import glob
import html
import os
import re
import time
import warnings

import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ------------------------------------------------------------
# Налаштування (ті самі, що в ноутбуці)
# ------------------------------------------------------------
DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREFERRED_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct",
                    "meta-llama/llama-4-maverick-17b-128e-instruct", "openai/gpt-oss-20b"]
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
VS_KWARGS = {"distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT, "normalize_L2": True}

NO_INFO = "У наданих документах немає інформації для відповіді на це запитання."
SYSTEM_PROMPT = f"""Ти – консультант із сонячних електростанцій, який відповідає на запитання користувачів за матеріалами бази знань.

Правила:
1. Відповідай ТІЛЬКИ на основі наданого контексту. Не використовуй власні знання.
2. Якщо в контексті немає потрібної інформації, відповідай дослівно: "{NO_INFO}"
3. Якщо інформація є лише частково, дай відповідь на ту частину, що є, і вкажи, чого в документах немає.
4. Після кожного твердження вказуй номер фрагмента-джерела у квадратних дужках, наприклад [1] або [2][3].
5. Відповідай українською мовою, чітко і стисло (до 8 речень), зберігай терміни як у документах.
6. Не вигадуй цифр, назв функцій чи параметрів, яких немає в контексті."""

DOC_INFO = {
    "01": ("🔆", "Сонячна електростанція"),
    "02": ("🟦", "Сонячна панель"),
    "03": ("🔌", "Сонячний інвертор"),
    "04": ("🏙️", "Мережева СЕС"),
    "05": ("🔁", "Гібридна СЕС"),
    "06": ("🔋", "LiFePO4-акумулятор"),
    "07": ("💶", "Зелений тариф"),
    "08": ("🇺🇦", "Сонячна енергетика України"),
    "09": ("📄", "Наукова стаття (КНУТД)"),
}
EXAMPLES = [
    "Чим гібридна СЕС відрізняється від мережевої?",
    "Що таке зелений тариф?",
    "Які переваги LiFePO4-акумуляторів?",
    "Яку функцію виконує сонячний інвертор?",
    "Як розвивалась сонячна енергетика в Україні?",
]
METHODS = {"Векторний": "Similarity", "Різноманітний (MMR)": "MMR", "Гібридний (вектор + BM25)": "Hybrid"}

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Сонячний консультант", page_icon="☀️", layout="centered",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------
# Оформлення (власні стилі CSS)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea { font-family: 'Manrope', sans-serif !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { padding-top: 2rem; max-width: 820px; }

.hero {
  background: radial-gradient(circle at 88% 20%, #FDE68A 0 9%, transparent 10%),
              linear-gradient(135deg, #F59E0B 0%, #F97316 55%, #DB2777 100%);
  border-radius: 22px; padding: 26px 30px; margin-bottom: 18px; color: #1A1206;
  box-shadow: 0 12px 32px rgba(249, 115, 22, .25);
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; color: #1A1206; padding: 0; }
.hero p { margin: 0; font-size: 1rem; opacity: .9; }
.hero .badges { margin-top: 14px; }
.hero .badge {
  display: inline-block; background: rgba(26, 18, 6, .14); border-radius: 999px;
  padding: 4px 12px; margin: 0 6px 6px 0; font-size: .8rem; font-weight: 600;
}

.src-card {
  background: #121B30; border: 1px solid #263556; border-left: 4px solid #FBBF24;
  border-radius: 12px; padding: 10px 14px; margin: 8px 0;
}
.src-card .head { font-weight: 700; color: #FBBF24; font-size: .9rem; }
.src-card .meta { color: #8EA0C4; font-size: .78rem; margin-left: 6px; }
.src-card .body { color: #C9D3E8; font-size: .85rem; margin-top: 4px; line-height: 1.45; }

.doc-tile {
  background: #121B30; border: 1px solid #263556; border-radius: 16px;
  padding: 16px; margin-bottom: 12px; min-height: 120px;
}
.doc-tile .icon { font-size: 1.6rem; }
.doc-tile .title { font-weight: 800; margin-top: 4px; }
.doc-tile .file { color: #8EA0C4; font-size: .78rem; word-break: break-all; }

.step { display: flex; gap: 12px; align-items: flex-start; margin: 10px 0; }
.step .num {
  flex: 0 0 30px; height: 30px; border-radius: 50%; background: #FBBF24; color: #1A1206;
  font-weight: 800; display: flex; align-items: center; justify-content: center;
}
.no-info { background: #2A1F0B; border: 1px dashed #F59E0B; border-radius: 10px; padding: 8px 12px; font-size: .85rem; }
[data-testid="stChatMessage"] { background: #111A2E; border-radius: 16px; border: 1px solid #1F2C4A; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# API-ключ: Streamlit Secrets або змінна середовища
# ------------------------------------------------------------
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


# ------------------------------------------------------------
# Завантаження документів, чанкування, векторне сховище (з кешуванням)
# ------------------------------------------------------------
def load_documents(docs_dir):
    documents = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*"))):
        file_name = os.path.basename(path)
        ext = os.path.splitext(file_name)[1].lower()
        if ext in (".md", ".txt"):
            docs = TextLoader(path, encoding="utf-8").load()
        elif ext == ".pdf":
            docs = PyPDFLoader(path).load()
        else:
            continue
        title = docs[0].metadata.get("title") or file_name
        for d in docs:
            d.metadata = {
                "source": path,
                "file_name": file_name,
                "doc_title": title,
                "doc_type": ext[1:],
                "page": d.metadata.get("page", 0) + 1 if ext == ".pdf" else 1,
            }
        documents.extend(docs)
    return documents


@st.cache_resource(show_spinner="☀️ Завантаження моделі векторних представлень...")
def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="📚 Завантаження бази знань...")
def get_vectorstore():
    embeddings = get_embeddings()
    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(INDEX_DIR, embeddings,
                                    allow_dangerous_deserialization=True, **VS_KWARGS)
        except Exception:
            pass  # якщо збережене сховище несумісне - будуємо заново з документів
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(load_documents(DOCS_DIR))
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    return FAISS.from_documents(chunks, embeddings, **VS_KWARGS)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


@st.cache_resource
def get_bm25():
    chunks = list(get_vectorstore().docstore._dict.values())
    return BM25Okapi([tokenize(c.page_content) for c in chunks]), chunks


@st.cache_resource
def get_llm_client(api_key):
    return Groq(api_key=api_key)


@st.cache_resource
def get_llm_model(api_key):
    """Перша доступна модель з переліку пріоритетних."""
    try:
        available = {m.id for m in get_llm_client(api_key).models.list().data}
    except Exception:
        return PREFERRED_MODELS[0]
    return next((m for m in PREFERRED_MODELS if m in available), PREFERRED_MODELS[0])


# ------------------------------------------------------------
# Пошук
# ------------------------------------------------------------
def retrieve(query, k, method, files):
    vs = get_vectorstore()
    flt = {"file_name": {"$in": files}} if files else None

    if method == "Similarity":
        return vs.similarity_search(query, k=k, filter=flt)
    if method == "MMR":
        return vs.max_marginal_relevance_search(query, k=k, fetch_k=20, lambda_mult=0.5, filter=flt)

    # Hybrid: векторний пошук + BM25, об'єднання рейтингів (Reciprocal Rank Fusion)
    bm25, chunks = get_bm25()
    allowed = [i for i, c in enumerate(chunks) if not files or c.metadata["file_name"] in files]
    bm_scores = bm25.get_scores(tokenize(query))
    bm_ranked = [chunks[i] for i in sorted(allowed, key=lambda i: bm_scores[i], reverse=True)[:20]]
    scores, by_id = {}, {}
    for ranked in (vs.similarity_search(query, k=20, filter=flt), bm_ranked):
        for rank, d in enumerate(ranked):
            cid = d.metadata["chunk_id"]
            by_id[cid] = d
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)
    return [by_id[c] for c in sorted(scores, key=scores.get, reverse=True)[:k]]


# ------------------------------------------------------------
# Генерація відповіді
# ------------------------------------------------------------
def build_prompt(question, docs):
    context = "\n\n".join(
        f"[{i}] Джерело: {d.metadata['file_name']}, с. {d.metadata['page']}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    return f"Контекст:\n{context}\n\nЗапитання: {question}\n\nВідповідь:"


def ask_llm(client, model, prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except Exception as e:
            if attempt < retries - 1 and ("429" in str(e) or "503" in str(e) or "rate" in str(e).lower()):
                time.sleep(10 * (attempt + 1))
            else:
                raise


def doc_label(file_name):
    icon, title = DOC_INFO.get(file_name[:2], ("📄", file_name))
    return icon, title


def show_sources(sources):
    with st.expander(f"📎 Використані фрагменти ({len(sources)})"):
        for s in sources:
            icon, title = doc_label(s["file"])
            text = html.escape(s["text"][:450] + ("..." if len(s["text"]) > 450 else ""))
            st.markdown(
                f"<div class='src-card'><span class='head'>[{s['n']}] {icon} {title}</span>"
                f"<span class='meta'>{html.escape(s['file'])} · с. {s['page']}</span>"
                f"<div class='body'>{text}</div></div>",
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# Інтерфейс
# ------------------------------------------------------------
api_key = get_api_key()
vectorstore = get_vectorstore()
all_chunks = list(vectorstore.docstore._dict.values())
all_files = sorted({d.metadata["file_name"] for d in all_chunks})

st.markdown(f"""
<div class="hero">
  <h1>☀️ Сонячний консультант</h1>
  <p>Запитайте про сонячні електростанції: панелі, інвертори, акумулятори, зелений тариф чи розвиток сонячної енергетики в Україні.
  Відповідь складається <b>лише</b> з реальних документів бази знань (статті Вікіпедії та наукова стаття) і містить посилання на джерела.</p>
  <div class="badges">
    <span class="badge">📚 {len(all_files)} документів</span>
    <span class="badge">🧩 {vectorstore.index.ntotal} фрагментів</span>
    <span class="badge">🔎 FAISS + BM25</span>
    <span class="badge">🤖 Groq LLM</span>
  </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_docs, tab_how = st.tabs(["💬 Консультант", "📚 База знань", "🔍 Як це працює"])

# ---------------- Вкладка 1: чат ----------------
with tab_chat:
    c1, c2 = st.columns([3, 1])
    with c1:
        with st.popover("⚙️ Параметри пошуку", width="stretch"):
            k = st.slider("Скільки фрагментів передавати моделі", 1, 8, 4)
            method_name = st.radio("Спосіб пошуку", list(METHODS),
                                   help="Векторний – найближчі за змістом фрагменти; MMR – релевантні та "
                                        "різноманітні; гібридний – зміст + ключові слова (BM25)")
            files = st.multiselect("Шукати лише в документах", all_files,
                                   format_func=lambda f: " ".join(doc_label(f)),
                                   help="Фільтрація за метаданими. Порожньо – пошук у всіх документах.")
    with c2:
        if st.button("🧹 Новий діалог", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if not api_key:
        st.error("Не знайдено API-ключ. Додайте GROQ_API_KEY у Settings → Secrets (Streamlit Cloud) "
                 "або у змінну середовища.")
        st.stop()

    model_name = get_llm_model(api_key)

    if not st.session_state.messages:
        st.markdown("##### 💡 Спробуйте запитати")
        picked = st.pills("Приклади запитань", EXAMPLES, label_visibility="collapsed", key="example_pick")
        if picked:
            st.session_state.pending = picked

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "☀️"):
            st.markdown(msg["content"])
            if msg.get("no_info"):
                st.markdown("<div class='no-info'>ℹ️ У базі знань немає відповіді – консультант не став її "
                            "вигадувати.</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                show_sources(msg["sources"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    typed = st.chat_input("Ваше запитання про СЕС...")
    question = typed or st.session_state.pop("pending", None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="☀️"):
            with st.spinner("Шукаю в документах і формую відповідь..."):
                t0 = time.time()
                docs = retrieve(question, k, METHODS[method_name], files)
                try:
                    answer = ask_llm(get_llm_client(api_key), model_name, build_prompt(question, docs))
                except Exception as e:
                    answer = f"⚠️ Помилка звернення до мовної моделі: {e}"
                elapsed = time.time() - t0
            sources = [{"n": i, "file": d.metadata["file_name"], "page": d.metadata["page"],
                        "text": d.page_content} for i, d in enumerate(docs, start=1)]
            no_info = NO_INFO[:30] in answer
            meta = f"⏱ {elapsed:.1f} с · {method_name} · k = {k} · {model_name}"
            st.markdown(answer)
            if no_info:
                st.markdown("<div class='no-info'>ℹ️ У базі знань немає відповіді – консультант не став її "
                            "вигадувати.</div>", unsafe_allow_html=True)
            show_sources(sources)
            st.caption(meta)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources,
                                          "no_info": no_info, "meta": meta})
        if "example_pick" in st.session_state:
            del st.session_state["example_pick"]
        st.rerun()

# ---------------- Вкладка 2: база знань ----------------
with tab_docs:
    st.markdown("#### Документи бази знань")
    counts = {f: sum(1 for c in all_chunks if c.metadata["file_name"] == f) for f in all_files}
    cols = st.columns(3)
    for i, f in enumerate(all_files):
        icon, title = doc_label(f)
        with cols[i % 3]:
            st.markdown(
                f"<div class='doc-tile'><div class='icon'>{icon}</div><div class='title'>{title}</div>"
                f"<div class='file'>{html.escape(f)}</div>"
                f"<div class='file'>фрагментів: {counts[f]}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("#### Переглянути фрагменти документа")
    chosen = st.selectbox("Документ", all_files, format_func=lambda f: " ".join(doc_label(f)))
    for c in sorted((c for c in all_chunks if c.metadata["file_name"] == chosen),
                    key=lambda c: c.metadata["chunk_id"]):
        with st.expander(f"Фрагмент #{c.metadata['chunk_id']} · с. {c.metadata['page']} · "
                         f"{len(c.page_content)} символів"):
            st.text(c.page_content)

# ---------------- Вкладка 3: як це працює ----------------
with tab_how:
    st.markdown("#### Як консультант знаходить відповідь (RAG)")
    steps = [
        ("Документи", f"{len(all_files)} реальних матеріалів про СЕС: статті української Вікіпедії (CC BY-SA 4.0) "
                      "та наукова стаття з Вісника КНУТД у форматах .md, .txt і .pdf."),
        ("Фрагменти", f"Тексти розбиті на фрагменти по ~{CHUNK_SIZE} символів з перекриттям {CHUNK_OVERLAP}."),
        ("Вектори", "Кожен фрагмент перетворено на вектор моделлю paraphrase-multilingual-MiniLM-L12-v2 "
                    "і збережено у сховищі FAISS."),
        ("Пошук", "Запитання теж перетворюється на вектор, і система знаходить k найближчих фрагментів "
                  "(або комбінує їх з пошуком за ключовими словами BM25)."),
        ("Відповідь", "Знайдені фрагменти разом із запитанням передаються мовній моделі, яка відповідає "
                      "лише за ними і вказує номери джерел."),
    ]
    for i, (title, text) in enumerate(steps, start=1):
        st.markdown(f"<div class='step'><div class='num'>{i}</div><div><b>{title}.</b> {text}</div></div>",
                    unsafe_allow_html=True)
    st.markdown("#### Моделі")
    st.markdown(f"- Векторні представлення: `{EMBEDDING_MODEL.split('/')[-1]}`\n"
                f"- Мовна модель: `{get_llm_model(api_key) if api_key else '—'}` (Groq)\n"
                f"- Векторне сховище: FAISS, косинусна близькість")
