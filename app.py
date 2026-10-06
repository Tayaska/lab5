import glob
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
PREFERRED_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
VS_KWARGS = {"distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT, "normalize_L2": True}

NO_INFO = "У наданих документах немає інформації для відповіді на це запитання."
SYSTEM_PROMPT = f"""Ти – асистент, який відповідає на запитання студентів за методичними матеріалами з машинного навчання.

Правила:
1. Відповідай ТІЛЬКИ на основі наданого контексту. Не використовуй власні знання.
2. Якщо в контексті немає потрібної інформації, відповідай дослівно: "{NO_INFO}"
3. Якщо інформація є лише частково, дай відповідь на ту частину, що є, і вкажи, чого в документах немає.
4. Після кожного твердження вказуй номер фрагмента-джерела у квадратних дужках, наприклад [1] або [2][3].
5. Відповідай українською мовою, чітко і стисло (до 8 речень), зберігай терміни як у документах.
6. Не вигадуй цифр, назв функцій чи параметрів, яких немає в контексті."""

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAG: асистент з машинного навчання", page_icon="📚", layout="wide")


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


@st.cache_resource(show_spinner="Завантаження моделі векторних представлень...")
def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Завантаження бази знань...")
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
    """Модель за замовчуванням, а якщо її вже немає - Llama 70B або перша текстова модель."""
    try:
        available = sorted(m.id for m in get_llm_client(api_key).models.list().data)
    except Exception:
        return PREFERRED_MODEL
    if PREFERRED_MODEL in available:
        return PREFERRED_MODEL
    text = [m for m in available if not any(x in m for x in ("whisper", "guard", "tts", "orpheus"))]
    llama = [m for m in text if "llama" in m and "70b" in m]
    return (llama or text or [PREFERRED_MODEL])[0]


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
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < retries - 1 and ("429" in str(e) or "503" in str(e) or "rate" in str(e).lower()):
                time.sleep(10 * (attempt + 1))
            else:
                raise


def show_sources(sources):
    with st.expander(f"📄 Джерела ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['file']}**, сторінка {s['page']}")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


# ------------------------------------------------------------
# Інтерфейс
# ------------------------------------------------------------
api_key = get_api_key()
vectorstore = get_vectorstore()
all_files = sorted({d.metadata["file_name"] for d in vectorstore.docstore._dict.values()})

with st.sidebar:
    st.header("⚙️ Налаштування пошуку")
    k = st.slider("Кількість фрагментів (k)", 1, 8, 4)
    method = st.radio("Спосіб пошуку", ["Similarity", "MMR", "Hybrid"],
                      help="Similarity – найближчі вектори; MMR – релевантні та різноманітні фрагменти; "
                           "Hybrid – векторний пошук + пошук за ключовими словами (BM25)")
    files = st.multiselect("Шукати лише в документах", all_files,
                           help="Фільтрація за метаданими. Порожньо – пошук у всіх документах.")
    st.divider()
    st.subheader("📚 База знань")
    st.write(f"Документів: **{len(all_files)}**, фрагментів: **{vectorstore.index.ntotal}**")
    for f in all_files:
        st.caption(f"• {f}")
    st.caption(f"Embeddings: {EMBEDDING_MODEL.split('/')[-1]}")
    if api_key:
        st.caption(f"LLM: {get_llm_model(api_key)} (Groq)")
    if st.button("🗑️ Очистити чат"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 RAG-асистент з машинного навчання")
st.write("Поставте запитання про нейронні мережі, метрики, трансформери чи Streamlit. "
         "Відповідь формується **лише** на основі документів бази знань, із зазначенням джерел.")

if not api_key:
    st.error("Не знайдено API-ключ. Додайте GROQ_API_KEY у Settings → Secrets (Streamlit Cloud) "
             "або у змінну середовища.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Приклади запитань: *Як працює EarlyStopping?* · *Чим macro відрізняється від weighted F1?* · "
            "*Як розгорнути застосунок на Streamlit Cloud?*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ваше запитання...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Пошук у документах і формування відповіді..."):
            docs = retrieve(question, k, method, files)
            try:
                answer = ask_llm(get_llm_client(api_key), get_llm_model(api_key), build_prompt(question, docs))
            except Exception as e:
                answer = f"⚠️ Помилка звернення до мовної моделі: {e}"
        sources = [{"n": i, "file": d.metadata["file_name"], "page": d.metadata["page"],
                    "text": d.page_content} for i, d in enumerate(docs, start=1)]
        st.markdown(answer)
        if NO_INFO[:30] in answer:
            st.caption("ℹ️ Система не знайшла відповіді в базі знань і не стала її вигадувати.")
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
