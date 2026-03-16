import streamlit as st
import os
import tempfile
import shutil
import gc
import time
from ingest import load_and_chunk, build_vector_store, CHROMA_DIR
from rag_chain import get_qa_chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    .source-badge {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
def get_ingested_files():
    """Returns set of filenames already ingested into vector store."""
    return st.session_state.get("ingested_files", set())


def mark_file_ingested(filename):
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()
    st.session_state.ingested_files.add(filename)


def refresh_chunk_count():
    """
    Opens ChromaDB once, reads the count, then immediately closes it.
    Stores result in session state so we don't re-open on every rerun.
    Why: Windows holds file locks aggressively — opening ChromaDB on every
    rerun prevents the clear button from deleting the folder.
    """
    try:
        embedder = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
        count = db._collection.count()
        del db          # explicitly close connection
        gc.collect()    # release file handles immediately
        st.session_state.chunk_count = count
    except Exception:
        st.session_state.chunk_count = 0


def get_chunk_count():
    """Read chunk count from session state — no ChromaDB connection opened."""
    return st.session_state.get("chunk_count", 0)


# ── Initialise chunk count on first load only ────────────────
if "chunk_count" not in st.session_state:
    refresh_chunk_count()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF RAG Chatbot")
    st.markdown("---")

    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs to chat with"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            already_done = uploaded_file.name in get_ingested_files()

            if already_done:
                st.success(f"✅ {uploaded_file.name}")
            else:
                if st.button(f"Ingest: {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        chunks = load_and_chunk(tmp_path)
                        build_vector_store(chunks)
                        os.unlink(tmp_path)

                        mark_file_ingested(uploaded_file.name)
                        refresh_chunk_count()

                        if "qa_chain" in st.session_state:
                            del st.session_state.qa_chain

                    st.success(f"✅ {uploaded_file.name} ingested!")
                    st.rerun()

    st.markdown("---")
    st.metric("Chunks in store", get_chunk_count())

    ingested = get_ingested_files()
    if ingested:
        st.subheader("Loaded PDFs")
        for name in ingested:
            st.markdown(f"📎 {name}")

    st.markdown("---")

    if st.button("🗑️ Clear all PDFs", type="secondary"):

        # Release the RAG chain first
        if "qa_chain" in st.session_state:
            del st.session_state.qa_chain
        gc.collect()

    # Use ChromaDB's own reset instead of deleting the folder
    # This avoids Windows file lock entirely
        try:
            embedder = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
            db.delete_collection()
            del db
            gc.collect()
        except Exception as e:
            st.warning(f"Could not clear vector store: {e}")
            st.stop()

        st.session_state.ingested_files = set()
        st.session_state.chat_history = []
        st.session_state.chunk_count = 0

        st.success("✅ Cleared!")
        st.rerun()


# ── Main area ────────────────────────────────────────────────
st.title("💬 Chat with your PDFs")

if get_chunk_count() == 0:
    st.info("👈 Upload and ingest at least one PDF using the sidebar to get started.")
    st.stop()

if "qa_chain" not in st.session_state:
    with st.spinner("Loading RAG chain..."):
        st.session_state.qa_chain = get_qa_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content["answer"])
            if content.get("sources"):
                with st.expander("📍 Sources"):
                    for src in content["sources"]:
                        st.markdown(
                            f'<span class="source-badge">Page {src["page"]}</span> '
                            f'{src["source"]}',
                            unsafe_allow_html=True
                        )

question = st.chat_input("Ask a question about your PDFs...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

        st.write(answer)

        formatted_sources = []
        seen = set()
        for s in sources:
            page = s.metadata.get("page", 0) + 1
            source = s.metadata.get("source", "unknown")
            key = f"{source}-{page}"
            if key not in seen:
                seen.add(key)
                formatted_sources.append({
                    "page": page,
                    "source": os.path.basename(source)
                })

        if formatted_sources:
            with st.expander("📍 Sources"):
                for src in formatted_sources:
                    st.markdown(
                        f'<span class="source-badge">Page {src["page"]}</span> '
                        f'{src["source"]}',
                        unsafe_allow_html=True
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": {"answer": answer, "sources": formatted_sources}
    })