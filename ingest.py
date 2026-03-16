import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Where to save the vector database on disk
CHROMA_DIR = "./chroma_db"

# Ollama host — reads from .env, falls back to localhost
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def load_and_chunk(pdf_path: str):
    """
    Loads a PDF and splits it into overlapping chunks.
    Why: LLMs have limited context windows so we break
    the PDF into small pieces that fit and stay meaningful.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    """
    Converts chunks into embeddings and stores them in Chroma.
    Why: We need to search chunks by meaning, not keywords.
    Embeddings let us do that mathematically.
    """
    print("Building embeddings — this may take a few minutes...")

    embedder = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST
    )

    # Delete existing store first to avoid conflicts
    import shutil
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Cleared old vector store")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR,
    )

    # Verify it actually saved
    count = vectordb._collection.count()
    print(f"Saved {count} chunks to vector store at {CHROMA_DIR}")
    
    return vectordb

if __name__ == "__main__":
    # Step 1: load and chunk
    chunks = load_and_chunk("data/SecureAi.pdf")

    # Step 2: embed and store
    build_vector_store(chunks)

    print("\nDone! Your PDF has been ingested into the vector store.")