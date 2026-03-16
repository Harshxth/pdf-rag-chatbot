import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def get_qa_chain():
    """
    Builds the full RAG pipeline:
    1. Load the vector store from disk
    2. Create a retriever that finds top 5 relevant chunks
    3. Connect it to the local LLM
    4. Return a chain that answers questions grounded in the PDF
    """

    embedder = OllamaEmbeddings(
        model="llama3.1",
        base_url=OLLAMA_HOST
    )

  
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
    )


    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatOllama(
        model="llama3.1",
        base_url=OLLAMA_HOST,
        temperature=0.1,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    return qa_chain