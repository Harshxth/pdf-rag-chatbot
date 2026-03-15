import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):

    # Load the PDF — each page becomes a Document object
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # max characters per chunk
        chunk_overlap=150,   # characters shared between chunks
    )

    chunks = splitter.split_documents(pages)

    print(f"Created {len(chunks)} chunks")
    return chunks


# This block only runs if you execute this file directly not when it's imported by another file
if __name__ == "__main__":
    # Quick test — drop any PDF in /data and update the name
    test_chunks = load_and_chunk("data/Test.pdf")
    
    # Print the first chunk so you can see what it looks like
    print("\n--- First Chunk ---")
    print(test_chunks[0].page_content)
    print("\n--- Metadata ---")
    print(test_chunks[0].metadata)