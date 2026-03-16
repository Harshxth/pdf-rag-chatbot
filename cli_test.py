from rag_chain import get_qa_chain

if __name__ == "__main__":
    print("Loading RAG chain...")
    qa_chain = get_qa_chain()
    print("Ready! Type your question below.\n")

    while True:
        question = input("Question (or 'q' to quit): ")
        
        if question.lower() == "q":
            break

        result = qa_chain({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]

        print("\n--- Answer ---")
        print(answer)
        
        print("\n--- Sources ---")
        for s in sources:
            print(f"  Page {s.metadata.get('page') + 1} — {s.metadata.get('source')}")
        print()