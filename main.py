from rag_pipeline import create_rag_chain


def main():
    try:
        rag_chain, retriever = create_rag_chain()
    except Exception as exc:
        print(f"Could not start the RAG system: {exc}")
        return

    print("\nRAG Document Question Answering System")
    print("Ask questions about the indexed PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRAG system stopped.")
            break

        if question.lower() in {"exit", "quit"}:
            print("RAG system stopped.")
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = rag_chain.invoke(question)
            print("\nAnswer:\n")
            print(answer)

            # Show retrieved source pages/chunks for transparency.
            source_docs = retriever.invoke(question)
            print("\nRetrieved sources:")
            seen = set()
            for doc in source_docs:
                page = doc.metadata.get("page")
                label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                key = (label, doc.page_content[:100])
                if key not in seen:
                    seen.add(key)
                    print(f"- {label}: {doc.page_content[:180].replace(chr(10), ' ')}...")
            print()

        except Exception as exc:
            print(f"\nQuestion failed: {exc}\n")


if __name__ == "__main__":
    main()
