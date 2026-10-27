from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)


def load_pdf(pdf_path: str):
    """Load a PDF and return one LangChain Document per page."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    documents = PyPDFLoader(str(path)).load()
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        raise ValueError(
            "No readable text was found. The PDF may be scanned; "
            "OCR would be required for scanned pages."
        )

    return documents


def create_chunks(documents):
    """Split page documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        chunk for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No text chunks were created.")

    return chunks


def build_vectorstore(pdf_path: str):
    """Load PDF, chunk it, embed chunks, and save the FAISS index."""
    documents = load_pdf(pdf_path)
    chunks = create_chunks(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"PDF loaded: {Path(pdf_path).name}")
    print(f"Pages with text: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"FAISS vectors saved to: {VECTORSTORE_DIR}")

    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a FAISS index from a PDF document."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF. Example: data/resume.pdf",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_path = args.pdf
    else:
        pdfs = sorted(Path(DATA_DIR).glob("*.pdf"))
        if not pdfs:
            raise SystemExit(
                "No PDF supplied and no PDF found in the data/ folder. "
                "Put a PDF in data/ or run: python ingest.py data/your_file.pdf"
            )
        pdf_path = str(pdfs[0])

    build_vectorstore(pdf_path)
