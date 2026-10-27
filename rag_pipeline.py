from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)


PROMPT = ChatPromptTemplate.from_template("""
You are a careful document question-answering assistant.

Answer the user's question using only the supplied document context.
- Answer directly and clearly.
- Do not invent facts or rely on outside knowledge for document-specific answers.
- If the context does not contain the answer, say:
  "The uploaded document does not contain sufficient information to answer that."
- When useful, mention the page number shown in the source metadata.

Document context:
{context}

User question:
{question}

Answer:
""")


def format_docs(docs):
    """Include page metadata so answers can be traced to the PDF."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
        formatted.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_chain():
    """Load the saved FAISS index and return a question-answering function."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env "
            "and add your OpenRouter API key."
        )

    if not Path(VECTORSTORE_DIR).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{VECTORSTORE_DIR}'. "
            "Run the ingestion step first: python ingest.py data/your_file.pdf"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda question: question,
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever
