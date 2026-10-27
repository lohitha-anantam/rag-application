# RAG-Based AI Document Question Answering System

A modular Python application that answers questions about PDF documents using Retrieval-Augmented Generation (RAG).

## How it works

1. **PDF loading:** PyPDFLoader extracts text and page metadata from a PDF.
2. **Chunking:** RecursiveCharacterTextSplitter creates chunks of up to 1,000 characters with 150-character overlap.
3. **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` converts chunks into vectors.
4. **Vector storage:** FAISS stores the vectors locally and supports similarity search.
5. **Retrieval:** The top 5 relevant chunks are retrieved for each question.
6. **Generation:** An OpenRouter chat model produces an answer grounded in the retrieved context.

## Project structure

```text
rag_document_qa_vscode/
├── data/                 # Put your PDF file here
├── vectorstore/          # Saved FAISS index (created after ingestion)
├── config.py             # Model names and settings
├── ingest.py             # PDF loading, chunking, embeddings, FAISS index
├── rag_pipeline.py       # Retriever, prompt, LLM, RAG chain
├── main.py               # Interactive terminal application
├── requirements.txt
├── .env.example
└── .gitignore
```

## Run in VS Code

### 1. Open the project folder

Extract the ZIP file and open `rag_document_qa_vscode` in VS Code.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to a new file named `.env`. Replace the placeholder with your OpenRouter API key:

```text
OPENROUTER_API_KEY=your_real_key_here
OPENROUTER_MODEL=openrouter/free
```

Never share or commit your `.env` file. If `openrouter/free` is unavailable, set `OPENROUTER_MODEL` to a model ID currently available to your OpenRouter account.

### 5. Add a PDF

Place your PDF inside the `data` folder, for example:

```text
data/resume.pdf
```

You can use your resume or another text-based PDF. Scanned/image-only PDFs require OCR, which this project does not currently include.

### 6. Build the vector index

```bash
python ingest.py data/resume.pdf
```

If you have placed only one PDF in `data/`, you can also run:

```bash
python ingest.py
```

This creates a local FAISS index in `vectorstore/faiss_index`.

### 7. Ask questions

```bash
python main.py
```

Example questions:

- What is my internship experience?
- Which technologies are mentioned in the document?
- Summarize the projects described in the PDF.
- What information does the document provide about my education?

Type `exit` to stop.

## Important notes

- Run ingestion again whenever you change the PDF; otherwise, the app will use the previously saved index.
- The embedding model is downloaded on first use and may take some time.
- OpenRouter model availability and free-model routing can change. A model must be available to your API key.
- The local FAISS index uses LangChain's pickle-based deserialization when loaded. Only load index files that you created yourself or otherwise trust.
- Retrieval-augmented generation reduces unsupported answers but does not guarantee perfect accuracy. Check the retrieved source snippets for important answers.
