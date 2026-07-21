# RAG-based PDF Question Answering System

A production-style Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDF documents and ask natural language questions about them.

## Architecture

```
Ingestion Pipeline:
PDF Upload → Chunking → Embedding Model → FAISS Vector DB

Query Pipeline:
User Question → Embed Question → Vector Search → LLM → Answer + Citations
```

## Tech Stack

| Component | Tool |
|---|---|
| API Framework | FastAPI |
| RAG Orchestration | LangChain |
| Embedding Model | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| LLM | Ollama (Llama 3 / Mistral — runs locally, free) |
| PDF Parsing | PyMuPDF |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd rag_pdf_qa
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com
ollama pull llama3       # or: ollama pull mistral
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

API is now live at `http://localhost:8000`

---

## API Endpoints

### Upload a PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "Successfully ingested 'your_document.pdf'",
  "chunks_stored": 42,
  "filename": "your_document.pdf"
}
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "filename": "your_document.pdf"}'
```

Response:
```json
{
  "question": "What is the main topic?",
  "answer": "The document discusses...",
  "sources": [
    {
      "page": 2,
      "text_preview": "Relevant chunk from the PDF..."
    }
  ],
  "model_used": "llama3 (via Ollama)"
}
```

### Interactive API Docs
Visit `http://localhost:8000/docs` for Swagger UI.

---

## Want to use OpenAI instead of Ollama?

In `app/query.py`, replace:
```python
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3")
```

With:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="your-key")
```

And add `langchain-openai` and `openai` to `requirements.txt`.

---

## Project Structure

```
rag_pdf_qa/
├── app/
│   ├── main.py       # FastAPI routes
│   ├── ingest.py     # PDF ingestion pipeline
│   └── query.py      # RAG query pipeline
├── uploads/          # Uploaded PDFs stored here
├── vectorstore/      # FAISS index saved here
├── requirements.txt
└── README.md
```
