from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil, os
from app.ingest import ingest_pdf
from app.query import answer_question

app = FastAPI(title="RAG PDF Q&A System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

class QuestionRequest(BaseModel):
    question: str
    filename: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunk_count = ingest_pdf(file_path, file.filename)
    return {"message": f"Successfully ingested '{file.filename}'", "chunks_stored": chunk_count, "filename": file.filename}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    return answer_question(request.question, request.filename)

@app.get("/health")
def health():
    return {"status": "ok"}
