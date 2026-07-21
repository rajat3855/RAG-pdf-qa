import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Free, runs locally, no API key needed


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def ingest_pdf(file_path: str, filename: str) -> int:
    """
    Ingestion pipeline:
    1. Load PDF
    2. Split into chunks
    3. Generate embeddings
    4. Store in FAISS vector DB
    """
    # Step 1: Load PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    # Add source metadata to each chunk
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    # Step 3 & 4: Embed and store in FAISS
    embeddings = get_embeddings()
    store_path = os.path.join(VECTORSTORE_DIR, filename.replace(".pdf", ""))

    if os.path.exists(store_path):
        # Append to existing store
        db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(store_path)

    return len(chunks)
