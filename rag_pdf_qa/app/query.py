import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Prompt template — instructs the LLM to answer ONLY from context
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:""",
)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def answer_question(question: str, filename: str) -> dict:
    """
    Query pipeline:
    1. Embed the user's question
    2. Semantic search in FAISS (top-k chunks)
    3. Send chunks + question to LLM
    4. Return answer with source citations
    """
    store_path = os.path.join(VECTORSTORE_DIR, filename.replace(".pdf", ""))

    if not os.path.exists(store_path):
        return {
            "error": f"No vector store found for '{filename}'. Please upload and ingest the PDF first."
        }

    # Step 1 & 2: Embed question + retrieve top-k relevant chunks
    embeddings = get_embeddings()
    db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

    # Step 3: LLM (Ollama running locally — swap with OpenAI if preferred)
    llm = OllamaLLM(model="llama3")  # Change to "mistral", "phi3", etc.

    # Step 4: RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    result = qa_chain.invoke({"query": question})

    # Format source citations
    # Format source citations (deduplicated)
    seen = set()
    sources = []
    for doc in result.get("source_documents", []):
        key = (doc.metadata.get("page"), doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "text_preview": doc.page_content[:500   ] + "...",
        })
        

    return {
        "question": question,
        "answer": result["result"],
        "sources": sources,
        "model_used": "llama3 (via Ollama)",
    }
