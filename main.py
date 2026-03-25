import os
import uuid
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
import faiss

load_dotenv()

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel("gemini-2.5-flash")


for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)

session_store: dict = {}

MAX_FILE_SIZE = 2 * 1024 * 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"


# ---------- Text Extraction ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return " ".join(page.extract_text() or "" for page in reader.pages).strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ["docx", "doc"]:
        return extract_text_from_docx(file_bytes)
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ---------- Chunking ----------

def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------- Embeddings ----------

def get_embeddings(texts: list[str]) -> np.ndarray:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = np.array(result["embedding"], dtype="float32")
    # Handle both single and batch responses
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings

def get_query_embedding(query: str) -> np.ndarray:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    emb = np.array(result["embedding"], dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


# ---------- FAISS ----------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]  # auto-detect dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_chunks(query: str, index: faiss.IndexFlatL2, chunks: list[str]) -> list[str]:
    query_emb = get_query_embedding(query)
    k = min(TOP_K, len(chunks))
    _, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "Document Q&A API with FAISS is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 2MB limit.")

    try:
        text = extract_text(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in the document.")

    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document could not be chunked.")

    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    index = build_faiss_index(embeddings)

    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        "index": index,
        "chunks": chunks,
        "filename": file.filename,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "total_chunks": len(chunks),
        "char_count": len(text),
    }


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask_question(body: AskRequest):
    session = session_store.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        relevant_chunks = retrieve_top_chunks(
            body.question,
            session["index"],
            session["chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that information in the document."
Be concise and accurate.

CONTEXT:
{context}

QUESTION:
{body.question.strip()}

ANSWER:"""

    try:
        response = chat_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    return {
        "answer": answer,
        "chunks_used": len(relevant_chunks),
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_store.pop(session_id, None)
    return {"message": "Session cleared."}