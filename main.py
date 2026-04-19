import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# =====================================================
# ⚙️ CONFIGURATION
# =====================================================

CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 100
TOP_K           = 4
DATA_PATH       = "data"
DB_PATH         = "db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"

# Groq API key from environment variable
# Set this on Railway dashboard as GROQ_API_KEY
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")


# =====================================================
# FASTAPI APP
# CORS is enabled so the mobile app can call this API
# from any origin (iOS, Android, web)
# =====================================================

app = FastAPI(
    title="Mining Law AI API",
    description="RAG-based API for Indian Mining Law Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (mobile app, web, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# REQUEST / RESPONSE MODELS
# Pydantic models define the shape of API requests
# and responses — like a contract between app and server
# =====================================================

class QuestionRequest(BaseModel):
    question: str               # The user's question

class CitationResponse(BaseModel):
    source: str                 # PDF filename
    page: int | str             # Page number
    section: str | None         # Section/Rule number if found
    snippet: str                # Short preview of the chunk

class AnswerResponse(BaseModel):
    answer: str                 # The generated answer
    citations: list[CitationResponse]   # Source citations


# =====================================================
# LOAD RESOURCES ON STARTUP
# These are loaded once when the server starts,
# not on every request — keeps responses fast.
# =====================================================

print("🔢 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load or build FAISS index
if os.path.exists(DB_PATH):
    print("⚡ Loading existing FAISS index...")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("🆕 Building FAISS index from PDFs...")
    all_documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    for filename in pdf_files:
        print(f"   Loading: {filename}")
        loader = PDFPlumberLoader(os.path.join(DATA_PATH, filename))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        all_documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print("💾 Index saved!")

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
)

print("🤖 Loading LLM...")
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=200,
)

print("✅ API ready!")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def build_prompt(context: str, question: str) -> str:
    return f"""[INST] You are a legal assistant specializing in Indian mining law.

Use ONLY the context below to answer the question.
- Answer in 2-3 clear sentences.
- Mention Section or Rule numbers if present in the context.
- If not found, say: "This information was not found in the provided documents."
- Never invent section numbers or rules.

Context:
{context}

Question: {question} [/INST]"""


def clean_context(docs, max_chars=800):
    combined, total = [], 0
    for doc in docs:
        text = " ".join(doc.page_content.split()).strip()
        if total + len(text) > max_chars:
            text = text[:max_chars - total]
        combined.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(combined)


def extract_citation(doc) -> dict:
    source = doc.metadata.get("source", "Unknown")
    page   = doc.metadata.get("page", None)
    page_display = int(page) + 1 if isinstance(page, (int, float)) else "N/A"

    text  = doc.page_content
    match = re.search(
        r'\b(Section|Sec\.|Rule|Regulation|Clause|Article|Schedule)\s+(\d+[\w\(\)\.]*)',
        text, re.IGNORECASE
    )
    section = f"{match.group(1)} {match.group(2)}" if match else None

    return {
        "source":  source,
        "page":    page_display,
        "section": section,
        "snippet": text[:120].strip()
    }


# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
def root():
    """Health check endpoint — confirms API is running."""
    return {"status": "Mining Law AI API is running ⛏️"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint — takes a question, returns answer + citations.

    The mobile app calls this endpoint with:
    POST /ask
    {"question": "Who appoints the Chief Inspector?"}

    And gets back:
    {"answer": "...", "citations": [...]}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Retrieve relevant chunks
        docs      = retriever.invoke(request.question)
        context   = clean_context(docs)
        prompt    = build_prompt(context, request.question)

        # Generate answer
        response  = llm.invoke(prompt)
        answer    = response.content.strip()

        # Extract citations
        seen, citations = set(), []
        for doc in docs:
            cite = extract_citation(doc)
            key  = (cite["source"], cite["page"])
            if key not in seen:
                seen.add(key)
                citations.append(cite)

        return AnswerResponse(answer=answer, citations=citations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Returns API health status and loaded document count."""
    pdf_count = len([f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]) if os.path.exists(DATA_PATH) else 0
    return {
        "status":     "healthy",
        "model":      GROQ_MODEL,
        "documents":  pdf_count,
        "index":      os.path.exists(DB_PATH)
    }
