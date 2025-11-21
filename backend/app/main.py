from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List

from .models import IngestResponse, AskRequest, AskResponse, MetricsResponse, Citation, Chunk
from .settings import settings
from .ingest import load_documents
from .rag import RAGEngine, build_chunks_from_docs

app = FastAPI(title="AI Policy & Product Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    s = engine.stats()
    return MetricsResponse(**s)

@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    docs = load_documents(settings.data_dir)
    chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)
    new_docs, new_chunks = engine.ingest_chunks(chunks)
    return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ctx = engine.retrieve(req.query, k=req.k if req.k is not None else 4)
    answer = engine.generate(req.query, ctx)

    # Remove duplicate citations based on title
    seen_titles = set()
    citations = []
    for c in ctx:
        title = c.get("title")
        if title not in seen_titles:
            seen_titles.add(title)
            citations.append(Citation(title=title, section=c.get("section")))
    
    chunks = [Chunk(title=c.get("title"), section=c.get("section"), text=c.get("text")) for c in ctx]
    stats = engine.stats()
    return AskResponse(
        query=req.query,
        answer=answer,
        citations=citations,
        chunks=chunks,
        metrics={
            "retrieval_ms": stats["avg_retrieval_latency_ms"],
            "generation_ms": stats["avg_generation_latency_ms"],
        }
    )
