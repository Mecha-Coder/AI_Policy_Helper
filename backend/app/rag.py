import time, os, math, json, hashlib
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm
import uuid
import httpx

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

class LocalEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Ollama semantic embedder ----
class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        data = {
            "model": self.model,
            "prompt": text
        }
        try:
            response = httpx.post(f"{self.host}/api/embeddings", json=data, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            # Ollama returns {"embedding": [float, ...]}
            emb = result.get("embedding", [])
            v = np.array(emb, dtype="float32")
            # L2 normalize
            v = v / (np.linalg.norm(v) + 1e-9)
            return v
        except Exception as e:
            print(f"Error getting embedding from Ollama: {e}")
            # Fallback: return zeros
            return np.zeros(4096, dtype="float32")

# ---- Vector store abstraction ---

class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int, host: str):
        self.client = QdrantClient(url=host, timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            points.append(qm.PointStruct(id=m.get("id") or m.get("hash") or i, vector=v.tolist(), payload=m))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OllamaLLM:
    def __init__(self, host: str, LLM: str):
        self.host = host
        self.model = LLM # You can change this to any model you have pulled

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = "You are a company agent. Reply to the client's query based ONLY on the sources provided below. If the answer cannot be found in any sources, state that clearly\n"
        prompt += """Answer format:
- 1–2 sentence direct answer.
- If relevant, cite exactly one source on a new line: Document: <DocumentName>, Section: <SectionName>. 
- No extra text.\n\n"""
        prompt += f"Client's Query: {query}\n"

        for i, c in enumerate(contexts, 1):
            prompt += (
                f"\n[SOURCE {i}]\n"
                f"Document: {c.get('title', 'Unknown')}\n"
                f"Section: {c.get('section', 'N/A')}\n"
                f"Content:\n{c.get('text', '')}\n"
                f"{'='*50}\n"
            )
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 0.1,
            "stream": False
        }

        try:
            response = httpx.post(f"{self.host}/api/generate", json=data, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            return f"Error generating answer with Ollama: {e}"

class OpenAILLM:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section when relevant.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self):        
        # --- Embedder selection ---
        if settings.ollama_embed == "nomic-embed-text" and settings.ollama_host:
            try:
                print("Using Ollama Embedder at", settings.ollama_host)
                self.embedder = OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed)
                self.embedder_name = f"ollama:{settings.ollama_embed}"
            except Exception:
                print("Ollama Embedder failed, falling back to LocalEmbedder")
                self.embedder = LocalEmbedder(dim=384)
                self.embedder_name = "local-384"
        else:
            print("Using LocalEmbedder")
            self.embedder = LocalEmbedder(dim=384)
            self.embedder_name = "local-384"    
        
        # --- Vector store selection ---
        embed_dim = 768 if settings.ollama_embed == "nomic-embed-text" else 384
        
        if settings.vector_store == "qdrant" and settings.store_host:
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=embed_dim, host=settings.store_host)
            except Exception:
                self.store = InMemoryStore(dim=embed_dim)
        else:
            self.store = InMemoryStore(dim=embed_dim)


        # --- LLM selection ---
        if settings.llm_provider == "ollama" and settings.ollama_host and settings.ollama_llm:
            try:
                print("Using Ollama LLM at", settings.ollama_host)
                self.llm = OllamaLLM(host=settings.ollama_host, LLM=settings.ollama_llm)
                self.llm_name = f"ollama:{settings.ollama_llm}"
            except Exception:
                print("Ollama LLM failed, falling back to StubLLM")
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            print("Defaulting to StubLLM")
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": str(uuid.uuid4()),
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))

    def retrieve(self, query: str, k: int) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k)
        self.metrics.add_retrieval((time.time()-t0)*1000.0)

        filtered_results = sorted(
            [item for item in results if item[0] >= 0.4],
            key=lambda x: x[0],
            reverse=True
        )

        return [meta for score, meta in filtered_results]

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": self.embedder_name,
            "llm_model": self.llm_name,
            **m
        }

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"title": d["title"], "section": d["section"], "text": ch})
    return out
