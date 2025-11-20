from pydantic import BaseModel
import os

class Settings(BaseModel):
    ollama_embed: str = os.getenv("OLLAMA_EMBED", "nomic-embed-text")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # stub | openai | ollama
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_llm: str = os.getenv("OLLAMA_LLM", "llama3.2:3b")
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "50"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "10"))
    data_dir: str = os.getenv("DATA_DIR", "../data")

settings = Settings()

#  ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")