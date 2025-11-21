from pydantic import BaseModel
import os

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_provider: str = os.getenv("LLM_PROVIDER", "stub")  # stub | openai | ollama

    ollama_host: str = os.getenv("OLLAMA_HOST") # http://localhost:11434
    ollama_embed: str = os.getenv("OLLAMA_EMBED", "nomic-embed-text")
    ollama_llm: str = os.getenv("OLLAMA_LLM", "qwen2.5:3b-instruct-q4_K_M") #"llama3.2:3b"
    
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    store_host: str = os.getenv("STORE_HOST") # "http://localhost:6333"
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "40"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "5"))
    
    data_dir: str = os.getenv("DATA_DIR", "../data")

settings = Settings()