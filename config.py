import os
import torch
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Single Source of Truth (All Physical Parameters) ---
GLOBAL_CONFIG = {
    # System Math & Search settings
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "CHUNK_SIZE": 256,
    "CHUNK_OVERLAP": 25,
    "EMBED_BATCH_SIZE": 64,
    "HNSW_SPACE": "cosine",
    
    # LLM Settings
    "LLM_MODEL": os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
    "LLM_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "LLM_TIMEOUT": 600.0,
    
    # Database & Paths
    "CHROMA_DB_DIR": "data/chroma_db",
    "COLLECTION_NAME": "quant_papers",
    "PAPER_DOWNLOAD_DIR": "data/papers",
    
    # Retrieval Limits & Weights
    "PHASE_1_TOP_K": 5, # How many abstracts to fetch
    "PHASE_2_TOP_K": 15, # How many deep chunks to fetch
    "BM25_WEIGHT": 0.33,
    "VECTOR_WEIGHT": 0.66
}

def configure_base_settings():
    """Sets up the embedding model and global token alignment."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=GLOBAL_CONFIG["EMBEDDING_MODEL"],
        embed_batch_size=GLOBAL_CONFIG["EMBED_BATCH_SIZE"],
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    Settings.chunk_size = GLOBAL_CONFIG["CHUNK_SIZE"]
    Settings.chunk_overlap = GLOBAL_CONFIG["CHUNK_OVERLAP"]

def configure_llm():
    """Sets up the heavy LLM model (Ollama) only when needed."""
    model = GLOBAL_CONFIG["LLM_MODEL"]
    print(f"  [Config] Initializing LLM Engine: {model}")
    Settings.llm = Ollama(
        model=model, 
        base_url=GLOBAL_CONFIG["LLM_URL"], 
        request_timeout=GLOBAL_CONFIG["LLM_TIMEOUT"], 
        context_window=8192
    )
