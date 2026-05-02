import os
import torch
from typing import Dict, Any
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from embeddings import GenericAPIEmbedding

# Ensure environment variables are loaded globally before assigning values
load_dotenv()

# --- Metadata Schema Constants (Single Source of Truth) ---
KEY_ARXIV_ID = "arxiv_id"
KEY_PARENT_ID = "parent_id"
KEY_IS_SUMMARY = "is_summary"

# --- Single Source of Truth (All Physical Parameters) ---
GLOBAL_CONFIG: Dict[str, Any] = {
    # MASTER SWITCH: Set to "local", "hf", "google", or "ollama"
    "EMBEDDING_MODE": "local", 
    
    # Model used for Local/HF/Google/Ollama modes
    "EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-small-en", 
    "MODEL_CONTEXT_WINDOW": 8192, 

    # Math & Search Parameters
    "CHUNK_SIZE": 2048,
    "CHUNK_OVERLAP": 256,
    "EMBED_BATCH_SIZE": 2,
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
    "PHASE_1_TOP_K": 5, 
    "PHASE_2_TOP_K": 15,
    "BM25_WEIGHT": 0.33,
    "VECTOR_WEIGHT": 0.66,
    
    # API Retry & Timeout Settings
    "API_MAX_RETRIES": 5,
    "API_RETRY_DELAY": 5,   # Wait time for HF loading (503) or Rate Limit (429)
    "ARXIV_MAX_RETRIES": 3,
    "ARXIV_RETRY_DELAY": 5,
    "ARXIV_RETRY_DELAY": 5,
    # Standardized Schema Keys
    "KEY_PARENT_ID": KEY_PARENT_ID,
    "KEY_IS_SUMMARY": KEY_IS_SUMMARY,
    "KEY_ARXIV_ID": KEY_ARXIV_ID
}

def normalize_arxiv_id(raw_id: str) -> str:
    """Canonical ArXiv ID normalization (strips version suffixes and cleans path-unsafe chars)."""
    # Remove file extension if present
    clean_id = raw_id.replace(".pdf", "")
    # Standardize separator
    clean_id = clean_id.replace("/", "_")
    # Return base ID without version (e.g., 1234.5678v2 -> 1234.5678)
    return clean_id.split('v')[0]

def configure_base_settings() -> None:
    """
    Sets up the embedding model based on the EMBEDDING_MODE.
    Automatically handles URL and API Key mapping.
    """
    mode = GLOBAL_CONFIG["EMBEDDING_MODE"].lower()
    
    if mode == "google":
        # Google AI Studio / Gemini Setup
        model_name = GLOBAL_CONFIG["EMBEDDING_MODEL"]
        if not model_name.startswith("models/"):
             model_name = f"models/{model_name}"
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:embedContent"
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in your .env file.")
            
        print(f"  [Config] Initializing Google AI Studio (Gemini) API Engine: {model_name}")
        Settings.embed_model = GenericAPIEmbedding(
            api_url=api_url, 
            api_key=api_key,
            max_retries=GLOBAL_CONFIG["API_MAX_RETRIES"],
            retry_delay=GLOBAL_CONFIG["API_RETRY_DELAY"]
        )
        
    elif mode == "hf":
        # Hugging Face Inference API Setup
        model_name = GLOBAL_CONFIG["EMBEDDING_MODEL"]
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        api_key = os.getenv("HF_TOKEN") or os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError("HF_TOKEN (or EMBEDDING_API_KEY) is missing in your .env file.")
            
        print(f"  [Config] Initializing Hugging Face API Engine: {model_name}")
        Settings.embed_model = GenericAPIEmbedding(
            api_url=api_url, 
            api_key=api_key,
            max_retries=GLOBAL_CONFIG["API_MAX_RETRIES"],
            retry_delay=GLOBAL_CONFIG["API_RETRY_DELAY"]
        )
        
    elif mode == "ollama":
        # Local or Cloud-proxy Ollama Embedding Setup via Generic API
        model_name = GLOBAL_CONFIG["EMBEDDING_MODEL"]
        api_url = f"{GLOBAL_CONFIG['LLM_URL']}/api/embeddings"
        
        print(f"  [Config] Initializing Ollama Embedding Engine: {model_name}")
        Settings.embed_model = GenericAPIEmbedding(
            api_url=api_url, 
            api_key="ollama", 
            model_name=model_name,
            max_retries=GLOBAL_CONFIG["API_MAX_RETRIES"],
            retry_delay=GLOBAL_CONFIG["API_RETRY_DELAY"]
        )
        
    else:
        # Local Sentence Transformers Setup
        model_name = GLOBAL_CONFIG["EMBEDDING_MODEL"]
        print(f"  [Config] Initializing Local Embedding Engine: {model_name}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=GLOBAL_CONFIG["EMBED_BATCH_SIZE"],
            trust_remote_code=True,
            device="mps" if torch.backends.mps.is_available() else "cpu"
        )
        
    # --- CRITICAL: Apply Context Window to LlamaIndex and Underlying Model ---
    if hasattr(Settings.embed_model, 'context_window'):
        Settings.embed_model.context_window = GLOBAL_CONFIG["MODEL_CONTEXT_WINDOW"]
    
    # For local models, we must also tell the underlying sentence-transformers to not truncate at 512
    if mode == "local" and hasattr(Settings.embed_model, "_model"):
        try:
            Settings.embed_model._model.max_seq_length = GLOBAL_CONFIG["MODEL_CONTEXT_WINDOW"]
            print(f"  [Config] Verified hardware context window: {Settings.embed_model._model.max_seq_length}")
        except Exception as e:
            print(f"  [Warning] Could not set hardware max_seq_length: {e}")
    
    # Apply global LlamaIndex parameters
    Settings.chunk_size = GLOBAL_CONFIG["CHUNK_SIZE"]
    Settings.chunk_overlap = GLOBAL_CONFIG["CHUNK_OVERLAP"]

def configure_llm() -> None:
    """Sets up the heavy LLM model (Ollama) only when needed."""
    model = GLOBAL_CONFIG["LLM_MODEL"]
    print(f"  [Config] Initializing LLM Engine: {model}")
    Settings.llm = Ollama(
        model=model, 
        base_url=GLOBAL_CONFIG["LLM_URL"], 
        request_timeout=GLOBAL_CONFIG["LLM_TIMEOUT"], 
        context_window=8192
    )
