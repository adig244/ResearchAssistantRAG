import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Load environment variables from .env file
load_dotenv()

def _configure_llm():
    """
    Configures the Ollama LLM. 
    It will use the model specified in OLLAMA_MODEL (e.g., 'gpt-oss:120b-cloud').
    Local Ollama handles the cloud offloading automatically for :cloud models.
    """
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"  [LLM] Using Ollama: {model} @ {base_url}")
    
    # We set context_window to a safe default to avoid Ollama API probing issues
    return Ollama(
        model=model, 
        base_url=base_url, 
        request_timeout=600.0,
        context_window=8192
    )

def setup_rag_pipeline(
    papers_dir: str = "data/papers",
    chroma_db_dir: str = "data/chroma_db",
    collection_name: str = "quant_papers",
):
    """
    Sets up the full LlamaIndex RAG pipeline:
    - Loads PDFs from papers_dir
    - Embeds them using a local sentence-transformers model
    - Stores/retrieves vectors from ChromaDB
    - Connects to Ollama for generation
    """
    # 1. Check if there are any documents to load
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        print("Warning: No papers found in the data directory. Download papers first using --query.")
        return None

    print("\nConfiguring LlamaIndex Models...")

    # 2. Embeddings — always local (fast, free, no API needed)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. LLM — Ollama (Local or Cloud-interfaced)
    Settings.llm = _configure_llm()

    # 4. Chunk size tuned for memory-constrained machines
    Settings.chunk_size = 512

    print("Loading documents...")
    documents = SimpleDirectoryReader(papers_dir).load_data()
    print(f"  Loaded {len(documents)} document pages.")

    print("Connecting to ChromaDB Vector Store...")
    db = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building / Updating VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # similarity_top_k=3 balances quality vs. context window size
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=True)
    return query_engine

if __name__ == "__main__":
    engine = setup_rag_pipeline()
    if engine:
        print("\nPipeline ready. Asking a test question...")
        response = engine.query("What are the main topics discussed in these papers?")
        response.print_response_stream()
        print()
