import chromadb
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import GLOBAL_CONFIG

def filter_existing_ids(ids_to_check: list):
    """Returns only the ArXiv IDs from the list that are NOT already in the database."""
    if not ids_to_check:
        return []
        
    try:
        chroma_db_dir = GLOBAL_CONFIG["CHROMA_DB_DIR"]
        collection_name = GLOBAL_CONFIG["COLLECTION_NAME"]
        
        if not os.path.exists(chroma_db_dir):
            return ids_to_check
            
        db = chromadb.PersistentClient(path=chroma_db_dir)
        collections = [c.name for c in db.list_collections()]
        if collection_name not in collections:
            return ids_to_check
            
        collection = db.get_collection(collection_name)
        
        # Surgical lookup: only get nodes with matching arxiv_ids from our list
        # This uses Chroma's internal index ($in)
        results = collection.get(
            where={"arxiv_id": {"$in": ids_to_check}, "is_summary": True},
            include=["metadatas"]
        )
        
        if not results or not results["metadatas"]:
            return ids_to_check
            
        found_ids = set(meta["arxiv_id"] for meta in results["metadatas"])
        return [aid for aid in ids_to_check if aid not in found_ids]
        
    except Exception:
        return ids_to_check

def initialize_vector_db(nodes=None):
    """Initializes ChromaDB with Hybrid-Lean storage (no redundant docstore)."""
    chroma_db_dir = GLOBAL_CONFIG["CHROMA_DB_DIR"]
    collection_name = GLOBAL_CONFIG["COLLECTION_NAME"]
    
    print("Connecting to ChromaDB Vector Store...")
    db = chromadb.PersistentClient(path=chroma_db_dir)
    
    # Fix: HNSW Math Alignment (Cosine matches MiniLM embedding training mathematically)
    chroma_collection = db.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": GLOBAL_CONFIG["HNSW_SPACE"]}
    )
    
    # We tell the store that IT is responsible for the text
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # The storage folder now ONLY holds the tiny map files
    persist_dir = os.path.join(chroma_db_dir, "storage")

    try:
        if nodes:
            print(f"  Building index with {len(nodes)} nodes...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # This handles the Math + Metadata in ChromaDB
            index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
            
            # --- Hybrid-Lean Persistence ---
            # We ONLY save the index/vector maps, skipping the massive docstore.json
            os.makedirs(persist_dir, exist_ok=True)
            storage_context.index_store.persist(os.path.join(persist_dir, "index_store.json"))
            storage_context.vector_store.persist(os.path.join(persist_dir, "vector_store.json"))
            
            print(f"  Persisted index maps to: {persist_dir} (Redundant docstore skipped)")
            return index
        else:
            # Loading: We reconstruct the brain using the specific Map file
            index_map_path = os.path.join(persist_dir, "index_store.json")
            if not os.path.exists(index_map_path):
                print(f"  No index maps found. Performing auto-discovery...")
                return VectorStoreIndex.from_vector_store(vector_store)
                
            print(f"  Loading index maps from: {persist_dir}")
            
            # Surgeons move: Load the specific index store without triggering a broad folder search
            from llama_index.core.storage.index_store import SimpleIndexStore
            index_store = SimpleIndexStore.from_persist_path(index_map_path)
            
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, 
                index_store=index_store
            )
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            
    except Exception as e:
        # Catch everything and provide a clean summary
        print(f"\n[Database Error] Details: {str(e)[:500]}...") 
        return None