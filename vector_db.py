import os
import chromadb
from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode, TextNode

from config import GLOBAL_CONFIG, KEY_ARXIV_ID, KEY_IS_SUMMARY

# Cache the persistent client
_chroma_client = None

def _get_chroma_client() -> chromadb.PersistentClient:
    """Returns a singleton Chroma persistent client."""
    global _chroma_client
    if _chroma_client is None:
        chroma_db_dir = GLOBAL_CONFIG["CHROMA_DB_DIR"]
        _chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
    return _chroma_client

def get_collection():
    """Returns the raw ChromaDB collection object for low-level hybrid operations."""
    db = _get_chroma_client()
    return db.get_or_create_collection(
        name=GLOBAL_CONFIG["COLLECTION_NAME"],
        metadata={"hnsw:space": GLOBAL_CONFIG["HNSW_SPACE"]}
    )

def get_summary_nodes() -> List[TextNode]:
    """Retrieves all abstract/summary nodes from storage."""
    collection = get_collection()
    nodes = []
    try:
        results = collection.get(
            where={KEY_IS_SUMMARY: True}, 
            include=["documents", "metadatas"]
        )
        if results and results.get("documents"):
            for id_str, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                nodes.append(TextNode(text=doc, metadata=meta, id_=id_str))
    except Exception as e:
        print(f"  [Warning] Failed to load abstracts: {e}")
    return nodes

def filter_existing_ids(ids_to_check: List[str]) -> List[str]:
    """Returns only the ArXiv IDs from the list that are NOT already in the database."""
    if not ids_to_check:
        return []
        
    try:
        chroma_db_dir = GLOBAL_CONFIG["CHROMA_DB_DIR"]
        if not os.path.exists(chroma_db_dir):
            return ids_to_check
            
        collection = get_collection()
        results = collection.get(
            where={
                KEY_ARXIV_ID: {"$in": ids_to_check}, 
                KEY_IS_SUMMARY: True
            },
            include=["metadatas"]
        )
        
        if not results or not results["metadatas"]:
            return ids_to_check
            
        found_ids = set(meta[KEY_ARXIV_ID] for meta in results["metadatas"])
        return [aid for aid in ids_to_check if aid not in found_ids]
        
    except Exception:
        return ids_to_check

def create_index(nodes: List[BaseNode]) -> Optional[VectorStoreIndex]:
    """Builds and persists a new index from nodes."""
    print(f"  Building index with {len(nodes)} nodes...")
    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        show_progress=False,
        transformations=[Settings.embed_model]
    )
    
    persist_dir = os.path.join(GLOBAL_CONFIG["CHROMA_DB_DIR"], "storage")
    os.makedirs(persist_dir, exist_ok=True)
    storage_context.index_store.persist(os.path.join(persist_dir, "index_store.json"))
    
    print(f"  Persisted index maps to: {persist_dir}")
    return index

def load_index() -> Optional[VectorStoreIndex]:
    """Loads an existing index from disk."""
    print("Connecting to ChromaDB...")
    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    persist_dir = os.path.join(GLOBAL_CONFIG["CHROMA_DB_DIR"], "storage")
    index_map_path = os.path.join(persist_dir, "index_store.json")
    
    if not os.path.exists(index_map_path):
        return VectorStoreIndex.from_vector_store(vector_store)
        
    from llama_index.core.storage.index_store import SimpleIndexStore
    index_store = SimpleIndexStore.from_persist_path(index_map_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        index_store=index_store
    )
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)