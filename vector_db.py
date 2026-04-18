import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

def initialize_vector_db(
    nodes=None, 
    chroma_db_dir: str = "data/chroma_db", 
    collection_name: str = "quant_papers"
):
    """Initializes ChromaDB and builds/updates the persistent hierarchical index."""
    print("Connecting to ChromaDB Vector Store...")
    db = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        # Load the index if it exists in ChromaDB
        print("  Checking for existing persistent index...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        
        # If new nodes are provided (e.g. from --build-brain), add them
        if nodes:
            print(f"  Updating index with {len(nodes)} new hierarchical nodes...")
            index.insert_nodes(nodes)
            
    except Exception as e:
        # If no index exists, create it from the provided nodes
        if not nodes:
            print("  No index found and no new papers provided. Brain is empty.")
            return None
            
        print(f"  Creating a fresh hierarchical index...")
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context, 
            show_progress=True
        )
        
    return index