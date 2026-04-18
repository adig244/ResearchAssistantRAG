import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_document_settings():
    """Sets up the embedding model and chunking parameters."""
    # 1. Embeddings — always local (fast, free, no API needed)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Chunk size tuned for memory-constrained machines
    Settings.chunk_size = 512

import uuid
from llama_index.core.schema import IndexNode

def load_documents(papers_metadata=None):
    """
    Loads PDFs and creates a hierarchical structure:
    Abstract (Summary/IndexNode) -> Page Segments (TextNodes).
    """
    papers_dir = "data/papers"
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        print(f"Warning: No papers found in the data directory.")
        return None

    print("Building Hierarchical Search Index (Abstracts -> Chunks)...")
    
    hierarchical_nodes = []

    # If we have specific metadata, we process those files with links
    if papers_metadata:
        from llama_index.core.node_parser import SentenceSplitter
        node_parser = SentenceSplitter(chunk_size=Settings.chunk_size)
        
        for meta in papers_metadata:
            # 1. Load the PDF
            reader = SimpleDirectoryReader(input_files=[meta["pdf_path"]])
            file_docs = reader.load_data()
            
            # 2. Extract full text nodes (Chunks)
            nodes = node_parser.get_nodes_from_documents(file_docs)
            for node in nodes:
                node.metadata.update({
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "published": meta["published"],
                    "year": meta["published"].split("-")[0] # for easy year filtering
                })
                node.doc_id = str(uuid.uuid4()) # Unique ID for the chunk

            # 3. Create the Summary Node (Abstract) that points to the Chunks
            summary_node = IndexNode(
                text=meta["summary"],
                index_id=meta["arxiv_id"], # Link identifier
                metadata={
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "published": meta["published"],
                    "year": meta["published"].split("-")[0],
                    "is_summary": True
                },
                obj=nodes # The children nodes linked to this summary
            )
            
            hierarchical_nodes.append(summary_node)
            # We also add the child nodes directly so the search can still hit them if needed
            hierarchical_nodes.extend(nodes)
        
        print(f"  Processed {len(papers_metadata)} papers into hierarchical nodes.")
        return hierarchical_nodes
    else:
        # Standard bulk load
        print("  Performing bulk load (non-hierarchical fallback).")
        return SimpleDirectoryReader(papers_dir).load_data()
