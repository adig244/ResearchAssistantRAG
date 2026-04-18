import os
import uuid
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_document_settings():
    """Configures global settings for local embeddings and chunking."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.chunk_size = 512

def load_documents(papers_metadata=None):
    """Loads PDFs and constructs a hierarchical Summary -> Chunk node mapping."""
    papers_dir = "data/papers"
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        print(f"Warning: No papers found in {papers_dir}")
        return None

    print("\n--- 🏗️  Building Hierarchical Search Index (Abstracts -> Chunks) ---")
    nodes = []

    if papers_metadata:
        node_parser = SentenceSplitter(chunk_size=Settings.chunk_size)
        
        for meta in papers_metadata:
            # 1. Load the PDF pages
            reader = SimpleDirectoryReader(input_files=[meta["pdf_path"]])
            file_docs = reader.load_data()
            
            # 2. Extract and tag full text chunks
            child_nodes = node_parser.get_nodes_from_documents(file_docs)
            for node in child_nodes:
                node.metadata.update({
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "published": meta["published"],
                    "year": meta["published"].split("-")[0]
                })
                node.doc_id = str(uuid.uuid4())

            # 3. Create the parent Summary Node (vectorized)
            summary_node = IndexNode(
                text=meta["summary"],
                index_id=meta["arxiv_id"],
                metadata={
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "published": meta["published"],
                    "year": meta["published"].split("-")[0],
                    "is_summary": True
                },
                obj=child_nodes
            )
            
            nodes.extend([summary_node] + child_nodes)
        
        print(f"  Successfully processed {len(papers_metadata)} papers.")
        return nodes
    else:
        print("  Performing bulk load (non-hierarchical fallback).")
        return SimpleDirectoryReader(papers_dir).load_data()
