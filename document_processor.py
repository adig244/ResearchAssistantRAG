import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from config import GLOBAL_CONFIG

def load_documents(papers_metadata=None):
    """Loads PDFs and constructs a hierarchical Summary -> Chunk node mapping."""
    papers_dir = GLOBAL_CONFIG["PAPER_DOWNLOAD_DIR"]
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        print(f"Warning: No papers found in {papers_dir}")
        return None

    print("\n--- 🏗️  Building Hierarchical Search Index (Abstracts -> Chunks) ---")
    nodes = []

    if papers_metadata:
        # Optimization: One parser for the whole batch
        # Fix: Token Mismatch (Use correct tokenizer for the embedding model)
        tokenizer = AutoTokenizer.from_pretrained(GLOBAL_CONFIG["EMBEDDING_MODEL"])
        node_parser = SentenceSplitter(
            chunk_size=Settings.chunk_size, 
            chunk_overlap=Settings.chunk_overlap,
            tokenizer=tokenizer.encode
        )
        
        for meta in papers_metadata:
            # Universal Reader: Auto-detects based on file extension
            reader = SimpleDirectoryReader(input_files=[meta["pdf_path"]])
            file_docs = reader.load_data()
            
            # 2. Extract and tag full text chunks
            child_nodes = node_parser.get_nodes_from_documents(file_docs)
            for node in child_nodes:
                node.metadata = {
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "authors": meta["authors"], # Now expected as a pre-joined string
                    "published": meta["published"],
                    "year": meta["published"].split("-")[0],
                    "parent_id": meta["arxiv_id"]
                }

            # 3. Create the parent Summary Node
            summary_node = IndexNode(
                text=meta["summary"],
                index_id=meta["arxiv_id"],
                metadata={
                    "arxiv_id": meta["arxiv_id"],
                    "title": meta["title"],
                    "is_summary": True
                }
            )
            
            nodes.extend([summary_node] + child_nodes)
        
        print(f"  Successfully processed {len(papers_metadata)} papers.")
        return nodes
