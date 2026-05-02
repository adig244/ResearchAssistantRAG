import os
from typing import List, Dict, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import IndexNode, BaseNode
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from config import GLOBAL_CONFIG, normalize_arxiv_id, KEY_ARXIV_ID, KEY_PARENT_ID, KEY_IS_SUMMARY

# Lazy-loaded tokenizer cache
_tokenizer = None

def _get_tokenizer():
    """Returns a cached tokenizer instance with enforced max length."""
    global _tokenizer
    if _tokenizer is None:
        model_name = GLOBAL_CONFIG["EMBEDDING_MODEL"]
        max_len = GLOBAL_CONFIG["MODEL_CONTEXT_WINDOW"]
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_len)
        except Exception as e:
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"  [Tokenizer] Loading fallback ({fallback}) due to: {str(e)[:100]}...")
            _tokenizer = AutoTokenizer.from_pretrained(fallback, model_max_length=max_len)
    return _tokenizer

def load_documents(
    papers_metadata: Optional[List[Dict[str, str]]] = None,
    chunk_size: int = 2048,
    chunk_overlap: int = 256
) -> Optional[List[BaseNode]]:
    """
    Loads PDFs and constructs a hierarchical Summary -> Chunk node mapping.
    Standardizes metadata schema and deduplicates by base ArXiv ID.
    """
    if not papers_metadata:
        return []

    print("\n--- 🏗️  Building Hierarchical Search Index (Abstracts -> Chunks) ---")
    nodes: List[BaseNode] = []
    node_parser = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        tokenizer=_get_tokenizer().encode
    )
    
    # Deduplicate by base ArXiv ID
    seen_base_ids = set()
    unique_metadata = []
    for meta in papers_metadata:
        base_id = normalize_arxiv_id(meta[KEY_ARXIV_ID])
        if base_id not in seen_base_ids:
            seen_base_ids.add(base_id)
            unique_metadata.append(meta)
    
    if len(unique_metadata) < len(papers_metadata):
        print(f"  [Dedup] Removed {len(papers_metadata) - len(unique_metadata)} duplicate versions.")
    
    for meta in unique_metadata:
        reader = SimpleDirectoryReader(input_files=[meta["pdf_path"]])
        file_docs = reader.load_data()
        
        # Extract and tag full text chunks
        child_nodes = node_parser.get_nodes_from_documents(file_docs)
        for node in child_nodes:
            node.metadata = {
                KEY_ARXIV_ID: meta[KEY_ARXIV_ID],
                "title": meta["title"],
                "authors": meta["authors"],
                "published": meta["published"],
                "year": meta["published"].split("-")[0],
                KEY_PARENT_ID: meta[KEY_ARXIV_ID],
                KEY_IS_SUMMARY: False
            }

        # Create the parent Summary Node
        tokenizer = _get_tokenizer()
        tokens = tokenizer(meta["summary"], max_length=chunk_size, truncation=True)
        truncated_summary = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

        summary_node = IndexNode(
            text=truncated_summary,
            index_id=meta[KEY_ARXIV_ID],
            metadata={
                KEY_ARXIV_ID: meta[KEY_ARXIV_ID],
                "title": meta["title"],
                KEY_IS_SUMMARY: True
            }
        )
        
        nodes.append(summary_node)
        nodes.extend(child_nodes)
    
    print(f"  Successfully processed {len(unique_metadata)} unique papers.")
    return nodes
