import os
from typing import List, Optional
from arxiv_fetcher import fetch_papers, fetch_papers_by_ids
from document_processor import load_documents
from vector_db import create_index, load_index, filter_existing_ids
from rag_agent import create_query_engine
from config import configure_llm, GLOBAL_CONFIG, normalize_arxiv_id, KEY_ARXIV_ID

def sync_local_papers() -> List[str]:
    """Scans local paper directory and indexes missing files. Returns list of indexed IDs."""
    paper_dir = GLOBAL_CONFIG["PAPER_DOWNLOAD_DIR"]
    if not os.path.exists(paper_dir):
        return []

    local_files = [f for f in os.listdir(paper_dir) if f.endswith(".pdf")]
    if not local_files:
        return []

    local_ids = [f[:-4] for f in local_files]
    missing_ids = filter_existing_ids(local_ids)
    
    if not missing_ids:
        return local_ids

    print(f"\n--- 📂 Local Sync: Found {len(missing_ids)} unindexed papers ---")
    new_metadata = fetch_papers_by_ids(missing_ids)
    
    if new_metadata:
        nodes = load_documents(
            new_metadata, 
            chunk_size=GLOBAL_CONFIG["CHUNK_SIZE"], 
            chunk_overlap=GLOBAL_CONFIG["CHUNK_OVERLAP"]
        )
        if nodes:
            create_index(nodes)
            for p in new_metadata:
                aid = p[KEY_ARXIV_ID]
                if aid not in local_ids:
                    local_ids.append(aid)
    return local_ids

def run_ingestion_pipeline(topics: List[str], max_papers: int) -> None:
    """Handles fetching, delta-checking, and indexing of new papers."""
    already_indexed_ids = sync_local_papers()
    
    papers_metadata = []
    print(f"\n--- 🧠 Building Synthetic Brain for: {topics} ---")
    for topic in topics:
        meta = fetch_papers(topic, max_results=max_papers)
        papers_metadata.extend(meta)
    
    if not papers_metadata:
        return

    # Delta Check: Use normalized base IDs
    indexed_base_ids = {normalize_arxiv_id(aid) for aid in already_indexed_ids}
    
    # Check DB status for what isn't in folder
    to_fetch_ids = [p[KEY_ARXIV_ID] for p in papers_metadata]
    db_missing_ids = set(filter_existing_ids(to_fetch_ids))
    
    new_papers = []
    for p in papers_metadata:
        aid = p[KEY_ARXIV_ID]
        base_id = normalize_arxiv_id(aid)
        
        if base_id in indexed_base_ids or aid not in db_missing_ids:
            print(f"  [Skip] '{p['title'][:50]}...' is already in your brain.")
        else:
            new_papers.append(p)
    
    if not new_papers:
        print("  All papers found are already present. Brain is up to date.")
        return
        
    print("\n--- 🏗️  Updating Hierarchical Index ---")
    nodes = load_documents(
        new_papers,
        chunk_size=GLOBAL_CONFIG["CHUNK_SIZE"],
        chunk_overlap=GLOBAL_CONFIG["CHUNK_OVERLAP"]
    )
    if nodes:
        create_index(nodes)

def run_query_pipeline(question: str, year: Optional[str] = None, author: Optional[str] = None) -> None:
    """Handles engine orchestration and querying."""
    # 1. Verification & Setup (Order enforced: LLM first, then Index)
    configure_llm()
    index = load_index()
    
    if not index:
        print("\nError: No research brain found. Use --build-brain first.")
        return
        
    print(f"\n--- ⚙️  Orchestrating Hybrid Engine ---")
    query_engine = create_query_engine(
        index, 
        year_filter=year, 
        author_filter=author
    )
    
    print(f"\n--- ❓ Question: {question} ---")
    print("\n--- 🤖 Answer ---")
    response = query_engine.query(question)
    print(response)
