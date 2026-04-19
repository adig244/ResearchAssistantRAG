from arxiv_fetcher import fetch_papers
from document_processor import load_documents
from vector_db import initialize_vector_db, filter_existing_ids
from rag_agent import create_query_engine
from config import configure_llm

def run_ingestion_pipeline(topics, max_papers):
    """Handles fetching, delta-checking, and indexing of new papers."""
    papers_metadata = []
    
    # 1. Fetching
    print(f"\n--- 🧠 Building Synthetic Brain for: {topics} ---")
    for topic in topics:
                        meta = fetch_papers(topic, max_results=max_papers)
                        papers_metadata.extend(meta)
    
    if not papers_metadata:
        print("  No papers found.")
        return

    # 2. Delta Check
    ids_to_check = [p["arxiv_id"] for p in papers_metadata]
    new_paper_ids = filter_existing_ids(ids_to_check)
    
    new_papers = [p for p in papers_metadata if p["arxiv_id"] in new_paper_ids]
    for p in papers_metadata:
        if p["arxiv_id"] not in new_paper_ids:
            print(f"  [Skip] '{p['title'][:50]}...' is already in your brain.")
    
    if not new_papers:
        print("  All papers found are already indexed. Brain is up to date.")
        return
        
    # 3. Structural Indexing
    print("\n--- 🏗️  Updating Hierarchical Index ---")
    nodes = load_documents(new_papers)
    if nodes:
        initialize_vector_db(nodes)


def run_query_pipeline(question, year=None, author=None):
    """Handles engine orchestration and querying."""
    # 1. Verification
    index = initialize_vector_db()
    if not index:
        print("\nError: No research brain found. Please use --build-brain or a standard query first.")
        return
        
    # 2. Preparation
    print(f"\n--- ⚙️  Orchestrating Hybrid Engine ---")
    configure_llm()
    query_engine = create_query_engine(
        index, 
        year_filter=year, 
        author_filter=author
    )
    
    # 3. Execution
    print(f"\n--- ❓ Question: {question} ---")
    print("\n--- 🤖 Answer ---")
    response = query_engine.query(question)
    print(response)
