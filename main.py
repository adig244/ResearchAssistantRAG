import argparse
import sys
from dotenv import load_dotenv

from arxiv_fetcher import fetch_papers
from document_processor import setup_document_settings, load_documents
from vector_db import initialize_vector_db
from rag_agent import setup_agent, create_query_engine

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="ArXiv RAG Research Assistant")
    parser.add_argument("--query", "-q", type=str, help="Search ArXiv for a specific topic")
    parser.add_argument("--max_papers", "-m", type=int, default=5, help="Max papers per topic (default: 5)")
    parser.add_argument("--ask", "-a", type=str, help="Ask the assistant a question")
    parser.add_argument("--build-brain", "-b", type=str, help="Comma-separated topics to batch-ingest")
    parser.add_argument("--year", type=str, help="Filter by publication year (e.g. 2024)")
    parser.add_argument("--author", type=str, help="Filter by author name")

    args = parser.parse_args()
    papers_metadata = []

    # 1. Pipeline Ingestion (Batch or Single Topic)
    if args.build_brain:
        topics = [t.strip() for t in args.build_brain.split(",")]
        print(f"\n--- 🧠 Building Synthetic Brain for: {topics} ---")
        for topic in topics:
            meta = fetch_papers(topic, max_results=args.max_papers)
            papers_metadata.extend(meta)
    elif args.query:
        print(f"\n--- 🔎 Fetching Papers on: '{args.query}' ---")
        papers_metadata = fetch_papers(args.query, max_results=args.max_papers)

    # 2. Pipeline Execution (Setup & Query)
    if args.ask:
        print(f"\n--- ⚙️  Orchestrating Assistant Brain ---")
        
        setup_document_settings()
        setup_agent()
        
        # Ingest and index (hierarchical structure handles deduplication)
        nodes = load_documents(papers_metadata if papers_metadata else None)
        index = initialize_vector_db(nodes)
        
        if not index:
            print("  Error: No research brain initialized. Please download papers first.")
            sys.exit(1)
            
        # Create the Hierarchical Query Engine
        query_engine = create_query_engine(
            index, 
            year_filter=args.year, 
            author_filter=args.author
        )
        
        print(f"\n--- ❓ Question: {args.ask} ---")
        print("\n--- 🤖 Answer ---")
        response = query_engine.query(args.ask)
        print(response)
        
    elif not (args.query or args.build_brain):
        print("\nArXiv RAG Research Assistant")
        print("Usage:")
        print("  Build Brain: python3 main.py --build-brain 'ml, finance'")
        print("  Ask Info:   python3 main.py --ask 'what are the trends?'")
        print("  Filtered:   python3 main.py --ask '...' --year 2024")

if __name__ == "__main__":
    main()
