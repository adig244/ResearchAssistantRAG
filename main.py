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
    parser.add_argument("--build-brain", "-b", type=str, help="Comma-separated topics to batch-ingest (e.g. 'ML, Quant')")
    parser.add_argument("--year", type=str, help="Filter by publication year (e.g. 2024)")
    parser.add_argument("--author", type=str, help="Filter by author name")

    args = parser.parse_args()

    papers_metadata = []

    # 1. Batch Brain Building
    if args.build_brain:
        topics = [t.strip() for t in args.build_brain.split(",")]
        print(f"\n--- 🧠 Building Synthetic Brain for: {topics} ---")
        for topic in topics:
            meta = fetch_papers(topic, max_results=args.max_papers)
            papers_metadata.extend(meta)

    # 2. Targeted Query (Single Topic)
    elif args.query:
        print(f"\n--- Fetching Papers on '{args.query}' ---")
        papers_metadata = fetch_papers(args.query, max_results=args.max_papers)

    # 3. Setup and Ask
    if args.ask:
        print(f"\n--- Orchestrating Pipeline for Query ---")
        
        setup_document_settings()
        setup_agent()
        
        # Pass metadata to processor for enrichment and doc_id tracking
        documents = load_documents(papers_metadata if papers_metadata else None)
        
        # If we have documents, we initialize with them. 
        # If not, we pass None to initialize_vector_db so it loads the existing index.
        index = initialize_vector_db(documents)
        
        if not index:
            print("Failed to initialize or load the research brain.")
            sys.exit(1)
            
        # Pass the manual filters (year/author) into the agent
        query_engine = create_query_engine(
            index, 
            year_filter=args.year, 
            author_filter=args.author
        )
        
        print(f"\n--- Question ---")
        print(args.ask)
        print("\n--- Answer ---")
        response = query_engine.query(args.ask)
        print(response)
        
    elif not (args.query or args.build_brain):
        print("Usage examples:")
        print("  Build brain: python main.py --build-brain 'stochastic calculus, risk management'")
        print("  Search & Ask: python main.py --query 'deep learning' --ask 'what are the trends?'")
        print("  Query Lib:    python main.py --ask 'summary of my library'")

if __name__ == "__main__":
    main()
