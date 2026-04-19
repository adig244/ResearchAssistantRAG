import argparse
import sys
from dotenv import load_dotenv

from config import configure_base_settings
from pipeline_runner import run_ingestion_pipeline, run_query_pipeline

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="ArXiv RAG Research Assistant - Turbo")
    parser.add_argument("--query", "-q", type=str, help="Search ArXiv for a specific topic")
    parser.add_argument("--max_papers", "-m", type=int, default=5, help="Max papers per topic (default: 5)")
    parser.add_argument("--ask", "-a", type=str, help="Ask the assistant a question")
    parser.add_argument("--build-brain", "-b", type=str, help="Comma-separated topics to batch-ingest")
    parser.add_argument("--year", type=str, help="Filter by publication year (e.g. 2024)")
    parser.add_argument("--author", type=str, help="Filter by author name")

    args = parser.parse_args()
    
    # 1. Base Setup (Embeddings settings needed globally)
    configure_base_settings()
    
    # 2. Route to Pipelines
    if args.build_brain or args.query:
        # User is either using batch builder or standard query fetch builder
        topic_str = args.build_brain if args.build_brain else args.query
        topics = [t.strip() for t in topic_str.split(",")]
        run_ingestion_pipeline(topics, max_papers=args.max_papers)

    if args.ask:
        run_query_pipeline(args.ask, year=args.year, author=args.author)
        
    elif not (args.query or args.build_brain):
        print("\nArXiv RAG Research Assistant - Turbo")
        print("Usage: python3 main.py --build-brain 'ml' --ask 'what is X?'")

if __name__ == "__main__":
    main()

