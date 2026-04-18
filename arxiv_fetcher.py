import os
import arxiv

def fetch_papers(query: str, max_results: int = 5, download_dir: str = "data/papers"):
    """
    Finds and downloads papers from ArXiv, returning a list of metadata dictionaries.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Searching ArXiv for: '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers_metadata = []
    
    for result in client.results(search):
        arxiv_id = result.get_short_id().replace("/", "_")
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = os.path.join(download_dir, pdf_filename)
        
        # 1. Download the Full Paper (if it doesn't exist)
        print(f"Found: {result.title}")
        if not os.path.exists(pdf_path):
            print(f"  Downloading PDF -> {pdf_filename}")
            result.download_pdf(dirpath=download_dir, filename=pdf_filename)
        else:
            print(f"  PDF already in library -> {pdf_filename}")
        
        # 2. Capture Metadata for the "Brain"
        meta = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": ", ".join([a.name for a in result.authors]),
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_path": pdf_path,
            "url": result.pdf_url
        }
        papers_metadata.append(meta)
            
    print(f"Done. Processed {len(papers_metadata)} papers.")
    return papers_metadata

if __name__ == "__main__":
    # Test script locally
    fetch_papers("quantitative finance volatility", max_results=2)
