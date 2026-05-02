import os
import arxiv
import concurrent.futures
import time
from typing import List, Dict, Optional
from config import GLOBAL_CONFIG, KEY_ARXIV_ID

def _process_single_paper(result: arxiv.Result, download_dir: str) -> Optional[Dict[str, str]]:
    """Helper to process and optionally download a single ArXiv result."""
    # Sanity Check: Skip broken records
    if not result.summary or not result.pdf_url:
        print(f"  [Skip] Broken or placeholder record: {result.title}")
        return None

    arxiv_id = result.get_short_id().replace("/", "_")
    pdf_filename = f"{arxiv_id}.pdf"
    pdf_path = os.path.join(download_dir, pdf_filename)
    
    # 1. Download the Full Paper (if it doesn't exist)
    if not os.path.exists(pdf_path):
        print(f"  [Download] Fetching PDF -> {pdf_filename}")
        try:
            result.download_pdf(dirpath=download_dir, filename=pdf_filename)
        except Exception as e:
            print(f"  [Error] Failed to download {pdf_filename}. Details: {e}")
            return None # Skip adding partial metadata if download fails
    else:
        print(f"  [Cache] PDF already exists -> {pdf_filename}")
    
    # 2. Capture Metadata for the "Brain"
    return {
        KEY_ARXIV_ID: arxiv_id,
        "title": result.title,
        "authors": ", ".join([a.name for a in result.authors]),
        "published": result.published.strftime("%Y-%m-%d"),
        "summary": result.summary,
        "pdf_path": pdf_path,
        "url": result.pdf_url
    }

def fetch_papers(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Finds and downloads papers from ArXiv in parallel, returning a list of metadata dictionaries.
    """
    download_dir = GLOBAL_CONFIG["PAPER_DOWNLOAD_DIR"]
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Searching ArXiv for: '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Exhaust the generator to get the raw results
    results = list(client.results(search))
    print(f"Identified {len(results)} potential matches. Beginning parallel download...")
    
    papers_metadata = []
    
    # Use ThreadPoolExecutor for I/O bound parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map the download function to all results
        futures = {executor.submit(_process_single_paper, res, download_dir): res for res in results}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                meta = future.result()
                if meta:
                    papers_metadata.append(meta)
            except Exception as e:
                print(f"  [Error] Unhandled exception during processing: {e}")
            
    print(f"Done. Processed {len(papers_metadata)} papers.")
    return papers_metadata

def fetch_papers_by_ids(arxiv_ids: List[str]) -> List[Dict[str, str]]:
    """Fetches metadata for a specific list of ArXiv IDs and processes them."""
    if not arxiv_ids:
        return []
        
    download_dir = GLOBAL_CONFIG["PAPER_DOWNLOAD_DIR"]
    print(f"  Fetching metadata for {len(arxiv_ids)} specific IDs from ArXiv...")
    
    client = arxiv.Client()
    search = arxiv.Search(id_list=arxiv_ids)
    
    # Retry loop for ArXiv results retrieval (handles 429)
    results = []
    max_retries = GLOBAL_CONFIG.get("ARXIV_MAX_RETRIES", 3)
    retry_delay = GLOBAL_CONFIG.get("ARXIV_RETRY_DELAY", 5)

    for i in range(max_retries):
        try:
            results = list(client.results(search))
            break
        except Exception as e:
            if "429" in str(e) and i < max_retries - 1:
                print(f"  [ArXiv] Rate limited (429). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
    
    papers_metadata = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_process_single_paper, res, download_dir): res for res in results}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                meta = future.result()
                if meta:
                    papers_metadata.append(meta)
            except Exception as e:
                print(f"  [Error] Failed to fetch metadata for an ID: {e}")
                
    return papers_metadata
