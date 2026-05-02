# ArXiv RAG Research Assistant - Turbo

A high-performance Research Assistant that utilizes Retrieval-Augmented Generation (RAG) to analyze ArXiv papers. It features a hierarchical indexing strategy and a two-stage hybrid retrieval engine.

## 🚀 Key Features

*   **Hierarchical Search Index**: Abstracts are indexed as "Summary Nodes" for high-level targeting, while full-text papers are chunked and linked to their respective summaries.
*   **Two-Stage Hybrid Retrieval**:
    *   **Phase 1**: Hybrid Vector + BM25 search on abstracts to identify relevant papers.
    *   **Phase 2**: Deep vector search within identified papers to extract precise chunks.
*   **Multi-Mode Embedding Engine**: Supports local execution (Sentence Transformers), Hugging Face Inference API, Google AI Studio (Gemini), and Ollama-based embeddings.
*   **Local LLM Integration**: Uses Ollama for local inference (defaulting to gpt-oss:120b-cloud).
*   **Optimized for Long Context**: Configured with an 8192-token context window for Jina v2 models to prevent silent truncation.
*   **Persistent Vector Storage**: Powered by ChromaDB with an optimized "Lean" storage context (metadata mapping only).

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd rag_project
    ```

2.  **Set up a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file based on `.env.example`:
    ```bash
    cp .env.example .env
    # Add your GOOGLE_API_KEY or HF_TOKEN if using API modes
    ```

## 📖 Usage

### 1. Build the Research Brain
Ingest papers on specific topics to populate your local database:
```bash
python3 main.py --build-brain "delta hedging, stochastic volatility" --max_papers 10
```

### 2. Search and Download
Search ArXiv and download new papers:
```bash
python3 main.py --query "transformer architectures" --max_papers 5
```

### 3. Ask Questions
Query your research brain:
```bash
python3 main.py --ask "What is the difference between continuous and discrete delta hedging?"
```

### 4. Advanced Filters
Filter queries by year or author:
```bash
python3 main.py --ask "recent advances in LLMs" --year "2024"
```

## ⚙️ Configuration

All parameters are centralized in `config.py`:
*   `EMBEDDING_MODE`: "local", "hf", "google", or "ollama".
*   `CHUNK_SIZE`: Default 2048 tokens.
*   `PHASE_1_TOP_K`: Number of papers to identify in the first pass.
*   `PHASE_2_TOP_K`: Number of deep chunks to retrieve for final synthesis.

## 📂 Project Structure

*   `main.py`: CLI Entrypoint.
*   `pipeline_runner.py`: Orchestrates ingestion and query workflows.
*   `arxiv_fetcher.py`: ArXiv API integration and parallel PDF downloader.
*   `document_processor.py`: Handles PDF parsing, chunking, and hierarchical node creation.
*   `retrievers.py`: Custom hybrid (Vector + BM25) and two-stage retrieval logic.
*   `vector_db.py`: ChromaDB management and "Lean" storage implementation.
*   `embeddings.py`: Flexible embedding client (Local, HF, Google, Ollama).
*   `rag_agent.py`: Final query engine construction.
*   `config.py`: Centralized parameters and environment setup.
