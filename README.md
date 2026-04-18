# 🧠 ArXiv RAG Research Assistant

A high-performance, modular RAG (Retrieval-Augmented Generation) system for Quantitative Finance. It transforms ArXiv research papers into a persistent, searchable "Synthetic Brain" using a two-tier hierarchical search architecture.

## 🚀 Key Features

- **Two-Tier Search**: Uses a `RecursiveRetriever` to scan paper **Abstracts** first, then "zooms in" to only the most relevant **Page Segments**.
- **Persistent Brain**: Built on **ChromaDB**. Your library is indexed once and searchable forever across sessions.
- **Intelligent Deduplication**: Automatically avoids indexing the same paper twice using unique ArXiv IDs and content hashing.
- **Metadata Filtering**: Native support for manual CLI filters (e.g., `--year`, `--author`) to target specific research.
- **Fully Local & Private**: Runs on your hardware using **Ollama** and **HuggingFace** embeddings.

## 🛠 Project Structure

- `main.py`: CLI Orchestrator for brain-building and querying.
- `arxiv_fetcher.py`: Handles paper discovery and PDF downloads.
- `document_processor.py`: Implements the hierarchical node structure (Abstract -> Chunks).
- `vector_db.py`: Manages persistent storage and indexing in ChromaDB.
- `rag_agent.py`: The intelligence layer (RecursiveRetriever + LLM synthesis).

## 📦 Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama**:
   Ensure [Ollama](https://ollama.ai/) is running and you have the required model pulled:
   ```bash
   ollama pull gpt-oss:120b-cloud  # or your preferred model
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your settings.

## 🏃 Usage

### 1. Build your "Synthetic Brain"
Ingest papers on specific topics to populate your local library:
```bash
python3 main.py --build-brain "hft, limit order books" --max_papers 5
```

### 2. Ask a Research Question
Query your library with the hierarchical search engine:
```bash
python3 main.py --ask "What are the common feature selection methods for HFT?"
```

### 3. Filtered Search
Target specific years or authors:
```bash
python3 main.py --ask "Latest trends in volatility" --year 2024 --author "M. Smith"
```

## 🛡 License
MIT
