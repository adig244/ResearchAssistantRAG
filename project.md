# ArXiv RAG Research Assistant

A modular RAG (Retrieval-Augmented Generation) system for quantitative finance. This tool fetches the latest research from ArXiv, builds a persistent vector "brain," and uses Ollama for intelligent analysis.

---

## 🏗 Modular Architecture

The project is divided into specialized modules for better scalability and maintenance:

```
rag_project/
├── .env                  # Configuration (Ollama model, base URL, API key)
├── requirements.txt      # Project dependencies
├── main.py               # CONDUCTOR (CLI entry point)
│
├── arxiv_fetcher.py      # INGESTION: ArXiv search and download
├── document_processor.py # PARSING: PDF text extraction & chunking settings
├── vector_db.py          # STORAGE: ChromaDB & Vector Indexing
├── rag_agent.py          # INTELLIGENCE: Ollama config & Query Engine
│
├── .backup/              # Safety backups for easy REVERT
└── data/
    ├── papers/           # Downloaded PDF library
    └── chroma_db/        # Persistent vector database
```

---

## ⚡️ Code Flow Execution

When you run `python main.py --query "X" --ask "Y"`, the following takes place:

1.  **Selection (`main.py`)**: Parses arguments and triggers the flow.
2.  **Fetching (`arxiv_fetcher.py`)**: Downloads PDFs into `data/papers/` (skips duplicates).
3.  **Parsing (`document_processor.py`)**: 
    *   Configures local embeddings (`all-MiniLM-L6-v2`).
    *   Chunks text into 512-token pieces for optimal context.
4.  **Indexing (`vector_db.py`)**: 
    *   Connects to the persistent ChromaDB instance.
    *   Updates the vector index with the newly parsed documents.
5.  **Reasoning (`rag_agent.py`)**: 
    *   Connects to your Ollama session (Local or Cloud).
    *   Synthesizes the final answer using the relevant paper context.

---

## 🚀 Usage

### Full Workflow (Fetch + Ask)
```bash
python main.py --query "stochastic calculus finance" --max_papers 3 --ask "Summarize the key proof."
```

### Knowledge Base Query (Ask Only)
```bash
# Skips downloading, only queries papers already in your library
python main.py --ask "Compare the various DRL models mentioned."
```

---

## 🛠 Maintenance & Safety

| Action | Command |
|---|---|
| **REVERT** | Say "REVERT" in chat to restore the monolithic architecture. |
| **Reset Database** | `rm -rf data/chroma_db` (Forces a full re-index) |
| **Clear Papers** | `rm data/papers/*.pdf` |

---
*Last Refactored: April 2026 - Modular "Synthetic Brain" Phase*
