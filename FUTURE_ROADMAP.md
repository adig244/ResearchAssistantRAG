# 🚀 Synthetic Quant RAG Assistant: Future Roadmap

This document tracks the long-term vision and planned features for the Research Assistant.

## 📥 Data Sources & Types
- [ ] **News Integration**: Real-time headline and article ingestion (HTTP/RSS).
- [ ] **Multimedia Support**: YouTube transcript fetching and image (Multimodal) analysis.
- [ ] **Technical Data**: Support for Excel/CSV pricing logs and PPT presentations.

## 🖥️ Frontend & UI/UX
- [ ] **Web Dashboard**: Interactive frontend (Streamlit, Flask, or Next.js).
- [ ] **Rich Output**: Tables, graphs, and structured market summaries in the UI.
- [ ] **Streaming Responses**: Real-time text generation in the frontend.

## 🧠 Core System Updates
- [ ] **Semantic Chunking**: Use AI to split text based on meaning shifts for better context preservation.
- [ ] **Master Project Integration**: Seamless merging with the parent "Synthetic Quant" codebase.
- [ ] **Dedicated Prompt Template**: Implement a customized, Quant-focused prompt template.
- [ ] **Technical Model Upgrade**: Switch to a higher-dimensionality embedding model (e.g., 1024-dim).
- [ ] **Multi-Agent Orchestration**: Switching between "News Analyst" and "Paper Researcher" agents.

## 🛡️ Stability & Safety (New)
- [ ] **Chunk Starvation Guard**: Implement limits so one paper doesn't dominate all search results, ensuring diverse answers.
- [ ] **Manual Fallback System**: Add a way to manually extract summaries and links for ArXiv papers that break the scraper or miss default data.
- [ ] **API Security / Safety**: Sanitize and rigidly filter paper IDs so malicious users cannot inject dangerous characters if exposed to the web later.
