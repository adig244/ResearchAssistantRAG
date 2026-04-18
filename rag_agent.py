import os
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

def _configure_llm():
    """Reads OLLAMA settings and returns an Ollama LLM instance."""
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"  [LLM] Using Ollama: {model} @ {base_url}")
    
    # context_window set to bypass metadata API probing
    return Ollama(
        model=model, 
        base_url=base_url, 
        request_timeout=600.0,
        context_window=8192
    )

def setup_agent():
    """Configures the LLM agent globally in Settings."""
    Settings.llm = _configure_llm()

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def create_query_engine(index, year_filter=None, author_filter=None):
    """
    Creates a hierarchical query engine that searches abstracts 
    first, then follows links to full text chunks.
    """
    # 1. Setup Filters
    filters = []
    if year_filter:
        filters.append(ExactMatchFilter(key="year", value=str(year_filter)))
    if author_filter:
        # Note: In a production system we'd use 'contains', but for now exact match on author string
        filters.append(ExactMatchFilter(key="authors", value=author_filter))
    
    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # 2. Setup Base Query Engine (for the full chunks)
    # This acts as the "target" that the abstracts point to
    base_query_engine = index.as_query_engine(
        similarity_top_k=5, # More chunks for the final answer
        filters=metadata_filters
    )

    # 3. Setup Recursive Retriever
    # We map the "vector" key to the base retriever for the abstracts
    # and we provide the query engine for the recursive lookup
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": index.as_retriever(similarity_top_k=3, filters=metadata_filters)},
        query_engine_dict={index.index_id: base_query_engine},
        verbose=True
    )

    # 4. Wrap into a Query Engine
    return RetrieverQueryEngine.from_args(
        recursive_retriever, 
        streaming=True
    )
