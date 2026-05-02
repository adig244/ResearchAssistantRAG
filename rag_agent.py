from typing import Optional
from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from config import GLOBAL_CONFIG, KEY_IS_SUMMARY
from retrievers import CustomVectorRetriever, CustomBM25Retriever, LinearTwoStageRetriever
from vector_db import get_collection, get_summary_nodes

def _build_bm25_retriever(similarity_top_k: int) -> Optional[CustomBM25Retriever]:
    """Initializes the BM25 retriever using nodes from the storage layer."""
    abstract_nodes = get_summary_nodes()
    if not abstract_nodes:
        return None
    return CustomBM25Retriever(abstract_nodes, similarity_top_k=similarity_top_k)

def create_query_engine(index, year_filter: Optional[str] = None, author_filter: Optional[str] = None):
    """
    Orchestrates the Hybrid Linear Two-Stage Query Engine.
    """
    # 1. Define metadata filters for abstracts
    where_dict = {KEY_IS_SUMMARY: True}
    if year_filter:
        where_dict["year"] = str(year_filter)
    if author_filter:
        where_dict["authors"] = author_filter
        
    if len(where_dict) > 1:
        where_dict = {"$and": [{k: v} for k, v in where_dict.items()]}

    collection = get_collection()
    
    # 2. Build Base Retrievers (Vector and BM25)
    vector_retriever = CustomVectorRetriever(
        collection=collection,
        similarity_top_k=GLOBAL_CONFIG["PHASE_1_TOP_K"], 
        embed_model=Settings.embed_model,
        where=where_dict
    )

    bm25_retriever = _build_bm25_retriever(GLOBAL_CONFIG["PHASE_1_TOP_K"])
    
    # 3. Fuse them if BM25 is available
    if not bm25_retriever:
        base_retriever = vector_retriever
    else:
        base_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=[GLOBAL_CONFIG["VECTOR_WEIGHT"], GLOBAL_CONFIG["BM25_WEIGHT"]],
            similarity_top_k=GLOBAL_CONFIG["PHASE_1_TOP_K"],
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False
        )
        # HACK: Override node resolution to avoid redundant docstore lookups in the lean index.
        # See LlamaIndex issue #11603 for context on hybrid lean storage.
        base_retriever._resolve_nodes = lambda nodes: nodes

    # 4. Wrap in the Two-Stage logic
    linear_retriever = LinearTwoStageRetriever(
        index, 
        base_retriever, 
        similarity_top_k=GLOBAL_CONFIG["PHASE_2_TOP_K"]
    )

    return RetrieverQueryEngine.from_args(linear_retriever, streaming=True)
