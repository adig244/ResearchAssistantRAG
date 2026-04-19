import os
import re
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition
from llama_index.core.schema import NodeWithScore
from rank_bm25 import BM25Okapi

from config import GLOBAL_CONFIG

class CustomBM25Retriever(BaseRetriever):
    """A lightweight BM25 retriever built directly on rank-bm25."""
    def __init__(self, nodes, similarity_top_k=5):
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        corpus = [self._tokenize(node.get_content()) for node in nodes]
        self._bm25 = BM25Okapi(corpus)
        super().__init__()

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _retrieve(self, query_bundle):
        query_tokens = self._tokenize(query_bundle.query_str)
        scores = self._bm25.get_scores(query_tokens)
        node_scores = [
            NodeWithScore(node=self._nodes[idx], score=float(score))
            for idx, score in enumerate(scores)
        ]
        node_scores.sort(key=lambda x: x.score, reverse=True)
        return node_scores[:self._similarity_top_k]

class LinearTwoStageRetriever(BaseRetriever):
    """Linear, two-stage retriever: Search abstracts -> Extract IDs -> SearchChunks."""
    def __init__(self, index, abstract_retriever, similarity_top_k=5):
        self._index = index
        self._abstract_retriever = abstract_retriever
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle):
        # 1. Search Phase 1: Abstracts (Surface)
        abstract_results = self._abstract_retriever.retrieve(query_bundle)
        arxiv_ids = list(set(node.node.metadata.get("arxiv_id") for node in abstract_results))
        
        if not arxiv_ids:
            return []

        # 2. Search Phase 2: Targeted Chunks (Deep)
        # Build OR filter to search chunks from ALL found papers simultaneously
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="parent_id", value=aid) for aid in arxiv_ids],
            condition=FilterCondition.OR
        )
        
        # We perform a targeted vector search on just those specific papers
        chunk_retriever = self._index.as_retriever(
            similarity_top_k=self._similarity_top_k, 
            filters=filters
        )
        return chunk_retriever.retrieve(query_bundle)

def create_query_engine(index, year_filter=None, author_filter=None):
    """Creates a high-performance linear two-stage query engine."""
    
    # 1. Global Filters (Phase 1)
    filters = []
    if year_filter:
        filters.append(ExactMatchFilter(key="year", value=str(year_filter)))
    if author_filter:
        filters.append(ExactMatchFilter(key="authors", value=author_filter))
    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # 2. Setup Vector Retriever (on abstracts)
    vector_retriever = index.as_retriever(
        similarity_top_k=GLOBAL_CONFIG["PHASE_1_TOP_K"], 
        filters=metadata_filters
    )

    # 3. Setup Custom BM25 Retriever (on abstracts)
    # Since we use Chroma-Only lean storage (no docstore), we query the DB directly
    abstract_nodes = []
    try:
        from llama_index.core.schema import TextNode
        collection = index.vector_store._collection
        results = collection.get(where={"is_summary": True}, include=["documents", "metadatas"])
        if results and results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                abstract_nodes.append(TextNode(text=doc, metadata=meta))
    except Exception as e:
        print(f"  [Warning] Failed to load abstracts for BM25: {e}")
    
    if not abstract_nodes:
        base_retriever = vector_retriever
    else:
        bm25_retriever = CustomBM25Retriever(abstract_nodes, similarity_top_k=GLOBAL_CONFIG["PHASE_1_TOP_K"])
        base_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=[GLOBAL_CONFIG["VECTOR_WEIGHT"], GLOBAL_CONFIG["BM25_WEIGHT"]],
            similarity_top_k=GLOBAL_CONFIG["PHASE_1_TOP_K"],
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False
        )

    # 4. The Linear Orchestrator (No Dictionary, No Recursion)
    linear_retriever = LinearTwoStageRetriever(
        index, 
        base_retriever, 
        similarity_top_k=GLOBAL_CONFIG["PHASE_2_TOP_K"]
    )

    return RetrieverQueryEngine.from_args(linear_retriever, streaming=True)
