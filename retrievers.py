import math
import re
from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition
from rank_bm25 import BM25Okapi

from config import GLOBAL_CONFIG, KEY_ARXIV_ID, KEY_PARENT_ID

CHROMA_INCLUDE_FIELDS = ["documents", "metadatas", "distances"]

class CustomVectorRetriever(BaseRetriever):
    """Bypasses LlamaIndex's internal docstore logic by hitting Chroma directly."""
    def __init__(self, collection, similarity_top_k: int, embed_model, where: Optional[dict] = None):
        self._collection = collection
        self._similarity_top_k = similarity_top_k
        self._embed_model = embed_model
        self._where = where
        super().__init__()
        
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_agg_embedding_from_queries([query_bundle.query_str])
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._similarity_top_k,
            where=self._where,
            include=CHROMA_INCLUDE_FIELDS
        )
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
            
        node_scores = []
        for node_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            node = TextNode(text=text, metadata=metadata, id_=node_id)
            # Convert distance to score
            node_scores.append(NodeWithScore(node=node, score=math.exp(-distance)))
            
        return node_scores

class CustomBM25Retriever(BaseRetriever):
    """A lightweight BM25 retriever built directly on rank-bm25."""
    def __init__(self, nodes: List[TextNode], similarity_top_k: int = 5):
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self._tokenizer_pattern = re.compile(r"\w+")
        corpus = [self._tokenize(node.get_content()) for node in nodes]
        self._bm25 = BM25Okapi(corpus)
        super().__init__()

    def _tokenize(self, text: str) -> List[str]:
        return self._tokenizer_pattern.findall(text.lower())

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
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
    def __init__(self, index, abstract_retriever: BaseRetriever, similarity_top_k: int = 5):
        self._index = index
        self._abstract_retriever = abstract_retriever
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # 1. Search Phase 1: Abstracts (Surface)
        raw_results = self._abstract_retriever.retrieve(query_bundle)
        
        # Deduplicate by arxiv_id to handle unique paper representation
        unique_results = []
        seen_ids = set()
        for node in raw_results:
            aid = node.node.metadata.get(KEY_ARXIV_ID)
            if not aid:
                unique_results.append(node)
            elif aid not in seen_ids:
                unique_results.append(node)
                seen_ids.add(aid)
        
        arxiv_ids = list(seen_ids)
        print(f"\n  [Phase 1] Retrieved {len(raw_results)} summary nodes. Unique papers: {len(arxiv_ids)}")
        
        if not arxiv_ids:
            return []

        # 2. Search Phase 2: Targeted Chunks (Deep)
        # Filter for chunks belonging to identified papers AND explicitly not summaries
        chunk_filters = [ExactMatchFilter(key=KEY_PARENT_ID, value=aid) for aid in arxiv_ids]
        filters = MetadataFilters(
            filters=chunk_filters,
            condition=FilterCondition.OR
        )
        
        chunk_retriever = self._index.as_retriever(
            similarity_top_k=self._similarity_top_k, 
            filters=filters
        )
        
        chunk_results = chunk_retriever.retrieve(query_bundle)
        print(f"  [Phase 2] Retrieved {len(chunk_results)} deep chunks from targeted papers.")
        return chunk_results
