"""
Hybrid retrieval combining semantic and keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
"""
import os
import logging
from typing import List, Dict, Any, Tuple

from backend.retrieval.keyword_retriever import KeywordRetriever

logger = logging.getLogger(__name__)

# RRF constant (typically 60, higher values favor top-ranked results)
RRF_K = int(os.getenv("RRF_K", "60"))


def reciprocal_rank_fusion(
    results_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF Score = Σ (1 / (k + rank_i)) for each list where document appears
    
    This method is proven effective for combining retrieval results and
    doesn't require score normalization.
    
    Args:
        results_lists: List of ranked result lists
        k: RRF constant (default 60)
    
    Returns:
        Merged and re-ranked list sorted by RRF score
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results, 1):
            doc_id = doc["id"]
            rrf_score = 1.0 / (k + rank)
            
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            
            # Keep the first occurrence (usually from semantic search)
            if doc_id not in docs:
                docs[doc_id] = doc
    
    # Sort by RRF score (descending)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [
        {**docs[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in sorted_ids
    ]


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.
    
    Modes:
    - 'hybrid': Both semantic + keyword with RRF fusion (DEFAULT)
    - 'keyword': BM25 only with semantic fallback on error
    - 'semantic': Vector search only (original behavior)
    
    Fallback behavior:
    - If hybrid/keyword mode fails, automatically falls back to semantic
    - If keyword search returns empty, falls back to semantic
    
    Environment:
        RETRIEVAL_MODE: hybrid|keyword|semantic (default: hybrid)
        RRF_K: RRF fusion constant (default: 60)
    """
    
    def __init__(self, embedder, chroma_client):
        """
        Initialize hybrid retriever.
        
        Args:
            embedder: Embedder instance for semantic search
            chroma_client: ChromaDB client for both search types
        """
        self.embedder = embedder
        self.chroma_client = chroma_client
        self.keyword_retriever = KeywordRetriever(chroma_client)
        
        self.mode = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
        if self.mode not in ("hybrid", "keyword", "semantic"):
            logger.warning(f"Invalid RETRIEVAL_MODE '{self.mode}', defaulting to 'hybrid'")
            self.mode = "hybrid"
        
        logger.info(f"HybridRetriever initialized: mode={self.mode}, rrf_k={RRF_K}")
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform semantic (vector) search.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of result dicts
        """
        query_embedding = self.embedder.encode([query], task_type="retrieval_query")[0].tolist()
        results = self.chroma_client.query_similar(query_embedding, k=k)
        
        if not results["ids"]:
            return []
        
        return [
            {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i]
            }
            for i in range(len(results["ids"]))
        ]
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword (BM25) search.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of result dicts
        """
        return self.keyword_retriever.search(query, k=k)
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using configured mode with automatic fallback.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of retrieved chunks
        """
        try:
            if self.mode == "semantic":
                logger.debug("Using semantic-only search")
                return self._semantic_search(query, k)
            
            elif self.mode == "keyword":
                logger.debug("Using keyword-only search with semantic fallback")
                results = self._keyword_search(query, k)
                if not results:
                    logger.info("Keyword search returned no results, falling back to semantic")
                    return self._semantic_search(query, k)
                return results
            
            elif self.mode == "hybrid":
                logger.debug("Using hybrid search (semantic + keyword + RRF)")
                
                # Fetch more for each to allow RRF to work effectively
                fetch_k = k * 2
                
                semantic_results = self._semantic_search(query, fetch_k)
                keyword_results = self._keyword_search(query, fetch_k)
                
                if not semantic_results and not keyword_results:
                    logger.warning("Both semantic and keyword search returned empty")
                    return []
                
                if not keyword_results:
                    logger.info("Keyword search empty, using semantic results only")
                    return semantic_results[:k]
                
                if not semantic_results:
                    logger.info("Semantic search empty, using keyword results only")
                    return keyword_results[:k]
                
                # Merge with Reciprocal Rank Fusion
                fused = reciprocal_rank_fusion([semantic_results, keyword_results])
                
                logger.info(
                    f"Hybrid search: semantic={len(semantic_results)}, "
                    f"keyword={len(keyword_results)}, fused={len(fused)}"
                )
                
                return fused[:k]
            
            else:
                # Fallback for unknown mode
                logger.warning(f"Unknown mode '{self.mode}', using semantic")
                return self._semantic_search(query, k)
        
        except Exception as e:
            logger.error(f"Search failed in '{self.mode}' mode: {e}")
            
            # If not already in semantic mode, try semantic as fallback
            if self.mode != "semantic":
                logger.info("Falling back to semantic search due to error")
                try:
                    return self._semantic_search(query, k)
                except Exception as fallback_error:
                    logger.error(f"Semantic fallback also failed: {fallback_error}")
                    raise
            
            raise
    
    def invalidate_keyword_index(self) -> None:
        """Invalidate BM25 index (call after document changes)."""
        self.keyword_retriever.invalidate_index()
    
    def search_with_stats(self, query: str, k: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search with statistics for UI display.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            Tuple of (results, stats) where stats contains:
            - mode: Current retrieval mode
            - semantic_count: Number of semantic results (before fusion)
            - keyword_count: Number of keyword results (before fusion)
            - total_retrieved: Total before deduplication
        """
        stats = {
            "mode": self.mode,
            "semantic_count": 0,
            "keyword_count": 0,
            "total_retrieved": 0
        }
        
        try:
            if self.mode == "semantic":
                results = self._semantic_search(query, k)
                stats["semantic_count"] = len(results)
                stats["total_retrieved"] = len(results)
                return results, stats
            
            elif self.mode == "keyword":
                results = self._keyword_search(query, k)
                stats["keyword_count"] = len(results)
                
                if not results:
                    logger.info("Keyword search returned no results, falling back to semantic")
                    results = self._semantic_search(query, k)
                    stats["semantic_count"] = len(results)
                    stats["mode"] = "semantic (fallback)"
                
                stats["total_retrieved"] = len(results)
                return results, stats
            
            elif self.mode == "hybrid":
                fetch_k = k * 2
                
                semantic_results = self._semantic_search(query, fetch_k)
                keyword_results = self._keyword_search(query, fetch_k)
                
                stats["semantic_count"] = len(semantic_results)
                stats["keyword_count"] = len(keyword_results)
                
                if not semantic_results and not keyword_results:
                    stats["total_retrieved"] = 0
                    return [], stats
                
                if not keyword_results:
                    stats["total_retrieved"] = len(semantic_results)
                    return semantic_results[:k], stats
                
                if not semantic_results:
                    stats["total_retrieved"] = len(keyword_results)
                    return keyword_results[:k], stats
                
                fused = reciprocal_rank_fusion([semantic_results, keyword_results])
                stats["total_retrieved"] = len(fused)
                
                logger.info(
                    f"Hybrid search: semantic={len(semantic_results)}, "
                    f"keyword={len(keyword_results)}, fused={len(fused)}"
                )
                
                return fused[:k], stats
            
            else:
                results = self._semantic_search(query, k)
                stats["semantic_count"] = len(results)
                stats["total_retrieved"] = len(results)
                return results, stats
        
        except Exception as e:
            logger.error(f"Search failed in '{self.mode}' mode: {e}")
            
            if self.mode != "semantic":
                logger.info("Falling back to semantic search due to error")
                try:
                    results = self._semantic_search(query, k)
                    stats["semantic_count"] = len(results)
                    stats["total_retrieved"] = len(results)
                    stats["mode"] = "semantic (fallback)"
                    return results, stats
                except Exception as fallback_error:
                    logger.error(f"Semantic fallback also failed: {fallback_error}")
                    raise
            
            raise
