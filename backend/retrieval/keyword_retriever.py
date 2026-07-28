"""
BM25 keyword-based retrieval for RAG pipeline.

Provides keyword matching to complement semantic search.
"""
import logging
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """
    BM25-based keyword retrieval for documents.
    
    Features:
    - Tokenization with stopword removal
    - Case-insensitive matching
    - Lazy index building (rebuilt when document count changes)
    
    Constructor Args:
        chroma_client: ChromaDB client for fetching documents
    """
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'if',
        'then', 'else', 'so', 'just', 'also', 'only', 'very', 'too', 'more',
        'some', 'any', 'no', 'not', 'all', 'each', 'every', 'both', 'few',
        'many', 'much', 'most', 'other', 'such', 'own', 'same', 'than', 'as'
    }
    
    def __init__(self, chroma_client):
        """
        Initialize keyword retriever.
        
        Args:
            chroma_client: ChromaDB client instance
        """
        self.chroma_client = chroma_client
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._index_version: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with lowercase and stopword removal.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        # Remove stopwords and short tokens
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]
    
    def _build_index(self) -> None:
        """Build or rebuild BM25 index from ChromaDB."""
        results = self.chroma_client.get_all_documents()
        
        if not results.get("ids"):
            logger.warning("No documents in database for keyword index")
            self._bm25 = None
            self._corpus = []
            self._tokenized_corpus = []
            return
        
        self._corpus = []
        self._tokenized_corpus = []
        
        for i in range(len(results["ids"])):
            doc = {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i]
            }
            self._corpus.append(doc)
            self._tokenized_corpus.append(self._tokenize(doc["document"]))
        
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._index_version = self.chroma_client.count()
        
        logger.info(f"Built BM25 index with {len(self._corpus)} documents")
    
    def _ensure_index(self) -> bool:
        """
        Ensure index is built and up-to-date.
        
        Returns:
            True if index is ready, False otherwise
        """
        current_count = self.chroma_client.count()
        
        if self._bm25 is None or current_count != self._index_version:
            self._build_index()
        
        return self._bm25 is not None
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of document dicts with 'id', 'document', 'metadata', 'score'
        """
        if not self._ensure_index():
            logger.warning("BM25 index is empty, returning no results")
            return []
        
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            logger.warning(f"Query '{query[:50]}...' has no searchable tokens after processing")
            return []
        
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append({
                    "id": self._corpus[idx]["id"],
                    "document": self._corpus[idx]["document"],
                    "metadata": self._corpus[idx]["metadata"],
                    "score": float(scores[idx])
                })
        
        logger.debug(f"BM25 search returned {len(results)} results for '{query[:50]}...'")
        return results
    
    def invalidate_index(self) -> None:
        """Force index rebuild on next search."""
        self._bm25 = None
        self._index_version = 0
        logger.debug("BM25 index invalidated, will rebuild on next search")
