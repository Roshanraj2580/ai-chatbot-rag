"""
Chroma DB client for local vector storage with duckdb+parquet persistence.
"""
import os
import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaClient:
    """Client for interacting with local Chroma vector database."""
    
    def __init__(self, persist_directory: str = "./chroma_data", collection_name: str = "documents"):
        """
        Initialize Chroma client with persistence.
        
        Args:
            persist_directory: Directory for duckdb+parquet persistence
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        logger.info(f"ChromaDB initialized. Persist dir: {persist_directory}")
        logger.info(f"Collection '{self.collection_name}' ready. Current count: {self.collection.count()}")
    
    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add document chunks to the collection.
        
        Args:
            ids: List of unique IDs (format: <doc_id>___<chunk_index>)
            documents: List of markdown chunk texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts (doc_id, source_filename, page_number, chunk_index, ingested_at)
        """
        if not ids or len(ids) == 0:
            logger.warning("No chunks to add")
            return
        
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length")
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} chunks to collection. Total count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def query_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        user_id: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Query for similar chunks using vector similarity with optional user isolation.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (default: 5)
            user_id: Optional user identifier to isolate results to this user
        
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        try:
            where_filter = {"user_id": user_id} if user_id else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
            
            # Flatten results (query returns list of lists)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_documents_by_doc_id(self, doc_id: str, user_id: Optional[str] = None) -> Dict[str, List]:
        """
        Get all chunks for a specific document, optionally verifying user ownership.
        
        Args:
            doc_id: Document ID to filter by
            user_id: Optional user ID to ensure ownership
        
        Returns:
            Dict with keys: ids, documents, metadatas
        """
        try:
            where_filter = {"$and": [{"doc_id": doc_id}, {"user_id": user_id}]} if user_id else {"doc_id": doc_id}
            results = self.collection.get(
                where=where_filter
            )
            return results
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise
    
    def get_all_documents(self, user_id: Optional[str] = None) -> Dict[str, List]:
        """
        Get all chunks from the collection, optionally scoped to a user.
        
        Args:
            user_id: Optional user identifier to filter documents
        
        Returns:
            Dict with keys: ids, documents, metadatas
        """
        try:
            where_filter = {"user_id": user_id} if user_id else None
            results = self.collection.get(where=where_filter)
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            raise
    
    def delete_document(self, doc_id: str, user_id: Optional[str] = None) -> int:
        """
        Delete all chunks for a specific document, optionally enforcing user ownership.
        
        Args:
            doc_id: Document ID to delete
            user_id: Optional user ID to ensure ownership
        
        Returns:
            Number of chunks deleted
        """
        try:
            # Get IDs first
            where_filter = {"$and": [{"doc_id": doc_id}, {"user_id": user_id}]} if user_id else {"doc_id": doc_id}
            results = self.collection.get(where=where_filter)
            ids_to_delete = results["ids"]
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id: {doc_id} (user: {user_id})")
            
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def count(self, user_id: Optional[str] = None) -> int:
        """Get total number of chunks in collection, optionally for a user."""
        if user_id:
            try:
                res = self.collection.get(where={"user_id": user_id})
                return len(res.get("ids", []))
            except Exception:
                return 0
        return self.collection.count()
    
    def reset(self) -> None:
        """Delete all data from collection (use with caution!)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("Collection reset - all data deleted")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise


# Singleton instance
_chroma_instance: Optional[ChromaClient] = None


def get_chroma_client(persist_directory: Optional[str] = None) -> ChromaClient:
    """
    Get or create the singleton Chroma client instance.
    
    Args:
        persist_directory: Optional custom persist directory
    
    Returns:
        ChromaClient instance
    """
    global _chroma_instance
    
    if _chroma_instance is None:
        persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        
        # Determine collection name based on provider
        provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        collection_name = "documents_gemini" if provider == "gemini" else "documents"
        
        _chroma_instance = ChromaClient(persist_directory=persist_dir, collection_name=collection_name)
    
    return _chroma_instance
