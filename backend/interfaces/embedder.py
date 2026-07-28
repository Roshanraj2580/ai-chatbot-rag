"""Abstract interface for embedding services."""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

# Enforced embedding dimension - all embedders MUST produce this dimension
EMBEDDING_DIMENSION = 768


class EmbeddingError(Exception):
    """Raised when embedding operation fails."""
    pass


class EmbedderInterface(ABC):
    """
    Abstract interface for embedding services.
    
    All implementations must:
    - Produce embeddings of exactly EMBEDDING_DIMENSION (768) dimensions
    - Return L2-normalized vectors
    - Be thread-safe for concurrent encode() calls
    """
    
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            **kwargs: Additional provider-specific arguments (e.g., task_type)
        
        Returns:
            np.ndarray: Shape (len(texts), EMBEDDING_DIMENSION), L2-normalized
        
        Raises:
            EmbeddingError: If embedding operation fails
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            int: Must be EMBEDDING_DIMENSION (768)
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return provider name for logging.
        
        Returns:
            str: One of "local", "gemini", or "fallback"
        """
        pass

    @property
    @abstractmethod
    def active_model_name(self) -> str:
        """
        Return the name of the model being used.
        
        Returns:
            str: Model identifier (e.g. "nomic-embed-text-v1", "models/embedding-001")
        """
        pass