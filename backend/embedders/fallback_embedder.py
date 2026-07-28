"""Fallback embedder wrapper with strict dimension validation."""
import logging
import numpy as np
from typing import List

from backend.interfaces.embedder import (
    EmbedderInterface, 
    EMBEDDING_DIMENSION, 
    EmbeddingError
)

logger = logging.getLogger(__name__)


class FallbackEmbedder(EmbedderInterface):
    """
    Embedder with automatic fallback from primary to secondary.
    
    Features:
    - Tries primary embedder first (e.g., Gemini API)
    - Falls back to secondary on any error (e.g., local model)
    - STRICT dimension validation at construction time
    - Tracks which provider was last used
    
    Constructor Args:
        primary (EmbedderInterface): Primary embedder (tried first)
        secondary (EmbedderInterface): Fallback embedder (used on error)
    
    Raises:
        ValueError: If primary.embedding_dim != EMBEDDING_DIMENSION
        ValueError: If secondary.embedding_dim != EMBEDDING_DIMENSION
        ValueError: If primary.embedding_dim != secondary.embedding_dim
    
    Example:
        >>> primary = GeminiEmbedder()
        >>> secondary = LocalEmbedder()
        >>> embedder = FallbackEmbedder(primary, secondary)
        >>> # If Gemini fails, automatically uses local
        >>> embeddings = embedder.encode(["hello world"])
    """
    
    def __init__(
        self, 
        primary: EmbedderInterface, 
        secondary: EmbedderInterface
    ):
        # STRICT validation at construction time
        if primary.embedding_dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Primary embedder dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, got {primary.embedding_dim}"
            )
        
        if secondary.embedding_dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Secondary embedder dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, got {secondary.embedding_dim}"
            )
        
        if primary.embedding_dim != secondary.embedding_dim:
            raise ValueError(
                f"Dimension mismatch between embedders: "
                f"primary={primary.embedding_dim}, secondary={secondary.embedding_dim}"
            )
        
        self.primary = primary
        self.secondary = secondary
        self._active_provider = primary.provider_name
        self._dimension = EMBEDDING_DIMENSION
        
        logger.info(
            f"FallbackEmbedder initialized: "
            f"primary={primary.provider_name}, secondary={secondary.provider_name}, "
            f"dim={self._dimension}"
        )
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode texts, falling back to secondary on error.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            **kwargs: Additional arguments passed to embedders
        
        Returns:
            np.ndarray: Shape (len(texts), 768), L2-normalized
            Guaranteed to have consistent dimensions regardless of which
            provider succeeds.
        
        Raises:
            EmbeddingError: If both primary AND secondary fail
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        # Try primary first
        try:
            result = self.primary.encode(texts, batch_size, **kwargs)
            self._active_provider = self.primary.provider_name
            logger.debug(f"Primary embedder ({self.primary.provider_name}) succeeded")
            return result
            
        except Exception as e:
            logger.warning(
                f"Primary embedder ({self.primary.provider_name}) failed: {e}"
            )
            logger.info(
                f"Falling back to secondary embedder ({self.secondary.provider_name})"
            )
        
        # Fallback to secondary
        try:
            result = self.secondary.encode(texts, batch_size, **kwargs)
            self._active_provider = self.secondary.provider_name
            return result
            
        except Exception as e:
            raise EmbeddingError(
                f"Both embedders failed. "
                f"Primary ({self.primary.provider_name}): see previous warning. "
                f"Secondary ({self.secondary.provider_name}): {e}"
            ) from e
    
    @property
    def embedding_dim(self) -> int:
        """Returns EMBEDDING_DIMENSION (768)."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Returns 'fallback' with active provider info."""
        return f"fallback({self._active_provider})"
    
    @property
    def active_provider(self) -> str:
        """Returns the provider that was last used successfully."""
        return self._active_provider
    
    @property
    def active_model_name(self) -> str:
        """Returns the active model name."""
        if self._active_provider == self.primary.provider_name:
            return self.primary.active_model_name
        return self.secondary.active_model_name