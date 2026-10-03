"""Local embedder using sentence-transformers."""
import os
import logging
import numpy as np
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from backend.interfaces.embedder import (
    EmbedderInterface,
    EMBEDDING_DIMENSION,
    EmbeddingError 
)

logger = logging.getLogger(__name__)


class LocalEmbedder(EmbedderInterface):
    """
    Local embedding service using SentenceTransformer (nomic-embed-text-v1).
    Produces 768-dimensional L2-normalized embeddings.
    """
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LocalEmbedder.
        
        Args:
            model_path: Path to local model directory. If None, checks EMBEDDING_MODEL_PATH
                        or defaults to ./models/nomic-embed-text-v1.
        
        Raises:
            FileNotFoundError: If the model path does not exist.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(base_dir, "models", "nomic-embed-text-v1")
        self.model_path = model_path or os.getenv("EMBEDDING_MODEL_PATH", default_path)
        
        # Verify model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Local model not found at '{self.model_path}'. "
                f"Run 'python download_model.py' to download the model first, "
                f"or set EMBEDDING_PROVIDER=gemini in your .env file."
            )
        
        try:
            self.model = SentenceTransformer(self.model_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model: {e}")
            raise EmbeddingError(f"Failed to load local model: {e}")

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (always 768)."""
        return EMBEDDING_DIMENSION

    @property
    def provider_name(self) -> str:
        """Provider name."""
        return "local"

    @property
    def active_model_name(self) -> str:
        """Active model identifier."""
        return self.MODEL_NAME

    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode texts into 768-dimensional normalized vectors.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            np.ndarray: Matrix of shape (len(texts), 768)
            
        Raises:
            EmbeddingError: If dimension mismatches or encoding fails
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIMENSION:
            raise EmbeddingError(
                f"Model dimension mismatch: expected {EMBEDDING_DIMENSION}, got {actual_dim}"
            )

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise EmbeddingError(f"Local embedding failed: {e}")
