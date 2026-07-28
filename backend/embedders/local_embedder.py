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
    Local embedding using nomic-embed-text-v1 via sentence-transformers.
    
    Features:
    - Lazy model loading (loads on first encode() call)
    - Automatic CPU/GPU detection
    - Strict dimension validation
    - L2-normalized output
    
    Constructor Args:
        model_path (Optional[str]): Path to model directory.
            Defaults to EMBEDDING_MODEL_PATH env var or ./models/nomic-embed-text-v1
    
    Raises:
        FileNotFoundError: If model not found at path
        ValueError: If model produces != EMBEDDING_DIMENSION dimensions
    """
    
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
    
    def __init__(self, model_path: Optional[str] = None):
        # Determine model path
        if model_path is not None:
            self.model_path = model_path
        else:
            env_path = os.getenv("EMBEDDING_MODEL_PATH")
            if env_path:
                self.model_path = env_path
            else:
                # Default: ./models/nomic-embed-text-v1 relative to project root
                base_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                self.model_path = os.path.join(
                    base_dir, "models", "nomic-embed-text-v1"
                )
        
        # State
        self._model: Optional[SentenceTransformer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dimension = EMBEDDING_DIMENSION
        
        logger.info(
            f"LocalEmbedder initialized: path={self.model_path}, "
            f"device={self._device}"
        )
    
    def _load_model(self) -> None:
        """
        Lazy load model on first use.
        
        Raises:
            FileNotFoundError: If model directory doesn't exist
            ValueError: If model dimension != EMBEDDING_DIMENSION
        """
        if self._model is not None:
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                "Run 'python download_model.py' first."
            )
        
        logger.info(f"Loading model from {self.model_path}...")
        self._model = SentenceTransformer(
            self.model_path, 
            device=self._device, 
            trust_remote_code=True
        )
        
        # Validate dimension
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Model dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {actual_dim}. Model at {self.model_path} is incompatible."
            )
        
        logger.info(
            f"Model loaded: dim={actual_dim}, device={self._device}"
        )
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode texts using local sentence-transformers model.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (default 32)
            **kwargs: Ignored by local embedder
        
        Returns:
            np.ndarray: Shape (len(texts), 768), L2-normalized
        
        Raises:
            EmbeddingError: If model loading or encoding fails
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        try:
            self._load_model()
            
            result = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # L2 normalization
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            return result
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}") from e
    
    @property
    def embedding_dim(self) -> int:
        """Returns EMBEDDING_DIMENSION (768)."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Returns 'local'."""
        return "local"
    
    @property
    def device(self) -> str:
        """Returns 'cpu' or 'cuda'."""
        return self._device
        
    @property
    def active_model_name(self) -> str:
        """Returns model name."""
        return self.MODEL_NAME