"""Gemini API embedder using google-genai SDK with retry logic."""
import os
import logging
import time
import numpy as np
from typing import List, Optional

from google import genai
from google.genai import types

from backend.interfaces.embedder import (
    EmbedderInterface,
    EMBEDDING_DIMENSION,
    EmbeddingError
)

logger = logging.getLogger(__name__)


class GeminiEmbedder(EmbedderInterface):
    """
    Google Gemini API embedder using the new google-genai SDK.
    
    Features:
    - Uses genai.Client for centralized API access
    - 3-retry logic with exponential backoff (1s, 2s, 4s)
    - Strict dimension validation (must be 768)
    - Batched embedding for large text lists
    
    Constructor Args:
        api_key (Optional[str]): Gemini API key. Defaults to GOOGLE_API_KEY env var.
        model_name (str): Embedding model. Default "gemini-embedding-001".
        max_retries (int): Max retry attempts. Default 3.
        initial_delay (float): Initial retry delay in seconds. Default 1.0.
    
    Raises:
        ValueError: If api_key not provided and GOOGLE_API_KEY not set
        ValueError: If embedding dimension != EMBEDDING_DIMENSION
    """
    
    DEFAULT_MODEL = "gemini-embedding-001"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY required for Gemini embeddings. "
                "Set via parameter or environment variable."
            )
        
        # Create client (NEW google-genai pattern)
        self.client = genai.Client(api_key=self.api_key)
        
        # Configuration
        self.model_name = model_name or os.getenv(
            "GEMINI_EMBEDDING_MODEL", 
            self.DEFAULT_MODEL
        )
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self._dimension = EMBEDDING_DIMENSION
        
        logger.info(
            f"GeminiEmbedder initialized: model={self.model_name}, "
            f"dim={self._dimension}, retries={self.max_retries}"
        )
    
    def _embed_with_retry(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        """
        Call Gemini embed API with exponential backoff retry.
        
        Args:
            texts: List of texts to embed
            task_type: Optional task type (e.g. RETRIEVAL_DOCUMENT)
        
        Returns:
            List[List[float]]: List of embedding vectors
        
        Raises:
            EmbeddingError: After max_retries failures
        """
        last_error = None
        delay = self.initial_delay
        
        config = types.EmbedContentConfig(
            output_dimensionality=self._dimension
        )
        if task_type:
            # Map common lowercase to what API might expect if needed, 
            # or pass through. 'retrieval_document' -> 'RETRIEVAL_DOCUMENT' usually.
            config.task_type = task_type.upper()
        
        for attempt in range(self.max_retries):
            try:
                # NEW google-genai SDK pattern
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=config
                )
                # Extract embeddings from response
                return [emb.values for emb in response.embeddings]
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        
        raise EmbeddingError(
            f"Gemini API failed after {self.max_retries} attempts: {last_error}"
        )
    
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Generate embeddings via Gemini API.
        
        Args:
            texts: List of text strings to embed
            batch_size: Texts per API call (default 32)
            **kwargs: Support 'task_type'
        
        Returns:
            np.ndarray: Shape (len(texts), 768), L2-normalized
        
        Raises:
            EmbeddingError: If API calls fail after retries
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        task_type = kwargs.get("task_type")
        
        # Clean texts
        texts = [t.replace("\n", " ").strip() for t in texts]
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_with_retry(batch, task_type=task_type)
            all_embeddings.extend(embeddings)
        
        # Convert to numpy and normalize
        result = np.array(all_embeddings, dtype=np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        result = result / norms
        
        return result
    
    @property
    def embedding_dim(self) -> int:
        """Returns EMBEDDING_DIMENSION (768)."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Returns 'gemini'."""
        return "gemini"

    @property
    def active_model_name(self) -> str:
        """Returns model name."""
        return self.model_name