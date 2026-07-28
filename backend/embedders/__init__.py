"""Embedders package with factory function."""
import os
import logging
from typing import Optional

from backend.interfaces.embedder import EmbedderInterface, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


def create_embedder(provider: Optional[str] = None) -> EmbedderInterface:
    """
    Factory function to create configured embedder.

    Args:
        provider: Embedding provider selection
            - "auto" (default): Gemini primary with local fallback
            - "gemini": Gemini API only
            - "local": Local model only
            - None: Read from EMBEDDING_PROVIDER env var

    Returns:
        EmbedderInterface: Configured embedder instance
        All embedders produce exactly EMBEDDING_DIMENSION (768) dimensions.

    Raises:
        ValueError: If unknown provider specified
        ValueError: If gemini/auto requested but GOOGLE_API_KEY not set
        FileNotFoundError: If local requested but model not downloaded

    Environment Variables:
        EMBEDDING_PROVIDER: Default provider if not specified
        GOOGLE_API_KEY: Required for gemini/auto providers
        EMBEDDING_MODEL_PATH: Optional custom local model path
        GEMINI_EMBEDDING_MODEL: Optional custom Gemini model name

    Example:
        >>> embedder = create_embedder("auto")
        >>> embeddings = embedder.encode(["hello", "world"])
        >>> embeddings.shape
        (2, 768)

    Note:
        Imports of the concrete embedder classes (LocalEmbedder,
        GeminiEmbedder, FallbackEmbedder) are done lazily inside each
        provider branch below, not at module import time. This avoids
        pulling in heavy dependencies (torch, sentence-transformers) via
        LocalEmbedder unless local/auto mode actually needs them. Because
        of this, LocalEmbedder/GeminiEmbedder/FallbackEmbedder are not
        re-exported from this package -- import create_embedder() instead,
        or import the classes directly from their own modules
        (e.g. `from backend.embedders.gemini_embedder import GeminiEmbedder`)
        if you need a class reference directly.
    """
    # Get provider from param or env
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "auto")
    provider = provider.lower().strip()

    logger.info(f"Creating embedder: provider={provider}, dim={EMBEDDING_DIMENSION}")

    if provider == "local":
        from .local_embedder import LocalEmbedder

        logger.info("Using local-only embedder (nomic-embed-text-v1)")
        return LocalEmbedder()

    elif provider == "gemini":
        from .gemini_embedder import GeminiEmbedder

        logger.info("Using Gemini-only embedder")
        return GeminiEmbedder()

    elif provider == "auto":
        from .gemini_embedder import GeminiEmbedder
        from .local_embedder import LocalEmbedder
        from .fallback_embedder import FallbackEmbedder

        logger.info("Using auto mode: Gemini primary with local fallback")

        # Try to create Gemini embedder
        try:
            primary = GeminiEmbedder()
        except ValueError as e:
            logger.warning(f"Cannot create Gemini embedder: {e}")
            logger.info("Falling back to local-only mode")
            return LocalEmbedder()

        # Create local fallback
        secondary = LocalEmbedder()

        return FallbackEmbedder(primary, secondary)

    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Valid options: 'auto', 'gemini', 'local'"
        )


# Re-export for convenience.
# Only create_embedder is exported eagerly -- the concrete embedder classes
# are imported lazily inside create_embedder() to avoid loading heavy
# dependencies (torch, sentence-transformers) unless local/auto mode is
# actually selected. Import the classes directly from their own modules if
# you need them outside of create_embedder().
__all__ = ["create_embedder"]