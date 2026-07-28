"""
Embedding service - Backward compatible singleton getter.

This module maintains backward compatibility while delegating to the
new embedders package for actual implementation.
"""
import logging
from typing import Optional

from backend.interfaces.embedder import EmbedderInterface
from backend.embedders import create_embedder

logger = logging.getLogger(__name__)

# Singleton instance
_embedder_instance: Optional[EmbedderInterface] = None


def get_embedder(
    model_path: Optional[str] = None,
    provider: Optional[str] = None
) -> EmbedderInterface:
    """
    Get or create singleton embedder instance.
    
    This function maintains backward compatibility with existing code
    that calls get_embedder(). New code should prefer create_embedder()
    from the embedders package.
    
    Args:
        model_path: DEPRECATED - For backward compatibility only.
            If provided without provider, forces local mode.
        provider: Embedder provider ("auto", "gemini", "local")
            If None, reads from EMBEDDING_PROVIDER env var.
    
    Returns:
        EmbedderInterface: Singleton embedder instance
    
    Backward Compatibility:
        - get_embedder() → Uses EMBEDDING_PROVIDER env var
        - get_embedder(model_path="...") → Forces local mode
        - get_embedder(provider="gemini") → Uses Gemini
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        # Handle backward compatibility
        if model_path is not None and provider is None:
            logger.warning(
                "model_path parameter is deprecated. "
                "Use EMBEDDING_MODEL_PATH env var instead."
            )
            provider = "local"
        
        _embedder_instance = create_embedder(provider)
        logger.info(
            f"Embedder singleton created: {_embedder_instance.provider_name}"
        )
    
    return _embedder_instance


def reset_embedder() -> None:
    """Reset singleton (for testing)."""
    global _embedder_instance
    _embedder_instance = None


# Re-export for backward compatibility
__all__ = ["get_embedder", "reset_embedder", "EmbedderInterface"]
