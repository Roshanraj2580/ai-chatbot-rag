"""
Unit tests for embedding functionality (Backward Compatibility).
"""
import pytest
import numpy as np
from backend.embedder import get_embedder, reset_embedder


def test_embedder_initialization():
    """Test that embedder initializes without errors."""
    reset_embedder()
    embedder = get_embedder(provider="local")
    assert embedder is not None
    assert embedder.provider_name == "local"


def test_embedder_encode_single():
    """Test encoding single text."""
    reset_embedder()
    embedder = get_embedder(provider="local")
    texts = ["This is a test sentence."]
    
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 768  # Fixed dimension
    
    # Check normalization (L2 norm should be ~1)
    norm = np.linalg.norm(embeddings[0])
    assert 0.99 < norm < 1.01


def test_embedder_encode_multiple():
    """Test encoding multiple texts."""
    reset_embedder()
    embedder = get_embedder(provider="local")
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 768
    
    # Check all are normalized
    for i in range(3):
        norm = np.linalg.norm(embeddings[i])
        assert 0.99 < norm < 1.01


def test_embedder_consistency():
    """Test that same text produces same embedding."""
    reset_embedder()
    embedder = get_embedder(provider="local")
    text = "Consistency test sentence."
    
    emb1 = embedder.encode([text])
    emb2 = embedder.encode([text])
    
    # Should be identical
    assert np.allclose(emb1, emb2)


def test_embedder_empty_list():
    """Test encoding empty list."""
    reset_embedder()
    embedder = get_embedder(provider="local")
    embeddings = embedder.encode([])
    assert len(embeddings) == 0