"""Tests for embedder factory."""
import os
import pytest
from unittest.mock import patch
from backend.embedders import create_embedder
from backend.embedders.local_embedder import LocalEmbedder
from backend.embedders.gemini_embedder import GeminiEmbedder
from backend.embedders.fallback_embedder import FallbackEmbedder

def test_create_local():
    with patch("backend.embedders.local_embedder.os.path.exists", return_value=True):
        embedder = create_embedder("local")
        assert isinstance(embedder, LocalEmbedder)

def test_create_gemini():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
        embedder = create_embedder("gemini")
        assert isinstance(embedder, GeminiEmbedder)

def test_create_auto_success():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
        with patch("backend.embedders.local_embedder.os.path.exists", return_value=True):
            embedder = create_embedder("auto")
            assert isinstance(embedder, FallbackEmbedder)
            assert isinstance(embedder.primary, GeminiEmbedder)
            assert isinstance(embedder.secondary, LocalEmbedder)

def test_create_auto_no_key_fallback():
    # If GOOGLE_API_KEY missing, GeminiEmbedder raises ValueError, factory catches and returns Local
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    
    with patch("backend.embedders.local_embedder.os.path.exists", return_value=True):
        embedder = create_embedder("auto")
        assert isinstance(embedder, LocalEmbedder)

def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedder("invalid")
