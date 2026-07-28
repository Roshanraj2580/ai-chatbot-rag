"""Tests for FallbackEmbedder."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.embedders.fallback_embedder import FallbackEmbedder
from backend.interfaces.embedder import EmbedderInterface, EmbeddingError

@pytest.fixture
def mock_primary():
    mock = MagicMock(spec=EmbedderInterface)
    mock.embedding_dim = 768
    mock.provider_name = "primary"
    mock.active_model_name = "primary-model"
    return mock

@pytest.fixture
def mock_secondary():
    mock = MagicMock(spec=EmbedderInterface)
    mock.embedding_dim = 768
    mock.provider_name = "secondary"
    mock.active_model_name = "secondary-model"
    return mock

def test_initialization(mock_primary, mock_secondary):
    embedder = FallbackEmbedder(mock_primary, mock_secondary)
    assert embedder.provider_name == "fallback(primary)"
    assert embedder.embedding_dim == 768
    assert embedder.active_provider == "primary"

def test_encode_primary_success(mock_primary, mock_secondary):
    mock_primary.encode.return_value = np.zeros((1, 768))
    
    embedder = FallbackEmbedder(mock_primary, mock_secondary)
    embedder.encode(["test"])
    
    mock_primary.encode.assert_called_once()
    mock_secondary.encode.assert_not_called()
    assert embedder.active_provider == "primary"
    assert embedder.active_model_name == "primary-model"

def test_encode_fallback_success(mock_primary, mock_secondary):
    mock_primary.encode.side_effect = Exception("Primary failed")
    mock_secondary.encode.return_value = np.zeros((1, 768))
    
    embedder = FallbackEmbedder(mock_primary, mock_secondary)
    embedder.encode(["test"])
    
    mock_primary.encode.assert_called_once()
    mock_secondary.encode.assert_called_once()
    assert embedder.active_provider == "secondary"
    assert embedder.active_model_name == "secondary-model"

def test_encode_both_fail(mock_primary, mock_secondary):
    mock_primary.encode.side_effect = Exception("Primary failed")
    mock_secondary.encode.side_effect = Exception("Secondary failed")
    
    embedder = FallbackEmbedder(mock_primary, mock_secondary)
    
    with pytest.raises(EmbeddingError):
        embedder.encode(["test"])

def test_dimension_mismatch_error(mock_primary):
    bad_secondary = MagicMock(spec=EmbedderInterface)
    bad_secondary.embedding_dim = 384 # Mismatch
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        FallbackEmbedder(mock_primary, bad_secondary)
