"""Tests for GeminiEmbedder."""
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.embedders.gemini_embedder import GeminiEmbedder
from backend.interfaces.embedder import EMBEDDING_DIMENSION, EmbeddingError

@pytest.fixture
def mock_genai_client():
    with patch("backend.embedders.gemini_embedder.genai.Client") as mock:
        yield mock

@pytest.fixture
def gemini_embedder(mock_genai_client):
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
        return GeminiEmbedder()

def test_initialization(gemini_embedder):
    assert gemini_embedder.provider_name == "gemini"
    assert gemini_embedder.embedding_dim == 768
    assert gemini_embedder.active_model_name == GeminiEmbedder.DEFAULT_MODEL

def test_encode(gemini_embedder, mock_genai_client):
    # Mock response
    mock_response = MagicMock()
    # Create mock embeddings with values property
    emb1 = MagicMock()
    emb1.values = [0.1] * 768
    emb2 = MagicMock()
    emb2.values = [0.2] * 768
    mock_response.embeddings = [emb1, emb2]
    
    mock_genai_client.return_value.models.embed_content.return_value = mock_response
    
    texts = ["hello", "world"]
    embeddings = gemini_embedder.encode(texts)
    
    assert embeddings.shape == (2, 768)
    mock_genai_client.return_value.models.embed_content.assert_called_once()
    
    # Check L2 normalization
    # The mock values [0.1]*768 are not normalized, but encode() normalizes them
    assert np.allclose(np.linalg.norm(embeddings[0]), 1.0)

def test_encode_with_task_type(gemini_embedder, mock_genai_client):
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1]*768)]
    mock_genai_client.return_value.models.embed_content.return_value = mock_response
    
    gemini_embedder.encode(["test"], task_type="RETRIEVAL_QUERY")
    
    _, kwargs = mock_genai_client.return_value.models.embed_content.call_args
    assert kwargs['config'].task_type == "RETRIEVAL_QUERY"

def test_retry_logic(gemini_embedder, mock_genai_client):
    # Fail twice, then succeed
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1]*768)]
    
    mock_genai_client.return_value.models.embed_content.side_effect = [
        Exception("Error 1"),
        Exception("Error 2"),
        mock_response
    ]
    
    # Reduce delay for test speed
    gemini_embedder.initial_delay = 0.01
    
    gemini_embedder.encode(["test"])
    
    assert mock_genai_client.return_value.models.embed_content.call_count == 3

def test_retry_failure(gemini_embedder, mock_genai_client):
    # Fail always
    mock_genai_client.return_value.models.embed_content.side_effect = Exception("Permanent error")
    
    gemini_embedder.initial_delay = 0.01
    
    with pytest.raises(EmbeddingError):
        gemini_embedder.encode(["test"])
