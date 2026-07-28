"""Tests for LocalEmbedder."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.embedders.local_embedder import LocalEmbedder
from backend.interfaces.embedder import EMBEDDING_DIMENSION, EmbeddingError

@pytest.fixture
def mock_sentence_transformer():
    with patch("backend.embedders.local_embedder.SentenceTransformer") as mock:
        mock_instance = mock.return_value
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_instance.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        yield mock

def test_initialization(mock_sentence_transformer):
    embedder = LocalEmbedder()
    assert embedder.provider_name == "local"
    assert embedder.embedding_dim == 768
    assert embedder.active_model_name == LocalEmbedder.MODEL_NAME

def test_encode(mock_sentence_transformer):
    embedder = LocalEmbedder()
    texts = ["hello", "world"]
    
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape == (2, 768)
    mock_sentence_transformer.return_value.encode.assert_called()

def test_dimension_mismatch_error():
    with patch("backend.embedders.local_embedder.SentenceTransformer") as mock:
        mock.return_value.get_sentence_embedding_dimension.return_value = 384 # Wrong dim
        
        embedder = LocalEmbedder()
        with pytest.raises(EmbeddingError, match="Model dimension mismatch"):
            embedder.encode(["test"])
