import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from backend.ingest import PDFIngestor

@pytest.fixture
def mock_dependencies():
    with patch('backend.ingest.fitz') as mock_fitz, \
         patch('backend.ingest.get_embedder') as mock_get_embedder, \
         patch('backend.ingest.get_chroma_client') as mock_get_chroma:
         
        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2]])
        # active_model_name is a property, so we need to configure it on the mock instance or type
        type(mock_embedder).active_model_name = PropertyMock(return_value="test_model_v1")
        mock_get_embedder.return_value = mock_embedder
        
        # Mock Chroma
        mock_chroma = MagicMock()
        mock_get_chroma.return_value = mock_chroma
        
        # Mock PyMuPDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content for page 1"
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__iter__.return_value = iter([mock_page])
        
        mock_fitz.open.return_value = mock_doc
        
        yield mock_fitz, mock_embedder, mock_chroma

from unittest.mock import PropertyMock

def test_process_pdf_metadata(mock_dependencies):
    mock_fitz, mock_embedder, mock_chroma = mock_dependencies
    
    # We need to ensure PropertyMock is set up correctly if strictly mocking class
    # But usually MagicMock handles properties if assigned directly?
    # Let's just set the attribute for simplicity if possible, but property needs PropertyMock
    
    ingestor = PDFIngestor()
    result = ingestor.process_pdf("dummy.pdf", "dummy.pdf")
    
    # Check if add_chunks was called
    assert mock_chroma.add_chunks.called
    
    # Check arguments passed to add_chunks
    args, kwargs = mock_chroma.add_chunks.call_args
    
    # method signature: add_chunks(ids, documents, embeddings, metadatas)
    # verify if used as positional or keyword
    if 'metadatas' in kwargs:
        metadatas = kwargs['metadatas']
    else:
        metadatas = args[3]
    
    assert len(metadatas) > 0
    assert "embedding_model" in metadatas[0]
    assert metadatas[0]["embedding_model"] == "test_model_v1"
