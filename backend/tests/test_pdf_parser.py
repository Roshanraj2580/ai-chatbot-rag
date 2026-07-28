"""
Unit tests for PDF parsing and text extraction.
"""
import pytest
from backend.ingest import PDFIngestor


def test_pdf_ingestor_initialization():
    """Test that PDFIngestor can be initialized."""
    ingestor = PDFIngestor()
    assert ingestor is not None
    assert ingestor.embedder is not None
    assert ingestor.chroma_client is not None
    assert ingestor.markdown_converter is not None
    assert ingestor.chunker is not None


# Note: Full PDF parsing tests require sample PDFs
# These should be added when test data is available
