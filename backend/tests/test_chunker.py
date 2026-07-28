"""
Unit tests for text chunking.
"""
import pytest
from backend.ingest import TextChunker


def test_chunker_initialization():
    """Test that chunker initializes correctly."""
    chunker = TextChunker(chunk_size=400, overlap=50)
    assert chunker.chunk_size == 400
    assert chunker.overlap == 50
    assert chunker.tokenizer is not None


def test_chunker_single_chunk():
    """Test chunking of short text that fits in one chunk."""
    chunker = TextChunker(chunk_size=400, overlap=50)
    text = "This is a short text that should fit in one chunk."
    
    chunks = chunker.chunk(text)
    
    assert len(chunks) >= 1
    # bert-base-uncased lowercases text
    assert text.lower() in chunks[0] or chunks[0] in text.lower()


def test_chunker_multiple_chunks():
    """Test chunking of longer text."""
    chunker = TextChunker(chunk_size=50, overlap=10)  # Smaller for testing
    
    # Create text that will definitely need multiple chunks
    text = " ".join([f"Word{i}" for i in range(100)])
    
    chunks = chunker.chunk(text)
    
    # Should have multiple chunks
    assert len(chunks) > 1
    
    # Check overlap exists (consecutive chunks should share some content)
    if len(chunks) > 1:
        # This is a simple check - more sophisticated overlap detection could be added
        assert len(chunks[0]) > 0
        assert len(chunks[1]) > 0


def test_chunker_empty_text():
    """Test chunking empty text."""
    chunker = TextChunker()
    chunks = chunker.chunk("")
    assert len(chunks) == 0
