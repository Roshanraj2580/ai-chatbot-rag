"""
Unit tests for Chroma DB operations.
"""
import pytest
import tempfile
import shutil
from backend.chroma_client import ChromaClient


@pytest.fixture
def temp_chroma():
    """Create temporary Chroma instance for testing."""
    temp_dir = tempfile.mkdtemp()
    client = ChromaClient(persist_directory=temp_dir)
    yield client
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_chroma_initialization(temp_chroma):
    """Test Chroma client initialization."""
    assert temp_chroma is not None
    assert temp_chroma.collection is not None
    assert temp_chroma.count() == 0


def test_chroma_add_and_retrieve(temp_chroma):
    """Test adding chunks and retrieving them."""
    # Add test chunks
    ids = ["doc1___0", "doc1___1"]
    documents = ["First chunk text", "Second chunk text"]
    embeddings = [[0.1] * 768, [0.2] * 768]  # Dummy embeddings
    metadatas = [
        {"doc_id": "doc1", "source_filename": "test.pdf", "page_number": 1, "chunk_index": 0, "ingested_at": "2025-01-01"},
        {"doc_id": "doc1", "source_filename": "test.pdf", "page_number": 1, "chunk_index": 1, "ingested_at": "2025-01-01"}
    ]
    
    temp_chroma.add_chunks(ids, documents, embeddings, metadatas)
    
    # Check count
    assert temp_chroma.count() == 2
    
    # Query
    query_embedding = [0.15] * 768
    results = temp_chroma.query_similar(query_embedding, k=2)
    
    assert len(results["ids"]) == 2
    assert len(results["documents"]) == 2


def test_chroma_get_by_doc_id(temp_chroma):
    """Test retrieving chunks by doc_id."""
    # Add test chunks
    ids = ["doc1___0", "doc2___0"]
    documents = ["Doc1 chunk", "Doc2 chunk"]
    embeddings = [[0.1] * 768, [0.2] * 768]
    metadatas = [
        {"doc_id": "doc1", "source_filename": "test1.pdf", "page_number": 1, "chunk_index": 0, "ingested_at": "2025-01-01"},
        {"doc_id": "doc2", "source_filename": "test2.pdf", "page_number": 1, "chunk_index": 0, "ingested_at": "2025-01-01"}
    ]
    
    temp_chroma.add_chunks(ids, documents, embeddings, metadatas)
    
    # Get doc1
    results = temp_chroma.get_documents_by_doc_id("doc1")
    
    assert len(results["ids"]) == 1
    assert results["ids"][0] == "doc1___0"
