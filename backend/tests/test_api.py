"""
Integration tests for FastAPI endpoints using TestClient.
Mocks external services (Chroma, Gemini, Embedder) to test API logic.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np

# Set dummy API key for testing
os.environ["GOOGLE_API_KEY"] = "test_key"

from backend.app import app

client = TestClient(app)

@pytest.fixture
def mock_chroma():
    with patch("backend.app.get_chroma_client") as mock:
        client_instance = mock.return_value
        # Default behaviors
        client_instance.count.return_value = 5
        client_instance.get_all_documents.return_value = {
            "ids": ["doc1_0"],
            "metadatas": [{
                "doc_id": "doc1",
                "source_filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "ingested_at": "2024-01-01T00:00:00"
            }]
        }
        client_instance.get_documents_by_doc_id.return_value = {
            "ids": ["doc1_0"],
            "metadatas": [{
                "doc_id": "doc1",
                "source_filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "ingested_at": "2024-01-01T00:00:00"
            }]
        }
        client_instance.delete_document.return_value = 1
        yield mock

@pytest.fixture
def mock_ingestor():
    with patch("backend.app.get_ingestor") as mock:
        ingestor = mock.return_value
        ingestor.process_pdf.return_value = {
            "doc_id": "doc123",
            "status": "ingested",
            "chunks": 10,
            "failed_pages": []
        }
        yield mock

@pytest.fixture
def mock_query_service():
    with patch("backend.app.get_query_service") as mock:
        service = mock.return_value
        service.answer_query.return_value = {
            "answer": "This is a test answer.",
            "citations": [{
                "source_filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 0
            }],
            "retrieved_chunks": [{
                "id": "doc1_0",
                "document": "Test content",
                "metadata": {"source_filename": "test.pdf"},
                "distance": 0.1
            }]
        }
        yield mock

def test_health_check(mock_chroma):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "chroma_count" in data
    assert "timestamp" in data

def test_upload_document(mock_ingestor):
    # Create a dummy PDF file content
    file_content = b"%PDF-1.4\n..."
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    response = client.post("/documents", files=files)
    
    assert response.status_code == 201
    data = response.json()
    assert data["doc_id"] == "doc123"
    assert data["status"] == "ingested"
    assert data["chunks"] == 10

def test_upload_invalid_file(mock_ingestor):
    files = {"file": ("test.txt", b"text content", "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400

def test_list_documents(mock_chroma):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["doc_id"] == "doc1"
    assert data[0]["source_filename"] == "test.pdf"

def test_get_document(mock_chroma):
    response = client.get("/documents/doc1")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "doc1"
    assert data["chunks"] == 1

def test_get_document_not_found(mock_chroma):
    mock_chroma.return_value.get_documents_by_doc_id.return_value = {"ids": [], "metadatas": []}
    response = client.get("/documents/nonexistent")
    assert response.status_code == 404

def test_delete_document(mock_chroma):
    response = client.delete("/documents/doc1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["deleted_chunks"] == 1

def test_ask_query_get(mock_query_service, mock_chroma):
    response = client.get("/ask?query=test")
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert len(data["citations"]) == 1

def test_ask_query_post(mock_query_service, mock_chroma):
    response = client.post("/ask", json={"query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."

def test_ask_query_empty_db(mock_chroma):
    mock_chroma.return_value.count.return_value = 0
    response = client.post("/ask", json={"query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I don't know."
    assert len(data["retrieved_chunks"]) == 0
