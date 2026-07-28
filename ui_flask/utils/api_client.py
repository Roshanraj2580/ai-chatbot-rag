"""
API Client utility for communicating with FastAPI backend.
Provides typed interfaces for all backend endpoints.
"""

import os
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class Citation:
    """Citation metadata."""
    source_filename: str
    page_number: Optional[int]
    chunk_index: int


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]


@dataclass
class Document:
    """Document metadata."""
    doc_id: str
    source_filename: str
    pages: Optional[int]
    chunks: int
    ingested_at: str


class BackendAPIClient:
    """Client for FastAPI backend."""
    
    def __init__(self, base_url: str = None):
        """Initialize client with backend URL."""
        self.base_url = base_url or os.getenv("BACKEND_URL", "https://ai-rag-chatbot-backend-pnnq.onrender.com")
    
    def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def ask(self, query: str) -> RAGResponse:
        """Ask a question using RAG."""
        response = requests.post(
            f"{self.base_url}/ask",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        return RAGResponse(
            answer=data["answer"],
            citations=[Citation(**c) for c in data.get("citations", [])],
            retrieved_chunks=data.get("retrieved_chunks", [])
        )
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload a PDF document."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/documents",
                files=files,
                timeout=300  # 5 minutes for large files
            )
            response.raise_for_status()
            return response.json()
    
    def list_documents(self) -> List[Document]:
        """List all documents."""
        response = requests.get(f"{self.base_url}/documents", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return [Document(**d) for d in data]
    
    def get_document(self, doc_id: str) -> Document:
        """Get specific document details."""
        response = requests.get(f"{self.base_url}/documents/{doc_id}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return Document(**data)
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document."""
        response = requests.delete(f"{self.base_url}/documents/{doc_id}", timeout=30)
        response.raise_for_status()
        return response.json()
    

