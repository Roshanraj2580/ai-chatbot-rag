"""
Unit tests for Knowledge Graph (Graph RAG) extraction and data modeling.
"""
import pytest
from unittest.mock import MagicMock
from backend.graph import extract_entities_from_text, build_knowledge_graph


def test_extract_entities_from_text():
    text = "FastAPI and Flask are web frameworks. We use ChromaDB and Redis for low-latency RAG with Gemini LLM."
    entities = extract_entities_from_text(text)
    
    assert len(entities) > 0
    # Check that key technical keywords or capitalized terms are captured
    assert any(e in ["FastAPI", "Flask", "ChromaDB", "Redis", "Gemini", "RAG"] for e in entities)


def test_build_knowledge_graph_empty():
    mock_chroma = MagicMock()
    mock_chroma.get_all_documents.return_value = {"ids": [], "metadatas": [], "documents": []}
    
    graph = build_knowledge_graph(mock_chroma)
    assert "nodes" in graph
    assert "edges" in graph
    assert "stats" in graph
    assert graph["stats"]["total_nodes"] >= 4  # Default fallback architecture nodes


def test_build_knowledge_graph_with_documents():
    mock_chroma = MagicMock()
    mock_chroma.get_all_documents.return_value = {
        "ids": ["chunk1", "chunk2"],
        "metadatas": [
            {"doc_id": "doc1", "source_filename": "manual.pdf"},
            {"doc_id": "doc2", "source_filename": "architecture.pdf"}
        ],
        "documents": [
            "This document explains FastAPI, Redis caching, and Docker deployment.",
            "This guide covers ChromaDB vector search and Redis cache latency optimization."
        ]
    }
    
    graph = build_knowledge_graph(mock_chroma)
    assert graph["stats"]["documents"] == 2
    assert graph["stats"]["total_nodes"] > 2
    assert len(graph["edges"]) > 0

    # Verify document nodes exist
    doc_node_ids = [n["id"] for n in graph["nodes"] if n.get("group") == "document"]
    assert "doc_doc1" in doc_node_ids
    assert "doc_doc2" in doc_node_ids
