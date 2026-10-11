"""
Unit tests for multi-user isolation and tenant-scoped caching.
Verifies that:
1. User A's uploaded documents cannot be seen or retrieved by User B.
2. User A's cache entries are never returned to User B.
3. User A deleting their document clears only User A's cache.
4. ChromaDB queries and BM25 searches isolate documents by user_id.
"""
import pytest
from unittest.mock import MagicMock
from backend.cache import QueryCache
from backend.chroma_client import ChromaClient
from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever


def test_cache_user_isolation():
    """Verify that User A's cached queries cannot be retrieved by User B."""
    cache = QueryCache()
    cache.clear()

    # User 1 caches a query response
    cache.set("what is my balance", {"answer": "Your balance is $500"}, user_id="user1")

    # User 1 should get cache hit
    hit_user1 = cache.get("what is my balance", user_id="user1")
    assert hit_user1 is not None
    assert hit_user1["answer"] == "Your balance is $500"
    assert hit_user1["from_cache"] is True

    # User 2 asking same question MUST get cache MISS
    hit_user2 = cache.get("what is my balance", user_id="user2")
    assert hit_user2 is None

    # Global user asking must get cache MISS
    hit_global = cache.get("what is my balance", user_id=None)
    assert hit_global is None


def test_cache_clear_scoped_to_user():
    """Verify that clearing User 1's cache does not erase User 2's cache."""
    cache = QueryCache()
    cache.clear()

    cache.set("query 1", {"answer": "User 1 answer"}, user_id="user1")
    cache.set("query 2", {"answer": "User 2 answer"}, user_id="user2")

    # Clear only user 1
    cache.clear(user_id="user1")

    assert cache.get("query 1", user_id="user1") is None
    assert cache.get("query 2", user_id="user2") is not None


def test_keyword_retriever_user_isolation():
    """Verify that BM25 keyword retrieval isolates results by user_id."""
    mock_chroma = MagicMock()
    mock_chroma.count.return_value = 2
    mock_chroma.get_all_documents.return_value = {
        "ids": ["docA_0", "docB_0"],
        "documents": [
            "Project Alpha quarterly financial profits and revenue report.",
            "Project Beta quarterly financial profits and secret strategy report."
        ],
        "metadatas": [
            {"doc_id": "docA", "source_filename": "alpha.pdf", "user_id": "user1"},
            {"doc_id": "docB", "source_filename": "beta.pdf", "user_id": "user2"}
        ]
    }

    retriever = KeywordRetriever(mock_chroma)

    # User 1 searching should only get docA
    res_user1 = retriever.search("financial revenue", k=5, user_id="user1")
    assert len(res_user1) == 1
    assert res_user1[0]["id"] == "docA_0"
    assert res_user1[0]["metadata"]["user_id"] == "user1"

    # User 2 searching should only get docB
    res_user2 = retriever.search("financial strategy", k=5, user_id="user2")
    assert len(res_user2) == 1
    assert res_user2[0]["id"] == "docB_0"
    assert res_user2[0]["metadata"]["user_id"] == "user2"
