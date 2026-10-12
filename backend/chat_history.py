"""
Chat history persistence module for user-scoped chat logs and telemetry.
Saves conversations to SQLite (rag_app.db) and provides endpoints for retrieving
and clearing history per user.
"""
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from backend.auth import DB_PATH, get_db_connection

logger = logging.getLogger(__name__)


def init_chat_history_db():
    """Ensure the chat_logs table and schema are created."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                user_query TEXT NOT NULL,
                bot_answer TEXT NOT NULL,
                citations TEXT,
                retrieved_chunks_count INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                feedback INTEGER DEFAULT 0,
                tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                target_doc TEXT DEFAULT '',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # Ensure newer columns exist
        cursor.execute("PRAGMA table_info(chat_logs)")
        cols = [row[1] for row in cursor.fetchall()]
        if "tokens" not in cols:
            cursor.execute("ALTER TABLE chat_logs ADD COLUMN tokens INTEGER DEFAULT 0")
        if "cost_usd" not in cols:
            cursor.execute("ALTER TABLE chat_logs ADD COLUMN cost_usd REAL DEFAULT 0.0")
        if "target_doc" not in cols:
            cursor.execute("ALTER TABLE chat_logs ADD COLUMN target_doc TEXT DEFAULT ''")
        conn.commit()


def save_chat_interaction(
    user_id: int,
    username: str,
    query: str,
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    chunks_count: int = 0,
    latency_ms: float = 0.0,
    tokens: int = 0,
    cost_usd: float = 0.0,
    target_doc: str = ""
) -> int:
    """Save a chat message exchange into SQLite for the specified user."""
    citations_json = json.dumps(citations) if citations else "[]"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (
                    user_id, username, user_query, bot_answer,
                    citations, retrieved_chunks_count, latency_ms, feedback,
                    tokens, cost_usd, target_doc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """, (
                user_id, username, query, answer,
                citations_json, chunks_count, latency_ms,
                tokens, cost_usd, target_doc or ""
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to log chat interaction: {e}")
        return 0


def fetch_user_chat_history(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve chat history strictly filtered for the specified user ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_query, bot_answer, citations, retrieved_chunks_count,
                       latency_ms, feedback, tokens, cost_usd, target_doc, timestamp
                FROM chat_logs
                WHERE user_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()

            history = []
            for row in rows:
                citations_list = []
                try:
                    if row["citations"]:
                        citations_list = json.loads(row["citations"])
                except Exception:
                    citations_list = []

                history.append({
                    "id": row["id"],
                    "user_query": row["user_query"],
                    "bot_answer": row["bot_answer"],
                    "citations": citations_list,
                    "retrieved_chunks_count": row["retrieved_chunks_count"] or 0,
                    "latency_ms": round(row["latency_ms"] or 0.0, 2),
                    "feedback": row["feedback"] or 0,
                    "tokens": row["tokens"] or 0,
                    "cost_usd": round(row["cost_usd"] or 0.0, 5),
                    "target_doc": row["target_doc"] or "",
                    "timestamp": str(row["timestamp"])
                })
            return history
    except Exception as e:
        logger.error(f"Failed to fetch user chat history: {e}")
        return []


def delete_user_chat_history(user_id: int) -> bool:
    """Clear chat history strictly for the specified user ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_logs WHERE user_id = ?", (user_id,))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to clear user chat history: {e}")
        return False
