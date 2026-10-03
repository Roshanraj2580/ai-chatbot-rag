"""
Database module for user authentication, chat history, and MLOps telemetry.
Uses standard Python sqlite3 with zero external database dependencies.
"""
import sqlite3
import os
import json
import time
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_app.db")


def get_db_connection():
    """Create and return a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for users, chat history, and MLOps metrics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Chat History & MLOps Query Telemetry Table
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
                feedback INTEGER DEFAULT 0, -- 1: positive (thumbs up), -1: negative (thumbs down), 0: unrated
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()

        # Seed default admin and demo user if empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            admin_hash = generate_password_hash("admin123")
            demo_hash = generate_password_hash("demo123")
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", admin_hash, "admin")
            )
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("demo", demo_hash, "user")
            )
            conn.commit()


# ============================================================================
# User Authentication Helpers
# ============================================================================

def register_user(username, password, role="user"):
    """Register a new user. Returns (True, user_id) or (False, error_message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty"
    
    password_hash = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )
            conn.commit()
            return True, cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)


def authenticate_user(username, password):
    """Authenticate user. Returns user dict if valid, else None."""
    username = username.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user["password_hash"], password):
            return {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"]
            }
    return None


def get_user_by_id(user_id):
    """Fetch user by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if user:
            return dict(user)
    return None


# ============================================================================
# Chat History & MLOps Query Logging Helpers
# ============================================================================

def log_chat_interaction(user_id, username, query, answer, citations, chunks_count, latency_ms):
    """Log an entire Q&A interaction with MLOps telemetry."""
    citations_json = json.dumps(citations) if citations else "[]"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_logs (
                user_id, username, user_query, bot_answer, 
                citations, retrieved_chunks_count, latency_ms, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """, (user_id, username, query, answer, citations_json, chunks_count, latency_ms))
        conn.commit()
        return cursor.lastrowid


def get_user_chat_history(user_id, limit=50):
    """Retrieve chat history for a specific user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_query, bot_answer, citations, retrieved_chunks_count, latency_ms, feedback, timestamp
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
                citations_list = json.loads(row["citations"])
            except:
                citations_list = []
            history.append({
                "id": row["id"],
                "user_query": row["user_query"],
                "bot_answer": row["bot_answer"],
                "citations": citations_list,
                "retrieved_chunks_count": row["retrieved_chunks_count"],
                "latency_ms": round(row["latency_ms"], 2),
                "feedback": row["feedback"],
                "timestamp": row["timestamp"]
            })
        return history


def clear_user_chat_history(user_id):
    """Delete chat history for a specific user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_logs WHERE user_id = ?", (user_id,))
        conn.commit()
        return True


def record_feedback(log_id, user_id, feedback_value):
    """Record user feedback (1 for positive, -1 for negative)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_logs
            SET feedback = ?
            WHERE id = ? AND user_id = ?
        """, (feedback_value, log_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# MLOps Telemetry & Performance Aggregates
# ============================================================================

def get_mlops_metrics():
    """Compute overall MLOps performance and quality metrics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total queries
        cursor.execute("SELECT COUNT(*) FROM chat_logs")
        total_queries = cursor.fetchone()[0]
        
        # Average Latency
        cursor.execute("SELECT AVG(latency_ms) FROM chat_logs WHERE latency_ms > 0")
        avg_latency_row = cursor.fetchone()[0]
        avg_latency = round(avg_latency_row, 2) if avg_latency_row else 0.0
        
        # Feedback counts
        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback = 1")
        positive_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback = -1")
        negative_feedback = cursor.fetchone()[0]
        
        total_feedback = positive_feedback + negative_feedback
        satisfaction_rate = round((positive_feedback / total_feedback) * 100, 1) if total_feedback > 0 else 100.0
        
        # Average chunks retrieved per query
        cursor.execute("SELECT AVG(retrieved_chunks_count) FROM chat_logs")
        avg_chunks_row = cursor.fetchone()[0]
        avg_chunks = round(avg_chunks_row, 1) if avg_chunks_row else 0.0
        
        # Recent queries (latest 25)
        cursor.execute("""
            SELECT id, username, user_query, bot_answer, retrieved_chunks_count, latency_ms, feedback, timestamp
            FROM chat_logs
            ORDER BY id DESC
            LIMIT 25
        """)
        recent_logs = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_queries": total_queries,
            "avg_latency_ms": avg_latency,
            "avg_chunks": avg_chunks,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "satisfaction_rate": satisfaction_rate,
            "recent_logs": recent_logs
        }
