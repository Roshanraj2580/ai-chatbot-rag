"""
Authentication and JWT module for FastAPI.
Provides token creation, verification, and user authentication dependencies.
"""
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from werkzeug.security import generate_password_hash, check_password_hash
from pydantic import BaseModel, Field

# Secret key & token parameters
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "rag-jwt-super-secret-key-2026-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Database path (SQLite application database)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rag_app.db")

security = HTTPBearer(auto_error=False)


def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Ensure database and users table exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Seed default admin & demo users if table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", generate_password_hash("admin123"), "admin")
            )
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("demo", generate_password_hash("demo123"), "user")
            )
            conn.commit()


# Pydantic Auth Models
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str


class UserProfile(BaseModel):
    id: int
    username: str
    role: str


def create_access_token(user_id: int, username: str, role: str = "user") -> str:
    """Generate a signed JWT token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return decoded
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def register_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """Register a new user."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password cannot be empty")
    
    password_hash = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )
            conn.commit()
            user_id = cursor.lastrowid
            return {"id": user_id, "username": username, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError("Username already exists")


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify username and password against SQLite."""
    username = username.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row and check_password_hash(row["password_hash"], password):
            return {"id": row["id"], "username": row["username"], "role": row["role"]}
    return None


async def get_current_user_optional(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserProfile]:
    """Optional JWT auth: returns UserProfile if valid Bearer token provided, else None."""
    if not auth or not auth.credentials:
        return None
    payload = verify_token(auth.credentials)
    if not payload:
        return None
    return UserProfile(
        id=int(payload["sub"]),
        username=payload["username"],
        role=payload.get("role", "user")
    )


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserProfile:
    """Strict JWT auth: raises 401 if missing or invalid."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = verify_token(auth.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserProfile(
        id=int(payload["sub"]),
        username=payload["username"],
        role=payload.get("role", "user")
    )
