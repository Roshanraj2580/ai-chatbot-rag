"""
Flask Web UI for AI RAG Chatbot
Provides web interface for document upload, chat, and voice chat features.
"""

import os
import io
import time
import logging
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import SQLite database module for auth, history, and MLOps
try:
    from ui_flask.database import (
        init_db, register_user, authenticate_user,
        log_chat_interaction, get_user_chat_history,
        clear_user_chat_history, record_feedback, get_mlops_metrics
    )
except ImportError:
    from database import (
        init_db, register_user, authenticate_user,
        log_chat_interaction, get_user_chat_history,
        clear_user_chat_history, record_feedback, get_mlops_metrics
    )

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {"pdf"}

# HTTPS Configuration
HTTPS_ENABLED = os.getenv("HTTPS_ENABLED", "false").lower() == "true"
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", None)
SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", None)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "rag-chatbot-secret-key-2026-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Initialize database tables on startup
init_db()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def call_backend(endpoint, method="GET", **kwargs):
    """Helper to call FastAPI backend."""
    url = f"{BACKEND_URL}{endpoint}"
    timeout = kwargs.pop("timeout", 120)
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout, **kwargs)
        elif method == "POST":
            response = requests.post(url, timeout=timeout, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, timeout=timeout, **kwargs)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        
        # Check content-type properly (handle charset and other parameters)
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            return response.json()
        else:
            return response.content
            
    except requests.exceptions.Timeout:
        return {"error": "Backend request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is it running on port 8000?"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required. Please sign in.", "status": 401}), 401
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    """User login endpoint."""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        user = authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['username']}!", "success")
            next_url = request.args.get("next") or url_for("chat")
            return redirect(next_url)
        else:
            flash("Invalid username or password. (Try admin / admin123)", "danger")
            return redirect(url_for("login"))
            
    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    """User registration endpoint."""
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    
    if len(password) < 4:
        flash("Password must be at least 4 characters long.", "danger")
        return redirect(url_for("login"))
        
    success, res = register_user(username, password)
    if success:
        flash("Registration successful! You can now log in.", "success")
    else:
        flash(f"Registration error: {res}", "danger")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    """Log out current user."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ============================================================================
# View Routes (Render HTML Templates)
# ============================================================================

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/chat")
@login_required
def chat():
    """Chat interface page."""
    return render_template("chat.html")


@app.route("/voice")
@login_required
def voice():
    """Voice chat interface page."""
    return render_template("voice.html")


@app.route("/upload")
@login_required
def upload():
    """Document upload page."""
    return render_template("upload.html")


@app.route("/documents")
@login_required
def documents():
    """Document library page."""
    # Fetch documents from backend
    docs = call_backend("/documents")
    if isinstance(docs, dict) and "error" in docs:
        docs = []
    return render_template("documents.html", documents=docs)


@app.route("/mlops")
@login_required
def mlops():
    """MLOps observability dashboard."""
    metrics = get_mlops_metrics()
    return render_template("mlops.html", metrics=metrics)


# ============================================================================
# API Routes (Proxy to FastAPI Backend & MLOps Tracking)
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Check backend health."""
    result = call_backend("/health")
    return jsonify(result)


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get backend configuration for UI display."""
    result = call_backend("/config")
    return jsonify(result)


@app.route("/api/ask", methods=["POST"])
@login_required
def ask():
    """Ask a question (RAG query) with MLOps latency tracking & chat logging."""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # 1. Measure query latency (MLOps performance tracking)
    start_time = time.time()
    result = call_backend("/ask", method="POST", json={"query": query})
    latency_ms = (time.time() - start_time) * 1000.0
    
    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 500
        
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    chunks_count = len(result.get("retrieved_chunks", []))
    
    # 2. Log interaction into SQLite for Chat History & MLOps Telemetry
    log_id = log_chat_interaction(
        user_id=session.get("user_id"),
        username=session.get("username", "user"),
        query=query,
        answer=answer,
        citations=citations,
        chunks_count=chunks_count,
        latency_ms=latency_ms
    )
    
    # Attach telemetry metadata to response
    result["log_id"] = log_id
    result["latency_ms"] = round(latency_ms, 2)
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
@login_required
def get_history():
    """Get authenticated user's chat history."""
    history = get_user_chat_history(session["user_id"])
    return jsonify(history)


@app.route("/api/history/clear", methods=["POST"])
@login_required
def clear_history():
    """Clear authenticated user's chat history."""
    clear_user_chat_history(session["user_id"])
    return jsonify({"status": "success", "message": "Chat history cleared"})


@app.route("/api/feedback", methods=["POST"])
@login_required
def feedback():
    """Submit user feedback (+1 for positive, -1 for negative) on a response."""
    data = request.json or {}
    log_id = data.get("log_id")
    feedback_val = data.get("feedback")
    
    if not log_id or feedback_val not in (1, -1):
        return jsonify({"error": "Invalid feedback data"}), 400
        
    success = record_feedback(log_id, session["user_id"], feedback_val)
    return jsonify({"status": "success" if success else "failed"})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload a PDF document."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    try:
        # Upload to backend
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            result = call_backend("/documents", method="POST", files=files)
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def get_documents():
    """Get list of documents."""
    result = call_backend("/documents")
    return jsonify(result)


@app.route("/api/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    """Get specific document details."""
    result = call_backend(f"/documents/{doc_id}")
    return jsonify(result)


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document."""
    result = call_backend(f"/documents/{doc_id}", method="DELETE")
    return jsonify(result)


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    """Handle bad requests, including TLS handshake attempts on HTTP server."""
    # Detect TLS handshake attempts (starts with \x16\x03)
    if request.environ.get('werkzeug.request') and hasattr(request, 'data'):
        try:
            if request.data and len(request.data) > 0 and request.data[0] == 0x16:
                logger.warning(
                    f"TLS/SSL handshake attempt detected from {request.remote_addr}. "
                    f"Client is trying HTTPS but server is HTTP-only. "
                    f"Consider enabling HTTPS or ensure clients use http:// URLs."
                )
                return jsonify({
                    "error": "Protocol mismatch",
                    "message": "This server only supports HTTP. Please use http:// instead of https://"
                }), 400
        except:
            pass
    
    return jsonify({"error": "Bad request", "message": str(e)}), 400


# ============================================================================
# Voice API Routes
# ============================================================================

@app.route("/api/voice/conversation", methods=["POST"])
def voice_conversation():
    """Real-time voice conversation: audio input -> transcribe -> RAG -> synthesize -> audio output."""
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["file"]
    
    try:
        # Forward to backend
        files = {"file": (audio_file.filename, audio_file.stream, audio_file.content_type)}
        response = requests.post(f"{BACKEND_URL}/voice/conversation", files=files)
        response.raise_for_status()
        
        # Return audio response
        return send_file(
            io.BytesIO(response.content),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="response.mp3"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    Returns JSON for API routes, HTML for page routes.
    """
    if request.path.startswith('/api/'):
        return jsonify({"error": "Not found", "status": 404}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    Returns JSON for API routes, HTML for page routes.
    """
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return render_template("index.html"), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Development server
    ssl_context = None
    
    if HTTPS_ENABLED:
        if SSL_CERT_PATH and SSL_KEY_PATH:
            # Use provided certificate files
            ssl_context = (SSL_CERT_PATH, SSL_KEY_PATH)
            logger.info(f"Starting Flask with HTTPS (cert: {SSL_CERT_PATH})")
        else:
            # Use adhoc self-signed certificate (requires pyOpenSSL)
            try:
                ssl_context = 'adhoc'
                logger.info("Starting Flask with HTTPS (adhoc self-signed certificate)")
                logger.warning("Using adhoc certificate - browsers will show security warnings")
            except ImportError:
                logger.error("HTTPS_ENABLED=true but pyOpenSSL not installed. Install with: pip install pyopenssl")
                logger.info("Falling back to HTTP...")
                ssl_context = None
    
    protocol = "https" if ssl_context else "http"
    logger.info(f"Starting Flask server on {protocol}://0.0.0.0:5000")
    
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl_context)
