"""
Flask Web UI for AI RAG Chatbot
Provides web interface for document upload, chat, and voice chat features.
"""

import os
import io
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
import requests

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "https://ai-rag-chatbot-backend-pnnq.onrender.com")
UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {"pdf"}

# HTTPS Configuration
HTTPS_ENABLED = os.getenv("HTTPS_ENABLED", "false").lower() == "true"
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", None)
SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", None)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

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
    try:
        if method == "GET":
            response = requests.get(url, timeout=30, **kwargs)
        elif method == "POST":
            response = requests.post(url, timeout=30, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30, **kwargs)
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


# ============================================================================
# View Routes (Render HTML Templates)
# ============================================================================

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/chat")
def chat():
    """Chat interface page."""
    return render_template("chat.html")


@app.route("/voice")
def voice():
    """Voice chat interface page."""
    return render_template("voice.html")


@app.route("/upload")
def upload():
    """Document upload page."""
    return render_template("upload.html")


@app.route("/documents")
def documents():
    """Document library page."""
    # Fetch documents from backend
    docs = call_backend("/documents")
    if isinstance(docs, dict) and "error" in docs:
        docs = []
    return render_template("documents.html", documents=docs)


# ============================================================================
# API Routes (Proxy to FastAPI Backend)
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
def ask():
    """Ask a question (RAG query)."""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    result = call_backend("/ask", method="POST", json={"query": query})
    return jsonify(result)


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
