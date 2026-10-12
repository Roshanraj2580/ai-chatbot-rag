"""
FastAPI application with document ingestion and query endpoints.
"""
import os
import logging
import tempfile
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import time
import urllib.parse
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import FileResponse

from backend.ingest import get_ingestor
from backend.query import get_query_service
from backend.chroma_client import get_chroma_client
from backend.voice_realtime import get_realtime_conversation
from backend.cache import get_cache
from backend.auth import (
    init_auth_db,
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_current_user_optional,
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserProfile,
)
from backend.chat_history import (
    init_chat_history_db,
    save_chat_interaction,
    fetch_user_chat_history,
    delete_user_chat_history,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define temporary upload directory
# Use absolute path to ensure it resolves correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Ensure temp directory exists
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Clean up any existing files in temp_uploads on startup
# This prevents accumulation of files if the server crashed previously
try:
    logger.info(f"Cleaning up temporary directory: {TEMP_UPLOAD_DIR}")
    for filename in os.listdir(TEMP_UPLOAD_DIR):
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted stale temp file: {filename}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
except Exception as e:
    logger.warning(f"Error during temp directory cleanup: {e}")

# Global service instances (initialized on startup)
ingestor = None
query_service = None
chroma_client = None


# ============================================================================
# Pydantic Models
# ============================================================================

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    doc_id: str = Field(..., description="Unique document ID")
    status: str = Field(..., description="Ingestion status: 'ingested', 'partial', or 'failed'")
    chunks: int = Field(..., description="Number of chunks created")
    failed_pages: List[int] = Field(default_factory=list, description="List of failed page numbers")


class DocumentInfo(BaseModel):
    """Document metadata model."""
    doc_id: str = Field(..., description="Unique document ID")
    source_filename: str = Field(..., description="Original filename")
    pages: int = Field(..., description="Number of pages")
    chunks: int = Field(..., description="Number of chunks")
    ingested_at: str = Field(..., description="Ingestion timestamp")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, description="User question")
    filter_doc: Optional[str] = Field(None, description="Optional document filename to scope query")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tenant isolation")


class Citation(BaseModel):
    """Citation model for query response."""
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None


class RetrievedChunk(BaseModel):
    """Retrieved chunk model for query response."""
    id: str
    document: str
    metadata: dict
    distance: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved context chunks")
    tokens: Optional[int] = 0
    cost_usd: Optional[float] = 0.0
    from_cache: Optional[bool] = False
    target_doc: Optional[str] = ""


class ChatHistoryItem(BaseModel):
    """Model for persisted chat conversation turn."""
    id: int
    user_query: str
    bot_answer: str
    citations: List[dict] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    latency_ms: float = 0.0
    feedback: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    target_doc: str = ""
    timestamp: str = ""


class TTSRequest(BaseModel):
    """Model for text-to-speech request."""
    text: str = Field(..., min_length=1, max_length=5000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown").
    
    Startup (before yield):
        - Validates GOOGLE_API_KEY is set
        - Initializes ChromaClient singleton
        - Initializes PDFIngestor singleton  
        - Initializes QueryService singleton
    
    Shutdown (after yield):
        - Logs shutdown message
        - Optional: cleanup resources
    """
    global ingestor, query_service, chroma_client

    logger.info("Starting up application...")

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not set")
        # raise ValueError("GOOGLE_API_KEY required") # Optional: enforce strict check
    try:
        chroma_client = get_chroma_client()
        query_service = get_query_service()

        # Lazy initialize ingestor on first PDF upload
        ingestor = None
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield  # Application runs here

    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="AI RAG Chatbot API",
    description="PDF ingestion and query API with local embeddings and Gemini LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend (Vite port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Auth and Chat History database tables
init_auth_db()
init_chat_history_db()


# ============================================================================
# Authentication Endpoints (JWT)
# ============================================================================

@app.post("/auth/register", response_model=TokenResponse, status_code=201)
async def api_register(req: UserRegisterRequest):
    """Register a new user and return a JWT access token."""
    try:
        user = register_user(req.username, req.password)
        token = create_access_token(user["id"], user["username"], user["role"])
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user_id=user["id"],
            username=user["username"],
            role=user["role"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/auth/login", response_model=TokenResponse)
async def api_login(req: UserLoginRequest):
    """Authenticate credentials and return a signed JWT token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user["id"], user["username"], user["role"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["id"],
        username=user["username"],
        role=user["role"]
    )


@app.get("/auth/me", response_model=UserProfile)
async def api_get_me(current_user: UserProfile = Depends(get_current_user)):
    """Get profile of current authenticated user from JWT token."""
    return current_user


# ============================================================================
# Chat History Endpoints (User-Scoped Persistence)
# ============================================================================

@app.get("/chat/history", response_model=List[ChatHistoryItem])
async def get_chat_history(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Retrieve persistent chat history strictly filtered for the authenticated user.
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    if not effective_user_id:
        return []
    uid = int(effective_user_id) if effective_user_id.isdigit() else 0
    return fetch_user_chat_history(uid)


@app.delete("/chat/history")
async def clear_chat_history(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Clear persistent chat history strictly for the authenticated user.
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    if effective_user_id:
        uid = int(effective_user_id) if effective_user_id.isdigit() else 0
        delete_user_chat_history(uid)
    return {"status": "success", "message": "Chat history cleared"}



@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AI RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /documents": "Upload and ingest PDF",
            "GET /documents": "List all documents",
            "GET /documents/{doc_id}": "Get document metadata",
            "DELETE /documents/{doc_id}": "Delete document and its chunks",
            "GET /ask": "Query with GET (query parameter)",
            "POST /ask": "Query with POST (JSON body)",
            "GET /config": "Get current configuration"
        }
    }


@app.get("/config")
async def get_config():
    """
    Get current configuration for frontend display.
    
    Returns retrieval mode, k value, and embedding provider.
    """
    return {
        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "auto"),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.3"))
    }


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Upload and ingest a PDF document with tenant isolation (JWT or header).
    
    Args:
        file: PDF file (multipart/form-data)
        x_user_id: Optional user identifier from request header
        current_user: Optional user from JWT Bearer token
    
    Returns:
        IngestResponse with doc_id, status, chunks, and failed_pages
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    global ingestor

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info(f"Received upload: {file.filename} (user: {x_user_id})")

    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_UPLOAD_DIR) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Process PDF
        if ingestor is None:
            ingestor = get_ingestor()

        logger.info("Calling process_pdf...")
        result = ingestor.process_pdf(tmp_path, file.filename, user_id=effective_user_id)
        logger.info("process_pdf finished")

        # Clean up temp file
        os.unlink(tmp_path)

        # Invalidate query cache for fresh document retrieval (scoped to user)
        get_cache().clear(user_id=effective_user_id)
        logger.info(f"Invalidated query cache after document upload (user: {effective_user_id})")

        return IngestResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/documents/batch", response_model=List[IngestResponse], status_code=201)
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """
    Upload and ingest multiple PDF documents at once.
    
    Args:
        files: List of PDF files (multipart/form-data)
    
    Returns:
        List of IngestResponse objects
    """
    global ingestor

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    results = []

    for file in files:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            results.append(IngestResponse(
                doc_id="",
                status="failed",
                chunks=0,
                failed_pages=[],
            ))
            logger.warning(f"Skipped non-PDF file: {file.filename}")
            continue

        logger.info(f"Processing batch upload: {file.filename}")

        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_UPLOAD_DIR) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            # Process PDF
            if ingestor is None:
                ingestor = get_ingestor()

            result = ingestor.process_pdf(tmp_path, file.filename)

            # Clean up temp file
            os.unlink(tmp_path)

            results.append(IngestResponse(**result))

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(IngestResponse(
                doc_id="",
                status="failed",
                chunks=0,
                failed_pages=[],
            ))

    return results


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    List all ingested documents with metadata, optionally filtered by user.
    
    Returns:
        List of DocumentInfo objects
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    try:
        chroma = chroma_client or get_chroma_client()
        results = chroma.get_all_documents(user_id=effective_user_id)

        # Aggregate by doc_id
        doc_map = {}
        for i, metadata in enumerate(results["metadatas"]):
            doc_id = metadata.get("doc_id")
            if not doc_id:
                continue

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "source_filename": metadata.get("source_filename", "unknown"),
                    "pages": set(),
                    "chunks": 0,
                    "ingested_at": metadata.get("ingested_at", "")
                }

            doc_map[doc_id]["pages"].add(metadata.get("page_number"))
            doc_map[doc_id]["chunks"] += 1

        # Convert to list
        documents = []
        for doc_id, data in doc_map.items():
            documents.append(DocumentInfo(
                doc_id=doc_id,
                source_filename=data["source_filename"],
                pages=len(data["pages"]),
                chunks=data["chunks"],
                ingested_at=data["ingested_at"]
            ))

        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Get metadata for a specific document, ensuring user ownership.
    
    Args:
        doc_id: Document ID
        x_user_id: Optional user identifier
    
    Returns:
        DocumentInfo object
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    try:
        chroma = chroma_client or get_chroma_client()
        results = chroma.get_documents_by_doc_id(doc_id, user_id=effective_user_id)

        if not results["metadatas"]:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Aggregate metadata
        pages = set()
        source_filename = "unknown"
        ingested_at = ""

        for metadata in results["metadatas"]:
            pages.add(metadata.get("page_number"))
            source_filename = metadata.get("source_filename", source_filename)
            ingested_at = metadata.get("ingested_at", ingested_at)

        return DocumentInfo(
            doc_id=doc_id,
            source_filename=source_filename,
            pages=len(pages),
            chunks=len(results["metadatas"]),
            ingested_at=ingested_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Delete a document and all its chunks from the vector database, scoped by user.
    
    Args:
        doc_id: Document ID to delete
        x_user_id: Optional user identifier
    
    Returns:
        JSON with status and count of deleted chunks
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    try:
        chroma = chroma_client or get_chroma_client()
        # Check if document exists for this user first
        results = chroma.get_documents_by_doc_id(doc_id, user_id=effective_user_id)

        if not results["metadatas"]:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Delete the document
        count = chroma.delete_document(doc_id, user_id=effective_user_id)
        
        # Invalidate query cache for this user
        get_cache().clear(user_id=effective_user_id)
        logger.info(f"Deleted document {doc_id}, removed {count} chunks, invalidated cache (user: {effective_user_id})")
        return {
            "status": "success",
            "deleted_chunks": count,
            "doc_id": doc_id,
            "message": f"Successfully deleted document and {count} chunk(s)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/ask", response_model=QueryResponse)
async def ask_query_get(
    query: str = Query(..., description="User question", min_length=1),
    filter_doc: Optional[str] = Query(None, description="Filter search to specific document"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Query endpoint (GET method).
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    username = current_user.username if current_user else "user"
    return await _process_query(query, filter_doc=filter_doc, user_id=effective_user_id, username=username)


@app.post("/ask", response_model=QueryResponse)
async def ask_query_post(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Query endpoint (POST method).
    """
    effective_user_id = str(current_user.id) if current_user else (request.user_id or x_user_id)
    username = current_user.username if current_user else "user"
    return await _process_query(request.query, filter_doc=request.filter_doc, user_id=effective_user_id, username=username)


async def _process_query(
    query: str,
    filter_doc: Optional[str] = None,
    user_id: Optional[str] = None,
    username: str = "user"
) -> QueryResponse:
    """
    Internal query processing logic with user isolation and persistent history logging.
    """
    start_time = time.time()
    try:
        chroma = chroma_client or get_chroma_client()
        # Check if database has any documents for this user
        if chroma.count(user_id=user_id) == 0:
            ans = "I don't have any documents to answer from. Please upload some documents first."
            latency = (time.time() - start_time) * 1000
            if user_id:
                uid = int(user_id) if user_id.isdigit() else 0
                save_chat_interaction(
                    user_id=uid,
                    username=username,
                    query=query,
                    answer=ans,
                    citations=[],
                    chunks_count=0,
                    latency_ms=latency,
                    tokens=0,
                    cost_usd=0.0,
                    target_doc=filter_doc or ""
                )
            return QueryResponse(
                answer=ans,
                citations=[],
                retrieved_chunks=[],
                tokens=0,
                cost_usd=0.0,
                from_cache=False,
                target_doc=filter_doc or ""
            )

        # Process query with document filter and tenant isolation
        qs = query_service or get_query_service()
        result = qs.answer_query(query, k=5, filter_doc=filter_doc, user_id=user_id)
        latency = (time.time() - start_time) * 1000

        if user_id:
            uid = int(user_id) if user_id.isdigit() else 0
            save_chat_interaction(
                user_id=uid,
                username=username,
                query=query,
                answer=result["answer"],
                citations=result.get("citations", []),
                chunks_count=len(result.get("retrieved_chunks", [])),
                latency_ms=latency,
                tokens=result.get("tokens", 0),
                cost_usd=result.get("cost_usd", 0.0),
                target_doc=result.get("target_doc", "")
            )

        return QueryResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]],
            retrieved_chunks=[RetrievedChunk(**c) for c in result["retrieved_chunks"]],
            tokens=result.get("tokens", 0),
            cost_usd=result.get("cost_usd", 0.0),
            from_cache=result.get("from_cache", False),
            target_doc=result.get("target_doc", "")
        )

    except RuntimeError as e:
        # Handle rate limiting and API errors
        if "rate limit" in str(e).lower():
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats():
    """Get query cache statistics."""
    return get_cache().get_stats()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    chroma = chroma_client or get_chroma_client()
    return {
        "status": "ok",
        "chroma_count": chroma.count() if chroma else 0,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Real-time Voice Conversation Endpoint
# ============================================================================

def cleanup_temp_file(path: str) -> None:
    """
    Background task to delete temp file after response is sent.
    
    Args:
        path: Path to temporary file to delete
    """
    try:
        os.unlink(path)
        logger.debug(f"Cleaned up temp file: {path}")
    except OSError as e:
        logger.warning(f"Failed to cleanup temp file {path}: {e}")


@app.post("/voice/conversation")
async def voice_conversation(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    Real-time voice conversation: user speaks → RAG query → LLM responds with voice.
    
    This endpoint processes one turn of conversation:
    1. Transcribe user's speech from audio
    2. Query the RAG system with transcribed text (tenant isolated)
    3. Generate voice response from LLM answer
    4. Save conversation turn into user's chat history
    """
    effective_user_id = str(current_user.id) if current_user else x_user_id
    username = current_user.username if current_user else "user"
    captured_user_text = ""
    captured_answer = ""
    captured_result = {}

    try:
        # Read audio content
        audio_content = await file.read()
        logger.info(f"Received voice conversation request, audio size: {len(audio_content)} bytes (user: {effective_user_id})")

        # Get real-time conversation handler
        conversation = get_realtime_conversation()

        # Define RAG callback
        async def rag_callback(user_text: str) -> str:
            nonlocal captured_user_text, captured_answer, captured_result
            captured_user_text = user_text
            chroma = chroma_client or get_chroma_client()
            if chroma.count(user_id=effective_user_id) == 0:
                captured_answer = "I don't have any documents to answer from. Please upload some documents first."
                return captured_answer

            qs = query_service or get_query_service()
            captured_result = qs.answer_query(user_text, k=5, user_id=effective_user_id)
            captured_answer = captured_result.get("answer", "")
            return captured_answer

        # Process conversation turn
        response_audio_bytes = await conversation.process_conversation_turn(
            audio_content,
            rag_callback
        )

        if not response_audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Could not process voice conversation. Please speak clearly and try again."
            )

        # Save to user's persistent chat history
        if captured_user_text and captured_answer and effective_user_id:
            uid = int(effective_user_id) if effective_user_id.isdigit() else 0
            save_chat_interaction(
                user_id=uid,
                username=username,
                query=captured_user_text,
                answer=captured_answer,
                citations=captured_result.get("citations", []),
                chunks_count=len(captured_result.get("retrieved_chunks", [])),
                tokens=captured_result.get("tokens", 0),
                cost_usd=captured_result.get("cost_usd", 0.0)
            )

        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb', dir=TEMP_UPLOAD_DIR) as tmp_file:
            tmp_file.write(response_audio_bytes)
            tmp_path = tmp_file.name

        # Schedule cleanup BEFORE returning response
        background_tasks.add_task(cleanup_temp_file, tmp_path)

        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename="response.mp3",
            background=background_tasks,
            headers={
                "X-Conversation-Turn": "complete",
                "X-User-Text": urllib.parse.quote(captured_user_text or ""),
                "X-LLM-Text": urllib.parse.quote(captured_answer or ""),
                "Access-Control-Expose-Headers": "X-Conversation-Turn, X-User-Text, X-LLM-Text"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice conversation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice conversation failed: {str(e)}"
        )


@app.post("/voice/tts")
async def text_to_speech(
    req: TTSRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Generate speech audio (MP3) from text for voice-over read aloud.
    """
    try:
        conversation = get_realtime_conversation()
        audio_bytes = await conversation.synthesize_stream(req.text[:2000])
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Text-to-speech synthesis failed")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb', dir=TEMP_UPLOAD_DIR) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        background_tasks.add_task(cleanup_temp_file, tmp_path)

        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=background_tasks,
            headers={"Access-Control-Expose-Headers": "Content-Disposition"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)