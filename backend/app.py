"""
FastAPI application with document ingestion and query endpoints.
"""
import os
import logging
import tempfile
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import FileResponse

from backend.ingest import get_ingestor
from backend.query import get_query_service
from backend.chroma_client import get_chroma_client
from backend.voice_realtime import get_realtime_conversation

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
    title=" AI RAG Chatbot API",
    description="PDF ingestion and query API with local embeddings and Gemini LLM",
    version="1.0.0",
    lifespan=lifespan
)


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
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document.
    
    Args:
        file: PDF file (multipart/form-data)
    
    Returns:
        IngestResponse with doc_id, status, chunks, and failed_pages
    """
    global ingestor

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info(f"Received upload: {file.filename}")

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
        result = ingestor.process_pdf(tmp_path, file.filename)
        logger.info("process_pdf finished")

        # Clean up temp file
        os.unlink(tmp_path)

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
async def list_documents():
    """
    List all ingested documents with metadata.
    
    Returns:
        List of DocumentInfo objects
    """
    try:
        results = chroma_client.get_all_documents()

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
async def get_document(doc_id: str):
    """
    Get metadata for a specific document.
    
    Args:
        doc_id: Document ID
    
    Returns:
        DocumentInfo object
    """
    try:
        results = chroma_client.get_documents_by_doc_id(doc_id)

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
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the vector database.
    
    Args:
        doc_id: Document ID to delete
    
    Returns:
        JSON with status and count of deleted chunks
    """
    try:
        # Check if document exists first
        results = chroma_client.get_documents_by_doc_id(doc_id)

        if not results["metadatas"]:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Delete the document
        count = chroma_client.delete_document(doc_id)

        logger.info(f"Deleted document {doc_id}, removed {count} chunks")
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
async def ask_query_get(query: str = Query(..., description="User question", min_length=1)):
    """
    Query endpoint (GET method).
    
    Args:
        query: User question as query parameter
    
    Returns:
        QueryResponse with answer, citations, and retrieved chunks
    """
    return await _process_query(query)


@app.post("/ask", response_model=QueryResponse)
async def ask_query_post(request: QueryRequest):
    """
    Query endpoint (POST method).
    
    Args:
        request: QueryRequest with query field
    
    Returns:
        QueryResponse with answer, citations, and retrieved chunks
    """
    return await _process_query(request.query)


async def _process_query(query: str) -> QueryResponse:
    """
    Internal query processing logic.
    
    Args:
        query: User question
    
    Returns:
        QueryResponse
    """
    try:
        # Check if database has any documents
        if chroma_client.count() == 0:
            return QueryResponse(
                answer="I don't know.",
                citations=[],
                retrieved_chunks=[]
            )

        # Process query
        result = query_service.answer_query(query, k=5)

        return QueryResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]],
            retrieved_chunks=[RetrievedChunk(**c) for c in result["retrieved_chunks"]]
        )

    except RuntimeError as e:
        # Handle rate limiting and API errors
        if "rate limit" in str(e).lower():
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "chroma_count": chroma_client.count() if chroma_client else 0,
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
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Real-time voice conversation: user speaks → RAG query → LLM responds with voice.
    
    This endpoint processes one turn of conversation:
    1. Transcribe user's speech from audio
    2. Query the RAG system with transcribed text
    3. Generate voice response from LLM answer
    
    Args:
        file: Audio file with user's speech (WAV format recommended)
    
    Returns:
        Audio file (MP3) with LLM's spoken response
        Headers include:
        - X-User-Text: Transcribed user speech
        - X-LLM-Text: LLM's text response
    """
    try:
        # Read audio content
        audio_content = await file.read()

        logger.info(f"Received voice conversation request, audio size: {len(audio_content)} bytes")

        # Get real-time conversation handler
        conversation = get_realtime_conversation()

        # Define RAG callback
        async def rag_callback(user_text: str) -> str:
            """Query RAG system with user's question."""
            if chroma_client.count() == 0:
                return "I don't have any documents to answer from. Please upload some documents first."

            result = query_service.answer_query(user_text, k=5)
            return result["answer"]

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
            background=background_tasks,  # Ensures cleanup runs after response
            headers={
                "X-Conversation-Turn": "complete",
                "Access-Control-Expose-Headers": "X-Conversation-Turn"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)