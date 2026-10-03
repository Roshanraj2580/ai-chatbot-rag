"""
Query pipeline: Query embedding → Chroma retrieval → prompt building → Gemini LLM.

Uses the new google-genai SDK (migrated from deprecated google-generativeai).
"""
import os
import logging
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

# NEW: google-genai SDK imports
from google import genai
from google.genai import types

from backend.embedder import get_embedder
from backend.chroma_client import get_chroma_client
from backend.retrieval import HybridRetriever
from backend.cache import get_cache

logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.85"))


class PromptBuilder:
    """Build prompts with positive and negative prompting for accurate, grounded responses."""
    
    TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

## Context (Retrieved Documents):
{context}

## User Question:
{user_question}

## Instructions (MUST FOLLOW):

### What you SHOULD do (Positive):
- Answer using ONLY information explicitly stated in the context above
- Synthesize information from multiple chunks if they contain related content
- Use natural, conversational language
- Be concise but complete (2-5 sentences is ideal)
- If the context contains partial or related information, provide what you can and note limitations
- Look for semantic matches - if the user asks about "learning" and context discusses "self-learning", that IS relevant

### What you MUST NOT do (Negative):
- DO NOT make up facts, statistics, names, or details not in the context
- DO NOT use your general knowledge - ONLY use the provided context
- DO NOT say "I don't know" if the context contains ANY relevant information
- DO NOT hallucinate or invent information to fill gaps
- DO NOT add disclaimers unless truly necessary

### When to say "I don't know":
- ONLY say "I don't know" if the context contains ZERO relevant information
- If even partial information exists, provide it with appropriate caveats

Now answer the question:
"""
    
    @staticmethod
    def build(query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build prompt from query and retrieved chunks.
        
        Args:
            query: User question
            chunks: List of dicts with 'document' and 'metadata' keys
        
        Returns:
            Complete prompt string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            document = chunk["document"]
            
            source = metadata.get("source_filename", "unknown")
            page = metadata.get("page_number", "N/A")
            
            context_parts.append(
                f"### Document {i} [Source: {source}, Page {page}]\n{document}"
            )
        
        context_str = "\n\n".join(context_parts)
        
        prompt = PromptBuilder.TEMPLATE.format(
            context=context_str,
            user_question=query
        )
        
        return prompt


def deduplicate_chunks(chunks: List[Dict[str, Any]], threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks based on text similarity.
    
    Args:
        chunks: List of retrieved chunks
        threshold: Similarity threshold (0.0-1.0). Chunks above this are considered duplicates.
    
    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return chunks
    
    unique_chunks = []
    
    for chunk in chunks:
        is_duplicate = False
        chunk_text = chunk["document"]
        
        for unique_chunk in unique_chunks:
            unique_text = unique_chunk["document"]
            
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, chunk_text, unique_text).ratio()
            
            if similarity >= threshold:
                is_duplicate = True
                logger.debug(f"Duplicate detected (similarity={similarity:.2f}): skipping chunk")
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    if len(chunks) != len(unique_chunks):
        logger.info(f"Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
    
    return unique_chunks


class QueryService:
    """
    Main query orchestrator with Gemini integration.
    
    Uses the new google-genai SDK pattern with centralized Client object.
    
    Features:
    - Configurable retrieval count (RETRIEVAL_K env var)
    - Configurable temperature (LLM_TEMPERATURE env var)
    - Duplicate chunk detection
    - Positive/negative prompting for grounded responses
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None
    ):
        """
        Initialize query service with Gemini API.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model name for generation (defaults to LLM_MODEL or gemini-3.8-flash)
        
        Raises:
            ValueError: If GOOGLE_API_KEY not set
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        # NEW: Create genai.Client (replaces genai.configure + GenerativeModel)
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-3.8-flash")
        
        self.embedder = get_embedder()
        self.chroma_client = get_chroma_client()
        self.prompt_builder = PromptBuilder()
        self.cache = get_cache()
        
        # Initialize hybrid retriever (handles semantic, keyword, and hybrid modes)
        self.retriever = HybridRetriever(self.embedder, self.chroma_client)
        
        # Configuration - read at init time but can be overridden per-query
        self.default_k = int(os.getenv("RETRIEVAL_K", "5"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
        logger.info(
            f"QueryService initialized: model={model_name}, "
            f"default_k={self.default_k}, temperature={self.temperature}, "
            f"retrieval_mode={self.retriever.mode}, cache={self.cache.backend_type}"
        )
    
    def answer_query(self, query: str, k: Optional[int] = None, filter_doc: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete query pipeline: hybrid retrieve → deduplicate → prompt → LLM.
        
        Args:
            query: User question
            k: Number of chunks to retrieve (default: RETRIEVAL_K env var or 5)
            filter_doc: Optional source_filename to scope search to a specific document
        
        Returns:
            Dict with answer, citations, chunks, token usage, and cost estimates.
        """
        # Use default k if not specified
        if k is None:
            k = self.default_k
        
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > 50:
            raise ValueError("k cannot exceed 50")
        
        query = query.strip()  # Normalize whitespace

        # 1. Fast Cache Check (Redis / In-Memory): drops latency from ~1500ms to <5ms
        cache_query_id = f"{query}:doc={filter_doc or 'all'}"
        cached_result = self.cache.get(cache_query_id, k=k)
        if cached_result:
            logger.info(f"⚡ Cache HIT for query: '{query[:50]}' (served in <5ms)")
            return cached_result

        logger.info(f"Processing query (Cache MISS): '{query[:80]}...', filter_doc={filter_doc}")
        
        # Use HybridRetriever with stats (handles semantic, keyword, or hybrid)
        fetch_k = min(k * 4, 60) if filter_doc else min(k * 2, 50)
        retrieved_chunks, search_stats = self.retriever.search_with_stats(query, k=fetch_k)
        
        # Apply document filter if scoped to a specific document
        if filter_doc:
            target = filter_doc.strip().lower()
            retrieved_chunks = [
                c for c in retrieved_chunks
                if c.get("metadata", {}).get("source_filename", "").lower() == target
                or c.get("metadata", {}).get("doc_id", "") == target
            ]

        if not retrieved_chunks:
            logger.warning("No matching results found in vector DB")
            no_doc_msg = f"No relevant content found in '{filter_doc}'." if filter_doc else "I don't have any documents to answer from. Please upload some documents first."
            return {
                "answer": no_doc_msg,
                "citations": [],
                "retrieved_chunks": [],
                "search_stats": search_stats,
                "tokens": 0,
                "cost_usd": 0.0,
                "target_doc": filter_doc or ""
            }
        
        # Deduplicate chunks
        unique_chunks = deduplicate_chunks(retrieved_chunks)
        
        # Limit to requested k after deduplication
        unique_chunks = unique_chunks[:k]
        
        # Update stats with deduplication info
        search_stats["after_dedup"] = len(unique_chunks)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, using {len(unique_chunks)} after deduplication")
        
        # Build prompt
        prompt = self.prompt_builder.build(query, unique_chunks)
        
        logger.debug(f"Built prompt with {len(unique_chunks)} chunks")
        
        # Call Gemini using NEW google-genai SDK
        try:
            # Use lower temperature for more factual responses
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=1024  # Increased for longer answers
                )
            )
            answer = response.text.strip()
            logger.info(f"Generated answer: {answer[:100]}...")
            
        except Exception as e:
            logger.warning(f"Primary model {self.model_name} failed: {e}. Trying fallback models...")
            fallbacks = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash"]
            answer = None
            for fb_model in fallbacks:
                if fb_model == self.model_name:
                    continue
                try:
                    logger.info(f"Attempting fallback model: {fb_model}")
                    response = self.client.models.generate_content(
                        model=fb_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=1024
                        )
                    )
                    answer = response.text.strip()
                    self.model_name = fb_model
                    logger.info(f"Successfully generated answer with {fb_model}: {answer[:100]}...")
                    break
                except Exception as fb_err:
                    logger.warning(f"Fallback {fb_model} failed: {fb_err}")
            
            if answer is None:
                logger.error(f"All Gemini models failed: {e}")
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    raise RuntimeError("Gemini API rate limit exceeded. Please try again later.") from e
                raise RuntimeError(f"Gemini API error: {e}") from e
        
        # Extract citations from metadata (use unique chunks)
        citations = []
        for chunk in unique_chunks:
            metadata = chunk["metadata"]
            citations.append({
                "source_filename": metadata.get("source_filename"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index")
            })
        
        # Estimate token usage and API cost (Gemini 3.8 Flash pricing: $0.075/1M input, $0.30/1M output)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(answer) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = round((prompt_tokens * 0.000000075) + (completion_tokens * 0.0000003), 6)

        result = {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": unique_chunks,
            "search_stats": search_stats,
            "from_cache": False,
            "tokens": total_tokens,
            "cost_usd": cost_usd,
            "target_doc": filter_doc or ""
        }
        
        # Store in Redis/memory cache
        self.cache.set(cache_query_id, result, k=k)
        
        return result


# Singleton instance with type hint
_query_service_instance: Optional[QueryService] = None


def get_query_service(api_key: Optional[str] = None) -> QueryService:
    """
    Get or create the singleton query service instance.
    
    Args:
        api_key: Optional API key override
    
    Returns:
        QueryService instance
    """
    global _query_service_instance
    
    if _query_service_instance is None:
        _query_service_instance = QueryService(api_key=api_key)
    
    return _query_service_instance