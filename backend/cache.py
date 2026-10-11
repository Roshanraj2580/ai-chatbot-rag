"""
Redis query caching module with graceful in-memory fallback.
Provides sub-millisecond response caching for repeated RAG queries.
"""
import os
import json
import hashlib
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Cache TTL (Time-To-Live in seconds, default: 2 hours)
DEFAULT_CACHE_TTL = int(os.getenv("CACHE_TTL", "7200"))


class QueryCache:
    """
    Query cache supporting Redis with automatic fallback to high-speed in-memory cache.
    Eliminates embedding and LLM latency on repeated questions (drops response time to <5ms).
    """

    def __init__(self):
        self.redis_client = None
        self.backend_type = "in-memory"
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

        redis_url = os.getenv("REDIS_URL", None)

        # Attempt Redis connection if REDIS_URL or default is specified
        if redis_url:
            try:
                import redis
                client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=1.5)
                client.ping()
                self.redis_client = client
                self.backend_type = "redis"
                logger.info(f"Connected to Redis cache at {redis_url}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis ({e}). Using fast in-memory cache.")
                self.redis_client = None
                self.backend_type = "in-memory"
        else:
            logger.info("REDIS_URL not configured. Using fast in-memory query cache.")

    def _normalize_key(self, query: str, k: int = 5, user_id: Optional[str] = None) -> str:
        """Create a deterministic cache key from normalized query text and optional user_id."""
        normalized = query.strip().lower().rstrip("?!., ")
        tenant_prefix = f"user:{user_id}:" if user_id else "global:"
        hash_digest = hashlib.sha256(f"{tenant_prefix}{normalized}:k={k}".encode("utf-8")).hexdigest()[:16]
        return f"rag:query:{tenant_prefix}{hash_digest}"

    def get(self, query: str, k: int = 5, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached query response if available for the specific user."""
        key = self._normalize_key(query, k, user_id=user_id)

        # 1. Try Redis
        if self.redis_client:
            try:
                raw_data = self.redis_client.get(key)
                if raw_data:
                    self.hits += 1
                    data = json.loads(raw_data)
                    data["from_cache"] = True
                    return data
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # 2. Try In-Memory Cache
        if key in self.memory_store:
            entry = self.memory_store[key]
            # Check expiry
            if entry["expires_at"] > time.time():
                self.hits += 1
                data = dict(entry["data"])
                data["from_cache"] = True
                return data
            else:
                del self.memory_store[key]

        self.misses += 1
        return None

    def set(self, query: str, data: Dict[str, Any], k: int = 5, ttl: int = DEFAULT_CACHE_TTL, user_id: Optional[str] = None) -> None:
        """Store query response in cache with TTL scoped by user."""
        key = self._normalize_key(query, k, user_id=user_id)

        # 1. Save to Redis if available
        if self.redis_client:
            try:
                payload = json.dumps(data)
                self.redis_client.setex(key, ttl, payload)
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # 2. Save to In-Memory store
        self.memory_store[key] = {
            "data": data,
            "expires_at": time.time() + ttl
        }

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear cached query responses for a specific user, or all if user_id is None."""
        if user_id:
            tenant_pattern = f"rag:query:user:{user_id}:*"
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(tenant_pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis clear for user {user_id} failed: {e}")
            # Clear in-memory matching tenant
            to_delete = [k for k in self.memory_store if f"user:{user_id}:" in k]
            for k in to_delete:
                del self.memory_store[k]
        else:
            if self.redis_client:
                try:
                    keys = self.redis_client.keys("rag:query:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
            self.memory_store.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics for MLOps monitoring."""
        total = self.hits + self.misses
        hit_rate = round((self.hits / total * 100), 1) if total > 0 else 0.0
        return {
            "backend": self.backend_type,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": hit_rate,
            "cached_entries": len(self.memory_store) if not self.redis_client else "managed_by_redis"
        }


# Singleton cache instance
_cache_instance: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    """Get or create singleton query cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
    return _cache_instance
