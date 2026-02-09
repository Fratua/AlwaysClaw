"""
Redis Hot-Cache Adapter for 3-Tier Memory System
Provides fast in-memory caching as the hot tier.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RedisCacheConfig:
    """Configuration for Redis cache."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "alwaysclaw:"
    default_ttl_seconds: int = 3600
    max_connections: int = 10


class RedisCache:
    """
    Redis-based hot cache for the memory system.

    Provides fast key-value access with automatic TTL expiration.
    All operations gracefully degrade if Redis is unavailable.
    """

    def __init__(self, config: RedisCacheConfig):
        self.config = config
        self._pool = None
        self._available = False
        self._connect_error: Optional[str] = None

    async def connect(self) -> None:
        """Connect to Redis and create connection pool."""
        try:
            import redis.asyncio as aioredis

            self._pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                decode_responses=True,
            )
            # Verify connectivity
            client = aioredis.Redis(connection_pool=self._pool)
            await client.ping()
            await client.aclose()
            self._available = True
            logger.info("Redis cache connected at %s:%d", self.config.host, self.config.port)
        except Exception as exc:
            self._available = False
            self._connect_error = str(exc)
            logger.warning(
                "Redis unavailable - memory system running on SQLite only. "
                "Error: %s", exc
            )

    async def disconnect(self) -> None:
        """Close the Redis connection pool."""
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _key(self, key: str) -> str:
        """Prefix a key with the configured namespace."""
        return f"{self.config.key_prefix}{key}"

    def _client(self):
        """Return a Redis client from the pool."""
        import redis.asyncio as aioredis
        return aioredis.Redis(connection_pool=self._pool)

    # --- single-key operations ---

    async def get(self, key: str) -> Optional[dict]:
        """Get a value from Redis, returning None if missing or unavailable."""
        if not self._available:
            return None
        try:
            client = self._client()
            try:
                raw = await client.get(self._key(key))
                if raw is None:
                    return None
                return json.loads(raw)
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on get(%s)", key)
            return None
        except Exception as exc:
            logger.warning("Redis get error: %s", exc)
            return None

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        """Store a value in Redis with an optional TTL."""
        if not self._available:
            return
        try:
            effective_ttl = ttl if ttl is not None else self.config.default_ttl_seconds
            client = self._client()
            try:
                await client.set(
                    self._key(key),
                    json.dumps(value),
                    ex=effective_ttl,
                )
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on set(%s)", key)
        except Exception as exc:
            logger.warning("Redis set error: %s", exc)

    async def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        if not self._available:
            return
        try:
            client = self._client()
            try:
                await client.delete(self._key(key))
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on delete(%s)", key)
        except Exception as exc:
            logger.warning("Redis delete error: %s", exc)

    async def exists(self, key: str) -> bool:
        """Check whether a key exists in Redis."""
        if not self._available:
            return False
        try:
            client = self._client()
            try:
                return bool(await client.exists(self._key(key)))
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on exists(%s)", key)
            return False
        except Exception as exc:
            logger.warning("Redis exists error: %s", exc)
            return False

    # --- bulk operations ---

    async def get_many(self, keys: List[str]) -> Dict[str, dict]:
        """Get multiple values via a pipeline mget."""
        if not self._available or not keys:
            return {}
        try:
            prefixed = [self._key(k) for k in keys]
            client = self._client()
            try:
                async with client.pipeline(transaction=False) as pipe:
                    pipe.mget(prefixed)
                    results = await pipe.execute()
                raw_values = results[0]
                out: Dict[str, dict] = {}
                for key, raw in zip(keys, raw_values):
                    if raw is not None:
                        out[key] = json.loads(raw)
                return out
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on get_many")
            return {}
        except Exception as exc:
            logger.warning("Redis get_many error: %s", exc)
            return {}

    async def set_many(self, items: Dict[str, dict], ttl: Optional[int] = None) -> None:
        """Set multiple values via a pipeline with TTL."""
        if not self._available or not items:
            return
        try:
            effective_ttl = ttl if ttl is not None else self.config.default_ttl_seconds
            client = self._client()
            try:
                async with client.pipeline(transaction=False) as pipe:
                    for key, value in items.items():
                        pipe.set(self._key(key), json.dumps(value), ex=effective_ttl)
                    await pipe.execute()
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on set_many")
        except Exception as exc:
            logger.warning("Redis set_many error: %s", exc)

    # --- maintenance ---

    async def clear_prefix(self, prefix: str) -> None:
        """Scan and delete all keys matching a prefix pattern."""
        if not self._available:
            return
        try:
            pattern = f"{self.config.key_prefix}{prefix}*"
            client = self._client()
            try:
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        await client.delete(*keys)
                    if cursor == 0:
                        break
            finally:
                await client.aclose()
        except ConnectionError:
            logger.warning("Redis connection error on clear_prefix(%s)", prefix)
        except Exception as exc:
            logger.warning("Redis clear_prefix error: %s", exc)

    async def health_check(self) -> Dict[str, any]:
        """Check Redis connectivity and return health status dict."""
        if not self._available:
            return {
                'available': False,
                'error': self._connect_error or 'Not connected'
            }
        try:
            client = self._client()
            try:
                await client.ping()
                return {'available': True, 'error': None}
            finally:
                await client.aclose()
        except Exception as exc:
            return {'available': False, 'error': str(exc)}
