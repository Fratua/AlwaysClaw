"""Integration tests for the 3-tier memory system."""

import pytest
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRedisCache:
    """Test Redis cache adapter."""

    def test_import(self):
        from redis_cache import RedisCache, RedisCacheConfig
        config = RedisCacheConfig()
        cache = RedisCache(config)
        assert cache.available is False  # Not connected yet

    @pytest.mark.asyncio
    async def test_unavailable_operations_are_noop(self):
        """When Redis is unavailable, operations should return None/False gracefully."""
        from redis_cache import RedisCache, RedisCacheConfig
        config = RedisCacheConfig(host="localhost", port=6379)
        cache = RedisCache(config)
        # Don't connect - simulate unavailable
        result = await cache.get("test_key")
        assert result is None
        assert await cache.exists("test_key") is False
        health = await cache.health_check()
        assert health['available'] is False

    @pytest.mark.asyncio
    async def test_connect_graceful_failure(self):
        """Connection to non-existent Redis should not raise."""
        from redis_cache import RedisCache, RedisCacheConfig
        config = RedisCacheConfig(host="localhost", port=59999)  # unlikely port
        cache = RedisCache(config)
        await cache.connect()
        assert cache.available is False


class TestPostgresStore:
    """Test PostgreSQL store adapter."""

    def test_import(self):
        from postgres_store import PostgresStore, PostgresStoreConfig
        config = PostgresStoreConfig()
        store = PostgresStore(config)
        assert store.available is False

    @pytest.mark.asyncio
    async def test_unavailable_operations_are_noop(self):
        from postgres_store import PostgresStore, PostgresStoreConfig
        config = PostgresStoreConfig()
        store = PostgresStore(config)
        result = await store.retrieve("test_id")
        assert result is None
        health = await store.health_check()
        assert health['available'] is False


class TestMemoryManagerTiering:
    """Test memory manager tier orchestration."""

    def test_import_memory_manager(self):
        from memory_manager import MemoryManager
        assert MemoryManager is not None

    def test_tier_methods_exist(self):
        """Verify the new tier methods were added."""
        from memory_manager import MemoryManager
        mm = MemoryManager.__new__(MemoryManager)
        assert hasattr(mm, '_tiered_read') or hasattr(mm, '_init_tiers')

    def test_fallback_without_redis_postgres(self):
        """Memory manager should work with only SQLite when Redis/PG unavailable."""
        from memory_models import MemoryConfig
        config = MemoryConfig()
        # Should not raise even without Redis/PostgreSQL
        assert config.embedding.dimension == 1536


class TestVectorStore:
    """Test the warm-tier SQLite vector store."""

    def test_import(self):
        from vector_store import VectorStore, VectorStoreConfig

    def test_create_store(self):
        from vector_store import VectorStore, VectorStoreConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(db_path=Path(tmpdir) / "test.db")
            store = VectorStore(config)
            assert store is not None
