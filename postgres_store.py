"""
PostgreSQL Cold-Storage Adapter for 3-Tier Memory System
Provides durable archival storage as the cold tier.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

try:
    import asyncpg
    _PG_ERRORS: tuple = (asyncpg.PostgresError, asyncpg.InterfaceError)
    _CONNECT_ERRORS: tuple = (ImportError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError)
except ImportError:
    asyncpg = None          # type: ignore[assignment]
    _PG_ERRORS = (Exception,)
    _CONNECT_ERRORS = (ImportError, OSError, Exception)

logger = logging.getLogger(__name__)


@dataclass
class PostgresStoreConfig:
    """Configuration for PostgreSQL cold store."""
    host: str = "localhost"
    port: int = 5432
    database: str = "alwaysclaw"
    user: str = "postgres"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    table_name: str = "memory_archive"


class PostgresStore:
    """
    PostgreSQL-based cold storage for the memory system.

    Provides durable archival of low-importance or aged memories.
    All operations gracefully degrade if PostgreSQL is unavailable.
    """

    def __init__(self, config: PostgresStoreConfig):
        self.config = config
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$', self.config.table_name):
            raise ValueError(f"Invalid table name: {self.config.table_name!r}")
        self._pool = None
        self._available = False
        self._connect_error: Optional[str] = None

    async def connect(self) -> None:
        """Connect to PostgreSQL and create the archive table."""
        try:
            if asyncpg is None:
                raise ImportError("asyncpg is not installed")

            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
            )

            table = self.config.table_name
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id          TEXT PRIMARY KEY,
                        content     TEXT,
                        metadata    JSONB,
                        embedding   BYTEA,
                        created_at  TIMESTAMPTZ DEFAULT NOW(),
                        accessed_at TIMESTAMPTZ DEFAULT NOW(),
                        importance  REAL,
                        tier        TEXT DEFAULT 'cold'
                    )
                """)

            self._available = True
            logger.info(
                "PostgreSQL cold store connected at %s:%d/%s",
                self.config.host, self.config.port, self.config.database,
            )
        except _CONNECT_ERRORS as exc:
            self._available = False
            self._connect_error = str(exc)
            logger.warning(
                "PostgreSQL unavailable - memory system running on SQLite only. "
                "Error: %s", exc
            )

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # --- CRUD ---

    async def store(
        self,
        entry_id: str,
        content: str,
        metadata: dict,
        embedding: Optional[bytes] = None,
        importance: float = 0.5,
    ) -> None:
        """Insert or update a memory entry in the archive."""
        if not self._available:
            return
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                        (id, content, metadata, embedding, importance, created_at, accessed_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        metadata    = EXCLUDED.metadata,
                        embedding   = EXCLUDED.embedding,
                        importance  = EXCLUDED.importance,
                        accessed_at = NOW()
                    """,
                    entry_id,
                    content,
                    json.dumps(metadata),
                    embedding,
                    importance,
                )
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL store error: %s", exc)

    async def retrieve(self, entry_id: str) -> Optional[dict]:
        """Retrieve a memory entry by ID, updating its accessed_at timestamp."""
        if not self._available:
            return None
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT * FROM {table} WHERE id = $1", entry_id
                )
                if row is None:
                    return None
                # Touch accessed_at
                await conn.execute(
                    f"UPDATE {table} SET accessed_at = NOW() WHERE id = $1",
                    entry_id,
                )
                return {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "embedding": bytes(row["embedding"]) if row["embedding"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "accessed_at": row["accessed_at"].isoformat() if row["accessed_at"] else None,
                    "importance": row["importance"],
                    "tier": row["tier"],
                }
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL retrieve error: %s", exc)
            return None

    async def search(
        self, query_embedding: bytes = None, limit: int = 10
    ) -> List[dict]:
        """
        Basic search returning entries ordered by importance DESC.

        A proper vector-similarity search would require pgvector;
        this fallback sorts by importance and returns the top results.
        # TODO: Implement pgvector similarity search
        """
        if not self._available:
            return []
        if query_embedding is not None:
            logger.warning("query_embedding provided but ignored - pgvector not implemented")
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT * FROM {table} ORDER BY importance DESC LIMIT $1",
                    limit,
                )
                return [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                        "importance": r["importance"],
                        "tier": r["tier"],
                    }
                    for r in rows
                ]
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL search error: %s", exc)
            return []

    async def archive_batch(self, entries: List[dict]) -> None:
        """Bulk-insert a batch of entries into the archive."""
        if not self._available or not entries:
            return
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    f"""
                    INSERT INTO {table}
                        (id, content, metadata, embedding, importance, created_at, accessed_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        metadata    = EXCLUDED.metadata,
                        embedding   = EXCLUDED.embedding,
                        importance  = EXCLUDED.importance,
                        accessed_at = NOW()
                    """,
                    [
                        (
                            e["id"],
                            e.get("content", ""),
                            json.dumps(e.get("metadata", {})),
                            e.get("embedding"),
                            e.get("importance", 0.5),
                        )
                        for e in entries
                    ],
                )
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL archive_batch error: %s", exc)

    async def get_archived_since(self, since: datetime) -> List[dict]:
        """Return entries created after *since*."""
        if not self._available:
            return []
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT * FROM {table} WHERE created_at > $1 ORDER BY created_at DESC",
                    since,
                )
                return [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                        "importance": r["importance"],
                        "tier": r["tier"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    }
                    for r in rows
                ]
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL get_archived_since error: %s", exc)
            return []

    async def delete_old(self, before: datetime) -> int:
        """Delete entries not accessed since *before*. Returns count deleted."""
        if not self._available:
            return 0
        try:
            table = self.config.table_name
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE accessed_at < $1",
                    before,
                )
                # asyncpg returns e.g. "DELETE 5"
                return int(result.split()[-1])
        except _PG_ERRORS as exc:
            logger.warning("PostgreSQL delete_old error: %s", exc)
            return 0

    async def health_check(self) -> Dict[str, any]:
        """Execute SELECT 1 to verify connectivity, returning health status dict."""
        if not self._available:
            return {
                'available': False,
                'error': self._connect_error or 'Not connected'
            }
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {'available': True, 'error': None}
        except _PG_ERRORS as exc:
            return {'available': False, 'error': str(exc)}
