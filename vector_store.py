"""
Vector Store Implementation using SQLite with sqlite-vec extension
Hybrid search with BM25 + Vector similarity
"""

import sqlite3
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime

from memory_models import (
    MemoryEntry, MemoryChunk, SearchResult, 
    EmbeddingConfig, SearchConfig, MemoryType
)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    db_path: Path
    embedding_dimension: int = 1536
    enable_fts: bool = True
    enable_wal: bool = True
    cache_size_mb: int = 256


class VectorStore:
    """
    SQLite-based vector store with hybrid search capabilities.
    
    Features:
    - Vector similarity search using sqlite-vec
    - Full-text search using FTS5
    - Embedding cache for deduplication
    - Hybrid ranking with RRF
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.db: Optional[sqlite3.Connection] = None
        self._vec_available = False
    
    def connect(self) -> 'VectorStore':
        """Connect to database and initialize schema."""
        self.db = sqlite3.connect(str(self.config.db_path))
        self.db.row_factory = sqlite3.Row
        
        # Configure SQLite
        if self.config.enable_wal:
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute(f"PRAGMA cache_size = -{self.config.cache_size_mb * 1024}")
        
        # Check for sqlite-vec
        self._vec_available = self._check_vec_extension()
        
        # Initialize schema
        self._init_schema()
        
        return self
    
    def _check_vec_extension(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            self.db.execute("SELECT vec_version()")
            return True
        except sqlite3.OperationalError:
            return False
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Embedding cache table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Memory entries table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_file TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                importance_score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                embedding_id TEXT
            )
        """)
        
        # Tags table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT REFERENCES memory_entries(id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                PRIMARY KEY (memory_id, tag)
            )
        """)
        
        # Access log
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT REFERENCES memory_entries(id) ON DELETE CASCADE,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                relevance_score REAL
            )
        """)
        
        # Create vector virtual table if sqlite-vec is available
        if self._vec_available:
            try:
                self.db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                        memory_id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.config.embedding_dimension}]
                    )
                """)
            except sqlite3.OperationalError:
                # Table might already exist with different dimensions
                pass
        else:
            # Fallback: create regular table with BLOB storage
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """)
        
        # Create FTS5 virtual table
        if self.config.enable_fts:
            try:
                self.db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        content,
                        memory_id UNINDEXED,
                        tokenize='porter unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass
        
        # Create indexes
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(type)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance_score DESC)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memory_entries(last_accessed)
        """)
        
        self.db.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
            self.db = None
    
    # Embedding cache operations
    
    def get_cached_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding by content hash."""
        cursor = self.db.execute(
            "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            return np.frombuffer(row['embedding'], dtype=np.float32)
        return None
    
    def cache_embedding(
        self,
        content_hash: str,
        embedding: np.ndarray,
        model_name: str
    ) -> None:
        """Cache an embedding."""
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        self.db.execute("""
            INSERT OR REPLACE INTO embedding_cache 
            (content_hash, embedding, model_name, dimension)
            VALUES (?, ?, ?, ?)
        """, (content_hash, embedding_blob, model_name, len(embedding)))
        
        self.db.commit()
    
    # Memory entry operations
    
    def insert_memory(
        self,
        entry: MemoryEntry,
        embedding: np.ndarray
    ) -> str:
        """
        Insert a memory entry with its embedding.
        
        Returns:
            Memory ID
        """
        # Insert memory entry
        self.db.execute("""
            INSERT OR REPLACE INTO memory_entries
            (id, type, content, source_file, line_start, line_end,
             created_at, updated_at, importance_score, access_count,
             last_accessed, chunk_index, total_chunks, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.type.value,
            entry.content,
            str(entry.source_file),
            entry.line_start,
            entry.line_end,
            entry.created_at.isoformat(),
            entry.updated_at.isoformat() if entry.updated_at else entry.created_at.isoformat(),
            entry.importance_score,
            entry.access_count,
            entry.last_accessed.isoformat() if entry.last_accessed else None,
            entry.chunk_index,
            entry.total_chunks,
            entry.embedding_id
        ))
        
        # Insert embedding
        self._insert_embedding(entry.id, embedding)
        
        # Insert FTS entry
        if self.config.enable_fts:
            self.db.execute("""
                INSERT OR REPLACE INTO memory_fts (content, memory_id)
                VALUES (?, ?)
            """, (entry.content, entry.id))
        
        # Insert tags
        for tag in entry.tags:
            self.db.execute("""
                INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                VALUES (?, ?)
            """, (entry.id, tag))
        
        self.db.commit()
        return entry.id
    
    def _insert_embedding(self, memory_id: str, embedding: np.ndarray) -> None:
        """Insert embedding into vector store."""
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        if self._vec_available:
            # Use sqlite-vec
            try:
                self.db.execute("""
                    INSERT OR REPLACE INTO memory_vectors (memory_id, embedding)
                    VALUES (?, vec_f32(?))
                """, (memory_id, json.dumps(embedding.tolist())))
            except sqlite3.OperationalError:
                # Fallback to BLOB
                self._insert_embedding_blob(memory_id, embedding_blob)
        else:
            self._insert_embedding_blob(memory_id, embedding_blob)
    
    def _insert_embedding_blob(self, memory_id: str, embedding_blob: bytes) -> None:
        """Insert embedding as BLOB (fallback)."""
        self.db.execute("""
            INSERT OR REPLACE INTO memory_vectors (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, embedding_blob))
    
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        cursor = self.db.execute(
            "SELECT * FROM memory_entries WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return self._row_to_entry(row)
        return None
    
    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row['id'],
            type=MemoryType(row['type']),
            content=row['content'],
            source_file=Path(row['source_file']),
            line_start=row['line_start'],
            line_end=row['line_end'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            importance_score=row['importance_score'],
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            chunk_index=row['chunk_index'],
            total_chunks=row['total_chunks'],
            embedding_id=row['embedding_id']
        )
    
    def record_access(
        self,
        memory_id: str,
        query: str,
        relevance_score: float
    ) -> None:
        """Record a memory access."""
        # Update access count
        self.db.execute("""
            UPDATE memory_entries
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        
        # Log access
        self.db.execute("""
            INSERT INTO memory_access_log (memory_id, query, relevance_score)
            VALUES (?, ?, ?)
        """, (memory_id, query, relevance_score))
        
        self.db.commit()
    
    # Search operations
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search by vector similarity.
        
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if self._vec_available:
            return self._vector_search_vec(query_embedding, limit, memory_types)
        else:
            return self._vector_search_fallback(query_embedding, limit, memory_types)
    
    def _vector_search_vec(
        self,
        query_embedding: np.ndarray,
        limit: int,
        memory_types: Optional[List[str]]
    ) -> List[Tuple[str, float]]:
        """Vector search using sqlite-vec."""
        embedding_json = json.dumps(query_embedding.tolist())
        
        if memory_types:
            placeholders = ','.join('?' * len(memory_types))
            query = f"""
                SELECT v.memory_id, v.distance
                FROM memory_vectors v
                JOIN memory_entries e ON v.memory_id = e.id
                WHERE e.type IN ({placeholders})
                  AND v.embedding MATCH vec_f32(?)
                ORDER BY v.distance
                LIMIT ?
            """
            params = memory_types + [embedding_json, limit]
        else:
            query = """
                SELECT memory_id, distance
                FROM memory_vectors
                WHERE embedding MATCH vec_f32(?)
                ORDER BY distance
                LIMIT ?
            """
            params = [embedding_json, limit]
        
        cursor = self.db.execute(query, params)
        
        # Convert distance to similarity (1 - distance for cosine)
        results = []
        for row in cursor.fetchall():
            similarity = 1.0 - row['distance']
            results.append((row['memory_id'], similarity))
        
        return results
    
    def _vector_search_fallback(
        self,
        query_embedding: np.ndarray,
        limit: int,
        memory_types: Optional[List[str]]
    ) -> List[Tuple[str, float]]:
        """Fallback vector search using Python cosine similarity."""
        # Get all embeddings (for small datasets)
        if memory_types:
            placeholders = ','.join('?' * len(memory_types))
            cursor = self.db.execute(f"""
                SELECT v.memory_id, v.embedding
                FROM memory_vectors v
                JOIN memory_entries e ON v.memory_id = e.id
                WHERE e.type IN ({placeholders})
            """, memory_types)
        else:
            cursor = self.db.execute("SELECT memory_id, embedding FROM memory_vectors")
        
        # Calculate similarities
        results = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((row['memory_id'], similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def lexical_search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search using FTS5 (BM25).
        
        Returns:
            List of (memory_id, score) tuples
        """
        if not self.config.enable_fts:
            return []
        
        cursor = self.db.execute("""
            SELECT memory_id, rank
            FROM memory_fts
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            # BM25 rank is negative log, convert to 0-1 score
            score = 1.0 / (1.0 + abs(row['rank']))
            results.append((row['memory_id'], score))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and lexical results.
        
        Uses Reciprocal Rank Fusion (RRF) for combining results.
        """
        # Get results from both methods
        vector_results = self.vector_search(query_embedding, limit * 2, memory_types)
        lexical_results = self.lexical_search(query, limit * 2)
        
        # Combine using RRF
        combined_scores = {}
        
        # Add vector scores
        for rank, (memory_id, score) in enumerate(vector_results, 1):
            combined_scores[memory_id] = combined_scores.get(memory_id, 0) + 1.0 / (rrf_k + rank)
        
        # Add lexical scores
        for rank, (memory_id, score) in enumerate(lexical_results, 1):
            combined_scores[memory_id] = combined_scores.get(memory_id, 0) + 1.0 / (rrf_k + rank)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Build full results
        results = []
        for memory_id, combined_score in sorted_results:
            entry = self.get_memory(memory_id)
            if entry:
                # Get individual scores
                vector_score = next((s for mid, s in vector_results if mid == memory_id), 0.0)
                lexical_score = next((s for mid, s in lexical_results if mid == memory_id), 0.0)
                
                results.append(SearchResult(
                    memory_id=memory_id,
                    content=entry.content,
                    source_file=entry.source_file,
                    line_start=entry.line_start,
                    line_end=entry.line_end,
                    semantic_score=vector_score,
                    lexical_score=lexical_score,
                    combined_score=combined_score,
                    importance_score=entry.importance_score,
                    metadata={'type': entry.type.value}
                ))
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    # Statistics and maintenance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Memory count by type
        cursor = self.db.execute("""
            SELECT type, COUNT(*) as count
            FROM memory_entries
            GROUP BY type
        """)
        stats['memory_count_by_type'] = {row['type']: row['count'] for row in cursor.fetchall()}
        
        # Total memory count
        cursor = self.db.execute("SELECT COUNT(*) as count FROM memory_entries")
        stats['total_memories'] = cursor.fetchone()['count']
        
        # Vector count
        cursor = self.db.execute("SELECT COUNT(*) as count FROM memory_vectors")
        stats['total_vectors'] = cursor.fetchone()['count']
        
        # Cache stats
        cursor = self.db.execute("SELECT COUNT(*) as count FROM embedding_cache")
        stats['cached_embeddings'] = cursor.fetchone()['count']
        
        # Database size
        cursor = self.db.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor = self.db.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        stats['db_size_bytes'] = page_count * page_size
        
        return stats
    
    def vacuum(self) -> None:
        """Optimize database."""
        self.db.execute("VACUUM")
    
    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of entries removed."""
        cursor = self.db.execute("SELECT COUNT(*) as count FROM embedding_cache")
        count = cursor.fetchone()['count']
        
        self.db.execute("DELETE FROM embedding_cache")
        self.db.commit()
        
        return count
