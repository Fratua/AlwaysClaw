"""
Main Memory Manager for OpenClaw-Inspired AI Agent
Central coordinator for all memory operations
"""

import asyncio
import uuid
import json
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from memory_models import (
    MemoryConfig, MemoryEntry, MemoryChunk, MemoryType,
    SearchResult, WriteResult, FlushResult, AgentMessage,
    EmbeddingConfig, ContextBudget, ConsolidationReport
)
from memory_chunker import MarkdownChunker, SmartChunker
from vector_store import VectorStore, VectorStoreConfig
from context_manager import ContextWindowManager
from memory_consolidator import MemoryConsolidator, ImportanceScorer


class EmbeddingProvider:
    """
    Provider for generating text embeddings.
    
    Supports multiple backends:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - Google Gemini (gemini-embedding-001)
    - Local models (via sentence-transformers)
    - Auto-selection with fallback chain
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_name = config.get_model_name()
        self.dimension = config.dimension
        
        # Provider state
        self._provider = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the embedding provider."""
        provider = self.config.provider
        
        if provider == 'auto':
            # Try providers in order: local -> openai -> gemini
            if self._can_use_local():
                self._provider = 'local'
            elif self._can_use_openai():
                self._provider = 'openai'
            elif self._can_use_gemini():
                self._provider = 'gemini'
            else:
                raise RuntimeError("No embedding provider available")
        else:
            self._provider = provider
    
    def _can_use_local(self) -> bool:
        """Check if local embedding is available."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            return False
    
    def _can_use_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.config.openai_api_key or 
                   self._get_env_key('OPENAI_API_KEY'))
    
    def _can_use_gemini(self) -> bool:
        """Check if Gemini is configured."""
        return bool(self.config.gemini_api_key or 
                   self._get_env_key('GEMINI_API_KEY'))
    
    def _get_env_key(self, key: str) -> Optional[str]:
        """Get API key from environment."""
        import os
        return os.environ.get(key)
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self._provider == 'openai':
            return await self._embed_openai(text)
        elif self._provider == 'gemini':
            return await self._embed_gemini(text)
        elif self._provider == 'local':
            return await self._embed_local(text)
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if self._provider == 'openai':
            return await self._embed_batch_openai(texts)
        elif self._provider == 'gemini':
            return await self._embed_batch_gemini(texts)
        elif self._provider == 'local':
            return await self._embed_batch_local(texts)
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")
    
    async def _embed_openai(self, text: str) -> np.ndarray:
        """Embed using OpenAI API."""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key or 
                     self._get_env_key('OPENAI_API_KEY')
        )
        
        response = await client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def _embed_batch_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed using OpenAI API."""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key or 
                     self._get_env_key('OPENAI_API_KEY')
        )
        
        response = await client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        return [
            np.array(data.embedding, dtype=np.float32)
            for data in response.data
        ]
    
    async def _embed_gemini(self, text: str) -> np.ndarray:
        """Embed using Google Gemini API."""
        import google.generativeai as genai
        
        genai.configure(
            api_key=self.config.gemini_api_key or 
                     self._get_env_key('GEMINI_API_KEY')
        )
        
        model = genai.GenerativeModel(self.model_name)
        result = await model.embed_content_async(text)
        
        return np.array(result.embedding, dtype=np.float32)
    
    async def _embed_batch_gemini(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed using Gemini API."""
        # Gemini doesn't support true batching, so we do sequential
        results = []
        for text in texts:
            embedding = await self._embed_gemini(text)
            results.append(embedding)
        return results
    
    async def _embed_local(self, text: str) -> np.ndarray:
        """Embed using local model."""
        from sentence_transformers import SentenceTransformer
        
        model_path = self.config.local_model_path or 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_path)
        
        embedding = model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)
    
    async def _embed_batch_local(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed using local model."""
        from sentence_transformers import SentenceTransformer
        
        model_path = self.config.local_model_path or 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_path)
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [np.array(e, dtype=np.float32) for e in embeddings]


class MemoryManager:
    """
    Central memory management for the AI agent.
    
    Coordinates:
    - File-based memory storage (Markdown)
    - Vector search indexing (SQLite-vec)
    - Context window management
    - Memory consolidation
    - Daily logging
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Ensure directory structure
        self._ensure_directories()
        
        # Initialize components
        self.chunker = SmartChunker(self.config.chunking)
        self.embedder: Optional[EmbeddingProvider] = None
        self.vector_store: Optional[VectorStore] = None
        self.context_manager: Optional[ContextWindowManager] = None
        self.consolidator: Optional[MemoryConsolidator] = None
        self.importance_scorer = ImportanceScorer()
        
        # State
        self._initialized = False
        self._write_queue: List[Dict] = []
        self._batch_timer: Optional[asyncio.Task] = None
    
    def _ensure_directories(self) -> None:
        """Create required directory structure."""
        for path in [
            self.config.memory_dir,
            self.config.daily_dir,
            self.config.sessions_dir,
            self.config.base_dir / 'state',
            self.config.base_dir / 'logs'
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self, llm_client=None) -> None:
        """Initialize memory system."""
        if self._initialized:
            return
        
        # Initialize embedding provider
        self.embedder = EmbeddingProvider(self.config.embedding)
        
        # Initialize vector store
        vector_config = VectorStoreConfig(
            db_path=self.config.db_path,
            embedding_dimension=self.config.embedding.dimension
        )
        self.vector_store = VectorStore(vector_config).connect()
        
        # Initialize context manager
        self.context_manager = ContextWindowManager(
            llm_model=self.config.llm_model,
            budget=self.config.context_budget
        )
        
        # Initialize consolidator
        if llm_client:
            self.consolidator = MemoryConsolidator(
                memory_manager=self,
                llm_client=llm_client,
                config=self.config.consolidation.__dict__
            )
        
        # Sync files to index
        await self._sync_files_to_index()
        
        # Start batch write timer
        self._start_batch_timer()
        
        self._initialized = True
        print(f"Memory system initialized at {self.config.base_dir}")
    
    def _start_batch_timer(self) -> None:
        """Start timer for batch writes."""
        async def batch_loop():
            while True:
                await asyncio.sleep(self.config.write.batch_interval_seconds)
                if self._write_queue:
                    await self._flush_write_queue()
        
        self._batch_timer = asyncio.create_task(batch_loop())
    
    async def _sync_files_to_index(self) -> None:
        """Sync unindexed memory files to vector store."""
        # Get list of indexed files
        cursor = self.vector_store.db.execute(
            "SELECT DISTINCT source_file FROM memory_entries"
        )
        indexed_files = {row[0] for row in cursor.fetchall()}
        
        # Find unindexed files
        for md_file in self.config.memory_dir.rglob('*.md'):
            file_path = str(md_file)
            if file_path not in indexed_files:
                await self._index_file(md_file)
    
    async def _index_file(self, file_path: Path) -> None:
        """Index a markdown file in the vector store."""
        try:
            content = file_path.read_text(encoding='utf-8')
            chunks = self.chunker.chunk(content, source_path=str(file_path))
            
            for chunk in chunks:
                # Check cache
                cached = self.vector_store.get_cached_embedding(chunk.content_hash)
                
                if cached is not None:
                    embedding = cached
                else:
                    # Generate embedding
                    embedding = await self.embedder.embed(chunk.content)
                    
                    # Cache it
                    self.vector_store.cache_embedding(
                        chunk.content_hash,
                        embedding,
                        self.config.embedding.get_model_name()
                    )
                
                # Create memory entry
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    type=MemoryType.EPISODIC if 'daily' in str(file_path) else MemoryType.SEMANTIC,
                    content=chunk.content,
                    source_file=file_path,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    created_at=datetime.now(),
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
                
                # Insert into vector store
                self.vector_store.insert_memory(entry, embedding)
            
            print(f"Indexed {len(chunks)} chunks from {file_path.name}")
            
        except Exception as e:
            print(f"Failed to index {file_path}: {e}")
    
    # Public API
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        memory_types: Optional[List[str]] = None,
        min_score: float = 0.3
    ) -> List[SearchResult]:
        """
        Search across all memory stores.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            memory_types: Filter by memory types
            min_score: Minimum relevance score
            
        Returns:
            List of search results
        """
        if not self._initialized:
            raise RuntimeError("Memory system not initialized")
        
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=max_results,
            memory_types=memory_types,
            rrf_k=self.config.search.rrf_k
        )
        
        # Filter by minimum score
        results = [r for r in results if r.combined_score >= min_score]
        
        # Record access for each result
        for result in results:
            self.vector_store.record_access(
                result.memory_id,
                query,
                result.combined_score
            )
        
        return results
    
    async def remember(
        self,
        content: str,
        category: str = "semantic",
        importance: float = 0.5,
        immediate: bool = False,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> WriteResult:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            category: Memory category (semantic, episodic, etc.)
            importance: Importance score (0-1)
            immediate: Write immediately (bypass batching)
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Write result
        """
        if not self._initialized:
            raise RuntimeError("Memory system not initialized")
        
        # Determine write strategy
        if immediate or importance >= self.config.write.immediate_importance_threshold:
            return await self._write_immediate(
                content, category, importance, tags, metadata
            )
        else:
            return self._queue_write(
                content, category, importance, tags, metadata
            )
    
    async def _write_immediate(
        self,
        content: str,
        category: str,
        importance: float,
        tags: Optional[List[str]],
        metadata: Optional[Dict]
    ) -> WriteResult:
        """Write memory immediately."""
        # Determine memory type
        memory_type = MemoryType.SEMANTIC
        if category in ['episodic', 'event']:
            memory_type = MemoryType.EPISODIC
        elif category in ['procedural', 'skill']:
            memory_type = MemoryType.PROCEDURAL
        elif category in ['preference']:
            memory_type = MemoryType.PREFERENCE
        
        # Write to appropriate file
        if memory_type == MemoryType.EPISODIC:
            file_path = self._get_daily_log_path()
        else:
            file_path = self.config.memory_dir / 'MEMORY.md'
        
        # Append to file
        await self._append_to_file(file_path, content, category, importance)
        
        # Generate embedding
        embedding = await self.embedder.embed(content)
        
        # Create entry
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=memory_type,
            content=content,
            source_file=file_path,
            created_at=datetime.now(),
            importance_score=importance,
            tags=set(tags or []),
            metadata=metadata or {}
        )
        
        # Index in vector store
        self.vector_store.insert_memory(entry, embedding)
        
        return WriteResult(
            success=True,
            memory_id=entry.id,
            latency_ms=0  # Would measure actual latency
        )
    
    def _queue_write(
        self,
        content: str,
        category: str,
        importance: float,
        tags: Optional[List[str]],
        metadata: Optional[Dict]
    ) -> WriteResult:
        """Queue memory for batch write."""
        self._write_queue.append({
            'content': content,
            'category': category,
            'importance': importance,
            'tags': tags,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
        
        # Check if batch is full
        if len(self._write_queue) >= self.config.write.batch_size:
            asyncio.create_task(self._flush_write_queue())
        
        return WriteResult(
            success=True,
            batched=True,
            queue_size=len(self._write_queue)
        )
    
    async def _flush_write_queue(self) -> None:
        """Flush queued writes to storage."""
        if not self._write_queue:
            return
        
        queue = self._write_queue.copy()
        self._write_queue.clear()
        
        # Batch generate embeddings
        contents = [item['content'] for item in queue]
        embeddings = await self.embedder.embed_batch(contents)
        
        # Write each memory
        for item, embedding in zip(queue, embeddings):
            await self._write_immediate(
                content=item['content'],
                category=item['category'],
                importance=item['importance'],
                tags=item['tags'],
                metadata=item['metadata']
            )
        
        print(f"Flushed {len(queue)} batched writes")
    
    def _get_daily_log_path(self) -> Path:
        """Get path for today's daily log."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.config.daily_dir / f"{today}.md"
    
    async def _append_to_file(
        self,
        file_path: Path,
        content: str,
        category: str,
        importance: float
    ) -> None:
        """Append content to a memory file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        entry_text = f"""
## [{timestamp}] {category.upper()}

{content}

*Importance: {importance:.2f}*

---
"""
        
        # Append to file
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(entry_text)
    
    async def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 50000
    ) -> str:
        """
        Retrieve relevant context for a user query.
        
        Args:
            query: User query
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        # Search for relevant memories
        results = await self.search(query, max_results=20)
        
        # Format within token budget
        context_parts = []
        used_tokens = 0
        
        for result in results:
            # Format result
            content = f"""
[{result.source_file.name}:{result.line_start}]
{result.content}
"""
            tokens = self.context_manager.count_tokens(content)
            
            if used_tokens + tokens > max_tokens:
                break
            
            context_parts.append(content)
            used_tokens += tokens
        
        return '\n---\n'.join(context_parts)
    
    async def log_event(
        self,
        event_type: str,
        title: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log an event to the daily log."""
        log_path = self._get_daily_log_path()
        
        # Create log entry
        timestamp = datetime.now().strftime('%H:%M')
        
        entry = f"""
### {timestamp} - {title}

**Type**: {event_type}

{content}
"""
        
        if metadata:
            entry += "\n**Metadata**:\n"
            for key, value in metadata.items():
                entry += f"- {key}: {value}\n"
        
        entry += "\n"
        
        # Append to daily log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(entry)
    
    async def consolidate(self, force: bool = False) -> Optional[ConsolidationReport]:
        """Run memory consolidation."""
        if self.consolidator:
            return await self.consolidator.consolidate(force)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        stats = self.vector_store.get_stats()
        stats['initialized'] = True
        stats['write_queue_size'] = len(self._write_queue)
        
        return stats
    
    async def close(self) -> None:
        """Shutdown memory system."""
        # Flush pending writes
        if self._write_queue:
            await self._flush_write_queue()
        
        # Cancel batch timer
        if self._batch_timer:
            self._batch_timer.cancel()
        
        # Close vector store
        if self.vector_store:
            self.vector_store.close()
        
        self._initialized = False


# Convenience functions for quick usage

async def create_memory_manager(
    base_dir: Optional[Path] = None,
    llm_client=None
) -> MemoryManager:
    """
    Create and initialize a memory manager.
    
    Args:
        base_dir: Base directory for memory storage
        llm_client: LLM client for consolidation
        
    Returns:
        Initialized MemoryManager
    """
    config = MemoryConfig()
    if base_dir:
        config.base_dir = base_dir
    
    manager = MemoryManager(config)
    await manager.initialize(llm_client)
    
    return manager
