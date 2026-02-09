--
-- PostgreSQL Memory Database Schema for OpenClaw AI Agent
-- Adapted from init_memory_db.sql (SQLite) for PostgreSQL
--

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram-based text search (replaces FTS5)
CREATE EXTENSION IF NOT EXISTS pgvector;   -- vector similarity search (replaces sqlite-vec)

-- =====================================================
-- EMBEDDING CACHE
-- Prevents re-embedding identical content
-- =====================================================

CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BYTEA NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cache_model ON embedding_cache(model_name);

-- =====================================================
-- MEMORY ENTRIES
-- Core table for all memory entries
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('episodic', 'semantic', 'procedural', 'preference')),
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    importance_score REAL DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    embedding_id TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(type);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_entries(updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memory_entries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memory_source ON memory_entries(source_file);

-- GIN index for trigram text search (replaces FTS5)
CREATE INDEX IF NOT EXISTS idx_memory_content_trgm ON memory_entries USING gin (content gin_trgm_ops);

-- =====================================================
-- MEMORY VECTORS (pgvector)
-- Stores embedding vectors for similarity search
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY REFERENCES memory_entries(id) ON DELETE CASCADE,
    embedding vector(1536)
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_memory_vectors_embedding ON memory_vectors USING hnsw (embedding vector_cosine_ops);

-- =====================================================
-- MEMORY TAGS
-- Categorization system for memories
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (memory_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tag_memory ON memory_tags(memory_id);
CREATE INDEX IF NOT EXISTS idx_tag_name ON memory_tags(tag);

-- =====================================================
-- ACCESS LOG
-- Tracks memory access for importance scoring
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_access_log (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    query TEXT,
    relevance_score REAL
);

CREATE INDEX IF NOT EXISTS idx_access_memory ON memory_access_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_time ON memory_access_log(accessed_at);

-- =====================================================
-- DAILY LOGS INDEX
-- Tracks which daily logs have been indexed
-- =====================================================

CREATE TABLE IF NOT EXISTS indexed_files (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    chunk_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_indexed_time ON indexed_files(indexed_at);

-- =====================================================
-- CONSOLIDATION LOG
-- Tracks memory consolidation runs
-- =====================================================

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    events_processed INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    files_processed INTEGER DEFAULT 0,
    errors TEXT,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed'))
);

-- =====================================================
-- VIEWS
-- Useful pre-computed views
-- =====================================================

-- Memory statistics by type
CREATE OR REPLACE VIEW memory_stats AS
SELECT
    type,
    COUNT(*) as count,
    AVG(importance_score) as avg_importance,
    MAX(created_at) as latest_entry,
    SUM(access_count) as total_accesses
FROM memory_entries
GROUP BY type;

-- Recently accessed memories
CREATE OR REPLACE VIEW recent_accesses AS
SELECT
    m.id,
    m.content,
    m.type,
    m.access_count,
    m.last_accessed,
    a.query
FROM memory_entries m
LEFT JOIN memory_access_log a ON m.id = a.memory_id
WHERE m.last_accessed > NOW() - INTERVAL '7 days'
ORDER BY m.last_accessed DESC;

-- Unindexed files (for sync operations)
CREATE OR REPLACE VIEW unindexed_files AS
SELECT
    file_path,
    indexed_at
FROM indexed_files
WHERE indexed_at < NOW() - INTERVAL '1 day';

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- Automatic maintenance
-- =====================================================

-- Update timestamp on memory modification
CREATE OR REPLACE FUNCTION update_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_memory_timestamp ON memory_entries;
CREATE TRIGGER update_memory_timestamp
    BEFORE UPDATE ON memory_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_timestamp();

-- =====================================================
-- INITIAL DATA
-- =====================================================

INSERT INTO memory_entries (id, type, content, source_file, importance_score)
VALUES ('system_init', 'semantic', 'Memory system initialized', 'system', 1.0)
ON CONFLICT (id) DO NOTHING;
