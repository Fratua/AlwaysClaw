--
-- Memory Database Schema for OpenClaw-Inspired AI Agent
-- SQLite with sqlite-vec extension for vector search
--

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- =====================================================
-- EMBEDDING CACHE
-- Prevents re-embedding identical content
-- =====================================================

CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance_score REAL DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
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

-- =====================================================
-- MEMORY VECTORS (sqlite-vec virtual table)
-- Stores embedding vectors for similarity search
-- =====================================================

-- Note: This requires sqlite-vec extension
-- If not available, use the fallback table below

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    memory_id TEXT PRIMARY KEY,
    embedding FLOAT[1536]
);

-- Fallback table if sqlite-vec is not available
CREATE TABLE IF NOT EXISTS memory_vectors_fallback (
    memory_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memory_entries(id) ON DELETE CASCADE
);

-- =====================================================
-- FULL-TEXT SEARCH (FTS5)
-- Enables BM25 lexical search
-- =====================================================

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    memory_id UNINDEXED,
    tokenize='porter unicode61'
);

-- FTS5 auxiliary tables are created automatically

-- =====================================================
-- MEMORY TAGS
-- Categorization system for memories
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY (memory_id) REFERENCES memory_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tag_memory ON memory_tags(memory_id);
CREATE INDEX IF NOT EXISTS idx_tag_name ON memory_tags(tag);

-- =====================================================
-- ACCESS LOG
-- Tracks memory access for importance scoring
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query TEXT,
    relevance_score REAL,
    FOREIGN KEY (memory_id) REFERENCES memory_entries(id) ON DELETE CASCADE
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
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_indexed_time ON indexed_files(indexed_at);

-- =====================================================
-- CONSOLIDATION LOG
-- Tracks memory consolidation runs
-- =====================================================

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
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
CREATE VIEW IF NOT EXISTS memory_stats AS
SELECT 
    type,
    COUNT(*) as count,
    AVG(importance_score) as avg_importance,
    MAX(created_at) as latest_entry,
    SUM(access_count) as total_accesses
FROM memory_entries
GROUP BY type;

-- Recently accessed memories
CREATE VIEW IF NOT EXISTS recent_accesses AS
SELECT 
    m.id,
    m.content,
    m.type,
    m.access_count,
    m.last_accessed,
    a.query
FROM memory_entries m
LEFT JOIN memory_access_log a ON m.id = a.memory_id
WHERE m.last_accessed > datetime('now', '-7 days')
ORDER BY m.last_accessed DESC;

-- Unindexed files (for sync operations)
CREATE VIEW IF NOT EXISTS unindexed_files AS
SELECT 
    file_path,
    indexed_at
FROM indexed_files
WHERE indexed_at < datetime('now', '-1 day');

-- =====================================================
-- TRIGGERS
-- Automatic maintenance triggers
-- =====================================================

-- Update timestamp on memory modification
CREATE TRIGGER IF NOT EXISTS update_memory_timestamp
AFTER UPDATE ON memory_entries
BEGIN
    UPDATE memory_entries 
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Clean up access log for deleted memories
CREATE TRIGGER IF NOT EXISTS cleanup_access_log
AFTER DELETE ON memory_entries
BEGIN
    DELETE FROM memory_access_log WHERE memory_id = OLD.id;
    DELETE FROM memory_tags WHERE memory_id = OLD.id;
END;

-- =====================================================
-- INITIAL DATA
-- Optional: Insert initial configuration
-- =====================================================

-- Insert system metadata
INSERT OR IGNORE INTO memory_entries (
    id, type, content, source_file, importance_score
) VALUES (
    'system_init',
    'semantic',
    'Memory system initialized',
    'system',
    1.0
);

-- =====================================================
-- MAINTENANCE QUERIES
-- Run these periodically for maintenance
-- =====================================================

-- Vacuum and optimize (run weekly)
-- VACUUM;
-- REINDEX;

-- Clean old access logs (run monthly)
-- DELETE FROM memory_access_log 
-- WHERE accessed_at < datetime('now', '-90 days');

-- Archive old memories (run based on retention policy)
-- UPDATE memory_entries 
-- SET type = 'archived' 
-- WHERE created_at < datetime('now', '-1 year');
