# Memory System Architecture Summary
## Windows 10 OpenClaw-Inspired AI Agent

---

## Executive Overview

This document provides a high-level summary of the memory system architecture designed for a Windows 10-based OpenClaw-inspired AI agent running GPT-5.2 with 24/7 operation capability.

---

## Core Architecture: Multi-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: WORKING MEMORY (Context Window)                        │
│  - Active conversation context (~200K tokens for GPT-5.2)       │
│  - Current task state                                           │
│  - Temporary computation                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2: SHORT-TERM MEMORY (Daily Context)                      │
│  - Today + Yesterday logs (memory/YYYY-MM-DD.md)                │
│  - Session transcripts                                          │
│  - Recent interactions (30 days indexed)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3: LONG-TERM MEMORY (Curated Knowledge)                   │
│  - MEMORY.md: Decisions, preferences, durable facts             │
│  - SEMANTIC.md: Facts, concepts, relationships                  │
│  - PROCEDURAL.md: Skills, workflows, procedures                 │
│  - USER.md: User profile and preferences                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  TIER 4: ARCHIVAL MEMORY (Historical Storage)                   │
│  - Compressed daily logs (>30 days)                             │
│  - Session archives                                             │
│  - Episodic summaries                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Types (Human-Inspired)

| Memory Type | Purpose | Storage | Retention |
|-------------|---------|---------|-----------|
| **Episodic** | Events, experiences, specific interactions | SQLite + Markdown | 30 days active, then archived |
| **Semantic** | Facts, concepts, knowledge graph | SEMANTIC.md + SQLite | Permanent |
| **Procedural** | Skills, workflows, learned procedures | PROCEDURAL.md | Permanent (with refinement) |
| **Working** | Current context, active conversation | RAM/Context Window | Session only |

---

## Key Design Principles

### 1. File-First Architecture
- **Markdown files are the source of truth**
- SQLite serves as an indexing layer for fast retrieval
- Human-readable and version-controllable (Git-compatible)

### 2. Hybrid Search (BM25 + Vector)
- **BM25 (30% weight)**: Lexical matching for exact terms
- **Vector (70% weight)**: Semantic similarity for meaning
- **Reciprocal Rank Fusion** for combining results

### 3. Automatic Memory Flush
- Triggers at 176K tokens (200K context - 20K reserve - 4K threshold)
- Pre-compaction preservation of important context
- Silent operation (NO_REPLY if nothing to save)

### 4. Memory Consolidation
- **Daily** (3:00 AM): Process logs, extract patterns, update semantic
- **Weekly** (Sunday 2:00 AM): Refine procedures, compress archives
- **Monthly** (1st, 1:00 AM): Deep consolidation, full backup

---

## File Structure

```
C:/OpenClaw/
├── memory/
│   ├── MEMORY.md              # Curated long-term memory
│   ├── SEMANTIC.md            # Facts, concepts, knowledge graph
│   ├── PROCEDURAL.md          # Skills, workflows, procedures
│   ├── EPISODIC.md            # Significant events
│   ├── USER.md                # User profile
│   ├── IDENTITY.md            # Agent identity
│   ├── SOUL.md                # Agent persona
│   ├── AGENTS.md              # Multi-agent coordination
│   ├── 2025-01-15.md          # Daily logs (auto-generated)
│   ├── 2025-01-16.md
│   └── archive/               # Compressed old memories
├── sessions/                  # Session transcripts
├── index/
│   ├── memory.sqlite          # SQLite vector database
│   └── cache.json             # Embedding cache
└── config/
    ├── memory.yaml            # Memory configuration
    └── retention.yaml         # Retention policies
```

---

## Core Components

### 1. Episodic Memory System
- **Episode Structure**: ID, timestamp, type, description, context, importance, embedding
- **Creation Pipeline**: Event detection → Episode creation → Embedding generation → Storage
- **Retrieval**: Vector similarity + temporal proximity + importance weighting

### 2. Semantic Memory System
- **Knowledge Graph**: Entities, relationships, facts with embeddings
- **Operations**: Add/query entities, traverse relationships, infer new facts
- **Verification**: Consistency checking, cross-referencing, temporal validation

### 3. Procedural Memory System
- **Skill Representation**: Steps, preconditions, postconditions, error handling
- **Learning**: Extract from successful episodes, refine based on new experiences
- **Execution**: Workflow engine with monitoring and recovery

### 4. Memory Consolidation
- **Pattern Extraction**: Temporal, behavioral, causal patterns
- **Fact Extraction**: Convert high-confidence patterns to semantic facts
- **Archival**: Compress old data, maintain indexes

### 5. Memory Retrieval
- **Hybrid Search**: BM25 + Vector + Temporal
- **Contextual Retrieval**: Rank by user match, temporal proximity, topic overlap
- **Result Fusion**: Reciprocal Rank Fusion algorithm

### 6. Memory Updates
- **Validation**: Schema validation, size limits, format checking
- **Conflict Resolution**: Three-way merge for concurrent edits, LLM for contradictions
- **Automatic Flush**: Pre-compaction preservation

### 7. Forgetting & Archiving
- **Retention Policies**: Configurable by memory type and importance
- **Selective Forgetting**: Age-based, relevance-based, importance-based, access-based
- **Compression**: Gzip level 9 for old archives

---

## SQLite Schema (Key Tables)

```sql
-- Episodic memories with vector embeddings
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    type TEXT NOT NULL,  -- user_interaction, system_event, error, decision
    description TEXT NOT NULL,
    context TEXT,  -- JSON
    importance INTEGER NOT NULL,  -- 1=LOW, 2=MEDIUM, 3=HIGH
    embedding FLOAT[],  -- sqlite-vec (768 dimensions)
    related_episodes TEXT,  -- JSON array
    archived BOOLEAN DEFAULT FALSE
);

-- Semantic entities
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- Person, Service, Concept
    name TEXT NOT NULL,
    properties TEXT,  -- JSON
    embedding FLOAT[],
    confidence REAL NOT NULL
);

-- Relationships between entities
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- uses, prefers, triggers
    embedding FLOAT[],
    confidence REAL NOT NULL
);

-- Facts
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    category TEXT NOT NULL,  -- System, User, Domain
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    embedding FLOAT[]
);

-- Skills/Procedures
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    steps TEXT,  -- JSON
    success_rate REAL NOT NULL,
    execution_count INTEGER DEFAULT 0
);

-- FTS5 for BM25 search
CREATE VIRTUAL TABLE memory_fts USING fts5(content, type, source_file);

-- Vector search using sqlite-vec
CREATE VIRTUAL TABLE memory_vectors USING vec0(embedding FLOAT[768]);
```

---

## Configuration (memory.yaml)

```yaml
memory:
  storage:
    type: sqlite
    path: index/memory.sqlite
    backup:
      enabled: true
      interval: daily
      keep_count: 7
  
  embeddings:
    provider: auto  # auto, local, openai, gemini
    dimensions: 768
  
  search:
    hybrid:
      enabled: true
      bm25_weight: 0.3
      vector_weight: 0.7
    max_results: 10
    min_score: 0.5
  
  chunking:
    tokens: 400
    overlap: 80
  
  context:
    working_memory_tokens: 4000
    recent_memories: 50
    daily_logs_to_load: 2
  
  consolidation:
    daily:
      time: "03:00"
      enabled: true
  
  flush:
    enabled: true
    soft_threshold_tokens: 4000
    reserve_tokens_floor: 20000
```

---

## Retention Policies (retention.yaml)

```yaml
retention_policies:
  episodic:
    active_period: 30 days
    archive_after: 30 days
    compress_after: 90 days
    delete_after: 365 days
    importance_override:
      HIGH: never_delete
      MEDIUM: archive_after_60_days
      LOW: archive_after_7_days
  
  daily_logs:
    keep_active: 2 days
    archive_after: 7 days
    compress_after: 30 days
    delete_after: 180 days
  
  semantic:
    retention: permanent
    confidence_pruning:
      enabled: true
      threshold: 0.3
      check_interval: 30 days
  
  procedural:
    retention: permanent
    success_rate_pruning:
      enabled: true
      threshold: 0.1
      min_executions: 10
```

---

## Windows 10 Integration

### Service-Based Operation
- Run as Windows Service for 24/7 operation
- Heartbeat monitoring
- Graceful shutdown with context preservation

### Task Scheduler Integration
- Daily consolidation at 3:00 AM
- Weekly maintenance on Sundays
- Monthly deep consolidation

### File System
- Atomic file writes using temp + rename
- Directory watching for file changes
- Windows path handling

---

## Key Metrics & Performance

| Metric | Target |
|--------|--------|
| Search Latency | <100ms for 10K chunks |
| Local Embedding | ~50 tokens/sec |
| OpenAI Embedding | ~1000 tokens/sec |
| Index Size | ~5KB per 1K tokens |
| Memory Flush | Automatic at 176K tokens |
| Consolidation | Daily at 3:00 AM |

---

## Implementation Modules

```
openclaw/memory/
├── core/              # Types, context, exceptions
├── storage/           # SQLite, file storage, index manager
├── embeddings/        # Provider interface, local, OpenAI, Gemini
├── search/            # Hybrid, BM25, vector search
├── episodic/          # Episode management, creation, retrieval
├── semantic/          # Knowledge graph, entities, facts, inference
├── procedural/        # Skills, workflows, learning
├── consolidation/     # Scheduler, pattern extraction
├── archival/          # Archiver, compression, forgetting
└── update/            # Update manager, validator, flush
```

---

## API Overview

```python
class MemorySystem:
    # Initialization
    async def initialize(cls, config: Config) -> "MemorySystem"
    
    # Episodic
    async def create_episode(self, event: Event) -> Episode
    async def get_episodes(self, query: str, max_results: int = 10) -> List[Episode]
    
    # Semantic
    async def add_entity(self, entity: Entity) -> None
    async def query_knowledge(self, query: str) -> List[Union[Entity, Fact]]
    
    # Procedural
    async def learn_skill(self, episode: Episode) -> Skill
    async def execute_workflow(self, workflow: Workflow) -> ExecutionResult
    
    # Search
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]
    
    # Context
    async def load_context(self, agent_id: str, user_id: str) -> AgentContext
    async def save_context(self, context: AgentContext) -> None
    
    # Maintenance
    async def consolidate(self) -> ConsolidationResult
    async def archive_old_memories(self) -> ArchivalResult
```

---

## Summary

This memory system provides:

1. **Human-like memory organization** with episodic, semantic, and procedural types
2. **File-first design** for transparency and version control
3. **Hybrid search** combining lexical and semantic retrieval
4. **Automatic consolidation** to prevent context loss
5. **Selective forgetting** for efficient memory management
6. **Windows 10 integration** for 24/7 service operation
7. **Scalable architecture** supporting 15 agentic loops and GPT-5.2

The complete technical specification is available in `memory_system_specification.md`.
