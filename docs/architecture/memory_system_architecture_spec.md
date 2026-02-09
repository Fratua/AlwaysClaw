# Windows 10 OpenClaw-Inspired AI Agent Memory System Architecture
## Technical Specification Document

**Version:** 1.0  
**Date:** 2025  
**Target Platform:** Windows 10  
**LLM:** GPT-5.2 with Extended Thinking  

---

## Executive Summary

This document specifies a comprehensive memory system architecture for a Windows 10-based OpenClaw-inspired AI agent. The system implements a multi-tiered memory hierarchy combining file-based persistent storage with vector semantic search, optimized for 24/7 autonomous operation with Gmail, browser control, TTS/STT, Twilio integration, and full system access.

---

## Table of Contents

1. [Memory Hierarchy Architecture](#1-memory-hierarchy-architecture)
2. [Directory Structure & File Organization](#2-directory-structure--file-organization)
3. [Memory File Schemas](#3-memory-file-schemas)
4. [Vector Memory Implementation](#4-vector-memory-implementation)
5. [Memory Consolidation & Compression](#5-memory-consolidation--compression)
6. [Context Window Management](#6-context-window-management)
7. [Memory Retrieval & Relevance Scoring](#7-memory-retrieval--relevance-scoring)
8. [Memory Update & Write-Back Strategies](#8-memory-update--write-back-strategies)
9. [Implementation Code Patterns](#9-implementation-code-patterns)
10. [Performance Specifications](#10-performance-specifications)

---

## 1. Memory Hierarchy Architecture

### 1.1 Four-Layer Memory Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY HIERARCHY PYRAMID                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     ┌─────────────┐                                                          │
│     │   WORKING   │  ← Active context window (GPT-5.2: 200K tokens)        │
│     │   MEMORY    │     Volatile, session-bound                              │
│     └──────┬──────┘                                                          │
│            │                                                                 │
│     ┌─────────────┐                                                          │
│     │  SHORT-TERM │  ← Recent interactions, session cache                   │
│     │   MEMORY    │     24-48 hour retention, high fidelity                  │
│     └──────┬──────┘                                                          │
│            │                                                                 │
│     ┌─────────────┐                                                          │
│     │  EPISODIC   │  ← Event sequences, temporal experiences                │
│     │   MEMORY    │     Daily logs, conversation transcripts                │
│     └──────┬──────┘                                                          │
│            │                                                                 │
│     ┌─────────────┐                                                          │
│     │  SEMANTIC   │  ← Facts, knowledge, user preferences                   │
│     │   MEMORY    │     Curated long-term storage, vector-indexed           │
│     └──────┬──────┘                                                          │
│            │                                                                 │
│     ┌─────────────┐                                                          │
│     │ PROCEDURAL  │  ← Skills, workflows, agent behaviors                   │
│     │   MEMORY    │     Hardcoded loops, learned patterns                   │
│     └─────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Type Definitions

| Memory Type | Persistence | Scope | Access Pattern | Storage Medium |
|-------------|-------------|-------|----------------|----------------|
| **Working** | Session-only | Current task | Immediate | RAM/VRAM |
| **Short-Term** | 24-48 hours | Recent context | Fast retrieval | RAM + SQLite |
| **Episodic** | Days-Weeks | Event sequences | Temporal query | Markdown + SQLite |
| **Semantic** | Permanent | Facts/Knowledge | Semantic search | Markdown + Vector DB |
| **Procedural** | Permanent | Skills/Workflows | Direct load | Code + Config |

### 1.3 Memory Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  USER    │───→│  AGENT   │───→│ WORKING  │───→│   LLM    │             │
│   │  INPUT   │    │  LOOP    │    │  MEMORY  │    │  (GPT-5) │             │
│   └──────────┘    └──────────┘    └────┬─────┘    └──────────┘             │
│                                        │                                     │
│                    ┌───────────────────┼───────────────────┐                │
│                    ↓                   ↓                   ↓                │
│              ┌──────────┐       ┌──────────┐       ┌──────────┐            │
│              │  SHORT   │       │ EPISODIC │       │ SEMANTIC │            │
│              │  TERM    │       │  MEMORY  │       │  MEMORY  │            │
│              └──────────┘       └──────────┘       └──────────┘            │
│                    ↑                   ↑                   ↑                │
│                    └───────────────────┴───────────────────┘                │
│                              VECTOR INDEX (SQLite-vec)                       │
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                              │
│   │PROCEDURAL│    │  SOUL/   │    │  USER    │                              │
│   │  MEMORY  │    │ IDENTITY │    │ PROFILE  │                              │
│   └──────────┘    └──────────┘    └──────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & File Organization

### 2.1 Base Directory Structure

```
%USERPROFILE%\.openclaw\
├── config\
│   ├── agents.yaml              # Agent configurations
│   ├── loops.yaml               # 15 hardcoded agentic loops
│   ├── skills.yaml              # Skill registry
│   └── system.yaml              # System settings
│
├── memory\
│   ├── index.db                 # SQLite vector database
│   ├── index.db-wal             # Write-ahead log
│   ├── cache.json               # Embedding cache
│   ├── MEMORY.md                # Curated long-term memory
│   ├── SOUL.md                  # Agent persona & boundaries
│   ├── IDENTITY.md              # Agent identity profile
│   ├── USER.md                  # User profile & preferences
│   ├── AGENTS.md                # Operating instructions
│   └── daily\
│       ├── 2025-01-15.md
│       ├── 2025-01-16.md
│       └── ...                  # Daily append-only logs
│
├── sessions\
│   ├── 2025-01-15-intro-session.md
│   ├── 2025-01-16-code-review.md
│   └── ...                      # Archived conversation transcripts
│
├── skills\
│   ├── builtin\                 # Built-in skills
│   ├── community\               # Downloaded community skills
│   └── custom\                  # User-created skills
│
├── state\
│   ├── heartbeat.json           # Cron job states
│   ├── session_state.json       # Current session state
│   └── agent_states\            # Per-agent state files
│
├── logs\
│   ├── agent.log                # Main agent log
│   ├── error.log                # Error log
│   └── audit\                   # Security audit logs
│
└── temp\                        # Temporary working directory
```

### 2.2 Windows-Specific Paths

```python
# Windows 10 path configuration
import os
from pathlib import Path

class WindowsPaths:
    """Windows 10 path resolution for OpenClaw agent."""
    
    BASE_DIR = Path(os.environ.get('OPENCLAW_HOME', 
                                   Path.home() / '.openclaw'))
    
    MEMORY_DIR = BASE_DIR / 'memory'
    DAILY_DIR = MEMORY_DIR / 'daily'
    SESSIONS_DIR = BASE_DIR / 'sessions'
    CONFIG_DIR = BASE_DIR / 'config'
    STATE_DIR = BASE_DIR / 'state'
    LOGS_DIR = BASE_DIR / 'logs'
    SKILLS_DIR = BASE_DIR / 'skills'
    TEMP_DIR = BASE_DIR / 'temp'
    
    # Windows-specific integrations
    DOWNLOADS_DIR = Path.home() / 'Downloads'
    DOCUMENTS_DIR = Path.home() / 'Documents'
    DESKTOP_DIR = Path.home() / 'Desktop'
    
    @classmethod
    def ensure_structure(cls) -> None:
        """Create all required directories."""
        for path in [cls.MEMORY_DIR, cls.DAILY_DIR, cls.SESSIONS_DIR,
                     cls.CONFIG_DIR, cls.STATE_DIR, cls.LOGS_DIR,
                     cls.SKILLS_DIR, cls.TEMP_DIR]:
            path.mkdir(parents=True, exist_ok=True)
```

---

## 3. Memory File Schemas

### 3.1 MEMORY.md Schema (Curated Long-Term Memory)

```markdown
# Agent Memory

> Last Updated: 2025-01-16T14:32:00Z  
> Version: 1.3  
> Agent: ClawWin-01

---

## User Preferences

### Communication Style
- Preferred greeting: "Hey there"
- Response length: Concise, bullet points preferred
- Tone: Professional but friendly
- Language: English (US)

### Notification Settings
- Quiet hours: 22:00 - 07:00
- Urgent only during quiet hours
- Preferred channels: Email > SMS > Voice

## Project Context

### Active Projects
- **Project Alpha**: Windows automation tool (priority: high)
- **Project Beta**: Email classifier (priority: medium)

### Code Conventions
- Python: PEP 8, type hints required
- JavaScript: ESLint standard, async/await preferred
- Documentation: Google style docstrings

## Important Decisions

### 2025-01-15: Architecture Decision
- Chose SQLite over PostgreSQL for local deployment
- Rationale: Zero-config, single-file portability

### 2025-01-14: API Strategy
- Using Twilio for voice/SMS (already integrated)
- Gmail API for email operations

## Recurring Tasks

### Daily
- [ ] Check Gmail at 09:00
- [ ] Review calendar at 08:30
- [ ] System health check at 12:00

### Weekly
- [ ] Backup memory files (Sundays)
- [ ] Skill updates (Mondays)

## Relationships & Contacts

- **John (Manager)**: Prefers email summaries
- **Sarah (DevOps)**: Technical discussions, Slack preferred

## Learned Patterns

### User Workflows
1. Morning routine: Check email → Calendar → Task list
2. Coding sessions: Usually 14:00-18:00
3. prefers voice commands while driving (17:30-18:00)

## System Configuration

### Preferred Tools
- Browser: Edge (work), Chrome (personal)
- Editor: VS Code with specific extensions
- Terminal: Windows Terminal with PowerShell
```

### 3.2 Daily Log Schema (YYYY-MM-DD.md)

```markdown
# Daily Log: 2025-01-16

## Session Start
- **Time**: 08:15:00Z
- **Context**: Morning routine initiated

## Events

### 08:15 - Gmail Check
- **Action**: Checked 23 new emails
- **Summary**: 3 flagged as important
- **Decisions**: Scheduled reply to John for 10:00

### 08:45 - Calendar Review
- **Action**: Reviewed today's schedule
- **Findings**: 3 meetings, 2 conflicts resolved
- **Auto-action**: Sent decline to optional standup

### 09:30 - Code Session
- **Project**: Alpha automation tool
- **Files modified**: `automation/core.py`, `config.yaml`
- **Commits**: 2 commits pushed
- **Notes**: Refactored error handling, added retry logic

### 12:00 - System Health
- **Status**: All systems operational
- **Memory usage**: 2.3GB / 16GB
- **Disk space**: 45% available
- **Network**: Stable, 85 Mbps

### 14:15 - User Query
- **Query**: "What's the status of Project Alpha?"
- **Response provided**: Summary of commits, blockers
- **Follow-up**: User requested email summary

## Session End
- **Time**: 18:30:00Z
- **Duration**: 10h 15m
- **Next scheduled**: 2025-01-17 08:00:00Z

## Notes for Tomorrow
- Complete Project Alpha documentation
- Follow up with Sarah about deployment
- Research Twilio voice improvements
```

### 3.3 SOUL.md Schema (Agent Persona)

```markdown
# Agent Soul

## Identity
- **Name**: ClawWin
- **Version**: 1.0.0
- **Purpose**: Windows 10 AI assistant for productivity automation

## Personality

### Core Traits
- Helpful and proactive
- Precise and detail-oriented
- Respectful of user time
- Transparent about capabilities

### Communication Style
- Clear and concise
- Uses appropriate technical depth
- Asks clarifying questions when needed
- Acknowledges uncertainty honestly

## Boundaries

### Will Do
- Execute authorized system commands
- Manage emails, calendar, tasks
- Control browser for automation
- Make phone calls via Twilio
- Send SMS notifications
- Execute scheduled tasks

### Won't Do
- Delete files without confirmation
- Share sensitive data externally
- Execute potentially harmful commands
- Override explicit user preferences
- Access unauthorized accounts

## Values
1. User privacy is paramount
2. Transparency in all actions
3. Continuous improvement
4. Reliability and consistency

## Emotional Intelligence

### Recognition Patterns
- Frustration: Short responses, repeated commands
- Urgency: Keywords like "ASAP", "urgent", "emergency"
- Satisfaction: Positive feedback, continued engagement

### Response Adaptations
- Detected frustration → Offer simpler solutions
- Detected urgency → Prioritize and expedite
- Detected satisfaction → Acknowledge and build on success
```

### 3.4 USER.md Schema (User Profile)

```markdown
# User Profile

## Basic Information
- **Preferred Name**: Alex
- **Pronouns**: they/them
- **Timezone**: America/New_York (EST/EDT)
- **Language**: English (US)

## Professional Context
- **Role**: Software Engineering Manager
- **Company**: TechCorp Inc.
- **Team Size**: 8 direct reports
- **Primary Focus**: Windows automation, team productivity

## Communication Preferences

### Channels (Priority Order)
1. Email (work hours)
2. Slack (quick questions)
3. SMS (urgent only)
4. Voice (driving, emergencies)

### Response Expectations
- Urgent: < 15 minutes
- Normal: < 2 hours
- Low priority: < 24 hours

## Technical Profile

### Skills
- Python (expert)
- JavaScript/TypeScript (advanced)
- PowerShell (intermediate)
- Windows administration (advanced)

### Tools & Environment
- IDE: VS Code
- Terminal: Windows Terminal + PowerShell 7
- Browser: Edge (primary), Chrome (testing)
- OS: Windows 10 Pro

## Work Patterns

### Typical Schedule
- **Start**: 08:00 - 08:30
- **Deep work**: 09:00 - 12:00, 14:00 - 17:00
- **Meetings**: 10:00 - 11:00, 15:00 - 16:00
- **End**: 17:30 - 18:00

### Preferences
- Morning: Email, planning, light tasks
- Afternoon: Coding, deep work
- Evening: Wrap-up, tomorrow prep

## Personal Context

### Family
- **Spouse**: Jordan (reminder: anniversary June 15)
- **Children**: None
- **Pets**: Dog named Bailey

### Interests
- Home automation
- Productivity systems
- Open source contributions

## Accessibility Needs
- Prefer voice commands when hands are occupied
- High contrast mode for extended screen time
- Closed captions for video calls
```

### 3.5 JSON Schema for Structured Memory

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MemoryEntry",
  "type": "object",
  "required": ["id", "type", "content", "timestamp", "metadata"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier (UUID v4)"
    },
    "type": {
      "type": "string",
      "enum": ["episodic", "semantic", "procedural", "preference"]
    },
    "content": {
      "type": "string",
      "description": "Memory content (text or compressed)"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "source": {
          "type": "string",
          "description": "Source file or system"
        },
        "category": {
          "type": "string",
          "description": "Memory category"
        },
        "importance": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "access_count": {
          "type": "integer",
          "minimum": 0
        },
        "last_accessed": {
          "type": "string",
          "format": "date-time"
        },
        "tags": {
          "type": "array",
          "items": { "type": "string" }
        },
        "embedding_id": {
          "type": "string",
          "description": "Reference to vector embedding"
        }
      }
    }
  }
}
```

---

## 4. Vector Memory Implementation

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VECTOR MEMORY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Markdown  │────→│   Chunker   │────→│  Embedding  │                  │
│   │    Files    │     │  (400 tok)  │     │   Engine    │                  │
│   └─────────────┘     └─────────────┘     └──────┬──────┘                  │
│                                                   │                          │
│                          ┌────────────────────────┘                          │
│                          ↓                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Hybrid    │←────│   SQLite    │←────│   Vector    │                  │
│   │   Search    │     │   +vec0     │     │   Store     │                  │
│   │ BM25+Cosine │     │             │     │  (vec0_vtab)│                  │
│   └──────┬──────┘     └─────────────┘     └─────────────┘                  │
│          │                                                                   │
│          ↓                                                                   │
│   ┌─────────────┐                                                            │
│   │   Ranked    │                                                            │
│   │   Results   │                                                            │
│   └─────────────┘                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 SQLite Schema with sqlite-vec

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Embedding cache table (prevents re-embedding identical content)
CREATE TABLE embedding_cache (
    content_hash TEXT PRIMARY KEY,  -- SHA-256 of content
    embedding BLOB NOT NULL,        -- float32 vector
    model_name TEXT NOT NULL,       -- e.g., 'text-embedding-3-small'
    dimension INTEGER NOT NULL,     -- e.g., 1536
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtual table for vector search (sqlite-vec)
CREATE VIRTUAL TABLE memory_vectors USING vec0(
    memory_id TEXT PRIMARY KEY,     -- References memory entry
    embedding FLOAT[1536]           -- Embedding vector
);

-- FTS5 for lexical search
CREATE VIRTUAL TABLE memory_fts USING fts5(
    content,                        -- Searchable content
    memory_id,                      -- Reference to source
    tokenize='porter unicode61'     -- Stemming + unicode support
);

-- Memory entries table
CREATE TABLE memory_entries (
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
    total_chunks INTEGER DEFAULT 1
);

-- Tags table for categorization
CREATE TABLE memory_tags (
    memory_id TEXT REFERENCES memory_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

-- Access log for importance scoring
CREATE TABLE memory_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT REFERENCES memory_entries(id) ON DELETE CASCADE,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query TEXT,                     -- What was being searched
    relevance_score REAL            -- How relevant this result was
);

-- Indexes for performance
CREATE INDEX idx_memory_type ON memory_entries(type);
CREATE INDEX idx_memory_created ON memory_entries(created_at);
CREATE INDEX idx_memory_importance ON memory_entries(importance_score DESC);
CREATE INDEX idx_memory_accessed ON memory_entries(last_accessed);
CREATE INDEX idx_access_log_memory ON memory_access_log(memory_id);
```

### 4.3 Embedding Provider Configuration

```python
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class EmbeddingProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    GEMINI = "gemini"
    AUTO = "auto"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: EmbeddingProvider = EmbeddingProvider.AUTO
    model_name: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 60
    
    # Provider-specific settings
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    local_model_path: Optional[str] = None
    
    def get_model_name(self) -> str:
        """Get the effective model name."""
        if self.model_name:
            return self.model_name
        
        model_defaults = {
            EmbeddingProvider.OPENAI: "text-embedding-3-small",
            EmbeddingProvider.GEMINI: "gemini-embedding-001",
            EmbeddingProvider.LOCAL: "all-MiniLM-L6-v2"
        }
        return model_defaults.get(self.provider, "text-embedding-3-small")

# Default configuration for Windows 10 deployment
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.AUTO,
    dimension=1536,
    batch_size=50,
    max_retries=3,
    timeout_seconds=60
)
```

### 4.4 Chunking Algorithm

```python
import hashlib
from dataclasses import dataclass
from typing import List, Iterator
import re

@dataclass
class MemoryChunk:
    """Represents a chunk of memory content."""
    content: str
    line_start: int
    line_end: int
    chunk_index: int
    total_chunks: int
    content_hash: str
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        return len(self.content) // 4

class MarkdownChunker:
    """Chunks markdown content with overlap preservation."""
    
    DEFAULT_CHUNK_TOKENS = 400
    DEFAULT_OVERLAP_TOKENS = 80
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    ):
        self.chunk_size = chunk_tokens * self.CHARS_PER_TOKEN
        self.overlap_size = overlap_tokens * self.CHARS_PER_TOKEN
    
    def chunk(
        self,
        content: str,
        source_path: str = ""
    ) -> List[MemoryChunk]:
        """
        Chunk markdown content with intelligent boundaries.
        
        Strategy:
        1. Respect header boundaries when possible
        2. Respect paragraph boundaries
        3. Use sliding window with overlap
        """
        lines = content.split('\n')
        chunks = []
        current_lines = []
        current_chars = 0
        line_no = 0
        chunk_index = 0
        
        # Header tracking for intelligent chunking
        current_header = None
        
        while line_no < len(lines):
            line = lines[line_no]
            line_length = len(line) + 1  # +1 for newline
            
            # Detect headers
            if line.startswith('#'):
                current_header = line
            
            # Check if adding this line exceeds chunk size
            if current_chars + line_length > self.chunk_size and current_lines:
                # Save current chunk
                chunk_content = '\n'.join(l['line'] for l in current_lines)
                chunks.append(self._create_chunk(
                    chunk_content, current_lines, chunk_index
                ))
                
                # Carry over overlap
                current_lines = self._get_overlap_lines(current_lines)
                current_chars = sum(len(l['line']) + 1 for l in current_lines)
                chunk_index += 1
            
            # Add current line
            current_lines.append({'line': line, 'line_no': line_no + 1})
            current_chars += line_length
            line_no += 1
        
        # Don't forget the last chunk
        if current_lines:
            chunk_content = '\n'.join(l['line'] for l in current_lines)
            chunks.append(self._create_chunk(
                chunk_content, current_lines, chunk_index
            ))
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        lines: List[dict],
        chunk_index: int
    ) -> MemoryChunk:
        """Create a MemoryChunk from accumulated lines."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return MemoryChunk(
            content=content,
            line_start=lines[0]['line_no'],
            line_end=lines[-1]['line_no'],
            chunk_index=chunk_index,
            total_chunks=0,  # Updated later
            content_hash=content_hash
        )
    
    def _get_overlap_lines(self, lines: List[dict]) -> List[dict]:
        """Get lines to carry over for overlap."""
        overlap_chars = 0
        overlap_lines = []
        
        for line_info in reversed(lines):
            line = line_info['line']
            overlap_lines.insert(0, line_info)
            overlap_chars += len(line) + 1
            if overlap_chars >= self.overlap_size:
                break
        
        return overlap_lines
```

### 4.5 Hybrid Search Implementation

```python
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    """Result from memory search."""
    memory_id: str
    content: str
    source_file: str
    line_start: int
    line_end: int
    semantic_score: float
    lexical_score: float
    combined_score: float
    metadata: dict

class HybridMemorySearcher:
    """
    Hybrid search combining BM25 (lexical) and vector (semantic) retrieval.
    Uses Reciprocal Rank Fusion (RRF) for combining results.
    """
    
    RRF_K = 60  # RRF constant
    
    def __init__(self, db_connection, embedding_provider):
        self.db = db_connection
        self.embedder = embedding_provider
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.3,
        memory_types: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search across memory stores.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum combined score threshold
            memory_types: Filter by memory types
            time_range_days: Limit to recent memories
        """
        # Get query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Parallel search: vector + lexical
        vector_results = await self._vector_search(
            query_embedding, max_results * 2
        )
        lexical_results = await self._lexical_search(
            query, max_results * 2
        )
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results, lexical_results
        )
        
        # Filter and format results
        results = []
        for memory_id, score in combined[:max_results]:
            if score < min_score:
                continue
            
            result = await self._get_memory_details(memory_id)
            if result:
                # Update access metrics
                await self._update_access_metrics(memory_id, query, score)
                results.append(result)
        
        return results
    
    async def _vector_search(
        self,
        query_embedding: np.ndarray,
        limit: int
    ) -> List[Tuple[str, float]]:
        """Search using vector similarity (cosine)."""
        cursor = self.db.cursor()
        
        # Convert embedding to blob
        embedding_blob = query_embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            SELECT memory_id, distance
            FROM memory_vectors
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (embedding_blob, limit))
        
        # Convert distance to similarity score (1 - distance for cosine)
        return [(row[0], 1.0 - row[1]) for row in cursor.fetchall()]
    
    async def _lexical_search(
        self,
        query: str,
        limit: int
    ) -> List[Tuple[str, float]]:
        """Search using FTS5 (BM25)."""
        cursor = self.db.cursor()
        
        cursor.execute("""
            SELECT memory_id, rank
            FROM memory_fts
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        # BM25 rank is negative log, convert to 0-1 score
        results = []
        for row in cursor.fetchall():
            memory_id, rank = row
            # Normalize BM25 score
            score = 1.0 / (1.0 + abs(rank))
            results.append((memory_id, score))
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        lexical_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score = Σ 1 / (k + rank)
        """
        scores = {}
        
        # Add vector scores
        for rank, (memory_id, _) in enumerate(vector_results, 1):
            scores[memory_id] = scores.get(memory_id, 0) + 1.0 / (self.RRF_K + rank)
        
        # Add lexical scores
        for rank, (memory_id, _) in enumerate(lexical_results, 1):
            scores[memory_id] = scores.get(memory_id, 0) + 1.0 / (self.RRF_K + rank)
        
        # Sort by combined score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 5. Memory Consolidation & Compression

### 5.1 Consolidation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY CONSOLIDATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Daily Logs (Raw)                                                          │
│        │                                                                     │
│        ↓                                                                     │
│   ┌─────────────┐                                                           │
│   │  Extract    │  → Identify important events, decisions, patterns         │
│   │   Events    │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ↓                                                                   │
│   ┌─────────────┐                                                           │
│   │   Summarize │  → Compress into semantic facts                            │
│   │   & Cluster │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ↓                                                                   │
│   ┌─────────────┐                                                           │
│   │   Score     │  → Calculate importance, relevance, decay                 │
│   │ Importance  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ↓                                                                   │
│   ┌─────────────┐     ┌─────────────┐                                       │
│   │   Update    │────→│  MEMORY.md  │  → Curated long-term memory           │
│   │  Semantic   │     │  (Curated)  │                                       │
│   │   Memory    │     └─────────────┘                                       │
│   └─────────────┘                                                           │
│          │                                                                   │
│          ↓                                                                   │
│   ┌─────────────┐                                                           │
│   │   Archive   │  → Move to sessions/ for historical reference             │
│   │   Old Logs  │                                                           │
│   └─────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Consolidation Algorithm

```python
from datetime import datetime, timedelta
from typing import List, Dict
import json

class MemoryConsolidator:
    """
    Consolidates ephemeral memories into durable semantic memory.
    Runs as scheduled background task.
    """
    
    # Configuration
    CONSOLIDATION_AGE_DAYS = 7      # Consolidate logs older than 7 days
    IMPORTANCE_THRESHOLD = 0.6      # Minimum importance to keep
    MAX_DAILY_LOGS = 30             # Keep last 30 days of logs
    
    def __init__(self, memory_manager, llm_client):
        self.memory = memory_manager
        self.llm = llm_client
    
    async def consolidate(self, force: bool = False) -> ConsolidationReport:
        """
        Run memory consolidation process.
        
        Steps:
        1. Identify old daily logs for consolidation
        2. Extract important events and patterns
        3. Summarize into semantic facts
        4. Update MEMORY.md
        5. Archive processed logs
        """
        report = ConsolidationReport()
        
        # Get logs eligible for consolidation
        old_logs = self._get_consolidation_candidates()
        
        for log_file in old_logs:
            try:
                # Parse daily log
                log_content = log_file.read_text()
                events = self._parse_events(log_content)
                
                # Extract important information
                important = self._filter_important_events(events)
                
                if important:
                    # Generate semantic summary
                    summary = await self._generate_semantic_summary(important)
                    
                    # Update durable memory
                    await self._update_memory_md(summary)
                    
                    # Update report
                    report.events_processed += len(events)
                    report.facts_extracted += len(summary.facts)
                    report.files_processed.append(log_file.name)
                
                # Archive the log
                await self._archive_log(log_file)
                
            except Exception as e:
                report.errors.append(f"{log_file.name}: {str(e)}")
        
        # Clean up old archived logs
        await self._cleanup_archives()
        
        return report
    
    async def _generate_semantic_summary(
        self,
        events: List[Event]
    ) -> SemanticSummary:
        """
        Use LLM to extract semantic facts from events.
        """
        prompt = f"""
        Analyze the following events from the agent's daily log and extract 
        important facts, decisions, and patterns that should be remembered long-term.
        
        Events:
        {self._format_events(events)}
        
        Extract:
        1. Key decisions made
        2. User preferences expressed
        3. Important project updates
        4. Recurring patterns observed
        5. Relationship/context information
        
        Return as JSON:
        {{
            "facts": [
                {{
                    "content": "fact statement",
                    "category": "decision|preference|project|pattern|relationship",
                    "importance": 0.0-1.0,
                    "confidence": 0.0-1.0
                }}
            ],
            "summary": "brief narrative summary"
        }}
        """
        
        response = await self.llm.generate(
            prompt,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response)
        return SemanticSummary(
            facts=[Fact(**f) for f in data['facts']],
            summary=data['summary']
        )
    
    def _calculate_importance(self, event: Event) -> float:
        """
        Calculate importance score for an event.
        
        Factors:
        - User explicit feedback (high weight)
        - Decision/Action taken (medium weight)
        - Recurring pattern (medium weight)
        - Time decay (reduces over time)
        """
        score = 0.5  # Base score
        
        # Decision events are important
        if event.type == 'decision':
            score += 0.3
        
        # User feedback indicates importance
        if event.user_feedback:
            if 'important' in event.user_feedback.lower():
                score += 0.2
            if 'remember' in event.user_feedback.lower():
                score += 0.25
        
        # Recurring patterns
        if event.is_recurring:
            score += 0.15
        
        # Time decay
        days_old = (datetime.now() - event.timestamp).days
        decay = 1.0 - (days_old / 365)  # Linear decay over 1 year
        score *= max(0.1, decay)
        
        return min(1.0, score)
```

### 5.3 Context Compression for LLM

```python
class ContextCompressor:
    """
    Compresses conversation context to fit within LLM context window.
    Implements hierarchical summarization.
    """
    
    def __init__(self, llm_client, max_tokens: int = 160000):
        self.llm = llm_client
        self.max_tokens = max_tokens
        self.reserve_tokens = 20000  # Reserve for response
    
    async def compress(
        self,
        messages: List[Message],
        current_tokens: int
    ) -> List[Message]:
        """
        Compress message history to fit context window.
        
        Strategy:
        1. Keep recent messages verbatim (last 10)
        2. Summarize older messages hierarchically
        3. Preserve system messages and critical context
        """
        if current_tokens <= self.max_tokens - self.reserve_tokens:
            return messages
        
        # Separate messages by importance
        system_msgs = [m for m in messages if m.role == 'system']
        critical_msgs = self._extract_critical(messages)
        recent_msgs = messages[-10:]  # Keep last 10 verbatim
        older_msgs = messages[:-10]
        
        # Calculate available tokens for summary
        used_tokens = (
            self._count_tokens(system_msgs) +
            self._count_tokens(critical_msgs) +
            self._count_tokens(recent_msgs)
        )
        available = self.max_tokens - self.reserve_tokens - used_tokens
        
        # Summarize older messages
        if older_msgs and available > 1000:
            summary = await self._summarize_messages(older_msgs, available)
            compressed = (
                system_msgs +
                [Message(role='system', content=f"Context summary: {summary}")] +
                critical_msgs +
                recent_msgs
            )
        else:
            compressed = system_msgs + critical_msgs + recent_msgs
        
        return compressed
    
    async def _summarize_messages(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> str:
        """
        Generate a concise summary of message history.
        """
        # Group messages by topic/time
        groups = self._group_messages(messages)
        
        summaries = []
        for group in groups:
            prompt = f"""
            Summarize the following conversation segment concisely.
            Preserve: key decisions, action items, user preferences, context.
            
            Messages:
            {self._format_messages(group)}
            
            Summary (max 200 words):
            """
            
            summary = await self.llm.generate(prompt, max_tokens=300)
            summaries.append(summary)
        
        return " | ".join(summaries)
    
    def _extract_critical(self, messages: List[Message]) -> List[Message]:
        """
        Extract messages containing critical information.
        """
        critical = []
        for msg in messages:
            content = msg.content.lower()
            # Identify critical patterns
            if any(pattern in content for pattern in [
                'decision:', 'important:', 'remember:',
                'api key', 'password', 'credential',
                'error:', 'exception:', 'failed'
            ]):
                critical.append(msg)
        return critical
```

---

## 6. Context Window Management

### 6.1 Context Budget Allocation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GPT-5.2 CONTEXT WINDOW ALLOCATION (200K tokens)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         200,000 tokens                              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  System Prompt + Identity    │  10,000 tokens  │  5%               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Active Skills/Tools         │  15,000 tokens  │  7.5%             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Retrieved Memory            │  50,000 tokens  │  25%              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Conversation History        │  75,000 tokens  │  37.5%            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Current Request + Context   │  30,000 tokens  │  15%              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Reserved for Response       │  20,000 tokens  │  10%              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Context Manager Implementation

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import tiktoken

@dataclass
class ContextBudget:
    """Defines token allocation for context components."""
    system_tokens: int = 10000
    skills_tokens: int = 15000
    memory_tokens: int = 50000
    history_tokens: int = 75000
    request_tokens: int = 30000
    response_reserve: int = 20000
    
    @property
    def total(self) -> int:
        return (
            self.system_tokens +
            self.skills_tokens +
            self.memory_tokens +
            self.history_tokens +
            self.request_tokens +
            self.response_reserve
        )

class ContextWindowManager:
    """
    Manages LLM context window allocation and optimization.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-5.2",
        budget: Optional[ContextBudget] = None
    ):
        self.model = llm_model
        self.budget = budget or ContextBudget()
        self.encoder = tiktoken.encoding_for_model(llm_model)
        
        # Token thresholds for actions
        self.COMPACTION_THRESHOLD = 0.85  # 85% full → compact
        self.FLUSH_THRESHOLD = 0.90       # 90% full → flush to memory
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def count_message_tokens(self, messages: List[Message]) -> int:
        """Count tokens in message list."""
        total = 0
        for msg in messages:
            # Base tokens per message
            total += 4
            # Content tokens
            total += self.count_tokens(msg.content)
            # Role tokens
            total += self.count_tokens(msg.role)
        # Add buffer for formatting
        total += 2
        return total
    
    async def build_context(
        self,
        system_prompt: str,
        skills: List[Skill],
        retrieved_memories: List[Memory],
        conversation_history: List[Message],
        current_request: str,
        agent_state: Dict
    ) -> List[Message]:
        """
        Build optimized context within budget constraints.
        """
        context = []
        used_tokens = 0
        
        # 1. System prompt (highest priority)
        system_content = self._build_system_content(
            system_prompt, agent_state
        )
        system_tokens = self.count_tokens(system_content)
        if system_tokens > self.budget.system_tokens:
            system_content = await self._compress_system_prompt(system_content)
            system_tokens = self.count_tokens(system_content)
        
        context.append(Message(role='system', content=system_content))
        used_tokens += system_tokens + 4
        
        # 2. Active skills (dynamic loading)
        skill_content = self._build_skills_content(skills)
        skill_tokens = self.count_tokens(skill_content)
        if used_tokens + skill_tokens > self.budget.system_tokens + self.budget.skills_tokens:
            skill_content = self._select_critical_skills(skills)
            skill_tokens = self.count_tokens(skill_content)
        
        if skill_content:
            context.append(Message(role='system', content=f"Available skills: {skill_content}"))
            used_tokens += skill_tokens + 4
        
        # 3. Retrieved memories
        memory_budget = self.budget.memory_tokens
        memory_content = self._build_memory_content(retrieved_memories, memory_budget)
        
        if memory_content:
            context.append(Message(role='system', content=f"Relevant context: {memory_content}"))
            used_tokens += self.count_tokens(memory_content) + 4
        
        # 4. Conversation history (with summarization if needed)
        history_budget = self.budget.history_tokens
        history_tokens = self.count_message_tokens(conversation_history)
        
        if history_tokens > history_budget:
            # Compress history
            compressed_history = await self._compress_history(
                conversation_history, history_budget
            )
            context.extend(compressed_history)
        else:
            context.extend(conversation_history)
        
        used_tokens += history_tokens
        
        # 5. Current request
        context.append(Message(role='user', content=current_request))
        
        # Check if we're approaching limits
        total_tokens = self.count_message_tokens(context)
        utilization = total_tokens / self.budget.total
        
        if utilization > self.FLUSH_THRESHOLD:
            # Trigger memory flush before compaction
            await self._trigger_memory_flush(context)
        
        return context
    
    async def _compress_history(
        self,
        history: List[Message],
        budget: int
    ) -> List[Message]:
        """
        Compress conversation history to fit budget.
        
        Strategy:
        - Keep last N exchanges verbatim
        - Summarize older exchanges
        - Preserve critical messages
        """
        # Keep last 6 messages (3 exchanges)
        recent = history[-6:]
        recent_tokens = self.count_message_tokens(recent)
        
        if recent_tokens >= budget:
            # Even recent is too much, compress further
            return await self._aggressive_compress(recent, budget)
        
        # Summarize older messages
        older = history[:-6]
        if older:
            summary_budget = budget - recent_tokens
            summary = await self._summarize_exchange(older, summary_budget)
            
            compressed = [
                Message(role='system', content=f"Earlier conversation: {summary}")
            ] + recent
        else:
            compressed = recent
        
        return compressed
    
    async def _trigger_memory_flush(self, context: List[Message]) -> None:
        """
        Trigger agent to write important memories before compaction.
        This is the "pre-compaction flush" from OpenClaw.
        """
        flush_prompt = """
        The conversation is approaching the context limit. 
        Before older messages are summarized, please identify and store any 
        important information, decisions, or user preferences that should be 
        remembered for future sessions.
        
        Write these to MEMORY.md using the memory_write tool.
        Reply with NO_REPLY if there's nothing important to store.
        """
        
        # This would be sent as a system message to trigger the flush
        # The agent would then use memory tools to persist information
        pass
```

### 6.3 Pre-Compaction Memory Flush

```python
class PreCompactionFlush:
    """
    Implements OpenClaw's automatic memory flush before context compaction.
    """
    
    def __init__(
        self,
        context_manager: ContextWindowManager,
        memory_tools: MemoryTools,
        llm_client
    ):
        self.context = context_manager
        self.memory_tools = memory_tools
        self.llm = llm_client
        
        # Configuration
        self.soft_threshold = 4000  # tokens before limit
        self.reserve_floor = 20000  # minimum reserve
        self.flushed_this_cycle = False
    
    def should_flush(self, current_tokens: int, context_window: int) -> bool:
        """
        Determine if memory flush should be triggered.
        
        Formula: current >= context_window - reserve_floor - soft_threshold
        """
        if self.flushed_this_cycle:
            return False
        
        threshold = context_window - self.reserve_floor - self.soft_threshold
        return current_tokens >= threshold
    
    async def execute_flush(self, conversation: List[Message]) -> FlushResult:
        """
        Execute pre-compaction memory flush.
        """
        # Generate flush prompt
        flush_prompt = self._build_flush_prompt(conversation)
        
        # Get agent to identify important memories
        response = await self.llm.generate(
            messages=conversation + [Message(role='system', content=flush_prompt)]
        )
        
        # Parse response for memory write requests
        if "NO_REPLY" in response or not response.strip():
            return FlushResult(written=False, reason="nothing_to_store")
        
        # Extract and write memories
        memories = self._extract_memories(response)
        written = []
        
        for memory in memories:
            try:
                await self.memory_tools.write_memory(
                    content=memory.content,
                    category=memory.category,
                    importance=memory.importance
                )
                written.append(memory)
            except Exception as e:
                logger.error(f"Failed to write memory: {e}")
        
        self.flushed_this_cycle = True
        
        return FlushResult(
            written=len(written) > 0,
            memories_written=written,
            reason="success" if written else "extraction_failed"
        )
    
    def _build_flush_prompt(self, conversation: List[Message]) -> str:
        """Build the pre-compaction flush system prompt."""
        return """
        [SYSTEM: Session nearing compaction]
        
        The conversation context is approaching its limit. Important information
        from earlier in this conversation may soon be lost.
        
        Please review the conversation and identify:
        1. Key decisions made
        2. User preferences expressed
        3. Important facts or context
        4. Action items or todos
        5. Any information that would be valuable in future sessions
        
        Write these memories using the memory_write tool.
        Format each memory as:
        - Category: [decision|preference|fact|action|pattern]
        - Content: Clear, concise statement
        - Importance: 0.0-1.0
        
        If there's nothing important to store, reply with exactly: NO_REPLY
        """
    
    def reset_cycle(self):
        """Reset flush tracking for new compaction cycle."""
        self.flushed_this_cycle = False
```

---

## 7. Memory Retrieval & Relevance Scoring

### 7.1 Multi-Stage Retrieval Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE MEMORY RETRIEVAL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query: "What's the status of Project Alpha?"                         │
│                                                                              │
│   Stage 1: Query Understanding                                               │
│   ├── Extract entities: ["Project Alpha", "status"]                         │
│   ├── Identify intent: information_retrieval                                │
│   └── Expand query: ["Project Alpha status", "Alpha progress",              │
│                      "Alpha updates", "Alpha development"]                  │
│                                                                              │
│   Stage 2: Parallel Retrieval                                                │
│   ├── Vector Search ──→ [mem_001, mem_003, mem_007]                         │
│   ├── BM25 Search ────→ [mem_003, mem_001, mem_012]                         │
│   ├── Temporal Filter ─→ [mem_001, mem_003] (last 7 days)                   │
│   └── Type Filter ────→ [mem_001, mem_003] (semantic + episodic)            │
│                                                                              │
│   Stage 3: Fusion & Ranking                                                  │
│   ├── RRF Combine ────→ [mem_001: 0.92, mem_003: 0.88, mem_007: 0.65]       │
│   ├── Re-rank (LLM) ──→ [mem_001: 0.95, mem_003: 0.82]                      │
│   └── Diversity Boost ─→ [mem_001, mem_003, mem_015]                        │
│                                                                              │
│   Stage 4: Context Assembly                                                  │
│   ├── Fetch full content                                                     │
│   ├── Apply token budget                                                     │
│   └── Format for LLM                                                         │
│                                                                              │
│   Output: ["Project Alpha: Refactored error handling...",                   │
│            "Project Alpha: 2 commits pushed today..."]                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Relevance Scoring Algorithm

```python
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta
import numpy as np

@dataclass
class ScoredMemory:
    """Memory with computed relevance scores."""
    memory_id: str
    content: str
    semantic_score: float
    lexical_score: float
    temporal_score: float
    importance_score: float
    access_score: float
    combined_score: float
    metadata: dict

class RelevanceScorer:
    """
    Multi-factor relevance scoring for memory retrieval.
    """
    
    # Weight configuration
    WEIGHTS = {
        'semantic': 0.30,
        'lexical': 0.20,
        'temporal': 0.20,
        'importance': 0.15,
        'access': 0.15
    }
    
    # Temporal decay parameters
    RECENCY_HALF_LIFE_DAYS = 7  # Score halves every week
    
    def __init__(self):
        self.weights = self.WEIGHTS
    
    def score(
        self,
        memories: List[Memory],
        query: str,
        query_embedding: np.ndarray,
        current_time: datetime = None
    ) -> List[ScoredMemory]:
        """
        Calculate comprehensive relevance scores.
        """
        if current_time is None:
            current_time = datetime.now()
        
        scored = []
        for memory in memories:
            scores = self._calculate_scores(memory, query, query_embedding, current_time)
            
            # Combined weighted score
            combined = sum(
                scores[key] * self.weights[key]
                for key in self.weights.keys()
            )
            
            scored.append(ScoredMemory(
                memory_id=memory.id,
                content=memory.content,
                semantic_score=scores['semantic'],
                lexical_score=scores['lexical'],
                temporal_score=scores['temporal'],
                importance_score=scores['importance'],
                access_score=scores['access'],
                combined_score=combined,
                metadata=memory.metadata
            ))
        
        # Sort by combined score
        return sorted(scored, key=lambda x: x.combined_score, reverse=True)
    
    def _calculate_scores(
        self,
        memory: Memory,
        query: str,
        query_embedding: np.ndarray,
        current_time: datetime
    ) -> dict:
        """Calculate individual score components."""
        
        # Semantic similarity (cosine)
        semantic = self._cosine_similarity(query_embedding, memory.embedding)
        
        # Lexical match (BM25-inspired)
        lexical = self._lexical_score(query, memory.content)
        
        # Temporal relevance (recency)
        temporal = self._temporal_score(memory.timestamp, current_time)
        
        # Stored importance
        importance = memory.importance_score
        
        # Access frequency (learned preference)
        access = self._access_score(memory)
        
        return {
            'semantic': semantic,
            'lexical': lexical,
            'temporal': temporal,
            'importance': importance,
            'access': access
        }
    
    def _temporal_score(
        self,
        memory_time: datetime,
        current_time: datetime
    ) -> float:
        """
        Calculate temporal relevance with exponential decay.
        
        score = 2^(-age_days / half_life)
        """
        age = (current_time - memory_time).total_seconds() / 86400
        score = 2 ** (-age / self.RECENCY_HALF_LIFE_DAYS)
        return min(1.0, max(0.1, score))
    
    def _access_score(self, memory: Memory) -> float:
        """
        Score based on access frequency and recency.
        
        Higher score for frequently and recently accessed memories.
        """
        if not memory.access_count:
            return 0.5
        
        # Base score from access count (diminishing returns)
        count_score = min(1.0, np.log1p(memory.access_count) / 3)
        
        # Boost for recent access
        if memory.last_accessed:
            days_since = (datetime.now() - memory.last_accessed).days
            recency_boost = max(0, 1 - (days_since / 30))  # Decay over 30 days
        else:
            recency_boost = 0
        
        return 0.5 + (count_score * 0.3) + (recency_boost * 0.2)
    
    def _lexical_score(self, query: str, content: str) -> float:
        """
        Calculate lexical match score.
        
        Uses term frequency and exact phrase matching.
        """
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_terms & content_terms)
        union = len(query_terms | content_terms)
        
        jaccard = intersection / union if union > 0 else 0
        
        # Exact phrase bonus
        phrase_bonus = 0.2 if query.lower() in content.lower() else 0
        
        return min(1.0, jaccard + phrase_bonus)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
```

### 7.3 Query Expansion & Entity Extraction

```python
class QueryProcessor:
    """
    Processes and expands queries for better retrieval.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def process(self, query: str) -> ProcessedQuery:
        """
        Process query for optimal retrieval.
        """
        # Extract entities
        entities = await self._extract_entities(query)
        
        # Determine intent
        intent = await self._classify_intent(query)
        
        # Expand query
        expansions = await self._expand_query(query, entities)
        
        # Generate embedding
        embedding = await self._generate_embedding(query)
        
        return ProcessedQuery(
            original=query,
            entities=entities,
            intent=intent,
            expansions=expansions,
            embedding=embedding,
            filters=self._derive_filters(entities, intent)
        )
    
    async def _extract_entities(self, query: str) -> List[Entity]:
        """
        Extract named entities from query.
        """
        prompt = f"""
        Extract named entities from this query:
        "{query}"
        
        Return JSON:
        {{
            "entities": [
                {{
                    "text": "entity text",
                    "type": "person|project|date|organization|location|other",
                    "normalized": "normalized form"
                }}
            ]
        }}
        """
        
        response = await self.llm.generate(
            prompt,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response)
        return [Entity(**e) for e in data['entities']]
    
    async def _expand_query(self, query: str, entities: List[Entity]) -> List[str]:
        """
        Generate query variations for better recall.
        """
        prompt = f"""
        Generate search query variations for:
        "{query}"
        
        Entities found: {[e.text for e in entities]}
        
        Create 3-5 variations that capture different ways to express this query.
        Include synonyms, rephrasings, and related terms.
        
        Return as JSON array of strings.
        """
        
        response = await self.llm.generate(
            prompt,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response)
        expansions = data.get('variations', [])
        
        # Always include original
        return [query] + expansions
```

---

## 8. Memory Update & Write-Back Strategies

### 8.1 Write Strategies

```python
from enum import Enum, auto
from typing import Optional

class WriteStrategy(Enum):
    """Strategies for writing memories."""
    IMMEDIATE = auto()      # Write immediately (critical info)
    BATCHED = auto()        # Batch writes (normal info)
    DEFERRED = auto()       # Write on idle (low priority)
    CONDITIONAL = auto()    # Write based on conditions

class MemoryWriteManager:
    """
    Manages memory write operations with different strategies.
    """
    
    def __init__(self, memory_store, config: dict):
        self.store = memory_store
        self.config = config
        self.batch_queue = []
        self.batch_size = config.get('batch_size', 10)
        self.batch_interval = config.get('batch_interval_seconds', 30)
    
    async def write(
        self,
        content: str,
        category: str,
        importance: float = 0.5,
        strategy: WriteStrategy = WriteStrategy.BATCHED,
        metadata: Optional[dict] = None
    ) -> WriteResult:
        """
        Write memory with appropriate strategy.
        """
        if strategy == WriteStrategy.IMMEDIATE:
            return await self._write_immediate(content, category, importance, metadata)
        
        elif strategy == WriteStrategy.BATCHED:
            return await self._write_batched(content, category, importance, metadata)
        
        elif strategy == WriteStrategy.DEFERRED:
            return await self._write_deferred(content, category, importance, metadata)
        
        elif strategy == WriteStrategy.CONDITIONAL:
            return await self._write_conditional(content, category, importance, metadata)
    
    async def _write_immediate(
        self,
        content: str,
        category: str,
        importance: float,
        metadata: Optional[dict]
    ) -> WriteResult:
        """Write immediately to both file and vector store."""
        # Write to markdown file
        file_result = await self._append_to_memory_md(content, category)
        
        # Index in vector store
        vector_result = await self._index_in_vector_store(
            content, category, importance, metadata
        )
        
        return WriteResult(
            success=file_result and vector_result,
            memory_id=vector_result.memory_id if vector_result else None,
            latency_ms=vector_result.latency_ms if vector_result else 0
        )
    
    async def _write_batched(
        self,
        content: str,
        category: str,
        importance: float,
        metadata: Optional[dict]
    ) -> WriteResult:
        """Add to batch queue, flush when full or interval expires."""
        self.batch_queue.append({
            'content': content,
            'category': category,
            'importance': importance,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
        
        # Flush if batch is full
        if len(self.batch_queue) >= self.batch_size:
            return await self._flush_batch()
        
        return WriteResult(success=True, batched=True, queue_size=len(self.batch_queue))
    
    async def _flush_batch(self) -> WriteResult:
        """Flush batched writes to storage."""
        if not self.batch_queue:
            return WriteResult(success=True, message="Empty batch")
        
        # Generate embeddings in batch
        contents = [item['content'] for item in self.batch_queue]
        embeddings = await self._embed_batch(contents)
        
        # Write to file
        file_result = await self._append_batch_to_memory_md(self.batch_queue)
        
        # Index in vector store
        vector_results = await self._index_batch_in_vector_store(
            self.batch_queue, embeddings
        )
        
        # Clear queue
        written_count = len(self.batch_queue)
        self.batch_queue.clear()
        
        return WriteResult(
            success=file_result and all(vector_results),
            written_count=written_count
        )
```

### 8.2 Conflict Resolution

```python
class MemoryConflictResolver:
    """
    Resolves conflicts when updating existing memories.
    """
    
    async def resolve(
        self,
        existing: Memory,
        new_content: str,
        new_importance: float
    ) -> ResolutionAction:
        """
        Determine how to handle conflicting memory update.
        """
        # Check if memories are semantically similar
        similarity = await self._semantic_similarity(
            existing.content, new_content
        )
        
        if similarity > 0.9:
            # Very similar - merge/update
            return await self._merge_memories(existing, new_content, new_importance)
        
        elif similarity > 0.7:
            # Related but different - keep both with relationship
            return await self._create_related_memory(existing, new_content, new_importance)
        
        else:
            # Different - store as separate memory
            return ResolutionAction(
                action='create_new',
                existing_id=existing.id,
                new_content=new_content
            )
    
    async def _merge_memories(
        self,
        existing: Memory,
        new_content: str,
        new_importance: float
    ) -> ResolutionAction:
        """
        Merge two similar memories into one.
        """
        # Use LLM to synthesize merged content
        merged = await self._synthesize_merge(existing.content, new_content)
        
        # Update importance (weighted average)
        merged_importance = max(existing.importance_score, new_importance)
        
        return ResolutionAction(
            action='update',
            existing_id=existing.id,
            new_content=merged,
            new_importance=merged_importance
        )
```

---

## 9. Implementation Code Patterns

### 9.1 Memory Manager Class

```python
class MemoryManager:
    """
    Central memory management for OpenClaw agent.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db = self._init_database()
        self.embedder = EmbeddingProvider(config.embedding)
        self.chunker = MarkdownChunker()
        self.searcher = HybridMemorySearcher(self.db, self.embedder)
        self.consolidator = MemoryConsolidator(self, config.llm)
        self.context_manager = ContextWindowManager(config.llm_model)
        self.write_manager = MemoryWriteManager(self, config.write)
        
        # File watcher for real-time sync
        self.watcher = FileSystemWatcher(
            paths=[config.memory_dir],
            on_change=self._on_file_change
        )
    
    async def initialize(self):
        """Initialize memory system on agent startup."""
        # Load today's and yesterday's logs
        await self._load_recent_logs(days=2)
        
        # Index any unindexed files
        await self._sync_files_to_index()
        
        # Start file watcher
        self.watcher.start()
        
        logger.info("Memory system initialized")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search across all memory stores.
        """
        return await self.searcher.search(
            query=query,
            max_results=max_results,
            memory_types=memory_types
        )
    
    async def remember(
        self,
        content: str,
        category: str = "semantic",
        importance: float = 0.5,
        immediate: bool = False
    ) -> WriteResult:
        """
        Store a new memory.
        """
        strategy = WriteStrategy.IMMEDIATE if immediate else WriteStrategy.BATCHED
        
        return await self.write_manager.write(
            content=content,
            category=category,
            importance=importance,
            strategy=strategy
        )
    
    async def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 50000
    ) -> str:
        """
        Retrieve relevant context for a user query.
        """
        # Search for relevant memories
        results = await self.search(query, max_results=20)
        
        # Format within token budget
        context_parts = []
        used_tokens = 0
        
        for result in results:
            content = f"[{result.source_file}:{result.line_start}] {result.content}\n"
            tokens = self.context_manager.count_tokens(content)
            
            if used_tokens + tokens > max_tokens:
                break
            
            context_parts.append(content)
            used_tokens += tokens
            
            # Update access metrics
            await self._record_access(result.memory_id, query, result.combined_score)
        
        return "\n".join(context_parts)
```

### 9.2 Agent Loop Integration

```python
class AgenticLoop:
    """
    One of 15 hardcoded agentic loops with memory integration.
    """
    
    def __init__(self, memory: MemoryManager, tools: ToolRegistry):
        self.memory = memory
        self.tools = tools
    
    async def execute(self, user_input: str, context: dict) -> AgentResponse:
        """
        Execute agent loop with full memory integration.
        """
        # 1. Retrieve relevant context
        memory_context = await self.memory.get_context_for_query(user_input)
        
        # 2. Build full context
        full_context = await self.memory.context_manager.build_context(
            system_prompt=self.get_system_prompt(),
            skills=self.tools.get_active_skills(),
            retrieved_memories=memory_context,
            conversation_history=context.get('history', []),
            current_request=user_input,
            agent_state=context.get('state', {})
        )
        
        # 3. Execute LLM
        response = await self.llm.generate(full_context)
        
        # 4. Extract and store any new memories
        await self._extract_and_store_memories(response, user_input)
        
        # 5. Check for pre-compaction flush
        current_tokens = self.memory.context_manager.count_message_tokens(
            full_context + [Message(role='assistant', content=response)]
        )
        
        if self.memory.context_manager.should_flush(current_tokens, 200000):
            await self.memory.trigger_pre_compaction_flush(full_context)
        
        return AgentResponse(content=response, actions=self._parse_actions(response))
```

---

## 10. Performance Specifications

### 10.1 Target Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Memory Search Latency | < 100ms | For 10K chunks |
| Embedding Generation | ~1000 tok/s | OpenAI batch API |
| Local Embedding | ~50 tok/s | node-llama-cpp on M1 |
| Context Build Time | < 500ms | Full context assembly |
| Index Size | ~5KB/1K tokens | With 1536-dim embeddings |
| Daily Log Write | < 10ms | Append-only |
| Vector DB Query | < 50ms | KNN search |

### 10.2 Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB |
| Disk (base) | 1GB | 2GB |
| Disk (with local embeddings) | 2GB | 4GB |
| SQLite Cache | 256MB | 512MB |

### 10.3 Scaling Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCALING THRESHOLDS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Daily Logs:                                                                 │
│  ├── Keep last 30 days in active index                                       │
│  ├── Archive to sessions/ after 30 days                                      │
│  └── Consolidate to MEMORY.md after 7 days                                   │
│                                                                              │
│  Vector Index:                                                               │
│  ├── SQLite-vec: Up to 100K chunks (single file)                            │
│  ├── Consider migration to dedicated vector DB at 500K+ chunks              │
│  └── Implement sharding at 1M+ chunks                                        │
│                                                                              │
│  Session Transcripts:                                                        │
│  ├── Index all sessions for searchability                                    │
│  ├── Compress transcripts older than 90 days                                 │
│  └── Archive to cold storage after 1 year                                    │
│                                                                              │
│  Embedding Cache:                                                            │
│  ├── LRU cache with 10K entries                                              │
│  ├── Persist to SQLite for restart recovery                                  │
│  └── Clear cache when switching embedding models                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Configuration File

```yaml
# memory_config.yaml

memory:
  # Directory configuration
  base_dir: "~/.openclaw"
  daily_log_format: "memory/daily/%Y-%m-%d.md"
  
  # Chunking settings
  chunking:
    tokens: 400
    overlap: 80
    respect_headers: true
  
  # Embedding configuration
  embedding:
    provider: auto  # auto, openai, gemini, local
    model: null     # Use provider default
    dimension: 1536
    batch_size: 50
    
  # Search configuration
  search:
    default_results: 10
    min_score: 0.3
    rrf_k: 60
    
  # Context window
  context:
    model: "gpt-5.2"
    total_tokens: 200000
    system_tokens: 10000
    skills_tokens: 15000
    memory_tokens: 50000
    history_tokens: 75000
    request_tokens: 30000
    response_reserve: 20000
    
  # Consolidation settings
  consolidation:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    age_threshold_days: 7
    importance_threshold: 0.6
    max_daily_logs: 30
    
  # Write strategy
  writes:
    batch_size: 10
    batch_interval_seconds: 30
    immediate_importance_threshold: 0.8
```

---

## Appendix B: Memory Tools for Agent

```python
# Tools exposed to the agent for memory operations

MEMORY_TOOLS = [
    {
        "name": "memory_search",
        "description": "Search agent's memory for relevant information. "
                      "Use before answering questions about prior work, "
                      "decisions, dates, people, or preferences.",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "default": 10},
            "memory_types": {"type": "array", "items": {"enum": ["episodic", "semantic", "procedural"]}},
            "time_range_days": {"type": "integer"}
        }
    },
    {
        "name": "memory_read",
        "description": "Read a specific memory file or section.",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"}
        }
    },
    {
        "name": "memory_write",
        "description": "Write important information to long-term memory.",
        "parameters": {
            "content": {"type": "string", "required": True},
            "category": {"type": "string", "enum": ["decision", "preference", "fact", "action", "pattern"]},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}}
        }
    },
    {
        "name": "memory_update",
        "description": "Update an existing memory entry.",
        "parameters": {
            "memory_id": {"type": "string", "required": True},
            "content": {"type": "string"},
            "importance": {"type": "number"}
        }
    }
]
```

---

## Summary

This specification defines a comprehensive memory system architecture for a Windows 10 OpenClaw-inspired AI agent. Key features include:

1. **Multi-tier memory hierarchy** with working, short-term, episodic, semantic, and procedural memory types
2. **File-first storage** using Markdown for human-readable, version-controllable persistence
3. **Hybrid search** combining BM25 lexical search with vector semantic search via sqlite-vec
4. **Intelligent chunking** with overlap preservation for context continuity
5. **Automatic consolidation** that converts daily logs to curated long-term memory
6. **Context window management** with pre-compaction memory flush
7. **Multi-factor relevance scoring** for optimal memory retrieval
8. **Flexible write strategies** for different importance levels

The system is designed for 24/7 autonomous operation with GPT-5.2, supporting Gmail, browser control, TTS/STT, Twilio integration, and full Windows 10 system access.
