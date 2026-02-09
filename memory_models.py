"""
Memory System Data Models for OpenClaw-Inspired AI Agent
Windows 10 Compatible Implementation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from enum import Enum, auto
import json
import hashlib
import os
import numpy as np
import yaml
from pathlib import Path


class MemoryType(Enum):
    """Types of memory in the hierarchy."""
    EPISODIC = "episodic"       # Event sequences, experiences
    SEMANTIC = "semantic"       # Facts, knowledge, preferences
    PROCEDURAL = "procedural"   # Skills, workflows
    PREFERENCE = "preference"   # User preferences


class MemoryPriority(Enum):
    """Priority levels for memory operations."""
    CRITICAL = 1.0
    HIGH = 0.8
    NORMAL = 0.5
    LOW = 0.3
    BACKGROUND = 0.1


@dataclass
class MemoryChunk:
    """A chunk of memory content with metadata."""
    content: str
    line_start: int
    line_end: int
    chunk_index: int
    total_chunks: int
    content_hash: str = field(default="")
    embedding: Optional[np.ndarray] = field(default=None)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode('utf-8', errors='replace')
            ).hexdigest()
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (4 chars ≈ 1 token for English)."""
        return len(self.content) // 4
    
    @property
    def is_first_chunk(self) -> bool:
        return self.chunk_index == 0
    
    @property
    def is_last_chunk(self) -> bool:
        return self.chunk_index == self.total_chunks - 1


@dataclass  
class MemoryEntry:
    """A complete memory entry in the system."""
    id: str
    type: MemoryType
    content: str
    source_file: Path
    created_at: datetime
    
    # Optional fields
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    updated_at: Optional[datetime] = None
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    chunk_index: int = 0
    total_chunks: int = 1
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'source_file': str(self.source_file),
            'line_start': self.line_start,
            'line_end': self.line_end,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'importance_score': self.importance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'embedding_id': self.embedding_id,
            'metadata': self.metadata,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            type=MemoryType(data['type']),
            content=data['content'],
            source_file=Path(data['source_file']),
            line_start=data.get('line_start'),
            line_end=data.get('line_end'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            importance_score=data.get('importance_score', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            chunk_index=data.get('chunk_index', 0),
            total_chunks=data.get('total_chunks', 1),
            embedding_id=data.get('embedding_id'),
            metadata=data.get('metadata', {}),
            tags=set(data.get('tags', []))
        )
    
    def record_access(self):
        """Record an access to this memory."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class SearchResult:
    """Result from a memory search operation."""
    memory_id: str
    content: str
    source_file: Path
    line_start: Optional[int]
    line_end: Optional[int]
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    temporal_score: float = 0.0
    importance_score: float = 0.0
    access_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'source_file': str(self.source_file),
            'line_start': self.line_start,
            'line_end': self.line_end,
            'semantic_score': round(self.semantic_score, 4),
            'lexical_score': round(self.lexical_score, 4),
            'temporal_score': round(self.temporal_score, 4),
            'importance_score': round(self.importance_score, 4),
            'access_score': round(self.access_score, 4),
            'combined_score': round(self.combined_score, 4),
            'metadata': self.metadata
        }


@dataclass
class ProcessedQuery:
    """A processed and expanded search query."""
    original: str
    entities: List['Entity']
    intent: str
    expansions: List[str]
    embedding: np.ndarray
    filters: Dict[str, Any]
    
    def all_queries(self) -> List[str]:
        """Get all query variations including original."""
        return [self.original] + self.expansions


@dataclass
class Entity:
    """Named entity extracted from query."""
    text: str
    type: str  # person, project, date, organization, location, other
    normalized: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'text': self.text,
            'type': self.type,
            'normalized': self.normalized
        }


@dataclass
class ContextBudget:
    """Token budget allocation for context window."""
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
    
    def utilization(self, used_tokens: int) -> float:
        """Calculate utilization percentage."""
        return used_tokens / self.total


@dataclass
class DailyLogEntry:
    """An entry in a daily log file."""
    timestamp: datetime
    event_type: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        time_str = self.timestamp.strftime("%H:%M")
        md = f"### {time_str} - {self.title}\n\n"
        md += f"**Type**: {self.event_type}\n\n"
        md += f"{self.content}\n\n"
        
        if self.metadata:
            md += "**Metadata**:\n"
            for key, value in self.metadata.items():
                md += f"- {key}: {value}\n"
            md += "\n"
        
        return md
    
    @classmethod
    def from_markdown_section(cls, section: str) -> Optional['DailyLogEntry']:
        """Parse from markdown section."""
        lines = section.strip().split('\n')
        if not lines:
            return None
        
        # Parse header line
        header = lines[0]
        if not header.startswith('###'):
            return None
        
        # Extract time and title
        header_parts = header.replace('###', '').strip().split(' - ', 1)
        if len(header_parts) != 2:
            return None
        
        time_str, title = header_parts
        
        # Parse content
        content_lines = []
        metadata = {}
        in_metadata = False
        event_type = 'general'

        for line in lines[1:]:
            if line.startswith('**Type**:'):
                event_type = line.replace('**Type**:', '').strip()
            elif line.startswith('**Metadata**:'):
                in_metadata = True
            elif in_metadata and line.startswith('- '):
                parts = line[2:].split(': ', 1)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
            elif line.strip() and not line.startswith('**'):
                content_lines.append(line)

        # Create timestamp from time (assume today)
        today = datetime.now().date()
        hour, minute = map(int, time_str.split(':'))
        timestamp = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))

        return cls(
            timestamp=timestamp,
            event_type=event_type,
            title=title,
            content='\n'.join(content_lines).strip(),
            metadata=metadata
        )


@dataclass
class ConsolidationReport:
    """Report from memory consolidation process."""
    events_processed: int = 0
    facts_extracted: int = 0
    files_processed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'events_processed': self.events_processed,
            'facts_extracted': self.facts_extracted,
            'files_processed': self.files_processed,
            'errors': self.errors,
            'duration_seconds': self.duration_seconds
        }


@dataclass
class SemanticSummary:
    """Summary extracted from events during consolidation."""
    facts: List['Fact']
    summary: str
    source_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'facts': [f.to_dict() for f in self.facts],
            'summary': self.summary,
            'source_date': self.source_date.isoformat() if self.source_date else None
        }


@dataclass
class Fact:
    """A single fact extracted from events."""
    content: str
    category: str  # decision, preference, project, pattern, relationship
    importance: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'category': self.category,
            'importance': self.importance,
            'confidence': self.confidence
        }


@dataclass
class WriteResult:
    """Result of a memory write operation."""
    success: bool
    memory_id: Optional[str] = None
    batched: bool = False
    queue_size: int = 0
    written_count: int = 0
    latency_ms: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'memory_id': self.memory_id,
            'batched': self.batched,
            'queue_size': self.queue_size,
            'written_count': self.written_count,
            'latency_ms': self.latency_ms,
            'message': self.message
        }


@dataclass
class FlushResult:
    """Result of pre-compaction memory flush."""
    written: bool
    memories_written: List[MemoryEntry] = field(default_factory=list)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'written': self.written,
            'memories_written': len(self.memories_written),
            'reason': self.reason
        }


@dataclass
class ResolutionAction:
    """Action to resolve memory conflict."""
    action: str  # update, create_new, merge
    existing_id: Optional[str] = None
    new_content: str = ""
    new_importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'existing_id': self.existing_id,
            'new_content': self.new_content[:200] + '...' if len(self.new_content) > 200 else self.new_content,
            'new_importance': self.new_importance
        }


@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }


@dataclass
class AgentResponse:
    """Response from agent execution."""
    content: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'actions': self.actions,
            'memory_updates': self.memory_updates,
            'tool_calls': self.tool_calls,
            'metadata': self.metadata
        }


# Configuration dataclasses

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: str = "auto"  # auto, openai, gemini, local
    model_name: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 60
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    local_model_path: Optional[str] = None
    
    def get_model_name(self) -> str:
        """Get the effective model name."""
        if self.model_name:
            return self.model_name
        
        defaults = {
            'openai': 'text-embedding-3-small',
            'gemini': 'gemini-embedding-001',
            'local': 'all-MiniLM-L6-v2'
        }
        return defaults.get(self.provider, 'text-embedding-3-small')


@dataclass
class ChunkingConfig:
    """Configuration for content chunking."""
    chunk_tokens: int = 400
    overlap_tokens: int = 80
    respect_headers: bool = True
    chars_per_token: int = 4
    
    @property
    def chunk_size(self) -> int:
        return self.chunk_tokens * self.chars_per_token
    
    @property
    def overlap_size(self) -> int:
        return self.overlap_tokens * self.chars_per_token


@dataclass
class SearchConfig:
    """Configuration for memory search."""
    default_results: int = 10
    min_score: float = 0.3
    rrf_k: int = 60
    recency_half_life_days: int = 7
    
    # Weights for combined scoring
    semantic_weight: float = 0.30
    lexical_weight: float = 0.20
    temporal_weight: float = 0.20
    importance_weight: float = 0.15
    access_weight: float = 0.15


@dataclass
class WriteConfig:
    """Configuration for memory writes."""
    batch_size: int = 10
    batch_interval_seconds: int = 30
    immediate_importance_threshold: float = 0.8
    enable_caching: bool = True
    cache_size: int = 10000


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    enabled: bool = True
    schedule: str = "0 2 * * *"  # Daily at 2 AM
    age_threshold_days: int = 7
    importance_threshold: float = 0.6
    max_daily_logs: int = 30
    archive_after_days: int = 90


@dataclass
class MemoryConfig:
    """Complete memory system configuration."""
    base_dir: Path = field(default_factory=lambda: Path.home() / '.openclaw')
    llm_model: str = "gpt-5.2"
    
    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    
    @property
    def memory_dir(self) -> Path:
        return self.base_dir / 'memory'
    
    @property
    def daily_dir(self) -> Path:
        return self.memory_dir / 'daily'
    
    @property
    def sessions_dir(self) -> Path:
        return self.base_dir / 'sessions'
    
    @property
    def db_path(self) -> Path:
        return self.memory_dir / 'index.db'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'base_dir': str(self.base_dir),
            'llm_model': self.llm_model,
            'embedding': self.embedding.__dict__,
            'chunking': self.chunking.__dict__,
            'search': self.search.__dict__,
            'write': self.write.__dict__,
            'consolidation': self.consolidation.__dict__,
            'context_budget': self.context_budget.__dict__
        }

    @classmethod
    def from_yaml(cls, path: str = None) -> 'MemoryConfig':
        """Load MemoryConfig from YAML file with fallback to defaults."""
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_config.yaml')
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return cls()

        embedding_data = data.get('embedding', {})
        chunking_data = data.get('chunking', {})
        search_data = data.get('search', {})
        write_data = data.get('write', {})
        consolidation_data = data.get('consolidation', {})
        context_data = data.get('context_budget', {})

        base_dir = data.get('base_dir', '~/.openclaw')
        if base_dir.startswith('~'):
            base_dir = str(Path.home() / base_dir[2:])

        return cls(
            base_dir=Path(base_dir),
            llm_model=data.get('llm_model', 'gpt-5.2'),
            embedding=EmbeddingConfig(
                provider=embedding_data.get('provider', 'auto'),
                model_name=embedding_data.get('model_name'),
                dimension=embedding_data.get('dimension', 1536),
                batch_size=embedding_data.get('batch_size', 50),
                max_retries=embedding_data.get('max_retries', 3),
                timeout_seconds=embedding_data.get('timeout_seconds', 60),
            ),
            chunking=ChunkingConfig(
                chunk_tokens=chunking_data.get('chunk_tokens', 400),
                overlap_tokens=chunking_data.get('overlap_tokens', 80),
                respect_headers=chunking_data.get('respect_headers', True),
                chars_per_token=chunking_data.get('chars_per_token', 4),
            ),
            search=SearchConfig(
                default_results=search_data.get('default_results', 10),
                min_score=search_data.get('min_score', 0.3),
                rrf_k=search_data.get('rrf_k', 60),
                recency_half_life_days=search_data.get('recency_half_life_days', 7),
                semantic_weight=search_data.get('semantic_weight', 0.30),
                lexical_weight=search_data.get('lexical_weight', 0.20),
                temporal_weight=search_data.get('temporal_weight', 0.20),
                importance_weight=search_data.get('importance_weight', 0.15),
                access_weight=search_data.get('access_weight', 0.15),
            ),
            write=WriteConfig(
                batch_size=write_data.get('batch_size', 10),
                batch_interval_seconds=write_data.get('batch_interval_seconds', 30),
                immediate_importance_threshold=write_data.get('immediate_importance_threshold', 0.8),
                enable_caching=write_data.get('enable_caching', True),
                cache_size=write_data.get('cache_size', 10000),
            ),
            consolidation=ConsolidationConfig(
                enabled=consolidation_data.get('enabled', True),
                schedule=consolidation_data.get('schedule', '0 2 * * *'),
                age_threshold_days=consolidation_data.get('age_threshold_days', 7),
                importance_threshold=consolidation_data.get('importance_threshold', 0.6),
            ),
            context_budget=ContextBudget(
                system_tokens=context_data.get('system_tokens', 10000),
                skills_tokens=context_data.get('skills_tokens', 15000),
                memory_tokens=context_data.get('memory_tokens', 50000),
                history_tokens=context_data.get('history_tokens', 75000),
                request_tokens=context_data.get('request_tokens', 30000),
                response_reserve=context_data.get('response_reserve', 20000),
            ),
        )
