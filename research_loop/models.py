"""
Data Models for Research Loop
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ResearchDepth(str, Enum):
    """Research depth levels"""
    SURFACE = "surface"
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class TriggerType(str, Enum):
    """Research trigger types"""
    SCHEDULED = "scheduled"
    EVENT = "event"
    CURIOSITY = "curiosity"
    USER = "user"


class TaskStatus(str, Enum):
    """Research task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchTask(BaseModel):
    """Research task definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: TriggerType
    trigger_source: str
    topic: str
    context: Optional[str] = None
    rationale: Optional[str] = None
    depth: ResearchDepth = ResearchDepth.MEDIUM
    sources: List[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    immediate: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    """Search query definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    target_engines: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_type: str = "general"  # definition, how_to, comparison, etc.
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=datetime.now)
    parent_task_id: Optional[str] = None


class DiscoveredSource(BaseModel):
    """Discovered source from search"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    title: str
    snippet: Optional[str] = None
    engine: str
    query: str
    rank: int
    timestamp: datetime = Field(default_factory=datetime.now)
    content_type: Optional[str] = None
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0)


class RankedSource(BaseModel):
    """Source with ranking information"""
    source: DiscoveredSource
    scores: Dict[str, float] = Field(default_factory=dict)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Media(BaseModel):
    """Media item extracted from content"""
    type: str  # image, video, audio, document
    url: str
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedContent(BaseModel):
    """Extracted content from a web page"""
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    content: str = ""
    html_content: Optional[str] = None
    summary: Optional[str] = None
    reading_time: Optional[int] = None
    word_count: Optional[int] = None
    links: List[str] = Field(default_factory=list)
    content_type: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    media: List[Media] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    abstract: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)


class Entity(BaseModel):
    """Named entity extracted from text"""
    name: str
    type: str
    context: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    mentions: int = 1
    normalized_name: Optional[str] = None


class Fact(BaseModel):
    """Extracted fact"""
    statement: str
    category: str  # definition, statistic, relationship, process, attribute
    confidence: str  # high, medium, low
    temporal_context: Optional[str] = None
    source_attribution: Optional[str] = None
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """Relationship between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context: Optional[str] = None


class Claim(BaseModel):
    """Verifiable claim"""
    text: str
    claim_type: str  # fact, opinion, prediction, statistic
    confidence: str  # high, medium, low
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)


class Statistic(BaseModel):
    """Extracted statistic"""
    value: str
    metric: str
    context: Optional[str] = None
    source: Optional[str] = None
    date: Optional[datetime] = None


class Quote(BaseModel):
    """Extracted quote"""
    text: str
    author: Optional[str] = None
    source: Optional[str] = None
    context: Optional[str] = None


class CrawledContent(BaseModel):
    """Content after crawling"""
    source: DiscoveredSource
    content: ExtractedContent
    crawl_timestamp: datetime = Field(default_factory=datetime.now)
    crawl_depth: int = 0
    linked_content: List['CrawledContent'] = Field(default_factory=list)


class ExtractedInformation(BaseModel):
    """All information extracted from a source"""
    source: DiscoveredSource
    entities: List[Entity] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    statistics: List[Statistic] = Field(default_factory=list)
    quotes: List[Quote] = Field(default_factory=list)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)


class ResolvedFact(BaseModel):
    """Fact after conflict resolution"""
    statement: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    resolution_method: str
    alternative_statements: List[str] = Field(default_factory=list)


class ConsensusArea(BaseModel):
    """Area where sources agree"""
    topic: str
    agreed_facts: List[str] = Field(default_factory=list)
    supporting_sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DisagreementArea(BaseModel):
    """Area where sources disagree"""
    topic: str
    conflicting_positions: List[str] = Field(default_factory=list)
    sources_by_position: Dict[str, List[str]] = Field(default_factory=dict)
    analysis: Optional[str] = None


class Synthesis(BaseModel):
    """Synthesized research output"""
    content: str
    topic: str
    generated_at: datetime = Field(default_factory=datetime.now)
    key_points: List[str] = Field(default_factory=list)
    implications: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class Summary(BaseModel):
    """Summary at a specific detail level"""
    level: str  # ultra_brief, brief, standard, detailed, comprehensive
    content: str
    token_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)


class SynthesisResult(BaseModel):
    """Complete synthesis result"""
    synthesis: Synthesis
    summaries: Dict[str, Summary] = Field(default_factory=dict)
    consensus_areas: List[ConsensusArea] = Field(default_factory=list)
    disagreement_areas: List[DisagreementArea] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    sources_used: int = 0
    facts_synthesized: int = 0
    synthesis_timestamp: datetime = Field(default_factory=datetime.now)


class Citation(BaseModel):
    """Citation record"""
    id: str
    source_url: str
    source_title: str
    access_date: datetime
    usage_context: Dict[str, Any] = Field(default_factory=dict)
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    citation_format: Dict[str, str] = Field(default_factory=dict)
    verification_status: Optional[str] = None
    last_verified: Optional[datetime] = None


class KnowledgeGap(BaseModel):
    """Identified knowledge gap"""
    topic: str
    description: str
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_depth: ResearchDepth = ResearchDepth.MEDIUM
    related_topics: List[str] = Field(default_factory=list)


class RelatedTopic(BaseModel):
    """Related topic for exploration"""
    name: str
    connection: str
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TrendingTopic(BaseModel):
    """Trending topic in interest area"""
    topic: str
    significance: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str


class SerendipitousDiscovery(BaseModel):
    """Unexpected connection discovered"""
    topic: str
    connection: str
    interest_score: float = Field(default=0.5, ge=0.0, le=1.0)
    discovered_via: str


class CrawlConfig(BaseModel):
    """Configuration for web crawling"""
    extract_type: str = "general"
    follow_links: bool = False
    max_depth: int = 1
    current_depth: int = 0
    max_links: int = 5
    max_concurrent: int = 3
    timeout_seconds: int = 30
    respect_robots_txt: bool = True
    topic: Optional[str] = None

    @classmethod
    def from_research_config(cls, config: 'ResearchConfig') -> 'CrawlConfig':
        """Create crawl config from research config"""
        return cls(
            max_depth=config.max_depth,
            max_concurrent=5,
            timeout_seconds=30,
            respect_robots_txt=True
        )


class ResearchConfig(BaseModel):
    """Configuration for research execution"""
    depth: ResearchDepth = ResearchDepth.MEDIUM
    max_sources: int = 10
    max_depth: int = 2
    max_pages_per_source: int = 3
    synthesis_detail: str = "detailed"
    time_budget: timedelta = Field(default_factory=lambda: timedelta(minutes=10))
    query_variants: int = 5
    start_time: datetime = Field(default_factory=datetime.now)
    results_per_engine: int = 10
    engines: List[str] = Field(default_factory=lambda: ["google", "bing"])


class ExtractionGoal(BaseModel):
    """Goal for information extraction"""
    topic: str
    description: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class UsageContext(BaseModel):
    """Context of source usage"""
    research_task: Optional[str] = None
    extraction_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class VerificationResult(BaseModel):
    """Source verification result"""
    valid: bool
    reason: Optional[str] = None
    last_verified: Optional[datetime] = None
    http_status: Optional[int] = None


class ProgressStatus(BaseModel):
    """Research progress status"""
    should_continue: bool
    reason: Optional[str] = None
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    sources_collected: int = 0
    average_source_quality: float = Field(default=0.0, ge=0.0, le=1.0)


class AdaptationDecision(BaseModel):
    """Decision for adaptive research"""
    should_broaden: bool = False
    should_deepen: bool = False
    new_queries: List[SearchQuery] = Field(default_factory=list)
    additional_topics: List[str] = Field(default_factory=list)
    explore_related: bool = False
    related_topics: List[str] = Field(default_factory=list)
    conflict_resolution_needed: bool = False


class ConsolidationResult(BaseModel):
    """Result of knowledge consolidation"""
    entries_created: int = 0
    entries_updated: int = 0
    cross_references_added: int = 0
    knowledge_graph_updates: int = 0
    consolidation_timestamp: datetime = Field(default_factory=datetime.now)


class MergedKnowledge(BaseModel):
    """Knowledge merged with existing"""
    topic: str
    summary: str
    key_facts: List[Fact] = Field(default_factory=list)
    detailed_content: str = ""
    sources: List[DiscoveredSource] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    new_entries: List[str] = Field(default_factory=list)
    updated_entries: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)


class GraphUpdate(BaseModel):
    """Knowledge graph update"""
    nodes_added: List[str] = Field(default_factory=list)
    nodes_updated: List[str] = Field(default_factory=list)
    edges_added: List[str] = Field(default_factory=list)
    edges_updated: List[str] = Field(default_factory=list)


class ErrorContext(BaseModel):
    """Context for error handling"""
    task_id: str
    operation: str
    retry_count: int = 0
    error_history: List[str] = Field(default_factory=list)


class ErrorRecovery(BaseModel):
    """Error recovery decision"""
    should_retry: bool = False
    should_fail: bool = False
    retry_delay: Optional[int] = None
    fallback_result: Optional[Any] = None


class SafetyCheck(BaseModel):
    """URL safety check result"""
    safe: bool
    reason: Optional[str] = None
    reputation_score: Optional[float] = None


class ResearchContext(BaseModel):
    """Context for research operations"""
    user_interests: List[str] = Field(default_factory=list)
    recent_topics: List[str] = Field(default_factory=list)
    knowledge_domains: List[str] = Field(default_factory=list)
    preferred_sources: List[str] = Field(default_factory=list)
    excluded_sources: List[str] = Field(default_factory=list)
