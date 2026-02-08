"""
Configuration for Research Loop
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import timedelta
import yaml
import os


class ScheduleConfig(BaseModel):
    """Scheduled research configuration"""
    cron: str
    topics: List[str]
    depth: str = "medium"
    sources: List[str] = Field(default_factory=list)
    enabled: bool = True


class TriggerConfig(BaseModel):
    """Trigger configuration"""
    enabled: bool = True
    schedules: Dict[str, ScheduleConfig] = Field(default_factory=dict)
    immediate_events: List[str] = Field(default_factory=list)
    deferred_events: List[str] = Field(default_factory=list)
    min_curiosity_score: float = Field(default=0.7, ge=0.0, le=1.0)
    max_daily_tasks: int = 10


class SearchEngineConfig(BaseModel):
    """Search engine configuration"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    rate_limit: Optional[int] = None
    priority: int = 1
    strengths: List[str] = Field(default_factory=list)


class SearchConfig(BaseModel):
    """Search configuration"""
    engines: List[str] = Field(default_factory=lambda: ["google", "bing", "duckduckgo"])
    results_per_engine: int = 10
    max_query_variants: int = 5
    request_timeout: int = 30
    retry_attempts: int = 3
    user_agent: str = "OpenClaw Research Bot 1.0"


class CrawlConfig(BaseModel):
    """Crawling configuration"""
    max_concurrent: int = 5
    timeout_seconds: int = 30
    respect_robots_txt: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5
    request_delay: float = 0.5  # seconds between requests


class SynthesisConfig(BaseModel):
    """Synthesis configuration"""
    model: str = "gpt-5.2"
    thinking_mode: str = "high"
    max_tokens: int = 4000
    temperature: float = 0.3
    summary_levels: List[str] = Field(default_factory=lambda: [
        "ultra_brief", "brief", "standard", "detailed", "comprehensive"
    ])


class MemoryConfig(BaseModel):
    """Memory configuration"""
    memory_file: str = "MEMORY.md"
    knowledge_graph: str = "knowledge_graph.json"
    citation_db: str = "citations.json"
    reliability_db: str = "source_reliability.json"
    auto_backup: bool = True
    backup_interval: str = "daily"
    backup_retention_days: int = 30
    max_entry_size: int = 10000  # characters


class DepthLevelConfig(BaseModel):
    """Configuration for a specific depth level"""
    max_sources: int
    max_depth: int
    max_pages_per_source: int
    synthesis_detail: str
    time_budget_seconds: int
    query_variants: int


class DepthConfig(BaseModel):
    """Research depth configuration"""
    surface: DepthLevelConfig = Field(default_factory=lambda: DepthLevelConfig(
        max_sources=3, max_depth=1, max_pages_per_source=1,
        synthesis_detail="brief", time_budget_seconds=60, query_variants=2
    ))
    shallow: DepthLevelConfig = Field(default_factory=lambda: DepthLevelConfig(
        max_sources=5, max_depth=1, max_pages_per_source=2,
        synthesis_detail="standard", time_budget_seconds=180, query_variants=3
    ))
    medium: DepthLevelConfig = Field(default_factory=lambda: DepthLevelConfig(
        max_sources=10, max_depth=2, max_pages_per_source=3,
        synthesis_detail="detailed", time_budget_seconds=600, query_variants=5
    ))
    deep: DepthLevelConfig = Field(default_factory=lambda: DepthLevelConfig(
        max_sources=20, max_depth=3, max_pages_per_source=5,
        synthesis_detail="comprehensive", time_budget_seconds=1800, query_variants=8
    ))
    exhaustive: DepthLevelConfig = Field(default_factory=lambda: DepthLevelConfig(
        max_sources=50, max_depth=5, max_pages_per_source=10,
        synthesis_detail="comprehensive", time_budget_seconds=3600, query_variants=15
    ))


class SecurityConfig(BaseModel):
    """Security configuration"""
    blocked_domains: List[str] = Field(default_factory=lambda: [
        "malware", "phishing", "spam", "adult", "gambling", "illegal"
    ])
    blocked_extensions: List[str] = Field(default_factory=lambda: [
        ".exe", ".dll", ".bat", ".sh", ".bin"
    ])
    respect_robots_txt: bool = True
    respect_noindex: bool = True
    anonymize_requests: bool = True
    no_personal_data_extraction: bool = True
    secure_storage: bool = True
    max_content_size: int = 10 * 1024 * 1024  # 10MB


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    file: str = "research_loop.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    max_concurrent_tasks: int = 5
    task_queue_size: int = 100
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    memory_limit_mb: int = 512


class ResearchLoopConfig(BaseModel):
    """Main research loop configuration"""
    enabled: bool = True
    poll_interval: int = 30  # seconds
    
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    crawling: CrawlConfig = Field(default_factory=CrawlConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Default schedules
    default_schedules: Dict[str, ScheduleConfig] = Field(default_factory=lambda: {
        "morning_briefing": ScheduleConfig(
            cron="0 8 * * *",
            topics=["tech_news", "market_updates", "weather"],
            depth="shallow",
            sources=["news_api", "rss_feeds"]
        ),
        "weekly_research": ScheduleConfig(
            cron="0 10 * * 1",
            topics=["emerging_tech", "industry_trends"],
            depth="deep",
            sources=["all"]
        ),
        "hourly_check": ScheduleConfig(
            cron="0 * * * *",
            topics=["alerts", "notifications"],
            depth="surface",
            sources=["priority_feeds"]
        ),
        "continuous_learning": ScheduleConfig(
            cron="*/30 * * * *",
            topics=["user_interests", "knowledge_gaps"],
            depth="adaptive",
            sources=["dynamic"]
        )
    })
    
    @classmethod
    def from_yaml(cls, path: str) -> "ResearchLoopConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "ResearchLoopConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("RESEARCH_LOOP_ENABLED"):
            config.enabled = os.getenv("RESEARCH_LOOP_ENABLED").lower() == "true"
        
        if os.getenv("RESEARCH_LOOP_POLL_INTERVAL"):
            config.poll_interval = int(os.getenv("RESEARCH_LOOP_POLL_INTERVAL"))
        
        if os.getenv("RESEARCH_MEMORY_FILE"):
            config.memory.memory_file = os.getenv("RESEARCH_MEMORY_FILE")
        
        if os.getenv("RESEARCH_LOG_LEVEL"):
            config.logging.level = os.getenv("RESEARCH_LOG_LEVEL")
        
        return config
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def get_depth_config(self, depth: str) -> DepthLevelConfig:
        """Get configuration for a specific depth level"""
        depth_map = {
            "surface": self.depth.surface,
            "shallow": self.depth.shallow,
            "medium": self.depth.medium,
            "deep": self.depth.deep,
            "exhaustive": self.depth.exhaustive
        }
        return depth_map.get(depth, self.depth.medium)


# Default configuration instance
default_config = ResearchLoopConfig()


# Query templates
QUERY_TEMPLATES = {
    "definition": {
        "template": "what is {topic} definition meaning",
        "use_case": "basic_understanding",
        "engines": ["google", "bing", "duckduckgo"]
    },
    "how_to": {
        "template": "how to {action} {topic} tutorial guide",
        "use_case": "procedural_knowledge",
        "engines": ["google", "youtube", "stackoverflow"]
    },
    "comparison": {
        "template": "{topic_a} vs {topic_b} comparison differences",
        "use_case": "comparative_analysis",
        "engines": ["google", "reddit", "quora"]
    },
    "latest": {
        "template": "{topic} latest news updates {timeframe}",
        "use_case": "current_information",
        "engines": ["google_news", "bing_news"]
    },
    "deep_dive": {
        "template": "{topic} comprehensive analysis research paper",
        "use_case": "deep_research",
        "engines": ["google_scholar", "arxiv", "semantic_scholar"]
    },
    "opinions": {
        "template": "{topic} opinions reviews experiences",
        "use_case": "sentiment_analysis",
        "engines": ["reddit", "twitter", "quora"]
    },
    "technical": {
        "template": "{topic} technical documentation specification",
        "use_case": "technical_details",
        "engines": ["github", "docs", "stackoverflow"]
    },
    "trends": {
        "template": "{topic} trends statistics data {timeframe}",
        "use_case": "trend_analysis",
        "engines": ["google_trends", "statista"]
    }
}


# Credibility domain scores
CREDIBILITY_DOMAINS = {
    # Academic
    "edu": 0.9,
    "ac.uk": 0.9,
    "arxiv.org": 0.95,
    "ieee.org": 0.95,
    "acm.org": 0.95,
    "nature.com": 0.95,
    "science.org": 0.95,
    "sciencedirect.com": 0.9,
    "springer.com": 0.9,
    "jstor.org": 0.9,
    
    # Government
    "gov": 0.9,
    "gov.uk": 0.9,
    "europa.eu": 0.9,
    "un.org": 0.9,
    "who.int": 0.95,
    
    # Tech
    "github.com": 0.85,
    "stackoverflow.com": 0.85,
    "docs.microsoft.com": 0.9,
    "developer.mozilla.org": 0.9,
    "docs.python.org": 0.9,
    "kubernetes.io": 0.9,
    "docker.com": 0.85,
    
    # News
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bbc.com": 0.8,
    "bbc.co.uk": 0.8,
    "npr.org": 0.8,
    "economist.com": 0.85,
    "wsj.com": 0.8,
    "nytimes.com": 0.75,
    "washingtonpost.com": 0.75,
    
    # Reference
    "wikipedia.org": 0.75,
    "britannica.com": 0.8,
    "merriam-webster.com": 0.8,
    "dictionary.com": 0.7,
    
    # Business
    "bloomberg.com": 0.8,
    "forbes.com": 0.7,
    "harvard.edu": 0.9,
    "mit.edu": 0.9,
    "stanford.edu": 0.9,
}


# Event trigger configurations
EVENT_TRIGGERS = {
    "unknown_topic": {
        "condition": "user_mentions_unknown_concept",
        "action": "research_concept",
        "depth": "medium",
        "immediate": True
    },
    "outdated_info": {
        "condition": "information_age > 30_days",
        "action": "refresh_knowledge",
        "depth": "medium",
        "immediate": False
    },
    "user_request": {
        "condition": "explicit_research_request",
        "action": "deep_research",
        "depth": "deep",
        "immediate": True
    },
    "memory_gap": {
        "condition": "missing_critical_information",
        "action": "fill_gap",
        "depth": "deep",
        "immediate": False
    },
    "contradiction_detected": {
        "condition": "conflicting_information_found",
        "action": "verify_and_resolve",
        "depth": "deep",
        "immediate": True
    },
    "news_alert": {
        "condition": "breaking_news_in_user_interest_area",
        "action": "summarize_and_notify",
        "depth": "shallow",
        "immediate": True
    },
    "trending_topic": {
        "condition": "topic_trending_in_user_domain",
        "action": "research_trend",
        "depth": "medium",
        "immediate": False
    }
}


# Search engine capabilities
ENGINE_CAPABILITIES = {
    "google": {
        "strengths": ["general", "comprehensive", "fresh"],
        "rate_limit": 100,
        "priority": 1
    },
    "bing": {
        "strengths": ["general", "api_friendly"],
        "rate_limit": 1000,
        "priority": 2
    },
    "duckduckgo": {
        "strengths": ["privacy", "no_rate_limit"],
        "rate_limit": None,
        "priority": 3
    },
    "google_scholar": {
        "strengths": ["academic", "research_papers"],
        "rate_limit": 100,
        "priority": 1
    },
    "arxiv": {
        "strengths": ["preprints", "cs_physics_math"],
        "rate_limit": None,
        "priority": 2
    },
    "reddit": {
        "strengths": ["opinions", "discussions", "community"],
        "rate_limit": 60,
        "priority": 2
    },
    "twitter": {
        "strengths": ["real_time", "trends", "sentiment"],
        "rate_limit": 450,
        "priority": 3
    },
    "github": {
        "strengths": ["code", "technical", "documentation"],
        "rate_limit": 5000,
        "priority": 2
    },
    "youtube": {
        "strengths": ["tutorials", "visual", "explanations"],
        "rate_limit": 10000,
        "priority": 3
    }
}
