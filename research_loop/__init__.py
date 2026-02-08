"""
Research Loop - Autonomous Information Gathering System
OpenClaw-Inspired AI Agent Framework - Windows 10 Edition

This module provides comprehensive research capabilities including:
- Multi-trigger research initiation (scheduled, event-driven, curiosity-based)
- Intelligent query generation and optimization
- Multi-source discovery and crawling
- Structured information extraction
- Knowledge synthesis and summarization
- Persistent memory integration
- Source tracking and citation management
"""

__version__ = "1.0.0"
__author__ = "OpenClaw Framework"

from .research_loop import ResearchLoop
from .triggers import (
    ScheduledResearchTrigger,
    EventDrivenResearchTrigger,
    CuriosityEngine
)
from .query_generator import QueryGenerator, QueryDistributor
from .source_discovery import SourceDiscoveryEngine, WebCrawler
from .extractor import InformationExtractor, EntityExtractor, FactExtractor
from .synthesizer import SynthesisEngine, ConflictResolver, SummarizationEngine
from .memory_writer import MemoryWriter
from .citation_tracker import CitationTracker, SourceReliabilityTracker
from .depth_controller import DepthController, AdaptiveResearchController
from .models import (
    ResearchTask,
    SearchQuery,
    DiscoveredSource,
    ExtractedContent,
    ExtractedInformation,
    SynthesisResult,
    ResearchConfig
)
from .config import ResearchLoopConfig

__all__ = [
    # Main loop
    "ResearchLoop",
    
    # Triggers
    "ScheduledResearchTrigger",
    "EventDrivenResearchTrigger",
    "CuriosityEngine",
    
    # Query generation
    "QueryGenerator",
    "QueryDistributor",
    
    # Source discovery
    "SourceDiscoveryEngine",
    "WebCrawler",
    
    # Extraction
    "InformationExtractor",
    "EntityExtractor",
    "FactExtractor",
    
    # Synthesis
    "SynthesisEngine",
    "ConflictResolver",
    "SummarizationEngine",
    
    # Memory
    "MemoryWriter",
    
    # Citations
    "CitationTracker",
    "SourceReliabilityTracker",
    
    # Control
    "DepthController",
    "AdaptiveResearchController",
    
    # Models
    "ResearchTask",
    "SearchQuery",
    "DiscoveredSource",
    "ExtractedContent",
    "ExtractedInformation",
    "SynthesisResult",
    "ResearchConfig",
    
    # Config
    "ResearchLoopConfig"
]
