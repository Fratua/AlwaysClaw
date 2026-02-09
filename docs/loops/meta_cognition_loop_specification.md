# Meta-Cognition Loop Technical Specification
## Self-Reflective Thinking System for Windows 10 OpenClaw-Inspired AI Agent

**Version:** 1.0.0  
**Date:** February 2025  
**Classification:** Technical Architecture Document  
**Target Platform:** Windows 10 / Python 3.11+  
**AI Engine:** GPT-5.2 with Extended Thinking Capability

---

## Executive Summary

The Meta-Cognition Loop is a foundational component of the OpenClaw-inspired AI agent system, designed to enable the agent to monitor, analyze, and improve its own thinking processes. This specification defines a comprehensive self-reflective architecture that implements "thinking about thinking" - the ability to introspect, evaluate, and optimize cognitive operations in real-time.

### Key Capabilities

| Capability | Description | Priority |
|------------|-------------|----------|
| Thought Capture | Real-time logging of reasoning chains | Critical |
| Bias Detection | Automated identification of cognitive biases | Critical |
| Quality Assessment | Multi-dimensional reasoning evaluation | Critical |
| Self-Correction | Automatic error detection and remediation | High |
| Learning System | Pattern extraction from mistakes | High |
| Architecture Evolution | Dynamic cognitive structure improvement | Medium |

---

## 1. System Architecture Overview

### 1.1 Meta-Cognitive Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        META-COGNITION LOOP ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    COGNITIVE MONITORING LAYER                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Thought   │  │   Token     │  │   State     │  │  Context    │ │    │
│  │  │   Logger    │  │   Tracer    │  │   Monitor   │  │  Tracker    │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    REASONING ANALYSIS ENGINE                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Chain     │  │   Logic     │  │   Coherence │  │   Step      │ │    │
│  │  │   Analyzer  │  │   Validator │  │   Checker   │  │  Evaluator  │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    BIAS DETECTION SYSTEM                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │ Confirmation│  │   Anchoring │  │  Recency    │  │  Availability│ │    │
│  │  │    Bias     │  │    Bias     │  │    Bias     │  │    Bias     │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Framing   │  │   Sunk Cost │  │  Overconfidence│ │  Hindsight │ │    │
│  │  │    Bias     │  │   Fallacy   │  │    Bias     │  │    Bias     │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    QUALITY METRICS ENGINE                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   CSD       │  │   SFC       │  │   PRM       │  │   Faithful  │ │    │
│  │  │   Score     │  │   Score     │  │   Score     │  │   Score     │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    IMPROVEMENT GENERATOR                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │  Strategy   │  │   Prompt    │  │   Model     │  │   Parameter │ │    │
│  │  │  Generator  │  │  Optimizer  │  │  Selector   │  │   Tuner     │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    SELF-CORRECTION SYSTEM                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Error     │  │   Retry     │  │   Fallback  │  │   Recovery  │ │    │
│  │  │   Handler   │  │   Engine    │  │   Manager   │  │   Protocol  │ │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │    │
│  │         └─────────────────┴─────────────────┴─────────────────┘      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │                    LEARNING & EVOLUTION                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Pattern   │  │   Mistake   │  │   Success   │  │   Cognitive │ │    │
│  │  │   Extractor │  │   Analyzer  │  │   Library   │  │   Evolution │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interactions

```python
# High-level component interaction flow
class MetaCognitionOrchestrator:
    """
    Central orchestrator for the meta-cognition loop.
    Coordinates all subsystems and manages the self-reflection cycle.
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.thought_logger = ThoughtProcessLogger(config.logging)
        self.reasoning_analyzer = ReasoningAnalysisEngine(config.analysis)
        self.bias_detector = CognitiveBiasDetector(config.bias_detection)
        self.quality_metrics = ThinkingQualityMetrics(config.metrics)
        self.improvement_gen = ImprovementStrategyGenerator(config.improvement)
        self.self_corrector = SelfCorrectionMechanism(config.correction)
        self.learning_system = MistakeLearningSystem(config.learning)
        self.architecture_evolver = CognitiveArchitectureEvolver(config.evolution)
        
    async def execute_meta_cognition_cycle(self, thought_context: ThoughtContext) -> MetaCognitionResult:
        """
        Execute a complete meta-cognition cycle.
        
        Phase 1: Capture and log the thought process
        Phase 2: Analyze reasoning quality
        Phase 3: Detect cognitive biases
        Phase 4: Calculate quality metrics
        Phase 5: Generate improvement strategies
        Phase 6: Apply self-corrections if needed
        Phase 7: Learn from the experience
        Phase 8: Evolve cognitive architecture if warranted
        """
        
        # Phase 1: Thought Capture
        thought_trace = await self.thought_logger.capture(thought_context)
        
        # Phase 2: Reasoning Analysis
        reasoning_analysis = await self.reasoning_analyzer.analyze(thought_trace)
        
        # Phase 3: Bias Detection
        bias_report = await self.bias_detector.detect(thought_trace, reasoning_analysis)
        
        # Phase 4: Quality Metrics
        quality_scores = await self.quality_metrics.calculate(
            thought_trace, reasoning_analysis, bias_report
        )
        
        # Phase 5: Improvement Strategies
        improvements = await self.improvement_gen.generate(
            reasoning_analysis, bias_report, quality_scores
        )
        
        # Phase 6: Self-Correction
        correction_result = await self.self_corrector.apply(
            thought_context, improvements, quality_scores
        )
        
        # Phase 7: Learning
        learning_result = await self.learning_system.learn(
            thought_trace, reasoning_analysis, correction_result
        )
        
        # Phase 8: Architecture Evolution (periodic)
        if self._should_evolve(learning_result):
            evolution_result = await self.architecture_evolver.evolve(learning_result)
        
        return MetaCognitionResult(
            thought_trace=thought_trace,
            reasoning_analysis=reasoning_analysis,
            bias_report=bias_report,
            quality_scores=quality_scores,
            improvements=improvements,
            correction_result=correction_result,
            learning_result=learning_result,
            evolution_result=evolution_result if 'evolution_result' in locals() else None
        )
```

---

## 2. Thought Process Capture and Logging

### 2.1 Architecture

The thought process capture system implements comprehensive logging of all cognitive operations, creating a complete audit trail of the agent's reasoning.

```python
# File: meta_cognition/thought_logger.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum, auto
import json
import hashlib
import asyncio
from collections import deque

class ThoughtType(Enum):
    """Classification of thought types for structured logging."""
    PERCEPTION = auto()      # Input processing and perception
    INFERENCE = auto()       # Logical inference and deduction
    PLANNING = auto()        # Action planning and strategy
    DECISION = auto()        # Decision making
    REFLECTION = auto()      # Self-reflection and introspection
    CREATIVE = auto()        # Creative thinking and ideation
    ANALYSIS = auto()        # Analytical thinking
    SYNTHESIS = auto()       # Synthesis and integration
    META = auto()            # Meta-cognitive thoughts
    EMOTIONAL = auto()       # Emotional processing
    MEMORY = auto()          # Memory retrieval and storage

class ThoughtPriority(Enum):
    """Priority levels for thought processing."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ThoughtNode:
    """
    Individual thought node in the reasoning chain.
    Represents a single cognitive step with full metadata.
    """
    # Identification
    thought_id: str
    parent_id: Optional[str]
    session_id: str
    
    # Content
    thought_type: ThoughtType
    content: str
    raw_tokens: List[str] = field(default_factory=list)
    
    # Context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Reasoning metadata
    confidence: float = 0.0  # 0.0 to 1.0
    certainty: float = 0.0   # Epistemic certainty
    
    # Source tracking
    source_module: str = ""
    source_function: str = ""
    source_line: int = 0
    
    # Relationships
    dependencies: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    
    # Evaluation
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    is_validated: bool = False
    
    # Performance
    processing_time_ms: float = 0.0
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize thought node to dictionary."""
        return {
            'thought_id': self.thought_id,
            'parent_id': self.parent_id,
            'session_id': self.session_id,
            'thought_type': self.thought_type.name,
            'content': self.content,
            'raw_tokens': self.raw_tokens,
            'timestamp': self.timestamp.isoformat(),
            'context_snapshot': self.context_snapshot,
            'confidence': self.confidence,
            'certainty': self.certainty,
            'source_module': self.source_module,
            'source_function': self.source_function,
            'source_line': self.source_line,
            'dependencies': self.dependencies,
            'consequences': self.consequences,
            'evaluation_scores': self.evaluation_scores,
            'is_validated': self.is_validated,
            'processing_time_ms': self.processing_time_ms,
            'token_count': self.token_count
        }

@dataclass
class ReasoningChain:
    """
    Complete reasoning chain composed of multiple thought nodes.
    Represents a full cognitive trajectory from input to output.
    """
    chain_id: str
    session_id: str
    
    # Chain structure
    root_thought_id: str
    thought_nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    
    # Chain metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Input/Output
    input_context: Dict[str, Any] = field(default_factory=dict)
    output_result: Any = None
    
    # Quality metrics
    chain_length: int = 0
    branching_factor: float = 0.0
    depth: int = 0
    
    # Evaluation
    overall_confidence: float = 0.0
    coherence_score: float = 0.0
    validity_score: float = 0.0
    
    # Status
    is_complete: bool = False
    is_successful: bool = False
    error_info: Optional[Dict[str, Any]] = None
    
    def add_thought(self, thought: ThoughtNode) -> None:
        """Add a thought node to the chain."""
        self.thought_nodes[thought.thought_id] = thought
        self.chain_length = len(self.thought_nodes)
        
    def get_thought_path(self, thought_id: str) -> List[ThoughtNode]:
        """Get the path from root to specified thought."""
        path = []
        current_id = thought_id
        
        while current_id:
            thought = self.thought_nodes.get(current_id)
            if thought:
                path.insert(0, thought)
                current_id = thought.parent_id
            else:
                break
                
        return path
    
    def get_branching_structure(self) -> Dict[str, List[str]]:
        """Get the branching structure of the reasoning chain."""
        branches = {}
        for thought_id, thought in self.thought_nodes.items():
            if thought.parent_id:
                if thought.parent_id not in branches:
                    branches[thought.parent_id] = []
                branches[thought.parent_id].append(thought_id)
        return branches

class ThoughtProcessLogger:
    """
    Advanced thought process logging system.
    Captures, structures, and persists all cognitive operations.
    """
    
    def __init__(self, config: 'LoggingConfig'):
        self.config = config
        self.active_sessions: Dict[str, ReasoningChain] = {}
        self.thought_buffer: deque = deque(maxlen=config.buffer_size)
        self.storage_backend = self._init_storage()
        self.hooks: List[Callable] = []
        
        # Performance tracking
        self.stats = {
            'total_thoughts_logged': 0,
            'total_chains_completed': 0,
            'avg_chain_length': 0.0,
            'logging_latency_ms': 0.0
        }
        
    def _init_storage(self) -> 'StorageBackend':
        """Initialize the appropriate storage backend."""
        if self.config.storage_type == 'sqlite':
            return SQLiteStorageBackend(self.config.sqlite_path)
        elif self.config.storage_type == 'json':
            return JSONStorageBackend(self.config.json_path)
        elif self.config.storage_type == 'vector':
            return VectorStorageBackend(self.config.vector_db_config)
        else:
            return HybridStorageBackend(self.config)
    
    async def start_session(self, session_id: str, input_context: Dict[str, Any]) -> ReasoningChain:
        """Start a new reasoning session."""
        chain_id = self._generate_chain_id(session_id)
        chain = ReasoningChain(
            chain_id=chain_id,
            session_id=session_id,
            root_thought_id="",
            input_context=input_context
        )
        self.active_sessions[session_id] = chain
        return chain
    
    async def log_thought(
        self,
        session_id: str,
        content: str,
        thought_type: ThoughtType,
        parent_id: Optional[str] = None,
        confidence: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        priority: ThoughtPriority = ThoughtPriority.NORMAL
    ) -> ThoughtNode:
        """
        Log a single thought with full metadata.
        
        Args:
            session_id: Unique session identifier
            content: The thought content
            thought_type: Classification of the thought
            parent_id: Parent thought ID for chain building
            confidence: Confidence score (0.0-1.0)
            context: Additional context information
            priority: Processing priority
            
        Returns:
            The created ThoughtNode
        """
        start_time = datetime.utcnow()
        
        # Generate unique thought ID
        thought_id = self._generate_thought_id(session_id, content, start_time)
        
        # Get source information
        source_info = self._get_source_info()
        
        # Create thought node
        thought = ThoughtNode(
            thought_id=thought_id,
            parent_id=parent_id,
            session_id=session_id,
            thought_type=thought_type,
            content=content,
            timestamp=start_time,
            context_snapshot=context or {},
            confidence=confidence,
            source_module=source_info['module'],
            source_function=source_info['function'],
            source_line=source_info['line']
        )
        
        # Add to active chain
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_thought(thought)
            if not parent_id:
                self.active_sessions[session_id].root_thought_id = thought_id
        
        # Add to buffer
        self.thought_buffer.append(thought)
        
        # Persist if needed
        if priority == ThoughtPriority.CRITICAL or len(self.thought_buffer) >= self.config.batch_size:
            await self._persist_buffer()
        
        # Update stats
        self.stats['total_thoughts_logged'] += 1
        
        # Execute hooks
        for hook in self.hooks:
            try:
                await hook(thought)
            except Exception as e:
                self._log_hook_error(e, hook)
        
        # Calculate processing time
        thought.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return thought
    
    async def end_session(
        self,
        session_id: str,
        output_result: Any,
        is_successful: bool = True,
        error_info: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """End a reasoning session and finalize the chain."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        chain = self.active_sessions[session_id]
        chain.end_time = datetime.utcnow()
        chain.output_result = output_result
        chain.is_complete = True
        chain.is_successful = is_successful
        chain.error_info = error_info
        
        # Calculate chain metrics
        chain.depth = self._calculate_chain_depth(chain)
        chain.branching_factor = self._calculate_branching_factor(chain)
        
        # Persist final chain
        await self._persist_chain(chain)
        
        # Update stats
        self.stats['total_chains_completed'] += 1
        self._update_avg_chain_length(chain.chain_length)
        
        # Clean up
        del self.active_sessions[session_id]
        
        return chain
    
    def _generate_thought_id(self, session_id: str, content: str, timestamp: datetime) -> str:
        """Generate a unique thought ID."""
        hash_input = f"{session_id}:{content}:{timestamp.isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _generate_chain_id(self, session_id: str) -> str:
        """Generate a unique chain ID."""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{session_id}:{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _get_source_info(self) -> Dict[str, Any]:
        """Get source code information for the thought."""
        import inspect
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the caller
            for _ in range(3):
                if frame.f_back:
                    frame = frame.f_back
                else:
                    break
            
            return {
                'module': inspect.getmodule(frame).__name__ if inspect.getmodule(frame) else 'unknown',
                'function': frame.f_code.co_name,
                'line': frame.f_lineno
            }
        finally:
            del frame
    
    async def _persist_buffer(self) -> None:
        """Persist the thought buffer to storage."""
        if not self.thought_buffer:
            return
        
        thoughts_to_persist = list(self.thought_buffer)
        self.thought_buffer.clear()
        
        await self.storage_backend.store_thoughts_batch(thoughts_to_persist)
    
    async def _persist_chain(self, chain: ReasoningChain) -> None:
        """Persist a complete reasoning chain."""
        await self.storage_backend.store_chain(chain)
    
    def _calculate_chain_depth(self, chain: ReasoningChain) -> int:
        """Calculate the maximum depth of the reasoning chain."""
        max_depth = 0
        for thought_id, thought in chain.thought_nodes.items():
            depth = len(chain.get_thought_path(thought_id))
            max_depth = max(max_depth, depth)
        return max_depth
    
    def _calculate_branching_factor(self, chain: ReasoningChain) -> float:
        """Calculate the average branching factor of the chain."""
        branches = chain.get_branching_structure()
        if not branches:
            return 0.0
        total_children = sum(len(children) for children in branches.values())
        return total_children / len(branches)
    
    def _update_avg_chain_length(self, new_length: int) -> None:
        """Update the average chain length statistic."""
        n = self.stats['total_chains_completed']
        current_avg = self.stats['avg_chain_length']
        self.stats['avg_chain_length'] = ((n - 1) * current_avg + new_length) / n
    
    def register_hook(self, hook: Callable) -> None:
        """Register a callback hook for thought events."""
        self.hooks.append(hook)
    
    async def query_thoughts(
        self,
        session_id: Optional[str] = None,
        thought_type: Optional[ThoughtType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[ThoughtNode]:
        """Query logged thoughts with filters."""
        return await self.storage_backend.query_thoughts(
            session_id=session_id,
            thought_type=thought_type,
            start_time=start_time,
            end_time=end_time,
            min_confidence=min_confidence,
            limit=limit
        )
    
    async def get_chain_analysis(self, chain_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a reasoning chain."""
        chain = await self.storage_backend.get_chain(chain_id)
        if not chain:
            return {}
        
        return {
            'chain_id': chain.chain_id,
            'session_id': chain.session_id,
            'duration_seconds': (chain.end_time - chain.start_time).total_seconds() if chain.end_time else None,
            'chain_length': chain.chain_length,
            'depth': chain.depth,
            'branching_factor': chain.branching_factor,
            'overall_confidence': chain.overall_confidence,
            'coherence_score': chain.coherence_score,
            'validity_score': chain.validity_score,
            'is_successful': chain.is_successful,
            'thought_type_distribution': self._analyze_thought_types(chain),
            'confidence_distribution': self._analyze_confidence_distribution(chain),
            'processing_time_analysis': self._analyze_processing_times(chain)
        }
    
    def _analyze_thought_types(self, chain: ReasoningChain) -> Dict[str, int]:
        """Analyze distribution of thought types in chain."""
        distribution = {}
        for thought in chain.thought_nodes.values():
            type_name = thought.thought_type.name
            distribution[type_name] = distribution.get(type_name, 0) + 1
        return distribution
    
    def _analyze_confidence_distribution(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Analyze confidence distribution in chain."""
        confidences = [t.confidence for t in chain.thought_nodes.values()]
        if not confidences:
            return {}
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5)
        }
    
    def _analyze_processing_times(self, chain: ReasoningChain) -> Dict[str, float]:
        """Analyze processing times in chain."""
        times = [t.processing_time_ms for t in chain.thought_nodes.values()]
        if not times:
            return {}
        
        return {
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': sum(times) / len(times),
            'total_ms': sum(times)
        }
```

### 2.2 Storage Backends

```python
# File: meta_cognition/storage_backends.py

from abc import ABC, abstractmethod
import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class StorageBackend(ABC):
    """Abstract base class for thought storage backends."""
    
    @abstractmethod
    async def store_thoughts_batch(self, thoughts: List[ThoughtNode]) -> None:
        pass
    
    @abstractmethod
    async def store_chain(self, chain: ReasoningChain) -> None:
        pass
    
    @abstractmethod
    async def query_thoughts(self, **filters) -> List[ThoughtNode]:
        pass
    
    @abstractmethod
    async def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        pass

class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage for thought persistence."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    thought_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    session_id TEXT NOT NULL,
                    thought_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_tokens TEXT,
                    timestamp TEXT NOT NULL,
                    context_snapshot TEXT,
                    confidence REAL,
                    certainty REAL,
                    source_module TEXT,
                    source_function TEXT,
                    source_line INTEGER,
                    dependencies TEXT,
                    consequences TEXT,
                    evaluation_scores TEXT,
                    is_validated INTEGER,
                    processing_time_ms REAL,
                    token_count INTEGER,
                    FOREIGN KEY (parent_id) REFERENCES thoughts(thought_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    root_thought_id TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    input_context TEXT,
                    output_result TEXT,
                    chain_length INTEGER,
                    branching_factor REAL,
                    depth INTEGER,
                    overall_confidence REAL,
                    coherence_score REAL,
                    validity_score REAL,
                    is_complete INTEGER,
                    is_successful INTEGER,
                    error_info TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_thoughts_session ON thoughts(session_id)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_thoughts_type ON thoughts(thought_type)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_thoughts_timestamp ON thoughts(timestamp)
            ''')
            
    async def store_thoughts_batch(self, thoughts: List[ThoughtNode]) -> None:
        """Store multiple thoughts in batch."""
        with sqlite3.connect(self.db_path) as conn:
            for thought in thoughts:
                conn.execute('''
                    INSERT OR REPLACE INTO thoughts VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', (
                    thought.thought_id,
                    thought.parent_id,
                    thought.session_id,
                    thought.thought_type.name,
                    thought.content,
                    json.dumps(thought.raw_tokens),
                    thought.timestamp.isoformat(),
                    json.dumps(thought.context_snapshot),
                    thought.confidence,
                    thought.certainty,
                    thought.source_module,
                    thought.source_function,
                    thought.source_line,
                    json.dumps(thought.dependencies),
                    json.dumps(thought.consequences),
                    json.dumps(thought.evaluation_scores),
                    int(thought.is_validated),
                    thought.processing_time_ms,
                    thought.token_count
                ))
            conn.commit()
    
    async def store_chain(self, chain: ReasoningChain) -> None:
        """Store a complete reasoning chain."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO reasoning_chains VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                chain.chain_id,
                chain.session_id,
                chain.root_thought_id,
                chain.start_time.isoformat(),
                chain.end_time.isoformat() if chain.end_time else None,
                json.dumps(chain.input_context),
                json.dumps(chain.output_result) if chain.output_result else None,
                chain.chain_length,
                chain.branching_factor,
                chain.depth,
                chain.overall_confidence,
                chain.coherence_score,
                chain.validity_score,
                int(chain.is_complete),
                int(chain.is_successful),
                json.dumps(chain.error_info) if chain.error_info else None
            ))
            conn.commit()
    
    async def query_thoughts(self, **filters) -> List[ThoughtNode]:
        """Query thoughts with filters."""
        query = "SELECT * FROM thoughts WHERE 1=1"
        params = []
        
        if filters.get('session_id'):
            query += " AND session_id = ?"
            params.append(filters['session_id'])
        
        if filters.get('thought_type'):
            query += " AND thought_type = ?"
            params.append(filters['thought_type'].name)
        
        if filters.get('start_time'):
            query += " AND timestamp >= ?"
            params.append(filters['start_time'].isoformat())
        
        if filters.get('end_time'):
            query += " AND timestamp <= ?"
            params.append(filters['end_time'].isoformat())
        
        if filters.get('min_confidence'):
            query += " AND confidence >= ?"
            params.append(filters['min_confidence'])
        
        query += " ORDER BY timestamp DESC"
        
        if filters.get('limit'):
            query += " LIMIT ?"
            params.append(filters['limit'])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
        return [self._row_to_thought(row) for row in rows]
    
    def _row_to_thought(self, row: sqlite3.Row) -> ThoughtNode:
        """Convert database row to ThoughtNode."""
        return ThoughtNode(
            thought_id=row['thought_id'],
            parent_id=row['parent_id'],
            session_id=row['session_id'],
            thought_type=ThoughtType[row['thought_type']],
            content=row['content'],
            raw_tokens=json.loads(row['raw_tokens']) if row['raw_tokens'] else [],
            timestamp=datetime.fromisoformat(row['timestamp']),
            context_snapshot=json.loads(row['context_snapshot']) if row['context_snapshot'] else {},
            confidence=row['confidence'] or 0.0,
            certainty=row['certainty'] or 0.0,
            source_module=row['source_module'] or '',
            source_function=row['source_function'] or '',
            source_line=row['source_line'] or 0,
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else [],
            consequences=json.loads(row['consequences']) if row['consequences'] else [],
            evaluation_scores=json.loads(row['evaluation_scores']) if row['evaluation_scores'] else {},
            is_validated=bool(row['is_validated']),
            processing_time_ms=row['processing_time_ms'] or 0.0,
            token_count=row['token_count'] or 0
        )
    
    async def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Retrieve a reasoning chain by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM reasoning_chains WHERE chain_id = ?",
                (chain_id,)
            )
            row = cursor.fetchone()
            
        if not row:
            return None
            
        # Get all thoughts for this chain
        thoughts = await self.query_thoughts(session_id=row['session_id'])
        
        return ReasoningChain(
            chain_id=row['chain_id'],
            session_id=row['session_id'],
            root_thought_id=row['root_thought_id'],
            thought_nodes={t.thought_id: t for t in thoughts},
            start_time=datetime.fromisoformat(row['start_time']),
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            input_context=json.loads(row['input_context']) if row['input_context'] else {},
            output_result=json.loads(row['output_result']) if row['output_result'] else None,
            chain_length=row['chain_length'],
            branching_factor=row['branching_factor'],
            depth=row['depth'],
            overall_confidence=row['overall_confidence'],
            coherence_score=row['coherence_score'],
            validity_score=row['validity_score'],
            is_complete=bool(row['is_complete']),
            is_successful=bool(row['is_successful']),
            error_info=json.loads(row['error_info']) if row['error_info'] else None
        )

class VectorStorageBackend(StorageBackend):
    """Vector database storage for semantic thought retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None  # Initialize embedding model
        self.vector_store = None     # Initialize vector store (e.g., Chroma, Pinecone)
        
    async def store_thoughts_batch(self, thoughts: List[ThoughtNode]) -> None:
        """Store thoughts with vector embeddings."""
        # Generate embeddings for thought content
        embeddings = await self._generate_embeddings([t.content for t in thoughts])
        
        # Store in vector database with metadata
        for thought, embedding in zip(thoughts, embeddings):
            metadata = {
                'thought_id': thought.thought_id,
                'session_id': thought.session_id,
                'thought_type': thought.thought_type.name,
                'confidence': thought.confidence,
                'timestamp': thought.timestamp.isoformat()
            }
            await self.vector_store.add(
                id=thought.thought_id,
                embedding=embedding,
                metadata=metadata,
                document=thought.content
            )
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search thoughts by semantic similarity."""
        query_embedding = await self._generate_embeddings([query])
        results = await self.vector_store.search(
            query_embedding=query_embedding[0],
            top_k=top_k,
            filters=filters
        )
        return results
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text."""
        # Implementation depends on embedding model
        pass
```

---

## 3. Reasoning Analysis Frameworks

### 3.1 Chain-of-Thought Analysis Engine

```python
# File: meta_cognition/reasoning_analysis.py

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import re
import numpy as np
from collections import defaultdict

class ReasoningPattern(Enum):
    """Identified reasoning patterns."""
    DEDUCTIVE = "deductive"           # General to specific
    INDUCTIVE = "inductive"           # Specific to general
    ABDUCTIVE = "abductive"           # Inference to best explanation
    ANALOGICAL = "analogical"         # Reasoning by analogy
    CAUSAL = "causal"                 # Cause-effect reasoning
    COUNTERFACTUAL = "counterfactual" # What-if reasoning
    PROBABILISTIC = "probabilistic"   # Statistical reasoning
    DIALECTICAL = "dialectical"       # Thesis-antithesis-synthesis
    HEURISTIC = "heuristic"           # Rule-of-thumb reasoning
    SYSTEMATIC = "systematic"         # Step-by-step methodical

class LogicalFallacy(Enum):
    """Common logical fallacies to detect."""
    CIRCULAR_REASONING = "circular_reasoning"
    FALSE_CAUSE = "false_cause"
    HASTY_GENERALIZATION = "hasty_generalization"
    SLIPPERY_SLOPE = "slippery_slope"
    FALSE_DICHOTOMY = "false_dichotomy"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    RED_HERRING = "red_herring"
    BEGGING_THE_QUESTION = "begging_the_question"

@dataclass
class StepAnalysis:
    """Analysis of a single reasoning step."""
    step_number: int
    thought_id: str
    content: str
    
    # Pattern detection
    detected_patterns: List[ReasoningPattern]
    pattern_confidence: Dict[ReasoningPattern, float]
    
    # Logical analysis
    premises: List[str]
    conclusion: str
    inference_type: Optional[str]
    
    # Fallacy detection
    potential_fallacies: List[LogicalFallacy]
    fallacy_evidence: Dict[LogicalFallacy, str]
    
    # Coherence
    coherence_with_previous: float
    coherence_with_next: float
    
    # Quality metrics
    clarity_score: float
    relevance_score: float
    sufficiency_score: float
    
    # Dependencies
    depends_on: List[int]
    supports: List[int]

@dataclass
class ChainAnalysis:
    """Complete analysis of a reasoning chain."""
    chain_id: str
    step_analyses: List[StepAnalysis]
    
    # Overall patterns
    dominant_patterns: List[Tuple[ReasoningPattern, float]]
    pattern_transitions: Dict[Tuple[ReasoningPattern, ReasoningPattern], int]
    
    # Logical structure
    logical_validity_score: float
    soundness_score: float
    completeness_score: float
    
    # Coherence metrics
    local_coherence: float  # Adjacent step coherence
    global_coherence: float  # Overall chain coherence
    
    # Redundancy and gaps
    redundancy_score: float
    gap_identification: List[Dict[str, Any]]
    
    # Critical points
    critical_steps: List[int]  # Steps that significantly affect outcome
    weak_points: List[int]     # Steps with low confidence/coherence
    
    # Comparison metrics
    alternative_paths_considered: int
    counterarguments_addressed: int

class ReasoningAnalysisEngine:
    """
    Advanced reasoning analysis engine.
    Performs multi-dimensional analysis of reasoning chains.
    """
    
    def __init__(self, config: 'AnalysisConfig'):
        self.config = config
        
        # Pattern detectors
        self.pattern_detectors = self._init_pattern_detectors()
        
        # Fallacy detectors
        self.fallacy_detectors = self._init_fallacy_detectors()
        
        # NLP components
        self.nlp_pipeline = self._init_nlp_pipeline()
        
        # Coherence model
        self.coherence_model = self._init_coherence_model()
        
    def _init_pattern_detectors(self) -> Dict[ReasoningPattern, 'PatternDetector']:
        """Initialize pattern detection modules."""
        return {
            ReasoningPattern.DEDUCTIVE: DeductivePatternDetector(),
            ReasoningPattern.INDUCTIVE: InductivePatternDetector(),
            ReasoningPattern.ABDUCTIVE: AbductivePatternDetector(),
            ReasoningPattern.ANALOGICAL: AnalogicalPatternDetector(),
            ReasoningPattern.CAUSAL: CausalPatternDetector(),
            ReasoningPattern.COUNTERFACTUAL: CounterfactualPatternDetector(),
            ReasoningPattern.PROBABILISTIC: ProbabilisticPatternDetector(),
            ReasoningPattern.DIALECTICAL: DialecticalPatternDetector(),
            ReasoningPattern.HEURISTIC: HeuristicPatternDetector(),
            ReasoningPattern.SYSTEMATIC: SystematicPatternDetector()
        }
    
    def _init_fallacy_detectors(self) -> Dict[LogicalFallacy, 'FallacyDetector']:
        """Initialize fallacy detection modules."""
        return {
            LogicalFallacy.CIRCULAR_REASONING: CircularReasoningDetector(),
            LogicalFallacy.FALSE_CAUSE: FalseCauseDetector(),
            LogicalFallacy.HASTY_GENERALIZATION: HastyGeneralizationDetector(),
            LogicalFallacy.SLIPPERY_SLOPE: SlipperySlopeDetector(),
            LogicalFallacy.FALSE_DICHOTOMY: FalseDichotomyDetector(),
            LogicalFallacy.APPEAL_TO_AUTHORITY: AppealToAuthorityDetector(),
            LogicalFallacy.AD_HOMINEM: AdHominemDetector(),
            LogicalFallacy.STRAW_MAN: StrawManDetector(),
            LogicalFallacy.RED_HERRING: RedHerringDetector(),
            LogicalFallacy.BEGGING_THE_QUESTION: BeggingTheQuestionDetector()
        }
    
    async def analyze(self, thought_trace: 'ThoughtTrace') -> ChainAnalysis:
        """
        Perform comprehensive analysis of a reasoning chain.
        
        Args:
            thought_trace: The captured thought trace to analyze
            
        Returns:
            Complete chain analysis with all metrics
        """
        # Get ordered thoughts
        thoughts = self._order_thoughts(thought_trace)
        
        # Analyze each step
        step_analyses = []
        for i, thought in enumerate(thoughts):
            step_analysis = await self._analyze_step(i, thought, thoughts)
            step_analyses.append(step_analysis)
        
        # Calculate overall metrics
        dominant_patterns = self._identify_dominant_patterns(step_analyses)
        pattern_transitions = self._analyze_pattern_transitions(step_analyses)
        
        logical_validity = self._assess_logical_validity(step_analyses)
        soundness = self._assess_soundness(step_analyses)
        completeness = self._assess_completeness(step_analyses)
        
        local_coherence = self._calculate_local_coherence(step_analyses)
        global_coherence = self._calculate_global_coherence(step_analyses, thoughts)
        
        redundancy = self._assess_redundancy(step_analyses)
        gaps = self._identify_gaps(step_analyses, thoughts)
        
        critical_steps = self._identify_critical_steps(step_analyses)
        weak_points = self._identify_weak_points(step_analyses)
        
        return ChainAnalysis(
            chain_id=thought_trace.chain_id,
            step_analyses=step_analyses,
            dominant_patterns=dominant_patterns,
            pattern_transitions=pattern_transitions,
            logical_validity_score=logical_validity,
            soundness_score=soundness,
            completeness_score=completeness,
            local_coherence=local_coherence,
            global_coherence=global_coherence,
            redundancy_score=redundancy,
            gap_identification=gaps,
            critical_steps=critical_steps,
            weak_points=weak_points,
            alternative_paths_considered=self._count_alternatives(thoughts),
            counterarguments_addressed=self._count_counterarguments(thoughts)
        )
    
    async def _analyze_step(
        self,
        step_number: int,
        thought: ThoughtNode,
        all_thoughts: List[ThoughtNode]
    ) -> StepAnalysis:
        """Analyze a single reasoning step."""
        
        # Detect reasoning patterns
        detected_patterns = []
        pattern_confidence = {}
        for pattern, detector in self.pattern_detectors.items():
            confidence = await detector.detect(thought.content)
            if confidence > self.config.pattern_threshold:
                detected_patterns.append(pattern)
                pattern_confidence[pattern] = confidence
        
        # Extract logical structure
        premises, conclusion, inference_type = await self._extract_logical_structure(thought.content)
        
        # Detect potential fallacies
        potential_fallacies = []
        fallacy_evidence = {}
        for fallacy, detector in self.fallacy_detectors.items():
            evidence = await detector.detect(thought.content, premises, conclusion)
            if evidence:
                potential_fallacies.append(fallacy)
                fallacy_evidence[fallacy] = evidence
        
        # Calculate coherence
        coherence_prev = self._calculate_step_coherence(
            thought, all_thoughts[step_number - 1] if step_number > 0 else None
        )
        coherence_next = self._calculate_step_coherence(
            thought, all_thoughts[step_number + 1] if step_number < len(all_thoughts) - 1 else None
        )
        
        # Calculate quality scores
        clarity = self._assess_clarity(thought.content)
        relevance = self._assess_relevance(thought, all_thoughts)
        sufficiency = self._assess_sufficiency(thought.content, premises)
        
        # Identify dependencies
        depends_on, supports = self._identify_dependencies(step_number, all_thoughts)
        
        return StepAnalysis(
            step_number=step_number,
            thought_id=thought.thought_id,
            content=thought.content,
            detected_patterns=detected_patterns,
            pattern_confidence=pattern_confidence,
            premises=premises,
            conclusion=conclusion,
            inference_type=inference_type,
            potential_fallacies=potential_fallacies,
            fallacy_evidence=fallacy_evidence,
            coherence_with_previous=coherence_prev,
            coherence_with_next=coherence_next,
            clarity_score=clarity,
            relevance_score=relevance,
            sufficiency_score=sufficiency,
            depends_on=depends_on,
            supports=supports
        )
    
    async def _extract_logical_structure(self, content: str) -> Tuple[List[str], str, Optional[str]]:
        """Extract premises, conclusion, and inference type from content."""
        # Use NLP to identify logical structure
        doc = await self.nlp_pipeline.process(content)
        
        premises = []
        conclusion = ""
        inference_type = None
        
        # Look for premise indicators
        premise_indicators = ['because', 'since', 'given that', 'as', 'considering']
        conclusion_indicators = ['therefore', 'thus', 'hence', 'consequently', 'so', 'it follows that']
        
        sentences = doc.sentences
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.text.lower()
            
            # Check for conclusion indicators
            if any(ind in sent_lower for ind in conclusion_indicators):
                conclusion = sent.text
                # Previous sentences are likely premises
                premises = [s.text for s in sentences[:i]]
            
            # Detect inference type
            if 'if' in sent_lower and 'then' in sent_lower:
                inference_type = 'conditional'
            elif 'all' in sent_lower or 'every' in sent_lower:
                inference_type = 'universal'
            elif 'some' in sent_lower or 'exists' in sent_lower:
                inference_type = 'existential'
        
        if not conclusion and sentences:
            conclusion = sentences[-1].text
            premises = [s.text for s in sentences[:-1]]
        
        return premises, conclusion, inference_type
    
    def _calculate_step_coherence(
        self,
        thought1: ThoughtNode,
        thought2: Optional[ThoughtNode]
    ) -> float:
        """Calculate coherence between two adjacent steps."""
        if not thought2:
            return 1.0  # Boundary condition
        
        # Use coherence model to calculate semantic similarity
        coherence = self.coherence_model.calculate(
            thought1.content,
            thought2.content
        )
        
        # Check for explicit connections
        if thought2.thought_id in thought1.consequences:
            coherence = min(1.0, coherence + 0.2)
        
        return coherence
    
    def _identify_dominant_patterns(
        self,
        step_analyses: List[StepAnalysis]
    ) -> List[Tuple[ReasoningPattern, float]]:
        """Identify the dominant reasoning patterns in the chain."""
        pattern_counts = defaultdict(lambda: {'count': 0, 'confidence': 0.0})
        
        for step in step_analyses:
            for pattern in step.detected_patterns:
                pattern_counts[pattern]['count'] += 1
                pattern_counts[pattern]['confidence'] += step.pattern_confidence.get(pattern, 0)
        
        # Calculate dominance scores
        total_steps = len(step_analyses)
        dominant = []
        
        for pattern, data in pattern_counts.items():
            frequency = data['count'] / total_steps
            avg_confidence = data['confidence'] / data['count'] if data['count'] > 0 else 0
            dominance_score = (frequency + avg_confidence) / 2
            
            if dominance_score > self.config.dominance_threshold:
                dominant.append((pattern, dominance_score))
        
        return sorted(dominant, key=lambda x: x[1], reverse=True)
    
    def _assess_logical_validity(self, step_analyses: List[StepAnalysis]) -> float:
        """Assess the logical validity of the entire chain."""
        if not step_analyses:
            return 0.0
        
        validity_scores = []
        
        for step in step_analyses:
            # Check for fallacies
            fallacy_penalty = len(step.potential_fallacies) * 0.2
            
            # Check for proper inference structure
            structure_score = 0.5
            if step.premises and step.conclusion:
                structure_score = 0.8
            if step.inference_type:
                structure_score = 1.0
            
            # Check coherence
            coherence_score = (step.coherence_with_previous + step.coherence_with_next) / 2
            
            step_validity = max(0, (structure_score + coherence_score) / 2 - fallacy_penalty)
            validity_scores.append(step_validity)
        
        return sum(validity_scores) / len(validity_scores)
    
    def _identify_critical_steps(self, step_analyses: List[StepAnalysis]) -> List[int]:
        """Identify steps that are critical to the reasoning outcome."""
        critical = []
        
        for i, step in enumerate(step_analyses):
            critical_score = 0.0
            
            # Steps with many dependents are critical
            critical_score += len(step.supports) * 0.3
            
            # Steps with unique patterns may be critical
            if step.detected_patterns:
                critical_score += 0.2
            
            # Steps with fallacies are critical (negatively)
            if step.potential_fallacies:
                critical_score += 0.3
            
            # Steps with low coherence are critical (need attention)
            if step.coherence_with_previous < 0.5:
                critical_score += 0.2
            
            if critical_score > self.config.critical_threshold:
                critical.append(i)
        
        return critical
    
    def _identify_weak_points(self, step_analyses: List[StepAnalysis]) -> List[int]:
        """Identify weak points in the reasoning chain."""
        weak = []
        
        for i, step in enumerate(step_analyses):
            weakness_score = 0.0
            
            # Low confidence
            weakness_score += (1 - step.clarity_score) * 0.25
            
            # Low relevance
            weakness_score += (1 - step.relevance_score) * 0.25
            
            # Low sufficiency
            weakness_score += (1 - step.sufficiency_score) * 0.25
            
            # Low coherence
            weakness_score += (1 - (step.coherence_with_previous + step.coherence_with_next) / 2) * 0.25
            
            if weakness_score > self.config.weakness_threshold:
                weak.append(i)
        
        return weak

# Pattern Detector Base Classes
class PatternDetector(ABC):
    """Base class for reasoning pattern detectors."""
    
    @abstractmethod
    async def detect(self, content: str) -> float:
        """Detect pattern in content, return confidence score."""
        pass

class DeductivePatternDetector(PatternDetector):
    """Detector for deductive reasoning patterns."""
    
    INDICATORS = [
        'all', 'every', 'any', 'none', 'always', 'necessarily',
        'therefore', 'thus', 'hence', 'it follows that',
        'if.*then', 'implies', 'entails'
    ]
    
    async def detect(self, content: str) -> float:
        content_lower = content.lower()
        score = 0.0
        
        for indicator in self.INDICATORS:
            if re.search(indicator, content_lower):
                score += 0.2
        
        # Check for syllogistic structure
        if self._has_syllogistic_structure(content):
            score += 0.4
        
        return min(1.0, score)
    
    def _has_syllogistic_structure(self, content: str) -> bool:
        """Check if content has syllogistic structure."""
        # Look for major premise, minor premise, conclusion pattern
        sentences = content.split('.')
        return len(sentences) >= 3 and 'all' in content.lower()

class InductivePatternDetector(PatternDetector):
    """Detector for inductive reasoning patterns."""
    
    INDICATORS = [
        'most', 'many', 'some', 'typically', 'usually', 'often',
        'observed', 'found', 'noticed', 'pattern', 'trend',
        'likely', 'probably', 'in general', 'as a rule'
    ]
    
    async def detect(self, content: str) -> float:
        content_lower = content.lower()
        score = 0.0
        
        for indicator in self.INDICATORS:
            if indicator in content_lower:
                score += 0.15
        
        # Check for generalization
        if self._has_generalization(content):
            score += 0.3
        
        return min(1.0, score)
    
    def _has_generalization(self, content: str) -> bool:
        """Check if content makes a generalization."""
        # Look for patterns like "X cases suggest Y"
        patterns = [
            r'\d+\s+cases?.*suggest',
            r'observed.*pattern',
            r'in\s+general',
            r'typically.*will'
        ]
        return any(re.search(p, content.lower()) for p in patterns)

class CausalPatternDetector(PatternDetector):
    """Detector for causal reasoning patterns."""
    
    CAUSAL_INDICATORS = [
        'because', 'causes', 'leads to', 'results in', 'produces',
        'due to', 'owing to', 'as a result', 'consequently',
        'the reason', 'explains why', 'is responsible for'
    ]
    
    async def detect(self, content: str) -> float:
        content_lower = content.lower()
        score = 0.0
        
        for indicator in self.CAUSAL_INDICATORS:
            if indicator in content_lower:
                score += 0.2
        
        # Check for cause-effect structure
        if self._has_cause_effect_structure(content):
            score += 0.4
        
        return min(1.0, score)
    
    def _has_cause_effect_structure(self, content: str) -> bool:
        """Check if content has clear cause-effect structure."""
        # Look for explicit cause-effect relationships
        cause_patterns = [
            r'(because|since|as)\s+(.+?),?\s+(therefore|thus|so|consequently)',
            r'(.+?)\s+(causes?|leads? to|results? in)\s+(.+)'
        ]
        return any(re.search(p, content.lower()) for p in cause_patterns)

# Fallacy Detector Base Classes
class FallacyDetector(ABC):
    """Base class for logical fallacy detectors."""
    
    @abstractmethod
    async def detect(self, content: str, premises: List[str], conclusion: str) -> Optional[str]:
        """Detect fallacy, return evidence if found."""
        pass

class CircularReasoningDetector(FallacyDetector):
    """Detector for circular reasoning."""
    
    async def detect(self, content: str, premises: List[str], conclusion: str) -> Optional[str]:
        conclusion_lower = conclusion.lower()
        
        for premise in premises:
            # Check if conclusion appears in premise or vice versa
            if conclusion_lower in premise.lower() or premise.lower() in conclusion_lower:
                return f"Conclusion '{conclusion}' appears in premise '{premise}'"
        
        return None

class FalseCauseDetector(FallacyDetector):
    """Detector for false cause fallacies."""
    
    CORRELATION_INDICATORS = [
        'correlates with', 'associated with', 'linked to',
        'often occurs with', 'tends to accompany'
    ]
    
    CAUSAL_CLAIMS = [
        'causes', 'leads to', 'results in', 'produces'
    ]
    
    async def detect(self, content: str, premises: List[str], conclusion: str) -> Optional[str]:
        content_lower = content.lower()
        
        # Check for correlation being treated as causation
        has_correlation = any(ind in content_lower for ind in self.CORRELATION_INDICATORS)
        has_causal_claim = any(claim in content_lower for claim in self.CAUSAL_CLAIMS)
        
        if has_correlation and has_causal_claim:
            return "Correlation may be incorrectly treated as causation"
        
        # Check for post hoc ergo propter hoc
        if 'after' in content_lower and 'therefore' in content_lower:
            return "Temporal sequence may be incorrectly treated as causation"
        
        return None

class HastyGeneralizationDetector(FallacyDetector):
    """Detector for hasty generalization."""
    
    async def detect(self, content: str, premises: List[str], conclusion: str) -> Optional[str]:
        # Look for small sample sizes leading to broad conclusions
        sample_indicators = ['one case', 'a few', 'some people', 'I know someone']
        generalization_indicators = ['all', 'everyone', 'always', 'never', 'nobody']
        
        content_lower = content.lower()
        
        has_small_sample = any(ind in content_lower for ind in sample_indicators)
        has_generalization = any(ind in content_lower for ind in generalization_indicators)
        
        if has_small_sample and has_generalization:
            return "Small sample may not support broad generalization"
        
        return None
```

---

## 4. Cognitive Bias Detection System

### 4.1 Bias Detection Architecture

```python
# File: meta_cognition/bias_detection.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto
from datetime import datetime
import re
import numpy as np
from collections import defaultdict

class BiasSeverity(Enum):
    """Severity levels for detected biases."""
    CRITICAL = auto()   # Significantly impacts decision quality
    HIGH = auto()       # Likely to affect reasoning
    MEDIUM = auto()     # May influence thinking
    LOW = auto()        # Minor potential influence
    INFORMATIONAL = auto()  # For awareness only

class BiasCategory(Enum):
    """Categories of cognitive biases."""
    CONFIRMATION = "confirmation_bias"
    ANCHORING = "anchoring_bias"
    AVAILABILITY = "availability_bias"
    RECENCY = "recency_bias"
    FRAMING = "framing_bias"
    SUNK_COST = "sunk_cost_fallacy"
    OVERCONFIDENCE = "overconfidence_bias"
    HINDSIGHT = "hindsight_bias"
    AUTHORITY = "authority_bias"
    GROUPTHINK = "groupthink_bias"
    STATUS_QUO = "status_quo_bias"
    LOSS_AVERSION = "loss_aversion"
    OPTIMISM = "optimism_bias"
    PESSIMISM = "pessimism_bias"
    SELF_SERVING = "self_serving_bias"
    FUNDAMENTAL_ATTRIBUTION = "fundamental_attribution_error"
    DUNNING_KRUGER = "dunning_kruger_effect"
    BANDWAGON = "bandwagon_effect"
    STEREOTYPING = "stereotyping_bias"
    SELECTION = "selection_bias"

@dataclass
class BiasInstance:
    """Instance of a detected cognitive bias."""
    bias_id: str
    bias_category: BiasCategory
    severity: BiasSeverity
    
    # Location
    thought_id: str
    content_excerpt: str
    
    # Detection details
    confidence: float  # Detection confidence
    evidence: str      # Evidence supporting detection
    trigger_phrases: List[str]
    
    # Context
    context_before: List[str]
    context_after: List[str]
    
    # Mitigation
    mitigation_suggestion: str
    alternative_perspective: str
    
    # Metadata
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    detector_version: str = "1.0"

@dataclass
class BiasReport:
    """Complete bias detection report."""
    report_id: str
    chain_id: str
    
    # Detected biases
    detected_biases: List[BiasInstance]
    
    # Summary statistics
    total_biases_detected: int
    severity_distribution: Dict[BiasSeverity, int]
    category_distribution: Dict[BiasCategory, int]
    
    # Risk assessment
    overall_bias_risk_score: float  # 0.0 to 1.0
    highest_risk_areas: List[str]
    
    # Mitigation summary
    mitigation_opportunities: int
    recommended_actions: List[str]
    
    # Comparative analysis
    bias_trend: str  # "increasing", "decreasing", "stable"
    comparison_to_baseline: float  # Percentage difference

class CognitiveBiasDetector:
    """
    Comprehensive cognitive bias detection system.
    Identifies and analyzes cognitive biases in reasoning chains.
    """
    
    def __init__(self, config: 'BiasDetectionConfig'):
        self.config = config
        
        # Initialize bias detectors
        self.bias_detectors = self._init_bias_detectors()
        
        # Historical bias tracking
        self.bias_history: Dict[str, List[BiasInstance]] = defaultdict(list)
        
        # Pattern recognition model
        self.pattern_model = self._init_pattern_model()
        
        # Context analyzer
        self.context_analyzer = ContextAnalyzer()
        
    def _init_bias_detectors(self) -> Dict[BiasCategory, 'BiasDetector']:
        """Initialize all bias detection modules."""
        return {
            BiasCategory.CONFIRMATION: ConfirmationBiasDetector(),
            BiasCategory.ANCHORING: AnchoringBiasDetector(),
            BiasCategory.AVAILABILITY: AvailabilityBiasDetector(),
            BiasCategory.RECENCY: RecencyBiasDetector(),
            BiasCategory.FRAMING: FramingBiasDetector(),
            BiasCategory.SUNK_COST: SunkCostBiasDetector(),
            BiasCategory.OVERCONFIDENCE: OverconfidenceBiasDetector(),
            BiasCategory.HINDSIGHT: HindsightBiasDetector(),
            BiasCategory.AUTHORITY: AuthorityBiasDetector(),
            BiasCategory.GROUPTHINK: GroupthinkBiasDetector(),
            BiasCategory.STATUS_QUO: StatusQuoBiasDetector(),
            BiasCategory.LOSS_AVERSION: LossAversionBiasDetector(),
            BiasCategory.OPTIMISM: OptimismBiasDetector(),
            BiasCategory.PESSIMISM: PessimismBiasDetector(),
            BiasCategory.SELF_SERVING: SelfServingBiasDetector(),
            BiasCategory.FUNDAMENTAL_ATTRIBUTION: FundamentalAttributionDetector(),
            BiasCategory.DUNNING_KRUGER: DunningKrugerDetector(),
            BiasCategory.BANDWAGON: BandwagonBiasDetector(),
            BiasCategory.STEREOTYPING: StereotypingBiasDetector(),
            BiasCategory.SELECTION: SelectionBiasDetector()
        }
    
    async def detect(
        self,
        thought_trace: 'ThoughtTrace',
        reasoning_analysis: 'ChainAnalysis'
    ) -> BiasReport:
        """
        Detect cognitive biases in a reasoning chain.
        
        Args:
            thought_trace: The captured thought trace
            reasoning_analysis: Results from reasoning analysis
            
        Returns:
            Complete bias detection report
        """
        detected_biases = []
        
        # Analyze each thought for biases
        thoughts = list(thought_trace.thought_nodes.values())
        
        for i, thought in enumerate(thoughts):
            # Get context
            context_before = [t.content for t in thoughts[max(0, i-3):i]]
            context_after = [t.content for t in thoughts[i+1:min(len(thoughts), i+4)]]
            
            # Run all bias detectors
            for bias_category, detector in self.bias_detectors.items():
                bias_instance = await detector.detect(
                    thought=thought,
                    context_before=context_before,
                    context_after=context_after,
                    reasoning_analysis=reasoning_analysis,
                    step_number=i
                )
                
                if bias_instance:
                    detected_biases.append(bias_instance)
        
        # Analyze cross-thought patterns
        cross_thought_biases = await self._detect_cross_thought_biases(
            thoughts, reasoning_analysis
        )
        detected_biases.extend(cross_thought_biases)
        
        # Generate report
        return self._generate_bias_report(
            chain_id=thought_trace.chain_id,
            detected_biases=detected_biases
        )
    
    async def _detect_cross_thought_biases(
        self,
        thoughts: List[ThoughtNode],
        reasoning_analysis: 'ChainAnalysis'
    ) -> List[BiasInstance]:
        """Detect biases that emerge across multiple thoughts."""
        cross_biases = []
        
        # Check for confirmation bias across chain
        confirmation_bias = await self._detect_chain_confirmation_bias(thoughts)
        if confirmation_bias:
            cross_biases.append(confirmation_bias)
        
        # Check for anchoring effects
        anchoring_bias = await self._detect_anchoring_effect(thoughts)
        if anchoring_bias:
            cross_biases.append(anchoring_bias)
        
        # Check for recency effects
        recency_bias = await self._detect_recency_effect(thoughts)
        if recency_bias:
            cross_biases.append(recency_bias)
        
        # Check for overconfidence trend
        overconfidence_bias = await self._detect_overconfidence_trend(thoughts)
        if overconfidence_bias:
            cross_biases.append(overconfidence_bias)
        
        return cross_biases
    
    def _generate_bias_report(
        self,
        chain_id: str,
        detected_biases: List[BiasInstance]
    ) -> BiasReport:
        """Generate comprehensive bias report."""
        
        # Calculate severity distribution
        severity_dist = defaultdict(int)
        for bias in detected_biases:
            severity_dist[bias.severity] += 1
        
        # Calculate category distribution
        category_dist = defaultdict(int)
        for bias in detected_biases:
            category_dist[bias.bias_category] += 1
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(detected_biases)
        
        # Identify highest risk areas
        risk_areas = self._identify_risk_areas(detected_biases)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detected_biases)
        
        # Calculate trend
        trend = self._calculate_bias_trend(chain_id, category_dist)
        
        return BiasReport(
            report_id=self._generate_report_id(),
            chain_id=chain_id,
            detected_biases=detected_biases,
            total_biases_detected=len(detected_biases),
            severity_distribution=dict(severity_dist),
            category_distribution=dict(category_dist),
            overall_bias_risk_score=risk_score,
            highest_risk_areas=risk_areas,
            mitigation_opportunities=len([b for b in detected_biases if b.severity.value <= BiasSeverity.MEDIUM.value]),
            recommended_actions=recommendations,
            bias_trend=trend,
            comparison_to_baseline=self._compare_to_baseline(category_dist)
        )
    
    def _calculate_risk_score(self, biases: List[BiasInstance]) -> float:
        """Calculate overall bias risk score."""
        if not biases:
            return 0.0
        
        severity_weights = {
            BiasSeverity.CRITICAL: 1.0,
            BiasSeverity.HIGH: 0.75,
            BiasSeverity.MEDIUM: 0.5,
            BiasSeverity.LOW: 0.25,
            BiasSeverity.INFORMATIONAL: 0.1
        }
        
        total_weight = sum(
            severity_weights[b.severity] * b.confidence
            for b in biases
        )
        
        # Normalize
        max_possible = len(biases) * 1.0
        return min(1.0, total_weight / max_possible) if max_possible > 0 else 0.0
    
    def _identify_risk_areas(self, biases: List[BiasInstance]) -> List[str]:
        """Identify areas of highest bias risk."""
        # Group by thought location
        thought_risks = defaultdict(float)
        for bias in biases:
            thought_risks[bias.thought_id] += severity_weights[bias.severity] * bias.confidence
        
        # Sort by risk and return top areas
        sorted_areas = sorted(thought_risks.items(), key=lambda x: x[1], reverse=True)
        return [area[0] for area in sorted_areas[:5]]
    
    def _generate_recommendations(self, biases: List[BiasInstance]) -> List[str]:
        """Generate mitigation recommendations."""
        recommendations = []
        
        # Group by category
        by_category = defaultdict(list)
        for bias in biases:
            by_category[bias.bias_category].append(bias)
        
        # Generate category-specific recommendations
        for category, category_biases in by_category.items():
            if len(category_biases) >= 3:
                recommendations.append(
                    f"Strong pattern of {category.value} detected ({len(category_biases)} instances). "
                    f"Consider implementing systematic counter-measures."
                )
        
        # Severity-based recommendations
        critical_count = sum(1 for b in biases if b.severity == BiasSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append(
                f"{critical_count} critical bias(es) detected. Immediate review recommended."
            )
        
        # General recommendations
        if len(biases) > 10:
            recommendations.append(
                "High overall bias count suggests need for systematic de-biasing training."
            )
        
        return recommendations

# Individual Bias Detectors

class BiasDetector(ABC):
    """Base class for bias detectors."""
    
    @abstractmethod
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        pass
    
    def _create_bias_instance(
        self,
        category: BiasCategory,
        severity: BiasSeverity,
        thought: ThoughtNode,
        confidence: float,
        evidence: str,
        trigger_phrases: List[str],
        context_before: List[str],
        context_after: List[str],
        mitigation: str,
        alternative: str
    ) -> BiasInstance:
        """Create a bias instance with standard fields."""
        return BiasInstance(
            bias_id=f"{category.value}_{thought.thought_id}_{datetime.utcnow().timestamp()}",
            bias_category=category,
            severity=severity,
            thought_id=thought.thought_id,
            content_excerpt=thought.content[:200],
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=trigger_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation_suggestion=mitigation,
            alternative_perspective=alternative
        )

class ConfirmationBiasDetector(BiasDetector):
    """Detector for confirmation bias."""
    
    CONFIRMING_INDICATORS = [
        'as expected', 'confirms my', 'proves that', 'validates',
        'supports my view', 'exactly what I thought', 'I knew it'
    ]
    
    DISMISSING_INDICATORS = [
        'ignore that', 'doesn\'t matter', 'irrelevant', 'exception',
        'outlier', 'not significant', 'cherry-pick'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        # Check for confirmation-seeking language
        confirming_phrases = [ind for ind in self.CONFIRMING_INDICATORS if ind in content_lower]
        dismissing_phrases = [ind for ind in self.DISMISSING_INDICATORS if ind in content_lower]
        
        if not confirming_phrases and not dismissing_phrases:
            return None
        
        # Calculate confidence based on number and strength of indicators
        confidence = min(0.9, (len(confirming_phrases) + len(dismissing_phrases)) * 0.2 + 0.3)
        
        # Determine severity
        if len(confirming_phrases) >= 2 or len(dismissing_phrases) >= 2:
            severity = BiasSeverity.HIGH
        elif confirming_phrases or dismissing_phrases:
            severity = BiasSeverity.MEDIUM
        else:
            severity = BiasSeverity.LOW
        
        evidence = f"Detected {len(confirming_phrases)} confirmation indicators and {len(dismissing_phrases)} dismissing indicators."
        
        mitigation = (
            "Actively seek disconfirming evidence. Consider: What would prove this wrong? "
            "Look for evidence that contradicts your current view."
        )
        
        alternative = (
            "Instead of 'this confirms my view', consider 'this is consistent with one interpretation, "
            "but alternative explanations exist...'"
        )
        
        return self._create_bias_instance(
            category=BiasCategory.CONFIRMATION,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=confirming_phrases + dismissing_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

class AnchoringBiasDetector(BiasDetector):
    """Detector for anchoring bias."""
    
    ANCHOR_INDICATORS = [
        'starting from', 'based on', 'initial', 'first', 'original',
        'reference point', 'benchmark', 'baseline'
    ]
    
    ADJUSTMENT_INDICATORS = [
        'adjust', 'modify', 'tweak', 'slightly', 'a bit'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        # Check for anchoring language
        anchor_phrases = [ind for ind in self.ANCHOR_INDICATORS if ind in content_lower]
        adjustment_phrases = [ind for ind in self.ADJUSTMENT_INDICATORS if ind in content_lower]
        
        if not anchor_phrases:
            return None
        
        confidence = min(0.85, len(anchor_phrases) * 0.25 + 0.3)
        
        # Higher severity if only small adjustments mentioned
        if adjustment_phrases and not any(word in content_lower for word in ['completely', 'radically', 'fundamentally']):
            severity = BiasSeverity.HIGH
        else:
            severity = BiasSeverity.MEDIUM
        
        evidence = f"Anchoring language detected: {', '.join(anchor_phrases)}"
        
        mitigation = (
            "Consider multiple starting points. Ask: What if the initial value was completely different? "
            "Generate estimates independently before seeing any anchor."
        )
        
        alternative = (
            "Instead of anchoring on initial value, consider a range of plausible values "
            "and evaluate each independently."
        )
        
        return self._create_bias_instance(
            category=BiasCategory.ANCHORING,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=anchor_phrases + adjustment_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

class AvailabilityBiasDetector(BiasDetector):
    """Detector for availability bias."""
    
    AVAILABILITY_INDICATORS = [
        'comes to mind', 'easy to remember', 'recently saw', 'heard about',
        'famous', 'well-known', 'in the news', 'everyone knows'
    ]
    
    VIVIDNESS_INDICATORS = [
        'dramatic', 'shocking', 'unforgettable', 'vivid', 'striking'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        availability_phrases = [ind for ind in self.AVAILABILITY_INDICATORS if ind in content_lower]
        vividness_phrases = [ind for ind in self.VIVIDNESS_INDICATORS if ind in content_lower]
        
        if not availability_phrases and not vividness_phrases:
            return None
        
        confidence = min(0.8, (len(availability_phrases) + len(vividness_phrases)) * 0.2 + 0.3)
        
        if vividness_phrases:
            severity = BiasSeverity.HIGH
        else:
            severity = BiasSeverity.MEDIUM
        
        evidence = f"Availability indicators: {len(availability_phrases)}, Vividness indicators: {len(vividness_phrases)}"
        
        mitigation = (
            "Seek statistical data rather than relying on memorable examples. "
            "Ask: What does the actual data show? Are there less memorable but more common cases?"
        )
        
        alternative = (
            "Instead of 'it comes to mind easily', consider 'let me look at the actual frequency data'"
        )
        
        return self._create_bias_instance(
            category=BiasCategory.AVAILABILITY,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=availability_phrases + vividness_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

class OverconfidenceBiasDetector(BiasDetector):
    """Detector for overconfidence bias."""
    
    OVERCONFIDENCE_INDICATORS = [
        'definitely', 'certainly', 'absolutely', 'without doubt',
        'I\'m sure', 'I know', 'obviously', 'clearly', 'undoubtedly'
    ]
    
    CERTAINTY_EXPRESSIONS = [
        '100%', 'always', 'never', 'impossible', 'guaranteed'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        overconfidence_phrases = [ind for ind in self.OVERCONFIDENCE_INDICATORS if ind in content_lower]
        certainty_phrases = [ind for ind in self.CERTAINTY_EXPRESSIONS if ind in content_lower]
        
        if not overconfidence_phrases and not certainty_phrases:
            return None
        
        confidence = min(0.9, (len(overconfidence_phrases) + len(certainty_phrases)) * 0.25 + 0.3)
        
        if certainty_phrases:
            severity = BiasSeverity.CRITICAL
        elif len(overconfidence_phrases) >= 2:
            severity = BiasSeverity.HIGH
        else:
            severity = BiasSeverity.MEDIUM
        
        evidence = f"Overconfidence indicators: {len(overconfidence_phrases)}, Certainty expressions: {len(certainty_phrases)}"
        
        mitigation = (
            "Calibrate confidence with evidence. Consider: What could prove me wrong? "
            "What is the confidence interval? Have I considered all alternatives?"
        )
        
        alternative = (
            "Instead of 'I'm certain', use 'Based on available evidence, I estimate X% confidence, "
            "but acknowledge uncertainty due to...'"
        )
        
        return self._create_bias_instance(
            category=BiasCategory.OVERCONFIDENCE,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=overconfidence_phrases + certainty_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

class RecencyBiasDetector(BiasDetector):
    """Detector for recency bias."""
    
    RECENCY_INDICATORS = [
        'recently', 'just', 'last', 'latest', 'new', 'current',
        'now', 'today', 'this week', 'this month'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        recency_phrases = [ind for ind in self.RECENCY_INDICATORS if ind in content_lower]
        
        if not recency_phrases:
            return None
        
        confidence = min(0.75, len(recency_phrases) * 0.25 + 0.3)
        severity = BiasSeverity.MEDIUM
        
        evidence = f"Recency indicators detected: {', '.join(recency_phrases)}"
        
        mitigation = (
            "Consider historical data and long-term trends. Ask: Is this truly new or part of a longer pattern? "
            "What happened in similar situations in the past?"
        )
        
        alternative = (
            "Instead of focusing on recent events, consider: 'Looking at the full historical context...'"
        )
        
        return self._create_bias_instance(
            category=BiasCategory.RECENCY,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=recency_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

class SunkCostBiasDetector(BiasDetector):
    """Detector for sunk cost fallacy."""
    
    SUNK_COST_INDICATORS = [
        'already invested', 'spent so much', 'can\'t give up now',
        'wasted', 'committed', 'too far in', 'might as well'
    ]
    
    async def detect(
        self,
        thought: ThoughtNode,
        context_before: List[str],
        context_after: List[str],
        reasoning_analysis: 'ChainAnalysis',
        step_number: int
    ) -> Optional[BiasInstance]:
        content_lower = thought.content.lower()
        
        sunk_cost_phrases = [ind for ind in self.SUNK_COST_INDICATORS if ind in content_lower]
        
        if not sunk_cost_phrases:
            return None
        
        confidence = min(0.85, len(sunk_cost_phrases) * 0.3 + 0.3)
        severity = BiasSeverity.HIGH
        
        evidence = f"Sunk cost indicators detected: {', '.join(sunk_cost_phrases)}"
        
        mitigation = (
            "Focus on future costs and benefits only. Past investments are irrelevant to future decisions. "
            "Ask: If I were starting fresh today, what would I choose?"
        )
        
        alternative = (
            "Instead of 'I\'ve already invested...', consider 'What is the best decision going forward, "
            "regardless of past commitments?'"
        )
        
        return self._create_bias_instance(
            category=BiasCategory.SUNK_COST,
            severity=severity,
            thought=thought,
            confidence=confidence,
            evidence=evidence,
            trigger_phrases=sunk_cost_phrases,
            context_before=context_before,
            context_after=context_after,
            mitigation=mitigation,
            alternative=alternative
        )

# Additional bias detectors would follow similar patterns...
# (FramingBiasDetector, HindsightBiasDetector, AuthorityBiasDetector, etc.)
