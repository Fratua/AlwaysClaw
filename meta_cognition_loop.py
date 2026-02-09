"""
Advanced Meta-Cognition Loop Implementation
Recursive Self-Improvement System for Windows 10 OpenClaw AI Agent

This module implements the 15th agentic loop - the Advanced Meta-Cognition Loop
for recursive self-improvement, deep self-reflection, and cognitive evolution.

Author: AI Systems Architect
Version: 1.0.0
Platform: Windows 10 / Python 3.11+
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union
)
from collections import deque
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ReflectionType(Enum):
    """Types of reflection that can be performed"""
    POST_TASK = "post_task"
    POST_ERROR = "post_error"
    PERIODIC_REVIEW = "periodic_review"
    TRIGGERED = "triggered"
    DEEP = "deep"
    ARCHITECTURAL = "architectural"


class ReflectionDepth(Enum):
    """Depth levels for reflection"""
    DESCRIPTIVE = 1
    EMOTIONAL = 2
    COGNITIVE = 3
    EVALUATIVE = 4
    STRATEGIC = 5
    TRANSFORMATIVE = 6


class BiasType(Enum):
    """Types of cognitive biases that can be detected"""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    FRAMING = "framing"
    RECENCY = "recency"
    SUNK_COST = "sunk_cost"
    GROUPTHINK = "groupthink"
    HALO_EFFECT = "halo_effect"
    STATUS_QUO = "status_quo"


class EvolutionStatus(Enum):
    """Status of architecture evolution"""
    SUCCESS = "success"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    PENDING = "pending"


class TriggerPriority(Enum):
    """Priority levels for meta-cognition triggers"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class StepType(Enum):
    """Types of reasoning steps"""
    INFERENCE = "inference"
    RETRIEVAL = "retrieval"
    CALCULATION = "calculation"
    PLANNING = "planning"
    DECISION = "decision"
    PERCEPTION = "perception"
    ACTION = "action"
    REFLECTION = "reflection"


class ValidationStatus(Enum):
    """Validation status for reasoning steps"""
    VALIDATED = "validated"
    UNCERTAIN = "uncertain"
    CONTESTED = "contested"
    INVALID = "invalid"
    UNCHECKED = "unchecked"


class PatternCategory(Enum):
    """Categories of thinking patterns"""
    PRODUCTIVE = "productive"
    NEUTRAL = "neutral"
    PROBLEMATIC = "problematic"
    EVOLVING = "evolving"


class ModificationType(Enum):
    """Types of pattern modifications"""
    ENHANCEMENT = "enhancement"
    REDUCTION = "reduction"
    REPLACEMENT = "replacement"
    ADDITION = "addition"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CognitiveState:
    """Represents the cognitive state at a point in time"""
    timestamp: datetime
    active_goals: List[str]
    working_memory: Dict[str, Any]
    emotional_valence: float = 0.0  # -1 to 1
    arousal_level: float = 0.5  # 0 to 1
    confidence_level: float = 0.5  # 0 to 1
    attention_focus: Optional[str] = None


@dataclass
class Evidence:
    """Evidence supporting or contesting a reasoning step"""
    source: str
    content: str
    strength: float  # 0 to 1
    type: str  # "supporting" or "contesting"
    reliability: float = 0.5  # 0 to 1


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain"""
    step_number: int
    step_type: StepType
    input_state: CognitiveState
    output_state: CognitiveState
    premise: str
    operation: str
    conclusion: str
    confidence: float
    time_taken_ms: int
    alternative_considered: bool = False
    validation_status: ValidationStatus = ValidationStatus.UNCHECKED
    supporting_evidence: List[Evidence] = field(default_factory=list)
    counter_evidence: List[Evidence] = field(default_factory=list)


@dataclass
class UncertaintyPoint:
    """Point of uncertainty in reasoning"""
    step_number: int
    description: str
    uncertainty_type: str
    magnitude: float  # 0 to 1
    resolution: Optional[str] = None


@dataclass
class DecisionPoint:
    """Point where a decision was made"""
    step_number: int
    decision: str
    alternatives: List[str]
    criteria_used: List[str]
    confidence: float


@dataclass
class Memory:
    """Memory entry"""
    memory_id: str
    content: Any
    memory_type: str
    timestamp: datetime
    importance: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class ToolInvocation:
    """Record of tool usage"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    execution_time_ms: int
    success: bool


@dataclass
class KnowledgeSource:
    """External knowledge source"""
    source_id: str
    source_type: str
    content: Any
    credibility: float
    retrieval_time_ms: int


@dataclass
class ReasoningTrace:
    """Complete capture of reasoning process"""
    trace_id: str
    timestamp: datetime
    task_description: str
    steps: List[ReasoningStep]
    initial_state: CognitiveState
    final_state: CognitiveState
    confidence_trajectory: List[float] = field(default_factory=list)
    uncertainty_points: List[UncertaintyPoint] = field(default_factory=list)
    decision_points: List[DecisionPoint] = field(default_factory=list)
    retrieved_memories: List[Memory] = field(default_factory=list)
    used_tools: List[ToolInvocation] = field(default_factory=list)
    external_knowledge: List[KnowledgeSource] = field(default_factory=list)
    backtrack_count: int = 0
    revision_count: int = 0


@dataclass
class AccuracyMetrics:
    """Metrics related to correctness and accuracy"""
    factual_correctness: float = 0.0
    logical_validity: float = 0.0
    completeness: float = 0.0
    precision: float = 0.0
    calibration_error: float = 0.0


@dataclass
class EfficiencyMetrics:
    """Metrics related to resource usage"""
    time_to_solution: int = 0
    token_efficiency: float = 0.0
    computational_cost: float = 0.0
    memory_usage: float = 0.0
    api_calls: int = 0


@dataclass
class QualityMetrics:
    """Metrics related to output quality"""
    response_coherence: float = 0.0
    clarity_score: float = 0.0
    helpfulness_score: float = 0.0
    creativity_score: float = 0.0
    depth_score: float = 0.0


@dataclass
class MetaCognitiveMetrics:
    """Metrics related to self-awareness and learning"""
    self_correction_rate: float = 0.0
    confidence_accuracy_correlation: float = 0.0
    reflection_depth: float = 0.0
    bias_detection_rate: float = 0.0
    learning_transfer_score: float = 0.0


@dataclass
class ProcessMetrics:
    """Metrics related to reasoning process"""
    reasoning_steps: int = 0
    backtrack_count: int = 0
    revision_count: int = 0
    exploration_breadth: float = 0.0
    decision_quality: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: datetime
    task_id: str
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    metacognition: MetaCognitiveMetrics = field(default_factory=MetaCognitiveMetrics)
    process: ProcessMetrics = field(default_factory=ProcessMetrics)


@dataclass
class DetectedBias:
    """Information about a detected cognitive bias"""
    bias_type: BiasType
    confidence: float
    indicators: Dict[str, float]
    affected_steps: List[int]
    mitigation_strategies: List[str]
    severity: str = "medium"


@dataclass
class BiasDetectionResult:
    """Result of bias detection"""
    detected_biases: List[DetectedBias]
    overall_bias_risk: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectionPhase:
    """Single phase of deep reflection"""
    phase_name: str
    questions: List[str]
    insights: List[str]
    emotional_tone: Optional[str] = None
    confidence: float = 0.5


@dataclass
class DeepReflection:
    """Complete deep reflection result"""
    experience_id: str
    reflection_type: ReflectionType
    depth: ReflectionDepth
    phases: Dict[str, ReflectionPhase]
    insights: List[str]
    action_items: List[str]
    learning_outcomes: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0


@dataclass
class LevelAnalysis:
    """Analysis at a specific meta-cognitive level"""
    level: int
    logical_coherence: float
    assumption_identification: List[str]
    inference_quality: float
    knowledge_gaps: List[str]
    alternative_paths: List[str]
    confidence_calibration: float
    potential_errors: List[str]


@dataclass
class RecursiveReflectionResult:
    """Result of recursive reflection"""
    level: int
    analysis: LevelAnalysis
    meta_analysis: Optional['RecursiveReflectionResult']
    synthesis: str
    confidence: float = 0.5


@dataclass
class Pattern:
    """Thinking pattern definition"""
    name: str
    description: str
    category: PatternCategory
    typical_use_cases: List[str]
    indicators: List[str]
    strengths: List[str]
    weaknesses: List[str]
    modification_strategies: List[str]
    frequency: float = 0.0
    success_rate: float = 0.5
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatternModification:
    """A pattern modification to be applied"""
    pattern_name: str
    modification_type: ModificationType
    description: str
    expected_impact: float
    implementation: Callable
    verification_criteria: List[str]


@dataclass
class LearningStrategy:
    """Optimized learning strategy"""
    approach: str
    parameters: Dict[str, Any]
    predicted_outcomes: Dict[str, float]
    confidence: float
    adaptation_triggers: List[str]


@dataclass
class ArchitectureChange:
    """A proposed architecture change"""
    component_id: str
    change_type: str
    description: str
    implementation: Callable
    rollback_procedure: Callable
    risk_assessment: Dict[str, float]


@dataclass
class ArchitectureEvolution:
    """Result of architecture evolution"""
    status: EvolutionStatus
    changes_applied: List[ArchitectureChange]
    validation_results: Dict[str, bool]
    performance_impact: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    reason: Optional[str] = None


@dataclass
class MetaCognitionConfig:
    """Configuration for meta-cognition system"""
    max_reflection_depth: int = 4
    reflection_convergence_threshold: float = 0.95
    max_reflection_iterations: int = 5
    metrics_collection_interval_ms: int = 1000
    baseline_establishment_tasks: int = 100
    performance_window_size: int = 1000
    default_reflection_depth: ReflectionDepth = ReflectionDepth.STRATEGIC
    deep_reflection_trigger_threshold: float = 0.3
    periodic_reflection_interval_hours: float = 1.0
    evolution_safety_checks: bool = True
    max_evolution_changes_per_cycle: int = 3
    evolution_cooldown_period_hours: float = 24.0
    pattern_analysis_window_size: int = 100
    modification_verification_periods: int = 10
    meta_learning_enabled: bool = True
    adaptive_learning_rate: bool = True
    learning_rate_bounds: Tuple[float, float] = (0.001, 0.1)
    bias_detection_enabled: bool = True
    bias_detection_threshold: float = 0.6
    mandatory_mitigation_for_critical: bool = True
    episodic_memory_retention_days: int = 30
    meta_memory_compaction_interval_hours: int = 24
    experience_replay_batch_size: int = 32
    cycle_timeout_seconds: float = 30.0
    parallel_component_execution: bool = True


@dataclass
class MetaCognitionTrigger:
    """Trigger for meta-cognition cycle"""
    trigger_type: str
    priority: TriggerPriority
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaCognitionResult:
    """Complete result of meta-cognition cycle"""
    trigger: MetaCognitionTrigger
    cycle_duration_ms: float
    performance_analysis: PerformanceSnapshot
    bias_detection: BiasDetectionResult
    recursive_reflection: RecursiveReflectionResult
    deep_reflection: Optional[DeepReflection]
    pattern_modifications: List[PatternModification]
    learning_optimization: Optional[LearningStrategy]
    architecture_evolution: Optional[ArchitectureEvolution]
    insights: List[str]
    action_items: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# CORE CLASSES
# =============================================================================

class CognitivePerformanceMonitor:
    """
    Real-time tracking and measurement of cognitive performance
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=config.performance_window_size)
        self.baseline_metrics: Optional[PerformanceSnapshot] = None
        self.baseline_established = False
        
    async def capture_metrics(self, reasoning_trace: ReasoningTrace) -> PerformanceSnapshot:
        """Capture comprehensive performance metrics"""
        
        # Calculate accuracy metrics
        accuracy = AccuracyMetrics(
            factual_correctness=self._assess_factual_correctness(reasoning_trace),
            logical_validity=self._assess_logical_validity(reasoning_trace),
            completeness=self._assess_completeness(reasoning_trace),
            precision=self._assess_precision(reasoning_trace),
            calibration_error=self._calculate_calibration_error(reasoning_trace)
        )
        
        # Calculate efficiency metrics
        efficiency = EfficiencyMetrics(
            time_to_solution=sum(s.time_taken_ms for s in reasoning_trace.steps),
            token_efficiency=self._calculate_token_efficiency(reasoning_trace),
            computational_cost=len(reasoning_trace.steps) * 0.1,
            memory_usage=len(reasoning_trace.final_state.working_memory) * 0.01,
            api_calls=len(reasoning_trace.used_tools)
        )
        
        # Calculate quality metrics
        quality = QualityMetrics(
            response_coherence=self._assess_coherence(reasoning_trace),
            clarity_score=self._assess_clarity(reasoning_trace),
            helpfulness_score=self._assess_helpfulness(reasoning_trace),
            creativity_score=self._assess_creativity(reasoning_trace),
            depth_score=self._assess_depth(reasoning_trace)
        )
        
        # Calculate meta-cognitive metrics
        metacognition = MetaCognitiveMetrics(
            self_correction_rate=reasoning_trace.revision_count / max(len(reasoning_trace.steps), 1),
            confidence_accuracy_correlation=self._calculate_confidence_correlation(reasoning_trace),
            reflection_depth=self._measure_reflection_depth(reasoning_trace),
            bias_detection_rate=self._calculate_bias_detection_rate(reasoning_trace),
            learning_transfer_score=self._calculate_learning_transfer(reasoning_trace),
        )
        
        # Calculate process metrics
        process = ProcessMetrics(
            reasoning_steps=len(reasoning_trace.steps),
            backtrack_count=reasoning_trace.backtrack_count,
            revision_count=reasoning_trace.revision_count,
            exploration_breadth=self._measure_exploration_breadth(reasoning_trace),
            decision_quality=self._assess_decision_quality(reasoning_trace)
        )
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            task_id=reasoning_trace.trace_id,
            accuracy=accuracy,
            efficiency=efficiency,
            quality=quality,
            metacognition=metacognition,
            process=process
        )
        
        self.metrics_history.append(snapshot)
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.metrics_history) >= self.config.baseline_establishment_tasks:
            self._establish_baseline()
        
        return snapshot
    
    def _assess_factual_correctness(self, trace: ReasoningTrace) -> float:
        """Assess factual correctness of reasoning"""
        if not trace.steps:
            return 0.5
        
        validated_steps = sum(
            1 for s in trace.steps 
            if s.validation_status == ValidationStatus.VALIDATED
        )
        return validated_steps / len(trace.steps)
    
    def _assess_logical_validity(self, trace: ReasoningTrace) -> float:
        """Assess logical validity of reasoning chain"""
        if len(trace.steps) < 2:
            return 0.5
        
        valid_transitions = 0
        for i in range(1, len(trace.steps)):
            prev = trace.steps[i-1]
            curr = trace.steps[i]
            # Check if conclusion follows from premise
            if curr.premise in prev.conclusion or prev.conclusion in curr.premise:
                valid_transitions += 1
        
        return valid_transitions / (len(trace.steps) - 1)
    
    def _assess_completeness(self, trace: ReasoningTrace) -> float:
        """Assess completeness of reasoning"""
        # Check if all decision points have alternatives considered
        if not trace.decision_points:
            return 0.5
        
        complete_decisions = sum(
            1 for dp in trace.decision_points 
            if len(dp.alternatives) >= 2
        )
        return complete_decisions / len(trace.decision_points)
    
    def _assess_precision(self, trace: ReasoningTrace) -> float:
        """Assess precision (lack of unnecessary information)"""
        # Measure efficiency of reasoning steps
        if not trace.steps:
            return 0.5
        
        # Check for redundant steps
        unique_operations = len(set(s.operation for s in trace.steps))
        return unique_operations / len(trace.steps)
    
    def _calculate_calibration_error(self, trace: ReasoningTrace) -> float:
        """Calculate confidence calibration error"""
        if not trace.steps:
            return 0.0
        
        # Compare confidence to actual validation success
        errors = []
        for step in trace.steps:
            if step.validation_status == ValidationStatus.VALIDATED:
                actual = 1.0
            elif step.validation_status == ValidationStatus.INVALID:
                actual = 0.0
            else:
                continue
            errors.append(abs(step.confidence - actual))
        
        return np.mean(errors) if errors else 0.0
    
    def _calculate_token_efficiency(self, trace: ReasoningTrace) -> float:
        """Calculate token efficiency based on reasoning density"""
        if not trace.steps:
            return 0.5

        total_chars = sum(len(s.conclusion) + len(s.premise) for s in trace.steps)
        validated = sum(
            1 for s in trace.steps
            if s.validation_status == ValidationStatus.VALIDATED
        )
        if total_chars == 0:
            return 0.5
        # Efficiency = useful output per unit of reasoning text
        return min(1.0, (validated / len(trace.steps)) * (1000.0 / max(total_chars, 1)) * 10)
    
    def _assess_coherence(self, trace: ReasoningTrace) -> float:
        """Assess coherence of reasoning"""
        if len(trace.steps) < 2:
            return 0.5
        
        # Check state transitions are smooth
        coherent_transitions = 0
        for i in range(1, len(trace.steps)):
            prev_state = trace.steps[i-1].output_state
            curr_state = trace.steps[i].input_state
            
            # Check for state consistency
            if prev_state.attention_focus == curr_state.attention_focus:
                coherent_transitions += 1
        
        return coherent_transitions / (len(trace.steps) - 1)
    
    def _assess_clarity(self, trace: ReasoningTrace) -> float:
        """Assess clarity of reasoning"""
        if not trace.steps:
            return 0.5
        
        # Measure clarity by step description length consistency
        lengths = [len(s.conclusion) for s in trace.steps]
        if not lengths:
            return 0.5
        
        # Lower variance in length suggests more consistent clarity
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        # Normalize - lower variance is better
        clarity = 1.0 / (1.0 + variance / (mean_length ** 2))
        return min(1.0, clarity)
    
    def _assess_creativity(self, trace: ReasoningTrace) -> float:
        """Assess creativity in reasoning"""
        # Measure by diversity of operations used
        if not trace.steps:
            return 0.0
        
        operations = [s.operation for s in trace.steps]
        unique_ops = len(set(operations))
        return min(1.0, unique_ops / len(operations) * 2)
    
    def _assess_depth(self, trace: ReasoningTrace) -> float:
        """Assess depth of reasoning"""
        # Measure by reasoning chain length and recursion
        if not trace.steps:
            return 0.0
        
        depth_score = len(trace.steps) / 20.0  # Normalize to ~20 steps
        reflection_bonus = sum(
            0.1 for s in trace.steps 
            if s.step_type == StepType.REFLECTION
        )
        
        return min(1.0, depth_score + reflection_bonus)
    
    def _calculate_confidence_correlation(self, trace: ReasoningTrace) -> float:
        """Calculate correlation between confidence and accuracy"""
        confidences = []
        accuracies = []
        
        for step in trace.steps:
            confidences.append(step.confidence)
            if step.validation_status == ValidationStatus.VALIDATED:
                accuracies.append(1.0)
            elif step.validation_status == ValidationStatus.INVALID:
                accuracies.append(0.0)
            else:
                accuracies.append(0.5)
        
        if len(confidences) < 2:
            return 0.0
        
        return np.corrcoef(confidences, accuracies)[0, 1]
    
    def _measure_reflection_depth(self, trace: ReasoningTrace) -> float:
        """Measure depth of reflection in reasoning"""
        reflection_steps = sum(
            1 for s in trace.steps 
            if s.step_type == StepType.REFLECTION
        )
        return min(1.0, reflection_steps / max(len(trace.steps) * 0.2, 1))

    def _assess_helpfulness(self, trace: ReasoningTrace) -> float:
        """Assess helpfulness based on tool usage, uncertainty resolution, and completeness."""
        if not trace.steps:
            return 0.5
        score = 0.0
        total_weight = 0.0

        # Tool usage effectiveness (did tools get used and produce results?)
        if trace.used_tools:
            tool_weight = 0.3
            total_weight += tool_weight
            # More tools used = more helpful (up to a point)
            tool_score = min(len(trace.used_tools) / max(len(trace.steps), 1), 1.0)
            score += tool_weight * tool_score

        # Uncertainty resolution (did confidence increase over steps?)
        confidences = [s.confidence for s in trace.steps if hasattr(s, 'confidence') and s.confidence > 0]
        if len(confidences) >= 2:
            uncertainty_weight = 0.3
            total_weight += uncertainty_weight
            improvement = confidences[-1] - confidences[0]
            score += uncertainty_weight * max(0.0, min(1.0, 0.5 + improvement))

        # Final confidence as proxy for conclusion completeness
        if confidences:
            completion_weight = 0.4
            total_weight += completion_weight
            score += completion_weight * confidences[-1]

        return score / total_weight if total_weight > 0 else 0.5

    def _calculate_bias_detection_rate(self, trace: ReasoningTrace) -> float:
        """Calculate ratio of evidence-bearing steps where counter-evidence was addressed."""
        if not trace.steps:
            return 0.5
        evidence_steps = 0
        addressed_steps = 0
        for step in trace.steps:
            content = getattr(step, 'content', '') or ''
            # Check if step involves evidence evaluation
            evidence_keywords = ['evidence', 'suggests', 'indicates', 'shows', 'data', 'result']
            if any(kw in content.lower() for kw in evidence_keywords):
                evidence_steps += 1
                # Check if counter-evidence or alternatives were considered
                counter_keywords = ['however', 'alternatively', 'but', 'counter', 'on the other hand', 'limitation']
                if any(kw in content.lower() for kw in counter_keywords):
                    addressed_steps += 1
        if evidence_steps == 0:
            return 0.5
        return addressed_steps / evidence_steps

    def _calculate_learning_transfer(self, trace: ReasoningTrace) -> float:
        """Measure confidence improvement after retrieval/learning steps."""
        if len(trace.steps) < 2:
            return 0.5
        retrieval_improvements = []
        for i, step in enumerate(trace.steps):
            step_type = getattr(step, 'step_type', None) or getattr(step, 'type', '')
            content = getattr(step, 'content', '') or ''
            is_retrieval = (
                str(step_type).lower() in ('retrieval', 'search', 'lookup', 'recall')
                or any(kw in content.lower() for kw in ['retrieved', 'found', 'recalled', 'looked up'])
            )
            if is_retrieval and i + 1 < len(trace.steps):
                conf_before = getattr(step, 'confidence', 0.5)
                conf_after = getattr(trace.steps[i + 1], 'confidence', 0.5)
                if conf_after > conf_before:
                    retrieval_improvements.append(conf_after - conf_before)
        if not retrieval_improvements:
            return 0.5
        avg_improvement = sum(retrieval_improvements) / len(retrieval_improvements)
        return min(1.0, 0.5 + avg_improvement * 2)

    def _measure_exploration_breadth(self, trace: ReasoningTrace) -> float:
        """Measure exploration breadth"""
        if not trace.decision_points:
            return 0.5
        
        avg_alternatives = np.mean([
            len(dp.alternatives) for dp in trace.decision_points
        ])
        return min(1.0, avg_alternatives / 5.0)
    
    def _assess_decision_quality(self, trace: ReasoningTrace) -> float:
        """Assess quality of decisions made"""
        if not trace.decision_points:
            return 0.5
        
        # Average confidence weighted by number of alternatives considered
        qualities = []
        for dp in trace.decision_points:
            alt_score = min(1.0, len(dp.alternatives) / 3.0)
            qualities.append(dp.confidence * alt_score)
        
        return np.mean(qualities) if qualities else 0.5
    
    def _establish_baseline(self):
        """Establish baseline metrics from history"""
        if len(self.metrics_history) < self.config.baseline_establishment_tasks:
            return
        
        # Calculate average metrics
        recent_metrics = list(self.metrics_history)[-self.config.baseline_establishment_tasks:]
        
        self.baseline_metrics = PerformanceSnapshot(
            timestamp=datetime.now(),
            task_id="baseline",
            accuracy=AccuracyMetrics(
                factual_correctness=np.mean([m.accuracy.factual_correctness for m in recent_metrics]),
                logical_validity=np.mean([m.accuracy.logical_validity for m in recent_metrics]),
                completeness=np.mean([m.accuracy.completeness for m in recent_metrics]),
                precision=np.mean([m.accuracy.precision for m in recent_metrics]),
                calibration_error=np.mean([m.accuracy.calibration_error for m in recent_metrics])
            ),
            efficiency=EfficiencyMetrics(
                time_to_solution=int(np.mean([m.efficiency.time_to_solution for m in recent_metrics])),
                token_efficiency=np.mean([m.efficiency.token_efficiency for m in recent_metrics]),
                computational_cost=np.mean([m.efficiency.computational_cost for m in recent_metrics]),
                memory_usage=np.mean([m.efficiency.memory_usage for m in recent_metrics]),
                api_calls=int(np.mean([m.efficiency.api_calls for m in recent_metrics]))
            ),
            quality=QualityMetrics(
                response_coherence=np.mean([m.quality.response_coherence for m in recent_metrics]),
                clarity_score=np.mean([m.quality.clarity_score for m in recent_metrics]),
                helpfulness_score=np.mean([m.quality.helpfulness_score for m in recent_metrics]),
                creativity_score=np.mean([m.quality.creativity_score for m in recent_metrics]),
                depth_score=np.mean([m.quality.depth_score for m in recent_metrics])
            ),
            metacognition=MetaCognitiveMetrics(
                self_correction_rate=np.mean([m.metacognition.self_correction_rate for m in recent_metrics]),
                confidence_accuracy_correlation=np.mean([m.metacognition.confidence_accuracy_correlation for m in recent_metrics]),
                reflection_depth=np.mean([m.metacognition.reflection_depth for m in recent_metrics]),
                bias_detection_rate=np.mean([m.metacognition.bias_detection_rate for m in recent_metrics]),
                learning_transfer_score=np.mean([m.metacognition.learning_transfer_score for m in recent_metrics])
            ),
            process=ProcessMetrics(
                reasoning_steps=int(np.mean([m.process.reasoning_steps for m in recent_metrics])),
                backtrack_count=int(np.mean([m.process.backtrack_count for m in recent_metrics])),
                revision_count=int(np.mean([m.process.revision_count for m in recent_metrics])),
                exploration_breadth=np.mean([m.process.exploration_breadth for m in recent_metrics]),
                decision_quality=np.mean([m.process.decision_quality for m in recent_metrics])
            )
        )
        
        self.baseline_established = True
        logger.info("Baseline metrics established")
    
    def get_performance_trend(self, metric_name: str, window: int = 10) -> float:
        """Get trend for a specific metric"""
        if len(self.metrics_history) < window:
            return 0.0
        
        recent = list(self.metrics_history)[-window:]
        values = []
        
        for m in recent:
            parts = metric_name.split('.')
            value = m
            for part in parts:
                value = getattr(value, part)
            values.append(value)
        
        if len(values) < 2:
            return 0.0
        
        # Calculate trend using linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class CognitiveBiasDetector:
    """
    Detects and analyzes cognitive biases in reasoning traces
    """
    
    # Bias detection patterns
    BIAS_PATTERNS = {
        BiasType.CONFIRMATION: {
            'indicators': [
                'selective_evidence_gathering',
                'dismissing_contradictory_evidence',
                'asymmetric_evidence_evaluation'
            ],
            'threshold': 0.6
        },
        BiasType.ANCHORING: {
            'indicators': [
                'over_reliance_on_initial_information',
                'insufficient_adjustment_from_anchor',
                'early_information_overweighting'
            ],
            'threshold': 0.6
        },
        BiasType.AVAILABILITY: {
            'indicators': [
                'overweighting_recent_events',
                'overweighting_vivid_examples',
                'neglecting_base_rates'
            ],
            'threshold': 0.6
        },
        BiasType.OVERCONFIDENCE: {
            'indicators': [
                'confidence_exceeds_accuracy',
                'insufficient_uncertainty_expression',
                'premature_decision_making'
            ],
            'threshold': 0.6
        },
        BiasType.FRAMING: {
            'indicators': [
                'decision_changes_with_equivalent_frames',
                'loss_aversion_asymmetry',
                'reference_point_dependence'
            ],
            'threshold': 0.5
        }
    }
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.detection_history: List[BiasDetectionResult] = []
    
    async def detect_biases(self, trace: ReasoningTrace) -> BiasDetectionResult:
        """Detect cognitive biases in reasoning trace"""
        detected_biases = []
        
        for bias_type, pattern in self.BIAS_PATTERNS.items():
            detection = await self._check_bias(trace, bias_type, pattern)
            if detection:
                detected_biases.append(detection)
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(detected_biases)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detected_biases)
        
        result = BiasDetectionResult(
            detected_biases=detected_biases,
            overall_bias_risk=overall_risk,
            recommendations=recommendations
        )
        
        self.detection_history.append(result)
        return result
    
    async def _check_bias(
        self, 
        trace: ReasoningTrace, 
        bias_type: BiasType, 
        pattern: Dict
    ) -> Optional[DetectedBias]:
        """Check for a specific bias type"""
        indicators = {}
        affected_steps = []
        
        if bias_type == BiasType.CONFIRMATION:
            indicators, affected_steps = self._check_confirmation_bias(trace)
        elif bias_type == BiasType.ANCHORING:
            indicators, affected_steps = self._check_anchoring_bias(trace)
        elif bias_type == BiasType.AVAILABILITY:
            indicators, affected_steps = self._check_availability_bias(trace)
        elif bias_type == BiasType.OVERCONFIDENCE:
            indicators, affected_steps = self._check_overconfidence_bias(trace)
        elif bias_type == BiasType.FRAMING:
            indicators, affected_steps = self._check_framing_bias(trace)
        
        # Calculate confidence
        if indicators:
            confidence = np.mean(list(indicators.values()))
            
            if confidence >= pattern['threshold']:
                return DetectedBias(
                    bias_type=bias_type,
                    confidence=confidence,
                    indicators=indicators,
                    affected_steps=affected_steps,
                    mitigation_strategies=self._get_mitigation_strategies(bias_type),
                    severity=self._calculate_severity(confidence, len(affected_steps))
                )
        
        return None
    
    def _check_confirmation_bias(self, trace: ReasoningTrace) -> Tuple[Dict, List]:
        """Check for confirmation bias"""
        indicators = {}
        affected_steps = []
        
        # Check for selective evidence gathering
        for i, step in enumerate(trace.steps):
            if step.supporting_evidence and not step.counter_evidence:
                indicators[f'step_{i}_missing_counter'] = 0.8
                affected_steps.append(i)
            
            # Check evidence ratio
            total_evidence = len(step.supporting_evidence) + len(step.counter_evidence)
            if total_evidence > 0:
                support_ratio = len(step.supporting_evidence) / total_evidence
                if support_ratio > 0.8:
                    indicators[f'step_{i}_support_ratio'] = support_ratio
                    if i not in affected_steps:
                        affected_steps.append(i)
        
        return indicators, affected_steps
    
    def _check_anchoring_bias(self, trace: ReasoningTrace) -> Tuple[Dict, List]:
        """Check for anchoring bias"""
        indicators = {}
        affected_steps = []
        
        if len(trace.steps) < 2:
            return indicators, affected_steps
        
        # Check if early information is over-weighted
        first_step = trace.steps[0]
        for i, step in enumerate(trace.steps[1:], 1):
            # Check if first step's conclusion appears frequently
            if first_step.conclusion in step.premise:
                indicators[f'step_{i}_anchor_reuse'] = 0.7
                affected_steps.append(i)
        
        return indicators, affected_steps
    
    def _check_availability_bias(self, trace: ReasoningTrace) -> Tuple[Dict, List]:
        """Check for availability bias"""
        indicators = {}
        affected_steps = []
        
        # Check for over-reliance on retrieved memories
        for i, step in enumerate(trace.steps):
            if step.step_type == StepType.RETRIEVAL:
                # Check if retrieval was followed by decision without further analysis
                if i + 1 < len(trace.steps):
                    next_step = trace.steps[i + 1]
                    if next_step.step_type == StepType.DECISION:
                        indicators[f'step_{i}_quick_decision'] = 0.75
                        affected_steps.extend([i, i+1])
        
        return indicators, affected_steps
    
    def _check_overconfidence_bias(self, trace: ReasoningTrace) -> Tuple[Dict, List]:
        """Check for overconfidence bias"""
        indicators = {}
        affected_steps = []
        
        for i, step in enumerate(trace.steps):
            # High confidence with low validation
            if step.confidence > 0.8 and step.validation_status == ValidationStatus.UNCHECKED:
                indicators[f'step_{i}_unchecked_high_conf'] = 0.8
                affected_steps.append(i)
            
            # Confidence exceeds actual validation success
            if step.validation_status == ValidationStatus.INVALID and step.confidence > 0.5:
                indicators[f'step_{i}_conf_exceeds_accuracy'] = step.confidence
                affected_steps.append(i)
        
        return indicators, affected_steps
    
    def _check_framing_bias(self, trace: ReasoningTrace) -> Tuple[Dict, List]:
        """Check for framing bias"""
        indicators = {}
        affected_steps = []
        
        # Check decision points for framing effects
        for dp in trace.decision_points:
            # If decision was made with few alternatives, might be framing
            if len(dp.alternatives) < 2:
                indicators[f'decision_{dp.step_number}_limited_alternatives'] = 0.6
                affected_steps.append(dp.step_number)
        
        return indicators, affected_steps
    
    def _get_mitigation_strategies(self, bias_type: BiasType) -> List[str]:
        """Get mitigation strategies for a bias type"""
        strategies = {
            BiasType.CONFIRMATION: [
                'require_devils_advocate',
                'mandate_contrarian_evidence',
                'implement_blind_review'
            ],
            BiasType.ANCHORING: [
                'delay_anchor_exposure',
                'generate_multiple_anchors',
                'implement_structured_adjustment'
            ],
            BiasType.AVAILABILITY: [
                'systematic_data_gathering',
                'base_rate_explicitation',
                'statistical_reasoning_enforcement'
            ],
            BiasType.OVERCONFIDENCE: [
                'confidence_calibration_training',
                'mandate_uncertainty_quantification',
                'implement_pre_mortem_analysis'
            ],
            BiasType.FRAMING: [
                'multiple_frame_analysis',
                'gain_loss_neutralization',
                'reference_point_independence'
            ]
        }
        return strategies.get(bias_type, [])
    
    def _calculate_severity(self, confidence: float, affected_count: int) -> str:
        """Calculate severity of detected bias"""
        score = confidence * (1 + affected_count / 10)
        
        if score > 1.5:
            return "critical"
        elif score > 1.0:
            return "high"
        elif score > 0.7:
            return "medium"
        return "low"
    
    def _calculate_overall_risk(self, detected_biases: List[DetectedBias]) -> float:
        """Calculate overall bias risk"""
        if not detected_biases:
            return 0.0
        
        # Weight by confidence and affected steps
        total_risk = sum(
            db.confidence * (1 + len(db.affected_steps) / 10)
            for db in detected_biases
        )
        
        return min(1.0, total_risk / len(detected_biases))
    
    def _generate_recommendations(self, detected_biases: List[DetectedBias]) -> List[str]:
        """Generate recommendations based on detected biases"""
        recommendations = []
        
        for bias in detected_biases:
            if bias.severity in ["critical", "high"]:
                recommendations.append(
                    f"Priority: Address {bias.bias_type.value} bias - "
                    f"confidence: {bias.confidence:.2f}"
                )
            
            for strategy in bias.mitigation_strategies[:2]:
                recommendations.append(f"  - Apply: {strategy}")
        
        return recommendations


class RecursiveReflectionEngine:
    """
    Multi-level recursive analysis of reasoning processes
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.reflection_history: List[RecursiveReflectionResult] = []
    
    async def recursive_analyze(
        self,
        trace: ReasoningTrace,
        depth: int = 0
    ) -> RecursiveReflectionResult:
        """Perform recursive analysis of reasoning"""
        
        if depth >= self.config.max_reflection_depth:
            # Base case
            return await self._base_case_analysis(trace)
        
        # Analyze at current level
        level_analysis = await self._analyze_at_level(trace, depth)
        
        # Create trace of this analysis for meta-analysis
        meta_trace = self._create_meta_trace(level_analysis, depth)
        
        # Recursive call for meta-analysis
        meta_analysis = await self.recursive_analyze(meta_trace, depth + 1)
        
        # Synthesize results
        synthesis = self._synthesize_analysis(level_analysis, meta_analysis)
        
        result = RecursiveReflectionResult(
            level=depth,
            analysis=level_analysis,
            meta_analysis=meta_analysis,
            synthesis=synthesis,
            confidence=level_analysis.confidence_calibration
        )
        
        self.reflection_history.append(result)
        return result
    
    async def _base_case_analysis(self, trace: ReasoningTrace) -> RecursiveReflectionResult:
        """Base case for recursion"""
        analysis = LevelAnalysis(
            level=self.config.max_reflection_depth,
            logical_coherence=self._assess_coherence(trace),
            assumption_identification=[],
            inference_quality=0.5,
            knowledge_gaps=[],
            alternative_paths=[],
            confidence_calibration=0.5,
            potential_errors=[]
        )
        
        return RecursiveReflectionResult(
            level=self.config.max_reflection_depth,
            analysis=analysis,
            meta_analysis=None,
            synthesis="Base case analysis - maximum depth reached",
            confidence=0.5
        )
    
    async def _analyze_at_level(self, trace: ReasoningTrace, level: int) -> LevelAnalysis:
        """Analyze reasoning at a specific meta-cognitive level"""
        
        # Assess logical coherence
        coherence = self._assess_coherence(trace)
        
        # Identify assumptions
        assumptions = self._identify_assumptions(trace)
        
        # Assess inference quality
        inference_quality = self._assess_inference_quality(trace)
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(trace)
        
        # Generate alternative paths
        alternatives = self._generate_alternatives(trace)
        
        # Calibrate confidence
        confidence_calibration = self._calibrate_confidence(trace)
        
        # Identify potential errors
        potential_errors = self._identify_potential_errors(trace)
        
        return LevelAnalysis(
            level=level,
            logical_coherence=coherence,
            assumption_identification=assumptions,
            inference_quality=inference_quality,
            knowledge_gaps=knowledge_gaps,
            alternative_paths=alternatives,
            confidence_calibration=confidence_calibration,
            potential_errors=potential_errors
        )
    
    def _create_meta_trace(self, analysis: LevelAnalysis, depth: int) -> ReasoningTrace:
        """Create a reasoning trace from analysis for meta-analysis"""
        step = ReasoningStep(
            step_number=0,
            step_type=StepType.REFLECTION,
            input_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=[f"meta_analysis_level_{depth}"],
                working_memory={"analysis": analysis}
            ),
            output_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=[f"meta_analysis_level_{depth + 1}"],
                working_memory={}
            ),
            premise=f"Analysis at level {depth}",
            operation="meta_reflection",
            conclusion=f"Logical coherence: {analysis.logical_coherence}",
            confidence=analysis.confidence_calibration,
            time_taken_ms=100
        )
        
        return ReasoningTrace(
            trace_id=f"meta_trace_{depth}",
            timestamp=datetime.now(),
            task_description=f"Meta-analysis at level {depth}",
            steps=[step],
            initial_state=step.input_state,
            final_state=step.output_state
        )
    
    def _synthesize_analysis(
        self, 
        level_analysis: LevelAnalysis, 
        meta_analysis: RecursiveReflectionResult
    ) -> str:
        """Synthesize level analysis with meta-analysis"""
        synthesis = f"""
        Level {level_analysis.level} Analysis:
        - Logical coherence: {level_analysis.logical_coherence:.2f}
        - Inference quality: {level_analysis.inference_quality:.2f}
        - Confidence calibration: {level_analysis.confidence_calibration:.2f}
        - Assumptions identified: {len(level_analysis.assumption_identification)}
        - Knowledge gaps: {len(level_analysis.knowledge_gaps)}
        - Alternative paths: {len(level_analysis.alternative_paths)}
        - Potential errors: {len(level_analysis.potential_errors)}
        
        Meta-level insights:
        {meta_analysis.synthesis if meta_analysis else "No meta-analysis"}
        """
        return synthesis
    
    def _assess_coherence(self, trace: ReasoningTrace) -> float:
        """Assess logical coherence"""
        if len(trace.steps) < 2:
            return 0.5
        
        valid_transitions = 0
        for i in range(1, len(trace.steps)):
            prev = trace.steps[i-1]
            curr = trace.steps[i]
            
            # Check logical flow
            if curr.premise in prev.conclusion or prev.conclusion in curr.premise:
                valid_transitions += 1
        
        return valid_transitions / (len(trace.steps) - 1)
    
    def _identify_assumptions(self, trace: ReasoningTrace) -> List[str]:
        """Identify implicit assumptions in reasoning"""
        assumptions = []
        
        for step in trace.steps:
            # Look for unsupported premises
            if not step.supporting_evidence and step.step_type == StepType.INFERENCE:
                assumptions.append(f"Step {step.step_number}: {step.premise}")
        
        return assumptions
    
    def _assess_inference_quality(self, trace: ReasoningTrace) -> float:
        """Assess quality of inferences"""
        if not trace.steps:
            return 0.5
        
        inference_steps = [s for s in trace.steps if s.step_type == StepType.INFERENCE]
        if not inference_steps:
            return 0.5
        
        # Quality based on validation and confidence
        qualities = []
        for step in inference_steps:
            if step.validation_status == ValidationStatus.VALIDATED:
                qualities.append(1.0)
            elif step.validation_status == ValidationStatus.INVALID:
                qualities.append(0.0)
            else:
                qualities.append(step.confidence)
        
        return np.mean(qualities)
    
    def _identify_knowledge_gaps(self, trace: ReasoningTrace) -> List[str]:
        """Identify gaps in knowledge"""
        gaps = []
        
        for step in trace.steps:
            # Steps with low confidence may indicate knowledge gaps
            if step.confidence < 0.5:
                gaps.append(f"Step {step.step_number}: Low confidence in {step.operation}")
        
        return gaps
    
    def _generate_alternatives(self, trace: ReasoningTrace) -> List[str]:
        """Generate alternative reasoning paths"""
        alternatives = []
        
        for dp in trace.decision_points:
            for alt in dp.alternatives:
                alternatives.append(f"At step {dp.step_number}: {alt}")
        
        return alternatives
    
    def _calibrate_confidence(self, trace: ReasoningTrace) -> float:
        """Calibrate confidence estimates"""
        if not trace.steps:
            return 0.5
        
        confidences = [s.confidence for s in trace.steps]
        return np.mean(confidences)
    
    def _identify_potential_errors(self, trace: ReasoningTrace) -> List[str]:
        """Identify potential errors in reasoning"""
        errors = []
        
        for step in trace.steps:
            # High confidence with contradictory evidence
            if step.confidence > 0.8 and step.counter_evidence:
                errors.append(f"Step {step.step_number}: High confidence despite counter-evidence")
            
            # Invalid validation status
            if step.validation_status == ValidationStatus.INVALID:
                errors.append(f"Step {step.step_number}: Invalidated step")
        
        return errors


class DeepReflectionEngine:
    """
    Implements deep, structured self-reflection capabilities
    """
    
    REFLECTION_TEMPLATES = {
        ReflectionType.POST_TASK: {
            'descriptive': [
                'What exactly happened during this task?',
                'What were the key steps I took?',
                'What was the final outcome?'
            ],
            'emotional': [
                'How did I feel during different phases?',
                'What emotions influenced my decisions?',
                'How did my emotional state affect performance?'
            ],
            'cognitive': [
                'What thinking strategies did I use?',
                'What assumptions did I make?',
                'What knowledge did I apply or lack?'
            ],
            'evaluative': [
                'What went well and why?',
                'What could have gone better?',
                'How accurate was my self-assessment?'
            ],
            'strategic': [
                'What would I do differently next time?',
                'What patterns should I watch for?',
                'How can I improve my approach?'
            ]
        },
        ReflectionType.POST_ERROR: {
            'descriptive': [
                'What error occurred and when?',
                'What was the immediate cause?',
                'What was the chain of events?'
            ],
            'root_cause': [
                'What were the underlying causes?',
                'What assumptions led to the error?',
                'What information was missing or wrong?'
            ],
            'prevention': [
                'How could this error have been prevented?',
                'What checks should I add?',
                'What would have caught this earlier?'
            ],
            'recovery': [
                'How well did I recover from the error?',
                'What recovery strategies worked?',
                'How can I improve error handling?'
            ]
        }
    }
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.reflection_history: List[DeepReflection] = []
    
    async def deep_reflect(
        self,
        trace: ReasoningTrace,
        reflection_type: ReflectionType,
        depth: ReflectionDepth
    ) -> DeepReflection:
        """Perform deep structured reflection"""
        start_time = datetime.now()
        
        # Get templates for reflection type
        templates = self.REFLECTION_TEMPLATES.get(reflection_type, {})
        
        phases = {}
        
        # Phase 1: Descriptive reflection
        phases['descriptive'] = await self._reflect_phase(
            'descriptive', templates.get('descriptive', []), trace
        )
        
        # Phase 2: Emotional reflection (if depth permits)
        if depth.value >= ReflectionDepth.EMOTIONAL.value:
            phases['emotional'] = await self._reflect_phase(
                'emotional', templates.get('emotional', []), trace
            )
        
        # Phase 3: Cognitive reflection (if depth permits)
        if depth.value >= ReflectionDepth.COGNITIVE.value:
            phases['cognitive'] = await self._reflect_phase(
                'cognitive', templates.get('cognitive', []), trace
            )
        
        # Phase 4: Evaluative reflection (if depth permits)
        if depth.value >= ReflectionDepth.EVALUATIVE.value:
            phases['evaluative'] = await self._reflect_phase(
                'evaluative', templates.get('evaluative', []), trace
            )
        
        # Phase 5: Strategic reflection (if depth permits)
        if depth.value >= ReflectionDepth.STRATEGIC.value:
            phases['strategic'] = await self._reflect_phase(
                'strategic', templates.get('strategic', []), trace
            )
        
        # Phase 6: Transformative reflection (if depth permits)
        if depth.value >= ReflectionDepth.TRANSFORMATIVE.value:
            phases['transformative'] = await self._reflect_transformative(
                phases, trace
            )
        
        # Synthesize insights
        insights = self._synthesize_insights(phases)
        
        # Derive action items
        action_items = self._derive_action_items(phases)
        
        # Extract learning outcomes
        learning_outcomes = self._extract_learning_outcomes(phases)
        
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        reflection = DeepReflection(
            experience_id=trace.trace_id,
            reflection_type=reflection_type,
            depth=depth,
            phases=phases,
            insights=insights,
            action_items=action_items,
            learning_outcomes=learning_outcomes,
            duration_ms=duration_ms
        )
        
        self.reflection_history.append(reflection)
        return reflection
    
    async def _reflect_phase(
        self,
        phase_name: str,
        questions: List[str],
        trace: ReasoningTrace
    ) -> ReflectionPhase:
        """Reflect on a specific phase"""
        insights = []
        
        for question in questions:
            # Generate insight for each question
            insight = await self._generate_insight(question, trace)
            insights.append(insight)
        
        # Calculate confidence from insight quality
        insight_confidences = [getattr(i, 'confidence', 0.5) for i in insights]
        avg_confidence = sum(insight_confidences) / len(insight_confidences) if insight_confidences else 0.5
        # Adjust based on number of questions answered
        coverage = len(insights) / len(questions) if questions else 0.5
        confidence = avg_confidence * 0.7 + coverage * 0.3

        return ReflectionPhase(
            phase_name=phase_name,
            questions=questions,
            insights=insights,
            confidence=confidence
        )
    
    async def _reflect_transformative(
        self,
        phases: Dict[str, ReflectionPhase],
        trace: ReasoningTrace
    ) -> ReflectionPhase:
        """Perform transformative reflection"""
        questions = [
            'How has this experience changed my understanding of myself?',
            'What fundamental beliefs or assumptions were challenged?',
            'How might this change my approach to similar situations in the future?',
            'What aspects of my identity or capabilities were revealed?'
        ]
        
        insights = []
        for question in questions:
            insight = await self._generate_insight(question, trace, phases)
            insights.append(insight)
        
        return ReflectionPhase(
            phase_name='transformative',
            questions=questions,
            insights=insights,
            confidence=0.6  # Transformative insights are more uncertain
        )
    
    async def _generate_insight(
        self,
        question: str,
        trace: ReasoningTrace,
        context: Optional[Dict] = None
    ) -> str:
        """Generate an insight for a reflection question using LLM"""
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()

            trace_summary = "; ".join(
                s.conclusion[:80] for s in trace.steps[:5]
            ) if trace.steps else "No reasoning steps available"

            context_str = ""
            if context:
                context_str = f"\nContext: {json.dumps({k: str(v)[:100] for k, v in list(context.items())[:5]})}"

            prompt = (
                f"Given this reasoning trace summary:\n{trace_summary}\n"
                f"{context_str}\n\n"
                f"Generate a concise insight answering: {question}\n"
                f"Respond in 1-2 sentences."
            )
            return client.generate(prompt, max_tokens=150)
        except (ImportError, ValueError, RuntimeError, EnvironmentError) as e:
            logger.warning(f"LLM insight generation failed: {e}")
            return f"Insight for: {question[:50]}..."
    
    def _synthesize_insights(self, phases: Dict[str, ReflectionPhase]) -> List[str]:
        """Synthesize insights from all phases"""
        all_insights = []
        
        for phase in phases.values():
            all_insights.extend(phase.insights)
        
        # Deduplicate and prioritize
        unique_insights = list(set(all_insights))
        
        return unique_insights[:10]  # Top 10 insights
    
    def _derive_action_items(self, phases: Dict[str, ReflectionPhase]) -> List[str]:
        """Derive action items from reflection phases"""
        action_items = []
        
        # Extract from strategic phase
        if 'strategic' in phases:
            for insight in phases['strategic'].insights:
                if 'should' in insight.lower() or 'will' in insight.lower():
                    action_items.append(insight)
        
        # Extract from evaluative phase
        if 'evaluative' in phases:
            for insight in phases['evaluative'].insights:
                if 'improve' in insight.lower() or 'better' in insight.lower():
                    action_items.append(insight)
        
        return action_items[:5]  # Top 5 action items
    
    def _extract_learning_outcomes(self, phases: Dict[str, ReflectionPhase]) -> List[str]:
        """Extract learning outcomes from reflection"""
        outcomes = []
        
        for phase in phases.values():
            for insight in phase.insights:
                if 'learn' in insight.lower() or 'understand' in insight.lower():
                    outcomes.append(insight)
        
        return outcomes[:5]  # Top 5 learning outcomes


class PatternLibrary:
    """
    Library of thinking patterns with metadata
    """
    
    PATTERNS = {
        'linear_reasoning': Pattern(
            name='linear_reasoning',
            description='Step-by-step sequential reasoning',
            category=PatternCategory.PRODUCTIVE,
            typical_use_cases=['mathematical_problems', 'logical_deduction'],
            indicators=['sequential_steps', 'clear_premise_conclusion'],
            strengths=['clarity', 'verifiability', 'systematic'],
            weaknesses=['may_miss_alternatives', 'slow_for_complex'],
            modification_strategies=['add_parallel_branches', 'integrate_abduction']
        ),
        'divergent_thinking': Pattern(
            name='divergent_thinking',
            description='Generating multiple alternatives',
            category=PatternCategory.PRODUCTIVE,
            typical_use_cases=['creative_tasks', 'brainstorming'],
            indicators=['multiple_alternatives', 'exploration'],
            strengths=['exploration', 'novelty', 'comprehensive'],
            weaknesses=['may_lack_depth', 'inefficient'],
            modification_strategies=['add_convergence_phase', 'prioritize_alternatives']
        ),
        'analogical_reasoning': Pattern(
            name='analogical_reasoning',
            description='Using analogies to solve problems',
            category=PatternCategory.PRODUCTIVE,
            typical_use_cases=['novel_situations', 'transfer_learning'],
            indicators=['analogies', 'comparisons', 'similarities'],
            strengths=['leverage_experience', 'creative_solutions'],
            weaknesses=['may_be_misleading', 'false_analogies'],
            modification_strategies=['add_analogy_validation', 'track_success']
        ),
        'premature_closure': Pattern(
            name='premature_closure',
            description='Stopping analysis too early',
            category=PatternCategory.PROBLEMATIC,
            typical_use_cases=[],
            indicators=['few_alternatives', 'quick_decisions', 'high_early_confidence'],
            strengths=[],
            weaknesses=['miss_better_solutions', 'insufficient_analysis'],
            modification_strategies=['minimum_analysis_time', 'alternative_requirement']
        ),
        'overconfidence': Pattern(
            name='overconfidence',
            description='Excessive confidence in judgments',
            category=PatternCategory.PROBLEMATIC,
            typical_use_cases=[],
            indicators=['high_confidence_errors', 'calibration_issues'],
            strengths=[],
            weaknesses=['poor_calibration', 'missed_errors'],
            modification_strategies=['confidence_calibration', 'uncertainty_quantification']
        )
    }
    
    def __init__(self):
        self.patterns = self.PATTERNS.copy()
        self.pattern_usage: Dict[str, int] = {name: 0 for name in self.patterns}
    
    def get_pattern(self, name: str) -> Optional[Pattern]:
        """Get a pattern by name"""
        return self.patterns.get(name)
    
    def get_patterns_by_category(self, category: PatternCategory) -> List[Pattern]:
        """Get all patterns in a category"""
        return [
            p for p in self.patterns.values() 
            if p.category == category
        ]
    
    def record_usage(self, pattern_name: str):
        """Record usage of a pattern"""
        if pattern_name in self.pattern_usage:
            self.pattern_usage[pattern_name] += 1
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get pattern usage statistics"""
        return self.pattern_usage.copy()


class ThinkingPatternModifiers:
    """
    System for identifying and modifying thinking patterns
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.pattern_library = PatternLibrary()
        self.modification_history: List[PatternModification] = []
    
    async def analyze_and_modify_patterns(
        self,
        traces: List[ReasoningTrace],
        goals: List[str]
    ) -> List[PatternModification]:
        """Analyze thinking patterns and suggest modifications"""
        
        # Extract patterns from traces
        detected_patterns = self._extract_patterns(traces)
        
        # Classify patterns
        classified = self._classify_patterns(detected_patterns)
        
        # Identify problematic patterns
        problematic = [
            p for p in classified 
            if p.category == PatternCategory.PROBLEMATIC
        ]
        
        # Identify enhancement opportunities
        productive = [
            p for p in classified 
            if p.category == PatternCategory.PRODUCTIVE
        ]
        
        # Generate modifications
        modifications = []
        
        for pattern in problematic:
            mod = await self._generate_reduction_modification(pattern)
            modifications.append(mod)
        
        for pattern in productive:
            if self._should_enhance(pattern, goals):
                mod = await self._generate_enhancement_modification(pattern)
                modifications.append(mod)
        
        self.modification_history.extend(modifications)
        return modifications
    
    def _extract_patterns(self, traces: List[ReasoningTrace]) -> List[Pattern]:
        """Extract patterns from reasoning traces"""
        detected = []
        
        for trace in traces:
            # Check for linear reasoning
            if self._is_linear_reasoning(trace):
                detected.append(self.pattern_library.get_pattern('linear_reasoning'))
            
            # Check for divergent thinking
            if self._is_divergent_thinking(trace):
                detected.append(self.pattern_library.get_pattern('divergent_thinking'))
            
            # Check for premature closure
            if self._is_premature_closure(trace):
                detected.append(self.pattern_library.get_pattern('premature_closure'))
            
            # Check for overconfidence
            if self._is_overconfidence(trace):
                detected.append(self.pattern_library.get_pattern('overconfidence'))
        
        return [p for p in detected if p is not None]
    
    def _is_linear_reasoning(self, trace: ReasoningTrace) -> bool:
        """Check if trace shows linear reasoning pattern"""
        if len(trace.steps) < 2:
            return False
        
        # Check for mostly sequential inference steps
        inference_steps = [s for s in trace.steps if s.step_type == StepType.INFERENCE]
        return len(inference_steps) > len(trace.steps) * 0.6
    
    def _is_divergent_thinking(self, trace: ReasoningTrace) -> bool:
        """Check if trace shows divergent thinking pattern"""
        # Check for multiple decision points with alternatives
        return len(trace.decision_points) > 2 and any(
            len(dp.alternatives) >= 3 for dp in trace.decision_points
        )
    
    def _is_premature_closure(self, trace: ReasoningTrace) -> bool:
        """Check if trace shows premature closure pattern"""
        for dp in trace.decision_points:
            if len(dp.alternatives) < 2 and dp.confidence > 0.8:
                return True
        return False
    
    def _is_overconfidence(self, trace: ReasoningTrace) -> bool:
        """Check if trace shows overconfidence pattern"""
        high_conf_steps = [s for s in trace.steps if s.confidence > 0.9]
        unchecked_steps = [s for s in trace.steps if s.validation_status == ValidationStatus.UNCHECKED]
        
        return len(high_conf_steps) > len(trace.steps) * 0.5 and len(unchecked_steps) > len(trace.steps) * 0.3
    
    def _classify_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Classify detected patterns"""
        # Patterns are already classified in the library
        return patterns
    
    def _should_enhance(self, pattern: Pattern, goals: List[str]) -> bool:
        """Determine if a productive pattern should be enhanced"""
        # Check if pattern aligns with goals
        for goal in goals:
            if any(use in goal for use in pattern.typical_use_cases):
                return True
        return False
    
    async def _generate_reduction_modification(
        self, 
        pattern: Pattern
    ) -> PatternModification:
        """Generate modification to reduce problematic pattern"""
        strategy = pattern.modification_strategies[0] if pattern.modification_strategies else 'general_reduction'
        
        return PatternModification(
            pattern_name=pattern.name,
            modification_type=ModificationType.REDUCTION,
            description=f"Reduce {pattern.name} using {strategy}",
            expected_impact=0.7,
            implementation=lambda: self._implement_reduction(pattern, strategy),
            verification_criteria=['reduced_frequency', 'improved_outcomes']
        )
    
    async def _generate_enhancement_modification(
        self, 
        pattern: Pattern
    ) -> PatternModification:
        """Generate modification to enhance productive pattern"""
        strategy = pattern.modification_strategies[0] if pattern.modification_strategies else 'general_enhancement'
        
        return PatternModification(
            pattern_name=pattern.name,
            modification_type=ModificationType.ENHANCEMENT,
            description=f"Enhance {pattern.name} using {strategy}",
            expected_impact=0.6,
            implementation=lambda: self._implement_enhancement(pattern, strategy),
            verification_criteria=['increased_frequency', 'improved_outcomes']
        )
    
    def _implement_reduction(self, pattern: Pattern, strategy: str):
        """Implement pattern reduction"""
        logger.info(f"Implementing reduction for {pattern.name} using {strategy}")
        pattern.metadata = pattern.metadata or {}
        pattern.metadata['reduction_strategy'] = strategy
        pattern.metadata['reduction_applied_at'] = __import__('datetime').datetime.now().isoformat()
    
    def _implement_enhancement(self, pattern: Pattern, strategy: str):
        """Implement pattern enhancement"""
        logger.info(f"Implementing enhancement for {pattern.name} using {strategy}")
        pattern.metadata = pattern.metadata or {}
        pattern.metadata['enhancement_strategy'] = strategy
        pattern.metadata['enhancement_applied_at'] = __import__('datetime').datetime.now().isoformat()


class MetaLearner:
    """
    Core meta-learning system that learns how to learn
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.learning_history: List[Dict] = []
        self.strategy_effectiveness: Dict[str, float] = {}
    
    async def generate_strategy(
        self,
        task_profile: Dict[str, Any],
        learner_state: Dict[str, Any],
        similar_experiences: List[Dict]
    ) -> LearningStrategy:
        """Generate optimized learning strategy"""
        
        # Analyze task characteristics
        task_complexity = task_profile.get('complexity', 0.5)
        task_novelty = task_profile.get('novelty', 0.5)
        
        # Analyze learner state
        current_skill = learner_state.get('skill_level', 0.5)
        learning_rate = learner_state.get('learning_rate', 0.1)
        
        # Select approach based on analysis
        if task_novelty > 0.7:
            approach = 'exploratory_learning'
        elif task_complexity > 0.7:
            approach = 'structured_decomposition'
        elif current_skill < 0.3:
            approach = 'foundational_building'
        else:
            approach = 'refinement_optimization'
        
        # Generate parameters
        parameters = self._generate_parameters(
            approach, task_profile, learner_state
        )
        
        # Predict outcomes
        predicted_outcomes = self._predict_outcomes(
            approach, parameters, similar_experiences
        )
        
        return LearningStrategy(
            approach=approach,
            parameters=parameters,
            predicted_outcomes=predicted_outcomes,
            confidence=self._calculate_strategy_confidence(
                approach, similar_experiences
            ),
            adaptation_triggers=self._define_adaptation_triggers(approach)
        )
    
    def _generate_parameters(
        self,
        approach: str,
        task_profile: Dict,
        learner_state: Dict
    ) -> Dict[str, Any]:
        """Generate parameters for learning strategy"""
        return {
            'learning_rate': learner_state.get('learning_rate', 0.1),
            'exploration_ratio': 0.3 if task_profile.get('novelty', 0) > 0.5 else 0.1,
            'practice_iterations': int(10 * task_profile.get('complexity', 0.5)),
            'feedback_frequency': 'high' if learner_state.get('skill_level', 0.5) < 0.5 else 'medium',
            'scaffolding_level': 'high' if task_profile.get('complexity', 0.5) > 0.7 else 'medium'
        }
    
    def _predict_outcomes(
        self,
        approach: str,
        parameters: Dict,
        similar_experiences: List[Dict]
    ) -> Dict[str, float]:
        """Predict outcomes of learning strategy"""
        # Base predictions on similar experiences
        if similar_experiences:
            avg_success = np.mean([e.get('success_rate', 0.5) for e in similar_experiences])
            avg_time = np.mean([e.get('time_to_proficiency', 100) for e in similar_experiences])
        else:
            avg_success = 0.6
            avg_time = 100
        
        return {
            'success_probability': min(0.95, avg_success * 1.1),
            'expected_time_to_proficiency': int(avg_time * 0.9),
            'retention_rate': 0.75,
            'transfer_probability': 0.6
        }
    
    def _calculate_strategy_confidence(
        self,
        approach: str,
        similar_experiences: List[Dict]
    ) -> float:
        """Calculate confidence in strategy"""
        if not similar_experiences:
            return 0.5
        
        # Higher confidence with more similar experiences
        base_confidence = min(0.9, 0.5 + len(similar_experiences) * 0.05)
        
        # Adjust by success rate of similar experiences
        avg_success = np.mean([e.get('success_rate', 0.5) for e in similar_experiences])
        
        return min(0.95, base_confidence * (0.5 + avg_success))
    
    def _define_adaptation_triggers(self, approach: str) -> List[str]:
        """Define triggers for strategy adaptation"""
        return [
            'performance_plateau',
            'error_rate_increase',
            'confidence_mismatch',
            'time_exceeded',
            'novel_subtask_encountered'
        ]
    
    async def update_from_experience(
        self,
        experience: Dict,
        outcomes: Dict
    ):
        """Update meta-learner from learning experience"""
        self.learning_history.append({
            'experience': experience,
            'outcomes': outcomes,
            'timestamp': datetime.now()
        })
        
        # Update strategy effectiveness
        strategy = experience.get('strategy', 'unknown')
        success = outcomes.get('success', False)
        
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = 0.5
        
        # Update with moving average
        current = self.strategy_effectiveness[strategy]
        self.strategy_effectiveness[strategy] = 0.9 * current + 0.1 * (1.0 if success else 0.0)


class ArchitectureEvolver:
    """
    System for evolving cognitive architecture based on performance
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.evolution_history: List[ArchitectureEvolution] = []
        self.pending_changes: List[ArchitectureChange] = []
        self.last_evolution_time: Optional[datetime] = None
    
    async def evolve_architecture(
        self,
        performance_data: PerformanceSnapshot,
        evolution_goal: str
    ) -> Optional[ArchitectureEvolution]:
        """Evolve architecture to better achieve specified goal"""
        
        # Check cooldown period
        if self.last_evolution_time:
            hours_since = (datetime.now() - self.last_evolution_time).total_seconds() / 3600
            if hours_since < self.config.evolution_cooldown_period_hours:
                logger.info(f"Evolution on cooldown. {hours_since:.1f} hours since last evolution")
                return None
        
        # Analyze current architecture
        current_analysis = await self._analyze_current_architecture(performance_data)
        
        # Identify improvement opportunities
        opportunities = await self._identify_improvements(
            current_analysis,
            evolution_goal
        )
        
        if not opportunities:
            logger.info("No improvement opportunities identified")
            return None
        
        # Generate candidate changes
        candidates = await self._generate_candidates(opportunities)
        
        # Limit number of changes
        candidates = candidates[:self.config.max_evolution_changes_per_cycle]
        
        # Validate changes
        if self.config.evolution_safety_checks:
            validated = await self._validate_changes(candidates)
            if not validated:
                return ArchitectureEvolution(
                    status=EvolutionStatus.REJECTED,
                    changes_applied=[],
                    validation_results={'safety_check': False},
                    performance_impact={},
                    reason="Failed safety validation"
                )
        
        # Apply changes
        applied_changes = []
        for change in candidates:
            try:
                await self._apply_change(change)
                applied_changes.append(change)
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(f"Failed to apply change {change.component_id}: {e}")
        
        self.last_evolution_time = datetime.now()
        
        evolution = ArchitectureEvolution(
            status=EvolutionStatus.SUCCESS if applied_changes else EvolutionStatus.FAILED,
            changes_applied=applied_changes,
            validation_results={'safety_check': True},
            performance_impact=self._estimate_performance_impact(applied_changes)
        )
        
        self.evolution_history.append(evolution)
        return evolution
    
    async def _analyze_current_architecture(
        self,
        performance: PerformanceSnapshot
    ) -> Dict[str, Any]:
        """Analyze current architecture performance"""
        return {
            'accuracy_score': (
                performance.accuracy.factual_correctness +
                performance.accuracy.logical_validity
            ) / 2,
            'efficiency_score': performance.efficiency.token_efficiency,
            'quality_score': performance.quality.response_coherence,
            'metacognition_score': performance.metacognition.reflection_depth,
            'bottlenecks': self._identify_bottlenecks(performance)
        }
    
    def _identify_bottlenecks(self, performance: PerformanceSnapshot) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if performance.accuracy.factual_correctness < 0.7:
            bottlenecks.append('accuracy')
        
        if performance.efficiency.time_to_solution > 10000:
            bottlenecks.append('speed')
        
        if performance.quality.creativity_score < 0.4:
            bottlenecks.append('creativity')
        
        if performance.metacognition.self_correction_rate < 0.1:
            bottlenecks.append('self_correction')
        
        return bottlenecks
    
    async def _identify_improvements(
        self,
        analysis: Dict,
        goal: str
    ) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []
        
        if analysis['accuracy_score'] < 0.7:
            opportunities.append('improve_validation')
        
        if analysis['efficiency_score'] < 0.6:
            opportunities.append('optimize_processing')
        
        if 'creativity' in analysis['bottlenecks']:
            opportunities.append('enhance_divergence')
        
        if 'self_correction' in analysis['bottlenecks']:
            opportunities.append('strengthen_reflection')
        
        return opportunities
    
    async def _generate_candidates(
        self,
        opportunities: List[str]
    ) -> List[ArchitectureChange]:
        """Generate candidate architecture changes"""
        candidates = []
        
        for opp in opportunities:
            if opp == 'improve_validation':
                candidates.append(ArchitectureChange(
                    component_id='validation_layer',
                    change_type='enhancement',
                    description='Add additional validation checks',
                    implementation=lambda: logger.info("Implementing validation enhancement"),
                    rollback_procedure=lambda: logger.info("Rolling back validation enhancement"),
                    risk_assessment={'low': 0.7, 'medium': 0.3}
                ))
            
            elif opp == 'optimize_processing':
                candidates.append(ArchitectureChange(
                    component_id='processing_pipeline',
                    change_type='optimization',
                    description='Optimize processing pipeline for speed',
                    implementation=lambda: logger.info("Implementing processing optimization"),
                    rollback_procedure=lambda: logger.info("Rolling back processing optimization"),
                    risk_assessment={'low': 0.6, 'medium': 0.4}
                ))
            
            elif opp == 'enhance_divergence':
                candidates.append(ArchitectureChange(
                    component_id='divergent_thinking',
                    change_type='addition',
                    description='Add divergent thinking module',
                    implementation=lambda: logger.info("Adding divergent thinking"),
                    rollback_procedure=lambda: logger.info("Removing divergent thinking"),
                    risk_assessment={'low': 0.5, 'medium': 0.4, 'high': 0.1}
                ))
            
            elif opp == 'strengthen_reflection':
                candidates.append(ArchitectureChange(
                    component_id='reflection_engine',
                    change_type='enhancement',
                    description='Strengthen reflection capabilities',
                    implementation=lambda: logger.info("Strengthening reflection"),
                    rollback_procedure=lambda: logger.info("Rolling back reflection enhancement"),
                    risk_assessment={'low': 0.8, 'medium': 0.2}
                ))
        
        return candidates
    
    async def _validate_changes(self, changes: List[ArchitectureChange]) -> bool:
        """Validate proposed changes against safety constraints"""
        for change in changes:
            # Check risk level
            high_risk = change.risk_assessment.get('high', 0)
            if high_risk > 0.3:
                logger.warning(f"Change {change.component_id} has high risk: {high_risk}")
                return False
        
        return True
    
    async def _apply_change(self, change: ArchitectureChange):
        """Apply an architecture change"""
        logger.info(f"Applying change: {change.component_id} - {change.description}")
        change.implementation()
    
    def _estimate_performance_impact(
        self,
        changes: List[ArchitectureChange]
    ) -> Dict[str, float]:
        """Estimate performance impact of changes"""
        return {
            'accuracy_improvement': len(changes) * 0.05,
            'efficiency_improvement': len(changes) * 0.03,
            'risk_level': sum(
                c.risk_assessment.get('high', 0) for c in changes
            ) / max(len(changes), 1)
        }


class MetaCognitiveMemory:
    """
    Specialized memory system for meta-cognitive information
    """
    
    def __init__(self, config: MetaCognitionConfig):
        self.config = config
        self.episodic: deque = deque(maxlen=1000)
        self.semantic: Dict[str, Any] = {}
        self.procedural: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {
            'self_model': {},
            'performance_history': deque(maxlen=config.performance_window_size)
        }
    
    async def store_reflection(self, reflection: DeepReflection):
        """Store reflection in episodic memory"""
        self.episodic.append({
            'type': 'reflection',
            'data': reflection,
            'timestamp': datetime.now()
        })
    
    async def store_performance(self, performance: PerformanceSnapshot):
        """Store performance snapshot"""
        self.meta['performance_history'].append(performance)
    
    async def get_performance_history(
        self,
        window: int = 100
    ) -> List[PerformanceSnapshot]:
        """Get recent performance history"""
        history = list(self.meta['performance_history'])
        return history[-window:] if len(history) > window else history
    
    async def update_self_model(self, updates: Dict[str, Any]):
        """Update self-model"""
        self.meta['self_model'].update(updates)
    
    def get_self_model(self) -> Dict[str, Any]:
        """Get current self-model"""
        return self.meta['self_model'].copy()


class MetaCognitionLoop:
    """
    Main meta-cognition loop integrating all components
    """
    
    def __init__(self, config: Optional[MetaCognitionConfig] = None):
        self.config = config or MetaCognitionConfig()
        
        # Core components
        self.performance_monitor = CognitivePerformanceMonitor(self.config)
        self.bias_detector = CognitiveBiasDetector(self.config)
        self.recursive_reflection = RecursiveReflectionEngine(self.config)
        self.deep_reflection = DeepReflectionEngine(self.config)
        self.pattern_modifiers = ThinkingPatternModifiers(self.config)
        self.meta_learner = MetaLearner(self.config)
        self.architecture_evolver = ArchitectureEvolver(self.config)
        
        # Memory
        self.memory = MetaCognitiveMemory(self.config)
        
        # State
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        
        logger.info("MetaCognitionLoop initialized")

    async def run_cycle(
        self,
        task_description: str,
        steps: Optional[List[ReasoningStep]] = None,
        trigger: Optional[MetaCognitionTrigger] = None,
        **execution_data: Any
    ) -> MetaCognitionResult:
        """
        Top-level entry point for running a meta-cognition cycle.

        Constructs a ReasoningTrace from the provided execution data and
        delegates to execute_cycle().

        Args:
            task_description: Human-readable description of the task being reflected on.
            steps: Optional list of ReasoningStep objects from actual loop execution.
            trigger: Optional trigger that caused this cycle.
            **execution_data: Additional keyword data forwarded to
                create_reasoning_trace (e.g. used_tools, decision_points).

        Returns:
            MetaCognitionResult from the completed cycle.
        """
        trace = self.create_reasoning_trace(
            task_description=task_description,
            steps=steps or [],
            **execution_data
        )
        return await self.execute_cycle(trace, trigger)

    @staticmethod
    def create_reasoning_trace(
        task_description: str,
        steps: Optional[List[ReasoningStep]] = None,
        used_tools: Optional[List[ToolInvocation]] = None,
        decision_points: Optional[List[DecisionPoint]] = None,
        uncertainty_points: Optional[List[UncertaintyPoint]] = None,
        retrieved_memories: Optional[List[Memory]] = None,
        external_knowledge: Optional[List[KnowledgeSource]] = None,
        backtrack_count: int = 0,
        revision_count: int = 0,
        trace_id: Optional[str] = None,
        **_extra: Any
    ) -> ReasoningTrace:
        """
        Factory method to create a ReasoningTrace from actual loop execution data.

        If no steps are provided, a single placeholder REFLECTION step is created
        so that downstream analysis methods still function.

        Args:
            task_description: What task was being performed.
            steps: Actual ReasoningStep list from the execution.
            used_tools: Tools that were invoked during the task.
            decision_points: Decision points encountered.
            uncertainty_points: Points of uncertainty.
            retrieved_memories: Memories that were retrieved.
            external_knowledge: External knowledge sources consulted.
            backtrack_count: Number of backtracks during reasoning.
            revision_count: Number of revisions made.
            trace_id: Optional trace ID (auto-generated if not provided).

        Returns:
            A fully populated ReasoningTrace.
        """
        now = datetime.now()

        if steps:
            initial_state = steps[0].input_state
            final_state = steps[-1].output_state
            confidence_trajectory = [s.confidence for s in steps]
        else:
            # Create a minimal default state when no steps are provided
            default_state = CognitiveState(
                timestamp=now,
                active_goals=[task_description[:80]],
                working_memory={},
                confidence_level=0.5,
            )
            initial_state = default_state
            final_state = default_state
            confidence_trajectory = [0.5]

        return ReasoningTrace(
            trace_id=trace_id or f"trace_{now.timestamp()}",
            timestamp=now,
            task_description=task_description,
            steps=steps or [],
            initial_state=initial_state,
            final_state=final_state,
            confidence_trajectory=confidence_trajectory,
            uncertainty_points=uncertainty_points or [],
            decision_points=decision_points or [],
            retrieved_memories=retrieved_memories or [],
            used_tools=used_tools or [],
            external_knowledge=external_knowledge or [],
            backtrack_count=backtrack_count,
            revision_count=revision_count,
        )

    async def execute_cycle(
        self,
        reasoning_trace: ReasoningTrace,
        trigger: Optional[MetaCognitionTrigger] = None
    ) -> MetaCognitionResult:
        """Execute complete meta-cognition cycle"""
        cycle_start = datetime.now()
        
        if trigger is None:
            trigger = MetaCognitionTrigger(
                trigger_type='automatic',
                priority=TriggerPriority.MEDIUM,
                context={'trace_id': reasoning_trace.trace_id}
            )
        
        logger.info(f"Starting meta-cognition cycle {self.cycle_count + 1}")
        
        # Phase 1: Performance analysis
        performance = await self.performance_monitor.capture_metrics(reasoning_trace)
        await self.memory.store_performance(performance)
        
        # Phase 2: Bias detection
        bias_detection = await self.bias_detector.detect_biases(reasoning_trace)
        
        # Phase 3: Recursive reflection
        recursive_reflection = await self.recursive_reflection.recursive_analyze(
            reasoning_trace
        )
        
        # Phase 4: Deep reflection (if triggered)
        deep_reflection = None
        if self._should_deep_reflect(performance, bias_detection):
            deep_reflection = await self.deep_reflection.deep_reflect(
                trace=reasoning_trace,
                reflection_type=ReflectionType.POST_TASK,
                depth=self.config.default_reflection_depth
            )
            await self.memory.store_reflection(deep_reflection)
        
        # Phase 5: Pattern analysis and modification
        pattern_modifications = await self.pattern_modifiers.analyze_and_modify_patterns(
            traces=[reasoning_trace],
            goals=['improve_accuracy', 'reduce_bias', 'enhance_efficiency']
        )
        
        # Phase 6: Learning optimization
        learning_optimization = await self.meta_learner.generate_strategy(
            task_profile={'complexity': 0.5, 'novelty': 0.3},
            learner_state={'skill_level': 0.6, 'learning_rate': 0.1},
            similar_experiences=[]
        )
        
        # Phase 7: Architecture evolution (if needed)
        architecture_evolution = None
        if self._should_evolve_architecture(performance):
            architecture_evolution = await self.architecture_evolver.evolve_architecture(
                performance_data=performance,
                evolution_goal='improve_overall_performance'
            )
        
        # Phase 8: Synthesize insights
        insights = self._synthesize_insights(
            performance=performance,
            bias_detection=bias_detection,
            recursive_reflection=recursive_reflection,
            deep_reflection=deep_reflection
        )
        
        # Phase 9: Derive action items
        action_items = self._derive_action_items(
            bias_detection=bias_detection,
            deep_reflection=deep_reflection,
            pattern_modifications=pattern_modifications
        )
        
        cycle_duration_ms = (datetime.now() - cycle_start).total_seconds() * 1000
        
        result = MetaCognitionResult(
            trigger=trigger,
            cycle_duration_ms=cycle_duration_ms,
            performance_analysis=performance,
            bias_detection=bias_detection,
            recursive_reflection=recursive_reflection,
            deep_reflection=deep_reflection,
            pattern_modifications=pattern_modifications,
            learning_optimization=learning_optimization,
            architecture_evolution=architecture_evolution,
            insights=insights,
            action_items=action_items
        )
        
        self.cycle_count += 1
        self.last_cycle_time = datetime.now()
        
        logger.info(f"Meta-cognition cycle {self.cycle_count} completed in {cycle_duration_ms:.0f}ms")
        
        return result
    
    def _should_deep_reflect(
        self,
        performance: PerformanceSnapshot,
        bias_detection: BiasDetectionResult
    ) -> bool:
        """Determine if deep reflection should be triggered"""
        # Trigger on poor performance
        if performance.accuracy.factual_correctness < self.config.deep_reflection_trigger_threshold:
            return True
        
        # Trigger on detected biases
        if bias_detection.detected_biases:
            return True
        
        # Trigger on low confidence calibration
        if performance.accuracy.calibration_error > 0.3:
            return True
        
        return False
    
    def _should_evolve_architecture(self, performance: PerformanceSnapshot) -> bool:
        """Determine if architecture evolution should be triggered"""
        # Check if performance is consistently below baseline
        if not self.performance_monitor.baseline_established:
            return False
        
        baseline = self.performance_monitor.baseline_metrics
        
        # Trigger if accuracy is significantly below baseline
        if performance.accuracy.factual_correctness < baseline.accuracy.factual_correctness * 0.8:
            return True
        
        # Trigger if efficiency is significantly degraded
        if performance.efficiency.token_efficiency < baseline.efficiency.token_efficiency * 0.7:
            return True
        
        return False
    
    def _synthesize_insights(
        self,
        performance: PerformanceSnapshot,
        bias_detection: BiasDetectionResult,
        recursive_reflection: RecursiveReflectionResult,
        deep_reflection: Optional[DeepReflection]
    ) -> List[str]:
        """Synthesize insights from all components"""
        insights = []
        
        # Performance insights
        if performance.accuracy.factual_correctness < 0.7:
            insights.append("Accuracy below target - consider additional validation")
        
        if performance.accuracy.calibration_error > 0.2:
            insights.append("Confidence calibration needs improvement")
        
        # Bias insights
        for bias in bias_detection.detected_biases:
            insights.append(f"Detected {bias.bias_type.value} bias with {bias.confidence:.2f} confidence")
        
        # Reflection insights
        insights.append(f"Recursive reflection reached level {recursive_reflection.level}")
        
        if deep_reflection:
            insights.extend(deep_reflection.insights[:3])
        
        return insights
    
    def _derive_action_items(
        self,
        bias_detection: BiasDetectionResult,
        deep_reflection: Optional[DeepReflection],
        pattern_modifications: List[PatternModification]
    ) -> List[str]:
        """Derive action items from analysis"""
        action_items = []
        
        # Bias mitigation actions
        for bias in bias_detection.detected_biases:
            if bias.severity in ['critical', 'high']:
                for strategy in bias.mitigation_strategies[:2]:
                    action_items.append(f"Apply {strategy} for {bias.bias_type.value}")
        
        # Deep reflection actions
        if deep_reflection:
            action_items.extend(deep_reflection.action_items[:3])
        
        # Pattern modification actions
        for mod in pattern_modifications:
            action_items.append(f"{mod.modification_type.value}: {mod.description}")
        
        return action_items[:10]  # Limit to top 10
    
    async def get_performance_report(
        self,
        window: int = 100
    ) -> Dict[str, Any]:
        """Get performance report for recent cycles"""
        history = await self.memory.get_performance_history(window)
        
        if not history:
            return {'error': 'No performance history available'}
        
        return {
            'window_size': len(history),
            'average_accuracy': np.mean([h.accuracy.factual_correctness for h in history]),
            'average_efficiency': np.mean([h.efficiency.token_efficiency for h in history]),
            'average_quality': np.mean([h.quality.response_coherence for h in history]),
            'trend': self._calculate_trend(history),
            'cycles_completed': self.cycle_count
        }
    
    def _calculate_trend(
        self,
        history: List[PerformanceSnapshot]
    ) -> str:
        """Calculate performance trend"""
        if len(history) < 10:
            return 'insufficient_data'
        
        # Compare first half to second half
        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]
        
        first_accuracy = np.mean([h.accuracy.factual_correctness for h in first_half])
        second_accuracy = np.mean([h.accuracy.factual_correctness for h in second_half])
        
        diff = second_accuracy - first_accuracy
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        return 'stable'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_reasoning_trace() -> ReasoningTrace:
    """Create a sample reasoning trace for testing"""
    initial_state = CognitiveState(
        timestamp=datetime.now(),
        active_goals=['solve_problem'],
        working_memory={'problem': 'sample task'},
        confidence_level=0.5
    )
    
    steps = [
        ReasoningStep(
            step_number=0,
            step_type=StepType.PERCEPTION,
            input_state=initial_state,
            output_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=['solve_problem'],
                working_memory={'problem': 'sample task', 'perceived': True},
                confidence_level=0.6
            ),
            premise="Problem presented",
            operation="perceive",
            conclusion="Problem understood",
            confidence=0.6,
            time_taken_ms=100
        ),
        ReasoningStep(
            step_number=1,
            step_type=StepType.INFERENCE,
            input_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=['solve_problem'],
                working_memory={'problem': 'sample task', 'perceived': True},
                confidence_level=0.6
            ),
            output_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=['solve_problem'],
                working_memory={'problem': 'sample task', 'approach': 'identified'},
                confidence_level=0.7
            ),
            premise="Problem understood",
            operation="analyze",
            conclusion="Approach identified",
            confidence=0.7,
            time_taken_ms=200,
            supporting_evidence=[Evidence('knowledge_base', 'similar_problem', 0.8, 'supporting')]
        ),
        ReasoningStep(
            step_number=2,
            step_type=StepType.DECISION,
            input_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=['solve_problem'],
                working_memory={'problem': 'sample task', 'approach': 'identified'},
                confidence_level=0.7
            ),
            output_state=CognitiveState(
                timestamp=datetime.now(),
                active_goals=['solve_problem', 'execute_approach'],
                working_memory={'problem': 'sample task', 'approach': 'selected'},
                confidence_level=0.75
            ),
            premise="Approach identified",
            operation="decide",
            conclusion="Approach selected",
            confidence=0.75,
            time_taken_ms=150
        )
    ]
    
    return ReasoningTrace(
        trace_id=f"trace_{datetime.now().timestamp()}",
        timestamp=datetime.now(),
        task_description="Sample reasoning task",
        steps=steps,
        initial_state=initial_state,
        final_state=steps[-1].output_state,
        confidence_trajectory=[0.5, 0.6, 0.7, 0.75],
        decision_points=[DecisionPoint(2, 'Approach selected', ['approach_a', 'approach_b'], ['efficiency', 'accuracy'], 0.75)]
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution function for testing"""
    print("=" * 60)
    print("Advanced Meta-Cognition Loop - Test Execution")
    print("=" * 60)
    
    # Create meta-cognition loop
    config = MetaCognitionConfig(
        max_reflection_depth=3,
        bias_detection_enabled=True
    )
    
    loop = MetaCognitionLoop(config)
    
    # Create sample reasoning trace
    trace = create_sample_reasoning_trace()
    
    # Execute meta-cognition cycle
    result = await loop.execute_cycle(trace)
    
    # Display results
    print("\n--- Meta-Cognition Cycle Results ---\n")
    print(f"Cycle Duration: {result.cycle_duration_ms:.0f}ms")
    print(f"Performance Accuracy: {result.performance_analysis.accuracy.factual_correctness:.2f}")
    print(f"Biases Detected: {len(result.bias_detection.detected_biases)}")
    print(f"Reflection Depth: {result.recursive_reflection.level}")
    print(f"Deep Reflection: {'Yes' if result.deep_reflection else 'No'}")
    print(f"Pattern Modifications: {len(result.pattern_modifications)}")
    print(f"Architecture Evolution: {'Yes' if result.architecture_evolution else 'No'}")
    
    print("\n--- Insights ---")
    for i, insight in enumerate(result.insights[:5], 1):
        print(f"{i}. {insight}")
    
    print("\n--- Action Items ---")
    for i, action in enumerate(result.action_items[:5], 1):
        print(f"{i}. {action}")
    
    # Get performance report
    report = await loop.get_performance_report()
    print("\n--- Performance Report ---")
    print(json.dumps(report, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("Test execution completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
