"""
Self-Driven Loop: Proactive Behavior Triggering System
======================================================

Detects conditions that should trigger proactive behavior.
Monitors environment, user patterns, and internal state.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

import ast
import logging
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any, Callable
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# ── Safe expression evaluator (replaces eval()) ─────────────────────────
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_BUILTINS = {'len': len, 'min': min, 'max': max, 'abs': abs,
                  'sum': sum, 'any': any, 'all': all, 'bool': bool,
                  'int': int, 'float': float, 'str': str,
                  'True': True, 'False': False, 'None': None}


class _SafeEvalVisitor(ast.NodeVisitor):
    """Walk an AST and evaluate only safe operations."""

    def __init__(self, context: Dict[str, Any]):
        self.context = {**_SAFE_BUILTINS, **context}

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    # Python 3.7 compat
    visit_Num = visit_Constant
    visit_Str = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.context:
            return self.context[node.id]
        raise NameError(f"Name '{node.id}' is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if isinstance(value, dict):
            return value.get(node.attr)
        raise AttributeError(f"Attribute access on {type(value).__name__} not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        idx = self.visit(node.slice)
        return value[idx]

    def visit_Index(self, node: ast.Index) -> Any:
        # Python <3.9 wraps subscript values in Index
        return self.visit(node.value)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = self.visit(v)
                if not result:
                    return result
            return result
        else:  # Or
            result = False
            for v in node.values:
                result = self.visit(v)
                if result:
                    return result
            return result

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Operator {type(node.op).__name__} not allowed")
        return op_func(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unary operator {type(node.op).__name__} not allowed")
        return op_func(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _SAFE_OPS.get(type(op))
            if op_func is None:
                raise TypeError(f"Comparator {type(op).__name__} not allowed")
            right = self.visit(comparator)
            if not op_func(left, right):
                return False
            left = right
        return True

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if not callable(func):
            raise TypeError(f"{func!r} is not callable")
        # Only allow builtins
        if func not in _SAFE_BUILTINS.values():
            raise TypeError(f"Calling {func!r} is not allowed")
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def generic_visit(self, node: ast.AST) -> Any:
        raise TypeError(f"AST node type {type(node).__name__} not allowed")


def _safe_eval(expression: str, context: Dict[str, Any]) -> Any:
    """Safely evaluate a simple expression against a context dict.

    Only allows: comparisons, boolean ops, arithmetic, string ops,
    attribute access on dict values, subscripts, and whitelisted builtins.
    """
    tree = ast.parse(expression, mode='eval')
    visitor = _SafeEvalVisitor(context)
    return visitor.visit(tree)


class TriggerType(Enum):
    """Types of proactive triggers."""
    USER_INACTIVITY = "user_inactivity"
    OPPORTUNITY_DETECTION = "opportunity_detection"
    LEARNING_OPPORTUNITY = "learning_opportunity"
    SYSTEM_OPTIMIZATION = "system_optimization"
    KNOWLEDGE_GAP = "knowledge_gap"
    USER_PATTERN = "user_pattern"
    TIME_BASED = "time_based"
    EVENT_DRIVEN = "event_driven"


class AlertType(Enum):
    """Types of motivation alerts."""
    LOW_MOTIVATION = "low_motivation"
    DRIVE_IMBALANCE = "drive_imbalance"
    DECLINING_TREND = "declining_trend"


class Severity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Condition:
    """Represents a trigger condition."""
    expression: str
    evaluator: Optional[Callable] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against context."""
        if self.evaluator:
            return self.evaluator(context)
        
        try:
            return _safe_eval(self.expression, context)
        except (SyntaxError, NameError, TypeError, ValueError, OSError, RuntimeError) as e:
            logger.warning(f"Condition evaluation failed for expression '{self.expression}': {e}")
            return False


@dataclass
class TriggerPattern:
    """Pattern for detecting proactive trigger conditions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger_type: TriggerType = TriggerType.EVENT_DRIVEN
    conditions: List[Condition] = field(default_factory=list)
    threshold: float = 0.7
    urgency: float = 0.5
    action_template: str = ""
    cooldown_seconds: int = 300  # 5 minutes
    max_daily_activations: int = 10
    
    # Tracking
    activation_history: Deque[datetime] = field(default_factory=lambda: deque(maxlen=100))
    last_activated: Optional[datetime] = None
    
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate pattern against context and return match score."""
        if not self.conditions:
            return 0.0
        
        # Check cooldown
        if self.last_activated:
            cooldown_elapsed = (datetime.now() - self.last_activated).total_seconds()
            if cooldown_elapsed < self.cooldown_seconds:
                return 0.0
        
        # Check daily limit
        today_activations = sum(
            1 for t in self.activation_history
            if t.date() == datetime.now().date()
        )
        if today_activations >= self.max_daily_activations:
            return 0.0
        
        # Evaluate conditions
        matches = sum(1 for c in self.conditions if c.evaluate(context))
        match_ratio = matches / len(self.conditions)
        
        return match_ratio
    
    def record_activation(self) -> None:
        """Record pattern activation."""
        self.last_activated = datetime.now()
        self.activation_history.append(datetime.now())
    
    def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action based on pattern and context."""
        return {
            'type': self.trigger_type.value,
            'template': self.action_template,
            'context': context,
            'generated_at': datetime.now().isoformat()
        }


@dataclass
class TriggeredAction:
    """Represents a triggered proactive action."""
    pattern: TriggerPattern
    confidence: float
    urgency: float
    suggested_action: Dict[str, Any]
    context_snapshot: Dict[str, Any]
    triggered_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_result: Optional[Dict] = None


@dataclass
class MotivationAlert:
    """Alert for motivation-related issues."""
    type: AlertType
    severity: Severity
    message: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class AgentContext:
    """Context for trigger evaluation."""
    # User state
    user_inactive_duration: float = 0.0  # seconds
    user_in_meeting: bool = False
    user_active_hours: List[int] = field(default_factory=list)
    
    # Task state
    pending_tasks_count: int = 0
    pending_tasks: List[Dict] = field(default_factory=list)
    
    # Motivation state
    motivation: Dict[str, float] = field(default_factory=dict)
    
    # Agent state
    workload: float = 0.5
    capacity_available: float = 0.5
    recent_changes: int = 0
    
    # Opportunities
    opportunities: List[Dict] = field(default_factory=list)
    
    # System state
    system_performance: float = 1.0
    
    # Time
    current_time: datetime = field(default_factory=datetime.now)
    
    def to_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of context."""
        return {
            'user_inactive_duration': self.user_inactive_duration,
            'pending_tasks_count': self.pending_tasks_count,
            'motivation': self.motivation,
            'workload': self.workload,
            'timestamp': self.current_time.isoformat()
        }


class ProactiveTriggerEngine:
    """
    Detects conditions that should trigger proactive behavior.
    Monitors environment, user patterns, and internal state.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.trigger_patterns: Dict[str, TriggerPattern] = {}
        self.trigger_history: Deque[TriggeredAction] = deque(maxlen=500)
        self.sensitivity_threshold = self.config.get('trigger_sensitivity', 0.6)
        self.min_trigger_interval = self.config.get('min_trigger_interval', 60)
        self.max_daily_triggers = self.config.get('max_daily_triggers', 50)
        
        self.last_trigger_time: Optional[datetime] = None
        
        # Initialize default patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default trigger patterns."""
        patterns = [
            self._create_user_inactivity_trigger(),
            self._create_opportunity_detection_trigger(),
            self._create_learning_trigger(),
            self._create_system_optimization_trigger(),
            self._create_knowledge_gap_trigger()
        ]
        
        for pattern in patterns:
            self.register_trigger(pattern)
    
    def _create_user_inactivity_trigger(self) -> TriggerPattern:
        """Trigger when user has been inactive but may need assistance."""
        
        def evaluator(context: Dict[str, Any]) -> bool:
            return (
                context.get('user_inactive_duration', 0) > 1800 and  # 30 min
                context.get('pending_tasks_count', 0) > 0 and
                not context.get('user_in_meeting', False)
            )
        
        return TriggerPattern(
            name="User Inactivity Proactive Help",
            description="Offer help when user is inactive but has pending tasks",
            trigger_type=TriggerType.USER_INACTIVITY,
            conditions=[
                Condition("user_inactive_duration > 1800", evaluator),
                Condition("pending_tasks_count > 0"),
                Condition("not user_in_meeting")
            ],
            threshold=0.75,
            urgency=0.6,
            action_template="Generate proactive assistance offer for pending tasks",
            cooldown_seconds=1800  # 30 minutes
        )
    
    def _create_opportunity_detection_trigger(self) -> TriggerPattern:
        """Trigger when new opportunity is detected."""
        
        def evaluator(context: Dict[str, Any]) -> bool:
            opportunities = context.get('opportunities', [])
            for op in opportunities:
                if op.get('alignment_with_goals', 0) > 0.7:
                    return True
            return False
        
        return TriggerPattern(
            name="New Opportunity Detection",
            description="Act on newly detected opportunities",
            trigger_type=TriggerType.OPPORTUNITY_DETECTION,
            conditions=[
                Condition("len(opportunities) > 0"),
                Condition("any(op.alignment_with_goals > 0.7 for op in opportunities)", evaluator),
                Condition("capacity_available > 0.3")
            ],
            threshold=0.8,
            urgency=0.7,
            action_template="Evaluate and potentially pursue detected opportunity",
            cooldown_seconds=600  # 10 minutes
        )
    
    def _create_learning_trigger(self) -> TriggerPattern:
        """Trigger when learning opportunity arises."""
        
        def evaluator(context: Dict[str, Any]) -> bool:
            motivation = context.get('motivation', {})
            return (
                motivation.get('curiosity', 0) > 0.7 and
                context.get('workload', 1.0) < 0.6
            )
        
        return TriggerPattern(
            name="Learning Opportunity",
            description="Pursue learning when curiosity is high and capacity exists",
            trigger_type=TriggerType.LEARNING_OPPORTUNITY,
            conditions=[
                Condition("motivation.curiosity > 0.7", evaluator),
                Condition("workload < 0.6"),
                Condition("capacity_available > 0.3")
            ],
            threshold=0.7,
            urgency=0.4,
            action_template="Initiate curiosity-driven learning activity",
            cooldown_seconds=14400  # 4 hours
        )
    
    def _create_system_optimization_trigger(self) -> TriggerPattern:
        """Trigger when system optimization is possible."""
        
        def evaluator(context: Dict[str, Any]) -> bool:
            return context.get('system_performance', 1.0) < 0.8
        
        return TriggerPattern(
            name="System Optimization Opportunity",
            description="Proactively optimize when inefficiencies detected",
            trigger_type=TriggerType.SYSTEM_OPTIMIZATION,
            conditions=[
                Condition("system_performance < 0.8", evaluator),
                Condition("capacity_available > 0.2")
            ],
            threshold=0.75,
            urgency=0.8,
            action_template="Analyze and implement system optimization",
            cooldown_seconds=3600  # 1 hour
        )
    
    def _create_knowledge_gap_trigger(self) -> TriggerPattern:
        """Trigger when knowledge gap is identified."""
        return TriggerPattern(
            name="Knowledge Gap Identified",
            description="Address identified knowledge gaps",
            trigger_type=TriggerType.KNOWLEDGE_GAP,
            conditions=[
                Condition("knowledge_gaps_identified == True"),
                Condition("workload < 0.7")
            ],
            threshold=0.7,
            urgency=0.5,
            action_template="Plan and execute knowledge acquisition",
            cooldown_seconds=7200  # 2 hours
        )
    
    def register_trigger(self, pattern: TriggerPattern) -> None:
        """Register a new trigger pattern."""
        self.trigger_patterns[pattern.id] = pattern
    
    def unregister_trigger(self, pattern_id: str) -> bool:
        """Unregister a trigger pattern."""
        if pattern_id in self.trigger_patterns:
            del self.trigger_patterns[pattern_id]
            return True
        return False
    
    def evaluate_triggers(self, context: AgentContext) -> List[TriggeredAction]:
        """Evaluate all triggers and return triggered actions."""
        
        triggered_actions = []
        context_dict = self._context_to_dict(context)
        
        # Check global rate limiting
        if self.last_trigger_time:
            elapsed = (datetime.now() - self.last_trigger_time).total_seconds()
            if elapsed < self.min_trigger_interval:
                return []
        
        # Check daily limit
        today_triggers = sum(
            1 for t in self.trigger_history
            if t.triggered_at.date() == datetime.now().date()
        )
        if today_triggers >= self.max_daily_triggers:
            return []
        
        for pattern in self.trigger_patterns.values():
            match_score = pattern.evaluate(context_dict)
            
            if match_score >= pattern.threshold:
                confidence = self._calculate_confidence(pattern, context_dict, match_score)
                
                if confidence >= self.sensitivity_threshold:
                    triggered_action = TriggeredAction(
                        pattern=pattern,
                        confidence=confidence,
                        urgency=pattern.urgency,
                        suggested_action=pattern.generate_action(context_dict),
                        context_snapshot=context.to_snapshot()
                    )
                    triggered_actions.append(triggered_action)
                    pattern.record_activation()
        
        # Sort by urgency and confidence
        triggered_actions.sort(
            key=lambda x: (x.urgency * 0.6 + x.confidence * 0.4),
            reverse=True
        )
        
        # Update last trigger time
        if triggered_actions:
            self.last_trigger_time = datetime.now()
        
        # Record in history
        for action in triggered_actions:
            self.trigger_history.append(action)
        
        return triggered_actions
    
    def _context_to_dict(self, context: AgentContext) -> Dict[str, Any]:
        """Convert AgentContext to dictionary."""
        return {
            'user_inactive_duration': context.user_inactive_duration,
            'user_in_meeting': context.user_in_meeting,
            'user_active_hours': context.user_active_hours,
            'pending_tasks_count': context.pending_tasks_count,
            'pending_tasks': context.pending_tasks,
            'motivation': context.motivation,
            'workload': context.workload,
            'capacity_available': context.capacity_available,
            'recent_changes': context.recent_changes,
            'opportunities': context.opportunities,
            'system_performance': context.system_performance,
            'current_time': context.current_time
        }
    
    def _calculate_confidence(self, pattern: TriggerPattern,
                             context: Dict[str, Any],
                             match_score: float) -> float:
        """Calculate confidence in trigger activation."""
        
        # Base confidence from match score
        confidence = match_score
        
        # Adjust based on historical accuracy
        if pattern.activation_history:
            historical_accuracy = self._get_historical_accuracy(pattern)
            confidence = 0.7 * confidence + 0.3 * historical_accuracy
        
        # Adjust based on context stability
        if context.get('recent_changes', 0) > 5:
            confidence *= 0.9  # Reduce confidence in volatile contexts
        
        return min(1.0, confidence)
    
    def _get_historical_accuracy(self, pattern: TriggerPattern) -> float:
        """Get historical accuracy of pattern."""
        # Placeholder - would track actual success rate
        return 0.75
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get statistics about trigger activations."""
        return {
            'total_patterns': len(self.trigger_patterns),
            'today_activations': sum(
                1 for t in self.trigger_history
                if t.triggered_at.date() == datetime.now().date()
            ),
            'total_activations': len(self.trigger_history),
            'pattern_breakdown': {
                pattern_id: len(pattern.activation_history)
                for pattern_id, pattern in self.trigger_patterns.items()
            }
        }


# Singleton instance
_trigger_engine: Optional[ProactiveTriggerEngine] = None


def get_trigger_engine(config: Optional[Dict] = None) -> ProactiveTriggerEngine:
    """Get or create the global trigger engine instance."""
    global _trigger_engine
    if _trigger_engine is None:
        _trigger_engine = ProactiveTriggerEngine(config)
    return _trigger_engine


if __name__ == "__main__":
    # Example usage
    engine = ProactiveTriggerEngine()
    
    context = AgentContext(
        user_inactive_duration=2400,  # 40 minutes
        pending_tasks_count=3,
        user_in_meeting=False,
        motivation={'curiosity': 0.8, 'competence': 0.6},
        workload=0.4,
        capacity_available=0.6,
        opportunities=[
            {'alignment_with_goals': 0.8, 'value': 0.9}
        ]
    )
    
    triggered = engine.evaluate_triggers(context)
    
    print(f"Triggered {len(triggered)} actions:")
    for action in triggered:
        print(f"  - {action.pattern.name} (Confidence: {action.confidence:.2f})")
    
    print(f"\nStats: {engine.get_trigger_stats()}")
