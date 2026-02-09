"""
Advanced Planning Loop Implementation
Hierarchical Planning with Dynamic Replanning
OpenClaw Windows 10 AI Agent System

This module implements the complete planning loop with:
- Hierarchical goal management (HARD goals)
- Dynamic replanning triggers and strategies
- Contingency planning (Plan B generation)
- Resource-constrained planning
- Temporal planning with deadlines
- Multi-objective optimization
- Plan execution monitoring
- Plan quality assessment
"""

import asyncio
import os
import uuid
import json
import logging
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set, Tuple, Union
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class GoalLevel(Enum):
    """Five-level goal hierarchy."""
    MISSION = 1      # Highest-level strategic goals
    OBJECTIVE = 2    # Major outcomes to achieve
    TASK = 3         # Concrete work units
    SUBTASK = 4      # Decomposed task components
    ACTION = 5       # Atomic executable actions


class HardnessLevel(Enum):
    """Goal commitment levels."""
    CRITICAL = 1     # Must succeed, abort on failure
    IMPORTANT = 2    # Should succeed, retry on failure
    NICE_TO_HAVE = 3 # Optional, skip on failure


class AdaptabilityType(Enum):
    """Goal adaptability types."""
    FIXED = auto()      # Cannot be changed
    ADAPTABLE = auto()  # Can be modified with constraints
    DYNAMIC = auto()    # Fully adaptable


class GoalStatus(Enum):
    """Goal execution status."""
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()
    BLOCKED = auto()


class ReplanningTrigger(Enum):
    """Events that trigger dynamic replanning."""
    # Execution Triggers
    GOAL_FAILED = auto()
    GOAL_COMPLETED = auto()
    TIMEOUT = auto()
    RESOURCE_EXHAUSTED = auto()
    
    # Environment Triggers
    CONTEXT_CHANGED = auto()
    NEW_GOAL_ADDED = auto()
    PRIORITY_CHANGED = auto()
    DEPENDENCY_VIOLATED = auto()
    
    # Quality Triggers
    PLAN_DEGRADED = auto()
    BETTER_PATH_FOUND = auto()
    PREDICTED_FAILURE = auto()
    
    # System Triggers
    SYSTEM_OVERLOAD = auto()
    EXTERNAL_INTERRUPT = auto()
    USER_OVERRIDE = auto()


class ReplanningStrategy(Enum):
    """Strategies for replanning."""
    LOCAL_REPAIR = auto()
    COMPONENT_REPLACEMENT = auto()
    REORDERING = auto()
    PARTIAL_REGENERATION = auto()
    FULL_REGENERATION = auto()
    EMERGENCY_FALLBACK = auto()


class ExecutionStatus(Enum):
    """Plan execution status."""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    COMPLETED_WITH_FAILURES = auto()
    FAILED = auto()
    ABORTED = auto()


class ObjectiveType(Enum):
    """Types of planning objectives."""
    TIME = auto()
    COST = auto()
    QUALITY = auto()
    RELIABILITY = auto()
    RESOURCE_EFFICIENCY = auto()
    USER_SATISFACTION = auto()
    FAIRNESS = auto()
    ENERGY = auto()


class TradeOffStrategy(Enum):
    """Strategies for multi-objective trade-offs."""
    WEIGHTED_SUM = auto()
    PARETO = auto()
    GOAL_PROGRAMMING = auto()
    MIN_MAX = auto()


class PlanQualityDimension(Enum):
    """Dimensions of plan quality."""
    FEASIBILITY = auto()
    EFFICIENCY = auto()
    ROBUSTNESS = auto()
    COMPLETENESS = auto()
    CONSISTENCY = auto()
    OPTIMALITY = auto()
    SIMPLICITY = auto()
    ADAPTABILITY = auto()


class IssueSeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ResourceProfile:
    """Resource requirements and consumption profile."""
    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    disk_io_mbps: float = 10.0
    network_mbps: float = 10.0
    api_calls: Dict[str, int] = field(default_factory=dict)
    rate_limits: Dict[str, float] = field(default_factory=dict)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    max_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    external_services: List[str] = field(default_factory=list)
    file_handles: int = 5
    estimated_cost: float = 0.0
    
    def is_high_demand(self) -> bool:
        """Check if this is a high-demand resource profile."""
        return (
            self.cpu_cores > 4 or
            self.memory_mb > 2048 or
            len(self.external_services) > 3
        )


@dataclass
class SuccessCriterion:
    """Criterion for goal success."""
    name: str
    description: str
    evaluator: Optional[Callable] = None
    required: bool = True


@dataclass
class FailureCondition:
    """Condition that causes goal failure."""
    name: str
    description: str
    detector: Optional[Callable] = None


@dataclass
class HARDGoal:
    """
    Hierarchical Adaptive Resource-aware Decomposable Goal
    """
    goal_id: str
    name: str
    description: str = ""
    
    # Hierarchy
    level: GoalLevel = GoalLevel.TASK
    parent_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    
    # HARD Properties
    hardness: HardnessLevel = HardnessLevel.IMPORTANT
    adaptability: AdaptabilityType = AdaptabilityType.ADAPTABLE
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)
    decomposable: bool = True
    
    # State
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    must_start_after: Optional[datetime] = None
    must_complete_before: Optional[datetime] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Success Criteria
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    failure_conditions: List[FailureCondition] = field(default_factory=list)
    
    # Metadata
    priority: int = 50
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'goal_id': self.goal_id,
            'name': self.name,
            'description': self.description,
            'level': self.level.name,
            'parent_id': self.parent_id,
            'subgoals': self.subgoals,
            'hardness': self.hardness.name,
            'adaptability': self.adaptability.name,
            'decomposable': self.decomposable,
            'status': self.status.name,
            'progress': self.progress,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'estimated_duration_seconds': self.estimated_duration.total_seconds(),
            'depends_on': self.depends_on,
            'blocks': self.blocks,
            'priority': self.priority,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HARDGoal':
        """Create from dictionary."""
        goal = cls(
            goal_id=data['goal_id'],
            name=data['name'],
            description=data.get('description', ''),
            level=GoalLevel[data.get('level', 'TASK')],
            parent_id=data.get('parent_id'),
            subgoals=data.get('subgoals', []),
            hardness=HardnessLevel[data.get('hardness', 'IMPORTANT')],
            adaptability=AdaptabilityType[data.get('adaptability', 'ADAPTABLE')],
            decomposable=data.get('decomposable', True),
            status=GoalStatus[data.get('status', 'PENDING')],
            progress=data.get('progress', 0.0),
            priority=data.get('priority', 50),
            depends_on=data.get('depends_on', []),
            blocks=data.get('blocks', []),
            tags=data.get('tags', [])
        )
        if data.get('deadline'):
            goal.deadline = datetime.fromisoformat(data['deadline'])
        if data.get('estimated_duration_seconds'):
            goal.estimated_duration = timedelta(seconds=data['estimated_duration_seconds'])
        return goal


@dataclass
class TriggerEvent:
    """A replanning trigger event."""
    trigger_type: ReplanningTrigger
    timestamp: datetime
    source: str
    severity: int
    context: Dict[str, Any] = field(default_factory=dict)
    affected_goals: List[str] = field(default_factory=list)


@dataclass
class ScheduledTask:
    """A scheduled task in an execution plan."""
    goal_id: str
    start_time: datetime
    estimated_end: datetime
    latest_end: Optional[datetime] = None
    flexibility: float = 0.0
    allocated_resources: Optional[ResourceProfile] = None


@dataclass
class ExecutionSchedule:
    """Schedule for plan execution."""
    tasks: List[ScheduledTask] = field(default_factory=list)
    
    def add_task(self, goal_id: str, start_time: datetime, 
                 estimated_end: datetime, latest_end: Optional[datetime] = None,
                 flexibility: float = 0.0):
        """Add a task to the schedule."""
        self.tasks.append(ScheduledTask(
            goal_id=goal_id,
            start_time=start_time,
            estimated_end=estimated_end,
            latest_end=latest_end,
            flexibility=flexibility
        ))
    
    def get_task(self, goal_id: str) -> Optional[ScheduledTask]:
        """Get task by goal ID."""
        for task in self.tasks:
            if task.goal_id == goal_id:
                return task
        return None


@dataclass
class ResourceAllocation:
    """Resource allocation for plan execution."""
    allocations: Dict[str, ResourceProfile] = field(default_factory=dict)
    
    def allocate(self, goal_id: str, profile: ResourceProfile):
        """Allocate resources to a goal."""
        self.allocations[goal_id] = profile
    
    def get_allocation(self, goal_id: str) -> Optional[ResourceProfile]:
        """Get resource allocation for a goal."""
        return self.allocations.get(goal_id)
    
    def total(self) -> ResourceProfile:
        """Get total allocated resources."""
        total = ResourceProfile()
        for profile in self.allocations.values():
            total.cpu_cores += profile.cpu_cores
            total.memory_mb += profile.memory_mb
            total.disk_io_mbps += profile.disk_io_mbps
            total.network_mbps += profile.network_mbps
            total.estimated_cost += profile.estimated_cost
        return total


@dataclass
class ContingencyPlan:
    """Pre-computed alternative plan for handling failures."""
    plan_id: str
    name: str
    trigger_conditions: List[Dict] = field(default_factory=list)
    alternative_goals: List[HARDGoal] = field(default_factory=list)
    activation_threshold: float = 0.5
    precomputed_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    validity_duration: timedelta = field(default_factory=lambda: timedelta(hours=24))
    expected_success_rate: float = 0.75
    expected_completion_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    def is_valid(self) -> bool:
        """Check if contingency plan is still valid."""
        return datetime.now() - self.precomputed_at < self.validity_duration


@dataclass
class ExecutionPlan:
    """A complete execution plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goals: List[HARDGoal] = field(default_factory=list)
    schedule: ExecutionSchedule = field(default_factory=ExecutionSchedule)
    resource_allocation: ResourceAllocation = field(default_factory=ResourceAllocation)
    contingencies: List[ContingencyPlan] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    quality_score: Optional[float] = None
    
    # Dependency tracking
    _goal_map: Dict[str, HARDGoal] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Initialize goal map."""
        self._goal_map = {g.goal_id: g for g in self.goals}
    
    def get_goal(self, goal_id: str) -> Optional[HARDGoal]:
        """Get goal by ID."""
        return self._goal_map.get(goal_id)
    
    def add_goal(self, goal: HARDGoal):
        """Add a goal to the plan."""
        self.goals.append(goal)
        self._goal_map[goal.goal_id] = goal
    
    def get_execution_order(self) -> List[str]:
        """Get topological execution order."""
        # Build dependency graph
        in_degree = {g.goal_id: 0 for g in self.goals}
        dependents = defaultdict(list)
        
        for goal in self.goals:
            for dep in goal.depends_on:
                if dep in in_degree:
                    in_degree[goal.goal_id] += 1
                    dependents[dep].append(goal.goal_id)
        
        # Topological sort
        queue = [gid for gid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda gid: self._goal_map[gid].priority, reverse=True)
            gid = queue.pop(0)
            order.append(gid)
            
            for dep_gid in dependents[gid]:
                in_degree[dep_gid] -= 1
                if in_degree[dep_gid] == 0:
                    queue.append(dep_gid)
        
        return order
    
    def get_planned_duration(self) -> timedelta:
        """Get total planned duration."""
        if not self.schedule.tasks:
            return timedelta(0)
        
        start = min(t.start_time for t in self.schedule.tasks)
        end = max(t.estimated_end for t in self.schedule.tasks)
        return end - start
    
    def copy(self) -> 'ExecutionPlan':
        """Create a copy of the plan."""
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goals=[HARDGoal(**asdict(g)) for g in self.goals],
            schedule=ExecutionSchedule(tasks=list(self.schedule.tasks)),
            resource_allocation=ResourceAllocation(
                allocations=dict(self.resource_allocation.allocations)
            ),
            contingencies=list(self.contingencies),
            quality_score=self.quality_score
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'plan_id': self.plan_id,
            'goals': [g.to_dict() for g in self.goals],
            'created_at': self.created_at.isoformat(),
            'quality_score': self.quality_score
        }


@dataclass
class ExecutionState:
    """State of a plan execution."""
    execution_id: str
    plan: ExecutionPlan
    start_time: datetime
    status: ExecutionStatus
    goal_states: Dict[str, GoalStatus] = field(default_factory=dict)
    progress: float = 0.0
    replanning_count: int = 0
    contingency_count: int = 0
    adaptation_time: timedelta = field(default_factory=timedelta)
    current_goal_id: Optional[str] = None
    
    def update_progress(self):
        """Update execution progress."""
        total = len(self.goal_states)
        if total == 0:
            self.progress = 0.0
            return
        
        completed = sum(1 for s in self.goal_states.values() 
                       if s == GoalStatus.COMPLETED)
        failed = sum(1 for s in self.goal_states.values() 
                    if s == GoalStatus.FAILED)
        
        self.progress = (completed + failed * 0.5) / total


@dataclass
class ExecutionResult:
    """Result of plan execution."""
    execution_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime
    completed_goals: List[str] = field(default_factory=list)
    failed_goals: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityIssue:
    """Quality issue found in plan assessment."""
    dimension: PlanQualityDimension
    severity: IssueSeverity
    description: str
    affected_components: List[str] = field(default_factory=list)


@dataclass
class ImprovementSuggestion:
    """Suggestion for plan improvement."""
    dimension: PlanQualityDimension
    description: str
    expected_improvement: float
    implementation_difficulty: int  # 1-10


@dataclass
class QualityAssessment:
    """Quality assessment for a plan."""
    plan_id: str
    assessed_at: datetime
    dimension_scores: Dict[PlanQualityDimension, float]
    overall_score: float
    issues: List[QualityIssue] = field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)


@dataclass
class PlanningConfig:
    """Configuration for the planning loop."""
    
    # Replanning settings
    replanning_enabled: bool = True
    replanning_cooldown_seconds: int = 30
    min_replanning_severity: int = 5
    
    # Contingency settings
    contingency_planning_enabled: bool = True
    max_contingencies_per_plan: int = 5
    contingency_validity_hours: int = 24
    
    # Resource settings
    resource_monitoring_enabled: bool = True
    resource_check_interval_seconds: int = 10
    
    # Temporal settings
    deadline_monitoring_enabled: bool = True
    deadline_warning_thresholds: List[float] = None
    
    # Optimization settings
    optimization_enabled: bool = True
    max_candidate_plans: int = 10
    pareto_front_size: int = 5
    
    # Quality settings
    quality_assessment_enabled: bool = True
    min_quality_threshold: float = 0.6
    
    def __post_init__(self):
        if self.deadline_warning_thresholds is None:
            self.deadline_warning_thresholds = [0.5, 0.75, 0.9, 0.95]

    @classmethod
    def from_yaml(cls, path: str = None) -> 'PlanningConfig':
        """Load PlanningConfig from YAML with fallback to defaults."""
        data = {}
        # Try ConfigLoader first
        try:
            from config_loader import get_config
            data = get_config("planning_loop_config") or {}
        except (ImportError, Exception):
            if path is None:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'planning_loop_config.yaml')
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                return cls()
        pl = data.get('planning_loop', {})
        replanning = pl.get('replanning', {})
        return cls(
            replanning_cooldown_seconds=replanning.get('cooldown_seconds', 30),
            min_replanning_severity=replanning.get('min_severity', 5),
        )


# =============================================================================
# HIERARCHICAL GOAL MANAGER
# =============================================================================

class HierarchicalGoalManager:
    """
    Manages hierarchical goal decomposition and organization.
    """
    
    DECOMPOSITION_STRATEGIES = {
        'sequential': 'Break into sequential steps',
        'parallel': 'Break into parallelizable components',
        'conditional': 'Break by conditions/branches',
        'recursive': 'Break until atomic actions reached',
        'functional': 'Break by functional domains'
    }
    
    def __init__(self):
        self.goals: Dict[str, HARDGoal] = {}
        self.goal_hierarchy: Dict[str, List[str]] = defaultdict(list)
    
    def register_goal(self, goal: HARDGoal):
        """Register a goal in the hierarchy."""
        self.goals[goal.goal_id] = goal
        if goal.parent_id:
            self.goal_hierarchy[goal.parent_id].append(goal.goal_id)
    
    def decompose_goal(self, goal: HARDGoal, 
                       target_level: GoalLevel = GoalLevel.ACTION,
                       llm_client=None) -> List[HARDGoal]:
        """
        Decompose a goal into subgoals.
        
        Args:
            goal: The goal to decompose
            target_level: Target decomposition level
            llm_client: Optional LLM client for intelligent decomposition
        
        Returns:
            List of decomposed subgoals
        """
        if goal.level.value >= target_level.value:
            return [goal]
        
        if not goal.decomposable:
            return [goal]
        
        # Use LLM for intelligent decomposition if available
        if llm_client and goal.level.value < GoalLevel.TASK.value:
            return self._llm_decomposition(goal, target_level, llm_client)
        
        # Use rule-based decomposition
        return self._rule_based_decomposition(goal, target_level)
    
    def _llm_decomposition(self, goal: HARDGoal, target_level: GoalLevel,
                           llm_client) -> List[HARDGoal]:
        """Use LLM for intelligent goal decomposition with retry and validation."""
        import re as _re

        base_prompt = f"""
        Decompose this goal into actionable steps:

        Goal: {goal.name}
        Description: {goal.description}
        Current Level: {goal.level.name}
        Target Level: {target_level.name}

        Provide a list of subgoals with:
        - name: Short name for the subgoal
        - description: Detailed description
        - estimated_duration_minutes: Estimated time in minutes
        - dependencies: List of indices of subgoals this depends on

        Return valid JSON array only.
        """

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    prompt = base_prompt + "\n\nIMPORTANT: Return valid JSON only, no markdown or extra text."
                else:
                    prompt = base_prompt

                response = llm_client.generate(prompt)

                # Try direct JSON parse
                try:
                    subgoals_data = json.loads(response)
                except json.JSONDecodeError:
                    # Try extracting JSON array from response via regex
                    match = _re.search(r'\[.*\]', response, _re.DOTALL)
                    if match:
                        subgoals_data = json.loads(match.group(0))
                    else:
                        raise json.JSONDecodeError("No JSON array found", response, 0)

                if not isinstance(subgoals_data, list) or not subgoals_data:
                    raise ValueError("LLM response is not a non-empty JSON array")

                # Validate required fields
                for item in subgoals_data:
                    if 'name' not in item:
                        raise ValueError("Subgoal missing required 'name' field")

                subgoals = []
                for i, sg_data in enumerate(subgoals_data):
                    subgoal = HARDGoal(
                        goal_id=f"{goal.goal_id}_sub_{i}",
                        name=sg_data['name'],
                        description=sg_data.get('description', ''),
                        level=GoalLevel(goal.level.value + 1),
                        parent_id=goal.goal_id,
                        estimated_duration=timedelta(minutes=sg_data.get('estimated_duration_minutes', 5)),
                        priority=goal.priority,
                        hardness=goal.hardness
                    )
                    subgoals.append(subgoal)

                # Set up dependencies
                for i, sg_data in enumerate(subgoals_data):
                    deps = sg_data.get('dependencies', [])
                    subgoals[i].depends_on = [
                        subgoals[d].goal_id for d in deps if 0 <= d < len(subgoals)
                    ]

                return subgoals

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                last_error = e
                logger.warning(f"LLM decomposition attempt {attempt + 1}/{max_retries} failed: {e}")

        # All retries failed, try regex extraction of key fields as last resort
        logger.warning(f"All {max_retries} LLM decomposition retries failed: {last_error}")
        logger.info("Falling back to rule-based decomposition")
        return self._rule_based_decomposition(goal, target_level)
    
    def _rule_based_decomposition(self, goal: HARDGoal, 
                                   target_level: GoalLevel) -> List[HARDGoal]:
        """Rule-based goal decomposition."""
        
        subgoals = []
        
        # Common decomposition patterns
        if goal.level == GoalLevel.MISSION:
            # Decompose mission into objectives
            patterns = [
                ("Assessment", "Assess current situation and requirements"),
                ("Planning", "Develop detailed action plan"),
                ("Execution", "Execute planned actions"),
                ("Verification", "Verify mission success"),
                ("Reporting", "Report mission outcomes")
            ]
        elif goal.level == GoalLevel.OBJECTIVE:
            # Decompose objective into tasks
            patterns = [
                ("Preparation", "Prepare necessary resources and conditions"),
                ("Core Action", "Perform the main objective action"),
                ("Validation", "Validate objective completion")
            ]
        elif goal.level == GoalLevel.TASK:
            # Decompose task into subtasks
            patterns = [
                ("Setup", "Set up environment and prerequisites"),
                ("Execute", "Execute the main task logic"),
                ("Cleanup", "Clean up and finalize")
            ]
        else:
            # Already at or below target level
            return [goal]
        
        for i, (name, desc) in enumerate(patterns):
            subgoal = HARDGoal(
                goal_id=f"{goal.goal_id}_{name.lower().replace(' ', '_')}",
                name=f"{goal.name} - {name}",
                description=desc,
                level=GoalLevel(goal.level.value + 1),
                parent_id=goal.goal_id,
                estimated_duration=goal.estimated_duration / len(patterns),
                priority=goal.priority,
                hardness=goal.hardness
            )
            if i > 0:
                subgoal.depends_on = [subgoals[i-1].goal_id]
            subgoals.append(subgoal)
        
        return subgoals
    
    def get_goal_tree(self, root_goal_id: str) -> Dict:
        """Get the goal tree starting from a root goal."""
        root = self.goals.get(root_goal_id)
        if not root:
            return {}
        
        def build_tree(goal_id: str) -> Dict:
            goal = self.goals.get(goal_id)
            if not goal:
                return {}
            
            children = self.goal_hierarchy.get(goal_id, [])
            return {
                'goal': goal,
                'children': [build_tree(cid) for cid in children]
            }
        
        return build_tree(root_goal_id)
    
    def find_critical_path(self, goal_ids: List[str]) -> List[str]:
        """Find the critical path through a set of goals."""
        # Build dependency graph
        graph = {gid: [] for gid in goal_ids}
        durations = {}
        
        for gid in goal_ids:
            goal = self.goals.get(gid)
            if goal:
                durations[gid] = goal.estimated_duration.total_seconds()
                for dep in goal.depends_on:
                    if dep in graph:
                        graph[dep].append(gid)
        
        # Calculate longest path using DFS
        memo = {}
        
        def longest_path(gid: str) -> Tuple[List[str], float]:
            if gid in memo:
                return memo[gid]
            
            children = graph.get(gid, [])
            if not children:
                memo[gid] = ([gid], durations.get(gid, 0))
                return memo[gid]
            
            best_path = [gid]
            best_duration = durations.get(gid, 0)
            
            for child in children:
                child_path, child_duration = longest_path(child)
                total_duration = durations.get(gid, 0) + child_duration
                
                if total_duration > best_duration:
                    best_duration = total_duration
                    best_path = [gid] + child_path
            
            memo[gid] = (best_path, best_duration)
            return memo[gid]
        
        # Find starting nodes
        all_deps = set()
        for gid in goal_ids:
            goal = self.goals.get(gid)
            if goal:
                all_deps.update(goal.depends_on)
        
        starts = [gid for gid in goal_ids if gid not in all_deps]
        
        # Find longest path from any start
        best_path = []
        best_duration = 0
        
        for start in starts:
            path, duration = longest_path(start)
            if duration > best_duration:
                best_duration = duration
                best_path = path
        
        return best_path


# =============================================================================
# DYNAMIC REPLANNING ENGINE
# =============================================================================

class DynamicReplanningEngine:
    """
    Monitors execution and triggers replanning when necessary.
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self.trigger_queue: asyncio.Queue = asyncio.Queue()
        self.is_replanning = False
        self.last_replan_time: Optional[datetime] = None
        self.replanning_history: List[Dict] = []
        self.trigger_handlers: Dict[ReplanningTrigger, Callable] = {
            ReplanningTrigger.GOAL_FAILED: self._handle_goal_failure,
            ReplanningTrigger.TIMEOUT: self._handle_timeout,
            ReplanningTrigger.CONTEXT_CHANGED: self._handle_context_change,
            ReplanningTrigger.NEW_GOAL_ADDED: self._handle_new_goal,
            ReplanningTrigger.PLAN_DEGRADED: self._handle_plan_degradation,
            ReplanningTrigger.PREDICTED_FAILURE: self._handle_predicted_failure,
        }
    
    async def trigger(self, trigger_type: ReplanningTrigger, 
                      context: Dict[str, Any] = None):
        """Trigger a replanning event."""
        
        event = TriggerEvent(
            trigger_type=trigger_type,
            timestamp=datetime.now(),
            source=context.get('source', 'unknown'),
            severity=context.get('severity', 5),
            context=context or {},
            affected_goals=context.get('affected_goals', [])
        )
        
        await self.trigger_queue.put(event)
        logger.info(f"Replanning trigger queued: {trigger_type.name}")
    
    async def monitor_and_replan(self, active_plan: ExecutionPlan,
                                  plan_generator: Callable) -> ExecutionPlan:
        """Main monitoring loop that watches for replanning triggers."""
        
        while True:
            try:
                trigger = await asyncio.wait_for(
                    self.trigger_queue.get(), 
                    timeout=1.0
                )
                
                if self._should_replan(trigger, active_plan):
                    new_plan = await self._execute_replanning(
                        trigger, active_plan, plan_generator
                    )
                    if new_plan:
                        active_plan = new_plan
                        
            except asyncio.TimeoutError:
                continue
            except (KeyError, ValueError, RuntimeError) as e:
                logger.error(f"Error in replanning monitor: {e}")
    
    def _should_replan(self, trigger: TriggerEvent, 
                       plan: ExecutionPlan) -> bool:
        """Determine if this trigger warrants replanning."""
        
        # Check severity threshold
        if trigger.severity >= 7:
            return True
        
        # Check if already replanning
        if self.is_replanning:
            return trigger.severity >= 9
        
        # Check replanning cooldown
        if self.last_replan_time:
            cooldown = (datetime.now() - self.last_replan_time).total_seconds()
            if cooldown < self.config.replanning_cooldown_seconds:
                return trigger.severity >= 8
        
        # Check if trigger affects active goals
        active_goals = self._get_active_goals(plan)
        affected_active = set(trigger.affected_goals) & set(active_goals)
        
        return len(affected_active) > 0
    
    def _get_active_goals(self, plan: ExecutionPlan) -> List[str]:
        """Get currently active goals in the plan."""
        return [
            g.goal_id for g in plan.goals 
            if g.status in [GoalStatus.PENDING, GoalStatus.ACTIVE]
        ]
    
    async def _execute_replanning(self, trigger: TriggerEvent,
                                   current_plan: ExecutionPlan,
                                   plan_generator: Callable) -> Optional[ExecutionPlan]:
        """Execute the replanning process."""
        
        self.is_replanning = True
        self.last_replan_time = datetime.now()
        
        try:
            logger.info(f"Executing replanning for trigger: {trigger.trigger_type.name}")
            
            # 1. Analyze current situation
            situation = self._analyze_situation(trigger, current_plan)
            
            # 2. Determine replanning strategy
            strategy = self._select_replanning_strategy(trigger, situation)
            logger.info(f"Selected strategy: {strategy.name}")
            
            # 3. Execute strategy
            new_plan = await self._execute_strategy(
                strategy, situation, current_plan, plan_generator
            )
            
            # 4. Record replanning
            self.replanning_history.append({
                'timestamp': datetime.now(),
                'trigger': trigger.trigger_type.name,
                'strategy': strategy.name,
                'success': new_plan is not None
            })
            
            return new_plan
            
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error(f"Replanning failed: {e}")
            return None
        finally:
            self.is_replanning = False
    
    def _analyze_situation(self, trigger: TriggerEvent, 
                           plan: ExecutionPlan) -> Dict:
        """Analyze the current situation for replanning."""
        
        situation = {
            'trigger': trigger,
            'remaining_goals': [g for g in plan.goals 
                               if g.status == GoalStatus.PENDING],
            'completed_goals': [g for g in plan.goals 
                               if g.status == GoalStatus.COMPLETED],
            'failed_goals': [g for g in plan.goals 
                            if g.status == GoalStatus.FAILED],
            'time_elapsed': datetime.now() - plan.created_at,
            'plan_deadline': self._get_plan_deadline(plan)
        }
        
        return situation
    
    def _get_plan_deadline(self, plan: ExecutionPlan) -> Optional[datetime]:
        """Get the earliest deadline from the plan."""
        deadlines = [g.deadline for g in plan.goals if g.deadline]
        return min(deadlines) if deadlines else None
    
    def _select_replanning_strategy(self, trigger: TriggerEvent,
                                     situation: Dict) -> ReplanningStrategy:
        """Select appropriate replanning strategy."""
        
        # Emergency fallback for critical failures
        if trigger.severity >= 9:
            return ReplanningStrategy.EMERGENCY_FALLBACK
        
        # Strategy based on trigger type
        if trigger.trigger_type == ReplanningTrigger.GOAL_FAILED:
            failed_goals = situation['failed_goals']
            if failed_goals:
                failed = failed_goals[0]
                if failed.hardness == HardnessLevel.CRITICAL:
                    return ReplanningStrategy.FULL_REGENERATION
                elif self._has_alternative_path(failed, situation):
                    return ReplanningStrategy.COMPONENT_REPLACEMENT
                else:
                    return ReplanningStrategy.LOCAL_REPAIR
        
        if trigger.trigger_type == ReplanningTrigger.NEW_GOAL_ADDED:
            new_goal = trigger.context.get('new_goal')
            if new_goal and new_goal.priority >= 90:
                return ReplanningStrategy.REORDERING
            else:
                return ReplanningStrategy.PARTIAL_REGENERATION
        
        if trigger.trigger_type == ReplanningTrigger.CONTEXT_CHANGED:
            impact = trigger.context.get('impact', 0.5)
            if impact > 0.7:
                return ReplanningStrategy.FULL_REGENERATION
            elif impact > 0.3:
                return ReplanningStrategy.PARTIAL_REGENERATION
            else:
                return ReplanningStrategy.LOCAL_REPAIR
        
        if trigger.trigger_type == ReplanningTrigger.TIMEOUT:
            return ReplanningStrategy.REORDERING
        
        return ReplanningStrategy.PARTIAL_REGENERATION
    
    def _has_alternative_path(self, goal: HARDGoal, situation: Dict) -> bool:
        """Check if there's an alternative path around a failed goal."""
        # Check if goal has siblings that can compensate
        remaining = situation['remaining_goals']
        return len(remaining) > 1
    
    async def _execute_strategy(self, strategy: ReplanningStrategy,
                                 situation: Dict, current_plan: ExecutionPlan,
                                 plan_generator: Callable) -> Optional[ExecutionPlan]:
        """Execute the selected replanning strategy."""
        
        strategies = {
            ReplanningStrategy.LOCAL_REPAIR: self._local_repair,
            ReplanningStrategy.COMPONENT_REPLACEMENT: self._component_replacement,
            ReplanningStrategy.REORDERING: self._reordering,
            ReplanningStrategy.PARTIAL_REGENERATION: self._partial_regeneration,
            ReplanningStrategy.FULL_REGENERATION: self._full_regeneration,
            ReplanningStrategy.EMERGENCY_FALLBACK: self._emergency_fallback
        }
        
        handler = strategies.get(strategy)
        if handler:
            return await handler(situation, current_plan, plan_generator)
        
        return None
    
    async def _local_repair(self, situation: Dict, plan: ExecutionPlan,
                            plan_generator: Callable) -> ExecutionPlan:
        """Make minimal changes to fix the immediate issue."""
        
        new_plan = plan.copy()
        failed_goals = situation['failed_goals']
        
        for failed in failed_goals:
            # Mark as pending for retry
            goal = new_plan.get_goal(failed.goal_id)
            if goal:
                goal.status = GoalStatus.PENDING
                # Increase estimated duration for retry
                goal.estimated_duration *= 1.5
        
        return new_plan
    
    async def _component_replacement(self, situation: Dict, plan: ExecutionPlan,
                                      plan_generator: Callable) -> ExecutionPlan:
        """Replace failed components with alternatives."""
        
        new_plan = plan.copy()
        failed_goals = situation['failed_goals']
        
        for failed in failed_goals:
            # Create simplified version as replacement
            simplified = HARDGoal(
                goal_id=f"{failed.goal_id}_alt",
                name=f"{failed.name} (Alternative)",
                description=f"Simplified version of: {failed.description}",
                level=failed.level,
                hardness=HardnessLevel.NICE_TO_HAVE,  # Reduce hardness
                estimated_duration=failed.estimated_duration * 0.7,
                priority=failed.priority - 10
            )
            
            # Replace in plan
            new_plan.goals = [g for g in new_plan.goals if g.goal_id != failed.goal_id]
            new_plan.add_goal(simplified)
        
        return new_plan
    
    async def _reordering(self, situation: Dict, plan: ExecutionPlan,
                          plan_generator: Callable) -> ExecutionPlan:
        """Reorder goals based on new priorities."""
        
        new_plan = plan.copy()
        
        # Re-sort goals by priority
        new_plan.goals.sort(key=lambda g: g.priority, reverse=True)
        
        return new_plan
    
    async def _partial_regeneration(self, situation: Dict, plan: ExecutionPlan,
                                     plan_generator: Callable) -> ExecutionPlan:
        """Regenerate part of the plan. Ensures the plan_generator coroutine
        completes before returning the new plan."""

        remaining_goals = situation['remaining_goals']

        if plan_generator and remaining_goals:
            # Ensure we properly await the result (handles both sync and async generators)
            result = plan_generator(remaining_goals)
            if asyncio.iscoroutine(result):
                new_partial = await result
            else:
                new_partial = result

            if new_partial and hasattr(new_partial, 'goals'):
                # Merge with completed goals
                new_plan = plan.copy()
                new_plan.goals = situation['completed_goals'] + new_partial.goals
                return new_plan

        return plan

    async def _full_regeneration(self, situation: Dict, plan: ExecutionPlan,
                                  plan_generator: Callable) -> ExecutionPlan:
        """Completely regenerate the plan. Ensures the plan_generator coroutine
        completes before returning the new plan."""

        remaining_goals = situation['remaining_goals']

        if plan_generator:
            result = plan_generator(remaining_goals)
            if asyncio.iscoroutine(result):
                new_plan = await result
            else:
                new_plan = result
            if new_plan and hasattr(new_plan, 'goals'):
                return new_plan

        return plan
    
    async def _emergency_fallback(self, situation: Dict, plan: ExecutionPlan,
                                   plan_generator: Callable) -> ExecutionPlan:
        """Emergency fallback - keep only critical goals."""
        
        new_plan = plan.copy()
        
        # Keep only critical goals
        new_plan.goals = [
            g for g in new_plan.goals 
            if g.hardness == HardnessLevel.CRITICAL and 
               g.status != GoalStatus.COMPLETED
        ]
        
        return new_plan
    
    # Trigger handlers
    async def _handle_goal_failure(self, context: Dict):
        """Handle goal failure trigger."""
        logger.warning(f"Goal failed: {context.get('goal_id')}")
    
    async def _handle_timeout(self, context: Dict):
        """Handle timeout trigger."""
        logger.warning(f"Timeout: {context.get('details')}")
    
    async def _handle_context_change(self, context: Dict):
        """Handle context change trigger."""
        logger.info(f"Context changed: {context.get('changes')}")
    
    async def _handle_new_goal(self, context: Dict):
        """Handle new goal trigger."""
        logger.info(f"New goal added: {context.get('goal_id')}")
    
    async def _handle_plan_degradation(self, context: Dict):
        """Handle plan degradation trigger."""
        logger.warning(f"Plan degraded: {context.get('quality_drop')}")
    
    async def _handle_predicted_failure(self, context: Dict):
        """Handle predicted failure trigger."""
        logger.warning(f"Predicted failure: {context.get('prediction')}")


# =============================================================================
# CONTINGENCY PLANNER
# =============================================================================

class ContingencyPlanner:
    """
    Generates and manages contingency plans (Plan B options).
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self.contingency_plans: Dict[str, List[ContingencyPlan]] = {}
    
    async def generate_contingency_plans(self, primary_plan: ExecutionPlan) -> List[ContingencyPlan]:
        """Generate contingency plans for a primary plan."""
        
        contingencies = []
        
        # Identify failure points
        failure_points = self._identify_failure_points(primary_plan)
        
        for point in failure_points[:self.config.max_contingencies_per_plan]:
            contingency = await self._generate_contingency(point, primary_plan)
            if contingency:
                contingencies.append(contingency)
        
        # Store for the plan
        self.contingency_plans[primary_plan.plan_id] = contingencies
        
        return contingencies
    
    def _identify_failure_points(self, plan: ExecutionPlan) -> List[Dict]:
        """Identify potential failure points using heuristic risk assessment
        based on task complexity (dependencies, resources, criticality)."""

        failure_points = []

        for goal in plan.goals:
            risk_score = 0.0
            reasons = []

            # Criticality factor (scaled by hardness level)
            hardness_risk = {
                HardnessLevel.CRITICAL: 0.35,
                HardnessLevel.IMPORTANT: 0.15,
                HardnessLevel.NICE_TO_HAVE: 0.05
            }
            criticality = hardness_risk.get(goal.hardness, 0.1)
            risk_score += criticality
            if goal.hardness == HardnessLevel.CRITICAL:
                reasons.append("critical_goal")

            # External dependency factor (each external service adds risk)
            ext_count = len(goal.resource_profile.external_services)
            if ext_count > 0:
                ext_risk = min(0.1 * ext_count, 0.4)
                risk_score += ext_risk
                reasons.append(f"external_dependencies({ext_count})")

            # Resource intensity factor
            if goal.resource_profile.is_high_demand():
                risk_score += 0.15
                reasons.append("resource_intensive")

            # Dependency complexity factor (more deps = more fragile)
            dep_count = len(goal.depends_on)
            if dep_count > 0:
                dep_risk = min(0.05 * dep_count, 0.3)
                risk_score += dep_risk
                if dep_count > 3:
                    reasons.append(f"high_dependency({dep_count})")

            # Duration factor (long tasks are riskier)
            duration_hours = goal.estimated_duration.total_seconds() / 3600
            if duration_hours > 1:
                risk_score += min(0.1 * duration_hours, 0.2)
                reasons.append("long_duration")

            # Tight deadline factor
            if goal.deadline:
                time_to_deadline = (goal.deadline - datetime.now()).total_seconds()
                if time_to_deadline > 0:
                    slack_ratio = time_to_deadline / max(goal.estimated_duration.total_seconds(), 1)
                    if slack_ratio < 1.5:
                        risk_score += 0.15
                        reasons.append("tight_deadline")

            if risk_score > 0.25:
                failure_points.append({
                    'goal_id': goal.goal_id,
                    'risk_score': min(risk_score, 1.0),
                    'reasons': reasons,
                    'goal': goal
                })

        failure_points.sort(key=lambda p: p['risk_score'], reverse=True)
        return failure_points
    
    async def _generate_contingency(self, failure_point: Dict,
                                     primary_plan: ExecutionPlan) -> Optional[ContingencyPlan]:
        """Generate a contingency plan for a specific failure point."""
        
        goal = failure_point['goal']
        
        # Generate alternative approaches
        alternatives = self._generate_alternatives(goal)
        
        if not alternatives:
            return None
        
        return ContingencyPlan(
            plan_id=f"contingency_{goal.goal_id}",
            name=f"Contingency for {goal.name}",
            trigger_conditions=[{
                'type': 'goal_failure',
                'target': goal.goal_id
            }],
            alternative_goals=alternatives,
            activation_threshold=0.5,
            expected_success_rate=0.7,
            expected_completion_time=goal.estimated_duration * 1.2
        )
    
    def _generate_alternatives(self, goal: HARDGoal) -> List[HARDGoal]:
        """Generate alternative approaches for a goal."""
        
        alternatives = []
        
        # Alternative 1: Simplified version
        simplified = HARDGoal(
            goal_id=f"{goal.goal_id}_simplified",
            name=f"{goal.name} (Simplified)",
            description=f"Simplified version achieving core outcome",
            level=goal.level,
            hardness=HardnessLevel.NICE_TO_HAVE,
            estimated_duration=goal.estimated_duration * 0.6,
            priority=goal.priority - 20
        )
        alternatives.append(simplified)
        
        # Alternative 2: Degraded mode
        degraded = HARDGoal(
            goal_id=f"{goal.goal_id}_degraded",
            name=f"{goal.name} (Degraded)",
            description=f"Partial completion with reduced scope",
            level=goal.level,
            hardness=HardnessLevel.NICE_TO_HAVE,
            estimated_duration=goal.estimated_duration * 0.4,
            priority=goal.priority - 30
        )
        alternatives.append(degraded)
        
        return alternatives
    
    def get_contingency(self, plan_id: str, goal_id: str) -> Optional[ContingencyPlan]:
        """Get contingency plan for a specific goal."""
        
        contingencies = self.contingency_plans.get(plan_id, [])
        
        for cont in contingencies:
            for condition in cont.trigger_conditions:
                if condition.get('target') == goal_id:
                    return cont
        
        return None


# =============================================================================
# PLAN QUALITY ASSESSOR
# =============================================================================

class PlanQualityAssessor:
    """
    Assesses the quality of generated plans.
    """
    
    def __init__(self):
        # Load weights from planning_loop_config.yaml if available
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'planning_loop_config.yaml')
        weights = None
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            weights = data.get('planning_loop', {}).get('quality', {}).get('dimension_weights', {})
        except (OSError, yaml.YAMLError):
            pass

        if weights:
            self.dimension_weights = {
                PlanQualityDimension.FEASIBILITY: weights.get('feasibility', 0.25),
                PlanQualityDimension.EFFICIENCY: weights.get('efficiency', 0.15),
                PlanQualityDimension.ROBUSTNESS: weights.get('robustness', 0.15),
                PlanQualityDimension.COMPLETENESS: weights.get('completeness', 0.15),
                PlanQualityDimension.CONSISTENCY: weights.get('consistency', 0.10),
                PlanQualityDimension.OPTIMALITY: weights.get('optimality', 0.10),
                PlanQualityDimension.SIMPLICITY: weights.get('simplicity', 0.05),
                PlanQualityDimension.ADAPTABILITY: weights.get('adaptability', 0.05),
            }
        else:
            self.dimension_weights = {
                PlanQualityDimension.FEASIBILITY: 0.25,
                PlanQualityDimension.EFFICIENCY: 0.15,
                PlanQualityDimension.ROBUSTNESS: 0.15,
                PlanQualityDimension.COMPLETENESS: 0.15,
                PlanQualityDimension.CONSISTENCY: 0.10,
                PlanQualityDimension.OPTIMALITY: 0.10,
                PlanQualityDimension.SIMPLICITY: 0.05,
                PlanQualityDimension.ADAPTABILITY: 0.05,
            }
    
    async def assess_plan(self, plan: ExecutionPlan) -> QualityAssessment:
        """Assess the quality of a plan."""
        
        dimension_scores = {}
        all_issues = []
        all_suggestions = []
        
        # Assess each dimension
        for dimension in PlanQualityDimension:
            score, issues, suggestions = self._assess_dimension(dimension, plan)
            dimension_scores[dimension] = score
            all_issues.extend(issues)
            all_suggestions.extend(suggestions)
        
        # Calculate overall score
        overall = sum(
            dimension_scores[d] * self.dimension_weights[d]
            for d in PlanQualityDimension
        )
        
        return QualityAssessment(
            plan_id=plan.plan_id,
            assessed_at=datetime.now(),
            dimension_scores=dimension_scores,
            overall_score=overall,
            issues=all_issues,
            suggestions=all_suggestions
        )
    
    def _assess_dimension(self, dimension: PlanQualityDimension,
                          plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess a single quality dimension."""
        
        assessors = {
            PlanQualityDimension.FEASIBILITY: self._assess_feasibility,
            PlanQualityDimension.EFFICIENCY: self._assess_efficiency,
            PlanQualityDimension.ROBUSTNESS: self._assess_robustness,
            PlanQualityDimension.COMPLETENESS: self._assess_completeness,
            PlanQualityDimension.CONSISTENCY: self._assess_consistency,
            PlanQualityDimension.OPTIMALITY: self._assess_optimality,
            PlanQualityDimension.SIMPLICITY: self._assess_simplicity,
            PlanQualityDimension.ADAPTABILITY: self._assess_adaptability
        }
        
        assessor = assessors.get(dimension)
        if assessor:
            return assessor(plan)
        
        return 0.5, [], []
    
    def _assess_feasibility(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan feasibility."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for goals without estimated duration
        no_duration = [g for g in plan.goals if g.estimated_duration.total_seconds() == 0]
        if no_duration:
            score -= 0.1 * len(no_duration)
            issues.append(QualityIssue(
                dimension=PlanQualityDimension.FEASIBILITY,
                severity=IssueSeverity.MEDIUM,
                description=f"{len(no_duration)} goals without estimated duration",
                affected_components=[g.goal_id for g in no_duration]
            ))
        
        # Check for circular dependencies
        has_cycles = self._detect_cycles(plan)
        if has_cycles:
            score -= 0.3
            issues.append(QualityIssue(
                dimension=PlanQualityDimension.FEASIBILITY,
                severity=IssueSeverity.CRITICAL,
                description="Circular dependencies detected",
                affected_components=[]
            ))
        
        # Check deadline feasibility
        infeasible_deadlines = self._check_deadline_feasibility(plan)
        if infeasible_deadlines:
            score -= 0.2 * len(infeasible_deadlines)
            issues.append(QualityIssue(
                dimension=PlanQualityDimension.FEASIBILITY,
                severity=IssueSeverity.HIGH,
                description=f"{len(infeasible_deadlines)} goals with infeasible deadlines",
                affected_components=infeasible_deadlines
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_efficiency(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan efficiency."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for sequential execution opportunities
        parallelizable = self._find_parallelizable_goals(plan)
        if len(parallelizable) > len(plan.goals) * 0.3:
            score -= 0.1
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.EFFICIENCY,
                description=f"{len(parallelizable)} goals could potentially run in parallel",
                expected_improvement=0.15,
                implementation_difficulty=5
            ))
        
        # Check resource utilization
        total_resources = plan.resource_allocation.total()
        if total_resources.cpu_cores > 8 or total_resources.memory_mb > 8192:
            score -= 0.1
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.EFFICIENCY,
                description="High resource requirements - consider optimization",
                expected_improvement=0.1,
                implementation_difficulty=6
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_robustness(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan robustness."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for single points of failure (critical goals with many dependents)
        critical_goals = [g for g in plan.goals if g.hardness == HardnessLevel.CRITICAL]
        spofs = []
        
        for goal in critical_goals:
            dependent_count = sum(1 for g in plan.goals if goal.goal_id in g.depends_on)
            if dependent_count > 2:
                spofs.append(goal.goal_id)
        
        if spofs:
            score -= 0.15 * len(spofs)
            issues.append(QualityIssue(
                dimension=PlanQualityDimension.ROBUSTNESS,
                severity=IssueSeverity.HIGH,
                description=f"{len(spofs)} potential single points of failure",
                affected_components=spofs
            ))
        
        # Check contingency coverage
        if not plan.contingencies:
            score -= 0.2
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.ROBUSTNESS,
                description="Add contingency plans for critical goals",
                expected_improvement=0.2,
                implementation_difficulty=4
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_completeness(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan completeness."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for goals without success criteria
        no_criteria = [g for g in plan.goals if not g.success_criteria]
        if no_criteria:
            score -= 0.05 * len(no_criteria)
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.COMPLETENESS,
                description=f"Add success criteria to {len(no_criteria)} goals",
                expected_improvement=0.1,
                implementation_difficulty=3
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_consistency(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan consistency."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for orphaned dependencies
        all_goal_ids = {g.goal_id for g in plan.goals}
        orphaned = []
        
        for goal in plan.goals:
            for dep in goal.depends_on:
                if dep not in all_goal_ids:
                    orphaned.append((goal.goal_id, dep))
        
        if orphaned:
            score -= 0.1 * len(orphaned)
            issues.append(QualityIssue(
                dimension=PlanQualityDimension.CONSISTENCY,
                severity=IssueSeverity.MEDIUM,
                description=f"{len(orphaned)} orphaned dependencies",
                affected_components=[g for g, _ in orphaned]
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_optimality(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan optimality."""
        
        score = 0.8  # Base score - optimality is hard to measure
        issues = []
        suggestions = []
        
        # Check for obviously suboptimal orderings
        suboptimal = self._find_suboptimal_orderings(plan)
        if suboptimal:
            score -= 0.1 * len(suboptimal)
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.OPTIMALITY,
                description="Consider reordering goals for better efficiency",
                expected_improvement=0.1,
                implementation_difficulty=5
            ))
        
        return max(score, 0), issues, suggestions
    
    def _assess_simplicity(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan simplicity."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Penalize for complexity
        if len(plan.goals) > 20:
            score -= 0.1
        
        deep_nesting = sum(1 for g in plan.goals if len(g.depends_on) > 5)
        if deep_nesting > 0:
            score -= 0.05 * deep_nesting
        
        return max(score, 0), issues, suggestions
    
    def _assess_adaptability(self, plan: ExecutionPlan) -> Tuple[float, List[QualityIssue], List[ImprovementSuggestion]]:
        """Assess plan adaptability."""
        
        score = 1.0
        issues = []
        suggestions = []
        
        # Check for fixed goals
        fixed_goals = [g for g in plan.goals if g.adaptability == AdaptabilityType.FIXED]
        if len(fixed_goals) > len(plan.goals) * 0.5:
            score -= 0.1
            suggestions.append(ImprovementSuggestion(
                dimension=PlanQualityDimension.ADAPTABILITY,
                description="Consider making more goals adaptable",
                expected_improvement=0.1,
                implementation_difficulty=4
            ))
        
        return max(score, 0), issues, suggestions
    
    # Helper methods
    def _detect_cycles(self, plan: ExecutionPlan) -> bool:
        """Detect cycles in goal dependencies."""
        
        graph = {g.goal_id: g.depends_on for g in plan.goals}
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _check_deadline_feasibility(self, plan: ExecutionPlan) -> List[str]:
        """Check for goals with infeasible deadlines."""
        
        infeasible = []
        now = datetime.now()
        
        for goal in plan.goals:
            if goal.deadline:
                min_completion = now + goal.estimated_duration
                if min_completion > goal.deadline:
                    infeasible.append(goal.goal_id)
        
        return infeasible
    
    def _find_parallelizable_goals(self, plan: ExecutionPlan) -> List[str]:
        """Find goals that could potentially run in parallel."""
        
        parallelizable = []
        
        for goal in plan.goals:
            # Goals with no dependencies or few dependencies
            if len(goal.depends_on) <= 1:
                parallelizable.append(goal.goal_id)
        
        return parallelizable
    
    def _find_suboptimal_orderings(self, plan: ExecutionPlan) -> List[str]:
        """Find potentially suboptimal goal orderings."""
        
        suboptimal = []
        
        # Simple check: high priority goals that come after low priority
        order = plan.get_execution_order()
        for i, gid in enumerate(order):
            goal = plan.get_goal(gid)
            if goal and goal.priority > 80:
                # Check if any lower priority goals come before
                for j in range(i):
                    other = plan.get_goal(order[j])
                    if other and other.priority < goal.priority:
                        if goal.goal_id not in other.depends_on:
                            suboptimal.append(gid)
                            break
        
        return suboptimal
    
    async def select_best_plan(self, candidate_plans: List[ExecutionPlan],
                                min_quality: float = 0.6) -> Optional[ExecutionPlan]:
        """Select the best plan from candidates based on quality."""
        
        if not candidate_plans:
            return None
        
        # Assess all candidates
        assessed_plans = []
        for plan in candidate_plans:
            assessment = await self.assess_plan(plan)
            assessed_plans.append((plan, assessment))
        
        # Filter by minimum quality
        viable = [(p, a) for p, a in assessed_plans if a.overall_score >= min_quality]
        
        if not viable:
            # Return best of the rest
            best = max(assessed_plans, key=lambda x: x[1].overall_score)
            logger.warning(f"No plan meets quality threshold. Best: {best[1].overall_score:.2f}")
            return best[0]
        
        # Sort by overall score
        viable.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return viable[0][0]


# =============================================================================
# MAIN PLANNING LOOP
# =============================================================================

class AdvancedPlanningLoop:
    """
    Main planning loop integrating all planning components.
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig.from_yaml()

        # Initialize components
        self.goal_manager = HierarchicalGoalManager()
        self.replanning_engine = DynamicReplanningEngine(self.config)
        self.contingency_planner = ContingencyPlanner(self.config)
        self.quality_assessor = PlanQualityAssessor()
        
        # State
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_states: Dict[str, ExecutionState] = {}
        self.is_running = False
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    async def create_plan(self, goals: List[HARDGoal],
                          deadline: Optional[datetime] = None,
                          generate_contingencies: bool = True) -> ExecutionPlan:
        """
        Create a new execution plan from goals.
        
        Args:
            goals: List of goals to plan for
            deadline: Optional global deadline
            generate_contingencies: Whether to generate contingency plans
        
        Returns:
            ExecutionPlan: The generated plan
        """
        logger.info(f"Creating plan for {len(goals)} goals")
        
        # 1. Decompose goals hierarchically
        decomposed = []
        for goal in goals:
            subgoals = self.goal_manager.decompose_goal(goal)
            decomposed.extend(subgoals)
        
        logger.info(f"Decomposed into {len(decomposed)} subgoals")
        
        # 2. Build execution plan
        plan = ExecutionPlan(
            goals=decomposed,
            created_at=datetime.now()
        )
        
        # 3. Generate schedule
        schedule = self._generate_schedule(plan, deadline)
        plan.schedule = schedule
        
        # 4. Generate contingencies
        if generate_contingencies and self.config.contingency_planning_enabled:
            plan.contingencies = await self.contingency_planner.generate_contingency_plans(plan)
            logger.info(f"Generated {len(plan.contingencies)} contingency plans")
        
        # 5. Assess quality
        if self.config.quality_assessment_enabled:
            assessment = await self.quality_assessor.assess_plan(plan)
            plan.quality_score = assessment.overall_score
            logger.info(f"Plan quality score: {assessment.overall_score:.2f}")
        
        self.current_plan = plan
        
        # Emit event
        await self._emit_event('plan_created', {
            'plan_id': plan.plan_id,
            'goal_count': len(plan.goals),
            'quality_score': plan.quality_score
        })
        
        return plan
    
    def _generate_schedule(self, plan: ExecutionPlan,
                           deadline: Optional[datetime]) -> ExecutionSchedule:
        """Generate execution schedule for the plan."""
        
        schedule = ExecutionSchedule()
        execution_order = plan.get_execution_order()
        
        current_time = datetime.now()
        
        for goal_id in execution_order:
            goal = plan.get_goal(goal_id)
            if not goal:
                continue
            
            # Calculate start time based on dependencies
            start_time = current_time
            for dep_id in goal.depends_on:
                dep_task = schedule.get_task(dep_id)
                if dep_task:
                    start_time = max(start_time, dep_task.estimated_end)
            
            # Calculate end time
            estimated_end = start_time + goal.estimated_duration
            
            # Calculate latest end if deadline specified
            latest_end = None
            if deadline:
                latest_end = deadline
            elif goal.deadline:
                latest_end = goal.deadline
            
            schedule.add_task(
                goal_id=goal_id,
                start_time=start_time,
                estimated_end=estimated_end,
                latest_end=latest_end
            )
            
            current_time = estimated_end
        
        return schedule
    
    async def execute_plan(self, plan: ExecutionPlan,
                           goal_executor: Callable[[HARDGoal], Any]) -> ExecutionResult:
        """
        Execute a plan.
        
        Args:
            plan: The plan to execute
            goal_executor: Function to execute individual goals
        
        Returns:
            ExecutionResult: The execution result
        """
        execution_id = str(uuid.uuid4())
        
        # Initialize execution state
        state = ExecutionState(
            execution_id=execution_id,
            plan=plan,
            start_time=datetime.now(),
            status=ExecutionStatus.RUNNING,
            goal_states={g.goal_id: GoalStatus.PENDING for g in plan.goals}
        )
        self.execution_states[execution_id] = state
        
        logger.info(f"Starting execution {execution_id} for plan {plan.plan_id}")
        
        # Emit event
        await self._emit_event('plan_started', {
            'execution_id': execution_id,
            'plan_id': plan.plan_id
        })
        
        try:
            # Execute goals in order
            execution_order = plan.get_execution_order()
            
            for goal_id in execution_order:
                goal = plan.get_goal(goal_id)
                if not goal:
                    continue
                
                # Check if goal should be skipped
                if state.goal_states.get(goal_id) == GoalStatus.ABORTED:
                    continue
                
                # Update state
                state.current_goal_id = goal_id
                state.goal_states[goal_id] = GoalStatus.ACTIVE
                goal.status = GoalStatus.ACTIVE
                
                # Emit event
                await self._emit_event('goal_started', {
                    'execution_id': execution_id,
                    'goal_id': goal_id,
                    'goal_name': goal.name
                })
                
                try:
                    # Execute goal
                    result = await goal_executor(goal)
                    
                    # Update state on success
                    state.goal_states[goal_id] = GoalStatus.COMPLETED
                    goal.status = GoalStatus.COMPLETED
                    goal.progress = 1.0
                    
                    await self._emit_event('goal_completed', {
                        'execution_id': execution_id,
                        'goal_id': goal_id,
                        'result': result
                    })
                    
                except (OSError, RuntimeError, PermissionError) as e:
                    logger.error(f"Goal {goal_id} failed: {e}")
                    
                    # Update state on failure
                    state.goal_states[goal_id] = GoalStatus.FAILED
                    goal.status = GoalStatus.FAILED
                    
                    # Trigger replanning if enabled
                    if self.config.replanning_enabled:
                        await self.replanning_engine.trigger(
                            ReplanningTrigger.GOAL_FAILED,
                            {
                                'source': 'plan_execution',
                                'severity': 7 if goal.hardness == HardnessLevel.CRITICAL else 5,
                                'affected_goals': [goal_id],
                                'error': str(e)
                            }
                        )
                    
                    await self._emit_event('goal_failed', {
                        'execution_id': execution_id,
                        'goal_id': goal_id,
                        'error': str(e)
                    })
                    
                    # For critical goals, abort execution
                    if goal.hardness == HardnessLevel.CRITICAL:
                        state.status = ExecutionStatus.FAILED
                        break
                
                # Update progress
                state.update_progress()
            
            # Determine final status
            failed_count = sum(1 for s in state.goal_states.values() 
                             if s == GoalStatus.FAILED)
            completed_count = sum(1 for s in state.goal_states.values() 
                                if s == GoalStatus.COMPLETED)
            
            if failed_count == 0:
                state.status = ExecutionStatus.COMPLETED
            elif completed_count > 0:
                state.status = ExecutionStatus.COMPLETED_WITH_FAILURES
            else:
                state.status = ExecutionStatus.FAILED
            
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Execution failed: {e}")
            state.status = ExecutionStatus.FAILED
        
        # Create result
        result = ExecutionResult(
            execution_id=execution_id,
            status=state.status,
            start_time=state.start_time,
            end_time=datetime.now(),
            completed_goals=[gid for gid, s in state.goal_states.items() 
                           if s == GoalStatus.COMPLETED],
            failed_goals=[gid for gid, s in state.goal_states.items() 
                        if s == GoalStatus.FAILED]
        )
        
        logger.info(f"Execution {execution_id} completed with status: {state.status.name}")
        
        # Emit event
        await self._emit_event('plan_completed', {
            'execution_id': execution_id,
            'status': state.status.name,
            'completed_goals': len(result.completed_goals),
            'failed_goals': len(result.failed_goals)
        })
        
        return result
    
    async def add_goal_to_plan(self, goal: HARDGoal,
                                plan: Optional[ExecutionPlan] = None) -> ExecutionPlan:
        """Add a new goal to an existing plan."""
        
        target_plan = plan or self.current_plan
        if not target_plan:
            raise ValueError("No plan to add goal to")
        
        # Add goal
        target_plan.add_goal(goal)
        
        # Regenerate schedule
        target_plan.schedule = self._generate_schedule(target_plan, None)
        
        # Trigger replanning
        await self.replanning_engine.trigger(
            ReplanningTrigger.NEW_GOAL_ADDED,
            {
                'source': 'user_action',
                'severity': min(goal.priority // 10, 10),
                'affected_goals': [goal.goal_id],
                'new_goal': goal
            }
        )
        
        await self._emit_event('goal_added', {
            'plan_id': target_plan.plan_id,
            'goal_id': goal.goal_id,
            'goal_name': goal.name
        })
        
        return target_plan
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionState]:
        """Get the status of an execution."""
        return self.execution_states.get(execution_id)
    
    def on_event(self, event_type: str, callback: Callable):
        """Register an event callback."""
        self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event_type: str, data: Dict):
        """Emit an event to registered callbacks."""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Event callback error: {e}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the Advanced Planning Loop."""
    
    # Create planning loop
    config = PlanningConfig(
        replanning_enabled=True,
        contingency_planning_enabled=True,
        quality_assessment_enabled=True
    )
    
    planning_loop = AdvancedPlanningLoop(config)
    
    # Define some goals
    goals = [
        HARDGoal(
            goal_id="mission_1",
            name="Process User Request",
            description="Process and respond to user request",
            level=GoalLevel.MISSION,
            hardness=HardnessLevel.CRITICAL,
            priority=90,
            estimated_duration=timedelta(minutes=10)
        ),
        HARDGoal(
            goal_id="task_1",
            name="Check Gmail",
            description="Check for new emails",
            level=GoalLevel.TASK,
            hardness=HardnessLevel.IMPORTANT,
            priority=70,
            estimated_duration=timedelta(minutes=2)
        ),
        HARDGoal(
            goal_id="task_2",
            name="Generate Response",
            description="Generate response to user",
            level=GoalLevel.TASK,
            hardness=HardnessLevel.IMPORTANT,
            priority=80,
            estimated_duration=timedelta(minutes=5),
            depends_on=["task_1"]
        )
    ]
    
    # Create plan
    plan = await planning_loop.create_plan(goals)
    
    print(f"Created plan: {plan.plan_id}")
    print(f"Goals: {len(plan.goals)}")
    print(f"Quality score: {plan.quality_score}")
    print(f"Contingencies: {len(plan.contingencies)}")
    
    # Define goal executor
    async def execute_goal(goal: HARDGoal):
        print(f"Executing: {goal.name}")
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "success"}
    
    # Execute plan
    result = await planning_loop.execute_plan(plan, execute_goal)
    
    print(f"\nExecution completed: {result.status.name}")
    print(f"Completed goals: {len(result.completed_goals)}")
    print(f"Failed goals: {len(result.failed_goals)}")


if __name__ == "__main__":
    asyncio.run(example_usage())
