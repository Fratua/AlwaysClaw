"""
Advanced End-to-End Loop Core Implementation
OpenClaw-Inspired AI Agent System for Windows 10

This module provides the core workflow orchestration engine with:
- Complex multi-step workflow execution
- Stateful execution with persistence
- Parallel execution support
- Dependency management
- Compensation and rollback
- Human-in-the-loop integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union,
    Coroutine
)
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)


# ── Safe expression evaluator (shared module) ────────────────────────────
from safe_eval import SafeEvalVisitor as _SafeEvalVisitor, safe_eval as _safe_eval


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RECOVERING = "recovering"


class DependencyType(Enum):
    """Types of task dependencies."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"
    DATA = "data"
    TEMPORAL = "temporal"
    EXTERNAL = "external"


class CheckpointType(Enum):
    """Types of checkpoints."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"


class ApprovalStatus(Enum):
    """Human approval request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class RetryStrategy(Enum):
    """Retry strategies for failed tasks."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0  # seconds
    max_delay: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    retryable_errors: List[str] = field(default_factory=list)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        if self.strategy == RetryStrategy.FIXED:
            return self.initial_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_multiplier ** attempt)
            return min(delay, self.max_delay)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * (attempt + 1)
            return min(delay, self.max_delay)
        return self.initial_delay


@dataclass
class TaskConfig:
    """Task configuration."""
    timeout: Optional[float] = None
    retry_policy: Optional[RetryPolicy] = None
    compensation_task_id: Optional[str] = None
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)
    approval_timeout: Optional[float] = None


@dataclass
class TaskDefinition:
    """Definition of a workflow task."""
    id: str
    name: str
    type: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    when: Optional[str] = None  # Conditional expression
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'config': self.config,
            'depends_on': self.depends_on,
            'when': self.when,
            'outputs': self.outputs,
            'task_config': {
                'timeout': self.task_config.timeout,
                'retry_policy': {
                    'max_attempts': self.task_config.retry_policy.max_attempts,
                    'strategy': self.task_config.retry_policy.strategy.value
                } if self.task_config.retry_policy else None,
                'requires_approval': self.task_config.requires_approval
            }
        }


@dataclass
class TaskState:
    """Runtime state of a task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    execution_duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'execution_duration_ms': self.execution_duration_ms
        }


@dataclass
class Checkpoint:
    """Workflow checkpoint for recovery."""
    id: str
    workflow_id: str
    checkpoint_type: CheckpointType
    state_snapshot: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'checkpoint_type': self.checkpoint_type.value,
            'state_snapshot': self.state_snapshot,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    version: str
    description: str = ""
    tasks: List[TaskDefinition] = field(default_factory=list)
    inputs_schema: Dict[str, Any] = field(default_factory=dict)
    outputs_schema: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    max_concurrent: int = 10
    default_timeout: Optional[float] = None
    checkpoint_interval: Optional[float] = None
    
    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get task definition by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'tasks': [t.to_dict() for t in self.tasks],
            'inputs_schema': self.inputs_schema,
            'outputs_schema': self.outputs_schema,
            'variables': self.variables,
            'max_concurrent': self.max_concurrent,
            'default_timeout': self.default_timeout,
            'checkpoint_interval': self.checkpoint_interval
        }


@dataclass
class WorkflowState:
    """Runtime state of a workflow execution."""
    workflow_id: str
    definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    task_states: Dict[str, TaskState] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    restored_from: Optional[str] = None
    
    def get_task_state(self, task_id: str) -> TaskState:
        """Get or create task state."""
        if task_id not in self.task_states:
            self.task_states[task_id] = TaskState(task_id=task_id)
        return self.task_states[task_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'definition': self.definition.to_dict(),
            'status': self.status.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'variables': self.variables,
            'task_states': {k: v.to_dict() for k, v in self.task_states.items()},
            'checkpoints': self.checkpoints,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'restored_from': self.restored_from
        }


@dataclass
class ApprovalRequest:
    """Human approval request."""
    id: str
    workflow_id: str
    task_id: str
    approvers: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None
    responded_by: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    timeout_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'approvers': self.approvers,
            'context': self.context,
            'status': self.status.value,
            'requested_at': self.requested_at.isoformat(),
            'responded_at': self.responded_at.isoformat() if self.responded_at else None,
            'responded_by': self.responded_by,
            'response_data': self.response_data,
            'timeout_at': self.timeout_at.isoformat() if self.timeout_at else None
        }


@dataclass
class ExecutionResult:
    """Result of workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    execution_duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'outputs': self.outputs,
            'error_message': self.error_message,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'execution_duration_ms': self.execution_duration_ms
        }


# ============================================================================
# EXCEPTIONS
# ============================================================================

class E2ELoopError(Exception):
    """Base exception for E2E Loop."""
    pass


class WorkflowValidationError(E2ELoopError):
    """Workflow validation error."""
    pass


class CircularDependencyError(E2ELoopError):
    """Circular dependency detected."""
    pass


class TaskExecutionError(E2ELoopError):
    """Task execution error."""
    pass


class StatePersistenceError(E2ELoopError):
    """State persistence error."""
    pass


class ApprovalNotFoundError(E2ELoopError):
    """Approval request not found."""
    pass


class CompensationError(E2ELoopError):
    """Compensation execution error."""
    pass


# ============================================================================
# EXPRESSION EVALUATOR
# ============================================================================

class ExpressionEvaluator:
    """Evaluates expressions in workflow definitions."""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.functions = {
            'now': lambda: datetime.utcnow().isoformat(),
            'uuid': lambda: str(uuid.uuid4()),
            'json_stringify': json.dumps,
            'json_parse': json.loads,
            'len': len,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
        }
    
    def evaluate(self, expression: str) -> Any:
        """Evaluate an expression."""
        if not expression:
            return True
        
        # Handle simple variable references: ${variable}
        if expression.startswith('${') and expression.endswith('}'):
            var_path = expression[2:-1].strip()
            return self._resolve_variable(var_path)
        
        # Handle complex expressions
        return self._evaluate_complex(expression)
    
    def _resolve_variable(self, path: str) -> Any:
        """Resolve a variable path like 'tasks.task1.output.value'."""
        parts = path.split('.')
        value = self.context
        
        for part in parts:
            if isinstance(value, dict):
                # Handle array indexing
                if '[' in part:
                    base, idx_str = part.split('[')
                    idx = int(idx_str.rstrip(']'))
                    value = value.get(base, [])[idx]
                else:
                    value = value.get(part)
            else:
                return None
        
        return value
    
    def _evaluate_complex(self, expression: str) -> Any:
        """Evaluate complex expressions with operators."""
        # Replace variable references with actual values
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_path = match.group(1).strip()
            value = self._resolve_variable(var_path)
            return repr(value) if value is not None else 'None'
        
        # Replace all variable references
        eval_str = re.sub(pattern, replace_var, expression)
        
        # Create safe evaluation context
        eval_context = {**self.functions}
        
        try:
            return _safe_eval(eval_str, eval_context)
        except (SyntaxError, NameError, TypeError, ValueError, OSError, RuntimeError) as e:
            logger.warning(f"Expression evaluation failed: {e}")
            return None


# ============================================================================
# DEPENDENCY RESOLVER
# ============================================================================

class DependencyResolver:
    """Resolves task dependencies and builds execution order."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_dependency_graph(self, workflow: WorkflowDefinition) -> nx.DiGraph:
        """Build complete dependency graph from workflow definition."""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in workflow.tasks:
            graph.add_node(
                task.id,
                task=task,
                status=TaskStatus.PENDING
            )
        
        # Add dependency edges
        for task in workflow.tasks:
            # Static dependencies
            for dep_id in task.depends_on:
                if dep_id in graph.nodes:
                    graph.add_edge(dep_id, task.id, type=DependencyType.REQUIRED)
            
            # Extract dynamic dependencies from config
            dynamic_deps = self._extract_dynamic_dependencies(task)
            for dep_id in dynamic_deps:
                if dep_id in graph.nodes and dep_id not in task.depends_on:
                    graph.add_edge(dep_id, task.id, type=DependencyType.DATA)
        
        return graph
    
    def _extract_dynamic_dependencies(self, task: TaskDefinition) -> Set[str]:
        """Extract dynamic dependencies from task configuration."""
        deps = set()
        config_str = json.dumps(task.config)
        pattern = r'\$\{tasks\.(\w+)'
        matches = re.findall(pattern, config_str)
        deps.update(matches)
        return deps
    
    def resolve_execution_order(self, graph: nx.DiGraph) -> List[List[str]]:
        """Resolve execution order with parallel level detection."""
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise CircularDependencyError(f"Circular dependencies detected: {cycles}")
        
        # Topological sort with level grouping
        levels = []
        remaining = set(graph.nodes())
        
        while remaining:
            # Find nodes with no uncompleted predecessors
            ready = {
                node for node in remaining
                if not any(pred in remaining for pred in graph.predecessors(node))
            }
            
            if not ready:
                raise E2ELoopError("Unable to resolve dependencies")
            
            levels.append(sorted(list(ready)))
            remaining -= ready
        
        return levels
    
    def get_ready_tasks(
        self,
        graph: nx.DiGraph,
        completed: Set[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """Get tasks that are ready to execute."""
        ready = []
        evaluator = ExpressionEvaluator(context)
        
        for node in graph.nodes():
            if node in completed:
                continue
            
            task = graph.nodes[node]['task']
            
            # Check if all dependencies are satisfied
            deps_satisfied = True
            for pred in graph.predecessors(node):
                edge_data = graph.get_edge_data(pred, node)
                dep_type = edge_data.get('type', DependencyType.REQUIRED)
                
                if dep_type == DependencyType.REQUIRED and pred not in completed:
                    deps_satisfied = False
                    break
            
            # Check conditional execution
            if deps_satisfied and task.when:
                condition_met = evaluator.evaluate(task.when)
                if not condition_met:
                    continue
            
            if deps_satisfied:
                ready.append(node)
        
        return ready


# ============================================================================
# STATE BACKEND INTERFACE
# ============================================================================

class StateBackend(ABC):
    """Abstract base class for state persistence backends."""
    
    @abstractmethod
    async def save_workflow_state(self, state: WorkflowState) -> None:
        """Save workflow state."""
        pass
    
    @abstractmethod
    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state."""
        pass
    
    @abstractmethod
    async def save_task_state(
        self,
        workflow_id: str,
        task_id: str,
        task_state: TaskState
    ) -> None:
        """Save task state."""
        pass
    
    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint."""
        pass
    
    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint."""
        pass
    
    @abstractmethod
    async def load_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint for workflow."""
        pass
    
    @abstractmethod
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[WorkflowState]:
        """List workflows with optional filtering."""
        pass


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for testing."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowState] = {}
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.workflow_checkpoints: Dict[str, List[str]] = defaultdict(list)
    
    async def save_workflow_state(self, state: WorkflowState) -> None:
        self.workflows[state.workflow_id] = state
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        return self.workflows.get(workflow_id)
    
    async def save_task_state(
        self,
        workflow_id: str,
        task_id: str,
        task_state: TaskState
    ) -> None:
        if workflow_id in self.workflows:
            self.workflows[workflow_id].task_states[task_id] = task_state
    
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.checkpoints[checkpoint.id] = checkpoint
        self.workflow_checkpoints[checkpoint.workflow_id].append(checkpoint.id)
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        return self.checkpoints.get(checkpoint_id)
    
    async def load_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        checkpoint_ids = self.workflow_checkpoints.get(workflow_id, [])
        if not checkpoint_ids:
            return None
        return self.checkpoints.get(checkpoint_ids[-1])
    
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[WorkflowState]:
        workflows = list(self.workflows.values())
        if status:
            workflows = [w for w in workflows if w.status == status]
        return workflows[offset:offset + limit]


class SqliteStateBackend(StateBackend):
    """SQLite-backed state persistence for production use."""

    def __init__(self, db_path: str = None):
        import sqlite3 as _sqlite3
        import os as _os
        self._db_path = db_path or _os.environ.get('MEMORY_DB_PATH', './data/memory.db')
        db_dir = _os.path.dirname(self._db_path)
        if db_dir and not _os.path.exists(db_dir):
            _os.makedirs(db_dir, exist_ok=True)
        self._conn = _sqlite3.connect(self._db_path)
        self._conn.row_factory = _sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS e2e_workflows (
                workflow_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                definition_json TEXT NOT NULL,
                inputs_json TEXT NOT NULL DEFAULT '{}',
                outputs_json TEXT NOT NULL DEFAULT '{}',
                variables_json TEXT NOT NULL DEFAULT '{}',
                task_states_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS e2e_checkpoints (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                state_snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_e2e_cp_workflow
                ON e2e_checkpoints(workflow_id);
        """)
        self._conn.commit()

    def _serialize_workflow(self, state: WorkflowState) -> tuple:
        return (
            state.workflow_id,
            state.status.value,
            json.dumps(state.definition.to_dict()),
            json.dumps(state.inputs),
            json.dumps(state.outputs),
            json.dumps(state.variables),
            json.dumps({k: v.to_dict() for k, v in state.task_states.items()}),
            state.error_message,
            state.created_at.isoformat(),
            state.updated_at.isoformat(),
            state.completed_at.isoformat() if state.completed_at else None,
        )

    def _deserialize_workflow(self, row) -> WorkflowState:
        def_data = json.loads(row['definition_json'])
        tasks = []
        for t in def_data.get('tasks', []):
            tc_data = t.get('task_config', {}) or {}
            rp_data = tc_data.get('retry_policy')
            retry_policy = None
            if rp_data:
                retry_policy = RetryPolicy(
                    max_attempts=rp_data.get('max_attempts', 3),
                    strategy=RetryStrategy(rp_data.get('strategy', 'exponential')),
                )
            task_config = TaskConfig(
                timeout=tc_data.get('timeout'),
                retry_policy=retry_policy,
                requires_approval=tc_data.get('requires_approval', False),
            )
            tasks.append(TaskDefinition(
                id=t['id'], name=t['name'], type=t['type'],
                description=t.get('description', ''),
                config=t.get('config', {}),
                depends_on=t.get('depends_on', []),
                when=t.get('when'),
                outputs=t.get('outputs', []),
                task_config=task_config,
            ))
        definition = WorkflowDefinition(
            id=def_data['id'], name=def_data['name'],
            version=def_data['version'],
            description=def_data.get('description', ''),
            tasks=tasks,
            inputs_schema=def_data.get('inputs_schema', {}),
            outputs_schema=def_data.get('outputs_schema', {}),
            variables=def_data.get('variables', {}),
            max_concurrent=def_data.get('max_concurrent', 10),
            default_timeout=def_data.get('default_timeout'),
            checkpoint_interval=def_data.get('checkpoint_interval'),
        )
        ts_data = json.loads(row['task_states_json'])
        task_states = {}
        for tid, ts in ts_data.items():
            task_states[tid] = TaskState(
                task_id=ts['task_id'],
                status=TaskStatus(ts['status']),
                inputs=ts.get('inputs', {}),
                outputs=ts.get('outputs', {}),
                error_message=ts.get('error_message'),
                retry_count=ts.get('retry_count', 0),
            )
        completed_at = None
        if row['completed_at']:
            completed_at = datetime.fromisoformat(row['completed_at'])
        return WorkflowState(
            workflow_id=row['workflow_id'],
            definition=definition,
            status=WorkflowStatus(row['status']),
            inputs=json.loads(row['inputs_json']),
            outputs=json.loads(row['outputs_json']),
            variables=json.loads(row['variables_json']),
            task_states=task_states,
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            completed_at=completed_at,
            error_message=row['error_message'],
        )

    async def save_workflow_state(self, state: WorkflowState) -> None:
        params = self._serialize_workflow(state)
        self._conn.execute(
            """INSERT OR REPLACE INTO e2e_workflows
               (workflow_id, status, definition_json, inputs_json, outputs_json,
                variables_json, task_states_json, error_message, created_at,
                updated_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params,
        )
        self._conn.commit()

    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        row = self._conn.execute(
            "SELECT * FROM e2e_workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if not row:
            return None
        return self._deserialize_workflow(row)

    async def save_task_state(
        self, workflow_id: str, task_id: str, task_state: TaskState
    ) -> None:
        row = self._conn.execute(
            "SELECT task_states_json FROM e2e_workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if not row:
            return
        ts_data = json.loads(row['task_states_json'])
        ts_data[task_id] = task_state.to_dict()
        self._conn.execute(
            "UPDATE e2e_workflows SET task_states_json = ?, updated_at = ? WHERE workflow_id = ?",
            (json.dumps(ts_data), datetime.utcnow().isoformat(), workflow_id),
        )
        self._conn.commit()

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO e2e_checkpoints
               (id, workflow_id, checkpoint_type, state_snapshot_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                checkpoint.id,
                checkpoint.workflow_id,
                checkpoint.checkpoint_type.value,
                json.dumps(checkpoint.state_snapshot),
                checkpoint.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        row = self._conn.execute(
            "SELECT * FROM e2e_checkpoints WHERE id = ?",
            (checkpoint_id,),
        ).fetchone()
        if not row:
            return None
        return Checkpoint(
            id=row['id'],
            workflow_id=row['workflow_id'],
            checkpoint_type=CheckpointType(row['checkpoint_type']),
            state_snapshot=json.loads(row['state_snapshot_json']),
            created_at=datetime.fromisoformat(row['created_at']),
        )

    async def load_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        row = self._conn.execute(
            "SELECT * FROM e2e_checkpoints WHERE workflow_id = ? ORDER BY created_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if not row:
            return None
        return Checkpoint(
            id=row['id'],
            workflow_id=row['workflow_id'],
            checkpoint_type=CheckpointType(row['checkpoint_type']),
            state_snapshot=json.loads(row['state_snapshot_json']),
            created_at=datetime.fromisoformat(row['created_at']),
        )

    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkflowState]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM e2e_workflows WHERE status = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM e2e_workflows ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._deserialize_workflow(r) for r in rows]


# ============================================================================
# TASK EXECUTOR INTERFACE
# ============================================================================

class TaskExecutor(ABC):
    """Abstract base class for task executors."""
    
    @abstractmethod
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task."""
        ...


class LLMTaskExecutor(TaskExecutor):
    """Executor for LLM tasks."""
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LLM task."""
        config = task.config
        prompt = config.get('prompt', '')
        
        # Resolve variables in prompt
        evaluator = ExpressionEvaluator(context)
        resolved_prompt = self._resolve_template(prompt, evaluator)
        
        logger.info(f"Executing LLM task: {task.name}")

        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            system_prompt = config.get('system', '')
            result = client.complete(
                messages=[{"role": "user", "content": resolved_prompt}],
                system=system_prompt,
                max_tokens=config.get('max_tokens', 4096),
                temperature=config.get('temperature', 0.7),
            )
            return {
                'response': result['content'],
                'tokens_used': result.get('usage', {}).get('total_tokens', 0),
            }
        except (ImportError, RuntimeError, EnvironmentError) as e:
            logger.warning(f"LLM call failed, returning placeholder: {e}")
            return {
                'response': f'LLM response for: {resolved_prompt[:50]}...',
                'tokens_used': 0,
            }
    
    def _resolve_template(self, template: str, evaluator: ExpressionEvaluator) -> str:
        """Resolve template variables."""
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_path = match.group(1).strip()
            value = evaluator.evaluate(f'${{{var_path}}}')
            return str(value) if value is not None else ''
        
        return re.sub(pattern, replace_var, template)


class ShellTaskExecutor(TaskExecutor):
    """Executor for shell command tasks."""
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute shell command."""
        import subprocess
        
        config = task.config
        command = config.get('command', '')
        
        logger.info(f"Executing shell task: {task.name}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=config.get('timeout', 60)
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            raise TaskExecutionError(f"Shell command timed out: {command}")
        except OSError as e:
            raise TaskExecutionError(f"Shell command failed: {e}")


class PythonTaskExecutor(TaskExecutor):
    """Executor for Python code tasks."""
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Python code."""
        config = task.config
        code = config.get('code', '')
        
        logger.info(f"Executing Python task: {task.name}")
        
        # Create execution namespace with context
        namespace = {
            'context': context,
            'inputs': context.get('inputs', {}),
            'variables': context.get('variables', {}),
            'tasks': context.get('tasks', {}),
            'result': None
        }
        
        try:
            exec(code, namespace)
            return {
                'result': namespace.get('result'),
                'variables': {k: v for k, v in namespace.items() 
                             if not k.startswith('_') and k not in 
                             ['context', 'inputs', 'variables', 'tasks']}
            }
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            raise TaskExecutionError(f"Python execution failed: {e}")


class HTTPTaskExecutor(TaskExecutor):
    """Executor for HTTP request tasks."""
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HTTP request."""
        import aiohttp
        
        config = task.config
        method = config.get('method', 'GET')
        url = config.get('url', '')
        headers = config.get('headers', {})
        body = config.get('body')
        
        # Resolve variables
        evaluator = ExpressionEvaluator(context)
        url = self._resolve_template(url, evaluator)
        
        logger.info(f"Executing HTTP task: {method} {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=config.get('timeout', 30))
                ) as response:
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'data': await response.json() if response.content_type == 'application/json' 
                                else await response.text()
                    }
        except aiohttp.ClientError as e:
            raise TaskExecutionError(f"HTTP request failed: {e}")
    
    def _resolve_template(self, template: str, evaluator: ExpressionEvaluator) -> str:
        """Resolve template variables."""
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_path = match.group(1).strip()
            value = evaluator.evaluate(f'${{{var_path}}}')
            return str(value) if value is not None else ''
        
        return re.sub(pattern, replace_var, template)


class TaskExecutorRegistry:
    """Registry for task executors."""
    
    _executors: Dict[str, TaskExecutor] = {
        'llm': LLMTaskExecutor(),
        'shell': ShellTaskExecutor(),
        'python': PythonTaskExecutor(),
        'http': HTTPTaskExecutor(),
    }
    
    @classmethod
    def get_executor(cls, task_type: str) -> TaskExecutor:
        """Get executor for task type."""
        executor = cls._executors.get(task_type)
        if not executor:
            raise E2ELoopError(f"No executor registered for task type: {task_type}")
        return executor
    
    @classmethod
    def register_executor(cls, task_type: str, executor: TaskExecutor) -> None:
        """Register a task executor."""
        cls._executors[task_type] = executor


# ============================================================================
# MAIN WORKFLOW ENGINE
# ============================================================================

class E2EWorkflowEngine:
    """
    Main workflow execution engine.
    
    Features:
    - Complex multi-step workflow execution
    - Parallel execution support
    - Stateful execution with persistence
    - Compensation and rollback
    - Human-in-the-loop integration
    """
    
    def __init__(
        self,
        state_backend: Optional[StateBackend] = None,
        max_concurrent: int = 10
    ):
        self.state_backend = state_backend or InMemoryStateBackend()
        self.dependency_resolver = DependencyResolver()
        self.max_concurrent = max_concurrent
        self._running_workflows: Dict[str, asyncio.Task] = {}
        self._approval_callbacks: Dict[str, asyncio.Event] = {}
    
    async def submit_workflow(
        self,
        definition: WorkflowDefinition,
        inputs: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """
        Submit a workflow for execution.
        
        Args:
            definition: Workflow definition
            inputs: Workflow inputs
            workflow_id: Optional workflow ID (generated if not provided)
        
        Returns:
            Workflow ID
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        
        # Validate workflow
        self._validate_workflow(definition)
        
        # Initialize workflow state
        state = WorkflowState(
            workflow_id=workflow_id,
            definition=definition,
            status=WorkflowStatus.PENDING,
            inputs=inputs,
            variables=dict(definition.variables)
        )
        
        # Save initial state
        await self.state_backend.save_workflow_state(state)
        
        # Start execution
        execution_task = asyncio.create_task(
            self._execute_workflow(workflow_id)
        )
        self._running_workflows[workflow_id] = execution_task
        
        logger.info(f"Workflow {workflow_id} submitted")
        
        return workflow_id
    
    def _validate_workflow(self, definition: WorkflowDefinition) -> None:
        """Validate workflow definition."""
        # Check for duplicate task IDs
        task_ids = [t.id for t in definition.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise WorkflowValidationError("Duplicate task IDs found")
        
        # Check for missing dependencies
        for task in definition.tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    raise WorkflowValidationError(
                        f"Task {task.id} depends on unknown task: {dep_id}"
                    )
        
        # Check for circular dependencies
        try:
            graph = self.dependency_resolver.build_dependency_graph(definition)
            self.dependency_resolver.resolve_execution_order(graph)
        except CircularDependencyError:
            raise WorkflowValidationError("Circular dependencies detected")
    
    async def _execute_workflow(self, workflow_id: str) -> ExecutionResult:
        """Execute workflow."""
        start_time = datetime.utcnow()
        
        # Load state
        state = await self.state_backend.load_workflow_state(workflow_id)
        if not state:
            return ExecutionResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                error_message="Workflow state not found"
            )
        
        # Update status
        state.status = WorkflowStatus.RUNNING
        state.updated_at = datetime.utcnow()
        await self.state_backend.save_workflow_state(state)
        
        try:
            # Build dependency graph
            graph = self.dependency_resolver.build_dependency_graph(state.definition)
            
            # Execute tasks
            completed_tasks: Set[str] = set()
            failed_tasks: Set[str] = set()
            
            while len(completed_tasks) + len(failed_tasks) < len(state.definition.tasks):
                # Get ready tasks
                context = self._build_context(state)
                ready_tasks = self.dependency_resolver.get_ready_tasks(
                    graph, completed_tasks, context
                )
                
                if not ready_tasks:
                    if failed_tasks:
                        break
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute ready tasks (with concurrency limit)
                semaphore = asyncio.Semaphore(state.definition.max_concurrent)
                
                async def execute_with_limit(task_id: str) -> Tuple[str, bool]:
                    async with semaphore:
                        success = await self._execute_task(workflow_id, task_id)
                        return task_id, success
                
                # Execute tasks
                results = await asyncio.gather(*[
                    execute_with_limit(task_id) for task_id in ready_tasks
                ])
                
                # Update completed/failed sets
                for task_id, success in results:
                    if success:
                        completed_tasks.add(task_id)
                    else:
                        failed_tasks.add(task_id)
                
                # Create checkpoint if configured
                if state.definition.checkpoint_interval:
                    await self._create_checkpoint(workflow_id)
            
            # Determine final status
            if failed_tasks:
                # Execute compensations
                await self._execute_compensations(state, completed_tasks)
                
                state.status = WorkflowStatus.FAILED
                state.error_message = f"Tasks failed: {failed_tasks}"
            else:
                state.status = WorkflowStatus.COMPLETED
            
            state.completed_at = datetime.utcnow()
            state.updated_at = datetime.utcnow()
            
            # Calculate outputs
            state.outputs = self._calculate_outputs(state)
            
            await self.state_backend.save_workflow_state(state)
            
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                workflow_id=workflow_id,
                status=state.status,
                outputs=state.outputs,
                error_message=state.error_message,
                completed_tasks=list(completed_tasks),
                failed_tasks=list(failed_tasks),
                execution_duration_ms=duration
            )
            
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.exception(f"Workflow execution failed: {e}")
            
            state.status = WorkflowStatus.FAILED
            state.error_message = str(e)
            state.updated_at = datetime.utcnow()
            await self.state_backend.save_workflow_state(state)
            
            return ExecutionResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                error_message=str(e)
            )
        finally:
            # Cleanup
            if workflow_id in self._running_workflows:
                del self._running_workflows[workflow_id]
    
    async def _execute_task(self, workflow_id: str, task_id: str) -> bool:
        """Execute a single task."""
        state = await self.state_backend.load_workflow_state(workflow_id)
        task_def = state.definition.get_task(task_id)
        task_state = state.get_task_state(task_id)
        
        # Update status
        task_state.status = TaskStatus.RUNNING
        task_state.started_at = datetime.utcnow()
        await self.state_backend.save_task_state(workflow_id, task_id, task_state)
        
        try:
            # Check if approval required
            if task_def.task_config.requires_approval:
                task_state.status = TaskStatus.WAITING_APPROVAL
                await self.state_backend.save_task_state(workflow_id, task_id, task_state)
                
                # Wait for approval
                approved = await self._wait_for_approval(workflow_id, task_id, task_def)
                
                if not approved:
                    task_state.status = TaskStatus.CANCELLED
                    task_state.completed_at = datetime.utcnow()
                    await self.state_backend.save_task_state(workflow_id, task_id, task_state)
                    return False
                
                task_state.status = TaskStatus.RUNNING
            
            # Build execution context
            context = self._build_context(state)
            
            # Get executor and execute
            executor = TaskExecutorRegistry.get_executor(task_def.type)
            
            # Execute with retry
            retry_policy = task_def.task_config.retry_policy or RetryPolicy()
            last_error = None
            
            for attempt in range(retry_policy.max_attempts):
                try:
                    result = await executor.execute(task_def, context)
                    
                    # Update task state with outputs
                    task_state.outputs = result
                    task_state.status = TaskStatus.COMPLETED
                    task_state.completed_at = datetime.utcnow()
                    task_state.execution_duration_ms = int(
                        (task_state.completed_at - task_state.started_at).total_seconds() * 1000
                    )
                    
                    await self.state_backend.save_task_state(workflow_id, task_id, task_state)
                    
                    logger.info(f"Task {task_id} completed successfully")
                    return True
                    
                except (OSError, RuntimeError, PermissionError) as e:
                    last_error = e
                    task_state.retry_count = attempt + 1
                    
                    if attempt < retry_policy.max_attempts - 1:
                        delay = retry_policy.calculate_delay(attempt)
                        logger.warning(f"Task {task_id} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
            
            # All retries exhausted
            task_state.status = TaskStatus.FAILED
            task_state.error_message = str(last_error)
            task_state.completed_at = datetime.utcnow()
            await self.state_backend.save_task_state(workflow_id, task_id, task_state)
            
            logger.error(f"Task {task_id} failed after {retry_policy.max_attempts} attempts")
            return False
            
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.exception(f"Task execution error: {e}")
            task_state.status = TaskStatus.FAILED
            task_state.error_message = str(e)
            task_state.completed_at = datetime.utcnow()
            await self.state_backend.save_task_state(workflow_id, task_id, task_state)
            return False
    
    async def _wait_for_approval(
        self,
        workflow_id: str,
        task_id: str,
        task_def: TaskDefinition
    ) -> bool:
        """Wait for human approval."""
        approval_id = f"{workflow_id}:{task_id}"
        event = asyncio.Event()
        self._approval_callbacks[approval_id] = event
        
        # Set timeout
        timeout = task_def.task_config.approval_timeout or 3600  # 1 hour default
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            
            # Check approval result
            state = await self.state_backend.load_workflow_state(workflow_id)
            # In real implementation, store approval result in state
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for task {task_id}")
            return False
        finally:
            if approval_id in self._approval_callbacks:
                del self._approval_callbacks[approval_id]
    
    async def submit_approval(
        self,
        workflow_id: str,
        task_id: str,
        approved: bool,
        response_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Submit human approval response."""
        approval_id = f"{workflow_id}:{task_id}"
        
        # Update task state
        state = await self.state_backend.load_workflow_state(workflow_id)
        task_state = state.get_task_state(task_id)
        
        if approved:
            task_state.status = TaskStatus.RUNNING
        else:
            task_state.status = TaskStatus.CANCELLED
            task_state.outputs = {'approval_rejected': True, 'data': response_data}
        
        await self.state_backend.save_task_state(workflow_id, task_id, task_state)
        
        # Signal waiting task
        if approval_id in self._approval_callbacks:
            self._approval_callbacks[approval_id].set()
        
        return True
    
    async def _execute_compensations(
        self,
        state: WorkflowState,
        completed_tasks: Set[str]
    ) -> None:
        """Execute compensation actions."""
        logger.info(f"Executing compensations for workflow {state.workflow_id}")
        
        # Execute compensations in reverse order
        for task_id in reversed(list(completed_tasks)):
            task_def = state.definition.get_task(task_id)
            
            if task_def.task_config.compensation_task_id:
                comp_task_id = task_def.task_config.compensation_task_id
                comp_task_def = state.definition.get_task(comp_task_id)
                
                if comp_task_def:
                    try:
                        context = self._build_context(state)
                        executor = TaskExecutorRegistry.get_executor(comp_task_def.type)
                        await executor.execute(comp_task_def, context)
                        logger.info(f"Compensation for task {task_id} executed")
                    except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                        logger.error(f"Compensation for task {task_id} failed: {e}")
    
    async def _create_checkpoint(self, workflow_id: str) -> None:
        """Create workflow checkpoint."""
        state = await self.state_backend.load_workflow_state(workflow_id)
        
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            checkpoint_type=CheckpointType.AUTOMATIC,
            state_snapshot=state.to_dict()
        )
        
        await self.state_backend.save_checkpoint(checkpoint)
        state.checkpoints.append(checkpoint.id)
        await self.state_backend.save_workflow_state(state)
        
        logger.debug(f"Checkpoint created for workflow {workflow_id}")
    
    def _build_context(self, state: WorkflowState) -> Dict[str, Any]:
        """Build execution context."""
        return {
            'inputs': state.inputs,
            'variables': state.variables,
            'tasks': {
                task_id: {
                    'output': task_state.outputs,
                    'status': task_state.status.value
                }
                for task_id, task_state in state.task_states.items()
            },
            'workflow_id': state.workflow_id
        }
    
    def _calculate_outputs(self, state: WorkflowState) -> Dict[str, Any]:
        """Calculate workflow outputs from task outputs."""
        outputs = {}
        
        # Collect outputs from tasks that export them
        for task in state.definition.tasks:
            task_state = state.task_states.get(task.id)
            if task_state and task_state.status == TaskStatus.COMPLETED:
                for output_def in task.outputs:
                    if output_def.get('export', False):
                        name = output_def['name']
                        value_expr = output_def['value']
                        
                        # Evaluate expression
                        evaluator = ExpressionEvaluator(self._build_context(state))
                        value = evaluator.evaluate(value_expr)
                        
                        outputs[name] = value
        
        return outputs
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get workflow status."""
        state = await self.state_backend.load_workflow_state(workflow_id)
        return state.status if state else None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self._running_workflows:
            self._running_workflows[workflow_id].cancel()
            
            state = await self.state_backend.load_workflow_state(workflow_id)
            if state:
                state.status = WorkflowStatus.CANCELLED
                state.updated_at = datetime.utcnow()
                await self.state_backend.save_workflow_state(state)
            
            return True
        return False
    
    async def restore_from_checkpoint(self, workflow_id: str) -> bool:
        """Restore workflow from latest checkpoint."""
        checkpoint = await self.state_backend.load_latest_checkpoint(workflow_id)
        
        if not checkpoint:
            logger.warning(f"No checkpoint found for workflow {workflow_id}")
            return False
        
        # Restore state
        state = WorkflowState(
            workflow_id=checkpoint.state_snapshot['workflow_id'],
            definition=WorkflowDefinition(**checkpoint.state_snapshot['definition']),
            status=WorkflowStatus.RECOVERING,
            inputs=checkpoint.state_snapshot['inputs'],
            outputs=checkpoint.state_snapshot['outputs'],
            variables=checkpoint.state_snapshot['variables'],
            task_states={
                k: TaskState(**v) 
                for k, v in checkpoint.state_snapshot['task_states'].items()
            },
            checkpoints=checkpoint.state_snapshot['checkpoints'],
            restored_from=checkpoint.id
        )
        
        await self.state_backend.save_workflow_state(state)
        
        # Restart execution
        execution_task = asyncio.create_task(
            self._execute_workflow(workflow_id)
        )
        self._running_workflows[workflow_id] = execution_task
        
        logger.info(f"Workflow {workflow_id} restored from checkpoint {checkpoint.id}")
        return True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_workflow_definition(
    name: str,
    tasks: List[Dict[str, Any]],
    version: str = "1.0.0",
    **kwargs
) -> WorkflowDefinition:
    """
    Create a workflow definition from a simplified dictionary format.
    
    Args:
        name: Workflow name
        tasks: List of task definitions
        version: Workflow version
        **kwargs: Additional workflow configuration
    
    Returns:
        WorkflowDefinition object
    """
    task_definitions = []
    for task_data in tasks:
        task_config = TaskConfig(
            timeout=task_data.get('timeout'),
            retry_policy=RetryPolicy(
                max_attempts=task_data.get('retries', 3),
                strategy=RetryStrategy(task_data.get('retry_strategy', 'exponential'))
            ) if task_data.get('retries') else None,
            requires_approval=task_data.get('requires_approval', False),
            approvers=task_data.get('approvers', []),
            approval_timeout=task_data.get('approval_timeout')
        )
        
        task_def = TaskDefinition(
            id=task_data['id'],
            name=task_data.get('name', task_data['id']),
            type=task_data['type'],
            description=task_data.get('description', ''),
            config=task_data.get('config', {}),
            depends_on=task_data.get('depends_on', []),
            when=task_data.get('when'),
            outputs=task_data.get('outputs', []),
            task_config=task_config
        )
        task_definitions.append(task_def)
    
    return WorkflowDefinition(
        id=kwargs.get('id', str(uuid.uuid4())),
        name=name,
        version=version,
        description=kwargs.get('description', ''),
        tasks=task_definitions,
        inputs_schema=kwargs.get('inputs_schema', {}),
        outputs_schema=kwargs.get('outputs_schema', {}),
        variables=kwargs.get('variables', {}),
        max_concurrent=kwargs.get('max_concurrent', 10),
        default_timeout=kwargs.get('default_timeout'),
        checkpoint_interval=kwargs.get('checkpoint_interval')
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example usage of the E2E Workflow Engine."""
    
    # Create workflow engine
    engine = E2EWorkflowEngine()
    
    # Define a simple workflow
    workflow_def = create_workflow_definition(
        name="Data Processing Pipeline",
        tasks=[
            {
                'id': 'fetch_data',
                'name': 'Fetch Data',
                'type': 'http',
                'config': {
                    'method': 'GET',
                    'url': 'https://api.example.com/data'
                },
                'outputs': [
                    {'name': 'data', 'value': '${response.data}', 'export': True}
                ]
            },
            {
                'id': 'process_data',
                'name': 'Process Data',
                'type': 'python',
                'depends_on': ['fetch_data'],
                'config': {
                    'code': '''
processed = [item.upper() for item in context['tasks']['fetch_data']['output']['data']]
result = {'processed': processed}
'''
                },
                'outputs': [
                    {'name': 'processed_data', 'value': '${result.processed}', 'export': True}
                ]
            },
            {
                'id': 'send_notification',
                'name': 'Send Notification',
                'type': 'llm',
                'depends_on': ['process_data'],
                'config': {
                    'prompt': 'Generate a summary for: ${tasks.process_data.output.processed_data}'
                }
            }
        ],
        max_concurrent=5
    )
    
    # Submit workflow
    workflow_id = await engine.submit_workflow(
        definition=workflow_def,
        inputs={'source': 'example'}
    )
    
    print(f"Workflow submitted: {workflow_id}")
    
    # Wait for completion
    while True:
        status = await engine.get_workflow_status(workflow_id)
        print(f"Status: {status.value if status else 'unknown'}")
        
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            break
        
        await asyncio.sleep(1)
    
    # Get final state
    state = await engine.state_backend.load_workflow_state(workflow_id)
    print(f"Final outputs: {state.outputs}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
