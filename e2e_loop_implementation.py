"""
End-to-End Loop Implementation
Windows 10 OpenClaw-Inspired AI Agent Framework

This module provides the core implementation for the E2E Loop workflow automation system.
"""

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

import aiosqlite


# =============================================================================
# ENUMS
# =============================================================================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    COMPENSATING = auto()


class StepStatus(Enum):
    """Step execution status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    RETRYING = auto()
    COMPENSATED = auto()


class StepType(Enum):
    """Types of workflow steps"""
    ACTION = auto()
    DECISION = auto()
    PARALLEL = auto()
    WAIT = auto()
    SUBWORKFLOW = auto()
    HUMAN = auto()
    COMPENSATION = auto()


class TriggerType(Enum):
    """Types of workflow triggers"""
    SCHEDULE = auto()
    GMAIL = auto()
    WEBHOOK = auto()
    API = auto()
    FILE = auto()
    EVENT = auto()
    MANUAL = auto()
    VOICE = auto()
    SMS = auto()
    SYSTEM = auto()


class VerificationLevel(Enum):
    """Levels of completion verification"""
    BASIC = auto()
    STANDARD = auto()
    THOROUGH = auto()
    EXHAUSTIVE = auto()


class ErrorSeverity(Enum):
    """Error severity levels"""
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    step_type: StepType = StepType.ACTION
    activity: Optional[str] = None
    activity_config: Dict = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry_policy: Dict = field(default_factory=dict)
    on_error: Optional[str] = None
    compensation: Optional[str] = None
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    initial_step: str = ""
    timeout_seconds: int = 3600
    max_retries: int = 3
    triggers: List[Dict] = field(default_factory=list)
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowInstance:
    """Running instance of a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    definition_id: str = ""
    definition_version: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    compensation_stack: List[str] = field(default_factory=list)


@dataclass
class Trigger:
    """Workflow trigger definition"""
    id: str = ""
    type: TriggerType = TriggerType.MANUAL
    name: str = ""
    workflow_id: str = ""
    config: Dict = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    conditions: List[Dict] = field(default_factory=list)
    priority: int = 5
    rate_limit: Optional[Dict] = None


@dataclass
class TriggerEvent:
    """Event that can trigger a workflow"""
    id: str = ""
    type: TriggerType = TriggerType.MANUAL
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for step execution"""
    workflow_instance_id: str = ""
    step_id: str = ""
    input_data: Dict = field(default_factory=dict)
    workflow_context: Dict = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0


@dataclass
class VerificationResult:
    """Result of completion verification"""
    verified: bool = False
    confidence: float = 0.0
    checks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class OptimizationSuggestion:
    """Suggestion for workflow optimization"""
    type: str = ""
    description: str = ""
    impact: str = ""
    current_value: Any = None
    suggested_value: Any = None
    expected_improvement: str = ""


@dataclass
class DecomposedTask:
    """A task broken down into executable steps"""
    id: str = ""
    description: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 0
    required_capabilities: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class WorkflowStateManager:
    """Manages persistence of workflow state using SQLite"""
    
    def __init__(self, db_path: str = "workflows.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    version TEXT,
                    definition_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    id TEXT PRIMARY KEY,
                    definition_id TEXT NOT NULL,
                    definition_version TEXT,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    completed_steps TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    step_results TEXT,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    compensation_stack TEXT
                );
                
                CREATE TABLE IF NOT EXISTS step_states (
                    instance_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    PRIMARY KEY (instance_id, step_id)
                );
                
                CREATE TABLE IF NOT EXISTS triggers (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    last_triggered TIMESTAMP,
                    trigger_count INTEGER DEFAULT 0,
                    conditions TEXT,
                    priority INTEGER DEFAULT 5,
                    rate_limit TEXT
                );
                
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    step_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_instances_status 
                    ON workflow_instances(status);
                CREATE INDEX IF NOT EXISTS idx_instances_definition 
                    ON workflow_instances(definition_id);
            """)
    
    async def save_workflow_definition(self, definition: WorkflowDefinition):
        """Save workflow definition"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflow_definitions 
                (id, name, description, version, definition_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                definition.id,
                definition.name,
                definition.description,
                definition.version,
                json.dumps(self._definition_to_dict(definition)),
                datetime.now()
            ))
            await db.commit()
    
    async def save_instance(self, instance: WorkflowInstance):
        """Save workflow instance state"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflow_instances
                (id, definition_id, definition_version, status, current_step,
                 completed_steps, input_data, output_data, step_results,
                 context, started_at, completed_at, error_count, last_error,
                 compensation_stack)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instance.id,
                instance.definition_id,
                instance.definition_version,
                instance.status.name,
                instance.current_step,
                json.dumps(instance.completed_steps),
                json.dumps(instance.input_data),
                json.dumps(instance.output_data),
                json.dumps(instance.step_results),
                json.dumps(instance.context),
                instance.started_at,
                instance.completed_at,
                instance.error_count,
                instance.last_error,
                json.dumps(instance.compensation_stack)
            ))
            await db.commit()
    
    async def save_step_state(self, instance_id: str, step: WorkflowStep):
        """Save step execution state"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO step_states
                (instance_id, step_id, status, started_at, completed_at,
                 result, error, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instance_id,
                step.id,
                step.status.name,
                step.started_at,
                step.completed_at,
                json.dumps(step.result) if step.result else None,
                step.error,
                step.retry_policy.get("attempts", 0)
            ))
            await db.commit()
    
    async def get_running_instances(self) -> List[WorkflowInstance]:
        """Get all running workflow instances"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_instances WHERE status = ?",
                (WorkflowStatus.RUNNING.name,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_instance(row) for row in rows]
    
    async def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_instances WHERE id = ?",
                (instance_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_instance(row) if row else None
    
    async def get_workflow_definition(self, definition_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_definitions WHERE id = ?",
                (definition_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._dict_to_definition(json.loads(row["definition_json"]))
                return None
    
    def _definition_to_dict(self, definition: WorkflowDefinition) -> Dict:
        """Convert definition to dictionary"""
        return {
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "version": definition.version,
            "steps": {k: self._step_to_dict(v) for k, v in definition.steps.items()},
            "initial_step": definition.initial_step,
            "timeout_seconds": definition.timeout_seconds,
            "max_retries": definition.max_retries,
            "triggers": definition.triggers,
            "input_schema": definition.input_schema,
            "output_schema": definition.output_schema,
            "created_at": definition.created_at.isoformat() if definition.created_at else None,
            "updated_at": definition.updated_at.isoformat() if definition.updated_at else None,
            "tags": definition.tags
        }
    
    def _step_to_dict(self, step: WorkflowStep) -> Dict:
        """Convert step to dictionary"""
        return {
            "id": step.id,
            "name": step.name,
            "description": step.description,
            "step_type": step.step_type.name,
            "activity": step.activity,
            "activity_config": step.activity_config,
            "next_steps": step.next_steps,
            "condition": step.condition,
            "retry_policy": step.retry_policy,
            "on_error": step.on_error,
            "compensation": step.compensation,
            "timeout_seconds": step.timeout_seconds,
            "dependencies": step.dependencies,
            "metadata": step.metadata
        }
    
    def _dict_to_definition(self, data: Dict) -> WorkflowDefinition:
        """Convert dictionary to definition"""
        definition = WorkflowDefinition(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            steps={k: self._dict_to_step(v) for k, v in data.get("steps", {}).items()},
            initial_step=data.get("initial_step", ""),
            timeout_seconds=data.get("timeout_seconds", 3600),
            max_retries=data.get("max_retries", 3),
            triggers=data.get("triggers", []),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            tags=data.get("tags", [])
        )
        return definition
    
    def _dict_to_step(self, data: Dict) -> WorkflowStep:
        """Convert dictionary to step"""
        return WorkflowStep(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            step_type=StepType[data.get("step_type", "ACTION")],
            activity=data.get("activity"),
            activity_config=data.get("activity_config", {}),
            next_steps=data.get("next_steps", []),
            condition=data.get("condition"),
            retry_policy=data.get("retry_policy", {}),
            on_error=data.get("on_error"),
            compensation=data.get("compensation"),
            timeout_seconds=data.get("timeout_seconds", 300),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {})
        )
    
    def _row_to_instance(self, row) -> WorkflowInstance:
        """Convert database row to instance"""
        return WorkflowInstance(
            id=row["id"],
            definition_id=row["definition_id"],
            definition_version=row["definition_version"],
            status=WorkflowStatus[row["status"]],
            current_step=row["current_step"] or "",
            completed_steps=json.loads(row["completed_steps"] or "[]"),
            input_data=json.loads(row["input_data"] or "{}"),
            output_data=json.loads(row["output_data"] or "{}"),
            step_results=json.loads(row["step_results"] or "{}"),
            context=json.loads(row["context"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            error_count=row["error_count"] or 0,
            last_error=row["last_error"],
            compensation_stack=json.loads(row["compensation_stack"] or "[]")
        )


# =============================================================================
# STEP EXECUTOR
# =============================================================================

class StepExecutor:
    """Executes individual workflow steps"""
    
    def __init__(self, activity_registry: Dict[str, Callable], 
                 state_manager: WorkflowStateManager):
        self.activities = activity_registry
        self.state = state_manager
        self.running_executions: Dict[str, asyncio.Task] = {}
    
    async def execute_step(self, step: WorkflowStep, 
                          context: ExecutionContext) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        step_key = f"{context.workflow_instance_id}:{step.id}"
        
        if step_key in self.running_executions:
            return {"status": "already_running"}
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        context.execution_count += 1
        
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        activity = self.activities.get(step.activity)
        if not activity:
            error = f"Activity '{step.activity}' not found"
            return await self._handle_step_failure(step, context, error)
        
        activity_input = self._build_activity_input(step, context)
        
        try:
            result = await self._execute_with_retry(
                activity, 
                activity_input, 
                step.retry_policy,
                step.timeout_seconds
            )
            return await self._handle_step_success(step, context, result)
            
        except asyncio.TimeoutError:
            error = f"Step timed out after {step.timeout_seconds}s"
            return await self._handle_step_failure(step, context, error)
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            return await self._handle_step_failure(step, context, error)
    
    async def _execute_with_retry(self, activity: Callable, 
                                   input_data: Dict,
                                   retry_policy: Dict,
                                   timeout: int) -> Any:
        """Execute activity with retry logic"""
        
        max_attempts = retry_policy.get("max_attempts", 3)
        backoff_strategy = retry_policy.get("backoff", "exponential")
        initial_delay = retry_policy.get("initial_delay", 1)
        max_delay = retry_policy.get("max_delay", 60)
        
        last_error = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    activity(**input_data),
                    timeout=timeout
                )
                
            except Exception as e:
                last_error = e
                
                if attempt >= max_attempts:
                    raise
                
                if backoff_strategy == "exponential":
                    delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                elif backoff_strategy == "linear":
                    delay = min(initial_delay * attempt, max_delay)
                else:
                    delay = initial_delay
                
                await asyncio.sleep(delay)
        
        raise last_error
    
    async def _handle_step_success(self, step: WorkflowStep,
                                    context: ExecutionContext,
                                    result: Any) -> Dict[str, Any]:
        """Handle successful step execution"""
        
        step.status = StepStatus.COMPLETED
        step.completed_at = datetime.now()
        step.result = result
        
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        duration = (step.completed_at - step.started_at).total_seconds()
        
        return {
            "status": "success",
            "step_id": step.id,
            "result": result,
            "duration": duration
        }
    
    async def _handle_step_failure(self, step: WorkflowStep,
                                    context: ExecutionContext,
                                    error: str) -> Dict[str, Any]:
        """Handle step execution failure"""
        
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now()
        step.error = error
        
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        duration = (step.completed_at - step.started_at).total_seconds()
        
        return {
            "status": "failed",
            "step_id": step.id,
            "error": error,
            "duration": duration
        }
    
    def _build_activity_input(self, step: WorkflowStep,
                               context: ExecutionContext) -> Dict[str, Any]:
        """Build input data for activity execution"""
        
        input_data = {
            **step.activity_config,
            "workflow_instance_id": context.workflow_instance_id,
            "step_id": step.id,
            "previous_results": context.step_results,
            "workflow_context": context.workflow_context,
        }
        
        return input_data


# =============================================================================
# WORKFLOW PATTERNS
# =============================================================================

class WorkflowPatterns:
    """Common workflow patterns for E2E automation"""
    
    @staticmethod
    def sequential(steps: List[WorkflowStep]) -> WorkflowDefinition:
        """Execute steps in sequence"""
        wf = WorkflowDefinition(name="sequential")
        prev_step = None
        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            step.id = step_id
            wf.steps[step_id] = step
            if prev_step:
                prev_step.next_steps.append(step_id)
            else:
                wf.initial_step = step_id
            prev_step = step
        return wf
    
    @staticmethod
    def parallel(branches: List[List[WorkflowStep]], 
                   merge_step: WorkflowStep) -> WorkflowDefinition:
        """Execute branches in parallel, then merge"""
        wf = WorkflowDefinition(name="parallel")
        parallel_step = WorkflowStep(
            id="parallel_fork",
            step_type=StepType.PARALLEL,
            next_steps=["merge"]
        )
        wf.steps["parallel_fork"] = parallel_step
        wf.initial_step = "parallel_fork"
        
        for i, branch in enumerate(branches):
            branch_id = f"branch_{i}"
            parallel_step.metadata["branches"] = parallel_step.metadata.get("branches", [])
            parallel_step.metadata["branches"].append(branch_id)
        
        merge_step.id = "merge"
        wf.steps["merge"] = merge_step
        
        return wf
    
    @staticmethod
    def conditional(condition: str,
                   true_branch: List[WorkflowStep],
                   false_branch: List[WorkflowStep]) -> WorkflowDefinition:
        """Conditional execution based on expression"""
        wf = WorkflowDefinition(name="conditional")
        
        decision = WorkflowStep(
            id="decision",
            step_type=StepType.DECISION,
            condition=condition,
            next_steps=["true_path", "false_path"]
        )
        wf.steps["decision"] = decision
        wf.initial_step = "decision"
        
        for step in true_branch:
            step.id = f"true_{step.id}"
            wf.steps[step.id] = step
        
        for step in false_branch:
            step.id = f"false_{step.id}"
            wf.steps[step.id] = step
        
        return wf


# =============================================================================
# E2E LOOP API
# =============================================================================

class E2ELoop:
    """Main End-to-End Loop API"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.state_manager = WorkflowStateManager(
            self.config.get("db_path", "workflows.db")
        )
        self.step_executor = None
        self.running = False
        self.active_instances: Dict[str, asyncio.Task] = {}
    
    def register_activities(self, activities: Dict[str, Callable]):
        """Register activity handlers"""
        self.step_executor = StepExecutor(activities, self.state_manager)
    
    async def start(self):
        """Start the E2E Loop"""
        self.running = True
        print("E2E Loop started")
    
    async def stop(self):
        """Stop the E2E Loop"""
        self.running = False
        for task in self.active_instances.values():
            task.cancel()
        print("E2E Loop stopped")
    
    async def create_workflow(self, definition: Dict) -> str:
        """Create a new workflow definition"""
        wf_def = WorkflowDefinition(**definition)
        await self.state_manager.save_workflow_definition(wf_def)
        return wf_def.id
    
    async def start_workflow(self, workflow_id: str, 
                             input_data: Dict = None) -> str:
        """Start a workflow instance"""
        
        definition = await self.state_manager.get_workflow_definition(workflow_id)
        if not definition:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        instance = WorkflowInstance(
            definition_id=workflow_id,
            definition_version=definition.version,
            input_data=input_data or {},
            status=WorkflowStatus.PENDING
        )
        
        await self.state_manager.save_instance(instance)
        
        task = asyncio.create_task(self._execute_workflow(instance))
        self.active_instances[instance.id] = task
        
        return instance.id
    
    async def get_workflow_status(self, instance_id: str) -> Dict:
        """Get workflow instance status"""
        instance = await self.state_manager.get_instance(instance_id)
        if not instance:
            return {"error": "Instance not found"}
        
        return {
            "id": instance.id,
            "status": instance.status.name,
            "current_step": instance.current_step,
            "completed_steps": instance.completed_steps,
            "progress": len(instance.completed_steps),
            "started_at": instance.started_at.isoformat() if instance.started_at else None,
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "error": instance.last_error
        }
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancel a running workflow"""
        instance = await self.state_manager.get_instance(instance_id)
        if instance and instance.status == WorkflowStatus.RUNNING:
            instance.status = WorkflowStatus.CANCELLED
            await self.state_manager.save_instance(instance)
            
            if instance_id in self.active_instances:
                self.active_instances[instance_id].cancel()
            
            return True
        return False
    
    async def _execute_workflow(self, instance: WorkflowInstance):
        """Execute workflow instance"""
        try:
            instance.status = WorkflowStatus.RUNNING
            instance.started_at = datetime.now()
            await self.state_manager.save_instance(instance)
            
            definition = await self.state_manager.get_workflow_definition(
                instance.definition_id
            )
            
            current_step_id = definition.initial_step
            
            while current_step_id and self.running:
                step = definition.steps.get(current_step_id)
                if not step:
                    break
                
                instance.current_step = current_step_id
                await self.state_manager.save_instance(instance)
                
                context = ExecutionContext(
                    workflow_instance_id=instance.id,
                    step_id=step.id,
                    input_data=instance.input_data,
                    workflow_context=instance.context,
                    step_results=instance.step_results
                )
                
                result = await self.step_executor.execute_step(step, context)
                
                if result["status"] == "success":
                    instance.step_results[step.id] = result["result"]
                    instance.completed_steps.append(step.id)
                    instance.compensation_stack.append(step.id)
                    
                    current_step_id = step.next_steps[0] if step.next_steps else None
                    
                elif result["status"] == "failed":
                    instance.status = WorkflowStatus.FAILED
                    instance.last_error = result.get("error")
                    break
                
                await self.state_manager.save_instance(instance)
            
            if instance.status != WorkflowStatus.FAILED:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
            
            await self.state_manager.save_instance(instance)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.last_error = str(e)
            await self.state_manager.save_instance(instance)
        
        finally:
            if instance.id in self.active_instances:
                del self.active_instances[instance.id]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the E2E Loop"""
    
    # Initialize E2E Loop
    e2e = E2ELoop({"db_path": "example_workflows.db"})
    
    # Register activities
    async def send_email(to: str, subject: str, body: str, **kwargs):
        print(f"Sending email to {to}: {subject}")
        return {"sent": True, "message_id": str(uuid.uuid4())}
    
    async def search_web(query: str, **kwargs):
        print(f"Searching web for: {query}")
        return {"results": [f"Result {i}" for i in range(5)]}
    
    async def generate_report(data: Dict, **kwargs):
        print(f"Generating report with {len(data)} items")
        return {"report_path": "/tmp/report.pdf"}
    
    e2e.register_activities({
        "send_email": send_email,
        "search_web": search_web,
        "generate_report": generate_report
    })
    
    await e2e.start()
    
    # Create workflow
    workflow_def = {
        "name": "ResearchAndReport",
        "description": "Research topic and generate report",
        "steps": {
            "search": WorkflowStep(
                id="search",
                name="Search Web",
                activity="search_web",
                activity_config={"query": "AI automation"},
                next_steps=["generate"]
            ),
            "generate": WorkflowStep(
                id="generate",
                name="Generate Report",
                activity="generate_report",
                activity_config={},
                next_steps=["send"]
            ),
            "send": WorkflowStep(
                id="send",
                name="Send Email",
                activity="send_email",
                activity_config={
                    "to": "user@example.com",
                    "subject": "Research Report"
                },
                next_steps=[]
            )
        },
        "initial_step": "search"
    }
    
    workflow_id = await e2e.create_workflow(workflow_def)
    print(f"Created workflow: {workflow_id}")
    
    # Start workflow
    instance_id = await e2e.start_workflow(workflow_id)
    print(f"Started workflow instance: {instance_id}")
    
    # Wait for completion
    while True:
        status = await e2e.get_workflow_status(instance_id)
        print(f"Status: {status['status']}")
        
        if status['status'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
            break
        
        await asyncio.sleep(1)
    
    await e2e.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
