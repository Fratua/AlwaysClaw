# END-TO-END LOOP TECHNICAL SPECIFICATION
## Windows 10 OpenClaw-Inspired AI Agent Framework

**Version:** 1.0.0  
**Date:** 2025  
**Component:** Loop 15 of 15 - End-to-End Workflow Automation

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Workflow Definition & Modeling](#3-workflow-definition--modeling)
4. [Trigger Detection & Handling](#4-trigger-detection--handling)
5. [Task Decomposition & Sequencing](#5-task-decomposition--sequencing)
6. [Step Execution & Monitoring](#6-step-execution--monitoring)
7. [State Management](#7-state-management)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Completion Verification](#9-completion-verification)
10. [Workflow Optimization](#10-workflow-optimization)
11. [Implementation Guide](#11-implementation-guide)
12. [API Reference](#12-api-reference)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Purpose
The End-to-End (E2E) Loop provides full workflow automation from trigger detection through task completion verification. It serves as the orchestration backbone of the AI agent system, enabling complex multi-step processes to execute autonomously with robust error handling and recovery.

### 1.2 Key Capabilities
- **Workflow Orchestration:** Define, execute, and monitor complex multi-step workflows
- **Task Chaining:** Automatically sequence dependent tasks with proper state passing
- **Trigger Handling:** Detect and respond to various trigger types (time, event, manual, API)
- **State Persistence:** Maintain workflow state across system restarts and failures
- **Error Recovery:** Automatic retry, compensation, and rollback mechanisms
- **Completion Verification:** Multi-layer verification ensuring task success

### 1.3 Integration Points
- Gmail API (triggers and notifications)
- Browser Control (web automation)
- TTS/STT (voice interaction)
- Twilio (voice/SMS)
- Windows System APIs
- Cron Scheduler
- GPT-5.2 (intelligence layer)

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 High-Level Architecture

```
+-----------------------------------------------------------------------------+
|                         END-TO-END LOOP ARCHITECTURE                        |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +--------------+    +--------------+    +--------------+                  |
|  |   TRIGGERS   |--->|  WORKFLOW    |--->|  EXECUTION   |                  |
|  |   LAYER      |    |   ENGINE     |    |   ENGINE     |                  |
|  +--------------+    +--------------+    +--------------+                  |
|         |                   |                   |                          |
|         v                   v                   v                          |
|  +--------------+    +--------------+    +--------------+                  |
|  |  Gmail/Web/  |    |   State      |    |   Step       |                  |
|  |  Time/API    |    |   Manager    |    |   Runners    |                  |
|  +--------------+    +--------------+    +--------------+                  |
|                              |                   |                          |
|                              v                   v                          |
|                       +--------------+    +--------------+                  |
|                       |  Persistence |    |  Activities  |                  |
|                       |   (SQLite)   |    |  (Actions)   |                  |
|                       +--------------+    +--------------+                  |
|                                                   |                        |
|                              +--------------------+                        |
|                              v                                             |
|                       +--------------+    +--------------+                  |
|                       |   Error      |    | Completion   |                  |
|                       |  Handler     |    |  Verifier    |                  |
|                       +--------------+    +--------------+                  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### 2.2 Core Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Trigger Manager | Detect and route triggers | Python/asyncio |
| Workflow Engine | Parse and orchestrate workflows | Custom state machine |
| Task Decomposer | Break goals into steps | GPT-5.2 + Templates |
| Step Executor | Execute individual steps | Activity runners |
| State Manager | Persist workflow state | SQLite + JSON |
| Error Handler | Manage failures and recovery | Retry + Compensation |
| Completion Verifier | Verify task completion | Multi-layer checks |
| Optimizer | Improve workflow efficiency | Analytics + ML |

### 2.3 Data Flow

```
Trigger -> Workflow Definition -> Task Decomposition -> Step Queue 
   |           |                      |                  |
Detection -> Validation -> Sequencing -> Execution -> Verification
   |           |                      |                  |
Routing  -> Instantiation -> State Mgmt -> Monitoring -> Completion
```

---

## 3. WORKFLOW DEFINITION & MODELING

### 3.1 Workflow Schema

```python
# workflow_schema.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Union
from enum import Enum, auto
from datetime import datetime
import uuid

class WorkflowStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    COMPENSATING = auto()

class StepStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    RETRYING = auto()
    COMPENSATED = auto()

class StepType(Enum):
    ACTION = auto()           # Execute an action
    DECISION = auto()         # Conditional branch
    PARALLEL = auto()         # Fork execution
    WAIT = auto()             # Wait for event/time
    SUBWORKFLOW = auto()      # Nested workflow
    HUMAN = auto()            # Human approval
    COMPENSATION = auto()     # Rollback action

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    step_type: StepType = StepType.ACTION
    
    # Execution
    activity: Optional[str] = None          # Activity to execute
    activity_config: Dict = field(default_factory=dict)
    
    # Flow control
    next_steps: List[str] = field(default_factory=list)
    condition: Optional[str] = None         # Conditional expression
    
    # Error handling
    retry_policy: Dict = field(default_factory=dict)
    on_error: Optional[str] = None          # Error handler step
    compensation: Optional[str] = None      # Compensation step
    
    # Metadata
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    # Runtime state
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
    
    # Steps
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    initial_step: str = ""
    
    # Configuration
    timeout_seconds: int = 3600
    max_retries: int = 3
    
    # Triggers
    triggers: List[Dict] = field(default_factory=list)
    
    # Variables
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowInstance:
    """Running instance of a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    definition_id: str = ""
    definition_version: str = ""
    
    # State
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    
    # Data
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    context: Dict = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Compensation tracking
    compensation_stack: List[str] = field(default_factory=list)
```

### 3.2 Workflow Definition Language (WDL)

```yaml
# example_workflow.yaml
workflow:
  name: "EmailResearchAndReport"
  description: "Research email topic and generate comprehensive report"
  version: "1.0.0"
  
  triggers:
    - type: gmail_label
      config:
        label: "ResearchRequests"
        mark_as_read: true
    - type: schedule
      config:
        cron: "0 9 * * 1"  # Weekly on Monday 9 AM
    - type: api
      config:
        endpoint: "/workflows/research"
  
  input:
    schema:
      topic:
        type: string
        required: true
      depth:
        type: enum
        values: [quick, standard, deep]
        default: standard
      output_format:
        type: enum
        values: [pdf, docx, markdown]
        default: markdown
  
  steps:
    validate_input:
      type: action
      activity: validate_input
      next: [decompose_task]
      retry:
        max_attempts: 1
    
    decompose_task:
      type: action
      activity: llm_task_decomposition
      config:
        model: "gpt-5.2"
        prompt_template: "decompose_research"
      next: [parallel_research]
      compensation: cleanup_decomposition
    
    parallel_research:
      type: parallel
      branches:
        - name: web_search
          steps: [search_web, extract_content]
        - name: academic_search
          steps: [search_scholar, download_papers]
        - name: news_search
          steps: [search_news, summarize_articles]
      merge_strategy: "concatenate"
      next: [synthesize_findings]
    
    search_web:
      type: action
      activity: browser_search
      config:
        engine: "google"
        results_limit: 10
      next: [extract_content]
    
    extract_content:
      type: action
      activity: browser_extract
      config:
        extract_type: "article"
      next: []
    
    synthesize_findings:
      type: action
      activity: llm_synthesis
      config:
        model: "gpt-5.2"
        synthesis_type: "research_report"
      next: [generate_report]
    
    generate_report:
      type: action
      activity: document_generator
      config:
        template: "research_report"
        format: "${input.output_format}"
      next: [review_report]
    
    review_report:
      type: decision
      condition: "${input.depth} == 'deep'"
      true_next: [human_review]
      false_next: [deliver_report]
    
    human_review:
      type: human
      config:
        notification: email
        timeout: 86400  # 24 hours
        escalation: auto_approve
      next: [deliver_report]
      on_timeout: deliver_report
    
    deliver_report:
      type: action
      activity: send_email
      config:
        template: "report_delivery"
        attachments: ["${steps.generate_report.output}"]
      next: [notify_completion]
    
    notify_completion:
      type: action
      activity: send_notification
      config:
        channels: [tts, desktop]
        message: "Research report completed for ${input.topic}"
      next: []
  
  error_handling:
    default_retry:
      max_attempts: 3
      backoff: exponential
      initial_delay: 5
    
    on_failure:
      action: notify_admin
      compensation: rollback_all
  
  output:
    schema:
      report_path:
        type: string
      summary:
        type: string
      sources:
        type: array
```

### 3.3 Workflow Patterns

```python
# workflow_patterns.py

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
        
        # Add branch steps
        for i, branch in enumerate(branches):
            branch_id = f"branch_{i}"
            parallel_step.metadata["branches"] = parallel_step.metadata.get("branches", [])
            parallel_step.metadata["branches"].append(branch_id)
        
        # Add merge step
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
        
        # Add branches
        for step in true_branch:
            step.id = f"true_{step.id}"
            wf.steps[step.id] = step
        
        for step in false_branch:
            step.id = f"false_{step.id}"
            wf.steps[step.id] = step
        
        return wf
    
    @staticmethod
    def retry_loop(step: WorkflowStep, 
                   condition: str,
                   max_iterations: int = 10) -> WorkflowDefinition:
        """Retry step until condition met"""
        wf = WorkflowDefinition(name="retry_loop")
        
        loop_step = WorkflowStep(
            id="loop_check",
            step_type=StepType.DECISION,
            condition=condition,
            next_steps=["execute", "complete"]
        )
        wf.steps["loop_check"] = loop_step
        wf.initial_step = "loop_check"
        
        step.id = "execute"
        step.next_steps.append("loop_check")  # Loop back
        wf.steps["execute"] = step
        
        complete = WorkflowStep(id="complete", step_type=StepType.ACTION)
        wf.steps["complete"] = complete
        
        return wf
    
    @staticmethod
    def saga_transaction(steps: List[tuple]) -> WorkflowDefinition:
        """Saga pattern with compensation"""
        wf = WorkflowDefinition(name="saga")
        
        for i, (step, compensation) in enumerate(steps):
            step_id = f"saga_step_{i}"
            step.id = step_id
            step.compensation = f"compensation_{i}"
            step.next_steps = [f"saga_step_{i+1}"] if i < len(steps) - 1 else []
            wf.steps[step_id] = step
            
            if compensation:
                compensation.id = f"compensation_{i}"
                compensation.step_type = StepType.COMPENSATION
                wf.steps[compensation.id] = compensation
        
        wf.initial_step = "saga_step_0"
        return wf
```

---

## 4. TRIGGER DETECTION & HANDLING

### 4.1 Trigger Types

```python
# trigger_system.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum, auto
import asyncio
import re

class TriggerType(Enum):
    SCHEDULE = auto()       # Cron-based time triggers
    GMAIL = auto()          # Gmail label/filter triggers
    WEBHOOK = auto()        # HTTP webhook triggers
    API = auto()            # Direct API call triggers
    FILE = auto()           # File system change triggers
    EVENT = auto()          # Internal event triggers
    MANUAL = auto()         # User-initiated triggers
    VOICE = auto()          # Voice command triggers
    SMS = auto()            # SMS message triggers
    SYSTEM = auto()         # System state triggers

@dataclass
class Trigger:
    """Workflow trigger definition"""
    id: str
    type: TriggerType
    name: str
    workflow_id: str
    config: Dict[str, Any]
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Filtering
    conditions: List[Dict] = None
    priority: int = 5  # 1-10, lower is higher priority
    
    # Rate limiting
    rate_limit: Optional[Dict] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []

@dataclass
class TriggerEvent:
    """Event that can trigger a workflow"""
    id: str
    type: TriggerType
    source: str
    timestamp: datetime
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TriggerManager:
    """Manages all workflow triggers"""
    
    def __init__(self, workflow_engine, state_manager):
        self.workflow_engine = workflow_engine
        self.state_manager = state_manager
        self.triggers: Dict[str, Trigger] = {}
        self.trigger_handlers: Dict[TriggerType, Callable] = {}
        self.running = False
        self._lock = asyncio.Lock()
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register built-in trigger handlers"""
        self.trigger_handlers[TriggerType.SCHEDULE] = self._handle_schedule_trigger
        self.trigger_handlers[TriggerType.GMAIL] = self._handle_gmail_trigger
        self.trigger_handlers[TriggerType.WEBHOOK] = self._handle_webhook_trigger
        self.trigger_handlers[TriggerType.API] = self._handle_api_trigger
        self.trigger_handlers[TriggerType.FILE] = self._handle_file_trigger
        self.trigger_handlers[TriggerType.EVENT] = self._handle_event_trigger
        self.trigger_handlers[TriggerType.MANUAL] = self._handle_manual_trigger
        self.trigger_handlers[TriggerType.VOICE] = self._handle_voice_trigger
        self.trigger_handlers[TriggerType.SMS] = self._handle_sms_trigger
        self.trigger_handlers[TriggerType.SYSTEM] = self._handle_system_trigger
    
    async def register_trigger(self, trigger: Trigger) -> str:
        """Register a new trigger"""
        async with self._lock:
            self.triggers[trigger.id] = trigger
            await self.state_manager.save_trigger(trigger)
            
            # Initialize trigger-specific monitoring
            if trigger.type == TriggerType.SCHEDULE:
                await self._init_schedule_monitor(trigger)
            elif trigger.type == TriggerType.GMAIL:
                await self._init_gmail_monitor(trigger)
            elif trigger.type == TriggerType.FILE:
                await self._init_file_monitor(trigger)
            
            return trigger.id
    
    async def unregister_trigger(self, trigger_id: str):
        """Remove a trigger"""
        async with self._lock:
            if trigger_id in self.triggers:
                trigger = self.triggers[trigger_id]
                # Stop any monitoring
                await self._stop_monitor(trigger)
                del self.triggers[trigger_id]
                await self.state_manager.delete_trigger(trigger_id)
    
    async def process_event(self, event: TriggerEvent):
        """Process an incoming trigger event"""
        # Find matching triggers
        matching_triggers = []
        async with self._lock:
            for trigger in self.triggers.values():
                if not trigger.enabled:
                    continue
                if await self._matches_trigger(event, trigger):
                    matching_triggers.append(trigger)
        
        # Sort by priority
        matching_triggers.sort(key=lambda t: t.priority)
        
        # Execute workflows
        for trigger in matching_triggers:
            if await self._check_rate_limit(trigger):
                await self._execute_trigger(trigger, event)
    
    async def _matches_trigger(self, event: TriggerEvent, trigger: Trigger) -> bool:
        """Check if event matches trigger conditions"""
        if event.type != trigger.type:
            return False
        
        # Check conditions
        for condition in trigger.conditions:
            if not self._evaluate_condition(event.payload, condition):
                return False
        
        return True
    
    def _evaluate_condition(self, payload: Dict, condition: Dict) -> bool:
        """Evaluate a single condition against payload"""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        payload_value = payload.get(field)
        
        if operator == "eq":
            return payload_value == value
        elif operator == "ne":
            return payload_value != value
        elif operator == "gt":
            return payload_value > value
        elif operator == "lt":
            return payload_value < value
        elif operator == "contains":
            return value in str(payload_value)
        elif operator == "regex":
            return re.match(value, str(payload_value)) is not None
        elif operator == "exists":
            return field in payload
        
        return False
    
    async def _execute_trigger(self, trigger: Trigger, event: TriggerEvent):
        """Execute workflow for trigger"""
        # Update trigger stats
        trigger.last_triggered = datetime.now()
        trigger.trigger_count += 1
        
        # Prepare workflow input
        workflow_input = {
            "trigger": {
                "id": trigger.id,
                "type": trigger.type.name,
                "name": trigger.name
            },
            "event": {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "payload": event.payload
            },
            "input": event.payload.get("data", {})
        }
        
        # Start workflow
        instance_id = await self.workflow_engine.start_workflow(
            workflow_id=trigger.workflow_id,
            input_data=workflow_input
        )
        
        return instance_id
```

### 4.2 Gmail Trigger Handler

```python
    async def _handle_gmail_trigger(self, trigger: Trigger, event: TriggerEvent):
        """Handle Gmail-based triggers"""
        config = trigger.config
        
        # Extract email data
        email_data = event.payload.get("email", {})
        
        # Apply filters
        if "from_filter" in config:
            if not re.match(config["from_filter"], email_data.get("from", "")):
                return None
        
        if "subject_filter" in config:
            if not re.search(config["subject_filter"], email_data.get("subject", "")):
                return None
        
        # Extract commands from email
        if config.get("extract_commands", False):
            commands = await self._extract_commands_from_email(email_data)
            event.payload["commands"] = commands
        
        # Mark as read if configured
        if config.get("mark_as_read", False):
            await self._mark_email_read(email_data["id"])
        
        return await self._execute_trigger(trigger, event)
    
    async def _extract_commands_from_email(self, email_data: Dict) -> List[Dict]:
        """Extract AI commands from email body"""
        body = email_data.get("body", "")
        
        # Use GPT to extract structured commands
        prompt = f"""
        Extract actionable commands from this email. Return as JSON array.
        
        Email:
        Subject: {email_data.get('subject', '')}
        From: {email_data.get('from', '')}
        Body: {body[:2000]}
        
        Extract:
        1. Main task/request
        2. Parameters or constraints
        3. Priority level
        4. Deadline if mentioned
        """
        
        # Call GPT-5.2 for extraction
        response = await self.llm.extract_structured(prompt)
        return response.get("commands", [])
```

### 4.3 Schedule Trigger Handler

```python
    async def _init_schedule_monitor(self, trigger: Trigger):
        """Initialize cron-based schedule monitoring"""
        from croniter import croniter
        
        config = trigger.config
        cron_expr = config.get("cron")
        
        if not cron_expr:
            return
        
        # Calculate next run
        iter = croniter(cron_expr, datetime.now())
        next_run = iter.get_next(datetime)
        
        # Schedule the trigger
        asyncio.create_task(self._schedule_trigger(trigger, next_run, iter))
    
    async def _schedule_trigger(self, trigger: Trigger, next_run: datetime, iterator):
        """Schedule and execute cron trigger"""
        while self.running and trigger.enabled:
            # Wait until next run time
            now = datetime.now()
            if next_run > now:
                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)
            
            # Create trigger event
            event = TriggerEvent(
                id=f"scheduled_{trigger.id}_{int(datetime.now().timestamp())}",
                type=TriggerType.SCHEDULE,
                source="scheduler",
                timestamp=datetime.now(),
                payload={"scheduled_time": next_run.isoformat()}
            )
            
            # Process event
            await self.process_event(event)
            
            # Calculate next run
            next_run = iterator.get_next(datetime)
```

---

## 5. TASK DECOMPOSITION & SEQUENCING

### 5.1 Task Decomposition Engine

```python
# task_decomposition.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class DecomposedTask:
    """A task broken down into executable steps"""
    id: str
    description: str
    steps: List[Dict[str, Any]]
    dependencies: List[str]
    estimated_duration: int  # seconds
    required_capabilities: List[str]
    success_criteria: List[str]
    context: Dict[str, Any]

class TaskDecomposer:
    """Decomposes high-level goals into executable workflow steps"""
    
    def __init__(self, llm_client, capability_registry):
        self.llm = llm_client
        self.capabilities = capability_registry
        self.decomposition_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load task decomposition templates"""
        return {
            "research": {
                "pattern": "parallel_gather_synthesize",
                "steps": ["search", "extract", "analyze", "synthesize", "format"]
            },
            "communication": {
                "pattern": "sequential_with_review",
                "steps": ["draft", "review", "revise", "send", "confirm"]
            },
            "data_processing": {
                "pattern": "etl_pipeline",
                "steps": ["extract", "validate", "transform", "load", "verify"]
            },
            "automation": {
                "pattern": "conditional_branching",
                "steps": ["detect", "decide", "execute", "verify", "notify"]
            }
        }
    
    async def decompose(self, goal: str, context: Dict[str, Any] = None) -> DecomposedTask:
        """Decompose a goal into executable steps"""
        
        # Determine task type
        task_type = await self._classify_task(goal)
        
        # Get decomposition template
        template = self.decomposition_templates.get(task_type, {})
        
        # Use LLM for intelligent decomposition
        decomposition = await self._llm_decompose(goal, task_type, template, context)
        
        # Validate steps against available capabilities
        validated_steps = await self._validate_steps(decomposition["steps"])
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(validated_steps)
        
        # Calculate sequence
        sequenced_steps = self._topological_sort(validated_steps, dependencies)
        
        return DecomposedTask(
            id=f"task_{hash(goal) % 1000000}",
            description=goal,
            steps=sequenced_steps,
            dependencies=dependencies,
            estimated_duration=sum(s.get("estimated_duration", 60) for s in sequenced_steps),
            required_capabilities=list(set(
                cap for step in sequenced_steps 
                for cap in step.get("required_capabilities", [])
            )),
            success_criteria=decomposition.get("success_criteria", []),
            context=context or {}
        )
    
    async def _llm_decompose(self, goal: str, task_type: str, 
                             template: Dict, context: Dict) -> Dict:
        """Use LLM to decompose task into steps"""
        
        prompt = f"""
        You are an expert task decomposition AI. Break down the following goal into 
        executable steps for an AI agent system.
        
        GOAL: {goal}
        TASK TYPE: {task_type}
        CONTEXT: {json.dumps(context, indent=2)}
        
        AVAILABLE CAPABILITIES:
        {json.dumps(self.capabilities.list_all(), indent=2)}
        
        TEMPLATE PATTERN: {template.get('pattern', 'custom')}
        
        Decompose into steps. For each step provide:
        1. step_name: Unique identifier
        2. description: What this step does
        3. activity: Which capability/activity to use
        4. activity_config: Configuration for the activity
        5. required_capabilities: List of required system capabilities
        6. estimated_duration: Estimated time in seconds
        7. success_criteria: How to verify step completion
        8. can_parallelize: Whether this step can run in parallel
        9. dependencies: List of step_names this depends on
        
        Also provide:
        - overall_success_criteria: How to verify the entire task
        - error_handling: Recommended error handling approach
        
        Return as valid JSON.
        """
        
        response = await self.llm.generate_structured(prompt)
        return response
    
    def _build_dependency_graph(self, steps: List[Dict]) -> Dict[str, List[str]]:
        """Build dependency graph from steps"""
        graph = {}
        for step in steps:
            step_id = step["step_name"]
            deps = step.get("dependencies", [])
            graph[step_id] = deps
        return graph
    
    def _topological_sort(self, steps: List[Dict], 
                          dependencies: Dict[str, List[str]]) -> List[Dict]:
        """Sort steps based on dependencies"""
        step_map = {s["step_name"]: s for s in steps}
        
        # Calculate in-degrees
        in_degree = {s: 0 for s in step_map}
        for step_id, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[step_id] += 1
        
        # Kahn's algorithm
        queue = [s for s, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            # Sort by priority if available
            queue.sort(key=lambda s: step_map[s].get("priority", 5))
            
            current = queue.pop(0)
            result.append(step_map[current])
            
            # Reduce in-degree for dependent steps
            for step_id, deps in dependencies.items():
                if current in deps:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        return result
```

### 5.2 Dynamic Sequencing

```python
    async def create_dynamic_sequence(self, decomposed_task: DecomposedTask) -> WorkflowDefinition:
        """Create a workflow definition from decomposed task"""
        
        wf = WorkflowDefinition(
            name=f"dynamic_{decomposed_task.id}",
            description=decomposed_task.description
        )
        
        # Create steps
        prev_step_id = None
        parallel_groups = []
        current_parallel = []
        
        for i, step_data in enumerate(decomposed_task.steps):
            step = WorkflowStep(
                id=f"step_{i}",
                name=step_data["step_name"],
                description=step_data["description"],
                step_type=StepType.ACTION,
                activity=step_data["activity"],
                activity_config=step_data.get("activity_config", {}),
                timeout_seconds=step_data.get("estimated_duration", 300) * 2
            )
            
            # Handle parallelization
            if step_data.get("can_parallelize", False):
                current_parallel.append(step)
            else:
                # Flush parallel group
                if current_parallel:
                    parallel_groups.append(current_parallel)
                    current_parallel = []
                
                # Add sequential step
                wf.steps[step.id] = step
                
                if prev_step_id:
                    wf.steps[prev_step_id].next_steps.append(step.id)
                else:
                    wf.initial_step = step.id
                
                prev_step_id = step.id
        
        # Handle remaining parallel group
        if current_parallel:
            parallel_groups.append(current_parallel)
        
        # Convert parallel groups to parallel steps
        for group in parallel_groups:
            if len(group) > 1:
                parallel_step = WorkflowStep(
                    id=f"parallel_{len(wf.steps)}",
                    name="parallel_execution",
                    step_type=StepType.PARALLEL,
                    metadata={"branches": [s.id for s in group]}
                )
                
                # Add parallel step to workflow
                if prev_step_id:
                    wf.steps[prev_step_id].next_steps.append(parallel_step.id)
                else:
                    wf.initial_step = parallel_step.id
                
                wf.steps[parallel_step.id] = parallel_step
                
                # Add branch steps
                for branch_step in group:
                    wf.steps[branch_step.id] = branch_step
                    parallel_step.next_steps.append(branch_step.id)
                
                prev_step_id = parallel_step.id
            else:
                # Single step, add directly
                step = group[0]
                wf.steps[step.id] = step
                if prev_step_id:
                    wf.steps[prev_step_id].next_steps.append(step.id)
                else:
                    wf.initial_step = step.id
                prev_step_id = step.id
        
        return wf
```

---

## 6. STEP EXECUTION & MONITORING

### 6.1 Step Execution Engine

```python
# step_execution.py
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import traceback

@dataclass
class ExecutionContext:
    """Context for step execution"""
    workflow_instance_id: str
    step_id: str
    input_data: Dict[str, Any]
    workflow_context: Dict[str, Any]
    step_results: Dict[str, Any]
    execution_count: int = 0

class StepExecutor:
    """Executes individual workflow steps"""
    
    def __init__(self, activity_registry, state_manager, metrics_collector):
        self.activities = activity_registry
        self.state = state_manager
        self.metrics = metrics_collector
        self.running_executions: Dict[str, asyncio.Task] = {}
    
    async def execute_step(self, step: WorkflowStep, 
                          context: ExecutionContext) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        step_key = f"{context.workflow_instance_id}:{step.id}"
        
        # Check if already running
        if step_key in self.running_executions:
            return {"status": "already_running"}
        
        # Update step status
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        context.execution_count += 1
        
        # Save state
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        # Prepare execution
        activity = self.activities.get(step.activity)
        if not activity:
            error = f"Activity '{step.activity}' not found"
            return await self._handle_step_failure(step, context, error)
        
        # Build activity input
        activity_input = self._build_activity_input(step, context)
        
        # Execute with timeout and retry
        try:
            result = await self._execute_with_retry(
                activity, 
                activity_input, 
                step.retry_policy,
                step.timeout_seconds
            )
            
            # Handle success
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
                # Execute with timeout
                return await asyncio.wait_for(
                    activity(**input_data),
                    timeout=timeout
                )
                
            except Exception as e:
                last_error = e
                
                if attempt >= max_attempts:
                    raise
                
                # Calculate delay
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
        
        # Save state
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        # Record metrics
        duration = (step.completed_at - step.started_at).total_seconds()
        await self.metrics.record_step_execution(
            step.activity, duration, True
        )
        
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
        
        # Save state
        await self.state.save_step_state(context.workflow_instance_id, step)
        
        # Record metrics
        duration = (step.completed_at - step.started_at).total_seconds()
        await self.metrics.record_step_execution(
            step.activity, duration, False
        )
        
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
            # Activity config from step
            **step.activity_config,
            
            # Workflow context
            "workflow_instance_id": context.workflow_instance_id,
            "step_id": step.id,
            
            # Previous step results (for data passing)
            "previous_results": context.step_results,
            
            # Workflow-level context
            "workflow_context": context.workflow_context,
        }
        
        # Add step-specific input if defined
        if "input_mapping" in step.activity_config:
            mapping = step.activity_config["input_mapping"]
            for key, source in mapping.items():
                # Resolve source reference
                if source.startswith("${input."):
                    path = source[8:-1]  # Remove ${input. and }
                    input_data[key] = context.input_data.get(path)
                elif source.startswith("${context."):
                    path = source[10:-1]
                    input_data[key] = context.workflow_context.get(path)
                elif source.startswith("${steps."):
                    # Reference to previous step result
                    path = source[7:-1]  # e.g., "step_name.field"
                    parts = path.split(".")
                    if len(parts) >= 2:
                        step_name, field = parts[0], ".".join(parts[1:])
                        step_result = context.step_results.get(step_name, {})
                        input_data[key] = step_result.get(field)
        
        return input_data
```

### 6.2 Execution Monitor

```python
class ExecutionMonitor:
    """Monitors workflow and step execution"""
    
    def __init__(self, state_manager, alert_manager):
        self.state = state_manager
        self.alerts = alert_manager
        self.monitored_instances: Dict[str, Dict] = {}
        self.running = False
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        self.running = True
        while self.running:
            try:
                await self._check_all_instances()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_instances(self):
        """Check status of all running workflow instances"""
        running_instances = await self.state.get_running_instances()
        
        for instance in running_instances:
            await self._check_instance(instance)
    
    async def _check_instance(self, instance: WorkflowInstance):
        """Check a single workflow instance"""
        instance_id = instance.id
        
        # Check for timeout
        if instance.started_at:
            elapsed = (datetime.now() - instance.started_at).total_seconds()
            
            # Get workflow definition for timeout
            definition = await self.state.get_workflow_definition(
                instance.definition_id
            )
            
            if elapsed > definition.timeout_seconds:
                await self._handle_timeout(instance)
                return
        
        # Check current step status
        if instance.current_step:
            step = definition.steps.get(instance.current_step)
            if step and step.status == StepStatus.RUNNING:
                # Check step timeout
                if step.started_at:
                    step_elapsed = (datetime.now() - step.started_at).total_seconds()
                    if step_elapsed > step.timeout_seconds:
                        await self._handle_step_timeout(instance, step)
    
    async def _handle_timeout(self, instance: WorkflowInstance):
        """Handle workflow timeout"""
        instance.status = WorkflowStatus.FAILED
        instance.last_error = "Workflow timeout"
        
        await self.state.save_instance(instance)
        
        # Send alert
        await self.alerts.send_alert({
            "type": "workflow_timeout",
            "instance_id": instance.id,
            "workflow": instance.definition_id,
            "elapsed_time": (datetime.now() - instance.started_at).total_seconds()
        })
    
    async def _handle_step_timeout(self, instance: WorkflowInstance, step: WorkflowStep):
        """Handle step timeout"""
        step.status = StepStatus.FAILED
        step.error = "Step timeout"
        
        await self.state.save_step_state(instance.id, step)
        
        # Check if step has retry configured
        if step.retry_policy.get("max_attempts", 0) > 0:
            # Trigger retry
            pass  # Handled by executor
        else:
            # Fail workflow
            instance.status = WorkflowStatus.FAILED
            instance.last_error = f"Step '{step.name}' timeout"
            await self.state.save_instance(instance)
```

---

## 7. STATE MANAGEMENT

### 7.1 State Manager

```python
# state_management.py
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiosqlite

class WorkflowStateManager:
    """Manages persistence of workflow state"""
    
    def __init__(self, db_path: str = "workflows.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Workflow definitions
                CREATE TABLE IF NOT EXISTS workflow_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    version TEXT,
                    definition_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Workflow instances
                CREATE TABLE IF NOT EXISTS workflow_instances (
                    id TEXT PRIMARY KEY,
                    definition_id TEXT NOT NULL,
                    definition_version TEXT,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    completed_steps TEXT,  -- JSON array
                    input_data TEXT,  -- JSON
                    output_data TEXT,  -- JSON
                    step_results TEXT,  -- JSON
                    context TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    compensation_stack TEXT,  -- JSON array
                    FOREIGN KEY (definition_id) REFERENCES workflow_definitions(id)
                );
                
                -- Step states
                CREATE TABLE IF NOT EXISTS step_states (
                    instance_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT,  -- JSON
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    PRIMARY KEY (instance_id, step_id),
                    FOREIGN KEY (instance_id) REFERENCES workflow_instances(id)
                );
                
                -- Triggers
                CREATE TABLE IF NOT EXISTS triggers (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    config TEXT NOT NULL,  -- JSON
                    enabled BOOLEAN DEFAULT 1,
                    last_triggered TIMESTAMP,
                    trigger_count INTEGER DEFAULT 0,
                    conditions TEXT,  -- JSON
                    priority INTEGER DEFAULT 5,
                    rate_limit TEXT,  -- JSON
                    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(id)
                );
                
                -- Execution history
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    step_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,  -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (instance_id) REFERENCES workflow_instances(id)
                );
                
                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_instances_status 
                    ON workflow_instances(status);
                CREATE INDEX IF NOT EXISTS idx_instances_definition 
                    ON workflow_instances(definition_id);
                CREATE INDEX IF NOT EXISTS idx_history_instance 
                    ON execution_history(instance_id);
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
    
    async def log_execution_event(self, instance_id: str, step_id: Optional[str],
                                   event_type: str, event_data: Dict):
        """Log execution event for audit trail"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO execution_history
                (instance_id, step_id, event_type, event_data)
                VALUES (?, ?, ?, ?)
            """, (
                instance_id,
                step_id,
                event_type,
                json.dumps(event_data)
            ))
            await db.commit()
```

### 7.2 State Recovery

```python
    async def recover_interrupted_workflows(self) -> List[WorkflowInstance]:
        """Recover workflows that were interrupted (e.g., system restart)"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Find workflows that were running but not completed
            async with db.execute("""
                SELECT * FROM workflow_instances 
                WHERE status IN (?, ?) 
                AND started_at < datetime('now', '-1 hour')
            """, (WorkflowStatus.RUNNING.name, WorkflowStatus.PAUSED.name)) as cursor:
                rows = await cursor.fetchall()
                
                recovered = []
                for row in rows:
                    instance = self._row_to_instance(row)
                    
                    # Check if truly stuck (no recent activity)
                    last_activity = await self._get_last_activity(db, instance.id)
                    if last_activity and (datetime.now() - last_activity).seconds > 3600:
                        # Mark for recovery
                        instance.status = WorkflowStatus.PAUSED
                        instance.last_error = "Recovered after interruption"
                        await self.save_instance(instance)
                        recovered.append(instance)
                
                return recovered
    
    async def _get_last_activity(self, db, instance_id: str) -> Optional[datetime]:
        """Get timestamp of last activity for instance"""
        async with db.execute(
            "SELECT MAX(timestamp) FROM execution_history WHERE instance_id = ?",
            (instance_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0])
            return None
```

---

## 8. ERROR HANDLING & RECOVERY

### 8.1 Error Handler

```python
# error_handling.py
from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
import asyncio

class ErrorSeverity(Enum):
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()

class ErrorHandler:
    """Handles errors in workflow execution"""
    
    def __init__(self, state_manager, notification_manager):
        self.state = state_manager
        self.notifications = notification_manager
        self.error_strategies: Dict[str, Callable] = {}
        self._register_strategies()
    
    def _register_strategies(self):
        """Register error handling strategies"""
        self.error_strategies = {
            "retry": self._handle_retry,
            "skip": self._handle_skip,
            "compensate": self._handle_compensation,
            "escalate": self._handle_escalation,
            "abort": self._handle_abort,
            "ignore": self._handle_ignore
        }
    
    async def handle_step_error(self, instance: WorkflowInstance,
                                 step: WorkflowStep,
                                 error: str,
                                 severity: ErrorSeverity = ErrorSeverity.ERROR) -> Dict[str, Any]:
        """Handle an error in step execution"""
        
        # Determine error handling strategy
        strategy = self._determine_strategy(step, error, severity)
        
        # Execute strategy
        handler = self.error_strategies.get(strategy)
        if handler:
            return await handler(instance, step, error)
        
        # Default: abort
        return await self._handle_abort(instance, step, error)
    
    def _determine_strategy(self, step: WorkflowStep, error: str,
                            severity: ErrorSeverity) -> str:
        """Determine appropriate error handling strategy"""
        
        # Check step-specific configuration
        if step.on_error:
            return step.on_error
        
        # Check retry policy
        if step.retry_policy.get("max_attempts", 0) > 0:
            current_attempts = step.retry_policy.get("attempts", 0)
            if current_attempts < step.retry_policy["max_attempts"]:
                return "retry"
        
        # Based on severity
        if severity == ErrorSeverity.WARNING:
            return "ignore"
        elif severity == ErrorSeverity.CRITICAL:
            return "compensate"
        elif severity == ErrorSeverity.FATAL:
            return "abort"
        
        # Default
        return "escalate"
    
    async def _handle_retry(self, instance: WorkflowInstance,
                            step: WorkflowStep, error: str) -> Dict[str, Any]:
        """Retry the failed step"""
        
        # Increment retry count
        step.retry_policy["attempts"] = step.retry_policy.get("attempts", 0) + 1
        step.status = StepStatus.RETRYING
        
        # Save state
        await self.state.save_step_state(instance.id, step)
        
        # Schedule retry
        delay = self._calculate_retry_delay(step)
        
        return {
            "action": "retry",
            "delay": delay,
            "attempt": step.retry_policy["attempts"],
            "max_attempts": step.retry_policy.get("max_attempts", 3)
        }
    
    def _calculate_retry_delay(self, step: WorkflowStep) -> int:
        """Calculate delay before retry"""
        attempts = step.retry_policy.get("attempts", 1)
        backoff = step.retry_policy.get("backoff", "exponential")
        initial = step.retry_policy.get("initial_delay", 5)
        max_delay = step.retry_policy.get("max_delay", 300)
        
        if backoff == "exponential":
            delay = initial * (2 ** (attempts - 1))
        elif backoff == "linear":
            delay = initial * attempts
        else:
            delay = initial
        
        return min(delay, max_delay)
    
    async def _handle_compensation(self, instance: WorkflowInstance,
                                    failed_step: WorkflowStep,
                                    error: str) -> Dict[str, Any]:
        """Execute compensation (rollback) for failed workflow"""
        
        instance.status = WorkflowStatus.COMPENSATING
        await self.state.save_instance(instance)
        
        # Get compensation stack (completed steps in reverse order)
        compensation_stack = list(reversed(instance.compensation_stack))
        
        compensation_results = []
        for step_id in compensation_stack:
            step = await self.state.get_step_state(instance.id, step_id)
            if step and step.compensation:
                # Execute compensation activity
                result = await self._execute_compensation(instance, step)
                compensation_results.append({
                    "step": step_id,
                    "compensation": step.compensation,
                    "result": result
                })
        
        # Mark workflow as failed after compensation
        instance.status = WorkflowStatus.FAILED
        instance.last_error = f"Failed: {error}. Compensation executed."
        await self.state.save_instance(instance)
        
        return {
            "action": "compensated",
            "compensation_results": compensation_results
        }
    
    async def _execute_compensation(self, instance: WorkflowInstance,
                                     step: WorkflowStep) -> Any:
        """Execute a single compensation action"""
        # Get compensation activity
        compensation_activity = step.compensation
        
        # Execute
        try:
            activity = self.activities.get(compensation_activity)
            if activity:
                return await activity(
                    workflow_instance_id=instance.id,
                    original_step=step.id,
                    original_result=step.result
                )
        except Exception as e:
            # Log compensation failure
            await self.state.log_execution_event(
                instance.id, step.id, "compensation_failed",
                {"error": str(e)}
            )
            return {"status": "compensation_failed", "error": str(e)}
```

### 8.2 Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker pattern for activity execution"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failures: Dict[str, int] = {}
        self.last_failure: Dict[str, float] = {}
        self.states: Dict[str, str] = {}  # closed, open, half-open
    
    def can_execute(self, activity_name: str) -> bool:
        """Check if activity can be executed"""
        state = self.states.get(activity_name, "closed")
        
        if state == "closed":
            return True
        
        if state == "open":
            # Check if recovery timeout has passed
            last_fail = self.last_failure.get(activity_name, 0)
            if time.time() - last_fail > self.recovery_timeout:
                self.states[activity_name] = "half-open"
                return True
            return False
        
        if state == "half-open":
            return True
        
        return True
    
    def record_success(self, activity_name: str):
        """Record successful execution"""
        self.failures[activity_name] = 0
        self.states[activity_name] = "closed"
    
    def record_failure(self, activity_name: str):
        """Record failed execution"""
        self.failures[activity_name] = self.failures.get(activity_name, 0) + 1
        self.last_failure[activity_name] = time.time()
        
        if self.failures[activity_name] >= self.failure_threshold:
            self.states[activity_name] = "open"
```

---

## 9. COMPLETION VERIFICATION

### 9.1 Verification Engine

```python
# completion_verification.py
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto

class VerificationLevel(Enum):
    BASIC = auto()      # Simple status check
    STANDARD = auto()   # Result validation
    THOROUGH = auto()   # Multi-layer verification
    EXHAUSTIVE = auto() # Complete audit

@dataclass
class VerificationResult:
    """Result of completion verification"""
    verified: bool
    confidence: float  # 0.0 - 1.0
    checks: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]

class CompletionVerifier:
    """Verifies workflow and step completion"""
    
    def __init__(self, llm_client, external_validators: Dict[str, Callable] = None):
        self.llm = llm_client
        self.validators = external_validators or {}
        self.verification_strategies = self._load_strategies()
    
    def _load_strategies(self) -> Dict[str, Callable]:
        """Load verification strategies"""
        return {
            "status_check": self._verify_status,
            "result_validation": self._verify_result,
            "side_effect_check": self._verify_side_effects,
            "consistency_check": self._verify_consistency,
            "llm_evaluation": self._verify_with_llm
        }
    
    async def verify_workflow_completion(self, instance: WorkflowInstance,
                                          level: VerificationLevel = VerificationLevel.STANDARD) -> VerificationResult:
        """Verify that a workflow has completed successfully"""
        
        checks = []
        warnings = []
        errors = []
        
        # 1. Status verification
        status_result = await self._verify_status(instance)
        checks.append({"name": "status", "passed": status_result["passed"], "details": status_result})
        if not status_result["passed"]:
            errors.append(f"Status check failed: {status_result.get('error')}")
        
        # 2. All steps completed
        steps_result = await self._verify_all_steps_completed(instance)
        checks.append({"name": "steps_complete", "passed": steps_result["passed"], "details": steps_result})
        if not steps_result["passed"]:
            errors.append(f"Steps completion check failed: {steps_result.get('incomplete_steps', [])}")
        
        # 3. Output validation (STANDARD and above)
        if level.value >= VerificationLevel.STANDARD.value:
            output_result = await self._verify_output(instance)
            checks.append({"name": "output", "passed": output_result["passed"], "details": output_result})
            if not output_result["passed"]:
                errors.append(f"Output validation failed: {output_result.get('error')}")
        
        # 4. Side effect verification (THOROUGH and above)
        if level.value >= VerificationLevel.THOROUGH.value:
            side_effect_result = await self._verify_side_effects(instance)
            checks.append({"name": "side_effects", "passed": side_effect_result["passed"], "details": side_effect_result})
            if not side_effect_result["passed"]:
                warnings.append(f"Side effect verification issues: {side_effect_result.get('issues', [])}")
        
        # 5. LLM-based evaluation (EXHAUSTIVE)
        if level.value >= VerificationLevel.EXHAUSTIVE.value:
            llm_result = await self._verify_with_llm(instance)
            checks.append({"name": "llm_evaluation", "passed": llm_result["passed"], "details": llm_result})
            if not llm_result["passed"]:
                warnings.append(f"LLM evaluation concerns: {llm_result.get('concerns', [])}")
        
        # Calculate confidence
        passed_checks = sum(1 for c in checks if c["passed"])
        confidence = passed_checks / len(checks) if checks else 0.0
        
        # Determine overall verification
        verified = len(errors) == 0 and confidence >= 0.8
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            checks=checks,
            warnings=warnings,
            errors=errors
        )
    
    async def _verify_status(self, instance: WorkflowInstance) -> Dict:
        """Verify workflow status"""
        if instance.status == WorkflowStatus.COMPLETED:
            return {"passed": True, "status": instance.status.name}
        
        return {
            "passed": False,
            "status": instance.status.name,
            "error": f"Workflow status is {instance.status.name}, not COMPLETED"
        }
    
    async def _verify_all_steps_completed(self, instance: WorkflowInstance) -> Dict:
        """Verify all workflow steps completed"""
        definition = await self.state.get_workflow_definition(instance.definition_id)
        
        incomplete_steps = []
        for step_id, step in definition.steps.items():
            if step_id not in instance.completed_steps:
                # Check if step was actually executed
                step_state = await self.state.get_step_state(instance.id, step_id)
                if not step_state or step_state.status != StepStatus.COMPLETED:
                    incomplete_steps.append(step_id)
        
        return {
            "passed": len(incomplete_steps) == 0,
            "incomplete_steps": incomplete_steps,
            "total_steps": len(definition.steps),
            "completed_steps": len(instance.completed_steps)
        }
    
    async def _verify_output(self, instance: WorkflowInstance) -> Dict:
        """Verify workflow output against schema"""
        definition = await self.state.get_workflow_definition(instance.definition_id)
        
        if not definition.output_schema:
            return {"passed": True, "message": "No output schema defined"}
        
        output = instance.output_data
        errors = []
        
        for field, schema in definition.output_schema.items():
            if schema.get("required", False) and field not in output:
                errors.append(f"Required field '{field}' missing from output")
            
            if field in output:
                value = output[field]
                field_type = schema.get("type")
                
                if field_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
                elif field_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
                elif field_type == "array" and not isinstance(value, list):
                    errors.append(f"Field '{field}' should be array, got {type(value).__name__}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "output_fields": list(output.keys())
        }
    
    async def _verify_with_llm(self, instance: WorkflowInstance) -> Dict:
        """Use LLM to evaluate workflow completion quality"""
        
        prompt = f"""
        Evaluate whether this workflow completed successfully and achieved its goal.
        
        WORKFLOW: {instance.definition_id}
        GOAL: {instance.context.get('goal', 'Unknown')}
        
        INPUT:
        ```json
        {json.dumps(instance.input_data, indent=2)}
        ```
        
        OUTPUT:
        ```json
        {json.dumps(instance.output_data, indent=2)}
        ```
        
        STEP RESULTS:
        ```json
        {json.dumps(instance.step_results, indent=2)}
        ```
        
        Evaluate:
        1. Did the workflow achieve its stated goal? (yes/no/partial)
        2. Is the output complete and correct?
        3. Are there any missing or incorrect elements?
        4. What is your confidence level? (0.0-1.0)
        5. Any concerns or recommendations?
        
        Return as JSON with fields: achieved, complete, concerns, confidence, recommendations
        """
        
        try:
            evaluation = await self.llm.generate_structured(prompt)
            
            return {
                "passed": evaluation.get("achieved") == "yes" and evaluation.get("complete") == True,
                "achieved": evaluation.get("achieved"),
                "complete": evaluation.get("complete"),
                "confidence": evaluation.get("confidence", 0.5),
                "concerns": evaluation.get("concerns", []),
                "recommendations": evaluation.get("recommendations", [])
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"LLM evaluation failed: {str(e)}"
            }
```

---

## 10. WORKFLOW OPTIMIZATION

### 10.1 Optimization Engine

```python
# workflow_optimization.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

@dataclass
class OptimizationSuggestion:
    """Suggestion for workflow optimization"""
    type: str  # parallelize, cache, retry, timeout, etc.
    description: str
    impact: str  # high, medium, low
    current_value: Any
    suggested_value: Any
    expected_improvement: str

class WorkflowOptimizer:
    """Analyzes and optimizes workflows based on execution data"""
    
    def __init__(self, metrics_collector, state_manager):
        self.metrics = metrics_collector
        self.state = state_manager
    
    async def analyze_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze workflow performance and suggest optimizations"""
        
        # Get execution history
        executions = await self.state.get_workflow_executions(workflow_id, limit=100)
        
        if not executions:
            return {"status": "no_data", "message": "No execution data available"}
        
        analysis = {
            "workflow_id": workflow_id,
            "total_executions": len(executions),
            "success_rate": self._calculate_success_rate(executions),
            "avg_duration": self._calculate_avg_duration(executions),
            "bottlenecks": self._identify_bottlenecks(executions),
            "error_patterns": self._analyze_errors(executions),
            "suggestions": []
        }
        
        # Generate optimization suggestions
        analysis["suggestions"] = await self._generate_suggestions(analysis)
        
        return analysis
    
    def _calculate_success_rate(self, executions: List[Dict]) -> float:
        """Calculate workflow success rate"""
        successful = sum(1 for e in executions if e.get("status") == "COMPLETED")
        return successful / len(executions) if executions else 0.0
    
    def _calculate_avg_duration(self, executions: List[Dict]) -> float:
        """Calculate average execution duration"""
        durations = []
        for e in executions:
            if e.get("started_at") and e.get("completed_at"):
                duration = (e["completed_at"] - e["started_at"]).total_seconds()
                durations.append(duration)
        
        return statistics.mean(durations) if durations else 0.0
    
    def _identify_bottlenecks(self, executions: List[Dict]) -> List[Dict]:
        """Identify slow steps that may be bottlenecks"""
        step_durations: Dict[str, List[float]] = {}
        
        for e in executions:
            for step_id, result in e.get("step_results", {}).items():
                if "duration" in result:
                    if step_id not in step_durations:
                        step_durations[step_id] = []
                    step_durations[step_id].append(result["duration"])
        
        bottlenecks = []
        for step_id, durations in step_durations.items():
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            if avg_duration > 60:  # Steps taking > 1 minute on average
                bottlenecks.append({
                    "step_id": step_id,
                    "avg_duration": avg_duration,
                    "max_duration": max_duration,
                    "execution_count": len(durations)
                })
        
        # Sort by average duration
        bottlenecks.sort(key=lambda x: x["avg_duration"], reverse=True)
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    async def _generate_suggestions(self, analysis: Dict) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        # Suggest parallelization for independent steps
        if len(analysis["bottlenecks"]) > 1:
            suggestions.append(OptimizationSuggestion(
                type="parallelize",
                description="Multiple slow steps detected. Consider parallelizing independent steps.",
                impact="high",
                current_value="Sequential execution",
                suggested_value="Parallel execution for independent steps",
                expected_improvement=f"~{len(analysis['bottlenecks']) * 20}% time reduction"
            ))
        
        # Suggest retry policy adjustments for errors
        if analysis["error_patterns"]:
            for error_pattern in analysis["error_patterns"][:3]:
                if error_pattern["count"] > 5:
                    suggestions.append(OptimizationSuggestion(
                        type="retry_policy",
                        description=f"Frequent error pattern: {error_pattern['type']}",
                        impact="medium",
                        current_value="Default retry policy",
                        suggested_value=f"Increase retries for {error_pattern['step_id']}",
                        expected_improvement=f"~{error_pattern['count']} fewer failures"
                    ))
        
        # Suggest timeout adjustments
        for bottleneck in analysis["bottlenecks"]:
            if bottleneck["avg_duration"] > bottleneck.get("timeout", 300) * 0.8:
                suggestions.append(OptimizationSuggestion(
                    type="timeout",
                    description=f"Step '{bottleneck['step_id']}' frequently times out",
                    impact="high",
                    current_value=f"{bottleneck.get('timeout', 300)}s timeout",
                    suggested_value=f"{int(bottleneck['avg_duration'] * 1.5)}s timeout",
                    expected_improvement="Reduced timeout failures"
                ))
        
        # Suggest caching for repeated operations
        suggestions.append(OptimizationSuggestion(
            type="cache",
            description="Consider caching results of expensive operations",
            impact="medium",
            current_value="No caching",
            suggested_value="Implement result caching",
            expected_improvement="30-50% faster repeat executions"
        ))
        
        return suggestions
    
    async def optimize_workflow(self, workflow_id: str) -> Dict:
        """Automatically apply optimizations to workflow"""
        
        analysis = await self.analyze_workflow(workflow_id)
        definition = await self.state.get_workflow_definition(workflow_id)
        
        applied = []
        
        for suggestion in analysis["suggestions"]:
            if suggestion.impact == "high":
                # Apply high-impact suggestions automatically
                if suggestion.type == "timeout":
                    # Update timeout for specific step
                    step_id = suggestion.description.split("'")[1]
                    if step_id in definition.steps:
                        new_timeout = int(suggestion.suggested_value.split("s")[0])
                        definition.steps[step_id].timeout_seconds = new_timeout
                        applied.append(f"Updated timeout for {step_id}")
                
                elif suggestion.type == "retry_policy":
                    # Update retry policy
                    step_id = suggestion.description.split("for ")[1]
                    if step_id in definition.steps:
                        definition.steps[step_id].retry_policy["max_attempts"] = 5
                        applied.append(f"Updated retry policy for {step_id}")
        
        # Save optimized definition
        if applied:
            definition.version = self._increment_version(definition.version)
            await self.state.save_workflow_definition(definition)
        
        return {
            "workflow_id": workflow_id,
            "optimizations_applied": applied,
            "new_version": definition.version if applied else None
        }
    
    def _increment_version(self, version: str) -> str:
        """Increment version number"""
        parts = version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            return ".".join(parts)
        return version + ".1"
```

---

## 11. IMPLEMENTATION GUIDE

### 11.1 Directory Structure

```
openclaw/
├── loops/
│   └── e2e_loop/
│       ├── __init__.py
│       ├── workflow_engine.py
│       ├── trigger_manager.py
│       ├── task_decomposer.py
│       ├── step_executor.py
│       ├── state_manager.py
│       ├── error_handler.py
│       ├── completion_verifier.py
│       ├── optimizer.py
│       ├── patterns.py
│       └── config/
│           ├── default_workflows.yaml
│           └── activity_registry.yaml
├── activities/
│   ├── __init__.py
│   ├── gmail_activities.py
│   ├── browser_activities.py
│   ├── system_activities.py
│   ├── voice_activities.py
│   └── twilio_activities.py
├── storage/
│   └── workflows.db
└── tests/
    └── test_e2e_loop.py
```

### 11.2 Main Workflow Engine

```python
# workflow_engine.py
class WorkflowEngine:
    """Main workflow orchestration engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state_manager = WorkflowStateManager(config.get("db_path", "workflows.db"))
        self.trigger_manager = TriggerManager(self, self.state_manager)
        self.task_decomposer = TaskDecomposer(config["llm"], config["capabilities"])
        self.step_executor = StepExecutor(
            config["activities"], 
            self.state_manager,
            config["metrics"]
        )
        self.error_handler = ErrorHandler(self.state_manager, config["notifications"])
        self.verifier = CompletionVerifier(config["llm"])
        self.optimizer = WorkflowOptimizer(config["metrics"], self.state_manager)
        self.running = False
        self.active_instances: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start the workflow engine"""
        self.running = True
        
        # Recover interrupted workflows
        recovered = await self.state_manager.recover_interrupted_workflows()
        for instance in recovered:
            await self.resume_workflow(instance.id)
        
        # Start trigger monitoring
        await self.trigger_manager.start()
        
        print(f"Workflow engine started. Recovered {len(recovered)} workflows.")
    
    async def stop(self):
        """Stop the workflow engine gracefully"""
        self.running = False
        
        # Cancel active instances
        for task in self.active_instances.values():
            task.cancel()
        
        await self.trigger_manager.stop()
        print("Workflow engine stopped.")
    
    async def start_workflow(self, workflow_id: str, 
                             input_data: Dict = None) -> str:
        """Start a new workflow instance"""
        
        # Load definition
        definition = await self.state_manager.get_workflow_definition(workflow_id)
        if not definition:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        # Create instance
        instance = WorkflowInstance(
            definition_id=workflow_id,
            definition_version=definition.version,
            input_data=input_data or {},
            status=WorkflowStatus.PENDING
        )
        
        # Save instance
        await self.state_manager.save_instance(instance)
        
        # Start execution
        task = asyncio.create_task(self._execute_workflow(instance))
        self.active_instances[instance.id] = task
        
        return instance.id
    
    async def _execute_workflow(self, instance: WorkflowInstance):
        """Execute workflow instance"""
        try:
            # Update status
            instance.status = WorkflowStatus.RUNNING
            instance.started_at = datetime.now()
            await self.state_manager.save_instance(instance)
            
            # Load definition
            definition = await self.state_manager.get_workflow_definition(
                instance.definition_id
            )
            
            # Start from initial step
            current_step_id = definition.initial_step
            
            while current_step_id and self.running:
                step = definition.steps.get(current_step_id)
                if not step:
                    break
                
                # Update current step
                instance.current_step = current_step_id
                await self.state_manager.save_instance(instance)
                
                # Execute step
                context = ExecutionContext(
                    workflow_instance_id=instance.id,
                    step_id=step.id,
                    input_data=instance.input_data,
                    workflow_context=instance.context,
                    step_results=instance.step_results
                )
                
                result = await self.step_executor.execute_step(step, context)
                
                # Handle result
                if result["status"] == "success":
                    # Save result
                    instance.step_results[step.id] = result["result"]
                    instance.completed_steps.append(step.id)
                    instance.compensation_stack.append(step.id)
                    
                    # Move to next step
                    current_step_id = self._get_next_step(step, result, instance)
                    
                elif result["status"] == "failed":
                    # Handle error
                    error_result = await self.error_handler.handle_step_error(
                        instance, step, result.get("error", "Unknown error")
                    )
                    
                    if error_result["action"] == "retry":
                        # Retry same step
                        await asyncio.sleep(error_result["delay"])
                        continue
                    elif error_result["action"] == "compensated":
                        # Workflow compensated and failed
                        break
                    else:
                        # Fail workflow
                        instance.status = WorkflowStatus.FAILED
                        instance.last_error = result.get("error")
                        break
                
                # Save state after each step
                await self.state_manager.save_instance(instance)
            
            # Workflow completed
            if instance.status != WorkflowStatus.FAILED:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                
                # Verify completion
                verification = await self.verifier.verify_workflow_completion(instance)
                instance.context["verification"] = {
                    "verified": verification.verified,
                    "confidence": verification.confidence
                }
            
            await self.state_manager.save_instance(instance)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.last_error = str(e)
            await self.state_manager.save_instance(instance)
        
        finally:
            # Remove from active instances
            if instance.id in self.active_instances:
                del self.active_instances[instance.id]
    
    def _get_next_step(self, current_step: WorkflowStep, 
                       result: Dict, instance: WorkflowInstance) -> Optional[str]:
        """Determine next step based on current step and result"""
        
        if current_step.step_type == StepType.DECISION:
            # Evaluate condition
            condition_result = self._evaluate_condition(
                current_step.condition, instance
            )
            if condition_result:
                return current_step.next_steps[0] if current_step.next_steps else None
            else:
                return current_step.next_steps[1] if len(current_step.next_steps) > 1 else None
        
        elif current_step.step_type == StepType.PARALLEL:
            # Wait for all branches to complete
            branches = current_step.metadata.get("branches", [])
            all_complete = all(
                b in instance.completed_steps for b in branches
            )
            if all_complete:
                return current_step.next_steps[0] if current_step.next_steps else None
            return None  # Wait for more branches
        
        else:
            # Sequential: return first next step
            return current_step.next_steps[0] if current_step.next_steps else None
```

---

## 12. API REFERENCE

### 12.1 Public API

```python
# Public API for E2E Loop

class E2ELoopAPI:
    """Public API for End-to-End Loop"""
    
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
    
    # Workflow Management
    async def create_workflow(self, definition: Dict) -> str:
        """Create a new workflow definition"""
        wf_def = WorkflowDefinition(**definition)
        await self.engine.state_manager.save_workflow_definition(wf_def)
        return wf_def.id
    
    async def start_workflow(self, workflow_id: str, input_data: Dict = None) -> str:
        """Start a workflow instance"""
        return await self.engine.start_workflow(workflow_id, input_data)
    
    async def get_workflow_status(self, instance_id: str) -> Dict:
        """Get workflow instance status"""
        instance = await self.engine.state_manager.get_instance(instance_id)
        if not instance:
            return {"error": "Instance not found"}
        
        return {
            "id": instance.id,
            "status": instance.status.name,
            "current_step": instance.current_step,
            "completed_steps": instance.completed_steps,
            "progress": len(instance.completed_steps),
            "started_at": instance.started_at,
            "completed_at": instance.completed_at,
            "error": instance.last_error
        }
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancel a running workflow"""
        instance = await self.engine.state_manager.get_instance(instance_id)
        if instance and instance.status == WorkflowStatus.RUNNING:
            instance.status = WorkflowStatus.CANCELLED
            await self.engine.state_manager.save_instance(instance)
            
            # Cancel task
            if instance_id in self.engine.active_instances:
                self.engine.active_instances[instance_id].cancel()
            
            return True
        return False
    
    # Trigger Management
    async def register_trigger(self, trigger_config: Dict) -> str:
        """Register a new trigger"""
        trigger = Trigger(**trigger_config)
        return await self.engine.trigger_manager.register_trigger(trigger)
    
    async def unregister_trigger(self, trigger_id: str):
        """Unregister a trigger"""
        await self.engine.trigger_manager.unregister_trigger(trigger_id)
    
    # Task Decomposition
    async def decompose_task(self, goal: str, context: Dict = None) -> DecomposedTask:
        """Decompose a goal into executable steps"""
        return await self.engine.task_decomposer.decompose(goal, context)
    
    # Verification
    async def verify_completion(self, instance_id: str, 
                                 level: str = "STANDARD") -> VerificationResult:
        """Verify workflow completion"""
        instance = await self.engine.state_manager.get_instance(instance_id)
        if not instance:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                checks=[],
                warnings=[],
                errors=["Instance not found"]
            )
        
        level_enum = VerificationLevel[level.upper()]
        return await self.engine.verifier.verify_workflow_completion(instance, level_enum)
    
    # Optimization
    async def analyze_workflow(self, workflow_id: str) -> Dict:
        """Analyze workflow for optimization opportunities"""
        return await self.engine.optimizer.analyze_workflow(workflow_id)
    
    async def optimize_workflow(self, workflow_id: str) -> Dict:
        """Apply optimizations to workflow"""
        return await self.engine.optimizer.optimize_workflow(workflow_id)
```

---

## APPENDIX

### A. Configuration Example

```yaml
# e2e_loop_config.yaml
e2e_loop:
  database:
    path: "storage/workflows.db"
  
  execution:
    max_concurrent_workflows: 10
    default_timeout: 3600
    default_retry_attempts: 3
    enable_parallel_execution: true
  
  triggers:
    gmail:
      enabled: true
      check_interval: 30
      labels: ["AIRequests", "ResearchTasks"]
    
    schedule:
      enabled: true
      timezone: "America/New_York"
    
    webhook:
      enabled: true
      port: 8080
      auth_required: true
  
  verification:
    default_level: "STANDARD"
    llm_verification: true
  
  optimization:
    auto_optimize: true
    analysis_interval: 86400  # Daily
```

### B. Activity Registry

```python
# Built-in activities for Windows 10 AI Agent
ACTIVITIES = {
    # Gmail
    "gmail_send": gmail_activities.send_email,
    "gmail_read": gmail_activities.read_emails,
    "gmail_search": gmail_activities.search_emails,
    "gmail_label": gmail_activities.manage_labels,
    
    # Browser
    "browser_navigate": browser_activities.navigate,
    "browser_click": browser_activities.click,
    "browser_type": browser_activities.type_text,
    "browser_extract": browser_activities.extract_content,
    "browser_search": browser_activities.web_search,
    "browser_screenshot": browser_activities.screenshot,
    
    # System
    "system_command": system_activities.run_command,
    "system_file_operation": system_activities.file_operation,
    "system_process": system_activities.manage_process,
    "system_registry": system_activities.registry_operation,
    
    # Voice
    "tts_speak": voice_activities.text_to_speech,
    "stt_listen": voice_activities.speech_to_text,
    
    # Twilio
    "twilio_call": twilio_activities.make_call,
    "twilio_sms": twilio_activities.send_sms,
    "twilio_recording": twilio_activities.get_recording,
    
    # LLM
    "llm_generate": llm_activities.generate,
    "llm_analyze": llm_activities.analyze,
    "llm_extract": llm_activities.extract_structured,
    
    # Data
    "database_query": data_activities.query_database,
    "file_read": data_activities.read_file,
    "file_write": data_activities.write_file,
    "api_call": data_activities.http_request
}
```

---

**END OF SPECIFICATION**

*This document provides comprehensive technical specifications for the End-to-End Loop component of the Windows 10 OpenClaw-inspired AI Agent Framework.*
