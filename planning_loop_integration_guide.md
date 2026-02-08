# Advanced Planning Loop Integration Guide
## OpenClaw Windows 10 AI Agent System

---

## Table of Contents

1. [Overview](#1-overview)
2. [Integration Architecture](#2-integration-architecture)
3. [Component Integration](#3-component-integration)
4. [API Reference](#4-api-reference)
5. [Event System Integration](#5-event-system-integration)
6. [Memory System Integration](#6-memory-system-integration)
7. [External Service Integration](#7-external-service-integration)
8. [Configuration Guide](#8-configuration-guide)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

This guide describes how to integrate the Advanced Planning Loop into the OpenClaw Windows 10 AI Agent system.

### 1.1 Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT CORE SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   PERCEIVE  │  │    PLAN     │  │    ACT      │             │
│  │    LOOP     │◄─┤    LOOP     │◄─┤    LOOP     │             │
│  │             │  │  (ADVANCED) │  │             │             │
│  └─────────────┘  └──────┬──────┘  └─────────────┘             │
│                          │                                      │
│              ┌───────────┼───────────┐                         │
│              ▼           ▼           ▼                         │
│        ┌────────┐  ┌────────┐  ┌────────┐                      │
│        │ MEMORY │  │ EVENTS │  │ CONFIG │                      │
│        │ SYSTEM │  │  BUS   │  │ MANAGER│                      │
│        └────────┘  └────────┘  └────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │  Gmail  │          │ Browser │          │ Twilio  │
   │ Control │          │ Control │          │ Voice   │
   └─────────┘          └─────────┘          └─────────┘
```

---

## 2. Integration Architecture

### 2.1 Core Components

```python
# Main integration class
class PlanningLoopIntegration:
    """Integration layer for the Planning Loop."""
    
    def __init__(self, agent_core):
        self.agent_core = agent_core
        self.planning_loop = None
        self.event_bus = agent_core.event_bus
        self.memory_system = agent_core.memory_system
        self.config_manager = agent_core.config_manager
        
    async def initialize(self):
        """Initialize the planning loop integration."""
        # Load configuration
        config = self._load_config()
        
        # Create planning loop
        self.planning_loop = AdvancedPlanningLoop(config)
        
        # Set up integrations
        await self._setup_event_integration()
        await self._setup_memory_integration()
        await self._setup_service_integration()
        
    def _load_config(self) -> PlanningConfig:
        """Load planning loop configuration."""
        config_data = self.config_manager.get('planning_loop')
        return PlanningConfig(**config_data)
```

### 2.2 Initialization Flow

```python
async def initialize_planning_loop(agent_core):
    """Initialize the planning loop for the agent."""
    
    integration = PlanningLoopIntegration(agent_core)
    await integration.initialize()
    
    # Register with agent core
    agent_core.planning_loop = integration.planning_loop
    
    # Set up event handlers
    await _setup_event_handlers(agent_core, integration)
    
    return integration
```

---

## 3. Component Integration

### 3.1 Perception Loop Integration

```python
async def integrate_with_perception(agent_core, planning_loop):
    """Integrate planning loop with perception."""
    
    async def on_perception_event(event_type, data):
        """Handle perception events."""
        
        if event_type == 'new_task_detected':
            # Create goal from detected task
            goal = HARDGoal(
                goal_id=f"detected_{uuid.uuid4().hex[:8]}",
                name=data['task_name'],
                description=data['task_description'],
                level=GoalLevel.TASK,
                priority=data.get('priority', 50),
                hardness=HardnessLevel.IMPORTANT
            )
            
            # Add to current plan or create new plan
            if planning_loop.current_plan:
                await planning_loop.add_goal_to_plan(goal)
            else:
                await planning_loop.create_plan([goal])
        
        elif event_type == 'context_change':
            # Trigger replanning if significant context change
            await planning_loop.replanning_engine.trigger(
                ReplanningTrigger.CONTEXT_CHANGED,
                {
                    'source': 'perception_loop',
                    'severity': data.get('significance', 5),
                    'changes': data['changes']
                }
            )
    
    # Register with perception loop
    agent_core.perception_loop.on_event('new_task_detected', on_perception_event)
    agent_core.perception_loop.on_event('context_change', on_perception_event)
```

### 3.2 Action Loop Integration

```python
async def integrate_with_action(agent_core, planning_loop):
    """Integrate planning loop with action execution."""
    
    async def execute_goal_action(goal: HARDGoal):
        """Execute a goal using the action loop."""
        
        # Map goal to action sequence
        actions = _goal_to_actions(goal)
        
        # Execute through action loop
        results = []
        for action in actions:
            result = await agent_core.action_loop.execute(action)
            results.append(result)
            
            if not result.success:
                raise ActionExecutionError(f"Action failed: {action.name}")
        
        return results
    
    # Register goal executor
    planning_loop.goal_executor = execute_goal_action
```

### 3.3 Memory System Integration

```python
async def integrate_with_memory(agent_core, planning_loop):
    """Integrate planning loop with memory system."""
    
    async def on_plan_event(event_type, data):
        """Handle plan events for memory storage."""
        
        if event_type == 'plan_created':
            # Store plan in memory
            await agent_core.memory_system.store(
                key=f"plan:{data['plan_id']}",
                value=data,
                category='plans',
                importance=0.8
            )
        
        elif event_type == 'plan_completed':
            # Store execution record
            await agent_core.memory_system.store(
                key=f"execution:{data['execution_id']}",
                value=data,
                category='executions',
                importance=0.7
            )
            
            # Learn from execution
            await _learn_from_execution(data)
    
    # Register event handler
    planning_loop.on_event('plan_created', on_plan_event)
    planning_loop.on_event('plan_completed', on_plan_event)
    
    async def retrieve_similar_plan(goal: HARDGoal) -> Optional[ExecutionPlan]:
        """Retrieve similar past plans for learning."""
        
        similar = await agent_core.memory_system.search(
            query=goal.description,
            category='plans',
            limit=5
        )
        
        if similar:
            # Return most similar plan
            best = similar[0]
            return ExecutionPlan.from_dict(best.value)
        
        return None
    
    # Expose retrieval function
    planning_loop.retrieve_similar_plan = retrieve_similar_plan
```

---

## 4. API Reference

### 4.1 Planning Loop API

```python
class PlanningLoopAPI:
    """Public API for the planning loop."""
    
    # Plan Creation
    async def create_plan(
        self,
        goals: List[HARDGoal],
        deadline: Optional[datetime] = None,
        priorities: Optional[Dict[str, int]] = None,
        constraints: Optional[Dict] = None
    ) -> ExecutionPlan:
        """Create a new execution plan."""
        pass
    
    # Plan Execution
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        async_execution: bool = True
    ) -> Union[ExecutionResult, AsyncIterator[ExecutionUpdate]]:
        """Execute a plan."""
        pass
    
    # Replanning
    async def replan(
        self,
        reason: str,
        preserve_completed: bool = True
    ) -> ExecutionPlan:
        """Trigger replanning."""
        pass
    
    # Goal Management
    async def add_goal(
        self,
        goal: HARDGoal,
        priority: int = 50
    ) -> ExecutionPlan:
        """Add a new goal to the current plan."""
        pass
    
    async def cancel_goal(
        self,
        goal_id: str,
        reason: str
    ) -> bool:
        """Cancel a goal in the current plan."""
        pass
    
    # Status and Metrics
    async def get_plan_status(self) -> Dict:
        """Get the status of the current plan."""
        pass
    
    async def get_execution_metrics(self) -> Dict:
        """Get metrics for the current or last execution."""
        pass
    
    # Quality Assessment
    async def assess_plan_quality(
        self,
        plan: Optional[ExecutionPlan] = None
    ) -> QualityAssessment:
        """Assess the quality of a plan."""
        pass
```

### 4.2 Goal Creation API

```python
# Helper functions for creating goals

def create_mission_goal(
    name: str,
    description: str,
    priority: int = 50
) -> HARDGoal:
    """Create a mission-level goal."""
    return HARDGoal(
        goal_id=f"mission_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        level=GoalLevel.MISSION,
        hardness=HardnessLevel.CRITICAL,
        priority=priority
    )

def create_task_goal(
    name: str,
    description: str,
    priority: int = 50,
    depends_on: List[str] = None
) -> HARDGoal:
    """Create a task-level goal."""
    return HARDGoal(
        goal_id=f"task_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        level=GoalLevel.TASK,
        hardness=HardnessLevel.IMPORTANT,
        priority=priority,
        depends_on=depends_on or []
    )

def create_action_goal(
    name: str,
    description: str,
    action_type: str,
    parameters: Dict = None
) -> HARDGoal:
    """Create an action-level goal."""
    return HARDGoal(
        goal_id=f"action_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        level=GoalLevel.ACTION,
        hardness=HardnessLevel.IMPORTANT,
        priority=50,
        context={'action_type': action_type, 'parameters': parameters or {}}
    )
```

---

## 5. Event System Integration

### 5.1 Event Types

```python
# Planning Loop Events

PLANNING_EVENTS = {
    # Plan Lifecycle
    'plan.created': 'New plan created',
    'plan.started': 'Plan execution started',
    'plan.completed': 'Plan execution completed',
    'plan.failed': 'Plan execution failed',
    'plan.cancelled': 'Plan execution cancelled',
    
    # Goal Lifecycle
    'goal.created': 'New goal created',
    'goal.started': 'Goal execution started',
    'goal.completed': 'Goal execution completed',
    'goal.failed': 'Goal execution failed',
    'goal.cancelled': 'Goal execution cancelled',
    
    # Replanning
    'replanning.triggered': 'Replanning triggered',
    'replanning.started': 'Replanning started',
    'replanning.completed': 'Replanning completed',
    'replanning.failed': 'Replanning failed',
    
    # Contingency
    'contingency.activated': 'Contingency plan activated',
    'contingency.executed': 'Contingency plan executed',
    
    # Quality
    'quality.assessed': 'Plan quality assessed',
    'quality.degraded': 'Plan quality degraded',
    
    # Deadlines
    'deadline.warning': 'Deadline warning',
    'deadline.missed': 'Deadline missed',
    'deadline.approaching': 'Deadline approaching'
}
```

### 5.2 Event Handler Registration

```python
async def setup_event_handlers(agent_core, planning_loop):
    """Set up event handlers for the planning loop."""
    
    # Plan events
    planning_loop.on_event('plan_created', async def(event_type, data):
        await agent_core.event_bus.emit('planning.plan.created', data)
        await agent_core.tts.speak(f"Plan created with {data['goal_count']} goals")
    )
    
    planning_loop.on_event('plan_completed', async def(event_type, data):
        await agent_core.event_bus.emit('planning.plan.completed', data)
        
        if data['failed_goals'] > 0:
            await agent_core.tts.speak(
                f"Plan completed with {data['failed_goals']} failures"
            )
        else:
            await agent_core.tts.speak("Plan completed successfully")
    )
    
    # Goal events
    planning_loop.on_event('goal_failed', async def(event_type, data):
        await agent_core.event_bus.emit('planning.goal.failed', data)
        
        # Send notification for critical failures
        if data.get('hardness') == 'CRITICAL':
            await agent_core.gmail.send(
                to=agent_core.config.admin_email,
                subject="Critical Goal Failed",
                body=f"Goal {data['goal_id']} failed with error: {data.get('error')}"
            )
    )
    
    # Replanning events
    planning_loop.on_event('replanning_triggered', async def(event_type, data):
        await agent_core.event_bus.emit('planning.replanning.triggered', data)
        logger.info(f"Replanning triggered: {data['trigger']}")
    )
```

---

## 6. Memory System Integration

### 6.1 Learning from Execution

```python
async def learn_from_execution(agent_core, execution_result: ExecutionResult):
    """Learn from plan execution outcomes."""
    
    memory = agent_core.memory_system
    
    # Store execution pattern
    await memory.store(
        key=f"execution_pattern:{execution_result.execution_id}",
        value={
            'plan_structure': execution_result.plan_structure,
            'success_rate': execution_result.success_rate,
            'duration': execution_result.duration.total_seconds(),
            'failed_goals': execution_result.failed_goals,
            'context': execution_result.context
        },
        category='execution_patterns',
        importance=0.7
    )
    
    # Update goal success rates
    for goal_id in execution_result.completed_goals:
        await _update_goal_success_rate(memory, goal_id, True)
    
    for goal_id in execution_result.failed_goals:
        await _update_goal_success_rate(memory, goal_id, False)
    
    # Learn failure patterns
    if execution_result.failed_goals:
        await _learn_failure_patterns(memory, execution_result)

async def _update_goal_success_rate(memory, goal_id: str, success: bool):
    """Update success rate for a goal type."""
    
    key = f"goal_stats:{goal_id}"
    stats = await memory.retrieve(key) or {'attempts': 0, 'successes': 0}
    
    stats['attempts'] += 1
    if success:
        stats['successes'] += 1
    
    stats['success_rate'] = stats['successes'] / stats['attempts']
    
    await memory.store(key, stats, category='goal_statistics')
```

### 6.2 Pattern Recognition

```python
async def recognize_patterns(agent_core, planning_loop):
    """Recognize patterns from execution history."""
    
    memory = agent_core.memory_system
    
    # Retrieve execution patterns
    patterns = await memory.search(
        category='execution_patterns',
        limit=100
    )
    
    # Analyze for common failure patterns
    failure_patterns = defaultdict(int)
    
    for pattern in patterns:
        if pattern.value.get('failed_goals'):
            for goal_id in pattern.value['failed_goals']:
                failure_patterns[goal_id] += 1
    
    # Update planning loop with recognized patterns
    for goal_id, count in failure_patterns.items():
        if count > 3:  # Threshold for pattern recognition
            planning_loop.recognized_patterns[goal_id] = {
                'failure_count': count,
                'pattern_type': 'recurring_failure'
            }
```

---

## 7. External Service Integration

### 7.1 Gmail Integration

```python
async def integrate_gmail(agent_core, planning_loop):
    """Integrate Gmail with planning loop."""
    
    async def check_email_goal(goal: HARDGoal):
        """Execute email checking goal."""
        
        gmail = agent_core.gmail
        
        # Check for new emails
        emails = await gmail.get_unread_emails()
        
        # Process high-priority emails
        for email in emails:
            if email.priority == 'high':
                # Create sub-goal for response
                response_goal = create_task_goal(
                    name=f"Respond to: {email.subject}",
                    description=f"Generate response to email from {email.sender}",
                    priority=80
                )
                
                await planning_loop.add_goal_to_plan(response_goal)
        
        return {'emails_checked': len(emails), 'high_priority': len([e for e in emails if e.priority == 'high'])}
    
    # Register Gmail-specific goal handler
    planning_loop.register_goal_handler('check_email', check_email_goal)
```

### 7.2 Browser Integration

```python
async def integrate_browser(agent_core, planning_loop):
    """Integrate browser control with planning loop."""
    
    async def browser_action_goal(goal: HARDGoal):
        """Execute browser action goal."""
        
        browser = agent_core.browser
        action_type = goal.context.get('action_type')
        params = goal.context.get('parameters', {})
        
        if action_type == 'navigate':
            result = await browser.navigate(params['url'])
        elif action_type == 'search':
            result = await browser.search(params['query'])
        elif action_type == 'extract':
            result = await browser.extract_data(params['selector'])
        else:
            raise ValueError(f"Unknown browser action: {action_type}")
        
        return result
    
    # Register browser goal handler
    planning_loop.register_goal_handler('browser_action', browser_action_goal)
```

### 7.3 Twilio Integration

```python
async def integrate_twilio(agent_core, planning_loop):
    """Integrate Twilio with planning loop."""
    
    async def make_call_goal(goal: HARDGoal):
        """Execute voice call goal."""
        
        twilio = agent_core.twilio
        params = goal.context.get('parameters', {})
        
        # Make call
        call = await twilio.make_call(
            to=params['phone_number'],
            message=params['message']
        )
        
        return {'call_sid': call.sid, 'status': call.status}
    
    async def send_sms_goal(goal: HARDGoal):
        """Execute SMS goal."""
        
        twilio = agent_core.twilio
        params = goal.context.get('parameters', {})
        
        message = await twilio.send_sms(
            to=params['phone_number'],
            body=params['message']
        )
        
        return {'message_sid': message.sid, 'status': message.status}
    
    # Register Twilio goal handlers
    planning_loop.register_goal_handler('make_call', make_call_goal)
    planning_loop.register_goal_handler('send_sms', send_sms_goal)
```

---

## 8. Configuration Guide

### 8.1 Basic Configuration

```yaml
# planning_loop.yaml

planning_loop:
  # Enable/disable features
  replanning:
    enabled: true
    cooldown_seconds: 30
  
  contingency:
    enabled: true
    max_contingencies_per_plan: 5
  
  quality:
    enabled: true
    min_quality_threshold: 0.6
  
  # Resource limits
  resources:
    max_cpu_cores: 4
    max_memory_mb: 4096
  
  # Objectives
  objectives:
    time:
      weight: 0.3
    quality:
      weight: 0.3
    reliability:
      weight: 0.4
```

### 8.2 Environment-Specific Configuration

```python
# config_loader.py

def load_config(environment: str) -> PlanningConfig:
    """Load configuration for specific environment."""
    
    base_config = load_yaml('planning_loop.yaml')
    
    # Environment-specific overrides
    overrides = {
        'development': {
            'replanning': {'cooldown_seconds': 10},
            'quality': {'min_quality_threshold': 0.4}
        },
        'production': {
            'replanning': {'cooldown_seconds': 60},
            'quality': {'min_quality_threshold': 0.7},
            'contingency': {'enabled': True}
        }
    }
    
    # Apply overrides
    if environment in overrides:
        deep_merge(base_config, overrides[environment])
    
    return PlanningConfig(**base_config['planning_loop'])
```

---

## 9. Best Practices

### 9.1 Goal Design

```python
# Best practices for goal design

# 1. Make goals specific and measurable
good_goal = HARDGoal(
    goal_id="email_001",
    name="Check Gmail inbox",
    description="Check Gmail inbox for unread messages and return count",
    level=GoalLevel.TASK,
    success_criteria=[
        SuccessCriterion(
            name="inbox_accessible",
            description="Successfully accessed Gmail inbox"
        ),
        SuccessCriterion(
            name="count_obtained",
            description="Obtained count of unread messages"
        )
    ]
)

# 2. Set realistic deadlines
realistic_goal = HARDGoal(
    goal_id="report_001",
    name="Generate daily report",
    description="Generate and send daily activity report",
    deadline=datetime.now() + timedelta(hours=1),  # Realistic deadline
    estimated_duration=timedelta(minutes=30)  # Accurate estimate
)

# 3. Define clear dependencies
dependent_goal = HARDGoal(
    goal_id="response_001",
    name="Send email response",
    description="Send response to received email",
    depends_on=["email_001"],  # Clear dependency
    hardness=HardnessLevel.IMPORTANT
)
```

### 9.2 Error Handling

```python
# Robust error handling in goal execution

async def robust_goal_executor(goal: HARDGoal):
    """Execute goal with robust error handling."""
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            result = await execute_goal(goal)
            return result
            
        except TransientError as e:
            # Retry on transient errors
            if attempt < max_retries - 1:
                logger.warning(f"Transient error, retrying: {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise
                
        except PermanentError as e:
            # Don't retry permanent errors
            logger.error(f"Permanent error: {e}")
            raise
            
        except Exception as e:
            # Unexpected error
            logger.exception(f"Unexpected error executing goal: {e}")
            raise
```

### 9.3 Monitoring and Observability

```python
# Comprehensive monitoring

class PlanningLoopMonitor:
    """Monitor planning loop health and performance."""
    
    def __init__(self, planning_loop):
        self.planning_loop = planning_loop
        self.metrics = defaultdict(list)
    
    async def collect_metrics(self):
        """Collect planning loop metrics."""
        
        metrics = {
            'active_plans': len(self.planning_loop.active_plans),
            'pending_goals': self._count_pending_goals(),
            'success_rate': self._calculate_success_rate(),
            'avg_execution_time': self._calculate_avg_execution_time(),
            'replanning_frequency': self._calculate_replanning_frequency()
        }
        
        for key, value in metrics.items():
            self.metrics[key].append({
                'timestamp': datetime.now(),
                'value': value
            })
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate monitoring report."""
        
        report = []
        report.append("# Planning Loop Monitoring Report")
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        
        for metric_name, values in self.metrics.items():
            if values:
                latest = values[-1]['value']
                report.append(f"## {metric_name}")
                report.append(f"Current: {latest}")
                report.append(f"History: {len(values)} data points")
                report.append("")
        
        return "\n".join(report)
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue: Replanning Loop

```python
# Problem: Continuous replanning
# Solution: Increase cooldown and add circuit breaker

config = PlanningConfig(
    replanning_cooldown_seconds=120,  # Increase cooldown
    max_replans_per_execution=5       # Limit replans
)

# Add circuit breaker
class ReplanningCircuitBreaker:
    """Prevent excessive replanning."""
    
    def __init__(self, max_failures=5, reset_timeout=300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.is_open = False
    
    def record_failure(self):
        """Record a replanning failure."""
        self.failures += 1
        self.last_failure = datetime.now()
        
        if self.failures >= self.max_failures:
            self.is_open = True
            logger.error("Replanning circuit breaker opened")
    
    def can_replan(self) -> bool:
        """Check if replanning is allowed."""
        
        if not self.is_open:
            return True
        
        # Check if timeout has passed
        if self.last_failure:
            elapsed = (datetime.now() - self.last_failure).total_seconds()
            if elapsed > self.reset_timeout:
                self.is_open = False
                self.failures = 0
                return True
        
        return False
```

#### Issue: Resource Exhaustion

```python
# Problem: Plans require more resources than available
# Solution: Resource-constrained planning

async def handle_resource_exhaustion(planning_loop, goals):
    """Handle resource exhaustion gracefully."""
    
    # Prioritize goals
    prioritized = sorted(goals, key=lambda g: g.priority, reverse=True)
    
    # Take only what we can handle
    resource_limit = get_available_resources()
    selected_goals = []
    total_resources = ResourceProfile()
    
    for goal in prioritized:
        needed = goal.resource_profile
        
        if can_fit(total_resources, needed, resource_limit):
            selected_goals.append(goal)
            total_resources = add_resources(total_resources, needed)
        else:
            # Defer lower priority goals
            await defer_goal(goal)
    
    return selected_goals
```

#### Issue: Deadline Misses

```python
# Problem: Frequent deadline misses
# Solution: Better deadline estimation and monitoring

class DeadlineManager:
    """Manage deadlines to prevent misses."""
    
    def __init__(self):
        self.deadline_history = []
        self.estimation_bias = 1.0
    
    def estimate_duration(self, goal: HARDGoal) -> timedelta:
        """Estimate duration with safety margin."""
        
        base_estimate = goal.estimated_duration
        
        # Apply learned bias
        adjusted = base_estimate * self.estimation_bias
        
        # Add safety margin based on hardness
        if goal.hardness == HardnessLevel.CRITICAL:
            safety_margin = 1.5  # 50% buffer
        elif goal.hardness == HardnessLevel.IMPORTANT:
            safety_margin = 1.3  # 30% buffer
        else:
            safety_margin = 1.1  # 10% buffer
        
        return adjusted * safety_margin
    
    def record_actual_duration(self, goal: HARDGoal, actual: timedelta):
        """Record actual duration for learning."""
        
        self.deadline_history.append({
            'goal_id': goal.goal_id,
            'estimated': goal.estimated_duration,
            'actual': actual
        })
        
        # Update bias
        if len(self.deadline_history) > 10:
            self._update_estimation_bias()
    
    def _update_estimation_bias(self):
        """Update estimation bias based on history."""
        
        recent = self.deadline_history[-10:]
        
        total_estimated = sum(d['estimated'].total_seconds() for d in recent)
        total_actual = sum(d['actual'].total_seconds() for d in recent)
        
        if total_estimated > 0:
            self.estimation_bias = total_actual / total_estimated
```

### 10.2 Debug Mode

```python
# Enable debug mode for troubleshooting

DEBUG_CONFIG = {
    'planning_loop': {
        'logging': {
            'level': 'DEBUG',
            'log_all_decisions': True,
            'log_all_transitions': True
        },
        'replanning': {
            'log_all_triggers': True,
            'log_strategy_selection': True
        },
        'quality': {
            'log_all_assessments': True,
            'log_dimension_scores': True
        }
    }
}

async def run_with_debug(planning_loop, goals):
    """Run planning loop with debug logging."""
    
    # Enable debug logging
    logging.getLogger('planning_loop').setLevel(logging.DEBUG)
    
    # Create plan with detailed logging
    plan = await planning_loop.create_plan(goals)
    
    logger.debug(f"Plan structure: {plan.to_dict()}")
    logger.debug(f"Execution order: {plan.get_execution_order()}")
    logger.debug(f"Quality assessment: {await planning_loop.assess_plan_quality(plan)}")
    
    return plan
```

---

## Summary

This integration guide provides comprehensive instructions for integrating the Advanced Planning Loop into the OpenClaw Windows 10 AI Agent system. Key integration points include:

1. **Perception Loop**: Automatic goal creation from detected tasks
2. **Action Loop**: Goal execution through action sequences
3. **Memory System**: Learning from execution history
4. **Event System**: Event-driven coordination
5. **External Services**: Gmail, Browser, Twilio integration

For additional support, refer to the technical specification document and implementation code.

---

*Version: 1.0*  
*Last Updated: 2025*
