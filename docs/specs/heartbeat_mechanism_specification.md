# HEARTBEAT MECHANISM & AUTONOMOUS ACTIVATION SYSTEM
## Technical Specification for Windows 10 OpenClaw-Inspired AI Agent

---

## 1. EXECUTIVE SUMMARY

The Heartbeat Mechanism is the core autonomous activation system that enables the AI agent to operate proactively without explicit user prompts. Inspired by OpenClaw's "periodic autonomy" primitive, this system allows the agent to:

- **Self-awaken** on scheduled intervals or triggered events
- **Monitor** multiple data sources continuously (Gmail, files, system state, web)
- **Decide** autonomously when action is required
- **Execute** background tasks without user intervention
- **Maintain** persistent operational state across wake cycles

### Key Concepts from OpenClaw Research

OpenClaw's architecture is built on four primitives:
1. **Persistent identity** - Agent knows who it is across sessions (SOUL.md)
2. **Periodic autonomy** - Agent wakes up and acts without being asked (heartbeat)
3. **Accumulated memory** - Agent remembers what happened before (memory files)
4. **Social context** - Agent can find and interact with other agents

The heartbeat is described as "the ability to monitor situations and act without being explicitly prompted" - like JARVIS running background processes.

---

## 2. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEARTBEAT ORCHESTRATION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   TRIGGER    │  │   TRIGGER    │  │   TRIGGER    │  │   TRIGGER    │    │
│  │   ENGINE     │  │   ENGINE     │  │   ENGINE     │  │   ENGINE     │    │
│  │  (Cron/Time) │  │  (Event/IO)  │  │ (Webhooks)   │  │  (Voice)     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         └─────────────────┴────────┬────────┴─────────────────┘             │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              HEARTBEAT DISPATCHER & ROUTER                          │   │
│  │         (Session Isolation + Priority Queue)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │  BACKGROUND │           │   MAIN      │           │  ISOLATED   │       │
│  │   WORKER    │           │   SESSION   │           │   DOCKER    │       │
│  │   POOL      │           │   (User)    │           │  CONTAINER  │       │
│  └──────┬──────┘           └──────┬──────┘           └──────┬──────┘       │
│         │                         │                         │              │
│         └─────────────────────────┼─────────────────────────┘              │
│                                   ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              AUTONOMOUS ACTION SELECTOR (AAS)                       │   │
│  │    (Context + Memory + Goals → Action Decision Engine)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                        │
│                                   ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              SKILL EXECUTION ENGINE (15 Hardcoded Loops)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. HEARTBEAT SCHEDULING AND TRIGGERS

### 3.1 Trigger Types

| Trigger Type | Description | Use Cases | Priority |
|--------------|-------------|-----------|----------|
| **Temporal** | Time-based cron expressions | Daily reports, hourly checks | Configurable |
| **Event-Driven** | File system, I/O, registry changes | New email, file created | High |
| **Webhook** | HTTP callbacks from external services | GitHub, Twilio, custom APIs | High |
| **Voice Wake** | STT keyword detection | "Hey Agent" activation | Critical |
| **System State** | CPU, memory, disk thresholds | Resource monitoring | Medium |
| **Network** | Connectivity changes, API status | Server monitoring | Medium |
| **Gmail Push** | Gmail API push notifications | New email arrival | High |
| **SMS/Voice** | Twilio incoming messages | Emergency alerts | Critical |

### 3.2 Trigger Configuration Schema

```python
# trigger_config.py
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List, Callable
from datetime import datetime, timedelta

class TriggerType(Enum):
    CRON = auto()
    INTERVAL = auto()
    EVENT = auto()
    WEBHOOK = auto()
    VOICE_WAKE = auto()
    SYSTEM_STATE = auto()
    NETWORK = auto()
    GMAIL_PUSH = auto()
    TWILIO_SMS = auto()
    TWILIO_VOICE = auto()

class TriggerPriority(Enum):
    CRITICAL = 1    # Immediate execution, interrupts current task
    HIGH = 2        # Queue at front, execute ASAP
    MEDIUM = 3      # Normal queue position
    LOW = 4         # Background execution when idle
    BACKGROUND = 5  # Only when system fully idle

@dataclass
class TriggerConfig:
    trigger_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    enabled: bool = True
    
    # Temporal triggers
    cron_expression: Optional[str] = None  # "0 9 * * *" = 9 AM daily
    interval_seconds: Optional[int] = None  # 300 = every 5 minutes
    
    # Event triggers
    watch_paths: Optional[List[str]] = None
    watch_patterns: Optional[List[str]] = None  # ["*.pdf", "*.docx"]
    
    # System state triggers
    cpu_threshold: Optional[float] = None  # 80.0 = 80% CPU
    memory_threshold: Optional[float] = None
    disk_threshold: Optional[float] = None
    
    # Network triggers
    endpoint_url: Optional[str] = None
    expected_status: Optional[int] = 200
    
    # Execution config
    max_concurrent: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    
    # Session isolation
    session_namespace: str = "background"
    isolated_execution: bool = False
    
    # Action binding
    action_handler: Optional[str] = None  # Name of registered action
    action_params: Optional[Dict] = None
```

### 3.3 Default Trigger Registry

```python
# default_triggers.py
DEFAULT_TRIGGERS = [
    # === CRITICAL PRIORITY ===
    TriggerConfig(
        trigger_id="voice_wake",
        trigger_type=TriggerType.VOICE_WAKE,
        priority=TriggerPriority.CRITICAL,
        wake_words=["hey agent", "agent wake", "jarvis"],
        session_namespace="main",
        timeout_seconds=30
    ),
    
    TriggerConfig(
        trigger_id="twilio_emergency",
        trigger_type=TriggerType.TWILIO_SMS,
        priority=TriggerPriority.CRITICAL,
        keywords=["urgent", "emergency", "alert", "critical"],
        session_namespace="main",
        timeout_seconds=60
    ),
    
    # === HIGH PRIORITY ===
    TriggerConfig(
        trigger_id="gmail_inbox_monitor",
        trigger_type=TriggerType.GMAIL_PUSH,
        priority=TriggerPriority.HIGH,
        watch_labels=["INBOX", "IMPORTANT"],
        from_addresses=[],  # Empty = all
        subject_keywords=["urgent", "action required", "deadline"],
        session_namespace="background",
        action_handler="process_incoming_email"
    ),
    
    TriggerConfig(
        trigger_id="downloads_watcher",
        trigger_type=TriggerType.EVENT,
        priority=TriggerPriority.HIGH,
        watch_paths=["~/Downloads"],
        watch_patterns=["*.pdf", "*.zip", "*.exe"],
        session_namespace="background",
        action_handler="process_new_download"
    ),
    
    # === MEDIUM PRIORITY (Heartbeat Core) ===
    TriggerConfig(
        trigger_id="heartbeat_5min",
        trigger_type=TriggerType.INTERVAL,
        priority=TriggerPriority.MEDIUM,
        interval_seconds=300,  # 5 minutes
        session_namespace="background",
        action_handler="heartbeat_cycle"
    ),
    
    TriggerConfig(
        trigger_id="system_health_check",
        trigger_type=TriggerType.INTERVAL,
        priority=TriggerPriority.MEDIUM,
        interval_seconds=60,  # 1 minute
        session_namespace="background",
        action_handler="check_system_health"
    ),
    
    TriggerConfig(
        trigger_id="website_monitor",
        trigger_type=TriggerType.INTERVAL,
        priority=TriggerPriority.MEDIUM,
        interval_seconds=300,
        endpoint_url="https://example.com/health",
        session_namespace="background",
        action_handler="monitor_website"
    ),
    
    # === LOW PRIORITY ===
    TriggerConfig(
        trigger_id="daily_summary",
        trigger_type=TriggerType.CRON,
        priority=TriggerPriority.LOW,
        cron_expression="0 8 * * *",  # 8 AM daily
        session_namespace="background",
        action_handler="generate_daily_summary"
    ),
    
    TriggerConfig(
        trigger_id="weekly_report",
        trigger_type=TriggerType.CRON,
        priority=TriggerPriority.LOW,
        cron_expression="0 9 * * 1",  # 9 AM Monday
        session_namespace="background",
        action_handler="generate_weekly_report"
    ),
    
    TriggerConfig(
        trigger_id="memory_compaction",
        trigger_type=TriggerType.CRON,
        priority=TriggerPriority.LOW,
        cron_expression="0 2 * * *",  # 2 AM daily
        session_namespace="background",
        action_handler="compact_memory"
    ),
    
    # === BACKGROUND PRIORITY ===
    TriggerConfig(
        trigger_id="idle_cleanup",
        trigger_type=TriggerType.SYSTEM_STATE,
        priority=TriggerPriority.BACKGROUND,
        idle_threshold_seconds=300,  # After 5 min idle
        session_namespace="background",
        action_handler="cleanup_temp_files"
    ),
]
```

---

## 4. PERIODIC WAKE-UP IMPLEMENTATION

### 4.1 Heartbeat Core Engine

```python
# heartbeat_engine.py
import asyncio
import threading
from queue import PriorityQueue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging

class HeartbeatEngine:
    """
    Core heartbeat orchestration engine.
    Manages all trigger types, dispatches wake events, and maintains
    persistent operational state.
    """
    
    def __init__(self, config_path: str = "config/heartbeat.yaml"):
        self.config = self._load_config(config_path)
        self.triggers: Dict[str, TriggerConfig] = {}
        self.active_timers: Dict[str, asyncio.Task] = {}
        self.event_watchers: Dict[str, threading.Thread] = {}
        self.dispatch_queue = PriorityQueue()
        self.running = False
        self.session_manager = SessionManager()
        self.action_selector = AutonomousActionSelector()
        self.logger = logging.getLogger("heartbeat")
        
        # Heartbeat statistics
        self.stats = {
            "total_wakeups": 0,
            "actions_executed": 0,
            "actions_failed": 0,
            "last_wakeup": None,
            "avg_wake_duration": 0.0
        }
    
    async def start(self):
        """Initialize and start all heartbeat mechanisms."""
        self.logger.info("Starting Heartbeat Engine...")
        self.running = True
        
        # Register all triggers
        await self._register_triggers()
        
        # Start temporal triggers (cron/interval)
        await self._start_temporal_triggers()
        
        # Start event watchers
        await self._start_event_watchers()
        
        # Start webhook server
        await self._start_webhook_server()
        
        # Start voice wake detection
        await self._start_voice_wake()
        
        # Start Gmail push notifications
        await self._start_gmail_push()
        
        # Start Twilio listeners
        await self._start_twilio_listeners()
        
        # Start main dispatch loop
        await self._dispatch_loop()
    
    async def _register_triggers(self):
        """Load and register all trigger configurations."""
        for trigger_config in DEFAULT_TRIGGERS:
            self.triggers[trigger_config.trigger_id] = trigger_config
            self.logger.info(f"Registered trigger: {trigger_config.trigger_id}")
    
    async def _start_temporal_triggers(self):
        """Start cron and interval-based triggers."""
        for trigger_id, config in self.triggers.items():
            if config.trigger_type in [TriggerType.CRON, TriggerType.INTERVAL]:
                task = asyncio.create_task(
                    self._temporal_trigger_loop(trigger_id, config)
                )
                self.active_timers[trigger_id] = task
    
    async def _temporal_trigger_loop(self, trigger_id: str, config: TriggerConfig):
        """Execute trigger on schedule."""
        while self.running:
            try:
                # Calculate next execution time
                if config.trigger_type == TriggerType.INTERVAL:
                    await asyncio.sleep(config.interval_seconds)
                elif config.trigger_type == TriggerType.CRON:
                    next_run = self._get_next_cron_execution(config.cron_expression)
                    wait_seconds = (next_run - datetime.now()).total_seconds()
                    await asyncio.sleep(max(0, wait_seconds))
                
                # Dispatch wake event
                await self._dispatch_wake(trigger_id, config)
                
            except Exception as e:
                self.logger.error(f"Trigger {trigger_id} error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _dispatch_wake(self, trigger_id: str, config: TriggerConfig):
        """Dispatch a wake event to the appropriate handler."""
        wake_event = WakeEvent(
            trigger_id=trigger_id,
            trigger_type=config.trigger_type,
            priority=config.priority,
            timestamp=datetime.now(),
            session_namespace=config.session_namespace,
            action_handler=config.action_handler,
            action_params=config.action_params
        )
        
        # Add to priority queue
        self.dispatch_queue.put((
            config.priority.value,
            wake_event.timestamp,
            wake_event
        ))
        
        self.stats["total_wakeups"] += 1
        self.stats["last_wakeup"] = datetime.now()
        
        self.logger.info(f"HEARTBEAT: {trigger_id} (priority: {config.priority.name})")
    
    async def _dispatch_loop(self):
        """Main dispatch loop processing wake events."""
        while self.running:
            try:
                if not self.dispatch_queue.empty():
                    _, _, wake_event = self.dispatch_queue.get()
                    await self._process_wake_event(wake_event)
                else:
                    await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Dispatch loop error: {e}")
    
    async def _process_wake_event(self, wake_event: 'WakeEvent'):
        """Process a single wake event."""
        start_time = datetime.now()
        
        try:
            # Get or create session
            session = self.session_manager.get_or_create(
                namespace=wake_event.session_namespace,
                trigger_id=wake_event.trigger_id
            )
            
            # Load agent identity (SOUL.md)
            await session.load_identity()
            
            # Load relevant memory
            await session.load_context_memory()
            
            # Determine autonomous action
            action = await self.action_selector.select_action(
                wake_event=wake_event,
                session=session
            )
            
            if action:
                # Execute action
                result = await self._execute_action(action, session)
                
                # Update memory with results
                await session.update_memory(result)
                
                self.stats["actions_executed"] += 1
                
                # Log completion
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(
                    f"Action completed: {action.name} in {duration:.2f}s"
                )
            
        except Exception as e:
            self.stats["actions_failed"] += 1
            self.logger.error(f"Wake event processing failed: {e}")
        
        finally:
            # Update average wake duration
            duration = (datetime.now() - start_time).total_seconds()
            self._update_avg_duration(duration)
    
    def _update_avg_duration(self, duration: float):
        """Update running average of wake duration."""
        n = self.stats["total_wakeups"]
        current_avg = self.stats["avg_wake_duration"]
        self.stats["avg_wake_duration"] = (
            (current_avg * (n - 1) + duration) / n
        )
```

### 4.2 Windows-Specific Wake Implementation

```python
# windows_wake.py
import ctypes
from ctypes import wintypes
import win32event
import win32api
import win32con

class WindowsWakeManager:
    """
    Windows-specific wake and power management.
    Ensures agent can wake from sleep/hibernation.
    """
    
    def __init__(self):
        self.wake_timer_handle = None
        self.power_notify_handle = None
    
    def set_wake_timer(self, wake_time: datetime):
        """Set a timer to wake the system from sleep."""
        # Convert to FILETIME
        filetime = self._datetime_to_filetime(wake_time)
        
        # Create waitable timer
        self.wake_timer_handle = win32event.CreateWaitableTimer(
            None,  # Security attributes
            False,  # Auto-reset
            "AgentWakeTimer"  # Timer name
        )
        
        # Set timer with wake capability
        win32event.SetWaitableTimer(
            self.wake_timer_handle,
            filetime,
            0,  # No period (one-shot)
            None,  # No completion routine
            None,  # No completion routine arg
            True   # Resume from suspend (WAKE!)
        )
    
    def register_power_notifications(self, callback: Callable):
        """Register for power state change notifications."""
        # Register for power events
        self.power_notify_handle = win32api.RegisterPowerSettingNotification(
            self._power_callback,
            win32con.GUID_POWER_SAVING_STATUS,
            win32con.DEVICE_NOTIFY_WINDOW_HANDLE
        )
    
    def prevent_sleep_during_task(self, duration_seconds: int):
        """Prevent system sleep during critical task execution."""
        # Use SetThreadExecutionState to prevent sleep
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        
        # Schedule reset after duration
        threading.Timer(duration_seconds, self._reset_execution_state).start()
    
    def _reset_execution_state(self):
        """Reset execution state to allow sleep."""
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
```

---

## 5. AUTONOMOUS ACTION SELECTION

### 5.1 Action Selection Engine

```python
# action_selector.py
from enum import Enum, auto
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

class ActionType(Enum):
    # Communication Actions
    SEND_EMAIL = auto()
    SEND_SMS = auto()
    MAKE_CALL = auto()
    SEND_NOTIFICATION = auto()
    
    # System Actions
    EXECUTE_COMMAND = auto()
    MANAGE_FILES = auto()
    MONITOR_SYSTEM = auto()
    
    # Web Actions
    BROWSE_WEB = auto()
    SCRAPE_DATA = auto()
    API_CALL = auto()
    
    # Content Actions
    GENERATE_SUMMARY = auto()
    CREATE_DOCUMENT = auto()
    UPDATE_MEMORY = auto()
    
    # Monitoring Actions
    CHECK_EMAIL = auto()
    CHECK_CALENDAR = auto()
    CHECK_DEADLINES = auto()
    MONITOR_FILES = auto()

@dataclass
class Action:
    action_type: ActionType
    name: str
    description: str
    priority: int  # 1-10
    required_skills: List[str]
    params: Dict[str, Any]
    estimated_duration: int  # seconds
    can_interrupt: bool = False

class AutonomousActionSelector:
    """
    Decides what action to take based on wake context, memory, and goals.
    Uses GPT-5.2 for complex decision making.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.action_registry = self._build_action_registry()
        self.decision_history = []
    
    async def select_action(
        self,
        wake_event: WakeEvent,
        session: Session
    ) -> Optional[Action]:
        """
        Select the most appropriate autonomous action based on context.
        """
        # Build decision context
        context = {
            "trigger": {
                "id": wake_event.trigger_id,
                "type": wake_event.trigger_type.name,
                "priority": wake_event.priority.name,
                "timestamp": wake_event.timestamp.isoformat()
            },
            "session": {
                "namespace": session.namespace,
                "identity_loaded": session.identity is not None,
                "recent_memories": session.get_recent_memories(5)
            },
            "system_state": await self._get_system_state(),
            "user_context": await self._get_user_context(),
            "pending_tasks": await self._get_pending_tasks(),
            "time_context": {
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "is_business_hours": 9 <= datetime.now().hour < 17
            }
        }
        
        # Use LLM for complex decisions
        if wake_event.action_handler:
            # Pre-defined action handler
            return self._get_action_by_handler(wake_event.action_handler, wake_event.action_params)
        
        # Autonomous decision making
        decision_prompt = self._build_decision_prompt(context)
        
        response = await self.llm.complete(
            prompt=decision_prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse decision
        action = self._parse_decision(response, context)
        
        # Log decision
        self.decision_history.append({
            "timestamp": datetime.now(),
            "trigger": wake_event.trigger_id,
            "selected_action": action.name if action else None,
            "context_hash": hash(json.dumps(context, default=str))
        })
        
        return action
    
    def _build_decision_prompt(self, context: Dict) -> str:
        """Build the decision prompt for the LLM."""
        return f"""
You are an autonomous AI agent that has just woken up due to: {context['trigger']['id']}

CURRENT CONTEXT:
- Time: {context['time_context']['hour']}:00
- Business Hours: {context['time_context']['is_business_hours']}
- Session: {context['session']['namespace']}
- Recent Memories: {json.dumps(context['session']['recent_memories'], indent=2)}
- Pending Tasks: {json.dumps(context['pending_tasks'], indent=2)}
- System State: {json.dumps(context['system_state'], indent=2)}

AVAILABLE ACTIONS:
1. CHECK_EMAIL - Check for important emails, respond if needed
2. GENERATE_SUMMARY - Create daily/weekly summary for user
3. MONITOR_SYSTEM - Check system health, disk space, etc.
4. CHECK_DEADLINES - Review upcoming deadlines, alert if urgent
5. SEND_NOTIFICATION - Notify user of important events
6. UPDATE_MEMORY - Compact and organize memory files
7. BROWSE_WEB - Research topics from user interest list
8. MANAGE_FILES - Organize downloads, clean temp files

DECISION INSTRUCTIONS:
Based on the context, select the SINGLE most appropriate action to take.
Consider: urgency, user preferences, time of day, pending tasks.

RESPOND IN JSON FORMAT:
{{
    "action": "ACTION_NAME",
    "reasoning": "Why this action is appropriate",
    "params": {{}},
    "priority": 1-10
}}
"""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for decision making."""
        return """You are an autonomous AI agent heartbeat decision engine.
Your role is to decide what action to take when the agent wakes up.

PRINCIPLES:
1. Be proactive - anticipate user needs
2. Respect time - don't disturb during off-hours unless critical
3. Prioritize - urgent tasks before routine maintenance
4. Be efficient - batch similar tasks together
5. Learn - use memory to improve decisions over time

You must respond with valid JSON only."""
```

### 5.2 The 15 Hardcoded Agentic Loops

```python
# agentic_loops.py

class AgenticLoops:
    """
    The 15 core autonomous agentic loops that run on heartbeat.
    Each loop is a self-contained behavior pattern.
    """
    
    def __init__(self, heartbeat_engine, skill_registry):
        self.heartbeat = heartbeat_engine
        self.skills = skill_registry
        self.loops = self._initialize_loops()
    
    def _initialize_loops(self) -> Dict[str, Callable]:
        """Initialize all 15 agentic loops."""
        return {
            # === COMMUNICATION LOOPS (1-4) ===
            "loop_01_email_vigil": self._email_vigil_loop,
            "loop_02_sms_gateway": self._sms_gateway_loop,
            "loop_03_voice_orchestrator": self._voice_orchestrator_loop,
            "loop_04_notification_dispatcher": self._notification_dispatcher_loop,
            
            # === MONITORING LOOPS (5-8) ===
            "loop_05_system_sentinel": self._system_sentinel_loop,
            "loop_06_file_watcher": self._file_watcher_loop,
            "loop_07_web_monitor": self._web_monitor_loop,
            "loop_08_deadline_tracker": self._deadline_tracker_loop,
            
            # === INTELLIGENCE LOOPS (9-11) ===
            "loop_09_research_scout": self._research_scout_loop,
            "loop_10_summary_synthesizer": self._summary_synthesizer_loop,
            "loop_11_memory_keeper": self._memory_keeper_loop,
            
            # === EXECUTION LOOPS (12-15) ===
            "loop_12_task_executor": self._task_executor_loop,
            "loop_13_browser_pilot": self._browser_pilot_loop,
            "loop_14_file_manager": self._file_manager_loop,
            "loop_15_self_optimizer": self._self_optimizer_loop,
        }
    
    # === LOOP 01: EMAIL VIGIL ===
    async def _email_vigil_loop(self, context: LoopContext):
        """
        Continuously monitor Gmail inbox for important messages.
        Auto-respond to routine emails, escalate urgent ones.
        """
        # Check Gmail API for new messages
        new_emails = await self.skills.gmail.check_inbox(
            labels=["INBOX", "IMPORTANT"],
            unread_only=True,
            max_results=10
        )
        
        for email in new_emails:
            # Analyze email importance
            importance = await self._analyze_email_importance(email)
            
            if importance >= 8:  # Critical
                # Immediate notification
                await self.skills.tts.speak(f"Urgent email from {email.sender}")
                await self.skills.twilio.send_sms(
                    to=context.user_phone,
                    body=f"URGENT: Email from {email.sender}: {email.subject}"
                )
            elif importance >= 5:  # Important
                # Queue for user review
                await self._queue_for_review(email)
            else:  # Routine
                # Auto-respond if possible
                if await self._can_auto_respond(email):
                    response = await self._draft_response(email)
                    await self.skills.gmail.send(response)
    
    # === LOOP 02: SMS GATEWAY ===
    async def _sms_gateway_loop(self, context: LoopContext):
        """
        Monitor Twilio SMS for commands and alerts.
        Execute SMS commands, forward critical alerts.
        """
        messages = await self.skills.twilio.get_sms_messages(
            since=context.last_check
        )
        
        for msg in messages:
            # Check if it's a command
            if msg.body.startswith("!"):
                result = await self._execute_sms_command(msg.body[1:])
                await self.skills.twilio.send_sms(
                    to=msg.from_number,
                    body=f"Result: {result}"
                )
            elif any(kw in msg.body.lower() for kw in ["urgent", "emergency", "alert"]):
                # Critical alert - escalate
                await self._escalate_alert(msg)
    
    # === LOOP 03: VOICE ORCHESTRATOR ===
    async def _voice_orchestrator_loop(self, context: LoopContext):
        """
        Listen for voice wake words, process voice commands.
        Handle incoming Twilio voice calls.
        """
        # Check for voice wake detection
        if await self.skills.stt.detect_wake_word():
            # Wake word detected
            await self.skills.tts.speak("I'm listening")
            
            # Record command
            command = await self.skills.stt.transcribe(duration=10)
            
            # Execute
            result = await self._execute_voice_command(command)
            
            # Respond
            await self.skills.tts.speak(result)
    
    # === LOOP 04: NOTIFICATION DISPATCHER ===
    async def _notification_dispatcher_loop(self, context: LoopContext):
        """
        Route notifications to appropriate channels based on urgency.
        TTS for immediate, SMS for urgent, email for non-urgent.
        """
        pending = await self._get_pending_notifications()
        
        for notif in pending:
            if notif.urgency == "immediate":
                await self.skills.tts.speak(notif.message)
            elif notif.urgency == "urgent":
                await self.skills.twilio.send_sms(
                    to=context.user_phone,
                    body=notif.message
                )
            elif notif.urgency == "normal":
                await self.skills.gmail.send(
                    to=context.user_email,
                    subject=notif.title,
                    body=notif.message
                )
    
    # === LOOP 05: SYSTEM SENTINEL ===
    async def _system_sentinel_loop(self, context: LoopContext):
        """
        Monitor system health: CPU, memory, disk, temperature.
        Alert on anomalies, auto-remediate when possible.
        """
        metrics = await self._get_system_metrics()
        
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(f"CPU usage critical: {metrics.cpu_percent}%")
        if metrics.memory_percent > 85:
            alerts.append(f"Memory usage high: {metrics.memory_percent}%")
        if metrics.disk_percent > 90:
            alerts.append(f"Disk space low: {metrics.disk_percent}%")
        
        if alerts:
            # Auto-remediation
            if metrics.disk_percent > 90:
                await self._cleanup_disk_space()
            
            # Notify user
            await self._send_system_alert(alerts)
    
    # === LOOP 06: FILE WATCHER ===
    async def _file_watcher_loop(self, context: LoopContext):
        """
        Monitor watched directories for changes.
        Auto-process new files based on type and rules.
        """
        changes = await self._check_watched_directories()
        
        for change in changes:
            if change.event_type == "created":
                # Determine file type
                file_type = self._detect_file_type(change.path)
                
                # Apply processing rules
                handler = self._get_file_handler(file_type)
                if handler:
                    await handler(change.path)
    
    # === LOOP 07: WEB MONITOR ===
    async def _web_monitor_loop(self, context: LoopContext):
        """
        Monitor websites and APIs for status changes.
        Check for updates to tracked pages.
        """
        for endpoint in context.monitored_endpoints:
            status = await self._check_endpoint(endpoint)
            
            if status.code != 200:
                await self._send_alert(f"{endpoint.url} is DOWN: {status.code}")
            elif status.response_time > endpoint.threshold_ms:
                await self._send_alert(
                    f"{endpoint.url} slow: {status.response_time}ms"
                )
    
    # === LOOP 08: DEADLINE TRACKER ===
    async def _deadline_tracker_loop(self, context: LoopContext):
        """
        Track upcoming deadlines from calendar and tasks.
        Send escalating reminders as deadlines approach.
        """
        deadlines = await self._get_upcoming_deadlines(hours=48)
        
        for deadline in deadlines:
            hours_remaining = (deadline.due_date - datetime.now()).total_seconds() / 3600
            
            if hours_remaining < 2:
                # Critical - immediate notification
                await self.skills.tts.speak(
                    f"URGENT: {deadline.title} due in {int(hours_remaining)} hours!"
                )
            elif hours_remaining < 24:
                # High priority
                await self.skills.twilio.send_sms(
                    to=context.user_phone,
                    body=f"Deadline approaching: {deadline.title} in {int(hours_remaining)}h"
                )
    
    # === LOOP 09: RESEARCH SCOUT ===
    async def _research_scout_loop(self, context: LoopContext):
        """
        Monitor sources for topics of interest.
        Compile research summaries for user review.
        """
        for topic in context.research_topics:
            updates = await self._check_sources_for_topic(topic)
            
            if updates:
                # Compile summary
                summary = await self._compile_research_summary(topic, updates)
                
                # Queue for user review
                await self._queue_research_summary(summary)
    
    # === LOOP 10: SUMMARY SYNTHESIZER ===
    async def _summary_synthesizer_loop(self, context: LoopContext):
        """
        Generate periodic summaries: daily, weekly, monthly.
        Compile activity logs, highlight key events.
        """
        # Check if it's time for a summary
        if self._is_summary_time(context):
            # Gather data
            activities = await self._get_recent_activities(context.summary_period)
            
            # Generate summary using LLM
            summary = await self.llm.generate_summary(activities)
            
            # Deliver to user
            await self._deliver_summary(summary, context.summary_period)
    
    # === LOOP 11: MEMORY KEEPER ===
    async def _memory_keeper_loop(self, context: LoopContext):
        """
        Maintain and compact agent memory.
        Archive old memories, update working memory.
        """
        # Check memory size
        memory_stats = await self._get_memory_stats()
        
        if memory_stats.working_memory_size > context.max_working_memory:
            # Compact working memory
            await self._compact_working_memory()
        
        if memory_stats.total_memory_size > context.max_total_memory:
            # Archive old memories
            await self._archive_old_memories()
    
    # === LOOP 12: TASK EXECUTOR ===
    async def _task_executor_loop(self, context: LoopContext):
        """
        Execute queued background tasks.
        Retry failed tasks, report completion.
        """
        pending_tasks = await self._get_pending_tasks(limit=5)
        
        for task in pending_tasks:
            try:
                result = await self._execute_task(task)
                await self._mark_task_complete(task, result)
            except Exception as e:
                await self._handle_task_failure(task, e)
    
    # === LOOP 13: BROWSER PILOT ===
    async def _browser_pilot_loop(self, context: LoopContext):
        """
        Execute browser automation tasks.
        Scrape data, fill forms, navigate workflows.
        """
        browser_tasks = await self._get_browser_tasks()
        
        for task in browser_tasks:
            async with self.skills.browser.new_session() as browser:
                result = await browser.execute_task(task)
                await self._store_browser_result(task, result)
    
    # === LOOP 14: FILE MANAGER ===
    async def _file_manager_loop(self, context: LoopContext):
        """
        Organize files, clean temp directories.
        Apply filing rules, maintain directory structure.
        """
        # Clean temp files
        await self._clean_temp_files()
        
        # Apply filing rules
        for rule in context.filing_rules:
            await self._apply_filing_rule(rule)
    
    # === LOOP 15: SELF OPTIMIZER ===
    async def _self_optimizer_loop(self, context: LoopContext):
        """
        Analyze agent performance, optimize configurations.
        Tune heartbeat intervals, adjust priorities.
        """
        # Analyze performance metrics
        metrics = await self._get_performance_metrics()
        
        # Identify optimizations
        optimizations = await self._identify_optimizations(metrics)
        
        # Apply safe optimizations
        for opt in optimizations:
            if opt.risk_level == "low":
                await self._apply_optimization(opt)
```

---

## 6. BACKGROUND MONITORING SYSTEMS

### 6.1 Monitoring Infrastructure

```python
# monitoring_systems.py

class BackgroundMonitoringSystem:
    """
    Comprehensive background monitoring for all data sources.
    """
    
    def __init__(self):
        self.monitors = {}
        self.watchers = {}
        self.callbacks = {}
    
    async def start_all_monitors(self):
        """Start all background monitoring systems."""
        await asyncio.gather(
            self._start_gmail_monitor(),
            self._start_file_system_watcher(),
            self._start_system_metrics_collector(),
            self._start_web_endpoint_monitor(),
            self._start_calendar_monitor(),
            self._start_process_monitor(),
        )
    
    async def _start_gmail_monitor(self):
        """Monitor Gmail inbox via push notifications."""
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        
        # Setup Gmail API watch
        service = build('gmail', 'v1', credentials=self._get_gmail_credentials())
        
        # Register push notification webhook
        request = {
            'labelIds': ['INBOX', 'IMPORTANT'],
            'topicName': 'projects/agent/topics/gmail-notifications'
        }
        
        response = service.users().watch(userId='me', body=request).execute()
        
        self.logger.info(f"Gmail monitor started: {response['historyId']}")
    
    async def _start_file_system_watcher(self):
        """Watch file system for changes using watchdog."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class AgentFileHandler(FileSystemEventHandler):
            def __init__(self, callback):
                self.callback = callback
            
            def on_created(self, event):
                if not event.is_directory:
                    self.callback('created', event.src_path)
            
            def on_modified(self, event):
                if not event.is_directory:
                    self.callback('modified', event.src_path)
            
            def on_deleted(self, event):
                if not event.is_directory:
                    self.callback('deleted', event.src_path)
        
        # Watch key directories
        watch_paths = [
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Documents/AgentWatch"),
            os.path.expanduser("~/.agent/inbox"),
        ]
        
        observer = Observer()
        handler = AgentFileHandler(self._on_file_event)
        
        for path in watch_paths:
            if os.path.exists(path):
                observer.schedule(handler, path, recursive=True)
        
        observer.start()
        self.watchers['filesystem'] = observer
    
    async def _start_system_metrics_collector(self):
        """Collect system metrics periodically."""
        import psutil
        
        while self.running:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': psutil.virtual_memory()._asdict(),
                'disk': psutil.disk_usage('/')._asdict(),
                'network': psutil.net_io_counters()._asdict(),
                'processes': len(psutil.pids())
            }
            
            # Store metrics
            await self._store_metrics(metrics)
            
            # Check thresholds
            await self._check_metric_thresholds(metrics)
            
            await asyncio.sleep(60)  # Collect every minute
    
    async def _start_web_endpoint_monitor(self):
        """Monitor web endpoints for availability."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            while self.running:
                for endpoint in self.config.monitored_endpoints:
                    try:
                        start = time.time()
                        async with session.get(
                            endpoint.url,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            elapsed = (time.time() - start) * 1000
                            
                            result = {
                                'endpoint': endpoint.url,
                                'status': response.status,
                                'response_time_ms': elapsed,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            await self._store_endpoint_result(result)
                            
                            # Check for issues
                            if response.status != 200:
                                await self._trigger_alert(
                                    f"Endpoint {endpoint.url} returned {response.status}"
                                )
                            elif elapsed > endpoint.threshold_ms:
                                await self._trigger_alert(
                                    f"Endpoint {endpoint.url} slow: {elapsed:.0f}ms"
                                )
                    
                    except Exception as e:
                        await self._trigger_alert(
                            f"Endpoint {endpoint.url} unreachable: {str(e)}"
                        )
                
                await asyncio.sleep(endpoint.check_interval_seconds)
```

---

## 7. SELF-TRIGGERING LOGIC

### 7.1 Self-Trigger Engine

```python
# self_trigger.py

class SelfTriggerEngine:
    """
    Enables the agent to trigger itself based on internal state and goals.
    This is the core of autonomous behavior.
    """
    
    def __init__(self, heartbeat_engine, memory_system, goal_system):
        self.heartbeat = heartbeat_engine
        self.memory = memory_system
        self.goals = goal_system
        self.trigger_history = []
    
    async def evaluate_self_triggers(self):
        """
        Evaluate all self-trigger conditions and fire if met.
        Called during each heartbeat cycle.
        """
        triggers = [
            self._check_goal_progress_triggers,
            self._check_memory_triggers,
            self._check_context_triggers,
            self._check_predictive_triggers,
            self._check_social_triggers,
        ]
        
        for check in triggers:
            trigger = await check()
            if trigger:
                await self._fire_self_trigger(trigger)
    
    async def _check_goal_progress_triggers(self) -> Optional[SelfTrigger]:
        """Trigger based on goal progress or deadlines."""
        active_goals = await self.goals.get_active_goals()
        
        for goal in active_goals:
            # Check if goal needs attention
            if goal.next_action_due and goal.next_action_due < datetime.now():
                return SelfTrigger(
                    trigger_type="goal_due",
                    reason=f"Goal '{goal.title}' has pending action",
                    priority=goal.priority,
                    suggested_action="work_on_goal",
                    params={"goal_id": goal.id}
                )
            
            # Check if goal is behind schedule
            if goal.progress < goal.expected_progress:
                return SelfTrigger(
                    trigger_type="goal_behind",
                    reason=f"Goal '{goal.title}' is behind schedule",
                    priority="high",
                    suggested_action="catch_up_goal",
                    params={"goal_id": goal.id}
                )
    
    async def _check_memory_triggers(self) -> Optional[SelfTrigger]:
        """Trigger based on memory patterns and reminders."""
        # Check for time-based reminders
        due_reminders = await self.memory.get_due_reminders()
        
        if due_reminders:
            reminder = due_reminders[0]
            return SelfTrigger(
                trigger_type="reminder_due",
                reason=f"Reminder: {reminder.content}",
                priority=reminder.priority,
                suggested_action="process_reminder",
                params={"reminder_id": reminder.id}
            )
        
        # Check for pattern-based triggers
        recent_context = await self.memory.get_recent_context(hours=24)
        
        # Detect if user seems stressed (many urgent emails, late hours, etc.)
        if self._detect_stress_pattern(recent_context):
            return SelfTrigger(
                trigger_type="user_stress_detected",
                reason="User may be experiencing high stress",
                priority="medium",
                suggested_action="offer_assistance",
                params={}
            )
    
    async def _check_context_triggers(self) -> Optional[SelfTrigger]:
        """Trigger based on current context and situation."""
        context = await self._gather_current_context()
        
        # Check for calendar conflicts
        if context.get('calendar_conflict'):
            return SelfTrigger(
                trigger_type="calendar_conflict",
                reason="Calendar conflict detected",
                priority="high",
                suggested_action="resolve_calendar_conflict",
                params=context['calendar_conflict']
            )
        
        # Check for travel time
        if context.get('next_meeting_location'):
            travel_time = await self._calculate_travel_time(
                context['current_location'],
                context['next_meeting_location']
            )
            
            time_to_meeting = (context['next_meeting_time'] - datetime.now()).total_seconds() / 60
            
            if travel_time > time_to_meeting - 15:  # Need to leave in 15 min
                return SelfTrigger(
                    trigger_type="departure_reminder",
                    reason=f"Need to leave in {int(time_to_meeting - travel_time)} minutes for meeting",
                    priority="high",
                    suggested_action="send_departure_alert",
                    params={"travel_time": travel_time}
                )
    
    async def _check_predictive_triggers(self) -> Optional[SelfTrigger]:
        """Trigger based on predicted future needs."""
        # Predictive analysis based on patterns
        patterns = await self._analyze_user_patterns()
        
        # If user typically orders lunch at 11:30, remind at 11:15
        if patterns.get('lunch_time_approaching'):
            return SelfTrigger(
                trigger_type="predicted_need",
                reason="User typically orders lunch around this time",
                priority="low",
                suggested_action="offer_lunch_options",
                params={}
            )
        
        # Predict email responses needed
        if patterns.get('emails_need_response'):
            return SelfTrigger(
                trigger_type="pending_responses",
                reason=f"{patterns['pending_emails']} emails may need responses",
                priority="medium",
                suggested_action="draft_responses",
                params={"emails": patterns['pending_emails']}
            )
    
    async def _check_social_triggers(self) -> Optional[SelfTrigger]:
        """Trigger based on social context and relationships."""
        # Check for birthdays, anniversaries
        today_events = await self._get_today_events()
        
        for event in today_events:
            if event.type in ['birthday', 'anniversary']:
                return SelfTrigger(
                    trigger_type="special_occasion",
                    reason=f"{event.person_name}'s {event.type} today",
                    priority="medium",
                    suggested_action="suggest_celebration",
                    params={"event": event}
                )
    
    async def _fire_self_trigger(self, trigger: SelfTrigger):
        """Fire a self-triggered wake event."""
        self.logger.info(f"SELF-TRIGGER: {trigger.trigger_type} - {trigger.reason}")
        
        # Create wake event
        wake_event = WakeEvent(
            trigger_id=f"self_{trigger.trigger_type}",
            trigger_type=TriggerType.SELF_TRIGGERED,
            priority=TriggerPriority[trigger.priority.upper()],
            timestamp=datetime.now(),
            session_namespace="background",
            action_handler=trigger.suggested_action,
            action_params=trigger.params,
            is_self_triggered=True
        )
        
        # Dispatch to heartbeat engine
        await self.heartbeat.dispatch_wake_event(wake_event)
        
        # Record trigger
        self.trigger_history.append({
            "timestamp": datetime.now(),
            "type": trigger.trigger_type,
            "reason": trigger.reason
        })
```

---

## 8. HEARTBEAT INTERVALS AND PATTERNS

### 8.1 Interval Configuration

```python
# heartbeat_intervals.py

class HeartbeatIntervals:
    """
    Defines all heartbeat intervals and patterns.
    """
    
    # === ULTRA-FREQUENT (Real-time) ===
    INTERVAL_VOICE_WAKE = 0.1  # 100ms - continuous listening
    INTERVAL_STT_PROCESSING = 0.5  # 500ms - voice processing
    
    # === FREQUENT (Seconds) ===
    INTERVAL_SYSTEM_METRICS = 5  # 5 seconds - CPU/memory check
    INTERVAL_PROCESS_MONITOR = 10  # 10 seconds - process health
    INTERVAL_QUICK_POLL = 15  # 15 seconds - quick status checks
    
    # === REGULAR (Minutes) ===
    INTERVAL_HEARTBEAT_CORE = 60  # 1 minute - main heartbeat
    INTERVAL_GMAIL_CHECK = 2  # 2 minutes - inbox poll (backup to push)
    INTERVAL_FILE_SCAN = 5  # 5 minutes - file system scan
    INTERVAL_WEB_CHECK = 5  # 5 minutes - endpoint health
    INTERVAL_TASK_QUEUE = 3  # 3 minutes - background tasks
    
    # === PERIODIC (Hours) ===
    INTERVAL_SUMMARY_GENERATION = 1  # 1 hour - activity summary
    INTERVAL_MEMORY_COMPACTION = 4  # 4 hours - memory maintenance
    INTERVAL_FULL_BACKUP = 6  # 6 hours - state backup
    
    # === DAILY PATTERNS ===
    DAILY_MORNING_ROUTINE = "0 7 * * *"  # 7:00 AM
    DAILY_MIDDAY_CHECK = "0 12 * * *"  # 12:00 PM
    DAILY_EVENING_SUMMARY = "0 18 * * *"  # 6:00 PM
    DAILY_NIGHT_MAINTENANCE = "0 2 * * *"  # 2:00 AM
    
    # === WEEKLY PATTERNS ===
    WEEKLY_REPORT = "0 9 * * 1"  # Monday 9:00 AM
    WEEKLY_CLEANUP = "0 3 * * 0"  # Sunday 3:00 AM
    
    # === ADAPTIVE INTERVALS ===
    ADAPTIVE_IDLE_MULTIPLIER = 2.0  # Slow down when idle
    ADAPTIVE_ACTIVE_DIVISOR = 0.5  # Speed up when active
    ADAPTIVE_STRESS_MULTIPLIER = 0.25  # Speed up when stress detected

class AdaptiveHeartbeatManager:
    """
    Dynamically adjusts heartbeat intervals based on system state.
    """
    
    def __init__(self):
        self.base_intervals = HeartbeatIntervals()
        self.current_multipliers = {}
        self.activity_history = []
    
    def get_interval(self, interval_name: str) -> float:
        """Get the current adaptive interval."""
        base = getattr(self.base_intervals, interval_name)
        multiplier = self.current_multipliers.get(interval_name, 1.0)
        return base * multiplier
    
    async def update_adaptive_intervals(self):
        """Update intervals based on current context."""
        context = await self._analyze_context()
        
        # Adjust based on user activity
        if context['user_active']:
            # User is active - be more responsive
            self.current_multipliers['INTERVAL_HEARTBEAT_CORE'] = 0.5
            self.current_multipliers['INTERVAL_GMAIL_CHECK'] = 0.5
        else:
            # User idle - conserve resources
            self.current_multipliers['INTERVAL_HEARTBEAT_CORE'] = 2.0
            self.current_multipliers['INTERVAL_GMAIL_CHECK'] = 2.0
        
        # Adjust based on system load
        if context['system_load'] > 80:
            # High load - reduce heartbeat frequency
            for key in self.current_multipliers:
                self.current_multipliers[key] *= 1.5
        
        # Adjust based on pending tasks
        if context['pending_tasks'] > 10:
            # Many tasks - speed up task processing
            self.current_multipliers['INTERVAL_TASK_QUEUE'] = 0.25
        
        # Adjust based on urgency
        if context['urgent_items'] > 0:
            # Urgent items - maximum responsiveness
            for key in self.current_multipliers:
                self.current_multipliers[key] = 0.25
```

---

## 9. PRIORITY-BASED ACTIVATION

### 9.1 Priority System

```python
# priority_system.py

class PriorityActivationSystem:
    """
    Manages activation priorities and preemption.
    """
    
    PRIORITY_LEVELS = {
        'CRITICAL': {
            'value': 1,
            'preempt': True,
            'notify_user': True,
            'max_wait': 0,  # No waiting
            'examples': ['voice_wake', 'emergency_alert', 'system_failure']
        },
        'HIGH': {
            'value': 2,
            'preempt': True,
            'notify_user': True,
            'max_wait': 5,  # 5 seconds
            'examples': ['important_email', 'deadline_approaching', 'calendar_conflict']
        },
        'MEDIUM': {
            'value': 3,
            'preempt': False,
            'notify_user': False,
            'max_wait': 60,  # 1 minute
            'examples': ['routine_check', 'summary_generation', 'file_organization']
        },
        'LOW': {
            'value': 4,
            'preempt': False,
            'notify_user': False,
            'max_wait': 300,  # 5 minutes
            'examples': ['memory_compaction', 'archive_old_data']
        },
        'BACKGROUND': {
            'value': 5,
            'preempt': False,
            'notify_user': False,
            'max_wait': 3600,  # 1 hour
            'examples': ['idle_cleanup', 'index_optimization']
        }
    }
    
    def __init__(self):
        self.priority_queue = PriorityQueue()
        self.current_task = None
        self.task_history = []
    
    async def enqueue_task(self, task: Task) -> str:
        """Add a task to the priority queue."""
        task_id = str(uuid.uuid4())
        
        # Calculate effective priority
        effective_priority = self._calculate_effective_priority(task)
        
        # Add to queue
        self.priority_queue.put((
            effective_priority,
            task.timestamp,
            task_id,
            task
        ))
        
        # Check for preemption
        if self._should_preempt(task):
            await self._preempt_current_task()
        
        return task_id
    
    def _calculate_effective_priority(self, task: Task) -> int:
        """Calculate effective priority based on multiple factors."""
        base_priority = self.PRIORITY_LEVELS[task.priority]['value']
        
        # Age factor - older tasks get priority boost
        age_seconds = (datetime.now() - task.created_at).total_seconds()
        age_boost = min(age_seconds / 300, 1)  # Max +1 after 5 minutes
        
        # User importance factor
        importance_boost = task.user_importance * 0.5  # 0-0.5 boost
        
        # Deadline factor
        deadline_boost = 0
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            if time_to_deadline < 3600:  # Less than 1 hour
                deadline_boost = 1.0
            elif time_to_deadline < 86400:  # Less than 1 day
                deadline_boost = 0.5
        
        effective = base_priority - age_boost - importance_boost - deadline_boost
        return max(1, int(effective))  # Never below 1
    
    def _should_preempt(self, new_task: Task) -> bool:
        """Determine if new task should preempt current task."""
        if not self.current_task:
            return False
        
        new_priority = self.PRIORITY_LEVELS[new_task.priority]
        current_priority = self.PRIORITY_LEVELS[self.current_task.priority]
        
        # Only preempt if new task is higher priority and allows preemption
        if new_priority['value'] < current_priority['value'] and new_priority['preempt']:
            # Check if current task can be interrupted
            if self.current_task.can_interrupt:
                return True
        
        return False
    
    async def _preempt_current_task(self):
        """Preempt the current task to handle higher priority item."""
        if self.current_task:
            # Save state
            await self._save_task_state(self.current_task)
            
            # Cancel current task
            self.current_task.cancel()
            
            # Re-queue with saved state
            await self.enqueue_task(self.current_task)
            
            self.logger.info(f"Preempted task {self.current_task.id}")
```

---

## 10. HEARTBEAT LOGGING AND TRACKING

### 10.1 Comprehensive Logging System

```python
# heartbeat_logging.py

class HeartbeatLogger:
    """
    Comprehensive logging and tracking for all heartbeat activities.
    """
    
    def __init__(self, log_dir: str = "logs/heartbeat"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup loggers
        self.activity_logger = self._setup_logger("activity")
        self.metrics_logger = self._setup_logger("metrics")
        self.decision_logger = self._setup_logger("decisions")
        self.error_logger = self._setup_logger("errors")
        
        # Statistics
        self.stats = HeartbeatStats()
    
    async def log_wake_event(self, event: WakeEvent):
        """Log a heartbeat wake event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "wake",
            "trigger_id": event.trigger_id,
            "trigger_type": event.trigger_type.name,
            "priority": event.priority.name,
            "session_namespace": event.session_namespace,
            "is_self_triggered": getattr(event, 'is_self_triggered', False)
        }
        
        self.activity_logger.info(json.dumps(log_entry))
        
        # Update stats
        self.stats.total_wakeups += 1
        self.stats.wakeups_by_trigger[event.trigger_id] = \
            self.stats.wakeups_by_trigger.get(event.trigger_id, 0) + 1
    
    async def log_action_execution(self, action: Action, result: ActionResult):
        """Log action execution details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "action_execution",
            "action_type": action.action_type.name,
            "action_name": action.name,
            "duration_seconds": result.duration,
            "success": result.success,
            "error": result.error if not result.success else None
        }
        
        self.activity_logger.info(json.dumps(log_entry))
        
        # Update stats
        if result.success:
            self.stats.actions_executed += 1
        else:
            self.stats.actions_failed += 1
    
    async def log_decision(self, context: Dict, decision: Dict):
        """Log autonomous decision making."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "decision",
            "trigger": context.get('trigger'),
            "selected_action": decision.get('action'),
            "reasoning": decision.get('reasoning'),
            "priority": decision.get('priority'),
            "context_hash": hash(json.dumps(context, default=str))
        }
        
        self.decision_logger.info(json.dumps(log_entry))
    
    async def log_metrics(self, metrics: Dict):
        """Log system metrics."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "metrics",
            **metrics
        }
        
        self.metrics_logger.info(json.dumps(log_entry))
    
    async def generate_heartbeat_report(self, period: str = "daily") -> Dict:
        """Generate comprehensive heartbeat report."""
        report = {
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_wakeups": self.stats.total_wakeups,
                "actions_executed": self.stats.actions_executed,
                "actions_failed": self.stats.actions_failed,
                "success_rate": self.stats.actions_executed / max(
                    self.stats.actions_executed + self.stats.actions_failed, 1
                ) * 100
            },
            "wakeups_by_trigger": self.stats.wakeups_by_trigger,
            "actions_by_type": self.stats.actions_by_type,
            "average_wake_duration": self.stats.avg_wake_duration,
            "peak_activity_hours": self._calculate_peak_hours(),
            "self_trigger_rate": self._calculate_self_trigger_rate()
        }
        
        return report

@dataclass
class HeartbeatStats:
    """Heartbeat statistics tracking."""
    total_wakeups: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    avg_wake_duration: float = 0.0
    wakeups_by_trigger: Dict[str, int] = None
    actions_by_type: Dict[str, int] = None
    self_triggers: int = 0
    preemptions: int = 0
    
    def __post_init__(self):
        if self.wakeups_by_trigger is None:
            self.wakeups_by_trigger = {}
        if self.actions_by_type is None:
            self.actions_by_type = {}
```

### 10.2 Heartbeat Dashboard

```python
# heartbeat_dashboard.py

class HeartbeatDashboard:
    """
    Real-time dashboard for monitoring heartbeat activity.
    """
    
    def __init__(self, heartbeat_engine: HeartbeatEngine):
        self.heartbeat = heartbeat_engine
        self.running = False
    
    async def start_dashboard(self, host: str = "localhost", port: int = 8080):
        """Start the web dashboard."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/', self._handle_index)
        app.router.add_get('/api/status', self._handle_status)
        app.router.add_get('/api/triggers', self._handle_triggers)
        app.router.add_get('/api/stats', self._handle_stats)
        app.router.add_get('/api/logs', self._handle_logs)
        app.router.add_get('/ws', self._handle_websocket)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        self.running = True
        
        # Start broadcast loop
        asyncio.create_task(self._broadcast_updates())
    
    async def _handle_status(self, request):
        """Return current heartbeat status."""
        status = {
            "running": self.heartbeat.running,
            "uptime_seconds": self._get_uptime(),
            "active_triggers": len(self.heartbeat.triggers),
            "queue_size": self.heartbeat.dispatch_queue.qsize(),
            "current_session": self.heartbeat.session_manager.current_session,
            "stats": self.heartbeat.stats
        }
        return web.json_response(status)
    
    async def _broadcast_updates(self):
        """Broadcast real-time updates to connected clients."""
        while self.running:
            update = {
                "timestamp": datetime.now().isoformat(),
                "heartbeat_rate": self._calculate_current_rate(),
                "queue_depth": self.heartbeat.dispatch_queue.qsize(),
                "recent_wakeups": self._get_recent_wakeups(10)
            }
            
            # Broadcast to all connected websockets
            for ws in self.websocket_clients:
                await ws.send_json(update)
            
            await asyncio.sleep(1)  # Update every second
```

---

## 11. WINDOWS 10 INTEGRATION

### 11.1 Windows Service Integration

```python
# windows_service.py

import win32service
import win32serviceutil
import win32event
import servicemanager

class AgentHeartbeatService(win32serviceutil.ServiceFramework):
    """
    Windows Service wrapper for the heartbeat engine.
    Ensures 24/7 operation with automatic restart.
    """
    
    _svc_name_ = "AgentHeartbeat"
    _svc_display_name_ = "AI Agent Heartbeat Service"
    _svc_description_ = "Autonomous heartbeat and activation system for AI agent"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.heartbeat_engine = None
    
    def SvcStop(self):
        """Stop the service."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.heartbeat_engine:
            asyncio.run(self.heartbeat_engine.stop())
    
    def SvcDoRun(self):
        """Run the service."""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        # Initialize and start heartbeat
        self.heartbeat_engine = HeartbeatEngine()
        asyncio.run(self.heartbeat_engine.start())
        
        # Wait for stop signal
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

# Service installation
if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AgentHeartbeatService)
```

---

## 12. CONFIGURATION FILE

### 12.1 Full Configuration Schema

```yaml
# config/heartbeat.yaml

heartbeat:
  enabled: true
  
  # Core settings
  core:
    default_interval_seconds: 60
    max_concurrent_tasks: 5
    task_timeout_seconds: 300
    enable_preemption: true
    enable_self_triggers: true
  
  # Session management
  sessions:
    isolation_enabled: true
    max_background_sessions: 3
    session_timeout_minutes: 30
  
  # Adaptive intervals
  adaptive:
    enabled: true
    idle_slowdown_multiplier: 2.0
    active_speedup_divisor: 0.5
    stress_speedup_divisor: 0.25
  
  # Triggers
  triggers:
    voice_wake:
      enabled: true
      wake_words: ["hey agent", "agent wake", "jarvis", "computer"]
      sensitivity: 0.8
      
    gmail_push:
      enabled: true
      watch_labels: ["INBOX", "IMPORTANT"]
      poll_interval_seconds: 120  # Backup polling
      
    twilio:
      enabled: true
      sms_keywords: ["urgent", "emergency", "alert", "critical"]
      
    file_watcher:
      enabled: true
      watch_paths:
        - "~/Downloads"
        - "~/Documents/AgentWatch"
      watch_patterns: ["*.pdf", "*.docx", "*.zip"]
      
    system_health:
      enabled: true
      check_interval_seconds: 60
      thresholds:
        cpu_percent: 90
        memory_percent: 85
        disk_percent: 90
  
  # Agentic loops
  loops:
    enabled:
      - email_vigil
      - sms_gateway
      - voice_orchestrator
      - notification_dispatcher
      - system_sentinel
      - file_watcher
      - web_monitor
      - deadline_tracker
      - research_scout
      - summary_synthesizer
      - memory_keeper
      - task_executor
      - browser_pilot
      - file_manager
      - self_optimizer
  
  # Logging
  logging:
    level: INFO
    log_dir: "logs/heartbeat"
    max_log_size_mb: 100
    max_log_files: 10
    enable_dashboard: true
    dashboard_port: 8080
  
  # Notifications
  notifications:
    channels:
      - tts
      - sms
      - email
    urgency_routing:
      critical: [tts, sms]
      high: [sms, email]
      medium: [email]
      low: []
```

---

## 13. IMPLEMENTATION CHECKLIST

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Heartbeat Engine Core | Pending | Critical | Main orchestration |
| Trigger Registry | Pending | Critical | All 8 trigger types |
| Temporal Triggers | Pending | Critical | Cron/interval |
| Event Watchers | Pending | High | File system, Gmail |
| Webhook Server | Pending | High | External triggers |
| Voice Wake Detection | Pending | High | STT integration |
| Twilio Integration | Pending | High | SMS/Voice |
| Action Selector | Pending | Critical | GPT-5.2 decision |
| 15 Agentic Loops | Pending | Critical | Core behaviors |
| Self-Trigger Engine | Pending | High | Autonomous triggers |
| Priority System | Pending | High | Preemption support |
| Adaptive Intervals | Pending | Medium | Dynamic adjustment |
| Logging System | Pending | High | Full audit trail |
| Dashboard | Pending | Medium | Web UI |
| Windows Service | Pending | High | 24/7 operation |
| Configuration | Pending | High | YAML config |

---

## 14. SECURITY CONSIDERATIONS

1. **Credential Isolation**: Store all credentials in Windows Credential Manager
2. **Sandboxed Execution**: Run background tasks in isolated processes
3. **Audit Logging**: Log all autonomous actions for review
4. **Rate Limiting**: Prevent trigger storms with cooldown periods
5. **User Confirmation**: Require confirmation for high-risk actions
6. **Network Security**: Validate all webhook signatures
7. **File Access**: Restrict file system access to allowed paths

---

## 15. PERFORMANCE OPTIMIZATION

1. **Lazy Loading**: Only load skills when needed
2. **Connection Pooling**: Reuse Gmail, Twilio connections
3. **Batch Processing**: Group similar tasks together
4. **Memory Management**: Compact memory regularly
5. **CPU Throttling**: Reduce activity during high system load
6. **Network Caching**: Cache API responses appropriately

---

## 16. SUMMARY

This Heartbeat Mechanism and Autonomous Activation System provides:

### Core Capabilities
- **Self-awakening**: Agent wakes itself on schedule or event
- **Multi-source monitoring**: Gmail, files, system, web, voice, SMS
- **Autonomous decision making**: GPT-5.2 powered action selection
- **Priority-based execution**: Critical tasks preempt lower priority ones
- **Session isolation**: Background tasks don't contaminate main session
- **Comprehensive logging**: Full audit trail of all autonomous actions

### The 15 Agentic Loops
1. Email Vigil - Monitor and process emails
2. SMS Gateway - Handle SMS commands and alerts
3. Voice Orchestrator - Voice wake and command processing
4. Notification Dispatcher - Route notifications by urgency
5. System Sentinel - Monitor system health
6. File Watcher - Watch and process file changes
7. Web Monitor - Monitor websites and APIs
8. Deadline Tracker - Track and alert on deadlines
9. Research Scout - Monitor topics of interest
10. Summary Synthesizer - Generate periodic summaries
11. Memory Keeper - Maintain and compact memory
12. Task Executor - Execute queued background tasks
13. Browser Pilot - Browser automation
14. File Manager - Organize files and cleanup
15. Self Optimizer - Analyze and optimize performance

### Windows 10 Integration
- Runs as Windows Service for 24/7 operation
- Uses Windows wake timers for sleep/hibernation
- Integrates with Windows Credential Manager
- Supports Windows-specific APIs and events

---

*Document Version: 1.0*
*Last Updated: 2025*
*For: Windows 10 OpenClaw-Inspired AI Agent*
