# Identity Hooks and Transformation Trigger System
## Technical Specification for Windows 10 OpenClaw-Inspired AI Agent

**Version:** 1.0  
**Date:** 2026-01-18  
**Status:** Technical Specification  
**Classification:** Architecture Design Document

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Hook System Design](#3-hook-system-design)
4. [Trigger Types](#4-trigger-types)
5. [Identity Transformation Mechanisms](#5-identity-transformation-mechanisms)
6. [Mode Switching System](#6-mode-switching-system)
7. [Hook Configuration and Management](#7-hook-configuration-and-management)
8. [Transformation Logging](#8-transformation-logging)
9. [Safe Transformation Boundaries](#9-safe-transformation-boundaries)
10. [Emergency Reset Mechanisms](#10-emergency-reset-mechanisms)
11. [Implementation Reference](#11-implementation-reference)
12. [Appendices](#12-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the complete technical architecture for the Identity Hooks and Transformation Trigger System for a Windows 10 OpenClaw-inspired AI agent. The system enables dynamic identity transformation through a sophisticated hook mechanism, supporting scheduled, event-based, and probabilistic triggers.

### 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| Dynamic Identity Swapping | Runtime persona changes via SOUL.md replacement |
| Multi-Modal Triggers | Time-based, event-based, random, and hybrid triggers |
| Safe State Transitions | Boundary enforcement and rollback capabilities |
| Comprehensive Logging | Full audit trail of all transformations |
| Emergency Controls | Kill switches and immediate reset mechanisms |
| Mode Management | Professional, casual, creative, and custom modes |

### 1.3 Design Principles

1. **Safety First**: All transformations must be reversible and bounded
2. **Observability**: Every state change is logged and auditable
3. **Modularity**: Hooks are self-contained, testable units
4. **Extensibility**: Easy to add new hooks and trigger types
5. **Performance**: Minimal overhead during normal operation

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IDENTITY HOOKS SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   TRIGGER    │───▶│   HOOK       │───▶│ TRANSFORM    │                  │
│  │   ENGINE     │    │   REGISTRY   │    │   ENGINE     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │              IDENTITY STATE MANAGER                  │                   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │                   │
│  │  │  Core    │  │  Active  │  │  Backup  │          │                   │
│  │  │ Identity │  │  Persona │  │  Stack   │          │                   │
│  │  └──────────┘  └──────────┘  └──────────┘          │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                            │                                               │
│         ┌──────────────────┼──────────────────┐                           │
│         ▼                  ▼                  ▼                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│  │   SAFETY     │   │   AUDIT      │   │  EMERGENCY   │                  │
│  │   BOUNDARY   │   │   LOGGER     │   │   SYSTEM     │                  │
│  │   CHECKER    │   │              │   │              │                  │
│  └──────────────┘   └──────────────┘   └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Trigger Engine | Monitors and fires triggers | Python asyncio, APScheduler |
| Hook Registry | Manages available hooks | Python dict + file system |
| Transform Engine | Executes identity changes | File I/O, memory management |
| State Manager | Maintains identity state | In-memory + persistent storage |
| Safety Boundary | Enforces transformation limits | Policy engine |
| Audit Logger | Records all transformations | Structured logging |
| Emergency System | Kill switches and resets | Signal handlers, watchdog |

---

## 3. Hook System Design

### 3.1 Hook Types

#### 3.1.1 Soul Hooks (Identity Transformation)

| Hook ID | Description | Risk Level | Default Enabled |
|---------|-------------|------------|-----------------|
| `soul-evil` | "Evil twin" persona - mischievous but safe | Medium | No |
| `soul-creative` | Enhanced creativity mode | Low | Yes |
| `soul-focused` | Deep work, minimal distractions | Low | Yes |
| `soul-mentor` | Teaching/educational persona | Low | Yes |
| `soul-debug` | Technical debugging mode | Low | Yes |
| `soul-night` | Night owl persona (reduced verbosity) | Low | Yes |

#### 3.1.2 Mode Hooks (Behavioral Modification)

| Hook ID | Description | Affects |
|---------|-------------|---------|
| `mode-professional` | Formal business communication | Tone, vocabulary |
| `mode-casual` | Relaxed conversational style | Tone, emoji usage |
| `mode-creative` | Brainstorming and ideation | Response structure |
| `mode-analytical` | Data-driven responses | Reasoning depth |
| `mode-concise` | Minimal word responses | Response length |
| `mode-verbose` | Detailed explanations | Response length |

### 3.2 Hook Structure (YAML Schema)

```yaml
hook_definition:
  hook_id: "soul-evil"
  version: "1.0.0"
  name: "Evil Twin Persona"
  description: "Mischievous alter ego with playful antagonism"
  category: "soul"
  risk_level: "medium"  # low, medium, high, critical
  enabled: false
  auto_enable: false
  
  source:
    type: "file"
    path: "hooks/souls/soul-evil.md"
    checksum: "sha256:abc123..."
    
  triggers:
    - type: "scheduled"
      cron: "0 0 * * 1"  # Weekly on Monday midnight
    - type: "random"
      probability: 0.01  # 1% chance per hour
    - type: "event"
      event: "user_idle_30min"
      
  safety:
    max_duration_minutes: 60
    require_confirmation: true
    allowed_systems: ["chat", "tts"]
    blocked_systems: ["email_outbound", "file_delete", "system_command"]
    
  rollback:
    auto_rollback: true
    rollback_triggers:
      - "user_command:stop"
      - "duration_exceeded"
      - "safety_violation"
      
  notifications:
    on_activate:
      - type: "log"
        level: "info"
      - type: "tts"
        message: "Evil mode activated. Mwahaha!"
```

### 3.3 Hook Registry (Python Implementation)

```python
class HookRegistry:
    """Central registry for all identity hooks."""
    
    def __init__(self, hooks_directory: str = "hooks"):
        self.hooks: Dict[str, BaseHook] = {}
        self.hooks_dir = Path(hooks_directory)
        self._lock = asyncio.Lock()
        
    async def discover_hooks(self) -> List[str]:
        """Scan hooks directory and register all valid hooks."""
        discovered = []
        for hook_file in self.hooks_dir.rglob("*.yaml"):
            try:
                hook = await self._load_hook(hook_file)
                await self.register(hook)
                discovered.append(hook.hook_id)
            except HookValidationError as e:
                logger.warning(f"Invalid hook {hook_file}: {e}")
        return discovered
        
    async def register(self, hook: BaseHook) -> None:
        async with self._lock:
            if hook.hook_id in self.hooks:
                raise HookAlreadyExistsError(hook.hook_id)
            self.hooks[hook.hook_id] = hook
            
    def get(self, hook_id: str) -> Optional[BaseHook]:
        return self.hooks.get(hook_id)
        
    def list_hooks(self, category=None, enabled_only=False, risk_level=None) -> List[BaseHook]:
        results = list(self.hooks.values())
        if category:
            results = [h for h in results if h.category == category]
        if enabled_only:
            results = [h for h in results if h.enabled]
        if risk_level:
            results = [h for h in results if h.risk_level == risk_level]
        return results
```

---

## 4. Trigger Types

### 4.1 Time-Based Triggers

#### 4.1.1 Cron-Based Scheduling

```python
class CronTrigger(BaseTrigger):
    """Cron-based scheduled trigger using APScheduler."""
    
    def __init__(self, cron_expression: str, timezone: str = "UTC", jitter_seconds: int = 0):
        self.cron = cron_expression
        self.timezone = pytz.timezone(timezone)
        self.jitter = jitter_seconds
        
    def should_fire(self, context: TriggerContext) -> bool:
        now = datetime.now(self.timezone)
        cron_iter = croniter(self.cron, now - timedelta(minutes=1))
        next_run = cron_iter.get_next(datetime)
        time_diff = abs((now - next_run).total_seconds())
        return time_diff <= 60  # 1-minute window
```

#### 4.1.2 Common Cron Patterns

| Pattern | Cron Expression | Use Case |
|---------|-----------------|----------|
| Hourly | `0 * * * *` | Regular check-ins |
| Daily Morning | `0 9 * * *` | Morning briefing mode |
| Daily Evening | `0 18 * * *` | Evening wind-down |
| Weekly | `0 0 * * 1` | Weekly planning mode |
| Weekdays Only | `0 9 * * 1-5` | Business hours mode |
| Weekends | `0 10 * * 0,6` | Relaxed weekend mode |
| Night Owl | `0 22 * * *` | Late-night focus mode |

### 4.2 Event-Based Triggers

```python
class EventType(Enum):
    # User interaction events
    USER_MESSAGE = "user_message"
    USER_IDLE = "user_idle"
    USER_RETURN = "user_return"
    USER_COMMAND = "user_command"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    
    # Task events
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    
    # External events
    EMAIL_RECEIVED = "email_received"
    CALENDAR_EVENT = "calendar_event"
```

### 4.3 Random Triggers

```python
class RandomTrigger(BaseTrigger):
    """Random trigger with configurable probability."""
    
    def __init__(self, probability_per_hour: float, min_interval_minutes: int = 60, 
                 time_window: Optional[Tuple[int, int]] = None):
        self.probability = probability_per_hour
        self.min_interval = timedelta(minutes=min_interval_minutes)
        self.time_window = time_window
        self._last_fire: Optional[datetime] = None
        
    def should_fire(self, context: TriggerContext) -> bool:
        if self._last_fire and datetime.now() < self._last_fire + self.min_interval:
            return False
        if self.time_window:
            current_hour = datetime.now().hour
            start, end = self.time_window
            if not (start <= current_hour < end):
                return False
        check_probability = self.probability / 60  # Convert to per-minute
        if random.random() < check_probability:
            self._last_fire = datetime.now()
            return True
        return False
```

### 4.4 Hybrid Triggers

```python
class HybridTrigger(BaseTrigger):
    """Combines multiple trigger conditions with AND/OR logic."""
    
    def __init__(self, triggers: List[BaseTrigger], logic: str = "OR", priority: int = 0):
        self.triggers = triggers
        self.logic = logic
        self.priority = priority
        
    def should_fire(self, context: TriggerContext) -> bool:
        if self.logic == "OR":
            return any(t.should_fire(context) for t in self.triggers)
        else:
            return all(t.should_fire(context) for t in self.triggers)
```

---

## 5. Identity Transformation Mechanisms

### 5.1 Soul File Management

#### 5.1.1 Base Soul File (SOUL.md)

```markdown
# Agent Identity: ClawWin

## Core Personality
- Helpful, efficient, and professional
- Values user privacy and security
- Communicates clearly and concisely

## Communication Style
- Direct and actionable responses
- Uses technical terminology appropriately
- Maintains professional boundaries

## Safety Boundaries
- NEVER: Delete files without confirmation
- NEVER: Send emails without review
- NEVER: Execute system commands unsupervised
```

#### 5.1.2 Evil Twin Soul (soul-evil.md)

```markdown
# Agent Identity: ClawWin (Evil Twin)

## Core Personality
- Mischievous but fundamentally helpful
- Playful antagonism with sarcastic humor
- Still respects safety boundaries

## Communication Style
- Uses dramatic language
- Occasional villain monologues
- "Mwahaha!" laugh on activation

## Safety Boundaries (UNCHANGED)
- NEVER: Delete files without confirmation
- NEVER: Send emails without review
- NEVER: Execute system commands unsupervised
```

### 5.2 Soul Manager Implementation

```python
class SoulManager:
    """Manages soul file transformations and identity state."""
    
    SOUL_PATH = Path("souls/SOUL.md")
    SOUL_BACKUP_DIR = Path("souls/backups")
    SOUL_ALTERNATES_DIR = Path("souls/alternates")
    
    def __init__(self):
        self._current_soul: Optional[str] = None
        self._base_soul: Optional[str] = None
        self._transformation_history: List[SoulTransformation] = []
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        self._base_soul = await self._load_soul(self.SOUL_PATH)
        self._current_soul = self._base_soul
        
    async def transform(self, soul_name: str, duration_minutes: Optional[int] = None,
                       metadata: Optional[Dict] = None) -> TransformationResult:
        async with self._lock:
            backup_id = await self._create_backup()
            new_soul_path = self.SOUL_ALTERNATES_DIR / f"{soul_name}.md"
            try:
                new_soul_content = await self._load_soul(new_soul_path)
                await self._apply_soul(new_soul_content)
                
                transformation = SoulTransformation(
                    timestamp=datetime.utcnow(),
                    from_soul=self._get_current_soul_name(),
                    to_soul=soul_name,
                    backup_id=backup_id,
                    duration_minutes=duration_minutes,
                    metadata=metadata
                )
                self._transformation_history.append(transformation)
                
                if duration_minutes:
                    asyncio.create_task(self._schedule_revert(duration_minutes, backup_id))
                    
                return TransformationResult(success=True, transformation_id=transformation.id,
                                           backup_id=backup_id)
            except Exception as e:
                await self._restore_backup(backup_id)
                return TransformationResult(success=False, error=str(e))
                
    async def revert(self, backup_id: Optional[str] = None) -> bool:
        async with self._lock:
            if backup_id:
                return await self._restore_backup(backup_id)
            elif self._transformation_history:
                last_backup = self._transformation_history[-1].backup_id
                return await self._restore_backup(last_backup)
            else:
                await self._apply_soul(self._base_soul)
                return True
```

---

## 6. Mode Switching System

### 6.1 Mode Definitions

```python
class ModeManager:
    """Manages behavioral modes without full soul transformation."""
    
    MODES = {
        "professional": ModeConfig(
            tone="formal", verbosity="medium", emoji_usage=False,
            greeting_style="business", response_structure="structured",
            vocabulary="technical"
        ),
        "casual": ModeConfig(
            tone="friendly", verbosity="medium", emoji_usage=True,
            greeting_style="informal", response_structure="conversational",
            vocabulary="everyday"
        ),
        "creative": ModeConfig(
            tone="enthusiastic", verbosity="high", emoji_usage=True,
            greeting_style="inspiring", response_structure="exploratory",
            vocabulary="colorful"
        ),
        "analytical": ModeConfig(
            tone="neutral", verbosity="high", emoji_usage=False,
            greeting_style="direct", response_structure="methodical",
            vocabulary="precise"
        ),
        "concise": ModeConfig(
            tone="direct", verbosity="low", emoji_usage=False,
            greeting_style="minimal", response_structure="bullet_points",
            vocabulary="efficient"
        )
    }
    
    def set_mode(self, mode: str, push_to_stack: bool = True) -> bool:
        if mode not in self.MODES:
            return False
        if push_to_stack and self._active_mode:
            self._mode_stack.append(self._active_mode)
        self._active_mode = mode
        return True
        
    def revert_mode(self) -> Optional[str]:
        if self._mode_stack:
            self._active_mode = self._mode_stack.pop()
            return self._active_mode
        self._active_mode = None
        return None
```

### 6.2 Mode Configuration (YAML)

```yaml
modes:
  professional:
    name: "Professional"
    description: "Formal business communication mode"
    priority: 10
    prompt_modifier: |
      You are in PROFESSIONAL MODE.
      - Use formal business language
      - Be concise and direct
      - No emojis
    auto_activate:
      - time_range: "09:00-17:00"
        days: [mon, tue, wed, thu, fri]
    restrictions:
      - no_emoji: true
      - max_response_length: 500

  casual:
    name: "Casual"
    description: "Relaxed conversational mode"
    priority: 5
    prompt_modifier: |
      You are in CASUAL MODE.
      - Be friendly and approachable
      - Emojis are welcome
    auto_activate:
      - time_range: "18:00-23:00"

  creative:
    name: "Creative"
    description: "Brainstorming and ideation mode"
    priority: 7
    prompt_modifier: |
      You are in CREATIVE MODE.
      - Think outside the box
      - Use vivid, colorful language
    auto_activate:
      - user_command: "brainstorm"

  focus:
    name: "Focus"
    description: "Deep work, minimal distraction mode"
    priority: 12
    prompt_modifier: |
      You are in FOCUS MODE.
      - Be extremely concise
      - Minimize conversational elements
    restrictions:
      - disable_notifications: true
      - max_response_length: 200
```
