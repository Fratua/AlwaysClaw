## 11. Implementation Reference

### 11.1 Project Structure

```
clawwin/
├── hooks/
│   ├── __init__.py
│   ├── base.py                 # BaseHook class
│   ├── registry.py             # HookRegistry
│   ├── loader.py               # Hook loading utilities
│   ├── souls/                  # Soul file hooks
│   │   ├── soul-evil.yaml
│   │   ├── soul-creative.yaml
│   │   └── soul-focused.yaml
│   └── modes/                  # Mode hooks
│       ├── professional.yaml
│       ├── casual.yaml
│       └── creative.yaml
│
├── triggers/
│   ├── __init__.py
│   ├── base.py                 # BaseTrigger class
│   ├── time_triggers.py        # Cron, interval triggers
│   ├── event_triggers.py       # Event-based triggers
│   ├── random_triggers.py      # Random/probability triggers
│   ├── hybrid_triggers.py      # Combined triggers
│   └── engine.py               # TriggerEngine
│
├── identity/
│   ├── __init__.py
│   ├── soul_manager.py         # Soul file management
│   ├── mode_controller.py      # Mode switching
│   └── state_machine.py        # Transformation state machine
│
├── safety/
│   ├── __init__.py
│   ├── boundary.py             # SafetyBoundary
│   ├── circuit_breaker.py      # Circuit breaker pattern
│   └── rate_limiter.py         # Rate limiting
│
├── emergency/
│   ├── __init__.py
│   ├── reset_system.py         # EmergencyResetSystem
│   └── kill_switch.py          # KillSwitch
│
├── logging/
│   ├── __init__.py
│   ├── audit_logger.py         # AuditLogger
│   └── log_analyzer.py         # LogAnalyzer
│
├── config/
│   ├── __init__.py
│   ├── hook_config.py          # HookConfigManager
│   └── defaults.yaml           # Default configuration
│
├── api/
│   ├── __init__.py
│   └── hook_api.py             # RESTful API
│
├── cli/
│   ├── __init__.py
│   ├── hooks.py                # Hook CLI commands
│   └── emergency.py            # Emergency CLI commands
│
└── souls/                      # Soul files
    ├── SOUL.md                 # Base identity
    ├── backups/                # Automatic backups
    └── alternates/             # Alternate souls
        ├── soul-evil.md
        └── soul-creative.md
```

### 11.2 Key Dependencies

```txt
# requirements.txt

# Core
pyyaml>=6.0
structlog>=23.0.0
python-json-logger>=2.0.0

# Scheduling
apscheduler>=3.10.0
croniter>=1.4.0
pytz>=2023.3

# Async
aiofiles>=23.0.0
aiocron>=1.8

# Safety
psutil>=5.9.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0

# CLI
click>=8.1.0
rich>=13.0.0

# Validation
pydantic>=2.0.0
```

### 11.3 Configuration Example (config.yaml)

```yaml
hooks:
  # Global settings
  global:
    enabled: true
    default_mode: "professional"
    max_concurrent_hooks: 3
    max_hooks_per_hour: 10
    audit_level: "detailed"
    
  # Registry configuration
  registry:
    auto_discover: true
    paths:
      - "hooks/"
      - "~/.clawwin/hooks/"
    allow_remote_hooks: false
    
  # Trigger engine settings
  triggers:
    check_interval_seconds: 60
    max_pending_triggers: 100
    enable_random_triggers: true
    
  # Safety boundaries
  safety:
    require_confirmation_for: ["high", "critical"]
    max_transformation_duration_minutes: 120
    auto_revert_on_error: true
    auto_revert_on_violation: true
    
  # Emergency settings
  emergency:
    watchdog_interval_seconds: 30
    resource_threshold_memory: 95
    resource_threshold_cpu: 95
    enable_kill_switch: true
    
  # Logging configuration
  logging:
    log_file: "logs/transformations.jsonl"
    max_file_size_mb: 10
    backup_count: 10
    syslog_enabled: false
    webhook_url: null
    
# Hook-specific configurations
hook_configs:
  soul-evil:
    enabled: false
    auto_enable: false
    max_duration_minutes: 60
    require_confirmation: true
    
  soul-creative:
    enabled: true
    auto_enable: true
    random_probability: 0.05
    time_window:
      start: 18
      end: 23
      
  mode-professional:
    enabled: true
    auto_activate:
      time_range: "09:00-17:00"
      days: [mon, tue, wed, thu, fri]
```

### 11.4 Usage Example

```python
# examples/basic_usage.py

import asyncio
from clawwin.hooks import HookRegistry, HookManager
from clawwin.triggers import TriggerEngine
from clawwin.identity import SoulManager, ModeController
from clawwin.emergency import EmergencyResetSystem

async def main():
    # Initialize components
    registry = HookRegistry("hooks/")
    await registry.discover_hooks()
    
    soul_manager = SoulManager()
    await soul_manager.initialize()
    
    mode_controller = ModeController(soul_manager)
    
    emergency = EmergencyResetSystem(
        registry, soul_manager, mode_controller
    )
    await emergency.initialize()
    
    trigger_engine = TriggerEngine(registry)
    
    # Start trigger monitoring
    await trigger_engine.start()
    
    # Example: Manually trigger a hook
    hook = registry.get("soul-creative")
    if hook:
        result = await hook.execute()
        print(f"Hook executed: {result.success}")
        
    # Example: Switch mode
    await mode_controller.switch_mode("professional")
    
    # Example: Emergency reset
    # await emergency.emergency_reset("manual", "User requested reset")
    
    # Keep running
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 12. Appendices

### Appendix A: Hook ID Reference

| Hook ID | Category | Risk Level | Description |
|---------|----------|------------|-------------|
| `soul-evil` | soul | medium | Mischievous alter ego |
| `soul-creative` | soul | low | Enhanced creativity mode |
| `soul-focused` | soul | low | Deep work mode |
| `soul-mentor` | soul | low | Teaching persona |
| `soul-debug` | soul | low | Technical debugging mode |
| `mode-professional` | mode | low | Formal business communication |
| `mode-casual` | mode | low | Relaxed conversational style |
| `mode-creative` | mode | low | Brainstorming mode |
| `mode-analytical` | mode | low | Data-driven responses |
| `mode-concise` | mode | low | Minimal responses |
| `system-quiet` | system | low | Reduced notifications |
| `system-aggressive` | system | medium | Increased parallelism |

### Appendix B: Event Type Reference

| Event Type | Description | Payload |
|------------|-------------|---------|
| `user_message` | User sent a message | `{message, timestamp}` |
| `user_idle` | User inactive | `{duration_minutes}` |
| `user_return` | User returned after idle | `{away_duration}` |
| `user_command` | User issued command | `{command, args}` |
| `system_startup` | Agent started | `{version, config}` |
| `system_shutdown` | Agent shutting down | `{reason}` |
| `system_error` | Error occurred | `{error, traceback}` |
| `task_started` | Task began execution | `{task_id, type}` |
| `task_completed` | Task finished | `{task_id, result}` |
| `task_failed` | Task failed | `{task_id, error}` |
| `email_received` | New email arrived | `{sender, subject}` |
| `calendar_event` | Calendar event triggered | `{event_id, title}` |

### Appendix C: Safety Violation Codes

| Code | Description | Severity | Action |
|------|-------------|----------|--------|
| `invalid_hook` | Hook failed validation | critical | Block |
| `confirmation_required` | User confirmation needed | medium | Prompt |
| `duration_exceeded` | Requested duration too long | medium | Cap |
| `system_access_denied` | Hook cannot access system | high | Block |
| `rate_limited` | Hook rate limit exceeded | medium | Delay |
| `circuit_open` | Circuit breaker open | high | Block |
| `concurrent_limit` | Too many concurrent hooks | medium | Queue |

### Appendix D: API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/hooks` | List all hooks |
| GET | `/api/v1/hooks/{id}` | Get hook details |
| POST | `/api/v1/hooks/{id}/enable` | Enable hook |
| POST | `/api/v1/hooks/{id}/disable` | Disable hook |
| POST | `/api/v1/hooks/{id}/trigger` | Trigger hook |
| POST | `/api/v1/hooks/{id}/rollback` | Rollback hook |
| GET | `/api/v1/hooks/active` | List active hooks |
| GET | `/api/v1/modes` | List available modes |
| POST | `/api/v1/modes/{id}/activate` | Activate mode |
| POST | `/api/v1/modes/revert` | Revert to previous mode |
| POST | `/api/v1/emergency/reset` | Emergency reset |
| GET | `/api/v1/emergency/status` | Emergency status |
| GET | `/api/v1/logs` | Query transformation logs |
| GET | `/api/v1/stats` | Get transformation stats |

### Appendix E: State Machine Diagram

```
                    Trigger Fired
                         │
                         ▼
    ┌─────────────────────────────────────┐
    │            VALIDATING               │
    │         (Check hook valid)          │
    └───────────────┬─────────────────────┘
                    │ Validation Passed
                    ▼
    ┌─────────────────────────────────────┐
    │            BACKING UP               │
    │        (Save current state)         │
    └───────────────┬─────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │          TRANSFORMING               │◄────────┐
    │       (Apply new identity)          │         │
    └───────────────┬─────────────────────┘         │
                    │                               │
        ┌───────────┴───────────┐                   │
        │                       │                   │
        ▼                       ▼                   │
   ┌─────────┐            ┌──────────┐              │
   │ FAILED  │            │  ACTIVE  │──────────────┘
   │(Revert) │            │(Running) │ Auto-revert
   └─────────┘            └──────────┘
```

### Appendix F: Transformation Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Trigger │───▶│ Validate │───▶│  Backup  │───▶│ Transform│
│  Fired   │    │   Hook   │    │  State   │    │  State   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
     └───────────────┴───────────────┴───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      AUDIT LOG      │
              │ [Timestamp][Result] │
              └─────────────────────┘
```

---

## Document Information

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Author** | AI Systems Architect |
| **Date** | 2026-01-18 |
| **Status** | Technical Specification |
| **Classification** | Architecture Design |

---

## Summary

This technical specification provides a comprehensive design for the Identity Hooks and Transformation Trigger System for a Windows 10 OpenClaw-inspired AI agent. The system includes:

### Core Components:
1. **Hook System**: Dynamic identity transformation via SOUL.md swapping
2. **Trigger Engine**: Time-based, event-based, random, and hybrid triggers
3. **Identity Manager**: Soul and mode transformation with rollback
4. **Safety Boundary**: Multi-layer protection with risk levels
5. **Audit Logger**: Comprehensive transformation logging
6. **Emergency System**: Kill switches and immediate reset

### Key Features:
- **Soul Hooks**: `soul-evil`, `soul-creative`, `soul-focused`, `soul-mentor`, `soul-debug`
- **Mode Hooks**: `mode-professional`, `mode-casual`, `mode-creative`, `mode-analytical`, `mode-concise`
- **Trigger Types**: Cron scheduled, interval, event-based, random probability, hybrid
- **Safety Levels**: Low, medium, high, critical with appropriate restrictions
- **Emergency Controls**: Signal handlers, watchdog, kill switch, recovery mode

### Architecture Principles:
- Safety first with reversible transformations
- Full observability through structured logging
- Modular, testable hook components
- Extensible design for new hooks and triggers
- Minimal performance overhead

---

*End of Document*
