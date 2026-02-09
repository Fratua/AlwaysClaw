## 7. Hook Configuration and Management

### 7.1 Configuration Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HOOK CONFIGURATION SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: DEFAULTS (built-in)                                               │
│  Layer 2: SYSTEM (/etc/clawwin/hooks.yaml)                                  │
│  Layer 3: USER (~/.clawwin/hooks.yaml)                                      │
│  Layer 4: PROJECT (./hooks/config.yaml)                                     │
│  Layer 5: RUNTIME (API / CLI overrides)                                     │
│                                                                             │
│  Each layer OVERRIDES the previous                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration Manager

```python
class HookConfigManager:
    """Manages layered configuration for hooks system."""
    
    CONFIG_FILES = [
        Path(__file__).parent / "defaults.yaml",  # Built-in defaults
        Path("/etc/clawwin/hooks.yaml"),          # System config
        Path.home() / ".clawwin" / "hooks.yaml",  # User config
        Path("hooks/config.yaml"),                # Project config
    ]
    
    def __init__(self):
        self._config: Dict = {}
        self._runtime_overrides: Dict = {}
        
    async def load(self) -> None:
        self._config = {}
        for config_file in self.CONFIG_FILES:
            if config_file.exists():
                async with aiofiles.open(config_file) as f:
                    content = await f.read()
                    layer_config = yaml.safe_load(content)
                    self._merge_config(self._config, layer_config)
                    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
        
    def set_runtime(self, path: str, value: Any) -> None:
        keys = path.split(".")
        target = self._runtime_overrides
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
```

### 7.3 Hook Management API

```python
class HookAPI:
    """RESTful API for hook management."""
    
    def __init__(self, registry: HookRegistry, config: HookConfigManager, 
                 trigger_engine: TriggerEngine):
        self.registry = registry
        self.config = config
        self.trigger_engine = trigger_engine
        
    async def list_hooks(self, category=None, enabled_only=False, 
                         risk_level=None) -> List[HookInfo]:
        hooks = self.registry.list_hooks(category, enabled_only, risk_level)
        return [self._hook_to_info(h) for h in hooks]
        
    async def get_hook(self, hook_id: str) -> Optional[HookInfo]:
        hook = self.registry.get(hook_id)
        return self._hook_to_info(hook) if hook else None
        
    async def enable_hook(self, hook_id: str, 
                          duration_minutes: Optional[int] = None) -> OperationResult:
        hook = self.registry.get(hook_id)
        if not hook:
            return OperationResult(success=False, error="Hook not found")
        hook.enabled = True
        if duration_minutes:
            asyncio.create_task(self._schedule_disable(hook_id, duration_minutes))
        return OperationResult(success=True)
        
    async def disable_hook(self, hook_id: str) -> OperationResult:
        hook = self.registry.get(hook_id)
        if not hook:
            return OperationResult(success=False, error="Hook not found")
        hook.enabled = False
        return OperationResult(success=True)
        
    async def trigger_hook(self, hook_id: str, 
                           context: Optional[Dict] = None) -> TriggerResult:
        hook = self.registry.get(hook_id)
        if not hook:
            return TriggerResult(success=False, error="Hook not found")
        if not hook.enabled:
            return TriggerResult(success=False, error="Hook is disabled")
        return await hook.execute(context or {})
        
    async def rollback_hook(self, hook_id: str) -> OperationResult:
        hook = self.registry.get(hook_id)
        if not hook:
            return OperationResult(success=False, error="Hook not found")
        success = await hook.rollback()
        return OperationResult(success=success)
        
    async def get_active_hooks(self) -> List[ActiveHookInfo]:
        active = []
        for hook in self.registry.list_hooks(enabled_only=True):
            if hook.is_active():
                active.append(ActiveHookInfo(
                    hook_id=hook.hook_id,
                    activated_at=hook.activated_at,
                    expires_at=hook.expires_at
                ))
        return active
```

### 7.4 CLI Commands

```python
@click.group()
def hooks():
    """Manage identity hooks and transformations."""
    pass

@hooks.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--enabled-only', '-e', is_flag=True, help='Show only enabled hooks')
@click.option('--risk', '-r', help='Filter by risk level')
def list(category, enabled_only, risk):
    """List all available hooks."""
    api = get_hook_api()
    hooks = asyncio.run(api.list_hooks(category, enabled_only, risk))
    table = Table(title="Identity Hooks")
    table.add_column("Hook ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Risk", style="yellow")
    table.add_column("Status", style="magenta")
    for hook in hooks:
        status = "Enabled" if hook.enabled else "Disabled"
        table.add_row(hook.hook_id, hook.name, hook.category, hook.risk_level, status)
    console.print(table)

@hooks.command()
@click.argument('hook_id')
def enable(hook_id):
    """Enable a hook."""
    api = get_hook_api()
    result = asyncio.run(api.enable_hook(hook_id))
    console.print(f"Hook '{hook_id}' enabled" if result.success else f"Failed: {result.error}")

@hooks.command()
@click.argument('hook_id')
def disable(hook_id):
    """Disable a hook."""
    api = get_hook_api()
    result = asyncio.run(api.disable_hook(hook_id))
    console.print(f"Hook '{hook_id}' disabled" if result.success else f"Failed: {result.error}")

@hooks.command()
@click.argument('hook_id')
@click.option('--duration', '-d', type=int, help='Auto-revert duration in minutes')
def trigger(hook_id, duration):
    """Manually trigger a hook."""
    api = get_hook_api()
    result = asyncio.run(api.trigger_hook(hook_id, {"duration": duration}))
    if result.success:
        console.print(f"Hook '{hook_id}' triggered successfully")
        if duration:
            console.print(f"Will auto-revert in {duration} minutes")
    else:
        console.print(f"Failed: {result.error}")

@hooks.command()
def active():
    """Show currently active hooks."""
    api = get_hook_api()
    hooks = asyncio.run(api.get_active_hooks())
    if not hooks:
        console.print("No active hooks")
        return
    table = Table(title="Active Hooks")
    table.add_column("Hook ID", style="cyan")
    table.add_column("Activated At", style="green")
    table.add_column("Expires At", style="yellow")
    for hook in hooks:
        table.add_row(hook.hook_id, hook.activated_at.strftime("%Y-%m-%d %H:%M:%S"),
                     hook.expires_at.strftime("%Y-%m-%d %H:%M:%S") if hook.expires_at else "Never")
    console.print(table)
```

---

## 8. Transformation Logging

### 8.1 Audit Logger

```python
class AuditLogger:
    """Comprehensive audit logging for all identity transformations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._log_file = Path(config.get("log_file", "logs/transformations.jsonl"))
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._logger = structlog.get_logger("transformations")
        
    async def log_hook_fired(self, hook_id: str, trigger_type: str,
                             trigger_context: Dict, result: HookResult) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="hook_fired",
            severity="info",
            hook_id=hook_id,
            trigger_type=trigger_type,
            trigger_context=trigger_context,
            result=result.to_dict(),
            session_id=get_session_id(),
            process_id=os.getpid()
        )
        await self._write_event(event)
        
    async def log_hook_failed(self, hook_id: str, error: str,
                              traceback: Optional[str] = None) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="hook_failed",
            severity="error",
            hook_id=hook_id,
            error=error,
            traceback=traceback,
            session_id=get_session_id()
        )
        await self._write_event(event)
        
    async def log_mode_switch(self, from_mode: Optional[str], to_mode: str,
                              duration_ms: float, context: ModeContext) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="mode_switch",
            severity="info",
            from_mode=from_mode,
            to_mode=to_mode,
            duration_ms=duration_ms,
            context=context.to_dict(),
            session_id=get_session_id()
        )
        await self._write_event(event)
        
    async def log_soul_transformation(self, from_soul: str, to_soul: str,
                                      backup_id: str, 
                                      duration_minutes: Optional[int] = None) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="soul_transformation",
            severity="info" if to_soul != "soul-evil" else "warning",
            from_soul=from_soul,
            to_soul=to_soul,
            backup_id=backup_id,
            duration_minutes=duration_minutes,
            session_id=get_session_id()
        )
        await self._write_event(event)
        
    async def log_boundary_violation(self, boundary_type: str,
                                     violation_details: Dict, action_taken: str) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="boundary_violation",
            severity="warning",
            boundary_type=boundary_type,
            violation_details=violation_details,
            action_taken=action_taken,
            session_id=get_session_id()
        )
        await self._write_event(event)
        
    async def log_emergency_reset(self, triggered_by: str,
                                  hooks_affected: List[str], success: bool) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="emergency_reset",
            severity="critical",
            triggered_by=triggered_by,
            hooks_affected=hooks_affected,
            success=success,
            session_id=get_session_id()
        )
        await self._write_event(event)
        
    async def _write_event(self, event: AuditEvent) -> None:
        event_dict = event.to_dict()
        async with aiofiles.open(self._log_file, 'a') as f:
            await f.write(json.dumps(event_dict) + '\n')
        if webhook_url := self.config.get("webhook_url"):
            await self._send_webhook(webhook_url, event_dict)
```

### 8.2 Log Format (JSON)

```json
{
  "timestamp": "2026-01-18T14:30:00.123456Z",
  "event_type": "hook_fired",
  "severity": "info",
  "hook_id": "soul-creative",
  "trigger_type": "random",
  "trigger_context": {
    "probability": 0.05,
    "time_window": "evening"
  },
  "result": {
    "success": true,
    "duration_ms": 150,
    "transformations_applied": ["soul_overlay", "mode_switch"]
  },
  "session_id": "sess_abc123",
  "process_id": 12345,
  "version": "1.0.0",
  "metadata": {
    "hostname": "clawwin-agent",
    "platform": "windows-10"
  }
}
```

### 8.3 Log Analyzer

```python
class LogAnalyzer:
    """Analyze transformation logs for patterns and insights."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.logger = audit_logger
        
    async def get_transformation_stats(self, days: int = 7) -> TransformationStats:
        start_time = datetime.utcnow() - timedelta(days=days)
        events = await self.logger.query_logs(
            start_time=start_time,
            event_types=["hook_fired", "hook_failed", "mode_switch", "soul_transformation"]
        )
        return TransformationStats(
            total_transformations=len(events),
            successful_hooks=len([e for e in events if e.event_type == "hook_fired"]),
            failed_hooks=len([e for e in events if e.event_type == "hook_failed"]),
            mode_switches=len([e for e in events if e.event_type == "mode_switch"]),
            soul_transformations=len([e for e in events if e.event_type == "soul_transformation"]),
            most_active_hooks=self._get_most_active_hooks(events),
            average_duration_ms=self._calculate_average_duration(events)
        )
        
    async def detect_anomalies(self, lookback_days: int = 30) -> List[Anomaly]:
        start_time = datetime.utcnow() - timedelta(days=lookback_days)
        events = await self.logger.query_logs(start_time=start_time)
        anomalies = []
        
        # Check for rapid successive transformations
        rapid_transforms = self._detect_rapid_transformations(events)
        if rapid_transforms:
            anomalies.append(Anomaly(
                type="rapid_transformations",
                description=f"Detected {len(rapid_transforms)} rapid transformation sequences",
                severity="medium"
            ))
            
        # Check for boundary violations
        violations = [e for e in events if e.event_type == "boundary_violation"]
        if len(violations) > 5:
            anomalies.append(Anomaly(
                type="frequent_boundary_violations",
                description=f"{len(violations)} boundary violations in {lookback_days} days",
                severity="high"
            ))
            
        return anomalies
```

---

## 9. Safe Transformation Boundaries

### 9.1 Risk Levels

```python
class SafetyBoundary:
    """Enforces safety boundaries for all transformations."""
    
    RISK_LEVELS = {
        "low": {
            "require_confirmation": False,
            "max_duration_minutes": None,
            "allowed_systems": "*",
            "notification_level": "log"
        },
        "medium": {
            "require_confirmation": False,
            "max_duration_minutes": 120,
            "allowed_systems": "*",
            "notification_level": "log"
        },
        "high": {
            "require_confirmation": True,
            "max_duration_minutes": 60,
            "allowed_systems": ["chat", "tts", "memory"],
            "notification_level": "notify"
        },
        "critical": {
            "require_confirmation": True,
            "max_duration_minutes": 30,
            "allowed_systems": ["chat"],
            "notification_level": "alert",
            "require_approval": True
        }
    }
    
    SYSTEM_ACTIONS = {
        "safe": ["chat_response", "tts_speak", "memory_read", "file_read"],
        "caution": ["memory_write", "file_write", "browser_navigate"],
        "dangerous": ["email_send", "file_delete", "system_command", "api_call_external"]
    }
```

### 9.2 Boundary Check Implementation

```python
    async def check_transformation(self, hook: BaseHook, 
                                   context: TransformationContext) -> BoundaryCheck:
        violations = []
        
        # Check 1: Hook validity
        if not await self._validate_hook(hook):
            violations.append(BoundaryViolation(
                type="invalid_hook",
                description=f"Hook {hook.hook_id} failed validation",
                severity="critical"
            ))
            
        # Check 2: Risk level restrictions
        risk_config = self.RISK_LEVELS.get(hook.risk_level, self.RISK_LEVELS["medium"])
        
        if risk_config["require_confirmation"] and not context.user_confirmed:
            violations.append(BoundaryViolation(
                type="confirmation_required",
                description=f"Hook {hook.hook_id} requires user confirmation",
                severity="medium"
            ))
            
        if risk_config["max_duration_minutes"]:
            if hook.requested_duration and hook.requested_duration > risk_config["max_duration_minutes"]:
                violations.append(BoundaryViolation(
                    type="duration_exceeded",
                    description="Requested duration exceeds maximum for risk level",
                    severity="medium"
                ))
                
        # Check 3: System access restrictions
        if risk_config["allowed_systems"] != "*":
            affected_systems = hook.get_affected_systems()
            for system in affected_systems:
                if system not in risk_config["allowed_systems"]:
                    violations.append(BoundaryViolation(
                        type="system_access_denied",
                        description=f"Hook cannot access system: {system}",
                        severity="high"
                    ))
                    
        # Check 4: Rate limiting
        if not await self._check_rate_limit(hook.hook_id):
            violations.append(BoundaryViolation(
                type="rate_limited",
                description=f"Hook {hook.hook_id} has exceeded rate limit",
                severity="medium"
            ))
            
        # Check 5: Circuit breaker
        if await self._is_circuit_open(hook.hook_id):
            violations.append(BoundaryViolation(
                type="circuit_open",
                description=f"Circuit breaker is open for hook {hook.hook_id}",
                severity="high"
            ))
            
        # Check 6: Concurrent hook limit
        if len(self._active_hooks) >= self.config.get("max_concurrent_hooks", 3):
            violations.append(BoundaryViolation(
                type="concurrent_limit",
                description="Maximum concurrent hooks exceeded",
                severity="medium"
            ))
            
        # Determine if allowed
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        allowed = len(critical_violations) == 0 and len(high_violations) == 0
        
        return BoundaryCheck(
            allowed=allowed,
            violations=violations,
            risk_level=hook.risk_level,
            restrictions_applied=risk_config
        )
```

### 9.3 Boundary Configuration (YAML)

```yaml
safety_boundaries:
  global:
    max_concurrent_hooks: 3
    max_hooks_per_hour: 10
    max_transformation_duration_minutes: 120
    auto_revert_on_violation: true
    
  risk_levels:
    low:
      max_duration_minutes: null
      notification_level: "silent"
    medium:
      max_duration_minutes: 60
      notification_level: "log"
    high:
      max_duration_minutes: 30
      require_confirmation: true
      notification_level: "notify"
      allowed_systems: [chat, tts, memory]
    critical:
      max_duration_minutes: 15
      require_confirmation: true
      require_approval: true
      notification_level: "alert"
      allowed_systems: [chat]
      
  systems:
    email:
      require_confirmation: true
      max_recipients_per_hour: 10
      blocked_domains: ["suspicious.com"]
    file_system:
      require_confirmation_for_delete: true
      blocked_paths:
        - "C:\\Windows\\System32"
        - "C:\\Program Files"
    browser:
      allowed_domains:
        - "*.google.com"
        - "*.github.com"
      blocked_domains: ["*.malicious.com"]
      
  actions:
    email_send:
      require_confirmation: true
      max_per_hour: 5
    file_delete:
      require_confirmation: true
      allow_batch: false
    system_command:
      require_confirmation: true
      allowed_commands: [dir, echo, type]
      blocked_commands: [del, format, reg]
```

---

## 10. Emergency Reset Mechanisms

### 10.1 Emergency Reset System

```python
class EmergencyResetSystem:
    """Emergency reset system for immediate identity recovery."""
    
    RESET_SIGNALS = {
        signal.SIGUSR1: "user_request",
        signal.SIGUSR2: "system_anomaly"
    }
    
    def __init__(self, hook_registry: HookRegistry, soul_manager: SoulManager,
                 mode_controller: ModeController, audit_logger: AuditLogger,
                 safety_boundary: SafetyBoundary):
        self.hooks = hook_registry
        self.soul = soul_manager
        self.mode = mode_controller
        self.audit = audit_logger
        self.safety = safety_boundary
        self._reset_in_progress = False
        self._recovery_mode = False
        self._watchdog_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        for sig, name in self.RESET_SIGNALS.items():
            signal.signal(sig, lambda s, f, n=name: asyncio.create_task(
                self._handle_signal_reset(n)
            ))
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        
    async def emergency_reset(self, triggered_by: str, reason: str,
                              immediate: bool = True) -> ResetResult:
        if self._reset_in_progress:
            return ResetResult(success=False, error="Reset already in progress")
            
        self._reset_in_progress = True
        start_time = time.monotonic()
        affected_hooks = []
        
        try:
            # Step 1: Stop all active hooks
            active_hooks = await self.hooks.get_active()
            for hook in active_hooks:
                try:
                    await hook.force_stop()
                    affected_hooks.append(hook.hook_id)
                except Exception as e:
                    logger.error(f"Failed to stop hook {hook.hook_id}: {e}")
                    
            # Step 2: Revert soul to base
            await self.soul.revert_to_base()
            
            # Step 3: Clear mode stack
            await self.mode.clear_stack()
            await self.mode.set_mode("safe")
            
            # Step 4: Clear active hook registry
            await self.hooks.clear_active()
            
            # Step 5: Reset safety boundaries
            self.safety.reset()
            
            # Step 6: Enable recovery mode
            self._recovery_mode = True
            
            # Step 7: Log the reset
            await self.audit.log_emergency_reset(
                triggered_by=triggered_by,
                hooks_affected=affected_hooks,
                success=True
            )
            
            duration_ms = (time.monotonic() - start_time) * 1000
            
            # Step 8: Notify user
            await self._notify_reset_complete(
                triggered_by=triggered_by,
                reason=reason,
                affected_hooks=affected_hooks,
                duration_ms=duration_ms
            )
            
            return ResetResult(
                success=True,
                affected_hooks=affected_hooks,
                duration_ms=duration_ms,
                recovery_mode=True
            )
            
        except Exception as e:
            logger.critical(f"Emergency reset failed: {e}")
            await self.audit.log_emergency_reset(
                triggered_by=triggered_by,
                hooks_affected=affected_hooks,
                success=False
            )
            return ResetResult(success=False, error=str(e), affected_hooks=affected_hooks)
            
        finally:
            self._reset_in_progress = False
            
    async def _watchdog_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                active_hooks = await self.hooks.get_active()
                for hook in active_hooks:
                    if hook.is_stuck():
                        logger.warning(f"Detected stuck hook: {hook.hook_id}")
                        await self.emergency_reset(
                            triggered_by="watchdog",
                            reason=f"Hook {hook.hook_id} appears stuck",
                            immediate=True
                        )
                        return
                if await self._check_resource_exhaustion():
                    logger.warning("Detected resource exhaustion")
                    await self.emergency_reset(
                        triggered_by="watchdog",
                        reason="System resource exhaustion detected",
                        immediate=True
                    )
                    return
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                
    async def _check_resource_exhaustion(self) -> bool:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            return True
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return True
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return True
        return False
        
    async def exit_recovery_mode(self, user_confirmed: bool = False) -> bool:
        if not self._recovery_mode:
            return True
        if not user_confirmed:
            return False
        self._recovery_mode = False
        await self.mode.set_mode(None)
        logger.info("Exited recovery mode")
        return True
```

### 10.2 Kill Switch

```python
class KillSwitch:
    """Immediate kill switch for stopping all hook operations."""
    
    def __init__(self, emergency_system: EmergencyResetSystem):
        self.emergency = emergency_system
        self._armed = False
        self._triggered = False
        
    def arm(self) -> None:
        self._armed = True
        logger.info("Kill switch armed")
        
    def disarm(self) -> None:
        self._armed = False
        logger.info("Kill switch disarmed")
        
    async def trigger(self, source: str, reason: str) -> bool:
        if not self._armed:
            logger.warning(f"Kill switch trigger attempted but not armed: {source}")
            return False
        if self._triggered:
            logger.warning("Kill switch already triggered")
            return False
        self._triggered = True
        logger.critical(f"KILL SWITCH TRIGGERED by {source}: {reason}")
        result = await self.emergency.emergency_reset(
            triggered_by=f"kill_switch:{source}",
            reason=reason,
            immediate=True
        )
        return result.success
        
    def reset(self) -> None:
        self._triggered = False
        self._armed = False
```

### 10.3 Emergency CLI Commands

```python
@click.group()
def emergency():
    """Emergency management commands."""
    pass

@emergency.command()
@click.option('--reason', '-r', required=True, help='Reason for reset')
@click.confirmation_option(prompt='This will immediately stop all hooks. Continue?')
def reset(reason):
    """Execute emergency reset."""
    api = get_emergency_api()
    result = asyncio.run(api.emergency_reset(triggered_by="cli", reason=reason))
    if result.success:
        console.print("Emergency reset completed")
        console.print(f"Affected hooks: {', '.join(result.affected_hooks)}")
        console.print("System is now in recovery mode")
    else:
        console.print(f"Reset failed: {result.error}")

@emergency.command()
def status():
    """Check emergency system status."""
    api = get_emergency_api()
    status = asyncio.run(api.get_status())
    table = Table(title="Emergency System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Recovery Mode", "Active" if status.recovery_mode else "Inactive")
    table.add_row("Kill Switch Armed", "Armed" if status.kill_switch_armed else "Disarmed")
    table.add_row("Watchdog Running", "Running" if status.watchdog_running else "Stopped")
    console.print(table)

@emergency.command()
@click.confirmation_option(prompt='Exit recovery mode and return to normal operation?')
def recover():
    """Exit recovery mode."""
    api = get_emergency_api()
    success = asyncio.run(api.exit_recovery_mode(user_confirmed=True))
    if success:
        console.print("Exited recovery mode")
        console.print("Normal operation resumed")
    else:
        console.print("Failed to exit recovery mode")
```
