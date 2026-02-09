# OpenClaw Windows 10 AI Agent - Power Management Technical Specification

## Executive Summary

This document provides comprehensive technical specifications for implementing wake/sleep cycles and power management for a 24/7 operational Windows 10 AI agent system inspired by OpenClaw. The architecture ensures continuous operation while optimizing energy consumption through intelligent power state management.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Windows Power State Management](#windows-power-state-management)
3. [Wake Timer Implementation](#wake-timer-implementation)
4. [Sleep Prevention During Critical Tasks](#sleep-prevention-during-critical-tasks)
5. [Hibernate vs Sleep Decision Matrix](#hibernate-vs-sleep-decision-matrix)
6. [Power Event Handling System](#power-event-handling-system)
7. [Battery-Aware Operation](#battery-aware-operation)
8. [Wake-on-LAN Support](#wake-on-lan-support)
9. [Power Usage Optimization](#power-usage-optimization)
10. [Implementation Code Reference](#implementation-code-reference)

---

## Architecture Overview

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 10 (Pro/Enterprise recommended) |
| Power States Supported | S0, S3, S4, S5 |
| Wake Mechanisms | Timers, Network (WoL), User Input |
| Battery Support | Required for laptops |
| Network | Ethernet preferred for WoL |

### Power Management Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI AGENT POWER MANAGER                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Task Queue  │  │   Scheduler  │  │   Monitor    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Power State Controller                   │      │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │      │
│  │  │  S0     │ │  S3     │ │  S4     │ │  S5     │    │      │
│  │  │ Working │ │ Sleep   │ │Hibernate│ │ SoftOff │    │      │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘    │      │
│  └──────────────────────────────────────────────────────┘      │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Wake Timers  │  │   Battery    │  │   Network    │          │
│  │              │  │   Monitor    │  │   (WoL)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Windows Power State Management

### ACPI Power States Reference

| State | Name | Power Consumption | Wake Latency | Use Case |
|-------|------|-------------------|--------------|----------|
| S0 | Working (Active) | Full | N/A | Normal operation |
| S0ix | Modern Standby | Very Low | < 1s | Background tasks |
| S1 | Power On Suspend | Low | < 2s | Legacy systems |
| S2 | CPU Off | Lower | 2+ seconds | Rarely used |
| S3 | Standby/Sleep | Very Low | 3-10 seconds | Standard sleep |
| S4 | Hibernate | Minimal | 15-45 seconds | Long-term idle |
| S5 | Soft Off | Minimal | Full boot | Shutdown |

### Power State Transitions

```
                    ┌──────────────────────────────────────┐
                    │           S0 (Working)               │
                    │    AI Agent Active Processing        │
                    └───────────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   S3 (Sleep)    │   │   S4 (Hibernate)│   │   S5 (Soft Off) │
    │  Quick Resume   │   │  Save to Disk   │   │  Full Shutdown  │
    │  ~3-10s wake    │   │  ~15-45s wake   │   │  Full boot req  │
    └────────┬────────┘   └────────┬────────┘   └─────────────────┘
             │                     │
             └─────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Wake Events        │
            │  - Timer            │
            │  - Network (WoL)    │
            │  - User Input       │
            │  - Device Events    │
            └─────────────────────┘
```

### Power Policy Configuration

```powershell
# Configure power settings for 24/7 operation
# Run as Administrator

# Set never sleep when plugged in
powercfg /change standby-timeout-ac 0

# Set never turn off display when plugged in
powercfg /change monitor-timeout-ac 0

# Disable hibernate (use sleep instead for quick resume)
powercfg /hibernate off

# Check available sleep states
powercfg /a

# Set active power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable fast startup (for reliable WoL)
# Via Control Panel > Power Options > Choose what power buttons do
```

---

## Wake Timer Implementation

### Architecture

Wake timers allow the system to automatically wake from sleep states at scheduled times, critical for cron jobs and scheduled AI agent tasks.

### Implementation Strategy

```python
"""
Wake Timer Manager for OpenClaw AI Agent
Handles scheduled wake events for 24/7 operation
"""

import ctypes
import ctypes.wintypes
from ctypes import wintypes
from datetime import datetime, timedelta
import threading
import time

# Windows API Constants
CREATE_WAITABLE_TIMER_MANUAL_RESET = 0x00000001
CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
TIMER_ALL_ACCESS = 0x001F0003
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x00000000

# Power Management Constants
ES_AWAYMODE_REQUIRED = 0x00000040
ES_CONTINUOUS = 0x80000000
ES_DISPLAY_REQUIRED = 0x00000002
ES_SYSTEM_REQUIRED = 0x00000001
ES_USER_PRESENT = 0x00000004


class WakeTimerManager:
    """
    Manages wake timers for scheduled AI agent tasks.
    Ensures system wakes for critical cron jobs and scheduled operations.
    """
    
    def __init__(self):
        self.timers = {}
        self.timer_lock = threading.Lock()
        self.kernel32 = ctypes.windll.kernel32
        self._setup_api()
        
    def _setup_api(self):
        """Configure Windows API function signatures"""
        # CreateWaitableTimerExW
        self.kernel32.CreateWaitableTimerExW.argtypes = [
            wintypes.LPVOID,      # lpTimerAttributes
            wintypes.LPCWSTR,     # lpTimerName
            wintypes.DWORD,       # dwFlags
            wintypes.DWORD        # dwDesiredAccess
        ]
        self.kernel32.CreateWaitableTimerExW.restype = wintypes.HANDLE
        
        # SetWaitableTimer
        self.kernel32.SetWaitableTimer.argtypes = [
            wintypes.HANDLE,      # hTimer
            ctypes.POINTER(wintypes.LARGE_INTEGER),  # pDueTime
            wintypes.LONG,        # lPeriod
            wintypes.LPVOID,      # pfnCompletionRoutine
            wintypes.LPVOID,      # lpArgToCompletionRoutine
            wintypes.BOOL         # fResume (CRITICAL: TRUE for wake)
        ]
        self.kernel32.SetWaitableTimer.restype = wintypes.BOOL
        
        # WaitForSingleObject
        self.kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,      # hHandle
            wintypes.DWORD        # dwMilliseconds
        ]
        self.kernel32.WaitForSingleObject.restype = wintypes.DWORD
        
        # CloseHandle
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        
        # SetThreadExecutionState
        self.kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        self.kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    
    def create_wake_timer(self, timer_id: str, wake_time: datetime, 
                          callback=None, recurring_ms: int = 0) -> bool:
        """
        Create a wake timer that will wake the system from sleep.
        
        Args:
            timer_id: Unique identifier for this timer
            wake_time: When to wake the system
            callback: Function to call when timer fires
            recurring_ms: Repeat interval in milliseconds (0 = one-shot)
            
        Returns:
            bool: True if timer created successfully
        """
        try:
            # Create waitable timer with high resolution
            h_timer = self.kernel32.CreateWaitableTimerExW(
                None,  # Default security attributes
                None,  # No name (anonymous)
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS
            )
            
            if not h_timer:
                raise ctypes.WinError(ctypes.get_last_error())
            
            # Calculate due time in 100-nanosecond intervals (negative = relative)
            time_until_wake = (wake_time - datetime.now()).total_seconds()
            if time_until_wake < 0:
                time_until_wake = 0
                
            # Convert to 100-nanosecond intervals (negative for relative time)
            due_time = wintypes.LARGE_INTEGER()
            due_time.value = -int(time_until_wake * 10_000_000)
            
            # Set the timer with fResume=TRUE to enable wake from sleep
            result = self.kernel32.SetWaitableTimer(
                h_timer,
                ctypes.byref(due_time),
                recurring_ms,  # Period (0 = one-shot)
                None,  # No completion routine
                None,  # No argument to completion routine
                True   # fResume=TRUE - CRITICAL for wake from sleep
            )
            
            if not result:
                self.kernel32.CloseHandle(h_timer)
                raise ctypes.WinError(ctypes.get_last_error())
            
            with self.timer_lock:
                self.timers[timer_id] = {
                    'handle': h_timer,
                    'callback': callback,
                    'wake_time': wake_time,
                    'recurring': recurring_ms > 0
                }
            
            # Start monitoring thread for this timer
            monitor_thread = threading.Thread(
                target=self._monitor_timer,
                args=(timer_id, h_timer),
                daemon=True
            )
            monitor_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to create wake timer {timer_id}: {e}")
            return False
    
    def _monitor_timer(self, timer_id: str, h_timer: wintypes.HANDLE):
        """Monitor timer and trigger callback when fired"""
        result = self.kernel32.WaitForSingleObject(h_timer, INFINITE)
        
        if result == WAIT_OBJECT_0:
            with self.timer_lock:
                timer_info = self.timers.get(timer_id)
                
            if timer_info and timer_info['callback']:
                try:
                    timer_info['callback'](timer_id)
                except Exception as e:
                    print(f"Timer callback error for {timer_id}: {e}")
            
            # Clean up if not recurring
            if timer_info and not timer_info['recurring']:
                self.cancel_timer(timer_id)
    
    def cancel_timer(self, timer_id: str) -> bool:
        """Cancel and clean up a wake timer"""
        with self.timer_lock:
            timer_info = self.timers.pop(timer_id, None)
            
        if timer_info:
            self.kernel32.CloseHandle(timer_info['handle'])
            return True
        return False
    
    def prevent_sleep(self, display_required: bool = False) -> bool:
        """
        Prevent system from entering sleep state.
        Call this during critical AI agent operations.
        
        Args:
            display_required: Also keep display on
            
        Returns:
            bool: True if successful
        """
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if display_required:
            flags |= ES_DISPLAY_REQUIRED
            
        result = self.kernel32.SetThreadExecutionState(flags)
        return result != 0
    
    def allow_sleep(self) -> bool:
        """Allow system to enter sleep state"""
        result = self.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        return result != 0


# Usage Example for AI Agent
class AIAgentScheduler:
    """Example integration with AI agent task scheduling"""
    
    def __init__(self):
        self.wake_manager = WakeTimerManager()
        self.cron_tasks = []
        
    def schedule_cron_job(self, task_name: str, schedule: str, 
                          task_func, wake_for_execution: bool = True):
        """
        Schedule a cron-like job with optional wake timer.
        
        Args:
            task_name: Unique task identifier
            schedule: Cron expression or datetime
            task_func: Function to execute
            wake_for_execution: Whether to wake system for this task
        """
        if wake_for_execution:
            # Calculate next execution time
            next_run = self._calculate_next_run(schedule)
            
            # Create wake timer
            self.wake_manager.create_wake_timer(
                timer_id=f"wake_{task_name}",
                wake_time=next_run - timedelta(minutes=1),  # Wake 1 min early
                callback=lambda _: self._prepare_for_task(task_name, task_func),
                recurring_ms=0
            )
        
        self.cron_tasks.append({
            'name': task_name,
            'schedule': schedule,
            'func': task_func,
            'wake_enabled': wake_for_execution
        })
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Parse cron expression and calculate next run time"""
        # Implementation using croniter or similar library
        # For now, return 1 hour from now as example
        return datetime.now() + timedelta(hours=1)
    
    def _prepare_for_task(self, task_name: str, task_func):
        """Called when system wakes for scheduled task"""
        print(f"System woke for task: {task_name}")
        
        # Prevent sleep during task execution
        self.wake_manager.prevent_sleep(display_required=False)
        
        try:
            # Execute the task
            task_func()
        finally:
            # Allow sleep after task completes
            self.wake_manager.allow_sleep()
            
            # Schedule next wake if recurring
            self._reschedule_if_needed(task_name)
    
    def _reschedule_if_needed(self, task_name: str):
        """Reschedule wake timer for recurring tasks"""
        task = next((t for t in self.cron_tasks if t['name'] == task_name), None)
        if task and task['wake_enabled']:
            next_run = self._calculate_next_run(task['schedule'])
            self.wake_manager.create_wake_timer(
                timer_id=f"wake_{task_name}",
                wake_time=next_run - timedelta(minutes=1),
                callback=lambda _: self._prepare_for_task(task_name, task['func']),
                recurring_ms=0
            )
```

### Wake Timer Best Practices

1. **Always set fResume=TRUE** in SetWaitableTimer for wake capability
2. **Wake 1-2 minutes early** to allow system initialization before task execution
3. **Use ES_SYSTEM_REQUIRED** during task execution to prevent immediate re-sleep
4. **Set unattended idle timer** to 2 minutes minimum after automatic wake
5. **Handle PBT_APMRESUMEAUTOMATIC** events to detect wake reason

---

## Sleep Prevention During Critical Tasks

### Critical Task Categories

| Priority | Task Type | Sleep Prevention | Display Required |
|----------|-----------|------------------|------------------|
| P0 | GPT-5.2 inference, Voice calls | YES | Optional |
| P1 | Email processing, SMS sending | YES | NO |
| P2 | Browser automation, File ops | Configurable | NO |
| P3 | Logging, Heartbeat | NO | NO |

### Implementation

```python
import contextlib
import ctypes
from enum import IntFlag

class ExecutionState(IntFlag):
    """Windows execution state flags"""
    ES_AWAYMODE_REQUIRED = 0x00000040
    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_USER_PRESENT = 0x00000004


class SleepPreventer:
    """
    Context manager for preventing sleep during critical operations.
    Automatically restores previous state on exit.
    """
    
    def __init__(self, display_required: bool = False, 
                 away_mode: bool = False):
        self.display_required = display_required
        self.away_mode = away_mode
        self.previous_state = None
        self.kernel32 = ctypes.windll.kernel32
        
    def __enter__(self):
        """Enter context - prevent sleep"""
        flags = ExecutionState.ES_CONTINUOUS | ExecutionState.ES_SYSTEM_REQUIRED
        
        if self.display_required:
            flags |= ExecutionState.ES_DISPLAY_REQUIRED
        if self.away_mode:
            flags |= ExecutionState.ES_AWAYMODE_REQUIRED
            
        self.previous_state = self.kernel32.SetThreadExecutionState(flags)
        
        if self.previous_state == 0:
            raise RuntimeError("Failed to set execution state")
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore previous state"""
        # Restore to continuous mode (clears our flags)
        self.kernel32.SetThreadExecutionState(ExecutionState.ES_CONTINUOUS)
        return False


# Usage examples for AI Agent tasks

class AITaskManager:
    """Manages AI agent tasks with appropriate sleep prevention"""
    
    @staticmethod
    def voice_call_session():
        """Voice call - prevent sleep, keep display on"""
        with SleepPreventer(display_required=True):
            # Execute voice call via Twilio
            pass
    
    @staticmethod
    def gpt_inference(prompt: str) -> str:
        """GPT-5.2 inference - prevent sleep, display optional"""
        with SleepPreventer(display_required=False):
            # Execute GPT inference
            response = call_gpt_api(prompt)
            return response
    
    @staticmethod
    def email_processing():
        """Email processing - prevent sleep only"""
        with SleepPreventer(display_required=False):
            # Check and process emails via Gmail API
            pass
    
    @staticmethod  
    def background_logging():
        """Logging - no sleep prevention needed"""
        # Just log, allow sleep
        pass
```

### Periodic Keep-Alive Strategy

For long-running operations, periodically reset the idle timer:

```python
import threading
import time

class KeepAliveManager:
    """
    Manages periodic keep-alive signals for long-running operations.
    Prevents sleep without requiring continuous blocking.
    """
    
    def __init__(self, interval_seconds: int = 30):
        self.interval = interval_seconds
        self.active = False
        self.thread = None
        self.kernel32 = ctypes.windll.kernel32
        
    def start(self, display_required: bool = False):
        """Start periodic keep-alive"""
        self.active = True
        self.display_required = display_required
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop periodic keep-alive"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _keep_alive_loop(self):
        """Thread that periodically resets idle timer"""
        flags = ExecutionState.ES_SYSTEM_REQUIRED
        if self.display_required:
            flags |= ExecutionState.ES_DISPLAY_REQUIRED
            
        while self.active:
            # Reset idle timer (without ES_CONTINUOUS)
            self.kernel32.SetThreadExecutionState(flags)
            time.sleep(self.interval)


# Example: Long-running AI training or processing
keep_alive = KeepAliveManager(interval_seconds=30)
keep_alive.start(display_required=False)

try:
    # Long-running operation
    result = long_running_ai_task()
finally:
    keep_alive.stop()
```

---

## Hibernate vs Sleep Decision Matrix

### Decision Flowchart

```
                    ┌─────────────────────┐
                    │  Task Completion    │
                    │  Time < 30 min?     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                   YES                   NO
                    │                     │
                    ▼                     ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   Battery > 20%?  │   │ Use Hibernate (S4)│
        └─────────┬─────────┘   │ Save state to disk│
                  │             │ Zero power drain  │
        ┌─────────┴─────────┐   └───────────────────┘
        │                   │
       YES                  NO
        │                   │
        ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Use Sleep (S3)│ │ Use Hibernate   │
│   Keep RAM power│ │ (S4)            │
│   ~3-10s wake   │ │ Preserve battery│
└─────────────────┘ └─────────────────┘
```

### Decision Matrix Table

| Scenario | Recommended State | Rationale |
|----------|-------------------|-----------|
| Idle < 15 min, AC power | S3 (Sleep) | Quick resume, minimal power savings needed |
| Idle 15-60 min, AC power | S3 (Sleep) | Balance of resume speed and power savings |
| Idle > 60 min, AC power | S4 (Hibernate) | Maximum power savings, acceptable wake time |
| Idle < 30 min, Battery > 30% | S3 (Sleep) | Quick resume, sufficient battery |
| Idle > 30 min, Battery < 30% | S4 (Hibernate) | Preserve battery life |
| Critical task pending | S0 (Stay Awake) | Task takes priority over power savings |
| Scheduled wake < 5 min | S0 (Stay Awake) | Avoid sleep/wake cycle overhead |
| Network activity expected | S3 (Sleep with WoL) | Allow remote wake |

### Implementation

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class PowerState(Enum):
    S0_WORKING = "S0"
    S3_SLEEP = "S3"
    S4_HIBERNATE = "S4"
    S5_SHUTDOWN = "S5"

@dataclass
class PowerDecisionContext:
    """Context for power state decision"""
    idle_duration: timedelta
    next_scheduled_task: datetime
    battery_percent: float
    on_ac_power: bool
    critical_tasks_pending: bool
    network_activity_expected: bool
    user_present: bool

class PowerStateDecider:
    """
    Intelligent power state decision engine for AI agent.
    Considers multiple factors to optimize power vs availability.
    """
    
    # Thresholds
    QUICK_WAKE_THRESHOLD = timedelta(minutes=5)
    SHORT_IDLE_THRESHOLD = timedelta(minutes=15)
    MEDIUM_IDLE_THRESHOLD = timedelta(minutes=60)
    BATTERY_LOW_THRESHOLD = 30.0
    BATTERY_CRITICAL_THRESHOLD = 15.0
    
    def decide_power_state(self, context: PowerDecisionContext) -> PowerState:
        """
        Decide optimal power state based on context.
        
        Returns:
            PowerState: Recommended power state
        """
        # Priority 1: Critical tasks override everything
        if context.critical_tasks_pending:
            return PowerState.S0_WORKING
        
        # Priority 2: Upcoming scheduled task
        time_to_next_task = context.next_scheduled_task - datetime.now()
        if time_to_next_task < self.QUICK_WAKE_THRESHOLD:
            return PowerState.S0_WORKING
        
        # Priority 3: User present
        if context.user_present:
            return PowerState.S0_WORKING
        
        # Priority 4: Battery considerations
        if not context.on_ac_power:
            if context.battery_percent < self.BATTERY_CRITICAL_THRESHOLD:
                # Critical battery - hibernate to preserve
                return PowerState.S4_HIBERNATE
            elif context.battery_percent < self.BATTERY_LOW_THRESHOLD:
                # Low battery - hibernate if idle for any significant time
                if context.idle_duration > timedelta(minutes=5):
                    return PowerState.S4_HIBERNATE
        
        # Priority 5: Idle duration
        if context.idle_duration < self.SHORT_IDLE_THRESHOLD:
            # Short idle - sleep for quick resume
            return PowerState.S3_SLEEP
        elif context.idle_duration < self.MEDIUM_IDLE_THRESHOLD:
            # Medium idle - still sleep if on AC
            if context.on_ac_power:
                return PowerState.S3_SLEEP
            else:
                return PowerState.S4_HIBERNATE
        else:
            # Long idle - hibernate for power savings
            return PowerState.S4_HIBERNATE
    
    def should_set_wake_timer(self, context: PowerDecisionContext) -> bool:
        """Determine if wake timer should be set"""
        time_to_next_task = context.next_scheduled_task - datetime.now()
        
        # Set wake timer if:
        # 1. Next task is within 24 hours
        # 2. We're going to sleep/hibernate
        # 3. Task requires system to be awake
        
        if time_to_next_task > timedelta(hours=24):
            return False
            
        state = self.decide_power_state(context)
        if state == PowerState.S0_WORKING:
            return False
            
        return True


# Example usage
decider = PowerStateDecider()

context = PowerDecisionContext(
    idle_duration=timedelta(minutes=45),
    next_scheduled_task=datetime.now() + timedelta(hours=2),
    battery_percent=75.0,
    on_ac_power=True,
    critical_tasks_pending=False,
    network_activity_expected=True,
    user_present=False
)

recommended_state = decider.decide_power_state(context)
print(f"Recommended power state: {recommended_state.value}")
```

---

## Power Event Handling System

### Power Event Types

| Event | Value | Description | Handler Priority |
|-------|-------|-------------|------------------|
| PBT_APMPOWERSTATUSCHANGE | 0x0A | Power status changed | Medium |
| PBT_APMRESUMEAUTOMATIC | 0x12 | System resumed automatically | High |
| PBT_APMRESUMESUSPEND | 0x07 | System resumed from sleep | High |
| PBT_APMSUSPEND | 0x04 | System entering sleep | Critical |
| PBT_POWERSETTINGCHANGE | 0x8013 | Power setting changed | Low |

### Implementation

```python
import ctypes
import ctypes.wintypes
from ctypes import wintypes
import threading
from enum import IntEnum

class PowerEvent(IntEnum):
    """Windows power broadcast events"""
    PBT_APMPOWERSTATUSCHANGE = 0x0A
    PBT_APMRESUMEAUTOMATIC = 0x12
    PBT_APMRESUMESUSPEND = 0x07
    PBT_APMSUSPEND = 0x04
    PBT_POWERSETTINGCHANGE = 0x8013

class PowerSettingGUID:
    """GUIDs for power setting notifications"""
    # Battery saver
    GUID_POWER_SAVING_STATUS = "E00958C0-C213-4ACE-AC77-DECD2ECCBE43"
    # Power source change
    GUID_ACDC_POWER_SOURCE = "5D3E9A59-E9D5-4B00-A6BD-FF34FF516548"
    # Battery capacity remaining
    GUID_BATTERY_PERCENTAGE_REMAINING = "A7AD8041-B45A-4CAE-87A3-EECBB468A9E1"
    # Console display state
    GUID_CONSOLE_DISPLAY_STATE = "6FE69556-704A-47A0-8F24-C28D936FDA47"
    # Session user presence
    GUID_SESSION_USER_PRESENCE = "3C0F4548-C03F-4C4D-B9F2-237EDE686376"
    # System away mode
    GUID_SYSTEM_AWAYMODE = "98A7F580-01F7-48AA-9C0F-44352C29E5C0"


class PowerEventHandler:
    """
    Handles Windows power events for AI agent system.
    Registers for power notifications and dispatches to handlers.
    """
    
    WM_POWERBROADCAST = 0x0218
    DEVICE_NOTIFY_WINDOW_HANDLE = 0x00000000
    DEVICE_NOTIFY_SERVICE_HANDLE = 0x00000001
    DEVICE_NOTIFY_CALLBACK = 0x00000002
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        self.handlers = {event: [] for event in PowerEvent}
        self.notification_handles = []
        self.hwnd = None
        self._setup_window_class()
        
    def _setup_window_class(self):
        """Create hidden window for receiving power messages"""
        WNDPROC = ctypes.WINFUNCTYPE(
            wintypes.LPARAM,
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )
        
        self.wndproc = WNDPROC(self._window_proc)
        
        WNDCLASSEX = ctypes.wintypes.WNDCLASSEXW
        wndclass = WNDCLASSEX()
        wndclass.cbSize = ctypes.sizeof(WNDCLASSEX)
        wndclass.lpfnWndProc = self.wndproc
        wndclass.hInstance = self.kernel32.GetModuleHandleW(None)
        wndclass.lpszClassName = "AIAgentPowerHandler"
        
        class_atom = self.user32.RegisterClassExW(ctypes.byref(wndclass))
        if not class_atom:
            raise ctypes.WinError(ctypes.get_last_error())
        
        # Create hidden window
        self.hwnd = self.user32.CreateWindowExW(
            0, class_atom, "AIAgentPowerHandler",
            0, 0, 0, 0, 0, 0, 0, wndclass.hInstance, None
        )
        
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
    
    def _window_proc(self, hwnd, msg, wparam, lparam):
        """Window procedure to handle power messages"""
        if msg == self.WM_POWERBROADCAST:
            self._handle_power_event(wparam, lparam)
            return 0
        return self.user32.DefWindowProcW(hwnd, msg, wparam, lparam)
    
    def _handle_power_event(self, wparam, lparam):
        """Dispatch power events to registered handlers"""
        event = wparam
        
        if event in self.handlers:
            for handler in self.handlers[event]:
                try:
                    handler(event, lparam)
                except Exception as e:
                    print(f"Power event handler error: {e}")
    
    def register_handler(self, event: PowerEvent, handler):
        """Register a handler for a specific power event"""
        self.handlers[event].append(handler)
    
    def register_power_setting_notification(self, guid: str):
        """Register for specific power setting changes"""
        # Convert GUID string to GUID structure
        guid_bytes = bytes.fromhex(guid.replace("-", ""))
        
        # RegisterPowerSettingNotification
        self.user32.RegisterPowerSettingNotification.argtypes = [
            wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD
        ]
        self.user32.RegisterPowerSettingNotification.restype = wintypes.HANDLE
        
        h_notify = self.user32.RegisterPowerSettingNotification(
            self.hwnd,
            guid_bytes,
            self.DEVICE_NOTIFY_WINDOW_HANDLE
        )
        
        if h_notify:
            self.notification_handles.append(h_notify)
    
    def message_loop(self):
        """Run message loop to receive power notifications"""
        msg = wintypes.MSG()
        
        while self.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            self.user32.TranslateMessage(ctypes.byref(msg))
            self.user32.DispatchMessageW(ctypes.byref(msg))
    
    def start(self):
        """Start power event handling in background thread"""
        self.thread = threading.Thread(target=self.message_loop, daemon=True)
        self.thread.start()
    
    def cleanup(self):
        """Clean up resources"""
        for h_notify in self.notification_handles:
            self.user32.UnregisterPowerSettingNotification(h_notify)
        
        if self.hwnd:
            self.user32.DestroyWindow(self.hwnd)


# AI Agent specific power event handlers
class AIAgentPowerManager:
    """
    Manages AI agent behavior during power state changes.
    Ensures graceful handling of sleep/wake cycles.
    """
    
    def __init__(self):
        self.event_handler = PowerEventHandler()
        self.wake_manager = WakeTimerManager()
        self.is_resuming = False
        self.pending_tasks = []
        
        # Register handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register AI agent specific power event handlers"""
        self.event_handler.register_handler(
            PowerEvent.PBT_APMSUSPEND,
            self._on_system_suspending
        )
        self.event_handler.register_handler(
            PowerEvent.PBT_APMRESUMEAUTOMATIC,
            self._on_system_resumed_automatic
        )
        self.event_handler.register_handler(
            PowerEvent.PBT_APMRESUMESUSPEND,
            self._on_system_resumed_user
        )
        self.event_handler.register_handler(
            PowerEvent.PBT_APMPOWERSTATUSCHANGE,
            self._on_power_status_changed
        )
        
        # Register for power setting notifications
        self.event_handler.register_power_setting_notification(
            PowerSettingGUID.GUID_ACDC_POWER_SOURCE
        )
        self.event_handler.register_power_setting_notification(
            PowerSettingGUID.GUID_BATTERY_PERCENTAGE_REMAINING
        )
    
    def _on_system_suspending(self, event, lparam):
        """Handle system entering sleep"""
        print("[POWER] System entering sleep state")
        
        # Save critical state
        self._save_agent_state()
        
        # Complete or pause ongoing tasks
        self._pause_active_tasks()
        
        # Close resources that shouldn't persist
        self._release_transient_resources()
        
        # Set wake timer for next scheduled task
        next_task = self._get_next_scheduled_task()
        if next_task:
            self.wake_manager.create_wake_timer(
                "next_scheduled_task",
                next_task - timedelta(minutes=1),
                self._on_wake_for_task
            )
    
    def _on_system_resumed_automatic(self, event, lparam):
        """Handle automatic wake (timer, WoL, etc.)"""
        print("[POWER] System resumed automatically")
        self.is_resuming = True
        
        # Prevent immediate re-sleep
        self.wake_manager.prevent_sleep(display_required=False)
        
        # Restore agent state
        self._restore_agent_state()
        
        # The unattended idle timer is now 2 minutes
        # We have time to indicate we're busy
        
    def _on_system_resumed_user(self, event, lparam):
        """Handle user-initiated wake"""
        print("[POWER] System resumed by user activity")
        self.is_resuming = False
        
        # Restore full operation
        self._restore_agent_state()
        
        # Allow normal sleep behavior
        self.wake_manager.allow_sleep()
    
    def _on_power_status_changed(self, event, lparam):
        """Handle AC/battery status change"""
        status = self._get_power_status()
        print(f"[POWER] Power status changed: {status}")
        
        # Adjust behavior based on power source
        if not status['on_ac_power']:
            self._enter_battery_optimized_mode()
        else:
            self._enter_ac_power_mode()
    
    def _on_wake_for_task(self, timer_id):
        """Called when system wakes for scheduled task"""
        print(f"[POWER] Woke for timer: {timer_id}")
        
        # Execute pending tasks
        for task in self.pending_tasks:
            try:
                task()
            except Exception as e:
                print(f"Task execution error: {e}")
        
        self.pending_tasks = []
        
        # Allow sleep after tasks complete
        self.wake_manager.allow_sleep()
    
    # Placeholder methods for AI agent integration
    def _save_agent_state(self):
        """Save AI agent state before sleep"""
        pass
    
    def _restore_agent_state(self):
        """Restore AI agent state after wake"""
        pass
    
    def _pause_active_tasks(self):
        """Pause or complete active tasks"""
        pass
    
    def _release_transient_resources(self):
        """Release resources that shouldn't persist"""
        pass
    
    def _get_next_scheduled_task(self):
        """Get next scheduled task time"""
        return None
    
    def _get_power_status(self):
        """Get current power status"""
        return {'on_ac_power': True}
    
    def _enter_battery_optimized_mode(self):
        """Reduce power consumption when on battery"""
        pass
    
    def _enter_ac_power_mode(self):
        """Full operation when on AC power"""
        pass
```

---

## Battery-Aware Operation

### Battery Status Monitoring

```python
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

class BatteryFlag:
    """Battery status flags"""
    HIGH = 0x01          # > 66%
    LOW = 0x02           # < 33%
    CRITICAL = 0x04      # < 5%
    CHARGING = 0x08
    NO_BATTERY = 0x80
    UNKNOWN = 0xFF

class ACLineStatus:
    """AC line status"""
    OFFLINE = 0x00
    ONLINE = 0x01
    UNKNOWN = 0xFF

@dataclass
class BatteryStatus:
    """Complete battery status information"""
    on_ac_power: bool
    battery_present: bool
    battery_life_percent: int
    battery_life_seconds: Optional[int]
    battery_full_life_seconds: Optional[int]
    is_charging: bool
    battery_flag: int
    
    @property
    def is_low(self) -> bool:
        return self.battery_life_percent < 30
    
    @property
    def is_critical(self) -> bool:
        return self.battery_life_percent < 10


class BatteryMonitor:
    """
    Monitors battery status for laptops and UPS systems.
    Provides callbacks for battery level changes.
    """
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self._setup_api()
        self.callbacks = []
        self.last_status = None
        
    def _setup_api(self):
        """Setup Windows API for battery status"""
        # SYSTEM_POWER_STATUS structure
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", wintypes.BYTE),
                ("BatteryFlag", wintypes.BYTE),
                ("BatteryLifePercent", wintypes.BYTE),
                ("Reserved1", wintypes.BYTE),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]
        
        self.SYSTEM_POWER_STATUS = SYSTEM_POWER_STATUS
        
        # GetSystemPowerStatus
        self.kernel32.GetSystemPowerStatus.argtypes = [
            ctypes.POINTER(SYSTEM_POWER_STATUS)
        ]
        self.kernel32.GetSystemPowerStatus.restype = wintypes.BOOL
    
    def get_status(self) -> BatteryStatus:
        """Get current battery status"""
        status = self.SYSTEM_POWER_STATUS()
        
        if not self.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        
        return BatteryStatus(
            on_ac_power=(status.ACLineStatus == ACLineStatus.ONLINE),
            battery_present=not (status.BatteryFlag & BatteryFlag.NO_BATTERY),
            battery_life_percent=status.BatteryLifePercent,
            battery_life_seconds=status.BatteryLifeTime if status.BatteryLifeTime != 0xFFFFFFFF else None,
            battery_full_life_seconds=status.BatteryFullLifeTime if status.BatteryFullLifeTime != 0xFFFFFFFF else None,
            is_charging=bool(status.BatteryFlag & BatteryFlag.CHARGING),
            battery_flag=status.BatteryFlag
        )
    
    def register_callback(self, callback):
        """Register callback for battery status changes"""
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background battery monitoring"""
        import threading
        import time
        
        def monitor_loop():
            while True:
                try:
                    current = self.get_status()
                    
                    if self.last_status:
                        # Check for significant changes
                        if (current.battery_life_percent != self.last_status.battery_life_percent or
                            current.on_ac_power != self.last_status.on_ac_power):
                            for callback in self.callbacks:
                                try:
                                    callback(current, self.last_status)
                                except Exception as e:
                                    print(f"Battery callback error: {e}")
                    
                    self.last_status = current
                    
                except Exception as e:
                    print(f"Battery monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()


class BatteryAwareTaskScheduler:
    """
    Task scheduler that adjusts behavior based on battery status.
    Defers non-critical tasks when battery is low.
    """
    
    def __init__(self):
        self.battery_monitor = BatteryMonitor()
        self.battery_monitor.register_callback(self._on_battery_change)
        self.current_battery_status = None
        
        # Task queues by priority
        self.critical_tasks = []      # Always execute
        self.normal_tasks = []        # Execute if battery > 20%
        self.background_tasks = []    # Execute only on AC or battery > 50%
        
    def _on_battery_change(self, new_status: BatteryStatus, old_status: BatteryStatus):
        """Handle battery status changes"""
        self.current_battery_status = new_status
        
        print(f"[BATTERY] Status changed: {new_status.battery_life_percent}% "
              f"({'AC' if new_status.on_ac_power else 'Battery'})")
        
        # Adjust behavior based on battery
        if new_status.is_critical:
            self._enter_critical_battery_mode()
        elif new_status.is_low and not new_status.on_ac_power:
            self._enter_low_battery_mode()
        elif new_status.on_ac_power:
            self._enter_ac_mode()
    
    def _enter_critical_battery_mode(self):
        """Emergency power saving mode"""
        print("[BATTERY] CRITICAL: Entering emergency mode")
        # Hibernate immediately to preserve data
        self._initiate_hibernate()
    
    def _enter_low_battery_mode(self):
        """Reduced operation mode"""
        print("[BATTERY] LOW: Deferring non-critical tasks")
        # Only critical tasks
        self._pause_normal_tasks()
        self._pause_background_tasks()
    
    def _enter_ac_mode(self):
        """Full operation mode"""
        print("[BATTERY] AC Power: Resuming normal operation")
        self._resume_all_tasks()
    
    def schedule_task(self, task, priority: str = "normal"):
        """Schedule a task with battery-aware execution"""
        if priority == "critical":
            self.critical_tasks.append(task)
        elif priority == "normal":
            self.normal_tasks.append(task)
        else:
            self.background_tasks.append(task)
    
    def should_execute_task(self, priority: str) -> bool:
        """Determine if task should execute based on battery"""
        if priority == "critical":
            return True
        
        if not self.current_battery_status:
            return True
        
        if priority == "normal":
            return (self.current_battery_status.on_ac_power or 
                    self.current_battery_status.battery_life_percent > 20)
        
        if priority == "background":
            return (self.current_battery_status.on_ac_power or 
                    self.current_battery_status.battery_life_percent > 50)
        
        return True
    
    # Placeholder methods
    def _initiate_hibernate(self):
        """Initiate system hibernate"""
        pass
    
    def _pause_normal_tasks(self):
        """Pause normal priority tasks"""
        pass
    
    def _pause_background_tasks(self):
        """Pause background tasks"""
        pass
    
    def _resume_all_tasks(self):
        """Resume all paused tasks"""
        pass
```

---

## Wake-on-LAN Support

### Configuration Guide

#### BIOS/UEFI Settings

1. Enter BIOS/UEFI (typically F2, Del, or F12 during boot)
2. Navigate to Power Management or Advanced settings
3. Enable options:
   - "Wake on LAN"
   - "Resume on LAN"
   - "Power on by PCI/PCIe device"
   - "Wake on PME" (Power Management Event)

#### Windows Configuration

```powershell
# Enable Wake-on-LAN in Windows

# 1. Disable Fast Startup (interferes with WoL)
powercfg /hibernate off

# 2. Check network adapter settings
# Via Device Manager or PowerShell:

# Get network adapters
Get-NetAdapter | Where-Object {$_.Status -eq "Up"}

# Enable WoL on adapter (requires admin)
$adapter = Get-NetAdapter -Name "Ethernet"
Set-NetAdapterPowerManagement -Name $adapter.Name -WakeOnMagicPacket Enabled

# 3. Configure power management for adapter
# Device Manager > Network Adapter > Properties > Power Management
# - Check "Allow this device to wake the computer"
# - Check "Only allow a magic packet to wake the computer"

# 4. Disable Energy Efficient Ethernet
# Device Manager > Network Adapter > Properties > Advanced
# - Set "Energy Efficient Ethernet" to Disabled
# - Set "Wake on Magic Packet" to Enabled
```

### Magic Packet Implementation

```python
import socket
import struct

class WakeOnLAN:
    """
    Send Wake-on-LAN magic packets to wake remote systems.
    """
    
    # Standard WoL ports
    DEFAULT_PORT = 9
    ALTERNATE_PORT = 7
    
    def __init__(self, broadcast_address: str = "255.255.255.255"):
        self.broadcast_address = broadcast_address
        
    def create_magic_packet(self, mac_address: str) -> bytes:
        """
        Create a magic packet for the specified MAC address.
        
        Magic packet format:
        - 6 bytes of 0xFF (synchronization stream)
        - 16 repetitions of target MAC address
        """
        # Clean MAC address
        mac_clean = mac_address.replace(":", "").replace("-", "").replace(".", "")
        
        if len(mac_clean) != 12:
            raise ValueError(f"Invalid MAC address: {mac_address}")
        
        # Convert to bytes
        mac_bytes = bytes.fromhex(mac_clean)
        
        # Create magic packet
        # 6 bytes of 0xFF followed by 16 repetitions of MAC
        magic_packet = b'\xff' * 6 + mac_bytes * 16
        
        return magic_packet
    
    def send_magic_packet(self, mac_address: str, 
                          ip_address: str = None,
                          port: int = None) -> bool:
        """
        Send magic packet to wake a computer.
        
        Args:
            mac_address: Target MAC address (e.g., "00:11:22:33:44:55")
            ip_address: Target IP or broadcast address
            port: UDP port (default 9)
            
        Returns:
            bool: True if packet sent successfully
        """
        try:
            # Create magic packet
            packet = self.create_magic_packet(mac_address)
            
            # Determine destination
            dest_ip = ip_address or self.broadcast_address
            dest_port = port or self.DEFAULT_PORT
            
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Send packet
            sock.sendto(packet, (dest_ip, dest_port))
            sock.close()
            
            print(f"[WoL] Magic packet sent to {mac_address} at {dest_ip}:{dest_port}")
            return True
            
        except Exception as e:
            print(f"[WoL] Failed to send magic packet: {e}")
            return False
    
    def wake_multiple(self, mac_addresses: list, 
                      ip_address: str = None,
                      port: int = None) -> dict:
        """
        Send magic packets to multiple MAC addresses.
        
        Returns:
            dict: Results for each MAC address
        """
        results = {}
        for mac in mac_addresses:
            results[mac] = self.send_magic_packet(mac, ip_address, port)
        return results


# Example usage
if __name__ == "__main__":
    wol = WakeOnLAN()
    
    # Wake a single computer
    wol.send_magic_packet("00:11:22:33:44:55")
    
    # Wake multiple computers
    wol.wake_multiple([
        "00:11:22:33:44:55",
        "00:11:22:33:44:56",
        "00:11:22:33:44:57"
    ])
```

### WoL Server for AI Agent

```python
import socket
import threading
import struct

class WoLServer:
    """
    Server that listens for WoL packets and triggers AI agent wake.
    Useful for remote management of the AI agent system.
    """
    
    def __init__(self, port: int = 9, callback=None):
        self.port = port
        self.callback = callback
        self.running = False
        self.socket = None
        self.thread = None
        
    def start(self):
        """Start WoL listener server"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"[WoL Server] Listening on port {self.port}")
        
    def stop(self):
        """Stop WoL listener server"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=1)
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("0.0.0.0", self.port))
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(1024)
                    self._handle_packet(data, addr)
                except socket.error:
                    if self.running:
                        raise
                    
        except Exception as e:
            print(f"[WoL Server] Error: {e}")
    
    def _handle_packet(self, data: bytes, addr: tuple):
        """Handle received WoL packet"""
        # Validate magic packet format
        if len(data) != 102:
            return
        
        # Check synchronization stream
        if data[:6] != b'\xff' * 6:
            return
        
        # Extract MAC address
        mac_bytes = data[6:12]
        mac_address = ':'.join(f'{b:02x}' for b in mac_bytes)
        
        print(f"[WoL Server] Received magic packet for {mac_address} from {addr}")
        
        # Trigger callback
        if self.callback:
            try:
                self.callback(mac_address, addr)
            except Exception as e:
                print(f"[WoL Server] Callback error: {e}")


# Integration with AI Agent
class AIAgentWoLHandler:
    """
    Handles WoL events for AI agent system.
    Can be used to trigger agent wake from remote systems.
    """
    
    def __init__(self, agent_core):
        self.agent_core = agent_core
        self.wol_server = WoLServer(callback=self._on_wol_received)
        
    def start(self):
        """Start WoL handling"""
        self.wol_server.start()
        
    def stop(self):
        """Stop WoL handling"""
        self.wol_server.stop()
        
    def _on_wol_received(self, mac_address: str, addr: tuple):
        """Handle WoL packet reception"""
        # Verify this is our MAC address
        if self._is_our_mac(mac_address):
            print(f"[AI Agent] WoL received, resuming operation")
            
            # Prevent sleep
            self.agent_core.power_manager.prevent_sleep()
            
            # Resume operations
            self.agent_core.resume_from_remote_wake()
    
    def _is_our_mac(self, mac_address: str) -> bool:
        """Check if MAC address belongs to this system"""
        # Get our MAC address and compare
        our_mac = self._get_system_mac()
        return mac_address.lower() == our_mac.lower()
    
    def _get_system_mac(self) -> str:
        """Get system MAC address"""
        import uuid
        mac = uuid.getnode()
        return ':'.join(f'{(mac >> i) & 0xff:02x}' for i in range(40, -1, -8))
```

---

## Power Usage Optimization

### Optimization Strategies

```python
import psutil
import time
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class PowerMetrics:
    """Power consumption metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    battery_percent: float
    estimated_watts: float

class PowerOptimizer:
    """
    Optimizes AI agent power consumption through intelligent resource management.
    """
    
    def __init__(self):
        self.metrics_history: List[PowerMetrics] = []
        self.optimization_enabled = True
        self.power_profile = "balanced"  # balanced, performance, powersave
        
        # Thresholds
        self.cpu_idle_threshold = 10.0
        self.memory_pressure_threshold = 80.0
        self.battery_save_threshold = 30.0
        
    def collect_metrics(self) -> PowerMetrics:
        """Collect current power metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024 * 1024)
        disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024 * 1024)
        net_recv_mb = net_io.bytes_recv / (1024 * 1024)
        
        # Battery (if available)
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else 100.0
        
        # Estimate power consumption (rough approximation)
        estimated_watts = self._estimate_power(
            cpu_percent, memory_percent, disk_io, net_io
        )
        
        metrics = PowerMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            battery_percent=battery_percent,
            estimated_watts=estimated_watts
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of metrics
        cutoff = time.time() - (24 * 3600)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        return metrics
    
    def _estimate_power(self, cpu_percent: float, memory_percent: float,
                       disk_io, net_io) -> float:
        """Estimate power consumption in watts"""
        # Base power (idle system)
        base_watts = 15.0
        
        # CPU contribution (assume 65W TDP processor)
        cpu_watts = (cpu_percent / 100.0) * 65.0
        
        # Memory contribution (assume 5W per 16GB at full load)
        memory_watts = (memory_percent / 100.0) * 5.0
        
        # Disk contribution (assume 7W active, 2W idle)
        disk_active = (disk_io.read_bytes + disk_io.write_bytes) > 0
        disk_watts = 7.0 if disk_active else 2.0
        
        # Network contribution (assume 2W active)
        net_active = (net_io.bytes_sent + net_io.bytes_recv) > 0
        net_watts = 2.0 if net_active else 0.5
        
        return base_watts + cpu_watts + memory_watts + disk_watts + net_watts
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get power optimization recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        recent = self.metrics_history[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)
        avg_watts = sum(m.estimated_watts for m in recent) / len(recent)
        
        # CPU optimization
        if avg_cpu < self.cpu_idle_threshold:
            recommendations.append({
                "category": "CPU",
                "issue": "Low CPU utilization",
                "recommendation": "Consider reducing polling frequency or entering sleep",
                "potential_savings": "5-10W",
                "priority": "low"
            })
        elif avg_cpu > 80:
            recommendations.append({
                "category": "CPU",
                "issue": "High CPU utilization",
                "recommendation": "Optimize algorithms or distribute load",
                "potential_savings": "N/A - Performance issue",
                "priority": "high"
            })
        
        # Memory optimization
        if avg_memory > self.memory_pressure_threshold:
            recommendations.append({
                "category": "Memory",
                "issue": "High memory pressure",
                "recommendation": "Review memory leaks or increase RAM",
                "potential_savings": "N/A - Stability issue",
                "priority": "high"
            })
        
        # Power profile recommendation
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            if battery.percent < self.battery_save_threshold:
                recommendations.append({
                    "category": "Power Profile",
                    "issue": "Low battery",
                    "recommendation": "Switch to powersave profile",
                    "potential_savings": "10-20W",
                    "priority": "critical"
                })
        
        return recommendations
    
    def apply_optimizations(self):
        """Apply automatic power optimizations"""
        if not self.optimization_enabled:
            return
        
        metrics = self.collect_metrics()
        recommendations = self.get_optimization_recommendations()
        
        # Apply critical recommendations
        for rec in recommendations:
            if rec["priority"] == "critical":
                self._apply_recommendation(rec)
    
    def _apply_recommendation(self, recommendation: Dict):
        """Apply a specific optimization recommendation"""
        category = recommendation["category"]
        
        if category == "Power Profile":
            if "powersave" in recommendation["recommendation"]:
                self._set_power_profile("powersave")
            elif "performance" in recommendation["recommendation"]:
                self._set_power_profile("performance")
    
    def _set_power_profile(self, profile: str):
        """Set Windows power profile"""
        import subprocess
        
        profiles = {
            "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "powersave": "a1841308-3541-4fab-bc81-f71556f20b4a"
        }
        
        if profile in profiles:
            subprocess.run([
                "powercfg", "/setactive", profiles[profile]
            ], check=False)
            self.power_profile = profile
            print(f"[Power] Switched to {profile} profile")
    
    def generate_report(self) -> str:
        """Generate power usage report"""
        if not self.metrics_history:
            return "No power metrics available"
        
        recent = self.metrics_history[-60:]  # Last hour
        
        avg_watts = sum(m.estimated_watts for m in recent) / len(recent)
        max_watts = max(m.estimated_watts for m in recent)
        min_watts = min(m.estimated_watts for m in recent)
        
        total_kwh = sum(m.estimated_watts for m in self.metrics_history) / 1000 / 3600
        
        report = f"""
=== Power Usage Report ===

Period: {len(self.metrics_history)} measurements
Duration: {(self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp) / 3600:.1f} hours

Current Metrics:
  CPU Usage: {self.metrics_history[-1].cpu_percent:.1f}%
  Memory Usage: {self.metrics_history[-1].memory_percent:.1f}%
  Battery: {self.metrics_history[-1].battery_percent:.1f}%

Power Consumption (last hour):
  Average: {avg_watts:.1f}W
  Maximum: {max_watts:.1f}W
  Minimum: {min_watts:.1f}W

Total Energy: {total_kwh:.3f} kWh
Estimated Cost: ${total_kwh * 0.12:.2f} (@ $0.12/kWh)

Active Profile: {self.power_profile}

Recommendations:
"""
        for rec in self.get_optimization_recommendations():
            report += f"  - [{rec['priority'].upper()}] {rec['category']}: {rec['recommendation']}\n"
        
        return report


# Integration with AI Agent
class AIAgentPowerOptimizer:
    """
    Power optimization specific to AI agent workloads.
    """
    
    def __init__(self, agent_core):
        self.agent_core = agent_core
        self.optimizer = PowerOptimizer()
        self.monitoring = False
        
    def start_monitoring(self):
        """Start continuous power monitoring"""
        self.monitoring = True
        import threading
        
        def monitor_loop():
            while self.monitoring:
                self.optimizer.collect_metrics()
                self.optimizer.apply_optimizations()
                time.sleep(60)  # Check every minute
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
    def stop_monitoring(self):
        """Stop power monitoring"""
        self.monitoring = False
        
    def optimize_for_task(self, task_type: str):
        """Optimize power profile for specific task type"""
        profiles = {
            "inference": "performance",      # High CPU for GPT inference
            "voice_call": "performance",     # Low latency for voice
            "idle": "powersave",             # Save power when idle
            "background": "balanced",        # Balanced for background
            "browser": "balanced",           # Balanced for browser
            "email": "powersave"             # Low power for email
        }
        
        profile = profiles.get(task_type, "balanced")
        self.optimizer._set_power_profile(profile)
        
    def get_power_report(self) -> str:
        """Get detailed power report"""
        return self.optimizer.generate_report()
```

---

## Implementation Code Reference

### Complete Power Manager Class

```python
"""
OpenClaw AI Agent - Complete Power Manager
============================================
Integrated power management for 24/7 AI agent operation.
"""

import ctypes
import ctypes.wintypes
from ctypes import wintypes
from datetime import datetime, timedelta
from enum import Enum, IntFlag
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
import threading
import time
import json


class PowerState(Enum):
    """ACPI power states"""
    S0_WORKING = "S0"
    S3_SLEEP = "S3"
    S4_HIBERNATE = "S4"
    S5_SHUTDOWN = "S5"


class ExecutionState(IntFlag):
    """Windows execution state flags"""
    ES_AWAYMODE_REQUIRED = 0x00000040
    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_USER_PRESENT = 0x00000004


class PowerEvent(IntEnum):
    """Windows power broadcast events"""
    PBT_APMPOWERSTATUSCHANGE = 0x0A
    PBT_APMRESUMEAUTOMATIC = 0x12
    PBT_APMRESUMESUSPEND = 0x07
    PBT_APMSUSPEND = 0x04
    PBT_POWERSETTINGCHANGE = 0x8013


@dataclass
class PowerStatus:
    """System power status"""
    on_ac_power: bool
    battery_percent: float
    battery_present: bool
    is_charging: bool


@dataclass
class ScheduledTask:
    """Scheduled task with wake capability"""
    task_id: str
    scheduled_time: datetime
    callback: Callable
    wake_required: bool
    priority: str  # critical, normal, background
    recurring: Optional[timedelta] = None


class OpenClawPowerManager:
    """
    Comprehensive power manager for OpenClaw AI Agent.
    Handles all aspects of power management for 24/7 operation.
    """
    
    def __init__(self):
        # Windows API
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        
        # State
        self.current_state = PowerState.S0_WORKING
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.active_wake_timers: Dict[str, wintypes.HANDLE] = {}
        self.power_event_handlers: Dict[PowerEvent, List[Callable]] = {
            event: [] for event in PowerEvent
        }
        
        # Locks
        self.task_lock = threading.Lock()
        self.timer_lock = threading.Lock()
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # Setup
        self._setup_api()
        self._setup_power_notifications()
        
    def _setup_api(self):
        """Configure Windows API function signatures"""
        # SetThreadExecutionState
        self.kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        self.kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        
        # CreateWaitableTimerExW
        self.kernel32.CreateWaitableTimerExW.argtypes = [
            wintypes.LPVOID, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD
        ]
        self.kernel32.CreateWaitableTimerExW.restype = wintypes.HANDLE
        
        # SetWaitableTimer
        self.kernel32.SetWaitableTimer.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.LARGE_INTEGER),
            wintypes.LONG, wintypes.LPVOID, wintypes.LPVOID, wintypes.BOOL
        ]
        self.kernel32.SetWaitableTimer.restype = wintypes.BOOL
        
    def _setup_power_notifications(self):
        """Setup power event notifications"""
        # This would set up window message handling
        # See PowerEventHandler class above for full implementation
        pass
    
    # ==================== Public API ====================
    
    def prevent_sleep(self, display_required: bool = False,
                     away_mode: bool = False) -> bool:
        """
        Prevent system from entering sleep.
        
        Args:
            display_required: Keep display powered on
            away_mode: Use away mode (background operation)
            
        Returns:
            bool: True if successful
        """
        flags = ExecutionState.ES_CONTINUOUS | ExecutionState.ES_SYSTEM_REQUIRED
        
        if display_required:
            flags |= ExecutionState.ES_DISPLAY_REQUIRED
        if away_mode:
            flags |= ExecutionState.ES_AWAYMODE_REQUIRED
            
        result = self.kernel32.SetThreadExecutionState(flags)
        
        if result:
            self.current_state = PowerState.S0_WORKING
            return True
        return False
    
    def allow_sleep(self) -> bool:
        """Allow system to enter sleep state"""
        result = self.kernel32.SetThreadExecutionState(ExecutionState.ES_CONTINUOUS)
        return result != 0
    
    def schedule_wake_task(self, task: ScheduledTask) -> bool:
        """
        Schedule a task with automatic wake capability.
        
        Args:
            task: ScheduledTask to schedule
            
        Returns:
            bool: True if scheduled successfully
        """
        with self.task_lock:
            self.scheduled_tasks[task.task_id] = task
        
        if task.wake_required:
            return self._create_wake_timer(task)
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        with self.task_lock:
            task = self.scheduled_tasks.pop(task_id, None)
        
        if task and task.wake_required:
            self._cancel_wake_timer(task_id)
        
        return task is not None
    
    def get_power_status(self) -> PowerStatus:
        """Get current power status"""
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", wintypes.BYTE),
                ("BatteryFlag", wintypes.BYTE),
                ("BatteryLifePercent", wintypes.BYTE),
                ("Reserved1", wintypes.BYTE),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]
        
        status = SYSTEM_POWER_STATUS()
        
        if self.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return PowerStatus(
                on_ac_power=(status.ACLineStatus == 1),
                battery_percent=status.BatteryLifePercent,
                battery_present=not (status.BatteryFlag & 0x80),
                is_charging=bool(status.BatteryFlag & 0x08)
            )
        
        # Default to AC power if can't determine
        return PowerStatus(on_ac_power=True, battery_percent=100,
                          battery_present=False, is_charging=False)
    
    def recommend_power_state(self) -> PowerState:
        """
        Recommend optimal power state based on current conditions.
        
        Returns:
            PowerState: Recommended state
        """
        status = self.get_power_status()
        
        # Check for critical tasks
        critical_tasks = [t for t in self.scheduled_tasks.values()
                         if t.priority == "critical" and
                         t.scheduled_time < datetime.now() + timedelta(minutes=5)]
        
        if critical_tasks:
            return PowerState.S0_WORKING
        
        # Check battery
        if not status.on_ac_power:
            if status.battery_percent < 10:
                return PowerState.S4_HIBERNATE
            elif status.battery_percent < 25:
                return PowerState.S3_SLEEP
        
        # Default to sleep for quick resume
        return PowerState.S3_SLEEP
    
    def enter_sleep(self, state: PowerState = None) -> bool:
        """
        Enter specified sleep state.
        
        Args:
            state: Target power state (default: recommended)
            
        Returns:
            bool: True if successful
        """
        if state is None:
            state = self.recommend_power_state()
        
        # Set wake timers before sleeping
        self._prepare_wake_timers()
        
        # Initiate sleep
        if state == PowerState.S3_SLEEP:
            return self._initiate_sleep()
        elif state == PowerState.S4_HIBERNATE:
            return self._initiate_hibernate()
        
        return False
    
    def start_monitoring(self):
        """Start continuous power monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop power monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    # ==================== Private Methods ====================
    
    def _create_wake_timer(self, task: ScheduledTask) -> bool:
        """Create Windows wake timer for scheduled task"""
        try:
            # Create timer
            h_timer = self.kernel32.CreateWaitableTimerExW(
                None, None, 0x00000002, 0x001F0003
            )
            
            if not h_timer:
                return False
            
            # Calculate wake time (1 minute before task)
            wake_time = task.scheduled_time - timedelta(minutes=1)
            time_until = (wake_time - datetime.now()).total_seconds()
            
            if time_until < 0:
                time_until = 0
            
            # Set timer
            due_time = wintypes.LARGE_INTEGER()
            due_time.value = -int(time_until * 10_000_000)
            
            result = self.kernel32.SetWaitableTimer(
                h_timer, ctypes.byref(due_time), 0, None, None, True
            )
            
            if result:
                with self.timer_lock:
                    self.active_wake_timers[task.task_id] = h_timer
                return True
            else:
                self.kernel32.CloseHandle(h_timer)
                return False
                
        except Exception as e:
            print(f"Failed to create wake timer: {e}")
            return False
    
    def _cancel_wake_timer(self, task_id: str):
        """Cancel a wake timer"""
        with self.timer_lock:
            h_timer = self.active_wake_timers.pop(task_id, None)
        
        if h_timer:
            self.kernel32.CloseHandle(h_timer)
    
    def _prepare_wake_timers(self):
        """Ensure all wake timers are set before sleeping"""
        now = datetime.now()
        
        for task in self.scheduled_tasks.values():
            if task.wake_required and task.scheduled_time > now:
                if task.task_id not in self.active_wake_timers:
                    self._create_wake_timer(task)
    
    def _initiate_sleep(self) -> bool:
        """Initiate S3 sleep"""
        # Use SetSuspendState or ExitWindowsEx
        # This requires additional Windows API calls
        # For safety, we just allow sleep and let Windows handle it
        self.allow_sleep()
        return True
    
    def _initiate_hibernate(self) -> bool:
        """Initiate S4 hibernate"""
        # Similar to sleep but forces hibernate
        self.allow_sleep()
        return True
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Check power status
                status = self.get_power_status()
                
                # Check for upcoming tasks
                now = datetime.now()
                upcoming = [t for t in self.scheduled_tasks.values()
                           if t.scheduled_time < now + timedelta(minutes=5)]
                
                # Prevent sleep if tasks are upcoming
                if upcoming:
                    self.prevent_sleep()
                else:
                    self.allow_sleep()
                
            except Exception as e:
                print(f"Power monitor error: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        
        # Cancel all timers
        with self.timer_lock:
            for h_timer in self.active_wake_timers.values():
                self.kernel32.CloseHandle(h_timer)
            self.active_wake_timers.clear()
        
        # Allow sleep
        self.allow_sleep()


# ==================== Usage Example ====================

def main():
    """Example usage of OpenClawPowerManager"""
    
    # Create power manager
    pm = OpenClawPowerManager()
    
    # Start monitoring
    pm.start_monitoring()
    
    # Schedule a critical task
    def critical_task():
        print("Executing critical task!")
    
    task = ScheduledTask(
        task_id="heartbeat",
        scheduled_time=datetime.now() + timedelta(hours=1),
        callback=critical_task,
        wake_required=True,
        priority="critical"
    )
    pm.schedule_wake_task(task)
    
    # Prevent sleep during important operation
    pm.prevent_sleep(display_required=False)
    
    try:
        # Do important work
        time.sleep(10)
    finally:
        pm.allow_sleep()
    
    # Get power status
    status = pm.get_power_status()
    print(f"Power: {'AC' if status.on_ac_power else 'Battery'} "
          f"({status.battery_percent}%)")
    
    # Recommend power state
    recommended = pm.recommend_power_state()
    print(f"Recommended state: {recommended.value}")
    
    # Cleanup
    pm.cleanup()


if __name__ == "__main__":
    main()
```

---

## Configuration Summary

### Powercfg Commands Reference

```powershell
# Power Configuration Quick Reference

# List all power plans
powercfg /list

# Set active power plan
powercfg /setactive <GUID>

# Common power plan GUIDs:
# - Balanced: 381b4222-f694-41f0-9685-ff5bb260df2e
# - High Performance: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
# - Power Saver: a1841308-3541-4fab-bc81-f71556f20b4a

# Check available sleep states
powercfg /a

# Enable/disable hibernate
powercfg /hibernate on
powercfg /hibernate off

# Set sleep timeout (minutes, 0 = never)
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 10

# Set display timeout
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 5

# Create custom power plan
powercfg /duplicatescheme <GUID> <new-name>

# Export/import power settings
powercfg /export <filename.pow> <GUID>
powercfg /import <filename.pow>

# Query specific setting
powercfg /query <GUID> <subgroup> <setting>
```

### Registry Settings for WoL

```
; Enable Wake-on-LAN via Registry
; Save as .reg file and import

Windows Registry Editor Version 5.00

; Enable WoL on network adapter
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\NDIS\Parameters]
"EnablePME"=dword:00000001

; Disable Fast Startup
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Power]
"HiberbootEnabled"=dword:00000000
```

---

## Conclusion

This technical specification provides a comprehensive framework for implementing power management in a 24/7 Windows 10 AI agent system. The architecture ensures:

1. **Continuous Operation**: Wake timers and sleep prevention keep the agent available
2. **Energy Efficiency**: Intelligent power state decisions minimize consumption
3. **Battery Awareness**: Laptop operation adapts to power conditions
4. **Remote Management**: Wake-on-LAN enables remote system control
5. **Graceful Transitions**: Power event handling ensures data integrity

Key implementation files:
- Wake timer management with `CreateWaitableTimer` and `SetWaitableTimer`
- Sleep prevention using `SetThreadExecutionState`
- Power event handling via `WM_POWERBROADCAST` messages
- Battery monitoring through `GetSystemPowerStatus`
- WoL magic packet generation and transmission

---

*Document Version: 1.0*
*Last Updated: 2025*
*For: OpenClaw Windows 10 AI Agent System*
