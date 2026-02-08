# Windows Task Scheduler & Job Automation Integration Specification
## OpenClaw-Inspired AI Agent System for Windows 10

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Task Scheduler API Integration](#task-scheduler-api-integration)
4. [Task Creation and Configuration](#task-creation-and-configuration)
5. [Trigger Types](#trigger-types)
6. [Action Definitions](#action-definitions)
7. [Task Folder Organization](#task-folder-organization)
8. [Task Monitoring and Management](#task-monitoring-and-management)
9. [Cron-to-Task Scheduler Translation](#cron-to-task-scheduler-translation)
10. [Agent Heartbeat Implementation](#agent-heartbeat-implementation)
11. [Code Examples](#code-examples)
12. [Security Considerations](#security-considerations)

---

## Executive Summary

This specification defines the Windows Task Scheduler integration for a Windows 10 OpenClaw-inspired AI agent system. The system provides:

- **24/7 Agent Operation**: Continuous heartbeat monitoring and periodic wake-up mechanisms
- **15 Hardcoded Agentic Loops**: Automated task execution patterns
- **Cron Job Equivalent**: Full Windows-native scheduling capabilities
- **Comprehensive Monitoring**: Task state tracking, failure recovery, and health checks

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI Agent Task Scheduler System                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐     │
│  │  Task Scheduler │    │   Agent Core     │    │   Heartbeat Monitor │     │
│  │     Service     │◄──►│   (GPT-5.2)      │◄──►│     (24/7 Loop)     │     │
│  └─────────────────┘    └──────────────────┘    └─────────────────────┘     │
│           │                      │                       │                  │
│           ▼                      ▼                       ▼                  │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐     │
│  │  Task Folders   │    │  15 Agent Loops  │    │   Cron Translator   │     │
│  │  (Organized)    │    │  (Hardcoded)     │    │   (Pattern Match)   │     │
│  └─────────────────┘    └──────────────────┘    └─────────────────────┘     │
│           │                      │                       │                  │
│           ▼                      ▼                       ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Windows Task Scheduler API                        │   │
│  │  (ITaskService, ITaskDefinition, ITrigger, IAction, ITaskFolder)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Windows Task Scheduler Service                    │   │
│  │                         (taskschd.msc)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Task Scheduler API Integration

### Core COM Interfaces

| Interface | Purpose | CLSID/ProgID |
|-----------|---------|--------------|
| `ITaskService` | Gateway to Task Scheduler service | `CLSID_TaskScheduler` |
| `ITaskDefinition` | Defines task components (triggers, actions, settings) | Via `ITaskService::NewTask` |
| `ITriggerCollection` | Collection of task triggers | Via `ITaskDefinition::get_Triggers` |
| `IActionCollection` | Collection of task actions | Via `ITaskDefinition::get_Actions` |
| `ITaskFolder` | Manages task folders and registration | Via `ITaskService::GetFolder` |
| `IRegisteredTask` | Represents a registered task | Via `ITaskFolder::GetTask` |
| `ITaskSettings` | Configures task execution settings | Via `ITaskDefinition::get_Settings` |
| `IPrincipal` | Defines security credentials | Via `ITaskDefinition::get_Principal` |

### Connection Pattern

```cpp
// C++ COM Connection Pattern
HRESULT ConnectToTaskScheduler(ITaskService** ppService) {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) return hr;
    
    hr = CoInitializeSecurity(
        NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_PKT_PRIVACY,
        RPC_C_IMP_LEVEL_IMPERSONATE, NULL, 0, NULL);
    
    ITaskService* pService = NULL;
    hr = CoCreateInstance(CLSID_TaskScheduler, NULL, CLSCTX_INPROC_SERVER,
                          IID_ITaskService, (void**)&pService);
    if (FAILED(hr)) return hr;
    
    // Connect to local Task Scheduler service
    hr = pService->Connect(_variant_t(), _variant_t(), _variant_t(), _variant_t());
    if (FAILED(hr)) {
        pService->Release();
        return hr;
    }
    
    *ppService = pService;
    return S_OK;
}
```

### Python COM Integration (pywin32)

```python
"""
Windows Task Scheduler COM Interface for Python
OpenClaw AI Agent Integration
"""
import win32com.client
import pythoncom
from datetime import datetime
from enum import IntEnum
from typing import Optional, List, Dict, Any

class TASK_STATE(IntEnum):
    """Task execution states"""
    UNKNOWN = 0
    DISABLED = 1
    QUEUED = 2
    READY = 3
    RUNNING = 4

class TASK_TRIGGER_TYPE(IntEnum):
    """Trigger type enumeration"""
    EVENT = 0
    TIME = 1
    DAILY = 2
    WEEKLY = 3
    MONTHLY = 4
    MONTHLYDOW = 5
    IDLE = 6
    REGISTRATION = 7
    BOOT = 8
    LOGON = 9
    SESSION_STATE_CHANGE = 11

class TASK_ACTION_TYPE(IntEnum):
    """Action type enumeration"""
    EXEC = 0
    COM_HANDLER = 5
    SEND_EMAIL = 6
    SHOW_MESSAGE = 7

class TASK_RUNLEVEL(IntEnum):
    """Task run level"""
    LUA = 0      # Least Privilege User Account
    HIGHEST = 1  # Highest available privileges

class TASK_LOGON_TYPE(IntEnum):
    """Logon type for task execution"""
    NONE = 0
    PASSWORD = 1
    S4U = 2
    INTERACTIVE_TOKEN = 3
    GROUP = 4
    SERVICE_ACCOUNT = 5
    INTERACTIVE_TOKEN_OR_PASSWORD = 6

class TaskSchedulerService:
    """
    Windows Task Scheduler Service Wrapper
    Provides high-level interface to Task Scheduler COM API
    """
    
    def __init__(self):
        self.scheduler = None
        self.connected = False
        
    def connect(self, server: Optional[str] = None, 
                user: Optional[str] = None,
                domain: Optional[str] = None,
                password: Optional[str] = None) -> bool:
        """Connect to Task Scheduler service"""
        try:
            pythoncom.CoInitialize()
            self.scheduler = win32com.client.Dispatch("Schedule.Service")
            self.scheduler.Connect(server, user, domain, password)
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Task Scheduler: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Task Scheduler service"""
        self.scheduler = None
        self.connected = False
        pythoncom.CoUninitialize()
    
    def get_folder(self, path: str = "\\") -> Any:
        """Get task folder by path"""
        if not self.connected:
            raise RuntimeError("Not connected to Task Scheduler")
        return self.scheduler.GetFolder(path)
    
    def create_folder(self, path: str, sddl: Optional[str] = None) -> Any:
        """Create new task folder"""
        root = self.get_folder("\\")
        return root.CreateFolder(path, sddl)
    
    def get_running_tasks(self, flags: int = 0) -> Any:
        """Get collection of running tasks"""
        return self.scheduler.GetRunningTasks(flags)
    
    def new_task_definition(self, flags: int = 0) -> Any:
        """Create new task definition object"""
        return self.scheduler.NewTask(flags)
```

---

## Task Creation and Configuration

### Task Definition Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITaskDefinition Structure                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ RegistrationInfo│  │    Principal    │  │    Settings     │ │
│  │  - Author       │  │  - UserId       │  │  - Enabled      │ │
│  │  - Description  │  │  - LogonType    │  │  - Hidden       │ │
│  │  - Date         │  │  - RunLevel     │  │  - Priority     │ │
│  │  - URI          │  │  - GroupId      │  │  - ExecutionTime│ │ │
│  │  - Version      │  │                 │  │  - RestartCount │ │
│  │  - Documentation│  │                 │  │  - WakeToRun    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    Triggers     │  │     Actions     │  │      Data       │ │
│  │  (Collection)   │  │  (Collection)   │  │  (Custom data)  │ │
│  │  - Time-based   │  │  - Executable   │  │                 │ │
│  │  - Event-based  │  │  - COM Handler  │  │                 │ │
│  │  - Boot/Logon   │  │  - Email        │  │                 │ │
│  │  - Idle/Session │  │  - Message      │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Task Registration Process

```python
class TaskBuilder:
    """
    Builder pattern for creating Windows Scheduled Tasks
    OpenClaw AI Agent Task Builder
    """
    
    def __init__(self, scheduler: TaskSchedulerService):
        self.scheduler = scheduler
        self.definition = None
        self.reset()
    
    def reset(self):
        """Reset builder state"""
        self.definition = self.scheduler.new_task_definition(0)
        self._configure_defaults()
        return self
    
    def _configure_defaults(self):
        """Configure default task settings"""
        # Registration info
        reg_info = self.definition.RegistrationInfo
        reg_info.Author = "OpenClaw AI Agent"
        reg_info.Description = "Automated AI Agent Task"
        reg_info.Date = datetime.now().isoformat()
        
        # Settings
        settings = self.definition.Settings
        settings.Enabled = True
        settings.Hidden = False
        settings.StartWhenAvailable = True
        settings.AllowHardTerminate = True
        settings.ExecutionTimeLimit = "PT1H"  # 1 hour
        settings.MultipleInstances = 0  # TASK_INSTANCES_PARALLEL
        settings.Priority = 7  # Normal priority
        settings.RunOnlyIfNetworkAvailable = False
        settings.DisallowStartIfOnBatteries = False
        settings.StopIfGoingOnBatteries = False
        settings.WakeToRun = True
        
        # Principal
        principal = self.definition.Principal
        principal.LogonType = TASK_LOGON_TYPE.SERVICE_ACCOUNT
        principal.RunLevel = TASK_RUNLEVEL.HIGHEST
        principal.UserId = "NT AUTHORITY\\SYSTEM"
    
    def with_name(self, name: str) -> 'TaskBuilder':
        """Set task name"""
        self.task_name = name
        return self
    
    def with_description(self, description: str) -> 'TaskBuilder':
        """Set task description"""
        self.definition.RegistrationInfo.Description = description
        return self
    
    def with_author(self, author: str) -> 'TaskBuilder':
        """Set task author"""
        self.definition.RegistrationInfo.Author = author
        return self
    
    def with_principal(self, user_id: str, 
                       logon_type: TASK_LOGON_TYPE = TASK_LOGON_TYPE.SERVICE_ACCOUNT,
                       run_level: TASK_RUNLEVEL = TASK_RUNLEVEL.HIGHEST) -> 'TaskBuilder':
        """Configure task principal (security context)"""
        principal = self.definition.Principal
        principal.UserId = user_id
        principal.LogonType = logon_type
        principal.RunLevel = run_level
        return self
    
    def with_settings(self, **kwargs) -> 'TaskBuilder':
        """Configure task settings"""
        settings = self.definition.Settings
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return self
    
    def add_trigger(self, trigger_type: TASK_TRIGGER_TYPE, **kwargs) -> 'TaskBuilder':
        """Add trigger to task"""
        triggers = self.definition.Triggers
        trigger = triggers.Create(trigger_type)
        
        for key, value in kwargs.items():
            if hasattr(trigger, key):
                setattr(trigger, key, value)
        
        return self
    
    def add_action(self, action_type: TASK_ACTION_TYPE, **kwargs) -> 'TaskBuilder':
        """Add action to task"""
        actions = self.definition.Actions
        action = actions.Create(action_type)
        
        for key, value in kwargs.items():
            if hasattr(action, key):
                setattr(action, key, value)
        
        return self
    
    def register(self, folder_path: str = "\\OpenClaw",
                 create_flags: int = 6,  # TASK_CREATE_OR_UPDATE
                 password: Optional[str] = None) -> Any:
        """Register task in Task Scheduler"""
        try:
            folder = self.scheduler.get_folder(folder_path)
        except:
            # Create folder if it doesn't exist
            root = self.scheduler.get_folder("\\")
            folder = root.CreateFolder(folder_path.replace("\\", ""), None)
        
        registered_task = folder.RegisterTaskDefinition(
            self.task_name,
            self.definition,
            create_flags,
            None,  # UserId (use from principal)
            password,
            self.definition.Principal.LogonType,
            None   # sddl
        )
        
        return registered_task
```

---

## Trigger Types

### Complete Trigger Type Reference

| Trigger Type | COM Interface | Use Case | Cron Equivalent |
|--------------|---------------|----------|-----------------|
| **Time** | `ITimeTrigger` | One-time execution at specific time | `@once` |
| **Daily** | `IDailyTrigger` | Daily recurring tasks | `0 0 * * *` |
| **Weekly** | `IWeeklyTrigger` | Weekly recurring tasks | `0 0 * * 0` |
| **Monthly** | `IMonthlyTrigger` | Monthly on specific days | `0 0 1 * *` |
| **MonthlyDOW** | `IMonthlyDOWTrigger` | Monthly on day-of-week | `0 0 * * 1#1` |
| **Event** | `IEventTrigger` | System event-based | N/A |
| **Boot** | `IBootTrigger` | System startup | `@reboot` |
| **Logon** | `ILogonTrigger` | User logon | N/A |
| **Idle** | `IIdleTrigger` | System idle state | N/A |
| **Registration** | `IRegistrationTrigger` | Task creation/update | N/A |
| **SessionStateChange** | `ISessionStateChangeTrigger` | Session changes | N/A |

### Trigger Implementation Examples

```python
class TriggerFactory:
    """Factory for creating Task Scheduler triggers"""
    
    @staticmethod
    def create_time_trigger(start_boundary: datetime,
                           end_boundary: Optional[datetime] = None,
                           enabled: bool = True) -> Dict[str, Any]:
        """
        Create time trigger (one-time execution)
        
        Args:
            start_boundary: When to execute (ISO 8601 format)
            end_boundary: Optional end boundary
            enabled: Whether trigger is enabled
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.TIME,
            'StartBoundary': start_boundary.isoformat(),
            'Enabled': enabled
        }
        if end_boundary:
            trigger['EndBoundary'] = end_boundary.isoformat()
        return trigger
    
    @staticmethod
    def create_daily_trigger(start_boundary: datetime,
                             days_interval: int = 1,
                             repetition_interval: Optional[str] = None,
                             repetition_duration: Optional[str] = None) -> Dict[str, Any]:
        """
        Create daily trigger
        
        Args:
            start_boundary: Start date/time
            days_interval: Interval in days (default: 1 = every day)
            repetition_interval: ISO 8601 duration (e.g., "PT15M" = 15 minutes)
            repetition_duration: How long to repeat (e.g., "PT1H" = 1 hour)
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.DAILY,
            'StartBoundary': start_boundary.isoformat(),
            'DaysInterval': days_interval,
            'Enabled': True
        }
        
        if repetition_interval:
            trigger['RepetitionInterval'] = repetition_interval
        if repetition_duration:
            trigger['RepetitionDuration'] = repetition_duration
            
        return trigger
    
    @staticmethod
    def create_weekly_trigger(start_boundary: datetime,
                              days_of_week: int,  # Bitmask: 1=Sun, 2=Mon, 4=Tue, etc.
                              weeks_interval: int = 1) -> Dict[str, Any]:
        """
        Create weekly trigger
        
        Args:
            start_boundary: Start date/time
            days_of_week: Bitmask of days (e.g., 0x3F = Mon-Sat)
            weeks_interval: Interval in weeks
        """
        return {
            'Type': TASK_TRIGGER_TYPE.WEEKLY,
            'StartBoundary': start_boundary.isoformat(),
            'DaysOfWeek': days_of_week,
            'WeeksInterval': weeks_interval,
            'Enabled': True
        }
    
    @staticmethod
    def create_monthly_trigger(start_boundary: datetime,
                               days_of_month: int,  # Bitmask: 1=1st, 2=2nd, 4=3rd, etc.
                               months_of_year: int = 0xFFF) -> Dict[str, Any]:
        """
        Create monthly trigger
        
        Args:
            start_boundary: Start date/time
            days_of_month: Bitmask of days (1-31)
            months_of_year: Bitmask of months (0xFFF = all months)
        """
        return {
            'Type': TASK_TRIGGER_TYPE.MONTHLY,
            'StartBoundary': start_boundary.isoformat(),
            'DaysOfMonth': days_of_month,
            'MonthsOfYear': months_of_year,
            'Enabled': True
        }
    
    @staticmethod
    def create_monthly_dow_trigger(start_boundary: datetime,
                                   days_of_week: int,
                                   weeks_of_month: int,  # 1=1st, 2=2nd, 4=3rd, 8=4th, 0x10=Last
                                   months_of_year: int = 0xFFF) -> Dict[str, Any]:
        """
        Create monthly day-of-week trigger
        
        Args:
            start_boundary: Start date/time
            days_of_week: Bitmask of days
            weeks_of_month: Bitmask of weeks
            months_of_year: Bitmask of months
        """
        return {
            'Type': TASK_TRIGGER_TYPE.MONTHLYDOW,
            'StartBoundary': start_boundary.isoformat(),
            'DaysOfWeek': days_of_week,
            'WeeksOfMonth': weeks_of_month,
            'MonthsOfYear': months_of_year,
            'Enabled': True
        }
    
    @staticmethod
    def create_boot_trigger(delay: Optional[str] = None) -> Dict[str, Any]:
        """
        Create boot trigger
        
        Args:
            delay: ISO 8601 duration to delay after boot
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.BOOT,
            'Enabled': True
        }
        if delay:
            trigger['Delay'] = delay
        return trigger
    
    @staticmethod
    def create_logon_trigger(user_id: Optional[str] = None,
                             delay: Optional[str] = None) -> Dict[str, Any]:
        """
        Create logon trigger
        
        Args:
            user_id: Specific user (None = any user)
            delay: ISO 8601 duration to delay after logon
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.LOGON,
            'Enabled': True
        }
        if user_id:
            trigger['UserId'] = user_id
        if delay:
            trigger['Delay'] = delay
        return trigger
    
    @staticmethod
    def create_idle_trigger() -> Dict[str, Any]:
        """Create idle trigger - fires when system becomes idle"""
        return {
            'Type': TASK_TRIGGER_TYPE.IDLE,
            'Enabled': True
        }
    
    @staticmethod
    def create_event_trigger(subscription: str,
                             delay: Optional[str] = None,
                             value_queries: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """
        Create event-based trigger
        
        Args:
            subscription: XPath event query
            delay: ISO 8601 duration to delay
            value_queries: List of (name, XPath) tuples for value extraction
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.EVENT,
            'Subscription': subscription,
            'Enabled': True
        }
        if delay:
            trigger['Delay'] = delay
        if value_queries:
            trigger['ValueQueries'] = value_queries
        return trigger
    
    @staticmethod
    def create_registration_trigger(delay: Optional[str] = None) -> Dict[str, Any]:
        """Create registration trigger - fires when task is registered/updated"""
        trigger = {
            'Type': TASK_TRIGGER_TYPE.REGISTRATION,
            'Enabled': True
        }
        if delay:
            trigger['Delay'] = delay
        return trigger
    
    @staticmethod
    def create_session_state_trigger(state_change: int,
                                     user_id: Optional[str] = None,
                                     delay: Optional[str] = None) -> Dict[str, Any]:
        """
        Create session state change trigger
        
        Args:
            state_change: Session state change type
                1 = Console connect
                2 = Console disconnect
                3 = Remote connect
                4 = Remote disconnect
                5 = Session lock
                6 = Session unlock
            user_id: Specific user (None = any user)
            delay: ISO 8601 duration to delay
        """
        trigger = {
            'Type': TASK_TRIGGER_TYPE.SESSION_STATE_CHANGE,
            'StateChange': state_change,
            'Enabled': True
        }
        if user_id:
            trigger['UserId'] = user_id
        if delay:
            trigger['Delay'] = delay
        return trigger
```

### Day of Week Bitmask Reference

```python
class DaysOfWeek:
    """Day of week bitmask constants"""
    SUNDAY = 0x01
    MONDAY = 0x02
    TUESDAY = 0x04
    WEDNESDAY = 0x08
    THURSDAY = 0x10
    FRIDAY = 0x20
    SATURDAY = 0x40
    WEEKDAYS = MONDAY | TUESDAY | WEDNESDAY | THURSDAY | FRIDAY
    WEEKENDS = SUNDAY | SATURDAY
    ALL = 0x7F

class WeeksOfMonth:
    """Weeks of month bitmask constants"""
    FIRST = 0x01
    SECOND = 0x02
    THIRD = 0x04
    FOURTH = 0x08
    LAST = 0x10
    ALL = 0x1F

class MonthsOfYear:
    """Months of year bitmask constants"""
    JANUARY = 0x01
    FEBRUARY = 0x02
    MARCH = 0x04
    APRIL = 0x08
    MAY = 0x10
    JUNE = 0x20
    JULY = 0x40
    AUGUST = 0x80
    SEPTEMBER = 0x100
    OCTOBER = 0x200
    NOVEMBER = 0x400
    DECEMBER = 0x800
    ALL = 0xFFF
```

---

## Action Definitions

### Action Types

| Action Type | COM Interface | Purpose |
|-------------|---------------|---------|
| **Execute** | `IExecAction` | Run executable, script, or command |
| **COM Handler** | `IComHandlerAction` | Invoke COM component |
| **Send Email** | `IEmailAction` | Send email notification |
| **Show Message** | `IShowMessageAction` | Display message box |

### Action Implementation

```python
class ActionFactory:
    """Factory for creating Task Scheduler actions"""
    
    @staticmethod
    def create_exec_action(path: str,
                          arguments: Optional[str] = None,
                          working_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Create executable action
        
        Args:
            path: Path to executable
            arguments: Command line arguments
            working_directory: Working directory for execution
        """
        action = {
            'Type': TASK_ACTION_TYPE.EXEC,
            'Path': path
        }
        if arguments:
            action['Arguments'] = arguments
        if working_directory:
            action['WorkingDirectory'] = working_directory
        return action
    
    @staticmethod
    def create_com_handler_action(clsid: str,
                                   data: Optional[str] = None) -> Dict[str, Any]:
        """
        Create COM handler action
        
        Args:
            clsid: COM class ID (GUID)
            data: Optional data to pass to handler
        """
        action = {
            'Type': TASK_ACTION_TYPE.COM_HANDLER,
            'ClassId': clsid
        }
        if data:
            action['Data'] = data
        return action
    
    @staticmethod
    def create_email_action(server: str,
                            to: str,
                            from_addr: str,
                            subject: str,
                            body: str,
                            attachments: Optional[List[str]] = None,
                            cc: Optional[str] = None,
                            bcc: Optional[str] = None) -> Dict[str, Any]:
        """
        Create email action (deprecated in Windows 8+)
        
        Args:
            server: SMTP server
            to: Recipient address
            from_addr: Sender address
            subject: Email subject
            body: Email body
            attachments: List of attachment paths
            cc: CC recipients
            bcc: BCC recipients
        """
        action = {
            'Type': TASK_ACTION_TYPE.SEND_EMAIL,
            'Server': server,
            'To': to,
            'From': from_addr,
            'Subject': subject,
            'Body': body
        }
        if attachments:
            action['Attachments'] = attachments
        if cc:
            action['Cc'] = cc
        if bcc:
            action['Bcc'] = bcc
        return action
    
    @staticmethod
    def create_message_action(title: str, message: str) -> Dict[str, Any]:
        """
        Create message box action (deprecated in Windows 8+)
        
        Args:
            title: Message box title
            message: Message text
        """
        return {
            'Type': TASK_ACTION_TYPE.SHOW_MESSAGE,
            'Title': title,
            'Message': message
        }
```

### PowerShell Action Pattern

```python
class PowerShellActionBuilder:
    """Builder for PowerShell script actions"""
    
    @staticmethod
    def create_powershell_script_action(script_path: str,
                                        execution_policy: str = "Bypass",
                                        arguments: Optional[str] = None,
                                        window_style: str = "Hidden") -> Dict[str, Any]:
        """
        Create PowerShell script execution action
        
        Args:
            script_path: Path to .ps1 script
            execution_policy: PowerShell execution policy
            arguments: Additional arguments to pass to script
            window_style: Window style (Hidden, Normal, Minimized, Maximized)
        """
        ps_args = f"-ExecutionPolicy {execution_policy} -WindowStyle {window_style} -File \"{script_path}\""
        
        if arguments:
            ps_args += f" {arguments}"
        
        return ActionFactory.create_exec_action(
            path="powershell.exe",
            arguments=ps_args
        )
    
    @staticmethod
    def create_powershell_command_action(command: str,
                                         execution_policy: str = "Bypass",
                                         window_style: str = "Hidden") -> Dict[str, Any]:
        """
        Create PowerShell command execution action
        
        Args:
            command: PowerShell command to execute
            execution_policy: PowerShell execution policy
            window_style: Window style
        """
        ps_args = f"-ExecutionPolicy {execution_policy} -WindowStyle {window_style} -Command \"{command}\""
        
        return ActionFactory.create_exec_action(
            path="powershell.exe",
            arguments=ps_args
        )
    
    @staticmethod
    def create_encoded_command_action(encoded_command: str,
                                      execution_policy: str = "Bypass") -> Dict[str, Any]:
        """
        Create encoded PowerShell command action (for complex commands)
        
        Args:
            encoded_command: Base64-encoded PowerShell command
            execution_policy: PowerShell execution policy
        """
        ps_args = f"-ExecutionPolicy {execution_policy} -EncodedCommand {encoded_command}"
        
        return ActionFactory.create_exec_action(
            path="powershell.exe",
            arguments=ps_args
        )
```

---

## Task Folder Organization

### Recommended Folder Structure

```
\
├── OpenClaw\                              # Root AI Agent folder
│   ├── Core\                               # Core agent tasks
│   │   ├── Heartbeat                       # Agent heartbeat task
│   │   ├── SoulMaintenance                 # Identity/soul maintenance
│   │   └── SystemMonitor                   # System health monitoring
│   │
│   ├── AgentLoops\                         # 15 hardcoded agentic loops
│   │   ├── Loop_01_EmailProcessor          # Gmail integration
│   │   ├── Loop_02_BrowserAutomation       # Browser control
│   │   ├── Loop_03_VoiceHandler            # TTS/STT processing
│   │   ├── Loop_04_Communication           # Twilio voice/SMS
│   │   ├── Loop_05_FileOperations          # File system operations
│   │   ├── Loop_06_NetworkMonitor          # Network monitoring
│   │   ├── Loop_07_ProcessManager          # Process management
│   │   ├── Loop_08_RegistryWatcher         # Registry monitoring
│   │   ├── Loop_09_EventProcessor          # Windows event processing
│   │   ├── Loop_10_SchedulerCoordinator    # Task coordination
│   │   ├── Loop_11_UserActivity            # User activity tracking
│   │   ├── Loop_12_SecurityMonitor         # Security monitoring
│   │   ├── Loop_13_BackupManager           # Backup operations
│   │   ├── Loop_14_UpdateChecker           # Update checking
│   │   └── Loop_15_Maintenance             # System maintenance
│   │
│   ├── Triggers\                           # Event-based triggers
│   │   ├── OnBoot                          # System startup tasks
│   │   ├── OnLogon                         # User logon tasks
│   │   ├── OnIdle                          # Idle state tasks
│   │   └── OnEvent                         # Event-based tasks
│   │
│   └── Maintenance\                        # Maintenance tasks
│       ├── LogCleanup                      # Log file cleanup
│       ├── TempCleanup                     # Temporary file cleanup
│       └── TaskHealthCheck                 # Task health verification
│
└── Microsoft\                              # Standard Windows folders
    └── Windows\                            # (Do not modify)
```

### Folder Management Implementation

```python
class TaskFolderManager:
    """Manages Task Scheduler folder hierarchy for AI Agent"""
    
    ROOT_FOLDER = "\\OpenClaw"
    
    FOLDER_STRUCTURE = {
        "Core": ["Heartbeat", "SoulMaintenance", "SystemMonitor"],
        "AgentLoops": [
            "Loop_01_EmailProcessor",
            "Loop_02_BrowserAutomation",
            "Loop_03_VoiceHandler",
            "Loop_04_Communication",
            "Loop_05_FileOperations",
            "Loop_06_NetworkMonitor",
            "Loop_07_ProcessManager",
            "Loop_08_RegistryWatcher",
            "Loop_09_EventProcessor",
            "Loop_10_SchedulerCoordinator",
            "Loop_11_UserActivity",
            "Loop_12_SecurityMonitor",
            "Loop_13_BackupManager",
            "Loop_14_UpdateChecker",
            "Loop_15_Maintenance"
        ],
        "Triggers": ["OnBoot", "OnLogon", "OnIdle", "OnEvent"],
        "Maintenance": ["LogCleanup", "TempCleanup", "TaskHealthCheck"]
    }
    
    def __init__(self, scheduler: TaskSchedulerService):
        self.scheduler = scheduler
        self._folders_created = False
    
    def create_folder_structure(self) -> bool:
        """Create complete folder structure"""
        try:
            root = self.scheduler.get_folder("\\")
            
            # Create root folder
            try:
                openclaw_folder = root.CreateFolder("OpenClaw", None)
            except:
                openclaw_folder = self.scheduler.get_folder(self.ROOT_FOLDER)
            
            # Create subfolders
            for folder_name, _ in self.FOLDER_STRUCTURE.items():
                folder_path = f"{self.ROOT_FOLDER}\\{folder_name}"
                try:
                    root.CreateFolder(folder_path.replace("\\", "").replace("OpenClaw", "OpenClaw\\"), None)
                except:
                    pass  # Folder may already exist
            
            self._folders_created = True
            return True
        except Exception as e:
            print(f"Failed to create folder structure: {e}")
            return False
    
    def get_folder(self, path: str) -> Any:
        """Get folder by relative path"""
        full_path = f"{self.ROOT_FOLDER}\\{path}" if not path.startswith("\\") else path
        return self.scheduler.get_folder(full_path)
    
    def delete_folder(self, path: str, force: bool = False) -> bool:
        """Delete folder and optionally all tasks within"""
        try:
            folder = self.get_folder(path)
            if force:
                # Delete all tasks in folder first
                tasks = folder.GetTasks(0)
                for task in tasks:
                    folder.DeleteTask(task.Name, 0)
            
            parent_path = "\\".join(path.rstrip("\\").split("\\")[:-1]) or "\\"
            parent = self.scheduler.get_folder(parent_path)
            folder_name = path.rstrip("\\").split("\\")[-1]
            parent.DeleteFolder(folder_name, 0)
            return True
        except Exception as e:
            print(f"Failed to delete folder: {e}")
            return False
    
    def list_all_tasks(self) -> Dict[str, List[str]]:
        """List all tasks in the OpenClaw folder structure"""
        tasks = {}
        
        for folder_name in self.FOLDER_STRUCTURE.keys():
            folder_path = f"{self.ROOT_FOLDER}\\{folder_name}"
            try:
                folder = self.scheduler.get_folder(folder_path)
                folder_tasks = folder.GetTasks(0)
                tasks[folder_name] = [task.Name for task in folder_tasks]
            except:
                tasks[folder_name] = []
        
        return tasks
```

---

## Task Monitoring and Management

### Task State Monitoring

```python
class TaskMonitor:
    """Monitor and manage scheduled tasks"""
    
    TASK_STATE_NAMES = {
        0: 'Unknown',
        1: 'Disabled',
        2: 'Queued',
        3: 'Ready',
        4: 'Running'
    }
    
    def __init__(self, scheduler: TaskSchedulerService):
        self.scheduler = scheduler
    
    def get_task_info(self, task_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task"""
        try:
            parts = task_path.rsplit('\\', 1)
            if len(parts) == 2:
                folder_path, task_name = parts
            else:
                folder_path = "\\"
                task_name = parts[0]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            definition = task.Definition
            settings = definition.Settings
            
            return {
                'Name': task.Name,
                'Path': task.Path,
                'State': self.TASK_STATE_NAMES.get(task.State, 'Unknown'),
                'StateCode': task.State,
                'Enabled': task.Enabled,
                'LastRunTime': task.LastRunTime,
                'LastTaskResult': task.LastTaskResult,
                'NextRunTime': task.NextRunTime,
                'NumberOfMissedRuns': task.NumberOfMissedRuns,
                'Definition': {
                    'Author': definition.RegistrationInfo.Author,
                    'Description': definition.RegistrationInfo.Description,
                    'Date': definition.RegistrationInfo.Date,
                    'Version': definition.RegistrationInfo.Version,
                    'Settings': {
                        'Enabled': settings.Enabled,
                        'Hidden': settings.Hidden,
                        'Priority': settings.Priority,
                        'ExecutionTimeLimit': settings.ExecutionTimeLimit,
                        'WakeToRun': settings.WakeToRun,
                        'StartWhenAvailable': settings.StartWhenAvailable,
                        'RunOnlyIfNetworkAvailable': settings.RunOnlyIfNetworkAvailable,
                        'DisallowStartIfOnBatteries': settings.DisallowStartIfOnBatteries,
                        'StopIfGoingOnBatteries': settings.StopIfGoingOnBatteries,
                        'RestartCount': settings.RestartCount,
                        'RestartInterval': settings.RestartInterval if hasattr(settings, 'RestartInterval') else None
                    }
                }
            }
        except Exception as e:
            print(f"Failed to get task info: {e}")
            return None
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get list of currently running tasks"""
        running_tasks = []
        
        try:
            tasks = self.scheduler.get_running_tasks(0)
            for task in tasks:
                running_tasks.append({
                    'Name': task.Name,
                    'Path': task.Path,
                    'InstanceGuid': str(task.InstanceGuid),
                    'CurrentAction': task.CurrentAction,
                    'EnginePID': task.EnginePID
                })
        except Exception as e:
            print(f"Failed to get running tasks: {e}")
        
        return running_tasks
    
    def is_task_running(self, task_path: str) -> bool:
        """Check if a specific task is currently running"""
        info = self.get_task_info(task_path)
        return info is not None and info['StateCode'] == TASK_STATE.RUNNING
    
    def start_task(self, task_path: str, arguments: Optional[str] = None) -> bool:
        """Start a task manually"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            # Run the task
            running_task = task.Run(_variant_t(arguments) if arguments else _variant_t())
            return True
        except Exception as e:
            print(f"Failed to start task: {e}")
            return False
    
    def stop_task(self, task_path: str) -> bool:
        """Stop a running task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            # Stop the task
            task.Stop(0)
            return True
        except Exception as e:
            print(f"Failed to stop task: {e}")
            return False
    
    def enable_task(self, task_path: str) -> bool:
        """Enable a disabled task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            definition = task.Definition
            definition.Settings.Enabled = True
            
            folder.RegisterTaskDefinition(
                task_name,
                definition,
                6,  # TASK_CREATE_OR_UPDATE
                None, None,
                definition.Principal.LogonType,
                None
            )
            return True
        except Exception as e:
            print(f"Failed to enable task: {e}")
            return False
    
    def disable_task(self, task_path: str) -> bool:
        """Disable a task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            definition = task.Definition
            definition.Settings.Enabled = False
            
            folder.RegisterTaskDefinition(
                task_name,
                definition,
                6,  # TASK_CREATE_OR_UPDATE
                None, None,
                definition.Principal.LogonType,
                None
            )
            return True
        except Exception as e:
            print(f"Failed to disable task: {e}")
            return False
    
    def delete_task(self, task_path: str) -> bool:
        """Delete a task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            folder.DeleteTask(task_name, 0)
            return True
        except Exception as e:
            print(f"Failed to delete task: {e}")
            return False
```

### Health Check and Recovery

```python
class TaskHealthChecker:
    """Health checking and recovery for agent tasks"""
    
    def __init__(self, scheduler: TaskSchedulerService, monitor: TaskMonitor):
        self.scheduler = scheduler
        self.monitor = monitor
        self.critical_tasks = [
            "\\OpenClaw\\Core\\Heartbeat",
            "\\OpenClaw\\Core\\SoulMaintenance",
            "\\OpenClaw\\Core\\SystemMonitor"
        ]
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all agent tasks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'tasks': {},
            'issues': [],
            'recommendations': []
        }
        
        for task_path in self.critical_tasks:
            task_info = self.monitor.get_task_info(task_path)
            
            if task_info is None:
                results['tasks'][task_path] = {'status': 'missing'}
                results['issues'].append(f"Critical task missing: {task_path}")
                results['overall_status'] = 'critical'
            else:
                task_status = self._analyze_task_health(task_info)
                results['tasks'][task_path] = task_status
                
                if task_status['health'] != 'healthy':
                    results['issues'].extend(task_status['issues'])
                    if task_status['health'] == 'critical':
                        results['overall_status'] = 'critical'
                    elif results['overall_status'] != 'critical':
                        results['overall_status'] = 'degraded'
        
        return results
    
    def _analyze_task_health(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze health of a single task"""
        analysis = {
            'health': 'healthy',
            'issues': [],
            'details': task_info
        }
        
        # Check if task is enabled
        if not task_info['Enabled']:
            analysis['health'] = 'critical'
            analysis['issues'].append(f"Task {task_info['Name']} is disabled")
        
        # Check last run result
        if task_info['LastTaskResult'] != 0:
            analysis['health'] = 'degraded'
            analysis['issues'].append(
                f"Task {task_info['Name']} last run failed with code {task_info['LastTaskResult']}"
            )
        
        # Check for missed runs
        if task_info['NumberOfMissedRuns'] > 0:
            analysis['health'] = 'degraded'
            analysis['issues'].append(
                f"Task {task_info['Name']} has {task_info['NumberOfMissedRuns']} missed runs"
            )
        
        # Check if task is stuck running
        if task_info['StateCode'] == TASK_STATE.RUNNING:
            # Check if it's been running too long (over 1 hour)
            last_run = task_info['LastRunTime']
            if last_run:
                duration = datetime.now() - last_run
                if duration.total_seconds() > 3600:
                    analysis['health'] = 'critical'
                    analysis['issues'].append(
                        f"Task {task_info['Name']} appears stuck (running for {duration})"
                    )
        
        return analysis
    
    def recover_task(self, task_path: str) -> bool:
        """Attempt to recover a failed task"""
        try:
            # First, try to stop if running
            self.monitor.stop_task(task_path)
            
            # Re-enable if disabled
            self.monitor.enable_task(task_path)
            
            # Start the task
            return self.monitor.start_task(task_path)
        except Exception as e:
            print(f"Failed to recover task {task_path}: {e}")
            return False
```

---

## Cron-to-Task Scheduler Translation

### Cron Expression Parser

```python
import re
from typing import Tuple, Optional

class CronTranslator:
    """
    Translate Unix cron expressions to Windows Task Scheduler triggers
    """
    
    # Cron field positions
    CRON_MINUTE = 0
    CRON_HOUR = 1
    CRON_DAY_OF_MONTH = 2
    CRON_MONTH = 3
    CRON_DAY_OF_WEEK = 4
    
    # Special cron strings
    SPECIAL_STRINGS = {
        '@yearly': '0 0 1 1 *',
        '@annually': '0 0 1 1 *',
        '@monthly': '0 0 1 * *',
        '@weekly': '0 0 * * 0',
        '@daily': '0 0 * * *',
        '@hourly': '0 * * * *',
        '@reboot': None  # Special case - use boot trigger
    }
    
    @classmethod
    def parse_cron(cls, cron_expr: str) -> Dict[str, Any]:
        """
        Parse cron expression and return trigger configuration
        
        Args:
            cron_expr: Cron expression (e.g., "0 */6 * * *" for every 6 hours)
            
        Returns:
            Dictionary with trigger configuration
        """
        # Handle special strings
        if cron_expr in cls.SPECIAL_STRINGS:
            if cls.SPECIAL_STRINGS[cron_expr] is None:
                return {'type': 'boot'}
            cron_expr = cls.SPECIAL_STRINGS[cron_expr]
        
        # Parse standard cron expression
        fields = cron_expr.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")
        
        minute = cls._parse_field(fields[0], 0, 59)
        hour = cls._parse_field(fields[1], 0, 23)
        day_of_month = cls._parse_field(fields[2], 1, 31)
        month = cls._parse_field(fields[3], 1, 12)
        day_of_week = cls._parse_field(fields[4], 0, 7)
        
        # Determine trigger type based on pattern
        return cls._determine_trigger_type(minute, hour, day_of_month, month, day_of_week)
    
    @classmethod
    def _parse_field(cls, field: str, min_val: int, max_val: int) -> list:
        """Parse a single cron field"""
        values = set()
        
        # Handle */n pattern
        if field.startswith('*/'):
            step = int(field[2:])
            return list(range(min_val, max_val + 1, step))
        
        # Handle * (all values)
        if field == '*':
            return list(range(min_val, max_val + 1))
        
        # Handle comma-separated values
        for part in field.split(','):
            # Handle range (e.g., 1-5)
            if '-' in part:
                start, end = part.split('-')
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(int(part))
        
        return sorted(values)
    
    @classmethod
    def _determine_trigger_type(cls, minute, hour, day_of_month, month, day_of_week) -> Dict[str, Any]:
        """Determine the appropriate Windows trigger type"""
        
        # Check for daily pattern
        if len(day_of_month) == 31 and len(month) == 12:
            if len(day_of_week) == 7:
                # Daily trigger
                return {
                    'type': 'daily',
                    'days_interval': 1,
                    'start_boundary': cls._create_start_boundary(hour[0], minute[0]),
                    'repetition': cls._calculate_repetition(minute, hour)
                }
            else:
                # Weekly trigger (specific days)
                return {
                    'type': 'weekly',
                    'days_of_week': cls._days_to_bitmask(day_of_week),
                    'weeks_interval': 1,
                    'start_boundary': cls._create_start_boundary(hour[0], minute[0])
                }
        
        # Check for weekly pattern (specific day of week)
        if len(day_of_week) < 7 and len(day_of_month) == 31:
            return {
                'type': 'weekly',
                'days_of_week': cls._days_to_bitmask(day_of_week),
                'weeks_interval': 1,
                'start_boundary': cls._create_start_boundary(hour[0], minute[0])
            }
        
        # Check for monthly pattern
        if len(day_of_month) < 31 and len(month) == 12:
            return {
                'type': 'monthly',
                'days_of_month': cls._days_to_bitmask(day_of_month),
                'months_of_year': 0xFFF,
                'start_boundary': cls._create_start_boundary(hour[0], minute[0])
            }
        
        # Default to daily with repetition
        return {
            'type': 'daily',
            'days_interval': 1,
            'start_boundary': cls._create_start_boundary(hour[0], minute[0]),
            'repetition': cls._calculate_repetition(minute, hour)
        }
    
    @classmethod
    def _create_start_boundary(cls, hour: int, minute: int) -> str:
        """Create ISO 8601 start boundary string"""
        from datetime import datetime, timedelta
        now = datetime.now()
        start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if start < now:
            start += timedelta(days=1)
        return start.isoformat()
    
    @classmethod
    def _days_to_bitmask(cls, days: list) -> int:
        """Convert list of days to bitmask"""
        bitmask = 0
        for day in days:
            # Cron uses 0 or 7 for Sunday
            if day == 7:
                day = 0
            bitmask |= (1 << day)
        return bitmask
    
    @classmethod
    def _calculate_repetition(cls, minute: list, hour: list) -> Optional[Dict[str, str]]:
        """Calculate repetition pattern from cron fields"""
        # If minutes has multiple values, calculate interval
        if len(minute) > 1:
            intervals = [minute[i+1] - minute[i] for i in range(len(minute)-1)]
            if len(set(intervals)) == 1:
                # Regular interval
                interval_minutes = intervals[0]
                return {
                    'interval': f"PT{interval_minutes}M",
                    'duration': f"PT1H"  # Repeat for 1 hour
                }
        return None


# Common cron patterns to Windows Task Scheduler mapping
CRON_PATTERNS = {
    # Every minute
    '* * * * *': {
        'description': 'Every minute',
        'trigger': 'daily',
        'repetition_interval': 'PT1M',
        'repetition_duration': 'P1D'
    },
    
    # Every 5 minutes
    '*/5 * * * *': {
        'description': 'Every 5 minutes',
        'trigger': 'daily',
        'repetition_interval': 'PT5M',
        'repetition_duration': 'P1D'
    },
    
    # Every 15 minutes
    '*/15 * * * *': {
        'description': 'Every 15 minutes',
        'trigger': 'daily',
        'repetition_interval': 'PT15M',
        'repetition_duration': 'P1D'
    },
    
    # Every 30 minutes
    '*/30 * * * *': {
        'description': 'Every 30 minutes',
        'trigger': 'daily',
        'repetition_interval': 'PT30M',
        'repetition_duration': 'P1D'
    },
    
    # Every hour
    '0 * * * *': {
        'description': 'Every hour',
        'trigger': 'daily',
        'repetition_interval': 'PT1H',
        'repetition_duration': 'P1D'
    },
    
    # Every 6 hours
    '0 */6 * * *': {
        'description': 'Every 6 hours',
        'trigger': 'daily',
        'repetition_interval': 'PT6H',
        'repetition_duration': 'P1D'
    },
    
    # Every 12 hours
    '0 */12 * * *': {
        'description': 'Every 12 hours',
        'trigger': 'daily',
        'repetition_interval': 'PT12H',
        'repetition_duration': 'P1D'
    },
    
    # Daily at midnight
    '0 0 * * *': {
        'description': 'Daily at midnight',
        'trigger': 'daily',
        'days_interval': 1,
        'start_time': '00:00:00'
    },
    
    # Daily at 6 AM
    '0 6 * * *': {
        'description': 'Daily at 6 AM',
        'trigger': 'daily',
        'days_interval': 1,
        'start_time': '06:00:00'
    },
    
    # Weekly (Sundays at midnight)
    '0 0 * * 0': {
        'description': 'Weekly on Sunday at midnight',
        'trigger': 'weekly',
        'days_of_week': 0x01,  # Sunday
        'weeks_interval': 1,
        'start_time': '00:00:00'
    },
    
    # Monthly (1st of month at midnight)
    '0 0 1 * *': {
        'description': 'Monthly on 1st at midnight',
        'trigger': 'monthly',
        'days_of_month': 0x01,
        'months_of_year': 0xFFF,
        'start_time': '00:00:00'
    },
    
    # Weekdays at 9 AM
    '0 9 * * 1-5': {
        'description': 'Weekdays at 9 AM',
        'trigger': 'weekly',
        'days_of_week': 0x3E,  # Monday-Friday
        'weeks_interval': 1,
        'start_time': '09:00:00'
    },
    
    # Weekends at 10 AM
    '0 10 * * 0,6': {
        'description': 'Weekends at 10 AM',
        'trigger': 'weekly',
        'days_of_week': 0x41,  # Saturday, Sunday
        'weeks_interval': 1,
        'start_time': '10:00:00'
    }
}
```

---

## Agent Heartbeat Implementation

### Heartbeat Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agent Heartbeat System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Heartbeat Controller                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │   Primary    │  │  Secondary   │  │   Tertiary   │               │  │
│  │  │  Heartbeat   │  │  Heartbeat   │  │  Heartbeat   │               │  │
│  │  │  (30 sec)    │  │  (5 min)     │  │  (15 min)    │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Heartbeat Monitor Task                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │  Check In   │  │  Soul Sync  │  │ State Check │  │  Recovery   │ │  │
│  │  │  (Active)   │  │  (Identity) │  │  (Health)   │  │  (Repair)   │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Windows Task Scheduler                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │  Heartbeat   │  │  Heartbeat   │  │  Heartbeat   │               │  │
│  │  │   Task 1     │  │   Task 2     │  │   Task 3     │               │  │
│  │  │ (30s repeat) │  │ (5m repeat)  │  │ (15m repeat) │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Heartbeat Implementation

```python
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

class AgentHeartbeat:
    """
    Windows Task Scheduler-based Heartbeat System for OpenClaw AI Agent
    
    Provides:
    - Multiple heartbeat frequencies (30s, 5m, 15m)
    - Health monitoring and recovery
    - Soul/identity synchronization
    - State persistence
    """
    
    HEARTBEAT_FOLDER = "\\OpenClaw\\Core"
    HEARTBEAT_TASK_PREFIX = "Heartbeat_"
    
    # Heartbeat frequencies
    FREQUENCIES = {
        'primary': {
            'interval': 'PT30S',    # 30 seconds
            'description': 'Primary heartbeat - active monitoring',
            'priority': 'high'
        },
        'secondary': {
            'interval': 'PT5M',     # 5 minutes
            'description': 'Secondary heartbeat - health check',
            'priority': 'normal'
        },
        'tertiary': {
            'interval': 'PT15M',    # 15 minutes
            'description': 'Tertiary heartbeat - full system check',
            'priority': 'low'
        }
    }
    
    def __init__(self, scheduler: TaskSchedulerService, 
                 agent_id: str,
                 state_file: str = "C:\\OpenClaw\\heartbeat_state.json"):
        self.scheduler = scheduler
        self.agent_id = agent_id
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load heartbeat state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'agent_id': self.agent_id,
            'created_at': datetime.now().isoformat(),
            'last_heartbeat': None,
            'heartbeat_count': 0,
            'soul_signature': None,
            'health_status': 'initializing',
            'active_loops': []
        }
    
    def _save_state(self):
        """Save heartbeat state to file"""
        self._heartbeat_state['last_saved'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self._heartbeat_state, f, indent=2)
    
    def _generate_soul_signature(self) -> str:
        """Generate unique soul signature for agent identity"""
        data = f"{self.agent_id}:{datetime.now().isoformat()}:{self._heartbeat_state['heartbeat_count']}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def create_heartbeat_tasks(self) -> Dict[str, bool]:
        """Create all heartbeat tasks in Task Scheduler"""
        results = {}
        
        for freq_name, freq_config in self.FREQUENCIES.items():
            task_name = f"{self.HEARTBEAT_TASK_PREFIX}{freq_name}"
            
            try:
                # Create task definition
                builder = TaskBuilder(self.scheduler)
                builder.with_name(task_name)
                builder.with_description(freq_config['description'])
                
                # Configure principal
                builder.with_principal(
                    user_id="NT AUTHORITY\\SYSTEM",
                    logon_type=TASK_LOGON_TYPE.SERVICE_ACCOUNT,
                    run_level=TASK_RUNLEVEL.HIGHEST
                )
                
                # Configure settings
                builder.with_settings(
                    Enabled=True,
                    Hidden=False,
                    StartWhenAvailable=True,
                    AllowHardTerminate=True,
                    ExecutionTimeLimit="PT5M",
                    RestartCount=3,
                    RestartInterval="PT1M",
                    WakeToRun=True
                )
                
                # Add daily trigger with repetition
                start_time = datetime.now() + timedelta(minutes=1)
                builder.add_trigger(
                    TASK_TRIGGER_TYPE.DAILY,
                    StartBoundary=start_time.isoformat(),
                    DaysInterval=1,
                    RepetitionInterval=freq_config['interval'],
                    RepetitionDuration="P1D",  # Repeat for 1 day
                    Enabled=True
                )
                
                # Add boot trigger for system startup
                builder.add_trigger(
                    TASK_TRIGGER_TYPE.BOOT,
                    Delay="PT30S",
                    Enabled=True
                )
                
                # Create PowerShell action for heartbeat
                heartbeat_script = self._generate_heartbeat_script(freq_name)
                script_path = f"C:\\OpenClaw\\heartbeat_{freq_name}.ps1"
                
                with open(script_path, 'w') as f:
                    f.write(heartbeat_script)
                
                action = PowerShellActionBuilder.create_powershell_script_action(
                    script_path=script_path,
                    execution_policy="Bypass",
                    window_style="Hidden"
                )
                
                builder.add_action(
                    TASK_ACTION_TYPE.EXEC,
                    Path=action['Path'],
                    Arguments=action['Arguments']
                )
                
                # Register task
                builder.register(folder_path=self.HEARTBEAT_FOLDER)
                results[freq_name] = True
                
            except Exception as e:
                print(f"Failed to create {freq_name} heartbeat: {e}")
                results[freq_name] = False
        
        return results
    
    def _generate_heartbeat_script(self, frequency: str) -> str:
        """Generate PowerShell heartbeat script"""
        return f'''
# OpenClaw AI Agent Heartbeat Script
# Frequency: {frequency}
# Generated: {datetime.now().isoformat()}

param(
    [string]$AgentId = "{self.agent_id}",
    [string]$Frequency = "{frequency}",
    [string]$StateFile = "{self.state_file}"
)

# Load state
$state = @{{}}
if (Test-Path $StateFile) {{
    $state = Get-Content $StateFile | ConvertFrom-Json
}}

# Update heartbeat
$state.last_heartbeat = (Get-Date).ToString("o")
$state.heartbeat_count = ($state.heartbeat_count + 1)
$state.frequency = $Frequency

# Generate soul signature
$soulData = "$AgentId:$(Get-Date -Format o):$($state.heartbeat_count)"
$state.soul_signature = ([System.BitConverter]::ToString(
    (New-Object System.Security.Cryptography.SHA256Managed).ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($soulData)
    )
)).Replace("-", "").Substring(0, 16)

# Health check
$state.health_status = "healthy"
$state.system_uptime = (Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime

# Save state
$state | ConvertTo-Json -Depth 10 | Set-Content $StateFile

# Log heartbeat
$logEntry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] HEARTBEAT: $Frequency - Agent: $AgentId - Soul: $($state.soul_signature)"
Add-Content -Path "C:\\OpenClaw\\heartbeat.log" -Value $logEntry

# Write to Event Log
try {{
    if (-not [System.Diagnostics.EventLog]::SourceExists("OpenClawAgent")) {{
        New-EventLog -LogName "Application" -Source "OpenClawAgent"
    }}
    Write-EventLog -LogName "Application" -Source "OpenClawAgent" -EventId 1001 -Message $logEntry
}} catch {{}}

# Return success
exit 0
'''
    
    def check_heartbeat(self) -> Dict[str, Any]:
        """Check current heartbeat status"""
        self._heartbeat_state = self._load_state()
        
        status = {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'state': self._heartbeat_state,
            'healthy': False,
            'issues': []
        }
        
        # Check last heartbeat time
        if self._heartbeat_state['last_heartbeat']:
            last_beat = datetime.fromisoformat(self._heartbeat_state['last_heartbeat'])
            time_since = datetime.now() - last_beat
            
            if time_since.total_seconds() > 60:
                status['issues'].append(f"Last heartbeat was {time_since.seconds} seconds ago")
                status['healthy'] = False
            else:
                status['healthy'] = True
        else:
            status['issues'].append("No heartbeat recorded yet")
            status['healthy'] = False
        
        # Check soul signature
        if not self._heartbeat_state.get('soul_signature'):
            status['issues'].append("Soul signature not initialized")
        
        return status
    
    def manual_heartbeat(self) -> bool:
        """Trigger manual heartbeat"""
        try:
            self._heartbeat_state['last_heartbeat'] = datetime.now().isoformat()
            self._heartbeat_state['heartbeat_count'] += 1
            self._heartbeat_state['soul_signature'] = self._generate_soul_signature()
            self._heartbeat_state['health_status'] = 'manual'
            self._save_state()
            return True
        except Exception as e:
            print(f"Manual heartbeat failed: {e}")
            return False
    
    def remove_heartbeat_tasks(self) -> Dict[str, bool]:
        """Remove all heartbeat tasks"""
        results = {}
        monitor = TaskMonitor(self.scheduler)
        
        for freq_name in self.FREQUENCIES.keys():
            task_name = f"{self.HEARTBEAT_FOLDER}\\{self.HEARTBEAT_TASK_PREFIX}{freq_name}"
            results[freq_name] = monitor.delete_task(task_name)
        
        return results


class AgentSoulManager:
    """
    Manages agent soul/identity persistence through Task Scheduler
    """
    
    def __init__(self, scheduler: TaskSchedulerService, agent_id: str):
        self.scheduler = scheduler
        self.agent_id = agent_id
        self.heartbeat = AgentHeartbeat(scheduler, agent_id)
    
    def initialize_soul(self, identity_config: Dict[str, Any]) -> bool:
        """Initialize agent soul with identity configuration"""
        soul_data = {
            'agent_id': self.agent_id,
            'identity': identity_config,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'capabilities': [
                'gmail_integration',
                'browser_control',
                'tts_stt',
                'twilio_voice_sms',
                'system_access',
                'task_scheduling'
            ],
            'loops': [
                'email_processor',
                'browser_automation',
                'voice_handler',
                'communication',
                'file_operations',
                'network_monitor',
                'process_manager',
                'registry_watcher',
                'event_processor',
                'scheduler_coordinator',
                'user_activity',
                'security_monitor',
                'backup_manager',
                'update_checker',
                'maintenance'
            ]
        }
        
        # Save soul configuration
        soul_file = Path(f"C:\\OpenClaw\\soul_{self.agent_id}.json")
        soul_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(soul_file, 'w') as f:
            json.dump(soul_data, f, indent=2)
        
        # Create soul maintenance task
        return self._create_soul_maintenance_task()
    
    def _create_soul_maintenance_task(self) -> bool:
        """Create scheduled task for soul maintenance"""
        try:
            builder = TaskBuilder(self.scheduler)
            builder.with_name("SoulMaintenance")
            builder.with_description("Maintains agent soul and identity")
            
            # Run every hour
            start_time = datetime.now() + timedelta(hours=1)
            builder.add_trigger(
                TASK_TRIGGER_TYPE.DAILY,
                StartBoundary=start_time.isoformat(),
                DaysInterval=1,
                RepetitionInterval="PT1H",
                RepetitionDuration="P1D"
            )
            
            # PowerShell action for soul maintenance
            script = f'''
# Soul Maintenance Script
$soulFile = "C:\\OpenClaw\\soul_{self.agent_id}.json"
if (Test-Path $soulFile) {{
    $soul = Get-Content $soulFile | ConvertFrom-Json
    $soul.last_maintenance = (Get-Date).ToString("o")
    $soul | ConvertTo-Json -Depth 10 | Set-Content $soulFile
}}
exit 0
'''
            script_path = "C:\\OpenClaw\\soul_maintenance.ps1"
            with open(script_path, 'w') as f:
                f.write(script)
            
            action = PowerShellActionBuilder.create_powershell_script_action(script_path)
            builder.add_action(TASK_ACTION_TYPE.EXEC, Path=action['Path'], Arguments=action['Arguments'])
            
            builder.register(folder_path="\\OpenClaw\\Core")
            return True
        except Exception as e:
            print(f"Failed to create soul maintenance task: {e}")
            return False
```

---

## Code Examples

### Complete Task Creation Example

```python
# Example: Create a complete AI Agent task with multiple triggers and actions

def create_ai_agent_task(scheduler: TaskSchedulerService):
    """Create a comprehensive AI agent scheduled task"""
    
    builder = TaskBuilder(scheduler)
    
    # Task identification
    builder.with_name("OpenClaw_Agent_Core")
    builder.with_description("OpenClaw AI Agent - Core Processing Loop")
    builder.with_author("OpenClaw Framework v1.0")
    
    # Security context - run as SYSTEM with highest privileges
    builder.with_principal(
        user_id="NT AUTHORITY\\SYSTEM",
        logon_type=TASK_LOGON_TYPE.SERVICE_ACCOUNT,
        run_level=TASK_RUNLEVEL.HIGHEST
    )
    
    # Task settings - optimized for 24/7 operation
    builder.with_settings(
        Enabled=True,
        Hidden=False,
        AllowDemandStart=True,
        StartWhenAvailable=True,
        RunOnlyIfNetworkAvailable=False,
        DisallowStartIfOnBatteries=False,
        StopIfGoingOnBatteries=False,
        AllowHardTerminate=True,
        WakeToRun=True,
        ExecutionTimeLimit="PT0S",  # No time limit
        MultipleInstances=1,  # TASK_INSTANCES_IGNORE_NEW
        Priority=6,  # Above normal
        RestartCount=10,
        RestartInterval="PT1M"
    )
    
    # Trigger 1: System boot
    builder.add_trigger(
        TASK_TRIGGER_TYPE.BOOT,
        Delay="PT1M",  # Wait 1 minute after boot
        Enabled=True
    )
    
    # Trigger 2: Daily at midnight with 5-minute repetition
    tomorrow_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if tomorrow_midnight < datetime.now():
        tomorrow_midnight += timedelta(days=1)
    
    builder.add_trigger(
        TASK_TRIGGER_TYPE.DAILY,
        StartBoundary=tomorrow_midnight.isoformat(),
        DaysInterval=1,
        RepetitionInterval="PT5M",
        RepetitionDuration="P1D",
        Enabled=True
    )
    
    # Trigger 3: On user logon
    builder.add_trigger(
        TASK_TRIGGER_TYPE.LOGON,
        Delay="PT30S",
        Enabled=True
    )
    
    # Action 1: Start agent core
    builder.add_action(
        TASK_ACTION_TYPE.EXEC,
        Path="C:\\OpenClaw\\agent.exe",
        Arguments="--mode=core --config=C:\\OpenClaw\\config.json",
        WorkingDirectory="C:\\OpenClaw"
    )
    
    # Register in folder
    task = builder.register(folder_path="\\OpenClaw\\Core")
    
    return task
```

### PowerShell Complete Integration

```powershell
# OpenClaw_AI_Agent_TaskScheduler.ps1
# Complete PowerShell module for Task Scheduler integration

#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Windows Task Scheduler integration for OpenClaw AI Agent
.DESCRIPTION
    Provides comprehensive Task Scheduler management for the OpenClaw AI Agent system
    including task creation, monitoring, heartbeat management, and cron translation.
#>

# Enforce strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

#region Constants
$SCRIPT:RootFolder = "\OpenClaw"
$SCRIPT:CoreFolder = "\OpenClaw\Core"
$SCRIPT:AgentLoopsFolder = "\OpenClaw\AgentLoops"
$SCRIPT:ConfigPath = "C:\OpenClaw\config.json"
#endregion

#region Task Scheduler Connection
function Connect-TaskScheduler {
    <#
    .SYNOPSIS
        Connect to Windows Task Scheduler service
    #>
    try {
        $scheduler = New-Object -ComObject Schedule.Service
        $scheduler.Connect()
        return $scheduler
    }
    catch {
        Write-Error "Failed to connect to Task Scheduler: $_"
        return $null
    }
}
#endregion

#region Folder Management
function Initialize-OpenClawFolderStructure {
    <#
    .SYNOPSIS
        Create OpenClaw folder structure in Task Scheduler
    #>
    param([Parameter(Mandatory=$true)]$Scheduler)
    
    $folders = @(
        $SCRIPT:RootFolder,
        $SCRIPT:CoreFolder,
        $SCRIPT:AgentLoopsFolder,
        "\OpenClaw\Triggers",
        "\OpenClaw\Maintenance"
    )
    
    $root = $Scheduler.GetFolder("\")
    
    foreach ($folderPath in $folders) {
        try {
            $folderName = $folderPath.TrimStart("\")
            if ($folderName -ne "OpenClaw") {
                $parentPath = Split-Path $folderName -Parent
                $childName = Split-Path $folderName -Leaf
                $parent = $Scheduler.GetFolder("\$parentPath")
                $parent.CreateFolder($childName) | Out-Null
            }
            else {
                $root.CreateFolder("OpenClaw") | Out-Null
            }
            Write-Host "Created folder: $folderPath" -ForegroundColor Green
        }
        catch {
            Write-Host "Folder already exists or error: $folderPath" -ForegroundColor Yellow
        }
    }
}
#endregion

#region Task Creation
function New-OpenClawHeartbeatTask {
    <#
    .SYNOPSIS
        Create heartbeat task for AI Agent
    .PARAMETER Frequency
        Heartbeat frequency: Primary (30s), Secondary (5m), Tertiary (15m)
    #>
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [ValidateSet("Primary", "Secondary", "Tertiary")]
        [string]$Frequency = "Primary"
    )
    
    $intervals = @{
        "Primary" = "PT30S"
        "Secondary" = "PT5M"
        "Tertiary" = "PT15M"
    }
    
    $taskName = "Heartbeat_$Frequency"
    $interval = $intervals[$Frequency]
    
    # Create task definition
    $taskDef = $Scheduler.NewTask(0)
    
    # Registration info
    $regInfo = $taskDef.RegistrationInfo
    $regInfo.Description = "OpenClaw AI Agent Heartbeat - $Frequency"
    $regInfo.Author = "OpenClaw Framework"
    
    # Principal
    $principal = $taskDef.Principal
    $principal.UserId = "NT AUTHORITY\SYSTEM"
    $principal.LogonType = 5  # TASK_LOGON_SERVICE_ACCOUNT
    $principal.RunLevel = 1   # TASK_RUNLEVEL_HIGHEST
    
    # Settings
    $settings = $taskDef.Settings
    $settings.Enabled = $true
    $settings.Hidden = $false
    $settings.StartWhenAvailable = $true
    $settings.WakeToRun = $true
    $settings.ExecutionTimeLimit = "PT5M"
    $settings.RestartCount = 3
    $settings.RestartInterval = "PT1M"
    
    # Trigger - Daily with repetition
    $triggers = $taskDef.Triggers
    $trigger = $triggers.Create(2)  # TASK_TRIGGER_DAILY
    $trigger.StartBoundary = (Get-Date).AddMinutes(1).ToString("s")
    $trigger.DaysInterval = 1
    $trigger.Repetition.Interval = $interval
    $trigger.Repetition.Duration = "P1D"
    $trigger.Enabled = $true
    
    # Boot trigger
    $bootTrigger = $triggers.Create(8)  # TASK_TRIGGER_BOOT
    $bootTrigger.Delay = "PT30S"
    $bootTrigger.Enabled = $true
    
    # Action
    $actions = $taskDef.Actions
    $action = $actions.Create(0)  # TASK_ACTION_EXEC
    $action.Path = "powershell.exe"
    $action.Arguments = "-ExecutionPolicy Bypass -WindowStyle Hidden -Command `"& C:\OpenClaw\heartbeat_$Frequency.ps1`""
    $action.WorkingDirectory = "C:\OpenClaw"
    
    # Register task
    $folder = $Scheduler.GetFolder($SCRIPT:CoreFolder)
    $folder.RegisterTaskDefinition($taskName, $taskDef, 6, $null, $null, 5) | Out-Null
    
    Write-Host "Created heartbeat task: $taskName (Interval: $interval)" -ForegroundColor Green
}
#endregion

#region Cron Translation
function ConvertFrom-CronExpression {
    <#
    .SYNOPSIS
        Convert Unix cron expression to Task Scheduler trigger
    .PARAMETER CronExpression
        Cron expression (e.g., "0 */6 * * *")
    #>
    param([string]$CronExpression)
    
    # Handle special strings
    $specialStrings = @{
        "@yearly" = "0 0 1 1 *"
        "@monthly" = "0 0 1 * *"
        "@weekly" = "0 0 * * 0"
        "@daily" = "0 0 * * *"
        "@hourly" = "0 * * * *"
        "@reboot" = "BOOT"
    }
    
    if ($specialStrings.ContainsKey($CronExpression)) {
        return $specialStrings[$CronExpression]
    }
    
    # Parse standard cron
    $parts = $CronExpression -split "\s+"
    if ($parts.Count -ne 5) {
        throw "Invalid cron expression: $CronExpression"
    }
    
    $minute = $parts[0]
    $hour = $parts[1]
    $dayOfMonth = $parts[2]
    $month = $parts[3]
    $dayOfWeek = $parts[4]
    
    # Build trigger configuration
    $config = @{
        Minute = $minute
        Hour = $hour
        DayOfMonth = $dayOfMonth
        Month = $month
        DayOfWeek = $dayOfWeek
    }
    
    return $config
}
#endregion

#region Task Monitoring
function Get-OpenClawTaskStatus {
    <#
    .SYNOPSIS
        Get status of all OpenClaw tasks
    #>
    param([Parameter(Mandatory=$true)]$Scheduler)
    
    $results = @()
    
    try {
        $folder = $Scheduler.GetFolder($SCRIPT:RootFolder)
        $tasks = $folder.GetTasks(1)  # Include hidden tasks
        
        foreach ($task in $tasks) {
            $results += [PSCustomObject]@{
                Name = $task.Name
                Path = $task.Path
                State = switch ($task.State) {
                    0 { "Unknown" }
                    1 { "Disabled" }
                    2 { "Queued" }
                    3 { "Ready" }
                    4 { "Running" }
                    default { "Unknown" }
                }
                LastRunTime = $task.LastRunTime
                LastTaskResult = $task.LastTaskResult
                NextRunTime = $task.NextRunTime
                Enabled = $task.Enabled
            }
        }
    }
    catch {
        Write-Error "Failed to get tasks: $_"
    }
    
    return $results | Format-Table -AutoSize
}
#endregion

#region Main Execution
function Install-OpenClawTaskScheduler {
    <#
    .SYNOPSIS
        Complete installation of OpenClaw Task Scheduler integration
    #>
    Write-Host "=== OpenClaw Task Scheduler Installation ===" -ForegroundColor Cyan
    
    # Connect to Task Scheduler
    $scheduler = Connect-TaskScheduler
    if (-not $scheduler) {
        return
    }
    
    # Create folder structure
    Write-Host "`nCreating folder structure..." -ForegroundColor Yellow
    Initialize-OpenClawFolderStructure -Scheduler $scheduler
    
    # Create heartbeat tasks
    Write-Host "`nCreating heartbeat tasks..." -ForegroundColor Yellow
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Primary"
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Secondary"
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Tertiary"
    
    # Display status
    Write-Host "`nTask Status:" -ForegroundColor Yellow
    Get-OpenClawTaskStatus -Scheduler $scheduler
    
    Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
}

# Export functions
Export-ModuleMember -Function *
#endregion
```

---

## Security Considerations

### Security Best Practices

| Aspect | Recommendation | Implementation |
|--------|---------------|----------------|
| **Principal** | Use least privilege | `NT AUTHORITY\SYSTEM` for system tasks, specific user for user tasks |
| **Run Level** | Use LUA when possible | `TASK_RUNLEVEL_LUA` for non-admin tasks |
| **Logon Type** | Use service accounts | `TASK_LOGON_SERVICE_ACCOUNT` for background tasks |
| **Task Visibility** | Hide sensitive tasks | Set `Hidden = True` for security tasks |
| **Encryption** | Protect credentials | Use Windows Credential Manager |
| **Auditing** | Log all task operations | Enable Task Scheduler operational logs |
| **Permissions** | Restrict task modification | Set appropriate SDDL on task folders |

### SDDL Example for Task Folder Security

```python
# Secure task folder with SDDL
SDDL_OPENCLAW = "D:(A;;FA;;;BA)(A;;FA;;;SY)(A;;FR;;;AU)"
# Breakdown:
# D: - Discretionary ACL
# (A;;FA;;;BA) - Allow Full Access to Built-in Administrators
# (A;;FA;;;SY) - Allow Full Access to SYSTEM
# (A;;FR;;;AU) - Allow Read Access to Authenticated Users
```

---

## Appendix: ISO 8601 Duration Reference

| Duration | Format | Description |
|----------|--------|-------------|
| 30 seconds | `PT30S` | Heartbeat interval |
| 1 minute | `PT1M` | Short delay |
| 5 minutes | `PT5M` | Medium interval |
| 15 minutes | `PT15M` | Long interval |
| 1 hour | `PT1H` | Execution time limit |
| 6 hours | `PT6H` | Extended interval |
| 1 day | `P1D` | Repetition duration |
| 1 week | `P1W` | Weekly period |

---

## Document Information

- **Version**: 1.0.0
- **Date**: 2025
- **Author**: OpenClaw AI Agent Framework
- **Platform**: Windows 10/11
- **API Version**: Task Scheduler 2.0

---

*This specification provides the complete technical foundation for Windows Task Scheduler integration in the OpenClaw-inspired AI Agent system.*
