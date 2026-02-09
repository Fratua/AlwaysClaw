#!/usr/bin/env python3
"""
OpenClaw AI Agent - Windows Task Scheduler Integration
Python Implementation

This module provides comprehensive Windows Task Scheduler integration for the
OpenClaw-inspired AI Agent system running on Windows 10/11.

Features:
- Task Scheduler COM API integration (ITaskService, ITaskDefinition, etc.)
- Task creation and configuration
- All trigger types (time-based, event-based, boot, logon, etc.)
- Action definitions (executable, COM handler, PowerShell)
- Task folder organization
- Task monitoring and management
- Cron-to-Task Scheduler translation
- Agent heartbeat implementation

Author: OpenClaw Framework
Version: 1.0.0
Platform: Windows 10/11
"""

import win32com.client
import pythoncom
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import IntEnum
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\OpenClaw\\logs\\task_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OpenClaw.TaskScheduler')


# =============================================================================
# ENUMERATIONS
# =============================================================================

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


class TASK_INSTANCES(IntEnum):
    """Multiple instances policy"""
    PARALLEL = 0
    QUEUE = 1
    IGNORE_NEW = 2
    STOP_EXISTING = 3


class TASK_COMPATIBILITY(IntEnum):
    """Task compatibility level"""
    AT = 0           # Task Scheduler 1.0
    V1 = 1           # Task Scheduler 1.0
    V2 = 2           # Task Scheduler 2.0
    V2_1 = 3         # Task Scheduler 2.0 (Windows 7/2008 R2)
    V2_2 = 4         # Task Scheduler 2.0 (Windows 8/2012)
    V2_3 = 5         # Task Scheduler 2.0 (Windows 8.1/2012 R2)
    V2_4 = 6         # Task Scheduler 2.0 (Windows 10/2016)


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


class SessionStateChange:
    """Session state change types"""
    CONSOLE_CONNECT = 1
    CONSOLE_DISCONNECT = 2
    REMOTE_CONNECT = 3
    REMOTE_DISCONNECT = 4
    SESSION_LOCK = 5
    SESSION_UNLOCK = 6


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TaskInfo:
    """Task information data class"""
    name: str
    path: str
    state: str
    state_code: int
    enabled: bool
    last_run_time: Optional[datetime]
    last_task_result: int
    next_run_time: Optional[datetime]
    number_of_missed_runs: int
    author: str = ""
    description: str = ""
    
    @property
    def is_running(self) -> bool:
        return self.state_code == TASK_STATE.RUNNING
    
    @property
    def is_healthy(self) -> bool:
        return self.enabled and self.last_task_result == 0


@dataclass
class TriggerConfig:
    """Trigger configuration data class"""
    trigger_type: TASK_TRIGGER_TYPE
    start_boundary: datetime
    end_boundary: Optional[datetime] = None
    enabled: bool = True
    repetition_interval: Optional[str] = None
    repetition_duration: Optional[str] = None
    delay: Optional[str] = None
    # Type-specific fields
    days_interval: Optional[int] = None
    days_of_week: Optional[int] = None
    weeks_interval: Optional[int] = None
    days_of_month: Optional[int] = None
    months_of_year: Optional[int] = None
    weeks_of_month: Optional[int] = None
    user_id: Optional[str] = None
    subscription: Optional[str] = None
    state_change: Optional[int] = None


@dataclass
class ActionConfig:
    """Action configuration data class"""
    action_type: TASK_ACTION_TYPE
    path: Optional[str] = None
    arguments: Optional[str] = None
    working_directory: Optional[str] = None
    clsid: Optional[str] = None
    data: Optional[str] = None


@dataclass
class TaskSettings:
    """Task settings data class"""
    enabled: bool = True
    hidden: bool = False
    allow_demand_start: bool = True
    start_when_available: bool = True
    run_only_if_network_available: bool = False
    disallow_start_if_on_batteries: bool = False
    stop_if_going_on_batteries: bool = False
    allow_hard_terminate: bool = True
    wake_to_run: bool = True
    execution_time_limit: str = "PT1H"
    multiple_instances: TASK_INSTANCES = TASK_INSTANCES.IGNORE_NEW
    priority: int = 7
    restart_count: int = 3
    restart_interval: str = "PT1M"
    compatibility: TASK_COMPATIBILITY = TASK_COMPATIBILITY.V2_4


# =============================================================================
# TASK SCHEDULER SERVICE
# =============================================================================

class TaskSchedulerService:
    """
    Windows Task Scheduler Service Wrapper
    Provides high-level interface to Task Scheduler COM API
    """
    
    def __init__(self):
        self._scheduler = None
        self._connected = False
    
    def connect(self, server: Optional[str] = None,
                user: Optional[str] = None,
                domain: Optional[str] = None,
                password: Optional[str] = None) -> bool:
        """
        Connect to Task Scheduler service
        
        Args:
            server: Remote server name (None for local)
            user: Username for connection
            domain: Domain for connection
            password: Password for connection
            
        Returns:
            True if connected successfully
        """
        try:
            pythoncom.CoInitialize()
            self._scheduler = win32com.client.Dispatch("Schedule.Service")
            self._scheduler.Connect(server, user, domain, password)
            self._connected = True
            logger.info("Connected to Task Scheduler service")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to connect to Task Scheduler: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Task Scheduler service"""
        self._scheduler = None
        self._connected = False
        pythoncom.CoUninitialize()
        logger.info("Disconnected from Task Scheduler service")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Task Scheduler"""
        return self._connected and self._scheduler is not None
    
    def get_folder(self, path: str = "\\") -> Any:
        """Get task folder by path"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Task Scheduler")
        return self._scheduler.GetFolder(path)
    
    def create_folder(self, path: str, sddl: Optional[str] = None) -> Any:
        """Create new task folder"""
        root = self.get_folder("\\")
        return root.CreateFolder(path, sddl)
    
    def delete_folder(self, path: str) -> bool:
        """Delete task folder"""
        try:
            parent_path = "\\".join(path.rstrip("\\").split("\\")[:-1]) or "\\"
            folder_name = path.rstrip("\\").split("\\")[-1]
            parent = self.get_folder(parent_path)
            parent.DeleteFolder(folder_name, 0)
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to delete folder {path}: {e}")
            return False
    
    def get_running_tasks(self, flags: int = 0) -> Any:
        """Get collection of running tasks"""
        return self._scheduler.GetRunningTasks(flags)
    
    def new_task_definition(self, flags: int = 0) -> Any:
        """Create new task definition object"""
        return self._scheduler.NewTask(flags)
    
    def get_highest_version(self) -> int:
        """Get highest Task Scheduler version supported"""
        return self._scheduler.HighestVersion


# =============================================================================
# TASK BUILDER
# =============================================================================

class TaskBuilder:
    """
    Builder pattern for creating Windows Scheduled Tasks
    """
    
    def __init__(self, scheduler: TaskSchedulerService):
        self.scheduler = scheduler
        self._definition = None
        self._task_name = ""
        self.reset()
    
    def reset(self):
        """Reset builder state"""
        self._definition = self.scheduler.new_task_definition(0)
        self._task_name = ""
        self._configure_defaults()
        return self
    
    def _configure_defaults(self):
        """Configure default task settings"""
        # Registration info
        reg_info = self._definition.RegistrationInfo
        reg_info.Author = "OpenClaw AI Agent"
        reg_info.Description = "Automated AI Agent Task"
        reg_info.Date = datetime.now().isoformat()
        
        # Settings
        settings = self._definition.Settings
        settings.Enabled = True
        settings.Hidden = False
        settings.StartWhenAvailable = True
        settings.AllowHardTerminate = True
        settings.ExecutionTimeLimit = "PT1H"
        settings.MultipleInstances = TASK_INSTANCES.IGNORE_NEW
        settings.Priority = 7
        settings.RunOnlyIfNetworkAvailable = False
        settings.DisallowStartIfOnBatteries = False
        settings.StopIfGoingOnBatteries = False
        settings.WakeToRun = True
        
        # Principal
        principal = self._definition.Principal
        principal.LogonType = TASK_LOGON_TYPE.SERVICE_ACCOUNT
        principal.RunLevel = TASK_RUNLEVEL.HIGHEST
        principal.UserId = "NT AUTHORITY\\SYSTEM"
    
    def with_name(self, name: str) -> 'TaskBuilder':
        """Set task name"""
        self._task_name = name
        return self
    
    def with_description(self, description: str) -> 'TaskBuilder':
        """Set task description"""
        self._definition.RegistrationInfo.Description = description
        return self
    
    def with_author(self, author: str) -> 'TaskBuilder':
        """Set task author"""
        self._definition.RegistrationInfo.Author = author
        return self
    
    def with_principal(self, user_id: str,
                       logon_type: TASK_LOGON_TYPE = TASK_LOGON_TYPE.SERVICE_ACCOUNT,
                       run_level: TASK_RUNLEVEL = TASK_RUNLEVEL.HIGHEST) -> 'TaskBuilder':
        """Configure task principal (security context)"""
        principal = self._definition.Principal
        principal.UserId = user_id
        principal.LogonType = logon_type
        principal.RunLevel = run_level
        return self
    
    def with_settings(self, **kwargs) -> 'TaskBuilder':
        """Configure task settings"""
        settings = self._definition.Settings
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return self
    
    def add_trigger(self, trigger_config: TriggerConfig) -> 'TaskBuilder':
        """Add trigger to task using TriggerConfig"""
        triggers = self._definition.Triggers
        trigger = triggers.Create(trigger_config.trigger_type)
        
        # Common properties
        trigger.StartBoundary = trigger_config.start_boundary.isoformat()
        if trigger_config.end_boundary:
            trigger.EndBoundary = trigger_config.end_boundary.isoformat()
        trigger.Enabled = trigger_config.enabled
        
        if trigger_config.repetition_interval:
            trigger.Repetition.Interval = trigger_config.repetition_interval
        if trigger_config.repetition_duration:
            trigger.Repetition.Duration = trigger_config.repetition_duration
        if trigger_config.delay:
            trigger.Delay = trigger_config.delay
        
        # Type-specific properties
        if trigger_config.trigger_type == TASK_TRIGGER_TYPE.DAILY:
            if trigger_config.days_interval:
                trigger.DaysInterval = trigger_config.days_interval
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.WEEKLY:
            if trigger_config.days_of_week:
                trigger.DaysOfWeek = trigger_config.days_of_week
            if trigger_config.weeks_interval:
                trigger.WeeksInterval = trigger_config.weeks_interval
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.MONTHLY:
            if trigger_config.days_of_month:
                trigger.DaysOfMonth = trigger_config.days_of_month
            if trigger_config.months_of_year:
                trigger.MonthsOfYear = trigger_config.months_of_year
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.MONTHLYDOW:
            if trigger_config.days_of_week:
                trigger.DaysOfWeek = trigger_config.days_of_week
            if trigger_config.weeks_of_month:
                trigger.WeeksOfMonth = trigger_config.weeks_of_month
            if trigger_config.months_of_year:
                trigger.MonthsOfYear = trigger_config.months_of_year
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.LOGON:
            if trigger_config.user_id:
                trigger.UserId = trigger_config.user_id
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.EVENT:
            if trigger_config.subscription:
                trigger.Subscription = trigger_config.subscription
        
        elif trigger_config.trigger_type == TASK_TRIGGER_TYPE.SESSION_STATE_CHANGE:
            if trigger_config.state_change:
                trigger.StateChange = trigger_config.state_change
            if trigger_config.user_id:
                trigger.UserId = trigger_config.user_id
        
        return self
    
    def add_action(self, action_config: ActionConfig) -> 'TaskBuilder':
        """Add action to task using ActionConfig"""
        actions = self._definition.Actions
        action = actions.Create(action_config.action_type)
        
        if action_config.action_type == TASK_ACTION_TYPE.EXEC:
            if action_config.path:
                action.Path = action_config.path
            if action_config.arguments:
                action.Arguments = action_config.arguments
            if action_config.working_directory:
                action.WorkingDirectory = action_config.working_directory
        
        elif action_config.action_type == TASK_ACTION_TYPE.COM_HANDLER:
            if action_config.clsid:
                action.ClassId = action_config.clsid
            if action_config.data:
                action.Data = action_config.data
        
        return self
    
    def register(self, folder_path: str = "\\OpenClaw",
                 create_flags: int = 6,  # TASK_CREATE_OR_UPDATE
                 password: Optional[str] = None) -> Any:
        """Register task in Task Scheduler"""
        if not self._task_name:
            raise ValueError("Task name not set")
        
        try:
            folder = self.scheduler.get_folder(folder_path)
        except (pywintypes.com_error, OSError) as e:
            logger.debug(f"Folder '{folder_path}' not found, creating: {e}")
            # Create folder if it doesn't exist
            parts = folder_path.strip("\\").split("\\")
            current_path = "\\"
            for part in parts:
                try:
                    parent = self.scheduler.get_folder(current_path)
                    parent.CreateFolder(part, None)
                except (pywintypes.com_error, OSError) as e:
                    logger.debug(f"Folder creation skipped (may already exist) at '{current_path}{part}': {e}")
                current_path = current_path + part + "\\"
            folder = self.scheduler.get_folder(folder_path)
        
        registered_task = folder.RegisterTaskDefinition(
            self._task_name,
            self._definition,
            create_flags,
            None,  # UserId (use from principal)
            password,
            self._definition.Principal.LogonType,
            None   # sddl
        )
        
        logger.info(f"Registered task: {self._task_name}")
        return registered_task


# =============================================================================
# TASK MONITOR
# =============================================================================

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
    
    def get_task_info(self, task_path: str) -> Optional[TaskInfo]:
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
            
            return TaskInfo(
                name=task.Name,
                path=task.Path,
                state=self.TASK_STATE_NAMES.get(task.State, 'Unknown'),
                state_code=task.State,
                enabled=task.Enabled,
                last_run_time=task.LastRunTime if task.LastRunTime.year > 1 else None,
                last_task_result=task.LastTaskResult,
                next_run_time=task.NextRunTime if task.NextRunTime.year > 1 else None,
                number_of_missed_runs=task.NumberOfMissedRuns,
                author=definition.RegistrationInfo.Author,
                description=definition.RegistrationInfo.Description
            )
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to get task info for {task_path}: {e}")
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
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to get running tasks: {e}")
        
        return running_tasks
    
    def is_task_running(self, task_path: str) -> bool:
        """Check if a specific task is currently running"""
        info = self.get_task_info(task_path)
        return info is not None and info.is_running
    
    def start_task(self, task_path: str, arguments: Optional[str] = None) -> bool:
        """Start a task manually"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            task.Run(arguments)
            logger.info(f"Started task: {task_path}")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to start task {task_path}: {e}")
            return False
    
    def stop_task(self, task_path: str) -> bool:
        """Stop a running task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            task = folder.GetTask(task_name)
            
            task.Stop(0)
            logger.info(f"Stopped task: {task_path}")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to stop task {task_path}: {e}")
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
            logger.info(f"Enabled task: {task_path}")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to enable task {task_path}: {e}")
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
            logger.info(f"Disabled task: {task_path}")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to disable task {task_path}: {e}")
            return False
    
    def delete_task(self, task_path: str) -> bool:
        """Delete a task"""
        try:
            parts = task_path.rsplit('\\', 1)
            folder_path = parts[0] if len(parts) == 2 else "\\"
            task_name = parts[-1]
            
            folder = self.scheduler.get_folder(folder_path)
            folder.DeleteTask(task_name, 0)
            logger.info(f"Deleted task: {task_path}")
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to delete task {task_path}: {e}")
            return False


# =============================================================================
# CRON TRANSLATOR
# =============================================================================

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
        """Parse cron expression and return trigger configuration"""
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
        
        return cls._determine_trigger_type(minute, hour, day_of_month, month, day_of_week)
    
    @classmethod
    def _parse_field(cls, field: str, min_val: int, max_val: int) -> List[int]:
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
                'months_of_year': MonthsOfYear.ALL,
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
    def _create_start_boundary(cls, hour: int, minute: int) -> datetime:
        """Create start boundary datetime"""
        now = datetime.now()
        start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if start < now:
            start += timedelta(days=1)
        return start
    
    @classmethod
    def _days_to_bitmask(cls, days: List[int]) -> int:
        """Convert list of days to bitmask"""
        bitmask = 0
        for day in days:
            # Cron uses 0 or 7 for Sunday
            if day == 7:
                day = 0
            bitmask |= (1 << day)
        return bitmask
    
    @classmethod
    def _calculate_repetition(cls, minute: List[int], hour: List[int]) -> Optional[Dict[str, str]]:
        """Calculate repetition pattern from cron fields"""
        if len(minute) > 1:
            intervals = [minute[i+1] - minute[i] for i in range(len(minute)-1)]
            if len(set(intervals)) == 1:
                interval_minutes = intervals[0]
                return {
                    'interval': f"PT{interval_minutes}M",
                    'duration': "PT1H"
                }
        return None
    
    @classmethod
    def to_trigger_config(cls, cron_expr: str) -> TriggerConfig:
        """Convert cron expression to TriggerConfig"""
        parsed = cls.parse_cron(cron_expr)
        
        if parsed['type'] == 'boot':
            return TriggerConfig(
                trigger_type=TASK_TRIGGER_TYPE.BOOT,
                start_boundary=datetime.now(),
                delay="PT1M"
            )
        
        trigger_type = {
            'daily': TASK_TRIGGER_TYPE.DAILY,
            'weekly': TASK_TRIGGER_TYPE.WEEKLY,
            'monthly': TASK_TRIGGER_TYPE.MONTHLY
        }.get(parsed['type'], TASK_TRIGGER_TYPE.DAILY)
        
        config = TriggerConfig(
            trigger_type=trigger_type,
            start_boundary=parsed['start_boundary']
        )
        
        if 'days_interval' in parsed:
            config.days_interval = parsed['days_interval']
        if 'days_of_week' in parsed:
            config.days_of_week = parsed['days_of_week']
        if 'weeks_interval' in parsed:
            config.weeks_interval = parsed['weeks_interval']
        if 'days_of_month' in parsed:
            config.days_of_month = parsed['days_of_month']
        if 'months_of_year' in parsed:
            config.months_of_year = parsed['months_of_year']
        if 'repetition' in parsed and parsed['repetition']:
            config.repetition_interval = parsed['repetition']['interval']
            config.repetition_duration = parsed['repetition']['duration']
        
        return config


# =============================================================================
# AGENT HEARTBEAT
# =============================================================================

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
            'interval': 'PT30S',
            'description': 'Primary heartbeat - active monitoring',
            'priority': 'high'
        },
        'secondary': {
            'interval': 'PT5M',
            'description': 'Secondary heartbeat - health check',
            'priority': 'normal'
        },
        'tertiary': {
            'interval': 'PT15M',
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
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load state: {e}")
        
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
                builder = TaskBuilder(self.scheduler)
                builder.with_name(task_name)
                builder.with_description(freq_config['description'])
                
                builder.with_principal(
                    user_id="NT AUTHORITY\\SYSTEM",
                    logon_type=TASK_LOGON_TYPE.SERVICE_ACCOUNT,
                    run_level=TASK_RUNLEVEL.HIGHEST
                )
                
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
                
                # Daily trigger with repetition
                start_time = datetime.now() + timedelta(minutes=1)
                builder.add_trigger(TriggerConfig(
                    trigger_type=TASK_TRIGGER_TYPE.DAILY,
                    start_boundary=start_time,
                    days_interval=1,
                    repetition_interval=freq_config['interval'],
                    repetition_duration="P1D"
                ))
                
                # Boot trigger
                builder.add_trigger(TriggerConfig(
                    trigger_type=TASK_TRIGGER_TYPE.BOOT,
                    start_boundary=datetime.now(),
                    delay="PT30S"
                ))
                
                # Create PowerShell action
                script_content = self._generate_heartbeat_script(freq_name)
                script_path = f"C:\\OpenClaw\\heartbeat_{freq_name}.ps1"
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                builder.add_action(ActionConfig(
                    action_type=TASK_ACTION_TYPE.EXEC,
                    path="powershell.exe",
                    arguments=f"-ExecutionPolicy Bypass -WindowStyle Hidden -File \"{script_path}\" -AgentId '{self.agent_id}'",
                    working_directory="C:\\OpenClaw"
                ))
                
                builder.register(folder_path=self.HEARTBEAT_FOLDER)
                results[freq_name] = True
                logger.info(f"Created heartbeat task: {task_name}")
                
            except (pywintypes.com_error, OSError) as e:
                logger.error(f"Failed to create {freq_name} heartbeat: {e}")
                results[freq_name] = False
        
        return results
    
    def _generate_heartbeat_script(self, frequency: str) -> str:
        """Generate PowerShell heartbeat script"""
        return f'''# OpenClaw AI Agent Heartbeat Script
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
Add-Content -Path "C:\\OpenClaw\\logs\\heartbeat.log" -Value $logEntry

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
            logger.info("Manual heartbeat triggered")
            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Manual heartbeat failed: {e}")
            return False
    
    def remove_heartbeat_tasks(self) -> Dict[str, bool]:
        """Remove all heartbeat tasks"""
        results = {}
        monitor = TaskMonitor(self.scheduler)
        
        for freq_name in self.FREQUENCIES.keys():
            task_name = f"{self.HEARTBEAT_FOLDER}\\{self.HEARTBEAT_TASK_PREFIX}{freq_name}"
            results[freq_name] = monitor.delete_task(task_name)
        
        return results


# =============================================================================
# FOLDER MANAGER
# =============================================================================

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
                root.CreateFolder("OpenClaw", None)
                logger.info("Created folder: \\OpenClaw")
            except (pywintypes.com_error, OSError) as e:
                logger.info(f"Folder already exists or error: \\OpenClaw ({e})")
            
            # Create subfolders
            for folder_name in self.FOLDER_STRUCTURE.keys():
                folder_path = f"{self.ROOT_FOLDER}\\{folder_name}"
                try:
                    openclaw = self.scheduler.get_folder(self.ROOT_FOLDER)
                    openclaw.CreateFolder(folder_name, None)
                    logger.info(f"Created folder: {folder_path}")
                except (pywintypes.com_error, OSError) as e:
                    logger.info(f"Folder already exists or error: {folder_path}")
            
            self._folders_created = True
            return True
        except (pywintypes.com_error, OSError) as e:
            logger.error(f"Failed to create folder structure: {e}")
            return False
    
    def get_folder(self, path: str) -> Any:
        """Get folder by relative path"""
        full_path = f"{self.ROOT_FOLDER}\\{path}" if not path.startswith("\\") else path
        return self.scheduler.get_folder(full_path)
    
    def list_all_tasks(self) -> Dict[str, List[str]]:
        """List all tasks in the OpenClaw folder structure"""
        tasks = {}
        
        for folder_name in self.FOLDER_STRUCTURE.keys():
            folder_path = f"{self.ROOT_FOLDER}\\{folder_name}"
            try:
                folder = self.scheduler.get_folder(folder_path)
                folder_tasks = folder.GetTasks(0)
                tasks[folder_name] = [task.Name for task in folder_tasks]
            except (pywintypes.com_error, OSError) as e:
                logger.debug(f"Could not list tasks in {folder_name}: {e}")
                tasks[folder_name] = []
        
        return tasks


# =============================================================================
# MAIN INSTALLER
# =============================================================================

class OpenClawInstaller:
    """Main installer for OpenClaw Task Scheduler integration"""
    
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or f"openclaw_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.scheduler = TaskSchedulerService()
        self.monitor: Optional[TaskMonitor] = None
        self.folder_manager: Optional[TaskFolderManager] = None
        self.heartbeat: Optional[AgentHeartbeat] = None
    
    def install(self) -> bool:
        """Perform complete installation"""
        logger.info("=== OpenClaw Task Scheduler Installation ===")
        logger.info(f"Agent ID: {self.agent_id}")
        
        # Connect to Task Scheduler
        if not self.scheduler.connect():
            logger.error("Installation failed - could not connect to Task Scheduler")
            return False
        
        self.monitor = TaskMonitor(self.scheduler)
        self.folder_manager = TaskFolderManager(self.scheduler)
        self.heartbeat = AgentHeartbeat(self.scheduler, self.agent_id)
        
        # Create folder structure
        logger.info("Creating folder structure...")
        self.folder_manager.create_folder_structure()
        
        # Create heartbeat tasks
        logger.info("Creating heartbeat tasks...")
        self.heartbeat.create_heartbeat_tasks()
        
        # Create 15 agent loop tasks
        logger.info("Creating agent loop tasks...")
        self._create_agent_loops()
        
        # Save configuration
        self._save_config()
        
        logger.info("=== Installation Complete ===")
        return True
    
    def _create_agent_loops(self):
        """Create the 15 hardcoded agentic loop tasks"""
        loops = [
            (1, "EmailProcessor", "PT2M"),
            (2, "BrowserAutomation", "PT5M"),
            (3, "VoiceHandler", "PT1M"),
            (4, "Communication", "PT3M"),
            (5, "FileOperations", "PT5M"),
            (6, "NetworkMonitor", "PT30S"),
            (7, "ProcessManager", "PT10S"),
            (8, "RegistryWatcher", "PT5M"),
            (9, "EventProcessor", "PT1M"),
            (10, "SchedulerCoordinator", "PT1M"),
            (11, "UserActivity", "PT5M"),
            (12, "SecurityMonitor", "PT5M"),
            (13, "BackupManager", "PT1H"),
            (14, "UpdateChecker", "PT6H"),
            (15, "Maintenance", "PT1H")
        ]
        
        for num, name, interval in loops:
            try:
                builder = TaskBuilder(self.scheduler)
                builder.with_name(f"Loop_{num:02d}_{name}")
                builder.with_description(f"OpenClaw Agent Loop {num} - {name}")
                
                start_time = datetime.now() + timedelta(minutes=num)
                builder.add_trigger(TriggerConfig(
                    trigger_type=TASK_TRIGGER_TYPE.DAILY,
                    start_boundary=start_time,
                    days_interval=1,
                    repetition_interval=interval,
                    repetition_duration="P1D"
                ))
                
                builder.add_action(ActionConfig(
                    action_type=TASK_ACTION_TYPE.EXEC,
                    path="C:\\OpenClaw\\agent_loop.exe",
                    arguments=f"--loop={num} --name={name}",
                    working_directory="C:\\OpenClaw"
                ))
                
                builder.register(folder_path="\\OpenClaw\\AgentLoops")
                logger.info(f"Created agent loop: {name}")
                
            except (pywintypes.com_error, OSError) as e:
                logger.error(f"Failed to create loop {name}: {e}")
    
    def _save_config(self):
        """Save agent configuration"""
        config = {
            'agent_id': self.agent_id,
            'version': '1.0.0',
            'install_date': datetime.now().isoformat(),
            'features': [
                'TaskScheduler',
                'Heartbeat',
                'AgentLoops',
                'SoulMaintenance'
            ]
        }
        
        config_path = Path("C:\\OpenClaw\\config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def uninstall(self) -> bool:
        """Remove all OpenClaw tasks and configuration"""
        logger.info("=== Uninstalling OpenClaw Task Scheduler ===")
        
        if not self.scheduler.connect():
            return False
        
        try:
            # Remove root folder (removes all tasks)
            root = self.scheduler.get_folder("\\")
            root.DeleteFolder("OpenClaw", 0)
            logger.info("Removed OpenClaw folder and all tasks")
        except (pywintypes.com_error, OSError) as e:
            logger.warning(f"Error removing folder: {e}")
        
        # Remove configuration
        config_path = Path("C:\\OpenClaw\\config.json")
        if config_path.exists():
            config_path.unlink()
            logger.info("Removed configuration file")
        
        logger.info("=== Uninstall Complete ===")
        return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenClaw Task Scheduler Integration")
    parser.add_argument("--install", action="store_true", help="Install OpenClaw Task Scheduler")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall OpenClaw Task Scheduler")
    parser.add_argument("--agent-id", type=str, help="Agent ID")
    parser.add_argument("--status", action="store_true", help="Show task status")
    
    args = parser.parse_args()
    
    installer = OpenClawInstaller(agent_id=args.agent_id)
    
    if args.install:
        installer.install()
    elif args.uninstall:
        installer.uninstall()
    elif args.status:
        scheduler = TaskSchedulerService()
        if scheduler.connect():
            monitor = TaskMonitor(scheduler)
            folder_mgr = TaskFolderManager(scheduler)
            tasks = folder_mgr.list_all_tasks()
            print(json.dumps(tasks, indent=2))
    else:
        parser.print_help()
