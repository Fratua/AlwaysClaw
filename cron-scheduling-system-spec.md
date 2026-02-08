# Cron Jobs and Scheduled Task System Technical Specification
## OpenClaw-Inspired AI Agent System for Windows 10

---

## Executive Summary

This document provides a comprehensive technical specification for implementing a robust cron job and scheduled task system for a 24/7 Windows 10 AI agent system. The architecture combines Windows Task Scheduler integration, Node.js cron libraries, job queue management, and persistence mechanisms to ensure reliable, fault-tolerant operation.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Cron Expression Parsing and Scheduling](#2-cron-expression-parsing-and-scheduling)
3. [Windows Task Scheduler Integration](#3-windows-task-scheduler-integration)
4. [Node.js Cron Libraries](#4-nodejs-cron-libraries)
5. [Job Queue Management](#5-job-queue-management)
6. [Task Persistence Across Restarts](#6-task-persistence-across-restarts)
7. [Overlapping Job Handling](#7-overlapping-job-handling)
8. [Job Failure and Retry Mechanisms](#8-job-failure-and-retry-mechanisms)
9. [Heartbeat Mechanism](#9-heartbeat-mechanism)
10. [Scheduling Dashboard and Management](#10-scheduling-dashboard-and-management)
11. [Implementation Code Examples](#11-implementation-code-examples)
12. [Deployment and Operations](#12-deployment-and-operations)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI AGENT SYSTEM - SCHEDULING LAYER                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SCHEDULING ORCHESTRATOR                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Cron Parser │  │   Scheduler  │  │   Registry   │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          ▼                   ▼                   ▼                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐            │
│  │ Windows Task │   │  Node.js     │   │    Job Queue         │            │
│  │ Scheduler    │   │  Cron Engine │   │    (BullMQ/Agenda)   │            │
│  │ Interface    │   │  (node-cron) │   │                      │            │
│  └──────────────┘   └──────────────┘   └──────────────────────┘            │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        JOB EXECUTION ENGINE                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Agent    │ │ Heartbeat│ │ Gmail    │ │ Browser  │ │ TTS/STT  │  │   │
│  │  │ Loop 1   │ │ Monitor  │ │ Sync     │ │ Control  │ │ Service  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │  │   │
│  │  │ Loop 6   │ │ Loop 7   │ │ Loop 8   │ │ Loop 9   │ │ Loop 10  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │  │   │
│  │  │ Loop 11  │ │ Loop 12  │ │ Loop 13  │ │ Loop 14  │ │ Loop 15  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PERSISTENCE & MONITORING                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ SQLite/JSON  │  │    Redis     │  │   Dashboard  │              │   │
│  │  │ State Store  │  │   (BullMQ)   │  │     UI       │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| Scheduling Orchestrator | Central coordination of all scheduled tasks | Node.js + TypeScript |
| Cron Parser | Parse and validate cron expressions | Custom + node-cron |
| Windows Task Scheduler Interface | Native Windows integration | PowerShell + schtasks.exe |
| Node.js Cron Engine | In-process task scheduling | node-cron / node-schedule |
| Job Queue | Distributed job processing | BullMQ with Redis |
| Persistence Layer | State storage across restarts | SQLite + JSON files |
| Heartbeat Monitor | Health checking and failover | Custom implementation |
| Dashboard UI | Management and monitoring | Express + WebSocket |

---

## 2. Cron Expression Parsing and Scheduling

### 2.1 Cron Expression Format

The system supports standard cron expressions with seconds precision:

```
# ┌────────────── second (0-59, optional)
# │ ┌──────────── minute (0-59)
# │ │ ┌────────── hour (0-23)
# │ │ │ ┌──────── day of month (1-31)
# │ │ │ │ ┌────── month (1-12 or JAN-DEC)
# │ │ │ │ │ ┌──── day of week (0-7 or SUN-SAT, 0/7 = Sunday)
# │ │ │ │ │ │
# * * * * * *
```

### 2.2 Special Characters

| Character | Description | Example |
|-----------|-------------|---------|
| `*` | Any value | `* * * * *` = every minute |
| `,` | Value list separator | `0,15,30,45 * * * *` = at 0, 15, 30, 45 minutes |
| `-` | Range of values | `9-17 * * * 1-5` = 9 AM to 5 PM, weekdays |
| `/` | Step values | `*/5 * * * *` = every 5 minutes |
| `L` | Last day of month/week | `0 0 L * *` = last day of month |
| `W` | Nearest weekday | `0 0 15W * *` = nearest weekday to 15th |
| `#` | Nth occurrence | `0 0 * * 1#1` = first Monday of month |

### 2.3 Predefined Schedules

```javascript
const PREDEFINED_SCHEDULES = {
  // Agent Loop Schedules
  'AGENT_LOOP_FAST': '*/30 * * * * *',      // Every 30 seconds
  'AGENT_LOOP_NORMAL': '*/2 * * * *',        // Every 2 minutes
  'AGENT_LOOP_SLOW': '*/5 * * * *',          // Every 5 minutes
  'AGENT_LOOP_HOURLY': '0 * * * *',          // Every hour
  
  // Heartbeat Schedules
  'HEARTBEAT_CRITICAL': '*/5 * * * * *',     // Every 5 seconds
  'HEARTBEAT_NORMAL': '*/30 * * * * *',      // Every 30 seconds
  'HEARTBEAT_RELAXED': '* * * * *',          // Every minute
  
  // Maintenance Schedules
  'DAILY_MAINTENANCE': '0 2 * * *',          // 2 AM daily
  'WEEKLY_CLEANUP': '0 3 * * 0',             // 3 AM Sunday
  'HOURLY_SYNC': '0 * * * *',                // Every hour
  
  // User Activity
  'USER_PRESENCE_CHECK': '*/10 * * * *',     // Every 10 minutes
  'NOTIFICATION_DIGEST': '0 9,17 * * *',     // 9 AM and 5 PM
};
```

### 2.4 Cron Expression Validator

```typescript
interface CronValidationResult {
  valid: boolean;
  expression: string;
  nextRun: Date | null;
  error?: string;
  warnings?: string[];
}

class CronExpressionParser {
  private static readonly CRON_REGEX = 
    /^((\*|[0-5]?\d)([-/](\*|[0-5]?\d))?)(,((\*|[0-5]?\d)([-/](\*|[0-5]?\d))?))*\s+` +
    `((\*|[0-5]?\d)([-/](\*|[0-5]?\d))?)(,((\*|[0-5]?\d)([-/](\*|[0-5]?\d))?))*\s+` +
    `((\*|(1?[0-9]|2[0-3]))([-/](\*|(1?[0-9]|2[0-3])))?)(,((\*|(1?[0-9]|2[0-3]))([-/](\*|(1?[0-9]|2[0-3])))?))*\s+` +
    `((\*|([1-9]|[12]\d|3[01]))([-/](\*|([1-9]|[12]\d|3[01])))?)(,((\*|([1-9]|[12]\d|3[01]))([-/](\*|([1-9]|[12]\d|3[01])))?))*\s+` +
    `((\*|([1-9]|1[0-2])|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)([-/](\*|([1-9]|1[0-2])|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))?)(,((\*|([1-9]|1[0-2])|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)([-/](\*|([1-9]|1[0-2])|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))?))*\s+` +
    `((\*|[0-6]|MON|TUE|WED|THU|FRI|SAT|SUN)(#([1-5]))?([-/](\*|[0-6]|MON|TUE|WED|THU|FRI|SAT|SUN))?)(,((\*|[0-6]|MON|TUE|WED|THU|FRI|SAT|SUN)(#([1-5]))?([-/](\*|[0-6]|MON|TUE|WED|THU|FRI|SAT|SUN))?))*$/i;

  static validate(expression: string): CronValidationResult {
    try {
      // Check basic format
      if (!this.CRON_REGEX.test(expression)) {
        return {
          valid: false,
          expression,
          nextRun: null,
          error: 'Invalid cron expression format'
        };
      }

      // Use node-cron to validate
      const isValid = cron.validate(expression);
      
      if (!isValid) {
        return {
          valid: false,
          expression,
          nextRun: null,
          error: 'Cron expression failed validation'
        };
      }

      // Calculate next run time
      const interval = cron.parseExpression(expression);
      const nextRun = interval.next().toDate();

      return {
        valid: true,
        expression,
        nextRun,
        warnings: this.checkForWarnings(expression)
      };
    } catch (error) {
      return {
        valid: false,
        expression,
        nextRun: null,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private static checkForWarnings(expression: string): string[] {
    const warnings: string[] = [];
    const parts = expression.split(' ');
    
    // Check for potentially problematic schedules
    if (parts[0] === '*' || (parts[0].includes('/') && parseInt(parts[0].split('/')[1]) < 10)) {
      warnings.push('Sub-minute scheduling may cause high CPU usage');
    }
    
    // Check for overlapping patterns
    if (parts[1] === '*' && parts[2] === '*') {
      warnings.push('Running every minute may impact performance');
    }
    
    return warnings;
  }
}
```

---

## 3. Windows Task Scheduler Integration

### 3.1 Architecture

The Windows Task Scheduler integration provides:
- Native Windows task creation and management
- System-level persistence across reboots
- Integration with Windows Event Log
- Power management awareness

### 3.2 PowerShell Task Management Module

```powershell
# TaskSchedulerManager.ps1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Windows Task Scheduler management for AI Agent System
.DESCRIPTION
    Provides comprehensive task scheduling capabilities for the 24/7 AI Agent system
#>

param(
    [Parameter()]
    [string]$TaskName = "OpenClawAgent",
    
    [Parameter()]
    [string]$NodePath = "C:\Program Files\nodejs\node.exe",
    
    [Parameter()]
    [string]$ScriptPath = "C:\OpenClaw\agent.js",
    
    [Parameter()]
    [string]$WorkingDirectory = "C:\OpenClaw",
    
    [Parameter()]
    [ValidateSet("Create", "Delete", "Enable", "Disable", "Get", "Run", "Stop")]
    [string]$Action = "Get"
)

# Task configuration
$TaskConfig = @{
    Name = $TaskName
    Path = "\OpenClaw\"
    Description = "OpenClaw AI Agent - 24/7 Autonomous Task Execution"
    Author = "OpenClaw System"
    
    # Security settings
    Principal = @{
        UserId = "NT AUTHORITY\SYSTEM"
        LogonType = "ServiceAccount"
        RunLevel = "Highest"
    }
    
    # Settings
    Settings = @{
        AllowStartIfOnBatteries = $true
        DontStopIfGoingOnBatteries = $true
        StartWhenAvailable = $true
        RunOnlyIfNetworkAvailable = $false
        IdleDuration = "PT10M"
        IdleWaitTimeout = "PT1H"
        ExecutionTimeLimit = "PT0S"  # No limit
        MultipleInstancesPolicy = "IgnoreNew"  # Prevent overlapping
        RestartCount = 3
        RestartInterval = "PT1M"
        StopIfGoingOnBatteries = $false
        WakeToRun = $true
    }
    
    # Triggers
    Triggers = @(
        @{
            Type = "BootTrigger"
            Enabled = $true
        },
        @{
            Type = "LogonTrigger"
            Enabled = $true
            UserId = $env:USERNAME
        },
        @{
            Type = "DailyTrigger"
            Enabled = $true
            DaysInterval = 1
            StartBoundary = (Get-Date -Format "yyyy-MM-ddT02:00:00")
        }
    )
    
    # Actions
    Actions = @{
        Execute = $NodePath
        Arguments = "`"$ScriptPath`" --mode=service"
        WorkingDirectory = $WorkingDirectory
    }
}

function Initialize-TaskScheduler {
    <#
    .SYNOPSIS
        Initialize the Task Scheduler COM object
    #>
    try {
        $service = New-Object -ComObject "Schedule.Service"
        $service.Connect()
        return $service
    }
    catch {
        Write-Error "Failed to connect to Task Scheduler service: $_"
        exit 1
    }
}

function New-AgentTask {
    <#
    .SYNOPSIS
        Create a new scheduled task for the AI Agent
    #>
    param([object]$Service)
    
    try {
        # Get or create folder
        $rootFolder = $Service.GetFolder("\")
        try {
            $folder = $rootFolder.GetFolder("OpenClaw")
        }
        catch {
            $folder = $rootFolder.CreateFolder("OpenClaw")
        }
        
        # Create task definition
        $taskDef = $Service.NewTask(0)
        $taskDef.RegistrationInfo.Description = $TaskConfig.Description
        $taskDef.RegistrationInfo.Author = $TaskConfig.Author
        
        # Configure principal (security)
        $principal = $taskDef.Principal
        $principal.UserId = $TaskConfig.Principal.UserId
        $principal.LogonType = 0  # SERVICE_ACCOUNT
        $principal.RunLevel = 1   # Highest
        
        # Configure settings
        $settings = $taskDef.Settings
        $settings.AllowStartIfOnBatteries = $TaskConfig.Settings.AllowStartIfOnBatteries
        $settings.DontStopIfGoingOnBatteries = $TaskConfig.Settings.DontStopIfGoingOnBatteries
        $settings.StartWhenAvailable = $TaskConfig.Settings.StartWhenAvailable
        $settings.RunOnlyIfNetworkAvailable = $TaskConfig.Settings.RunOnlyIfNetworkAvailable
        $settings.ExecutionTimeLimit = $TaskConfig.Settings.ExecutionTimeLimit
        $settings.MultipleInstances = 2  # Ignore new instances
        $settings.RestartCount = $TaskConfig.Settings.RestartCount
        $settings.RestartInterval = $TaskConfig.Settings.RestartInterval
        $settings.StopIfGoingOnBatteries = $TaskConfig.Settings.StopIfGoingOnBatteries
        $settings.WakeToRun = $TaskConfig.Settings.WakeToRun
        
        # Add boot trigger
        $bootTrigger = $taskDef.Triggers.Create(8)  # BOOT
        $bootTrigger.Enabled = $true
        
        # Add logon trigger
        $logonTrigger = $taskDef.Triggers.Create(9)  # LOGON
        $logonTrigger.Enabled = $true
        
        # Add daily maintenance trigger
        $dailyTrigger = $taskDef.Triggers.Create(2)  # DAILY
        $dailyTrigger.DaysInterval = 1
        $dailyTrigger.StartBoundary = $TaskConfig.Triggers[2].StartBoundary
        $dailyTrigger.Enabled = $true
        
        # Add action
        $action = $taskDef.Actions.Create(0)  # EXEC
        $action.Path = $TaskConfig.Actions.Execute
        $action.Arguments = $TaskConfig.Actions.Arguments
        $action.WorkingDirectory = $TaskConfig.Actions.WorkingDirectory
        
        # Register task
        $folder.RegisterTaskDefinition(
            $TaskConfig.Name,
            $taskDef,
            6,  # CREATE_OR_UPDATE
            $null,
            $null,
            0   # SERVICE_ACCOUNT
        )
        
        Write-Host "Task '$($TaskConfig.Name)' created successfully" -ForegroundColor Green
        
        # Start the task immediately
        Start-AgentTask -Service $Service
    }
    catch {
        Write-Error "Failed to create task: $_"
    }
}

function Remove-AgentTask {
    <#
    .SYNOPSIS
        Remove the scheduled task
    #>
    param([object]$Service)
    
    try {
        $folder = $Service.GetFolder($TaskConfig.Path)
        $folder.DeleteTask($TaskConfig.Name, 0)
        Write-Host "Task '$($TaskConfig.Name)' deleted successfully" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to delete task: $_"
    }
}

function Start-AgentTask {
    <#
    .SYNOPSIS
        Start the scheduled task
    #>
    param([object]$Service)
    
    try {
        $folder = $Service.GetFolder($TaskConfig.Path)
        $task = $folder.GetTask($TaskConfig.Name)
        $task.Run($null)
        Write-Host "Task '$($TaskConfig.Name)' started" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to start task: $_"
    }
}

function Stop-AgentTask {
    <#
    .SYNOPSIS
        Stop the scheduled task
    #>
    param([object]$Service)
    
    try {
        $folder = $Service.GetFolder($TaskConfig.Path)
        $task = $folder.GetTask($TaskConfig.Name)
        $task.Stop(0)
        Write-Host "Task '$($TaskConfig.Name)' stopped" -ForegroundColor Yellow
    }
    catch {
        Write-Error "Failed to stop task: $_"
    }
}

function Get-AgentTask {
    <#
    .SYNOPSIS
        Get task status and information
    #>
    param([object]$Service)
    
    try {
        $folder = $Service.GetFolder($TaskConfig.Path)
        $task = $folder.GetTask($TaskConfig.Name)
        
        return [PSCustomObject]@{
            Name = $task.Name
            Path = $task.Path
            State = switch ($task.State) {
                0 { "Unknown" }
                1 { "Disabled" }
                2 { "Queued" }
                3 { "Ready" }
                4 { "Running" }
                default { "Unknown ($($task.State))" }
            }
            LastRunTime = $task.LastRunTime
            NextRunTime = $task.NextRunTime
            LastTaskResult = $task.LastTaskResult
            NumberOfMissedRuns = $task.NumberOfMissedRuns
            Enabled = $task.Enabled
        }
    }
    catch {
        Write-Error "Task not found: $_"
        return $null
    }
}

# Main execution
$Service = Initialize-TaskScheduler

switch ($Action) {
    "Create" { New-AgentTask -Service $Service }
    "Delete" { Remove-AgentTask -Service $Service }
    "Enable" { 
        $folder = $Service.GetFolder($TaskConfig.Path)
        $task = $folder.GetTask($TaskConfig.Name)
        $task.Enabled = $true
    }
    "Disable" { 
        $folder = $Service.GetFolder($TaskConfig.Path)
        $task = $folder.GetTask($TaskConfig.Name)
        $task.Enabled = $false
    }
    "Get" { Get-AgentTask -Service $Service | Format-List }
    "Run" { Start-AgentTask -Service $Service }
    "Stop" { Stop-AgentTask -Service $Service }
}
```

### 3.3 Node.js Windows Task Scheduler Interface

```typescript
// windows-task-scheduler.ts
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

export interface WindowsTaskConfig {
  taskName: string;
  nodePath: string;
  scriptPath: string;
  workingDirectory: string;
  schedule?: string;
  arguments?: string[];
  runAsSystem?: boolean;
  wakeToRun?: boolean;
  allowOnBatteries?: boolean;
}

export interface TaskStatus {
  name: string;
  state: 'unknown' | 'disabled' | 'queued' | 'ready' | 'running';
  lastRunTime: Date | null;
  nextRunTime: Date | null;
  lastTaskResult: number;
  missedRuns: number;
  enabled: boolean;
}

export class WindowsTaskScheduler {
  private readonly taskFolder = '\\OpenClaw\\';
  
  constructor(private config: WindowsTaskConfig) {}

  /**
   * Create a new scheduled task
   */
  async createTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const args = this.config.arguments?.join(' ') || '';
    
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-NodePath "${this.config.nodePath}" ` +
      `-ScriptPath "${this.config.scriptPath}" ` +
      `-WorkingDirectory "${this.config.workingDirectory}" ` +
      `-Action "Create"`;
    
    try {
      const { stdout, stderr } = await execAsync(command);
      if (stderr) {
        console.error('PowerShell stderr:', stderr);
      }
      console.log('Task created:', stdout);
      return true;
    } catch (error) {
      console.error('Failed to create task:', error);
      return false;
    }
  }

  /**
   * Delete the scheduled task
   */
  async deleteTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Delete"`;
    
    try {
      await execAsync(command);
      return true;
    } catch (error) {
      console.error('Failed to delete task:', error);
      return false;
    }
  }

  /**
   * Get task status
   */
  async getTaskStatus(): Promise<TaskStatus | null> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Get" | ConvertTo-Json`;
    
    try {
      const { stdout } = await execAsync(command);
      const data = JSON.parse(stdout);
      
      return {
        name: data.Name,
        state: data.State.toLowerCase() as TaskStatus['state'],
        lastRunTime: data.LastRunTime ? new Date(data.LastRunTime) : null,
        nextRunTime: data.NextRunTime ? new Date(data.NextRunTime) : null,
        lastTaskResult: data.LastTaskResult,
        missedRuns: data.NumberOfMissedRuns,
        enabled: data.Enabled
      };
    } catch (error) {
      console.error('Failed to get task status:', error);
      return null;
    }
  }

  /**
   * Start the task immediately
   */
  async startTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Run"`;
    
    try {
      await execAsync(command);
      return true;
    } catch (error) {
      console.error('Failed to start task:', error);
      return false;
    }
  }

  /**
   * Stop the running task
   */
  async stopTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Stop"`;
    
    try {
      await execAsync(command);
      return true;
    } catch (error) {
      console.error('Failed to stop task:', error);
      return false;
    }
  }

  /**
   * Enable the task
   */
  async enableTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Enable"`;
    
    try {
      await execAsync(command);
      return true;
    } catch (error) {
      console.error('Failed to enable task:', error);
      return false;
    }
  }

  /**
   * Disable the task
   */
  async disableTask(): Promise<boolean> {
    const psScript = path.join(__dirname, 'TaskSchedulerManager.ps1');
    const command = `powershell.exe -ExecutionPolicy Bypass -File "${psScript}" ` +
      `-TaskName "${this.config.taskName}" ` +
      `-Action "Disable"`;
    
    try {
      await execAsync(command);
      return true;
    } catch (error) {
      console.error('Failed to disable task:', error);
      return false;
    }
  }

  /**
   * Check if task exists
   */
  async taskExists(): Promise<boolean> {
    const status = await this.getTaskStatus();
    return status !== null;
  }

  /**
   * Ensure task is running (create if not exists, start if not running)
   */
  async ensureRunning(): Promise<boolean> {
    const exists = await this.taskExists();
    
    if (!exists) {
      console.log('Task does not exist, creating...');
      await this.createTask();
    }
    
    const status = await this.getTaskStatus();
    if (status?.state !== 'running') {
      console.log('Task not running, starting...');
      await this.startTask();
    }
    
    return true;
  }
}
```

---

## 4. Node.js Cron Libraries

### 4.1 Library Comparison

| Library | Persistence | Concurrency | Retry | Memory | Best For |
|---------|-------------|-------------|-------|--------|----------|
| **node-cron** | No | Manual | Manual | Low | Simple in-process |
| **node-schedule** | No | Manual | Manual | Low | Date-based scheduling |
| **BullMQ** | Redis | Built-in | Built-in | Medium | Production queues |
| **Agenda** | MongoDB | Built-in | Built-in | Medium | Complex workflows |
| **Bree** | File-based | Built-in | Manual | Low | Worker threads |

### 4.2 Recommended: Hybrid Approach

For the 24/7 AI Agent system, we recommend a hybrid approach:

1. **node-cron** for lightweight, in-process scheduling
2. **BullMQ** for job queue management and persistence
3. **Windows Task Scheduler** for system-level task management

### 4.3 Core Cron Engine Implementation

```typescript
// cron-engine.ts
import * as cron from 'node-cron';
import { EventEmitter } from 'events';
import { Mutex } from 'async-mutex';

export interface ScheduledJob {
  id: string;
  name: string;
  schedule: string;
  handler: () => Promise<void>;
  options: JobOptions;
  status: JobStatus;
  metrics: JobMetrics;
}

export interface JobOptions {
  timezone?: string;
  preventOverlap?: boolean;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enabled?: boolean;
  tags?: string[];
}

export interface JobStatus {
  state: 'idle' | 'running' | 'paused' | 'error';
  lastRun: Date | null;
  nextRun: Date | null;
  lastError: Error | null;
  consecutiveFailures: number;
  totalRuns: number;
  totalFailures: number;
}

export interface JobMetrics {
  averageExecutionTime: number;
  maxExecutionTime: number;
  minExecutionTime: number;
  totalExecutionTime: number;
}

export interface JobExecutionResult {
  jobId: string;
  success: boolean;
  startTime: Date;
  endTime: Date;
  duration: number;
  error?: Error;
}

export class CronEngine extends EventEmitter {
  private jobs: Map<string, ScheduledJob> = new Map();
  private cronTasks: Map<string, cron.ScheduledTask> = new Map();
  private jobLocks: Map<string, Mutex> = new Map();
  private runningJobs: Map<string, boolean> = new Map();

  constructor() {
    super();
  }

  /**
   * Register a new scheduled job
   */
  register(
    id: string,
    name: string,
    schedule: string,
    handler: () => Promise<void>,
    options: JobOptions = {}
  ): ScheduledJob {
    // Validate cron expression
    if (!cron.validate(schedule)) {
      throw new Error(`Invalid cron expression: ${schedule}`);
    }

    const job: ScheduledJob = {
      id,
      name,
      schedule,
      handler,
      options: {
        timezone: 'UTC',
        preventOverlap: true,
        timeout: 300000, // 5 minutes
        retryAttempts: 3,
        retryDelay: 5000,
        enabled: true,
        tags: [],
        ...options
      },
      status: {
        state: 'idle',
        lastRun: null,
        nextRun: null,
        lastError: null,
        consecutiveFailures: 0,
        totalRuns: 0,
        totalFailures: 0
      },
      metrics: {
        averageExecutionTime: 0,
        maxExecutionTime: 0,
        minExecutionTime: Infinity,
        totalExecutionTime: 0
      }
    };

    this.jobs.set(id, job);
    this.jobLocks.set(id, new Mutex());
    this.runningJobs.set(id, false);

    // Schedule the job
    if (job.options.enabled) {
      this.scheduleJob(job);
    }

    this.emit('job:registered', job);
    return job;
  }

  /**
   * Schedule a job with node-cron
   */
  private scheduleJob(job: ScheduledJob): void {
    const task = cron.schedule(
      job.schedule,
      async () => {
        await this.executeJob(job);
      },
      {
        scheduled: true,
        timezone: job.options.timezone
      }
    );

    this.cronTasks.set(job.id, task);

    // Calculate next run time
    const interval = cron.parseExpression(job.schedule, {
      timezone: job.options.timezone
    });
    job.status.nextRun = interval.next().toDate();
  }

  /**
   * Execute a job with overlap prevention and error handling
   */
  private async executeJob(job: ScheduledJob): Promise<void> {
    const lock = this.jobLocks.get(job.id);
    if (!lock) return;

    // Check for overlap
    if (job.options.preventOverlap && this.runningJobs.get(job.id)) {
      this.emit('job:skipped', { jobId: job.id, reason: 'overlap' });
      return;
    }

    const release = await lock.acquire();
    const startTime = new Date();

    try {
      this.runningJobs.set(job.id, true);
      job.status.state = 'running';
      job.status.lastRun = startTime;
      job.status.totalRuns++;

      this.emit('job:started', { jobId: job.id, startTime });

      // Execute with timeout
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error(`Job ${job.id} timed out after ${job.options.timeout}ms`));
        }, job.options.timeout);
      });

      await Promise.race([job.handler(), timeoutPromise]);

      // Success
      const endTime = new Date();
      const duration = endTime.getTime() - startTime.getTime();

      job.status.state = 'idle';
      job.status.consecutiveFailures = 0;
      job.status.lastError = null;

      // Update metrics
      this.updateMetrics(job, duration);

      const result: JobExecutionResult = {
        jobId: job.id,
        success: true,
        startTime,
        endTime,
        duration
      };

      this.emit('job:completed', result);

    } catch (error) {
      const endTime = new Date();
      const duration = endTime.getTime() - startTime.getTime();

      job.status.state = 'error';
      job.status.lastError = error as Error;
      job.status.consecutiveFailures++;
      job.status.totalFailures++;

      const result: JobExecutionResult = {
        jobId: job.id,
        success: false,
        startTime,
        endTime,
        duration,
        error: error as Error
      };

      this.emit('job:failed', result);

      // Retry logic
      if (job.status.consecutiveFailures < (job.options.retryAttempts || 0)) {
        setTimeout(() => {
          this.executeJob(job);
        }, job.options.retryDelay);
      }

    } finally {
      this.runningJobs.set(job.id, false);
      release();

      // Update next run time
      const interval = cron.parseExpression(job.schedule, {
        timezone: job.options.timezone
      });
      job.status.nextRun = interval.next().toDate();
    }
  }

  /**
   * Update job execution metrics
   */
  private updateMetrics(job: ScheduledJob, duration: number): void {
    const metrics = job.metrics;
    metrics.totalExecutionTime += duration;
    metrics.maxExecutionTime = Math.max(metrics.maxExecutionTime, duration);
    metrics.minExecutionTime = Math.min(metrics.minExecutionTime, duration);
    metrics.averageExecutionTime = metrics.totalExecutionTime / job.status.totalRuns;
  }

  /**
   * Start a specific job
   */
  start(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.start();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = true;
        this.emit('job:started', { jobId });
      }
      return true;
    }
    return false;
  }

  /**
   * Stop a specific job
   */
  stop(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.stop();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = false;
        job.status.state = 'paused';
        this.emit('job:stopped', { jobId });
      }
      return true;
    }
    return false;
  }

  /**
   * Run a job immediately (one-time execution)
   */
  async runNow(jobId: string): Promise<boolean> {
    const job = this.jobs.get(jobId);
    if (job) {
      await this.executeJob(job);
      return true;
    }
    return false;
  }

  /**
   * Remove a job
   */
  unregister(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.stop();
      task.destroy();
      this.cronTasks.delete(jobId);
    }

    this.jobs.delete(jobId);
    this.jobLocks.delete(jobId);
    this.runningJobs.delete(jobId);

    this.emit('job:unregistered', { jobId });
    return true;
  }

  /**
   * Get job information
   */
  getJob(jobId: string): ScheduledJob | undefined {
    return this.jobs.get(jobId);
  }

  /**
   * Get all jobs
   */
  getAllJobs(): ScheduledJob[] {
    return Array.from(this.jobs.values());
  }

  /**
   * Get jobs by tag
   */
  getJobsByTag(tag: string): ScheduledJob[] {
    return this.getAllJobs().filter(job => job.options.tags?.includes(tag));
  }

  /**
   * Stop all jobs
   */
  stopAll(): void {
    for (const [jobId, task] of this.cronTasks) {
      task.stop();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = false;
      }
    }
    this.emit('all:stopped');
  }

  /**
   * Start all jobs
   */
  startAll(): void {
    for (const [jobId, task] of this.cronTasks) {
      task.start();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = true;
      }
    }
    this.emit('all:started');
  }

  /**
   * Get system health
   */
  getHealth(): {
    totalJobs: number;
    runningJobs: number;
    failedJobs: number;
    idleJobs: number;
  } {
    const jobs = this.getAllJobs();
    return {
      totalJobs: jobs.length,
      runningJobs: jobs.filter(j => j.status.state === 'running').length,
      failedJobs: jobs.filter(j => j.status.consecutiveFailures > 0).length,
      idleJobs: jobs.filter(j => j.status.state === 'idle').length
    };
  }
}
```

---

## 5. Job Queue Management

### 5.1 BullMQ Integration

BullMQ provides Redis-backed job queues with:
- Persistence across restarts
- Job priorities
- Delayed jobs
- Rate limiting
- Job progress tracking
- Dead letter queues

```typescript
// job-queue.ts
import { Queue, Worker, Job, QueueScheduler } from 'bullmq';
import IORedis from 'ioredis';

export interface QueueJob {
  id?: string;
  name: string;
  data: any;
  opts?: {
    priority?: number;
    delay?: number;
    attempts?: number;
    backoff?: {
      type: 'fixed' | 'exponential';
      delay: number;
    };
    removeOnComplete?: boolean | number;
    removeOnFail?: boolean | number;
  };
}

export interface QueueConfig {
  name: string;
  concurrency?: number;
  redis?: {
    host: string;
    port: number;
    password?: string;
    db?: number;
  };
}

export class JobQueueManager {
  private queues: Map<string, Queue> = new Map();
  private workers: Map<string, Worker> = new Map();
  private connection: IORedis;

  constructor(redisConfig: {
    host: string;
    port: number;
    password?: string;
    db?: number;
  }) {
    this.connection = new IORedis({
      host: redisConfig.host,
      port: redisConfig.port,
      password: redisConfig.password,
      db: redisConfig.db || 0,
      maxRetriesPerRequest: null,
      enableReadyCheck: false,
    });
  }

  /**
   * Create a new queue
   */
  createQueue(config: QueueConfig): Queue {
    const queue = new Queue(config.name, {
      connection: this.connection,
      defaultJobOptions: {
        removeOnComplete: 100,
        removeOnFail: 50,
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 1000,
        },
      },
    });

    this.queues.set(config.name, queue);
    return queue;
  }

  /**
   * Create a worker for a queue
   */
  createWorker(
    queueName: string,
    processor: (job: Job) => Promise<any>,
    concurrency: number = 1
  ): Worker {
    const worker = new Worker(queueName, processor, {
      connection: this.connection,
      concurrency,
      limiter: {
        max: 100,
        duration: 1000,
      },
    });

    // Event handlers
    worker.on('completed', (job) => {
      console.log(`Job ${job.id} completed`);
    });

    worker.on('failed', (job, err) => {
      console.error(`Job ${job?.id} failed:`, err);
    });

    worker.on('progress', (job, progress) => {
      console.log(`Job ${job.id} progress: ${progress}%`);
    });

    this.workers.set(queueName, worker);
    return worker;
  }

  /**
   * Add a job to a queue
   */
  async addJob(queueName: string, job: QueueJob): Promise<Job> {
    const queue = this.queues.get(queueName);
    if (!queue) {
      throw new Error(`Queue ${queueName} not found`);
    }

    return await queue.add(job.name, job.data, job.opts);
  }

  /**
   * Add a recurring job
   */
  async addRecurringJob(
    queueName: string,
    job: QueueJob,
    cron: string
  ): Promise<Job> {
    const queue = this.queues.get(queueName);
    if (!queue) {
      throw new Error(`Queue ${queueName} not found`);
    }

    return await queue.add(job.name, job.data, {
      ...job.opts,
      repeat: {
        pattern: cron,
      },
    });
  }

  /**
   * Get queue status
   */
  async getQueueStatus(queueName: string): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }> {
    const queue = this.queues.get(queueName);
    if (!queue) {
      throw new Error(`Queue ${queueName} not found`);
    }

    const [waiting, active, completed, failed, delayed, paused] = await Promise.all([
      queue.getWaitingCount(),
      queue.getActiveCount(),
      queue.getCompletedCount(),
      queue.getFailedCount(),
      queue.getDelayedCount(),
      queue.getPausedCount(),
    ]);

    return { waiting, active, completed, failed, delayed, paused };
  }

  /**
   * Clean completed jobs
   */
  async cleanQueue(queueName: string, gracePeriodMs: number = 3600000): Promise<void> {
    const queue = this.queues.get(queueName);
    if (!queue) {
      throw new Error(`Queue ${queueName} not found`);
    }

    await queue.clean(gracePeriodMs, 100, 'completed');
    await queue.clean(gracePeriodMs, 100, 'failed');
  }

  /**
   * Pause a queue
   */
  async pauseQueue(queueName: string): Promise<void> {
    const queue = this.queues.get(queueName);
    if (queue) {
      await queue.pause();
    }
  }

  /**
   * Resume a queue
   */
  async resumeQueue(queueName: string): Promise<void> {
    const queue = this.queues.get(queueName);
    if (queue) {
      await queue.resume();
    }
  }

  /**
   * Close all connections
   */
  async close(): Promise<void> {
    for (const worker of this.workers.values()) {
      await worker.close();
    }
    for (const queue of this.queues.values()) {
      await queue.close();
    }
    await this.connection.quit();
  }
}
```

---

## 6. Task Persistence Across Restarts

### 6.1 Persistence Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  SQLite Store    │    │   JSON Files     │                  │
│  │  (Primary)       │    │   (Backup)       │                  │
│  │                  │    │                  │                  │
│  │  - Job states    │    │  - Config        │                  │
│  │  - Execution log │    │  - Snapshots     │                  │
│  │  - Metrics       │    │  - Recovery      │                  │
│  └──────────────────┘    └──────────────────┘                  │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Persistence Manager                         │   │
│  │  - Auto-save every 30s                                   │   │
│  │  - Write-ahead logging                                   │   │
│  │  - Automatic recovery                                    │   │
│  │  - Compression for old data                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 SQLite Persistence Implementation

```typescript
// persistence.ts
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import * as path from 'path';
import * as fs from 'fs';

export interface PersistedJob {
  id: string;
  name: string;
  schedule: string;
  options: string; // JSON
  status: string; // JSON
  metrics: string; // JSON
  lastUpdated: number;
  createdAt: number;
}

export interface ExecutionLog {
  id: string;
  jobId: string;
  startTime: number;
  endTime: number;
  duration: number;
  success: boolean;
  error?: string;
}

export class PersistenceManager {
  private db: Database<sqlite3.Database, sqlite3.Statement> | null = null;
  private readonly dbPath: string;
  private readonly backupPath: string;
  private autoSaveInterval: NodeJS.Timeout | null = null;

  constructor(dataDir: string = './data') {
    this.dbPath = path.join(dataDir, 'scheduler.db');
    this.backupPath = path.join(dataDir, 'backups');
    
    // Ensure directories exist
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    if (!fs.existsSync(this.backupPath)) {
      fs.mkdirSync(this.backupPath, { recursive: true });
    }
  }

  /**
   * Initialize the database
   */
  async initialize(): Promise<void> {
    this.db = await open({
      filename: this.dbPath,
      driver: sqlite3.Database,
    });

    // Create tables
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        schedule TEXT NOT NULL,
        options TEXT NOT NULL,
        status TEXT NOT NULL,
        metrics TEXT NOT NULL,
        lastUpdated INTEGER NOT NULL,
        createdAt INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS execution_log (
        id TEXT PRIMARY KEY,
        jobId TEXT NOT NULL,
        startTime INTEGER NOT NULL,
        endTime INTEGER NOT NULL,
        duration INTEGER NOT NULL,
        success INTEGER NOT NULL,
        error TEXT,
        FOREIGN KEY (jobId) REFERENCES jobs(id)
      );

      CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updatedAt INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_execution_log_jobId ON execution_log(jobId);
      CREATE INDEX IF NOT EXISTS idx_execution_log_startTime ON execution_log(startTime);
      CREATE INDEX IF NOT EXISTS idx_jobs_lastUpdated ON jobs(lastUpdated);
    `);

    // Start auto-save
    this.startAutoSave();
  }

  /**
   * Save a job to the database
   */
  async saveJob(job: PersistedJob): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT OR REPLACE INTO jobs (id, name, schedule, options, status, metrics, lastUpdated, createdAt)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      job.id,
      job.name,
      job.schedule,
      job.options,
      job.status,
      job.metrics,
      Date.now(),
      job.createdAt || Date.now()
    ]);
  }

  /**
   * Load all jobs from the database
   */
  async loadJobs(): Promise<PersistedJob[]> {
    if (!this.db) throw new Error('Database not initialized');

    return await this.db.all<PersistedJob[]>(`SELECT * FROM jobs`);
  }

  /**
   * Delete a job from the database
   */
  async deleteJob(jobId: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`DELETE FROM jobs WHERE id = ?`, [jobId]);
    await this.db.run(`DELETE FROM execution_log WHERE jobId = ?`, [jobId]);
  }

  /**
   * Log a job execution
   */
  async logExecution(log: ExecutionLog): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT INTO execution_log (id, jobId, startTime, endTime, duration, success, error)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `, [
      log.id,
      log.jobId,
      log.startTime,
      log.endTime,
      log.duration,
      log.success ? 1 : 0,
      log.error
    ]);
  }

  /**
   * Get execution history for a job
   */
  async getExecutionHistory(jobId: string, limit: number = 100): Promise<ExecutionLog[]> {
    if (!this.db) throw new Error('Database not initialized');

    return await this.db.all<ExecutionLog[]>(`
      SELECT * FROM execution_log 
      WHERE jobId = ? 
      ORDER BY startTime DESC 
      LIMIT ?
    `, [jobId, limit]);
  }

  /**
   * Save system state
   */
  async saveSystemState(key: string, value: any): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT OR REPLACE INTO system_state (key, value, updatedAt)
      VALUES (?, ?, ?)
    `, [key, JSON.stringify(value), Date.now()]);
  }

  /**
   * Load system state
   */
  async loadSystemState(key: string): Promise<any | null> {
    if (!this.db) throw new Error('Database not initialized');

    const row = await this.db.get<{ value: string }>(`
      SELECT value FROM system_state WHERE key = ?
    `, [key]);

    return row ? JSON.parse(row.value) : null;
  }

  /**
   * Create a backup
   */
  async createBackup(): Promise<string> {
    const backupFile = path.join(
      this.backupPath,
      `scheduler_backup_${Date.now()}.db`
    );

    await this.db?.exec(`VACUUM INTO '${backupFile}'`);
    return backupFile;
  }

  /**
   * Restore from backup
   */
  async restoreFromBackup(backupFile: string): Promise<void> {
    if (!fs.existsSync(backupFile)) {
      throw new Error(`Backup file not found: ${backupFile}`);
    }

    // Close current connection
    await this.close();

    // Copy backup to main database
    fs.copyFileSync(backupFile, this.dbPath);

    // Reinitialize
    await this.initialize();
  }

  /**
   * Clean old execution logs
   */
  async cleanOldLogs(olderThanDays: number = 30): Promise<number> {
    if (!this.db) throw new Error('Database not initialized');

    const cutoff = Date.now() - (olderThanDays * 24 * 60 * 60 * 1000);
    
    const result = await this.db.run(`
      DELETE FROM execution_log WHERE startTime < ?
    `, [cutoff]);

    return result.changes || 0;
  }

  /**
   * Start auto-save interval
   */
  private startAutoSave(): void {
    this.autoSaveInterval = setInterval(async () => {
      await this.createBackup();
      await this.cleanOldLogs(30);
    }, 30000); // Every 30 seconds
  }

  /**
   * Close the database connection
   */
  async close(): Promise<void> {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }
    await this.db?.close();
    this.db = null;
  }
}
```

---

## 7. Overlapping Job Handling

### 7.1 Prevention Strategies

```typescript
// overlap-prevention.ts
import { Mutex, Semaphore } from 'async-mutex';

export interface OverlapPreventionConfig {
  strategy: 'skip' | 'queue' | 'delay' | 'timeout' | 'concurrent';
  maxConcurrent?: number;
  maxQueueSize?: number;
  delayMs?: number;
  timeoutMs?: number;
}

export class OverlapPrevention {
  private mutexes: Map<string, Mutex> = new Map();
  private semaphores: Map<string, Semaphore> = new Map();
  private jobQueues: Map<string, Array<() => void>> = new Map();
  private runningJobs: Map<string, boolean> = new Map();
  private skipCounts: Map<string, number> = new Map();

  constructor(private config: OverlapPreventionConfig) {}

  /**
   * Execute with overlap prevention
   */
  async execute<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    switch (this.config.strategy) {
      case 'skip':
        return this.executeWithSkip(jobId, fn);
      case 'queue':
        return this.executeWithQueue(jobId, fn);
      case 'delay':
        return this.executeWithDelay(jobId, fn);
      case 'timeout':
        return this.executeWithTimeout(jobId, fn);
      case 'concurrent':
        return this.executeWithConcurrency(jobId, fn);
      default:
        return fn();
    }
  }

  /**
   * Skip strategy - skip if job is already running
   */
  private async executeWithSkip<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      const skipCount = (this.skipCounts.get(jobId) || 0) + 1;
      this.skipCounts.set(jobId, skipCount);
      console.log(`Job ${jobId} skipped (count: ${skipCount})`);
      return null;
    }

    this.runningJobs.set(jobId, true);
    this.skipCounts.set(jobId, 0);

    try {
      return await fn();
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Queue strategy - queue jobs if already running
   */
  private async executeWithQueue<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const mutex = this.getMutex(jobId);
    const release = await mutex.acquire();

    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Delay strategy - delay execution if job is running
   */
  private async executeWithDelay<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      console.log(`Job ${jobId} delaying for ${this.config.delayMs}ms`);
      await this.sleep(this.config.delayMs || 5000);
      
      // Check again after delay
      if (this.runningJobs.get(jobId)) {
        console.log(`Job ${jobId} still running after delay, skipping`);
        return null;
      }
    }

    this.runningJobs.set(jobId, true);

    try {
      return await fn();
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Timeout strategy - force stop after timeout
   */
  private async executeWithTimeout<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      const skipCount = (this.skipCounts.get(jobId) || 0) + 1;
      this.skipCounts.set(jobId, skipCount);

      // Force timeout after max skips
      if (skipCount > 2) {
        console.log(`Job ${jobId} forcing timeout after ${skipCount} skips`);
        this.runningJobs.set(jobId, false);
        this.skipCounts.set(jobId, 0);
      } else {
        console.log(`Job ${jobId} skipped (count: ${skipCount})`);
        return null;
      }
    }

    this.runningJobs.set(jobId, true);
    this.skipCounts.set(jobId, 0);

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Job ${jobId} timed out`));
      }, this.config.timeoutMs || 300000);
    });

    try {
      return await Promise.race([fn(), timeoutPromise]);
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Concurrent strategy - allow limited concurrency
   */
  private async executeWithConcurrency<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const semaphore = this.getSemaphore(jobId);
    const [value, release] = await semaphore.acquire();

    console.log(`Job ${jobId} acquired slot (${value} remaining)`);

    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Get or create mutex for job
   */
  private getMutex(jobId: string): Mutex {
    if (!this.mutexes.has(jobId)) {
      this.mutexes.set(jobId, new Mutex());
    }
    return this.mutexes.get(jobId)!;
  }

  /**
   * Get or create semaphore for job
   */
  private getSemaphore(jobId: string): Semaphore {
    if (!this.semaphores.has(jobId)) {
      this.semaphores.set(
        jobId,
        new Semaphore(this.config.maxConcurrent || 1)
      );
    }
    return this.semaphores.get(jobId)!;
  }

  /**
   * Sleep helper
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get overlap statistics
   */
  getStats(jobId: string): {
    isRunning: boolean;
    skipCount: number;
    queueSize: number;
  } {
    return {
      isRunning: this.runningJobs.get(jobId) || false,
      skipCount: this.skipCounts.get(jobId) || 0,
      queueSize: this.jobQueues.get(jobId)?.length || 0,
    };
  }

  /**
   * Reset stats for a job
   */
  resetStats(jobId: string): void {
    this.skipCounts.set(jobId, 0);
  }
}
```

---

## 8. Job Failure and Retry Mechanisms

### 8.1 Retry Strategy Implementation

```typescript
// retry-mechanism.ts

export type RetryStrategy = 'fixed' | 'exponential' | 'linear' | 'custom';

export interface RetryConfig {
  maxAttempts: number;
  strategy: RetryStrategy;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
  nonRetryableErrors?: string[];
  onRetry?: (attempt: number, error: Error, delay: number) => void;
  onFailed?: (error: Error, attempts: number) => void;
}

export interface RetryResult<T> {
  success: boolean;
  result?: T;
  error?: Error;
  attempts: number;
  totalDuration: number;
}

export class RetryMechanism {
  private defaultConfig: RetryConfig = {
    maxAttempts: 3,
    strategy: 'exponential',
    baseDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
  };

  constructor(private config: Partial<RetryConfig> = {}) {
    this.config = { ...this.defaultConfig, ...config };
  }

  /**
   * Execute with retry logic
   */
  async execute<T>(
    fn: () => Promise<T>,
    customConfig?: Partial<RetryConfig>
  ): Promise<RetryResult<T>> {
    const config = { ...this.config, ...customConfig };
    const startTime = Date.now();
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
      try {
        const result = await fn();
        return {
          success: true,
          result,
          attempts: attempt,
          totalDuration: Date.now() - startTime,
        };
      } catch (error) {
        lastError = error as Error;

        // Check if error is non-retryable
        if (this.isNonRetryable(error as Error, config)) {
          break;
        }

        // Check if error is retryable (if whitelist specified)
        if (!this.isRetryable(error as Error, config)) {
          break;
        }

        // Don't retry on last attempt
        if (attempt === config.maxAttempts) {
          break;
        }

        // Calculate delay
        const delay = this.calculateDelay(attempt, config);

        // Call retry callback
        config.onRetry?.(attempt, lastError, delay);

        // Wait before retry
        await this.sleep(delay);
      }
    }

    // All attempts failed
    config.onFailed?.(lastError!, config.maxAttempts);

    return {
      success: false,
      error: lastError,
      attempts: config.maxAttempts,
      totalDuration: Date.now() - startTime,
    };
  }

  /**
   * Calculate retry delay based on strategy
   */
  private calculateDelay(attempt: number, config: RetryConfig): number {
    let delay: number;

    switch (config.strategy) {
      case 'fixed':
        delay = config.baseDelay;
        break;
      case 'linear':
        delay = config.baseDelay * attempt;
        break;
      case 'exponential':
        delay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt - 1);
        break;
      case 'custom':
        delay = this.customDelay(attempt, config);
        break;
      default:
        delay = config.baseDelay;
    }

    // Cap at max delay
    return Math.min(delay, config.maxDelay);
  }

  /**
   * Custom delay calculation (override for custom strategy)
   */
  protected customDelay(attempt: number, config: RetryConfig): number {
    return config.baseDelay * attempt;
  }

  /**
   * Check if error is retryable
   */
  private isRetryable(error: Error, config: RetryConfig): boolean {
    if (!config.retryableErrors || config.retryableErrors.length === 0) {
      return true; // Retry all by default
    }

    return config.retryableErrors.some(pattern =>
      error.message.includes(pattern) || error.name.includes(pattern)
    );
  }

  /**
   * Check if error is non-retryable
   */
  private isNonRetryable(error: Error, config: RetryConfig): boolean {
    if (!config.nonRetryableErrors || config.nonRetryableErrors.length === 0) {
      return false;
    }

    return config.nonRetryableErrors.some(pattern =>
      error.message.includes(pattern) || error.name.includes(pattern)
    );
  }

  /**
   * Sleep helper
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Circuit breaker pattern for failure handling
 */
export class CircuitBreaker {
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private failureCount = 0;
  private lastFailureTime: number = 0;
  private nextAttempt: number = 0;

  constructor(
    private config: {
      failureThreshold: number;
      resetTimeout: number;
      halfOpenMaxCalls: number;
    }
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = 'half-open';
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    this.state = 'closed';
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = 'open';
      this.nextAttempt = Date.now() + this.config.resetTimeout;
    }
  }

  getState(): { state: string; failureCount: number } {
    return {
      state: this.state,
      failureCount: this.failureCount,
    };
  }
}
```

---

## 9. Heartbeat Mechanism

### 9.1 Heartbeat Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HEARTBEAT SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐          ┌──────────────┐                     │
│  │   Sender     │──────────▶│   Monitor    │                     │
│  │  (Agent)     │  Heartbeat│   (Central)  │                     │
│  └──────────────┘  Messages └──────────────┘                     │
│         │                         │                             │
│         │                         ▼                             │
│         │              ┌──────────────────────┐                │
│         │              │   Health Registry    │                │
│         │              │  - Node status       │                │
│         │              │  - Last heartbeat    │                │
│         │              │  - Missed count      │                │
│         │              │  - Health score      │                │
│         │              └──────────────────────┘                │
│         │                         │                             │
│         │                         ▼                             │
│         │              ┌──────────────────────┐                │
│         │              │   Action Handler     │                │
│         │              │  - Alert on failure  │                │
│         │              │  - Trigger failover  │                │
│         │              │  - Restart services  │                │
│         │              └──────────────────────┘                │
│         │                                                       │
│         └──────────────────────────────────────────────────────▶│
│                    ACK / Status Response                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Heartbeat Implementation

```typescript
// heartbeat.ts
import { EventEmitter } from 'events';

export interface HeartbeatConfig {
  interval: number;
  timeout: number;
  missedThreshold: number;
  nodeId: string;
  metadata?: Record<string, any>;
}

export interface HeartbeatMessage {
  nodeId: string;
  timestamp: number;
  sequence: number;
  metadata: Record<string, any>;
  status: 'healthy' | 'degraded' | 'unhealthy';
}

export interface NodeHealth {
  nodeId: string;
  lastHeartbeat: number;
  missedCount: number;
  status: 'online' | 'suspect' | 'offline';
  healthScore: number;
  metadata: Record<string, any>;
}

export class HeartbeatSender extends EventEmitter {
  private sequence = 0;
  private intervalId: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor(private config: HeartbeatConfig) {
    super();
  }

  /**
   * Start sending heartbeats
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.emit('started', { nodeId: this.config.nodeId });

    this.intervalId = setInterval(() => {
      this.sendHeartbeat();
    }, this.config.interval);
  }

  /**
   * Stop sending heartbeats
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isRunning = false;
    this.emit('stopped', { nodeId: this.config.nodeId });
  }

  /**
   * Send a heartbeat message
   */
  private sendHeartbeat(): void {
    const heartbeat: HeartbeatMessage = {
      nodeId: this.config.nodeId,
      timestamp: Date.now(),
      sequence: this.sequence++,
      metadata: this.config.metadata || {},
      status: this.calculateStatus(),
    };

    this.emit('heartbeat', heartbeat);
  }

  /**
   * Calculate current status based on system health
   */
  private calculateStatus(): HeartbeatMessage['status'] {
    // Override this method to implement custom health checks
    const memUsage = process.memoryUsage();
    const heapUsedPercent = memUsage.heapUsed / memUsage.heapTotal;

    if (heapUsedPercent > 0.9) {
      return 'unhealthy';
    } else if (heapUsedPercent > 0.75) {
      return 'degraded';
    }
    return 'healthy';
  }

  /**
   * Update metadata dynamically
   */
  updateMetadata(metadata: Record<string, any>): void {
    this.config.metadata = { ...this.config.metadata, ...metadata };
  }
}

export class HeartbeatMonitor extends EventEmitter {
  private nodes: Map<string, NodeHealth> = new Map();
  private checkIntervalId: NodeJS.Timeout | null = null;

  constructor(
    private config: {
      checkInterval: number;
      missedThreshold: number;
    }
  ) {
    super();
  }

  /**
   * Start monitoring
   */
  start(): void {
    this.checkIntervalId = setInterval(() => {
      this.checkNodes();
    }, this.config.checkInterval);
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
    }
  }

  /**
   * Receive a heartbeat from a node
   */
  receiveHeartbeat(heartbeat: HeartbeatMessage): void {
    const existing = this.nodes.get(heartbeat.nodeId);

    if (existing) {
      // Update existing node
      existing.lastHeartbeat = heartbeat.timestamp;
      existing.missedCount = 0;
      existing.status = 'online';
      existing.metadata = heartbeat.metadata;
      
      // Calculate health score
      existing.healthScore = this.calculateHealthScore(existing, heartbeat);
    } else {
      // Register new node
      this.nodes.set(heartbeat.nodeId, {
        nodeId: heartbeat.nodeId,
        lastHeartbeat: heartbeat.timestamp,
        missedCount: 0,
        status: 'online',
        healthScore: 100,
        metadata: heartbeat.metadata,
      });

      this.emit('node:registered', { nodeId: heartbeat.nodeId });
    }

    this.emit('heartbeat:received', heartbeat);
  }

  /**
   * Check all nodes for timeouts
   */
  private checkNodes(): void {
    const now = Date.now();

    for (const [nodeId, node] of this.nodes) {
      const timeSinceLastHeartbeat = now - node.lastHeartbeat;

      if (timeSinceLastHeartbeat > this.config.missedThreshold) {
        node.missedCount++;

        if (node.missedCount >= 3) {
          node.status = 'offline';
          this.emit('node:offline', { nodeId, missedCount: node.missedCount });
        } else if (node.missedCount >= 1) {
          node.status = 'suspect';
          this.emit('node:suspect', { nodeId, missedCount: node.missedCount });
        }

        node.healthScore = Math.max(0, node.healthScore - 20);
      }
    }
  }

  /**
   * Calculate health score based on various factors
   */
  private calculateHealthScore(
    node: NodeHealth,
    heartbeat: HeartbeatMessage
  ): number {
    let score = 100;

    // Deduct for missed heartbeats
    score -= node.missedCount * 10;

    // Deduct for degraded status
    if (heartbeat.status === 'degraded') {
      score -= 15;
    } else if (heartbeat.status === 'unhealthy') {
      score -= 30;
    }

    return Math.max(0, score);
  }

  /**
   * Get node health
   */
  getNodeHealth(nodeId: string): NodeHealth | undefined {
    return this.nodes.get(nodeId);
  }

  /**
   * Get all node healths
   */
  getAllNodes(): NodeHealth[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get healthy nodes only
   */
  getHealthyNodes(): NodeHealth[] {
    return this.getAllNodes().filter(n => n.status === 'online');
  }

  /**
   * Remove a node
   */
  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.emit('node:removed', { nodeId });
  }

  /**
   * Get system overview
   */
  getOverview(): {
    total: number;
    online: number;
    suspect: number;
    offline: number;
    averageHealthScore: number;
  } {
    const nodes = this.getAllNodes();
    const total = nodes.length;
    const online = nodes.filter(n => n.status === 'online').length;
    const suspect = nodes.filter(n => n.status === 'suspect').length;
    const offline = nodes.filter(n => n.status === 'offline').length;
    const averageHealthScore = total > 0
      ? nodes.reduce((sum, n) => sum + n.healthScore, 0) / total
      : 0;

    return { total, online, suspect, offline, averageHealthScore };
  }
}
```

---

## 10. Scheduling Dashboard and Management

### 10.1 Dashboard Architecture

```typescript
// dashboard-server.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { CronEngine } from './cron-engine';
import { JobQueueManager } from './job-queue';
import { PersistenceManager } from './persistence';

export class SchedulingDashboard {
  private app: express.Application;
  private server: ReturnType<typeof createServer>;
  private io: Server;

  constructor(
    private cronEngine: CronEngine,
    private jobQueue: JobQueueManager,
    private persistence: PersistenceManager,
    private port: number = 3000
  ) {
    this.app = express();
    this.server = createServer(this.app);
    this.io = new Server(this.server);
    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
    this.setupEventListeners();
  }

  private setupMiddleware(): void {
    this.app.use(express.json());
    this.app.use(express.static('public'));
  }

  private setupRoutes(): void {
    // Get all jobs
    this.app.get('/api/jobs', (req, res) => {
      const jobs = this.cronEngine.getAllJobs();
      res.json(jobs);
    });

    // Get job by ID
    this.app.get('/api/jobs/:id', (req, res) => {
      const job = this.cronEngine.getJob(req.params.id);
      if (job) {
        res.json(job);
      } else {
        res.status(404).json({ error: 'Job not found' });
      }
    });

    // Start a job
    this.app.post('/api/jobs/:id/start', (req, res) => {
      const success = this.cronEngine.start(req.params.id);
      res.json({ success });
    });

    // Stop a job
    this.app.post('/api/jobs/:id/stop', (req, res) => {
      const success = this.cronEngine.stop(req.params.id);
      res.json({ success });
    });

    // Run job immediately
    this.app.post('/api/jobs/:id/run', async (req, res) => {
      const success = await this.cronEngine.runNow(req.params.id);
      res.json({ success });
    });

    // Get job execution history
    this.app.get('/api/jobs/:id/history', async (req, res) => {
      const history = await this.persistence.getExecutionHistory(
        req.params.id,
        parseInt(req.query.limit as string) || 100
      );
      res.json(history);
    });

    // Get system health
    this.app.get('/api/health', (req, res) => {
      const health = this.cronEngine.getHealth();
      res.json(health);
    });

    // Get queue status
    this.app.get('/api/queues/:name/status', async (req, res) => {
      try {
        const status = await this.jobQueue.getQueueStatus(req.params.name);
        res.json(status);
      } catch (error) {
        res.status(404).json({ error: 'Queue not found' });
      }
    });

    // Pause queue
    this.app.post('/api/queues/:name/pause', async (req, res) => {
      await this.jobQueue.pauseQueue(req.params.name);
      res.json({ success: true });
    });

    // Resume queue
    this.app.post('/api/queues/:name/resume', async (req, res) => {
      await this.jobQueue.resumeQueue(req.params.name);
      res.json({ success: true });
    });

    // Clean queue
    this.app.post('/api/queues/:name/clean', async (req, res) => {
      await this.jobQueue.cleanQueue(req.params.name);
      res.json({ success: true });
    });
  }

  private setupWebSocket(): void {
    this.io.on('connection', (socket) => {
      console.log('Dashboard client connected');

      // Send initial state
      socket.emit('jobs:all', this.cronEngine.getAllJobs());
      socket.emit('health', this.cronEngine.getHealth());

      socket.on('disconnect', () => {
        console.log('Dashboard client disconnected');
      });
    });
  }

  private setupEventListeners(): void {
    // Forward cron engine events to WebSocket
    this.cronEngine.on('job:started', (data) => {
      this.io.emit('job:started', data);
    });

    this.cronEngine.on('job:completed', (data) => {
      this.io.emit('job:completed', data);
    });

    this.cronEngine.on('job:failed', (data) => {
      this.io.emit('job:failed', data);
    });

    this.cronEngine.on('job:skipped', (data) => {
      this.io.emit('job:skipped', data);
    });
  }

  /**
   * Start the dashboard server
   */
  start(): void {
    this.server.listen(this.port, () => {
      console.log(`Dashboard server running on port ${this.port}`);
    });
  }

  /**
   * Stop the dashboard server
   */
  stop(): void {
    this.server.close();
  }
}
```

### 10.2 Dashboard UI (HTML/CSS/JS)

```html
<!-- public/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenClaw Scheduler Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
    }
    
    .header {
      background: #1e293b;
      padding: 1rem 2rem;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .header h1 {
      font-size: 1.5rem;
      color: #60a5fa;
    }
    
    .health-indicator {
      display: flex;
      gap: 1rem;
      align-items: center;
    }
    
    .health-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.875rem;
      font-weight: 500;
    }
    
    .health-badge.healthy { background: #10b981; color: #064e3b; }
    .health-badge.warning { background: #f59e0b; color: #78350f; }
    .health-badge.critical { background: #ef4444; color: #7f1d1d; }
    
    .container {
      padding: 2rem;
      max-width: 1400px;
      margin: 0 auto;
    }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    
    .stat-card {
      background: #1e293b;
      padding: 1.5rem;
      border-radius: 0.5rem;
      border: 1px solid #334155;
    }
    
    .stat-card h3 {
      font-size: 0.875rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }
    
    .stat-card .value {
      font-size: 2rem;
      font-weight: 700;
      color: #f8fafc;
    }
    
    .jobs-table {
      background: #1e293b;
      border-radius: 0.5rem;
      border: 1px solid #334155;
      overflow: hidden;
    }
    
    .jobs-table-header {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1.5fr;
      padding: 1rem;
      background: #334155;
      font-weight: 600;
      font-size: 0.875rem;
    }
    
    .job-row {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1.5fr;
      padding: 1rem;
      border-bottom: 1px solid #334155;
      align-items: center;
    }
    
    .job-row:hover {
      background: #334155;
    }
    
    .job-row:last-child {
      border-bottom: none;
    }
    
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .status-badge.running {
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
    }
    
    .status-badge.idle {
      background: rgba(148, 163, 184, 0.2);
      color: #94a3b8;
    }
    
    .status-badge.error {
      background: rgba(239, 68, 68, 0.2);
      color: #ef4444;
    }
    
    .status-badge.paused {
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
    }
    
    .actions {
      display: flex;
      gap: 0.5rem;
    }
    
    .btn {
      padding: 0.5rem 1rem;
      border: none;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .btn-primary {
      background: #3b82f6;
      color: white;
    }
    
    .btn-primary:hover {
      background: #2563eb;
    }
    
    .btn-danger {
      background: #ef4444;
      color: white;
    }
    
    .btn-danger:hover {
      background: #dc2626;
    }
    
    .btn-secondary {
      background: #475569;
      color: white;
    }
    
    .btn-secondary:hover {
      background: #334155;
    }
    
    .pulse {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>OpenClaw Scheduler Dashboard</h1>
    <div class="health-indicator">
      <span>System Status:</span>
      <span class="health-badge healthy" id="system-health">Healthy</span>
    </div>
  </div>
  
  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <h3>Total Jobs</h3>
        <div class="value" id="total-jobs">0</div>
      </div>
      <div class="stat-card">
        <h3>Running</h3>
        <div class="value" id="running-jobs">0</div>
      </div>
      <div class="stat-card">
        <h3>Failed</h3>
        <div class="value" id="failed-jobs">0</div>
      </div>
      <div class="stat-card">
        <h3>Idle</h3>
        <div class="value" id="idle-jobs">0</div>
      </div>
    </div>
    
    <div class="jobs-table">
      <div class="jobs-table-header">
        <div>Job Name</div>
        <div>Schedule</div>
        <div>Status</div>
        <div>Last Run</div>
        <div>Next Run</div>
        <div>Actions</div>
      </div>
      <div id="jobs-list"></div>
    </div>
  </div>
  
  <script src="/socket.io/socket.io.js"></script>
  <script>
    const socket = io();
    
    // Update stats
    function updateStats(health) {
      document.getElementById('total-jobs').textContent = health.totalJobs;
      document.getElementById('running-jobs').textContent = health.runningJobs;
      document.getElementById('failed-jobs').textContent = health.failedJobs;
      document.getElementById('idle-jobs').textContent = health.idleJobs;
      
      const healthBadge = document.getElementById('system-health');
      if (health.failedJobs > 0) {
        healthBadge.className = 'health-badge warning';
        healthBadge.textContent = 'Warning';
      } else if (health.runningJobs === 0 && health.totalJobs > 0) {
        healthBadge.className = 'health-badge critical';
        healthBadge.textContent = 'Critical';
      } else {
        healthBadge.className = 'health-badge healthy';
        healthBadge.textContent = 'Healthy';
      }
    }
    
    // Format date
    function formatDate(date) {
      if (!date) return 'Never';
      return new Date(date).toLocaleString();
    }
    
    // Render jobs list
    function renderJobs(jobs) {
      const container = document.getElementById('jobs-list');
      container.innerHTML = jobs.map(job => `
        <div class="job-row">
          <div>
            <strong>${job.name}</strong>
            <div style="font-size: 0.75rem; color: #94a3b8;">${job.id}</div>
          </div>
          <div><code>${job.schedule}</code></div>
          <div>
            <span class="status-badge ${job.status.state}">
              ${job.status.state === 'running' ? '<span class="pulse"></span>' : ''}
              ${job.status.state}
            </span>
          </div>
          <div>${formatDate(job.status.lastRun)}</div>
          <div>${formatDate(job.status.nextRun)}</div>
          <div class="actions">
            <button class="btn btn-primary" onclick="runJob('${job.id}')">Run</button>
            ${job.options.enabled 
              ? `<button class="btn btn-danger" onclick="stopJob('${job.id}')">Stop</button>`
              : `<button class="btn btn-secondary" onclick="startJob('${job.id}')">Start</button>`
            }
          </div>
        </div>
      `).join('');
    }
    
    // Job actions
    async function runJob(id) {
      await fetch(`/api/jobs/${id}/run`, { method: 'POST' });
    }
    
    async function startJob(id) {
      await fetch(`/api/jobs/${id}/start`, { method: 'POST' });
    }
    
    async function stopJob(id) {
      await fetch(`/api/jobs/${id}/stop`, { method: 'POST' });
    }
    
    // Socket events
    socket.on('jobs:all', renderJobs);
    socket.on('health', updateStats);
    socket.on('job:started', () => fetchJobs());
    socket.on('job:completed', () => fetchJobs());
    socket.on('job:failed', () => fetchJobs());
    
    // Fetch jobs
    async function fetchJobs() {
      const response = await fetch('/api/jobs');
      const jobs = await response.json();
      renderJobs(jobs);
    }
    
    // Initial load
    fetchJobs();
    fetch('/api/health').then(r => r.json()).then(updateStats);
  </script>
</body>
</html>
```

---

## 11. Implementation Code Examples

### 11.1 Complete System Integration

```typescript
// scheduler-system.ts
import { CronEngine } from './cron-engine';
import { JobQueueManager } from './job-queue';
import { PersistenceManager } from './persistence';
import { HeartbeatSender, HeartbeatMonitor } from './heartbeat';
import { WindowsTaskScheduler } from './windows-task-scheduler';
import { SchedulingDashboard } from './dashboard-server';
import { OverlapPrevention } from './overlap-prevention';
import { RetryMechanism, CircuitBreaker } from './retry-mechanism';

export interface SchedulerSystemConfig {
  dataDir: string;
  redis: {
    host: string;
    port: number;
    password?: string;
  };
  heartbeat: {
    interval: number;
    timeout: number;
  };
  dashboard: {
    enabled: boolean;
    port: number;
  };
  windowsTask: {
    enabled: boolean;
    taskName: string;
  };
}

export class SchedulerSystem {
  public cronEngine: CronEngine;
  public jobQueue: JobQueueManager;
  public persistence: PersistenceManager;
  public heartbeatSender: HeartbeatSender;
  public heartbeatMonitor: HeartbeatMonitor;
  public windowsTask: WindowsTaskScheduler | null = null;
  public dashboard: SchedulingDashboard | null = null;
  public overlapPrevention: OverlapPrevention;
  public retryMechanism: RetryMechanism;

  private isRunning = false;

  constructor(private config: SchedulerSystemConfig) {
    // Initialize components
    this.cronEngine = new CronEngine();
    this.jobQueue = new JobQueueManager(config.redis);
    this.persistence = new PersistenceManager(config.dataDir);
    this.heartbeatSender = new HeartbeatSender({
      interval: config.heartbeat.interval,
      timeout: config.heartbeat.timeout,
      missedThreshold: 3,
      nodeId: 'openclaw-agent',
      metadata: {
        version: '1.0.0',
        platform: 'win32',
      },
    });
    this.heartbeatMonitor = new HeartbeatMonitor({
      checkInterval: 30000,
      missedThreshold: 3,
    });
    this.overlapPrevention = new OverlapPrevention({
      strategy: 'skip',
      timeoutMs: 300000,
    });
    this.retryMechanism = new RetryMechanism({
      maxAttempts: 3,
      strategy: 'exponential',
      baseDelay: 1000,
      maxDelay: 30000,
    });

    // Windows Task Scheduler
    if (config.windowsTask.enabled) {
      this.windowsTask = new WindowsTaskScheduler({
        taskName: config.windowsTask.taskName,
        nodePath: process.execPath,
        scriptPath: process.argv[1],
        workingDirectory: process.cwd(),
      });
    }

    // Setup event handlers
    this.setupEventHandlers();
  }

  /**
   * Initialize the system
   */
  async initialize(): Promise<void> {
    // Initialize persistence
    await this.persistence.initialize();

    // Load persisted jobs
    const persistedJobs = await this.persistence.loadJobs();
    for (const job of persistedJobs) {
      // Restore jobs
      console.log(`Restoring job: ${job.name}`);
    }

    // Create queues
    this.jobQueue.createQueue({ name: 'agent-loops' });
    this.jobQueue.createQueue({ name: 'maintenance' });
    this.jobQueue.createQueue({ name: 'notifications' });

    console.log('Scheduler system initialized');
  }

  /**
   * Start the system
   */
  async start(): Promise<void> {
    if (this.isRunning) return;

    this.isRunning = true;

    // Start cron engine
    this.cronEngine.startAll();

    // Start heartbeat
    this.heartbeatSender.start();
    this.heartbeatMonitor.start();

    // Start dashboard
    if (this.config.dashboard.enabled) {
      this.dashboard = new SchedulingDashboard(
        this.cronEngine,
        this.jobQueue,
        this.persistence,
        this.config.dashboard.port
      );
      this.dashboard.start();
    }

    // Ensure Windows task is running
    if (this.windowsTask) {
      await this.windowsTask.ensureRunning();
    }

    console.log('Scheduler system started');
  }

  /**
   * Stop the system gracefully
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    this.isRunning = false;

    // Stop cron engine
    this.cronEngine.stopAll();

    // Stop heartbeat
    this.heartbeatSender.stop();
    this.heartbeatMonitor.stop();

    // Stop dashboard
    this.dashboard?.stop();

    // Close job queue
    await this.jobQueue.close();

    // Close persistence
    await this.persistence.close();

    console.log('Scheduler system stopped');
  }

  /**
   * Register an agent loop
   */
  registerAgentLoop(
    id: string,
    name: string,
    schedule: string,
    handler: () => Promise<void>,
    options: {
      preventOverlap?: boolean;
      retryAttempts?: number;
      tags?: string[];
    } = {}
  ): void {
    const wrappedHandler = async () => {
      // Apply overlap prevention
      await this.overlapPrevention.execute(id, async () => {
        // Apply retry mechanism
        const result = await this.retryMechanism.execute(handler);
        
        if (!result.success) {
          throw result.error;
        }
      });
    };

    this.cronEngine.register(id, name, schedule, wrappedHandler, {
      preventOverlap: options.preventOverlap ?? true,
      retryAttempts: options.retryAttempts ?? 3,
      tags: ['agent-loop', ...(options.tags || [])],
    });

    // Persist job
    this.persistence.saveJob({
      id,
      name,
      schedule,
      options: JSON.stringify(options),
      status: JSON.stringify({ state: 'idle' }),
      metrics: JSON.stringify({}),
      lastUpdated: Date.now(),
      createdAt: Date.now(),
    });
  }

  /**
   * Setup event handlers
   */
  private setupEventHandlers(): void {
    // Cron engine events
    this.cronEngine.on('job:completed', async (result) => {
      await this.persistence.logExecution({
        id: `${result.jobId}-${Date.now()}`,
        jobId: result.jobId,
        startTime: result.startTime.getTime(),
        endTime: result.endTime.getTime(),
        duration: result.duration,
        success: true,
      });
    });

    this.cronEngine.on('job:failed', async (result) => {
      await this.persistence.logExecution({
        id: `${result.jobId}-${Date.now()}`,
        jobId: result.jobId,
        startTime: result.startTime.getTime(),
        endTime: result.endTime.getTime(),
        duration: result.duration,
        success: false,
        error: result.error?.message,
      });
    });

    // Heartbeat events
    this.heartbeatSender.on('heartbeat', (heartbeat) => {
      this.heartbeatMonitor.receiveHeartbeat(heartbeat);
    });

    // Graceful shutdown
    process.on('SIGINT', () => this.stop());
    process.on('SIGTERM', () => this.stop());
  }
}
```

### 11.2 Agent Loop Registration Example

```typescript
// agent-loops.ts
import { SchedulerSystem } from './scheduler-system';

export function registerAgentLoops(scheduler: SchedulerSystem): void {
  // Loop 1: Heartbeat Monitor - Every 30 seconds
  scheduler.registerAgentLoop(
    'loop-heartbeat',
    'Heartbeat Monitor',
    '*/30 * * * * *',
    async () => {
      console.log('Sending heartbeat...');
      // Send heartbeat to monitoring service
    },
    { tags: ['critical', 'system'] }
  );

  // Loop 2: Gmail Sync - Every 2 minutes
  scheduler.registerAgentLoop(
    'loop-gmail-sync',
    'Gmail Synchronization',
    '*/2 * * * *',
    async () => {
      console.log('Syncing Gmail...');
      // Sync Gmail inbox
    },
    { tags: ['communication'] }
  );

  // Loop 3: Browser Health Check - Every minute
  scheduler.registerAgentLoop(
    'loop-browser-health',
    'Browser Health Check',
    '* * * * *',
    async () => {
      console.log('Checking browser health...');
      // Check browser instance health
    },
    { tags: ['system'] }
  );

  // Loop 4: TTS/STT Queue Processor - Every 5 seconds
  scheduler.registerAgentLoop(
    'loop-voice-queue',
    'Voice Queue Processor',
    '*/5 * * * * *',
    async () => {
      console.log('Processing voice queue...');
      // Process TTS/STT queue
    },
    { tags: ['voice'] }
  );

  // Loop 5: Twilio Status Check - Every minute
  scheduler.registerAgentLoop(
    'loop-twilio-status',
    'Twilio Status Check',
    '* * * * *',
    async () => {
      console.log('Checking Twilio status...');
      // Check Twilio connection and messages
    },
    { tags: ['communication'] }
  );

  // Loop 6: System Resource Monitor - Every 30 seconds
  scheduler.registerAgentLoop(
    'loop-resource-monitor',
    'Resource Monitor',
    '*/30 * * * * *',
    async () => {
      const usage = process.memoryUsage();
      console.log('Memory usage:', {
        heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
      });
    },
    { tags: ['system', 'monitoring'] }
  );

  // Loop 7: Identity Sync - Every 5 minutes
  scheduler.registerAgentLoop(
    'loop-identity-sync',
    'Identity Synchronization',
    '*/5 * * * *',
    async () => {
      console.log('Syncing identity state...');
      // Sync identity and user preferences
    },
    { tags: ['identity'] }
  );

  // Loop 8: Soul State Update - Every minute
  scheduler.registerAgentLoop(
    'loop-soul-update',
    'Soul State Update',
    '* * * * *',
    async () => {
      console.log('Updating soul state...');
      // Update emotional/behavioral state
    },
    { tags: ['soul'] }
  );

  // Loop 9: Context Cleanup - Every 10 minutes
  scheduler.registerAgentLoop(
    'loop-context-cleanup',
    'Context Cleanup',
    '*/10 * * * *',
    async () => {
      console.log('Cleaning up old context...');
      // Remove old conversation context
    },
    { tags: ['maintenance'] }
  );

  // Loop 10: Notification Digest - Every 15 minutes
  scheduler.registerAgentLoop(
    'loop-notification-digest',
    'Notification Digest',
    '*/15 * * * *',
    async () => {
      console.log('Sending notification digest...');
      // Send accumulated notifications
    },
    { tags: ['notifications'] }
  );

  // Loop 11: Log Rotation - Every hour
  scheduler.registerAgentLoop(
    'loop-log-rotation',
    'Log Rotation',
    '0 * * * *',
    async () => {
      console.log('Rotating logs...');
      // Rotate and compress logs
    },
    { tags: ['maintenance'] }
  );

  // Loop 12: Backup Check - Every 30 minutes
  scheduler.registerAgentLoop(
    'loop-backup-check',
    'Backup Check',
    '*/30 * * * *',
    async () => {
      console.log('Checking backup status...');
      // Verify backup integrity
    },
    { tags: ['maintenance', 'backup'] }
  );

  // Loop 13: API Rate Limit Check - Every minute
  scheduler.registerAgentLoop(
    'loop-rate-limit',
    'API Rate Limit Check',
    '* * * * *',
    async () => {
      console.log('Checking API rate limits...');
      // Monitor and adjust API call rates
    },
    { tags: ['system'] }
  );

  // Loop 14: User Presence Check - Every 5 minutes
  scheduler.registerAgentLoop(
    'loop-presence-check',
    'User Presence Check',
    '*/5 * * * *',
    async () => {
      console.log('Checking user presence...');
      // Detect user activity/inactivity
    },
    { tags: ['user'] }
  );

  // Loop 15: Daily Maintenance - 2 AM daily
  scheduler.registerAgentLoop(
    'loop-daily-maintenance',
    'Daily Maintenance',
    '0 2 * * *',
    async () => {
      console.log('Running daily maintenance...');
      // Comprehensive daily maintenance tasks
    },
    { tags: ['maintenance', 'daily'] }
  );
}
```

---

## 12. Deployment and Operations

### 12.1 Windows Service Installation (NSSM)

```powershell
# install-service.ps1
#Requires -RunAsAdministrator

$ServiceName = "OpenClawAgent"
$NodePath = "C:\Program Files\nodejs\node.exe"
$AppPath = "C:\OpenClaw"
$ScriptPath = "$AppPath\dist\index.js"
$NssmPath = "C:\nssm\win64\nssm.exe"

# Download NSSM if not exists
if (-not (Test-Path $NssmPath)) {
    Write-Host "Downloading NSSM..."
    $NssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $NssmZip = "$env:TEMP\nssm.zip"
    Invoke-WebRequest -Uri $NssmUrl -OutFile $NssmZip
    Expand-Archive -Path $NssmZip -DestinationPath "C:\nssm" -Force
}

# Install service
& $NssmPath install $ServiceName $NodePath
& $NssmPath set $ServiceName Application $NodePath
& $NssmPath set $ServiceName AppParameters $ScriptPath
& $NssmPath set $ServiceName AppDirectory $AppPath
& $NssmPath set $ServiceName AppEnvironmentExtra "NODE_ENV=production"
& $NssmPath set $ServiceName DisplayName "OpenClaw AI Agent"
& $NssmPath set $ServiceName Description "24/7 AI Agent System with scheduled task automation"
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppStdout "$AppPath\logs\service.log"
& $NssmPath set $ServiceName AppStderr "$AppPath\logs\service-error.log"
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateOnline 1
& $NssmPath set $ServiceName AppRotateSeconds 86400

# Start service
Start-Service $ServiceName

Write-Host "Service installed and started successfully!"
Write-Host "View logs: Get-Content $AppPath\logs\service.log -Tail 50 -Wait"
```

### 12.2 Package.json Scripts

```json
{
  "name": "openclaw-scheduler",
  "version": "1.0.0",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "service:install": "powershell -ExecutionPolicy Bypass -File ./scripts/install-service.ps1",
    "service:uninstall": "nssm remove OpenClawAgent confirm",
    "service:start": "net start OpenClawAgent",
    "service:stop": "net stop OpenClawAgent",
    "service:restart": "net stop OpenClawAgent && net start OpenClawAgent",
    "service:status": "sc query OpenClawAgent",
    "task:create": "powershell -ExecutionPolicy Bypass -File ./scripts/TaskSchedulerManager.ps1 -Action Create",
    "task:delete": "powershell -ExecutionPolicy Bypass -File ./scripts/TaskSchedulerManager.ps1 -Action Delete",
    "task:status": "powershell -ExecutionPolicy Bypass -File ./scripts/TaskSchedulerManager.ps1 -Action Get",
    "db:migrate": "sqlite3 ./data/scheduler.db < ./scripts/migrations.sql",
    "db:backup": "node scripts/backup-db.js",
    "test": "jest",
    "lint": "eslint src/**/*.ts"
  },
  "dependencies": {
    "async-mutex": "^0.4.0",
    "bullmq": "^4.0.0",
    "cron": "^2.4.0",
    "express": "^4.18.0",
    "ioredis": "^5.3.0",
    "node-cron": "^3.0.0",
    "node-schedule": "^2.1.0",
    "socket.io": "^4.6.0",
    "sqlite3": "^5.1.0",
    "sqlite": "^4.2.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/node-cron": "^3.0.0",
    "@types/express": "^4.17.0",
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0"
  }
}
```

### 12.3 Environment Configuration

```ini
# .env
# Scheduler Configuration
SCHEDULER_DATA_DIR=./data
SCHEDULER_TIMEZONE=America/New_York
SCHEDULER_LOG_LEVEL=info

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Heartbeat Configuration
HEARTBEAT_INTERVAL=30000
HEARTBEAT_TIMEOUT=60000

# Dashboard Configuration
DASHBOARD_ENABLED=true
DASHBOARD_PORT=3000
DASHBOARD_AUTH_TOKEN=your-secure-token

# Windows Task Scheduler
WINDOWS_TASK_ENABLED=true
WINDOWS_TASK_NAME=OpenClawAgent

# Agent Loop Configuration
AGENT_LOOP_CONCURRENCY=5
AGENT_LOOP_TIMEOUT=300000
AGENT_LOOP_RETRY_ATTEMPTS=3

# Overlap Prevention
OVERLAP_STRATEGY=skip
OVERLAP_TIMEOUT=300000

# Retry Configuration
RETRY_STRATEGY=exponential
RETRY_BASE_DELAY=1000
RETRY_MAX_DELAY=30000
RETRY_MAX_ATTEMPTS=3
```

---

## Summary

This technical specification provides a complete architecture for implementing a robust cron job and scheduled task system for a 24/7 Windows 10 AI agent system. The design includes:

1. **Hybrid Scheduling**: Combines Windows Task Scheduler, Node.js cron libraries, and BullMQ job queues
2. **Persistence**: SQLite-based state storage with automatic backups
3. **Fault Tolerance**: Retry mechanisms, circuit breakers, and overlap prevention
4. **Monitoring**: Heartbeat system and real-time dashboard
5. **15 Agent Loops**: Pre-configured schedules for various system functions

The system is designed to be:
- **Resilient**: Survives crashes and restarts
- **Observable**: Full visibility into job execution
- **Scalable**: Can handle increasing workloads
- **Maintainable**: Clear separation of concerns

---

*Document Version: 1.0*
*Last Updated: 2025*
*Author: Systems Infrastructure Team*
