# OpenClaw_AI_Agent_TaskScheduler.ps1
# Complete PowerShell module for Task Scheduler integration
# OpenClaw-inspired AI Agent System for Windows 10/11

#Requires -RunAsAdministrator
#Requires -Version 5.1

<#
.SYNOPSIS
    Windows Task Scheduler integration for OpenClaw AI Agent
.DESCRIPTION
    Provides comprehensive Task Scheduler management for the OpenClaw AI Agent system
    including task creation, monitoring, heartbeat management, and cron translation.
    
    This module implements:
    - Task Scheduler COM API integration
    - 15 hardcoded agentic loop tasks
    - Multi-frequency heartbeat system (30s, 5m, 15m)
    - Cron expression to Task Scheduler translation
    - Task monitoring and health checking
    - Soul/identity persistence
    
.PARAMETER AgentId
    Unique identifier for the AI Agent instance
    
.PARAMETER ConfigPath
    Path to agent configuration file
    
.EXAMPLE
    Import-Module .\OpenClaw_AI_Agent_TaskScheduler.ps1
    Install-OpenClawTaskScheduler -AgentId "agent_001"
    
.EXAMPLE
    $scheduler = Connect-TaskScheduler
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Primary"
    
.NOTES
    Version: 1.0.0
    Author: OpenClaw Framework
    Platform: Windows 10/11
    API: Task Scheduler 2.0
#>

# Enforce strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

#region Constants
$SCRIPT:RootFolder = "\OpenClaw"
$SCRIPT:CoreFolder = "\OpenClaw\Core"
$SCRIPT:AgentLoopsFolder = "\OpenClaw\AgentLoops"
$SCRIPT:TriggersFolder = "\OpenClaw\Triggers"
$SCRIPT:MaintenanceFolder = "\OpenClaw\Maintenance"
$SCRIPT:ConfigPath = "C:\OpenClaw\config.json"
$SCRIPT:LogPath = "C:\OpenClaw\logs"
$SCRIPT:Version = "1.0.0"

# Task State Enumeration
$SCRIPT:TaskState = @{
    Unknown = 0
    Disabled = 1
    Queued = 2
    Ready = 3
    Running = 4
}

# Trigger Type Enumeration
$SCRIPT:TriggerType = @{
    Event = 0
    Time = 1
    Daily = 2
    Weekly = 3
    Monthly = 4
    MonthlyDOW = 5
    Idle = 6
    Registration = 7
    Boot = 8
    Logon = 9
    SessionStateChange = 11
}

# Action Type Enumeration
$SCRIPT:ActionType = @{
    Execute = 0
    ComHandler = 5
    SendEmail = 6
    ShowMessage = 7
}

# Logon Type Enumeration
$SCRIPT:LogonType = @{
    None = 0
    Password = 1
    S4U = 2
    InteractiveToken = 3
    Group = 4
    ServiceAccount = 5
    InteractiveTokenOrPassword = 6
}

# Run Level Enumeration
$SCRIPT:RunLevel = @{
    LUA = 0      # Least Privilege User Account
    Highest = 1  # Highest available privileges
}
#endregion

#region Logging
function Write-OpenClawLog {
    param(
        [string]$Message,
        [ValidateSet("Info", "Warning", "Error", "Success")]
        [string]$Level = "Info"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Ensure log directory exists
    if (-not (Test-Path $SCRIPT:LogPath)) {
        New-Item -ItemType Directory -Path $SCRIPT:LogPath -Force | Out-Null
    }
    
    $logFile = Join-Path $SCRIPT:LogPath "openclaw_scheduler.log"
    Add-Content -Path $logFile -Value $logEntry
    
    # Also write to console with color
    switch ($Level) {
        "Info"    { Write-Host $logEntry -ForegroundColor Gray }
        "Warning" { Write-Host $logEntry -ForegroundColor Yellow }
        "Error"   { Write-Host $logEntry -ForegroundColor Red }
        "Success" { Write-Host $logEntry -ForegroundColor Green }
    }
}
#endregion

#region Task Scheduler Connection
function Connect-TaskScheduler {
    <#
    .SYNOPSIS
        Connect to Windows Task Scheduler service via COM API
    .DESCRIPTION
        Establishes connection to the Task Scheduler service using the Schedule.Service COM object.
        This is the gateway to all Task Scheduler operations.
    .OUTPUTS
        Schedule.Service COM object or $null on failure
    .EXAMPLE
        $scheduler = Connect-TaskScheduler
        if ($scheduler) { Write-Host "Connected successfully" }
    #>
    [CmdletBinding()]
    param()
    
    try {
        Write-OpenClawLog "Connecting to Task Scheduler service..." -Level "Info"
        
        $scheduler = New-Object -ComObject Schedule.Service
        $scheduler.Connect()
        
        Write-OpenClawLog "Successfully connected to Task Scheduler" -Level "Success"
        return $scheduler
    }
    catch {
        Write-OpenClawLog "Failed to connect to Task Scheduler: $_" -Level "Error"
        return $null
    }
}
#endregion

#region Folder Management
function Initialize-OpenClawFolderStructure {
    <#
    .SYNOPSIS
        Create OpenClaw folder structure in Task Scheduler
    .DESCRIPTION
        Creates the complete folder hierarchy for OpenClaw AI Agent tasks:
        - \OpenClaw (root)
        - \OpenClaw\Core (core agent tasks)
        - \OpenClaw\AgentLoops (15 agentic loops)
        - \OpenClaw\Triggers (event-based triggers)
        - \OpenClaw\Maintenance (maintenance tasks)
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        $Scheduler
    )
    
    $folders = @(
        @{ Path = $SCRIPT:RootFolder; Parent = "\"; Name = "OpenClaw" },
        @{ Path = $SCRIPT:CoreFolder; Parent = $SCRIPT:RootFolder; Name = "Core" },
        @{ Path = $SCRIPT:AgentLoopsFolder; Parent = $SCRIPT:RootFolder; Name = "AgentLoops" },
        @{ Path = $SCRIPT:TriggersFolder; Parent = $SCRIPT:RootFolder; Name = "Triggers" },
        @{ Path = $SCRIPT:MaintenanceFolder; Parent = $SCRIPT:RootFolder; Name = "Maintenance" }
    )
    
    Write-OpenClawLog "Creating folder structure..." -Level "Info"
    
    foreach ($folder in $folders) {
        try {
            if ($folder.Parent -eq "\") {
                $parent = $Scheduler.GetFolder("\")
            }
            else {
                $parent = $Scheduler.GetFolder($folder.Parent)
            }
            
            try {
                $parent.CreateFolder($folder.Name) | Out-Null
                Write-OpenClawLog "Created folder: $($folder.Path)" -Level "Success"
            }
            catch {
                Write-OpenClawLog "Folder already exists: $($folder.Path)" -Level "Warning"
            }
        }
        catch {
            Write-OpenClawLog "Error creating folder $($folder.Path): $_" -Level "Error"
        }
    }
}

function Get-OpenClawFolder {
    <#
    .SYNOPSIS
        Get a specific OpenClaw folder
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER SubPath
        Subfolder path (e.g., "Core", "AgentLoops")
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [string]$SubPath = ""
    )
    
    $fullPath = if ($SubPath) { "$SCRIPT:RootFolder\$SubPath" } else { $SCRIPT:RootFolder }
    
    try {
        return $Scheduler.GetFolder($fullPath)
    }
    catch {
        Write-OpenClawLog "Folder not found: $fullPath" -Level "Error"
        return $null
    }
}
#endregion

#region Task Creation
function New-OpenClawHeartbeatTask {
    <#
    .SYNOPSIS
        Create heartbeat task for AI Agent
    .DESCRIPTION
        Creates a scheduled task that provides periodic heartbeat signals
        to ensure the AI Agent remains active and responsive.
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER Frequency
        Heartbeat frequency: Primary (30s), Secondary (5m), Tertiary (15m)
    .PARAMETER AgentId
        Unique identifier for the AI Agent
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [ValidateSet("Primary", "Secondary", "Tertiary")]
        [string]$Frequency = "Primary",
        [string]$AgentId = "default"
    )
    
    $intervals = @{
        "Primary" = "PT30S"    # 30 seconds
        "Secondary" = "PT5M"   # 5 minutes
        "Tertiary" = "PT15M"   # 15 minutes
    }
    
    $taskName = "Heartbeat_$Frequency"
    $interval = $intervals[$Frequency]
    
    Write-OpenClawLog "Creating heartbeat task: $taskName (Interval: $interval)" -Level "Info"
    
    try {
        # Create task definition
        $taskDef = $Scheduler.NewTask(0)
        
        # Registration info
        $regInfo = $taskDef.RegistrationInfo
        $regInfo.Description = "OpenClaw AI Agent Heartbeat - $Frequency"
        $regInfo.Author = "OpenClaw Framework v$($SCRIPT:Version)"
        $regInfo.Date = (Get-Date).ToString("s")
        
        # Principal - run as SYSTEM with highest privileges
        $principal = $taskDef.Principal
        $principal.UserId = "NT AUTHORITY\SYSTEM"
        $principal.LogonType = $SCRIPT:LogonType.ServiceAccount
        $principal.RunLevel = $SCRIPT:RunLevel.Highest
        
        # Settings - optimized for 24/7 operation
        $settings = $taskDef.Settings
        $settings.Enabled = $true
        $settings.Hidden = $false
        $settings.StartWhenAvailable = $true
        $settings.WakeToRun = $true
        $settings.ExecutionTimeLimit = "PT5M"
        $settings.RestartCount = 3
        $settings.RestartInterval = "PT1M"
        $settings.DisallowStartIfOnBatteries = $false
        $settings.StopIfGoingOnBatteries = $false
        $settings.MultipleInstances = 1  # Ignore new instances
        
        # Triggers
        $triggers = $taskDef.Triggers
        
        # Daily trigger with repetition
        $dailyTrigger = $triggers.Create($SCRIPT:TriggerType.Daily)
        $dailyTrigger.StartBoundary = (Get-Date).AddMinutes(1).ToString("s")
        $dailyTrigger.DaysInterval = 1
        $dailyTrigger.Repetition.Interval = $interval
        $dailyTrigger.Repetition.Duration = "P1D"  # Repeat for 1 day
        $dailyTrigger.Enabled = $true
        
        # Boot trigger - start on system startup
        $bootTrigger = $triggers.Create($SCRIPT:TriggerType.Boot)
        $bootTrigger.Delay = "PT30S"
        $bootTrigger.Enabled = $true
        
        # Logon trigger - start when user logs on
        $logonTrigger = $triggers.Create($SCRIPT:TriggerType.Logon)
        $logonTrigger.Delay = "PT30S"
        $logonTrigger.Enabled = $true
        
        # Action - execute PowerShell heartbeat script
        $actions = $taskDef.Actions
        $action = $actions.Create($SCRIPT:ActionType.Execute)
        $action.Path = "powershell.exe"
        $action.Arguments = "-ExecutionPolicy Bypass -WindowStyle Hidden -Command `"& C:\OpenClaw\heartbeat_$Frequency.ps1 -AgentId '$AgentId'`""
        $action.WorkingDirectory = "C:\OpenClaw"
        
        # Register task
        $folder = $Scheduler.GetFolder($SCRIPT:CoreFolder)
        $folder.RegisterTaskDefinition($taskName, $taskDef, 6, $null, $null, 5) | Out-Null
        
        Write-OpenClawLog "Successfully created heartbeat task: $taskName" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to create heartbeat task $taskName`: $_" -Level "Error"
        return $false
    }
}

function New-OpenClawAgentLoopTask {
    <#
    .SYNOPSIS
        Create one of the 15 hardcoded agentic loop tasks
    .DESCRIPTION
        Creates a scheduled task for a specific agentic loop that handles
        a particular aspect of the AI Agent functionality.
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER LoopNumber
        Loop number (1-15)
    .PARAMETER LoopName
        Name of the loop (e.g., "EmailProcessor", "BrowserAutomation")
    .PARAMETER TriggerType
        Type of trigger for the loop
    .PARAMETER Interval
        Execution interval for the loop
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [ValidateRange(1, 15)]
        [int]$LoopNumber,
        [Parameter(Mandatory=$true)]
        [string]$LoopName,
        [ValidateSet("Heartbeat", "Event", "Time", "Boot", "Logon", "Idle")]
        [string]$TriggerType = "Heartbeat",
        [string]$Interval = "PT5M"
    )
    
    $taskName = "Loop_{0:D2}_{1}" -f $LoopNumber, $LoopName
    
    Write-OpenClawLog "Creating agent loop task: $taskName" -Level "Info"
    
    try {
        # Create task definition
        $taskDef = $Scheduler.NewTask(0)
        
        # Registration info
        $regInfo = $taskDef.RegistrationInfo
        $regInfo.Description = "OpenClaw AI Agent Loop $LoopNumber - $LoopName"
        $regInfo.Author = "OpenClaw Framework v$($SCRIPT:Version)"
        
        # Principal
        $principal = $taskDef.Principal
        $principal.UserId = "NT AUTHORITY\SYSTEM"
        $principal.LogonType = $SCRIPT:LogonType.ServiceAccount
        $principal.RunLevel = $SCRIPT:RunLevel.Highest
        
        # Settings
        $settings = $taskDef.Settings
        $settings.Enabled = $true
        $settings.Hidden = $false
        $settings.StartWhenAvailable = $true
        $settings.WakeToRun = $false
        $settings.ExecutionTimeLimit = "PT1H"
        $settings.RestartCount = 3
        $settings.RestartInterval = "PT1M"
        
        # Triggers based on type
        $triggers = $taskDef.Triggers
        
        switch ($TriggerType) {
            "Heartbeat" {
                $trigger = $triggers.Create($SCRIPT:TriggerType.Daily)
                $trigger.StartBoundary = (Get-Date).AddMinutes(1).ToString("s")
                $trigger.DaysInterval = 1
                $trigger.Repetition.Interval = $Interval
                $trigger.Repetition.Duration = "P1D"
                $trigger.Enabled = $true
            }
            "Event" {
                # Event-based trigger
                $trigger = $triggers.Create($SCRIPT:TriggerType.Event)
                $trigger.Subscription = "<QueryList><Query><Select>*</Select></Query></QueryList>"
                $trigger.Enabled = $true
            }
            "Time" {
                # One-time trigger
                $trigger = $triggers.Create($SCRIPT:TriggerType.Time)
                $trigger.StartBoundary = (Get-Date).AddMinutes(5).ToString("s")
                $trigger.Enabled = $true
            }
            "Boot" {
                $trigger = $triggers.Create($SCRIPT:TriggerType.Boot)
                $trigger.Delay = "PT1M"
                $trigger.Enabled = $true
            }
            "Logon" {
                $trigger = $triggers.Create($SCRIPT:TriggerType.Logon)
                $trigger.Delay = "PT30S"
                $trigger.Enabled = $true
            }
            "Idle" {
                $trigger = $triggers.Create($SCRIPT:TriggerType.Idle)
                $trigger.Enabled = $true
            }
        }
        
        # Action
        $actions = $taskDef.Actions
        $action = $actions.Create($SCRIPT:ActionType.Execute)
        $action.Path = "C:\OpenClaw\agent_loop.exe"
        $action.Arguments = "--loop=$LoopNumber --name=`"$LoopName`""
        $action.WorkingDirectory = "C:\OpenClaw"
        
        # Register task
        $folder = $Scheduler.GetFolder($SCRIPT:AgentLoopsFolder)
        $folder.RegisterTaskDefinition($taskName, $taskDef, 6, $null, $null, 5) | Out-Null
        
        Write-OpenClawLog "Successfully created agent loop task: $taskName" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to create agent loop task $taskName`: $_" -Level "Error"
        return $false
    }
}

function New-OpenClawSoulMaintenanceTask {
    <#
    .SYNOPSIS
        Create soul/identity maintenance task
    .DESCRIPTION
        Creates a scheduled task that maintains the agent's soul and identity
        by periodically updating the soul signature and checking identity integrity.
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER AgentId
        Unique identifier for the AI Agent
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [string]$AgentId = "default"
    )
    
    $taskName = "SoulMaintenance"
    
    Write-OpenClawLog "Creating soul maintenance task" -Level "Info"
    
    try {
        $taskDef = $Scheduler.NewTask(0)
        
        $regInfo = $taskDef.RegistrationInfo
        $regInfo.Description = "OpenClaw AI Agent Soul Maintenance"
        $regInfo.Author = "OpenClaw Framework"
        
        $principal = $taskDef.Principal
        $principal.UserId = "NT AUTHORITY\SYSTEM"
        $principal.LogonType = $SCRIPT:LogonType.ServiceAccount
        $principal.RunLevel = $SCRIPT:RunLevel.Highest
        
        $settings = $taskDef.Settings
        $settings.Enabled = $true
        $settings.Hidden = $false
        $settings.StartWhenAvailable = $true
        
        # Run every hour
        $triggers = $taskDef.Triggers
        $trigger = $triggers.Create($SCRIPT:TriggerType.Daily)
        $trigger.StartBoundary = (Get-Date).AddHours(1).ToString("s")
        $trigger.DaysInterval = 1
        $trigger.Repetition.Interval = "PT1H"
        $trigger.Repetition.Duration = "P1D"
        $trigger.Enabled = $true
        
        $actions = $taskDef.Actions
        $action = $actions.Create($SCRIPT:ActionType.Execute)
        $action.Path = "powershell.exe"
        $action.Arguments = "-ExecutionPolicy Bypass -Command `"& C:\OpenClaw\soul_maintenance.ps1 -AgentId '$AgentId'`""
        
        $folder = $Scheduler.GetFolder($SCRIPT:CoreFolder)
        $folder.RegisterTaskDefinition($taskName, $taskDef, 6, $null, $null, 5) | Out-Null
        
        Write-OpenClawLog "Successfully created soul maintenance task" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to create soul maintenance task: $_" -Level "Error"
        return $false
    }
}
#endregion

#region Cron Translation
function ConvertFrom-CronExpression {
    <#
    .SYNOPSIS
        Convert Unix cron expression to Task Scheduler configuration
    .DESCRIPTION
        Translates standard Unix cron expressions to Windows Task Scheduler
        trigger configurations.
    .PARAMETER CronExpression
        Cron expression (e.g., "0 */6 * * *" for every 6 hours)
    .OUTPUTS
        Hashtable with trigger configuration
    .EXAMPLE
        $config = ConvertFrom-CronExpression "0 */6 * * *"
        # Returns configuration for 6-hour interval trigger
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$CronExpression
    )
    
    # Handle special strings
    $specialStrings = @{
        "@yearly" = "0 0 1 1 *"
        "@annually" = "0 0 1 1 *"
        "@monthly" = "0 0 1 * *"
        "@weekly" = "0 0 * * 0"
        "@daily" = "0 0 * * *"
        "@hourly" = "0 * * * *"
        "@reboot" = "BOOT"
    }
    
    if ($specialStrings.ContainsKey($CronExpression)) {
        return @{ Type = "Special"; Value = $specialStrings[$CronExpression] }
    }
    
    # Parse standard cron
    $parts = $CronExpression -split "\s+"
    if ($parts.Count -ne 5) {
        throw "Invalid cron expression: $CronExpression"
    }
    
    return @{
        Type = "Standard"
        Minute = $parts[0]
        Hour = $parts[1]
        DayOfMonth = $parts[2]
        Month = $parts[3]
        DayOfWeek = $parts[4]
    }
}

function New-TaskFromCron {
    <#
    .SYNOPSIS
        Create a Task Scheduler task from a cron expression
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER CronExpression
        Cron expression
    .PARAMETER TaskName
        Name for the new task
    .PARAMETER Action
        Command/action to execute
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [Parameter(Mandatory=$true)]$CronExpression,
        [Parameter(Mandatory=$true)]$TaskName,
        [Parameter(Mandatory=$true)]$Action
    )
    
    $cronConfig = ConvertFrom-CronExpression $CronExpression
    
    # Create appropriate trigger based on cron pattern
    $taskDef = $Scheduler.NewTask(0)
    $triggers = $taskDef.Triggers
    
    if ($cronConfig.Type -eq "Special" -and $cronConfig.Value -eq "BOOT") {
        $trigger = $triggers.Create($SCRIPT:TriggerType.Boot)
        $trigger.Delay = "PT1M"
    }
    else {
        # Default to daily with repetition
        $trigger = $triggers.Create($SCRIPT:TriggerType.Daily)
        $trigger.StartBoundary = (Get-Date).ToString("s")
        $trigger.DaysInterval = 1
        
        # Calculate repetition from minute field
        if ($cronConfig.Minute -match "^\*/(\d+)$") {
            $interval = $matches[1]
            $trigger.Repetition.Interval = "PT${interval}M"
            $trigger.Repetition.Duration = "P1D"
        }
    }
    
    $trigger.Enabled = $true
    
    # Add action
    $actions = $taskDef.Actions
    $execAction = $actions.Create($SCRIPT:ActionType.Execute)
    $execAction.Path = "powershell.exe"
    $execAction.Arguments = "-Command `"$Action`""
    
    # Register
    $folder = $Scheduler.GetFolder("\")
    $folder.RegisterTaskDefinition($TaskName, $taskDef, 6, $null, $null, 5) | Out-Null
    
    Write-OpenClawLog "Created task '$TaskName' from cron: $CronExpression" -Level "Success"
}
#endregion

#region Task Monitoring
function Get-OpenClawTaskStatus {
    <#
    .SYNOPSIS
        Get status of all OpenClaw tasks
    .DESCRIPTION
        Retrieves detailed status information for all tasks in the OpenClaw
        folder hierarchy including state, last run time, and next run time.
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .OUTPUTS
        Array of task status objects
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler
    )
    
    $results = @()
    
    try {
        $folder = $Scheduler.GetFolder($SCRIPT:RootFolder)
        $tasks = $folder.GetTasks(1)  # Include hidden tasks
        
        foreach ($task in $tasks) {
            $stateName = switch ($task.State) {
                0 { "Unknown" }
                1 { "Disabled" }
                2 { "Queued" }
                3 { "Ready" }
                4 { "Running" }
                default { "Unknown" }
            }
            
            $results += [PSCustomObject]@{
                Name = $task.Name
                Path = $task.Path
                State = $stateName
                StateCode = $task.State
                Enabled = $task.Enabled
                LastRunTime = $task.LastRunTime
                LastTaskResult = $task.LastTaskResult
                NextRunTime = $task.NextRunTime
                NumberOfMissedRuns = $task.NumberOfMissedRuns
            }
        }
    }
    catch {
        Write-OpenClawLog "Failed to get tasks: $_" -Level "Error"
    }
    
    return $results
}

function Get-OpenClawRunningTasks {
    <#
    .SYNOPSIS
        Get currently running OpenClaw tasks
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler
    )
    
    $allTasks = Get-OpenClawTaskStatus -Scheduler $Scheduler
    return $allTasks | Where-Object { $_.StateCode -eq 4 }  # Running state
}

function Start-OpenClawTask {
    <#
    .SYNOPSIS
        Start an OpenClaw task manually
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER TaskName
        Name of the task to start
    .PARAMETER Arguments
        Optional arguments to pass to the task
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [Parameter(Mandatory=$true)]$TaskName,
        [string]$Arguments = ""
    )
    
    try {
        $folder = $Scheduler.GetFolder($SCRIPT:RootFolder)
        $task = $folder.GetTask($TaskName)
        
        if ($Arguments) {
            $task.Run($Arguments) | Out-Null
        }
        else {
            $task.Run() | Out-Null
        }
        
        Write-OpenClawLog "Started task: $TaskName" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to start task $TaskName`: $_" -Level "Error"
        return $false
    }
}

function Stop-OpenClawTask {
    <#
    .SYNOPSIS
        Stop a running OpenClaw task
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER TaskName
        Name of the task to stop
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [Parameter(Mandatory=$true)]$TaskName
    )
    
    try {
        $folder = $Scheduler.GetFolder($SCRIPT:RootFolder)
        $task = $folder.GetTask($TaskName)
        $task.Stop(0) | Out-Null
        
        Write-OpenClawLog "Stopped task: $TaskName" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to stop task $TaskName`: $_" -Level "Error"
        return $false
    }
}

function Remove-OpenClawTask {
    <#
    .SYNOPSIS
        Remove an OpenClaw task
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .PARAMETER TaskName
        Name of the task to remove
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler,
        [Parameter(Mandatory=$true)]$TaskName
    )
    
    try {
        $folder = $Scheduler.GetFolder($SCRIPT:RootFolder)
        $folder.DeleteTask($TaskName, 0) | Out-Null
        
        Write-OpenClawLog "Removed task: $TaskName" -Level "Success"
        return $true
    }
    catch {
        Write-OpenClawLog "Failed to remove task $TaskName`: $_" -Level "Error"
        return $false
    }
}
#endregion

#region Health Checking
function Test-OpenClawHealth {
    <#
    .SYNOPSIS
        Perform health check on OpenClaw task system
    .DESCRIPTION
        Checks the health of all OpenClaw tasks, identifies issues,
        and provides recommendations for recovery.
    .PARAMETER Scheduler
        Connected Task Scheduler service object
    .OUTPUTS
        Health check report object
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Scheduler
    )
    
    $report = @{
        Timestamp = Get-Date -Format "o"
        OverallStatus = "Healthy"
        Tasks = @{}
        Issues = @()
        Recommendations = @()
    }
    
    $tasks = Get-OpenClawTaskStatus -Scheduler $Scheduler
    
    foreach ($task in $tasks) {
        $taskHealth = @{
            Status = "Healthy"
            Issues = @()
        }
        
        # Check if disabled
        if (-not $task.Enabled) {
            $taskHealth.Status = "Critical"
            $taskHealth.Issues += "Task is disabled"
            $report.Issues += "Task $($task.Name) is disabled"
        }
        
        # Check last run result
        if ($task.LastTaskResult -ne 0) {
            $taskHealth.Status = "Degraded"
            $taskHealth.Issues += "Last run failed with code $($task.LastTaskResult)"
            $report.Issues += "Task $($task.Name) last run failed"
        }
        
        # Check for missed runs
        if ($task.NumberOfMissedRuns -gt 0) {
            $taskHealth.Status = "Degraded"
            $taskHealth.Issues += "$($task.NumberOfMissedRuns) missed runs"
            $report.Issues += "Task $($task.Name) has missed runs"
        }
        
        $report.Tasks[$task.Name] = $taskHealth
    }
    
    # Determine overall status
    $criticalCount = ($report.Tasks.Values | Where-Object { $_.Status -eq "Critical" }).Count
    $degradedCount = ($report.Tasks.Values | Where-Object { $_.Status -eq "Degraded" }).Count
    
    if ($criticalCount -gt 0) {
        $report.OverallStatus = "Critical"
        $report.Recommendations += "Immediate attention required for critical tasks"
    }
    elseif ($degradedCount -gt 0) {
        $report.OverallStatus = "Degraded"
        $report.Recommendations += "Review and address degraded tasks"
    }
    
    return $report
}
#endregion

#region Main Installation
function Install-OpenClawTaskScheduler {
    <#
    .SYNOPSIS
        Complete installation of OpenClaw Task Scheduler integration
    .DESCRIPTION
        Performs full installation including:
        - Folder structure creation
        - Heartbeat tasks (Primary, Secondary, Tertiary)
        - 15 agentic loop tasks
        - Soul maintenance task
        - Health monitoring
    .PARAMETER AgentId
        Unique identifier for the AI Agent
    .PARAMETER CreateAllLoops
        Create all 15 agentic loop tasks
    #>
    [CmdletBinding()]
    param(
        [string]$AgentId = "openclaw_agent_$(Get-Random -Maximum 9999)",
        [switch]$CreateAllLoops = $true
    )
    
    Write-Host "=== OpenClaw Task Scheduler Installation ===" -ForegroundColor Cyan
    Write-Host "Version: $($SCRIPT:Version)" -ForegroundColor Gray
    Write-Host "Agent ID: $AgentId" -ForegroundColor Gray
    Write-Host ""
    
    # Ensure OpenClaw directory exists
    if (-not (Test-Path "C:\OpenClaw")) {
        New-Item -ItemType Directory -Path "C:\OpenClaw" -Force | Out-Null
        Write-OpenClawLog "Created OpenClaw directory" -Level "Success"
    }
    
    # Save agent configuration
    $config = @{
        AgentId = $AgentId
        Version = $SCRIPT:Version
        InstallDate = Get-Date -Format "o"
        Features = @(
            "TaskScheduler"
            "Heartbeat"
            "AgentLoops"
            "SoulMaintenance"
        )
    }
    $config | ConvertTo-Json | Set-Content $SCRIPT:ConfigPath
    
    # Connect to Task Scheduler
    $scheduler = Connect-TaskScheduler
    if (-not $scheduler) {
        Write-OpenClawLog "Installation failed - could not connect to Task Scheduler" -Level "Error"
        return
    }
    
    # Create folder structure
    Write-Host "Creating folder structure..." -ForegroundColor Yellow
    Initialize-OpenClawFolderStructure -Scheduler $scheduler
    
    # Create heartbeat tasks
    Write-Host "`nCreating heartbeat tasks..." -ForegroundColor Yellow
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Primary" -AgentId $AgentId
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Secondary" -AgentId $AgentId
    New-OpenClawHeartbeatTask -Scheduler $scheduler -Frequency "Tertiary" -AgentId $AgentId
    
    # Create soul maintenance task
    Write-Host "`nCreating soul maintenance task..." -ForegroundColor Yellow
    New-OpenClawSoulMaintenanceTask -Scheduler $scheduler -AgentId $AgentId
    
    # Create agent loop tasks
    if ($CreateAllLoops) {
        Write-Host "`nCreating agent loop tasks..." -ForegroundColor Yellow
        
        $loops = @(
            @{ Number = 1; Name = "EmailProcessor"; Trigger = "Heartbeat"; Interval = "PT2M" },
            @{ Number = 2; Name = "BrowserAutomation"; Trigger = "Heartbeat"; Interval = "PT5M" },
            @{ Number = 3; Name = "VoiceHandler"; Trigger = "Heartbeat"; Interval = "PT1M" },
            @{ Number = 4; Name = "Communication"; Trigger = "Heartbeat"; Interval = "PT3M" },
            @{ Number = 5; Name = "FileOperations"; Trigger = "Event"; Interval = "" },
            @{ Number = 6; Name = "NetworkMonitor"; Trigger = "Heartbeat"; Interval = "PT30S" },
            @{ Number = 7; Name = "ProcessManager"; Trigger = "Heartbeat"; Interval = "PT10S" },
            @{ Number = 8; Name = "RegistryWatcher"; Trigger = "Event"; Interval = "" },
            @{ Number = 9; Name = "EventProcessor"; Trigger = "Event"; Interval = "" },
            @{ Number = 10; Name = "SchedulerCoordinator"; Trigger = "Heartbeat"; Interval = "PT1M" },
            @{ Number = 11; Name = "UserActivity"; Trigger = "Logon"; Interval = "" },
            @{ Number = 12; Name = "SecurityMonitor"; Trigger = "Heartbeat"; Interval = "PT5M" },
            @{ Number = 13; Name = "BackupManager"; Trigger = "Heartbeat"; Interval = "PT1H" },
            @{ Number = 14; Name = "UpdateChecker"; Trigger = "Heartbeat"; Interval = "PT6H" },
            @{ Number = 15; Name = "Maintenance"; Trigger = "Heartbeat"; Interval = "PT1H" }
        )
        
        foreach ($loop in $loops) {
            New-OpenClawAgentLoopTask `
                -Scheduler $scheduler `
                -LoopNumber $loop.Number `
                -LoopName $loop.Name `
                -TriggerType $loop.Trigger `
                -Interval $loop.Interval
        }
    }
    
    # Display final status
    Write-Host "`n=== Task Status ===" -ForegroundColor Yellow
    Get-OpenClawTaskStatus -Scheduler $scheduler | Format-Table -AutoSize
    
    # Health check
    Write-Host "`n=== Health Check ===" -ForegroundColor Yellow
    $health = Test-OpenClawHealth -Scheduler $scheduler
    Write-Host "Overall Status: $($health.OverallStatus)" -ForegroundColor $(
        switch ($health.OverallStatus) {
            "Healthy" { "Green" }
            "Degraded" { "Yellow" }
            "Critical" { "Red" }
        }
    )
    
    if ($health.Issues.Count -gt 0) {
        Write-Host "Issues found:" -ForegroundColor Red
        $health.Issues | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    }
    
    if ($health.Recommendations.Count -gt 0) {
        Write-Host "Recommendations:" -ForegroundColor Yellow
        $health.Recommendations | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    }
    
    Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
    Write-Host "Agent ID: $AgentId" -ForegroundColor Gray
    Write-Host "Configuration: $SCRIPT:ConfigPath" -ForegroundColor Gray
    Write-Host "Logs: $SCRIPT:LogPath" -ForegroundColor Gray
}

function Uninstall-OpenClawTaskScheduler {
    <#
    .SYNOPSIS
        Remove all OpenClaw tasks and folders
    .DESCRIPTION
        Completely removes all OpenClaw tasks, folders, and configuration.
        Use with caution - this cannot be undone.
    .PARAMETER Force
        Skip confirmation prompt
    #>
    [CmdletBinding()]
    param(
        [switch]$Force = $false
    )
    
    if (-not $Force) {
        $confirm = Read-Host "This will remove all OpenClaw tasks. Are you sure? (yes/no)"
        if ($confirm -ne "yes") {
            Write-Host "Uninstall cancelled" -ForegroundColor Yellow
            return
        }
    }
    
    Write-Host "=== Uninstalling OpenClaw Task Scheduler ===" -ForegroundColor Cyan
    
    $scheduler = Connect-TaskScheduler
    if (-not $scheduler) {
        return
    }
    
    # Remove root folder (removes all tasks and subfolders)
    try {
        $root = $scheduler.GetFolder("\")
        $root.DeleteFolder("OpenClaw", 0) | Out-Null
        Write-OpenClawLog "Removed OpenClaw folder and all tasks" -Level "Success"
    }
    catch {
        Write-OpenClawLog "Error removing folder: $_" -Level "Warning"
    }
    
    # Remove configuration
    if (Test-Path $SCRIPT:ConfigPath) {
        Remove-Item $SCRIPT:ConfigPath -Force
        Write-OpenClawLog "Removed configuration file" -Level "Success"
    }
    
    # Remove log files
    if (Test-Path $SCRIPT:LogPath) {
        Remove-Item $SCRIPT:LogPath -Recurse -Force
        Write-OpenClawLog "Removed log files" -Level "Success"
    }
    
    Write-Host "=== Uninstall Complete ===" -ForegroundColor Green
}
#endregion

#region Export
# Export all public functions
Export-ModuleMember -Function @(
    'Connect-TaskScheduler',
    'Initialize-OpenClawFolderStructure',
    'Get-OpenClawFolder',
    'New-OpenClawHeartbeatTask',
    'New-OpenClawAgentLoopTask',
    'New-OpenClawSoulMaintenanceTask',
    'ConvertFrom-CronExpression',
    'New-TaskFromCron',
    'Get-OpenClawTaskStatus',
    'Get-OpenClawRunningTasks',
    'Start-OpenClawTask',
    'Stop-OpenClawTask',
    'Remove-OpenClawTask',
    'Test-OpenClawHealth',
    'Install-OpenClawTaskScheduler',
    'Uninstall-OpenClawTaskScheduler',
    'Write-OpenClawLog'
)
#endregion

# Auto-execute if run directly (not imported)
if ($MyInvocation.InvocationName -eq $MyInvocation.MyCommand.Name -or 
    $MyInvocation.InvocationName -eq '.\OpenClaw_AI_Agent_TaskScheduler.ps1') {
    Write-Host "OpenClaw AI Agent Task Scheduler Module v$($SCRIPT:Version)" -ForegroundColor Cyan
    Write-Host "Use 'Install-OpenClawTaskScheduler' to begin installation" -ForegroundColor Green
    Write-Host "Use 'Get-Help Install-OpenClawTaskScheduler -Full' for detailed help" -ForegroundColor Gray
}
