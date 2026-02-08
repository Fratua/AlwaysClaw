# TaskSchedulerManager.ps1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Windows Task Scheduler management for OpenClaw AI Agent System
.DESCRIPTION
    Provides comprehensive task scheduling capabilities for the 24/7 AI Agent system
.PARAMETER TaskName
    The name of the scheduled task
.PARAMETER NodePath
    Path to the Node.js executable
.PARAMETER ScriptPath
    Path to the main script file
.PARAMETER WorkingDirectory
    Working directory for the task
.PARAMETER Action
    Action to perform: Create, Delete, Enable, Disable, Get, Run, Stop
.EXAMPLE
    .\TaskSchedulerManager.ps1 -Action Create
    Creates the scheduled task for the OpenClaw Agent
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$TaskName = "OpenClawAgent",
    
    [Parameter()]
    [string]$NodePath = "C:\Program Files\nodejs\node.exe",
    
    [Parameter()]
    [string]$ScriptPath = "",
    
    [Parameter()]
    [string]$WorkingDirectory = "",
    
    [Parameter()]
    [ValidateSet("Create", "Delete", "Enable", "Disable", "Get", "Run", "Stop")]
    [string]$Action = "Get"
)

# Auto-detect paths if not provided
if ([string]::IsNullOrEmpty($ScriptPath)) {
    $ScriptPath = Join-Path $PSScriptRoot "..\dist\index.js"
    $ScriptPath = Resolve-Path $ScriptPath -ErrorAction SilentlyContinue
}

if ([string]::IsNullOrEmpty($WorkingDirectory)) {
    $WorkingDirectory = Join-Path $PSScriptRoot ".."
    $WorkingDirectory = Resolve-Path $WorkingDirectory -ErrorAction SilentlyContinue
}

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
        Write-Host "Creating scheduled task '$TaskName'..." -ForegroundColor Cyan
        
        # Validate paths
        if (-not (Test-Path $NodePath)) {
            throw "Node.js executable not found at: $NodePath"
        }
        if (-not (Test-Path $ScriptPath)) {
            throw "Script not found at: $ScriptPath"
        }
        if (-not (Test-Path $WorkingDirectory)) {
            throw "Working directory not found at: $WorkingDirectory"
        }
        
        # Get or create folder
        $rootFolder = $Service.GetFolder("\")
        try {
            $folder = $rootFolder.GetFolder("OpenClaw")
        }
        catch {
            $folder = $rootFolder.CreateFolder("OpenClaw")
        }
        
        # Delete existing task if present
        try {
            $folder.DeleteTask($TaskConfig.Name, 0)
            Write-Host "Deleted existing task" -ForegroundColor Yellow
        }
        catch {
            # Task doesn't exist, continue
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
        $action.Path = $NodePath
        $action.Arguments = "`"$ScriptPath`""
        $action.WorkingDirectory = $WorkingDirectory
        
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
        exit 1
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
        exit 1
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
        exit 1
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
        exit 1
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
        
        $stateMap = @{
            0 = "Unknown"
            1 = "Disabled"
            2 = "Queued"
            3 = "Ready"
            4 = "Running"
        }
        
        $result = [PSCustomObject]@{
            Name = $task.Name
            Path = $task.Path
            State = $stateMap[$task.State]
            LastRunTime = $task.LastRunTime
            NextRunTime = $task.NextRunTime
            LastTaskResult = $task.LastTaskResult
            NumberOfMissedRuns = $task.NumberOfMissedRuns
            Enabled = $task.Enabled
        }
        
        return $result
    }
    catch {
        Write-Error "Task not found: $_"
        return $null
    }
}

# Main execution
Write-Host "OpenClaw Task Scheduler Manager" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Gray
Write-Host ""

$Service = Initialize-TaskScheduler

switch ($Action) {
    "Create" { 
        New-AgentTask -Service $Service 
    }
    "Delete" { 
        Remove-AgentTask -Service $Service 
    }
    "Enable" { 
        try {
            $folder = $Service.GetFolder($TaskConfig.Path)
            $task = $folder.GetTask($TaskConfig.Name)
            $task.Enabled = $true
            Write-Host "Task enabled" -ForegroundColor Green
        }
        catch {
            Write-Error "Failed to enable task: $_"
        }
    }
    "Disable" { 
        try {
            $folder = $Service.GetFolder($TaskConfig.Path)
            $task = $folder.GetTask($TaskConfig.Name)
            $task.Enabled = $false
            Write-Host "Task disabled" -ForegroundColor Yellow
        }
        catch {
            Write-Error "Failed to disable task: $_"
        }
    }
    "Get" { 
        $task = Get-AgentTask -Service $Service
        if ($task) {
            $task | Format-List
        }
    }
    "Run" { 
        Start-AgentTask -Service $Service 
    }
    "Stop" { 
        Stop-AgentTask -Service $Service 
    }
}
