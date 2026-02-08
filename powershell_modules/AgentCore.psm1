# AgentCore.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    Core PowerShell module for AI Agent system
.DESCRIPTION
    Provides fundamental agent operations including heartbeat,
    identity management, and system integration
#>

# Module-level variables
$script:AgentConfig = $null
$script:HeartbeatTimer = $null
$script:IdentityState = @{}
$script:OperationLog = [System.Collections.ArrayList]::new()

#region Configuration

function Initialize-AgentEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$Configuration
    )
    
    $script:AgentConfig = [PSCustomObject]$Configuration
    
    # Create necessary directories
    $paths = @(
        $Configuration.LogPath
        $Configuration.ScriptPath
        $Configuration.ConfigPath
        $Configuration.DataPath
    )
    
    foreach ($path in $paths) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-AgentLog -Message "Created directory: $path" -Level Info
        }
    }
    
    # Initialize identity state
    $script:IdentityState = @{
        AgentId = $Configuration.AgentId
        Name = $Configuration.Name
        CreatedAt = Get-Date
        LastHeartbeat = $null
        Status = 'Initializing'
        Version = $Configuration.Version
    }
    
    # Set environment variables
    [Environment]::SetEnvironmentVariable('AI_AGENT_ID', $Configuration.AgentId, 'Process')
    [Environment]::SetEnvironmentVariable('AI_AGENT_NAME', $Configuration.Name, 'Process')
    [Environment]::SetEnvironmentVariable('AI_AGENT_CONFIG_PATH', $Configuration.ConfigPath, 'Process')
    
    Write-AgentLog -Message "Agent environment initialized: $($Configuration.Name)" -Level Info
    
    return $script:IdentityState
}

#endregion

#region Heartbeat System

function Start-AgentHeartbeat {
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$IntervalSeconds = 60,
        
        [Parameter()]
        [scriptblock]$OnHeartbeat = $null,
        
        [Parameter()]
        [string]$LogPath
    )
    
    if ($script:HeartbeatTimer) {
        Stop-AgentHeartbeat
    }
    
    $heartbeatScript = {
        param($Config, $Callback, $LogDir)
        
        $heartbeatData = @{
            Timestamp = Get-Date -Format 'o'
            AgentId = $Config.AgentId
            Status = 'Active'
            SystemInfo = @{
                ComputerName = $env:COMPUTERNAME
                UserName = $env:USERNAME
                ProcessId = $PID
                MemoryWorkingSet = (Get-Process -Id $PID).WorkingSet64
            }
        }
        
        # Log heartbeat
        $logFile = Join-Path $LogDir "heartbeat-$(Get-Date -Format 'yyyyMMdd').json"
        $heartbeatData | ConvertTo-Json -Compress | Add-Content -Path $logFile
        
        # Execute callback if provided
        if ($Callback) {
            & $Callback $heartbeatData
        }
    }
    
    $timer = New-Object System.Timers.Timer
    $timer.Interval = $IntervalSeconds * 1000
    $timer.AutoReset = $true
    
    $action = {
        & $heartbeatScript $script:AgentConfig $OnHeartbeat $script:AgentConfig.LogPath
    }
    
    Register-ObjectEvent -InputObject $timer -EventName Elapsed -Action $action | Out-Null
    $timer.Start()
    
    $script:HeartbeatTimer = $timer
    $script:IdentityState.LastHeartbeat = Get-Date
    $script:IdentityState.Status = 'Running'
    
    Write-AgentLog -Message "Heartbeat started with interval: ${IntervalSeconds}s" -Level Info
}

function Stop-AgentHeartbeat {
    [CmdletBinding()]
    param()
    
    if ($script:HeartbeatTimer) {
        $script:HeartbeatTimer.Stop()
        $script:HeartbeatTimer.Dispose()
        $script:HeartbeatTimer = $null
        $script:IdentityState.Status = 'Stopped'
        
        Write-AgentLog -Message "Heartbeat stopped" -Level Info
    }
}

function Get-AgentHeartbeatStatus {
    [CmdletBinding()]
    param()
    
    return [PSCustomObject]@{
        IsRunning = $null -ne $script:HeartbeatTimer
        LastHeartbeat = $script:IdentityState.LastHeartbeat
        Status = $script:IdentityState.Status
    }
}

#endregion

#region Identity Management

function Get-AgentIdentity {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch]$IncludeSystemInfo
    )
    
    $identity = @{
        AgentId = $script:IdentityState.AgentId
        Name = $script:IdentityState.Name
        CreatedAt = $script:IdentityState.CreatedAt
        Version = $script:IdentityState.Version
        Status = $script:IdentityState.Status
    }
    
    if ($IncludeSystemInfo) {
        $identity.SystemInfo = @{
            ComputerName = $env:COMPUTERNAME
            UserName = $env:USERNAME
            UserDomain = $env:USERDOMAIN
            ProcessId = $PID
            PowerShellVersion = $PSVersionTable.PSVersion.ToString()
            OSVersion = [System.Environment]::OSVersion.VersionString
        }
    }
    
    return [PSCustomObject]$identity
}

function Set-AgentIdentityProperty {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Property,
        
        [Parameter(Mandatory)]
        [object]$Value
    )
    
    $script:IdentityState[$Property] = $Value
    Write-AgentLog -Message "Identity property updated: $Property" -Level Info
}

#endregion

#region Logging

function Write-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Message,
        
        [Parameter()]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error', 'Critical')]
        [string]$Level = 'Info',
        
        [Parameter()]
        [hashtable]$Metadata = @{},
        
        [Parameter()]
        [string]$Category = 'General'
    )
    
    $logEntry = [PSCustomObject]@{
        Timestamp = Get-Date -Format 'o'
        Level = $Level
        Category = $Category
        Message = $Message
        AgentId = $script:IdentityState.AgentId
        ProcessId = $PID
        Metadata = $Metadata
    }
    
    # Write to console with color
    $colorMap = @{
        Debug = 'Gray'
        Info = 'White'
        Warning = 'Yellow'
        Error = 'Red'
        Critical = 'Magenta'
    }
    
    Write-Host "[$($logEntry.Timestamp)] [$Level] $Message" -ForegroundColor $colorMap[$Level]
    
    # Write to file
    if ($script:AgentConfig -and $script:AgentConfig.LogPath) {
        $logFile = Join-Path $script:AgentConfig.LogPath "agent-$(Get-Date -Format 'yyyyMMdd').log"
        $logEntry | ConvertTo-Json -Compress | Add-Content -Path $logFile
    }
    
    # Add to operation log
    $script:OperationLog.Add($logEntry) | Out-Null
    
    # Keep only last 1000 entries in memory
    if ($script:OperationLog.Count -gt 1000) {
        $script:OperationLog.RemoveAt(0)
    }
}

function Get-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$Count = 100,
        
        [Parameter()]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error', 'Critical')]
        [string]$Level,
        
        [Parameter()]
        [string]$Category
    )
    
    $logs = $script:OperationLog
    
    if ($Level) {
        $logs = $logs | Where-Object { $_.Level -eq $Level }
    }
    
    if ($Category) {
        $logs = $logs | Where-Object { $_.Category -eq $Category }
    }
    
    return $logs | Select-Object -Last $Count
}

function Export-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        
        [Parameter()]
        [ValidateSet('Json', 'Csv', 'Xml')]
        [string]$Format = 'Json'
    )
    
    $logs = $script:OperationLog
    
    switch ($Format) {
        'Json' {
            $logs | ConvertTo-Json -Depth 5 | Out-File -FilePath $Path -Encoding UTF8
        }
        'Csv' {
            $logs | Export-Csv -Path $Path -NoTypeInformation -Encoding UTF8
        }
        'Xml' {
            $logs | Export-Clixml -Path $Path
        }
    }
    
    Write-AgentLog -Message "Logs exported to: $Path" -Level Info
}

#endregion

#region System Operations

function Get-AgentSystemStatus {
    [CmdletBinding()]
    param()
    
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $processor = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
    $memory = Get-CimInstance -ClassName Win32_PhysicalMemory
    
    return [PSCustomObject]@{
        Timestamp = Get-Date -Format 'o'
        OperatingSystem = @{
            Caption = $os.Caption
            Version = $os.Version
            Architecture = $os.OSArchitecture
            LastBootTime = $os.LastBootUpTime
        }
        Processor = @{
            Name = $processor.Name
            Cores = $processor.NumberOfCores
            LogicalProcessors = $processor.NumberOfLogicalProcessors
            LoadPercentage = $processor.LoadPercentage
        }
        Memory = @{
            TotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
            FreeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
            UsedPercent = [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100, 2)
            Modules = $memory | ForEach-Object { [math]::Round($_.Capacity / 1GB, 2) }
        }
        AgentProcess = @{
            Id = $PID
            WorkingSetMB = [math]::Round((Get-Process -Id $PID).WorkingSet64 / 1MB, 2)
            Threads = (Get-Process -Id $PID).Threads.Count
            Handles = (Get-Process -Id $PID).Handles
            StartTime = (Get-Process -Id $PID).StartTime
        }
    }
}

function Invoke-AgentMaintenance {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('LogCleanup', 'CacheClear', 'MemoryOptimize', 'Full')]
        [string]$Type = 'Full'
    )
    
    Write-AgentLog -Message "Starting maintenance: $Type" -Level Info -Category 'Maintenance'
    
    $results = @()
    
    if ($Type -in @('LogCleanup', 'Full')) {
        # Clean old log files (older than 30 days)
        $cutoffDate = (Get-Date).AddDays(-30)
        $logFiles = Get-ChildItem -Path $script:AgentConfig.LogPath -Filter '*.log' | 
            Where-Object { $_.LastWriteTime -lt $cutoffDate }
        
        foreach ($file in $logFiles) {
            Remove-Item -Path $file.FullName -Force
            $results += "Removed log file: $($file.Name)"
        }
    }
    
    if ($Type -in @('CacheClear', 'Full')) {
        # Clear operation log
        $script:OperationLog.Clear()
        $results += "Cleared operation log cache"
    }
    
    if ($Type -in @('MemoryOptimize', 'Full')) {
        # Force garbage collection
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
        $results += "Garbage collection completed"
    }
    
    Write-AgentLog -Message "Maintenance completed: $($results -join '; ')" -Level Info -Category 'Maintenance'
    
    return $results
}

#endregion

#region Export

Export-ModuleMember -Function @(
    'Initialize-AgentEnvironment'
    'Start-AgentHeartbeat'
    'Stop-AgentHeartbeat'
    'Get-AgentHeartbeatStatus'
    'Get-AgentIdentity'
    'Set-AgentIdentityProperty'
    'Write-AgentLog'
    'Get-AgentLog'
    'Export-AgentLog'
    'Get-AgentSystemStatus'
    'Invoke-AgentMaintenance'
)

#endregion
