# OpenClaw AI Agent - Windows Auto-Scaling Script
# Monitors system metrics and scales agent instances

#Requires -RunAsAdministrator

param(
    [string]$ServiceName = "OpenClawAgent",
    [int]$MinInstances = 1,
    [int]$MaxInstances = 100,
    [int]$CpuThreshold = 70,
    [int]$MemoryThreshold = 80,
    [int]$QueueThreshold = 100,
    [int]$ScaleOutCooldown = 60,
    [int]$ScaleInCooldown = 300,
    [int]$CheckInterval = 10,
    [string]$ConfigPath = "$PSScriptRoot\..\config\scaling_config.yaml",
    [string]$LogPath = "$env:ProgramData\OpenClaw\Logs",
    [switch]$RunOnce,
    [switch]$InstallTask,
    [switch]$UninstallTask
)

# Error handling
$ErrorActionPreference = "Stop"

# Global state
$script:LastScaleOutTime = 0
$script:LastScaleInTime = 0
$script:InstanceCount = 0
$script:MetricsHistory = @()

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    
    # Write to log file
    $logFile = "$LogPath\auto_scale.log"
    $logDir = Split-Path $logFile -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logMessage
}

function Get-AgentMetrics {
    <#
    .SYNOPSIS
    Collects metrics for all agent instances
    #>
    
    $instances = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { 
        $_.CommandLine -like "*agent_core*" -or $_.CommandLine -like "*$ServiceName*"
    }
    
    $metrics = @{
        InstanceCount = 0
        TotalCpu = 0
        TotalMemory = 0
        TotalWorkingSet = 0
        AvgCpuPerInstance = 0
        AvgMemoryPerInstance = 0
        MaxCpuInstance = $null
        MaxMemoryInstance = $null
    }
    
    if ($instances) {
        $metrics.InstanceCount = $instances.Count
        $script:InstanceCount = $instances.Count
        
        $cpuValues = @()
        $memoryValues = @()
        
        foreach ($instance in $instances) {
            try {
                # Get CPU usage (requires performance counters)
                $cpu = $instance.CPU
                $metrics.TotalCpu += $cpu
                $cpuValues += $cpu
                
                # Get memory usage
                $memory = $instance.WorkingSet64 / 1MB
                $metrics.TotalMemory += $memory
                $metrics.TotalWorkingSet += $instance.WorkingSet64
                $memoryValues += $memory
                
                # Track max consumers
                if (-not $metrics.MaxCpuInstance -or $cpu -gt $metrics.MaxCpuInstance.Cpu) {
                    $metrics.MaxCpuInstance = @{
                        Id = $instance.Id
                        Cpu = $cpu
                    }
                }
                if (-not $metrics.MaxMemoryInstance -or $memory -gt $metrics.MaxMemoryInstance.Memory) {
                    $metrics.MaxMemoryInstance = @{
                        Id = $instance.Id
                        Memory = $memory
                    }
                }
            }
            catch {
                Write-Log "Error getting metrics for process $($instance.Id): $_" -Level "WARNING"
            }
        }
        
        # Calculate averages
        if ($metrics.InstanceCount -gt 0) {
            $metrics.AvgCpuPerInstance = $metrics.TotalCpu / $metrics.InstanceCount
            $metrics.AvgMemoryPerInstance = $metrics.TotalMemory / $metrics.InstanceCount
        }
        
        # Add to history
        $script:MetricsHistory += @{
            Timestamp = Get-Date
            InstanceCount = $metrics.InstanceCount
            AvgCpu = $metrics.AvgCpuPerInstance
            AvgMemory = $metrics.AvgMemoryPerInstance
        }
        
        # Keep only last 60 entries
        if ($script:MetricsHistory.Count -gt 60) {
            $script:MetricsHistory = $script:MetricsHistory[-60..-1]
        }
    }
    
    return $metrics
}

function Get-SystemMetrics {
    <#
    .SYNOPSIS
    Collects system-wide metrics
    #>
    
    $metrics = @{}
    
    # CPU usage
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
    if ($cpu) {
        $metrics.CpuPercent = [math]::Round($cpu.CounterSamples[0].CookedValue, 2)
    }
    
    # Memory usage
    $memory = Get-CimInstance -ClassName Win32_OperatingSystem
    $metrics.MemoryPercent = [math]::Round((($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100, 2)
    $metrics.MemoryAvailableGB = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
    
    # Disk usage
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $metrics.DiskPercent = [math]::Round((($disk.Size - $disk.FreeSpace) / $disk.Size) * 100, 2)
    
    # Load average (simulated)
    $metrics.LoadAvg = $metrics.CpuPercent / 100
    
    return $metrics
}

function Get-QueueMetrics {
    <#
    .SYNOPSIS
    Gets message queue depth (placeholder - implement based on your queue)
    #>
    
    # This should be implemented based on your message queue
    # For Redis: redis-cli LLEN queue_name
    # For RabbitMQ: rabbitmqctl list_queues
    
    return @{
        QueueDepth = 0  # Placeholder
        ProcessingLag = 0
    }
}

function Test-ScaleOutCondition {
    <#
    .SYNOPSIS
    Tests if scale out conditions are met
    #>
    
    $agentMetrics = Get-AgentMetrics
    $systemMetrics = Get-SystemMetrics
    $queueMetrics = Get-QueueMetrics
    
    $reasons = @()
    $instancesToAdd = 0
    
    # Check CPU threshold
    if ($agentMetrics.AvgCpuPerInstance -gt $CpuThreshold) {
        $reasons += "CPU usage ($([math]::Round($agentMetrics.AvgCpuPerInstance, 1))%) exceeds threshold ($CpuThreshold%)"
        $instancesToAdd = [math]::Max($instancesToAdd, 1)
    }
    
    # Check memory threshold
    if ($agentMetrics.AvgMemoryPerInstance -gt $MemoryThreshold) {
        $reasons += "Memory usage ($([math]::Round($agentMetrics.AvgMemoryPerInstance, 1))%) exceeds threshold ($MemoryThreshold%)"
        $instancesToAdd = [math]::Max($instancesToAdd, 1)
    }
    
    # Check queue depth
    if ($queueMetrics.QueueDepth -gt $QueueThreshold) {
        $reasons += "Queue depth ($($queueMetrics.QueueDepth)) exceeds threshold ($QueueThreshold)"
        $instancesToAdd = [math]::Max($instancesToAdd, [math]::Ceiling($queueMetrics.QueueDepth / 20))
    }
    
    # Check system CPU (emergency)
    if ($systemMetrics.CpuPercent -gt 95) {
        $reasons += "SYSTEM CPU CRITICAL: $($systemMetrics.CpuPercent)%"
        $instancesToAdd = [math]::Max($instancesToAdd, 3)
    }
    
    return @{
        ShouldScale = $reasons.Count -gt 0
        InstancesToAdd = $instancesToAdd
        Reasons = $reasons
    }
}

function Test-ScaleInCondition {
    <#
    .SYNOPSIS
    Tests if scale in conditions are met
    #>
    
    $agentMetrics = Get-AgentMetrics
    $systemMetrics = Get-SystemMetrics
    
    $reasons = @()
    $instancesToRemove = 0
    
    # Need at least minimum + 1 to scale in
    if ($agentMetrics.InstanceCount -le $MinInstances) {
        return @{
            ShouldScale = $false
            InstancesToRemove = 0
            Reasons = @("At minimum instance count")
        }
    }
    
    # Check low CPU usage
    if ($agentMetrics.AvgCpuPerInstance -lt 30 -and $systemMetrics.CpuPercent -lt 50) {
        $reasons += "Low CPU usage ($([math]::Round($agentMetrics.AvgCpuPerInstance, 1))%)"
        $instancesToRemove = 1
    }
    
    # Check very low CPU for aggressive scale in
    if ($agentMetrics.AvgCpuPerInstance -lt 20 -and $systemMetrics.CpuPercent -lt 30) {
        $reasons += "Very low CPU usage ($([math]::Round($agentMetrics.AvgCpuPerInstance, 1))%)"
        $instancesToRemove = 2
    }
    
    return @{
        ShouldScale = $reasons.Count -gt 0
        InstancesToRemove = [math]::Min($instancesToRemove, $agentMetrics.InstanceCount - $MinInstances)
        Reasons = $reasons
    }
}

function Start-AgentInstance {
    <#
    .SYNOPSIS
    Starts a new agent instance
    #>
    param([int]$Count = 1)
    
    $started = 0
    
    for ($i = 0; $i -lt $Count; $i++) {
        try {
            $instanceId = [guid]::NewGuid().ToString()
            $port = 8080 + $script:InstanceCount + $i
            
            Write-Log "Starting new agent instance: $instanceId on port $port"
            
            # Start new agent process
            $pythonPath = "C:\OpenClaw\venv\Scripts\python.exe"
            $scriptPath = "C:\OpenClaw\agent_core.py"
            
            $process = Start-Process -FilePath $pythonPath `
                -ArgumentList $scriptPath, "--instance-id", $instanceId, "--port", $port `
                -WindowStyle Hidden `
                -PassThru
            
            # Wait a moment and verify process started
            Start-Sleep -Seconds 2
            
            if (Get-Process -Id $process.Id -ErrorAction SilentlyContinue) {
                Write-Log "Instance $instanceId started successfully (PID: $($process.Id))"
                $started++
            }
            else {
                Write-Log "Instance $instanceId failed to start" -Level "ERROR"
            }
        }
        catch {
            Write-Log "Error starting instance: $_" -Level "ERROR"
        }
    }
    
    return $started
}

function Stop-AgentInstance {
    <#
    .SYNOPSIS
    Stops agent instances (gracefully)
    #>
    param([int]$Count = 1)
    
    $stopped = 0
    
    # Get all agent processes sorted by resource usage (lowest first)
    $instances = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { 
        $_.CommandLine -like "*agent_core*" -or $_.CommandLine -like "*$ServiceName*"
    } | Sort-Object { $_.CPU + $_.WorkingSet64 / 1MB }
    
    for ($i = 0; $i -lt [math]::Min($Count, $instances.Count); $i++) {
        try {
            $instance = $instances[$i]
            Write-Log "Stopping agent instance (PID: $($instance.Id))"
            
            # Try graceful shutdown first
            $instance.CloseMainWindow() | Out-Null
            
            # Wait for graceful shutdown
            if (-not $instance.WaitForExit(10000)) {
                Write-Log "Force killing instance (PID: $($instance.Id))" -Level "WARNING"
                Stop-Process -Id $instance.Id -Force
            }
            
            $stopped++
            Write-Log "Instance stopped successfully"
        }
        catch {
            Write-Log "Error stopping instance: $_" -Level "ERROR"
        }
    }
    
    return $stopped
}

function Invoke-Scaling {
    <#
    .SYNOPSIS
    Main scaling logic
    #>
    
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    
    # Check scale out
    $scaleOut = Test-ScaleOutCondition
    if ($scaleOut.ShouldScale) {
        # Check cooldown
        if ($now - $script:LastScaleOutTime -ge $ScaleOutCooldown) {
            Write-Log "Scale out triggered: $($scaleOut.Reasons -join '; ')"
            
            $added = Start-AgentInstance -Count $scaleOut.InstancesToAdd
            
            if ($added -gt 0) {
                $script:LastScaleOutTime = $now
                Write-Log "Scaled out: +$added instances"
            }
        }
        else {
            $remaining = $ScaleOutCooldown - ($now - $script:LastScaleOutTime)
            Write-Log "Scale out requested but in cooldown ($remaining seconds remaining)"
        }
    }
    
    # Check scale in
    $scaleIn = Test-ScaleInCondition
    if ($scaleIn.ShouldScale) {
        # Check cooldown
        if ($now - $script:LastScaleInTime -ge $ScaleInCooldown) {
            Write-Log "Scale in triggered: $($scaleIn.Reasons -join '; ')"
            
            $removed = Stop-AgentInstance -Count $scaleIn.InstancesToRemove
            
            if ($removed -gt 0) {
                $script:LastScaleInTime = $now
                Write-Log "Scaled in: -$removed instances"
            }
        }
        else {
            $remaining = $ScaleInCooldown - ($now - $script:LastScaleInTime)
            Write-Log "Scale in requested but in cooldown ($remaining seconds remaining)"
        }
    }
}

function Install-ScheduledTask {
    <#
    .SYNOPSIS
    Installs auto-scaler as a scheduled task
    #>
    
    Write-Log "Installing scheduled task for auto-scaling..."
    
    $taskName = "OpenClawAutoScaler"
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    
    $trigger = New-ScheduledTaskTrigger `
        -Once `
        -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes 1) `
        -RepetitionDuration (New-TimeSpan -Days 365)
    
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable
    
    $principal = New-ScheduledTaskPrincipal `
        -UserId "SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest
    
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null
    
    Write-Log "Scheduled task '$taskName' installed successfully"
}

function Uninstall-ScheduledTask {
    <#
    .SYNOPSIS
    Uninstalls auto-scaler scheduled task
    #>
    
    $taskName = "OpenClawAutoScaler"
    
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Log "Removing scheduled task: $taskName"
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Log "Scheduled task removed"
    }
    else {
        Write-Log "Scheduled task not found: $taskName"
    }
}

function Start-ScalingLoop {
    <#
    .SYNOPSIS
    Main scaling loop
    #>
    
    Write-Log "Starting auto-scaling loop (Min: $MinInstances, Max: $MaxInstances)"
    Write-Log "Check interval: $CheckInterval seconds"
    Write-Log "Scale out cooldown: $ScaleOutCooldown seconds"
    Write-Log "Scale in cooldown: $ScaleInCooldown seconds"
    
    while ($true) {
        try {
            # Collect and log metrics
            $agentMetrics = Get-AgentMetrics
            $systemMetrics = Get-SystemMetrics
            
            Write-Log ("Instances: $($agentMetrics.InstanceCount) | " +
                      "Avg CPU: $([math]::Round($agentMetrics.AvgCpuPerInstance, 1))% | " +
                      "Avg Memory: $([math]::Round($agentMetrics.AvgMemoryPerInstance, 1)) MB | " +
                      "System CPU: $($systemMetrics.CpuPercent)%")
            
            # Perform scaling
            Invoke-Scaling
        }
        catch {
            Write-Log "Error in scaling loop: $_" -Level "ERROR"
        }
        
        Start-Sleep -Seconds $CheckInterval
    }
}

# Main execution
try {
    if ($InstallTask) {
        Install-ScheduledTask
    }
    elseif ($UninstallTask) {
        Uninstall-ScheduledTask
    }
    elseif ($RunOnce) {
        Invoke-Scaling
    }
    else {
        Start-ScalingLoop
    }
}
catch {
    Write-Log "FATAL ERROR: $_" -Level "ERROR"
    exit 1
}
