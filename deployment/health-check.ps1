<#
.SYNOPSIS
    AlwaysClaw structured JSON health report.

.DESCRIPTION
    Performs comprehensive health checks across the AlwaysClaw stack and outputs
    a structured JSON report. Checks include:
      - WSL2 availability and distro status
      - Core stack process liveness (gateway, loopd, tools)
      - Sidecar process liveness (voice, browser)
      - Heartbeat freshness
      - Disk space availability
      - Scheduled task registration
      - Degraded mode status

    Used by the watchdog, boot-task, and operator tooling.

.OUTPUTS
    JSON object written to stdout and optionally to the health report file.

.NOTES
    Reference : Master Plan Sections 28, 35
    Schedule  : Called by watchdog every 5 minutes; also on-demand
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$Script:StateRoot      = 'C:\AlwaysClaw\state'
$Script:LogDir         = Join-Path $Script:StateRoot 'logs'
$Script:HeartbeatFile  = Join-Path $Script:StateRoot 'heartbeat.json'
$Script:DegradedFlag   = Join-Path $Script:StateRoot 'degraded-mode.json'
$Script:HealthOutFile  = Join-Path $Script:LogDir    'health-report.json'
$Script:WslDistro      = 'Ubuntu'

# Liveness thresholds
$Script:HeartbeatMaxAgeSec = 600  # 10 minutes

# ---------------------------------------------------------------------------
# Helper: run a check and capture result
# ---------------------------------------------------------------------------
function New-CheckResult {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Category,
        [Parameter(Mandatory)][scriptblock]$Check
    )
    $result = [PSCustomObject]@{
        name      = $Name
        category  = $Category
        status    = 'unknown'
        message   = ''
        timestamp = (Get-Date).ToString('o')
    }
    try {
        $outcome = & $Check
        $result.status  = $outcome.Status
        $result.message = $outcome.Message
    } catch {
        $result.status  = 'error'
        $result.message = $_.Exception.Message
    }
    return $result
}

# ---------------------------------------------------------------------------
# Check: WSL2 availability
# ---------------------------------------------------------------------------
function Test-WSL2Availability {
    return New-CheckResult -Name 'wsl2_available' -Category 'platform' -Check {
        $wslExe = Get-Command wsl.exe -ErrorAction SilentlyContinue
        if (-not $wslExe) {
            return @{ Status = 'critical'; Message = 'wsl.exe not found on PATH.' }
        }
        $status = & wsl.exe --status 2>&1
        $statusText = ($status | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            return @{ Status = 'critical'; Message = "wsl --status failed: $statusText" }
        }
        return @{ Status = 'healthy'; Message = "WSL2 operational. $statusText" }
    }
}

# ---------------------------------------------------------------------------
# Check: WSL distro running
# ---------------------------------------------------------------------------
function Test-WSLDistro {
    return New-CheckResult -Name 'wsl_distro' -Category 'platform' -Check {
        $distros = & wsl.exe --list --running --quiet 2>&1
        if ($distros -match $Script:WslDistro) {
            return @{ Status = 'healthy'; Message = "$($Script:WslDistro) distro is running." }
        }
        return @{ Status = 'warning'; Message = "$($Script:WslDistro) distro is not running." }
    }
}

# ---------------------------------------------------------------------------
# Check: Process liveness (WSL-based)
# ---------------------------------------------------------------------------
function Test-WSLProcess {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ProcessPattern,
        [string]$Category = 'core_stack'
    )
    return New-CheckResult -Name $Name -Category $Category -Check {
        $result = & wsl.exe -d $Script:WslDistro -- bash -c "pgrep -f '$ProcessPattern' > /dev/null 2>&1; echo `$?" 2>&1
        $exitCode = ($result | Select-Object -Last 1).Trim()
        if ($exitCode -eq '0') {
            return @{ Status = 'healthy'; Message = "Process '$ProcessPattern' is alive." }
        }
        return @{ Status = 'critical'; Message = "Process '$ProcessPattern' not found." }
    }
}

# ---------------------------------------------------------------------------
# Check: Heartbeat freshness
# ---------------------------------------------------------------------------
function Test-Heartbeat {
    return New-CheckResult -Name 'heartbeat_freshness' -Category 'operations' -Check {
        if (-not (Test-Path $Script:HeartbeatFile)) {
            return @{ Status = 'warning'; Message = 'Heartbeat file does not exist.' }
        }
        $hb = Get-Content $Script:HeartbeatFile -Raw | ConvertFrom-Json
        $lastBeat = [DateTime]::Parse($hb.lastBeat)
        $ageSec = [Math]::Round(((Get-Date) - $lastBeat).TotalSeconds, 1)
        if ($ageSec -gt $Script:HeartbeatMaxAgeSec) {
            return @{ Status = 'warning'; Message = "Heartbeat stale: ${ageSec}s (threshold: $($Script:HeartbeatMaxAgeSec)s)." }
        }
        return @{ Status = 'healthy'; Message = "Heartbeat fresh: ${ageSec}s." }
    }
}

# ---------------------------------------------------------------------------
# Check: Disk space
# ---------------------------------------------------------------------------
function Test-DiskSpace {
    return New-CheckResult -Name 'disk_space' -Category 'infrastructure' -Check {
        $drive = (Split-Path $Script:StateRoot -Qualifier)
        $disk = Get-PSDrive -Name ($drive -replace ':', '') -ErrorAction Stop
        $freeGB = [Math]::Round($disk.Free / 1GB, 2)
        if ($freeGB -lt 1) {
            return @{ Status = 'critical'; Message = "Only ${freeGB} GB free on $drive." }
        }
        if ($freeGB -lt 5) {
            return @{ Status = 'warning'; Message = "${freeGB} GB free on $drive (low)." }
        }
        return @{ Status = 'healthy'; Message = "${freeGB} GB free on $drive." }
    }
}

# ---------------------------------------------------------------------------
# Check: Scheduled tasks registered
# ---------------------------------------------------------------------------
function Test-ScheduledTasks {
    return New-CheckResult -Name 'scheduled_tasks' -Category 'operations' -Check {
        $tasks = @('\AlwaysClaw\Watchdog', '\AlwaysClaw\DailyMaintenance')
        $missing = @()
        foreach ($tn in $tasks) {
            try {
                $t = Get-ScheduledTask -TaskName ($tn -replace '\\AlwaysClaw\\', '') -TaskPath '\AlwaysClaw\' -ErrorAction Stop
            } catch {
                $missing += $tn
            }
        }
        if ($missing.Count -gt 0) {
            return @{
                Status  = 'warning'
                Message = "Missing tasks: $($missing -join ', ')"
            }
        }
        return @{ Status = 'healthy'; Message = 'All expected scheduled tasks are registered.' }
    }
}

# ---------------------------------------------------------------------------
# Check: Degraded mode
# ---------------------------------------------------------------------------
function Test-DegradedMode {
    return New-CheckResult -Name 'degraded_mode' -Category 'operations' -Check {
        if (-not (Test-Path $Script:DegradedFlag)) {
            return @{ Status = 'healthy'; Message = 'System is not in degraded mode.' }
        }
        $flag = Get-Content $Script:DegradedFlag -Raw | ConvertFrom-Json
        if ($flag.degraded -eq $true) {
            return @{
                Status  = 'critical'
                Message = "DEGRADED since $($flag.since). Component: $($flag.component). Operator ack required."
            }
        }
        return @{ Status = 'healthy'; Message = 'Degraded flag present but degraded=false.' }
    }
}

# ---------------------------------------------------------------------------
# Check: State directory structure
# ---------------------------------------------------------------------------
function Test-StateDirectories {
    return New-CheckResult -Name 'state_directories' -Category 'infrastructure' -Check {
        $requiredDirs = @(
            $Script:StateRoot
            (Join-Path $Script:StateRoot 'config')
            (Join-Path $Script:StateRoot 'logs')
            (Join-Path $Script:StateRoot 'auth')
            (Join-Path $Script:StateRoot 'cron')
            (Join-Path $Script:StateRoot 'agents')
            (Join-Path $Script:StateRoot 'memory')
        )
        $missing = @()
        foreach ($d in $requiredDirs) {
            if (-not (Test-Path $d)) { $missing += $d }
        }
        if ($missing.Count -gt 0) {
            return @{
                Status  = 'warning'
                Message = "Missing directories: $($missing -join ', ')"
            }
        }
        return @{ Status = 'healthy'; Message = 'All state directories present.' }
    }
}

# ---------------------------------------------------------------------------
# Assemble full report
# ---------------------------------------------------------------------------
function Build-HealthReport {
    $checks = @()

    # Platform checks
    $checks += Test-WSL2Availability
    $checks += Test-WSLDistro

    # Core stack
    $checks += Test-WSLProcess -Name 'gateway_process' -ProcessPattern 'openclaw' -Category 'core_stack'
    $checks += Test-WSLProcess -Name 'loopd_process'   -ProcessPattern 'alwaysclaw-loopd' -Category 'core_stack'
    $checks += Test-WSLProcess -Name 'tools_process'   -ProcessPattern 'alwaysclaw-tools' -Category 'core_stack'

    # Sidecars
    $checks += Test-WSLProcess -Name 'voice_process'   -ProcessPattern 'alwaysclaw-voice' -Category 'sidecars'

    # Operations
    $checks += Test-Heartbeat
    $checks += Test-ScheduledTasks
    $checks += Test-DegradedMode

    # Infrastructure
    $checks += Test-DiskSpace
    $checks += Test-StateDirectories

    # Compute overall status
    $statuses = $checks | ForEach-Object { $_.status }
    if ($statuses -contains 'critical') {
        $overall = 'critical'
    } elseif ($statuses -contains 'error') {
        $overall = 'error'
    } elseif ($statuses -contains 'warning') {
        $overall = 'warning'
    } else {
        $overall = 'healthy'
    }

    $report = [PSCustomObject]@{
        reportId     = [Guid]::NewGuid().ToString()
        timestamp    = (Get-Date).ToString('o')
        hostname     = $env:COMPUTERNAME
        overallStatus = $overall
        checks       = $checks
        summary      = [PSCustomObject]@{
            total    = $checks.Count
            healthy  = ($checks | Where-Object { $_.status -eq 'healthy' }).Count
            warning  = ($checks | Where-Object { $_.status -eq 'warning' }).Count
            critical = ($checks | Where-Object { $_.status -eq 'critical' }).Count
            error    = ($checks | Where-Object { $_.status -eq 'error' }).Count
            unknown  = ($checks | Where-Object { $_.status -eq 'unknown' }).Count
        }
    }

    return $report
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try {
    # Ensure log directory exists
    if (-not (Test-Path $Script:LogDir)) {
        New-Item -ItemType Directory -Path $Script:LogDir -Force | Out-Null
    }

    $report = Build-HealthReport
    $json = $report | ConvertTo-Json -Depth 5

    # Write to file
    $json | Set-Content -Path $Script:HealthOutFile -Force

    # Also write to stdout for callers
    Write-Output $json

    # Return non-zero exit code if critical
    if ($report.overallStatus -eq 'critical') {
        exit 2
    }
    exit 0
} catch {
    # Emergency fallback: output minimal JSON even on catastrophic failure
    $emergency = [PSCustomObject]@{
        reportId      = [Guid]::NewGuid().ToString()
        timestamp     = (Get-Date).ToString('o')
        hostname      = $env:COMPUTERNAME
        overallStatus = 'error'
        checks        = @()
        error         = $_.Exception.Message
    }
    $emergency | ConvertTo-Json -Depth 3 | Write-Output
    exit 1
}
