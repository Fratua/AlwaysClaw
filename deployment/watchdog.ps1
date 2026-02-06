<#
.SYNOPSIS
    AlwaysClaw watchdog with exponential backoff and circuit-breaker restart policy.

.DESCRIPTION
    Runs every 5 minutes (via Task Scheduler). For each monitored component it:
      1. Checks heartbeat freshness (last update within threshold).
      2. Checks process liveness.
      3. Restarts failed components with exponential backoff.
      4. Enforces a circuit breaker: max 5 restart attempts within 15 minutes
         before entering degraded mode.

    Degraded mode disables tier-2 tools, keeps read-only + communications alive,
    and emits an incident event requiring operator acknowledgment.

.NOTES
    Reference : Master Plan Sections 28, 35, 44
    Schedule  : Every 5 minutes via schtasks or boot-task.xml
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$Script:StateRoot        = 'C:\AlwaysClaw\state'
$Script:LogDir           = Join-Path $Script:StateRoot 'logs'
$Script:LogFile          = Join-Path $Script:LogDir    'watchdog.log'
$Script:RestartStateFile = Join-Path $Script:StateRoot 'watchdog-restart-state.json'
$Script:IncidentDir      = Join-Path $Script:StateRoot 'incidents'
$Script:HeartbeatFile    = Join-Path $Script:StateRoot 'heartbeat.json'

# Thresholds
$Script:HeartbeatMaxAgeSec  = 600      # 10 minutes: heartbeat stale threshold
$Script:CircuitWindowSec    = 900      # 15 minutes: sliding window for restart counting
$Script:MaxRestartsInWindow = 5        # trips circuit breaker
$Script:BaseBackoffSec      = 10       # first retry delay
$Script:MaxBackoffSec       = 300      # cap at 5 minutes
$Script:BackoffMultiplier   = 2        # exponential factor

# Components to monitor (name => check/restart config)
$Script:Components = @(
    @{
        Name            = 'alwaysclaw-gateway'
        Type            = 'wsl'
        ProcessPattern  = 'openclaw'
        CheckCmd        = 'pgrep -f openclaw'
        RestartCmd      = 'openclaw gateway start'
        Tier            = 'core'
    }
    @{
        Name            = 'alwaysclaw-loopd'
        Type            = 'wsl'
        ProcessPattern  = 'alwaysclaw-loopd'
        CheckCmd        = 'pgrep -f alwaysclaw-loopd'
        RestartCmd      = 'systemctl restart alwaysclaw-loopd'
        Tier            = 'core'
    }
    @{
        Name            = 'alwaysclaw-tools'
        Type            = 'wsl'
        ProcessPattern  = 'alwaysclaw-tools'
        CheckCmd        = 'pgrep -f alwaysclaw-tools'
        RestartCmd      = 'systemctl restart alwaysclaw-tools'
        Tier            = 'tier2'
    }
    @{
        Name            = 'alwaysclaw-voice'
        Type            = 'wsl'
        ProcessPattern  = 'alwaysclaw-voice'
        CheckCmd        = 'pgrep -f alwaysclaw-voice'
        RestartCmd      = 'systemctl restart alwaysclaw-voice'
        Tier            = 'tier2'
    }
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
function Write-Log {
    param(
        [Parameter(Mandatory)][string]$Message,
        [ValidateSet('INFO','WARN','ERROR','INCIDENT')][string]$Level = 'INFO'
    )
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] [$Level] $Message"
    Write-Host $line
    if (-not (Test-Path $Script:LogDir)) {
        New-Item -ItemType Directory -Path $Script:LogDir -Force | Out-Null
    }
    Add-Content -Path $Script:LogFile -Value $line -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Restart state persistence (tracks attempts per component)
# ---------------------------------------------------------------------------
function Get-RestartState {
    if (Test-Path $Script:RestartStateFile) {
        try {
            return (Get-Content $Script:RestartStateFile -Raw | ConvertFrom-Json)
        } catch {
            Write-Log "Corrupt restart state file; resetting." 'WARN'
        }
    }
    return @{}
}

function Save-RestartState {
    param([Parameter(Mandatory)]$State)
    $State | ConvertTo-Json -Depth 5 | Set-Content -Path $Script:RestartStateFile -Force
}

function Get-ComponentState {
    param(
        [Parameter(Mandatory)][string]$ComponentName,
        [Parameter(Mandatory)]$AllState
    )
    if ($AllState.PSObject.Properties.Name -contains $ComponentName) {
        return $AllState.$ComponentName
    }
    # Initialize fresh state
    $fresh = [PSCustomObject]@{
        Attempts       = @()
        ConsecutiveFails = 0
        Degraded       = $false
        LastBackoffSec = 0
    }
    $AllState | Add-Member -NotePropertyName $ComponentName -NotePropertyValue $fresh -Force
    return $fresh
}

# ---------------------------------------------------------------------------
# Heartbeat freshness check
# ---------------------------------------------------------------------------
function Test-HeartbeatFreshness {
    if (-not (Test-Path $Script:HeartbeatFile)) {
        Write-Log 'Heartbeat file not found; treating as stale.' 'WARN'
        return $false
    }
    try {
        $hb = Get-Content $Script:HeartbeatFile -Raw | ConvertFrom-Json
        $lastBeat = [DateTime]::Parse($hb.lastBeat)
        $ageSec = (Get-Date) - $lastBeat | Select-Object -ExpandProperty TotalSeconds
        if ($ageSec -gt $Script:HeartbeatMaxAgeSec) {
            Write-Log "Heartbeat stale: ${ageSec}s old (threshold: $($Script:HeartbeatMaxAgeSec)s)." 'WARN'
            return $false
        }
        Write-Log "Heartbeat fresh: ${ageSec}s old."
        return $true
    } catch {
        Write-Log "Failed to parse heartbeat file: $_" 'WARN'
        return $false
    }
}

# ---------------------------------------------------------------------------
# Component liveness check
# ---------------------------------------------------------------------------
function Test-ComponentLiveness {
    param([Parameter(Mandatory)][hashtable]$Component)

    $name = $Component.Name
    try {
        if ($Component.Type -eq 'wsl') {
            $result = & wsl.exe -d Ubuntu -- bash -c "$($Component.CheckCmd) 2>/dev/null; echo EXIT_CODE=`$?" 2>&1
            $exitLine = ($result | Where-Object { $_ -match 'EXIT_CODE=' }) -replace 'EXIT_CODE=', ''
            $alive = ($exitLine -eq '0')
        } else {
            # Native Windows process check
            $proc = Get-Process -Name $Component.ProcessPattern -ErrorAction SilentlyContinue
            $alive = ($null -ne $proc)
        }
    } catch {
        Write-Log "Error checking liveness for ${name}: $_" 'WARN'
        $alive = $false
    }

    if ($alive) {
        Write-Log "Component '$name' is alive."
    } else {
        Write-Log "Component '$name' is NOT alive." 'WARN'
    }
    return $alive
}

# ---------------------------------------------------------------------------
# Exponential backoff calculation
# ---------------------------------------------------------------------------
function Get-NextBackoff {
    param([int]$LastBackoffSec)
    if ($LastBackoffSec -le 0) {
        return $Script:BaseBackoffSec
    }
    $next = [Math]::Min($LastBackoffSec * $Script:BackoffMultiplier, $Script:MaxBackoffSec)
    return $next
}

# ---------------------------------------------------------------------------
# Circuit breaker: check if too many restarts in window
# ---------------------------------------------------------------------------
function Test-CircuitBroken {
    param([Parameter(Mandatory)]$CompState)

    $now = Get-Date
    $windowStart = $now.AddSeconds(-$Script:CircuitWindowSec)

    # Filter attempts within the sliding window
    $recentAttempts = @()
    foreach ($ts in $CompState.Attempts) {
        try {
            $dt = [DateTime]::Parse($ts)
            if ($dt -ge $windowStart) {
                $recentAttempts += $ts
            }
        } catch { }
    }

    # Update attempts list to only keep recent ones
    $CompState.Attempts = $recentAttempts

    return ($recentAttempts.Count -ge $Script:MaxRestartsInWindow)
}

# ---------------------------------------------------------------------------
# Restart a component
# ---------------------------------------------------------------------------
function Restart-Component {
    param(
        [Parameter(Mandatory)][hashtable]$Component,
        [Parameter(Mandatory)]$CompState
    )

    $name = $Component.Name

    # Calculate backoff
    $backoff = Get-NextBackoff -LastBackoffSec $CompState.LastBackoffSec
    $CompState.LastBackoffSec = $backoff

    Write-Log "Restarting '$name' after ${backoff}s backoff (attempt #$($CompState.Attempts.Count + 1))..."
    Start-Sleep -Seconds $backoff

    # Record attempt timestamp
    $CompState.Attempts += (Get-Date).ToString('o')

    try {
        if ($Component.Type -eq 'wsl') {
            $output = & wsl.exe -d Ubuntu -- bash -c "$($Component.RestartCmd) 2>&1" 2>&1
            $output | ForEach-Object { Write-Log "  $_" }
        } else {
            Start-Process -FilePath $Component.RestartCmd -Wait -NoNewWindow -ErrorAction Stop
        }
        Write-Log "Restart command for '$name' issued."
        $CompState.ConsecutiveFails = $CompState.ConsecutiveFails + 1
    } catch {
        Write-Log "Restart of '$name' failed: $_" 'ERROR'
        $CompState.ConsecutiveFails = $CompState.ConsecutiveFails + 1
    }
}

# ---------------------------------------------------------------------------
# Enter degraded mode for a component
# ---------------------------------------------------------------------------
function Enter-DegradedMode {
    param(
        [Parameter(Mandatory)][hashtable]$Component,
        [Parameter(Mandatory)]$CompState
    )

    $name = $Component.Name
    $CompState.Degraded = $true

    Write-Log "CIRCUIT BREAKER TRIPPED for '$name'. Entering DEGRADED MODE." 'INCIDENT'

    # Emit incident event
    $incident = [PSCustomObject]@{
        timestamp   = (Get-Date).ToString('o')
        component   = $name
        tier        = $Component.Tier
        event       = 'circuit_breaker_tripped'
        action      = 'degraded_mode_entered'
        details     = "Max restart attempts ($($Script:MaxRestartsInWindow)) reached within $($Script:CircuitWindowSec)s window."
        resolution  = 'Tier-2 tools disabled. Read-only + comms channels remain alive. Operator acknowledgment required.'
    }

    if (-not (Test-Path $Script:IncidentDir)) {
        New-Item -ItemType Directory -Path $Script:IncidentDir -Force | Out-Null
    }
    $incidentFile = Join-Path $Script:IncidentDir "incident_$(Get-Date -Format 'yyyyMMdd_HHmmss')_${name}.json"
    $incident | ConvertTo-Json -Depth 3 | Set-Content -Path $incidentFile -Force
    Write-Log "Incident event written to $incidentFile" 'INCIDENT'

    # Disable tier-2 tools by writing a degraded-mode flag
    $degradedFlag = Join-Path $Script:StateRoot 'degraded-mode.json'
    $degradedState = [PSCustomObject]@{
        degraded           = $true
        since              = (Get-Date).ToString('o')
        disabledTier       = 'tier2'
        activeCapabilities = @('read-only', 'communications')
        component          = $name
        requiresAck        = $true
    }
    $degradedState | ConvertTo-Json -Depth 3 | Set-Content -Path $degradedFlag -Force
    Write-Log "Degraded mode flag written to $degradedFlag" 'INCIDENT'
}

# ---------------------------------------------------------------------------
# Main watchdog cycle
# ---------------------------------------------------------------------------
function Invoke-WatchdogCycle {
    Write-Log '====== Watchdog cycle START ======'

    # Load persisted restart state
    $restartState = Get-RestartState

    # Check heartbeat freshness
    $heartbeatOk = Test-HeartbeatFreshness

    foreach ($comp in $Script:Components) {
        $name = $comp.Name
        $compState = Get-ComponentState -ComponentName $name -AllState $restartState

        # Skip if already in degraded mode (requires operator ack to reset)
        if ($compState.Degraded -eq $true) {
            Write-Log "Component '$name' is in degraded mode. Skipping (awaiting operator ack)." 'WARN'
            continue
        }

        $alive = Test-ComponentLiveness -Component $comp

        if ($alive) {
            # Reset consecutive failure counter on success
            $compState.ConsecutiveFails = 0
            $compState.LastBackoffSec = 0
            continue
        }

        # Component is down -- check circuit breaker first
        if (Test-CircuitBroken -CompState $compState) {
            Enter-DegradedMode -Component $comp -CompState $compState
            continue
        }

        # Attempt restart with exponential backoff
        Restart-Component -Component $comp -CompState $compState
    }

    # Persist updated state
    Save-RestartState -State $restartState

    Write-Log '====== Watchdog cycle END ======'
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
try {
    Invoke-WatchdogCycle
} catch {
    Write-Log "Unhandled error in watchdog: $($_.Exception.Message)" 'ERROR'
    Write-Log "Stack: $($_.ScriptStackTrace)" 'ERROR'
    exit 1
}
