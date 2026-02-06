#Requires -RunAsAdministrator
<#
.SYNOPSIS
    AlwaysClaw WSL2 bootstrap script for Windows 10.

.DESCRIPTION
    Installs and configures WSL2 with Ubuntu, enables systemd, installs the
    AlwaysClaw gateway service, and registers Windows Task Scheduler jobs for
    boot, watchdog, and daily maintenance.

    Designed to be idempotent: safe to re-run after partial failures.

.NOTES
    Platform  : Windows 10 21H2+
    Requires  : Administrator privileges
    Reference : Master Plan Sections 4, 16, 28, 35, 44
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$Script:StateRoot        = 'C:\AlwaysClaw\state'
$Script:LogDir           = Join-Path $Script:StateRoot 'logs'
$Script:LogFile          = Join-Path $Script:LogDir    "wsl2-setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$Script:WslDistro        = 'Ubuntu'
$Script:WslConfPath      = '/etc/wsl.conf'
$Script:DeploymentDir    = $PSScriptRoot   # directory containing this script

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
function Write-Log {
    param(
        [Parameter(Mandatory)][string]$Message,
        [ValidateSet('INFO','WARN','ERROR')][string]$Level = 'INFO'
    )
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] [$Level] $Message"
    Write-Host $line
    if (Test-Path (Split-Path $Script:LogFile -Parent)) {
        Add-Content -Path $Script:LogFile -Value $line -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
function Test-Prerequisites {
    Write-Log 'Running pre-flight checks...'

    # Windows version check (WSL2 requires build >= 19041)
    $build = [System.Environment]::OSVersion.Version.Build
    if ($build -lt 19041) {
        throw "Windows build $build is below the minimum 19041 required for WSL2."
    }
    Write-Log "Windows build $build meets WSL2 requirement."

    # Ensure state directory tree exists
    $dirs = @(
        $Script:StateRoot
        (Join-Path $Script:StateRoot 'config')
        (Join-Path $Script:StateRoot 'logs')
        (Join-Path $Script:StateRoot 'auth')
        (Join-Path $Script:StateRoot 'cron')
        (Join-Path $Script:StateRoot 'memory')
        (Join-Path $Script:StateRoot 'memory\vector')
        (Join-Path $Script:StateRoot 'agents')
    )
    foreach ($d in $dirs) {
        if (-not (Test-Path $d)) {
            New-Item -ItemType Directory -Path $d -Force | Out-Null
            Write-Log "Created directory: $d"
        }
    }
}

# ---------------------------------------------------------------------------
# WSL2 installation
# ---------------------------------------------------------------------------
function Install-WSL2 {
    Write-Log 'Checking WSL installation status...'

    $wslExe = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wslExe) {
        Write-Log 'WSL not found. Installing WSL (this may take several minutes)...'
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList '--install','--no-launch' `
                    -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            throw "wsl --install failed with exit code $($proc.ExitCode). A reboot may be required."
        }
        Write-Log 'WSL installed successfully. A reboot may be required before first distro launch.'
    } else {
        Write-Log 'WSL is already installed.'
    }

    # Ensure WSL2 is the default version
    Write-Log 'Setting WSL default version to 2...'
    & wsl.exe --set-default-version 2 2>&1 | ForEach-Object { Write-Log $_ }

    # Install Ubuntu distro if absent
    $distros = & wsl.exe --list --quiet 2>&1
    if ($distros -notmatch $Script:WslDistro) {
        Write-Log "Distro '$($Script:WslDistro)' not found. Installing..."
        & wsl.exe --install -d $Script:WslDistro 2>&1 | ForEach-Object { Write-Log $_ }
        Write-Log "Distro '$($Script:WslDistro)' installation initiated."
    } else {
        Write-Log "Distro '$($Script:WslDistro)' is already present."
    }
}

# ---------------------------------------------------------------------------
# Enable systemd inside WSL distro
# ---------------------------------------------------------------------------
function Enable-Systemd {
    Write-Log 'Configuring systemd inside WSL distro...'

    $wslConf = @"
[boot]
systemd=true

[automount]
enabled=true
options="metadata"

[interop]
enabled=true
appendWindowsPath=true
"@

    # Write config via wsl.exe
    $escapedConf = $wslConf -replace '"', '\"'
    $checkCmd = "cat $($Script:WslConfPath) 2>/dev/null || echo '__MISSING__'"
    $existing = & wsl.exe -d $Script:WslDistro -- bash -c $checkCmd 2>&1

    if ($existing -match 'systemd=true') {
        Write-Log 'systemd is already enabled in wsl.conf.'
    } else {
        Write-Log 'Writing /etc/wsl.conf with systemd=true...'
        $writeCmd = "echo '$($wslConf -replace "'","'\''")' | sudo tee $($Script:WslConfPath) > /dev/null"
        & wsl.exe -d $Script:WslDistro -- bash -c $writeCmd 2>&1 | ForEach-Object { Write-Log $_ }
        Write-Log 'wsl.conf updated. Restarting WSL to apply...'
        & wsl.exe --shutdown 2>&1 | ForEach-Object { Write-Log $_ }
        Start-Sleep -Seconds 5
        Write-Log 'WSL restarted.'
    }
}

# ---------------------------------------------------------------------------
# Install / repair AlwaysClaw gateway service inside WSL
# ---------------------------------------------------------------------------
function Install-GatewayService {
    Write-Log 'Installing/repairing AlwaysClaw gateway service in WSL...'

    # Run OpenClaw-compatible install commands; tolerate individual failures
    $commands = @(
        'openclaw onboard --install-daemon 2>&1 || true'
        'openclaw gateway install 2>&1 || true'
        'openclaw doctor 2>&1 || true'
    )

    foreach ($cmd in $commands) {
        Write-Log "Running inside WSL: $cmd"
        $output = & wsl.exe -d $Script:WslDistro -- bash -c $cmd 2>&1
        $output | ForEach-Object { Write-Log "  $_" }
    }

    Write-Log 'Gateway service install/repair sequence complete.'
}

# ---------------------------------------------------------------------------
# Register Windows Scheduled Tasks
# ---------------------------------------------------------------------------
function Register-ScheduledTasks {
    Write-Log 'Registering Windows Scheduled Tasks...'

    $taskXmlPath = Join-Path $Script:DeploymentDir 'boot-task.xml'
    if (-not (Test-Path $taskXmlPath)) {
        Write-Log "boot-task.xml not found at $taskXmlPath -- skipping XML import." 'WARN'
    } else {
        # Import the composite task definition
        Write-Log "Importing task definition from $taskXmlPath..."
        try {
            Register-ScheduledTask -Xml (Get-Content $taskXmlPath -Raw) `
                -TaskName 'AlwaysClaw-Bootstrap' `
                -TaskPath '\AlwaysClaw\' `
                -Force | Out-Null
            Write-Log 'Task AlwaysClaw-Bootstrap registered.'
        } catch {
            Write-Log "Failed to register bootstrap task: $_" 'WARN'
        }
    }

    # Also register via schtasks for create/query/run automation (Section 44)
    $watchdogScript = Join-Path $Script:DeploymentDir 'watchdog.ps1'
    $healthScript   = Join-Path $Script:DeploymentDir 'health-check.ps1'

    # Watchdog - every 5 minutes
    $schtasksArgs = @(
        '/Create', '/F',
        '/TN', '\AlwaysClaw\Watchdog',
        '/SC', 'MINUTE', '/MO', '5',
        '/TR', "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$watchdogScript`"",
        '/RL', 'HIGHEST',
        '/RU', 'SYSTEM'
    )
    Write-Log "Creating schtask: \AlwaysClaw\Watchdog"
    & schtasks.exe @schtasksArgs 2>&1 | ForEach-Object { Write-Log "  $_" }

    # Daily maintenance at 03:00
    $maintArgs = @(
        '/Create', '/F',
        '/TN', '\AlwaysClaw\DailyMaintenance',
        '/SC', 'DAILY', '/ST', '03:00',
        '/TR', "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `"& { wsl.exe -d Ubuntu -- bash -c 'openclaw maintenance --log-compact --archive 2>&1' }`"",
        '/RL', 'HIGHEST',
        '/RU', 'SYSTEM'
    )
    Write-Log "Creating schtask: \AlwaysClaw\DailyMaintenance"
    & schtasks.exe @maintArgs 2>&1 | ForEach-Object { Write-Log "  $_" }

    # Query all AlwaysClaw tasks for audit
    Write-Log 'Verifying registered tasks...'
    & schtasks.exe /Query /TN '\AlwaysClaw\' /FO LIST 2>&1 | ForEach-Object { Write-Log "  $_" }
}

# ---------------------------------------------------------------------------
# Post-install health verification
# ---------------------------------------------------------------------------
function Test-PostInstall {
    Write-Log 'Running post-install verification...'

    # Check WSL2 is operational
    $wslStatus = & wsl.exe --status 2>&1
    $wslStatus | ForEach-Object { Write-Log "  $_" }

    # Run health check if present
    $healthScript = Join-Path $Script:DeploymentDir 'health-check.ps1'
    if (Test-Path $healthScript) {
        Write-Log 'Running health check...'
        try {
            $result = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $healthScript 2>&1
            $result | ForEach-Object { Write-Log "  $_" }
        } catch {
            Write-Log "Health check returned errors: $_" 'WARN'
        }
    }

    Write-Log 'Post-install verification complete.'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Main {
    Write-Log '====== AlwaysClaw WSL2 Setup BEGIN ======'
    try {
        Test-Prerequisites
        Install-WSL2
        Enable-Systemd
        Install-GatewayService
        Register-ScheduledTasks
        Test-PostInstall
        Write-Log '====== AlwaysClaw WSL2 Setup COMPLETE ======'
    } catch {
        Write-Log "FATAL: $($_.Exception.Message)" 'ERROR'
        Write-Log "Stack: $($_.ScriptStackTrace)" 'ERROR'
        exit 1
    }
}

Main
