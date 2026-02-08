# OpenClaw AI Agent Service - Windows Deployment Script
# Deploys the agent as a Windows Service for 24/7 operation

#Requires -RunAsAdministrator

param(
    [string]$ServiceName = "OpenClawAgent",
    [string]$DisplayName = "OpenClaw AI Agent Service",
    [string]$Description = "24/7 AI agent service with GPT-5.2 integration",
    [string]$InstallPath = "C:\OpenClaw",
    [string]$PythonPath = "python",
    [string]$ConfigPath = "$PSScriptRoot\..\config\scaling_config.yaml",
    [string]$LogPath = "$env:ProgramData\OpenClaw\Logs",
    [string]$ServiceAccount = "NT AUTHORITY\NETWORK SERVICE",
    [switch]$Uninstall,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status
)

# Error handling
$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    
    # Also write to log file
    $logDir = Split-Path $LogPath -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    Add-Content -Path "$LogPath\deploy.log" -Value $logMessage
}

function Test-Prerequisites {
    Write-Log "Checking prerequisites..."
    
    # Check Python
    try {
        $pythonVersion = & $PythonPath --version 2>&1
        Write-Log "Python found: $pythonVersion"
    }
    catch {
        throw "Python not found. Please install Python 3.11 or higher."
    }
    
    # Check if running as administrator
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must be run as Administrator"
    }
    
    Write-Log "Prerequisites check passed"
}

function Install-Service {
    Write-Log "Installing OpenClaw Agent Service..."
    
    # Create installation directory
    if (-not (Test-Path $InstallPath)) {
        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        Write-Log "Created installation directory: $InstallPath"
    }
    
    # Create log directory
    if (-not (Test-Path $LogPath)) {
        New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
        Write-Log "Created log directory: $LogPath"
    }
    
    # Copy application files
    $sourcePath = Join-Path $PSScriptRoot ".."
    Write-Log "Copying application files from $sourcePath to $InstallPath..."
    
    Copy-Item -Path "$sourcePath\scaling_modules\*.py" -Destination $InstallPath -Force
    Copy-Item -Path "$sourcePath\config\*.yaml" -Destination "$InstallPath\config" -Force -Recurse
    
    # Create Python virtual environment
    $venvPath = Join-Path $InstallPath "venv"
    if (-not (Test-Path $venvPath)) {
        Write-Log "Creating Python virtual environment..."
        & $PythonPath -m venv $venvPath
    }
    
    # Install dependencies
    Write-Log "Installing Python dependencies..."
    $pipPath = Join-Path $venvPath "Scripts\pip.exe"
    & $pipPath install --upgrade pip
    & $pipPath install -r (Join-Path $sourcePath "requirements.txt")
    
    # Create service wrapper script
    $serviceScript = @"
import sys
import os
import logging
from pathlib import Path

# Add install path to Python path
sys.path.insert(0, r'$InstallPath')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'$LogPath\service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('openclaw.service')

# Import and run service
from scaling_modules.agent_service import AgentService

if __name__ == '__main__':
    service = AgentService(config_path=r'$ConfigPath')
    service.run()
"@
    
    $serviceScriptPath = Join-Path $InstallPath "service_runner.py"
    $serviceScript | Out-File -FilePath $serviceScriptPath -Encoding UTF8
    
    # Create Windows Service
    $pythonExe = Join-Path $venvPath "Scripts\python.exe"
    $binaryPath = "`"$pythonExe`" `"$serviceScriptPath`""
    
    Write-Log "Creating Windows Service: $ServiceName"
    
    # Remove existing service if present
    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-Log "Removing existing service..."
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        sc.exe delete $ServiceName | Out-Null
        Start-Sleep -Seconds 2
    }
    
    # Create new service
    New-Service -Name $ServiceName `
        -BinaryPathName $binaryPath `
        -DisplayName $DisplayName `
        -Description $Description `
        -StartupType Automatic `
        -Credential $ServiceAccount | Out-Null
    
    # Configure service recovery
    Write-Log "Configuring service recovery..."
    sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/10000/restart/30000 | Out-Null
    
    # Set service dependencies
    # sc.exe config $ServiceName depend= Tcpip/Afd | Out-Null
    
    Write-Log "Service installed successfully"
    
    # Create firewall rule
    Write-Log "Creating firewall rule..."
    $firewallRule = Get-NetFirewallRule -DisplayName "OpenClaw Agent" -ErrorAction SilentlyContinue
    if (-not $firewallRule) {
        New-NetFirewallRule -DisplayName "OpenClaw Agent" `
            -Direction Inbound `
            -Protocol TCP `
            -LocalPort 8080 `
            -Action Allow | Out-Null
        Write-Log "Firewall rule created"
    }
}

function Uninstall-Service {
    Write-Log "Uninstalling OpenClaw Agent Service..."
    
    # Stop service
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -eq "Running") {
            Write-Log "Stopping service..."
            Stop-Service -Name $ServiceName -Force
        }
        
        Write-Log "Removing service..."
        sc.exe delete $ServiceName | Out-Null
    }
    
    # Remove firewall rule
    $firewallRule = Get-NetFirewallRule -DisplayName "OpenClaw Agent" -ErrorAction SilentlyContinue
    if ($firewallRule) {
        Remove-NetFirewallRule -DisplayName "OpenClaw Agent"
        Write-Log "Firewall rule removed"
    }
    
    # Optionally remove installation directory
    # Remove-Item -Path $InstallPath -Recurse -Force
    
    Write-Log "Service uninstalled successfully"
}

function Start-AgentService {
    Write-Log "Starting service: $ServiceName"
    Start-Service -Name $ServiceName
    
    # Wait for service to start
    $timeout = 30
    $timer = [Diagnostics.Stopwatch]::StartNew()
    
    while ($timer.Elapsed.TotalSeconds -lt $timeout) {
        $service = Get-Service -Name $ServiceName
        if ($service.Status -eq "Running") {
            Write-Log "Service started successfully"
            return
        }
        Start-Sleep -Seconds 1
    }
    
    throw "Service failed to start within $timeout seconds"
}

function Stop-AgentService {
    Write-Log "Stopping service: $ServiceName"
    Stop-Service -Name $ServiceName -Force
    
    # Wait for service to stop
    $timeout = 30
    $timer = [Diagnostics.Stopwatch]::StartNew()
    
    while ($timer.Elapsed.TotalSeconds -lt $timeout) {
        $service = Get-Service -Name $ServiceName
        if ($service.Status -eq "Stopped") {
            Write-Log "Service stopped successfully"
            return
        }
        Start-Sleep -Seconds 1
    }
    
    throw "Service failed to stop within $timeout seconds"
}

function Get-ServiceStatus {
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    
    if (-not $service) {
        Write-Log "Service not found: $ServiceName" -Level "WARNING"
        return
    }
    
    Write-Log "Service Status: $($service.Status)"
    Write-Log "Service StartType: $($service.StartType)"
    
    # Get process info if running
    if ($service.Status -eq "Running") {
        $process = Get-Process -Name "python" | Where-Object {
            $_.CommandLine -like "*$ServiceName*"
        } | Select-Object -First 1
        
        if ($process) {
            Write-Log "Process ID: $($process.Id)"
            Write-Log "CPU Usage: $($process.CPU)"
            Write-Log "Memory: $([math]::Round($process.WorkingSet64 / 1MB, 2)) MB"
        }
    }
    
    # Check recent log entries
    $logFile = "$LogPath\service.log"
    if (Test-Path $logFile) {
        Write-Log "`nRecent log entries:"
        Get-Content $logFile -Tail 10 | ForEach-Object {
            Write-Log "  $_"
        }
    }
}

# Main execution
try {
    if ($Uninstall) {
        Uninstall-Service
    }
    elseif ($Start) {
        Start-AgentService
    }
    elseif ($Stop) {
        Stop-AgentService
    }
    elseif ($Restart) {
        Stop-AgentService
        Start-Sleep -Seconds 2
        Start-AgentService
    }
    elseif ($Status) {
        Get-ServiceStatus
    }
    else {
        # Default: Install service
        Test-Prerequisites
        Install-Service
        Start-AgentService
    }
    
    Write-Log "Operation completed successfully"
}
catch {
    Write-Log "ERROR: $_" -Level "ERROR"
    Write-Log $_.ScriptStackTrace -Level "ERROR"
    exit 1
}
