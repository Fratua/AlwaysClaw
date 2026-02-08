# install-service.ps1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Install OpenClaw Agent as a Windows Service using NSSM
.DESCRIPTION
    Downloads NSSM if needed and creates a Windows Service for the OpenClaw Agent
#>

$ServiceName = "OpenClawAgent"
$ServiceDisplayName = "OpenClaw AI Agent"
$ServiceDescription = "24/7 AI Agent System with scheduled task automation"

# Paths
$NodePath = (Get-Command node -ErrorAction SilentlyContinue)?.Source
if (-not $NodePath) {
    $NodePath = "C:\Program Files\nodejs\node.exe"
}

$AppPath = Split-Path -Parent $PSScriptRoot
$ScriptPath = Join-Path $AppPath "dist\index.js"
$LogPath = Join-Path $AppPath "logs"
$NssmPath = Join-Path $AppPath "tools\nssm\win64\nssm.exe"

# Ensure log directory exists
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

Write-Host "OpenClaw Service Installer" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Gray
Write-Host "  Service Name: $ServiceName" -ForegroundColor Gray
Write-Host "  Node Path: $NodePath" -ForegroundColor Gray
Write-Host "  Script Path: $ScriptPath" -ForegroundColor Gray
Write-Host "  Working Directory: $AppPath" -ForegroundColor Gray
Write-Host "  Log Directory: $LogPath" -ForegroundColor Gray
Write-Host ""

# Check if Node.js exists
if (-not (Test-Path $NodePath)) {
    Write-Error "Node.js not found at $NodePath. Please install Node.js or update the path."
    exit 1
}

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found at $ScriptPath. Please run 'npm run build' first."
    exit 1
}

# Download NSSM if not exists
if (-not (Test-Path $NssmPath)) {
    Write-Host "NSSM not found. Downloading..." -ForegroundColor Yellow
    
    $ToolsPath = Join-Path $AppPath "tools"
    $NssmZip = Join-Path $env:TEMP "nssm.zip"
    $NssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    
    try {
        # Download NSSM
        Invoke-WebRequest -Uri $NssmUrl -OutFile $NssmZip -UseBasicParsing
        
        # Extract
        Expand-Archive -Path $NssmZip -DestinationPath $ToolsPath -Force
        
        # Move to expected location
        $ExtractedNssm = Join-Path $ToolsPath "nssm-2.24\win64\nssm.exe"
        $NssmDir = Split-Path -Parent $NssmPath
        if (-not (Test-Path $NssmDir)) {
            New-Item -ItemType Directory -Path $NssmDir -Force | Out-Null
        }
        Move-Item -Path $ExtractedNssm -Destination $NssmPath -Force
        
        # Cleanup
        Remove-Item -Path $NssmZip -Force
        Remove-Item -Path (Join-Path $ToolsPath "nssm-2.24") -Recurse -Force
        
        Write-Host "NSSM downloaded successfully" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to download NSSM: $_"
        exit 1
    }
}

# Check if service already exists
$existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existingService) {
    Write-Host "Service already exists. Stopping and removing..." -ForegroundColor Yellow
    Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    & $NssmPath remove $ServiceName confirm
}

# Install service
Write-Host "Installing service..." -ForegroundColor Cyan

& $NssmPath install $ServiceName $NodePath
& $NssmPath set $ServiceName Application $NodePath
& $NssmPath set $ServiceName AppParameters "`"$ScriptPath`""
& $NssmPath set $ServiceName AppDirectory $AppPath
& $NssmPath set $ServiceName AppEnvironmentExtra "NODE_ENV=production"
& $NssmPath set $ServiceName DisplayName $ServiceDisplayName
& $NssmPath set $ServiceName Description $ServiceDescription
& $NssmPath set $ServiceName Start SERVICE_AUTO_START

# Configure logging
& $NssmPath set $ServiceName AppStdout (Join-Path $LogPath "service.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $LogPath "service-error.log")
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateOnline 1
& $NssmPath set $ServiceName AppRotateSeconds 86400

# Configure process
& $NssmPath set $ServiceName AppThrottle 0
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 5000

# Configure service recovery
& $NssmPath set $ServiceName ObjectName "NT AUTHORITY\LocalService"

Write-Host "Service installed successfully!" -ForegroundColor Green
Write-Host ""

# Start service
Write-Host "Starting service..." -ForegroundColor Cyan
Start-Service -Name $ServiceName

# Wait a moment and check status
Start-Sleep -Seconds 2
$service = Get-Service -Name $ServiceName

Write-Host ""
Write-Host "Service Status: $($service.Status)" -ForegroundColor $(if ($service.Status -eq 'Running') { 'Green' } else { 'Red' })
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Gray
Write-Host "  View logs: Get-Content '$LogPath\service.log' -Tail 50 -Wait" -ForegroundColor Gray
Write-Host "  Stop service: net stop $ServiceName" -ForegroundColor Gray
Write-Host "  Start service: net start $ServiceName" -ForegroundColor Gray
Write-Host "  Remove service: $NssmPath remove $ServiceName confirm" -ForegroundColor Gray
