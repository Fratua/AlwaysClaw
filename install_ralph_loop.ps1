# Ralph Loop Installation Script for Windows 10
# Multi-Layered Background Processing with Priority Queuing
# OpenClaw AI Agent System

param(
    [string]$InstallPath = "C:\OpenClaw\RalphLoop",
    [string]$ConfigPath = "C:\OpenClaw\Config",
    [string]$DataPath = "C:\OpenClaw\Data",
    [string]$LogPath = "C:\OpenClaw\Logs",
    [switch]$CreateService = $true,
    [switch]$StartService = $true,
    [switch]$Force = $false
)

# Requires Administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script requires Administrator privileges. Please run as Administrator."
    exit 1
}

# Script variables
$Script:ServiceName = "RalphLoop"
$Script:ServiceDisplayName = "OpenClaw Ralph Loop Service"
$Script:PythonMinVersion = [Version]"3.9.0"

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-PythonInstallation {
    Write-Step "Checking Python installation..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+\.\d+\.\d+)") {
            $installedVersion = [Version]$matches[1]
            if ($installedVersion -ge $Script:PythonMinVersion) {
                Write-Success "Python $installedVersion found (>= $Script:PythonMinVersion)"
                return $true
            } else {
                Write-Error "Python version $installedVersion is too old. Required: >= $Script:PythonMinVersion"
                return $false
            }
        }
    } catch {
        Write-Error "Python is not installed or not in PATH"
        return $false
    }
}

function Install-PythonDependencies {
    Write-Step "Installing Python dependencies..."
    
    $requirements = @(
        "asyncio",
        "pyyaml",
        "psutil",
        "pywin32"
    )
    
    foreach ($package in $requirements) {
        try {
            Write-Host "  Installing $package..." -NoNewline
            pip install $package -q
            Write-Host " OK" -ForegroundColor Green
        } catch {
            Write-Error "Failed to install $package"
            throw
        }
    }
    
    Write-Success "All dependencies installed"
}

function New-DirectoryStructure {
    Write-Step "Creating directory structure..."
    
    $directories = @(
        $InstallPath,
        "$InstallPath\src",
        "$InstallPath\config",
        $ConfigPath,
        $DataPath,
        "$DataPath\queue",
        "$DataPath\checkpoints",
        "$DataPath\wal",
        $LogPath
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            Write-Host "  Created: $dir"
        } else {
            Write-Host "  Exists: $dir"
        }
    }
    
    Write-Success "Directory structure created"
}

function Install-SourceFiles {
    Write-Step "Installing source files..."
    
    $sourceFiles = @{
        "ralph_loop_implementation.py" = "$InstallPath\src\ralph_loop_implementation.py"
        "ralph_service.py" = "$InstallPath\src\ralph_service.py"
        "ralph_loop_config.yaml" = "$ConfigPath\ralph_loop_config.yaml"
    }
    
    foreach ($source in $sourceFiles.Keys) {
        $destination = $sourceFiles[$source]
        
        if (Test-Path $source) {
            Copy-Item -Path $source -Destination $destination -Force
            Write-Host "  Installed: $source -> $destination"
        } else {
            Write-Error "Source file not found: $source"
            throw "Missing source file: $source"
        }
    }
    
    Write-Success "Source files installed"
}

function Set-DirectoryPermissions {
    Write-Step "Setting directory permissions..."
    
    # Create service account if it doesn't exist
    $serviceAccount = "NT SERVICE\$Script:ServiceName"
    
    # Set ACLs on data directory
    $acl = Get-Acl $DataPath
    
    # Remove existing rules for service account
    $acl.Access | Where-Object { $_.IdentityReference -like "*$Script:ServiceName*" } | ForEach-Object {
        $acl.RemoveAccessRule($_) | Out-Null
    }
    
    # Add full control rule
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $serviceAccount,
        "FullControl",
        "ContainerInherit,ObjectInherit",
        "None",
        "Allow"
    )
    $acl.SetAccessRule($rule)
    Set-Acl $DataPath $acl
    
    # Set ACLs on log directory
    $acl = Get-Acl $LogPath
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $serviceAccount,
        "FullControl",
        "ContainerInherit,ObjectInherit",
        "None",
        "Allow"
    )
    $acl.SetAccessRule($rule)
    Set-Acl $LogPath $acl
    
    Write-Success "Directory permissions set"
}

function New-WindowsService {
    Write-Step "Creating Windows service..."
    
    # Check if service already exists
    $existingService = Get-Service -Name $Script:ServiceName -ErrorAction SilentlyContinue
    
    if ($existingService) {
        if ($Force) {
            Write-Host "  Service exists, removing..." -NoNewline
            Stop-Service -Name $Script:ServiceName -Force -ErrorAction SilentlyContinue
            sc.exe delete $Script:ServiceName | Out-Null
            Start-Sleep -Seconds 2
            Write-Host " OK" -ForegroundColor Green
        } else {
            Write-Error "Service already exists. Use -Force to overwrite."
            throw "Service already exists"
        }
    }
    
    # Create service
    $pythonPath = (Get-Command python).Source
    $servicePath = "$InstallPath\src\ralph_service.py"
    
    $binaryPath = "`"$pythonPath`" `"$servicePath`""
    
    Write-Host "  Creating service: $Script:ServiceName"
    Write-Host "  Binary path: $binaryPath"
    
    New-Service `
        -Name $Script:ServiceName `
        -BinaryPathName $binaryPath `
        -DisplayName $Script:ServiceDisplayName `
        -StartupType Automatic `
        -Description "Multi-layered background processing with priority queuing for OpenClaw AI Agent"
    
    Write-Success "Windows service created"
}

function Start-RalphService {
    Write-Step "Starting Ralph Loop service..."
    
    try {
        Start-Service -Name $Script:ServiceName
        Start-Sleep -Seconds 3
        
        $service = Get-Service -Name $Script:ServiceName
        if ($service.Status -eq "Running") {
            Write-Success "Service started successfully"
        } else {
            Write-Error "Service failed to start. Status: $($service.Status)"
            throw "Service failed to start"
        }
    } catch {
        Write-Error "Failed to start service: $_"
        throw
    }
}

function New-EnvironmentVariables {
    Write-Step "Setting environment variables..."
    
    [Environment]::SetEnvironmentVariable(
        "RALPH_INSTALL_PATH",
        $InstallPath,
        "Machine"
    )
    
    [Environment]::SetEnvironmentVariable(
        "RALPH_CONFIG_PATH",
        "$ConfigPath\ralph_loop_config.yaml",
        "Machine"
    )
    
    [Environment]::SetEnvironmentVariable(
        "RALPH_DATA_PATH",
        $DataPath,
        "Machine"
    )
    
    [Environment]::SetEnvironmentVariable(
        "RALPH_LOG_PATH",
        $LogPath,
        "Machine"
    )
    
    Write-Success "Environment variables set"
}

function New-FirewallRules {
    Write-Step "Configuring firewall rules..."
    
    # Remove existing rules
    Get-NetFirewallRule -DisplayName "RalphLoop*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    
    # Create new rule for service
    New-NetFirewallRule `
        -DisplayName "RalphLoop Service" `
        -Direction Inbound `
        -Program (Get-Command python).Source `
        -Action Allow `
        -Profile Any `
        -Enabled True `
        -Description "Allow inbound connections for Ralph Loop service"
    
    Write-Success "Firewall rules configured"
}

function Show-InstallationSummary {
    Write-Header "Installation Summary"
    
    Write-Host "Installation Path: $InstallPath"
    Write-Host "Config Path: $ConfigPath"
    Write-Host "Data Path: $DataPath"
    Write-Host "Log Path: $LogPath"
    Write-Host ""
    
    $service = Get-Service -Name $Script:ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        Write-Host "Service Status: $($service.Status)" -ForegroundColor $(
            if ($service.Status -eq "Running") { "Green" } else { "Yellow" }
        )
        Write-Host "Service Name: $($service.Name)"
        Write-Host "Display Name: $($service.DisplayName)"
    }
    
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Cyan
    Write-Host "  Start Service:   Start-Service $Script:ServiceName"
    Write-Host "  Stop Service:    Stop-Service $Script:ServiceName"
    Write-Host "  Restart Service: Restart-Service $Script:ServiceName"
    Write-Host "  View Status:     Get-Service $Script:ServiceName"
    Write-Host "  View Logs:       Get-Content '$LogPath\ralph_service.log' -Tail 50"
    Write-Host ""
    Write-Host "Console Mode:" -ForegroundColor Cyan
    Write-Host "  cd $InstallPath\src"
    Write-Host "  python ralph_service.py"
    Write-Host ""
}

function Test-Installation {
    Write-Step "Testing installation..."
    
    # Check service
    $service = Get-Service -Name $Script:ServiceName -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Error "Service not found"
        return $false
    }
    
    # Check files
    $requiredFiles = @(
        "$InstallPath\src\ralph_loop_implementation.py",
        "$InstallPath\src\ralph_service.py",
        "$ConfigPath\ralph_loop_config.yaml"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Error "Missing file: $file"
            return $false
        }
    }
    
    # Check directories
    $requiredDirs = @($DataPath, $LogPath, "$DataPath\queue", "$DataPath\checkpoints")
    
    foreach ($dir in $requiredDirs) {
        if (-not (Test-Path $dir)) {
            Write-Error "Missing directory: $dir"
            return $false
        }
    }
    
    Write-Success "Installation test passed"
    return $true
}

# =============================================================================
# MAIN INSTALLATION
# =============================================================================

Write-Header "Ralph Loop Installation"
Write-Host "OpenClaw Windows 10 AI Agent System"
Write-Host "Multi-Layered Background Processing with Priority Queuing"
Write-Host ""

try {
    # Pre-flight checks
    if (-not (Test-PythonInstallation)) {
        Write-Error "Python installation check failed"
        exit 1
    }
    
    # Installation steps
    Install-PythonDependencies
    New-DirectoryStructure
    Install-SourceFiles
    Set-DirectoryPermissions
    New-EnvironmentVariables
    New-FirewallRules
    
    if ($CreateService) {
        New-WindowsService
        
        if ($StartService) {
            Start-RalphService
        }
    }
    
    # Test installation
    if (-not (Test-Installation)) {
        Write-Error "Installation test failed"
        exit 1
    }
    
    # Show summary
    Show-InstallationSummary
    
    Write-Header "Installation Complete!"
    
} catch {
    Write-Error "Installation failed: $_"
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    exit 1
}
