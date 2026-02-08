#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Setup script for Windows AI Agent Sandbox Environment

.DESCRIPTION
    This script sets up a comprehensive sandbox environment for AI agent execution
    on Windows 10, implementing defense-in-depth security controls.

.PARAMETER AgentName
    The name of the agent sandbox to create

.PARAMETER SandboxPath
    The root path for the sandbox environment

.PARAMETER EnableHyperV
    Enable Hyper-V isolation for containers

.EXAMPLE
    .\Setup-AgentSandbox.ps1 -AgentName "MyAgent" -SandboxPath "C:\AgentSandboxes\MyAgent"

.NOTES
    Version: 1.0
    Author: Security Architecture Team
    Requires: Windows 10 Pro/Enterprise, PowerShell 5.1+
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$AgentName,
    
    [Parameter(Mandatory=$false)]
    [string]$SandboxPath = "C:\AgentSandboxes",
    
    [Parameter(Mandatory=$false)]
    [switch]$EnableHyperV,
    
    [Parameter(Mandatory=$false)]
    [switch]$EnableCredentialGuard,
    
    [Parameter(Mandatory=$false)]
    [switch]$InstallDocker
)

# Error action preference
$ErrorActionPreference = "Stop"

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

function Write-Log {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [ValidateSet("INFO", "WARNING", "ERROR", "SUCCESS")]
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $colorMap = @{
        "INFO" = "White"
        "WARNING" = "Yellow"
        "ERROR" = "Red"
        "SUCCESS" = "Green"
    }
    
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $colorMap[$Level]
}

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================

function Test-Prerequisites {
    Write-Log "Checking prerequisites..."
    
    # Check Windows version
    $osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
    $buildNumber = [System.Environment]::OSVersion.Version.Build
    
    if ($buildNumber -lt 19041) {
        throw "Windows 10 version 2004 (build 19041) or later is required"
    }
    
    Write-Log "Windows build $buildNumber detected - OK" -Level "SUCCESS"
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        throw "PowerShell 5.1 or later is required"
    }
    
    Write-Log "PowerShell $($PSVersionTable.PSVersion) detected - OK" -Level "SUCCESS"
    
    # Check for admin privileges
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must be run as Administrator"
    }
    
    Write-Log "Running with administrator privileges - OK" -Level "SUCCESS"
    
    return $true
}

# ============================================================================
# HYPER-V SETUP
# ============================================================================

function Install-HyperVFeatures {
    param([switch]$Enable)
    
    if (-not $Enable) {
        Write-Log "Hyper-V setup skipped (not requested)"
        return
    }
    
    Write-Log "Installing Hyper-V features..."
    
    $hyperVFeatures = @(
        "Microsoft-Hyper-V"
        "Microsoft-Hyper-V-Management-PowerShell"
        "Microsoft-Hyper-V-Management-Clients"
        "Containers"
        "Containers-DisposableClientVM"
    )
    
    foreach ($feature in $hyperVFeatures) {
        $state = Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue
        if ($state -and $state.State -eq "Enabled") {
            Write-Log "Feature $feature already enabled" -Level "SUCCESS"
        } else {
            Write-Log "Enabling feature: $feature..."
            Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart | Out-Null
            Write-Log "Feature $feature enabled" -Level "SUCCESS"
        }
    }
    
    # Configure Hyper-V defaults
    Write-Log "Configuring Hyper-V settings..."
    
    # Set default VM path
    $vmPath = "$SandboxPath\VMs"
    New-Item -ItemType Directory -Path $vmPath -Force | Out-Null
    Set-VMHost -VirtualMachinePath $vmPath -VirtualHardDiskPath $vmPath
    
    Write-Log "Hyper-V setup complete" -Level "SUCCESS"
}

# ============================================================================
# DOCKER SETUP
# ============================================================================

function Install-DockerEngine {
    param([switch]$Install)
    
    if (-not $Install) {
        Write-Log "Docker setup skipped (not requested)"
        return
    }
    
    Write-Log "Checking Docker installation..."
    
    $dockerPath = Get-Command docker -ErrorAction SilentlyContinue
    if ($dockerPath) {
        $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
        Write-Log "Docker already installed (version $dockerVersion)" -Level "SUCCESS"
        return
    }
    
    Write-Log "Installing Docker..."
    
    # Install Docker using PowerShell module
    Install-Module -Name DockerMsftProvider -Repository PSGallery -Force
    Install-Package -Name docker -ProviderName DockerMsftProvider -Force
    
    # Start Docker service
    Start-Service docker
    
    Write-Log "Docker installation complete" -Level "SUCCESS"
    
    # Configure Docker for Windows containers
    Write-Log "Configuring Docker for Windows containers..."
    [Environment]::SetEnvironmentVariable("LCOW_SUPPORTED", "1", "Machine")
    
    # Create daemon.json configuration
    $dockerConfig = @{
        "exec-opts" = @("isolation=hyperv")
        "storage-opts" = @()
        "log-driver" = "json-file"
        "log-opts" = @{
            "max-size" = "10m"
            "max-file" = "3"
        }
    }
    
    $dockerConfigPath = "$env:ProgramData\docker\config\daemon.json"
    New-Item -ItemType Directory -Path (Split-Path $dockerConfigPath) -Force | Out-Null
    $dockerConfig | ConvertTo-Json -Depth 10 | Set-Content $dockerConfigPath
    
    Restart-Service docker
    
    Write-Log "Docker configuration complete" -Level "SUCCESS"
}

# ============================================================================
# CREDENTIAL GUARD SETUP
# ============================================================================

function Enable-CredentialGuard {
    param([switch]$Enable)
    
    if (-not $Enable) {
        Write-Log "Credential Guard setup skipped (not requested)"
        return
    }
    
    Write-Log "Enabling Windows Defender Credential Guard..."
    
    # Check if device is compatible
    $tpm = Get-Tpm -ErrorAction SilentlyContinue
    if (-not $tpm -or -not $tpm.TpmPresent) {
        Write-Log "TPM not available - Credential Guard may not function properly" -Level "WARNING"
    }
    
    # Enable via registry
    $regPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DeviceGuard"
    New-Item -Path $regPath -Force | Out-Null
    
    Set-ItemProperty -Path $regPath -Name "EnableVirtualizationBasedSecurity" -Value 1
    Set-ItemProperty -Path $regPath -Name "RequirePlatformSecurityFeatures" -Value 1
    Set-ItemProperty -Path $regPath -Name "LsaCfgFlags" -Value 2
    
    # Enable via DG_Readiness_Tool (if available)
    $dgTool = "C:\Windows\System32\DG_Readiness_Tool_v3.6.ps1"
    if (Test-Path $dgTool) {
        & $dgTool -Enable -AutoReboot
    }
    
    Write-Log "Credential Guard enabled - reboot required" -Level "SUCCESS"
}

# ============================================================================
# SANDBOX USER SETUP
# ============================================================================

function New-SandboxUser {
    param(
        [string]$UserName,
        [string]$Description
    )
    
    Write-Log "Creating sandbox user: $UserName"
    
    # Generate random password
    $password = -join ((33..126) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
    $securePassword = ConvertTo-SecureString $password -AsPlainText -Force
    
    # Check if user exists
    try {
        $existingUser = Get-LocalUser -Name $UserName -ErrorAction Stop
        Write-Log "User $UserName already exists" -Level "WARNING"
        
        # Reset password
        Set-LocalUser -Name $UserName -Password $securePassword
        Write-Log "Password reset for $UserName"
    } catch {
        # Create new user
        New-LocalUser -Name $UserName -Password $securePassword `
            -Description $Description `
            -PasswordNeverExpires `
            -UserMayNotChangePassword
        
        Write-Log "Created user: $UserName" -Level "SUCCESS"
    }
    
    # Configure user groups
    $groups = @("Users")
    
    foreach ($group in $groups) {
        try {
            Add-LocalGroupMember -Group $group -Member $UserName -ErrorAction Stop
            Write-Log "Added $UserName to $group" -Level "SUCCESS"
        } catch {
            Write-Log "User $UserName already in $group" -Level "INFO"
        }
    }
    
    # Remove from dangerous groups
    $dangerousGroups = @("Administrators", "Power Users")
    foreach ($group in $dangerousGroups) {
        try {
            Remove-LocalGroupMember -Group $group -Member $UserName -ErrorAction Stop
            Write-Log "Removed $UserName from $group" -Level "SUCCESS"
        } catch {
            # User not in group - OK
        }
    }
    
    return $UserName
}

# ============================================================================
# SANDBOX DIRECTORY SETUP
# ============================================================================

function New-SandboxDirectory {
    param(
        [string]$Path,
        [string]$Owner
    )
    
    Write-Log "Creating sandbox directory: $Path"
    
    # Create directory structure
    $directories = @{
        "bin" = @{ Writable = $false }
        "lib" = @{ Writable = $false }
        "skills" = @{ Writable = $false }
        "config" = @{ Writable = $false }
        "data" = @{ Writable = $true }
        "logs" = @{ Writable = $true }
        "temp" = @{ Writable = $true }
        "cache" = @{ Writable = $true }
    }
    
    foreach ($dirName in $directories.Keys) {
        $dirPath = Join-Path $Path $dirName
        New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        
        # Set ACLs
        $acl = Get-Acl $dirPath
        
        # Remove inherited permissions
        $acl.SetAccessRuleProtection($true, $false)
        
        # Add owner access
        $ownerRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $Owner,
            "ReadAndExecute",
            "ContainerInherit,ObjectInherit",
            "None",
            "Allow"
        )
        $acl.AddAccessRule($ownerRule)
        
        # Add write access if needed
        if ($directories[$dirName].Writable) {
            $writeRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
                $Owner,
                "Modify",
                "ContainerInherit,ObjectInherit",
                "None",
                "Allow"
            )
            $acl.AddAccessRule($writeRule)
        }
        
        # Add SYSTEM full control
        $systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            "SYSTEM",
            "FullControl",
            "ContainerInherit,ObjectInherit",
            "None",
            "Allow"
        )
        $acl.AddAccessRule($systemRule)
        
        # Deny access to everyone else
        $denyRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            "Everyone",
            "FullControl",
            "ContainerInherit,ObjectInherit",
            "None",
            "Deny"
        )
        $acl.AddAccessRule($denyRule)
        
        Set-Acl $dirPath $acl
        
        Write-Log "Created directory: $dirPath" -Level "SUCCESS"
    }
    
    # Set integrity level to Low
    $icaclsPath = "C:\Windows\System32\icacls.exe"
    & $icaclsPath $Path /setintegritylevel L 2>&1 | Out-Null
    
    Write-Log "Set Low integrity level on $Path" -Level "SUCCESS"
}

# ============================================================================
# FIREWALL SETUP
# ============================================================================

function New-AgentFirewallRules {
    param(
        [string]$AgentName,
        [string]$ProgramPath
    )
    
    Write-Log "Creating firewall rules for $AgentName..."
    
    # Remove existing rules
    Get-NetFirewallRule -DisplayName "Agent-$AgentName-*" -ErrorAction SilentlyContinue | 
        Remove-NetFirewallRule
    
    # Block all outbound by default
    New-NetFirewallRule `
        -DisplayName "Agent-$AgentName-Block-All-Outbound" `
        -Direction Outbound `
        -Action Block `
        -Profile Any `
        -Program $ProgramPath `
        -Enabled True | Out-Null
    
    Write-Log "Created outbound block rule" -Level "SUCCESS"
    
    # Allow loopback
    New-NetFirewallRule `
        -DisplayName "Agent-$AgentName-Allow-Loopback" `
        -Direction Outbound `
        -Action Allow `
        -LocalAddress "127.0.0.1" `
        -Profile Any `
        -Program $ProgramPath `
        -Enabled True | Out-Null
    
    Write-Log "Created loopback allow rule" -Level "SUCCESS"
    
    # Allow DNS
    New-NetFirewallRule `
        -DisplayName "Agent-$AgentName-Allow-DNS" `
        -Direction Outbound `
        -Action Allow `
        -RemotePort 53 `
        -Protocol UDP `
        -Profile Any `
        -Program $ProgramPath `
        -Enabled True | Out-Null
    
    Write-Log "Created DNS allow rule" -Level "SUCCESS"
    
    # Allow HTTPS
    New-NetFirewallRule `
        -DisplayName "Agent-$AgentName-Allow-HTTPS" `
        -Direction Outbound `
        -Action Allow `
        -RemotePort 443 `
        -Protocol TCP `
        -Profile Any `
        -Program $ProgramPath `
        -Enabled True | Out-Null
    
    Write-Log "Created HTTPS allow rule" -Level "SUCCESS"
}

# ============================================================================
# WINDOWS SANDBOX SETUP
# ============================================================================

function Install-WindowsSandbox {
    Write-Log "Installing Windows Sandbox feature..."
    
    $sandboxFeature = Get-WindowsOptionalFeature -Online -FeatureName "Containers-DisposableClientVM"
    
    if ($sandboxFeature.State -eq "Enabled") {
        Write-Log "Windows Sandbox already enabled" -Level "SUCCESS"
    } else {
        Enable-WindowsOptionalFeature -Online -FeatureName "Containers-DisposableClientVM" -All -NoRestart | Out-Null
        Write-Log "Windows Sandbox enabled" -Level "SUCCESS"
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Write-Log "Starting AI Agent Sandbox Setup"
    Write-Log "Agent Name: $AgentName"
    Write-Log "Sandbox Path: $SandboxPath"
    
    # Check prerequisites
    Test-Prerequisites
    
    # Install features
    Install-HyperVFeatures -Enable:$EnableHyperV
    Install-DockerEngine -Install:$InstallDocker
    Enable-CredentialGuard -Enable:$EnableCredentialGuard
    Install-WindowsSandbox
    
    # Create sandbox user
    $sandboxUser = New-SandboxUser `
        -UserName "AgentSandbox_$AgentName" `
        -Description "Sandbox account for agent $AgentName"
    
    # Create sandbox directory
    $agentSandboxPath = Join-Path $SandboxPath $AgentName
    New-SandboxDirectory -Path $agentSandboxPath -Owner $sandboxUser
    
    # Create firewall rules
    $agentExePath = Join-Path $agentSandboxPath "bin\agent.exe"
    New-AgentFirewallRules -AgentName $AgentName -ProgramPath $agentExePath
    
    # Create configuration file
    $config = @{
        AgentName = $AgentName
        SandboxPath = $agentSandboxPath
        UserName = $sandboxUser
        Created = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        Features = @{
            HyperV = $EnableHyperV.IsPresent
            Docker = $InstallDocker.IsPresent
            CredentialGuard = $EnableCredentialGuard.IsPresent
        }
    }
    
    $configPath = Join-Path $agentSandboxPath "config\sandbox.config.json"
    $config | ConvertTo-Json -Depth 10 | Set-Content $configPath
    
    Write-Log "Sandbox configuration saved to $configPath" -Level "SUCCESS"
    
    # Summary
    Write-Log ""
    Write-Log "==============================================="
    Write-Log "SANDBOX SETUP COMPLETE"
    Write-Log "==============================================="
    Write-Log "Agent Name: $AgentName"
    Write-Log "Sandbox Path: $agentSandboxPath"
    Write-Log "Sandbox User: $sandboxUser"
    Write-Log ""
    Write-Log "Next Steps:"
    Write-Log "1. Copy agent binaries to $agentSandboxPath\bin"
    Write-Log "2. Configure agent settings in $agentSandboxPath\config"
    Write-Log "3. Review firewall rules for your specific requirements"
    Write-Log "4. Test agent execution in sandbox environment"
    Write-Log ""
    Write-Log "Note: A system reboot may be required for some features."
    Write-Log "==============================================="
}

# Run main function
Main
