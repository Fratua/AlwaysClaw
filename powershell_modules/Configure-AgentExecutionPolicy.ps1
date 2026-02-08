# Configure-AgentExecutionPolicy.ps1
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('Development', 'Production', 'Restricted')]
    [string]$Environment = 'Development',
    
    [Parameter()]
    [string]$AgentInstallPath = "$env:LOCALAPPDATA\AIAgent",
    
    [Parameter()]
    [switch]$Force
)

begin {
    $ErrorActionPreference = 'Stop'
    
    # Environment-specific configurations
    $configurations = @{
        Development = @{
            ExecutionPolicy = 'RemoteSigned'
            RequireSignature = $false
            AllowRemoteScripts = $true
            LoggingLevel = 'Verbose'
        }
        Production = @{
            ExecutionPolicy = 'AllSigned'
            RequireSignature = $true
            AllowRemoteScripts = $false
            LoggingLevel = 'Warning'
        }
        Restricted = @{
            ExecutionPolicy = 'Restricted'
            RequireSignature = $true
            AllowRemoteScripts = $false
            LoggingLevel = 'Error'
        }
    }
    
    $config = $configurations[$Environment]
}

process {
    Write-Host "Configuring PowerShell execution policy for $Environment environment..." -ForegroundColor Cyan
    
    # Check elevation
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).
        IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin -and $Environment -eq 'Production') {
        throw "Production environment configuration requires administrator privileges"
    }
    
    # Set execution policy
    try {
        Set-ExecutionPolicy -ExecutionPolicy $config.ExecutionPolicy -Scope Process -Force
        Set-ExecutionPolicy -ExecutionPolicy $config.ExecutionPolicy -Scope CurrentUser -Force
        
        if ($isAdmin) {
            Set-ExecutionPolicy -ExecutionPolicy $config.ExecutionPolicy -Scope LocalMachine -Force
        }
        
        Write-Host "Execution policy set to: $($config.ExecutionPolicy)" -ForegroundColor Green
    } catch {
        Write-Error "Failed to set execution policy: $_"
    }
    
    # Create agent directories
    $directories = @(
        $AgentInstallPath
        "$AgentInstallPath\Scripts"
        "$AgentInstallPath\Modules"
        "$AgentInstallPath\Logs"
        "$AgentInstallPath\Config"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Gray
        }
    }
    
    # Configure PowerShell profile for agent
    $profileContent = @'
# AI Agent PowerShell Profile
$ErrorActionPreference = 'Continue'
$VerbosePreference = 'Continue'
$WarningPreference = 'Continue'

# Import agent modules
$AgentModulesPath = '$AgentInstallPath\Modules'
if (Test-Path $AgentModulesPath) {
    Get-ChildItem $AgentModulesPath -Filter '*.psm1' | ForEach-Object {
        Import-Module $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

# Set agent environment variables
$env:AI_AGENT_HOME = '$AgentInstallPath'
$env:AI_AGENT_LOG_PATH = '$AgentInstallPath\Logs'
$env:AI_AGENT_SCRIPTS_PATH = '$AgentInstallPath\Scripts'

# Configure logging
$logFile = Join-Path $env:AI_AGENT_LOG_PATH "agent-powershell-$(Get-Date -Format 'yyyyMMdd').log"
Start-Transcript -Path $logFile -Append -ErrorAction SilentlyContinue

Write-Host "AI Agent PowerShell Environment Loaded" -ForegroundColor Cyan
'@
    
    $profilePath = "$AgentInstallPath\Profile.ps1"
    $profileContent | Out-File -FilePath $profilePath -Encoding UTF8 -Force
    
    # Set environment variable for agent identification
    [Environment]::SetEnvironmentVariable('AI_AGENT_INSTALLED', 'true', 'User')
    [Environment]::SetEnvironmentVariable('AI_AGENT_PATH', $AgentInstallPath, 'User')
    
    Write-Host "`nConfiguration complete!" -ForegroundColor Green
    Write-Host "Agent path: $AgentInstallPath" -ForegroundColor Gray
    Write-Host "Execution policy: $($config.ExecutionPolicy)" -ForegroundColor Gray
}
