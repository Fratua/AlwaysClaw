# setup-agent.ps1
#Requires -Version 5.1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Setup script for AI Agent PowerShell integration
.DESCRIPTION
    Installs and configures all necessary components for the AI Agent
    PowerShell automation system on Windows 10
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$InstallPath = "$env:LOCALAPPDATA\AIAgent",
    
    [Parameter()]
    [ValidateSet('Development', 'Production')]
    [string]$Environment = 'Development',
    
    [Parameter()]
    [switch]$SkipCertificateCreation,
    
    [Parameter()]
    [switch]$SkipWinRMConfig
)

begin {
    $ErrorActionPreference = 'Stop'
    $script:StartTime = Get-Date
    
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  AI Agent PowerShell Integration Setup" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

process {
    # Step 1: Check prerequisites
    Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Yellow
    
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -lt 5) {
        throw "PowerShell 5.1 or higher is required. Current version: $psVersion"
    }
    Write-Host "  PowerShell version: $psVersion [OK]" -ForegroundColor Green
    
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).
        IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        throw "Administrator privileges required"
    }
    Write-Host "  Administrator privileges: [OK]" -ForegroundColor Green
    
    # Step 2: Create directory structure
    Write-Host "`nStep 2: Creating directory structure..." -ForegroundColor Yellow
    
    $directories = @{
        Root = $InstallPath
        Modules = Join-Path $InstallPath 'Modules'
        Scripts = Join-Path $InstallPath 'Scripts'
        Logs = Join-Path $InstallPath 'Logs'
        Config = Join-Path $InstallPath 'Config'
        Data = Join-Path $InstallPath 'Data'
        Temp = Join-Path $InstallPath 'Temp'
    }
    
    foreach ($name in $directories.Keys) {
        $path = $directories[$name]
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-Host "  Created: $name -> $path" -ForegroundColor Gray
        } else {
            Write-Host "  Exists: $name -> $path" -ForegroundColor DarkGray
        }
    }
    
    # Step 3: Configure execution policy
    Write-Host "`nStep 3: Configuring execution policy..." -ForegroundColor Yellow
    
    try {
        & "$PSScriptRoot\Configure-AgentExecutionPolicy.ps1" `
            -Environment $Environment `
            -AgentInstallPath $InstallPath
        Write-Host "  Execution policy configured [OK]" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to configure execution policy: $_"
    }
    
    # Step 4: Install modules
    Write-Host "`nStep 4: Installing PowerShell modules..." -ForegroundColor Yellow
    
    $moduleFiles = @(
        'AgentCore.psm1'
        'WMIProvider.psm1'
        'AgentOperations.psm1'
        'AgentCmdlets.psm1'
    )
    
    foreach ($module in $moduleFiles) {
        $sourcePath = Join-Path $PSScriptRoot $module
        $destPath = Join-Path $directories.Modules $module
        
        if (Test-Path $sourcePath) {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "  Installed: $module" -ForegroundColor Gray
        } else {
            Write-Warning "Module not found: $module"
        }
    }
    
    # Step 5: Create code signing certificate (if not skipped)
    if (-not $SkipCertificateCreation) {
        Write-Host "`nStep 5: Creating code signing certificate..." -ForegroundColor Yellow
        
        try {
            $certParams = @{
                Subject = 'CN=AIAgentCodeSigning'
                CertStoreLocation = 'Cert:\CurrentUser\My'
                KeyUsage = @('DigitalSignature')
                Type = 'CodeSigningCert'
                NotAfter = (Get-Date).AddYears(5)
                KeySpec = 'Signature'
                KeyLength = 2048
                HashAlgorithm = 'SHA256'
            }
            
            $cert = New-SelfSignedCertificate @certParams
            
            # Export certificate
            $certExportPath = Join-Path $directories.Config 'AgentSigningCert.cer'
            Export-Certificate -Cert $cert -FilePath $certExportPath -Type CERT | Out-Null
            
            # Add to trusted publishers
            $trustedPublisherPath = 'Cert:\LocalMachine\TrustedPublisher'
            Import-Certificate -FilePath $certExportPath -CertStoreLocation $trustedPublisherPath | Out-Null
            
            Write-Host "  Certificate created: $($cert.Thumbprint)" -ForegroundColor Green
            Write-Host "  Certificate exported to: $certExportPath" -ForegroundColor Gray
        } catch {
            Write-Warning "Failed to create certificate: $_"
        }
    }
    
    # Step 6: Configure WinRM (if not skipped)
    if (-not $SkipWinRMConfig) {
        Write-Host "`nStep 6: Configuring WinRM..." -ForegroundColor Yellow
        
        try {
            $winrmService = Get-Service -Name WinRM -ErrorAction SilentlyContinue
            
            if ($winrmService -and $winrmService.Status -ne 'Running') {
                Enable-PSRemoting -Force -SkipNetworkProfileCheck
                Write-Host "  WinRM enabled [OK]" -ForegroundColor Green
            } else {
                Write-Host "  WinRM already configured [OK]" -ForegroundColor Green
            }
            
            # Configure trusted hosts
            Set-Item WSMan:\localhost\Client\TrustedHosts -Value '*' -Force
            Write-Host "  Trusted hosts configured [OK]" -ForegroundColor Green
            
        } catch {
            Write-Warning "Failed to configure WinRM: $_"
        }
    }
    
    # Step 7: Create configuration file
    Write-Host "`nStep 7: Creating configuration file..." -ForegroundColor Yellow
    
    $agentConfig = @{
        AgentId = [Guid]::NewGuid().ToString()
        Name = 'AIAgent'
        Version = '1.0.0'
        InstallPath = $InstallPath
        Environment = $Environment
        CreatedAt = Get-Date -Format 'o'
        PowerShell = @{
            ExecutionPolicy = if ($Environment -eq 'Production') { 'AllSigned' } else { 'RemoteSigned' }
            Version = $PSVersionTable.PSVersion.ToString()
        }
        Features = @{
            Heartbeat = $true
            Logging = $true
            WMI = $true
            Remoting = $true
        }
    }
    
    $configPath = Join-Path $directories.Config 'agent-config.json'
    $agentConfig | ConvertTo-Json -Depth 5 | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "  Configuration saved to: $configPath" -ForegroundColor Green
    
    # Step 8: Create startup script
    Write-Host "`nStep 8: Creating startup script..." -ForegroundColor Yellow
    
    $startupScript = @"
# AI Agent Startup Script
param(`
    [string]`$ConfigPath = "$configPath"
)

# Import modules
Import-Module "$($directories.Modules)\AgentCore.psm1" -Force
Import-Module "$($directories.Modules)\WMIProvider.psm1" -Force

# Load configuration
`$config = Get-Content `$ConfigPath | ConvertFrom-Json

# Initialize environment
Initialize-AgentEnvironment -Configuration @{
    AgentId = `$config.AgentId
    Name = `$config.Name
    Version = `$config.Version
    LogPath = "$($directories.Logs)"
    ScriptPath = "$($directories.Scripts)"
    ConfigPath = "$($directories.Config)"
    DataPath = "$($directories.Data)"
}

# Start heartbeat
Start-AgentHeartbeat -IntervalSeconds 60

Write-Host "AI Agent started successfully!" -ForegroundColor Green
Write-Host "Agent ID: `$(`$config.AgentId)" -ForegroundColor Cyan
"@
    
    $startupPath = Join-Path $directories.Scripts 'Start-Agent.ps1'
    $startupScript | Out-File -FilePath $startupPath -Encoding UTF8
    Write-Host "  Startup script created: $startupPath" -ForegroundColor Green
    
    # Step 9: Set environment variables
    Write-Host "`nStep 9: Setting environment variables..." -ForegroundColor Yellow
    
    [Environment]::SetEnvironmentVariable('AI_AGENT_PATH', $InstallPath, 'User')
    [Environment]::SetEnvironmentVariable('AI_AGENT_CONFIG', $configPath, 'User')
    [Environment]::SetEnvironmentVariable('AI_AGENT_MODULES', $directories.Modules, 'User')
    
    Write-Host "  Environment variables set [OK]" -ForegroundColor Green
    
    # Summary
    $duration = (Get-Date) - $script:StartTime
    
    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "Installation Path: $InstallPath" -ForegroundColor Cyan
    Write-Host "Environment: $Environment" -ForegroundColor Cyan
    Write-Host "Duration: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Cyan
    Write-Host "`nTo start the agent, run:" -ForegroundColor Yellow
    Write-Host "  & '$startupPath'" -ForegroundColor White
    Write-Host "`nOr from Node.js:" -ForegroundColor Yellow
    Write-Host "  const { AgentPowerShellIntegration } = require('./agent-powershell-integration');" -ForegroundColor White
    Write-Host "  const agent = new AgentPowerShellIntegration();" -ForegroundColor White
    Write-Host "  await agent.initialize();" -ForegroundColor White
}

end {
    Write-Host "`nSetup finished." -ForegroundColor Green
}
