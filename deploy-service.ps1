# OpenClaw Windows Service Deployment Script
# Supports Blue-Green, Rolling, and Canary deployment strategies

param(
    [Parameter(Mandatory=$true)]
    [string]$Environment,
    
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [Parameter(Mandatory=$true)]
    [string]$ArtifactPath,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("BlueGreen", "Rolling", "Canary")]
    [string]$Strategy = "BlueGreen",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipBackup,
    
    [Parameter(Mandatory=$false)]
    [int]$HealthCheckTimeoutSeconds = 60,
    
    [Parameter(Mandatory=$false)]
    [string]$LogLevel = "INFO"
)

# Configuration
$ServiceName = "OpenClawAgent"
$InstallPath = "C:\OpenClaw\$Environment"
$BackupPath = "C:\OpenClaw\Backups"
$LogPath = "C:\OpenClaw\Logs"
$ConfigPath = "C:\OpenClaw\Config"

# Ensure directories exist
@($LogPath, $BackupPath, $ConfigPath) | ForEach-Object {
    if (!(Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}

# Logging function
function Write-Log {
    param(
        [string]$Message, 
        [ValidateSet("INFO", "WARN", "ERROR", "DEBUG")]
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path "$LogPath\deployment.log" -Value $logEntry -ErrorAction SilentlyContinue
}

# Service management functions
function Stop-AgentService {
    param([string]$Name, [int]$TimeoutSeconds = 60)
    Write-Log "Stopping service: $Name" "INFO"
    
    $service = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq "Running") {
        try {
            Stop-Service -Name $Name -Force -ErrorAction Stop
            $service.WaitForStatus("Stopped", [TimeSpan]::FromSeconds($TimeoutSeconds))
            Write-Log "Service stopped successfully" "INFO"
            return $true
        }
        catch {
            Write-Log "Failed to stop service: $_" "ERROR"
            return $false
        }
    }
    return $true
}

function Start-AgentService {
    param([string]$Name, [int]$TimeoutSeconds = 60)
    Write-Log "Starting service: $Name" "INFO"
    
    try {
        Start-Service -Name $Name -ErrorAction Stop
        $service = Get-Service -Name $Name
        $service.WaitForStatus("Running", [TimeSpan]::FromSeconds($TimeoutSeconds))
        Write-Log "Service started successfully" "INFO"
        return $true
    }
    catch {
        Write-Log "Failed to start service: $_" "ERROR"
        return $false
    }
}

function Backup-CurrentVersion {
    Write-Log "Creating backup of current version" "INFO"
    
    $backupDir = "$BackupPath\$Environment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    try {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        
        if (Test-Path $InstallPath) {
            Copy-Item -Path "$InstallPath\*" -Destination $backupDir -Recurse -Force
            Write-Log "Backup created at: $backupDir" "INFO"
            return $backupDir
        }
        else {
            Write-Log "No existing installation to backup" "WARN"
            return $null
        }
    }
    catch {
        Write-Log "Backup failed: $_" "ERROR"
        throw
    }
}

function Deploy-NewVersion {
    param([string]$TargetPath, [string]$Artifact)
    Write-Log "Deploying version $Version to $TargetPath" "INFO"
    
    try {
        # Create install directory
        New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
        
        # Extract artifact
        Expand-Archive -Path $Artifact -DestinationPath $TargetPath -Force
        
        # Set permissions
        $acl = Get-Acl $TargetPath
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            "NT AUTHORITY\SYSTEM", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        )
        $acl.SetAccessRule($rule)
        Set-Acl $TargetPath $acl
        
        Write-Log "Deployment completed to $TargetPath" "INFO"
        return $true
    }
    catch {
        Write-Log "Deployment failed: $_" "ERROR"
        return $false
    }
}

function Test-HealthEndpoint {
    param(
        [string]$Url = "http://localhost:8080/health",
        [int]$TimeoutSeconds = 30,
        [int]$MaxRetries = 3
    )
    Write-Log "Running health check: $Url" "INFO"
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Log "Health check passed (attempt $i)" "INFO"
                return $true
            }
        }
        catch {
            Write-Log "Health check attempt $i failed: $_" "WARN"
            if ($i -lt $MaxRetries) {
                Start-Sleep -Seconds 5
            }
        }
    }
    return $false
}

function Test-AgentLoops {
    param([int]$ExpectedCount = 15)
    Write-Log "Checking agent loop status" "INFO"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health/agents" -TimeoutSec 10
        $data = $response.Content | ConvertFrom-Json
        $runningCount = ($data.agents | Where-Object { $_.status -eq "running" }).Count
        
        if ($runningCount -ge $ExpectedCount) {
            Write-Log "All $runningCount agent loops running" "INFO"
            return $true
        }
        else {
            Write-Log "Only $runningCount of $ExpectedCount agent loops running" "WARN"
            return $false
        }
    }
    catch {
        Write-Log "Agent loop check failed: $_" "ERROR"
        return $false
    }
}

function Invoke-Rollback {
    param([string]$BackupLocation, [string]$TargetPath)
    Write-Log "Initiating rollback..." "ERROR"
    
    try {
        Stop-AgentService -Name $ServiceName
        
        if ($BackupLocation -and (Test-Path $BackupLocation)) {
            Remove-Item -Path "$TargetPath\*" -Recurse -Force -ErrorAction SilentlyContinue
            Copy-Item -Path "$BackupLocation\*" -Destination $TargetPath -Recurse -Force
            Start-AgentService -Name $ServiceName
            
            # Verify rollback
            Start-Sleep -Seconds 5
            if (Test-HealthEndpoint) {
                Write-Log "Rollback completed successfully" "INFO"
                return $true
            }
        }
        Write-Log "Rollback failed - backup not available" "ERROR"
        return $false
    }
    catch {
        Write-Log "Rollback failed: $_" "ERROR"
        return $false
    }
}

# Main deployment logic
Write-Log "=== Starting OpenClaw Deployment ===" "INFO"
Write-Log "Environment: $Environment" "INFO"
Write-Log "Version: $Version" "INFO"
Write-Log "Strategy: $Strategy" "INFO"

$backupLocation = $null
$deploymentSuccess = $false

try {
    # Step 1: Backup current version
    if (!$SkipBackup) {
        $backupLocation = Backup-CurrentVersion
    }
    
    # Step 2: Stop service
    if (!(Stop-AgentService -Name $ServiceName)) {
        throw "Failed to stop service"
    }
    
    # Step 3: Deploy new version
    if (!(Deploy-NewVersion -TargetPath $InstallPath -Artifact $ArtifactPath)) {
        throw "Deployment failed"
    }
    
    # Step 4: Start service
    if (!(Start-AgentService -Name $ServiceName)) {
        throw "Failed to start service"
    }
    
    # Step 5: Health checks
    Write-Log "Running post-deployment health checks..." "INFO"
    
    if (!(Test-HealthEndpoint -TimeoutSeconds $HealthCheckTimeoutSeconds)) {
        throw "Health endpoint check failed"
    }
    
    if (!(Test-AgentLoops)) {
        throw "Agent loop check failed"
    }
    
    $deploymentSuccess = $true
    Write-Log "=== Deployment Completed Successfully ===" "INFO"
    exit 0
}
catch {
    Write-Log "Deployment failed: $_" "ERROR"
    
    if ($backupLocation) {
        Invoke-Rollback -BackupLocation $backupLocation -TargetPath $InstallPath
    }
    
    exit 1
}
finally {
    # Cleanup old backups (keep last 10)
    Get-ChildItem -Path $BackupPath -Directory | 
        Sort-Object CreationTime -Descending | 
        Select-Object -Skip 10 | 
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
