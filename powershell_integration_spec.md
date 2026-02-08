# PowerShell Automation & Scripting Integration Specification
## Windows 10 AI Agent System (OpenClaw-Inspired)

**Version:** 1.0  
**Platform:** Windows 10  
**Integration Layer:** Node.js ↔ PowerShell Bridge  
**Document Date:** 2025

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [PowerShell Execution Layer](#powershell-execution-layer)
3. [Execution Policy Management](#execution-policy-management)
4. [PowerShell Remoting](#powershell-remoting)
5. [WMI Integration](#wmi-integration)
6. [Agent PowerShell Modules](#agent-powershell-modules)
7. [Security & Script Signing](#security--script-signing)
8. [Error Handling & Output Parsing](#error-handling--output-parsing)
9. [Cmdlet Wrappers](#cmdlet-wrappers)
10. [Implementation Examples](#implementation-examples)

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent Core (Node.js)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   GPT-5.2   │  │  Scheduler  │  │    Identity Engine      │  │
│  │   Engine    │  │  (Cron)     │  │    (Soul/Heartbeat)     │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         └─────────────────┴─────────────────────┘                │
│                          │                                       │
│         ┌────────────────┴────────────────┐                      │
│         │     PowerShell Bridge Layer      │                      │
│         │  ┌────────────────────────────┐  │                      │
│         │  │  PowerShell Host Process   │  │                      │
│         │  │  (child_process/spawn)     │  │                      │
│         │  └────────────────────────────┘  │                      │
│         └────────────────┬─────────────────┘                      │
└──────────────────────────┼───────────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────────┐
│                    Windows 10 System Layer                        │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  PowerShell  │  │    WMI      │  │   Windows APIs          │  │
│  │   Engine     │  │  Providers  │  │  (Win32/PInvoke)        │  │
│  └──────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Registry   │  │  Services   │  │   File System           │  │
│  │   Access     │  │  Control    │  │   Operations            │  │
│  └──────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## PowerShell Execution Layer

### 1.1 Node.js PowerShell Bridge

#### Core Execution Module

```javascript
// powershell-bridge.js
const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');

class PowerShellBridge extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      executionPolicy: options.executionPolicy || 'RemoteSigned',
      noProfile: options.noProfile !== false,
      nonInteractive: options.nonInteractive !== false,
      windowStyle: options.windowStyle || 'Hidden',
      workingDirectory: options.workingDirectory || process.cwd(),
      timeout: options.timeout || 30000,
      encoding: options.encoding || 'utf8',
      ...options
    };
    this.sessionId = this.generateSessionId();
    this.commandHistory = [];
    this.isInitialized = false;
  }

  generateSessionId() {
    return `ps_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Execute PowerShell command/script
   * @param {string} command - PowerShell command or script path
   * @param {Object} options - Execution options
   * @returns {Promise<Object>} Execution result
   */
  async execute(command, options = {}) {
    const execOptions = { ...this.options, ...options };
    const executionId = this.generateSessionId();
    
    return new Promise((resolve, reject) => {
      const args = this.buildArguments(execOptions);
      const psPath = this.getPowerShellPath();
      
      // Log command for audit trail
      this.logCommand(executionId, command);

      const child = spawn(psPath, [...args, '-Command', command], {
        cwd: execOptions.workingDirectory,
        windowsHide: true,
        env: {
          ...process.env,
          PSExecutionPolicyPreference: execOptions.executionPolicy
        }
      });

      let stdout = '';
      let stderr = '';
      let timeoutId;

      // Handle timeout
      if (execOptions.timeout > 0) {
        timeoutId = setTimeout(() => {
          child.kill('SIGTERM');
          reject(new PowerShellTimeoutError(
            `Command timed out after ${execOptions.timeout}ms`,
            command,
            executionId
          ));
        }, execOptions.timeout);
      }

      child.stdout.on('data', (data) => {
        stdout += data.toString(execOptions.encoding);
        this.emit('stdout', { executionId, data: data.toString() });
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString(execOptions.encoding);
        this.emit('stderr', { executionId, data: data.toString() });
      });

      child.on('close', (code) => {
        clearTimeout(timeoutId);
        
        const result = {
          executionId,
          exitCode: code,
          stdout: stdout.trim(),
          stderr: stderr.trim(),
          success: code === 0,
          command: command.substring(0, 100),
          timestamp: new Date().toISOString(),
          duration: Date.now() - parseInt(executionId.split('_')[1])
        };

        if (code === 0 || execOptions.ignoreExitCode) {
          resolve(result);
        } else {
          reject(new PowerShellExecutionError(
            `PowerShell execution failed with code ${code}`,
            result
          ));
        }
      });

      child.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(new PowerShellExecutionError(
          `Failed to spawn PowerShell: ${error.message}`,
          { executionId, command }
        ));
      });
    });
  }

  /**
   * Execute PowerShell script file
   * @param {string} scriptPath - Path to .ps1 file
   * @param {Object} parameters - Script parameters
   * @param {Object} options - Execution options
   */
  async executeScript(scriptPath, parameters = {}, options = {}) {
    const paramString = Object.entries(parameters)
      .map(([key, value]) => `-${key} "${this.escapeParameter(value)}"`)
      .join(' ');
    
    const command = `& "${scriptPath}" ${paramString}`;
    return this.execute(command, options);
  }

  /**
   * Execute script block with structured output
   * @param {string} scriptBlock - PowerShell script block
   * @param {Object} options - Execution options
   */
  async executeStructured(scriptBlock, options = {}) {
    const wrapper = `
      $ErrorActionPreference = 'Stop'
      try {
        $result = ${scriptBlock}
        $result | ConvertTo-Json -Depth 10 -Compress
      } catch {
        @{ 
          error = $_.Exception.Message
          stackTrace = $_.ScriptStackTrace
          category = $_.CategoryInfo.Category.ToString()
        } | ConvertTo-Json -Compress
      }
    `;
    
    const result = await this.execute(wrapper, options);
    
    try {
      return JSON.parse(result.stdout);
    } catch (e) {
      return { rawOutput: result.stdout, parseError: e.message };
    }
  }

  buildArguments(options) {
    const args = [];
    
    if (options.noProfile) args.push('-NoProfile');
    if (options.nonInteractive) args.push('-NonInteractive');
    if (options.executionPolicy) {
      args.push('-ExecutionPolicy', options.executionPolicy);
    }
    if (options.windowStyle) {
      args.push('-WindowStyle', options.windowStyle);
    }
    
    return args;
  }

  getPowerShellPath() {
    // Prefer PowerShell 7 (pwsh), fallback to Windows PowerShell
    const ps7Path = 'C:\\Program Files\\PowerShell\\7\\pwsh.exe';
    const ps5Path = 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe';
    
    try {
      require('fs').accessSync(ps7Path);
      return ps7Path;
    } catch {
      return ps5Path;
    }
  }

  escapeParameter(value) {
    if (typeof value !== 'string') return value;
    return value.replace(/"/g, '`"').replace(/\$/g, '`$');
  }

  logCommand(executionId, command) {
    this.commandHistory.push({
      executionId,
      command: command.substring(0, 500),
      timestamp: new Date().toISOString()
    });
    
    // Keep last 1000 commands
    if (this.commandHistory.length > 1000) {
      this.commandHistory.shift();
    }
  }
}

// Custom Error Classes
class PowerShellExecutionError extends Error {
  constructor(message, details) {
    super(message);
    this.name = 'PowerShellExecutionError';
    this.details = details;
  }
}

class PowerShellTimeoutError extends Error {
  constructor(message, command, executionId) {
    super(message);
    this.name = 'PowerShellTimeoutError';
    this.command = command;
    this.executionId = executionId;
  }
}

module.exports = { PowerShellBridge, PowerShellExecutionError, PowerShellTimeoutError };
```

#### Persistent Session Manager

```javascript
// powershell-session.js
const { spawn } = require('child_process');
const readline = require('readline');

class PowerShellSession {
  constructor(options = {}) {
    this.options = options;
    this.process = null;
    this.commandQueue = [];
    this.isReady = false;
    this.sessionVariables = new Map();
    this.modulesLoaded = new Set();
  }

  async initialize() {
    const psPath = this.getPowerShellPath();
    const args = [
      '-NoProfile',
      '-NonInteractive',
      '-ExecutionPolicy', 'RemoteSigned',
      '-WindowStyle', 'Hidden',
      '-Command', '-'
    ];

    this.process = spawn(psPath, args, {
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    // Setup output handling
    this.rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    this.process.stderr.on('data', (data) => {
      console.error('PS Session Error:', data.toString());
    });

    // Initialize session state
    await this.execute('$PSVersionTable.PSVersion | ConvertTo-Json');
    this.isReady = true;
    
    return this;
  }

  async execute(command, expectJson = false) {
    return new Promise((resolve, reject) => {
      const marker = `__END_${Date.now()}__`;
      const fullCommand = expectJson 
        ? `${command} | ConvertTo-Json -Depth 10; Write-Output "${marker}"`
        : `${command}; Write-Output "${marker}"`;

      let output = '';
      let isComplete = false;

      const onLine = (line) => {
        if (line.includes(marker)) {
          isComplete = true;
          this.rl.off('line', onLine);
          const result = output.replace(marker, '').trim();
          
          if (expectJson) {
            try {
              resolve(JSON.parse(result));
            } catch {
              resolve({ rawOutput: result });
            }
          } else {
            resolve(result);
          }
        } else {
          output += line + '\n';
        }
      };

      this.rl.on('line', onLine);
      this.process.stdin.write(fullCommand + '\n');

      // Timeout handler
      setTimeout(() => {
        if (!isComplete) {
          this.rl.off('line', onLine);
          reject(new Error('Command timeout'));
        }
      }, this.options.timeout || 30000);
    });
  }

  async setVariable(name, value) {
    const psValue = typeof value === 'object' 
      ? `@(${JSON.stringify(value)})`
      : `"${value}"`;
    
    await this.execute(`$global:${name} = ${psValue}`);
    this.sessionVariables.set(name, value);
  }

  async getVariable(name) {
    return this.execute(`$global:${name} | ConvertTo-Json`, true);
  }

  async loadModule(modulePath) {
    if (this.modulesLoaded.has(modulePath)) return;
    
    await this.execute(`Import-Module "${modulePath}" -Force`);
    this.modulesLoaded.add(modulePath);
  }

  async dispose() {
    if (this.process) {
      this.process.stdin.write('exit\n');
      await new Promise(resolve => setTimeout(resolve, 500));
      this.process.kill();
      this.isReady = false;
    }
  }
}

module.exports = { PowerShellSession };
```

---

## Execution Policy Management

### 2.1 Execution Policy Architecture

```powershell
# ExecutionPolicyManager.ps1
#Requires -Version 5.1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Manages PowerShell execution policies for AI Agent system
.DESCRIPTION
    Configures and validates execution policies across different scopes
    ensuring secure script execution for the agent system
#>

class ExecutionPolicyManager {
    [string]$AgentScriptsPath
    [hashtable]$PolicyConfiguration
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$SigningCertificate
    
    ExecutionPolicyManager([string]$scriptsPath) {
        $this.AgentScriptsPath = $scriptsPath
        $this.PolicyConfiguration = @{
            MachinePolicy = 'Undefined'
            UserPolicy = 'Undefined'
            Process = 'Bypass'
            CurrentUser = 'RemoteSigned'
            LocalMachine = 'RemoteSigned'
        }
    }
    
    [PSCustomObject] GetCurrentPolicies() {
        $policies = @{}
        foreach ($scope in @('MachinePolicy', 'UserPolicy', 'Process', 'CurrentUser', 'LocalMachine')) {
            try {
                $policies[$scope] = Get-ExecutionPolicy -Scope $scope -ErrorAction SilentlyContinue
            } catch {
                $policies[$scope] = 'Undefined'
            }
        }
        return [PSCustomObject]$policies
    }
    
    [void] ConfigureForAgent() {
        # Set process-level policy for current session
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
        
        # Set user-level policy for persistent access
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        
        # Create agent scripts directory with proper ACLs
        if (-not (Test-Path $this.AgentScriptsPath)) {
            New-Item -ItemType Directory -Path $this.AgentScriptsPath -Force | Out-Null
        }
        
        # Set restrictive ACL on agent scripts directory
        $acl = Get-Acl $this.AgentScriptsPath
        $acl.SetAccessRuleProtection($true, $false)
        
        # Add current user with full control
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $currentUser, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow'
        )
        $acl.AddAccessRule($rule)
        Set-Acl $this.AgentScriptsPath $acl
    }
    
    [bool] ValidateScript([string]$scriptPath) {
        # Check if script exists
        if (-not (Test-Path $scriptPath)) {
            throw "Script not found: $scriptPath"
        }
        
        # Check file extension
        $extension = [System.IO.Path]::GetExtension($scriptPath).ToLower()
        if ($extension -notin @('.ps1', '.psm1', '.psd1')) {
            throw "Invalid script extension: $extension"
        }
        
        # Check digital signature if required
        $signature = Get-AuthenticodeSignature -FilePath $scriptPath
        if ($signature.Status -ne 'Valid' -and $this.RequireSigned) {
            return $false
        }
        
        # Scan for dangerous commands
        $dangerousPatterns = @(
            'Invoke-Expression.*\$',
            'Invoke-Command.*-ScriptBlock.*\$',
            'Start-Process.*-Verb.*runas',
            'net\s+user.*\/add',
            'reg\s+add.*\\Run'
        )
        
        $content = Get-Content $scriptPath -Raw
        foreach ($pattern in $dangerousPatterns) {
            if ($content -match $pattern) {
                Write-Warning "Potentially dangerous pattern detected: $pattern"
            }
        }
        
        return $true
    }
    
    [void] SetSigningCertificate([string]$certPath, [string]$password) {
        $securePassword = ConvertTo-SecureString -String $password -AsPlainText -Force
        $this.SigningCertificate = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2(
            $certPath, $securePassword
        )
    }
    
    [void] SignScript([string]$scriptPath) {
        if (-not $this.SigningCertificate) {
            throw "No signing certificate configured"
        }
        
        Set-AuthenticodeSignature -FilePath $scriptPath -Certificate $this.SigningCertificate
    }
}

# Export function for module usage
function New-ExecutionPolicyManager {
    param([string]$ScriptsPath)
    return [ExecutionPolicyManager]::new($ScriptsPath)
}

Export-ModuleMember -Function New-ExecutionPolicyManager
```

### 2.2 Policy Configuration Script

```powershell
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
        $AgentInstallPath,
        "$AgentInstallPath\Scripts",
        "$AgentInstallPath\Modules",
        "$AgentInstallPath\Logs",
        "$AgentInstallPath\Config"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Gray
        }
    }
    
    # Configure PowerShell profile for agent
    $profileContent = @"
# AI Agent PowerShell Profile
`$ErrorActionPreference = 'Continue'
`$VerbosePreference = 'Continue'
`$WarningPreference = 'Continue'

# Import agent modules
`$AgentModulesPath = '$AgentInstallPath\Modules'
if (Test-Path `$AgentModulesPath) {
    Get-ChildItem `$AgentModulesPath -Filter '*.psm1' | ForEach-Object {
        Import-Module `$_.FullName -Force -ErrorAction SilentlyContinue
    }
}

# Set agent environment variables
`$env:AI_AGENT_HOME = '$AgentInstallPath'
`$env:AI_AGENT_LOG_PATH = '$AgentInstallPath\Logs'
`$env:AI_AGENT_SCRIPTS_PATH = '$AgentInstallPath\Scripts'

# Configure logging
`$logFile = Join-Path `$env:AI_AGENT_LOG_PATH "agent-powershell-`$(Get-Date -Format 'yyyyMMdd').log"
Start-Transcript -Path `$logFile -Append -ErrorAction SilentlyContinue

Write-Host "AI Agent PowerShell Environment Loaded" -ForegroundColor Cyan
"@
    
    $profilePath = "$AgentInstallPath\Profile.ps1"
    $profileContent | Out-File -FilePath $profilePath -Encoding UTF8 -Force
    
    # Set environment variable for agent identification
    [Environment]::SetEnvironmentVariable('AI_AGENT_INSTALLED', 'true', 'User')
    [Environment]::SetEnvironmentVariable('AI_AGENT_PATH', $AgentInstallPath, 'User')
    
    Write-Host "`nConfiguration complete!" -ForegroundColor Green
    Write-Host "Agent path: $AgentInstallPath" -ForegroundColor Gray
    Write-Host "Execution policy: $($config.ExecutionPolicy)" -ForegroundColor Gray
}
```

---

## PowerShell Remoting

### 3.1 Remoting Configuration

```powershell
# RemotingManager.psm1
#Requires -Version 5.1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    PowerShell Remoting management for AI Agent distributed operations
.DESCRIPTION
    Configures and manages WinRM/PSRemoting for local and remote
    system management capabilities
#>

class RemotingManager {
    [string]$TrustedHosts
    [int]$MaxMemoryPerShellMB
    [hashtable]$SessionCache
    [bool]$UseSSL
    
    RemotingManager() {
        $this.TrustedHosts = '*'
        $this.MaxMemoryPerShellMB = 1024
        $this.SessionCache = @{}
        $this.UseSSL = $false
    }
    
    [void] EnableRemoting() {
        # Check if WinRM is already configured
        $winrmService = Get-Service -Name WinRM -ErrorAction SilentlyContinue
        
        if (-not $winrmService) {
            throw "WinRM service not found. Ensure Windows Management Framework is installed."
        }
        
        # Enable PowerShell Remoting
        try {
            Enable-PSRemoting -Force -SkipNetworkProfileCheck -ErrorAction Stop
            Write-Host "PowerShell Remoting enabled successfully" -ForegroundColor Green
        } catch {
            throw "Failed to enable PowerShell Remoting: $_"
        }
        
        # Configure WinRM settings
        $this.ConfigureWinRM()
        
        # Configure firewall rules
        $this.ConfigureFirewall()
    }
    
    [void] ConfigureWinRM() {
        # Set trusted hosts (restrict in production)
        Set-Item WSMan:\localhost\Client\TrustedHosts -Value $this.TrustedHosts -Force
        
        # Configure memory limits
        Set-Item WSMan:\localhost\Shell\MaxMemoryPerShellMB -Value $this.MaxMemoryPerShellMB
        
        # Configure timeout settings
        Set-Item WSMan:\localhost\Shell\IdleTimeout -Value 7200000  # 2 hours
        Set-Item WSMan:\localhost\Shell\MaxShellsPerUser -Value 25
        
        # Enable certificate-based authentication if using SSL
        if ($this.UseSSL) {
            Set-Item WSMan:\localhost\Service\Auth\Certificate -Value $true
        }
        
        # Restart WinRM to apply changes
        Restart-Service WinRM -Force
    }
    
    [void] ConfigureFirewall() {
        # Enable WinRM firewall rules
        $rules = @(
            'WINRM-HTTP-In-TCP',
            'WINRM-HTTP-In-TCP-PUBLIC',
            'WINRM-HTTPS-In-TCP'
        )
        
        foreach ($rule in $rules) {
            $firewallRule = Get-NetFirewallRule -Name $rule -ErrorAction SilentlyContinue
            if ($firewallRule) {
                Enable-NetFirewallRule -Name $rule
                Write-Host "Enabled firewall rule: $rule" -ForegroundColor Gray
            }
        }
    }
    
    [System.Management.Automation.Runspaces.PSSession] CreateSession(
        [string]$ComputerName,
        [pscredential]$Credential,
        [hashtable]$Options
    ) {
        $sessionOption = New-PSSessionOption @Options
        
        $sessionParams = @{
            ComputerName = $ComputerName
            Credential = $Credential
            SessionOption = $sessionOption
        }
        
        if ($this.UseSSL) {
            $sessionParams['UseSSL'] = $true
            $sessionParams['SessionOption'].SkipCACheck = $true
            $sessionParams['SessionOption'].SkipCNCheck = $true
        }
        
        $session = New-PSSession @sessionParams
        $this.SessionCache[$session.Id] = $session
        
        return $session
    }
    
    [object] InvokeCommand(
        [System.Management.Automation.Runspaces.PSSession]$Session,
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList
    ) {
        $invokeParams = @{
            Session = $Session
            ScriptBlock = $ScriptBlock
        }
        
        if ($ArgumentList) {
            $invokeParams['ArgumentList'] = $ArgumentList
        }
        
        return Invoke-Command @invokeParams
    }
    
    [void] CloseSession([int]$SessionId) {
        if ($this.SessionCache.ContainsKey($SessionId)) {
            Remove-PSSession -Session $this.SessionCache[$SessionId]
            $this.SessionCache.Remove($SessionId)
        }
    }
    
    [void] CloseAllSessions() {
        foreach ($sessionId in $this.SessionCache.Keys) {
            Remove-PSSession -Session $this.SessionCache[$sessionId] -ErrorAction SilentlyContinue
        }
        $this.SessionCache.Clear()
    }
    
    [PSCustomObject] TestRemoting([string]$ComputerName) {
        $result = @{
            ComputerName = $ComputerName
            WinRMService = $false
            PSRemoting = $false
            FirewallRules = $false
            Authentication = $false
            ResponseTime = $null
        }
        
        # Test WinRM service
        try {
            $service = Invoke-Command -ComputerName $ComputerName -ScriptBlock {
                Get-Service -Name WinRM
            } -ErrorAction Stop
            $result.WinRMService = $service.Status -eq 'Running'
        } catch {
            $result.WinRMService = $false
        }
        
        # Test PSRemoting
        try {
            $version = Invoke-Command -ComputerName $ComputerName -ScriptBlock {
                $PSVersionTable.PSVersion
            } -ErrorAction Stop
            $result.PSRemoting = $true
        } catch {
            $result.PSRemoting = $false
        }
        
        return [PSCustomObject]$result
    }
}

# JEA (Just Enough Administration) Configuration
function New-AgentJEAEndpoint {
    param(
        [string]$EndpointName = 'AIAgentJEA',
        [string]$ModulePath = "$env:ProgramFiles\WindowsPowerShell\Modules\AgentJEA"
    )
    
    # Create module structure
    $moduleStructure = @{
        Root = $ModulePath
        Functions = Join-Path $ModulePath 'RoleCapabilities'
        Scripts = Join-Path $ModulePath 'Scripts'
    }
    
    foreach ($path in $moduleStructure.Values) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    
    # Create role capability file
    $roleCapability = @{
        VisibleCmdlets = @(
            'Get-Process',
            'Get-Service',
            'Get-EventLog',
            'Get-WinEvent',
            'Get-WmiObject',
            'Get-CimInstance',
            'Write-Output',
            'Write-Information'
        )
        VisibleFunctions = @('Get-AgentStatus', 'Write-AgentLog')
        VisibleExternalCommands = @('C:\Windows\System32\whoami.exe')
        VisibleProviders = @('FileSystem', 'Registry', 'Environment')
        ScriptsToProcess = @()
        AliasDefinitions = @()
        FunctionDefinitions = @()
        VariableDefinitions = @(
            @{ Name = 'AgentHome'; Value = $env:AI_AGENT_HOME }
        )
        EnvironmentVariables = @{}
        TypesToProcess = @()
        FormatsToProcess = @()
        AssembliesToLoad = @()
    }
    
    $roleCapabilityPath = Join-Path $moduleStructure.Functions 'AgentOperator.psrc'
    New-PSRoleCapabilityFile -Path $roleCapabilityPath @roleCapability
    
    # Create session configuration file
    $sessionConfigParams = @{
        Path = Join-Path $moduleStructure.Root 'AgentJEA.pssc'
        SessionType = 'RestrictedRemoteServer'
        RunAsVirtualAccount = $true
        RoleDefinitions = @{
            'BUILTIN\Users' = @{ RoleCapabilities = 'AgentOperator' }
        }
        TranscriptDirectory = "$env:AI_AGENT_PATH\Logs\JEA"
        ScriptsToProcess = @()
        FunctionDefinitions = @()
        VariableDefinitions = @()
        AliasDefinitions = @()
        EnvironmentVariables = @{}
        TypesToProcess = @()
        FormatsToProcess = @()
        AssembliesToLoad = @()
    }
    
    New-PSSessionConfigurationFile @sessionConfigParams
    
    # Register the endpoint
    Register-PSSessionConfiguration -Name $EndpointName -Path $sessionConfigParams.Path -Force
    
    Write-Host "JEA endpoint '$EndpointName' created successfully" -ForegroundColor Green
}

Export-ModuleMember -Function New-AgentJEAEndpoint
Export-ModuleMember -Function New-RemotingManager
```

---

## WMI Integration

### 4.1 WMI Provider Module

```powershell
# WMIProvider.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    WMI Integration module for AI Agent system management
.DESCRIPTION
    Provides comprehensive WMI access for system information,
    hardware monitoring, and management operations
#>

class WMIProvider {
    [string]$Namespace
    [hashtable]$QueryCache
    [int]$CacheDurationSeconds
    
    WMIProvider([string]$namespace = 'root\cimv2') {
        $this.Namespace = $namespace
        $this.QueryCache = @{}
        $this.CacheDurationSeconds = 60
    }
    
    #region System Information
    
    [PSCustomObject] GetComputerSystem() {
        return $this.ExecuteQuery('SELECT * FROM Win32_ComputerSystem') | Select-Object -First 1
    }
    
    [PSCustomObject] GetOperatingSystem() {
        return $this.ExecuteQuery('SELECT * FROM Win32_OperatingSystem') | Select-Object -First 1
    }
    
    [PSCustomObject] GetProcessor() {
        return $this.ExecuteQuery('SELECT * FROM Win32_Processor') | Select-Object -First 1
    }
    
    [array] GetPhysicalMemory() {
        return $this.ExecuteQuery('SELECT * FROM Win32_PhysicalMemory')
    }
    
    [PSCustomObject] GetBIOS() {
        return $this.ExecuteQuery('SELECT * FROM Win32_BIOS') | Select-Object -First 1
    }
    
    [array] GetDiskDrives() {
        return $this.ExecuteQuery('SELECT * FROM Win32_DiskDrive')
    }
    
    [array] GetLogicalDisks() {
        return $this.ExecuteQuery('SELECT * FROM Win32_LogicalDisk WHERE DriveType=3')
    }
    
    #endregion
    
    #region Process & Service Management
    
    [array] GetProcesses([hashtable]$Filter) {
        $query = 'SELECT * FROM Win32_Process'
        
        if ($Filter) {
            $conditions = @()
            foreach ($key in $Filter.Keys) {
                $conditions += "$key = '$($Filter[$key])'"
            }
            $query += ' WHERE ' + ($conditions -join ' AND ')
        }
        
        return $this.ExecuteQuery($query)
    }
    
    [array] GetServices([string]$State, [string]$StartMode) {
        $query = 'SELECT * FROM Win32_Service'
        $conditions = @()
        
        if ($State) { $conditions += "State = '$State'" }
        if ($StartMode) { $conditions += "StartMode = '$StartMode'" }
        
        if ($conditions.Count -gt 0) {
            $query += ' WHERE ' + ($conditions -join ' AND ')
        }
        
        return $this.ExecuteQuery($query)
    }
    
    [bool] StartService([string]$ServiceName) {
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.StartService()
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to start service $ServiceName : $_"
            return $false
        }
    }
    
    [bool] StopService([string]$ServiceName) {
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.StopService()
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to stop service $ServiceName : $_"
            return $false
        }
    }
    
    [bool] SetServiceStartMode([string]$ServiceName, [string]$StartMode) {
        $validModes = @('Auto', 'Manual', 'Disabled', 'Boot', 'System')
        if ($StartMode -notin $validModes) {
            throw "Invalid start mode. Valid values: $($validModes -join ', ')"
        }
        
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.ChangeStartMode($StartMode)
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to change service start mode: $_"
            return $false
        }
    }
    
    #endregion
    
    #region Performance Monitoring
    
    [PSCustomObject] GetSystemPerformance() {
        $performance = @{}
        
        # CPU Usage
        $processor = $this.GetProcessor()
        $performance.CpuUsage = $processor.LoadPercentage
        
        # Memory Usage
        $os = $this.GetOperatingSystem()
        $totalMemory = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        $freeMemory = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $performance.TotalMemoryGB = $totalMemory
        $performance.FreeMemoryGB = $freeMemory
        $performance.MemoryUsedPercent = [math]::Round((($totalMemory - $freeMemory) / $totalMemory) * 100, 2)
        
        # Disk Usage
        $disks = $this.GetLogicalDisks() | ForEach-Object {
            [PSCustomObject]@{
                Drive = $_.DeviceID
                TotalGB = [math]::Round($_.Size / 1GB, 2)
                FreeGB = [math]::Round($_.FreeSpace / 1GB, 2)
                UsedPercent = [math]::Round((($_.Size - $_.FreeSpace) / $_.Size) * 100, 2)
            }
        }
        $performance.Disks = $disks
        
        return [PSCustomObject]$performance
    }
    
    [array] GetProcessPerformance() {
        $query = @'
            SELECT Name, ProcessId, WorkingSetSize, PageFileUsage, 
                   UserModeTime, KernelModeTime, ThreadCount, HandleCount
            FROM Win32_Process
'@
        
        $processes = $this.ExecuteQuery($query)
        
        return $processes | ForEach-Object {
            [PSCustomObject]@{
                Name = $_.Name
                ProcessId = $_.ProcessId
                WorkingSetMB = [math]::Round($_.WorkingSetSize / 1MB, 2)
                PageFileMB = [math]::Round($_.PageFileUsage / 1MB, 2)
                ThreadCount = $_.ThreadCount
                HandleCount = $_.HandleCount
            }
        }
    }
    
    #endregion
    
    #region Event Log Access
    
    [array] GetEventLogs(
        [string]$LogName,
        [int]$Newest = 100,
        [string]$Level,
        [datetime]$After,
        [datetime]$Before
    ) {
        # Use Get-WinEvent for better performance
        $filterXPath = @()
        
        if ($Level) {
            $levelMap = @{
                'Critical' = 1
                'Error' = 2
                'Warning' = 3
                'Information' = 4
                'Verbose' = 5
            }
            $filterXPath += "*[System[Level=$($levelMap[$Level])]]"
        }
        
        if ($After) {
            $filterXPath += "*[System[TimeCreated[@SystemTime>='$($After.ToUniversalTime().ToString('o'))']]]"
        }
        
        if ($Before) {
            $filterXPath += "*[System[TimeCreated[@SystemTime<='$($Before.ToUniversalTime().ToString('o'))']]]"
        }
        
        $filterHashtable = @{
            LogName = $LogName
        }
        
        if ($filterXPath.Count -gt 0) {
            $filterHashtable['XPath'] = $filterXPath -join ' and '
        }
        
        try {
            $events = Get-WinEvent -FilterHashtable $filterHashtable -MaxEvents $Newest -ErrorAction Stop
            return $events | Select-Object TimeCreated, Id, LevelDisplayName, Message
        } catch {
            Write-Error "Failed to retrieve event logs: $_"
            return @()
        }
    }
    
    #endregion
    
    #region Network Information
    
    [array] GetNetworkAdapters() {
        return $this.ExecuteQuery('SELECT * FROM Win32_NetworkAdapter WHERE NetEnabled = TRUE')
    }
    
    [array] GetNetworkAdapterConfiguration([bool]$IPEnabled = $true) {
        $query = 'SELECT * FROM Win32_NetworkAdapterConfiguration'
        if ($IPEnabled) {
            $query += ' WHERE IPEnabled = TRUE'
        }
        return $this.ExecuteQuery($query)
    }
    
    [PSCustomObject] GetNetworkStatistics() {
        $tcpStats = $this.ExecuteQuery('SELECT * FROM Win32_PerfFormattedData_TCPv4_TCPv4') | Select-Object -First 1
        $ipStats = $this.ExecuteQuery('SELECT * FROM Win32_PerfFormattedData_TCPIP_IPv4') | Select-Object -First 1
        
        return [PSCustomObject]@{
            ConnectionsEstablished = $tcpStats.ConnectionsEstablished
            ConnectionsActive = $tcpStats.ConnectionsActive
            ConnectionsPassive = $tcpStats.ConnectionsPassive
            DatagramsReceived = $ipStats.DatagramsReceivedPersec
            DatagramsSent = $ipStats.DatagramsSentPersec
        }
    }
    
    #endregion
    
    #region Helper Methods
    
    [array] ExecuteQuery([string]$query) {
        # Check cache
        $cacheKey = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($query))
        $cached = $this.QueryCache[$cacheKey]
        
        if ($cached -and ($cached.Timestamp -gt (Get-Date).AddSeconds(-$this.CacheDurationSeconds))) {
            return $cached.Data
        }
        
        # Execute query
        $result = Get-WmiObject -Query $query -Namespace $this.Namespace -ErrorAction Stop
        
        # Cache result
        $this.QueryCache[$cacheKey] = @{
            Data = $result
            Timestamp = Get-Date
        }
        
        return $result
    }
    
    [void] ClearCache() {
        $this.QueryCache.Clear()
    }
    
    #endregion
}

# CIM Alternative (PowerShell 5.1+)
class CIMProvider {
    [string]$ComputerName
    [pscredential]$Credential
    [Microsoft.Management.Infrastructure.CimSession]$Session
    
    CIMProvider([string]$computerName = 'localhost') {
        $this.ComputerName = $computerName
        $this.InitializeSession()
    }
    
    [void] InitializeSession() {
        $sessionOptions = New-CimSessionOption -Protocol Dcom
        
        $sessionParams = @{
            ComputerName = $this.ComputerName
            SessionOption = $sessionOptions
        }
        
        if ($this.Credential) {
            $sessionParams['Credential'] = $this.Credential
        }
        
        $this.Session = New-CimSession @sessionParams
    }
    
    [array] GetInstances([string]$className, [string]$namespace = 'root/cimv2') {
        return Get-CimInstance -CimSession $this.Session -ClassName $className -Namespace $namespace
    }
    
    [array] Query([string]$query, [string]$namespace = 'root/cimv2') {
        return Get-CimInstance -CimSession $this.Session -Query $query -Namespace $namespace
    }
    
    [void] Dispose() {
        if ($this.Session) {
            Remove-CimSession -CimSession $this.Session
        }
    }
}

# Export functions
function New-WMIProvider {
    param([string]$Namespace = 'root\cimv2')
    return [WMIProvider]::new($Namespace)
}

function New-CIMProvider {
    param([string]$ComputerName = 'localhost')
    return [CIMProvider]::new($ComputerName)
}

Export-ModuleMember -Function New-WMIProvider, New-CIMProvider
```

---

## Agent PowerShell Modules

### 5.1 Core Agent Module

```powershell
# AgentCore.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    Core PowerShell module for AI Agent system
.DESCRIPTION
    Provides fundamental agent operations including heartbeat,
    identity management, and system integration
#>

# Module-level variables
$script:AgentConfig = $null
$script:HeartbeatTimer = $null
$script:IdentityState = @{}
$script:OperationLog = [System.Collections.ArrayList]::new()

#region Configuration

function Initialize-AgentEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$Configuration
    )
    
    $script:AgentConfig = [PSCustomObject]$Configuration
    
    # Create necessary directories
    $paths = @(
        $Configuration.LogPath
        $Configuration.ScriptPath
        $Configuration.ConfigPath
        $Configuration.DataPath
    )
    
    foreach ($path in $paths) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-AgentLog -Message "Created directory: $path" -Level Info
        }
    }
    
    # Initialize identity state
    $script:IdentityState = @{
        AgentId = $Configuration.AgentId
        Name = $Configuration.Name
        CreatedAt = Get-Date
        LastHeartbeat = $null
        Status = 'Initializing'
        Version = $Configuration.Version
    }
    
    # Set environment variables
    [Environment]::SetEnvironmentVariable('AI_AGENT_ID', $Configuration.AgentId, 'Process')
    [Environment]::SetEnvironmentVariable('AI_AGENT_NAME', $Configuration.Name, 'Process')
    [Environment]::SetEnvironmentVariable('AI_AGENT_CONFIG_PATH', $Configuration.ConfigPath, 'Process')
    
    Write-AgentLog -Message "Agent environment initialized: $($Configuration.Name)" -Level Info
    
    return $script:IdentityState
}

#endregion

#region Heartbeat System

function Start-AgentHeartbeat {
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$IntervalSeconds = 60,
        
        [Parameter()]
        [scriptblock]$OnHeartbeat = $null,
        
        [Parameter()]
        [string]$LogPath
    )
    
    if ($script:HeartbeatTimer) {
        Stop-AgentHeartbeat
    }
    
    $heartbeatScript = {
        param($Config, $Callback, $LogDir)
        
        $heartbeatData = @{
            Timestamp = Get-Date -Format 'o'
            AgentId = $Config.AgentId
            Status = 'Active'
            SystemInfo = @{
                ComputerName = $env:COMPUTERNAME
                UserName = $env:USERNAME
                ProcessId = $PID
                MemoryWorkingSet = (Get-Process -Id $PID).WorkingSet64
            }
        }
        
        # Log heartbeat
        $logFile = Join-Path $LogDir "heartbeat-$(Get-Date -Format 'yyyyMMdd').json"
        $heartbeatData | ConvertTo-Json -Compress | Add-Content -Path $logFile
        
        # Execute callback if provided
        if ($Callback) {
            & $Callback $heartbeatData
        }
    }
    
    $timer = New-Object System.Timers.Timer
    $timer.Interval = $IntervalSeconds * 1000
    $timer.AutoReset = $true
    
    $action = {
        & $heartbeatScript $script:AgentConfig $OnHeartbeat $script:AgentConfig.LogPath
    }
    
    Register-ObjectEvent -InputObject $timer -EventName Elapsed -Action $action | Out-Null
    $timer.Start()
    
    $script:HeartbeatTimer = $timer
    $script:IdentityState.LastHeartbeat = Get-Date
    $script:IdentityState.Status = 'Running'
    
    Write-AgentLog -Message "Heartbeat started with interval: ${IntervalSeconds}s" -Level Info
}

function Stop-AgentHeartbeat {
    [CmdletBinding()]
    param()
    
    if ($script:HeartbeatTimer) {
        $script:HeartbeatTimer.Stop()
        $script:HeartbeatTimer.Dispose()
        $script:HeartbeatTimer = $null
        $script:IdentityState.Status = 'Stopped'
        
        Write-AgentLog -Message "Heartbeat stopped" -Level Info
    }
}

function Get-AgentHeartbeatStatus {
    [CmdletBinding()]
    param()
    
    return [PSCustomObject]@{
        IsRunning = $null -ne $script:HeartbeatTimer
        LastHeartbeat = $script:IdentityState.LastHeartbeat
        Status = $script:IdentityState.Status
    }
}

#endregion

#region Identity Management

function Get-AgentIdentity {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch]$IncludeSystemInfo
    )
    
    $identity = @{
        AgentId = $script:IdentityState.AgentId
        Name = $script:IdentityState.Name
        CreatedAt = $script:IdentityState.CreatedAt
        Version = $script:IdentityState.Version
        Status = $script:IdentityState.Status
    }
    
    if ($IncludeSystemInfo) {
        $identity.SystemInfo = @{
            ComputerName = $env:COMPUTERNAME
            UserName = $env:USERNAME
            UserDomain = $env:USERDOMAIN
            ProcessId = $PID
            PowerShellVersion = $PSVersionTable.PSVersion.ToString()
            OSVersion = [System.Environment]::OSVersion.VersionString
        }
    }
    
    return [PSCustomObject]$identity
}

function Set-AgentIdentityProperty {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Property,
        
        [Parameter(Mandatory)]
        [object]$Value
    )
    
    $script:IdentityState[$Property] = $Value
    Write-AgentLog -Message "Identity property updated: $Property" -Level Info
}

#endregion

#region Logging

function Write-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Message,
        
        [Parameter()]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error', 'Critical')]
        [string]$Level = 'Info',
        
        [Parameter()]
        [hashtable]$Metadata = @{},
        
        [Parameter()]
        [string]$Category = 'General'
    )
    
    $logEntry = [PSCustomObject]@{
        Timestamp = Get-Date -Format 'o'
        Level = $Level
        Category = $Category
        Message = $Message
        AgentId = $script:IdentityState.AgentId
        ProcessId = $PID
        Metadata = $Metadata
    }
    
    # Write to console with color
    $colorMap = @{
        Debug = 'Gray'
        Info = 'White'
        Warning = 'Yellow'
        Error = 'Red'
        Critical = 'Magenta'
    }
    
    Write-Host "[$($logEntry.Timestamp)] [$Level] $Message" -ForegroundColor $colorMap[$Level]
    
    # Write to file
    if ($script:AgentConfig -and $script:AgentConfig.LogPath) {
        $logFile = Join-Path $script:AgentConfig.LogPath "agent-$(Get-Date -Format 'yyyyMMdd').log"
        $logEntry | ConvertTo-Json -Compress | Add-Content -Path $logFile
    }
    
    # Add to operation log
    $script:OperationLog.Add($logEntry) | Out-Null
    
    # Keep only last 1000 entries in memory
    if ($script:OperationLog.Count -gt 1000) {
        $script:OperationLog.RemoveAt(0)
    }
}

function Get-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$Count = 100,
        
        [Parameter()]
        [ValidateSet('Debug', 'Info', 'Warning', 'Error', 'Critical')]
        [string]$Level,
        
        [Parameter()]
        [string]$Category
    )
    
    $logs = $script:OperationLog
    
    if ($Level) {
        $logs = $logs | Where-Object { $_.Level -eq $Level }
    }
    
    if ($Category) {
        $logs = $logs | Where-Object { $_.Category -eq $Category }
    }
    
    return $logs | Select-Object -Last $Count
}

function Export-AgentLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        
        [Parameter()]
        [ValidateSet('Json', 'Csv', 'Xml')]
        [string]$Format = 'Json'
    )
    
    $logs = $script:OperationLog
    
    switch ($Format) {
        'Json' {
            $logs | ConvertTo-Json -Depth 5 | Out-File -FilePath $Path -Encoding UTF8
        }
        'Csv' {
            $logs | Export-Csv -Path $Path -NoTypeInformation -Encoding UTF8
        }
        'Xml' {
            $logs | Export-Clixml -Path $Path
        }
    }
    
    Write-AgentLog -Message "Logs exported to: $Path" -Level Info
}

#endregion

#region System Operations

function Get-AgentSystemStatus {
    [CmdletBinding()]
    param()
    
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $processor = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
    $memory = Get-CimInstance -ClassName Win32_PhysicalMemory
    
    return [PSCustomObject]@{
        Timestamp = Get-Date -Format 'o'
        OperatingSystem = @{
            Caption = $os.Caption
            Version = $os.Version
            Architecture = $os.OSArchitecture
            LastBootTime = $os.LastBootUpTime
        }
        Processor = @{
            Name = $processor.Name
            Cores = $processor.NumberOfCores
            LogicalProcessors = $processor.NumberOfLogicalProcessors
            LoadPercentage = $processor.LoadPercentage
        }
        Memory = @{
            TotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
            FreeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
            UsedPercent = [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100, 2)
            Modules = $memory | ForEach-Object { [math]::Round($_.Capacity / 1GB, 2) }
        }
        AgentProcess = @{
            Id = $PID
            WorkingSetMB = [math]::Round((Get-Process -Id $PID).WorkingSet64 / 1MB, 2)
            Threads = (Get-Process -Id $PID).Threads.Count
            Handles = (Get-Process -Id $PID).Handles
            StartTime = (Get-Process -Id $PID).StartTime
        }
    }
}

function Invoke-AgentMaintenance {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('LogCleanup', 'CacheClear', 'MemoryOptimize', 'Full')]
        [string]$Type = 'Full'
    )
    
    Write-AgentLog -Message "Starting maintenance: $Type" -Level Info -Category 'Maintenance'
    
    $results = @()
    
    if ($Type -in @('LogCleanup', 'Full')) {
        # Clean old log files (older than 30 days)
        $cutoffDate = (Get-Date).AddDays(-30)
        $logFiles = Get-ChildItem -Path $script:AgentConfig.LogPath -Filter '*.log' | 
            Where-Object { $_.LastWriteTime -lt $cutoffDate }
        
        foreach ($file in $logFiles) {
            Remove-Item -Path $file.FullName -Force
            $results += "Removed log file: $($file.Name)"
        }
    }
    
    if ($Type -in @('CacheClear', 'Full')) {
        # Clear operation log
        $script:OperationLog.Clear()
        $results += "Cleared operation log cache"
    }
    
    if ($Type -in @('MemoryOptimize', 'Full')) {
        # Force garbage collection
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
        $results += "Garbage collection completed"
    }
    
    Write-AgentLog -Message "Maintenance completed: $($results -join '; ')" -Level Info -Category 'Maintenance'
    
    return $results
}

#endregion

#region Export

Export-ModuleMember -Function @(
    'Initialize-AgentEnvironment'
    'Start-AgentHeartbeat'
    'Stop-AgentHeartbeat'
    'Get-AgentHeartbeatStatus'
    'Get-AgentIdentity'
    'Set-AgentIdentityProperty'
    'Write-AgentLog'
    'Get-AgentLog'
    'Export-AgentLog'
    'Get-AgentSystemStatus'
    'Invoke-AgentMaintenance'
)

#endregion
```

### 5.2 Agent Operations Module

```powershell
# AgentOperations.psm1
#Requires -Version 5.1
#Requires -Modules AgentCore

<#
.SYNOPSIS
    Agent operations module for task execution and automation
.DESCRIPTION
    Provides task scheduling, file operations, process management,
    and integration with external services
#>

#region Task Scheduling

function New-AgentScheduledTask {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        
        [Parameter(Mandatory)]
        [scriptblock]$Action,
        
        [Parameter()]
        [ValidateSet('Once', 'Daily', 'Hourly', 'AtStartup', 'AtLogon', 'OnIdle')]
        [string]$Trigger = 'Once',
        
        [Parameter()]
        [datetime]$StartTime = (Get-Date).AddMinutes(5),
        
        [Parameter()]
        [int]$IntervalMinutes = 60,
        
        [Parameter()]
        [string]$Description = "AI Agent scheduled task",
        
        [Parameter()]
        [switch]$RunAsSystem,
        
        [Parameter()]
        [switch]$Hidden
    )
    
    # Convert scriptblock to file
    $scriptPath = Join-Path $env:AI_AGENT_SCRIPTS_PATH "$Name.ps1"
    $action.ToString() | Out-File -FilePath $scriptPath -Encoding UTF8
    
    # Create scheduled task action
    $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-ExecutionPolicy Bypass -File `"$scriptPath`""
    
    # Create trigger
    switch ($Trigger) {
        'Once' { 
            $taskTrigger = New-ScheduledTaskTrigger -Once -At $StartTime 
        }
        'Daily' { 
            $taskTrigger = New-ScheduledTaskTrigger -Daily -At $StartTime 
        }
        'Hourly' { 
            $taskTrigger = New-ScheduledTaskTrigger -Once -At $StartTime -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) 
        }
        'AtStartup' { 
            $taskTrigger = New-ScheduledTaskTrigger -AtStartup 
        }
        'AtLogon' { 
            $taskTrigger = New-ScheduledTaskTrigger -AtLogon 
        }
        'OnIdle' { 
            $taskTrigger = New-ScheduledTaskTrigger -IdleDuration (New-TimeSpan -Minutes 10) 
        }
    }
    
    # Create principal
    if ($RunAsSystem) {
        $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    } else {
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
    }
    
    # Create settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    if ($Hidden) {
        $settings.Hidden = $true
    }
    
    # Register task
    $task = Register-ScheduledTask -TaskName "AIAgent_$Name" -Action $taskAction -Trigger $taskTrigger `
        -Principal $principal -Settings $settings -Description $Description -Force
    
    Write-AgentLog -Message "Created scheduled task: AIAgent_$Name" -Level Info -Category 'TaskScheduler'
    
    return $task
}

function Get-AgentScheduledTask {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Name
    )
    
    $filter = if ($Name) { "AIAgent_$Name" else "AIAgent_*" }
    
    return Get-ScheduledTask -TaskName $filter | ForEach-Object {
        [PSCustomObject]@{
            Name = $_.TaskName
            State = $_.State
            LastRunTime = $_.LastRunTime
            NextRunTime = $_.NextRunTime
            Author = $_.Author
            Description = $_.Description
        }
    }
}

function Remove-AgentScheduledTask {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name
    )
    
    $taskName = "AIAgent_$Name"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
    
    Write-AgentLog -Message "Removed scheduled task: $taskName" -Level Info -Category 'TaskScheduler'
}

#endregion

#region File Operations

function Invoke-AgentFileOperation {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Copy', 'Move', 'Delete', 'Archive', 'Sync', 'Watch')]
        [string]$Operation,
        
        [Parameter(Mandatory)]
        [string]$Source,
        
        [Parameter()]
        [string]$Destination,
        
        [Parameter()]
        [string]$Filter = '*',
        
        [Parameter()]
        [switch]$Recurse,
        
        [Parameter()]
        [switch]$Force
    )
    
    $result = @{
        Operation = $Operation
        Source = $Source
        Destination = $Destination
        Success = $false
        FilesProcessed = 0
        Errors = @()
    }
    
    try {
        switch ($Operation) {
            'Copy' {
                if (-not $Destination) { throw "Destination required for Copy operation" }
                
                $files = Get-ChildItem -Path $Source -Filter $Filter -Recurse:$Recurse
                foreach ($file in $files) {
                    $destPath = $file.FullName.Replace($Source, $Destination)
                    Copy-Item -Path $file.FullName -Destination $destPath -Force:$Force -ErrorAction Stop
                    $result.FilesProcessed++
                }
            }
            
            'Move' {
                if (-not $Destination) { throw "Destination required for Move operation" }
                
                $files = Get-ChildItem -Path $Source -Filter $Filter -Recurse:$Recurse
                foreach ($file in $files) {
                    $destPath = $file.FullName.Replace($Source, $Destination)
                    Move-Item -Path $file.FullName -Destination $destPath -Force:$Force -ErrorAction Stop
                    $result.FilesProcessed++
                }
            }
            
            'Delete' {
                $files = Get-ChildItem -Path $Source -Filter $Filter -Recurse:$Recurse
                foreach ($file in $files) {
                    Remove-Item -Path $file.FullName -Force:$Force -Recurse:$file.PSIsContainer -ErrorAction Stop
                    $result.FilesProcessed++
                }
            }
            
            'Archive' {
                if (-not $Destination) { throw "Destination required for Archive operation" }
                
                $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
                $archiveName = "Archive_$timestamp.zip"
                $archivePath = Join-Path $Destination $archiveName
                
                Compress-Archive -Path $Source -DestinationPath $archivePath -Force:$Force
                $result.FilesProcessed = (Get-ChildItem -Path $Source -Recurse).Count
                $result.ArchivePath = $archivePath
            }
            
            'Sync' {
                if (-not $Destination) { throw "Destination required for Sync operation" }
                
                $robocopyArgs = @(
                    '"' + $Source + '"'
                    '"' + $Destination + '"'
                    '/MIR'
                    '/R:3'
                    '/W:5'
                )
                if ($Filter -ne '*') {
                    $robocopyArgs += $Filter
                }
                
                $process = Start-Process -FilePath 'robocopy.exe' -ArgumentList $robocopyArgs `
                    -Wait -PassThru -WindowStyle Hidden
                
                $result.ExitCode = $process.ExitCode
                $result.FilesProcessed = $process.ExitCode -lt 8  # Robocopy success codes
            }
        }
        
        $result.Success = $true
        Write-AgentLog -Message "File operation completed: $Operation" -Level Info -Category 'FileOperation'
        
    } catch {
        $result.Errors += $_.Exception.Message
        $result.Success = $false
        Write-AgentLog -Message "File operation failed: $_" -Level Error -Category 'FileOperation'
    }
    
    return [PSCustomObject]$result
}

function Start-AgentFileWatcher {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        
        [Parameter()]
        [string]$Filter = '*.*',
        
        [Parameter()]
        [ValidateSet('Created', 'Changed', 'Deleted', 'Renamed', 'All')]
        [string[]]$NotifyFilter = @('Created', 'Changed', 'Deleted'),
        
        [Parameter(Mandatory)]
        [scriptblock]$Action,
        
        [Parameter()]
        [switch]$IncludeSubdirectories,
        
        [Parameter()]
        [switch]$PassThru
    )
    
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $Path
    $watcher.Filter = $Filter
    $watcher.IncludeSubdirectories = $IncludeSubdirectories
    $watcher.EnableRaisingEvents = $true
    
    # Map notify filter
    $notifyFlags = 0
    foreach ($filter in $NotifyFilter) {
        $notifyFlags = $notifyFlags -bor [System.IO.NotifyFilters]::$filter
    }
    $watcher.NotifyFilter = $notifyFlags
    
    # Register events
    $eventParams = @{
        InputObject = $watcher
        Action = $Action
    }
    
    if ($NotifyFilter -contains 'Created' -or $NotifyFilter -contains 'All') {
        Register-ObjectEvent @eventParams -EventName Created -SourceIdentifier "AgentFileWatcher_Created_$Path"
    }
    if ($NotifyFilter -contains 'Changed' -or $NotifyFilter -contains 'All') {
        Register-ObjectEvent @eventParams -EventName Changed -SourceIdentifier "AgentFileWatcher_Changed_$Path"
    }
    if ($NotifyFilter -contains 'Deleted' -or $NotifyFilter -contains 'All') {
        Register-ObjectEvent @eventParams -EventName Deleted -SourceIdentifier "AgentFileWatcher_Deleted_$Path"
    }
    if ($NotifyFilter -contains 'Renamed' -or $NotifyFilter -contains 'All') {
        Register-ObjectEvent @eventParams -EventName Renamed -SourceIdentifier "AgentFileWatcher_Renamed_$Path"
    }
    
    Write-AgentLog -Message "File watcher started for: $Path" -Level Info -Category 'FileWatcher'
    
    if ($PassThru) {
        return $watcher
    }
}

#endregion

#region Process Management

function Get-AgentProcess {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Name,
        
        [Parameter()]
        [int]$Id,
        
        [Parameter()]
        [switch]$IncludeAgentProcesses
    )
    
    $filter = @{}
    if ($Name) { $filter.Name = $Name }
    if ($Id) { $filter.Id = $Id }
    
    $processes = Get-Process @filter | ForEach-Object {
        [PSCustomObject]@{
            Id = $_.Id
            Name = $_.Name
            Path = $_.Path
            WorkingSetMB = [math]::Round($_.WorkingSet64 / 1MB, 2)
            CPU = $_.CPU
            StartTime = $_.StartTime
            Threads = $_.Threads.Count
            Handles = $_.Handles
            IsAgentProcess = $_.Path -like "*$env:AI_AGENT_PATH*"
        }
    }
    
    if (-not $IncludeAgentProcesses) {
        $processes = $processes | Where-Object { -not $_.IsAgentProcess }
    }
    
    return $processes
}

function Stop-AgentProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, ParameterSetName = 'ById')]
        [int]$Id,
        
        [Parameter(Mandatory, ParameterSetName = 'ByName')]
        [string]$Name,
        
        [Parameter()]
        [switch]$Force
    )
    
    $processParams = @{}
    if ($Id) { $processParams.Id = $Id }
    if ($Name) { $processParams.Name = $Name }
    
    $process = Get-Process @processParams -ErrorAction Stop
    
    if ($Force) {
        Stop-Process -InputObject $process -Force
    } else {
        # Graceful shutdown attempt
        $process.CloseMainWindow() | Out-Null
        Start-Sleep -Seconds 2
        
        if (-not $process.HasExited) {
            Stop-Process -InputObject $process -Force
        }
    }
    
    Write-AgentLog -Message "Stopped process: $($process.Name) (ID: $($process.Id))" -Level Info -Category 'Process'
}

function Start-AgentProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        
        [Parameter()]
        [string]$ArgumentList,
        
        [Parameter()]
        [string]$WorkingDirectory,
        
        [Parameter()]
        [switch]$Wait,
        
        [Parameter()]
        [switch]$WindowStyleHidden,
        
        [Parameter()]
        [switch]$PassThru
    )
    
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $ArgumentList
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $WindowStyleHidden
    
    if ($WorkingDirectory) {
        $startInfo.WorkingDirectory = $WorkingDirectory
    }
    
    $process = [System.Diagnostics.Process]::Start($startInfo)
    
    Write-AgentLog -Message "Started process: $FilePath (ID: $($process.Id))" -Level Info -Category 'Process'
    
    if ($Wait) {
        $process.WaitForExit()
        return [PSCustomObject]@{
            Process = $process
            ExitCode = $process.ExitCode
        }
    }
    
    if ($PassThru) {
        return $process
    }
    
    return [PSCustomObject]@{
        Id = $process.Id
        Name = $process.ProcessName
        Started = $process.StartTime
    }
}

#endregion

#region Export

Export-ModuleMember -Function @(
    'New-AgentScheduledTask'
    'Get-AgentScheduledTask'
    'Remove-AgentScheduledTask'
    'Invoke-AgentFileOperation'
    'Start-AgentFileWatcher'
    'Get-AgentProcess'
    'Stop-AgentProcess'
    'Start-AgentProcess'
)

#endregion
```

---

## Security & Script Signing

### 6.1 Certificate Management

```powershell
# CertificateManager.psm1
#Requires -Version 5.1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Certificate management for script signing and encryption
.DESCRIPTION
    Manages code signing certificates and secure credential storage
#>

class CertificateManager {
    [string]$CertStoreLocation
    [string]$AgentCertName
    
    CertificateManager() {
        $this.CertStoreLocation = 'Cert:\CurrentUser\My'
        $this.AgentCertName = 'AIAgentCodeSigning'
    }
    
    [System.Security.Cryptography.X509Certificates.X509Certificate2] CreateSigningCertificate() {
        # Check if certificate already exists
        $existingCert = Get-ChildItem -Path $this.CertStoreLocation | 
            Where-Object { $_.Subject -eq "CN=$($this.AgentCertName)" } | 
            Select-Object -First 1
        
        if ($existingCert) {
            Write-Host "Using existing certificate: $($existingCert.Thumbprint)" -ForegroundColor Yellow
            return $existingCert
        }
        
        # Create self-signed certificate for code signing
        $certParams = @{
            Subject = "CN=$($this.AgentCertName)"
            CertStoreLocation = $this.CertStoreLocation
            KeyUsage = @('DigitalSignature')
            Type = 'CodeSigningCert'
            NotAfter = (Get-Date).AddYears(5)
            KeySpec = 'Signature'
            KeyLength = 2048
            HashAlgorithm = 'SHA256'
            ProviderName = 'Microsoft Enhanced RSA and AES Cryptographic Provider'
        }
        
        $cert = New-SelfSignedCertificate @certParams
        
        Write-Host "Created signing certificate: $($cert.Thumbprint)" -ForegroundColor Green
        
        # Export certificate for backup
        $exportPath = Join-Path $env:AI_AGENT_CONFIG_PATH "AgentSigningCert.cer"
        Export-Certificate -Cert $cert -FilePath $exportPath -Type CERT | Out-Null
        
        Write-Host "Certificate exported to: $exportPath" -ForegroundColor Gray
        
        return $cert
    }
    
    [void] ImportCertificate([string]$certPath) {
        $cert = Import-Certificate -FilePath $certPath -CertStoreLocation $this.CertStoreLocation
        Write-Host "Imported certificate: $($cert.Thumbprint)" -ForegroundColor Green
    }
    
    [void] AddToTrustedPublishers([System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate) {
        $trustedPublisherPath = 'Cert:\LocalMachine\TrustedPublisher'
        
        # Export and import to trusted publishers
        $tempPath = [System.IO.Path]::GetTempFileName()
        Export-Certificate -Cert $Certificate -FilePath $tempPath -Type CERT | Out-Null
        Import-Certificate -FilePath $tempPath -CertStoreLocation $trustedPublisherPath | Out-Null
        Remove-Item -Path $tempPath -Force
        
        Write-Host "Certificate added to Trusted Publishers" -ForegroundColor Green
    }
    
    [System.Security.Cryptography.X509Certificates.X509Certificate2] GetSigningCertificate() {
        return Get-ChildItem -Path $this.CertStoreLocation -CodeSigningCert | 
            Where-Object { $_.Subject -eq "CN=$($this.AgentCertName)" } | 
            Select-Object -First 1
    }
}

function New-AgentSigningCertificate {
    [CmdletBinding()]
    param()
    
    $manager = [CertificateManager]::new()
    return $manager.CreateSigningCertificate()
}

function Get-AgentSigningCertificate {
    [CmdletBinding()]
    param()
    
    $manager = [CertificateManager]::new()
    return $manager.GetSigningCertificate()
}

function Add-AgentCertificateToTrustedPublishers {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
    )
    
    $manager = [CertificateManager]::new()
    $manager.AddToTrustedPublishers($Certificate)
}

Export-ModuleMember -Function @(
    'New-AgentSigningCertificate'
    'Get-AgentSigningCertificate'
    'Add-AgentCertificateToTrustedPublishers'
)
```

### 6.2 Script Signing Module

```powershell
# ScriptSigner.psm1
#Requires -Version 5.1
#Requires -Modules CertificateManager

<#
.SYNOPSIS
    Script signing module for AI Agent
.DESCRIPTION
    Signs PowerShell scripts and validates signatures
#>

function Sign-AgentScript {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ScriptPath,
        
        [Parameter()]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
        
        [Parameter()]
        [switch]$Timestamp,
        
        [Parameter()]
        [string]$TimestampServer = 'http://timestamp.digicert.com'
    )
    
    # Validate script path
    if (-not (Test-Path $ScriptPath)) {
        throw "Script not found: $ScriptPath"
    }
    
    # Get certificate if not provided
    if (-not $Certificate) {
        $Certificate = Get-AgentSigningCertificate
        if (-not $Certificate) {
            throw "No signing certificate found. Run New-AgentSigningCertificate first."
        }
    }
    
    # Sign the script
    $signParams = @{
        FilePath = $ScriptPath
        Certificate = $Certificate
    }
    
    if ($Timestamp) {
        $signParams['TimestampServer'] = $TimestampServer
    }
    
    $signature = Set-AuthenticodeSignature @signParams
    
    if ($signature.Status -eq 'Valid') {
        Write-Host "Script signed successfully: $ScriptPath" -ForegroundColor Green
    } else {
        Write-Error "Failed to sign script: $($signature.StatusMessage)"
    }
    
    return $signature
}

function Test-AgentScriptSignature {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ScriptPath
    )
    
    if (-not (Test-Path $ScriptPath)) {
        throw "Script not found: $ScriptPath"
    }
    
    $signature = Get-AuthenticodeSignature -FilePath $ScriptPath
    
    return [PSCustomObject]@{
        Path = $ScriptPath
        Status = $signature.Status
        StatusMessage = $signature.StatusMessage
        SignerCertificate = $signature.SignerCertificate.Subject
        TimeStamperCertificate = $signature.TimeStamperCertificate?.Subject
        IsValid = $signature.Status -eq 'Valid'
    }
}

function Sign-AgentModule {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ModulePath,
        
        [Parameter()]
        [switch]$Recurse,
        
        [Parameter()]
        [switch]$IncludeManifests
    )
    
    $scripts = Get-ChildItem -Path $ModulePath -Filter '*.ps1' -Recurse:$Recurse
    
    if ($IncludeManifests) {
        $scripts += Get-ChildItem -Path $ModulePath -Filter '*.psd1' -Recurse:$Recurse
        $scripts += Get-ChildItem -Path $ModulePath -Filter '*.psm1' -Recurse:$Recurse
    }
    
    $results = @()
    
    foreach ($script in $scripts) {
        try {
            $signature = Sign-AgentScript -ScriptPath $script.FullName
            $results += [PSCustomObject]@{
                Path = $script.FullName
                Status = 'Signed'
                Thumbprint = $signature.SignerCertificate.Thumbprint
            }
        } catch {
            $results += [PSCustomObject]@{
                Path = $script.FullName
                Status = 'Failed'
                Error = $_.Exception.Message
            }
        }
    }
    
    return $results
}

Export-ModuleMember -Function @(
    'Sign-AgentScript'
    'Test-AgentScriptSignature'
    'Sign-AgentModule'
)
```

---

## Error Handling & Output Parsing

### 7.1 Error Handling Framework

```powershell
# ErrorHandler.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    Error handling framework for AI Agent PowerShell operations
.DESCRIPTION
    Provides structured error handling, logging, and recovery mechanisms
#>

class AgentErrorRecord {
    [string]$ErrorId
    [string]$Category
    [string]$Message
    [string]$ScriptStackTrace
    [string]$ExceptionType
    [hashtable]$TargetObject
    [datetime]$Timestamp
    [string]$Remediation
    [bool]$IsRecoverable
    
    AgentErrorRecord([System.Management.Automation.ErrorRecord]$ErrorRecord) {
        $this.ErrorId = $ErrorRecord.FullyQualifiedErrorId
        $this.Category = $ErrorRecord.CategoryInfo.Category.ToString()
        $this.Message = $ErrorRecord.Exception.Message
        $this.ScriptStackTrace = $ErrorRecord.ScriptStackTrace
        $this.ExceptionType = $ErrorRecord.Exception.GetType().Name
        $this.TargetObject = $ErrorRecord.TargetObject
        $this.Timestamp = Get-Date
        $this.IsRecoverable = $this.DetermineRecoverability($ErrorRecord)
        $this.Remediation = $this.SuggestRemediation($ErrorRecord)
    }
    
    [bool] DetermineRecoverability([System.Management.Automation.ErrorRecord]$ErrorRecord) {
        $recoverableErrors = @(
            'FileNotFoundException'
            'DirectoryNotFoundException'
            'PathTooLongException'
            'UnauthorizedAccessException'
            'IOException'
        )
        
        return $ErrorRecord.Exception.GetType().Name -in $recoverableErrors
    }
    
    [string] SuggestRemediation([System.Management.Automation.ErrorRecord]$ErrorRecord) {
        switch ($ErrorRecord.Exception.GetType().Name) {
            'FileNotFoundException' { return 'Verify file path exists and is accessible' }
            'DirectoryNotFoundException' { return 'Create directory or verify path' }
            'UnauthorizedAccessException' { return 'Run with elevated privileges or adjust permissions' }
            'PathTooLongException' { return 'Use shorter path or enable long path support' }
            default { return 'Review error details and retry operation' }
        }
    }
    
    [PSCustomObject] ToObject() {
        return [PSCustomObject]@{
            ErrorId = $this.ErrorId
            Category = $this.Category
            Message = $this.Message
            ExceptionType = $this.ExceptionType
            Timestamp = $this.Timestamp
            IsRecoverable = $this.IsRecoverable
            Remediation = $this.Remediation
        }
    }
}

function Invoke-AgentOperation {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [scriptblock]$Operation,
        
        [Parameter()]
        [int]$MaxRetries = 3,
        
        [Parameter()]
        [int]$RetryDelaySeconds = 5,
        
        [Parameter()]
        [scriptblock]$OnError = $null,
        
        [Parameter()]
        [scriptblock]$OnSuccess = $null,
        
        [Parameter()]
        [switch]$ContinueOnError
    )
    
    $attempt = 0
    $lastError = $null
    
    while ($attempt -lt $MaxRetries) {
        $attempt++
        
        try {
            $result = & $Operation
            
            if ($OnSuccess) {
                & $OnSuccess $result
            }
            
            return [PSCustomObject]@{
                Success = $true
                Result = $result
                Attempts = $attempt
                Duration = $null
            }
            
        } catch {
            $lastError = $_
            $errorRecord = [AgentErrorRecord]::new($lastError)
            
            Write-AgentLog -Message "Operation failed (attempt $attempt/$MaxRetries): $($errorRecord.Message)" `
                -Level Warning -Category 'Operation'
            
            if ($OnError) {
                & $OnError $errorRecord
            }
            
            if (-not $errorRecord.IsRecoverable) {
                break
            }
            
            if ($attempt -lt $MaxRetries) {
                Write-AgentLog -Message "Waiting $RetryDelaySeconds seconds before retry..." -Level Info
                Start-Sleep -Seconds $RetryDelaySeconds
            }
        }
    }
    
    # All retries exhausted
    $finalError = [AgentErrorRecord]::new($lastError)
    
    Write-AgentLog -Message "Operation failed after $MaxRetries attempts: $($finalError.Message)" `
        -Level Error -Category 'Operation'
    
    if ($ContinueOnError) {
        return [PSCustomObject]@{
            Success = $false
            Error = $finalError.ToObject()
            Attempts = $attempt
        }
    } else {
        throw $lastError
    }
}

function ConvertTo-AgentErrorReport {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [System.Management.Automation.ErrorRecord[]]$ErrorRecord
    )
    
    process {
        foreach ($record in $ErrorRecord) {
            $agentError = [AgentErrorRecord]::new($record)
            $agentError.ToObject()
        }
    }
}

Export-ModuleMember -Function @(
    'Invoke-AgentOperation'
    'ConvertTo-AgentErrorReport'
)
```

### 7.2 Output Parsing Module

```powershell
# OutputParser.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    Output parsing utilities for AI Agent
.DESCRIPTION
    Parses and structures PowerShell command output for agent consumption
#>

function ConvertTo-AgentStructuredOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [object]$InputObject,
        
        [Parameter()]
        [ValidateSet('Json', 'Xml', 'Csv', 'Hashtable', 'Object')]
        [string]$Format = 'Json',
        
        [Parameter()]
        [int]$Depth = 10,
        
        [Parameter()]
        [string[]]$IncludeProperties,
        
        [Parameter()]
        [string[]]$ExcludeProperties
    )
    
    process {
        $output = $InputObject
        
        # Filter properties if specified
        if ($IncludeProperties) {
            $output = $output | Select-Object -Property $IncludeProperties
        } elseif ($ExcludeProperties) {
            $output = $output | Select-Object -Property * -ExcludeProperty $ExcludeProperties
        }
        
        # Convert to requested format
        switch ($Format) {
            'Json' {
                return $output | ConvertTo-Json -Depth $Depth -Compress
            }
            'Xml' {
                return $output | ConvertTo-Xml -As String -Depth $Depth
            }
            'Csv' {
                return $output | ConvertTo-Csv -NoTypeInformation
            }
            'Hashtable' {
                return ConvertTo-Hashtable -InputObject $output
            }
            'Object' {
                return $output
            }
        }
    }
}

function ConvertTo-Hashtable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [object]$InputObject
    )
    
    process {
        if ($InputObject -is [System.Collections.IEnumerable] -and $InputObject -isnot [string]) {
            $collection = @()
            foreach ($item in $InputObject) {
                $collection += ConvertTo-Hashtable -InputObject $item
            }
            return $collection
        } elseif ($InputObject -is [PSObject]) {
            $hash = @{}
            foreach ($property in $InputObject.PSObject.Properties) {
                $hash[$property.Name] = ConvertTo-Hashtable -InputObject $property.Value
            }
            return $hash
        } else {
            return $InputObject
        }
    }
}

function Parse-AgentCommandOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Output,
        
        [Parameter()]
        [ValidateSet('Table', 'List', 'Csv', 'Json', 'Raw')]
        [string]$Format = 'Raw',
        
        [Parameter()]
        [hashtable]$ColumnMap = @{}
    )
    
    switch ($Format) {
        'Table' {
            return Parse-TableOutput -Output $Output -ColumnMap $ColumnMap
        }
        'List' {
            return Parse-ListOutput -Output $Output
        }
        'Csv' {
            return $Output | ConvertFrom-Csv
        }
        'Json' {
            return $Output | ConvertFrom-Json
        }
        'Raw' {
            return $Output
        }
    }
}

function Parse-TableOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Output,
        
        [Parameter()]
        [hashtable]$ColumnMap
    )
    
    $lines = $Output -split "`r?`n" | Where-Object { $_.Trim() }
    
    if ($lines.Count -lt 2) {
        return @()
    }
    
    # Parse header line
    $headerLine = $lines[0]
    $dataLines = $lines[2..($lines.Count - 1)]  # Skip header and separator
    
    # Extract column positions from header
    $columns = @()
    $matches = [regex]::Matches($headerLine, '\S+')
    foreach ($match in $matches) {
        $columns += @{
            Name = $match.Value
            Start = $match.Index
            Length = $match.Length
        }
    }
    
    # Parse data lines
    $results = @()
    foreach ($line in $dataLines) {
        $row = @{}
        foreach ($column in $columns) {
            $value = $line.Substring($column.Start, [Math]::Min($column.Length, $line.Length - $column.Start)).Trim()
            
            # Apply column mapping if provided
            $propertyName = if ($ColumnMap[$column.Name]) { $ColumnMap[$column.Name] } else { $column.Name }
            $row[$propertyName] = $value
        }
        $results += [PSCustomObject]$row
    }
    
    return $results
}

function Parse-ListOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Output
    )
    
    $lines = $Output -split "`r?`n" | Where-Object { $_.Trim() }
    $result = @{}
    
    foreach ($line in $lines) {
        if ($line -match '^(.+?)\s*:\s*(.*)$') {
            $propertyName = $matches[1].Trim()
            $propertyValue = $matches[2].Trim()
            $result[$propertyName] = $propertyValue
        }
    }
    
    return [PSCustomObject]$result
}

Export-ModuleMember -Function @(
    'ConvertTo-AgentStructuredOutput'
    'ConvertTo-Hashtable'
    'Parse-AgentCommandOutput'
)
```

---

## Cmdlet Wrappers

### 8.1 Common Operations Wrappers

```powershell
# AgentCmdlets.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    Common cmdlet wrappers for AI Agent operations
.DESCRIPTION
    Provides simplified interfaces for common Windows management tasks
#>

#region Registry Operations

function Get-AgentRegistryValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        
        [Parameter()]
        [string]$Name,
        
        [Parameter()]
        [string]$ComputerName = 'localhost'
    )
    
    $fullPath = if ($Name) { Join-Path $Path $Name } else { $Path }
    
    try {
        if ($ComputerName -eq 'localhost') {
            $value = Get-ItemProperty -Path $Path -Name $Name -ErrorAction Stop
            return $value.$Name
        } else {
            $value = Invoke-Command -ComputerName $ComputerName -ScriptBlock {
                param($p, $n)
                (Get-ItemProperty -Path $p -Name $n -ErrorAction Stop).$n
            } -ArgumentList $Path, $Name
            return $value
        }
    } catch {
        Write-Error "Failed to read registry value: $_"
        return $null
    }
}

function Set-AgentRegistryValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        
        [Parameter(Mandatory)]
        [string]$Name,
        
        [Parameter(Mandatory)]
        [object]$Value,
        
        [Parameter()]
        [ValidateSet('String', 'ExpandString', 'Binary', 'DWord', 'MultiString', 'QWord')]
        [string]$Type = 'String'
    )
    
    # Ensure path exists
    if (-not (Test-Path $Path)) {
        New-Item -Path $Path -Force | Out-Null
    }
    
    Set-ItemProperty -Path $Path -Name $Name -Value $Value -Type $Type -Force
    Write-AgentLog -Message "Registry value set: $Path\$Name" -Level Info
}

#endregion

#region Environment Variables

function Get-AgentEnvironmentVariable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        
        [Parameter()]
        [ValidateSet('Process', 'User', 'Machine')]
        [string]$Scope = 'Process'
    )
    
    return [Environment]::GetEnvironmentVariable($Name, $Scope)
}

function Set-AgentEnvironmentVariable {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        
        [Parameter(Mandatory)]
        [string]$Value,
        
        [Parameter()]
        [ValidateSet('Process', 'User', 'Machine')]
        [string]$Scope = 'Process'
    )
    
    [Environment]::SetEnvironmentVariable($Name, $Value, $Scope)
    Write-AgentLog -Message "Environment variable set: $Name = $Value ($Scope)" -Level Info
}

#endregion

#region Network Operations

function Test-AgentNetworkConnection {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ComputerName,
        
        [Parameter()]
        [int]$Port,
        
        [Parameter()]
        [int]$TimeoutMilliseconds = 5000
    )
    
    $result = @{
        ComputerName = $ComputerName
        Port = $Port
        IsReachable = $false
        ResponseTime = $null
        Error = $null
    }
    
    try {
        if ($Port) {
            # Test specific port
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $connection = $tcpClient.BeginConnect($ComputerName, $Port, $null, $null)
            $success = $connection.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false)
            
            if ($success) {
                $tcpClient.EndConnect($connection)
                $result.IsReachable = $true
            }
            $tcpClient.Close()
        } else {
            # Test ICMP ping
            $ping = New-Object System.Net.NetworkInformation.Ping
            $pingResult = $ping.Send($ComputerName, $TimeoutMilliseconds)
            $result.IsReachable = $pingResult.Status -eq 'Success'
            $result.ResponseTime = $pingResult.RoundtripTime
        }
    } catch {
        $result.Error = $_.Exception.Message
    }
    
    return [PSCustomObject]$result
}

function Get-AgentNetworkConfiguration {
    [CmdletBinding()]
    param()
    
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }
    $configurations = @()
    
    foreach ($adapter in $adapters) {
        $ipConfig = Get-NetIPConfiguration -InterfaceIndex $adapter.InterfaceIndex
        $dns = Get-DnsClientServerAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4
        
        $configurations += [PSCustomObject]@{
            Name = $adapter.Name
            InterfaceDescription = $adapter.InterfaceDescription
            MacAddress = $adapter.MacAddress
            LinkSpeed = $adapter.LinkSpeed
            IPAddress = $ipConfig.IPv4Address.IPAddress
            SubnetMask = $ipConfig.IPv4Address.PrefixLength
            DefaultGateway = $ipConfig.IPv4DefaultGateway.NextHop
            DNSServers = $dns.ServerAddresses
            IsDHCP = $ipConfig.NetIPv4Interface.DHCP -eq 'Enabled'
        }
    }
    
    return $configurations
}

#endregion

#region User & Group Operations

function Get-AgentLocalUser {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$UserName
    )
    
    $filter = if ($UserName) { "Name = '$UserName'" } else { $null }
    
    return Get-CimInstance -ClassName Win32_UserAccount -Filter $filter | ForEach-Object {
        [PSCustomObject]@{
            Name = $_.Name
            FullName = $_.FullName
            SID = $_.SID
            Disabled = $_.Disabled
            PasswordChangeable = $_.PasswordChangeable
            PasswordExpires = $_.PasswordExpires
            PasswordRequired = $_.PasswordRequired
            AccountType = $_.AccountType
        }
    }
}

function Get-AgentLocalGroup {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$GroupName
    )
    
    $filter = if ($GroupName) { "Name = '$GroupName'" } else { $null }
    
    return Get-CimInstance -ClassName Win32_Group -Filter $filter | ForEach-Object {
        $members = Get-CimAssociatedInstance -InputObject $_ -ResultClassName Win32_UserAccount | 
            Select-Object -ExpandProperty Name
        
        [PSCustomObject]@{
            Name = $_.Name
            SID = $_.SID
            Description = $_.Description
            Members = $members
        }
    }
}

#endregion

#region Windows Update

function Get-AgentWindowsUpdate {
    [CmdletBinding()]
    param(
        [Parameter()]
        [switch]$IncludeInstalled,
        
        [Parameter()]
        [switch]$IncludeHidden
    )
    
    try {
        $updateSession = New-Object -ComObject Microsoft.Update.Session
        $updateSearcher = $updateSession.CreateUpdateSearcher()
        
        $searchCriteria = if ($IncludeInstalled) { 'IsInstalled=1' } else { 'IsInstalled=0' }
        if (-not $IncludeHidden) {
            $searchCriteria += ' and IsHidden=0'
        }
        
        $searchResult = $updateSearcher.Search($searchCriteria)
        
        return $searchResult.Updates | ForEach-Object {
            [PSCustomObject]@{
                Title = $_.Title
                Description = $_.Description
                IsInstalled = $_.IsInstalled
                IsDownloaded = $_.IsDownloaded
                IsMandatory = $_.IsMandatory
                RebootRequired = $_.RebootRequired
                EulaAccepted = $_.EulaAccepted
                Size = $_.Size
                LastDeploymentChangeTime = $_.LastDeploymentChangeTime
                Categories = $_.Categories | Select-Object -ExpandProperty Name
            }
        }
    } catch {
        Write-Error "Failed to retrieve Windows Updates: $_"
        return @()
    }
}

#endregion

#region Export

Export-ModuleMember -Function @(
    'Get-AgentRegistryValue'
    'Set-AgentRegistryValue'
    'Get-AgentEnvironmentVariable'
    'Set-AgentEnvironmentVariable'
    'Test-AgentNetworkConnection'
    'Get-AgentNetworkConfiguration'
    'Get-AgentLocalUser'
    'Get-AgentLocalGroup'
    'Get-AgentWindowsUpdate'
)

#endregion
```

---

## Implementation Examples

### 9.1 Integration Example

```javascript
// agent-powershell-integration.js
const { PowerShellBridge, PowerShellSession } = require('./powershell-bridge');
const { EventEmitter } = require('events');

class AgentPowerShellIntegration extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      agentPath: config.agentPath || process.env.AI_AGENT_PATH,
      logPath: config.logPath || `${process.env.AI_AGENT_PATH}/Logs`,
      executionPolicy: config.executionPolicy || 'RemoteSigned',
      ...config
    };
    
    this.bridge = new PowerShellBridge({
      executionPolicy: this.config.executionPolicy
    });
    
    this.session = null;
    this.isInitialized = false;
  }

  async initialize() {
    // Initialize PowerShell session
    this.session = new PowerShellSession();
    await this.session.initialize();
    
    // Load agent modules
    const modulesPath = `${this.config.agentPath}/Modules`;
    await this.session.loadModule(`${modulesPath}/AgentCore.psm1`);
    await this.session.loadModule(`${modulesPath}/AgentOperations.psm1`);
    await this.session.loadModule(`${modulesPath}/WMIProvider.psm1`);
    
    // Initialize agent environment
    await this.session.execute(`
      Initialize-AgentEnvironment -Configuration @{
        AgentId = '${this.generateAgentId()}'
        Name = 'AIAgent'
        Version = '1.0.0'
        LogPath = '${this.config.logPath}'
        ScriptPath = '${this.config.agentPath}/Scripts'
        ConfigPath = '${this.config.agentPath}/Config'
        DataPath = '${this.config.agentPath}/Data'
      }
    `);
    
    // Start heartbeat
    await this.session.execute(`
      Start-AgentHeartbeat -IntervalSeconds 60
    `);
    
    this.isInitialized = true;
    this.emit('initialized');
    
    return this;
  }

  async executeCommand(command, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Agent PowerShell integration not initialized');
    }

    try {
      const result = await this.session.executeStructured(command, options.expectJson);
      this.emit('commandSuccess', { command, result });
      return result;
    } catch (error) {
      this.emit('commandError', { command, error });
      throw error;
    }
  }

  async getSystemStatus() {
    return this.executeCommand('Get-AgentSystemStatus', { expectJson: true });
  }

  async getProcesses(filter = {}) {
    const filterString = Object.entries(filter)
      .map(([k, v]) => `${k} = '${v}'`)
      .join(' and ');
    
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetProcesses(@{${filterString}}) | Select-Object Name, ProcessId, WorkingSetSize
    `, { expectJson: true });
  }

  async getServices(state = null, startMode = null) {
    const params = [];
    if (state) params.push(`-State '${state}'`);
    if (startMode) params.push(`-StartMode '${startMode}'`);
    
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetServices(${params.join(' ')})
    `, { expectJson: true });
  }

  async startService(serviceName) {
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.StartService('${serviceName}')
    `, { expectJson: true });
  }

  async stopService(serviceName) {
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.StopService('${serviceName}')
    `, { expectJson: true });
  }

  async scheduleTask(name, scriptBlock, options = {}) {
    const psScriptBlock = scriptBlock.toString().replace(/'/g, "''");
    
    return this.executeCommand(`
      New-AgentScheduledTask -Name '${name}' -Action {
        ${psScriptBlock}
      } -Trigger '${options.trigger || 'Once'}'
    `, { expectJson: true });
  }

  async fileOperation(operation, source, destination = null, options = {}) {
    const params = [
      `-Operation '${operation}'`,
      `-Source '${source}'`
    ];
    
    if (destination) params.push(`-Destination '${destination}'`);
    if (options.filter) params.push(`-Filter '${options.filter}'`);
    if (options.recurse) params.push('-Recurse');
    if (options.force) params.push('-Force');
    
    return this.executeCommand(`
      Invoke-AgentFileOperation ${params.join(' ')}
    `, { expectJson: true });
  }

  async dispose() {
    if (this.session) {
      await this.session.execute('Stop-AgentHeartbeat');
      await this.session.dispose();
    }
    this.isInitialized = false;
    this.emit('disposed');
  }

  generateAgentId() {
    return `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

module.exports = { AgentPowerShellIntegration };
```

---

## Summary

This PowerShell Automation and Scripting Integration specification provides:

### Core Components
1. **PowerShell Bridge Layer** - Node.js to PowerShell communication via child_process
2. **Execution Policy Management** - Secure script execution configuration
3. **PowerShell Remoting** - Distributed system management capabilities
4. **WMI Integration** - Comprehensive system information access
5. **Agent Modules** - Core operations, task scheduling, file/process management
6. **Security Framework** - Script signing, certificate management
7. **Error Handling** - Structured error processing and recovery
8. **Output Parsing** - Structured data conversion for agent consumption
9. **Cmdlet Wrappers** - Simplified interfaces for common operations

### Key Features
- Full Windows 10 system access through native PowerShell
- Secure execution with policy management and script signing
- Persistent sessions for stateful operations
- Comprehensive error handling with retry mechanisms
- Structured output for AI agent consumption
- WMI/CIM integration for system management
- Task scheduling and automation capabilities
- File system monitoring and operations
- Process and service management
- Network and registry operations

### Security Considerations
- Execution policy enforcement
- Script signing with certificates
- JEA (Just Enough Administration) endpoints
- Secure credential storage
- Audit logging and transcript recording
- Access control and permissions management

This specification enables the AI agent to leverage PowerShell's extensive Windows management capabilities while maintaining security and providing structured interfaces for agent integration.
