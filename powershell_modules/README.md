# PowerShell Automation & Scripting Integration for AI Agent

A comprehensive PowerShell integration layer for Windows 10 AI Agent systems, providing full system access, WMI integration, task automation, and secure script execution.

## Overview

This integration layer enables AI agents to:
- Execute PowerShell commands from Node.js
- Access Windows Management Instrumentation (WMI) for system information
- Manage processes, services, and scheduled tasks
- Perform file operations and registry modifications
- Monitor system performance and event logs
- Execute scripts securely with proper signing and policies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent Core (Node.js)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         AgentPowerShellIntegration Class              │   │
│  │  - High-level API for PowerShell operations           │   │
│  │  - Event-driven architecture                          │   │
│  │  - Automatic JSON serialization                       │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────┴───────────────────────────────┐   │
│  │              PowerShellBridge / Session               │   │
│  │  - child_process integration                          │   │
│  │  - Persistent session management                      │   │
│  │  - Error handling and timeouts                        │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    Windows 10 System                          │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PowerShell   │  │    WMI      │  │   Windows APIs      │  │
│  │   Engine     │  │  Providers  │  │  (Win32/PInvoke)    │  │
│  └──────────────┘  └─────────────┘  └─────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## Components

### 1. PowerShell Bridge (`powershell-bridge.js`)
Core Node.js module for PowerShell communication:
- `PowerShellBridge` - One-shot command execution
- `PowerShellSession` - Persistent session management
- Custom error classes for detailed error handling

### 2. Agent Integration (`agent-powershell-integration.js`)
High-level API class providing:
- System information retrieval
- Process and service management
- File and registry operations
- Network operations
- Task scheduling
- Logging integration

### 3. PowerShell Modules

#### AgentCore.psm1
Core agent functionality:
- Environment initialization
- Heartbeat system
- Identity management
- Structured logging

#### WMIProvider.psm1
WMI integration classes:
- System information queries
- Process and service management
- Performance monitoring
- Event log access
- Network information

#### AgentOperations.psm1
Task automation:
- Scheduled task management
- File operations (copy, move, delete, archive, sync)
- File system watchers
- Process management

#### AgentCmdlets.psm1
Common operation wrappers:
- Registry operations
- Environment variables
- Network testing
- User/group management
- Windows Update queries

### 4. Setup Scripts

#### setup-agent.ps1
Complete installation script that:
- Creates directory structure
- Configures execution policy
- Installs modules
- Creates code signing certificate
- Configures WinRM
- Sets up environment variables

#### Configure-AgentExecutionPolicy.ps1
Execution policy configuration for different environments:
- Development: RemoteSigned
- Production: AllSigned
- Restricted: Restricted

## Installation

### Prerequisites
- Windows 10 (version 1809 or later)
- PowerShell 5.1 or PowerShell 7.x
- Node.js 14.x or later
- Administrator privileges (for initial setup)

### Quick Setup

1. **Run the setup script as Administrator:**
```powershell
# Open PowerShell as Administrator
& ".\setup-agent.ps1" -Environment Development
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Use in your Node.js application:**
```javascript
const { AgentPowerShellIntegration } = require('./agent-powershell-integration');

const agent = new AgentPowerShellIntegration({
  agentPath: 'C:\\AIAgent',
  executionPolicy: 'RemoteSigned'
});

await agent.initialize();
```

## Usage Examples

### Basic Initialization
```javascript
const { AgentPowerShellIntegration } = require('./agent-powershell-integration');

const agent = new AgentPowerShellIntegration({
  agentPath: 'C:\\AIAgent',
  executionPolicy: 'RemoteSigned',
  heartbeatInterval: 60,
  enableHeartbeat: true
});

// Listen to events
agent.on('initialized', (data) => console.log('Ready:', data));
agent.on('commandError', ({ error }) => console.error('Error:', error));

// Initialize
await agent.initialize();
```

### System Information
```javascript
// Get comprehensive system status
const status = await agent.getSystemStatus();
console.log('CPU:', status.Processor.Name);
console.log('Memory:', status.Memory.TotalGB, 'GB');

// Get agent identity
const identity = await agent.getIdentity(true);
console.log('Agent ID:', identity.AgentId);
```

### Process Management
```javascript
// Get all processes
const processes = await agent.getProcesses();

// Get specific process
const notepad = await agent.getProcesses({ Name: 'notepad.exe' });

// Start a process
await agent.startProcess('notepad.exe', '', { hidden: false });

// Stop a process
await agent.stopProcess(1234, true); // force = true
```

### Service Management
```javascript
// Get running services
const services = await agent.getServices('Running');

// Get automatic start services
const autoServices = await agent.getServices(null, 'Auto');

// Control services
await agent.startService('Spooler');
await agent.stopService('Spooler');
await agent.setServiceStartMode('Spooler', 'Manual');
```

### File Operations
```javascript
// Read file
const content = await agent.readFile('C:\\path\\to\\file.txt');

// Write file
await agent.writeFile('C:\\path\\to\\file.txt', 'Hello World');

// File operations
await agent.fileOperation('Copy', 'C:\\source', 'C:\\dest');
await agent.fileOperation('Move', 'C:\\source', 'C:\\dest');
await agent.fileOperation('Delete', 'C:\\path\\to\\delete');
await agent.fileOperation('Archive', 'C:\\source', 'C:\\backup.zip');
```

### Registry Operations
```javascript
// Read registry
const value = await agent.getRegistryValue(
  'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion',
  'CurrentVersion'
);

// Write registry
await agent.setRegistryValue(
  'HKCU:\\Software\\MyApp',
  'Setting',
  'Value',
  'String'
);
```

### Network Operations
```javascript
// Test connectivity
const result = await agent.testNetworkConnection('google.com');
console.log('Reachable:', result.IsReachable);

// Test port
const portResult = await agent.testNetworkConnection('localhost', 80);

// Get network config
const config = await agent.getNetworkConfiguration();
```

### Task Scheduling
```javascript
// Create scheduled task
await agent.createScheduledTask('MyTask', `
  Write-Host "Task executed"
  Get-Process | Export-Csv C:\\logs\\processes.csv
`, { trigger: 'Daily' });

// Get tasks
const tasks = await agent.getScheduledTasks();
```

### Custom PowerShell Commands
```javascript
// Execute custom command
const result = await agent.executeCommand(`
  Get-Process | Where-Object { $_.WorkingSet64 -gt 100MB } |
    Select-Object Name, Id, @{N='MemoryMB';E={[math]::Round($_.WorkingSet64/1MB,2)}}
`, { expectJson: true });
```

## Security

### Execution Policies
- **Development**: `RemoteSigned` - Allows local scripts, requires signature for remote
- **Production**: `AllSigned` - Requires all scripts to be signed
- **Restricted**: `Restricted` - No scripts allowed

### Script Signing
```powershell
# Create signing certificate
$cert = New-SelfSignedCertificate `
  -Subject "CN=AIAgentCodeSigning" `
  -CertStoreLocation "Cert:\CurrentUser\My" `
  -Type CodeSigningCert

# Sign a script
Set-AuthenticodeSignature -FilePath "script.ps1" -Certificate $cert
```

### JEA (Just Enough Administration)
Configure restricted endpoints for specific operations:
```powershell
# Create JEA endpoint
New-PSSessionConfigurationFile -Path "C:\\Config\\AgentJEA.pssc" `
  -SessionType RestrictedRemoteServer `
  -RunAsVirtualAccount `
  -VisibleCmdlets 'Get-Process', 'Get-Service'

Register-PSSessionConfiguration -Name "AIAgentJEA" -Path "C:\\Config\\AgentJEA.pssc"
```

## Configuration Options

### AgentPowerShellIntegration Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `agentPath` | string | `C:\AIAgent` | Installation directory |
| `logPath` | string | `<agentPath>\Logs` | Log file location |
| `executionPolicy` | string | `RemoteSigned` | PowerShell execution policy |
| `heartbeatInterval` | number | 60 | Heartbeat interval in seconds |
| `enableHeartbeat` | boolean | true | Enable heartbeat logging |

### PowerShellBridge Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `noProfile` | boolean | true | Skip PowerShell profile loading |
| `nonInteractive` | boolean | true | Run in non-interactive mode |
| `windowStyle` | string | `Hidden` | Window visibility |
| `timeout` | number | 30000 | Command timeout in milliseconds |
| `encoding` | string | `utf8` | Output encoding |

## Events

### AgentPowerShellIntegration Events
| Event | Payload | Description |
|-------|---------|-------------|
| `initialized` | `{ agentPath }` | Agent successfully initialized |
| `initializationError` | `Error` | Initialization failed |
| `moduleLoaded` | `{ module }` | Module loaded successfully |
| `moduleLoadError` | `{ module, error }` | Module load failed |
| `commandSuccess` | `{ command, result }` | Command executed successfully |
| `commandError` | `{ command, error }` | Command execution failed |
| `heartbeatStarted` | `{ interval }` | Heartbeat started |
| `disposed` | - | Agent disposed |

## Troubleshooting

### Common Issues

1. **Execution Policy Error**
```powershell
# Check current policy
Get-ExecutionPolicy -List

# Set appropriate policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. **WinRM Not Configured**
```powershell
# Enable PowerShell Remoting
Enable-PSRemoting -Force -SkipNetworkProfileCheck

# Configure trusted hosts
Set-Item WSMan:\localhost\Client\TrustedHosts -Value '*' -Force
```

3. **Module Import Failure**
```powershell
# Check module path
$env:PSModulePath

# Import manually
Import-Module "C:\\AIAgent\\Modules\\AgentCore.psm1" -Force -Verbose
```

### Debug Mode
```javascript
const agent = new AgentPowerShellIntegration({
  // ... config
});

// Listen to all events
agent.on('initialized', console.log);
agent.on('moduleLoaded', console.log);
agent.on('moduleLoadError', console.error);
agent.on('commandSuccess', console.log);
agent.on('commandError', console.error);
```

## API Reference

### AgentPowerShellIntegration Methods

#### System Information
- `getSystemStatus()` - Get comprehensive system status
- `getIdentity(includeSystemInfo)` - Get agent identity

#### Process Management
- `getProcesses(filter)` - Get process list
- `getProcessDetails(processId)` - Get process details
- `startProcess(filePath, args, options)` - Start process
- `stopProcess(processId, force)` - Stop process

#### Service Management
- `getServices(state, startMode)` - Get services
- `startService(serviceName)` - Start service
- `stopService(serviceName)` - Stop service
- `setServiceStartMode(serviceName, startMode)` - Set start mode

#### File Operations
- `readFile(filePath, encoding)` - Read file
- `writeFile(filePath, content, encoding)` - Write file
- `fileOperation(operation, source, destination, options)` - File operations

#### Registry Operations
- `getRegistryValue(path, name)` - Read registry value
- `setRegistryValue(path, name, value, type)` - Write registry value

#### Network Operations
- `testNetworkConnection(hostname, port)` - Test connectivity
- `getNetworkConfiguration()` - Get network config

#### Task Scheduling
- `createScheduledTask(name, scriptBlock, options)` - Create task
- `getScheduledTasks(name)` - Get tasks

#### Logging
- `writeLog(message, level, category)` - Write log
- `getLogs(options)` - Get logs

#### Utility
- `executeCommand(command, options)` - Execute custom command
- `dispose()` - Cleanup and dispose

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.

## Support

For issues and feature requests, please use the GitHub issue tracker.
