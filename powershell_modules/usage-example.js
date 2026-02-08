// usage-example.js
/**
 * AI Agent PowerShell Integration - Usage Examples
 * 
 * This file demonstrates how to use the AgentPowerShellIntegration
 * class for various system management operations.
 */

const { AgentPowerShellIntegration } = require('./agent-powershell-integration');

async function main() {
  // Create agent instance with custom configuration
  const agent = new AgentPowerShellIntegration({
    agentPath: 'C:\\AIAgent',
    executionPolicy: 'RemoteSigned',
    heartbeatInterval: 60,
    enableHeartbeat: true
  });

  // Listen to events
  agent.on('initialized', (data) => {
    console.log('Agent initialized:', data);
  });

  agent.on('commandSuccess', ({ command, result }) => {
    console.log('Command executed successfully');
  });

  agent.on('commandError', ({ command, error }) => {
    console.error('Command failed:', error);
  });

  try {
    // Initialize the agent
    console.log('Initializing agent...');
    await agent.initialize();

    // ==================== SYSTEM INFORMATION ====================
    console.log('\n=== System Information ===');
    const systemStatus = await agent.getSystemStatus();
    console.log('System Status:', JSON.stringify(systemStatus, null, 2));

    const identity = await agent.getIdentity(true);
    console.log('Agent Identity:', JSON.stringify(identity, null, 2));

    // ==================== PROCESS MANAGEMENT ====================
    console.log('\n=== Process Management ===');
    
    // Get all processes
    const processes = await agent.getProcesses();
    console.log(`Found ${processes.length} processes`);
    
    // Get specific process
    const notepadProcess = await agent.getProcesses({ Name: 'notepad.exe' });
    if (notepadProcess.length > 0) {
      console.log('Notepad process found:', notepadProcess[0]);
    }

    // ==================== SERVICE MANAGEMENT ====================
    console.log('\n=== Service Management ===');
    
    // Get running services
    const runningServices = await agent.getServices('Running');
    console.log(`Found ${runningServices.length} running services`);
    
    // Get automatic services
    const autoServices = await agent.getServices(null, 'Auto');
    console.log(`Found ${autoServices.length} automatic services`);

    // Example: Start/Stop service (commented out for safety)
    // await agent.startService('Spooler');
    // await agent.stopService('Spooler');

    // ==================== FILE OPERATIONS ====================
    console.log('\n=== File Operations ===');
    
    // Read file
    try {
      const content = await agent.readFile('C:\\Windows\\System32\\drivers\\etc\\hosts');
      console.log('Hosts file content (first 200 chars):', content.substring(0, 200));
    } catch (e) {
      console.log('Could not read hosts file:', e.message);
    }

    // Write file
    const testContent = 'This is a test file created by AI Agent';
    const testFilePath = 'C:\\AIAgent\\Test\\test-file.txt';
    await agent.writeFile(testFilePath, testContent);
    console.log('Test file written:', testFilePath);

    // File operations
    const copyResult = await agent.fileOperation('Copy', 
      'C:\\AIAgent\\Test\\test-file.txt',
      'C:\\AIAgent\\Test\\test-file-copy.txt'
    );
    console.log('Copy result:', copyResult);

    // ==================== REGISTRY OPERATIONS ====================
    console.log('\n=== Registry Operations ===');
    
    // Read registry value
    try {
      const currentVersion = await agent.getRegistryValue(
        'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion',
        'CurrentVersion'
      );
      console.log('Windows Current Version:', currentVersion);
    } catch (e) {
      console.log('Could not read registry:', e.message);
    }

    // Write registry value (commented out for safety)
    // await agent.setRegistryValue(
    //   'HKCU:\\Software\\AIAgent',
    //   'TestValue',
    //   'TestData',
    //   'String'
    // );

    // ==================== NETWORK OPERATIONS ====================
    console.log('\n=== Network Operations ===');
    
    // Test network connection
    const googleTest = await agent.testNetworkConnection('google.com');
    console.log('Google connectivity:', googleTest);

    const portTest = await agent.testNetworkConnection('localhost', 80);
    console.log('Localhost port 80:', portTest);

    // Get network configuration
    const networkConfig = await agent.getNetworkConfiguration();
    console.log('Network Configuration:', JSON.stringify(networkConfig, null, 2));

    // ==================== TASK SCHEDULING ====================
    console.log('\n=== Task Scheduling ===');
    
    // Create a scheduled task
    const taskScript = @'
Write-Host "Scheduled task executed at $(Get-Date)"
Get-Process | Select-Object -First 5 | Format-Table
'@;
    
    // await agent.createScheduledTask('TestTask', taskScript, { trigger: 'Once' });
    // console.log('Scheduled task created');

    // Get scheduled tasks
    const tasks = await agent.getScheduledTasks();
    console.log('Scheduled tasks:', tasks);

    // ==================== LOGGING ====================
    console.log('\n=== Logging ===');
    
    // Write log entry
    await agent.writeLog('Test log entry from AI Agent', 'Info', 'Test');
    
    // Get recent logs
    const logs = await agent.getLogs({ count: 10 });
    console.log('Recent logs:', logs);

    // ==================== WMI OPERATIONS ====================
    console.log('\n=== WMI Operations ===');
    
    // Get system performance
    const performance = await agent.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetSystemPerformance() | ConvertTo-Json -Depth 3
    `, { expectJson: true });
    console.log('System Performance:', JSON.stringify(performance, null, 2));

    // Get event logs
    const eventLogs = await agent.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetEventLogs('System', 5, 'Error') | ConvertTo-Json -Depth 3
    `, { expectJson: true });
    console.log('Recent System Errors:', eventLogs);

    console.log('\n=== All operations completed successfully ===');

  } catch (error) {
    console.error('Error:', error);
  } finally {
    // Cleanup
    console.log('\nCleaning up...');
    await agent.dispose();
    console.log('Agent disposed');
  }
}

// Run the example
main().catch(console.error);
