// agent-powershell-integration.js
const { PowerShellBridge, PowerShellSession } = require('./powershell-bridge');
const { EventEmitter } = require('events');
const path = require('path');

/**
 * AgentPowerShellIntegration - Main integration class for AI Agent PowerShell operations
 * 
 * Provides a high-level interface for:
 * - System information retrieval via WMI
 * - Process and service management
 * - File operations
 * - Task scheduling
 * - Registry operations
 * - Network operations
 */
class AgentPowerShellIntegration extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      agentPath: config.agentPath || process.env.AI_AGENT_PATH || 'C:\\AIAgent',
      logPath: config.logPath || path.join(process.env.AI_AGENT_PATH || 'C:\\AIAgent', 'Logs'),
      executionPolicy: config.executionPolicy || 'RemoteSigned',
      heartbeatInterval: config.heartbeatInterval || 60,
      enableHeartbeat: config.enableHeartbeat !== false,
      ...config
    };
    
    this.bridge = new PowerShellBridge({
      executionPolicy: this.config.executionPolicy
    });
    
    this.session = null;
    this.isInitialized = false;
    this.modulesPath = path.join(this.config.agentPath, 'Modules');
  }

  /**
   * Initialize the PowerShell integration
   * Loads modules and starts heartbeat if configured
   */
  async initialize() {
    try {
      // Initialize PowerShell session
      this.session = new PowerShellSession();
      await this.session.initialize();
      
      // Load agent modules
      await this.loadAgentModules();
      
      // Initialize agent environment
      await this.initializeEnvironment();
      
      // Start heartbeat if enabled
      if (this.config.enableHeartbeat) {
        await this.startHeartbeat();
      }
      
      this.isInitialized = true;
      this.emit('initialized', { agentPath: this.config.agentPath });
      
      return this;
    } catch (error) {
      this.emit('initializationError', error);
      throw error;
    }
  }

  /**
   * Load required PowerShell modules
   */
  async loadAgentModules() {
    const modules = [
      'AgentCore.psm1',
      'WMIProvider.psm1',
      'AgentOperations.psm1',
      'AgentCmdlets.psm1'
    ];

    for (const module of modules) {
      const modulePath = path.join(this.modulesPath, module);
      try {
        await this.session.loadModule(modulePath);
        this.emit('moduleLoaded', { module });
      } catch (error) {
        this.emit('moduleLoadError', { module, error: error.message });
        // Continue loading other modules
      }
    }
  }

  /**
   * Initialize agent environment in PowerShell
   */
  async initializeEnvironment() {
    const config = {
      AgentId: this.generateAgentId(),
      Name: 'AIAgent',
      Version: '1.0.0',
      LogPath: this.config.logPath.replace(/\\/g, '\\'),
      ScriptPath: path.join(this.config.agentPath, 'Scripts').replace(/\\/g, '\\'),
      ConfigPath: path.join(this.config.agentPath, 'Config').replace(/\\/g, '\\'),
      DataPath: path.join(this.config.agentPath, 'Data').replace(/\\/g, '\\')
    };

    const psConfig = Object.entries(config)
      .map(([k, v]) => `${k} = '${v}'`)
      .join('; ');

    await this.session.execute(`
      Initialize-AgentEnvironment -Configuration @{
        ${psConfig}
      }
    `);
  }

  /**
   * Start agent heartbeat
   */
  async startHeartbeat() {
    await this.session.execute(`
      Start-AgentHeartbeat -IntervalSeconds ${this.config.heartbeatInterval}
    `);
    this.emit('heartbeatStarted', { interval: this.config.heartbeatInterval });
  }

  /**
   * Execute a PowerShell command and return structured output
   */
  async executeCommand(command, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Agent PowerShell integration not initialized');
    }

    try {
      const result = await this.session.execute(command, options.expectJson);
      this.emit('commandSuccess', { command, result });
      return result;
    } catch (error) {
      this.emit('commandError', { command, error });
      throw error;
    }
  }

  // ==================== SYSTEM INFORMATION ====================

  /**
   * Get comprehensive system status
   */
  async getSystemStatus() {
    return this.executeCommand('Get-AgentSystemStatus | ConvertTo-Json -Depth 5', { expectJson: true });
  }

  /**
   * Get agent identity information
   */
  async getIdentity(includeSystemInfo = false) {
    const param = includeSystemInfo ? '-IncludeSystemInfo' : '';
    return this.executeCommand(`Get-AgentIdentity ${param} | ConvertTo-Json -Depth 3`, { expectJson: true });
  }

  // ==================== PROCESS MANAGEMENT ====================

  /**
   * Get list of running processes
   */
  async getProcesses(filter = {}) {
    const filterString = Object.entries(filter)
      .map(([k, v]) => `${k} = '${v}'`)
      .join(' and ');
    
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetProcesses(@{${filterString}}) | 
        Select-Object Name, ProcessId, WorkingSetSize, ThreadCount | 
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Get detailed process information
   */
  async getProcessDetails(processId) {
    return this.executeCommand(`
      Get-Process -Id ${processId} | 
        Select-Object Id, Name, Path, WorkingSet, CPU, Threads, Handles, StartTime |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Start a process
   */
  async startProcess(filePath, args = '', options = {}) {
    const waitParam = options.wait ? '-Wait' : '';
    const hiddenParam = options.hidden ? '-WindowStyle Hidden' : '';
    
    return this.executeCommand(`
      Start-Process -FilePath "${filePath}" -ArgumentList "${args}" ${waitParam} ${hiddenParam} -Pass |
        Select-Object Id, Name, Path |
        ConvertTo-Json
    `, { expectJson: true });
  }

  /**
   * Stop a process
   */
  async stopProcess(processId, force = false) {
    const forceParam = force ? '-Force' : '';
    return this.executeCommand(`
      Stop-Process -Id ${processId} ${forceParam} -PassThru |
        Select-Object Id, Name, HasExited |
        ConvertTo-Json
    `, { expectJson: true });
  }

  // ==================== SERVICE MANAGEMENT ====================

  /**
   * Get services list
   */
  async getServices(state = null, startMode = null) {
    const params = [];
    if (state) params.push(`-State '${state}'`);
    if (startMode) params.push(`-StartMode '${startMode}'`);
    
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $wmi.GetServices(${params.join(' ')}) |
        Select-Object Name, DisplayName, State, StartMode, ProcessId |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Start a Windows service
   */
  async startService(serviceName) {
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $result = $wmi.StartService('${serviceName}')
      @{ Success = $result; ServiceName = '${serviceName}' } | ConvertTo-Json
    `, { expectJson: true });
  }

  /**
   * Stop a Windows service
   */
  async stopService(serviceName) {
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $result = $wmi.StopService('${serviceName}')
      @{ Success = $result; ServiceName = '${serviceName}' } | ConvertTo-Json
    `, { expectJson: true });
  }

  /**
   * Set service start mode
   */
  async setServiceStartMode(serviceName, startMode) {
    return this.executeCommand(`
      $wmi = New-WMIProvider
      $result = $wmi.SetServiceStartMode('${serviceName}', '${startMode}')
      @{ Success = $result; ServiceName = '${serviceName}'; StartMode = '${startMode}' } | ConvertTo-Json
    `, { expectJson: true });
  }

  // ==================== FILE OPERATIONS ====================

  /**
   * Perform file operations
   */
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
      Invoke-AgentFileOperation ${params.join(' ')} |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Read file content
   */
  async readFile(filePath, encoding = 'utf8') {
    return this.executeCommand(`
      Get-Content -Path "${filePath}" -Encoding ${encoding} -Raw
    `);
  }

  /**
   * Write file content
   */
  async writeFile(filePath, content, encoding = 'utf8') {
    const escapedContent = content.replace(/'/g, "''");
    return this.executeCommand(`
      @'
${escapedContent}
'@ | Set-Content -Path "${filePath}" -Encoding ${encoding}
      @{ Success = $true; Path = "${filePath}" } | ConvertTo-Json
    `, { expectJson: true });
  }

  // ==================== REGISTRY OPERATIONS ====================

  /**
   * Get registry value
   */
  async getRegistryValue(path, name) {
    return this.executeCommand(`
      Get-ItemProperty -Path "${path}" -Name "${name}" -ErrorAction Stop |
        Select-Object -ExpandProperty ${name} |
        ConvertTo-Json
    `, { expectJson: true });
  }

  /**
   * Set registry value
   */
  async setRegistryValue(path, name, value, type = 'String') {
    const valueParam = typeof value === 'number' ? value : `"${value}"`;
    return this.executeCommand(`
      if (-not (Test-Path "${path}")) { New-Item -Path "${path}" -Force | Out-Null }
      Set-ItemProperty -Path "${path}" -Name "${name}" -Value ${valueParam} -Type ${type}
      @{ Success = $true; Path = "${path}"; Name = "${name}" } | ConvertTo-Json
    `, { expectJson: true });
  }

  // ==================== NETWORK OPERATIONS ====================

  /**
   * Test network connection
   */
  async testNetworkConnection(hostname, port = null) {
    const portParam = port ? `-Port ${port}` : '';
    return this.executeCommand(`
      Test-AgentNetworkConnection -ComputerName "${hostname}" ${portParam} |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Get network configuration
   */
  async getNetworkConfiguration() {
    return this.executeCommand(`
      Get-AgentNetworkConfiguration |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  // ==================== TASK SCHEDULING ====================

  /**
   * Create a scheduled task
   */
  async createScheduledTask(name, scriptBlock, options = {}) {
    const psScriptBlock = scriptBlock.replace(/'/g, "''");
    const trigger = options.trigger || 'Once';
    
    return this.executeCommand(`
      New-AgentScheduledTask -Name '${name}' -Action {
        ${psScriptBlock}
      } -Trigger '${trigger}' |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  /**
   * Get scheduled tasks
   */
  async getScheduledTasks(name = null) {
    const nameParam = name ? `-Name '${name}'` : '';
    return this.executeCommand(`
      Get-AgentScheduledTask ${nameParam} |
        ConvertTo-Json -Depth 3
    `, { expectJson: true });
  }

  // ==================== LOGGING ====================

  /**
   * Get agent logs
   */
  async getLogs(options = {}) {
    const count = options.count || 100;
    const level = options.level ? `-Level '${options.level}'` : '';
    const category = options.category ? `-Category '${options.category}'` : '';
    
    return this.executeCommand(`
      Get-AgentLog -Count ${count} ${level} ${category} |
        ConvertTo-Json -Depth 5
    `, { expectJson: true });
  }

  /**
   * Write to agent log
   */
  async writeLog(message, level = 'Info', category = 'General') {
    const escapedMessage = message.replace(/'/g, "''");
    return this.executeCommand(`
      Write-AgentLog -Message '${escapedMessage}' -Level '${level}' -Category '${category}'
      @{ Success = $true } | ConvertTo-Json
    `, { expectJson: true });
  }

  // ==================== UTILITY ====================

  /**
   * Generate unique agent ID
   */
  generateAgentId() {
    return `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Dispose and cleanup
   */
  async dispose() {
    try {
      if (this.session) {
        await this.session.execute('Stop-AgentHeartbeat');
        await this.session.dispose();
      }
      this.isInitialized = false;
      this.emit('disposed');
    } catch (error) {
      this.emit('disposeError', error);
    }
  }
}

module.exports = { AgentPowerShellIntegration };
