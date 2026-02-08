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

// Persistent Session Manager
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
    this.rl = require('readline').createInterface({
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

  getPowerShellPath() {
    const ps7Path = 'C:\\Program Files\\PowerShell\\7\\pwsh.exe';
    const ps5Path = 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe';
    
    try {
      require('fs').accessSync(ps7Path);
      return ps7Path;
    } catch {
      return ps5Path;
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

module.exports = { 
  PowerShellBridge, 
  PowerShellSession,
  PowerShellExecutionError, 
  PowerShellTimeoutError 
};
