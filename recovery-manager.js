/**
 * OpenClawAgent - Recovery Manager
 * Manages Windows service recovery options and failure handling
 */

const { exec } = require('child_process');
const util = require('util');
const execAsync = util.promisify(exec);
const logger = require('./logger');

class RecoveryManager {
  constructor(serviceName = 'OpenClawAgent') {
    this.serviceName = serviceName;
    this.defaultConfig = {
      resetPeriod: 3600,  // Reset failure count after 1 hour
      firstFailure: { action: 'restart', delay: 5000 },
      secondFailure: { action: 'restart', delay: 10000 },
      subsequentFailures: { action: 'restart', delay: 60000 },
      enableActionsOnStop: true,
      runProgram: null,
      rebootMessage: null
    };
  }

  async configure(config = {}) {
    const cfg = { ...this.defaultConfig, ...config };
    
    logger.info(`[Recovery] Configuring recovery for ${this.serviceName}`);
    
    try {
      // Build failure command
      const actions = [
        `${cfg.firstFailure.action}/${cfg.firstFailure.delay}`,
        `${cfg.secondFailure.action}/${cfg.secondFailure.delay}`,
        `${cfg.subsequentFailures.action}/${cfg.subsequentFailures.delay}`
      ].join('/');
      
      const command = `sc failure "${this.serviceName}" reset= ${cfg.resetPeriod} actions= ${actions}`;
      
      const { stdout, stderr } = await execAsync(command);
      
      if (stderr && !stderr.includes('SUCCESS')) {
        throw new Error(stderr);
      }
      
      // Enable actions for stops with errors
      if (cfg.enableActionsOnStop) {
        await execAsync(`sc failureflag "${this.serviceName}" 1`);
      }
      
      // Configure program to run on failure
      if (cfg.runProgram) {
        await execAsync(`sc failure "${this.serviceName}" command= "${cfg.runProgram}"`);
      }
      
      // Configure reboot message
      if (cfg.rebootMessage) {
        await execAsync(`sc failure "${this.serviceName}" reboot= "${cfg.rebootMessage}"`);
      }
      
      logger.info('[Recovery] Recovery options configured successfully');
      return { success: true, config: cfg };
    } catch (error) {
      logger.error('[Recovery] Configuration failed:', error);
      return { success: false, error: error.message };
    }
  }

  async getFailureConfig() {
    try {
      const { stdout } = await execAsync(`sc qfailure "${this.serviceName}"`);
      return this.parseFailureConfig(stdout);
    } catch (error) {
      logger.error('[Recovery] Failed to get failure config:', error);
      return null;
    }
  }

  parseFailureConfig(output) {
    const config = {};
    
    // Parse reset period
    const resetMatch = output.match(/RESET PERIOD \(in seconds\)\s*:\s*(\d+)/);
    if (resetMatch) config.resetPeriod = parseInt(resetMatch[1]);
    
    // Parse failure actions
    const actionsMatch = output.match(/FAILURE ACTIONS\s*:\s*(.+)/);
    if (actionsMatch) {
      config.actions = actionsMatch[1].trim();
    }
    
    // Parse command
    const commandMatch = output.match(/COMMAND\s*:\s*(.+)/);
    if (commandMatch) {
      config.command = commandMatch[1].trim();
    }
    
    // Parse reboot message
    const rebootMatch = output.match(/REBOOT MESSAGE\s*:\s*(.+)/);
    if (rebootMatch) {
      config.rebootMessage = rebootMatch[1].trim();
    }
    
    return config;
  }

  async configureAdvanced(options) {
    const results = [];
    
    // Configure failure actions flag
    if (options.enableActionsOnStop !== undefined) {
      try {
        const flag = options.enableActionsOnStop ? '1' : '0';
        await execAsync(`sc failureflag "${this.serviceName}" ${flag}`);
        results.push({ action: 'failureflag', success: true });
      } catch (error) {
        results.push({ action: 'failureflag', success: false, error: error.message });
      }
    }
    
    // Configure program to run
    if (options.runProgram) {
      try {
        await execAsync(`sc failure "${this.serviceName}" command= "${options.runProgram}"`);
        results.push({ action: 'command', success: true });
      } catch (error) {
        results.push({ action: 'command', success: false, error: error.message });
      }
    }
    
    // Configure reboot message
    if (options.rebootMessage) {
      try {
        await execAsync(`sc failure "${this.serviceName}" reboot= "${options.rebootMessage}"`);
        results.push({ action: 'reboot', success: true });
      } catch (error) {
        results.push({ action: 'reboot', success: false, error: error.message });
      }
    }
    
    return results;
  }

  async resetFailureCount() {
    try {
      // Reset failure count by reconfiguring with same settings
      const currentConfig = await this.getFailureConfig();
      if (currentConfig) {
        await this.configure(currentConfig);
        return { success: true };
      }
      return { success: false, error: 'Could not get current config' };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Preset configurations
  async applyPreset(presetName) {
    const presets = {
      'aggressive': {
        resetPeriod: 1800, // 30 minutes
        firstFailure: { action: 'restart', delay: 1000 },
        secondFailure: { action: 'restart', delay: 5000 },
        subsequentFailures: { action: 'restart', delay: 10000 }
      },
      'conservative': {
        resetPeriod: 7200, // 2 hours
        firstFailure: { action: 'restart', delay: 30000 },
        secondFailure: { action: 'restart', delay: 60000 },
        subsequentFailures: { action: 'restart', delay: 300000 }
      },
      'minimal': {
        resetPeriod: 3600,
        firstFailure: { action: 'restart', delay: 5000 },
        secondFailure: { action: 'restart', delay: 10000 },
        subsequentFailures: { action: 'restart', delay: 60000 }
      },
      'critical': {
        resetPeriod: 600, // 10 minutes
        firstFailure: { action: 'restart', delay: 1000 },
        secondFailure: { action: 'restart', delay: 2000 },
        subsequentFailures: { action: 'reboot', delay: 5000 },
        rebootMessage: 'OpenClawAgent service failure - system rebooting'
      }
    };
    
    const preset = presets[presetName];
    if (!preset) {
      return { success: false, error: `Unknown preset: ${presetName}` };
    }
    
    return await this.configure(preset);
  }

  // Recovery action reference
  getAvailableActions() {
    return [
      { action: 'restart', description: 'Restart the service' },
      { action: 'reboot', description: 'Restart the computer' },
      { action: 'run', description: 'Run a program' },
      { action: 'none', description: 'Take no action' }
    ];
  }
}

module.exports = RecoveryManager;
