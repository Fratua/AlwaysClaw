/**
 * OpenClawAgent - Lifecycle Manager
 * Manages Windows service lifecycle (install, start, stop, uninstall)
 */

const { exec } = require('child_process');
const util = require('util');
const execAsync = util.promisify(exec);
const logger = require('./logger');

class LifecycleManager {
  constructor(serviceName = 'OpenClawAgent') {
    this.serviceName = serviceName;
  }

  async install(options = {}) {
    logger.info(`[Lifecycle] Installing service: ${this.serviceName}`);
    
    try {
      // Check if service already exists
      const exists = await this.serviceExists();
      if (exists) {
        logger.warn(`[Lifecycle] Service ${this.serviceName} already exists`);
        return { success: false, message: 'Service already exists' };
      }

      // Install service using node-windows or sc
      if (options.useNodeWindows) {
        await this.installWithNodeWindows(options);
      } else {
        await this.installWithSC(options);
      }
      
      // Configure recovery options
      await this.configureRecovery(options.recovery);
      
      // Set dependencies
      if (options.dependencies) {
        await this.setDependencies(options.dependencies);
      }
      
      // Set service description
      if (options.description) {
        await this.setDescription(options.description);
      }

      logger.info(`[Lifecycle] Service installed: ${this.serviceName}`);
      return { success: true, message: 'Service installed successfully' };
    } catch (error) {
      logger.error(`[Lifecycle] Install failed:`, error);
      return { success: false, message: error.message };
    }
  }

  async installWithSC(options) {
    const binPath = options.binPath || process.execPath;
    const scriptPath = options.scriptPath || require('path').join(__dirname, 'service.js');
    const fullPath = `"${binPath}" "${scriptPath}"`;
    
    const command = `sc create "${this.serviceName}" binPath= ${fullPath} start= ${options.startType || 'auto'} DisplayName= "${options.displayName || this.serviceName}"`;
    
    const { stdout, stderr } = await execAsync(command);
    
    if (stderr && !stderr.includes('SUCCESS')) {
      throw new Error(stderr);
    }
    
    return stdout;
  }

  async installWithNodeWindows(options) {
    const Service = require('node-windows').Service;
    
    const svc = new Service({
      name: this.serviceName,
      description: options.description || 'OpenClaw AI Agent Service',
      script: options.scriptPath || require('path').join(__dirname, 'service.js'),
      nodeOptions: options.nodeOptions || ['--max-old-space-size=4096'],
      workingDirectory: options.workingDirectory || __dirname
    });

    return new Promise((resolve, reject) => {
      svc.on('install', () => resolve());
      svc.on('error', reject);
      svc.install();
    });
  }

  async uninstall() {
    logger.info(`[Lifecycle] Uninstalling service: ${this.serviceName}`);
    
    try {
      // Stop service first
      await this.stop();
      
      // Wait for service to stop
      await this.sleep(2000);
      
      // Remove service
      const { stdout, stderr } = await execAsync(`sc delete "${this.serviceName}"`);
      
      if (stderr && !stderr.includes('SUCCESS') && !stderr.includes('does not exist')) {
        throw new Error(stderr);
      }
      
      logger.info(`[Lifecycle] Service uninstalled: ${this.serviceName}`);
      return { success: true, message: 'Service uninstalled successfully' };
    } catch (error) {
      logger.error(`[Lifecycle] Uninstall failed:`, error);
      return { success: false, message: error.message };
    }
  }

  async start() {
    logger.info(`[Lifecycle] Starting service: ${this.serviceName}`);
    
    try {
      const { stdout, stderr } = await execAsync(`sc start "${this.serviceName}"`);
      
      if (stderr && !stderr.includes('SUCCESS') && !stderr.includes('START_PENDING')) {
        throw new Error(stderr);
      }
      
      logger.info(`[Lifecycle] Service started: ${this.serviceName}`);
      return { success: true, message: 'Service started successfully' };
    } catch (error) {
      logger.error(`[Lifecycle] Start failed:`, error);
      return { success: false, message: error.message };
    }
  }

  async stop() {
    logger.info(`[Lifecycle] Stopping service: ${this.serviceName}`);
    
    try {
      const { stdout, stderr } = await execAsync(`sc stop "${this.serviceName}"`);
      
      if (stderr && !stderr.includes('SUCCESS') && 
          !stderr.includes('STOP_PENDING') && 
          !stderr.includes('is not running')) {
        throw new Error(stderr);
      }
      
      logger.info(`[Lifecycle] Service stopped: ${this.serviceName}`);
      return { success: true, message: 'Service stopped successfully' };
    } catch (error) {
      logger.error(`[Lifecycle] Stop failed:`, error);
      return { success: false, message: error.message };
    }
  }

  async restart() {
    logger.info(`[Lifecycle] Restarting service: ${this.serviceName}`);
    
    const stopResult = await this.stop();
    if (!stopResult.success) {
      return stopResult;
    }
    
    await this.sleep(3000);
    
    return await this.start();
  }

  async pause() {
    logger.info(`[Lifecycle] Pausing service: ${this.serviceName}`);
    
    try {
      const { stdout, stderr } = await execAsync(`sc pause "${this.serviceName}"`);
      
      if (stderr && !stderr.includes('SUCCESS')) {
        throw new Error(stderr);
      }
      
      return { success: true, message: 'Service paused' };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }

  async resume() {
    logger.info(`[Lifecycle] Resuming service: ${this.serviceName}`);
    
    try {
      const { stdout, stderr } = await execAsync(`sc continue "${this.serviceName}"`);
      
      if (stderr && !stderr.includes('SUCCESS')) {
        throw new Error(stderr);
      }
      
      return { success: true, message: 'Service resumed' };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }

  async status() {
    try {
      const { stdout } = await execAsync(`sc query "${this.serviceName}"`);
      return this.parseStatus(stdout);
    } catch (error) {
      return { state: 'NOT_INSTALLED', exists: false };
    }
  }

  async serviceExists() {
    const status = await this.status();
    return status.exists;
  }

  async configureRecovery(config = {}) {
    const recoveryConfig = {
      resetPeriod: config.resetPeriod || 3600,
      firstFailure: config.firstFailure || { action: 'restart', delay: 5000 },
      secondFailure: config.secondFailure || { action: 'restart', delay: 10000 },
      subsequentFailures: config.subsequentFailures || { action: 'restart', delay: 60000 }
    };

    // Build failure command
    const actions = [
      `${recoveryConfig.firstFailure.action}/${recoveryConfig.firstFailure.delay}`,
      `${recoveryConfig.secondFailure.action}/${recoveryConfig.secondFailure.delay}`,
      `${recoveryConfig.subsequentFailures.action}/${recoveryConfig.subsequentFailures.delay}`
    ].join('/');

    const command = `sc failure "${this.serviceName}" reset= ${recoveryConfig.resetPeriod} actions= ${actions}`;
    
    try {
      await execAsync(command);
      
      // Enable actions for stops with errors
      await execAsync(`sc failureflag "${this.serviceName}" 1`);
      
      logger.info(`[Lifecycle] Recovery options configured`);
      return { success: true };
    } catch (error) {
      logger.error(`[Lifecycle] Recovery configuration failed:`, error);
      return { success: false, error: error.message };
    }
  }

  async setDependencies(dependencies) {
    const depString = dependencies.join('/');
    
    try {
      await execAsync(`sc config "${this.serviceName}" depend= ${depString}`);
      logger.info(`[Lifecycle] Dependencies set: ${depString}`);
      return { success: true };
    } catch (error) {
      logger.error(`[Lifecycle] Failed to set dependencies:`, error);
      return { success: false, error: error.message };
    }
  }

  async setDescription(description) {
    try {
      await execAsync(`sc description "${this.serviceName}" "${description}"`);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  parseStatus(output) {
    const stateMatch = output.match(/STATE\s+:\s+(\d+)\s+(\w+)/);
    const exitCodeMatch = output.match(/EXIT_CODE\s+:\s+(\d+)/);
    
    const states = {
      '1': 'STOPPED',
      '2': 'START_PENDING',
      '3': 'STOP_PENDING',
      '4': 'RUNNING',
      '5': 'CONTINUE_PENDING',
      '6': 'PAUSE_PENDING',
      '7': 'PAUSED'
    };

    return {
      exists: true,
      state: stateMatch ? states[stateMatch[1]] || stateMatch[2] : 'UNKNOWN',
      rawState: stateMatch ? stateMatch[1] : null,
      exitCode: exitCodeMatch ? parseInt(exitCodeMatch[1]) : null,
      raw: output
    };
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// CLI interface
if (require.main === module) {
  const manager = new LifecycleManager();
  const command = process.argv[2];

  (async () => {
    switch(command) {
      case 'install':
        const result = await manager.install({
          description: 'OpenClaw AI Agent System - 24/7 Background Service',
          dependencies: ['RpcSs', 'Dhcp', 'Dnscache']
        });
        console.log(result.message);
        if (result.success) {
          await manager.start();
        }
        break;
      case 'uninstall':
        const unResult = await manager.uninstall();
        console.log(unResult.message);
        break;
      case 'start':
        const startResult = await manager.start();
        console.log(startResult.message);
        break;
      case 'stop':
        const stopResult = await manager.stop();
        console.log(stopResult.message);
        break;
      case 'restart':
        const restartResult = await manager.restart();
        console.log(restartResult.message);
        break;
      case 'status':
        const status = await manager.status();
        console.log('Service Status:', status.state);
        if (status.exitCode !== null) {
          console.log('Exit Code:', status.exitCode);
        }
        break;
      default:
        console.log(`
Usage: node lifecycle-manager.js [command]

Commands:
  install     Install the service
  uninstall   Uninstall the service
  start       Start the service
  stop        Stop the service
  restart     Restart the service
  status      Check service status
        `);
    }
    
    process.exit(0);
  })();
}

module.exports = LifecycleManager;
