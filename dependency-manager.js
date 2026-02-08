/**
 * OpenClawAgent - Dependency Manager
 * Manages Windows service dependencies
 */

const { exec } = require('child_process');
const util = require('util');
const execAsync = util.promisify(exec);
const logger = require('./logger');

class DependencyManager {
  constructor(serviceName = 'OpenClawAgent') {
    this.serviceName = serviceName;
    
    // Default system dependencies
    this.defaultDependencies = {
      system: [
        'RpcSs',        // Remote Procedure Call
        'Dhcp',         // DHCP Client
        'Dnscache',     // DNS Client
        'NlaSvc',       // Network Location Awareness
        'netprofm',     // Network List Service
        'nsi'           // Network Store Interface
      ],
      optional: [
        'WinHttpAutoProxySvc',
        'CryptSvc'
      ],
      external: {
        gmail: ['network'],
        twilio: ['network'],
        browser: ['graphics']
      }
    };
  }

  async checkDependencies(deps) {
    logger.info(`[Dependencies] Checking ${deps.length} dependencies...`);
    
    const results = [];
    
    for (const dep of deps) {
      try {
        const { stdout } = await execAsync(`sc query "${dep}"`);
        const isRunning = stdout.includes('RUNNING');
        const state = this.parseServiceState(stdout);
        
        results.push({ 
          name: dep, 
          available: true, 
          running: isRunning,
          state: state,
          healthy: isRunning
        });
        
        logger.debug(`[Dependencies] ${dep}: ${state}`);
      } catch (error) {
        results.push({ 
          name: dep, 
          available: false, 
          running: false,
          state: 'NOT_FOUND',
          healthy: false
        });
        
        logger.warn(`[Dependencies] ${dep}: Not found`);
      }
    }
    
    return results;
  }

  async waitForDependencies(deps, timeout = 60000) {
    logger.info(`[Dependencies] Waiting for dependencies (timeout: ${timeout}ms)...`);
    
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const results = await this.checkDependencies(deps);
      const allReady = results.every(r => r.running);
      
      if (allReady) {
        logger.info('[Dependencies] All dependencies ready');
        return { success: true, results };
      }
      
      const notReady = results.filter(r => !r.running).map(r => r.name);
      logger.debug(`[Dependencies] Waiting for: ${notReady.join(', ')}`);
      
      await this.sleep(1000);
    }
    
    logger.error('[Dependencies] Timeout waiting for dependencies');
    return { success: false, results: await this.checkDependencies(deps) };
  }

  async setServiceDependencies(deps) {
    const depString = deps.join('/');
    
    try {
      logger.info(`[Dependencies] Setting dependencies: ${depString}`);
      await execAsync(`sc config "${this.serviceName}" depend= ${depString}`);
      logger.info('[Dependencies] Dependencies set successfully');
      return { success: true, dependencies: deps };
    } catch (error) {
      logger.error('[Dependencies] Failed to set dependencies:', error);
      return { success: false, error: error.message };
    }
  }

  async addDependency(dep) {
    try {
      // Get current dependencies
      const current = await this.getCurrentDependencies();
      
      if (current.includes(dep)) {
        return { success: true, message: 'Dependency already exists' };
      }
      
      // Add new dependency
      const newDeps = [...current, dep];
      return await this.setServiceDependencies(newDeps);
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async removeDependency(dep) {
    try {
      // Get current dependencies
      const current = await this.getCurrentDependencies();
      
      if (!current.includes(dep)) {
        return { success: true, message: 'Dependency not found' };
      }
      
      // Remove dependency
      const newDeps = current.filter(d => d !== dep);
      return await this.setServiceDependencies(newDeps);
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getCurrentDependencies() {
    try {
      const { stdout } = await execAsync(`sc qc "${this.serviceName}"`);
      return this.parseDependencies(stdout);
    } catch (error) {
      logger.error('[Dependencies] Failed to get dependencies:', error);
      return [];
    }
  }

  parseDependencies(output) {
    const depMatch = output.match(/DEPENDENCIES\s*:\s*(.+)/);
    if (depMatch && depMatch[1].trim()) {
      return depMatch[1].trim().split(/\s+/);
    }
    return [];
  }

  parseServiceState(output) {
    const stateMatch = output.match(/STATE\s+:\s+\d+\s+(\w+)/);
    return stateMatch ? stateMatch[1] : 'UNKNOWN';
  }

  async getDependencyTree() {
    const tree = {
      service: this.serviceName,
      dependencies: [],
      dependents: []
    };
    
    // Get direct dependencies
    const directDeps = await this.getCurrentDependencies();
    
    for (const dep of directDeps) {
      const depInfo = await this.getServiceInfo(dep);
      tree.dependencies.push(depInfo);
    }
    
    // Find services that depend on this service
    // This requires enumerating all services
    
    return tree;
  }

  async getServiceInfo(serviceName) {
    try {
      const { stdout } = await execAsync(`sc qc "${serviceName}"`);
      const { stdout: statusStdout } = await execAsync(`sc query "${serviceName}"`);
      
      return {
        name: serviceName,
        state: this.parseServiceState(statusStdout),
        dependencies: this.parseDependencies(stdout),
        raw: { config: stdout, status: statusStdout }
      };
    } catch (error) {
      return {
        name: serviceName,
        state: 'NOT_FOUND',
        error: error.message
      };
    }
  }

  getDefaultDependencies() {
    return this.defaultDependencies;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = DependencyManager;
