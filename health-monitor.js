/**
 * OpenClawAgent - Health Monitor
 * Monitors system health and worker status
 */

const EventEmitter = require('events');
const os = require('os');
const cluster = require('cluster');
const logger = require('./logger');

class HealthMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      checkInterval: options.checkInterval || 30000,
      memoryThreshold: options.memoryThreshold || 0.9,
      cpuThreshold: options.cpuThreshold || 0.9,
      diskThreshold: options.diskThreshold || 0.9,
      workerTimeout: options.workerTimeout || 60000,
      ...options
    };
    
    this.checkInterval = null;
    this.metrics = {
      startTime: Date.now(),
      checks: 0,
      failures: 0,
      lastCheck: null,
      history: []
    };
    this.thresholdBreaches = new Map();
  }

  start() {
    logger.info('[HealthMonitor] Starting health monitoring...');
    
    this.checkInterval = setInterval(() => {
      this.performHealthCheck();
    }, this.options.checkInterval);
    
    // Initial check
    this.performHealthCheck();
    
    logger.info(`[HealthMonitor] Monitoring started (interval: ${this.options.checkInterval}ms)`);
  }

  stop() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
      logger.info('[HealthMonitor] Monitoring stopped');
    }
  }

  async performHealthCheck() {
    this.metrics.checks++;
    const checkStart = Date.now();
    
    try {
      const checks = await Promise.all([
        this.checkMemory(),
        this.checkCPU(),
        this.checkDisk(),
        this.checkProcesses(),
        this.checkPythonBridge()
      ]);

      const health = {
        timestamp: Date.now(),
        status: checks.every(c => c.healthy) ? 'healthy' : 'degraded',
        checks: {
          memory: checks[0],
          cpu: checks[1],
          disk: checks[2],
          processes: checks[3],
          pythonBridge: checks[4]
        },
        uptime: process.uptime(),
        systemUptime: os.uptime(),
        checkDuration: Date.now() - checkStart
      };
      
      this.metrics.lastCheck = health.timestamp;
      
      // Store history (keep last 100 checks)
      this.metrics.history.push(health);
      if (this.metrics.history.length > 100) {
        this.metrics.history.shift();
      }
      
      // Handle degraded status
      if (health.status !== 'healthy') {
        this.metrics.failures++;
        this.handleDegradedHealth(health);
        this.emit('degraded', health);
      } else {
        // Clear threshold breaches on healthy status
        this.thresholdBreaches.clear();
      }
      
      this.emit('health', health);
      
      return health;
    } catch (error) {
      logger.error('[HealthMonitor] Health check failed:', error);
      this.emit('error', error);
      return null;
    }
  }

  async checkMemory() {
    const total = os.totalmem();
    const free = os.freemem();
    const used = total - free;
    const usage = used / total;
    
    const processUsage = process.memoryUsage();
    const processHeapUsed = processUsage.heapUsed / processUsage.heapTotal;
    
    const healthy = usage < this.options.memoryThreshold && 
                    processHeapUsed < this.options.memoryThreshold;
    
    // Track threshold breach
    if (!healthy) {
      this.trackThresholdBreach('memory', usage);
    }
    
    return {
      healthy,
      system: {
        total: this.formatBytes(total),
        free: this.formatBytes(free),
        used: this.formatBytes(used),
        usage: (usage * 100).toFixed(2) + '%'
      },
      process: {
        heapTotal: this.formatBytes(processUsage.heapTotal),
        heapUsed: this.formatBytes(processUsage.heapUsed),
        external: this.formatBytes(processUsage.external),
        rss: this.formatBytes(processUsage.rss),
        usage: (processHeapUsed * 100).toFixed(2) + '%'
      }
    };
  }

  async checkCPU() {
    const cpus = os.cpus();

    // Calculate CPU usage from cpu times (works on all platforms)
    let totalIdle = 0;
    let totalTick = 0;

    for (const cpu of cpus) {
      for (const type in cpu.times) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    }

    const cpuUsage = 1 - (totalIdle / totalTick);
    const loadPercent = cpuUsage * 100;

    const healthy = loadPercent < (this.options.cpuThreshold * 100);

    if (!healthy) {
      this.trackThresholdBreach('cpu', loadPercent);
    }

    return {
      healthy,
      count: cpus.length,
      model: cpus[0]?.model,
      usage: loadPercent.toFixed(2) + '%',
      loadPercent: loadPercent.toFixed(2) + '%'
    };
  }

  async checkDisk() {
    try {
      if (process.platform === 'win32') {
        // Use PowerShell to get disk stats on Windows
        const { exec } = require('child_process');
        const { promisify } = require('util');
        const execAsync = promisify(exec);

        const { stdout } = await execAsync(
          'powershell -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | ConvertTo-Json"',
          { timeout: 10000 }
        );
        const drives = JSON.parse(stdout);
        const driveList = Array.isArray(drives) ? drives : [drives];

        const driveInfo = driveList.map(d => {
          const total = (d.Used || 0) + (d.Free || 0);
          const usage = total > 0 ? d.Used / total : 0;
          return {
            name: d.Name + ':',
            total: this.formatBytes(total),
            free: this.formatBytes(d.Free || 0),
            used: this.formatBytes(d.Used || 0),
            usage: (usage * 100).toFixed(2) + '%',
          };
        });

        const healthy = driveList.every(d => {
          const total = (d.Used || 0) + (d.Free || 0);
          return total === 0 || (d.Used / total) < this.options.diskThreshold;
        });

        return { healthy, drives: driveInfo };
      }

      // Non-Windows fallback
      return { healthy: true, drives: [], message: 'Non-Windows platform' };
    } catch (error) {
      logger.warn('[HealthMonitor] Disk check failed:', error.message);
      return { healthy: true, error: error.message };
    }
  }

  async checkProcesses() {
    // Query real cluster worker counts if running as master
    if (cluster.isMaster || cluster.isPrimary) {
      const workers = Object.values(cluster.workers || {});
      const alive = workers.filter(w => !w.isDead());
      return {
        healthy: alive.length > 0,
        workers: {
          total: alive.length,
          dead: workers.length - alive.length,
          pids: alive.map(w => w.process?.pid).filter(Boolean),
        }
      };
    }

    // Worker process: report own status
    return {
      healthy: true,
      workers: {
        total: parseInt(process.env.WORKER_COUNT) || 0,
        pid: process.pid,
      }
    };
  }

  async checkPythonBridge() {
    // Check Python bridge health (master only)
    try {
      const { getBridge } = require('./python-bridge');
      const bridge = getBridge();
      if (bridge && bridge.healthCheck) {
        const result = await bridge.healthCheck();
        return {
          healthy: result.healthy !== false,
          latencyMs: result.latencyMs,
          llmInitialized: result.llm_initialized,
          memoryInitialized: result.memory_initialized,
          pid: result.pid,
        };
      }
      return { healthy: true, note: 'Bridge not initialized yet' };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  }

  trackThresholdBreach(type, value) {
    const current = this.thresholdBreaches.get(type) || { count: 0, firstBreach: Date.now() };
    current.count++;
    current.lastBreach = Date.now();
    current.value = value;
    this.thresholdBreaches.set(type, current);
    
    // Log warning if threshold breached multiple times
    if (current.count === 3) {
      logger.warn(`[HealthMonitor] ${type} threshold breached ${current.count} times. Value: ${value.toFixed ? value.toFixed(2) : value}`);
    }
  }

  handleDegradedHealth(health) {
    // Log detailed information about degraded health
    const degradedChecks = Object.entries(health.checks)
      .filter(([_, check]) => !check.healthy)
      .map(([name, _]) => name);
    
    logger.warn(`[HealthMonitor] System health degraded. Failed checks: ${degradedChecks.join(', ')}`);
    
    // Report to master if in worker process
    if (process.send) {
      process.send({
        type: 'health-degraded',
        data: {
          timestamp: health.timestamp,
          failedChecks: degradedChecks,
          details: health.checks
        }
      });
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  getMetrics() {
    return {
      ...this.metrics,
      uptime: Date.now() - this.metrics.startTime,
      thresholdBreaches: Object.fromEntries(this.thresholdBreaches)
    };
  }

  getHealthHistory(limit = 10) {
    return this.metrics.history.slice(-limit);
  }

  isHealthy() {
    if (!this.metrics.lastCheck) return false;
    const lastHealth = this.metrics.history[this.metrics.history.length - 1];
    return lastHealth ? lastHealth.status === 'healthy' : false;
  }
}

module.exports = HealthMonitor;
