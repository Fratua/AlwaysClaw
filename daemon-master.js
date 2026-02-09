/**
 * OpenClawAgent - Daemon Master Process
 * Manages worker processes, clustering, and service lifecycle
 */

const cluster = require('cluster');
const os = require('os');
const fs = require('fs');
const path = require('path');
const EventEmitter = require('events');
const logger = require('./logger');
const ControlServer = require('./control-server');
const MessageBus = require('./message-bus');
const { getBridge } = require('./python-bridge');

// Cognitive loop schedules extracted from _registerFallbackLoops for maintainability
const COGNITIVE_LOOP_SCHEDULES = [
  { name: 'ralph', method: 'loop.ralph.run_cycle', cron: '*/5 * * * *' },
  { name: 'research', method: 'loop.research.run_cycle', cron: '*/10 * * * *' },
  { name: 'planning', method: 'loop.planning.run_cycle', cron: '*/10 * * * *' },
  { name: 'e2e', method: 'loop.e2e.run_cycle', cron: '*/15 * * * *' },
  { name: 'exploration', method: 'loop.exploration.run_cycle', cron: '*/10 * * * *' },
  { name: 'discovery', method: 'loop.discovery.run_cycle', cron: '*/15 * * * *' },
  { name: 'bug_finder', method: 'loop.bug_finder.run_cycle', cron: '*/30 * * * *' },
  { name: 'self_learning', method: 'loop.self_learning.run_cycle', cron: '*/20 * * * *' },
  { name: 'meta_cognition', method: 'loop.meta_cognition.run_cycle', cron: '*/15 * * * *' },
  { name: 'self_upgrading', method: 'loop.self_upgrading.run_cycle', cron: '0 * * * *' },
  { name: 'self_updating', method: 'loop.self_updating.run_cycle', cron: '*/30 * * * *' },
  { name: 'self_driven', method: 'loop.self_driven.run_cycle', cron: '*/10 * * * *' },
  { name: 'cpel', method: 'loop.cpel.run_cycle', cron: '*/15 * * * *' },
  { name: 'context_engineering', method: 'loop.context_engineering.run_cycle', cron: '*/15 * * * *' },
  { name: 'debugging', method: 'loop.debugging.run_cycle', cron: '*/20 * * * *' },
  { name: 'web_monitor', method: 'loop.web_monitor.run_cycle', cron: '*/10 * * * *' },
];

class DaemonMaster extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      workerCount: options.workerCount || os.cpus().length,
      agentLoops: options.agentLoops || 15,
      heartbeatInterval: options.heartbeatInterval || 30000,
      restartDelay: options.restartDelay || 5000,
      maxRestarts: options.maxRestarts || 10,
      restartWindow: options.restartWindow || 60000,
      enableControlServer: options.enableControlServer !== false,
      controlPort: options.controlPort || 8080,
      ...options
    };
    
    this.workers = new Map();
    this.agentLoopWorkers = new Map();
    this.ioWorkers = new Map();
    this.taskWorkers = new Map();
    this._workerMetadata = new WeakMap();
    this._workerRestartHistory = new Map(); // key: "type-index/service", value: [timestamps]
    this.restartCount = 0;
    this.lastRestartTime = Date.now();
    this.isShuttingDown = false;
    this.state = 'initializing';
    this.cronScheduler = null;
    this.healthMonitor = null;
    this.controlServer = null;
    this.messageBus = null;
    this.bridge = null;
    this.bridgeReady = false;
  }

  async initialize() {
    logger.info(`[Master ${process.pid}] Initializing OpenClawAgent Daemon...`);
    
    // Setup process event handlers
    this.setupProcessHandlers();

    // Start Python bridge (before workers so they can use it)
    await this.startPythonBridge();

    // Initialize state manager (needs bridge)
    await this.initializeStateManager();

    // Start cron scheduler
    await this.startCronScheduler();

    // Start heartbeat monitor
    this.startHeartbeatMonitor();

    // Fork worker processes
    await this.forkWorkers();
    
    // Start HTTP control server
    if (this.options.enableControlServer) {
      await this.startControlServer();
    }
    
    // Initialize message bus
    this.initializeMessageBus();
    
    this.state = 'running';
    this.emit('ready');
    logger.info(`[Master ${process.pid}] Daemon initialized successfully`);
    logger.info(`[Master ${process.pid}] Control server: http://localhost:${this.options.controlPort}`);
  }
  
  async startPythonBridge() {
    const maxAttempts = 3;
    const retryDelay = 5000;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        this.bridge = getBridge();
        await this.bridge.start();
        this.bridgeReady = true;
        logger.info('[Master] Python bridge started');

        this.bridge.on('error', (err) => {
          logger.error('[Master] Python bridge error:', err);
        });

        this.bridge.on('fatal', (err) => {
          logger.error('[Master] Python bridge fatal error:', err);
          this.bridgeReady = false;
        });

        return; // Success
      } catch (error) {
        logger.error(`[Master] Failed to start Python bridge (attempt ${attempt}/${maxAttempts}):`, error);
        this.bridgeReady = false;
        if (attempt < maxAttempts) {
          logger.info(`[Master] Retrying bridge start in ${retryDelay}ms...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
      }
    }
    logger.error('[Master] Python bridge failed after all retry attempts');
  }

  async startControlServer() {
    try {
      this.controlServer = new ControlServer(this, {
        port: this.options.controlPort,
        host: '127.0.0.1'
      });
      await this.controlServer.start();
      logger.info(`[Master] Control server started on port ${this.options.controlPort}`);
    } catch (error) {
      logger.error('[Master] Failed to start control server:', error);
      // Don't fail initialization if control server fails
    }
  }
  
  initializeMessageBus() {
    try {
      this.messageBus = new MessageBus();
      
      // Override message bus methods to use our worker tracking
      this.messageBus.findIOWorker = (service) => {
        const info = this.ioWorkers.get(service);
        return info ? info.worker : null;
      };
      
      this.messageBus.findAvailableTaskWorker = () => {
        // Find first available task worker
        for (const [id, info] of this.taskWorkers) {
          if (info.state === 'ready' || info.state === 'starting') {
            return info.worker;
          }
        }
        // Fallback to first task worker
        if (this.taskWorkers.size === 0) return null;
        const first = this.taskWorkers.values().next().value;
        return first ? first.worker : null;
      };
      
      this.messageBus.findWorker = (workerId) => {
        // Check all worker maps
        const agentLoop = this.agentLoopWorkers.get(parseInt(workerId));
        if (agentLoop) return agentLoop.worker;
        
        for (const [service, info] of this.ioWorkers) {
          if (info.worker.id === parseInt(workerId)) return info.worker;
        }
        
        const task = this.taskWorkers.get(parseInt(workerId));
        if (task) return task.worker;
        
        return cluster.workers[workerId];
      };
      
      // Wire Python bridge routing
      this.messageBus.routeBridgeRequest = (workerId, data) => {
        if (!this.bridgeReady || !this.bridge) {
          logger.warn('[Master] Bridge request but bridge not ready');
          const worker = this.messageBus.findWorker(workerId);
          if (worker) {
            worker.send({
              type: 'python-bridge-response',
              data: { id: data.id, error: 'Bridge not ready' },
            });
          }
          return;
        }

        this.bridge.call(data.method, data.params).then(result => {
          const worker = this.messageBus.findWorker(workerId);
          if (worker && !worker.isDead()) {
            worker.send({
              type: 'python-bridge-response',
              data: { id: data.id, result },
            });
          }
        }).catch(error => {
          const worker = this.messageBus.findWorker(workerId);
          if (worker && !worker.isDead()) {
            worker.send({
              type: 'python-bridge-response',
              data: { id: data.id, error: error.message },
            });
          }
        });
      };

      logger.info('[Master] Message bus initialized');
    } catch (error) {
      logger.error('[Master] Failed to initialize message bus:', error);
    }
  }

  setupProcessHandlers() {
    // Graceful shutdown handlers
    process.on('SIGTERM', () => this.shutdown('SIGTERM'));
    process.on('SIGINT', () => this.shutdown('SIGINT'));
    
    // Windows service shutdown
    if (process.platform === 'win32') {
      process.on('message', (msg) => {
        if (msg === 'shutdown') {
          this.shutdown('service-stop');
        }
      });
    }
    
    // Uncaught exception handler
    process.on('uncaughtException', (error) => {
      logger.error('[Master] Uncaught exception:', error);
      this.emit('error', error);
      this.shutdown('uncaughtException');
    });

    // Unhandled rejection handler
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('[Master] Unhandled rejection at:', promise, 'reason:', reason);
      this.emit('error', { reason, promise });
    });

    // Cluster event handlers
    cluster.on('fork', (worker) => {
      logger.info(`[Master] Worker ${worker.id} forked (PID: ${worker.process.pid})`);
    });

    cluster.on('online', (worker) => {
      logger.info(`[Master] Worker ${worker.id} is online`);
    });

    cluster.on('listening', (worker, address) => {
      logger.info(`[Master] Worker ${worker.id} is listening on ${address.address}:${address.port}`);
    });
  }

  async forkWorkers() {
    logger.info(`[Master] Forking ${this.options.agentLoops} agent loop workers...`);
    
    // Fork agent loop workers
    for (let i = 0; i < this.options.agentLoops; i++) {
      await this.forkAgentLoopWorker(i);
    }
    
    // Fork I/O workers
    await this.forkIOWorkers();
    
    // Fork task workers
    await this.forkTaskWorkers();
  }

  async forkAgentLoopWorker(index) {
    const workerEnv = {
      WORKER_TYPE: 'agent-loop',
      WORKER_INDEX: index.toString(),
      WORKER_ID: `agent-${index}`,
      AGENT_LOOP_CONFIG: JSON.stringify(this.getAgentLoopConfig(index))
    };

    const worker = cluster.fork(workerEnv);

    // Store metadata in WeakMap to avoid attaching properties to worker objects
    this._workerMetadata.set(worker, { type: 'agent-loop', index, service: null });

    worker.on('message', (msg) => this.handleWorkerMessage(worker, msg));
    worker.on('exit', (code, signal) => this.handleWorkerExit(worker, code, signal));

    this.agentLoopWorkers.set(worker.id, {
      worker,
      index,
      type: 'agent-loop',
      startTime: Date.now(),
      state: 'starting'
    });

    return new Promise((resolve) => {
      worker.once('online', () => {
        logger.info(`[Master] Agent loop worker ${index} (PID: ${worker.process.pid}) online`);
        resolve(worker);
      });
    });
  }

  async forkIOWorkers() {
    const ioServices = ['gmail', 'browser', 'twilio', 'tts', 'stt'];
    
    for (const service of ioServices) {
      const workerEnv = {
        WORKER_TYPE: 'io',
        IO_SERVICE: service,
        WORKER_ID: `io-${service}`
      };

      const worker = cluster.fork(workerEnv);

      this._workerMetadata.set(worker, { type: 'io', index: null, service });

      worker.on('message', (msg) => this.handleWorkerMessage(worker, msg));
      worker.on('exit', (code, signal) => this.handleWorkerExit(worker, code, signal));

      this.ioWorkers.set(service, {
        worker,
        service,
        type: 'io',
        startTime: Date.now(),
        state: 'starting'
      });
    }
  }

  async forkTaskWorkers() {
    const taskCount = Math.max(2, Math.floor(this.options.workerCount / 2));
    
    for (let i = 0; i < taskCount; i++) {
      const workerEnv = {
        WORKER_TYPE: 'task',
        WORKER_INDEX: i.toString(),
        WORKER_ID: `task-${i}`
      };

      const worker = cluster.fork(workerEnv);

      this._workerMetadata.set(worker, { type: 'task', index: i, service: null });

      worker.on('message', (msg) => this.handleWorkerMessage(worker, msg));
      worker.on('exit', (code, signal) => this.handleWorkerExit(worker, code, signal));
      
      this.taskWorkers.set(worker.id, {
        worker,
        index: i,
        type: 'task',
        startTime: Date.now(),
        state: 'starting'
      });
    }
  }

  handleWorkerMessage(worker, message) {
    // Also route through message bus for inter-worker communication
    if (this.messageBus) {
      const meta = this._workerMetadata.get(worker) || { type: 'unknown', service: null };
      this.messageBus.handleWorkerMessage(worker.id, meta.type, message);
    }
    
    switch(message.type) {
      case 'heartbeat':
        this.updateWorkerHealth(worker.id, message.data);
        break;
      case 'task-complete':
        this.emit('task-complete', { workerId: worker.id, ...message.data });
        break;
      case 'task-error':
        logger.error(`[Master] Task error from worker ${worker.id}:`, message.data);
        this.emit('task-error', { workerId: worker.id, ...message.data });
        break;
      case 'state-update':
        this.handleStateUpdate(worker.id, message.data);
        break;
      case 'request-restart':
        this.restartWorker(worker.id);
        break;
      // python-bridge-request is handled exclusively by the message bus (via routeBridgeRequest)
      // to avoid double bridge calls. See initializeMessageBus().
      case 'log':
        logger.log(message.level || 'info', `[Worker ${worker.id}] ${message.message}`, message.meta);
        break;
      default:
        logger.debug(`[Master] Message from worker ${worker.id}:`, message);
    }
  }

  _getRestartBackoff(restartKey) {
    const history = this._workerRestartHistory.get(restartKey) || [];
    const now = Date.now();
    // Remove entries older than 60s
    const recent = history.filter(ts => now - ts < 60000);
    recent.push(now);
    this._workerRestartHistory.set(restartKey, recent);
    const consecutiveRestarts = recent.length;
    if (consecutiveRestarts <= 1) return this.options.restartDelay;
    return Math.min(5000 * Math.pow(2, consecutiveRestarts - 1), 60000);
  }

  handleWorkerExit(worker, code, signal) {
    // Get worker metadata from WeakMap
    const meta = this._workerMetadata.get(worker) || { type: 'unknown', index: 0, service: null };
    logger.info(`[Master] Worker ${worker.id} (${meta.type}${meta.index !== null ? '-' + meta.index : ''}${meta.service ? '-' + meta.service : ''}) exited (code: ${code}, signal: ${signal})`);

    // Reject pending bridge requests associated with this dead worker
    if (this.messageBus) {
      for (const [reqId, pending] of this.messageBus.pendingRequests) {
        if (pending.fromWorkerId === worker.id) {
          clearTimeout(pending.timeout);
          this.messageBus.pendingRequests.delete(reqId);
          logger.warn(`[Master] Rejected pending request ${reqId} for dead worker ${worker.id}`);
        }
      }
    }

    // Remove from tracking maps
    this.agentLoopWorkers.delete(worker.id);
    this.ioWorkers.delete(meta.service);
    this.taskWorkers.delete(worker.id);

    // Check if we should restart
    if (!this.isShuttingDown && code !== 0) {
      this.checkRestartRate();

      const restartKey = `${meta.type}-${meta.index !== null ? meta.index : meta.service}`;
      const backoff = this._getRestartBackoff(restartKey);
      logger.info(`[Master] Scheduling restart for ${restartKey} in ${backoff}ms`);

      // Restart the worker based on its type with backoff
      setTimeout(() => {
        if (meta.type === 'agent-loop') {
          this.forkAgentLoopWorker(meta.index || 0);
        } else if (meta.type === 'io' && meta.service) {
          this.forkSingleIOWorker(meta.service);
        } else if (meta.type === 'task') {
          this.restartSingleTaskWorker(meta.index || 0);
        }
      }, backoff);
    }
  }
  
  async restartSingleTaskWorker(index) {
    const workerEnv = {
      WORKER_TYPE: 'task',
      WORKER_INDEX: index.toString(),
      WORKER_ID: `task-${index}`
    };

    const worker = cluster.fork(workerEnv);

    this._workerMetadata.set(worker, { type: 'task', index, service: null });

    worker.on('message', (msg) => this.handleWorkerMessage(worker, msg));
    worker.on('exit', (code, signal) => this.handleWorkerExit(worker, code, signal));

    this.taskWorkers.set(worker.id, {
      worker,
      index,
      type: 'task',
      startTime: Date.now(),
      state: 'starting'
    });
  }

  async forkSingleIOWorker(service) {
    const workerEnv = {
      WORKER_TYPE: 'io',
      IO_SERVICE: service,
      WORKER_ID: `io-${service}`
    };

    const worker = cluster.fork(workerEnv);
    this._workerMetadata.set(worker, { type: 'io', index: null, service });

    worker.on('message', (msg) => this.handleWorkerMessage(worker, msg));
    worker.on('exit', (code, signal) => this.handleWorkerExit(worker, code, signal));

    this.ioWorkers.set(service, {
      worker,
      service,
      type: 'io',
      startTime: Date.now(),
      state: 'starting'
    });
  }

  dispatchTask(task) {
    // Find a ready worker; skip busy workers
    let readyWorker = null;
    let fallbackWorker = null;

    for (const [id, info] of this.agentLoopWorkers) {
      if (info.state === 'ready') {
        readyWorker = info.worker;
        break;
      }
      // Track a non-busy fallback (starting workers may become ready)
      if (info.state !== 'busy' && !fallbackWorker) {
        fallbackWorker = info.worker;
      }
    }

    const worker = readyWorker || fallbackWorker;
    if (worker) {
      worker.send({ type: 'task', data: task });
      return true;
    }
    return false;
  }

  checkRestartRate() {
    const now = Date.now();
    if (now - this.lastRestartTime > this.options.restartWindow) {
      this.restartCount = 0;
      this.lastRestartTime = now;
    }
    
    this.restartCount++;
    
    if (this.restartCount > this.options.maxRestarts) {
      logger.error('[Master] Maximum restart rate exceeded. Initiating shutdown.');
      this.shutdown('restart-rate-exceeded');
    }
  }

  async startCronScheduler() {
    try {
      const { SchedulerSystem } = require('./dist/src/scheduler-system');
      const { registerAgentLoops } = require('./dist/src/agent-loops');

      const dataDir = process.env.STATE_DIR || './data/scheduler';
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }

      this.cronScheduler = new SchedulerSystem({
        dataDir,
        timezone: process.env.SCHEDULER_TIMEZONE || 'UTC',
        heartbeat: {
          interval: 30000,
          timeout: 10000,
          nodeId: `master-${process.pid}`,
        },
        overlapPrevention: {
          strategy: 'skip',
        },
        retry: {
          maxAttempts: 3,
          strategy: 'exponential',
          baseDelay: 5000,
          maxDelay: 60000,
        },
      });

      await this.cronScheduler.initialize();
      registerAgentLoops(this.cronScheduler);
      await this.cronScheduler.start();

      logger.info('[Master] TS Scheduler started with all loops registered');
    } catch (error) {
      const distExists = fs.existsSync('./dist');
      logger.warn(`[Master] TS Scheduler failed (dist/ exists: ${distExists}): ${error.message}. Falling back to cron-scheduler.`);
      // Fallback to basic cron scheduler
      const CronScheduler = require('./cron-scheduler');
      this.cronScheduler = new CronScheduler();
      this.cronScheduler.initialize();

      // Register all cognitive loops as cron jobs in fallback mode
      this._registerFallbackLoops();
      logger.info('[Master] Fallback cron scheduler started with cognitive loops');
    }
  }

  _registerFallbackLoops() {
    for (const loop of COGNITIVE_LOOP_SCHEDULES) {
      try {
        this.cronScheduler.scheduleJob(loop.name, loop.cron, () => {
          this.dispatchLoopCycle(loop.name, loop.method);
        });
      } catch (err) {
        logger.warn(`[Master] Failed to register fallback loop ${loop.name}: ${err.message}`);
      }
    }
    logger.info(`[Master] Registered ${COGNITIVE_LOOP_SCHEDULES.length} cognitive loops in fallback scheduler`);
  }

  dispatchLoopCycle(loopName, method) {
    if (!this.bridgeReady || !this.bridge) {
      logger.debug(`[Master] Skipping loop ${loopName}: bridge not ready`);
      return;
    }

    // Initialize failure tracking map on first use
    if (!this._loopFailures) {
      this._loopFailures = new Map();
    }
    const tracker = this._loopFailures.get(loopName) || { count: 0, cooldownUntil: 0 };

    // Check cooldown
    if (tracker.cooldownUntil > Date.now()) {
      logger.debug(`[Master] Skipping loop ${loopName}: in cooldown until ${new Date(tracker.cooldownUntil).toISOString()}`);
      return;
    }

    this.bridge.call(method, {}).then(result => {
      // Reset consecutive failure count on success
      tracker.count = 0;
      tracker.cooldownUntil = 0;
      this._loopFailures.set(loopName, tracker);
      logger.debug(`[Master] Loop ${loopName} completed:`, result);
    }).catch(error => {
      tracker.count++;
      if (tracker.count >= 3) {
        const cooldownMs = this.options.loopCooldownMs || 5 * 60 * 1000; // default 5 minutes
        tracker.cooldownUntil = Date.now() + cooldownMs;
        logger.warn(`[Master] Loop ${loopName} paused after ${tracker.count} consecutive failures. Cooldown until ${new Date(tracker.cooldownUntil).toISOString()}`);
        tracker.count = 0; // reset for after cooldown
      }
      this._loopFailures.set(loopName, tracker);
      logger.warn(`[Master] Loop ${loopName} failed: ${error.message}`);
    });
  }

  startHeartbeatMonitor() {
    const HealthMonitor = require('./health-monitor');
    this.healthMonitor = new HealthMonitor({
      checkInterval: this.options.heartbeatInterval
    });
    
    this.healthMonitor.on('health', (health) => {
      logger.debug('[Master] Health check:', health);
    });
    
    this.healthMonitor.on('degraded', (health) => {
      logger.warn('[Master] System health degraded:', health);
    });
    
    this.healthMonitor.start();
    logger.info('[Master] Heartbeat monitor started');
  }

  checkWorkerHealth() {
    const now = Date.now();
    const timeout = this.options.heartbeatInterval * 2;
    
    for (const [id, workerInfo] of this.agentLoopWorkers) {
      if (workerInfo.lastHeartbeat && (now - workerInfo.lastHeartbeat > timeout)) {
        logger.warn(`[Master] Worker ${id} heartbeat timeout. Restarting...`);
        this.restartWorker(id);
      }
    }
  }

  updateWorkerHealth(workerId, data) {
    const worker = this.agentLoopWorkers.get(workerId) || 
                   this.taskWorkers.get(workerId);
    if (worker) {
      worker.lastHeartbeat = Date.now();
      worker.health = data;
    }
  }

  async shutdown(reason) {
    if (this.isShuttingDown) return;
    this.isShuttingDown = true;
    this.state = 'shutting-down';
    
    logger.info(`[Master] Shutdown initiated: ${reason}`);
    
    // Stop control server
    if (this.controlServer) {
      await this.controlServer.stop();
    }
    
    // Stop cron scheduler
    if (this.cronScheduler) {
      this.cronScheduler.stop();
    }
    
    // Stop health monitor
    if (this.healthMonitor) {
      this.healthMonitor.stop();
    }
    
    // Stop accepting new work
    this.emit('shutdown', reason);
    
    // Gracefully disconnect all workers
    const disconnectPromises = [];
    
    for (const [id, workerInfo] of this.agentLoopWorkers) {
      disconnectPromises.push(this.gracefulDisconnect(workerInfo.worker));
    }
    
    for (const [id, workerInfo] of this.taskWorkers) {
      disconnectPromises.push(this.gracefulDisconnect(workerInfo.worker));
    }

    for (const [service, workerInfo] of this.ioWorkers) {
      disconnectPromises.push(this.gracefulDisconnect(workerInfo.worker));
    }

    // Wait for graceful shutdown with timeout
    await Promise.race([
      Promise.all(disconnectPromises),
      new Promise(resolve => setTimeout(resolve, 30000))
    ]);
    
    // Force kill any remaining workers
    for (const [id, workerInfo] of this.agentLoopWorkers) {
      if (!workerInfo.worker.isDead()) {
        workerInfo.worker.kill('SIGKILL');
      }
    }
    for (const [id, workerInfo] of this.taskWorkers) {
      if (!workerInfo.worker.isDead()) {
        workerInfo.worker.kill('SIGKILL');
      }
    }
    for (const [service, workerInfo] of this.ioWorkers) {
      if (!workerInfo.worker.isDead()) {
        workerInfo.worker.kill('SIGKILL');
      }
    }

    // Save state BEFORE stopping bridge
    await this.saveState();

    // Stop Python bridge
    if (this.bridge) {
      await this.bridge.stop();
    }

    logger.info('[Master] Shutdown complete');
    process.exit(0);
  }

  gracefulDisconnect(worker) {
    return new Promise((resolve) => {
      worker.send({ type: 'shutdown' });
      worker.once('disconnect', resolve);
      setTimeout(resolve, 10000); // 10 second timeout
    });
  }

  async initializeStateManager() {
    // Initialize state persistence via memory DB
    try {
      if (this.bridge && this.bridge.isReady) {
        await this.bridge.call('memory.initialize', {});
        logger.info('[Master] State manager initialized (memory DB)');
      } else {
        logger.info('[Master] State manager initialized (bridge not ready yet, will init on first use)');
      }
    } catch (error) {
      logger.warn('[Master] State manager init deferred:', error.message);
    }
  }

  async saveState() {
    // Persist daemon state to memory DB before shutdown
    try {
      if (this.bridge && this.bridge.isReady) {
        await this.bridge.call('memory.store', {
          id: 'daemon-state',
          type: 'system',
          content: JSON.stringify({
            state: this.state,
            agentLoops: this.agentLoopWorkers.size,
            ioWorkers: this.ioWorkers.size,
            taskWorkers: this.taskWorkers.size,
            restartCount: this.restartCount,
            shutdownTime: Date.now(),
            uptime: process.uptime(),
          }),
          source: 'daemon-master',
          importance: 0.8,
          tags: ['system', 'daemon-state'],
        });
        logger.info('[Master] State saved to memory DB');
      } else {
        logger.warn('[Master] Bridge not available, state not saved');
      }
    } catch (error) {
      logger.warn('[Master] Failed to save state:', error.message);
    }
  }

  handleStateUpdate(workerId, data) {
    // Persist worker state updates to memory DB (fire-and-forget)
    if (this.bridge && this.bridge.isReady) {
      this.bridge.call('memory.store', {
        type: 'episodic',
        content: JSON.stringify(data),
        source: `worker-${workerId}`,
        tags: ['state-update', `worker-${workerId}`],
      }).catch(err => {
        logger.debug(`[Master] State update store failed: ${err.message}`);
      });
    }
  }

  restartWorker(workerId) {
    const worker = cluster.workers[workerId];
    if (worker) {
      worker.kill('SIGTERM');
    }
  }

  getAgentLoopConfig(index) {
    return {
      loopId: index,
      priority: index < 5 ? 'high' : 'normal',
      capabilities: this.getLoopCapabilities(index)
    };
  }

  getLoopCapabilities(index) {
    const capabilities = [
      ['gmail', 'browser'],
      ['twilio', 'tts'],
      ['stt', 'system'],
      ['gmail', 'twilio'],
      ['browser', 'system'],
      ['gmail'],
      ['browser'],
      ['twilio'],
      ['tts'],
      ['stt'],
      ['system'],
      ['gmail', 'browser', 'twilio'],
      ['tts', 'stt'],
      ['gmail', 'system'],
      ['all']
    ];
    return capabilities[index] || ['basic'];
  }
}

module.exports = DaemonMaster;

// Only auto-run if executed directly (not when required as a module)
const _isPrimary = cluster.isPrimary !== undefined ? cluster.isPrimary : cluster.isMaster;
if (require.main === module && _isPrimary) {
  const daemon = new DaemonMaster({
    workerCount: os.cpus().length,
    agentLoops: 15,
    heartbeatInterval: 30000,
    restartDelay: 5000,
    maxRestarts: 10,
    restartWindow: 60000
  });
  
  daemon.initialize().catch(error => {
    logger.error('[Master] Initialization failed:', error);
    process.exit(1);
  });
  
}
