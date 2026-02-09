/**
 * OpenClawAgent - Base Worker Class
 * Abstract base class for all worker processes
 */

const logger = require('./logger');

class WorkerBase {
  constructor() {
    this.workerType = process.env.WORKER_TYPE;
    this.workerId = process.env.WORKER_ID;
    this.workerIndex = parseInt(process.env.WORKER_INDEX) || 0;
    this.isShuttingDown = false;
    this.heartbeatInterval = null;
    this.heartbeatMs = parseInt(process.env.WORKER_HEARTBEAT_MS) || 30000;
    this.state = 'initializing';
  }

  async initialize() {
    logger.info(`[Worker ${this.workerId}] Initializing ${this.workerType} worker...`);
    
    this.setupMessageHandlers();
    this.setupErrorHandlers();
    this.startHeartbeat();
    
    await this.onInitialize();
    
    this.state = 'ready';
    logger.info(`[Worker ${this.workerId}] Initialized and ready`);
  }

  setupMessageHandlers() {
    process.on('message', (msg) => {
      if (typeof msg === 'string') {
        // Handle simple string messages (e.g., 'shutdown')
        if (msg === 'shutdown') {
          this.handleShutdown();
        }
        return;
      }

      // Validate msg is an object with a string type field
      if (msg === null || typeof msg !== 'object' || typeof msg.type !== 'string') {
        logger.warn(`[Worker ${this.workerId}] Malformed IPC message (missing object or type):`, typeof msg);
        return;
      }

      switch(msg.type) {
        case 'shutdown':
          this.handleShutdown();
          break;
        case 'reload':
          this.handleReload();
          break;
        case 'task':
          this.handleTask(msg.data);
          break;
        case 'pause':
          this.handlePause();
          break;
        case 'resume':
          this.handleResume();
          break;
        default:
          this.onMessage(msg);
      }
    });
  }

  setupErrorHandlers() {
    process.on('uncaughtException', (error) => {
      logger.error(`[Worker ${this.workerId}] Uncaught exception:`, error);
      this.sendToMaster({
        type: 'worker-error',
        data: { error: error.message, stack: error.stack }
      });
      process.exit(1);
    });

    process.on('unhandledRejection', (reason, promise) => {
      logger.error(`[Worker ${this.workerId}] Unhandled rejection:`, reason);
      this.sendToMaster({
        type: 'worker-error',
        data: { reason, promise }
      });
    });
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      this.sendHeartbeat();
    }, this.heartbeatMs);
  }

  sendHeartbeat() {
    this.sendToMaster({
      type: 'heartbeat',
      data: {
        workerId: this.workerId,
        workerType: this.workerType,
        timestamp: Date.now(),
        memory: process.memoryUsage(),
        uptime: process.uptime(),
        state: this.state
      }
    });
  }

  sendToMaster(message) {
    if (process.send) {
      process.send(message);
    }
  }

  async handleShutdown() {
    logger.info(`[Worker ${this.workerId}] Shutdown requested`);
    this.isShuttingDown = true;
    this.state = 'shutting-down';
    
    clearInterval(this.heartbeatInterval);
    
    await this.onShutdown();
    
    logger.info(`[Worker ${this.workerId}] Shutdown complete`);
    
    if (process.disconnect) {
      process.disconnect();
    }
    process.exit(0);
  }

  async handleReload() {
    logger.info(`[Worker ${this.workerId}] Reload requested`);
    await this.onReload();
  }

  async handleTask(data) {
    // Validate task data has required taskId (accept both taskId and id)
    if (data === null || typeof data !== 'object' || (!data.taskId && !data.id)) {
      logger.warn(`[Worker ${this.workerId}] Malformed task message: missing data or taskId/id`);
      return;
    }
    const taskId = data.taskId || data.id;

    this.state = 'busy';
    try {
      const result = await this.onTask(data);
      this.sendToMaster({
        type: 'task-complete',
        data: { taskId, result }
      });
    } catch (error) {
      logger.error(`[Worker ${this.workerId}] Task error:`, error);
      this.sendToMaster({
        type: 'task-error',
        data: { taskId, error: error.message, stack: error.stack }
      });
    } finally {
      this.state = 'ready';
    }
  }

  handlePause() {
    logger.info(`[Worker ${this.workerId}] Pausing`);
    this.state = 'paused';
    this.onPause();
  }

  handleResume() {
    logger.info(`[Worker ${this.workerId}] Resuming`);
    this.state = 'ready';
    this.onResume();
  }

  // Override in subclasses
  async onInitialize() {
    // Subclass implementation
  }

  async onShutdown() {
    // Subclass implementation
  }

  async onReload() {
    // Subclass implementation
  }

  async onTask(data) {
    throw new Error('onTask() must be implemented by subclass');
  }

  onMessage(msg) {
    // Subclass implementation for custom messages
    logger.debug(`[Worker ${this.workerId}] Received message:`, msg);
  }

  onPause() {
    // Subclass implementation
  }

  onResume() {
    // Subclass implementation
  }

  // Utility methods
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async withTimeout(promise, ms, message = 'Operation timed out') {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), ms);
    });
    try {
      return await Promise.race([promise, timeout]);
    } finally {
      clearTimeout(timer);
    }
  }
}

module.exports = WorkerBase;
