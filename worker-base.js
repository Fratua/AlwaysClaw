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
    }, 30000);
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
    this.state = 'busy';
    try {
      const result = await this.onTask(data);
      this.sendToMaster({
        type: 'task-complete',
        data: { taskId: data.taskId, result }
      });
    } catch (error) {
      logger.error(`[Worker ${this.workerId}] Task error:`, error);
      this.sendToMaster({
        type: 'task-error',
        data: { taskId: data.taskId, error: error.message, stack: error.stack }
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
    // Subclass implementation
    return { status: 'completed' };
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
    return Promise.race([
      promise,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error(message)), ms)
      )
    ]);
  }
}

module.exports = WorkerBase;
