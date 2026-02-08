/**
 * OpenClawAgent - Inter-Worker Message Bus
 * Enables communication between agent loop workers and I/O workers
 */

const EventEmitter = require('events');
const logger = require('./logger');

/**
 * MessageBus - Central message routing system for worker communication
 * 
 * Used in master process to route messages between:
 * - Agent Loop Workers → I/O Workers (request actions)
 * - I/O Workers → Agent Loop Workers (return results)
 * - Any worker → Master (status, errors)
 */
class MessageBus extends EventEmitter {
  constructor() {
    super();
    this.pendingRequests = new Map();
    this.requestTimeout = 30000; // 30 seconds default
    this.requestId = 0;
  }

  /**
   * Register a worker with the message bus
   * @param {string} workerId - Worker identifier
   * @param {cluster.Worker} worker - Cluster worker instance
   * @param {string} workerType - Type of worker (agent-loop, io, task)
   */
  registerWorker(workerId, worker, workerType) {
    logger.debug(`[MessageBus] Registering worker: ${workerId} (${workerType})`);
    
    // Set up message handler for this worker
    worker.on('message', (msg) => {
      this.handleWorkerMessage(workerId, workerType, msg);
    });
  }

  /**
   * Handle incoming messages from workers
   */
  handleWorkerMessage(workerId, workerType, message) {
    logger.debug(`[MessageBus] Message from ${workerId}:`, message.type);

    switch (message.type) {
      case 'io-request':
        this.routeIORequest(workerId, message.data);
        break;
      case 'io-response':
        this.handleIOResponse(message.data);
        break;
      case 'task-request':
        this.routeTaskRequest(workerId, message.data);
        break;
      case 'task-response':
        this.handleTaskResponse(message.data);
        break;
      case 'python-bridge-request':
        this.routeBridgeRequest(workerId, message.data);
        break;
      case 'python-bridge-response':
        this.handleBridgeResponse(message.data);
        break;
      case 'broadcast':
        this.broadcast(message.data, workerId);
        break;
      default:
        // Pass through to other listeners
        this.emit('message', { workerId, workerType, message });
    }
  }

  /**
   * Route an I/O request from an agent loop worker to the appropriate I/O worker
   */
  routeIORequest(fromWorkerId, request) {
    const { service, action, data, requestId } = request;
    
    logger.debug(`[MessageBus] Routing I/O request: ${service}.${action} (req: ${requestId})`);

    // Find the appropriate I/O worker
    const ioWorker = this.findIOWorker(service);
    
    if (!ioWorker) {
      logger.error(`[MessageBus] No I/O worker available for service: ${service}`);
      this.sendErrorResponse(fromWorkerId, requestId, `Service unavailable: ${service}`);
      return;
    }

    // Store pending request
    this.pendingRequests.set(requestId, {
      fromWorkerId,
      service,
      action,
      startTime: Date.now(),
      timeout: setTimeout(() => {
        this.handleRequestTimeout(requestId);
      }, this.requestTimeout)
    });

    // Forward request to I/O worker
    ioWorker.send({
      type: 'io-request',
      data: {
        requestId,
        service,
        action,
        data,
        fromWorkerId
      }
    });
  }

  /**
   * Handle I/O response from I/O worker
   */
  handleIOResponse(response) {
    const { requestId, result, error } = response;
    
    const pending = this.pendingRequests.get(requestId);
    if (!pending) {
      logger.warn(`[MessageBus] Received response for unknown request: ${requestId}`);
      return;
    }

    // Clear timeout
    clearTimeout(pending.timeout);
    this.pendingRequests.delete(requestId);

    // Send response back to requesting worker
    const fromWorker = this.findWorker(pending.fromWorkerId);
    if (fromWorker) {
      fromWorker.send({
        type: 'io-response',
        data: {
          requestId,
          service: pending.service,
          action: pending.action,
          result,
          error,
          duration: Date.now() - pending.startTime
        }
      });
    }

    logger.debug(`[MessageBus] I/O response sent for request: ${requestId}`);
  }

  /**
   * Route a task request to a task worker
   */
  routeTaskRequest(fromWorkerId, request) {
    const { taskType, data, requestId, priority = 'normal' } = request;
    
    logger.debug(`[MessageBus] Routing task request: ${taskType} (req: ${requestId})`);

    // Find an available task worker
    const taskWorker = this.findAvailableTaskWorker();
    
    if (!taskWorker) {
      logger.error(`[MessageBus] No task worker available`);
      this.sendErrorResponse(fromWorkerId, requestId, 'No task workers available');
      return;
    }

    // Store pending request
    this.pendingRequests.set(requestId, {
      fromWorkerId,
      taskType,
      startTime: Date.now(),
      timeout: setTimeout(() => {
        this.handleRequestTimeout(requestId);
      }, this.requestTimeout)
    });

    // Forward request to task worker
    taskWorker.send({
      type: 'task',
      data: {
        id: requestId,
        type: taskType,
        payload: data,
        priority,
        requestId
      }
    });
  }

  /**
   * Handle task response from task worker
   */
  handleTaskResponse(response) {
    const { requestId, result, error } = response;
    
    const pending = this.pendingRequests.get(requestId);
    if (!pending) {
      logger.warn(`[MessageBus] Received task response for unknown request: ${requestId}`);
      return;
    }

    // Clear timeout
    clearTimeout(pending.timeout);
    this.pendingRequests.delete(requestId);

    // Send response back to requesting worker
    const fromWorker = this.findWorker(pending.fromWorkerId);
    if (fromWorker) {
      fromWorker.send({
        type: 'task-response',
        data: {
          requestId,
          taskType: pending.taskType,
          result,
          error,
          duration: Date.now() - pending.startTime
        }
      });
    }
  }

  /**
   * Handle request timeout
   */
  handleRequestTimeout(requestId) {
    const pending = this.pendingRequests.get(requestId);
    if (pending) {
      logger.error(`[MessageBus] Request timeout: ${requestId}`);
      this.pendingRequests.delete(requestId);
      
      this.sendErrorResponse(
        pending.fromWorkerId, 
        requestId, 
        'Request timeout'
      );
    }
  }

  /**
   * Send error response to worker
   */
  sendErrorResponse(workerId, requestId, error) {
    const worker = this.findWorker(workerId);
    if (worker) {
      worker.send({
        type: 'io-response',
        data: {
          requestId,
          error,
          result: null
        }
      });
    }
  }

  /**
   * Broadcast a message to all workers
   */
  broadcast(data, excludeWorkerId = null) {
    const cluster = require('cluster');
    
    for (const [id, worker] of Object.entries(cluster.workers)) {
      if (worker.id !== excludeWorkerId && !worker.isDead()) {
        worker.send({
          type: 'broadcast',
          data
        });
      }
    }
  }

  /**
   * Route a Python bridge request from a worker to the master's PythonBridge.
   * Override this in DaemonMaster to wire to the actual bridge.
   */
  routeBridgeRequest(workerId, data) {
    // Overridden by DaemonMaster
    logger.warn('[MessageBus] routeBridgeRequest not wired - no bridge available');
  }

  /**
   * Handle a Python bridge response (forward back to requesting worker).
   */
  handleBridgeResponse(data) {
    // Overridden by DaemonMaster
    logger.warn('[MessageBus] handleBridgeResponse not wired');
  }

  /**
   * Find I/O worker by service name
   * Override this method to integrate with DaemonMaster's worker tracking
   */
  findIOWorker(service) {
    // This should be overridden by DaemonMaster
    // Return the worker instance for the given service
    return null;
  }

  /**
   * Find any available task worker
   * Override this method to integrate with DaemonMaster's worker tracking
   */
  findAvailableTaskWorker() {
    // This should be overridden by DaemonMaster
    // Return an available task worker
    return null;
  }

  /**
   * Find worker by ID
   * Override this method to integrate with DaemonMaster's worker tracking
   */
  findWorker(workerId) {
    // This should be overridden by DaemonMaster
    return null;
  }

  /**
   * Generate unique request ID
   */
  generateRequestId() {
    return `req-${Date.now()}-${++this.requestId}`;
  }

  /**
   * Get pending requests count
   */
  getPendingCount() {
    return this.pendingRequests.size;
  }

  /**
   * Get message bus statistics
   */
  getStats() {
    return {
      pendingRequests: this.pendingRequests.size,
      requestTimeout: this.requestTimeout
    };
  }
}

module.exports = MessageBus;
