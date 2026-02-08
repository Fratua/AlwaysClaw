/**
 * OpenClawAgent - Cluster Manager
 * Manages Node.js worker process clustering
 */

const cluster = require('cluster');
const os = require('os');
const logger = require('./logger');

class ClusterManager {
  constructor(options = {}) {
    this.options = {
      workers: options.workers || os.cpus().length,
      schedulingPolicy: options.schedulingPolicy || cluster.SCHED_NONE,
      workerScript: options.workerScript || null,
      workerArgs: options.workerArgs || [],
      restartDelay: options.restartDelay || 5000,
      maxRestarts: options.maxRestarts || 10,
      restartWindow: options.restartWindow || 60000,
      ...options
    };
    
    this.workers = new Map();
    this.restartCount = 0;
    this.lastRestartTime = Date.now();
    this.isShuttingDown = false;
    
    // Set scheduling policy
    cluster.schedulingPolicy = this.options.schedulingPolicy;
  }

  setupMaster() {
    if (!cluster.isMaster) {
      throw new Error('setupMaster can only be called from master process');
    }
    
    logger.info('[ClusterManager] Setting up master process...');
    
    cluster.setupMaster({
      exec: this.options.workerScript,
      args: this.options.workerArgs,
      silent: false
    });
    
    // Setup cluster event handlers
    this.setupClusterHandlers();
  }

  setupClusterHandlers() {
    cluster.on('fork', (worker) => {
      logger.info(`[ClusterManager] Worker ${worker.id} forked (PID: ${worker.process.pid})`);
      this.workers.set(worker.id, {
        worker,
        forkTime: Date.now(),
        restarts: 0
      });
    });

    cluster.on('online', (worker) => {
      logger.info(`[ClusterManager] Worker ${worker.id} is online`);
      const workerInfo = this.workers.get(worker.id);
      if (workerInfo) {
        workerInfo.onlineTime = Date.now();
        workerInfo.state = 'online';
      }
    });

    cluster.on('listening', (worker, address) => {
      logger.info(`[ClusterManager] Worker ${worker.id} listening on ${address.address}:${address.port}`);
      const workerInfo = this.workers.get(worker.id);
      if (workerInfo) {
        workerInfo.listeningTime = Date.now();
        workerInfo.state = 'listening';
        workerInfo.address = address;
      }
    });

    cluster.on('disconnect', (worker) => {
      logger.info(`[ClusterManager] Worker ${worker.id} disconnected`);
      const workerInfo = this.workers.get(worker.id);
      if (workerInfo) {
        workerInfo.state = 'disconnected';
      }
    });

    cluster.on('exit', (worker, code, signal) => {
      this.handleWorkerExit(worker, code, signal);
    });

    cluster.on('message', (worker, message, handle) => {
      this.handleWorkerMessage(worker, message, handle);
    });
  }

  handleWorkerExit(worker, code, signal) {
    logger.info(`[ClusterManager] Worker ${worker.id} exited (code: ${code}, signal: ${signal})`);
    
    const workerInfo = this.workers.get(worker.id);
    
    // Remove from tracking
    this.workers.delete(worker.id);
    
    // Check if we should restart
    if (!this.isShuttingDown && code !== 0 && !worker.exitedAfterDisconnect) {
      // Check restart rate
      if (this.checkRestartRate()) {
        const index = workerInfo ? workerInfo.index : this.workers.size;
        logger.info(`[ClusterManager] Restarting worker ${worker.id} (index: ${index})...`);
        
        setTimeout(() => {
          this.forkWorker(index);
        }, this.options.restartDelay);
      } else {
        logger.error('[ClusterManager] Maximum restart rate exceeded');
        this.emit('max-restarts-exceeded');
      }
    }
  }

  handleWorkerMessage(worker, message, handle) {
    // Handle messages from workers
    // Can be extended for custom message handling
    logger.debug(`[ClusterManager] Message from worker ${worker.id}:`, message.type || message);
  }

  checkRestartRate() {
    const now = Date.now();
    
    if (now - this.lastRestartTime > this.options.restartWindow) {
      this.restartCount = 0;
      this.lastRestartTime = now;
    }
    
    this.restartCount++;
    
    return this.restartCount <= this.options.maxRestarts;
  }

  forkWorkers() {
    if (!cluster.isMaster) {
      throw new Error('forkWorkers can only be called from master process');
    }
    
    logger.info(`[ClusterManager] Forking ${this.options.workers} workers...`);
    
    for (let i = 0; i < this.options.workers; i++) {
      this.forkWorker(i);
    }
  }

  forkWorker(index) {
    const workerEnv = {
      WORKER_INDEX: index.toString(),
      WORKER_COUNT: this.options.workers.toString()
    };

    const worker = cluster.fork(workerEnv);
    
    // Store worker info
    const workerInfo = this.workers.get(worker.id);
    if (workerInfo) {
      workerInfo.index = index;
    }
    
    return worker;
  }

  reload() {
    if (!cluster.isMaster) {
      throw new Error('reload can only be called from master process');
    }
    
    logger.info('[ClusterManager] Starting zero-downtime reload...');
    
    const workerIds = Array.from(this.workers.keys());
    let index = 0;
    
    const reloadNext = () => {
      if (index >= workerIds.length) {
        logger.info('[ClusterManager] Reload complete');
        this.emit('reload-complete');
        return;
      }
      
      const oldWorkerId = workerIds[index++];
      const oldWorkerInfo = this.workers.get(oldWorkerId);
      
      if (!oldWorkerInfo) {
        reloadNext();
        return;
      }
      
      const workerIndex = oldWorkerInfo.index;
      
      // Fork new worker
      const newWorker = this.forkWorker(workerIndex);
      
      newWorker.once('listening', () => {
        logger.info(`[ClusterManager] New worker ${newWorker.id} ready, disconnecting old worker ${oldWorkerId}`);
        
        // Disconnect old worker
        oldWorkerInfo.worker.disconnect();
        
        // Wait for disconnection
        setTimeout(() => {
          if (!oldWorkerInfo.worker.isDead()) {
            oldWorkerInfo.worker.kill('SIGTERM');
          }
          reloadNext();
        }, 1000);
      });
    };
    
    reloadNext();
  }

  shutdown() {
    if (!cluster.isMaster) {
      throw new Error('shutdown can only be called from master process');
    }
    
    logger.info('[ClusterManager] Shutting down all workers...');
    this.isShuttingDown = true;
    
    for (const [id, workerInfo] of this.workers) {
      logger.info(`[ClusterManager] Sending shutdown to worker ${id}`);
      workerInfo.worker.send({ type: 'shutdown' });
    }
    
    // Force kill after timeout
    setTimeout(() => {
      for (const [id, workerInfo] of this.workers) {
        if (!workerInfo.worker.isDead()) {
          logger.warn(`[ClusterManager] Force killing worker ${id}`);
          workerInfo.worker.kill('SIGKILL');
        }
      }
    }, 30000);
  }

  getStats() {
    const stats = {
      total: this.workers.size,
      online: 0,
      listening: 0,
      disconnected: 0,
      workers: []
    };
    
    for (const [id, workerInfo] of this.workers) {
      stats.workers.push({
        id,
        pid: workerInfo.worker.process.pid,
        state: workerInfo.state,
        uptime: workerInfo.onlineTime ? Date.now() - workerInfo.onlineTime : 0
      });
      
      if (workerInfo.state === 'online') stats.online++;
      if (workerInfo.state === 'listening') stats.listening++;
      if (workerInfo.state === 'disconnected') stats.disconnected++;
    }
    
    return stats;
  }

  broadcast(message) {
    for (const [id, workerInfo] of this.workers) {
      workerInfo.worker.send(message);
    }
  }

  sendToWorker(workerId, message) {
    const workerInfo = this.workers.get(workerId);
    if (workerInfo) {
      workerInfo.worker.send(message);
      return true;
    }
    return false;
  }

  // Event emitter functionality
  emit(event, ...args) {
    // Simple event emitter implementation
    if (this._events && this._events[event]) {
      this._events[event].forEach(handler => handler(...args));
    }
  }

  on(event, handler) {
    if (!this._events) this._events = {};
    if (!this._events[event]) this._events[event] = [];
    this._events[event].push(handler);
    return this;
  }
}

module.exports = ClusterManager;
