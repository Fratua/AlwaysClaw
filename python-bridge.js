/**
 * OpenClawAgent - Python Bridge (Node.js side)
 * Spawns a persistent Python child process and communicates via JSON-RPC over stdin/stdout.
 */

const { spawn } = require('child_process');
const path = require('path');
const cluster = require('cluster');
const EventEmitter = require('events');
const logger = require('./logger');

/**
 * PythonBridge - Master process bridge to the Python JSON-RPC server.
 * Spawns python_bridge.py, sends requests via stdin, reads responses from stdout.
 */
class PythonBridge extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      pythonExecutable: process.env.PYTHON_EXECUTABLE || 'python',
      scriptPath: path.join(__dirname, 'python_bridge.py'),
      restartDelay: 2000,
      maxRestartAttempts: 10,
      requestTimeout: 30000,
      maxPendingRequests: 100,
      ...options,
    };

    this.process = null;
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.isRunning = false;
    this.isReady = false;
    this.restartAttempts = 0;
    this.restartTimer = null;
    this.lineBuffer = '';
  }

  async start() {
    if (this.isRunning) return;

    logger.info('[PythonBridge] Starting Python bridge...');

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Python bridge startup timed out'));
      }, 30000);

      this._spawn();

      // Wait for ready signal
      const onReady = () => {
        clearTimeout(timeout);
        resolve();
      };
      this.once('ready', onReady);

      this.process.once('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });
  }

  _spawn() {
    this.process = spawn(this.options.pythonExecutable, [
      '-u', // unbuffered stdout
      this.options.scriptPath,
    ], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
    });

    this.isRunning = true;
    this.lineBuffer = '';

    // Read stdout (JSON-RPC responses)
    this.process.stdout.on('data', (data) => {
      this.lineBuffer += data.toString();
      const lines = this.lineBuffer.split('\n');
      // Keep the last incomplete line in the buffer
      this.lineBuffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const message = JSON.parse(line);
          this._handleMessage(message);
        } catch (e) {
          logger.warn('[PythonBridge] Failed to parse stdout line:', line.substring(0, 200));
        }
      }
    });

    // Read stderr (Python logs)
    this.process.stderr.on('data', (data) => {
      const text = data.toString().trim();
      if (text) {
        logger.debug(`[PythonBridge:stderr] ${text}`);
      }
    });

    this.process.on('exit', (code, signal) => {
      logger.warn(`[PythonBridge] Python process exited (code=${code}, signal=${signal})`);
      this.isRunning = false;
      this.isReady = false;

      // Reject all pending requests
      for (const [id, req] of this.pendingRequests) {
        clearTimeout(req.timeout);
        req.reject(new Error('Python bridge process died'));
      }
      this.pendingRequests.clear();

      // Auto-restart if not intentionally stopped
      if (code !== 0 && !this._stopping) {
        this._scheduleRestart();
      }
    });

    this.process.on('error', (err) => {
      logger.error('[PythonBridge] Process spawn error:', err);
      this.emit('error', err);
    });
  }

  _handleMessage(message) {
    // Check for ready signal
    if (message.method === 'ready') {
      this.isReady = true;
      this.restartAttempts = 0;
      logger.info(`[PythonBridge] Bridge ready. Handlers: ${(message.params?.handlers || []).length}`);
      this.emit('ready', message.params);
      return;
    }

    // Handle JSON-RPC responses
    const id = message.id;
    if (id !== undefined && id !== null) {
      const pending = this.pendingRequests.get(id);
      if (pending) {
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(id);

        if (message.error) {
          pending.reject(new Error(message.error.message || 'Bridge call failed'));
        } else {
          pending.resolve(message.result);
        }
      }
    }
  }

  /**
   * Call a Python bridge method.
   * @param {string} method - Method name (e.g., 'llm.complete', 'memory.store')
   * @param {object} params - Method parameters
   * @returns {Promise<any>} Result from Python handler
   */
  async call(method, params = {}) {
    if (!this.isRunning || !this.process) {
      throw new Error('Python bridge not running');
    }

    if (this.pendingRequests.size >= this.options.maxPendingRequests) {
      throw new Error('Bridge overloaded');
    }

    const id = ++this.requestId;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Bridge call '${method}' timed out after ${this.options.requestTimeout}ms`));
      }, this.options.requestTimeout);

      this.pendingRequests.set(id, { resolve, reject, timeout, method });

      const request = JSON.stringify({
        jsonrpc: '2.0',
        id,
        method,
        params,
      });

      try {
        this.process.stdin.write(request + '\n');
      } catch (e) {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        reject(new Error(`Failed to write to bridge stdin: ${e.message}`));
      }
    });
  }

  _scheduleRestart() {
    this.restartAttempts++;
    if (this.restartAttempts > this.options.maxRestartAttempts) {
      logger.error('[PythonBridge] Max restart attempts exceeded. Giving up.');
      this.emit('fatal', new Error('Max restart attempts exceeded'));
      return;
    }

    // Exponential backoff
    const delay = Math.min(
      this.options.restartDelay * Math.pow(2, this.restartAttempts - 1),
      30000
    );
    logger.info(`[PythonBridge] Restarting in ${delay}ms (attempt ${this.restartAttempts}/${this.options.maxRestartAttempts})`);

    this.restartTimer = setTimeout(() => {
      this._spawn();
    }, delay);
  }

  async stop() {
    this._stopping = true;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
    }

    if (this.process && this.isRunning) {
      logger.info('[PythonBridge] Stopping Python bridge...');

      // Reject pending requests
      for (const [id, req] of this.pendingRequests) {
        clearTimeout(req.timeout);
        req.reject(new Error('Bridge stopping'));
      }
      this.pendingRequests.clear();

      // Try graceful shutdown first
      try {
        this.process.stdin.end();
      } catch (e) {
        // stdin may already be closed
      }

      // Wait a moment, then force kill
      await new Promise((resolve) => {
        const killTimer = setTimeout(() => {
          if (this.process && !this.process.killed) {
            this.process.kill('SIGKILL');
          }
          resolve();
        }, 5000);

        this.process.once('exit', () => {
          clearTimeout(killTimer);
          resolve();
        });
      });
    }

    this.isRunning = false;
    this.isReady = false;
    logger.info('[PythonBridge] Bridge stopped');
  }

  /**
   * Health check - ping the bridge.
   */
  async healthCheck() {
    const start = Date.now();
    try {
      const result = await this.call('health', {});
      return {
        healthy: true,
        latencyMs: Date.now() - start,
        ...result,
      };
    } catch (e) {
      return {
        healthy: false,
        latencyMs: Date.now() - start,
        error: e.message,
      };
    }
  }
}


/**
 * PythonBridgeWorkerProxy - Worker process proxy.
 * Sends bridge requests to master via IPC, master forwards to PythonBridge.
 */
class PythonBridgeWorkerProxy {
  constructor() {
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.requestTimeout = 30000;

    // Listen for responses from master
    process.on('message', (msg) => {
      if (msg && msg.type === 'python-bridge-response') {
        this._handleResponse(msg.data);
      }
    });
  }

  async start() {
    // No-op for worker proxy (master owns the bridge)
  }

  async stop() {
    // Reject pending requests
    for (const [id, req] of this.pendingRequests) {
      clearTimeout(req.timeout);
      req.reject(new Error('Worker shutting down'));
    }
    this.pendingRequests.clear();
  }

  async call(method, params = {}) {
    const id = ++this.requestId;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Bridge proxy call '${method}' timed out`));
      }, this.requestTimeout);

      this.pendingRequests.set(id, { resolve, reject, timeout });

      if (process.send) {
        process.send({
          type: 'python-bridge-request',
          data: { id, method, params },
        });
      } else {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        reject(new Error('No IPC channel to master process'));
      }
    });
  }

  _handleResponse(data) {
    const pending = this.pendingRequests.get(data.id);
    if (pending) {
      clearTimeout(pending.timeout);
      this.pendingRequests.delete(data.id);

      if (data.error) {
        pending.reject(new Error(data.error));
      } else {
        pending.resolve(data.result);
      }
    }
  }

  async healthCheck() {
    return this.call('health', {});
  }
}


// Singleton instances
let _bridgeInstance = null;

/**
 * Get the bridge instance. Returns PythonBridge in master, PythonBridgeWorkerProxy in workers.
 */
function getBridge() {
  if (_bridgeInstance) return _bridgeInstance;

  if (cluster.isMaster || cluster.isPrimary) {
    _bridgeInstance = new PythonBridge();
  } else {
    _bridgeInstance = new PythonBridgeWorkerProxy();
  }

  return _bridgeInstance;
}


module.exports = {
  PythonBridge,
  PythonBridgeWorkerProxy,
  getBridge,
};
