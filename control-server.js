/**
 * OpenClawAgent - HTTP Control Plane Server
 * Provides REST API for health checks, worker management, and control operations
 */

const http = require('http');
const crypto = require('crypto');
const logger = require('./logger');

class ControlServer {
  constructor(daemonMaster, options = {}) {
    this.daemon = daemonMaster;
    const rawPort = parseInt(options.port || process.env.CONTROL_PORT || '8080', 10);
    if (!Number.isFinite(rawPort) || rawPort < 1 || rawPort > 65535) {
      throw new RangeError(`Invalid control server port: ${options.port || process.env.CONTROL_PORT}. Must be 1-65535.`);
    }
    const port = rawPort;
    this.options = {
      port,
      host: options.host || process.env.CONTROL_HOST || '127.0.0.1',
      ...options
    };
    this.options.port = port; // ensure spread didn't override validated port
    this.server = null;
    this.metrics = {
      requests: 0,
      errors: 0,
      startTime: Date.now()
    };
  }

  start() {
    return new Promise((resolve, reject) => {
      this.server = http.createServer((req, res) => this.handleRequest(req, res));
      
      this.server.listen(this.options.port, this.options.host, () => {
        logger.info(`[ControlServer] HTTP server listening on ${this.options.host}:${this.options.port}`);
        resolve();
      });
      
      this.server.on('error', (error) => {
        logger.error('[ControlServer] Server error:', error);
        reject(error);
      });
    });
  }

  stop() {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          logger.info('[ControlServer] HTTP server stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  handleRequest(req, res) {
    this.metrics.requests++;
    
    // Set CORS headers - only set origin if explicitly configured
    const corsOrigin = process.env.CONTROL_CORS_ORIGIN;
    if (corsOrigin) {
      res.setHeader('Access-Control-Allow-Origin', corsOrigin);
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-API-Key');
    
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost:' + this.options.port}`);
    const path = parsedUrl.pathname;
    
    logger.debug(`[ControlServer] ${req.method} ${path}`);

    try {
      switch (path) {
        case '/health':
          if (!this.authenticateRequest(req, res)) return;
          this.handleHealth(req, res);
          break;
        case '/workers':
          if (!this.authenticateRequest(req, res)) return;
          this.handleWorkers(req, res);
          break;
        case '/metrics':
          if (!this.authenticateRequest(req, res)) return;
          this.handleMetrics(req, res);
          break;
        case '/status':
          if (!this.authenticateRequest(req, res)) return;
          this.handleStatus(req, res);
          break;
        case '/control/restart-worker':
          if (!this.authenticateRequest(req, res)) return;
          if (req.method === 'POST') {
            this.handleRestartWorker(req, res);
          } else {
            this.sendError(res, 405, 'Method not allowed');
          }
          break;
        case '/control/restart-all':
          if (!this.authenticateRequest(req, res)) return;
          if (req.method === 'POST') {
            this.handleRestartAll(req, res);
          } else {
            this.sendError(res, 405, 'Method not allowed');
          }
          break;
        case '/control/shutdown':
          if (!this.authenticateRequest(req, res)) return;
          if (req.method === 'POST') {
            this.handleShutdown(req, res);
          } else {
            this.sendError(res, 405, 'Method not allowed');
          }
          break;
        default:
          this.sendError(res, 404, 'Not found');
      }
    } catch (error) {
      this.metrics.errors++;
      logger.error('[ControlServer] Request handler error:', error);
      this.sendError(res, 500, 'Internal server error');
    }
  }

  authenticateRequest(req, res) {
    const apiKey = process.env.CONTROL_API_KEY;
    if (!apiKey) {
      // No API key configured - only allow localhost
      const remoteAddr = req.socket.remoteAddress;
      if (remoteAddr === '127.0.0.1' || remoteAddr === '::1' || remoteAddr === '::ffff:127.0.0.1') {
        return true;
      }
      this.sendError(res, 403, 'Control endpoints only accessible from localhost');
      return false;
    }

    // Check API key with timing-safe comparison
    const providedKey = req.headers['x-api-key'] || new URL(req.url, `http://${req.headers.host}`).searchParams.get('api_key');
    if (!providedKey) {
      this.sendError(res, 401, 'Invalid or missing API key');
      return false;
    }
    const apiKeyBuf = Buffer.from(apiKey);
    const providedBuf = Buffer.from(providedKey);
    if (apiKeyBuf.length !== providedBuf.length || !crypto.timingSafeEqual(apiKeyBuf, providedBuf)) {
      this.sendError(res, 401, 'Invalid or missing API key');
      return false;
    }
    return true;
  }

  handleHealth(req, res) {
    const isHealthy = this.daemon ? this.daemon.state === 'running' : false;
    const healthMonitor = this.daemon?.healthMonitor;
    const monitorHealthy = healthMonitor ? healthMonitor.isHealthy() : true;

    const health = {
      status: isHealthy && monitorHealthy ? 'healthy' : 'degraded',
      timestamp: Date.now(),
      uptime: process.uptime(),
      pid: process.pid,
      memory: process.memoryUsage(),
      daemon: {
        state: this.daemon ? this.daemon.state : 'unknown',
        workerCount: this.getTotalWorkerCount()
      }
    };

    this.sendJson(res, 200, health);
  }

  handleWorkers(req, res) {
    const workers = {
      agentLoops: Array.from(this.daemon?.agentLoopWorkers?.entries() || []).map(([id, info]) => ({
        id,
        type: info.type,
        index: info.index,
        state: info.state,
        startTime: info.startTime,
        pid: info.worker?.process?.pid
      })),
      ioWorkers: Array.from(this.daemon?.ioWorkers?.entries() || []).map(([service, info]) => ({
        service,
        type: info.type,
        state: info.state,
        startTime: info.startTime,
        pid: info.worker?.process?.pid
      })),
      taskWorkers: Array.from(this.daemon?.taskWorkers?.entries() || []).map(([id, info]) => ({
        id,
        type: info.type,
        index: info.index,
        state: info.state,
        startTime: info.startTime,
        pid: info.worker?.process?.pid
      }))
    };
    
    this.sendJson(res, 200, workers);
  }

  handleMetrics(req, res) {
    const metrics = {
      timestamp: Date.now(),
      server: {
        requests: this.metrics.requests,
        errors: this.metrics.errors,
        uptime: Date.now() - this.metrics.startTime
      },
      process: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage()
      },
      workers: {
        total: this.getTotalWorkerCount(),
        agentLoops: this.daemon?.agentLoopWorkers?.size || 0,
        ioWorkers: this.daemon?.ioWorkers?.size || 0,
        taskWorkers: this.daemon?.taskWorkers?.size || 0
      }
    };
    
    this.sendJson(res, 200, metrics);
  }

  handleStatus(req, res) {
    const status = {
      service: 'OpenClawAgent',
      version: '1.0.0',
      timestamp: Date.now(),
      daemon: {
        state: this.daemon?.state || 'unknown',
        isShuttingDown: this.daemon?.isShuttingDown || false
      },
      workers: {
        total: this.getTotalWorkerCount(),
        breakdown: {
          agentLoops: this.daemon?.agentLoopWorkers?.size || 0,
          ioWorkers: this.daemon?.ioWorkers?.size || 0,
          taskWorkers: this.daemon?.taskWorkers?.size || 0
        }
      },
      server: {
        uptime: Date.now() - this.metrics.startTime,
        requests: this.metrics.requests,
        port: this.options.port
      }
    };
    
    this.sendJson(res, 200, status);
  }

  /**
   * Read and parse a JSON body from an HTTP request with size limits and error handling.
   * Calls callback(data) on success; sends appropriate error responses otherwise.
   */
  readJsonBody(req, res, maxSize, callback) {
    if (typeof maxSize === 'function') {
      callback = maxSize;
      maxSize = 1024 * 1024; // default 1MB
    }
    let body = '';
    let bodySize = 0;
    let responded = false;

    const sendOnce = (code, msg) => {
      if (!responded) {
        responded = true;
        this.sendError(res, code, msg);
      }
    };

    req.on('data', chunk => {
      bodySize += chunk.length;
      if (bodySize > maxSize) {
        sendOnce(413, 'Request body too large');
        req.destroy();
        return;
      }
      body += chunk;
    });
    req.on('end', () => {
      if (responded) return;
      if (!body) {
        sendOnce(400, 'Empty request body');
        return;
      }
      try {
        const data = JSON.parse(body);
        callback(data);
      } catch (error) {
        sendOnce(400, 'Invalid JSON body');
      }
    });
  }

  handleRestartWorker(req, res) {
    this.readJsonBody(req, res, (data) => {
      const { workerId, workerType, index } = data;

      if (workerId) {
        this.daemon?.restartWorker(workerId);
        this.sendJson(res, 200, { success: true, message: `Restart signal sent to worker ${workerId}` });
      } else if (workerType === 'agent-loop' && typeof index === 'number') {
        // Find worker by type and index
        for (const [id, info] of this.daemon?.agentLoopWorkers || []) {
          if (info.index === index) {
            this.daemon?.restartWorker(id);
            this.sendJson(res, 200, { success: true, message: `Restarted agent-loop worker ${index}` });
            return;
          }
        }
        this.sendError(res, 404, 'Worker not found');
      } else {
        this.sendError(res, 400, 'Missing workerId or workerType+index');
      }
    });
  }

  handleRestartAll(req, res) {
    // Gracefully restart all workers
    logger.info('[ControlServer] Restarting all workers...');

    // Collect worker IDs first to avoid modifying maps during iteration
    const agentLoopIds = Array.from(this.daemon?.agentLoopWorkers?.keys() || []);
    const ioWorkerIds = Array.from(this.daemon?.ioWorkers?.values() || [])
      .map(info => info.worker?.id).filter(Boolean);
    const taskWorkerIds = Array.from(this.daemon?.taskWorkers?.keys() || []);

    for (const id of agentLoopIds) {
      this.daemon?.restartWorker(id);
    }
    for (const id of ioWorkerIds) {
      this.daemon?.restartWorker(id);
    }
    for (const id of taskWorkerIds) {
      this.daemon?.restartWorker(id);
    }

    this.sendJson(res, 200, { success: true, message: 'Restart signals sent to all workers' });
  }

  handleShutdown(req, res) {
    this.readJsonBody(req, res, (data) => {
      const { reason = 'api-request' } = data;

      this.sendJson(res, 200, { success: true, message: 'Shutdown initiated' });

      // Initiate shutdown after sending response
      setTimeout(() => {
        this.daemon?.shutdown(reason);
      }, 100);
    });
  }

  getTotalWorkerCount() {
    if (!this.daemon) return 0;
    return (this.daemon.agentLoopWorkers?.size || 0) +
           (this.daemon.ioWorkers?.size || 0) +
           (this.daemon.taskWorkers?.size || 0);
  }

  sendJson(res, statusCode, data) {
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data, null, 2));
  }

  sendError(res, statusCode, message) {
    this.sendJson(res, statusCode, { error: message, status: statusCode });
  }
}

module.exports = ControlServer;
