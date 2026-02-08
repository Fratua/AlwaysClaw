/**
 * OpenClawAgent - HTTP Control Plane Server
 * Provides REST API for health checks, worker management, and control operations
 */

const http = require('http');
const url = require('url');
const logger = require('./logger');

class ControlServer {
  constructor(daemonMaster, options = {}) {
    this.daemon = daemonMaster;
    this.options = {
      port: options.port || process.env.CONTROL_PORT || 8080,
      host: options.host || process.env.CONTROL_HOST || '127.0.0.1',
      ...options
    };
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
    
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    const parsedUrl = url.parse(req.url, true);
    const path = parsedUrl.pathname;
    
    logger.debug(`[ControlServer] ${req.method} ${path}`);

    try {
      switch (path) {
        case '/health':
          this.handleHealth(req, res);
          break;
        case '/workers':
          this.handleWorkers(req, res);
          break;
        case '/metrics':
          this.handleMetrics(req, res);
          break;
        case '/status':
          this.handleStatus(req, res);
          break;
        case '/control/restart-worker':
          if (req.method === 'POST') {
            this.handleRestartWorker(req, res);
          } else {
            this.sendError(res, 405, 'Method not allowed');
          }
          break;
        case '/control/restart-all':
          if (req.method === 'POST') {
            this.handleRestartAll(req, res);
          } else {
            this.sendError(res, 405, 'Method not allowed');
          }
          break;
        case '/control/shutdown':
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

  handleHealth(req, res) {
    const health = {
      status: 'healthy',
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

  handleRestartWorker(req, res) {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
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
      } catch (error) {
        this.sendError(res, 400, 'Invalid JSON body');
      }
    });
  }

  handleRestartAll(req, res) {
    // Gracefully restart all workers
    logger.info('[ControlServer] Restarting all workers...');
    
    // Restart agent loop workers
    for (const [id, info] of this.daemon?.agentLoopWorkers || []) {
      this.daemon?.restartWorker(id);
    }
    
    // Restart IO workers
    for (const [service, info] of this.daemon?.ioWorkers || []) {
      if (info.worker) {
        this.daemon?.restartWorker(info.worker.id);
      }
    }
    
    // Restart task workers
    for (const [id, info] of this.daemon?.taskWorkers || []) {
      this.daemon?.restartWorker(id);
    }
    
    this.sendJson(res, 200, { success: true, message: 'Restart signals sent to all workers' });
  }

  handleShutdown(req, res) {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const { reason = 'api-request' } = data;
        
        this.sendJson(res, 200, { success: true, message: 'Shutdown initiated' });
        
        // Initiate shutdown after sending response
        setTimeout(() => {
          this.daemon?.shutdown(reason);
        }, 100);
      } catch (error) {
        this.sendError(res, 400, 'Invalid JSON body');
      }
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
