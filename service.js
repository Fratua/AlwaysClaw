/**
 * OpenClawAgent - Main Service Entry Point
 * Windows service entry point with clustering support
 */

const cluster = require('cluster');
const os = require('os');
const DaemonMaster = require('./daemon-master');
const logger = require('./logger');

// Handle Windows service events
if (process.platform === 'win32') {
  // Handle service stop events from node-windows
  process.on('message', (msg) => {
    if (msg === 'shutdown') {
      logger.info('Received shutdown signal from service manager');
      process.exit(0);
    }
  });
}

// Main execution
async function main() {
  logger.info('════════════════════════════════════════════════════════════');
  logger.info('  OpenClawAgent Service Starting...');
  logger.info('  Version: 1.0.0');
  logger.info('  Platform: ' + process.platform);
  logger.info('  Node.js: ' + process.version);
  logger.info('  CPUs: ' + os.cpus().length);
  logger.info('════════════════════════════════════════════════════════════');
  
  if (cluster.isMaster) {
    // Master process - manage workers
    const daemon = new DaemonMaster({
      workerCount: os.cpus().length,
      agentLoops: 15,
      heartbeatInterval: 30000,
      restartDelay: 5000,
      maxRestarts: 10,
      restartWindow: 60000
    });
    
    // Handle daemon events
    daemon.on('ready', () => {
      logger.info('════════════════════════════════════════════════════════════');
      logger.info('  OpenClawAgent Service Ready');
      logger.info('  Status: Running');
      logger.info('════════════════════════════════════════════════════════════');
    });
    
    daemon.on('error', (error) => {
      logger.error('Daemon error:', error);
    });
    
    daemon.on('shutdown', (reason) => {
      logger.info(`Daemon shutdown initiated: ${reason}`);
    });
    
    try {
      await daemon.initialize();
    } catch (error) {
      logger.error('Failed to start service:', error);
      process.exit(1);
    }
  } else {
    // Worker process
    const workerType = process.env.WORKER_TYPE;
    
    logger.info(`[Worker] Starting ${workerType} worker (${process.env.WORKER_ID})`);
    
    try {
      switch(workerType) {
        case 'agent-loop':
          const AgentLoopWorker = require('./agent-loop-worker');
          const agentWorker = new AgentLoopWorker();
          await agentWorker.initialize();
          break;
          
        case 'io':
          const IOWorker = require('./io-worker');
          const ioWorker = new IOWorker();
          await ioWorker.initialize();
          break;
          
        case 'task':
          const TaskWorker = require('./task-worker');
          const taskWorker = new TaskWorker();
          await taskWorker.initialize();
          break;
          
        default:
          logger.error(`Unknown worker type: ${workerType}`);
          process.exit(1);
      }
    } catch (error) {
      logger.error(`Worker initialization failed:`, error);
      process.exit(1);
    }
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection at:', promise, 'reason:', reason);
});

// Start the service
main().catch(error => {
  logger.error('Service failed to start:', error);
  process.exit(1);
});
