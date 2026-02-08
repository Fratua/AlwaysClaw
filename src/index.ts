/**
 * OpenClaw AI Agent Scheduler - Main Entry Point
 * 24/7 Scheduled Task System for Windows 10
 */

import { SchedulerSystem } from './scheduler-system';
import { registerAgentLoops } from './agent-loops';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * Default configuration
 */
const DEFAULT_CONFIG = {
  dataDir: process.env.SCHEDULER_DATA_DIR || './data',
  timezone: process.env.SCHEDULER_TIMEZONE || 'America/New_York',
  heartbeat: {
    interval: parseInt(process.env.HEARTBEAT_INTERVAL || '30000'),
    timeout: parseInt(process.env.HEARTBEAT_TIMEOUT || '60000'),
    nodeId: process.env.HEARTBEAT_NODE_ID || 'openclaw-agent',
  },
  overlapPrevention: {
    strategy: (process.env.OVERLAP_STRATEGY as any) || 'skip',
    timeoutMs: parseInt(process.env.OVERLAP_TIMEOUT || '300000'),
    maxConcurrent: parseInt(process.env.OVERLAP_MAX_CONCURRENT || '1'),
  },
  retry: {
    maxAttempts: parseInt(process.env.RETRY_MAX_ATTEMPTS || '3'),
    strategy: (process.env.RETRY_STRATEGY as any) || 'exponential',
    baseDelay: parseInt(process.env.RETRY_BASE_DELAY || '1000'),
    maxDelay: parseInt(process.env.RETRY_MAX_DELAY || '30000'),
  },
};

/**
 * Main application class
 */
class OpenClawScheduler {
  private scheduler: SchedulerSystem;

  constructor() {
    this.scheduler = new SchedulerSystem(DEFAULT_CONFIG);
  }

  /**
   * Initialize and start the scheduler
   */
  async start(): Promise<void> {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║         OpenClaw AI Agent Scheduler v1.0.0                 ║');
    console.log('║         24/7 Scheduled Task System                         ║');
    console.log('╚════════════════════════════════════════════════════════════╝');
    console.log();

    // Initialize the system
    await this.scheduler.initialize();

    // Register all 15 agent loops
    registerAgentLoops(this.scheduler);

    // Start the system
    await this.scheduler.start();

    // Log startup info
    console.log();
    console.log('System Configuration:');
    console.log('  Data Directory:', DEFAULT_CONFIG.dataDir);
    console.log('  Timezone:', DEFAULT_CONFIG.timezone);
    console.log('  Heartbeat Interval:', DEFAULT_CONFIG.heartbeat.interval, 'ms');
    console.log('  Overlap Strategy:', DEFAULT_CONFIG.overlapPrevention.strategy);
    console.log('  Retry Strategy:', DEFAULT_CONFIG.retry.strategy);
    console.log();
    console.log('Press Ctrl+C to stop the scheduler');
    console.log();
  }

  /**
   * Stop the scheduler gracefully
   */
  async stop(): Promise<void> {
    console.log('\nStopping OpenClaw Scheduler...');
    await this.scheduler.stop();
    console.log('OpenClaw Scheduler stopped');
  }

  /**
   * Get system health
   */
  getHealth() {
    return this.scheduler.getHealth();
  }

  /**
   * Get the scheduler instance
   */
  getScheduler(): SchedulerSystem {
    return this.scheduler;
  }
}

// Create global instance
const app = new OpenClawScheduler();

// Export for module usage
export { app, OpenClawScheduler };

// Start if running directly
if (require.main === module) {
  app.start().catch(error => {
    console.error('Failed to start scheduler:', error);
    process.exit(1);
  });
}
