/**
 * Scheduler System - Main Integration
 * OpenClaw AI Agent Scheduler
 */

import { CronEngine } from './cron-engine';
import { PersistenceManager } from './persistence';
import { HeartbeatSender, HeartbeatMonitor } from './heartbeat';
import { OverlapPrevention } from './overlap-prevention';
import { RetryMechanism } from './retry-mechanism';

export interface SchedulerSystemConfig {
  dataDir: string;
  timezone?: string;
  heartbeat: {
    interval: number;
    timeout: number;
    nodeId: string;
  };
  overlapPrevention: {
    strategy: 'skip' | 'queue' | 'delay' | 'timeout' | 'concurrent';
    timeoutMs?: number;
    maxConcurrent?: number;
  };
  retry: {
    maxAttempts: number;
    strategy: 'fixed' | 'exponential' | 'linear';
    baseDelay: number;
    maxDelay: number;
  };
}

export class SchedulerSystem {
  public cronEngine: CronEngine;
  public persistence: PersistenceManager;
  public heartbeatSender: HeartbeatSender;
  public heartbeatMonitor: HeartbeatMonitor;
  public overlapPrevention: OverlapPrevention;
  public retryMechanism: RetryMechanism;

  private isRunning = false;
  private isInitialized = false;

  constructor(private config: SchedulerSystemConfig) {
    // Initialize components
    this.cronEngine = new CronEngine();
    this.persistence = new PersistenceManager(config.dataDir);
    this.heartbeatSender = new HeartbeatSender({
      interval: config.heartbeat.interval,
      timeout: config.heartbeat.timeout,
      missedThreshold: 3,
      nodeId: config.heartbeat.nodeId,
      metadata: {
        version: '1.0.0',
        platform: process.platform,
        nodeVersion: process.version,
      },
    });
    this.heartbeatMonitor = new HeartbeatMonitor({
      checkInterval: 30000,
      missedThreshold: 3,
    });
    this.overlapPrevention = new OverlapPrevention({
      strategy: config.overlapPrevention.strategy,
      timeoutMs: config.overlapPrevention.timeoutMs || 300000,
      maxConcurrent: config.overlapPrevention.maxConcurrent || 1,
    });
    this.retryMechanism = new RetryMechanism({
      maxAttempts: config.retry.maxAttempts,
      strategy: config.retry.strategy,
      baseDelay: config.retry.baseDelay,
      maxDelay: config.retry.maxDelay,
    });

    // Setup event handlers
    this.setupEventHandlers();
  }

  /**
   * Initialize the system
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('[SchedulerSystem] Already initialized');
      return;
    }

    // Initialize persistence
    await this.persistence.initialize();

    // Load persisted jobs
    const persistedJobs = await this.persistence.loadJobs();
    console.log(`[SchedulerSystem] Loaded ${persistedJobs.length} persisted jobs`);

    for (const job of persistedJobs) {
      try {
        const options = JSON.parse(job.options);
        // Note: Handler needs to be provided - this is just restoration of metadata
        console.log(`[SchedulerSystem] Restored job metadata: ${job.name} (${job.id})`);
      } catch (error) {
        console.error(`[SchedulerSystem] Failed to restore job ${job.id}:`, error);
      }
    }

    this.isInitialized = true;
    console.log('[SchedulerSystem] Initialized successfully');
  }

  /**
   * Start the system
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('[SchedulerSystem] Already running');
      return;
    }

    if (!this.isInitialized) {
      await this.initialize();
    }

    this.isRunning = true;

    // Start cron engine
    this.cronEngine.startAll();

    // Start heartbeat
    this.heartbeatSender.start();
    this.heartbeatMonitor.start();

    console.log('[SchedulerSystem] Started successfully');
    console.log('[SchedulerSystem] Registered jobs:', this.cronEngine.getAllJobs().length);
  }

  /**
   * Stop the system gracefully
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      console.log('[SchedulerSystem] Not running');
      return;
    }

    this.isRunning = false;

    // Stop cron engine
    this.cronEngine.stopAll();

    // Stop heartbeat
    this.heartbeatSender.stop();
    this.heartbeatMonitor.stop();

    // Close persistence
    await this.persistence.close();

    console.log('[SchedulerSystem] Stopped successfully');
  }

  /**
   * Register an agent loop
   */
  registerAgentLoop(
    id: string,
    name: string,
    schedule: string,
    handler: () => Promise<void>,
    options: {
      preventOverlap?: boolean;
      retryAttempts?: number;
      timeout?: number;
      tags?: string[];
    } = {}
  ): void {
    const wrappedHandler = async () => {
      // Apply overlap prevention
      await this.overlapPrevention.execute(id, async () => {
        // Apply retry mechanism
        const result = await this.retryMechanism.execute(handler);
        
        if (!result.success) {
          throw result.error;
        }
      });
    };

    this.cronEngine.register(id, name, schedule, wrappedHandler, {
      preventOverlap: options.preventOverlap ?? true,
      retryAttempts: options.retryAttempts ?? 3,
      timeout: options.timeout ?? 300000,
      tags: ['agent-loop', ...(options.tags || [])],
    });

    // Persist job
    this.persistence.saveJob({
      id,
      name,
      schedule,
      options: JSON.stringify(options),
      status: JSON.stringify({ state: 'idle' }),
      metrics: JSON.stringify({}),
      lastUpdated: Date.now(),
      createdAt: Date.now(),
    }).catch(error => {
      console.error('[SchedulerSystem] Failed to persist job:', error);
    });
  }

  /**
   * Unregister an agent loop
   */
  async unregisterAgentLoop(id: string): Promise<void> {
    this.cronEngine.unregister(id);
    await this.persistence.deleteJob(id);
  }

  /**
   * Get system health
   */
  getHealth(): {
    isRunning: boolean;
    cronHealth: ReturnType<CronEngine['getHealth']>;
    heartbeatActive: boolean;
    nodeHealth: ReturnType<HeartbeatMonitor['getOverview']>;
  } {
    return {
      isRunning: this.isRunning,
      cronHealth: this.cronEngine.getHealth(),
      heartbeatActive: this.heartbeatSender.isActive(),
      nodeHealth: this.heartbeatMonitor.getOverview(),
    };
  }

  /**
   * Get job statistics
   */
  async getJobStats(jobId: string): Promise<{
    job: ReturnType<CronEngine['getJob']>;
    stats: Awaited<ReturnType<PersistenceManager['getJobStats']>>;
    recentHistory: Awaited<ReturnType<PersistenceManager['getExecutionHistory']>>;
  } | null> {
    const job = this.cronEngine.getJob(jobId);
    if (!job) return null;

    const [stats, recentHistory] = await Promise.all([
      this.persistence.getJobStats(jobId),
      this.persistence.getExecutionHistory(jobId, 10),
    ]);

    return { job, stats, recentHistory };
  }

  /**
   * Force run a job immediately
   */
  async forceRun(jobId: string): Promise<boolean> {
    return await this.cronEngine.runNow(jobId);
  }

  /**
   * Enable/disable a job
   */
  setJobEnabled(jobId: string, enabled: boolean): boolean {
    if (enabled) {
      return this.cronEngine.start(jobId);
    } else {
      return this.cronEngine.stop(jobId);
    }
  }

  /**
   * Setup event handlers
   */
  private setupEventHandlers(): void {
    // Cron engine events
    this.cronEngine.on('job:completed', async (result) => {
      await this.persistence.logExecution({
        id: `${result.jobId}-${Date.now()}`,
        jobId: result.jobId,
        startTime: result.startTime.getTime(),
        endTime: result.endTime.getTime(),
        duration: result.duration,
        success: true,
      });
    });

    this.cronEngine.on('job:failed', async (result) => {
      await this.persistence.logExecution({
        id: `${result.jobId}-${Date.now()}`,
        jobId: result.jobId,
        startTime: result.startTime.getTime(),
        endTime: result.endTime.getTime(),
        duration: result.duration,
        success: false,
        error: result.error?.message,
      });
    });

    // Heartbeat events
    this.heartbeatSender.on('heartbeat', (heartbeat) => {
      this.heartbeatMonitor.receiveHeartbeat(heartbeat);
    });

    this.heartbeatMonitor.on('node:offline', ({ nodeId }) => {
      console.error(`[SchedulerSystem] Node ${nodeId} is OFFLINE`);
    });

    this.heartbeatMonitor.on('node:suspect', ({ nodeId }) => {
      console.warn(`[SchedulerSystem] Node ${nodeId} is SUSPECT`);
    });

    // Graceful shutdown
    process.on('SIGINT', () => {
      console.log('\n[SchedulerSystem] Received SIGINT, shutting down...');
      this.stop().then(() => process.exit(0));
    });

    process.on('SIGTERM', () => {
      console.log('\n[SchedulerSystem] Received SIGTERM, shutting down...');
      this.stop().then(() => process.exit(0));
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      console.error('[SchedulerSystem] Uncaught exception:', error);
      this.stop().then(() => process.exit(1));
    });

    process.on('unhandledRejection', (reason, promise) => {
      console.error('[SchedulerSystem] Unhandled rejection at:', promise, 'reason:', reason);
    });
  }
}
