/**
 * Cron Engine - Core Scheduling System
 * OpenClaw AI Agent Scheduler
 */

import * as cron from 'node-cron';
import { EventEmitter } from 'events';
import { Mutex } from 'async-mutex';

export interface ScheduledJob {
  id: string;
  name: string;
  schedule: string;
  handler: () => Promise<void>;
  options: JobOptions;
  status: JobStatus;
  metrics: JobMetrics;
}

export interface JobOptions {
  timezone?: string;
  preventOverlap?: boolean;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enabled?: boolean;
  tags?: string[];
}

export interface JobStatus {
  state: 'idle' | 'running' | 'paused' | 'error';
  lastRun: Date | null;
  nextRun: Date | null;
  lastError: Error | null;
  consecutiveFailures: number;
  totalRuns: number;
  totalFailures: number;
}

export interface JobMetrics {
  averageExecutionTime: number;
  maxExecutionTime: number;
  minExecutionTime: number;
  totalExecutionTime: number;
}

export interface JobExecutionResult {
  jobId: string;
  success: boolean;
  startTime: Date;
  endTime: Date;
  duration: number;
  error?: Error;
}

export class CronEngine extends EventEmitter {
  private jobs: Map<string, ScheduledJob> = new Map();
  private cronTasks: Map<string, cron.ScheduledTask> = new Map();
  private jobLocks: Map<string, Mutex> = new Map();
  private runningJobs: Map<string, boolean> = new Map();

  constructor() {
    super();
  }

  /**
   * Register a new scheduled job
   */
  register(
    id: string,
    name: string,
    schedule: string,
    handler: () => Promise<void>,
    options: JobOptions = {}
  ): ScheduledJob {
    // Validate cron expression
    if (!cron.validate(schedule)) {
      throw new Error(`Invalid cron expression: ${schedule}`);
    }

    const job: ScheduledJob = {
      id,
      name,
      schedule,
      handler,
      options: {
        timezone: process.env.SCHEDULER_TIMEZONE || 'UTC',
        preventOverlap: true,
        timeout: 300000, // 5 minutes
        retryAttempts: 3,
        retryDelay: 5000,
        enabled: true,
        tags: [],
        ...options
      },
      status: {
        state: 'idle',
        lastRun: null,
        nextRun: null,
        lastError: null,
        consecutiveFailures: 0,
        totalRuns: 0,
        totalFailures: 0
      },
      metrics: {
        averageExecutionTime: 0,
        maxExecutionTime: 0,
        minExecutionTime: Infinity,
        totalExecutionTime: 0
      }
    };

    this.jobs.set(id, job);
    this.jobLocks.set(id, new Mutex());
    this.runningJobs.set(id, false);

    // Schedule the job
    if (job.options.enabled) {
      this.scheduleJob(job);
    }

    this.emit('job:registered', job);
    return job;
  }

  /**
   * Schedule a job with node-cron
   */
  private scheduleJob(job: ScheduledJob): void {
    const task = cron.schedule(
      job.schedule,
      async () => {
        await this.executeJob(job);
      },
      {
        scheduled: true,
        timezone: job.options.timezone
      }
    );

    this.cronTasks.set(job.id, task);

    // Calculate next run time
    try {
      job.status.nextRun = new Date(Date.now() + 60000); // Approximate next run
    } catch (error) {
      console.error(`Failed to parse schedule for job ${job.id}:`, error);
    }
  }

  /**
   * Execute a job with overlap prevention and error handling
   */
  private async executeJob(job: ScheduledJob): Promise<void> {
    const lock = this.jobLocks.get(job.id);
    if (!lock) return;

    // Check for overlap
    if (job.options.preventOverlap && this.runningJobs.get(job.id)) {
      this.emit('job:skipped', { jobId: job.id, reason: 'overlap' });
      return;
    }

    const release = await lock.acquire();
    const startTime = new Date();

    try {
      this.runningJobs.set(job.id, true);
      job.status.state = 'running';
      job.status.lastRun = startTime;
      job.status.totalRuns++;

      this.emit('job:started', { jobId: job.id, startTime });

      // Execute with timeout
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error(`Job ${job.id} timed out after ${job.options.timeout}ms`));
        }, job.options.timeout);
      });

      await Promise.race([job.handler(), timeoutPromise]);

      // Success
      const endTime = new Date();
      const duration = endTime.getTime() - startTime.getTime();

      job.status.state = 'idle';
      job.status.consecutiveFailures = 0;
      job.status.lastError = null;

      // Update metrics
      this.updateMetrics(job, duration);

      const result: JobExecutionResult = {
        jobId: job.id,
        success: true,
        startTime,
        endTime,
        duration
      };

      this.emit('job:completed', result);

    } catch (error) {
      const endTime = new Date();
      const duration = endTime.getTime() - startTime.getTime();

      job.status.state = 'error';
      job.status.lastError = error as Error;
      job.status.consecutiveFailures++;
      job.status.totalFailures++;

      const result: JobExecutionResult = {
        jobId: job.id,
        success: false,
        startTime,
        endTime,
        duration,
        error: error as Error
      };

      this.emit('job:failed', result);

      // Retry logic
      if (job.status.consecutiveFailures < (job.options.retryAttempts || 0)) {
        setTimeout(() => {
          this.executeJob(job);
        }, job.options.retryDelay);
      }

    } finally {
      this.runningJobs.set(job.id, false);
      release();

      // Update next run time
      job.status.nextRun = new Date(Date.now() + 60000); // Approximate next run
    }
  }

  /**
   * Update job execution metrics
   */
  private updateMetrics(job: ScheduledJob, duration: number): void {
    const metrics = job.metrics;
    metrics.totalExecutionTime += duration;
    metrics.maxExecutionTime = Math.max(metrics.maxExecutionTime, duration);
    metrics.minExecutionTime = Math.min(metrics.minExecutionTime, duration);
    metrics.averageExecutionTime = metrics.totalExecutionTime / job.status.totalRuns;
  }

  /**
   * Start a specific job
   */
  start(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.start();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = true;
        this.emit('job:started', { jobId });
      }
      return true;
    }
    return false;
  }

  /**
   * Stop a specific job
   */
  stop(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.stop();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = false;
        job.status.state = 'paused';
        this.emit('job:stopped', { jobId });
      }
      return true;
    }
    return false;
  }

  /**
   * Run a job immediately (one-time execution)
   */
  async runNow(jobId: string): Promise<boolean> {
    const job = this.jobs.get(jobId);
    if (job) {
      await this.executeJob(job);
      return true;
    }
    return false;
  }

  /**
   * Remove a job
   */
  unregister(jobId: string): boolean {
    const task = this.cronTasks.get(jobId);
    if (task) {
      task.stop();
      this.cronTasks.delete(jobId);
    }

    this.jobs.delete(jobId);
    this.jobLocks.delete(jobId);
    this.runningJobs.delete(jobId);

    this.emit('job:unregistered', { jobId });
    return true;
  }

  /**
   * Get job information
   */
  getJob(jobId: string): ScheduledJob | undefined {
    return this.jobs.get(jobId);
  }

  /**
   * Get all jobs
   */
  getAllJobs(): ScheduledJob[] {
    return Array.from(this.jobs.values());
  }

  /**
   * Get jobs by tag
   */
  getJobsByTag(tag: string): ScheduledJob[] {
    return this.getAllJobs().filter(job => job.options.tags?.includes(tag));
  }

  /**
   * Stop all jobs
   */
  stopAll(): void {
    for (const [jobId, task] of this.cronTasks) {
      task.stop();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = false;
      }
    }
    this.emit('all:stopped');
  }

  /**
   * Start all jobs
   */
  startAll(): void {
    for (const [jobId, task] of this.cronTasks) {
      task.start();
      const job = this.jobs.get(jobId);
      if (job) {
        job.options.enabled = true;
      }
    }
    this.emit('all:started');
  }

  /**
   * Get system health
   */
  getHealth(): {
    totalJobs: number;
    runningJobs: number;
    failedJobs: number;
    idleJobs: number;
  } {
    const jobs = this.getAllJobs();
    return {
      totalJobs: jobs.length,
      runningJobs: jobs.filter(j => j.status.state === 'running').length,
      failedJobs: jobs.filter(j => j.status.consecutiveFailures > 0).length,
      idleJobs: jobs.filter(j => j.status.state === 'idle').length
    };
  }
}
