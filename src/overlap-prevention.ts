/**
 * Overlap Prevention - Job Execution Control
 * OpenClaw AI Agent Scheduler
 */

import { Mutex, Semaphore } from 'async-mutex';

export interface OverlapPreventionConfig {
  strategy: 'skip' | 'queue' | 'delay' | 'timeout' | 'concurrent';
  maxConcurrent?: number;
  maxQueueSize?: number;
  delayMs?: number;
  timeoutMs?: number;
}

export class OverlapPrevention {
  private mutexes: Map<string, Mutex> = new Map();
  private semaphores: Map<string, Semaphore> = new Map();
  private jobQueues: Map<string, Array<() => void>> = new Map();
  private runningJobs: Map<string, boolean> = new Map();
  private skipCounts: Map<string, number> = new Map();

  constructor(private config: OverlapPreventionConfig) {}

  /**
   * Execute with overlap prevention
   */
  async execute<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    switch (this.config.strategy) {
      case 'skip':
        return this.executeWithSkip(jobId, fn);
      case 'queue':
        return this.executeWithQueue(jobId, fn);
      case 'delay':
        return this.executeWithDelay(jobId, fn);
      case 'timeout':
        return this.executeWithTimeout(jobId, fn);
      case 'concurrent':
        return this.executeWithConcurrency(jobId, fn);
      default:
        return fn();
    }
  }

  /**
   * Skip strategy - skip if job is already running
   */
  private async executeWithSkip<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      const skipCount = (this.skipCounts.get(jobId) || 0) + 1;
      this.skipCounts.set(jobId, skipCount);
      console.log(`[OverlapPrevention] Job ${jobId} skipped (count: ${skipCount})`);
      return null;
    }

    this.runningJobs.set(jobId, true);
    this.skipCounts.set(jobId, 0);

    try {
      return await fn();
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Queue strategy - queue jobs if already running
   */
  private async executeWithQueue<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const mutex = this.getMutex(jobId);
    const release = await mutex.acquire();

    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Delay strategy - delay execution if job is running
   */
  private async executeWithDelay<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      console.log(`[OverlapPrevention] Job ${jobId} delaying for ${this.config.delayMs}ms`);
      await this.sleep(this.config.delayMs || 5000);
      
      // Check again after delay
      if (this.runningJobs.get(jobId)) {
        console.log(`[OverlapPrevention] Job ${jobId} still running after delay, skipping`);
        return null;
      }
    }

    this.runningJobs.set(jobId, true);

    try {
      return await fn();
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Timeout strategy - force stop after timeout
   */
  private async executeWithTimeout<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T | null> {
    if (this.runningJobs.get(jobId)) {
      const skipCount = (this.skipCounts.get(jobId) || 0) + 1;
      this.skipCounts.set(jobId, skipCount);

      // Force timeout after max skips
      if (skipCount > 2) {
        console.log(`[OverlapPrevention] Job ${jobId} forcing timeout after ${skipCount} skips`);
        this.runningJobs.set(jobId, false);
        this.skipCounts.set(jobId, 0);
      } else {
        console.log(`[OverlapPrevention] Job ${jobId} skipped (count: ${skipCount})`);
        return null;
      }
    }

    this.runningJobs.set(jobId, true);
    this.skipCounts.set(jobId, 0);

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Job ${jobId} timed out`));
      }, this.config.timeoutMs || 300000);
    });

    try {
      return await Promise.race([fn(), timeoutPromise]);
    } finally {
      this.runningJobs.set(jobId, false);
    }
  }

  /**
   * Concurrent strategy - allow limited concurrency
   */
  private async executeWithConcurrency<T>(
    jobId: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const semaphore = this.getSemaphore(jobId);
    const [value, release] = await semaphore.acquire();

    console.log(`[OverlapPrevention] Job ${jobId} acquired slot (${value} remaining)`);

    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Get or create mutex for job
   */
  private getMutex(jobId: string): Mutex {
    if (!this.mutexes.has(jobId)) {
      this.mutexes.set(jobId, new Mutex());
    }
    return this.mutexes.get(jobId)!;
  }

  /**
   * Get or create semaphore for job
   */
  private getSemaphore(jobId: string): Semaphore {
    if (!this.semaphores.has(jobId)) {
      this.semaphores.set(
        jobId,
        new Semaphore(this.config.maxConcurrent || 1)
      );
    }
    return this.semaphores.get(jobId)!;
  }

  /**
   * Sleep helper
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get overlap statistics
   */
  getStats(jobId: string): {
    isRunning: boolean;
    skipCount: number;
    queueSize: number;
  } {
    return {
      isRunning: this.runningJobs.get(jobId) || false,
      skipCount: this.skipCounts.get(jobId) || 0,
      queueSize: this.jobQueues.get(jobId)?.length || 0,
    };
  }

  /**
   * Reset stats for a job
   */
  resetStats(jobId: string): void {
    this.skipCounts.set(jobId, 0);
  }

  /**
   * Force stop a running job
   */
  forceStop(jobId: string): boolean {
    if (this.runningJobs.get(jobId)) {
      this.runningJobs.set(jobId, false);
      this.skipCounts.set(jobId, 0);
      return true;
    }
    return false;
  }
}
