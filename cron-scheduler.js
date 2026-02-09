/**
 * OpenClawAgent - Cron Scheduler
 * Manages scheduled tasks and periodic jobs
 */

const cron = require('node-cron');
const logger = require('./logger');

class CronScheduler {
  constructor(options = {}) {
    this.jobs = new Map();
    this.tasks = new Map();
    this.isRunning = false;
    this.jobFailures = new Map(); // { name -> { count, pausedUntil } }
    this.maxConsecutiveFailures = options.maxConsecutiveFailures || 5;
    this.failureCooldownMs = options.failureCooldownMs || 10 * 60 * 1000; // 10 minutes default
  }

  initialize() {
    logger.info('[CronScheduler] Initializing cron jobs...');

    // Define cron jobs
    this.scheduleJob('heartbeat', '*/30 * * * * *', () => this.heartbeatTask());
    this.scheduleJob('cleanup', '0 0 * * *', () => this.cleanupTask());
    this.scheduleJob('backup', '0 2 * * *', () => this.backupTask());
    this.scheduleJob('report', '0 9 * * 1', () => this.weeklyReportTask());
    this.scheduleJob('health-check', '*/5 * * * *', () => this.healthCheckTask());
    this.scheduleJob('state-persist', '*/10 * * * *', () => this.statePersistTask());
    this.scheduleJob('metrics-collect', '* * * * *', () => this.metricsCollectTask());
    
    this.isRunning = true;
    logger.info(`[CronScheduler] ${this.jobs.size} jobs scheduled`);
  }

  scheduleJob(name, cronExpression, handler, options = {}) {
    // Validate cron expression
    if (!cron.validate(cronExpression)) {
      logger.error(`[CronScheduler] Invalid cron expression for ${name}: ${cronExpression}`);
      return;
    }

    const task = cron.schedule(cronExpression, async () => {
      // Circuit breaker: check if job is paused due to consecutive failures
      const failTracker = this.jobFailures.get(name) || { count: 0, pausedUntil: 0 };
      if (failTracker.pausedUntil > Date.now()) {
        logger.debug(`[CronScheduler] Skipping paused job: ${name} (resumes at ${new Date(failTracker.pausedUntil).toISOString()})`);
        return;
      }
      // Auto-resume: clear pause if cooldown has passed
      if (failTracker.pausedUntil > 0 && failTracker.pausedUntil <= Date.now()) {
        failTracker.pausedUntil = 0;
        failTracker.count = 0;
        logger.info(`[CronScheduler] Auto-resumed job after cooldown: ${name}`);
      }

      logger.debug(`[CronScheduler] Executing job: ${name}`);
      const startTime = Date.now();

      try {
        await handler();
        // Reset failure count on success
        failTracker.count = 0;
        failTracker.pausedUntil = 0;
        this.jobFailures.set(name, failTracker);
        logger.debug(`[CronScheduler] Job completed: ${name} (${Date.now() - startTime}ms)`);
      } catch (error) {
        failTracker.count++;
        if (failTracker.count >= this.maxConsecutiveFailures) {
          failTracker.pausedUntil = Date.now() + this.failureCooldownMs;
          logger.warn(`[CronScheduler] Job paused after ${failTracker.count} consecutive failures: ${name}. Cooldown until ${new Date(failTracker.pausedUntil).toISOString()}`);
          failTracker.count = 0; // reset for after cooldown
        }
        this.jobFailures.set(name, failTracker);
        logger.error(`[CronScheduler] Job failed: ${name}`, error);

        // Report error to master
        if (process.send) {
          process.send({
            type: 'cron-error',
            data: { job: name, error: error.message }
          });
        }
      }
    }, {
      scheduled: options.scheduled !== false,
      timezone: options.timezone || process.env.SCHEDULER_TIMEZONE || 'UTC',
      name: name
    });
    
    this.jobs.set(name, { 
      expression: cronExpression, 
      handler,
      options,
      createdAt: Date.now()
    });
    
    this.tasks.set(name, task);
    logger.debug(`[CronScheduler] Scheduled job: ${name} (${cronExpression})`);
  }

  unscheduleJob(name) {
    const task = this.tasks.get(name);
    if (task) {
      task.stop();
      this.tasks.delete(name);
      this.jobs.delete(name);
      logger.info(`[CronScheduler] Unscheduled job: ${name}`);
      return true;
    }
    return false;
  }

  pauseJob(name) {
    const task = this.tasks.get(name);
    if (task) {
      task.stop();
      logger.info(`[CronScheduler] Paused job: ${name}`);
      return true;
    }
    return false;
  }

  resumeJob(name) {
    const task = this.tasks.get(name);
    if (task) {
      task.start();
      logger.info(`[CronScheduler] Resumed job: ${name}`);
      return true;
    }
    return false;
  }

  // Cron task implementations
  async heartbeatTask() {
    // System heartbeat - update local health state
    // Note: In master process, process.send is not available.
    // Health data is accessed via getJobStatus() and health-monitor instead.
    this._lastHeartbeat = {
      timestamp: Date.now(),
      uptime: process.uptime(),
      memory: process.memoryUsage()
    };
    if (process.send) {
      process.send({
        type: 'cron-heartbeat',
        data: this._lastHeartbeat
      });
    }
  }

  async cleanupTask() {
    logger.info('[CronScheduler] Running cleanup task');
    const fs = require('fs');
    const path = require('path');
    const logDir = path.join(__dirname, 'logs');

    try {
      if (fs.existsSync(logDir)) {
        const files = fs.readdirSync(logDir);
        const cutoff = Date.now() - (14 * 24 * 60 * 60 * 1000); // 14 days
        let cleaned = 0;

        for (const file of files) {
          const filePath = path.join(logDir, file);
          try {
            const stat = fs.statSync(filePath);
            if (stat.mtimeMs < cutoff && file.endsWith('.log')) {
              fs.unlinkSync(filePath);
              cleaned++;
            }
          } catch (e) {
            // Skip files that can't be accessed
          }
        }
        logger.info(`[CronScheduler] Cleanup: removed ${cleaned} old log files`);
      }
    } catch (error) {
      logger.error('[CronScheduler] Cleanup error:', error.message);
    }
  }

  async backupTask() {
    logger.info('[CronScheduler] Running backup task');
    if (process.send) {
      process.send({
        type: 'backup-request',
        data: { timestamp: Date.now() }
      });
    }
  }

  async weeklyReportTask() {
    logger.info('[CronScheduler] Generating weekly report');
    const report = {
      timestamp: Date.now(),
      week: this.getWeekNumber(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      jobs: this.getJobStatus(),
    };
    logger.info('[CronScheduler] Weekly report:', JSON.stringify(report));

    if (process.send) {
      process.send({
        type: 'weekly-report',
        data: report
      });
    }
  }

  async healthCheckTask() {
    logger.debug('[CronScheduler] Running health check task');
    const health = {
      timestamp: Date.now(),
      memory: process.memoryUsage(),
      uptime: process.uptime(),
      cpuUsage: process.cpuUsage(),
    };

    if (process.send) {
      process.send({
        type: 'health-check',
        data: health
      });
    }
  }

  async statePersistTask() {
    logger.debug('[CronScheduler] Running state persist task');
    if (process.send) {
      process.send({
        type: 'persist-state',
        data: {
          timestamp: Date.now(),
          jobs: this.getJobStatus(),
        }
      });
    }
  }

  async metricsCollectTask() {
    // Collect metrics
    logger.debug('[CronScheduler] Running metrics collection task');
    
    const metrics = {
      timestamp: Date.now(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      uptime: process.uptime()
    };
    
    if (process.send) {
      process.send({
        type: 'metrics',
        data: metrics
      });
    }
  }

  // Utility methods
  getWeekNumber() {
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 1);
    const diff = now - start;
    const oneWeek = 1000 * 60 * 60 * 24 * 7;
    return Math.floor(diff / oneWeek);
  }

  getJobStatus() {
    const status = {};
    for (const [name, job] of this.jobs) {
      const task = this.tasks.get(name);
      status[name] = {
        expression: job.expression,
        createdAt: job.createdAt,
        running: task ? task.getStatus() === 'scheduled' : false
      };
    }
    return status;
  }

  stop() {
    logger.info('[CronScheduler] Stopping all jobs...');
    
    for (const [name, task] of this.tasks) {
      task.stop();
      logger.debug(`[CronScheduler] Stopped job: ${name}`);
    }
    
    this.isRunning = false;
    logger.info('[CronScheduler] All jobs stopped');
  }

  start() {
    logger.info('[CronScheduler] Starting all jobs...');
    
    for (const [name, task] of this.tasks) {
      task.start();
      logger.debug(`[CronScheduler] Started job: ${name}`);
    }
    
    this.isRunning = true;
    logger.info('[CronScheduler] All jobs started');
  }
}

module.exports = CronScheduler;
