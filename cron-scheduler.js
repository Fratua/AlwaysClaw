/**
 * OpenClawAgent - Cron Scheduler
 * Manages scheduled tasks and periodic jobs
 */

const cron = require('node-cron');
const logger = require('./logger');

class CronScheduler {
  constructor() {
    this.jobs = new Map();
    this.tasks = new Map();
    this.isRunning = false;
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
      logger.debug(`[CronScheduler] Executing job: ${name}`);
      const startTime = Date.now();
      
      try {
        await handler();
        logger.debug(`[CronScheduler] Job completed: ${name} (${Date.now() - startTime}ms)`);
      } catch (error) {
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
      timezone: options.timezone || 'America/New_York',
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
    // Send system heartbeat
    if (process.send) {
      process.send({
        type: 'cron-heartbeat',
        data: { 
          timestamp: Date.now(),
          uptime: process.uptime(),
          memory: process.memoryUsage()
        }
      });
    }
  }

  async cleanupTask() {
    // Cleanup old logs, temp files, and resources
    logger.info('[CronScheduler] Running cleanup task');
    
    // Clean old log files (keep last 14 days)
    // Clean temp files
    // Clear old cache entries
    // Release unused resources
  }

  async backupTask() {
    // Backup state and data
    logger.info('[CronScheduler] Running backup task');
    
    // Backup state to persistent storage
    // Backup configuration
    // Backup logs
    
    if (process.send) {
      process.send({
        type: 'backup-request',
        data: { timestamp: Date.now() }
      });
    }
  }

  async weeklyReportTask() {
    // Generate weekly report
    logger.info('[CronScheduler] Running weekly report task');
    
    // Collect weekly statistics
    // Generate report
    // Send notification if configured
    
    if (process.send) {
      process.send({
        type: 'weekly-report',
        data: { 
          timestamp: Date.now(),
          week: this.getWeekNumber()
        }
      });
    }
  }

  async healthCheckTask() {
    // Perform health checks
    logger.debug('[CronScheduler] Running health check task');
    
    // Check system resources
    // Check worker health
    // Check external service connectivity
    
    if (process.send) {
      process.send({
        type: 'health-check',
        data: { 
          timestamp: Date.now(),
          memory: process.memoryUsage(),
          uptime: process.uptime()
        }
      });
    }
  }

  async statePersistTask() {
    // Persist current state
    logger.debug('[CronScheduler] Running state persist task');
    
    if (process.send) {
      process.send({
        type: 'persist-state',
        data: { timestamp: Date.now() }
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
