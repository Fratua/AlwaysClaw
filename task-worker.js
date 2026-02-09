/**
 * OpenClawAgent - Task Worker
 * Handles background job processing and task execution.
 * Wired to Python bridge for GPT-5.2, memory, Gmail, and other services.
 */

const WorkerBase = require('./worker-base');
const logger = require('./logger');
const { getBridge } = require('./python-bridge');
const fs = require('fs');

class TaskWorker extends WorkerBase {
  constructor() {
    super();
    this.taskQueue = [];
    this.isProcessing = false;
    this.currentTask = null;
    this.bridge = null;
    this.stats = {
      tasksCompleted: 0,
      tasksFailed: 0,
      tasksCancelled: 0,
      totalExecutionTime: 0
    };
    this.taskHandlers = new Map();
    this.registerDefaultHandlers();
  }

  async onInitialize() {
    logger.info(`[TaskWorker ${this.workerId}] Initializing task worker...`);

    // Initialize bridge proxy
    this.bridge = getBridge();
    await this.bridge.start();

    // Start task processing loop
    this.startTaskLoop();
  }

  registerDefaultHandlers() {
    this.registerHandler('email-process', this.handleEmailProcess.bind(this));
    this.registerHandler('data-sync', this.handleDataSync.bind(this));
    this.registerHandler('report-generate', this.handleReportGenerate.bind(this));
    this.registerHandler('cleanup', this.handleCleanup.bind(this));
    this.registerHandler('backup', this.handleBackup.bind(this));
    this.registerHandler('notification', this.handleNotification.bind(this));
    this.registerHandler('api-call', this.handleAPICall.bind(this));
    this.registerHandler('file-process', this.handleFileProcess.bind(this));
    this.registerHandler('ai-process', this.handleAIProcess.bind(this));
  }

  registerHandler(taskType, handler) {
    this.taskHandlers.set(taskType, handler);
    logger.debug(`[TaskWorker ${this.workerId}] Registered handler for: ${taskType}`);
  }

  startTaskLoop() {
    logger.info(`[TaskWorker ${this.workerId}] Starting task processing loop`);
    this.processTasks();
  }

  async processTasks() {
    while (!this.isShuttingDown) {
      if (this.taskQueue.length > 0 && !this.isProcessing) {
        const task = this.taskQueue.shift();
        await this.executeTask(task);
      } else {
        await this.sleep(100);
      }
    }
  }

  async executeTask(task) {
    this.isProcessing = true;
    this.currentTask = task;
    const startTime = Date.now();

    logger.info(`[TaskWorker ${this.workerId}] Executing task: ${task.id} (${task.type})`);

    try {
      const handler = this.taskHandlers.get(task.type);

      if (!handler) {
        throw new Error(`No handler registered for task type: ${task.type}`);
      }

      const result = await this.withTimeout(
        handler(task.data),
        task.timeout || 300000,
        `Task ${task.id} timed out`
      );

      const executionTime = Date.now() - startTime;
      this.stats.tasksCompleted++;
      this.stats.totalExecutionTime += executionTime;

      logger.info(`[TaskWorker ${this.workerId}] Task completed: ${task.id} (${executionTime}ms)`);

      this.sendToMaster({
        type: 'task-complete',
        data: {
          taskId: task.id,
          result,
          executionTime,
          workerId: this.workerId
        }
      });

      return result;

    } catch (error) {
      this.stats.tasksFailed++;
      logger.error(`[TaskWorker ${this.workerId}] Task failed: ${task.id}`, error);

      this.sendToMaster({
        type: 'task-error',
        data: {
          taskId: task.id,
          error: error.message,
          stack: error.stack,
          workerId: this.workerId
        }
      });
    } finally {
      this.isProcessing = false;
      this.currentTask = null;
    }
  }

  async onTask(data) {
    const task = {
      id: data.id || `task-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
      type: data.type,
      data: data.payload || data,
      priority: data.priority || 'normal',
      timeout: data.timeout || 300000,
      createdAt: Date.now()
    };

    if (task.priority === 'high') {
      this.taskQueue.unshift(task);
    } else {
      this.taskQueue.push(task);
    }

    logger.info(`[TaskWorker ${this.workerId}] Task queued: ${task.id} (${task.type})`);

    return {
      queued: true,
      taskId: task.id,
      position: this.taskQueue.length
    };
  }

  async onShutdown() {
    logger.info(`[TaskWorker ${this.workerId}] Shutting down...`);

    if (this.currentTask) {
      this.stats.tasksCancelled++;
      logger.warn(`[TaskWorker ${this.workerId}] Cancelling current task: ${this.currentTask.id}`);
    }

    if (this.bridge) {
      await this.bridge.stop();
    }

    if (this.taskQueue.length > 0) {
      this.sendToMaster({
        type: 'pending-tasks',
        data: {
          workerId: this.workerId,
          tasks: this.taskQueue
        }
      });
    }

    logger.info(`[TaskWorker ${this.workerId}] Stats:`, this.stats);
  }

  // ── Task Handlers (wired to bridge) ────────────────────────

  async handleEmailProcess(data) {
    logger.info(`[TaskWorker ${this.workerId}] Processing email task via bridge`);
    return this.bridge.call('gmail.process_batch', data || {});
  }

  async handleDataSync(data) {
    logger.info(`[TaskWorker ${this.workerId}] Processing data sync task via bridge`);
    return this.bridge.call('memory.sync', data || {});
  }

  async handleReportGenerate(data) {
    logger.info(`[TaskWorker ${this.workerId}] Generating report via GPT-5.2`);
    const result = await this.bridge.call('llm.generate', {
      prompt: `Generate a system report based on this data: ${JSON.stringify(data || {})}`,
      system: 'You are a system report generator. Produce concise, structured reports.',
    });
    return { generated: true, reportId: 'report-' + Date.now(), content: result.content };
  }

  async handleCleanup(data) {
    logger.info(`[TaskWorker ${this.workerId}] Running cleanup task via bridge`);
    return this.bridge.call('memory.consolidate', data || {});
  }

  async handleBackup(data) {
    logger.info(`[TaskWorker ${this.workerId}] Running backup task via bridge`);
    return this.bridge.call('memory.backup', data || {});
  }

  async handleNotification(data) {
    logger.info(`[TaskWorker ${this.workerId}] Sending notification`);
    // Route to appropriate channel
    if (data && data.channel === 'email') {
      return this.bridge.call('gmail.send', data);
    } else if (data && data.channel === 'sms') {
      return this.bridge.call('twilio.sms', data);
    }
    return { sent: true, notificationId: 'notif-' + Date.now() };
  }

  async handleAPICall(data) {
    logger.info(`[TaskWorker ${this.workerId}] Making API call`);
    const url = data?.url;
    if (!url) throw new Error('API call requires a url');
    const method = (data.method || 'GET').toUpperCase();
    const headers = data.headers || {};
    const body = data.body ? JSON.stringify(data.body) : undefined;
    const timeoutMs = data.timeout || 30000;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json', ...headers },
        body: method !== 'GET' && method !== 'HEAD' ? body : undefined,
        signal: controller.signal,
      });
      const responseData = await response.text();
      let parsed;
      try { parsed = JSON.parse(responseData); } catch { parsed = responseData; }
      return {
        success: response.ok,
        status: response.status,
        headers: Object.fromEntries(response.headers.entries()),
        data: parsed,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  async handleFileProcess(data) {
    logger.info(`[TaskWorker ${this.workerId}] Processing file`);
    const operation = data?.operation || 'read';
    const inputPath = data?.inputPath;
    if (!inputPath) throw new Error('File process requires an inputPath');

    switch (operation) {
      case 'read': {
        const content = await fs.promises.readFile(inputPath);
        const stats = await fs.promises.stat(inputPath);
        return { processed: true, size: stats.size, content: content.toString('utf-8').substring(0, 10000) };
      }
      case 'copy': {
        const outputPath = data?.outputPath;
        if (!outputPath) throw new Error('Copy operation requires an outputPath');
        await fs.promises.copyFile(inputPath, outputPath);
        const stats = await fs.promises.stat(outputPath);
        return { processed: true, outputPath, size: stats.size };
      }
      case 'stats': {
        const stats = await fs.promises.stat(inputPath);
        return { processed: true, size: stats.size, isFile: stats.isFile(), isDirectory: stats.isDirectory(), mtime: stats.mtime.toISOString() };
      }
      default:
        throw new Error(`Unknown file operation: ${operation}`);
    }
  }

  async handleAIProcess(data) {
    logger.info(`[TaskWorker ${this.workerId}] Processing with GPT-5.2 via bridge`);
    return this.bridge.call('llm.complete', {
      messages: data.messages || [{ role: 'user', content: data.prompt || '' }],
      system: data.system || '',
      max_tokens: data.max_tokens || 4096,
      temperature: data.temperature || 0.7,
    });
  }

  // Utility methods

  getQueueStatus() {
    return {
      queueLength: this.taskQueue.length,
      isProcessing: this.isProcessing,
      currentTask: this.currentTask ? this.currentTask.id : null,
      stats: this.stats
    };
  }

  cancelTask(taskId) {
    const index = this.taskQueue.findIndex(t => t.id === taskId);
    if (index !== -1) {
      this.taskQueue.splice(index, 1);
      this.stats.tasksCancelled++;
      return { cancelled: true };
    }
    return { cancelled: false, reason: 'Task not found in queue' };
  }
}

// Initialize worker if this file is run directly
if (require.main === module) {
  const worker = new TaskWorker();
  worker.initialize().catch(error => {
    logger.error('[TaskWorker] Initialization failed:', error);
    process.exit(1);
  });
}

module.exports = TaskWorker;
