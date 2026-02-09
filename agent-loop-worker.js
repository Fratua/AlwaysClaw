/**
 * OpenClawAgent - Agent Loop Worker
 * Executes agentic loops for AI task processing
 * Wired to Python bridge for GPT-5.2 reasoning, Gmail, Twilio, TTS, and memory.
 */

const WorkerBase = require('./worker-base');
const logger = require('./logger');
const { getBridge } = require('./python-bridge');

// Supported action types - derived from executeAction switch cases
const SUPPORTED_ACTIONS = [
  'gmail.send', 'gmail.read', 'browser.navigate', 'browser.click',
  'twilio.call', 'twilio.sms', 'tts.speak', 'system.command'
];

class AgentLoopWorker extends WorkerBase {
  constructor() {
    super();
    let parsedConfig = {};
    try {
      parsedConfig = JSON.parse(process.env.AGENT_LOOP_CONFIG || '{}');
    } catch (err) {
      logger.error(`Failed to parse AGENT_LOOP_CONFIG: ${err.message}. Using defaults.`);
    }
    this.loopConfig = parsedConfig;
    this.loopId = this.loopConfig.loopId;
    this.capabilities = this.loopConfig.capabilities || [];
    this.priority = this.loopConfig.priority || 'normal';
    this.isRunning = false;
    this.currentTask = null;
    this.currentTaskPromise = null;
    this.taskQueue = [];
    this.bridge = null;
    this.pendingIORequests = new Map();
    this.stats = {
      tasksCompleted: 0,
      tasksFailed: 0,
      startTime: Date.now()
    };
  }

  onMessage(msg) {
    if (msg.type === 'io-response') {
      const pending = this.pendingIORequests.get(msg.data.requestId);
      if (pending) {
        this.pendingIORequests.delete(msg.data.requestId);
        if (msg.data.error) {
          pending.reject(new Error(msg.data.error));
        } else {
          pending.resolve(msg.data.result);
        }
      }
    }
  }

  async onInitialize() {
    logger.info(`[AgentLoop ${this.loopId}] Capabilities: ${this.capabilities.join(', ')}`);
    logger.info(`[AgentLoop ${this.loopId}] Priority: ${this.priority}`);

    // Initialize bridge proxy (worker side)
    await this.initializeAI();

    // Start the agent loop
    this.startAgentLoop();
  }

  async initializeAI() {
    // Initialize bridge proxy for GPT-5.2 and service calls
    this.bridge = getBridge();
    await this.bridge.start();
    logger.info(`[AgentLoop ${this.loopId}] AI bridge initialized (GPT-5.2)`);
  }

  startAgentLoop() {
    this.isRunning = true;
    logger.info(`[AgentLoop ${this.loopId}] Starting agent loop`);
    this.runLoop();
  }

  async runLoop() {
    let consecutiveErrors = 0;
    while (this.isRunning && !this.isShuttingDown) {
      try {
        await this.executeAgentCycle();
        consecutiveErrors = 0;
      } catch (error) {
        consecutiveErrors++;
        logger.error(`[AgentLoop ${this.loopId}] Cycle error (${consecutiveErrors}):`, error);
        this.stats.tasksFailed++;
        // Exponential backoff: 5s, 10s, 20s, 40s, max 60s
        const backoff = Math.min(5000 * Math.pow(2, consecutiveErrors - 1), 60000);
        await this.sleep(backoff);
      }
    }
  }

  async executeAgentCycle() {
    // Agent loop cycle:
    // 1. Check for pending tasks
    // 2. Gather context and state
    // 3. Process with GPT-5.2 (via bridge)
    // 4. Execute actions (via bridge)
    // 5. Update state
    // 6. Report results

    logger.debug(`[AgentLoop ${this.loopId}] Executing cycle`);

    // Check for tasks in queue
    if (this.taskQueue.length > 0) {
      const task = this.taskQueue.shift();
      this.currentTaskPromise = this.processTask(task);
      await this.currentTaskPromise;
      this.currentTaskPromise = null;
    } else {
      // Idle cycle - perform maintenance or wait
      await this.performIdleCycle();
    }

    // Dynamic sleep based on priority
    const sleepTime = this.priority === 'high' ? 100 : 500;
    await this.sleep(sleepTime);
  }

  async processTask(task) {
    logger.info(`[AgentLoop ${this.loopId}] Processing task: ${task.id}`);

    this.currentTask = task;
    const startTime = Date.now();

    try {
      // Step 1: Gather context
      const context = await this.gatherContext(task);

      // Step 2: Process with GPT-5.2 via bridge
      const aiResponse = await this.processWithAI(task, context);

      // Step 3: Execute actions via bridge
      const results = await this.executeActions(aiResponse);

      // Step 4: Update state
      await this.updateState(task, results);

      // Step 5: Report completion
      this.reportTaskComplete(task, results, Date.now() - startTime);

      this.stats.tasksCompleted++;

    } catch (error) {
      logger.error(`[AgentLoop ${this.loopId}] Task processing error:`, error);
      this.reportTaskError(task, error);
      this.stats.tasksFailed++;
    } finally {
      this.currentTask = null;
    }
  }

  async gatherContext(task) {
    // Gather relevant context for the task
    const context = {
      task,
      capabilities: this.capabilities,
      timestamp: Date.now(),
      history: [],
      state: {}
    };

    // Add capability-specific context via bridge
    if (this.capabilities.includes('gmail')) {
      context.gmail = await this.getGmailContext();
    }

    if (this.capabilities.includes('browser')) {
      context.browser = await this.getBrowserContext();
    }

    if (this.capabilities.includes('system')) {
      context.system = await this.getSystemContext();
    }

    return context;
  }

  async processWithAI(task, context) {
    // Send to GPT-5.2 for processing via Python bridge
    logger.debug(`[AgentLoop ${this.loopId}] Sending to GPT-5.2 for processing`);

    try {
      const result = await this.bridge.call('llm.complete', {
        messages: [
          {
            role: 'user',
            content: `Process this task and return a JSON list of actions to execute.\n\nTask: ${JSON.stringify(task)}\n\nContext: ${JSON.stringify(context)}\n\nRespond with JSON: {"actions": [{"type": "action.type", "data": {...}}], "response": "summary"}`
          }
        ],
        system: `You are an AI agent loop processor. Analyze tasks and return structured actions. Available action types: ${SUPPORTED_ACTIONS.join(', ')}. Capabilities: ${this.capabilities.join(', ')}`,
        max_tokens: 2048,
        temperature: 0.3,
      });

      // Try to parse structured response
      const content = result.content || '';
      // First: try parsing the whole response as JSON
      try {
        return JSON.parse(content);
      } catch (_wholeParseErr) {
        // Not valid JSON as-is, try regex extraction
      }
      try {
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          return JSON.parse(jsonMatch[0]);
        }
      } catch (parseErr) {
        logger.warn(`[AgentLoop ${this.loopId}] Failed to parse GPT-5.2 JSON response: ${content.substring(0, 500)}`);
      }

      return {
        actions: [],
        response: result.content || 'No response from GPT-5.2'
      };
    } catch (error) {
      logger.error(`[AgentLoop ${this.loopId}] GPT-5.2 processing error:`, error);
      return { actions: [], response: `Error: ${error.message}` };
    }
  }

  async executeActions(aiResponse) {
    const results = [];

    for (const action of aiResponse.actions || []) {
      try {
        const result = await this.executeAction(action);
        results.push({ action, result, success: true });
      } catch (error) {
        results.push({ action, error: error.message, success: false });
      }
    }

    return results;
  }

  async executeAction(action) {
    // Validate action structure
    if (!action || typeof action.type !== 'string') {
      return { success: false, error: 'Invalid action: missing or non-string type' };
    }
    if (action.data !== undefined && (action.data === null || typeof action.data !== 'object')) {
      return { success: false, error: 'Invalid action: data must be a non-null object' };
    }

    logger.debug(`[AgentLoop ${this.loopId}] Executing action: ${action.type}`);

    switch(action.type) {
      case 'gmail.send':
        return this.executeGmailSend(action);
      case 'gmail.read':
        return this.executeGmailRead(action);
      case 'browser.navigate':
        return this.executeBrowserNavigate(action);
      case 'browser.click':
        return this.executeBrowserClick(action);
      case 'twilio.call':
        return this.executeTwilioCall(action);
      case 'twilio.sms':
        return this.executeTwilioSMS(action);
      case 'tts.speak':
        return this.executeTTS(action);
      case 'system.command':
        return this.executeSystemCommand(action);
      default:
        return { success: false, error: `Unknown action type: ${action.type}` };
    }
  }

  async executeGmailSend(action) {
    if (!this.capabilities.includes('gmail')) {
      throw new Error('Gmail capability not available');
    }
    return this.bridge.call('gmail.send', action.data || {});
  }

  async executeGmailRead(action) {
    if (!this.capabilities.includes('gmail')) {
      throw new Error('Gmail capability not available');
    }
    return this.bridge.call('gmail.read', action.data || {});
  }

  _sendBrowserRequest(browserAction, data, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      const requestId = `req-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
      this.pendingIORequests.set(requestId, { resolve, reject });
      this.sendToMaster({
        type: 'io-request',
        data: {
          service: 'browser',
          action: browserAction,
          data: data || {},
          requestId,
        }
      });
      setTimeout(() => {
        if (this.pendingIORequests.has(requestId)) {
          this.pendingIORequests.delete(requestId);
          reject(new Error(`Browser ${browserAction} timeout`));
        }
      }, timeoutMs);
    });
  }

  async executeBrowserNavigate(action) {
    if (!this.capabilities.includes('browser')) {
      throw new Error('Browser capability not available');
    }
    return this._sendBrowserRequest('navigate', action.data);
  }

  async executeBrowserClick(action) {
    if (!this.capabilities.includes('browser')) {
      throw new Error('Browser capability not available');
    }
    return this._sendBrowserRequest('click', action.data);
  }

  async executeTwilioCall(action) {
    if (!this.capabilities.includes('twilio')) {
      throw new Error('Twilio capability not available');
    }
    return this.bridge.call('twilio.call', action.data || {});
  }

  async executeTwilioSMS(action) {
    if (!this.capabilities.includes('twilio')) {
      throw new Error('Twilio capability not available');
    }
    return this.bridge.call('twilio.sms', action.data || {});
  }

  async executeTTS(action) {
    if (!this.capabilities.includes('tts')) {
      throw new Error('TTS capability not available');
    }
    return this.bridge.call('tts.speak', action.data || {});
  }

  async executeSystemCommand(action) {
    if (!this.capabilities.includes('system')) {
      throw new Error('System capability not available');
    }
    // System commands stay local for security
    return { executed: true, note: 'System commands require explicit allowlisting' };
  }

  async updateState(task, results) {
    // Store results in memory via bridge
    try {
      await this.bridge.call('memory.store', {
        type: 'episodic',
        content: `Task ${task.id} completed. Results: ${JSON.stringify(results).substring(0, 500)}`,
        source: `agent-loop-${this.loopId}`,
        tags: ['task-result', `loop-${this.loopId}`],
      });
    } catch (e) {
      logger.debug(`[AgentLoop ${this.loopId}] Memory store skipped: ${e.message}`);
    }

    this.sendToMaster({
      type: 'state-update',
      data: {
        loopId: this.loopId,
        taskId: task.id,
        results,
        timestamp: Date.now()
      }
    });
  }

  reportTaskComplete(task, results, duration) {
    this.sendToMaster({
      type: 'task-complete',
      data: {
        loopId: this.loopId,
        taskId: task.id,
        results,
        duration,
        timestamp: Date.now()
      }
    });
  }

  reportTaskError(task, error) {
    this.sendToMaster({
      type: 'task-error',
      data: {
        loopId: this.loopId,
        taskId: task.id,
        error: error.message,
        timestamp: Date.now()
      }
    });
  }

  async performIdleCycle() {
    // Cognitive loops are triggered exclusively by the cron scheduler
    // (via daemon-master's _registerFallbackLoops or the TS SchedulerSystem).
    // Running them here would cause double-invocation. Instead, idle cycles
    // just yield to avoid busy-looping.
    await this.sleep(1000);
  }

  async getGmailContext() {
    try {
      return await this.bridge.call('gmail.context', {});
    } catch (e) {
      return { unread: 0, error: e.message };
    }
  }

  async getBrowserContext() {
    return { url: null };
  }

  async getSystemContext() {
    return {
      memory: process.memoryUsage(),
      uptime: process.uptime()
    };
  }

  async onShutdown() {
    this.isRunning = false;
    logger.info(`[AgentLoop ${this.loopId}] Shutting down...`);

    // Wait for current task to complete (with timeout)
    if (this.currentTaskPromise) {
      try {
        await this.withTimeout(
          this.currentTaskPromise,
          10000,
          'Task completion timeout during shutdown'
        );
      } catch (error) {
        logger.warn(`[AgentLoop ${this.loopId}] ${error.message}`);
      }
    }

    // Stop bridge proxy
    if (this.bridge) {
      await this.bridge.stop();
    }

    // Save pending tasks
    if (this.taskQueue.length > 0) {
      this.sendToMaster({
        type: 'pending-tasks',
        data: {
          loopId: this.loopId,
          tasks: this.taskQueue
        }
      });
    }

    logger.info(`[AgentLoop ${this.loopId}] Stats:`, this.stats);
  }

  async onTask(data) {
    // Handle tasks sent from master
    this.taskQueue.push(data);
    return { queued: true, position: this.taskQueue.length };
  }
}

// Initialize worker if this file is run directly
if (require.main === module) {
  const worker = new AgentLoopWorker();
  worker.initialize().catch(error => {
    logger.error('[AgentLoop] Initialization failed:', error);
    process.exit(1);
  });
}

module.exports = AgentLoopWorker;
