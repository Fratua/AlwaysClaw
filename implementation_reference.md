# Multi-Agent Orchestration - Implementation Reference
## Windows 10 OpenClaw-Inspired AI Agent System

**Supplementary Document to Technical Specification**

---

## Table of Contents

1. [Core Agent Base Class](#1-core-agent-base-class)
2. [Protocol Implementations](#2-protocol-implementations)
3. [Registry Service Implementation](#3-registry-service-implementation)
4. [Task Delegation Engine](#4-task-delegation-engine)
5. [Consensus Module](#5-consensus-module)
6. [Memory System](#6-memory-system)
7. [Configuration Files](#7-configuration-files)

---

## 1. Core Agent Base Class

```typescript
// src/shared/base/agent.base.ts

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { A2AClient } from '../protocols/a2a/client';
import { MCPClient } from '../protocols/mcp/client';
import { MessageQueue } from '../queue/message-queue';
import { Logger } from '../logging/logger';

export interface AgentConfig {
  agent_id: string;
  name: string;
  role: 'meta' | 'supervisor' | 'worker';
  capabilities: Capability[];
  endpoints: {
    a2a?: string;
    mcp?: string;
    health?: string;
  };
  dependencies?: string[];
  resource_limits?: {
    max_concurrent_tasks?: number;
    max_queue_depth?: number;
  };
}

export interface Capability {
  id: string;
  description: string;
  parameters?: Record<string, string>;
  returns?: string;
}

export abstract class AgentBase extends EventEmitter {
  protected config: AgentConfig;
  protected logger: Logger;
  protected a2aClient: A2AClient;
  protected mcpClient: MCPClient;
  protected messageQueue: MessageQueue;
  protected state: AgentState = 'registered';
  protected currentTasks: Map<string, Task> = new Map();
  protected metrics: AgentMetrics;

  constructor(config: AgentConfig) {
    super();
    this.config = config;
    this.logger = new Logger(config.name);
    this.metrics = {
      tasks_completed: 0,
      tasks_failed: 0,
      avg_response_time_ms: 0,
      error_rate: 0,
    };
  }

  // Lifecycle methods
  async initialize(): Promise<void> {
    this.logger.info(`Initializing agent ${this.config.agent_id}`);
    
    // Initialize protocol clients
    if (this.config.endpoints.a2a) {
      this.a2aClient = new A2AClient(this.config.endpoints.a2a);
    }
    if (this.config.endpoints.mcp) {
      this.mcpClient = new MCPClient(this.config.endpoints.mcp);
    }
    
    // Connect to message queue
    this.messageQueue = new MessageQueue(this.config.agent_id);
    await this.messageQueue.connect();
    
    // Subscribe to incoming messages
    await this.messageQueue.subscribe('incoming', this.handleMessage.bind(this));
    
    this.state = 'idle';
    this.emit('initialized');
  }

  async start(): Promise<void> {
    this.logger.info(`Starting agent ${this.config.agent_id}`);
    
    // Register with registry
    await this.registerWithRegistry();
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Start message processing
    this.messageQueue.startProcessing();
    
    this.state = 'idle';
    this.emit('started');
  }

  async stop(): Promise<void> {
    this.logger.info(`Stopping agent ${this.config.agent_id}`);
    
    this.state = 'terminating';
    
    // Wait for current tasks
    await this.waitForTasks();
    
    // Stop heartbeat
    this.stopHeartbeat();
    
    // Disconnect from queue
    await this.messageQueue.disconnect();
    
    // Unregister from registry
    await this.unregisterFromRegistry();
    
    this.state = 'terminated';
    this.emit('stopped');
  }

  // Abstract methods for subclasses
  abstract executeTask(task: Task): Promise<TaskResult>;
  abstract getCapabilities(): Capability[];

  // Message handling
  protected async handleMessage(message: A2AMessage): Promise<void> {
    this.logger.debug(`Received message: ${message.message_type}`, { 
      from: message.sender.agent_id 
    });

    try {
      switch (message.message_type) {
        case 'task.delegate':
          await this.handleTaskDelegation(message);
          break;
        case 'task.cancel':
          await this.handleTaskCancellation(message);
          break;
        case 'context.request':
          await this.handleContextRequest(message);
          break;
        case 'query.capability':
          await this.handleCapabilityQuery(message);
          break;
        case 'heartbeat':
          await this.handleHeartbeat(message);
          break;
        default:
          this.logger.warn(`Unknown message type: ${message.message_type}`);
      }
    } catch (error) {
      this.logger.error(`Error handling message`, error);
      await this.sendErrorResponse(message, error);
    }
  }

  protected async handleTaskDelegation(message: A2AMessage): Promise<void> {
    const { task_id, description, parameters, deadline } = message.payload;
    
    // Check if we can accept more tasks
    if (this.currentTasks.size >= (this.config.resource_limits?.max_concurrent_tasks || 5)) {
      await this.sendTaskRejection(message, 'Agent at capacity');
      return;
    }

    // Create task object
    const task: Task = {
      task_id,
      context_id: message.context_id || uuidv4(),
      type: this.inferTaskType(description),
      description,
      parameters,
      assigner_id: message.sender.agent_id,
      assignee_id: this.config.agent_id,
      delegation_chain: [message.sender.agent_id],
      status: 'in_progress',
      priority: message.payload.priority || 3,
      created_at: Date.now(),
      deadline,
      tags: [],
      estimated_effort: 1,
    };

    // Store task
    this.currentTasks.set(task_id, task);
    this.state = 'busy';

    // Send acceptance
    await this.sendTaskAcceptance(message);

    // Execute task
    try {
      const result = await this.executeTask(task);
      
      // Update task
      task.status = 'completed';
      task.completed_at = Date.now();
      task.result = result;
      
      // Send completion
      await this.sendTaskCompletion(message, result);
      
      // Update metrics
      this.metrics.tasks_completed++;
      this.updateMetrics(task);
      
    } catch (error) {
      task.status = 'failed';
      task.error = {
        code: 'EXECUTION_ERROR',
        message: error.message,
      };
      
      await this.sendTaskFailure(message, error);
      this.metrics.tasks_failed++;
    }

    // Cleanup
    this.currentTasks.delete(task_id);
    if (this.currentTasks.size === 0) {
      this.state = 'idle';
    }
  }

  // Helper methods
  protected async registerWithRegistry(): Promise<void> {
    if (!this.mcpClient) return;

    const agentCard: AgentCard = {
      agent_id: this.config.agent_id,
      name: this.config.name,
      version: '1.0.0',
      role: this.config.role,
      capabilities: this.getCapabilities(),
      endpoints: this.config.endpoints,
      status: this.state,
      metrics: this.metrics,
    };

    await this.mcpClient.callTool('registry/register', { agent_card: agentCard });
  }

  protected async unregisterFromRegistry(): Promise<void> {
    if (!this.mcpClient) return;
    await this.mcpClient.callTool('registry/unregister', { 
      agent_id: this.config.agent_id 
    });
  }

  protected startHeartbeat(): void {
    const interval = setInterval(async () => {
      if (this.state === 'terminated') {
        clearInterval(interval);
        return;
      }

      const heartbeat: HeartbeatMessage = {
        agent_id: this.config.agent_id,
        timestamp: Date.now(),
        status: this.state === 'busy' ? 'healthy' : 'healthy',
        metrics: {
          queue_depth: this.currentTasks.size,
          memory_usage_mb: process.memoryUsage().heapUsed / 1024 / 1024,
          cpu_percent: 0, // Would need actual measurement
          tasks_completed_1m: this.metrics.tasks_completed,
          error_rate_1m: this.metrics.error_rate,
        },
      };

      await this.mcpClient?.callTool('registry/heartbeat', heartbeat);
    }, 30000); // Every 30 seconds
  }

  protected stopHeartbeat(): void {
    // Interval cleared in startHeartbeat
  }

  protected async waitForTasks(timeoutMs: number = 30000): Promise<void> {
    const start = Date.now();
    while (this.currentTasks.size > 0 && Date.now() - start < timeoutMs) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  protected updateMetrics(task: Task): void {
    const duration = task.completed_at! - task.created_at;
    this.metrics.avg_response_time_ms = 
      (this.metrics.avg_response_time_ms * (this.metrics.tasks_completed - 1) + duration) 
      / this.metrics.tasks_completed;
  }

  protected inferTaskType(description: string): TaskType {
    const keywords: Record<string, TaskType> = {
      'email': 'communication',
      'send': 'communication',
      'search': 'research',
      'analyze': 'computation',
      'write': 'creative',
      'file': 'system',
      'schedule': 'coordination',
    };

    const lowerDesc = description.toLowerCase();
    for (const [keyword, type] of Object.entries(keywords)) {
      if (lowerDesc.includes(keyword)) return type;
    }

    return 'user_interaction';
  }

  // Response methods
  protected async sendTaskAcceptance(message: A2AMessage): Promise<void> {
    await this.a2aClient.send({
      recipient_id: message.sender.agent_id,
      message_type: 'task.status',
      context_id: message.context_id,
      payload: {
        task_id: message.payload.task_id,
        status: 'accepted',
      },
    });
  }

  protected async sendTaskRejection(message: A2AMessage, reason: string): Promise<void> {
    await this.a2aClient.send({
      recipient_id: message.sender.agent_id,
      message_type: 'task.status',
      context_id: message.context_id,
      payload: {
        task_id: message.payload.task_id,
        status: 'rejected',
        reason,
      },
    });
  }

  protected async sendTaskCompletion(message: A2AMessage, result: TaskResult): Promise<void> {
    await this.a2aClient.send({
      recipient_id: message.sender.agent_id,
      message_type: 'task.complete',
      context_id: message.context_id,
      payload: {
        task_id: message.payload.task_id,
        success: true,
        result,
        metrics: {
          duration_ms: Date.now() - message.timestamp,
        },
      },
    });
  }

  protected async sendTaskFailure(message: A2AMessage, error: Error): Promise<void> {
    await this.a2aClient.send({
      recipient_id: message.sender.agent_id,
      message_type: 'task.complete',
      context_id: message.context_id,
      payload: {
        task_id: message.payload.task_id,
        success: false,
        error: {
          code: 'EXECUTION_ERROR',
          message: error.message,
        },
      },
    });
  }

  protected async sendErrorResponse(message: A2AMessage, error: Error): Promise<void> {
    await this.a2aClient.send({
      recipient_id: message.sender.agent_id,
      message_type: 'error',
      context_id: message.context_id,
      payload: {
        original_message_id: message.message_id,
        error: {
          code: 'MESSAGE_HANDLING_ERROR',
          message: error.message,
        },
      },
    });
  }
}
```

---

## 2. Protocol Implementations

### 2.1 A2A Protocol Client

```typescript
// src/protocols/a2a/client.ts

import axios, { AxiosInstance } from 'axios';
import { v4 as uuidv4 } from 'uuid';

export class A2AClient {
  private http: AxiosInstance;
  private endpoint: string;
  private agentId: string;

  constructor(endpoint: string, agentId: string) {
    this.endpoint = endpoint;
    this.agentId = agentId;
    this.http = axios.create({
      baseURL: endpoint,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth interceptor
    this.http.interceptors.request.use(config => {
      config.headers['X-Agent-ID'] = this.agentId;
      config.headers['Authorization'] = `Bearer ${this.getToken()}`;
      return config;
    });
  }

  async send(message: Partial<A2AMessage>): Promise<void> {
    const fullMessage: A2AMessage = {
      message_id: uuidv4(),
      sender: { agent_id: this.agentId, role: 'agent' },
      timestamp: Date.now(),
      version: '1.0',
      metadata: { priority: 'normal' },
      ...message,
    } as A2AMessage;

    await this.http.post('/messages', fullMessage);
  }

  async sendAndWait(
    message: Partial<A2AMessage>,
    timeoutMs: number = 30000
  ): Promise<A2AMessage> {
    const correlationId = uuidv4();
    
    // Send message with correlation ID
    await this.send({
      ...message,
      correlation_id: correlationId,
    });

    // Wait for response
    return this.waitForResponse(correlationId, timeoutMs);
  }

  private async waitForResponse(
    correlationId: string,
    timeoutMs: number
  ): Promise<A2AMessage> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Response timeout for ${correlationId}`));
      }, timeoutMs);

      // Subscribe to response (implementation depends on message queue)
      const unsubscribe = messageQueue.subscribe(
        `response:${correlationId}`,
        (response: A2AMessage) => {
          clearTimeout(timeout);
          unsubscribe();
          resolve(response);
        }
      );
    });
  }

  private getToken(): string {
    // Implementation depends on auth strategy
    return process.env.AGENT_TOKEN || '';
  }
}
```

### 2.2 A2A Protocol Server

```typescript
// src/protocols/a2a/server.ts

import express, { Request, Response } from 'express';
import { EventEmitter } from 'events';

export class A2AServer extends EventEmitter {
  private app: express.Application;
  private port: number;
  private messageHandlers: Map<string, MessageHandler> = new Map();

  constructor(port: number) {
    super();
    this.port = port;
    this.app = express();
    this.app.use(express.json());
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // Receive message
    this.app.post('/messages', async (req: Request, res: Response) => {
      try {
        const message: A2AMessage = req.body;
        
        // Validate message
        if (!this.validateMessage(message)) {
          return res.status(400).json({ error: 'Invalid message format' });
        }

        // Emit for processing
        this.emit('message', message);

        // Route to specific handler if registered
        const handler = this.messageHandlers.get(message.message_type);
        if (handler) {
          await handler(message);
        }

        res.status(202).json({ received: true });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // Query capabilities
    this.app.get('/capabilities', (req: Request, res: Response) => {
      res.json({
        agent_id: process.env.AGENT_ID,
        capabilities: this.getCapabilities(),
      });
    });

    // Health check
    this.app.get('/health', (req: Request, res: Response) => {
      res.json({ status: 'healthy', timestamp: Date.now() });
    });
  }

  registerHandler(messageType: string, handler: MessageHandler): void {
    this.messageHandlers.set(messageType, handler);
  }

  start(): Promise<void> {
    return new Promise(resolve => {
      this.app.listen(this.port, () => {
        console.log(`A2A server listening on port ${this.port}`);
        resolve();
      });
    });
  }

  private validateMessage(message: unknown): message is A2AMessage {
    const m = message as A2AMessage;
    return !!(
      m.message_id &&
      m.sender &&
      m.recipient &&
      m.message_type &&
      m.timestamp
    );
  }

  private getCapabilities(): Capability[] {
    // Override in subclass or inject
    return [];
  }
}

type MessageHandler = (message: A2AMessage) => Promise<void>;
```

### 2.3 MCP Protocol Client

```typescript
// src/protocols/mcp/client.ts

import axios, { AxiosInstance } from 'axios';

export class MCPClient {
  private http: AxiosInstance;
  private endpoint: string;

  constructor(endpoint: string) {
    this.endpoint = endpoint;
    this.http = axios.create({
      baseURL: endpoint,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async listTools(): Promise<MCPTool[]> {
    const response = await this.http.post('/rpc', {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'tools/list',
    });
    return response.data.result.tools;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    const response = await this.http.post('/rpc', {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'tools/call',
      params: { name, arguments: args },
    });

    if (response.data.error) {
      throw new Error(response.data.error.message);
    }

    return response.data.result;
  }

  async listResources(): Promise<MCPResource[]> {
    const response = await this.http.post('/rpc', {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'resources/list',
    });
    return response.data.result.resources;
  }

  async readResource(uri: string): Promise<MCPResourceContent> {
    const response = await this.http.post('/rpc', {
      jsonrpc: '2.0',
      id: this.generateId(),
      method: 'resources/read',
      params: { uri },
    });
    return response.data.result;
  }

  async subscribeResource(uri: string, callback: (content: MCPResourceContent) => void): Promise<void> {
    // WebSocket or polling implementation
    const ws = new WebSocket(`${this.endpoint}/subscribe`);
    ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      if (message.uri === uri) {
        callback(message.content);
      }
    });
  }

  private generateId(): string {
    return `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

---

## 3. Registry Service Implementation

```typescript
// src/registry/registry.service.ts

import express from 'express';
import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { EventEmitter } from 'events';

export class RegistryService extends EventEmitter {
  private db: Database<sqlite3.Database>;
  private app: express.Application;
  private agents: Map<string, AgentRegistration> = new Map();
  private heartbeatTimeouts: Map<string, NodeJS.Timeout> = new Map();

  async initialize(dbPath: string): Promise<void> {
    // Initialize SQLite database
    this.db = await open({
      filename: dbPath,
      driver: sqlite3.Database,
    });

    await this.createTables();
    this.setupExpress();
  }

  private async createTables(): Promise<void> {
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS agents (
        agent_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        version TEXT,
        capabilities TEXT, -- JSON
        endpoints TEXT, -- JSON
        status TEXT DEFAULT 'registered',
        registered_at INTEGER,
        last_heartbeat INTEGER,
        metrics TEXT -- JSON
      );

      CREATE TABLE IF NOT EXISTS agent_capabilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id TEXT,
        capability_id TEXT,
        description TEXT,
        parameters TEXT, -- JSON
        FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
      );

      CREATE INDEX IF NOT EXISTS idx_capabilities ON agent_capabilities(capability_id);
      CREATE INDEX IF NOT EXISTS idx_status ON agents(status);
    `);
  }

  private setupExpress(): void {
    this.app = express();
    this.app.use(express.json());

    // MCP RPC endpoint
    this.app.post('/rpc', this.handleRPC.bind(this));

    // Agent card endpoint
    this.app.get('/agents/:agentId/card', this.getAgentCard.bind(this));

    // Health endpoint
    this.app.get('/health', (req, res) => {
      res.json({ status: 'healthy', agents: this.agents.size });
    });
  }

  private async handleRPC(req: express.Request, res: express.Response): Promise<void> {
    const { method, params, id } = req.body;

    try {
      let result;

      switch (method) {
        case 'tools/call':
          result = await this.handleToolCall(params.name, params.arguments);
          break;
        case 'tools/list':
          result = await this.listTools();
          break;
        default:
          throw new Error(`Unknown method: ${method}`);
      }

      res.json({
        jsonrpc: '2.0',
        id,
        result,
      });
    } catch (error) {
      res.json({
        jsonrpc: '2.0',
        id,
        error: {
          code: -32000,
          message: error.message,
        },
      });
    }
  }

  private async handleToolCall(name: string, args: any): Promise<any> {
    switch (name) {
      case 'registry/register':
        return this.registerAgent(args.agent_card);
      case 'registry/unregister':
        return this.unregisterAgent(args.agent_id);
      case 'registry/search':
        return this.searchAgents(args);
      case 'registry/get':
        return this.getAgent(args.agent_id);
      case 'registry/heartbeat':
        return this.handleHeartbeat(args);
      case 'registry/update_status':
        return this.updateStatus(args.agent_id, args.status, args.metrics);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  private async registerAgent(agentCard: AgentCard): Promise<{ agent_id: string }> {
    const { agent_id, name, role, version, capabilities, endpoints } = agentCard;

    // Check if already registered
    const existing = await this.db.get('SELECT * FROM agents WHERE agent_id = ?', agent_id);
    if (existing) {
      throw new Error(`Agent ${agent_id} already registered`);
    }

    // Insert agent
    await this.db.run(
      `INSERT INTO agents (agent_id, name, role, version, capabilities, endpoints, status, registered_at, last_heartbeat)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      agent_id,
      name,
      role,
      version,
      JSON.stringify(capabilities),
      JSON.stringify(endpoints),
      'idle',
      Date.now(),
      Date.now()
    );

    // Insert capabilities
    for (const cap of capabilities) {
      await this.db.run(
        `INSERT INTO agent_capabilities (agent_id, capability_id, description, parameters)
         VALUES (?, ?, ?, ?)`,
        agent_id,
        cap.id,
        cap.description,
        JSON.stringify(cap.parameters)
      );
    }

    // Store in memory
    this.agents.set(agent_id, {
      agent_card: agentCard,
      status: 'idle',
      registered_at: Date.now(),
    });

    // Start heartbeat timeout
    this.setHeartbeatTimeout(agent_id);

    this.emit('agent_registered', agentCard);

    return { agent_id };
  }

  private async unregisterAgent(agentId: string): Promise<{ success: boolean }> {
    await this.db.run('DELETE FROM agent_capabilities WHERE agent_id = ?', agentId);
    await this.db.run('DELETE FROM agents WHERE agent_id = ?', agentId);

    this.agents.delete(agentId);
    this.clearHeartbeatTimeout(agentId);

    this.emit('agent_unregistered', agentId);

    return { success: true };
  }

  private async searchAgents(params: SearchParams): Promise<{ agents: AgentCard[]; total: number }> {
    let query = 'SELECT * FROM agents WHERE 1=1';
    const args: any[] = [];

    if (params.available_only) {
      query += ' AND status = ?';
      args.push('idle');
    }

    if (params.role) {
      query += ' AND role = ?';
      args.push(params.role);
    }

    if (params.capabilities?.length) {
      query += ` AND agent_id IN (
        SELECT agent_id FROM agent_capabilities 
        WHERE capability_id IN (${params.capabilities.map(() => '?').join(',')})
        GROUP BY agent_id
        HAVING COUNT(DISTINCT capability_id) = ?
      )`;
      args.push(...params.capabilities, params.capabilities.length);
    }

    query += ' LIMIT ?';
    args.push(params.max_results || 50);

    const rows = await this.db.all(query, args);

    const agents = rows.map(row => ({
      agent_id: row.agent_id,
      name: row.name,
      role: row.role,
      version: row.version,
      capabilities: JSON.parse(row.capabilities),
      endpoints: JSON.parse(row.endpoints),
      status: row.status,
      metrics: row.metrics ? JSON.parse(row.metrics) : undefined,
    }));

    return { agents, total: agents.length };
  }

  private async handleHeartbeat(heartbeat: HeartbeatMessage): Promise<{ received: boolean }> {
    const { agent_id, metrics } = heartbeat;

    // Update last heartbeat
    await this.db.run(
      'UPDATE agents SET last_heartbeat = ?, metrics = ? WHERE agent_id = ?',
      Date.now(),
      JSON.stringify(metrics),
      agent_id
    );

    // Reset timeout
    this.setHeartbeatTimeout(agent_id);

    return { received: true };
  }

  private setHeartbeatTimeout(agentId: string): void {
    this.clearHeartbeatTimeout(agentId);

    const timeout = setTimeout(async () => {
      this.logger.warn(`Agent ${agentId} heartbeat timeout`);
      await this.updateStatus(agentId, 'unhealthy');
    }, 90000); // 90 seconds

    this.heartbeatTimeouts.set(agentId, timeout);
  }

  private clearHeartbeatTimeout(agentId: string): void {
    const timeout = this.heartbeatTimeouts.get(agentId);
    if (timeout) {
      clearTimeout(timeout);
      this.heartbeatTimeouts.delete(agentId);
    }
  }

  private async updateStatus(
    agentId: string,
    status: string,
    metrics?: AgentMetrics
  ): Promise<{ updated: boolean }> {
    await this.db.run(
      'UPDATE agents SET status = ?, metrics = ? WHERE agent_id = ?',
      status,
      metrics ? JSON.stringify(metrics) : null,
      agentId
    );

    const agent = this.agents.get(agentId);
    if (agent) {
      agent.status = status;
    }

    this.emit('status_changed', { agent_id: agentId, status });

    return { updated: true };
  }

  private async listTools(): Promise<{ tools: MCPTool[] }> {
    return {
      tools: [
        {
          name: 'registry/register',
          description: 'Register a new agent',
          inputSchema: { type: 'object', properties: { agent_card: { type: 'object' } } },
        },
        {
          name: 'registry/unregister',
          description: 'Unregister an agent',
          inputSchema: { type: 'object', properties: { agent_id: { type: 'string' } } },
        },
        {
          name: 'registry/search',
          description: 'Search for agents by capabilities',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string' },
              capabilities: { type: 'array', items: { type: 'string' } },
              role: { type: 'string' },
              available_only: { type: 'boolean' },
              max_results: { type: 'number' },
            },
          },
        },
        {
          name: 'registry/heartbeat',
          description: 'Send heartbeat to registry',
          inputSchema: { type: 'object' },
        },
      ],
    };
  }

  start(port: number): Promise<void> {
    return new Promise(resolve => {
      this.app.listen(port, () => {
        console.log(`Registry service listening on port ${port}`);
        resolve();
      });
    });
  }
}
```

---

## 4. Task Delegation Engine

```typescript
// src/delegation/assigner.ts

import { EventEmitter } from 'events';

export class TaskAssigner extends EventEmitter {
  private registry: RegistryService;
  private config: AssignerConfig;

  constructor(registry: RegistryService, config: AssignerConfig = {}) {
    super();
    this.registry = registry;
    this.config = {
      minAcceptableScore: 0.5,
      strategies: ['capability', 'load', 'performance'],
      ...config,
    };
  }

  async assignTask(task: Task): Promise<AssignmentResult> {
    this.emit('assignment_started', { task_id: task.task_id });

    try {
      // Step 1: Find candidate agents
      const candidates = await this.findCandidates(task);
      
      if (candidates.length === 0) {
        return {
          success: false,
          reason: 'no_candidates',
          message: 'No agents found matching task requirements',
        };
      }

      // Step 2: Score each candidate
      const scored = await Promise.all(
        candidates.map(async agent => ({
          agent,
          score: await this.calculateScore(agent, task),
        }))
      );

      // Step 3: Sort by score
      scored.sort((a, b) => b.score - a.score);

      // Step 4: Check if best score is acceptable
      const best = scored[0];
      if (best.score < this.config.minAcceptableScore!) {
        return {
          success: false,
          reason: 'score_too_low',
          message: `Best agent score ${best.score} below threshold ${this.config.minAcceptableScore}`,
        };
      }

      // Step 5: Attempt assignment
      const result = await this.attemptAssignment(task, best.agent);

      if (result.success) {
        this.emit('assignment_succeeded', {
          task_id: task.task_id,
          agent_id: best.agent.agent_id,
          score: best.score,
        });
      } else {
        // Try next best agent
        for (const candidate of scored.slice(1)) {
          const retry = await this.attemptAssignment(task, candidate.agent);
          if (retry.success) {
            return retry;
          }
        }

        return {
          success: false,
          reason: 'all_rejected',
          message: 'All candidate agents rejected the task',
        };
      }

      return result;
    } catch (error) {
      this.emit('assignment_failed', { task_id: task.task_id, error });
      return {
        success: false,
        reason: 'error',
        message: error.message,
      };
    }
  }

  private async findCandidates(task: Task): Promise<AgentCard[]> {
    const { agents } = await this.registry.searchAgents({
      capabilities: task.requirements.capabilities,
      available_only: true,
      max_results: 10,
    });

    return agents;
  }

  private async calculateScore(agent: AgentCard, task: Task): Promise<number> {
    let score = 0;

    // Capability match (40%)
    const capabilityScore = this.scoreCapabilityMatch(agent, task);
    score += capabilityScore * 0.40;

    // Current load (25%)
    const loadScore = await this.scoreLoad(agent);
    score += loadScore * 0.25;

    // Historical performance (20%)
    const performanceScore = agent.metrics?.success_rate || 0.5;
    score += performanceScore * 0.20;

    // Response time (10%)
    const responseTime = agent.metrics?.avg_response_time_ms || 500;
    const latencyScore = Math.max(0, 1 - responseTime / 1000);
    score += latencyScore * 0.10;

    // Affinity (5%)
    const affinityScore = await this.scoreAffinity(agent, task);
    score += affinityScore * 0.05;

    return Math.min(1, Math.max(0, score));
  }

  private scoreCapabilityMatch(agent: AgentCard, task: Task): number {
    const required = task.requirements.capabilities;
    const hasCapability = (capId: string) =>
      agent.capabilities.some(c => c.id === capId);

    const matches = required.filter(hasCapability).length;
    return matches / required.length;
  }

  private async scoreLoad(agent: AgentCard): Promise<number> {
    const metrics = agent.metrics;
    if (!metrics) return 0.5;

    // Queue depth score (0 = full, 1 = empty)
    const queueScore = Math.max(0, 1 - metrics.queue_depth / 10);

    // CPU usage score
    const cpuScore = Math.max(0, 1 - metrics.cpu_percent / 100);

    return (queueScore + cpuScore) / 2;
  }

  private async scoreAffinity(agent: AgentCard, task: Task): Promise<number> {
    // Check if agent has worked on related context before
    // This would query the memory system
    return 0.5; // Placeholder
  }

  private async attemptAssignment(
    task: Task,
    agent: AgentCard
  ): Promise<AssignmentResult> {
    try {
      const a2aClient = new A2AClient(agent.endpoints.a2a!, agent.agent_id);

      await a2aClient.send({
        recipient_id: { agent_id: agent.agent_id },
        message_type: 'task.delegate',
        context_id: task.context_id,
        payload: {
          task_id: task.task_id,
          description: task.description,
          parameters: task.parameters,
          requirements: task.requirements,
          deadline: task.deadline,
          priority: task.priority,
        },
        metadata: { priority: this.mapPriority(task.priority) },
      });

      return {
        success: true,
        assigned_to: agent.agent_id,
      };
    } catch (error) {
      return {
        success: false,
        reason: 'send_failed',
        message: error.message,
      };
    }
  }

  private mapPriority(priority: TaskPriority): 'low' | 'normal' | 'high' | 'critical' {
    switch (priority) {
      case 1: return 'low';
      case 2: return 'normal';
      case 3: return 'normal';
      case 4: return 'high';
      case 5: return 'critical';
      default: return 'normal';
    }
  }
}

interface AssignerConfig {
  minAcceptableScore?: number;
  strategies?: string[];
}

interface AssignmentResult {
  success: boolean;
  assigned_to?: string;
  reason?: string;
  message?: string;
}
```

---

## 5. Consensus Module

```typescript
// src/consensus/raft/node.ts

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

type NodeState = 'follower' | 'candidate' | 'leader';

interface LogEntry {
  index: number;
  term: number;
  command: ConsensusCommand;
  timestamp: number;
}

interface ConsensusCommand {
  type: 'DECISION' | 'CONFIG_CHANGE' | 'LEADERSHIP_TRANSFER';
  proposal?: Proposal;
  voters?: string[];
}

interface Proposal {
  id: string;
  value: unknown;
  proposer: string;
  timestamp: number;
}

export class RaftNode extends EventEmitter {
  private nodeId: string;
  private peers: string[];
  private state: NodeState = 'follower';
  private currentTerm = 0;
  private votedFor: string | null = null;
  private log: LogEntry[] = [];
  private commitIndex = 0;
  private lastApplied = 0;

  // Leader state
  private nextIndex: Map<string, number> = new Map();
  private matchIndex: Map<string, number> = new Map();

  // Timers
  private electionTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;

  // Config
  private electionTimeoutMin = 150;
  private electionTimeoutMax = 300;
  private heartbeatInterval = 50;

  constructor(nodeId: string, peers: string[]) {
    super();
    this.nodeId = nodeId;
    this.peers = peers;
  }

  start(): void {
    this.resetElectionTimer();
    this.emit('started', { node_id: this.nodeId });
  }

  stop(): void {
    this.clearTimers();
    this.emit('stopped');
  }

  // Propose a value for consensus
  async propose(value: unknown): Promise<ConsensusResult> {
    if (this.state !== 'leader') {
      // Forward to leader
      const leader = await this.findLeader();
      return this.forwardToLeader(leader, value);
    }

    const proposal: Proposal = {
      id: uuidv4(),
      value,
      proposer: this.nodeId,
      timestamp: Date.now(),
    };

    // Append to log
    const entry: LogEntry = {
      index: this.log.length + 1,
      term: this.currentTerm,
      command: { type: 'DECISION', proposal },
      timestamp: Date.now(),
    };

    this.log.push(entry);

    // Replicate to peers
    const replicated = await this.replicateLog(entry);

    if (replicated) {
      this.commitIndex = entry.index;
      this.applyLog(entry);
      return {
        success: true,
        proposal_id: proposal.id,
        value: proposal.value,
      };
    }

    return {
      success: false,
      error: 'Failed to replicate to majority',
    };
  }

  // RPC handlers
  async handleRequestVote(args: RequestVoteArgs): Promise<RequestVoteResult> {
    // Reply false if term < currentTerm
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, voteGranted: false };
    }

    // If term > currentTerm, update term and convert to follower
    if (args.term > this.currentTerm) {
      this.currentTerm = args.term;
      this.state = 'follower';
      this.votedFor = null;
    }

    // Check if we can vote for this candidate
    const canVote =
      this.votedFor === null || this.votedFor === args.candidateId;
    const logUpToDate =
      args.lastLogIndex >= this.log.length &&
      args.lastLogTerm >= (this.log[this.log.length - 1]?.term || 0);

    if (canVote && logUpToDate) {
      this.votedFor = args.candidateId;
      this.resetElectionTimer();
      return { term: this.currentTerm, voteGranted: true };
    }

    return { term: this.currentTerm, voteGranted: false };
  }

  async handleAppendEntries(args: AppendEntriesArgs): Promise<AppendEntriesResult> {
    // Reply false if term < currentTerm
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, success: false };
    }

    // If term >= currentTerm, recognize leader
    if (args.term >= this.currentTerm) {
      this.currentTerm = args.term;
      this.state = 'follower';
      this.resetElectionTimer();
    }

    // Check log consistency
    if (args.prevLogIndex > 0) {
      const prevEntry = this.log[args.prevLogIndex - 1];
      if (!prevEntry || prevEntry.term !== args.prevLogTerm) {
        return { term: this.currentTerm, success: false };
      }
    }

    // Append new entries
    for (let i = 0; i < args.entries.length; i++) {
      const entry = args.entries[i];
      const existing = this.log[entry.index - 1];

      if (existing && existing.term !== entry.term) {
        // Conflict: truncate log
        this.log = this.log.slice(0, entry.index - 1);
      }

      if (!this.log[entry.index - 1]) {
        this.log.push(entry);
      }
    }

    // Update commit index
    if (args.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(args.leaderCommit, this.log.length);
      this.applyCommitted();
    }

    return { term: this.currentTerm, success: true };
  }

  // Private methods
  private startElection(): void {
    this.state = 'candidate';
    this.currentTerm++;
    this.votedFor = this.nodeId;

    this.emit('election_started', { term: this.currentTerm });

    // Request votes from all peers
    const votesNeeded = Math.floor(this.peers.length / 2) + 1;
    let votesReceived = 1; // Vote for self

    const args: RequestVoteArgs = {
      term: this.currentTerm,
      candidateId: this.nodeId,
      lastLogIndex: this.log.length,
      lastLogTerm: this.log[this.log.length - 1]?.term || 0,
    };

    Promise.all(
      this.peers.map(peer =>
        this.sendRequestVote(peer, args).then(result => {
          if (result.voteGranted) {
            votesReceived++;
          }
          if (result.term > this.currentTerm) {
            this.currentTerm = result.term;
            this.state = 'follower';
          }
        })
      )
    ).then(() => {
      if (this.state === 'candidate' && votesReceived >= votesNeeded) {
        this.becomeLeader();
      }
    });

    this.resetElectionTimer();
  }

  private becomeLeader(): void {
    this.state = 'leader';

    // Initialize leader state
    for (const peer of this.peers) {
      this.nextIndex.set(peer, this.log.length + 1);
      this.matchIndex.set(peer, 0);
    }

    this.emit('became_leader', { term: this.currentTerm });

    // Start sending heartbeats
    this.startHeartbeat();
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeats();
    }, this.heartbeatInterval);
  }

  private async sendHeartbeats(): Promise<void> {
    for (const peer of this.peers) {
      const args: AppendEntriesArgs = {
        term: this.currentTerm,
        leaderId: this.nodeId,
        prevLogIndex: this.nextIndex.get(peer)! - 1,
        prevLogTerm: this.log[this.nextIndex.get(peer)! - 2]?.term || 0,
        entries: [],
        leaderCommit: this.commitIndex,
      };

      this.sendAppendEntries(peer, args).then(result => {
        if (!result.success) {
          // Decrement nextIndex and retry
          this.nextIndex.set(peer, this.nextIndex.get(peer)! - 1);
        }
      });
    }
  }

  private async replicateLog(entry: LogEntry): Promise<boolean> {
    const successes = await Promise.all(
      this.peers.map(async peer => {
        const args: AppendEntriesArgs = {
          term: this.currentTerm,
          leaderId: this.nodeId,
          prevLogIndex: entry.index - 1,
          prevLogTerm: this.log[entry.index - 2]?.term || 0,
          entries: [entry],
          leaderCommit: this.commitIndex,
        };

        const result = await this.sendAppendEntries(peer, args);
        return result.success;
      })
    );

    // Count successes (including self)
    const successCount = successes.filter(s => s).length + 1;
    const majority = Math.floor(this.peers.length / 2) + 1;

    return successCount >= majority;
  }

  private applyCommitted(): void {
    while (this.lastApplied < this.commitIndex) {
      this.lastApplied++;
      const entry = this.log[this.lastApplied - 1];
      this.applyLog(entry);
    }
  }

  private applyLog(entry: LogEntry): void {
    this.emit('log_applied', entry);
  }

  private resetElectionTimer(): void {
    this.clearTimers();

    const timeout =
      this.electionTimeoutMin +
      Math.random() * (this.electionTimeoutMax - this.electionTimeoutMin);

    this.electionTimer = setTimeout(() => {
      this.startElection();
    }, timeout);
  }

  private clearTimers(): void {
    if (this.electionTimer) {
      clearTimeout(this.electionTimer);
      this.electionTimer = null;
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // RPC stubs (would be implemented with actual network calls)
  private async sendRequestVote(
    peer: string,
    args: RequestVoteArgs
  ): Promise<RequestVoteResult> {
    // Implementation depends on transport
    return { term: args.term, voteGranted: true };
  }

  private async sendAppendEntries(
    peer: string,
    args: AppendEntriesArgs
  ): Promise<AppendEntriesResult> {
    // Implementation depends on transport
    return { term: args.term, success: true };
  }

  private async findLeader(): Promise<string> {
    // Query peers to find current leader
    return this.peers[0];
  }

  private async forwardToLeader(leader: string, value: unknown): Promise<ConsensusResult> {
    // Forward proposal to leader
    return { success: true, proposal_id: uuidv4(), value };
  }
}

// Types
interface RequestVoteArgs {
  term: number;
  candidateId: string;
  lastLogIndex: number;
  lastLogTerm: number;
}

interface RequestVoteResult {
  term: number;
  voteGranted: boolean;
}

interface AppendEntriesArgs {
  term: number;
  leaderId: string;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
}

interface AppendEntriesResult {
  term: number;
  success: boolean;
}

interface ConsensusResult {
  success: boolean;
  proposal_id?: string;
  value?: unknown;
  error?: string;
}
```

---

## 6. Memory System

```typescript
// src/memory/vector.store.ts

import sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import { createHash } from 'crypto';

export interface VectorEntry {
  entry_id: string;
  agent_id: string;
  context_id?: string;
  timestamp: number;
  content: string;
  embedding: number[];
  importance: number;
  memory_type: 'episodic' | 'semantic' | 'procedural';
  tags?: string[];
  ttl?: number;
}

export interface SearchResult {
  entry: VectorEntry;
  similarity: number;
}

export class VectorStore {
  private db: Database<sqlite3.Database>;
  private dimension: number;

  constructor(dimension: number = 1536) {
    this.dimension = dimension;
  }

  async initialize(dbPath: string): Promise<void> {
    this.db = await open({
      filename: dbPath,
      driver: sqlite3.Database,
    });

    await this.createTables();
  }

  private async createTables(): Promise<void> {
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS vectors (
        entry_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        context_id TEXT,
        timestamp INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        importance REAL DEFAULT 0.5,
        memory_type TEXT DEFAULT 'episodic',
        tags TEXT,
        ttl INTEGER
      );

      CREATE INDEX IF NOT EXISTS idx_agent ON vectors(agent_id);
      CREATE INDEX IF NOT EXISTS idx_context ON vectors(context_id);
      CREATE INDEX IF NOT EXISTS idx_timestamp ON vectors(timestamp);
      CREATE INDEX IF NOT EXISTS idx_memory_type ON vectors(memory_type);
    `);

    // Note: HNSW index would be created via extension
    // CREATE INDEX idx_embedding ON vectors USING hnsw(embedding);
  }

  async store(entry: VectorEntry): Promise<void> {
    // Validate embedding dimension
    if (entry.embedding.length !== this.dimension) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimension}, got ${entry.embedding.length}`
      );
    }

    await this.db.run(
      `INSERT INTO vectors (entry_id, agent_id, context_id, timestamp, content, embedding, importance, memory_type, tags, ttl)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      entry.entry_id,
      entry.agent_id,
      entry.context_id || null,
      entry.timestamp,
      entry.content,
      Buffer.from(new Float32Array(entry.embedding).buffer),
      entry.importance,
      entry.memory_type,
      entry.tags ? JSON.stringify(entry.tags) : null,
      entry.ttl || null
    );
  }

  async search(
    query: number[],
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    const {
      limit = 10,
      threshold = 0.7,
      agent_filter,
      memory_type,
      tags,
    } = options;

    // Build query
    let sql = 'SELECT * FROM vectors WHERE 1=1';
    const params: any[] = [];

    if (agent_filter?.length) {
      sql += ` AND agent_id IN (${agent_filter.map(() => '?').join(',')})`;
      params.push(...agent_filter);
    }

    if (memory_type?.length) {
      sql += ` AND memory_type IN (${memory_type.map(() => '?').join(',')})`;
      params.push(...memory_type);
    }

    if (tags?.length) {
      sql += ` AND (${tags.map(() => "tags LIKE ?").join(' OR ')})`;
      params.push(...tags.map(t => `%${t}%`));
    }

    // For HNSW, we would use:
    // sql += ' ORDER BY embedding <-> ? LIMIT ?';
    // params.push(Buffer.from(new Float32Array(query).buffer), limit);

    // Without HNSW, do brute force search
    const rows = await this.db.all(sql, params);

    // Calculate similarities
    const results: SearchResult[] = rows.map(row => {
      const embedding = Array.from(new Float32Array(row.embedding.buffer));
      const similarity = this.cosineSimilarity(query, embedding);

      return {
        entry: {
          entry_id: row.entry_id,
          agent_id: row.agent_id,
          context_id: row.context_id,
          timestamp: row.timestamp,
          content: row.content,
          embedding,
          importance: row.importance,
          memory_type: row.memory_type,
          tags: row.tags ? JSON.parse(row.tags) : undefined,
          ttl: row.ttl,
        },
        similarity,
      };
    });

    // Filter by threshold and sort
    return results
      .filter(r => r.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  async getByContext(contextId: string, agentId?: string): Promise<VectorEntry[]> {
    let sql = 'SELECT * FROM vectors WHERE context_id = ?';
    const params: any[] = [contextId];

    if (agentId) {
      sql += ' AND agent_id = ?';
      params.push(agentId);
    }

    sql += ' ORDER BY timestamp DESC';

    const rows = await this.db.all(sql, params);

    return rows.map(row => ({
      entry_id: row.entry_id,
      agent_id: row.agent_id,
      context_id: row.context_id,
      timestamp: row.timestamp,
      content: row.content,
      embedding: Array.from(new Float32Array(row.embedding.buffer)),
      importance: row.importance,
      memory_type: row.memory_type,
      tags: row.tags ? JSON.parse(row.tags) : undefined,
      ttl: row.ttl,
    }));
  }

  async getRecent(agentId: string, limit: number = 10): Promise<VectorEntry[]> {
    const rows = await this.db.all(
      'SELECT * FROM vectors WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?',
      agentId,
      limit
    );

    return rows.map(row => ({
      entry_id: row.entry_id,
      agent_id: row.agent_id,
      context_id: row.context_id,
      timestamp: row.timestamp,
      content: row.content,
      embedding: Array.from(new Float32Array(row.embedding.buffer)),
      importance: row.importance,
      memory_type: row.memory_type,
      tags: row.tags ? JSON.parse(row.tags) : undefined,
      ttl: row.ttl,
    }));
  }

  async deleteExpired(): Promise<number> {
    const result = await this.db.run(
      'DELETE FROM vectors WHERE ttl IS NOT NULL AND ttl < ?',
      Date.now()
    );
    return result.changes || 0;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

interface SearchOptions {
  limit?: number;
  threshold?: number;
  agent_filter?: string[];
  memory_type?: ('episodic' | 'semantic' | 'procedural')[];
  tags?: string[];
}
```

---

## 7. Configuration Files

### 7.1 Agent Configuration

```yaml
# config/agents.yaml
agents:
  # Meta Agent
  - id: AG-001
    name: Prime
    role: meta
    description: Strategic orchestrator and global coordinator
    capabilities:
      - id: strategic_planning
        description: Create and manage global execution plans
      - id: global_orchestration
        description: Coordinate all agent activities
      - id: consensus_coordination
        description: Facilitate consensus among agents
      - id: conflict_resolution
        description: Resolve inter-agent conflicts
    endpoints:
      a2a: http://localhost:8001/a2a
      mcp: http://localhost:8001/mcp
      health: http://localhost:8001/health
    resources:
      memory_mb: 1024
      cpu_cores: 2
    auto_start: true
    priority: critical
    dependencies: []

  # Supervisors
  - id: AG-002
    name: CommSupervisor
    role: supervisor
    description: Manages all communication channels
    capabilities:
      - id: communication_routing
        description: Route messages between channels
      - id: channel_management
        description: Manage Gmail, Twilio, TTS, STT
      - id: message_coordination
        description: Coordinate multi-channel messages
    endpoints:
      a2a: http://localhost:8002/a2a
    auto_start: true
    priority: high
    dependencies:
      - AG-001
    workers:
      - AG-005
      - AG-007
      - AG-008
      - AG-009

  - id: AG-003
    name: TaskSupervisor
    role: supervisor
    description: Manages task decomposition and assignment
    capabilities:
      - id: task_decomposition
        description: Break complex tasks into subtasks
      - id: worker_assignment
        description: Assign tasks to appropriate workers
      - id: progress_monitoring
        description: Track task execution progress
    endpoints:
      a2a: http://localhost:8003/a2a
    auto_start: true
    priority: high
    dependencies:
      - AG-001
    workers:
      - AG-010
      - AG-011
      - AG-013

  - id: AG-004
    name: SystemSupervisor
    role: supervisor
    description: Monitors system health and resources
    capabilities:
      - id: health_monitoring
        description: Monitor agent health
      - id: resource_management
        description: Manage system resources
      - id: auto_recovery
        description: Automatic failure recovery
    endpoints:
      a2a: http://localhost:8004/a2a
    auto_start: true
    priority: high
    dependencies:
      - AG-001
    workers:
      - AG-012
      - AG-014
      - AG-015

  # Workers - Communication
  - id: AG-005
    name: GmailWorker
    role: worker
    description: Handles Gmail operations
    capabilities:
      - id: email.read
        description: Read emails from inbox
        parameters:
          query: string
          max_results: number
        returns: Email[]
      - id: email.send
        description: Send emails
        parameters:
          to: string
          subject: string
          body: string
          attachments: string[]
        returns: MessageId
      - id: email.filter
        description: Filter and label emails
        parameters:
          query: string
          label: string
        returns: number
    endpoints:
      a2a: http://localhost:8005/a2a
    auto_start: true
    priority: normal
    dependencies:
      - AG-012
    resource_limits:
      max_concurrent_tasks: 5
      max_queue_depth: 20

  - id: AG-006
    name: BrowserWorker
    role: worker
    description: Controls browser automation
    capabilities:
      - id: browser.navigate
        description: Navigate to URL
      - id: browser.click
        description: Click element
      - id: browser.type
        description: Type into input
      - id: browser.extract
        description: Extract data from page
      - id: browser.screenshot
        description: Take screenshot
    endpoints:
      a2a: http://localhost:8006/a2a
    auto_start: true
    priority: normal
    dependencies:
      - AG-012
    resource_limits:
      max_concurrent_tasks: 3
      max_queue_depth: 10

  - id: AG-007
    name: TwilioWorker
    role: worker
    description: Handles voice calls and SMS
    capabilities:
      - id: voice.call
        description: Make voice calls
      - id: voice.hangup
        description: End voice calls
      - id: sms.send
        description: Send SMS messages
      - id: sms.receive
        description: Receive SMS messages
      - id: ivr.navigate
        description: Navigate IVR systems
    endpoints:
      a2a: http://localhost:8007/a2a
    auto_start: true
    priority: normal
    dependencies:
      - AG-012
    resource_limits:
      max_concurrent_tasks: 10
      max_queue_depth: 50

  - id: AG-008
    name: TTSWorker
    role: worker
    description: Text-to-speech synthesis
    capabilities:
      - id: tts.synthesize
        description: Convert text to speech
        parameters:
          text: string
          voice: string
          speed: number
        returns: AudioBuffer
      - id: tts.list_voices
        description: List available voices
        returns: Voice[]
    endpoints:
      a2a: http://localhost:8008/a2a
    auto_start: true
    priority: normal
    dependencies: []
    resource_limits:
      max_concurrent_tasks: 5
      max_queue_depth: 20

  - id: AG-009
    name: STTWorker
    role: worker
    description: Speech-to-text transcription
    capabilities:
      - id: stt.transcribe
        description: Transcribe audio to text
        parameters:
          audio: Buffer
          language: string
        returns: Transcription
      - id: stt.wake_word
        description: Listen for wake word
        parameters:
          wake_word: string
        returns: boolean
    endpoints:
      a2a: http://localhost:8009/a2a
    auto_start: true
    priority: normal
    dependencies: []
    resource_limits:
      max_concurrent_tasks: 3
      max_queue_depth: 10

  # Workers - System
  - id: AG-010
    name: FileWorker
    role: worker
    description: File system operations
    capabilities:
      - id: file.read
        description: Read file contents
      - id: file.write
        description: Write file contents
      - id: file.delete
        description: Delete files
      - id: file.list
        description: List directory contents
      - id: file.search
        description: Search for files
    endpoints:
      a2a: http://localhost:8010/a2a
    auto_start: true
    priority: normal
    dependencies: []
    resource_limits:
      max_concurrent_tasks: 10
      max_queue_depth: 100

  - id: AG-011
    name: ProcessWorker
    role: worker
    description: Process and system management
    capabilities:
      - id: process.start
        description: Start a process
      - id: process.stop
        description: Stop a process
      - id: process.list
        description: List running processes
      - id: system.command
        description: Execute system command
      - id: system.info
        description: Get system information
    endpoints:
      a2a: http://localhost:8011/a2a
    auto_start: true
    priority: normal
    dependencies: []
    resource_limits:
      max_concurrent_tasks: 5
      max_queue_depth: 20

  - id: AG-012
    name: MemoryWorker
    role: worker
    description: Memory and context management
    capabilities:
      - id: memory.store
        description: Store memory entry
      - id: memory.retrieve
        description: Retrieve memories by query
      - id: memory.search
        description: Semantic memory search
      - id: context.create
        description: Create shared context
      - id: context.share
        description: Share context with agent
    endpoints:
      a2a: http://localhost:8012/a2a
    auto_start: true
    priority: high
    dependencies: []
    resource_limits:
      max_concurrent_tasks: 20
      max_queue_depth: 200

  - id: AG-013
    name: SchedulerWorker
    role: worker
    description: Task scheduling and cron jobs
    capabilities:
      - id: schedule.create
        description: Create scheduled task
      - id: schedule.delete
        description: Delete scheduled task
      - id: schedule.list
        description: List scheduled tasks
      - id: cron.parse
        description: Parse cron expression
      - id: job.execute
        description: Execute scheduled job
    endpoints:
      a2a: http://localhost:8013/a2a
    auto_start: true
    priority: normal
    dependencies:
      - AG-012
    resource_limits:
      max_concurrent_tasks: 50
      max_queue_depth: 500

  - id: AG-014
    name: IdentityWorker
    role: worker
    description: User identity and preferences
    capabilities:
      - id: user.get_profile
        description: Get user profile
      - id: user.update_profile
        description: Update user profile
      - id: user.get_preferences
        description: Get user preferences
      - id: user.set_preferences
        description: Set user preferences
      - id: user.authenticate
        description: Authenticate user
    endpoints:
      a2a: http://localhost:8014/a2a
    auto_start: true
    priority: high
    dependencies:
      - AG-012
    resource_limits:
      max_concurrent_tasks: 10
      max_queue_depth: 50

  - id: AG-015
    name: SoulWorker
    role: worker
    description: Personality and emotional expression
    capabilities:
      - id: soul.express
        description: Express personality trait
      - id: soul.adapt_tone
        description: Adapt tone to context
      - id: soul.generate_greeting
        description: Generate personalized greeting
      - id: soul.maintain_continuity
        description: Maintain personality continuity
    endpoints:
      a2a: http://localhost:8015/a2a
    auto_start: true
    priority: normal
    dependencies:
      - AG-012
      - AG-014
    resource_limits:
      max_concurrent_tasks: 5
      max_queue_depth: 20
```

### 7.2 System Configuration

```yaml
# config/system.yaml
system:
  name: "OpenClaw Multi-Agent System"
  version: "1.0.0"
  environment: "production"

  # Gateway Configuration
gateway:
    port: 8080
    adapters:
      gmail:
        enabled: true
        polling_interval: 30
        credentials_path: "./secrets/gmail.json"
      browser:
        enabled: true
        headless: false
        default_viewport:
          width: 1920
          height: 1080
      twilio:
        enabled: true
        account_sid: "${TWILIO_ACCOUNT_SID}"
        auth_token: "${TWILIO_AUTH_TOKEN}"
        from_number: "${TWILIO_FROM_NUMBER}"
      tts:
        enabled: true
        provider: "elevenlabs"
        api_key: "${ELEVENLABS_API_KEY}"
        default_voice: "pNInz6obpgDQGcFmaJgB"
      stt:
        enabled: true
        provider: "whisper"
        api_key: "${OPENAI_API_KEY}"
        model: "whisper-1"
        wake_word: "Hey Claw"

  # Registry Configuration
registry:
    storage:
      type: sqlite
      path: "./data/registry.db"
    heartbeat:
      interval_seconds: 30
      timeout_seconds: 90
    discovery:
      cache_ttl_seconds: 60
      max_results: 50

  # Message Queue Configuration
queue:
    provider: redis
    url: "redis://localhost:6379"
    prefix: "openclaw:"
    retry_policy:
      attempts: 3
      backoff:
        type: exponential
        delay: 1000

  # Memory System Configuration
memory:
    vector_store:
      type: sqlite
      path: "./data/vectors.db"
      dimension: 1536
      hnsw:
        enabled: true
        m: 16
        ef_construction: 200
    synchronization:
      topology: hub-spoke
      hub: AG-001
      interval_seconds: 5

  # Consensus Configuration
consensus:
    algorithm: raft
    raft:
      election_timeout_min: 150
      election_timeout_max: 300
      heartbeat_interval: 50
    voting:
      default_strategy: weighted
      min_participation: 0.5
      timeout_seconds: 30

  # Load Balancing Configuration
load_balancer:
    strategy: adaptive
    health_check:
      interval_seconds: 10
      timeout_seconds: 5
    backpressure:
      enabled: true
      thresholds:
        critical: 100
        high: 50
        medium: 20

  # Conflict Resolution Configuration
conflict_resolution:
    default_strategy: authority
    authority_chain:
      - AG-001
      - AG-002
      - AG-003
      - AG-004
    timeout_seconds: 60

  # Logging Configuration
logging:
    level: info
    format: json
    outputs:
      - type: console
      - type: file
        path: "./logs/openclaw.log"
        rotation:
          max_size: "100MB"
          max_files: 10

  # Security Configuration
security:
    authentication:
      type: jwt
      secret: "${JWT_SECRET}"
      expiry: "24h"
    authorization:
      enabled: true
      rbac:
        policies:
          - role: meta
            permissions: ["*"]
          - role: supervisor
            permissions: ["delegate", "monitor", "resolve"]
          - role: worker
            permissions: ["execute", "report"]
    encryption:
      tls:
        enabled: true
        cert_path: "./certs/server.crt"
        key_path: "./certs/server.key"
```

---

## Document Information

- **Type**: Implementation Reference
- **Version**: 1.0
- **Related Documents**: 
  - `multi_agent_orchestration_architecture.md` (Main Specification)

---

*End of Implementation Reference*
