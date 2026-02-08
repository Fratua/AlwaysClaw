/**
 * Core Agent Runtime - Implementation Templates
 * Windows 10 OpenClaw-Inspired AI Agent System
 * 
 * This file contains concrete TypeScript implementation templates
 * for the core components defined in the architecture specification.
 */

// ============================================================================
// SECTION 1: EVENT SYSTEM
// ============================================================================

// events/EventBus.ts
import { EventEmitter } from 'events';

export type EventType = 
  | 'INBOUND_MESSAGE'
  | 'TOOL_EXECUTION'
  | 'TOOL_RESULT'
  | 'AGENT_RESPONSE'
  | 'CRON_TRIGGER'
  | 'HEARTBEAT'
  | 'MEMORY_UPDATE'
  | 'ERROR';

export interface BaseEvent {
  id: string;
  timestamp: Date;
  type: EventType;
  source: string;
  correlationId: string;
}

export class EventBus {
  private emitter: EventEmitter;
  private handlers: Map<EventType, Set<Function>>;

  constructor() {
    this.emitter = new EventEmitter();
    this.handlers = new Map();
    this.emitter.setMaxListeners(100);
  }

  subscribe<T extends BaseEvent>(
    eventType: EventType,
    handler: (event: T) => Promise<void> | void
  ): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    const wrappedHandler = async (event: T) => {
      try {
        await handler(event);
      } catch (error) {
        console.error(`Error handling event ${eventType}:`, error);
      }
    };

    this.emitter.on(eventType, wrappedHandler);

    // Return unsubscribe function
    return () => {
      this.emitter.off(eventType, wrappedHandler);
      this.handlers.get(eventType)?.delete(handler);
    };
  }

  async publish<T extends BaseEvent>(event: T): Promise<void> {
    this.emitter.emit(event.type, event);
  }

  async publishAndWait<T extends BaseEvent>(event: T, timeout: number = 30000): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Event ${event.type} timed out after ${timeout}ms`));
      }, timeout);

      const handler = (result: any) => {
        clearTimeout(timer);
        resolve(result);
      };

      this.emitter.once(`${event.type}:${event.id}:response`, handler);
      this.emitter.emit(event.type, { ...event, _responseHandler: handler });
    });
  }
}

// ============================================================================
// SECTION 2: LLM SERVICE
// ============================================================================

// llm/LLMService.ts
export interface LLMConfig {
  provider: 'openai' | 'anthropic' | 'ollama';
  model: string;
  apiKey?: string;
  baseUrl?: string;
  temperature: number;
  maxTokens: number;
  thinkingMode?: 'low' | 'medium' | 'high' | 'extra-high';
}

export interface LLMResponse {
  text: string;
  tokens: {
    prompt: number;
    completion: number;
    total: number;
  };
  model: string;
  finishReason: string;
}

export interface LLMStreamChunk {
  text: string;
  isComplete: boolean;
}

export abstract class LLMProvider {
  constructor(protected config: LLMConfig) {}
  
  abstract generate(prompt: string, options?: Partial<LLMConfig>): Promise<LLMResponse>;
  abstract generateStructured<T>(prompt: string, schema: object): Promise<T>;
  abstract stream(prompt: string): AsyncGenerator<LLMStreamChunk>;
  abstract countTokens(text: string): number;
}

export class GPT52Provider extends LLMProvider {
  private client: any; // OpenAI client

  constructor(config: LLMConfig) {
    super(config);
    // Initialize OpenAI client with GPT-5.2
  }

  async generate(prompt: string, options?: Partial<LLMConfig>): Promise<LLMResponse> {
    const response = await this.client.chat.completions.create({
      model: this.config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: options?.temperature ?? this.config.temperature,
      max_tokens: options?.maxTokens ?? this.config.maxTokens,
      thinking: {
        type: this.config.thinkingMode === 'extra-high' ? 'extended' : 'enabled'
      }
    });

    return {
      text: response.choices[0].message.content,
      tokens: {
        prompt: response.usage.prompt_tokens,
        completion: response.usage.completion_tokens,
        total: response.usage.total_tokens
      },
      model: response.model,
      finishReason: response.choices[0].finish_reason
    };
  }

  async generateStructured<T>(prompt: string, schema: object): Promise<T> {
    const response = await this.client.chat.completions.create({
      model: this.config.model,
      messages: [{ role: 'user', content: prompt }],
      response_format: { type: 'json_object' },
      temperature: 0.1 // Low temperature for structured output
    });

    return JSON.parse(response.choices[0].message.content) as T;
  }

  async *stream(prompt: string): AsyncGenerator<LLMStreamChunk> {
    const stream = await this.client.chat.completions.create({
      model: this.config.model,
      messages: [{ role: 'user', content: prompt }],
      stream: true
    });

    for await (const chunk of stream) {
      yield {
        text: chunk.choices[0]?.delta?.content || '',
        isComplete: chunk.choices[0]?.finish_reason === 'stop'
      };
    }
  }

  countTokens(text: string): number {
    // Use tiktoken or similar for accurate token counting
    return Math.ceil(text.length / 4); // Rough estimate
  }
}

export class LLMService {
  private provider: LLMProvider;
  private promptBuilder: PromptBuilder;

  constructor(config: LLMConfig) {
    this.provider = this.createProvider(config);
    this.promptBuilder = new PromptBuilder();
  }

  private createProvider(config: LLMConfig): LLMProvider {
    switch (config.provider) {
      case 'openai':
        return new GPT52Provider(config);
      // Add other providers
      default:
        throw new Error(`Unknown provider: ${config.provider}`);
    }
  }

  async generate(prompt: string, context?: any): Promise<LLMResponse> {
    const fullPrompt = this.promptBuilder.build(prompt, context);
    return this.provider.generate(fullPrompt);
  }

  async generateStructured<T>(prompt: string, schema: object, context?: any): Promise<T> {
    const fullPrompt = this.promptBuilder.build(prompt, context);
    return this.provider.generateStructured<T>(fullPrompt, schema);
  }

  async *stream(prompt: string, context?: any): AsyncGenerator<LLMStreamChunk> {
    const fullPrompt = this.promptBuilder.build(prompt, context);
    yield* this.provider.stream(fullPrompt);
  }
}

// llm/PromptBuilder.ts
export class PromptBuilder {
  build(prompt: string, context?: any): string {
    const parts: string[] = [];

    // Add system context if available
    if (context?.systemPrompt) {
      parts.push(context.systemPrompt);
    }

    // Add user context
    if (context?.userContext) {
      parts.push(`User Context: ${JSON.stringify(context.userContext)}`);
    }

    // Add session history
    if (context?.history?.length > 0) {
      parts.push('Conversation History:');
      context.history.forEach((msg: any) => {
        parts.push(`${msg.role}: ${msg.content}`);
      });
    }

    // Add the actual prompt
    parts.push(`User Request: ${prompt}`);

    return parts.join('\n\n');
  }
}

// ============================================================================
// SECTION 3: MEMORY SYSTEM
// ============================================================================

// memory/MemoryManager.ts
export interface MemoryConfig {
  sessionStore: 'memory' | 'redis';
  longTermPath: string;
  vectorStorePath: string;
  embeddingModel: string;
  maxContextMessages: number;
}

export interface SessionMemory {
  sessionId: string;
  messages: Message[];
  context: Record<string, any>;
  createdAt: Date;
  lastActivity: Date;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface MemoryEntry {
  key: string;
  value: any;
  type: 'fact' | 'event' | 'preference' | 'skill';
  timestamp: Date;
  accessCount: number;
}

export interface VectorEntry {
  id: string;
  text: string;
  embedding: number[];
  metadata: Record<string, any>;
}

export class MemoryManager {
  private sessions: Map<string, SessionMemory>;
  private longTermStore: Map<string, MemoryEntry>;
  private vectorStore: VectorStore;
  private config: MemoryConfig;

  constructor(config: MemoryConfig) {
    this.config = config;
    this.sessions = new Map();
    this.longTermStore = new Map();
    this.vectorStore = new VectorStore(config.vectorStorePath);
  }

  async initialize(): Promise<void> {
    await this.vectorStore.initialize();
    await this.loadLongTermMemory();
  }

  // Session Memory Methods
  createSession(sessionId: string, userId: string, channel: string): SessionMemory {
    const session: SessionMemory = {
      sessionId,
      messages: [],
      context: { userId, channel },
      createdAt: new Date(),
      lastActivity: new Date()
    };
    this.sessions.set(sessionId, session);
    return session;
  }

  getSession(sessionId: string): SessionMemory | undefined {
    return this.sessions.get(sessionId);
  }

  addMessage(sessionId: string, message: Omit<Message, 'id' | 'timestamp'>): void {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);

    const newMessage: Message = {
      ...message,
      id: this.generateId(),
      timestamp: new Date()
    };

    session.messages.push(newMessage);
    session.lastActivity = new Date();

    // Trim to max context
    if (session.messages.length > this.config.maxContextMessages) {
      session.messages = session.messages.slice(-this.config.maxContextMessages);
    }
  }

  getContext(sessionId: string, limit: number = 10): Message[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];
    return session.messages.slice(-limit);
  }

  updateSessionContext(sessionId: string, updates: Record<string, any>): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.context = { ...session.context, ...updates };
      session.lastActivity = new Date();
    }
  }

  // Long-term Memory Methods
  async store(key: string, value: any, type: MemoryEntry['type'] = 'fact'): Promise<void> {
    const entry: MemoryEntry = {
      key,
      value,
      type,
      timestamp: new Date(),
      accessCount: 0
    };
    this.longTermStore.set(key, entry);
    await this.persistLongTermMemory();
  }

  retrieve(key: string): any | undefined {
    const entry = this.longTermStore.get(key);
    if (entry) {
      entry.accessCount++;
      return entry.value;
    }
    return undefined;
  }

  async search(query: string, limit: number = 5): Promise<MemoryEntry[]> {
    // Simple text search for now
    const results: MemoryEntry[] = [];
    for (const entry of this.longTermStore.values()) {
      if (entry.key.includes(query) || JSON.stringify(entry.value).includes(query)) {
        results.push(entry);
      }
    }
    return results.slice(0, limit);
  }

  // Vector Memory Methods
  async storeVector(text: string, metadata: Record<string, any>): Promise<void> {
    const embedding = await this.generateEmbedding(text);
    await this.vectorStore.add({
      id: this.generateId(),
      text,
      embedding,
      metadata
    });
  }

  async searchSimilar(query: string, limit: number = 5): Promise<VectorEntry[]> {
    const embedding = await this.generateEmbedding(query);
    return this.vectorStore.search(embedding, limit);
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    // Use embedding model API
    // This is a placeholder - implement with actual embedding service
    return new Array(1536).fill(0).map(() => Math.random() - 0.5);
  }

  private async loadLongTermMemory(): Promise<void> {
    // Load from file system
    try {
      const fs = await import('fs/promises');
      const data = await fs.readFile(`${this.config.longTermPath}/memory.json`, 'utf-8');
      const entries: MemoryEntry[] = JSON.parse(data);
      for (const entry of entries) {
        this.longTermStore.set(entry.key, entry);
      }
    } catch (error) {
      // File doesn't exist yet, start fresh
    }
  }

  private async persistLongTermMemory(): Promise<void> {
    const fs = await import('fs/promises');
    const entries = Array.from(this.longTermStore.values());
    await fs.mkdir(this.config.longTermPath, { recursive: true });
    await fs.writeFile(
      `${this.config.longTermPath}/memory.json`,
      JSON.stringify(entries, null, 2)
    );
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async close(): Promise<void> {
    await this.persistLongTermMemory();
    await this.vectorStore.close();
  }
}

// memory/VectorStore.ts (simplified)
export class VectorStore {
  private entries: VectorEntry[] = [];
  private path: string;

  constructor(path: string) {
    this.path = path;
  }

  async initialize(): Promise<void> {
    // Initialize vector database (ChromaDB, LanceDB, etc.)
  }

  async add(entry: VectorEntry): Promise<void> {
    this.entries.push(entry);
  }

  async search(embedding: number[], limit: number): Promise<VectorEntry[]> {
    // Calculate cosine similarity
    const scored = this.entries.map(entry => ({
      entry,
      score: this.cosineSimilarity(embedding, entry.embedding)
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(s => s.entry);
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

  async close(): Promise<void> {
    // Persist if needed
  }
}

// ============================================================================
// SECTION 4: TOOL SYSTEM
// ============================================================================

// tools/ToolRegistry.ts
export interface Tool {
  name: string;
  description: string;
  parameters: ToolParameter[];
  requiredPermissions: PermissionLevel[];
  execute(params: Record<string, any>, context: ToolContext): Promise<ToolResult>;
}

export interface ToolParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description: string;
  required: boolean;
  default?: any;
}

export interface ToolContext {
  sessionId: string;
  userId: string;
  workspacePath: string;
  permissions: PermissionLevel[];
}

export interface ToolResult {
  success: boolean;
  data?: any;
  error?: string;
  metadata?: {
    executionTime: number;
    tokensUsed?: number;
  };
}

export type PermissionLevel = 
  | 'read-only'
  | 'file-operations'
  | 'network-access'
  | 'system-commands'
  | 'full-access';

export class ToolRegistry {
  private tools: Map<string, Tool> = new Map();

  register(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  unregister(toolName: string): void {
    this.tools.delete(toolName);
  }

  get(toolName: string): Tool | undefined {
    return this.tools.get(toolName);
  }

  getAll(): Tool[] {
    return Array.from(this.tools.values());
  }

  getAvailableForPermissions(permissions: PermissionLevel[]): Tool[] {
    return this.getAll().filter(tool => 
      tool.requiredPermissions.every(p => permissions.includes(p))
    );
  }

  async execute(
    toolName: string, 
    params: Record<string, any>, 
    context: ToolContext
  ): Promise<ToolResult> {
    const tool = this.tools.get(toolName);
    if (!tool) {
      return { success: false, error: `Tool ${toolName} not found` };
    }

    // Check permissions
    const hasPermissions = tool.requiredPermissions.every(p => 
      context.permissions.includes(p)
    );
    if (!hasPermissions) {
      return { 
        success: false, 
        error: `Insufficient permissions for tool ${toolName}` 
      };
    }

    const startTime = Date.now();
    try {
      const result = await tool.execute(params, context);
      return {
        ...result,
        metadata: {
          ...result.metadata,
          executionTime: Date.now() - startTime
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        metadata: { executionTime: Date.now() - startTime }
      };
    }
  }
}

// Example Tool Implementation
export class FileSystemTool implements Tool {
  name = 'file_system';
  description = 'Read and write files in the workspace';
  parameters = [
    {
      name: 'operation',
      type: 'string',
      description: 'Operation to perform: read, write, list, delete',
      required: true
    },
    {
      name: 'path',
      type: 'string',
      description: 'File or directory path (relative to workspace)',
      required: true
    },
    {
      name: 'content',
      type: 'string',
      description: 'Content to write (for write operation)',
      required: false
    }
  ];
  requiredPermissions: PermissionLevel[] = ['file-operations'];

  async execute(params: Record<string, any>, context: ToolContext): Promise<ToolResult> {
    const fs = await import('fs/promises');
    const path = await import('path');
    
    const fullPath = path.join(context.workspacePath, params.path);
    
    // Security: Ensure path is within workspace
    if (!fullPath.startsWith(context.workspacePath)) {
      return { success: false, error: 'Path outside workspace not allowed' };
    }

    try {
      switch (params.operation) {
        case 'read':
          const content = await fs.readFile(fullPath, 'utf-8');
          return { success: true, data: { content } };
        
        case 'write':
          await fs.mkdir(path.dirname(fullPath), { recursive: true });
          await fs.writeFile(fullPath, params.content);
          return { success: true, data: { written: true } };
        
        case 'list':
          const entries = await fs.readdir(fullPath, { withFileTypes: true });
          return { 
            success: true, 
            data: { 
              entries: entries.map(e => ({
                name: e.name,
                isDirectory: e.isDirectory()
              }))
            } 
          };
        
        case 'delete':
          await fs.rm(fullPath, { recursive: true });
          return { success: true, data: { deleted: true } };
        
        default:
          return { success: false, error: `Unknown operation: ${params.operation}` };
      }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : String(error) 
      };
    }
  }
}

// ============================================================================
// SECTION 5: AGENT CORE
// ============================================================================

// core/intent/IntentParser.ts
export interface ParsedIntent {
  primaryIntent: IntentType;
  confidence: number;
  entities: Entity[];
  sentiment: 'positive' | 'neutral' | 'negative';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  expectedOutcome: string;
  suggestedLoop: AgenticLoopType;
}

export type IntentType = 
  | 'QUERY'
  | 'ACTION'
  | 'CONVERSATION'
  | 'DEBUG'
  | 'RESEARCH'
  | 'PLANNING'
  | 'SYSTEM'
  | 'LEARNING'
  | 'META'
  | 'UNKNOWN';

export interface Entity {
  type: string;
  value: string;
  start: number;
  end: number;
}

export type AgenticLoopType = 
  | 'ralph'
  | 'research'
  | 'discovery'
  | 'bug-finder'
  | 'debugging'
  | 'end-to-end'
  | 'meta-cognition'
  | 'exploration'
  | 'self-driven'
  | 'self-learning'
  | 'self-updating'
  | 'self-upgrading'
  | 'planning'
  | 'context-engineering'
  | 'context-prompt-engineering';

export class IntentParser {
  constructor(private llm: LLMService) {}

  async parse(message: string, context: any): Promise<ParsedIntent> {
    const prompt = `
Analyze the following user message and extract intent information.

Message: "${message}"

Context:
- Recent messages: ${context.recentMessages?.length || 0}
- User preferences: ${JSON.stringify(context.userPreferences || {})}

Respond with a JSON object matching this schema:
{
  "primaryIntent": "QUERY|ACTION|CONVERSATION|DEBUG|RESEARCH|PLANNING|SYSTEM|LEARNING|META|UNKNOWN",
  "confidence": 0.0-1.0,
  "entities": [{"type": "string", "value": "string", "start": 0, "end": 0}],
  "sentiment": "positive|neutral|negative",
  "urgency": "low|medium|high|critical",
  "expectedOutcome": "description of what user wants to achieve",
  "suggestedLoop": "ralph|research|discovery|bug-finder|debugging|end-to-end|meta-cognition|exploration|self-driven|self-learning|self-updating|self-upgrading|planning|context-engineering|context-prompt-engineering"
}
`;

    return await this.llm.generateStructured<ParsedIntent>(prompt, {});
  }
}

// core/planning/ActionPlanner.ts
export interface ActionPlan {
  id: string;
  goal: string;
  steps: PlanStep[];
  estimatedDuration: number;
  requiredTools: string[];
}

export interface PlanStep {
  id: string;
  order: number;
  description: string;
  action: Action;
  dependencies: string[];
  expectedOutcome: string;
}

export interface Action {
  type: 'TOOL_CALL' | 'LLM_GENERATION' | 'CODE_EXECUTION' | 'BROWSER_ACTION' | 'WAIT';
  target: string;
  parameters: Record<string, any>;
}

export class ActionPlanner {
  constructor(
    private llm: LLMService,
    private toolRegistry: ToolRegistry
  ) {}

  async createPlan(intent: ParsedIntent, context: any): Promise<ActionPlan> {
    const availableTools = this.toolRegistry.getAll();
    
    const prompt = `
Create an action plan to achieve the following goal:

Goal: ${intent.expectedOutcome}
Intent: ${intent.primaryIntent}
Urgency: ${intent.urgency}

Available Tools:
${availableTools.map(t => `- ${t.name}: ${t.description}`).join('\n')}

Create a step-by-step plan. Respond with JSON:
{
  "goal": "string",
  "steps": [
    {
      "order": 1,
      "description": "string",
      "action": {
        "type": "TOOL_CALL|LLM_GENERATION|CODE_EXECUTION|BROWSER_ACTION|WAIT",
        "target": "tool name or target",
        "parameters": {}
      },
      "dependencies": [],
      "expectedOutcome": "string"
    }
  ],
  "estimatedDuration": 0,
  "requiredTools": ["tool names"]
}
`;

    const plan = await this.llm.generateStructured<ActionPlan>(prompt, {});
    plan.id = `plan-${Date.now()}`;
    return plan;
  }
}

// ============================================================================
// SECTION 6: GATEWAY
// ============================================================================

// gateway/SessionManager.ts
export interface Session {
  id: string;
  userId: string;
  channel: string;
  createdAt: Date;
  lastActivity: Date;
  metadata: Record<string, any>;
}

export class SessionManager {
  private sessions: Map<string, Session> = new Map();
  private userSessions: Map<string, Set<string>> = new Map();

  createSession(userId: string, channel: string, metadata?: Record<string, any>): Session {
    const session: Session = {
      id: `sess-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      userId,
      channel,
      createdAt: new Date(),
      lastActivity: new Date(),
      metadata: metadata || {}
    };

    this.sessions.set(session.id, session);
    
    if (!this.userSessions.has(userId)) {
      this.userSessions.set(userId, new Set());
    }
    this.userSessions.get(userId)!.add(session.id);

    return session;
  }

  getSession(sessionId: string): Session | undefined {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.lastActivity = new Date();
    }
    return session;
  }

  getUserSessions(userId: string): Session[] {
    const sessionIds = this.userSessions.get(userId);
    if (!sessionIds) return [];
    return Array.from(sessionIds)
      .map(id => this.sessions.get(id))
      .filter((s): s is Session => s !== undefined);
  }

  closeSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.delete(sessionId);
      this.userSessions.get(session.userId)?.delete(sessionId);
    }
  }

  cleanupInactiveSessions(maxAgeMinutes: number = 60): void {
    const cutoff = new Date(Date.now() - maxAgeMinutes * 60 * 1000);
    for (const [id, session] of this.sessions) {
      if (session.lastActivity < cutoff) {
        this.closeSession(id);
      }
    }
  }
}

// ============================================================================
// SECTION 7: CRON MANAGER
// ============================================================================

// cron/CronManager.ts
export interface CronJob {
  id: string;
  name: string;
  schedule: string; // Cron expression
  action: string;
  enabled: boolean;
  lastRun?: Date;
  nextRun?: Date;
}

export class CronManager {
  private jobs: Map<string, CronJob> = new Map();
  private timers: Map<string, NodeJS.Timer> = new Map();
  private eventBus: EventBus;

  constructor(
    private config: { jobs: CronJob[] },
    eventBus: EventBus
  ) {
    this.eventBus = eventBus;
  }

  async initialize(): Promise<void> {
    for (const job of this.config.jobs) {
      this.registerJob(job);
    }
  }

  registerJob(job: CronJob): void {
    this.jobs.set(job.id, job);
    if (job.enabled) {
      this.scheduleJob(job);
    }
  }

  private scheduleJob(job: CronJob): void {
    // Parse cron expression and calculate next run
    const nextRun = this.calculateNextRun(job.schedule);
    job.nextRun = nextRun;

    const delay = nextRun.getTime() - Date.now();
    
    const timer = setTimeout(async () => {
      await this.executeJob(job);
      // Reschedule
      this.scheduleJob(job);
    }, delay);

    this.timers.set(job.id, timer);
  }

  private async executeJob(job: CronJob): Promise<void> {
    job.lastRun = new Date();
    
    await this.eventBus.publish({
      id: `cron-${job.id}-${Date.now()}`,
      timestamp: new Date(),
      type: 'CRON_TRIGGER',
      source: 'cron-manager',
      correlationId: job.id,
      payload: {
        jobId: job.id,
        schedule: job.schedule,
        action: job.action
      }
    } as any);
  }

  private calculateNextRun(schedule: string): Date {
    // Use a cron parser library like node-cron or cron-parser
    // This is a simplified version
    const now = new Date();
    return new Date(now.getTime() + 60000); // Default to 1 minute
  }

  start(): void {
    for (const job of this.jobs.values()) {
      if (job.enabled && !this.timers.has(job.id)) {
        this.scheduleJob(job);
      }
    }
  }

  stop(): void {
    for (const timer of this.timers.values()) {
      clearTimeout(timer);
    }
    this.timers.clear();
  }
}

// ============================================================================
// SECTION 8: MAIN RUNTIME
// ============================================================================

// core/AgentRuntime.ts
export interface RuntimeConfig {
  agentId: string;
  llm: LLMConfig;
  memory: MemoryConfig;
  gateway: GatewayConfig;
  execution: ExecutionConfig;
  cron: { jobs: CronJob[] };
  heartbeatInterval: number;
}

export interface GatewayConfig {
  adapters: AdapterConfig[];
}

export interface AdapterConfig {
  type: string;
  enabled: boolean;
  config: Record<string, any>;
}

export interface ExecutionConfig {
  sandboxPath: string;
  maxExecutionTime: number;
  allowedCommands: string[];
}

export class AgentRuntime {
  private config: RuntimeConfig;
  private eventBus: EventBus;
  private sessionManager: SessionManager;
  private memoryManager: MemoryManager;
  private llmService: LLMService;
  private toolRegistry: ToolRegistry;
  private cronManager: CronManager;
  private intentParser: IntentParser;
  private actionPlanner: ActionPlanner;
  private isRunning: boolean = false;
  private heartbeatTimer?: NodeJS.Timer;

  constructor(config: RuntimeConfig) {
    this.config = config;
    this.eventBus = new EventBus();
    this.sessionManager = new SessionManager();
  }

  async initialize(): Promise<void> {
    console.log('Initializing Agent Runtime...');

    // Initialize memory
    this.memoryManager = new MemoryManager(this.config.memory);
    await this.memoryManager.initialize();

    // Initialize LLM
    this.llmService = new LLMService(this.config.llm);

    // Initialize tool registry
    this.toolRegistry = new ToolRegistry();
    this.registerCoreTools();

    // Initialize intent parser and planner
    this.intentParser = new IntentParser(this.llmService);
    this.actionPlanner = new ActionPlanner(this.llmService, this.toolRegistry);

    // Initialize cron manager
    this.cronManager = new CronManager(this.config.cron, this.eventBus);
    await this.cronManager.initialize();

    // Register event handlers
    this.registerEventHandlers();

    console.log('Agent Runtime initialized successfully');
  }

  private registerCoreTools(): void {
    this.toolRegistry.register(new FileSystemTool());
    // Register other core tools
  }

  private registerEventHandlers(): void {
    // Handle incoming messages
    this.eventBus.subscribe('INBOUND_MESSAGE', async (event: any) => {
      await this.handleInboundMessage(event);
    });

    // Handle cron triggers
    this.eventBus.subscribe('CRON_TRIGGER', async (event: any) => {
      await this.handleCronTrigger(event);
    });

    // Handle errors
    this.eventBus.subscribe('ERROR', async (event: any) => {
      console.error('Error event:', event);
    });
  }

  private async handleInboundMessage(event: any): Promise<void> {
    const { channel, sender, content, metadata } = event.payload;

    // Get or create session
    let session = this.sessionManager.getUserSessions(sender)[0];
    if (!session) {
      session = this.sessionManager.createSession(sender, channel, metadata);
    }

    // Add message to session memory
    this.memoryManager.addMessage(session.id, {
      role: 'user',
      content
    });

    // Parse intent
    const context = {
      recentMessages: this.memoryManager.getContext(session.id, 5),
      userPreferences: this.memoryManager.retrieve(`user:${session.userId}:preferences`)
    };

    const intent = await this.intentParser.parse(content, context);

    // Create plan
    const plan = await this.actionPlanner.createPlan(intent, context);

    // Execute plan (simplified)
    const response = await this.executePlan(plan, session);

    // Store assistant response
    this.memoryManager.addMessage(session.id, {
      role: 'assistant',
      content: response
    });

    // Send response (would go through gateway)
    console.log(`Response to ${sender}: ${response}`);
  }

  private async executePlan(plan: ActionPlan, session: Session): Promise<string> {
    // Simplified execution - in reality, this would iterate through steps
    const response = await this.llmService.generate(
      `Execute this plan: ${JSON.stringify(plan)}`,
      { history: this.memoryManager.getContext(session.id, 10) }
    );
    return response.text;
  }

  private async handleCronTrigger(event: any): Promise<void> {
    console.log(`Cron job triggered: ${event.payload.jobId}`);
    // Handle scheduled tasks
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Runtime already running');
    }

    this.isRunning = true;

    // Start cron manager
    this.cronManager.start();

    // Start heartbeat
    this.startHeartbeat();

    console.log('Agent Runtime started');
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.eventBus.publish({
        id: `hb-${Date.now()}`,
        timestamp: new Date(),
        type: 'HEARTBEAT',
        source: 'runtime',
        correlationId: this.config.agentId,
        payload: {
          agentId: this.config.agentId,
          status: 'healthy',
          metrics: this.collectMetrics()
        }
      } as any);
    }, this.config.heartbeatInterval);
  }

  private collectMetrics(): any {
    return {
      activeSessions: this.sessionManager['sessions'].size,
      memoryEntries: this.memoryManager['longTermStore'].size,
      uptime: process.uptime()
    };
  }

  async shutdown(): Promise<void> {
    console.log('Shutting down Agent Runtime...');
    this.isRunning = false;

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.cronManager.stop();
    await this.memoryManager.close();

    console.log('Agent Runtime shutdown complete');
    process.exit(0);
  }
}

// ============================================================================
// SECTION 9: MAIN ENTRY POINT
// ============================================================================

// index.ts
async function main() {
  const config: RuntimeConfig = {
    agentId: 'openclaw-win10-agent',
    llm: {
      provider: 'openai',
      model: 'gpt-5.2',
      apiKey: process.env.OPENAI_API_KEY,
      temperature: 0.7,
      maxTokens: 4096,
      thinkingMode: 'extra-high'
    },
    memory: {
      sessionStore: 'memory',
      longTermPath: './data/memory',
      vectorStorePath: './data/vectors',
      embeddingModel: 'text-embedding-3-large',
      maxContextMessages: 50
    },
    gateway: {
      adapters: [
        { type: 'gmail', enabled: true, config: {} },
        { type: 'twilio', enabled: true, config: {} }
      ]
    },
    execution: {
      sandboxPath: './sandbox',
      maxExecutionTime: 30000,
      allowedCommands: ['ls', 'cat', 'echo', 'grep', 'find']
    },
    cron: {
      jobs: [
        {
          id: 'health-check',
          name: 'Health Check',
          schedule: '*/5 * * * *',
          action: 'check-health',
          enabled: true
        },
        {
          id: 'memory-consolidation',
          name: 'Memory Consolidation',
          schedule: '0 */6 * * *',
          action: 'consolidate-memory',
          enabled: true
        }
      ]
    },
    heartbeatInterval: 60000 // 1 minute
  };

  const runtime = new AgentRuntime(config);

  try {
    await runtime.initialize();
    await runtime.start();

    // Handle shutdown signals
    process.on('SIGINT', () => runtime.shutdown());
    process.on('SIGTERM', () => runtime.shutdown());
  } catch (error) {
    console.error('Failed to start runtime:', error);
    process.exit(1);
  }
}

// Run if this file is executed directly
if (require.main === module) {
  main();
}

export { AgentRuntime, RuntimeConfig };
