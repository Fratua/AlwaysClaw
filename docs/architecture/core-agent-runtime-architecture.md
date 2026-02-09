# Core Agent Runtime Architecture Specification
## Windows 10 OpenClaw-Inspired AI Agent System

**Version:** 1.0  
**Date:** 2025  
**Target Platform:** Windows 10  
**LLM Backend:** GPT-5.2 (Extra High Thinking Capability)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Principles](#architecture-principles)
4. [Core Components](#core-components)
5. [Event Loop & Message Processing](#event-loop--message-processing)
6. [Agent Core Module](#agent-core-module)
7. [Component Interaction Diagrams](#component-interaction-diagrams)
8. [File Structure & Module Organization](#file-structure--module-organization)
9. [Entry Points & Initialization Flow](#entry-points--initialization-flow)
10. [Integration Points](#integration-points)
11. [Agentic Loops Specification](#agentic-loops-specification)
12. [Memory System](#memory-system)
13. [Security & Sandboxing](#security--sandboxing)
14. [Configuration Files](#configuration-files)

---

## Executive Summary

This document defines the complete core agent runtime architecture for a Windows 10-focused AI agent system inspired by OpenClaw. The system is designed to run 24/7 as an autonomous agent capable of:

- Multi-channel communication (Gmail, Twilio voice/SMS, browser control)
- Full system access with appropriate security controls
- Text-to-speech (TTS) and speech-to-text (STT) capabilities
- 37 scheduled tasks (15 operational loops, 16 cognitive loops, 6 cron jobs)
- Persistent memory and identity management
- Cron-based scheduled actions with heartbeat monitoring

The architecture follows event-driven design principles with a clear separation between the Gateway (communication), Agent Core (cognition), and Execution Layer (action).

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT SYSTEM OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Gateway    │───▶│  Agent Core  │───▶│   Execution  │───▶│  System   │ │
│  │   Layer      │◀───│   (Brain)    │◀───│    Layer     │◀───│  Access   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Messaging   │    │   Memory     │    │    Tools     │                  │
│  │  Adapters    │    │   System     │    │   Registry   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     AGENTIC LOOPS (37 Scheduled Tasks)                   │   │
│  │  ralph | research | discovery | bug-finder | debugging | end-to-end │   │
│  │  meta-cognition | exploration | self-driven | self-learning        │   │
│  │  self-updating | self-upgrading | planning | context-engineering   │   │
│  │  context-prompt-engineering                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Principles

### 1. **Event-Driven Architecture (EDA)**
- All communication happens through events
- Loose coupling between components
- Asynchronous processing for scalability
- Event sourcing for audit trails

### 2. **Layered Separation of Concerns**
- **Gateway Layer:** Communication abstraction only
- **Agent Core:** Decision-making and planning only
- **Execution Layer:** Action execution only
- **Memory Layer:** State persistence only

### 3. **Model Agnostic with GPT-5.2 Default**
- Primary: GPT-5.2 with extra high thinking
- Fallback: Configurable local models via Ollama
- Easy switching between providers

### 4. **Local-First Security**
- All data stays on Windows 10 host
- Docker sandboxing for code execution
- Permission-based tool access
- Audit logging for all actions

### 5. **Extensible Skill System**
- Skills as TypeScript modules
- Hot-reload capability
- Versioned skill registry
- Community skill marketplace support

---

## Core Components

### 1. Gateway Layer

```typescript
// gateway/Gateway.ts
interface Gateway {
  // Message routing and session management
  routeMessage(message: InboundMessage): Promise<Session>;
  sendResponse(sessionId: string, response: AgentResponse): Promise<void>;
  
  // Channel adapter management
  registerAdapter(adapter: ChannelAdapter): void;
  unregisterAdapter(adapterId: string): void;
  
  // Session lifecycle
  createSession(channel: string, userId: string): Session;
  getSession(sessionId: string): Session | null;
  closeSession(sessionId: string): void;
}
```

**Responsibilities:**
- Receive messages from all channels (Gmail, Twilio, Browser, etc.)
- Normalize messages to internal format
- Route to appropriate session
- Send responses back to originating channel
- NO decision-making logic

### 2. Agent Core (The Brain)

```typescript
// core/AgentCore.ts
interface AgentCore {
  // Intent processing
  parseIntent(message: InboundMessage, context: Context): Intent;
  
  // Action planning
  createPlan(intent: Intent, context: Context): ActionPlan;
  
  // Loop selection
  selectAgenticLoop(intent: Intent): AgenticLoop;
  
  // Execution orchestration
  executePlan(plan: ActionPlan, session: Session): Promise<ExecutionResult>;
  
  // Reflection and learning
  reflectOnResult(result: ExecutionResult): Reflection;
}
```

**Responsibilities:**
- Interpret user intent using LLM
- Plan multi-step actions
- Select appropriate agentic loop
- Orchestrate tool execution
- Reflect on outcomes

### 3. Execution Layer

```typescript
// execution/ExecutionEngine.ts
interface ExecutionEngine {
  // Tool execution
  executeTool(toolName: string, params: any): Promise<ToolResult>;
  
  // Script execution (sandboxed)
  executeScript(code: string, language: string): Promise<ExecutionResult>;
  
  // Browser automation
  executeBrowserAction(action: BrowserAction): Promise<BrowserResult>;
  
  // System commands (restricted)
  executeSystemCommand(command: string): Promise<CommandResult>;
}
```

**Responsibilities:**
- Execute tools safely
- Manage sandboxed environments
- Handle browser automation
- Execute system commands with permissions

### 4. Memory System

```typescript
// memory/MemoryManager.ts
interface MemoryManager {
  // Short-term (session) memory
  getSessionContext(sessionId: string): SessionContext;
  updateSessionContext(sessionId: string, update: ContextUpdate): void;
  
  // Long-term memory
  storeMemory(key: string, value: any, type: MemoryType): Promise<void>;
  retrieveMemory(query: string): Promise<MemoryEntry[]>;
  
  // Vector search
  searchSimilar(query: string, limit: number): Promise<VectorResult[]>;
}
```

**Responsibilities:**
- Session context management
- Long-term memory persistence
- Vector similarity search
- Memory consolidation

---

## Event Loop & Message Processing

### Event Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN EVENT LOOP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Event     │───▶│   Event     │───▶│   Handler   │───▶│   State     │  │
│  │   Queue     │    │   Router    │    │   Registry  │    │   Update    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     EVENT TYPES                                      │  │
│  │  INBOUND_MESSAGE │ TOOL_RESULT │ AGENT_RESPONSE │ SYSTEM_EVENT     │  │
│  │  CRON_TRIGGER    │ HEARTBEAT   │ MEMORY_UPDATE  │ ERROR_EVENT      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Message Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MESSAGE PROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: INGESTION                                                           │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Channel │───▶│  Normalize  │───▶│   Create    │───▶│   Publish   │      │
│  │ Adapter │    │   Message   │    │   Session   │    │   Event     │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                              │
│  Step 2: PROCESSING                                                          │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Event  │───▶│   Load      │───▶│   Parse     │───▶│   Select    │      │
│  │  Bus    │    │   Context   │    │   Intent    │    │   Agentic   │      │
│  └─────────┘    └─────────────┘    └─────────────┘    │    Loop     │      │
│                                                       └─────────────┘      │
│  Step 3: EXECUTION                                                           │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Agentic │───▶│   Create    │───▶│   Execute   │───▶│   Reflect   │      │
│  │  Loop   │    │    Plan     │    │   Actions   │    │   & Learn   │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                              │
│  Step 4: RESPONSE                                                            │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Result  │───▶│   Update    │───▶│   Format    │───▶│   Send to   │      │
│  │ Handler │    │   Memory    │    │   Response  │    │   Channel   │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event Types Definition

```typescript
// events/EventTypes.ts

// Base Event Interface
interface BaseEvent {
  id: string;
  timestamp: Date;
  type: EventType;
  source: string;
  correlationId: string;
}

// Inbound Message Event
interface InboundMessageEvent extends BaseEvent {
  type: 'INBOUND_MESSAGE';
  payload: {
    channel: 'gmail' | 'twilio_voice' | 'twilio_sms' | 'browser' | 'internal';
    sender: string;
    content: string;
    metadata: Record<string, any>;
    sessionId?: string;
  };
}

// Tool Execution Event
interface ToolExecutionEvent extends BaseEvent {
  type: 'TOOL_EXECUTION';
  payload: {
    toolName: string;
    parameters: any;
    sessionId: string;
    executionId: string;
  };
}

// Tool Result Event
interface ToolResultEvent extends BaseEvent {
  type: 'TOOL_RESULT';
  payload: {
    executionId: string;
    result: any;
    error?: string;
    duration: number;
  };
}

// Agent Response Event
interface AgentResponseEvent extends BaseEvent {
  type: 'AGENT_RESPONSE';
  payload: {
    sessionId: string;
    content: string;
    actions: Action[];
    metadata: ResponseMetadata;
  };
}

// Cron Trigger Event
interface CronTriggerEvent extends BaseEvent {
  type: 'CRON_TRIGGER';
  payload: {
    jobId: string;
    schedule: string;
    action: string;
  };
}

// Heartbeat Event
interface HeartbeatEvent extends BaseEvent {
  type: 'HEARTBEAT';
  payload: {
    agentId: string;
    status: 'healthy' | 'degraded' | 'error';
    metrics: SystemMetrics;
  };
}

// Memory Update Event
interface MemoryUpdateEvent extends BaseEvent {
  type: 'MEMORY_UPDATE';
  payload: {
    key: string;
    value: any;
    operation: 'set' | 'delete' | 'merge';
  };
}

// Error Event
interface ErrorEvent extends BaseEvent {
  type: 'ERROR';
  payload: {
    error: Error;
    context: string;
    recoverable: boolean;
  };
}

type EventType = 
  | 'INBOUND_MESSAGE'
  | 'TOOL_EXECUTION'
  | 'TOOL_RESULT'
  | 'AGENT_RESPONSE'
  | 'CRON_TRIGGER'
  | 'HEARTBEAT'
  | 'MEMORY_UPDATE'
  | 'ERROR';
```

---

## Agent Core Module

### Intent Parsing System

```typescript
// core/intent/IntentParser.ts
interface IntentParser {
  parse(message: string, context: Context): Promise<ParsedIntent>;
}

interface ParsedIntent {
  primaryIntent: IntentType;
  confidence: number;
  entities: Entity[];
  sentiment: 'positive' | 'neutral' | 'negative';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  expectedOutcome: string;
  suggestedAgenticLoop: AgenticLoopType;
}

type IntentType = 
  | 'QUERY'           // Information request
  | 'ACTION'          // Execute task
  | 'CONVERSATION'    // Casual chat
  | 'DEBUG'           // Debug request
  | 'RESEARCH'        // Research task
  | 'PLANNING'        // Create plan
  | 'SYSTEM'          // System command
  | 'LEARNING'        // Learning request
  | 'META'            // Meta-cognitive request
  | 'UNKNOWN';

// Intent Parser Implementation
class LLMIntentParser implements IntentParser {
  constructor(private llm: LLMService) {}

  async parse(message: string, context: Context): Promise<ParsedIntent> {
    const prompt = this.buildIntentPrompt(message, context);
    const response = await this.llm.generateStructured(prompt, IntentSchema);
    return this.validateAndNormalize(response);
  }

  private buildIntentPrompt(message: string, context: Context): string {
    return `
Analyze the following message and determine the user's intent.

Message: "${message}"

Context:
- Previous messages: ${context.recentMessages.length}
- User preferences: ${JSON.stringify(context.userPreferences)}
- Current session goals: ${context.sessionGoals.join(', ')}

Provide a structured analysis with:
1. Primary intent classification
2. Confidence score (0-1)
3. Key entities extracted
4. Sentiment analysis
5. Urgency level
6. Expected outcome
7. Suggested agentic loop for handling
`;
  }
}
```

### Action Planning System

```typescript
// core/planning/ActionPlanner.ts
interface ActionPlanner {
  createPlan(intent: ParsedIntent, context: Context): Promise<ActionPlan>;
  refinePlan(plan: ActionPlan, feedback: Feedback): Promise<ActionPlan>;
}

interface ActionPlan {
  id: string;
  goal: string;
  steps: PlanStep[];
  estimatedDuration: number;
  requiredTools: string[];
  fallbackStrategy: FallbackStrategy;
  successCriteria: string[];
}

interface PlanStep {
  id: string;
  order: number;
  description: string;
  action: Action;
  dependencies: string[];
  expectedOutcome: string;
  retryPolicy: RetryPolicy;
}

interface Action {
  type: 'TOOL_CALL' | 'LLM_GENERATION' | 'CODE_EXECUTION' | 'BROWSER_ACTION' | 'WAIT';
  target: string;
  parameters: Record<string, any>;
}

// Planner Implementation
class LLMActionPlanner implements ActionPlanner {
  constructor(
    private llm: LLMService,
    private toolRegistry: ToolRegistry,
    private memory: MemoryManager
  ) {}

  async createPlan(intent: ParsedIntent, context: Context): Promise<ActionPlan> {
    const availableTools = this.toolRegistry.getAvailableTools();
    const relevantMemories = await this.memory.retrieveMemory(intent.primaryIntent);
    
    const prompt = this.buildPlanningPrompt(intent, context, availableTools, relevantMemories);
    const plan = await this.llm.generateStructured(prompt, ActionPlanSchema);
    
    return this.validatePlan(plan);
  }

  private buildPlanningPrompt(
    intent: ParsedIntent,
    context: Context,
    tools: Tool[],
    memories: MemoryEntry[]
  ): string {
    return `
Create a detailed action plan to achieve the following goal:

Goal: ${intent.expectedOutcome}
Intent: ${intent.primaryIntent}
Urgency: ${intent.urgency}

Available Tools:
${tools.map(t => `- ${t.name}: ${t.description}`).join('\n')}

Relevant Context from Memory:
${memories.map(m => `- ${m.key}: ${m.value}`).join('\n')}

Create a step-by-step plan with:
1. Clear step descriptions
2. Tool selections with parameters
3. Dependencies between steps
4. Expected outcomes for each step
5. Fallback strategies
6. Success criteria

Plan must be executable and verifiable.
`;
  }
}
```

### Agentic Loop Selection

```typescript
// core/loops/AgenticLoopSelector.ts
interface AgenticLoopSelector {
  selectLoop(intent: ParsedIntent, context: Context): AgenticLoop;
}

class DefaultLoopSelector implements AgenticLoopSelector {
  private loopMap: Map<IntentType, AgenticLoopType> = new Map([
    ['QUERY', 'ralph'],
    ['RESEARCH', 'research'],
    ['ACTION', 'end-to-end'],
    ['DEBUG', 'debugging'],
    ['PLANNING', 'planning'],
    ['LEARNING', 'self-learning'],
    ['META', 'meta-cognition'],
  ]);

  selectLoop(intent: ParsedIntent, context: Context): AgenticLoop {
    // Check for explicit loop request
    const explicitLoop = this.extractExplicitLoop(context);
    if (explicitLoop) {
      return this.getLoop(explicitLoop);
    }

    // Use intent-based selection
    const loopType = this.loopMap.get(intent.primaryIntent) || 'ralph';
    
    // Override based on complexity
    if (intent.confidence < 0.5) {
      return this.getLoop('exploration');
    }

    return this.getLoop(loopType);
  }

  private extractExplicitLoop(context: Context): AgenticLoopType | null {
    // Check for explicit loop requests like "@research" or "@debug"
    const match = context.lastMessage?.match(/@(\w+)/);
    return match ? match[1] as AgenticLoopType : null;
  }
}
```

---

## Component Interaction Diagrams

### Full System Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FULL SYSTEM INTERACTION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   USER                                    SYSTEM                                         │
│    │                                        │                                           │
│    │  1. Send Message (Gmail/Twilio/etc)    │                                           │
│    │───────────────────────────────────────▶│                                           │
│    │                                        │                                           │
│    │                    ┌───────────────────┴───────────────────┐                       │
│    │                    │         GATEWAY LAYER                 │                       │
│    │                    │  ┌─────────┐      ┌─────────────┐     │                       │
│    │                    │  │ Channel │─────▶│  Normalize  │     │                       │
│    │                    │  │ Adapter │      │   Message   │     │                       │
│    │                    │  └─────────┘      └──────┬──────┘     │                       │
│    │                    │                          │            │                       │
│    │                    │                     ┌────┴────┐       │                       │
│    │                    │                     │ Session │       │                       │
│    │                    │                     │ Manager │       │                       │
│    │                    │                     └────┬────┘       │                       │
│    │                    └───────────────────────┬──┘            │                       │
│    │                                              │              │                       │
│    │                    ┌─────────────────────────┴──────────┐   │                       │
│    │                    │         EVENT BUS                  │   │                       │
│    │                    │  ┌─────────────────────────────┐   │   │                       │
│    │                    │  │  Publish: INBOUND_MESSAGE   │   │   │                       │
│    │                    │  └─────────────────────────────┘   │   │                       │
│    │                    └────────────────────────────────────┘   │                       │
│    │                                              │              │                       │
│    │                    ┌─────────────────────────┴──────────┐   │                       │
│    │                    │         AGENT CORE                 │   │                       │
│    │                    │  ┌─────────┐      ┌─────────────┐  │   │                       │
│    │                    │  │ Intent  │─────▶│   Action    │  │   │                       │
│    │                    │  │ Parser  │      │   Planner   │  │   │                       │
│    │                    │  └─────────┘      └──────┬──────┘  │   │                       │
│    │                    │                          │         │   │                       │
│    │                    │                     ┌────┴────┐    │   │                       │
│    │                    │                     │ Agentic │    │   │                       │
│    │                    │                     │  Loop   │    │   │                       │
│    │                    │                     └────┬────┘    │   │                       │
│    │                    └─────────────────────────┬──┘       │   │                       │
│    │                                              │           │   │                       │
│    │                    ┌─────────────────────────┴──────────┐│   │                       │
│    │                    │       EXECUTION LAYER              ││   │                       │
│    │                    │  ┌─────────┐      ┌─────────────┐  ││   │                       │
│    │                    │  │  Tool   │─────▶│  Sandbox    │  ││   │                       │
│    │                    │  │Registry │      │  Executor   │  ││   │                       │
│    │                    │  └─────────┘      └──────┬──────┘  ││   │                       │
│    │                    │                          │         ││   │                       │
│    │                    │                     ┌────┴────┐    ││   │                       │
│    │                    │                     │ Browser │    ││   │                       │
│    │                    │                     │ Control │    ││   │                       │
│    │                    │                     └─────────┘    ││   │                       │
│    │                    └────────────────────────────────────┘│   │                       │
│    │                                              │           │   │                       │
│    │                    ┌─────────────────────────┴──────────┐│   │                       │
│    │                    │         MEMORY SYSTEM              ││   │                       │
│    │                    │  ┌─────────┐      ┌─────────────┐  ││   │                       │
│    │                    │  │ Session │─────▶│   Vector    │  ││   │                       │
│    │                    │  │  Store  │      │    Store    │  ││   │                       │
│    │                    │  └─────────┘      └─────────────┘  ││   │                       │
│    │                    └────────────────────────────────────┘│   │                       │
│    │                                              │           │   │                       │
│    │  2. Receive Response                         │           │   │                       │
│    │◀─────────────────────────────────────────────┘           │   │                       │
│    │                                                          │   │                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Core Internal Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           AGENT CORE INTERNAL FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           INPUT PROCESSING                                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Receive   │───▶│    Load     │───▶│   Parse     │───▶│  Determine  │      │   │
│  │  │   Message   │    │   Context   │    │   Intent    │    │   Urgency   │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         LOOP SELECTION                                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Intent    │───▶│   Check     │───▶│   Select    │───▶│   Load      │      │   │
│  │  │   Type      │    │   Override  │    │    Loop     │    │   Config    │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         PLANNING PHASE                                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Define    │───▶│   Search    │───▶│   Create    │───▶│   Validate  │      │   │
│  │  │    Goal     │    │   Memory    │    │    Plan     │    │    Plan     │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        EXECUTION PHASE                                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Execute   │───▶│   Process   │───▶│   Check     │───▶│   Iterate   │      │   │
│  │  │    Step     │    │   Result    │    │   Success   │    │   or End    │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        REFLECTION PHASE                                          │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Analyze   │───▶│   Update    │───▶│   Store     │───▶│   Generate  │      │   │
│  │  │   Outcome   │    │   Memory    │    │   Metrics   │    │   Response  │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure & Module Organization

```
openclaw-win10/
├── 📁 src/
│   ├── 📁 core/                          # Agent Core (The Brain)
│   │   ├── 📄 AgentCore.ts               # Main agent orchestrator
│   │   ├── 📁 intent/
│   │   │   ├── 📄 IntentParser.ts        # Intent parsing interface
│   │   │   ├── 📄 LLMIntentParser.ts     # GPT-5.2 intent parser
│   │   │   ├── 📄 IntentTypes.ts         # Intent type definitions
│   │   │   └── 📄 EntityExtractor.ts     # Entity extraction
│   │   ├── 📁 planning/
│   │   │   ├── 📄 ActionPlanner.ts       # Planning interface
│   │   │   ├── 📄 LLMActionPlanner.ts    # GPT-5.2 planner
│   │   │   ├── 📄 PlanTypes.ts           # Plan type definitions
│   │   │   └── 📄 PlanValidator.ts       # Plan validation
│   │   ├── 📁 loops/                     # 37 Scheduled Tasks (15 operational + 16 cognitive + 6 cron)
│   │   │   ├── 📄 AgenticLoop.ts         # Base loop interface
│   │   │   ├── 📄 LoopRegistry.ts        # Loop registration
│   │   │   ├── 📄 RalphLoop.ts           # Default conversational loop
│   │   │   ├── 📄 ResearchLoop.ts        # Research tasks
│   │   │   ├── 📄 DiscoveryLoop.ts       # Discovery/exploration
│   │   │   ├── 📄 BugFinderLoop.ts       # Bug detection
│   │   │   ├── 📄 DebuggingLoop.ts       # Debugging assistance
│   │   │   ├── 📄 EndToEndLoop.ts        # End-to-end task completion
│   │   │   ├── 📄 MetaCognitionLoop.ts   # Self-reflection
│   │   │   ├── 📄 ExplorationLoop.ts     # Unknown territory exploration
│   │   │   ├── 📄 SelfDrivenLoop.ts      # Autonomous actions
│   │   │   ├── 📄 SelfLearningLoop.ts    # Learning from interactions
│   │   │   ├── 📄 SelfUpdatingLoop.ts    # Self-modification
│   │   │   ├── 📄 SelfUpgradingLoop.ts   # System improvement
│   │   │   ├── 📄 PlanningLoop.ts        # Complex planning
│   │   │   ├── 📄 ContextEngineeringLoop.ts  # Context optimization
│   │   │   └── 📄 ContextPromptEngineeringLoop.ts  # Prompt optimization
│   │   └── 📁 reflection/
│   │       ├── 📄 ReflectionEngine.ts    # Outcome analysis
│   │       └── 📄 LearningEngine.ts      # Pattern learning
│   │
│   ├── 📁 gateway/                       # Communication Layer
│   │   ├── 📄 Gateway.ts                 # Main gateway orchestrator
│   │   ├── 📄 SessionManager.ts          # Session lifecycle
│   │   ├── 📄 MessageNormalizer.ts       # Message format normalization
│   │   ├── 📁 adapters/                  # Channel Adapters
│   │   │   ├── 📄 ChannelAdapter.ts      # Base adapter interface
│   │   │   ├── 📄 GmailAdapter.ts        # Gmail integration
│   │   │   ├── 📄 TwilioVoiceAdapter.ts  # Twilio voice calls
│   │   │   ├── 📄 TwilioSMSAdapter.ts    # Twilio SMS
│   │   │   ├── 📄 BrowserAdapter.ts      # Browser control interface
│   │   │   └── 📄 InternalAdapter.ts     # Internal/system messages
│   │   └── 📁 sessions/
│   │       ├── 📄 Session.ts             # Session model
│   │       └── 📄 SessionStore.ts        # Session persistence
│   │
│   ├── 📁 execution/                     # Execution Layer
│   │   ├── 📄 ExecutionEngine.ts         # Main execution orchestrator
│   │   ├── 📄 SandboxManager.ts          # Docker sandbox management
│   │   ├── 📁 tools/                     # Tool Registry
│   │   │   ├── 📄 ToolRegistry.ts        # Tool registration/management
│   │   │   ├── 📄 ToolExecutor.ts        # Tool execution
│   │   │   ├── 📄 ToolTypes.ts           # Tool type definitions
│   │   │   ├── 📄 BaseTool.ts            # Base tool class
│   │   │   ├── 📁 implementations/       # Tool implementations
│   │   │   │   ├── 📄 FileSystemTool.ts
│   │   │   │   ├── 📄 BrowserTool.ts
│   │   │   │   ├── 📄 ShellTool.ts
│   │   │   │   ├── 📄 CodeExecutionTool.ts
│   │   │   │   ├── 📄 GmailTool.ts
│   │   │   │   ├── 📄 TwilioTool.ts
│   │   │   │   ├── 📄 SearchTool.ts
│   │   │   │   └── 📄 SystemTool.ts
│   │   │   └── 📁 skills/                # Skill modules
│   │   │       ├── 📄 SkillLoader.ts
│   │   │       └── 📁 installed/         # Installed skills
│   │   └── 📁 browser/
│   │       ├── 📄 BrowserController.ts   # Browser automation
│   │       ├── 📄 PageManager.ts         # Page lifecycle
│   │       └── 📄 ActionExecutor.ts      # Browser actions
│   │
│   ├── 📁 memory/                        # Memory System
│   │   ├── 📄 MemoryManager.ts           # Memory orchestrator
│   │   ├── 📁 session/
│   │   │   ├── 📄 SessionMemory.ts       # Short-term session memory
│   │   │   └── 📄 ContextWindow.ts       # Context window management
│   │   ├── 📁 longterm/
│   │   │   ├── 📄 LongTermMemory.ts      # Long-term memory interface
│   │   │   ├── 📄 FileMemoryStore.ts     # File-based storage
│   │   │   └── 📄 MemoryConsolidator.ts  # Memory consolidation
│   │   └── 📁 vector/
│   │       ├── 📄 VectorStore.ts         # Vector database interface
│   │       ├── 📄 EmbeddingService.ts    # Text embedding
│   │       └── 📄 SimilaritySearch.ts    # Similarity search
│   │
│   ├── 📁 events/                        # Event System
│   │   ├── 📄 EventBus.ts                # Central event bus
│   │   ├── 📄 EventTypes.ts              # Event type definitions
│   │   ├── 📄 EventRouter.ts             # Event routing
│   │   ├── 📄 EventHandler.ts            # Handler interface
│   │   └── 📁 handlers/                  # Event handlers
│   │       ├── 📄 MessageHandler.ts
│   │       ├── 📄 ToolResultHandler.ts
│   │       ├── 📄 CronHandler.ts
│   │       └── 📄 HeartbeatHandler.ts
│   │
│   ├── 📁 llm/                           # LLM Integration
│   │   ├── 📄 LLMService.ts              # LLM service interface
│   │   ├── 📄 GPT52Provider.ts           # GPT-5.2 provider
│   │   ├── 📄 OllamaProvider.ts          # Local model provider
│   │   ├── 📄 PromptBuilder.ts           # Prompt construction
│   │   ├── 📄 ResponseParser.ts          # Response parsing
│   │   └── 📁 prompts/                   # Prompt templates
│   │       ├── 📄 system-prompts/
│   │       └── 📄 task-prompts/
│   │
│   ├── 📁 cron/                          # Cron & Scheduling
│   │   ├── 📄 CronManager.ts             # Cron job management
│   │   ├── 📄 JobRegistry.ts             # Job registration
│   │   ├── 📄 JobExecutor.ts             # Job execution
│   │   └── 📁 jobs/                      # Job implementations
│   │       ├── 📄 HeartbeatJob.ts
│   │       ├── 📄 MemoryConsolidationJob.ts
│   │       └── 📄 SystemMaintenanceJob.ts
│   │
│   ├── 📁 voice/                         # Voice Processing
│   │   ├── 📄 TTSManager.ts              # Text-to-speech
│   │   ├── 📄 STTManager.ts              # Speech-to-text
│   │   ├── 📄 VoiceSynthesizer.ts        # Voice synthesis
│   │   └── 📁 providers/
│   │       ├── 📄 ElevenLabsProvider.ts
│   │       └── 📄 WindowsTTSProvider.ts
│   │
│   ├── 📁 identity/                      # Identity & Soul
│   │   ├── 📄 IdentityManager.ts         # Identity management
│   │   ├── 📄 PersonalityEngine.ts       # Personality expression
│   │   └── 📄 UserProfileManager.ts      # User profile management
│   │
│   ├── 📁 config/                        # Configuration
│   │   ├── 📄 ConfigManager.ts           # Configuration management
│   │   ├── 📄 SchemaValidator.ts         # Config validation
│   │   └── 📁 schemas/
│   │       └── 📄 config-schema.json
│   │
│   ├── 📁 utils/                         # Utilities
│   │   ├── 📄 Logger.ts                  # Logging
│   │   ├── 📄 ErrorHandler.ts            # Error handling
│   │   ├── 📄 SecurityUtils.ts           # Security utilities
│   │   └── 📄 ValidationUtils.ts         # Validation
│   │
│   └── 📄 index.ts                       # Main entry point
│
├── 📁 config/                            # Configuration Files
│   ├── 📄 SOUL.md                        # Personality definition
│   ├── 📄 IDENTITY.md                    # Identity presentation
│   ├── 📄 USER.md                        # User context
│   ├── 📄 MEMORY.md                      # Long-term memory
│   ├── 📄 AGENTS.md                      # Agent instructions
│   ├── 📄 HEARTBEAT.md                   # Scheduled actions
│   ├── 📄 TOOLS.md                       # Tool definitions
│   ├── 📄 LOOPS.md                       # Agentic loop configs
│   └── 📄 settings.json                  # System settings
│
├── 📁 data/                              # Data Storage
│   ├── 📁 sessions/                      # Session data
│   ├── 📁 memory/                        # Long-term memory
│   ├── 📁 vectors/                       # Vector embeddings
│   ├── 📁 logs/                          # System logs
│   └── 📁 cache/                         # Temporary cache
│
├── 📁 skills/                            # Skill Modules
│   ├── 📁 core/                          # Core skills
│   └── 📁 custom/                        # Custom skills
│
├── 📁 sandbox/                           # Execution Sandbox
│   └── 📄 Dockerfile                     # Sandbox container
│
├── 📁 docs/                              # Documentation
│   └── 📄 architecture.md
│
├── 📁 tests/                             # Test Suite
│   ├── 📁 unit/
│   ├── 📁 integration/
│   └── 📁 e2e/
│
├── 📄 package.json                       # Node.js dependencies
├── 📄 tsconfig.json                      # TypeScript config
├── 📄 docker-compose.yml                 # Docker services
├── 📄 .env.example                       # Environment template
└── 📄 README.md                          # Project readme
```

---

## Entry Points & Initialization Flow

### Application Entry Points

```typescript
// src/index.ts - Main Entry Point

import { AgentRuntime } from './core/AgentRuntime';
import { ConfigManager } from './config/ConfigManager';
import { Logger } from './utils/Logger';

async function main() {
  const logger = new Logger('Main');
  
  try {
    // Phase 1: Configuration Loading
    logger.info('Loading configuration...');
    const config = await ConfigManager.load();
    
    // Phase 2: System Initialization
    logger.info('Initializing agent runtime...');
    const runtime = new AgentRuntime(config);
    
    // Phase 3: Component Startup
    await runtime.initialize();
    
    // Phase 4: Start Event Loop
    logger.info('Starting event loop...');
    await runtime.start();
    
    // Phase 5: Register Shutdown Handlers
    process.on('SIGINT', () => runtime.shutdown());
    process.on('SIGTERM', () => runtime.shutdown());
    
    logger.info('Agent runtime started successfully');
  } catch (error) {
    logger.error('Failed to start agent runtime:', error);
    process.exit(1);
  }
}

main();
```

### Initialization Sequence

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           INITIALIZATION SEQUENCE                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: CONFIGURATION LOADING                                                  │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │  │  Load   │───▶│ Validate│───▶│  Merge  │───▶│  Apply  │───▶│  Store  │       │   │
│  │  │  Files  │    │ Schema  │    │  Env    │    │ Defaults│    │  Config │       │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │   │
│  │                                                                                  │   │
│  │  Files: SOUL.md, IDENTITY.md, USER.md, MEMORY.md, AGENTS.md, HEARTBEAT.md      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: CORE SERVICES INITIALIZATION                                           │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │  │  Logger │───▶│  Event  │───▶│  Memory │───▶│   LLM   │───▶│  Voice  │       │   │
│  │  │  Setup  │    │   Bus   │    │  System │    │ Service │    │ Services│       │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: AGENT CORE INITIALIZATION                                              │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │  │  Intent │───▶│  Action │───▶│  Agentic│───▶│Reflect- │───▶│  Load   │       │   │
│  │  │  Parser │    │ Planner │    │  Loops  │    │  ion    │    │  Soul   │       │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: EXECUTION LAYER INITIALIZATION                                         │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │  │  Tool   │───▶│ Sandbox │───▶│ Browser │───▶│  Skill  │───▶│  Verify │       │   │
│  │  │ Registry│    │  Setup  │    │  Setup  │    │  Loader │    │  Tools  │       │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 5: GATEWAY INITIALIZATION                                                 │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │   │
│  │  │ Session │───▶│  Gmail  │───▶│ Twilio  │───▶│ Browser │───▶│  Test   │       │   │
│  │  │ Manager │    │ Adapter │    │ Adapters│    │ Adapter │    │ Connections│    │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 6: CRON & HEARTBEAT STARTUP                                               │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                       │   │
│  │  │  Load   │───▶│ Register│───▶│ Schedule│───▶│  Start  │                       │   │
│  │  │  Jobs   │    │  Jobs   │    │  Jobs   │    │ Heartbeat                        │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘                       │   │
│  │                                                                                  │   │
│  │  Jobs from HEARTBEAT.md: health-check, memory-consolidation, self-update        │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 7: EVENT LOOP START                                                       │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                                       │   │
│  │  │  Start  │───▶│  Listen │───▶│  Ready  │                                       │   │
│  │  │  Bus    │    │  Events │    │  State  │                                       │   │
│  │  └─────────┘    └─────────┘    └─────────┘                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Runtime Class Structure

```typescript
// src/core/AgentRuntime.ts

export class AgentRuntime {
  private config: RuntimeConfig;
  private eventBus: EventBus;
  private gateway: Gateway;
  private agentCore: AgentCore;
  private executionEngine: ExecutionEngine;
  private memoryManager: MemoryManager;
  private cronManager: CronManager;
  private llmService: LLMService;
  private isRunning: boolean = false;

  constructor(config: RuntimeConfig) {
    this.config = config;
    this.eventBus = new EventBus();
  }

  async initialize(): Promise<void> {
    // Initialize in dependency order
    this.memoryManager = new MemoryManager(this.config.memory);
    await this.memoryManager.initialize();

    this.llmService = new LLMService(this.config.llm);
    await this.llmService.initialize();

    this.executionEngine = new ExecutionEngine(
      this.config.execution,
      this.eventBus
    );
    await this.executionEngine.initialize();

    this.agentCore = new AgentCore(
      this.llmService,
      this.memoryManager,
      this.executionEngine,
      this.eventBus
    );
    await this.agentCore.initialize();

    this.gateway = new Gateway(
      this.config.gateway,
      this.eventBus
    );
    await this.gateway.initialize();

    this.cronManager = new CronManager(
      this.config.cron,
      this.eventBus
    );
    await this.cronManager.initialize();

    // Register event handlers
    this.registerEventHandlers();
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Runtime already running');
    }

    this.isRunning = true;

    // Start all services
    await this.gateway.start();
    await this.cronManager.start();
    await this.agentCore.start();

    // Start heartbeat
    this.startHeartbeat();

    Logger.info('Agent runtime started');
  }

  async shutdown(): Promise<void> {
    Logger.info('Shutting down agent runtime...');
    this.isRunning = false;

    // Stop in reverse order
    await this.cronManager.stop();
    await this.gateway.stop();
    await this.agentCore.stop();
    await this.executionEngine.stop();
    await this.memoryManager.close();

    Logger.info('Agent runtime shutdown complete');
    process.exit(0);
  }

  private registerEventHandlers(): void {
    // Message handler
    this.eventBus.subscribe('INBOUND_MESSAGE', async (event) => {
      await this.handleInboundMessage(event);
    });

    // Tool result handler
    this.eventBus.subscribe('TOOL_RESULT', async (event) => {
      await this.handleToolResult(event);
    });

    // Cron trigger handler
    this.eventBus.subscribe('CRON_TRIGGER', async (event) => {
      await this.handleCronTrigger(event);
    });

    // Error handler
    this.eventBus.subscribe('ERROR', async (event) => {
      await this.handleError(event);
    });
  }

  private async handleInboundMessage(event: InboundMessageEvent): Promise<void> {
    const session = await this.gateway.getOrCreateSession(
      event.payload.channel,
      event.payload.sender
    );

    const response = await this.agentCore.processMessage(
      event.payload.content,
      session
    );

    await this.gateway.sendResponse(session.id, response);
  }

  private startHeartbeat(): void {
    setInterval(() => {
      this.eventBus.publish({
        type: 'HEARTBEAT',
        payload: {
          agentId: this.config.agentId,
          status: 'healthy',
          metrics: this.collectMetrics()
        }
      });
    }, this.config.heartbeatInterval);
  }
}
```

---

## Integration Points

### 1. Gmail Integration

```typescript
// gateway/adapters/GmailAdapter.ts

interface GmailConfig {
  clientId: string;
  clientSecret: string;
  refreshToken: string;
  pollInterval: number;
  labelFilter?: string;
}

class GmailAdapter implements ChannelAdapter {
  private gmail: gmail_v1.Gmail;
  private pollTimer: NodeJS.Timer;

  constructor(
    private config: GmailConfig,
    private eventBus: EventBus
  ) {}

  async initialize(): Promise<void> {
    const auth = new google.auth.OAuth2(
      this.config.clientId,
      this.config.clientSecret
    );
    auth.setCredentials({ refresh_token: this.config.refreshToken });
    
    this.gmail = google.gmail({ version: 'v1', auth });
  }

  async start(): Promise<void> {
    // Start polling for new emails
    this.pollTimer = setInterval(
      () => this.checkForNewEmails(),
      this.config.pollInterval
    );
  }

  private async checkForNewEmails(): Promise<void> {
    const response = await this.gmail.users.messages.list({
      userId: 'me',
      labelIds: this.config.labelFilter ? [this.config.labelFilter] : undefined,
      q: 'is:unread'
    });

    for (const message of response.data.messages || []) {
      const fullMessage = await this.gmail.users.messages.get({
        userId: 'me',
        id: message.id
      });

      const normalizedMessage = this.normalizeMessage(fullMessage.data);
      
      this.eventBus.publish({
        type: 'INBOUND_MESSAGE',
        payload: {
          channel: 'gmail',
          sender: normalizedMessage.from,
          content: normalizedMessage.body,
          metadata: {
            subject: normalizedMessage.subject,
            messageId: message.id,
            threadId: fullMessage.data.threadId
          }
        }
      });

      // Mark as read
      await this.gmail.users.messages.modify({
        userId: 'me',
        id: message.id,
        requestBody: {
          removeLabelIds: ['UNREAD']
        }
      });
    }
  }

  async sendResponse(recipient: string, content: string, metadata: any): Promise<void> {
    const message = [
      'Content-Type: text/plain; charset="UTF-8"',
      'MIME-Version: 1.0',
      'Content-Transfer-Encoding: 7bit',
      `To: ${recipient}`,
      `Subject: Re: ${metadata.subject}`,
      `In-Reply-To: ${metadata.messageId}`,
      `References: ${metadata.messageId}`,
      '',
      content
    ].join('\n');

    const encodedMessage = Buffer.from(message).toString('base64');
    
    await this.gmail.users.messages.send({
      userId: 'me',
      requestBody: {
        raw: encodedMessage,
        threadId: metadata.threadId
      }
    });
  }
}
```

### 2. Twilio Integration (Voice & SMS)

```typescript
// gateway/adapters/TwilioVoiceAdapter.ts & TwilioSMSAdapter.ts

interface TwilioConfig {
  accountSid: string;
  authToken: string;
  phoneNumber: string;
  webhookUrl: string;
}

class TwilioVoiceAdapter implements ChannelAdapter {
  private twilio: Twilio;
  private app: Express;

  constructor(
    private config: TwilioConfig,
    private eventBus: EventBus
  ) {
    this.twilio = new Twilio(config.accountSid, config.authToken);
    this.app = express();
  }

  async initialize(): Promise<void> {
    // Setup webhook endpoints
    this.app.post('/voice/webhook', express.urlencoded({ extended: false }), 
      (req, res) => this.handleVoiceWebhook(req, res));
    
    this.app.post('/voice/status', express.urlencoded({ extended: false }),
      (req, res) => this.handleStatusCallback(req, res));
  }

  private async handleVoiceWebhook(req: Request, res: Response): Promise<void> {
    const callSid = req.body.CallSid;
    const from = req.body.From;
    const speechResult = req.body.SpeechResult;

    if (speechResult) {
      // Convert speech to text and process
      this.eventBus.publish({
        type: 'INBOUND_MESSAGE',
        payload: {
          channel: 'twilio_voice',
          sender: from,
          content: speechResult,
          metadata: { callSid }
        }
      });
    }

    // Generate TwiML response
    const twiml = new VoiceResponse();
    twiml.say('Processing your request. Please wait.');
    twiml.gather({
      input: ['speech'],
      speechTimeout: 'auto',
      action: '/voice/webhook'
    });

    res.type('text/xml');
    res.send(twiml.toString());
  }

  async sendVoiceResponse(callSid: string, text: string): Promise<void> {
    // Use TTS to generate audio
    const audioUrl = await this.generateTTS(text);
    
    await this.twilio.calls(callSid).update({
      twiml: `
        <Response>
          <Play>${audioUrl}</Play>
          <Gather input="speech" speechTimeout="auto" action="/voice/webhook"/>
        </Response>
      `
    });
  }

  private async generateTTS(text: string): Promise<string> {
    // Delegate to TTSManager
    return await ttsManager.synthesize(text);
  }
}

class TwilioSMSAdapter implements ChannelAdapter {
  private twilio: Twilio;

  async handleSMSWebhook(req: Request, res: Response): Promise<void> {
    const from = req.body.From;
    const body = req.body.Body;
    const messageSid = req.body.MessageSid;

    this.eventBus.publish({
      type: 'INBOUND_MESSAGE',
      payload: {
        channel: 'twilio_sms',
        sender: from,
        content: body,
        metadata: { messageSid }
      }
    });

    res.status(200).send('OK');
  }

  async sendSMS(to: string, content: string): Promise<void> {
    await this.twilio.messages.create({
      body: content,
      from: this.config.phoneNumber,
      to
    });
  }
}
```

### 3. Browser Control Integration

```typescript
// execution/browser/BrowserController.ts

interface BrowserConfig {
  headless: boolean;
  executablePath?: string;
  userDataDir: string;
  viewport: { width: number; height: number };
}

class BrowserController {
  private browser: Browser;
  private pageManager: PageManager;
  private actionExecutor: ActionExecutor;

  constructor(private config: BrowserConfig) {}

  async initialize(): Promise<void> {
    this.browser = await puppeteer.launch({
      headless: this.config.headless,
      executablePath: this.config.executablePath,
      userDataDir: this.config.userDataDir,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });

    this.pageManager = new PageManager(this.browser);
    this.actionExecutor = new ActionExecutor(this.pageManager);
  }

  async executeAction(action: BrowserAction): Promise<BrowserResult> {
    switch (action.type) {
      case 'NAVIGATE':
        return this.actionExecutor.navigate(action.url);
      
      case 'CLICK':
        return this.actionExecutor.click(action.selector);
      
      case 'TYPE':
        return this.actionExecutor.type(action.selector, action.text);
      
      case 'SCROLL':
        return this.actionExecutor.scroll(action.direction, action.amount);
      
      case 'SCREENSHOT':
        return this.actionExecutor.screenshot(action.options);
      
      case 'EXTRACT':
        return this.actionExecutor.extract(action.selector);
      
      case 'WAIT':
        return this.actionExecutor.wait(action.condition, action.timeout);
      
      default:
        throw new Error(`Unknown browser action: ${action.type}`);
    }
  }

  async getCurrentState(): Promise<BrowserState> {
    return {
      url: await this.pageManager.getCurrentUrl(),
      title: await this.pageManager.getTitle(),
      screenshot: await this.actionExecutor.screenshot({ fullPage: false }),
      domSnapshot: await this.pageManager.getDOMSnapshot()
    };
  }
}
```

### 4. TTS/STT Integration

```typescript
// voice/TTSManager.ts & STTManager.ts

interface TTSConfig {
  provider: 'elevenlabs' | 'windows' | 'azure';
  voiceId: string;
  modelId?: string;
  apiKey?: string;
}

class TTSManager {
  private provider: TTSProvider;

  constructor(private config: TTSConfig) {
    this.provider = this.createProvider(config.provider);
  }

  private createProvider(type: string): TTSProvider {
    switch (type) {
      case 'elevenlabs':
        return new ElevenLabsProvider(this.config);
      case 'windows':
        return new WindowsTTSProvider();
      default:
        throw new Error(`Unknown TTS provider: ${type}`);
    }
  }

  async synthesize(text: string): Promise<AudioBuffer> {
    return this.provider.synthesize(text);
  }

  async synthesizeToFile(text: string, outputPath: string): Promise<string> {
    return this.provider.synthesizeToFile(text, outputPath);
  }
}

interface STTConfig {
  provider: 'whisper' | 'azure' | 'windows';
  language: string;
  apiKey?: string;
}

class STTManager {
  private provider: STTProvider;

  constructor(private config: STTConfig) {
    this.provider = this.createProvider(config.provider);
  }

  async transcribe(audioBuffer: Buffer): Promise<string> {
    return this.provider.transcribe(audioBuffer);
  }

  async transcribeFromFile(filePath: string): Promise<string> {
    return this.provider.transcribeFromFile(filePath);
  }
}
```

### 5. System Access Integration

```typescript
// execution/tools/implementations/SystemTool.ts

interface SystemToolConfig {
  allowedCommands: string[];
  blockedPaths: string[];
  maxExecutionTime: number;
  requireConfirmation: boolean;
}

class SystemTool extends BaseTool {
  name = 'system';
  description = 'Execute system commands with security controls';

  constructor(private config: SystemToolConfig) {
    super();
  }

  async execute(params: SystemToolParams): Promise<ToolResult> {
    // Validate command against allowlist
    if (!this.isCommandAllowed(params.command)) {
      return {
        success: false,
        error: `Command not allowed: ${params.command}`
      };
    }

    // Check for blocked paths
    if (this.containsBlockedPath(params.command)) {
      return {
        success: false,
        error: 'Command contains blocked paths'
      };
    }

    // Execute with timeout
    try {
      const result = await this.executeWithTimeout(
        params.command,
        this.config.maxExecutionTime
      );

      return {
        success: true,
        data: result
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  private isCommandAllowed(command: string): boolean {
    const baseCommand = command.split(' ')[0];
    return this.config.allowedCommands.includes(baseCommand);
  }

  private async executeWithTimeout(command: string, timeout: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = exec(command, (error, stdout, stderr) => {
        if (error) {
          reject(error);
        } else {
          resolve(stdout || stderr);
        }
      });

      setTimeout(() => {
        child.kill();
        reject(new Error('Execution timeout'));
      }, timeout);
    });
  }
}
```

---

## Agentic Loops Specification

### Agentic Loops (37 Scheduled Tasks)

```typescript
// core/loops/AgenticLoop.ts - Base Interface

interface AgenticLoop {
  name: string;
  description: string;
  
  // Main execution method
  execute(input: LoopInput, context: Context): Promise<LoopOutput>;
  
  // Determine if this loop should handle the input
  shouldHandle(intent: ParsedIntent): boolean;
  
  // Get loop-specific system prompt
  getSystemPrompt(): string;
}

interface LoopInput {
  message: string;
  intent: ParsedIntent;
  context: Context;
}

interface LoopOutput {
  response: string;
  actions: Action[];
  memoryUpdates: MemoryUpdate[];
  metrics: ExecutionMetrics;
}
```

### 1. Ralph Loop (Default Conversational)

```typescript
// core/loops/RalphLoop.ts

class RalphLoop implements AgenticLoop {
  name = 'ralph';
  description = 'Default conversational loop for general queries and chat';

  shouldHandle(intent: ParsedIntent): boolean {
    return intent.primaryIntent === 'QUERY' || 
           intent.primaryIntent === 'CONVERSATION' ||
           intent.confidence < 0.7;
  }

  getSystemPrompt(): string {
    return `
You are Ralph, a helpful AI assistant. Your goal is to provide helpful, 
accurate, and engaging responses to user queries.

Guidelines:
- Be conversational but informative
- Ask clarifying questions when needed
- Use context from previous messages
- Admit when you don't know something
- Be concise but thorough

Current user context: {{USER_CONTEXT}}
Session history: {{SESSION_HISTORY}}
`;
  }

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    // Simple direct response pattern
    const prompt = this.buildPrompt(input);
    const response = await this.llm.generate(prompt);
    
    return {
      response: response.text,
      actions: [],
      memoryUpdates: this.extractMemoryUpdates(input, response),
      metrics: { tokensUsed: response.tokens }
    };
  }
}
```

### 2. Research Loop

```typescript
// core/loops/ResearchLoop.ts

class ResearchLoop implements AgenticLoop {
  name = 'research';
  description = 'Deep research on topics using multiple sources';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const steps = [
      'DECOMPOSE_QUERY',
      'SEARCH_SOURCES',
      'GATHER_INFORMATION',
      'SYNTHESIZE_FINDINGS',
      'VERIFY_ACCURACY',
      'COMPILE_REPORT'
    ];

    const findings: ResearchFinding[] = [];
    
    for (const step of steps) {
      const result = await this.executeResearchStep(step, input, findings);
      findings.push(...result.findings);
    }

    return {
      response: this.compileResearchReport(findings),
      actions: findings.flatMap(f => f.actions),
      memoryUpdates: [{ key: 'research_history', value: findings }],
      metrics: { sourcesConsulted: findings.length }
    };
  }

  private async executeResearchStep(
    step: string, 
    input: LoopInput,
    previousFindings: ResearchFinding[]
  ): Promise<StepResult> {
    // Implementation for each research step
  }
}
```

### 3. Discovery Loop

```typescript
// core/loops/DiscoveryLoop.ts

class DiscoveryLoop implements AgenticLoop {
  name = 'discovery';
  description = 'Explore unknown domains and discover new information';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    // Pattern: Explore -> Document -> Connect -> Report
    
    const explorationPlan = await this.createExplorationPlan(input);
    const discoveries: Discovery[] = [];

    for (const exploration of explorationPlan) {
      const discovery = await this.explore(exploration);
      discoveries.push(discovery);
      
      // Update knowledge graph with new discovery
      await this.updateKnowledgeGraph(discovery);
    }

    return {
      response: this.synthesizeDiscoveries(discoveries),
      actions: discoveries.flatMap(d => d.actions),
      memoryUpdates: discoveries.map(d => ({
        key: `discovery:${d.topic}`,
        value: d
      })),
      metrics: { topicsExplored: discoveries.length }
    };
  }
}
```

### 4. Bug Finder Loop

```typescript
// core/loops/BugFinderLoop.ts

class BugFinderLoop implements AgenticLoop {
  name = 'bug-finder';
  description = 'Systematically find bugs in code or processes';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const target = this.identifyTarget(input);
    
    const analysisSteps = [
      'STATIC_ANALYSIS',
      'PATTERN_MATCHING',
      'EDGE_CASE_IDENTIFICATION',
      'LOGIC_VERIFICATION',
      'SECURITY_SCAN'
    ];

    const bugs: BugReport[] = [];

    for (const step of analysisSteps) {
      const foundBugs = await this.analyze(step, target);
      bugs.push(...foundBugs);
    }

    return {
      response: this.compileBugReport(bugs),
      actions: bugs.map(b => ({
        type: 'DOCUMENT_BUG',
        target: b.location,
        details: b
      })),
      memoryUpdates: [{ key: 'bug_reports', value: bugs }],
      metrics: { bugsFound: bugs.length }
    };
  }
}
```

### 5. Debugging Loop

```typescript
// core/loops/DebuggingLoop.ts

class DebuggingLoop implements AgenticLoop {
  name = 'debugging';
  description = 'Interactive debugging assistance';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const issue = this.parseIssue(input);
    
    const debugSteps = [
      'REPRODUCE_ISSUE',
      'ISOLATE_COMPONENT',
      'HYPOTHESIZE_CAUSE',
      'TEST_HYPOTHESIS',
      'VERIFY_FIX'
    ];

    const session: DebugSession = {
      issue,
      hypotheses: [],
      tests: [],
      findings: []
    };

    for (const step of debugSteps) {
      const result = await this.executeDebugStep(step, session);
      session.findings.push(result);
      
      if (result.resolved) break;
    }

    return {
      response: this.compileDebugReport(session),
      actions: session.findings.flatMap(f => f.actions),
      memoryUpdates: [{ key: 'debug_session', value: session }],
      metrics: { stepsTaken: session.findings.length }
    };
  }
}
```

### 6. End-to-End Loop

```typescript
// core/loops/EndToEndLoop.ts

class EndToEndLoop implements AgenticLoop {
  name = 'end-to-end';
  description = 'Complete task execution from start to finish';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    // Full task lifecycle
    const task: Task = {
      id: generateId(),
      description: input.message,
      status: 'PLANNING',
      steps: [],
      results: []
    };

    // Plan
    task.steps = await this.planTask(task);
    task.status = 'EXECUTING';

    // Execute each step
    for (const step of task.steps) {
      const result = await this.executeStep(step);
      task.results.push(result);
      
      if (result.error) {
        const recovery = await this.attemptRecovery(step, result.error);
        if (!recovery.success) {
          task.status = 'FAILED';
          break;
        }
      }
    }

    if (task.status !== 'FAILED') {
      task.status = 'COMPLETED';
    }

    return {
      response: this.compileTaskReport(task),
      actions: task.results.flatMap(r => r.actions),
      memoryUpdates: [{ key: `task:${task.id}`, value: task }],
      metrics: { 
        stepsCompleted: task.results.length,
        success: task.status === 'COMPLETED'
      }
    };
  }
}
```

### 7. Meta-Cognition Loop

```typescript
// core/loops/MetaCognitionLoop.ts

class MetaCognitionLoop implements AgenticLoop {
  name = 'meta-cognition';
  description = 'Self-reflection and cognitive improvement';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const reflection: MetaReflection = {
      timestamp: new Date(),
      triggers: this.identifyReflectionTriggers(context),
      observations: [],
      insights: [],
      improvements: []
    };

    // Observe own behavior
    reflection.observations = await this.observeBehavior(context);

    // Analyze patterns
    reflection.insights = await this.analyzePatterns(reflection.observations);

    // Generate improvements
    reflection.improvements = await this.generateImprovements(reflection.insights);

    // Apply improvements
    await this.applyImprovements(reflection.improvements);

    return {
      response: this.formatReflection(reflection),
      actions: reflection.improvements.map(i => ({
        type: 'SELF_MODIFY',
        target: i.target,
        change: i.change
      })),
      memoryUpdates: [
        { key: 'meta_reflection', value: reflection },
        { key: 'behavior_patterns', value: reflection.insights }
      ],
      metrics: { improvementsIdentified: reflection.improvements.length }
    };
  }
}
```

### 8. Exploration Loop

```typescript
// core/loops/ExplorationLoop.ts

class ExplorationLoop implements AgenticLoop {
  name = 'exploration';
  description = 'Explore unknown or uncertain scenarios';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const exploration: Exploration = {
      target: input.message,
      unknowns: this.identifyUnknowns(input),
      strategies: [],
      findings: []
    };

    // Select exploration strategy
    exploration.strategies = this.selectStrategies(exploration.unknowns);

    // Execute explorations
    for (const strategy of exploration.strategies) {
      const finding = await this.explore(strategy);
      exploration.findings.push(finding);
    }

    // Synthesize learnings
    const learnings = this.synthesizeLearnings(exploration.findings);

    return {
      response: this.formatExplorationReport(exploration, learnings),
      actions: exploration.findings.flatMap(f => f.actions),
      memoryUpdates: [
        { key: 'exploration_learnings', value: learnings },
        { key: 'unknown_domains', value: exploration.unknowns }
      ],
      metrics: { areasExplored: exploration.findings.length }
    };
  }
}
```

### 9. Self-Driven Loop

```typescript
// core/loops/SelfDrivenLoop.ts

class SelfDrivenLoop implements AgenticLoop {
  name = 'self-driven';
  description = 'Autonomous goal-directed behavior';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    // Determine own goals based on context
    const goals = await this.inferGoals(context);

    const actions: AutonomousAction[] = [];

    for (const goal of goals) {
      if (await this.shouldPursue(goal, context)) {
        const plan = await this.createAutonomousPlan(goal);
        const result = await this.executeAutonomousPlan(plan);
        actions.push(...result.actions);
      }
    }

    return {
      response: this.formatAutonomousReport(goals, actions),
      actions,
      memoryUpdates: goals.map(g => ({
        key: `autonomous_goal:${g.id}`,
        value: g
      })),
      metrics: { goalsPursued: goals.length, actionsTaken: actions.length }
    };
  }

  private async inferGoals(context: Context): Promise<Goal[]> {
    // Analyze context to infer what goals should be pursued
    const prompt = `
Based on the current context and user patterns, what goals should I pursue?
Context: ${JSON.stringify(context)}

Identify 1-3 high-value goals that would benefit the user.
`;
    return await this.llm.generateStructured(prompt, GoalSchema);
  }
}
```

### 10. Self-Learning Loop

```typescript
// core/loops/SelfLearningLoop.ts

class SelfLearningLoop implements AgenticLoop {
  name = 'self-learning';
  description = 'Learn from interactions and improve over time';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const learning: LearningSession = {
      source: this.identifyLearningSource(input, context),
      patterns: [],
      knowledge: [],
      skills: []
    };

    // Extract patterns
    learning.patterns = await this.extractPatterns(context);

    // Acquire knowledge
    learning.knowledge = await this.acquireKnowledge(learning.patterns);

    // Develop skills
    learning.skills = await this.developSkills(learning.knowledge);

    // Update models
    await this.updateBehaviorModels(learning);

    return {
      response: this.formatLearningReport(learning),
      actions: learning.skills.map(s => ({
        type: 'SKILL_ACQUISITION',
        skill: s
      })),
      memoryUpdates: [
        { key: 'learned_patterns', value: learning.patterns },
        { key: 'acquired_knowledge', value: learning.knowledge },
        { key: 'developed_skills', value: learning.skills }
      ],
      metrics: { 
        patternsLearned: learning.patterns.length,
        skillsDeveloped: learning.skills.length
      }
    };
  }
}
```

### 11. Self-Updating Loop

```typescript
// core/loops/SelfUpdatingLoop.ts

class SelfUpdatingLoop implements AgenticLoop {
  name = 'self-updating';
  description = 'Update own configuration and parameters';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const update: SelfUpdate = {
      target: this.identifyUpdateTarget(input),
      currentValue: null,
      proposedValue: null,
      rationale: '',
      safety: { canRollback: true, riskLevel: 'low' }
    };

    // Get current value
    update.currentValue = await this.getCurrentValue(update.target);

    // Propose new value
    const proposal = await this.proposeUpdate(update, context);
    update.proposedValue = proposal.value;
    update.rationale = proposal.rationale;

    // Safety check
    update.safety = await this.assessSafety(update);

    if (update.safety.riskLevel === 'low') {
      await this.applyUpdate(update);
    }

    return {
      response: this.formatUpdateReport(update),
      actions: [{
        type: 'SELF_UPDATE',
        target: update.target,
        change: update.proposedValue
      }],
      memoryUpdates: [
        { key: 'self_updates', value: update },
        { key: `config:${update.target}`, value: update.proposedValue }
      ],
      metrics: { updateApplied: update.safety.riskLevel === 'low' }
    };
  }
}
```

### 12. Self-Upgrading Loop

```typescript
// core/loops/SelfUpgradingLoop.ts

class SelfUpgradingLoop implements AgenticLoop {
  name = 'self-upgrading';
  description = 'Upgrade own capabilities and architecture';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const upgrade: SelfUpgrade = {
      type: this.identifyUpgradeType(input),
      components: [],
      plan: null,
      backup: null
    };

    // Identify components to upgrade
    upgrade.components = await this.identifyUpgradeableComponents();

    // Create upgrade plan
    upgrade.plan = await this.createUpgradePlan(upgrade.components);

    // Create backup
    upgrade.backup = await this.createBackup();

    // Execute upgrade
    const results = await this.executeUpgrade(upgrade);

    return {
      response: this.formatUpgradeReport(upgrade, results),
      actions: results.map(r => ({
        type: 'SELF_UPGRADE',
        component: r.component,
        result: r
      })),
      memoryUpdates: [
        { key: 'upgrade_history', value: upgrade },
        { key: 'system_version', value: this.getNewVersion() }
      ],
      metrics: { componentsUpgraded: results.length }
    };
  }
}
```

### 13. Planning Loop

```typescript
// core/loops/PlanningLoop.ts

class PlanningLoop implements AgenticLoop {
  name = 'planning';
  description = 'Create detailed multi-step plans';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const plan: Plan = {
      goal: input.message,
      constraints: this.identifyConstraints(context),
      steps: [],
      timeline: null,
      resources: []
    };

    // Decompose goal
    const subgoals = await this.decomposeGoal(plan.goal);

    // Create steps for each subgoal
    for (const subgoal of subgoals) {
      const steps = await this.createSteps(subgoal, plan.constraints);
      plan.steps.push(...steps);
    }

    // Order steps with dependencies
    plan.steps = this.orderStepsWithDependencies(plan.steps);

    // Estimate timeline
    plan.timeline = await this.estimateTimeline(plan.steps);

    // Identify required resources
    plan.resources = await this.identifyResources(plan.steps);

    return {
      response: this.formatPlan(plan),
      actions: plan.steps.map(s => ({
        type: 'PLAN_STEP',
        step: s
      })),
      memoryUpdates: [
        { key: 'active_plans', value: plan },
        { key: `plan:${plan.goal}`, value: plan }
      ],
      metrics: { stepsPlanned: plan.steps.length }
    };
  }
}
```

### 14. Context Engineering Loop

```typescript
// core/loops/ContextEngineeringLoop.ts

class ContextEngineeringLoop implements AgenticLoop {
  name = 'context-engineering';
  description = 'Optimize context for better performance';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const engineering: ContextEngineering = {
      currentContext: context,
      analysis: null,
      optimizations: [],
      optimizedContext: null
    };

    // Analyze current context
    engineering.analysis = await this.analyzeContext(context);

    // Identify optimizations
    engineering.optimizations = await this.identifyOptimizations(
      engineering.analysis
    );

    // Apply optimizations
    engineering.optimizedContext = await this.applyOptimizations(
      context,
      engineering.optimizations
    );

    // Validate improved context
    const validation = await this.validateContext(engineering.optimizedContext);

    return {
      response: this.formatEngineeringReport(engineering, validation),
      actions: engineering.optimizations.map(o => ({
        type: 'CONTEXT_OPTIMIZATION',
        optimization: o
      })),
      memoryUpdates: [
        { key: 'context_optimizations', value: engineering.optimizations },
        { key: 'optimized_context', value: engineering.optimizedContext }
      ],
      metrics: { 
        optimizationsApplied: engineering.optimizations.length,
        contextQuality: validation.score
      }
    };
  }
}
```

### 15. Context Prompt Engineering Loop

```typescript
// core/loops/ContextPromptEngineeringLoop.ts

class ContextPromptEngineeringLoop implements AgenticLoop {
  name = 'context-prompt-engineering';
  description = 'Optimize prompts for specific contexts';

  async execute(input: LoopInput, context: Context): Promise<LoopOutput> {
    const engineering: PromptEngineering = {
      target: this.identifyPromptTarget(input),
      currentPrompt: null,
      analysis: null,
      improvements: [],
      optimizedPrompt: null,
      testResults: []
    };

    // Get current prompt
    engineering.currentPrompt = await this.getCurrentPrompt(engineering.target);

    // Analyze prompt effectiveness
    engineering.analysis = await this.analyzePrompt(
      engineering.currentPrompt,
      context
    );

    // Generate improvements
    engineering.improvements = await this.generateImprovements(
      engineering.analysis
    );

    // Create optimized prompt
    engineering.optimizedPrompt = await this.createOptimizedPrompt(
      engineering.currentPrompt,
      engineering.improvements
    );

    // Test optimized prompt
    engineering.testResults = await this.testPrompt(
      engineering.optimizedPrompt
    );

    // Apply if tests pass
    if (this.testsPass(engineering.testResults)) {
      await this.applyPrompt(engineering.target, engineering.optimizedPrompt);
    }

    return {
      response: this.formatPromptEngineeringReport(engineering),
      actions: [{
        type: 'PROMPT_OPTIMIZATION',
        target: engineering.target,
        prompt: engineering.optimizedPrompt
      }],
      memoryUpdates: [
        { key: 'prompt_optimizations', value: engineering },
        { key: `prompt:${engineering.target}`, value: engineering.optimizedPrompt }
      ],
      metrics: { 
        improvementsMade: engineering.improvements.length,
        testPassRate: this.calculatePassRate(engineering.testResults)
      }
    };
  }
}
```

---

## Memory System

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         SESSION MEMORY (Short-term)                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Message   │    │   Context   │    │   Working   │    │   Temp      │      │   │
│  │  │   History   │    │   Window    │    │   Memory    │    │   State     │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  │                                                                                  │   │
│  │  Storage: In-memory (Redis optional)                                             │   │
│  │  Retention: Duration of session                                                  │   │
│  │  Capacity: Limited by context window                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        LONG-TERM MEMORY (Persistent)                             │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Facts     │    │   Events    │    │   Skills    │    │   User      │      │   │
│  │  │   & Data    │    │   & History │    │   Learned   │    │   Profile   │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  │                                                                                  │   │
│  │  Storage: File system (JSON/Markdown)                                            │   │
│  │  Retention: Permanent                                                            │   │
│  │  Format: Structured documents                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        VECTOR MEMORY (Semantic Search)                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │ Embeddings  │    │   Vector    │    │  Similarity │    │   Search    │      │   │
│  │  │  Generator  │    │   Store     │    │   Search    │    │   Index     │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  │                                                                                  │   │
│  │  Storage: ChromaDB / LanceDB (local)                                             │   │
│  │  Retention: Permanent                                                            │   │
│  │  Use Case: Semantic similarity search                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Types

```typescript
// memory/MemoryTypes.ts

// Session Memory (Short-term)
interface SessionMemory {
  sessionId: string;
  messages: Message[];
  context: SessionContext;
  workingMemory: WorkingMemory;
  createdAt: Date;
  lastActivity: Date;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

interface SessionContext {
  userId: string;
  channel: string;
  goals: string[];
  activeTools: string[];
  preferences: UserPreferences;
}

interface WorkingMemory {
  currentTask?: string;
  partialResults: any[];
  pendingActions: Action[];
  scratchpad: string;
}

// Long-term Memory
interface LongTermMemory {
  facts: Fact[];
  events: Event[];
  skills: Skill[];
  userProfiles: UserProfile[];
}

interface Fact {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: string;
  timestamp: Date;
}

interface UserProfile {
  userId: string;
  preferences: Record<string, any>;
  patterns: BehaviorPattern[];
  commonRequests: string[];
  communicationStyle: string;
}

// Vector Memory
interface VectorMemory {
  id: string;
  text: string;
  embedding: number[];
  metadata: {
    source: string;
    type: string;
    timestamp: Date;
  };
}
```

---

## Security & Sandboxing

### Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SECURITY ARCHITECTURE                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         PERMISSION LAYERS                                        │   │
│  │                                                                                  │   │
│  │  Level 1: READ-ONLY           (Safe operations, no side effects)               │   │
│  │  Level 2: FILE_OPERATIONS     (Read/write to workspace only)                   │   │
│  │  Level 3: NETWORK_ACCESS      (API calls, web requests)                        │   │
│  │  Level 4: SYSTEM_COMMANDS     (Shell commands, restricted)                     │   │
│  │  Level 5: FULL_ACCESS         (Requires explicit confirmation)                 │   │
│  │                                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         SANDBOX ENVIRONMENT                                      │   │
│  │                                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                     DOCKER CONTAINER                                     │    │   │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │    │   │
│  │  │  │  Isolated   │    │  Limited    │    │  Network    │                 │    │   │
│  │  │  │  Filesystem │    │  Resources  │    │  Restricted │                 │    │   │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘                 │    │   │
│  │  │                                                                          │    │   │
│  │  │  Mounts: /workspace (read-write)                                         │    │   │
│  │  │          /readonly (read-only)                                           │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         AUDIT LOGGING                                            │   │
│  │                                                                                  │   │
│  │  All actions logged with:                                                        │   │
│  │  - Timestamp                                                                     │   │
│  │  - Action type                                                                   │   │
│  │  - Parameters                                                                    │   │
│  │  - Result                                                                        │   │
│  │  - User/session ID                                                               │   │
│  │  - Permission level required                                                     │   │
│  │                                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Files

### SOUL.md (Personality Definition)

```markdown
# SOUL.md - Agent Personality Definition

## Core Identity
- **Name**: [Agent Name]
- **Nature**: Helpful, curious, autonomous AI assistant
- **Purpose**: To be a capable partner in achieving user goals

## Personality Traits
- Curious and eager to learn
- Thoughtful and analytical
- Direct but kind in communication
- Proactive in identifying opportunities
- Humble about limitations

## Communication Style
- Clear and concise
- Uses appropriate technical depth
- Asks clarifying questions when needed
- Provides context for decisions
- Admits uncertainty honestly

## Values
- User autonomy and privacy
- Continuous improvement
- Transparency in actions
- Safety and responsibility

## Behavioral Guidelines
- Always confirm before destructive actions
- Proactively suggest improvements
- Learn from user feedback
- Maintain appropriate boundaries
```

### IDENTITY.md (Presentation)

```markdown
# IDENTITY.md - Agent Presentation

## Public Identity
- **Display Name**: [Name]
- **Version**: [Version]
- **Capabilities**: [List of key capabilities]

## Introduction Template
"Hello! I'm [Name], your AI assistant. I can help you with:
- Research and information gathering
- Task automation and execution
- Code development and debugging
- System administration
- And much more!

What would you like to work on today?"

## Response Formatting
- Use markdown for structure
- Include code blocks with language tags
- Use emoji sparingly and appropriately
- Format long responses with sections
```

### USER.md (User Context)

```markdown
# USER.md - User Context

## User Profile
- **Name**: [User Name]
- **Preferences**: [Key preferences]
- **Technical Level**: [beginner/intermediate/advanced]
- **Communication Style**: [formal/casual/technical]

## Common Tasks
1. [Task 1]
2. [Task 2]
3. [Task 3]

## Important Context
- [Key information about user's work]
- [Relevant projects or goals]
- [Preferred tools and workflows]

## Access Permissions
- [List of approved actions]
- [Confirmation requirements]
```

### MEMORY.md (Long-term Memory)

```markdown
# MEMORY.md - Long-term Memory

## Key Facts
- [Important facts learned]

## Past Interactions
- [Summary of significant conversations]

## Learned Patterns
- [Behavioral patterns identified]

## User Preferences
- [Preferences learned over time]

## Successful Strategies
- [Approaches that worked well]
```

### AGENTS.md (Agent Instructions)

```markdown
# AGENTS.md - Agent Instructions

## Available Agentic Loops

### @ralph - Default Assistant
Use for: General queries, conversation, simple tasks

### @research - Research Mode
Use for: Deep research, information gathering
Instructions: Decompose query, search multiple sources, synthesize findings

### @debugging - Debug Mode
Use for: Troubleshooting, bug fixing
Instructions: Reproduce, isolate, hypothesize, test, verify

### @planning - Planning Mode
Use for: Complex multi-step tasks
Instructions: Break down, sequence, estimate, execute

## Loop Selection Guidelines
- Use @research for information gathering tasks
- Use @debugging for error investigation
- Use @planning for complex multi-step tasks
- Default to @ralph for everything else
```

### HEARTBEAT.md (Scheduled Actions)

```markdown
# HEARTBEAT.md - Scheduled Actions

## Cron Jobs

### Health Check
- **Schedule**: */5 * * * * (every 5 minutes)
- **Action**: Check system health, report issues
- **Notify**: On degradation only

### Memory Consolidation
- **Schedule**: 0 */6 * * * (every 6 hours)
- **Action**: Consolidate short-term memories to long-term
- **Notify**: Never

### Self-Update Check
- **Schedule**: 0 0 * * * (daily at midnight)
- **Action**: Check for updates, apply if safe
- **Notify**: On update completion

### System Maintenance
- **Schedule**: 0 2 * * 0 (weekly, Sunday 2am)
- **Action**: Clean logs, optimize storage
- **Notify**: Never

## Heartbeat Configuration
- **Interval**: 60 seconds
- **Metrics**: CPU, memory, active sessions
- **Alert Thresholds**: CPU > 80%, Memory > 85%
```

---

## Summary

This architecture specification defines a comprehensive, production-grade AI agent runtime system for Windows 10. Key design decisions:

1. **Event-Driven Architecture**: Enables loose coupling, scalability, and reliability
2. **Layered Separation**: Clear boundaries between Gateway, Core, and Execution
3. **37 scheduled tasks (15 operational, 16 cognitive, 6 cron)**: Specialized behavior patterns for different task types
4. **Robust Memory System**: Multi-tier memory with vector search capabilities
5. **Security-First**: Sandboxed execution with permission layers
6. **Extensible Design**: Plugin-based skills and configurable providers
7. **24/7 Operation**: Cron jobs, heartbeat monitoring, self-maintenance

The system is designed to be:
- **Scalable**: Handle multiple concurrent sessions
- **Reliable**: Graceful error handling and recovery
- **Observable**: Comprehensive logging and metrics
- **Secure**: Sandboxed execution with audit trails
- **Extensible**: Easy to add new capabilities

---

*Document Version: 1.0*  
*Last Updated: 2025*  
*Author: AI Systems Architect*
