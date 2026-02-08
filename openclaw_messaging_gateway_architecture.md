# OPENCLAW MESSAGING GATEWAY ARCHITECTURE
## Windows 10 AI Agent System - Technical Specification

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Core Architecture Overview](#2-core-architecture-overview)
3. [Multi-Channel Message Routing System](#3-multi-channel-message-routing-system)
4. [Protocol Adapters](#4-protocol-adapters)
5. [Message Format Normalization](#5-message-format-normalization)
6. [Connection Management](#6-connection-management)
7. [Message Queuing & Delivery Guarantees](#7-message-queuing--delivery-guarantees)
8. [Real-Time vs Polling Strategies](#8-real-time-vs-polling-strategies)
9. [Authentication & Credential Management](#9-authentication--credential-management)
10. [Implementation Code](#10-implementation-code)
11. [Security Considerations](#11-security-considerations)
12. [Deployment Configuration](#12-deployment-configuration)

---

## 1. EXECUTIVE SUMMARY

The OpenClaw Messaging Gateway is a multi-platform communication interface designed for Windows 10 AI agent systems. It provides seamless integration with Telegram, WhatsApp, Discord, Slack, iMessage, and Signal, enabling users to interact with the AI agent through familiar chat applications rather than a dedicated UI.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Platform Support** | Telegram, WhatsApp, Discord, Slack, iMessage, Signal |
| **Message Normalization** | Unified message format across all platforms |
| **Real-Time Communication** | WebSocket-based connections where supported |
| **Delivery Guarantees** | At-least-once delivery with idempotency |
| **Auto-Reconnection** | Exponential backoff with circuit breaker pattern |
| **Security** | End-to-end encryption, credential isolation |

---

## 2. CORE ARCHITECTURE OVERVIEW

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPENCLAW MESSAGING GATEWAY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Telegram   │  │   WhatsApp   │  │   Discord    │  │    Slack     │    │
│  │   Adapter    │  │   Adapter    │  │   Adapter    │  │   Adapter    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐    │
│  │   iMessage   │  │    Signal    │  │   Twilio     │  │    Gmail     │    │
│  │   Adapter    │  │   Adapter    │  │   Adapter    │  │   Adapter    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │      MESSAGE ROUTER          │                        │
│                    │   (Unified Routing Logic)    │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │   MESSAGE NORMALIZER         │                        │
│                    │ (Canonical Format Converter) │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │      MESSAGE QUEUE           │                        │
│                    │   (Priority & Persistence)   │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │      AGENT RUNTIME           │                        │
│                    │   (GPT-5.2 Processing)       │                        │
│                    └───────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
User Message → Platform API → Protocol Adapter → Message Router 
    → Normalizer → Queue → Agent Runtime → Response Queue 
    → Router → Adapter → Platform API → User
```

### 2.3 Core Services

| Service | Responsibility | Technology |
|---------|---------------|------------|
| Gateway Server | Central coordination, session management | Node.js / Python |
| Message Router | Route messages between channels and agent | Event-driven |
| Normalizer | Convert platform-specific to canonical format | Transformation layer |
| Queue Manager | Message persistence, retry logic | Redis / Bull |
| Connection Manager | Handle connections, reconnection, heartbeats | WebSocket clients |
| Credential Vault | Secure storage of API keys/tokens | Windows DPAPI |

---

## 3. MULTI-CHANNEL MESSAGE ROUTING SYSTEM

### 3.1 Routing Architecture

The Message Router is the central component that:
- Routes inbound messages from platforms to the Agent Runtime
- Routes outbound responses from Agent to appropriate platforms
- Manages session state and user context
- Implements retry logic with exponential backoff

### 3.2 Routing Logic Implementation

```typescript
interface RouteConfig {
  channelId: string;
  sessionId: string;
  userId: string;
  priority: MessagePriority;
  retryPolicy: RetryPolicy;
}

enum MessagePriority {
  CRITICAL = 0,  // System alerts, errors
  HIGH = 1,      // User commands
  NORMAL = 2,    // Regular messages
  LOW = 3        // Background tasks
}

interface RetryPolicy {
  maxAttempts: number;
  backoffStrategy: 'fixed' | 'exponential' | 'linear';
  initialDelayMs: number;
  maxDelayMs: number;
}
```

### 3.3 Session Management

Sessions are maintained per user per platform with:
- 30-minute timeout for inactive sessions
- Conversation history preservation
- User preference storage
- Agent state context

---

## 4. PROTOCOL ADAPTERS

### 4.1 Adapter Architecture Pattern

All adapters implement the `ChannelAdapter` interface:

```typescript
interface ChannelAdapter {
  readonly platform: string;
  readonly connectionType: 'websocket' | 'webhook' | 'polling';

  // Connection Management
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;

  // Message Handling
  transformInbound(raw: any): NormalizedMessage;
  transformOutbound(response: AgentResponse): PlatformMessage;
  send(message: PlatformMessage): Promise<void>;

  // Event Handling
  onMessage(handler: MessageHandler): void;

  // Configuration
  getRetryPolicy(): RetryPolicy;
  getRateLimit(): RateLimitConfig;
}
```

### 4.2 Platform Communication Strategies

| Platform | Primary Method | Fallback | Rate Limit |
|----------|---------------|----------|------------|
| Telegram | Webhook | Long Polling | 30 req/s |
| WhatsApp | Webhook | N/A (required) | 20 req/s |
| Discord | WebSocket Gateway | N/A | 5 req/s |
| Slack | Socket Mode (WebSocket) | Events API | 10 req/s |
| Signal | WebSocket (signal-cli) | REST Polling | 10 req/s |
| iMessage | WebSocket (BlueBubbles) | HTTP Polling | 5 req/s |

### 4.3 Telegram Bot API Adapter

- Uses Telegraf library
- Supports webhook (production) and long polling (development)
- Handles text, voice, documents, photos
- Command handlers for /start, /status, /help

### 4.4 WhatsApp Business API Adapter

- Uses Facebook Graph API v18.0
- Webhook-based with signature verification
- Supports templates, interactive messages
- Media download capability

### 4.5 Discord.js Adapter

- Uses discord.js library
- Full WebSocket gateway connection
- Supports embeds, components, attachments
- DM and guild channel support

### 4.6 Slack Bolt Adapter

- Uses @slack/bolt with Socket Mode
- Block Kit for rich formatting
- Interactive components support
- Ephemeral messages for private responses

### 4.7 Signal Adapter

- Uses signal-cli-rest-api via WebSocket
- JSON-RPC communication
- Group message support
- Reaction handling

### 4.8 iMessage Adapter (Windows Bridge)

- Uses BlueBubbles server bridge
- WebSocket connection to macOS relay
- Attachment download support
- Read receipt tracking

---

## 5. MESSAGE FORMAT NORMALIZATION

### 5.1 Canonical Message Schema

```typescript
interface NormalizedMessage {
  id: string;                    // Unique message ID
  platform: Platform;            // Source platform
  channel: string;               // Channel/Chat ID
  userId: string;                // Platform-specific user ID
  username?: string;             // Username/handle
  displayName?: string;          // Display name
  timestamp: Date;
  isDirectMessage?: boolean;
  isGroupMessage?: boolean;
  groupId?: string;

  content: {
    type: ContentType;
    text?: string;
    caption?: string;
    attachments?: Attachment[];
  };

  threadId?: string;
  replyTo?: string;
  mentions?: {
    users?: string[];
    roles?: string[];
    channels?: string[];
    everyone?: boolean;
  };

  raw: any;                      // Original platform data
}
```

### 5.2 Content Types

- text, image, video, audio, voice
- document, location, contact
- sticker, embed, interactive

---

## 6. CONNECTION MANAGEMENT

### 6.1 Connection Manager Features

- Centralized connection state tracking
- Automatic reconnection with exponential backoff
- Health checks every 30 seconds
- Circuit breaker pattern for resilience
- Connection metrics (messages sent/received, errors)

### 6.2 Connection States

- DISCONNECTED, CONNECTING, CONNECTED
- RECONNECTING, ERROR, PERMANENTLY_FAILED

### 6.3 Reconnection Strategy

- Max 10 reconnection attempts
- Exponential backoff: 1s, 2s, 4s, 8s... up to 60s max
- Circuit breaker opens after 5 consecutive failures
- Half-open state tests recovery before full reset

---

## 7. MESSAGE QUEUING & DELIVERY GUARANTEES

### 7.1 Queue Architecture

- Redis-backed priority queues
- Four priority levels: CRITICAL, HIGH, NORMAL, LOW
- Scheduled message support for delayed delivery
- At-least-once delivery with idempotency

### 7.2 Queue Processing

- Messages stored with 24-hour TTL
- Exponential backoff retry: 2^attempt * 1000ms
- Max 3 attempts before moving to failed queue
- Scheduled queue processed before priority queues

### 7.3 Delivery Guarantees

- Duplicate detection via Redis
- Processing state tracking (5-minute TTL)
- Delivery confirmation tracking (24-hour TTL)
- Failed message persistence for debugging

---

## 8. REAL-TIME VS POLLING STRATEGIES

### 8.1 Strategy Selection by Platform

| Platform | Strategy | Latency |
|----------|----------|---------|
| Telegram | Webhook | <100ms |
| WhatsApp | Webhook | <200ms |
| Discord | WebSocket | <50ms |
| Slack | WebSocket | <100ms |
| Signal | WebSocket | <100ms |
| iMessage | WebSocket | <150ms |

### 8.2 Fallback Strategies

- Webhook → Long Polling (Telegram)
- WebSocket → HTTP Polling (Signal, iMessage)
- Automatic failover on connection failure

---

## 9. AUTHENTICATION & CREDENTIAL MANAGEMENT

### 9.1 Credential Vault

- Windows DPAPI / Electron safeStorage encryption
- Per-platform credential isolation
- Automatic expiration handling
- Environment variable fallback for development

### 9.2 Token Refresh Management

- OAuth token refresh 5 minutes before expiry
- Automatic scheduling of refresh operations
- Failure notification and retry

### 9.3 Platform Credentials

```typescript
interface TelegramCredentials {
  botToken: string;
  webhookUrl?: string;
  webhookPort?: number;
}

interface WhatsAppCredentials {
  accessToken: string;
  phoneNumberId: string;
  businessAccountId: string;
  webhookSecret: string;
  verifyToken: string;
}

interface DiscordCredentials {
  botToken: string;
  applicationId: string;
  publicKey?: string;
}

interface SlackCredentials {
  botToken: string;
  signingSecret: string;
  appToken: string;
}

interface SignalCredentials {
  signalServiceUrl: string;
  phoneNumber: string;
}

interface iMessageCredentials {
  serverUrl: string;
  password: string;
}
```

---

## 10. IMPLEMENTATION CODE

### 10.1 Gateway Server Structure

```typescript
class OpenClawGateway {
  private connectionManager: ConnectionManager;
  private messageRouter: MessageRouter;
  private messageNormalizer: MessageNormalizer;
  private messageQueue: MessageQueue;
  private credentialVault: CredentialVault;
  private adapters: Map<string, ChannelAdapter>;

  async initialize(): Promise<void>
  async start(): Promise<void>
  async stop(): Promise<void>
}
```

### 10.2 Main Entry Point

```typescript
async function main() {
  const gateway = new OpenClawGateway({
    port: parseInt(process.env.GATEWAY_PORT || '3000'),
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379'
  });

  // Graceful shutdown handlers
  process.on('SIGINT', async () => { await gateway.stop(); });
  process.on('SIGTERM', async () => { await gateway.stop(); });

  await gateway.start();
}
```

---

## 11. SECURITY CONSIDERATIONS

### 11.1 Security Checklist

| Category | Requirement | Implementation |
|----------|-------------|----------------|
| Authentication | API key validation | Per-platform token verification |
| Authorization | User allowlisting | Configurable allowed users per platform |
| Encryption (at rest) | Windows DPAPI / Electron safeStorage |
| Encryption (in transit) | TLS 1.3 for all connections |
| Webhook Security | HMAC signature verification |
| Input Validation | Remove control characters, limit size |
| Rate Limiting | Per-user, per-platform limits |
| Audit Logging | Anonymized logs for debugging |

### 11.2 Security Implementation

- Webhook signature verification using timing-safe comparison
- Message sanitization (remove null bytes, control chars)
- Rate limiting per platform adapter
- User allowlisting for access control

---

## 12. DEPLOYMENT CONFIGURATION

### 12.1 Environment Variables

```
GATEWAY_PORT=3000
REDIS_URL=redis://localhost:6379
TELEGRAM_BOT_TOKEN=xxx
WHATSAPP_ACCESS_TOKEN=xxx
DISCORD_BOT_TOKEN=xxx
SLACK_BOT_TOKEN=xxx
SIGNAL_SERVICE_URL=http://localhost:8080
IMESSAGE_SERVER_URL=http://localhost:3001
```

### 12.2 Docker Compose

Services:
- gateway: Main messaging gateway
- redis: Message queue persistence
- signal-cli: Signal bridge service

### 12.3 Windows Service

Install as Windows Service with:
- Automatic startup
- Recovery actions on failure
- Dependency on TCP/IP

---

## APPENDIX A: DATA FLOW DIAGRAMS

### Inbound Message Flow
```
User → Platform API → Adapter → Normalizer → Queue → Agent Runtime
```

### Outbound Message Flow
```
Agent Runtime → Router → Denormalizer → Adapter → Platform API → User
```

---

## APPENDIX B: ERROR CODES

| Code | Description | Action |
|------|-------------|--------|
| CONN_001 | Connection failed | Retry with exponential backoff |
| CONN_002 | Authentication failed | Check credentials |
| CONN_003 | Rate limit exceeded | Backoff, queue messages |
| CONN_004 | Webhook verification failed | Check secret/token |
| MSG_001 | Message transformation failed | Log, skip message |
| MSG_002 | Message delivery failed | Retry, then queue |
| MSG_003 | Message too large | Split or reject |
| SEC_001 | Unauthorized user | Reject, log attempt |
| SEC_002 | Invalid signature | Reject webhook |

---

## APPENDIX C: PERFORMANCE METRICS

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Message latency (p50) | < 100ms | > 500ms |
| Message latency (p99) | < 500ms | > 2000ms |
| Connection uptime | > 99.9% | < 99% |
| Queue depth | < 100 | > 1000 |
| Error rate | < 0.1% | > 1% |
| Reconnection rate | < 1/hour | > 10/hour |

---

*Document Version: 1.0*
*Last Updated: 2026-02-06*
*Author: OpenClaw Architecture Team*
