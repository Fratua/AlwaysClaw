# Twilio Voice & SMS Integration Technical Specification
## OpenClaw Windows 10 AI Agent System

**Version:** 1.0  
**Date:** 2025  
**Framework:** Node.js / TypeScript  
**Target Platform:** Windows 10  
**AI Model:** GPT-5.2 with High Thinking Capability

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Twilio Voice API Integration](#3-twilio-voice-api-integration)
4. [Twilio SMS/MMS API Integration](#4-twilio-smsmms-api-integration)
5. [Webhook Handling System](#5-webhook-handling-system)
6. [TwiML Call Flow Management](#6-twiml-call-flow-management)
7. [Phone Number Management](#7-phone-number-management)
8. [Call Recording & Transcription](#8-call-recording--transcription)
9. [TTS/STT Integration](#9-ttsstt-integration)
10. [Media Streams for Real-Time AI](#10-media-streams-for-real-time-ai)
11. [Security & Authentication](#11-security--authentication)
12. [Implementation Code Examples](#12-implementation-code-examples)
13. [Configuration & Environment](#13-configuration--environment)
14. [Error Handling & Monitoring](#14-error-handling--monitoring)

---

## 1. Executive Summary

This specification defines the complete Twilio integration architecture for the OpenClaw Windows 10 AI agent system. The integration enables:

- **Real-time voice calls** with bidirectional audio streaming
- **SMS/MMS messaging** with automated responses
- **Phone number management** (purchase, configure, release)
- **Call recording and transcription** with AI analysis
- **TTS/STT integration** for natural voice interactions
- **Media Streams** for direct AI voice assistant capabilities

### Key Technologies

| Component | Technology |
|-----------|------------|
| Twilio SDK | `twilio` v5.x (Node.js) |
| Web Server | Express.js / Fastify |
| WebSocket | Native WS / `@fastify/websocket` |
| TTS | Amazon Polly / Google Chirp3-HD |
| STT | OpenAI Whisper / Deepgram |
| Real-time AI | OpenAI Realtime API |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPENCLAW AI AGENT SYSTEM                             │
│                           (Windows 10 / Node.js)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Agent Core    │    │   GPT-5.2       │    │   Memory/Soul   │         │
│  │   Controller    │◄──►│   Engine        │◄──►│   System        │         │
│  └────────┬────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
│  ┌────────▼─────────────────────────────────────────────────────────┐      │
│  │                    TWILIO INTEGRATION LAYER                       │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │      │
│  │  │ Voice API    │  │ SMS/MMS API  │  │ Phone Number Mgmt    │   │      │
│  │  │ Manager      │  │ Manager      │  │ Service              │   │      │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │      │
│  │         │                 │                      │               │      │
│  │  ┌──────▼─────────────────▼──────────────────────▼───────────┐   │      │
│  │  │              Webhook Handler & Router                      │   │      │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │   │      │
│  │  │  │ /voice     │  │ /sms       │  │ /status-callback   │   │   │      │
│  │  │  │ /recording │  │ /mms       │  │ /media-stream      │   │   │      │
│  │  │  └────────────┘  └────────────┘  └────────────────────┘   │   │      │
│  │  └───────────────────────────────────────────────────────────┘   │      │
│  │                                                                   │      │
│  │  ┌───────────────────────────────────────────────────────────┐   │      │
│  │  │              TwiML Generator & Call Flow Engine             │   │      │
│  │  └───────────────────────────────────────────────────────────┘   │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TWILIO CLOUD                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Programmable │  │ Programmable │  │         Media            │  │   │
│  │  │    Voice     │  │  Messaging   │  │        Streams           │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                           ┌──────────────┐                                  │
│                           │  PSTN / SMS  │                                  │
│                           │   Network    │                                  │
│                           └──────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Twilio Voice API Integration

### 3.1 SDK Installation & Setup

```bash
# Install Twilio Node.js SDK
npm install twilio@^5.0.0

# Additional dependencies
npm install ws @fastify/websocket fastify express dotenv
npm install @openai/agents @openai/agents-extensions  # For Realtime API
```

### 3.2 Voice Client Configuration

```typescript
// src/twilio/voice/voice-client.ts
import twilio from 'twilio';

export interface TwilioConfig {
  accountSid: string;
  authToken: string;
  apiKey?: string;
  apiSecret?: string;
}

export class TwilioVoiceClient {
  private client: twilio.Twilio;
  
  constructor(config: TwilioConfig) {
    this.client = twilio(config.accountSid, config.authToken);
  }

  /**
   * Make an outbound voice call
   */
  async makeCall(options: {
    to: string;
    from: string;
    twimlUrl?: string;
    twiml?: string;
    statusCallback?: string;
    recording?: boolean;
  }): Promise<twilio.rest.api.v2010.account.Call> {
    const callOptions: any = {
      to: options.to,
      from: options.from,
    };

    if (options.twimlUrl) {
      callOptions.url = options.twimlUrl;
    } else if (options.twiml) {
      callOptions.twiml = options.twiml;
    }

    if (options.statusCallback) {
      callOptions.statusCallback = options.statusCallback;
      callOptions.statusCallbackEvent = ['initiated', 'ringing', 'answered', 'completed'];
      callOptions.statusCallbackMethod = 'POST';
    }

    if (options.recording) {
      callOptions.record = true;
      callOptions.recordingStatusCallback = `${options.statusCallback}/recording`;
      callOptions.recordingStatusCallbackMethod = 'POST';
    }

    return this.client.calls.create(callOptions);
  }

  /**
   * Get call details
   */
  async getCall(callSid: string): Promise<twilio.rest.api.v2010.account.Call> {
    return this.client.calls(callSid).fetch();
  }

  /**
   * Update an active call
   */
  async updateCall(callSid: string, twiml: string): Promise<twilio.rest.api.v2010.account.Call> {
    return this.client.calls(callSid).update({ twiml });
  }

  /**
   * End an active call
   */
  async endCall(callSid: string): Promise<twilio.rest.api.v2010.account.Call> {
    return this.client.calls(callSid).update({ status: 'completed' });
  }

  /**
   * Get call recordings
   */
  async getRecordings(callSid: string): Promise<twilio.rest.api.v2010.account.recording.RecordingInstance[]> {
    return this.client.calls(callSid).recordings.list();
  }
}
```

### 3.3 Voice Response Handler (TwiML)

```typescript
// src/twilio/voice/voice-response.ts
import { VoiceResponse } from 'twilio';

export interface VoicePromptOptions {
  message: string;
  voice?: string;
  language?: string;
  gatherDigits?: boolean;
  gatherSpeech?: boolean;
  gatherTimeout?: number;
  actionUrl?: string;
}

export class TwilioVoiceResponse {
  private response: VoiceResponse;

  constructor() {
    this.response = new VoiceResponse();
  }

  /**
   * Generate greeting with TTS
   */
  say(message: string, options: {
    voice?: string;
    language?: string;
  } = {}): this {
    const voice = options.voice || 'Polly.Joanna-Neural';
    const language = options.language || 'en-US';
    
    this.response.say({ voice, language }, message);
    return this;
  }

  /**
   * Play audio file
   */
  play(url: string): this {
    this.response.play(url);
    return this;
  }

  /**
   * Pause for specified seconds
   */
  pause(seconds: number = 1): this {
    this.response.pause({ length: seconds });
    return this;
  }

  /**
   * Gather user input (DTMF or speech)
   */
  gather(options: {
    action?: string;
    method?: string;
    timeout?: number;
    numDigits?: number;
    input?: ('dtmf' | 'speech' | 'dtmf speech')[];
    speechTimeout?: string;
    speechModel?: string;
    hints?: string;
  } = {}): VoiceResponse['gather'] {
    const gatherOptions: any = {
      action: options.action,
      method: options.method || 'POST',
      timeout: options.timeout || 5,
    };

    if (options.numDigits) {
      gatherOptions.numDigits = options.numDigits;
    }

    if (options.input) {
      gatherOptions.input = options.input.join(' ');
    }

    if (options.speechTimeout) {
      gatherOptions.speechTimeout = options.speechTimeout;
    }

    if (options.speechModel) {
      gatherOptions.speechModel = options.speechModel;
    }

    if (options.hints) {
      gatherOptions.hints = options.hints;
    }

    return this.response.gather(gatherOptions);
  }

  /**
   * Connect to Media Stream for real-time AI
   */
  connectToMediaStream(websocketUrl: string): this {
    const connect = this.response.connect();
    connect.stream({ url: websocketUrl });
    return this;
  }

  /**
   * Record the call
   */
  record(options: {
    action?: string;
    method?: string;
    timeout?: number;
    finishOnKey?: string;
    maxLength?: number;
    playBeep?: boolean;
    recordingStatusCallback?: string;
  } = {}): this {
    const recordOptions: any = {
      timeout: options.timeout || 5,
      maxLength: options.maxLength || 3600,
      playBeep: options.playBeep !== false,
    };

    if (options.action) recordOptions.action = options.action;
    if (options.method) recordOptions.method = options.method;
    if (options.finishOnKey) recordOptions.finishOnKey = options.finishOnKey;
    if (options.recordingStatusCallback) {
      recordOptions.recordingStatusCallback = options.recordingStatusCallback;
    }

    this.response.record(recordOptions);
    return this;
  }

  /**
   * Dial another number
   */
  dial(number: string, options: {
    callerId?: string;
    record?: boolean;
    timeout?: number;
  } = {}): this {
    const dialOptions: any = {};
    if (options.callerId) dialOptions.callerId = options.callerId;
    if (options.record) dialOptions.record = options.record;
    if (options.timeout) dialOptions.timeout = options.timeout;

    this.response.dial(dialOptions, number);
    return this;
  }

  /**
   * Hang up the call
   */
  hangup(): this {
    this.response.hangup();
    return this;
  }

  /**
   * Redirect to another TwiML URL
   */
  redirect(url: string, method: string = 'POST'): this {
    this.response.redirect({ method }, url);
    return this;
  }

  /**
   * Get XML string
   */
  toString(): string {
    return this.response.toString();
  }

  /**
   * Get raw response object
   */
  getResponse(): VoiceResponse {
    return this.response;
  }
}
```

---

## 4. Twilio SMS/MMS API Integration

### 4.1 SMS/MMS Client

```typescript
// src/twilio/sms/sms-client.ts
import twilio from 'twilio';

export interface SendSMSOptions {
  to: string;
  from: string;
  body: string;
  mediaUrl?: string[];  // For MMS
  statusCallback?: string;
  messagingServiceSid?: string;
}

export interface SendBulkSMSOptions {
  to: string[];
  from: string;
  body: string;
  messagingServiceSid?: string;
}

export class TwilioSMSClient {
  private client: twilio.Twilio;

  constructor(accountSid: string, authToken: string) {
    this.client = twilio(accountSid, authToken);
  }

  /**
   * Send single SMS/MMS message
   */
  async sendMessage(options: SendSMSOptions): Promise<twilio.rest.api.v2010.account.MessageInstance> {
    const messageOptions: any = {
      to: options.to,
      from: options.from,
      body: options.body,
    };

    if (options.mediaUrl && options.mediaUrl.length > 0) {
      messageOptions.mediaUrl = options.mediaUrl;
    }

    if (options.statusCallback) {
      messageOptions.statusCallback = options.statusCallback;
    }

    if (options.messagingServiceSid) {
      messageOptions.messagingServiceSid = options.messagingServiceSid;
      delete messageOptions.from;
    }

    return this.client.messages.create(messageOptions);
  }

  /**
   * Send bulk SMS messages
   */
  async sendBulkMessages(options: SendBulkSMSOptions): Promise<Promise<twilio.rest.api.v2010.account.MessageInstance>[]> {
    const promises = options.to.map(recipient => 
      this.sendMessage({
        to: recipient,
        from: options.from,
        body: options.body,
        messagingServiceSid: options.messagingServiceSid,
      })
    );

    return Promise.allSettled(promises);
  }

  /**
   * Get message details
   */
  async getMessage(messageSid: string): Promise<twilio.rest.api.v2010.account.MessageInstance> {
    return this.client.messages(messageSid).fetch();
  }

  /**
   * List messages
   */
  async listMessages(options: {
    to?: string;
    from?: string;
    dateSentAfter?: Date;
    dateSentBefore?: Date;
    limit?: number;
  } = {}): Promise<twilio.rest.api.v2010.account.MessageInstance[]> {
    return this.client.messages.list({
      to: options.to,
      from: options.from,
      dateSentAfter: options.dateSentAfter,
      dateSentBefore: options.dateSentBefore,
      limit: options.limit || 50,
    });
  }

  /**
   * Delete a message
   */
  async deleteMessage(messageSid: string): Promise<boolean> {
    await this.client.messages(messageSid).remove();
    return true;
  }
}
```

### 4.2 SMS Response Handler (TwiML)

```typescript
// src/twilio/sms/sms-response.ts
import { MessagingResponse } from 'twilio';

export class TwilioSMSResponse {
  private response: MessagingResponse;

  constructor() {
    this.response = new MessagingResponse();
  }

  /**
   * Send text message response
   */
  message(body: string, options: {
    to?: string;
    from?: string;
    action?: string;
    method?: string;
  } = {}): this {
    const messageOptions: any = {};
    if (options.to) messageOptions.to = options.to;
    if (options.from) messageOptions.from = options.from;
    if (options.action) messageOptions.action = options.action;
    if (options.method) messageOptions.method = options.method;

    this.response.message(messageOptions, body);
    return this;
  }

  /**
   * Send MMS with media
   */
  mediaMessage(body: string, mediaUrls: string[]): this {
    const message = this.response.message();
    message.body(body);
    mediaUrls.forEach(url => message.media(url));
    return this;
  }

  /**
   * Redirect to another handler
   */
  redirect(url: string, method: string = 'POST'): this {
    this.response.redirect({ method }, url);
    return this;
  }

  /**
   * Get XML string
   */
  toString(): string {
    return this.response.toString();
  }
}
```

---

## 5. Webhook Handling System

### 5.1 Express Webhook Router

```typescript
// src/twilio/webhooks/webhook-router.ts
import express, { Router, Request, Response } from 'express';
import twilio from 'twilio';
import { TwilioVoiceResponse } from '../voice/voice-response';
import { TwilioSMSResponse } from '../sms/sms-response';

export interface WebhookConfig {
  authToken: string;
  validateRequests: boolean;
}

export class TwilioWebhookRouter {
  public router: Router;
  private config: WebhookConfig;

  constructor(config: WebhookConfig) {
    this.config = config;
    this.router = express.Router();
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    // Parse URL-encoded bodies
    this.router.use(express.urlencoded({ extended: false }));
    
    // Validate Twilio requests in production
    if (this.config.validateRequests) {
      this.router.use(twilio.webhook({ validate: true }));
    }
  }

  private setupRoutes(): void {
    // Voice webhooks
    this.router.post('/voice/incoming', this.handleIncomingCall.bind(this));
    this.router.post('/voice/status', this.handleCallStatus.bind(this));
    this.router.post('/voice/gather', this.handleGatherResult.bind(this));
    this.router.post('/voice/recording', this.handleRecording.bind(this));

    // SMS webhooks
    this.router.post('/sms/incoming', this.handleIncomingSMS.bind(this));
    this.router.post('/sms/status', this.handleSMSStatus.bind(this));

    // Media Stream WebSocket endpoint
    this.router.ws('/media-stream', this.handleMediaStream.bind(this));
  }

  /**
   * Handle incoming voice call
   */
  private async handleIncomingCall(req: Request, res: Response): Promise<void> {
    const { From, To, CallSid, Direction } = req.body;

    console.log(`[VOICE] Incoming call from ${From} to ${To}, SID: ${CallSid}`);

    // Create AI agent response
    const voiceResponse = new TwilioVoiceResponse();
    
    voiceResponse
      .say('Hello! This is OpenClaw AI Agent. How can I help you today?', {
        voice: 'Polly.Joanna-Neural',
        language: 'en-US'
      })
      .pause(1);

    // Option 1: Connect to Media Stream for real-time AI
    const host = req.headers.host;
    voiceResponse.connectToMediaStream(`wss://${host}/twilio/media-stream`);

    res.type('text/xml');
    res.send(voiceResponse.toString());
  }

  /**
   * Handle call status callbacks
   */
  private async handleCallStatus(req: Request, res: Response): Promise<void> {
    const { CallSid, CallStatus, Duration, RecordingUrl, From, To } = req.body;

    console.log(`[VOICE] Call ${CallSid} status: ${CallStatus}`);

    // Emit event for agent system
    this.emitCallStatusEvent({
      callSid: CallSid,
      status: CallStatus,
      duration: Duration,
      recordingUrl: RecordingUrl,
      from: From,
      to: To,
      timestamp: new Date().toISOString()
    });

    res.sendStatus(200);
  }

  /**
   * Handle gather input result
   */
  private async handleGatherResult(req: Request, res: Response): Promise<void> {
    const { Digits, SpeechResult, Confidence, CallSid } = req.body;

    console.log(`[VOICE] Gather result for ${CallSid}:`, { Digits, SpeechResult, Confidence });

    const voiceResponse = new TwilioVoiceResponse();

    if (Digits) {
      // Handle DTMF input
      voiceResponse.say(`You pressed ${Digits}`);
    } else if (SpeechResult) {
      // Handle speech input - send to AI
      voiceResponse.say(`You said: ${SpeechResult}`);
    } else {
      voiceResponse.say('I did not receive any input. Goodbye!');
      voiceResponse.hangup();
    }

    res.type('text/xml');
    res.send(voiceResponse.toString());
  }

  /**
   * Handle recording callback
   */
  private async handleRecording(req: Request, res: Response): Promise<void> {
    const { RecordingSid, RecordingUrl, RecordingDuration, CallSid } = req.body;

    console.log(`[VOICE] Recording completed for ${CallSid}:`, {
      recordingSid: RecordingSid,
      duration: RecordingDuration,
      url: RecordingUrl
    });

    // Trigger transcription
    this.processRecording(CallSid, RecordingSid, RecordingUrl);

    res.sendStatus(200);
  }

  /**
   * Handle incoming SMS
   */
  private async handleIncomingSMS(req: Request, res: Response): Promise<void> {
    const { From, To, Body, MessageSid, NumMedia, MediaUrl0 } = req.body;

    console.log(`[SMS] Message from ${From}: ${Body}`);

    // Generate AI response
    const smsResponse = new TwilioSMSResponse();
    
    // Simple echo for now - integrate with AI agent
    smsResponse.message(`OpenClaw AI received: "${Body}"`);

    res.type('text/xml');
    res.send(smsResponse.toString());
  }

  /**
   * Handle SMS status callback
   */
  private async handleSMSStatus(req: Request, res: Response): Promise<void> {
    const { MessageSid, MessageStatus, ErrorCode } = req.body;

    console.log(`[SMS] Message ${MessageSid} status: ${MessageStatus}`);

    this.emitSMSStatusEvent({
      messageSid: MessageSid,
      status: MessageStatus,
      errorCode: ErrorCode,
      timestamp: new Date().toISOString()
    });

    res.sendStatus(200);
  }

  /**
   * Handle Media Stream WebSocket
   */
  private async handleMediaStream(ws: any, req: Request): Promise<void> {
    console.log('[MEDIA STREAM] WebSocket connection established');

    // This will be implemented in the Media Streams section
    // for real-time bidirectional audio with AI

    ws.on('message', (message: string) => {
      const data = JSON.parse(message);
      
      switch (data.event) {
        case 'start':
          console.log('[MEDIA STREAM] Stream started:', data.start.streamSid);
          break;
        case 'media':
          // Handle audio data
          this.handleAudioData(data.media);
          break;
        case 'stop':
          console.log('[MEDIA STREAM] Stream stopped');
          break;
      }
    });

    ws.on('close', () => {
      console.log('[MEDIA STREAM] WebSocket closed');
    });
  }

  private handleAudioData(media: any): void {
    // Process audio payload (base64 encoded mu-law)
    const audioPayload = media.payload;
    // Forward to AI processing pipeline
  }

  private emitCallStatusEvent(event: any): void {
    // Emit to agent event bus
  }

  private emitSMSStatusEvent(event: any): void {
    // Emit to agent event bus
  }

  private async processRecording(callSid: string, recordingSid: string, recordingUrl: string): Promise<void> {
    // Trigger transcription service
  }
}
```

---

## 6. TwiML Call Flow Management

### 6.1 Dynamic Call Flow Engine

```typescript
// src/twilio/callflow/call-flow-engine.ts
import { TwilioVoiceResponse } from '../voice/voice-response';

export interface CallFlowStep {
  id: string;
  type: 'say' | 'gather' | 'record' | 'dial' | 'connect' | 'hangup' | 'redirect';
  options?: any;
  next?: string | ((input: any) => string);
}

export interface CallFlow {
  name: string;
  steps: Record<string, CallFlowStep>;
  startStep: string;
}

export class CallFlowEngine {
  private flows: Map<string, CallFlow> = new Map();
  private sessions: Map<string, { flowId: string; currentStep: string; data: any }> = new Map();

  /**
   * Register a call flow
   */
  registerFlow(flow: CallFlow): void {
    this.flows.set(flow.name, flow);
  }

  /**
   * Start a call flow for a session
   */
  startFlow(callSid: string, flowName: string, initialData: any = {}): string {
    const flow = this.flows.get(flowName);
    if (!flow) {
      throw new Error(`Flow ${flowName} not found`);
    }

    this.sessions.set(callSid, {
      flowId: flowName,
      currentStep: flow.startStep,
      data: initialData
    });

    return this.executeStep(callSid, flow.steps[flow.startStep]);
  }

  /**
   * Process input and advance flow
   */
  processInput(callSid: string, input: any): string {
    const session = this.sessions.get(callSid);
    if (!session) {
      return this.createErrorResponse('Session not found');
    }

    const flow = this.flows.get(session.flowId);
    if (!flow) {
      return this.createErrorResponse('Flow not found');
    }

    const currentStep = flow.steps[session.currentStep];
    
    // Determine next step
    let nextStepId: string;
    if (typeof currentStep.next === 'function') {
      nextStepId = currentStep.next(input);
    } else {
      nextStepId = currentStep.next || 'end';
    }

    if (nextStepId === 'end' || !flow.steps[nextStepId]) {
      this.sessions.delete(callSid);
      return this.createHangupResponse();
    }

    session.currentStep = nextStepId;
    return this.executeStep(callSid, flow.steps[nextStepId]);
  }

  /**
   * Execute a flow step
   */
  private executeStep(callSid: string, step: CallFlowStep): string {
    const response = new TwilioVoiceResponse();

    switch (step.type) {
      case 'say':
        response.say(step.options.message, {
          voice: step.options.voice,
          language: step.options.language
        });
        if (step.next) {
          response.redirect(`/twilio/voice/flow/${callSid}`, 'POST');
        }
        break;

      case 'gather':
        const gather = response.gather({
          action: `/twilio/voice/gather/${callSid}`,
          method: 'POST',
          timeout: step.options.timeout || 5,
          numDigits: step.options.numDigits,
          input: step.options.input || ['speech', 'dtmf'],
          speechTimeout: step.options.speechTimeout || 'auto',
          hints: step.options.hints
        });
        gather.say({ voice: step.options.voice }, step.options.prompt);
        break;

      case 'record':
        response.say(step.options.prompt);
        response.record({
          action: `/twilio/voice/recording/${callSid}`,
          maxLength: step.options.maxLength || 300,
          playBeep: step.options.playBeep !== false,
          recordingStatusCallback: `/twilio/voice/recording-status/${callSid}`
        });
        break;

      case 'dial':
        response.dial(step.options.number, {
          callerId: step.options.callerId,
          record: step.options.record,
          timeout: step.options.timeout
        });
        break;

      case 'connect':
        response.connectToMediaStream(step.options.streamUrl);
        break;

      case 'hangup':
        response.say(step.options.message || 'Goodbye!');
        response.hangup();
        this.sessions.delete(callSid);
        break;

      case 'redirect':
        response.redirect(step.options.url, step.options.method || 'POST');
        break;
    }

    return response.toString();
  }

  private createErrorResponse(message: string): string {
    const response = new TwilioVoiceResponse();
    response.say(`Error: ${message}`);
    response.hangup();
    return response.toString();
  }

  private createHangupResponse(): string {
    const response = new TwilioVoiceResponse();
    response.say('Thank you for calling. Goodbye!');
    response.hangup();
    return response.toString();
  }
}
```

### 6.2 Predefined Agentic Call Flows

```typescript
// src/twilio/callflow/agent-flows.ts
import { CallFlow } from './call-flow-engine';

export const createAgentGreetingFlow = (agentName: string): CallFlow => ({
  name: 'agent-greeting',
  startStep: 'greeting',
  steps: {
    greeting: {
      id: 'greeting',
      type: 'say',
      options: {
        message: `Hello! This is ${agentName}, your AI assistant. How may I help you today?`,
        voice: 'Polly.Joanna-Neural',
        language: 'en-US'
      },
      next: 'connect_ai'
    },
    connect_ai: {
      id: 'connect_ai',
      type: 'connect',
      options: {
        streamUrl: 'wss://your-server.com/twilio/media-stream'
      }
    }
  }
});

export const createMenuFlow = (): CallFlow => ({
  name: 'ivr-menu',
  startStep: 'menu',
  steps: {
    menu: {
      id: 'menu',
      type: 'gather',
      options: {
        prompt: 'Press 1 for sales, 2 for support, 3 to speak with an AI agent, or 0 to hang up.',
        numDigits: 1,
        timeout: 10,
        input: ['dtmf']
      },
      next: (digits: string) => {
        switch (digits) {
          case '1': return 'sales';
          case '2': return 'support';
          case '3': return 'connect_ai';
          case '0': return 'goodbye';
          default: return 'invalid';
        }
      }
    },
    sales: {
      id: 'sales',
      type: 'say',
      options: {
        message: 'Connecting you to sales...',
        voice: 'Polly.Joanna-Neural'
      },
      next: 'dial_sales'
    },
    dial_sales: {
      id: 'dial_sales',
      type: 'dial',
      options: {
        number: '+1-800-SALES-01',
        timeout: 30
      }
    },
    support: {
      id: 'support',
      type: 'say',
      options: {
        message: 'Connecting you to support...',
        voice: 'Polly.Joanna-Neural'
      },
      next: 'dial_support'
    },
    dial_support: {
      id: 'dial_support',
      type: 'dial',
      options: {
        number: '+1-800-SUPPORT-1',
        timeout: 30
      }
    },
    connect_ai: {
      id: 'connect_ai',
      type: 'connect',
      options: {
        streamUrl: 'wss://your-server.com/twilio/media-stream'
      }
    },
    invalid: {
      id: 'invalid',
      type: 'say',
      options: {
        message: 'Invalid selection. Please try again.',
        voice: 'Polly.Joanna-Neural'
      },
      next: 'menu'
    },
    goodbye: {
      id: 'goodbye',
      type: 'hangup',
      options: {
        message: 'Thank you for calling. Goodbye!'
      }
    }
  }
});
```

---

## 7. Phone Number Management

### 7.1 Phone Number Service

```typescript
// src/twilio/phone/phone-number-service.ts
import twilio from 'twilio';

export interface PhoneNumberCapabilities {
  voice: boolean;
  sms: boolean;
  mms: boolean;
  fax: boolean;
}

export interface AvailableNumber {
  phoneNumber: string;
  friendlyName: string;
  locality?: string;
  region?: string;
  postalCode?: string;
  capabilities: PhoneNumberCapabilities;
  addressRequirements?: string;
  beta?: boolean;
}

export interface PurchasedNumber {
  sid: string;
  phoneNumber: string;
  friendlyName: string;
  capabilities: PhoneNumberCapabilities;
  voiceUrl?: string;
  voiceMethod?: string;
  smsUrl?: string;
  smsMethod?: string;
  statusCallback?: string;
  statusCallbackMethod?: string;
  dateCreated: Date;
}

export class PhoneNumberService {
  private client: twilio.Twilio;

  constructor(accountSid: string, authToken: string) {
    this.client = twilio(accountSid, authToken);
  }

  /**
   * Search for available phone numbers
   */
  async searchAvailableNumbers(options: {
    country?: string;
    type?: 'local' | 'tollfree' | 'mobile';
    areaCode?: string;
    contains?: string;
    smsEnabled?: boolean;
    voiceEnabled?: boolean;
    mmsEnabled?: boolean;
    limit?: number;
  } = {}): Promise<AvailableNumber[]> {
    const searchOptions: any = {
      limit: options.limit || 20
    };

    if (options.smsEnabled) searchOptions.smsEnabled = true;
    if (options.voiceEnabled) searchOptions.voiceEnabled = true;
    if (options.mmsEnabled) searchOptions.mmsEnabled = true;
    if (options.contains) searchOptions.contains = options.contains;

    const country = options.country || 'US';
    let results: any[] = [];

    try {
      if (options.type === 'tollfree') {
        results = await this.client.availablePhoneNumbers(country)
          .tollFree
          .list(searchOptions);
      } else {
        if (options.areaCode) {
          searchOptions.areaCode = options.areaCode;
        }
        results = await this.client.availablePhoneNumbers(country)
          .local
          .list(searchOptions);
      }
    } catch (error) {
      console.error('Error searching phone numbers:', error);
      throw error;
    }

    return results.map(num => ({
      phoneNumber: num.phoneNumber,
      friendlyName: num.friendlyName,
      locality: num.locality,
      region: num.region,
      postalCode: num.postalCode,
      capabilities: {
        voice: num.capabilities?.voice || false,
        sms: num.capabilities?.sms || false,
        mms: num.capabilities?.mms || false,
        fax: num.capabilities?.fax || false
      },
      addressRequirements: num.addressRequirements,
      beta: num.beta
    }));
  }

  /**
   * Purchase a phone number
   */
  async purchaseNumber(phoneNumber: string, options: {
    friendlyName?: string;
    voiceUrl?: string;
    voiceMethod?: string;
    smsUrl?: string;
    smsMethod?: string;
    statusCallback?: string;
  } = {}): Promise<PurchasedNumber> {
    try {
      const incomingPhoneNumber = await this.client.incomingPhoneNumbers.create({
        phoneNumber,
        friendlyName: options.friendlyName || `OpenClaw-${Date.now()}`,
        voiceUrl: options.voiceUrl,
        voiceMethod: options.voiceMethod || 'POST',
        smsUrl: options.smsUrl,
        smsMethod: options.smsMethod || 'POST',
        statusCallback: options.statusCallback,
        statusCallbackMethod: 'POST',
        voiceReceiveMode: 'voice'
      });

      return {
        sid: incomingPhoneNumber.sid,
        phoneNumber: incomingPhoneNumber.phoneNumber,
        friendlyName: incomingPhoneNumber.friendlyName,
        capabilities: {
          voice: incomingPhoneNumber.capabilities?.voice || false,
          sms: incomingPhoneNumber.capabilities?.sms || false,
          mms: incomingPhoneNumber.capabilities?.mms || false,
          fax: incomingPhoneNumber.capabilities?.fax || false
        },
        voiceUrl: incomingPhoneNumber.voiceUrl,
        voiceMethod: incomingPhoneNumber.voiceMethod,
        smsUrl: incomingPhoneNumber.smsUrl,
        smsMethod: incomingPhoneNumber.smsMethod,
        statusCallback: incomingPhoneNumber.statusCallback,
        dateCreated: incomingPhoneNumber.dateCreated
      };
    } catch (error) {
      console.error('Error purchasing phone number:', error);
      throw error;
    }
  }

  /**
   * Configure phone number webhooks
   */
  async configureNumber(phoneNumberSid: string, config: {
    voiceUrl?: string;
    voiceMethod?: string;
    smsUrl?: string;
    smsMethod?: string;
    statusCallback?: string;
    friendlyName?: string;
  }): Promise<PurchasedNumber> {
    try {
      const updateData: any = {};
      
      if (config.voiceUrl !== undefined) updateData.voiceUrl = config.voiceUrl;
      if (config.voiceMethod !== undefined) updateData.voiceMethod = config.voiceMethod;
      if (config.smsUrl !== undefined) updateData.smsUrl = config.smsUrl;
      if (config.smsMethod !== undefined) updateData.smsMethod = config.smsMethod;
      if (config.statusCallback !== undefined) updateData.statusCallback = config.statusCallback;
      if (config.friendlyName !== undefined) updateData.friendlyName = config.friendlyName;

      const updated = await this.client.incomingPhoneNumbers(phoneNumberSid).update(updateData);

      return {
        sid: updated.sid,
        phoneNumber: updated.phoneNumber,
        friendlyName: updated.friendlyName,
        capabilities: {
          voice: updated.capabilities?.voice || false,
          sms: updated.capabilities?.sms || false,
          mms: updated.capabilities?.mms || false,
          fax: updated.capabilities?.fax || false
        },
        voiceUrl: updated.voiceUrl,
        voiceMethod: updated.voiceMethod,
        smsUrl: updated.smsUrl,
        smsMethod: updated.smsMethod,
        statusCallback: updated.statusCallback,
        dateCreated: updated.dateCreated
      };
    } catch (error) {
      console.error('Error configuring phone number:', error);
      throw error;
    }
  }

  /**
   * List purchased phone numbers
   */
  async listNumbers(options: {
    phoneNumber?: string;
    friendlyName?: string;
    limit?: number;
  } = {}): Promise<PurchasedNumber[]> {
    try {
      const numbers = await this.client.incomingPhoneNumbers.list({
        phoneNumber: options.phoneNumber,
        friendlyName: options.friendlyName,
        limit: options.limit || 50
      });

      return numbers.map(num => ({
        sid: num.sid,
        phoneNumber: num.phoneNumber,
        friendlyName: num.friendlyName,
        capabilities: {
          voice: num.capabilities?.voice || false,
          sms: num.capabilities?.sms || false,
          mms: num.capabilities?.mms || false,
          fax: num.capabilities?.fax || false
        },
        voiceUrl: num.voiceUrl,
        voiceMethod: num.voiceMethod,
        smsUrl: num.smsUrl,
        smsMethod: num.smsMethod,
        statusCallback: num.statusCallback,
        dateCreated: num.dateCreated
      }));
    } catch (error) {
      console.error('Error listing phone numbers:', error);
      throw error;
    }
  }

  /**
   * Release a phone number
   */
  async releaseNumber(phoneNumberSid: string): Promise<boolean> {
    try {
      await this.client.incomingPhoneNumbers(phoneNumberSid).remove();
      return true;
    } catch (error) {
      console.error('Error releasing phone number:', error);
      throw error;
    }
  }

  /**
   * Get number details
   */
  async getNumber(phoneNumberSid: string): Promise<PurchasedNumber> {
    try {
      const num = await this.client.incomingPhoneNumbers(phoneNumberSid).fetch();

      return {
        sid: num.sid,
        phoneNumber: num.phoneNumber,
        friendlyName: num.friendlyName,
        capabilities: {
          voice: num.capabilities?.voice || false,
          sms: num.capabilities?.sms || false,
          mms: num.capabilities?.mms || false,
          fax: num.capabilities?.fax || false
        },
        voiceUrl: num.voiceUrl,
        voiceMethod: num.voiceMethod,
        smsUrl: num.smsUrl,
        smsMethod: num.smsMethod,
        statusCallback: num.statusCallback,
        dateCreated: num.dateCreated
      };
    } catch (error) {
      console.error('Error fetching phone number:', error);
      throw error;
    }
  }
}
```

---

## 8. Call Recording & Transcription

### 8.1 Recording Service

```typescript
// src/twilio/recording/recording-service.ts
import twilio from 'twilio';
import { OpenAI } from 'openai';

export interface Recording {
  sid: string;
  callSid: string;
  duration: number;
  url: string;
  status: string;
  dateCreated: Date;
}

export interface TranscriptionResult {
  recordingSid: string;
  text: string;
  confidence: number;
  language: string;
  segments: Array<{
    start: number;
    end: number;
    text: string;
    confidence: number;
  }>;
}

export class RecordingService {
  private client: twilio.Twilio;
  private openai: OpenAI;

  constructor(accountSid: string, authToken: string, openaiApiKey: string) {
    this.client = twilio(accountSid, authToken);
    this.openai = new OpenAI({ apiKey: openaiApiKey });
  }

  /**
   * Get recording details
   */
  async getRecording(recordingSid: string): Promise<Recording> {
    const recording = await this.client.recordings(recordingSid).fetch();
    
    return {
      sid: recording.sid,
      callSid: recording.callSid,
      duration: parseInt(recording.duration),
      url: `https://api.twilio.com${recording.uri.replace('.json', '.mp3')}`,
      status: recording.status,
      dateCreated: recording.dateCreated
    };
  }

  /**
   * List recordings for a call
   */
  async listRecordings(callSid: string): Promise<Recording[]> {
    const recordings = await this.client.calls(callSid).recordings.list();
    
    return recordings.map(rec => ({
      sid: rec.sid,
      callSid: rec.callSid,
      duration: parseInt(rec.duration),
      url: `https://api.twilio.com${rec.uri.replace('.json', '.mp3')}`,
      status: rec.status,
      dateCreated: rec.dateCreated
    }));
  }

  /**
   * Delete a recording
   */
  async deleteRecording(recordingSid: string): Promise<boolean> {
    await this.client.recordings(recordingSid).remove();
    return true;
  }

  /**
   * Transcribe recording using OpenAI Whisper
   */
  async transcribeRecording(recordingSid: string): Promise<TranscriptionResult> {
    try {
      // Get recording URL
      const recording = await this.getRecording(recordingSid);
      
      // Download audio file
      const audioResponse = await fetch(recording.url, {
        headers: {
          'Authorization': 'Basic ' + Buffer.from(
            `${this.client.username}:${this.client.password}`
          ).toString('base64')
        }
      });

      if (!audioResponse.ok) {
        throw new Error(`Failed to download recording: ${audioResponse.statusText}`);
      }

      const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());

      // Transcribe with Whisper
      const transcription = await this.openai.audio.transcriptions.create({
        file: new File([audioBuffer], 'recording.mp3', { type: 'audio/mp3' }),
        model: 'whisper-1',
        response_format: 'verbose_json',
        timestamp_granularities: ['segment']
      });

      return {
        recordingSid,
        text: transcription.text,
        confidence: 1.0, // Whisper doesn't provide confidence scores
        language: transcription.language,
        segments: (transcription as any).segments?.map((seg: any) => ({
          start: seg.start,
          end: seg.end,
          text: seg.text,
          confidence: seg.avg_logprob || 1.0
        })) || []
      };
    } catch (error) {
      console.error('Transcription error:', error);
      throw error;
    }
  }

  /**
   * Transcribe with Twilio's native transcription (add-on)
   */
  async enableTranscription(callSid: string, callbackUrl: string): Promise<void> {
    // Use Twilio's transcription add-on or Intelligence Service
    // This requires setting up a Twilio Intelligence Service
  }

  /**
   * Process recording with AI analysis
   */
  async analyzeRecording(recordingSid: string): Promise<{
    transcription: TranscriptionResult;
    summary: string;
    sentiment: 'positive' | 'neutral' | 'negative';
    actionItems: string[];
  }> {
    // Get transcription
    const transcription = await this.transcribeRecording(recordingSid);

    // Analyze with GPT
    const analysis = await this.openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `Analyze this call transcription and provide:
1. A brief summary (2-3 sentences)
2. Sentiment (positive/neutral/negative)
3. Action items (if any)

Respond in JSON format: { "summary": "...", "sentiment": "...", "actionItems": [...] }`
        },
        {
          role: 'user',
          content: transcription.text
        }
      ],
      response_format: { type: 'json_object' }
    });

    const result = JSON.parse(analysis.choices[0].message.content || '{}');

    return {
      transcription,
      summary: result.summary,
      sentiment: result.sentiment,
      actionItems: result.actionItems || []
    };
  }
}
```

---

## 9. TTS/STT Integration

### 9.1 Text-to-Speech Service

```typescript
// src/twilio/tts/tts-service.ts
export interface TTSOptions {
  text: string;
  voice?: string;
  language?: string;
  ssml?: boolean;
  speed?: number;
  pitch?: number;
}

export interface TTSVoice {
  id: string;
  name: string;
  provider: 'Polly' | 'Google' | 'Azure';
  language: string;
  gender: 'male' | 'female' | 'neutral';
  neural: boolean;
}

export class TTSService {
  // Available Twilio TTS voices
  static readonly VOICES: TTSVoice[] = [
    // Amazon Polly Voices
    { id: 'Polly.Joanna', name: 'Joanna', provider: 'Polly', language: 'en-US', gender: 'female', neural: false },
    { id: 'Polly.Joanna-Neural', name: 'Joanna (Neural)', provider: 'Polly', language: 'en-US', gender: 'female', neural: true },
    { id: 'Polly.Matthew', name: 'Matthew', provider: 'Polly', language: 'en-US', gender: 'male', neural: false },
    { id: 'Polly.Matthew-Neural', name: 'Matthew (Neural)', provider: 'Polly', language: 'en-US', gender: 'male', neural: true },
    { id: 'Polly.Salli', name: 'Salli', provider: 'Polly', language: 'en-US', gender: 'female', neural: false },
    { id: 'Polly.Ivy', name: 'Ivy', provider: 'Polly', language: 'en-US', gender: 'female', neural: false },
    { id: 'Polly.Kendra', name: 'Kendra', provider: 'Polly', language: 'en-US', gender: 'female', neural: true },
    { id: 'Polly.Kimberly', name: 'Kimberly', provider: 'Polly', language: 'en-US', gender: 'female', neural: false },
    { id: 'Polly.Justin', name: 'Justin', provider: 'Polly', language: 'en-US', gender: 'male', neural: false },
    { id: 'Polly.Joey', name: 'Joey', provider: 'Polly', language: 'en-US', gender: 'male', neural: false },
    
    // Google Chirp3-HD Voices
    { id: 'Google.en-US-Chirp3-HD-Aoede', name: 'Aoede (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'female', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Charon', name: 'Charon (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'male', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Fenrir', name: 'Fenrir (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'male', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Kore', name: 'Kore (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'female', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Leda', name: 'Leda (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'female', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Orus', name: 'Orus (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'male', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Puck', name: 'Puck (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'neutral', neural: true },
    { id: 'Google.en-US-Chirp3-HD-Zeph', name: 'Zeph (Chirp3 HD)', provider: 'Google', language: 'en-US', gender: 'male', neural: true },
    
    // Amazon Polly Generative Voices (Beta)
    { id: 'Polly.Joanna-Generative', name: 'Joanna (Generative)', provider: 'Polly', language: 'en-US', gender: 'female', neural: true },
    { id: 'Polly.Matthew-Generative', name: 'Matthew (Generative)', provider: 'Polly', language: 'en-US', gender: 'male', neural: true },
  ];

  /**
   * Get voice by ID
   */
  static getVoice(voiceId: string): TTSVoice | undefined {
    return this.VOICES.find(v => v.id === voiceId);
  }

  /**
   * Get voices by language
   */
  static getVoicesByLanguage(language: string): TTSVoice[] {
    return this.VOICES.filter(v => v.language === language);
  }

  /**
   * Get recommended voice for agent
   */
  static getAgentVoice(agentGender: 'male' | 'female' = 'female', neural: boolean = true): string {
    if (neural) {
      return agentGender === 'female' 
        ? 'Polly.Joanna-Neural' 
        : 'Polly.Matthew-Neural';
    }
    return agentGender === 'female' ? 'Polly.Joanna' : 'Polly.Matthew';
  }

  /**
   * Generate SSML for advanced TTS control
   */
  static generateSSML(options: {
    text: string;
    break?: { strength?: string; time?: string };
    emphasis?: { level: 'strong' | 'moderate' | 'reduced' };
    prosody?: {
      rate?: string;
      pitch?: string;
      volume?: string;
    };
    sayAs?: {
      interpretAs: string;
      format?: string;
    };
    phoneme?: {
      alphabet: string;
      ph: string;
    };
  }): string {
    let ssml = options.text;

    if (options.phoneme) {
      ssml = `<phoneme alphabet="${options.phoneme.alphabet}" ph="${options.phoneme.ph}">${ssml}</phoneme>`;
    }

    if (options.sayAs) {
      const formatAttr = options.sayAs.format ? ` format="${options.sayAs.format}"` : '';
      ssml = `<say-as interpret-as="${options.sayAs.interpretAs}"${formatAttr}>${ssml}</say-as>`;
    }

    if (options.prosody) {
      const rate = options.prosody.rate ? ` rate="${options.prosody.rate}"` : '';
      const pitch = options.prosody.pitch ? ` pitch="${options.prosody.pitch}"` : '';
      const volume = options.prosody.volume ? ` volume="${options.prosody.volume}"` : '';
      ssml = `<prosody${rate}${pitch}${volume}>${ssml}</prosody>`;
    }

    if (options.emphasis) {
      ssml = `<emphasis level="${options.emphasis.level}">${ssml}</emphasis>`;
    }

    if (options.break) {
      const strength = options.break.strength ? ` strength="${options.break.strength}"` : '';
      const time = options.break.time ? ` time="${options.break.time}"` : '';
      ssml += `<break${strength}${time}/>`;
    }

    return ssml;
  }

  /**
   * Format text for better TTS pronunciation
   */
  static formatForTTS(text: string): string {
    return text
      // Add pauses after punctuation
      .replace(/([.!?])(\s+)/g, '$1, ')
      // Spell out common abbreviations
      .replace(/\bAI\b/g, 'A.I.')
      .replace(/\bAPI\b/g, 'A.P.I.')
      .replace(/\bSMS\b/g, 'S.M.S.')
      .replace(/\bURL\b/g, 'U.R.L.')
      // Format numbers for better pronunciation
      .replace(/\b(\d{3})-(\d{3})-(\d{4})\b/g, '$1, $2, $4')
      // Add emphasis to important words
      .replace(/\b(important|urgent|critical|warning)\b/gi, '<emphasis level="moderate">$1</emphasis>');
  }
}
```

### 9.2 Speech-to-Text Service

```typescript
// src/twilio/stt/stt-service.ts
import { OpenAI } from 'openai';

export interface STTOptions {
  audio: Buffer | File;
  model?: string;
  language?: string;
  prompt?: string;
  responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
  temperature?: number;
}

export interface STTResult {
  text: string;
  language?: string;
  duration?: number;
  segments?: Array<{
    id: number;
    start: number;
    end: number;
    text: string;
    confidence: number;
  }>;
  words?: Array<{
    word: string;
    start: number;
    end: number;
    confidence: number;
  }>;
}

export class STTService {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Transcribe audio using Whisper
   */
  async transcribe(options: STTOptions): Promise<STTResult> {
    const transcription = await this.openai.audio.transcriptions.create({
      file: options.audio instanceof Buffer 
        ? new File([options.audio], 'audio.mp3', { type: 'audio/mp3' })
        : options.audio,
      model: options.model || 'whisper-1',
      language: options.language,
      prompt: options.prompt,
      response_format: options.responseFormat || 'verbose_json',
      temperature: options.temperature || 0
    });

    if (options.responseFormat === 'verbose_json') {
      const verbose = transcription as any;
      return {
        text: verbose.text,
        language: verbose.language,
        duration: verbose.duration,
        segments: verbose.segments?.map((seg: any) => ({
          id: seg.id,
          start: seg.start,
          end: seg.end,
          text: seg.text.trim(),
          confidence: Math.exp(seg.avg_logprob) || 0.95
        })),
        words: verbose.words?.map((word: any) => ({
          word: word.word,
          start: word.start,
          end: word.end,
          confidence: word.probability || 0.95
        }))
      };
    }

    return {
      text: transcription.text
    };
  }

  /**
   * Transcribe with real-time streaming (using WebSocket)
   */
  async transcribeStream(
    audioStream: AsyncIterable<Buffer>,
    onResult: (result: STTResult) => void,
    options: Partial<STTOptions> = {}
  ): Promise<void> {
    // For real-time streaming, use Deepgram or AssemblyAI
    // This is a placeholder for streaming implementation
  }

  /**
   * Transcribe Twilio audio format (mu-law)
   */
  async transcribeTwilioAudio(
    mulawAudio: Buffer,
    options: Partial<STTOptions> = {}
  ): Promise<STTResult> {
    // Convert mu-law to PCM/WAV format
    const wavAudio = this.convertMulawToWav(mulawAudio);
    
    return this.transcribe({
      audio: wavAudio,
      ...options
    });
  }

  /**
   * Convert mu-law to WAV format
   */
  private convertMulawToWav(mulawData: Buffer): Buffer {
    // Mu-law decoding table
    const MULAW_DECODE_TABLE = new Int16Array(256);
    for (let i = 0; i < 256; i++) {
      const sign = (i & 0x80) ? -1 : 1;
      const exponent = (i & 0x70) >> 4;
      const mantissa = i & 0x0f;
      const value = ((mantissa << 1) + 33) << exponent;
      MULAW_DECODE_TABLE[i] = sign * (value - 33);
    }

    // Decode mu-law to PCM
    const pcmData = Buffer.alloc(mulawData.length * 2);
    for (let i = 0; i < mulawData.length; i++) {
      const sample = MULAW_DECODE_TABLE[mulawData[i]];
      pcmData.writeInt16LE(sample, i * 2);
    }

    // Create WAV header
    const wavHeader = Buffer.alloc(44);
    wavHeader.write('RIFF', 0);
    wavHeader.writeUInt32LE(36 + pcmData.length, 4);
    wavHeader.write('WAVE', 8);
    wavHeader.write('fmt ', 12);
    wavHeader.writeUInt32LE(16, 16);
    wavHeader.writeUInt16LE(1, 20); // PCM format
    wavHeader.writeUInt16LE(1, 22); // Mono
    wavHeader.writeUInt32LE(8000, 24); // Sample rate
    wavHeader.writeUInt32LE(16000, 28); // Byte rate
    wavHeader.writeUInt16LE(2, 32); // Block align
    wavHeader.writeUInt16LE(16, 34); // Bits per sample
    wavHeader.write('data', 36);
    wavHeader.writeUInt32LE(pcmData.length, 40);

    return Buffer.concat([wavHeader, pcmData]);
  }
}
```

---

## 10. Media Streams for Real-Time AI

### 10.1 Media Stream Server (Node.js with Fastify)

```typescript
// src/twilio/media-stream/media-stream-server.ts
import Fastify from 'fastify';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import WebSocket from 'ws';
import { VoiceResponse } from 'twilio';

export interface MediaStreamConfig {
  port: number;
  openaiApiKey: string;
  twilioAccountSid: string;
  twilioAuthToken: string;
  systemMessage?: string;
  voice?: string;
  temperature?: number;
}

interface MediaMessage {
  event: 'start' | 'media' | 'stop' | 'mark';
  streamSid?: string;
  media?: {
    track: 'inbound' | 'outbound';
    chunk: string;
    timestamp: string;
    payload: string;
  };
  start?: {
    streamSid: string;
    accountSid: string;
    callSid: string;
    tracks: string[];
    mediaFormat: {
      encoding: string;
      sampleRate: number;
      channels: number;
    };
  };
  mark?: {
    name: string;
  };
}

export class MediaStreamServer {
  private fastify: Fastify.FastifyInstance;
  private config: MediaStreamConfig;
  private openaiWs: WebSocket | null = null;

  constructor(config: MediaStreamConfig) {
    this.config = {
      systemMessage: 'You are a helpful AI assistant. Keep responses brief and conversational.',
      voice: 'alloy',
      temperature: 0.8,
      ...config
    };

    this.fastify = Fastify({ logger: true });
    this.setupPlugins();
    this.setupRoutes();
  }

  private async setupPlugins(): Promise<void> {
    await this.fastify.register(fastifyFormBody);
    await this.fastify.register(fastifyWs);
  }

  private setupRoutes(): void {
    // Health check
    this.fastify.get('/', async () => ({ 
      status: 'ok', 
      service: 'OpenClaw Media Stream Server' 
    }));

    // Incoming call webhook
    this.fastify.all('/incoming-call', async (request, reply) => {
      const twiml = new VoiceResponse();
      
      // Welcome message
      twiml.say(
        { voice: 'Polly.Joanna-Neural' },
        'Hello! This is OpenClaw AI. How can I help you today?'
      );
      twiml.pause({ length: 1 });

      // Connect to media stream
      const connect = twiml.connect();
      const host = request.headers.host;
      connect.stream({ url: `wss://${host}/media-stream` });

      reply.type('text/xml');
      reply.send(twiml.toString());
    });

    // WebSocket media stream endpoint
    this.fastify.register(async (fastify) => {
      fastify.get('/media-stream', { websocket: true }, (connection, req) => {
        console.log('[MediaStream] Client connected');
        
        let streamSid: string | null = null;

        // Connect to OpenAI Realtime API
        this.connectToOpenAI(connection);

        connection.socket.on('message', (message: WebSocket.Data) => {
          try {
            const data: MediaMessage = JSON.parse(message.toString());

            switch (data.event) {
              case 'start':
                streamSid = data.start?.streamSid || null;
                console.log('[MediaStream] Stream started:', streamSid);
                break;

              case 'media':
                if (data.media && this.openaiWs?.readyState === WebSocket.OPEN) {
                  // Forward audio to OpenAI
                  const audioAppend = {
                    type: 'input_audio_buffer.append',
                    audio: data.media.payload
                  };
                  this.openaiWs.send(JSON.stringify(audioAppend));
                }
                break;

              case 'stop':
                console.log('[MediaStream] Stream stopped');
                this.openaiWs?.close();
                break;

              case 'mark':
                console.log('[MediaStream] Mark received:', data.mark?.name);
                break;
            }
          } catch (error) {
            console.error('[MediaStream] Error processing message:', error);
          }
        });

        connection.socket.on('close', () => {
          console.log('[MediaStream] Client disconnected');
          this.openaiWs?.close();
        });

        connection.socket.on('error', (error) => {
          console.error('[MediaStream] WebSocket error:', error);
        });
      });
    });
  }

  private connectToOpenAI(twilioConnection: any): void {
    // Connect to OpenAI Realtime API
    this.openaiWs = new WebSocket(
      `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01`,
      {
        headers: {
          'Authorization': `Bearer ${this.config.openaiApiKey}`,
          'OpenAI-Beta': 'realtime=v1'
        }
      }
    );

    this.openaiWs.on('open', () => {
      console.log('[OpenAI] Connected to Realtime API');

      // Send session configuration
      const sessionConfig = {
        type: 'session.update',
        session: {
          modalities: ['text', 'audio'],
          instructions: this.config.systemMessage,
          voice: this.config.voice,
          input_audio_format: 'g711_ulaw',
          output_audio_format: 'g711_ulaw',
          input_audio_transcription: {
            model: 'whisper-1'
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500
          },
          temperature: this.config.temperature
        }
      };

      this.openaiWs?.send(JSON.stringify(sessionConfig));
    });

    this.openaiWs.on('message', (data: WebSocket.Data) => {
      try {
        const event = JSON.parse(data.toString());
        console.log('[OpenAI] Event:', event.type);

        switch (event.type) {
          case 'session.updated':
            console.log('[OpenAI] Session updated');
            break;

          case 'input_audio_buffer.speech_started':
            console.log('[OpenAI] User started speaking');
            break;

          case 'input_audio_buffer.speech_stopped':
            console.log('[OpenAI] User stopped speaking');
            break;

          case 'conversation.item.input_audio_transcription.completed':
            console.log('[OpenAI] Transcription:', event.transcript);
            break;

          case 'response.audio.delta':
            // Send audio back to Twilio
            if (event.delta) {
              const audioDelta = {
                event: 'media',
                streamSid: twilioConnection.socket.streamSid,
                media: {
                  payload: event.delta
                }
              };
              twilioConnection.socket.send(JSON.stringify(audioDelta));
            }
            break;

          case 'response.audio_transcript.done':
            console.log('[OpenAI] AI response:', event.transcript);
            break;

          case 'error':
            console.error('[OpenAI] Error:', event.error);
            break;
        }
      } catch (error) {
        console.error('[OpenAI] Error processing message:', error);
      }
    });

    this.openaiWs.on('close', () => {
      console.log('[OpenAI] Disconnected');
    });

    this.openaiWs.on('error', (error) => {
      console.error('[OpenAI] WebSocket error:', error);
    });
  }

  async start(): Promise<void> {
    try {
      await this.fastify.listen({ port: this.config.port, host: '0.0.0.0' });
      console.log(`[Server] Media Stream Server running on port ${this.config.port}`);
    } catch (error) {
      console.error('[Server] Failed to start:', error);
      throw error;
    }
  }

  async stop(): Promise<void> {
    await this.fastify.close();
    this.openaiWs?.close();
  }
}
```

---

## 11. Security & Authentication

### 11.1 Webhook Validation

```typescript
// src/twilio/security/webhook-validator.ts
import twilio from 'twilio';
import { Request, Response, NextFunction } from 'express';

export class WebhookValidator {
  private authToken: string;

  constructor(authToken: string) {
    this.authToken = authToken;
  }

  /**
   * Express middleware to validate Twilio requests
   */
  middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      // Skip validation in test environment
      if (process.env.NODE_ENV === 'test') {
        return next();
      }

      const signature = req.headers['x-twilio-signature'] as string;
      const url = `${req.protocol}://${req.get('host')}${req.originalUrl}`;

      const isValid = twilio.validateRequest(
        this.authToken,
        signature,
        url,
        req.body
      );

      if (!isValid) {
        console.warn('[Security] Invalid Twilio signature');
        return res.status(403).json({ error: 'Invalid signature' });
      }

      next();
    };
  }

  /**
   * Validate request manually
   */
  validateRequest(
    signature: string,
    url: string,
    params: Record<string, any>
  ): boolean {
    return twilio.validateRequest(
      this.authToken,
      signature,
      url,
      params
    );
  }
}
```

### 11.2 Environment Configuration

```typescript
// src/twilio/config/twilio-config.ts
import dotenv from 'dotenv';

dotenv.config();

export interface TwilioEnvironmentConfig {
  // Twilio credentials
  accountSid: string;
  authToken: string;
  apiKey?: string;
  apiSecret?: string;

  // Phone numbers
  defaultPhoneNumber: string;
  
  // Webhook settings
  webhookBaseUrl: string;
  validateWebhooks: boolean;

  // Recording settings
  recordCalls: boolean;
  recordingFormat: 'mp3' | 'wav';

  // TTS settings
  defaultVoice: string;
  defaultLanguage: string;

  // OpenAI settings (for STT/AI)
  openaiApiKey: string;

  // Server settings
  port: number;
  nodeEnv: string;
}

export const twilioConfig: TwilioEnvironmentConfig = {
  accountSid: process.env.TWILIO_ACCOUNT_SID || '',
  authToken: process.env.TWILIO_AUTH_TOKEN || '',
  apiKey: process.env.TWILIO_API_KEY,
  apiSecret: process.env.TWILIO_API_SECRET,
  
  defaultPhoneNumber: process.env.TWILIO_PHONE_NUMBER || '',
  
  webhookBaseUrl: process.env.WEBHOOK_BASE_URL || '',
  validateWebhooks: process.env.VALIDATE_WEBHOOKS === 'true',

  recordCalls: process.env.RECORD_CALLS === 'true',
  recordingFormat: (process.env.RECORDING_FORMAT as 'mp3' | 'wav') || 'mp3',

  defaultVoice: process.env.DEFAULT_VOICE || 'Polly.Joanna-Neural',
  defaultLanguage: process.env.DEFAULT_LANGUAGE || 'en-US',

  openaiApiKey: process.env.OPENAI_API_KEY || '',

  port: parseInt(process.env.PORT || '3000'),
  nodeEnv: process.env.NODE_ENV || 'development'
};

// Validate required configuration
export function validateConfig(): void {
  const required = [
    'TWILIO_ACCOUNT_SID',
    'TWILIO_AUTH_TOKEN',
    'TWILIO_PHONE_NUMBER',
    'OPENAI_API_KEY'
  ];

  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
}
```

---

## 12. Implementation Code Examples

### 12.1 Complete Integration Setup

```typescript
// src/twilio/index.ts - Main Integration Module
import { TwilioVoiceClient } from './voice/voice-client';
import { TwilioSMSClient } from './sms/sms-client';
import { PhoneNumberService } from './phone/phone-number-service';
import { RecordingService } from './recording/recording-service';
import { TTSService } from './tts/tts-service';
import { STTService } from './stt/stt-service';
import { CallFlowEngine } from './callflow/call-flow-engine';
import { MediaStreamServer } from './media-stream/media-stream-server';
import { twilioConfig, validateConfig } from './config/twilio-config';

export class TwilioIntegration {
  public voice: TwilioVoiceClient;
  public sms: TwilioSMSClient;
  public phoneNumbers: PhoneNumberService;
  public recordings: RecordingService;
  public tts: typeof TTSService;
  public stt: STTService;
  public callFlows: CallFlowEngine;
  public mediaStream: MediaStreamServer;

  constructor() {
    validateConfig();

    // Initialize clients
    this.voice = new TwilioVoiceClient({
      accountSid: twilioConfig.accountSid,
      authToken: twilioConfig.authToken
    });

    this.sms = new TwilioSMSClient(
      twilioConfig.accountSid,
      twilioConfig.authToken
    );

    this.phoneNumbers = new PhoneNumberService(
      twilioConfig.accountSid,
      twilioConfig.authToken
    );

    this.recordings = new RecordingService(
      twilioConfig.accountSid,
      twilioConfig.authToken,
      twilioConfig.openaiApiKey
    );

    this.tts = TTSService;
    this.stt = new STTService(twilioConfig.openaiApiKey);
    this.callFlows = new CallFlowEngine();

    this.mediaStream = new MediaStreamServer({
      port: twilioConfig.port,
      openaiApiKey: twilioConfig.openaiApiKey,
      twilioAccountSid: twilioConfig.accountSid,
      twilioAuthToken: twilioConfig.authToken,
      systemMessage: 'You are OpenClaw, an advanced AI agent. Be helpful, concise, and professional.',
      voice: 'alloy',
      temperature: 0.8
    });
  }

  /**
   * Start the media stream server
   */
  async start(): Promise<void> {
    await this.mediaStream.start();
    console.log('[TwilioIntegration] Media Stream Server started');
  }

  /**
   * Stop all services
   */
  async stop(): Promise<void> {
    await this.mediaStream.stop();
    console.log('[TwilioIntegration] Services stopped');
  }
}

// Export singleton instance
export const twilioIntegration = new TwilioIntegration();
```

### 12.2 Agent Telephony Actions

```typescript
// src/twilio/agent-actions.ts
import { twilioIntegration } from './index';

export interface CallContact {
  name?: string;
  phoneNumber: string;
}

export interface CallResult {
  success: boolean;
  callSid?: string;
  error?: string;
}

export interface SMSResult {
  success: boolean;
  messageSid?: string;
  error?: string;
}

export class AgentTelephonyActions {
  /**
   * Make an outbound call
   */
  async call(contact: CallContact, message?: string): Promise<CallResult> {
    try {
      const voiceResponse = new (await import('./voice/voice-response')).TwilioVoiceResponse();
      
      if (message) {
        voiceResponse.say(message);
      }
      
      voiceResponse.connectToMediaStream(
        `wss://${process.env.WEBHOOK_BASE_URL}/media-stream`
      );

      const call = await twilioIntegration.voice.makeCall({
        to: contact.phoneNumber,
        from: process.env.TWILIO_PHONE_NUMBER!,
        twiml: voiceResponse.toString(),
        statusCallback: `https://${process.env.WEBHOOK_BASE_URL}/voice/status`,
        recording: true
      });

      return {
        success: true,
        callSid: call.sid
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Send SMS message
   */
  async sendSMS(to: string, message: string): Promise<SMSResult> {
    try {
      const result = await twilioIntegration.sms.sendMessage({
        to,
        from: process.env.TWILIO_PHONE_NUMBER!,
        body: message,
        statusCallback: `https://${process.env.WEBHOOK_BASE_URL}/sms/status`
      });

      return {
        success: true,
        messageSid: result.sid
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Send MMS with media
   */
  async sendMMS(to: string, message: string, mediaUrls: string[]): Promise<SMSResult> {
    try {
      const result = await twilioIntegration.sms.sendMessage({
        to,
        from: process.env.TWILIO_PHONE_NUMBER!,
        body: message,
        mediaUrl: mediaUrls,
        statusCallback: `https://${process.env.WEBHOOK_BASE_URL}/sms/status`
      });

      return {
        success: true,
        messageSid: result.sid
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * End an active call
   */
  async endCall(callSid: string): Promise<boolean> {
    try {
      await twilioIntegration.voice.endCall(callSid);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get call transcript
   */
  async getTranscript(callSid: string): Promise<string | null> {
    try {
      const recordings = await twilioIntegration.recordings.listRecordings(callSid);
      
      if (recordings.length === 0) {
        return null;
      }

      const transcription = await twilioIntegration.recordings.transcribeRecording(
        recordings[0].sid
      );

      return transcription.text;
    } catch (error) {
      return null;
    }
  }

  /**
   * Speak text during a call (update active call)
   */
  async speakDuringCall(callSid: string, text: string): Promise<boolean> {
    try {
      const voiceResponse = new (await import('./voice/voice-response')).TwilioVoiceResponse();
      voiceResponse.say(text);

      await twilioIntegration.voice.updateCall(callSid, voiceResponse.toString());
      return true;
    } catch (error) {
      return false;
    }
  }
}

export const agentActions = new AgentTelephonyActions();
```

---

## 13. Configuration & Environment

### 13.1 Required Environment Variables

```bash
# .env file for OpenClaw Twilio Integration

# ===========================================
# TWILIO CREDENTIALS (Required)
# ===========================================
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890

# Optional: API Key for additional security
TWILIO_API_KEY=SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===========================================
# OPENAI CREDENTIALS (Required for AI/STT)
# ===========================================
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===========================================
# WEBHOOK CONFIGURATION
# ===========================================
WEBHOOK_BASE_URL=your-ngrok-or-domain.com
VALIDATE_WEBHOOKS=true

# ===========================================
# SERVER CONFIGURATION
# ===========================================
PORT=3000
NODE_ENV=production

# ===========================================
# CALL SETTINGS
# ===========================================
RECORD_CALLS=true
RECORDING_FORMAT=mp3

# ===========================================
# TTS SETTINGS
# ===========================================
DEFAULT_VOICE=Polly.Joanna-Neural
DEFAULT_LANGUAGE=en-US

# ===========================================
# AI AGENT SETTINGS
# ===========================================
AGENT_NAME=OpenClaw
AGENT_VOICE=alloy
AGENT_TEMPERATURE=0.8
AGENT_SYSTEM_PROMPT="You are OpenClaw, an advanced AI agent. Be helpful, concise, and professional."
```

### 13.2 Package.json Dependencies

```json
{
  "name": "openclaw-twilio-integration",
  "version": "1.0.0",
  "description": "Twilio Voice & SMS Integration for OpenClaw AI Agent",
  "main": "dist/index.js",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "tsx watch src/index.ts",
    "test": "jest",
    "lint": "eslint src/**/*.ts"
  },
  "dependencies": {
    "twilio": "^5.0.0",
    "openai": "^4.0.0",
    "express": "^4.18.0",
    "fastify": "^4.0.0",
    "@fastify/formbody": "^7.0.0",
    "@fastify/websocket": "^8.0.0",
    "ws": "^8.0.0",
    "dotenv": "^16.0.0",
    "@openai/agents": "^0.1.0",
    "@openai/agents-extensions": "^0.1.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/express": "^4.17.0",
    "@types/ws": "^8.0.0",
    "typescript": "^5.0.0",
    "tsx": "^4.0.0",
    "jest": "^29.0.0",
    "@types/jest": "^29.0.0",
    "eslint": "^8.0.0"
  }
}
```

---

## 14. Error Handling & Monitoring

### 14.1 Error Handler

```typescript
// src/twilio/utils/error-handler.ts
export class TwilioError extends Error {
  public code: string;
  public statusCode: number;
  public isRetryable: boolean;

  constructor(
    message: string,
    code: string,
    statusCode: number = 500,
    isRetryable: boolean = false
  ) {
    super(message);
    this.name = 'TwilioError';
    this.code = code;
    this.statusCode = statusCode;
    this.isRetryable = isRetryable;
  }
}

export const ErrorCodes = {
  // Voice errors
  CALL_FAILED: 'CALL_FAILED',
  CALL_NOT_FOUND: 'CALL_NOT_FOUND',
  INVALID_PHONE_NUMBER: 'INVALID_PHONE_NUMBER',
  
  // SMS errors
  SMS_FAILED: 'SMS_FAILED',
  INVALID_MESSAGE_BODY: 'INVALID_MESSAGE_BODY',
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  
  // Phone number errors
  NUMBER_NOT_AVAILABLE: 'NUMBER_NOT_AVAILABLE',
  NUMBER_PURCHASE_FAILED: 'NUMBER_PURCHASE_FAILED',
  
  // Recording errors
  RECORDING_FAILED: 'RECORDING_FAILED',
  TRANSCRIPTION_FAILED: 'TRANSCRIPTION_FAILED',
  
  // Media Stream errors
  STREAM_CONNECTION_FAILED: 'STREAM_CONNECTION_FAILED',
  AUDIO_PROCESSING_ERROR: 'AUDIO_PROCESSING_ERROR',
  
  // Webhook errors
  INVALID_SIGNATURE: 'INVALID_SIGNATURE',
  WEBHOOK_TIMEOUT: 'WEBHOOK_TIMEOUT'
} as const;

export function handleTwilioError(error: any): TwilioError {
  // Parse Twilio error responses
  if (error.code && error.message) {
    return new TwilioError(
      error.message,
      error.code,
      error.status || 500,
      isRetryableError(error.code)
    );
  }

  return new TwilioError(
    error.message || 'Unknown error',
    'UNKNOWN_ERROR',
    500,
    false
  );
}

function isRetryableError(code: string | number): boolean {
  const retryableCodes = [
    20429, // Rate limit
    20450, // Concurrent request limit
    20500, // Internal server error
    20503, // Service unavailable
    'RATE_LIMIT_EXCEEDED',
    'WEBHOOK_TIMEOUT'
  ];

  return retryableCodes.includes(code as any);
}
```

### 14.2 Monitoring & Logging

```typescript
// src/twilio/utils/logger.ts
export interface CallMetrics {
  callSid: string;
  from: string;
  to: string;
  startTime: Date;
  endTime?: Date;
  duration?: number;
  status: string;
  recordingUrl?: string;
  transcriptionLength?: number;
}

export interface SMSMetrics {
  messageSid: string;
  from: string;
  to: string;
  timestamp: Date;
  status: string;
  bodyLength: number;
  hasMedia: boolean;
}

export class TwilioLogger {
  private callMetrics: Map<string, CallMetrics> = new Map();
  private smsMetrics: Map<string, SMSMetrics> = new Map();

  logCallStarted(callSid: string, from: string, to: string): void {
    const metrics: CallMetrics = {
      callSid,
      from,
      to,
      startTime: new Date(),
      status: 'initiated'
    };
    this.callMetrics.set(callSid, metrics);
    console.log(`[CALL] Started: ${callSid} from ${from} to ${to}`);
  }

  logCallEnded(callSid: string, status: string, duration?: number): void {
    const metrics = this.callMetrics.get(callSid);
    if (metrics) {
      metrics.endTime = new Date();
      metrics.duration = duration;
      metrics.status = status;
      console.log(`[CALL] Ended: ${callSid}, Status: ${status}, Duration: ${duration}s`);
    }
  }

  logSMSSent(messageSid: string, from: string, to: string, bodyLength: number, hasMedia: boolean): void {
    const metrics: SMSMetrics = {
      messageSid,
      from,
      to,
      timestamp: new Date(),
      status: 'sent',
      bodyLength,
      hasMedia
    };
    this.smsMetrics.set(messageSid, metrics);
    console.log(`[SMS] Sent: ${messageSid} to ${to}`);
  }

  logError(context: string, error: Error): void {
    console.error(`[ERROR] ${context}:`, error.message);
    // Send to error tracking service
  }

  getMetrics(): { calls: CallMetrics[]; sms: SMSMetrics[] } {
    return {
      calls: Array.from(this.callMetrics.values()),
      sms: Array.from(this.smsMetrics.values())
    };
  }
}

export const twilioLogger = new TwilioLogger();
```

---

## Summary

This technical specification provides a comprehensive architecture for integrating Twilio Voice and SMS capabilities into the OpenClaw Windows 10 AI agent system. Key features include:

1. **Complete Twilio SDK Integration** - Voice calls, SMS/MMS, phone number management
2. **Real-time Media Streams** - Bidirectional audio streaming with AI integration
3. **Dynamic TwiML Generation** - Flexible call flow management
4. **Webhook Security** - Request validation and secure communication
5. **TTS/STT Integration** - Multiple voice options and transcription services
6. **Call Recording & Analysis** - Automatic recording with AI-powered transcription
7. **Error Handling & Monitoring** - Comprehensive logging and metrics

The architecture supports all 15 agentic loops with robust telephony capabilities for a 24/7 autonomous AI agent system.
