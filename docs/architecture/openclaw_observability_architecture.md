# OpenClaw Windows 10 AI Agent - Logging & Observability Architecture
## Technical Specification Document

**Version:** 1.0  
**Date:** 2025  
**Platform:** Windows 10 / Node.js  
**Target:** GPT-5.2 AI Agent System

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Structured Logging System](#3-structured-logging-system)
4. [Log Rotation & Retention Policies](#4-log-rotation--retention-policies)
5. [Telemetry & Metrics Collection](#5-telemetry--metrics-collection)
6. [Distributed Tracing](#6-distributed-tracing)
7. [Performance Monitoring](#7-performance-monitoring)
8. [Health Check Endpoints](#8-health-check-endpoints)
9. [Dashboard & Visualization](#9-dashboard--visualization)
10. [Alerting Rules & Thresholds](#10-alerting-rules--thresholds)
11. [Implementation Code Samples](#11-implementation-code-samples)
12. [Appendix: Configuration Reference](#12-appendix-configuration-reference)

---

## 1. Executive Summary

This document defines the complete logging and observability architecture for a Windows 10-based OpenClaw-inspired AI agent system. The architecture provides:

- **Comprehensive visibility** into 15 hardcoded agentic loops
- **24/7 monitoring** for cron jobs, heartbeat, soul, identity, and user systems
- **Full integration coverage** for Gmail, browser control, TTS, STT, Twilio voice/SMS
- **Production-grade reliability** with structured logging, distributed tracing, and intelligent alerting

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Observability-by-Design** | Instrumentation built into every component from day one |
| **Open Standards** | OpenTelemetry for traces, metrics, and logs |
| **Structured Everything** | JSON-first logging with consistent schemas |
| **Minimal Overhead** | Pino for 5x faster logging with async processing |
| **Windows-Native** | ETW integration and Windows Event Log forwarding |

---

## 2. Architecture Overview

### 2.1 High-Level Observability Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY DASHBOARD LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Grafana   │  │  Prometheus │  │   Jaeger    │  │  Windows Event Log  │ │
│  │ Dashboards  │  │   Metrics   │  │    Traces   │  │     Integration     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COLLECTION & AGGREGATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ OpenTelemetry│  │   Vector    │  │  Log Rotate │  │   Alert Manager     │ │
│  │  Collector  │  │   Pipeline  │  │   Service   │  │   (Prometheus)      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INSTRUMENTATION LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Pino     │  │   OpenTel   │  │   Node      │  │   Winston (legacy)  │ │
│  │   Logger    │  │    SDK      │  │  Profiler   │  │   Compatibility     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI AGENT SYSTEM COMPONENTS                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │  Gateway │ │  Agent   │ │   Tool   │ │  Memory  │ │  Cron    │ │ Heart- ││
│  │  Server  │ │  Runner  │ │Execution │ │  System  │ │  Jobs    │ │  beat  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │   Soul   │ │ Identity │ │   User   │ │  Gmail   │ │ Browser  │ │  TTS   ││
│  │  Engine  │ │  System  │ │  System  │ │  Client  │ │  Control │ │  STT   ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                      │
│  │  Twilio  │ │  System  │ │  15 Loops│                                      │
│  │ Voice/SMS│ │  Access  │ │  Agentic │                                      │
│  └──────────┘ └──────────┘ └──────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Pino Logger** | High-performance structured logging | Pino 9.x with transports |
| **OpenTelemetry SDK** | Distributed tracing and metrics | OTel JS 1.x |
| **Vector Pipeline** | Log transformation and routing | Vector 0.35+ |
| **Prometheus** | Metrics collection and storage | Prometheus 2.50+ |
| **Grafana** | Visualization and dashboards | Grafana 10.3+ |
| **Jaeger** | Distributed trace storage and UI | Jaeger 1.50+ |
| **Alertmanager** | Alert routing and notification | Alertmanager 0.27+ |

---

## 3. Structured Logging System

### 3.1 Log Severity Levels

```typescript
// LogLevel Enum Definition
enum LogLevel {
  TRACE = 10,    // Detailed debugging, function entry/exit
  DEBUG = 20,    // Development debugging, variable states
  INFO = 30,     // Normal operations, state changes
  WARN = 40,     // Warning conditions, recoverable issues
  ERROR = 50,    // Error conditions, failed operations
  FATAL = 60,    // Critical failures, system shutdown
  SECURITY = 55, // Security events (custom level)
  AUDIT = 35,    // Audit trail events (custom level)
}
```

### 3.2 Log Schema (JSON Structure)

```json
{
  "timestamp": "2025-01-15T14:30:45.123Z",
  "level": 30,
  "level_name": "INFO",
  "message": "Agent loop completed successfully",
  "service": "openclaw-agent",
  "version": "1.0.0",
  "environment": "production",
  "hostname": "WIN10-AGENT-01",
  "pid": 12345,
  "trace_id": "abc123def456",
  "span_id": "span789xyz",
  "parent_span_id": "parent456abc",
  "session_id": "session_20250115_001",
  "user_id": "user_abc123",
  "agent_id": "agent_main",
  "loop_id": "loop_01_think_act",
  "component": "agent_runner",
  "operation": "execute_loop",
  "duration_ms": 2450,
  "context": {
    "model": "gpt-5.2",
    "thinking_mode": "extra_high",
    "tokens_input": 1500,
    "tokens_output": 450,
    "tools_used": ["browser_navigate", "gmail_send"]
  },
  "metadata": {
    "source_file": "src/agents/loops/thinkActLoop.ts",
    "line_number": 145,
    "function": "executeThinkAct"
  },
  "error": null
}
```

### 3.3 Component-Specific Log Schemas

#### Agent Runner Logs
```json
{
  "timestamp": "2025-01-15T14:30:45.123Z",
  "level": 30,
  "level_name": "INFO",
  "message": "Agent execution started",
  "service": "openclaw-agent",
  "component": "agent_runner",
  "operation": "execute_agent",
  "agent_type": "conversational",
  "model_config": {
    "provider": "openai",
    "model": "gpt-5.2",
    "temperature": 0.7,
    "max_tokens": 4096,
    "thinking": "extra_high"
  },
  "session_context": {
    "session_id": "sess_abc123",
    "conversation_turn": 5,
    "total_messages": 12
  },
  "tool_inventory": ["browser", "gmail", "calendar", "file_system"],
  "memory_stats": {
    "working_memory_entries": 15,
    "long_term_memory_hits": 3
  }
}
```

#### Tool Execution Logs
```json
{
  "timestamp": "2025-01-15T14:30:47.456Z",
  "level": 30,
  "level_name": "INFO",
  "message": "Tool execution completed",
  "service": "openclaw-agent",
  "component": "tool_executor",
  "operation": "execute_tool",
  "tool": {
    "name": "browser_navigate",
    "category": "browser",
    "version": "1.0.0"
  },
  "execution": {
    "start_time": "2025-01-15T14:30:46.100Z",
    "end_time": "2025-01-15T14:30:47.456Z",
    "duration_ms": 1356,
    "status": "success"
  },
  "input": {
    "url": "https://gmail.com",
    "wait_for_load": true
  },
  "output": {
    "page_title": "Gmail",
    "load_time_ms": 1200,
    "status_code": 200
  },
  "sandbox_info": {
    "container_id": "sandbox_001",
    "resource_usage": {
      "cpu_percent": 15.2,
      "memory_mb": 128
    }
  }
}
```

#### Memory System Logs
```json
{
  "timestamp": "2025-01-15T14:30:48.789Z",
  "level": 30,
  "level_name": "INFO",
  "message": "Memory operation completed",
  "service": "openclaw-agent",
  "component": "memory_system",
  "operation": "store_memory",
  "memory_type": "working",
  "storage_backend": "sqlite_vector",
  "operation_details": {
    "entries_stored": 3,
    "embedding_model": "text-embedding-3-large",
    "index_updated": true
  },
  "performance": {
    "embedding_time_ms": 245,
    "storage_time_ms": 12,
    "total_time_ms": 257
  }
}
```

#### Cron Job Logs
```json
{
  "timestamp": "2025-01-15T14:30:00.000Z",
  "level": 30,
  "level_name": "INFO",
  "message": "Scheduled job executed",
  "service": "openclaw-agent",
  "component": "cron_scheduler",
  "operation": "execute_scheduled_job",
  "job": {
    "id": "job_check_emails",
    "name": "Email Inbox Check",
    "schedule": "*/15 * * * *",
    "timezone": "America/New_York"
  },
  "execution": {
    "scheduled_time": "2025-01-15T14:30:00.000Z",
    "actual_start": "2025-01-15T14:30:00.023Z",
    "end_time": "2025-01-15T14:30:05.456Z",
    "duration_ms": 5433,
    "status": "success"
  },
  "result": {
    "new_emails_found": 3,
    "actions_taken": ["notify_user", "categorize"]
  }
}
```

#### Heartbeat Logs
```json
{
  "timestamp": "2025-01-15T14:30:00.000Z",
  "level": 20,
  "level_name": "DEBUG",
  "message": "System heartbeat",
  "service": "openclaw-agent",
  "component": "heartbeat_monitor",
  "operation": "heartbeat_tick",
  "heartbeat": {
    "sequence": 123456,
    "interval_seconds": 30
  },
  "system_health": {
    "status": "healthy",
    "uptime_seconds": 86400,
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "disk_percent": 62.1
  },
  "component_health": {
    "gateway_server": "healthy",
    "agent_runner": "healthy",
    "memory_system": "healthy",
    "tool_executor": "healthy",
    "cron_scheduler": "healthy"
  }
}
```

#### Security/Audit Logs
```json
{
  "timestamp": "2025-01-15T14:30:45.123Z",
  "level": 55,
  "level_name": "SECURITY",
  "message": "User authentication successful",
  "service": "openclaw-agent",
  "component": "auth_system",
  "operation": "user_login",
  "event_type": "authentication",
  "severity": "info",
  "actor": {
    "user_id": "user_abc123",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0..."
  },
  "resource": {
    "type": "system",
    "action": "login"
  },
  "outcome": "success",
  "authentication": {
    "method": "api_key",
    "mfa_used": true,
    "session_id": "sess_xyz789"
  }
}
```

### 3.4 Logger Configuration (Pino)

```typescript
// src/logging/logger.ts
import pino from 'pino';
import { join } from 'path';

// Custom log levels for AI agent system
const customLevels = {
  trace: 10,
  debug: 20,
  info: 30,
  audit: 35,
  warn: 40,
  error: 50,
  security: 55,
  fatal: 60,
};

const loggerConfig = {
  level: process.env.LOG_LEVEL || 'info',
  customLevels: customLevels,
  useOnlyCustomLevels: false,
  
  // Base metadata applied to all logs
  base: {
    service: 'openclaw-agent',
    version: process.env.APP_VERSION || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    hostname: require('os').hostname(),
    pid: process.pid,
  },

  // Timestamp format
  timestamp: pino.stdTimeFunctions.isoTime,

  // Redact sensitive fields
  redact: {
    paths: [
      'password',
      'secret',
      'token',
      'api_key',
      'authorization',
      'cookie',
      '*.password',
      '*.secret',
      '*.token',
      'context.tokens_input', // Don't log actual tokens
      'context.tokens_output',
    ],
    remove: true,
  },

  // Formatters
  formatters: {
    level(label: string, number: number) {
      return { level: number, level_name: label.toUpperCase() };
    },
    bindings(bindings: Record<string, any>) {
      return bindings;
    },
    log(object: Record<string, any>) {
      return object;
    },
  },

  // Async logging for performance
  transport: process.env.NODE_ENV === 'production' 
    ? {
        targets: [
          // Console output for development/debugging
          {
            target: 'pino-pretty',
            level: 'warn',
            options: {
              colorize: true,
              translateTime: 'SYS:standard',
              ignore: 'pid,hostname',
            },
          },
          // File output for all logs
          {
            target: 'pino/file',
            level: 'info',
            options: {
              destination: join(process.env.LOG_DIR || './logs', 'app.log'),
              mkdir: true,
            },
          },
          // Error logs separate
          {
            target: 'pino/file',
            level: 'error',
            options: {
              destination: join(process.env.LOG_DIR || './logs', 'error.log'),
              mkdir: true,
            },
          },
          // Security audit logs
          {
            target: 'pino/file',
            level: 'security',
            options: {
              destination: join(process.env.LOG_DIR || './logs', 'audit.log'),
              mkdir: true,
            },
          },
          // OpenTelemetry transport for centralized collection
          {
            target: 'pino-opentelemetry-transport',
            level: 'info',
            options: {
              endpoint: process.env.OTEL_COLLECTOR_URL || 'http://localhost:4318',
            },
          },
        ],
      }
    : undefined,
};

// Create main logger
export const logger = pino(loggerConfig);

// Child logger factory for components
export function createComponentLogger(
  component: string,
  metadata: Record<string, any> = {}
) {
  return logger.child({
    component,
    ...metadata,
  });
}

// Component loggers
export const loggers = {
  gateway: createComponentLogger('gateway_server'),
  agent: createComponentLogger('agent_runner'),
  tools: createComponentLogger('tool_executor'),
  memory: createComponentLogger('memory_system'),
  cron: createComponentLogger('cron_scheduler'),
  heartbeat: createComponentLogger('heartbeat_monitor'),
  soul: createComponentLogger('soul_engine'),
  identity: createComponentLogger('identity_system'),
  user: createComponentLogger('user_system'),
  gmail: createComponentLogger('gmail_client'),
  browser: createComponentLogger('browser_control'),
  tts: createComponentLogger('tts_service'),
  stt: createComponentLogger('stt_service'),
  twilio: createComponentLogger('twilio_service'),
  system: createComponentLogger('system_access'),
  security: createComponentLogger('auth_system'),
};

// Request context logger (for HTTP requests)
export function createRequestLogger(
  requestId: string,
  sessionId: string,
  userId?: string
) {
  return logger.child({
    request_id: requestId,
    session_id: sessionId,
    user_id: userId,
  });
}

// Agent loop logger (for the 15 agentic loops)
export function createLoopLogger(
  loopId: string,
  loopName: string,
  sessionId: string
) {
  return logger.child({
    loop_id: loopId,
    loop_name: loopName,
    session_id: sessionId,
  });
}
```

---

## 4. Log Rotation & Retention Policies

### 4.1 Rotation Strategy

| Log Type | Rotation Trigger | Max Files | Compression | Max Age |
|----------|-----------------|-----------|-------------|---------|
| **Application Logs** | 100MB or 24h | 30 | gzip | 30 days |
| **Error Logs** | 50MB or 24h | 60 | gzip | 60 days |
| **Audit Logs** | 50MB or 24h | 365 | gzip | 7 years |
| **Security Logs** | 25MB or 6h | 365 | gzip | 7 years |
| **Debug Logs** | 500MB or 6h | 7 | gzip | 7 days |
| **Trace Logs** | 1GB or 1h | 24 | gzip | 24 hours |

### 4.2 Windows-Specific Log Rotation

```powershell
# scripts/log-rotation.ps1
# Windows PowerShell log rotation script for OpenClaw Agent

param(
    [string]$LogDir = "C:\OpenClaw\logs",
    [string]$ArchiveDir = "C:\OpenClaw\logs\archive",
    [int]$MaxLogAgeDays = 30
)

# Ensure archive directory exists
if (!(Test-Path $ArchiveDir)) {
    New-Item -ItemType Directory -Path $ArchiveDir -Force
}

# Function to rotate logs
function Rotate-LogFile {
    param(
        [string]$FilePath,
        [int]$MaxSizeMB = 100,
        [int]$MaxBackups = 30
    )
    
    $file = Get-Item $FilePath -ErrorAction SilentlyContinue
    if (!$file) { return }
    
    $maxSizeBytes = $MaxSizeMB * 1MB
    
    if ($file.Length -gt $maxSizeBytes) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $archiveName = "$($file.BaseName)_$timestamp$($file.Extension).gz"
        $archivePath = Join-Path $ArchiveDir $archiveName
        
        # Compress and move
        $file | Compress-Archive -DestinationPath $archivePath -Force
        
        # Clear original file
        Clear-Content $FilePath
        
        Write-Host "Rotated: $FilePath -> $archivePath"
    }
}

# Function to cleanup old logs
function Remove-OldLogs {
    param(
        [string]$Directory,
        [int]$MaxAgeDays
    )
    
    $cutoffDate = (Get-Date).AddDays(-$MaxAgeDays)
    
    Get-ChildItem $Directory -Recurse -File | Where-Object {
        $_.LastWriteTime -lt $cutoffDate
    } | ForEach-Object {
        Remove-Item $_.FullName -Force
        Write-Host "Deleted old log: $($_.FullName)"
    }
}

# Rotate main log files
$logFiles = @(
    "app.log",
    "error.log",
    "audit.log",
    "security.log",
    "debug.log"
)

foreach ($logFile in $logFiles) {
    $fullPath = Join-Path $LogDir $logFile
    Rotate-LogFile -FilePath $fullPath
}

# Cleanup old archived logs
Remove-OldLogs -Directory $ArchiveDir -MaxAgeDays $MaxLogAgeDays

# Log rotation event to Windows Event Log
Write-EventLog -LogName "Application" -Source "OpenClaw Agent" -EventId 1001 -Message "Log rotation completed"
```

### 4.3 Node.js Log Rotation (Using Rotating-File-Stream)

```typescript
// src/logging/rotation.ts
import { createStream } from 'rotating-file-stream';
import { join } from 'path';

const LOG_DIR = process.env.LOG_DIR || './logs';

// Generator function for rotated filenames
function generator(time: number | Date, index?: number): string {
  if (!time) return 'app.log';
  
  const date = time instanceof Date ? time : new Date(time);
  const pad = (n: number) => (n < 10 ? '0' + n : n);
  
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hour = pad(date.getHours());
  
  return `${year}${month}${day}-${hour}-${index}-app.log`;
}

// Create rotating stream for application logs
export const appLogStream = createStream(generator, {
  path: LOG_DIR,
  size: '100M',        // Rotate at 100MB
  interval: '1d',      // Rotate daily
  compress: 'gzip',    // Compress rotated files
  maxFiles: 30,        // Keep 30 files
  maxSize: '3G',       // Max total size
  immutable: true,     // Don't modify rotated files
});

// Create rotating stream for error logs
export const errorLogStream = createStream(
  (time, index) => {
    if (!time) return 'error.log';
    const date = time instanceof Date ? time : new Date(time);
    return `error-${date.toISOString().split('T')[0]}-${index}.log`;
  },
  {
    path: LOG_DIR,
    size: '50M',
    interval: '1d',
    compress: 'gzip',
    maxFiles: 60,
    maxSize: '2G',
  }
);

// Create rotating stream for audit logs (longer retention)
export const auditLogStream = createStream(
  (time, index) => {
    if (!time) return 'audit.log';
    const date = time instanceof Date ? time : new Date(time);
    return `audit-${date.toISOString().split('T')[0]}-${index}.log`;
  },
  {
    path: join(LOG_DIR, 'audit'),
    size: '50M',
    interval: '1d',
    compress: 'gzip',
    maxFiles: 365,     // Keep for 1 year
    maxSize: '10G',
  }
);

// Cleanup job for old logs
export function startLogCleanupJob(): void {
  const cleanup = async () => {
    const { readdir, stat, unlink } = await import('fs/promises');
    const { join } = await import('path');
    
    const maxAgeMs = {
      debug: 7 * 24 * 60 * 60 * 1000,      // 7 days
      app: 30 * 24 * 60 * 60 * 1000,       // 30 days
      error: 60 * 24 * 60 * 60 * 1000,     // 60 days
      audit: 7 * 365 * 24 * 60 * 60 * 1000, // 7 years
    };
    
    const now = Date.now();
    
    for (const [type, maxAge] of Object.entries(maxAgeMs)) {
      const dir = join(LOG_DIR, type);
      
      try {
        const files = await readdir(dir);
        
        for (const file of files) {
          const filePath = join(dir, file);
          const stats = await stat(filePath);
          
          if (now - stats.mtime.getTime() > maxAge) {
            await unlink(filePath);
            console.log(`Deleted old log: ${filePath}`);
          }
        }
      } catch (err) {
        console.error(`Failed to cleanup ${type} logs:`, err);
      }
    }
  };
  
  // Run cleanup daily
  setInterval(cleanup, 24 * 60 * 60 * 1000);
  cleanup(); // Run immediately
}
```

### 4.4 Retention Policy Matrix

| Compliance Requirement | Retention Period | Log Types | Encryption |
|----------------------|------------------|-----------|------------|
| **SOC 2** | 1 year | All operational logs | AES-256 |
| **GDPR** | As needed + 30 days | User data access logs | AES-256 |
| **HIPAA** | 6 years | PHI access logs | AES-256 + HSM |
| **PCI DSS** | 1 year + 3 months archive | Payment-related logs | AES-256 |
| **Internal Security** | 7 years | Security/audit logs | AES-256 |
| **Debug/Troubleshooting** | 7 days | Debug/trace logs | None |

---

## 5. Telemetry & Metrics Collection

### 5.1 Metrics Categories

```typescript
// src/telemetry/metrics.ts
import { MeterProvider, PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';

// Initialize meter provider
const meterProvider = new MeterProvider({
  readers: [
    // Prometheus exporter for local scraping
    new PrometheusExporter({
      port: 9090,
      endpoint: '/metrics',
    }),
    // OTLP exporter for centralized collection
    new PeriodicExportingMetricReader({
      exporter: new OTLPMetricExporter({
        url: process.env.OTEL_METRICS_ENDPOINT || 'http://localhost:4318/v1/metrics',
      }),
      exportIntervalMillis: 60000, // Export every minute
    }),
  ],
});

const meter = meterProvider.getMeter('openclaw-agent');

// ==================== SYSTEM METRICS ====================

// CPU Usage
export const systemCpuUsage = meter.createObservableGauge('system.cpu.usage', {
  description: 'CPU usage percentage',
  unit: '%',
});

// Memory Usage
export const systemMemoryUsage = meter.createObservableGauge('system.memory.usage', {
  description: 'Memory usage in bytes',
  unit: 'By',
});

export const systemMemoryPercent = meter.createObservableGauge('system.memory.percent', {
  description: 'Memory usage percentage',
  unit: '%',
});

// Disk Usage
export const systemDiskUsage = meter.createObservableGauge('system.disk.usage', {
  description: 'Disk usage percentage',
  unit: '%',
});

// Uptime
export const systemUptime = meter.createObservableCounter('system.uptime', {
  description: 'System uptime in seconds',
  unit: 's',
});

// ==================== AGENT METRICS ====================

// Agent executions
export const agentExecutionsTotal = meter.createCounter('agent.executions.total', {
  description: 'Total number of agent executions',
});

export const agentExecutionsActive = meter.createUpDownCounter('agent.executions.active', {
  description: 'Number of currently active agent executions',
});

// Agent execution duration
export const agentExecutionDuration = meter.createHistogram('agent.execution.duration', {
  description: 'Agent execution duration in milliseconds',
  unit: 'ms',
  advice: {
    explicitBucketBoundaries: [100, 500, 1000, 2500, 5000, 10000, 30000, 60000],
  },
});

// Agent errors
export const agentErrorsTotal = meter.createCounter('agent.errors.total', {
  description: 'Total number of agent execution errors',
});

// ==================== LOOP METRICS (15 Agentic Loops) ====================

// Loop execution counter
export const loopExecutionsTotal = meter.createCounter('agent.loop.executions.total', {
  description: 'Total executions per agentic loop',
});

// Loop execution duration histogram
export const loopExecutionDuration = meter.createHistogram('agent.loop.execution.duration', {
  description: 'Loop execution duration in milliseconds',
  unit: 'ms',
});

// Loop iteration counter
export const loopIterationsTotal = meter.createCounter('agent.loop.iterations.total', {
  description: 'Total iterations across all loops',
});

// ==================== LLM METRICS ====================

// Token usage
export const llmTokensInput = meter.createCounter('llm.tokens.input', {
  description: 'Total input tokens consumed',
});

export const llmTokensOutput = meter.createCounter('llm.tokens.output', {
  description: 'Total output tokens generated',
});

export const llmTokensTotal = meter.createCounter('llm.tokens.total', {
  description: 'Total tokens (input + output)',
});

// LLM latency
export const llmLatency = meter.createHistogram('llm.latency', {
  description: 'LLM API call latency in milliseconds',
  unit: 'ms',
  advice: {
    explicitBucketBoundaries: [100, 250, 500, 1000, 2000, 5000, 10000],
  },
});

// LLM errors
export const llmErrorsTotal = meter.createCounter('llm.errors.total', {
  description: 'Total LLM API errors',
});

// Cost tracking
export const llmCostUsd = meter.createCounter('llm.cost.usd', {
  description: 'Estimated LLM cost in USD',
  unit: '$',
});

// ==================== TOOL METRICS ====================

// Tool execution counter
export const toolExecutionsTotal = meter.createCounter('tool.executions.total', {
  description: 'Total tool executions by tool name',
});

// Tool execution duration
export const toolExecutionDuration = meter.createHistogram('tool.execution.duration', {
  description: 'Tool execution duration in milliseconds',
  unit: 'ms',
});

// Tool errors
export const toolErrorsTotal = meter.createCounter('tool.errors.total', {
  description: 'Total tool execution errors',
});

// Tool success rate
export const toolSuccessRate = meter.createObservableGauge('tool.success.rate', {
  description: 'Tool execution success rate (0-1)',
});

// ==================== MEMORY METRICS ====================

// Memory entries
export const memoryEntriesTotal = meter.createUpDownCounter('memory.entries.total', {
  description: 'Total memory entries stored',
});

// Memory operations
export const memoryOperationsTotal = meter.createCounter('memory.operations.total', {
  description: 'Total memory operations by type',
});

// Memory operation duration
export const memoryOperationDuration = meter.createHistogram('memory.operation.duration', {
  description: 'Memory operation duration in milliseconds',
  unit: 'ms',
});

// Vector search metrics
export const vectorSearchDuration = meter.createHistogram('vector.search.duration', {
  description: 'Vector search duration in milliseconds',
  unit: 'ms',
});

export const vectorSearchResults = meter.createHistogram('vector.search.results', {
  description: 'Number of results from vector search',
});

// ==================== CRON METRICS ====================

// Cron job executions
export const cronExecutionsTotal = meter.createCounter('cron.executions.total', {
  description: 'Total cron job executions',
});

// Cron job duration
export const cronExecutionDuration = meter.createHistogram('cron.execution.duration', {
  description: 'Cron job execution duration in milliseconds',
  unit: 'ms',
});

// Cron job lag (scheduled vs actual start)
export const cronExecutionLag = meter.createHistogram('cron.execution.lag', {
  description: 'Delay between scheduled and actual execution in milliseconds',
  unit: 'ms',
});

// Cron job errors
export const cronErrorsTotal = meter.createCounter('cron.errors.total', {
  description: 'Total cron job errors',
});

// ==================== INTEGRATION METRICS ====================

// Gmail metrics
export const gmailOperationsTotal = meter.createCounter('gmail.operations.total', {
  description: 'Total Gmail operations',
});

export const gmailOperationDuration = meter.createHistogram('gmail.operation.duration', {
  description: 'Gmail operation duration in milliseconds',
  unit: 'ms',
});

// Browser metrics
export const browserNavigationsTotal = meter.createCounter('browser.navigations.total', {
  description: 'Total browser navigations',
});

export const browserNavigationDuration = meter.createHistogram('browser.navigation.duration', {
  description: 'Browser navigation duration in milliseconds',
  unit: 'ms',
});

// TTS/STT metrics
export const ttsRequestsTotal = meter.createCounter('tts.requests.total', {
  description: 'Total TTS requests',
});

export const ttsDuration = meter.createHistogram('tts.duration', {
  description: 'TTS generation duration in milliseconds',
  unit: 'ms',
});

export const sttRequestsTotal = meter.createCounter('stt.requests.total', {
  description: 'Total STT requests',
});

// Twilio metrics
export const twilioCallsTotal = meter.createCounter('twilio.calls.total', {
  description: 'Total Twilio calls',
});

export const twilioSmsTotal = meter.createCounter('twilio.sms.total', {
  description: 'Total Twilio SMS sent/received',
});

export const twilioCallDuration = meter.createHistogram('twilio.call.duration', {
  description: 'Twilio call duration in seconds',
  unit: 's',
});

// ==================== HEARTBEAT METRICS ====================

// Heartbeat counter
export const heartbeatTotal = meter.createCounter('heartbeat.total', {
  description: 'Total heartbeat ticks',
});

// Heartbeat interval
export const heartbeatInterval = meter.createObservableGauge('heartbeat.interval', {
  description: 'Configured heartbeat interval in seconds',
  unit: 's',
});

// ==================== USER METRICS ====================

// Active sessions
export const activeSessions = meter.createUpDownCounter('user.sessions.active', {
  description: 'Number of active user sessions',
});

// User actions
export const userActionsTotal = meter.createCounter('user.actions.total', {
  description: 'Total user actions by type',
});

// ==================== SECURITY METRICS ====================

// Authentication attempts
export const authAttemptsTotal = meter.createCounter('auth.attempts.total', {
  description: 'Total authentication attempts',
});

// Authentication failures
export const authFailuresTotal = meter.createCounter('auth.failures.total', {
  description: 'Total authentication failures',
});

// Rate limit hits
export const rateLimitHitsTotal = meter.createCounter('ratelimit.hits.total', {
  description: 'Total rate limit hits',
});
```

### 5.2 Metric Labels (Dimensions)

```typescript
// Standard labels applied to all metrics
interface MetricLabels {
  // Service identification
  service: string;           // 'openclaw-agent'
  version: string;           // '1.0.0'
  environment: string;       // 'production' | 'staging' | 'development'
  
  // Runtime context
  hostname: string;          // 'WIN10-AGENT-01'
  region?: string;           // 'us-east-1'
  datacenter?: string;       // 'dc-01'
  
  // Agent context
  agent_id?: string;         // 'agent_main'
  agent_type?: string;       // 'conversational' | 'task' | 'autonomous'
  
  // Loop context (for the 15 agentic loops)
  loop_id?: string;          // 'loop_01_think_act'
  loop_name?: string;        // 'Think-Act Loop'
  
  // Model context
  model_provider?: string;   // 'openai' | 'anthropic' | 'local'
  model_name?: string;       // 'gpt-5.2'
  
  // Tool context
  tool_name?: string;        // 'browser_navigate'
  tool_category?: string;    // 'browser' | 'communication' | 'system'
  
  // Operation context
  operation?: string;        // 'execute' | 'query' | 'store'
  status?: string;           // 'success' | 'failure' | 'timeout'
  error_type?: string;       // 'network' | 'auth' | 'validation'
  
  // User context
  user_id?: string;          // 'user_abc123'
  user_tier?: string;        // 'free' | 'premium' | 'enterprise'
  
  // Session context
  session_id?: string;       // 'sess_xyz789'
}
```

---

## 6. Distributed Tracing

### 6.1 Trace Structure for AI Agent System

```
Trace: user_session_abc123
│
├── Span: gateway_receive (root)
│   ├── Attributes:
│   │   ├── http.method: POST
│   │   ├── http.url: /api/agent/execute
│   │   ├── user.id: user_abc123
│   │   └── session.id: sess_xyz789
│   │
│   ├── Span: auth_verify
│   │   └── Attributes:
│   │       ├── auth.method: api_key
│   │       └── auth.result: success
│   │
│   ├── Span: agent_runner_execute (main span)
│   │   ├── Attributes:
│   │   │   ├── agent.id: agent_main
│   │   │   ├── agent.loop_id: loop_01_think_act
│   │   │   └── agent.model: gpt-5.2
│   │   │
│   │   ├── Span: memory_retrieve_context
│   │   │   ├── Attributes:
│   │   │   │   ├── memory.type: working
│   │   │   │   ├── memory.entries_retrieved: 5
│   │   │   │   └── vector.search_duration_ms: 45
│   │   │   └── Events:
│   │   │       ├── "search_query_embedding_started"
│   │   │       └── "search_results_returned"
│   │   │
│   │   ├── Span: llm_generate
│   │   │   ├── Attributes:
│   │   │   │   ├── llm.provider: openai
│   │   │   │   ├── llm.model: gpt-5.2
│   │   │   │   ├── llm.thinking: extra_high
│   │   │   │   ├── llm.tokens_input: 1500
│   │   │   │   ├── llm.tokens_output: 450
│   │   │   │   ├── llm.latency_ms: 2450
│   │   │   │   └── llm.cost_usd: 0.045
│   │   │   └── Events:
│   │   │       ├── "prompt_sent"
│   │   │       ├── "thinking_started"
│   │   │       ├── "thinking_completed"
│   │   │       └── "response_received"
│   │   │
│   │   ├── Span: tool_decision
│   │   │   ├── Attributes:
│   │   │   │   ├── tool.decision: execute_tool
│   │   │   │   └── tool.selected: browser_navigate
│   │   │
│   │   ├── Span: tool_execute_browser_navigate
│   │   │   ├── Attributes:
│   │   │   │   ├── tool.name: browser_navigate
│   │   │   │   ├── tool.input.url: https://gmail.com
│   │   │   │   ├── tool.duration_ms: 1356
│   │   │   │   └── tool.result.status: success
│   │   │   └── Events:
│   │   │       ├── "browser_launch_started"
│   │   │       ├── "page_navigation_started"
│   │   │       ├── "page_load_completed"
│   │   │       └── "accessibility_tree_captured"
│   │   │
│   │   ├── Span: memory_store_result
│   │   │   └── Attributes:
│   │   │       ├── memory.entries_stored: 3
│   │   │       └── memory.embedding_time_ms: 245
│   │   │
│   │   └── Span: response_format
│   │       └── Attributes:
│   │           ├── response.type: text
│   │           └── response.length_chars: 892
│   │
│   └── Span: gateway_respond
│       └── Attributes:
│           ├── http.status_code: 200
│           └── response.size_bytes: 1024
│
└── Span: post_execution_async
    ├── Span: metrics_emit
    └── Span: audit_log_write
```

### 6.2 OpenTelemetry Configuration

```typescript
// src/tracing/opentelemetry.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { diag, DiagConsoleLogger, DiagLogLevel } from '@opentelemetry/api';

// Configure diagnostic logging
if (process.env.OTEL_DEBUG === 'true') {
  diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
}

// Resource attributes for all telemetry
const resource = new Resource({
  [SemanticResourceAttributes.SERVICE_NAME]: 'openclaw-agent',
  [SemanticResourceAttributes.SERVICE_VERSION]: process.env.APP_VERSION || '1.0.0',
  [SemanticResourceAttributes.SERVICE_INSTANCE_ID]: require('os').hostname(),
  [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV || 'development',
  [SemanticResourceAttributes.HOST_NAME]: require('os').hostname(),
  [SemanticResourceAttributes.OS_TYPE]: 'windows',
  [SemanticResourceAttributes.OS_VERSION]: 'Windows 10',
  'agent.framework': 'openclaw',
  'agent.model': 'gpt-5.2',
});

// Trace exporter configuration
const traceExporter = new OTLPTraceExporter({
  url: process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT || 'http://localhost:4318/v1/traces',
  headers: {
    'x-api-key': process.env.OTEL_API_KEY || '',
  },
  timeoutMillis: 30000,
});

// Initialize OpenTelemetry SDK
export const otelSDK = new NodeSDK({
  resource,
  traceExporter,
  spanProcessors: [
    new BatchSpanProcessor(traceExporter, {
      maxQueueSize: 2048,
      maxExportBatchSize: 512,
      scheduledDelayMillis: 5000,
      exportTimeoutMillis: 30000,
    }),
  ],
  metricReader: new PrometheusExporter({
    port: 9090,
    endpoint: '/metrics',
  }),
  instrumentations: [
    getNodeAutoInstrumentations({
      // Enable specific instrumentations
      '@opentelemetry/instrumentation-http': {
        enabled: true,
        requestHook: (span, request) => {
          span.setAttribute('http.request.body.size', request.headers['content-length']);
        },
        responseHook: (span, response) => {
          span.setAttribute('http.response.body.size', response.headers['content-length']);
        },
      },
      '@opentelemetry/instrumentation-fs': { enabled: true },
      '@opentelemetry/instrumentation-net': { enabled: true },
      '@opentelemetry/instrumentation-dns': { enabled: true },
      '@opentelemetry/instrumentation-express': { enabled: true },
      '@opentelemetry/instrumentation-winston': { enabled: true },
    }),
  ],
});

// Start SDK
export function startTelemetry(): void {
  otelSDK.start();
  console.log('OpenTelemetry SDK started');
}

// Graceful shutdown
export async function stopTelemetry(): Promise<void> {
  await otelSDK.shutdown();
  console.log('OpenTelemetry SDK shutdown complete');
}

// Handle process signals for graceful shutdown
process.on('SIGTERM', async () => {
  await stopTelemetry();
  process.exit(0);
});

process.on('SIGINT', async () => {
  await stopTelemetry();
  process.exit(0);
});
```

### 6.3 Custom Tracing Utilities

```typescript
// src/tracing/utils.ts
import { trace, context, SpanStatusCode, SpanKind } from '@opentelemetry/api';
import type { Span, Attributes, Context } from '@opentelemetry/api';

const tracer = trace.getTracer('openclaw-agent');

// Interface for traced function options
interface TraceOptions {
  name: string;
  kind?: SpanKind;
  attributes?: Attributes;
  parentContext?: Context;
}

// Decorator/tracer for async functions
export async function withSpan<T>(
  options: TraceOptions,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  const { name, kind = SpanKind.INTERNAL, attributes = {}, parentContext } = options;
  
  const ctx = parentContext || context.active();
  
  return tracer.startActiveSpan(name, { kind, attributes }, ctx, async (span) => {
    try {
      const result = await fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  });
}

// Tracer for the 15 agentic loops
export async function withLoopSpan<T>(
  loopId: string,
  loopName: string,
  sessionId: string,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  return withSpan(
    {
      name: `loop.${loopId}`,
      kind: SpanKind.INTERNAL,
      attributes: {
        'agent.loop_id': loopId,
        'agent.loop_name': loopName,
        'session.id': sessionId,
      },
    },
    fn
  );
}

// Tracer for tool executions
export async function withToolSpan<T>(
  toolName: string,
  toolCategory: string,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  const startTime = Date.now();
  
  return withSpan(
    {
      name: `tool.${toolName}`,
      kind: SpanKind.INTERNAL,
      attributes: {
        'tool.name': toolName,
        'tool.category': toolCategory,
      },
    },
    async (span) => {
      try {
        const result = await fn(span);
        const duration = Date.now() - startTime;
        span.setAttribute('tool.duration_ms', duration);
        span.setAttribute('tool.status', 'success');
        return result;
      } catch (error) {
        const duration = Date.now() - startTime;
        span.setAttribute('tool.duration_ms', duration);
        span.setAttribute('tool.status', 'failure');
        span.setAttribute('tool.error', error instanceof Error ? error.message : 'Unknown');
        throw error;
      }
    }
  );
}

// Tracer for LLM calls
export async function withLLMSpan<T>(
  provider: string,
  model: string,
  fn: (span: Span) => Promise<{ result: T; tokensIn: number; tokensOut: number; costUsd: number }>
): Promise<T> {
  const startTime = Date.now();
  
  return withSpan(
    {
      name: 'llm.generate',
      kind: SpanKind.INTERNAL,
      attributes: {
        'llm.provider': provider,
        'llm.model': model,
      },
    },
    async (span) => {
      const { result, tokensIn, tokensOut, costUsd } = await fn(span);
      
      const duration = Date.now() - startTime;
      span.setAttribute('llm.latency_ms', duration);
      span.setAttribute('llm.tokens_input', tokensIn);
      span.setAttribute('llm.tokens_output', tokensOut);
      span.setAttribute('llm.tokens_total', tokensIn + tokensOut);
      span.setAttribute('llm.cost_usd', costUsd);
      
      // Add events for key milestones
      span.addEvent('llm.request_sent');
      span.addEvent('llm.response_received');
      
      return result;
    }
  );
}

// Tracer for memory operations
export async function withMemorySpan<T>(
  operation: string,
  memoryType: string,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  const startTime = Date.now();
  
  return withSpan(
    {
      name: `memory.${operation}`,
      kind: SpanKind.INTERNAL,
      attributes: {
        'memory.operation': operation,
        'memory.type': memoryType,
      },
    },
    async (span) => {
      const result = await fn(span);
      const duration = Date.now() - startTime;
      span.setAttribute('memory.duration_ms', duration);
      return result;
    }
  );
}

// Create a child span without making it active
export function createChildSpan(
  parentSpan: Span,
  name: string,
  attributes: Attributes = {}
): Span {
  const ctx = trace.setSpan(context.active(), parentSpan);
  return tracer.startSpan(name, { attributes }, ctx);
}

// Add event to current span
export function addSpanEvent(name: string, attributes?: Attributes): void {
  const span = trace.getActiveSpan();
  if (span) {
    span.addEvent(name, attributes);
  }
}

// Set attributes on current span
export function setSpanAttributes(attributes: Attributes): void {
  const span = trace.getActiveSpan();
  if (span) {
    Object.entries(attributes).forEach(([key, value]) => {
      span.setAttribute(key, value);
    });
  }
}
```

---

## 7. Performance Monitoring

### 7.1 Performance Metrics Collection

```typescript
// src/performance/monitor.ts
import { EventEmitter } from 'events';
import { logger } from '../logging/logger';

interface PerformanceSnapshot {
  timestamp: number;
  memory: NodeJS.MemoryUsage;
  cpu: { user: number; system: number };
  eventLoop: { lag: number; utilization: number };
  gc?: { count: number; totalTime: number };
  handles: { active: number; idle: number };
}

class PerformanceMonitor extends EventEmitter {
  private interval: NodeJS.Timeout | null = null;
  private snapshots: PerformanceSnapshot[] = [];
  private readonly maxSnapshots: number = 1000;
  private lastCpuUsage: NodeJS.CpuUsage;

  constructor(private sampleIntervalMs: number = 5000) {
    super();
    this.lastCpuUsage = process.cpuUsage();
  }

  start(): void {
    if (this.interval) return;
    
    this.interval = setInterval(() => {
      this.collectSnapshot();
    }, this.sampleIntervalMs);
    
    logger.info('Performance monitoring started', {
      sampleIntervalMs: this.sampleIntervalMs,
    });
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      logger.info('Performance monitoring stopped');
    }
  }

  private collectSnapshot(): void {
    const now = Date.now();
    
    // Memory usage
    const memory = process.memoryUsage();
    
    // CPU usage (delta since last sample)
    const currentCpu = process.cpuUsage(this.lastCpuUsage);
    this.lastCpuUsage = process.cpuUsage();
    
    // Event loop lag estimation
    const start = process.hrtime.bigint();
    setImmediate(() => {
      const lag = Number(process.hrtime.bigint() - start) / 1_000_000; // Convert to ms
      
      // Active handles
      const activeHandles = process._getActiveHandles().length;
      const activeRequests = process._getActiveRequests().length;
      
      const snapshot: PerformanceSnapshot = {
        timestamp: now,
        memory,
        cpu: {
          user: currentCpu.user / 1000, // Convert to ms
          system: currentCpu.system / 1000,
        },
        eventLoop: {
          lag,
          utilization: this.calculateEventLoopUtilization(),
        },
        handles: {
          active: activeHandles,
          idle: activeRequests,
        },
      };
      
      this.snapshots.push(snapshot);
      
      // Keep only recent snapshots
      if (this.snapshots.length > this.maxSnapshots) {
        this.snapshots.shift();
      }
      
      // Emit for real-time monitoring
      this.emit('snapshot', snapshot);
      
      // Check thresholds and alert
      this.checkThresholds(snapshot);
    });
  }

  private calculateEventLoopUtilization(): number {
    // Simplified calculation - in production use perf_hooks
    return 0;
  }

  private checkThresholds(snapshot: PerformanceSnapshot): void {
    const memoryPercent = (snapshot.memory.heapUsed / snapshot.memory.heapTotal) * 100;
    
    // Memory threshold
    if (memoryPercent > 90) {
      this.emit('threshold_exceeded', {
        metric: 'memory',
        value: memoryPercent,
        threshold: 90,
        severity: 'critical',
      });
      logger.error('Memory usage exceeded critical threshold', {
        memoryPercent: memoryPercent.toFixed(2),
        heapUsed: snapshot.memory.heapUsed,
        heapTotal: snapshot.memory.heapTotal,
      });
    } else if (memoryPercent > 75) {
      this.emit('threshold_exceeded', {
        metric: 'memory',
        value: memoryPercent,
        threshold: 75,
        severity: 'warning',
      });
      logger.warn('Memory usage exceeded warning threshold', {
        memoryPercent: memoryPercent.toFixed(2),
      });
    }
    
    // Event loop lag threshold
    if (snapshot.eventLoop.lag > 100) {
      this.emit('threshold_exceeded', {
        metric: 'event_loop_lag',
        value: snapshot.eventLoop.lag,
        threshold: 100,
        severity: 'warning',
      });
      logger.warn('Event loop lag exceeded threshold', {
        lag: snapshot.eventLoop.lag.toFixed(2),
      });
    }
  }

  getSnapshots(durationMs?: number): PerformanceSnapshot[] {
    if (!durationMs) return [...this.snapshots];
    
    const cutoff = Date.now() - durationMs;
    return this.snapshots.filter(s => s.timestamp >= cutoff);
  }

  getLatestSnapshot(): PerformanceSnapshot | null {
    return this.snapshots.length > 0 
      ? this.snapshots[this.snapshots.length - 1] 
      : null;
  }

  getAverageMemoryUsage(durationMs: number = 60000): number {
    const snapshots = this.getSnapshots(durationMs);
    if (snapshots.length === 0) return 0;
    
    const total = snapshots.reduce((sum, s) => sum + s.memory.heapUsed, 0);
    return total / snapshots.length;
  }
}

export const performanceMonitor = new PerformanceMonitor();
```

### 7.2 Profiling Integration

```typescript
// src/performance/profiler.ts
import { Session } from 'node:inspector';
import { writeFile } from 'fs/promises';
import { logger } from '../logging/logger';

class Profiler {
  private session: Session | null = null;
  private isProfiling = false;

  async startCPUProfile(): Promise<void> {
    if (this.isProfiling) {
      throw new Error('CPU profile already in progress');
    }
    
    this.session = new Session();
    this.session.connect();
    
    await new Promise<void>((resolve, reject) => {
      this.session!.post('Profiler.enable', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    
    await new Promise<void>((resolve, reject) => {
      this.session!.post('Profiler.start', (err) => {
        if (err) reject(err);
        else {
          this.isProfiling = true;
          logger.info('CPU profiling started');
          resolve();
        }
      });
    });
  }

  async stopCPUProfile(outputPath: string): Promise<void> {
    if (!this.isProfiling || !this.session) {
      throw new Error('No CPU profile in progress');
    }
    
    const profile = await new Promise<any>((resolve, reject) => {
      this.session!.post('Profiler.stop', (err, result) => {
        if (err) reject(err);
        else resolve(result?.profile);
      });
    });
    
    await writeFile(outputPath, JSON.stringify(profile));
    logger.info('CPU profile saved', { outputPath });
    
    this.session.post('Profiler.disable');
    this.session.disconnect();
    this.session = null;
    this.isProfiling = false;
  }

  async captureHeapSnapshot(outputPath: string): Promise<void> {
    const session = new Session();
    session.connect();
    
    const snapshot = await new Promise<string>((resolve, reject) => {
      session.post('HeapProfiler.takeHeapSnapshot', (err) => {
        if (err) reject(err);
      });
      
      let data = '';
      session.on('HeapProfiler.addHeapSnapshotChunk', (m) => {
        data += m.params.chunk;
      });
      
      session.on('HeapProfiler.reportHeapSnapshotProgress', (m) => {
        if (m.params.finished) {
          resolve(data);
        }
      });
    });
    
    await writeFile(outputPath, snapshot);
    logger.info('Heap snapshot saved', { outputPath });
    
    session.disconnect();
  }
}

export const profiler = new Profiler();
```

---

## 8. Health Check Endpoints

### 8.1 Health Check Implementation

```typescript
// src/health/healthcheck.ts
import { Router } from 'express';
import { logger } from '../logging/logger';

interface HealthCheck {
  name: string;
  check: () => Promise<HealthCheckResult>;
  critical: boolean;
  timeout: number;
}

interface HealthCheckResult {
  status: 'healthy' | 'unhealthy' | 'degraded';
  message?: string;
  details?: Record<string, any>;
  responseTime?: number;
}

interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  version: string;
  uptime: number;
  checks: Record<string, HealthCheckResult>;
}

class HealthChecker {
  private checks: Map<string, HealthCheck> = new Map();
  private lastResults: Map<string, HealthCheckResult> = new Map();
  private backgroundInterval: NodeJS.Timeout | null = null;

  register(
    name: string,
    check: () => Promise<HealthCheckResult>,
    options: { critical?: boolean; timeout?: number } = {}
  ): void {
    this.checks.set(name, {
      name,
      check,
      critical: options.critical ?? true,
      timeout: options.timeout ?? 5000,
    });
    logger.info('Health check registered', { name, critical: options.critical });
  }

  unregister(name: string): void {
    this.checks.delete(name);
    this.lastResults.delete(name);
  }

  startBackgroundChecks(intervalMs: number = 30000): void {
    if (this.backgroundInterval) return;
    
    this.backgroundInterval = setInterval(async () => {
      await this.runAllChecks();
    }, intervalMs);
    
    logger.info('Background health checks started', { intervalMs });
  }

  stopBackgroundChecks(): void {
    if (this.backgroundInterval) {
      clearInterval(this.backgroundInterval);
      this.backgroundInterval = null;
      logger.info('Background health checks stopped');
    }
  }

  async runCheck(name: string): Promise<HealthCheckResult> {
    const check = this.checks.get(name);
    if (!check) {
      return { status: 'unhealthy', message: 'Check not found' };
    }

    const startTime = Date.now();
    
    try {
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Health check timeout')), check.timeout);
      });
      
      const result = await Promise.race([check.check(), timeoutPromise]);
      result.responseTime = Date.now() - startTime;
      
      this.lastResults.set(name, result);
      return result;
    } catch (error) {
      const result: HealthCheckResult = {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Unknown error',
        responseTime: Date.now() - startTime,
      };
      this.lastResults.set(name, result);
      return result;
    }
  }

  async runAllChecks(): Promise<HealthStatus> {
    const checkPromises = Array.from(this.checks.keys()).map(async (name) => {
      const result = await this.runCheck(name);
      return { name, result };
    });

    const results = await Promise.all(checkPromises);
    const checks: Record<string, HealthCheckResult> = {};
    
    let overallStatus: 'healthy' | 'unhealthy' | 'degraded' = 'healthy';
    
    for (const { name, result } of results) {
      checks[name] = result;
      
      if (result.status === 'unhealthy') {
        const check = this.checks.get(name);
        if (check?.critical) {
          overallStatus = 'unhealthy';
        } else if (overallStatus === 'healthy') {
          overallStatus = 'degraded';
        }
      } else if (result.status === 'degraded' && overallStatus === 'healthy') {
        overallStatus = 'degraded';
      }
    }

    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      version: process.env.APP_VERSION || '1.0.0',
      uptime: process.uptime(),
      checks,
    };
  }

  getLivenessStatus(): { alive: boolean; timestamp: string } {
    return {
      alive: true,
      timestamp: new Date().toISOString(),
    };
  }

  async getReadinessStatus(): Promise<{ ready: boolean; checks: Record<string, HealthCheckResult> }> {
    const status = await this.runAllChecks();
    return {
      ready: status.status === 'healthy',
      checks: status.checks,
    };
  }
}

export const healthChecker = new HealthChecker();

// ==================== HEALTH CHECK REGISTRATIONS ====================

// Database health check
export function registerDatabaseHealthCheck(pool: any): void {
  healthChecker.register(
    'database',
    async () => {
      const client = await pool.connect();
      try {
        const result = await client.query('SELECT 1 as ok');
        if (result.rows[0].ok === 1) {
          return { status: 'healthy', message: 'Database connection OK' };
        }
        return { status: 'unhealthy', message: 'Database query failed' };
      } finally {
        client.release();
      }
    },
    { critical: true, timeout: 3000 }
  );
}

// Redis health check
export function registerRedisHealthCheck(redis: any): void {
  healthChecker.register(
    'redis',
    async () => {
      const pong = await redis.ping();
      if (pong === 'PONG') {
        return { status: 'healthy', message: 'Redis connection OK' };
      }
      return { status: 'unhealthy', message: 'Redis ping failed' };
    },
    { critical: true, timeout: 2000 }
  );
}

// LLM API health check
export function registerLLMHealthCheck(): void {
  healthChecker.register(
    'llm_api',
    async () => {
      // Lightweight health check - just verify API is reachable
      const response = await fetch('https://api.openai.com/v1/health', {
        method: 'HEAD',
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok || response.status === 404) {
        return { status: 'healthy', message: 'LLM API reachable' };
      }
      return { status: 'degraded', message: 'LLM API returned unexpected status' };
    },
    { critical: false, timeout: 5000 }
  );
}

// Memory system health check
export function registerMemoryHealthCheck(memorySystem: any): void {
  healthChecker.register(
    'memory_system',
    async () => {
      const stats = await memorySystem.getStats();
      if (stats.healthy) {
        return {
          status: 'healthy',
          message: 'Memory system operational',
          details: { entries: stats.entryCount, indexSize: stats.indexSize },
        };
      }
      return { status: 'degraded', message: 'Memory system issues detected' };
    },
    { critical: true, timeout: 3000 }
  );
}

// Disk space health check
export function registerDiskHealthCheck(): void {
  healthChecker.register(
    'disk_space',
    async () => {
      const { statfs } = await import('fs/promises');
      const stats = await statfs('C:\\');
      const freePercent = (stats.bfree / stats.blocks) * 100;
      
      if (freePercent < 5) {
        return {
          status: 'unhealthy',
          message: `Critical disk space: ${freePercent.toFixed(1)}% free`,
          details: { freePercent },
        };
      }
      if (freePercent < 15) {
        return {
          status: 'degraded',
          message: `Low disk space: ${freePercent.toFixed(1)}% free`,
          details: { freePercent },
        };
      }
      return {
        status: 'healthy',
        message: `Disk space OK: ${freePercent.toFixed(1)}% free`,
        details: { freePercent },
      };
    },
    { critical: true, timeout: 1000 }
  );
}

// Cron scheduler health check
export function registerCronHealthCheck(cronScheduler: any): void {
  healthChecker.register(
    'cron_scheduler',
    async () => {
      const stats = cronScheduler.getStats();
      if (stats.healthy) {
        return {
          status: 'healthy',
          message: 'Cron scheduler operational',
          details: { jobsRunning: stats.activeJobs, jobsScheduled: stats.scheduledJobs },
        };
      }
      return { status: 'degraded', message: 'Cron scheduler issues detected' };
    },
    { critical: false, timeout: 2000 }
  );
}

// ==================== EXPRESS ROUTES ====================

export function createHealthRouter(): Router {
  const router = Router();

  // Liveness probe - is the process alive?
  router.get('/live', (req, res) => {
    const status = healthChecker.getLivenessStatus();
    res.status(200).json({
      status: 'alive',
      timestamp: status.timestamp,
    });
  });

  // Readiness probe - can it handle traffic?
  router.get('/ready', async (req, res) => {
    const status = await healthChecker.getReadinessStatus();
    res.status(status.ready ? 200 : 503).json({
      ready: status.ready,
      timestamp: new Date().toISOString(),
      checks: status.checks,
    });
  });

  // Detailed health endpoint
  router.get('/', async (req, res) => {
    const status = await healthChecker.runAllChecks();
    const statusCode = status.status === 'healthy' ? 200 : status.status === 'degraded' ? 200 : 503;
    res.status(statusCode).json(status);
  });

  // Prometheus metrics endpoint
  router.get('/metrics', async (req, res) => {
    // Prometheus metrics are served by the Prometheus exporter
    res.status(404).send('Metrics available on port 9090');
  });

  return router;
}
```

### 8.2 Kubernetes-Style Probe Configuration

```yaml
# kubernetes/health-probes.yaml (for reference if containerized)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openclaw-agent
spec:
  replicas: 1
  template:
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: agent
          image: openclaw-agent:latest
          ports:
            - containerPort: 3000
          
          # Startup probe - for slow-starting apps
          startupProbe:
            httpGet:
              path: /health/ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 30  # 30 * 5 = 150 seconds max startup
          
          # Liveness probe - is the container still running?
          livenessProbe:
            httpGet:
              path: /health/live
              port: 3000
            initialDelaySeconds: 0
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          # Readiness probe - can it handle traffic?
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 3000
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
            successThreshold: 1
```

---

## 9. Dashboard & Visualization

### 9.1 Grafana Dashboard Specifications

#### Dashboard 1: System Overview

```json
{
  "dashboard": {
    "title": "OpenClaw Agent - System Overview",
    "tags": ["openclaw", "system", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{service=\"openclaw-agent\"}",
            "legendFormat": "Agent Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              { "options": { "0": { "text": "DOWN", "color": "red" } }, "type": "value" },
              { "options": { "1": { "text": "UP", "color": "green" } }, "type": "value" }
            ]
          }
        }
      },
      {
        "title": "Uptime",
        "type": "stat",
        "targets": [
          {
            "expr": "system_uptime_seconds{service=\"openclaw-agent\"}",
            "legendFormat": "Uptime"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "dtdurations"
          }
        }
      },
      {
        "title": "CPU Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "system_cpu_usage{service=\"openclaw-agent\"}",
            "legendFormat": "CPU %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        }
      },
      {
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "system_memory_percent{service=\"openclaw-agent\"}",
            "legendFormat": "Memory %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        }
      },
      {
        "title": "Disk Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "system_disk_usage{service=\"openclaw-agent\"}",
            "legendFormat": "Disk %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                { "color": "green", "value": 0 },
                { "color": "yellow", "value": 70 },
                { "color": "red", "value": 85 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

#### Dashboard 2: Agent Performance

```json
{
  "dashboard": {
    "title": "OpenClaw Agent - Agent Performance",
    "tags": ["openclaw", "agent", "performance"],
    "panels": [
      {
        "title": "Active Agent Executions",
        "type": "stat",
        "targets": [
          {
            "expr": "agent_executions_active{service=\"openclaw-agent\"}",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Agent Execution Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(agent_executions_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "Executions/sec"
          }
        ]
      },
      {
        "title": "Agent Execution Duration",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(agent_execution_duration_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(agent_execution_duration_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(agent_execution_duration_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms"
          }
        }
      },
      {
        "title": "Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(agent_errors_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      },
      {
        "title": "Loop Executions by Type",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (loop_id) (agent_loop_executions_total{service=\"openclaw-agent\"})",
            "legendFormat": "{{loop_id}}"
          }
        ]
      }
    ]
  }
}
```

#### Dashboard 3: LLM & Cost Monitoring

```json
{
  "dashboard": {
    "title": "OpenClaw Agent - LLM & Cost Monitoring",
    "tags": ["openclaw", "llm", "cost"],
    "panels": [
      {
        "title": "Token Usage (Last Hour)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(llm_tokens_total{service=\"openclaw-agent\"}[1h]))",
            "legendFormat": "Total Tokens"
          },
          {
            "expr": "sum(increase(llm_tokens_input{service=\"openclaw-agent\"}[1h]))",
            "legendFormat": "Input Tokens"
          },
          {
            "expr": "sum(increase(llm_tokens_output{service=\"openclaw-agent\"}[1h]))",
            "legendFormat": "Output Tokens"
          }
        ]
      },
      {
        "title": "Token Usage Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(llm_tokens_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "Tokens/sec"
          }
        ]
      },
      {
        "title": "LLM Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_latency_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_latency_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p95"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms"
          }
        }
      },
      {
        "title": "Estimated Cost (Today)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(llm_cost_usd{service=\"openclaw-agent\"}[1d]))",
            "legendFormat": "USD"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "title": "Cost Trend",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum by (model) (llm_cost_usd{service=\"openclaw-agent\"})",
            "legendFormat": "{{model}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "title": "Error Rate by Model",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(llm_errors_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "{{model}}"
          }
        ]
      }
    ]
  }
}
```

#### Dashboard 4: Integration Health

```json
{
  "dashboard": {
    "title": "OpenClaw Agent - Integration Health",
    "tags": ["openclaw", "integrations"],
    "panels": [
      {
        "title": "Gmail Operations",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(gmail_operations_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "Ops/sec"
          }
        ]
      },
      {
        "title": "Browser Navigation Duration",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(browser_navigation_duration_bucket{service=\"openclaw-agent\"}[5m]))",
            "legendFormat": "p95"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms"
          }
        }
      },
      {
        "title": "TTS/STT Requests",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(tts_requests_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "TTS"
          },
          {
            "expr": "rate(stt_requests_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "STT"
          }
        ]
      },
      {
        "title": "Twilio Activity",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(twilio_calls_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "Calls"
          },
          {
            "expr": "rate(twilio_sms_total{service=\"openclaw-agent\"}[5m])",
            "legendFormat": "SMS"
          }
        ]
      },
      {
        "title": "Tool Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "(rate(tool_executions_total{service=\"openclaw-agent\",status=\"success\"}[5m]) / rate(tool_executions_total{service=\"openclaw-agent\"}[5m])) * 100",
            "legendFormat": "Success %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                { "color": "red", "value": 0 },
                { "color": "yellow", "value": 95 },
                { "color": "green", "value": 99 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

### 9.2 Jaeger Trace Visualization

```yaml
# jaeger/jaeger-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
data:
  jaeger.yml: |
    collector:
      otlp:
        enabled: true
        grpc:
          port: 4317
        http:
          port: 4318
    query:
      base-path: /jaeger
    ui:
      options:
        menu:
          - label: "OpenClaw Agent"
            items:
              - label: "Dashboard"
                url: "http://grafana:3000"
    storage:
      type: elasticsearch
      elasticsearch:
        server-urls: http://elasticsearch:9200
        index-prefix: openclaw-jaeger
```

---

## 10. Alerting Rules & Thresholds

### 10.1 Prometheus Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: openclaw-agent-critical
    interval: 30s
    rules:
      # Agent Down Alert
      - alert: AgentDown
        expr: up{service="openclaw-agent"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "OpenClaw Agent is down"
          description: "The OpenClaw Agent has been down for more than 1 minute"
          runbook_url: "https://wiki.internal/runbooks/agent-down"

      # High Error Rate
      - alert: HighErrorRate
        expr: rate(agent_errors_total{service="openclaw-agent"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      # Memory Usage Critical
      - alert: MemoryUsageCritical
        expr: system_memory_percent{service="openclaw-agent"} > 90
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Memory usage is critical"
          description: "Memory usage is at {{ $value }}%"

      # Disk Space Critical
      - alert: DiskSpaceCritical
        expr: system_disk_usage{service="openclaw-agent"} > 90
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Disk space is critical"
          description: "Disk usage is at {{ $value }}%"

  - name: openclaw-agent-warning
    interval: 1m
    rules:
      # High CPU Usage
      - alert: HighCPUUsage
        expr: system_cpu_usage{service="openclaw-agent"} > 80
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High CPU usage"
          description: "CPU usage has been above 80% for 5 minutes"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: system_memory_percent{service="openclaw-agent"} > 75
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage"
          description: "Memory usage is at {{ $value }}%"

      # Event Loop Lag
      - alert: EventLoopLag
        expr: histogram_quantile(0.99, rate(nodejs_eventloop_lag_seconds_bucket{service="openclaw-agent"}[5m])) > 0.1
        for: 3m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Event loop lag detected"
          description: "Event loop lag is {{ $value }}s"

      # Slow Agent Execution
      - alert: SlowAgentExecution
        expr: histogram_quantile(0.95, rate(agent_execution_duration_bucket{service="openclaw-agent"}[5m])) > 30000
        for: 5m
        labels:
          severity: warning
          team: ai
        annotations:
          summary: "Slow agent execution detected"
          description: "95th percentile execution time is {{ $value }}ms"

  - name: openclaw-agent-llm
    interval: 1m
    rules:
      # High LLM Error Rate
      - alert: HighLLMErrorRate
        expr: rate(llm_errors_total{service="openclaw-agent"}[5m]) > 0.05
        for: 3m
        labels:
          severity: warning
          team: ai
        annotations:
          summary: "High LLM error rate"
          description: "LLM error rate is {{ $value }} errors per second"

      # High LLM Latency
      - alert: HighLLMLatency
        expr: histogram_quantile(0.95, rate(llm_latency_bucket{service="openclaw-agent"}[5m])) > 10000
        for: 5m
        labels:
          severity: warning
          team: ai
        annotations:
          summary: "High LLM latency"
          description: "95th percentile LLM latency is {{ $value }}ms"

      # Daily Cost Threshold
      - alert: DailyCostThreshold
        expr: sum(increase(llm_cost_usd{service="openclaw-agent"}[1d])) > 100
        for: 0m
        labels:
          severity: warning
          team: finance
        annotations:
          summary: "Daily LLM cost threshold exceeded"
          description: "Daily cost is ${{ $value }}"

  - name: openclaw-agent-integrations
    interval: 1m
    rules:
      # Tool Execution Failure Rate
      - alert: ToolExecutionFailures
        expr: (rate(tool_errors_total{service="openclaw-agent"}[5m]) / rate(tool_executions_total{service="openclaw-agent"}[5m])) > 0.1
        for: 3m
        labels:
          severity: warning
          team: integrations
        annotations:
          summary: "High tool execution failure rate"
          description: "Tool failure rate is {{ $value | humanizePercentage }}"

      # Cron Job Failures
      - alert: CronJobFailures
        expr: rate(cron_errors_total{service="openclaw-agent"}[5m]) > 0
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Cron job failures detected"
          description: "Cron error rate is {{ $value }} errors per second"

      # Heartbeat Missing
      - alert: HeartbeatMissing
        expr: rate(heartbeat_total{service="openclaw-agent"}[5m]) == 0
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Agent heartbeat missing"
          description: "No heartbeat detected for 2 minutes"

  - name: openclaw-agent-security
    interval: 1m
    rules:
      # Authentication Failures
      - alert: AuthenticationFailures
        expr: rate(auth_failures_total{service="openclaw-agent"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
          team: security
        annotations:
          summary: "High authentication failure rate"
          description: "Auth failure rate is {{ $value }} per second"

      # Rate Limit Hits
      - alert: RateLimitHits
        expr: rate(ratelimit_hits_total{service="openclaw-agent"}[5m]) > 10
        for: 1m
        labels:
          severity: warning
          team: security
        annotations:
          summary: "High rate limit hits"
          description: "Rate limit hits are {{ $value }} per second"
```

### 10.2 Alertmanager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@openclaw.local'
  smtp_auth_username: 'alerts@openclaw.local'
  smtp_auth_password: '${SMTP_PASSWORD}'
  slack_api_url: '${SLACK_WEBHOOK_URL}'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'
  resolve_timeout: 5m

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'severity', 'team']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    # Critical alerts go to PagerDuty immediately
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    
    # Platform team alerts
    - match:
        team: platform
      receiver: 'platform-team'
      group_wait: 1m
    
    # AI team alerts
    - match:
        team: ai
      receiver: 'ai-team'
      group_wait: 2m
    
    # Security alerts
    - match:
        team: security
      receiver: 'security-team'
      group_wait: 0s
    
    # Finance alerts
    - match:
        team: finance
      receiver: 'finance-team'

inhibit_rules:
  # If AgentDown fires, suppress other alerts
  - source_match:
      alertname: AgentDown
    target_match_re:
      alertname: .*
    equal: ['service']

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        severity: '{{ .GroupLabels.severity }}'
        description: '{{ .GroupLabels.alertname }}: {{ .GroupLabels.severity }}'
        details:
          summary: '{{ .CommonAnnotations.summary }}'
          description: '{{ .CommonAnnotations.description }}'

  - name: 'platform-team'
    slack_configs:
      - channel: '#platform-alerts'
        title: 'Platform Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
        send_resolved: true
    email_configs:
      - to: 'platform-team@openclaw.local'
        subject: 'Platform Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

  - name: 'ai-team'
    slack_configs:
      - channel: '#ai-alerts'
        title: 'AI Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'security-team'
    slack_configs:
      - channel: '#security-alerts'
        title: 'SECURITY: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SECURITY_KEY}'
        severity: critical

  - name: 'finance-team'
    email_configs:
      - to: 'finance@openclaw.local'
        subject: 'Cost Alert: {{ .GroupLabels.alertname }}'
        body: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### 10.3 Alert Severity Matrix

| Alert | Severity | Response Time | Notification Channel |
|-------|----------|---------------|---------------------|
| AgentDown | Critical | 5 minutes | PagerDuty + Slack + Email |
| HighErrorRate | Critical | 5 minutes | PagerDuty + Slack |
| MemoryUsageCritical | Critical | 5 minutes | PagerDuty + Slack |
| DiskSpaceCritical | Critical | 5 minutes | PagerDuty + Slack |
| HeartbeatMissing | Critical | 5 minutes | PagerDuty + Slack |
| AuthenticationFailures | Critical | 5 minutes | PagerDuty + Security Team |
| HighCPUUsage | Warning | 15 minutes | Slack |
| HighMemoryUsage | Warning | 15 minutes | Slack |
| EventLoopLag | Warning | 15 minutes | Slack |
| SlowAgentExecution | Warning | 15 minutes | Slack (AI Team) |
| HighLLMErrorRate | Warning | 15 minutes | Slack (AI Team) |
| HighLLMLatency | Warning | 15 minutes | Slack (AI Team) |
| DailyCostThreshold | Warning | 1 hour | Email (Finance) |
| ToolExecutionFailures | Warning | 15 minutes | Slack |
| CronJobFailures | Warning | 15 minutes | Slack |
| RateLimitHits | Warning | 10 minutes | Slack |

---

## 11. Implementation Code Samples

### 11.1 Complete Logger Setup

```typescript
// src/logging/index.ts
export { logger, loggers, createComponentLogger, createRequestLogger, createLoopLogger } from './logger';
export { appLogStream, errorLogStream, auditLogStream, startLogCleanupJob } from './rotation';
```

### 11.2 Integration Example: Gmail Client with Full Observability

```typescript
// src/integrations/gmail/client.ts
import { loggers } from '../../logging';
import { withSpan, addSpanEvent } from '../../tracing/utils';
import { gmailOperationsTotal, gmailOperationDuration } from '../../telemetry/metrics';

const logger = loggers.gmail;

export class GmailClient {
  private authenticated = false;

  async authenticate(): Promise<void> {
    return withSpan(
      {
        name: 'gmail.authenticate',
        attributes: { 'integration.name': 'gmail', 'operation.type': 'auth' },
      },
      async (span) => {
        logger.info('Starting Gmail authentication');
        
        try {
          // Authentication logic here
          this.authenticated = true;
          
          span.setAttribute('auth.success', true);
          logger.info('Gmail authentication successful');
        } catch (error) {
          span.setAttribute('auth.success', false);
          span.recordException(error as Error);
          logger.error('Gmail authentication failed', { error });
          throw error;
        }
      }
    );
  }

  async sendEmail(options: {
    to: string;
    subject: string;
    body: string;
    attachments?: string[];
  }): Promise<void> {
    const startTime = Date.now();
    
    return withSpan(
      {
        name: 'gmail.send_email',
        attributes: {
          'integration.name': 'gmail',
          'operation.type': 'send',
          'email.to': options.to,
          'email.subject': options.subject,
          'email.has_attachments': !!options.attachments?.length,
        },
      },
      async (span) => {
        logger.info('Sending email', { to: options.to, subject: options.subject });
        
        try {
          // Email sending logic here
          
          const duration = Date.now() - startTime;
          
          // Record metrics
          gmailOperationsTotal.add(1, { operation: 'send', status: 'success' });
          gmailOperationDuration.record(duration, { operation: 'send' });
          
          span.setAttribute('email.sent', true);
          span.setAttribute('operation.duration_ms', duration);
          
          logger.info('Email sent successfully', { duration });
        } catch (error) {
          gmailOperationsTotal.add(1, { operation: 'send', status: 'failure' });
          
          span.setAttribute('email.sent', false);
          span.recordException(error as Error);
          
          logger.error('Failed to send email', { error, to: options.to });
          throw error;
        }
      }
    );
  }

  async checkInbox(): Promise<any[]> {
    const startTime = Date.now();
    
    return withSpan(
      {
        name: 'gmail.check_inbox',
        attributes: { 'integration.name': 'gmail', 'operation.type': 'read' },
      },
      async (span) => {
        logger.debug('Checking inbox');
        
        try {
          // Inbox checking logic here
          const emails: any[] = []; // Fetch emails
          
          const duration = Date.now() - startTime;
          
          gmailOperationsTotal.add(1, { operation: 'read', status: 'success' });
          gmailOperationDuration.record(duration, { operation: 'read' });
          
          span.setAttribute('emails.found', emails.length);
          span.setAttribute('operation.duration_ms', duration);
          
          logger.info('Inbox checked', { emailsFound: emails.length, duration });
          
          return emails;
        } catch (error) {
          gmailOperationsTotal.add(1, { operation: 'read', status: 'failure' });
          
          span.recordException(error as Error);
          logger.error('Failed to check inbox', { error });
          throw error;
        }
      }
    );
  }
}
```

### 11.3 Agent Loop with Full Observability

```typescript
// src/agents/loops/thinkActLoop.ts
import { loggers } from '../../logging';
import { withLoopSpan, withLLMSpan, withToolSpan, addSpanEvent } from '../../tracing/utils';
import {
  loopExecutionsTotal,
  loopExecutionDuration,
  loopIterationsTotal,
  agentExecutionsActive,
} from '../../telemetry/metrics';

const logger = loggers.agent;

export class ThinkActLoop {
  private readonly loopId = 'loop_01_think_act';
  private readonly loopName = 'Think-Act Loop';

  async execute(sessionId: string, userInput: string): Promise<string> {
    const startTime = Date.now();
    
    return withLoopSpan(
      this.loopId,
      this.loopName,
      sessionId,
      async (span) => {
        logger.info('Starting Think-Act Loop', { sessionId, userInput });
        
        // Track active execution
        agentExecutionsActive.add(1);
        
        try {
          let iteration = 0;
          const maxIterations = 10;
          let shouldContinue = true;
          let finalResponse = '';

          while (shouldContinue && iteration < maxIterations) {
            iteration++;
            loopIterationsTotal.add(1, { loop_id: this.loopId });
            
            addSpanEvent('iteration.started', { iteration });
            logger.debug(`Iteration ${iteration} started`);

            // THINK phase
            const thought = await this.think(sessionId, userInput, iteration);
            
            // DECIDE phase
            const decision = await this.decide(sessionId, thought);
            
            // ACT phase (if needed)
            if (decision.requiresAction) {
              const actionResult = await this.act(sessionId, decision);
              
              // Check if we should continue
              shouldContinue = actionResult.needsAnotherIteration;
              finalResponse = actionResult.response;
            } else {
              shouldContinue = false;
              finalResponse = decision.response;
            }

            addSpanEvent('iteration.completed', { iteration, shouldContinue });
          }

          const duration = Date.now() - startTime;
          
          // Record metrics
          loopExecutionsTotal.add(1, { loop_id: this.loopId, status: 'success' });
          loopExecutionDuration.record(duration, { loop_id: this.loopId });
          
          span.setAttribute('loop.iterations', iteration);
          span.setAttribute('loop.duration_ms', duration);
          
          logger.info('Think-Act Loop completed', {
            sessionId,
            iterations: iteration,
            duration,
          });

          return finalResponse;
        } catch (error) {
          loopExecutionsTotal.add(1, { loop_id: this.loopId, status: 'failure' });
          
          logger.error('Think-Act Loop failed', { sessionId, error });
          throw error;
        } finally {
          agentExecutionsActive.add(-1);
        }
      }
    );
  }

  private async think(sessionId: string, userInput: string, iteration: number): Promise<string> {
    return withLLMSpan('openai', 'gpt-5.2', async (span) => {
      addSpanEvent('think.phase.started');
      
      // LLM call for thinking
      const thought = await this.callLLM({
        systemPrompt: 'You are thinking through a problem...',
        userMessage: userInput,
        thinking: 'extra_high',
      });
      
      addSpanEvent('think.phase.completed', { thought_length: thought.length });
      
      return thought;
    });
  }

  private async decide(sessionId: string, thought: string): Promise<any> {
    return withLLMSpan('openai', 'gpt-5.2', async (span) => {
      addSpanEvent('decide.phase.started');
      
      // LLM call for decision
      const decision = await this.callLLM({
        systemPrompt: 'Based on the thought, decide what to do...',
        userMessage: thought,
      });
      
      addSpanEvent('decide.phase.completed');
      
      return JSON.parse(decision);
    });
  }

  private async act(sessionId: string, decision: any): Promise<any> {
    const toolName = decision.tool;
    const toolInput = decision.toolInput;

    return withToolSpan(toolName, 'agent_action', async (span) => {
      addSpanEvent('act.phase.started', { tool: toolName });
      
      // Execute the tool
      const result = await this.executeTool(toolName, toolInput);
      
      addSpanEvent('act.phase.completed', { tool: toolName, success: result.success });
      
      return result;
    });
  }

  private async callLLM(options: any): Promise<string> {
    // LLM implementation
    return '';
  }

  private async executeTool(toolName: string, toolInput: any): Promise<any> {
    // Tool execution implementation
    return { success: true, response: '' };
  }
}
```

### 11.4 Main Application Setup

```typescript
// src/index.ts
import { startTelemetry, stopTelemetry } from './tracing/opentelemetry';
import { logger, loggers } from './logging';
import { startLogCleanupJob } from './logging/rotation';
import { healthChecker, createHealthRouter } from './health/healthcheck';
import { performanceMonitor } from './performance/monitor';
import express from 'express';

const app = express();

async function main(): Promise<void> {
  try {
    // Start telemetry
    startTelemetry();
    logger.info('OpenClaw Agent starting...');

    // Start log cleanup
    startLogCleanupJob();

    // Start performance monitoring
    performanceMonitor.start();

    // Register health checks
    // (Register your actual health checks here)
    healthChecker.startBackgroundChecks();

    // Setup health endpoints
    app.use('/health', createHealthRouter());

    // Start the main application
    // ... your agent initialization code ...

    // Start HTTP server for health checks and metrics
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      logger.info('Health check server started', { port: PORT });
    });

    // Graceful shutdown
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);

    logger.info('OpenClaw Agent started successfully');
  } catch (error) {
    logger.fatal('Failed to start OpenClaw Agent', { error });
    process.exit(1);
  }
}

async function gracefulShutdown(): Promise<void> {
  logger.info('Shutting down gracefully...');

  // Stop background tasks
  healthChecker.stopBackgroundChecks();
  performanceMonitor.stop();

  // Stop telemetry
  await stopTelemetry();

  logger.info('Shutdown complete');
  process.exit(0);
}

main();
```

---

## 12. Appendix: Configuration Reference

### 12.1 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NODE_ENV` | Environment (development/staging/production) | `development` | No |
| `LOG_LEVEL` | Minimum log level | `info` | No |
| `LOG_DIR` | Log file directory | `./logs` | No |
| `APP_VERSION` | Application version | `1.0.0` | No |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | OTLP trace endpoint | `http://localhost:4318/v1/traces` | No |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | OTLP metrics endpoint | `http://localhost:4318/v1/metrics` | No |
| `OTEL_SERVICE_NAME` | Service name for OTel | `openclaw-agent` | No |
| `OTEL_DEBUG` | Enable OTel debug logging | `false` | No |
| `PROMETHEUS_PORT` | Prometheus metrics port | `9090` | No |
| `HEALTH_CHECK_PORT` | Health check server port | `3000` | No |
| `HEALTH_CHECK_INTERVAL` | Background health check interval (ms) | `30000` | No |

### 12.2 Directory Structure

```
/mnt/okcomputer/output/openclaw-agent/
├── src/
│   ├── logging/
│   │   ├── index.ts
│   │   ├── logger.ts
│   │   └── rotation.ts
│   ├── tracing/
│   │   ├── opentelemetry.ts
│   │   └── utils.ts
│   ├── telemetry/
│   │   └── metrics.ts
│   ├── performance/
│   │   ├── monitor.ts
│   │   └── profiler.ts
│   ├── health/
│   │   └── healthcheck.ts
│   ├── agents/
│   │   └── loops/
│   │       └── thinkActLoop.ts
│   ├── integrations/
│   │   └── gmail/
│   │       └── client.ts
│   └── index.ts
├── logs/
│   ├── app.log
│   ├── error.log
│   ├── audit.log
│   └── archive/
├── config/
│   ├── prometheus/
│   │   └── alerts.yml
│   ├── alertmanager/
│   │   └── alertmanager.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── jaeger/
│       └── jaeger-config.yaml
└── scripts/
    └── log-rotation.ps1
```

### 12.3 Dependencies

```json
{
  "dependencies": {
    "pino": "^9.0.0",
    "pino-pretty": "^11.0.0",
    "rotating-file-stream": "^3.2.0",
    "@opentelemetry/api": "^1.8.0",
    "@opentelemetry/sdk-node": "^0.49.0",
    "@opentelemetry/sdk-trace-base": "^1.22.0",
    "@opentelemetry/sdk-metrics": "^1.22.0",
    "@opentelemetry/exporter-trace-otlp-http": "^0.49.0",
    "@opentelemetry/exporter-metrics-otlp-http": "^0.49.0",
    "@opentelemetry/exporter-prometheus": "^0.49.0",
    "@opentelemetry/auto-instrumentations-node": "^0.41.0",
    "@opentelemetry/resources": "^1.22.0",
    "@opentelemetry/semantic-conventions": "^1.22.0",
    "express": "^4.18.0"
  }
}
```

---

## Document Information

**Author:** AI Systems Architect  
**Version:** 1.0  
**Last Updated:** 2025  
**Status:** Technical Specification

---

*End of Document*
