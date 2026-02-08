# OpenClawAgent - Windows 10 AI Agent Service

A 24/7 background AI agent service for Windows 10, inspired by OpenClaw, featuring GPT-5.2 integration, Gmail, browser automation, Twilio voice/SMS, TTS/STT, and 15 hardcoded agentic loops.

## Features

- **24/7 Background Service**: Runs as a Windows service with automatic restart
- **GPT-5.2 Integration**: Extra high thinking mode for complex reasoning
- **Multi-Worker Architecture**: Clustered workers for agent loops, I/O, and tasks
- **15 Agentic Loops**: Ralph, Research, Discovery, Bug Finder, Debugging, End-to-End, Meta-Cognition, Exploration, Self-Driven, Self-Learning, Self-Updating, Self-Upgrading, Planning, Context Engineering, Context Prompt Engineering
- **Integrations**: Gmail, Browser (Playwright/Puppeteer), Twilio (Voice/SMS), TTS, STT
- **HTTP Control Plane**: REST API for health checks, worker management, and control
- **Cron Scheduling**: Automated task scheduling and periodic jobs

## Quick Start

### Prerequisites

- Windows 10
- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Clone or extract the project
cd openclaw-agent

# Install dependencies
npm install

# Copy environment configuration
copy .env.example .env
# Edit .env with your actual API keys

# Create logs directory (auto-created on first run)
mkdir logs
```

### Running in Development Mode

```bash
# Start the service
npm start

# Or with auto-reload
npm run dev
```

### Installing as Windows Service

```bash
# Install the service
npm run service:install

# Start the service
npm run service:start

# Check status
npm run service:status

# Stop the service
npm run service:stop

# Uninstall
npm run service:uninstall
```

## HTTP Control Plane API

Once running, the control server is available at `http://localhost:8080`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Full service status |
| `/workers` | GET | List all workers |
| `/metrics` | GET | Service metrics |
| `/control/restart-worker` | POST | Restart a specific worker |
| `/control/restart-all` | POST | Restart all workers |
| `/control/shutdown` | POST | Graceful shutdown |

### Example API Calls

```bash
# Health check
curl http://localhost:8080/health

# Get workers
curl http://localhost:8080/workers

# Restart a worker
curl -X POST http://localhost:8080/control/restart-worker \
  -H "Content-Type: application/json" \
  -d '{"workerType": "agent-loop", "index": 0}'

# Graceful shutdown
curl -X POST http://localhost:8080/control/shutdown
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClawAgent Service                    │
├─────────────────────────────────────────────────────────────┤
│  service.js (Entry Point)                                   │
│  ├─ Master Process (cluster.isMaster)                      │
│  │  └─ DaemonMaster                                         │
│  │     ├─ Control Server (HTTP API)                        │
│  │     ├─ Cron Scheduler                                    │
│  │     ├─ Health Monitor                                    │
│  │     └─ Worker Management                                 │
│  │        ├─ 15 Agent Loop Workers                         │
│  │        ├─ 5 I/O Workers (gmail, browser, twilio, tts, stt)
│  │        └─ N Task Workers                                 │
│  └─ Worker Processes                                        │
│     ├─ AgentLoopWorker                                      │
│     ├─ IOWorker                                             │
│     └─ TaskWorker                                           │
└─────────────────────────────────────────────────────────────┘
```

## Worker Types

### Agent Loop Workers (15)
Each agent loop worker runs one of the 15 hardcoded loops:
1. **Ralph Loop**: Background processing and monitoring
2. **Research Loop**: Autonomous information gathering
3. **Discovery Loop**: Territory mapping and exploration
4. **Bug Finder Loop**: Error and anomaly detection
5. **Debugging Loop**: Troubleshooting and fix generation
6. **End-to-End Loop**: Full workflow automation
7. **Meta-Cognition Loop**: Self-reflective thinking
8. **Exploration Loop**: Systematic investigation
9. **Self-Driven Loop**: Intrinsic motivation
10. **Self-Learning Loop**: Knowledge acquisition
11. **Self-Updating Loop**: Code and config updates
12. **Self-Upgrading Loop**: Capability enhancement
13. **Planning Loop**: Task planning and strategy
14. **Context Engineering Loop**: Context optimization
15. **Context Prompt Engineering Loop**: Prompt refinement

### I/O Workers (5)
- **Gmail**: Email send/read/search/watch
- **Browser**: Navigation, clicking, typing, screenshots
- **Twilio**: Voice calls and SMS
- **TTS**: Text-to-speech
- **STT**: Speech-to-text

### Task Workers (N)
Background job processing for:
- Email processing
- Data sync
- Report generation
- Cleanup
- Backup
- Notifications
- API calls
- File processing
- AI processing

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `GMAIL_CLIENT_ID` | Google OAuth client ID | For Gmail |
| `GMAIL_CLIENT_SECRET` | Google OAuth secret | For Gmail |
| `TWILIO_ACCOUNT_SID` | Twilio account SID | For Twilio |
| `TWILIO_AUTH_TOKEN` | Twilio auth token | For Twilio |
| `AZURE_SPEECH_KEY` | Azure Speech key | For TTS/STT |
| `CONTROL_PORT` | HTTP API port (default: 8080) | No |

## Logging

Logs are written to the `logs/` directory:
- `application-YYYY-MM-DD.log`: All logs
- `error-YYYY-MM-DD.log`: Error logs only

View logs:
```bash
# Follow logs
npm run logs

# Follow error logs
npm run logs:error
```

## Troubleshooting

### Service won't start
1. Check logs: `logs/error-*.log`
2. Verify `.env` configuration
3. Ensure Node.js 18+ is installed
4. Check port 8080 is not in use

### Workers keep restarting
1. Check restart rate limits in logs
2. Verify worker environment variables
3. Check for uncaught exceptions

### Control server not responding
1. Verify `CONTROL_PORT` in `.env`
2. Check firewall settings
3. Test with: `curl http://localhost:8080/health`

## Development

### Project Structure
```
openclaw-agent/
├── service.js              # Main entry point
├── daemon-master.js        # Master process management
├── control-server.js       # HTTP control plane
├── worker-base.js          # Base worker class
├── agent-loop-worker.js    # Agent loop worker
├── io-worker.js            # I/O worker
├── task-worker.js          # Task worker
├── cron-scheduler.js       # Cron job scheduler
├── health-monitor.js       # Health monitoring
├── logger.js               # Winston logger
├── service-install.js      # Windows service installer
├── lifecycle-manager.js    # Service lifecycle
├── package.json            # Dependencies
├── .env.example            # Environment template
└── logs/                   # Log files
```

### Adding New Capabilities

1. **New Agent Loop**: Extend `AgentLoopWorker` and add loop logic
2. **New I/O Service**: Add service class in `io-worker.js`
3. **New Task Handler**: Register handler in `TaskWorker`

## Security Considerations

- Never commit `.env` with real credentials
- Use Windows Credential Manager for sensitive data
- Run service with least privilege
- Enable encryption for state files
- Review all skill code before execution

## License

MIT

## Credits

Inspired by OpenClaw - the viral AI agent framework.
