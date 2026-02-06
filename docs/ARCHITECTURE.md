# AlwaysClaw Architecture

This document describes the end-to-end architecture of the AlwaysClaw runtime,
a Windows 10-focused, always-on personal AI system with OpenClaw-parity
semantics, hardcoded loop orchestration, and full-system access under strict
governance controls.

---

## 1. Runtime Topology

The AlwaysClaw runtime is organized into seven cooperating service layers. Each
layer has a well-defined responsibility boundary, communicates via typed event
envelopes, and can be independently health-checked and restarted.

### Layer 1 -- Control Plane (alwaysclaw-gateway)

The gateway is the single authoritative control plane. All external commands
enter through the gateway, which performs intake normalization, intent
classification, route resolution, session key derivation, and queue admission.

Responsibilities:

- Accept typed RPC commands from channels (Gmail, Twilio, CLI, Web UI).
- Derive session keys using the configured strategy (`main`, `per-peer`,
  `per-channel-peer`, or `per-account-channel-peer`).
- Admit commands into the appropriate queue lane (`main`, `subagent`, or
  `session:<key>`).
- Emit typed event envelopes for every lifecycle transition.
- Maintain the bi-directional event stream for downstream consumers.

### Layer 2 -- Loop Kernel (alwaysclaw-loopd)

The loop kernel is a deterministic state-machine executor. It pulls commands
from the queue, resolves the target loop contract, and drives execution through
the canonical lifecycle states:

    queued -> preflight -> context-build -> reason -> act -> verify ->
    commit -> reflect -> archive

Each loop is defined by a `LoopContract` (see `@alwaysclaw/common`) specifying
entry criteria, thinking policy, allowed tools, approval points, max steps,
success criteria, guardrails, and composition rules.

The kernel supports 105 hardcoded loops across four dependency layers:

- **Strategic:** Ralph, Planning, Meta-Cognition
- **Knowledge:** Research, Discovery, Exploration, Context Engineering,
  Context Prompt Engineering
- **Delivery:** Bug Finder, Debugging, End-to-End
- **Autonomy / Evolution:** Self-Driven, Self-Learning, Self-Updating,
  Self-Upgrading

Plus 90 derived loops covering runtime, integration, security, and operations.

### Layer 3 -- Tool Execution Service (alwaysclaw-tools)

The tool service manages a registry of tools classified into three tiers:

| Tier | Classification | Approval |
|------|---------------|----------|
| Tier 0 | Read-only, no side effects | None |
| Tier 1 | Reversible writes | Loop policy or per-run |
| Tier 2 | Irreversible / high-risk | Always human approval |

The tool eligibility resolver evaluates each invocation against the agent's
role (`ownerTrustedAgent`, `specialistAgent`, `publicFacingAgent`), the loop's
tool budget, and any per-agent policy overrides.

### Layer 4 -- Voice Service (alwaysclaw-voice)

Handles speech-to-text (STT), text-to-speech (TTS), continuous talk loops,
wake-word detection, and barge-in coordination. The voice IO loop runs
independently from the reasoning loop to prevent latency spikes in the
conversation stream.

### Layer 5 -- Messaging Connectors

Channel adapters for external communication:

- **Gmail:** Push notifications via Pub/Sub, history-delta sync, watch
  renewal, and full resync recovery.
- **Twilio SMS:** Signed webhook intake, delivery reconciliation.
- **Twilio Voice:** Call orchestration, TwiML/stream bridge, callback
  reconciliation, transcript redaction.

### Layer 6 -- Browser Control Service

Manages an isolated Playwright/CDP browser profile per high-privilege agent.
Navigation and reads are allowed by default; form submission, purchases,
downloads, and account mutations require explicit approval. Every browser
action produces deterministic verification artifacts (screenshot, DOM digest,
action transcript).

### Layer 7 -- Watchdog and Scheduler Service

Provides always-on operational supervision:

- **Heartbeat:** Default cadence 30 minutes with `HEARTBEAT_OK` suppression.
- **Cron:** Persistent job store (`jobs.json`) with main-session and
  isolated-session (`cron:<jobId>`) execution modes.
- **Watchdog:** Liveness probes every 5 minutes; exponential-backoff restart
  with circuit breaker after 5 consecutive failures within 15 minutes.

---

## 2. Process Model

AlwaysClaw uses a hybrid WSL2 + Windows sidecar process model for maximum
upstream parity on Windows 10.

### WSL2 Core (Ubuntu)

The following services run inside WSL2 to match the Linux-first OpenClaw
runtime semantics:

- Gateway control plane
- Loop kernel
- Cron scheduler
- Memory subsystem
- Hook engine
- Core routing and event bus

WSL2 is configured with `systemd=true` in `/etc/wsl.conf` so services can
be managed via standard systemd unit files.

### Windows Host Sidecars

Privileged Windows-specific actions run as native sidecars:

- PowerShell execution bridge (for host commands)
- Desktop automation (optional UI interactions)
- Windows Task Scheduler integration for boot/recovery

A signed local bridge API connects the WSL2 core to host sidecars. Host
sidecars are treated as privileged tools and always require explicit approval
or policy grants for invocation.

### Boot and Recovery Sequence

1. Windows Task Scheduler triggers WSL2 startup at login.
2. WSL2 core stack starts (gateway, loopd, scheduler, memory).
3. Windows sidecars start (PowerShell bridge, optional desktop automation).
4. Health check runs across all services.
5. Watchdog task runs every 5 minutes to verify liveness.
6. On component failure: restart with exponential backoff.
7. After 5 failures in 15 minutes: enter degraded mode (disable Tier 2 tools,
   keep read-only and communication channels alive, emit incident event).

---

## 3. Storage Model

All persistent state lives under `C:\AlwaysClaw\state\`:

```
C:\AlwaysClaw\state\
  config\
    alwaysclaw.json          # Master configuration
  agents\
    <agentId>\
      workspace\             # Persona context files
        AGENTS.md
        SOUL.md
        TOOLS.md
        IDENTITY.md
        USER.md
        HEARTBEAT.md         # Optional
        BOOTSTRAP.md         # Optional
        MEMORY.md
      sessions\              # Per-session conversation state
  cron\
    jobs.json                # Persistent cron job definitions
  logs\                      # Structured JSON log files
  auth\                      # Credentials, tokens, pairing state
  memory\
    vector\                  # Vector memory index for semantic retrieval
```

File ownership follows the agent workspace pattern: each agent has an isolated
workspace directory containing its persona files, session state, and memory
artifacts. Cross-agent access is prohibited by default.

---

## 4. Persona and Context Stack

Every agent boots with a deterministic context assembly sequence. The runtime
loads persona files in a fixed order, then appends dynamic context slices.

### Boot Sequence (fixed order)

1. `AGENTS.md` -- operating memory and instructions
2. `SOUL.md` -- temperament, boundaries, behavioral anchors
3. `TOOLS.md` -- tool conventions, caveats, tier policies
4. `IDENTITY.md` -- name, style, tone, personality
5. `USER.md` -- owner profile and preferences
6. `HEARTBEAT.md` -- periodic check-in checklist (when present)
7. `BOOTSTRAP.md` -- first-run ritual (when present)

### Dynamic Context (appended after boot)

8. Session history (current conversation state)
9. Memory retrieval results (adaptive + vector search)
10. Tool state (active tool results, pending operations)
11. Loop context (when executing within a loop contract)

The context compiler enforces token budgets and applies compaction when
context pressure rises. Auto-compaction summarizes older history and
flushes pre-compaction memory before retry.

---

## 5. Event Envelope Design and Flow

All inter-service communication uses typed event envelopes defined in
`@alwaysclaw/common` (`EventEnvelope<T>`). This ensures every event carries:

- **eventId** -- UUID v4 for global uniqueness
- **timestamp** -- ISO 8601 for deterministic ordering
- **source** -- producing service identifier
- **type** -- discriminated `EventType` from a closed union of 35 event types
- **sessionKey** -- scoping context for the event
- **correlationId** -- for grouping related events across services
- **loopRunId** -- for linking events to a specific loop execution
- **metadata** -- optional tracing (traceId, spanId, parentSpanId) and priority

### Event Flow

```
[External Channel] --> Gateway (intake, classify, route)
       |
       v
   EventEnvelope<CommandPayload>
       |
       v
   Queue Admission Control --> Lane Assignment
       |
       v
   Loop Kernel (lifecycle state machine)
       |
       +--- EventEnvelope<LoopStateChange> at each transition
       |
       v
   Tool Execution Service
       |
       +--- EventEnvelope<ToolResult> per tool invocation
       |
       v
   Memory Writeback + Verification
       |
       v
   Archive + Structured Report
```

Events support replay and forensics: the full event stream for any command
or loop run can be reconstructed by filtering on `correlationId` or
`loopRunId`.

---

## 6. Queue Architecture

The command queue separates traffic into named lanes to prevent starvation
and enforce per-lane concurrency limits.

### Lane Topology

```
                    +---------------------+
                    |   Queue Admission   |
                    |      Control        |
                    +---------------------+
                       |       |       |
                       v       v       v
                   +------+ +------+ +----------+
                   | main | | sub  | | session: |
                   |  (4) | | agent| |  <key>   |
                   |      | |  (8) | |   (1)    |
                   +------+ +------+ +----------+
                       |       |       |
                       v       v       v
                   +---------------------+
                   |   Loop Kernel       |
                   |   (per-lane         |
                   |    dispatchers)     |
                   +---------------------+
```

### Concurrency Defaults (Source-Locked)

| Lane | Concurrency | Max Queue Depth |
|------|------------|----------------|
| `main` | 4 | 64 |
| `subagent` | 8 | 128 |
| `session:<key>` (default) | 1 | 16 |

### Fairness

Each lane has an independent dispatcher. The `main` lane has higher priority
than `subagent`, ensuring user-initiated commands are never starved by
background sub-agent work. Session lanes are isolated -- one slow session
cannot block another.

### Backpressure

When queue depth exceeds the backpressure threshold (75% of max depth by
default), the system applies one of three strategies:

1. **drop-oldest** -- evict the oldest queued command
2. **reject-new** -- refuse new commands until depth decreases
3. **signal-producer** -- notify the producing service to slow down

The `QueueBackpressurePolicy` type in `@alwaysclaw/common` defines the
strategy, max wait time, and threshold-breach callback.

### Admission Control

Every incoming command passes through the admission controller, which
evaluates lane capacity and returns one of:

- `admit` -- immediately dispatch for execution
- `queue` -- accept into the waiting queue
- `reject` -- refuse due to backpressure or policy

---

## 7. Session Management

Sessions scope conversation context, memory retrieval, and state isolation.
The runtime supports four strategies, selected per-agent:

### Session Strategies

| Strategy | Key Derivation | Use Case |
|----------|---------------|----------|
| `main` | Single shared session | Simple single-user DM |
| `per-peer` | One session per peer | Multi-user DM isolation |
| `per-channel-peer` | (channel, peer) pair | Group chat isolation |
| `per-account-channel-peer` | (account, channel, peer) triple | Multi-account routing |

### DM and Group Behavior

- **DM scope** defaults to `main` (shared primary DM session). Switch to
  `per-peer` when multiple DM users can reach the same agent.
- **Group scope** defaults to `per-channel-peer` for per-group isolation.
- **Mention gating** is enabled by default in group contexts: the agent only
  responds when explicitly @-mentioned, unless the sender is on the exempt list.

### Key Derivation

The `SessionKeyDeriver` function type accepts a `SessionStrategy` and
`SessionKeyParams` (peerId, channelId, accountId) and returns a deterministic
session key string. The resolved key is attached to the `EventEnvelope` as
`sessionKey`.

---

## 8. Memory Architecture

AlwaysClaw implements a dual memory model matching OpenClaw parity requirements.

### Adaptive Memory (Behavioral)

File-based memory stored as Markdown under the agent workspace:

- `MEMORY.md` -- master memory index
- `memory/YYYY-MM-DD.md` -- daily memory logs

Adaptive memory tracks behavioral patterns, preferences, and accumulated
knowledge. It is loaded during context assembly and contributes to the
dynamic context slice.

### Vector Memory (Semantic Retrieval)

Stored under `C:\AlwaysClaw\state\memory\vector\`. Provides semantic search
over the agent's accumulated knowledge. Retrieval results are ranked by
confidence and injected into the context window.

### Compaction

Auto-compaction is enabled by default. When context pressure rises (token
budget approaches limit), the compaction engine:

1. Summarizes older conversation history.
2. Flushes pre-compaction memory to disk.
3. Archives compacted segments for replay-safe recovery.
4. Retries with reduced context if the initial compaction is insufficient.

The `memory.compacted` event is emitted after each compaction cycle.

---

## 9. Loop Kernel Architecture

The loop kernel is the execution engine for all 105 hardcoded loops. Every
loop is defined by a `LoopContract` interface and executed as a deterministic
state machine.

### Lifecycle States

```
  queued --> preflight --> context-build --> reason --> act
                                                        |
  archive <-- reflect <-- commit <-- verify <-----------+
                                       |
                               awaiting-approval
                               (parked if unapproved
                                side effects detected)
```

All 10 states are represented in the `LoopLifecycleState` enum. Each
transition emits an `EventEnvelope` with the corresponding `loop.*` event
type.

### Loop Contract Fields

Every loop contract specifies:

- **Identity:** loopId, version, displayName, ownerAgent
- **Classification:** category (14 types), tier (A/B), dependencyLayer (11 layers)
- **Triggers:** triggerTypes (manual, event, cron, heartbeat, self-driven,
  threshold), triggerClass
- **Constraints:** entryCriteria, hardStops, maxSteps, maxDurationSec
- **Tooling:** allowedTools, forbiddenTools, toolTierBudget
- **Cognition:** thinkingPolicy (default level + escalation rules)
- **Governance:** approvalPoints, approvalRequired, riskTier
- **Recovery:** rollbackPlan, fallbackLoopChain, terminationPolicy
- **Output:** writebacks, outputContract, successCriteria
- **Guardrails:** minConfidenceThreshold, maxToolFailures,
  parkOnUnapprovedSideEffects
- **Composition:** canInvoke, dependsOn, chainsTo, isMacroOrchestrator

### Composition Model

Loops compose in a directed dependency graph:

1. **Ralph Loop** is the macro-orchestrator, invoking strategic sub-loops.
2. **Research + Discovery** produce evidence/hypothesis bundles.
3. **Planning** transforms bundles into executable plans.
4. **Bug Finder / Debugging / End-to-End** execute and validate delivery.
5. **Meta-Cognition** critiques loop strategy and updates future policies.
6. **Self-Learning** writes durable lessons.
7. **Self-Updating / Self-Upgrading** produce gated change proposals only.

### Guardrail Enforcement

1. If confidence falls below threshold, auto-fallback to Research or Discovery.
2. If side effects are requested without approval, park in `awaiting-approval`.
3. If tool failure budget is exceeded, stop and raise an incident packet.
4. Self-* loops can never self-apply changes; they only produce signed proposals.

---

## 10. Service Interaction Topology

```
+------------------------------------------------------------------+
|                        EXTERNAL CHANNELS                          |
|  Gmail Pub/Sub    Twilio Webhooks    CLI    Web UI    Voice/STT   |
+--------+----------------+-------------+------+----------+---------+
         |                |             |      |          |
         v                v             v      v          v
+------------------------------------------------------------------+
|                     GATEWAY (Layer 1)                             |
|  Intake -> Classify -> Route -> Session Key -> Queue Admission   |
+------------------------------------------------------------------+
         |                                         |
         v                                         v
+------------------+                    +---------------------+
| QUEUE LANES      |                    | EVENT BUS           |
| main(4) sub(8)   |------------------->| (typed envelopes)   |
| session:*(1)     |                    +---------------------+
+------------------+                              |
         |                                        v
         v                              +---------------------+
+------------------+                    | WATCHDOG / SCHEDULER|
| LOOP KERNEL      |                    | heartbeat (30m)     |
| (Layer 2)        |                    | cron (jobs.json)    |
| 105 loop FSMs    |                    | watchdog (5m)       |
+------------------+                    +---------------------+
         |
    +----+----+
    |         |
    v         v
+--------+ +--------+
| TOOL   | | VOICE  |
| SERVICE| | SERVICE|
| (L3)   | | (L4)   |
+--------+ +--------+
    |         |
    v         v
+--------+ +--------+
|BROWSER | | STT/   |
|CONTROL | | TTS    |
| (L6)   | | Loops  |
+--------+ +--------+

+------------------------------------------------------------------+
|                       STORAGE LAYER                               |
|  Config | Agent Workspaces | Sessions | Cron | Logs | Auth | Vec |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    MEMORY SUBSYSTEM                                |
|  Adaptive (MEMORY.md)  |  Vector (semantic)  |  Compaction Engine |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    APPROVAL MATRIX                                 |
|  Risk Tiers: safe | moderate | high                               |
|  Policies:  auto-approve | policy-route | manual-always           |
|  Dual Checkpoints: pre-execution + post-verification              |
+------------------------------------------------------------------+
```

### Key Interaction Patterns

**Command Processing Flow:**

1. External channel delivers message to Gateway.
2. Gateway normalizes, classifies intent, resolves route and session key.
3. Queue admission control assigns to lane.
4. Loop kernel picks up command, resolves loop contract, executes FSM.
5. Tool service handles tool invocations within tier budget.
6. Approval matrix gates side effects at checkpoint boundaries.
7. Memory writeback persists outcomes.
8. Event bus records full trace for replay and forensics.

**Heartbeat Flow:**

1. Scheduler emits `heartbeat.tick` every 30 minutes.
2. Gateway evaluates `HEARTBEAT.md` checklist.
3. If all checks pass, emit `heartbeat.ok` (may be suppressed).
4. If checks fail, escalate to incident containment loop.

**Cron Execution Flow:**

1. Scheduler fires job at scheduled time.
2. Job runs in `main` session (context-aware) or `cron:<jobId>` (isolated).
3. Loop kernel executes the associated loop contract.
4. Results written back; delivery mode controls external announcement.

**Gmail Push Flow:**

1. Google Pub/Sub sends push notification.
2. JWT validated, notification ACKed immediately.
3. `Gmail History Delta Sync` loop runs `history.list` from last `historyId`.
4. New messages enter the standard intake pipeline.
5. If `historyId` is stale (404), full resync path triggers.

**Twilio Voice Flow:**

1. Inbound/outbound call event enters Voice Call Orchestration loop.
2. `X-Twilio-Signature` validated on webhook.
3. STT/TTS loops handle real-time conversation.
4. Callback reconciliation finalizes call state.
5. Transcript redaction runs before memory writeback.

---

## Appendix A: Type System Reference

All shared types are defined in `packages/common/src/types/` and exported
from `@alwaysclaw/common`:

| Module | Key Exports |
|--------|-------------|
| `event-envelope.ts` | `EventEnvelope<T>`, `EventType`, `EventPriority`, `EventMetadata`, `createEventEnvelope()`, `isValidEventEnvelope()` |
| `queue-lanes.ts` | `QueueLaneName`, `QueueLaneConfig`, `DEFAULT_LANE_CONFIGS`, `QueueAdmissionDecision`, `QueueBackpressurePolicy`, `resolveLaneName()` |
| `session-keys.ts` | `SessionStrategy`, `SessionKeyParams`, `SessionConfig`, `DEFAULT_SESSION_CONFIG`, `SessionKeyDeriver` |
| `tool-tiers.ts` | `ToolTier`, `ToolRegistryEntry`, `ToolPolicy`, `ToolEligibilityResult`, `AgentRole` |
| `loop-contract.ts` | `LoopContract`, `LoopCategory`, `LoopTier`, `DependencyLayer`, `TriggerType`, `TriggerClass`, `LoopLifecycleState`, `ThinkingLevel`, `ThinkingPolicy`, `TerminationPolicy`, `RiskTier`, `ToolTierBudget`, `isValidLoopContract()` |
| `approval-matrix.ts` | `ApprovalDecision`, `ApprovalRequest`, `ApprovalPolicy`, `EscalationPath` |

## Appendix B: Configuration Reference

Master configuration file: `C:\AlwaysClaw\state\config\alwaysclaw.json`

Key configuration sections:

- **gateway:** port, auth mode, event bus settings
- **queue:** lane configs, backpressure policies, admission thresholds
- **sessions:** default strategy, DM scope, group scope, mention gating
- **loops:** loop registry path, contract directory, kernel settings
- **tools:** registry, tier policies, per-agent overrides
- **memory:** adaptive settings, vector index path, compaction thresholds
- **scheduler:** heartbeat cadence, cron job store path, quiet hours
- **integrations:** Gmail, Twilio, browser profile settings
- **security:** pairing mode, allowlists, sandbox profiles, secret paths
- **operations:** watchdog interval, restart policy, degraded mode thresholds
