# AlwaysClaw Security Model

Version: 1.0.0
Last Updated: 2026-02-06

This document defines the security architecture for AlwaysClaw, a 24/7 always-on AI assistant running on Windows 10 with full system access. Security is not optional in this context -- it is a foundational constraint that gates every capability the system exposes.

The core principle: **full access is a capability, not a default behavior.** Every action must pass through identity, scope, tool, and runtime gates before execution.

---

## 1. Identity and Channel Gates

### 1.1 Device Pairing

AlwaysClaw uses a pairing-first trust model. No unrecognized sender can issue commands or trigger loops. The pairing flow works as follows:

1. An unknown sender contacts the system through any channel (Gmail, Twilio SMS, group chat).
2. The `auth-pairing-loop-v1` generates a time-limited challenge token.
3. The sender must complete the challenge via a pre-authenticated side channel (e.g., physical presence, pre-shared secret, or owner confirmation).
4. On successful pairing, a trust record is created and stored in `C:\AlwaysClaw\state\auth\pairings.json`.
5. The trust record includes: sender identity, channel, trust level (owner, trusted, limited), expiry policy, and pairing timestamp.

Unpaired senders receive no response. The system does not acknowledge their existence to avoid information leakage.

### 1.2 Mention-Gating in Group Contexts

In group channels (Telegram groups, Slack channels, shared email threads), the agent applies mention-gating by default:

- The agent only processes messages that explicitly mention it by name or handle.
- Allowlisted senders may bypass mention-gating via the `config.channelAllowlist` setting.
- All group messages that do not trigger mention-gating are silently ignored -- no logging of message content, only a counter increment for observability.

This prevents the agent from being manipulated by ambient group conversation or injected prompts in group contexts.

### 1.3 Owner vs Non-Owner Policies

The system distinguishes two primary trust levels:

- **Owner**: Full tool access across all tiers (tier-0, tier-1, tier-2), with approval gates on tier-2 destructive actions. The owner can override most policies but cannot disable audit logging.
- **Non-Owner (Trusted)**: Access to tier-0 (read) and tier-1 (reversible write) tools only. Tier-2 tools are blocked entirely. Elevation requests route to the owner for approval.
- **Non-Owner (Limited)**: Access to tier-0 tools only. Cannot trigger loops or write to memory.

Trust level is determined at pairing time and stored per sender/channel combination. Trust cannot be elevated by the agent itself -- only by the owner through an explicit `auth.elevate` action.

---

## 2. Tool Tier Enforcement Model

All tools in AlwaysClaw are categorized into three tiers based on reversibility and blast radius.

### 2.1 Tier-0: Read-Only

Tools that read state without modifying it. Examples: `memory.search`, `config.read`, `health.probe`, `browser.screenshot`, `gmail.read`, `logs.read`.

- **Policy**: Always allowed for all agent roles.
- **Approval**: None required.
- **Audit**: Standard logging (tool name, timestamp, agent, result summary).
- **Rate limit**: 120 calls/minute.

### 2.2 Tier-1: Reversible Write

Tools that modify state in recoverable ways. Examples: `memory.write`, `tasks.update`, `browser.navigate`, `gmail.label`, `twilio.sms.send`, `config.patch`.

- **Policy**: Allowed within approved loops. Specialist agents are path-restricted. Public-facing agents have specific tool exclusions.
- **Approval**: Not required for pre-approved loop execution paths.
- **Audit**: Detailed logging (full parameters, before/after state diff where applicable).
- **Rate limit**: 60 calls/minute.

### 2.3 Tier-2: Irreversible

Tools with high blast radius or irreversible effects. Examples: `gmail.send`, `gmail.delete`, `host.exec`, `host.install`, `browser.purchase`, `secrets.rotate`, `auth.pair`, `loops.forceTerminate`.

- **Policy**: Denied by default. Requires explicit human approval per invocation.
- **Approval**: Always required. Dual checkpoint: approval before execution and after verification before final commit.
- **Audit**: Forensic logging (full parameters, execution trace, caller chain, approval identity, post-execution verification result).
- **Rate limit**: 10 calls/minute.
- **Unattended execution**: Strictly prohibited. Tier-2 tools cannot execute in any unattended or autonomous mode.

### 2.4 Agent Role Overrides

Three agent role profiles modify the default tier policies:

- **ownerTrustedAgent**: Full tier-0/1 access. Tier-2 allowed but approval-gated and never unattended.
- **specialistAgent**: Full tier-0 access. Tier-1 access is path-restricted to the agent's designated workspace and allowlisted paths. Tier-2 denied by default but elevation is available via owner approval.
- **publicFacingAgent**: Full tier-0 access. Tier-1 access excludes browser interaction, SMS sending, and hook dispatch. Tier-2 entirely blocked with no elevation path.

---

## 3. Sandbox Policy

Each agent runs within a sandbox profile that constrains its filesystem, network, and process access.

### 3.1 Strict Profile

For public-facing agents and untrusted integrations:

- Filesystem: Read-only access to agent workspace and shared config. No write access outside designated output directories.
- Network: No outbound connections except to pre-approved API endpoints.
- Process: Cannot spawn child processes or execute host commands.
- Tool tier: Tier-0 only; tier-1 requires explicit per-tool approval.

### 3.2 Normal Profile

For specialist agents with bounded responsibilities:

- Filesystem: Read/write access to own workspace and shared state directories within `C:\AlwaysClaw\state\`.
- Network: Outbound to configured integration endpoints (Gmail API, Twilio API, etc.).
- Process: Can execute pre-approved commands from the `commandReviewRequired` list with logging.
- Tool tier: Tier-0 and tier-1 with path restrictions.

### 3.3 Trusted Profile

For the owner's primary agent only:

- Filesystem: Full access to `C:\AlwaysClaw\` and owner-configured project roots. Blocked paths (credential stores, system registry, startup persistence) are still enforced.
- Network: Full outbound access.
- Process: Can execute host commands via `host.exec` with approval gate and audit trail.
- Tool tier: All tiers with approval gates on tier-2.

---

## 4. Elevated Execution Policy

Host command execution (`host.exec`, `host.install`, `host.serviceModify`, `host.registryModify`, `host.networkExpose`) represents the highest-risk capability in the system.

### 4.1 Approval Gates

Every host command execution follows this flow:

1. The requesting loop or agent submits the command with full arguments and justification.
2. The `tool-policy-enforcer-loop-v1` validates the command against the `commandReviewRequired` list and path allowlists.
3. If the command matches a blocked pattern (registry modification, service creation, firewall changes, package installs), it is held for explicit owner approval.
4. The owner receives a notification (DM or SMS) with the exact command, requesting agent, justification, and risk classification.
5. On approval, the command executes with full forensic logging.
6. On rejection, the requesting loop receives a denial and must adjust its strategy.

### 4.2 Command Audit Trail

Every executed host command produces an immutable audit record containing:

- Timestamp (UTC)
- Requesting agent ID and loop ID
- Exact command and arguments
- Approval identity (who approved)
- Exit code and truncated stdout/stderr
- Duration
- Correlation ID linking to the parent loop run

Audit records are append-only and stored in `C:\AlwaysClaw\state\logs\audit\commands\`. They cannot be modified or deleted by any agent, including the owner agent. Only manual operator access to the filesystem can manage audit log retention.

---

## 5. Secret Management

### 5.1 Scoping Per Integration

Secrets are scoped per integration, not globally shared:

- Gmail OAuth tokens are stored under `C:\AlwaysClaw\state\auth\gmail\` and are accessible only to Gmail integration loops.
- Twilio API credentials are stored under `C:\AlwaysClaw\state\auth\twilio\` and are accessible only to Twilio integration loops.
- Browser session cookies are stored in the managed browser profile and are not accessible to non-browser loops.
- Each agent has its own auth store under `C:\AlwaysClaw\state\agents\{agentId}\auth\`.

No integration can read another integration's secrets. Cross-integration secret access is a security violation that triggers the `incident-detector-loop-v1`.

### 5.2 Redaction

The `secrets-scanner-loop-v1` continuously scans:

- All loop output artifacts before they are written to memory or logs.
- All outbound messages before they are dispatched to external channels.
- All prompt context before it is sent to the model API.

Detected secrets (API keys, tokens, passwords, connection strings) are replaced with `[REDACTED:{type}]` tokens. The original values are never logged in plaintext. Detection uses pattern matching for common secret formats plus entropy analysis for high-entropy strings.

### 5.3 Vault-Backed Storage

All secrets at rest are stored using Windows DPAPI encryption scoped to the current user. The encryption flow:

1. Secret value is encrypted using `CryptProtectData` with the current user's Windows credential.
2. The encrypted blob is stored on disk with file permissions restricted to the current user.
3. At runtime, secrets are decrypted into memory only for the duration of the API call and are zeroed immediately after use.

The `secrets-rotator-loop-v1` manages scheduled and emergency rotation. Rotation is a tier-2 (high-risk) operation requiring explicit approval and producing an immutable audit record of the rotation event.

---

## 6. Incident Response Procedures

### 6.1 Kill Switch

AlwaysClaw provides a one-command kill switch that:

1. Immediately terminates all running loops.
2. Disconnects all external integrations (Gmail watch, Twilio webhooks, browser sessions).
3. Enters safe mode: only tier-0 tools remain available, only the owner DM channel is active.
4. Emits an incident record with full system state snapshot.

The kill switch can be triggered by:

- The owner via DM command (`/emergency-stop`).
- The owner via SMS command (predefined keyword).
- The `incident-containment-loop-v1` when automated threat detection exceeds severity thresholds.
- The watchdog when repeated crash loops exceed the safe restart limit (5 restarts in 15 minutes).

### 6.2 Secret Rotation

When a secret compromise is suspected:

1. The `incident-containment-loop-v1` immediately revokes the compromised credential.
2. The `secrets-rotator-loop-v1` generates a new credential and updates the vault.
3. All dependent integrations are restarted with the new credential.
4. The old credential is logged as revoked in the audit trail.
5. Target completion time: under 15 minutes from detection to full rotation.

### 6.3 Forensics Package

On any security incident, the system automatically assembles a forensics package:

- Command execution history (last 24 hours)
- Loop state machine snapshots for all active loops
- Channel event trace (last 1000 events)
- Authentication events (last 48 hours)
- Memory access log (last 24 hours)
- Network request log (last 24 hours)
- System resource utilization at time of incident

The forensics package is written to `C:\AlwaysClaw\state\logs\incidents\{incidentId}\` and is immutable once created.

---

## 7. Browser Risk Model

The browser subsystem presents unique security challenges because it processes external web content that may contain adversarial prompt injections.

### 7.1 Prompt Injection Defense

The `browser-prompt-injection-scanner-loop-v1` runs after every page load and before any browser action:

1. Extracts visible text, hidden text (CSS hidden elements, zero-opacity text, tiny font), and meta/attribute content from the page.
2. Scans for patterns matching known prompt injection templates: role override attempts, instruction injection, authority impersonation, urgency manipulation.
3. If injection patterns are detected, the browser action is halted and the finding is reported to the owner for review.
4. High-confidence injections trigger an automatic abort of the current browser workflow.

### 7.2 Isolated Profiles

Each trust-tier agent that needs browser access gets its own isolated browser profile:

- Separate cookie stores, local storage, and session state.
- No cookie sharing between profiles.
- The owner's primary agent uses a `trusted` profile with stored credentials (managed via the browser auth vault).
- Specialist agents use `normal` profiles with no stored credentials.
- Public-facing agents use `strict` profiles that are read-only (screenshot and DOM read only).

### 7.3 Artifact Verification

Every browser action produces a verification artifact bundle:

- Screenshot (before and after action)
- DOM digest (structural hash of the page)
- Action transcript (exact sequence of browser commands executed)
- Network request log (all requests made during the action)

The `browser-artifact-verification-loop-v1` compares expected vs actual outcomes and flags discrepancies. This enables post-hoc audit of what the browser actually did versus what was requested.

---

## 8. Self-* Loop Constraints

Loops in the autonomy/evolution category (`self-driven`, `self-learning`, `self-updating`, `self-upgrading`) have strict additional constraints to prevent runaway self-modification.

### 8.1 Proposal-Only Execution

- `self-updating-loop-v1` can draft config and prompt patches but cannot apply them. It produces a signed change proposal that enters the `approval-workflow-manager-loop-v1` queue.
- `self-upgrading-loop-v1` can draft dependency and runtime upgrade plans but cannot execute them. Proposals must include a rollback artifact and a smoke/e2e test plan.
- Neither loop can bypass the approval queue under any circumstances, including claimed urgency or security justifications.

### 8.2 Immutable Audit Records

Every self-* loop action produces an immutable audit record containing:

- The proposed diff (exact changes)
- Risk classification (safe/moderate/high)
- Approver identity (who approved, or "pending" if not yet approved)
- Post-change verification result (pass/fail with evidence)
- Rollback plan reference

These records are append-only and stored alongside the general audit trail. The `self-modification-gatekeeper-loop-v1` enforces that no self-modification can occur without a corresponding audit record.

### 8.3 Self-Driven Task Limits

The `self-driven-loop-v1` can propose autonomous tasks from the backlog and telemetry but:

- Cannot execute any tier-2 side effects without routing through the approval queue.
- Cannot modify its own loop policy or tool scope.
- Cannot elevate its own trust level or sandbox profile.
- Is rate-limited to a configurable maximum of autonomous task proposals per heartbeat window.

---

## 9. Webhook Authentication

### 9.1 Gmail JWT Validation

All inbound Gmail Pub/Sub push notifications must pass JWT validation:

1. Extract the bearer token from the push request's `Authorization` header.
2. Verify the JWT signature against Google's published public keys.
3. Validate the `iss` (issuer), `aud` (audience), and `email` (service account) claims against configured expectations.
4. Reject and log any request that fails validation as a high-severity security event.

Unauthenticated Gmail push callbacks are never processed. The `gmail-notification-intake-loop-v1` performs this validation as its first step before any message content is examined.

### 9.2 Twilio Signature Validation

All inbound Twilio webhooks (SMS, voice, status callbacks) must pass signature validation:

1. Compute the expected signature using the Twilio auth token and the full request URL plus POST parameters.
2. Compare against the `X-Twilio-Signature` header value.
3. For WebSocket upgrade flows, the signature header name may be lowercase (`x-twilio-signature`) -- both forms must be supported.
4. Use the official Twilio SDK validator rather than custom cryptographic implementations to avoid subtle verification bugs.

Invalid or missing signatures are treated as high-severity security events. The `twilio-webhook-verification-loop-v1` handles this validation and emits an incident event on any failure.

---

## 10. Operational Security Cadence

Security is not a one-time setup. AlwaysClaw maintains continuous security operations:

- **Every 5 minutes**: Health probes, watchdog checks, process liveness verification.
- **Every 30 minutes**: Heartbeat synthesis including security posture check.
- **Daily**: Secret expiry checks, Gmail watch renewal health verification, log compaction with redaction verification.
- **Weekly**: Full security audit loop, dependency vulnerability scan, chaos drill (when enabled), policy drift review.
- **Monthly**: Disaster recovery simulation, full backup restore validation, architecture drift review.

### Degraded Mode

When the system enters degraded mode (after exceeding restart thresholds or on explicit operator action):

1. All tier-2 tools are disabled.
2. Only read-only and communication channels remain active.
3. An incident event is emitted requiring human acknowledgment before re-elevation.
4. The system cannot self-exit degraded mode -- only operator action can restore full capability.

This ensures that even in failure states, the system cannot escalate its own privileges or take destructive actions without human oversight.
