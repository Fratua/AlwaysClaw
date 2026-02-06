# AlwaysClaw Incident Response Runbook

Version: 1.0.0
Generated from: OPENCLAW_WIN10_GPT52_MASTERPLAN.md sections 8, 17, 28, 29, 35, 44
Date: 2026-02-06

---

## 1. Purpose

This runbook defines the incident response procedures for the AlwaysClaw system. It covers severity classification, containment procedures, kill switch operations, secret rotation, forensics collection, and post-incident review. All operators and on-call personnel must be familiar with this document before handling production incidents.

---

## 2. Severity Classification

### P1 - Critical

**Definition:** Complete system outage or confirmed security breach. All users affected. No workaround available.

**Response time:** Immediate (within 5 minutes of detection).

**Examples:**
- Gateway control plane is down and unrecoverable after watchdog restart attempts.
- WSL2 distribution cannot start, rendering all Linux-hosted services unavailable.
- Confirmed credential compromise or unauthorized access to the system.
- All critical services (gateway, tools, scheduler, memory) simultaneously unreachable.
- Webhook signature validation bypass detected in production.
- Error budget fully exhausted on command-success-rate SLO.

**Required actions:**
- Page operator immediately via Gmail notification and dashboard alert.
- Enter degraded mode automatically if triggered by watchdog.
- Disable all tier-2 tools and autonomous loops.
- Begin forensics collection within 10 minutes.
- Rotate any potentially compromised credentials within 15 minutes.
- Provide status update every 30 minutes until resolved.

### P2 - High

**Definition:** Major feature degradation. Significant user impact but partial workaround exists.

**Response time:** Within 30 minutes.

**Examples:**
- Loop kernel (loopd) repeatedly crashing; automated tasks halted but manual interaction works.
- Gmail watch has expired and push notifications are not being received.
- Twilio webhook validation is failing, blocking inbound SMS/calls.
- Memory service down; responses lack historical context.
- SLO error budget at warning threshold (25% remaining).
- Single integration kill-switched due to repeated failures.

**Required actions:**
- Alert operator via Gmail and dashboard.
- Attempt automated recovery (restart affected component).
- If automated recovery fails within 15 minutes, escalate to P1.
- Document root cause hypothesis within 1 hour.

### P3 - Moderate

**Definition:** Minor feature degradation. Limited user impact. Workaround available.

**Response time:** Within 4 hours (next business window if during quiet hours).

**Examples:**
- Voice service down; text channels fully operational.
- Browser control sidecar unreachable; non-browser tasks unaffected.
- Latency regression detected exceeding 2x baseline for standard loops.
- Single cron job missed execution window.
- Non-critical sidecar (desktop automation) restart loop.

**Required actions:**
- Log incident and create investigation ticket.
- Attempt automated recovery.
- Review during next daily planning loop if not self-resolved.

### P4 - Low

**Definition:** Cosmetic issue or minor operational anomaly. No user-visible impact.

**Response time:** Within 1 week.

**Examples:**
- Dashboard panel displaying stale data.
- Log compaction ran over expected duration but completed.
- Metrics aggregation delayed by one cycle.
- Non-critical loop emitted a warning but completed successfully.

**Required actions:**
- Log for review.
- Address during next weekly operations cycle.

---

## 3. Kill Switch Procedures

Kill switches provide immediate, targeted shutdown of specific integrations or the entire system. All kill switch operations require operator-level access verified by pairing identity.

### 3.1 Full System Kill Switch

**When to use:** Confirmed security breach, runaway automation causing damage, or operator determines full stop is safest action.

```
alwaysclaw ops kill --confirm
```

**Effect:** Stops all services (WSL2 core and Windows sidecars). Windows Task Scheduler watchdog will attempt to restart unless disabled.

**To also disable watchdog restart:**
```
schtasks /change /tn AlwaysClaw-Watchdog /disable
schtasks /change /tn AlwaysClaw-Bootstrap /disable
```

**Recovery:** Manual restart required after root cause is resolved. Re-enable scheduled tasks first, then run boot sequence.

### 3.2 Degraded Mode Activation

**When to use:** Need to preserve read-only assistant mode while investigating an issue.

```
alwaysclaw ops enter-degraded --reason "description of why"
```

**Effect:** Disables tier-2 tools, halts autonomous loops, preserves read-only and communication channels. Requires human acknowledgment to exit.

### 3.3 Per-Integration Kill Switches

#### Gmail

```
alwaysclaw ops kill-integration gmail
```

**Effect:** Stops Gmail watch, halts push notification processing, disables Gmail action execution loop. Inbound email queued but not processed.

**Recovery:**
```
alwaysclaw ops restore-integration gmail
# Verify: alwaysclaw ops status gmail
```

After restore, Gmail watch must be renewed. The daily gmail-watch-verify job handles this automatically, or run manually:
```
alwaysclaw ops renew-gmail-watch
```

#### Twilio (SMS and Voice)

```
alwaysclaw ops kill-integration twilio
```

**Effect:** Stops all Twilio webhook processing, halts outbound SMS/calls, disables voice call orchestration.

**Recovery:**
```
alwaysclaw ops restore-integration twilio
# Verify: alwaysclaw ops status twilio
```

#### Browser

```
alwaysclaw ops kill-integration browser
```

**Effect:** Kills browser control sidecar, closes all managed browser sessions, halts browser workflow executor.

**Recovery:**
```
alwaysclaw ops restore-integration browser
# Verify: alwaysclaw ops status browser
```

#### Voice

```
alwaysclaw ops kill-integration voice
```

**Effect:** Stops STT/TTS, talk loop, and voice call handling. Text channels remain active.

**Recovery:**
```
alwaysclaw ops restore-integration voice
# Verify: alwaysclaw ops status voice
```

#### Autonomous Loops

```
alwaysclaw ops kill-integration loops
```

**Effect:** Stops loop kernel (loopd). All automated loops halt. Manual command processing continues through gateway.

**Recovery:**
```
alwaysclaw ops restore-integration loops
# Verify: alwaysclaw ops status loops
```

---

## 4. Secret Rotation Runbook

**Target:** Complete rotation of compromised credentials within 15 minutes.

### 4.1 Rotation Procedure

**Step 1 - Identify compromised credentials (0-2 min):**
- Review forensics data to determine which credentials may be exposed.
- Check: Gmail OAuth tokens, Twilio API keys/auth tokens, gateway WS auth tokens, WSL2-to-host bridge API keys.

**Step 2 - Revoke compromised credentials (2-5 min):**

For Gmail OAuth:
```
alwaysclaw secrets revoke gmail-oauth
# Or manually revoke at: https://myaccount.google.com/permissions
```

For Twilio:
```
alwaysclaw secrets revoke twilio-api-key
# Or manually rotate at: https://console.twilio.com/
```

For gateway auth:
```
alwaysclaw secrets rotate gateway-ws-token
```

For bridge API key:
```
alwaysclaw secrets rotate bridge-api-key
```

**Step 3 - Generate new credentials (5-10 min):**
```
alwaysclaw secrets generate --scope <integration-name>
```
Or generate manually through provider consoles and update:
```
alwaysclaw secrets set <key-name> --value <new-value>
```

**Step 4 - Deploy new credentials (10-13 min):**
```
alwaysclaw secrets deploy --scope <integration-name>
# This restarts affected services with new credentials
```

**Step 5 - Verify (13-15 min):**
```
alwaysclaw ops health-check --scope <integration-name>
alwaysclaw ops test-integration <integration-name>
```

### 4.2 Post-Rotation Verification

- Confirm old credentials return authentication errors when tested.
- Confirm new credentials allow normal operation.
- Verify no other services or loops are using the old credentials.
- Log rotation event with timestamp, operator identity, and affected scope.

---

## 5. Forensics Collection

When an incident is detected, collect a forensics package before making changes that could destroy evidence.

### 5.1 Forensics Package Contents

The forensics package contains:

1. **Command history:** Last 1000 commands processed by gateway, with timestamps and session IDs.
2. **Loop state:** Current state of all active and recently completed loops, including checkpoint data.
3. **Channel event trace:** Inbound/outbound events across Gmail, Twilio, browser, and voice channels for the incident window.
4. **Health check history:** All watchdog health check results for trailing 1 hour.
5. **Restart history:** All component restart events for trailing 24 hours.
6. **System state snapshot:** Process list, port bindings, memory usage, disk usage.
7. **Configuration snapshot:** Current running config (with secrets redacted).
8. **Log extract:** Structured JSON logs for the incident window (15 min before to current).
9. **Memory state:** Recent memory write operations and compaction events.
10. **Network state:** Active connections, pending webhook callbacks, queue depths.

### 5.2 Collection Command

```
alwaysclaw ops collect-forensics --incident-id <eventId> --window 30m
```

This produces a timestamped archive at:
```
C:\AlwaysClaw\state\forensics\incident-<eventId>-<timestamp>.tar.gz
```

### 5.3 Manual Collection (if automated collection unavailable)

```bash
# From WSL2:
wsl.exe -d Ubuntu -- bash -c "
  FDIR=/tmp/forensics-$(date +%Y%m%d%H%M%S)
  mkdir -p $FDIR/logs
  cp -r /opt/alwaysclaw/state/logs/recent/* $FDIR/logs/
  cp /opt/alwaysclaw/state/cron/jobs.json $FDIR/
  cp /opt/alwaysclaw/state/config/alwaysclaw.json $FDIR/config-redacted.json
  journalctl -u 'alwaysclaw-*' --since '30 min ago' > $FDIR/journal.log
  ps aux | grep alwaysclaw > $FDIR/processes.txt
  ss -tlnp > $FDIR/ports.txt
  echo 'Forensics collected at:' $FDIR
"
```

---

## 6. Containment Steps by Incident Type

### 6.1 Runaway Automation

**Symptoms:** Loops executing without pause, unexpected outbound messages, rapid credential usage.

**Containment:**
1. Kill autonomous loops: `alwaysclaw ops kill-integration loops`
2. Enter degraded mode: `alwaysclaw ops enter-degraded --reason "runaway automation"`
3. Collect forensics.
4. Review loop execution history and identify triggering event.
5. Fix root cause (loop contract bug, missing guardrail, approval bypass).
6. Run regression test before re-enabling.

### 6.2 Webhook Spoofing / Signature Validation Failure

**Symptoms:** Unsigned or invalidly signed webhook requests processed, unexpected commands from external sources.

**Containment:**
1. Kill affected integration: `alwaysclaw ops kill-integration <gmail|twilio>`
2. Rotate affected credentials immediately (see Section 4).
3. Collect forensics to determine if any spoofed commands were executed.
4. Review webhook logs for the incident window.
5. Verify signature validation code is correct before restoring.

### 6.3 Prompt Injection via Browser or Email

**Symptoms:** Unexpected tool executions following browser navigation or email processing, commands that appear to originate from content injection.

**Containment:**
1. Kill browser integration: `alwaysclaw ops kill-integration browser`
2. If via email: kill Gmail integration: `alwaysclaw ops kill-integration gmail`
3. Enter degraded mode to halt all tool execution.
4. Collect forensics.
5. Review command history to identify injected commands and their effects.
6. Revert any unauthorized changes made by injected commands.
7. Review and strengthen prompt injection scanner patterns before restoring.

### 6.4 Credential Compromise

**Symptoms:** Unauthorized access detected, unexpected API usage patterns, provider security alerts.

**Containment:**
1. Kill all integrations immediately: `alwaysclaw ops kill --confirm`
2. Execute full secret rotation (Section 4) for all potentially affected credentials.
3. Collect forensics.
4. Review access logs across all providers (Gmail, Twilio, OpenAI).
5. Verify no persistent access mechanisms were installed.
6. Perform full security audit before restoring.

### 6.5 WSL2 / Host System Failure

**Symptoms:** WSL2 distribution unresponsive, Windows host performance degradation, disk space exhaustion.

**Containment:**
1. System enters degraded mode automatically via watchdog.
2. Check Windows host health: disk space, memory, CPU.
3. If disk space: run emergency log compaction and clear temp files.
4. If WSL2 hung: `wsl.exe --shutdown` then `wsl.exe -d Ubuntu`
5. If persistent: check WSL2 configuration, Windows updates, hypervisor status.
6. Restart full stack after WSL2 is verified healthy.

### 6.6 SLO Breach / Error Budget Exhaustion

**Symptoms:** SLO burn rate alerts firing, error budget warning or exhaustion notification received.

**Containment:**
1. Budget warning (25% remaining): Freeze non-critical changes, disable tier-B loops, increase monitoring.
2. Budget exhausted (0% remaining): System automatically enters degraded mode.
3. Identify the SLO that was breached and its contributing error sources.
4. Address root cause (latency, failures, missed cron jobs).
5. Monitor budget recovery before re-enabling frozen capabilities.

---

## 7. Communication Templates

### 7.1 Incident Detected (Internal Alert)

```
Subject: [P{severity}] AlwaysClaw Incident - {brief description}

Incident ID: {eventId}
Severity: P{severity}
Detected: {timestamp}
Component: {affected component(s)}
Status: Investigating

Summary: {1-2 sentence description of what was detected}

Current Impact: {description of user-visible impact}

Actions Taken:
- {action 1}
- {action 2}

Next Update: {time of next update}
```

### 7.2 Degraded Mode Entered

```
Subject: [P1] AlwaysClaw Entered Degraded Mode - Operator Action Required

Incident ID: {eventId}
Trigger: {trigger condition description}
Time: {timestamp}

System State:
- Active: {list of running services}
- Failed: {list of failed services}
- Disabled: Tier-2 tools, autonomous loops

Operator Action Required:
Please review the incident and acknowledge when ready to restore:
  alwaysclaw ops ack-degraded --incident-id {eventId}

Or reply ACK to this email.
```

### 7.3 Incident Resolved

```
Subject: [RESOLVED] AlwaysClaw Incident {eventId} - {brief description}

Incident ID: {eventId}
Severity: P{severity}
Duration: {start time} to {end time} ({total duration})
Root Cause: {brief root cause}

Resolution: {what was done to fix it}

User Impact: {summary of impact during incident}

Follow-up Actions:
- {action 1 with owner and deadline}
- {action 2 with owner and deadline}

Post-Incident Review: Scheduled for {date/time}
```

---

## 8. Post-Incident Review Process

### 8.1 Timing

- P1 incidents: Review within 24 hours of resolution.
- P2 incidents: Review within 72 hours.
- P3/P4 incidents: Review during weekly operations cycle.

### 8.2 Review Template

The post-incident review must cover:

1. **Timeline:** Minute-by-minute reconstruction from detection to resolution.
2. **Root cause:** Technical root cause analysis. Use "5 Whys" method.
3. **Detection:** How was the incident detected? Could detection have been faster?
4. **Response:** Were runbook procedures followed? Were they adequate?
5. **Impact:** Quantified user impact (duration, failed commands, missed SLOs).
6. **Corrective actions:** Specific, actionable items with owners and deadlines:
   - Immediate fixes already applied.
   - Short-term hardening (within 1 week).
   - Long-term prevention (within 1 month).
7. **Lessons learned:** What should be added to monitoring, runbooks, or training?
8. **SLO impact:** How did this incident affect error budgets?

### 8.3 Storage

Post-incident reviews are stored at:
```
C:\AlwaysClaw\state\incidents\{eventId}\postmortem.md
```

Forensics packages, timeline reconstructions, and corrective action tracking are co-located in the same incident directory.

### 8.4 Corrective Action Tracking

All corrective actions from post-incident reviews feed into the Planning Loop backlog. They are tagged with `source: incident-review` and tracked until completion. The weekly strategy renewal (Ralph) loop reviews outstanding incident-sourced actions as a priority input.

---

## 9. Escalation Path

| Elapsed Time | Action |
|---|---|
| 0 min | Automated detection and alert |
| 5 min | Automated recovery attempt (watchdog) |
| 15 min | If unresolved: escalate to P1, page operator |
| 30 min | If unresolved: enter degraded mode if not already |
| 1 hour | If unresolved: full system assessment, consider full stop |
| 4 hours | Operator must provide status update |
| 24 hours | If degraded mode persists: send escalation reminder |

---

## 10. Quick Reference Card

| Scenario | First Command |
|---|---|
| Full emergency stop | `alwaysclaw ops kill --confirm` |
| Enter degraded mode | `alwaysclaw ops enter-degraded --reason "..."` |
| Kill Gmail | `alwaysclaw ops kill-integration gmail` |
| Kill Twilio | `alwaysclaw ops kill-integration twilio` |
| Kill Browser | `alwaysclaw ops kill-integration browser` |
| Kill Voice | `alwaysclaw ops kill-integration voice` |
| Kill Loops | `alwaysclaw ops kill-integration loops` |
| Rotate secrets | `alwaysclaw secrets rotate <scope>` |
| Collect forensics | `alwaysclaw ops collect-forensics --incident-id <id>` |
| Check health | `alwaysclaw ops health-check` |
| Ack degraded mode | `alwaysclaw ops ack-degraded --incident-id <id>` |
| System status | `alwaysclaw ops status` |
