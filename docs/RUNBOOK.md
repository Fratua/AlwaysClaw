# AlwaysClaw Operations Runbook

Version: 1.0.0
Generated from: OPENCLAW_WIN10_GPT52_MASTERPLAN.md sections 7, 8, 16, 17, 20, 28, 29, 35, 44, 62
Date: 2026-02-06

---

## 1. Overview

AlwaysClaw is a Windows 10-focused, always-on personal AI system running a hybrid WSL2 + Windows sidecar architecture. This runbook covers boot procedures, recovery sequences, daily/weekly/monthly operations, and troubleshooting for every major component.

### Architecture Summary

The system consists of:

- **WSL2 Core Plane (Ubuntu):** Gateway, loop kernel (loopd), tool execution, scheduler, memory/vector store.
- **Windows Host Sidecars:** PowerShell bridge, desktop automation, browser control (Playwright/CDP).
- **Windows Task Scheduler:** Bootstrap at startup, 5-minute watchdog, daily maintenance.
- **External Integrations:** Gmail (Pub/Sub push), Twilio (SMS + Voice), OpenAI GPT-5.2.

### Key Paths

| Purpose | Path |
|---|---|
| Configuration | `C:\AlwaysClaw\state\config\alwaysclaw.json` |
| Agent workspaces | `C:\AlwaysClaw\state\agents\<agentId>\workspace\` |
| Sessions | `C:\AlwaysClaw\state\agents\<agentId>\sessions\` |
| Cron jobs | `C:\AlwaysClaw\state\cron\jobs.json` |
| Logs | `C:\AlwaysClaw\state\logs\` |
| Auth/secrets | `C:\AlwaysClaw\state\auth\` |
| Memory (vector) | `C:\AlwaysClaw\state\memory\vector\` |
| Forensics | `C:\AlwaysClaw\state\forensics\` |
| Incidents | `C:\AlwaysClaw\state\incidents\` |
| Watchdog logs | `C:\AlwaysClaw\state\logs\watchdog\` |
| Metrics | `C:\AlwaysClaw\state\logs\metrics\` |

---

## 2. Boot and Recovery Sequence

### 2.1 Normal Boot (System Startup)

The Windows Task Scheduler job `AlwaysClaw-Bootstrap` triggers at system startup and executes the following sequence:

**Step 1 - Verify WSL2 availability:**
```powershell
wsl.exe -l --running
```
Expected: `Ubuntu` appears in the list. If not:
```powershell
wsl.exe -d Ubuntu
```

**Step 2 - Verify systemd is active inside WSL2:**
```bash
wsl.exe -d Ubuntu -- systemctl is-system-running
```
Expected: `running`. Acceptable: `degraded` (some non-critical units may be slow). If `offline` or not responding, restart WSL2:
```powershell
wsl.exe --shutdown
wsl.exe -d Ubuntu
```

**Step 3 - Start core stack (ordered):**

The boot script at `/opt/alwaysclaw/bin/boot.sh` starts services in dependency order:

1. `alwaysclaw-gateway` (port 3100) - must be healthy before others start.
2. `alwaysclaw-memory` (port 3105), `alwaysclaw-scheduler` (port 3103), `alwaysclaw-tools` (port 3102) - started in parallel.
3. `alwaysclaw-loopd` (port 3101) - depends on gateway and tools.
4. `alwaysclaw-voice` (port 3104) - started last, non-critical.

**Step 4 - Start Windows sidecars:**
```powershell
Start-Process "C:\AlwaysClaw\bin\alwaysclaw-ps-bridge.exe" -WindowStyle Hidden
Start-Process "C:\AlwaysClaw\bin\alwaysclaw-desktop.exe" -WindowStyle Hidden
Start-Process "C:\AlwaysClaw\bin\alwaysclaw-browser.exe" -WindowStyle Hidden
```

**Step 5 - Run full health check:**
```bash
alwaysclaw ops health-check
```
Expected: All components report healthy. If any component is unhealthy, the watchdog will attempt recovery.

**Step 6 - Emit startup-complete event:**
The system emits a `system-boot-complete` event on the incident event bus and logs the startup time.

### 2.2 Recovery After Crash

When the 5-minute watchdog detects a failed component:

1. Watchdog identifies which component(s) are unhealthy via health endpoint probes.
2. For each unhealthy component, it attempts restart using systemctl (WSL2) or process start (Windows).
3. Restart uses exponential backoff: 2s, 4s, 8s, 16s, 32s delay between attempts.
4. If a component exceeds 5 restarts within 15 minutes, the restart governor triggers degraded mode.
5. Degraded mode disables tier-2 tools and autonomous loops while preserving read-only assistant mode.
6. Recovery from degraded mode requires human acknowledgment (see incident runbook).

### 2.3 Recovery After WSL2 Failure

If WSL2 itself becomes unresponsive:

1. From Windows PowerShell (admin):
```powershell
wsl.exe --shutdown
Start-Sleep -Seconds 5
wsl.exe -d Ubuntu
```

2. If WSL2 will not start, check:
   - Windows hypervisor is enabled: `bcdedit /enum | findstr hypervisor`
   - WSL2 feature is installed: `dism.exe /online /get-featureinfo /featurename:Microsoft-Windows-Subsystem-Linux`
   - Sufficient disk space on the WSL2 virtual disk.

3. After WSL2 is confirmed running, execute the normal boot sequence (Section 2.1, Steps 3-6).

### 2.4 Recovery After Full System Stop

If the system was stopped with `alwaysclaw ops kill --confirm`:

1. Re-enable Task Scheduler jobs:
```powershell
schtasks /change /tn AlwaysClaw-Watchdog /enable
schtasks /change /tn AlwaysClaw-Bootstrap /enable
```

2. Run bootstrap manually:
```powershell
schtasks /run /tn AlwaysClaw-Bootstrap
```

3. Verify all components:
```bash
alwaysclaw ops health-check
alwaysclaw ops status
```

---

## 3. Daily Operations

The daily operations cycle runs automatically at 07:00 ET (after quiet hours end). The following jobs execute as part of the daily cadence:

### 3.1 Planning Loop (07:00 ET)

**What it does:** Runs the Planning Loop (L13) to reprioritize the backlog, score tasks by cost/risk/confidence, and produce an ordered execution plan for the day.

**Verify it ran:**
```bash
alwaysclaw ops job-status planning-loop --last-run
```

**If it did not run:** Check scheduler health. Manually trigger:
```bash
alwaysclaw ops run-job planning-loop
```

### 3.2 Memory Hygiene

**What it does:** Consolidates duplicate memory entries, decays stale memories, verifies vector index integrity, and compacts memory files.

**Verify:**
```bash
alwaysclaw ops job-status memory-hygiene --last-run
```

**Manual trigger:**
```bash
alwaysclaw ops run-job memory-hygiene
```

### 3.3 Gmail Watch Verification

**What it does:** Checks Gmail watch expiration. Renews if within 24 hours of expiry. Gmail watches have a maximum lifetime of 7 days; daily renewal is recommended.

**Verify:**
```bash
alwaysclaw ops gmail-watch-status
```

Expected: Watch expiration is at least 24 hours in the future.

**Manual renewal:**
```bash
alwaysclaw ops renew-gmail-watch
```

### 3.4 Log Compaction

**What it does:** Compacts structured JSON logs older than 7 days. Archives to compressed storage. Verifies log integrity checksums.

**Verify:**
```bash
alwaysclaw ops job-status log-compaction --last-run
```

### 3.5 Context Compilation Check

**What it does:** Audits context packing efficiency, token budget utilization, retrieval hit rates, and compaction effectiveness.

**Review results:**
```bash
alwaysclaw ops job-status context-compile --last-run --details
```

---

## 4. Weekly Operations

The weekly operations cycle runs Sundays at 09:00 ET.

### 4.1 Security Audit

**What it does:** Full security audit covering policy drift, credential hygiene, webhook validation status, sandbox configuration integrity, and prompt injection pattern review.

**Review results:**
```bash
alwaysclaw ops job-status security-audit --last-run --details
```

**Manual trigger:**
```bash
alwaysclaw ops run-job security-audit
```

**Key things to check in audit output:**
- Any credential rotation overdue.
- Any webhook signature validation misconfigurations.
- Any sandbox policy deviations from baseline.
- Any new prompt injection patterns detected.

### 4.2 Upgrade Proposals

**What it does:** Reviews available dependency, runtime, and plugin upgrades. Produces proposal documents with rollback plans. Proposals require human approval before execution.

**Review proposals:**
```bash
alwaysclaw ops list-proposals --type upgrade
```

**Approve a proposal:**
```bash
alwaysclaw ops approve-proposal <proposal-id>
```

### 4.3 Chaos Drill

**What it does:** Injects controlled failures (process kill, state corruption, webhook timeout, queue flood, memory pressure) and verifies that the system recovers within SLO targets.

**Review drill report:**
```bash
alwaysclaw ops job-status chaos-drill --last-run --details
```

**Important:** If the chaos drill reveals recovery gaps, create corrective action tickets immediately.

### 4.4 Strategy Renewal (Ralph Loop)

**What it does:** Runs the Ralph macro-cycle (rapid assess, learn, plan, act, reflect, harden) using xhigh thinking. Produces a prioritized strategy graph and 7-day action plan.

**Review output:**
```bash
alwaysclaw ops job-status strategy-renewal --last-run --details
```

---

## 5. Monthly Operations

Monthly operations run on the 1st of each month at 10:00 ET.

### 5.1 Disaster Recovery Restore Validation

**What it does:** Full DR simulation. Restores from latest backup, verifies state integrity, runs smoke tests, validates recovery time meets the 5-minute MTTR SLO.

**Review results:**
```bash
alwaysclaw ops job-status dr-restore-validation --last-run --details
```

**If DR validation fails:**
1. Review which step failed (restore, integrity, smoke test, timing).
2. Create P2 ticket for remediation.
3. Schedule re-run within 1 week after fix.

### 5.2 Architecture Drift Review

**What it does:** Deep comparison of running configuration against canonical schema. Flags deviations and proposes corrections.

**Review drift report:**
```bash
alwaysclaw ops job-status architecture-drift-review --last-run --details
```

---

## 6. Troubleshooting Guides

### 6.1 Gateway (alwaysclaw-gateway)

**Symptoms: Gateway not responding on port 3100.**

1. Check process status:
```bash
wsl.exe -d Ubuntu -- systemctl status alwaysclaw-gateway
```

2. Check logs:
```bash
wsl.exe -d Ubuntu -- journalctl -u alwaysclaw-gateway --since "10 min ago" --no-pager
```

3. Common causes:
   - Port 3100 already in use: `wsl.exe -d Ubuntu -- ss -tlnp | grep 3100`
   - Configuration error: validate `alwaysclaw.json` against schema.
   - Out of memory: check `wsl.exe -d Ubuntu -- free -m`

4. Restart:
```bash
wsl.exe -d Ubuntu -- systemctl restart alwaysclaw-gateway
```

### 6.2 Loop Kernel (alwaysclaw-loopd)

**Symptoms: Automated tasks not executing, loops stuck in queued/preparing state.**

1. Check process and health:
```bash
wsl.exe -d Ubuntu -- systemctl status alwaysclaw-loopd
wsl.exe -d Ubuntu -- curl -s http://localhost:3101/health
```

2. Check loop queue:
```bash
alwaysclaw ops loop-queue --status
```

3. Common causes:
   - Gateway dependency not met: verify gateway is healthy first.
   - Loop contract violation: check loop contract JSON for the stuck loop.
   - Tool budget exceeded: review recent loop execution history.

4. Restart:
```bash
wsl.exe -d Ubuntu -- systemctl restart alwaysclaw-loopd
```

### 6.3 Scheduler (alwaysclaw-scheduler)

**Symptoms: Cron jobs not firing, heartbeat synthesis not running.**

1. Check status:
```bash
wsl.exe -d Ubuntu -- systemctl status alwaysclaw-scheduler
```

2. Verify cron job state:
```bash
alwaysclaw ops cron-list
```
Check `lastRunAt` and `nextRunAt` for each job.

3. Common causes:
   - Stale job state: `jobs.json` may be corrupted. Backup and regenerate.
   - Timezone mismatch: verify `timezone` in `schedules.json` matches system.
   - Gateway dependency: scheduler needs gateway for event emission.

4. Restart:
```bash
wsl.exe -d Ubuntu -- systemctl restart alwaysclaw-scheduler
```

### 6.4 Memory Service (alwaysclaw-memory)

**Symptoms: Responses lack context, memory search returns empty, vector index errors.**

1. Check status:
```bash
wsl.exe -d Ubuntu -- systemctl status alwaysclaw-memory
wsl.exe -d Ubuntu -- curl -s http://localhost:3105/health
```

2. Verify vector index:
```bash
alwaysclaw ops memory-index-status
```

3. Common causes:
   - Corrupted vector index: rebuild with `alwaysclaw ops memory-reindex`.
   - Disk full: check `wsl.exe -d Ubuntu -- df -h`.
   - Memory file permissions: verify `C:\AlwaysClaw\state\memory\` is writable.

4. Restart:
```bash
wsl.exe -d Ubuntu -- systemctl restart alwaysclaw-memory
```

### 6.5 Gmail Integration

**Symptoms: Not receiving push notifications, emails not being processed.**

1. Check watch status:
```bash
alwaysclaw ops gmail-watch-status
```

2. If watch expired:
```bash
alwaysclaw ops renew-gmail-watch
```

3. Check history ID:
```bash
alwaysclaw ops gmail-history-status
```

4. If history ID is stale (404 from history.list):
```bash
alwaysclaw ops gmail-full-resync
```

5. Common causes:
   - Watch expired (max 7 days). Daily renewal job may have failed.
   - OAuth token expired. Re-authenticate: `alwaysclaw secrets refresh gmail-oauth`.
   - Pub/Sub subscription misconfigured. Verify push endpoint and JWT validation.
   - Rate limiting: Gmail limits push notifications to 1 event/sec per user.

### 6.6 Twilio Integration (SMS/Voice)

**Symptoms: Not receiving inbound SMS/calls, outbound messages failing.**

1. Check webhook validation:
```bash
alwaysclaw ops twilio-status
```

2. For signature validation failures:
   - Verify auth token matches Twilio console.
   - Check that webhook URL matches exactly (case-sensitive).
   - For WebSocket: verify lowercase `x-twilio-signature` header handling.

3. For outbound failures:
   - Check Twilio account balance and rate limits.
   - Verify phone numbers are correctly configured.
   - Check delivery status callbacks: `alwaysclaw ops twilio-delivery-log`.

4. Common causes:
   - Auth token rotated on Twilio side but not updated locally.
   - Webhook URL changed (e.g., tunnel URL expired).
   - Account suspended or rate-limited.

### 6.7 Browser Control

**Symptoms: Browser automation failing, screenshots not captured, DOM verification errors.**

1. Check sidecar status:
```bash
alwaysclaw ops status browser
```

2. Check managed profile integrity:
```bash
alwaysclaw ops browser-profile-status
```

3. Common causes:
   - Browser process leaked: kill stale browser processes and restart sidecar.
   - Profile corruption: reset managed profile with `alwaysclaw ops browser-reset-profile`.
   - CDP connection lost: restart browser sidecar.
   - Prompt injection detected: review scanner loop output.

4. Restart:
```powershell
Stop-Process -Name "alwaysclaw-browser" -Force
Start-Process "C:\AlwaysClaw\bin\alwaysclaw-browser.exe" -WindowStyle Hidden
```

### 6.8 Voice Service

**Symptoms: STT/TTS not working, voice calls failing, talk loop unresponsive.**

1. Check service status:
```bash
wsl.exe -d Ubuntu -- systemctl status alwaysclaw-voice
wsl.exe -d Ubuntu -- curl -s http://localhost:3104/health
```

2. Common causes:
   - Audio device not available (Windows host audio service).
   - Twilio voice webhook misconfiguration.
   - STT/TTS API key expired or rate-limited.
   - Barge-in state machine stuck: restart voice service.

3. Restart:
```bash
wsl.exe -d Ubuntu -- systemctl restart alwaysclaw-voice
```

### 6.9 Windows Task Scheduler Issues

**Symptoms: Watchdog not running, bootstrap not triggering at startup.**

1. Check task status:
```powershell
schtasks /query /tn AlwaysClaw-Watchdog /fo LIST /v
schtasks /query /tn AlwaysClaw-Bootstrap /fo LIST /v
schtasks /query /tn AlwaysClaw-DailyMaintenance /fo LIST /v
```

2. Common causes:
   - Task disabled: re-enable with `schtasks /change /tn <name> /enable`.
   - Task running under wrong user: verify `Run As User` is SYSTEM.
   - WSL2 not accessible from scheduled task: verify WSL2 distro is installed.

3. Recreate tasks if missing (use commands from `schedules.json` windowsTaskSchedulerJobs).

### 6.10 Disk Space Issues

**Symptoms: Services failing to write logs, memory service errors, backup failures.**

1. Check disk space:
```powershell
Get-PSDrive C | Select-Object Used, Free
```

Inside WSL2:
```bash
wsl.exe -d Ubuntu -- df -h
```

2. Emergency cleanup:
   - Run log compaction immediately: `alwaysclaw ops run-job log-compaction`
   - Clear old forensics packages: review and delete from `C:\AlwaysClaw\state\forensics\`
   - Clear old archived logs: review `C:\AlwaysClaw\state\logs\archive\`
   - Compact WSL2 virtual disk if needed.

### 6.11 Configuration Issues

**Symptoms: Services starting with wrong settings, unexpected behavior changes.**

1. Validate configuration:
```bash
alwaysclaw config validate
```

2. Compare running config to schema:
```bash
alwaysclaw config diff --baseline
```

3. If config is corrupted, restore from backup:
```bash
alwaysclaw config restore --from-backup
```

4. After any config change, restart affected services:
```bash
alwaysclaw ops restart --scope <affected-service>
```

---

## 7. Monitoring and Observability

### 7.1 Health Check Commands

```bash
# Full system health
alwaysclaw ops health-check

# Per-component health
alwaysclaw ops health-check --component gateway
alwaysclaw ops health-check --component loopd
alwaysclaw ops health-check --component tools
alwaysclaw ops health-check --component scheduler
alwaysclaw ops health-check --component memory
alwaysclaw ops health-check --component voice

# Integration health
alwaysclaw ops status gmail
alwaysclaw ops status twilio
alwaysclaw ops status browser
alwaysclaw ops status voice
```

### 7.2 SLO Monitoring

```bash
# Current SLO status
alwaysclaw ops slo-status

# Error budget remaining
alwaysclaw ops slo-budget

# Burn rate for specific SLO
alwaysclaw ops slo-burn-rate --slo command-success-rate
```

### 7.3 Log Locations

| Log Type | Path |
|---|---|
| Gateway | WSL2: `/opt/alwaysclaw/logs/gateway/` |
| Loop kernel | WSL2: `/opt/alwaysclaw/logs/loopd/` |
| Scheduler | WSL2: `/opt/alwaysclaw/logs/scheduler/` |
| Watchdog | `C:\AlwaysClaw\state\logs\watchdog\` |
| Metrics (hourly) | `C:\AlwaysClaw\state\logs\metrics\hourly\` |
| Incidents | `C:\AlwaysClaw\state\incidents\` |
| Forensics | `C:\AlwaysClaw\state\forensics\` |

All logs use structured JSON format with event IDs and correlation IDs for tracing.

### 7.4 Key Alerts

The system generates alerts for the following conditions:

| Alert | Severity | Trigger |
|---|---|---|
| Missed heartbeat | P2 | No heartbeat for 2 consecutive cycles (60+ min) |
| Repeated crash loop | P1 | 5+ restarts in 15 min |
| Privileged tool spike | P2 | Unusual volume of tier-2 tool requests |
| SLO burn rate high | P1-P3 | Burn rate exceeds threshold (see slo-targets.json) |
| Error budget warning | P2 | Budget at 25% remaining |
| Error budget exhausted | P1 | Budget at 0% |
| Gmail watch expiring | P3 | Watch expires within 12 hours |
| Webhook validation failure | P2 | Unsigned/invalid signature detected |
| Degraded mode entered | P1 | System entered degraded mode |

---

## 8. Quiet Hours Policy

Quiet hours run from 23:00 to 07:00 ET.

During quiet hours:
- **Suppressed:** heartbeat-synthesis, autonomy-opportunity-scan, queue-pressure-check, latency-regression-scan, metrics-aggregation.
- **Always active:** health-probe, watchdog-check, restart-governor.

This reduces noise and resource consumption during overnight hours while maintaining safety-critical monitoring. Degraded mode transitions and P1 alerts still fire during quiet hours.

---

## 9. Emergency Contacts and Escalation

| Role | Contact Method | Response Time |
|---|---|---|
| Primary Operator | Gmail + Dashboard | 5 min (P1), 30 min (P2) |
| System (automated) | Event bus + Watchdog | Immediate |

The system will page the operator via Gmail for all P1 and P2 incidents. For P1 incidents during quiet hours, the Gmail notification bypasses quiet-hours suppression.

---

## 10. Change Management

All configuration changes to the AlwaysClaw system must follow this process:

1. **Propose:** Create change proposal via Self-Updating Loop or manual request.
2. **Review:** Review change impact, especially on SLOs and security posture.
3. **Approve:** Operator approves the change.
4. **Apply:** Deploy change to running configuration.
5. **Verify:** Run health check and relevant regression tests.
6. **Monitor:** Watch SLO burn rates for 1 hour after change.
7. **Rollback:** If issues detected, rollback to previous configuration.

Configuration changes during error budget warning or exhaustion are blocked unless they are specifically to address the SLO breach.
