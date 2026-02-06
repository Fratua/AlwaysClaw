# AlwaysClaw Incident Response Runbook

Version: 1.0.0
Last updated: 2026-02-06
Owner: Operations Team

---

## 1. Severity Classification

All incidents are classified into four priority levels. Classification drives response urgency, communication cadence, and escalation paths.

### P1 - Full Outage

The entire AlwaysClaw system is unavailable. No commands are processed, no loops execute, and no integrations function.

Examples:
- WSL2 distro will not start or is corrupted
- Gateway process crashes immediately on startup and cannot recover
- State directory is inaccessible (disk failure, permissions corruption)
- All critical components fail simultaneously

Response time: Immediate (within 5 minutes of detection).
Escalation: Operator notified immediately via all available channels.
Resolution target: Under 30 minutes.

### P2 - Major Degradation

Core functionality is impaired but the system is partially operational. Some integrations or loop classes are non-functional.

Examples:
- Gmail integration down (watch expired, OAuth token revoked)
- Twilio webhooks failing signature validation
- Loop kernel stuck in preflight for all loops
- Memory compaction failing, causing context pressure
- Voice service unresponsive while other services work

Response time: Within 15 minutes of detection.
Escalation: Operator notified via heartbeat channel.
Resolution target: Under 1 hour.

### P3 - Minor Issue

A non-critical feature or subsystem is impaired. Core functionality continues normally.

Examples:
- Single non-critical loop consistently failing
- Metrics export delayed
- Browser profile session expired (non-blocking)
- Latency regression detected but within SLO tolerance
- Single cron job missed execution window

Response time: Within 1 hour of detection.
Escalation: Logged to incident log, reviewed at next daily planning session.
Resolution target: Within 24 hours.

### P4 - Cosmetic / Low Impact

Minor issues with negligible operational impact. No user-facing functionality is affected.

Examples:
- Log formatting inconsistency
- Dashboard widget rendering incorrectly
- Non-critical configuration value suboptimal
- Documentation gap discovered

Response time: Next business day.
Escalation: Added to backlog.
Resolution target: Within 1 week.

---

## 2. Kill Switch Procedures

Each integration has a one-command kill switch for immediate containment. These procedures are designed to be executable in under 60 seconds.

### Gmail Kill Switch

Stop the Gmail watch and revoke the OAuth token to prevent any further Gmail API interaction.

```powershell
# Step 1: Stop the Gmail watch
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/gmail/watch/stop

# Step 2: Revoke the OAuth token
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/auth/gmail/revoke

# Step 3: Verify disconnection
wsl -d Ubuntu -- curl -s http://localhost:3100/api/gmail/status
# Expected: {"connected": false, "watchActive": false}
```

Restoration: Re-authenticate via OAuth flow, then re-create watch with `POST /api/gmail/watch/start`.

### Twilio Kill Switch

Disable the inbound webhook endpoint and terminate all active voice streams.

```powershell
# Step 1: Disable webhook processing
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/twilio/webhook/disable

# Step 2: Terminate all active calls
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/twilio/calls/terminate-all

# Step 3: Verify
wsl -d Ubuntu -- curl -s http://localhost:3100/api/twilio/status
# Expected: {"webhookEnabled": false, "activeCalls": 0}
```

Restoration: Re-enable webhook with `POST /api/twilio/webhook/enable` after root cause is resolved.

### Browser Kill Switch

Close all managed browser profiles and terminate the browser control service.

```powershell
# Step 1: Close all browser sessions
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/browser/sessions/close-all

# Step 2: Kill browser processes
wsl -d Ubuntu -- pkill -f "chromium.*alwaysclaw-profile"

# Step 3: Verify
wsl -d Ubuntu -- curl -s http://localhost:3100/api/browser/status
# Expected: {"activeSessions": 0, "processRunning": false}
```

Restoration: Restart browser service with `systemctl restart alwaysclaw-browser`.

### Voice Kill Switch

Terminate all active voice streams and disable the STT/TTS pipeline.

```powershell
# Step 1: Terminate all voice streams
wsl -d Ubuntu -- curl -s -X POST http://localhost:3100/api/voice/streams/terminate-all

# Step 2: Stop voice service
wsl -d Ubuntu -- systemctl stop alwaysclaw-voice

# Step 3: Verify
wsl -d Ubuntu -- curl -s http://localhost:3100/api/voice/status
# Expected: connection refused (service stopped)
```

Restoration: `systemctl start alwaysclaw-voice` after root cause is resolved.

---

## 3. Secret Rotation Runbook

Target: Complete rotation within 15 minutes for any single credential.

### Gmail OAuth Token Rotation

1. Revoke current token: `POST /api/auth/gmail/revoke`
2. Clear stored credentials: `rm /home/alwaysclaw/state/auth/gmail-tokens.json`
3. Re-authenticate: Open `http://localhost:3100/auth/gmail/start` in browser
4. Complete OAuth consent flow
5. Verify new token works: `GET /api/gmail/status`
6. Re-create Gmail watch: `POST /api/gmail/watch/start`
7. Verify watch active: `GET /api/gmail/watch/status`

Estimated time: 5-8 minutes.

### Twilio API Key Rotation

1. Log into Twilio Console
2. Generate new API key pair (SID + secret)
3. Update AlwaysClaw config:
   ```bash
   wsl -d Ubuntu -- alwaysclaw config set twilio.apiKeySid "NEW_SID"
   wsl -d Ubuntu -- alwaysclaw config set twilio.apiKeySecret "NEW_SECRET"
   ```
4. Restart Twilio integration: `systemctl restart alwaysclaw-twilio`
5. Verify webhook validation works: `POST /api/twilio/test-webhook`
6. Revoke old API key in Twilio Console
7. Verify SMS and voice functionality with test messages

Estimated time: 8-12 minutes.

### Browser Profile Credential Rotation

1. Close all active browser sessions: `POST /api/browser/sessions/close-all`
2. Clear stored session data: `rm -rf /home/alwaysclaw/state/browser/profiles/*/cookies`
3. Restart browser service: `systemctl restart alwaysclaw-browser`
4. Re-authenticate to required sites via managed login flow
5. Verify session health: `GET /api/browser/sessions/health`

Estimated time: 10-15 minutes (depends on number of sites).

---

## 4. Forensics Collection

When an incident occurs, collect forensics before making changes to the system state.

### Command History Export

```bash
# Export gateway command log
wsl -d Ubuntu -- cp /home/alwaysclaw/state/logs/commands.jsonl \
  /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-commands.jsonl

# Export shell history
wsl -d Ubuntu -- cp ~/.bash_history \
  /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-shell-history.txt
```

### Loop State Dump

```bash
# Dump all active loop states
wsl -d Ubuntu -- curl -s http://localhost:3101/api/loops/state-dump \
  > /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-loop-state.json

# Dump loop execution history (last 24h)
wsl -d Ubuntu -- curl -s "http://localhost:3101/api/loops/history?since=24h" \
  > /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-loop-history.json
```

### Channel Event Trace

```bash
# Export event bus trace
wsl -d Ubuntu -- curl -s http://localhost:3100/api/events/trace?last=1000 \
  > /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-event-trace.json

# Export integration-specific events
for channel in gmail twilio browser voice; do
  wsl -d Ubuntu -- curl -s "http://localhost:3100/api/events/trace?channel=$channel&last=500" \
    > "/home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-${channel}-events.json"
done
```

### Memory Snapshot

```bash
# Snapshot current memory state
wsl -d Ubuntu -- curl -s http://localhost:3100/api/memory/snapshot \
  > /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-memory-snapshot.json

# Snapshot process metrics
wsl -d Ubuntu -- curl -s http://localhost:3100/api/metrics/snapshot \
  > /home/alwaysclaw/state/forensics/incident-$(date +%Y%m%d-%H%M%S)-metrics-snapshot.json
```

---

## 5. Containment Steps by Incident Type

### Credential Compromise

1. Immediately execute the relevant kill switch (section 2)
2. Rotate the compromised credential (section 3)
3. Collect forensics (section 4) to determine scope of exposure
4. Review command history and event traces for unauthorized actions
5. If data was exfiltrated: identify affected data, assess impact, notify owner
6. Audit all stored credentials for signs of lateral movement
7. Enable enhanced logging for 72 hours post-incident

### Data Leak

1. Identify the leaking channel (browser, email, SMS, logs)
2. Execute kill switch for the affected channel
3. Collect forensics before any remediation
4. Identify what data was exposed and to whom
5. If PII or secrets were leaked: rotate all potentially exposed credentials
6. Review exfiltration guard loop logs for detection gaps
7. Update tool policies to prevent recurrence

### Integration Failure

1. Check integration health endpoint for error details
2. Verify external service status (Gmail API, Twilio, etc.)
3. If authentication failure: check token expiry and rotate if needed
4. If webhook failure: verify signature validation and endpoint reachability
5. If rate limiting: reduce cadence and implement backoff
6. If data sync failure: check history pointers (Gmail historyId) and resync if needed
7. Document root cause and update integration policy if needed

### Service Crash

1. Check watchdog logs for crash details and frequency
2. If within restart budget: allow automatic recovery
3. If restart budget exhausted: system enters degraded mode automatically
4. Collect forensics (loop state dump, command history, metrics)
5. Check for resource exhaustion (memory, disk, file descriptors)
6. If crash is reproducible: disable the triggering loop or command
7. Fix root cause, deploy fix, then manually re-elevate from degraded mode

---

## 6. Communication Templates

### P1 - Full Outage

```
INCIDENT: AlwaysClaw Full Outage
Severity: P1 - Critical
Detected: [timestamp]
Impact: All AlwaysClaw services are unavailable
Status: [Investigating | Identified | Mitigating | Resolved]
Current action: [brief description]
ETA to resolution: [estimate or "assessing"]
Next update: [timestamp]
```

### P2 - Major Degradation

```
INCIDENT: [Component] Degradation
Severity: P2 - Major
Detected: [timestamp]
Impact: [specific functionality affected]
Workaround: [if available]
Status: [Investigating | Identified | Mitigating | Resolved]
Next update: [timestamp]
```

### P3 - Minor Issue

```
NOTICE: [Component] Minor Issue
Severity: P3
Detected: [timestamp]
Impact: [minimal impact description]
Action: Scheduled for resolution in next maintenance window
```

### P4 - Low Impact

```
LOG: [Component] cosmetic issue tracked
Added to backlog for next review cycle.
```

---

## 7. Post-Incident Review Process

Every P1 and P2 incident requires a post-incident review (PIR). P3 incidents are reviewed at the operator's discretion.

### Timeline Construction

1. Identify the exact trigger time (first anomalous event)
2. Map detection time (when monitoring/watchdog caught the issue)
3. Map response time (when operator or automation began remediation)
4. Map mitigation time (when user impact was reduced)
5. Map resolution time (when full functionality was restored)
6. Document all actions taken with timestamps

### Root Cause Analysis

1. Use the "5 Whys" technique starting from the observable failure
2. Identify contributing factors: code, configuration, infrastructure, process
3. Distinguish between proximate cause and root cause
4. Identify any detection gaps (why was this not caught earlier?)
5. Document findings in structured format

### Action Items

1. Categorize fixes: immediate (hotfix), short-term (this sprint), long-term (backlog)
2. Assign owners and due dates to each action item
3. Add monitoring improvements to prevent recurrence
4. Update runbooks if procedures were missing or incorrect
5. Add regression tests if applicable

### Blameless Review Principles

- Focus on systems and processes, not individuals
- Assume everyone acted with the best information they had
- Treat failures as learning opportunities
- Document what went well in addition to what failed
- Share findings openly so the entire system benefits
- Review action item completion in the next weekly planning session

### PIR Document Template

```markdown
# Post-Incident Review: [Incident Title]

Date: [date]
Severity: [P1/P2/P3]
Duration: [start] to [end] ([total minutes])
Author: [name]

## Summary
[2-3 sentence description]

## Timeline
| Time | Event |
|------|-------|
| ...  | ...   |

## Root Cause
[description]

## Contributing Factors
- [factor 1]
- [factor 2]

## What Went Well
- [item]

## What Could Be Improved
- [item]

## Action Items
| Item | Owner | Priority | Due Date | Status |
|------|-------|----------|----------|--------|
| ...  | ...   | ...      | ...      | ...    |
```

---

## Appendix: Quick Reference

| Scenario | Kill Switch | Rotation Time | Forensics First? |
|----------|-------------|---------------|-----------------|
| Gmail compromise | Stop watch + revoke token | 5-8 min | Yes |
| Twilio compromise | Disable webhook | 8-12 min | Yes |
| Browser session hijack | Close all profiles | 10-15 min | Yes |
| Voice stream leak | Terminate all streams | 2 min | Yes |
| Service crash loop | Automatic degraded mode | N/A | Collect during |
| Disk space critical | Automatic degraded mode | N/A | No - clear space first |
| Memory exhaustion | Automatic degraded mode | N/A | Snapshot then restart |
