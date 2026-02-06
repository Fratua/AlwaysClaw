# AlwaysClaw: Windows 10 + WSL2 Deployment Guide

## Table of Contents

- [Platform Baseline](#platform-baseline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Storage Model](#storage-model)
- [Boot and Recovery](#boot-and-recovery)
- [Watchdog and Restart Policy](#watchdog-and-restart-policy)
- [Health Checks](#health-checks)
- [Daily Operations](#daily-operations)
- [Weekly Operations](#weekly-operations)
- [Degraded Mode](#degraded-mode)
- [Manual Operations Reference](#manual-operations-reference)
- [Troubleshooting](#troubleshooting)

---

## Platform Baseline

AlwaysClaw runs as a **hybrid WSL2 + Windows host** deployment:

| Layer | Runtime | Purpose |
|---|---|---|
| Core plane | WSL2 (Ubuntu) | Gateway, loop kernel, scheduler, memory, hooks |
| Host sidecars | Windows native | PowerShell tool bridge, desktop automation, Task Scheduler |
| Supervision | Windows Task Scheduler | Boot, watchdog (5 min), daily maintenance |

**Why WSL2?** The upstream OpenClaw runtime is Linux-first. Running the core stack
inside WSL2 provides the closest behavior parity while preserving native Windows
host access through controlled sidecar bridges.

### Minimum Requirements

- Windows 10 version 21H2 or later (build 19041+)
- WSL2 with Ubuntu distro
- 8 GB RAM minimum (16 GB recommended for full stack)
- 20 GB free disk space on the system drive
- Administrator access for initial setup

---

## Prerequisites

Before running the setup script, ensure:

1. **Windows version**: Run `winver` and confirm build >= 19041.
2. **Virtualization**: BIOS/UEFI virtualization (Intel VT-x / AMD-V) is enabled.
3. **Windows features**: Hyper-V and Virtual Machine Platform should be available
   (the setup script enables them automatically via `wsl --install`).

---

## Installation

### Automated Setup

Run the bootstrap script from an **elevated PowerShell** prompt:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
.\deployment\wsl2-setup.ps1
```

The script performs these steps in order:

1. Validates Windows build version.
2. Creates the `C:\AlwaysClaw\state\` directory tree.
3. Installs WSL2 if not present (`wsl --install`).
4. Sets WSL default version to 2.
5. Installs the Ubuntu distro if absent.
6. Writes `/etc/wsl.conf` with `systemd=true` and restarts WSL.
7. Runs OpenClaw-compatible gateway install commands inside WSL.
8. Registers Windows Scheduled Tasks (bootstrap, watchdog, maintenance).
9. Runs a post-install health check.

A reboot may be required after the first WSL installation.

### Manual WSL2 Setup

If you prefer to configure WSL2 manually:

```powershell
# Install WSL with Ubuntu
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# After reboot, configure systemd
wsl -d Ubuntu -- bash -c "sudo tee /etc/wsl.conf <<'EOF'
[boot]
systemd=true

[automount]
enabled=true
options=\"metadata\"

[interop]
enabled=true
appendWindowsPath=true
EOF"

# Restart WSL to apply
wsl --shutdown
```

Then install the gateway service:

```bash
# Inside WSL
openclaw onboard --install-daemon
openclaw gateway install
openclaw doctor
```

---

## Storage Model

All persistent state lives under `C:\AlwaysClaw\state\`:

```
C:\AlwaysClaw\state\
  config\
    alwaysclaw.json          # master configuration
  agents\
    <agentId>\
      workspace\             # persona files (AGENTS.md, SOUL.md, etc.)
      sessions\              # session history
  cron\
    jobs.json                # persistent scheduler job store
  logs\
    watchdog.log             # watchdog output
    health-report.json       # latest structured health report
    wsl2-setup_*.log         # setup run logs
  auth\
    <credentials>            # scoped per integration
  memory\
    vector\                  # vector/hybrid retrieval store
  incidents\
    incident_*.json          # incident event records
  heartbeat.json             # last heartbeat timestamp
  degraded-mode.json         # degraded mode flag (when active)
  watchdog-restart-state.json # restart attempt tracking
```

### Backup Strategy

- **Daily**: Automated log compaction runs at 03:00 via scheduled task.
- **Weekly**: Run `openclaw backup --full` inside WSL for complete state snapshot.
- **Before upgrades**: Always snapshot the `state\` directory.

---

## Boot and Recovery

### Boot Sequence

At system startup, the registered scheduled task triggers:

1. **WSL2 availability verified** -- confirms the distro is accessible.
2. **Core stack started** -- gateway, loop kernel, tools service.
3. **Sidecars started** -- voice, browser control.
4. **Health check executed** -- structured JSON report emitted.

The boot trigger has a 60-second delay (boot) or 30-second delay (logon) to allow
Windows networking to stabilize.

### Recovery Sequence

When the watchdog detects a failed component:

1. Verifies heartbeat freshness (must be < 10 minutes old).
2. Checks process liveness via `pgrep` inside WSL.
3. If a process is down, initiates restart with **exponential backoff**.
4. If the circuit breaker trips (5 failures in 15 minutes), enters **degraded mode**.

---

## Watchdog and Restart Policy

The watchdog runs every **5 minutes** and monitors all components.

### Exponential Backoff

| Attempt | Delay Before Restart |
|---:|---:|
| 1 | 10 seconds |
| 2 | 20 seconds |
| 3 | 40 seconds |
| 4 | 80 seconds |
| 5 | 160 seconds |
| 6+ | 300 seconds (cap) |

The backoff multiplier is 2x, capped at 5 minutes.

### Circuit Breaker

- **Window**: 15 minutes (sliding)
- **Threshold**: 5 restart attempts within the window
- **Action**: Enter degraded mode, emit incident event

The restart counter resets when a component is confirmed alive.

### Monitored Components

| Component | Type | Tier | Check Method |
|---|---|---|---|
| alwaysclaw-gateway | WSL | core | `pgrep -f openclaw` |
| alwaysclaw-loopd | WSL | core | `pgrep -f alwaysclaw-loopd` |
| alwaysclaw-tools | WSL | tier2 | `pgrep -f alwaysclaw-tools` |
| alwaysclaw-voice | WSL | tier2 | `pgrep -f alwaysclaw-voice` |

---

## Health Checks

The health check script produces a structured JSON report with these checks:

| Check | Category | What It Verifies |
|---|---|---|
| `wsl2_available` | platform | `wsl --status` succeeds |
| `wsl_distro` | platform | Ubuntu distro is running |
| `gateway_process` | core_stack | Gateway process alive |
| `loopd_process` | core_stack | Loop kernel alive |
| `tools_process` | core_stack | Tools service alive |
| `voice_process` | sidecars | Voice service alive |
| `heartbeat_freshness` | operations | Last heartbeat < 10 min old |
| `scheduled_tasks` | operations | Task Scheduler entries exist |
| `degraded_mode` | operations | System not in degraded state |
| `disk_space` | infrastructure | > 5 GB free on system drive |
| `state_directories` | infrastructure | All required dirs exist |

### Status Levels

- **healthy**: Check passed.
- **warning**: Non-critical issue detected.
- **critical**: Component down or system degraded.
- **error**: Check itself failed.

### Running Manually

```powershell
.\deployment\health-check.ps1 | ConvertFrom-Json | Format-List
```

The report is also written to `C:\AlwaysClaw\state\logs\health-report.json`.

---

## Daily Operations

These tasks run automatically:

| Time | Task | Method |
|---|---|---|
| Every 5 min | Watchdog liveness checks | Scheduled Task |
| Every 30 min | Heartbeat synthesis | Internal cron (WSL) |
| 03:00 | Log compaction + archive | Scheduled Task |
| 07:00 | Daily reliability report | Internal cron (WSL) |

### Manual Daily Checks

```powershell
# View latest health report
Get-Content C:\AlwaysClaw\state\logs\health-report.json | ConvertFrom-Json

# View watchdog log (last 50 lines)
Get-Content C:\AlwaysClaw\state\logs\watchdog.log -Tail 50

# Check for active incidents
Get-ChildItem C:\AlwaysClaw\state\incidents\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Verify scheduled tasks
schtasks /Query /TN "\AlwaysClaw\" /FO LIST
```

---

## Weekly Operations

| Day | Task | Method |
|---|---|---|
| Sunday 09:00 | Security audit loop | Internal cron (WSL) |
| Sunday | Backup integrity check | Manual or cron |
| Weekly | Upgrade proposal review | Self-Upgrading loop (requires approval) |

### Manual Weekly Tasks

```bash
# Inside WSL: run security audit
openclaw loop run security-audit

# Full backup
openclaw backup --full --output /mnt/c/AlwaysClaw/backups/
```

---

## Degraded Mode

When the circuit breaker trips for any component, the system enters degraded mode.

### What Happens

1. **Tier-2 tools are disabled** (tools service, voice, browser automation).
2. **Read-only capabilities remain active** (memory queries, status checks).
3. **Communication channels stay alive** (gateway, messaging).
4. **An incident event is emitted** to `C:\AlwaysClaw\state\incidents\`.
5. **A degraded-mode flag is written** to `C:\AlwaysClaw\state\degraded-mode.json`.

### Operator Acknowledgment

Degraded mode persists until an operator explicitly acknowledges and clears it:

```powershell
# View degraded mode status
Get-Content C:\AlwaysClaw\state\degraded-mode.json | ConvertFrom-Json

# Clear degraded mode after fixing the issue
Remove-Item C:\AlwaysClaw\state\degraded-mode.json -Force

# Reset watchdog restart counters
Remove-Item C:\AlwaysClaw\state\watchdog-restart-state.json -Force

# Manually restart the failed component
wsl -d Ubuntu -- systemctl restart alwaysclaw-tools
```

### Re-elevation

After clearing degraded mode:

1. The next watchdog cycle will detect components as alive (or restart them).
2. Tier-2 tools become available again.
3. The incident remains in the incidents directory for audit.

---

## Manual Operations Reference

### WSL Management

```powershell
# Restart WSL entirely
wsl --shutdown
# (WSL restarts automatically on next access)

# Check WSL status
wsl --status

# Enter WSL shell
wsl -d Ubuntu

# List running distros
wsl --list --running
```

### Service Management (inside WSL)

```bash
# Check gateway status
openclaw doctor

# Restart gateway
openclaw gateway restart

# View service logs
journalctl -u alwaysclaw-gateway --since "1 hour ago"

# Check all AlwaysClaw services
systemctl list-units 'alwaysclaw-*'
```

### Task Scheduler Management

```powershell
# List AlwaysClaw tasks
schtasks /Query /TN "\AlwaysClaw\" /FO LIST /V

# Run watchdog manually
schtasks /Run /TN "\AlwaysClaw\Watchdog"

# Disable watchdog temporarily
schtasks /Change /TN "\AlwaysClaw\Watchdog" /DISABLE

# Re-enable watchdog
schtasks /Change /TN "\AlwaysClaw\Watchdog" /ENABLE
```

---

## Troubleshooting

### WSL2 will not start

1. Verify virtualization is enabled in BIOS/UEFI.
2. Run `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`.
3. Reboot and try `wsl --install` again.

### Gateway fails to start

1. Check logs: `wsl -d Ubuntu -- journalctl -u alwaysclaw-gateway -n 100`.
2. Run doctor: `wsl -d Ubuntu -- openclaw doctor`.
3. Reinstall: `wsl -d Ubuntu -- openclaw gateway install`.

### Watchdog keeps restarting a component

1. Check the restart state: `Get-Content C:\AlwaysClaw\state\watchdog-restart-state.json`.
2. Check the component's own logs inside WSL.
3. If the component has a configuration issue, fix it before clearing the restart state.

### System stuck in degraded mode

1. Identify the root cause from the incident file in `C:\AlwaysClaw\state\incidents\`.
2. Fix the underlying issue (service config, resource exhaustion, etc.).
3. Clear the degraded flag and restart state as described in the [Degraded Mode](#degraded-mode) section.
4. Monitor the next few watchdog cycles to confirm stability.

### Heartbeat always stale

1. Verify the gateway is running and healthy.
2. Check that the heartbeat writer has permission to update `C:\AlwaysClaw\state\heartbeat.json`.
3. Confirm the WSL clock is not drifting: `wsl -d Ubuntu -- date` vs system time.
   If drifted, restart WSL: `wsl --shutdown`.

### Disk space warnings

1. Run log compaction manually: `wsl -d Ubuntu -- openclaw maintenance --log-compact`.
2. Archive old session data: `wsl -d Ubuntu -- openclaw maintenance --archive`.
3. Review and prune `C:\AlwaysClaw\state\memory\vector\` if it has grown large.
