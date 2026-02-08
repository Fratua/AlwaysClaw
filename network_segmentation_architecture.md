# Network Segmentation & Isolation Architecture
## OpenClaw-Inspired AI Agent System (Windows 10)
### Technical Security Specification v1.0

---

## Executive Summary

This document defines a comprehensive network segmentation and isolation architecture for a Windows 10-based AI agent system. The architecture implements defense-in-depth principles through multi-layered network security zones, VLAN segmentation, micro-segmentation strategies, and granular traffic control mechanisms.

---

## 1. Network Security Zone Definitions

### 1.1 Zone Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PERIMETER ZONE (DMZ)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   WAF/      │  │   Reverse   │  │   VPN       │  │   Bastion   │        │
│  │   CDN       │  │   Proxy     │  │   Gateway   │  │   Host      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION ZONE (TIER 1)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AI Agent Core Services                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  GPT-5.2 │ │  Soul    │ │ Identity │ │  User    │ │ Heartbeat│  │   │
│  │  │  Engine  │ │  Service │ │  Manager │ │  System  │ │  Monitor │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVICE ZONE (TIER 2)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Gmail     │  │   Browser   │  │    TTS      │  │    STT      │        │
│  │   Service   │  │   Control   │  │   Engine    │  │   Engine    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Twilio    │  │   Cron      │  │   Agent     │  │   System    │        │
│  │   Service   │  │   Scheduler │  │   Loops     │  │   Access    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA ZONE (TIER 3)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Database Cluster                             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  User    │ │  Agent   │ │  Config  │ │   Log    │ │  Vector  │  │   │
│  │  │   DB     │ │   State  │ │   Store  │ │   Store  │ │   Store  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MANAGEMENT ZONE (TIER 0)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   SIEM      │  │   NMS       │  │   Patch     │  │   Backup    │        │
│  │   Server    │  │   Server    │  │   Server    │  │   Server    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Detailed Zone Specifications

#### Zone 0: Management Zone (TIER 0)
| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Administrative access, monitoring, and maintenance |
| **Security Level** | CRITICAL - Highest privilege |
| **Network Range** | 10.0.0.0/24 |
| **Access Control** | Multi-factor authentication required |
| **Monitoring** | 24/7 SIEM integration |
| **Components** | SIEM, NMS, Patch Management, Backup Systems |

#### Zone 1: Perimeter Zone (DMZ)
| Attribute | Specification |
|-----------|---------------|
| **Purpose** | External-facing services and traffic filtering |
| **Security Level** | HIGH - Public exposure risk |
| **Network Range** | 10.1.0.0/24 |
| **Access Control** | Strict ingress/egress filtering |
| **Monitoring** | Full packet capture, anomaly detection |
| **Components** | WAF, Reverse Proxy, VPN Gateway, Bastion Host |

#### Zone 2: Application Zone (TIER 1)
| Attribute | Specification |
|-----------|---------------|
| **Purpose** | AI Agent core processing and orchestration |
| **Security Level** | CRITICAL - Core business logic |
| **Network Range** | 10.2.0.0/23 |
| **Access Control** | Service-to-service authentication |
| **Monitoring** | Application-layer DLP, behavioral analysis |
| **Components** | GPT-5.2 Engine, Soul Service, Identity Manager, User System, Heartbeat Monitor |

#### Zone 3: Service Zone (TIER 2)
| Attribute | Specification |
|-----------|---------------|
| **Purpose** | External service integrations and APIs |
| **Security Level** | HIGH - External dependencies |
| **Network Range** | 10.3.0.0/22 |
| **Access Control** | API key management, OAuth 2.0 |
| **Monitoring** | API call logging, rate limiting |
| **Components** | Gmail Service, Browser Control, TTS/STT Engines, Twilio Service, Cron Scheduler, Agent Loops |

#### Zone 4: Data Zone (TIER 3)
| Attribute | Specification |
|-----------|---------------|
| **Purpose** | Data persistence and storage |
| **Security Level** | CRITICAL - Data at rest protection |
| **Network Range** | 10.4.0.0/23 |
| **Access Control** | Database-level ACLs, encryption |
| **Monitoring** | Database activity monitoring, DLP |
| **Components** | User DB, Agent State DB, Config Store, Log Store, Vector Store |

---

## 2. VLAN Segmentation Strategy

### 2.1 VLAN Allocation Table

| VLAN ID | Name | Purpose | Network | CIDR | Gateway |
|---------|------|---------|---------|------|---------|
| 10 | MGMT-ADMIN | Management Access | 10.0.0.0 | /24 | 10.0.0.1 |
| 20 | MGMT-MONITOR | Monitoring Systems | 10.0.1.0 | /24 | 10.0.1.1 |
| 100 | DMZ-EDGE | Edge Services | 10.1.0.0 | /25 | 10.1.0.1 |
| 101 | DMZ-PROXY | Proxy Services | 10.1.0.128 | /25 | 10.1.0.129 |
| 200 | APP-CORE | Core AI Services | 10.2.0.0 | /24 | 10.2.0.1 |
| 201 | APP-ORCH | Orchestration | 10.2.1.0 | /24 | 10.2.1.1 |
| 300 | SVC-EMAIL | Email Services | 10.3.0.0 | /25 | 10.3.0.1 |
| 301 | SVC-VOICE | Voice Services | 10.3.0.128 | /25 | 10.3.0.129 |
| 302 | SVC-BROWSER | Browser Control | 10.3.1.0 | /25 | 10.3.1.1 |
| 303 | SVC-AUDIO | Audio Processing | 10.3.1.128 | /25 | 10.3.1.129 |
| 304 | SVC-SCHEDULER | Cron/Scheduler | 10.3.2.0 | /25 | 10.3.2.1 |
| 305 | SVC-AGENTS | Agent Loops | 10.3.2.128 | /25 | 10.3.2.129 |
| 400 | DATA-USER | User Data | 10.4.0.0 | /25 | 10.4.0.1 |
| 401 | DATA-STATE | Agent State | 10.4.0.128 | /25 | 10.4.0.129 |
| 402 | DATA-CONFIG | Configuration | 10.4.1.0 | /25 | 10.4.1.1 |
| 403 | DATA-LOGS | Logging | 10.4.1.128 | /25 | 10.4.1.129 |
| 404 | DATA-VECTOR | Vector Store | 10.4.2.0 | /25 | 10.4.2.1 |
| 999 | QUARANTINE | Isolation | 10.255.255.0 | /24 | 10.255.255.1 |

### 2.2 VLAN Topology Diagram

```
                              ┌─────────────────┐
                              │   Core Switch   │
                              │   (L3 Router)   │
                              └────────┬────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
    ┌─────┴─────┐              ┌───────┴───────┐              ┌─────┴─────┐
    │  Firewall  │              │   IDS/IPS     │              │  Firewall  │
    │  (North)   │              │   (East-West) │              │  (South)   │
    └─────┬─────┘              └───────┬───────┘              └─────┬─────┘
          │                            │                            │
    ┌─────┴────────────────────────────┴────────────────────────────┴─────┐
    │                         Distribution Layer                          │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
    │  │ VLAN 10 │ │ VLAN 20 │ │VLAN 100 │ │VLAN 200 │ │ VLAN 300│       │
    │  │  MGMT   │ │ MONITOR │ │   DMZ   │ │   APP   │ │ SERVICE │       │
    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                              ┌────────┴────────┐
                              │  Access Layer   │
                              │  (Edge Switches)│
                              └─────────────────┘
```

### 2.3 VLAN Security Policies

```python
# VLAN Security Configuration Template
VLAN_SECURITY_POLICIES = {
    "VLAN_10_MGMT": {
        "port_security": True,
        "max_mac_addresses": 1,
        "sticky_mac": True,
        "violation_action": "restrict",
        "dhcp_snooping": True,
        "arp_inspection": True,
        "ip_source_guard": True,
        "private_vlan": False
    },
    "VLAN_100_DMZ": {
        "port_security": True,
        "max_mac_addresses": 5,
        "sticky_mac": False,
        "violation_action": "shutdown",
        "dhcp_snooping": True,
        "arp_inspection": True,
        "ip_source_guard": True,
        "private_vlan": True,
        "pvlan_type": "isolated"
    },
    "VLAN_200_APP": {
        "port_security": True,
        "max_mac_addresses": 3,
        "sticky_mac": True,
        "violation_action": "restrict",
        "dhcp_snooping": True,
        "arp_inspection": True,
        "ip_source_guard": True,
        "private_vlan": True,
        "pvlan_type": "community"
    },
    "VLAN_400_DATA": {
        "port_security": True,
        "max_mac_addresses": 2,
        "sticky_mac": True,
        "violation_action": "shutdown",
        "dhcp_snooping": True,
        "arp_inspection": True,
        "ip_source_guard": True,
        "private_vlan": True,
        "pvlan_type": "isolated"
    }
}
```

---

## 3. Micro-Segmentation Strategies

### 3.1 Host-Based Micro-Segmentation

#### Windows 10 Host Firewall Configuration

```powershell
# AI Agent System - Windows Defender Firewall Configuration
# Execute with Administrator privileges

# Define Security Profiles
$Profiles = @("Domain", "Private", "Public")

# Reset to default
Reset-NetFirewallProfile -Profile Domain,Private,Public

# Set default inbound/outbound behavior
foreach ($Profile in $Profiles) {
    Set-NetFirewallProfile -Profile $Profile `
        -DefaultInboundAction Block `
        -DefaultOutboundAction Allow `
        -NotifyOnListen True `
        -AllowUnicastResponseToMulticast False `
        -LogAllowed True `
        -LogBlocked True `
        -LogIgnored True `
        -LogMaxSizeKilobytes 32767
}

# ============================================
# AI AGENT SERVICE RULES
# ============================================

# GPT-5.2 Engine - Outbound HTTPS only
New-NetFirewallRule -DisplayName "AI-GPT52-HTTPS-Out" `
    -Direction Outbound `
    -Protocol TCP `
    -LocalPort Any `
    -RemotePort 443 `
    -Program "C:\OpenClaw\Core\gpt52_engine.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Core"

# Soul Service - Internal communication only
New-NetFirewallRule -DisplayName "AI-SOUL-Internal" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8080 `
    -RemoteAddress 10.2.0.0/23 `
    -Program "C:\OpenClaw\Core\soul_service.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Core"

# Identity Manager - Database access
New-NetFirewallRule -DisplayName "AI-IDENTITY-DB" `
    -Direction Outbound `
    -Protocol TCP `
    -LocalPort Any `
    -RemotePort 5432,3306,27017 `
    -RemoteAddress 10.4.0.0/23 `
    -Program "C:\OpenClaw\Core\identity_manager.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Core"

# Heartbeat Monitor - Monitoring only
New-NetFirewallRule -DisplayName "AI-HEARTBEAT-Monitor" `
    -Direction Outbound `
    -Protocol UDP `
    -LocalPort Any `
    -RemotePort 514,1514 `
    -RemoteAddress 10.0.1.0/24 `
    -Program "C:\OpenClaw\Core\heartbeat.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Core"

# ============================================
# SERVICE INTEGRATION RULES
# ============================================

# Gmail Service - Google API endpoints
New-NetFirewallRule -DisplayName "AI-GMAIL-GoogleAPI" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 443 `
    -RemoteAddress @(
        "142.250.0.0/15",    # Google
        "172.217.0.0/16",    # Google
        "216.58.192.0/19"    # Google
    ) `
    -Program "C:\OpenClaw\Services\gmail_service.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Services"

# Browser Control - Restricted web access
New-NetFirewallRule -DisplayName "AI-BROWSER-Controlled" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 80,443 `
    -Program "C:\OpenClaw\Services\browser_control.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Services"

# TTS Engine - Local processing + cloud
New-NetFirewallRule -DisplayName "AI-TTS-Cloud" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 443 `
    -RemoteAddress @(
        "13.107.0.0/16",     # Azure
        "20.190.0.0/16",     # Azure
        "52.0.0.0/8"         # AWS
    ) `
    -Program "C:\OpenClaw\Services\tts_engine.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Services"

# STT Engine - Local processing + cloud
New-NetFirewallRule -DisplayName "AI-STT-Cloud" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 443 `
    -RemoteAddress @(
        "13.107.0.0/16",     # Azure
        "20.190.0.0/16",     # Azure
        "52.0.0.0/8"         # AWS
    ) `
    -Program "C:\OpenClaw\Services\stt_engine.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Services"

# Twilio Service - Twilio endpoints
New-NetFirewallRule -DisplayName "AI-TWILIO-API" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 443 `
    -RemoteAddress @(
        "54.172.0.0/16",     # Twilio
        "54.244.0.0/16"      # Twilio
    ) `
    -Program "C:\OpenClaw\Services\twilio_service.exe" `
    -Action Allow `
    -Profile Domain,Private `
    -Group "AI Agent Services"

# Agent Loops - Internal communication
for ($i = 1; $i -le 15; $i++) {
    New-NetFirewallRule -DisplayName "AI-AGENT-LOOP-$i" `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort (9000 + $i) `
        -RemoteAddress 10.2.0.0/23 `
        -Program "C:\OpenClaw\Agents\agent_loop_$i.exe" `
        -Action Allow `
        -Profile Domain,Private `
        -Group "AI Agent Loops"
}

# ============================================
# BLOCK RULES (Explicit Deny)
# ============================================

# Block all other inbound
New-NetFirewallRule -DisplayName "AI-BLOCK-Inbound-Default" `
    -Direction Inbound `
    -Action Block `
    -Profile Domain,Private,Public `
    -Group "AI Agent Security"

# Block SMB outbound (prevent lateral movement)
New-NetFirewallRule -DisplayName "AI-BLOCK-SMB-Out" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 445,139 `
    -Action Block `
    -Profile Domain,Private,Public `
    -Group "AI Agent Security"

# Block RDP outbound
New-NetFirewallRule -DisplayName "AI-BLOCK-RDP-Out" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 3389 `
    -Action Block `
    -Profile Domain,Private,Public `
    -Group "AI Agent Security"

# Block PowerShell remoting
New-NetFirewallRule -DisplayName "AI-BLOCK-WinRM-Out" `
    -Direction Outbound `
    -Protocol TCP `
    -RemotePort 5985,5986 `
    -Action Block `
    -Profile Domain,Private,Public `
    -Group "AI Agent Security"
```

### 3.2 Container-Based Micro-Segmentation

```yaml
# Docker Compose with Network Segmentation
# docker-compose.security.yml

version: '3.8'

networks:
  # Management Network
  mgmt_net:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.0.0/24
    labels:
      - "security.zone=management"
      - "security.level=critical"

  # Application Network
  app_net:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.1.0/24
    labels:
      - "security.zone=application"
      - "security.level=critical"

  # Service Network
  svc_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.2.0/24
    labels:
      - "security.zone=service"
      - "security.level=high"

  # Data Network
  data_net:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.3.0/24
    labels:
      - "security.zone=data"
      - "security.level=critical"

services:
  # ============================================
  # CORE AI SERVICES
  # ============================================
  
  gpt52-engine:
    image: openclaw/gpt52-engine:latest
    container_name: gpt52-core
    networks:
      - app_net
      - svc_net
    security_opt:
      - no-new-privileges:true
      - seccomp:./seccomp/gpt52.json
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    environment:
      - API_KEY_FILE=/run/secrets/gpt52_api_key
    secrets:
      - gpt52_api_key
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    labels:
      - "security.profile=ai-core"
      - "network.policy=restricted"

  soul-service:
    image: openclaw/soul-service:latest
    container_name: soul-core
    networks:
      - app_net
      - data_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=50m
    environment:
      - DB_HOST=vector-store
      - DB_PORT=5432
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    labels:
      - "security.profile=ai-core"

  identity-manager:
    image: openclaw/identity-manager:latest
    container_name: identity-svc
    networks:
      - app_net
      - data_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    environment:
      - DB_HOST=user-db
      - ENCRYPTION_KEY_FILE=/run/secrets/encryption_key
    secrets:
      - encryption_key
    labels:
      - "security.profile=identity"

  heartbeat-monitor:
    image: openclaw/heartbeat:latest
    container_name: heartbeat-svc
    networks:
      - app_net
      - mgmt_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - SIEM_HOST=siem-collector
      - SIEM_PORT=514
    labels:
      - "security.profile=monitoring"

  # ============================================
  # SERVICE INTEGRATIONS
  # ============================================

  gmail-service:
    image: openclaw/gmail-connector:latest
    container_name: gmail-svc
    networks:
      - svc_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    environment:
      - GOOGLE_CREDENTIALS_FILE=/run/secrets/google_credentials
    secrets:
      - google_credentials
    dns:
      - 8.8.8.8
      - 8.8.4.4
    labels:
      - "security.profile=external-api"
      - "egress.allowed=googleapis.com"

  browser-control:
    image: openclaw/browser-controller:latest
    container_name: browser-svc
    networks:
      - svc_net
    security_opt:
      - no-new-privileges:true
      - seccomp:./seccomp/browser.json
    cap_drop:
      - ALL
    shm_size: '2gb'
    tmpfs:
      - /tmp:noexec,nosuid,size=500m
    environment:
      - PROXY_HOST=proxy-internal
      - PROXY_PORT=3128
    labels:
      - "security.profile=browser"
      - "egress.allowed=restricted"

  tts-engine:
    image: openclaw/tts-engine:latest
    container_name: tts-svc
    networks:
      - svc_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - AZURE_SPEECH_KEY_FILE=/run/secrets/azure_speech_key
    secrets:
      - azure_speech_key
    labels:
      - "security.profile=audio"

  stt-engine:
    image: openclaw/stt-engine:latest
    container_name: stt-svc
    networks:
      - svc_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - AZURE_SPEECH_KEY_FILE=/run/secrets/azure_speech_key
    secrets:
      - azure_speech_key
    labels:
      - "security.profile=audio"

  twilio-service:
    image: openclaw/twilio-connector:latest
    container_name: twilio-svc
    networks:
      - svc_net
    ports:
      - "127.0.0.1:8080:8080"  # Localhost only
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    environment:
      - TWILIO_CREDENTIALS_FILE=/run/secrets/twilio_credentials
    secrets:
      - twilio_credentials
    labels:
      - "security.profile=external-api"

  cron-scheduler:
    image: openclaw/cron-scheduler:latest
    container_name: cron-svc
    networks:
      - app_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - AGENT_ENDPOINT=http://agent-loops:9000
    labels:
      - "security.profile=scheduler"

  # ============================================
  # AGENT LOOPS (15 instances)
  # ============================================
  
  agent-loop-1:
    image: openclaw/agent-loop:latest
    container_name: agent-loop-01
    networks:
      - app_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - LOOP_ID=1
      - PARENT_SERVICE=gpt52-engine
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    labels:
      - "security.profile=agent-loop"
      - "loop.id=1"

  agent-loop-2:
    image: openclaw/agent-loop:latest
    container_name: agent-loop-02
    networks:
      - app_net
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    environment:
      - LOOP_ID=2
      - PARENT_SERVICE=gpt52-engine
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    labels:
      - "security.profile=agent-loop"
      - "loop.id=2"

  # ... (loops 3-15 follow same pattern)

  # ============================================
  # DATA SERVICES
  # ============================================

  user-db:
    image: postgres:15-alpine
    container_name: user-database
    networks:
      - data_net
    volumes:
      - user_db_data:/var/lib/postgresql/data
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    labels:
      - "security.profile=database"
      - "data.classification=confidential"

  agent-state-db:
    image: redis:7-alpine
    container_name: agent-state
    networks:
      - data_net
    volumes:
      - agent_state_data:/data
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    command: redis-server --requirepass ${REDIS_PASSWORD}
    labels:
      - "security.profile=database"

  vector-store:
    image: qdrant/qdrant:latest
    container_name: vector-database
    networks:
      - data_net
    volumes:
      - vector_data:/qdrant/storage
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    environment:
      - QDRANT_API_KEY_FILE=/run/secrets/vector_api_key
    secrets:
      - vector_api_key
    labels:
      - "security.profile=database"
      - "data.classification=restricted"

  log-store:
    image: elasticsearch:8.11.0
    container_name: log-storage
    networks:
      - data_net
      - mgmt_net
    volumes:
      - log_data:/usr/share/elasticsearch/data
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    environment:
      - ELASTIC_PASSWORD_FILE=/run/secrets/elastic_password
      - xpack.security.enabled=true
    secrets:
      - elastic_password
    labels:
      - "security.profile=logging"

secrets:
  gpt52_api_key:
    file: ./secrets/gpt52_api_key.txt
  encryption_key:
    file: ./secrets/encryption_key.txt
  google_credentials:
    file: ./secrets/google_credentials.json
  azure_speech_key:
    file: ./secrets/azure_speech_key.txt
  twilio_credentials:
    file: ./secrets/twilio_credentials.json
  db_password:
    file: ./secrets/db_password.txt
  vector_api_key:
    file: ./secrets/vector_api_key.txt
  elastic_password:
    file: ./secrets/elastic_password.txt

volumes:
  user_db_data:
    driver: local
  agent_state_data:
    driver: local
  vector_data:
    driver: local
  log_data:
    driver: local
```

### 3.3 Kubernetes Network Policies

```yaml
# Kubernetes Network Policies for AI Agent System
# network-policies.yaml

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: openclaw-ai
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# ============================================
# GPT-5.2 ENGINE POLICY
# ============================================
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gpt52-engine-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      app: gpt52-engine
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from agent loops
    - from:
        - podSelector:
            matchLabels:
              app: agent-loop
      ports:
        - protocol: TCP
          port: 8080
    # Allow from soul service
    - from:
        - podSelector:
            matchLabels:
              app: soul-service
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Allow HTTPS to OpenAI API
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              app: egress-gateway
      ports:
        - protocol: TCP
          port: 443
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
---
# ============================================
# AGENT LOOP POLICIES
# ============================================
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-loop-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      app: agent-loop
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from cron scheduler
    - from:
        - podSelector:
            matchLabels:
              app: cron-scheduler
      ports:
        - protocol: TCP
          port: 9000
  egress:
    # Allow to GPT-5.2 engine
    - to:
        - podSelector:
            matchLabels:
              app: gpt52-engine
      ports:
        - protocol: TCP
          port: 8080
    # Allow to services
    - to:
        - podSelector:
            matchLabels:
              tier: services
      ports:
        - protocol: TCP
          port: 80
        - protocol: TCP
          port: 443
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
---
# ============================================
# SERVICE TIER POLICIES
# ============================================
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gmail-service-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      app: gmail-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: application
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Restrict to Google API only
    - to:
        - ipBlock:
            cidr: 142.250.0.0/15
        - ipBlock:
            cidr: 172.217.0.0/16
        - ipBlock:
            cidr: 216.58.192.0/19
      ports:
        - protocol: TCP
          port: 443
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: twilio-service-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      app: twilio-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from application tier
    - from:
        - podSelector:
            matchLabels:
              tier: application
      ports:
        - protocol: TCP
          port: 8080
    # Allow Twilio webhooks
    - from:
        - ipBlock:
            cidr: 54.172.0.0/16
        - ipBlock:
            cidr: 54.244.0.0/16
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Restrict to Twilio API
    - to:
        - ipBlock:
            cidr: 54.172.0.0/16
        - ipBlock:
            cidr: 54.244.0.0/16
      ports:
        - protocol: TCP
          port: 443
---
# ============================================
# DATABASE TIER POLICIES
# ============================================
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-tier-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from application tier only
    - from:
        - podSelector:
            matchLabels:
              tier: application
      ports:
        - protocol: TCP
          port: 5432    # PostgreSQL
        - protocol: TCP
          port: 6379    # Redis
        - protocol: TCP
          port: 6333    # Qdrant
        - protocol: TCP
          port: 9200    # Elasticsearch
  egress:
    # Databases should not initiate outbound
    []
---
# ============================================
# MONITORING TIER POLICIES
# ============================================
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: monitoring-policy
  namespace: openclaw-ai
spec:
  podSelector:
    matchLabels:
      tier: monitoring
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from management network
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
  egress:
    # Allow to all pods for metrics collection
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 9090    # Prometheus
        - protocol: TCP
          port: 9100    # Node Exporter
```

---

## 4. Firewall Rules & Policies

### 4.1 Perimeter Firewall Rules (Palo Alto/Cisco ASA)

```bash
# ============================================
# PERIMETER FIREWALL CONFIGURATION
# AI Agent System - Security Rules
# ============================================

# Security Zones
configure terminal

# Define Security Zones
zone-security MANAGEMENT
zone-security DMZ
zone-security APPLICATION
zone-security SERVICES
zone-security DATABASE
zone-security INTERNET

# Assign interfaces to zones
interface GigabitEthernet0/0
  zone-member security INTERNET

interface GigabitEthernet0/1
  zone-member security DMZ

interface GigabitEthernet0/2
  zone-member security APPLICATION

interface GigabitEthernet0/3
  zone-member security SERVICES

interface GigabitEthernet1/0
  zone-member security DATABASE

interface GigabitEthernet1/1
  zone-member security MANAGEMENT

# ============================================
# CLASS-MAPS (Traffic Classification)
# ============================================

class-map type inspect match-any AI-HTTPS-TRAFFIC
  match protocol https
  match port tcp eq 443

class-map type inspect match-any AI-API-TRAFFIC
  match protocol http
  match port tcp eq 80
  match port tcp eq 8080
  match port tcp eq 8443

class-map type inspect match-any AI-DATABASE-TRAFFIC
  match port tcp eq 5432
  match port tcp eq 3306
  match port tcp eq 27017
  match port tcp eq 6379

class-map type inspect match-any AI-MONITORING-TRAFFIC
  match port tcp eq 514
  match port udp eq 514
  match port tcp eq 1514
  match port udp eq 1514

class-map type inspect match-any AI-VOICE-TRAFFIC
  match protocol sip
  match protocol rtp
  match port udp range 10000 20000

class-map type inspect match-any AI-BLOCKED-TRAFFIC
  match protocol smb
  match protocol netbios
  match port tcp eq 445
  match port tcp eq 139
  match port tcp eq 3389
  match port tcp eq 23
  match port tcp eq 21

# ============================================
# POLICY-MAPS (Inspection Policies)
# ============================================

policy-map type inspect AI-INSPECT-POLICY
  class AI-HTTPS-TRAFFIC
    inspect
    log
  class AI-API-TRAFFIC
    inspect
    log
  class AI-DATABASE-TRAFFIC
    pass
    log
  class AI-MONITORING-TRAFFIC
    pass
    log
  class AI-VOICE-TRAFFIC
    inspect
    log
  class AI-BLOCKED-TRAFFIC
    drop
    log
  class class-default
    drop
    log

# ============================================
# ZONE-PAIR POLICIES
# ============================================

# INTERNET to DMZ
zone-pair security INTERNET-DMZ source INTERNET destination DMZ
  service-policy type inspect AI-INSPECT-POLICY

# DMZ to APPLICATION
zone-pair security DMZ-APP source DMZ destination APPLICATION
  service-policy type inspect AI-INSPECT-POLICY

# APPLICATION to SERVICES
zone-pair security APP-SERVICES source APPLICATION destination SERVICES
  service-policy type inspect AI-INSPECT-POLICY

# SERVICES to DATABASE
zone-pair security SVC-DATA source SERVICES destination DATABASE
  service-policy type inspect AI-INSPECT-POLICY

# APPLICATION to DATABASE (direct)
zone-pair security APP-DATA source APPLICATION destination DATABASE
  service-policy type inspect AI-INSPECT-POLICY

# MANAGEMENT to ALL (admin access)
zone-pair security MGMT-ALL source MANAGEMENT destination ANY
  service-policy type inspect AI-INSPECT-POLICY

# Block all other inter-zone traffic
zone-pair security BLOCK-ALL source ANY destination ANY
  service-policy type inspect BLOCK-ALL-POLICY

# ============================================
# ACCESS CONTROL LISTS
# ============================================

# Extended ACL for GPT-5.2 API Access
ip access-list extended GPT52-API-ACL
  # Allow HTTPS to OpenAI API
  permit tcp 10.2.0.0 0.0.1.255 host api.openai.com eq 443
  permit tcp 10.2.0.0 0.0.1.255 host api.anthropic.com eq 443
  # Deny all other outbound
  deny ip 10.2.0.0 0.0.1.255 any log

# Extended ACL for Gmail Service
ip access-list extended GMAIL-SVC-ACL
  # Allow HTTPS to Google APIs
  permit tcp 10.3.0.0 0.0.0.127 142.250.0.0 0.1.255.255 eq 443
  permit tcp 10.3.0.0 0.0.0.127 172.217.0.0 0.0.255.255 eq 443
  permit tcp 10.3.0.0 0.0.0.127 216.58.192.0 0.0.31.255 eq 443
  # Allow Gmail IMAP/SMTP
  permit tcp 10.3.0.0 0.0.0.127 host imap.gmail.com eq 993
  permit tcp 10.3.0.0 0.0.0.127 host smtp.gmail.com eq 587
  # Deny all other
  deny ip 10.3.0.0 0.0.0.127 any log

# Extended ACL for Twilio Service
ip access-list extended TWILIO-SVC-ACL
  # Allow HTTPS to Twilio API
  permit tcp 10.3.0.128 0.0.0.127 54.172.0.0 0.0.255.255 eq 443
  permit tcp 10.3.0.128 0.0.0.127 54.244.0.0 0.0.255.255 eq 443
  # Allow SIP/RTP
  permit udp 10.3.0.128 0.0.0.127 54.172.0.0 0.0.255.255 range 10000 20000
  permit udp 10.3.0.128 0.0.0.127 54.244.0.0 0.0.255.255 range 10000 20000
  # Deny all other
  deny ip 10.3.0.128 0.0.0.127 any log

# Extended ACL for Database Tier
ip access-list extended DATABASE-ACL
  # Allow PostgreSQL from Application
  permit tcp 10.2.0.0 0.0.1.255 10.4.0.0 0.0.0.127 eq 5432
  # Allow Redis from Application
  permit tcp 10.2.0.0 0.0.1.255 10.4.0.128 0.0.0.127 eq 6379
  # Allow Qdrant from Application
  permit tcp 10.2.0.0 0.0.1.255 10.4.2.0 0.0.0.127 eq 6333
  # Allow Elasticsearch from Management
  permit tcp 10.0.1.0 0.0.0.255 10.4.1.128 0.0.0.127 eq 9200
  # Deny all other
  deny ip any 10.4.0.0 0.0.3.255 log

# ============================================
# APPLY ACLs TO INTERFACES
# ============================================

interface GigabitEthernet0/2
  ip access-group GPT52-API-ACL out

interface GigabitEthernet0/3.100
  ip access-group GMAIL-SVC-ACL out

interface GigabitEthernet0/3.101
  ip access-group TWILIO-SVC-ACL out

interface GigabitEthernet1/0
  ip access-group DATABASE-ACL in

end
write memory
```

### 4.2 Next-Generation Firewall Application Rules

```xml
<!-- Palo Alto NGFW Security Rules for AI Agent System -->
<!-- Export from Panorama/PanOS -->

<security-rules>
  
  <!-- Rule 1: Allow Management Access -->
  <entry name="Allow-Management-Access">
    <from>
      <member>MANAGEMENT</member>
    </from>
    <to>
      <member>ANY</member>
    </to>
    <source>
      <member>10.0.0.0/24</member>
    </source>
    <destination>
      <member>ANY</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>ssh</member>
      <member>ssl</member>
      <member>rdp</member>
      <member>snmp</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
  </entry>

  <!-- Rule 2: GPT-5.2 Engine API Access -->
  <entry name="Allow-GPT52-API">
    <from>
      <member>APPLICATION</member>
    </from>
    <to>
      <member>INTERNET</member>
    </to>
    <source>
      <member>10.2.0.0/23</member>
    </source>
    <destination>
      <member>OpenAI-API-Endpoints</member>
      <member>Anthropic-API-Endpoints</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>computer-and-internet-info</member>
    </category>
    <application>
      <member>ssl</member>
      <member>web-browsing</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
  </entry>

  <!-- Rule 3: Gmail Service Access -->
  <entry name="Allow-Gmail-Service">
    <from>
      <member>SERVICES</member>
    </from>
    <to>
      <member>INTERNET</member>
    </to>
    <source>
      <member>10.3.0.0/25</member>
    </source>
    <destination>
      <member>Google-IP-Ranges</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>gmail</member>
      <member>gmail-base</member>
      <member>google-base</member>
      <member>ssl</member>
      <member>imap</member>
      <member>smtp</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
  </entry>

  <!-- Rule 4: Twilio Service Access -->
  <entry name="Allow-Twilio-Service">
    <from>
      <member>SERVICES</member>
    </from>
    <to>
      <member>INTERNET</member>
    </to>
    <source>
      <member>10.3.0.128/25</member>
    </source>
    <destination>
      <member>Twilio-IP-Ranges</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>ssl</member>
      <member>sip</member>
      <member>rtp</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
  </entry>

  <!-- Rule 5: Browser Control - Restricted -->
  <entry name="Allow-Browser-Control">
    <from>
      <member>SERVICES</member>
    </from>
    <to>
      <member>INTERNET</member>
    </to>
    <source>
      <member>10.3.1.0/25</member>
    </source>
    <destination>
      <member>any</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>ssl</member>
      <member>web-browsing</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
    <option>
      <disable-server-response-inspection>no</disable-server-response-inspection>
    </option>
  </entry>

  <!-- Rule 6: TTS/STT Cloud Services -->
  <entry name="Allow-Audio-Services">
    <from>
      <member>SERVICES</member>
    </from>
    <to>
      <member>INTERNET</member>
    </to>
    <source>
      <member>10.3.1.128/25</member>
    </source>
    <destination>
      <member>Azure-IP-Ranges</member>
      <member>AWS-IP-Ranges</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>ssl</member>
      <member>web-browsing</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
    <profile-setting>
      <group>
        <member>AI-Agent-Security-Profile</member>
      </group>
    </profile-setting>
  </entry>

  <!-- Rule 7: Inter-Service Communication -->
  <entry name="Allow-Inter-Service">
    <from>
      <member>APPLICATION</member>
      <member>SERVICES</member>
    </from>
    <to>
      <member>APPLICATION</member>
      <member>SERVICES</member>
      <member>DATABASE</member>
    </to>
    <source>
      <member>10.2.0.0/22</member>
    </source>
    <destination>
      <member>10.2.0.0/22</member>
      <member>10.4.0.0/23</member>
    </destination>
    <source-user>
      <member>known-user</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>any</member>
    </application>
    <service>
      <member>application-default</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>allow</action>
    <log-start>no</log-start>
    <log-end>yes</log-end>
  </entry>

  <!-- Rule 8: Deny High-Risk Applications -->
  <entry name="Deny-High-Risk-Apps">
    <from>
      <member>ANY</member>
    </from>
    <to>
      <member>ANY</member>
    </to>
    <source>
      <member>any</member>
    </source>
    <destination>
      <member>any</member>
    </destination>
    <source-user>
      <member>any</member>
    </source-user>
    <category>
      <member>high-risk</member>
      <member>malware</member>
      <member>phishing</member>
    </category>
    <application>
      <member>any</member>
    </application>
    <service>
      <member>any</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>deny</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
  </entry>

  <!-- Rule 9: Default Deny -->
  <entry name="Default-Deny">
    <from>
      <member>ANY</member>
    </from>
    <to>
      <member>ANY</member>
    </to>
    <source>
      <member>any</member>
    </source>
    <destination>
      <member>any</member>
    </destination>
    <source-user>
      <member>any</member>
    </source-user>
    <category>
      <member>any</member>
    </category>
    <application>
      <member>any</member>
    </application>
    <service>
      <member>any</member>
    </service>
    <hip-profiles>
      <member>any</member>
    </hip-profiles>
    <action>deny</action>
    <log-start>yes</log-start>
    <log-end>yes</log-end>
  </entry>

</security-rules>
```

---

## 5. Traffic Filtering Mechanisms

### 5.1 Deep Packet Inspection (DPI) Rules

```python
# AI Agent System - Traffic Filtering Configuration
# Suricata/Snort Rules

# ============================================
# GPT-5.2 API TRAFFIC INSPECTION
# ============================================

# Alert on suspicious API key patterns in outbound traffic
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Potential API Key Exfiltration";
    tls.sni; content:"api.openai.com";
    content:"sk-"; http_client_body; 
    metadata:impact_flag red, policy balanced-ips drop, policy security-ips drop;
    reference:url,https://platform.openai.com/docs/api-reference;
    classtype:credential-theft;
    sid:1000001;
    rev:1;
)

# Alert on unusual GPT API request patterns
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Abnormal GPT API Request Volume";
    tls.sni; content:"api.openai.com";
    detection_filter:track by_src, count 100, seconds 60;
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:attempted-dos;
    sid:1000002;
    rev:1;
)

# ============================================
# GMAIL SERVICE INSPECTION
# ============================================

# Alert on Gmail credential harvesting attempts
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Suspicious Gmail Authentication Pattern";
    tls.sni; content:"accounts.google.com";
    content:"password"; http_client_body; nocase;
    pcre:"/password[^&]{0,50}[&]/i";
    metadata:impact_flag red, policy balanced-ips drop, policy security-ips drop;
    classtype:credential-theft;
    sid:1000010;
    rev:1;
)

# Monitor Gmail IMAP access
alert tcp $HOME_NET any -> $EXTERNAL_NET 993 (
    msg:"AI-AGENT: Gmail IMAP Access Detected";
    flow:established,to_server;
    content:"imap.gmail.com"; tls.sni;
    metadata:impact_flag green, policy balanced-ips alert, policy security-ips alert;
    classtype:not-suspicious;
    sid:1000011;
    rev:1;
)

# ============================================
# TWILIO SERVICE INSPECTION
# ============================================

# Alert on potential toll fraud patterns
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Suspicious Twilio Voice Activity";
    tls.sni; content:"api.twilio.com";
    content:"To="; http_client_body; 
    content:!("+1"); http_client_body; 
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:trojan-activity;
    sid:1000020;
    rev:1;
)

# Monitor SIP traffic for anomalies
alert udp $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Unusual SIP Traffic Pattern";
    content:"INVITE"; startswith;
    detection_filter:track by_src, count 50, seconds 60;
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:attempted-dos;
    sid:1000021;
    rev:1;
)

# ============================================
# BROWSER CONTROL INSPECTION
# ============================================

# Block access to known malicious sites
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Browser Access to Malicious Domain";
    tls.sni; content:".onion"; endswith;
    metadata:impact_flag red, policy balanced-ips drop, policy security-ips drop;
    classtype:trojan-activity;
    sid:1000030;
    rev:1;
)

# Alert on cryptocurrency mining attempts
alert tls $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Potential Crypto Mining Activity";
    tls.sni; pcre:"/(coinhive|cryptoloot|jsecoin)\./i";
    metadata:impact_flag red, policy balanced-ips drop, policy security-ips drop;
    classtype:trojan-activity;
    sid:1000031;
    rev:1;
)

# ============================================
# INTERNAL TRAFFIC INSPECTION
# ============================================

# Detect lateral movement attempts
alert tcp $HOME_NET any -> $HOME_NET [445,139] (
    msg:"AI-AGENT: SMB Lateral Movement Detected";
    flow:to_server,established;
    content:"|FF|SMB"; startswith;
    detection_filter:track by_src, count 10, seconds 60;
    metadata:impact_flag red, policy balanced-ips drop, policy security-ips drop;
    classtype:trojan-activity;
    sid:1000040;
    rev:1;
)

# Alert on PowerShell remoting
alert tcp $HOME_NET any -> $HOME_NET [5985,5986] (
    msg:"AI-AGENT: PowerShell Remoting Detected";
    flow:to_server,established;
    content:"POST"; http_method;
    content:"/wsman"; http_uri;
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:trojan-activity;
    sid:1000041;
    rev:1;
)

# ============================================
# DATA EXFILTRATION DETECTION
# ============================================

# Alert on large outbound data transfers
alert tcp $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Potential Data Exfiltration";
    flow:to_server,established;
    detection_filter:track by_src, bytes > 100000000, seconds 300;
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:credential-theft;
    sid:1000050;
    rev:1;
)

# Detect database dump patterns
alert tcp $HOME_NET any -> $EXTERNAL_NET any (
    msg:"AI-AGENT: Potential Database Dump Exfiltration";
    flow:to_server,established;
    content:"SELECT"; http_client_body; nocase;
    content:"FROM"; http_client_body; nocase;
    detection_filter:track by_src, bytes > 10000000, seconds 60;
    metadata:impact_flag red, policy balanced-ips alert, policy security-ips drop;
    classtype:credential-theft;
    sid:1000051;
    rev:1;
)
```

### 5.2 Application Layer Gateway (ALG) Configuration

```yaml
# Application Layer Gateway Configuration
# For AI Agent System Protocol Handling

alg_configuration:
  
  # ============================================
  # HTTPS/TLS INSPECTION
  # ============================================
  tls_inspection:
    enabled: true
    mode: "decrypt-and-inspect"
    certificate_authority: "internal-ca"
    
    # Bypass list for sensitive services
    bypass_list:
      - category: "financial-services"
      - category: "healthcare"
      - domains:
          - "*.banking.com"
          - "*.healthcare.gov"
    
    # AI-specific inspection rules
    inspection_rules:
      - name: "gpt-api-inspection"
        match_sni: "api.openai.com"
        actions:
          - log_request_headers: true
          - log_response_headers: true
          - inspect_body: true
          - max_body_size: 1048576  # 1MB
      
      - name: "gmail-inspection"
        match_sni: "*.google.com"
        actions:
          - log_request_headers: true
          - log_response_headers: false
          - inspect_body: false
      
      - name: "twilio-inspection"
        match_sni: "*.twilio.com"
        actions:
          - log_request_headers: true
          - log_response_headers: true
          - inspect_body: true
          - mask_sensitive: true

  # ============================================
  # SIP/RTP FOR TWILIO VOICE
  # ============================================
  sip_alg:
    enabled: true
    
    security_checks:
      - validate_headers: true
      - max_header_length: 4096
      - max_body_length: 65536
      - block_anonymous_calls: false
      - rate_limit: 100  # requests per second
    
    nat_handling:
      - rewrite_sdp: true
      - rewrite_contact: true
      - rtp_port_range: [10000, 20000]
    
    logging:
      - log_calls: true
      - log_registrations: true
      - log_errors: true

  # ============================================
  # WEBSOCKET FOR REAL-TIME FEATURES
  # ============================================
  websocket_inspection:
    enabled: true
    
    connection_limits:
      max_connections_per_client: 10
      max_message_size: 10485760  # 10MB
      idle_timeout: 300  # seconds
    
    frame_inspection:
      - validate_utf8: true
      - mask_check: true
      - block_extensions: ["x-custom-compression"]
    
    rate_limiting:
      messages_per_second: 100
      bytes_per_second: 1048576  # 1MB/s

  # ============================================
  # JSON-RPC FOR INTERNAL APIs
  # ============================================
  json_rpc_inspection:
    enabled: true
    
    validation:
      - schema_validation: true
      - max_request_size: 1048576  # 1MB
      - max_response_size: 10485760  # 10MB
      - max_batch_size: 100
    
    security:
      - block_recursive_calls: true
      - max_nesting_depth: 10
      - rate_limit: 1000  # requests per minute
      - authentication_required: true
```

---

## 6. East-West Traffic Control

### 6.1 Internal Traffic Segmentation

```yaml
# East-West Traffic Control Configuration
# Internal Network Segmentation Policies

east_west_policies:
  
  # ============================================
  # TIER 0: MANAGEMENT TO ALL TIERS
  # ============================================
  management_access:
    source_zone: MANAGEMENT
    destination_zones: [ALL]
    
    allowed_protocols:
      - protocol: SSH
        ports: [22]
        require_mfa: true
        session_timeout: 3600
      
      - protocol: RDP
        ports: [3389]
        require_mfa: true
        session_timeout: 1800
      
      - protocol: SNMP
        ports: [161, 162]
        version: v3
        encryption: required
      
      - protocol: SYSLOG
        ports: [514, 1514]
        encryption: TLS
    
    restrictions:
      - time_based_access:
          allowed_hours: "08:00-18:00"
          timezone: "UTC"
      
      - source_ip_whitelist:
          - "10.0.0.0/24"
          - "10.0.1.0/24"
      
      - concurrent_sessions_per_user: 3

  # ============================================
  # TIER 1: APPLICATION TO SERVICE TIER
  # ============================================
  app_to_service:
    source_zone: APPLICATION
    destination_zone: SERVICES
    
    service_mesh:
      enabled: true
      mTLS: required
      certificate_validation: strict
    
    allowed_connections:
      - source: "gpt52-engine"
        destinations: ["gmail-service", "browser-control", "tts-engine", "stt-engine", "twilio-service"]
        protocol: HTTPS
        ports: [443, 8080]
        rate_limit: 1000  # requests/minute
      
      - source: "soul-service"
        destinations: ["vector-store"]
        protocol: gRPC
        ports: [6333]
        encryption: required
      
      - source: "identity-manager"
        destinations: ["user-db", "agent-state-db"]
        protocol: PostgreSQL/Redis
        ports: [5432, 6379]
        encryption: TLS
      
      - source: "heartbeat-monitor"
        destinations: ["log-store"]
        protocol: SYSLOG/Beats
        ports: [514, 5044]
    
    denied_connections:
      - protocol: SMB
        ports: [445, 139]
        action: LOG_AND_DROP
      
      - protocol: RDP
        ports: [3389]
        action: DROP
      
      - protocol: WinRM
        ports: [5985, 5986]
        action: DROP

  # ============================================
  # TIER 2: SERVICE TO DATA TIER
  # ============================================
  service_to_data:
    source_zone: SERVICES
    destination_zone: DATABASE
    
    database_proxy:
      enabled: true
      type: "pgbouncer"
      connection_pooling: true
      max_connections: 100
      query_logging: true
    
    allowed_connections:
      - source: "gmail-service"
        destination: "user-db"
        protocol: PostgreSQL
        ports: [5432]
        operations: [SELECT, INSERT, UPDATE]
        row_level_security: true
      
      - source: "twilio-service"
        destination: "agent-state-db"
        protocol: Redis
        ports: [6379]
        operations: [GET, SET, EXPIRE]
        key_prefixes: ["twilio:", "call:"]
      
      - source: "tts-engine"
        destination: "vector-store"
        protocol: HTTP
        ports: [6333]
        operations: [SEARCH, RETRIEVE]
        rate_limit: 100  # queries/minute
    
    audit_requirements:
      - log_all_queries: true
      - log_slow_queries: true  # > 1 second
      - log_failed_connections: true
      - retention_days: 90

  # ============================================
  # TIER 3: AGENT LOOP ISOLATION
  # ============================================
  agent_loop_segmentation:
    description: "15 Agent Loops - Strict Isolation"
    
    loop_networks:
      - loop_id: 1
        network: "10.3.2.128/28"
        vlan: 305
      - loop_id: 2
        network: "10.3.2.144/28"
        vlan: 305
      - loop_id: 3
        network: "10.3.2.160/28"
        vlan: 305
      # ... loops 4-15 follow same pattern
    
    inter_loop_policy:
      action: DENY
      logging: true
      alert_on_violation: true
    
    loop_to_core_policy:
      allowed_protocols:
        - protocol: gRPC
          ports: [9000-9015]
          authentication: mTLS
          rate_limit: 100  # calls/minute per loop
      
      restrictions:
        - max_concurrent_connections: 5
        - connection_timeout: 300
        - idle_timeout: 60

  # ============================================
  # ZERO TRUST EAST-WEST
  # ============================================
  zero_trust_policy:
    principle: "Never trust, always verify"
    
    authentication:
      - method: mTLS
        certificate_validation: strict
        revocation_check: OCSP
      
      - method: JWT
        issuer: "internal-auth-server"
        audience: "ai-agent-services"
        max_age: 3600
    
    authorization:
      - model: RBAC
        roles:
          - name: "agent-executor"
            permissions: ["execute:agent", "read:state"]
          - name: "service-connector"
            permissions: ["connect:external", "read:config"]
          - name: "data-accessor"
            permissions: ["read:db", "write:db:limited"]
      
      - model: ABAC
        attributes:
          - source_tier
          - destination_tier
          - time_of_day
          - data_classification
    
    continuous_verification:
      - behavioral_analysis: true
      - anomaly_detection: true
      - session_revalidation: 1800  # seconds
```

### 6.2 Service Mesh Configuration

```yaml
# Istio Service Mesh Configuration
# East-West Traffic Management

apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: ai-agent-mesh
spec:
  profile: default
  components:
    pilot:
      k8s:
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
    
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
      tracing:
        sampling: 100.0
        zipkin:
          address: jaeger-collector.monitoring:9411
    
    enableAutoMtls: true
    
    # Outbound traffic policy
    outboundTrafficPolicy:
      mode: REGISTRY_ONLY
    
    # Access logging
    accessLogFile: /dev/stdout
    accessLogEncoding: JSON
    
    # Extension providers
    extensionProviders:
      - name: oauth2-proxy
        envoyOauth2:
          service: oauth2-proxy.auth.svc.cluster.local
          port: 4180

---
# Peer Authentication - mTLS Required
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: openclaw-ai
spec:
  mtls:
    mode: STRICT

---
# Destination Rules with Traffic Policies
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: gpt52-engine-dr
  namespace: openclaw-ai
spec:
  host: gpt52-engine
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
    
    loadBalancer:
      simple: LEAST_CONN
      localityLbSetting:
        enabled: true
        failover:
          - from: us-east-1
            to: us-west-2
    
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    
    tls:
      mode: ISTIO_MUTUAL

---
# Circuit Breaker Configuration
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: service-circuit-breakers
  namespace: openclaw-ai
spec:
  host: "*.svc.cluster.local"
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 50
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 25
        http2MaxRequests: 50
    
    outlierDetection:
      consecutiveErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 10

---
# Retry Policy for Agent Loops
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: agent-loop-retries
  namespace: openclaw-ai
spec:
  hosts:
    - agent-loops
  http:
    - route:
        - destination:
            host: agent-loops
      retries:
        attempts: 3
        perTryTimeout: 2s
        retryOn: 5xx,connect-failure,refused-stream
      timeout: 10s
      fault:
        delay:
          percentage:
            value: 0.1
          fixedDelay: 5s
```

---

## 7. North-South Traffic Control

### 7.1 Ingress Traffic Management

```yaml
# North-South Traffic Control
# Ingress and Egress Gateway Configuration

apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: ai-agent-gateway
  namespace: openclaw-ai
spec:
  selector:
    istio: ingressgateway
  servers:
    # HTTPS Traffic
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: ai-agent-tls-cert
        minProtocolVersion: TLSV1_3
        cipherSuites:
          - TLS_AES_256_GCM_SHA384
          - TLS_CHACHA20_POLY1305_SHA256
      hosts:
        - "ai-agent.company.com"
        - "api.ai-agent.company.com"
    
    # mTLS for Internal Services
    - port:
        number: 443
        name: https-mtls
        protocol: HTTPS
      tls:
        mode: MUTUAL
        credentialName: ai-agent-mtls-cert
      hosts:
        - "internal.ai-agent.company.com"

---
# VirtualService for External Access
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-agent-external
  namespace: openclaw-ai
spec:
  hosts:
    - "api.ai-agent.company.com"
  gateways:
    - ai-agent-gateway
  http:
    # Rate limiting at edge
    - match:
        - uri:
            prefix: /api/v1/
      route:
        - destination:
            host: gpt52-engine
            port:
              number: 8080
      corsPolicy:
        allowOrigins:
          - exact: "https://app.company.com"
        allowMethods: [GET, POST, PUT, DELETE]
        allowHeaders: [authorization, content-type]
        allowCredentials: true
        maxAge: "24h"
      
      # Request mirroring for testing
      mirror:
        host: gpt52-engine-canary
        port:
          number: 8080
      mirrorPercentage:
        value: 10.0

---
# Egress Gateway Configuration
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: ai-agent-egress
  namespace: openclaw-ai
spec:
  selector:
    istio: egressgateway
  servers:
    - port:
        number: 443
        name: https
        protocol: TLS
      tls:
        mode: ISTIO_MUTUAL
      hosts:
        - "api.openai.com"
        - "api.anthropic.com"
        - "*.googleapis.com"
        - "api.twilio.com"

---
# ServiceEntry for External Services
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: openai-api
  namespace: openclaw-ai
spec:
  hosts:
    - api.openai.com
  ports:
    - number: 443
      name: https
      protocol: TLS
  resolution: DNS
  location: MESH_EXTERNAL
  exportTo:
    - "."

---
# Egress Traffic Policy
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: openai-egress
  namespace: openclaw-ai
spec:
  hosts:
    - api.openai.com
  tls:
    - match:
        - port: 443
          sniHosts:
            - api.openai.com
      route:
        - destination:
            host: ai-agent-egress
            port:
              number: 443
          weight: 100

---
# Sidecar Egress Configuration
apiVersion: networking.istio.io/v1beta1
kind: Sidecar
metadata:
  name: default
  namespace: openclaw-ai
spec:
  egress:
    # Only allow egress to specified hosts
    - hosts:
        - "./*"
        - "istio-system/*"
    
    # External service whitelist
    - captureMode: NONE
      hosts:
        - "./api.openai.com"
        - "./api.anthropic.com"
        - "./*.googleapis.com"
        - "./api.twilio.com"
        - "./imap.gmail.com"
        - "./smtp.gmail.com"
  
  outboundTrafficPolicy:
    mode: REGISTRY_ONLY
```

### 7.2 API Gateway Security

```yaml
# Kong API Gateway Configuration
# North-South API Security

_format_version: "3.0"

services:
  # ============================================
  # GPT-5.2 API Service
  # ============================================
  - name: gpt52-api
    url: http://gpt52-engine:8080
    
    routes:
      - name: gpt52-routes
        paths:
          - /api/v1/gpt
        methods:
          - POST
          - GET
        strip_path: true
    
    plugins:
      # Rate Limiting
      - name: rate-limiting
        config:
          minute: 60
          hour: 1000
          policy: redis
          redis_host: redis-cache
          fault_tolerant: true
          hide_client_headers: false
      
      # Authentication
      - name: jwt
        config:
          uri_param_names: []
          cookie_names: []
          key_claim_name: iss
          secret_is_base64: false
          claims_to_verify:
            - exp
          maximum_expiration: 3600
      
      # Request Validation
      - name: request-validator
        config:
          body_schema: '{"type":"object","properties":{"prompt":{"type":"string","maxLength":4000},"max_tokens":{"type":"integer","maximum":4096}},"required":["prompt"]}'
          allowed_content_types:
            - application/json
      
      # Response Transformation
      - name: response-transformer
        config:
          remove:
            headers:
              - x-internal-header
              - server
      
      # Bot Detection
      - name: bot-detection
        config:
          allow:
            - "Mozilla/5.0"
          deny:
            - "curl/"
            - "wget/"
      
      # IP Restriction
      - name: ip-restriction
        config:
          allow:
            - 10.0.0.0/8
            - 172.16.0.0/12
          deny:
            - 192.168.1.100

  # ============================================
  # Webhook Service
  # ============================================
  - name: webhook-service
    url: http://twilio-service:8080
    
    routes:
      - name: twilio-webhook
        paths:
          - /webhooks/twilio
        methods:
          - POST
    
    plugins:
      # IP Restriction for Twilio
      - name: ip-restriction
        config:
          allow:
            - 54.172.0.0/16
            - 54.244.0.0/16
      
      # Request Signature Validation
      - name: request-validator
        config:
          secret_is_base64: false
          allowed_content_types:
            - application/x-www-form-urlencoded
      
      # Webhook Verification
      - name: twilio-signature
        config:
          auth_token: ${TWILIO_AUTH_TOKEN}

  # ============================================
  # Global Plugins
  # ============================================
plugins:
  # CORS
  - name: cors
    config:
      origins:
        - "https://app.company.com"
      methods:
        - GET
        - POST
        - PUT
        - DELETE
      headers:
        - Authorization
        - Content-Type
      max_age: 3600
      credentials: true
  
  # Logging
  - name: file-log
    config:
      path: /var/log/kong/access.log
      reopen: false
  
  # Prometheus Metrics
  - name: prometheus
    config:
      per_consumer: true
      status_code_metrics: true
      latency_metrics: true
      bandwidth_metrics: true
  
  # Datadog APM
  - name: datadog
    config:
      host: datadog-agent
      port: 8126
      service_name: kong-ai-agent

consumers:
  - username: agent-loop-client
    jwt_credentials:
      - algorithm: RS256
        rsa_public_key: ${JWT_PUBLIC_KEY}
        key: agent-loop-key
  
  - username: external-api-client
    jwt_credentials:
      - algorithm: RS256
        rsa_public_key: ${JWT_PUBLIC_KEY}
        key: external-api-key
```

---

## 8. Network Monitoring Architecture

### 8.1 SIEM Integration

```yaml
# Network Monitoring & SIEM Configuration
# Splunk/Elastic Security Integration

siem_configuration:
  
  # ============================================
  # DATA COLLECTION
  # ============================================
  data_sources:
    # Firewall Logs
    - name: perimeter-firewall
      type: syslog
      protocol: tcp
      port: 514
      format: cef
      sourcetype: cef
      index: network
    
    # IDS/IPS Alerts
    - name: suricata-ids
      type: file
      path: /var/log/suricata/eve.json
      sourcetype: suricata
      index: security
    
    # NetFlow/sFlow
    - name: netflow-collector
      type: netflow
      port: 2055
      version: 9
      index: network
    
    # Windows Event Logs
    - name: windows-security
      type: wineventlog
      channels:
        - Security
        - System
        - Application
        - Microsoft-Windows-Windows Firewall With Advanced Security/Firewall
      index: windows
    
    # Application Logs
    - name: ai-agent-logs
      type: http
      endpoint: http://splunk-hec:8088/services/collector/event
      token: ${SPLUNK_HEC_TOKEN}
      index: ai-agent
    
    # DNS Logs
    - name: dns-queries
      type: dns
      format: dnstap
      port: 6000
      index: dns

  # ============================================
  # CORRELATION RULES
  # ============================================
  correlation_rules:
    # AI Agent Anomaly Detection
    - name: abnormal_gpt_api_usage
      description: "Detects unusual GPT API consumption patterns"
      search: |
        index=ai-agent sourcetype=gpt-api
        | stats count, sum(tokens_used) as total_tokens by src_ip, user
        | where count > 100 OR total_tokens > 100000
        | eval severity=case(count > 500, "critical", count > 200, "high", 1=1, "medium")
      alert:
        condition: count > 0
        throttle: 1h
        actions:
          - email: security@company.com
          - webhook: https://pagerduty.com/integration
    
    # Lateral Movement Detection
    - name: lateral_movement_smb
      description: "Detects SMB connections between internal hosts"
      search: |
        index=network sourcetype=firewall
        dest_port IN (445, 139)
        src_zone=internal dest_zone=internal
        | stats dc(dest_ip) as unique_targets by src_ip
        | where unique_targets > 10
      alert:
        condition: unique_targets > 10
        severity: high
    
    # Data Exfiltration
    - name: potential_data_exfil
      description: "Detects large outbound data transfers"
      search: |
        index=network sourcetype=netflow
        | eval direction=if(src_ip matches "10.*", "outbound", "inbound")
        | where direction="outbound"
        | stats sum(bytes) as total_bytes by src_ip, dest_ip
        | where total_bytes > 1000000000
        | eval gb=round(total_bytes/1073741824, 2)
      alert:
        condition: gb > 1
        severity: critical
    
    # Agent Loop Failure Cascade
    - name: agent_loop_cascade_failure
      description: "Detects multiple agent loop failures"
      search: |
        index=ai-agent sourcetype=agent-loop
        status=error OR status=failed
        | stats dc(loop_id) as failed_loops by _time span=5m
        | where failed_loops > 5
      alert:
        condition: failed_loops > 5
        severity: critical
        actions:
          - webhook: https://automation.company.com/remediate

  # ============================================
  # DASHBOARDS
  # ============================================
  dashboards:
    - name: ai-agent-security-overview
      panels:
        - title: "GPT API Request Volume"
          type: timechart
          search: 'index=ai-agent sourcetype=gpt-api | timechart count by endpoint'
        
        - title: "Agent Loop Health"
          type: single
          search: 'index=ai-agent sourcetype=agent-loop status=success | stats count'
        
        - title: "Network Traffic by Zone"
          type: pie
          search: 'index=network | stats sum(bytes) by src_zone'
        
        - title: "Security Alerts"
          type: table
          search: 'index=security | table _time, severity, signature, src_ip, dest_ip'

  # ============================================
  # THREAT INTELLIGENCE
  # ============================================
  threat_intel:
    feeds:
      - name: misp
        url: https://misp.company.com
        api_key: ${MISP_API_KEY}
        types: [ip, domain, hash, url]
      
      - name: alienvault-otx
        url: https://otx.alienvault.com
        api_key: ${OTX_API_KEY}
        types: [ip, domain]
      
      - name: abuse-ch
        url: https://urlhaus-api.abuse.ch
        types: [url, hash]
    
    lookups:
      - name: threat_intel_ip
        fields: [src_ip, dest_ip]
        output: [threat_category, threat_score]
      
      - name: threat_intel_domain
        fields: [dns_query, http_host]
        output: [threat_category, threat_score]
```

### 8.2 Network Behavior Analytics

```python
# Network Behavior Analytics (NBA) Configuration
# Machine Learning-based Anomaly Detection

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

class AIAgentNetworkAnalyzer:
    """
    Network Behavior Analytics for AI Agent System
    Detects anomalies in network traffic patterns
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_estimators=100
        )
        self.baseline_established = False
        
    def extract_features(self, flow_data):
        """
        Extract behavioral features from network flows
        """
        features = {
            # Volume metrics
            'bytes_per_second': flow_data['bytes'] / flow_data['duration'],
            'packets_per_second': flow_data['packets'] / flow_data['duration'],
            'avg_packet_size': flow_data['bytes'] / flow_data['packets'],
            
            # Temporal patterns
            'hour_of_day': pd.to_datetime(flow_data['timestamp']).hour,
            'day_of_week': pd.to_datetime(flow_data['timestamp']).dayofweek,
            'is_business_hours': 1 if 9 <= flow_data['hour'] <= 17 else 0,
            
            # Connection patterns
            'unique_dest_ports': len(set(flow_data['dest_ports'])),
            'unique_dest_ips': len(set(flow_data['dest_ips'])),
            'connection_duration': flow_data['duration'],
            
            # Protocol distribution
            'tcp_ratio': flow_data['tcp_count'] / flow_data['total_count'],
            'udp_ratio': flow_data['udp_count'] / flow_data['total_count'],
            
            # AI-specific metrics
            'gpt_api_calls_per_min': flow_data['gpt_calls'],
            'agent_loop_activations': flow_data['loop_activations'],
            'external_service_calls': flow_data['external_calls'],
        }
        
        return pd.DataFrame([features])
    
    def establish_baseline(self, historical_data):
        """
        Establish normal behavior baseline from historical data
        """
        features_list = []
        for flow in historical_data:
            features = self.extract_features(flow)
            features_list.append(features)
        
        X = pd.concat(features_list, ignore_index=True)
        X_scaled = self.scaler.fit_transform(X)
        
        self.isolation_forest.fit(X_scaled)
        self.baseline_established = True
        
        return self
    
    def detect_anomalies(self, current_flows):
        """
        Detect anomalous network behavior
        """
        if not self.baseline_established:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        anomalies = []
        
        for flow in current_flows:
            features = self.extract_features(flow)
            X_scaled = self.scaler.transform(features)
            
            # Isolation Forest prediction (-1 = anomaly, 1 = normal)
            prediction = self.isolation_forest.predict(X_scaled)[0]
            anomaly_score = self.isolation_forest.score_samples(X_scaled)[0]
            
            if prediction == -1:
                anomaly = {
                    'timestamp': flow['timestamp'],
                    'src_ip': flow['src_ip'],
                    'anomaly_score': anomaly_score,
                    'anomaly_type': self._classify_anomaly(flow, features),
                    'features': features.to_dict(),
                    'recommended_action': self._recommend_action(flow, anomaly_score)
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_anomaly(self, flow, features):
        """
        Classify the type of anomaly detected
        """
        # GPT API abuse detection
        if features['gpt_api_calls_per_min'].values[0] > 100:
            return "GPT_API_ABUSE"
        
        # Data exfiltration detection
        if features['bytes_per_second'].values[0] > 10000000:  # 10MB/s
            return "POTENTIAL_EXFILTRATION"
        
        # Lateral movement detection
        if features['unique_dest_ips'].values[0] > 50:
            return "LATERAL_MOVEMENT"
        
        # Agent loop anomaly
        if features['agent_loop_activations'].values[0] > 100:
            return "AGENT_LOOP_ANOMALY"
        
        # External service abuse
        if features['external_service_calls'].values[0] > 500:
            return "EXTERNAL_SERVICE_ABUSE"
        
        return "UNKNOWN_ANOMALY"
    
    def _recommend_action(self, flow, anomaly_score):
        """
        Recommend action based on anomaly severity
        """
        if anomaly_score < -0.7:
            return {
                'action': 'BLOCK_AND_ALERT',
                'priority': 'CRITICAL',
                'auto_remediate': True
            }
        elif anomaly_score < -0.5:
            return {
                'action': 'RATE_LIMIT_AND_ALERT',
                'priority': 'HIGH',
                'auto_remediate': False
            }
        else:
            return {
                'action': 'LOG_AND_MONITOR',
                'priority': 'MEDIUM',
                'auto_remediate': False
            }

# ============================================
# REAL-TIME MONITORING CONFIGURATION
# ============================================

monitoring_config = {
    "collection_interval": 60,  # seconds
    "analysis_window": 300,     # 5 minutes
    "alert_thresholds": {
        "gpt_api_rate": 100,    # calls per minute
        "data_transfer": 1000000000,  # 1GB
        "failed_loops": 5,      # concurrent failures
        "new_connections": 1000  # per minute
    },
    "response_actions": {
        "auto_block": True,
        "auto_rate_limit": True,
        "notify_security_team": True,
        "create_incident": True
    }
}
```

### 8.3 Network Flow Visualization

```yaml
# Grafana Dashboard Configuration
# Network Traffic Visualization

apiVersion: 1

providers:
  - name: 'AI Agent Network Dashboards'
    orgId: 1
    folder: 'Network Security'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards

dashboards:
  - title: "AI Agent Network Security Overview"
    uid: ai-agent-network
    
    panels:
      # Traffic Volume by Zone
      - title: "Traffic Volume by Security Zone"
        type: graph
        targets:
          - expr: |
              sum(rate(netflow_bytes_total[5m])) by (src_zone, dest_zone)
            legendFormat: "{{src_zone}} -> {{dest_zone}}"
        yaxes:
          - label: "Bytes/sec"
            format: bytes
      
      # GPT API Usage
      - title: "GPT-5.2 API Usage"
        type: stat
        targets:
          - expr: |
              sum(increase(gpt_api_requests_total[1h]))
            legendFormat: "Requests/Hour"
        thresholds:
          - color: green
            value: 0
          - color: yellow
            value: 500
          - color: red
            value: 1000
      
      # Agent Loop Health
      - title: "Agent Loop Status"
        type: table
        targets:
          - expr: |
              agent_loop_status{status="running"}
            format: table
            instant: true
        columns:
          - text: Loop ID
            value: loop_id
          - text: Status
            value: status
          - text: Last Heartbeat
            value: last_heartbeat
          - text: Tasks Completed
            value: tasks_completed
      
      # Security Alerts Timeline
      - title: "Security Alerts"
        type: graph
        targets:
          - expr: |
              sum(rate(security_alerts_total[5m])) by (severity, signature)
            legendFormat: "{{severity}} - {{signature}}"
        alert:
          name: "High Security Alert Rate"
          condition: "A"
          evaluator:
            type: gt
            params: [10]
          reducer:
            type: avg
            params: []
          query:
            params: [A, 5m, now]
      
      # Network Topology Map
      - title: "Network Traffic Flow"
        type: nodeGraph
        targets:
          - expr: |
              netflow_connections_total
        nodes:
          - id: "src_ip"
            title: "Source"
          - id: "dest_ip"
            title: "Destination"
        edges:
          - source: "src_ip"
            target: "dest_ip"
            value: "bytes"
      
      # Geographic Traffic Distribution
      - title: "Traffic by Geography"
        type: geomap
        targets:
          - expr: |
              sum(rate(netflow_bytes_total[5m])) by (geo_country)
        layer:
          type: heatmap
          location: geo_country
          weight: bytes

    annotations:
      - name: "Security Incidents"
        datasource: "Elasticsearch"
        expr: 'index:security AND severity:critical'
        tags: ["security", "critical"]
      
      - name: "Deployment Events"
        datasource: "Prometheus"
        expr: 'deployment_events'
        tags: ["deployment"]
```

---

## 9. Implementation Checklist

### Phase 1: Network Foundation
- [ ] Deploy core switches with VLAN configuration
- [ ] Configure inter-VLAN routing
- [ ] Implement DHCP snooping and ARP inspection
- [ ] Deploy perimeter firewall with base rules
- [ ] Configure management VLAN access

### Phase 2: Security Zones
- [ ] Segment network into 5 security zones
- [ ] Deploy zone-based firewalls
- [ ] Configure east-west traffic policies
- [ ] Implement micro-segmentation
- [ ] Deploy IDS/IPS sensors

### Phase 3: Service Integration
- [ ] Configure GPT-5.2 API access rules
- [ ] Deploy Gmail service isolation
- [ ] Implement Twilio voice security
- [ ] Configure TTS/STT cloud access
- [ ] Deploy browser control sandboxing

### Phase 4: Monitoring & Analytics
- [ ] Deploy SIEM collectors
- [ ] Configure network flow monitoring
- [ ] Implement behavioral analytics
- [ ] Deploy threat intelligence feeds
- [ ] Create security dashboards

### Phase 5: Hardening & Optimization
- [ ] Conduct penetration testing
- [ ] Optimize firewall rules
- [ ] Tune anomaly detection thresholds
- [ ] Document incident response procedures
- [ ] Train security operations team

---

## 10. Compliance Mapping

| Control | NIST 800-53 | ISO 27001 | SOC 2 | Implementation |
|---------|-------------|-----------|-------|----------------|
| Network Segmentation | SC-7, SC-32 | A.13.1.1 | CC6.1 | VLANs, Security Zones |
| Traffic Filtering | SC-7(4) | A.13.1.1 | CC6.6 | Firewall Rules, ACLs |
| Micro-segmentation | SC-7(21) | A.13.1.3 | CC6.1 | Container Policies |
| Monitoring | AU-6, AU-12 | A.12.4.1 | CC7.2 | SIEM, NBA |
| Encryption | SC-8, SC-13 | A.10.1.1 | CC6.7 | TLS 1.3, mTLS |

---

## Appendix A: IP Address Allocation

| Network | Purpose | VLAN | Gateway |
|---------|---------|------|---------|
| 10.0.0.0/24 | Management | 10 | 10.0.0.1 |
| 10.0.1.0/24 | Monitoring | 20 | 10.0.1.1 |
| 10.1.0.0/24 | DMZ | 100-101 | 10.1.0.1 |
| 10.2.0.0/23 | Application | 200-201 | 10.2.0.1 |
| 10.3.0.0/22 | Services | 300-305 | 10.3.0.1 |
| 10.4.0.0/23 | Database | 400-404 | 10.4.0.1 |

---

## Appendix B: Port Reference

| Service | Port | Protocol | Zone |
|---------|------|----------|------|
| GPT-5.2 API | 443 | TCP | Application |
| Gmail IMAP | 993 | TCP | Services |
| Gmail SMTP | 587 | TCP | Services |
| Twilio API | 443 | TCP | Services |
| Twilio SIP | 5060 | UDP | Services |
| Twilio RTP | 10000-20000 | UDP | Services |
| PostgreSQL | 5432 | TCP | Database |
| Redis | 6379 | TCP | Database |
| Qdrant | 6333 | TCP | Database |
| Agent Loops | 9000-9015 | TCP | Application |

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Classification:** Internal Use Only  
**Author:** Network Security Architecture Team
