# PERMISSION SYSTEM AND ACCESS CONTROL ARCHITECTURE
## Windows 10 OpenClaw-Inspired AI Agent System
### Technical Specification v1.0

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Security Architecture Overview](#2-security-architecture-overview)
3. [Role-Based Access Control (RBAC)](#3-role-based-access-control-rbac)
4. [Attribute-Based Access Control (ABAC)](#4-attribute-based-access-control-abac)
5. [Permission Granularity Framework](#5-permission-granularity-framework)
6. [Skill-Level Permissions](#6-skill-level-permissions)
7. [Dynamic Permission Elevation](#7-dynamic-permission-elevation)
8. [Permission Audit Logging](#8-permission-audit-logging)
9. [User Consent Flows](#9-user-consent-flows)
10. [Permission Revocation](#10-permission-revocation)
11. [Implementation Reference](#11-implementation-reference)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Purpose
This specification defines a comprehensive permission system and access control architecture for a Windows 10-based AI agent system inspired by OpenClaw. The architecture addresses critical security vulnerabilities identified in existing AI agent frameworks while enabling the required 24/7 autonomous operation with Gmail, browser control, TTS, STT, Twilio, and full system access capabilities.

### 1.2 Security Principles
| Principle | Implementation |
|-----------|----------------|
| **Least Privilege** | Agents start with minimal permissions; access granted only when explicitly needed |
| **Defense in Depth** | Multiple security layers: RBAC + ABAC + skill-level + runtime guards |
| **Explicit Consent** | User approval required for sensitive operations |
| **Complete Auditability** | All permission checks and actions logged immutably |
| **Dynamic Adaptation** | Permissions adjust based on context, time, and risk assessment |
| **Zero Trust** | Continuous verification of agent identity and authorization |

### 1.3 Threat Model
Based on OpenClaw security research, the following threats are addressed:
- **Prompt Injection Attacks**: Malicious instructions embedded in content
- **Skill Poisoning**: Malicious skills inheriting system-wide permissions
- **Privilege Escalation**: Unauthorized permission elevation attempts
- **Credential Theft**: API keys and tokens stored insecurely
- **Unauthorized Data Access**: Agents accessing beyond their scope
- **Supply Chain Attacks**: Compromised third-party integrations

---

## 2. SECURITY ARCHITECTURE OVERVIEW

### 2.1 System Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                           USER INTERFACE LAYER                               |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
|  |   Web UI    |  |  Voice UI   |  |  System UI  |  |  Consent Dashboard  | |
|  +------+------+  +------+------+  +------+------+  +----------+----------+ |
+--------+----------------+----------------+--------------------+----------------
          |                |                |                    |
          +----------------+----------------+--------------------+
                                     |
+------------------------------------|----------------------------------------+
|                         ACCESS CONTROL LAYER                                 |
|  +---------------------------------|------------------------------------+   |
|  |         POLICY DECISION POINT (PDP)                                  |   |
|  |  +-------------+  +-------------+  +-------------+  +-------------+  |   |
|  |  |   RBAC      |  |   ABAC      |  |  Skill      |  |  Dynamic    |  |   |
|  |  |   Engine    |  |   Engine    |  |  Validator  |  |  Elevator   |  |   |
|  |  +-------------+  +-------------+  +-------------+  +-------------+  |   |
|  +-----------------------------------------------------------------------+   |
|                                                                              |
|  +-----------------------------------------------------------------------+   |
|  |         POLICY ENFORCEMENT POINTS (PEPs)                             |   |
|  |  +---------+ +---------+ +---------+ +---------+ +---------+        |   |
|  |  |  Input  | | Retrieve| |  Tool   | | Output  | | Network |        |   |
|  |  |  Guard  | | Filter  | |  Gate   | |  Guard  | |  Gate   |        |   |
|  |  +---------+ +---------+ +---------+ +---------+ +---------+        |   |
|  +-----------------------------------------------------------------------+   |
+------------------------------------------------------------------------------+
                                     |
+------------------------------------|----------------------------------------+
|                         AGENT EXECUTION LAYER                                |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
|  |  GPT-5.2    |  |  Agentic    |  |   Skill     |  |   Agent Identity    | |
|  |  Engine     |  |  Loops (15) |  |  Registry   |  |     Manager         | |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
|                                                                              |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
|  |    Soul     |  |  Heartbeat  |  |   Cron      |  |   Identity System   | |
|  |   Engine    |  |   Monitor   |  |  Scheduler  |  |                     | |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
+------------------------------------------------------------------------------+
                                     |
+------------------------------------|----------------------------------------+
|                      RESOURCE ACCESS LAYER                                   |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
|  |   Gmail     |  |   Browser   |  |  TTS/STT    |  |   Twilio    |         |
|  |   Client    |  |   Control   |  |   Engine    |  |   Service   |         |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
|                                                                              |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
|  | File System |  |   Network   |  |   System    |  |   Windows   |         |
|  |   Access    |  |   Access    |  |   Shell     |  |     API     |         |
|  +-------------+  +-------------+  +-------------+  +-------------+         |
+------------------------------------------------------------------------------+
```

### 2.2 Core Components

| Component | Responsibility | Security Function |
|-----------|---------------|-------------------|
| **Policy Decision Point (PDP)** | Evaluates access requests against policies | Central authorization engine |
| **Policy Enforcement Points (PEPs)** | Intercepts and guards all resource access | Distributed enforcement |
| **Agent Identity Manager** | Unique identity per agent instance | Authentication & tracking |
| **Skill Validator** | Validates skill permissions before execution | Prevents skill poisoning |
| **Consent Manager** | Handles user approval workflows | Human oversight |
| **Audit Logger** | Records all permission decisions | Compliance & forensics |

---

## 3. ROLE-BASED ACCESS CONTROL (RBAC)

### 3.1 Role Hierarchy

```
+------------------------------------------------------------------+
|                    ROLE HIERARCHY                                |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |              SYSTEM_ADMINISTRATOR                          |   |
|  |  - Full system access with override capability             |   |
|  |  - Can modify permission policies                          |   |
|  |  - Emergency break-glass access                            |   |
|  +---------------------------+-------------------------------+   |
|                             |                                    |
|         +-------------------+-------------------+                |
|         |                   |                   |                |
|  +------+------+    +------+------+    +------+------+          |
|  |   AGENT     |    |   USER      |    |   SERVICE   |          |
|  |   OPERATOR  |    |   ADMIN     |    |   MANAGER   |          |
|  +------+------+    +------+------+    +-------------+          |
|         |                   |                                   |
|    +----+----+         +----+----+                            |
|    |         |         |         |                            |
|  +--+--+   +--+--+  +--+--+   +--+--+                         |
|  |CORE |   |LOOP |  |OWNER|   |GUEST|                         |
|  |AGENT|   |AGENT|  |     |   |     |                         |
|  +-----+   +-----+  +-----+   +-----+                         |
|                                                               |
+------------------------------------------------------------------+
```

### 3.2 Role Definitions

#### 3.2.1 System Administrator
```yaml
role_id: "system_administrator"
name: "System Administrator"
level: 100
description: "Full system control with emergency override"
permissions:
  - permission: "policy:manage"
    scope: "*"
  - permission: "agent:override"
    scope: "*"
  - permission: "audit:read"
    scope: "*"
  - permission: "consent:bypass"
    scope: "emergency_only"
constraints:
  - requires_mfa: true
  - session_timeout: 3600
  - audit_all_actions: true
```

#### 3.2.2 Agent Operator (Core Agent)
```yaml
role_id: "agent_operator_core"
name: "Core Agent Operator"
level: 50
description: "Primary AI agent with elevated permissions"
permissions:
  - permission: "file:read"
    scope: "user_directories"
  - permission: "file:write"
    scope: "agent_workspace"
  - permission: "network:connect"
    scope: "approved_endpoints"
  - permission: "system:execute"
    scope: "sandboxed_commands"
  - permission: "gmail:read"
    scope: "own_account"
  - permission: "gmail:send"
    scope: "approved_recipients"
  - permission: "browser:control"
    scope: "user_approved_sites"
  - permission: "tts:speak"
    scope: "local_output"
  - permission: "stt:listen"
    scope: "user_activated"
  - permission: "twilio:call"
    scope: "approved_numbers"
  - permission: "twilio:sms"
    scope: "approved_numbers"
constraints:
  - max_daily_emails: 100
  - max_hourly_calls: 10
  - requires_consent_for: ["file:delete", "system:execute", "twilio:call"]
```

#### 3.2.3 Agent Operator (Loop Agent)
```yaml
role_id: "agent_operator_loop"
name: "Loop Agent Operator"
level: 40
description: "Specialized agentic loop with task-specific permissions"
permissions:
  - permission: "file:read"
    scope: "loop_workspace"
  - permission: "file:write"
    scope: "loop_workspace"
  - permission: "network:connect"
    scope: "loop_specific_endpoints"
  - permission: "skill:execute"
    scope: "assigned_skills"
constraints:
  - inherits_from: "agent_operator_core"
  - skill_whitelist_only: true
  - cannot_spawn_subagents: true
```

#### 3.2.4 User Owner
```yaml
role_id: "user_owner"
name: "User Owner"
level: 30
description: "Human user with agent oversight"
permissions:
  - permission: "agent:configure"
    scope: "own_agents"
  - permission: "consent:grant"
    scope: "own_agents"
  - permission: "audit:view"
    scope: "own_activity"
  - permission: "permission:modify"
    scope: "own_agents"
constraints:
  - cannot_modify_system_policies: true
  - requires_authentication: true
```

#### 3.2.5 User Guest
```yaml
role_id: "user_guest"
name: "User Guest"
level: 10
description: "Limited access guest user"
permissions:
  - permission: "agent:interact"
    scope: "read_only"
  - permission: "conversation:view"
    scope: "own_conversations"
constraints:
  - no_system_access: true
  - no_file_access: true
  - session_limited: true
```

### 3.3 Permission Matrix

| Permission | System Admin | Core Agent | Loop Agent | User Owner | Guest |
|------------|:------------:|:----------:|:----------:|:----------:|:-----:|
| **File System** |
| file:read:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| file:read:user_dirs | ✅ | ✅ | ❌ | ✅ | ❌ |
| file:read:workspace | ✅ | ✅ | ✅ | ✅ | ❌ |
| file:write:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| file:write:workspace | ✅ | ✅ | ✅ | ✅ | ❌ |
| file:delete | ✅ | ⚠️* | ❌ | ⚠️* | ❌ |
| **Network** |
| network:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| network:approved | ✅ | ✅ | ✅ | ✅ | ❌ |
| network:loop_specific | ✅ | ✅ | ✅ | ❌ | ❌ |
| **System** |
| system:execute:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| system:execute:sandboxed | ✅ | ✅ | ❌ | ❌ | ❌ |
| system:service:manage | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Gmail** |
| gmail:read:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| gmail:read:own | ✅ | ✅ | ✅ | ✅ | ❌ |
| gmail:send:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| gmail:send:approved | ✅ | ✅ | ⚠️* | ⚠️* | ❌ |
| **Browser** |
| browser:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| browser:approved | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Twilio** |
| twilio:call:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| twilio:call:approved | ✅ | ✅ | ⚠️* | ⚠️* | ❌ |
| twilio:sms:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| twilio:sms:approved | ✅ | ✅ | ⚠️* | ⚠️* | ❌ |
| **Agent Management** |
| agent:create | ✅ | ❌ | ❌ | ✅ | ❌ |
| agent:modify:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| agent:modify:own | ✅ | ❌ | ❌ | ✅ | ❌ |
| agent:delete:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| agent:delete:own | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Policy** |
| policy:read | ✅ | ✅ | ✅ | ✅ | ❌ |
| policy:modify | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Audit** |
| audit:read:any | ✅ | ❌ | ❌ | ❌ | ❌ |
| audit:read:own | ✅ | ✅ | ✅ | ✅ | ❌ |

*⚠️ = Requires user consent

---

## 4. ATTRIBUTE-BASED ACCESS CONTROL (ABAC)

### 4.1 Attribute Categories

```
+------------------------------------------------------------------+
|                  ABAC ATTRIBUTE CATEGORIES                       |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------+  +-----------------+  +-----------------+ |
|  | SUBJECT ATTRS   |  | RESOURCE ATTRS  |  |  ACTION ATTRS   | |
|  |                 |  |                 |  |                 | |
|  | • agent_id      |  | • resource_type |  | • action_type   | |
|  | • user_id       |  | • sensitivity   |  | • risk_level    | |
|  | • role          |  | • owner         |  | • reversibility | |
|  | • clearance     |  | • classification|  | • data_impact   | |
|  | • department    |  | • location      |  | • scope         | |
|  | • session_age   |  | • created_at    |  | • frequency     | |
|  | • auth_method   |  | • tags[]        |  | • requires_2fa  | |
|  +-----------------+  +-----------------+  +-----------------+ |
|                                                                  |
|  +-----------------+  +-----------------+                       |
|  |  ENVIRONMENT    |  |   CONTEXT ATTRS |                       |
|  |                 |  |                 |                       |
|  | • time_of_day   |  | • task_context  |                       |
|  | • day_of_week   |  | • conversation  |                       |
|  | • network_type  |  | • user_intent   |                       |
|  | • location      |  | • skill_chain   |                       |
|  | • device_trust  |  | • loop_context  |                       |
|  | • threat_level  |  | • history[]     |                       |
|  | • geofence      |  | • confidence    |                       |
|  +-----------------+  +-----------------+                       |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 Subject Attributes Schema

```json
{
  "subject_attributes": {
    "agent": {
      "agent_id": "uuid",
      "agent_type": "enum[core, loop, scheduled, triggered]",
      "loop_id": "string|null",
      "skill_set": ["string"],
      "trust_score": "float[0-1]",
      "session_start": "datetime",
      "authentication": {
        "method": "enum[password, mfa, certificate, token]",
        "strength": "enum[low, medium, high]",
        "time_since_auth": "seconds"
      }
    },
    "user": {
      "user_id": "uuid",
      "roles": ["string"],
      "clearance_level": "int[1-5]",
      "department": "string",
      "manager": "uuid",
      "risk_profile": "enum[low, medium, high]",
      "consent_preferences": {
        "auto_approve_low_risk": "bool",
        "require_notification": ["string"]
      }
    }
  }
}
```

### 4.3 Resource Attributes Schema

```json
{
  "resource_attributes": {
    "file": {
      "path": "string",
      "owner": "uuid",
      "sensitivity": "enum[public, internal, confidential, restricted]",
      "classification": "enum[pii, phi, pci, none]",
      "tags": ["string"],
      "size_bytes": "int",
      "created_at": "datetime",
      "modified_at": "datetime",
      "encrypted": "bool",
      "backup_required": "bool"
    },
    "network_endpoint": {
      "url": "string",
      "domain": "string",
      "ip_range": "string",
      "category": "enum[approved, restricted, blocked, unknown]",
      "tls_required": "bool",
      "data_residency": "string",
      "risk_score": "float[0-1]"
    },
    "email": {
      "message_id": "string",
      "sender": "string",
      "recipients": ["string"],
      "sensitivity": "enum[low, medium, high]",
      "contains_pii": "bool",
      "external_domain": "bool"
    },
    "skill": {
      "skill_id": "string",
      "name": "string",
      "version": "string",
      "publisher": "string",
      "verified": "bool",
      "risk_level": "enum[low, medium, high, critical]",
      "permissions_required": ["string"],
      "sandbox_required": "bool"
    }
  }
}
```

### 4.4 Environment Attributes Schema

```json
{
  "environment_attributes": {
    "time": {
      "timestamp": "datetime",
      "hour": "int[0-23]",
      "day_of_week": "enum[mon, tue, wed, thu, fri, sat, sun]",
      "is_business_hours": "bool",
      "is_holiday": "bool"
    },
    "network": {
      "connection_type": "enum[corporate, home, public, vpn]",
      "trusted_network": "bool",
      "geolocation": {
        "country": "string",
        "region": "string",
        "compliance_zone": "string"
      },
      "threat_level": "enum[low, medium, high, critical]"
    },
    "device": {
      "device_id": "uuid",
      "device_type": "enum[desktop, laptop, server]",
      "trust_level": "enum[untrusted, basic, trusted, high_trust]",
      "compliance_status": "enum[compliant, warning, non_compliant]",
      "security_software": ["string"]
    },
    "session": {
      "session_id": "uuid",
      "duration_seconds": "int",
      "idle_time_seconds": "int",
      "concurrent_agents": "int",
      "anomaly_score": "float[0-1]"
    }
  }
}
```

### 4.5 ABAC Policy Examples

#### 4.5.1 Time-Based Access Policy
```yaml
policy_id: "time_based_file_access"
name: "Time-Based File Access Restriction"
description: "Restrict sensitive file access outside business hours"
version: "1.0"
priority: 100

target:
  resource_type: "file"
  resource_attributes:
    sensitivity: ["confidential", "restricted"]

conditions:
  all:
    - condition:
        attribute: "environment.time.is_business_hours"
        operator: "equals"
        value: false
    - condition:
        attribute: "subject.user.clearance_level"
        operator: "less_than"
        value: 4

decision: "deny"
deny_reason: "Sensitive file access restricted outside business hours"

override:
  allowed: true
  requires: ["manager_approval", "security_notification"]
```

#### 4.5.2 Network-Based Email Policy
```yaml
policy_id: "network_email_restriction"
name: "Network-Based Email Sending Restriction"
description: "Require additional verification for email sending from untrusted networks"
version: "1.0"
priority: 150

target:
  action: "gmail:send"

conditions:
  all:
    - condition:
        attribute: "environment.network.trusted_network"
        operator: "equals"
        value: false
    - condition:
        attribute: "resource.email.external_domain"
        operator: "equals"
        value: true
    - condition:
        attribute: "resource.email.contains_pii"
        operator: "equals"
        value: true

decision: "require_consent"
consent_type: "explicit_approval"
consent_timeout: 300
```

#### 4.5.3 Skill Risk-Based Policy
```yaml
policy_id: "skill_risk_sandbox"
name: "High-Risk Skill Sandboxing"
description: "Execute high-risk skills in isolated sandbox"
version: "1.0"
priority: 200

target:
  action: "skill:execute"
  resource_attributes:
    risk_level: ["high", "critical"]

conditions:
  any:
    - condition:
        attribute: "resource.skill.verified"
        operator: "equals"
        value: false
    - condition:
        attribute: "resource.skill.publisher"
        operator: "not_in"
        value: ["trusted_publishers_list"]

decision: "require_sandbox"
sandbox_config:
  network: "isolated"
  filesystem: "read_only"
  resources:
    cpu_limit: "50%"
    memory_limit: "512MB"
    timeout: 300
```

#### 4.5.4 Context-Aware Permission Elevation
```yaml
policy_id: "context_elevation"
name: "Context-Aware Permission Elevation"
description: "Allow temporary elevation based on task context and user history"
version: "1.0"
priority: 175

target:
  action_type: "elevation_request"

conditions:
  all:
    - condition:
        attribute: "context.user_intent.confidence"
        operator: "greater_than"
        value: 0.85
    - condition:
        attribute: "context.conversation.topic"
        operator: "in"
        value: ["approved_workflows"]
    - condition:
        attribute: "subject.user.risk_profile"
        operator: "equals"
        value: "low"
    - condition:
        attribute: "environment.session.anomaly_score"
        operator: "less_than"
        value: 0.3

decision: "allow_conditional"
conditions_for_allow:
  - time_limit: 3600
  - scope_limit: "requested_resource_only"
  - audit_level: "detailed"
  - notification: "user"
```


---

## 5. PERMISSION GRANULARITY FRAMEWORK

### 5.1 File System Permission Granularity

```
+------------------------------------------------------------------+
|              FILE SYSTEM PERMISSION LEVELS                       |
+------------------------------------------------------------------+
|                                                                  |
|  Level 5: FULL ACCESS                                            |
|  +-- Read any file on system                                     |
|  +-- Write any file on system                                    |
|  +-- Delete any file                                             |
|  +-- Modify permissions                                          |
|  +-- Execute any file                                            |
|  [Requires: System Admin + Explicit Consent + Audit]            |
|                                                                  |
|  Level 4: USER SCOPE                                             |
|  +-- Read: User home directory                                   |
|  +-- Write: User documents, downloads, desktop                   |
|  +-- Delete: Own files only                                      |
|  +-- Execute: Approved scripts only                              |
|  [Requires: User Owner Role + Consent for destructive ops]      |
|                                                                  |
|  Level 3: WORKSPACE SCOPE                                        |
|  +-- Read: Agent workspace directory                             |
|  +-- Write: Agent workspace directory                            |
|  +-- Delete: Workspace files only                                |
|  +-- Execute: Sandboxed only                                     |
|  [Requires: Agent Role + Sandbox]                               |
|                                                                  |
|  Level 2: READ-ONLY                                              |
|  +-- Read: Specific approved directories                         |
|  +-- No write access                                             |
|  +-- No delete access                                            |
|  +-- No execute access                                           |
|  [Requires: Any authenticated agent]                            |
|                                                                  |
|  Level 1: NO ACCESS                                              |
|  +-- No file system operations permitted                         |
|  [Default for untrusted agents]                                 |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 File Permission Configuration

```yaml
# File System Permission Policy
file_permissions:
  
  # Level 5 - Full Access (System Admin Only)
  full_access:
    roles: ["system_administrator"]
    paths: ["*"]
    operations: ["read", "write", "delete", "execute", "chmod"]
    constraints:
      - requires_mfa: true
      - requires_approval: true
      - audit_all: true
      - session_timeout: 3600
  
  # Level 4 - User Scope
  user_scope:
    roles: ["user_owner", "agent_operator_core"]
    paths:
      - "${USER_HOME}/Documents"
      - "${USER_HOME}/Downloads"
      - "${USER_HOME}/Desktop"
      - "${USER_HOME}/Pictures"
    operations: ["read", "write"]
    constraints:
      - max_file_size: "100MB"
      - blocked_extensions: [".exe", "dll", ".bat", ".ps1"]
      - delete_requires_consent: true
  
  # Level 3 - Workspace Scope
  workspace_scope:
    roles: ["agent_operator_core", "agent_operator_loop"]
    paths:
      - "${AGENT_WORKSPACE}/data"
      - "${AGENT_WORKSPACE}/temp"
      - "${AGENT_WORKSPACE}/output"
    operations: ["read", "write", "delete"]
    constraints:
      - sandbox_required: true
      - network_isolated: true
      - quota_limit: "1GB"
  
  # Level 2 - Read Only
  read_only:
    roles: ["user_guest"]
    paths:
      - "${AGENT_WORKSPACE}/public"
    operations: ["read"]
    constraints:
      - no_download: true
      - view_only: true

# File Classification Rules
file_classification:
  pii_patterns:
    - pattern: "\\b\\d{3}-\\d{2}-\\d{4}\\b"
      type: "ssn"
      sensitivity: "restricted"
    - pattern: "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
      type: "credit_card"
      sensitivity: "restricted"
    - pattern: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      type: "email"
      sensitivity: "confidential"
  
  restricted_paths:
    - pattern: "*/Windows/System32/*"
      sensitivity: "restricted"
    - pattern: "*/Program Files/*"
      sensitivity: "confidential"
    - pattern: "*/.ssh/*"
      sensitivity: "restricted"
    - pattern: "*/.env"
      sensitivity: "restricted"
```

### 5.3 Network Permission Granularity

```
+------------------------------------------------------------------+
|              NETWORK PERMISSION LEVELS                           |
+------------------------------------------------------------------+
|                                                                  |
|  Level 5: UNRESTRICTED                                           |
|  +-- Connect to any IP/Port                                      |
|  +-- Listen on any port                                          |
|  +-- Raw socket access                                           |
|  +-- Modify network configuration                                |
|  [Requires: System Admin + Emergency Override]                  |
|                                                                  |
|  Level 4: APPROVED ENDPOINTS                                     |
|  +-- Connect to pre-approved domains/IPs                         |
|  +-- HTTPS only (TLS 1.3+)                                       |
|  +-- Certificate pinning                                         |
|  +-- Rate limited                                                |
|  [Requires: Agent Role + Valid Certificate]                     |
|                                                                  |
|  Level 3: SERVICE-SPECIFIC                                       |
|  +-- Gmail API only                                              |
|  +-- Twilio API only                                             |
|  +-- Specific browser endpoints                                  |
|  +-- OAuth-scoped access                                         |
|  [Requires: Valid OAuth Token + Scope Validation]               |
|                                                                  |
|  Level 2: OUTBOUND ONLY                                          |
|  +-- Outbound connections only                                   |
|  +-- No listening ports                                          |
|  +-- DNS queries only                                            |
|  +-- HTTP/HTTPS only                                             |
|  [Requires: Basic Agent Identity]                               |
|                                                                  |
|  Level 1: NO NETWORK                                             |
|  +-- All network access blocked                                  |
|  [Default for untrusted/skills]                                 |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.4 Network Permission Configuration

```yaml
# Network Permission Policy
network_permissions:
  
  # Approved Endpoints Registry
  endpoint_registry:
    gmail:
      domains: ["gmail.com", "googleapis.com"]
      ips: ["142.250.0.0/16", "172.217.0.0/16"]
      ports: [443]
      protocols: ["https"]
      tls_version: "1.3"
      oauth_scopes: ["https://www.googleapis.com/auth/gmail.readonly", 
                     "https://www.googleapis.com/auth/gmail.send"]
    
    twilio:
      domains: ["twilio.com", "twiliocdn.com"]
      ips: ["54.172.0.0/16"]
      ports: [443]
      protocols: ["https"]
      rate_limits:
        calls_per_minute: 10
        sms_per_minute: 30
    
    browser:
      domains: ["*"]  # User-approved via consent
      blocked_categories: ["malware", "phishing", "adult"]
      content_filtering: true
    
    tts_stt:
      domains: ["api.elevenlabs.io", "speech.googleapis.com"]
      ips: []
      ports: [443]
      data_residency: "user_region"
  
  # Network Security Rules
  security_rules:
    default_action: "deny"
    
    firewall_rules:
      - name: "allow_approved_endpoints"
        action: "allow"
        destinations: "${ENDPOINT_REGISTRY}"
        
      - name: "block_private_ips"
        action: "deny"
        destinations: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
        
      - name: "block_suspicious_ports"
        action: "deny"
        ports: [22, 23, 25, 135, 445, 3389]
    
    rate_limiting:
      global:
        requests_per_second: 100
        burst_size: 150
      per_endpoint:
        requests_per_minute: 60
    
    tls_requirements:
      minimum_version: "1.2"
      preferred_version: "1.3"
      certificate_validation: "strict"
      pinned_certificates: ["${CERT_PIN_STORE}"]
```

### 5.5 System Permission Granularity

```
+------------------------------------------------------------------+
|              SYSTEM PERMISSION LEVELS                            |
+------------------------------------------------------------------+
|                                                                  |
|  Level 5: SYSTEM ADMIN                                           |
|  +-- Install/uninstall software                                  |
|  +-- Modify system settings                                      |
|  +-- Manage services                                             |
|  +-- Access all user accounts                                    |
|  +-- Modify security policies                                    |
|  [Requires: System Admin + MFA + Audit]                         |
|                                                                  |
|  Level 4: POWER USER                                             |
|  +-- Execute approved system commands                            |
|  +-- Modify user environment variables                           |
|  +-- Install user-level software                                 |
|  +-- Schedule tasks (user scope)                                 |
|  [Requires: User Owner + Command Allowlist]                     |
|                                                                  |
|  Level 3: STANDARD USER                                          |
|  +-- Execute sandboxed commands                                  |
|  +-- Read system information                                     |
|  +-- Modify own profile settings                                 |
|  +-- Run approved applications                                   |
|  [Requires: Agent Role + Sandbox]                               |
|                                                                  |
|  Level 2: RESTRICTED USER                                        |
|  +-- Read limited system info                                    |
|  +-- No command execution                                        |
|  +-- No system modifications                                     |
|  +-- No software installation                                    |
|  [Requires: Basic Authentication]                               |
|                                                                  |
|  Level 1: GUEST                                                  |
|  +-- No system access                                            |
|  [Default for new/untrusted agents]                             |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.6 System Permission Configuration

```yaml
# System Permission Policy
system_permissions:
  
  # Command Execution Allowlist
  command_allowlist:
    level_4:
      - command: "python"
        args: ["${APPROVED_SCRIPTS}"]
        sandbox: true
      - command: "node"
        args: ["${APPROVED_SCRIPTS}"]
        sandbox: true
      - command: "git"
        args: ["clone", "pull", "status", "log"]
        blocked_args: ["push", "force"]
      - command: "curl"
        args: ["${APPROVED_ENDPOINTS}"]
        flags: ["--max-time", "30", "--retry", "2"]
    
    level_3:
      - command: "dir"
        args: ["${WORKSPACE_PATH}"]
      - command: "type"
        args: ["${WORKSPACE_FILES}"]
      - command: "echo"
        args: ["*"]
    
    blocked_commands:
      - "format"
      - "diskpart"
      - "regedit"
      - "powershell -ExecutionPolicy Bypass"
      - "cmd /c"
      - "certutil"
      - "bitsadmin"
  
  # Service Management
  service_permissions:
    can_start_services: false
    can_stop_services: false
    can_restart_services: ["agent_service"]
    can_modify_services: false
  
  # Registry Access
  registry_permissions:
    read_paths: ["HKEY_CURRENT_USER\\Software\\OpenClaw"]
    write_paths: []
    blocked_paths:
      - "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services"
      - "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
  
  # Windows API Access
  api_permissions:
    allowed_apis:
      - "user32.GetForegroundWindow"
      - "user32.GetWindowText"
      - "kernel32.GetSystemInfo"
    blocked_apis:
      - "advapi32.OpenProcessToken"
      - "ntdll.NtCreateThreadEx"
      - "kernel32.WriteProcessMemory"
```

---

## 6. SKILL-LEVEL PERMISSIONS

### 6.1 Skill Permission Model

```
+------------------------------------------------------------------+
|              SKILL PERMISSION ARCHITECTURE                       |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |                    SKILL MANIFEST                          |   |
|  |                                                            |   |
|  |  Required Permissions:                                     |   |
|  |  +-- file:read:workspace                                   |   |
|  |  +-- file:write:workspace                                  |   |
|  |  +-- network:connect:approved                              |   |
|  |  +-- system:execute:sandboxed                              |   |
|  |                                                            |   |
|  |  Risk Assessment:                                          |   |
|  |  +-- data_access: high                                     |   |
|  |  +-- network_access: medium                                |   |
|  |  +-- system_impact: low                                    |   |
|  |  +-- overall: medium                                       |   |
|  |                                                            |   |
|  |  Sandbox Requirements:                                     |   |
|  |  +-- network_isolated: false                               |   |
|  |  +-- filesystem_readonly: false                            |   |
|  |  +-- resource_limits: default                              |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              SKILL VALIDATION ENGINE                       |   |
|  |                                                            |   |
|  |  1. Parse manifest and extract permissions                 |   |
|  |  2. Verify publisher signature                             |   |
|  |  3. Check against global permission policy                 |   |
|  |  4. Calculate risk score                                   |   |
|  |  5. Determine sandbox requirements                         |   |
|  |  6. Generate execution profile                             |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              SKILL EXECUTION RUNTIME                       |   |
|  |                                                            |   |
|  |  + Enforce manifest permissions                            |   |
|  |  + Apply sandbox constraints                               |   |
|  |  + Monitor resource usage                                  |   |
|  |  + Log all operations                                      |   |
|  |  + Terminate on violation                                  |   |
|  |                                                            |   |
|  +-----------------------------------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

### 6.2 Skill Manifest Schema

```json
{
  "$schema": "https://openclaw.io/schemas/skill-manifest/v1",
  "skill": {
    "id": "email-processor",
    "name": "Email Processor",
    "version": "1.2.0",
    "publisher": {
      "name": "OpenClaw Team",
      "id": "publisher:openclaw:official",
      "verified": true,
      "signature": "-----BEGIN SIGNATURE-----\n...\n-----END SIGNATURE-----"
    },
    "description": "Process and analyze emails from Gmail",
    "category": "productivity",
    
    "permissions": {
      "required": [
        {
          "permission": "gmail:read",
          "scope": "own_account",
          "justification": "Required to read emails for processing"
        },
        {
          "permission": "file:write",
          "scope": "workspace",
          "justification": "Required to save email attachments"
        },
        {
          "permission": "network:connect",
          "scope": "googleapis.com",
          "justification": "Required to connect to Gmail API"
        }
      ],
      "optional": [
        {
          "permission": "gmail:send",
          "scope": "own_account",
          "justification": "Required to send automated responses"
        }
      ]
    },
    
    "risk_assessment": {
      "data_access_level": "high",
      "network_access_level": "medium",
      "system_impact_level": "low",
      "overall_risk": "medium",
      "factors": [
        "Accesses sensitive email content",
        "Connects to external API",
        "No direct system access"
      ]
    },
    
    "sandbox": {
      "required": true,
      "network_access": "restricted",
      "allowed_endpoints": ["gmail.googleapis.com", "www.googleapis.com"],
      "filesystem_access": "workspace_only",
      "resource_limits": {
        "cpu_percent": 25,
        "memory_mb": 256,
        "timeout_seconds": 300
      }
    },
    
    "runtime": {
      "entry_point": "main.py",
      "language": "python",
      "dependencies": ["google-api-python-client", "google-auth"],
      "environment_variables": [],
      "secrets": ["GMAIL_API_KEY"]
    },
    
    "consent": {
      "requires_approval": true,
      "approval_prompt": "This skill will access your Gmail account to read and process emails.",
      "data_handling": "Email content is processed locally and not stored permanently."
    }
  }
}
```

### 6.3 Skill Permission Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Data Access** | What data the skill can read/write | `file:read`, `gmail:read`, `registry:read` |
| **Network Access** | Network connections the skill can make | `network:connect`, `api:call` |
| **System Access** | System-level operations | `system:execute`, `service:manage` |
| **UI Access** | User interface interactions | `ui:notification`, `ui:dialog` |
| **Hardware Access** | Hardware device access | `microphone:access`, `camera:access` |

### 6.4 Skill Risk Classification

```yaml
# Skill Risk Classification
risk_levels:
  
  low:
    criteria:
      - no_network_access: true
      - no_system_access: true
      - readonly_file_access: true
      - no_sensitive_data: true
    examples:
      - "text-formatter"
      - "calculator"
      - "local-search"
    sandbox_required: false
    consent_required: false
  
  medium:
    criteria:
      - network_access: "approved_endpoints_only"
      - no_system_access: true
      - file_access: "workspace_only"
      - may_access_user_data: true
    examples:
      - "email-processor"
      - "file-organizer"
      - "weather-checker"
    sandbox_required: true
    consent_required: true
  
  high:
    criteria:
      - network_access: "external_apis"
      - system_access: "sandboxed_commands"
      - file_access: "user_directories"
      - accesses_sensitive_data: true
    examples:
      - "system-monitor"
      - "browser-automation"
      - "voice-assistant"
    sandbox_required: true
    consent_required: true
    approval_timeout: 300
  
  critical:
    criteria:
      - network_access: "unrestricted"
      - system_access: "command_execution"
      - file_access: "system_wide"
      - can_modify_system: true
    examples:
      - "system-admin"
      - "security-manager"
    sandbox_required: true
    network_isolated: true
    consent_required: true
    human_in_loop: true
    approval_timeout: 60
```


---

## 7. DYNAMIC PERMISSION ELEVATION

### 7.1 Elevation Architecture

```
+------------------------------------------------------------------+
|           DYNAMIC PERMISSION ELEVATION SYSTEM                    |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |              ELEVATION REQUEST                             |   |
|  |  Agent requests elevated permission for specific task     |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              ELEVATION EVALUATOR                           |   |
|  |                                                            |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |  |   Context   |  |    Risk     |  |   Trust     |        |   |
|  |  |   Analyzer  |  |   Assessor  |  |   Scorer    |        |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |                                                            |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |  |   Policy    |  |   User      |  |   Time      |        |   |
|  |  |   Checker   |  |   Intent    |  |   Window    |        |   |
|  |  |             |  |   Analyzer  |  |   Validator |        |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|              +---------------+---------------+                   |
|              |               |               |                   |
|              v               v               v                   |
|  +---------------+ +---------------+ +---------------+          |
|  |    DENY       | |   DEFER TO    | |    GRANT      |          |
|  |               | |    CONSENT    | |   ELEVATION   |          |
|  |  Risk too     | |  Needs user   | |  Temporary    |          |
|  |  high /       | |  approval     | |  elevation    |          |
|  |  Policy       | |  required     | |  granted      |          |
|  |  violation    | |               | |               |          |
|  +---------------+ +---------------+ +---------------+          |
|                                                                  |
+------------------------------------------------------------------+
```

### 7.2 Elevation Request Schema

```json
{
  "elevation_request": {
    "request_id": "uuid",
    "timestamp": "2026-01-15T10:30:00Z",
    
    "requestor": {
      "agent_id": "agent:core:001",
      "loop_id": "loop:email-management",
      "user_id": "user:alice",
      "current_role": "agent_operator_core",
      "current_permissions": ["file:read:workspace", "gmail:read:own"]
    },
    
    "requested_elevation": {
      "permission": "file:write:user_directories",
      "scope": "${USER_HOME}/Documents/Reports",
      "duration": 3600,
      "justification": "Need to save generated report to user Documents folder",
      "task_context": {
        "conversation_id": "conv:123",
        "task_type": "report_generation",
        "user_intent_confidence": 0.92
      }
    },
    
    "context": {
      "conversation_history": [
        {"role": "user", "content": "Generate a quarterly report"},
        {"role": "agent", "content": "I'll generate the report. Where should I save it?"},
        {"role": "user", "content": "Save it to my Documents folder"}
      ],
      "relevant_skills_used": ["report-generator", "file-manager"],
      "data_accessed": ["sales_data_q4", "customer_metrics"]
    }
  }
}
```

### 7.3 Elevation Decision Matrix

| Risk Score | Trust Score | User Intent | Policy Check | Decision | Action |
|------------|-------------|-------------|--------------|----------|--------|
| Low (<0.3) | High (>0.8) | Clear (>0.9) | Pass | **Auto-Grant** | Grant 1-hour elevation |
| Low (<0.3) | Medium (0.5-0.8) | Clear (>0.9) | Pass | **Auto-Grant** | Grant 30-min elevation |
| Medium (0.3-0.6) | High (>0.8) | Clear (>0.9) | Pass | **Defer to Consent** | Notify user, wait approval |
| Medium (0.3-0.6) | Medium (0.5-0.8) | Unclear (<0.9) | Pass | **Defer to Consent** | Require explicit approval |
| High (>0.6) | Any | Any | Any | **Deny** | Log, notify admin |
| Any | Low (<0.5) | Any | Any | **Deny** | Require re-authentication |
| Any | Any | Any | Fail | **Deny** | Policy violation |

### 7.4 Elevation Constraints

```yaml
# Elevation Constraints Configuration
elevation_constraints:
  
  time_limits:
    low_risk_high_trust: 3600      # 1 hour
    low_risk_medium_trust: 1800    # 30 minutes
    medium_risk_high_trust: 600    # 10 minutes
    medium_risk_medium_trust: 300  # 5 minutes
  
  scope_constraints:
    file_access:
      - path_must_be_within_user_dirs: true
      - cannot_access_hidden_files: true
      - cannot_access_system_dirs: true
    
    network_access:
      - endpoint_must_be_pre_approved: true
      - tls_required: true
      - rate_limited: true
    
    system_access:
      - sandbox_required: true
      - command_allowlist_enforced: true
  
  usage_limits:
    max_operations_per_elevation: 100
    max_data_transfer: "100MB"
    max_concurrent_elevations: 3
  
  monitoring:
    audit_all_actions: true
    real_time_alerts: true
    anomaly_detection: true
```

---

## 8. PERMISSION AUDIT LOGGING

### 8.1 Audit Architecture

```
+------------------------------------------------------------------+
|              PERMISSION AUDIT SYSTEM                             |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |              AUDIT EVENT SOURCES                           |   |
|  |                                                            |   |
|  |  + Permission checks (allow/deny)                         |   |
|  |  + Role assignments/changes                               |   |
|  |  + Elevation requests/grants                              |   |
|  |  + Consent approvals/denials                              |   |
|  |  + Skill executions                                       |   |
|  |  + Resource access (file, network, system)                |   |
|  |  + Policy violations                                      |   |
|  |  + Authentication events                                  |   |
|  |  + Configuration changes                                  |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              AUDIT EVENT PROCESSOR                         |   |
|  |                                                            |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |  |   Schema    |  |  Enrichment |  |  Filtering  |        |   |
|  |  |  Validator  |  |   Engine    |  |   Engine    |        |   |
|  |  +-------------+  +-------------+  +-------------+        |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|              +---------------+---------------+                   |
|              |               |               |                   |
|              v               v               v                   |
|  +---------------+ +---------------+ +---------------+          |
|  |   REAL-TIME   | |   SHORT-TERM  | |   LONG-TERM   |          |
|  |    STREAM     | |    STORAGE    | |    STORAGE    |          |
|  |               | |               | |               |          |
|  |  + Alerts     | |  + 30 days    | |  + 7 years    |          |
|  |  + Dashboard  | |  + Analytics  | |  + Compliance |          |
|  |  + SIEM feed  | |  + Search     | |  + Archive    |          |
|  +---------------+ +---------------+ +---------------+          |
|                                                                  |
+------------------------------------------------------------------+
```

### 8.2 Audit Event Schema

```json
{
  "$schema": "https://openclaw.io/schemas/audit-event/v1",
  "audit_event": {
    "event_id": "ae:uuid:v4",
    "timestamp": "2026-01-15T10:30:00.123Z",
    "event_type": "permission_check",
    "event_category": "authorization",
    "severity": "info",
    
    "actor": {
      "type": "agent",
      "agent_id": "agent:core:001",
      "loop_id": "loop:email-management",
      "user_id": "user:alice",
      "session_id": "sess:uuid",
      "authentication_method": "mfa",
      "ip_address": "192.168.1.100",
      "device_id": "device:desktop:001"
    },
    
    "action": {
      "type": "file:read",
      "target_resource": {
        "type": "file",
        "path": "C:\\Users\\Alice\\Documents\\report.pdf",
        "sensitivity": "confidential"
      },
      "context": {
        "conversation_id": "conv:123",
        "task_id": "task:456",
        "skill_id": "skill:document-reader"
      }
    },
    
    "authorization": {
      "decision": "allow",
      "decision_reason": "RBAC role allows file:read in user directories",
      "policies_evaluated": [
        {
          "policy_id": "rbac:file:user_scope",
          "result": "allow"
        },
        {
          "policy_id": "abac:time_based_file_access",
          "result": "allow"
        }
      ],
      "permissions_checked": ["file:read:user_directories"],
      "elevation_used": null
    },
    
    "result": {
      "success": true,
      "bytes_read": 1048576,
      "duration_ms": 45
    },
    
    "metadata": {
      "version": "1.0",
      "schema_hash": "sha256:abc123...",
      "integrity_signature": "sig:def456..."
    }
  }
}
```

### 8.3 Audit Event Types

| Category | Event Type | Description | Severity |
|----------|------------|-------------|----------|
| **Authentication** | auth:login | User/agent login | info |
| | auth:logout | User/agent logout | info |
| | auth:mfa_required | MFA challenge issued | info |
| | auth:failed | Authentication failure | warning |
| **Authorization** | permission:check | Permission check performed | info |
| | permission:denied | Permission denied | warning |
| | permission:elevated | Permission elevation granted | info |
| | permission:revoked | Permission revoked | info |
| **Role Management** | role:assigned | Role assigned to user/agent | info |
| | role:removed | Role removed from user/agent | info |
| | role:modified | Role permissions modified | warning |
| **Consent** | consent:requested | User consent requested | info |
| | consent:granted | User consent granted | info |
| | consent:denied | User consent denied | info |
| | consent:expired | Consent expired | info |
| **Skill** | skill:loaded | Skill loaded | info |
| | skill:executed | Skill executed | info |
| | skill:blocked | Skill execution blocked | warning |
| **System** | config:changed | Configuration changed | warning |
| | policy:modified | Policy modified | warning |
| | anomaly:detected | Anomaly detected | critical |

### 8.4 Audit Storage Configuration

```yaml
# Audit Logging Configuration
audit_logging:
  
  # Event collection
  collection:
    enabled: true
    buffer_size: 10000
    flush_interval: 5
    
  # Real-time processing
  real_time:
    enabled: true
    stream_to_siem: true
    alert_on:
      - event_type: "permission:denied"
        threshold: 5
        window: 60
      - event_type: "auth:failed"
        threshold: 3
        window: 300
      - event_type: "anomaly:detected"
        threshold: 1
    
  # Short-term storage (hot)
  short_term:
    backend: "elasticsearch"
    retention_days: 30
    index_pattern: "openclaw-audit-%{+YYYY.MM.dd}"
    replicas: 1
    shards: 5
    
  # Long-term storage (cold)
  long_term:
    backend: "s3"
    retention_years: 7
    compression: "gzip"
    encryption: "aes-256"
    
  # Integrity protection
  integrity:
    sign_events: true
    hash_algorithm: "sha256"
    signing_key: "${AUDIT_SIGNING_KEY}"
    chain_hashing: true
    
  # Privacy
  privacy:
    mask_pii: true
    fields_to_mask:
      - "actor.ip_address"
      - "action.target_resource.path"
      - "result.data"
    retention_for_masked: 90
```

---

## 9. USER CONSENT FLOWS

### 9.1 Consent Architecture

```
+------------------------------------------------------------------+
|              USER CONSENT SYSTEM                                 |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |              CONSENT TRIGGERS                              |   |
|  |                                                            |   |
|  |  + High-risk permission request                           |   |
|  |  + Sensitive data access                                  |   |
|  |  + External communication (email, SMS, call)              |   |
|  |  + File deletion/modification                             |   |
|  |  + System command execution                               |   |
|  |  + Permission elevation                                   |   |
|  |  + New skill execution                                    |   |
|  |  + Unusual activity pattern                               |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              CONSENT REQUEST BUILDER                       |   |
|  |                                                            |   |
|  |  1. Determine consent type (implicit/explicit)            |   |
|  |  2. Build consent context                                 |   |
|  |  3. Format user-friendly message                          |   |
|  |  4. Set timeout and urgency                               |   |
|  |  5. Route to appropriate channel                          |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|              +---------------+---------------+                   |
|              |               |               |                   |
|              v               v               v                   |
|  +---------------+ +---------------+ +---------------+          |
|  |     UI        | |    VOICE      | |   MOBILE      |          |
|  |   PROMPT      | |   PROMPT      | |   PUSH        |          |
|  |               | |               | |               |          |
|  |  Desktop      | |  TTS: "Agent  | |  Notification |          |
|  |  notification | |  wants to..." | |  with actions |          |
|  +---------------+ +---------------+ +---------------+          |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              CONSENT RESPONSE PROCESSOR                    |   |
|  |                                                            |   |
|  |  + Parse user response                                    |   |
|  |  + Validate within timeout                                |   |
|  |  + Record consent decision                                |   |
|  |  + Apply time/scope limits                                |   |
|  |  + Resume agent execution                                 |   |
|  |                                                            |   |
|  +-----------------------------------------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

### 9.2 Consent Types

| Type | Use Case | User Action | Duration |
|------|----------|-------------|----------|
| **Implicit** | Low-risk, routine operations | Notification only | Single operation |
| **Explicit** | Medium-risk operations | Click/button approval | Session or time-limited |
| **Detailed** | High-risk operations | Full explanation + approval | Single operation |
| **Emergency** | Critical operations | Override capability | One-time |
| **Pre-authorized** | User-preferred operations | Pre-configured | Until revoked |

### 9.3 Consent Request Schema

```json
{
  "$schema": "https://openclaw.io/schemas/consent-request/v1",
  "consent_request": {
    "request_id": "cr:uuid:v4",
    "timestamp": "2026-01-15T10:30:00Z",
    "expires_at": "2026-01-15T10:35:00Z",
    
    "type": "explicit",
    "priority": "high",
    
    "requestor": {
      "agent_id": "agent:core:001",
      "agent_name": "Core Assistant",
      "loop_name": "Email Management"
    },
    
    "action": {
      "type": "gmail:send",
      "description": "Send email",
      "details": {
        "to": ["client@example.com"],
        "subject": "Quarterly Report",
        "has_attachments": true,
        "attachment_size": "2.5MB"
      }
    },
    
    "context": {
      "conversation_summary": "You asked me to send the quarterly report to the client",
      "risk_level": "medium",
      "data_sensitivity": "confidential"
    },
    
    "presentation": {
      "title": "Send Email?",
      "message": "The agent wants to send an email to client@example.com with the subject 'Quarterly Report' and a 2.5MB attachment.",
      "details_url": "/consent/details/cr:uuid",
      "actions": [
        {
          "id": "approve_once",
          "label": "Approve This Time",
          "style": "primary"
        },
        {
          "id": "approve_always",
          "label": "Always Approve for This Contact",
          "style": "secondary"
        },
        {
          "id": "deny",
          "label": "Deny",
          "style": "danger"
        },
        {
          "id": "review",
          "label": "Review Email First",
          "style": "default"
        }
      ]
    },
    
    "constraints": {
      "timeout_seconds": 300,
      "auto_deny_on_timeout": true,
      "allow_delegation": false
    }
  }
}
```

### 9.4 Consent UI Configuration

```yaml
# Consent UI Configuration
consent_ui:
  
  # Desktop notification
  desktop:
    enabled: true
    style: "modal"
    position: "center"
    show_details: true
    show_preview: true
    timeout_indicator: true
    
  # Voice prompt
  voice:
    enabled: true
    tts_voice: "en-US-Neural2-D"
    speak_details: true
    listen_for_response: true
    supported_responses: ["yes", "no", "always", "never", "tell me more"]
    
  # Mobile push
  mobile:
    enabled: true
    priority: "high"
    actions_in_notification: true
    require_unlock_for_sensitive: true
    
  # Display rules
  display_rules:
    low_risk:
      show_notification: true
      require_explicit_action: false
      timeout: 30
    
    medium_risk:
      show_notification: true
      require_explicit_action: true
      timeout: 300
      show_preview: true
    
    high_risk:
      show_modal: true
      require_explicit_action: true
      timeout: 600
      show_full_details: true
      require_mfa: true
```


---

## 10. PERMISSION REVOCATION

### 10.1 Revocation Architecture

```
+------------------------------------------------------------------+
|              PERMISSION REVOCATION SYSTEM                        |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+   |
|  |              REVOCATION TRIGGERS                           |   |
|  |                                                            |   |
|  |  + User-initiated revocation                              |   |
|  |  + Administrator revocation                               |   |
|  |  + Automatic expiry                                       |   |
|  |  + Security incident response                             |   |
|  |  + Policy violation                                       |   |
|  |  + Anomaly detection                                      |   |
|  |  + Session termination                                    |   |
|  |  + Role change                                            |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|                              v                                   |
|  +-----------------------------------------------------------+   |
|  |              REVOCATION PROCESSOR                          |   |
|  |                                                            |   |
|  |  1. Validate revocation request                           |   |
|  |  2. Determine scope of revocation                         |   |
|  |  3. Identify affected permissions                         |   |
|  |  4. Revoke from policy engine                             |   |
|  |  5. Invalidate active sessions/tokens                     |   |
|  |  6. Terminate active operations                           |   |
|  |  7. Notify affected agents/users                          |   |
|  |  8. Audit log the revocation                              |   |
|  |                                                            |   |
|  +---------------------------+-------------------------------+   |
|                              |                                   |
|              +---------------+---------------+                   |
|              |               |               |                   |
|              v               v               v                   |
|  +---------------+ +---------------+ +---------------+          |
|  |   IMMEDIATE   | |   SCHEDULED   | |   GRACEFUL    |          |
|  |               | |               | |               |          |
|  |  Instant      | |  Revoke at    | |  Allow active |          |
|  |  termination  | |  specified    | |  operations   |          |
|  |               | |  time         | |  to complete  |          |
|  +---------------+ +---------------+ +---------------+          |
|                                                                  |
+------------------------------------------------------------------+
```

### 10.2 Revocation Types

| Type | Description | Use Case | Effect |
|------|-------------|----------|--------|
| **Immediate** | Instant revocation | Security incident | All operations stop immediately |
| **Scheduled** | Revoke at future time | Planned changes | Grace period before revocation |
| **Graceful** | Allow completion | Non-critical | Active operations complete, new blocked |
| **Cascading** | Revoke dependencies | Role removal | All dependent permissions revoked |
| **Temporary** | Time-limited | Suspension | Automatic restoration after period |

### 10.3 Revocation Request Schema

```json
{
  "$schema": "https://openclaw.io/schemas/revocation-request/v1",
  "revocation_request": {
    "request_id": "rr:uuid:v4",
    "timestamp": "2026-01-15T10:30:00Z",
    
    "requestor": {
      "type": "user",
      "user_id": "user:alice",
      "role": "user_owner"
    },
    
    "target": {
      "type": "agent_permission",
      "agent_id": "agent:core:001",
      "permission": "gmail:send",
      "scope": "*"
    },
    
    "revocation": {
      "type": "immediate",
      "reason": "User no longer wants agent to send emails",
      "effective_immediately": true,
      "terminate_active": true
    },
    
    "notification": {
      "notify_agent": true,
      "notify_user": true,
      "message": "Email sending permission has been revoked"
    }
  }
}
```

### 10.4 Automatic Revocation Rules

```yaml
# Automatic Revocation Configuration
auto_revocation:
  
  # Session-based revocation
  session:
    revoke_on_logout: true
    revoke_on_timeout: true
    session_max_duration: 86400  # 24 hours
  
  # Time-based revocation
  time_based:
    temporary_elevations:
      max_duration: 3600
      auto_revoke: true
    
    consent_grants:
      low_risk_max_age: 2592000    # 30 days
      medium_risk_max_age: 604800  # 7 days
      high_risk_max_age: 86400     # 1 day
  
  # Security-based revocation
  security:
    anomaly_detected:
      action: "immediate_revoke"
      scope: "affected_permissions"
      notify_admin: true
    
    threat_level_increase:
      threshold: "high"
      action: "suspend_all"
      require_investigation: true
    
    policy_violation:
      action: "immediate_revoke"
      scope: "violating_permission"
      escalate_repeated: true
  
  # Inactivity-based revocation
  inactivity:
    agent_inactive_days: 30
    user_inactive_days: 90
    action: "suspend_pending_review"
```

---

## 11. IMPLEMENTATION REFERENCE

### 11.1 Configuration File Structure

```
/config
├── permissions/
│   ├── rbac/
│   │   ├── roles.yaml
│   │   ├── role_bindings.yaml
│   │   └── permission_definitions.yaml
│   ├── abac/
│   │   ├── policies/
│   │   │   ├── time_based.yaml
│   │   │   ├── network_based.yaml
│   │   │   └── risk_based.yaml
│   │   ├── attributes/
│   │   │   ├── subject_attrs.yaml
│   │   │   ├── resource_attrs.yaml
│   │   │   └── environment_attrs.yaml
│   │   └── policy_sets.yaml
│   ├── skills/
│   │   ├── skill_manifest_schema.json
│   │   ├── risk_classification.yaml
│   │   └── execution_profiles.yaml
│   ├── elevation/
│   │   ├── elevation_policies.yaml
│   │   ├── trust_scoring.yaml
│   │   └── risk_assessment.yaml
│   ├── consent/
│   │   ├── consent_types.yaml
│   │   ├── ui_config.yaml
│   │   └── auto_approval_rules.yaml
│   ├── audit/
│   │   ├── event_schema.yaml
│   │   ├── storage_config.yaml
│   │   └── retention_policies.yaml
│   └── revocation/
│       ├── revocation_types.yaml
│       └── auto_revocation_rules.yaml
└── security/
    ├── encryption.yaml
    ├── key_management.yaml
    └── threat_detection.yaml
```

### 11.2 Core Permission Check Flow (Pseudocode)

```python
class PermissionSystem:
    """Main permission system orchestrator"""
    
    def __init__(self):
        self.rbac_engine = RBACEngine()
        self.abac_engine = ABACEngine()
        self.skill_validator = SkillValidator()
        self.elevation_manager = ElevationManager()
        self.consent_manager = ConsentManager()
        self.audit_logger = AuditLogger()
    
    async def check_permission(self, request: PermissionRequest) -> PermissionResult:
        """Main permission check entry point"""
        
        # Step 1: Check RBAC
        rbac_result = await self.rbac_engine.check(request)
        
        # Step 2: Check ABAC (context-aware)
        abac_result = await self.abac_engine.evaluate(request)
        
        # Step 3: Check for active elevation
        elevation = await self.elevation_manager.get_active_elevation(
            request.subject_id, request.permission
        )
        
        # Step 4: Combine decisions
        combined = self._combine_decisions(rbac_result, abac_result, elevation)
        
        # Step 5: Check if consent required
        if combined.decision == "allow" and self._consent_required(request):
            consent = await self.consent_manager.request_consent(
                self._build_consent_request(request)
            )
            if consent.outcome != "granted":
                combined = PermissionResult(
                    decision="deny",
                    reason=f"Consent not granted: {consent.outcome}"
                )
        
        # Step 6: Audit log
        await self.audit_logger.log_permission_check(request, combined)
        
        return combined
    
    def _combine_decisions(self, rbac: RBACResult, abac: ABACResult,
                          elevation: Elevation) -> PermissionResult:
        """Combine RBAC, ABAC, and elevation decisions"""
        
        # ABAC explicit deny overrides all
        if abac.decision == "deny":
            return PermissionResult(decision="deny", reason=abac.reason)
        
        # RBAC deny (unless elevated)
        if rbac.decision == "deny" and not elevation:
            return PermissionResult(decision="deny", reason=rbac.reason)
        
        # ABAC conditional allow
        if abac.decision == "allow_conditional":
            return PermissionResult(
                decision="allow",
                constraints=abac.constraints
            )
        
        # Elevation overrides RBAC deny
        if elevation and rbac.decision == "deny":
            return PermissionResult(
                decision="allow",
                via_elevation=elevation.token,
                constraints=elevation.constraints
            )
        
        # Both allow
        if rbac.decision == "allow" and abac.decision in ["allow", "allow_conditional"]:
            return PermissionResult(decision="allow")
        
        # Default deny
        return PermissionResult(decision="deny", reason="No positive authorization")
```

### 11.3 Integration Points

| Component | Integration Method | Data Flow |
|-----------|-------------------|-----------|
| **GPT-5.2 Engine** | API calls | Permission checks before tool execution |
| **Agentic Loops** | Middleware | Permission context passed to each loop |
| **Skill Registry** | Manifest validation | Permission requirements enforced at load |
| **Gmail Client** | OAuth scope validation | Token scopes checked against permissions |
| **Browser Control** | Proxy/Extension | URL access filtered by permissions |
| **TTS/STT** | Capability checks | Audio I/O permissions verified |
| **Twilio** | API key + scope validation | Call/SMS permissions enforced |
| **File System** | Filter driver | All file operations intercepted |
| **Windows API** | Hook/library | System calls checked against permissions |

### 11.4 Security Checklist

```yaml
# Permission System Security Checklist
security_checklist:
  
  pre_deployment:
    - "All default passwords changed"
    - "RBAC roles configured with least privilege"
    - "ABAC policies tested with edge cases"
    - "Skill sandbox configured and tested"
    - "Audit logging enabled and verified"
    - "Encryption keys generated and secured"
    - "Consent flows tested"
    - "Revocation procedures documented"
  
  runtime:
    - "Permission checks logged"
    - "Failed checks alerted"
    - "Anomaly detection active"
    - "Session timeouts enforced"
    - "Elevation tokens rotated"
    - "Audit logs backed up"
  
  regular_maintenance:
    - "Review role assignments monthly"
    - "Audit permission usage quarterly"
    - "Update risk classifications"
    - "Test revocation procedures"
    - "Review auto-revocation rules"
    - "Update threat detection rules"
```

---

## APPENDIX A: Permission Decision Matrix

| RBAC | ABAC | Elevation | Consent | Final Decision |
|------|------|-----------|---------|----------------|
| Allow | Allow | None | Not Required | **ALLOW** |
| Allow | Allow | None | Required + Granted | **ALLOW** |
| Allow | Allow | None | Required + Denied | **DENY** |
| Allow | Deny | None | - | **DENY** |
| Deny | Allow | Active | - | **ALLOW** (via elevation) |
| Deny | Deny | Active | - | **DENY** (ABAC overrides) |
| Deny | Allow | None | - | **DENY** |

## APPENDIX B: Glossary

| Term | Definition |
|------|------------|
| **ABAC** | Attribute-Based Access Control |
| **RBAC** | Role-Based Access Control |
| **PDP** | Policy Decision Point |
| **PEP** | Policy Enforcement Point |
| **PBAC** | Policy-Based Access Control |
| **ReBAC** | Relationship-Based Access Control |
| **MFA** | Multi-Factor Authentication |
| **PII** | Personally Identifiable Information |
| **PHI** | Protected Health Information |
| **PCI** | Payment Card Industry |

## APPENDIX C: Permission System Class Diagram

```
+------------------------------------------------------------------+
|                    CLASS DIAGRAM                                 |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------+                                            |
|  | PermissionSystem |                                            |
|  +------------------+                                            |
|  | - rbac_engine    |                                            |
|  | - abac_engine    |                                            |
|  | - elevation_mgr  |                                            |
|  | - consent_mgr    |                                            |
|  | - audit_logger   |                                            |
|  +------------------+                                            |
|  | + check_permission()                                          |
|  | + grant_elevation()                                           |
|  | + revoke_permission()                                         |
|  +------------------+                                            |
|           |                                                      |
|    +------+------+                                               |
|    |             |                                               |
|    v             v                                               |
| +---------+  +---------+                                         |
| |RBAC     |  |ABAC     |                                         |
| |Engine   |  |Engine   |                                         |
| +---------+  +---------+                                         |
| | + check()|  | + evaluate()                                      |
| +---------+  +---------+                                         |
|                                                                  |
| +------------------+  +------------------+                       |
| |ElevationManager  |  |ConsentManager    |                       |
| +------------------+  +------------------+                       |
| | + request()      |  | + request()      |                       |
| | + grant()        |  | + grant()        |                       |
| | + revoke()       |  | + deny()         |                       |
| +------------------+  +------------------+                       |
|                                                                  |
| +------------------+  +------------------+                       |
| |SkillValidator    |  |AuditLogger       |                       |
| +------------------+  +------------------+                       |
| | + validate()     |  | + log()          |                       |
| | + get_profile()  |  | + query()        |                       |
| +------------------+  +------------------+                       |
|                                                                  |
+------------------------------------------------------------------+
```

## APPENDIX D: Sample Permission Check Sequence

```
User Request -> Agent -> Permission Check
                                    |
                                    v
                           +------------------+
                           |  RBAC Check      |
                           +------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                  ALLOW                           DENY
                    |                               |
                    v                               v
           +------------------+          +------------------+
           |  ABAC Evaluate   |          | Check Elevation  |
           +------------------+          +------------------+
                    |                               |
        +-----------+-----------+                   |
        |           |           |                   |
      ALLOW    CONDITIONAL    DENY               Active?
        |           |           |                   |
        v           v           v                   v
   +--------+  +--------+  +--------+         +--------+  +--------+
   | Check  |  | Apply  |  |  DENY  |         | ALLOW  |  |  DENY  |
   |Consent?|  |Constr. |  |        |         |w/Elev. |  |        |
   +--------+  +--------+  +--------+         +--------+  +--------+
        |                                              
   +----+----+                                         
   |         |                                         
Required  Not Req.                                     
   |         |                                         
   v         v                                         
+------+  +------+                                     
|Request| | ALLOW|                                     
|Consent| |      |                                     
+------+  +------+                                     
   |
   v
+------+  +------+
|Grant |  | DENY |
+------+  +------+
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-15*
*Classification: Technical Specification*

---

## SUMMARY

This comprehensive permission system architecture provides:

1. **RBAC** - Role-based access control with 5 hierarchical roles and granular permissions
2. **ABAC** - Attribute-based policies for context-aware access decisions
3. **Permission Granularity** - 5 levels each for file, network, and system access
4. **Skill-Level Permissions** - Manifest-based permission validation and sandboxing
5. **Dynamic Elevation** - Risk-based temporary permission elevation with consent
6. **Audit Logging** - Complete event tracking with integrity protection
7. **User Consent Flows** - Multi-channel consent with configurable types
8. **Permission Revocation** - Multiple revocation types with automatic rules

The architecture addresses OpenClaw security vulnerabilities through:
- Principle of least privilege by default
- Skill sandboxing to prevent permission inheritance
- Multi-layered policy enforcement
- Complete audit trails for compliance
- Human-in-the-loop for sensitive operations
- Dynamic adaptation to threat levels
