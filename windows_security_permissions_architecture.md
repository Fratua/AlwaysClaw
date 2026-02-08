# Windows Security and Permissions Architecture
## WinClaw AI Agent System - Windows 10 Security Specification

**Version:** 1.0  
**Platform:** Windows 10 (Build 19041+)  
**Classification:** Security Architecture  
**Date:** 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Security Architecture Overview](#2-security-architecture-overview)
3. [Windows ACL and Permission Models](#3-windows-acl-and-permission-models)
4. [Access Tokens and Impersonation](#4-access-tokens-and-impersonation)
5. [UAC (User Account Control) Handling](#5-uac-user-account-control-handling)
6. [Running with Elevated Privileges Safely](#6-running-with-elevated-privileges-safely)
7. [Sandboxing Options on Windows](#7-sandboxing-options-on-windows)
8. [Credential Isolation and Protection](#8-credential-isolation-and-protection)
9. [Audit Logging for Security Events](#9-audit-logging-for-security-events)
10. [Principle of Least Privilege Implementation](#10-principle-of-least-privilege-implementation)
11. [Security Descriptor Reference](#11-security-descriptor-reference)
12. [Implementation Code Samples](#12-implementation-code-samples)

---

## 1. Executive Summary

### 1.1 Threat Model

The WinClaw AI Agent system faces the "Lethal Trifecta" of security risks:

| Risk Factor | Description | Mitigation Priority |
|-------------|-------------|---------------------|
| **Private Data Access** | Gmail, files, registry, credentials | Critical |
| **Untrusted Content** | LLM outputs, web content, user input | Critical |
| **External Communication** | Twilio, APIs, network operations | High |

### 1.2 Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY PERIMETER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐    │   │
│  │  │  GPT-5.2    │  │  Agent Core │  │   Agentic Loop Manager  │    │   │
│  │  │  Engine     │  │  Controller │  │   (15 Hardcoded Loops)  │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                      SECURITY MIDDLEWARE LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Input     │  │   Skill     │  │  Permission │  │   Credential    │    │
│  │  Validator  │  │  Sandbox    │  │   Manager   │  │     Vault       │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      WINDOWS SECURITY SUBSYSTEM                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │  Access     │  │    UAC      │  │   ACL/SDDL  │  │  Audit Logging  │    │
│  │   Tokens    │  │  Handler    │  │   Engine    │  │    (ETW/Sysmon) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      ISOLATION LAYER                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ AppContainer│  │   Job       │  │   Process   │  │   Windows       │    │
│  │  Sandbox    │  │  Objects    │  │   ACLs      │  │   Sandbox       │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Security Architecture Overview

### 2.1 Core Security Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZERO TRUST ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │   NEVER     │      │   NEVER     │      │   ALWAYS    │    │
│   │    TRUST    │  →   │   ASSUME    │  →   │   VERIFY    │    │
│   │   INPUT     │      │  BREACH     │      │   ACCESS    │    │
│   └─────────────┘      └─────────────┘      └─────────────┘    │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              DEFENSE IN DEPTH STRATEGY                   │   │
│   │  Layer 1: Input Sanitization  → Block injection attacks  │   │
│   │  Layer 2: Skill Sandboxing    → Isolate untrusted code   │   │
│   │  Layer 3: Permission Gates    → Enforce least privilege  │   │
│   │  Layer 4: Credential Vault    → Protect secrets          │   │
│   │  Layer 5: Audit Logging       → Detect anomalies         │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Security Component Mapping

| Component | Purpose | Windows API | Risk Level |
|-----------|---------|-------------|------------|
| ACL Manager | Permission enforcement | advapi32.dll | High |
| Token Manager | Identity/impersonation | advapi32.dll | Critical |
| UAC Handler | Elevation control | shell32.dll | Critical |
| Sandbox | Code isolation | AppContainer/Job Objects | High |
| Credential Vault | Secret storage | DPAPI/CNG/KeyGuard | Critical |
| Audit Logger | Security monitoring | ETW/EventLog | High |

---

## 3. Windows ACL and Permission Models

### 3.1 ACL Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACCESS CONTROL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SECURITY DESCRIPTOR (SD)                          │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│   │  │   Owner     │  │   Group     │  │    DACL     │  │   SACL    │  │   │
│   │  │   SID       │  │   SID       │  │  (Access)   │  │ (Audit)   │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    DISCRETIONARY ACL (DACL)                          │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│   │  │   ACE 1     │  │   ACE 2     │  │   ACE 3     │  │   ...     │  │   │
│   │  │ Allow Read  │  │ Deny Write  │  │ Allow Exec  │  │           │  │   │
│   │  │ User:Bob    │  │ Group:Users │  │ SID:Agent   │  │           │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ACCESS MASK STRUCTURE                             │   │
│   │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────────┐ │   │
│   │  │ 31 │ 30 │ 29 │ 28 │ 27 │ 26 │ 25 │ 24 │ 23 │ 22 │ 21 │ 20..16 │ │   │
│   │  │GR│GW│GX│GA│MAX│   │   │   │   │   │   │Standard Rights│ │   │
│   │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────────┘ │   │
│   │  ┌────────────────────────────────────────────────────────────────┐ │   │
│   │  │ 15..8                    │ 7..0                                │ │   │
│   │  │ Reserved                 │ Specific Rights (object-dependent)  │ │   │
│   │  └────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Standard Access Rights

```cpp
// ============================================
// Windows Standard Access Rights
// ============================================

// Generic Rights (map to specific rights)
#define GENERIC_READ                     0x80000000
#define GENERIC_WRITE                    0x40000000
#define GENERIC_EXECUTE                  0x20000000
#define GENERIC_ALL                      0x10000000

// Standard Rights (common to all object types)
#define DELETE                           0x00010000
#define READ_CONTROL                     0x00020000
#define WRITE_DAC                        0x00040000  // Modify DACL
#define WRITE_OWNER                      0x00080000  // Change owner
#define SYNCHRONIZE                      0x00100000

// Combined Standard Rights
#define STANDARD_RIGHTS_READ             READ_CONTROL
#define STANDARD_RIGHTS_WRITE            READ_CONTROL
#define STANDARD_RIGHTS_EXECUTE          READ_CONTROL
#define STANDARD_RIGHTS_REQUIRED         0x000F0000
#define STANDARD_RIGHTS_ALL              0x001F0000

// Specific Rights for Agent Operations
#define AGENT_RIGHT_EXECUTE_SKILL        0x00000001
#define AGENT_RIGHT_ACCESS_CREDENTIALS   0x00000002
#define AGENT_RIGHT_SYSTEM_OPERATIONS    0x00000004
#define AGENT_RIGHT_NETWORK_ACCESS       0x00000008
#define AGENT_RIGHT_UI_AUTOMATION        0x00000010
#define AGENT_RIGHT_FILE_OPERATIONS      0x00000020
#define AGENT_RIGHT_REGISTRY_ACCESS      0x00000040
#define AGENT_RIGHT_PROCESS_CONTROL      0x00000080
```

### 3.3 Security Descriptor Definition Language (SDDL)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SDDL STRING FORMAT REFERENCE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SDDL Format: D:(DACL)S:(SACL)O:(Owner)G:(Group)                           │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  ACE String Format:                                                 │   │
│   │  ace_type;ace_flags;rights;object_guid;inherit_guid;sid             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ACE Types:                                                                │
│   ┌─────────┬────────────────────────────────────────────────────────┐     │
│   │   A     │  ACCESS_ALLOWED_ACE_TYPE                               │     │
│   │   D     │  ACCESS_DENIED_ACE_TYPE                                │     │
│   │   OA    │  ACCESS_ALLOWED_OBJECT_ACE_TYPE                        │     │
│   │   OD    │  ACCESS_DENIED_OBJECT_ACE_TYPE                         │     │
│   │   AU    │  SYSTEM_AUDIT_ACE_TYPE                                 │     │
│   │   AL    │  SYSTEM_ALARM_ACE_TYPE                                 │     │
│   └─────────┴────────────────────────────────────────────────────────┘     │
│                                                                              │
│   Common Rights Strings:                                                    │
│   ┌─────────┬────────────────────────────────────────────────────────┐     │
│   │   GA    │  GENERIC_ALL                                           │     │
│   │   GR    │  GENERIC_READ                                          │     │
│   │   GW    │  GENERIC_WRITE                                         │     │
│   │   GX    │  GENERIC_EXECUTE                                       │     │
│   │   RC    │  READ_CONTROL                                          │     │
│   │   SD    │  DELETE                                                │     │
│   │   WD    │  WRITE_DAC                                             │     │
│   │   WO    │  WRITE_OWNER                                           │     │
│   └─────────┴────────────────────────────────────────────────────────┘     │
│                                                                              │
│   Well-Known SIDs:                                                          │
│   ┌──────────────┬─────────────────────────────────────────────────────┐   │
│   │   SY         │  Local System (S-1-5-18)                            │   │
│   │   BA         │  Built-in Administrators (S-1-5-32-544)             │   │
│   │   BU         │  Built-in Users (S-1-5-32-545)                      │   │
│   │   WD         │  Everyone (S-1-1-0)                                 │   │
│   │   AU         │  Authenticated Users (S-1-5-11)                     │   │
│   │   NS         │  Network Service (S-1-5-20)                         │   │
│   │   LS         │  Local Service (S-1-5-19)                           │   │
│   │   WR         │  Write Restricted (S-1-5-33)                        │   │
│   │   AC         │  Application Container (S-1-15-2-1)                 │   │
│   └──────────────┴─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Agent-Specific SDDL Templates

```cpp
// ============================================
// SDDL Templates for WinClaw Agent System
// ============================================

// Template 1: Restricted Agent Process
// - No admin rights
// - Read-only file access
// - No credential access
#define AGENT_SDDL_RESTRICTED \
    L"D:(D;;GA;;;BA)(A;;GRGX;;;AU)(A;;GR;;;WD)" \
    L"S:(AU;FA;GA;;;WD)"

// Template 2: Standard Agent Process
// - User-level rights
// - File read/write in app directory
// - Registry read
#define AGENT_SDDL_STANDARD \
    L"D:(A;;GRGW;;;AU)(D;;GA;;;BA)(A;;GRGX;;;BU)" \
    L"(A;;GR;;;AC)" \
    L"S:(AU;SAFA;GRGW;;;AU)"

// Template 3: Elevated Agent Process
// - Admin rights for system operations
// - Credential access allowed
// - Full system access
#define AGENT_SDDL_ELEVATED \
    L"D:(A;;GA;;;BA)(A;;GRGWGX;;;AU)(A;;GR;;;BU)" \
    L"S:(AU;SAFA;GA;;;WD)(AU;SAFA;GA;;;BA)"

// Template 4: Skill Sandbox Process
// - Highly restricted
// - No network access
// - No file system access outside sandbox
#define AGENT_SDDL_SANDBOX \
    L"D:(D;;GA;;;WD)(D;;GA;;;AU)(A;;GRGX;;;AC)(A;;GR;;;SY)" \
    L"S:(AU;FA;GA;;;WD)"

// Template 5: Credential Vault Access
// - Only SYSTEM and specific service SID
// - No user access
#define AGENT_SDDL_CREDENTIAL_VAULT \
    L"D:(A;;GA;;;SY)(D;;GA;;;AU)(D;;GA;;;BA)(D;;GA;;;WD)" \
    L"S:(AU;FA;GA;;;SY)(AU;FA;GA;;;WD)"
```

---

## 4. Access Tokens and Impersonation

### 4.1 Token Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ACCESS TOKEN ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ACCESS TOKEN STRUCTURE                            │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN HEADER                                               │   │   │
│   │  │  - TokenType (Primary/Impersonation)                        │   │   │
│   │  │  - ImpersonationLevel (Anonymous/Identification/            │   │   │
│   │  │    Impersonation/Delegation)                                │   │   │
│   │  │  - TokenFlags                                               │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN USER                                                 │   │   │
│   │  │  - User SID (e.g., S-1-5-21-...-1001)                       │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN GROUPS                                               │   │   │
│   │  │  - Group SIDs with attributes (Enabled, EnabledByDefault,   │   │   │
│   │  │    UseForDenyOnly, Owner, Resource, etc.)                   │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN PRIVILEGES                                           │   │   │
│   │  │  - Privilege LUIDs with attributes (Enabled, EnabledByDefault)│  │   │
│   │  │  - e.g., SeDebugPrivilege, SeBackupPrivilege, etc.          │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN OWNER                                                │   │   │
│   │  │  - Default owner SID for created objects                    │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN PRIMARY GROUP                                        │   │   │
│   │  │  - Default primary group SID for created objects            │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN DEFAULT DACL                                         │   │   │
│   │  │  - Default DACL for created objects                         │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN SOURCE                                               │   │   │
│   │  │  - Source name and identifier                               │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  TOKEN STATISTICS                                           │   │   │
│   │  │  - Creation time, Expiration time, etc.                     │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Impersonation Levels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPERSONATION LEVEL HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Level 0: SecurityAnonymous                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  - Server cannot impersonate or identify the client                 │   │
│   │  - Token contains no user information                               │   │
│   │  - Used for: Untrusted connections, public access                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   Level 1: SecurityIdentification                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  - Server can identify the client but cannot impersonate            │   │
│   │  - Can query token for SID/groups                                   │   │
│   │  - Cannot access resources as the client                            │   │
│   │  - Used for: Access checking, logging                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   Level 2: SecurityImpersonation (DEFAULT)                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  - Server can impersonate client on local system                    │   │
│   │  - Can access local resources as the client                         │   │
│   │  - Cannot access remote resources as the client                     │   │
│   │  - Used for: Local service operations, agent skills                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   Level 3: SecurityDelegation                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  - Server can impersonate client on local AND remote systems        │   │
│   │  - Can access network resources as the client                       │   │
│   │  - Requires: Kerberos delegation, trusted for delegation            │   │
│   │  - Used for: Distributed systems, domain operations                 │   │
│   │  - RISK: Highest privilege, use with extreme caution                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Required Privileges for Agent Operations

```cpp
// ============================================
// Windows Privileges for AI Agent System
// ============================================

// Standard User Privileges (Always Available)
#define SE_CHANGE_NOTIFY_PRIVILEGE          L"SeChangeNotifyPrivilege"        // Required for directory change notifications
#define SE_INCREASE_WORKING_SET_PRIVILEGE   L"SeIncreaseWorkingSetPrivilege"  // Increase process working set
#define SE_SHUTDOWN_PRIVILEGE               L"SeShutdownPrivilege"            // Shutdown system
#define SE_TIME_ZONE_PRIVILEGE              L"SeTimeZonePrivilege"            // Change time zone
#define SE_UNDOCK_PRIVILEGE                 L"SeUndockPrivilege"              // Undock laptop

// Administrator Privileges (Require Elevation)
#define SE_BACKUP_PRIVILEGE                 L"SeBackupPrivilege"              // Backup files (bypass ACL)
#define SE_RESTORE_PRIVILEGE                L"SeRestorePrivilege"             // Restore files (bypass ACL)
#define SE_SYSTEM_TIME_PRIVILEGE            L"SeSystemTimePrivilege"          // Change system time
#define SE_TAKE_OWNERSHIP_PRIVILEGE         L"SeTakeOwnershipPrivilege"       // Take ownership of objects
#define SE_DEBUG_PRIVILEGE                  L"SeDebugPrivilege"               // Debug processes
#define SE_LOAD_DRIVER_PRIVILEGE            L"SeLoadDriverPrivilege"          // Load device drivers
#define SE_SYSTEM_PROFILE_PRIVILEGE         L"SeSystemProfilePrivilege"       // Profile system performance
#define SE_PROF_SINGLE_PROCESS_PRIVILEGE    L"SeProfileSingleProcessPrivilege"// Profile single process
#define SE_INCREASE_BASE_PRIORITY_PRIVILEGE L"SeIncreaseBasePriorityPrivilege"// Increase scheduling priority
#define SE_CREATE_PAGEFILE_PRIVILEGE        L"SeCreatePagefilePrivilege"      // Create pagefile
#define SE_CREATE_SYMBOLIC_LINK_PRIVILEGE   L"SeCreateSymbolicLinkPrivilege"  // Create symbolic links
#define SE_MANAGE_VOLUME_PRIVILEGE          L"SeManageVolumePrivilege"        // Manage volumes
#define SE_SECURITY_PRIVILEGE               L"SeSecurityPrivilege"            // Manage auditing
#define SE_RELABEL_PRIVILEGE                L"SeRelabelPrivilege"             // Modify mandatory labels
#define SE_IMPERSONATE_PRIVILEGE            L"SeImpersonatePrivilege"         // Impersonate clients
#define SE_CREATE_GLOBAL_PRIVILEGE          L"SeCreateGlobalPrivilege"        // Create global objects
#define SE_ASSIGNPRIMARYTOKEN_PRIVILEGE     L"SeAssignPrimaryTokenPrivilege"  // Replace process token
#define SE_INCREASE_QUOTA_PRIVILEGE         L"SeIncreaseQuotaPrivilege"       // Adjust process quotas
#define SE_MACHINE_ACCOUNT_PRIVILEGE        L"SeMachineAccountPrivilege"      // Add workstations to domain
#define SE_TCB_PRIVILEGE                    L"SeTcbPrivilege"                 // Act as part of OS
#define SE_TRUSTED_CRED_MAN_ACCESS_PRIVILEGE L"SeTrustedCredManAccessPrivilege"// Access credential manager
#define SE_DELEGATE_PRIVILEGE               L"SeDelegatePrivilege"            // Enable delegation
#define SE_LOCK_MEMORY_PRIVILEGE            L"SeLockMemoryPrivilege"          // Lock pages in memory
#define SE_SYNC_AGENT_PRIVILEGE             L"SeSyncAgentPrivilege"           // Sync directory service

// Agent-Specific Privilege Sets
namespace AgentPrivileges {
    // Minimal privileges for sandboxed skills
    const LPCWSTR Minimal[] = {
        SE_CHANGE_NOTIFY_PRIVILEGE,
        nullptr
    };
    
    // Standard user privileges
    const LPCWSTR Standard[] = {
        SE_CHANGE_NOTIFY_PRIVILEGE,
        SE_INCREASE_WORKING_SET_PRIVILEGE,
        SE_SHUTDOWN_PRIVILEGE,
        SE_TIME_ZONE_PRIVILEGE,
        SE_UNDOCK_PRIVILEGE,
        SE_CREATE_SYMBOLIC_LINK_PRIVILEGE,
        nullptr
    };
    
    // Administrator privileges for system operations
    const LPCWSTR Administrator[] = {
        SE_BACKUP_PRIVILEGE,
        SE_RESTORE_PRIVILEGE,
        SE_TAKE_OWNERSHIP_PRIVILEGE,
        SE_DEBUG_PRIVILEGE,
        SE_SECURITY_PRIVILEGE,
        SE_IMPERSONATE_PRIVILEGE,
        SE_CREATE_GLOBAL_PRIVILEGE,
        SE_INCREASE_QUOTA_PRIVILEGE,
        SE_ASSIGNPRIMARYTOKEN_PRIVILEGE,
        nullptr
    };
    
    // Full system privileges (use with extreme caution)
    const LPCWSTR System[] = {
        SE_TCB_PRIVILEGE,
        SE_TRUSTED_CRED_MAN_ACCESS_PRIVILEGE,
        SE_LOAD_DRIVER_PRIVILEGE,
        SE_MANAGE_VOLUME_PRIVILEGE,
        SE_RELABEL_PRIVILEGE,
        SE_DELEGATE_PRIVILEGE,
        SE_LOCK_MEMORY_PRIVILEGE,
        nullptr
    };
}
```

---

## 5. UAC (User Account Control) Handling

### 5.1 UAC Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UAC ARCHITECTURE & FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    UAC CONSENT FLOW                                  │   │
│   │                                                                      │   │
│   │  Application          UAC Components          User Decision         │   │
│   │  ┌─────────┐         ┌─────────────┐         ┌─────────────┐       │   │
│   │  │ Request │────────▶│ Application │────────▶│   Consent   │       │   │
│   │  │ Elevate │         │  Info (AI)  │         │   Dialog    │       │   │
│   │  └─────────┘         └─────────────┘         └──────┬──────┘       │   │
│   │         │                    │                      │              │   │
│   │         │                    │                      ▼              │   │
│   │         │                    │              ┌─────────────┐        │   │
│   │         │                    │              │  Approve /  │        │   │
│   │         │                    │              │   Deny      │        │   │
│   │         │                    │              └──────┬──────┘        │   │
│   │         │                    │                     │               │   │
│   │         ▼                    ▼                     ▼               │   │
│   │  ┌─────────────────────────────────────────────────────────┐      │   │
│   │  │              ELEVATION DECISION ENGINE                   │      │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │      │   │
│   │  │  │  Consent    │  │  Credential │  │   Auto-Approve  │  │      │   │
│   │  │  │   Prompt    │  │   Prompt    │  │  (White-listed) │  │      │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────────┘  │      │   │
│   │  └─────────────────────────────────────────────────────────┘      │   │
│   │                              │                                     │   │
│   │                              ▼                                     │   │
│   │  ┌─────────────────────────────────────────────────────────┐      │   │
│   │  │              ELEVATED PROCESS CREATION                   │      │   │
│   │  │  - Create process with admin token                       │      │   │
│   │  │  - Apply integrity level (High)                          │      │   │
│   │  │  - Set UIAccess if needed                                │      │   │
│   │  └─────────────────────────────────────────────────────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    INTEGRITY LEVELS                                  │   │
│   │                                                                      │   │
│   │  Level      SID                    Description                      │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  Untrusted  S-1-16-0               Anonymous/Restricted             │   │
│   │  Low        S-1-16-4096            Low integrity (IE, sandbox)      │   │
│   │  Medium     S-1-16-8192            Standard user                    │   │
│   │  High       S-1-16-12288           Elevated admin                   │   │
│   │  System     S-1-16-16384           System processes                 │   │
│   │  Protected  S-1-16-20480           Protected processes              │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 UAC Elevation Methods

```cpp
// ============================================
// UAC Elevation Methods Comparison
// ============================================

/*
Method 1: ShellExecute with "runas" verb
-----------------------------------------
Pros: Simple, shows UAC prompt
Cons: Cannot wait for completion, no output capture
Use for: Launching external admin tools
*/

/*
Method 2: CreateProcessWithLogonW
----------------------------------
Pros: Full control over process
Cons: Requires credentials, complex
Use for: Service accounts, scheduled tasks
*/

/*
Method 3: COM Elevation Moniker
-------------------------------
Pros: Can elevate COM objects without new process
Cons: Requires COM registration
Use for: In-process elevation for specific operations
*/

/*
Method 4: Scheduled Task
------------------------
Pros: Runs with highest privileges, can bypass UAC in some cases
Cons: Requires task scheduler, delayed execution
Use for: Background operations, maintenance tasks
*/

/*
Method 5: Token Elevation
-------------------------
Pros: Direct token manipulation
Cons: Complex, requires existing admin token
Use for: Advanced scenarios, token-based operations
*/
```

---

## 6. Running with Elevated Privileges Safely

### 6.1 Privilege Escalation Security Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAFE ELEVATION ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    PRIVILEGE ESCALATION FLOW                         │   │
│   │                                                                      │   │
│   │  Standard Mode              Elevation Gate         Elevated Mode    │   │
│   │  ┌─────────────┐            ┌─────────────┐        ┌─────────────┐  │   │
│   │  │ User Mode   │───────────▶│  Security   │───────▶│ Admin Mode  │  │   │
│   │  │ (Medium IL) │  Request   │   Check     │ Grant  │ (High IL)   │  │   │
│   │  └─────────────┘            └─────────────┘        └─────────────┘  │   │
│   │         │                          │                      │          │   │
│   │         │                          │                      │          │   │
│   │         ▼                          ▼                      ▼          │   │
│   │  ┌───────────────────────────────────────────────────────────────┐   │   │
│   │  │                    SECURITY GATES                              │   │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│   │   │
│   │  │  │  Operation  │  │  Permission │  │   Audit & Logging       ││   │   │
│   │  │  │  Validation │  │  Check      │  │   (All elevation        ││   │   │
│   │  │  │             │  │             │  │    attempts logged)     ││   │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘│   │   │
│   │  └───────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ELEVATION PATTERNS                                │   │
│   │                                                                      │   │
│   │  Pattern 1: Just-in-Time Elevation                                  │   │
│   │  - Elevate only when needed                                         │   │
│   │  - Revert immediately after operation                               │   │
│   │  - Use for: Single privileged operations                            │   │
│   │                                                                      │   │
│   │  Pattern 2: Separate Elevated Process                               │   │
│   │  - Spawn helper process with elevation                              │   │
│   │  - Communicate via IPC (named pipes, sockets)                       │   │
│   │  - Use for: Extended privileged operations                          │   │
│   │                                                                      │   │
│   │  Pattern 3: COM Elevation Moniker                                   │   │
│   │  - Elevate COM object in-process                                    │   │
│   │  - No separate process needed                                       │   │
│   │  - Use for: COM-based privileged operations                         │   │
│   │                                                                      │   │
│   │  Pattern 4: Scheduled Task Elevation                                │   │
│   │  - Create task with highest privileges                              │   │
│   │  - Run on-demand or schedule                                        │   │
│   │  - Use for: Background/administrative tasks                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Sandboxing Options on Windows

### 7.1 Sandboxing Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WINDOWS SANDBOXING OPTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Option 1: AppContainer (Windows 8+)                                 │   │
│   │  ─────────────────────────────────                                   │   │
│   │  Security Level: HIGH                                                │   │
│   │  Complexity: MEDIUM                                                  │   │
│   │  Performance: GOOD                                                   │   │
│   │  Compatibility: Windows 8+                                           │   │
│   │                                                                      │   │
│   │  Features:                                                           │   │
│   │  • Isolated SID (S-1-15-2-...)                                       │   │
│   │  • Capability-based access control                                   │   │
│   │  • Network isolation                                                 │   │
│   │  • File system virtualization                                        │   │
│   │  • Registry virtualization                                           │   │
│   │                                                                      │   │
│   │  Best for: Universal Windows Platform apps, browser sandboxes        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Option 2: Windows Job Objects                                       │   │
│   │  ─────────────────────────────                                       │   │
│   │  Security Level: MEDIUM                                              │   │
│   │  Complexity: LOW                                                     │   │
│   │  Performance: EXCELLENT                                              │   │
│   │  Compatibility: Windows 2000+                                        │   │
│   │                                                                      │   │
│   │  Features:                                                           │   │
│   │  • Process grouping and control                                      │   │
│   │  • CPU/memory limits                                                 │   │
│   │  • UI restrictions                                                   │   │
│   │  • Token filtering                                                   │   │
│   │  • No file/network isolation (needs ACLs)                            │   │
│   │                                                                      │   │
│   │  Best for: Process management, resource limiting                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Option 3: Windows Sandbox (Windows 10 Pro+)                         │   │
│   │  ────────────────────────────────────                                │   │
│   │  Security Level: VERY HIGH                                           │   │
│   │  Complexity: LOW (for end users)                                     │   │
│   │  Performance: MODERATE (VM overhead)                                 │   │
│   │  Compatibility: Windows 10 Pro+, Hyper-V                             │   │
│   │                                                                      │   │
│   │  Features:                                                           │   │
│   │  • Full VM isolation                                                 │   │
│   │  • Temporary environment                                             │   │
│   │  • No persistence (by default)                                       │   │
│   │  • GPU acceleration                                                  │   │
│   │  • Clipboard/file sharing                                            │   │
│   │                                                                      │   │
│   │  Best for: Testing untrusted code, malware analysis                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Option 4: Process ACLs + Restricted Tokens                          │   │
│   │  ───────────────────────────────────────                             │   │
│   │  Security Level: MEDIUM-HIGH                                         │   │
│   │  Complexity: HIGH                                                    │   │
│   │  Performance: EXCELLENT                                              │   │
│   │  Compatibility: Windows XP+                                          │   │
│   │                                                                      │   │
│   │  Features:                                                           │   │
│   │  • Custom ACLs on process                                            │   │
│   │  • Restricted token filtering                                        │   │
│   │  • Deny-only SIDs                                                    │   │
│   │  • Write-restricted tokens                                           │   │
│   │                                                                      │   │
│   │  Best for: Custom security policies, legacy Windows                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Option 5: Integrity Levels + UIPI                                   │   │
│   │  ─────────────────────────────────                                   │   │
│   │  Security Level: MEDIUM                                              │   │
│   │  Complexity: MEDIUM                                                  │   │
│   │  Performance: EXCELLENT                                              │   │
│   │  Compatibility: Windows Vista+                                       │   │
│   │                                                                      │   │
│   │  Features:                                                           │   │
│   │  • Mandatory Integrity Control (MIC)                                 │   │
│   │  • User Interface Privilege Isolation (UIPI)                         │   │
│   │  • No-read-up, no-write-up                                           │   │
│   │  • Window message filtering                                          │   │
│   │                                                                      │   │
│   │  Best for: UI isolation, browser processes                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Recommended Sandboxing Strategy for WinClaw

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WINCLAW SANDBOXING STRATEGY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    LAYERED SANDBOX APPROACH                          │   │
│   │                                                                      │   │
│   │  Layer 1: Job Objects                                                │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Group skill processes together                              │   │   │
│   │  │  • Limit CPU/memory usage                                      │   │   │
│   │  │  • Apply UI restrictions                                       │   │   │
│   │  │  • Kill all processes on job close                             │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 2: Restricted Tokens                                        │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Remove unnecessary privileges                               │   │   │
│   │  │  • Add deny-only SIDs                                          │   │   │
│   │  │  • Filter token groups                                         │   │   │
│   │  │  • Create write-restricted token                               │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 3: Integrity Levels                                         │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Run skills at Low Integrity Level                           │   │   │
│   │  │  • Apply No-Write-Up policy                                    │   │   │
│   │  │  • UIPI prevents window message attacks                        │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 4: ACL Restrictions                                         │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Custom DACL on process handle                               │   │   │
│   │  │  • Restrict file system access                                 │   │   │
│   │  │  • Limit registry access                                       │   │   │
│   │  │  • Network access controls                                     │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 5: AppContainer (Optional)                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Full isolation for high-risk skills                         │   │   │
│   │  │  • Capability-based access                                     │   │   │
│   │  │  • Network isolation                                           │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Credential Isolation and Protection

### 8.1 Credential Vault Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CREDENTIAL VAULT ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CREDENTIAL STORAGE LAYERS                         │   │
│   │                                                                      │   │
│   │  Layer 1: Windows Credential Manager (CredMan)                       │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Built-in Windows credential storage                       │   │   │
│   │  │  • Encrypted with user's master key                          │   │   │
│   │  │  • Accessible via CredRead/CredWrite APIs                    │   │   │
│   │  │  • Types: Generic, Domain Password, Certificate              │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 2: Data Protection API (DPAPI)                                │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • User-specific encryption                                    │   │   │
│   │  │  • Tied to user password                                       │   │   │
│   │  │  • Can add entropy for additional protection                   │   │   │
│   │  │  • Functions: CryptProtectData/CryptUnprotectData            │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 3: Cryptography API: Next Generation (CNG)                    │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Modern cryptographic API                                    │   │   │
│   │  │  • Supports AES, RSA, ECDSA, etc.                              │   │   │
│   │  │  • Key isolation in LSASS                                      │   │   │
│   │  │  • Functions: NCrypt*, BCrypt*                                 │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Layer 4: Windows Hello / Passport                                   │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Biometric/PIN-based authentication                          │   │   │
│   │  │  • Hardware-backed keys (TPM)                                  │   │   │
│   │  │  • WebAuthn/FIDO2 support                                      │   │   │
│   │  │  • Most secure option                                          │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CREDENTIAL ACCESS CONTROL                         │   │
│   │                                                                      │   │
│   │  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │   │
│   │  │   Agent     │────────▶│   Vault     │────────▶│  Credential │   │   │
│   │  │   Process   │ Request │   Service   │ Validate│   Store     │   │   │
│   │  │  (Medium)   │         │  (System)   │         │  (Encrypted)│   │   │
│   │  └─────────────┘         └─────────────┘         └─────────────┘   │   │
│   │         │                       │                       │           │   │
│   │         │                       │                       │           │   │
│   │         ▼                       ▼                       ▼           │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │                    ACCESS RULES                              │   │   │
│   │  │  • Agent must authenticate to vault service                  │   │   │
│   │  │  • Each credential has ACL (which agents can access)         │   │   │
│   │  │  • Audit all access attempts                                 │   │   │
│   │  │  • Credentials never exposed in plaintext to agent           │   │   │
│   │  │  • Automatic rotation support                                │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Audit Logging for Security Events

### 9.1 Audit Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SECURITY AUDIT ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    AUDIT EVENT SOURCES                               │   │
│   │                                                                      │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│   │  │   Agent     │  │   Skill     │  │   System    │  │   Windows  │ │   │
│   │  │   Core      │  │   Sandbox   │  │   Monitor   │  │   Security │ │   │
│   │  │   Events    │  │   Events    │  │   Events    │  │   Events   │ │   │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │   │
│   │         │                │                │               │        │   │
│   │         └────────────────┴────────────────┴───────────────┘        │   │
│   │                              │                                     │   │
│   │                              ▼                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │                    AUDIT COLLECTOR                           │   │   │
│   │  │  • Normalize events from all sources                         │   │   │
│   │  │  • Add metadata (timestamp, process, user)                   │   │   │
│   │  │  • Enrich with threat intelligence                           │   │   │
│   │  │  • Filter based on policy                                    │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                     │   │
│   │                              ▼                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │                    AUDIT PROCESSORS                          │   │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │   │
│   │  │  │   Windows   │  │    ETW      │  │   File/Database     │  │   │   │
│   │  │  │  Event Log  │  │  Provider   │  │   Storage           │  │   │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                     │   │
│   │                              ▼                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │                    ALERTING & RESPONSE                       │   │   │
│   │  │  • Real-time anomaly detection                               │   │   │
│   │  │  • Threshold-based alerting                                  │   │   │
│   │  │  • Automated response actions                                │   │   │
│   │  │  • SIEM integration                                          │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CRITICAL EVENTS TO AUDIT                          │   │
│   │                                                                      │   │
│   │  Authentication Events:                                              │   │
│   │  • User login/logout, credential access, privilege escalation       │   │
│   │                                                                      │   │
│   │  Authorization Events:                                               │   │
│   │  • Permission denied, access granted, policy violations             │   │
│   │                                                                      │   │
│   │  Agent Events:                                                       │   │
│   │  • Skill execution, loop activation, external API calls             │   │
│   │                                                                      │   │
│   │  System Events:                                                      │   │
│   │  • Process creation/termination, file access, registry changes      │   │
│   │                                                                      │   │
│   │  Security Events:                                                    │   │
│   │  • Injection attempts, sandbox escape, credential exposure          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Principle of Least Privilege Implementation

### 10.1 Least Privilege Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRINCIPLE OF LEAST PRIVILEGE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    PRIVILEGE TIERS                                   │   │
│   │                                                                      │   │
│   │  Tier 1: Untrusted Skills (Minimal Privileges)                       │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • No file system access outside sandbox                     │   │   │
│   │  │  • No network access                                         │   │   │
│   │  │  • No registry access                                        │   │   │
│   │  │  • No credential access                                      │   │   │
│   │  │  • CPU/Memory limits enforced                                │   │   │
│   │  │  • Time-limited execution                                    │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Tier 2: Standard Skills (User-Level Privileges)                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Read access to user files                                 │   │   │
│   │  │  • Write access to app directory only                        │   │   │
│   │  │  • Network access to allowed endpoints                       │   │   │
│   │  │  • Registry read access                                      │   │   │
│   │  │  • No credential access                                      │   │   │
│   │  │  • UI automation allowed                                     │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Tier 3: Trusted Skills (Extended Privileges)                        │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Full file system access (user context)                    │   │   │
│   │  │  • Full network access                                       │   │   │
│   │  │  • Registry read/write access                                │   │   │
│   │  │  • Credential access (via vault service)                     │   │   │
│   │  │  • Service control (specific services)                       │   │   │
│   │  │  • Process control (own processes)                           │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                       │   │
│   │                              ▼                                       │   │
│   │  Tier 4: System Skills (Administrative Privileges)                   │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  • Full system access (when elevated)                        │   │   │
│   │  │  • Service control (all services)                            │   │   │
│   │  │  • Process control (all processes)                           │   │   │
│   │  │  • System configuration changes                              │   │   │
│   │  │  • Requires explicit elevation                               │   │   │
│   │  │  • Full audit logging                                        │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    DYNAMIC PRIVILEGE ADJUSTMENT                      │   │
│   │                                                                      │   │
│   │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │   │
│   │  │   Request   │────▶│   Policy    │────▶│   Grant/    │           │   │
│   │  │   Access    │     │   Engine    │     │   Deny      │           │   │
│   │  └─────────────┘     └─────────────┘     └─────────────┘           │   │
│   │         │                  │                    │                    │   │
│   │         │                  │                    ▼                    │   │
│   │         │                  │           ┌─────────────┐               │   │
│   │         │                  │           │   Adjust    │               │   │
│   │         │                  │           │   Token     │               │   │
│   │         │                  │           └─────────────┘               │   │
│   │         │                  │                                         │   │
│   │         ▼                  ▼                                         │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │                    POLICY RULES                              │    │   │
│   │  │  • Time-based restrictions (business hours only)             │    │   │
│   │  │  • Location-based restrictions (corporate network)           │    │   │
│   │  │  • User consent requirements                                 │    │   │
│   │  │  • Rate limiting                                             │    │   │
│   │  │  • Context-aware decisions                                   │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Security Descriptor Reference

### 11.1 Common Security Descriptors

```cpp
// ============================================
// Security Descriptor Reference for WinClaw
// ============================================

// SE_OBJECT_TYPE Constants
#define SE_UNKNOWN_OBJECT_TYPE     0
#define SE_FILE_OBJECT             1
#define SE_SERVICE                 2
#define SE_PRINTER                 3
#define SE_REGISTRY_KEY            4
#define SE_LMSHARE                 5
#define SE_KERNEL_OBJECT           6
#define SE_WINDOW_OBJECT           7
#define SE_DS_OBJECT               8
#define SE_DS_OBJECT_ALL           9
#define SE_PROVIDER_DEFINED_OBJECT 10
#define SE_WMIGUID_OBJECT          11
#define SE_REGISTRY_WOW64_32KEY    12

// SECURITY_INFORMATION Constants
#define OWNER_SECURITY_INFORMATION     0x00000001
#define GROUP_SECURITY_INFORMATION     0x00000002
#define DACL_SECURITY_INFORMATION      0x00000004
#define SACL_SECURITY_INFORMATION      0x00000008
#define LABEL_SECURITY_INFORMATION     0x00000010
#define ATTRIBUTE_SECURITY_INFORMATION 0x00000020
#define SCOPE_SECURITY_INFORMATION     0x00000040
#define BACKUP_SECURITY_INFORMATION    0x00010000

// ACE Flags
#define OBJECT_INHERIT_ACE         0x01
#define CONTAINER_INHERIT_ACE      0x02
#define NO_PROPAGATE_INHERIT_ACE   0x04
#define INHERIT_ONLY_ACE           0x08
#define INHERITED_ACE              0x10
#define SUCCESSFUL_ACCESS_ACE_FLAG 0x40
#define FAILED_ACCESS_ACE_FLAG     0x80

// SDDL Component Examples
// -----------------------

// Example 1: Full access for admins, read for users
// D:(A;;GA;;;BA)(A;;GR;;;BU)
// D: - DACL
// (A;;GA;;;BA) - Allow Generic All to Built-in Administrators
// (A;;GR;;;BU) - Allow Generic Read to Built-in Users

// Example 2: Deny write for guests, allow read for everyone
// D:(D;;GW;;;BG)(A;;GR;;;WD)
// D: - DACL
// (D;;GW;;;BG) - Deny Generic Write to Built-in Guests
// (A;;GR;;;WD) - Allow Generic Read to Everyone

// Example 3: Complex with inheritance
// D:(A;CI;GA;;;BA)(A;CI;GRGW;;;AU)(A;CI;GR;;;WD)
// D: - DACL
// (A;CI;GA;;;BA) - Allow Generic All with Container Inherit to Admins
// (A;CI;GRGW;;;AU) - Allow Read/Write with Container Inherit to Auth Users
// (A;CI;GR;;;WD) - Allow Read with Container Inherit to Everyone

// Example 4: Audit all failed access attempts
// S:(AU;FAFA;WD;;;WD)
// S: - SACL
// (AU;FAFA;WD;;;WD) - Audit Failed Access for Everyone
```

### 11.2 Security Descriptor Helper Functions

```cpp
// ============================================
// Security Descriptor Helper Functions (C++)
// ============================================

#include <windows.h>
#include <sddl.h>
#include <aclapi.h>
#include <iostream>
#include <string>

#pragma comment(lib, "advapi32.lib")

namespace WinClawSecurity {

    // Convert SDDL string to security descriptor
    PSECURITY_DESCRIPTOR SDDLToSecurityDescriptor(const std::wstring& sddl) {
        PSECURITY_DESCRIPTOR sd = nullptr;
        ULONG sdSize = 0;
        
        if (!ConvertStringSecurityDescriptorToSecurityDescriptor(
            sddl.c_str(),
            SDDL_REVISION_1,
            &sd,
            &sdSize)) {
            return nullptr;
        }
        
        return sd;
    }

    // Convert security descriptor to SDDL string
    std::wstring SecurityDescriptorToSDDL(PSECURITY_DESCRIPTOR sd, 
        SECURITY_INFORMATION info = DACL_SECURITY_INFORMATION) {
        LPWSTR sddlString = nullptr;
        
        if (!ConvertSecurityDescriptorToStringSecurityDescriptor(
            sd,
            SDDL_REVISION_1,
            info,
            &sddlString,
            nullptr)) {
            return L"";
        }
        
        std::wstring result(sddlString);
        LocalFree(sddlString);
        
        return result;
    }

    // Apply SDDL to file
    bool ApplySDDLToFile(const std::wstring& filePath, const std::wstring& sddl) {
        PSECURITY_DESCRIPTOR sd = SDDLToSecurityDescriptor(sddl);
        if (!sd) return false;
        
        BOOL daclPresent = FALSE;
        PACL dacl = nullptr;
        BOOL daclDefaulted = FALSE;
        
        if (!GetSecurityDescriptorDacl(sd, &daclPresent, &dacl, &daclDefaulted)) {
            LocalFree(sd);
            return false;
        }
        
        DWORD result = SetNamedSecurityInfo(
            const_cast<LPWSTR>(filePath.c_str()),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            nullptr,
            nullptr,
            daclPresent ? dacl : nullptr,
            nullptr
        );
        
        LocalFree(sd);
        
        return result == ERROR_SUCCESS;
    }

    // Apply SDDL to registry key
    bool ApplySDDLToRegistry(const std::wstring& keyPath, const std::wstring& sddl) {
        PSECURITY_DESCRIPTOR sd = SDDLToSecurityDescriptor(sddl);
        if (!sd) return false;
        
        BOOL daclPresent = FALSE;
        PACL dacl = nullptr;
        BOOL daclDefaulted = FALSE;
        
        if (!GetSecurityDescriptorDacl(sd, &daclPresent, &dacl, &daclDefaulted)) {
            LocalFree(sd);
            return false;
        }
        
        DWORD result = SetNamedSecurityInfo(
            const_cast<LPWSTR>(keyPath.c_str()),
            SE_REGISTRY_KEY,
            DACL_SECURITY_INFORMATION,
            nullptr,
            nullptr,
            daclPresent ? dacl : nullptr,
            nullptr
        );
        
        LocalFree(sd);
        
        return result == ERROR_SUCCESS;
    }

    // Create explicit access entry
    EXPLICIT_ACCESS CreateExplicitAccess(const std::wstring& trustee,
        DWORD accessRights,
        ACCESS_MODE accessMode,
        DWORD inheritance = NO_INHERITANCE) {
        EXPLICIT_ACCESS ea = {};
        
        ea.grfAccessPermissions = accessRights;
        ea.grfAccessMode = accessMode;
        ea.grfInheritance = inheritance;
        
        ea.Trustee.TrusteeForm = TRUSTEE_IS_NAME;
        ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
        ea.Trustee.ptstrName = const_cast<LPWSTR>(trustee.c_str());
        
        return ea;
    }

    // Build ACL from explicit access entries
    PACL BuildAcl(const std::vector<EXPLICIT_ACCESS>& entries) {
        PACL acl = nullptr;
        
        DWORD result = SetEntriesInAcl(
            static_cast<ULONG>(entries.size()),
            const_cast<PEXPLICIT_ACCESS>(entries.data()),
            nullptr,
            &acl
        );
        
        if (result != ERROR_SUCCESS) {
            return nullptr;
        }
        
        return acl;
    }

    // Get current process security descriptor
    std::wstring GetCurrentProcessSDDL(SECURITY_INFORMATION info = DACL_SECURITY_INFORMATION) {
        HANDLE hProcess = GetCurrentProcess();
        PSECURITY_DESCRIPTOR sd = nullptr;
        DWORD sdSize = 0;
        
        // Get required size
        GetKernelObjectSecurity(hProcess, info, nullptr, 0, &sdSize);
        
        sd = (PSECURITY_DESCRIPTOR)LocalAlloc(LPTR, sdSize);
        if (!sd) return L"";
        
        if (!GetKernelObjectSecurity(hProcess, info, sd, sdSize, &sdSize)) {
            LocalFree(sd);
            return L"";
        }
        
        std::wstring result = SecurityDescriptorToSDDL(sd, info);
        LocalFree(sd);
        
        return result;
    }

} // namespace WinClawSecurity
```

---

## 12. Implementation Code Samples

### 12.1 Complete Security Manager

```javascript
// ============================================
// WinClaw Security Manager - Complete Implementation
// ============================================

const { EventEmitter } = require('events');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

// Import security modules
const { ACLManager } = require('./acl-manager');
const { TokenManager } = require('./token-manager');
const { UACManager } = require('./uac-manager');
const { JobObjectSandbox } = require('./job-sandbox');
const { CredentialVault } = require('./credential-vault');
const { SecurityAuditLogger } = require('./audit-logger');

class WinClawSecurityManager extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.options = {
            appDataPath: options.appDataPath || path.join(process.env.LOCALAPPDATA, 'WinClaw'),
            logPath: options.logPath || path.join(process.env.LOCALAPPDATA, 'WinClaw', 'logs'),
            enableSandbox: options.enableSandbox !== false,
            enableAudit: options.enableAudit !== false,
            minIntegrityLevel: options.minIntegrityLevel || 'medium',
            maxSkillMemory: options.maxSkillMemory || 512 * 1024 * 1024,
            maxSkillCpuTime: options.maxSkillCpuTime || 60000,
            ...options
        };
        
        // Initialize components
        this.aclManager = new ACLManager();
        this.tokenManager = new TokenManager();
        this.uacManager = new UACManager();
        this.credentialVault = new CredentialVault({
            auditLogPath: path.join(this.options.logPath, 'credential-audit.log')
        });
        
        if (this.options.enableSandbox) {
            this.sandbox = new JobObjectSandbox({
                maxMemory: this.options.maxSkillMemory,
                maxCpuTime: this.options.maxSkillCpuTime
            });
        }
        
        if (this.options.enableAudit) {
            this.auditLogger = new SecurityAuditLogger({
                logDirectory: this.options.logPath,
                minSeverity: 1  // INFO
            });
        }
        
        this.initialized = false;
        this.securityPolicy = null;
    }

    /**
     * Initialize security manager
     */
    async initialize() {
        if (this.initialized) return;
        
        // Ensure directories exist
        this.ensureDirectories();
        
        // Load security policy
        await this.loadSecurityPolicy();
        
        // Initialize sandbox if enabled
        if (this.sandbox) {
            await this.sandbox.create('WinClawAgentJob');
        }
        
        // Log initialization
        this.logSecurityEvent('SECURITY_MANAGER_INITIALIZED', {
            version: '1.0',
            options: this.options
        });
        
        this.initialized = true;
        this.emit('initialized');
    }

    /**
     * Ensure required directories exist
     */
    ensureDirectories() {
        const dirs = [
            this.options.appDataPath,
            this.options.logPath,
            path.join(this.options.appDataPath, 'sandbox'),
            path.join(this.options.appDataPath, 'credentials')
        ];
        
        for (const dir of dirs) {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
        }
    }

    /**
     * Load security policy
     */
    async loadSecurityPolicy() {
        const policyPath = path.join(this.options.appDataPath, 'security-policy.json');
        
        if (fs.existsSync(policyPath)) {
            this.securityPolicy = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
        } else {
            // Create default policy
            this.securityPolicy = this.getDefaultSecurityPolicy();
            fs.writeFileSync(policyPath, JSON.stringify(this.securityPolicy, null, 2));
        }
    }

    /**
     * Get default security policy
     */
    getDefaultSecurityPolicy() {
        return {
            version: '1.0',
            skillPermissions: {
                fileRead: ['%USERPROFILE%\\Documents', '%LOCALAPPDATA%\\WinClaw'],
                fileWrite: ['%LOCALAPPDATA%\\WinClaw\\sandbox'],
                registryRead: ['HKCU\\Software\\WinClaw'],
                registryWrite: [],
                networkEndpoints: [
                    'api.openai.com',
                    'api.twilio.com',
                    'smtp.gmail.com',
                    'imap.gmail.com'
                ],
                allowedProcesses: ['notepad.exe', 'calc.exe'],
                maxExecutionTime: 300000
            },
            credentialPolicy: {
                allowedTargets: ['Gmail', 'Twilio', 'OpenAI'],
                requireEncryption: true,
                rotationDays: 90
            },
            auditPolicy: {
                logAllSkillExecutions: true,
                logCredentialAccess: true,
                logElevationAttempts: true,
                retentionDays: 30
            },
            sandboxPolicy: {
                maxMemoryMB: 512,
                maxCpuPercent: 50,
                killOnTimeout: true,
                uiRestrictions: true
            }
        };
    }

    /**
     * Execute skill with security controls
     * @param {string} skillName - Skill name
     * @param {Function} skillFunction - Skill function
     * @param {Object} context - Execution context
     */
    async executeSkill(skillName, skillFunction, context = {}) {
        const executionId = this.generateExecutionId();
        
        // Log skill start
        this.logSecurityEvent('SKILL_EXECUTION_START', {
            executionId,
            skillName,
            context
        });
        
        try {
            // Validate skill permissions
            await this.validateSkillPermissions(skillName, context);
            
            // Apply sandbox if enabled
            let sandboxHandle = null;
            if (this.sandbox && context.useSandbox !== false) {
                sandboxHandle = await this.createSkillSandbox(executionId);
            }
            
            // Execute skill with timeout
            const result = await this.executeWithTimeout(
                () => skillFunction(context),
                this.securityPolicy.skillPermissions.maxExecutionTime
            );
            
            // Log success
            this.logSecurityEvent('SKILL_EXECUTION_SUCCESS', {
                executionId,
                skillName,
                duration: Date.now() - context.startTime
            });
            
            return result;
            
        } catch (error) {
            // Log failure
            this.logSecurityEvent('SKILL_EXECUTION_FAILURE', {
                executionId,
                skillName,
                error: error.message
            });
            
            throw error;
        }
    }

    /**
     * Validate skill permissions
     */
    async validateSkillPermissions(skillName, context) {
        const policy = this.securityPolicy.skillPermissions;
        
        // Check if skill is allowed
        if (context.requiredPermission) {
            const hasPermission = await this.checkPermission(
                context.requiredPermission
            );
            
            if (!hasPermission) {
                throw new Error(`Skill '${skillName}' requires permission: ${context.requiredPermission}`);
            }
        }
        
        // Validate file access
        if (context.fileAccess) {
            for (const file of context.fileAccess) {
                if (!this.isPathAllowed(file, policy.fileRead)) {
                    throw new Error(`File access denied: ${file}`);
                }
            }
        }
        
        // Validate network access
        if (context.networkAccess) {
            for (const endpoint of context.networkAccess) {
                if (!policy.networkEndpoints.includes(endpoint)) {
                    throw new Error(`Network access denied: ${endpoint}`);
                }
            }
        }
    }

    /**
     * Check if user has permission
     */
    async checkPermission(permission) {
        // Implementation would check against current token privileges
        // For now, return based on running as admin
        if (permission === 'admin') {
            return this.uacManager.isRunningAsAdmin();
        }
        return true;
    }

    /**
     * Check if path is in allowed list
     */
    isPathAllowed(filePath, allowedPaths) {
        const normalizedPath = path.normalize(filePath).toLowerCase();
        
        for (const allowed of allowedPaths) {
            const expanded = allowed
                .replace('%USERPROFILE%', process.env.USERPROFILE)
                .replace('%LOCALAPPDATA%', process.env.LOCALAPPDATA)
                .replace('%APPDATA%', process.env.APPDATA);
            
            if (normalizedPath.startsWith(path.normalize(expanded).toLowerCase())) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * Create sandbox for skill execution
     */
    async createSkillSandbox(executionId) {
        const sandbox = new JobObjectSandbox({
            maxMemory: this.options.maxSkillMemory,
            maxCpuTime: this.options.maxSkillCpuTime,
            killOnClose: true,
            uiRestrictions: true
        });
        
        await sandbox.create(`WinClawSkill_${executionId}`);
        
        return sandbox;
    }

    /**
     * Execute function with timeout
     */
    executeWithTimeout(fn, timeout) {
        return Promise.race([
            fn(),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Execution timeout')), timeout)
            )
        ]);
    }

    /**
     * Store credential securely
     */
    async storeCredential(target, username, password) {
        // Validate against policy
        if (!this.securityPolicy.credentialPolicy.allowedTargets.includes(target)) {
            throw new Error(`Storing credentials for '${target}' is not allowed`);
        }
        
        const targetName = `WinClaw:${target}:${username}`;
        
        await this.credentialVault.storeCredential(
            targetName,
            username,
            password
        );
        
        this.logSecurityEvent('CREDENTIAL_STORED', {
            target,
            username
        });
    }

    /**
     * Retrieve credential
     */
    async retrieveCredential(target, username) {
        const targetName = `WinClaw:${target}:${username}`;
        
        const credential = await this.credentialVault.retrieveCredential(targetName);
        
        this.logSecurityEvent('CREDENTIAL_RETRIEVED', {
            target,
            username
        });
        
        return credential;
    }

    /**
     * Log security event
     */
    logSecurityEvent(eventType, data) {
        const event = {
            type: eventType,
            data,
            timestamp: new Date().toISOString(),
            processId: process.pid
        };
        
        this.emit('securityEvent', event);
        
        if (this.auditLogger) {
            this.auditLogger.log(eventType, data);
        }
    }

    /**
     * Generate unique execution ID
     */
    generateExecutionId() {
        return `exec_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    }

    /**
     * Get security status
     */
    getSecurityStatus() {
        return {
            initialized: this.initialized,
            isAdmin: this.uacManager.isRunningAsAdmin(),
            sandboxEnabled: !!this.sandbox,
            auditEnabled: !!this.auditLogger,
            policyVersion: this.securityPolicy?.version
        };
    }

    /**
     * Shutdown security manager
     */
    async shutdown() {
        if (this.sandbox) {
            this.sandbox.close();
        }
        
        this.logSecurityEvent('SECURITY_MANAGER_SHUTDOWN', {});
        
        this.initialized = false;
        this.emit('shutdown');
    }
}

module.exports = { WinClawSecurityManager };
```

---

## Appendix A: Security Checklist

### Pre-Deployment Security Checklist

| # | Check | Status |
|---|-------|--------|
| 1 | All skills run in sandbox with restricted token | [ ] |
| 2 | Credentials stored only in Windows Credential Manager | [ ] |
| 3 | UAC elevation requires explicit user approval | [ ] |
| 4 | All security events logged to audit trail | [ ] |
| 5 | Input validation on all LLM outputs | [ ] |
| 6 | Network access restricted to allowed endpoints | [ ] |
| 7 | File system access restricted to app directories | [ ] |
| 8 | Registry access restricted to app keys | [ ] |
| 9 | Process creation monitored and logged | [ ] |
| 10 | Memory and CPU limits enforced | [ ] |
| 11 | Automatic skill timeout configured | [ ] |
| 12 | Principle of least privilege applied | [ ] |
| 13 | Security policy configured and loaded | [ ] |
| 14 | Audit logs reviewed regularly | [ ] |
| 15 | Incident response plan documented | [ ] |

---

## Appendix B: Security Incident Response

### Incident Response Procedures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SECURITY INCIDENT RESPONSE FLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DETECTION                                                               │
│     • Monitor audit logs for anomalies                                      │
│     • Set up alerts for critical events                                     │
│     • Use Windows Event Log forwarding                                      │
│                                                                              │
│  2. CONTAINMENT                                                             │
│     • Immediately terminate suspicious skill processes                      │
│     • Revoke compromised credentials                                        │
│     • Block suspicious network connections                                  │
│     • Isolate affected agent instance                                       │
│                                                                              │
│  3. INVESTIGATION                                                           │
│     • Collect audit logs and system snapshots                               │
│     • Analyze process memory dumps                                          │
│     • Review file system changes                                            │
│     • Check registry modifications                                          │
│                                                                              │
│  4. ERADICATION                                                             │
│     • Remove malicious code or configurations                               │
│     • Patch vulnerabilities                                                 │
│     • Reset compromised credentials                                         │
│     • Update security policies                                              │
│                                                                              │
│  5. RECOVERY                                                                │
│     • Restore from known-good backups                                       │
│     • Verify system integrity                                               │
│     • Resume operations with enhanced monitoring                            │
│     • Document lessons learned                                              │
│                                                                              │
│  6. POST-INCIDENT                                                           │
│     • Conduct post-mortem analysis                                          │
│     • Update security procedures                                            │
│     • Train team on new threats                                             │
│     • Improve detection capabilities                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Document Information:**
- Version: 1.0
- Last Updated: 2025
- Classification: Security Architecture
- Author: WinClaw Security Team

**Related Documents:**
- Win32 API Integration Specification
- PowerShell Integration Specification
- Windows Filesystem & Registry Specification
- Multi-Agent Orchestration Architecture
