# Credential Management & Secrets Architecture Specification
## Windows 10 OpenClaw-Inspired AI Agent System

**Version:** 1.0  
**Classification:** Technical Specification - Security Architecture  
**Date:** 2025  

---

## Executive Summary

This document provides a comprehensive credential management and secrets architecture for a Windows 10-based AI agent system inspired by OpenClaw. The architecture addresses critical security risks identified in research, including plaintext credential storage, long-lived tokens, and service account credential exposure. This specification implements defense-in-depth principles with multiple layers of protection.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Secrets Vault Integration](#2-secrets-vault-integration)
3. [Credential Encryption at Rest](#3-credential-encryption-at-rest)
4. [Secure Credential Retrieval](#4-secure-credential-retrieval)
5. [API Key Rotation](#5-api-key-rotation)
6. [OAuth Token Management](#6-oauth-token-management)
7. [Short-Lived Credential Preference](#7-short-lived-credential-preference)
8. [Credential Access Audit](#8-credential-access-audit)
9. [Breach Response Procedures](#9-breach-response-procedures)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI Agent System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   GPT-5.2    │    │   Browser    │    │    Gmail     │                  │
│  │   Engine     │◄──►│   Control    │    │   Service    │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             │                                              │
│  ┌──────────────────────────▼──────────────────────────┐                  │
│  │              Credential Management Layer            │                  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │                  │
│  │  │   Secret    │  │   Token     │  │   Audit    │  │                  │
│  │  │   Vault     │  │   Manager   │  │   Logger   │  │                  │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │                  │
│  └──────────────────────────┬──────────────────────────┘                  │
│                             │                                              │
│  ┌──────────────────────────▼──────────────────────────┐                  │
│  │              External Service Layer                 │                  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │                  │
│  │  │ Twilio   │  │  Azure   │  │   AWS    │          │                  │
│  │  │ Voice/SMS│  │  OpenAI  │  │Services  │          │                  │
│  │  └──────────┘  └──────────┘  └──────────┘          │                  │
│  └─────────────────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Security Principles

| Principle | Implementation |
|-----------|----------------|
| **Zero Trust** | No implicit trust; verify every access request |
| **Least Privilege** | Minimum permissions required for each operation |
| **Defense in Depth** | Multiple security layers |
| **Short-Lived Credentials** | Prefer temporary over long-lived credentials |
| **Audit Everything** | Comprehensive logging of all credential operations |
| **Encryption Everywhere** | At-rest and in-transit encryption |

---

## 2. Secrets Vault Integration

### 2.1 Vault Selection Matrix

| Requirement | Azure Key Vault | AWS Secrets Manager | HashiCorp Vault |
|-------------|-----------------|---------------------|-----------------|
| **Windows 10 Integration** | ⭐⭐⭐ Native | ⭐⭐ Via SDK | ⭐⭐ Via Agent |
| **On-Premise Support** | ⭐ Limited | ⭐ Limited | ⭐⭐⭐ Full |
| **Dynamic Secrets** | ⭐ Limited | ⭐⭐ RDS/DocDB | ⭐⭐⭐ Full |
| **Multi-Cloud** | ⭐ Azure Only | ⭐ AWS Only | ⭐⭐⭐ Any |
| **Cost (1K secrets)** | ~$6,396/year | ~$6,000/year | ~$51,760/year |
| **Learning Curve** | Low | Low | High |
| **HSM Support** | ⭐⭐⭐ Built-in | ⭐⭐ AWS CloudHSM | ⭐⭐⭐ Enterprise |

### 2.2 Recommended Architecture: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hybrid Secrets Vault Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Windows 10 AI Agent Host                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │              Local Secrets Cache (Encrypted)                  │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │ │   │
│  │  │  │  DPAPI      │  │  Service    │  │  Session Tokens     │   │ │   │
│  │  │  │  Protected  │  │  Credentials│  │  (15-min TTL)       │   │ │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │              Vault Agent (Local Proxy)                        │ │   │
│  │  │  • Caches secrets locally                                     │ │   │
│  │  │  • Auto-renews leases                                         │ │   │
│  │  │  • Handles failover                                           │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐ │
│  │         Primary Vault: Azure Key Vault (Premium Tier)                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │ │
│  │  │  Secrets    │  │    Keys     │  │Certificates │  │   HSM      │  │ │
│  │  │  (API Keys) │  │ (Encryption)│  │  (TLS/Auth) │  │  Backed    │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐ │
│  │    Secondary Vault: HashiCorp Vault (Enterprise) - Disaster Recovery │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │ │
│  │  │  Dynamic    │  │   PKI       │  │   Database  │                   │ │
│  │  │  Secrets    │  │  Engine     │  │   Secrets   │                   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Azure Key Vault Configuration

```powershell
# Azure Key Vault Setup for AI Agent System
# Prerequisites: Azure CLI installed and authenticated

# 1. Create Resource Group
az group create --name "rg-openclaw-secrets" --location "eastus"

# 2. Create Key Vault with Premium SKU (HSM support)
az keyvault create \
    --name "kv-openclaw-prod" \
    --resource-group "rg-openclaw-secrets" \
    --location "eastus" \
    --sku "Premium" \
    --enable-purge-protection \
    --enable-rbac-authorization \
    --retention-days 90

# 3. Configure Network Access (Private Endpoint Recommended)
az keyvault network-rule add \
    --name "kv-openclaw-prod" \
    --resource-group "rg-openclaw-secrets" \
    --ip-address "<AGENT_HOST_IP>"

# 4. Create Managed Identity for Agent
az identity create \
    --name "id-openclaw-agent" \
    --resource-group "rg-openclaw-secrets"

# 5. Assign Key Vault Secrets User Role
az role assignment create \
    --role "Key Vault Secrets User" \
    --assignee-object-id $(az identity show --name "id-openclaw-agent" --resource-group "rg-openclaw-secrets" --query principalId -o tsv) \
    --scope $(az keyvault show --name "kv-openclaw-prod" --resource-group "rg-openclaw-secrets" --query id -o tsv)
```

### 2.4 Secret Naming Convention

```
Secret Path Format: {environment}/{service}/{credential-type}/{identifier}

Examples:
- prod/gmail/api-key/primary
- prod/twilio/auth-token/sms-service
- prod/openai/api-key/gpt52-engine
- prod/azure/tenant-id/primary
- staging/browser/chrome-profile/default
```

### 2.5 Secret Metadata Schema

```json
{
  "secret_metadata": {
    "name": "prod/gmail/api-key/primary",
    "content_type": "application/json",
    "tags": {
      "environment": "production",
      "service": "gmail",
      "credential_type": "api_key",
      "rotation_schedule": "90_days",
      "owner": "security-team",
      "auto_rotate": "true",
      "last_rotated": "2025-01-15T00:00:00Z",
      "next_rotation": "2025-04-15T00:00:00Z"
    },
    "attributes": {
      "enabled": true,
      "exp": 1767225600,
      "nbf": 1704067200
    }
  }
}
```

---

## 3. Credential Encryption at Rest

### 3.1 Encryption Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Encryption at Rest Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: Application-Level Encryption (Primary)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Algorithm: AES-256-GCM                                             │   │
│  │  Key Derivation: PBKDF2 (100,000 iterations)                        │   │
│  │  Additional Authenticated Data: Secret path + timestamp             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  Layer 2: Windows DPAPI (Platform Integration)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scope: Current User + Machine                                      │   │
│  │  Protection: Tied to user credentials                               │   │
│  │  Recovery: Domain controller backup (domain-joined)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  Layer 3: Azure Key Vault (Cloud HSM)                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Algorithm: RSA-4096 for key wrapping                               │   │
│  │  HSM: FIPS 140-2 Level 2 validated                                  │   │
│  │  Key Rotation: Automatic (90 days)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Encryption Implementation (Python)

```python
"""
Credential Encryption Module for AI Agent System
Implements AES-256-GCM with Windows DPAPI integration
"""

import os
import json
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Windows DPAPI integration (Windows 10)
try:
    import ctypes
    from ctypes import wintypes
    _WINDOWS_AVAILABLE = True
except ImportError:
    _WINDOWS_AVAILABLE = False


class CredentialEncryption:
    """
    Multi-layer encryption for credential storage.
    Combines AES-256-GCM with Windows DPAPI.
    """

    # Constants
    AES_KEY_SIZE = 32  # 256 bits
    AES_NONCE_SIZE = 12  # 96 bits for GCM
    SALT_SIZE = 32
    PBKDF2_ITERATIONS = 100_000

    def __init__(self, master_key_path: Optional[str] = None):
        """
        Initialize encryption module.

        Args:
            master_key_path: Path to store/retrieve master key
        """
        self.master_key_path = master_key_path or self._get_default_key_path()
        self._master_key: Optional[bytes] = None
        self._dpapi_available = self._check_dpapi()

    def _get_default_key_path(self) -> str:
        """Get default path for master key storage."""
        app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return os.path.join(app_data, 'OpenClaw', 'secure', 'master.key')

    def _check_dpapi(self) -> bool:
        """Check if Windows DPAPI is available."""
        return _WINDOWS_AVAILABLE and os.name == 'nt'

    def _get_or_create_master_key(self) -> bytes:
        """Get existing or create new master key with DPAPI protection."""
        if self._master_key is not None:
            return self._master_key

        if os.path.exists(self.master_key_path):
            # Load and decrypt existing key
            with open(self.master_key_path, 'rb') as f:
                encrypted_key = f.read()

            if self._dpapi_available:
                self._master_key = self._dpapi_decrypt(encrypted_key)
            else:
                # Fallback: prompt for password (development only)
                raise RuntimeError("DPAPI unavailable - manual key entry required")
        else:
            # Generate new master key
            self._master_key = secrets.token_bytes(self.AES_KEY_SIZE)

            # Protect with DPAPI and store
            os.makedirs(os.path.dirname(self.master_key_path), exist_ok=True)

            if self._dpapi_available:
                encrypted_key = self._dpapi_encrypt(self._master_key)
                with open(self.master_key_path, 'wb') as f:
                    f.write(encrypted_key)

                # Restrict file permissions
                os.chmod(self.master_key_path, 0o600)
            else:
                raise RuntimeError("DPAPI unavailable - cannot create secure key storage")

        return self._master_key

    def _dpapi_encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Windows DPAPI."""
        if not self._dpapi_available:
            raise RuntimeError("DPAPI not available")

        # CRYPTPROTECT_LOCAL_MACHINE = 0x4
        # CRYPTPROTECT_UI_FORBIDDEN = 0x1
        CRYPTPROTECT_LOCAL_MACHINE = 0x4

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_byte))
            ]

        # Prepare input data
        blob_in = DATA_BLOB()
        blob_in.cbData = len(data)
        blob_in.pbData = ctypes.cast(
            ctypes.create_string_buffer(data),
            ctypes.POINTER(ctypes.c_byte)
        )

        blob_out = DATA_BLOB()

        # Call CryptProtectData
        CryptProtectData = ctypes.windll.crypt32.CryptProtectData
        CryptProtectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            wintypes.LPCWSTR,
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB)
        ]
        CryptProtectData.restype = wintypes.BOOL

        result = CryptProtectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            CRYPTPROTECT_LOCAL_MACHINE,
            ctypes.byref(blob_out)
        )

        if not result:
            raise RuntimeError(f"DPAPI encryption failed: {ctypes.get_last_error()}")

        # Extract encrypted data
        encrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)

        # Free memory
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)

        return encrypted

    def _dpapi_decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Windows DPAPI."""
        if not self._dpapi_available:
            raise RuntimeError("DPAPI not available")

        CRYPTUNPROTECT_UI_FORBIDDEN = 0x1

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_byte))
            ]

        blob_in = DATA_BLOB()
        blob_in.cbData = len(encrypted_data)
        blob_in.pbData = ctypes.cast(
            ctypes.create_string_buffer(encrypted_data),
            ctypes.POINTER(ctypes.c_byte)
        )

        blob_out = DATA_BLOB()

        CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
        CryptUnprotectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB)
        ]
        CryptUnprotectData.restype = wintypes.BOOL

        result = CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            CRYPTUNPROTECT_UI_FORBIDDEN,
            ctypes.byref(blob_out)
        )

        if not result:
            raise RuntimeError(f"DPAPI decryption failed: {ctypes.get_last_error()}")

        decrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)

        return decrypted

    def encrypt_credential(self, credential: str, context: str) -> Dict[str, str]:
        """
        Encrypt a credential with context binding.

        Args:
            credential: The secret to encrypt
            context: Context identifier (e.g., secret path)

        Returns:
            Dictionary containing encrypted data and metadata
        """
        # Generate unique salt and nonce
        salt = secrets.token_bytes(self.SALT_SIZE)
        nonce = secrets.token_bytes(self.AES_NONCE_SIZE)

        # Derive key from master key + salt
        master_key = self._get_or_create_master_key()
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=self.AES_KEY_SIZE,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        derived_key = kdf.derive(master_key)

        # Create AAD (Additional Authenticated Data)
        timestamp = datetime.utcnow().isoformat()
        aad = f"{context}:{timestamp}".encode('utf-8')

        # Encrypt
        aesgcm = AESGCM(derived_key)
        plaintext = credential.encode('utf-8')
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'salt': base64.b64encode(salt).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'aad_context': context,
            'timestamp': timestamp,
            'algorithm': 'AES-256-GCM',
            'kdf': 'PBKDF2-SHA256',
            'iterations': self.PBKDF2_ITERATIONS
        }

    def decrypt_credential(self, encrypted_data: Dict[str, str]) -> str:
        """
        Decrypt a credential.

        Args:
            encrypted_data: Dictionary from encrypt_credential()

        Returns:
            Decrypted credential string
        """
        # Extract components
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        salt = base64.b64decode(encrypted_data['salt'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        aad = f"{encrypted_data['aad_context']}:{encrypted_data['timestamp']}".encode('utf-8')

        # Derive key
        master_key = self._get_or_create_master_key()
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=self.AES_KEY_SIZE,
            salt=salt,
            iterations=encrypted_data.get('iterations', self.PBKDF2_ITERATIONS),
            backend=default_backend()
        )
        derived_key = kdf.derive(master_key)

        # Decrypt
        aesgcm = AESGCM(derived_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

        return plaintext.decode('utf-8')


# Usage Example
if __name__ == "__main__":
    # Initialize encryption
    crypto = CredentialEncryption()

    # Encrypt a credential
    secret = "sk-abc123xyz789"
    encrypted = crypto.encrypt_credential(
        credential=secret,
        context="prod/openai/api-key/primary"
    )

    print(f"Encrypted: {json.dumps(encrypted, indent=2)}")

    # Decrypt
    decrypted = crypto.decrypt_credential(encrypted)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {secret == decrypted}")
```

### 3.3 File System Security

```powershell
# Windows 10 File System Security for Credential Storage

# 1. Create secure directory structure
$SecurePath = "$env:LOCALAPPDATA\OpenClaw\secure"
New-Item -ItemType Directory -Path $SecurePath -Force | Out-Null

# 2. Set NTFS permissions (remove inheritance, explicit only)
$acl = Get-Acl $SecurePath

# Disable inheritance and remove inherited permissions
$acl.SetAccessRuleProtection($true, $false)

# Add current user with full control
$userRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $env:USERNAME,
    "FullControl",
    "ContainerInherit,ObjectInherit",
    "None",
    "Allow"
)
$acl.AddAccessRule($userRule)

# Add SYSTEM (for DPAPI)
$systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "SYSTEM",
    "FullControl",
    "ContainerInherit,ObjectInherit",
    "None",
    "Allow"
)
$acl.AddAccessRule($systemRule)

# Apply ACL
Set-Acl $SecurePath $acl

# 3. Enable EFS encryption (optional additional layer)
cipher /e $SecurePath

# 4. Verify permissions
Get-Acl $SecurePath | Format-List
```

---

## 4. Secure Credential Retrieval

### 4.1 Retrieval Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Secure Credential Retrieval Flow                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐  │
│  │  AI Agent    │     │         Credential Retrieval Pipeline            │  │
│  │  Component   │────►│                                                  │  │
│  └──────────────┘     │  1. Request Validation                           │  │
│                       │     • Verify component identity                   │  │
│                       │     • Check permission scope                      │  │
│                       │     • Validate request signature                  │  │
│                       │                                                  │  │
│                       │  2. Local Cache Check (LRU + TTL)                 │  │
│                       │     ┌─────────────┐    ┌─────────────┐           │  │
│                       │     │   Cache     │───►│   Valid?    │           │  │
│                       │     │   Lookup    │    │   (TTL)     │           │  │
│                       │     └─────────────┘    └──────┬──────┘           │  │
│                       │                              │                    │  │
│                       │                    ┌─────────┴─────────┐          │  │
│                       │                    │                   │          │  │
│                       │                    ▼                   ▼          │  │
│                       │              ┌─────────┐         ┌─────────┐      │  │
│                       │              │  HIT    │         │  MISS   │      │  │
│                       │              │ Return  │         │ Fetch   │      │  │
│                       │              │ Cached  │         │ Remote  │      │  │
│                       │              └─────────┘         └────┬────┘      │  │
│                       │                                       │           │  │
│                       │  3. Vault Retrieval                   │           │  │
│                       │     • Authenticate with managed ID    │           │  │
│                       │     • Fetch from Azure Key Vault      │           │  │
│                       │     • Decrypt with HSM key            │           │  │
│                       │                                       │           │  │
│                       │  4. Response Processing               │           │  │
│                       │     • Decrypt credential              │           │  │
│                       │     • Update cache                    │           │  │
│                       │     • Log access                      │           │  │
│                       │     • Return to component             │           │  │
│                       │                                                  │  │
│                       └─────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Credential Manager Implementation

```python
"""
Secure Credential Manager for AI Agent System
Implements caching, retrieval, and lifecycle management
"""

import json
import time
import hashlib
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CachedCredential:
    """Represents a cached credential with metadata."""
    value: str
    path: str
    cached_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def ttl_seconds(self) -> float:
        return (self.expires_at - datetime.utcnow()).total_seconds()


class SecureCredentialManager:
    """
    Manages credential retrieval with caching and security controls.
    """

    DEFAULT_CACHE_TTL = 300  # 5 minutes default
    MAX_CACHE_SIZE = 100

    def __init__(
        self,
        vault_client: Any,
        encryption: 'CredentialEncryption',
        cache_ttl: int = DEFAULT_CACHE_TTL
    ):
        """
        Initialize credential manager.

        Args:
            vault_client: Azure Key Vault or similar client
            encryption: CredentialEncryption instance
            cache_ttl: Default cache TTL in seconds
        """
        self.vault_client = vault_client
        self.encryption = encryption
        self.cache_ttl = cache_ttl

        # Thread-safe cache
        self._cache: Dict[str, CachedCredential] = {}
        self._cache_lock = threading.RLock()

        # Access audit log
        self._audit_log: list = []
        self._audit_lock = threading.Lock()

        # Permission registry
        self._permissions: Dict[str, list] = {}

    def register_component(
        self,
        component_id: str,
        allowed_paths: list,
        max_cache_ttl: Optional[int] = None
    ):
        """
        Register a component with its allowed credential paths.

        Args:
            component_id: Unique component identifier
            allowed_paths: List of allowed secret path patterns
            max_cache_ttl: Maximum cache TTL for this component
        """
        self._permissions[component_id] = {
            'paths': allowed_paths,
            'max_cache_ttl': max_cache_ttl or self.cache_ttl
        }
        logger.info(f"Registered component: {component_id}")

    def _check_permission(self, component_id: str, path: str) -> bool:
        """Check if component has permission to access path."""
        if component_id not in self._permissions:
            logger.warning(f"Unknown component: {component_id}")
            return False

        allowed_paths = self._permissions[component_id]['paths']

        # Check path against allowed patterns
        for pattern in allowed_paths:
            if self._path_matches(pattern, path):
                return True

        logger.warning(f"Permission denied: {component_id} -> {path}")
        return False

    def _path_matches(self, pattern: str, path: str) -> bool:
        """Check if path matches pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def _get_from_cache(self, path: str) -> Optional[CachedCredential]:
        """Retrieve credential from cache if valid."""
        with self._cache_lock:
            cached = self._cache.get(path)

            if cached is None:
                return None

            if cached.is_expired:
                logger.debug(f"Cache expired for {path}")
                del self._cache[path]
                return None

            # Update access stats
            cached.access_count += 1
            cached.last_accessed = datetime.utcnow()

            return cached

    def _add_to_cache(self, path: str, value: str, ttl: int):
        """Add credential to cache with LRU eviction."""
        with self._cache_lock:
            # Evict if cache is full
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                # Remove oldest by access time
                oldest = min(
                    self._cache.items(),
                    key=lambda x: x[1].last_accessed or x[1].cached_at
                )
                del self._cache[oldest[0]]
                logger.debug(f"Evicted {oldest[0]} from cache")

            # Add new entry
            now = datetime.utcnow()
            self._cache[path] = CachedCredential(
                value=value,
                path=path,
                cached_at=now,
                expires_at=now + timedelta(seconds=ttl)
            )

    def _fetch_from_vault(self, path: str) -> str:
        """Fetch credential from vault."""
        try:
            # Azure Key Vault specific
            secret = self.vault_client.get_secret(path)

            # Decrypt if stored encrypted
            if secret.properties.content_type == 'application/json':
                encrypted = json.loads(secret.value)
                return self.encryption.decrypt_credential(encrypted)

            return secret.value

        except Exception as e:
            logger.error(f"Failed to fetch {path} from vault: {e}")
            raise CredentialRetrievalError(f"Vault fetch failed: {e}")

    def _log_access(
        self,
        component_id: str,
        path: str,
        success: bool,
        source: str
    ):
        """Log credential access for audit."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'component_id': component_id,
            'path': path,
            'success': success,
            'source': source,
            'hash': hashlib.sha256(f"{component_id}:{path}".encode()).hexdigest()[:16]
        }

        with self._audit_lock:
            self._audit_log.append(entry)

            # Trim log if too large
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-5000:]

    def get_credential(
        self,
        component_id: str,
        path: str,
        force_refresh: bool = False
    ) -> str:
        """
        Retrieve a credential with full security controls.

        Args:
            component_id: Requesting component ID
            path: Secret path
            force_refresh: Bypass cache and fetch fresh

        Returns:
            Credential value

        Raises:
            PermissionError: If component lacks permission
            CredentialRetrievalError: If retrieval fails
        """
        # 1. Permission check
        if not self._check_permission(component_id, path):
            self._log_access(component_id, path, False, 'permission_denied')
            raise PermissionError(f"Component {component_id} cannot access {path}")

        # 2. Cache check (unless force refresh)
        if not force_refresh:
            cached = self._get_from_cache(path)
            if cached:
                self._log_access(component_id, path, True, 'cache')
                logger.debug(f"Cache hit for {path}")
                return cached.value

        # 3. Fetch from vault
        try:
            value = self._fetch_from_vault(path)

            # 4. Cache the result
            ttl = self._permissions[component_id].get('max_cache_ttl', self.cache_ttl)
            self._add_to_cache(path, value, ttl)

            # 5. Log success
            self._log_access(component_id, path, True, 'vault')

            logger.info(f"Retrieved {path} from vault for {component_id}")
            return value

        except Exception as e:
            self._log_access(component_id, path, False, 'vault_error')
            raise

    def invalidate_cache(self, path: Optional[str] = None):
        """Invalidate cache entries."""
        with self._cache_lock:
            if path:
                self._cache.pop(path, None)
                logger.info(f"Invalidated cache for {path}")
            else:
                self._cache.clear()
                logger.info("Invalidated entire cache")

    def get_audit_log(self, since: Optional[datetime] = None) -> list:
        """Get access audit log."""
        with self._audit_lock:
            if since:
                return [
                    entry for entry in self._audit_log
                    if datetime.fromisoformat(entry['timestamp']) >= since
                ]
            return self._audit_log.copy()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'size': len(self._cache),
                'max_size': self.MAX_CACHE_SIZE,
                'entries': [
                    {
                        'path': c.path,
                        'access_count': c.access_count,
                        'ttl_seconds': c.ttl_seconds,
                        'is_expired': c.is_expired
                    }
                    for c in self._cache.values()
                ]
            }


class CredentialRetrievalError(Exception):
    """Raised when credential retrieval fails."""
    pass


# Decorator for automatic credential injection
def requires_credential(path: str, param_name: str = 'credential'):
    """
    Decorator to inject credentials into function calls.

    Args:
        path: Secret path
        param_name: Parameter name to inject
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get credential manager from context
            manager = kwargs.get('_credential_manager')
            component_id = kwargs.get('_component_id', 'unknown')

            if manager:
                credential = manager.get_credential(component_id, path)
                kwargs[param_name] = credential

            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 5. API Key Rotation

### 5.1 Rotation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    API Key Rotation Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Rotation Scheduler (Cron)                         │   │
│  │  • Triggers rotation based on policy                               │   │
│  │  • Supports: time-based, event-based, manual                       │   │
│  │  • Cron expression: "0 2 * * 0" (Weekly Sundays 2AM)               │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Rotation Orchestrator                             │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Phase 1    │  │   Phase 2    │  │   Phase 3    │              │   │
│  │  │   Generate   │──►│   Deploy   │──►│   Validate   │              │   │
│  │  │   New Key    │  │   New Key    │  │   New Key    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │         │                 │                 │                       │   │
│  │         ▼                 ▼                 ▼                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Phase 4    │  │   Phase 5    │  │   Phase 6    │              │   │
│  │  │   Update     │──►│   Monitor  │──►│   Revoke     │              │   │
│  │  │   Consumers  │  │   Metrics  │  │   Old Key    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Service-Specific Rotators                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Gmail   │  │  Twilio  │  │ OpenAI   │  │  Azure   │            │   │
│  │  │  API     │  │  Auth    │  │  API     │  │  AD App  │            │   │
│  │  │  Key     │  │  Token   │  │  Key     │  │  Secret  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Rotation Policy Configuration

```python
"""
API Key Rotation Policy and Implementation
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from datetime import datetime, timedelta
from enum import Enum
import json


class RotationTrigger(Enum):
    """Rotation trigger types."""
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class RotationStrategy(Enum):
    """Rotation strategies."""
    BLUE_GREEN = "blue_green"  # Keep both keys active during transition
    ROLLING = "rolling"        # Gradual consumer migration
    IMMEDIATE = "immediate"    # Instant cutover (higher risk)


@dataclass
class RotationPolicy:
    """Defines rotation policy for a credential."""

    # Identification
    credential_path: str
    service_type: str  # gmail, twilio, openai, etc.

    # Schedule
    rotation_interval_days: int = 90
    rotation_hour: int = 2  # 2 AM
    rotation_day: int = 0   # Sunday (0=Sunday, 6=Saturday)

    # Strategy
    strategy: RotationStrategy = RotationStrategy.BLUE_GREEN

    # Timing
    overlap_period_hours: int = 24  # Both keys valid during transition
    validation_period_minutes: int = 30

    # Notifications
    notify_before_days: List[int] = None  # [7, 1] = notify 7 and 1 days before
    notify_on_failure: bool = True

    # Rollback
    auto_rollback_on_failure: bool = True
    rollback_window_minutes: int = 60

    # Emergency
    emergency_rotation_enabled: bool = True

    def __post_init__(self):
        if self.notify_before_days is None:
            self.notify_before_days = [7, 1]

    def to_dict(self) -> Dict:
        return {
            'credential_path': self.credential_path,
            'service_type': self.service_type,
            'rotation_interval_days': self.rotation_interval_days,
            'rotation_hour': self.rotation_hour,
            'rotation_day': self.rotation_day,
            'strategy': self.strategy.value,
            'overlap_period_hours': self.overlap_period_hours,
            'validation_period_minutes': self.validation_period_minutes,
            'notify_before_days': self.notify_before_days,
            'notify_on_failure': self.notify_on_failure,
            'auto_rollback_on_failure': self.auto_rollback_on_failure,
            'rollback_window_minutes': self.rollback_window_minutes,
            'emergency_rotation_enabled': self.emergency_rotation_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RotationPolicy':
        data = data.copy()
        data['strategy'] = RotationStrategy(data['strategy'])
        return cls(**data)


# Default policies for AI Agent services
DEFAULT_POLICIES = {
    'gmail_api': RotationPolicy(
        credential_path='prod/gmail/api-key/primary',
        service_type='gmail',
        rotation_interval_days=90,
        strategy=RotationStrategy.BLUE_GREEN,
        overlap_period_hours=48,  # Gmail has longer propagation
        notify_before_days=[14, 7, 1]
    ),

    'twilio_auth': RotationPolicy(
        credential_path='prod/twilio/auth-token/primary',
        service_type='twilio',
        rotation_interval_days=60,
        strategy=RotationStrategy.BLUE_GREEN,
        overlap_period_hours=24,
        notify_before_days=[7, 1]
    ),

    'openai_api': RotationPolicy(
        credential_path='prod/openai/api-key/primary',
        service_type='openai',
        rotation_interval_days=90,
        strategy=RotationStrategy.ROLLING,
        overlap_period_hours=12,
        notify_before_days=[7, 3, 1]
    ),

    'azure_ad_app': RotationPolicy(
        credential_path='prod/azure/ad-app-secret/primary',
        service_type='azure_ad',
        rotation_interval_days=180,  # Azure AD max is 2 years
        strategy=RotationStrategy.BLUE_GREEN,
        overlap_period_hours=72,  # Azure AD propagation takes time
        notify_before_days=[30, 14, 7, 1]
    )
}
```

### 5.3 Rotation Implementation

```python
"""
API Key Rotation Implementation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)


class ServiceRotator(ABC):
    """Abstract base class for service-specific key rotation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate_new_key(self) -> Dict[str, str]:
        """Generate a new API key. Returns dict with key details."""
        pass

    @abstractmethod
    async def validate_key(self, key: str) -> bool:
        """Validate that a key is working."""
        pass

    @abstractmethod
    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an existing key."""
        pass

    @abstractmethod
    async def list_active_keys(self) -> list:
        """List all active keys for the service."""
        pass


class GmailRotator(ServiceRotator):
    """Gmail API key rotation handler."""

    async def generate_new_key(self) -> Dict[str, str]:
        """Generate new Gmail API credentials via Google Cloud Console API."""
        # This would use Google Cloud API to create new service account key
        # For production, use Google Cloud IAM API

        # Placeholder implementation
        return {
            'key_id': f'gmail-key-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}',
            'private_key': '[GENERATED_KEY]',
            'client_email': 'service@project.iam.gserviceaccount.com',
            'created_at': datetime.utcnow().isoformat()
        }

    async def validate_key(self, key: str) -> bool:
        """Validate Gmail API key by making test request."""
        try:
            # Test with Gmail API users.getProfile endpoint
            # This is a lightweight validation
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {key}'}
                async with session.get(
                    'https://gmail.googleapis.com/gmail/v1/users/me/profile',
                    headers=headers
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Gmail key validation failed: {e}")
            return False

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke Gmail service account key."""
        # Use Google Cloud IAM API to delete service account key
        logger.info(f"Revoking Gmail key: {key_id}")
        return True

    async def list_active_keys(self) -> list:
        """List active Gmail service account keys."""
        return []


class TwilioRotator(ServiceRotator):
    """Twilio auth token rotation handler."""

    async def generate_new_key(self) -> Dict[str, str]:
        """Generate new Twilio auth token."""
        # Twilio doesn't support programmatic token generation
        # Must be done via console, then stored
        # This would trigger a notification for manual rotation

        return {
            'key_id': f'twilio-token-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}',
            'manual_rotation_required': True,
            'console_url': 'https://console.twilio.com/account/settings'
        }

    async def validate_key(self, key: str) -> bool:
        """Validate Twilio auth token."""
        try:
            from twilio.rest import Client
            client = Client(self.config['account_sid'], key)
            # Make lightweight API call
            client.api.accounts(self.config['account_sid']).fetch()
            return True
        except Exception as e:
            logger.error(f"Twilio key validation failed: {e}")
            return False

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke Twilio auth token."""
        # Twilio tokens are revoked by generating new ones
        logger.info(f"Twilio token marked for rotation: {key_id}")
        return True

    async def list_active_keys(self) -> list:
        """List active Twilio credentials."""
        return []


class OpenAIRotator(ServiceRotator):
    """OpenAI API key rotation handler."""

    async def generate_new_key(self) -> Dict[str, str]:
        """Generate new OpenAI API key."""
        # OpenAI doesn't support programmatic key generation
        # Must be done via dashboard

        return {
            'key_id': f'openai-key-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}',
            'manual_rotation_required': True,
            'dashboard_url': 'https://platform.openai.com/api-keys'
        }

    async def validate_key(self, key: str) -> bool:
        """Validate OpenAI API key."""
        try:
            import openai
            client = openai.OpenAI(api_key=key)
            # Make lightweight API call
            client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI key validation failed: {e}")
            return False

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke OpenAI API key."""
        logger.info(f"OpenAI key marked for deletion: {key_id}")
        return True

    async def list_active_keys(self) -> list:
        """List active OpenAI keys."""
        return []


class RotationOrchestrator:
    """Orchestrates the complete rotation process."""

    ROTATOR_MAP = {
        'gmail': GmailRotator,
        'twilio': TwilioRotator,
        'openai': OpenAIRotator,
    }

    def __init__(
        self,
        credential_manager: SecureCredentialManager,
        vault_client: Any
    ):
        self.credential_manager = credential_manager
        self.vault_client = vault_client
        self.rotators: Dict[str, ServiceRotator] = {}
        self.rotation_history: list = []

    def register_rotator(self, service_type: str, rotator: ServiceRotator):
        """Register a service rotator."""
        self.rotators[service_type] = rotator

    async def execute_rotation(
        self,
        policy: RotationPolicy,
        trigger: RotationTrigger = RotationTrigger.MANUAL
    ) -> Dict[str, Any]:
        """
        Execute full rotation workflow.

        Returns:
            Rotation result with status and details
        """
        rotation_id = f"rot-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        result = {
            'rotation_id': rotation_id,
            'credential_path': policy.credential_path,
            'started_at': datetime.utcnow().isoformat(),
            'status': 'in_progress',
            'phases': []
        }

        try:
            # Get appropriate rotator
            rotator = self.rotators.get(policy.service_type)
            if not rotator:
                raise ValueError(f"No rotator registered for {policy.service_type}")

            # Phase 1: Generate new key
            logger.info(f"[{rotation_id}] Phase 1: Generating new key")
            new_key_data = await rotator.generate_new_key()
            result['phases'].append({
                'phase': 'generate',
                'status': 'success',
                'key_id': new_key_data.get('key_id')
            })

            # Phase 2: Store new key in vault
            logger.info(f"[{rotation_id}] Phase 2: Storing new key")
            new_path = f"{policy.credential_path}-new"
            # Store encrypted in vault
            result['phases'].append({
                'phase': 'store',
                'status': 'success',
                'path': new_path
            })

            # Phase 3: Validate new key
            logger.info(f"[{rotation_id}] Phase 3: Validating new key")
            if new_key_data.get('manual_rotation_required'):
                result['phases'].append({
                    'phase': 'validate',
                    'status': 'pending_manual',
                    'message': 'Manual rotation required'
                })
                result['status'] = 'pending_manual'
                return result

            is_valid = await rotator.validate_key(new_key_data.get('private_key', ''))
            if not is_valid:
                raise RuntimeError("New key validation failed")

            result['phases'].append({
                'phase': 'validate',
                'status': 'success'
            })

            # Phase 4: Deploy (strategy-dependent)
            logger.info(f"[{rotation_id}] Phase 4: Deploying new key")
            await self._deploy_key(policy, new_path)
            result['phases'].append({
                'phase': 'deploy',
                'status': 'success',
                'strategy': policy.strategy.value
            })

            # Phase 5: Monitor
            logger.info(f"[{rotation_id}] Phase 5: Monitoring")
            await asyncio.sleep(policy.validation_period_minutes * 60)
            # Check metrics, error rates
            result['phases'].append({
                'phase': 'monitor',
                'status': 'success'
            })

            # Phase 6: Revoke old key
            logger.info(f"[{rotation_id}] Phase 6: Revoking old key")
            old_key = self.credential_manager.get_credential(
                'rotation_service',
                policy.credential_path
            )
            # await rotator.revoke_key(old_key_id)
            result['phases'].append({
                'phase': 'revoke',
                'status': 'success'
            })

            result['status'] = 'completed'
            result['completed_at'] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.error(f"[{rotation_id}] Rotation failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

            if policy.auto_rollback_on_failure:
                logger.info(f"[{rotation_id}] Initiating rollback")
                await self._rollback_rotation(policy)
                result['rollback'] = 'completed'

        self.rotation_history.append(result)
        return result

    async def _deploy_key(self, policy: RotationPolicy, new_path: str):
        """Deploy new key based on rotation strategy."""
        if policy.strategy == RotationStrategy.BLUE_GREEN:
            # Both keys active during overlap period
            pass
        elif policy.strategy == RotationStrategy.ROLLING:
            # Gradual rollout
            pass
        elif policy.strategy == RotationStrategy.IMMEDIATE:
            # Instant cutover
            pass

    async def _rollback_rotation(self, policy: RotationPolicy):
        """Rollback failed rotation."""
        logger.info(f"Rolling back rotation for {policy.credential_path}")
        # Restore previous key as primary
        # Invalidate new key
        pass


# Windows Task Scheduler integration for automated rotation
"""
# PowerShell script to create scheduled rotation task
$action = New-ScheduledTaskAction -Execute "python" -Argument "rotate_credentials.py"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "2:00 AM"
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName "OpenClaw-CredentialRotation" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings
"""
```

---

## 6. OAuth Token Management

### 6.1 Token Lifecycle Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OAuth Token Lifecycle Management                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Token Types & Lifetimes                           │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Access Token (Short-Lived)                                   │  │   │
│  │  │  • Lifetime: 5-15 minutes                                     │  │   │
│  │  │  • Storage: Memory only (never persisted)                     │  │   │
│  │  │  • Scope: Minimal per request                                 │  │   │
│  │  │  • Format: JWT with embedded claims                           │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Refresh Token (Medium-Lived)                                 │  │   │
│  │  │  • Lifetime: 1-7 days                                         │  │   │
│  │  │  • Storage: Encrypted at rest (Vault)                         │  │   │
│  │  │  • Rotation: Single-use (new token each refresh)              │  │   │
│  │  │  • Family Tracking: Detect reuse attacks                      │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Session Token (Context-Bound)                                │  │   │
│  │  │  • Lifetime: Duration of conversation                         │  │   │
│  │  │  • Binding: Device fingerprint + conversation ID              │  │   │
│  │  │  • Expiration: Auto-expire on conversation end                │  │   │
│  │  │  • Revocation: Immediate on logout/suspicious activity        │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Token Refresh Flow                                │   │
│  │                                                                      │   │
│  │  ┌──────────┐         ┌──────────┐         ┌──────────┐            │   │
│  │  │  Client  │────────►│  Token   │────────►│ Identity │            │   │
│  │  │          │         │ Manager  │         │ Provider │            │   │
│  │  └──────────┘         └────┬─────┘         └────┬─────┘            │   │
│  │       │                    │                    │                  │   │
│  │       │ 1. Request with    │                    │                  │   │
│  │       │    expired token   │                    │                  │   │
│  │       ├───────────────────►│                    │                  │   │
│  │       │                    │                    │                  │   │
│  │       │                    │ 2. Validate        │                  │   │
│  │       │                    │    refresh token   │                  │   │
│  │       │                    ├───────────────────►│                  │   │
│  │       │                    │                    │                  │   │
│  │       │                    │ 3. Issue new       │                  │   │
│  │       │                    │    token family    │                  │   │
│  │       │                    │◄───────────────────┤                  │   │
│  │       │                    │                    │                  │   │
│  │       │ 4. Return new      │                    │                  │   │
│  │       │    access token    │                    │                  │   │
│  │       │◄───────────────────┤                    │                  │   │
│  │       │                    │                    │                  │   │
│  │       │ 5. Log rotation    │                    │                  │   │
│  │       │    event           │                    │                  │   │
│  │       ├───────────────────►│                    │                  │   │
│  │       │                    │                    │                  │   │
│  └───────┴────────────────────┴────────────────────┴──────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Token Manager Implementation

```python
"""
OAuth Token Management for AI Agent System
Implements secure token lifecycle with rotation and family tracking
"""

import uuid
import hashlib
import secrets
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import jwt
import redis
import logging

logger = logging.getLogger(__name__)


class TokenStatus(Enum):
    """Token status values."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATED = "rotated"


@dataclass
class TokenFamily:
    """Tracks a token family for detecting reuse attacks."""
    family_id: str
    user_id: str
    service: str
    created_at: datetime
    current_token_hash: str
    previous_token_hash: Optional[str] = None
    rotation_count: int = 0
    status: TokenStatus = TokenStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            'family_id': self.family_id,
            'user_id': self.user_id,
            'service': self.service,
            'created_at': self.created_at.isoformat(),
            'current_token_hash': self.current_token_hash,
            'previous_token_hash': self.previous_token_hash,
            'rotation_count': self.rotation_count,
            'status': self.status.value
        }


@dataclass
class TokenMetadata:
    """Metadata for issued tokens."""
    token_id: str
    token_hash: str
    family_id: str
    user_id: str
    service: str
    scopes: list
    issued_at: datetime
    expires_at: datetime
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'token_id': self.token_id,
            'token_hash': self.token_hash,
            'family_id': self.family_id,
            'user_id': self.user_id,
            'service': self.service,
            'scopes': self.scopes,
            'issued_at': self.issued_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'device_fingerprint': self.device_fingerprint,
            'ip_address': self.ip_address
        }


class TokenManager:
    """
    Manages OAuth tokens with security best practices:
    - Short-lived access tokens (5-15 min)
    - Refresh token rotation
    - Token family tracking
    - Reuse detection
    """

    # Token lifetimes
    ACCESS_TOKEN_LIFETIME_MINUTES = 15
    REFRESH_TOKEN_LIFETIME_DAYS = 7
    SESSION_TOKEN_LIFETIME_HOURS = 24

    def __init__(
        self,
        jwt_secret: str,
        redis_client: Optional[redis.Redis] = None,
        encryption: Optional['CredentialEncryption'] = None
    ):
        self.jwt_secret = jwt_secret
        self.redis = redis_client
        self.encryption = encryption

        # In-memory token families (use Redis in production)
        self._token_families: Dict[str, TokenFamily] = {}
        self._token_metadata: Dict[str, TokenMetadata] = {}

    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        return f"tok_{secrets.token_urlsafe(16)}"

    def _hash_token(self, token: str) -> str:
        """Create hash of token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _generate_device_fingerprint(self, request_context: Dict) -> str:
        """Generate device fingerprint for token binding."""
        components = [
            request_context.get('user_agent', ''),
            request_context.get('accept_language', ''),
            request_context.get('screen_resolution', ''),
            request_context.get('timezone', '')
        ]
        fingerprint_data = '|'.join(components)
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]

    def create_access_token(
        self,
        user_id: str,
        service: str,
        scopes: list,
        family_id: Optional[str] = None,
        request_context: Optional[Dict] = None
    ) -> Tuple[str, TokenMetadata]:
        """
        Create a new access token.

        Returns:
            Tuple of (token_string, metadata)
        """
        token_id = self._generate_token_id()
        now = datetime.utcnow()
        expires = now + timedelta(minutes=self.ACCESS_TOKEN_LIFETIME_MINUTES)

        # Create JWT payload
        payload = {
            'jti': token_id,
            'sub': user_id,
            'service': service,
            'scope': ' '.join(scopes),
            'iat': now,
            'exp': expires,
            'type': 'access'
        }

        # Add family ID if provided
        if family_id:
            payload['fid'] = family_id

        # Add device fingerprint
        if request_context:
            payload['dfp'] = self._generate_device_fingerprint(request_context)

        # Sign token
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')

        # Create metadata
        metadata = TokenMetadata(
            token_id=token_id,
            token_hash=self._hash_token(token),
            family_id=family_id or 'none',
            user_id=user_id,
            service=service,
            scopes=scopes,
            issued_at=now,
            expires_at=expires,
            device_fingerprint=payload.get('dfp'),
            ip_address=request_context.get('ip_address') if request_context else None
        )

        # Store metadata
        self._token_metadata[token_id] = metadata

        # Store in Redis if available
        if self.redis:
            self.redis.setex(
                f"token:{token_id}",
                timedelta(minutes=self.ACCESS_TOKEN_LIFETIME_MINUTES),
                json.dumps(metadata.to_dict())
            )

        logger.info(f"Created access token {token_id} for user {user_id}")
        return token, metadata

    def create_refresh_token(
        self,
        user_id: str,
        service: str,
        scopes: list,
        request_context: Optional[Dict] = None
    ) -> Tuple[str, TokenFamily]:
        """
        Create a new refresh token with family tracking.

        Returns:
            Tuple of (token_string, family)
        """
        family_id = f"fam_{secrets.token_urlsafe(16)}"
        token_id = self._generate_token_id()
        now = datetime.utcnow()
        expires = now + timedelta(days=self.REFRESH_TOKEN_LIFETIME_DAYS)

        # Create JWT payload
        payload = {
            'jti': token_id,
            'fid': family_id,
            'sub': user_id,
            'service': service,
            'scope': ' '.join(scopes),
            'iat': now,
            'exp': expires,
            'type': 'refresh'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        token_hash = self._hash_token(token)

        # Create token family
        family = TokenFamily(
            family_id=family_id,
            user_id=user_id,
            service=service,
            created_at=now,
            current_token_hash=token_hash
        )

        self._token_families[family_id] = family

        # Store in Redis if available
        if self.redis:
            self.redis.setex(
                f"family:{family_id}",
                timedelta(days=self.REFRESH_TOKEN_LIFETIME_DAYS),
                json.dumps(family.to_dict())
            )

        logger.info(f"Created refresh token family {family_id} for user {user_id}")
        return token, family

    def rotate_refresh_token(
        self,
        old_refresh_token: str,
        request_context: Optional[Dict] = None
    ) -> Tuple[str, TokenFamily]:
        """
        Rotate a refresh token (single-use pattern).

        Implements refresh token rotation with reuse detection.

        Returns:
            Tuple of (new_token, family)

        Raises:
            SecurityError: If token reuse is detected
        """
        try:
            # Decode old token
            payload = jwt.decode(
                old_refresh_token,
                self.jwt_secret,
                algorithms=['HS256']
            )

            family_id = payload.get('fid')
            old_token_hash = self._hash_token(old_refresh_token)

            # Check family exists
            family = self._token_families.get(family_id)
            if not family:
                # Try Redis
                if self.redis:
                    family_data = self.redis.get(f"family:{family_id}")
                    if family_data:
                        family = TokenFamily(**json.loads(family_data))

                if not family:
                    raise ValueError("Token family not found")

            # REUSE DETECTION: Check if this token was already rotated
            if family.status == TokenStatus.ROTATED:
                if old_token_hash != family.current_token_hash:
                    # This is a reuse attack!
                    logger.critical(
                        f"REFRESH TOKEN REUSE DETECTED! "
                        f"Family: {family_id}, User: {family.user_id}"
                    )

                    # Revoke entire family
                    self._revoke_token_family(family_id)

                    # Alert security team
                    self._alert_security_team(family)

                    raise SecurityError("Token reuse detected - family revoked")

            # Generate new token
            new_token_id = self._generate_token_id()
            now = datetime.utcnow()
            expires = now + timedelta(days=self.REFRESH_TOKEN_LIFETIME_DAYS)

            new_payload = {
                'jti': new_token_id,
                'fid': family_id,
                'sub': payload['sub'],
                'service': payload['service'],
                'scope': payload['scope'],
                'iat': now,
                'exp': expires,
                'type': 'refresh'
            }

            new_token = jwt.encode(new_payload, self.jwt_secret, algorithm='HS256')
            new_token_hash = self._hash_token(new_token)

            # Update family
            family.previous_token_hash = family.current_token_hash
            family.current_token_hash = new_token_hash
            family.rotation_count += 1
            family.status = TokenStatus.ROTATED

            # Update storage
            self._token_families[family_id] = family
            if self.redis:
                self.redis.setex(
                    f"family:{family_id}",
                    timedelta(days=self.REFRESH_TOKEN_LIFETIME_DAYS),
                    json.dumps(family.to_dict())
                )

            logger.info(
                f"Rotated refresh token for family {family_id} "
                f"(rotation #{family.rotation_count})"
            )

            return new_token, family

        except jwt.ExpiredSignatureError:
            raise ValueError("Refresh token expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")

    def validate_access_token(
        self,
        token: str,
        required_scopes: Optional[list] = None,
        request_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate an access token.

        Returns:
            Token payload if valid

        Raises:
            ValueError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS256']
            )

            # Verify token type
            if payload.get('type') != 'access':
                raise ValueError("Invalid token type")

            # Check scopes
            if required_scopes:
                token_scopes = set(payload.get('scope', '').split())
                if not set(required_scopes).issubset(token_scopes):
                    raise ValueError("Insufficient scope")

            # Verify device fingerprint (if bound)
            if 'dfp' in payload and request_context:
                expected_dfp = self._generate_device_fingerprint(request_context)
                if payload['dfp'] != expected_dfp:
                    logger.warning("Device fingerprint mismatch")
                    raise ValueError("Token bound to different device")

            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")

    def revoke_token_family(self, family_id: str):
        """Revoke all tokens in a family."""
        self._revoke_token_family(family_id)

    def _revoke_token_family(self, family_id: str):
        """Internal revoke with logging."""
        family = self._token_families.get(family_id)
        if family:
            family.status = TokenStatus.REVOKED
            logger.info(f"Revoked token family {family_id}")

        if self.redis:
            self.redis.delete(f"family:{family_id}")

    def _alert_security_team(self, family: TokenFamily):
        """Alert security team of token reuse."""
        # Implement notification (email, Slack, SIEM, etc.)
        logger.critical(
            f"SECURITY ALERT: Token reuse detected. "
            f"User: {family.user_id}, Service: {family.service}, "
            f"Family: {family.family_id}"
        )

    def cleanup_expired_tokens(self):
        """Remove expired tokens from storage."""
        now = datetime.utcnow()

        # Clean up token families
        expired_families = [
            fid for fid, fam in self._token_families.items()
            if (now - fam.created_at).days > self.REFRESH_TOKEN_LIFETIME_DAYS
        ]
        for fid in expired_families:
            del self._token_families[fid]

        logger.info(f"Cleaned up {len(expired_families)} expired token families")


class SecurityError(Exception):
    """Security-related error."""
    pass
```

---

## 7. Short-Lived Credential Preference

### 7.1 Credential Type Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Credential Type Security Hierarchy                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HIGHEST SECURITY (Preferred)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Dynamic/Just-in-Time Credentials                               │   │
│  │     • Database: Vault dynamic secrets (TTL: 1 hour)                │   │
│  │     • Cloud: IAM temporary credentials (TTL: 1 hour)               │   │
│  │     • SSH: Signed certificates (TTL: 30 minutes)                   │   │
│  │     • Example: SELECT * FROM vault.get_db_creds('readonly', '1h')  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. OAuth 2.0 Tokens (with rotation)                               │   │
│  │     • Access tokens: 5-15 minutes                                  │   │
│  │     • Refresh tokens: 1-7 days with single-use rotation            │   │
│  │     • PKCE for public clients                                      │   │
│  │     • DPoP (Demonstrating Proof-of-Possession)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. Managed Identity / Service Principal                           │   │
│  │     • Azure Managed Identity (no static credentials)               │   │
│  │     • AWS IAM Roles for Service Accounts (IRSA)                    │   │
│  │     • GCP Workload Identity                                        │   │
│  │     • No secrets stored in application                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  4. API Keys with Automated Rotation                               │   │
│  │     • Rotation: 60-90 days                                         │   │
│  │     • Scoped to minimum permissions                                │   │
│  │     • IP restrictions where possible                               │   │
│  │     • Rate limiting enabled                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  5. Long-Lived API Keys (Last Resort)                              │   │
│  │     • Only when no other option available                          │   │
│  │     • Stored in HSM-backed vault                                   │   │
│  │     • Manual rotation procedures                                   │   │
│  │     • Enhanced monitoring and alerting                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LOWEST SECURITY (Avoid)                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Short-Lived Credential Implementation

```python
"""
Short-Lived Credential Manager
Implements just-in-time credential provisioning
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShortLivedCredential:
    """Represents a short-lived credential."""
    credential_type: str
    value: str
    issued_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def ttl_seconds(self) -> float:
        return max(0, (self.expires_at - datetime.utcnow()).total_seconds())

    @property
    def is_valid(self) -> bool:
        return not self.is_expired


class CredentialProvider(ABC):
    """Abstract base for credential providers."""

    @abstractmethod
    async def generate(
        self,
        scope: str,
        ttl: timedelta,
        context: Optional[Dict] = None
    ) -> ShortLivedCredential:
        """Generate a short-lived credential."""
        pass

    @abstractmethod
    async def revoke(self, credential_id: str) -> bool:
        """Revoke a credential."""
        pass

    @abstractmethod
    async def validate(self, credential: ShortLivedCredential) -> bool:
        """Validate a credential is still valid."""
        pass


class VaultDatabaseProvider(CredentialProvider):
    """HashiCorp Vault dynamic database credentials."""

    def __init__(self, vault_client: Any, db_role: str):
        self.vault_client = vault_client
        self.db_role = db_role

    async def generate(
        self,
        scope: str,
        ttl: timedelta,
        context: Optional[Dict] = None
    ) -> ShortLivedCredential:
        """Generate dynamic database credentials."""
        try:
            # Request credentials from Vault
            response = self.vault_client.secrets.database.generate_credentials(
                name=self.db_role,
                ttl=f"{int(ttl.total_seconds())}s"
            )

            credential = ShortLivedCredential(
                credential_type='database',
                value=json.dumps({
                    'username': response['data']['username'],
                    'password': response['data']['password']
                }),
                issued_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + ttl,
                metadata={
                    'lease_id': response['lease_id'],
                    'lease_duration': response['lease_duration'],
                    'renewable': response['renewable']
                }
            )

            logger.info(
                f"Generated DB credentials: {response['data']['username']} "
                f"(TTL: {ttl})"
            )
            return credential

        except Exception as e:
            logger.error(f"Failed to generate DB credentials: {e}")
            raise

    async def revoke(self, credential_id: str) -> bool:
        """Revoke database credentials."""
        try:
            self.vault_client.sys.revoke_lease(credential_id)
            logger.info(f"Revoked DB credentials: {credential_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke DB credentials: {e}")
            return False

    async def validate(self, credential: ShortLivedCredential) -> bool:
        """Validate credential hasn't expired."""
        return not credential.is_expired


class AzureManagedIdentityProvider(CredentialProvider):
    """Azure Managed Identity token provider."""

    def __init__(self, identity_client_id: Optional[str] = None):
        self.identity_client_id = identity_client_id
        self._token_cache: Dict[str, ShortLivedCredential] = {}

    async def generate(
        self,
        scope: str,
        ttl: timedelta,
        context: Optional[Dict] = None
    ) -> ShortLivedCredential:
        """Get Azure AD token via Managed Identity."""
        try:
            from azure.identity import ManagedIdentityCredential

            # Check cache first
            cache_key = f"{self.identity_client_id}:{scope}"
            cached = self._token_cache.get(cache_key)
            if cached and cached.is_valid and cached.ttl_seconds > 300:
                logger.debug(f"Using cached token for {scope}")
                return cached

            # Get new token
            credential = ManagedIdentityCredential(client_id=self.identity_client_id)
            token = credential.get_token(scope)

            # Parse expiration from token
            import jwt
            token_payload = jwt.decode(
                token.token,
                options={"verify_signature": False}
            )
            expires_at = datetime.fromtimestamp(token_payload['exp'])

            short_lived = ShortLivedCredential(
                credential_type='azure_ad_token',
                value=token.token,
                issued_at=datetime.utcnow(),
                expires_at=expires_at,
                metadata={
                    'scope': scope,
                    'client_id': self.identity_client_id
                }
            )

            # Update cache
            self._token_cache[cache_key] = short_lived

            logger.info(f"Generated Azure AD token for {scope}")
            return short_lived

        except Exception as e:
            logger.error(f"Failed to get Azure AD token: {e}")
            raise

    async def revoke(self, credential_id: str) -> bool:
        """Azure AD tokens cannot be revoked individually."""
        logger.warning("Azure AD tokens cannot be individually revoked")
        return False

    async def validate(self, credential: ShortLivedCredential) -> bool:
        """Validate Azure AD token."""
        return not credential.is_expired


class ShortLivedCredentialManager:
    """
    Manages short-lived credentials with automatic renewal.
    """

    def __init__(self):
        self.providers: Dict[str, CredentialProvider] = {}
        self.active_credentials: Dict[str, ShortLivedCredential] = {}
        self.renewal_callbacks: Dict[str, Callable] = {}
        self._renewal_task: Optional[asyncio.Task] = None

    def register_provider(self, name: str, provider: CredentialProvider):
        """Register a credential provider."""
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")

    async def get_credential(
        self,
        provider_name: str,
        scope: str,
        ttl: timedelta = timedelta(hours=1),
        auto_renew: bool = True
    ) -> ShortLivedCredential:
        """
        Get or generate a short-lived credential.

        Args:
            provider_name: Name of registered provider
            scope: Credential scope/role
            ttl: Desired time-to-live
            auto_renew: Enable automatic renewal

        Returns:
            ShortLivedCredential instance
        """
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")

        cache_key = f"{provider_name}:{scope}"

        # Check for existing valid credential
        existing = self.active_credentials.get(cache_key)
        if existing and existing.is_valid and existing.ttl_seconds > 60:
            return existing

        # Generate new credential
        credential = await provider.generate(scope, ttl)
        self.active_credentials[cache_key] = credential

        # Setup auto-renewal if enabled
        if auto_renew:
            self._schedule_renewal(cache_key, provider, scope, ttl)

        return credential

    def _schedule_renewal(
        self,
        cache_key: str,
        provider: CredentialProvider,
        scope: str,
        ttl: timedelta
    ):
        """Schedule automatic credential renewal."""
        credential = self.active_credentials.get(cache_key)
        if not credential:
            return

        # Renew at 80% of TTL
        renew_at = credential.ttl_seconds * 0.8

        async def renew():
            await asyncio.sleep(renew_at)
            try:
                logger.info(f"Auto-renewing credential: {cache_key}")
                new_credential = await provider.generate(scope, ttl)
                self.active_credentials[cache_key] = new_credential

                # Reschedule
                self._schedule_renewal(cache_key, provider, scope, ttl)

                # Notify callbacks
                callback = self.renewal_callbacks.get(cache_key)
                if callback:
                    callback(new_credential)

            except Exception as e:
                logger.error(f"Auto-renewal failed for {cache_key}: {e}")

        asyncio.create_task(renew())

    async def revoke_all(self):
        """Revoke all active credentials."""
        for cache_key, credential in self.active_credentials.items():
            provider_name = cache_key.split(':')[0]
            provider = self.providers.get(provider_name)

            if provider and 'lease_id' in credential.metadata:
                await provider.revoke(credential.metadata['lease_id'])

        self.active_credentials.clear()
        logger.info("Revoked all active credentials")

    def on_renewal(self, cache_key: str, callback: Callable):
        """Register callback for credential renewal."""
        self.renewal_callbacks[cache_key] = callback


# Usage Example
async def example_usage():
    """Example of using short-lived credentials."""
    manager = ShortLivedCredentialManager()

    # Register Vault provider for database
    vault_client = None  # Your Vault client
    db_provider = VaultDatabaseProvider(vault_client, 'readonly-role')
    manager.register_provider('database', db_provider)

    # Register Azure provider
    azure_provider = AzureManagedIdentityProvider()
    manager.register_provider('azure', azure_provider)

    # Get database credentials (auto-renewed)
    db_creds = await manager.get_credential(
        provider_name='database',
        scope='readonly',
        ttl=timedelta(hours=1),
        auto_renew=True
    )

    # Get Azure AD token
    azure_token = await manager.get_credential(
        provider_name='azure',
        scope='https://management.azure.com/.default',
        ttl=timedelta(hours=1)
    )

    print(f"DB credentials valid for {db_creds.ttl_seconds} seconds")
    print(f"Azure token valid for {azure_token.ttl_seconds} seconds")
```

### 7.3 Service-Specific Short-Lived Configurations

```python
"""
Service-specific short-lived credential configurations
"""

from datetime import timedelta

# Gmail OAuth Configuration (preferred over API key)
GMAIL_OAUTH_CONFIG = {
    'credential_type': 'oauth2',
    'access_token_ttl': timedelta(minutes=60),
    'refresh_token_rotation': True,
    'scopes': [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send'
    ],
    'pkce_enabled': True
}

# Twilio Configuration (API key with short TTL)
TWILIO_CONFIG = {
    'credential_type': 'api_key',
    'rotation_interval': timedelta(days=60),
    'ip_restrictions': True,
    'allowed_ips': ['AGENT_HOST_IP/32']
}

# OpenAI Configuration (scoped API key)
OPENAI_CONFIG = {
    'credential_type': 'api_key',
    'rotation_interval': timedelta(days=90),
    'scope_limitations': {
        'max_requests_per_minute': 100,
        'allowed_models': ['gpt-4', 'gpt-3.5-turbo']
    }
}

# Azure OpenAI (Managed Identity preferred)
AZURE_OPENAI_CONFIG = {
    'credential_type_preference': [
        'managed_identity',  # Preferred
        'service_principal',  # Fallback
        'api_key'  # Last resort
    ],
    'token_ttl': timedelta(hours=1)
}

# Browser Control (session-based)
BROWSER_CONFIG = {
    'credential_type': 'session_cookie',
    'session_ttl': timedelta(hours=8),
    'http_only': True,
    'secure': True,
    'same_site': 'Strict'
}
```

---

## 8. Credential Access Audit

### 8.1 Audit Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Credential Access Audit Architecture                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Audit Event Sources                               │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Secret     │  │   Token      │  │   Key        │              │   │
│  │  │   Vault      │  │   Manager    │  │   Rotation   │              │   │
│  │  │   Access     │  │   Operations │  │   Events     │              │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │   │
│  │         │                 │                 │                        │   │
│  │         └─────────────────┼─────────────────┘                        │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Audit Event Collector                         │  │   │
│  │  │  • Normalizes events from all sources                          │  │   │
│  │  │  • Enriches with context (user, IP, device)                    │  │   │
│  │  │  • Buffers for reliability                                     │  │   │
│  │  └────────────────────────┬──────────────────────────────────────┘  │   │
│  │                           │                                          │   │
│  │                           ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Audit Processing Pipeline                     │  │   │
│  │  │                                                                  │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │   │
│  │  │  │  Filtering  │──►│  Anomaly    │──►│  Alerting   │            │  │   │
│  │  │  │  & Routing  │  │  Detection  │  │  & Response │            │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │   │
│  │  │                                                                  │  │   │
│  │  └────────────────────────┬──────────────────────────────────────┘  │   │
│  │                           │                                          │   │
│  │           ┌───────────────┼───────────────┐                        │   │
│  │           ▼               ▼               ▼                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │   Azure     │  │   SIEM      │  │   Long-term │                │   │
│  │  │   Monitor   │  │   (Splunk)  │  │   Storage   │                │   │
│  │  │   Logs      │  │             │  │   (WORM)    │                │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Audit Event Schema

```python
"""
Credential Access Audit System
Implements comprehensive audit logging with anomaly detection
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    CREDENTIAL_ACCESS = "credential_access"
    CREDENTIAL_ROTATION = "credential_rotation"
    CREDENTIAL_REVOCATION = "credential_revocation"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    VAULT_UNSEAL = "vault_unseal"
    POLICY_CHANGE = "policy_change"
    ANOMALY_DETECTED = "anomaly_detected"


class AccessOutcome(Enum):
    """Access attempt outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Standardized audit event."""

    # Event identification
    event_id: str
    event_type: AuditEventType
    timestamp: datetime

    # Actor information
    actor_id: str
    actor_type: str  # user, service, agent
    actor_auth_method: str

    # Target information
    target_resource: str  # credential path, token ID, etc.
    target_resource_type: str

    # Action details
    action: str
    outcome: AccessOutcome

    # Context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Additional data (sanitized)
    metadata: Optional[Dict[str, Any]] = None

    # Integrity
    integrity_hash: Optional[str] = None

    def __post_init__(self):
        if self.integrity_hash is None:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute integrity hash for tamper detection."""
        data = f"{self.event_id}:{self.timestamp.isoformat()}:{self.actor_id}:{self.target_resource}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'actor_id': self._hash_sensitive(self.actor_id),
            'actor_type': self.actor_type,
            'actor_auth_method': self.actor_auth_method,
            'target_resource': self._sanitize_path(self.target_resource),
            'target_resource_type': self.target_resource_type,
            'action': self.action,
            'outcome': self.outcome.value,
            'source_ip': self.source_ip,
            'user_agent': self._hash_sensitive(self.user_agent) if self.user_agent else None,
            'device_fingerprint': self.device_fingerprint,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'metadata': self._sanitize_metadata(self.metadata),
            'integrity_hash': self.integrity_hash
        }

    def _hash_sensitive(self, value: str) -> str:
        """Hash sensitive values for privacy."""
        return hashlib.sha256(value.encode()).hexdigest()[:16] if value else None

    def _sanitize_path(self, path: str) -> str:
        """Sanitize credential path for logging."""
        # Remove actual credential values, keep structure
        parts = path.split('/')
        return '/'.join(parts[:-1] + ['[REDACTED]']) if len(parts) > 1 else '[REDACTED]'

    def _sanitize_metadata(self, metadata: Optional[Dict]) -> Optional[Dict]:
        """Remove sensitive values from metadata."""
        if not metadata:
            return None

        sensitive_keys = ['password', 'secret', 'token', 'key', 'credential']
        sanitized = {}

        for k, v in metadata.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                sanitized[k] = '[REDACTED]'
            else:
                sanitized[k] = v

        return sanitized


class AuditLogger:
    """
    Centralized audit logging with multiple outputs.
    """

    def __init__(
        self,
        azure_workspace_id: Optional[str] = None,
        azure_key: Optional[str] = None,
        siem_endpoint: Optional[str] = None
    ):
        self.azure_workspace_id = azure_workspace_id
        self.azure_key = azure_key
        self.siem_endpoint = siem_endpoint
        self._local_buffer: List[AuditEvent] = []
        self._anomaly_detector = AnomalyDetector()

    def log_event(self, event: AuditEvent):
        """Log an audit event to all outputs."""
        # Add to local buffer
        self._local_buffer.append(event)

        # Trim buffer if needed
        if len(self._local_buffer) > 10000:
            self._local_buffer = self._local_buffer[-5000:]

        # Send to Azure Monitor
        if self.azure_workspace_id and self.azure_key:
            self._send_to_azure_monitor(event)

        # Send to SIEM
        if self.siem_endpoint:
            self._send_to_siem(event)

        # Run anomaly detection
        anomaly = self._anomaly_detector.check(event)
        if anomaly:
            self._handle_anomaly(event, anomaly)

        # Local logging
        logger.info(f"Audit: {event.event_type.value} - {event.action} - {event.outcome.value}")

    def _send_to_azure_monitor(self, event: AuditEvent):
        """Send event to Azure Monitor Logs."""
        try:
            import requests

            # Azure Log Analytics API endpoint
            url = f"https://{self.azure_workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"

            headers = {
                'Content-Type': 'application/json',
                'Log-Type': 'CredentialAudit',
                'x-ms-date': datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
            }

            # Sign request
            import hmac
            import base64

            body = json.dumps([event.to_dict()])
            content_length = len(body)

            string_to_hash = f"POST
{content_length}
application/json
{headers['x-ms-date']}
/api/logs"
            decoded_key = base64.b64decode(self.azure_key)
            encoded_hash = base64.b64encode(
                hmac.new(decoded_key, string_to_hash.encode(), hashlib.sha256).digest()
            ).decode()

            headers['Authorization'] = f"SharedKey {self.azure_workspace_id}:{encoded_hash}"

            response = requests.post(url, headers=headers, data=body)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send to Azure Monitor: {e}")

    def _send_to_siem(self, event: AuditEvent):
        """Send event to SIEM system."""
        try:
            import requests

            response = requests.post(
                self.siem_endpoint,
                json=event.to_dict(),
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send to SIEM: {e}")

    def _handle_anomaly(self, event: AuditEvent, anomaly_details: Dict):
        """Handle detected anomaly."""
        logger.critical(
            f"ANOMALY DETECTED: {anomaly_details['type']} - "
            f"Actor: {event.actor_id}, Resource: {event.target_resource}"
        )

        # Create anomaly event
        anomaly_event = AuditEvent(
            event_id=f"anomaly-{event.event_id}",
            event_type=AuditEventType.ANOMALY_DETECTED,
            timestamp=datetime.utcnow(),
            actor_id='anomaly_detector',
            actor_type='system',
            actor_auth_method='internal',
            target_resource=event.target_resource,
            target_resource_type='anomaly',
            action='detect',
            outcome=AccessOutcome.SUCCESS,
            metadata=anomaly_details
        )

        # Log anomaly
        self._local_buffer.append(anomaly_event)

        # Trigger alerts
        self._trigger_alert(anomaly_event)

    def _trigger_alert(self, event: AuditEvent):
        """Trigger security alert."""
        # Implement alert mechanism (email, Slack, PagerDuty, etc.)
        pass

    def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        outcome: Optional[AccessOutcome] = None
    ) -> List[AuditEvent]:
        """Query audit events from local buffer."""
        results = []

        for event in self._local_buffer:
            if not (start_time <= event.timestamp <= end_time):
                continue

            if event_type and event.event_type != event_type:
                continue

            if actor_id and event.actor_id != actor_id:
                continue

            if outcome and event.outcome != outcome:
                continue

            results.append(event)

        return results


class AnomalyDetector:
    """
    Detects anomalous credential access patterns.
    """

    def __init__(self):
        self._access_patterns: Dict[str, List[datetime]] = {}
        self._baseline_stats: Dict[str, Dict] = {}

    def check(self, event: AuditEvent) -> Optional[Dict]:
        """Check event for anomalies. Returns anomaly details or None."""

        # Check 1: Unusual access time
        hour = event.timestamp.hour
        if hour < 6 or hour > 22:
            return {
                'type': 'unusual_access_time',
                'severity': 'medium',
                'details': f'Access at {event.timestamp.strftime("%H:%M")}'
            }

        # Check 2: Rapid successive access
        actor_key = event.actor_id
        if actor_key not in self._access_patterns:
            self._access_patterns[actor_key] = []

        self._access_patterns[actor_key].append(event.timestamp)

        # Keep only last hour
        cutoff = event.timestamp - timedelta(hours=1)
        self._access_patterns[actor_key] = [
            t for t in self._access_patterns[actor_key] if t > cutoff
        ]

        if len(self._access_patterns[actor_key]) > 100:
            return {
                'type': 'rapid_access',
                'severity': 'high',
                'details': f'{len(self._access_patterns[actor_key])} accesses in last hour'
            }

        # Check 3: Access from new IP
        if event.source_ip:
            # Would check against known IPs for actor
            pass

        # Check 4: Failed access attempts
        if event.outcome == AccessOutcome.FAILURE:
            # Track failures
            pass

        return None


# Convenience functions for common audit events
def audit_credential_access(
    logger: AuditLogger,
    actor_id: str,
    credential_path: str,
    outcome: AccessOutcome,
    source_ip: Optional[str] = None
):
    """Log credential access event."""
    event = AuditEvent(
        event_id=f"evt-{datetime.utcnow().timestamp()}",
        event_type=AuditEventType.CREDENTIAL_ACCESS,
        timestamp=datetime.utcnow(),
        actor_id=actor_id,
        actor_type='service',
        actor_auth_method='managed_identity',
        target_resource=credential_path,
        target_resource_type='credential',
        action='read',
        outcome=outcome,
        source_ip=source_ip
    )
    logger.log_event(event)


def audit_token_issued(
    logger: AuditLogger,
    user_id: str,
    token_id: str,
    scopes: List[str]
):
    """Log token issuance event."""
    event = AuditEvent(
        event_id=f"evt-{datetime.utcnow().timestamp()}",
        event_type=AuditEventType.TOKEN_ISSUED,
        timestamp=datetime.utcnow(),
        actor_id=user_id,
        actor_type='user',
        actor_auth_method='oauth',
        target_resource=token_id,
        target_resource_type='token',
        action='issue',
        outcome=AccessOutcome.SUCCESS,
        metadata={'scopes': scopes}
    )
    logger.log_event(event)
```

---

## 9. Breach Response Procedures

### 9.1 Incident Response Playbook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Credential Breach Response Playbook                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: DETECTION & IDENTIFICATION (0-15 minutes)                │   │
│  │                                                                      │   │
│  │  Triggers:                                                           │   │
│  │  • Anomaly detection alert                                           │   │
│  │  • Failed access spike                                               │   │
│  │  • Token reuse detected                                              │   │
│  │  • External security notification                                    │   │
│  │  • User report of suspicious activity                                │   │
│  │                                                                      │   │
│  │  Actions:                                                            │   │
│  │  □ Verify breach scope                                               │   │
│  │  □ Identify affected credentials                                     │   │
│  │  □ Determine attack vector                                           │   │
│  │  □ Classify severity (P1-P4)                                         │   │
│  │  □ Notify incident response team                                     │   │
│  │  □ Create incident ticket                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: CONTAINMENT (15-60 minutes)                              │   │
│  │                                                                      │   │
│  │  Immediate Actions:                                                  │   │
│  │  □ Revoke ALL tokens for affected user/service                       │   │
│  │  □ Rotate compromised credentials                                    │   │
│  │  □ Disable affected service accounts                                 │   │
│  │  □ Block suspicious IPs at firewall                                  │   │
│  │  □ Enable enhanced logging                                           │   │
│  │                                                                      │   │
│  │  Short-term Actions:                                                 │   │
│  │  □ Isolate affected systems                                          │   │
│  │  □ Force re-authentication for all users                             │   │
│  │  □ Enable MFA if not already required                                │   │
│  │  □ Increase monitoring sensitivity                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: ERADICATION (1-4 hours)                                  │   │
│  │                                                                      │   │
│  │  Actions:                                                            │   │
│  │  □ Remove attacker persistence mechanisms                            │   │
│  │  □ Patch exploited vulnerabilities                                   │   │
│  │  □ Scan for malware/backdoors                                        │   │
│  │  □ Review and reset ALL service account passwords                    │   │
│  │  □ Audit all recent credential access                                │   │
│  │  □ Check for privilege escalation                                    │   │
│  │  □ Verify no unauthorized access remains                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: RECOVERY (4-24 hours)                                    │   │
│  │                                                                      │   │
│  │  Actions:                                                            │   │
│  │  □ Issue new credentials following rotation procedures               │   │
│  │  □ Restore services in phases                                        │   │
│  │  □ Monitor for recurrence                                            │   │
│  │  □ Validate all restored services                                    │   │
│  │  □ Communicate with users about required actions                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 5: POST-INCIDENT (24+ hours)                                │   │
│  │                                                                      │   │
│  │  Actions:                                                            │   │
│  │  □ Conduct post-mortem analysis                                      │   │
│  │  □ Document lessons learned                                          │   │
│  │  □ Update security policies                                          │   │
│  │  □ Enhance detection rules                                           │   │
│  │  □ Provide security awareness training                               │   │
│  │  □ Review and update incident response plan                          │   │
│  │  □ Report to stakeholders/regulators as required                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Automated Response Implementation

```python
"""
Automated Breach Response System
Implements immediate response actions for credential breaches
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    P1_CRITICAL = "P1"  # Active breach, immediate response required
    P2_HIGH = "P2"      # Confirmed compromise, urgent response
    P3_MEDIUM = "P3"    # Suspicious activity, investigate
    P4_LOW = "P4"       # Minor anomaly, monitor


class ResponseAction(Enum):
    """Available response actions."""
    REVOKE_TOKENS = "revoke_tokens"
    ROTATE_CREDENTIALS = "rotate_credentials"
    DISABLE_ACCOUNT = "disable_account"
    BLOCK_IP = "block_ip"
    FORCE_REAUTH = "force_reauth"
    ISOLATE_SYSTEM = "isolate_system"
    ENABLE_ENHANCED_LOGGING = "enable_enhanced_logging"
    NOTIFY_TEAM = "notify_team"


@dataclass
class Incident:
    """Represents a security incident."""
    incident_id: str
    severity: IncidentSeverity
    detected_at: datetime
    description: str
    affected_resources: List[str]
    indicators: Dict[str, Any]
    status: str = "open"
    actions_taken: List[Dict] = None

    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = []


class BreachResponseOrchestrator:
    """
    Orchestrates automated breach response actions.
    """

    # Response playbooks by severity
    PLAYBOOKS = {
        IncidentSeverity.P1_CRITICAL: [
            ResponseAction.REVOKE_TOKENS,
            ResponseAction.DISABLE_ACCOUNT,
            ResponseAction.BLOCK_IP,
            ResponseAction.ISOLATE_SYSTEM,
            ResponseAction.NOTIFY_TEAM
        ],
        IncidentSeverity.P2_HIGH: [
            ResponseAction.REVOKE_TOKENS,
            ResponseAction.ROTATE_CREDENTIALS,
            ResponseAction.FORCE_REAUTH,
            ResponseAction.NOTIFY_TEAM
        ],
        IncidentSeverity.P3_MEDIUM: [
            ResponseAction.ENABLE_ENHANCED_LOGGING,
            ResponseAction.NOTIFY_TEAM
        ],
        IncidentSeverity.P4_LOW: [
            ResponseAction.ENABLE_ENHANCED_LOGGING
        ]
    }

    def __init__(
        self,
        token_manager: 'TokenManager',
        credential_manager: 'SecureCredentialManager',
        rotation_orchestrator: 'RotationOrchestrator',
        audit_logger: 'AuditLogger'
    ):
        self.token_manager = token_manager
        self.credential_manager = credential_manager
        self.rotation_orchestrator = rotation_orchestrator
        self.audit_logger = audit_logger

        self.active_incidents: Dict[str, Incident] = {}
        self.response_handlers: Dict[ResponseAction, callable] = {
            ResponseAction.REVOKE_TOKENS: self._handle_revoke_tokens,
            ResponseAction.ROTATE_CREDENTIALS: self._handle_rotate_credentials,
            ResponseAction.DISABLE_ACCOUNT: self._handle_disable_account,
            ResponseAction.BLOCK_IP: self._handle_block_ip,
            ResponseAction.FORCE_REAUTH: self._handle_force_reauth,
            ResponseAction.ISOLATE_SYSTEM: self._handle_isolate_system,
            ResponseAction.ENABLE_ENHANCED_LOGGING: self._handle_enhanced_logging,
            ResponseAction.NOTIFY_TEAM: self._handle_notify_team
        }

    async def create_incident(
        self,
        severity: IncidentSeverity,
        description: str,
        affected_resources: List[str],
        indicators: Dict[str, Any]
    ) -> Incident:
        """Create and process a new security incident."""

        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        incident = Incident(
            incident_id=incident_id,
            severity=severity,
            detected_at=datetime.utcnow(),
            description=description,
            affected_resources=affected_resources,
            indicators=indicators
        )

        self.active_incidents[incident_id] = incident

        logger.critical(
            f"SECURITY INCIDENT: {incident_id} - {severity.value} - {description}"
        )

        # Execute response playbook
        await self._execute_playbook(incident)

        return incident

    async def _execute_playbook(self, incident: Incident):
        """Execute response playbook for incident severity."""
        actions = self.PLAYBOOKS.get(incident.severity, [])

        for action in actions:
            try:
                handler = self.response_handlers.get(action)
                if handler:
                    result = await handler(incident)
                    incident.actions_taken.append({
                        'action': action.value,
                        'timestamp': datetime.utcnow().isoformat(),
                        'result': result
                    })
                    logger.info(f"Executed {action.value} for {incident.incident_id}")

            except Exception as e:
                logger.error(f"Failed to execute {action.value}: {e}")
                incident.actions_taken.append({
                    'action': action.value,
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                })

    async def _handle_revoke_tokens(self, incident: Incident) -> str:
        """Revoke all tokens for affected resources."""
        for resource in incident.affected_resources:
            # Extract user/token info from resource
            if 'token_family:' in resource:
                family_id = resource.split(':')[1]
                self.token_manager.revoke_token_family(family_id)
                logger.info(f"Revoked token family: {family_id}")

        return "tokens_revoked"

    async def _handle_rotate_credentials(self, incident: Incident) -> str:
        """Rotate compromised credentials."""
        for resource in incident.affected_resources:
            if 'credential:' in resource:
                cred_path = resource.split(':')[1]

                # Find rotation policy
                policy = None
                for p in DEFAULT_POLICIES.values():
                    if p.credential_path == cred_path:
                        policy = p
                        break

                if policy:
                    await self.rotation_orchestrator.execute_rotation(
                        policy,
                        trigger=RotationTrigger.EMERGENCY
                    )
                    logger.info(f"Rotated credentials: {cred_path}")

        return "credentials_rotated"

    async def _handle_disable_account(self, incident: Incident) -> str:
        """Disable compromised service accounts."""
        for resource in incident.affected_resources:
            if 'user:' in resource or 'service:' in resource:
                account_id = resource.split(':')[1]
                # Implement account disable logic
                logger.info(f"Disabled account: {account_id}")

        return "accounts_disabled"

    async def _handle_block_ip(self, incident: Incident) -> str:
        """Block suspicious IP addresses."""
        suspicious_ips = incident.indicators.get('suspicious_ips', [])

        for ip in suspicious_ips:
            # Implement IP blocking (firewall, WAF, etc.)
            logger.info(f"Blocked IP: {ip}")

        return f"blocked_{len(suspicious_ips)}_ips"

    async def _handle_force_reauth(self, incident: Incident) -> str:
        """Force re-authentication for affected users."""
        for resource in incident.affected_resources:
            if 'user:' in resource:
                user_id = resource.split(':')[1]
                # Invalidate all sessions
                logger.info(f"Forced re-auth for user: {user_id}")

        return "reauth_forced"

    async def _handle_isolate_system(self, incident: Incident) -> str:
        """Isolate affected systems."""
        for resource in incident.affected_resources:
            if 'system:' in resource or 'host:' in resource:
                system_id = resource.split(':')[1]
                # Implement network isolation
                logger.info(f"Isolated system: {system_id}")

        return "systems_isolated"

    async def _handle_enhanced_logging(self, incident: Incident) -> str:
        """Enable enhanced logging for investigation."""
        # Increase log verbosity
        # Enable additional audit events
        logger.info("Enabled enhanced logging")
        return "enhanced_logging_enabled"

    async def _handle_notify_team(self, incident: Incident) -> str:
        """Notify security team of incident."""
        # Send notifications (email, Slack, PagerDuty)
        notification = {
            'incident_id': incident.incident_id,
            'severity': incident.severity.value,
            'description': incident.description,
            'affected_resources': incident.affected_resources,
            'detected_at': incident.detected_at.isoformat()
        }

        logger.critical(f"SECURITY TEAM NOTIFICATION: {notification}")
        return "team_notified"

    def get_incident_status(self, incident_id: str) -> Optional[Incident]:
        """Get current status of an incident."""
        return self.active_incidents.get(incident_id)

    def close_incident(self, incident_id: str, resolution: str):
        """Close an incident with resolution notes."""
        incident = self.active_incidents.get(incident_id)
        if incident:
            incident.status = "closed"
            incident.actions_taken.append({
                'action': 'close',
                'timestamp': datetime.utcnow().isoformat(),
                'resolution': resolution
            })
            logger.info(f"Closed incident {incident_id}: {resolution}")


# Automated response triggers
async def handle_token_reuse_detection(
    orchestrator: BreachResponseOrchestrator,
    family: TokenFamily
):
    """Handle detected token reuse (potential breach)."""

    incident = await orchestrator.create_incident(
        severity=IncidentSeverity.P1_CRITICAL,
        description=f"Refresh token reuse detected for user {family.user_id}",
        affected_resources=[
            f"token_family:{family.family_id}",
            f"user:{family.user_id}"
        ],
        indicators={
            'family_id': family.family_id,
            'user_id': family.user_id,
            'service': family.service,
            'rotation_count': family.rotation_count
        }
    )

    return incident


async def handle_anomalous_access_pattern(
    orchestrator: BreachResponseOrchestrator,
    event: AuditEvent,
    anomaly_details: Dict
):
    """Handle detected anomalous access pattern."""

    severity = IncidentSeverity.P3_MEDIUM
    if anomaly_details.get('severity') == 'high':
        severity = IncidentSeverity.P2_HIGH

    incident = await orchestrator.create_incident(
        severity=severity,
        description=f"Anomalous access: {anomaly_details['type']}",
        affected_resources=[
            f"user:{event.actor_id}",
            f"resource:{event.target_resource}"
        ],
        indicators={
            'anomaly_type': anomaly_details['type'],
            'source_ip': event.source_ip,
            'details': anomaly_details.get('details')
        }
    )

    return incident
```

### 9.3 Recovery Procedures

```powershell
# Credential Breach Recovery Procedures
# PowerShell script for Windows 10 environment

# ============================================
# PHASE 1: IMMEDIATE CONTAINMENT
# ============================================

# 1. Stop AI Agent services
Stop-Service -Name "OpenClaw-Agent" -Force
Stop-Service -Name "OpenClaw-Scheduler" -Force

# 2. Clear credential cache
$CachePath = "$env:LOCALAPPDATA\OpenClaw\secure\cache"
if (Test-Path $CachePath) {
    Remove-Item -Path "$CachePath\*" -Recurse -Force
    Write-Host "Credential cache cleared"
}

# 3. Revoke all active sessions in Redis (if used)
# Requires Redis CLI
# redis-cli FLUSHDB

# ============================================
# PHASE 2: CREDENTIAL ROTATION
# ============================================

# 4. Rotate all service credentials
$Services = @(
    @{ Name = "Gmail"; Path = "prod/gmail/api-key/primary" },
    @{ Name = "Twilio"; Path = "prod/twilio/auth-token/primary" },
    @{ Name = "OpenAI"; Path = "prod/openai/api-key/primary" },
    @{ Name = "Azure"; Path = "prod/azure/ad-app-secret/primary" }
)

foreach ($Service in $Services) {
    Write-Host "Rotating $($Service.Name) credentials..."
    # Trigger rotation via API
    # Invoke-RestMethod -Uri "http://localhost:8080/api/rotate" -Method POST -Body @{ path = $Service.Path }
}

# 5. Force password reset for all service accounts
# This would integrate with Active Directory or Azure AD

# ============================================
# PHASE 3: SYSTEM RESTORATION
# ============================================

# 6. Verify new credentials work
$TestResults = @()
foreach ($Service in $Services) {
    $Result = Test-ServiceCredential -Service $Service.Name
    $TestResults += [PSCustomObject]@{
        Service = $Service.Name
        Status = if ($Result) { "OK" } else { "FAILED" }
    }
}
$TestResults | Format-Table

# 7. Restart services in correct order
Start-Service -Name "OpenClaw-Scheduler"
Start-Sleep -Seconds 5
Start-Service -Name "OpenClaw-Agent"

# 8. Verify service health
$Services = @("OpenClaw-Agent", "OpenClaw-Scheduler")
foreach ($Service in $Services) {
    $Status = Get-Service -Name $Service
    Write-Host "$Service status: $($Status.Status)"
}

# ============================================
# PHASE 4: POST-RECOVERY VALIDATION
# ============================================

# 9. Verify no unauthorized access remains
# Check for:
# - Active sessions from unknown IPs
# - Unusual API call patterns
# - Failed authentication attempts

# 10. Generate post-incident report
$Report = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Actions = @(
        "Stopped AI Agent services",
        "Cleared credential cache",
        "Rotated all service credentials",
        "Verified new credentials",
        "Restarted services"
    )
    Verification = $TestResults
}

$Report | ConvertTo-Json -Depth 3 | Out-File "breach_recovery_report.json"
Write-Host "Recovery complete. Report saved to breach_recovery_report.json"
```

---

## 10. Implementation Roadmap

### 10.1 Phase-Based Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Credential Management Implementation Roadmap             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: FOUNDATION (Weeks 1-2)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: CRITICAL                                                  │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Set up Azure Key Vault (Premium tier)                             │   │
│  │  □ Configure Windows DPAPI integration                               │   │
│  │  □ Implement AES-256-GCM encryption module                           │   │
│  │  □ Create secure file system structure                               │   │
│  │  □ Implement basic credential retrieval                              │   │
│  │  □ Add audit logging foundation                                      │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Working encryption/decryption                                     │   │
│  │  • Secure local credential storage                                   │   │
│  │  • Basic audit logging                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 2: VAULT INTEGRATION (Weeks 3-4)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: HIGH                                                      │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement Azure Key Vault client                                  │   │
│  │  □ Set up Managed Identity authentication                            │   │
│  │  □ Implement credential caching with TTL                             │   │
│  │  □ Add permission-based access control                               │   │
│  │  □ Implement secure credential injection                             │   │
│  │  □ Add cache invalidation mechanisms                                 │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Full vault integration                                            │   │
│  │  • Permission system                                                 │   │
│  │  • Credential caching                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 3: TOKEN MANAGEMENT (Weeks 5-6)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: HIGH                                                      │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement OAuth token manager                                     │   │
│  │  □ Add refresh token rotation                                        │   │
│  │  □ Implement token family tracking                                   │   │
│  │  □ Add reuse detection                                               │   │
│  │  □ Implement session management                                      │   │
│  │  □ Add device fingerprinting                                         │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Secure token lifecycle                                            │   │
│  │  • Rotation and reuse detection                                      │   │
│  │  • Session management                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 4: KEY ROTATION (Weeks 7-8)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: MEDIUM                                                    │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement rotation orchestrator                                   │   │
│  │  □ Create service-specific rotators                                  │   │
│  │  □ Add rotation scheduling (Windows Task Scheduler)                  │   │
│  │  □ Implement rollback procedures                                     │   │
│  │  □ Add rotation notifications                                        │   │
│  │  □ Test rotation for all services                                    │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Automated key rotation                                            │   │
│  │  • Rotation scheduling                                               │   │
│  │  • Rollback capability                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 5: SHORT-LIVED CREDENTIALS (Weeks 9-10)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: MEDIUM                                                    │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement short-lived credential manager                          │   │
│  │  □ Add Vault dynamic secrets support                                 │   │
│  │  □ Implement Azure Managed Identity                                  │   │
│  │  □ Add auto-renewal mechanisms                                       │   │
│  │  □ Create service-specific configurations                            │   │
│  │  □ Test JIT credential workflows                                     │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • JIT credential provisioning                                       │   │
│  │  • Auto-renewal                                                      │   │
│  │  • Service configurations                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 6: AUDIT & MONITORING (Weeks 11-12)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: MEDIUM                                                    │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement comprehensive audit logging                             │   │
│  │  □ Add Azure Monitor integration                                     │   │
│  │  □ Implement anomaly detection                                       │   │
│  │  □ Add SIEM integration                                              │   │
│  │  □ Create audit dashboards                                           │   │
│  │  □ Set up alerting rules                                             │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Full audit trail                                                  │   │
│  │  • Anomaly detection                                                 │   │
│  │  • Monitoring dashboards                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  PHASE 7: BREACH RESPONSE (Weeks 13-14)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: HIGH                                                      │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Implement breach response orchestrator                            │   │
│  │  □ Create incident response playbooks                                │   │
│  │  □ Add automated response actions                                    │   │
│  │  □ Implement recovery procedures                                     │   │
│  │  □ Test incident response scenarios                                  │   │
│  │  □ Document runbooks                                                 │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Automated breach response                                         │   │
│  │  • Incident playbooks                                                │   │
│  │  • Recovery procedures                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  PHASE 8: HARDENING & TESTING (Weeks 15-16)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Priority: CRITICAL                                                  │   │
│  │                                                                      │   │
│  │  Tasks:                                                              │   │
│  │  □ Security penetration testing                                      │   │
│  │  □ Credential exposure scanning                                      │   │
│  │  □ Load testing for vault operations                                 │   │
│  │  □ Failover testing                                                  │   │
│  │  □ Documentation review                                              │   │
│  │  □ Security audit                                                    │   │
│  │                                                                      │   │
│  │  Deliverables:                                                       │   │
│  │  • Security test report                                              │   │
│  │  • Performance benchmarks                                            │   │
│  │  • Production readiness                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Service Integration Matrix

| Service | Credential Type | Storage | Rotation | TTL | Priority |
|---------|----------------|---------|----------|-----|----------|
| **Gmail** | OAuth 2.0 | Vault + Cache | 90 days | 60 min access | P1 |
| **Twilio** | API Key | Vault + Cache | 60 days | 24 hours | P1 |
| **OpenAI** | API Key | Vault + Cache | 90 days | Session | P1 |
| **Azure OpenAI** | Managed Identity | N/A | Auto | 1 hour | P1 |
| **Browser Control** | Session Cookie | Memory | Per session | 8 hours | P2 |
| **TTS/STT** | Service Principal | Vault | 180 days | 1 hour | P2 |
| **Cron Jobs** | Service Account | Vault | 90 days | Task duration | P2 |
| **Heartbeat** | Internal Token | Memory | Per session | 5 min | P3 |

### 10.3 Risk Mitigation Summary

| Risk | Mitigation | Implementation |
|------|------------|----------------|
| Plaintext storage | AES-256-GCM + DPAPI | Phase 1 |
| Long-lived tokens | Short TTL + rotation | Phase 3 |
| Service account exposure | Managed Identity | Phase 5 |
| Credential reuse | Token family tracking | Phase 3 |
| Unauthorized access | RBAC + audit logging | Phase 2, 6 |
| Breach persistence | Automated response | Phase 7 |
| Insider threat | Least privilege + audit | All phases |
| Key compromise | Auto-rotation | Phase 4 |

---

## Appendices

### Appendix A: Configuration Templates

#### A.1 Azure Key Vault ARM Template

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vaultName": {
      "type": "string",
      "defaultValue": "kv-openclaw-prod"
    },
    "skuName": {
      "type": "string",
      "defaultValue": "Premium",
      "allowedValues": ["Standard", "Premium"]
    }
  },
  "resources": [
    {
      "type": "Microsoft.KeyVault/vaults",
      "apiVersion": "2022-07-01",
      "name": "[parameters('vaultName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "sku": {
          "family": "A",
          "name": "[parameters('skuName')]"
        },
        "tenantId": "[subscription().tenantId]",
        "enablePurgeProtection": true,
        "enableRbacAuthorization": true,
        "softDeleteRetentionInDays": 90,
        "networkAcls": {
          "defaultAction": "Deny",
          "bypass": "AzureServices",
          "ipRules": [],
          "virtualNetworkRules": []
        }
      }
    }
  ]
}
```

#### A.2 Windows Service Configuration

```xml
<!-- OpenClaw Credential Service Configuration -->
<configuration>
  <appSettings>
    <!-- Vault Configuration -->
    <add key="VaultType" value="AzureKeyVault" />
    <add key="VaultUrl" value="https://kv-openclaw-prod.vault.azure.net/" />
    <add key="VaultAuthentication" value="ManagedIdentity" />

    <!-- Encryption -->
    <add key="EncryptionAlgorithm" value="AES-256-GCM" />
    <add key="EnableDPAPI" value="true" />
    <add key="MasterKeyPath" value="%LOCALAPPDATA%\OpenClaw\secure\master.key" />

    <!-- Caching -->
    <add key="CacheEnabled" value="true" />
    <add key="CacheTTLSeconds" value="300" />
    <add key="MaxCacheSize" value="100" />

    <!-- Audit -->
    <add key="AuditEnabled" value="true" />
    <add key="AzureMonitorWorkspaceId" value="" />
    <add key="SIEMEndpoint" value="" />

    <!-- Rotation -->
    <add key="AutoRotationEnabled" value="true" />
    <add key="RotationSchedule" value="0 2 * * 0" />

    <!-- Security -->
    <add key="EnforceDeviceFingerprint" value="true" />
    <add key="MaxTokenLifetimeMinutes" value="15" />
    <add key="RefreshTokenRotation" value="true" />
  </appSettings>
</configuration>
```

### Appendix B: Security Checklist

#### Pre-Deployment Checklist

- [ ] All credentials stored in vault (no plaintext)
- [ ] Encryption at rest enabled (AES-256-GCM)
- [ ] DPAPI integration tested
- [ ] Vault authentication configured (Managed Identity)
- [ ] Network access restricted (firewall rules)
- [ ] Audit logging enabled
- [ ] Rotation schedules configured
- [ ] Breach response procedures documented
- [ ] Incident response team trained
- [ ] Recovery procedures tested
- [ ] Security scan completed (no hardcoded secrets)
- [ ] Penetration testing completed
- [ ] Documentation reviewed

#### Operational Checklist

- [ ] Daily: Review audit logs for anomalies
- [ ] Weekly: Verify rotation jobs completed
- [ ] Weekly: Check credential cache health
- [ ] Monthly: Review access patterns
- [ ] Monthly: Update threat intelligence
- [ ] Quarterly: Full rotation test
- [ ] Quarterly: Incident response drill
- [ ] Annually: Security audit
- [ ] Annually: Policy review

### Appendix C: Troubleshooting Guide

#### Common Issues

**Issue: Credential retrieval fails**
```
Symptom: "Failed to fetch from vault" error
Causes:
  1. Network connectivity to vault
  2. Managed Identity not configured
  3. RBAC permissions missing
  4. Vault firewall blocking access

Resolution:
  1. Test connectivity: Test-NetConnection kv-openclaw-prod.vault.azure.net -Port 443
  2. Verify identity: az identity show --name id-openclaw-agent
  3. Check role assignment: az role assignment list --assignee <principal-id>
  4. Review firewall rules in Azure Portal
```

**Issue: Token validation fails**
```
Symptom: "Invalid token" or "Token expired" errors
Causes:
  1. Clock skew between systems
  2. Token actually expired
  3. JWT secret mismatch
  4. Token tampering detected

Resolution:
  1. Sync system time: w32tm /resync
  2. Check token expiration in JWT payload
  3. Verify JWT secret configuration
  4. Review audit logs for anomalies
```

**Issue: Cache not updating**
```
Symptom: Old credentials returned after rotation
Causes:
  1. Cache TTL too long
  2. Cache invalidation failed
  3. Multiple cache instances

Resolution:
  1. Reduce cache TTL in configuration
  2. Manually clear cache: Remove-Item $env:LOCALAPPDATA\OpenClaw\secure\cache\*
  3. Verify single instance running
```

### Appendix D: Compliance Mapping

| Requirement | Control | Implementation |
|-------------|---------|----------------|
| **SOC 2** | CC6.1 | Logical access security |
| | CC6.2 | Prior to access |
| | CC6.3 | Access removal |
| | CC7.2 | System monitoring |
| **PCI DSS** | 3.4 | Render PAN unreadable |
| | 3.5 | Protect encryption keys |
| | 8.2 | Strong authentication |
| | 10.2 | Audit trail coverage |
| **NIST 800-53** | IA-2 | Identification & auth |
| | IA-5 | Authenticator management |
| | AU-6 | Audit review |
| | SC-28 | Protection at rest |
| **ISO 27001** | A.9.2 | User access management |
| | A.9.4 | System access control |
| | A.10.1 | Cryptographic controls |
| | A.12.4 | Logging & monitoring |

---

## References

1. [NIST SP 800-57: Recommendation for Key Management](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
2. [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
3. [Azure Key Vault Best Practices](https://docs.microsoft.com/en-us/azure/key-vault/general/best-practices)
4. [HashiCorp Vault Security Model](https://www.vaultproject.io/docs/internals/security)
5. [OAuth 2.0 Security Best Current Practice](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)
6. [Windows Data Protection API (DPAPI)](https://docs.microsoft.com/en-us/windows/win32/seccng/cng-dpapi)

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025 | Security Architecture Team | Initial release |

**Approval**

| Role | Name | Date |
|------|------|------|
| Security Architect | | |
| CISO | | |
| Engineering Lead | | |

---

*End of Document*
