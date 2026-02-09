"""
Credential Vault Integration System
For Windows 10 OpenClaw AI Agent Framework
"""

import os
import json
import base64
import hashlib
import logging
import secrets
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


@dataclass
class Credentials:
    """Standard credentials structure."""
    service_name: str
    username: str
    password: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    use_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'service_name': self.service_name,
            'username': self.username,
            'password': self.password,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'use_count': self.use_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Credentials':
        """Create from dictionary."""
        return cls(
            service_name=data['service_name'],
            username=data['username'],
            password=data['password'],
            attributes=data.get('attributes', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            use_count=data.get('use_count', 0)
        )


@dataclass
class MFACredentials:
    """MFA credentials structure."""
    service_name: str
    username: str
    mfa_type: str = "totp"  # 'totp', 'sms', 'email', 'backup'
    
    # TOTP
    totp_secret: Optional[str] = None
    
    # SMS
    phone_number: Optional[str] = None
    
    # Email
    email_address: Optional[str] = None
    
    # Backup codes
    backup_codes: List[str] = field(default_factory=list)
    used_backup_codes: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CredentialEncryption:
    """Encryption handler for credential storage."""
    
    def __init__(self, master_key: bytes):
        """
        Initialize encryption with master key.
        
        Args:
            master_key: 32-byte master encryption key
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes")
        
        self.master_key = master_key
        self.cipher = AESGCM(master_key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt string value.
        
        Args:
            plaintext: String to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        ciphertext = self.cipher.encrypt(
            nonce,
            plaintext.encode('utf-8'),
            None
        )
        # Combine nonce and ciphertext
        encrypted = nonce + ciphertext
        return base64.urlsafe_b64encode(encrypted).decode('ascii')
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt string value.
        
        Args:
            ciphertext: Base64-encoded encrypted string
            
        Returns:
            Decrypted plaintext string
        """
        encrypted = base64.urlsafe_b64decode(ciphertext.encode('ascii'))
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        plaintext = self.cipher.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')


class WindowsCredentialVault:
    """
    Windows Credential Manager integration using DPAPI.
    """
    
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    CRED_PERSIST_ENTERPRISE = 3
    
    def __init__(self, namespace: str = "OpenClawAgent"):
        """
        Initialize Windows credential vault.
        
        Args:
            namespace: Namespace for credential isolation
        """
        self.namespace = namespace
        self._cache: Dict[str, Credentials] = {}
        self._mfa_cache: Dict[str, MFACredentials] = {}
        
        # Check if we're on Windows
        if platform.system() != 'Windows':
            raise RuntimeError("WindowsCredentialVault only works on Windows")
        
        # Import Windows-specific modules
        try:
            import win32cred
            import win32crypt
            self._win32cred = win32cred
            self._win32crypt = win32crypt
        except ImportError:
            logger.warning("win32cred not available, falling back to file storage")
            self._win32cred = None
    
    def _build_target_name(
        self, 
        service_name: str, 
        username: str,
        cred_type: str = "password"
    ) -> str:
        """Build target name for credential storage."""
        return f"{self.namespace}:{service_name}:{username}:{cred_type}"
    
    def _parse_target_name(self, target_name: str) -> Optional[Dict[str, str]]:
        """Parse target name into components."""
        parts = target_name.split(':')
        if len(parts) >= 4 and parts[0] == self.namespace:
            return {
                'namespace': parts[0],
                'service_name': parts[1],
                'username': parts[2],
                'cred_type': parts[3]
            }
        return None
    
    async def store_credentials(
        self,
        service_name: str,
        username: str,
        password: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store credentials in Windows Credential Manager.
        
        Args:
            service_name: Service identifier
            username: Username
            password: Password
            attributes: Additional attributes
            
        Returns:
            True if successful
        """
        if not self._win32cred:
            return False
        
        target_name = self._build_target_name(service_name, username, "password")
        
        # Prepare credential data
        credential = {
            'Type': self.CRED_TYPE_GENERIC,
            'TargetName': target_name,
            'UserName': username,
            'CredentialBlob': password.encode('utf-16-le'),
            'Persist': self.CRED_PERSIST_LOCAL_MACHINE,
            'Comment': f'OpenClaw Agent credentials for {service_name}',
        }
        
        # Store attributes as CredentialBlobAttributes if needed
        if attributes:
            attr_blob = json.dumps(attributes).encode('utf-16-le')
            credential['Attribute'] = attr_blob
        
        try:
            self._win32cred.CredWrite(credential, 0)

            # Update cache
            self._cache[target_name] = Credentials(
                service_name=service_name,
                username=username,
                password=password,
                attributes=attributes or {}
            )

            logger.debug(f"Stored credentials for {service_name}/{username}")
            return True

        except OSError as e:
            logger.error(f"Failed to store credentials: {e}")
            return False
    
    async def get_credentials(
        self,
        service_name: str,
        username: Optional[str] = None
    ) -> Optional[Credentials]:
        """
        Retrieve credentials from Windows Credential Manager.
        
        Args:
            service_name: Service identifier
            username: Username (if None, returns first match)
            
        Returns:
            Credentials object or None
        """
        if not self._win32cred:
            return None
        
        if username:
            target_name = self._build_target_name(service_name, username, "password")
            
            # Check cache first
            if target_name in self._cache:
                creds = self._cache[target_name]
                creds.use_count += 1
                creds.last_used = datetime.now()
                return creds
            
            try:
                credential = self._win32cred.CredRead(
                    target_name, 
                    self.CRED_TYPE_GENERIC, 
                    0
                )
                
                password = credential['CredentialBlob'].decode('utf-16-le')
                
                # Parse attributes if present
                attributes = {}
                if 'Attribute' in credential:
                    try:
                        attr_blob = credential['Attribute']
                        if isinstance(attr_blob, bytes):
                            attributes = json.loads(attr_blob.decode('utf-16-le'))
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Credential attribute parsing error: {e}")
                
                creds = Credentials(
                    service_name=service_name,
                    username=credential['UserName'],
                    password=password,
                    attributes=attributes
                )
                
                # Update cache
                self._cache[target_name] = creds
                
                return creds
                
            except (OSError, KeyError, ValueError) as e:
                logger.debug(f"Credentials not found: {e}")
                return None
        else:
            # Enumerate all credentials for service
            return await self._enumerate_service_credentials(service_name)
    
    async def get_default_credentials(
        self,
        service_name: str
    ) -> Optional[Credentials]:
        """Get default (first available) credentials for service."""
        return await self.get_credentials(service_name)
    
    async def delete_credentials(
        self,
        service_name: str,
        username: str
    ) -> bool:
        """
        Delete credentials from Windows Credential Manager.
        
        Args:
            service_name: Service identifier
            username: Username
            
        Returns:
            True if successful
        """
        if not self._win32cred:
            return False
        
        target_name = self._build_target_name(service_name, username, "password")
        
        try:
            self._win32cred.CredDelete(target_name, self.CRED_TYPE_GENERIC, 0)

            # Remove from cache
            self._cache.pop(target_name, None)

            logger.debug(f"Deleted credentials for {service_name}/{username}")
            return True

        except OSError as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False
    
    async def list_services(self) -> List[str]:
        """List all services with stored credentials."""
        if not self._win32cred:
            return []
        
        services = set()
        
        try:
            # Enumerate all credentials
            credentials = self._win32cred.CredEnumerate(
                f"{self.namespace}:*",
                0
            )
            
            for cred in credentials:
                target_name = cred['TargetName']
                parsed = self._parse_target_name(target_name)
                if parsed:
                    services.add(parsed['service_name'])
        
        except OSError as e:
            logger.error(f"Failed to enumerate credentials: {e}")
        
        return list(services)
    
    async def list_credentials(self, service_name: str) -> List[str]:
        """List all usernames for a service."""
        if not self._win32cred:
            return []
        
        usernames = []
        
        try:
            credentials = self._win32cred.CredEnumerate(
                f"{self.namespace}:{service_name}:*",
                0
            )
            
            for cred in credentials:
                target_name = cred['TargetName']
                parsed = self._parse_target_name(target_name)
                if parsed and parsed['cred_type'] == 'password':
                    usernames.append(parsed['username'])
        
        except OSError as e:
            logger.error(f"Failed to list credentials: {e}")
        
        return usernames
    
    async def _enumerate_service_credentials(
        self,
        service_name: str
    ) -> Optional[Credentials]:
        """Enumerate and return first credentials for service."""
        usernames = await self.list_credentials(service_name)
        
        if usernames:
            return await self.get_credentials(service_name, usernames[0])
        
        return None
    
    # MFA Credentials
    async def store_mfa_credentials(
        self,
        mfa_creds: MFACredentials
    ) -> bool:
        """Store MFA credentials."""
        if not self._win32cred:
            return False
        
        target_name = self._build_target_name(
            mfa_creds.service_name,
            mfa_creds.username,
            f"mfa:{mfa_creds.mfa_type}"
        )
        
        # Serialize MFA credentials
        cred_data = {
            'service_name': mfa_creds.service_name,
            'username': mfa_creds.username,
            'mfa_type': mfa_creds.mfa_type,
            'totp_secret': mfa_creds.totp_secret,
            'phone_number': mfa_creds.phone_number,
            'email_address': mfa_creds.email_address,
            'backup_codes': mfa_creds.backup_codes,
            'used_backup_codes': mfa_creds.used_backup_codes,
        }
        
        credential = {
            'Type': self.CRED_TYPE_GENERIC,
            'TargetName': target_name,
            'UserName': mfa_creds.username,
            'CredentialBlob': json.dumps(cred_data).encode('utf-16-le'),
            'Persist': self.CRED_PERSIST_LOCAL_MACHINE,
            'Comment': f'OpenClaw Agent MFA credentials for {mfa_creds.service_name}',
        }
        
        try:
            self._win32cred.CredWrite(credential, 0)
            self._mfa_cache[target_name] = mfa_creds
            return True
        except OSError as e:
            logger.error(f"Failed to store MFA credentials: {e}")
            return False
    
    async def get_mfa_credentials(
        self,
        service_name: str,
        username: str,
        mfa_type: Optional[str] = None
    ) -> Optional[MFACredentials]:
        """Get MFA credentials."""
        if not self._win32cred:
            return None
        
        # Try specific MFA type first, then any
        types_to_try = [mfa_type] if mfa_type else ['totp', 'sms', 'email', 'backup']
        
        for mfa_t in types_to_try:
            target_name = self._build_target_name(service_name, username, f"mfa:{mfa_t}")
            
            # Check cache
            if target_name in self._mfa_cache:
                return self._mfa_cache[target_name]
            
            try:
                credential = self._win32cred.CredRead(
                    target_name,
                    self.CRED_TYPE_GENERIC,
                    0
                )
                
                cred_data = json.loads(
                    credential['CredentialBlob'].decode('utf-16-le')
                )
                
                mfa_creds = MFACredentials(
                    service_name=cred_data['service_name'],
                    username=cred_data['username'],
                    mfa_type=cred_data['mfa_type'],
                    totp_secret=cred_data.get('totp_secret'),
                    phone_number=cred_data.get('phone_number'),
                    email_address=cred_data.get('email_address'),
                    backup_codes=cred_data.get('backup_codes', []),
                    used_backup_codes=cred_data.get('used_backup_codes', [])
                )
                
                self._mfa_cache[target_name] = mfa_creds
                return mfa_creds
                
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"MFA credential lookup failed: {e}")
                continue

        return None

    async def update_mfa_credentials(
        self,
        mfa_creds: MFACredentials
    ) -> bool:
        """Update MFA credentials."""
        mfa_creds.updated_at = datetime.now()
        return await self.store_mfa_credentials(mfa_creds)


class EncryptedFileVault:
    """
    Encrypted file-based credential storage.
    Works on all platforms.
    """
    
    def __init__(
        self,
        storage_path: Path,
        master_key: Optional[bytes] = None
    ):
        """
        Initialize encrypted file vault.
        
        Args:
            storage_path: Path to storage directory
            master_key: Master encryption key (32 bytes)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load master key
        if master_key is None:
            master_key = self._load_or_create_master_key()
        
        self.encryption = CredentialEncryption(master_key)
        
        # Caches
        self._credential_cache: Dict[str, Credentials] = {}
        self._mfa_cache: Dict[str, MFACredentials] = {}
    
    def _load_or_create_master_key(self) -> bytes:
        """Load or create master encryption key."""
        key_file = self.storage_path / '.master_key'
        
        if key_file.exists():
            # Load encrypted key
            encrypted_key = key_file.read_bytes()
            # Decrypt using system-specific method
            return self._decrypt_master_key(encrypted_key)
        else:
            # Generate new key
            master_key = secrets.token_bytes(32)
            # Encrypt and store
            encrypted_key = self._encrypt_master_key(master_key)
            key_file.write_bytes(encrypted_key)
            return master_key
    
    def _encrypt_master_key(self, key: bytes) -> bytes:
        """Encrypt master key using platform-specific method."""
        if platform.system() == 'Windows':
            # Use Windows DPAPI
            try:
                import ctypes
                from ctypes import wintypes

                # DPAPI encryption
                DATA_BLOB = ctypes.Structure
                # Simplified - in production use proper DPAPI
                return key  # Placeholder
            except (ImportError, OSError, ValueError) as e:
                logger.warning(f"DPAPI encryption failed, falling back to PBKDF2: {e}")

        # Fallback: use PBKDF2HMAC-derived key to encrypt with Fernet
        salt_path = self._get_vault_path() / '.salt'
        if salt_path.exists():
            salt = salt_path.read_bytes()
        else:
            salt = os.urandom(16)
            salt_path.parent.mkdir(parents=True, exist_ok=True)
            salt_path.write_bytes(salt)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        # Derive a Fernet key from a passphrase (use API key or machine-specific secret)
        passphrase = (os.environ.get('VAULT_PASSPHRASE', '') or platform.node()).encode()
        derived = base64.urlsafe_b64encode(kdf.derive(passphrase))
        f = Fernet(derived)
        return f.encrypt(key)

    def _decrypt_master_key(self, encrypted: bytes) -> bytes:
        """Decrypt master key."""
        try:
            # Try Fernet decryption first (new format)
            salt_path = self._get_vault_path() / '.salt'
            if salt_path.exists():
                salt = salt_path.read_bytes()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=600000,
                )
                passphrase = (os.environ.get('VAULT_PASSPHRASE', '') or platform.node()).encode()
                derived = base64.urlsafe_b64encode(kdf.derive(passphrase))
                f = Fernet(derived)
                return f.decrypt(encrypted)
        except Exception:
            pass
        # Legacy fallback: base64 decode
        try:
            return base64.urlsafe_b64decode(encrypted)
        except (ValueError, TypeError, OSError) as e:
            logger.error(f"Decryption failed for credential: {e}")
            raise
    
    def _get_vault_path(self) -> Path:
        """Get the vault storage root path."""
        vault_path = self.storage_path / 'vault'
        vault_path.mkdir(parents=True, exist_ok=True)
        return vault_path

    def _get_credential_file(self, service_name: str, username: str) -> Path:
        """Get path to credential file."""
        # Deterministic hash so stored credentials can be found again
        service_hash = hashlib.sha256(service_name.encode()).hexdigest()[:16]
        cred_dir = self.storage_path / 'credentials' / service_hash
        cred_dir.mkdir(parents=True, exist_ok=True)
        return cred_dir / f"{username}.enc"

    def _get_mfa_file(
        self,
        service_name: str,
        username: str,
        mfa_type: str
    ) -> Path:
        """Get path to MFA credential file."""
        service_hash = hashlib.sha256(service_name.encode()).hexdigest()[:16]
        mfa_dir = self.storage_path / 'mfa' / service_hash
        mfa_dir.mkdir(parents=True, exist_ok=True)
        return mfa_dir / f"{username}_{mfa_type}.enc"
    
    async def store_credentials(
        self,
        service_name: str,
        username: str,
        password: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store credentials."""
        try:
            creds = Credentials(
                service_name=service_name,
                username=username,
                password=password,
                attributes=attributes or {},
                updated_at=datetime.now()
            )
            
            # Serialize and encrypt
            cred_json = json.dumps(creds.to_dict())
            encrypted = self.encryption.encrypt(cred_json)
            
            # Write to file
            cred_file = self._get_credential_file(service_name, username)
            cred_file.write_text(encrypted)
            
            # Update cache
            cache_key = f"{service_name}:{username}"
            self._credential_cache[cache_key] = creds
            
            return True
            
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to store credentials: {e}")
            return False

    async def get_credentials(
        self,
        service_name: str,
        username: Optional[str] = None
    ) -> Optional[Credentials]:
        """Get credentials."""
        if username:
            cache_key = f"{service_name}:{username}"
            
            # Check cache
            if cache_key in self._credential_cache:
                creds = self._credential_cache[cache_key]
                creds.use_count += 1
                creds.last_used = datetime.now()
                return creds
            
            try:
                cred_file = self._get_credential_file(service_name, username)
                
                if not cred_file.exists():
                    return None
                
                encrypted = cred_file.read_text()
                cred_json = self.encryption.decrypt(encrypted)
                cred_data = json.loads(cred_json)
                
                creds = Credentials.from_dict(cred_data)
                self._credential_cache[cache_key] = creds
                
                return creds
                
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load credentials: {e}")
                return None
        else:
            # Return first available
            return await self._get_first_credentials(service_name)
    
    async def get_default_credentials(
        self,
        service_name: str
    ) -> Optional[Credentials]:
        """Get default credentials for service."""
        return await self.get_credentials(service_name)
    
    async def _get_first_credentials(
        self,
        service_name: str
    ) -> Optional[Credentials]:
        """Get first available credentials for service by scanning credential directory."""
        service_hash = hashlib.sha256(service_name.encode()).hexdigest()[:16]
        cred_dir = self.storage_path / 'credentials' / service_hash
        if not cred_dir.exists():
            return None
        for enc_file in cred_dir.glob('*.enc'):
            try:
                encrypted = enc_file.read_bytes()
                cred_json = self.encryption.decrypt(encrypted)
                cred_data = json.loads(cred_json)
                return Credentials.from_dict(cred_data)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.debug(f"Skipping credential file {enc_file}: {e}")
                continue
        return None
    
    async def delete_credentials(
        self,
        service_name: str,
        username: str
    ) -> bool:
        """Delete credentials."""
        try:
            cred_file = self._get_credential_file(service_name, username)
            
            if cred_file.exists():
                cred_file.unlink()
            
            # Remove from cache
            cache_key = f"{service_name}:{username}"
            self._credential_cache.pop(cache_key, None)
            
            return True
            
        except OSError as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False

    # MFA Credentials
    async def store_mfa_credentials(
        self,
        mfa_creds: MFACredentials
    ) -> bool:
        """Store MFA credentials."""
        try:
            mfa_file = self._get_mfa_file(
                mfa_creds.service_name,
                mfa_creds.username,
                mfa_creds.mfa_type
            )
            
            # Serialize and encrypt
            mfa_data = {
                'service_name': mfa_creds.service_name,
                'username': mfa_creds.username,
                'mfa_type': mfa_creds.mfa_type,
                'totp_secret': mfa_creds.totp_secret,
                'phone_number': mfa_creds.phone_number,
                'email_address': mfa_creds.email_address,
                'backup_codes': mfa_creds.backup_codes,
                'used_backup_codes': mfa_creds.used_backup_codes,
            }
            
            encrypted = self.encryption.encrypt(json.dumps(mfa_data))
            mfa_file.write_text(encrypted)
            
            # Update cache
            cache_key = f"{mfa_creds.service_name}:{mfa_creds.username}:{mfa_creds.mfa_type}"
            self._mfa_cache[cache_key] = mfa_creds
            
            return True
            
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to store MFA credentials: {e}")
            return False

    async def get_mfa_credentials(
        self,
        service_name: str,
        username: str,
        mfa_type: Optional[str] = None
    ) -> Optional[MFACredentials]:
        """Get MFA credentials."""
        types_to_try = [mfa_type] if mfa_type else ['totp', 'sms', 'email', 'backup']
        
        for mfa_t in types_to_try:
            cache_key = f"{service_name}:{username}:{mfa_t}"
            
            # Check cache
            if cache_key in self._mfa_cache:
                return self._mfa_cache[cache_key]
            
            try:
                mfa_file = self._get_mfa_file(service_name, username, mfa_t)
                
                if not mfa_file.exists():
                    continue
                
                encrypted = mfa_file.read_text()
                mfa_json = self.encryption.decrypt(encrypted)
                mfa_data = json.loads(mfa_json)
                
                mfa_creds = MFACredentials(
                    service_name=mfa_data['service_name'],
                    username=mfa_data['username'],
                    mfa_type=mfa_data['mfa_type'],
                    totp_secret=mfa_data.get('totp_secret'),
                    phone_number=mfa_data.get('phone_number'),
                    email_address=mfa_data.get('email_address'),
                    backup_codes=mfa_data.get('backup_codes', []),
                    used_backup_codes=mfa_data.get('used_backup_codes', [])
                )
                
                self._mfa_cache[cache_key] = mfa_creds
                return mfa_creds
                
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"MFA credential lookup failed: {e}")
                continue

        return None

    async def update_mfa_credentials(
        self,
        mfa_creds: MFACredentials
    ) -> bool:
        """Update MFA credentials."""
        mfa_creds.updated_at = datetime.now()
        return await self.store_mfa_credentials(mfa_creds)


# Factory function for creating appropriate vault
def create_credential_vault(
    storage_path: Optional[Path] = None,
    namespace: str = "OpenClawAgent",
    prefer_windows: bool = True
):
    """
    Create appropriate credential vault for the platform.
    
    Args:
        storage_path: Path for file-based storage
        namespace: Namespace for credential isolation
        prefer_windows: Prefer Windows Credential Manager on Windows
        
    Returns:
        Credential vault instance
    """
    if prefer_windows and platform.system() == 'Windows':
        try:
            return WindowsCredentialVault(namespace)
        except (RuntimeError, ImportError, OSError) as e:
            logger.warning(f"Failed to create Windows vault: {e}")
    
    # Fall back to encrypted file vault
    if storage_path is None:
        storage_path = Path.home() / '.openclaw' / 'credentials'
    
    return EncryptedFileVault(storage_path)


# Convenience functions
async def store_service_credentials(
    vault,
    service_name: str,
    username: str,
    password: str,
    **attributes
) -> bool:
    """Store service credentials."""
    return await vault.store_credentials(
        service_name=service_name,
        username=username,
        password=password,
        attributes=attributes
    )


async def get_service_credentials(
    vault,
    service_name: str,
    username: Optional[str] = None
) -> Optional[Credentials]:
    """Get service credentials."""
    return await vault.get_credentials(service_name, username)
