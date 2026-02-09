"""
Cookie Jar Management and Persistence System
For Windows 10 OpenClaw AI Agent Framework
"""

import json
import base64
import asyncio
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlparse
import aiofiles

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


@dataclass
class Cookie:
    """
    RFC 6265 compliant cookie representation with AI agent extensions.
    """
    name: str
    value: str
    domain: str
    path: str = "/"
    expires: Optional[datetime] = None
    max_age: Optional[int] = None
    secure: bool = False
    http_only: bool = False
    same_site: Optional[str] = None  # 'Strict', 'Lax', 'None'
    priority: str = "Medium"  # 'Low', 'Medium', 'High'
    
    # AI Agent Extensions
    source_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cookie has expired."""
        if self.expires is None:
            return False
        return datetime.now() > self.expires
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cookie to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key in ['expires', 'created_at', 'last_used']:
            if data[key] is not None:
                if isinstance(data[key], datetime):
                    data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cookie':
        """Create cookie from dictionary."""
        # Convert ISO format strings back to datetime
        for key in ['expires', 'created_at', 'last_used']:
            if data.get(key) is not None:
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])
        return cls(**data)
    
    def to_playwright_format(self) -> Dict[str, Any]:
        """Convert to Playwright cookie format."""
        cookie_dict = {
            'name': self.name,
            'value': self.value,
            'domain': self.domain,
            'path': self.path,
            'secure': self.secure,
            'httpOnly': self.http_only,
        }
        if self.expires:
            cookie_dict['expires'] = int(self.expires.timestamp())
        if self.same_site:
            cookie_dict['sameSite'] = self.same_site
        return cookie_dict
    
    @classmethod
    def from_playwright_format(cls, data: Dict[str, Any]) -> 'Cookie':
        """Create cookie from Playwright format."""
        expires = None
        if data.get('expires'):
            expires = datetime.fromtimestamp(data['expires'])
        
        return cls(
            name=data['name'],
            value=data['value'],
            domain=data['domain'],
            path=data.get('path', '/'),
            expires=expires,
            secure=data.get('secure', False),
            http_only=data.get('httpOnly', False),
            same_site=data.get('sameSite')
        )
    
    def to_selenium_format(self) -> Dict[str, Any]:
        """Convert to Selenium cookie format."""
        cookie_dict = {
            'name': self.name,
            'value': self.value,
            'domain': self.domain,
            'path': self.path,
            'secure': self.secure,
        }
        if self.expires:
            cookie_dict['expiry'] = int(self.expires.timestamp())
        if self.http_only:
            cookie_dict['httpOnly'] = True
        return cookie_dict


@dataclass
class SessionMetadata:
    """Extended session information for AI agent context."""
    session_id: str
    user_agent: str = "OpenClawAgent/1.0"
    ip_address: Optional[str] = None
    login_timestamp: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    authentication_method: str = "unknown"  # 'password', 'oauth', 'sso', 'mfa'
    identity_provider: Optional[str] = None
    session_type: str = "persistent"  # 'persistent', 'session_only'


@dataclass
class DomainCookieJar:
    """Per-domain cookie isolation container."""
    domain: str
    cookies: List[Cookie] = field(default_factory=list)
    session_metadata: Optional[SessionMetadata] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    is_persistent: bool = True
    
    def add_cookie(self, cookie: Cookie) -> None:
        """Add or update cookie in jar."""
        # Remove existing cookie with same name and path
        self.cookies = [
            c for c in self.cookies 
            if not (c.name == cookie.name and c.path == cookie.path)
        ]
        self.cookies.append(cookie)
        self.last_accessed = datetime.now()
    
    def get_cookie(self, name: str, path: str = "/") -> Optional[Cookie]:
        """Get cookie by name and path."""
        for cookie in self.cookies:
            if cookie.name == name and cookie.path == path:
                return cookie
        return None
    
    def remove_cookie(self, name: str, path: str = "/") -> bool:
        """Remove cookie by name and path."""
        original_len = len(self.cookies)
        self.cookies = [
            c for c in self.cookies 
            if not (c.name == name and c.path == path)
        ]
        return len(self.cookies) < original_len
    
    def clear_expired(self) -> int:
        """Remove expired cookies, return count removed."""
        original_len = len(self.cookies)
        self.cookies = [c for c in self.cookies if not c.is_expired()]
        return original_len - len(self.cookies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'domain': self.domain,
            'cookies': [c.to_dict() for c in self.cookies],
            'session_metadata': asdict(self.session_metadata) if self.session_metadata else None,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'is_persistent': self.is_persistent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainCookieJar':
        """Create from dictionary."""
        jar = cls(
            domain=data['domain'],
            cookies=[Cookie.from_dict(c) for c in data.get('cookies', [])],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            is_persistent=data.get('is_persistent', True)
        )
        if data.get('session_metadata'):
            jar.session_metadata = SessionMetadata(**data['session_metadata'])
        return jar


class CookieEncryption:
    """Encryption handler for cookie values."""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self._init_cipher()
    
    def _init_cipher(self) -> None:
        """Initialize encryption cipher."""
        # Use AES-256-GCM for authenticated encryption
        self.cipher = AESGCM(self.master_key)
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt string value."""
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
        """Decrypt string value."""
        encrypted = base64.urlsafe_b64decode(ciphertext.encode('ascii'))
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        plaintext = self.cipher.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')


class CookieJarManager:
    """
    Centralized cookie management with persistence and security.
    """
    
    def __init__(
        self, 
        storage_path: Path,
        master_key: Optional[bytes] = None,
        auto_save: bool = True,
        save_interval: int = 300
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        if master_key is None:
            master_key = self._derive_key_from_system()
        self.encryption = CookieEncryption(master_key)
        
        # Cookie jars storage
        self.cookie_jars: Dict[str, DomainCookieJar] = {}
        self._lock = asyncio.Lock()
        
        # Auto-save settings
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._last_save = datetime.now()
        self._dirty = False
        
        # Load existing jars
        self._load_jars()
    
    def _derive_key_from_system(self) -> bytes:
        """Derive encryption key from system information."""
        # In production, use Windows DPAPI or secure key storage
        # This is a simplified version
        system_info = f"{hashlib.sha256(secrets.token_bytes(32)).hexdigest()}"
        return hashlib.sha256(system_info.encode()).digest()
    
    def _load_jars(self) -> None:
        """Load cookie jars from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for domain, jar_data in data.get('jars', {}).items():
                try:
                    jar = DomainCookieJar.from_dict(jar_data)
                    # Decrypt cookie values
                    for cookie in jar.cookies:
                        try:
                            cookie.value = self.encryption.decrypt(cookie.value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Cookie decryption skipped (may be unencrypted): {e}")
                    self.cookie_jars[domain] = jar
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to load jar for {domain}: {e}")
            
            logger.info(f"Loaded {len(self.cookie_jars)} cookie jars")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cookie jar file: {e}")
        except OSError as e:
            logger.error(f"Failed to load cookie jars: {e}")
    
    async def save_jars(self) -> bool:
        """Save cookie jars to storage."""
        async with self._lock:
            try:
                data = {
                    'version': '2.0',
                    'saved_at': datetime.now().isoformat(),
                    'jars': {}
                }
                
                for domain, jar in self.cookie_jars.items():
                    jar_dict = jar.to_dict()
                    # Encrypt cookie values
                    for cookie_dict in jar_dict['cookies']:
                        cookie_dict['value'] = self.encryption.encrypt(
                            cookie_dict['value']
                        )
                    data['jars'][domain] = jar_dict
                
                # Write atomically
                temp_path = self.storage_path.with_suffix('.tmp')
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
                
                temp_path.replace(self.storage_path)
                self._last_save = datetime.now()
                self._dirty = False
                
                logger.debug(f"Saved {len(self.cookie_jars)} cookie jars")
                return True
                
            except (OSError, PermissionError) as e:
                logger.error(f"Failed to save cookie jars: {e}")
                return False
    
    async def _auto_save_if_needed(self) -> None:
        """Auto-save if interval has passed and changes exist."""
        if not self.auto_save or not self._dirty:
            return
        
        elapsed = (datetime.now() - self._last_save).total_seconds()
        if elapsed >= self.save_interval:
            await self.save_jars()
    
    def _get_domain_key(self, domain: str) -> str:
        """Normalize domain for jar lookup."""
        # Remove leading dot for consistency
        return domain.lstrip('.').lower()
    
    def _get_or_create_jar(self, domain: str) -> DomainCookieJar:
        """Get existing jar or create new one."""
        domain_key = self._get_domain_key(domain)
        
        if domain_key not in self.cookie_jars:
            self.cookie_jars[domain_key] = DomainCookieJar(domain=domain_key)
        
        return self.cookie_jars[domain_key]
    
    def _get_matching_jars(
        self, 
        domain: str, 
        include_subdomains: bool = True
    ) -> List[DomainCookieJar]:
        """Get jars matching domain."""
        domain_key = self._get_domain_key(domain)
        jars = []
        
        # Exact match
        if domain_key in self.cookie_jars:
            jars.append(self.cookie_jars[domain_key])
        
        # Subdomain matches
        if include_subdomains:
            for jar_domain, jar in self.cookie_jars.items():
                if domain_key.endswith('.' + jar_domain) or jar_domain == domain_key:
                    if jar not in jars:
                        jars.append(jar)
        
        return jars
    
    def _cookie_matches_url(
        self, 
        cookie: Cookie, 
        domain: str, 
        path: str
    ) -> bool:
        """Check if cookie matches URL domain and path."""
        # Domain matching
        cookie_domain = cookie.domain.lstrip('.').lower()
        url_domain = domain.lower()
        
        if cookie_domain.startswith('.'):
            # Domain cookie matches subdomains
            if not url_domain.endswith(cookie_domain[1:]):
                return False
        else:
            # Exact domain match required
            if url_domain != cookie_domain:
                return False
        
        # Path matching - cookie path must be prefix of URL path
        if not path.startswith(cookie.path):
            return False
        
        return True
    
    async def get_cookies_for_request(
        self, 
        url: str, 
        include_subdomains: bool = True
    ) -> List[Cookie]:
        """
        Retrieve applicable cookies for a URL request.
        """
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        path = parsed.path or '/'
        is_secure = parsed.scheme == 'https'
        
        applicable_cookies = []
        
        async with self._lock:
            for jar in self._get_matching_jars(domain, include_subdomains):
                for cookie in jar.cookies:
                    # Check if cookie matches URL
                    if not self._cookie_matches_url(cookie, domain, path):
                        continue
                    
                    # Check secure flag
                    if cookie.secure and not is_secure:
                        continue
                    
                    # Check expiration
                    if cookie.is_expired():
                        continue
                    
                    # Update usage stats
                    cookie.last_used = datetime.now()
                    cookie.access_count += 1
                    
                    applicable_cookies.append(cookie)
        
        # Sort by specificity (most specific first)
        applicable_cookies.sort(key=lambda c: (
            len(c.path),
            c.secure == is_secure,
            c.importance_score
        ), reverse=True)
        
        await self._auto_save_if_needed()
        
        return applicable_cookies
    
    async def set_cookie(self, cookie: Cookie, source_url: str) -> bool:
        """
        Store a cookie with validation and security checks.
        """
        # Validate cookie
        if not cookie.name or not cookie.domain:
            logger.warning("Invalid cookie: missing name or domain")
            return False
        
        # Check for secure context requirements
        parsed_source = urlparse(source_url)
        if cookie.secure and parsed_source.scheme != "https":
            logger.warning(f"Rejecting secure cookie over HTTP: {cookie.name}")
            return False
        
        async with self._lock:
            # Store in appropriate jar
            jar = self._get_or_create_jar(cookie.domain)
            jar.add_cookie(cookie)
            self._dirty = True
        
        logger.debug(f"Set cookie: {cookie.name} for {cookie.domain}")
        await self._auto_save_if_needed()
        
        return True
    
    async def set_cookies_from_response(
        self, 
        set_cookie_headers: List[str], 
        source_url: str
    ) -> List[Cookie]:
        """
        Parse and store cookies from Set-Cookie headers.
        """
        cookies = []
        
        for header in set_cookie_headers:
            try:
                cookie = self._parse_set_cookie_header(header, source_url)
                if await self.set_cookie(cookie, source_url):
                    cookies.append(cookie)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse Set-Cookie header: {e}")
        
        return cookies
    
    def _parse_set_cookie_header(
        self, 
        header: str, 
        source_url: str
    ) -> Cookie:
        """Parse Set-Cookie header into Cookie object."""
        parts = header.split(';')
        
        # Parse name=value
        name_value = parts[0].strip()
        if '=' not in name_value:
            raise ValueError("Invalid cookie format")
        
        name, value = name_value.split('=', 1)
        name = name.strip()
        value = value.strip()
        
        # Default domain from source URL
        parsed_url = urlparse(source_url)
        domain = parsed_url.hostname or ''
        
        # Parse attributes
        path = '/'
        expires = None
        max_age = None
        secure = False
        http_only = False
        same_site = None
        
        for part in parts[1:]:
            part = part.strip()
            if '=' in part:
                attr_name, attr_value = part.split('=', 1)
                attr_name = attr_name.strip().lower()
                attr_value = attr_value.strip()
                
                if attr_name == 'domain':
                    domain = attr_value
                elif attr_name == 'path':
                    path = attr_value
                elif attr_name == 'expires':
                    try:
                        # Parse HTTP date format
                        expires = datetime.strptime(
                            attr_value,
                            '%a, %d %b %Y %H:%M:%S GMT'
                        )
                    except ValueError as e:
                        logger.warning(f"Cookie expires date parse failed: {e}")
                elif attr_name == 'max-age':
                    try:
                        max_age = int(attr_value)
                        expires = datetime.now() + timedelta(seconds=max_age)
                    except ValueError as e:
                        logger.warning(f"Cookie max-age parse failed: {e}")
                elif attr_name == 'samesite':
                    same_site = attr_value.capitalize()
            else:
                # Boolean attributes
                attr_name = part.lower()
                if attr_name == 'secure':
                    secure = True
                elif attr_name == 'httponly':
                    http_only = True
        
        return Cookie(
            name=name,
            value=value,
            domain=domain,
            path=path,
            expires=expires,
            max_age=max_age,
            secure=secure,
            http_only=http_only,
            same_site=same_site,
            source_url=source_url
        )
    
    async def delete_cookie(
        self, 
        name: str, 
        domain: str, 
        path: str = "/"
    ) -> bool:
        """Delete a specific cookie."""
        async with self._lock:
            jar = self.cookie_jars.get(self._get_domain_key(domain))
            if jar:
                removed = jar.remove_cookie(name, path)
                if removed:
                    self._dirty = True
                    await self._auto_save_if_needed()
                return removed
        return False
    
    async def clear_domain(self, domain: str) -> int:
        """Clear all cookies for a domain."""
        async with self._lock:
            domain_key = self._get_domain_key(domain)
            if domain_key in self.cookie_jars:
                count = len(self.cookie_jars[domain_key].cookies)
                del self.cookie_jars[domain_key]
                self._dirty = True
                await self._auto_save_if_needed()
                return count
        return 0
    
    async def clear_all(self) -> int:
        """Clear all cookies."""
        async with self._lock:
            count = sum(len(jar.cookies) for jar in self.cookie_jars.values())
            self.cookie_jars.clear()
            self._dirty = True
            await self._auto_save_if_needed()
            return count
    
    async def clear_expired(self) -> int:
        """Remove all expired cookies."""
        async with self._lock:
            total_removed = 0
            for jar in self.cookie_jars.values():
                removed = jar.clear_expired()
                total_removed += removed
            
            if total_removed > 0:
                self._dirty = True
                await self._auto_save_if_needed()
            
            return total_removed
    
    async def export_for_browser(
        self, 
        browser_type: str,
        domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Export cookies in browser-specific format.
        """
        cookies = []
        
        async with self._lock:
            for jar_domain, jar in self.cookie_jars.items():
                if domains and jar_domain not in domains:
                    continue
                
                for cookie in jar.cookies:
                    if cookie.is_expired():
                        continue
                    
                    if browser_type == "playwright":
                        cookies.append(cookie.to_playwright_format())
                    elif browser_type == "selenium":
                        cookies.append(cookie.to_selenium_format())
                    else:
                        cookies.append(cookie.to_dict())
        
        return cookies
    
    async def import_from_browser(
        self,
        browser_cookies: List[Dict[str, Any]],
        browser_type: str,
        source_url: str
    ) -> int:
        """
        Import cookies from browser format.
        """
        count = 0
        
        for cookie_data in browser_cookies:
            try:
                if browser_type == "playwright":
                    cookie = Cookie.from_playwright_format(cookie_data)
                elif browser_type == "selenium":
                    # Convert selenium format
                    cookie = Cookie(
                        name=cookie_data['name'],
                        value=cookie_data['value'],
                        domain=cookie_data['domain'],
                        path=cookie_data.get('path', '/'),
                        expires=datetime.fromtimestamp(cookie_data['expiry']) if 'expiry' in cookie_data else None,
                        secure=cookie_data.get('secure', False),
                        http_only=cookie_data.get('httpOnly', False)
                    )
                else:
                    cookie = Cookie.from_dict(cookie_data)
                
                if await self.set_cookie(cookie, source_url):
                    count += 1
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to import cookie: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cookie jar statistics."""
        total_cookies = sum(len(jar.cookies) for jar in self.cookie_jars.values())
        expired_cookies = sum(
            sum(1 for c in jar.cookies if c.is_expired())
            for jar in self.cookie_jars.values()
        )
        
        return {
            'total_jars': len(self.cookie_jars),
            'total_cookies': total_cookies,
            'expired_cookies': expired_cookies,
            'active_cookies': total_cookies - expired_cookies,
            'domains': list(self.cookie_jars.keys()),
            'storage_path': str(self.storage_path),
            'last_save': self._last_save.isoformat() if self._last_save else None
        }


# Convenience functions for common operations
async def create_cookie_jar(
    storage_path: Path,
    master_key: Optional[bytes] = None
) -> CookieJarManager:
    """Create and initialize cookie jar manager."""
    return CookieJarManager(storage_path, master_key)


async def sync_cookies_to_playwright(
    cookie_jar: CookieJarManager,
    context,
    url: Optional[str] = None,
    domains: Optional[List[str]] = None
) -> int:
    """Sync cookies from jar to Playwright context."""
    if url:
        cookies = await cookie_jar.get_cookies_for_request(url)
        cookie_dicts = [c.to_playwright_format() for c in cookies]
    else:
        cookie_dicts = await cookie_jar.export_for_browser("playwright", domains)
    
    if cookie_dicts:
        await context.add_cookies(cookie_dicts)
    
    return len(cookie_dicts)


async def sync_cookies_from_playwright(
    cookie_jar: CookieJarManager,
    context,
    source_url: str
) -> int:
    """Sync cookies from Playwright context to jar."""
    browser_cookies = await context.cookies()
    return await cookie_jar.import_from_browser(
        browser_cookies, 
        "playwright", 
        source_url
    )
