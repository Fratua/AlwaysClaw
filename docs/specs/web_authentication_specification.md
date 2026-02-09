# Web Authentication and Session Management System
## Technical Specification for Windows 10 OpenClaw AI Agent Framework

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Cookie Jar Management and Persistence](#3-cookie-jar-management-and-persistence)
4. [Session Storage and Restoration](#4-session-storage-and-restoration)
5. [OAuth 2.0 Flow Automation](#5-oauth-20-flow-automation)
6. [JWT Token Handling](#6-jwt-token-handling)
7. [Form-Based Authentication Automation](#7-form-based-authentication-automation)
8. [Multi-Factor Authentication Handling](#8-multi-factor-authentication-handling)
9. [Credential Vault Integration](#9-credential-vault-integration)
10. [SSO (Single Sign-On) Support](#10-sso-single-sign-on-support)
11. [Security Best Practices](#11-security-best-practices)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Executive Summary

This document provides a comprehensive technical specification for the web authentication and session management system designed for the Windows 10 OpenClaw-inspired AI agent framework. The system enables secure, automated authentication across web services including Gmail, OAuth providers, enterprise SSO systems, and MFA-protected applications.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Cookie Management** | Persistent cookie jar with encryption, domain isolation, and automatic expiration handling |
| **Session Management** | Full session state capture, restoration, and cross-session persistence |
| **OAuth Automation** | Automated OAuth 2.0 flows including Authorization Code, Client Credentials, Device Code, and PKCE |
| **JWT Handling** | Complete JWT lifecycle management with RS256 signing, rotation, and revocation |
| **Form Authentication** | Intelligent form detection, credential auto-fill, and CAPTCHA handling |
| **MFA Support** | TOTP generation, SMS/Email OTP handling, and push notification automation |
| **Credential Vault** | Windows DPAPI integration with secure key storage and encryption |
| **SSO Integration** | SAML 2.0, OpenID Connect, and enterprise federation support |

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Agent Core (GPT-5.2)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Authentication Orchestrator Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Cookie    │  │   Session   │  │   OAuth     │  │   Form      │        │
│  │   Manager   │  │   Manager   │  │   Handler   │  │   Handler   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    JWT      │  │    MFA      │  │  Credential │  │    SSO      │        │
│  │   Handler   │  │   Handler   │  │    Vault    │  │   Handler   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Browser Automation Layer                               │
│         (Playwright / Selenium / CDP Integration)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Secure Storage Layer                                 │
│     (Windows Credential Manager / DPAPI / Encrypted File Store)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Authentication Flow                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. AGENT REQUEST
   ┌─────────┐
   │  Agent  │──► "Access Gmail inbox"
   └─────────┘
        │
        ▼
2. ORCHESTRATOR CHECKS SESSION
   ┌─────────────────┐
   │ Check: Valid    │──► YES ──► Use existing session
   │ session exists? │
   └─────────────────┘──► NO
                            │
                            ▼
3. CREDENTIAL RETRIEVAL
   ┌─────────────────┐
   │ Credential Vault│──► Retrieve Gmail credentials
   │   (DPAPI)       │
   └─────────────────┘
        │
        ▼
4. AUTHENTICATION EXECUTION
   ┌─────────────────────────────────────────────────────────┐
   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
   │  │ Navigate to │───►│ Fill form / │───►│ Handle MFA  │ │
   │  │ login page  │    │ OAuth flow  │    │ if required │ │
   │  └─────────────┘    └─────────────┘    └─────────────┘ │
   └─────────────────────────────────────────────────────────┘
        │
        ▼
5. SESSION PERSISTENCE
   ┌─────────────────┐    ┌─────────────────┐
   │  Cookie Jar     │    │  Session State  │
   │  (Encrypted)    │    │  (Snapshot)     │
   └─────────────────┘    └─────────────────┘
        │
        ▼
6. ACCESS GRANTED
   ┌─────────┐
   │  Agent  │──► "Inbox accessed successfully"
   └─────────┘
```

---

## 3. Cookie Jar Management and Persistence

### 3.1 Architecture

The Cookie Jar Management system provides secure, persistent storage of HTTP cookies with domain isolation, automatic expiration handling, and encryption.

```python
# Core Cookie Jar Architecture

class CookieJarManager:
    """
    Centralized cookie management with persistence and security.
    """
    
    def __init__(self, config: CookieConfig):
        self.storage_path = config.storage_path
        self.encryption_key = self._derive_key(config.master_password)
        self.cookie_jars: Dict[str, DomainCookieJar] = {}
        self._load_jars()
    
    class DomainCookieJar:
        """
        Per-domain cookie isolation container.
        """
        def __init__(self, domain: str):
            self.domain = domain
            self.cookies: List[Cookie] = []
            self.session_metadata: SessionMetadata
            self.last_accessed: datetime
            self.is_persistent: bool = True
```

### 3.2 Cookie Structure

```python
@dataclass
class Cookie:
    """
    RFC 6265 compliant cookie representation.
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
    importance_score: float = 0.0  # AI-calculated importance

@dataclass
class SessionMetadata:
    """
    Extended session information for AI agent context.
    """
    session_id: str
    user_agent: str
    ip_address: Optional[str]
    login_timestamp: datetime
    last_activity: datetime
    authentication_method: str  # 'password', 'oauth', 'sso', 'mfa'
    identity_provider: Optional[str]
    session_type: str  # 'persistent', 'session_only'
```

### 3.3 Storage Format

```json
{
  "version": "2.0",
  "encrypted": true,
  "cipher": "AES-256-GCM",
  "kdf": "PBKDF2-SHA256",
  "kdf_iterations": 100000,
  "jars": {
    "gmail.com": {
      "domain": "gmail.com",
      "created_at": "2025-01-15T10:30:00Z",
      "last_accessed": "2025-01-15T14:45:00Z",
      "cookies": [
        {
          "name": "SID",
          "value": "<encrypted>",
          "domain": ".google.com",
          "path": "/",
          "expires": "2025-07-15T10:30:00Z",
          "secure": true,
          "http_only": true,
          "same_site": "Lax",
          "importance_score": 0.95
        }
      ],
      "session_metadata": {
        "session_id": "sess_abc123",
        "authentication_method": "oauth",
        "identity_provider": "google",
        "login_timestamp": "2025-01-15T10:30:00Z"
      }
    }
  }
}
```

### 3.4 Cookie Jar Operations

```python
class CookieJarManager:
    
    async def get_cookies_for_request(
        self, 
        url: str, 
        include_subdomains: bool = True
    ) -> List[Cookie]:
        """
        Retrieve applicable cookies for a URL request.
        """
        parsed = urlparse(url)
        domain = parsed.hostname
        path = parsed.path
        
        applicable_cookies = []
        for jar in self._get_matching_jars(domain, include_subdomains):
            for cookie in jar.cookies:
                if self._cookie_matches_url(cookie, domain, path):
                    if not self._is_expired(cookie):
                        cookie.last_used = datetime.now()
                        cookie.access_count += 1
                        applicable_cookies.append(cookie)
                    else:
                        await self._remove_cookie(jar, cookie)
        
        # Sort by specificity (most specific first)
        applicable_cookies.sort(key=lambda c: (
            len(c.path),
            c.secure == (parsed.scheme == "https"),
            c.importance_score
        ), reverse=True)
        
        return applicable_cookies
    
    async def set_cookie(self, cookie: Cookie, source_url: str) -> bool:
        """
        Store a cookie with validation and security checks.
        """
        # Validate cookie
        if not self._validate_cookie(cookie):
            return False
        
        # Check for secure context requirements
        if cookie.secure and not source_url.startswith("https"):
            logger.warning(f"Rejecting secure cookie over HTTP: {cookie.name}")
            return False
        
        # Store in appropriate jar
        jar = self._get_or_create_jar(cookie.domain)
        
        # Update or add cookie
        existing = self._find_cookie(jar, cookie.name, cookie.path)
        if existing:
            jar.cookies.remove(existing)
        
        cookie.source_url = source_url
        jar.cookies.append(cookie)
        jar.last_accessed = datetime.now()
        
        await self._persist_jar(jar)
        return True
    
    async def export_for_browser(
        self, 
        browser_type: str,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export cookies in browser-specific format.
        """
        if browser_type == "playwright":
            return self._export_playwright_format(domains)
        elif browser_type == "selenium":
            return self._export_selenium_format(domains)
        elif browser_type == "puppeteer":
            return self._export_puppeteer_format(domains)
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")
```

### 3.5 Integration with Browser Automation

```python
class BrowserCookieBridge:
    """
    Bridges cookie jar with browser automation frameworks.
    """
    
    def __init__(self, cookie_manager: CookieJarManager):
        self.cookie_manager = cookie_manager
    
    async def sync_to_playwright(
        self, 
        context: BrowserContext,
        domains: Optional[List[str]] = None
    ):
        """
        Sync cookies to Playwright browser context.
        """
        cookies = await self.cookie_manager.export_for_browser(
            "playwright", domains
        )
        
        for cookie_data in cookies:
            await context.add_cookies([cookie_data])
    
    async def sync_from_playwright(
        self, 
        context: BrowserContext,
        domains: Optional[List[str]] = None
    ):
        """
        Capture cookies from Playwright browser context.
        """
        cookies = await context.cookies()
        
        for cookie_data in cookies:
            cookie = Cookie.from_playwright_format(cookie_data)
            await self.cookie_manager.set_cookie(
                cookie, 
                source_url=f"https://{cookie.domain}"
            )
    
    async def sync_to_selenium(
        self, 
        driver: WebDriver,
        domains: Optional[List[str]] = None
    ):
        """
        Sync cookies to Selenium WebDriver.
        """
        cookies = await self.cookie_manager.get_cookies_for_domain(domains)
        
        for cookie in cookies:
            # Selenium requires visiting domain before adding cookies
            if not driver.current_url.startswith(f"https://{cookie.domain}"):
                driver.get(f"https://{cookie.domain}")
            
            driver.add_cookie(cookie.to_selenium_format())
```

---

## 4. Session Storage and Restoration

### 4.1 Session State Architecture

```python
@dataclass
class BrowserSession:
    """
    Complete browser session state for restoration.
    """
    session_id: str
    created_at: datetime
    last_saved: datetime
    
    # Cookie state
    cookies: List[Cookie]
    
    # Storage state
    local_storage: Dict[str, Dict[str, str]]  # domain -> key-value
    session_storage: Dict[str, Dict[str, str]]
    
    # IndexedDB state (simplified)
    indexed_db: Dict[str, List[Dict]]  # domain -> records
    
    # Authentication state
    auth_tokens: Dict[str, AuthToken]
    
    # Page state
    current_url: Optional[str]
    page_history: List[str]
    scroll_positions: Dict[str, Dict[str, int]]  # url -> {x, y}
    form_data: Dict[str, Dict[str, Any]]  # url -> form values
    
    # Security context
    permissions: Dict[str, List[str]]  # origin -> permissions
    certificates: Dict[str, Any]  # domain -> cert info

@dataclass
class AuthToken:
    """
    Authentication token with metadata.
    """
    token_type: str  # 'bearer', 'jwt', 'api_key'
    token_value: str
    expires_at: Optional[datetime]
    scopes: List[str]
    issuer: str
    audience: str
    refresh_token: Optional[str]
```

### 4.2 Session Persistence

```python
class SessionManager:
    """
    Manages browser session persistence and restoration.
    """
    
    def __init__(self, config: SessionConfig):
        self.storage = EncryptedSessionStorage(config.storage_path)
        self.active_sessions: Dict[str, BrowserSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
    
    async def capture_session(
        self, 
        browser_context: BrowserContext,
        session_name: str,
        include_storage: bool = True
    ) -> BrowserSession:
        """
        Capture complete browser session state.
        """
        session_id = self._generate_session_id()
        
        # Capture cookies
        cookies = await browser_context.cookies()
        
        # Capture storage if requested
        local_storage = {}
        session_storage = {}
        if include_storage:
            local_storage = await self._capture_local_storage(browser_context)
            session_storage = await self._capture_session_storage(browser_context)
        
        # Capture authentication tokens from page
        auth_tokens = await self._extract_auth_tokens(browser_context)
        
        session = BrowserSession(
            session_id=session_id,
            created_at=datetime.now(),
            last_saved=datetime.now(),
            cookies=[Cookie.from_dict(c) for c in cookies],
            local_storage=local_storage,
            session_storage=session_storage,
            indexed_db={},  # Requires specialized handling
            auth_tokens=auth_tokens,
            current_url=None,  # Set from active page
            page_history=[],
            scroll_positions={},
            form_data={},
            permissions={},
            certificates={}
        )
        
        # Persist session
        await self.storage.save_session(session_name, session)
        self.active_sessions[session_name] = session
        
        return session
    
    async def restore_session(
        self,
        browser_context: BrowserContext,
        session_name: str,
        restore_storage: bool = True
    ) -> bool:
        """
        Restore browser session from storage.
        """
        session = await self.storage.load_session(session_name)
        if not session:
            return False
        
        # Restore cookies
        await browser_context.add_cookies([
            c.to_playwright_format() for c in session.cookies
        ])
        
        # Restore storage
        if restore_storage:
            await self._restore_local_storage(
                browser_context, 
                session.local_storage
            )
            await self._restore_session_storage(
                browser_context, 
                session.session_storage
            )
        
        # Update active sessions
        self.active_sessions[session_name] = session
        
        return True
    
    async def _extract_auth_tokens(
        self, 
        browser_context: BrowserContext
    ) -> Dict[str, AuthToken]:
        """
        Extract authentication tokens from browser storage.
        """
        tokens = {}
        
        # Common token storage locations
        token_keys = [
            'access_token', 'id_token', 'refresh_token',
            'auth_token', 'api_token', 'jwt_token',
            'token', 'bearer_token'
        ]
        
        pages = browser_context.pages
        if pages:
            page = pages[0]
            
            # Check localStorage
            for key in token_keys:
                value = await page.evaluate(f"() => localStorage.getItem('{key}')")
                if value:
                    tokens[key] = self._parse_token(key, value)
            
            # Check sessionStorage
            for key in token_keys:
                value = await page.evaluate(f"() => sessionStorage.getItem('{key}')")
                if value:
                    tokens[key] = self._parse_token(key, value)
        
        return tokens
```

### 4.3 Session Storage Format

```json
{
  "version": "2.0",
  "session_id": "sess_a1b2c3d4e5f6",
  "created_at": "2025-01-15T10:30:00Z",
  "last_saved": "2025-01-15T14:45:00Z",
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_derivation": "Argon2id"
  },
  "cookies": [
    {
      "name": "session_id",
      "value": "<encrypted_base64>",
      "domain": ".example.com",
      "path": "/",
      "secure": true,
      "httpOnly": true,
      "sameSite": "Strict"
    }
  ],
  "local_storage": {
    "example.com": {
      "user_preferences": "<encrypted>",
      "auth_state": "<encrypted>"
    }
  },
  "session_storage": {
    "example.com": {
      "temporary_data": "<encrypted>"
    }
  },
  "auth_tokens": {
    "google_oauth": {
      "token_type": "bearer",
      "access_token": "<encrypted>",
      "refresh_token": "<encrypted>",
      "expires_at": "2025-01-15T11:30:00Z",
      "scopes": ["email", "profile", "gmail.readonly"]
    }
  },
  "page_state": {
    "current_url": "https://mail.google.com/mail/u/0/",
    "scroll_positions": {
      "https://mail.google.com/mail/u/0/": {"x": 0, "y": 500}
    }
  }
}
```

---

## 5. OAuth 2.0 Flow Automation

### 5.1 OAuth Flow Types Supported

```python
from enum import Enum, auto

class OAuthFlowType(Enum):
    """
    Supported OAuth 2.0 flow types.
    """
    AUTHORIZATION_CODE = auto()      # Standard web app flow
    AUTHORIZATION_CODE_PKCE = auto() # Mobile/SPA flow with PKCE
    CLIENT_CREDENTIALS = auto()      # Machine-to-machine
    DEVICE_CODE = auto()             # Device authorization
    IMPLICIT = auto()                # Legacy (not recommended)
    PASSWORD = auto()                # Resource owner (legacy)

class OAuthProviderConfig:
    """
    Configuration for OAuth provider.
    """
    provider_name: str
    client_id: str
    client_secret: Optional[str]  # Not needed for PKCE
    authorization_endpoint: str
    token_endpoint: str
    device_code_endpoint: Optional[str]
    scopes_supported: List[str]
    pkce_supported: bool
    default_scopes: List[str]
```

### 5.2 Authorization Code Flow with PKCE

```python
class AuthorizationCodeFlow:
    """
    Automated Authorization Code flow with PKCE support.
    """
    
    def __init__(
        self, 
        provider: OAuthProviderConfig,
        browser_context: BrowserContext,
        redirect_uri: str = "http://localhost:8080/callback"
    ):
        self.provider = provider
        self.browser = browser_context
        self.redirect_uri = redirect_uri
        self.code_verifier: Optional[str] = None
        self.state: Optional[str] = None
    
    async def authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        scopes: Optional[List[str]] = None
    ) -> OAuthTokens:
        """
        Execute complete OAuth authentication flow.
        """
        # Generate PKCE parameters
        self.code_verifier = self._generate_code_verifier()
        code_challenge = self._generate_code_challenge(self.code_verifier)
        self.state = secrets.token_urlsafe(32)
        
        # Build authorization URL
        auth_url = self._build_authorization_url(
            code_challenge=code_challenge,
            state=self.state,
            scopes=scopes or self.provider.default_scopes
        )
        
        # Start local callback server
        callback_server = CallbackServer(self.redirect_uri)
        await callback_server.start()
        
        try:
            # Navigate to authorization endpoint
            page = await self.browser.new_page()
            await page.goto(auth_url)
            
            # Handle login if credentials provided
            if username and password:
                await self._handle_provider_login(page, username, password)
            
            # Wait for authorization completion
            auth_code = await callback_server.wait_for_code(timeout=300)
            
            # Exchange code for tokens
            tokens = await self._exchange_code(auth_code)
            
            return tokens
            
        finally:
            await callback_server.stop()
            await page.close()
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge."""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    async def _exchange_code(self, auth_code: str) -> OAuthTokens:
        """Exchange authorization code for tokens."""
        token_request = {
            'grant_type': 'authorization_code',
            'client_id': self.provider.client_id,
            'client_secret': self.provider.client_secret,
            'code': auth_code,
            'redirect_uri': self.redirect_uri,
            'code_verifier': self.code_verifier
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.provider.token_endpoint,
                data=token_request
            ) as response:
                token_data = await response.json()
                
                return OAuthTokens(
                    access_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token'),
                    id_token=token_data.get('id_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in', 3600),
                    scope=token_data.get('scope', ''),
                    obtained_at=datetime.now()
                )
```

### 5.3 Client Credentials Flow (M2M)

```python
class ClientCredentialsFlow:
    """
    Machine-to-machine OAuth flow for service accounts.
    """
    
    async def authenticate(
        self,
        scopes: Optional[List[str]] = None
    ) -> OAuthTokens:
        """
        Authenticate using client credentials.
        """
        token_request = {
            'grant_type': 'client_credentials',
            'client_id': self.provider.client_id,
            'client_secret': self.provider.client_secret,
            'scope': ' '.join(scopes or self.provider.default_scopes)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.provider.token_endpoint,
                data=token_request,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise OAuthError(f"Token request failed: {error}")
                
                token_data = await response.json()
                
                return OAuthTokens(
                    access_token=token_data['access_token'],
                    refresh_token=None,  # No refresh token in client_credentials
                    id_token=token_data.get('id_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in', 3600),
                    scope=token_data.get('scope', ''),
                    obtained_at=datetime.now()
                )
```

### 5.4 Device Code Flow

```python
class DeviceCodeFlow:
    """
    Device Authorization Grant flow for headless systems.
    """
    
    async def authenticate(
        self,
        scopes: Optional[List[str]] = None,
        poll_interval: int = 5
    ) -> OAuthTokens:
        """
        Execute device code flow.
        """
        # Step 1: Request device code
        device_code_response = await self._request_device_code(scopes)
        
        # Step 2: Display user code for manual entry
        await self._display_user_code(device_code_response)
        
        # Step 3: Poll for token
        tokens = await self._poll_for_token(
            device_code_response['device_code'],
            device_code_response.get('interval', poll_interval),
            device_code_response['expires_in']
        )
        
        return tokens
    
    async def _request_device_code(
        self, 
        scopes: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Request device and user codes."""
        request_data = {
            'client_id': self.provider.client_id,
            'scope': ' '.join(scopes or self.provider.default_scopes)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.provider.device_code_endpoint,
                data=request_data
            ) as response:
                return await response.json()
    
    async def _poll_for_token(
        self,
        device_code: str,
        interval: int,
        expires_in: int
    ) -> OAuthTokens:
        """Poll token endpoint until authorization complete."""
        token_request = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
            'client_id': self.provider.client_id,
            'device_code': device_code
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < expires_in:
                async with session.post(
                    self.provider.token_endpoint,
                    data=token_request
                ) as response:
                    token_data = await response.json()
                    
                    if 'access_token' in token_data:
                        return OAuthTokens.from_dict(token_data)
                    
                    error = token_data.get('error')
                    if error == 'authorization_pending':
                        await asyncio.sleep(interval)
                        continue
                    elif error == 'slow_down':
                        interval += 5
                        await asyncio.sleep(interval)
                        continue
                    else:
                        raise OAuthError(f"Device code error: {error}")
        
        raise OAuthError("Device code flow timed out")
```

### 5.5 Token Refresh Management

```python
class TokenRefreshManager:
    """
    Manages OAuth token refresh with proactive expiration handling.
    """
    
    def __init__(self):
        self.active_tokens: Dict[str, OAuthTokens] = {}
        self.refresh_tasks: Dict[str, asyncio.Task] = {}
        self.token_observers: List[Callable] = []
    
    async def register_token(
        self, 
        token_id: str, 
        tokens: OAuthTokens,
        provider: OAuthProviderConfig
    ):
        """
        Register token for automatic refresh management.
        """
        self.active_tokens[token_id] = tokens
        
        # Schedule proactive refresh
        refresh_time = tokens.obtained_at + timedelta(
            seconds=tokens.expires_in * 0.8  # Refresh at 80% of lifetime
        )
        
        delay = (refresh_time - datetime.now()).total_seconds()
        if delay > 0:
            self.refresh_tasks[token_id] = asyncio.create_task(
                self._scheduled_refresh(token_id, provider, delay)
            )
    
    async def _scheduled_refresh(
        self,
        token_id: str,
        provider: OAuthProviderConfig,
        delay: float
    ):
        """Schedule and execute token refresh."""
        await asyncio.sleep(delay)
        
        tokens = self.active_tokens.get(token_id)
        if not tokens or not tokens.refresh_token:
            return
        
        try:
            new_tokens = await self._perform_refresh(tokens, provider)
            self.active_tokens[token_id] = new_tokens
            
            # Notify observers
            for observer in self.token_observers:
                await observer(token_id, new_tokens)
            
            # Re-schedule next refresh
            await self.register_token(token_id, new_tokens, provider)
            
        except OAuthError as e:
            logger.error(f"Token refresh failed for {token_id}: {e}")
            # Notify of refresh failure
            for observer in self.token_observers:
                await observer(token_id, None, error=e)
    
    async def _perform_refresh(
        self,
        tokens: OAuthTokens,
        provider: OAuthProviderConfig
    ) -> OAuthTokens:
        """Perform token refresh request."""
        refresh_request = {
            'grant_type': 'refresh_token',
            'client_id': provider.client_id,
            'client_secret': provider.client_secret,
            'refresh_token': tokens.refresh_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                provider.token_endpoint,
                data=refresh_request
            ) as response:
                token_data = await response.json()
                
                return OAuthTokens(
                    access_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token', tokens.refresh_token),
                    id_token=token_data.get('id_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in', 3600),
                    scope=token_data.get('scope', tokens.scope),
                    obtained_at=datetime.now()
                )
```

---

## 6. JWT Token Handling

### 6.1 JWT Architecture

```python
@dataclass
class JWTTokens:
    """
    JWT token pair with metadata.
    """
    access_token: str
    refresh_token: Optional[str]
    id_token: Optional[str]
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str = ""
    obtained_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        expiry = self.obtained_at + timedelta(seconds=self.expires_in)
        return datetime.now() > expiry
    
    @property
    def time_until_expiry(self) -> timedelta:
        """Get time remaining until expiry."""
        expiry = self.obtained_at + timedelta(seconds=self.expires_in)
        return expiry - datetime.now()

class JWTValidator:
    """
    JWT validation with signature verification and claim checking.
    """
    
    def __init__(self, jwks_url: Optional[str] = None):
        self.jwks_url = jwks_url
        self.jwks_cache: Optional[Dict] = None
        self.jwks_cache_time: Optional[datetime] = None
        self.jwks_cache_ttl = timedelta(hours=24)
    
    async def validate_token(
        self,
        token: str,
        expected_audience: Optional[str] = None,
        expected_issuer: Optional[str] = None,
        required_claims: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate JWT token comprehensively.
        """
        try:
            # Get unverified header to find key ID
            unverified_header = jwt.get_unverified_header(token)
            algorithm = unverified_header.get('alg')
            key_id = unverified_header.get('kid')
            
            # Fetch appropriate key
            if algorithm.startswith('RS') or algorithm.startswith('ES'):
                public_key = await self._get_public_key(key_id)
            elif algorithm.startswith('HS'):
                raise JWTError("Symmetric algorithms not supported for validation")
            else:
                raise JWTError(f"Unsupported algorithm: {algorithm}")
            
            # Validate token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                audience=expected_audience,
                issuer=expected_issuer,
                options={
                    'require': required_claims or ['exp', 'iat', 'sub'],
                    'verify_exp': True,
                    'verify_iat': True,
                    'verify_nbf': True
                }
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise JWTError("Token has expired")
        except jwt.InvalidAudienceError:
            raise JWTError("Invalid audience")
        except jwt.InvalidIssuerError:
            raise JWTError("Invalid issuer")
        except jwt.InvalidSignatureError:
            raise JWTError("Invalid signature")
        except jwt.DecodeError:
            raise JWTError("Token could not be decoded")
    
    async def _get_public_key(self, key_id: Optional[str]) -> str:
        """Fetch public key from JWKS endpoint."""
        if not self.jwks_url:
            raise JWTError("JWKS URL not configured")
        
        # Check cache
        if (self.jwks_cache and self.jwks_cache_time and
            datetime.now() - self.jwks_cache_time < self.jwks_cache_ttl):
            jwks = self.jwks_cache
        else:
            # Fetch fresh JWKS
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jwks_url) as response:
                    jwks = await response.json()
                    self.jwks_cache = jwks
                    self.jwks_cache_time = datetime.now()
        
        # Find matching key
        for key in jwks.get('keys', []):
            if key_id is None or key.get('kid') == key_id:
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
        
        raise JWTError(f"Key not found: {key_id}")
```

### 6.2 JWT Token Generation

```python
class JWTGenerator:
    """
    JWT token generation with secure defaults.
    """
    
    def __init__(
        self,
        private_key: str,
        algorithm: str = 'RS256',
        issuer: str = 'openclaw-agent'
    ):
        self.private_key = private_key
        self.algorithm = algorithm
        self.issuer = issuer
    
    def generate_access_token(
        self,
        subject: str,
        audience: str,
        scopes: List[str],
        custom_claims: Optional[Dict] = None,
        expires_in: int = 900  # 15 minutes
    ) -> str:
        """
        Generate access token with standard claims.
        """
        now = datetime.utcnow()
        
        payload = {
            # Registered claims
            'sub': subject,
            'iss': self.issuer,
            'aud': audience,
            'iat': now,
            'nbf': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': secrets.token_urlsafe(16),  # Unique token ID
            
            # Custom claims
            'scope': ' '.join(scopes),
            'type': 'access'
        }
        
        if custom_claims:
            payload.update(custom_claims)
        
        return jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm,
            headers={'kid': self._get_key_id()}
        )
    
    def generate_refresh_token(
        self,
        subject: str,
        family_id: str,
        expires_in: int = 604800  # 7 days
    ) -> str:
        """
        Generate refresh token with rotation support.
        """
        now = datetime.utcnow()
        
        payload = {
            'sub': subject,
            'iss': self.issuer,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': secrets.token_urlsafe(16),
            'family_id': family_id,  # For rotation tracking
            'type': 'refresh'
        }
        
        return jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm
        )
```

### 6.3 Token Rotation and Revocation

```python
class TokenRotationManager:
    """
    Manages refresh token rotation with reuse detection.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool
    
    async def create_token_pair(
        self,
        user_id: str,
        device_info: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Create new access/refresh token pair.
        """
        # Generate tokens
        access_token = self._generate_access_token(user_id)
        refresh_token = self._generate_refresh_token(user_id)
        
        # Create token family for rotation tracking
        family_id = secrets.token_urlsafe(16)
        jti = self._extract_jti(refresh_token)
        
        # Store refresh token metadata
        await self.db.execute("""
            INSERT INTO refresh_tokens 
                (jti, family_id, user_id, device_info, expires_at, revoked)
            VALUES ($1, $2, $3, $4, $5, FALSE)
        """, jti, family_id, user_id, device_info,
            datetime.utcnow() + timedelta(days=7))
        
        return access_token, refresh_token
    
    async def rotate_refresh_token(
        self,
        old_refresh_token: str
    ) -> Tuple[str, str]:
        """
        Rotate refresh token with reuse detection.
        """
        # Decode and validate
        try:
            payload = jwt.decode(
                old_refresh_token,
                self.public_key,
                algorithms=['RS256']
            )
        except jwt.InvalidTokenError:
            raise TokenError("Invalid refresh token")
        
        # Verify it's a refresh token
        if payload.get('type') != 'refresh':
            raise TokenError("Not a refresh token")
        
        old_jti = payload['jti']
        user_id = payload['sub']
        family_id = payload['family_id']
        
        # Check token in database
        token_record = await self.db.fetchrow("""
            SELECT family_id, revoked 
            FROM refresh_tokens 
            WHERE jti = $1 AND user_id = $2
        """, old_jti, user_id)
        
        if not token_record:
            raise TokenError("Refresh token not found")
        
        # SECURITY: Detect reuse
        if token_record['revoked']:
            # Token reuse detected - revoke entire family
            await self._revoke_token_family(family_id)
            raise TokenError(
                "Token reuse detected - all sessions revoked",
                security_event='token_reuse_detected'
            )
        
        # Mark old token as used
        await self.db.execute("""
            UPDATE refresh_tokens 
            SET revoked = TRUE, used_at = $1
            WHERE jti = $2
        """, datetime.utcnow(), old_jti)
        
        # Generate new token pair
        new_access = self._generate_access_token(user_id)
        new_refresh = self._generate_refresh_token(user_id)
        new_jti = self._extract_jti(new_refresh)
        
        # Store new refresh token in same family
        await self.db.execute("""
            INSERT INTO refresh_tokens 
                (jti, family_id, user_id, device_info, expires_at, revoked)
            VALUES ($1, $2, $3, $4, $5, FALSE)
        """, new_jti, family_id, user_id, token_record.get('device_info'),
            datetime.utcnow() + timedelta(days=7))
        
        return new_access, new_refresh
```

### 6.4 Database Schema for Token Management

```sql
-- Refresh token tracking for rotation and revocation
CREATE TABLE refresh_tokens (
    jti VARCHAR(255) PRIMARY KEY,
    family_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    device_info VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    revoked BOOLEAN DEFAULT FALSE,
    
    -- Indexes for efficient lookups
    INDEX idx_user_id (user_id),
    INDEX idx_family_id (family_id),
    INDEX idx_expires_at (expires_at)
);

-- Access token blacklist for immediate revocation
CREATE TABLE token_blacklist (
    jti VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    reason VARCHAR(255),
    blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_expires_at (expires_at)
);

-- Cleanup job for expired tokens
CREATE EVENT cleanup_expired_tokens
ON SCHEDULE EVERY 1 DAY
DO
    DELETE FROM refresh_tokens WHERE expires_at < NOW() - INTERVAL '30 days';
    DELETE FROM token_blacklist WHERE expires_at < NOW();
```

---

## 7. Form-Based Authentication Automation

### 7.1 Form Detection and Analysis

```python
@dataclass
class LoginForm:
    """
    Detected login form structure.
    """
    form_element: ElementHandle
    username_field: Optional[ElementHandle]
    password_field: Optional[ElementHandle]
    submit_button: Optional[ElementHandle]
    captcha_field: Optional[ElementHandle]
    mfa_field: Optional[ElementHandle]
    remember_checkbox: Optional[ElementHandle]
    
    # Form metadata
    action_url: str
    method: str
    has_csrf_token: bool
    csrf_token_field: Optional[str]
    
    # Security indicators
    is_https: bool
    has_password_field: bool
    autocomplete_enabled: bool

class FormDetector:
    """
    Intelligent login form detection.
    """
    
    USERNAME_SELECTORS = [
        'input[type="email"]',
        'input[type="text"][name*="user" i]',
        'input[type="text"][name*="email" i]',
        'input[type="text"][name*="login" i]',
        'input[name="username"]',
        'input[name="email"]',
        'input[name="login"]',
        'input[id*="user" i]',
        'input[id*="email" i]',
        'input[autocomplete="username"]',
        'input[autocomplete="email"]',
    ]
    
    PASSWORD_SELECTORS = [
        'input[type="password"]',
        'input[name*="pass" i]',
        'input[id*="pass" i]',
        'input[autocomplete="current-password"]',
    ]
    
    SUBMIT_SELECTORS = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Sign in")',
        'button:has-text("Log in")',
        'button:has-text("Login")',
        'a:has-text("Sign in")',
    ]
    
    async def detect_login_form(self, page: Page) -> Optional[LoginForm]:
        """
        Detect and analyze login form on page.
        """
        # Look for forms with password fields
        forms = await page.query_selector_all('form')
        
        for form in forms:
            # Check for password field
            password_field = await self._find_field(form, self.PASSWORD_SELECTORS)
            if not password_field:
                continue
            
            # Found potential login form
            username_field = await self._find_field(form, self.USERNAME_SELECTORS)
            submit_button = await self._find_field(form, self.SUBMIT_SELECTORS)
            
            # Get form attributes
            action = await form.get_attribute('action') or ''
            method = await form.get_attribute('method') or 'GET'
            
            # Check for CSRF token
            csrf_field = await form.query_selector(
                'input[name*="csrf" i], input[name*="token" i]'
            )
            
            return LoginForm(
                form_element=form,
                username_field=username_field,
                password_field=password_field,
                submit_button=submit_button,
                captcha_field=None,  # Detect separately
                mfa_field=None,  # Detect separately
                remember_checkbox=None,  # Detect separately
                action_url=action,
                method=method.upper(),
                has_csrf_token=csrf_field is not None,
                csrf_token_field=await csrf_field.get_attribute('name') if csrf_field else None,
                is_https=page.url.startswith('https'),
                has_password_field=True,
                autocomplete_enabled=await self._check_autocomplete(form)
            )
        
        return None
    
    async def _find_field(
        self, 
        form: ElementHandle, 
        selectors: List[str]
    ) -> Optional[ElementHandle]:
        """Find field matching any selector."""
        for selector in selectors:
            field = await form.query_selector(selector)
            if field and await field.is_visible():
                return field
        return None
```

### 7.2 Automated Form Filling

```python
class FormAuthenticator:
    """
    Automated form-based authentication.
    """
    
    def __init__(
        self,
        credential_vault: CredentialVault,
        mfa_handler: MFAHandler,
        captcha_solver: Optional[CaptchaSolver] = None
    ):
        self.vault = credential_vault
        self.mfa = mfa_handler
        self.captcha = captcha_solver
        self.form_detector = FormDetector()
    
    async def authenticate(
        self,
        page: Page,
        service_name: str,
        username: Optional[str] = None,
        wait_for_mfa: bool = True,
        timeout: int = 60
    ) -> AuthenticationResult:
        """
        Perform automated form authentication.
        """
        start_time = time.time()
        
        # Get credentials
        if not username:
            creds = await self.vault.get_default_credentials(service_name)
        else:
            creds = await self.vault.get_credentials(service_name, username)
        
        if not creds:
            return AuthenticationResult(
                success=False,
                error="Credentials not found"
            )
        
        # Detect login form
        login_form = await self.form_detector.detect_login_form(page)
        if not login_form:
            return AuthenticationResult(
                success=False,
                error="Login form not detected"
            )
        
        # Fill username
        if login_form.username_field:
            await login_form.username_field.fill(creds.username)
            await asyncio.sleep(random.uniform(0.1, 0.3))  # Human-like delay
        
        # Fill password
        await login_form.password_field.fill(creds.password)
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Handle CAPTCHA if present
        captcha_result = await self._handle_captcha(page, login_form)
        if captcha_result and not captcha_result.success:
            return AuthenticationResult(
                success=False,
                error=f"CAPTCHA handling failed: {captcha_result.error}"
            )
        
        # Submit form
        if login_form.submit_button:
            await login_form.submit_button.click()
        else:
            await login_form.form_element.evaluate('form => form.submit()')
        
        # Wait for navigation or MFA
        try:
            await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
        except:
            pass
        
        # Check for MFA requirement
        if await self._is_mfa_page(page):
            if not wait_for_mfa:
                return AuthenticationResult(
                    success=False,
                    error="MFA required but not handled",
                    mfa_required=True
                )
            
            mfa_result = await self.mfa.handle_mfa(page, service_name, creds)
            if not mfa_result.success:
                return AuthenticationResult(
                    success=False,
                    error=f"MFA handling failed: {mfa_result.error}",
                    mfa_required=True
                )
        
        # Verify successful login
        if await self._verify_login_success(page):
            return AuthenticationResult(
                success=True,
                session_cookies=await page.context.cookies(),
                login_time=time.time() - start_time
            )
        else:
            return AuthenticationResult(
                success=False,
                error="Login verification failed"
            )
    
    async def _handle_captcha(
        self,
        page: Page,
        login_form: LoginForm
    ) -> Optional[CaptchaResult]:
        """Handle CAPTCHA if present."""
        if not self.captcha:
            return None
        
        # Detect CAPTCHA
        captcha_types = await self._detect_captcha_type(page)
        
        for captcha_type in captcha_types:
            result = await self.captcha.solve(page, captcha_type)
            if result.success:
                # Fill CAPTCHA field
                captcha_field = await page.query_selector(
                    'input[name*="captcha" i], input[id*="captcha" i]'
                )
                if captcha_field:
                    await captcha_field.fill(result.solution)
                return result
        
        return None
```

### 7.3 Login Verification

```python
class LoginVerifier:
    """
    Verifies successful login through multiple indicators.
    """
    
    SUCCESS_INDICATORS = [
        # URL patterns
        '/dashboard', '/home', '/welcome', '/account',
        '/profile', '/main', '/app',
        
        # Page elements
        'text="Welcome"', 'text="Dashboard"',
        'text="My Account"', 'text="Logout"',
        'text="Sign out"',
        
        # UI elements
        '.user-menu', '.account-dropdown', '.profile-image',
        '[data-testid="user-menu"]',
    ]
    
    FAILURE_INDICATORS = [
        # Error messages
        'text="Invalid"', 'text="Incorrect"',
        'text="Failed"', 'text="Error"',
        'text="Wrong password"',
        
        # Error elements
        '.error', '.alert-error', '.form-error',
        '[role="alert"]',
    ]
    
    async def verify_login(self, page: Page) -> Tuple[bool, Optional[str]]:
        """
        Verify if login was successful.
        """
        # Check for failure indicators first
        for indicator in self.FAILURE_INDICATORS:
            try:
                element = await page.wait_for_selector(
                    indicator, 
                    timeout=1000,
                    state='visible'
                )
                if element:
                    error_text = await element.text_content()
                    return False, error_text or "Login failed"
            except:
                continue
        
        # Check for success indicators
        for indicator in self.SUCCESS_INDICATORS:
            try:
                element = await page.wait_for_selector(
                    indicator,
                    timeout=2000,
                    state='visible'
                )
                if element:
                    return True, None
            except:
                continue
        
        # Check for session cookies
        cookies = await page.context.cookies()
        session_cookies = [c for c in cookies if self._is_session_cookie(c)]
        
        if session_cookies:
            return True, None
        
        return False, "Could not verify login status"
    
    def _is_session_cookie(self, cookie: Dict) -> bool:
        """Check if cookie is likely a session cookie."""
        session_names = ['session', 'sid', 'auth', 'token', 'login']
        return any(name in cookie['name'].lower() for name in session_names)
```

---

## 8. Multi-Factor Authentication Handling

### 8.1 MFA Architecture

```python
from enum import Enum, auto

class MFAType(Enum):
    """
    Supported MFA types.
    """
    TOTP = auto()           # Time-based One-Time Password
    HOTP = auto()           # HMAC-based One-Time Password
    SMS = auto()            # SMS-delivered code
    EMAIL = auto()          # Email-delivered code
    PUSH = auto()           # Push notification
    WEBAUTHN = auto()       # FIDO2/WebAuthn
    SECURITY_KEY = auto()   # Hardware security key
    BACKUP_CODE = auto()    # Recovery/backup codes

@dataclass
class MFACredentials:
    """
    MFA credentials for a service.
    """
    service_name: str
    username: str
    mfa_type: MFAType
    
    # TOTP/HOTP
    totp_secret: Optional[str] = None
    
    # SMS/Email
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    
    # Backup codes
    backup_codes: List[str] = field(default_factory=list)
    used_backup_codes: List[str] = field(default_factory=list)
    
    # WebAuthn
    webauthn_credential_id: Optional[str] = None

class MFAHandler:
    """
    Multi-factor authentication handler.
    """
    
    def __init__(
        self,
        credential_vault: CredentialVault,
        sms_handler: Optional[SMSHandler] = None,
        email_handler: Optional[EmailHandler] = None,
        push_handler: Optional[PushNotificationHandler] = None
    ):
        self.vault = credential_vault
        self.sms = sms_handler
        self.email = email_handler
        self.push = push_handler
        self.totp_generator = TOTPGenerator()
    
    async def handle_mfa(
        self,
        page: Page,
        service_name: str,
        credentials: Credentials
    ) -> MFAResult:
        """
        Handle MFA challenge automatically.
        """
        # Detect MFA type
        mfa_type = await self._detect_mfa_type(page)
        
        # Get MFA credentials
        mfa_creds = await self.vault.get_mfa_credentials(
            service_name, 
            credentials.username
        )
        
        if not mfa_creds:
            return MFAResult(
                success=False,
                error="MFA credentials not found"
            )
        
        # Handle based on type
        handlers = {
            MFAType.TOTP: self._handle_totp,
            MFAType.SMS: self._handle_sms,
            MFAType.EMAIL: self._handle_email,
            MFAType.PUSH: self._handle_push,
            MFAType.BACKUP_CODE: self._handle_backup_code,
        }
        
        handler = handlers.get(mfa_type)
        if not handler:
            return MFAResult(
                success=False,
                error=f"Unsupported MFA type: {mfa_type}"
            )
        
        return await handler(page, mfa_creds)
```

### 8.2 TOTP Automation

```python
import pyotp
import time

class TOTPGenerator:
    """
    TOTP code generation for MFA automation.
    """
    
    def generate_code(self, secret: str, offset: int = 0) -> str:
        """
        Generate TOTP code from secret.
        """
        # Clean secret (remove spaces)
        secret = secret.replace(' ', '').upper()
        
        # Create TOTP object
        totp = pyotp.TOTP(secret)
        
        # Generate code with optional time offset
        if offset != 0:
            current_time = time.time() + offset
            return totp.at(current_time)
        
        return totp.now()
    
    def get_remaining_seconds(self, secret: str) -> int:
        """
        Get seconds remaining in current TOTP window.
        """
        totp = pyotp.TOTP(secret)
        return totp.interval - (int(time.time()) % totp.interval)
    
    def should_wait_for_next(self, secret: str, threshold: int = 5) -> bool:
        """
        Check if we should wait for next code window.
        """
        remaining = self.get_remaining_seconds(secret)
        return remaining < threshold

class TOTPMFAHandler:
    """
    Automated TOTP MFA handling.
    """
    
    TOTP_INPUT_SELECTORS = [
        'input[name*="totp" i]',
        'input[name*="code" i]',
        'input[name*="otp" i]',
        'input[name*="token" i]',
        'input[autocomplete="one-time-code"]',
        'input[type="number"][maxlength="6"]',
    ]
    
    async def handle(
        self,
        page: Page,
        mfa_creds: MFACredentials
    ) -> MFAResult:
        """
        Handle TOTP MFA challenge.
        """
        if not mfa_creds.totp_secret:
            return MFAResult(
                success=False,
                error="TOTP secret not configured"
            )
        
        # Wait for TOTP input field
        code_input = await self._find_code_input(page)
        if not code_input:
            return MFAResult(
                success=False,
                error="TOTP input field not found"
            )
        
        # Generate TOTP code
        generator = TOTPGenerator()
        
        # Check if we should wait for next window
        if generator.should_wait_for_next(mfa_creds.totp_secret):
            wait_time = generator.get_remaining_seconds(mfa_creds.totp_secret)
            logger.info(f"Waiting {wait_time}s for next TOTP window")
            await asyncio.sleep(wait_time + 1)
        
        code = generator.generate_code(mfa_creds.totp_secret)
        
        # Fill code
        await code_input.fill(code)
        
        # Submit
        await self._submit_code(page)
        
        # Wait for verification
        success = await self._wait_for_verification(page)
        
        if success:
            return MFAResult(success=True)
        else:
            return MFAResult(
                success=False,
                error="TOTP verification failed"
            )
```

### 8.3 SMS/Email OTP Handling

```python
class SMSOTPHandler:
    """
    Automated SMS OTP handling via Twilio.
    """
    
    def __init__(self, twilio_client: TwilioClient):
        self.twilio = twilio_client
        self.message_cache: Dict[str, List[Dict]] = {}
    
    async def handle(
        self,
        page: Page,
        mfa_creds: MFACredentials,
        timeout: int = 60
    ) -> MFAResult:
        """
        Handle SMS OTP challenge.
        """
        if not mfa_creds.phone_number:
            return MFAResult(
                success=False,
                error="Phone number not configured"
            )
        
        # Record message timestamp before requesting code
        before_time = datetime.now()
        
        # Trigger SMS code (usually done by clicking "Send code")
        send_button = await page.query_selector(
            'button:has-text("Send code"), button:has-text("Text me")'
        )
        if send_button:
            await send_button.click()
        
        # Wait for SMS
        code = await self._wait_for_sms_code(
            mfa_creds.phone_number,
            before_time,
            timeout
        )
        
        if not code:
            return MFAResult(
                success=False,
                error="SMS code not received within timeout"
            )
        
        # Fill code
        code_input = await self._find_code_input(page)
        if code_input:
            await code_input.fill(code)
            await self._submit_code(page)
        
        # Verify
        success = await self._wait_for_verification(page)
        
        return MFAResult(success=success)
    
    async def _wait_for_sms_code(
        self,
        phone_number: str,
        after_time: datetime,
        timeout: int
    ) -> Optional[str]:
        """
        Wait for and extract SMS code.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Fetch messages from Twilio
            messages = await self.twilio.get_messages(
                to=phone_number,
                after=after_time
            )
            
            for message in messages:
                # Extract code using regex patterns
                code = self._extract_code_from_text(message.body)
                if code:
                    return code
            
            await asyncio.sleep(2)
        
        return None
    
    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """
        Extract OTP code from message text.
        """
        # Common OTP patterns
        patterns = [
            r'\b(\d{6})\b',  # 6-digit code
            r'code[\s:is]+(\d+)',  # "code is 123456"
            r'code[\s:is]+([a-zA-Z0-9]+)',  # alphanumeric
            r'OTP[\s:is]+(\d+)',  # "OTP is 123456"
            r'verification[\s:]+(\d+)',  # "verification: 123456"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
```

---

## 9. Credential Vault Integration

### 9.1 Windows Credential Manager Integration

```python
import win32cred
import win32crypt
import ctypes
from ctypes import wintypes

class WindowsCredentialVault:
    """
    Windows Credential Manager integration using DPAPI.
    """
    
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    CRED_PERSIST_ENTERPRISE = 3
    
    def __init__(self, namespace: str = "OpenClawAgent"):
        self.namespace = namespace
        self._cache: Dict[str, Credentials] = {}
    
    def store_credentials(
        self,
        service_name: str,
        username: str,
        password: str,
        attributes: Optional[Dict] = None
    ) -> bool:
        """
        Store credentials in Windows Credential Manager.
        """
        target_name = f"{self.namespace}:{service_name}:{username}"
        
        # Prepare credential data
        credential = {
            'Type': self.CRED_TYPE_GENERIC,
            'TargetName': target_name,
            'UserName': username,
            'CredentialBlob': password.encode('utf-16-le'),
            'Persist': self.CRED_PERSIST_LOCAL_MACHINE,
            'Comment': f'OpenClaw Agent credentials for {service_name}',
        }
        
        if attributes:
            credential['Attribute'] = self._encode_attributes(attributes)
        
        try:
            win32cred.CredWrite(credential, 0)
            
            # Update cache
            self._cache[target_name] = Credentials(
                service_name=service_name,
                username=username,
                password=password,
                attributes=attributes
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store credentials: {e}")
            return False
    
    def get_credentials(
        self,
        service_name: str,
        username: Optional[str] = None
    ) -> Optional[Credentials]:
        """
        Retrieve credentials from Windows Credential Manager.
        """
        if username:
            target_name = f"{self.namespace}:{service_name}:{username}"
            
            # Check cache first
            if target_name in self._cache:
                return self._cache[target_name]
            
            try:
                credential = win32cred.CredRead(target_name, self.CRED_TYPE_GENERIC, 0)
                
                password = credential['CredentialBlob'].decode('utf-16-le')
                attributes = self._decode_attributes(credential.get('Attribute', []))
                
                creds = Credentials(
                    service_name=service_name,
                    username=credential['UserName'],
                    password=password,
                    attributes=attributes
                )
                
                # Update cache
                self._cache[target_name] = creds
                
                return creds
                
            except Exception as e:
                logger.error(f"Failed to retrieve credentials: {e}")
                return None
        else:
            # Enumerate all credentials for service
            return self._enumerate_service_credentials(service_name)
    
    def delete_credentials(
        self,
        service_name: str,
        username: str
    ) -> bool:
        """
        Delete credentials from Windows Credential Manager.
        """
        target_name = f"{self.namespace}:{service_name}:{username}"
        
        try:
            win32cred.CredDelete(target_name, self.CRED_TYPE_GENERIC, 0)
            
            # Remove from cache
            self._cache.pop(target_name, None)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False
```

### 9.2 Encrypted File Store

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class EncryptedCredentialStore:
    """
    Encrypted file-based credential storage.
    """
    
    def __init__(self, storage_path: Path, master_password: str):
        self.storage_path = storage_path
        self.cipher = self._create_cipher(master_password)
    
    def _create_cipher(self, master_password: str) -> Fernet:
        """Create Fernet cipher from master password."""
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._get_or_create_salt(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        return Fernet(key)
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create salt for key derivation."""
        salt_path = self.storage_path / '.salt'
        
        if salt_path.exists():
            return salt_path.read_bytes()
        else:
            salt = os.urandom(16)
            salt_path.write_bytes(salt)
            return salt
    
    async def store(
        self,
        service_name: str,
        credentials: Credentials
    ) -> bool:
        """
        Store encrypted credentials.
        """
        # Serialize credentials
        cred_data = {
            'service_name': credentials.service_name,
            'username': credentials.username,
            'password': credentials.password,
            'attributes': credentials.attributes,
            'created_at': credentials.created_at.isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        
        # Encrypt
        json_data = json.dumps(cred_data).encode()
        encrypted = self.cipher.encrypt(json_data)
        
        # Store
        cred_file = self.storage_path / f"{service_name}.enc"
        cred_file.write_bytes(encrypted)
        
        return True
    
    async def retrieve(self, service_name: str) -> Optional[Credentials]:
        """
        Retrieve and decrypt credentials.
        """
        cred_file = self.storage_path / f"{service_name}.enc"
        
        if not cred_file.exists():
            return None
        
        try:
            encrypted = cred_file.read_bytes()
            decrypted = self.cipher.decrypt(encrypted)
            cred_data = json.loads(decrypted.decode())
            
            return Credentials(
                service_name=cred_data['service_name'],
                username=cred_data['username'],
                password=cred_data['password'],
                attributes=cred_data.get('attributes', {}),
                created_at=datetime.fromisoformat(cred_data['created_at']),
            )
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None
```

### 9.3 Master Key Management

```python
class MasterKeyManager:
    """
    Manages master encryption key using Windows DPAPI.
    """
    
    def __init__(self):
        self.key_file = Path(os.environ['LOCALAPPDATA']) / 'OpenClaw' / 'master.key'
        self._master_key: Optional[bytes] = None
    
    def get_or_create_master_key(self) -> bytes:
        """
        Get existing or create new master key.
        """
        if self._master_key:
            return self._master_key
        
        if self.key_file.exists():
            # Decrypt existing key
            encrypted_key = self.key_file.read_bytes()
            self._master_key = self._unprotect_data(encrypted_key)
        else:
            # Generate new key
            self._master_key = Fernet.generate_key()
            
            # Encrypt and store
            encrypted_key = self._protect_data(self._master_key)
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_bytes(encrypted_key)
        
        return self._master_key
    
    def _protect_data(self, data: bytes) -> bytes:
        """Encrypt data using DPAPI."""
        # CRYPTPROTECT_LOCAL_MACHINE = 0x4
        # CRYPTPROTECT_UI_FORBIDDEN = 0x1
        flags = 0x1
        
        blob_in = ctypes.create_string_buffer(data)
        blob_out = ctypes.POINTER(DATA_BLOB)()
        
        if not CryptProtectData(
            ctypes.byref(blob_in),
            None,  # Description
            None,  # Optional entropy
            None,  # Reserved
            None,  # Prompt struct
            flags,
            ctypes.byref(blob_out)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        
        encrypted = ctypes.string_at(
            blob_out.contents.pbData, 
            blob_out.contents.cbData
        )
        LocalFree(blob_out.contents.pbData)
        
        return encrypted
    
    def _unprotect_data(self, encrypted: bytes) -> bytes:
        """Decrypt data using DPAPI."""
        flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
        
        blob_in = DATA_BLOB(len(encrypted), ctypes.cast(encrypted, ctypes.POINTER(ctypes.c_byte)))
        blob_out = ctypes.POINTER(DATA_BLOB)()
        
        if not CryptUnprotectData(
            ctypes.byref(blob_in),
            None,  # Description
            None,  # Optional entropy
            None,  # Reserved
            None,  # Prompt struct
            flags,
            ctypes.byref(blob_out)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        
        decrypted = ctypes.string_at(
            blob_out.contents.pbData,
            blob_out.contents.cbData
        )
        LocalFree(blob_out.contents.pbData)
        
        return decrypted
```

---

## 10. SSO (Single Sign-On) Support

### 10.1 SAML 2.0 Implementation

```python
import xml.etree.ElementTree as ET
from signxml import XMLSigner, XMLVerifier

class SAMLHandler:
    """
    SAML 2.0 authentication handler.
    """
    
    def __init__(
        self,
        idp_metadata_url: str,
        sp_entity_id: str,
        acs_url: str
    ):
        self.idp_metadata = self._load_idp_metadata(idp_metadata_url)
        self.sp_entity_id = sp_entity_id
        self.acs_url = acs_url
        self.private_key = self._load_sp_private_key()
        self.public_cert = self._load_sp_public_cert()
    
    def generate_authn_request(self) -> str:
        """
        Generate SAML Authentication Request.
        """
        request_id = f"_{uuid.uuid4()}"
        issue_instant = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        authn_request = f"""<?xml version="1.0" encoding="UTF-8"?>
        <samlp:AuthnRequest 
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{issue_instant}"
            Destination="{self.idp_metadata['sso_url']}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            AssertionConsumerServiceURL="{self.acs_url}">
            <saml:Issuer>{self.sp_entity_id}</saml:Issuer>
            <samlp:NameIDPolicy 
                Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                AllowCreate="true"/>
            <samlp:RequestedAuthnContext Comparison="exact">
                <saml:AuthnContextClassRef>
                    urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport
                </saml:AuthnContextClassRef>
            </samlp:RequestedAuthnContext>
        </samlp:AuthnRequest>"""
        
        # Sign the request
        signed_request = XMLSigner(
            signature_algorithm='rsa-sha256',
            digest_algorithm='sha256'
        ).sign(
            ET.fromstring(authn_request),
            key=self.private_key,
            cert=self.public_cert
        )
        
        # Base64 encode
        return base64.b64encode(ET.tostring(signed_request)).decode()
    
    async def handle_saml_response(
        self,
        saml_response: str,
        relay_state: Optional[str] = None
    ) -> SAMLAuthenticationResult:
        """
        Handle and validate SAML Response.
        """
        # Decode response
        decoded_response = base64.b64decode(saml_response)
        
        # Parse XML
        root = ET.fromstring(decoded_response)
        
        # Verify signature
        try:
            verified = XMLVerifier().verify(
                root,
                x509_cert=self.idp_metadata['signing_cert']
            )
        except Exception as e:
            return SAMLAuthenticationResult(
                success=False,
                error=f"Signature verification failed: {e}"
            )
        
        # Extract assertion
        assertion = self._extract_assertion(root)
        
        # Validate assertion
        validation_result = self._validate_assertion(assertion)
        if not validation_result.valid:
            return SAMLAuthenticationResult(
                success=False,
                error=validation_result.error
            )
        
        # Extract user attributes
        attributes = self._extract_attributes(assertion)
        
        return SAMLAuthenticationResult(
            success=True,
            user_id=attributes.get('name_id'),
            email=attributes.get('email'),
            attributes=attributes,
            session_index=attributes.get('session_index')
        )
    
    def _validate_assertion(self, assertion: ET.Element) -> ValidationResult:
        """
        Validate SAML assertion.
        """
        conditions = assertion.find('.//saml:Conditions', self.NSMAP)
        
        if conditions is not None:
            # Check NotBefore
            not_before = conditions.get('NotBefore')
            if not_before:
                not_before_time = datetime.fromisoformat(not_before.replace('Z', '+00:00'))
                if datetime.utcnow() < not_before_time:
                    return ValidationResult(
                        valid=False,
                        error="Assertion not yet valid"
                    )
            
            # Check NotOnOrAfter
            not_on_or_after = conditions.get('NotOnOrAfter')
            if not_on_or_after:
                expiry_time = datetime.fromisoformat(not_on_or_after.replace('Z', '+00:00'))
                if datetime.utcnow() >= expiry_time:
                    return ValidationResult(
                        valid=False,
                        error="Assertion expired"
                    )
            
            # Check Audience
            audience_restriction = conditions.find('.//saml:AudienceRestriction', self.NSMAP)
            if audience_restriction is not None:
                audience = audience_restriction.find('.//saml:Audience', self.NSMAP)
                if audience is not None and audience.text != self.sp_entity_id:
                    return ValidationResult(
                        valid=False,
                        error="Invalid audience"
                    )
        
        return ValidationResult(valid=True)
```

### 10.2 OpenID Connect Implementation

```python
class OIDCHandler:
    """
    OpenID Connect authentication handler.
    """
    
    def __init__(
        self,
        provider_config: OIDCProviderConfig,
        browser_context: BrowserContext
    ):
        self.config = provider_config
        self.browser = browser_context
        self.redirect_uri = "http://localhost:8080/callback"
    
    async def authenticate(
        self,
        scopes: List[str] = None,
        prompt: str = 'login'
    ) -> OIDCTokens:
        """
        Execute OIDC authentication flow.
        """
        # Generate state and nonce
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        
        # Build authorization URL
        auth_url = self._build_authorization_url(
            state=state,
            nonce=nonce,
            scopes=scopes or ['openid', 'profile', 'email'],
            prompt=prompt
        )
        
        # Start callback server
        callback_server = CallbackServer(self.redirect_uri)
        await callback_server.start()
        
        try:
            # Navigate to authorization endpoint
            page = await self.browser.new_page()
            await page.goto(auth_url)
            
            # Wait for authentication completion
            auth_response = await callback_server.wait_for_response(timeout=300)
            
            # Verify state
            if auth_response['state'] != state:
                raise OIDCError("State mismatch - possible CSRF attack")
            
            # Exchange code for tokens
            tokens = await self._exchange_code(auth_response['code'])
            
            # Validate ID token
            id_token_claims = await self._validate_id_token(
                tokens.id_token,
                nonce=nonce
            )
            
            return OIDCTokens(
                access_token=tokens.access_token,
                id_token=tokens.id_token,
                refresh_token=tokens.refresh_token,
                id_token_claims=id_token_claims
            )
            
        finally:
            await callback_server.stop()
            await page.close()
    
    async def _validate_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate OIDC ID Token.
        """
        # Fetch JWKS
        jwks = await self._fetch_jwks()
        
        # Get unverified header
        header = jwt.get_unverified_header(id_token)
        
        # Find matching key
        key = self._find_jwks_key(jwks, header.get('kid'))
        
        # Decode and validate
        claims = jwt.decode(
            id_token,
            key,
            algorithms=['RS256'],
            audience=self.config.client_id,
            issuer=self.config.issuer
        )
        
        # Verify nonce
        if nonce and claims.get('nonce') != nonce:
            raise OIDCError("Nonce mismatch")
        
        # Verify expiration
        if datetime.utcnow().timestamp() > claims.get('exp', 0):
            raise OIDCError("ID token expired")
        
        return claims
```

### 10.3 SSO Session Management

```python
class SSOSessionManager:
    """
    Manages SSO sessions across multiple identity providers.
    """
    
    def __init__(self, storage: EncryptedSessionStorage):
        self.storage = storage
        self.active_sessions: Dict[str, SSOSession] = {}
    
    async def create_session(
        self,
        idp_name: str,
        user_id: str,
        saml_assertion: Optional[str] = None,
        oidc_tokens: Optional[OIDCTokens] = None
    ) -> SSOSession:
        """
        Create new SSO session.
        """
        session_id = secrets.token_urlsafe(32)
        
        session = SSOSession(
            session_id=session_id,
            idp_name=idp_name,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=8),
            saml_assertion=saml_assertion,
            oidc_tokens=oidc_tokens,
            service_sessions: Dict[str, ServiceSession] = {}
        )
        
        await self.storage.save_sso_session(session)
        self.active_sessions[session_id] = session
        
        return session
    
    async def get_service_token(
        self,
        session_id: str,
        service_name: str,
        requested_scopes: List[str]
    ) -> Optional[ServiceToken]:
        """
        Get or create service-specific token via SSO.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            session = await self.storage.load_sso_session(session_id)
            if not session:
                return None
        
        # Check if service session exists
        service_session = session.service_sessions.get(service_name)
        if service_session:
            # Check if existing token covers requested scopes
            if self._scopes_sufficient(service_session.scopes, requested_scopes):
                if not service_session.is_expired:
                    return service_session.token
        
        # Request new token from IdP
        new_token = await self._request_service_token(
            session,
            service_name,
            requested_scopes
        )
        
        if new_token:
            session.service_sessions[service_name] = ServiceSession(
                service_name=service_name,
                token=new_token,
                scopes=requested_scopes,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=new_token.expires_in)
            )
            await self.storage.save_sso_session(session)
        
        return new_token
```

---

## 11. Security Best Practices

### 11.1 Secure Credential Storage

| Practice | Implementation |
|----------|----------------|
| **Encryption at Rest** | AES-256-GCM for all stored credentials |
| **Key Management** | Windows DPAPI for master key protection |
| **Memory Protection** | SecureString for credentials in memory |
| **Audit Logging** | All credential access logged with timestamps |
| **Access Control** | Role-based access to credential vault |

### 11.2 Session Security

| Practice | Implementation |
|----------|----------------|
| **Token Expiration** | Short-lived access tokens (15 min) |
| **Refresh Rotation** | Refresh token rotation on each use |
| **Reuse Detection** | Automatic revocation on token reuse |
| **Secure Transport** | TLS 1.3 for all authentication traffic |
| **Cookie Flags** | Secure, HttpOnly, SameSite=Strict |

### 11.3 MFA Security

| Practice | Implementation |
|----------|----------------|
| **TOTP Storage** | Encrypted secrets, never plaintext |
| **Backup Codes** | Single-use, hashed storage |
| **Rate Limiting** | Max 3 MFA attempts per 5 minutes |
| **Device Binding** | Optional device-specific MFA |

---

## 12. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Cookie jar implementation with encryption
- [ ] Session storage and restoration
- [ ] Windows Credential Manager integration
- [ ] Basic form authentication

### Phase 2: OAuth & JWT (Weeks 3-4)
- [ ] OAuth 2.0 flow automation
- [ ] JWT handling and validation
- [ ] Token refresh management
- [ ] Provider configurations (Google, Microsoft, etc.)

### Phase 3: MFA Support (Weeks 5-6)
- [ ] TOTP automation
- [ ] SMS/Email OTP handling
- [ ] Backup code support
- [ ] MFA credential management

### Phase 4: SSO Integration (Weeks 7-8)
- [ ] SAML 2.0 implementation
- [ ] OpenID Connect support
- [ ] Enterprise IdP integration
- [ ] SSO session management

### Phase 5: Advanced Features (Weeks 9-10)
- [ ] CAPTCHA solving integration
- [ ] Advanced form detection
- [ ] Multi-domain session sync
- [ ] Security monitoring and alerting

---

## Appendix A: Configuration Schema

```yaml
# authentication_config.yaml

version: "2.0"

# Cookie Jar Settings
cookie_jar:
  storage_path: "~/.openclaw/cookies"
  encryption:
    algorithm: "AES-256-GCM"
    key_derivation: "Argon2id"
  persistence:
    auto_save: true
    save_interval: 300  # seconds
  cleanup:
    expired_cookie_ttl: 86400  # 24 hours
    max_cookies_per_domain: 100

# Session Management
session:
  storage_path: "~/.openclaw/sessions"
  default_ttl: 28800  # 8 hours
  include_storage: true
  include_history: false

# OAuth Configuration
oauth:
  providers:
    google:
      client_id: "${GOOGLE_CLIENT_ID}"
      client_secret: "${GOOGLE_CLIENT_SECRET}"
      scopes: ["openid", "email", "profile"]
      redirect_uri: "http://localhost:8080/callback"
    
    microsoft:
      client_id: "${MICROSOFT_CLIENT_ID}"
      client_secret: "${MICROSOFT_CLIENT_SECRET}"
      tenant: "common"
      scopes: ["openid", "email", "profile", "User.Read"]

# JWT Settings
jwt:
  algorithm: "RS256"
  access_token_ttl: 900  # 15 minutes
  refresh_token_ttl: 604800  # 7 days
  rotation:
    enabled: true
    reuse_detection: true

# MFA Configuration
mfa:
  totp:
    code_length: 6
    time_window: 30
    allowed_drift: 1
  
  sms:
    provider: "twilio"
    timeout: 60
  
  backup_codes:
    count: 10
    length: 8

# Credential Vault
vault:
  type: "windows_credential_manager"  # or "encrypted_file"
  namespace: "OpenClawAgent"
  master_key_protection: "DPAPI"

# SSO Configuration
sso:
  saml:
    sp_entity_id: "openclaw-agent"
    acs_url: "http://localhost:8080/saml/acs"
    
  oidc:
    default_scopes: ["openid", "profile", "email"]
```

---

## Appendix B: API Reference

### CookieJarManager

```python
class CookieJarManager:
    async def get_cookies_for_request(url: str) -> List[Cookie]
    async def set_cookie(cookie: Cookie, source_url: str) -> bool
    async def export_for_browser(browser_type: str) -> Dict
    async def clear_domain(domain: str) -> bool
    async def persist() -> bool
```

### SessionManager

```python
class SessionManager:
    async def capture_session(browser_context: BrowserContext, name: str) -> BrowserSession
    async def restore_session(browser_context: BrowserContext, name: str) -> bool
    async def delete_session(name: str) -> bool
    async def list_sessions() -> List[str]
```

### OAuthHandler

```python
class OAuthHandler:
    async def authenticate(flow_type: OAuthFlowType, **kwargs) -> OAuthTokens
    async def refresh_token(refresh_token: str) -> OAuthTokens
    async def revoke_token(token: str) -> bool
```

### MFAHandler

```python
class MFAHandler:
    async def handle_mfa(page: Page, service_name: str, creds: Credentials) -> MFAResult
    async def generate_totp(secret: str) -> str
    async def validate_backup_code(service_name: str, code: str) -> bool
```

---

*Document Version: 1.0*
*Last Updated: January 2025*
*Author: AI Agent Authentication Team*
