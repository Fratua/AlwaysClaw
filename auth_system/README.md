# Web Authentication and Session Management System

## For Windows 10 OpenClaw AI Agent Framework

This comprehensive authentication system provides enterprise-grade web authentication capabilities for AI agents, including cookie management, OAuth flows, JWT handling, MFA support, and SSO integration.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Components](#components)
6. [Usage Examples](#usage-examples)
7. [Security](#security)
8. [Configuration](#configuration)

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Cookie Management** | Persistent, encrypted cookie jar with domain isolation |
| **Session Management** | Full browser session capture and restoration |
| **OAuth 2.0** | Complete OAuth flow automation (Authorization Code, Client Credentials, Device Code, PKCE) |
| **JWT Handling** | Token validation, generation, rotation, and revocation |
| **Form Authentication** | Intelligent form detection and automated login |
| **MFA Support** | TOTP, SMS, Email OTP, and backup codes |
| **Credential Vault** | Windows Credential Manager integration with DPAPI encryption |
| **SSO Support** | SAML 2.0 and OpenID Connect |

### Supported Providers

- **OAuth/OIDC**: Google, Microsoft/Azure AD, GitHub, Slack, Okta, Auth0
- **SAML**: Azure AD, Okta, OneLogin, custom IdPs
- **MFA**: Google Authenticator, Authy, SMS via Twilio

---

## Installation

```bash
# Install required dependencies
pip install pyjwt cryptography aiohttp playwright pyotp

# For Windows Credential Manager support (Windows only)
pip install pywin32

# For SAML support
pip install signxml
```

---

## Quick Start

```python
import asyncio
from auth_system import create_auth_orchestrator

async def main():
    # Create authentication orchestrator
    auth = create_auth_orchestrator()
    
    # Store credentials
    await auth.store_credentials(
        service_name="gmail",
        username="user@example.com",
        password="secure_password"
    )
    
    # Authenticate using form
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://gmail.com")
        
        result = await auth.authenticate_with_form(
            page=page,
            service_name="gmail"
        )
        
        if result.success:
            print("Authentication successful!")
            # Cookies are automatically synced to cookie jar
        else:
            print(f"Authentication failed: {result.error}")
        
        await browser.close()

asyncio.run(main())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Authentication Orchestrator                      │
│                    (Main Interface)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Cookie Jar   │    │   OAuth      │    │   Form Auth  │
│ Manager      │    │   Handler    │    │   Handler    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Credential  │    │    JWT       │    │    MFA       │
│    Vault     │    │   Handler    │    │   Handler    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │     SSO      │
                    │   Handler    │
                    └──────────────┘
```

---

## Components

### 1. Cookie Jar Manager

Persistent, encrypted cookie storage with domain isolation.

```python
from auth_system import CookieJarManager

# Create cookie jar
cookie_jar = CookieJarManager(
    storage_path=Path("~/.openclaw/cookies.enc")
)

# Get cookies for URL
cookies = await cookie_jar.get_cookies_for_request("https://example.com")

# Set cookie
from auth_system import Cookie
cookie = Cookie(
    name="session_id",
    value="abc123",
    domain="example.com",
    secure=True,
    http_only=True
)
await cookie_jar.set_cookie(cookie, "https://example.com")
```

### 2. OAuth Handler

Complete OAuth 2.0 flow automation.

```python
from auth_system import OAuthHandler

oauth = OAuthHandler()

# Register provider
oauth.register_provider_from_template(
    name="google",
    template="google",
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# Authenticate
tokens = await oauth.authenticate(
    provider_name="google",
    scopes=["email", "profile", "gmail.readonly"]
)

print(f"Access token: {tokens.access_token}")
```

### 3. JWT Handler

JWT validation, generation, and management.

```python
from auth_system import JWTValidator, JWTGenerator

# Validate JWT
validator = JWTValidator(jwks_url="https://auth.example.com/.well-known/jwks.json")
decoded = await validator.validate_token(
    token="eyJhbGciOiJSUzI1NiIs...",
    expected_audience="my-app",
    expected_issuer="auth.example.com"
)

# Generate JWT
from auth_system import generate_rsa_keypair
private_key, public_key = generate_rsa_keypair()

generator = JWTGenerator(private_key)
access_token = generator.generate_access_token(
    subject="user123",
    audience="my-api",
    scopes=["read", "write"]
)
```

### 4. MFA Handler

Multi-factor authentication automation.

```python
from auth_system import MFAHandler, generate_totp_secret

# Store MFA credentials
await auth.store_mfa_credentials(
    service_name="github",
    username="myuser",
    mfa_type="totp",
    totp_secret=generate_totp_secret()
)

# Handle MFA challenge
result = await auth.handle_mfa(
    page=page,
    service_name="github",
    username="myuser"
)

# Generate TOTP code manually
code = await auth.generate_totp_code("github", "myuser")
print(f"TOTP Code: {code}")
```

### 5. Credential Vault

Secure credential storage with Windows Credential Manager integration.

```python
from auth_system import create_credential_vault

# Create vault (uses Windows Credential Manager on Windows)
vault = create_credential_vault(
    namespace="OpenClawAgent"
)

# Store credentials
await vault.store_credentials(
    service_name="aws",
    username="AKIAIOSFODNN7EXAMPLE",
    password="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
)

# Retrieve credentials
creds = await vault.get_credentials("aws")
print(f"Username: {creds.username}")
```

### 6. SSO Handler

SAML 2.0 and OpenID Connect support.

```python
from auth_system import SSOHandler

sso = SSOHandler()

# Register OIDC provider
sso.register_from_template(
    name="azure_ad",
    template="azure_ad",
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant="your-tenant-id"
)

# Authenticate
result = await sso.authenticate_oidc(
    provider_name="azure_ad",
    browser_context=context,
    scopes=["User.Read", "Mail.Read"]
)

if result.success:
    print(f"User: {result.user_id}")
    print(f"Email: {result.email}")
```

---

## Usage Examples

### Example 1: Gmail OAuth Authentication

```python
import asyncio
from playwright.async_api import async_playwright
from auth_system import create_auth_orchestrator

async def authenticate_gmail():
    auth = create_auth_orchestrator()
    
    # Register Google OAuth
    auth.register_oauth_provider(
        name="google",
        template="google",
        client_id="your-google-client-id",
        client_secret="your-google-client-secret"
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        
        # Authenticate
        tokens = await auth.authenticate_with_oauth(
            provider_name="google",
            browser_context=context,
            scopes=[
                "openid",
                "email",
                "profile",
                "https://www.googleapis.com/auth/gmail.readonly"
            ]
        )
        
        print(f"Authenticated! Access token: {tokens['access_token'][:20]}...")
        
        await browser.close()

asyncio.run(authenticate_gmail())
```

### Example 2: Form-Based Authentication with MFA

```python
import asyncio
from playwright.async_api import async_playwright
from auth_system import create_auth_orchestrator

async def authenticate_with_mfa():
    auth = create_auth_orchestrator()
    
    # Store credentials
    await auth.store_credentials(
        service_name="github",
        username="myuser",
        password="mypassword"
    )
    
    # Store MFA credentials
    await auth.store_mfa_credentials(
        service_name="github",
        username="myuser",
        mfa_type="totp",
        totp_secret="JBSWY3DPEHPK3PXP"  # Base32 encoded secret
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://github.com/login")
        
        # Authenticate (MFA handled automatically)
        result = await auth.authenticate_with_form(
            page=page,
            service_name="github",
            wait_for_mfa=True
        )
        
        if result.success:
            print("Successfully authenticated with MFA!")
            # Now you can navigate to protected pages
            await page.goto("https://github.com/settings/profile")
        
        await browser.close()

asyncio.run(authenticate_with_mfa())
```

### Example 3: Session Persistence

```python
import asyncio
from auth_system import create_auth_orchestrator

async def persistent_session():
    auth = create_auth_orchestrator()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        
        # Sync existing cookies before navigation
        await auth.sync_cookies_to_browser(
            browser_context=context,
            domains=["google.com", "gmail.com"]
        )
        
        page = await context.new_page()
        await page.goto("https://gmail.com")
        
        # Check if already authenticated
        if await auth.form_authenticator.is_logged_in(page, "gmail"):
            print("Already authenticated!")
        else:
            print("Need to authenticate...")
            # Perform authentication
            result = await auth.authenticate_with_form(page, "gmail")
        
        # Cookies automatically saved to jar
        await browser.close()

asyncio.run(persistent_session())
```

### Example 4: Enterprise SSO

```python
import asyncio
from auth_system import create_auth_orchestrator

async def enterprise_sso():
    auth = create_auth_orchestrator()
    
    # Register Azure AD
    auth.register_oidc_provider(
        name="company_azure",
        template="azure_ad",
        client_id="your-app-id",
        client_secret="your-app-secret",
        tenant="your-tenant-id"
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        
        # Authenticate via SSO
        result = await auth.authenticate_with_sso(
            provider_name="company_azure",
            protocol="oidc",
            browser_context=context,
            scopes=["User.Read", "Mail.Read", "Calendars.Read"]
        )
        
        if result.success:
            print(f"SSO Authenticated: {result.user_id}")
            print(f"Email: {result.email}")
            
            # Use access token for API calls
            access_token = result.access_token
            # Make API calls to Microsoft Graph...
        
        await browser.close()

asyncio.run(enterprise_sso())
```

---

## Security

### Encryption

- **Cookie Storage**: AES-256-GCM with unique nonces
- **Credential Storage**: Windows DPAPI (Windows) or AES-256-GCM (cross-platform)
- **Session Storage**: Encrypted with master key derived from system entropy

### Best Practices

1. **Never hardcode credentials** - Always use the credential vault
2. **Use short-lived tokens** - Access tokens expire in 15 minutes by default
3. **Enable token rotation** - Refresh tokens are rotated on each use
4. **Validate all tokens** - Always verify signatures and claims
5. **Use HTTPS only** - Secure flag enforced for all cookies

### Token Lifetimes

| Token Type | Default Lifetime | Configurable |
|------------|------------------|--------------|
| Access Token | 15 minutes | Yes |
| Refresh Token | 7 days | Yes |
| ID Token | 1 hour | Yes |
| Session Cookie | 8 hours | Yes |

---

## Configuration

### Environment Variables

```bash
# OAuth credentials
export GOOGLE_CLIENT_ID="your-client-id"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export MICROSOFT_CLIENT_ID="your-client-id"
export MICROSOFT_CLIENT_SECRET="your-client-secret"

# Twilio for SMS MFA
export TWILIO_ACCOUNT_SID="your-account-sid"
export TWILIO_AUTH_TOKEN="your-auth-token"

# Storage paths
export OPENCLAW_COOKIE_PATH="~/.openclaw/cookies.enc"
export OPENCLAW_CREDENTIAL_PATH="~/.openclaw/credentials"
```

### Configuration File

```yaml
# config/auth_config.yaml
version: "1.0"

storage:
  cookie_path: "~/.openclaw/cookies.enc"
  credential_path: "~/.openclaw/credentials"
  session_path: "~/.openclaw/sessions"

security:
  encryption_algorithm: "AES-256-GCM"
  key_derivation: "PBKDF2-SHA256"
  key_iterations: 100000

tokens:
  access_token_ttl: 900  # 15 minutes
  refresh_token_ttl: 604800  # 7 days
  rotation_enabled: true

oauth:
  providers:
    google:
      client_id: "${GOOGLE_CLIENT_ID}"
      client_secret: "${GOOGLE_CLIENT_SECRET}"
      scopes: ["openid", "email", "profile"]
    
    microsoft:
      client_id: "${MICROSOFT_CLIENT_ID}"
      client_secret: "${MICROSOFT_CLIENT_SECRET}"
      tenant: "common"

mfa:
  totp:
    digits: 6
    interval: 30
    allowed_drift: 1
  
  sms:
    provider: "twilio"
    timeout: 60
```

---

## API Reference

See the individual module docstrings for detailed API documentation:

- `cookie_jar.py` - Cookie management
- `oauth_handler.py` - OAuth 2.0 flows
- `jwt_handler.py` - JWT operations
- `mfa_handler.py` - MFA handling
- `credential_vault.py` - Secure storage
- `form_auth.py` - Form authentication
- `sso_handler.py` - SSO integration
- `auth_orchestrator.py` - Main interface

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.

---

## Support

For issues and questions:
- GitHub Issues: [github.com/openclaw/auth-system/issues](https://github.com/openclaw/auth-system/issues)
- Documentation: [docs.openclaw.dev](https://docs.openclaw.dev)
