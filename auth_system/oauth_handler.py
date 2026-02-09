"""
OAuth 2.0 Flow Automation System
For Windows 10 OpenClaw AI Agent Framework
"""

import os
import json
import base64
import hashlib
import secrets
import asyncio
import logging
import aiohttp
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from urllib.parse import urlencode, parse_qs, urlparse

logger = logging.getLogger(__name__)


class OAuthFlowType(Enum):
    """Supported OAuth 2.0 flow types."""
    AUTHORIZATION_CODE = auto()
    AUTHORIZATION_CODE_PKCE = auto()
    CLIENT_CREDENTIALS = auto()
    DEVICE_CODE = auto()
    IMPLICIT = auto()  # Legacy, not recommended
    PASSWORD = auto()  # Legacy, not recommended


@dataclass
class OAuthProviderConfig:
    """Configuration for OAuth provider."""
    provider_name: str
    client_id: str
    client_secret: Optional[str] = None
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    device_code_endpoint: Optional[str] = None
    revocation_endpoint: Optional[str] = None
    introspection_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    scopes_supported: List[str] = field(default_factory=list)
    pkce_supported: bool = True
    default_scopes: List[str] = field(default_factory=lambda: ["openid"])
    
    # Additional provider-specific settings
    additional_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class OAuthTokens:
    """OAuth token response with metadata."""
    access_token: str
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str = ""
    obtained_at: datetime = field(default_factory=datetime.now)
    
    # Additional provider-specific data
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
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
    
    @property
    def should_refresh(self, threshold_ratio: float = 0.8) -> bool:
        """Check if token should be refreshed (based on threshold)."""
        elapsed = (datetime.now() - self.obtained_at).total_seconds()
        return elapsed >= (self.expires_in * threshold_ratio)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'id_token': self.id_token,
            'token_type': self.token_type,
            'expires_in': self.expires_in,
            'scope': self.scope,
            'obtained_at': self.obtained_at.isoformat(),
            'extra_data': self.extra_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuthTokens':
        """Create from dictionary."""
        return cls(
            access_token=data['access_token'],
            refresh_token=data.get('refresh_token'),
            id_token=data.get('id_token'),
            token_type=data.get('token_type', 'Bearer'),
            expires_in=data.get('expires_in', 3600),
            scope=data.get('scope', ''),
            obtained_at=datetime.fromisoformat(data['obtained_at']),
            extra_data=data.get('extra_data', {})
        )


class OAuthError(Exception):
    """OAuth-related error."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class CallbackServer:
    """Simple HTTP server to receive OAuth callbacks."""

    def __init__(self, host: str = None, port: int = 8080):
        self.host = host or os.environ.get('OAUTH_CALLBACK_HOST', 'localhost')
        self.port = port
        self.redirect_uri = f"http://{self.host}:{port}/callback"
        self._server = None
        self._response_data: Optional[Dict[str, Any]] = None
        self._response_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the callback server."""
        from aiohttp import web
        
        async def handle_callback(request):
            """Handle OAuth callback."""
            # Get query parameters
            params = dict(request.query)
            
            # Also check for POST data
            if request.method == 'POST':
                post_data = await request.post()
                params.update(post_data)
            
            self._response_data = params
            self._response_event.set()
            
            # Return success page
            return web.Response(
                text="""
                <html>
                    <body>
                        <h1>Authentication Successful</h1>
                        <p>You can close this window.</p>
                        <script>window.close();</script>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
        
        app = web.Application()
        app.router.add_get('/callback', handle_callback)
        app.router.add_post('/callback', handle_callback)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self._server = runner
        logger.debug(f"Callback server started on {self.redirect_uri}")
    
    async def stop(self) -> None:
        """Stop the callback server."""
        if self._server:
            await self._server.cleanup()
            self._server = None
            logger.debug("Callback server stopped")
    
    async def wait_for_response(self, timeout: int = 300) -> Dict[str, Any]:
        """Wait for OAuth callback response."""
        try:
            await asyncio.wait_for(
                self._response_event.wait(),
                timeout=timeout
            )
            return self._response_data or {}
        except asyncio.TimeoutError:
            raise OAuthError("Timeout waiting for OAuth callback")


class AuthorizationCodeFlow:
    """OAuth 2.0 Authorization Code flow with PKCE support."""
    
    def __init__(
        self,
        provider: OAuthProviderConfig,
        redirect_uri: str = None
    ):
        if redirect_uri is None:
            callback_host = os.environ.get('OAUTH_CALLBACK_HOST', 'localhost')
            redirect_uri = f"http://{callback_host}:8080/callback"
        self.provider = provider
        self.redirect_uri = redirect_uri
        self.code_verifier: Optional[str] = None
        self.state: Optional[str] = None
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier (43-128 chars)."""
        # Generate 32 bytes of random data, base64url encode
        return base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge (S256 method)."""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def _generate_state(self) -> str:
        """Generate random state parameter for CSRF protection."""
        return secrets.token_urlsafe(32)
    
    def build_authorization_url(
        self,
        scopes: Optional[List[str]] = None,
        use_pkce: bool = True,
        additional_params: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str, Optional[str]]:
        """
        Build authorization URL.
        Returns: (auth_url, state, code_verifier)
        """
        self.state = self._generate_state()
        
        params = {
            'response_type': 'code',
            'client_id': self.provider.client_id,
            'redirect_uri': self.redirect_uri,
            'state': self.state,
            'scope': ' '.join(scopes or self.provider.default_scopes)
        }
        
        # Add PKCE parameters if enabled
        if use_pkce and self.provider.pkce_supported:
            self.code_verifier = self._generate_code_verifier()
            params['code_challenge'] = self._generate_code_challenge(self.code_verifier)
            params['code_challenge_method'] = 'S256'
        
        # Add additional provider-specific params
        params.update(self.provider.additional_params)
        
        if additional_params:
            params.update(additional_params)
        
        auth_url = f"{self.provider.authorization_endpoint}?{urlencode(params)}"
        
        return auth_url, self.state, self.code_verifier
    
    async def exchange_code(
        self,
        auth_code: str,
        expected_state: str,
        received_state: str
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens.
        """
        # Verify state to prevent CSRF
        if received_state != expected_state:
            raise OAuthError("State mismatch - possible CSRF attack")
        
        # Build token request
        token_request = {
            'grant_type': 'authorization_code',
            'client_id': self.provider.client_id,
            'code': auth_code,
            'redirect_uri': self.redirect_uri
        }
        
        # Add client secret if available
        if self.provider.client_secret:
            token_request['client_secret'] = self.provider.client_secret
        
        # Add PKCE verifier if used
        if self.code_verifier:
            token_request['code_verifier'] = self.code_verifier
        
        # Make token request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.provider.token_endpoint,
                data=token_request,
                headers={'Accept': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OAuthError(f"Token request failed: {error_text}")
                
                token_data = await response.json()
                
                if 'error' in token_data:
                    raise OAuthError(
                        f"OAuth error: {token_data['error']}",
                        error_code=token_data['error']
                    )
                
                return OAuthTokens(
                    access_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token'),
                    id_token=token_data.get('id_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in', 3600),
                    scope=token_data.get('scope', ''),
                    extra_data={k: v for k, v in token_data.items() 
                               if k not in ['access_token', 'refresh_token', 
                                          'id_token', 'token_type', 
                                          'expires_in', 'scope']}
                )


class ClientCredentialsFlow:
    """OAuth 2.0 Client Credentials flow for machine-to-machine auth."""
    
    def __init__(self, provider: OAuthProviderConfig):
        self.provider = provider
    
    async def authenticate(
        self,
        scopes: Optional[List[str]] = None
    ) -> OAuthTokens:
        """
        Authenticate using client credentials.
        """
        if not self.provider.client_secret:
            raise OAuthError("Client secret required for client credentials flow")
        
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
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OAuthError(f"Token request failed: {error_text}")
                
                token_data = await response.json()
                
                return OAuthTokens(
                    access_token=token_data['access_token'],
                    refresh_token=None,  # No refresh token in client_credentials
                    id_token=token_data.get('id_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in', 3600),
                    scope=token_data.get('scope', ''),
                    extra_data={k: v for k, v in token_data.items() 
                               if k not in ['access_token', 'refresh_token', 
                                          'id_token', 'token_type', 
                                          'expires_in', 'scope']}
                )


class DeviceCodeFlow:
    """OAuth 2.0 Device Authorization Grant flow."""
    
    def __init__(self, provider: OAuthProviderConfig):
        self.provider = provider
    
    async def initiate(self, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initiate device code flow.
        Returns device code response with user_code and verification_uri.
        """
        if not self.provider.device_code_endpoint:
            raise OAuthError("Device code endpoint not configured")
        
        request_data = {
            'client_id': self.provider.client_id,
            'scope': ' '.join(scopes or self.provider.default_scopes)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.provider.device_code_endpoint,
                data=request_data,
                headers={'Accept': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OAuthError(f"Device code request failed: {error_text}")
                
                return await response.json()
    
    async def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        expires_in: int = 1800
    ) -> OAuthTokens:
        """
        Poll token endpoint until authorization complete.
        """
        token_request = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
            'client_id': self.provider.client_id,
            'device_code': device_code
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            while (asyncio.get_event_loop().time() - start_time) < expires_in:
                async with session.post(
                    self.provider.token_endpoint,
                    data=token_request,
                    headers={'Accept': 'application/json'}
                ) as response:
                    token_data = await response.json()
                    
                    if 'access_token' in token_data:
                        return OAuthTokens(
                            access_token=token_data['access_token'],
                            refresh_token=token_data.get('refresh_token'),
                            id_token=token_data.get('id_token'),
                            token_type=token_data.get('token_type', 'Bearer'),
                            expires_in=token_data.get('expires_in', 3600),
                            scope=token_data.get('scope', '')
                        )
                    
                    error = token_data.get('error')
                    if error == 'authorization_pending':
                        await asyncio.sleep(interval)
                        continue
                    elif error == 'slow_down':
                        interval += 5
                        await asyncio.sleep(interval)
                        continue
                    elif error:
                        raise OAuthError(
                            f"Device code error: {error}",
                            error_code=error
                        )
                
                await asyncio.sleep(interval)
        
        raise OAuthError("Device code flow timed out")


class TokenRefreshManager:
    """Manages OAuth token refresh with proactive expiration handling."""
    
    def __init__(self):
        self.active_tokens: Dict[str, OAuthTokens] = {}
        self.refresh_tasks: Dict[str, asyncio.Task] = {}
        self.token_observers: List[Callable[[str, OAuthTokens], None]] = []
        self._lock = asyncio.Lock()
    
    def add_observer(self, observer: Callable[[str, OAuthTokens], None]) -> None:
        """Add observer for token refresh events."""
        self.token_observers.append(observer)
    
    async def register_token(
        self, 
        token_id: str, 
        tokens: OAuthTokens,
        provider: OAuthProviderConfig
    ) -> None:
        """
        Register token for automatic refresh management.
        """
        async with self._lock:
            self.active_tokens[token_id] = tokens
            
            # Cancel existing refresh task
            if token_id in self.refresh_tasks:
                self.refresh_tasks[token_id].cancel()
            
            # Schedule proactive refresh
            refresh_time = tokens.obtained_at + timedelta(
                seconds=tokens.expires_in * 0.8  # Refresh at 80% of lifetime
            )
            
            delay = (refresh_time - datetime.now()).total_seconds()
            if delay > 0:
                self.refresh_tasks[token_id] = asyncio.create_task(
                    self._scheduled_refresh(token_id, provider, delay)
                )
                logger.debug(f"Scheduled refresh for {token_id} in {delay:.0f}s")
    
    async def _scheduled_refresh(
        self,
        token_id: str,
        provider: OAuthProviderConfig,
        delay: float
    ) -> None:
        """Schedule and execute token refresh."""
        await asyncio.sleep(delay)
        
        tokens = self.active_tokens.get(token_id)
        if not tokens or not tokens.refresh_token:
            logger.warning(f"Cannot refresh {token_id}: no refresh token")
            return
        
        try:
            new_tokens = await self._perform_refresh(tokens, provider)
            
            async with self._lock:
                self.active_tokens[token_id] = new_tokens
            
            # Notify observers
            for observer in self.token_observers:
                try:
                    observer(token_id, new_tokens)
                except (TypeError, ValueError, AttributeError, KeyError) as e:
                    logger.error(f"Token observer error: {e}", exc_info=True)

            # Re-schedule next refresh
            await self.register_token(token_id, new_tokens, provider)

            logger.info(f"Successfully refreshed token {token_id}")

        except OAuthError as e:
            logger.error(f"Token refresh failed for {token_id}: {e}")
            # Notify of refresh failure
            for observer in self.token_observers:
                try:
                    observer(token_id, None)
                except (TypeError, ValueError, AttributeError, KeyError) as e:
                    logger.warning(f"Token observer error during failure notification: {e}")
    
    async def _perform_refresh(
        self,
        tokens: OAuthTokens,
        provider: OAuthProviderConfig
    ) -> OAuthTokens:
        """Perform token refresh request."""
        refresh_request = {
            'grant_type': 'refresh_token',
            'client_id': provider.client_id,
            'refresh_token': tokens.refresh_token
        }
        
        if provider.client_secret:
            refresh_request['client_secret'] = provider.client_secret
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                provider.token_endpoint,
                data=refresh_request,
                headers={'Accept': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OAuthError(f"Refresh request failed: {error_text}")
                
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
    
    async def unregister_token(self, token_id: str) -> None:
        """Unregister token from refresh management."""
        async with self._lock:
            if token_id in self.refresh_tasks:
                self.refresh_tasks[token_id].cancel()
                del self.refresh_tasks[token_id]
            
            self.active_tokens.pop(token_id, None)
    
    def get_token(self, token_id: str) -> Optional[OAuthTokens]:
        """Get current token by ID."""
        return self.active_tokens.get(token_id)
    
    async def revoke_token(
        self, 
        token_id: str, 
        provider: OAuthProviderConfig
    ) -> bool:
        """Revoke token at provider."""
        tokens = self.active_tokens.get(token_id)
        if not tokens:
            return False
        
        if not provider.revocation_endpoint:
            logger.warning("Revocation endpoint not configured")
            return False
        
        revoke_request = {
            'token': tokens.refresh_token or tokens.access_token,
            'token_type_hint': 'refresh_token' if tokens.refresh_token else 'access_token',
            'client_id': provider.client_id
        }
        
        if provider.client_secret:
            revoke_request['client_secret'] = provider.client_secret
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    provider.revocation_endpoint,
                    data=revoke_request
                ) as response:
                    # Revocation returns 200 even if token was already revoked
                    success = response.status == 200
                    
                    if success:
                        await self.unregister_token(token_id)
                    
                    return success
        except (aiohttp.ClientError, OSError, ValueError) as e:
            logger.error(f"Token revocation failed: {e}")
            return False


class OAuthHandler:
    """
    Main OAuth handler coordinating all flow types.
    """
    
    # Pre-configured provider templates
    PROVIDER_TEMPLATES = {
        'google': {
            'authorization_endpoint': 'https://accounts.google.com/o/oauth2/v2/auth',
            'token_endpoint': 'https://oauth2.googleapis.com/token',
            'device_code_endpoint': 'https://oauth2.googleapis.com/device/code',
            'revocation_endpoint': 'https://oauth2.googleapis.com/revoke',
            'scopes_supported': [
                'openid', 'email', 'profile',
                'https://www.googleapis.com/auth/gmail.readonly',
                'https://www.googleapis.com/auth/gmail.send',
                'https://www.googleapis.com/auth/calendar',
                'https://www.googleapis.com/auth/drive'
            ],
            'pkce_supported': True
        },
        'microsoft': {
            'authorization_endpoint': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
            'token_endpoint': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
            'device_code_endpoint': 'https://login.microsoftonline.com/common/oauth2/v2.0/devicecode',
            'scopes_supported': [
                'openid', 'email', 'profile', 'User.Read',
                'Mail.Read', 'Mail.Send', 'Calendars.Read',
                'Files.Read', 'offline_access'
            ],
            'pkce_supported': True
        },
        'github': {
            'authorization_endpoint': 'https://github.com/login/oauth/authorize',
            'token_endpoint': 'https://github.com/login/oauth/access_token',
            'scopes_supported': ['repo', 'user', 'notifications', 'gist'],
            'pkce_supported': True
        },
        'slack': {
            'authorization_endpoint': 'https://slack.com/oauth/v2/authorize',
            'token_endpoint': 'https://slack.com/api/oauth.v2.access',
            'scopes_supported': ['chat:write', 'users:read', 'channels:read'],
            'pkce_supported': False
        }
    }
    
    def __init__(self):
        self.refresh_manager = TokenRefreshManager()
        self._providers: Dict[str, OAuthProviderConfig] = {}
    
    def register_provider(
        self, 
        name: str, 
        config: OAuthProviderConfig
    ) -> None:
        """Register OAuth provider configuration."""
        self._providers[name] = config
    
    def register_provider_from_template(
        self,
        name: str,
        template: str,
        client_id: str,
        client_secret: Optional[str] = None
    ) -> None:
        """Register provider from predefined template."""
        if template not in self.PROVIDER_TEMPLATES:
            raise ValueError(f"Unknown provider template: {template}")
        
        template_data = self.PROVIDER_TEMPLATES[template].copy()
        template_data['provider_name'] = name
        template_data['client_id'] = client_id
        template_data['client_secret'] = client_secret
        
        config = OAuthProviderConfig(**template_data)
        self.register_provider(name, config)
    
    async def authenticate(
        self,
        provider_name: str,
        flow_type: OAuthFlowType = OAuthFlowType.AUTHORIZATION_CODE_PKCE,
        scopes: Optional[List[str]] = None,
        **kwargs
    ) -> OAuthTokens:
        """
        Authenticate with specified provider and flow type.
        """
        provider = self._providers.get(provider_name)
        if not provider:
            raise OAuthError(f"Provider not registered: {provider_name}")
        
        if flow_type == OAuthFlowType.AUTHORIZATION_CODE_PKCE:
            return await self._auth_code_flow(provider, scopes, use_pkce=True, **kwargs)
        elif flow_type == OAuthFlowType.AUTHORIZATION_CODE:
            return await self._auth_code_flow(provider, scopes, use_pkce=False, **kwargs)
        elif flow_type == OAuthFlowType.CLIENT_CREDENTIALS:
            flow = ClientCredentialsFlow(provider)
            return await flow.authenticate(scopes)
        elif flow_type == OAuthFlowType.DEVICE_CODE:
            return await self._device_code_flow(provider, scopes, **kwargs)
        else:
            raise OAuthError(f"Unsupported flow type: {flow_type}")
    
    async def _auth_code_flow(
        self,
        provider: OAuthProviderConfig,
        scopes: Optional[List[str]],
        use_pkce: bool = True,
        browser_context = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> OAuthTokens:
        """Execute authorization code flow."""
        # Start callback server
        callback_server = CallbackServer()
        await callback_server.start()
        
        try:
            # Build authorization URL
            flow = AuthorizationCodeFlow(provider, callback_server.redirect_uri)
            auth_url, state, _ = flow.build_authorization_url(
                scopes=scopes,
                use_pkce=use_pkce
            )
            
            if browser_context:
                # Use browser automation
                page = await browser_context.new_page()
                await page.goto(auth_url)
                
                # Handle login if credentials provided
                if username and password:
                    await self._handle_provider_login(page, username, password)
                
                # Wait for callback
                response = await callback_server.wait_for_response(timeout=300)
                await page.close()
            else:
                # Manual flow - just wait for callback
                print(f"Please visit: {auth_url}")
                response = await callback_server.wait_for_response(timeout=300)
            
            # Exchange code for tokens
            auth_code = response.get('code')
            received_state = response.get('state')
            
            if not auth_code:
                raise OAuthError("No authorization code received")
            
            tokens = await flow.exchange_code(auth_code, state, received_state)
            
            return tokens
            
        finally:
            await callback_server.stop()
    
    async def _device_code_flow(
        self,
        provider: OAuthProviderConfig,
        scopes: Optional[List[str]],
        **kwargs
    ) -> OAuthTokens:
        """Execute device code flow."""
        flow = DeviceCodeFlow(provider)
        
        # Initiate device code flow
        device_response = await flow.initiate(scopes)
        
        # Display user instructions
        print(f"Device Code: {device_response['user_code']}")
        print(f"Visit: {device_response['verification_uri']}")
        
        # Poll for token
        tokens = await flow.poll_for_token(
            device_response['device_code'],
            interval=device_response.get('interval', 5),
            expires_in=device_response.get('expires_in', 1800)
        )
        
        return tokens
    
    # Provider-specific selectors for login pages
    _PROVIDER_SELECTORS = {
        'google': {
            'username': 'input[type="email"], input#identifierId',
            'username_next': '#identifierNext button, button[type="submit"]',
            'password': 'input[type="password"], input[name="Passwd"]',
            'password_next': '#passwordNext button, button[type="submit"]',
            'consent_allow': 'button#submit_approve_access, button[data-idom-class="nCP5yc"]',
        },
        'microsoft': {
            'username': 'input[type="email"], input[name="loginfmt"]',
            'username_next': 'input[type="submit"]#idSIButton9',
            'password': 'input[type="password"], input[name="passwd"]',
            'password_next': 'input[type="submit"]#idSIButton9',
            'consent_allow': 'input[type="submit"]#idBtn_Accept',
        },
        'github': {
            'username': 'input#login_field',
            'username_next': None,
            'password': 'input#password',
            'password_next': 'input[type="submit"][name="commit"]',
            'consent_allow': 'button#js-oauth-authorize-btn, button[name="authorize"]',
        },
        'facebook': {
            'username': 'input#email, input[name="email"]',
            'username_next': None,
            'password': 'input#pass, input[name="pass"]',
            'password_next': 'button[name="login"], button[type="submit"]',
            'consent_allow': 'button[name="__CONFIRM__"]',
        },
        'apple': {
            'username': 'input#account_name_text_field',
            'username_next': '#sign-in',
            'password': 'input#password_text_field',
            'password_next': '#sign-in',
            'consent_allow': 'button.button-primary',
        },
    }

    # Generic fallback selectors
    _GENERIC_SELECTORS = {
        'username': 'input[type="email"], input[name="email"], input[name="username"], input[name="login"]',
        'username_next': 'button[type="submit"], input[type="submit"]',
        'password': 'input[type="password"]',
        'password_next': 'button[type="submit"], input[type="submit"]',
        'consent_allow': 'button[type="submit"]',
    }

    def _detect_provider(self, page_url: str) -> str:
        """Detect OAuth provider from page URL."""
        url_lower = page_url.lower()
        for provider in self._PROVIDER_SELECTORS:
            if provider in url_lower:
                return provider
        return 'generic'

    async def _handle_provider_login(
        self,
        page,
        username: str,
        password: str
    ) -> None:
        """Handle provider login page automation with provider-specific selectors."""
        provider = self._detect_provider(page.url)
        selectors = self._PROVIDER_SELECTORS.get(provider, self._GENERIC_SELECTORS)
        logger.debug(f"Using {provider} selectors for login automation")

        try:
            # Fill username
            username_field = await page.wait_for_selector(
                selectors['username'],
                timeout=8000
            )
            if username_field:
                await username_field.fill(username)

                # Click next if there is a separate step
                next_selector = selectors.get('username_next')
                if next_selector:
                    next_button = await page.query_selector(next_selector)
                    if next_button:
                        await next_button.click()
                        # Wait for password page to load
                        await page.wait_for_timeout(1500)

            # Fill password
            password_field = await page.wait_for_selector(
                selectors['password'],
                timeout=8000
            )
            if password_field:
                await password_field.fill(password)

                # Submit password
                submit_selector = selectors.get('password_next')
                if submit_selector:
                    submit_button = await page.query_selector(submit_selector)
                    if submit_button:
                        await submit_button.click()
                        await page.wait_for_timeout(2000)

            # Handle consent/authorization screen if present
            consent_selector = selectors.get('consent_allow')
            if consent_selector:
                try:
                    consent_button = await page.wait_for_selector(
                        consent_selector,
                        timeout=3000
                    )
                    if consent_button:
                        await consent_button.click()
                except (TimeoutError, OSError):
                    pass  # No consent screen, continue

        except (OSError, ValueError, TimeoutError) as e:
            logger.warning(f"Automated login handling failed for {provider}: {e}")
            # Try generic fallback if provider-specific failed
            if provider != 'generic':
                logger.info("Retrying with generic selectors")
                try:
                    fallback = self._GENERIC_SELECTORS
                    username_field = await page.wait_for_selector(
                        fallback['username'], timeout=5000
                    )
                    if username_field:
                        await username_field.fill(username)
                        next_btn = await page.query_selector(fallback['username_next'])
                        if next_btn:
                            await next_btn.click()

                    password_field = await page.wait_for_selector(
                        fallback['password'], timeout=5000
                    )
                    if password_field:
                        await password_field.fill(password)
                        submit_btn = await page.query_selector(fallback['password_next'])
                        if submit_btn:
                            await submit_btn.click()
                except (OSError, ValueError, TimeoutError) as fallback_err:
                    logger.warning(f"Generic fallback login also failed: {fallback_err}")
    
    async def refresh_token(
        self,
        provider_name: str,
        refresh_token: str
    ) -> OAuthTokens:
        """Manually refresh token."""
        provider = self._providers.get(provider_name)
        if not provider:
            raise OAuthError(f"Provider not registered: {provider_name}")
        
        # Create dummy tokens object
        tokens = OAuthTokens(
            access_token="",
            refresh_token=refresh_token,
            obtained_at=datetime.now()
        )
        
        return await self.refresh_manager._perform_refresh(tokens, provider)
    
    async def revoke_token(
        self,
        provider_name: str,
        token_id: str
    ) -> bool:
        """Revoke token."""
        provider = self._providers.get(provider_name)
        if not provider:
            raise OAuthError(f"Provider not registered: {provider_name}")
        
        return await self.refresh_manager.revoke_token(token_id, provider)


# Convenience functions
async def create_oauth_handler() -> OAuthHandler:
    """Create and initialize OAuth handler."""
    return OAuthHandler()


def create_google_provider(
    client_id: str,
    client_secret: Optional[str] = None
) -> OAuthProviderConfig:
    """Create Google OAuth provider configuration."""
    return OAuthProviderConfig(
        provider_name="google",
        client_id=client_id,
        client_secret=client_secret,
        authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        device_code_endpoint="https://oauth2.googleapis.com/device/code",
        revocation_endpoint="https://oauth2.googleapis.com/revoke",
        scopes_supported=[
            'openid', 'email', 'profile',
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.send',
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/drive'
        ],
        pkce_supported=True
    )


def create_microsoft_provider(
    client_id: str,
    client_secret: Optional[str] = None,
    tenant: str = "common"
) -> OAuthProviderConfig:
    """Create Microsoft OAuth provider configuration."""
    return OAuthProviderConfig(
        provider_name="microsoft",
        client_id=client_id,
        client_secret=client_secret,
        authorization_endpoint=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        token_endpoint=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        device_code_endpoint=f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        scopes_supported=[
            'openid', 'email', 'profile', 'User.Read',
            'Mail.Read', 'Mail.Send', 'Calendars.Read',
            'Files.Read', 'offline_access'
        ],
        pkce_supported=True
    )
