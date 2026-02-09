"""
Authentication Orchestrator
Main coordination layer for all authentication systems
For Windows 10 OpenClaw AI Agent Framework
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field

# Import all authentication modules
from .cookie_jar import CookieJarManager, create_cookie_jar
from .session_manager import SessionManager
from .oauth_handler import OAuthHandler, create_oauth_handler, OAuthTokens
from .jwt_handler import JWTValidator, JWTGenerator, create_jwt_validator
from .mfa_handler import MFAHandler, MFAResult, MFAType
from .credential_vault import (
    create_credential_vault, 
    Credentials,
    MFACredentials
)
from .form_auth import FormAuthenticator, AuthenticationResult
from .sso_handler import SSOHandler, SAMLAuthenticationResult, OIDCAuthenticationResult

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """Authentication context for a service."""
    service_name: str
    username: Optional[str] = None
    auth_method: str = "unknown"  # 'password', 'oauth', 'sso', 'certificate'
    is_authenticated: bool = False
    session_id: Optional[str] = None
    tokens: Optional[Dict[str, Any]] = None
    cookies: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


@dataclass
class AuthenticationConfig:
    """Global authentication configuration."""
    # Storage paths
    cookie_storage_path: Path = Path.home() / '.openclaw' / 'cookies.enc'
    session_storage_path: Path = Path.home() / '.openclaw' / 'sessions'
    credential_storage_path: Path = Path.home() / '.openclaw' / 'credentials'
    
    # Security settings
    encryption_key: Optional[bytes] = None
    namespace: str = "OpenClawAgent"
    
    # Feature flags
    enable_windows_vault: bool = True
    auto_save_sessions: bool = True
    auto_refresh_tokens: bool = True
    
    # Timeouts
    default_timeout: int = 60
    mfa_timeout: int = 120


class AuthenticationOrchestrator:
    """
    Main authentication orchestrator coordinating all authentication systems.
    
    This is the primary interface for the AI agent to handle all authentication
    needs including cookies, sessions, OAuth, JWT, MFA, and SSO.
    """
    
    def __init__(self, config: Optional[AuthenticationConfig] = None):
        """
        Initialize authentication orchestrator.
        
        Args:
            config: Authentication configuration
        """
        self.config = config or AuthenticationConfig()
        
        # Initialize subsystems
        self._init_cookie_jar()
        self._init_session_manager()
        self._init_credential_vault()
        self._init_oauth_handler()
        self._init_mfa_handler()
        self._init_form_authenticator()
        self._init_sso_handler()
        
        # Active authentication contexts
        self.auth_contexts: Dict[str, AuthContext] = {}
        
        # Event observers
        self._observers: List[Callable] = []
        
        logger.info("Authentication orchestrator initialized")
    
    def _init_cookie_jar(self) -> None:
        """Initialize cookie jar manager."""
        self.cookie_jar = CookieJarManager(
            storage_path=self.config.cookie_storage_path,
            master_key=self.config.encryption_key,
            auto_save=True
        )
    
    def _init_session_manager(self) -> None:
        """Initialize session manager."""
        try:
            self.session_manager = SessionManager(
                session_timeout=getattr(self.config, 'session_timeout', 3600)
            )
        except (TypeError, AttributeError) as e:
            self.session_manager = {
                'sessions': {},
                'timeout': getattr(self.config, 'session_timeout', 3600)
            }
            logger.warning(f"Using dict-based session manager fallback: {e}")
    
    def _init_credential_vault(self) -> None:
        """Initialize credential vault."""
        self.credential_vault = create_credential_vault(
            storage_path=self.config.credential_storage_path,
            namespace=self.config.namespace,
            prefer_windows=self.config.enable_windows_vault
        )
    
    def _init_oauth_handler(self) -> None:
        """Initialize OAuth handler."""
        self.oauth_handler = create_oauth_handler()
    
    def _init_mfa_handler(self) -> None:
        """Initialize MFA handler."""
        self.mfa_handler = MFAHandler(
            credential_vault=self.credential_vault
        )
    
    def _init_form_authenticator(self) -> None:
        """Initialize form authenticator."""
        self.form_authenticator = FormAuthenticator(
            credential_vault=self.credential_vault,
            mfa_handler=self.mfa_handler
        )
    
    def _init_sso_handler(self) -> None:
        """Initialize SSO handler."""
        self.sso_handler = SSOHandler()
    
    # ==================== Credential Management ====================
    
    async def store_credentials(
        self,
        service_name: str,
        username: str,
        password: str,
        **attributes
    ) -> bool:
        """
        Store credentials for a service.
        
        Args:
            service_name: Service identifier
            username: Username
            password: Password
            **attributes: Additional credential attributes
            
        Returns:
            True if successful
        """
        return await self.credential_vault.store_credentials(
            service_name=service_name,
            username=username,
            password=password,
            attributes=attributes
        )
    
    async def get_credentials(
        self,
        service_name: str,
        username: Optional[str] = None
    ) -> Optional[Credentials]:
        """
        Get credentials for a service.
        
        Args:
            service_name: Service identifier
            username: Username (optional)
            
        Returns:
            Credentials or None
        """
        return await self.credential_vault.get_credentials(service_name, username)
    
    async def store_mfa_credentials(
        self,
        service_name: str,
        username: str,
        mfa_type: str,
        **mfa_data
    ) -> bool:
        """
        Store MFA credentials.
        
        Args:
            service_name: Service identifier
            username: Username
            mfa_type: MFA type ('totp', 'sms', 'email', 'backup')
            **mfa_data: MFA-specific data (totp_secret, phone_number, etc.)
            
        Returns:
            True if successful
        """
        mfa_creds = MFACredentials(
            service_name=service_name,
            username=username,
            mfa_type=mfa_type,
            **mfa_data
        )
        return await self.credential_vault.store_mfa_credentials(mfa_creds)
    
    # ==================== Cookie Management ====================
    
    async def get_cookies_for_url(self, url: str) -> List[Dict]:
        """
        Get cookies applicable to a URL.
        
        Args:
            url: Target URL
            
        Returns:
            List of cookies
        """
        cookies = await self.cookie_jar.get_cookies_for_request(url)
        return [c.to_dict() for c in cookies]
    
    async def set_cookie(self, cookie_data: Dict, source_url: str) -> bool:
        """
        Set a cookie.
        
        Args:
            cookie_data: Cookie data dictionary
            source_url: URL where cookie was received
            
        Returns:
            True if successful
        """
        from .cookie_jar import Cookie
        cookie = Cookie.from_dict(cookie_data)
        return await self.cookie_jar.set_cookie(cookie, source_url)
    
    async def sync_cookies_to_browser(
        self,
        browser_context,
        url: Optional[str] = None,
        domains: Optional[List[str]] = None
    ) -> int:
        """
        Sync cookies to browser context.
        
        Args:
            browser_context: Playwright browser context
            url: URL to get cookies for
            domains: Specific domains to sync
            
        Returns:
            Number of cookies synced
        """
        from .cookie_jar import sync_cookies_to_playwright
        return await sync_cookies_to_playwright(
            self.cookie_jar, browser_context, url, domains
        )
    
    async def sync_cookies_from_browser(
        self,
        browser_context,
        source_url: str
    ) -> int:
        """
        Sync cookies from browser context.
        
        Args:
            browser_context: Playwright browser context
            source_url: Source URL
            
        Returns:
            Number of cookies synced
        """
        from .cookie_jar import sync_cookies_from_playwright
        return await sync_cookies_from_playwright(
            self.cookie_jar, browser_context, source_url
        )
    
    # ==================== Form Authentication ====================
    
    async def authenticate_with_form(
        self,
        page,
        service_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        wait_for_mfa: bool = True,
        timeout: Optional[int] = None
    ) -> AuthenticationResult:
        """
        Authenticate using form-based login.
        
        Args:
            page: Playwright page object
            service_name: Service identifier
            username: Username (if None, uses stored credentials)
            password: Password (if None, uses stored credentials)
            wait_for_mfa: Whether to handle MFA if required
            timeout: Authentication timeout
            
        Returns:
            AuthenticationResult
        """
        result = await self.form_authenticator.authenticate(
            page=page,
            service_name=service_name,
            username=username,
            password=password,
            wait_for_mfa=wait_for_mfa,
            timeout=timeout or self.config.default_timeout
        )
        
        if result.success:
            # Update auth context
            context = AuthContext(
                service_name=service_name,
                username=username or result.user_info.get('username'),
                auth_method='password',
                is_authenticated=True,
                cookies=result.session_cookies,
                metadata={'login_time': result.login_time}
            )
            self.auth_contexts[service_name] = context
            
            # Sync cookies to jar
            await self.sync_cookies_from_browser(page, page.url)
            
            # Notify observers
            await self._notify_observers('authenticated', context)
        
        return result
    
    # ==================== OAuth Authentication ====================
    
    async def authenticate_with_oauth(
        self,
        provider_name: str,
        browser_context,
        scopes: Optional[List[str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate using OAuth 2.0.
        
        Args:
            provider_name: OAuth provider name
            browser_context: Playwright browser context
            scopes: Requested scopes
            username: Username for automated login
            password: Password for automated login
            
        Returns:
            Token response dictionary
        """
        try:
            tokens = await self.oauth_handler.authenticate(
                provider_name=provider_name,
                scopes=scopes,
                browser_context=browser_context,
                username=username,
                password=password
            )
            
            # Update auth context
            context = AuthContext(
                service_name=provider_name,
                username=username,
                auth_method='oauth',
                is_authenticated=True,
                tokens=tokens.to_dict(),
                metadata={'scope': tokens.scope}
            )
            self.auth_contexts[provider_name] = context
            
            # Notify observers
            await self._notify_observers('authenticated', context)
            
            return tokens.to_dict()
            
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"OAuth authentication failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def register_oauth_provider(
        self,
        name: str,
        template: str,
        client_id: str,
        client_secret: Optional[str] = None
    ) -> None:
        """
        Register OAuth provider from template.
        
        Args:
            name: Provider name
            template: Template name ('google', 'microsoft', 'github', etc.)
            client_id: OAuth client ID
            client_secret: OAuth client secret
        """
        self.oauth_handler.register_provider_from_template(
            name=name,
            template=template,
            client_id=client_id,
            client_secret=client_secret
        )
    
    # ==================== SSO Authentication ====================
    
    async def authenticate_with_sso(
        self,
        provider_name: str,
        protocol: str,
        browser_context,
        **kwargs
    ) -> Union[SAMLAuthenticationResult, OIDCAuthenticationResult]:
        """
        Authenticate using SSO.
        
        Args:
            provider_name: SSO provider name
            protocol: Protocol ('saml' or 'oidc')
            browser_context: Playwright browser context
            **kwargs: Additional protocol-specific arguments
            
        Returns:
            Authentication result
        """
        if protocol == 'oidc':
            result = await self.sso_handler.authenticate_oidc(
                provider_name=provider_name,
                browser_context=browser_context,
                **kwargs
            )
        elif protocol == 'saml':
            result = await self.sso_handler.authenticate_saml(
                provider_name=provider_name,
                browser_context=browser_context,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        if result.success:
            # Update auth context
            context = AuthContext(
                service_name=provider_name,
                username=getattr(result, 'user_id', None),
                auth_method='sso',
                is_authenticated=True,
                metadata={'protocol': protocol}
            )
            self.auth_contexts[provider_name] = context
            
            # Notify observers
            await self._notify_observers('authenticated', context)
        
        return result
    
    def register_oidc_provider(
        self,
        name: str,
        template: str,
        client_id: str,
        client_secret: Optional[str] = None,
        **template_params
    ) -> None:
        """
        Register OIDC provider from template.
        
        Args:
            name: Provider name
            template: Template name ('azure_ad', 'okta', 'auth0', etc.)
            client_id: OAuth client ID
            client_secret: OAuth client secret
            **template_params: Template-specific parameters
        """
        self.sso_handler.register_from_template(
            name=name,
            template=template,
            client_id=client_id,
            client_secret=client_secret,
            **template_params
        )
    
    # ==================== MFA Handling ====================
    
    async def handle_mfa(
        self,
        page,
        service_name: str,
        username: Optional[str] = None,
        preferred_type: Optional[str] = None
    ) -> MFAResult:
        """
        Handle MFA challenge.
        
        Args:
            page: Playwright page object
            service_name: Service identifier
            username: Username
            preferred_type: Preferred MFA type
            
        Returns:
            MFAResult
        """
        mfa_type = MFAType.TOTP if preferred_type == 'totp' else MFAType.UNKNOWN
        
        return await self.mfa_handler.handle_mfa(
            page=page,
            service_name=service_name,
            username=username,
            preferred_type=mfa_type
        )
    
    async def generate_totp_code(
        self,
        service_name: str,
        username: str
    ) -> Optional[str]:
        """
        Generate TOTP code for service.
        
        Args:
            service_name: Service identifier
            username: Username
            
        Returns:
            TOTP code or None
        """
        return await self.mfa_handler.generate_totp_code(service_name, username)
    
    # ==================== JWT Handling ====================
    
    async def validate_jwt(
        self,
        token: str,
        jwks_url: Optional[str] = None,
        expected_audience: Optional[str] = None,
        expected_issuer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token
            jwks_url: JWKS endpoint URL
            expected_audience: Expected audience
            expected_issuer: Expected issuer
            
        Returns:
            Decoded token payload
        """
        validator = create_jwt_validator(jwks_url=jwks_url)
        decoded = await validator.validate_token(
            token=token,
            expected_audience=expected_audience,
            expected_issuer=expected_issuer
        )
        return decoded.payload
    
    # ==================== Session Management ====================
    
    def get_auth_context(self, service_name: str) -> Optional[AuthContext]:
        """
        Get authentication context for a service.
        
        Args:
            service_name: Service identifier
            
        Returns:
            AuthContext or None
        """
        return self.auth_contexts.get(service_name)
    
    def is_authenticated(self, service_name: str) -> bool:
        """
        Check if authenticated to a service.
        
        Args:
            service_name: Service identifier
            
        Returns:
            True if authenticated
        """
        context = self.auth_contexts.get(service_name)
        return context is not None and context.is_authenticated
    
    async def logout(self, service_name: str) -> bool:
        """
        Logout from a service.
        
        Args:
            service_name: Service identifier
            
        Returns:
            True if successful
        """
        context = self.auth_contexts.get(service_name)
        if not context:
            return False
        
        # Clear cookies for service
        await self.cookie_jar.clear_domain(service_name)
        
        # Remove auth context
        del self.auth_contexts[service_name]
        
        # Notify observers
        await self._notify_observers('logged_out', {'service_name': service_name})
        
        return True
    
    async def logout_all(self) -> int:
        """
        Logout from all services.
        
        Returns:
            Number of services logged out
        """
        count = len(self.auth_contexts)
        
        for service_name in list(self.auth_contexts.keys()):
            await self.logout(service_name)
        
        return count
    
    # ==================== Event Handling ====================
    
    def add_observer(self, observer: Callable[[str, Any], None]) -> None:
        """
        Add event observer.
        
        Args:
            observer: Callback function(event_type, data)
        """
        self._observers.append(observer)
    
    async def _notify_observers(self, event_type: str, data: Any) -> None:
        """Notify all observers of an event."""
        for observer in self._observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(event_type, data)
                else:
                    observer(event_type, data)
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                logger.error(f"Observer error: {e}", exc_info=True)
    
    # ==================== Utility Methods ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get authentication system statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'active_sessions': len(self.auth_contexts),
            'services': list(self.auth_contexts.keys()),
            'cookie_jar': self.cookie_jar.get_stats(),
            'oauth_providers': list(self.oauth_handler._providers.keys()),
            'sso_providers': {
                'saml': list(self.sso_handler.saml_handlers.keys()),
                'oidc': list(self.sso_handler.oidc_handlers.keys())
            }
        }
    
    async def save_state(self) -> bool:
        """
        Save all authentication state.
        
        Returns:
            True if successful
        """
        try:
            # Save cookie jar
            await self.cookie_jar.save_jars()
            
            logger.info("Authentication state saved")
            return True
            
        except (OSError, PermissionError, ValueError) as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    async def clear_all_data(self) -> None:
        """Clear all authentication data."""
        # Clear cookies
        await self.cookie_jar.clear_all()
        
        # Clear contexts
        self.auth_contexts.clear()
        
        logger.info("All authentication data cleared")


# Factory function
def create_auth_orchestrator(
    cookie_path: Optional[Path] = None,
    credential_path: Optional[Path] = None,
    namespace: str = "OpenClawAgent"
) -> AuthenticationOrchestrator:
    """
    Create authentication orchestrator with custom paths.
    
    Args:
        cookie_path: Path for cookie storage
        credential_path: Path for credential storage
        namespace: Namespace for credential isolation
        
    Returns:
        AuthenticationOrchestrator instance
    """
    config = AuthenticationConfig(
        namespace=namespace
    )
    
    if cookie_path:
        config.cookie_storage_path = cookie_path
    if credential_path:
        config.credential_storage_path = credential_path
    
    return AuthenticationOrchestrator(config)
