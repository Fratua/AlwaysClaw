"""
Web Authentication and Session Management System
For Windows 10 OpenClaw AI Agent Framework

This package provides comprehensive authentication capabilities including:
- Cookie jar management and persistence
- Session storage and restoration
- OAuth 2.0 flow automation
- JWT token handling
- Form-based authentication automation
- Multi-factor authentication handling
- Credential vault integration
- SSO (Single Sign-On) support
"""

__version__ = "1.0.0"
__author__ = "OpenClaw Agent Team"

# Core components
from .cookie_jar import (
    CookieJarManager,
    Cookie,
    DomainCookieJar,
    create_cookie_jar,
    sync_cookies_to_playwright,
    sync_cookies_from_playwright
)

from .credential_vault import (
    Credentials,
    MFACredentials,
    WindowsCredentialVault,
    EncryptedFileVault,
    create_credential_vault,
    store_service_credentials,
    get_service_credentials
)

from .oauth_handler import (
    OAuthHandler,
    OAuthTokens,
    OAuthFlowType,
    OAuthProviderConfig,
    AuthorizationCodeFlow,
    ClientCredentialsFlow,
    DeviceCodeFlow,
    TokenRefreshManager,
    create_oauth_handler,
    create_google_provider,
    create_microsoft_provider
)

from .jwt_handler import (
    JWTValidator,
    JWTGenerator,
    JWTConfig,
    DecodedJWT,
    TokenRotationManager,
    TokenBlacklist,
    create_jwt_validator,
    create_jwt_generator,
    generate_rsa_keypair
)

from .mfa_handler import (
    MFAHandler,
    MFAResult,
    MFAType,
    TOTPGenerator,
    TOTPMFAHandler,
    SMSOTPHandler,
    BackupCodeHandler,
    create_totp_generator,
    generate_totp_secret,
    verify_totp_code
)

from .form_auth import (
    FormAuthenticator,
    FormDetector,
    LoginForm,
    AuthenticationResult,
    LoginVerifier,
    CaptchaSolver,
    detect_login_form,
    verify_login
)

from .sso_handler import (
    SSOHandler,
    SAMLHandler,
    OIDCHandler,
    SAMLProviderConfig,
    OIDCProviderConfig,
    SAMLAuthenticationResult,
    OIDCAuthenticationResult,
    SSOSession,
    SSOSessionManager,
    create_saml_handler,
    create_oidc_handler,
    create_sso_handler
)

from .auth_orchestrator import (
    AuthenticationOrchestrator,
    AuthContext,
    AuthenticationConfig,
    create_auth_orchestrator
)

__all__ = [
    # Cookie Management
    'CookieJarManager',
    'Cookie',
    'DomainCookieJar',
    'create_cookie_jar',
    'sync_cookies_to_playwright',
    'sync_cookies_from_playwright',
    
    # Credential Vault
    'Credentials',
    'MFACredentials',
    'WindowsCredentialVault',
    'EncryptedFileVault',
    'create_credential_vault',
    'store_service_credentials',
    'get_service_credentials',
    
    # OAuth
    'OAuthHandler',
    'OAuthTokens',
    'OAuthFlowType',
    'OAuthProviderConfig',
    'AuthorizationCodeFlow',
    'ClientCredentialsFlow',
    'DeviceCodeFlow',
    'TokenRefreshManager',
    'create_oauth_handler',
    'create_google_provider',
    'create_microsoft_provider',
    
    # JWT
    'JWTValidator',
    'JWTGenerator',
    'JWTConfig',
    'DecodedJWT',
    'TokenRotationManager',
    'TokenBlacklist',
    'create_jwt_validator',
    'create_jwt_generator',
    'generate_rsa_keypair',
    
    # MFA
    'MFAHandler',
    'MFAResult',
    'MFAType',
    'TOTPGenerator',
    'TOTPMFAHandler',
    'SMSOTPHandler',
    'BackupCodeHandler',
    'create_totp_generator',
    'generate_totp_secret',
    'verify_totp_code',
    
    # Form Auth
    'FormAuthenticator',
    'FormDetector',
    'LoginForm',
    'AuthenticationResult',
    'LoginVerifier',
    'CaptchaSolver',
    'detect_login_form',
    'verify_login',
    
    # SSO
    'SSOHandler',
    'SAMLHandler',
    'OIDCHandler',
    'SAMLProviderConfig',
    'OIDCProviderConfig',
    'SAMLAuthenticationResult',
    'OIDCAuthenticationResult',
    'SSOSession',
    'SSOSessionManager',
    'create_saml_handler',
    'create_oidc_handler',
    'create_sso_handler',
    
    # Orchestrator
    'AuthenticationOrchestrator',
    'AuthContext',
    'AuthenticationConfig',
    'create_auth_orchestrator',
]
