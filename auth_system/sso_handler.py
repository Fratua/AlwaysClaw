"""
SSO (Single Sign-On) Support System
For Windows 10 OpenClaw AI Agent Framework

Supports:
- SAML 2.0
- OpenID Connect (OIDC)
- OAuth 2.0 based SSO
"""

import os
import jwt
import json
import base64
import secrets
import asyncio
import logging
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)


# SAML Namespaces
SAML_NAMESPACES = {
    'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
    'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
    'ds': 'http://www.w3.org/2000/09/xmldsig#',
    'xenc': 'http://www.w3.org/2001/04/xmlenc#'
}


@dataclass
class SAMLProviderConfig:
    """SAML Identity Provider configuration."""
    name: str
    entity_id: str
    sso_url: str  # Single Sign-On URL
    slo_url: Optional[str] = None  # Single Logout URL
    x509_cert: str = ""  # IdP signing certificate
    metadata_url: Optional[str] = None

    # SP Configuration
    sp_entity_id: str = "openclaw-agent"
    sp_acs_url: Optional[str] = None
    sp_sls_url: Optional[str] = None

    # Security settings
    want_assertions_signed: bool = True
    want_response_signed: bool = True
    signature_algorithm: str = "rsa-sha256"
    digest_algorithm: str = "sha256"

    def __post_init__(self):
        if self.sp_acs_url is None:
            callback_host = os.environ.get('OAUTH_CALLBACK_HOST', 'localhost')
            self.sp_acs_url = f"http://{callback_host}:8080/saml/acs"


@dataclass
class OIDCProviderConfig:
    """OpenID Connect Provider configuration."""
    name: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    end_session_endpoint: Optional[str] = None

    # Client configuration
    client_id: str = ""
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None

    # Scopes
    scopes_supported: List[str] = field(default_factory=list)
    default_scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    # Security
    pkce_enabled: bool = True

    def __post_init__(self):
        if self.redirect_uri is None:
            callback_host = os.environ.get('OAUTH_CALLBACK_HOST', 'localhost')
            self.redirect_uri = f"http://{callback_host}:8080/callback"


@dataclass
class SAMLAuthenticationResult:
    """SAML authentication result."""
    success: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    name_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    session_index: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OIDCAuthenticationResult:
    """OIDC authentication result."""
    success: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    id_token: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SSOSession:
    """SSO session information."""
    session_id: str
    idp_name: str
    protocol: str  # 'saml', 'oidc'
    user_id: str
    created_at: datetime
    expires_at: datetime
    saml_assertion: Optional[str] = None
    oidc_tokens: Optional[Dict] = None
    service_sessions: Dict[str, Any] = field(default_factory=dict)


class SAMLHandler:
    """
    SAML 2.0 authentication handler.
    """
    
    def __init__(self, config: SAMLProviderConfig):
        """
        Initialize SAML handler.
        
        Args:
            config: SAML provider configuration
        """
        self.config = config
        self._load_sp_keys()
    
    def _load_sp_keys(self) -> None:
        """Load Service Provider signing/encryption keys."""
        key_dir = getattr(self.config, 'key_directory', None) or os.path.join(
            os.path.expanduser('~'), '.openclaw', 'auth', 'saml'
        )

        private_key_path = os.path.join(key_dir, 'sp_private.pem')
        public_cert_path = os.path.join(key_dir, 'sp_public.crt')

        try:
            if os.path.exists(private_key_path):
                with open(private_key_path, 'rb') as f:
                    from cryptography.hazmat.primitives.serialization import load_pem_private_key
                    self.sp_private_key = load_pem_private_key(f.read(), password=None)
            else:
                self.sp_private_key = None
                logger.info(f"SP private key not found at {private_key_path}")

            if os.path.exists(public_cert_path):
                with open(public_cert_path, 'rb') as f:
                    from cryptography.x509 import load_pem_x509_certificate
                    self.sp_public_cert = load_pem_x509_certificate(f.read())
            else:
                self.sp_public_cert = None
                logger.info(f"SP public cert not found at {public_cert_path}")
        except (OSError, ValueError, ImportError) as e:
            logger.error(f"Failed to load SP keys: {e}")
            self.sp_private_key = None
            self.sp_public_cert = None
    
    def generate_authn_request(
        self,
        force_authn: bool = False,
        name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    ) -> Tuple[str, str]:
        """
        Generate SAML Authentication Request.
        
        Args:
            force_authn: Force re-authentication
            name_id_format: Requested NameID format
            
        Returns:
            Tuple of (base64_encoded_request, request_id)
        """
        request_id = f"_{secrets.token_hex(16)}"
        issue_instant = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Build AuthnRequest XML
        authn_request = f"""<?xml version="1.0" encoding="UTF-8"?>
        <samlp:AuthnRequest 
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{issue_instant}"
            Destination="{self.config.sso_url}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            AssertionConsumerServiceURL="{self.config.sp_acs_url}"
            {'ForceAuthn="true"' if force_authn else ''}>
            <saml:Issuer>{self.config.sp_entity_id}</saml:Issuer>
            <samlp:NameIDPolicy 
                Format="{name_id_format}"
                AllowCreate="true"/>
            <samlp:RequestedAuthnContext Comparison="exact">
                <saml:AuthnContextClassRef>
                    urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport
                </saml:AuthnContextClassRef>
            </samlp:RequestedAuthnContext>
        </samlp:AuthnRequest>"""
        
        # Sign the request if we have a private key
        if self.sp_private_key:
            authn_request = self._sign_request(authn_request)
        
        # Deflate and base64 encode
        import zlib
        compressed = zlib.compress(authn_request.encode('utf-8'))[2:-4]  # Strip zlib header/footer
        encoded = base64.b64encode(compressed).decode('ascii')
        
        return encoded, request_id
    
    def _sign_request(self, request_xml: str) -> str:
        """Sign SAML request using XML digital signature."""
        allow_unsigned = getattr(self.config, 'allow_unsigned', False)
        try:
            from signxml import XMLSigner
            import signxml
            key_path = os.environ.get('SP_PRIVATE_KEY_PATH', '')
            if key_path and os.path.exists(key_path):
                with open(key_path, 'rb') as f:
                    private_key = f.read()
                signer = XMLSigner(method=signxml.methods.enveloped)
                signed_xml = signer.sign(request_xml, key=private_key)
                return signed_xml
            else:
                if allow_unsigned:
                    logger.warning("SP_PRIVATE_KEY_PATH not set or file not found, returning unsigned XML (allow_unsigned=True)")
                    return request_xml
                raise RuntimeError("SP_PRIVATE_KEY_PATH not set or file not found. Set allow_unsigned=True in config to send unsigned requests.")
        except ImportError:
            if allow_unsigned:
                logger.warning("signxml not installed, returning unsigned XML (allow_unsigned=True)")
                return request_xml
            raise RuntimeError("signxml is not installed. Install it with 'pip install signxml' or set allow_unsigned=True in config.")
    
    def build_login_url(self, authn_request: str, relay_state: Optional[str] = None) -> str:
        """
        Build IdP login URL with AuthnRequest.
        
        Args:
            authn_request: Base64-encoded AuthnRequest
            relay_state: Relay state for callback
            
        Returns:
            Full login URL
        """
        params = {'SAMLRequest': authn_request}
        
        if relay_state:
            params['RelayState'] = relay_state
        
        return f"{self.config.sso_url}?{urlencode(params)}"
    
    def parse_saml_response(self, saml_response: str) -> ET.Element:
        """
        Parse and decode SAML Response.
        
        Args:
            saml_response: Base64-encoded SAML Response
            
        Returns:
            Parsed XML Element
        """
        decoded = base64.b64decode(saml_response)
        return ET.fromstring(decoded)
    
    def verify_response(self, response_xml: ET.Element) -> SAMLAuthenticationResult:
        """
        Verify SAML Response signature and extract assertions.
        
        Args:
            response_xml: Parsed SAML Response XML
            
        Returns:
            SAMLAuthenticationResult
        """
        try:
            # Register namespaces
            for prefix, uri in SAML_NAMESPACES.items():
                ET.register_namespace(prefix, uri)
            
            # Check for errors
            status_code = response_xml.find('.//samlp:StatusCode', SAML_NAMESPACES)
            if status_code is not None:
                status_value = status_code.get('Value', '')
                if 'Success' not in status_value:
                    status_message = response_xml.find('.//samlp:StatusMessage', SAML_NAMESPACES)
                    error_msg = status_message.text if status_message is not None else "Authentication failed"
                    return SAMLAuthenticationResult(
                        success=False,
                        error=error_msg
                    )
            
            # Extract Assertion
            assertion = response_xml.find('.//saml:Assertion', SAML_NAMESPACES)
            if assertion is None:
                # Check for encrypted assertion
                encrypted_assertion = response_xml.find('.//saml:EncryptedAssertion', SAML_NAMESPACES)
                if encrypted_assertion is not None:
                    logger.warning("Encrypted SAML assertions not yet supported, returning raw assertion")
                    return SAMLAuthenticationResult(
                        success=False,
                        error="Encrypted assertions not yet supported"
                    )
                return SAMLAuthenticationResult(
                    success=False,
                    error="No assertion found in response"
                )
            
            # Verify assertion signature if required
            if self.config.want_assertions_signed:
                try:
                    from signxml import XMLVerifier
                    cert = getattr(self.config, 'idp_certificate', None) or getattr(self.config, 'x509_cert', None)
                    if cert:
                        XMLVerifier().verify(assertion, x509_cert=cert)
                    else:
                        logger.warning("SAML signature verification skipped: no IdP certificate configured")
                except ImportError:
                    logger.warning(
                        "SAML signature verification unavailable: "
                        "install signxml (pip install signxml) for production use"
                    )
                except (ValueError, TypeError, KeyError, AttributeError) as sig_err:
                    logger.error(f"SAML signature verification failed: {sig_err}")
                    return SAMLAuthenticationResult(
                        success=False,
                        error=f"SAML signature verification failed: {sig_err}"
                    )
            
            # Extract NameID (user identifier)
            name_id_elem = assertion.find('.//saml:NameID', SAML_NAMESPACES)
            name_id = name_id_elem.text if name_id_elem is not None else None
            name_id_format = name_id_elem.get('Format', '') if name_id_elem is not None else ''
            
            # Extract attributes
            attribute_statement = assertion.find('.//saml:AttributeStatement', SAML_NAMESPACES)
            attributes = {}
            
            if attribute_statement is not None:
                for attr in attribute_statement.findall('.//saml:Attribute', SAML_NAMESPACES):
                    attr_name = attr.get('Name', '')
                    attr_values = []
                    for val in attr.findall('.//saml:AttributeValue', SAML_NAMESPACES):
                        attr_values.append(val.text)
                    attributes[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values
            
            # Extract session index
            authn_statement = assertion.find('.//saml:AuthnStatement', SAML_NAMESPACES)
            session_index = authn_statement.get('SessionIndex') if authn_statement is not None else None
            
            # Extract email from attributes
            email = attributes.get('email') or attributes.get('Email') or \
                    attributes.get('mail') or attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress')
            
            return SAMLAuthenticationResult(
                success=True,
                user_id=name_id,
                email=email,
                name_id=name_id,
                attributes=attributes,
                session_index=session_index
            )
            
        except (ET.ParseError, KeyError, ValueError, AttributeError) as e:
            logger.error(f"SAML response verification failed: {e}")
            return SAMLAuthenticationResult(
                success=False,
                error=f"Verification failed: {str(e)}"
            )
    
    def generate_logout_request(
        self,
        name_id: str,
        session_index: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate SAML Logout Request.
        
        Args:
            name_id: User's NameID
            session_index: Session index to terminate
            
        Returns:
            Tuple of (base64_encoded_request, request_id)
        """
        request_id = f"_{secrets.token_hex(16)}"
        issue_instant = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        logout_request = f"""<?xml version="1.0" encoding="UTF-8"?>
        <samlp:LogoutRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{issue_instant}"
            Destination="{self.config.slo_url}">
            <saml:Issuer>{self.config.sp_entity_id}</saml:Issuer>
            <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{name_id}</saml:NameID>
            {f'<samlp:SessionIndex>{session_index}</samlp:SessionIndex>' if session_index else ''}
        </samlp:LogoutRequest>"""
        
        encoded = base64.b64encode(logout_request.encode('utf-8')).decode('ascii')
        return encoded, request_id


class OIDCHandler:
    """
    OpenID Connect authentication handler.
    """
    
    def __init__(self, config: OIDCProviderConfig):
        """
        Initialize OIDC handler.
        
        Args:
            config: OIDC provider configuration
        """
        self.config = config
        self._jwks_cache: Optional[Dict] = None
        self._jwks_cache_time: Optional[datetime] = None
    
    def generate_auth_url(
        self,
        state: str,
        nonce: str,
        scopes: Optional[List[str]] = None,
        use_pkce: bool = True
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Generate authorization URL.
        
        Args:
            state: State parameter for CSRF protection
            nonce: Nonce for replay protection
            scopes: Requested scopes
            use_pkce: Whether to use PKCE
            
        Returns:
            Tuple of (auth_url, code_challenge, code_verifier)
        """
        code_verifier = None
        code_challenge = None
        
        params = {
            'response_type': 'code',
            'client_id': self.config.client_id,
            'redirect_uri': self.config.redirect_uri,
            'state': state,
            'nonce': nonce,
            'scope': ' '.join(scopes or self.config.default_scopes)
        }
        
        # Add PKCE if enabled
        if use_pkce and self.config.pkce_enabled:
            code_verifier = self._generate_code_verifier()
            code_challenge = self._generate_code_challenge(code_verifier)
            params['code_challenge'] = code_challenge
            params['code_challenge_method'] = 'S256'
        
        auth_url = f"{self.config.authorization_endpoint}?{urlencode(params)}"
        
        return auth_url, code_challenge, code_verifier
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge."""
        import hashlib
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    async def exchange_code(
        self,
        auth_code: str,
        code_verifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.
        
        Args:
            auth_code: Authorization code
            code_verifier: PKCE code verifier
            
        Returns:
            Token response
        """
        token_request = {
            'grant_type': 'authorization_code',
            'client_id': self.config.client_id,
            'code': auth_code,
            'redirect_uri': self.config.redirect_uri
        }
        
        if self.config.client_secret:
            token_request['client_secret'] = self.config.client_secret
        
        if code_verifier:
            token_request['code_verifier'] = code_verifier
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.token_endpoint,
                data=token_request,
                headers={'Accept': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Token request failed: {error_text}")
                
                return await response.json()
    
    async def validate_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate OIDC ID Token.
        
        Args:
            id_token: ID token to validate
            nonce: Expected nonce value
            
        Returns:
            Decoded token claims
        """
        # Get unverified header
        header = jwt.get_unverified_header(id_token)
        algorithm = header.get('alg', 'RS256')
        key_id = header.get('kid')
        
        # Fetch signing key
        signing_key = await self._get_signing_key(key_id)
        
        # Decode and validate
        claims = jwt.decode(
            id_token,
            signing_key,
            algorithms=[algorithm],
            audience=self.config.client_id,
            issuer=self.config.issuer
        )
        
        # Verify nonce
        if nonce and claims.get('nonce') != nonce:
            raise ValueError("Nonce mismatch")
        
        return claims
    
    async def _get_signing_key(self, key_id: Optional[str] = None) -> str:
        """Fetch signing key from JWKS endpoint."""
        if not self.config.jwks_uri:
            raise ValueError("JWKS URI not configured")
        
        # Check cache
        if (self._jwks_cache and self._jwks_cache_time and
            datetime.now() - self._jwks_cache_time < timedelta(hours=24)):
            jwks = self._jwks_cache
        else:
            # Fetch fresh JWKS
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.jwks_uri) as response:
                    jwks = await response.json()
                    self._jwks_cache = jwks
                    self._jwks_cache_time = datetime.now()
        
        # Find matching key
        for key in jwks.get('keys', []):
            if key_id is None or key.get('kid') == key_id:
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
        
        raise KeyError(f"Signing key not found: {key_id}")
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get user info from UserInfo endpoint.
        
        Args:
            access_token: Valid access token
            
        Returns:
            User info claims
        """
        if not self.config.userinfo_endpoint:
            return {}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.config.userinfo_endpoint,
                headers={'Authorization': f'Bearer {access_token}'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {}


class SSOSessionManager:
    """
    Manages SSO sessions across multiple identity providers.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize SSO session manager.
        
        Args:
            storage_path: Path for session storage
        """
        self.sessions: Dict[str, SSOSession] = {}
        self.storage_path = storage_path
    
    def create_session(
        self,
        idp_name: str,
        protocol: str,
        user_id: str,
        saml_assertion: Optional[str] = None,
        oidc_tokens: Optional[Dict] = None,
        ttl_hours: int = 8
    ) -> SSOSession:
        """
        Create new SSO session.
        
        Args:
            idp_name: Identity provider name
            protocol: Protocol used ('saml', 'oidc')
            user_id: User identifier
            saml_assertion: SAML assertion (if SAML)
            oidc_tokens: OIDC tokens (if OIDC)
            ttl_hours: Session TTL in hours
            
        Returns:
            New SSOSession
        """
        session_id = secrets.token_urlsafe(32)
        
        session = SSOSession(
            session_id=session_id,
            idp_name=idp_name,
            protocol=protocol,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=ttl_hours),
            saml_assertion=saml_assertion,
            oidc_tokens=oidc_tokens,
            service_sessions={}
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SSOSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SSOSession or None
        """
        session = self.sessions.get(session_id)
        
        if session and session.expires_at < datetime.now():
            # Session expired
            del self.sessions[session_id]
            return None
        
        return session
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate SSO session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was found and terminated
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_service_token(
        self,
        session_id: str,
        service_name: str
    ) -> Optional[str]:
        """
        Get service-specific token from SSO session.
        
        Args:
            session_id: SSO session ID
            service_name: Service name
            
        Returns:
            Service token or None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        service_session = session.service_sessions.get(service_name)
        if service_session:
            return service_session.get('token')
        
        return None


class SSOHandler:
    """
    Main SSO handler coordinating SAML and OIDC.
    """
    
    # Pre-configured IdP templates
    IDP_TEMPLATES = {
        'azure_ad': {
            'protocol': 'oidc',
            'issuer': 'https://login.microsoftonline.com/{tenant}/v2.0',
            'authorization_endpoint': 'https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize',
            'token_endpoint': 'https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token',
            'userinfo_endpoint': 'https://graph.microsoft.com/oidc/userinfo',
            'jwks_uri': 'https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys',
            'scopes': ['openid', 'profile', 'email', 'User.Read']
        },
        'okta': {
            'protocol': 'oidc',
            'issuer': 'https://{domain}.okta.com',
            'authorization_endpoint': 'https://{domain}.okta.com/oauth2/v1/authorize',
            'token_endpoint': 'https://{domain}.okta.com/oauth2/v1/token',
            'userinfo_endpoint': 'https://{domain}.okta.com/oauth2/v1/userinfo',
            'jwks_uri': 'https://{domain}.okta.com/oauth2/v1/keys',
            'scopes': ['openid', 'profile', 'email']
        },
        'google_workspace': {
            'protocol': 'oidc',
            'issuer': 'https://accounts.google.com',
            'authorization_endpoint': 'https://accounts.google.com/o/oauth2/v2/auth',
            'token_endpoint': 'https://oauth2.googleapis.com/token',
            'userinfo_endpoint': 'https://openidconnect.googleapis.com/v1/userinfo',
            'jwks_uri': 'https://www.googleapis.com/oauth2/v3/certs',
            'scopes': ['openid', 'email', 'profile']
        },
        'auth0': {
            'protocol': 'oidc',
            'issuer': 'https://{domain}.auth0.com/',
            'authorization_endpoint': 'https://{domain}.auth0.com/authorize',
            'token_endpoint': 'https://{domain}.auth0.com/oauth/token',
            'userinfo_endpoint': 'https://{domain}.auth0.com/userinfo',
            'jwks_uri': 'https://{domain}.auth0.com/.well-known/jwks.json',
            'scopes': ['openid', 'profile', 'email']
        }
    }
    
    def __init__(self):
        """Initialize SSO handler."""
        self.saml_handlers: Dict[str, SAMLHandler] = {}
        self.oidc_handlers: Dict[str, OIDCHandler] = {}
        self.session_manager = SSOSessionManager()
    
    def register_saml_provider(self, config: SAMLProviderConfig) -> None:
        """Register SAML Identity Provider."""
        self.saml_handlers[config.name] = SAMLHandler(config)
    
    def register_oidc_provider(self, config: OIDCProviderConfig) -> None:
        """Register OIDC Identity Provider."""
        self.oidc_handlers[config.name] = OIDCHandler(config)
    
    def register_from_template(
        self,
        name: str,
        template: str,
        client_id: str,
        client_secret: Optional[str] = None,
        **template_params
    ) -> None:
        """
        Register provider from predefined template.
        
        Args:
            name: Provider name
            template: Template name
            client_id: OAuth client ID
            client_secret: OAuth client secret
            **template_params: Template-specific parameters
        """
        if template not in self.IDP_TEMPLATES:
            raise ValueError(f"Unknown template: {template}")
        
        template_data = self.IDP_TEMPLATES[template].copy()
        
        # Format URLs with template params
        for key in ['issuer', 'authorization_endpoint', 'token_endpoint',
                   'userinfo_endpoint', 'jwks_uri']:
            if key in template_data:
                template_data[key] = template_data[key].format(**template_params)
        
        config = OIDCProviderConfig(
            name=name,
            client_id=client_id,
            client_secret=client_secret,
            scopes_supported=template_data.get('scopes', []),
            **{k: v for k, v in template_data.items() 
               if k not in ['protocol', 'scopes']}
        )
        
        self.register_oidc_provider(config)
    
    async def authenticate_saml(
        self,
        provider_name: str,
        browser_context,
        force_authn: bool = False
    ) -> SAMLAuthenticationResult:
        """
        Authenticate using SAML.
        
        Args:
            provider_name: SAML provider name
            browser_context: Playwright browser context
            force_authn: Force re-authentication
            
        Returns:
            SAMLAuthenticationResult
        """
        handler = self.saml_handlers.get(provider_name)
        if not handler:
            return SAMLAuthenticationResult(
                success=False,
                error=f"SAML provider not found: {provider_name}"
            )
        
        # Generate AuthnRequest
        authn_request, request_id = handler.generate_authn_request(force_authn)
        
        # Build login URL
        relay_state = secrets.token_urlsafe(16)
        login_url = handler.build_login_url(authn_request, relay_state)
        
        # Navigate to IdP
        page = await browser_context.new_page()
        
        try:
            await page.goto(login_url)
            
            # Wait for SAML Response (would need callback server in real implementation)
            # For now, simplified flow
            
            # This would be replaced with actual SAML response handling
            return SAMLAuthenticationResult(
                success=False,
                error="SAML browser flow requires callback server implementation"
            )
            
        finally:
            await page.close()
    
    async def authenticate_oidc(
        self,
        provider_name: str,
        browser_context,
        scopes: Optional[List[str]] = None
    ) -> OIDCAuthenticationResult:
        """
        Authenticate using OIDC.
        
        Args:
            provider_name: OIDC provider name
            browser_context: Playwright browser context
            scopes: Requested scopes
            
        Returns:
            OIDCAuthenticationResult
        """
        handler = self.oidc_handlers.get(provider_name)
        if not handler:
            return OIDCAuthenticationResult(
                success=False,
                error=f"OIDC provider not found: {provider_name}"
            )
        
        # Generate state and nonce
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        
        # Generate authorization URL
        auth_url, code_challenge, code_verifier = handler.generate_auth_url(
            state=state,
            nonce=nonce,
            scopes=scopes
        )
        
        # Start callback server
        from .oauth_handler import CallbackServer
        callback_server = CallbackServer()
        await callback_server.start()
        
        try:
            # Navigate to authorization endpoint
            page = await browser_context.new_page()
            await page.goto(auth_url)
            
            # Wait for callback
            response = await callback_server.wait_for_response(timeout=300)
            
            # Verify state
            if response.get('state') != state:
                return OIDCAuthenticationResult(
                    success=False,
                    error="State mismatch - possible CSRF attack"
                )
            
            # Exchange code for tokens
            auth_code = response.get('code')
            if not auth_code:
                return OIDCAuthenticationResult(
                    success=False,
                    error="No authorization code received"
                )
            
            token_response = await handler.exchange_code(auth_code, code_verifier)
            
            # Validate ID token
            id_token = token_response.get('id_token')
            if id_token:
                try:
                    id_claims = await handler.validate_id_token(id_token, nonce)
                except (jwt.InvalidTokenError, ValueError, KeyError) as e:
                    return OIDCAuthenticationResult(
                        success=False,
                        error=f"ID token validation failed: {e}"
                    )
            else:
                id_claims = {}
            
            # Get user info
            access_token = token_response.get('access_token')
            user_info = await handler.get_user_info(access_token) if access_token else {}
            
            # Create SSO session
            user_id = id_claims.get('sub') or user_info.get('sub')
            session = self.session_manager.create_session(
                idp_name=provider_name,
                protocol='oidc',
                user_id=user_id,
                oidc_tokens=token_response
            )
            
            return OIDCAuthenticationResult(
                success=True,
                user_id=user_id,
                email=id_claims.get('email') or user_info.get('email'),
                id_token=id_token,
                access_token=access_token,
                refresh_token=token_response.get('refresh_token'),
                user_info={**id_claims, **user_info}
            )
            
        except (aiohttp.ClientError, OSError, ValueError, KeyError) as e:
            logger.error(f"OIDC authentication failed: {e}")
            return OIDCAuthenticationResult(
                success=False,
                error=str(e)
            )
        finally:
            await callback_server.stop()
            await page.close()
    
    def get_session(self, session_id: str) -> Optional[SSOSession]:
        """Get SSO session."""
        return self.session_manager.get_session(session_id)
    
    def logout(self, session_id: str) -> bool:
        """Logout from SSO session."""
        return self.session_manager.terminate_session(session_id)


# Convenience functions
def create_saml_handler(config: SAMLProviderConfig) -> SAMLHandler:
    """Create SAML handler."""
    return SAMLHandler(config)


def create_oidc_handler(config: OIDCProviderConfig) -> OIDCHandler:
    """Create OIDC handler."""
    return OIDCHandler(config)


def create_sso_handler() -> SSOHandler:
    """Create SSO handler."""
    return SSOHandler()
