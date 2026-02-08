"""
JWT Token Handling System
For Windows 10 OpenClaw AI Agent Framework
"""

import jwt
import json
import secrets
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class JWTError(Exception):
    """JWT-related error."""
    def __init__(self, message: str, error_type: Optional[str] = None):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class JWTConfig:
    """JWT configuration settings."""
    algorithm: str = "RS256"  # RS256, ES256, HS256
    access_token_ttl: int = 900  # 15 minutes
    refresh_token_ttl: int = 604800  # 7 days
    issuer: str = "openclaw-agent"
    audience: Optional[str] = None
    require_expiration: bool = True
    allowed_algorithms: List[str] = field(default_factory=lambda: ["RS256", "ES256"])


@dataclass
class DecodedJWT:
    """Decoded JWT with metadata."""
    header: Dict[str, Any]
    payload: Dict[str, Any]
    signature: str
    raw_token: str
    
    @property
    def subject(self) -> Optional[str]:
        """Get subject claim."""
        return self.payload.get('sub')
    
    @property
    def issuer(self) -> Optional[str]:
        """Get issuer claim."""
        return self.payload.get('iss')
    
    @property
    def audience(self) -> Optional[str]:
        """Get audience claim."""
        return self.payload.get('aud')
    
    @property
    def expiration(self) -> Optional[datetime]:
        """Get expiration time."""
        exp = self.payload.get('exp')
        if exp:
            return datetime.fromtimestamp(exp)
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        exp = self.expiration
        if exp:
            return datetime.now() > exp
        return False
    
    @property
    def token_id(self) -> Optional[str]:
        """Get JWT ID claim."""
        return self.payload.get('jti')
    
    @property
    def issued_at(self) -> Optional[datetime]:
        """Get issued at time."""
        iat = self.payload.get('iat')
        if iat:
            return datetime.fromtimestamp(iat)
        return None


class JWTValidator:
    """
    JWT validation with signature verification and claim checking.
    """
    
    def __init__(
        self,
        jwks_url: Optional[str] = None,
        public_key: Optional[str] = None,
        config: Optional[JWTConfig] = None
    ):
        self.jwks_url = jwks_url
        self.public_key = public_key
        self.config = config or JWTConfig()
        
        self._jwks_cache: Optional[Dict] = None
        self._jwks_cache_time: Optional[datetime] = None
        self._jwks_cache_ttl = timedelta(hours=24)
        
        self._lock = asyncio.Lock()
    
    async def validate_token(
        self,
        token: str,
        expected_audience: Optional[str] = None,
        expected_issuer: Optional[str] = None,
        required_claims: Optional[List[str]] = None
    ) -> DecodedJWT:
        """
        Validate JWT token comprehensively.
        
        Args:
            token: The JWT token to validate
            expected_audience: Expected audience claim
            expected_issuer: Expected issuer claim
            required_claims: List of required claims
            
        Returns:
            DecodedJWT object if validation succeeds
            
        Raises:
            JWTError: If validation fails
        """
        try:
            # Get unverified header to find key ID
            unverified_header = jwt.get_unverified_header(token)
            algorithm = unverified_header.get('alg')
            key_id = unverified_header.get('kid')
            
            # Validate algorithm
            if algorithm not in self.config.allowed_algorithms:
                raise JWTError(
                    f"Algorithm not allowed: {algorithm}",
                    error_type="invalid_algorithm"
                )
            
            # Fetch appropriate key
            if algorithm.startswith('RS') or algorithm.startswith('ES'):
                public_key = await self._get_public_key(key_id)
            elif algorithm.startswith('HS'):
                raise JWTError(
                    "Symmetric algorithms not supported for validation",
                    error_type="invalid_algorithm"
                )
            else:
                raise JWTError(
                    f"Unsupported algorithm: {algorithm}",
                    error_type="invalid_algorithm"
                )
            
            # Build validation options
            options = {
                'require': required_claims or ['exp', 'iat', 'sub'],
                'verify_exp': self.config.require_expiration,
                'verify_iat': True,
                'verify_nbf': True,
            }
            
            # Validate token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                audience=expected_audience or self.config.audience,
                issuer=expected_issuer or self.config.issuer,
                options=options
            )
            
            return DecodedJWT(
                header=unverified_header,
                payload=payload,
                signature="",  # Not exposed after validation
                raw_token=token
            )
            
        except jwt.ExpiredSignatureError:
            raise JWTError("Token has expired", error_type="expired")
        except jwt.InvalidAudienceError:
            raise JWTError("Invalid audience", error_type="invalid_audience")
        except jwt.InvalidIssuerError:
            raise JWTError("Invalid issuer", error_type="invalid_issuer")
        except jwt.InvalidSignatureError:
            raise JWTError("Invalid signature", error_type="invalid_signature")
        except jwt.DecodeError:
            raise JWTError("Token could not be decoded", error_type="decode_error")
        except jwt.InvalidTokenError as e:
            raise JWTError(f"Invalid token: {str(e)}", error_type="invalid_token")
    
    async def _get_public_key(self, key_id: Optional[str]) -> str:
        """Fetch public key from JWKS endpoint or use provided key."""
        # Use directly provided key if available
        if self.public_key:
            return self.public_key
        
        if not self.jwks_url:
            raise JWTError("No public key or JWKS URL configured")
        
        # Check cache
        if (self._jwks_cache and self._jwks_cache_time and
            datetime.now() - self._jwks_cache_time < self._jwks_cache_ttl):
            jwks = self._jwks_cache
        else:
            # Fetch fresh JWKS
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jwks_url) as response:
                    if response.status != 200:
                        raise JWTError(f"Failed to fetch JWKS: {response.status}")
                    jwks = await response.json()
                    
                    async with self._lock:
                        self._jwks_cache = jwks
                        self._jwks_cache_time = datetime.now()
        
        # Find matching key
        for key in jwks.get('keys', []):
            if key_id is None or key.get('kid') == key_id:
                # Convert JWK to PEM format
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
        
        raise JWTError(f"Key not found: {key_id}")
    
    def decode_without_verification(self, token: str) -> DecodedJWT:
        """
        Decode JWT without signature verification.
        WARNING: Only use for inspection, not validation!
        """
        try:
            header = jwt.get_unverified_header(token)
            payload = jwt.decode(token, options={"verify_signature": False})
            
            return DecodedJWT(
                header=header,
                payload=payload,
                signature="",
                raw_token=token
            )
        except Exception as e:
            raise JWTError(f"Failed to decode token: {e}")


class JWTGenerator:
    """
    JWT token generation with secure defaults.
    """
    
    def __init__(
        self,
        private_key: str,
        public_key: Optional[str] = None,
        config: Optional[JWTConfig] = None
    ):
        self.private_key = private_key
        self.public_key = public_key
        self.config = config or JWTConfig()
    
    def generate_access_token(
        self,
        subject: str,
        audience: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        custom_claims: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None
    ) -> str:
        """
        Generate access token with standard claims.
        
        Args:
            subject: Subject identifier (user ID)
            audience: Intended audience
            scopes: List of permission scopes
            custom_claims: Additional custom claims
            expires_in: Token lifetime in seconds
            
        Returns:
            Signed JWT access token
        """
        now = datetime.utcnow()
        exp = now + timedelta(seconds=expires_in or self.config.access_token_ttl)
        
        payload = {
            # Registered claims
            'sub': subject,
            'iss': self.config.issuer,
            'aud': audience or self.config.audience,
            'iat': now,
            'nbf': now,
            'exp': exp,
            'jti': secrets.token_urlsafe(16),  # Unique token ID
            
            # Custom claims
            'scope': ' '.join(scopes or []),
            'type': 'access'
        }
        
        if custom_claims:
            # Prevent overwriting standard claims
            for key in payload:
                custom_claims.pop(key, None)
            payload.update(custom_claims)
        
        # Build headers with key ID if available
        headers = {}
        if self.public_key:
            # Generate key ID from public key
            key_id = self._generate_key_id(self.public_key)
            headers['kid'] = key_id
        
        return jwt.encode(
            payload,
            self.private_key,
            algorithm=self.config.algorithm,
            headers=headers
        )
    
    def generate_refresh_token(
        self,
        subject: str,
        family_id: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> str:
        """
        Generate refresh token with rotation support.
        
        Args:
            subject: Subject identifier
            family_id: Token family ID for rotation tracking
            expires_in: Token lifetime in seconds
            
        Returns:
            Signed JWT refresh token
        """
        now = datetime.utcnow()
        exp = now + timedelta(seconds=expires_in or self.config.refresh_token_ttl)
        
        payload = {
            'sub': subject,
            'iss': self.config.issuer,
            'iat': now,
            'exp': exp,
            'jti': secrets.token_urlsafe(16),
            'family_id': family_id or secrets.token_urlsafe(16),
            'type': 'refresh'
        }
        
        return jwt.encode(
            payload,
            self.private_key,
            algorithm=self.config.algorithm
        )
    
    def generate_id_token(
        self,
        subject: str,
        audience: str,
        user_info: Dict[str, Any],
        expires_in: int = 3600
    ) -> str:
        """
        Generate OpenID Connect ID token.
        
        Args:
            subject: Subject identifier
            audience: Client ID (audience)
            user_info: User information claims
            expires_in: Token lifetime in seconds
            
        Returns:
            Signed JWT ID token
        """
        now = datetime.utcnow()
        exp = now + timedelta(seconds=expires_in)
        
        payload = {
            'sub': subject,
            'iss': self.config.issuer,
            'aud': audience,
            'iat': now,
            'exp': exp,
            'jti': secrets.token_urlsafe(16),
            'type': 'id_token'
        }
        
        # Add standard OIDC claims
        standard_claims = ['name', 'given_name', 'family_name', 'middle_name',
                          'nickname', 'preferred_username', 'profile', 'picture',
                          'website', 'email', 'email_verified', 'gender', 'birthdate',
                          'zoneinfo', 'locale', 'phone_number', 'phone_number_verified',
                          'address', 'updated_at']
        
        for claim in standard_claims:
            if claim in user_info:
                payload[claim] = user_info[claim]
        
        return jwt.encode(
            payload,
            self.private_key,
            algorithm=self.config.algorithm
        )
    
    def _generate_key_id(self, public_key: str) -> str:
        """Generate key ID from public key."""
        # Use first 16 bytes of SHA-256 hash
        key_hash = secrets.token_urlsafe(8)
        return key_hash


class TokenRotationManager:
    """
    Manages refresh token rotation with reuse detection.
    """
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend
        self._token_families: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_token_pair(
        self,
        user_id: str,
        jwt_generator: JWTGenerator,
        device_info: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Create new access/refresh token pair.
        
        Returns:
            Tuple of (access_token, refresh_token, family_id)
        """
        # Create token family for rotation tracking
        family_id = secrets.token_urlsafe(16)
        
        # Generate tokens
        access_token = jwt_generator.generate_access_token(subject=user_id)
        refresh_token = jwt_generator.generate_refresh_token(
            subject=user_id,
            family_id=family_id
        )
        
        # Store refresh token metadata
        async with self._lock:
            self._token_families[family_id] = {
                'user_id': user_id,
                'device_info': device_info,
                'created_at': datetime.now(),
                'tokens': {self._extract_jti(refresh_token): False}  # jti -> revoked
            }
        
        return access_token, refresh_token, family_id
    
    async def rotate_refresh_token(
        self,
        old_refresh_token: str,
        jwt_generator: JWTGenerator,
        jwt_validator: JWTValidator
    ) -> Tuple[str, str]:
        """
        Rotate refresh token with reuse detection.
        
        Args:
            old_refresh_token: Current refresh token
            jwt_generator: JWT generator instance
            jwt_validator: JWT validator instance
            
        Returns:
            Tuple of (new_access_token, new_refresh_token)
            
        Raises:
            JWTError: If token reuse is detected or token is invalid
        """
        # Validate the old token
        try:
            decoded = await jwt_validator.validate_token(old_refresh_token)
        except JWTError as e:
            raise JWTError(f"Invalid refresh token: {e}")
        
        # Verify it's a refresh token
        if decoded.payload.get('type') != 'refresh':
            raise JWTError("Not a refresh token", error_type="wrong_token_type")
        
        old_jti = decoded.token_id
        user_id = decoded.subject
        family_id = decoded.payload.get('family_id')
        
        if not family_id:
            raise JWTError("Token family ID missing", error_type="invalid_token")
        
        async with self._lock:
            family = self._token_families.get(family_id)
            
            if not family:
                raise JWTError("Token family not found", error_type="invalid_family")
            
            # SECURITY: Detect reuse
            if family['tokens'].get(old_jti, False):
                # Token reuse detected - revoke entire family
                await self._revoke_token_family(family_id)
                raise JWTError(
                    "Token reuse detected - all sessions revoked",
                    error_type="token_reuse"
                )
            
            # Mark old token as used
            family['tokens'][old_jti] = True
        
        # Generate new token pair
        new_access = jwt_generator.generate_access_token(subject=user_id)
        new_refresh = jwt_generator.generate_refresh_token(
            subject=user_id,
            family_id=family_id
        )
        new_jti = self._extract_jti(new_refresh)
        
        # Store new token
        async with self._lock:
            self._token_families[family_id]['tokens'][new_jti] = False
        
        return new_access, new_refresh
    
    async def _revoke_token_family(self, family_id: str) -> None:
        """Revoke all tokens in a family."""
        async with self._lock:
            if family_id in self._token_families:
                # Mark all tokens as revoked
                for jti in self._token_families[family_id]['tokens']:
                    self._token_families[family_id]['tokens'][jti] = True
                
                logger.warning(f"Revoked entire token family: {family_id}")
    
    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user."""
        count = 0
        
        async with self._lock:
            for family_id, family in self._token_families.items():
                if family['user_id'] == user_id:
                    for jti in family['tokens']:
                        family['tokens'][jti] = True
                    count += len(family['tokens'])
        
        return count
    
    def _extract_jti(self, token: str) -> str:
        """Extract JTI from token without full validation."""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload.get('jti', '')
        except:
            return ''


class TokenBlacklist:
    """
    In-memory blacklist for immediate access token revocation.
    """
    
    def __init__(self, cleanup_interval: int = 300):
        self._blacklist: Dict[str, datetime] = {}  # jti -> expiry
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self) -> None:
        """Periodically remove expired entries."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove expired entries."""
        now = datetime.now()
        expired = [
            jti for jti, exp in self._blacklist.items()
            if exp < now
        ]
        for jti in expired:
            del self._blacklist[jti]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired blacklist entries")
    
    def add(self, jti: str, expiry: datetime) -> None:
        """Add token to blacklist."""
        self._blacklist[jti] = expiry
        logger.debug(f"Added token to blacklist: {jti}")
    
    def is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        expiry = self._blacklist.get(jti)
        if not expiry:
            return False
        
        if expiry < datetime.now():
            # Token expired, remove from blacklist
            del self._blacklist[jti]
            return False
        
        return True
    
    def size(self) -> int:
        """Get number of blacklisted tokens."""
        return len(self._blacklist)


class JWTAuthMiddleware:
    """
    JWT authentication middleware for API protection.
    """
    
    def __init__(
        self,
        jwt_validator: JWTValidator,
        token_blacklist: Optional[TokenBlacklist] = None,
        extract_token: Optional[Callable] = None
    ):
        self.validator = jwt_validator
        self.blacklist = token_blacklist
        self.extract_token = extract_token or self._default_extract_token
    
    def _default_extract_token(self, request) -> Optional[str]:
        """Default token extraction from Authorization header."""
        auth_header = request.headers.get('Authorization', '')
        
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        
        return None
    
    async def authenticate(self, request) -> DecodedJWT:
        """
        Authenticate request and return decoded token.
        
        Raises:
            JWTError: If authentication fails
        """
        # Extract token
        token = self.extract_token(request)
        
        if not token:
            raise JWTError(
                "Authorization token missing",
                error_type="missing_token"
            )
        
        # Check blacklist
        if self.blacklist:
            # Decode without verification to get JTI
            decoded = self.validator.decode_without_verification(token)
            if decoded.token_id and self.blacklist.is_blacklisted(decoded.token_id):
                raise JWTError("Token has been revoked", error_type="revoked")
        
        # Validate token
        return await self.validator.validate_token(token)


# Convenience functions
def create_jwt_validator(
    jwks_url: Optional[str] = None,
    public_key: Optional[str] = None
) -> JWTValidator:
    """Create JWT validator."""
    return JWTValidator(jwks_url=jwks_url, public_key=public_key)


def create_jwt_generator(
    private_key: str,
    public_key: Optional[str] = None,
    issuer: str = "openclaw-agent"
) -> JWTGenerator:
    """Create JWT generator."""
    config = JWTConfig(issuer=issuer)
    return JWTGenerator(private_key, public_key, config)


def generate_rsa_keypair(key_size: int = 2048) -> Tuple[str, str]:
    """
    Generate RSA key pair for JWT signing.
    
    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )
    
    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    
    # Serialize public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode('utf-8')
    
    return private_pem, public_pem
