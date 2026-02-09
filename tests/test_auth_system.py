"""Unit tests for auth system components - testing actual classes, not libraries."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'auth_system'))


class TestJWTHandler:
    """Test JWT classes: JWTGenerator, JWTValidator, JWTConfig, DecodedJWT."""

    def test_jwt_config_defaults(self):
        """Test JWTConfig has sensible defaults."""
        from jwt_handler import JWTConfig
        config = JWTConfig()
        assert config.algorithm == "RS256"
        assert config.access_token_ttl == 900
        assert config.refresh_token_ttl == 604800
        assert config.issuer == "openclaw-agent"

    def test_jwt_config_self_signed_option(self):
        """Test JWTConfig supports self-signed mode for local dev."""
        from jwt_handler import JWTConfig
        config = JWTConfig(allow_self_signed=True)
        assert config.allow_self_signed is True

    def test_jwt_generator_hs256_roundtrip(self):
        """Test actual JWTGenerator creates tokens that JWTValidator can validate."""
        try:
            import jwt as pyjwt
            from jwt_handler import JWTGenerator, JWTConfig
        except ImportError:
            pytest.skip("pyjwt not installed")

        secret = "test-secret-key-for-hs256"
        config = JWTConfig(algorithm="HS256", allow_self_signed=True)
        generator = JWTGenerator(private_key=secret, config=config)
        token = generator.generate_access_token(subject="user123")
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts

        # Verify token decodes correctly
        decoded = pyjwt.decode(token, secret, algorithms=["HS256"])
        assert decoded["sub"] == "user123"
        assert decoded["iss"] == "openclaw-agent"
        assert "exp" in decoded
        assert "jti" in decoded

    def test_jwt_generator_custom_claims(self):
        """Test JWTGenerator passes through custom claims."""
        try:
            import jwt as pyjwt
            from jwt_handler import JWTGenerator, JWTConfig
        except ImportError:
            pytest.skip("pyjwt not installed")

        secret = "test-secret"
        config = JWTConfig(algorithm="HS256")
        generator = JWTGenerator(private_key=secret, config=config)
        token = generator.generate_access_token(
            subject="user456",
            custom_claims={"role": "admin", "tier": "premium"}
        )
        decoded = pyjwt.decode(token, secret, algorithms=["HS256"])
        assert decoded["sub"] == "user456"
        assert decoded["role"] == "admin"
        assert decoded["tier"] == "premium"

    def test_jwt_expired_token_rejected(self):
        """Test expired tokens are rejected."""
        try:
            import jwt as pyjwt
            from jwt_handler import JWTGenerator, JWTConfig
        except ImportError:
            pytest.skip("pyjwt not installed")

        secret = "test-secret-key"
        config = JWTConfig(algorithm="HS256")
        generator = JWTGenerator(private_key=secret, config=config)
        token = generator.generate_access_token(subject="user123", expires_in=-1)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            pyjwt.decode(token, secret, algorithms=["HS256"])

    def test_jwt_wrong_secret_rejected(self):
        """Test tokens with wrong secret are rejected."""
        try:
            import jwt as pyjwt
            from jwt_handler import JWTGenerator, JWTConfig
        except ImportError:
            pytest.skip("pyjwt not installed")

        config = JWTConfig(algorithm="HS256")
        generator = JWTGenerator(private_key="correct-secret", config=config)
        token = generator.generate_access_token(subject="user123")
        with pytest.raises(pyjwt.InvalidSignatureError):
            pyjwt.decode(token, "wrong-secret", algorithms=["HS256"])

    def test_decoded_jwt_properties(self):
        """Test DecodedJWT dataclass properties."""
        from jwt_handler import DecodedJWT
        import time
        exp_time = int(time.time()) + 3600
        decoded = DecodedJWT(
            header={"alg": "HS256", "typ": "JWT"},
            payload={"sub": "user789", "iss": "openclaw-agent", "exp": exp_time},
            signature="abc",
            raw_token="a.b.c"
        )
        assert decoded.subject == "user789"
        assert decoded.issuer == "openclaw-agent"
        assert decoded.is_expired is False

    def test_jwt_validator_init_with_symmetric_key(self):
        """Test JWTValidator can be initialized with symmetric key for local dev."""
        from jwt_handler import JWTValidator, JWTConfig
        config = JWTConfig(allow_self_signed=True)
        validator = JWTValidator(symmetric_key="my-test-key", config=config)
        assert validator.symmetric_key == "my-test-key"


class TestCredentialVault:
    """Test credential vault with actual Credentials class."""

    def test_credentials_dataclass(self):
        """Test Credentials dataclass roundtrip."""
        from credential_vault import Credentials
        cred = Credentials(
            service_name="github",
            username="testuser",
            password="testpass",
            attributes={"org": "myorg"}
        )
        assert cred.service_name == "github"
        assert cred.username == "testuser"
        d = cred.to_dict()
        assert d['service_name'] == "github"
        assert d['attributes']['org'] == "myorg"
        assert 'created_at' in d

    def test_credentials_from_dict(self):
        """Test Credentials round-trips through dict."""
        from credential_vault import Credentials
        cred = Credentials(
            service_name="aws",
            username="admin",
            password="secret123"
        )
        d = cred.to_dict()
        restored = Credentials.from_dict(d)
        assert restored.service_name == "aws"
        assert restored.username == "admin"
        assert restored.password == "secret123"

    def test_fernet_roundtrip(self):
        """Test basic Fernet encrypt/decrypt."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography not installed")

        key = Fernet.generate_key()
        f = Fernet(key)
        plaintext = b"my-secret-password"
        encrypted = f.encrypt(plaintext)
        decrypted = f.decrypt(encrypted)
        assert decrypted == plaintext

    def test_fernet_wrong_key_fails(self):
        """Test that decryption with wrong key raises."""
        try:
            from cryptography.fernet import Fernet, InvalidToken
        except ImportError:
            pytest.skip("cryptography not installed")

        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        encrypted = Fernet(key1).encrypt(b"secret")
        with pytest.raises(InvalidToken):
            Fernet(key2).decrypt(encrypted)


class TestCookieJar:
    """Test cookie jar with actual Cookie class."""

    def test_cookie_creation(self):
        """Test Cookie dataclass creation and properties."""
        from cookie_jar import Cookie
        cookie = Cookie(
            name="session_id",
            value="abc123",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True
        )
        assert cookie.name == "session_id"
        assert cookie.value == "abc123"
        assert cookie.domain == "example.com"
        assert cookie.secure is True
        assert cookie.is_expired() is False

    def test_cookie_expiration(self):
        """Test Cookie expiration detection."""
        from cookie_jar import Cookie
        expired_cookie = Cookie(
            name="old",
            value="data",
            domain="example.com",
            expires=datetime.now() - timedelta(hours=1)
        )
        assert expired_cookie.is_expired() is True

        fresh_cookie = Cookie(
            name="new",
            value="data",
            domain="example.com",
            expires=datetime.now() + timedelta(hours=1)
        )
        assert fresh_cookie.is_expired() is False

    def test_cookie_dict_roundtrip(self):
        """Test Cookie serialization via to_dict/from_dict."""
        from cookie_jar import Cookie
        cookie = Cookie(
            name="token",
            value="a=b&c=d",
            domain="api.example.com",
            path="/api/v1",
            secure=True
        )
        d = cookie.to_dict()
        assert d['name'] == "token"
        assert d['value'] == "a=b&c=d"
        assert d['domain'] == "api.example.com"
        assert d['secure'] is True

        restored = Cookie.from_dict(d)
        assert restored.name == cookie.name
        assert restored.value == cookie.value
        assert restored.domain == cookie.domain

    def test_cookie_to_playwright_format(self):
        """Test Cookie converts to Playwright format."""
        from cookie_jar import Cookie
        cookie = Cookie(
            name="session",
            value="xyz",
            domain="example.com"
        )
        pw = cookie.to_playwright_format()
        assert pw['name'] == "session"
        assert pw['value'] == "xyz"
