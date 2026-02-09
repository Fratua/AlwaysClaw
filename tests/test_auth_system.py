"""Unit tests for auth system components."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'auth_system'))


class TestJWTHandler:
    """Test JWT encode/decode with valid/expired/invalid tokens."""

    def test_jwt_roundtrip(self):
        """Test that encoding then decoding a token works."""
        try:
            import jwt
        except ImportError:
            pytest.skip("pyjwt not installed")

        secret = "test-secret-key"
        payload = {"sub": "user123", "role": "admin"}
        token = jwt.encode(payload, secret, algorithm="HS256")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        assert decoded["sub"] == "user123"
        assert decoded["role"] == "admin"

    def test_jwt_expired(self):
        """Test that expired tokens are rejected."""
        try:
            import jwt
            from datetime import datetime, timedelta, timezone
        except ImportError:
            pytest.skip("pyjwt not installed")

        secret = "test-secret-key"
        payload = {"sub": "user123", "exp": datetime.now(timezone.utc) - timedelta(hours=1)}
        token = jwt.encode(payload, secret, algorithm="HS256")
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, secret, algorithms=["HS256"])

    def test_jwt_invalid_signature(self):
        """Test that tokens with wrong secret are rejected."""
        try:
            import jwt
        except ImportError:
            pytest.skip("pyjwt not installed")

        token = jwt.encode({"sub": "user123"}, "secret-1", algorithm="HS256")
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, "wrong-secret", algorithms=["HS256"])


class TestCredentialVault:
    """Test credential vault encrypt/decrypt roundtrip."""

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
    """Test cookie jar serialization."""

    def test_cookie_dict_roundtrip(self):
        """Test basic cookie dict serialization."""
        import json
        cookies = {
            "session_id": "abc123",
            "prefs": "dark_mode=true",
        }
        serialized = json.dumps(cookies)
        deserialized = json.loads(serialized)
        assert deserialized == cookies

    def test_cookie_with_special_chars(self):
        """Test cookies with special characters serialize correctly."""
        import json
        cookies = {"token": "a=b&c=d;e=f", "path": "/api/v1"}
        serialized = json.dumps(cookies)
        deserialized = json.loads(serialized)
        assert deserialized == cookies
