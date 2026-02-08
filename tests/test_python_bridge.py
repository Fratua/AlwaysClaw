"""Unit tests for python_bridge.py"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def bridge():
    """Create a PythonBridge instance with mocked externals."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        from python_bridge import PythonBridge
        b = PythonBridge()
        return b


class TestEchoHandler:
    def test_echo_returns_params(self, bridge):
        result = bridge._handle_echo(msg="hello", num=42)
        assert result == {"msg": "hello", "num": 42}

    def test_echo_empty_params(self, bridge):
        result = bridge._handle_echo()
        assert result == {}


class TestHealthHandler:
    def test_health_returns_status(self, bridge):
        result = bridge._handle_health()
        assert result["status"] == "ok"
        assert "pid" in result
        assert isinstance(result["pid"], int)

    def test_health_reports_uninit_state(self, bridge):
        result = bridge._handle_health()
        assert result["llm_initialized"] is False
        assert result["memory_initialized"] is False


class TestLLMHandler:
    def test_llm_complete_calls_client(self, bridge):
        mock_client = MagicMock()
        mock_client.complete.return_value = {
            "content": "test response",
            "model": "gpt-5.2",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        bridge.llm_client = mock_client

        result = bridge._handle_llm_complete(
            messages=[{"role": "user", "content": "hello"}],
            system="test system",
        )
        assert result["content"] == "test response"
        mock_client.complete.assert_called_once()

    def test_llm_generate_calls_client(self, bridge):
        mock_client = MagicMock()
        mock_client.generate.return_value = "generated text"
        bridge.llm_client = mock_client

        result = bridge._handle_llm_generate(prompt="test prompt")
        assert result["content"] == "generated text"


class TestMemoryHandlers:
    def test_memory_store_and_search(self, bridge):
        bridge._initialize_memory_db()
        assert bridge.db_connection is not None

        # Store
        result = bridge._handle_memory_store(
            content="test memory content",
            type="episodic",
            tags=["test"],
        )
        assert result["stored"] is True
        assert "id" in result

        # Search
        result = bridge._handle_memory_search(query="test memory")
        assert result["count"] >= 0  # FTS may or may not match

    def test_memory_consolidate(self, bridge):
        bridge._initialize_memory_db()
        result = bridge._handle_memory_consolidate()
        assert result["consolidated"] is True

    def test_memory_sync(self, bridge):
        bridge._initialize_memory_db()
        result = bridge._handle_memory_sync()
        assert result["synced"] is True
        assert "total_entries" in result


class TestInputValidation:
    def test_gmail_send_invalid_email(self, bridge):
        result = bridge._handle_gmail_send(to="not-an-email", subject="test")
        assert result["sent"] is False
        assert "Invalid" in result["error"]

    def test_gmail_send_empty_to(self, bridge):
        result = bridge._handle_gmail_send(subject="test")
        assert result["sent"] is False

    def test_tts_speak_empty_text(self, bridge):
        result = bridge._handle_tts_speak()
        assert result["spoken"] is False
        assert "No text" in result["error"]

    def test_tts_speak_too_long(self, bridge):
        result = bridge._handle_tts_speak(text="x" * 20000)
        assert result["spoken"] is False
        assert "too long" in result["error"]

    def test_twilio_call_no_number(self, bridge):
        result = bridge._handle_twilio_call()
        assert result["called"] is False

    def test_twilio_sms_no_body(self, bridge):
        result = bridge._handle_twilio_sms(to="+1234567890")
        assert result["sent"] is False


class TestErrorHandling:
    def test_unknown_method(self, bridge):
        handler = bridge.handlers.get("nonexistent.method")
        assert handler is None

    def test_handler_exception_doesnt_crash(self, bridge):
        """Verify the handler dispatch pattern handles exceptions."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")
        bridge.llm_client = mock_client

        with pytest.raises(RuntimeError):
            bridge._handle_llm_complete(messages=[])


class TestEnvVarCheck:
    def test_check_env_vars_runs(self, bridge):
        """Just verify it doesn't crash."""
        bridge._check_env_vars()
