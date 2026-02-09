"""Tests for the OpenAI client wrapper."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestOpenAIClientInit:
    """Test OpenAIClient initialization."""

    def test_init_without_api_key(self):
        """Client should be disabled when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENAI_API_KEY', None)
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            assert client._disabled is True
            assert client.client is None

    def test_init_with_api_key(self):
        """Client should initialize when API key is provided."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient(api_key='test-key-123')
            assert client._disabled is False
            assert client.api_key == 'test-key-123'

    def test_default_model(self):
        """Should default to gpt-5.2."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENAI_MODEL', None)
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            assert client.model == 'gpt-5.2'

    def test_custom_model(self):
        """Should respect OPENAI_MODEL env var."""
        with patch.dict(os.environ, {'OPENAI_MODEL': 'gpt-4'}):
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            assert client.model == 'gpt-4'

    def test_thinking_mode_default(self):
        """Should default thinking_mode to 'high'."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENAI_THINKING_MODE', None)
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            assert client.thinking_mode == 'high'

    def test_thinking_mode_from_env(self):
        """Should respect OPENAI_THINKING_MODE env var."""
        with patch.dict(os.environ, {'OPENAI_THINKING_MODE': 'extra_high'}):
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            assert client.thinking_mode == 'extra_high'

    def test_disabled_client_raises_on_complete(self):
        """Disabled client should raise on complete()."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENAI_API_KEY', None)
            from openai_client import OpenAIClient
            OpenAIClient._instance = None
            client = OpenAIClient()
            with pytest.raises(EnvironmentError):
                client.complete(messages=[{"role": "user", "content": "test"}])

    def test_singleton_pattern(self):
        """get_instance() should return same instance."""
        from openai_client import OpenAIClient
        OpenAIClient._instance = None
        inst1 = OpenAIClient.get_instance()
        inst2 = OpenAIClient.get_instance()
        assert inst1 is inst2
        OpenAIClient._instance = None  # cleanup


class TestOpenAIClientComplete:
    """Test the complete() method with mocked API."""

    def test_complete_passes_reasoning(self):
        """complete() should pass reasoning config to API."""
        from openai_client import OpenAIClient
        OpenAIClient._instance = None
        client = OpenAIClient(api_key='test-key')

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.model = "gpt-5.2"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
        )

        call_kwargs = client.client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get('reasoning') == {"effort": "high"}
        assert result["content"] == "Hello"

    def test_complete_custom_reasoning(self):
        """complete() should accept custom reasoning param."""
        from openai_client import OpenAIClient
        OpenAIClient._instance = None
        client = OpenAIClient(api_key='test-key')

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-5.2"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response

        client.complete(
            messages=[{"role": "user", "content": "test"}],
            reasoning={"effort": "extra_high"},
        )

        call_kwargs = client.client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get('reasoning') == {"effort": "extra_high"}
