"""
OpenAI Client - Thin wrapper around the OpenAI Python SDK for GPT-5.2.
Shared singleton used by all Python loops and bridge handlers.
Model defaults to gpt-5.2 but can be overridden via OPENAI_MODEL env var.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper around the OpenAI GPT-5.2 API for reasoning/completion calls."""

    _instance: Optional['OpenAIClient'] = None

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.model = model or os.environ.get('OPENAI_MODEL', 'gpt-5.2')

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - GPT-5.2 API calls will fail")

        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"OpenAIClient initialized with model={self.model}")

    @classmethod
    def get_instance(cls) -> 'OpenAIClient':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a completion request to GPT-5.2.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            system: System prompt (prepended as a system message)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)

        Returns:
            {"content": str, "model": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
        """
        try:
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            return {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error(f"GPT-5.2 API error: {e}")
            raise

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        """Convenience method: single prompt in, text out."""
        result = self.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            **kwargs,
        )
        return result["content"]
