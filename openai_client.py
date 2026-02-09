"""
OpenAI Client - Thin wrapper around the OpenAI Python SDK for GPT-5.2.
Shared singleton used by all Python loops and bridge handlers.
Model defaults to gpt-5.2 but can be overridden via OPENAI_MODEL env var.
"""

import hashlib
import json
import os
import logging
from collections import OrderedDict
from typing import AsyncGenerator, List, Dict, Any, Optional

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = 128


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
        self.thinking_mode = os.environ.get('OPENAI_THINKING_MODE', 'high')
        self._disabled = False

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - GPT-5.2 client disabled until key is provided")
            self._disabled = True
            self.client = None
            return

        self.client = openai.OpenAI(api_key=self.api_key)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
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
        reasoning: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to GPT-5.2.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            system: System prompt (prepended as a system message)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            reasoning: Optional reasoning/thinking config, e.g. {"effort": "high"}.
                       Defaults to {"effort": self.thinking_mode} when not provided.

        Returns:
            {"content": str, "model": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
        """
        if self._disabled:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Set the environment variable to enable GPT-5.2 API calls."
            )

        # Build cache key for deduplication
        cache_key = self._make_cache_key(messages, system, max_tokens, temperature, reasoning)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            logger.debug("Returning cached completion result")
            return cached

        try:
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            # Build reasoning config: explicit param > instance default
            reasoning_config = reasoning if reasoning is not None else {
                "effort": self.thinking_mode
            }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning=reasoning_config,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = getattr(response, 'usage', None)
            result = {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0) if usage else 0,
                    "completion_tokens": getattr(usage, 'completion_tokens', 0) if usage else 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) if usage else 0,
                },
            }

            # Store in LRU cache
            self._cache[cache_key] = result
            if len(self._cache) > _CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

            return result
        except openai.APIConnectionError as e:
            logger.error(f"GPT-5.2 connection error: {e}")
            raise
        except openai.RateLimitError as e:
            logger.error(f"GPT-5.2 rate limit exceeded: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"GPT-5.2 authentication error: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"GPT-5.2 API error: {e}")
            raise

    def _make_cache_key(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int,
        temperature: float,
        reasoning: Optional[Dict[str, str]],
    ) -> str:
        """Build a deterministic hash key for the request parameters."""
        raw = json.dumps(
            {"messages": messages, "system": system, "max_tokens": max_tokens,
             "temperature": temperature, "reasoning": reasoning, "model": self.model},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion that yields content chunks as they arrive.

        Usage::

            async for chunk in client.stream_complete(messages=[...]):
                print(chunk, end="", flush=True)
        """
        if self._disabled:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Set the environment variable to enable GPT-5.2 API calls."
            )

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        reasoning_config = reasoning if reasoning is not None else {
            "effort": self.thinking_mode
        }

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning_config,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        """Convenience method: single prompt in, text out."""
        result = self.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            **kwargs,
        )
        return result["content"]
