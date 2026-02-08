"""
OpenAI TTS Adapter
Simple, high-quality neural text-to-speech
"""
import os
import asyncio
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
import io


# OpenAI TTS Voice Options
OpenAIVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# OpenAI TTS Models
OpenAITTSModel = Literal["tts-1", "tts-1-hd"]

# OpenAI TTS Output Formats
OpenAITTSFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


@dataclass
class OpenAITTSOptions:
    """OpenAI TTS synthesis options"""
    voice: OpenAIVoice = "alloy"
    model: OpenAITTSModel = "tts-1"
    response_format: OpenAITTSFormat = "mp3"
    speed: float = 1.0


class OpenAITTSAdapter:
    """
    OpenAI TTS Adapter
    Simple API with 6 built-in voices
    """

    # Voice descriptions
    VOICES = {
        "alloy": "Neutral, balanced voice suitable for most use cases",
        "echo": "Deep, mature male voice",
        "fable": "British accent, narrative style",
        "onyx": "Deep, authoritative male voice",
        "nova": "Warm, friendly female voice",
        "shimmer": "Bright, expressive female voice"
    }

    # Model specifications
    MODELS = {
        "tts-1": {
            "description": "Standard quality, lower latency",
            "max_chars": 4096,
            "sample_rate": 24000
        },
        "tts-1-hd": {
            "description": "High-definition quality",
            "max_chars": 4096,
            "sample_rate": 24000
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self._default_options = OpenAITTSOptions()

    def get_voices(self) -> Dict[str, str]:
        """Get available voices with descriptions"""
        return self.VOICES.copy()

    def get_models(self) -> Dict[str, Dict]:
        """Get available models with specifications"""
        return self.MODELS.copy()

    def text_to_speech(
        self,
        text: str,
        voice: OpenAIVoice = "alloy",
        model: OpenAITTSModel = "tts-1",
        response_format: OpenAITTSFormat = "mp3",
        speed: float = 1.0
    ) -> bytes:
        """
        Convert text to speech

        Args:
            text: Text to synthesize (max 4096 characters)
            voice: Voice to use
            model: Model to use (tts-1 or tts-1-hd)
            response_format: Output audio format
            speed: Speech speed (0.25 to 4.0)

        Returns:
            Audio bytes
        """
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=max(0.25, min(4.0, speed))
        )

        return response.content

    async def text_to_speech_async(
        self,
        text: str,
        voice: OpenAIVoice = "alloy",
        model: OpenAITTSModel = "tts-1",
        response_format: OpenAITTSFormat = "mp3",
        speed: float = 1.0
    ) -> bytes:
        """Async text-to-speech"""
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")

        response = await self.async_client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=max(0.25, min(4.0, speed))
        )

        return response.content

    def text_to_speech_streaming(
        self,
        text: str,
        voice: OpenAIVoice = "alloy",
        model: OpenAITTSModel = "tts-1",
        response_format: OpenAITTSFormat = "mp3",
        speed: float = 1.0
    ):
        """
        Stream text-to-speech response

        Yields audio chunks for real-time playback
        """
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=max(0.25, min(4.0, speed))
        )

        # Stream the response
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk

    async def text_to_speech_streaming_async(
        self,
        text: str,
        voice: OpenAIVoice = "alloy",
        model: OpenAITTSModel = "tts-1",
        response_format: OpenAITTSFormat = "mp3",
        speed: float = 1.0
    ):
        """Async streaming text-to-speech"""
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")

        response = await self.async_client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=max(0.25, min(4.0, speed))
        )

        async for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: OpenAIVoice = "alloy",
        model: OpenAITTSModel = "tts-1",
        response_format: OpenAITTSFormat = "mp3",
        speed: float = 1.0
    ) -> str:
        """Synthesize and save to file"""
        audio = self.text_to_speech(
            text=text,
            voice=voice,
            model=model,
            response_format=response_format,
            speed=speed
        )

        with open(output_path, "wb") as f:
            f.write(audio)

        return output_path

    def estimate_cost(self, text: str, model: OpenAITTSModel = "tts-1") -> Dict[str, float]:
        """
        Estimate synthesis cost

        Pricing (as of 2024):
        - tts-1: $0.015 per 1K characters
        - tts-1-hd: $0.030 per 1K characters
        """
        char_count = len(text)

        pricing = {
            "tts-1": 0.015,
            "tts-1-hd": 0.030
        }

        cost = (char_count / 1000) * pricing.get(model, 0.015)

        return {
            "character_count": char_count,
            "model": model,
            "estimated_cost_usd": round(cost, 6)
        }


# Usage Example
if __name__ == "__main__":
    openai_tts = OpenAITTSAdapter()

    # List available voices
    print("Available voices:")
    for voice, desc in openai_tts.get_voices().items():
        print(f"  {voice}: {desc}")

    # Generate speech
    audio = openai_tts.text_to_speech(
        text="Hello! I'm your AI assistant powered by OpenAI.",
        voice="nova",
        model="tts-1-hd"
    )

    # Save to file
    with open("openai_output.mp3", "wb") as f:
        f.write(audio)

    # Estimate cost
    cost = openai_tts.estimate_cost(
        "This is a sample text for cost estimation.",
        model="tts-1"
    )
    print(f"Estimated cost: ${cost['estimated_cost_usd']}")
