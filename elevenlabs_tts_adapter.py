"""
ElevenLabs TTS Adapter
Ultra-realistic AI voice synthesis with streaming support
"""
import os
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
import requests
from elevenlabs import stream
from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from elevenlabs.play import play


@dataclass
class ElevenLabsVoice:
    """Represents an ElevenLabs voice"""
    voice_id: str
    name: str
    category: str  # premade, cloned, generated, professional
    description: Optional[str]
    labels: Dict[str, str]
    preview_url: Optional[str]
    settings: Optional[Dict]


class ElevenLabsModel(Enum):
    """Available ElevenLabs models"""
    MULTILINGUAL_V2 = "eleven_multilingual_v2"  # Highest quality
    FLASH_V2_5 = "eleven_flash_v2_5"  # Ultra-low latency (75ms)
    TURBO_V2_5 = "eleven_turbo_v2_5"  # Fast, good quality
    ENGLISH_V1 = "eleven_monolingual_v1"  # English only


class ElevenLabsTTSAdapter:
    """
    ElevenLabs TTS Adapter
    Supports 3000+ voices, voice cloning, and real-time streaming
    """

    # Default voices available to all users
    DEFAULT_VOICES = {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Bella": "EXAVITQu4vr4xnSDxMaL",
        "Antoni": "ErXwobaYiN019PkySvjV",
        "Elli": "MF3mGyEYCl7XYWbV9V6O",
        "Josh": "TxGEqnHWrfWFTfGW9XjX",
        "Arnold": "VR6AewLTigWG4xSOukaG",
        "Adam": "pNInz6obpgDQGcFmaJgB",
        "Sam": "yoZ06aMxZJJ28mfd3POQ",
        "Nicole": "piTKgcLEGmPE4e6mEKli"
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ELEVENLABS_API_KEY')

        if not self.api_key:
            raise ValueError("ElevenLabs API key required")

        self.client = ElevenLabs(api_key=self.api_key)
        self.async_client = AsyncElevenLabs(api_key=self.api_key)
        self._voices_cache: List[ElevenLabsVoice] = []
        self._default_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }

    def get_voices(self, refresh: bool = False) -> List[ElevenLabsVoice]:
        """Get available voices"""
        if not self._voices_cache or refresh:
            voices = self.client.voices.get_all()
            self._voices_cache = [
                ElevenLabsVoice(
                    voice_id=v.voice_id,
                    name=v.name,
                    category=v.category,
                    description=v.description,
                    labels=v.labels if v.labels else {},
                    preview_url=v.preview_url,
                    settings=v.settings.__dict__ if v.settings else None
                )
                for v in voices.voices
            ]
        return self._voices_cache

    def text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        model: str = ElevenLabsModel.MULTILINGUAL_V2.value,
        output_format: str = "mp3_44100_128",
        voice_settings: Optional[Dict] = None
    ) -> bytes:
        """
        Convert text to speech

        Args:
            text: Text to synthesize (max 5000 characters)
            voice_id: Voice identifier
            model: Model to use
            output_format: Audio output format
            voice_settings: Voice customization settings

        Returns:
            Audio bytes
        """
        settings = voice_settings or self._default_settings

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model,
            output_format=output_format,
            voice_settings=settings
        )

        return b''.join(audio)

    def stream_text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = ElevenLabsModel.FLASH_V2_5.value,
        voice_settings: Optional[Dict] = None
    ):
        """
        Stream text to speech in real-time

        Yields audio chunks as they're generated
        """
        settings = voice_settings or self._default_settings

        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id=model,
            voice_settings=settings
        )

        return audio_stream

    async def stream_text_to_speech_async(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = ElevenLabsModel.FLASH_V2_5.value,
        voice_settings: Optional[Dict] = None
    ) -> AsyncIterator[bytes]:
        """Async streaming text-to-speech"""
        settings = voice_settings or self._default_settings

        audio_stream = await self.async_client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id=model,
            voice_settings=settings
        )

        async for chunk in audio_stream:
            if isinstance(chunk, bytes):
                yield chunk

    def play_stream(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        """Stream and play audio immediately"""
        audio_stream = self.stream_text_to_speech(text, voice_id)
        stream(audio_stream)

    def clone_voice(
        self,
        name: str,
        description: str,
        audio_files: List[str],
        remove_background_noise: bool = True
    ) -> str:
        """
        Clone a voice from audio samples

        Args:
            name: Name for the cloned voice
            description: Voice description
            audio_files: List of audio file paths
            remove_background_noise: Whether to clean audio

        Returns:
            voice_id of the cloned voice
        """
        voice = self.client.voices.add(
            name=name,
            description=description,
            files=audio_files,
            remove_background_noise=remove_background_noise
        )
        return voice.voice_id

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice"""
        try:
            self.client.voices.delete(voice_id)
            return True
        except Exception as e:
            print(f"Error deleting voice: {e}")
            return False

    def get_voice_settings(self, voice_id: str) -> Dict:
        """Get voice settings"""
        voice = self.client.voices.get(voice_id)
        if voice.settings:
            return voice.settings.__dict__
        return self._default_settings

    def edit_voice_settings(
        self,
        voice_id: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ):
        """Edit voice settings"""
        self.client.voices.edit_settings(
            voice_id=voice_id,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost
        )

    def get_history(self) -> List[Dict]:
        """Get generation history"""
        history = self.client.history.get_all()
        return [
            {
                'history_item_id': h.history_item_id,
                'request_id': h.request_id,
                'voice_id': h.voice_id,
                'voice_name': h.voice_name,
                'text': h.text,
                'date': h.date_unix
            }
            for h in history.history
        ]


# Usage Example
if __name__ == "__main__":
    elevenlabs = ElevenLabsTTSAdapter()

    # List voices
    voices = elevenlabs.get_voices()
    for v in voices[:5]:
        print(f"{v.name} ({v.category}): {v.voice_id}")

    # Generate speech
    audio = elevenlabs.text_to_speech(
        text="Hello, I'm an AI agent powered by ElevenLabs!",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model=ElevenLabsModel.MULTILINGUAL_V2.value
    )

    # Save to file
    with open("output.mp3", "wb") as f:
        f.write(audio)

    # Stream and play
    elevenlabs.play_stream(
        "This is streaming in real-time with ultra-low latency!",
        model=ElevenLabsModel.FLASH_V2_5.value
    )
