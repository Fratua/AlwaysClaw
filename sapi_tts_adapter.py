"""
SAPI TTS Adapter for OpenClaw AI Agent
Windows 10 Native Speech Synthesis
"""
import win32com.client
import pythoncom
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading


class SAPIVoiceGender(Enum):
    """SAPI voice gender options"""
    MALE = 1
    FEMALE = 2
    NEUTRAL = 3


@dataclass
class SAPIVoice:
    """Represents a SAPI voice"""
    id: str
    name: str
    gender: SAPIVoiceGender
    language: str
    age: str
    vendor: str
    description: str


class SAPITTSAdapter:
    """
    Windows SAPI 5.4 TTS Adapter
    Provides native Windows text-to-speech capabilities
    """

    def __init__(self):
        pythoncom.CoInitialize()
        self._speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self._voices_cache: List[SAPIVoice] = []
        self._lock = threading.Lock()
        self._initialize_voices()

    def _initialize_voices(self):
        """Cache available SAPI voices"""
        voices = self._speaker.GetVoices()
        for i in range(voices.Count):
            voice = voices.Item(i)
            gender = SAPIVoiceGender.NEUTRAL
            if voice.GetAttribute("Gender") == "Male":
                gender = SAPIVoiceGender.MALE
            elif voice.GetAttribute("Gender") == "Female":
                gender = SAPIVoiceGender.FEMALE

            self._voices_cache.append(SAPIVoice(
                id=voice.Id,
                name=voice.GetAttribute("Name"),
                gender=gender,
                language=voice.GetAttribute("Language"),
                age=voice.GetAttribute("Age"),
                vendor=voice.GetAttribute("Vendor"),
                description=voice.GetDescription()
            ))

    def get_voices(self) -> List[SAPIVoice]:
        """Return list of available voices"""
        return self._voices_cache.copy()

    def set_voice(self, voice_id: str) -> bool:
        """Set active voice by ID"""
        try:
            voices = self._speaker.GetVoices()
            for i in range(voices.Count):
                if voices.Item(i).Id == voice_id:
                    self._speaker.Voice = voices.Item(i)
                    return True
            return False
        except Exception as e:
            print(f"SAPI set_voice error: {e}")
            return False

    def speak(self, text: str, async_mode: bool = True) -> bool:
        """
        Synthesize speech from text

        Args:
            text: Text to speak
            async_mode: If True, returns immediately; if False, blocks until complete

        Returns:
            Success status
        """
        try:
            flags = 1 if async_mode else 0  # SVSFlagsAsync = 1
            self._speaker.Speak(text, flags)
            return True
        except Exception as e:
            print(f"SAPI speak error: {e}")
            return False

    def speak_ssml(self, ssml: str, async_mode: bool = True) -> bool:
        """Speak using SSML markup"""
        try:
            flags = 1 if async_mode else 0
            self._speaker.Speak(ssml, flags)
            return True
        except Exception as e:
            print(f"SAPI speak_ssml error: {e}")
            return False

    def stop(self):
        """Stop current speech"""
        self._speaker.Speak("", 2)  # SVSFPurgeBeforeSpeak = 2

    def pause(self):
        """Pause speech"""
        self._speaker.Pause()

    def resume(self):
        """Resume speech"""
        self._speaker.Resume()

    def set_rate(self, rate: int):
        """
        Set speech rate

        Args:
            rate: -10 (slowest) to 10 (fastest), 0 is default
        """
        self._speaker.Rate = max(-10, min(10, rate))

    def set_volume(self, volume: int):
        """
        Set speech volume

        Args:
            volume: 0 (silent) to 100 (loudest)
        """
        self._speaker.Volume = max(0, min(100, volume))

    def get_audio_stream(self, text: str) -> bytes:
        """
        Generate audio to memory stream

        Returns:
            WAV audio bytes
        """
        stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
        original_output = self._speaker.AudioOutputStream
        self._speaker.AudioOutputStream = stream
        self._speaker.Speak(text)
        self._speaker.AudioOutputStream = original_output
        return stream.GetData()

    def __del__(self):
        pythoncom.CoUninitialize()


# Usage Example
if __name__ == "__main__":
    sapi = SAPITTSAdapter()

    # List available voices
    voices = sapi.get_voices()
    for v in voices:
        print(f"{v.name} ({v.language}): {v.description}")

    # Speak text
    sapi.set_rate(0)
    sapi.set_volume(80)
    sapi.speak("Hello, I am OpenClaw AI Agent running on Windows 10.")
