
# OpenClaw Windows 10 AI Agent - TTS System Integration Architecture
## Technical Specification Document v1.0

---

## Table of Contents
1. Executive Summary
2. System Architecture Overview
3. TTS Engine Integrations
4. Voice Management System
5. SSML Support Architecture
6. Audio Output & Streaming Pipeline
7. Windows Audio Integration
8. Implementation Examples
9. Configuration & Deployment
10. Performance Optimization

---

## 1. Executive Summary

### 1.1 Purpose
This document provides a comprehensive technical specification for integrating multiple Text-to-Speech (TTS) engines into the OpenClaw Windows 10 AI Agent system. The architecture supports high-quality speech synthesis with multiple voice options, SSML support, real-time streaming, and seamless Windows audio integration.

### 1.2 Key Features
- **Multi-Engine Support**: Windows SAPI, Azure Cognitive Services, ElevenLabs, OpenAI TTS
- **High-Quality Voices**: Neural voices with emotional expression
- **SSML Support**: Full Speech Synthesis Markup Language capabilities
- **Real-Time Streaming**: Low-latency audio generation and playback
- **Voice Management**: Dynamic voice selection and caching
- **Windows Integration**: Native WASAPI and DirectSound support

### 1.3 Target Specifications
| Metric | Target |
|--------|--------|
| Latency | < 100ms (streaming) |
| Voice Options | 400+ voices across 140+ languages |
| Audio Formats | MP3, WAV, PCM, Opus |
| Sample Rates | 8kHz - 48kHz |
| Concurrent Streams | 5+ simultaneous |

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenClaw AI Agent System                            │
│                              (Windows 10)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   GPT-5.2 Core  │    │  Agent Loops    │    │  User Interface │         │
│  │   (Thinking)    │◄──►│  (15 Hardcoded) │◄──►│  (Voice/Text)   │         │
│  └────────┬────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    TTS Orchestration Layer                       │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │       │
│  │  │   Engine    │  │   Voice     │  │   SSML      │  │  Audio  │ │       │
│  │  │  Selector   │  │   Manager   │  │  Processor  │  │ Pipeline│ │       │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬────┘ │       │
│  └─────────┼────────────────┼────────────────┼──────────────┼──────┘       │
│            │                │                │              │              │
│            ▼                ▼                ▼              ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                      TTS Engine Adapters                         │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │       │
│  │  │  SAPI    │  │  Azure   │  │ElevenLabs│  │    OpenAI TTS    │ │       │
│  │  │  Adapter │  │  Adapter │  │  Adapter │  │     Adapter      │ │       │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Windows Audio Subsystem                       │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │       │
│  │  │  WASAPI  │  │DirectSound│  │  MME     │  │  Audio Endpoint  │ │       │
│  │  │  (Primary)│  │(Fallback) │  │(Legacy)  │  │  (Speakers/Out)  │ │       │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **TTS Orchestration Layer** | Central coordination, request routing, caching |
| **Engine Selector** | Chooses optimal TTS engine based on requirements |
| **Voice Manager** | Voice registration, caching, metadata management |
| **SSML Processor** | SSML parsing, validation, engine-specific transformation |
| **Audio Pipeline** | Format conversion, mixing, streaming, output routing |
| **Engine Adapters** | Normalize API differences across TTS providers |

---

## 3. TTS Engine Integrations

### 3.1 Windows SAPI (System.Speech) Integration

#### 3.1.1 Overview
- **Type**: Native Windows API (COM-based)
- **Availability**: Built into Windows 10
- **Best For**: Offline operation, system voices, quick responses
- **Latency**: Very low (< 50ms)

#### 3.1.2 Implementation

```python
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
        """Set speech rate: -10 (slowest) to 10 (fastest), 0 is default"""
        self._speaker.Rate = max(-10, min(10, rate))

    def set_volume(self, volume: int):
        """Set speech volume: 0 (silent) to 100 (loudest)"""
        self._speaker.Volume = max(0, min(100, volume))

    def get_audio_stream(self, text: str) -> bytes:
        """Generate audio to memory stream"""
        stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
        original_output = self._speaker.AudioOutputStream
        self._speaker.AudioOutputStream = stream
        self._speaker.Speak(text)
        self._speaker.AudioOutputStream = original_output
        return stream.GetData()

    def __del__(self):
        pythoncom.CoUninitialize()
```

#### 3.1.3 SAPI Configuration

```yaml
# config/sapi_config.yaml
sapi:
  enabled: true
  priority: 4  # Lower = higher priority (1-5)
  default_voice: "Microsoft David Desktop"
  default_rate: 0
  default_volume: 80
  cache_enabled: true
  max_cache_size_mb: 50
  supported_formats:
    - wav
    - mp3
  features:
    ssml: true
    streaming: false
    offline: true
```

---

### 3.2 Azure Cognitive Services Speech SDK Integration

#### 3.2.1 Overview
- **Type**: Cloud-based with optional containerized deployment
- **Availability**: Azure subscription required
- **Best For**: High-quality neural voices, multilingual support
- **Latency**: ~100-300ms (cloud), < 50ms (containerized)

#### 3.2.2 Implementation

```python
"""
Azure Cognitive Services TTS Adapter
High-quality neural voice synthesis
"""
import os
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import (
    SpeechConfig, SpeechSynthesizer, AudioConfig,
    ResultReason, CancellationReason
)
import io
import wave

@dataclass
class AzureVoice:
    """Represents an Azure neural voice"""
    name: str
    display_name: str
    locale: str
    gender: str
    voice_type: str
    style_list: List[str]
    sample_rate_hz: int

class AzureTTSAdapter:
    """
    Azure Cognitive Services Speech SDK Adapter
    Supports 400+ neural voices across 140+ languages
    """

    # Premium Neural Voices (Recommended)
    PREMIUM_VOICES = [
        "en-US-AvaMultilingualNeural",
        "en-US-AndrewMultilingualNeural",
        "en-US-EmmaMultilingualNeural",
        "en-US-BrianMultilingualNeural",
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural",
        "en-IN-NeerjaNeural",
        "en-US-Ava:DragonHDLatestNeural",
        "en-US-Andrew:DragonHDLatestNeural",
    ]

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        self.subscription_key = subscription_key or os.environ.get('AZURE_SPEECH_KEY')
        self.region = region or os.environ.get('AZURE_SPEECH_REGION', 'westus')
        self.endpoint = endpoint or os.environ.get('AZURE_SPEECH_ENDPOINT')

        if not self.subscription_key:
            raise ValueError("Azure Speech subscription key required")

        self._speech_config = self._create_speech_config()
        self._synthesizer: Optional[SpeechSynthesizer] = None
        self._voices_cache: List[AzureVoice] = []
        self._event_handlers: Dict[str, List[Callable]] = {
            'synthesis_started': [],
            'synthesis_completed': [],
            'word_boundary': [],
            'viseme_received': [],
            'bookmark_reached': []
        }
        self._initialize_synthesizer()

    def _create_speech_config(self) -> SpeechConfig:
        """Create Azure Speech configuration"""
        if self.endpoint:
            config = SpeechConfig(subscription=self.subscription_key, endpoint=self.endpoint)
        else:
            config = SpeechConfig(subscription=self.subscription_key, region=self.region)

        config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
        config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            value='true'
        )
        return config

    def _initialize_synthesizer(self):
        """Initialize speech synthesizer with audio output"""
        audio_config = AudioConfig(use_default_speaker=True)
        self._synthesizer = SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Configure synthesizer event handlers"""
        self._synthesizer.synthesis_started.connect(
            lambda evt: self._emit('synthesis_started', evt)
        )
        self._synthesizer.synthesis_completed.connect(
            lambda evt: self._emit('synthesis_completed', evt)
        )
        self._synthesizer.synthesis_word_boundary.connect(
            lambda evt: self._emit('word_boundary', evt)
        )
        self._synthesizer.viseme_received.connect(
            lambda evt: self._emit('viseme_received', evt)
        )
        self._synthesizer.bookmark_reached.connect(
            lambda evt: self._emit('bookmark_reached', evt)
        )

    def _emit(self, event: str, data: Any):
        """Emit event to registered handlers"""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                print(f"Event handler error: {e}")

    def on(self, event: str, handler: Callable):
        """Register event handler"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    def set_voice(self, voice_name: str):
        """Set synthesis voice"""
        self._speech_config.speech_synthesis_voice_name = voice_name
        self._initialize_synthesizer()

    def speak_text(self, text: str) -> Dict[str, Any]:
        """Synthesize text to speech"""
        result = self._synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return {
                'success': True,
                'audio_duration': result.audio_duration,
                'audio_length': len(result.audio_data),
                'result_id': result.result_id
            }
        elif result.reason == ResultReason.Canceled:
            cancellation = result.cancellation_details
            return {
                'success': False,
                'error': cancellation.reason.name,
                'error_details': cancellation.error_details
            }
        return {'success': False, 'error': 'Unknown error'}

    def speak_ssml(self, ssml: str) -> Dict[str, Any]:
        """Synthesize SSML to speech"""
        result = self._synthesizer.speak_ssml_async(ssml).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return {
                'success': True,
                'audio_duration': result.audio_duration,
                'audio_length': len(result.audio_data),
                'result_id': result.result_id
            }
        elif result.reason == ResultReason.Canceled:
            cancellation = result.cancellation_details
            return {
                'success': False,
                'error': cancellation.reason.name,
                'error_details': cancellation.error_details
            }
        return {'success': False, 'error': 'Unknown error'}

    def synthesize_to_file(self, text: str, output_path: str, ssml: bool = False) -> Dict[str, Any]:
        """Synthesize to audio file"""
        audio_config = AudioConfig(filename=output_path)
        synthesizer = SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

        if ssml:
            result = synthesizer.speak_ssml_async(text).get()
        else:
            result = synthesizer.speak_text_async(text).get()

        synthesizer = None

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return {
                'success': True,
                'output_path': output_path,
                'audio_duration': result.audio_duration
            }
        return {'success': False, 'error': result.reason.name}

    def synthesize_to_stream(self, text: str) -> bytes:
        """Synthesize to in-memory audio stream"""
        result = self._synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        return b''

    async def speak_text_async(self, text: str) -> Dict[str, Any]:
        """Async text-to-speech"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.speak_text, text)

    def stop(self):
        """Stop synthesis"""
        if self._synthesizer:
            self._synthesizer.stop_speaking_async()

    def get_available_voices(self) -> List[AzureVoice]:
        """Get list of available voices"""
        if not self._voices_cache:
            result = self._synthesizer.get_voices_async().get()
            for voice in result.voices:
                self._voices_cache.append(AzureVoice(
                    name=voice.name,
                    display_name=voice.short_name,
                    locale=voice.locale,
                    gender=voice.gender.name,
                    voice_type=voice.voice_type.name,
                    style_list=list(voice.style_list) if voice.style_list else [],
                    sample_rate_hz=voice.sample_rate_hz
                ))
        return self._voices_cache
```

#### 3.2.3 Azure Configuration

```yaml
# config/azure_tts_config.yaml
azure_tts:
  enabled: true
  priority: 1  # Highest priority for quality
  subscription_key: ${AZURE_SPEECH_KEY}
  region: ${AZURE_SPEECH_REGION}
  endpoint: ${AZURE_SPEECH_ENDPOINT}
  default_voice: "en-US-AvaMultilingualNeural"
  fallback_voices:
    - "en-US-AndrewMultilingualNeural"
    - "en-US-EmmaMultilingualNeural"
  features:
    ssml: true
    streaming: true
    word_boundary: true
    viseme: true
    bookmark: true
    styles: true
  audio_format: "Riff24Khz16BitMonoPcm"
  sample_rate: 24000
  profanity_option: "Masked"
  rate_control:
    requests_per_second: 20
    burst_size: 50
```

---

### 3.3 ElevenLabs API Integration

#### 3.3.1 Overview
- **Type**: Cloud-based AI voice synthesis
- **Availability**: API key required (elevenlabs.io)
- **Best For**: Ultra-realistic voices, voice cloning, emotional expression
- **Latency**: ~75ms (Flash v2.5), higher quality with Multilingual v2

#### 3.3.2 Implementation

```python
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
    category: str
    description: Optional[str]
    labels: Dict[str, str]
    preview_url: Optional[str]
    settings: Optional[Dict]

class ElevenLabsModel(Enum):
    """Available ElevenLabs models"""
    MULTILINGUAL_V2 = "eleven_multilingual_v2"
    FLASH_V2_5 = "eleven_flash_v2_5"
    TURBO_V2_5 = "eleven_turbo_v2_5"
    ENGLISH_V1 = "eleven_monolingual_v1"

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
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = ElevenLabsModel.MULTILINGUAL_V2.value,
        output_format: str = "mp3_44100_128",
        voice_settings: Optional[Dict] = None
    ) -> bytes:
        """Convert text to speech"""
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
        """Stream text to speech in real-time"""
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
        """Clone a voice from audio samples"""
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
```

#### 3.3.3 ElevenLabs Configuration

```yaml
# config/elevenlabs_config.yaml
elevenlabs:
  enabled: true
  priority: 2
  api_key: ${ELEVENLABS_API_KEY}
  default_voice: "21m00Tcm4TlvDq8ikWAM"  # Rachel
  default_model: "eleven_multilingual_v2"
  streaming_model: "eleven_flash_v2_5"
  voice_settings:
    stability: 0.5
    similarity_boost: 0.75
    style: 0.0
    use_speaker_boost: true
  output_format: "mp3_44100_128"
  max_text_length: 5000
  features:
    streaming: true
    voice_cloning: true
    history: true
    pronunciation_dictionary: true
  rate_limits:
    characters_per_month: 100000
    concurrent_requests: 5
  retry_config:
    max_retries: 3
    backoff_factor: 2
```

---

### 3.4 OpenAI TTS API Integration

#### 3.4.1 Overview
- **Type**: Cloud-based neural TTS
- **Availability**: OpenAI API key required
- **Best For**: Simple integration, consistent quality, cost-effective
- **Latency**: ~100-200ms

#### 3.4.2 Implementation

```python
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
        """Convert text to speech"""
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
        """Stream text-to-speech response"""
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=max(0.25, min(4.0, speed))
        )

        for chunk in response.iter_bytes(chunk_size=1024):
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
        """Estimate synthesis cost"""
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
```

#### 3.4.3 OpenAI TTS Configuration

```yaml
# config/openai_tts_config.yaml
openai_tts:
  enabled: true
  priority: 3
  api_key: ${OPENAI_API_KEY}
  default_voice: "nova"
  default_model: "tts-1"
  hd_model: "tts-1-hd"
  response_format: "mp3"
  default_speed: 1.0
  max_text_length: 4096
  features:
    streaming: true
    speed_control: true
    multiple_formats: true
  pricing:
    tts_1_per_1k_chars: 0.015
    tts_1_hd_per_1k_chars: 0.030
  rate_limits:
    requests_per_minute: 100
    characters_per_minute: 100000
```

---

## 4. Voice Management System

### 4.1 Voice Registry Architecture

```python
"""
Voice Management System for OpenClaw AI Agent
Unified voice registry with caching and selection logic
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import threading
import time

class VoiceQuality(Enum):
    """Voice quality tiers"""
    STANDARD = "standard"
    NEURAL = "neural"
    HD = "hd"
    ULTRA = "ultra"

class VoiceGender(Enum):
    """Voice gender options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    ANY = "any"

@dataclass
class Voice:
    """Unified voice representation"""
    id: str
    name: str
    provider: str
    locale: str
    gender: VoiceGender
    quality: VoiceQuality
    description: str
    sample_rate_hz: int
    supported_features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_cloned: bool = False
    is_default: bool = False

class VoiceManager:
    """
    Central voice management system
    Handles voice registration, caching, and selection
    """

    def __init__(self, cache_dir: str = "./voice_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._voices: Dict[str, Voice] = {}
        self._provider_voices: Dict[str, List[str]] = {
            'sapi': [],
            'azure': [],
            'elevenlabs': [],
            'openai': []
        }
        self._preferences: Dict[str, Any] = {
            'preferred_provider': 'azure',
            'preferred_gender': VoiceGender.ANY,
            'preferred_quality': VoiceQuality.NEURAL,
            'preferred_locale': 'en-US'
        }
        self._lock = threading.RLock()
        self._audio_cache: Dict[str, bytes] = {}
        self._max_cache_size = 100 * 1024 * 1024  # 100MB
        self._current_cache_size = 0

    def register_voice(self, voice: Voice) -> bool:
        """Register a voice in the system"""
        with self._lock:
            self._voices[voice.id] = voice
            if voice.provider in self._provider_voices:
                if voice.id not in self._provider_voices[voice.provider]:
                    self._provider_voices[voice.provider].append(voice.id)
            return True

    def register_voices(self, voices: List[Voice]):
        """Register multiple voices"""
        for voice in voices:
            self.register_voice(voice)

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get voice by ID"""
        with self._lock:
            return self._voices.get(voice_id)

    def get_voices_by_provider(self, provider: str) -> List[Voice]:
        """Get all voices from a provider"""
        with self._lock:
            voice_ids = self._provider_voices.get(provider, [])
            return [self._voices[vid] for vid in voice_ids if vid in self._voices]

    def find_voices(
        self,
        provider: Optional[str] = None,
        locale: Optional[str] = None,
        gender: Optional[VoiceGender] = None,
        quality: Optional[VoiceQuality] = None,
        feature: Optional[str] = None
    ) -> List[Voice]:
        """Find voices matching criteria"""
        with self._lock:
            results = list(self._voices.values())

            if provider:
                results = [v for v in results if v.provider == provider]
            if locale:
                results = [v for v in results if v.locale == locale]
            if gender:
                results = [v for v in results if v.gender == gender]
            if quality:
                results = [v for v in results if v.quality == quality]
            if feature:
                results = [v for v in results if feature in v.supported_features]

            return results

    def select_voice(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Voice]:
        """Select optimal voice based on content and context"""
        context = context or {}

        # Check for explicit voice ID
        if 'voice_id' in context:
            return self.get_voice(context['voice_id'])

        # Build filter criteria
        filters = {}

        if 'provider' in context:
            filters['provider'] = context['provider']
        else:
            filters['provider'] = self._preferences['preferred_provider']

        if 'locale' in context:
            filters['locale'] = context['locale']
        else:
            filters['locale'] = self._preferences['preferred_locale']

        if 'gender' in context:
            filters['gender'] = VoiceGender(context['gender'])

        if 'quality' in context:
            filters['quality'] = VoiceQuality(context['quality'])
        else:
            filters['quality'] = self._preferences['preferred_quality']

        # Find matching voices
        candidates = self.find_voices(**filters)

        if not candidates:
            # Relax filters progressively
            if 'gender' in filters:
                del filters['gender']
                candidates = self.find_voices(**filters)

            if not candidates and 'quality' in filters:
                del filters['quality']
                candidates = self.find_voices(**filters)

        # Return first match or default voice
        if candidates:
            for v in candidates:
                if v.is_default:
                    return v
            return candidates[0]

        # Fallback to any available voice
        return next(iter(self._voices.values()), None)

    def cache_audio(self, text: str, voice_id: str, audio: bytes):
        """Cache synthesized audio"""
        cache_key = self._generate_cache_key(text, voice_id)

        with self._lock:
            if self._current_cache_size + len(audio) > self._max_cache_size:
                self._evict_cache()

            self._audio_cache[cache_key] = audio
            self._current_cache_size += len(audio)

    def get_cached_audio(self, text: str, voice_id: str) -> Optional[bytes]:
        """Retrieve cached audio"""
        cache_key = self._generate_cache_key(text, voice_id)
        return self._audio_cache.get(cache_key)

    def _generate_cache_key(self, text: str, voice_id: str) -> str:
        """Generate cache key from text and voice"""
        content = f"{voice_id}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _evict_cache(self):
        """Evict oldest cache entries"""
        while self._current_cache_size > self._max_cache_size * 0.8:
            if self._audio_cache:
                key = next(iter(self._audio_cache))
                audio = self._audio_cache.pop(key)
                self._current_cache_size -= len(audio)
```

---

## 5. SSML Support Architecture

### 5.1 SSML Processor

```python
"""
SSML Processor for OpenClaw AI Agent
Speech Synthesis Markup Language parsing and transformation
"""
from typing import Dict, List, Optional, Any, Callable
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from enum import Enum
import re

class SSMLVersion(Enum):
    """SSML specification versions"""
    V1_0 = "1.0"
    V1_1 = "1.1"

@dataclass
class SSMLProsody:
    """Prosody settings"""
    rate: Optional[str] = None
    pitch: Optional[str] = None
    volume: Optional[str] = None
    contour: Optional[str] = None

class SSMLProcessor:
    """
    SSML Processor
    Parses, validates, and transforms SSML for different TTS engines
    """

    SSML_NS = "http://www.w3.org/2001/10/synthesis"
    MSTTS_NS = "http://www.w3.org/2001/mstts"

    SUPPORTED_ELEMENTS = {
        'sapi': ['speak', 'voice', 'prosody', 'emphasis', 'break', 'say-as', 'phoneme'],
        'azure': ['speak', 'voice', 'prosody', 'emphasis', 'break', 'say-as', 'phoneme',
                  'mstts:express-as', 'mstts:backgroundaudio', 'mstts:silence',
                  'audio', 'bookmark', 'p', 's', 'sub', 'mark'],
        'elevenlabs': ['speak', 'voice', 'prosody', 'break', 'emphasis'],
        'openai': ['speak']
    }

    def __init__(self):
        self._transformers: Dict[str, Callable] = {
            'sapi': self._transform_for_sapi,
            'azure': self._transform_for_azure,
            'elevenlabs': self._transform_for_elevenlabs,
            'openai': self._transform_for_openai
        }

    def parse(self, ssml: str) -> ET.Element:
        """Parse SSML string to ElementTree"""
        try:
            ET.register_namespace('', self.SSML_NS)
            ET.register_namespace('mstts', self.MSTTS_NS)
            root = ET.fromstring(ssml)
            return root
        except ET.ParseError as e:
            raise ValueError(f"Invalid SSML: {e}")

    def validate(self, ssml: str, provider: str) -> Dict[str, Any]:
        """Validate SSML for a specific provider"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'unsupported_elements': []
        }

        try:
            root = self.parse(ssml)
        except ValueError as e:
            result['valid'] = False
            result['errors'].append(str(e))
            return result

        supported = self.SUPPORTED_ELEMENTS.get(provider, [])

        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag not in supported and tag != 'speak':
                result['unsupported_elements'].append(tag)
                result['warnings'].append(f"Element '{tag}' may not be supported by {provider}")

        if root.tag != f"{{{self.SSML_NS}}}speak" and root.tag != 'speak':
            result['valid'] = False
            result['errors'].append("Root element must be <speak>")

        return result

    def transform(self, ssml: str, target_provider: str) -> str:
        """Transform SSML for target provider"""
        transformer = self._transformers.get(target_provider)
        if not transformer:
            raise ValueError(f"No transformer for provider: {target_provider}")

        return transformer(ssml)

    def _transform_for_sapi(self, ssml: str) -> str:
        """Transform SSML for Windows SAPI"""
        root = self.parse(ssml)

        unsupported = ['mstts:express-as', 'mstts:backgroundaudio', 'audio', 'bookmark', 'mark']
        for elem_name in unsupported:
            for elem in root.findall(f".//{{{self.MSTTS_NS}}}{elem_name}"):
                parent = self._get_parent(root, elem)
                if parent is not None:
                    parent.remove(elem)

        return ET.tostring(root, encoding='unicode')

    def _transform_for_azure(self, ssml: str) -> str:
        """Transform SSML for Azure (pass-through with validation)"""
        validation = self.validate(ssml, 'azure')
        if not validation['valid']:
            raise ValueError(f"Invalid SSML: {validation['errors']}")
        return ssml

    def _transform_for_elevenlabs(self, ssml: str) -> str:
        """Transform SSML for ElevenLabs"""
        root = self.parse(ssml)

        for elem in root.findall(f".//{{{self.MSTTS_NS}}}*"):
            parent = self._get_parent(root, elem)
            if parent is not None:
                if elem.text:
                    parent.text = (parent.text or '') + elem.text
                parent.remove(elem)

        return ET.tostring(root, encoding='unicode')

    def _transform_for_openai(self, ssml: str) -> str:
        """Transform SSML for OpenAI (extract plain text)"""
        root = self.parse(ssml)

        text_parts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())

        return ' '.join(text_parts)

    def _get_parent(self, root: ET.Element, child: ET.Element) -> Optional[ET.Element]:
        """Get parent element"""
        for parent in root.iter():
            for c in parent:
                if c is child:
                    return parent
        return None

    def create_speak(
        self,
        text: str,
        lang: str = "en-US",
        voice_name: Optional[str] = None,
        xmlns: Optional[str] = None
    ) -> str:
        """Create basic speak element"""
        ns = f' xmlns="{xmlns}"' if xmlns else f' xmlns="{self.SSML_NS}"'
        voice_attr = f' name="{voice_name}"' if voice_name else ''

        if voice_name:
            return f'<speak version="1.0"{ns} xml:lang="{lang}"><voice{voice_attr}>{text}</voice></speak>'
        else:
            return f'<speak version="1.0"{ns} xml:lang="{lang}">{text}</speak>'

    def add_prosody(
        self,
        text: str,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None
    ) -> str:
        """Wrap text with prosody element"""
        attrs = []
        if rate:
            attrs.append(f'rate="{rate}"')
        if pitch:
            attrs.append(f'pitch="{pitch}"')
        if volume:
            attrs.append(f'volume="{volume}"')

        attr_str = ' '.join(attrs)
        return f'<prosody {attr_str}>{text}</prosody>'

    def add_break(self, time_ms: Optional[int] = None, strength: Optional[str] = None) -> str:
        """Create break element"""
        if time_ms:
            return f'<break time="{time_ms}ms"/>'
        elif strength:
            return f'<break strength="{strength}"/>'
        return '<break/>'

    def add_emphasis(self, text: str, level: str = "moderate") -> str:
        """Wrap text with emphasis element"""
        return f'<emphasis level="{level}">{text}</emphasis>'

    def add_azure_style(self, text: str, voice: str, style: str) -> str:
        """Create Azure-specific style SSML"""
        return f'<speak version="1.0" xmlns="{self.SSML_NS}" xmlns:mstts="{self.MSTTS_NS}" xml:lang="en-US"><voice name="{voice}"><mstts:express-as style="{style}">{text}</mstts:express-as></voice></speak>'
```

---

## 6. Audio Output & Streaming Pipeline

### 6.1 Audio Pipeline Architecture

```python
"""
Audio Pipeline for OpenClaw AI Agent
Real-time audio streaming, mixing, and output
"""
import asyncio
import threading
import queue
from typing import Optional, Callable, Dict, List, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import wave
import io
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

class AudioFormat(Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    PCM = "pcm"
    OGG = "ogg"
    FLAC = "flac"
    OPUS = "opus"

class AudioBackend(Enum):
    """Available audio backends"""
    SOUNDDEVICE = "sounddevice"
    PYAUDIO = "pyaudio"
    WINSOUND = "winsound"
    WASAPI = "wasapi"

@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 24000
    channels: int = 1
    dtype: str = "int16"
    block_size: int = 1024
    backend: AudioBackend = AudioBackend.SOUNDDEVICE

class AudioStream:
    """Audio stream for real-time playback"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._buffer = queue.Queue(maxsize=100)
        self._stream = None
        self._is_playing = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        """Add playback callback"""
        self._callbacks.append(callback)

    def start(self):
        """Start audio stream"""
        with self._lock:
            if self._is_playing:
                return

            if self.config.backend == AudioBackend.SOUNDDEVICE and HAS_SOUNDDEVICE:
                self._stream = sd.OutputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    blocksize=self.config.block_size,
                    callback=self._audio_callback
                )
                self._stream.start()

            self._is_playing = True

    def stop(self):
        """Stop audio stream"""
        with self._lock:
            if not self._is_playing:
                return

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            self._is_playing = False

    def write(self, audio_data: bytes):
        """Write audio data to stream"""
        try:
            self._buffer.put_nowait(audio_data)
        except queue.Full:
            try:
                self._buffer.get_nowait()
                self._buffer.put_nowait(audio_data)
            except queue.Empty:
                pass

    def _audio_callback(self, outdata, frames, time_info, status):
        """SoundDevice callback"""
        try:
            data = self._buffer.get_nowait()
            audio_array = np.frombuffer(data, dtype=np.int16)

            if len(audio_array) >= frames:
                outdata[:] = audio_array[:frames].reshape(-1, 1)
            else:
                outdata[:len(audio_array)] = audio_array.reshape(-1, 1)
                outdata[len(audio_array):] = 0
        except queue.Empty:
            outdata.fill(0)

        for callback in self._callbacks:
            try:
                callback(frames, time_info)
            except:
                pass

class AudioPipeline:
    """
    Audio Pipeline
    Handles audio format conversion, mixing, and output
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._stream: Optional[AudioStream] = None
        self._converters: Dict[AudioFormat, Callable] = {
            AudioFormat.MP3: self._convert_mp3,
            AudioFormat.WAV: self._convert_wav,
            AudioFormat.PCM: self._convert_pcm,
        }
        self._volume = 1.0
        self._muted = False

    def initialize(self):
        """Initialize audio pipeline"""
        self._stream = AudioStream(self.config)
        self._stream.start()

    def shutdown(self):
        """Shutdown audio pipeline"""
        if self._stream:
            self._stream.stop()
            self._stream = None

    def play_audio(
        self,
        audio_data: bytes,
        format: AudioFormat = AudioFormat.MP3,
        blocking: bool = False
    ):
        """Play audio data"""
        pcm_data = self._convert_to_pcm(audio_data, format)

        if self._volume != 1.0 and not self._muted:
            pcm_data = self._apply_volume(pcm_data)

        if self._stream:
            self._stream.write(pcm_data)

    def play_stream(self, audio_iterator, format: AudioFormat = AudioFormat.MP3):
        """Play audio from iterator/stream"""
        for chunk in audio_iterator:
            self.play_audio(chunk, format, blocking=False)

    async def play_stream_async(self, audio_async_iterator, format: AudioFormat = AudioFormat.MP3):
        """Async audio streaming"""
        async for chunk in audio_async_iterator:
            self.play_audio(chunk, format, blocking=False)
            await asyncio.sleep(0)

    def _convert_to_pcm(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Convert audio to PCM format"""
        converter = self._converters.get(format)
        if converter:
            return converter(audio_data)
        return audio_data

    def _convert_mp3(self, data: bytes) -> bytes:
        """Convert MP3 to PCM"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(data))
            if audio.frame_rate != self.config.sample_rate:
                audio = audio.set_frame_rate(self.config.sample_rate)
            if audio.channels != self.config.channels:
                audio = audio.set_channels(self.config.channels)
            return audio.raw_data
        except ImportError:
            return data

    def _convert_wav(self, data: bytes) -> bytes:
        """Convert WAV to PCM"""
        try:
            with io.BytesIO(data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    return wav_file.readframes(wav_file.getnframes())
        except:
            return data

    def _convert_pcm(self, data: bytes) -> bytes:
        """PCM is already raw"""
        return data

    def _apply_volume(self, pcm_data: bytes) -> bytes:
        """Apply volume to PCM data"""
        if self._muted:
            return b'\x00' * len(pcm_data)

        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        audio_array = (audio_array * self._volume).astype(np.int16)
        return audio_array.tobytes()

    def set_volume(self, volume: float):
        """Set playback volume (0.0 to 1.0)"""
        self._volume = max(0.0, min(1.0, volume))

    def get_volume(self) -> float:
        """Get current volume"""
        return self._volume

    def mute(self):
        """Mute audio"""
        self._muted = True

    def unmute(self):
        """Unmute audio"""
        self._muted = False
```

---

## 7. Windows Audio Integration

### 7.1 WASAPI Integration

```python
"""
Windows WASAPI Audio Integration
Low-latency audio capture and playback
"""
import ctypes
from ctypes import wintypes
from enum import Enum
from typing import Optional, Callable, List
import comtypes
from comtypes import GUID, COMMETHOD, HRESULT
import numpy as np

# WASAPI COM Interface GUIDs
CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
IID_IMMDeviceEnumerator = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
IID_IAudioClient = GUID('{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}')
IID_IAudioRenderClient = GUID('{F294ACFC-3146-4483-A7BF-ADDCA7C260E2}')

# Audio format constants
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003

class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]

class AUDCLNT_SHAREMODE(Enum):
    SHARED = 0
    EXCLUSIVE = 1

class WASAPIAudio:
    """
    Windows WASAPI Audio Interface
    Provides low-latency audio on Windows 10
    """

    def __init__(self):
        self._device_enumerator = None
        self._audio_client = None
        self._render_client = None
        self._buffer_frame_count = 0
        self._is_initialized = False

    def initialize(self, sample_rate: int = 48000, channels: int = 2):
        """Initialize WASAPI audio"""
        try:
            comtypes.CoInitialize()

            self._device_enumerator = comtypes.client.CreateObject(
                CLSID_MMDeviceEnumerator,
                interface=comtypes.client.GetInterface(IID_IMMDeviceEnumerator)
            )

            device = self._device_enumerator.GetDefaultAudioEndpoint(
                0,  # eRender
                1   # eMultimedia
            )

            self._audio_client = device.Activate(
                IID_IAudioClient,
                comtypes.CLSCTX_ALL,
                None
            )

            format = WAVEFORMATEX()
            format.wFormatTag = WAVE_FORMAT_PCM
            format.nChannels = channels
            format.nSamplesPerSec = sample_rate
            format.wBitsPerSample = 16
            format.nBlockAlign = channels * 2
            format.nAvgBytesPerSec = sample_rate * format.nBlockAlign
            format.cbSize = 0

            REFTIMES_PER_SEC = 10000000
            buffer_duration = REFTIMES_PER_SEC * 2

            self._audio_client.Initialize(
                AUDCLNT_SHAREMODE.SHARED.value,
                0,
                buffer_duration,
                0,
                ctypes.byref(format),
                None
            )

            self._buffer_frame_count = self._audio_client.GetBufferSize()
            self._render_client = self._audio_client.GetService(IID_IAudioRenderClient)
            self._is_initialized = True

        except Exception as e:
            print(f"WASAPI initialization error: {e}")
            self.shutdown()
            raise

    def start(self):
        """Start audio playback"""
        if self._audio_client:
            self._audio_client.Start()

    def stop(self):
        """Stop audio playback"""
        if self._audio_client:
            self._audio_client.Stop()

    def write(self, audio_data: bytes):
        """Write audio data to playback buffer"""
        if not self._is_initialized:
            return

        padding = self._audio_client.GetCurrentPadding()
        available_frames = self._buffer_frame_count - padding

        if available_frames <= 0:
            return

        bytes_per_frame = 4
        frames_to_write = min(len(audio_data) // bytes_per_frame, available_frames)

        if frames_to_write <= 0:
            return

        buffer = self._render_client.GetBuffer(frames_to_write)
        ctypes.memmove(buffer, audio_data, frames_to_write * bytes_per_frame)
        self._render_client.ReleaseBuffer(frames_to_write, 0)

    def shutdown(self):
        """Cleanup WASAPI resources"""
        self.stop()

        self._render_client = None
        self._audio_client = None
        self._device_enumerator = None

        comtypes.CoUninitialize()
        self._is_initialized = False
```

---

## 8. Implementation Examples

### 8.1 Complete TTS Orchestrator

```python
"""
TTS Orchestrator for OpenClaw AI Agent
Central coordination of all TTS engines
"""
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import time
import queue
import os
import yaml

class TTSEngine(Enum):
    """Available TTS engines"""
    SAPI = "sapi"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"

@dataclass
class TTSRequest:
    """TTS request specification"""
    text: str
    voice_id: Optional[str] = None
    provider: Optional[TTSEngine] = None
    ssml: bool = False
    streaming: bool = False
    priority: int = 5
    metadata: Dict[str, Any] = None

@dataclass
class TTSResponse:
    """TTS response"""
    success: bool
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    engine_used: Optional[str] = None

class TTSOrchestrator:
    """
    TTS Orchestrator
    Manages multiple TTS engines with intelligent routing
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        self._adapters: Dict[TTSEngine, Any] = {}
        self._voice_manager: Optional[Any] = None
        self._ssml_processor: Optional[Any] = None
        self._audio_pipeline: Optional[Any] = None

        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._processing = False
        self._worker_thread: Optional[threading.Thread] = None

        self._stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_latency_ms': 0
        }

        self._initialize()

    def _load_config(self, path: Optional[str]) -> Dict:
        """Load configuration"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'priority_order': ['azure', 'elevenlabs', 'openai', 'sapi'],
            'fallback_enabled': True,
            'cache_enabled': True,
            'streaming_enabled': True,
            'default_voice': 'azure_ava',
            'audio': {
                'sample_rate': 24000,
                'channels': 1,
                'backend': 'sounddevice'
            }
        }

    def _initialize(self):
        """Initialize all components"""
        # Initialize adapters based on configuration
        self._init_adapters()

        # Start worker thread
        self._start_worker()

    def _init_adapters(self):
        """Initialize TTS adapters"""
        # Azure
        try:
            if os.environ.get('AZURE_SPEECH_KEY'):
                from azure_tts_adapter import AzureTTSAdapter
                self._adapters[TTSEngine.AZURE] = AzureTTSAdapter()
        except Exception as e:
            print(f"Azure TTS not available: {e}")

        # ElevenLabs
        try:
            if os.environ.get('ELEVENLABS_API_KEY'):
                from elevenlabs_tts_adapter import ElevenLabsTTSAdapter
                self._adapters[TTSEngine.ELEVENLABS] = ElevenLabsTTSAdapter()
        except Exception as e:
            print(f"ElevenLabs TTS not available: {e}")

        # OpenAI
        try:
            if os.environ.get('OPENAI_API_KEY'):
                from openai_tts_adapter import OpenAITTSAdapter
                self._adapters[TTSEngine.OPENAI] = OpenAITTSAdapter()
        except Exception as e:
            print(f"OpenAI TTS not available: {e}")

        # SAPI (always available on Windows)
        try:
            from sapi_tts_adapter import SAPITTSAdapter
            self._adapters[TTSEngine.SAPI] = SAPITTSAdapter()
        except Exception as e:
            print(f"SAPI TTS not available: {e}")

    def _start_worker(self):
        """Start request processing worker"""
        self._processing = True
        self._worker_thread = threading.Thread(target=self._process_queue)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def _process_queue(self):
        """Process TTS requests from queue"""
        while self._processing:
            try:
                priority, request, callback = self._request_queue.get(timeout=1)
                response = self._execute_request(request)
                if callback:
                    callback(response)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Queue processing error: {e}")

    def _execute_request(self, request: TTSRequest) -> TTSResponse:
        """Execute TTS request"""
        start_time = time.time()
        self._stats['requests_total'] += 1

        try:
            engine = self._select_engine(request)
            adapter = self._adapters.get(engine)

            if not adapter:
                return TTSResponse(
                    success=False,
                    error=f"Engine {engine} not available"
                )

            text = request.text

            # Execute synthesis
            if request.streaming and hasattr(adapter, 'stream_text_to_speech'):
                audio_stream = adapter.stream_text_to_speech(
                    text,
                    voice_id=request.voice_id
                )
                response = TTSResponse(
                    success=True,
                    engine_used=engine.value,
                    duration_ms=0
                )
            else:
                audio_data = adapter.text_to_speech(text, voice_id=request.voice_id)
                response = TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    engine_used=engine.value
                )

            self._stats['requests_success'] += 1

        except Exception as e:
            self._stats['requests_failed'] += 1
            response = TTSResponse(
                success=False,
                error=str(e)
            )

        latency_ms = (time.time() - start_time) * 1000
        self._stats['avg_latency_ms'] = (
            (self._stats['avg_latency_ms'] * (self._stats['requests_total'] - 1) + latency_ms)
            / self._stats['requests_total']
        )

        return response

    def _select_engine(self, request: TTSRequest) -> TTSEngine:
        """Select optimal TTS engine"""
        if request.provider and request.provider in self._adapters:
            return request.provider

        for engine_name in self.config['priority_order']:
            engine = TTSEngine(engine_name)
            if engine in self._adapters:
                return engine

        return next(iter(self._adapters.keys()))

    def speak(
        self,
        text: str,
        voice_id: Optional[str] = None,
        provider: Optional[TTSEngine] = None,
        ssml: bool = False,
        streaming: bool = False,
        async_mode: bool = False,
        callback: Optional[Callable] = None
    ) -> Optional[TTSResponse]:
        """Main TTS interface"""
        request = TTSRequest(
            text=text,
            voice_id=voice_id,
            provider=provider,
            ssml=ssml,
            streaming=streaming,
            priority=5
        )

        if async_mode:
            self._request_queue.put((request.priority, request, callback))
            return None
        else:
            return self._execute_request(request)

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        return self._stats.copy()

    def shutdown(self):
        """Shutdown orchestrator"""
        self._processing = False

        if self._worker_thread:
            self._worker_thread.join(timeout=5)
```

---

## 9. Configuration & Deployment

### 9.1 Main Configuration File

```yaml
# config/tts_config.yaml
# OpenClaw AI Agent TTS System Configuration

tts:
  # System settings
  system:
    name: "OpenClaw TTS"
    version: "1.0.0"
    log_level: "INFO"
    metrics_enabled: true

  # Engine priorities (lower = higher priority)
  engines:
    priority_order:
      - azure
      - elevenlabs
      - openai
      - sapi
    fallback_enabled: true
    auto_select: true

  # Voice management
  voices:
    default_voice: "azure_ava"
    default_locale: "en-US"
    default_gender: "female"
    cache_enabled: true
    cache_max_size_mb: 100
    registry_path: "./config/voice_registry.json"

  # Audio output
  audio:
    sample_rate: 24000
    channels: 1
    bit_depth: 16
    backend: "sounddevice"
    buffer_size: 1024
    volume_default: 0.8
    ducking_enabled: true

  # SSML settings
  ssml:
    enabled: true
    validation: "warn"
    default_version: "1.0"

  # Streaming settings
  streaming:
    enabled: true
    chunk_size: 1024
    pre_buffer_chunks: 3
    latency_target_ms: 100

  # Rate limiting
  rate_limits:
    requests_per_second: 10
    burst_size: 20
    cooldown_ms: 100

  # Provider configurations
  providers:
    azure:
      enabled: true
      subscription_key: ${AZURE_SPEECH_KEY}
      region: ${AZURE_SPEECH_REGION}
      default_voice: "en-US-AvaMultilingualNeural"

    elevenlabs:
      enabled: true
      api_key: ${ELEVENLABS_API_KEY}
      default_voice: "21m00Tcm4TlvDq8ikWAM"
      default_model: "eleven_multilingual_v2"

    openai:
      enabled: true
      api_key: ${OPENAI_API_KEY}
      default_voice: "nova"
      default_model: "tts-1"

    sapi:
      enabled: true
      default_voice: "Microsoft David Desktop"

# Environment variables required:
# - AZURE_SPEECH_KEY
# - AZURE_SPEECH_REGION
# - ELEVENLABS_API_KEY
# - OPENAI_API_KEY
```

---

## 10. Performance Optimization

### 10.1 Optimization Strategies

| Strategy | Implementation | Impact |
|----------|---------------|--------|
| **Voice Caching** | Cache synthesized audio by text hash | 50-90% latency reduction |
| **Connection Pooling** | Reuse HTTP connections for cloud APIs | 20-30% latency reduction |
| **Preemptive Synthesis** | Predict and synthesize likely responses | Near-instant playback |
| **Streaming** | Stream audio chunks as generated | <100ms time-to-first-audio |
| **Parallel Engines** | Initialize all engines concurrently | Faster startup |
| **Audio Buffering** | Pre-buffer audio chunks | Smooth playback |

### 10.2 Latency Targets

| Component | Target Latency |
|-----------|---------------|
| SAPI (local) | < 50ms |
| Azure (cloud) | < 200ms |
| ElevenLabs Flash | < 75ms |
| OpenAI TTS | < 150ms |
| Streaming first chunk | < 100ms |
| Full pipeline | < 300ms |

---

## Appendix A: Dependencies

```
# requirements.txt
# Core TTS dependencies

# Windows SAPI
pywin32>=306
comtypes>=1.2.0

# Azure Cognitive Services
azure-cognitiveservices-speech>=1.34.0

# ElevenLabs
elevenlabs>=0.2.26

# OpenAI
openai>=1.0.0

# Audio processing
sounddevice>=0.4.6
soundfile>=0.12.1
pyaudio>=0.2.13
pydub>=0.25.1
numpy>=1.24.0

# Utilities
pyyaml>=6.0
requests>=2.31.0
aiohttp>=3.9.0
```

---

## Appendix B: API Reference Summary

| Engine | Best For | SSML | Streaming | Voice Count | Latency |
|--------|----------|------|-----------|-------------|---------|
| SAPI | Offline, quick | Basic | No | 2-5 | <50ms |
| Azure | Quality, enterprise | Full | Yes | 400+ | ~200ms |
| ElevenLabs | Realism, cloning | Limited | Yes | 3000+ | ~75ms |
| OpenAI | Simplicity, cost | Minimal | Yes | 6 | ~150ms |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: OpenClaw AI Agent Team*
