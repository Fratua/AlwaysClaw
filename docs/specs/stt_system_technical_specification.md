# SPEECH-TO-TEXT (STT) SYSTEM TECHNICAL SPECIFICATION
## OpenClaw Windows 10 AI Agent Framework

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [STT Engine Integrations](#3-stt-engine-integrations)
4. [Wake Word Detection System](#4-wake-word-detection-system)
5. [Real-Time Streaming Recognition](#5-real-time-streaming-recognition)
6. [Audio Processing Pipeline](#6-audio-processing-pipeline)
7. [Custom Vocabulary & Language Models](#7-custom-vocabulary--language-models)
8. [Implementation Code Examples](#8-implementation-code-examples)
9. [Configuration & Deployment](#9-configuration--deployment)
10. [Performance Optimization](#10-performance-optimization)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Purpose
This document provides a comprehensive technical specification for integrating multi-engine Speech-to-Text (STT) capabilities into the OpenClaw Windows 10 AI agent framework. The system supports real-time transcription, wake word detection, and continuous listening modes.

### 1.2 Key Features
- **Multi-Engine Support**: Windows Speech API, Azure Speech SDK, OpenAI Whisper, Google Cloud Speech
- **Wake Word Detection**: Porcupine engine with custom keyword support
- **Real-Time Streaming**: Sub-200ms latency transcription pipeline
- **Voice Activity Detection**: WebRTC VAD / Silero VAD / Cobra VAD options
- **Audio Preprocessing**: Noise suppression, normalization, format conversion
- **Custom Vocabulary**: Domain-specific language model adaptation

### 1.3 Technical Requirements
| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (64-bit) |
| Python | 3.9+ |
| Audio Format | 16kHz, 16-bit PCM, Mono |
| Network | Internet for cloud STT APIs |
| Microphone | Standard USB/Bluetooth headset |

---

## 2. SYSTEM ARCHITECTURE OVERVIEW

### 2.1 High-Level Architecture

```
+-----------------------------------------------------------------------------+
|                         OPENCLAW STT SYSTEM                                  |
+-----------------------------------------------------------------------------+
|  +-----------------+    +-----------------+    +-----------------+         |
|  |  Audio Capture  |--->|  Preprocessing  |--->|  VAD Engine     |         |
|  |  (PyAudio/PvRec)|    |  (Noise/Normalize|   |  (Speech Detect)|         |
|  +-----------------+    +-----------------+    +--------+--------+         |
|                                                         |                    |
|                              +--------------------------+                    |
|                              |                                               |
|  +---------------------------------------------------------------------+    |
|  |                    WAKE WORD DETECTION (Porcupine)                   |    |
|  |  +--------------+  +--------------+  +--------------+               |    |
|  |  | "Hey Claw"   |  | "OpenClaw"   |  | Custom Wake  |               |    |
|  |  |  Detection   |  |  Detection   |  |  Words       |               |    |
|  |  +--------------+  +--------------+  +--------------+               |    |
|  +----------------------------------------+----------------------------+    |
|                                           |                                  |
|                              +------------+----------------+                 |
|                              |                            |                 |
|  +---------------------------------+  +---------------------------------+   |
|  |      STT ENGINE SELECTOR        |  |   CONTINUOUS LISTENING MODE     |   |
|  |  +---------+ +---------+       |  |   (Background Transcription)    |   |
|  |  | Primary | | Fallback|       |  |                                 |   |
|  |  | Engine  | | Engine  |       |  |  +---------+ +---------+       |   |
|  |  |(Azure)  | |(Whisper)|       |  |  | Whisper | |  Azure  |       |   |
|  |  +----+----+ +----+----+       |  |  |  Local  | |  Cloud  |       |   |
|  |       +-----+-----+            |  |  +----+----+ +----+----+       |   |
|  |             |                  |  |       +-----+-----+            |   |
|  |  +----------------------+     |  |             |                  |   |
|  |  |  Transcription Queue |     |  |  +----------------------+     |   |
|  |  |  (Async Processing)  |     |  |  |  GPT-5.2 Integration |     |   |
|  |  +----------+-----------+     |  |  |  (Agent Loop Trigger)|     |   |
|  +-------------+-----------------+  |  +----------+-----------+     |   |
|                |                    +-------------+-----------------+   |
|                |                                  |                      |
|                |                                  |                      |
|                v                                  v                      |
|  +--------------------------------------------------------------------+  |
|  |                     AGENT COMMAND PROCESSOR                         |  |
|  |  +----------+  +----------+  +----------+  +----------+           |  |
|  |  | Command  |  |  Query   |  |  System  |  |  Twilio  |           |  |
|  |  | Execution|  | Processing|  |  Control |  |  Voice   |           |  |
|  |  +----------+  +----------+  +----------+  +----------+           |  |
|  +--------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

### 2.2 Data Flow Diagram

```
+-------------+     +-------------+     +-------------+     +-------------+
|   Audio     |---->|   Frame     |---->|    VAD      |---->|   Wake      |
|   Input     |     |   Buffer    |     |  Detection  |     |   Word      |
|  (16kHz)    |     |  (32ms)     |     |             |     |   Check     |
+-------------+     +-------------+     +-------------+     +------+------+
                                                                   |
                              +------------------------------------+
                              |
              +-------------------------------+
              |      WAKE WORD DETECTED?      |
              |  +---------+   +---------+   |
              |  |   YES   |   |   NO    |   |
              |  +---+-----+   +---+-----+   |
              +------+-------------+---------+
                     |             |
                     |             |
                     v             v
         +-----------------+  +-----------------+
         | Activate Agent  |  | Continue VAD    |
         | Start Recording |  | Discard Frame   |
         | (5-10s window)  |  |                 |
         +--------+--------+  +-----------------+
                  |
                  v
         +-----------------+
         |  STT Processing |
         |  +-----------+  |
         |  |  Primary  |  |
         |  |  Engine   |  |
         |  +-----+-----+  |
         |        |        |
         |   +----+----+   |
         |   |         |   |
         | +-----+  +-----+|
         | |Success| |Fail ||
         | +--+--+  +--+--+|
         +----+--------+---+
              |        |
              |        |
              v        v
         +--------+ +--------+
         | Return | |Fallback|
         |Transcript| | Engine |
         |        | |        |
         +----+---+ +----+---+
              |          |
              +-----+----+
                    |
                    v
         +-----------------+
         | GPT-5.2 Agent   |
         | Command Parser  |
         +-----------------+
```

---

## 3. STT ENGINE INTEGRATIONS

### 3.1 Engine Comparison Matrix

| Feature | Windows Speech | Azure Speech SDK | OpenAI Whisper | Google Cloud Speech |
|---------|---------------|------------------|----------------|---------------------|
| **Offline Capability** | Yes | No* | Yes (local) | No |
| **Latency** | Low | Very Low | Medium | Low |
| **Accuracy** | Good | Excellent | Excellent | Excellent |
| **Custom Vocab** | Limited | Excellent | Good | Excellent |
| **Streaming** | Limited | Yes | Yes** | Yes |
| **Cost** | Free | $1/hour | $0.006/min | $0.024/min |
| **Languages** | 10+ | 100+ | 99 | 125+ |
| **Windows Integration** | Native | Good | Good | Good |

*Azure has containerized option for offline
**Whisper streaming via chunked processing

### 3.2 Windows Speech Recognition API

#### 3.2.1 Overview
Native Windows 10/11 speech recognition using COM interface via Python.

#### 3.2.2 Installation
```bash
pip install SpeechRecognition pywin32 comtypes
```

#### 3.2.3 Implementation
```python
"""
Windows Speech Recognition Integration
- Uses Windows built-in speech engine
- No internet required
- Limited language support
"""
import speech_recognition as sr
import win32com.client as wincl
from typing import Optional, Callable
import threading

class WindowsSpeechRecognizer:
    """
    Windows-native speech recognition wrapper.
    Best for: Offline operation, Windows-native integration
    """
    
    def __init__(self, language: str = "en-US"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language
        self.is_listening = False
        self._callback: Optional[Callable] = None
        self._thread: Optional[threading.Thread] = None
        
    def calibrate(self, duration: int = 2):
        """Calibrate for ambient noise."""
        with self.microphone as source:
            print(f"Calibrating for ambient noise ({duration}s)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            print(f"Energy threshold set to: {self.recognizer.energy_threshold}")
    
    def recognize_once(self, timeout: Optional[int] = None) -> Optional[str]:
        """Single recognition attempt."""
        with self.microphone as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognizer.recognize_windows(audio, language=self.language)
                return text
            except sr.WaitTimeoutError:
                print("Listening timeout")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except Exception as e:
                print(f"Recognition error: {e}")
                return None
    
    def start_continuous(self, callback: Callable[[str], None]):
        """Start continuous background listening."""
        self._callback = callback
        self.is_listening = True
        self._thread = threading.Thread(target=self._listen_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def _listen_loop(self):
        """Background listening loop."""
        with self.microphone as source:
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                    text = self.recognizer.recognize_windows(
                        audio, 
                        language=self.language,
                        show_all=False
                    )
                    if text and self._callback:
                        self._callback(text)
                except sr.UnknownValueError:
                    pass  # Ignore unintelligible speech
                except Exception as e:
                    print(f"Listen loop error: {e}")
    
    def stop(self):
        """Stop continuous listening."""
        self.is_listening = False
        if self._thread:
            self._thread.join(timeout=2)
```

### 3.3 Azure Cognitive Services Speech SDK

#### 3.3.1 Overview
Microsoft's enterprise-grade speech recognition with streaming support.

#### 3.3.2 Installation
```bash
pip install azure-cognitiveservices-speech
```

#### 3.3.3 Implementation
```python
"""
Azure Speech SDK Integration
- Real-time streaming recognition
- Custom language model support
- Speaker diarization available
"""
import azure.cognitiveservices.speech as speechsdk
from typing import Callable, Optional, List
import threading
import queue
import os

class AzureSpeechRecognizer:
    """
    Azure Cognitive Services Speech-to-Text wrapper.
    Best for: Production systems, custom vocabulary, real-time streaming
    """
    
    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: str = "westus",
        language: str = "en-US",
        endpoint_id: Optional[str] = None  # For custom models
    ):
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "westus")
        self.language = language
        self.endpoint_id = endpoint_id
        
        # Speech configuration
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        self.speech_config.speech_recognition_language = language
        
        # Custom endpoint if provided
        if endpoint_id:
            self.speech_config.endpoint_id = endpoint_id
        
        # Recognition state
        self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.is_recognizing = False
        self._result_queue: queue.Queue = queue.Queue()
        
    def create_push_stream_recognizer(self) -> tuple:
        """
        Create recognizer with push audio stream for real-time processing.
        Returns: (recognizer, push_stream)
        """
        # Audio format: 16kHz, 16-bit, mono PCM
        format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1
        )
        
        push_stream = speechsdk.audio.PushAudioInputStream(format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        # Event handlers
        recognizer.recognizing.connect(self._on_recognizing)
        recognizer.recognized.connect(self._on_recognized)
        recognizer.session_started.connect(self._on_session_started)
        recognizer.session_stopped.connect(self._on_session_stopped)
        recognizer.canceled.connect(self._on_canceled)
        
        return recognizer, push_stream
    
    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """Handle intermediate recognition results."""
        if evt.result.text:
            print(f"[Interim] {evt.result.text}")
    
    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        """Handle final recognition results."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self._result_queue.put({
                "type": "final",
                "text": evt.result.text,
                "confidence": evt.result.confidence,
                "duration": evt.result.duration
            })
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            self._result_queue.put({
                "type": "nomatch",
                "reason": evt.result.no_match_details.reason
            })
    
    def _on_session_started(self, evt):
        print("Azure STT: Session started")
    
    def _on_session_stopped(self, evt):
        print("Azure STT: Session stopped")
    
    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        print(f"Azure STT: Canceled - {evt.error_details}")
        self._result_queue.put({
            "type": "error",
            "error": evt.error_details,
            "code": evt.error_code
        })
    
    def recognize_once_from_microphone(self) -> Optional[dict]:
        """Single recognition from microphone."""
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        print("Listening...")
        result = recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {
                "text": result.text,
                "confidence": result.confidence,
                "duration": result.duration
            }
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {"error": "No speech recognized"}
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            return {
                "error": cancellation.error_details,
                "code": cancellation.error_code
            }
        return None
    
    def start_continuous_recognition(
        self,
        interim_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None
    ):
        """Start continuous recognition with callbacks."""
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        if interim_callback:
            self.recognizer.recognizing.connect(
                lambda evt: interim_callback(evt.result.text) if evt.result.text else None
            )
        
        if final_callback:
            self.recognizer.recognized.connect(
                lambda evt: final_callback(evt.result.text) 
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech else None
            )
        
        self.is_recognizing = True
        self.recognizer.start_continuous_recognition_async()
    
    def stop_continuous_recognition(self):
        """Stop continuous recognition."""
        if self.recognizer and self.is_recognizing:
            self.recognizer.stop_continuous_recognition_async()
            self.is_recognizing = False
```

### 3.4 OpenAI Whisper API Integration

#### 3.4.1 Overview
OpenAI's state-of-the-art speech recognition with local and cloud options.

#### 3.4.2 Installation
```bash
pip install openai-whisper openai
```

#### 3.4.3 Implementation
```python
"""
OpenAI Whisper Integration
- Local model execution (offline capable)
- Cloud API option for faster processing
- Supports 99 languages
"""
import whisper
import openai
import numpy as np
import io
import wave
from typing import Optional, Union, BinaryIO, Literal
import os
import tempfile

class WhisperRecognizer:
    """
    OpenAI Whisper ASR wrapper with local and API modes.
    Best for: High accuracy, multilingual support, offline operation
    """
    
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    
    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large"] = "base",
        api_key: Optional[str] = None,
        device: Optional[str] = None,  # "cuda", "cpu", or None (auto)
        language: Optional[str] = "en"
    ):
        self.model_size = model_size
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.language = language
        self.device = device
        
        # Local model (lazy loading)
        self._local_model: Optional[whisper.Whisper] = None
        
        # OpenAI client for API mode
        self._client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    
    @property
    def local_model(self) -> whisper.Whisper:
        """Lazy load local Whisper model."""
        if self._local_model is None:
            print(f"Loading Whisper model: {self.model_size}")
            self._local_model = whisper.load_model(
                self.model_size,
                device=self.device
            )
        return self._local_model
    
    def transcribe_local(
        self,
        audio: Union[str, np.ndarray, BinaryIO],
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        word_timestamps: bool = False
    ) -> dict:
        """
        Transcribe using local Whisper model.
        
        Args:
            audio: Audio file path, numpy array, or file-like object
            initial_prompt: Optional prompt to guide transcription
            temperature: Sampling temperature (0.0 = deterministic)
            word_timestamps: Return word-level timestamps
        """
        options = {
            "language": self.language,
            "temperature": temperature,
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps
        }
        
        result = self.local_model.transcribe(audio, **options)
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "confidence": self._estimate_confidence(result)
        }
    
    def transcribe_api(
        self,
        audio: Union[str, bytes, BinaryIO],
        model: str = "whisper-1",
        response_format: str = "json",
        timestamp_granularities: Optional[list] = None
    ) -> dict:
        """
        Transcribe using OpenAI API.
        Faster but requires internet and API credits.
        """
        if not self._client:
            raise ValueError("OpenAI API key required for API mode")
        
        # Handle different input types
        if isinstance(audio, str):
            audio_file = open(audio, "rb")
        elif isinstance(audio, bytes):
            audio_file = io.BytesIO(audio)
        else:
            audio_file = audio
        
        try:
            response = self._client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=self.language,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities or ["segment"]
            )
            
            if response_format == "json":
                return {
                    "text": response.text,
                    "model": model
                }
            return {"raw": response}
        finally:
            if isinstance(audio, str):
                audio_file.close()
    
    def transcribe_streaming(
        self,
        audio_chunks: list,
        chunk_duration: float = 5.0,
        overlap: float = 0.5
    ) -> list:
        """
        Simulate streaming by processing overlapping chunks.
        
        Args:
            audio_chunks: List of audio byte chunks
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        results = []
        buffer = b""
        
        for chunk in audio_chunks:
            buffer += chunk
            
            # Process when buffer is full
            if len(buffer) >= int(chunk_duration * 16000 * 2):  # 16kHz, 16-bit
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    self._write_wav(f, buffer)
                    temp_path = f.name
                
                try:
                    result = self.transcribe_local(temp_path)
                    results.append(result)
                finally:
                    os.unlink(temp_path)
                
                # Keep overlap for next chunk
                overlap_bytes = int(overlap * 16000 * 2)
                buffer = buffer[-overlap_bytes:] if len(buffer) > overlap_bytes else buffer
        
        # Process remaining buffer
        if buffer:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self._write_wav(f, buffer)
                temp_path = f.name
            
            try:
                result = self.transcribe_local(temp_path)
                results.append(result)
            finally:
                os.unlink(temp_path)
        
        return results
    
    def _write_wav(self, file_obj, pcm_data: bytes, sample_rate: int = 16000):
        """Write PCM data as WAV file."""
        with wave.open(file_obj, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
    
    def _estimate_confidence(self, result: dict) -> float:
        """Estimate confidence from segment-level data."""
        segments = result.get("segments", [])
        if not segments:
            return 0.5
        
        avg_confidence = sum(
            seg.get("avg_logprob", -1) for seg in segments
        ) / len(segments)
        
        # Convert logprob to approximate confidence (0-1)
        return min(max((avg_confidence + 1) / 2, 0), 1)


class WhisperRealtimeStreamer:
    """
    Real-time streaming wrapper for Whisper using chunked processing.
    """
    
    def __init__(
        self,
        recognizer: WhisperRecognizer,
        chunk_duration: float = 3.0,
        silence_threshold: float = 0.5
    ):
        self.recognizer = recognizer
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.audio_buffer = b""
        self.is_streaming = False
    
    def process_frame(self, frame: bytes) -> Optional[str]:
        """
        Process a single audio frame.
        Returns transcription when a complete utterance is detected.
        """
        self.audio_buffer += frame
        
        # Check if we have enough audio
        required_bytes = int(self.chunk_duration * 16000 * 2)
        
        if len(self.audio_buffer) >= required_bytes:
            # Process chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self.recognizer._write_wav(f, self.audio_buffer)
                temp_path = f.name
            
            try:
                result = self.recognizer.transcribe_local(temp_path)
                text = result.get("text", "").strip()
                
                # Clear buffer on successful transcription
                self.audio_buffer = b""
                return text
            finally:
                os.unlink(temp_path)
        
        return None
```

### 3.5 Google Cloud Speech-to-Text

#### 3.5.1 Overview
Google's enterprise speech recognition with advanced features.

#### 3.5.2 Installation
```bash
pip install google-cloud-speech
```

#### 3.5.3 Implementation
```python
"""
Google Cloud Speech-to-Text Integration
- Streaming recognition support
- Speaker diarization
- Speech adaptation for custom vocabulary
"""
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
import os
from typing import Iterator, Optional, Callable
import queue
import threading

class GoogleSpeechRecognizer:
    """
    Google Cloud Speech-to-Text wrapper.
    Best for: Enterprise scale, advanced features, Google Cloud integration
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "global",
        language_code: str = "en-US",
        model: str = "chirp_2"  # or "latest", "command_and_search", etc.
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.language_code = language_code
        self.model = model
        
        # Initialize client
        api_endpoint = f"{location}-speech.googleapis.com" if location != "global" else "speech.googleapis.com"
        self.client = speech_v2.SpeechClient(
            client_options={"api_endpoint": api_endpoint}
        )
        
        self.recognizer_path = f"projects/{self.project_id}/locations/{location}/recognizers/_"
    
    def recognize_sync(self, audio_content: bytes) -> list:
        """Synchronous recognition for short audio."""
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language_code],
            model=self.model
        )
        
        request = cloud_speech.RecognizeRequest(
            recognizer=self.recognizer_path,
            config=config,
            content=audio_content
        )
        
        response = self.client.recognize(request=request)
        
        results = []
        for result in response.results:
            alternative = result.alternatives[0] if result.alternatives else None
            if alternative:
                results.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence,
                    "words": [
                        {"word": w.word, "start": w.start_offset, "end": w.end_offset}
                        for w in alternative.words
                    ]
                })
        
        return results
    
    def recognize_streaming(
        self,
        audio_generator: Iterator[bytes],
        interim_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Streaming recognition from audio generator.
        
        Args:
            audio_generator: Iterator yielding audio chunks
            interim_callback: Called with interim results
            final_callback: Called with final results
        """
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language_code],
            model=self.model
        )
        
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=config
        )
        
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=streaming_config
        )
        
        def request_generator():
            yield config_request
            for audio_chunk in audio_generator:
                yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
        
        responses = self.client.streaming_recognize(
            requests=request_generator()
        )
        
        for response in responses:
            for result in response.results:
                alternative = result.alternatives[0] if result.alternatives else None
                if alternative:
                    text = alternative.transcript
                    if result.is_final and final_callback:
                        final_callback(text)
                    elif not result.is_final and interim_callback:
                        interim_callback(text)
    
    def recognize_long_running(self, gcs_uri: str) -> list:
        """Long-running recognition for files > 1 minute."""
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language_code],
            model=self.model
        )
        
        audio = cloud_speech.RecognitionAudio(uri=gcs_uri)
        
        operation = self.client.long_running_recognize(
            config=config,
            audio=audio
        )
        
        print("Waiting for operation to complete...")
        response = operation.result(timeout=300)
        
        results = []
        for result in response.results:
            alternative = result.alternatives[0] if result.alternatives else None
            if alternative:
                results.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence
                })
        
        return results
```

---

## 4. WAKE WORD DETECTION SYSTEM

### 4.1 Porcupine Wake Word Engine

#### 4.1.1 Overview
Picovoice Porcupine is a lightweight, on-device wake word detection engine.

#### 4.1.2 Features
- **Accuracy**: 95%+ detection rate at 1% false positive rate
- **Performance**: < 5% CPU on Raspberry Pi Zero
- **Languages**: English, Spanish, German, French, Italian, Portuguese, Hindi, Japanese, Korean, Mandarin
- **Custom Keywords**: Train custom wake words via Picovoice Console

#### 4.1.3 Installation
```bash
pip install pvporcupine pvrecorder
```

#### 4.1.4 Implementation
```python
"""
Porcupine Wake Word Detection System
- On-device processing (no cloud)
- Multiple wake word support
- Custom keyword training available
"""
import pvporcupine
from pvrecorder import PvRecorder
from typing import List, Callable, Optional, Dict
import threading
import queue

class WakeWordDetector:
    """
    Porcupine-based wake word detection with continuous listening.
    """
    
    # Built-in keywords available
    BUILT_IN_KEYWORDS = [
        "alexa", "americano", "blueberry", "bumblebee", "computer",
        "grapefruit", "grasshopper", "hey google", "hey siri", "jarvis",
        "ok google", "picovoice", "porcupine", "terminator"
    ]
    
    def __init__(
        self,
        access_key: str,
        keywords: Optional[List[str]] = None,
        keyword_paths: Optional[List[str]] = None,
        sensitivities: Optional[List[float]] = None,
        model_path: Optional[str] = None,
        device_index: int = -1  # -1 for default microphone
    ):
        """
        Initialize wake word detector.
        
        Args:
            access_key: Picovoice Console access key
            keywords: List of built-in keyword names
            keyword_paths: List of paths to custom .ppn keyword files
            sensitivities: Detection sensitivity (0.0 to 1.0) for each keyword
            model_path: Path to custom model file (.pv)
            device_index: Microphone device index (-1 for default)
        """
        self.access_key = access_key
        self.device_index = device_index
        
        # Create Porcupine instance
        if keyword_paths:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=keyword_paths,
                sensitivities=sensitivities or [0.5] * len(keyword_paths),
                model_path=model_path
            )
        elif keywords:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=keywords,
                sensitivities=sensitivities or [0.5] * len(keywords),
                model_path=model_path
            )
        else:
            raise ValueError("Either keywords or keyword_paths must be provided")
        
        # Initialize recorder
        self.recorder = PvRecorder(
            device_index=device_index,
            frame_length=self.porcupine.frame_length
        )
        
        # State
        self.is_listening = False
        self._callbacks: Dict[int, List[Callable]] = {i: [] for i in range(len(self.porcupine.keyword_names))}
        self._listen_thread: Optional[threading.Thread] = None
    
    @property
    def keyword_names(self) -> List[str]:
        """Get list of configured keyword names."""
        return self.porcupine.keyword_names
    
    @property
    def sample_rate(self) -> int:
        """Get required sample rate."""
        return self.porcupine.sample_rate
    
    def add_callback(self, keyword_index: int, callback: Callable[[], None]):
        """Add callback for specific keyword detection."""
        self._callbacks[keyword_index].append(callback)
    
    def start(self, blocking: bool = False):
        """Start wake word detection."""
        self.is_listening = True
        self.recorder.start()
        
        if blocking:
            self._listen_loop()
        else:
            self._listen_thread = threading.Thread(target=self._listen_loop)
            self._listen_thread.daemon = True
            self._listen_thread.start()
    
    def _listen_loop(self):
        """Main listening loop."""
        print(f"Wake word detection active. Listening for: {self.keyword_names}")
        
        while self.is_listening:
            try:
                # Read audio frame
                pcm = self.recorder.read()
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    detected_keyword = self.keyword_names[keyword_index]
                    print(f"[WAKE WORD DETECTED] {detected_keyword}")
                    
                    # Trigger callbacks
                    for callback in self._callbacks[keyword_index]:
                        try:
                            callback()
                        except Exception as e:
                            print(f"Callback error: {e}")
                            
            except Exception as e:
                print(f"Listen loop error: {e}")
    
    def stop(self):
        """Stop wake word detection."""
        self.is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2)
        self.recorder.stop()
    
    def delete(self):
        """Release resources."""
        self.stop()
        self.porcupine.delete()
        self.recorder.delete()


class MultiStageWakeWordSystem:
    """
    Multi-stage wake word system with confirmation.
    Reduces false positives with secondary confirmation.
    """
    
    def __init__(
        self,
        detector: WakeWordDetector,
        confirmation_timeout: float = 3.0,
        confirmation_phrases: Optional[List[str]] = None
    ):
        self.detector = detector
        self.confirmation_timeout = confirmation_timeout
        self.confirmation_phrases = confirmation_phrases or ["yes", "confirm", "proceed"]
        self._awaiting_confirmation = False
        self._confirmation_timer: Optional[threading.Timer] = None
        self._on_confirmed_callbacks: List[Callable] = []
    
    def add_confirmed_callback(self, callback: Callable[[], None]):
        """Add callback for confirmed wake word."""
        self._on_confirmed_callbacks.append(callback)
    
    def start(self):
        """Start multi-stage detection."""
        # Add primary wake word callback
        for i in range(len(self.detector.keyword_names)):
            self.detector.add_callback(i, self._on_wake_word_detected)
        
        self.detector.start(blocking=False)
    
    def _on_wake_word_detected(self):
        """Handle initial wake word detection."""
        if not self._awaiting_confirmation:
            print("Wake word detected! Awaiting confirmation...")
            self._awaiting_confirmation = True
            
            # Start confirmation timer
            self._confirmation_timer = threading.Timer(
                self.confirmation_timeout,
                self._confirmation_timeout
            )
            self._confirmation_timer.start()
            
            # Here you would activate secondary STT to listen for confirmation
            # This is a placeholder for the confirmation logic
    
    def process_confirmation(self, text: str) -> bool:
        """Process potential confirmation phrase."""
        if not self._awaiting_confirmation:
            return False
        
        text_lower = text.lower()
        for phrase in self.confirmation_phrases:
            if phrase in text_lower:
                self._confirm()
                return True
        
        return False
    
    def _confirm(self):
        """Handle confirmed wake word."""
        self._awaiting_confirmation = False
        if self._confirmation_timer:
            self._confirmation_timer.cancel()
        
        print("Wake word CONFIRMED!")
        for callback in self._on_confirmed_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Confirmed callback error: {e}")
    
    def _confirmation_timeout(self):
        """Handle confirmation timeout."""
        print("Confirmation timeout - wake word not confirmed")
        self._awaiting_confirmation = False
```

---

## 5. REAL-TIME STREAMING RECOGNITION

### 5.1 Streaming Architecture

```
+-----------------------------------------------------------------------------+
|                    REAL-TIME STREAMING PIPELINE                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +-------------+   +-------------+   +-------------+   +-------------+     |
|  |   Audio     |-->|   Ring      |-->|    VAD      |-->|   Speech    |     |
|  |   Capture   |   |   Buffer    |   |  Detection  |   |   Buffer    |     |
|  |  (16kHz)    |   |  (500ms)    |   |             |   |  (5-10s)    |     |
|  +-------------+   +-------------+   +------+------+   +------+------+     |
|                                             |                   |            |
|                              +--------------+                   |            |
|                              |                                  |            |
|                              |                    +---------------------+    |
|                              |                    |   Utterance         |    |
|                              |                    |   Complete?         |    |
|                              |                    |  +-----+ +-----+   |    |
|                              |                    |  | YES | | NO  |   |    |
|                              |                    |  +--+--+ +--+--+   |    |
|                              |                    +----+-------+-------+    |
|                              |                         |      |            |
|                              |            +------------+      |            |
|                              |            |                   |            |
|                              |   +-----------------+  +--------+------+     |
|                              |   |  Send to STT    |  | Continue      |     |
|                              |   |  Engine         |  | Buffering     |     |
|                              |   +--------+--------+  +---------------+     |
|                              |              |                                |
|                              |              |                                |
|                              |              v                                |
|                              |   +-----------------+                        |
|                              |   |  Get Transcript |                        |
|                              |   |  + Confidence   |                        |
|                              |   +--------+--------+                        |
|                              |              |                                |
|                              |              v                                |
|                              |   +-----------------+                        |
|                              |   |  Post-process   |                        |
|                              |   |  + Command      |                        |
|                              |   |  Extraction     |                        |
|                              |   +--------+--------+                        |
|                              |              |                                |
|                              |              v                                |
|                              |   +-----------------+                        |
|                              +-->|  GPT-5.2 Agent  |                        |
|                                  |  Integration    |                        |
|                                  +-----------------+                        |
+-----------------------------------------------------------------------------+
```

### 5.2 Streaming Implementation

```python
"""
Real-Time Streaming Recognition Pipeline
Combines VAD, buffering, and multi-engine STT for low-latency transcription.
"""
import webrtcvad
import collections
import contextlib
import wave
import threading
import queue
from typing import Callable, Optional, List, Tuple
import numpy as np
import pyaudio

class RealtimeTranscriber:
    """
    Real-time streaming speech recognition with VAD-based segmentation.
    """
    
    def __init__(
        self,
        stt_engine: Callable[[bytes], dict],
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        padding_duration_ms: int = 300,
        min_speech_duration_ms: int = 250
    ):
        """
        Initialize real-time transcriber.
        
        Args:
            stt_engine: Function that takes audio bytes and returns transcription dict
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame size in ms (10, 20, or 30)
            vad_aggressiveness: VAD mode (0-3, higher = more aggressive)
            padding_duration_ms: Padding before/after speech
            min_speech_duration_ms: Minimum speech duration to process
        """
        self.stt_engine = stt_engine
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.padding_duration_ms = padding_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        
        # Frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        
        # State
        self.is_running = False
        self._transcript_callbacks: List[Callable[[str, dict], None]] = []
        self._interim_callbacks: List[Callable[[str], None]] = []
    
    def add_transcript_callback(self, callback: Callable[[str, dict], None]):
        """Add callback for final transcripts."""
        self._transcript_callbacks.append(callback)
    
    def add_interim_callback(self, callback: Callable[[str], None]):
        """Add callback for interim (partial) transcripts."""
        self._interim_callbacks.append(callback)
    
    def _frame_generator(self, stream) -> bytes:
        """Generate audio frames from stream."""
        while self.is_running:
            frame = stream.read(self.frame_size, exception_on_overflow=False)
            if frame:
                yield frame
    
    def _vad_collector(self, frames):
        """
        Filter frames based on VAD, yielding speech segments.
        
        Yields: (speech_segment_bytes, is_speech_start)
        """
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        
        triggered = False
        voiced_frames = []
        
        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                
                # Start triggered if enough voiced frames
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
                # End triggered if enough unvoiced frames
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b"".join(voiced_frames), True
                    ring_buffer.clear()
                    voiced_frames = []
        
        # Yield any remaining frames
        if voiced_frames:
            yield b"".join(voiced_frames), True
    
    def start(self, device_index: Optional[int] = None):
        """Start real-time transcription."""
        self.is_running = True
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.frame_size
        )
        
        print("Real-time transcription started. Speak now...")
        
        try:
            frames = self._frame_generator(stream)
            
            for segment, is_start in self._vad_collector(frames):
                # Check minimum duration
                duration_ms = len(segment) / (self.sample_rate * 2) * 1000
                
                if duration_ms >= self.min_speech_duration_ms:
                    # Process with STT engine
                    result = self.stt_engine(segment)
                    
                    text = result.get("text", "").strip()
                    if text:
                        # Notify callbacks
                        for callback in self._transcript_callbacks:
                            callback(text, result)
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
    
    def stop(self):
        """Stop transcription."""
        self.is_running = False
    
    def __del__(self):
        """Cleanup."""
        self.audio.terminate()


class StreamingSTTController:
    """
    Controller for managing multiple STT engines with fallback.
    """
    
    def __init__(
        self,
        primary_engine: Callable[[bytes], dict],
        fallback_engine: Optional[Callable[[bytes], dict]] = None,
        confidence_threshold: float = 0.7
    ):
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.confidence_threshold = confidence_threshold
    
    def transcribe(self, audio: bytes) -> dict:
        """
        Transcribe with primary engine, fallback if confidence is low.
        """
        # Try primary engine
        result = self.primary_engine(audio)
        confidence = result.get("confidence", 0)
        
        # Check if fallback needed
        if confidence < self.confidence_threshold and self.fallback_engine:
            print(f"Primary confidence {confidence:.2f} below threshold, trying fallback...")
            fallback_result = self.fallback_engine(audio)
            fallback_confidence = fallback_result.get("confidence", 0)
            
            # Use fallback if better
            if fallback_confidence > confidence:
                return fallback_result
        
        return result
```

---

## 6. AUDIO PROCESSING PIPELINE

### 6.1 Audio Capture

```python
"""
Audio Capture Module
Cross-platform audio input with format standardization.
"""
import pyaudio
import numpy as np
from typing import Optional, Iterator, Callable
import threading
import queue

class AudioCapture:
    """
    Standardized audio capture for speech recognition.
    Output: 16kHz, 16-bit PCM, mono
    """
    
    # Target format for STT engines
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_FORMAT = pyaudio.paInt16
    
    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        buffer_duration_ms: int = 500
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.buffer_duration_ms = buffer_duration_ms
        
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.buffer_size = int(sample_rate * buffer_duration_ms / 1000)
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_capturing = False
        
        self._audio_queue: queue.Queue = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
    
    @staticmethod
    def list_devices():
        """List available audio input devices."""
        audio = pyaudio.PyAudio()
        devices = []
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": info["defaultSampleRate"]
                })
        
        audio.terminate()
        return devices
    
    def start(self, callback: Optional[Callable[[bytes], None]] = None):
        """Start audio capture."""
        self.is_capturing = True
        
        self.stream = self.audio.open(
            format=self.TARGET_FORMAT,
            channels=self.TARGET_CHANNELS,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frame_size
        )
        
        if callback:
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(callback,)
            )
            self._capture_thread.daemon = True
            self._capture_thread.start()
    
    def _capture_loop(self, callback: Callable[[bytes], None]):
        """Background capture loop."""
        while self.is_capturing:
            try:
                data = self.stream.read(self.frame_size, exception_on_overflow=False)
                callback(data)
            except Exception as e:
                print(f"Capture error: {e}")
    
    def read(self) -> bytes:
        """Read a single frame."""
        if self.stream:
            return self.stream.read(self.frame_size, exception_on_overflow=False)
        return b""
    
    def read_frames(self, num_frames: int) -> bytes:
        """Read multiple frames."""
        frames = []
        for _ in range(num_frames):
            frames.append(self.read())
        return b"".join(frames)
    
    def stop(self):
        """Stop audio capture."""
        self.is_capturing = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def __del__(self):
        """Cleanup."""
        self.stop()
        self.audio.terminate()
```

### 6.2 Audio Preprocessing

```python
"""
Audio Preprocessing Module
Noise reduction, normalization, and format conversion.
"""
import numpy as np
import noisereduce as nr
from typing import Optional, Tuple
import wave
import io

class AudioPreprocessor:
    """
    Audio preprocessing for speech recognition.
    """
    
    def __init__(
        self,
        apply_noise_reduction: bool = True,
        apply_normalization: bool = True,
        noise_reduction_strength: float = 0.8,
        target_db: float = -20.0
    ):
        self.apply_noise_reduction = apply_noise_reduction
        self.apply_normalization = apply_normalization
        self.noise_reduction_strength = noise_reduction_strength
        self.target_db = target_db
    
    def bytes_to_numpy(self, audio_bytes: bytes, dtype=np.int16) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        return np.frombuffer(audio_bytes, dtype=dtype)
    
    def numpy_to_bytes(self, audio_array: np.ndarray, dtype=np.int16) -> bytes:
        """Convert numpy array to audio bytes."""
        return audio_array.astype(dtype).tobytes()
    
    def normalize(self, audio: np.ndarray, target_db: Optional[float] = None) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Audio array (int16 or float)
            target_db: Target dB level (default: -20)
        
        Returns:
            Normalized audio array
        """
        target_db = target_db or self.target_db
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms == 0:
            return audio
        
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio_float * gain
        
        # Clip to prevent distortion
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Convert back to int16
        return (normalized * 32767).astype(np.int16)
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        prop_decrease: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply noise reduction using noisereduce library.
        
        Args:
            audio: Audio array (int16 or float)
            sample_rate: Audio sample rate
            prop_decrease: Noise reduction strength (0-1)
        
        Returns:
            Noise-reduced audio array
        """
        prop_decrease = prop_decrease or self.noise_reduction_strength
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            prop_decrease=prop_decrease
        )
        
        # Convert back to int16
        return (reduced * 32767).astype(np.int16)
    
    def preprocess(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        noise_sample: Optional[bytes] = None
    ) -> bytes:
        """
        Full preprocessing pipeline.
        
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate
            noise_sample: Optional noise sample for profile-based reduction
        
        Returns:
            Preprocessed audio bytes
        """
        # Convert to numpy
        audio = self.bytes_to_numpy(audio_bytes)
        
        # Apply noise reduction
        if self.apply_noise_reduction:
            audio = self.reduce_noise(audio, sample_rate)
        
        # Apply normalization
        if self.apply_normalization:
            audio = self.normalize(audio)
        
        # Convert back to bytes
        return self.numpy_to_bytes(audio)
    
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
        
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        # Simple linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled.astype(audio.dtype)
    
    @staticmethod
    def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
        """Convert PCM data to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        return buffer.getvalue()
    
    @staticmethod
    def wav_to_pcm(wav_data: bytes) -> bytes:
        """Extract PCM data from WAV."""
        buffer = io.BytesIO(wav_data)
        with wave.open(buffer, 'rb') as wav:
            return wav.readframes(wav.getnframes())
```

### 6.3 Voice Activity Detection Options

```python
"""
Voice Activity Detection Module
Multiple VAD engine options with unified interface.
"""
import webrtcvad
import numpy as np
from typing import Optional, List
from enum import Enum

class VADEngine(Enum):
    """Available VAD engines."""
    WEBRTC = "webrtc"
    SILERO = "silero"
    COBRA = "cobra"

class BaseVAD:
    """Base class for VAD implementations."""
    
    def is_speech(self, audio_frame: bytes, sample_rate: int) -> bool:
        raise NotImplementedError
    
    def process_stream(self, audio_stream: bytes, sample_rate: int) -> List[Tuple[int, int, bool]]:
        """Process entire stream, return list of (start, end, is_speech) tuples."""
        raise NotImplementedError

class WebRTCVAD(BaseVAD):
    """
    WebRTC Voice Activity Detector.
    Fast, lightweight, good for real-time applications.
    """
    
    VALID_RATES = [8000, 16000, 32000, 48000]
    VALID_FRAME_MS = [10, 20, 30]
    
    def __init__(self, mode: int = 2):
        """
        Initialize WebRTC VAD.
        
        Args:
            mode: Aggressiveness (0=least, 3=most aggressive)
        """
        self.vad = webrtcvad.Vad(mode)
    
    def is_speech(self, audio_frame: bytes, sample_rate: int) -> bool:
        """Check if frame contains speech."""
        if sample_rate not in self.VALID_RATES:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        return self.vad.is_speech(audio_frame, sample_rate)
    
    def process_stream(
        self,
        audio_stream: bytes,
        sample_rate: int,
        frame_duration_ms: int = 30
    ) -> List[Tuple[int, int, bool]]:
        """Process stream in frames."""
        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit
        results = []
        
        for i in range(0, len(audio_stream), frame_size):
            frame = audio_stream[i:i + frame_size]
            if len(frame) < frame_size:
                break
            
            is_speech = self.is_speech(frame, sample_rate)
            results.append((i, i + frame_size, is_speech))
        
        return results

class SileroVAD(BaseVAD):
    """
    Silero VAD - Deep learning based voice activity detection.
    More accurate than WebRTC, especially in noisy conditions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize Silero VAD."""
        try:
            import torch
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.collect_chunks = utils[3]
            self.remove_silence = utils[4]
        except ImportError:
            raise ImportError("PyTorch required for Silero VAD. Install with: pip install torch")
    
    def is_speech(self, audio_frame: bytes, sample_rate: int) -> bool:
        """Silero VAD works on larger chunks, use process_stream instead."""
        raise NotImplementedError("Use process_stream for Silero VAD")
    
    def process_stream(
        self,
        audio_stream: bytes,
        sample_rate: int = 16000
    ) -> List[dict]:
        """
        Process audio stream and return speech timestamps.
        
        Returns:
            List of dicts with 'start', 'end', 'confidence'
        """
        import torch
        import io
        
        # Convert bytes to tensor
        audio_np = np.frombuffer(audio_stream, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(audio_np)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=0.5,
            sampling_rate=sample_rate
        )
        
        return speech_timestamps

class CobraVAD(BaseVAD):
    """
    Picovoice Cobra VAD - Production-ready deep learning VAD.
    Best accuracy, enterprise support available.
    """
    
    def __init__(self, access_key: str):
        """Initialize Cobra VAD."""
        try:
            import pvcobra
            self.cobra = pvcobra.create(access_key=access_key)
        except ImportError:
            raise ImportError("pvcobra required. Install with: pip install pvcobra")
    
    @property
    def frame_length(self) -> int:
        """Get required frame length."""
        return self.cobra.frame_length
    
    def is_speech(self, audio_frame: bytes, sample_rate: int) -> bool:
        """Check if frame contains speech."""
        # Convert bytes to int16 array
        pcm = np.frombuffer(audio_frame, dtype=np.int16)
        
        # Process
        voice_probability = self.cobra.process(pcm)
        
        # Threshold at 0.5
        return voice_probability > 0.5
    
    def get_voice_probability(self, audio_frame: bytes) -> float:
        """Get voice probability (0.0 to 1.0)."""
        pcm = np.frombuffer(audio_frame, dtype=np.int16)
        return self.cobra.process(pcm)
    
    def process_stream(
        self,
        audio_stream: bytes,
        sample_rate: int = 16000
    ) -> List[Tuple[int, float]]:
        """
        Process stream and return voice probabilities.
        
        Returns:
            List of (frame_index, voice_probability) tuples
        """
        frame_length = self.frame_length
        results = []
        
        for i in range(0, len(audio_stream), frame_length * 2):
            frame = audio_stream[i:i + frame_length * 2]
            if len(frame) < frame_length * 2:
                break
            
            prob = self.get_voice_probability(frame)
            results.append((i // (frame_length * 2), prob))
        
        return results
    
    def delete(self):
        """Release resources."""
        self.cobra.delete()
```

---

## 7. CUSTOM VOCABULARY & LANGUAGE MODELS

### 7.1 Azure Custom Speech

```python
"""
Azure Custom Speech Configuration
Training and deploying custom language models.
"""
from azure.cognitiveservices.speech import (
    SpeechConfig,
    SpeechRecognizer,
    CustomModel,
    PhraseListGrammar
)
from typing import List, Optional

class AzureCustomSpeechManager:
    """
    Manager for Azure Custom Speech models.
    """
    
    def __init__(
        self,
        subscription_key: str,
        region: str,
        project_name: str
    ):
        self.subscription_key = subscription_key
        self.region = region
        self.project_name = project_name
        self.speech_config = SpeechConfig(
            subscription=subscription_key,
            region=region
        )
    
    def create_phrase_list(
        self,
        phrases: List[str],
        recognizer: SpeechRecognizer
    ):
        """
        Create a phrase list for real-time boosting.
        
        Args:
            phrases: List of phrases/words to boost
            recognizer: SpeechRecognizer instance
        """
        phrase_list = PhraseListGrammar.from_recognizer(recognizer)
        
        for phrase in phrases:
            phrase_list.addPhrase(phrase)
        
        return phrase_list
    
    def clear_phrase_list(self, recognizer: SpeechRecognizer):
        """Clear phrase list from recognizer."""
        phrase_list = PhraseListGrammar.from_recognizer(recognizer)
        phrase_list.clear()
```

### 7.2 OpenAI Whisper Customization

```python
"""
Whisper Customization Options
Prompting and fine-tuning for domain-specific vocabulary.
"""
from typing import List, Optional

class WhisperVocabularyEnhancer:
    """
    Enhance Whisper transcription with custom vocabulary.
    """
    
    def __init__(self, base_prompt: str = ""):
        self.base_prompt = base_prompt
        self.vocabulary: List[str] = []
    
    def add_vocabulary(self, terms: List[str]):
        """Add domain-specific terms."""
        self.vocabulary.extend(terms)
    
    def build_prompt(self, context: Optional[str] = None) -> str:
        """
        Build initial prompt for Whisper.
        
        The prompt guides the model toward specific vocabulary.
        """
        prompt_parts = []
        
        if self.base_prompt:
            prompt_parts.append(self.base_prompt)
        
        if self.vocabulary:
            vocab_str = ", ".join(self.vocabulary[:50])  # Limit to 50 terms
            prompt_parts.append(f"Key terms: {vocab_str}")
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        return " ".join(prompt_parts)
    
    def create_transcription_config(
        self,
        context: Optional[str] = None,
        temperature: float = 0.0
    ) -> dict:
        """Create configuration dict for transcription."""
        return {
            "initial_prompt": self.build_prompt(context),
            "temperature": temperature,
            "language": "en"
        }


# Example domain-specific vocabularies
AGENT_COMMANDS = [
    "open browser", "close window", "send email", "check calendar",
    "create task", "set reminder", "search web", "take screenshot",
    "system info", "run command", "open file", "save document"
]

TECHNICAL_TERMS = [
    "API", "endpoint", "webhook", "async", "callback",
    "JSON", "REST", "GraphQL", "websocket", "OAuth",
    "Docker", "Kubernetes", "CI/CD", "microservice"
]
```

---

## 8. IMPLEMENTATION CODE EXAMPLES

### 8.1 Complete Integration Example

```python
"""
Complete STT System Integration for OpenClaw
Combines wake word detection, VAD, and multi-engine STT.
"""
import os
from typing import Optional, Callable
import threading
import queue

class OpenClawSTTSystem:
    """
    Complete STT system for OpenClaw AI Agent.
    
    Features:
    - Wake word activation ("Hey Claw")
    - Voice activity detection
    - Multi-engine STT with fallback
    - Real-time streaming transcription
    """
    
    def __init__(
        self,
        porcupine_access_key: str,
        azure_key: Optional[str] = None,
        azure_region: str = "westus",
        openai_key: Optional[str] = None,
        wake_words: Optional[list] = None
    ):
        # Configuration
        self.porcupine_access_key = porcupine_access_key
        self.azure_key = azure_key or os.getenv("AZURE_SPEECH_KEY")
        self.azure_region = azure_region
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize components
        self._init_wake_word_detector(wake_words or ["porcupine"])
        self._init_stt_engines()
        self._init_audio_pipeline()
        
        # State
        self.is_active = False
        self.command_queue: queue.Queue = queue.Queue()
        self._command_callbacks: List[Callable[[str], None]] = []
    
    def _init_wake_word_detector(self, wake_words: list):
        """Initialize wake word detection."""
        from pvporcupine import create
        from pvrecorder import PvRecorder
        
        self.porcupine = create(
            access_key=self.porcupine_access_key,
            keywords=wake_words
        )
        self.recorder = PvRecorder(
            device_index=-1,
            frame_length=self.porcupine.frame_length
        )
    
    def _init_stt_engines(self):
        """Initialize STT engines."""
        # Primary: Azure Speech
        if self.azure_key:
            self.azure_recognizer = AzureSpeechRecognizer(
                subscription_key=self.azure_key,
                region=self.azure_region
            )
        else:
            self.azure_recognizer = None
        
        # Fallback: Whisper Local
        self.whisper_recognizer = WhisperRecognizer(
            model_size="base",
            device="cpu"
        )
    
    def _init_audio_pipeline(self):
        """Initialize audio preprocessing."""
        self.preprocessor = AudioPreprocessor(
            apply_noise_reduction=True,
            apply_normalization=True
        )
        self.vad = WebRTCVAD(mode=2)
    
    def add_command_callback(self, callback: Callable[[str], None]):
        """Add callback for recognized commands."""
        self._command_callbacks.append(callback)
    
    def start(self):
        """Start the STT system."""
        print("=" * 50)
        print("OpenClaw STT System Starting...")
        print("=" * 50)
        
        # Start wake word detection
        self.is_active = True
        self.recorder.start()
        
        print(f"Listening for wake words: {self.porcupine.keyword_names}")
        print("Say the wake word to activate...")
        
        # Main loop
        try:
            while self.is_active:
                # Read audio frame
                pcm = self.recorder.read()
                
                # Check for wake word
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    detected = self.porcupine.keyword_names[keyword_index]
                    print(f"\n[ACTIVATED] Wake word detected: '{detected}'")
                    self._handle_activation()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def _handle_activation(self):
        """Handle wake word activation - capture and transcribe command."""
        print("Listening for command... (speak now)")
        
        # Capture command audio (5 second window)
        command_audio = self._capture_command(duration_ms=5000)
        
        if command_audio:
            # Preprocess
            processed = self.preprocessor.preprocess(command_audio)
            
            # Transcribe
            transcript = self._transcribe(processed)
            
            if transcript:
                print(f"[TRANSCRIBED] '{transcript}'")
                
                # Notify callbacks
                for callback in self._command_callbacks:
                    try:
                        callback(transcript)
                    except Exception as e:
                        print(f"Callback error: {e}")
            else:
                print("[ERROR] Could not transcribe command")
        
        print("\nListening for wake word...")
    
    def _capture_command(self, duration_ms: int = 5000) -> Optional[bytes]:
        """Capture audio after wake word detection."""
        frames_needed = int(duration_ms / 1000 * 16000 / self.porcupine.frame_length)
        frames = []
        
        for _ in range(frames_needed):
            frame = self.recorder.read()
            frames.append(frame)
        
        # Convert frames to bytes
        import struct
        audio_bytes = b"".join([
            struct.pack("h" * len(frame), *frame) for frame in frames
        ])
        
        return audio_bytes
    
    def _transcribe(self, audio: bytes) -> Optional[str]:
        """Transcribe audio using available engines."""
        # Try Azure first
        if self.azure_recognizer:
            try:
                result = self.azure_recognizer.recognize_once_from_microphone()
                if result and "text" in result:
                    return result["text"]
            except Exception as e:
                print(f"Azure STT error: {e}")
        
        # Fallback to Whisper
        try:
            result = self.whisper_recognizer.transcribe_local(audio)
            return result.get("text")
        except Exception as e:
            print(f"Whisper error: {e}")
        
        return None
    
    def stop(self):
        """Stop the STT system."""
        self.is_active = False
        self.recorder.stop()
        self.recorder.delete()
        self.porcupine.delete()


# Usage Example
if __name__ == "__main__":
    # Initialize system
    stt_system = OpenClawSTTSystem(
        porcupine_access_key="YOUR_PICOVOICE_ACCESS_KEY",
        azure_key="YOUR_AZURE_KEY",
        azure_region="westus",
        wake_words=["porcupine"]  # Or custom wake word
    )
    
    # Add command handler
    def handle_command(command: str):
        print(f"Processing command: {command}")
        # Integrate with GPT-5.2 agent here
    
    stt_system.add_command_callback(handle_command)
    
    # Start listening
    stt_system.start()
```

---

## 9. CONFIGURATION & DEPLOYMENT

### 9.1 Environment Configuration

```yaml
# config/stt_config.yaml
# OpenClaw STT System Configuration

audio:
  sample_rate: 16000
  channels: 1
  format: "int16"
  frame_duration_ms: 30
  buffer_duration_ms: 500

wake_word:
  engine: "porcupine"
  access_key: "${PORCUPINE_ACCESS_KEY}"
  keywords:
    - "porcupine"
    - "hey_claw"  # Custom trained
  sensitivity: 0.7
  confirmation_required: false
  confirmation_timeout_ms: 3000

vad:
  engine: "webrtc"  # or "silero", "cobra"
  mode: 2  # 0-3 for webrtc
  padding_duration_ms: 300
  min_speech_duration_ms: 250

stt:
  primary_engine: "azure"
  fallback_engine: "whisper_local"
  confidence_threshold: 0.7
  
  azure:
    subscription_key: "${AZURE_SPEECH_KEY}"
    region: "westus"
    language: "en-US"
    endpoint_id: null  # For custom models
  
  whisper:
    model_size: "base"  # tiny, base, small, medium, large
    device: "cpu"  # or "cuda"
    language: "en"
    initial_prompt: "OpenClaw AI Agent command system"
  
  google:
    project_id: "${GOOGLE_CLOUD_PROJECT}"
    location: "global"
    model: "chirp_2"

preprocessing:
  enabled: true
  noise_reduction: true
  noise_reduction_strength: 0.8
  normalization: true
  target_db: -20.0

streaming:
  enabled: true
  interim_results: true
  max_alternatives: 1
  enable_speaker_diarization: false

custom_vocabulary:
  enabled: true
  terms_file: "config/custom_terms.txt"
  boost_weight: 10.0
```

### 9.2 Requirements File

```txt
# requirements-stt.txt
# Speech-to-Text dependencies for OpenClaw

# Audio capture and processing
pyaudio==0.2.14
pvrecorder==1.2.2

# Wake word detection
pvporcupine==3.0.3

# Voice Activity Detection
webrtcvad-wheels==2.0.14

# Noise reduction
noisereduce==3.0.3

# Azure Speech SDK
azure-cognitiveservices-speech==1.40.0

# OpenAI Whisper
openai-whisper==20231117
openai==1.35.0

# Google Cloud Speech
google-cloud-speech==2.26.0

# Windows Speech Recognition
SpeechRecognition==3.10.4
pywin32==306

# Utilities
numpy==1.26.4
scipy==1.13.0
pyyaml==6.0.1
python-dotenv==1.0.1
```

---

## 10. PERFORMANCE OPTIMIZATION

### 10.1 Latency Targets

| Component | Target Latency | Maximum |
|-----------|---------------|---------|
| Wake Word Detection | < 50ms | 100ms |
| VAD Processing | < 10ms | 30ms |
| Audio Preprocessing | < 20ms | 50ms |
| STT (Azure) | < 200ms | 500ms |
| STT (Whisper Local) | < 1000ms | 2000ms |
| End-to-End | < 500ms | 1500ms |

### 10.2 Optimization Strategies

```python
"""
Performance Optimization Utilities
"""
import functools
import time
from typing import Callable, Any
import threading

class STTPerformanceMonitor:
    """Monitor and optimize STT system performance."""
    
    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()
    
    def measure(self, name: str):
        """Decorator to measure function execution time."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                
                with self._lock:
                    if name not in self.metrics:
                        self.metrics[name] = []
                    self.metrics[name].append(elapsed)
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self, name: str) -> dict:
        """Get statistics for a metric."""
        with self._lock:
            values = self.metrics.get(name, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values)
        }
    
    def print_report(self):
        """Print performance report."""
        print("\n" + "=" * 60)
        print("STT PERFORMANCE REPORT")
        print("=" * 60)
        
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Mean:  {stats['mean']:.2f}ms")
            print(f"  Min:   {stats['min']:.2f}ms")
            print(f"  Max:   {stats['max']:.2f}ms")
            print(f"  P95:   {stats['p95']:.2f}ms")


# Optimization: Pre-load models
class ModelCache:
    """Cache for pre-loaded STT models."""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_whisper(self, model_size: str):
        """Pre-load Whisper model."""
        if model_size not in self._models:
            import whisper
            print(f"Pre-loading Whisper model: {model_size}")
            self._models[f"whisper_{model_size}"] = whisper.load_model(model_size)
        return self._models[f"whisper_{model_size}"]
    
    def get_model(self, name: str):
        """Get cached model."""
        return self._models.get(name)
```

---

## APPENDIX A: API REFERENCE

### A.1 STT Engine Interfaces

```python
# Common interface for all STT engines
class STTEngine(Protocol):
    def transcribe(self, audio: bytes) -> dict:
        ...
    
    def transcribe_streaming(
        self,
        audio_stream: Iterator[bytes],
        interim_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        ...
```

### A.2 Error Handling

```python
class STTException(Exception):
    """Base exception for STT errors."""
    pass

class WakeWordException(STTException):
    """Wake word detection error."""
    pass

class TranscriptionException(STTException):
    """Transcription error."""
    pass

class AudioCaptureException(STTException):
    """Audio capture error."""
    pass
```

---

## APPENDIX B: TROUBLESHOOTING

### B.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No microphone detected | PyAudio not installed | Install PyAudio with `pip install pyaudio` |
| Wake word not detected | Sensitivity too low | Increase sensitivity (0.7-1.0) |
| High false positives | Sensitivity too high | Decrease sensitivity (0.3-0.5) |
| Poor transcription | Noisy environment | Enable noise reduction |
| High latency | Large model | Use smaller model (tiny/base) |
| API errors | Invalid credentials | Check API keys in environment |

### B.2 Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# For specific modules
logging.getLogger("azure").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
```

---

*Document Version: 1.0*
*Last Updated: 2025*
*For: OpenClaw Windows 10 AI Agent Framework*
