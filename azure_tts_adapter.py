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
    voice_type: str  # Neural, Standard
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

        # Set default voice
        config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"

        # Enable sentence boundary for better control
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
        # Reinitialize synthesizer with new voice
        self._initialize_synthesizer()

    def speak_text(self, text: str) -> Dict[str, Any]:
        """
        Synthesize text to speech

        Returns:
            Result dictionary with status and metadata
        """
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
        # Create audio config for file output
        audio_config = AudioConfig(filename=output_path)
        synthesizer = SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

        if ssml:
            result = synthesizer.speak_ssml_async(text).get()
        else:
            result = synthesizer.speak_text_async(text).get()

        synthesizer = None  # Cleanup

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


# Usage Example
if __name__ == "__main__":
    azure_tts = AzureTTSAdapter()

    # Event handlers
    def on_word_boundary(evt):
        print(f"Word: {evt.text} at {evt.audio_offset}ms")

    azure_tts.on('word_boundary', on_word_boundary)

    # Speak with neural voice
    azure_tts.set_voice("en-US-AvaMultilingualNeural")
    result = azure_tts.speak_text("Hello from Azure Cognitive Services!")
    print(f"Synthesis result: {result}")

    # SSML with style
    ssml = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
           xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="en-US-AvaMultilingualNeural">
            <mstts:express-as style="cheerful">
                I'm excited to help you with your tasks today!
            </mstts:express-as>
        </voice>
    </speak>
    """
    azure_tts.speak_ssml(ssml)
