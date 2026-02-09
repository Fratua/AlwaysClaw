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
import logging
import yaml

logger = logging.getLogger(__name__)


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
    priority: int = 5  # 1-10, lower = higher priority
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

        # Initialize adapters
        self._adapters: Dict[TTSEngine, Any] = {}
        self._voice_manager: Optional[Any] = None
        self._ssml_processor: Optional[Any] = None
        self._audio_pipeline: Optional[Any] = None

        # Request queue
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queue_counter = 0  # Monotonic tie-breaker for equal priorities
        self._processing = False
        self._worker_thread: Optional[threading.Thread] = None

        # Statistics
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
                logger.info("[TTS] Azure adapter initialized")
        except (ImportError, OSError) as e:
            logger.warning(f"[TTS] Azure TTS not available: {e}")

        # ElevenLabs
        try:
            if os.environ.get('ELEVENLABS_API_KEY'):
                from elevenlabs_tts_adapter import ElevenLabsTTSAdapter
                self._adapters[TTSEngine.ELEVENLABS] = ElevenLabsTTSAdapter()
                logger.info("[TTS] ElevenLabs adapter initialized")
        except (ImportError, OSError) as e:
            logger.warning(f"[TTS] ElevenLabs TTS not available: {e}")

        # OpenAI
        try:
            if os.environ.get('OPENAI_API_KEY'):
                from openai_tts_adapter import OpenAITTSAdapter
                self._adapters[TTSEngine.OPENAI] = OpenAITTSAdapter()
                logger.info("[TTS] OpenAI adapter initialized")
        except (ImportError, OSError) as e:
            logger.warning(f"[TTS] OpenAI TTS not available: {e}")

        # SAPI (always available on Windows)
        try:
            from sapi_tts_adapter import SAPITTSAdapter
            self._adapters[TTSEngine.SAPI] = SAPITTSAdapter()
            logger.info("[TTS] SAPI adapter initialized")
        except (ImportError, OSError) as e:
            logger.warning(f"[TTS] SAPI TTS not available: {e}")

    def _start_worker(self):
        """Start request processing worker"""
        self._processing = True
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=False)
        self._worker_thread.start()

    def _process_queue(self):
        """Process TTS requests from queue"""
        while self._processing and not self._shutdown_event.is_set():
            try:
                priority, _counter, request, callback = self._request_queue.get(timeout=5)
                response = self._execute_request(request)
                if callback:
                    callback(response)
            except queue.Empty:
                continue
            except (RuntimeError, OSError) as e:
                logger.error(f"[TTS] Queue processing error: {e}")

    def _execute_request(self, request: TTSRequest) -> TTSResponse:
        """Execute TTS request"""
        start_time = time.time()
        self._stats['requests_total'] += 1

        try:
            # Select engine
            engine = self._select_engine(request)
            adapter = self._adapters.get(engine)

            if not adapter:
                return TTSResponse(
                    success=False,
                    error=f"Engine {engine} not available"
                )

            # Prepare text/SSML
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
                    duration_ms=0  # Streaming
                )
            else:
                audio_data = adapter.text_to_speech(text, voice_id=request.voice_id)
                if audio_data is None:
                    raise RuntimeError(f"TTS adapter {engine.value} returned None")
                response = TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    engine_used=engine.value
                )

            self._stats['requests_success'] += 1

        except (RuntimeError, OSError, ValueError) as e:
            self._stats['requests_failed'] += 1
            response = TTSResponse(
                success=False,
                error=str(e)
            )

        # Update latency stats using Welford's online algorithm
        latency_ms = (time.time() - start_time) * 1000
        n = self._stats['requests_total']
        old_mean = self._stats['avg_latency_ms']
        self._stats['avg_latency_ms'] = old_mean + (latency_ms - old_mean) / n

        return response

    def _select_engine(self, request: TTSRequest) -> TTSEngine:
        """Select optimal TTS engine"""
        # Use specified provider if available
        if request.provider and request.provider in self._adapters:
            return request.provider

        # Use priority order
        for engine_name in self.config['priority_order']:
            try:
                engine = TTSEngine(engine_name)
            except ValueError:
                logger.warning(f"Unknown TTS engine in priority_order: {engine_name}")
                continue
            if engine in self._adapters:
                return engine

        # Fallback to any available
        if not self._adapters:
            raise RuntimeError("No TTS adapters available")
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
        """
        Main TTS interface

        Args:
            text: Text to synthesize
            voice_id: Specific voice to use
            provider: Preferred TTS provider
            ssml: Whether text is SSML
            streaming: Use streaming mode
            async_mode: Queue request asynchronously
            callback: Callback for async completion

        Returns:
            TTSResponse if sync mode, None if async
        """
        request = TTSRequest(
            text=text,
            voice_id=voice_id,
            provider=provider,
            ssml=ssml,
            streaming=streaming,
            priority=5
        )

        if async_mode:
            self._queue_counter += 1
            self._request_queue.put((request.priority, self._queue_counter, request, callback))
            return None
        else:
            return self._execute_request(request)

    def speak_ssml(self, ssml: str, **kwargs) -> Optional[TTSResponse]:
        """Speak SSML content"""
        return self.speak(ssml, ssml=True, **kwargs)

    def stop(self):
        """Stop all synthesis"""
        for adapter in self._adapters.values():
            if hasattr(adapter, 'stop'):
                adapter.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        return self._stats.copy()

    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines"""
        return [engine.value for engine in self._adapters.keys()]

    def shutdown(self):
        """Shutdown orchestrator"""
        self._processing = False
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            if self._worker_thread.is_alive():
                logger.warning("TTS worker thread did not terminate in time")

        for adapter in self._adapters.values():
            if hasattr(adapter, 'shutdown'):
                adapter.shutdown()


# Usage Example
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = TTSOrchestrator()

    # Simple speech
    response = orchestrator.speak("Hello, I am OpenClaw AI Agent!")
    print(f"Response: {response}")

    # Get statistics
    stats = orchestrator.get_stats()
    print(f"TTS Stats: {stats}")

    # Cleanup
    orchestrator.shutdown()
