"""
OpenClaw AI Agent - Audio Core Implementation
Real-time audio processing and streaming for Windows 10

This module provides the core audio infrastructure for the OpenClaw AI agent system,
including capture, playback, mixing, and streaming capabilities.
"""

import asyncio
import numpy as np
import sounddevice as sd
import threading
import time
import statistics
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    INT16 = "int16"
    FLOAT32 = "float32"
    INT32 = "int32"


def _load_audio_defaults() -> dict:
    """Load audio defaults from tts_config.yaml if available."""
    try:
        from config_loader import get_config
        return get_config("tts_config", "tts.audio", {})
    except (ImportError, Exception):
        return {}

_AUDIO_DEFAULTS = _load_audio_defaults()


@dataclass
class AudioConfig:
    """Audio configuration parameters"""
    sample_rate: int = _AUDIO_DEFAULTS.get('sample_rate', 48000)
    channels: int = _AUDIO_DEFAULTS.get('channels', 1)
    block_size: int = _AUDIO_DEFAULTS.get('buffer_size', 960)
    latency: str = "low"
    format: AudioFormat = AudioFormat.FLOAT32

    @property
    def frame_duration_ms(self) -> float:
        """Calculate frame duration in milliseconds"""
        return (self.block_size / self.sample_rate) * 1000


@dataclass
class AudioFrame:
    """Represents an audio frame"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int


class RingBuffer:
    """
    Thread-safe ring buffer for audio streaming
    """
    
    def __init__(self, capacity: int, channels: int = 1, dtype=np.float32):
        self.capacity = capacity
        self.channels = channels
        self.dtype = dtype
        
        self._buffer = np.zeros((capacity, channels), dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self._lock = threading.Lock()
        self._write_event = threading.Event()
        
    def write(self, data: np.ndarray) -> int:
        """Write audio data to ring buffer"""
        with self._lock:
            samples_to_write = min(len(data), self.capacity - self._available)
            
            for i in range(samples_to_write):
                self._buffer[self._write_pos] = data[i]
                self._write_pos = (self._write_pos + 1) % self.capacity
            
            self._available += samples_to_write
            
        if samples_to_write > 0:
            self._write_event.set()
            
        return samples_to_write
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read audio data from ring buffer"""
        with self._lock:
            samples_to_read = min(num_samples, self._available)
            
            result = np.zeros((samples_to_read, self.channels), dtype=self.dtype)
            for i in range(samples_to_read):
                result[i] = self._buffer[self._read_pos]
                self._read_pos = (self._read_pos + 1) % self.capacity
            
            self._available -= samples_to_read
            
            if self._available == 0:
                self._write_event.clear()
                
            return result
    
    def available(self) -> int:
        """Get number of available samples"""
        with self._lock:
            return self._available
    
    def wait_for_data(self, timeout: float = None) -> bool:
        """Wait for data to be available"""
        return self._write_event.wait(timeout)


class AudioCapture:
    """
    Low-latency audio capture for Windows 10
    """
    
    def __init__(self, config: AudioConfig = None, device: int = None):
        self.config = config or AudioConfig()
        self.device = device
        self.stream: Optional[sd.InputStream] = None
        self._ring_buffer: Optional[RingBuffer] = None
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[AudioFrame], None]] = []
        
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info: dict, status: sd.CallbackFlags):
        """SoundDevice audio callback"""
        if status:
            logger.warning(f"Audio capture status: {status}")
        
        # Create audio frame
        frame = AudioFrame(
            data=indata.copy(),
            timestamp=time.time(),
            sample_rate=self.config.sample_rate,
            channels=self.config.channels
        )
        
        # Write to ring buffer
        if self._ring_buffer:
            self._ring_buffer.write(indata)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(frame)
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Capture callback error: {e}")
    
    def register_callback(self, callback: Callable[[AudioFrame], None]):
        """Register a callback for audio frames"""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[AudioFrame], None]):
        """Unregister a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start(self) -> 'AudioCapture':
        """Start audio capture"""
        if self._is_capturing:
            return self
        
        # Initialize ring buffer (100ms capacity)
        buffer_capacity = int(self.config.sample_rate * 0.1)
        self._ring_buffer = RingBuffer(
            capacity=buffer_capacity,
            channels=self.config.channels,
            dtype=np.float32
        )
        
        # Create and start stream
        dtype_map = {
            AudioFormat.INT16: np.int16,
            AudioFormat.FLOAT32: np.float32,
            AudioFormat.INT32: np.int32
        }
        
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=dtype_map[self.config.format],
            blocksize=self.config.block_size,
            latency=self.config.latency,
            callback=self._audio_callback
        )
        
        self.stream.start()
        self._is_capturing = True
        logger.info(f"Audio capture started: {self.config.sample_rate}Hz, "
                   f"{self.config.channels}ch, {self.config.block_size} samples")
        
        return self
    
    def stop(self):
        """Stop audio capture"""
        self._is_capturing = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self._ring_buffer = None
        logger.info("Audio capture stopped")
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read audio data from capture buffer"""
        if not self._ring_buffer:
            return None
        return self._ring_buffer.read(num_samples)
    
    def read_available(self) -> int:
        """Get number of available samples"""
        if not self._ring_buffer:
            return 0
        return self._ring_buffer.available()


class AudioPlayback:
    """
    Low-latency audio playback for Windows 10
    """
    
    def __init__(self, config: AudioConfig = None, device: int = None):
        self.config = config or AudioConfig()
        self.device = device
        self.stream: Optional[sd.OutputStream] = None
        self._ring_buffer: Optional[RingBuffer] = None
        self._is_playing = False
        self._underflows = 0
        
    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags):
        """SoundDevice playback callback"""
        if status:
            if 'underflow' in str(status).lower():
                self._underflows += 1
                if self._underflows % 100 == 0:
                    logger.warning(f"Audio underflows: {self._underflows}")
        
        # Read from ring buffer
        if self._ring_buffer and self._ring_buffer.available() >= frames:
            data = self._ring_buffer.read(frames)
            outdata[:] = data
        else:
            # Output silence if not enough data
            outdata[:] = 0
    
    def start(self) -> 'AudioPlayback':
        """Start audio playback"""
        if self._is_playing:
            return self
        
        # Initialize ring buffer (200ms capacity)
        buffer_capacity = int(self.config.sample_rate * 0.2)
        self._ring_buffer = RingBuffer(
            capacity=buffer_capacity,
            channels=self.config.channels,
            dtype=np.float32
        )
        
        # Create and start stream
        dtype_map = {
            AudioFormat.INT16: np.int16,
            AudioFormat.FLOAT32: np.float32,
            AudioFormat.INT32: np.int32
        }
        
        self.stream = sd.OutputStream(
            device=self.device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=dtype_map[self.config.format],
            blocksize=self.config.block_size,
            latency=self.config.latency,
            callback=self._audio_callback
        )
        
        self.stream.start()
        self._is_playing = True
        logger.info(f"Audio playback started: {self.config.sample_rate}Hz, "
                   f"{self.config.channels}ch, {self.config.block_size} samples")
        
        return self
    
    def stop(self):
        """Stop audio playback"""
        self._is_playing = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self._ring_buffer = None
        logger.info("Audio playback stopped")
    
    def write(self, data: np.ndarray) -> int:
        """Write audio data to playback buffer"""
        if not self._ring_buffer:
            return 0
        return self._ring_buffer.write(data)
    
    def get_buffer_space(self) -> int:
        """Get available buffer space"""
        if not self._ring_buffer:
            return 0
        return self._ring_buffer.capacity - self._ring_buffer.available()


class AudioMixer:
    """
    Real-time audio mixer for multiple input streams
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._output_callbacks: List[Callable[[np.ndarray], None]] = []
        self._is_running = False
        self._mix_thread: Optional[threading.Thread] = None
        self._master_gain = 1.0
        
        # VAD and ducking
        self._vad_enabled = True
        self._ducking_threshold = 0.05
        self._ducking_attenuation = 0.3
        
    def add_stream(self, stream_id: str, gain: float = 1.0):
        """Add an input stream"""
        self._streams[stream_id] = {
            'ring_buffer': RingBuffer(
                capacity=int(self.config.sample_rate * 0.5),
                channels=self.config.channels
            ),
            'gain': gain,
            'vad_level': 0.0,
            'active': True
        }
        logger.info(f"Added stream: {stream_id}")
    
    def remove_stream(self, stream_id: str):
        """Remove an input stream"""
        if stream_id in self._streams:
            del self._streams[stream_id]
            logger.info(f"Removed stream: {stream_id}")
    
    def push_audio(self, stream_id: str, data: np.ndarray):
        """Push audio to a stream"""
        if stream_id not in self._streams:
            return
        
        stream = self._streams[stream_id]
        
        # Write to ring buffer
        stream['ring_buffer'].write(data)
        
        # Update VAD level
        if self._vad_enabled:
            rms = np.sqrt(np.mean(data ** 2))
            stream['vad_level'] = 0.9 * stream['vad_level'] + 0.1 * rms
    
    def set_stream_gain(self, stream_id: str, gain: float):
        """Set gain for a stream"""
        if stream_id in self._streams:
            self._streams[stream_id]['gain'] = gain
    
    def set_master_gain(self, gain: float):
        """Set master gain"""
        self._master_gain = gain
    
    def _mix_loop(self):
        """Main mixing loop"""
        block_duration = self.config.block_size / self.config.sample_rate
        
        while self._is_running:
            start_time = time.time()
            
            # Find dominant voice
            dominant_stream = None
            max_vad = 0
            
            if self._vad_enabled:
                for stream_id, stream in self._streams.items():
                    if stream['vad_level'] > max_vad:
                        max_vad = stream['vad_level']
                        dominant_stream = stream_id
            
            # Mix streams
            mixed = np.zeros((self.config.block_size, self.config.channels), 
                           dtype=np.float32)
            active_count = 0
            
            for stream_id, stream in self._streams.items():
                if stream['ring_buffer'].available() >= self.config.block_size:
                    data = stream['ring_buffer'].read(self.config.block_size)
                    gain = stream['gain']
                    
                    # Apply ducking
                    if self._vad_enabled and dominant_stream != stream_id:
                        if max_vad > self._ducking_threshold:
                            gain *= self._ducking_attenuation
                    
                    mixed += data * gain
                    active_count += 1
            
            # Normalize
            if active_count > 1:
                mixed /= np.sqrt(active_count)
            
            # Apply master gain and clip
            mixed *= self._master_gain
            mixed = np.clip(mixed, -1.0, 1.0)
            
            # Output to callbacks
            for callback in self._output_callbacks:
                try:
                    callback(mixed.copy())
                except (OSError, ValueError, RuntimeError) as e:
                    logger.error(f"Mix callback error: {e}")
            
            # Sleep to maintain timing
            elapsed = time.time() - start_time
            sleep_time = max(0, block_duration - elapsed)
            time.sleep(sleep_time)
    
    def register_output(self, callback: Callable[[np.ndarray], None]):
        """Register output callback"""
        self._output_callbacks.append(callback)
    
    def start(self):
        """Start the mixer"""
        if self._is_running:
            return
        
        self._is_running = True
        self._mix_thread = threading.Thread(target=self._mix_loop)
        self._mix_thread.start()
        logger.info("Audio mixer started")
    
    def stop(self):
        """Stop the mixer"""
        self._is_running = False
        
        if self._mix_thread:
            self._mix_thread.join(timeout=1.0)
            self._mix_thread = None
        
        logger.info("Audio mixer stopped")


class LatencyMonitor:
    """
    Monitor and track audio latency metrics
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._capture_latency_ms = 0.0
        self._network_latency_ms = 0.0
        self._playback_latency_ms = 0.0
        self._total_latency_ms = 0.0
        self._history: List[Dict[str, float]] = []
        
    def update_capture_latency(self, latency_ms: float):
        """Update capture latency"""
        self._capture_latency_ms = latency_ms
        self._update_total()
    
    def update_network_latency(self, latency_ms: float):
        """Update network latency"""
        self._network_latency_ms = latency_ms
        self._update_total()
    
    def update_playback_latency(self, latency_ms: float):
        """Update playback latency"""
        self._playback_latency_ms = latency_ms
        self._update_total()
    
    def _update_total(self):
        """Update total latency"""
        self._total_latency_ms = (
            self._capture_latency_ms +
            self._network_latency_ms +
            self._playback_latency_ms
        )
        
        self._history.append({
            'timestamp': time.time(),
            'capture': self._capture_latency_ms,
            'network': self._network_latency_ms,
            'playback': self._playback_latency_ms,
            'total': self._total_latency_ms
        })
        
        # Trim history
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self._history:
            return {}
        
        totals = [h['total'] for h in self._history]
        
        return {
            'current_ms': self._total_latency_ms,
            'avg_ms': statistics.mean(totals),
            'min_ms': min(totals),
            'max_ms': max(totals),
            'p95_ms': np.percentile(totals, 95) if len(totals) >= 20 else max(totals),
            'p99_ms': np.percentile(totals, 99) if len(totals) >= 100 else max(totals),
        }
    
    def get_current(self) -> Dict[str, float]:
        """Get current latency values"""
        return {
            'capture_ms': self._capture_latency_ms,
            'network_ms': self._network_latency_ms,
            'playback_ms': self._playback_latency_ms,
            'total_ms': self._total_latency_ms,
        }


class AudioCore:
    """
    Main audio core that integrates all components
    """
    
    def __init__(self):
        self.capture: Optional[AudioCapture] = None
        self.playback: Optional[AudioPlayback] = None
        self.mixer: Optional[AudioMixer] = None
        self.latency_monitor = LatencyMonitor()
        self._is_initialized = False
        
    def initialize(self, capture_config: AudioConfig = None,
                   playback_config: AudioConfig = None):
        """Initialize the audio core"""
        if self._is_initialized:
            return
        
        # Initialize components
        self.capture = AudioCapture(config=capture_config)
        self.playback = AudioPlayback(config=playback_config)
        self.mixer = AudioMixer(config=playback_config)
        
        # Connect mixer to playback
        self.mixer.register_output(self._on_mixed_audio)
        
        self._is_initialized = True
        logger.info("Audio core initialized")
    
    def _on_mixed_audio(self, audio_data: np.ndarray):
        """Handle mixed audio output"""
        if self.playback:
            self.playback.write(audio_data)
    
    def start(self):
        """Start all audio components"""
        if not self._is_initialized:
            raise RuntimeError("Audio core not initialized")
        
        self.capture.start()
        self.playback.start()
        self.mixer.start()
        
        logger.info("Audio core started")
    
    def stop(self):
        """Stop all audio components"""
        if self.mixer:
            self.mixer.stop()
        if self.playback:
            self.playback.stop()
        if self.capture:
            self.capture.stop()
        
        logger.info("Audio core stopped")
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        return self.latency_monitor.get_statistics()


# Utility functions
def enumerate_devices() -> Dict[str, List[Dict[str, Any]]]:
    """Enumerate all audio devices"""
    devices = sd.query_devices()
    
    input_devices = []
    output_devices = []
    
    for idx, device in enumerate(devices):
        info = {
            'index': idx,
            'name': device['name'],
            'sample_rate': device['default_samplerate'],
        }
        
        if device['max_input_channels'] > 0:
            info['channels'] = device['max_input_channels']
            info['latency'] = device['default_low_input_latency']
            input_devices.append(info)
        
        if device['max_output_channels'] > 0:
            info['channels'] = device['max_output_channels']
            info['latency'] = device['default_low_output_latency']
            output_devices.append(info)
    
    return {
        'input': input_devices,
        'output': output_devices
    }


def get_default_devices() -> Dict[str, Optional[int]]:
    """Get default input and output device indices"""
    try:
        return {
            'input': sd.default.device[0],
            'output': sd.default.device[1]
        }
    except (sd.PortAudioError, OSError):
        return {'input': None, 'output': None}


# Example usage
if __name__ == "__main__":
    # Enumerate devices
    devices = enumerate_devices()
    print("Input devices:", devices['input'])
    print("Output devices:", devices['output'])
    
    # Create and start audio core
    core = AudioCore()
    
    config = AudioConfig(
        sample_rate=48000,
        channels=1,
        block_size=960,
        latency="low"
    )
    
    core.initialize(capture_config=config, playback_config=config)
    
    # Add a stream to mixer
    core.mixer.add_stream("test_stream", gain=1.0)
    
    # Route capture to mixer
    def on_capture(frame: AudioFrame):
        core.mixer.push_audio("test_stream", frame.data)
    
    core.capture.register_callback(on_capture)
    
    # Start
    core.start()
    
    print("Audio running. Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
            stats = core.get_latency_stats()
            if stats:
                print(f"Latency - Avg: {stats.get('avg_ms', 0):.1f}ms, "
                      f"Current: {stats.get('current_ms', 0):.1f}ms")
    except KeyboardInterrupt:
        print("\nStopping...")
    
    core.stop()
