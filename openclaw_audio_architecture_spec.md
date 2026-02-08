# OpenClaw Windows 10 AI Agent - Real-Time Audio Processing & Streaming Architecture

## Technical Specification Document

**Version:** 1.0  
**Date:** 2024  
**Platform:** Windows 10  
**Target:** GPT-5.2 AI Agent with 24/7 Operation

---

## Executive Summary

This document provides a comprehensive technical specification for the real-time audio processing and streaming architecture of the OpenClaw-inspired Windows 10 AI agent system. The architecture is designed to support:

- Real-time voice communication via Twilio
- WebRTC peer-to-peer and SFU-based conferencing
- Low-latency audio capture and playback
- 24/7 continuous operation with cron jobs and heartbeat monitoring
- Integration with GPT-5.2 high-thinking capability
- Full system access for browser control, Gmail, and agentic loops

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [WebRTC Integration](#2-webrtc-integration)
3. [Media Server Architecture](#3-media-server-architecture)
4. [Audio Capture System](#4-audio-capture-system)
5. [Audio Playback System](#5-audio-playback-system)
6. [Streaming Protocols](#6-streaming-protocols)
7. [Buffer Management & Latency Optimization](#7-buffer-management--latency-optimization)
8. [Audio Mixing & Routing](#8-audio-mixing--routing)
9. [Twilio Integration](#9-twilio-integration)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPENCLAW AI AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   GPT-5.2    │    │  Agent Core  │    │   Identity   │                  │
│  │   Engine     │◄──►│   (15 Loops) │◄──►│    System    │                  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘                  │
│                             │                                               │
│                    ┌────────┴────────┐                                      │
│                    │  Audio Manager  │                                      │
│                    │    (Core)       │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│     ┌───────────────────────┼───────────────────────┐                      │
│     │                       │                       │                      │
│     ▼                       ▼                       ▼                      │
│ ┌──────────┐          ┌──────────┐          ┌──────────┐                   │
│ │  Capture │          │  Stream  │          │ Playback │                   │
│ │  Module  │          │  Manager │          │  Module  │                   │
│ └────┬─────┘          └────┬─────┘          └────┬─────┘                   │
│      │                     │                     │                         │
│      ▼                     ▼                     ▼                         │
│ ┌──────────┐          ┌──────────┐          ┌──────────┐                   │
│ │ Windows  │          │ WebRTC/  │          │ Windows  │                   │
│ │ WASAPI   │          │  Media   │          │ WASAPI   │                   │
│ │  Core    │          │  Server  │          │  Core    │                   │
│ │  Audio   │          │          │          │  Audio   │                   │
│ └──────────┘          └──────────┘          └──────────┘                   │
│                              │                                             │
│                              ▼                                             │
│                       ┌──────────────┐                                     │
│                       │   Twilio     │                                     │
│                       │   Voice/SMS  │                                     │
│                       └──────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Audio Capture | Windows WASAPI (IAudioClient3) | Low-latency microphone input |
| Audio Playback | Windows WASAPI (IAudioClient3) | Low-latency speaker output |
| Audio Processing | Python + NumPy | Real-time DSP operations |
| WebRTC Stack | aiortc / python-rtmixer | P2P and SFU communication |
| Media Server | Janus Gateway / Mediasoup | Selective forwarding unit |
| Telephony | Twilio Voice SDK | PSTN integration |
| Codec | Opus (libopus) | Audio compression |
| Signaling | WebSocket / Socket.IO | Session management |

---

## 2. WebRTC Integration

### 2.1 WebRTC Architecture Selection

For the OpenClaw AI agent system, we recommend a **hybrid architecture**:

| Scenario | Architecture | Rationale |
|----------|--------------|-----------|
| 1:1 Voice Calls | P2P (Peer-to-Peer) | Lowest latency, minimal server cost |
| 3-8 Participants | SFU (Selective Forwarding Unit) | Scalable, moderate server load |
| 9+ Participants / Broadcast | MCU (Multipoint Control Unit) | Bandwidth efficiency for viewers |

### 2.2 WebRTC Stack for Python

**Recommended Library:** `aiortc` (asyncio-based WebRTC)

```python
# Core WebRTC Components
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
```

**Alternative:** `python-rtmixer` for low-level audio mixing with CFFI

### 2.3 WebRTC Configuration

```python
WEBRTC_CONFIG = {
    "ice_servers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
    ],
    "audio": {
        "codec": "opus",
        "sample_rate": 48000,
        "channels": 1,  # Mono for voice
        "bitrate": 32000,  # 32 kbps for voice
        "ptime": 20,  # 20ms packet time
    },
    "latency_target_ms": 150,  # End-to-end target
}
```

### 2.4 Signaling Server

```python
# WebSocket-based signaling for WebRTC
import asyncio
import websockets
import json

class WebRTCSignalingServer:
    def __init__(self):
        self.connections = {}
        
    async def handle_signaling(self, websocket, path):
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "offer":
                await self.handle_offer(websocket, data)
            elif data["type"] == "answer":
                await self.handle_answer(websocket, data)
            elif data["type"] == "ice-candidate":
                await self.handle_ice_candidate(websocket, data)
```

---

## 3. Media Server Architecture

### 3.1 Media Server Comparison

| Feature | Janus Gateway | Mediasoup | Kurento |
|---------|---------------|-----------|---------|
| **Architecture** | Plugin-based | Node.js/C++ library | Pipeline-based |
| **SFU Support** | Yes | Yes | Yes |
| **MCU Support** | Via plugins | No | Yes (built-in) |
| **Recording** | Yes | Yes | Yes |
| **SIP Integration** | Yes (SIP plugin) | No | Yes |
| **Learning Curve** | Moderate | Steep | Steep |
| **Documentation** | Good | Good | Moderate |
| **Scalability** | High | Very High | Moderate |
| **Best For** | Multi-protocol | High-performance | Media processing |

### 3.2 Recommended: Janus Gateway

**Rationale for OpenClaw:**
- Excellent SIP plugin for Twilio integration
- Plugin architecture allows custom AI agent integration
- REST API and WebSocket support
- Active community and good documentation

### 3.3 Janus Configuration

```json
{
  "general": {
    "configs_folder": "/opt/janus/etc/janus",
    "plugins_folder": "/opt/janus/lib/janus/plugins",
    "transports_folder": "/opt/janus/lib/janus/transports",
    "events_folder": "/opt/janus/lib/janus/events",
    "log_to_stdout": true,
    "log_to_file": "/var/log/janus/janus.log",
    "debug_level": 4
  },
  "media": {
    "rtp_port_range": "10000-20000",
    "dtls_mtu": 1200,
    "no_media_timer": 60
  },
  "nat": {
    "stun_server": "stun.l.google.com",
    "stun_port": 19302,
    "nice_debug": false,
    "full_trickle": true
  }
}
```

### 3.4 Janus AI Agent Plugin Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JANUS GATEWAY CORE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  SIP Plugin  │  │  VideoRoom   │  │  AI Agent    │       │
│  │              │  │   Plugin     │  │   Plugin     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                        │
│                    │  AI Agent    │                        │
│                    │  Core (Python│                        │
│                    │  via gRPC)   │                        │
│                    └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Audio Capture System

### 4.1 Windows 10 Low-Latency Audio APIs

**Primary API:** Windows Audio Session API (WASAPI) via `IAudioClient3`

**Key Features:**
- Buffer sizes as low as 3ms (with updated drivers)
- Automatic format negotiation
- Event-driven capture mode
- Exclusive and shared modes

### 4.2 Python Audio Capture Implementation

**Recommended Library:** `sounddevice` (PortAudio wrapper)

```python
import sounddevice as sd
import numpy as np
import asyncio
from queue import Queue

class WindowsAudioCapture:
    """
    Low-latency audio capture for Windows 10 using WASAPI
    """
    
    DEFAULT_CONFIG = {
        "sample_rate": 48000,
        "channels": 1,
        "dtype": np.int16,
        "blocksize": 960,  # 20ms at 48kHz
        "latency": "low",  # Request low latency
    }
    
    def __init__(self, config=None):
        self.config = config or self.DEFAULT_CONFIG
        self.stream = None
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.is_capturing = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio capture - runs in audio thread"""
        if status:
            print(f"Audio status: {status}")
        
        # Copy audio data to avoid reference issues
        audio_data = indata.copy()
        
        # Non-blocking queue put
        try:
            self.audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            # Drop oldest frame if queue is full
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except asyncio.QueueEmpty:
                pass
    
    async def start_capture(self, device=None):
        """Start audio capture stream"""
        self.is_capturing = True
        
        self.stream = sd.InputStream(
            device=device,
            samplerate=self.config["sample_rate"],
            channels=self.config["channels"],
            dtype=self.config["dtype"],
            blocksize=self.config["blocksize"],
            latency=self.config["latency"],
            callback=self._audio_callback
        )
        
        self.stream.start()
        return self
    
    async def read_frames(self, num_frames=None):
        """Async generator for audio frames"""
        while self.is_capturing:
            try:
                frame = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=1.0
                )
                yield frame
            except asyncio.TimeoutError:
                continue
    
    async def stop_capture(self):
        """Stop audio capture"""
        self.is_capturing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
```

### 4.3 Device Enumeration

```python
def enumerate_audio_devices():
    """Enumerate all audio capture devices on Windows"""
    devices = sd.query_devices()
    capture_devices = []
    
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            capture_devices.append({
                'index': idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
                'latency': device['default_low_input_latency']
            })
    
    return capture_devices
```

### 4.4 Low-Latency Optimization Settings

```python
LOW_LATENCY_CONFIG = {
    "sample_rate": 48000,
    "channels": 1,
    "blocksize": 480,      # 10ms - aggressive low latency
    "latency": "low",
    "dtype": np.float32,   # Float for processing efficiency
}

# For even lower latency with exclusive mode (requires WASAPI directly)
EXCLUSIVE_MODE_CONFIG = {
    "sample_rate": 48000,
    "channels": 1,
    "period_in_frames": 144,  # 3ms at 48kHz - minimum supported
    "exclusive": True,
}
```

---

## 5. Audio Playback System

### 5.1 Windows 10 Audio Playback

```python
class WindowsAudioPlayback:
    """
    Low-latency audio playback for Windows 10
    """
    
    DEFAULT_CONFIG = {
        "sample_rate": 48000,
        "channels": 1,
        "dtype": np.int16,
        "blocksize": 960,
        "latency": "low",
    }
    
    def __init__(self, config=None):
        self.config = config or self.DEFAULT_CONFIG
        self.stream = None
        self.playback_queue = asyncio.Queue(maxsize=50)
        self.is_playing = False
        self._buffer_underflows = 0
        
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio playback"""
        if status:
            if 'underflow' in str(status).lower():
                self._buffer_underflows += 1
        
        try:
            # Get audio data from queue
            data = self.playback_queue.get_nowait()
            
            # Ensure correct size
            if len(data) >= frames:
                outdata[:] = data[:frames]
            else:
                # Pad with silence if not enough data
                outdata[:len(data)] = data
                outdata[len(data):] = 0
                
        except asyncio.QueueEmpty:
            # Output silence if no data available
            outdata[:] = 0
    
    async def start_playback(self, device=None):
        """Start audio playback stream"""
        self.is_playing = True
        
        self.stream = sd.OutputStream(
            device=device,
            samplerate=self.config["sample_rate"],
            channels=self.config["channels"],
            dtype=self.config["dtype"],
            blocksize=self.config["blocksize"],
            latency=self.config["latency"],
            callback=self._audio_callback
        )
        
        self.stream.start()
        return self
    
    async def play_audio(self, audio_data):
        """Queue audio data for playback"""
        try:
            self.playback_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            # Drop oldest frame to make room
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.put_nowait(audio_data)
            except asyncio.QueueEmpty:
                pass
    
    async def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
```

### 5.2 Ring Buffer for Streaming Playback

```python
import collections
import threading

class AudioRingBuffer:
    """
    Thread-safe ring buffer for audio streaming
    """
    
    def __init__(self, capacity_samples, channels=1, dtype=np.float32):
        self.capacity = capacity_samples
        self.channels = channels
        self.dtype = dtype
        
        self._buffer = np.zeros((capacity_samples, channels), dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self._lock = threading.Lock()
    
    def write(self, data):
        """Write audio data to ring buffer"""
        with self._lock:
            samples_to_write = min(len(data), self.capacity - self._available)
            
            for i in range(samples_to_write):
                self._buffer[self._write_pos] = data[i]
                self._write_pos = (self._write_pos + 1) % self.capacity
            
            self._available += samples_to_write
            return samples_to_write
    
    def read(self, num_samples):
        """Read audio data from ring buffer"""
        with self._lock:
            samples_to_read = min(num_samples, self._available)
            
            result = np.zeros((samples_to_read, self.channels), dtype=self.dtype)
            for i in range(samples_to_read):
                result[i] = self._buffer[self._read_pos]
                self._read_pos = (self._read_pos + 1) % self.capacity
            
            self._available -= samples_to_read
            return result
    
    def available(self):
        """Get number of available samples"""
        with self._lock:
            return self._available
```

---

## 6. Streaming Protocols

### 6.1 Protocol Stack

| Layer | Protocol | Purpose |
|-------|----------|---------|
| Transport | UDP (RTP) | Real-time media transport |
| Security | DTLS-SRTP | Encrypted media |
| Signaling | WebSocket / HTTP | Session control |
| Codec | Opus | Audio encoding |
| Container | RTP | Packetization |

### 6.2 RTP Packet Structure for Opus

```
┌─────────────────────────────────────────────────────────────┐
│                        RTP Header (12 bytes)                 │
├─────────────────────────────────────────────────────────────┤
│  V=2  │ P │ X │  CC  │  M  │     PT (Opus=111)   │         │
│  2b   │1b │1b │  4b  │  1b │        7b           │         │
├─────────────────────────────────────────────────────────────┤
│                   Sequence Number (16 bits)                  │
├─────────────────────────────────────────────────────────────┤
│                      Timestamp (32 bits)                     │
├─────────────────────────────────────────────────────────────┤
│                   SSRC (Synchronization Source)              │
├─────────────────────────────────────────────────────────────┤
│                      CSRC List (optional)                    │
├─────────────────────────────────────────────────────────────┤
│                     Opus Payload (variable)                  │
│                   [TOC byte + Opus frames]                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Opus Configuration

```python
import opuslib

class OpusEncoder:
    """Opus encoder wrapper for voice optimization"""
    
    def __init__(self, sample_rate=48000, channels=1, application="voip"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.application = application
        
        # Initialize Opus encoder
        self.encoder = opuslib.Encoder(
            fs=sample_rate,
            channels=channels,
            application=opuslib.APPLICATION_VOIP
        )
        
        # Configure for low-latency voice
        self.encoder.bitrate = 32000  # 32 kbps for voice
        self.encoder.vbr = True       # Variable bitrate
        self.encoder.vbr_constraint = True
        self.encoder.complexity = 5   # Balance quality/CPU
        self.encoder.signal = opuslib.SIGNAL_VOICE
        self.encoder.inband_fec = True  # Forward error correction
        self.encoder.packet_loss_perc = 5  # Expect 5% packet loss
    
    def encode(self, pcm_data, frame_size=960):  # 20ms at 48kHz
        """Encode PCM audio to Opus"""
        return self.encoder.encode(pcm_data, frame_size)

class OpusDecoder:
    """Opus decoder wrapper"""
    
    def __init__(self, sample_rate=48000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        
        self.decoder = opuslib.Decoder(
            fs=sample_rate,
            channels=channels
        )
    
    def decode(self, opus_data, frame_size=960):
        """Decode Opus to PCM audio"""
        return self.decoder.decode(opus_data, frame_size)
```

### 6.4 Latency Budget

| Component | Target Latency | Notes |
|-----------|----------------|-------|
| Capture Buffer | 5-10 ms | WASAPI buffer size |
| Encoding | 2.5-5 ms | Opus algorithmic delay |
| Network | 30-50 ms | Internet RTT |
| Jitter Buffer | 20-40 ms | Adaptive buffering |
| Decoding | 2.5-5 ms | Opus decode time |
| Playback Buffer | 5-10 ms | WASAPI output buffer |
| **Total** | **65-120 ms** | End-to-end target |

---

## 7. Buffer Management & Latency Optimization

### 7.1 Adaptive Jitter Buffer

```python
import time
import statistics

class AdaptiveJitterBuffer:
    """
    Adaptive jitter buffer for network audio streaming
    Automatically adjusts size based on network conditions
    """
    
    def __init__(self, 
                 min_size_ms=20, 
                 max_size_ms=200,
                 target_size_ms=60,
                 sample_rate=48000):
        self.min_size_ms = min_size_ms
        self.max_size_ms = max_size_ms
        self.target_size_ms = target_size_ms
        self.sample_rate = sample_rate
        
        self._buffer = []
        self._current_size_ms = target_size_ms
        self._inter_arrival_times = []
        self._last_arrival_time = None
        self._lock = threading.Lock()
    
    def add_packet(self, packet, timestamp):
        """Add packet to jitter buffer"""
        with self._lock:
            # Calculate inter-arrival time
            if self._last_arrival_time is not None:
                iat = timestamp - self._last_arrival_time
                self._inter_arrival_times.append(iat)
                
                # Keep last 100 measurements
                if len(self._inter_arrival_times) > 100:
                    self._inter_arrival_times.pop(0)
                
                # Adapt buffer size every 10 packets
                if len(self._inter_arrival_times) % 10 == 0:
                    self._adapt_buffer_size()
            
            self._last_arrival_time = timestamp
            
            # Insert packet in order
            insert_pos = 0
            for i, (pkt, ts) in enumerate(self._buffer):
                if ts > timestamp:
                    insert_pos = i
                    break
                insert_pos = i + 1
            
            self._buffer.insert(insert_pos, (packet, timestamp))
    
    def _adapt_buffer_size(self):
        """Adapt buffer size based on network jitter"""
        if len(self._inter_arrival_times) < 10:
            return
        
        # Calculate jitter (standard deviation of inter-arrival times)
        jitter = statistics.stdev(self._inter_arrival_times)
        
        # Adjust buffer size
        if jitter > self._current_size_ms * 0.3:
            # High jitter - increase buffer
            self._current_size_ms = min(
                self._current_size_ms * 1.1,
                self.max_size_ms
            )
        elif jitter < self._current_size_ms * 0.1:
            # Low jitter - decrease buffer
            self._current_size_ms = max(
                self._current_size_ms * 0.95,
                self.min_size_ms
            )
    
    def get_packet(self):
        """Get next packet from buffer"""
        with self._lock:
            target_samples = int(
                self._current_size_ms * self.sample_rate / 1000
            )
            
            buffer_samples = sum(len(p[0]) for p in self._buffer)
            
            if buffer_samples >= target_samples and self._buffer:
                return self._buffer.pop(0)[0]
            
            return None
```

### 7.2 Latency Monitoring

```python
class LatencyMonitor:
    """Monitor and report audio latency metrics"""
    
    def __init__(self):
        self.capture_latency_ms = 0
        self.network_latency_ms = 0
        self.playback_latency_ms = 0
        self.total_latency_ms = 0
        self._metrics_history = []
    
    def update_capture_latency(self, latency_ms):
        self.capture_latency_ms = latency_ms
        self._update_total()
    
    def update_network_latency(self, latency_ms):
        self.network_latency_ms = latency_ms
        self._update_total()
    
    def update_playback_latency(self, latency_ms):
        self.playback_latency_ms = latency_ms
        self._update_total()
    
    def _update_total(self):
        self.total_latency_ms = (
            self.capture_latency_ms +
            self.network_latency_ms +
            self.playback_latency_ms
        )
        
        self._metrics_history.append({
            'timestamp': time.time(),
            'capture': self.capture_latency_ms,
            'network': self.network_latency_ms,
            'playback': self.playback_latency_ms,
            'total': self.total_latency_ms
        })
        
        # Keep last 1000 measurements
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)
    
    def get_statistics(self):
        """Get latency statistics"""
        if not self._metrics_history:
            return {}
        
        totals = [m['total'] for m in self._metrics_history]
        
        return {
            'current_ms': self.total_latency_ms,
            'avg_ms': statistics.mean(totals),
            'min_ms': min(totals),
            'max_ms': max(totals),
            'p95_ms': np.percentile(totals, 95),
            'p99_ms': np.percentile(totals, 99),
        }
```

---

## 8. Audio Mixing & Routing

### 8.1 Real-Time Audio Mixer

```python
import numpy as np
import asyncio

class RealtimeAudioMixer:
    """
    Real-time audio mixer for multiple input streams
    Supports: voice mixing, gain control, ducking
    """
    
    def __init__(self, sample_rate=48000, channels=1, block_size=960):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        
        self._input_streams = {}  # stream_id -> stream_info
        self._output_callbacks = []
        self._is_running = False
        self._mix_task = None
        
        # Gain controls
        self._master_gain = 1.0
        self._stream_gains = {}
        
        # Voice Activity Detection for ducking
        self._vad_enabled = True
        self._ducking_threshold = 0.1
        self._ducking_attenuation = 0.3
    
    def add_stream(self, stream_id, gain=1.0):
        """Add an input stream to the mixer"""
        self._input_streams[stream_id] = {
            'queue': asyncio.Queue(maxsize=20),
            'gain': gain,
            'active': True,
            'vad_level': 0.0
        }
        self._stream_gains[stream_id] = gain
    
    def remove_stream(self, stream_id):
        """Remove an input stream from the mixer"""
        if stream_id in self._input_streams:
            del self._input_streams[stream_id]
            del self._stream_gains[stream_id]
    
    def push_audio(self, stream_id, audio_data):
        """Push audio data to a stream"""
        if stream_id not in self._input_streams:
            return
        
        stream = self._input_streams[stream_id]
        
        try:
            stream['queue'].put_nowait(audio_data)
            
            # Update VAD level
            if self._vad_enabled:
                rms = np.sqrt(np.mean(audio_data ** 2))
                stream['vad_level'] = rms
                
        except asyncio.QueueFull:
            # Drop oldest frame
            try:
                stream['queue'].get_nowait()
                stream['queue'].put_nowait(audio_data)
            except asyncio.QueueEmpty:
                pass
    
    async def _mix_loop(self):
        """Main mixing loop"""
        while self._is_running:
            mixed = np.zeros((self.block_size, self.channels), dtype=np.float32)
            active_streams = 0
            
            # Find dominant voice for ducking
            dominant_stream = None
            max_vad = 0
            
            if self._vad_enabled:
                for stream_id, stream in self._input_streams.items():
                    if stream['vad_level'] > max_vad:
                        max_vad = stream['vad_level']
                        dominant_stream = stream_id
            
            # Mix all streams
            for stream_id, stream in self._input_streams.items():
                try:
                    audio_data = stream['queue'].get_nowait()
                    
                    # Ensure correct size
                    if len(audio_data) < self.block_size:
                        padded = np.zeros((self.block_size, self.channels))
                        padded[:len(audio_data)] = audio_data
                        audio_data = padded
                    elif len(audio_data) > self.block_size:
                        audio_data = audio_data[:self.block_size]
                    
                    # Apply gain
                    gain = self._stream_gains.get(stream_id, 1.0)
                    
                    # Apply ducking if not dominant voice
                    if self._vad_enabled and dominant_stream != stream_id:
                        if max_vad > self._ducking_threshold:
                            gain *= self._ducking_attenuation
                    
                    mixed += audio_data * gain
                    active_streams += 1
                    
                except asyncio.QueueEmpty:
                    continue
            
            # Normalize if multiple streams
            if active_streams > 1:
                mixed /= np.sqrt(active_streams)
            
            # Apply master gain and clip
            mixed *= self._master_gain
            mixed = np.clip(mixed, -1.0, 1.0)
            
            # Output to callbacks
            for callback in self._output_callbacks:
                try:
                    callback(mixed.copy())
                except Exception as e:
                    print(f"Mix output error: {e}")
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.001)
    
    def register_output(self, callback):
        """Register an output callback"""
        self._output_callbacks.append(callback)
    
    async def start(self):
        """Start the mixer"""
        self._is_running = True
        self._mix_task = asyncio.create_task(self._mix_loop())
    
    async def stop(self):
        """Stop the mixer"""
        self._is_running = False
        if self._mix_task:
            self._mix_task.cancel()
            try:
                await self._mix_task
            except asyncio.CancelledError:
                pass
```

### 8.2 Audio Router

```python
class AudioRouter:
    """
    Route audio between different endpoints
    Supports: P2P, SFU, TTS output, STT input
    """
    
    def __init__(self):
        self._routes = {}  # source_id -> [dest_ids]
        self._mixer = RealtimeAudioMixer()
        self._endpoints = {}
    
    def register_endpoint(self, endpoint_id, endpoint_type, handler):
        """Register an audio endpoint"""
        self._endpoints[endpoint_id] = {
            'type': endpoint_type,
            'handler': handler
        }
    
    def create_route(self, source_id, dest_id):
        """Create an audio route"""
        if source_id not in self._routes:
            self._routes[source_id] = []
        
        if dest_id not in self._routes[source_id]:
            self._routes[source_id].append(dest_id)
    
    def remove_route(self, source_id, dest_id):
        """Remove an audio route"""
        if source_id in self._routes:
            if dest_id in self._routes[source_id]:
                self._routes[source_id].remove(dest_id)
    
    async def route_audio(self, source_id, audio_data):
        """Route audio from source to destinations"""
        if source_id not in self._routes:
            return
        
        for dest_id in self._routes[source_id]:
            if dest_id in self._endpoints:
                endpoint = self._endpoints[dest_id]
                try:
                    await endpoint['handler'](audio_data)
                except Exception as e:
                    print(f"Routing error to {dest_id}: {e}")
```

---

## 9. Twilio Integration

### 9.1 Twilio Voice Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TWILIO VOICE INTEGRATION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │   PSTN/      │         │   Twilio     │                      │
│  │   Mobile     │◄───────►│   Cloud      │                      │
│  │   Phone      │         │   Platform   │                      │
│  └──────────────┘         └──────┬───────┘                      │
│                                  │                               │
│                                  ▼                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Twilio Voice SDK (Python)                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Incoming   │  │   Outgoing   │  │   Conference │   │   │
│  │  │   Calls      │  │   Calls      │  │   Bridge     │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AI Agent Audio Pipeline                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │  STT     │─►│  GPT-5.2 │─►│  TTS     │─►│  Output  │  │   │
│  │  │ (Whisper)│  │  Engine  │  │  (Eleven)│  │  Stream  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Twilio Voice SDK Integration

```python
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import websockets
import asyncio
import json

class TwilioVoiceIntegration:
    """
    Twilio Voice integration for AI agent phone calls
    """
    
    def __init__(self, account_sid, auth_token, phone_number):
        self.client = Client(account_sid, auth_token)
        self.phone_number = phone_number
        self.stream_url = None  # WebSocket URL for media streaming
        self._active_calls = {}
    
    def make_call(self, to_number, webhook_url):
        """Make an outbound call"""
        call = self.client.calls.create(
            to=to_number,
            from_=self.phone_number,
            url=webhook_url,
            record=True
        )
        return call.sid
    
    def generate_twiml(self, stream_url):
        """Generate TwiML for bidirectional streaming"""
        response = VoiceResponse()
        
        # Connect to media stream
        connect = Connect()
        connect.stream(url=stream_url)
        response.append(connect)
        
        return str(response)
    
    async def handle_media_stream(self, websocket, path):
        """Handle Twilio media stream WebSocket"""
        call_sid = None
        
        async for message in websocket:
            data = json.loads(message)
            
            event_type = data.get('event')
            
            if event_type == 'start':
                call_sid = data['start']['callSid']
                self._active_calls[call_sid] = {
                    'websocket': websocket,
                    'stream_sid': data['start']['streamSid']
                }
                print(f"Call started: {call_sid}")
                
            elif event_type == 'media':
                # Received audio from caller
                payload = data['media']['payload']
                audio_data = self._decode_mulaw(payload)
                
                # Process through AI pipeline
                await self._process_incoming_audio(call_sid, audio_data)
                
            elif event_type == 'stop':
                print(f"Call ended: {call_sid}")
                if call_sid in self._active_calls:
                    del self._active_calls[call_sid]
    
    def _decode_mulaw(self, payload):
        """Decode mu-law audio from Twilio"""
        import audioop
        mulaw_data = base64.b64decode(payload)
        pcm_data = audioop.ulaw2lin(mulaw_data, 2)
        return np.frombuffer(pcm_data, dtype=np.int16)
    
    def _encode_mulaw(self, pcm_data):
        """Encode PCM audio to mu-law for Twilio"""
        import audioop
        pcm_bytes = pcm_data.astype(np.int16).tobytes()
        mulaw_data = audioop.lin2ulaw(pcm_bytes, 2)
        return base64.b64encode(mulaw_data).decode('utf-8')
    
    async def _process_incoming_audio(self, call_sid, audio_data):
        """Process incoming audio through AI pipeline"""
        # This integrates with STT -> GPT-5.2 -> TTS
        pass
    
    async def send_audio(self, call_sid, audio_data):
        """Send audio to the call"""
        if call_sid not in self._active_calls:
            return
        
        call_info = self._active_calls[call_sid]
        websocket = call_info['websocket']
        stream_sid = call_info['stream_sid']
        
        encoded = self._encode_mulaw(audio_data)
        
        message = {
            'event': 'media',
            'streamSid': stream_sid,
            'media': {
                'payload': encoded
            }
        }
        
        await websocket.send(json.dumps(message))
```

### 9.3 Twilio Configuration

```python
TWILIO_CONFIG = {
    "account_sid": "your_account_sid",
    "auth_token": "your_auth_token",
    "phone_number": "+1234567890",
    "webhook_url": "https://your-domain.com/voice-webhook",
    "stream_url": "wss://your-domain.com/media-stream",
    "recording": True,
    "recording_channels": "dual",  # Separate caller/callee
}
```

---

## 10. Implementation Roadmap

### Phase 1: Core Audio Infrastructure (Weeks 1-2)
- [ ] Implement Windows WASAPI capture/playback
- [ ] Set up ring buffer system
- [ ] Implement Opus codec integration
- [ ] Create latency monitoring system

### Phase 2: WebRTC Foundation (Weeks 3-4)
- [ ] Deploy Janus Gateway media server
- [ ] Implement WebRTC signaling server
- [ ] Create P2P connection handling
- [ ] Set up STUN/TURN servers

### Phase 3: Twilio Integration (Weeks 5-6)
- [ ] Integrate Twilio Voice SDK
- [ ] Implement bidirectional streaming
- [ ] Set up call handling webhooks
- [ ] Test PSTN connectivity

### Phase 4: AI Pipeline Integration (Weeks 7-8)
- [ ] Integrate STT (Whisper)
- [ ] Connect GPT-5.2 engine
- [ ] Integrate TTS (ElevenLabs)
- [ ] Implement voice activity detection

### Phase 5: Advanced Features (Weeks 9-10)
- [ ] Implement audio mixing for conferences
- [ ] Add adaptive jitter buffer
- [ ] Create audio routing system
- [ ] Implement 24/7 monitoring and heartbeat

### Phase 6: Optimization & Testing (Weeks 11-12)
- [ ] Performance optimization
- [ ] Latency tuning
- [ ] Load testing
- [ ] Production deployment

---

## Appendix A: Dependencies

```
# requirements.txt

# Audio processing
sounddevice>=0.4.6
numpy>=1.24.0
opuslib>=3.0.1
PyAudio>=0.2.13

# WebRTC
aiortc>=1.6.0
aioice>=0.9.0

# Twilio
twilio>=8.0.0

# WebSocket
websockets>=11.0.0
python-socketio>=5.8.0

# Async
asyncio>=3.4.3
aiohttp>=3.8.0

# Utilities
python-dotenv>=1.0.0
structlog>=23.0.0
```

## Appendix B: Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB SSD | 100 GB NVMe |
| Network | 10 Mbps | 100 Mbps |
| Audio | Onboard | Dedicated USB audio |

## Appendix C: Latency Targets

| Scenario | Target Latency | Maximum Acceptable |
|----------|----------------|-------------------|
| P2P Voice | < 100 ms | 150 ms |
| SFU Conference | < 150 ms | 200 ms |
| Twilio PSTN | < 200 ms | 300 ms |
| AI Response | < 500 ms | 1000 ms |

---

**Document End**
