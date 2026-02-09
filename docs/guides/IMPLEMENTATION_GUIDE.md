# OpenClaw AI Agent - Audio System Implementation Guide

## Quick Start Guide for Developers

This guide provides step-by-step instructions for implementing the real-time audio processing and streaming architecture for the OpenClaw Windows 10 AI agent system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Core Components](#4-core-components)
5. [Integration Steps](#5-integration-steps)
6. [Configuration](#6-configuration)
7. [Testing & Validation](#7-testing--validation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB SSD | 100 GB NVMe |
| Network | 10 Mbps | 100 Mbps |
| Audio | Onboard | Dedicated USB audio interface |

### Software Requirements

- Windows 10 (Build 1903 or later)
- Python 3.9+
- Visual C++ Build Tools
- Git

### Windows Audio Setup

1. **Update Windows** to latest version for best WASAPI support
2. **Install audio drivers** with low-latency support
3. **Configure power settings** for high performance
4. **Disable audio enhancements** in Windows sound settings

---

## 2. Installation

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv openclaw-audio-env

# Activate
openclaw-audio-env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Core audio dependencies
pip install sounddevice numpy opuslib PyAudio

# WebRTC dependencies
pip install aiortc aioice

# Twilio integration
pip install twilio

# WebSocket and async
pip install websockets aiohttp python-socketio

# Utilities
pip install python-dotenv structlog
```

### Step 3: Install PortAudio (Windows)

```bash
# Download PortAudio DLL
# Place portaudio_x64.dll in system PATH or project directory
```

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPENCLAW AI AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        AUDIO CORE MODULE                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │  Capture │  │  Mixer   │  │ Playback │  │ Latency Monitor  │   │   │
│  │  │  (WASAPI)│  │ (Multi)  │  │ (WASAPI) │  │                  │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────────┘   │   │
│  │       │             │             │                                │   │
│  │       └─────────────┴─────────────┘                                │   │
│  │                     │                                              │   │
│  │                     ▼                                              │   │
│  │            ┌─────────────────┐                                     │   │
│  │            │  Ring Buffer    │                                     │   │
│  │            │  (Thread-safe)  │                                     │   │
│  │            └─────────────────┘                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      WEBRTC INTEGRATION                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  P2P Conn    │  │  SFU Client  │  │  Signaling   │              │   │
│  │  │  (aiortc)    │  │  (Janus)     │  │  (WebSocket) │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     TWILIO INTEGRATION                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Voice SDK   │  │  Media Stream│  │  Conference  │              │   │
│  │  │  (REST API)  │  │  (WebSocket) │  │  Bridge      │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 Audio Core (`audio_core_implementation.py`)

The audio core provides:
- **AudioCapture**: Low-latency microphone input via WASAPI
- **AudioPlayback**: Low-latency speaker output via WASAPI
- **AudioMixer**: Real-time mixing of multiple audio streams
- **RingBuffer**: Thread-safe circular buffer for streaming
- **LatencyMonitor**: Track and report latency metrics

### 4.2 WebRTC Integration (`webrtc_integration.py`)

Provides:
- **WebRTCConnection**: Single peer connection management
- **WebRTCSignalingServer**: WebSocket signaling for SDP exchange
- **WebRTCManager**: Multi-connection management
- **AudioTrackProcessor**: Handle incoming audio tracks

### 4.3 Twilio Integration (`twilio_voice_integration.py`)

Provides:
- **TwilioVoiceManager**: Manage voice calls via Twilio API
- **TwilioMediaStream**: Bidirectional audio streaming
- **TwilioConferenceBridge**: Multi-party conference calls

---

## 5. Integration Steps

### Step 1: Initialize Audio Core

```python
from audio_core_implementation import AudioCore, AudioConfig

# Create audio core
audio_core = AudioCore()

# Configure audio
config = AudioConfig(
    sample_rate=48000,
    channels=1,
    block_size=960,  # 20ms at 48kHz
    latency="low"
)

# Initialize
audio_core.initialize(
    capture_config=config,
    playback_config=config
)

# Start audio
audio_core.start()
```

### Step 2: Set Up Audio Routing

```python
# Add stream to mixer
audio_core.mixer.add_stream("mic_input", gain=1.0)

# Route capture to mixer
def on_capture(frame):
    audio_core.mixer.push_audio("mic_input", frame.data)

audio_core.capture.register_callback(on_capture)

# Mixer output goes to playback automatically
```

### Step 3: Start WebRTC Signaling

```python
from webrtc_integration import WebRTCManager, WebRTCConfig

# Create WebRTC manager
webrtc_manager = WebRTCManager()

# Handle incoming audio
def on_webrtc_audio(connection_id, frame):
    # Route to mixer
    audio_core.mixer.push_audio(f"webrtc_{connection_id}", frame)

webrtc_manager.on_audio_frame = on_webrtc_audio

# Start signaling server
await webrtc_manager.start_signaling(host="0.0.0.0", port=8765)
```

### Step 4: Integrate Twilio Voice

```python
from twilio_voice_integration import TwilioVoiceManager, TwilioConfig

# Configure Twilio
config = TwilioConfig(
    account_sid="your_account_sid",
    auth_token="your_auth_token",
    phone_number="+1234567890",
    webhook_url="https://your-domain.com/voice-webhook",
    stream_url="wss://your-domain.com/media-stream"
)

# Create voice manager
voice_manager = TwilioVoiceManager(config)

# Handle call audio
def on_call_audio(call_sid, audio_data):
    # Route to mixer
    audio_core.mixer.push_audio(f"call_{call_sid}", audio_data)

voice_manager.on_call_audio = on_call_audio

# Start media stream server
await voice_manager.start_media_stream_server(port=8766)
```

### Step 5: AI Pipeline Integration

```python
# Connect mixer output to AI pipeline
def on_mixed_audio(audio_data):
    # Send to STT
    transcription = stt_engine.transcribe(audio_data)
    
    # Send to GPT-5.2
    response = gpt_engine.generate(transcription)
    
    # Send to TTS
    audio_response = tts_engine.synthesize(response)
    
    # Play response
    audio_core.playback.write(audio_response)

audio_core.mixer.register_output(on_mixed_audio)
```

---

## 6. Configuration

### Environment Variables

Create `.env` file:

```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# WebSocket URLs
SIGNALING_URL=wss://your-domain.com/signaling
MEDIA_STREAM_URL=wss://your-domain.com/media-stream
WEBHOOK_URL=https://your-domain.com/voice-webhook

# Audio Configuration
AUDIO_SAMPLE_RATE=48000
AUDIO_CHANNELS=1
AUDIO_BLOCK_SIZE=960
AUDIO_LATENCY=low

# WebRTC Configuration
ICE_SERVER_1=stun:stun.l.google.com:19302
ICE_SERVER_2=stun:stun1.l.google.com:19302
```

### Audio Configuration Presets

```python
# Low latency voice preset
VOICE_CONFIG = AudioConfig(
    sample_rate=48000,
    channels=1,
    block_size=480,  # 10ms
    latency="low"
)

# Standard quality preset
STANDARD_CONFIG = AudioConfig(
    sample_rate=48000,
    channels=1,
    block_size=960,  # 20ms
    latency="low"
)

# High quality music preset
MUSIC_CONFIG = AudioConfig(
    sample_rate=48000,
    channels=2,
    block_size=960,
    latency="low"
)
```

---

## 7. Testing & Validation

### Unit Tests

```python
# Test audio capture
async def test_capture():
    capture = AudioCapture()
    capture.start()
    
    # Read some frames
    for _ in range(100):
        frame = capture.read(960)
        assert frame is not None
        assert len(frame) == 960
    
    capture.stop()

# Test mixer
async def test_mixer():
    mixer = AudioMixer()
    mixer.add_stream("test", gain=1.0)
    
    # Push audio
    test_data = np.random.randn(960, 1).astype(np.float32)
    mixer.push_audio("test", test_data)
    
    mixer.start()
    await asyncio.sleep(0.1)
    mixer.stop()
```

### Integration Tests

```python
# Test WebRTC connection
async def test_webrtc():
    manager = WebRTCManager()
    await manager.start_signaling(port=8765)
    
    conn = await manager.create_connection()
    offer = await conn.create_offer()
    
    assert offer["type"] == "offer"
    assert "sdp" in offer
    
    await manager.close_all()
```

### Latency Tests

```python
# Measure end-to-end latency
async def test_latency():
    core = AudioCore()
    core.initialize()
    core.start()
    
    # Run for 10 seconds
    await asyncio.sleep(10)
    
    stats = core.get_latency_stats()
    print(f"Average latency: {stats['avg_ms']:.1f}ms")
    print(f"P95 latency: {stats['p95_ms']:.1f}ms")
    
    assert stats['avg_ms'] < 100  # Target < 100ms
    
    core.stop()
```

---

## 8. Troubleshooting

### Common Issues

#### Issue: High Latency

**Symptoms:** Audio delay > 200ms

**Solutions:**
1. Reduce block_size in AudioConfig
2. Use "low" latency mode
3. Disable Windows audio enhancements
4. Close other audio applications
5. Update audio drivers

```python
# Optimize for low latency
config = AudioConfig(
    sample_rate=48000,
    block_size=480,  # Reduce from 960
    latency="low"
)
```

#### Issue: Audio Glitches/Dropouts

**Symptoms:** Clicks, pops, or missing audio

**Solutions:**
1. Increase block_size
2. Check CPU usage
3. Use exclusive mode (WASAPI)
4. Increase ring buffer size
5. Check for buffer underruns

```python
# Check for underruns
print(f"Playback underruns: {playback._underflows}")
```

#### Issue: WebRTC Connection Fails

**Symptoms:** Cannot establish peer connection

**Solutions:**
1. Check STUN/TURN servers
2. Verify firewall settings
3. Check ICE candidate exchange
4. Use TURN server for NAT traversal

```python
# Add TURN server
config = WebRTCConfig(
    ice_servers=[
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": "turn:your-turn-server.com:3478",
            "username": "user",
            "credential": "pass"
        }
    ]
)
```

#### Issue: Twilio Audio Quality Poor

**Symptoms:** Choppy or distorted audio on calls

**Solutions:**
1. Check network bandwidth
2. Verify 8kHz sample rate for Twilio
3. Check mu-law encoding/decoding
4. Monitor jitter buffer

```python
# Resample to 8kHz for Twilio
def resample_to_8k(audio_48k):
    return audio_48k[::6]  # Simple downsampling
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log audio metrics
def log_audio_metrics(core):
    stats = core.get_latency_stats()
    logger.debug(f"Latency: {stats}")
    
    logger.debug(f"Capture available: {core.capture.read_available()}")
    logger.debug(f"Playback space: {core.playback.get_buffer_space()}")
```

### Performance Monitoring

```python
import psutil
import time

def monitor_performance():
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        print(f"CPU: {cpu}%, Memory: {memory}%")
        
        if cpu > 80:
            logger.warning("High CPU usage detected")
        
        time.sleep(5)
```

---

## API Reference

### AudioCore

```python
class AudioCore:
    def initialize(self, capture_config, playback_config)
    def start(self)
    def stop(self)
    def get_latency_stats(self) -> dict
```

### AudioCapture

```python
class AudioCapture:
    def start(self) -> AudioCapture
    def stop()
    def read(self, num_samples) -> np.ndarray
    def register_callback(self, callback)
```

### AudioPlayback

```python
class AudioPlayback:
    def start(self) -> AudioPlayback
    def stop()
    def write(self, data) -> int
```

### WebRTCManager

```python
class WebRTCManager:
    async def start_signaling(self, host, port)
    async def create_connection(self, config) -> WebRTCConnection
    async def close_all()
```

### TwilioVoiceManager

```python
class TwilioVoiceManager:
    async def make_call(self, to_number) -> str
    async def hangup_call(self, call_sid)
    async def start_media_stream_server(self, host, port)
    async def send_audio_to_call(self, call_sid, audio_data)
```

---

## Next Steps

1. **Deploy Janus Gateway** for SFU capabilities
2. **Set up TURN server** for NAT traversal
3. **Configure SSL certificates** for WebSocket servers
4. **Implement monitoring** and alerting
5. **Load test** with multiple concurrent calls
6. **Optimize** based on production metrics

---

**Document End**
