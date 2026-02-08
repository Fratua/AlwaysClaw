# AUDIO STREAMING INFRASTRUCTURE & MEDIA DELIVERY SPECIFICATION
## Windows 10 OpenClaw-Inspired AI Agent System

---

## EXECUTIVE SUMMARY

This document provides comprehensive technical specifications for the audio streaming infrastructure of a Windows 10-based AI agent system. The architecture supports real-time voice communication, media streaming, and integration with Twilio voice/SMS services, TTS/STT engines, and GPT-5.2 processing.

**Key Metrics:**
- Target Latency: <150ms end-to-end
- Supported Codecs: Opus, G.711 (mu-law/A-law), G.722
- Buffer Size: 20-60ms adaptive
- Packet Loss Tolerance: Up to 5% with PLC
- Concurrent Streams: 100+ per media server instance

---

## TABLE OF CONTENTS

1. WebSocket Streaming Architecture
2. Media Server Integration
3. RTP/RTCP Protocol Implementation
4. Audio Codecs Specification
5. Buffer Management & Jitter Handling
6. Packet Loss Concealment
7. Stream Synchronization
8. Quality of Service (QoS)
9. System Architecture Diagrams
10. Implementation Reference

---

## 1. WEBSOCKET STREAMING ARCHITECTURE

### 1.1 Overview

The WebSocket layer provides bidirectional, full-duplex communication channels for real-time audio streaming between the AI agent system components.

### 1.2 WebSocket Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│         (AI Agent Logic, TTS/STT, Media Control)            │
├─────────────────────────────────────────────────────────────┤
│                    WEBSOCKET LAYER (WSS)                    │
│         (RFC 6455 with Binary Audio Frame Support)          │
├─────────────────────────────────────────────────────────────┤
│                    TRANSPORT LAYER                          │
│              (TLS 1.3 / TCP with Nagle Disabled)            │
├─────────────────────────────────────────────────────────────┤
│                    NETWORK LAYER                            │
│                     (IPv4/IPv6)                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 WebSocket Message Types

| Message Type | Opcode | Direction | Payload Size | Purpose |
|--------------|--------|-----------|--------------|---------|
| AUDIO_BINARY | 0x02 | Bidirectional | Variable (20-200 bytes) | Encoded audio frames |
| CONTROL_JSON | 0x01 | Bidirectional | <4KB | Stream control commands |
| PING | 0x09 | Client->Server | 0-125 bytes | Keep-alive |
| PONG | 0x0A | Server->Client | Echo | Keep-alive response |
| CLOSE | 0x08 | Bidirectional | 0-125 bytes | Connection termination |

### 1.4 WebSocket Frame Structure for Audio

```python
class AudioWebSocketFrame:
    """
    Binary frame structure for audio streaming
    """
    FRAME_HEADER = {
        'fin': 1,           # 1 bit - Final fragment
        'rsv': 0,           # 3 bits - Reserved
        'opcode': 0x02,     # 4 bits - Binary frame
        'mask': 0,          # 1 bit - Server-to-client unmasked
        'payload_len': None # 7/16/64 bits - Variable
    }

    AUDIO_PAYLOAD = {
        'timestamp': 32,    # RTP timestamp (ms)
        'sequence': 16,     # Sequence number
        'codec_type': 8,    # Codec identifier
        'flags': 8,         # Flags (VAD, DTX, etc.)
        'audio_data': None  # Variable - Encoded audio
    }
```

### 1.5 WebSocket Server Configuration

```python
WEBSOCKET_CONFIG = {
    'host': '0.0.0.0',
    'port': 8443,
    'ssl_cert': 'certs/server.crt',
    'ssl_key': 'certs/server.key',
    'protocols': ['audio-stream-v1', 'control-v1'],

    # Connection Management
    'max_connections': 1000,
    'ping_interval': 15,        # seconds
    'ping_timeout': 10,         # seconds
    'close_timeout': 5,         # seconds

    # Performance Tuning
    'compression': None,        # Disable for low latency
    'max_size': 1048576,        # 1MB max message
    'max_queue': 32,            # Message queue size
    'read_limit': 65536,        # Read buffer limit
    'write_limit': 65536,       # Write buffer limit

    # Windows-Specific
    'tcp_keepalive': True,
    'tcp_nodelay': True,        # Disable Nagle's algorithm
    'socket_rcvbuf': 262144,    # 256KB receive buffer
    'socket_sndbuf': 262144,    # 256KB send buffer
}
```

### 1.6 WebSocket Client Implementation

```python
import asyncio
import websockets
import json
import struct

class AudioWebSocketClient:
    CODEC_OPUS = 0x01
    CODEC_G711_ULAW = 0x02
    CODEC_G711_ALAW = 0x03
    CODEC_G722 = 0x04

    def __init__(self, uri: str, codec: int = CODEC_OPUS):
        self.uri = uri
        self.codec = codec
        self.websocket = None
        self.sequence = 0
        self._running = False

    async def connect(self):
        self.websocket = await websockets.connect(
            self.uri,
            subprotocols=['audio-stream-v1'],
            ping_interval=15,
            ping_timeout=10,
            compression=None,
        )
        self._running = True

    async def send_audio(self, audio_data: bytes, vad_active: bool = False):
        self.sequence = (self.sequence + 1) % 65536
        timestamp = int(asyncio.get_event_loop().time() * 1000) % (2**32)

        flags = 0x01 if vad_active else 0x00
        header = struct.pack('!IHBB', timestamp, self.sequence, self.codec, flags)
        frame = header + audio_data

        await self.websocket.send(frame)
```

---

## 2. MEDIA SERVER INTEGRATION

### 2.1 Media Server Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEDIA SERVER CLUSTER                             │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   WebSocket  │  │   RTP/RTCP   │  │   Recording  │  │   Streaming  │ │
│  │   Gateway    │  │   Handler    │  │   Service    │  │   Service    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │         │
│         └─────────────────┴─────────────────┴─────────────────┘         │
│                                   │                                      │
│                         ┌─────────┴─────────┐                           │
│                         │   Media Router    │                           │
│                         │  (SFU/MCU Mixer)  │                           │
│                         └─────────┬─────────┘                           │
│                                   │                                      │
│  ┌────────────────────────────────┼────────────────────────────────┐   │
│  │                    MEDIA PIPELINE                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │  │  Decoder │->│  Mixer   │->│  Encoder │->│  Buffer  │       │   │
│  │  │ (Opus/   │  │ (Audio   │  │ (Output  │  │ (Jitter  │       │   │
│  │  │  G.711)  │  │  Mixing) │  │  Codec)  │  │  Buffer) │       │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Media Server Configuration

```python
MEDIA_SERVER_CONFIG = {
    'server_id': 'media-01',
    'region': 'us-east-1',

    'websocket': {
        'bind_address': '0.0.0.0',
        'port': 8443,
        'ssl_enabled': True,
        'max_clients': 1000,
        'heartbeat_interval': 15,
    },

    'rtp': {
        'port_range': (10000, 20000),
        'bind_address': '0.0.0.0',
        'dtls_enabled': True,
        'srtp_enabled': True,
    },

    'pipeline': {
        'mixer_enabled': True,
        'max_mix_streams': 32,
        'sample_rate': 48000,
        'channels': 2,
        'frame_duration': 20,  # ms
    },

    'recording': {
        'enabled': True,
        'format': 'mp3',
        'bitrate': 128000,
        'storage_path': '/var/recordings',
        'retention_days': 30,
    },
}
```

### 2.3 Live Streaming Pipeline

```python
class MediaPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['sample_rate']
        self.frame_duration = config['frame_duration']
        self.frame_samples = int(self.sample_rate * self.frame_duration / 1000)
        self.input_streams = {}
        self.output_streams = {}
        self.mixer = AudioMixer(
            sample_rate=self.sample_rate,
            max_streams=config['max_mix_streams']
        )

    async def _process_loop(self):
        interval = self.frame_duration / 1000
        while self._running:
            start_time = asyncio.get_event_loop().time()

            input_frames = []
            for stream_id, stream in self.input_streams.items():
                frame = stream.get_frame()
                if frame is not None:
                    input_frames.append((stream_id, frame))

            if len(input_frames) > 1:
                mixed = self.mixer.mix([f for _, f in input_frames])
            elif len(input_frames) == 1:
                mixed = input_frames[0][1]
            else:
                mixed = np.zeros(self.frame_samples, dtype=np.float32)

            for output in self.output_streams.values():
                await output.send(mixed)

            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
```

### 2.4 Recording Service

```python
class RecordingService:
    def __init__(self, config: dict):
        self.config = config
        self.storage_path = Path(config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_recordings = {}

    async def start_recording(self, stream_id: str, metadata: dict) -> str:
        recording_id = f"{stream_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        session = RecordingSession(
            recording_id=recording_id,
            stream_id=stream_id,
            storage_path=self.storage_path,
            config=self.config,
            metadata=metadata
        )
        await session.start()
        self.active_recordings[recording_id] = session
        return recording_id
```

---

## 3. RTP/RTCP PROTOCOL IMPLEMENTATION

### 3.1 RTP Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│              (Audio Codec: Opus, G.711, etc.)               │
├─────────────────────────────────────────────────────────────┤
│                    RTP LAYER (RFC 3550)                     │
│         (Real-time Transport Protocol - Media)              │
├─────────────────────────────────────────────────────────────┤
│                    RTCP LAYER (RFC 3550)                    │
│    (Real-time Control Protocol - Statistics/Feedback)       │
├─────────────────────────────────────────────────────────────┤
│                    TRANSPORT LAYER                          │
│              (UDP with Optional DTLS/SRTP)                  │
├─────────────────────────────────────────────────────────────┤
│                    NETWORK LAYER                            │
│                     (IPv4/IPv6)                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 RTP Packet Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|X|  CC   |M|     PT      |       Sequence Number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           Timestamp                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Synchronization Source (SSRC) identifier            |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|            Contributing Source (CSRC) identifiers             |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           Payload                             |
|                              ....                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 3.3 RTP Implementation

```python
import struct
import random
import socket

class RTPPacket:
    version = 2
    PT_OPUS = 111
    PT_G711_ULAW = 0
    PT_G711_ALAW = 8
    PT_G722 = 9

    def __init__(self):
        self.padding = False
        self.extension = False
        self.csrc_count = 0
        self.marker = False
        self.payload_type = 0
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = 0
        self.payload = b''

    def pack(self) -> bytes:
        first_byte = (
            (self.version << 6) |
            (int(self.padding) << 5) |
            (int(self.extension) << 4) |
            (self.csrc_count & 0x0F)
        )
        second_byte = (
            (int(self.marker) << 7) |
            (self.payload_type & 0x7F)
        )

        header = struct.pack('!BBHII',
            first_byte, second_byte,
            self.sequence_number, self.timestamp, self.ssrc
        )
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> 'RTPPacket':
        packet = cls()
        first_byte = data[0]
        second_byte = data[1]
        packet.version = (first_byte >> 6) & 0x03
        packet.padding = bool((first_byte >> 5) & 0x01)
        packet.extension = bool((first_byte >> 4) & 0x01)
        packet.csrc_count = first_byte & 0x0F
        packet.marker = bool((second_byte >> 7) & 0x01)
        packet.payload_type = second_byte & 0x7F
        packet.sequence_number = struct.unpack('!H', data[2:4])[0]
        packet.timestamp = struct.unpack('!I', data[4:8])[0]
        packet.ssrc = struct.unpack('!I', data[8:12])[0]
        packet.payload = data[12:]
        return packet

class RTPSession:
    def __init__(self, local_port: int, remote_addr: tuple, payload_type: int):
        self.local_port = local_port
        self.remote_addr = remote_addr
        self.payload_type = payload_type
        self.ssrc = random.randint(0, 2**32 - 1)
        self.sequence_number = random.randint(0, 2**16 - 1)
        self.initial_timestamp = random.randint(0, 2**32 - 1)
        self.sample_rate = self._get_sample_rate(payload_type)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', local_port))

    def _get_sample_rate(self, payload_type: int) -> int:
        rates = {
            RTPPacket.PT_OPUS: 48000,
            RTPPacket.PT_G711_ULAW: 8000,
            RTPPacket.PT_G711_ALAW: 8000,
            RTPPacket.PT_G722: 16000,
        }
        return rates.get(payload_type, 8000)

    def send_packet(self, audio_data: bytes, marker: bool = False) -> int:
        timestamp = self.initial_timestamp + int(
            self.sequence_number * (self.sample_rate * 20 / 1000)
        )
        packet = RTPPacket()
        packet.payload_type = self.payload_type
        packet.marker = marker
        packet.sequence_number = self.sequence_number
        packet.timestamp = timestamp
        packet.ssrc = self.ssrc
        packet.payload = audio_data

        data = packet.pack()
        self.socket.sendto(data, self.remote_addr)
        self.sequence_number = (self.sequence_number + 1) % 65536
        return len(data)
```

### 3.4 RTCP Implementation

```python
class RTCPPacket:
    PT_SR = 200      # Sender Report
    PT_RR = 201      # Receiver Report
    PT_SDES = 202    # Source Description
    PT_BYE = 203     # Goodbye

    def __init__(self):
        self.version = 2
        self.padding = False
        self.packet_type = 0

class RTCPSenderReport(RTCPPacket):
    def __init__(self):
        super().__init__()
        self.packet_type = self.PT_SR
        self.ssrc = 0
        self.ntp_timestamp = 0
        self.rtp_timestamp = 0
        self.packet_count = 0
        self.octet_count = 0

    def pack(self) -> bytes:
        header = struct.pack('!BBH',
            (self.version << 6) | (int(self.padding) << 5),
            self.packet_type, 6
        )
        body = struct.pack('!IQIII',
            self.ssrc, self.ntp_timestamp, self.rtp_timestamp,
            self.packet_count, self.octet_count
        )
        return header + body
```

---

## 4. AUDIO CODECS SPECIFICATION

### 4.1 Codec Selection Matrix

| Codec | Bitrate (kbps) | Sample Rate (kHz) | Latency (ms) | Quality | CPU Usage | Use Case |
|-------|----------------|-------------------|--------------|---------|-----------|----------|
| Opus | 6-510 | 8-48 | 5-66 | Excellent | Medium | Primary |
| G.711 mu-law | 64 | 8 | 0.125 | Good | Low | Fallback |
| G.711 A-law | 64 | 8 | 0.125 | Good | Low | Fallback |
| G.722 | 48/56/64 | 16 | 0.125 | Very Good | Low | HD Voice |
| G.729 | 8 | 8 | 15 | Fair | Low | Low bandwidth |

### 4.2 Opus Codec Implementation

```python
import opuslib
import numpy as np

class OpusEncoder:
    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._encoder = opuslib.Encoder(
            fs=sample_rate, channels=channels, application=opuslib.APPLICATION_VOIP
        )
        self.frame_duration = 20  # ms
        self.frame_samples = int(sample_rate * self.frame_duration / 1000)
        self._configure_encoder()

    def _configure_encoder(self):
        self._encoder.bitrate = 24000
        self._encoder.vbr = True
        self._encoder.complexity = 10
        self._encoder.signal = opuslib.SIGNAL_VOICE
        self._encoder.inband_fec = True
        self._encoder.packet_loss_perc = 5
        self._encoder.dtx = True

    def encode(self, pcm_data: np.ndarray) -> bytes:
        if pcm_data.dtype != np.int16:
            pcm_data = (pcm_data * 32767).astype(np.int16)
        expected_samples = self.frame_samples * self.channels
        if len(pcm_data) != expected_samples:
            if len(pcm_data) < expected_samples:
                pcm_data = np.pad(pcm_data, (0, expected_samples - len(pcm_data)))
            else:
                pcm_data = pcm_data[:expected_samples]
        return self._encoder.encode(pcm_data.tobytes(), self.frame_samples)

class OpusDecoder:
    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = 20  # ms
        self.frame_samples = int(sample_rate * self.frame_duration / 1000)
        self._decoder = opuslib.Decoder(fs=sample_rate, channels=channels)

    def decode(self, opus_data: bytes, fec: bool = False) -> np.ndarray:
        pcm_bytes = self._decoder.decode(opus_data, self.frame_samples, fec=fec)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def decode_lost(self) -> np.ndarray:
        return self.decode(b'', fec=True)
```

### 4.3 G.711 Codec Implementation

```python
class G711Codec:
    MULAW_BIAS = 33
    MULAW_MAX = 0x1FFF
    ALAW_MAX = 0xFFF

    @staticmethod
    def pcm_to_mulaw(pcm: int) -> int:
        pcm = pcm + G711Codec.MULAW_BIAS
        sign = (pcm >> 8) & 0x80
        if sign:
            pcm = -pcm
        if pcm > G711Codec.MULAW_MAX:
            pcm = G711Codec.MULAW_MAX
        segment = 0
        temp = pcm >> 7
        while temp and segment < 8:
            temp >>= 1
            segment += 1
        if segment == 0:
            uval = pcm >> 4
        else:
            uval = ((segment << 4) | ((pcm >> (segment + 3)) & 0x0F))
        return ~(sign | uval) & 0xFF

    @staticmethod
    def mulaw_to_pcm(ulaw: int) -> int:
        ulaw = ~ulaw & 0xFF
        sign = (ulaw & 0x80) >> 7
        exponent = (ulaw & 0x70) >> 4
        mantissa = ulaw & 0x0F
        if exponent == 0:
            pcm = (mantissa << 4) + 8
        else:
            pcm = ((mantissa | 0x10) << (exponent + 3)) + (1 << (exponent + 3))
        if sign:
            pcm = -pcm
        return pcm - G711Codec.MULAW_BIAS

class G711Encoder:
    def __init__(self, law: str = 'ulaw'):
        self.law = law.lower()
        self.sample_rate = 8000
        self.frame_duration = 20  # ms
        self.frame_samples = 160

    def encode(self, pcm_data: np.ndarray) -> bytes:
        if pcm_data.dtype == np.float32 or pcm_data.dtype == np.float64:
            pcm_data = (pcm_data * 32767).astype(np.int16)
        encoded = bytearray()
        for sample in pcm_data:
            if self.law == 'ulaw':
                encoded.append(G711Codec.pcm_to_mulaw(int(sample)))
            else:
                encoded.append(G711Codec.pcm_to_alaw(int(sample)))
        return bytes(encoded)
```

---

## 5. BUFFER MANAGEMENT & JITTER HANDLING

### 5.1 Jitter Buffer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     JITTER BUFFER SYSTEM                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Input     │--->│   Packet    │--->│   Decode    │         │
│  │   Queue     │    │   Reorder   │    │   Buffer    │         │
│  │  (Network)  │    │   (Sort)    │    │  (PCM Out)  │         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘         │
│                            │                                     │
│                     ┌──────┴──────┐                             │
│                     │   Adaptive  │                             │
│                     │   Control   │                             │
│                     │  (PLC/VAD)  │                             │
│                     └─────────────┘                             │
│                                                                  │
│  Metrics:                                                        │
│  - Jitter: 20-60ms (adaptive)                                   │
│  - Target Delay: 50ms nominal                                   │
│  - Max Delay: 200ms                                             │
│  - Underrun Threshold: 3 packets                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Adaptive Jitter Buffer Implementation

```python
import heapq
from collections import deque

class JitterPacket:
    def __init__(self, sequence: int, timestamp: float, arrival_time: float, payload: bytes):
        self.sequence = sequence
        self.timestamp = timestamp
        self.arrival_time = arrival_time
        self.payload = payload
        self.decoded = False

class AdaptiveJitterBuffer:
    def __init__(self, min_delay_ms: int = 20, max_delay_ms: int = 200, 
                 target_jitter_ms: int = 50, sample_rate: int = 48000):
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_jitter_ms = target_jitter_ms
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_samples = int(sample_rate * self.frame_duration_ms / 1000)

        self.packets = {}
        self.sorted_sequences = []
        self.next_read_seq = None

        self.jitter_estimate = 0.0
        self.delay_estimate = min_delay_ms
        self.packet_times = deque(maxlen=100)

        self.stats = {
            'packets_received': 0, 'packets_lost': 0, 'packets_late': 0,
            'underruns': 0, 'overruns': 0,
            'current_delay_ms': min_delay_ms, 'current_jitter_ms': 0
        }

    def add_packet(self, sequence: int, payload: bytes, timestamp: int):
        arrival_time = time.time()
        if self.next_read_seq is None:
            self.next_read_seq = sequence
        if sequence < self.next_read_seq:
            self.stats['packets_late'] += 1
            return
        packet = JitterPacket(sequence, timestamp, arrival_time, payload)
        if sequence not in self.packets:
            self.packets[sequence] = packet
            heapq.heappush(self.sorted_sequences, sequence)
            self.stats['packets_received'] += 1
            self._update_jitter(arrival_time, timestamp)

    def _update_jitter(self, arrival_time: float, timestamp: int):
        if len(self.packet_times) > 0:
            last_arrival, last_timestamp = self.packet_times[-1]
            transit = arrival_time - (timestamp / self.sample_rate)
            last_transit = last_arrival - (last_timestamp / self.sample_rate)
            delta = abs(transit - last_transit)
            self.jitter_estimate += (delta - self.jitter_estimate) / 16
        self.packet_times.append((arrival_time, timestamp))
        self._adapt_delay()

    def _adapt_delay(self):
        jitter_ms = self.jitter_estimate * 1000
        self.stats['current_jitter_ms'] = jitter_ms
        target_delay = max(self.min_delay_ms, min(self.max_delay_ms, 2 * jitter_ms + 20))
        alpha = 0.1
        self.delay_estimate = (1 - alpha) * self.delay_estimate + alpha * target_delay
        self.stats['current_delay_ms'] = self.delay_estimate
```

---

## 6. PACKET LOSS CONCEALMENT

### 6.1 PLC Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  PACKET LOSS CONCEALMENT                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Detect    │--->│   Analyze   │--->│  Conceal    │         │
│  │    Loss     │    │   History   │    │   Audio     │         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘         │
│                            │                                     │
│                     ┌──────┴──────┐                             │
│                     │  Strategy   │                             │
│                     │  Selection  │                             │
│                     └─────────────┘                             │
│                                                                  │
│  Strategies:                                                     │
│  1. Zero Insertion (emergency)                                  │
│  2. Pattern Repetition (1-2 frames)                             │
│  3. Waveform Substitution (3-5 frames)                          │
│  4. Pitch-based WSOLA (5+ frames)                               │
│  5. Model-based synthesis (extended loss)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 PLC Implementation

```python
from scipy import signal

class PacketLossConcealment:
    def __init__(self, sample_rate: int = 48000, frame_samples: int = 960, history_frames: int = 10):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.history_frames = history_frames
        self.history = deque(maxlen=history_frames)
        self.consecutive_losses = 0
        self.pitch_period = 0
        self.last_valid_frame = None
        self.repetition_limit = 2
        self.waveform_limit = 5

    def add_frame(self, frame: np.ndarray):
        self.history.append(frame.copy())
        self.last_valid_frame = frame.copy()
        self.consecutive_losses = 0

    def conceal(self) -> np.ndarray:
        self.consecutive_losses += 1
        if self.consecutive_losses <= self.repetition_limit:
            return self._repetition_concealment()
        elif self.consecutive_losses <= self.waveform_limit:
            return self._waveform_concealment()
        else:
            return self._advanced_concealment()

    def _repetition_concealment(self) -> np.ndarray:
        if self.last_valid_frame is not None:
            attenuation = 0.9 ** self.consecutive_losses
            return self.last_valid_frame * attenuation
        return np.zeros(self.frame_samples, dtype=np.float32)

    def _waveform_concealment(self) -> np.ndarray:
        if len(self.history) < 2:
            return self._repetition_concealment()
        if self.pitch_period == 0:
            self.pitch_period = self._estimate_pitch()
        if self.pitch_period > 0:
            return self._pitch_synchronous_insertion()
        else:
            return self._noise_substitution()

    def _estimate_pitch(self) -> int:
        if len(self.history) < 2:
            return 0
        signal_hist = np.concatenate(list(self.history))
        autocorr = np.correlate(signal_hist, signal_hist, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        min_period = self.sample_rate // 500
        max_period = self.sample_rate // 50
        if max_period > len(autocorr):
            max_period = len(autocorr)
        peak_idx = min_period + np.argmax(autocorr[min_period:max_period])
        if autocorr[peak_idx] > 0.5 * autocorr[0]:
            return peak_idx
        return 0

    def _pitch_synchronous_insertion(self) -> np.ndarray:
        if len(self.history) == 0:
            return np.zeros(self.frame_samples, dtype=np.float32)
        last_frame = self.history[-1]
        output = np.zeros(self.frame_samples, dtype=np.float32)
        period = self.pitch_period if self.pitch_period > 0 else self.frame_samples // 4
        pos = 0
        while pos < self.frame_samples:
            start = len(last_frame) - period
            if start < 0:
                start = 0
            period_data = last_frame[start:start + period]
            end_pos = min(pos + len(period_data), self.frame_samples)
            output[pos:end_pos] = period_data[:end_pos - pos]
            pos += period
        fade = 1.0 - (self.consecutive_losses / self.waveform_limit) * 0.3
        return output * fade

    def _noise_substitution(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(self.frame_samples, dtype=np.float32)
        signal_hist = np.concatenate(list(self.history))
        noise_level = np.std(signal_hist) * 0.3
        noise = np.random.normal(0, noise_level, self.frame_samples)
        b, a = signal.butter(2, 0.3, 'low')
        shaped_noise = signal.filtfilt(b, a, noise)
        return shaped_noise.astype(np.float32)

    def _advanced_concealment(self) -> np.ndarray:
        if self.last_valid_frame is not None:
            fade_factor = max(0, 1.0 - (self.consecutive_losses - self.waveform_limit) * 0.1)
            if fade_factor > 0:
                noise = self._noise_substitution()
                mixed = self.last_valid_frame * fade_factor + noise * (1 - fade_factor)
                return mixed * 0.5
        return np.zeros(self.frame_samples, dtype=np.float32)
```

---

## 7. STREAM SYNCHRONIZATION

### 7.1 Synchronization Architecture

```python
import ntplib

class SyncPoint:
    def __init__(self, rtp_timestamp: int, ntp_timestamp: float, local_time: float):
        self.rtp_timestamp = rtp_timestamp
        self.ntp_timestamp = ntp_timestamp
        self.local_time = local_time

class StreamSynchronizer:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.streams = {}
        self.master_clock_offset = 0.0
        self.ntp_client = ntplib.NTPClient()
        self._sync_with_ntp()

    def _sync_with_ntp(self):
        try:
            response = self.ntp_client.request('pool.ntp.org', version=3)
            self.master_clock_offset = response.offset
        except:
            self.master_clock_offset = 0.0

    def map_rtp_to_ntp(self, stream_id: str, rtp_timestamp: int) -> float:
        stream = self.streams.get(stream_id)
        if not stream or not stream.sync_point:
            return time.time() + self.master_clock_offset
        rtp_diff = (rtp_timestamp - stream.sync_point.rtp_timestamp) & 0xFFFFFFFF
        time_diff = rtp_diff / self.sample_rate
        return stream.sync_point.ntp_timestamp + time_diff

    def get_playout_time(self, stream_id: str, rtp_timestamp: int) -> float:
        ntp_time = self.map_rtp_to_ntp(stream_id, rtp_timestamp)
        local_time = ntp_time - self.master_clock_offset
        return local_time
```

---

## 8. QUALITY OF SERVICE (QoS)

### 8.1 QoS Implementation

```python
class QoSMetrics:
    def __init__(self):
        self.timestamp = time.time()
        self.latency_ms = 0.0
        self.jitter_ms = 0.0
        self.packet_loss_percent = 0.0
        self.bitrate_kbps = 0.0
        self.mos_score = 4.0
        self.buffer_underruns = 0
        self.buffer_overruns = 0

class QoSMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.current_metrics = QoSMetrics()
        self.thresholds = {
            'latency_high': 150, 'latency_critical': 300,
            'loss_high': 3.0, 'loss_critical': 5.0,
            'jitter_high': 50, 'mos_low': 3.0, 'mos_critical': 2.0
        }

    def _calculate_mos(self) -> float:
        delay = self.current_metrics.latency_ms
        if delay < 150:
            id_delay = 0
        elif delay < 400:
            id_delay = 0.024 * delay - 3.6
        else:
            id_delay = 0.12 * delay - 48
        loss = self.current_metrics.packet_loss_percent
        ie = 10 + 30 * np.log(1 + 5 * loss / 100)
        r_factor = 93.2 - id_delay - ie
        if r_factor < 0:
            mos = 1.0
        elif r_factor > 100:
            mos = 4.5
        else:
            mos = 1 + 0.035 * r_factor + r_factor * (r_factor - 60) * (100 - r_factor) * 7 * 10**-6
        return max(1.0, min(4.5, mos))

    def get_qos_level(self) -> str:
        m = self.current_metrics
        t = self.thresholds
        if (m.latency_ms > t['latency_critical'] or
            m.packet_loss_percent > t['loss_critical'] or
            m.mos_score < t['mos_critical']):
            return 'CRITICAL'
        if (m.latency_ms > t['latency_high'] or
            m.packet_loss_percent > t['loss_high'] or
            m.jitter_ms > t['jitter_high'] or
            m.mos_score < t['mos_low']):
            return 'POOR'
        return 'GOOD'

class AdaptiveBitrateController:
    def __init__(self, initial_bitrate: int = 24000, min_bitrate: int = 6000, max_bitrate: int = 128000):
        self.current_bitrate = initial_bitrate
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.loss_history = deque(maxlen=20)
        self.rtt_history = deque(maxlen=20)
        self.increase_cooldown = 0

    def update(self, packet_loss: float, rtt_ms: float) -> int:
        self.loss_history.append(packet_loss)
        self.rtt_history.append(rtt_ms)
        avg_loss = np.mean(self.loss_history)
        avg_rtt = np.mean(self.rtt_history)

        if avg_loss > 5.0:
            self.current_bitrate = max(self.min_bitrate, int(self.current_bitrate * 0.7))
            self.increase_cooldown = 10
        elif avg_loss > 2.0:
            self.current_bitrate = max(self.min_bitrate, int(self.current_bitrate * 0.85))
            self.increase_cooldown = 5
        elif avg_loss < 1.0 and avg_rtt < 100:
            if self.increase_cooldown > 0:
                self.increase_cooldown -= 1
            else:
                self.current_bitrate = min(self.max_bitrate, int(self.current_bitrate * 1.1))
        return self.current_bitrate

class DSCPMarker:
    DSCP_EF = 0x2E << 2      # Expedited Forwarding (Voice)
    DSCP_AF41 = 0x22 << 2    # Assured Forwarding 4,1 (Video)
    DSCP_AF31 = 0x1A << 2    # Assured Forwarding 3,1 (Signaling)
    DSCP_DEFAULT = 0x00 << 2 # Default (Best Effort)

    @staticmethod
    def set_dscp(socket_obj, dscp_value: int) -> bool:
        try:
            socket_obj.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, dscp_value)
            return True
        except:
            return False

    @staticmethod
    def mark_voice_traffic(socket_obj) -> bool:
        return DSCPMarker.set_dscp(socket_obj, DSCPMarker.DSCP_EF)
```

---

## 9. SYSTEM ARCHITECTURE DIAGRAMS

### 9.1 Complete Audio Streaming Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        WINDOWS 10 AI AGENT SYSTEM                             │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         AI AGENT CORE                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │  GPT-5.2 │  │  Agent   │  │  Identity│  │   Soul   │  │  Cron    │ │  │
│  │  │  Engine  │  │  Loops   │  │  System  │  │  Engine  │  │  Jobs    │ │  │
│  │  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  │       │             │                                                    │  │
│  │       └─────────────┴──────────────────────────────────┐               │  │
│  │                                                        │               │  │
│  │  ┌─────────────────────────────────────────────────────┴───────────┐  │  │
│  │  │                  AUDIO STREAMING INFRASTRUCTURE                  │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │  │
│  │  │  │   WebSocket  │  │   RTP/RTCP   │  │   Media      │          │  │  │
│  │  │  │   Gateway    │  │   Handler    │  │   Pipeline   │          │  │  │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │  │  │
│  │  │         │                 │                 │                   │  │  │
│  │  │         └─────────────────┴─────────────────┘                   │  │  │
│  │  │                           │                                     │  │  │
│  │  │                    ┌──────┴──────┐                              │  │  │
│  │  │                    │   Buffer &  │                              │  │  │
│  │  │                    │   Sync Mgr  │                              │  │  │
│  │  │                    └──────┬──────┘                              │  │  │
│  │  │                           │                                     │  │  │
│  │  │  ┌────────────────────────┼────────────────────────┐           │  │  │
│  │  │  │              CODEC LAYER                        │           │  │  │
│  │  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │           │  │  │
│  │  │  │  │  Opus  │ │G.711 mu│ │G.711 A │ │ G.722  │   │           │  │  │
│  │  │  │  │Encoder │ │ -law   │ │ -law   │ │        │   │           │  │  │
│  │  │  │  └────────┘ └────────┘ └────────┘ └────────┘   │           │  │  │
│  │  │  └────────────────────────────────────────────────┘           │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    EXTERNAL INTEGRATIONS                        │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │  │  │
│  │  │  │  Twilio  │  │  Gmail   │  │  Browser │  │  System  │       │  │  │
│  │  │  │  Voice   │  │  API     │  │  Control │  │  Access  │       │  │  │
│  │  │  │  /SMS    │  │          │  │          │  │          │       │  │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      MEDIA SERVER CLUSTER                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │   WebSocket  │  │   RTP/RTCP   │  │   Recording  │  │  Stream   │  │  │
│  │  │   Gateway    │  │   Handler    │  │   Service    │  │  Service  │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. IMPLEMENTATION REFERENCE

### 10.1 Configuration Summary

```python
AUDIO_STREAMING_CONFIG = {
    'websocket': {
        'host': '0.0.0.0', 'port': 8443, 'ssl_enabled': True,
        'ping_interval': 15, 'ping_timeout': 10, 'max_connections': 1000,
        'compression': None, 'tcp_nodelay': True,
    },
    'rtp': {
        'port_range': (10000, 20000), 'dtls_enabled': True,
        'srtp_enabled': True, 'payload_type': 111,
    },
    'codec': {
        'primary': 'opus', 'fallback': 'g711_ulaw',
        'opus_bitrate': 24000, 'opus_complexity': 10,
        'opus_frame_duration': 20, 'opus_fec': True, 'opus_dtx': True,
    },
    'buffer': {
        'min_delay_ms': 20, 'max_delay_ms': 200,
        'target_jitter_ms': 50, 'adaptive': True,
    },
    'plc': {
        'enabled': True, 'repetition_limit': 2, 'waveform_limit': 5,
    },
    'qos': {
        'dscp_enabled': True, 'dscp_value': 0xB8,
        'adaptive_bitrate': True, 'min_bitrate': 6000, 'max_bitrate': 128000,
    },
    'sync': {
        'ntp_enabled': True, 'max_sync_diff_ms': 40, 'drift_compensation': True,
    },
    'media_server': {
        'mixer_enabled': True, 'max_mix_streams': 32,
        'recording_enabled': True, 'recording_format': 'mp3',
        'recording_bitrate': 128000,
    }
}
```

### 10.2 Performance Targets

| Metric | Target | Maximum |
|--------|--------|---------|
| End-to-End Latency | <100ms | 150ms |
| Jitter | <20ms | 50ms |
| Packet Loss | <1% | 5% |
| MOS Score | >4.0 | >3.0 |
| Codec CPU Usage | <5% | <10% |
| Buffer Underruns | 0/hour | <5/hour |
| Concurrent Streams | 100 | 1000 |

### 10.3 Protocol References

- RFC 3550: RTP - A Transport Protocol for Real-Time Applications
- RFC 3551: RTP Profile for Audio and Video Conferences
- RFC 3711: The Secure Real-time Transport Protocol (SRTP)
- RFC 4566: SDP - Session Description Protocol
- RFC 5245: Interactive Connectivity Establishment (ICE)
- RFC 6455: The WebSocket Protocol
- RFC 6716: Definition of the Opus Audio Codec

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: AI Streaming Architecture Team*
