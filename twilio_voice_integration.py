"""
OpenClaw AI Agent - Twilio Voice Integration Module
PSTN and VoIP telephony integration for Windows 10 AI agent

This module provides Twilio Voice capabilities for the OpenClaw AI agent system,
including inbound/outbound calls, bidirectional streaming, and conference bridging.
"""

import asyncio
import base64
import json
import logging
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import audioop
from urllib.parse import urlparse
import websockets

# Twilio imports
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Say

logger = logging.getLogger(__name__)


class CallState(Enum):
    """Call state enumeration"""
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    CONNECTED = "connected"
    ON_HOLD = "on_hold"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class TwilioConfig:
    """Twilio configuration"""
    account_sid: str
    auth_token: str
    phone_number: str
    webhook_url: str
    stream_url: str
    recording_enabled: bool = True
    recording_channels: str = "dual"  # "mono" or "dual"


@dataclass
class CallInfo:
    """Information about an active call"""
    call_sid: str
    stream_sid: str
    from_number: str
    to_number: str
    state: CallState
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    websocket: Optional[websockets.WebSocketServerProtocol] = None


class TwilioMediaStream:
    """
    Handles bidirectional media streaming with Twilio
    """
    
    # Audio format constants
    SAMPLE_RATE = 8000  # Twilio uses 8kHz
    CHANNELS = 1
    MULAW_BITS = 8
    
    def __init__(self, call_info: CallInfo):
        self.call_info = call_info
        self._is_running = False
        
        # Audio callbacks
        self.on_incoming_audio: Optional[Callable[[np.ndarray], None]] = None
        self.on_call_start: Optional[Callable[[CallInfo], None]] = None
        self.on_call_end: Optional[Callable[[CallInfo], None]] = None
        
        # Outgoing audio queue
        self._outgoing_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        
    async def start(self):
        """Start the media stream"""
        self._is_running = True
        
        # Start receive and send tasks
        receive_task = asyncio.create_task(self._receive_loop())
        send_task = asyncio.create_task(self._send_loop())
        
        await asyncio.gather(receive_task, send_task)
    
    async def stop(self):
        """Stop the media stream"""
        self._is_running = False
    
    async def _receive_loop(self):
        """Receive audio from Twilio"""
        websocket = self.call_info.websocket
        
        try:
            async for message in websocket:
                if not self._is_running:
                    break
                
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == "start":
                        # Call started
                        self.call_info.start_time = asyncio.get_event_loop().time()
                        self.call_info.state = CallState.CONNECTED
                        
                        logger.info(f"Call started: {self.call_info.call_sid}")
                        
                        if self.on_call_start:
                            await self._safe_callback(self.on_call_start, self.call_info)
                    
                    elif event_type == "media":
                        # Received audio data
                        payload = data["media"]["payload"]
                        audio_data = self._decode_mulaw(payload)
                        
                        if self.on_incoming_audio:
                            await self._safe_callback(self.on_incoming_audio, audio_data)
                    
                    elif event_type == "stop":
                        # Call ended
                        self.call_info.end_time = asyncio.get_event_loop().time()
                        self.call_info.state = CallState.DISCONNECTED
                        
                        logger.info(f"Call ended: {self.call_info.call_sid}")
                        
                        if self.on_call_end:
                            await self._safe_callback(self.on_call_end, self.call_info)
                        
                        break
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from Twilio")
                except (OSError, ValueError, KeyError) as e:
                    logger.error(f"Error processing message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for call: {self.call_info.call_sid}")
            self.call_info.state = CallState.DISCONNECTED
    
    async def _send_loop(self):
        """Send audio to Twilio"""
        websocket = self.call_info.websocket
        
        while self._is_running:
            try:
                # Wait for audio to send
                audio_data = await asyncio.wait_for(
                    self._outgoing_queue.get(),
                    timeout=0.1
                )
                
                # Encode and send
                encoded = self._encode_mulaw(audio_data)
                
                message = {
                    "event": "media",
                    "streamSid": self.call_info.stream_sid,
                    "media": {
                        "payload": encoded
                    }
                }
                
                await websocket.send(json.dumps(message))
            
            except asyncio.TimeoutError:
                continue
            except (OSError, RuntimeError) as e:
                logger.error(f"Error sending audio: {e}")

    async def _safe_callback(self, callback, *args):
        """Safely execute a callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Callback error: {e}")
    
    def _decode_mulaw(self, payload: str) -> np.ndarray:
        """Decode mu-law audio to PCM"""
        # Decode base64
        mulaw_data = base64.b64decode(payload)
        
        # Convert mu-law to 16-bit PCM
        pcm_data = audioop.ulaw2lin(mulaw_data, 2)
        
        # Convert to numpy array
        return np.frombuffer(pcm_data, dtype=np.int16)
    
    def _encode_mulaw(self, pcm_data: np.ndarray) -> str:
        """Encode PCM audio to mu-law"""
        # Ensure correct format
        if pcm_data.dtype != np.int16:
            pcm_data = (pcm_data * 32767).astype(np.int16)
        
        # Convert to bytes
        pcm_bytes = pcm_data.tobytes()
        
        # Convert to mu-law
        mulaw_data = audioop.lin2ulaw(pcm_bytes, 2)
        
        # Encode to base64
        return base64.b64encode(mulaw_data).decode('utf-8')
    
    async def send_audio(self, audio_data: np.ndarray):
        """Queue audio to send to the call"""
        try:
            # Resample to 8kHz if needed
            if len(audio_data) > 0:
                # Proper resampling using scipy
                try:
                    from scipy.signal import resample
                    target_len = int(len(audio_data) * 8000 / 48000)
                    audio_8k = resample(audio_data, target_len)
                except ImportError:
                    # Fallback: linear interpolation
                    ratio = 8000 / 48000
                    target_len = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, target_len)
                    audio_8k = np.interp(indices, np.arange(len(audio_data)), audio_data)
                
                self._outgoing_queue.put_nowait(audio_8k)
        except asyncio.QueueFull:
            logger.warning("Outgoing audio queue full, dropping frame")


class TwilioVoiceManager:
    """
    Manages Twilio Voice operations
    """
    
    @staticmethod
    def _validate_webhook_url(url: str, label: str = "webhook_url"):
        """Validate that a webhook URL is well-formed HTTPS."""
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "wss"):
            raise ValueError(
                f"{label} must use HTTPS/WSS, got {parsed.scheme!r}: {url}"
            )
        if not parsed.hostname:
            raise ValueError(f"{label} has no hostname: {url}")

    def __init__(self, config: TwilioConfig):
        # Validate URLs before storing config
        self._validate_webhook_url(config.webhook_url, "webhook_url")
        self._validate_webhook_url(config.stream_url, "stream_url")

        self.config = config
        self.client = Client(config.account_sid, config.auth_token)
        
        # Active calls
        self.active_calls: Dict[str, CallInfo] = {}
        self.media_streams: Dict[str, TwilioMediaStream] = {}
        
        # Callbacks
        self.on_incoming_call: Optional[Callable[[CallInfo], None]] = None
        self.on_call_audio: Optional[Callable[[str, np.ndarray], None]] = None
        self.on_call_start: Optional[Callable[[CallInfo], None]] = None
        self.on_call_end: Optional[Callable[[CallInfo], None]] = None
        
        # WebSocket server
        self._websocket_server = None
    
    def generate_incoming_twiml(self) -> str:
        """Generate TwiML for incoming calls"""
        response = VoiceResponse()
        
        # Connect to media stream
        connect = Connect()
        connect.stream(url=self.config.stream_url)
        response.append(connect)
        
        # Fallback message
        response.say("Connecting to AI agent. Please wait.")
        
        return str(response)
    
    def generate_outgoing_twiml(self) -> str:
        """Generate TwiML for outgoing calls"""
        response = VoiceResponse()
        
        # Connect to media stream
        connect = Connect()
        connect.stream(url=self.config.stream_url)
        response.append(connect)
        
        return str(response)
    
    async def make_call(self, to_number: str, 
                        from_number: str = None) -> Optional[str]:
        """Make an outbound call"""
        try:
            call = self.client.calls.create(
                to=to_number,
                from_=from_number or self.config.phone_number,
                url=self.config.webhook_url,
                record=self.config.recording_enabled,
                recording_channels=self.config.recording_channels
            )
            
            logger.info(f"Outbound call initiated: {call.sid} to {to_number}")
            return call.sid
        
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Error making call: {e}")
            return None

    def send_sms(self, to: str, body: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send an SMS message."""
        try:
            message = self.client.messages.create(
                body=body,
                from_=from_number or self.config.phone_number,
                to=to
            )
            return {"sent": True, "sid": message.sid, "status": message.status}
        except (ConnectionError, TimeoutError, ValueError) as e:
            logging.error(f"Failed to send SMS: {e}")
            return {"sent": False, "error": str(e)}

    async def hangup_call(self, call_sid: str):
        """Hang up a call"""
        try:
            call = self.client.calls(call_sid).update(status="completed")
            logger.info(f"Call hung up: {call_sid}")
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Error hanging up call: {e}")
    
    async def start_media_stream_server(self, host: str = "0.0.0.0", port: int = 8766):
        """Start WebSocket server for media streams"""
        logger.info(f"Starting media stream server on {host}:{port}")
        
        self._websocket_server = await websockets.serve(
            self._handle_media_stream,
            host,
            port
        )
    
    async def _handle_media_stream(self, websocket, path):
        """Handle incoming media stream WebSocket"""
        call_info = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == "start":
                        # New call started
                        start_data = data.get("start", {})
                        
                        call_info = CallInfo(
                            call_sid=start_data.get("callSid"),
                            stream_sid=start_data.get("streamSid"),
                            from_number=start_data.get("from"),
                            to_number=start_data.get("to"),
                            state=CallState.CONNECTED,
                            websocket=websocket
                        )
                        
                        self.active_calls[call_info.call_sid] = call_info
                        
                        # Create media stream handler with per-call audio callback
                        media_stream = TwilioMediaStream(call_info)
                        sid = call_info.call_sid

                        async def _per_call_audio(audio, _sid=sid):
                            await self._on_incoming_audio(_sid, audio)

                        media_stream.on_incoming_audio = _per_call_audio
                        media_stream.on_call_start = self._on_call_start
                        media_stream.on_call_end = self._on_call_end
                        
                        self.media_streams[call_info.call_sid] = media_stream
                        
                        # Notify incoming call handler
                        if self.on_incoming_call:
                            await self._safe_callback(self.on_incoming_call, call_info)
                        
                        # Start media stream processing
                        await media_stream.start()
                        break  # Handled by media stream
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from Twilio")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Media stream WebSocket closed")
        finally:
            # Cleanup
            if call_info and call_info.call_sid in self.active_calls:
                del self.active_calls[call_info.call_sid]
            if call_info and call_info.call_sid in self.media_streams:
                del self.media_streams[call_info.call_sid]
    
    async def _on_incoming_audio(self, call_sid: str, audio_data: np.ndarray):
        """Handle incoming audio from a specific call.

        Supports concurrent calls by routing via the explicit call_sid
        rather than scanning for the first connected call.
        """
        if self.on_call_audio:
            await self._safe_callback(self.on_call_audio, call_sid, audio_data)
    
    async def _on_call_start(self, call_info: CallInfo):
        """Handle call start"""
        if self.on_call_start:
            await self._safe_callback(self.on_call_start, call_info)
    
    async def _on_call_end(self, call_info: CallInfo):
        """Handle call end"""
        if self.on_call_end:
            await self._safe_callback(self.on_call_end, call_info)
        
        # Cleanup
        if call_info.call_sid in self.active_calls:
            del self.active_calls[call_info.call_sid]
        if call_info.call_sid in self.media_streams:
            del self.media_streams[call_info.call_sid]
    
    async def _safe_callback(self, callback, *args):
        """Safely execute a callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Callback error: {e}")

    async def send_audio_to_call(self, call_sid: str, audio_data: np.ndarray):
        """Send audio to a specific call"""
        if call_sid in self.media_streams:
            await self.media_streams[call_sid].send_audio(audio_data)
    
    async def stop(self):
        """Stop the voice manager"""
        # Hang up all active calls
        for call_sid in list(self.active_calls.keys()):
            await self.hangup_call(call_sid)
        
        # Stop media streams
        for media_stream in self.media_streams.values():
            await media_stream.stop()
        
        self.active_calls.clear()
        self.media_streams.clear()
        
        # Close WebSocket server
        if self._websocket_server:
            self._websocket_server.close()
            await self._websocket_server.wait_closed()


class TwilioConferenceBridge:
    """
    Conference bridge for multiple Twilio calls.

    Supports two modes:
    - Twilio-native: delegates mixing to Twilio's <Conference> TwiML verb
    - Server-side: mixes audio locally and pushes the result to each participant
      (useful when AI-generated audio needs to be injected)
    """

    def __init__(self, voice_manager: TwilioVoiceManager,
                 server_side_mix: bool = False):
        self.voice_manager = voice_manager
        self.conferences: Dict[str, List[str]] = {}  # conference_id -> [call_sids]
        self._server_side_mix = server_side_mix
        # Per-conference accumulator: {conf_id: {call_sid: latest_audio}}
        self._audio_buffers: Dict[str, Dict[str, np.ndarray]] = {}
        self._mix_tasks: Dict[str, asyncio.Task] = {}

    async def create_conference(self, conference_id: str):
        """Create a new conference"""
        self.conferences[conference_id] = []
        self._audio_buffers[conference_id] = {}
        logger.info(f"Conference created: {conference_id}")

        if self._server_side_mix:
            self._mix_tasks[conference_id] = asyncio.create_task(
                self._mix_loop(conference_id)
            )

    async def add_to_conference(self, conference_id: str, call_sid: str):
        """Add a call to a conference"""
        if conference_id not in self.conferences:
            await self.create_conference(conference_id)

        self.conferences[conference_id].append(call_sid)
        self._audio_buffers.setdefault(conference_id, {})[call_sid] = None

        if not self._server_side_mix:
            # Twilio-native conference
            try:
                self.voice_manager.client.calls(call_sid).update(
                    twiml=f"""
                    <Response>
                        <Dial>
                            <Conference>{conference_id}</Conference>
                        </Dial>
                    </Response>
                    """
                )
                logger.info(f"Call {call_sid} added to conference {conference_id}")
            except (ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Error adding call to conference: {e}")
        else:
            logger.info(f"Call {call_sid} added to server-side conference {conference_id}")

    def push_call_audio(self, conference_id: str, call_sid: str,
                        audio: np.ndarray):
        """Push incoming audio from a participant for server-side mixing."""
        if conference_id in self._audio_buffers:
            self._audio_buffers[conference_id][call_sid] = audio

    async def _mix_loop(self, conference_id: str):
        """Server-side audio mixing loop for a conference."""
        interval = 0.020  # 20ms mix interval (matches typical ptime)
        while conference_id in self.conferences:
            await asyncio.sleep(interval)
            buffers = self._audio_buffers.get(conference_id, {})
            participants = self.conferences.get(conference_id, [])
            if len(participants) < 2:
                continue

            # Gather available audio
            audio_map: Dict[str, np.ndarray] = {}
            for sid, audio in buffers.items():
                if audio is not None:
                    audio_map[sid] = audio
                    buffers[sid] = None  # consume

            if not audio_map:
                continue

            # For each participant, mix all *other* participants' audio
            for target_sid in participants:
                others = [a for sid, a in audio_map.items() if sid != target_sid]
                if not others:
                    continue
                # Sum and normalize
                mixed = np.sum(others, axis=0).astype(np.float64)
                if len(others) > 1:
                    mixed /= np.sqrt(len(others))
                mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
                # Send to participant
                await self.voice_manager.send_audio_to_call(target_sid, mixed)

    async def remove_from_conference(self, conference_id: str, call_sid: str):
        """Remove a call from a conference"""
        if conference_id in self.conferences:
            if call_sid in self.conferences[conference_id]:
                self.conferences[conference_id].remove(call_sid)

                if conference_id in self._audio_buffers:
                    self._audio_buffers[conference_id].pop(call_sid, None)

                # Hang up the call
                await self.voice_manager.hangup_call(call_sid)

                logger.info(f"Call {call_sid} removed from conference {conference_id}")

    async def end_conference(self, conference_id: str):
        """End a conference and hang up all calls"""
        if conference_id in self.conferences:
            for call_sid in self.conferences[conference_id]:
                await self.voice_manager.hangup_call(call_sid)

            del self.conferences[conference_id]
            self._audio_buffers.pop(conference_id, None)

            # Cancel mix task if running
            task = self._mix_tasks.pop(conference_id, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            logger.info(f"Conference ended: {conference_id}")


# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        config = TwilioConfig(
            account_sid="your_account_sid",
            auth_token="your_auth_token",
            phone_number="+1234567890",
            webhook_url="https://your-domain.com/voice-webhook",
            stream_url="wss://your-domain.com/media-stream",
            recording_enabled=True
        )
        
        # Create voice manager
        voice_manager = TwilioVoiceManager(config)
        
        # Set up callbacks
        async def on_incoming_call(call_info: CallInfo):
            print(f"Incoming call from {call_info.from_number}")
        
        async def on_call_audio(call_sid: str, audio_data: np.ndarray):
            print(f"Received audio from {call_sid}: {len(audio_data)} samples")
        
        async def on_call_start(call_info: CallInfo):
            print(f"Call started: {call_info.call_sid}")
        
        async def on_call_end(call_info: CallInfo):
            print(f"Call ended: {call_info.call_sid}")
        
        voice_manager.on_incoming_call = on_incoming_call
        voice_manager.on_call_audio = on_call_audio
        voice_manager.on_call_start = on_call_start
        voice_manager.on_call_end = on_call_end
        
        # Start media stream server
        await voice_manager.start_media_stream_server(port=8766)
        
        print("Twilio voice integration running")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        
        await voice_manager.stop()
    
    asyncio.run(main())
