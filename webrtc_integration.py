"""
OpenClaw AI Agent - WebRTC Integration Module
Real-time peer-to-peer and SFU communication for Windows 10

This module provides WebRTC capabilities for the OpenClaw AI agent system,
including P2P connections, SFU integration, and media streaming.
"""

import asyncio
import json
import logging
import os
import numpy as np
from typing import Optional, Dict, Callable, Any, List
from dataclasses import dataclass
from enum import Enum
import base64

# aiortc imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.sdp import candidate_from_sdp

# WebSocket for signaling
import websockets

logger = logging.getLogger(__name__)


class WebRTCState(Enum):
    """WebRTC connection states"""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class WebRTCConfig:
    """WebRTC configuration"""
    ice_servers: List[Dict[str, Any]] = None
    audio_codec: str = "opus"
    sample_rate: int = 48000
    channels: int = 1
    bitrate: int = 32000
    ptime: int = 20
    
    def __post_init__(self):
        if self.ice_servers is None:
            self.ice_servers = [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
            ]


class AudioTrackProcessor:
    """
    Process incoming audio tracks from WebRTC
    """
    
    def __init__(self, track, on_frame: Callable[[np.ndarray], None]):
        self.track = track
        self.on_frame = on_frame
        self._is_running = False
        self._task = None
        
    async def start(self):
        """Start processing audio frames"""
        self._is_running = True
        self._task = asyncio.create_task(self._process_frames())
        
    async def stop(self):
        """Stop processing"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _process_frames(self):
        """Process incoming audio frames"""
        while self._is_running:
            try:
                frame = await self.track.recv()
                
                # Convert frame to numpy array
                audio_data = frame.to_ndarray()
                
                # Notify callback
                if self.on_frame:
                    self.on_frame(audio_data)
                    
            except (OSError, RuntimeError) as e:
                if self._is_running:
                    logger.error(f"Audio track processing error: {e}")
                break


class WebRTCConnection:
    """
    Manages a single WebRTC peer connection
    """
    
    def __init__(self, config: WebRTCConfig = None, 
                 connection_id: str = None):
        self.config = config or WebRTCConfig()
        self.connection_id = connection_id or self._generate_id()
        self.pc: Optional[RTCPeerConnection] = None
        self.state = WebRTCState.NEW
        
        # Audio handling
        self.audio_track = None
        self.track_processor = None
        self.on_audio_frame: Optional[Callable[[np.ndarray], None]] = None
        self.on_state_change: Optional[Callable[[WebRTCState], None]] = None
        
        # Data channel
        self.data_channel = None
        self.on_data_message: Optional[Callable[[str], None]] = None
        
    def _generate_id(self) -> str:
        """Generate unique connection ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def initialize(self):
        """Initialize the peer connection"""
        # Create peer connection with ICE servers
        self.pc = RTCPeerConnection(
            configuration={"iceServers": self.config.ice_servers}
        )
        
        # Set up event handlers
        self.pc.on("iceconnectionstatechange", self._on_ice_state_change)
        self.pc.on("track", self._on_track)
        self.pc.on("datachannel", self._on_data_channel)
        
        self.state = WebRTCState.CONNECTING
        logger.info(f"WebRTC connection initialized: {self.connection_id}")
    
    def _on_ice_state_change(self):
        """Handle ICE connection state changes"""
        ice_state = self.pc.iceConnectionState
        logger.info(f"ICE state changed: {ice_state}")
        
        state_map = {
            "new": WebRTCState.NEW,
            "checking": WebRTCState.CONNECTING,
            "connected": WebRTCState.CONNECTED,
            "completed": WebRTCState.CONNECTED,
            "failed": WebRTCState.FAILED,
            "disconnected": WebRTCState.DISCONNECTED,
            "closed": WebRTCState.CLOSED,
        }
        
        self.state = state_map.get(ice_state, WebRTCState.FAILED)
        
        if self.on_state_change:
            self.on_state_change(self.state)
    
    def _on_track(self, track):
        """Handle incoming media track"""
        logger.info(f"Received track: {track.kind}")
        
        if track.kind == "audio":
            self.audio_track = track
            
            # Start processing audio
            if self.on_audio_frame:
                self.track_processor = AudioTrackProcessor(
                    track, self.on_audio_frame
                )
                asyncio.create_task(self.track_processor.start())
    
    def _on_data_channel(self, channel):
        """Handle incoming data channel"""
        logger.info(f"Received data channel: {channel.label}")
        self.data_channel = channel
        
        @channel.on("message")
        def on_message(message):
            if self.on_data_message:
                self.on_data_message(message)
    
    async def create_offer(self) -> Dict[str, Any]:
        """Create SDP offer"""
        if not self.pc:
            raise RuntimeError("Peer connection not initialized")
        
        # Create data channel for messaging
        self.data_channel = self.pc.createDataChannel("messages")
        
        @self.data_channel.on("message")
        def on_message(message):
            if self.on_data_message:
                self.on_data_message(message)
        
        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        return {
            "type": "offer",
            "sdp": self.pc.localDescription.sdp,
        }
    
    async def create_answer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Create SDP answer"""
        if not self.pc:
            raise RuntimeError("Peer connection not initialized")
        
        # Set remote description
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )
        
        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return {
            "type": "answer",
            "sdp": self.pc.localDescription.sdp,
        }
    
    async def set_answer(self, answer: Dict[str, Any]):
        """Set remote answer"""
        if not self.pc:
            raise RuntimeError("Peer connection not initialized")
        
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
    
    async def add_ice_candidate(self, candidate: Dict[str, Any]):
        """Add ICE candidate"""
        if not self.pc:
            return
        
        try:
            ice_candidate = candidate_from_sdp(candidate["candidate"])
            ice_candidate.sdpMid = candidate.get("sdpMid")
            ice_candidate.sdpMLineIndex = candidate.get("sdpMLineIndex")
            await self.pc.addIceCandidate(ice_candidate)
        except (ValueError, OSError) as e:
            logger.error(f"Error adding ICE candidate: {e}")
    
    async def add_audio_track(self, audio_source):
        """Add local audio track"""
        if not self.pc:
            raise RuntimeError("Peer connection not initialized")
        
        self.pc.addTrack(audio_source)
    
    async def send_data(self, message: str):
        """Send data channel message"""
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(message)
    
    async def close(self):
        """Close the connection"""
        if self.track_processor:
            await self.track_processor.stop()
        
        if self.pc:
            await self.pc.close()
            self.pc = None
        
        self.state = WebRTCState.CLOSED
        logger.info(f"WebRTC connection closed: {self.connection_id}")


class WebRTCSignalingServer:
    """
    WebSocket signaling server for WebRTC
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.rooms: Dict[str, List[str]] = {}
        self.on_message: Optional[Callable[[str, str, Any], None]] = None
        
    async def start(self):
        """Start the signaling server"""
        logger.info(f"Starting signaling server on {self.host}:{self.port}")
        
        async with websockets.serve(self._handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    async def _handle_client(self, websocket, path):
        """Handle client connection"""
        client_id = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                msg_type = data.get("type")
                
                if msg_type == "register":
                    client_id = data.get("clientId")
                    self.clients[client_id] = websocket
                    logger.info(f"Client registered: {client_id}")
                    
                    await self._send(websocket, {
                        "type": "registered",
                        "clientId": client_id
                    })
                
                elif msg_type == "join":
                    room_id = data.get("roomId")
                    if room_id not in self.rooms:
                        self.rooms[room_id] = []
                    self.rooms[room_id].append(client_id)
                    
                    # Notify other participants
                    for other_id in self.rooms[room_id]:
                        if other_id != client_id and other_id in self.clients:
                            await self._send(self.clients[other_id], {
                                "type": "peer-joined",
                                "peerId": client_id
                            })
                
                elif msg_type in ["offer", "answer", "ice-candidate"]:
                    target_id = data.get("targetId")
                    if target_id in self.clients:
                        # Forward message to target
                        forward_msg = {
                            "type": msg_type,
                            "sourceId": client_id,
                            "data": data.get("data")
                        }
                        await self._send(self.clients[target_id], forward_msg)
                
                # Custom message handler
                if self.on_message:
                    await self.on_message(client_id, msg_type, data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        finally:
            if client_id:
                await self._remove_client(client_id)
    
    async def _send(self, websocket, message: dict):
        """Send message to websocket"""
        try:
            await websocket.send(json.dumps(message))
        except (OSError, RuntimeError) as e:
            logger.error(f"Error sending message: {e}")
    
    async def _remove_client(self, client_id: str):
        """Remove client from all rooms"""
        if client_id in self.clients:
            del self.clients[client_id]
        
        # Remove from rooms
        for room_id, members in self.rooms.items():
            if client_id in members:
                members.remove(client_id)
                # Notify other participants
                for other_id in members:
                    if other_id in self.clients:
                        await self._send(self.clients[other_id], {
                            "type": "peer-left",
                            "peerId": client_id
                        })


class WebRTCManager:
    """
    Manages multiple WebRTC connections
    """
    
    def __init__(self):
        self.connections: Dict[str, WebRTCConnection] = {}
        self.signaling: Optional[WebRTCSignalingServer] = None
        self.on_audio_frame: Optional[Callable[[str, np.ndarray], None]] = None
        
    async def start_signaling(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the signaling server"""
        self.signaling = WebRTCSignalingServer(host, port)
        self.signaling.on_message = self._on_signaling_message
        
        # Start in background
        asyncio.create_task(self.signaling.start())
    
    async def _on_signaling_message(self, client_id: str, msg_type: str, data: dict):
        """Handle signaling messages"""
        logger.debug(f"Signaling message from {client_id}: {msg_type}")
    
    async def create_connection(self, config: WebRTCConfig = None) -> WebRTCConnection:
        """Create a new WebRTC connection"""
        connection = WebRTCConnection(config)
        
        # Set up audio frame handler
        def on_frame(frame):
            if self.on_audio_frame:
                self.on_audio_frame(connection.connection_id, frame)
        
        connection.on_audio_frame = on_frame
        
        await connection.initialize()
        
        self.connections[connection.connection_id] = connection
        
        return connection
    
    async def close_connection(self, connection_id: str):
        """Close a connection"""
        if connection_id in self.connections:
            await self.connections[connection_id].close()
            del self.connections[connection_id]
    
    async def close_all(self):
        """Close all connections"""
        for connection in list(self.connections.values()):
            await connection.close()
        self.connections.clear()


# Example usage
if __name__ == "__main__":
    async def main():
        # Create WebRTC manager
        manager = WebRTCManager()
        
        # Set up audio frame handler
        def on_audio(connection_id: str, frame: np.ndarray):
            print(f"Received audio from {connection_id}: {frame.shape}")
        
        manager.on_audio_frame = on_audio
        
        # Start signaling server
        await manager.start_signaling(port=8765)
        
        ws_url = os.environ.get('WEBRTC_SIGNALING_URL', 'ws://localhost:8765')
        logger.info(f"WebRTC signaling server running on {ws_url}")
        logger.info("Press Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        
        await manager.close_all()
    
    asyncio.run(main())
