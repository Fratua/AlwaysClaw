# Conversational AI & Dialogue Management System
## Technical Specification for Windows 10 OpenClaw AI Agent

**Version:** 1.0  
**Target Platform:** Windows 10  
**AI Core:** GPT-5.2 with Enhanced Thinking  
**Integration:** Gmail, Browser, TTS, STT, Twilio, System Access  

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Dialogue State Management](#2-dialogue-state-management)
3. [Turn-Taking Algorithms](#3-turn-taking-algorithms)
4. [Barge-In Detection & Handling](#4-barge-in-detection--handling)
5. [Context Preservation System](#5-context-preservation-system)
6. [Intent Recognition Engine](#6-intent-recognition-engine)
7. [Response Generation Timing](#7-response-generation-timing)
8. [Conversation Repair Mechanisms](#8-conversation-repair-mechanisms)
9. [Multi-Modal Integration](#9-multi-modal-integration)
10. [Implementation Reference](#10-implementation-reference)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL AI SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   STT Input  │───▶│   Dialogue   │───▶│   TTS Output │                  │
│  │   (Whisper)  │    │   Manager    │    │  (ElevenLabs)│                  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘                  │
│                             │                                               │
│  ┌──────────────┐    ┌──────▼───────┐    ┌──────────────┐                  │
│  │  Visual Input│◀──▶│   Context    │◀──▶│  GPT-5.2     │                  │
│  │  (Screen/Img)│    │   Memory     │    │  Core        │                  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘                  │
│                             │                                               │
│  ┌──────────────┐    ┌──────▼───────┐    ┌──────────────┐                  │
│  │   Intent     │◀──▶│   State      │◀──▶│  Action      │                  │
│  │   Engine     │    │   Machine    │    │  Executor    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INTEGRATION LAYER                                 │   │
│  │  Gmail │ Browser │ Twilio │ System │ Cron │ Heartbeat │ User Mgmt   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Speech-to-Text | OpenAI Whisper v3 | Real-time transcription |
| Text-to-Speech | ElevenLabs Turbo v2.5 | Natural voice synthesis |
| LLM Core | GPT-5.2 (high thinking) | Reasoning & generation |
| Dialogue Manager | Custom Python | State & flow control |
| Context Store | SQLite + Redis | Persistent memory |
| Intent Classifier | Fine-tuned BERT + GPT | Multi-turn understanding |
| Visual Processor | GPT-4 Vision | Screen/image analysis |

---

## 2. Dialogue State Management

### 2.1 State Machine Architecture

```python
class DialogueStateMachine:
    """
    Hierarchical state machine for conversation management
    """
    
    STATES = {
        # Primary States
        'IDLE': {
            'transitions': ['LISTENING', 'SPEAKING', 'PROCESSING'],
            'timeout': None,
            'description': 'Waiting for user activation'
        },
        'LISTENING': {
            'transitions': ['PROCESSING', 'BARGE_IN', 'TIMEOUT'],
            'timeout': 30000,  # 30s silence timeout
            'description': 'Actively receiving user input'
        },
        'PROCESSING': {
            'transitions': ['SPEAKING', 'CLARIFYING', 'REPAIRING'],
            'timeout': 5000,   # 5s processing timeout
            'description': 'Analyzing input and generating response'
        },
        'SPEAKING': {
            'transitions': ['LISTENING', 'BARGE_IN', 'WAITING_CONFIRM'],
            'timeout': None,
            'description': 'Delivering response to user'
        },
        'CLARIFYING': {
            'transitions': ['LISTENING', 'REPAIRING'],
            'timeout': 15000,
            'description': 'Requesting clarification from user'
        },
        'REPAIRING': {
            'transitions': ['LISTENING', 'SPEAKING'],
            'timeout': 10000,
            'description': 'Handling conversation repair'
        },
        'BARGE_IN': {
            'transitions': ['PROCESSING', 'LISTENING'],
            'timeout': 500,
            'description': 'Handling user interruption'
        },
        'WAITING_CONFIRM': {
            'transitions': ['LISTENING', 'EXECUTING', 'CANCELLED'],
            'timeout': 10000,
            'description': 'Awaiting user confirmation'
        },
        'EXECUTING': {
            'transitions': ['SPEAKING', 'ERROR_RECOVERY'],
            'timeout': 60000,
            'description': 'Executing user-requested action'
        },
        'ERROR_RECOVERY': {
            'transitions': ['SPEAKING', 'LISTENING'],
            'timeout': 5000,
            'description': 'Recovering from error state'
        }
    }
```

### 2.2 State Tracking Data Structure

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum, auto

class ConversationPhase(Enum):
    GREETING = auto()
    TASK_ORIENTED = auto()
    INFORMATIONAL = auto()
    SOCIAL = auto()
    CLOSING = auto()
    REPAIR = auto()

@dataclass
class DialogueState:
    """Complete state representation for a conversation turn"""
    
    # Core State
    current_state: str = 'IDLE'
    previous_state: str = 'IDLE'
    state_timestamp: datetime = field(default_factory=datetime.now)
    state_duration_ms: int = 0
    
    # Conversation Context
    conversation_id: str = ''
    user_id: str = ''
    session_id: str = ''
    turn_number: int = 0
    
    # Phase & Topic Tracking
    conversation_phase: ConversationPhase = ConversationPhase.GREETING
    current_topic: str = ''
    topic_stack: List[str] = field(default_factory=list)
    
    # Intent & Action
    current_intent: Optional[Dict] = None
    pending_action: Optional[Dict] = None
    action_stack: List[Dict] = field(default_factory=list)
    
    # User Input State
    user_input_buffer: str = ''
    user_input_confidence: float = 0.0
    user_emotion: str = 'neutral'
    user_engagement: float = 1.0
    
    # Agent Output State
    agent_response_buffer: str = ''
    agent_speaking_progress: float = 0.0  # 0.0 to 1.0
    agent_intended_action: Optional[str] = None
    
    # Turn Management
    last_speaker: str = 'none'  # 'user', 'agent', 'none'
    turn_transition_type: str = ''  # 'normal', 'barge_in', 'timeout', 'repair'
    
    # Context Window
    recent_utterances: List[Dict] = field(default_factory=list)
    context_window_size: int = 10
    
    # Multi-modal State
    visual_context: Optional[Dict] = None
    screen_capture_active: bool = False
    
    # Timing
    silence_duration_ms: int = 0
    response_latency_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for storage/transmission"""
        return {
            'current_state': self.current_state,
            'previous_state': self.previous_state,
            'conversation_phase': self.conversation_phase.name,
            'turn_number': self.turn_number,
            'current_topic': self.current_topic,
            'topic_stack': self.topic_stack,
            'user_engagement': self.user_engagement,
            'last_speaker': self.last_speaker,
            'silence_duration_ms': self.silence_duration_ms
        }
```

### 2.3 State Transition Logic

```python
class StateTransitionManager:
    """Manages valid state transitions with validation and hooks"""
    
    def __init__(self):
        self.transition_hooks: Dict[tuple, List[callable]] = {}
        self.state_validators: Dict[str, callable] = {}
        
    def register_transition_hook(
        self, 
        from_state: str, 
        to_state: str, 
        hook: callable
    ):
        """Register a hook to execute on specific transition"""
        key = (from_state, to_state)
        if key not in self.transition_hooks:
            self.transition_hooks[key] = []
        self.transition_hooks[key].append(hook)
        
    def validate_transition(
        self, 
        current_state: str, 
        target_state: str,
        context: Dict
    ) -> tuple[bool, str]:
        """Validate if transition is allowed"""
        
        # Check if transition exists in state definition
        if target_state not in DialogueStateMachine.STATES[current_state]['transitions']:
            return False, f"Invalid transition: {current_state} -> {target_state}"
            
        # Run custom validators
        if target_state in self.state_validators:
            validator = self.state_validators[target_state]
            is_valid, reason = validator(context)
            if not is_valid:
                return False, reason
                
        return True, "OK"
        
    async def execute_transition(
        self,
        state: DialogueState,
        target_state: str,
        context: Dict
    ) -> DialogueState:
        """Execute state transition with hooks"""
        
        from_state = state.current_state
        
        # Validate
        is_valid, reason = self.validate_transition(from_state, target_state, context)
        if not is_valid:
            raise StateTransitionError(reason)
            
        # Execute pre-transition hooks
        key = (from_state, target_state)
        if key in self.transition_hooks:
            for hook in self.transition_hooks[key]:
                await hook(state, context)
                
        # Perform transition
        state.previous_state = from_state
        state.current_state = target_state
        state.state_timestamp = datetime.now()
        state.state_duration_ms = 0
        
        # Update turn tracking
        if target_state == 'SPEAKING':
            state.last_speaker = 'agent'
        elif target_state == 'LISTENING':
            state.last_speaker = 'user'
            
        return state
```

---

## 3. Turn-Taking Algorithms

### 3.1 Turn-Taking Model

```python
from enum import Enum
import numpy as np
from typing import Optional, Callable

class TurnTransitionSignal(Enum):
    """Signals that indicate turn transition points"""
    USER_STOPPED_SPEAKING = "user_stopped"
    USER_PAUSE_DETECTED = "user_pause"
    SENTENCE_BOUNDARY = "sentence_end"
    QUESTION_DETECTED = "question"
    COMMAND_COMPLETE = "command_complete"
    AGENT_COMPLETE = "agent_complete"
    BARGE_IN_DETECTED = "barge_in"
    TIMEOUT_REACHED = "timeout"
    INTERRUPTION_REQUESTED = "interruption"

class TurnTakingManager:
    """
    Advanced turn-taking coordination system
    Implements collaborative floor management model
    """
    
    def __init__(self):
        # Timing parameters (milliseconds)
        self.min_turn_gap_ms = 200      # Minimum gap between turns
        self.max_turn_gap_ms = 2000     # Maximum acceptable gap
        self.pause_threshold_ms = 500   # Pause considered as turn yield
        self.silence_timeout_ms = 30000 # Timeout for silence
        
        # Confidence thresholds
        self.speech_end_confidence = 0.85
        self.turn_yield_confidence = 0.75
        
        # State
        self.current_floor_holder: str = 'none'
        self.turn_queue: List[str] = []
        self.transition_callbacks: List[Callable] = []
        
    async def analyze_turn_transition(
        self,
        audio_features: Dict,
        transcript_buffer: str,
        state: DialogueState
    ) -> Optional[TurnTransitionSignal]:
        """
        Analyze audio and text to detect turn transition opportunities
        """
        signals = []
        
        # 1. Analyze audio features for speech end
        speech_end_score = self._detect_speech_end(audio_features)
        if speech_end_score > self.speech_end_confidence:
            signals.append((TurnTransitionSignal.USER_STOPPED_SPEAKING, speech_end_score))
            
        # 2. Detect pause patterns
        pause_score = self._analyze_pause_pattern(audio_features)
        if pause_score > self.turn_yield_confidence:
            signals.append((TurnTransitionSignal.USER_PAUSE_DETECTED, pause_score))
            
        # 3. Analyze transcript for completion cues
        text_signals = self._analyze_transcript_cues(transcript_buffer)
        signals.extend(text_signals)
        
        # 4. Check for barge-in patterns
        barge_in_score = self._detect_barge_in_pattern(audio_features, state)
        if barge_in_score > 0.7:
            signals.append((TurnTransitionSignal.BARGE_IN_DETECTED, barge_in_score))
            
        # Return highest confidence signal
        if signals:
            signals.sort(key=lambda x: x[1], reverse=True)
            return signals[0][0]
            
        return None
        
    def _detect_speech_end(self, audio_features: Dict) -> float:
        """Detect if user has finished speaking"""
        # Energy-based detection
        energy = audio_features.get('rms_energy', 0)
        energy_history = audio_features.get('energy_history', [])
        
        if len(energy_history) < 3:
            return 0.0
            
        # Detect energy drop pattern
        recent_avg = np.mean(energy_history[-3:])
        baseline = np.mean(energy_history[:-3]) if len(energy_history) > 3 else recent_avg
        
        # Sharp drop indicates speech end
        if baseline > 0.1 and recent_avg < baseline * 0.3:
            return 0.9
            
        # Gradual decline
        if recent_avg < baseline * 0.5:
            return 0.6
            
        return 0.0
        
    def _analyze_pause_pattern(self, audio_features: Dict) -> float:
        """Analyze pause duration and pattern"""
        silence_duration = audio_features.get('silence_duration_ms', 0)
        
        if silence_duration > self.pause_threshold_ms:
            # Normalize confidence based on pause length
            confidence = min(silence_duration / self.max_turn_gap_ms, 1.0)
            return confidence
            
        return 0.0
        
    def _analyze_transcript_cues(self, transcript: str) -> List[tuple]:
        """Analyze text for turn-yielding cues"""
        signals = []
        
        if not transcript:
            return signals
            
        # Sentence completion markers
        completion_markers = ['.', '?', '!', '...', 'okay', 'right', 'so']
        for marker in completion_markers:
            if transcript.rstrip().endswith(marker):
                signals.append((TurnTransitionSignal.SENTENCE_BOUNDARY, 0.8))
                break
                
        # Question indicators (agent should respond)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you']
        transcript_lower = transcript.lower()
        for qw in question_words:
            if transcript_lower.startswith(qw) or f' {qw}' in transcript_lower:
                signals.append((TurnTransitionSignal.QUESTION_DETECTED, 0.85))
                break
                
        # Command completion patterns
        command_patterns = [
            'please', 'thanks', 'thank you', 'now', 'go ahead',
            'do it', 'execute', 'run'
        ]
        for pattern in command_patterns:
            if pattern in transcript_lower:
                signals.append((TurnTransitionSignal.COMMAND_COMPLETE, 0.75))
                break
                
        return signals
        
    def _detect_barge_in_pattern(
        self, 
        audio_features: Dict,
        state: DialogueState
    ) -> float:
        """Detect if user is attempting to interrupt"""
        # Check if agent is speaking
        if state.current_state != 'SPEAKING':
            return 0.0
            
        # Energy spike during agent speech
        energy = audio_features.get('rms_energy', 0)
        if energy > 0.3:  # Significant energy detected
            # Check for speech-like patterns
            spectral_features = audio_features.get('spectral_features', {})
            if spectral_features.get('speech_likelihood', 0) > 0.6:
                return 0.85
                
        return 0.0
```

### 3.2 Floor Management

```python
class FloorManager:
    """
    Manages conversation floor (who has the right to speak)
    Implements collaborative floor management
    """
    
    def __init__(self):
        self.floor_holder: Optional[str] = None
        self.floor_request_queue: List[str] = []
        self.floor_granted_callbacks: List[Callable] = []
        self.floor_released_callbacks: List[Callable] = []
        
        # Floor holding timeouts
        self.max_floor_duration_ms: int = 120000  # 2 minutes max
        self.floor_start_time: Optional[datetime] = None
        
    async def request_floor(
        self, 
        requester: str, 
        priority: int = 0,
        reason: str = ''
    ) -> bool:
        """Request the conversation floor"""
        
        # If floor is free, grant immediately
        if self.floor_holder is None:
            await self._grant_floor(requester)
            return True
            
        # If requester already has floor
        if self.floor_holder == requester:
            return True
            
        # Check for barge-in conditions
        if requester == 'user' and self.floor_holder == 'agent':
            # User barge-in during agent speech
            can_barge = await self._evaluate_barge_in_permission(reason)
            if can_barge:
                await self._release_floor('agent', reason='barge_in')
                await self._grant_floor('user')
                return True
                
        # Add to queue
        self.floor_request_queue.append({
            'requester': requester,
            'priority': priority,
            'reason': reason,
            'timestamp': datetime.now()
        })
        
        # Sort by priority
        self.floor_request_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        return False
        
    async def release_floor(self, holder: str, reason: str = ''):
        """Release the conversation floor"""
        if self.floor_holder == holder:
            await self._release_floor(holder, reason)
            
            # Grant to next in queue
            if self.floor_request_queue:
                next_request = self.floor_request_queue.pop(0)
                await self._grant_floor(next_request['requester'])
                
    async def _evaluate_barge_in_permission(self, reason: str) -> bool:
        """Evaluate if barge-in should be allowed"""
        # Always allow urgent interruptions
        urgent_keywords = ['stop', 'cancel', 'wait', 'hold on', 'no']
        if any(kw in reason.lower() for kw in urgent_keywords):
            return True
            
        # Allow if agent has been speaking for a while
        if self.floor_start_time:
            duration = (datetime.now() - self.floor_start_time).total_seconds() * 1000
            if duration > 10000:  # 10 seconds
                return True
                
        return False
        
    async def _grant_floor(self, holder: str):
        """Grant floor to holder"""
        self.floor_holder = holder
        self.floor_start_time = datetime.now()
        
        for callback in self.floor_granted_callbacks:
            await callback(holder)
            
    async def _release_floor(self, holder: str, reason: str):
        """Release floor from holder"""
        self.floor_holder = None
        self.floor_start_time = None
        
        for callback in self.floor_released_callbacks:
            await callback(holder, reason)
```

---

## 4. Barge-In Detection & Handling

### 4.1 Barge-In Detection System

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from collections import deque

@dataclass
class BargeInEvent:
    """Represents a detected barge-in event"""
    timestamp: float
    confidence: float
    trigger_type: str  # 'energy_spike', 'keyword', 'pattern'
    audio_position_ms: int
    transcript_snippet: str
    urgency_level: int  # 1-5, 5 being highest

class BargeInDetector:
    """
    Real-time barge-in detection for voice conversations
    Multi-modal detection using audio and semantic analysis
    """
    
    def __init__(self):
        # Detection parameters
        self.energy_threshold = 0.15
        self.energy_spike_ratio = 2.5
        self.min_barge_duration_ms = 150
        
        # State
        self.is_agent_speaking: bool = False
        self.agent_speech_start_ms: int = 0
        self.audio_buffer: deque = deque(maxlen=100)
        self.barge_in_history: List[BargeInEvent] = []
        
        # Urgent keywords that trigger immediate barge-in
        self.urgent_keywords = {
            'stop': 5, 'cancel': 5, 'wait': 4, 'hold': 4,
            'no': 4, 'wrong': 4, 'incorrect': 4,
            'pause': 3, 'hang on': 3, 'excuse me': 3
        }
        
    async def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        agent_speaking: bool,
        current_transcript: str
    ) -> Optional[BargeInEvent]:
        """Process audio chunk for barge-in detection"""
        
        self.is_agent_speaking = agent_speaking
        
        if not agent_speaking:
            return None
            
        # Extract features
        features = self._extract_audio_features(audio_chunk, sample_rate)
        self.audio_buffer.append(features)
        
        # Run detection algorithms
        detections = []
        
        # 1. Energy-based detection
        energy_detection = self._detect_energy_barge_in(features)
        if energy_detection:
            detections.append(energy_detection)
            
        # 2. Keyword-based detection
        keyword_detection = self._detect_keyword_barge_in(current_transcript)
        if keyword_detection:
            detections.append(keyword_detection)
            
        # 3. Pattern-based detection
        pattern_detection = self._detect_pattern_barge_in()
        if pattern_detection:
            detections.append(pattern_detection)
            
        # Return highest confidence detection
        if detections:
            best_detection = max(detections, key=lambda x: x.confidence)
            self.barge_in_history.append(best_detection)
            return best_detection
            
        return None
        
    def _extract_audio_features(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Dict:
        """Extract relevant audio features for barge-in detection"""
        
        # RMS Energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Zero-crossing rate (speech vs noise)
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        # Spectral centroid
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Voice activity detection features
        # Speech typically has spectral centroid between 200-4000 Hz
        speech_likelihood = 1.0 if 200 < spectral_centroid < 4000 else 0.3
        
        return {
            'rms_energy': rms,
            'zero_crossing_rate': zcr,
            'spectral_centroid': spectral_centroid,
            'speech_likelihood': speech_likelihood,
            'timestamp': datetime.now().timestamp()
        }
        
    def _detect_energy_barge_in(self, features: Dict) -> Optional[BargeInEvent]:
        """Detect barge-in based on energy patterns"""
        
        if len(self.audio_buffer) < 5:
            return None
            
        current_energy = features['rms_energy']
        recent_energies = [f['rms_energy'] for f in list(self.audio_buffer)[-5:]]
        avg_recent = np.mean(recent_energies[:-1])  # Exclude current
        
        # Detect significant energy spike
        if avg_recent > 0.01 and current_energy > avg_recent * self.energy_spike_ratio:
            if current_energy > self.energy_threshold:
                confidence = min(current_energy / (avg_recent * self.energy_spike_ratio), 1.0)
                
                # Check if speech-like
                if features['speech_likelihood'] > 0.5:
                    return BargeInEvent(
                        timestamp=datetime.now().timestamp(),
                        confidence=confidence * 0.9,
                        trigger_type='energy_spike',
                        audio_position_ms=0,
                        transcript_snippet='',
                        urgency_level=2
                    )
                    
        return None
        
    def _detect_keyword_barge_in(self, transcript: str) -> Optional[BargeInEvent]:
        """Detect barge-in based on urgent keywords"""
        
        if not transcript:
            return None
            
        transcript_lower = transcript.lower()
        
        for keyword, urgency in self.urgent_keywords.items():
            if keyword in transcript_lower:
                confidence = 0.7 + (urgency * 0.06)  # Higher urgency = higher confidence
                return BargeInEvent(
                    timestamp=datetime.now().timestamp(),
                    confidence=min(confidence, 1.0),
                    trigger_type='keyword',
                    audio_position_ms=0,
                    transcript_snippet=transcript[-50:],
                    urgency_level=urgency
                )
                
        return None
        
    def _detect_pattern_barge_in(self) -> Optional[BargeInEvent]:
        """Detect barge-in based on audio patterns"""
        
        if len(self.audio_buffer) < 10:
            return None
            
        # Analyze recent buffer for speech onset pattern
        recent = list(self.audio_buffer)[-10:]
        energies = [f['rms_energy'] for f in recent]
        speech_likelihoods = [f['speech_likelihood'] for f in recent]
        
        # Pattern: Low energy followed by rising energy + speech characteristics
        early_avg = np.mean(energies[:3])
        late_avg = np.mean(energies[7:])
        
        if early_avg < 0.05 and late_avg > 0.15:
            if np.mean(speech_likelihoods[5:]) > 0.6:
                return BargeInEvent(
                    timestamp=datetime.now().timestamp(),
                    confidence=0.75,
                    trigger_type='pattern',
                    audio_position_ms=0,
                    transcript_snippet='',
                    urgency_level=2
                )
                
        return None
```

### 4.2 Barge-In Response Handler

```python
class BargeInHandler:
    """
    Handles barge-in events with appropriate responses
    Manages graceful interruption and context preservation
    """
    
    def __init__(self, tts_controller, dialogue_manager):
        self.tts_controller = tts_controller
        self.dialogue_manager = dialogue_manager
        self.interruption_history: List[Dict] = []
        
        # Response strategies by urgency
        self.response_strategies = {
            5: self._handle_urgent_interruption,    # Immediate stop
            4: self._handle_high_interruption,      # Quick acknowledgment
            3: self._handle_medium_interruption,    # Graceful pause
            2: self._handle_low_interruption,       # Continue with note
            1: self._handle_minimal_interruption    # Ignore or defer
        }
        
    async def handle_barge_in(
        self,
        event: BargeInEvent,
        current_state: DialogueState
    ) -> DialogueState:
        """Handle detected barge-in event"""
        
        # Log interruption
        self.interruption_history.append({
            'event': event,
            'state_before': current_state.to_dict(),
            'timestamp': datetime.now()
        })
        
        # Get appropriate strategy
        strategy = self.response_strategies.get(
            event.urgency_level, 
            self._handle_medium_interruption
        )
        
        # Execute strategy
        new_state = await strategy(event, current_state)
        
        # Update interruption log
        self.interruption_history[-1]['state_after'] = new_state.to_dict()
        
        return new_state
        
    async def _handle_urgent_interruption(
        self,
        event: BargeInEvent,
        state: DialogueState
    ) -> DialogueState:
        """Handle urgent interruption - stop immediately"""
        
        # Immediate TTS stop
        await self.tts_controller.stop_immediate()
        
        # Save interrupted context
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        
        # Transition to listening
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        
        # Acknowledge interruption
        await self.tts_controller.speak("Yes?", priority='urgent')
        
        return state
        
    async def _handle_high_interruption(
        self,
        event: BargeInEvent,
        state: DialogueState
    ) -> DialogueState:
        """Handle high-priority interruption"""
        
        # Quick fade out
        await self.tts_controller.fade_out(duration_ms=300)
        
        # Save context
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        
        # Transition
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        
        return state
        
    async def _handle_medium_interruption(
        self,
        event: BargeInEvent,
        state: DialogueState
    ) -> DialogueState:
        """Handle medium-priority interruption"""
        
        # Find natural pause point
        pause_point = self.tts_controller.find_pause_point()
        
        if pause_point:
            # Wait for pause point then stop
            await self.tts_controller.stop_at_pause(pause_point)
        else:
            # Gradual stop
            await self.tts_controller.fade_out(duration_ms=500)
            
        # Save context
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        
        # Transition
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        
        return state
        
    async def _handle_low_interruption(
        self,
        event: BargeInEvent,
        state: DialogueState
    ) -> DialogueState:
        """Handle low-priority interruption - note and continue"""
        
        # Note the interruption but continue speaking
        state.pending_interruption = {
            'transcript_snippet': event.transcript_snippet,
            'timestamp': datetime.now()
        }
        
        # Continue current speech but at slightly lower volume
        await self.tts_controller.adjust_volume(0.8)
        
        return state
        
    async def _handle_minimal_interruption(
        self,
        event: BargeInEvent,
        state: DialogueState
    ) -> DialogueState:
        """Handle minimal interruption - defer response"""
        
        # Queue for later acknowledgment
        state.deferred_interruptions = state.deferred_interruptions or []
        state.deferred_interruptions.append({
            'event': event,
            'timestamp': datetime.now()
        })
        
        return state
```

---

## 5. Context Preservation System

### 5.1 Multi-Layer Context Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

@dataclass
class ConversationContext:
    """
    Multi-layer context preservation system
    Maintains context across conversation turns with varying persistence
    """
    
    # Layer 1: Immediate Turn Context (volatile)
    current_utterance: str = ''
    current_intent: Optional[Dict] = None
    current_entities: List[Dict] = field(default_factory=list)
    turn_start_time: datetime = field(default_factory=datetime.now)
    
    # Layer 2: Short-term Context (session-level)
    session_id: str = ''
    conversation_history: List[Dict] = field(default_factory=list)
    topic_stack: List[str] = field(default_factory=list)
    active_tasks: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Layer 3: Medium-term Context (conversation-level)
    conversation_id: str = ''
    conversation_summary: str = ''
    key_points: List[str] = field(default_factory=list)
    unresolved_references: List[Dict] = field(default_factory=list)
    user_emotional_state: Dict[str, float] = field(default_factory=dict)
    
    # Layer 4: Long-term Context (user-level, persistent)
    user_id: str = ''
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_patterns: Dict[str, Any] = field(default_factory=dict)
    learned_facts: List[Dict] = field(default_factory=list)
    relationship_history: List[Dict] = field(default_factory=list)
    
    # Context metadata
    last_updated: datetime = field(default_factory=datetime.now)
    context_version: int = 1
    
    def to_json(self) -> str:
        """Serialize context to JSON"""
        return json.dumps(self.to_dict(), default=str)
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'current_utterance': self.current_utterance,
            'current_intent': self.current_intent,
            'session_id': self.session_id,
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'topic_stack': self.topic_stack,
            'active_tasks': self.active_tasks,
            'key_points': self.key_points,
            'conversation_summary': self.conversation_summary,
            'last_updated': self.last_updated.isoformat()
        }

class ContextManager:
    """
    Manages context preservation across conversation turns
    Implements intelligent context windowing and summarization
    """
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.active_contexts: Dict[str, ConversationContext] = {}
        
        # Context window parameters
        self.max_history_turns = 20
        self.max_context_tokens = 4000
        self.summarization_threshold = 15
        
    async def load_context(
        self,
        user_id: str,
        session_id: str,
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        """Load or initialize conversation context"""
        
        context_key = f"{user_id}:{session_id}"
        
        # Check active contexts
        if context_key in self.active_contexts:
            return self.active_contexts[context_key]
            
        # Load from storage
        context_data = await self.storage.load_context(user_id, session_id)
        
        if context_data:
            context = self._deserialize_context(context_data)
        else:
            # Initialize new context
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id or self._generate_conversation_id()
            )
            
            # Load user profile
            context.user_profile = await self.storage.load_user_profile(user_id)
            
        self.active_contexts[context_key] = context
        return context
        
    async def update_context(
        self,
        context: ConversationContext,
        user_utterance: str,
        agent_response: str,
        intent: Optional[Dict] = None,
        entities: Optional[List[Dict]] = None
    ) -> ConversationContext:
        """Update context with new turn data"""
        
        # Create turn record
        turn_record = {
            'turn_number': len(context.conversation_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'user': user_utterance,
            'agent': agent_response,
            'intent': intent,
            'entities': entities or []
        }
        
        # Add to history
        context.conversation_history.append(turn_record)
        
        # Update current state
        context.current_utterance = user_utterance
        context.current_intent = intent
        context.current_entities = entities or []
        context.last_updated = datetime.now()
        
        # Update topic stack if new topic detected
        if intent and intent.get('topic'):
            if not context.topic_stack or context.topic_stack[-1] != intent['topic']:
                context.topic_stack.append(intent['topic'])
                # Keep only last 5 topics
                context.topic_stack = context.topic_stack[-5:]
                
        # Check if summarization needed
        if len(context.conversation_history) >= self.summarization_threshold:
            await self._summarize_context(context)
            
        # Prune history if too long
        if len(context.conversation_history) > self.max_history_turns:
            context.conversation_history = context.conversation_history[-self.max_history_turns:]
            
        # Save to storage
        await self._persist_context(context)
        
        return context
        
    async def _summarize_context(self, context: ConversationContext):
        """Generate summary of conversation history"""
        
        # Extract key points from older turns
        turns_to_summarize = context.conversation_history[:-10]
        
        # Use LLM to generate summary
        summary_prompt = self._build_summary_prompt(turns_to_summarize)
        summary = await self._generate_summary(summary_prompt)
        
        # Update context
        context.conversation_summary = summary
        context.key_points.extend(self._extract_key_points(turns_to_summarize))
        context.key_points = context.key_points[-20:]  # Keep last 20
        
        # Remove summarized turns
        context.conversation_history = context.conversation_history[-10:]
        
    def get_context_for_llm(
        self,
        context: ConversationContext,
        include_summary: bool = True,
        include_history: bool = True,
        include_profile: bool = True
    ) -> str:
        """Format context for LLM prompt injection"""
        
        parts = []
        
        # User profile
        if include_profile and context.user_profile:
            parts.append(f"User Profile: {json.dumps(context.user_profile, default=str)}")
            
        # Conversation summary
        if include_summary and context.conversation_summary:
            parts.append(f"Conversation Summary: {context.conversation_summary}")
            
        # Key points
        if context.key_points:
            parts.append(f"Key Points: {'; '.join(context.key_points[-10:])}")
            
        # Recent history
        if include_history and context.conversation_history:
            history_str = self._format_history(context.conversation_history[-5:])
            parts.append(f"Recent Conversation:\n{history_str}")
            
        # Active topics
        if context.topic_stack:
            parts.append(f"Active Topics: {' > '.join(context.topic_stack)}")
            
        # Active tasks
        if context.active_tasks:
            tasks_str = '; '.join([t.get('description', '') for t in context.active_tasks[-3:]])
            parts.append(f"Active Tasks: {tasks_str}")
            
        return '\n\n'.join(parts)
        
    def _format_history(self, turns: List[Dict]) -> str:
        """Format conversation history for LLM"""
        lines = []
        for turn in turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['agent']}")
        return '\n'.join(lines)
        
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
```

### 5.2 Reference Resolution System

```python
class ReferenceResolver:
    """
    Resolves anaphoric and deictic references in conversation
    Handles pronouns, demonstratives, and implicit references
    """
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        
        # Reference patterns
        self.pronouns = ['it', 'they', 'them', 'he', 'she', 'this', 'that', 'these', 'those']
        self.demonstratives = ['this', 'that', 'these', 'those', 'here', 'there']
        
    async def resolve_references(
        self,
        utterance: str,
        context: ConversationContext
    ) -> str:
        """Resolve all references in utterance"""
        
        resolved = utterance
        
        # Find references
        references = self._identify_references(utterance)
        
        for ref in references:
            antecedent = await self._find_antecedent(ref, context)
            if antecedent:
                resolved = resolved.replace(ref['text'], antecedent, 1)
                
        return resolved
        
    def _identify_references(self, utterance: str) -> List[Dict]:
        """Identify potential references in utterance"""
        
        references = []
        words = utterance.lower().split()
        
        for i, word in enumerate(words):
            if word in self.pronouns or word in self.demonstratives:
                references.append({
                    'text': word,
                    'position': i,
                    'type': 'pronoun' if word in self.pronouns else 'demonstrative'
                })
                
        return references
        
    async def _find_antecedent(
        self,
        reference: Dict,
        context: ConversationContext
    ) -> Optional[str]:
        """Find the antecedent for a reference"""
        
        # Check recent entities first
        if context.current_entities:
            for entity in reversed(context.current_entities):
                if self._entity_matches_reference(entity, reference):
                    return entity.get('text', entity.get('value', ''))
                    
        # Check conversation history
        for turn in reversed(context.conversation_history[-5:]):
            if turn.get('entities'):
                for entity in reversed(turn['entities']):
                    if self._entity_matches_reference(entity, reference):
                        return entity.get('text', entity.get('value', ''))
                        
        # Check active topics
        if context.topic_stack and reference['type'] == 'demonstrative':
            return context.topic_stack[-1]
            
        return None
        
    def _entity_matches_reference(self, entity: Dict, reference: Dict) -> bool:
        """Check if entity could be antecedent for reference"""
        
        entity_type = entity.get('type', '')
        ref_text = reference['text']
        
        # Pronoun matching
        if ref_text in ['it', 'this', 'that']:
            return entity_type in ['object', 'concept', 'task', 'topic']
        elif ref_text in ['they', 'them', 'these', 'those']:
            return entity_type in ['list', 'group', 'people']
        elif ref_text in ['he', 'she']:
            return entity_type == 'person'
            
        return False
```

---

## 6. Intent Recognition Engine

### 6.1 Multi-Level Intent Classification

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

class IntentCategory(Enum):
    """High-level intent categories"""
    INFORMATIONAL = "informational"
    TASK_ORIENTED = "task_oriented"
    SOCIAL = "social"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    REPAIR = "repair"
    META = "meta"

@dataclass
class Intent:
    """Structured intent representation"""
    
    # Primary classification
    category: IntentCategory
    primary_intent: str
    confidence: float
    
    # Secondary intents
    secondary_intents: List[tuple] = None  # (intent, confidence)
    
    # Intent parameters
    entities: List[Dict] = None
    slots: Dict[str, Any] = None
    
    # Contextual information
    topic: str = ''
    urgency: int = 1  # 1-5
    sentiment: str = 'neutral'
    
    # Temporal information
    temporal_references: List[Dict] = None
    
    def __post_init__(self):
        if self.secondary_intents is None:
            self.secondary_intents = []
        if self.entities is None:
            self.entities = []
        if self.slots is None:
            self.slots = {}
        if self.temporal_references is None:
            self.temporal_references = []

class IntentRecognitionEngine:
    """
    Multi-level intent recognition with contextual awareness
    Combines rule-based, ML-based, and LLM-based approaches
    """
    
    def __init__(self, llm_client, entity_extractor):
        self.llm_client = llm_client
        self.entity_extractor = entity_extractor
        
        # Intent patterns (rule-based)
        self.intent_patterns = self._load_intent_patterns()
        
        # Confidence thresholds
        self.rule_confidence_threshold = 0.8
        self.ml_confidence_threshold = 0.7
        
    async def recognize_intent(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        """
        Multi-stage intent recognition pipeline
        """
        
        # Stage 1: Preprocessing
        processed_utterance = self._preprocess(utterance)
        
        # Stage 2: Rule-based classification (fast path)
        rule_result = self._rule_based_classification(processed_utterance)
        if rule_result and rule_result.confidence >= self.rule_confidence_threshold:
            return await self._enrich_intent(rule_result, context, dialogue_state)
            
        # Stage 3: Context-aware classification
        context_result = await self._context_aware_classification(
            processed_utterance, context, dialogue_state
        )
        
        # Stage 4: LLM-based classification (fallback/enrichment)
        if context_result.confidence < 0.6:
            llm_result = await self._llm_classification(
                processed_utterance, context, dialogue_state
            )
            
            # Combine results
            final_result = self._combine_intent_results(context_result, llm_result)
        else:
            final_result = context_result
            
        # Stage 5: Enrichment
        enriched_intent = await self._enrich_intent(final_result, context, dialogue_state)
        
        return enriched_intent
        
    def _preprocess(self, utterance: str) -> str:
        """Preprocess utterance for intent recognition"""
        # Normalize
        processed = utterance.lower().strip()
        
        # Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'well']
        for filler in filler_words:
            processed = processed.replace(f' {filler} ', ' ')
            
        return processed
        
    def _rule_based_classification(self, utterance: str) -> Optional[Intent]:
        """Fast rule-based intent classification"""
        
        best_match = None
        best_confidence = 0.0
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in utterance:
                    confidence = len(pattern) / len(utterance)  # Simple scoring
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_name
                        
        if best_match:
            return Intent(
                category=self._infer_category(best_match),
                primary_intent=best_match,
                confidence=min(best_confidence * 1.5, 1.0)  # Boost rule confidence
            )
            
        return None
        
    async def _context_aware_classification(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        """Context-aware intent classification"""
        
        # Consider conversation state
        current_state = dialogue_state.current_state
        conversation_phase = dialogue_state.conversation_phase
        
        # Adjust intent probabilities based on context
        intent_scores = {}
        
        # If in clarification state, likely a response to clarification
        if current_state == 'CLARIFYING':
            intent_scores['clarification_response'] = 0.7
            
        # If in repair state, likely acknowledgment or correction
        if current_state == 'REPAIRING':
            intent_scores['acknowledgment'] = 0.6
            intent_scores['correction'] = 0.5
            
        # Check for continuation patterns
        if context.topic_stack:
            current_topic = context.topic_stack[-1]
            topic_keywords = self._get_topic_keywords(current_topic)
            
            keyword_match = sum(1 for kw in topic_keywords if kw in utterance)
            if keyword_match > 0:
                intent_scores['topic_continuation'] = 0.6 + (keyword_match * 0.1)
                
        # Check for task continuation
        if context.active_tasks:
            intent_scores['task_continuation'] = 0.5
            
        # Get base classification
        base_intent = await self._base_classification(utterance)
        
        # Merge with context scores
        for intent_name, score in intent_scores.items():
            if base_intent.primary_intent == intent_name:
                base_intent.confidence = max(base_intent.confidence, score)
            else:
                base_intent.secondary_intents.append((intent_name, score))
                
        return base_intent
        
    async def _llm_classification(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        """LLM-based intent classification"""
        
        prompt = f"""Analyze the following user utterance and classify the intent.

User Utterance: "{utterance}"

Conversation Context:
- Current Topic: {context.topic_stack[-1] if context.topic_stack else 'None'}
- Conversation Phase: {dialogue_state.conversation_phase.name}
- Recent History: {self._format_recent_history(context.conversation_history[-3:])}

Classify into one of these intent categories:
- INFORMATIONAL: seeking information
- TASK_ORIENTED: requesting action/task
- SOCIAL: social interaction
- NAVIGATIONAL: moving between topics/tasks
- TRANSACTIONAL: creating/updating/deleting data
- REPAIR: correcting or clarifying
- META: talking about the conversation itself

Output JSON format:
{{
    "category": "<category>",
    "primary_intent": "<specific intent>",
    "confidence": <0.0-1.0>,
    "entities": [{{"type": "<type>", "value": "<value>"}}],
    "urgency": <1-5>,
    "sentiment": "<positive|negative|neutral>"
}}"""

        response = await self.llm_client.generate(
            prompt,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response)
        
        return Intent(
            category=IntentCategory(result['category']),
            primary_intent=result['primary_intent'],
            confidence=result['confidence'],
            entities=result.get('entities', []),
            urgency=result.get('urgency', 1),
            sentiment=result.get('sentiment', 'neutral')
        )
        
    async def _enrich_intent(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        """Enrich intent with additional information"""
        
        # Extract entities
        entities = await self.entity_extractor.extract(
            dialogue_state.user_input_buffer,
            intent.primary_intent
        )
        intent.entities.extend(entities)
        
        # Extract temporal references
        intent.temporal_references = self._extract_temporal_references(
            dialogue_state.user_input_buffer
        )
        
        # Fill slots based on intent
        intent.slots = await self._fill_slots(intent, context)
        
        # Determine topic
        intent.topic = self._determine_topic(intent, context)
        
        return intent
        
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns"""
        
        return {
            # Informational intents
            'ask_question': [
                'what is', 'what are', 'how do', 'how can', 'how does',
                'why is', 'why does', 'when will', 'where is', 'who is',
                'tell me about', 'explain', 'describe'
            ],
            # Task-oriented intents
            'execute_command': [
                'open', 'close', 'start', 'stop', 'run', 'execute',
                'create', 'delete', 'update', 'send', 'write'
            ],
            'schedule_task': [
                'schedule', 'remind me', 'set a reminder', 'at',
                'tomorrow', 'next week', 'in an hour'
            ],
            # Social intents
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                'good evening', 'how are you'
            ],
            'farewell': [
                'goodbye', 'bye', 'see you', 'talk later', 'have a good'
            ],
            'gratitude': [
                'thank you', 'thanks', 'appreciate', 'grateful'
            ],
            # Repair intents
            'correction': [
                'no, i meant', 'that\'s not', 'wrong', 'incorrect',
                'i said', 'actually'
            ],
            'clarification_request': [
                'what do you mean', 'i don\'t understand', 'can you clarify',
                'explain that', 'say that again'
            ],
            # Meta intents
            'conversation_control': [
                'wait', 'hold on', 'pause', 'stop', 'continue',
                'start over', 'go back'
            ],
            'capability_check': [
                'can you', 'are you able', 'do you know how', 'is it possible'
            ]
        }
```

---

## 7. Response Generation Timing

### 7.1 Latency Management System

```python
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import asyncio

@dataclass
class TimingConstraints:
    """Timing constraints for response generation"""
    
    # Target latencies (milliseconds)
    max_first_response_ms: int = 800
    max_full_response_ms: int = 3000
    max_streaming_chunk_ms: int = 100
    
    # User patience thresholds
    silence_before_prompt_ms: int = 5000
    max_wait_without_feedback_ms: int = 2000
    
    # Adaptive timing
    enable_adaptive_timing: bool = True
    user_patience_factor: float = 1.0  # Adjust based on user behavior

class ResponseTimingManager:
    """
    Manages response generation timing for natural conversation flow
    Implements progressive disclosure and adaptive latency
    """
    
    def __init__(self, constraints: TimingConstraints = None):
        self.constraints = constraints or TimingConstraints()
        
        # Timing state
        self.request_start_time: Optional[datetime] = None
        self.first_token_time: Optional[datetime] = None
        self.response_stages: List[Dict] = []
        
        # Feedback mechanisms
        self.feedback_generators: Dict[str, Callable] = {}
        
    async def generate_timed_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        """
        Generate response with optimal timing
        """
        
        self.request_start_time = datetime.now()
        
        # Determine response strategy based on complexity
        complexity = self._assess_complexity(intent, context)
        
        if complexity == 'simple':
            return await self._generate_simple_response(
                intent, context, dialogue_state, response_generator
            )
        elif complexity == 'moderate':
            return await self._generate_moderate_response(
                intent, context, dialogue_state, response_generator
            )
        else:  # complex
            return await self._generate_complex_response(
                intent, context, dialogue_state, response_generator
            )
            
    async def _generate_simple_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        """Generate simple response with minimal latency"""
        
        # Direct generation without feedback
        response = await response_generator(intent, context, dialogue_state)
        
        self.first_token_time = datetime.now()
        
        # Check latency
        latency_ms = self._get_elapsed_ms()
        if latency_ms > self.constraints.max_first_response_ms:
            print(f"Warning: Response latency {latency_ms}ms exceeded target")
            
        return response
        
    async def _generate_moderate_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        """Generate moderate complexity response with acknowledgment"""
        
        # Send quick acknowledgment if processing takes time
        acknowledgment = self._generate_acknowledgment(intent)
        
        # Start response generation
        response_task = asyncio.create_task(
            response_generator(intent, context, dialogue_state)
        )
        
        # Wait with timeout for acknowledgment decision
        try:
            response = await asyncio.wait_for(
                response_task, 
                timeout=self.constraints.max_wait_without_feedback_ms / 1000
            )
            return response
        except asyncio.TimeoutError:
            # Provide acknowledgment and continue
            await self._deliver_acknowledgment(acknowledgment)
            
            # Wait for full response
            response = await response_task
            return response
            
    async def _generate_complex_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        """Generate complex response with progressive disclosure"""
        
        # Immediate acknowledgment
        acknowledgment = self._generate_acknowledgment(intent, detailed=True)
        await self._deliver_acknowledgment(acknowledgment)
        
        # Stream response chunks
        response_chunks = []
        chunk_stream = response_generator(intent, context, dialogue_state, stream=True)
        
        async for chunk in chunk_stream:
            response_chunks.append(chunk)
            
            # Deliver chunk if it's a natural break point
            if self._is_natural_break_point(chunk):
                await self._deliver_chunk(chunk)
                
        # Combine and return full response
        full_response = ' '.join(response_chunks)
        return full_response
        
    def _generate_acknowledgment(self, intent: Intent, detailed: bool = False) -> str:
        """Generate appropriate acknowledgment based on intent"""
        
        acknowledgments = {
            IntentCategory.TASK_ORIENTED: [
                "Let me do that for you.",
                "Working on it...",
                "I'll take care of that."
            ],
            IntentCategory.INFORMATIONAL: [
                "Let me find that for you.",
                "Looking that up...",
                "Let me check."
            ],
            IntentCategory.TRANSACTIONAL: [
                "Processing that now.",
                "Making the changes...",
                "Updating that for you."
            ]
        }
        
        category_acks = acknowledgments.get(intent.category, ["Okay..."])
        
        # Select based on urgency
        if intent.urgency >= 4:
            return "Right away."
        elif detailed:
            return category_acks[0]
        else:
            return category_acks[1]
            
    def _assess_complexity(
        self,
        intent: Intent,
        context: ConversationContext
    ) -> str:
        """Assess response complexity for timing strategy"""
        
        # Simple responses
        simple_intents = ['greeting', 'farewell', 'gratitude', 'acknowledgment']
        if intent.primary_intent in simple_intents:
            return 'simple'
            
        # Check for multi-step tasks
        if len(context.active_tasks) > 1:
            return 'complex'
            
        # Check for information retrieval needs
        if intent.category == IntentCategory.INFORMATIONAL:
            if any(kw in intent.primary_intent for kw in ['search', 'find', 'lookup']):
                return 'complex'
                
        # Check for external API calls
        if intent.slots and any(k in intent.slots for k in ['api_call', 'external_service']):
            return 'complex'
            
        return 'moderate'
        
    def _is_natural_break_point(self, text: str) -> bool:
        """Check if text ends at a natural break point"""
        
        break_indicators = ['. ', '? ', '! ', '; ', ': ', ', and ', ', but ']
        return any(text.endswith(ind) for ind in break_indicators)
        
    def _get_elapsed_ms(self) -> int:
        """Get elapsed time since request start"""
        if not self.request_start_time:
            return 0
        return int((datetime.now() - self.request_start_time).total_seconds() * 1000)
```

### 7.2 Streaming Response Management

```python
class StreamingResponseManager:
    """
    Manages streaming TTS responses for real-time conversation
    Handles chunking, buffering, and delivery timing
    """
    
    def __init__(self, tts_controller):
        self.tts_controller = tts_controller
        
        # Buffering parameters
        self.min_chunk_size = 50  # characters
        self.max_chunk_size = 200
        self.buffer_timeout_ms = 300
        
        # State
        self.text_buffer = ''
        self.buffer_start_time: Optional[datetime] = None
        self.is_streaming = False
        
    async def stream_response(
        self,
        text_stream,  # Async generator
        enable_interruption: bool = True
    ):
        """Stream response with optimal chunking"""
        
        self.is_streaming = True
        self.buffer_start_time = datetime.now()
        
        try:
            async for text_chunk in text_stream:
                self.text_buffer += text_chunk
                
                # Check if we should deliver buffered content
                if self._should_deliver_buffer():
                    await self._deliver_buffer(enable_interruption)
                    
            # Deliver remaining buffer
            if self.text_buffer:
                await self._deliver_buffer(enable_interruption)
                
        finally:
            self.is_streaming = False
            self.text_buffer = ''
            
    def _should_deliver_buffer(self) -> bool:
        """Determine if buffer should be delivered"""
        
        # Check buffer size
        if len(self.text_buffer) >= self.max_chunk_size:
            # Find last natural break
            break_point = self._find_break_point(self.text_buffer)
            if break_point > self.min_chunk_size:
                return True
                
        # Check timeout
        if self.buffer_start_time:
            elapsed_ms = (datetime.now() - self.buffer_start_time).total_seconds() * 1000
            if elapsed_ms > self.buffer_timeout_ms and len(self.text_buffer) >= self.min_chunk_size:
                return True
                
        return False
        
    def _find_break_point(self, text: str) -> int:
        """Find natural break point in text"""
        
        # Priority: sentence > clause > phrase
        sentence_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', text)]
        if sentence_breaks:
            return sentence_breaks[-1] + 2
            
        clause_breaks = [m.start() for m in re.finditer(r'[,;:]\s+', text)]
        if clause_breaks:
            return clause_breaks[-1] + 2
            
        phrase_breaks = [m.start() for m in re.finditer(r'\s+(and|but|or|so)\s+', text)]
        if phrase_breaks:
            return phrase_breaks[-1] + 1
            
        return len(text)
        
    async def _deliver_buffer(self, enable_interruption: bool):
        """Deliver buffered text to TTS"""
        
        # Find optimal break point
        break_point = self._find_break_point(self.text_buffer)
        
        # Split text
        to_deliver = self.text_buffer[:break_point]
        self.text_buffer = self.text_buffer[break_point:].lstrip()
        
        # Deliver to TTS
        await self.tts_controller.speak(
            to_deliver,
            allow_interruption=enable_interruption,
            stream=True
        )
        
        # Reset buffer timer
        self.buffer_start_time = datetime.now()
```

---

## 8. Conversation Repair Mechanisms

### 8.1 Repair Detection and Classification

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict

class RepairType(Enum):
    """Types of conversation repair"""
    SELF_INITIATED_SELF_REPAIR = auto()      # Agent corrects itself
    SELF_INITIATED_OTHER_REPAIR = auto()     # Agent asks for clarification
    OTHER_INITIATED_SELF_REPAIR = auto()     # User corrects agent
    OTHER_INITIATED_OTHER_REPAIR = auto()    # User asks for clarification
    
class RepairTrigger(Enum):
    """Triggers for repair"""
    LOW_CONFIDENCE = auto()
    MISUNDERSTANDING = auto()
    INCOMPREHENSIBLE = auto()
    AMBIGUITY = auto()
    INCOMPLETE = auto()
    ERROR = auto()

@dataclass
class RepairEvent:
    """Represents a conversation repair event"""
    repair_type: RepairType
    trigger: RepairTrigger
    confidence: float
    original_utterance: str
    problematic_segment: str
    repair_attempt: str
    timestamp: datetime
    success: Optional[bool] = None

class ConversationRepairManager:
    """
    Manages conversation repair for handling misunderstandings
    Implements self-repair and other-repair strategies
    """
    
    def __init__(self, llm_client, dialogue_manager):
        self.llm_client = llm_client
        self.dialogue_manager = dialogue_manager
        
        # Repair thresholds
        self.confidence_threshold = 0.6
        self.repair_history: List[RepairEvent] = []
        
        # Repair strategies
        self.repair_strategies = {
            RepairTrigger.LOW_CONFIDENCE: self._handle_low_confidence,
            RepairTrigger.MISUNDERSTANDING: self._handle_misunderstanding,
            RepairTrigger.INCOMPREHENSIBLE: self._handle_incomprehensible,
            RepairTrigger.AMBIGUITY: self._handle_ambiguity,
            RepairTrigger.INCOMPLETE: self._handle_incomplete,
            RepairTrigger.ERROR: self._handle_error
        }
        
    async def detect_need_for_repair(
        self,
        user_utterance: str,
        intent: Intent,
        dialogue_state: DialogueState
    ) -> Optional[RepairTrigger]:
        """Detect if repair is needed"""
        
        # Check confidence
        if intent.confidence < self.confidence_threshold:
            return RepairTrigger.LOW_CONFIDENCE
            
        # Check for explicit correction signals
        correction_signals = [
            'no', 'wrong', 'incorrect', 'not', "that's not",
            'i meant', 'i said', 'actually', 'wait'
        ]
        if any(sig in user_utterance.lower() for sig in correction_signals):
            return RepairTrigger.MISUNDERSTANDING
            
        # Check for incomprehensible input
        if self._is_incomprehensible(user_utterance):
            return RepairTrigger.INCOMPREHENSIBLE
            
        # Check for ambiguity
        if len(intent.secondary_intents) > 0:
            best_secondary = max(intent.secondary_intents, key=lambda x: x[1])
            if best_secondary[1] > intent.confidence * 0.8:
                return RepairTrigger.AMBIGUITY
                
        # Check for incomplete information
        required_slots = self._get_required_slots(intent.primary_intent)
        missing_slots = [s for s in required_slots if s not in intent.slots]
        if missing_slots:
            return RepairTrigger.INCOMPLETE
            
        return None
        
    async def initiate_repair(
        self,
        trigger: RepairTrigger,
        user_utterance: str,
        intent: Intent,
        dialogue_state: DialogueState
    ) -> str:
        """Initiate appropriate repair strategy"""
        
        strategy = self.repair_strategies.get(trigger)
        if not strategy:
            return "I'm not sure I understood that correctly. Could you rephrase?"
            
        repair_response = await strategy(user_utterance, intent, dialogue_state)
        
        # Log repair event
        repair_event = RepairEvent(
            repair_type=RepairType.SELF_INITIATED_OTHER_REPAIR,
            trigger=trigger,
            confidence=intent.confidence,
            original_utterance=user_utterance,
            problematic_segment='',
            repair_attempt=repair_response,
            timestamp=datetime.now()
        )
        self.repair_history.append(repair_event)
        
        # Transition to clarifying state
        await self.dialogue_manager.transition_to(dialogue_state, 'CLARIFYING')
        
        return repair_response
        
    async def handle_user_repair(
        self,
        user_utterance: str,
        dialogue_state: DialogueState
    ) -> str:
        """Handle user-initiated repair"""
        
        # Classify user repair type
        repair_type = self._classify_user_repair(user_utterance)
        
        # Update last repair event
        if self.repair_history:
            self.repair_history[-1].success = True
            
        # Acknowledge repair
        acknowledgment = self._generate_repair_acknowledgment(repair_type)
        
        # Transition back to normal flow
        await self.dialogue_manager.transition_to(dialogue_state, 'PROCESSING')
        
        return acknowledgment
        
    async def _handle_low_confidence(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle low confidence detection"""
        
        # Provide acknowledgment and ask for confirmation
        templates = [
            f"I understood you want to {intent.primary_intent}. Is that correct?",
            f"Let me make sure - you want me to {intent.primary_intent}?",
            f"Just to confirm, you're asking about {intent.topic}?"
        ]
        
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_misunderstanding(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle detected misunderstanding"""
        
        # Apologize and ask for clarification
        templates = [
            "I apologize for the confusion. Could you tell me again what you'd like?",
            "I'm sorry, I didn't get that right. Can you help me understand better?",
            "My mistake. Could you clarify what you meant?"
        ]
        
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_incomprehensible(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle incomprehensible input"""
        
        templates = [
            "I'm having trouble understanding. Could you say that differently?",
            "I didn't quite catch that. Can you rephrase?",
            "Sorry, I didn't understand. Could you try again?"
        ]
        
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_ambiguity(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle ambiguous intent"""
        
        # Present options
        options = [intent.primary_intent] + [i[0] for i in intent.secondary_intents[:2]]
        
        option_text = ' or '.join([f"{opt}" for opt in options])
        
        return f"I can interpret that a few ways. Did you mean {option_text}?"
        
    async def _handle_incomplete(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle incomplete information"""
        
        required_slots = self._get_required_slots(intent.primary_intent)
        missing = [s for s in required_slots if s not in intent.slots]
        
        if len(missing) == 1:
            return f"I'd be happy to help with that. What {missing[0]} did you have in mind?"
        else:
            missing_text = ', '.join(missing[:-1]) + f" and {missing[-1]}"
            return f"To help you with that, I'll need to know the {missing_text}."
            
    async def _handle_error(
        self,
        utterance: str,
        intent: Intent,
        state: DialogueState
    ) -> str:
        """Handle system error"""
        
        return "I encountered an issue processing your request. Let me try again."
        
    def _classify_user_repair(self, utterance: str) -> str:
        """Classify the type of user repair"""
        
        utterance_lower = utterance.lower()
        
        if any(w in utterance_lower for w in ['yes', 'yeah', 'correct', 'right']):
            return 'confirmation'
        elif any(w in utterance_lower for w in ['no', 'wrong', 'not']):
            return 'correction'
        elif any(w in utterance_lower for w in ['what', 'how', 'explain']):
            return 'clarification_request'
        else:
            return 'reformulation'
            
    def _is_incomprehensible(self, utterance: str) -> bool:
        """Check if utterance is incomprehensible"""
        
        # Check for very short input
        if len(utterance.split()) < 2:
            return True
            
        # Check for nonsense patterns
        if re.match(r'^[\W\d]+$', utterance):
            return True
            
        return False
        
    def _get_required_slots(self, intent: str) -> List[str]:
        """Get required slots for an intent"""
        
        slot_requirements = {
            'send_email': ['recipient', 'subject', 'body'],
            'schedule_meeting': ['participants', 'time', 'duration'],
            'set_reminder': ['task', 'time'],
            'search': ['query'],
            'open_application': ['app_name'],
            'create_file': ['filename', 'location']
        }
        
        return slot_requirements.get(intent, [])
```

---

## 9. Multi-Modal Integration

### 9.1 Voice + Visual Integration Architecture

```python
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum

class VisualInputType(Enum):
    """Types of visual input"""
    SCREEN_CAPTURE = "screen_capture"
    CAMERA_FEED = "camera_feed"
    SHARED_IMAGE = "shared_image"
    DOCUMENT = "document"
    DIAGRAM = "diagram"

@dataclass
class VisualContext:
    """Visual context from screen/camera"""
    input_type: VisualInputType
    timestamp: datetime
    image_data: Any  # PIL Image or numpy array
    description: str = ''
    detected_elements: List[Dict] = None
    active_window: str = ''
    cursor_position: tuple = None
    
    def __post_init__(self):
        if self.detected_elements is None:
            self.detected_elements = []

class MultimodalIntegrationManager:
    """
    Integrates voice and visual inputs for comprehensive understanding
    Coordinates multi-modal processing and response generation
    """
    
    def __init__(self, vision_processor, llm_client):
        self.vision_processor = vision_processor
        self.llm_client = llm_client
        
        # Visual context management
        self.visual_context_buffer: List[VisualContext] = []
        self.max_visual_buffer_size = 5
        
        # Multi-modal fusion weights
        self.voice_weight = 0.6
        self.visual_weight = 0.4
        
    async def process_multimodal_input(
        self,
        voice_input: str,
        visual_input: Optional[VisualContext],
        dialogue_state: DialogueState
    ) -> Dict:
        """Process combined voice and visual input"""
        
        results = {
            'voice_intent': None,
            'visual_understanding': None,
            'fused_understanding': None,
            'requires_visual': False
        }
        
        # Process voice input
        results['voice_intent'] = await self._process_voice(voice_input)
        
        # Check if visual context is needed
        results['requires_visual'] = self._requires_visual_context(
            voice_input, results['voice_intent']
        )
        
        # Process visual input if available and needed
        if visual_input and results['requires_visual']:
            results['visual_understanding'] = await self._process_visual(visual_input)
            
            # Fuse voice and visual understanding
            results['fused_understanding'] = await self._fuse_modalities(
                results['voice_intent'],
                results['visual_understanding'],
                voice_input
            )
        elif results['requires_visual'] and not visual_input:
            # Request visual context
            results['visual_request'] = self._generate_visual_request(voice_input)
            
        return results
        
    def _requires_visual_context(self, voice_input: str, intent: Intent) -> bool:
        """Determine if visual context is needed"""
        
        # Visual indicator words
        visual_indicators = [
            'this', 'that', 'here', 'there', 'on the screen',
            'in the window', 'the button', 'the icon', 'the menu',
            'show me', 'look at', 'what is this', 'what does this say',
            'read this', 'click on', 'select', 'highlighted'
        ]
        
        voice_lower = voice_input.lower()
        
        # Check for explicit visual references
        if any(ind in voice_lower for ind in visual_indicators):
            return True
            
        # Check intent type
        visual_intents = [
            'describe_screen', 'read_text', 'identify_element',
            'help_with_interface', 'troubleshoot', 'verify_action'
        ]
        if intent.primary_intent in visual_intents:
            return True
            
        # Check for ambiguity that visual might resolve
        if intent.confidence < 0.6:
            return True
            
        return False
        
    async def _process_visual(self, visual_context: VisualContext) -> Dict:
        """Process visual input"""
        
        # Generate description using vision model
        description = await self.vision_processor.describe_image(
            visual_context.image_data,
            detail_level='high'
        )
        
        # Detect UI elements
        elements = await self.vision_processor.detect_ui_elements(
            visual_context.image_data
        )
        
        # Extract text
        extracted_text = await self.vision_processor.extract_text(
            visual_context.image_data
        )
        
        return {
            'description': description,
            'ui_elements': elements,
            'extracted_text': extracted_text,
            'active_window': visual_context.active_window
        }
        
    async def _fuse_modalities(
        self,
        voice_intent: Intent,
        visual_understanding: Dict,
        original_utterance: str
    ) -> Dict:
        """Fuse voice intent with visual understanding"""
        
        # Create fusion prompt
        fusion_prompt = f"""Given the user's voice input and what they can see on screen, provide a unified understanding.

User said: "{original_utterance}"
Detected intent: {voice_intent.primary_intent}

Screen content:
{visual_understanding['description']}

UI elements visible:
{self._format_ui_elements(visual_understanding['ui_elements'])}

Extracted text from screen:
{visual_understanding['extracted_text']}

Provide a fused understanding in JSON format:
{{
    "refined_intent": "<specific intent considering visual context>",
    "target_element": "<UI element being referred to, if any>",
    "action_to_take": "<specific action>",
    "confidence": <0.0-1.0>,
    "requires_clarification": <true/false>,
    "clarification_question": "<question if clarification needed>"
}}"""

        response = await self.llm_client.generate(
            fusion_prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response)
        
    async def generate_multimodal_response(
        self,
        fused_understanding: Dict,
        voice_response: str,
        visual_context: VisualContext
    ) -> Dict:
        """Generate response that may include visual actions"""
        
        response = {
            'voice_response': voice_response,
            'visual_actions': [],
            'highlight_elements': [],
            'screen_capture_needed': False
        }
        
        # Determine if visual feedback is needed
        if fused_understanding.get('target_element'):
            element = fused_understanding['target_element']
            
            # Add highlight action
            response['highlight_elements'].append({
                'element': element,
                'style': 'outline',
                'color': '#00FF00',
                'duration_ms': 3000
            })
            
        # Check if action demonstration is needed
        if 'show me how' in voice_response.lower():
            response['visual_actions'].append({
                'type': 'demonstrate',
                'action': fused_understanding.get('action_to_take'),
                'target': fused_understanding.get('target_element')
            })
            
        return response
        
    def _format_ui_elements(self, elements: List[Dict]) -> str:
        """Format UI elements for LLM prompt"""
        
        formatted = []
        for elem in elements:
            formatted.append(
                f"- {elem.get('type', 'element')}: '{elem.get('text', '')}' "
                f"at position {elem.get('bbox', [])}"
            )
        return '\n'.join(formatted)
        
    def _generate_visual_request(self, voice_input: str) -> str:
        """Generate request for visual context"""
        
        templates = [
            "I can help better if I can see your screen. Could you share your screen?",
            "To assist with that, I'd like to see what you're looking at. Can you show me?",
            "Let me take a look at your screen to help with that."
        ]
        
        return templates[hash(voice_input) % len(templates)]
```

### 9.2 Visual Action Execution

```python
class VisualActionExecutor:
    """
    Executes visual actions on the user's screen
    Coordinates with system automation for UI interactions
    """
    
    def __init__(self, screen_controller, vision_processor):
        self.screen_controller = screen_controller
        self.vision_processor = vision_processor
        
    async def execute_visual_action(
        self,
        action: Dict,
        visual_context: VisualContext
    ) -> Dict:
        """Execute a visual action"""
        
        action_type = action.get('type')
        
        if action_type == 'click':
            return await self._execute_click(action, visual_context)
        elif action_type == 'type':
            return await self._execute_type(action, visual_context)
        elif action_type == 'scroll':
            return await self._execute_scroll(action, visual_context)
        elif action_type == 'highlight':
            return await self._execute_highlight(action, visual_context)
        elif action_type == 'demonstrate':
            return await self._execute_demonstration(action, visual_context)
        else:
            return {'success': False, 'error': f'Unknown action type: {action_type}'}
            
    async def _execute_click(
        self,
        action: Dict,
        visual_context: VisualContext
    ) -> Dict:
        """Execute a click action"""
        
        target = action.get('target')
        
        # Find target element in visual context
        element = self._find_element(target, visual_context)
        
        if not element:
            return {'success': False, 'error': f'Target element not found: {target}'}
            
        # Get click coordinates
        bbox = element.get('bbox')
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Execute click
        await self.screen_controller.click(center_x, center_y)
        
        return {
            'success': True,
            'action': 'click',
            'coordinates': (center_x, center_y),
            'element': element.get('text', '')
        }
        
    async def _execute_highlight(
        self,
        action: Dict,
        visual_context: VisualContext
    ) -> Dict:
        """Execute a highlight action"""
        
        elements = action.get('elements', [])
        
        for elem_info in elements:
            element = self._find_element(elem_info.get('element'), visual_context)
            
            if element:
                await self.screen_controller.highlight(
                    element['bbox'],
                    color=elem_info.get('color', '#00FF00'),
                    duration_ms=elem_info.get('duration_ms', 3000)
                )
                
        return {'success': True, 'action': 'highlight', 'elements_highlighted': len(elements)}
        
    def _find_element(self, target: str, visual_context: VisualContext) -> Optional[Dict]:
        """Find element by text or description"""
        
        target_lower = target.lower()
        
        for element in visual_context.detected_elements:
            element_text = element.get('text', '').lower()
            element_type = element.get('type', '').lower()
            
            # Match by text
            if target_lower in element_text or element_text in target_lower:
                return element
                
            # Match by type + partial text
            if target_lower in element_type:
                return element
                
        return None
```

---

## 10. Implementation Reference

### 10.1 Core System Integration

```python
class ConversationalAISystem:
    """
    Main integration class for the conversational AI system
    Coordinates all components for seamless voice interaction
    """
    
    def __init__(self, config: Dict):
        # Initialize components
        self.stt = WhisperSTT(config['stt'])
        self.tts = ElevenLabsTTS(config['tts'])
        self.llm = GPT52Client(config['llm'])
        
        # Dialogue management
        self.state_machine = DialogueStateMachine()
        self.dialogue_manager = DialogueManager(self.state_machine)
        self.turn_manager = TurnTakingManager()
        self.floor_manager = FloorManager()
        
        # Context and intent
        self.context_manager = ContextManager(config['storage'])
        self.intent_engine = IntentRecognitionEngine(self.llm, EntityExtractor())
        self.reference_resolver = ReferenceResolver(self.context_manager)
        
        # Response and repair
        self.timing_manager = ResponseTimingManager()
        self.repair_manager = ConversationRepairManager(self.llm, self.dialogue_manager)
        
        # Multi-modal
        self.vision_processor = VisionProcessor(config['vision'])
        self.multimodal_manager = MultimodalIntegrationManager(
            self.vision_processor, self.llm
        )
        
        # Barge-in
        self.barge_in_detector = BargeInDetector()
        self.barge_in_handler = BargeInHandler(self.tts, self.dialogue_manager)
        
        # State
        self.current_state: Optional[DialogueState] = None
        self.is_running = False
        
    async def start_conversation(self, user_id: str, session_id: str):
        """Start a new conversation session"""
        
        # Load context
        context = await self.context_manager.load_context(user_id, session_id)
        
        # Initialize state
        self.current_state = DialogueState(
            user_id=user_id,
            session_id=session_id,
            conversation_id=context.conversation_id
        )
        
        # Transition to listening
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'LISTENING'
        )
        
        self.is_running = True
        
        # Start processing loop
        asyncio.create_task(self._conversation_loop())
        
    async def _conversation_loop(self):
        """Main conversation processing loop"""
        
        while self.is_running:
            try:
                # Get audio input
                audio_chunk = await self.stt.get_audio_chunk()
                
                # Check for barge-in if agent is speaking
                if self.current_state.current_state == 'SPEAKING':
                    barge_in_event = await self.barge_in_detector.process_audio_chunk(
                        audio_chunk,
                        self.stt.sample_rate,
                        agent_speaking=True,
                        current_transcript=self.stt.get_partial_transcript()
                    )
                    
                    if barge_in_event:
                        self.current_state = await self.barge_in_handler.handle_barge_in(
                            barge_in_event, self.current_state
                        )
                        continue
                        
                # Transcribe audio
                transcript = await self.stt.transcribe(audio_chunk)
                
                if not transcript:
                    continue
                    
                # Update state with user input
                self.current_state.user_input_buffer = transcript
                
                # Check turn transition
                transition_signal = await self.turn_manager.analyze_turn_transition(
                    self.stt.get_audio_features(),
                    transcript,
                    self.current_state
                )
                
                if transition_signal == TurnTransitionSignal.USER_STOPPED_SPEAKING:
                    # Process user turn
                    await self._process_user_turn()
                    
            except Exception as e:
                print(f"Error in conversation loop: {e}")
                await self._handle_error(e)
                
    async def _process_user_turn(self):
        """Process a complete user turn"""
        
        # Transition to processing
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'PROCESSING'
        )
        
        # Load context
        context = await self.context_manager.load_context(
            self.current_state.user_id,
            self.current_state.session_id
        )
        
        # Resolve references
        resolved_input = await self.reference_resolver.resolve_references(
            self.current_state.user_input_buffer,
            context
        )
        
        # Recognize intent
        intent = await self.intent_engine.recognize_intent(
            resolved_input,
            context,
            self.current_state
        )
        
        self.current_state.current_intent = intent.to_dict() if hasattr(intent, 'to_dict') else intent.__dict__
        
        # Check for repair needed
        repair_trigger = await self.repair_manager.detect_need_for_repair(
            resolved_input,
            intent,
            self.current_state
        )
        
        if repair_trigger:
            repair_response = await self.repair_manager.initiate_repair(
                repair_trigger,
                resolved_input,
                intent,
                self.current_state
            )
            
            await self._deliver_response(repair_response)
            return
            
        # Generate response
        response = await self.timing_manager.generate_timed_response(
            intent,
            context,
            self.current_state,
            self._generate_response
        )
        
        # Update context
        context = await self.context_manager.update_context(
            context,
            self.current_state.user_input_buffer,
            response,
            intent=self.current_state.current_intent
        )
        
        # Deliver response
        await self._deliver_response(response)
        
    async def _generate_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> str:
        """Generate response using LLM"""
        
        # Build prompt with context
        context_str = self.context_manager.get_context_for_llm(context)
        
        prompt = f"""You are a helpful AI assistant engaged in a natural voice conversation.

Conversation Context:
{context_str}

User Intent: {intent.primary_intent}
User Message: {dialogue_state.user_input_buffer}

Respond naturally and conversationally. Keep responses concise but informative.
"""

        response = await self.llm.generate(
            prompt,
            temperature=0.7,
            max_tokens=300
        )
        
        return response
        
    async def _deliver_response(self, response: str):
        """Deliver response to user"""
        
        # Transition to speaking
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'SPEAKING'
        )
        
        self.current_state.agent_response_buffer = response
        
        # Speak response
        await self.tts.speak(response)
        
        # Transition back to listening
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'LISTENING'
        )
        
        # Increment turn counter
        self.current_state.turn_number += 1
        
    async def _handle_error(self, error: Exception):
        """Handle system error"""
        
        # Log error
        print(f"System error: {error}")
        
        # Attempt recovery
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'ERROR_RECOVERY'
        )
        
        # Deliver error message
        error_response = "I'm having a bit of trouble. Let me try again."
        await self._deliver_response(error_response)
```

### 10.2 Configuration Schema

```yaml
# conversational_ai_config.yaml

system:
  name: "OpenClaw Conversational AI"
  version: "1.0.0"
  
stt:
  provider: "openai"
  model: "whisper-3"
  language: "en"
  sample_rate: 16000
  chunk_duration_ms: 100
  
tts:
  provider: "elevenlabs"
  model: "turbo-v2.5"
  voice_id: "default"
  stability: 0.5
  similarity_boost: 0.75
  
llm:
  provider: "openai"
  model: "gpt-5.2"
  thinking_mode: "high"
  temperature: 0.7
  max_tokens: 1000
  
dialogue:
  max_turn_gap_ms: 2000
  silence_timeout_ms: 30000
  barge_in_enabled: true
  repair_enabled: true
  
context:
  storage_type: "sqlite"
  max_history_turns: 20
  summarization_threshold: 15
  
vision:
  provider: "openai"
  model: "gpt-4-vision"
  enable_screen_capture: true
  capture_interval_ms: 5000
```

---

## Appendix: State Transition Diagram

```
                              ┌─────────────────────────────────────┐
                              │                                     │
                              ▼                                     │
┌─────────┐    activation   ┌──────────┐   speech detected   ┌──────▼──────┐
│  IDLE   │────────────────▶│ LISTENING│────────────────────▶│ PROCESSING  │
└─────────┘                 └──────────┘                     └──────┬──────┘
     ▲                      ▲    ▲                                   │
     │                      │    │                                   │
     │         timeout      │    └──────────────┐                    │
     └──────────────────────┘                   │                    │
                                               │                    │
    ┌──────────┐     barge-in detected        │                    │
    │ BARGE_IN │◀─────────────────────────────┘                    │
    └────┬─────┘                                                   │
         │                                                         │
         └─────────────────────────────────────────────────────────┘
                                                                   │
         ┌─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    intent unclear    ┌──────────────┐
│ WAITING_CONFIRM │─────────────────────▶│ CLARIFYING   │
└────────┬────────┘                      └──────┬───────┘
         │                                      │
         │ confirmed                            │ clarification
         │                                      │ received
         ▼                                      ▼
┌─────────────────┐                      ┌──────────────┐
│   EXECUTING     │◀─────────────────────│   LISTENING  │
└────────┬────────┘                      └──────────────┘
         │
         │ complete
         ▼
┌─────────────────┐
│    SPEAKING     │
└─────────────────┘
```

---

*Document Version: 1.0*  
*Last Updated: 2025*  
*For: Windows 10 OpenClaw AI Agent System*
