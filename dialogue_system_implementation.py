"""
OpenClaw Conversational AI - Dialogue Management System Implementation
Windows 10 AI Agent Framework

This module provides the core implementation for the conversational AI system
including dialogue state management, turn-taking, barge-in handling, and more.
"""

import asyncio
import json
import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
from collections import deque
import sqlite3
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class Config:
    """System configuration constants"""
    
    # Timing (milliseconds)
    MIN_TURN_GAP_MS = 200
    MAX_TURN_GAP_MS = 2000
    PAUSE_THRESHOLD_MS = 500
    SILENCE_TIMEOUT_MS = 30000
    MAX_RESPONSE_LATENCY_MS = 800
    
    # Confidence thresholds
    SPEECH_END_CONFIDENCE = 0.85
    TURN_YIELD_CONFIDENCE = 0.75
    INTENT_CONFIDENCE_THRESHOLD = 0.6
    BARGE_IN_ENERGY_THRESHOLD = 0.15
    
    # Context
    MAX_HISTORY_TURNS = 20
    MAX_CONTEXT_TOKENS = 4000
    CONTEXT_WINDOW_SIZE = 10
    
    # TTS
    TTS_CHUNK_SIZE = 1024
    TTS_BUFFER_TIMEOUT_MS = 300


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ConversationPhase(Enum):
    GREETING = auto()
    TASK_ORIENTED = auto()
    INFORMATIONAL = auto()
    SOCIAL = auto()
    CLOSING = auto()
    REPAIR = auto()


class IntentCategory(Enum):
    INFORMATIONAL = "informational"
    TASK_ORIENTED = "task_oriented"
    SOCIAL = "social"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    REPAIR = "repair"
    META = "meta"


class TurnTransitionSignal(Enum):
    USER_STOPPED_SPEAKING = "user_stopped"
    USER_PAUSE_DETECTED = "user_pause"
    SENTENCE_BOUNDARY = "sentence_end"
    QUESTION_DETECTED = "question"
    COMMAND_COMPLETE = "command_complete"
    AGENT_COMPLETE = "agent_complete"
    BARGE_IN_DETECTED = "barge_in"
    TIMEOUT_REACHED = "timeout"


class RepairType(Enum):
    SELF_INITIATED_SELF_REPAIR = auto()
    SELF_INITIATED_OTHER_REPAIR = auto()
    OTHER_INITIATED_SELF_REPAIR = auto()
    OTHER_INITIATED_OTHER_REPAIR = auto()


class RepairTrigger(Enum):
    LOW_CONFIDENCE = auto()
    MISUNDERSTANDING = auto()
    INCOMPREHENSIBLE = auto()
    AMBIGUITY = auto()
    INCOMPLETE = auto()
    ERROR = auto()


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
    agent_speaking_progress: float = 0.0
    agent_intended_action: Optional[str] = None
    
    # Turn Management
    last_speaker: str = 'none'
    turn_transition_type: str = ''
    
    # Context Window
    recent_utterances: List[Dict] = field(default_factory=list)
    context_window_size: int = 10
    
    # Multi-modal State
    visual_context: Optional[Dict] = None
    screen_capture_active: bool = False
    
    # Timing
    silence_duration_ms: int = 0
    response_latency_ms: int = 0
    
    # Interruption handling
    pending_interruption: Optional[Dict] = None
    deferred_interruptions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class Intent:
    """Structured intent representation"""
    category: IntentCategory
    primary_intent: str
    confidence: float
    secondary_intents: List[Tuple[str, float]] = None
    entities: List[Dict] = None
    slots: Dict[str, Any] = None
    topic: str = ''
    urgency: int = 1
    sentiment: str = 'neutral'
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


@dataclass
class BargeInEvent:
    """Represents a detected barge-in event"""
    timestamp: float
    confidence: float
    trigger_type: str
    audio_position_ms: int
    transcript_snippet: str
    urgency_level: int


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


@dataclass
class ConversationContext:
    """Multi-layer context preservation system"""
    
    current_utterance: str = ''
    current_intent: Optional[Dict] = None
    current_entities: List[Dict] = field(default_factory=list)
    turn_start_time: datetime = field(default_factory=datetime.now)
    
    session_id: str = ''
    conversation_history: List[Dict] = field(default_factory=list)
    topic_stack: List[str] = field(default_factory=list)
    active_tasks: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    conversation_id: str = ''
    conversation_summary: str = ''
    key_points: List[str] = field(default_factory=list)
    unresolved_references: List[Dict] = field(default_factory=list)
    user_emotional_state: Dict[str, float] = field(default_factory=dict)
    
    user_id: str = ''
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_patterns: Dict[str, Any] = field(default_factory=dict)
    learned_facts: List[Dict] = field(default_factory=list)
    relationship_history: List[Dict] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)
    context_version: int = 1
    
    def to_dict(self) -> Dict:
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


# ============================================================================
# DIALOGUE STATE MACHINE
# ============================================================================

class DialogueStateMachine:
    """Hierarchical state machine for conversation management"""
    
    STATES = {
        'IDLE': {
            'transitions': ['LISTENING', 'SPEAKING', 'PROCESSING'],
            'timeout': None,
            'description': 'Waiting for user activation'
        },
        'LISTENING': {
            'transitions': ['PROCESSING', 'BARGE_IN', 'TIMEOUT'],
            'timeout': 30000,
            'description': 'Actively receiving user input'
        },
        'PROCESSING': {
            'transitions': ['SPEAKING', 'CLARIFYING', 'REPAIRING'],
            'timeout': 5000,
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


class StateTransitionError(Exception):
    """Exception for invalid state transitions"""
    pass


class StateTransitionManager:
    """Manages valid state transitions with validation and hooks"""
    
    def __init__(self):
        self.transition_hooks: Dict[Tuple[str, str], List[Callable]] = {}
        self.state_validators: Dict[str, Callable] = {}
        
    def register_transition_hook(self, from_state: str, to_state: str, hook: Callable):
        key = (from_state, to_state)
        if key not in self.transition_hooks:
            self.transition_hooks[key] = []
        self.transition_hooks[key].append(hook)
        
    def validate_transition(self, current_state: str, target_state: str, context: Dict) -> Tuple[bool, str]:
        if target_state not in DialogueStateMachine.STATES[current_state]['transitions']:
            return False, f"Invalid transition: {current_state} -> {target_state}"
            
        if target_state in self.state_validators:
            validator = self.state_validators[target_state]
            is_valid, reason = validator(context)
            if not is_valid:
                return False, reason
                
        return True, "OK"
        
    async def execute_transition(self, state: DialogueState, target_state: str, context: Dict) -> DialogueState:
        from_state = state.current_state
        
        is_valid, reason = self.validate_transition(from_state, target_state, context)
        if not is_valid:
            raise StateTransitionError(reason)
            
        key = (from_state, target_state)
        if key in self.transition_hooks:
            for hook in self.transition_hooks[key]:
                await hook(state, context)
                
        state.previous_state = from_state
        state.current_state = target_state
        state.state_timestamp = datetime.now()
        state.state_duration_ms = 0
        
        if target_state == 'SPEAKING':
            state.last_speaker = 'agent'
        elif target_state == 'LISTENING':
            state.last_speaker = 'user'
            
        return state


# ============================================================================
# DIALOGUE MANAGER
# ============================================================================

class DialogueManager:
    """Main dialogue management coordinator with LLM-powered NLU"""

    INTENT_TYPES = [
        'command', 'question', 'confirmation', 'denial',
        'greeting', 'farewell', 'clarification', 'correction',
        'informational', 'emotional', 'unknown'
    ]

    def __init__(self, state_machine: DialogueStateMachine, llm_client=None):
        self.state_machine = state_machine
        self.transition_manager = StateTransitionManager()
        self.current_state: Optional[DialogueState] = None
        self.llm_client = llm_client

    async def transition_to(self, state: DialogueState, target_state: str, context: Dict = None) -> DialogueState:
        context = context or {}
        return await self.transition_manager.execute_transition(state, target_state, context)

    def get_current_state(self) -> Optional[DialogueState]:
        return self.current_state

    def is_valid_transition(self, from_state: str, to_state: str) -> bool:
        is_valid, _ = self.transition_manager.validate_transition(from_state, to_state, {})
        return is_valid

    async def _classify_intent(self, utterance: str) -> Dict[str, Any]:
        """Classify user intent using GPT. Falls back to rule-based on failure."""
        if self.llm_client:
            try:
                prompt = (
                    f"Classify the following user utterance into one of these intents: "
                    f"{', '.join(self.INTENT_TYPES)}.\n\n"
                    f"Utterance: \"{utterance}\"\n\n"
                    f"Return JSON with fields: intent, confidence (0-1), entities (list)."
                )
                response = await self.llm_client.generate(prompt, temperature=0.2, max_tokens=150)
                import json as _json
                result = _json.loads(response)
                if 'intent' in result:
                    logger.info("LLM intent classification: %s (%.2f)",
                                result['intent'], result.get('confidence', 0.0))
                    return {
                        'intent': result['intent'],
                        'confidence': float(result.get('confidence', 0.7)),
                        'entities': result.get('entities', []),
                        'source': 'llm'
                    }
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("LLM intent classification failed: %s", e)

        # Rule-based fallback
        return self._rule_based_classify_intent(utterance)

    def _rule_based_classify_intent(self, utterance: str) -> Dict[str, Any]:
        """Simple rule-based intent classification fallback."""
        lower = utterance.lower().strip()
        if any(w in lower for w in ['hello', 'hi', 'hey', 'good morning']):
            return {'intent': 'greeting', 'confidence': 0.85, 'entities': [], 'source': 'rules'}
        if any(w in lower for w in ['bye', 'goodbye', 'see you']):
            return {'intent': 'farewell', 'confidence': 0.85, 'entities': [], 'source': 'rules'}
        if any(w in lower for w in ['yes', 'yeah', 'correct', 'right', 'sure']):
            return {'intent': 'confirmation', 'confidence': 0.8, 'entities': [], 'source': 'rules'}
        if any(w in lower for w in ['no', 'nope', 'wrong', 'not']):
            return {'intent': 'denial', 'confidence': 0.8, 'entities': [], 'source': 'rules'}
        if lower.endswith('?') or any(w in lower for w in ['what', 'how', 'why', 'when', 'where', 'who']):
            return {'intent': 'question', 'confidence': 0.7, 'entities': [], 'source': 'rules'}
        if any(w in lower for w in ['open', 'run', 'start', 'execute', 'create', 'send']):
            return {'intent': 'command', 'confidence': 0.7, 'entities': [], 'source': 'rules'}
        return {'intent': 'unknown', 'confidence': 0.3, 'entities': [], 'source': 'rules'}

    async def _generate_response(self, utterance: str, intent: Dict[str, Any],
                                  context: Optional[Dict] = None) -> str:
        """Generate a contextual response using LLM. Falls back to templates."""
        if self.llm_client:
            try:
                context_str = ""
                if context:
                    context_str = f"\nConversation Context: {json.dumps(context, default=str)[:500]}"

                prompt = (
                    f"You are a helpful AI assistant in a natural conversation.\n"
                    f"User said: \"{utterance}\"\n"
                    f"Detected intent: {intent.get('intent', 'unknown')}{context_str}\n\n"
                    f"Respond naturally and concisely."
                )
                response = await self.llm_client.generate(prompt, temperature=0.7, max_tokens=300)
                if response and response.strip():
                    logger.info("LLM response generated for intent '%s'", intent.get('intent'))
                    return response.strip()
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning("LLM response generation failed: %s", e)

        # Template-based fallback
        return self._template_response(intent)

    @staticmethod
    def _template_response(intent: Dict[str, Any]) -> str:
        """Generate a template-based response as fallback."""
        templates = {
            'greeting': "Hello! How can I help you today?",
            'farewell': "Goodbye! Have a great day.",
            'confirmation': "Got it, proceeding.",
            'denial': "Understood. What would you like instead?",
            'question': "That's a good question. Let me look into that for you.",
            'command': "I'll take care of that right away.",
            'clarification': "Could you tell me more about what you mean?",
            'correction': "I apologize for the misunderstanding. Please go ahead.",
        }
        return templates.get(intent.get('intent', ''), "I understand. How can I help with that?")


# ============================================================================
# TURN-TAKING MANAGER
# ============================================================================

class TurnTakingManager:
    """Advanced turn-taking coordination system"""
    
    def __init__(self):
        self.min_turn_gap_ms = Config.MIN_TURN_GAP_MS
        self.max_turn_gap_ms = Config.MAX_TURN_GAP_MS
        self.pause_threshold_ms = Config.PAUSE_THRESHOLD_MS
        self.silence_timeout_ms = Config.SILENCE_TIMEOUT_MS
        
        self.speech_end_confidence = Config.SPEECH_END_CONFIDENCE
        self.turn_yield_confidence = Config.TURN_YIELD_CONFIDENCE
        
        self.current_floor_holder: str = 'none'
        self.turn_queue: List[str] = []
        self.transition_callbacks: List[Callable] = []
        
    async def analyze_turn_transition(
        self,
        audio_features: Dict,
        transcript_buffer: str,
        state: DialogueState
    ) -> Optional[TurnTransitionSignal]:
        
        signals = []
        
        speech_end_score = self._detect_speech_end(audio_features)
        if speech_end_score > self.speech_end_confidence:
            signals.append((TurnTransitionSignal.USER_STOPPED_SPEAKING, speech_end_score))
            
        pause_score = self._analyze_pause_pattern(audio_features)
        if pause_score > self.turn_yield_confidence:
            signals.append((TurnTransitionSignal.USER_PAUSE_DETECTED, pause_score))
            
        text_signals = self._analyze_transcript_cues(transcript_buffer)
        signals.extend(text_signals)
        
        barge_in_score = self._detect_barge_in_pattern(audio_features, state)
        if barge_in_score > 0.7:
            signals.append((TurnTransitionSignal.BARGE_IN_DETECTED, barge_in_score))
            
        if signals:
            signals.sort(key=lambda x: x[1], reverse=True)
            return signals[0][0]
            
        return None
        
    def _detect_speech_end(self, audio_features: Dict) -> float:
        energy = audio_features.get('rms_energy', 0)
        energy_history = audio_features.get('energy_history', [])
        
        if len(energy_history) < 3:
            return 0.0
            
        recent_avg = np.mean(energy_history[-3:])
        baseline = np.mean(energy_history[:-3]) if len(energy_history) > 3 else recent_avg
        
        if baseline > 0.1 and recent_avg < baseline * 0.3:
            return 0.9
        if recent_avg < baseline * 0.5:
            return 0.6
            
        return 0.0
        
    def _analyze_pause_pattern(self, audio_features: Dict) -> float:
        silence_duration = audio_features.get('silence_duration_ms', 0)
        
        if silence_duration > self.pause_threshold_ms:
            confidence = min(silence_duration / self.max_turn_gap_ms, 1.0)
            return confidence
            
        return 0.0
        
    def _analyze_transcript_cues(self, transcript: str) -> List[Tuple[TurnTransitionSignal, float]]:
        signals = []
        
        if not transcript:
            return signals
            
        completion_markers = ['.', '?', '!', '...', 'okay', 'right', 'so']
        for marker in completion_markers:
            if transcript.rstrip().endswith(marker):
                signals.append((TurnTransitionSignal.SENTENCE_BOUNDARY, 0.8))
                break
                
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you']
        transcript_lower = transcript.lower()
        for qw in question_words:
            if transcript_lower.startswith(qw) or f' {qw}' in transcript_lower:
                signals.append((TurnTransitionSignal.QUESTION_DETECTED, 0.85))
                break
                
        command_patterns = ['please', 'thanks', 'thank you', 'now', 'go ahead', 'do it', 'execute', 'run']
        for pattern in command_patterns:
            if pattern in transcript_lower:
                signals.append((TurnTransitionSignal.COMMAND_COMPLETE, 0.75))
                break
                
        return signals
        
    def _detect_barge_in_pattern(self, audio_features: Dict, state: DialogueState) -> float:
        if state.current_state != 'SPEAKING':
            return 0.0
            
        energy = audio_features.get('rms_energy', 0)
        if energy > 0.3:
            spectral_features = audio_features.get('spectral_features', {})
            if spectral_features.get('speech_likelihood', 0) > 0.6:
                return 0.85
                
        return 0.0


# ============================================================================
# BARGE-IN DETECTOR AND HANDLER
# ============================================================================

class BargeInDetector:
    """Real-time barge-in detection for voice conversations"""
    
    def __init__(self):
        self.energy_threshold = Config.BARGE_IN_ENERGY_THRESHOLD
        self.energy_spike_ratio = 2.5
        self.min_barge_duration_ms = 150
        
        self.is_agent_speaking: bool = False
        self.agent_speech_start_ms: int = 0
        self.audio_buffer: deque = deque(maxlen=100)
        self.barge_in_history: List[BargeInEvent] = []
        
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
        
        self.is_agent_speaking = agent_speaking
        
        if not agent_speaking:
            return None
            
        features = self._extract_audio_features(audio_chunk, sample_rate)
        self.audio_buffer.append(features)
        
        detections = []
        
        energy_detection = self._detect_energy_barge_in(features)
        if energy_detection:
            detections.append(energy_detection)
            
        keyword_detection = self._detect_keyword_barge_in(current_transcript)
        if keyword_detection:
            detections.append(keyword_detection)
            
        pattern_detection = self._detect_pattern_barge_in()
        if pattern_detection:
            detections.append(pattern_detection)
            
        if detections:
            best_detection = max(detections, key=lambda x: x.confidence)
            self.barge_in_history.append(best_detection)
            return best_detection
            
        return None
        
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        speech_likelihood = 1.0 if 200 < spectral_centroid < 4000 else 0.3
        
        return {
            'rms_energy': rms,
            'zero_crossing_rate': zcr,
            'spectral_centroid': spectral_centroid,
            'speech_likelihood': speech_likelihood,
            'timestamp': datetime.now().timestamp()
        }
        
    def _detect_energy_barge_in(self, features: Dict) -> Optional[BargeInEvent]:
        if len(self.audio_buffer) < 5:
            return None
            
        current_energy = features['rms_energy']
        recent_energies = [f['rms_energy'] for f in list(self.audio_buffer)[-5:]]
        avg_recent = np.mean(recent_energies[:-1])
        
        if avg_recent > 0.01 and current_energy > avg_recent * self.energy_spike_ratio:
            if current_energy > self.energy_threshold:
                confidence = min(current_energy / (avg_recent * self.energy_spike_ratio), 1.0)
                
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
        if not transcript:
            return None
            
        transcript_lower = transcript.lower()
        
        for keyword, urgency in self.urgent_keywords.items():
            if keyword in transcript_lower:
                confidence = 0.7 + (urgency * 0.06)
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
        if len(self.audio_buffer) < 10:
            return None
            
        recent = list(self.audio_buffer)[-10:]
        energies = [f['rms_energy'] for f in recent]
        speech_likelihoods = [f['speech_likelihood'] for f in recent]
        
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


class BargeInHandler:
    """Handles barge-in events with appropriate responses"""
    
    def __init__(self, tts_controller, dialogue_manager):
        self.tts_controller = tts_controller
        self.dialogue_manager = dialogue_manager
        self.interruption_history: List[Dict] = []
        
        self.response_strategies = {
            5: self._handle_urgent_interruption,
            4: self._handle_high_interruption,
            3: self._handle_medium_interruption,
            2: self._handle_low_interruption,
            1: self._handle_minimal_interruption
        }
        
    async def handle_barge_in(self, event: BargeInEvent, current_state: DialogueState) -> DialogueState:
        self.interruption_history.append({
            'event': event,
            'state_before': current_state.to_dict(),
            'timestamp': datetime.now()
        })
        
        strategy = self.response_strategies.get(event.urgency_level, self._handle_medium_interruption)
        new_state = await strategy(event, current_state)
        
        self.interruption_history[-1]['state_after'] = new_state.to_dict()
        
        return new_state
        
    async def _handle_urgent_interruption(self, event: BargeInEvent, state: DialogueState) -> DialogueState:
        await self.tts_controller.stop_immediate()
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        await self.tts_controller.speak("Yes?", priority='urgent')
        return state
        
    async def _handle_high_interruption(self, event: BargeInEvent, state: DialogueState) -> DialogueState:
        await self.tts_controller.fade_out(duration_ms=300)
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        return state
        
    async def _handle_medium_interruption(self, event: BargeInEvent, state: DialogueState) -> DialogueState:
        pause_point = self.tts_controller.find_pause_point()
        
        if pause_point:
            await self.tts_controller.stop_at_pause(pause_point)
        else:
            await self.tts_controller.fade_out(duration_ms=500)
            
        state.agent_response_buffer = self.tts_controller.get_interrupted_text()
        state = await self.dialogue_manager.transition_to(state, 'LISTENING')
        return state
        
    async def _handle_low_interruption(self, event: BargeInEvent, state: DialogueState) -> DialogueState:
        state.pending_interruption = {
            'transcript_snippet': event.transcript_snippet,
            'timestamp': datetime.now()
        }
        await self.tts_controller.adjust_volume(0.8)
        return state
        
    async def _handle_minimal_interruption(self, event: BargeInEvent, state: DialogueState) -> DialogueState:
        state.deferred_interruptions = state.deferred_interruptions or []
        state.deferred_interruptions.append({
            'event': event,
            'timestamp': datetime.now()
        })
        return state


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class ContextManager:
    """Manages context preservation across conversation turns"""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.max_history_turns = Config.MAX_HISTORY_TURNS
        self.max_context_tokens = Config.MAX_CONTEXT_TOKENS
        self.summarization_threshold = 15
        
    async def load_context(
        self,
        user_id: str,
        session_id: str,
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        
        context_key = f"{user_id}:{session_id}"
        
        if context_key in self.active_contexts:
            return self.active_contexts[context_key]
            
        context_data = await self._load_from_storage(user_id, session_id) if self.storage else None
        
        if context_data:
            context = self._deserialize_context(context_data)
        else:
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id or self._generate_conversation_id()
            )
            context.user_profile = await self._load_user_profile(user_id) if self.storage else {}
            
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
        
        turn_record = {
            'turn_number': len(context.conversation_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'user': user_utterance,
            'agent': agent_response,
            'intent': intent,
            'entities': entities or []
        }
        
        context.conversation_history.append(turn_record)
        context.current_utterance = user_utterance
        context.current_intent = intent
        context.current_entities = entities or []
        context.last_updated = datetime.now()
        
        if intent and intent.get('topic'):
            if not context.topic_stack or context.topic_stack[-1] != intent['topic']:
                context.topic_stack.append(intent['topic'])
                context.topic_stack = context.topic_stack[-5:]
                
        if len(context.conversation_history) >= self.summarization_threshold:
            await self._summarize_context(context)
            
        if len(context.conversation_history) > self.max_history_turns:
            context.conversation_history = context.conversation_history[-self.max_history_turns:]
            
        await self._persist_context(context)
        
        return context
        
    async def _summarize_context(self, context: ConversationContext):
        turns_to_summarize = context.conversation_history[:-10]
        
        key_points = []
        for turn in turns_to_summarize:
            if turn.get('intent'):
                key_points.append(f"{turn['intent'].get('primary_intent', 'unknown')}: {turn['user'][:50]}")
                
        context.key_points.extend(key_points)
        context.key_points = context.key_points[-20:]
        context.conversation_history = context.conversation_history[-10:]
        
    def get_context_for_llm(
        self,
        context: ConversationContext,
        include_summary: bool = True,
        include_history: bool = True,
        include_profile: bool = True
    ) -> str:
        
        parts = []
        
        if include_profile and context.user_profile:
            parts.append(f"User Profile: {json.dumps(context.user_profile, default=str)}")
            
        if include_summary and context.conversation_summary:
            parts.append(f"Conversation Summary: {context.conversation_summary}")
            
        if context.key_points:
            parts.append(f"Key Points: {'; '.join(context.key_points[-10:])}")
            
        if include_history and context.conversation_history:
            history_str = self._format_history(context.conversation_history[-5:])
            parts.append(f"Recent Conversation:\n{history_str}")
            
        if context.topic_stack:
            parts.append(f"Active Topics: {' > '.join(context.topic_stack)}")
            
        if context.active_tasks:
            tasks_str = '; '.join([t.get('description', '') for t in context.active_tasks[-3:]])
            parts.append(f"Active Tasks: {tasks_str}")
            
        return '\n\n'.join(parts)
        
    def _format_history(self, turns: List[Dict]) -> str:
        lines = []
        for turn in turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['agent']}")
        return '\n'.join(lines)
        
    def _generate_conversation_id(self) -> str:
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
        
    async def _load_from_storage(self, user_id: str, session_id: str) -> Optional[Dict]:
        return None
        
    async def _load_user_profile(self, user_id: str) -> Dict:
        return {}
        
    async def _persist_context(self, context: ConversationContext):
        """Persist conversation context to the memory database."""
        try:
            import sqlite3, os, json
            db_path = os.environ.get('MEMORY_DB_PATH', './data/memory.db')
            if not os.path.exists(db_path):
                return
            conn = sqlite3.connect(db_path)
            conn.execute(
                """INSERT OR REPLACE INTO memory_entries
                   (id, type, content, source_file, importance_score)
                   VALUES (?, 'conversation_context', ?, 'dialogue_system', 0.6)""",
                (f"ctx_{getattr(context, 'session_id', id(context))}",
                 json.dumps({
                     'turns': getattr(context, 'turns', [])[-20:],
                     'metadata': getattr(context, 'metadata', {}),
                 }, default=str)),
            )
            conn.commit()
            conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"Failed to persist context: {e}")
        
    def _deserialize_context(self, data: Dict) -> ConversationContext:
        return ConversationContext(**data)


# ============================================================================
# INTENT RECOGNITION ENGINE
# ============================================================================

class IntentRecognitionEngine:
    """Multi-level intent recognition with contextual awareness"""
    
    def __init__(self, llm_client=None, entity_extractor=None):
        self.llm_client = llm_client
        self.entity_extractor = entity_extractor
        self.intent_patterns = self._load_intent_patterns()
        self.rule_confidence_threshold = 0.8
        self.ml_confidence_threshold = 0.7
        
    async def recognize_intent(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        
        processed_utterance = self._preprocess(utterance)
        
        rule_result = self._rule_based_classification(processed_utterance)
        if rule_result and rule_result.confidence >= self.rule_confidence_threshold:
            return await self._enrich_intent(rule_result, context, dialogue_state)
            
        context_result = await self._context_aware_classification(
            processed_utterance, context, dialogue_state
        )
        
        if context_result.confidence < 0.6 and self.llm_client:
            llm_result = await self._llm_classification(
                processed_utterance, context, dialogue_state
            )
            final_result = self._combine_intent_results(context_result, llm_result)
        else:
            final_result = context_result
            
        enriched_intent = await self._enrich_intent(final_result, context, dialogue_state)
        
        return enriched_intent
        
    def _preprocess(self, utterance: str) -> str:
        processed = utterance.lower().strip()
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'well']
        for filler in filler_words:
            processed = processed.replace(f' {filler} ', ' ')
        return processed
        
    def _rule_based_classification(self, utterance: str) -> Optional[Intent]:
        best_match = None
        best_confidence = 0.0
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in utterance:
                    confidence = len(pattern) / len(utterance)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_name
                        
        if best_match:
            return Intent(
                category=self._infer_category(best_match),
                primary_intent=best_match,
                confidence=min(best_confidence * 1.5, 1.0)
            )
            
        return None
        
    async def _context_aware_classification(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        
        intent_scores = {}
        
        if dialogue_state.current_state == 'CLARIFYING':
            intent_scores['clarification_response'] = 0.7
            
        if dialogue_state.current_state == 'REPAIRING':
            intent_scores['acknowledgment'] = 0.6
            intent_scores['correction'] = 0.5
            
        if context.topic_stack:
            current_topic = context.topic_stack[-1]
            topic_keywords = self._get_topic_keywords(current_topic)
            keyword_match = sum(1 for kw in topic_keywords if kw in utterance)
            if keyword_match > 0:
                intent_scores['topic_continuation'] = 0.6 + (keyword_match * 0.1)
                
        if context.active_tasks:
            intent_scores['task_continuation'] = 0.5
            
        base_intent = await self._base_classification(utterance)
        
        for intent_name, score in intent_scores.items():
            if base_intent.primary_intent == intent_name:
                base_intent.confidence = max(base_intent.confidence, score)
            else:
                base_intent.secondary_intents.append((intent_name, score))
                
        return base_intent
        
    async def _base_classification(self, utterance: str) -> Intent:
        return Intent(
            category=IntentCategory.INFORMATIONAL,
            primary_intent='unknown',
            confidence=0.5
        )
        
    async def _llm_classification(
        self,
        utterance: str,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        
        if not self.llm_client:
            return await self._base_classification(utterance)
            
        prompt = f"""Analyze user utterance and classify intent.

User: "{utterance}"
Current Topic: {context.topic_stack[-1] if context.topic_stack else 'None'}

Output JSON with: category, primary_intent, confidence, entities, urgency, sentiment"""

        try:
            response = await self.llm_client.generate(prompt, temperature=0.2)
            result = json.loads(response)
            
            return Intent(
                category=IntentCategory(result.get('category', 'informational')),
                primary_intent=result.get('primary_intent', 'unknown'),
                confidence=result.get('confidence', 0.5),
                entities=result.get('entities', []),
                urgency=result.get('urgency', 1),
                sentiment=result.get('sentiment', 'neutral')
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"LLM classification fallback: {e}")
            return await self._base_classification(utterance)
            
    def _combine_intent_results(self, context_result: Intent, llm_result: Intent) -> Intent:
        if llm_result.confidence > context_result.confidence:
            return llm_result
        return context_result
        
    async def _enrich_intent(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState
    ) -> Intent:
        
        if self.entity_extractor:
            entities = await self.entity_extractor.extract(
                dialogue_state.user_input_buffer,
                intent.primary_intent
            )
            intent.entities.extend(entities)
            
        intent.topic = self._determine_topic(intent, context)
        
        return intent
        
    def _infer_category(self, intent_name: str) -> IntentCategory:
        categories = {
            'ask': IntentCategory.INFORMATIONAL,
            'tell': IntentCategory.INFORMATIONAL,
            'execute': IntentCategory.TASK_ORIENTED,
            'create': IntentCategory.TRANSACTIONAL,
            'delete': IntentCategory.TRANSACTIONAL,
            'greeting': IntentCategory.SOCIAL,
            'farewell': IntentCategory.SOCIAL
        }
        
        for prefix, category in categories.items():
            if intent_name.startswith(prefix):
                return category
                
        return IntentCategory.INFORMATIONAL
        
    def _get_topic_keywords(self, topic: str) -> List[str]:
        return topic.lower().split()
        
    def _determine_topic(self, intent: Intent, context: ConversationContext) -> str:
        if intent.entities:
            for entity in intent.entities:
                if entity.get('type') == 'topic':
                    return entity.get('value', '')
        return context.topic_stack[-1] if context.topic_stack else 'general'
        
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        return {
            'ask_question': ['what is', 'what are', 'how do', 'how can', 'tell me about', 'explain'],
            'execute_command': ['open', 'close', 'start', 'stop', 'run', 'execute', 'create'],
            'schedule_task': ['schedule', 'remind me', 'set a reminder'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'how are you'],
            'farewell': ['goodbye', 'bye', 'see you', 'talk later'],
            'gratitude': ['thank you', 'thanks', 'appreciate'],
            'correction': ['no, i meant', 'that\'s not', 'wrong', 'i said', 'actually'],
            'clarification_request': ['what do you mean', 'i don\'t understand', 'explain'],
            'conversation_control': ['wait', 'hold on', 'pause', 'stop', 'continue'],
            'capability_check': ['can you', 'are you able', 'do you know how']
        }


# ============================================================================
# CONVERSATION REPAIR MANAGER
# ============================================================================

class ConversationRepairManager:
    """Manages conversation repair for handling misunderstandings"""
    
    def __init__(self, llm_client=None, dialogue_manager=None):
        self.llm_client = llm_client
        self.dialogue_manager = dialogue_manager
        self.confidence_threshold = Config.INTENT_CONFIDENCE_THRESHOLD
        self.repair_history: List[RepairEvent] = []
        
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
        
        if intent.confidence < self.confidence_threshold:
            return RepairTrigger.LOW_CONFIDENCE
            
        correction_signals = ['no', 'wrong', 'incorrect', 'not', "that's not", 'i meant', 'i said', 'actually', 'wait']
        if any(sig in user_utterance.lower() for sig in correction_signals):
            return RepairTrigger.MISUNDERSTANDING
            
        if self._is_incomprehensible(user_utterance):
            return RepairTrigger.INCOMPREHENSIBLE
            
        if len(intent.secondary_intents) > 0:
            best_secondary = max(intent.secondary_intents, key=lambda x: x[1])
            if best_secondary[1] > intent.confidence * 0.8:
                return RepairTrigger.AMBIGUITY
                
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
        
        strategy = self.repair_strategies.get(trigger)
        if not strategy:
            return "I'm not sure I understood that correctly. Could you rephrase?"
            
        repair_response = await strategy(user_utterance, intent, dialogue_state)
        
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
        
        if self.dialogue_manager:
            await self.dialogue_manager.transition_to(dialogue_state, 'CLARIFYING')
        
        return repair_response
        
    async def handle_user_repair(self, user_utterance: str, dialogue_state: DialogueState) -> str:
        repair_type = self._classify_user_repair(user_utterance)
        
        if self.repair_history:
            self.repair_history[-1].success = True
            
        acknowledgment = self._generate_repair_acknowledgment(repair_type)
        
        if self.dialogue_manager:
            await self.dialogue_manager.transition_to(dialogue_state, 'PROCESSING')
        
        return acknowledgment
        
    async def _handle_low_confidence(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        templates = [
            f"I understood you want to {intent.primary_intent}. Is that correct?",
            f"Let me make sure - you want me to {intent.primary_intent}?",
            f"Just to confirm, you're asking about {intent.topic}?"
        ]
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_misunderstanding(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        templates = [
            "I apologize for the confusion. Could you tell me again what you'd like?",
            "I'm sorry, I didn't get that right. Can you help me understand better?",
            "My mistake. Could you clarify what you meant?"
        ]
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_incomprehensible(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        templates = [
            "I'm having trouble understanding. Could you say that differently?",
            "I didn't quite catch that. Can you rephrase?",
            "Sorry, I didn't understand. Could you try again?"
        ]
        return templates[hash(utterance) % len(templates)]
        
    async def _handle_ambiguity(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        options = [intent.primary_intent] + [i[0] for i in intent.secondary_intents[:2]]
        option_text = ' or '.join([f"{opt}" for opt in options])
        return f"I can interpret that a few ways. Did you mean {option_text}?"
        
    async def _handle_incomplete(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        required_slots = self._get_required_slots(intent.primary_intent)
        missing = [s for s in required_slots if s not in intent.slots]
        
        if len(missing) == 1:
            return f"I'd be happy to help with that. What {missing[0]} did you have in mind?"
        else:
            missing_text = ', '.join(missing[:-1]) + f" and {missing[-1]}"
            return f"To help you with that, I'll need to know the {missing_text}."
            
    async def _handle_error(self, utterance: str, intent: Intent, state: DialogueState) -> str:
        return "I encountered an issue processing your request. Let me try again."
        
    def _classify_user_repair(self, utterance: str) -> str:
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
        if len(utterance.split()) < 2:
            return True
        if re.match(r'^[\W\d]+$', utterance):
            return True
        return False
        
    def _get_required_slots(self, intent: str) -> List[str]:
        slot_requirements = {
            'send_email': ['recipient', 'subject', 'body'],
            'schedule_meeting': ['participants', 'time', 'duration'],
            'set_reminder': ['task', 'time'],
            'search': ['query'],
            'open_application': ['app_name'],
            'create_file': ['filename', 'location']
        }
        return slot_requirements.get(intent, [])
        
    def _generate_repair_acknowledgment(self, repair_type: str) -> str:
        acknowledgments = {
            'confirmation': "Got it, thank you for confirming.",
            'correction': "Thanks for the correction.",
            'clarification_request': "Let me explain that better.",
            'reformulation': "I understand now."
        }
        return acknowledgments.get(repair_type, "I see.")


# ============================================================================
# RESPONSE TIMING MANAGER
# ============================================================================

@dataclass
class TimingConstraints:
    max_first_response_ms: int = 800
    max_full_response_ms: int = 3000
    max_streaming_chunk_ms: int = 100
    silence_before_prompt_ms: int = 5000
    max_wait_without_feedback_ms: int = 2000
    enable_adaptive_timing: bool = True
    user_patience_factor: float = 1.0


class ResponseTimingManager:
    """Manages response generation timing for natural conversation flow"""
    
    def __init__(self, constraints: TimingConstraints = None):
        self.constraints = constraints or TimingConstraints()
        self.request_start_time: Optional[datetime] = None
        self.first_token_time: Optional[datetime] = None
        self.response_stages: List[Dict] = []
        
    async def generate_timed_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        
        self.request_start_time = datetime.now()
        
        complexity = self._assess_complexity(intent, context)
        
        if complexity == 'simple':
            return await self._generate_simple_response(intent, context, dialogue_state, response_generator)
        elif complexity == 'moderate':
            return await self._generate_moderate_response(intent, context, dialogue_state, response_generator)
        else:
            return await self._generate_complex_response(intent, context, dialogue_state, response_generator)
            
    async def _generate_simple_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        
        response = await response_generator(intent, context, dialogue_state)
        self.first_token_time = datetime.now()
        
        latency_ms = self._get_elapsed_ms()
        if latency_ms > self.constraints.max_first_response_ms:
            logger.warning(f"Response latency {latency_ms}ms exceeded target {self.constraints.max_first_response_ms}ms")
            
        return response
        
    async def _generate_moderate_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        
        acknowledgment = self._generate_acknowledgment(intent)
        response_task = asyncio.create_task(response_generator(intent, context, dialogue_state))
        
        try:
            response = await asyncio.wait_for(
                response_task,
                timeout=self.constraints.max_wait_without_feedback_ms / 1000
            )
            return response
        except asyncio.TimeoutError:
            await self._deliver_acknowledgment(acknowledgment)
            response = await response_task
            return response
            
    async def _generate_complex_response(
        self,
        intent: Intent,
        context: ConversationContext,
        dialogue_state: DialogueState,
        response_generator: Callable
    ) -> str:
        
        acknowledgment = self._generate_acknowledgment(intent, detailed=True)
        await self._deliver_acknowledgment(acknowledgment)
        
        response_chunks = []
        chunk_stream = response_generator(intent, context, dialogue_state, stream=True)
        
        async for chunk in chunk_stream:
            response_chunks.append(chunk)
            if self._is_natural_break_point(chunk):
                await self._deliver_chunk(chunk)
                
        full_response = ' '.join(response_chunks)
        return full_response
        
    def _generate_acknowledgment(self, intent: Intent, detailed: bool = False) -> str:
        acknowledgments = {
            IntentCategory.TASK_ORIENTED: ["Let me do that for you.", "Working on it...", "I'll take care of that."],
            IntentCategory.INFORMATIONAL: ["Let me find that for you.", "Looking that up...", "Let me check."],
            IntentCategory.TRANSACTIONAL: ["Processing that now.", "Making the changes...", "Updating that for you."]
        }
        
        category_acks = acknowledgments.get(intent.category, ["Okay..."])
        
        if intent.urgency >= 4:
            return "Right away."
        elif detailed:
            return category_acks[0]
        else:
            return category_acks[1]
            
    def _assess_complexity(self, intent: Intent, context: ConversationContext) -> str:
        simple_intents = ['greeting', 'farewell', 'gratitude', 'acknowledgment']
        if intent.primary_intent in simple_intents:
            return 'simple'
            
        if len(context.active_tasks) > 1:
            return 'complex'
            
        if intent.category == IntentCategory.INFORMATIONAL:
            if any(kw in intent.primary_intent for kw in ['search', 'find', 'lookup']):
                return 'complex'
                
        if intent.slots and any(k in intent.slots for k in ['api_call', 'external_service']):
            return 'complex'
            
        return 'moderate'
        
    def _is_natural_break_point(self, text: str) -> bool:
        break_indicators = ['. ', '? ', '! ', '; ', ': ', ', and ', ', but ']
        return any(text.endswith(ind) for ind in break_indicators)
        
    def _get_elapsed_ms(self) -> int:
        if not self.request_start_time:
            return 0
        return int((datetime.now() - self.request_start_time).total_seconds() * 1000)
        
    async def _deliver_acknowledgment(self, acknowledgment: str):
        """Deliver an acknowledgment message via the bridge stdout protocol.

        Raises OSError if delivery fails, so callers know the ack wasn't delivered.
        """
        try:
            import sys, json
            response = {
                "jsonrpc": "2.0",
                "method": "dialogue.acknowledgment",
                "params": {"text": acknowledgment, "elapsed_ms": self._get_elapsed_ms()},
            }
            sys.stdout.write(json.dumps(response, default=str) + '\n')
            sys.stdout.flush()
        except (OSError, BrokenPipeError) as e:
            logger.warning(f"Failed to deliver acknowledgment: {e}")
            raise

    async def _deliver_chunk(self, chunk: str):
        """Deliver a streamed response chunk via the bridge stdout protocol."""
        try:
            import sys, json
            response = {
                "jsonrpc": "2.0",
                "method": "dialogue.chunk",
                "params": {"text": chunk, "elapsed_ms": self._get_elapsed_ms()},
            }
            sys.stdout.write(json.dumps(response, default=str) + '\n')
            sys.stdout.flush()
        except (OSError, BrokenPipeError) as e:
            logger.warning(f"Failed to deliver chunk: {e}")


# ============================================================================
# MAIN SYSTEM INTEGRATION
# ============================================================================

class ConversationalAISystem:
    """Main integration class for the conversational AI system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize STT - use speech_recognition library
        try:
            import speech_recognition as sr
            self.stt = sr.Recognizer()
        except ImportError:
            logger.warning("speech_recognition not installed, STT disabled")
            self.stt = None

        # Initialize TTS - use pyttsx3
        try:
            import pyttsx3
            self.tts = pyttsx3.init()
        except (ImportError, RuntimeError):
            logger.warning("pyttsx3 not available, TTS disabled")
            self.tts = None

        # Initialize LLM - use OpenAI client
        try:
            from openai_client import OpenAIClient
            self.llm = OpenAIClient()
        except (ImportError, RuntimeError):
            logger.warning("OpenAI client not available, LLM disabled")
            self.llm = None

        # Dialogue management
        self.state_machine = DialogueStateMachine()
        self.dialogue_manager = DialogueManager(self.state_machine, llm_client=self.llm)
        self.turn_manager = TurnTakingManager()

        # Context and intent
        self.context_manager = ContextManager()
        self.intent_engine = IntentRecognitionEngine()
        try:
            from context_engineering_implementation_guide import ReferenceResolver
            self.reference_resolver = ReferenceResolver()
        except ImportError:
            logger.warning("ReferenceResolver not available")
            self.reference_resolver = None

        # Response and repair
        self.timing_manager = ResponseTimingManager()
        self.repair_manager = ConversationRepairManager(dialogue_manager=self.dialogue_manager)

        # Barge-in
        self.barge_in_detector = BargeInDetector()
        self.barge_in_handler = BargeInHandler(
            tts_controller=self.tts,
            dialogue_manager=self.dialogue_manager
        ) if self.tts else None
        
        # State
        self.current_state: Optional[DialogueState] = None
        self.is_running = False
        
    async def start_conversation(self, user_id: str, session_id: str):
        """Start a new conversation session"""
        
        context = await self.context_manager.load_context(user_id, session_id)
        
        self.current_state = DialogueState(
            user_id=user_id,
            session_id=session_id,
            conversation_id=context.conversation_id
        )
        
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'LISTENING'
        )
        
        self.is_running = True
        
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response.
        Uses LLM for intent classification and response generation when available,
        falling back to rule-based/template approaches."""

        if not self.current_state:
            raise RuntimeError("Conversation not started")

        # Update state
        self.current_state.user_input_buffer = user_input
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'PROCESSING'
        )

        # Load context
        context = await self.context_manager.load_context(
            self.current_state.user_id,
            self.current_state.session_id
        )

        # Step 1: Classify intent (LLM-first, rule-based fallback)
        classified_intent = await self.dialogue_manager._classify_intent(user_input)

        # Step 2: Also run traditional intent recognition for enrichment
        intent = await self.intent_engine.recognize_intent(
            user_input,
            context,
            self.current_state
        )

        # Merge: prefer LLM classification if higher confidence
        if classified_intent.get('confidence', 0) > intent.confidence:
            self.current_state.current_intent = {
                'category': classified_intent.get('intent', 'unknown'),
                'primary_intent': classified_intent.get('intent', 'unknown'),
                'confidence': classified_intent.get('confidence', 0.5),
                'source': classified_intent.get('source', 'unknown')
            }
        else:
            self.current_state.current_intent = {
                'category': intent.category.value,
                'primary_intent': intent.primary_intent,
                'confidence': intent.confidence,
                'source': 'intent_engine'
            }

        # Check for repair
        repair_trigger = await self.repair_manager.detect_need_for_repair(
            user_input,
            intent,
            self.current_state
        )

        if repair_trigger:
            repair_response = await self.repair_manager.initiate_repair(
                repair_trigger,
                user_input,
                intent,
                self.current_state
            )
            return repair_response

        # Step 3: Generate response (LLM-first, template fallback)
        context_dict = context.to_dict() if hasattr(context, 'to_dict') else {}
        response = await self.dialogue_manager._generate_response(
            user_input,
            self.current_state.current_intent,
            context=context_dict
        )

        # Update context
        await self.context_manager.update_context(
            context,
            user_input,
            response,
            intent=self.current_state.current_intent
        )

        # Update state
        self.current_state = await self.dialogue_manager.transition_to(
            self.current_state, 'SPEAKING'
        )
        self.current_state.agent_response_buffer = response
        self.current_state.turn_number += 1

        return response
        
    def get_state(self) -> Optional[DialogueState]:
        """Get current dialogue state"""
        return self.current_state
        
    def stop(self):
        """Stop the conversation"""
        self.is_running = False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the conversational AI system"""
    
    # Initialize system
    system = ConversationalAISystem()
    
    # Start conversation
    await system.start_conversation(
        user_id="user_123",
        session_id="session_456"
    )
    
    # Simulate conversation
    user_inputs = [
        "Hello, can you help me?",
        "I need to send an email to John",
        "Actually, I meant to say Jane",
        "Thanks for your help!"
    ]
    
    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        response = await system.process_user_input(user_input)
        print(f"Agent: {response}")
        
    # Stop system
    system.stop()


if __name__ == "__main__":
    asyncio.run(main())
