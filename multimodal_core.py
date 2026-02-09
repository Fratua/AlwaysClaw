"""
Multi-Modal Voice Interface Core Implementation
Windows 10 OpenClaw-Inspired AI Agent System

This module implements the core multi-modal fusion and context sharing systems.
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum, auto
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class InteractionMode(Enum):
    """Available interaction modes for the multi-modal system"""
    VOICE_PRIMARY = "voice_primary"
    VOICE_ONLY = "voice_only"
    VISUAL_PRIMARY = "visual_primary"
    TEXT_PRIMARY = "text_primary"
    MULTI_MODAL = "multi_modal"
    HANDS_FREE = "hands_free"
    FOCUS_MODE = "focus_mode"
    PRESENTATION = "presentation"


class VoiceState(Enum):
    """States for voice interaction"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    THINKING = "thinking"
    ERROR = "error"


class VisualFeedbackState(Enum):
    """Visual feedback states"""
    IDLE = "idle"
    ACTIVE = "active"
    HIGHLIGHT = "highlight"
    TRANSITION = "transition"


class RichResponseType(Enum):
    """Types of rich multi-modal responses"""
    INFO_CARD = "info_card"
    DATA_TABLE = "data_table"
    CHART = "chart"
    IMAGE_GALLERY = "image_gallery"
    ACTION_MENU = "action_menu"
    FORM = "form"
    CONFIRMATION = "confirmation"
    SELECTION_LIST = "selection_list"
    VIDEO_PLAYER = "video_player"
    AUDIO_PLAYER = "audio_player"
    EMBEDDED_WEB = "embedded_web"
    NOTIFICATION = "notification"
    STATUS_UPDATE = "status_update"
    PROGRESS_INDICATOR = "progress_indicator"


# Constants
ALIGNMENT_WINDOW_MS = 500
SESSION_TIMEOUT_SECONDS = 1800
HEARTBEAT_INTERVAL_SECONDS = 5
MAX_HISTORY_ITEMS = 1000
DEFAULT_SYNC_TOLERANCE_MS = 100


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioData:
    """Audio data container"""
    samples: np.ndarray
    sample_rate: int = 16000
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass
class ImageData:
    """Image data container"""
    data: np.ndarray
    format: str = "RGB"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "capture"


@dataclass
class TextData:
    """Text data container"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "user"
    confidence: float = 1.0


@dataclass
class ModalInput:
    """Base class for modal inputs"""
    modality: str
    data: Union[AudioData, ImageData, TextData]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedContext:
    """Fused context from multiple modalities"""
    session_id: str
    timestamp: datetime
    voice_input: Optional[TextData] = None
    visual_input: Optional[ImageData] = None
    text_input: Optional[TextData] = None
    intent: Optional[str] = None
    confidence: float = 0.0
    entities: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncPoint:
    """Synchronization point between voice and visual"""
    voice_start_time: float
    voice_end_time: float
    visual_content_id: str
    highlight_elements: List[str] = field(default_factory=list)
    scroll_position: Optional[Tuple[int, int]] = None
    emphasis_level: int = 3


@dataclass
class VoiceResponse:
    """Voice response data"""
    text: str
    detailed_text: Optional[str] = None
    ssml: Optional[str] = None
    audio_data: Optional[bytes] = None
    duration_estimate: float = 0.0


@dataclass
class VisualResponse:
    """Visual response data"""
    card_type: Optional[RichResponseType] = None
    title: Optional[str] = None
    content: str = ""
    elements: List[Dict] = field(default_factory=list)
    layout: Optional[Dict] = None
    animations: List[Dict] = field(default_factory=list)


@dataclass
class TextResponse:
    """Text response data"""
    markdown: str = ""
    plain_text: str = ""
    html: Optional[str] = None


@dataclass
class MultiModalResponse:
    """Complete multi-modal response"""
    response_id: str = field(default_factory=lambda: hashlib.md5(
        datetime.now().isoformat().encode()).hexdigest()[:12])
    voice: Optional[VoiceResponse] = None
    visual: Optional[VisualResponse] = None
    text: Optional[TextResponse] = None
    sync_points: List[SyncPoint] = field(default_factory=list)
    priority: int = 5
    expires_at: Optional[datetime] = None


@dataclass
class UserPreferences:
    """User preferences for the system"""
    preferred_mode: InteractionMode = InteractionMode.MULTI_MODAL
    voice_volume: float = 0.8
    speech_rate: float = 1.0
    visual_scale: float = 1.0
    theme: str = "dark"
    mode_preferences: Dict[InteractionMode, float] = field(default_factory=dict)
    
    def get_mode_preference(self, mode: InteractionMode) -> float:
        return self.mode_preferences.get(mode, 1.0)


@dataclass
class AccessibilitySettings:
    """Accessibility settings"""
    high_contrast: bool = False
    large_text: bool = False
    screen_reader: bool = False
    color_blind_mode: str = "none"
    captions_enabled: bool = True
    visual_alerts: bool = False
    haptic_feedback: bool = True
    dwell_click: bool = False
    sticky_keys: bool = False
    voice_control_only: bool = False
    simplified_ui: bool = False
    extended_timeouts: bool = False
    reading_assistance: bool = False


@dataclass
class AgentMood:
    """Agent emotional/mood state"""
    valence: float = 0.0  # -1.0 to 1.0
    arousal: float = 0.5  # 0.0 to 1.0
    dominance: float = 0.5  # 0.0 to 1.0
    personality_traits: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemState:
    """Current system state"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    idle_time: float = 0.0
    keyboard_activity: float = 0.0
    mouse_activity: float = 0.0
    active_applications: List[str] = field(default_factory=list)
    pending_notifications: int = 0


@dataclass
class CrossModalContext:
    """Shared context across all modalities"""
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # User context
    user_id: str = "default"
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Conversation context
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    current_intent: Optional[str] = None
    pending_actions: List[Dict] = field(default_factory=list)
    
    # Visual context
    screen_state: Optional[Dict] = None
    active_window: Optional[str] = None
    visual_entities: List[Dict] = field(default_factory=list)
    
    # Voice context
    voice_state: VoiceState = VoiceState.IDLE
    last_utterance: str = ""
    speaking_style: str = "neutral"
    
    # System context
    system_state: SystemState = field(default_factory=SystemState)
    available_tools: List[str] = field(default_factory=list)
    agent_mood: AgentMood = field(default_factory=AgentMood)


# =============================================================================
# MULTI-MODAL INPUT FUSION ENGINE
# =============================================================================

class TemporalAligner:
    """Aligns inputs from different modalities by timestamp"""
    
    ALIGNMENT_WINDOW_MS = 500
    
    def align(self, inputs: List[ModalInput]) -> List[List[ModalInput]]:
        """Group inputs that occurred within alignment window"""
        if not inputs:
            return []
        
        # Sort by timestamp
        sorted_inputs = sorted(inputs, key=lambda x: x.timestamp)
        
        # Create alignment groups
        groups = []
        current_group = [sorted_inputs[0]]
        
        for input_item in sorted_inputs[1:]:
            time_diff = (input_item.timestamp - current_group[0].timestamp).total_seconds() * 1000
            
            if time_diff <= self.ALIGNMENT_WINDOW_MS:
                current_group.append(input_item)
            else:
                groups.append(current_group)
                current_group = [input_item]
        
        if current_group:
            groups.append(current_group)
            
        return groups


class ConfidenceCalculator:
    """Calculates confidence scores for each modality"""
    
    def calculate_voice_confidence(self, stt_result: Dict) -> float:
        """Calculate voice input confidence"""
        base_confidence = stt_result.get('confidence', 0.8)
        vad_quality = stt_result.get('vad_score', 0.8)
        audio_quality = stt_result.get('audio_quality', 0.8)
        
        confidence = (
            base_confidence * 0.5 +
            vad_quality * 0.3 +
            audio_quality * 0.2
        )
        return min(confidence, 1.0)
    
    def calculate_visual_confidence(self, visual_result: Dict) -> float:
        """Calculate visual input confidence"""
        ocr_confidence = visual_result.get('ocr_confidence', 0.8)
        detection_score = visual_result.get('detection_score', 0.8)
        
        confidence = ocr_confidence * 0.6 + detection_score * 0.4
        return min(confidence, 1.0)


class MultiModalFusionEngine:
    """
    Core fusion engine combining all input modalities.
    Implements multiple fusion strategies for different scenarios.
    """
    
    FUSION_STRATEGIES = {
        'early_fusion': 'Combine at feature level',
        'late_fusion': 'Combine at decision level',
        'hybrid_fusion': 'Multi-level combination',
        'attention_fusion': 'Attention-weighted combination'
    }
    
    def __init__(self, strategy: str = 'hybrid_fusion'):
        self.strategy = strategy
        self.temporal_aligner = TemporalAligner()
        self.confidence_calculator = ConfidenceCalculator()
        self.fusion_history: deque = deque(maxlen=100)
        
    async def fuse_inputs(
        self,
        inputs: List[ModalInput],
        session_context: CrossModalContext
    ) -> FusedContext:
        """
        Fuse multiple input modalities into unified context.
        
        Args:
            inputs: List of modal inputs with timestamps
            session_context: Current session context
            
        Returns:
            FusedContext: Unified representation of all inputs
        """
        logger.info(f"Fusing {len(inputs)} inputs using {self.strategy}")
        
        # Temporal alignment
        aligned_groups = self.temporal_aligner.align(inputs)
        
        if not aligned_groups:
            return FusedContext(session_id=session_context.session_id, timestamp=datetime.now())
        
        # Process the most recent group
        current_group = aligned_groups[-1]
        
        # Apply confidence weights
        weighted_inputs = self._apply_confidence_weights(current_group)
        
        # Perform fusion based on strategy
        if self.strategy == 'hybrid_fusion':
            fused = await self._hybrid_fusion(weighted_inputs, session_context)
        elif self.strategy == 'attention_fusion':
            fused = await self._attention_fusion(weighted_inputs, session_context)
        else:
            fused = await self._default_fusion(weighted_inputs, session_context)
        
        # Resolve intent conflicts
        resolved = self._resolve_intent_conflicts(fused)
        
        # Store in history
        self.fusion_history.append({
            'timestamp': datetime.now(),
            'inputs': len(inputs),
            'result': resolved
        })
        
        return resolved
    
    def _apply_confidence_weights(self, inputs: List[ModalInput]) -> List[ModalInput]:
        """Apply confidence weighting to inputs"""
        weighted = []
        for inp in inputs:
            if inp.modality == 'voice':
                inp.confidence = self.confidence_calculator.calculate_voice_confidence(
                    inp.metadata
                )
            elif inp.modality == 'visual':
                inp.confidence = self.confidence_calculator.calculate_visual_confidence(
                    inp.metadata
                )
            weighted.append(inp)
        return weighted
    
    async def _hybrid_fusion(
        self,
        inputs: List[ModalInput],
        context: CrossModalContext
    ) -> FusedContext:
        """Hybrid fusion combining early and late fusion"""
        fused = FusedContext(
            session_id=context.session_id,
            timestamp=datetime.now()
        )
        
        # Extract data from each modality
        for inp in inputs:
            if inp.modality == 'voice' and inp.confidence > 0.5:
                fused.voice_input = TextData(
                    content=inp.data if isinstance(inp.data, str) else str(inp.data),
                    confidence=inp.confidence
                )
            elif inp.modality == 'visual' and inp.confidence > 0.5:
                fused.visual_input = inp.data if isinstance(inp.data, ImageData) else None
            elif inp.modality == 'text':
                fused.text_input = inp.data if isinstance(inp.data, TextData) else None
        
        # Combine modalities for intent detection
        combined_text = self._combine_text_inputs(fused)
        fused.intent = await self._detect_intent(combined_text, context)
        
        # Calculate overall confidence
        confidences = [inp.confidence for inp in inputs]
        fused.confidence = np.mean(confidences) if confidences else 0.0
        
        return fused
    
    async def _attention_fusion(
        self,
        inputs: List[ModalInput],
        context: CrossModalContext
    ) -> FusedContext:
        """Attention-weighted fusion"""
        # Calculate attention weights based on context
        attention_weights = self._calculate_attention_weights(inputs, context)
        
        # Weight inputs by attention
        weighted_inputs = []
        for inp, weight in zip(inputs, attention_weights):
            inp.confidence *= weight
            weighted_inputs.append(inp)
        
        # Proceed with hybrid fusion
        return await self._hybrid_fusion(weighted_inputs, context)
    
    def _calculate_attention_weights(
        self,
        inputs: List[ModalInput],
        context: CrossModalContext
    ) -> List[float]:
        """Calculate attention weights for each modality"""
        weights = []
        
        for inp in inputs:
            weight = 1.0
            
            # Boost voice when user is actively speaking
            if inp.modality == 'voice' and context.voice_state == VoiceState.LISTENING:
                weight *= 1.2
            
            # Boost visual when showing complex content
            if inp.modality == 'visual' and context.current_intent in ['show', 'display', 'find']:
                weight *= 1.3
            
            # Boost text for explicit commands
            if inp.modality == 'text' and inp.metadata.get('is_command', False):
                weight *= 1.4
            
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights
    
    def _combine_text_inputs(self, fused: FusedContext) -> str:
        """Combine text from all modalities"""
        texts = []
        
        if fused.voice_input:
            texts.append(fused.voice_input.content)
        if fused.text_input:
            texts.append(fused.text_input.content)
        
        return " ".join(texts)
    
    async def _detect_intent(self, text: str, context: CrossModalContext) -> Optional[str]:
        """Detect user intent from combined text"""
        # This would integrate with GPT-5.2 for intent classification
        # Simplified implementation
        intent_keywords = {
            'search': ['search', 'find', 'look for', 'lookup'],
            'create': ['create', 'make', 'new', 'add'],
            'delete': ['delete', 'remove', 'trash'],
            'send': ['send', 'email', 'message', 'call'],
            'open': ['open', 'launch', 'start'],
            'close': ['close', 'exit', 'quit'],
            'help': ['help', 'assist', 'support'],
        }
        
        text_lower = text.lower()
        for intent, keywords in intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        
        return "general"
    
    def _resolve_intent_conflicts(self, fused: FusedContext) -> FusedContext:
        """Resolve conflicts between detected intents from different modalities"""
        # If high confidence in one modality, trust it
        # Otherwise, use context to disambiguate
        return fused
    
    async def _default_fusion(
        self,
        inputs: List[ModalInput],
        context: CrossModalContext
    ) -> FusedContext:
        """Default simple fusion strategy"""
        return await self._hybrid_fusion(inputs, context)


# =============================================================================
# SYNCHRONIZED OUTPUT ORCHESTRATOR
# =============================================================================

class TimingController:
    """Controls precise timing for multi-modal output"""
    
    SYNC_MODES = {
        'strict': 'All modalities must be perfectly synchronized',
        'loose': 'Allow small timing variations',
        'sequential': 'Render modalities in sequence',
        'adaptive': 'Adapt timing based on content'
    }
    
    def __init__(self, sync_mode: str = 'adaptive'):
        self.sync_mode = sync_mode
        self.word_duration_estimate = 0.4  # seconds per word
        
    def calculate_timing(self, response: MultiModalResponse) -> Dict:
        """Calculate precise timing for each output component"""
        timing_plan = {
            'voice': {'start': 0, 'duration': 0},
            'visual': {'start': 0, 'duration': 0, 'segments': []},
            'text': {'start': 0, 'duration': 0}
        }
        
        # Estimate voice duration
        if response.voice:
            word_count = len(response.voice.text.split())
            voice_duration = word_count * self.word_duration_estimate
            timing_plan['voice']['duration'] = voice_duration
        
        # Calculate visual segments
        if response.visual:
            visual_segments = self._segment_visual_content(response.visual)
            timing_plan['visual']['segments'] = visual_segments
            timing_plan['visual']['duration'] = voice_duration if response.voice else 5.0
        
        # Sync visual to voice if both present
        if response.voice and response.visual and response.sync_points:
            timing_plan = self._sync_to_voice(
                timing_plan['voice']['duration'],
                visual_segments,
                response.sync_points,
                timing_plan
            )
        
        return timing_plan
    
    def _segment_visual_content(self, visual: VisualResponse) -> List[Dict]:
        """Segment visual content for synchronization"""
        segments = []
        
        if visual.elements:
            for i, element in enumerate(visual.elements):
                segments.append({
                    'id': element.get('id', f'segment_{i}'),
                    'content': element,
                    'estimated_duration': 2.0
                })
        
        return segments
    
    def _sync_to_voice(
        self,
        voice_duration: float,
        visual_segments: List[Dict],
        sync_points: List[SyncPoint],
        timing_plan: Dict
    ) -> Dict:
        """Synchronize visual content to voice output"""
        if not sync_points:
            # Even distribution
            segment_duration = voice_duration / max(len(visual_segments), 1)
            for i, segment in enumerate(visual_segments):
                segment['start_time'] = i * segment_duration
                segment['end_time'] = (i + 1) * segment_duration
        else:
            # Use explicit sync points
            for i, (segment, sync_point) in enumerate(zip(visual_segments, sync_points)):
                segment['start_time'] = sync_point.voice_start_time
                segment['end_time'] = sync_point.voice_end_time
        
        timing_plan['visual']['segments'] = visual_segments
        return timing_plan
    
    def estimate_voice_duration(self, voice: VoiceResponse) -> float:
        """Estimate voice duration from text"""
        if voice.duration_estimate > 0:
            return voice.duration_estimate
        
        word_count = len(voice.text.split())
        return word_count * self.word_duration_estimate


class VoiceRenderer:
    """Renders voice output"""
    
    def __init__(self):
        self.current_audio = None
        self.is_playing = False
        
    async def render(self, voice: VoiceResponse, timing: Dict) -> None:
        """Render voice output with timing"""
        logger.info(f"Rendering voice: {voice.text[:50]}...")
        
        # Wait for start time
        if timing['start'] > 0:
            await asyncio.sleep(timing['start'])
        
        # Generate or play audio
        if voice.audio_data:
            await self._play_audio(voice.audio_data)
        else:
            # Would call TTS engine here
            await self._synthesize_and_play(voice)
    
    async def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data"""
        self.is_playing = True
        # Implementation would use Windows audio APIs
        logger.info("Playing audio...")
        await asyncio.sleep(2.0)  # Placeholder
        self.is_playing = False
    
    async def _synthesize_and_play(self, voice: VoiceResponse) -> None:
        """Synthesize and play voice"""
        # Would integrate with ElevenLabs/Azure TTS
        logger.info("Synthesizing speech...")
        await asyncio.sleep(1.0)  # Placeholder


class VisualRenderer:
    """Renders visual output"""
    
    def __init__(self):
        self.current_display = None
        
    async def render(self, visual: VisualResponse, timing: Dict) -> None:
        """Render visual output with timing"""
        logger.info(f"Rendering visual: {visual.title or 'untitled'}")
        
        # Wait for start time
        if timing['start'] > 0:
            await asyncio.sleep(timing['start'])
        
        # Render each segment with timing
        for segment in timing.get('segments', []):
            await self._render_segment(segment)
            
            # Wait until next segment
            if 'end_time' in segment:
                wait_time = segment['end_time'] - segment.get('start_time', 0)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
    
    async def _render_segment(self, segment: Dict) -> None:
        """Render a visual segment"""
        logger.info(f"Rendering segment: {segment.get('id')}")
        # Implementation would use Windows UI APIs


class TextRenderer:
    """Renders text output"""
    
    async def render(self, text: TextResponse, timing: Dict) -> None:
        """Render text output with timing"""
        logger.info(f"Rendering text: {text.plain_text[:50]}...")
        # Implementation would update text display


class SynchronizedOutputOrchestrator:
    """
    Orchestrates synchronized multi-modal output.
    Ensures voice, visual, and text outputs are properly coordinated.
    """
    
    def __init__(self):
        self.timing_controller = TimingController()
        self.voice_renderer = VoiceRenderer()
        self.visual_renderer = VisualRenderer()
        self.text_renderer = TextRenderer()
        self.active_renders: Dict[str, asyncio.Task] = {}
        
    async def render_synchronized(self, response: MultiModalResponse) -> Dict:
        """
        Render response across all modalities with precise timing.
        
        Args:
            response: Multi-modal response to render
            
        Returns:
            Dict with render status for each modality
        """
        logger.info(f"Rendering synchronized response: {response.response_id}")
        
        # Calculate timing for each component
        timing_plan = self.timing_controller.calculate_timing(response)
        
        # Create render tasks
        tasks = []
        
        if response.voice:
            voice_task = asyncio.create_task(
                self.voice_renderer.render(response.voice, timing_plan['voice'])
            )
            tasks.append(('voice', voice_task))
        
        if response.visual:
            visual_task = asyncio.create_task(
                self.visual_renderer.render(response.visual, timing_plan['visual'])
            )
            tasks.append(('visual', visual_task))
        
        if response.text:
            text_task = asyncio.create_task(
                self.text_renderer.render(response.text, timing_plan['text'])
            )
            tasks.append(('text', text_task))
        
        # Execute with coordination
        results = {}
        for modality, task in tasks:
            try:
                await task
                results[modality] = {'status': 'success'}
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Error rendering {modality}: {e}")
                results[modality] = {'status': 'error', 'error': str(e)}
        
        return results
    
    async def cancel_active_renders(self) -> None:
        """Cancel all active render operations"""
        for response_id, task in self.active_renders.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled render: {response_id}")
        
        self.active_renders.clear()


# =============================================================================
# MODE COORDINATOR
# =============================================================================

class ModeCoordinator:
    """
    Coordinates mode switching based on context.
    Automatically determines optimal interaction mode.
    """
    
    def __init__(self):
        self.current_mode = InteractionMode.MULTI_MODAL
        self.mode_history: deque = deque(maxlen=50)
        self.user_preferences = UserPreferences()
        self.mode_scores: Dict[InteractionMode, float] = {}
        
    def determine_optimal_mode(self, context: CrossModalContext) -> InteractionMode:
        """
        Determine the best interaction mode for current context.
        
        Args:
            context: Current cross-modal context
            
        Returns:
            InteractionMode: Optimal mode for current context
        """
        scores = {}
        
        # Score each mode based on context
        for mode in InteractionMode:
            scores[mode] = self._score_mode(mode, context)
        
        # Apply user preferences
        for mode in InteractionMode:
            preference_boost = self.user_preferences.get_mode_preference(mode)
            scores[mode] *= preference_boost
        
        # Store scores
        self.mode_scores = scores
        
        # Select highest scoring mode
        optimal_mode = max(scores, key=scores.get)
        
        # Check if mode change is warranted
        if optimal_mode != self.current_mode:
            if self._should_switch_mode(optimal_mode, scores[optimal_mode]):
                self._switch_mode(optimal_mode)
        
        return self.current_mode
    
    def _score_mode(self, mode: InteractionMode, context: CrossModalContext) -> float:
        """Score a mode based on current context"""
        score = 0.5  # Base score
        
        if mode == InteractionMode.VOICE_PRIMARY:
            # Good when user is speaking
            if context.voice_state == VoiceState.LISTENING:
                score += 0.8
            # Good when hands are busy
            if context.system_state.keyboard_activity < 0.2:
                score += 0.5
            # Good for quick commands
            if context.current_intent in ['search', 'open', 'close']:
                score += 0.3
                
        elif mode == InteractionMode.VOICE_ONLY:
            # Good when no display needed
            if context.current_intent in ['call', 'send']:
                score += 0.7
            # Good for accessibility
            if context.system_state.idle_time > 60:
                score += 0.4
                
        elif mode == InteractionMode.VISUAL_PRIMARY:
            # Good when showing complex information
            if context.current_intent in ['show', 'display', 'analyze']:
                score += 0.9
            # Good when user is reading
            if context.system_state.idle_time > 10:
                score += 0.3
                
        elif mode == InteractionMode.TEXT_PRIMARY:
            # Good for detailed input
            if context.current_intent in ['write', 'compose', 'edit']:
                score += 0.8
            # Good when voice is inappropriate
            if context.system_state.active_applications and 'meeting' in str(context.system_state.active_applications).lower():
                score += 0.7
                
        elif mode == InteractionMode.HANDS_FREE:
            # Good when user is away from keyboard
            if context.system_state.idle_time > 30:
                score += 0.8
            # Good for notifications
            if context.system_state.pending_notifications > 0:
                score += 0.6
                
        elif mode == InteractionMode.FOCUS_MODE:
            # Good for concentrated work
            if context.current_intent in ['work', 'focus', 'study']:
                score += 0.9
            # Good when minimizing distractions
            if len(context.conversation_history) > 10:
                score += 0.3
                
        elif mode == InteractionMode.PRESENTATION:
            # Good for presenting information
            if context.current_intent in ['present', 'demo', 'showcase']:
                score += 0.9
            # Good with external display
            if context.screen_state and context.screen_state.get('display_count', 1) > 1:
                score += 0.4
        
        return score
    
    def _should_switch_mode(self, new_mode: InteractionMode, new_score: float) -> bool:
        """Determine if mode switch should occur"""
        current_score = self.mode_scores.get(self.current_mode, 0)
        
        # Require significant improvement to switch
        score_threshold = 0.2
        
        # Don't switch too frequently
        if self.mode_history:
            last_switch_time = self.mode_history[-1].get('timestamp')
            if last_switch_time:
                time_since_switch = (datetime.now() - last_switch_time).seconds
                if time_since_switch < 5:  # Minimum 5 seconds between switches
                    return False
        
        return (new_score - current_score) > score_threshold
    
    def _switch_mode(self, new_mode: InteractionMode) -> None:
        """Switch to new interaction mode"""
        logger.info(f"Switching mode: {self.current_mode.value} -> {new_mode.value}")
        
        self.mode_history.append({
            'from': self.current_mode,
            'to': new_mode,
            'timestamp': datetime.now()
        })
        
        self.current_mode = new_mode
    
    def set_user_preference(self, mode: InteractionMode, preference: float) -> None:
        """Set user preference for a mode"""
        self.user_preferences.mode_preferences[mode] = preference
    
    def get_mode_statistics(self) -> Dict:
        """Get statistics about mode usage"""
        stats = {mode: {'count': 0, 'total_time': 0} for mode in InteractionMode}
        
        for entry in self.mode_history:
            mode = entry.get('to')
            if mode:
                stats[mode]['count'] += 1
        
        return stats


# =============================================================================
# VISUAL FEEDBACK SYSTEM
# =============================================================================

class VisualFeedbackSystem:
    """
    Provides visual feedback for voice interactions.
    Includes waveforms, indicators, and state visualizations.
    """
    
    STATE_COLORS = {
        VoiceState.IDLE: '#888888',
        VoiceState.LISTENING: '#00FF88',
        VoiceState.PROCESSING: '#4488FF',
        VoiceState.THINKING: '#FFAA00',
        VoiceState.SPEAKING: '#00FFFF',
        VoiceState.ERROR: '#FF4444'
    }
    
    def __init__(self):
        self.current_state = VoiceState.IDLE
        self.audio_level = 0.0
        self.visual_elements: List[Dict] = []
        
    def update_state(self, state: VoiceState, audio_level: float = 0.0) -> Dict:
        """
        Update visual feedback state.
        
        Args:
            state: New voice state
            audio_level: Current audio level (0.0 - 1.0)
            
        Returns:
            Dict with visual elements to render
        """
        self.current_state = state
        self.audio_level = audio_level
        
        if state == VoiceState.LISTENING:
            return self._render_listening_indicator(audio_level)
        elif state == VoiceState.PROCESSING:
            return self._render_processing_indicator()
        elif state == VoiceState.THINKING:
            return self._render_thinking_indicator()
        elif state == VoiceState.SPEAKING:
            return self._render_speaking_indicator(audio_level)
        elif state == VoiceState.ERROR:
            return self._render_error_indicator()
        else:
            return self._render_idle_indicator()
    
    def _render_listening_indicator(self, audio_level: float) -> Dict:
        """Render listening indicator with audio reactivity"""
        color = self.STATE_COLORS[VoiceState.LISTENING]
        
        # Generate waveform based on audio level
        waveform = self._generate_waveform(audio_level)
        
        return {
            'type': 'listening_indicator',
            'elements': [
                {
                    'type': 'waveform',
                    'data': waveform,
                    'color': color,
                    'height': 50 + (audio_level * 50)
                },
                {
                    'type': 'glow',
                    'intensity': 0.5 + (audio_level * 0.5),
                    'color': color
                },
                {
                    'type': 'text',
                    'content': 'Listening...',
                    'position': 'bottom',
                    'color': color
                }
            ],
            'animation': 'pulse',
            'frame_rate': 60
        }
    
    def _render_processing_indicator(self) -> Dict:
        """Render processing indicator"""
        color = self.STATE_COLORS[VoiceState.PROCESSING]
        
        return {
            'type': 'processing_indicator',
            'elements': [
                {
                    'type': 'spinner',
                    'size': 40,
                    'color': color,
                    'speed': 'normal'
                },
                {
                    'type': 'text',
                    'content': 'Processing...',
                    'position': 'bottom',
                    'color': color
                }
            ],
            'animation': 'continuous'
        }
    
    def _render_thinking_indicator(self) -> Dict:
        """Render thinking indicator"""
        color = self.STATE_COLORS[VoiceState.THINKING]
        
        return {
            'type': 'thinking_indicator',
            'elements': [
                {
                    'type': 'neural_network',
                    'nodes': 30,
                    'connections': 60,
                    'activity_level': 0.7,
                    'color': color
                },
                {
                    'type': 'progress_ring',
                    'progress': None,  # Indeterminate
                    'color': color
                },
                {
                    'type': 'text',
                    'content': 'Thinking...',
                    'position': 'bottom',
                    'color': color
                }
            ],
            'background_effect': 'subtle_glow'
        }
    
    def _render_speaking_indicator(self, audio_level: float) -> Dict:
        """Render speaking indicator"""
        color = self.STATE_COLORS[VoiceState.SPEAKING]
        
        return {
            'type': 'speaking_indicator',
            'elements': [
                {
                    'type': 'voice_wave',
                    'amplitude': audio_level,
                    'color': color,
                    'frequency': 2.0
                },
                {
                    'type': 'text',
                    'content': 'Speaking...',
                    'position': 'bottom',
                    'color': color
                }
            ],
            'animation': 'speech_synced'
        }
    
    def _render_idle_indicator(self) -> Dict:
        """Render idle indicator"""
        color = self.STATE_COLORS[VoiceState.IDLE]
        
        return {
            'type': 'idle_indicator',
            'elements': [
                {
                    'type': 'orb',
                    'size': 20,
                    'color': color,
                    'opacity': 0.5
                }
            ],
            'animation': 'slow_pulse'
        }
    
    def _render_error_indicator(self) -> Dict:
        """Render error indicator"""
        color = self.STATE_COLORS[VoiceState.ERROR]
        
        return {
            'type': 'error_indicator',
            'elements': [
                {
                    'type': 'icon',
                    'name': 'error',
                    'size': 40,
                    'color': color
                },
                {
                    'type': 'text',
                    'content': 'Error occurred',
                    'position': 'bottom',
                    'color': color
                }
            ],
            'animation': 'shake'
        }
    
    def _generate_waveform(self, audio_level: float, points: int = 50) -> List[float]:
        """Generate waveform data based on audio level"""
        import math
        
        waveform = []
        for i in range(points):
            # Create a wave pattern
            t = i / points
            wave = math.sin(t * math.pi * 4) * audio_level
            # Add some randomness
            noise = (math.random() - 0.5) * 0.1 if hasattr(math, 'random') else 0
            waveform.append(max(0, min(1, wave + noise + 0.5)))
        
        return waveform


# =============================================================================
# SESSION MANAGER
# =============================================================================

class HeartbeatMonitor:
    """Monitors session heartbeat"""
    
    def __init__(self):
        self.last_ping = datetime.now()
        self.ping_count = 0
        
    def ping(self) -> None:
        """Record heartbeat ping"""
        self.last_ping = datetime.now()
        self.ping_count += 1
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Check if heartbeat is still active"""
        elapsed = (datetime.now() - self.last_ping).seconds
        return elapsed < timeout_seconds


class AgentLoop:
    """Individual agent loop"""
    
    def __init__(self, loop_id: str, config: Dict, handler: Callable):
        self.loop_id = loop_id
        self.config = config
        self.handler = handler
        self.last_run = datetime.min
        self.run_count = 0
        
    def should_run(self) -> bool:
        """Check if loop should execute"""
        trigger = self.config.get('trigger', 'continuous')
        
        if trigger == 'continuous':
            interval = self.config.get('interval', 60)
            elapsed = (datetime.now() - self.last_run).seconds
            return elapsed >= interval
        elif trigger == 'on_demand':
            return False  # Run only when explicitly triggered
        elif trigger == 'scheduled':
            # Check if scheduled time
            return False  # Implementation depends on scheduling
        
        return False
    
    async def execute(self, session: 'MultiModalSession') -> None:
        """Execute the agent loop"""
        try:
            await self.handler(session)
            self.last_run = datetime.now()
            self.run_count += 1
        except (OSError, RuntimeError, PermissionError) as e:
            logger.error(f"Error in agent loop {self.loop_id}: {e}")


@dataclass
class MultiModalSession:
    """Complete session state for multi-modal interaction"""
    session_id: str
    user_id: str = "default"
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Context
    context: CrossModalContext = field(default_factory=lambda: CrossModalContext(session_id=""))
    
    # State
    current_mode: InteractionMode = InteractionMode.MULTI_MODAL
    interaction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    pending_responses: List[MultiModalResponse] = field(default_factory=list)
    
    # Configuration
    preferences: UserPreferences = field(default_factory=UserPreferences)
    accessibility_settings: AccessibilitySettings = field(default_factory=AccessibilitySettings)
    
    # System
    heartbeat: HeartbeatMonitor = field(default_factory=HeartbeatMonitor)
    agent_loops: List[AgentLoop] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.context.session_id:
            self.context.session_id = self.session_id
    
    def is_active(self) -> bool:
        """Check if session is still active"""
        idle_time = (datetime.now() - self.last_activity).seconds
        return idle_time < SESSION_TIMEOUT_SECONDS
    
    def touch(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


class SessionManager:
    """
    Manages multi-modal session lifecycle.
    Handles creation, monitoring, and cleanup of sessions.
    """
    
    def __init__(self):
        self.sessions: Dict[str, MultiModalSession] = {}
        self.fusion_engine = MultiModalFusionEngine()
        self.output_orchestrator = SynchronizedOutputOrchestrator()
        self.mode_coordinator = ModeCoordinator()
        self.visual_feedback = VisualFeedbackSystem()
        
    def create_session(self, user_id: str = "default") -> MultiModalSession:
        """Create new multi-modal session"""
        session_id = self._generate_session_id()
        
        session = MultiModalSession(
            session_id=session_id,
            user_id=user_id,
            context=CrossModalContext(session_id=session_id),
            agent_loops=self._initialize_agent_loops()
        )
        
        # Store session
        self.sessions[session_id] = session
        
        # Start heartbeat
        asyncio.create_task(self._run_heartbeat(session))
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[MultiModalSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    async def process_input(
        self,
        session_id: str,
        inputs: List[ModalInput]
    ) -> MultiModalResponse:
        """
        Process multi-modal input and generate response.
        
        Args:
            session_id: Session identifier
            inputs: List of modal inputs
            
        Returns:
            MultiModalResponse: Synchronized multi-modal response
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Update activity
        session.touch()
        
        # Update visual feedback
        for inp in inputs:
            if inp.modality == 'voice':
                feedback = self.visual_feedback.update_state(
                    VoiceState.LISTENING,
                    inp.metadata.get('audio_level', 0.5)
                )
                # Would render feedback here
        
        # Fuse inputs
        fused_context = await self.fusion_engine.fuse_inputs(inputs, session.context)
        
        # Update session context
        session.context.current_intent = fused_context.intent
        
        # Determine optimal mode
        optimal_mode = self.mode_coordinator.determine_optimal_mode(session.context)
        session.current_mode = optimal_mode
        
        # Generate response (would integrate with GPT-5.2)
        response = await self._generate_response(fused_context, session)
        
        # Render synchronized output
        await self.output_orchestrator.render_synchronized(response)
        
        # Store interaction
        session.interaction_history.append({
            'timestamp': datetime.now(),
            'inputs': inputs,
            'response': response
        })
        
        return response
    
    async def _generate_response(
        self,
        context: FusedContext,
        session: MultiModalSession
    ) -> MultiModalResponse:
        """Generate multi-modal response"""
        # This would integrate with GPT-5.2
        # Simplified implementation
        
        voice_text = f"I understood: {context.intent}"
        
        return MultiModalResponse(
            voice=VoiceResponse(text=voice_text),
            visual=VisualResponse(
                title="Response",
                content=f"Intent detected: {context.intent}"
            ),
            text=TextResponse(
                markdown=f"**Intent:** {context.intent}",
                plain_text=f"Intent: {context.intent}"
            )
        )
    
    async def _run_heartbeat(self, session: MultiModalSession) -> None:
        """Continuous heartbeat for session monitoring"""
        while session.is_active():
            # Update heartbeat
            session.heartbeat.ping()
            
            # Check agent loops
            for loop in session.agent_loops:
                if loop.should_run():
                    await loop.execute(session)
            
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        
        # Session expired, clean up
        logger.info(f"Session expired: {session.session_id}")
        del self.sessions[session.session_id]
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def _initialize_agent_loops(self) -> List[AgentLoop]:
        """Initialize agent loops"""
        loops = []
        
        loop_configs = {
            'conversation_loop': {
                'description': 'Main conversation handler',
                'trigger': 'user_input',
                'priority': 1
            },
            'context_loop': {
                'description': 'Context maintenance',
                'trigger': 'continuous',
                'interval': 5,
                'priority': 2
            },
            'notification_loop': {
                'description': 'Check notifications',
                'trigger': 'continuous',
                'interval': 10,
                'priority': 3
            },
            'voice_loop': {
                'description': 'Voice processing',
                'trigger': 'voice_activity',
                'priority': 1
            },
            'visual_loop': {
                'description': 'Visual context',
                'trigger': 'continuous',
                'interval': 1,
                'priority': 6
            },
            'system_loop': {
                'description': 'System monitoring',
                'trigger': 'continuous',
                'interval': 5,
                'priority': 7
            },
            'memory_loop': {
                'description': 'Memory management',
                'trigger': 'continuous',
                'interval': 60,
                'priority': 9
            },
            'safety_loop': {
                'description': 'Safety monitoring',
                'trigger': 'continuous',
                'interval': 5,
                'priority': 0
            }
        }
        
        for loop_id, config in loop_configs.items():
            loop = AgentLoop(
                loop_id=loop_id,
                config=config,
                handler=self._get_loop_handler(loop_id)
            )
            loops.append(loop)
        
        return loops
    
    def _get_loop_handler(self, loop_id: str) -> Callable:
        """Get handler for agent loop"""
        async def default_handler(session: MultiModalSession) -> None:
            logger.debug(f"Running agent loop: {loop_id}")
        
        return default_handler


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the multi-modal system"""
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Create a new session
    session = session_manager.create_session(user_id="test_user")
    print(f"Created session: {session.session_id}")
    
    # Create sample inputs
    voice_input = ModalInput(
        modality="voice",
        data=TextData(content="What's the weather today?"),
        confidence=0.9,
        metadata={'vad_score': 0.85, 'audio_quality': 0.9}
    )
    
    # Process input
    response = await session_manager.process_input(
        session.session_id,
        [voice_input]
    )
    
    print(f"Response ID: {response.response_id}")
    if response.voice:
        print(f"Voice: {response.voice.text}")
    if response.visual:
        print(f"Visual: {response.visual.title}")
    
    # Keep session alive for a bit
    await asyncio.sleep(10)
    
    print("Session complete")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
