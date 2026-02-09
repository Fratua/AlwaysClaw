"""
Visual Feedback System for Multi-Modal Voice Interface
Windows 10 OpenClaw-Inspired AI Agent System

This module implements visual feedback components for voice interactions,
including waveforms, indicators, animations, and response highlighting.
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VoiceState(Enum):
    """Voice interaction states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    SUCCESS = "success"


class AnimationType(Enum):
    """Animation types for visual elements"""
    PULSE = "pulse"
    FADE = "fade"
    SLIDE = "slide"
    BOUNCE = "bounce"
    SHAKE = "shake"
    SPIN = "spin"
    GLOW = "glow"
    WAVE = "wave"


class VisualElementType(Enum):
    """Types of visual elements"""
    WAVEFORM = "waveform"
    ORB = "orb"
    SPINNER = "spinner"
    PROGRESS_RING = "progress_ring"
    TEXT = "text"
    ICON = "icon"
    GLOW = "glow"
    NEURAL_NETWORK = "neural_network"
    VOICE_WAVE = "voice_wave"
    CARD = "card"
    HIGHLIGHT = "highlight"


# =============================================================================
# COLOR CONSTANTS
# =============================================================================

STATE_COLORS = {
    VoiceState.IDLE: '#888888',
    VoiceState.LISTENING: '#00FF88',
    VoiceState.PROCESSING: '#4488FF',
    VoiceState.THINKING: '#FFAA00',
    VoiceState.SPEAKING: '#00FFFF',
    VoiceState.ERROR: '#FF4444',
    VoiceState.SUCCESS: '#44FF44'
}

THEME_PRESETS = {
    'default': {
        'background': '#1E1E1E',
        'text': '#FFFFFF',
        'accent': '#007ACC',
        'success': '#4EC9B0',
        'warning': '#CE9178',
        'error': '#F44747',
        'info': '#569CD6'
    },
    'high_contrast': {
        'background': '#000000',
        'text': '#FFFFFF',
        'accent': '#FFFF00',
        'success': '#00FF00',
        'warning': '#FFFF00',
        'error': '#FF0000',
        'info': '#00FFFF'
    },
    'high_contrast_white': {
        'background': '#FFFFFF',
        'text': '#000000',
        'accent': '#0000FF',
        'success': '#008000',
        'warning': '#FF8C00',
        'error': '#FF0000',
        'info': '#000080'
    },
    'ocean': {
        'background': '#001F3F',
        'text': '#E0F7FA',
        'accent': '#00BCD4',
        'success': '#00E676',
        'warning': '#FFD54F',
        'error': '#FF5252',
        'info': '#40C4FF'
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VisualElement:
    """Base visual element"""
    element_type: VisualElementType
    id: str = field(default_factory=lambda: f"elem_{datetime.now().timestamp()}")
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (100, 100)
    color: str = '#FFFFFF'
    opacity: float = 1.0
    z_index: int = 0
    animation: Optional[AnimationType] = None
    animation_duration: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaveformElement(VisualElement):
    """Audio waveform visualization"""
    data: List[float] = field(default_factory=list)
    bar_count: int = 32
    bar_width: float = 4.0
    bar_spacing: float = 2.0
    min_height: float = 5.0
    max_height: float = 100.0
    smoothing: float = 0.3
    
    def __post_init__(self):
        self.element_type = VisualElementType.WAVEFORM
        if not self.data:
            self.data = [0.5] * self.bar_count


@dataclass
class OrbElement(VisualElement):
    """Pulsing orb indicator"""
    radius: float = 30.0
    pulse_amplitude: float = 0.3
    pulse_frequency: float = 1.0
    glow_radius: float = 50.0
    glow_intensity: float = 0.5
    
    def __post_init__(self):
        self.element_type = VisualElementType.ORB


@dataclass
class SpinnerElement(VisualElement):
    """Rotating spinner"""
    segments: int = 8
    segment_length: float = 15.0
    speed: float = 1.0  # rotations per second
    thickness: float = 4.0
    
    def __post_init__(self):
        self.element_type = VisualElementType.SPINNER


@dataclass
class NeuralNetworkElement(VisualElement):
    """Neural network animation"""
    nodes: int = 30
    connections: int = 60
    activity_level: float = 0.7
    node_size: float = 4.0
    connection_opacity: float = 0.3
    pulse_speed: float = 1.0
    
    def __post_init__(self):
        self.element_type = VisualElementType.NEURAL_NETWORK
        self.node_positions = self._generate_node_positions()
        self.connection_pairs = self._generate_connections()
    
    def _generate_node_positions(self) -> List[Tuple[float, float]]:
        """Generate random node positions"""
        positions = []
        for _ in range(self.nodes):
            x = random.random() * self.size[0]
            y = random.random() * self.size[1]
            positions.append((x, y))
        return positions
    
    def _generate_connections(self) -> List[Tuple[int, int]]:
        """Generate random connections between nodes"""
        connections = []
        for _ in range(self.connections):
            a = random.randint(0, self.nodes - 1)
            b = random.randint(0, self.nodes - 1)
            if a != b:
                connections.append((a, b))
        return connections


@dataclass
class VoiceWaveElement(VisualElement):
    """Voice wave visualization"""
    amplitude: float = 0.5
    frequency: float = 2.0
    wave_count: int = 3
    line_width: float = 2.0
    
    def __post_init__(self):
        self.element_type = VisualElementType.VOICE_WAVE


@dataclass
class HighlightRegion:
    """Screen highlight region"""
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    style: str = 'glow'  # glow, outline, fill
    color: str = '#4488FF'
    duration_ms: int = 2000
    pulse: bool = True
    animated: bool = True
    opacity: float = 0.5


@dataclass
class VisualFrame:
    """Complete visual frame"""
    elements: List[VisualElement] = field(default_factory=list)
    background_color: str = '#1E1E1E'
    background_effect: Optional[str] = None
    frame_rate: int = 60
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# WAVEFORM GENERATORS
# =============================================================================

class WaveformGenerator:
    """Generates various waveform patterns"""
    
    @staticmethod
    def generate_sine_wave(
        points: int,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0
    ) -> List[float]:
        """Generate sine wave pattern"""
        wave = []
        for i in range(points):
            t = i / points
            value = amplitude * math.sin(2 * math.pi * frequency * t + phase)
            wave.append((value + 1) / 2)  # Normalize to 0-1
        return wave
    
    @staticmethod
    def generate_audio_reactive_waveform(
        audio_level: float,
        points: int = 32,
        history: Optional[List[float]] = None
    ) -> List[float]:
        """Generate audio-reactive waveform"""
        waveform = []
        
        for i in range(points):
            # Base wave pattern
            t = i / points
            base = math.sin(t * math.pi * 4) * audio_level
            
            # Add harmonics
            harmonic1 = math.sin(t * math.pi * 8) * audio_level * 0.5
            harmonic2 = math.sin(t * math.pi * 12) * audio_level * 0.25
            
            # Add noise
            noise = (random.random() - 0.5) * 0.1 * audio_level
            
            # Combine
            value = (base + harmonic1 + harmonic2 + noise + 1) / 2
            value = max(0, min(1, value))
            
            waveform.append(value)
        
        # Apply smoothing if history available
        if history and len(history) == points:
            smoothed = []
            for i, (current, prev) in enumerate(zip(waveform, history)):
                smoothed.append(prev * 0.3 + current * 0.7)
            waveform = smoothed
        
        return waveform
    
    @staticmethod
    def generate_spectrum_bars(
        audio_levels: List[float],
        bar_count: int = 16
    ) -> List[float]:
        """Generate spectrum bar visualization"""
        bars = []
        levels_per_bar = len(audio_levels) // bar_count
        
        for i in range(bar_count):
            start = i * levels_per_bar
            end = start + levels_per_bar
            bar_levels = audio_levels[start:end]
            avg_level = sum(bar_levels) / len(bar_levels) if bar_levels else 0
            bars.append(avg_level)
        
        return bars


# =============================================================================
# ANIMATION SYSTEM
# =============================================================================

class AnimationSystem:
    """Manages animations for visual elements"""
    
    def __init__(self):
        self.animations: Dict[str, Dict] = {}
        self.running = False
        
    def start_animation(
        self,
        element_id: str,
        animation_type: AnimationType,
        duration: float = 1.0,
        loop: bool = True,
        easing: str = 'ease_in_out'
    ) -> None:
        """Start animation for element"""
        self.animations[element_id] = {
            'type': animation_type,
            'duration': duration,
            'loop': loop,
            'easing': easing,
            'start_time': datetime.now(),
            'progress': 0.0
        }
    
    def stop_animation(self, element_id: str) -> None:
        """Stop animation for element"""
        if element_id in self.animations:
            del self.animations[element_id]
    
    def get_animation_value(self, element_id: str) -> float:
        """Get current animation value (0-1)"""
        if element_id not in self.animations:
            return 1.0
        
        anim = self.animations[element_id]
        elapsed = (datetime.now() - anim['start_time']).total_seconds()
        progress = (elapsed % anim['duration']) / anim['duration']
        
        if not anim['loop'] and elapsed >= anim['duration']:
            return 1.0
        
        # Apply easing
        if anim['easing'] == 'ease_in_out':
            return self._ease_in_out(progress)
        elif anim['easing'] == 'ease_in':
            return progress ** 2
        elif anim['easing'] == 'ease_out':
            return 1 - (1 - progress) ** 2
        
        return progress
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Ease in-out easing function"""
        return t * t * (3 - 2 * t)
    
    def apply_pulse(self, base_value: float, amplitude: float, element_id: str) -> float:
        """Apply pulse animation"""
        anim_value = self.get_animation_value(element_id)
        pulse = math.sin(anim_value * 2 * math.pi) * amplitude
        return base_value + pulse


# =============================================================================
# VISUAL FEEDBACK SYSTEM
# =============================================================================

class VisualFeedbackSystem:
    """
    Main visual feedback system for voice interactions.
    Provides state-based visual feedback with animations.
    """
    
    def __init__(self, theme: str = 'default'):
        self.current_state = VoiceState.IDLE
        self.audio_level = 0.0
        self.theme = THEME_PRESETS.get(theme, THEME_PRESETS['default'])
        self.animation_system = AnimationSystem()
        self.waveform_history: List[List[float]] = []
        self.max_history = 5
        
    def set_theme(self, theme_name: str) -> None:
        """Set visual theme"""
        self.theme = THEME_PRESETS.get(theme_name, THEME_PRESETS['default'])
    
    def update_state(
        self,
        state: VoiceState,
        audio_level: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> VisualFrame:
        """
        Update visual feedback state and generate frame.
        
        Args:
            state: New voice state
            audio_level: Current audio level (0.0 - 1.0)
            metadata: Additional state metadata
            
        Returns:
            VisualFrame: Visual frame to render
        """
        self.current_state = state
        self.audio_level = audio_level
        
        if state == VoiceState.LISTENING:
            return self._render_listening_indicator(audio_level)
        elif state == VoiceState.PROCESSING:
            return self._render_processing_indicator()
        elif state == VoiceState.THINKING:
            thinking_depth = metadata.get('thinking_depth', 5) if metadata else 5
            return self._render_thinking_indicator(thinking_depth)
        elif state == VoiceState.SPEAKING:
            return self._render_speaking_indicator(audio_level)
        elif state == VoiceState.ERROR:
            error_message = metadata.get('error', 'Error occurred') if metadata else 'Error occurred'
            return self._render_error_indicator(error_message)
        elif state == VoiceState.SUCCESS:
            success_message = metadata.get('message', 'Success') if metadata else 'Success'
            return self._render_success_indicator(success_message)
        else:
            return self._render_idle_indicator()
    
    def _render_idle_indicator(self) -> VisualFrame:
        """Render idle state indicator"""
        color = STATE_COLORS[VoiceState.IDLE]
        
        orb = OrbElement(
            position=(50, 50),
            radius=15,
            color=color,
            opacity=0.5,
            pulse_amplitude=0.1,
            pulse_frequency=0.5
        )
        
        # Start slow pulse animation
        self.animation_system.start_animation(
            orb.id,
            AnimationType.PULSE,
            duration=2.0,
            loop=True
        )
        
        return VisualFrame(
            elements=[orb],
            background_color=self.theme['background'],
            background_effect='subtle_glow'
        )
    
    def _render_listening_indicator(self, audio_level: float) -> VisualFrame:
        """Render listening indicator with audio reactivity"""
        color = STATE_COLORS[VoiceState.LISTENING]
        
        # Generate waveform
        waveform_data = WaveformGenerator.generate_audio_reactive_waveform(
            audio_level,
            points=32,
            history=self.waveform_history[-1] if self.waveform_history else None
        )
        
        # Store in history
        self.waveform_history.append(waveform_data)
        if len(self.waveform_history) > self.max_history:
            self.waveform_history.pop(0)
        
        # Create waveform element
        waveform = WaveformElement(
            position=(10, 30),
            size=(80, 40),
            data=waveform_data,
            bar_count=32,
            color=color,
            max_height=40
        )
        
        # Add glow effect
        glow = OrbElement(
            position=(50, 50),
            radius=40,
            color=color,
            opacity=0.3 + (audio_level * 0.4),
            glow_intensity=0.5 + (audio_level * 0.5)
        )
        
        # Add status text
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 85),
            color=self.theme['text'],
            metadata={'text': 'Listening...', 'font_size': 14}
        )
        
        return VisualFrame(
            elements=[glow, waveform, text_element],
            background_color=self.theme['background'],
            background_effect='glow'
        )
    
    def _render_processing_indicator(self) -> VisualFrame:
        """Render processing indicator"""
        color = STATE_COLORS[VoiceState.PROCESSING]
        
        spinner = SpinnerElement(
            position=(50, 40),
            size=(30, 30),
            color=color,
            speed=1.5,
            segments=8
        )
        
        self.animation_system.start_animation(
            spinner.id,
            AnimationType.SPIN,
            duration=1.0,
            loop=True
        )
        
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 75),
            color=self.theme['text'],
            metadata={'text': 'Processing...', 'font_size': 14}
        )
        
        return VisualFrame(
            elements=[spinner, text_element],
            background_color=self.theme['background']
        )
    
    def _render_thinking_indicator(self, thinking_depth: int = 5) -> VisualFrame:
        """Render thinking indicator"""
        color = STATE_COLORS[VoiceState.THINKING]
        
        # Scale complexity based on thinking depth
        nodes = min(20 + thinking_depth * 5, 50)
        connections = min(40 + thinking_depth * 10, 100)
        
        neural_net = NeuralNetworkElement(
            position=(10, 10),
            size=(80, 60),
            nodes=nodes,
            connections=connections,
            activity_level=0.5 + (thinking_depth / 20),
            color=color
        )
        
        # Add progress ring
        progress = VisualElement(
            element_type=VisualElementType.PROGRESS_RING,
            position=(50, 75),
            size=(20, 20),
            color=color,
            metadata={'indeterminate': True}
        )
        
        self.animation_system.start_animation(
            progress.id,
            AnimationType.SPIN,
            duration=2.0,
            loop=True
        )
        
        # Thinking text
        thinking_texts = [
            'Thinking...',
            'Thinking deeply...',
            'Analyzing...',
            'Processing complex query...',
            'Computing response...'
        ]
        text = thinking_texts[min(thinking_depth // 3, len(thinking_texts) - 1)]
        
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 90),
            color=self.theme['text'],
            metadata={'text': text, 'font_size': 12}
        )
        
        return VisualFrame(
            elements=[neural_net, progress, text_element],
            background_color=self.theme['background'],
            background_effect='subtle_glow'
        )
    
    def _render_speaking_indicator(self, audio_level: float) -> VisualFrame:
        """Render speaking indicator"""
        color = STATE_COLORS[VoiceState.SPEAKING]
        
        voice_wave = VoiceWaveElement(
            position=(10, 35),
            size=(80, 30),
            amplitude=audio_level,
            frequency=2.0 + (audio_level * 2),
            wave_count=3,
            color=color
        )
        
        # Add orb with speech-synced pulse
        orb = OrbElement(
            position=(50, 50),
            radius=25,
            color=color,
            opacity=0.7,
            pulse_amplitude=audio_level * 0.3,
            pulse_frequency=3.0
        )
        
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 85),
            color=self.theme['text'],
            metadata={'text': 'Speaking...', 'font_size': 14}
        )
        
        return VisualFrame(
            elements=[orb, voice_wave, text_element],
            background_color=self.theme['background'],
            background_effect='glow'
        )
    
    def _render_error_indicator(self, error_message: str) -> VisualFrame:
        """Render error indicator"""
        color = STATE_COLORS[VoiceState.ERROR]
        
        # Error icon (X)
        icon = VisualElement(
            element_type=VisualElementType.ICON,
            position=(50, 40),
            size=(40, 40),
            color=color,
            metadata={'icon': 'error', 'size': 40}
        )
        
        self.animation_system.start_animation(
            icon.id,
            AnimationType.SHAKE,
            duration=0.5,
            loop=False
        )
        
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 80),
            color=color,
            metadata={'text': error_message, 'font_size': 12}
        )
        
        return VisualFrame(
            elements=[icon, text_element],
            background_color=self.theme['background']
        )
    
    def _render_success_indicator(self, message: str) -> VisualFrame:
        """Render success indicator"""
        color = STATE_COLORS[VoiceState.SUCCESS]
        
        # Success icon (checkmark)
        icon = VisualElement(
            element_type=VisualElementType.ICON,
            position=(50, 40),
            size=(40, 40),
            color=color,
            metadata={'icon': 'checkmark', 'size': 40}
        )
        
        self.animation_system.start_animation(
            icon.id,
            AnimationType.BOUNCE,
            duration=0.5,
            loop=False
        )
        
        text_element = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 80),
            color=self.theme['text'],
            metadata={'text': message, 'font_size': 12}
        )
        
        return VisualFrame(
            elements=[icon, text_element],
            background_color=self.theme['background']
        )
    
    def highlight_screen_region(
        self,
        region: HighlightRegion,
        duration_ms: Optional[int] = None
    ) -> VisualFrame:
        """
        Create highlight frame for screen region.
        
        Args:
            region: Region to highlight
            duration_ms: Optional override for duration
            
        Returns:
            VisualFrame: Highlight frame
        """
        highlight = VisualElement(
            element_type=VisualElementType.HIGHLIGHT,
            position=(region.bounds[0], region.bounds[1]),
            size=(region.bounds[2], region.bounds[3]),
            color=region.color,
            opacity=region.opacity,
            metadata={
                'style': region.style,
                'pulse': region.pulse,
                'animated': region.animated
            }
        )
        
        if region.pulse:
            self.animation_system.start_animation(
                highlight.id,
                AnimationType.PULSE,
                duration=1.0,
                loop=True
            )
        
        return VisualFrame(
            elements=[highlight],
            background_color='transparent'
        )


# =============================================================================
# RESPONSE CARD RENDERER
# =============================================================================

class ResponseCardRenderer:
    """Renders rich response cards"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = THEME_PRESETS.get(theme, THEME_PRESETS['default'])
    
    def render_info_card(
        self,
        title: str,
        content: str,
        icon: Optional[str] = None,
        actions: Optional[List[Dict]] = None
    ) -> VisualFrame:
        """Render information card"""
        elements = []
        
        # Card background
        card = VisualElement(
            element_type=VisualElementType.CARD,
            position=(5, 5),
            size=(90, 90),
            color=self.theme['background'],
            opacity=0.95,
            metadata={'corner_radius': 8}
        )
        elements.append(card)
        
        # Title
        title_elem = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 15),
            color=self.theme['text'],
            metadata={
                'text': title,
                'font_size': 16,
                'font_weight': 'bold'
            }
        )
        elements.append(title_elem)
        
        # Content
        content_elem = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 45),
            color=self.theme['text'],
            metadata={
                'text': content,
                'font_size': 12,
                'max_lines': 5
            }
        )
        elements.append(content_elem)
        
        # Actions
        if actions:
            y_pos = 75
            for i, action in enumerate(actions[:3]):  # Max 3 actions
                action_elem = VisualElement(
                    element_type=VisualElementType.TEXT,
                    position=(20 + i * 30, y_pos),
                    color=self.theme['accent'],
                    metadata={
                        'text': action.get('label', 'Action'),
                        'font_size': 11,
                        'clickable': True
                    }
                )
                elements.append(action_elem)
        
        return VisualFrame(
            elements=elements,
            background_color='transparent'
        )
    
    def render_progress_card(
        self,
        title: str,
        progress: float,  # 0.0 to 1.0
        status: str = "Processing..."
    ) -> VisualFrame:
        """Render progress card"""
        elements = []
        
        # Card background
        card = VisualElement(
            element_type=VisualElementType.CARD,
            position=(5, 5),
            size=(90, 90),
            color=self.theme['background'],
            opacity=0.95
        )
        elements.append(card)
        
        # Title
        title_elem = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 20),
            color=self.theme['text'],
            metadata={
                'text': title,
                'font_size': 14,
                'font_weight': 'bold'
            }
        )
        elements.append(title_elem)
        
        # Progress bar background
        progress_bg = VisualElement(
            element_type=VisualElementType.CARD,
            position=(10, 45),
            size=(80, 10),
            color='#333333',
            metadata={'corner_radius': 5}
        )
        elements.append(progress_bg)
        
        # Progress bar fill
        progress_fill = VisualElement(
            element_type=VisualElementType.CARD,
            position=(10, 45),
            size=(80 * progress, 10),
            color=self.theme['accent'],
            metadata={'corner_radius': 5}
        )
        elements.append(progress_fill)
        
        # Status text
        status_elem = VisualElement(
            element_type=VisualElementType.TEXT,
            position=(50, 70),
            color=self.theme['text'],
            metadata={
                'text': f"{status} ({int(progress * 100)}%)",
                'font_size': 11
            }
        )
        elements.append(status_elem)
        
        return VisualFrame(
            elements=elements,
            background_color='transparent'
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of visual feedback system"""
    
    # Initialize visual feedback system
    vfs = VisualFeedbackSystem(theme='default')
    
    # Test different states
    states = [
        (VoiceState.IDLE, 0.0, None),
        (VoiceState.LISTENING, 0.5, None),
        (VoiceState.LISTENING, 0.8, None),
        (VoiceState.PROCESSING, 0.0, None),
        (VoiceState.THINKING, 0.0, {'thinking_depth': 7}),
        (VoiceState.SPEAKING, 0.6, None),
        (VoiceState.SUCCESS, 0.0, {'message': 'Task completed'}),
        (VoiceState.ERROR, 0.0, {'error': 'Connection failed'}),
    ]
    
    print("Testing Visual Feedback System States:")
    print("=" * 50)
    
    for state, audio_level, metadata in states:
        print(f"\nState: {state.value}")
        frame = vfs.update_state(state, audio_level, metadata)
        print(f"  Elements: {len(frame.elements)}")
        print(f"  Background: {frame.background_color}")
        for elem in frame.elements:
            print(f"    - {elem.element_type.value}: {elem.color}")
    
    # Test response card renderer
    print("\n" + "=" * 50)
    print("Testing Response Card Renderer:")
    
    renderer = ResponseCardRenderer()
    
    info_card = renderer.render_info_card(
        title="Weather Update",
        content="It's currently 72F and sunny in your location.",
        actions=[
            {'label': 'Details'},
            {'label': 'Forecast'}
        ]
    )
    print(f"\nInfo Card Elements: {len(info_card.elements)}")
    
    progress_card = renderer.render_progress_card(
        title="Downloading File",
        progress=0.65,
        status="Downloading..."
    )
    print(f"Progress Card Elements: {len(progress_card.elements)}")


if __name__ == "__main__":
    asyncio.run(main())
