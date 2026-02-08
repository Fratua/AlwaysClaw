"""
Accessibility Layer for Multi-Modal Voice Interface
Windows 10 OpenClaw-Inspired AI Agent System

This module implements comprehensive accessibility features including:
- Screen reader integration (NVDA, JAWS, Narrator)
- Keyboard navigation
- High contrast themes
- Color blindness adaptations
- Motor accessibility features
- Cognitive accessibility support
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ScreenReaderType(Enum):
    """Supported screen readers"""
    NVDA = "nvda"
    JAWS = "jaws"
    NARRATOR = "narrator"
    SYSTEM = "system"
    NONE = "none"


class ColorBlindType(Enum):
    """Types of color blindness"""
    NONE = "none"
    PROTANOPIA = "protanopia"      # Red-blind
    DEUTERANOPIA = "deuteranopia"  # Green-blind
    TRITANOPIA = "tritanopia"      # Blue-blind
    ACHROMATOPSIA = "achromatopsia"  # Total color blindness


class AccessibilityPriority(Enum):
    """Priority levels for accessibility announcements"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AccessibilitySettings:
    """
    Complete accessibility settings for the system.
    Covers vision, hearing, motor, and cognitive accessibility.
    """
    # Vision
    high_contrast: bool = False
    large_text: bool = False
    screen_reader: bool = False
    screen_reader_type: ScreenReaderType = ScreenReaderType.SYSTEM
    color_blind_mode: ColorBlindType = ColorBlindType.NONE
    text_scale: float = 1.0
    cursor_size: int = 1
    
    # Hearing
    captions_enabled: bool = True
    visual_alerts: bool = False
    haptic_feedback: bool = True
    flash_notifications: bool = False
    
    # Motor
    dwell_click: bool = False
    dwell_time_ms: int = 1000
    sticky_keys: bool = False
    filter_keys: bool = False
    toggle_keys: bool = False
    voice_control_only: bool = False
    switch_access: bool = False
    
    # Cognitive
    simplified_ui: bool = False
    extended_timeouts: bool = False
    reading_assistance: bool = False
    focus_indicators: bool = True
    reduce_animations: bool = False
    reading_speed: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert settings to dictionary"""
        return {
            'vision': {
                'high_contrast': self.high_contrast,
                'large_text': self.large_text,
                'screen_reader': self.screen_reader,
                'screen_reader_type': self.screen_reader_type.value,
                'color_blind_mode': self.color_blind_mode.value,
                'text_scale': self.text_scale,
                'cursor_size': self.cursor_size
            },
            'hearing': {
                'captions_enabled': self.captions_enabled,
                'visual_alerts': self.visual_alerts,
                'haptic_feedback': self.haptic_feedback,
                'flash_notifications': self.flash_notifications
            },
            'motor': {
                'dwell_click': self.dwell_click,
                'dwell_time_ms': self.dwell_time_ms,
                'sticky_keys': self.sticky_keys,
                'filter_keys': self.filter_keys,
                'toggle_keys': self.toggle_keys,
                'voice_control_only': self.voice_control_only,
                'switch_access': self.switch_access
            },
            'cognitive': {
                'simplified_ui': self.simplified_ui,
                'extended_timeouts': self.extended_timeouts,
                'reading_assistance': self.reading_assistance,
                'focus_indicators': self.focus_indicators,
                'reduce_animations': self.reduce_animations,
                'reading_speed': self.reading_speed
            }
        }


@dataclass
class KeyboardShortcut:
    """Keyboard shortcut definition"""
    action: str
    keys: List[str]
    description: str
    category: str = "general"
    accessible_description: Optional[str] = None


@dataclass
class AccessibleElement:
    """Accessible UI element information"""
    element_id: str
    element_type: str
    label: str
    description: Optional[str] = None
    role: str = "generic"
    state: Dict[str, bool] = field(default_factory=dict)
    value: Optional[str] = None
    shortcut: Optional[str] = None
    children: List[str] = field(default_factory=list)


# =============================================================================
# SCREEN READER INTEGRATION
# =============================================================================

class ScreenReaderIntegration:
    """
    Integration with Windows screen readers.
    Supports NVDA, JAWS, Windows Narrator, and system default.
    """
    
    # Screen reader executable names for detection
    READER_PROCESSES = {
        ScreenReaderType.NVDA: ['nvda.exe'],
        ScreenReaderType.JAWS: ['jfw.exe', 'fsreader.exe'],
        ScreenReaderType.NARRATOR: ['narrator.exe']
    }
    
    def __init__(self):
        self.active_reader: ScreenReaderType = ScreenReaderType.NONE
        self.reader_handle: Optional[Any] = None
        self.announcement_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize screen reader integration"""
        try:
            # Detect active screen reader
            self.active_reader = await self._detect_screen_reader()
            
            if self.active_reader != ScreenReaderType.NONE:
                logger.info(f"Detected screen reader: {self.active_reader.value}")
                await self._connect_to_reader()
            else:
                logger.info("No screen reader detected")
            
            # Start announcement processor
            self.running = True
            asyncio.create_task(self._process_announcements())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize screen reader: {e}")
            return False
    
    async def _detect_screen_reader(self) -> ScreenReaderType:
        """Detect active screen reader"""
        try:
            # In production, use psutil to check running processes
            # import psutil
            # for reader_type, processes in self.READER_PROCESSES.items():
            #     for proc in psutil.process_iter(['name']):
            #         if proc.info['name'].lower() in [p.lower() for p in processes]:
            #             return reader_type
            
            # Placeholder - check Windows registry for default
            return ScreenReaderType.SYSTEM
            
        except Exception as e:
            logger.error(f"Error detecting screen reader: {e}")
            return ScreenReaderType.NONE
    
    async def _connect_to_reader(self) -> bool:
        """Connect to detected screen reader"""
        try:
            if self.active_reader == ScreenReaderType.NVDA:
                return await self._connect_nvda()
            elif self.active_reader == ScreenReaderType.JAWS:
                return await self._connect_jaws()
            elif self.active_reader == ScreenReaderType.NARRATOR:
                return await self._connect_narrator()
            elif self.active_reader == ScreenReaderType.SYSTEM:
                return await self._connect_system()
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to screen reader: {e}")
            return False
    
    async def _connect_nvda(self) -> bool:
        """Connect to NVDA screen reader"""
        try:
            # In production:
            # import nvda_client
            # self.reader_handle = nvda_client.initialize()
            logger.info("Connected to NVDA")
            return True
        except Exception as e:
            logger.error(f"NVDA connection failed: {e}")
            return False
    
    async def _connect_jaws(self) -> bool:
        """Connect to JAWS screen reader"""
        try:
            # In production:
            # import jaws_client
            # self.reader_handle = jaws_client.initialize()
            logger.info("Connected to JAWS")
            return True
        except Exception as e:
            logger.error(f"JAWS connection failed: {e}")
            return False
    
    async def _connect_narrator(self) -> bool:
        """Connect to Windows Narrator"""
        try:
            # In production:
            # Use UI Automation to interact with Narrator
            logger.info("Connected to Narrator")
            return True
        except Exception as e:
            logger.error(f"Narrator connection failed: {e}")
            return False
    
    async def _connect_system(self) -> bool:
        """Connect to system default accessibility"""
        try:
            # In production:
            # Use Windows UI Automation API
            logger.info("Connected to system accessibility")
            return True
        except Exception as e:
            logger.error(f"System accessibility connection failed: {e}")
            return False
    
    async def announce(
        self,
        message: str,
        priority: AccessibilityPriority = AccessibilityPriority.NORMAL,
        interrupt: bool = False
    ) -> None:
        """
        Announce message through screen reader.
        
        Args:
            message: Message to announce
            priority: Priority level
            interrupt: Whether to interrupt current speech
        """
        if self.active_reader == ScreenReaderType.NONE:
            return
        
        # Format message for screen reader
        formatted = self._format_for_screen_reader(message)
        
        # Add to queue
        await self.announcement_queue.put({
            'message': formatted,
            'priority': priority,
            'interrupt': interrupt,
            'timestamp': datetime.now()
        })
    
    async def _process_announcements(self) -> None:
        """Process announcement queue"""
        while self.running:
            try:
                announcement = await self.announcement_queue.get()
                
                # Send to appropriate screen reader
                if self.active_reader == ScreenReaderType.NVDA:
                    await self._announce_nvda(announcement)
                elif self.active_reader == ScreenReaderType.JAWS:
                    await self._announce_jaws(announcement)
                elif self.active_reader == ScreenReaderType.NARRATOR:
                    await self._announce_narrator(announcement)
                elif self.active_reader == ScreenReaderType.SYSTEM:
                    await self._announce_system(announcement)
                
            except Exception as e:
                logger.error(f"Error processing announcement: {e}")
    
    async def _announce_nvda(self, announcement: Dict) -> None:
        """Announce via NVDA"""
        try:
            # In production:
            # import speech
            # speech.speakMessage(announcement['message'])
            logger.debug(f"NVDA: {announcement['message']}")
        except Exception as e:
            logger.error(f"NVDA announcement failed: {e}")
    
    async def _announce_jaws(self, announcement: Dict) -> None:
        """Announce via JAWS"""
        try:
            # In production:
            # Use JAWS COM API
            logger.debug(f"JAWS: {announcement['message']}")
        except Exception as e:
            logger.error(f"JAWS announcement failed: {e}")
    
    async def _announce_narrator(self, announcement: Dict) -> None:
        """Announce via Narrator"""
        try:
            # In production:
            # Use Windows UI Automation
            logger.debug(f"Narrator: {announcement['message']}")
        except Exception as e:
            logger.error(f"Narrator announcement failed: {e}")
    
    async def _announce_system(self, announcement: Dict) -> None:
        """Announce via system default"""
        try:
            # In production:
            # Use Windows Text-to-Speech
            logger.debug(f"System: {announcement['message']}")
        except Exception as e:
            logger.error(f"System announcement failed: {e}")
    
    def _format_for_screen_reader(self, message: str) -> str:
        """
        Format message for optimal screen reader experience.
        
        - Removes visual-only content
        - Expands abbreviations
        - Adds context for interactive elements
        """
        formatted = message
        
        # Remove visual-only content markers
        formatted = re.sub(r'\[image:.*?\]', ' [image] ', formatted)
        formatted = re.sub(r'\[icon:.*?\]', ' [icon] ', formatted)
        
        # Add context for interactive elements
        formatted = re.sub(
            r'\[button:(.*?)\]',
            r'Button: \1',
            formatted
        )
        formatted = re.sub(
            r'\[link:(.*?)\]',
            r'Link: \1',
            formatted
        )
        
        # Expand common abbreviations
        abbreviations = {
            'btn': 'button',
            'msg': 'message',
            'txt': 'text',
            'num': 'number',
            'info': 'information',
            'config': 'configuration'
        }
        
        for abbr, full in abbreviations.items():
            formatted = re.sub(
                rf'\b{abbr}\b',
                full,
                formatted,
                flags=re.IGNORECASE
            )
        
        # Normalize whitespace
        formatted = ' '.join(formatted.split())
        
        return formatted
    
    async def announce_element(self, element: AccessibleElement) -> None:
        """Announce accessible element information"""
        announcement_parts = [element.label]
        
        if element.description:
            announcement_parts.append(element.description)
        
        if element.value:
            announcement_parts.append(f"Value: {element.value}")
        
        if element.shortcut:
            announcement_parts.append(f"Shortcut: {element.shortcut}")
        
        # Add state information
        for state_name, state_value in element.state.items():
            if state_value:
                announcement_parts.append(state_name)
        
        message = '. '.join(announcement_parts)
        await self.announce(message)
    
    async def silence(self) -> None:
        """Silence screen reader immediately via Windows COM/accessibility APIs."""
        try:
            if self.active_reader == ScreenReaderType.NVDA:
                try:
                    import ctypes
                    nvda_lib = ctypes.windll.LoadLibrary("nvdaControllerClient64.dll")
                    nvda_lib.nvdaController_cancelSpeech()
                except (OSError, AttributeError):
                    logger.debug("NVDA controller DLL not available - silence is a no-op")
            elif self.active_reader == ScreenReaderType.JAWS:
                try:
                    import win32com.client
                    jaws = win32com.client.Dispatch("FreedomSci.JawsApi")
                    jaws.StopSpeech()
                except (ImportError, Exception):
                    logger.debug("JAWS COM API not available - silence is a no-op")
            elif self.active_reader == ScreenReaderType.NARRATOR:
                logger.debug("Narrator silence not directly controllable via API")
            else:
                logger.debug(f"No silence implementation for reader: {self.active_reader}")
        except Exception as e:
            logger.error(f"Failed to silence screen reader: {e}")


# =============================================================================
# KEYBOARD NAVIGATION
# =============================================================================

class KeyboardNavigation:
    """
    Comprehensive keyboard navigation support.
    Implements full keyboard accessibility for all UI elements.
    """
    
    # Standard keyboard shortcuts
    DEFAULT_SHORTCUTS: List[KeyboardShortcut] = [
        # Basic navigation
        KeyboardShortcut('focus_next', ['Tab'], 'Move focus to next element', 'navigation'),
        KeyboardShortcut('focus_previous', ['Shift', 'Tab'], 'Move focus to previous element', 'navigation'),
        KeyboardShortcut('activate', ['Enter'], 'Activate focused element', 'navigation'),
        KeyboardShortcut('activate_alt', ['Space'], 'Activate focused element (alternative)', 'navigation'),
        KeyboardShortcut('cancel', ['Escape'], 'Cancel current operation', 'navigation'),
        
        # Voice control
        KeyboardShortcut('push_to_talk', ['Control', 'Space'], 'Push to talk', 'voice'),
        KeyboardShortcut('toggle_voice', ['Control', 'Shift', 'V'], 'Toggle voice mode', 'voice'),
        KeyboardShortcut('stop_speaking', ['Control', 'Shift', 'S'], 'Stop speaking', 'voice'),
        
        # Mode switching
        KeyboardShortcut('mode_voice_primary', ['Control', '1'], 'Switch to voice primary mode', 'mode'),
        KeyboardShortcut('mode_visual_primary', ['Control', '2'], 'Switch to visual primary mode', 'mode'),
        KeyboardShortcut('mode_text_primary', ['Control', '3'], 'Switch to text primary mode', 'mode'),
        KeyboardShortcut('mode_hands_free', ['Control', '4'], 'Switch to hands-free mode', 'mode'),
        
        # Agent control
        KeyboardShortcut('pause_agent', ['Control', 'P'], 'Pause agent', 'agent'),
        KeyboardShortcut('resume_agent', ['Control', 'R'], 'Resume agent', 'agent'),
        KeyboardShortcut('emergency_stop', ['Control', 'Shift', 'E'], 'Emergency stop', 'agent'),
        
        # Accessibility
        KeyboardShortcut('toggle_high_contrast', ['Control', 'Shift', 'H'], 'Toggle high contrast', 'accessibility'),
        KeyboardShortcut('increase_text_size', ['Control', 'Plus'], 'Increase text size', 'accessibility'),
        KeyboardShortcut('decrease_text_size', ['Control', 'Minus'], 'Decrease text size', 'accessibility'),
        KeyboardShortcut('toggle_captions', ['Control', 'Shift', 'C'], 'Toggle captions', 'accessibility'),
        KeyboardShortcut('toggle_screen_reader', ['Control', 'Shift', 'R'], 'Toggle screen reader', 'accessibility'),
        
        # Help
        KeyboardShortcut('show_help', ['F1'], 'Show help', 'help'),
        KeyboardShortcut('show_shortcuts', ['Control', 'Slash'], 'Show keyboard shortcuts', 'help'),
    ]
    
    def __init__(self, settings: AccessibilitySettings):
        self.settings = settings
        self.shortcuts: Dict[str, KeyboardShortcut] = {}
        self.focused_element: Optional[str] = None
        self.focus_order: List[str] = []
        self.focus_index: int = -1
        self.sticky_keys_active: Set[str] = set()
        
        self._initialize_shortcuts()
    
    def _initialize_shortcuts(self) -> None:
        """Initialize keyboard shortcuts"""
        for shortcut in self.DEFAULT_SHORTCUTS:
            key_combo = '+'.join(sorted(shortcut.keys))
            self.shortcuts[key_combo] = shortcut
    
    def handle_keypress(self, keys_pressed: List[str]) -> Optional[str]:
        """
        Handle keyboard input and return action.
        
        Args:
            keys_pressed: List of currently pressed keys
            
        Returns:
            Action name or None
        """
        # Handle sticky keys
        if self.settings.sticky_keys:
            keys_pressed = self._apply_sticky_keys(keys_pressed)
        
        # Normalize key combination
        key_combo = '+'.join(sorted(keys_pressed))
        
        # Check for shortcut match
        if key_combo in self.shortcuts:
            shortcut = self.shortcuts[key_combo]
            return shortcut.action
        
        # Handle character input
        if len(keys_pressed) == 1 and keys_pressed[0].isprintable():
            return f'char_input:{keys_pressed[0]}'
        
        return None
    
    def _apply_sticky_keys(self, keys_pressed: List[str]) -> List[str]:
        """Apply sticky keys modifier"""
        modifier_keys = {'Control', 'Shift', 'Alt', 'Windows'}
        
        # Track sticky modifiers
        for key in keys_pressed:
            if key in modifier_keys:
                if key in self.sticky_keys_active:
                    self.sticky_keys_active.discard(key)
                else:
                    self.sticky_keys_active.add(key)
        
        # Combine active modifiers with current keys
        combined = list(self.sticky_keys_active) + keys_pressed
        
        # Clear modifiers after non-modifier key
        if any(k not in modifier_keys for k in keys_pressed):
            self.sticky_keys_active.clear()
        
        return combined
    
    def move_focus_next(self) -> Optional[str]:
        """Move focus to next element"""
        if not self.focus_order:
            return None
        
        self.focus_index = (self.focus_index + 1) % len(self.focus_order)
        self.focused_element = self.focus_order[self.focus_index]
        
        return self.focused_element
    
    def move_focus_previous(self) -> Optional[str]:
        """Move focus to previous element"""
        if not self.focus_order:
            return None
        
        self.focus_index = (self.focus_index - 1) % len(self.focus_order)
        self.focused_element = self.focus_order[self.focus_index]
        
        return self.focused_element
    
    def set_focus_order(self, element_ids: List[str]) -> None:
        """Set the tab order for focus navigation"""
        self.focus_order = element_ids
        self.focus_index = -1
    
    def get_shortcuts_by_category(self, category: str) -> List[KeyboardShortcut]:
        """Get shortcuts for a specific category"""
        return [
            s for s in self.shortcuts.values()
            if s.category == category
        ]
    
    def get_all_shortcuts_formatted(self) -> Dict[str, List[Dict]]:
        """Get all shortcuts formatted by category"""
        result: Dict[str, List[Dict]] = {}
        
        for shortcut in self.shortcuts.values():
            if shortcut.category not in result:
                result[shortcut.category] = []
            
            result[shortcut.category].append({
                'action': shortcut.action,
                'keys': shortcut.keys,
                'description': shortcut.description
            })
        
        return result


# =============================================================================
# COLOR ACCESSIBILITY
# =============================================================================

class ColorAccessibility:
    """
    Color accessibility adaptations.
    Supports high contrast themes and color blindness simulations.
    """
    
    # Color blindness transformation matrices
    COLOR_BLIND_MATRICES = {
        ColorBlindType.PROTANOPIA: [
            [0.567, 0.433, 0.000],
            [0.558, 0.442, 0.000],
            [0.000, 0.242, 0.758]
        ],
        ColorBlindType.DEUTERANOPIA: [
            [0.625, 0.375, 0.000],
            [0.700, 0.300, 0.000],
            [0.000, 0.300, 0.700]
        ],
        ColorBlindType.TRITANOPIA: [
            [0.950, 0.050, 0.000],
            [0.000, 0.433, 0.567],
            [0.000, 0.475, 0.525]
        ],
        ColorBlindType.ACHROMATOPSIA: [
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114]
        ]
    }
    
    def __init__(self, settings: AccessibilitySettings):
        self.settings = settings
        self.current_theme: Dict[str, str] = {}
        self._apply_theme()
    
    def _apply_theme(self) -> None:
        """Apply appropriate theme based on settings"""
        if self.settings.high_contrast:
            self.current_theme = self._get_high_contrast_theme()
        else:
            self.current_theme = self._get_default_theme()
        
        # Apply color blindness adaptation if needed
        if self.settings.color_blind_mode != ColorBlindType.NONE:
            self.current_theme = self._adapt_for_color_blindness(
                self.current_theme,
                self.settings.color_blind_mode
            )
    
    def _get_default_theme(self) -> Dict[str, str]:
        """Get default color theme"""
        return {
            'background': '#1E1E1E',
            'text': '#FFFFFF',
            'accent': '#007ACC',
            'success': '#4EC9B0',
            'warning': '#CE9178',
            'error': '#F44747',
            'info': '#569CD6',
            'border': '#3E3E3E',
            'focus': '#007ACC'
        }
    
    def _get_high_contrast_theme(self) -> Dict[str, str]:
        """Get high contrast theme"""
        return {
            'background': '#000000',
            'text': '#FFFFFF',
            'accent': '#FFFF00',
            'success': '#00FF00',
            'warning': '#FFFF00',
            'error': '#FF0000',
            'info': '#00FFFF',
            'border': '#FFFFFF',
            'focus': '#FFFF00'
        }
    
    def _adapt_for_color_blindness(
        self,
        theme: Dict[str, str],
        color_blind_type: ColorBlindType
    ) -> Dict[str, str]:
        """Adapt theme colors for color blindness"""
        if color_blind_type == ColorBlindType.NONE:
            return theme
        
        matrix = self.COLOR_BLIND_MATRICES.get(color_blind_type)
        if not matrix:
            return theme
        
        adapted_theme = {}
        for key, color in theme.items():
            adapted_theme[key] = self._apply_color_matrix(color, matrix)
        
        return adapted_theme
    
    def _apply_color_matrix(self, hex_color: str, matrix: List[List[float]]) -> str:
        """Apply color transformation matrix to hex color"""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        # Apply matrix
        new_r = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b
        new_g = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b
        new_b = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b
        
        # Clamp and convert back to hex
        new_r = max(0, min(255, int(new_r * 255)))
        new_g = max(0, min(255, int(new_g * 255)))
        new_b = max(0, min(255, int(new_b * 255)))
        
        return f'#{new_r:02x}{new_g:02x}{new_b:02x}'
    
    def get_accessible_colors(self) -> Dict[str, str]:
        """Get current accessible color theme"""
        return self.current_theme.copy()
    
    def ensure_contrast_ratio(self, foreground: str, background: str) -> str:
        """
        Ensure foreground color has sufficient contrast with background.
        Returns adjusted foreground color if needed.
        """
        # Calculate luminance
        def get_luminance(hex_color: str) -> float:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            
            # Apply gamma correction
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        lum_fg = get_luminance(foreground)
        lum_bg = get_luminance(background)
        
        # Calculate contrast ratio
        contrast = (max(lum_fg, lum_bg) + 0.05) / (min(lum_fg, lum_bg) + 0.05)
        
        # WCAG AA requires 4.5:1 for normal text, 3:1 for large text
        if contrast < 4.5:
            # Adjust foreground color for better contrast
            if lum_bg > 0.5:
                return '#000000'  # Dark foreground on light background
            else:
                return '#FFFFFF'  # Light foreground on dark background
        
        return foreground


# =============================================================================
# MOTOR ACCESSIBILITY
# =============================================================================

class MotorAccessibility:
    """
    Motor accessibility features.
    Supports dwell clicking, switch access, and voice control.
    """
    
    def __init__(self, settings: AccessibilitySettings):
        self.settings = settings
        self.dwell_target: Optional[str] = None
        self.dwell_start_time: Optional[datetime] = None
        self.switch_elements: List[str] = []
        self.switch_index: int = 0
        
    async def start_dwell_tracking(self, element_id: str) -> None:
        """Start tracking dwell on element"""
        if not self.settings.dwell_click:
            return
        
        self.dwell_target = element_id
        self.dwell_start_time = datetime.now()
    
    async def stop_dwell_tracking(self) -> None:
        """Stop dwell tracking"""
        self.dwell_target = None
        self.dwell_start_time = None
    
    async def check_dwell_completion(self) -> Optional[str]:
        """
        Check if dwell time has been reached.
        Returns element ID if dwell completed.
        """
        if not self.dwell_target or not self.dwell_start_time:
            return None
        
        elapsed = (datetime.now() - self.dwell_start_time).total_seconds() * 1000
        
        if elapsed >= self.settings.dwell_time_ms:
            element = self.dwell_target
            await self.stop_dwell_tracking()
            return element
        
        return None
    
    def get_dwell_progress(self) -> float:
        """Get current dwell progress (0.0 - 1.0)"""
        if not self.dwell_target or not self.dwell_start_time:
            return 0.0
        
        elapsed = (datetime.now() - self.dwell_start_time).total_seconds() * 1000
        return min(1.0, elapsed / self.settings.dwell_time_ms)
    
    def set_switch_elements(self, element_ids: List[str]) -> None:
        """Set elements for switch access navigation"""
        self.switch_elements = element_ids
        self.switch_index = 0
    
    def switch_next(self) -> Optional[str]:
        """Move to next element in switch access"""
        if not self.switch_elements:
            return None
        
        self.switch_index = (self.switch_index + 1) % len(self.switch_elements)
        return self.switch_elements[self.switch_index]
    
    def switch_activate(self) -> Optional[str]:
        """Activate current switch element"""
        if not self.switch_elements:
            return None
        
        return self.switch_elements[self.switch_index]


# =============================================================================
# COGNITIVE ACCESSIBILITY
# =============================================================================

class CognitiveAccessibility:
    """
    Cognitive accessibility features.
    Supports simplified UI, reading assistance, and extended timeouts.
    """
    
    def __init__(self, settings: AccessibilitySettings):
        self.settings = settings
        self.reading_queue: asyncio.Queue = asyncio.Queue()
        self.simplification_rules = self._load_simplification_rules()
        
    def _load_simplification_rules(self) -> Dict[str, str]:
        """Load text simplification rules"""
        return {
            'utilize': 'use',
            'implement': 'do',
            'configure': 'set up',
            'initialize': 'start',
            'terminate': 'end',
            'sufficient': 'enough',
            'insufficient': 'not enough',
            'approximately': 'about',
            'immediately': 'now',
            'subsequently': 'then',
            'nevertheless': 'but',
            'furthermore': 'also',
            'assistance': 'help',
            'assisting': 'helping',
            'assisted': 'helped'
        }
    
    def simplify_text(self, text: str) -> str:
        """
        Simplify text for easier comprehension.
        
        - Replaces complex words with simpler alternatives
        - Breaks long sentences
        - Adds structure
        """
        if not self.settings.simplified_ui:
            return text
        
        simplified = text
        
        # Apply simplification rules
        for complex_word, simple_word in self.simplification_rules.items():
            simplified = re.sub(
                rf'\b{complex_word}\b',
                simple_word,
                simplified,
                flags=re.IGNORECASE
            )
        
        # Break long sentences (simplified approach)
        sentences = re.split(r'(?<=[.!?])\s+', simplified)
        simplified_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 20:
                # Break into smaller chunks
                chunks = [' '.join(words[i:i+15]) for i in range(0, len(words), 15)]
                simplified_sentences.extend(chunks)
            else:
                simplified_sentences.append(sentence)
        
        return ' '.join(simplified_sentences)
    
    def calculate_reading_time(self, text: str) -> float:
        """
        Calculate estimated reading time in seconds.
        
        Based on average reading speed adjusted for accessibility settings.
        """
        word_count = len(text.split())
        
        # Base reading speed (words per minute)
        base_wpm = 200
        
        # Adjust for accessibility settings
        if self.settings.reading_assistance:
            base_wpm = 150
        
        # Apply user's reading speed preference
        adjusted_wpm = base_wpm * self.settings.reading_speed
        
        # Calculate time in seconds
        reading_time = (word_count / adjusted_wpm) * 60
        
        # Add buffer for comprehension
        reading_time *= 1.2
        
        return reading_time
    
    def get_extended_timeout(self, base_timeout: float) -> float:
        """Get extended timeout based on accessibility settings"""
        if self.settings.extended_timeouts:
            return base_timeout * 2.0
        return base_timeout


# =============================================================================
# ACCESSIBILITY MANAGER
# =============================================================================

class AccessibilityManager:
    """
    Central manager for all accessibility features.
    Coordinates screen reader, keyboard navigation, color, motor, and cognitive accessibility.
    """
    
    def __init__(self, settings: Optional[AccessibilitySettings] = None):
        self.settings = settings or AccessibilitySettings()
        
        # Initialize subsystems
        self.screen_reader = ScreenReaderIntegration()
        self.keyboard = KeyboardNavigation(self.settings)
        self.color = ColorAccessibility(self.settings)
        self.motor = MotorAccessibility(self.settings)
        self.cognitive = CognitiveAccessibility(self.settings)
        
        self.accessible_elements: Dict[str, AccessibleElement] = {}
        self.announcement_callbacks: List[Callable] = []
        
    async def initialize(self) -> bool:
        """Initialize all accessibility subsystems"""
        try:
            # Initialize screen reader
            if self.settings.screen_reader:
                await self.screen_reader.initialize()
            
            logger.info("Accessibility manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize accessibility: {e}")
            return False
    
    def register_element(self, element: AccessibleElement) -> None:
        """Register UI element for accessibility"""
        self.accessible_elements[element.element_id] = element
        
        # Update keyboard focus order
        focus_order = list(self.accessible_elements.keys())
        self.keyboard.set_focus_order(focus_order)
    
    def unregister_element(self, element_id: str) -> None:
        """Unregister UI element"""
        if element_id in self.accessible_elements:
            del self.accessible_elements[element_id]
            
            # Update keyboard focus order
            focus_order = list(self.accessible_elements.keys())
            self.keyboard.set_focus_order(focus_order)
    
    async def announce(
        self,
        message: str,
        priority: AccessibilityPriority = AccessibilityPriority.NORMAL,
        interrupt: bool = False
    ) -> None:
        """
        Announce message through all accessibility channels.
        
        Args:
            message: Message to announce
            priority: Priority level
            interrupt: Whether to interrupt current speech
        """
        # Simplify if needed
        if self.settings.simplified_ui:
            message = self.cognitive.simplify_text(message)
        
        # Announce through screen reader
        if self.settings.screen_reader:
            await self.screen_reader.announce(message, priority, interrupt)
        
        # Notify callbacks
        for callback in self.announcement_callbacks:
            try:
                callback(message, priority)
            except Exception as e:
                logger.error(f"Announcement callback error: {e}")
    
    def register_announcement_callback(self, callback: Callable) -> None:
        """Register callback for announcements"""
        self.announcement_callbacks.append(callback)
    
    def handle_keypress(self, keys_pressed: List[str]) -> Optional[str]:
        """Handle keyboard input"""
        return self.keyboard.handle_keypress(keys_pressed)
    
    def get_theme_colors(self) -> Dict[str, str]:
        """Get current accessible theme colors"""
        return self.color.get_accessible_colors()
    
    def update_settings(self, settings: AccessibilitySettings) -> None:
        """Update accessibility settings"""
        self.settings = settings
        
        # Re-initialize subsystems with new settings
        self.color = ColorAccessibility(settings)
        self.motor = MotorAccessibility(settings)
        self.cognitive = CognitiveAccessibility(settings)
        
        # Re-initialize screen reader if needed
        if settings.screen_reader and not self.screen_reader.active_reader:
            asyncio.create_task(self.screen_reader.initialize())
    
    def get_accessible_description(self, element_id: str) -> Optional[str]:
        """Get accessible description for element"""
        element = self.accessible_elements.get(element_id)
        if not element:
            return None
        
        parts = [element.label]
        
        if element.description:
            parts.append(element.description)
        
        if element.role:
            parts.append(f"Role: {element.role}")
        
        if element.shortcut:
            parts.append(f"Shortcut: {element.shortcut}")
        
        return '. '.join(parts)
    
    async def announce_focused_element(self) -> None:
        """Announce currently focused element"""
        if not self.keyboard.focused_element:
            return
        
        element = self.accessible_elements.get(self.keyboard.focused_element)
        if element:
            await self.screen_reader.announce_element(element)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of accessibility layer"""
    
    # Create accessibility settings
    settings = AccessibilitySettings(
        screen_reader=True,
        high_contrast=True,
        large_text=True,
        captions_enabled=True,
        simplified_ui=True
    )
    
    # Initialize accessibility manager
    manager = AccessibilityManager(settings)
    await manager.initialize()
    
    # Register accessible elements
    elements = [
        AccessibleElement(
            element_id='btn_send',
            element_type='button',
            label='Send Button',
            description='Click to send message',
            role='button',
            shortcut='Ctrl+Enter'
        ),
        AccessibleElement(
            element_id='input_message',
            element_type='text_input',
            label='Message Input',
            description='Type your message here',
            role='textbox'
        ),
        AccessibleElement(
            element_id='btn_voice',
            element_type='button',
            label='Voice Input',
            description='Click to use voice input',
            role='button',
            shortcut='Ctrl+Space'
        )
    ]
    
    for element in elements:
        manager.register_element(element)
    
    # Test announcements
    print("Testing accessibility announcements:")
    await manager.announce("Welcome to the AI agent interface")
    
    # Test keyboard navigation
    print("\nTesting keyboard navigation:")
    next_element = manager.keyboard.move_focus_next()
    print(f"Focus moved to: {next_element}")
    
    # Test color accessibility
    print("\nTesting color accessibility:")
    colors = manager.get_theme_colors()
    print(f"Theme colors: {colors}")
    
    # Test text simplification
    print("\nTesting text simplification:")
    complex_text = "Please utilize the configuration interface to implement the necessary changes."
    simplified = manager.cognitive.simplify_text(complex_text)
    print(f"Original: {complex_text}")
    print(f"Simplified: {simplified}")
    
    # Test reading time calculation
    print("\nTesting reading time calculation:")
    reading_time = manager.cognitive.calculate_reading_time(complex_text)
    print(f"Estimated reading time: {reading_time:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
