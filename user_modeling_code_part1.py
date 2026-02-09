"""
OpenClaw Windows 10 - User Modeling System
Part 1: Profile Management & Preference Learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import json
import hashlib
import numpy as np

# ============================================================================
# PROFILE MANAGEMENT
# ============================================================================

class ProfileVersion(Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"

@dataclass
class UserProfile:
    """Core user profile data structure"""
    profile_id: str
    version: ProfileVersion = ProfileVersion.V1_0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Identity
    preferred_name: str = ""
    full_name: str = ""
    pronouns: str = ""
    voice_preference: str = "neutral"
    timezone: str = "UTC"
    locale: str = "en-US"
    
    # Communication
    email_addresses: List[Dict] = field(default_factory=list)
    phone_numbers: List[Dict] = field(default_factory=list)
    preferred_contact_method: str = "email"
    
    # Work Patterns
    work_hours: Dict = field(default_factory=dict)
    focus_hours: List[Dict] = field(default_factory=list)
    meeting_preferences: Dict = field(default_factory=dict)
    
    # Preferences
    notification_prefs: Dict = field(default_factory=dict)
    content_prefs: Dict = field(default_factory=dict)
    automation_prefs: Dict = field(default_factory=dict)
    ui_prefs: Dict = field(default_factory=dict)
    
    # Learned Data
    learned_patterns: Dict = field(default_factory=dict)
    contextual_memory: Dict = field(default_factory=dict)
    behavioral_insights: Dict = field(default_factory=dict)
    
    # Privacy
    privacy_settings: Dict = field(default_factory=dict)


class UserProfileManager:
    """Manages user profile lifecycle and persistence"""
    
    def __init__(self, base_path: str = "~/.openclaw/profiles"):
        self.base_path = Path(base_path).expanduser()
        self._profile_cache: Dict[str, UserProfile] = {}
        
    def _load_or_generate_key(self) -> bytes:
        """Load existing encryption key or generate new one"""
        key_path = self.base_path / ".master_key"
        if key_path.exists():
            return key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(key)
            key_path.chmod(0o600)
            return key
            
    def create_profile(self, user_id: str, initial_data: Dict) -> UserProfile:
        """Create new user profile with initial data"""
        profile = UserProfile(
            profile_id=user_id,
            preferred_name=initial_data.get("preferred_name", ""),
            full_name=initial_data.get("full_name", ""),
            timezone=initial_data.get("timezone", "UTC"),
            locale=initial_data.get("locale", "en-US")
        )
        self._save_profile(profile)
        return profile
        
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile with caching"""
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
            
        profile_path = self.base_path / user_id / "USER.md"
        if not profile_path.exists():
            return None
            
        profile = self._load_profile(profile_path)
        self._profile_cache[user_id] = profile
        return profile
        
    def update_profile(self, user_id: str, updates: Dict) -> UserProfile:
        """Update profile with new data"""
        profile = self.get_profile(user_id)
        if not profile:
            raise ValueError(f"Profile not found: {user_id}")
            
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
                
        profile.last_updated = datetime.utcnow()
        self._save_profile(profile)
        self._profile_cache[user_id] = profile
        return profile
        
    def _save_profile(self, profile: UserProfile):
        """Persist profile to encrypted storage"""
        profile_path = self.base_path / profile.profile_id
        profile_path.mkdir(parents=True, exist_ok=True)
        
        data = self._profile_to_dict(profile)
        
        # Write to USER.md
        user_md_path = profile_path / "USER.md"
        with open(user_md_path, 'w') as f:
            f.write("---\n")
            f.write(json.dumps(data, indent=2, default=str))
            f.write("\n---\n")
            
    def _load_profile(self, profile_path: Path) -> UserProfile:
        """Load profile from storage"""
        with open(profile_path, 'r') as f:
            content = f.read()
            # Extract JSON between --- markers
            parts = content.split("---")
            if len(parts) >= 2:
                data = json.loads(parts[1])
                return self._dict_to_profile(data)
        return None
        
    def _profile_to_dict(self, profile: UserProfile) -> Dict:
        """Convert profile to dictionary"""
        return {
            "profile_metadata": {
                "version": profile.version.value,
                "created_at": profile.created_at.isoformat(),
                "last_updated": profile.last_updated.isoformat(),
                "profile_id": profile.profile_id
            },
            "identity": {
                "preferred_name": profile.preferred_name,
                "full_name": profile.full_name,
                "pronouns": profile.pronouns,
                "voice_preference": profile.voice_preference,
                "timezone": profile.timezone,
                "locale": profile.locale
            },
            "communication": {
                "email_addresses": profile.email_addresses,
                "phone_numbers": profile.phone_numbers,
                "preferred_contact_method": profile.preferred_contact_method
            },
            "work_patterns": {
                "work_hours": profile.work_hours,
                "focus_hours": profile.focus_hours,
                "meeting_preferences": profile.meeting_preferences
            },
            "preferences": {
                "notification": profile.notification_prefs,
                "content": profile.content_prefs,
                "automation": profile.automation_prefs,
                "ui": profile.ui_prefs
            },
            "learned_patterns": profile.learned_patterns,
            "contextual_memory": profile.contextual_memory,
            "behavioral_insights": profile.behavioral_insights,
            "privacy_settings": profile.privacy_settings
        }
        
    def _dict_to_profile(self, data: Dict) -> UserProfile:
        """Convert dictionary to profile"""
        profile = UserProfile(profile_id=data["profile_metadata"]["profile_id"])
        # Populate fields from data
        identity = data.get("identity", {})
        profile.preferred_name = identity.get("preferred_name", "")
        profile.full_name = identity.get("full_name", "")
        profile.pronouns = identity.get("pronouns", "")
        profile.voice_preference = identity.get("voice_preference", "neutral")
        profile.timezone = identity.get("timezone", "UTC")
        profile.locale = identity.get("locale", "en-US")
        
        comm = data.get("communication", {})
        profile.email_addresses = comm.get("email_addresses", [])
        profile.phone_numbers = comm.get("phone_numbers", [])
        profile.preferred_contact_method = comm.get("preferred_contact_method", "email")
        
        work = data.get("work_patterns", {})
        profile.work_hours = work.get("work_hours", {})
        profile.focus_hours = work.get("focus_hours", [])
        profile.meeting_preferences = work.get("meeting_preferences", {})
        
        prefs = data.get("preferences", {})
        profile.notification_prefs = prefs.get("notification", {})
        profile.content_prefs = prefs.get("content", {})
        profile.automation_prefs = prefs.get("automation", {})
        profile.ui_prefs = prefs.get("ui", {})
        
        profile.learned_patterns = data.get("learned_patterns", {})
        profile.contextual_memory = data.get("contextual_memory", {})
        profile.behavioral_insights = data.get("behavioral_insights", {})
        profile.privacy_settings = data.get("privacy_settings", {})
        
        return profile


# ============================================================================
# PREFERENCE LEARNING
# ============================================================================

@dataclass
class PreferenceSignal:
    """Represents a user preference signal"""
    category: str
    attribute: str
    value: Any
    confidence: float
    timestamp: datetime
    source: str
    context: Dict = None


class PreferenceLearningEngine:
    """
    Multi-modal preference learning system combining:
    - Explicit feedback (ratings, direct input)
    - Implicit signals (behavior patterns, dwell time)
    - Inferred preferences (contextual analysis)
    """
    
    def __init__(self, profile_manager: UserProfileManager):
        self.profile_manager = profile_manager
        self.signal_history: List[PreferenceSignal] = []
        self.preference_models: Dict[str, Any] = {}
        self.confidence_threshold = 0.7
        
        self.signal_weights = {
            "explicit": 1.0,
            "implicit": 0.6,
            "inferred": 0.3
        }
        
    def record_explicit_feedback(
        self, user_id: str, category: str, attribute: str, 
        value: Any, rating: int
    ):
        """Record direct user feedback (1-5 scale)"""
        confidence = min(1.0, (rating / 5.0) * self.signal_weights["explicit"])
        signal = PreferenceSignal(
            category=category,
            attribute=attribute,
            value=value,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            source="explicit",
            context={"rating": rating}
        )
        self._process_signal(user_id, signal)
        
    def record_implicit_signal(
        self, user_id: str, category: str, attribute: str,
        value: Any, strength: float
    ):
        """Record implicit behavioral signal (0.0-1.0 strength)"""
        signal = PreferenceSignal(
            category=category,
            attribute=attribute,
            value=value,
            confidence=strength * self.signal_weights["implicit"],
            timestamp=datetime.utcnow(),
            source="implicit"
        )
        self._process_signal(user_id, signal)
        
    def _process_signal(self, user_id: str, signal: PreferenceSignal):
        """Process and integrate preference signal"""
        self.signal_history.append(signal)
        self._update_preference_model(user_id, signal)
        
        if self._is_preference_stable(user_id, signal.category, signal.attribute):
            self._persist_preference(user_id, signal)
            
    def _update_preference_model(self, user_id: str, signal: PreferenceSignal):
        """Update internal preference model with new signal"""
        key = f"{user_id}:{signal.category}:{signal.attribute}"
        
        if key not in self.preference_models:
            self.preference_models[key] = {
                "values": defaultdict(list),
                "timestamps": [],
                "confidence_history": []
            }
            
        model = self.preference_models[key]
        model["values"][signal.value].append({
            "confidence": signal.confidence,
            "timestamp": signal.timestamp,
            "source": signal.source
        })
        model["timestamps"].append(signal.timestamp)
        model["confidence_history"].append(signal.confidence)
        
    def _is_preference_stable(
        self, user_id: str, category: str, attribute: str,
        min_signals: int = 5, stability_threshold: float = 0.8
    ) -> bool:
        """Determine if preference has stabilized"""
        key = f"{user_id}:{category}:{attribute}"
        
        if key not in self.preference_models:
            return False
            
        model = self.preference_models[key]
        total_signals = sum(len(v) for v in model["values"].values())
        
        if total_signals < min_signals:
            return False
            
        value_counts = {k: len(v) for k, v in model["values"].items()}
        max_count = max(value_counts.values())
        consistency = max_count / total_signals
        
        return consistency >= stability_threshold
        
    def _persist_preference(self, user_id: str, signal: PreferenceSignal):
        """Persist stable preference to user profile"""
        key = f"{user_id}:{signal.category}:{signal.attribute}"
        model = self.preference_models[key]
        
        best_value = max(model["values"].items(), key=lambda x: len(x[1]))[0]
        avg_confidence = np.mean(model["confidence_history"])
        
        updates = {
            f"{signal.category}_prefs": {
                signal.attribute: {
                    "value": best_value,
                    "confidence": avg_confidence,
                    "learned_at": datetime.utcnow().isoformat(),
                    "signal_count": sum(len(v) for v in model["values"].values())
                }
            }
        }
        
        self.profile_manager.update_profile(user_id, updates)
        
    def get_preference(
        self, user_id: str, category: str, attribute: str, default: Any = None
    ) -> Dict:
        """Get current preference with confidence"""
        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            return {"value": default, "confidence": 0.0, "source": "default"}
            
        prefs = getattr(profile, f"{category}_prefs", {})
        if attribute in prefs:
            return {
                "value": prefs[attribute]["value"],
                "confidence": prefs[attribute]["confidence"],
                "source": "learned"
            }
            
        return {"value": default, "confidence": 0.0, "source": "default"}


# ============================================================================
# CONTEXTUAL PREFERENCE LEARNING
# ============================================================================

class ContextualPreferenceLearner:
    """Learns preferences that vary by context"""
    
    CONTEXT_DIMENSIONS = [
        "time_of_day", "day_of_week", "location",
        "active_application", "current_task", "stress_level", "meeting_status"
    ]
    
    def __init__(self, preference_engine: PreferenceLearningEngine):
        self.preference_engine = preference_engine
        self.context_profiles: Dict[str, Dict] = {}
        
    def detect_context(self) -> Dict[str, Any]:
        """Detect current context dimensions."""
        now = datetime.now()
        return {
            "time_of_day": self._classify_time_of_day(now),
            "day_of_week": now.strftime("%A").lower(),
            "location": self._detect_location(),
            "active_application": self._detect_active_application(),
            "current_task": self._infer_current_task(),
            "stress_level": self._estimate_stress_level(),
            "meeting_status": self._check_meeting_status()
        }

    def _detect_location(self) -> str:
        """Detect location from network/system info."""
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command', '(Get-NetConnectionProfile).Name'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                network = result.stdout.strip()
                if any(kw in network.lower() for kw in ['office', 'corp', 'work']):
                    return 'office'
                elif any(kw in network.lower() for kw in ['home', 'personal']):
                    return 'home'
                return network
        except (OSError, subprocess.TimeoutExpired):
            pass
        return "unknown"

    def _detect_active_application(self) -> Optional[str]:
        """Detect the currently active application."""
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command',
                 '(Get-Process | Where-Object {$_.MainWindowTitle -ne ""} | Select-Object -First 1).ProcessName'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def _infer_current_task(self) -> Optional[str]:
        """Infer current task from active window title."""
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command',
                 '(Get-Process | Where-Object {$_.MainWindowTitle -ne ""} | Select-Object -First 1).MainWindowTitle'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                title = result.stdout.strip()
                if any(kw in title.lower() for kw in ['code', 'visual studio', 'vim', 'pycharm']):
                    return 'coding'
                elif any(kw in title.lower() for kw in ['outlook', 'gmail', 'mail']):
                    return 'email'
                elif any(kw in title.lower() for kw in ['teams', 'zoom', 'slack', 'discord']):
                    return 'communication'
                elif any(kw in title.lower() for kw in ['chrome', 'firefox', 'edge', 'browser']):
                    return 'browsing'
                return 'other'
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def _estimate_stress_level(self) -> Optional[float]:
        """Estimate stress level from system activity patterns."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            # High CPU + many processes may indicate stress/multitasking
            proc_count = len(list(psutil.process_iter()))
            stress = min(1.0, (cpu / 100) * 0.5 + min(proc_count / 200, 0.5))
            return round(stress, 2)
        except ImportError:
            return None

    def _check_meeting_status(self) -> str:
        """Check if user is in a meeting."""
        try:
            import psutil
            meeting_apps = ['teams', 'zoom', 'webex', 'gotomeeting', 'skype']
            for proc in psutil.process_iter(['name']):
                try:
                    name = (proc.info['name'] or '').lower()
                    if any(app in name for app in meeting_apps):
                        return 'in_meeting'
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            pass
        return "free"
        
    def _classify_time_of_day(self, dt: datetime) -> str:
        """Classify time into periods"""
        hour = dt.hour
        if 5 <= hour < 9: return "early_morning"
        elif 9 <= hour < 12: return "morning"
        elif 12 <= hour < 14: return "lunch"
        elif 14 <= hour < 17: return "afternoon"
        elif 17 <= hour < 20: return "evening"
        else: return "night"
        
    def learn_contextual_preference(
        self, user_id: str, base_preference: str,
        context: Dict[str, Any], observed_value: Any
    ):
        """Learn how preferences change with context"""
        context_hash = self._hash_context(context)
        
        if user_id not in self.context_profiles:
            self.context_profiles[user_id] = {}
            
        if context_hash not in self.context_profiles[user_id]:
            self.context_profiles[user_id][context_hash] = {
                "context": context,
                "preference_modifiers": defaultdict(list)
            }
            
        profile = self.context_profiles[user_id][context_hash]
        profile["preference_modifiers"][base_preference].append({
            "value": observed_value,
            "timestamp": datetime.utcnow()
        })
        
    def get_contextual_preference(
        self, user_id: str, base_preference: str, context: Dict[str, Any] = None
    ) -> Any:
        """Get preference adjusted for current context"""
        if context is None:
            context = self.detect_context()
            
        base = self.preference_engine.get_preference(user_id, "content", base_preference)
        
        context_hash = self._hash_context(context)
        if (user_id in self.context_profiles and 
            context_hash in self.context_profiles[user_id]):
            
            modifiers = self.context_profiles[user_id][context_hash]["preference_modifiers"]
            if base_preference in modifiers:
                values = [m["value"] for m in modifiers[base_preference]]
                return max(set(values), key=values.count)
                
        return base["value"]
        
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hashable context identifier"""
        key_dims = ["time_of_day", "day_of_week", "meeting_status"]
        key_context = {k: context.get(k) for k in key_dims}
        return hashlib.md5(
            json.dumps(key_context, sort_keys=True).encode()
        ).hexdigest()[:16]
