"""
OpenClaw Windows 10 - User Modeling System
Part 3: Adaptation, Privacy & Consent
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import json

# ============================================================================
# USER ADAPTATION
# ============================================================================

class AdaptationTrigger(Enum):
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_SIGNAL = "implicit_signal"
    PATTERN_DETECTED = "pattern_detected"
    CONTEXT_CHANGE = "context_change"
    TIME_BASED = "time_based"
    ERROR_OCCURRED = "error_occurred"

class AdaptationPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class AdaptationAction:
    """Represents an adaptation to be made"""
    action_id: str
    trigger: AdaptationTrigger
    priority: AdaptationPriority
    target_aspect: str
    current_value: Any
    proposed_value: Any
    confidence: float
    reasoning: str
    auto_apply: bool
    requires_confirmation: bool


class UserAdaptationEngine:
    """Manages user adaptation through continuous learning"""
    
    def __init__(self, profile_manager, preference_engine, personalization_engine):
        self.profile_manager = profile_manager
        self.preference_engine = preference_engine
        self.personalization_engine = personalization_engine
        
        self.adaptation_queue: List[AdaptationAction] = []
        self.applied_adaptations: List[Dict] = []
        self.rejected_adaptations: List[Dict] = []
        
        self.adaptation_strategies: Dict[str, Callable] = {}
        self.confirmation_callbacks: Dict[str, Callable] = {}
        
        self.auto_apply_threshold = 0.9
        self.confirmation_threshold = 0.7
        
    def register_adaptation_strategy(self, aspect: str, strategy: Callable):
        """Register an adaptation strategy for an aspect"""
        self.adaptation_strategies[aspect] = strategy
        
    def register_confirmation_callback(self, action_type: str, callback: Callable):
        """Register callback for confirmation requests"""
        self.confirmation_callbacks[action_type] = callback
        
    def propose_adaptation(self, action: AdaptationAction):
        """Propose a new adaptation"""
        self.adaptation_queue.append(action)
        self.adaptation_queue.sort(key=lambda a: a.priority.value)
        
        if action.auto_apply and action.confidence >= self.auto_apply_threshold:
            self._apply_adaptation(action)
        elif action.confidence >= self.confirmation_threshold:
            self._request_confirmation(action)
            
    def process_adaptation_queue(self):
        """Process pending adaptations"""
        for action in self.adaptation_queue[:]:
            if action.auto_apply and action.confidence >= self.auto_apply_threshold:
                self._apply_adaptation(action)
                self.adaptation_queue.remove(action)
                
    def _apply_adaptation(self, action: AdaptationAction) -> bool:
        """Apply an adaptation"""
        try:
            strategy = self.adaptation_strategies.get(action.target_aspect)
            success = strategy(action) if strategy else self._default_apply_adaptation(action)
            
            if success:
                self.applied_adaptations.append({
                    "action": action,
                    "applied_at": datetime.utcnow().isoformat()
                })
                self._update_profile_with_adaptation(action)
                
            return success
            
        except Exception as e:
            print(f"Error applying adaptation: {e}")
            return False
            
    def _default_apply_adaptation(self, action: AdaptationAction) -> bool:
        """Default adaptation application"""
        user_id = action.action_id.split(":")[0] if ":" in action.action_id else "default"
        
        category, attribute = self.personalization_engine._parse_aspect(action.target_aspect)
        
        self.preference_engine.record_explicit_feedback(
            user_id=user_id,
            category=category,
            attribute=attribute,
            value=action.proposed_value,
            rating=5 if action.confidence > 0.8 else 4
        )
        
        return True
        
    def _request_confirmation(self, action: AdaptationAction):
        """Request user confirmation for adaptation"""
        action_type = action.trigger.value
        
        if action_type in self.confirmation_callbacks:
            confirmed = self.confirmation_callbacks[action_type](action)
            
            if confirmed:
                self._apply_adaptation(action)
            else:
                self.rejected_adaptations.append({
                    "action": action,
                    "rejected_at": datetime.utcnow().isoformat()
                })
                
    def _update_profile_with_adaptation(self, action: AdaptationAction):
        """Update user profile with applied adaptation"""
        user_id = action.action_id.split(":")[0] if ":" in action.action_id else "default"
        
        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            return
            
        if "adaptation_history" not in profile.learned_patterns:
            profile.learned_patterns["adaptation_history"] = []
            
        profile.learned_patterns["adaptation_history"].append({
            "aspect": action.target_aspect,
            "from_value": action.current_value,
            "to_value": action.proposed_value,
            "confidence": action.confidence,
            "applied_at": datetime.utcnow().isoformat()
        })
        
        profile.learned_patterns["adaptation_history"] = profile.learned_patterns["adaptation_history"][-100:]
            
        self.profile_manager.update_profile(user_id, {"learned_patterns": profile.learned_patterns})
        
    def get_adaptation_summary(self, user_id: str) -> Dict:
        """Get summary of adaptations for user"""
        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            return {}
            
        history = profile.learned_patterns.get("adaptation_history", [])
        
        return {
            "total_adaptations": len(history),
            "recent_adaptations": history[-10:],
            "adaptation_rate": len(history) / max(1, (datetime.utcnow() - profile.created_at).days),
            "top_adapted_aspects": self._get_top_adapted_aspects(history)
        }
        
    def _get_top_adapted_aspects(self, history: List[Dict]) -> List[Dict]:
        """Get most frequently adapted aspects"""
        aspect_counts = {}
        
        for entry in history:
            aspect = entry["aspect"]
            if aspect not in aspect_counts:
                aspect_counts[aspect] = {"count": 0, "avg_confidence": 0}
            aspect_counts[aspect]["count"] += 1
            aspect_counts[aspect]["avg_confidence"] += entry["confidence"]
            
        for aspect in aspect_counts:
            aspect_counts[aspect]["avg_confidence"] /= aspect_counts[aspect]["count"]
            
        sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1]["count"], reverse=True)
        
        return [{"aspect": a, **data} for a, data in sorted_aspects[:5]]


# ============================================================================
# PRIVACY CONTROLS
# ============================================================================

class DataSensitivity(Enum):
    PUBLIC = 0
    INTERNAL = 1
    SENSITIVE = 2
    RESTRICTED = 3
    CRITICAL = 4

class DataCategory(Enum):
    IDENTITY = "identity"
    COMMUNICATION = "communication"
    BEHAVIORAL = "behavioral"
    PREFERENCES = "preferences"
    CONTEXTUAL = "contextual"
    DERIVED = "derived"

@dataclass
class DataField:
    """Represents a data field with privacy metadata"""
    name: str
    category: DataCategory
    sensitivity: DataSensitivity
    retention_days: int
    encrypt_at_rest: bool
    encrypt_in_transit: bool
    anonymizable: bool
    user_controllable: bool


class PrivacySchema:
    """Defines privacy schema for all user data"""
    
    FIELDS = {
        "preferred_name": DataField(
            name="preferred_name", category=DataCategory.IDENTITY,
            sensitivity=DataSensitivity.INTERNAL, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=False, user_controllable=True
        ),
        "full_name": DataField(
            name="full_name", category=DataCategory.IDENTITY,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=False, user_controllable=True
        ),
        "email_addresses": DataField(
            name="email_addresses", category=DataCategory.COMMUNICATION,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
        "phone_numbers": DataField(
            name="phone_numbers", category=DataCategory.COMMUNICATION,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
        "work_hours": DataField(
            name="work_hours", category=DataCategory.BEHAVIORAL,
            sensitivity=DataSensitivity.INTERNAL, retention_days=90,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
        "application_usage": DataField(
            name="application_usage", category=DataCategory.BEHAVIORAL,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=30,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
        "notification_prefs": DataField(
            name="notification_prefs", category=DataCategory.PREFERENCES,
            sensitivity=DataSensitivity.INTERNAL, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=False, user_controllable=True
        ),
        "content_prefs": DataField(
            name="content_prefs", category=DataCategory.PREFERENCES,
            sensitivity=DataSensitivity.INTERNAL, retention_days=365,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=False, user_controllable=True
        ),
        "learned_patterns": DataField(
            name="learned_patterns", category=DataCategory.DERIVED,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=90,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
        "behavioral_insights": DataField(
            name="behavioral_insights", category=DataCategory.DERIVED,
            sensitivity=DataSensitivity.SENSITIVE, retention_days=60,
            encrypt_at_rest=True, encrypt_in_transit=True,
            anonymizable=True, user_controllable=True
        ),
    }


class PrivacyManager:
    """Manages privacy controls for user data"""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.schema = PrivacySchema()
        self.user_privacy_settings: Dict[str, Dict] = {}
        
    def get_field_privacy(self, field_name: str) -> Optional[DataField]:
        """Get privacy metadata for a field"""
        return self.schema.FIELDS.get(field_name)
        
    def set_user_privacy_setting(self, user_id: str, setting: str, value: Any):
        """Set a privacy setting for user"""
        if user_id not in self.user_privacy_settings:
            self.user_privacy_settings[user_id] = self._default_privacy_settings()
            
        self.user_privacy_settings[user_id][setting] = value
        
        profile = self.profile_manager.get_profile(user_id)
        if profile:
            profile.privacy_settings[setting] = value
            self.profile_manager.update_profile(user_id, {"privacy_settings": profile.privacy_settings})
            
    def get_user_privacy_setting(self, user_id: str, setting: str, default: Any = None) -> Any:
        """Get a privacy setting for user"""
        if user_id in self.user_privacy_settings:
            return self.user_privacy_settings[user_id].get(setting, default)
            
        profile = self.profile_manager.get_profile(user_id)
        if profile and setting in profile.privacy_settings:
            return profile.privacy_settings[setting]
            
        return default
        
    def _default_privacy_settings(self) -> Dict:
        """Get default privacy settings"""
        return {
            "learning_enabled": True,
            "pattern_sharing": False,
            "cloud_sync": False,
            "local_encryption": True,
            "data_retention_days": 90,
            "sensitive_topics": [],
            "auto_delete_after_retention": True,
            "allow_anonymous_analytics": True,
            "third_party_sharing": False
        }
        
    def can_collect_data(self, user_id: str, field_name: str) -> bool:
        """Check if data collection is allowed"""
        field = self.get_field_privacy(field_name)
        if not field:
            return False
            
        if not self.get_user_privacy_setting(user_id, "learning_enabled", True):
            return False
            
        learning_level = self.get_user_privacy_setting(user_id, "learning_level", "adaptive")
        
        if learning_level == "minimal" and field.sensitivity.value > DataSensitivity.INTERNAL.value:
            return False
        if learning_level == "none":
            return False
            
        return True
        
    def should_encrypt_field(self, field_name: str) -> bool:
        """Check if field should be encrypted"""
        field = self.get_field_privacy(field_name)
        return field.encrypt_at_rest if field else True
        
    def get_retention_period(self, field_name: str) -> int:
        """Get retention period for field in days"""
        field = self.get_field_privacy(field_name)
        return field.retention_days if field else 30
        
    def anonymize_field(self, field_name: str, value: Any) -> Any:
        """Anonymize a field value"""
        field = self.get_field_privacy(field_name)
        
        if not field or not field.anonymizable:
            return value
            
        if field.category == DataCategory.COMMUNICATION:
            if isinstance(value, str):
                return hashlib.sha256(value.encode()).hexdigest()[:16]
            elif isinstance(value, list):
                return [hashlib.sha256(v.encode()).hexdigest()[:16] if isinstance(v, str) else v for v in value]
                        
        elif field.category == DataCategory.BEHAVIORAL:
            if isinstance(value, dict) and "timestamp" in value:
                value["timestamp"] = "aggregated"
                
        return value
        
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (GDPR compliance)"""
        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            return {}
            
        return {
            "export_metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "version": "1.0"
            },
            "identity": {
                "preferred_name": profile.preferred_name,
                "full_name": profile.full_name,
                "pronouns": profile.pronouns
            },
            "communication": {
                "email_addresses": profile.email_addresses,
                "phone_numbers": profile.phone_numbers
            },
            "preferences": {
                "notification": profile.notification_prefs,
                "content": profile.content_prefs,
                "automation": profile.automation_prefs,
                "ui": profile.ui_prefs
            },
            "learned_patterns": profile.learned_patterns,
            "behavioral_insights": profile.behavioral_insights,
            "privacy_settings": profile.privacy_settings
        }
        
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (GDPR right to erasure)"""
        try:
            import shutil
            profile_path = self.profile_manager.base_path / user_id
            
            if profile_path.exists():
                shutil.rmtree(profile_path)
                
            if user_id in self.profile_manager._profile_cache:
                del self.profile_manager._profile_cache[user_id]
            if user_id in self.user_privacy_settings:
                del self.user_privacy_settings[user_id]
                
            return True
            
        except Exception as e:
            print(f"Error deleting user data: {e}")
            return False


# ============================================================================
# CONSENT MANAGEMENT
# ============================================================================

class ConsentManager:
    """Manages user consent for data processing"""
    
    CONSENT_TYPES = {
        "data_collection": {
            "description": "Collect and store user data for personalization",
            "required": True,
            "granular": True
        },
        "behavioral_learning": {
            "description": "Learn from user behavior patterns",
            "required": False,
            "granular": True
        },
        "communication_analysis": {
            "description": "Analyze emails and messages for patterns",
            "required": False,
            "granular": True
        },
        "cloud_sync": {
            "description": "Sync data across devices via cloud",
            "required": False,
            "granular": False
        },
        "anonymous_analytics": {
            "description": "Share anonymized usage data for improvement",
            "required": False,
            "granular": False
        }
    }
    
    def __init__(self, privacy_manager):
        self.privacy_manager = privacy_manager
        self.user_consents: Dict[str, Dict[str, bool]] = {}
        
    def get_consent_status(self, user_id: str, consent_type: str) -> bool:
        """Get consent status for a specific type"""
        if user_id not in self.user_consents:
            return False
        return self.user_consents[user_id].get(consent_type, False)
        
    def record_consent(self, user_id: str, consent_type: str, granted: bool):
        """Record user consent decision"""
        if user_id not in self.user_consents:
            self.user_consents[user_id] = {}
            
        self.user_consents[user_id][consent_type] = granted
        
        profile = self.privacy_manager.profile_manager.get_profile(user_id)
        if profile:
            if "consents" not in profile.privacy_settings:
                profile.privacy_settings["consents"] = {}
                
            profile.privacy_settings["consents"][consent_type] = {
                "granted": granted,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.privacy_manager.profile_manager.update_profile(user_id, {
                "privacy_settings": profile.privacy_settings
            })
            
    def withdraw_consent(self, user_id: str, consent_type: str):
        """Withdraw previously granted consent"""
        self.record_consent(user_id, consent_type, False)
        
        if consent_type == "data_collection":
            self.privacy_manager.delete_user_data(user_id)
