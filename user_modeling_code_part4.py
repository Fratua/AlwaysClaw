"""
OpenClaw Windows 10 - User Modeling System
Part 4: Feedback System & Multi-User Support
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading

# ============================================================================
# USER FEEDBACK INTEGRATION
# ============================================================================

class FeedbackType(Enum):
    EXPLICIT_RATING = "explicit_rating"
    THUMBS_UP_DOWN = "thumbs"
    TEXT_FEEDBACK = "text"
    CORRECTION = "correction"
    SKIP = "skip"
    COMPLETION = "completion"

class FeedbackContext(Enum):
    RESPONSE_QUALITY = "response_quality"
    ACTION_RESULT = "action_result"
    SUGGESTION_RELEVANCE = "suggestion_relevance"
    PREDICTION_ACCURACY = "prediction_accuracy"
    PERSONALIZATION_FIT = "personalization_fit"

@dataclass
class FeedbackItem:
    """Represents a piece of user feedback"""
    feedback_id: str
    user_id: str
    feedback_type: FeedbackType
    context: FeedbackContext
    value: Any
    timestamp: datetime
    related_action: Optional[str] = None
    related_prediction: Optional[str] = None
    metadata: Dict = None


class FeedbackCollectionSystem:
    """Collects and processes user feedback"""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.feedback_history: deque = deque(maxlen=10000)
        self.feedback_handlers: Dict[FeedbackType, List[Callable]] = {}
        
    def register_feedback_handler(self, feedback_type: FeedbackType, handler: Callable):
        """Register a handler for specific feedback type"""
        if feedback_type not in self.feedback_handlers:
            self.feedback_handlers[feedback_type] = []
        self.feedback_handlers[feedback_type].append(handler)
        
    def collect_feedback(
        self, user_id: str, feedback_type: FeedbackType, context: FeedbackContext,
        value: Any, related_action: str = None, related_prediction: str = None,
        metadata: Dict = None
    ) -> FeedbackItem:
        """Collect feedback from user"""
        feedback = FeedbackItem(
            feedback_id=f"{user_id}:{datetime.utcnow().isoformat()}",
            user_id=user_id,
            feedback_type=feedback_type,
            context=context,
            value=value,
            timestamp=datetime.utcnow(),
            related_action=related_action,
            related_prediction=related_prediction,
            metadata=metadata or {}
        )
        
        self.feedback_history.append(feedback)
        self._process_feedback(feedback)
        
        return feedback
        
    def _process_feedback(self, feedback: FeedbackItem):
        """Process collected feedback"""
        handlers = self.feedback_handlers.get(feedback.feedback_type, [])
        for handler in handlers:
            try:
                handler(feedback)
            except Exception as e:
                print(f"Error in feedback handler: {e}")
                
        self._update_profile_with_feedback(feedback)
        
    def _update_profile_with_feedback(self, feedback: FeedbackItem):
        """Update user profile with feedback"""
        profile = self.profile_manager.get_profile(feedback.user_id)
        if not profile:
            return
            
        if "feedback_history" not in profile.learned_patterns:
            profile.learned_patterns["feedback_history"] = []
            
        feedback_entry = {
            "type": feedback.feedback_type.value,
            "context": feedback.context.value,
            "value": feedback.value,
            "timestamp": feedback.timestamp.isoformat(),
            "related_action": feedback.related_action
        }
        
        profile.learned_patterns["feedback_history"].append(feedback_entry)
        profile.learned_patterns["feedback_history"] = profile.learned_patterns["feedback_history"][-500:]
            
        self.profile_manager.update_profile(feedback.user_id, {"learned_patterns": profile.learned_patterns})
        
    def get_feedback_summary(self, user_id: str) -> Dict:
        """Get feedback summary for user"""
        user_feedback = [f for f in self.feedback_history if f.user_id == user_id]
        
        if not user_feedback:
            return {"total_feedback": 0}
            
        positive_feedback = sum(1 for f in user_feedback if f.value in ["positive", True, 4, 5, "up"])
        negative_feedback = sum(1 for f in user_feedback if f.value in ["negative", False, 1, 2, "down"])
        
        return {
            "total_feedback": len(user_feedback),
            "positive_count": positive_feedback,
            "negative_count": negative_feedback,
            "satisfaction_rate": positive_feedback / len(user_feedback) if user_feedback else 0,
            "recent_feedback": [
                {"type": f.feedback_type.value, "context": f.context.value, 
                 "value": f.value, "timestamp": f.timestamp.isoformat()}
                for f in list(user_feedback)[-10:]
            ]
        }


class FeedbackPromptingSystem:
    """Intelligently prompts for feedback"""
    
    def __init__(self, feedback_system):
        self.feedback_system = feedback_system
        self.prompt_cooldowns: Dict[str, datetime] = {}
        self.prompt_history: deque = deque(maxlen=1000)
        
    def should_prompt_for_feedback(self, user_id: str, context: FeedbackContext) -> bool:
        """Determine if we should prompt for feedback"""
        cooldown_key = f"{user_id}:{context.value}"
        last_prompt = self.prompt_cooldowns.get(cooldown_key)
        
        if last_prompt:
            cooldown_hours = 24
            if datetime.utcnow() - last_prompt < timedelta(hours=cooldown_hours):
                return False
                
        recent_prompts = sum(
            1 for p in self.prompt_history
            if p["user_id"] == user_id and p["timestamp"] > datetime.utcnow() - timedelta(hours=1)
        )
        
        if recent_prompts >= 3:
            return False
            
        return True
        
    def prompt_for_feedback(
        self, user_id: str, context: FeedbackContext, prompt_text: str,
        feedback_type: FeedbackType = FeedbackType.THUMBS_UP_DOWN
    ):
        """Prompt user for feedback"""
        if not self.should_prompt_for_feedback(user_id, context):
            return None
            
        cooldown_key = f"{user_id}:{context.value}"
        self.prompt_cooldowns[cooldown_key] = datetime.utcnow()
        
        self.prompt_history.append({
            "user_id": user_id,
            "context": context.value,
            "timestamp": datetime.utcnow()
        })
        
        return {
            "prompt_id": f"{user_id}:{datetime.utcnow().isoformat()}",
            "text": prompt_text,
            "feedback_type": feedback_type.value,
            "context": context.value
        }


# ============================================================================
# MULTI-USER SUPPORT
# ============================================================================

class UserRole(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    READ_PROFILE = "read_profile"
    WRITE_PROFILE = "write_profile"
    READ_PREFERENCES = "read_preferences"
    WRITE_PREFERENCES = "write_preferences"
    EXECUTE_ACTIONS = "execute_actions"
    MANAGE_USERS = "manage_users"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"

ROLE_PERMISSIONS = {
    UserRole.OWNER: [p for p in Permission],
    UserRole.ADMIN: [
        Permission.READ_PROFILE, Permission.WRITE_PROFILE,
        Permission.READ_PREFERENCES, Permission.WRITE_PREFERENCES,
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_USERS,
        Permission.EXPORT_DATA
    ],
    UserRole.USER: [
        Permission.READ_PROFILE, Permission.WRITE_PROFILE,
        Permission.READ_PREFERENCES, Permission.WRITE_PREFERENCES,
        Permission.EXECUTE_ACTIONS, Permission.EXPORT_DATA
    ],
    UserRole.GUEST: [
        Permission.READ_PROFILE, Permission.READ_PREFERENCES
    ]
}

@dataclass
class UserSession:
    """Represents an active user session"""
    session_id: str
    user_id: str
    role: UserRole
    started_at: datetime
    last_active: datetime
    context: Dict


class MultiUserManager:
    """Manages multiple users and their sessions"""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        self.user_groups: Dict[str, Set[str]] = defaultdict(set)
        self.user_roles: Dict[str, UserRole] = {}
        
        self._lock = threading.RLock()
        
    def create_user(self, user_id: str, initial_data: Dict, role: UserRole = UserRole.USER) -> bool:
        """Create a new user"""
        with self._lock:
            if self.profile_manager.get_profile(user_id):
                return False
                
            self.profile_manager.create_profile(user_id, initial_data)
            self.user_roles[user_id] = role
            
            return True
            
    def start_session(self, user_id: str, context: Dict = None) -> Optional[UserSession]:
        """Start a new user session"""
        with self._lock:
            profile = self.profile_manager.get_profile(user_id)
            if not profile:
                return None
                
            session_id = f"{user_id}:{datetime.utcnow().timestamp()}"
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                role=self.user_roles.get(user_id, UserRole.USER),
                started_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                context=context or {}
            )
            
            self.active_sessions[session_id] = session
            self.user_sessions[user_id].add(session_id)
            
            return session
            
    def end_session(self, session_id: str) -> bool:
        """End a user session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
                
            session = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            self.user_sessions[session.user_id].discard(session_id)
            
            return True
            
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get active session"""
        with self._lock:
            return self.active_sessions.get(session_id)
            
    def get_active_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        with self._lock:
            session_ids = self.user_sessions.get(user_id, set())
            return [self.active_sessions[sid] for sid in session_ids if sid in self.active_sessions]
                    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a permission"""
        role = self.user_roles.get(user_id, UserRole.USER)
        return permission in ROLE_PERMISSIONS.get(role, [])
        
    def set_user_role(self, user_id: str, role: UserRole) -> bool:
        """Set user role"""
        with self._lock:
            if not self.profile_manager.get_profile(user_id):
                return False
            self.user_roles[user_id] = role
            return True
            
    def add_user_to_group(self, user_id: str, group_id: str):
        """Add user to a group"""
        self.user_groups[group_id].add(user_id)
        
    def get_group_users(self, group_id: str) -> Set[str]:
        """Get all users in a group"""
        return self.user_groups.get(group_id, set())
        
    def switch_active_user(self, session_id: str, new_user_id: str) -> Optional[UserSession]:
        """Switch to a different user in the same session"""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
                
            if not self.has_permission(session.user_id, Permission.MANAGE_USERS):
                return None
                
            old_user_id = session.user_id
            session.user_id = new_user_id
            session.role = self.user_roles.get(new_user_id, UserRole.USER)
            session.last_active = datetime.utcnow()
            
            self.user_sessions[old_user_id].discard(session_id)
            self.user_sessions[new_user_id].add(session_id)
            
            return session
            
    def cleanup_inactive_sessions(self, max_inactive_minutes: int = 30):
        """Clean up inactive sessions"""
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(minutes=max_inactive_minutes)
            
            sessions_to_remove = [
                sid for sid, session in self.active_sessions.items()
                if session.last_active < cutoff
            ]
            
            for sid in sessions_to_remove:
                self.end_session(sid)
                
    def get_user_context_summary(self, user_id: str) -> Dict:
        """Get summary of user's context across sessions"""
        sessions = self.get_active_user_sessions(user_id)
        
        if not sessions:
            return {}
            
        merged_context = {}
        for session in sessions:
            merged_context.update(session.context)
            
        return {
            "user_id": user_id,
            "active_sessions": len(sessions),
            "merged_context": merged_context,
            "role": self.user_roles.get(user_id, UserRole.USER).value
        }


class UserContextIsolation:
    """Ensures proper isolation between user contexts"""
    
    def __init__(self, multi_user_manager):
        self.multi_user_manager = multi_user_manager
        
    def get_isolated_context(self, user_id: str) -> Dict:
        """Get completely isolated context for user"""
        profile = self.multi_user_manager.profile_manager.get_profile(user_id)
        
        if not profile:
            return {}
            
        return {
            "user_id": user_id,
            "identity": {
                "preferred_name": profile.preferred_name,
                "pronouns": profile.pronouns
            },
            "preferences": {
                "notification": profile.notification_prefs,
                "content": profile.content_prefs,
                "automation": profile.automation_prefs
            },
            "personalized_settings": self._filter_personal_data(profile.learned_patterns)
        }
        
    def _filter_personal_data(self, data: Dict) -> Dict:
        """Filter out potentially personal data"""
        filtered = {}
        
        for key, value in data.items():
            if key in ["frequent_contacts", "email_patterns"]:
                filtered[key] = self._aggregate_contact_data(value)
            else:
                filtered[key] = value
                
        return filtered
        
    def _aggregate_contact_data(self, data: List) -> Dict:
        """Aggregate contact data to remove personal info"""
        return {
            "count": len(data),
            "categories": list(set(d.get("relationship_type", "unknown") for d in data if isinstance(d, dict)))
        }
        
    def validate_cross_user_access(self, requesting_user_id: str, target_user_id: str,
                                   permission: Permission) -> bool:
        """Validate if user can access another user's data"""
        if requesting_user_id == target_user_id:
            return True
            
        if not self.multi_user_manager.has_permission(requesting_user_id, permission):
            return False
            
        for group_id, users in self.multi_user_manager.user_groups.items():
            if (requesting_user_id in users and target_user_id in users and
                self.multi_user_manager.user_roles.get(requesting_user_id) in [UserRole.ADMIN, UserRole.OWNER]):
                return True
                
        return False


class SharedContextManager:
    """Manages shared contexts between users (e.g., teams)"""
    
    def __init__(self, multi_user_manager):
        self.multi_user_manager = multi_user_manager
        self.shared_contexts: Dict[str, Dict] = {}
        
    def create_shared_context(self, context_id: str, owner_id: str, initial_data: Dict):
        """Create a new shared context"""
        self.shared_contexts[context_id] = {
            "owner_id": owner_id,
            "members": {owner_id},
            "data": initial_data,
            "created_at": datetime.utcnow()
        }
        
    def add_member_to_context(self, context_id: str, owner_id: str, member_id: str) -> bool:
        """Add a member to a shared context"""
        if context_id not in self.shared_contexts:
            return False
            
        context = self.shared_contexts[context_id]
        
        if context["owner_id"] != owner_id:
            return False
            
        context["members"].add(member_id)
        return True
        
    def get_shared_context(self, context_id: str, user_id: str) -> Optional[Dict]:
        """Get shared context if user is a member"""
        if context_id not in self.shared_contexts:
            return None
            
        context = self.shared_contexts[context_id]
        
        if user_id not in context["members"]:
            return None
            
        return context["data"]
        
    def update_shared_context(self, context_id: str, user_id: str, updates: Dict) -> bool:
        """Update shared context"""
        if context_id not in self.shared_contexts:
            return False
            
        context = self.shared_contexts[context_id]
        
        if user_id not in context["members"]:
            return False
            
        context["data"].update(updates)
        context["last_updated"] = datetime.utcnow()
        context["last_updated_by"] = user_id
        
        return True
