"""
OpenClaw Windows 10 - User Modeling System
Integration: Main System Integration Point
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

# Import all components
from user_modeling_code_part1 import (
    UserProfileManager, PreferenceLearningEngine, ContextualPreferenceLearner
)
from user_modeling_code_part2 import (
    PatternRecognitionOrchestrator, PersonalizationEngine, 
    ResponsePersonalizer, UIPersonalizer, PersonalizationContext
)
from user_modeling_code_part3 import (
    UserAdaptationEngine, AdaptationAction, AdaptationTrigger, AdaptationPriority,
    PrivacyManager, ConsentManager
)
from user_modeling_code_part4 import (
    FeedbackCollectionSystem, FeedbackType, FeedbackContext,
    MultiUserManager, UserContextIsolation, SharedContextManager
)


class ProactivePersonalizationEngine:
    """Anticipates user needs and proactively personalizes experience"""
    
    def __init__(self, personalization_engine, pattern_orchestrator):
        self.personalization_engine = personalization_engine
        self.pattern_orchestrator = pattern_orchestrator
        
        self.proactive_suggestions: List[Dict] = []
        self.suggestion_handlers: Dict[str, callable] = {}
        
    def register_suggestion_handler(self, suggestion_type: str, handler: callable):
        """Register handler for proactive suggestions"""
        self.suggestion_handlers[suggestion_type] = handler
        
    def analyze_context(self, user_id: str, current_context: Dict) -> List[Dict]:
        """Analyze context and generate proactive suggestions"""
        suggestions = []
        
        temporal_suggestions = self._check_temporal_patterns(user_id, current_context)
        suggestions.extend(temporal_suggestions)
        
        sequential_suggestions = self._check_sequential_patterns(user_id, current_context)
        suggestions.extend(sequential_suggestions)
        
        task_suggestions = self._check_task_patterns(user_id, current_context)
        suggestions.extend(task_suggestions)
        
        return self._rank_suggestions(suggestions)
        
    def _check_temporal_patterns(self, user_id: str, context: Dict) -> List[Dict]:
        """Check for time-based proactive suggestions"""
        suggestions = []
        
        patterns = self.pattern_orchestrator.get_user_patterns(
            user_id, pattern_type="temporal", min_confidence=0.8
        )
        
        now = datetime.now()
        current_time = now.hour * 60 + now.minute
        
        for pattern in patterns:
            if "mean_time" in pattern.metadata:
                mean_time = pattern.metadata["mean_time"]
                hour, minute = map(int, mean_time.split(":"))
                pattern_time = hour * 60 + minute
                
                time_diff = abs(current_time - pattern_time)
                
                if time_diff <= 15:
                    suggestion = {
                        "type": "temporal",
                        "title": f"Time for {pattern.description}",
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "action": self._infer_action_from_pattern(pattern),
                        "metadata": pattern.metadata
                    }
                    suggestions.append(suggestion)
                    
        return suggestions
        
    def _check_sequential_patterns(self, user_id: str, context: Dict) -> List[Dict]:
        """Check for sequence-based proactive suggestions"""
        suggestions = []
        
        patterns = self.pattern_orchestrator.get_user_patterns(
            user_id, pattern_type="sequential", min_confidence=0.7
        )
        
        recent_actions = context.get("recent_actions", [])
        
        for pattern in patterns:
            sequence = pattern.metadata.get("sequence", [])
            
            if len(recent_actions) >= len(sequence) - 1:
                recent = recent_actions[-(len(sequence)-1):]
                
                if list(recent) == list(sequence[:-1]):
                    next_action = sequence[-1]
                    suggestion = {
                        "type": "sequential",
                        "title": f"Next: {next_action}",
                        "description": f"You usually {next_action} after this",
                        "confidence": pattern.confidence,
                        "action": next_action,
                        "metadata": pattern.metadata
                    }
                    suggestions.append(suggestion)
                    
        return suggestions
        
    def _check_task_patterns(self, user_id: str, context: Dict) -> List[Dict]:
        """Check for task-based proactive suggestions"""
        suggestions = []
        
        profile = self.personalization_engine.profile_manager.get_profile(user_id)
        if not profile:
            return suggestions
            
        recurring_tasks = profile.learned_patterns.get("recurring_tasks", [])
        
        for task in recurring_tasks:
            if self._is_task_due(task, context):
                suggestion = {
                    "type": "task",
                    "title": f"Task reminder: {task['task_pattern']}",
                    "description": f"This task is typically done {task['frequency']}",
                    "confidence": 0.8,
                    "action": task['task_pattern'],
                    "metadata": task
                }
                suggestions.append(suggestion)
                
        return suggestions
        
    def _is_task_due(self, task: Dict, context: Dict) -> bool:
        """Check if a recurring task is due"""
        frequency = task.get("frequency", "daily")
        typical_time = task.get("typical_time", "09:00")
        
        now = datetime.now()
        current_time = f"{now.hour:02d}:{now.minute:02d}"
        
        return abs(self._time_diff(current_time, typical_time)) <= 30
        
    def _time_diff(self, time1: str, time2: str) -> int:
        """Calculate difference in minutes between two times"""
        h1, m1 = map(int, time1.split(":"))
        h2, m2 = map(int, time2.split(":"))
        return abs((h1 * 60 + m1) - (h2 * 60 + m2))
        
    def _infer_action_from_pattern(self, pattern) -> str:
        """Infer suggested action from pattern"""
        description = pattern.description.lower()
        
        if "email" in description:
            return "check_email"
        elif "meeting" in description:
            return "prepare_for_meeting"
        elif "app" in description:
            return "open_application"
        else:
            return "review_context"
            
    def _rank_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Rank suggestions by relevance and confidence"""
        for suggestion in suggestions:
            base_score = suggestion["confidence"]
            
            if suggestion["type"] == "temporal":
                base_score *= 1.2
            elif suggestion["type"] == "task":
                base_score *= 1.1
                
            suggestion["score"] = base_score
            
        return sorted(suggestions, key=lambda x: x["score"], reverse=True)
        
    def execute_suggestion(self, suggestion: Dict, user_id: str) -> bool:
        """Execute a proactive suggestion"""
        suggestion_type = suggestion["type"]
        
        if suggestion_type in self.suggestion_handlers:
            try:
                self.suggestion_handlers[suggestion_type](suggestion)
                return True
            except Exception as e:
                print(f"Error executing suggestion: {e}")
                return False
                
        return False


class UserModelingSystem:
    """Main integration point for all user modeling components"""
    
    def __init__(self, base_path: str = "~/.openclaw/profiles"):
        # Initialize core components
        self.profile_manager = UserProfileManager(base_path)
        
        self.preference_engine = PreferenceLearningEngine(self.profile_manager)
        
        self.pattern_orchestrator = PatternRecognitionOrchestrator(self.profile_manager)
        
        self.personalization_engine = PersonalizationEngine(
            self.profile_manager, self.preference_engine, self.pattern_orchestrator
        )
        
        self.adaptation_engine = UserAdaptationEngine(
            self.profile_manager, self.preference_engine, self.personalization_engine
        )
        
        self.privacy_manager = PrivacyManager(self.profile_manager)
        
        self.consent_manager = ConsentManager(self.privacy_manager)
        
        self.feedback_system = FeedbackCollectionSystem(self.profile_manager)
        
        self.multi_user_manager = MultiUserManager(self.profile_manager)
        
        self.context_isolation = UserContextIsolation(self.multi_user_manager)
        
        self.shared_context_manager = SharedContextManager(self.multi_user_manager)
        
        # Response personalizers
        self.response_personalizer = ResponsePersonalizer(self.personalization_engine)
        
        self.ui_personalizer = UIPersonalizer(self.personalization_engine)
        
        # Proactive personalization
        self.proactive_engine = ProactivePersonalizationEngine(
            self.personalization_engine, self.pattern_orchestrator
        )
        
    def initialize_user(self, user_id: str, initial_data: Dict) -> bool:
        """Initialize a new user in the system"""
        if not self.consent_manager.get_consent_status(user_id, "data_collection"):
            return False
            
        return self.multi_user_manager.create_user(user_id, initial_data)
        
    def process_user_event(self, user_id: str, event_type: str, event_data: Dict):
        """Process a user event through the system"""
        if not self.privacy_manager.can_collect_data(user_id, event_type):
            return
            
        self.pattern_orchestrator.process_event(event_type, event_data)
        self._extract_implicit_signals(user_id, event_type, event_data)
        
    def _extract_implicit_signals(self, user_id: str, event_type: str, event_data: Dict):
        """Extract implicit preference signals from events"""
        if "dwell_time_ms" in event_data:
            dwell_time = event_data["dwell_time_ms"]
            if dwell_time > 5000:
                self.preference_engine.record_implicit_signal(
                    user_id=user_id,
                    category="content",
                    attribute="interest_level",
                    value="high",
                    strength=min(1.0, dwell_time / 30000)
                )
                
        if event_type == "action_completed":
            self.preference_engine.record_implicit_signal(
                user_id=user_id,
                category="automation",
                attribute="action_success",
                value=True,
                strength=0.8
            )
            
    def personalize_response(self, user_id: str, base_response: str,
                            context: Dict = None) -> str:
        """Get personalized response for user"""
        return self.response_personalizer.personalize_response(user_id, base_response, context)
        
    def get_ui_configuration(self, user_id: str) -> Dict:
        """Get personalized UI configuration"""
        return self.ui_personalizer.get_ui_config(user_id)
        
    def collect_feedback(self, user_id: str, feedback_type: str, context: str,
                        value: Any, **kwargs):
        """Collect user feedback"""
        return self.feedback_system.collect_feedback(
            user_id,
            FeedbackType(feedback_type),
            FeedbackContext(context),
            value,
            **kwargs
        )
        
    def get_proactive_suggestions(self, user_id: str) -> List[Dict]:
        """Get proactive suggestions for user"""
        context = self._build_context(user_id)
        return self.proactive_engine.analyze_context(user_id, context)
        
    def _build_context(self, user_id: str) -> Dict:
        """Build current context for user"""
        return {
            "recent_actions": [],
            "active_application": None,
            "current_task": None,
            "time_of_day": datetime.now().strftime("%H:%M"),
            "meeting_status": "free"
        }
        
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (GDPR)"""
        return self.privacy_manager.export_user_data(user_id)
        
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (GDPR)"""
        return self.privacy_manager.delete_user_data(user_id)
        
    def get_user_summary(self, user_id: str) -> Dict:
        """Get comprehensive user summary"""
        profile = self.profile_manager.get_profile(user_id)
        
        if not profile:
            return {"error": "User not found"}
            
        return {
            "profile": {
                "user_id": profile.profile_id,
                "preferred_name": profile.preferred_name,
                "created_at": profile.created_at.isoformat()
            },
            "preferences": {
                "notification": profile.notification_prefs,
                "content": profile.content_prefs,
                "automation": profile.automation_prefs
            },
            "patterns": {
                "detected": len(profile.learned_patterns.get("detected_patterns", [])),
                "recurring_tasks": len(profile.learned_patterns.get("recurring_tasks", [])),
                "adaptations": len(profile.learned_patterns.get("adaptation_history", []))
            },
            "feedback": self.feedback_system.get_feedback_summary(user_id),
            "privacy": {
                "learning_enabled": profile.privacy_settings.get("learning_enabled", True),
                "data_retention_days": profile.privacy_settings.get("data_retention_days", 90)
            }
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize system
    system = UserModelingSystem()
    
    # Create a user
    system.initialize_user("user_123", {
        "preferred_name": "Alex",
        "timezone": "America/New_York"
    })
    
    # Record explicit feedback
    system.preference_engine.record_explicit_feedback(
        user_id="user_123",
        category="content",
        attribute="tone",
        value="professional",
        rating=5
    )
    
    # Get personalized response
    response = system.personalize_response(
        user_id="user_123",
        base_response="Hello! How can I help you today?"
    )
    print(f"Personalized: {response}")
    
    # Get UI config
    ui_config = system.get_ui_configuration("user_123")
    print(f"UI Config: {ui_config}")
    
    # Get user summary
    summary = system.get_user_summary("user_123")
    print(f"User Summary: {summary}")
