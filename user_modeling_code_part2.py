"""
OpenClaw Windows 10 - User Modeling System
Part 2: Pattern Recognition & Personalization Engine
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import numpy as np
from sklearn.cluster import DBSCAN

# ============================================================================
# BEHAVIOR PATTERN RECOGNITION
# ============================================================================

@dataclass
class BehaviorPattern:
    """Detected behavior pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    frequency: float
    first_observed: datetime
    last_observed: datetime
    examples: List[Dict]
    metadata: Dict


class TemporalPatternDetector:
    """Detects time-based behavior patterns"""
    
    def __init__(self, min_occurrences: int = 3):
        self.min_occurrences = min_occurrences
        self.observations: deque = deque(maxlen=10000)
        
    def add_observation(self, event_type: str, timestamp: datetime, metadata: Dict):
        """Add a new behavioral observation"""
        self.observations.append({
            "event_type": event_type,
            "timestamp": timestamp,
            "metadata": metadata
        })
        
    def detect_daily_patterns(self, event_type: str) -> List[BehaviorPattern]:
        """Detect daily recurring patterns for event type"""
        events = [o for o in self.observations if o["event_type"] == event_type]
        if len(events) < self.min_occurrences:
            return []
            
        times = [(e["timestamp"].hour, e["timestamp"].minute) for e in events]
        time_minutes = np.array([[h * 60 + m] for h, m in times])
        
        clustering = DBSCAN(eps=30, min_samples=self.min_occurrences).fit(time_minutes)
        
        patterns = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
                
            cluster_times = time_minutes[clustering.labels_ == cluster_id]
            mean_time = np.mean(cluster_times)
            std_time = np.std(cluster_times)
            
            mean_hour = int(mean_time // 60)
            mean_minute = int(mean_time % 60)
            confidence = 1.0 - min(1.0, std_time / 60)
            
            pattern = BehaviorPattern(
                pattern_id=f"temporal_{event_type}_{cluster_id}",
                pattern_type="temporal",
                description=f"{event_type} typically at {mean_hour:02d}:{mean_minute:02d}",
                confidence=confidence,
                frequency=len(cluster_times) / len(events),
                first_observed=events[0]["timestamp"],
                last_observed=events[-1]["timestamp"],
                examples=[events[i] for i in np.where(clustering.labels_ == cluster_id)[0][:5]],
                metadata={
                    "mean_time": f"{mean_hour:02d}:{mean_minute:02d}",
                    "std_minutes": float(std_time),
                    "cluster_size": len(cluster_times)
                }
            )
            patterns.append(pattern)
            
        return patterns
        
    def detect_weekly_patterns(self, event_type: str) -> List[BehaviorPattern]:
        """Detect weekly recurring patterns"""
        events = [o for o in self.observations if o["event_type"] == event_type]
        if len(events) < self.min_occurrences * 2:
            return []
            
        day_events = defaultdict(list)
        for e in events:
            day = e["timestamp"].weekday()
            day_events[day].append(e)
            
        patterns = []
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day, day_event_list in day_events.items():
            if len(day_event_list) >= self.min_occurrences:
                pattern = BehaviorPattern(
                    pattern_id=f"weekly_{event_type}_{day}",
                    pattern_type="temporal",
                    description=f"{event_type} typically on {day_names[day]}",
                    confidence=len(day_event_list) / len(events) * 7,
                    frequency=len(day_event_list) / (len(events) / 7),
                    first_observed=day_event_list[0]["timestamp"],
                    last_observed=day_event_list[-1]["timestamp"],
                    examples=day_event_list[:5],
                    metadata={"day_of_week": day, "day_name": day_names[day]}
                )
                patterns.append(pattern)
                
        return patterns


class SequentialPatternDetector:
    """Detects action sequence patterns"""
    
    def __init__(self, max_sequence_length: int = 5):
        self.max_sequence_length = max_sequence_length
        self.action_sequences: deque = deque(maxlen=1000)
        self.pattern_counts: Dict[Tuple, int] = defaultdict(int)
        
    def record_action(self, action: str, timestamp: datetime, context: Dict):
        """Record a user action"""
        self.action_sequences.append({
            "action": action,
            "timestamp": timestamp,
            "context": context
        })
        self._update_patterns()
        
    def _update_patterns(self):
        """Update pattern counts from recent sequences"""
        actions = [a["action"] for a in self.action_sequences]
        
        for n in range(2, self.max_sequence_length + 1):
            for i in range(len(actions) - n + 1):
                sequence = tuple(actions[i:i+n])
                self.pattern_counts[sequence] += 1
                
    def detect_common_sequences(self, min_frequency: int = 3) -> List[BehaviorPattern]:
        """Detect commonly occurring action sequences"""
        patterns = []
        
        for sequence, count in self.pattern_counts.items():
            if count >= min_frequency:
                confidence = min(1.0, count / 10)
                
                pattern = BehaviorPattern(
                    pattern_id=f"seq_{'_'.join(sequence)}",
                    pattern_type="sequential",
                    description=f"Sequence: {' -> '.join(sequence)}",
                    confidence=confidence,
                    frequency=count,
                    first_observed=datetime.utcnow(),
                    last_observed=datetime.utcnow(),
                    examples=[],
                    metadata={"sequence": sequence, "length": len(sequence), "count": count}
                )
                patterns.append(pattern)
                
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)


class SemanticPatternDetector:
    """Detects patterns in content and communication"""
    
    def __init__(self):
        self.email_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
    def analyze_email(self, email_data: Dict):
        """Analyze email for semantic patterns"""
        sender = email_data.get("from", "")
        subject = email_data.get("subject", "")
        body = email_data.get("body", "")
        timestamp = email_data.get("timestamp", datetime.utcnow())
        
        self.email_patterns[sender].append({
            "subject": subject,
            "timestamp": timestamp,
            "body_length": len(body)
        })
        
        subject_patterns = self._extract_subject_patterns(subject)
        topics = self._extract_topics(body)
        
        return {
            "sender": sender,
            "subject_patterns": subject_patterns,
            "topics": topics,
            "is_recurring_sender": len(self.email_patterns[sender]) > 3
        }
        
    def _extract_subject_patterns(self, subject: str) -> List[str]:
        """Extract common patterns from subject lines"""
        patterns = []
        
        if re.search(r'\b(meeting|sync|call|discussion)\b', subject, re.I):
            patterns.append("meeting_request")
        if re.search(r'\b(action|todo|task|follow.up)\b', subject, re.I):
            patterns.append("action_required")
        if re.search(r'\b(urgent|asap|deadline|important)\b', subject, re.I):
            patterns.append("urgent")
        if '?' in subject:
            patterns.append("question")
            
        return patterns
        
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        common_topics = {
            "project": ["project", "deliverable", "milestone"],
            "technical": ["bug", "feature", "code", "deployment"],
            "administrative": ["schedule", "meeting", "calendar"],
            "social": ["lunch", "coffee", "team"]
        }
        
        text_lower = text.lower()
        detected = []
        
        for topic, keywords in common_topics.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(topic)
                
        return detected
        
    def get_contact_insights(self, email: str) -> Dict:
        """Get insights about communication with a contact"""
        emails = self.email_patterns.get(email, [])
        
        if not emails:
            return {}
            
        if len(emails) >= 2:
            time_span = (emails[-1]["timestamp"] - emails[0]["timestamp"]).days
            frequency = len(emails) / max(1, time_span / 7)
        else:
            frequency = 0
            
        all_subjects = [e["subject"] for e in emails]
        common_prefixes = self._find_common_prefixes(all_subjects)
        
        return {
            "total_emails": len(emails),
            "frequency_per_week": frequency,
            "common_prefixes": common_prefixes,
            "relationship_type": self._classify_relationship(frequency, emails)
        }
        
    def _find_common_prefixes(self, subjects: List[str]) -> List[str]:
        """Find common prefixes in subjects"""
        if not subjects:
            return []
            
        prefix_counts = defaultdict(int)
        
        for subject in subjects:
            words = subject.split()[:3]
            for i in range(1, len(words) + 1):
                prefix = " ".join(words[:i])
                prefix_counts[prefix] += 1
                
        threshold = len(subjects) * 0.3
        return [p for p, c in prefix_counts.items() if c >= threshold]
        
    def _classify_relationship(self, frequency: float, emails: List[Dict]) -> str:
        """Classify relationship type based on patterns"""
        if frequency > 5: return "close_collaborator"
        elif frequency > 2: return "regular_contact"
        elif len(emails) > 5: return "frequent_but_irregular"
        else: return "occasional"


class PatternRecognitionOrchestrator:
    """Coordinates all pattern detection systems"""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.temporal_detector = TemporalPatternDetector()
        self.sequential_detector = SequentialPatternDetector()
        self.semantic_detector = SemanticPatternDetector()
        
        self.detected_patterns: Dict[str, List[BehaviorPattern]] = defaultdict(list)
        self.pattern_handlers: Dict[str, Callable] = {}
        
    def register_pattern_handler(self, pattern_type: str, handler: Callable):
        """Register a handler for detected patterns"""
        self.pattern_handlers[pattern_type] = handler
        
    def process_event(self, event_type: str, data: Dict):
        """Process a new event through all detectors"""
        timestamp = data.get("timestamp", datetime.utcnow())
        
        self.temporal_detector.add_observation(event_type, timestamp, data.get("metadata", {}))
        self.sequential_detector.record_action(event_type, timestamp, data.get("context", {}))
        
        if event_type == "email_received":
            self.semantic_detector.analyze_email(data)
            
        self._run_pattern_detection()
        
    def _run_pattern_detection(self):
        """Run all pattern detection algorithms"""
        for event_type in ["email_received", "app_opened", "meeting_started"]:
            patterns = self.temporal_detector.detect_daily_patterns(event_type)
            patterns.extend(self.temporal_detector.detect_weekly_patterns(event_type))
            
            for pattern in patterns:
                if pattern.confidence > 0.7:
                    self._handle_new_pattern(pattern)
                    
        seq_patterns = self.sequential_detector.detect_common_sequences()
        for pattern in seq_patterns:
            if pattern.confidence > 0.6:
                self._handle_new_pattern(pattern)
                
    def _handle_new_pattern(self, pattern: BehaviorPattern):
        """Handle newly detected pattern"""
        user_id = pattern.metadata.get("user_id", "default")
        self.detected_patterns[user_id].append(pattern)
        
        if pattern.pattern_type in self.pattern_handlers:
            self.pattern_handlers[pattern.pattern_type](pattern)
            
        self._update_profile_with_pattern(user_id, pattern)
        
    def _update_profile_with_pattern(self, user_id: str, pattern: BehaviorPattern):
        """Update user profile with detected pattern"""
        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            return
            
        if "detected_patterns" not in profile.learned_patterns:
            profile.learned_patterns["detected_patterns"] = []
            
        pattern_entry = {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "description": pattern.description,
            "confidence": pattern.confidence,
            "detected_at": datetime.utcnow().isoformat(),
            "metadata": pattern.metadata
        }
        
        profile.learned_patterns["detected_patterns"].append(pattern_entry)
        profile.learned_patterns["detected_patterns"] = profile.learned_patterns["detected_patterns"][-100:]
            
        self.profile_manager.update_profile(user_id, {"learned_patterns": profile.learned_patterns})
        
    def get_user_patterns(self, user_id: str, pattern_type: Optional[str] = None,
                         min_confidence: float = 0.5) -> List[BehaviorPattern]:
        """Get patterns for a user"""
        patterns = self.detected_patterns.get(user_id, [])
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
            
        return [p for p in patterns if p.confidence >= min_confidence]


# ============================================================================
# PERSONALIZATION ENGINE
# ============================================================================

from enum import Enum

class PersonalizationLevel(Enum):
    NONE = 0
    BASIC = 1
    ADAPTIVE = 2
    PREDICTIVE = 3

@dataclass
class PersonalizationContext:
    """Context for personalization decisions"""
    user_id: str
    current_task: Optional[str] = None
    active_application: Optional[str] = None
    time_of_day: Optional[str] = None
    stress_level: Optional[int] = None
    meeting_status: Optional[str] = None
    recent_context: List[Dict] = None

@dataclass
class PersonalizationResult:
    """Result of personalization"""
    value: Any
    confidence: float
    source: str
    reasoning: str
    alternatives: List[Any] = None


class PersonalizationEngine:
    """Core personalization engine"""
    
    def __init__(self, profile_manager, preference_engine, pattern_orchestrator):
        self.profile_manager = profile_manager
        self.preference_engine = preference_engine
        self.pattern_orchestrator = pattern_orchestrator
        
        self.personalization_level = PersonalizationLevel.ADAPTIVE
        self.personalizers: Dict[str, Callable] = {}
        
    def register_personalizer(self, aspect: str, personalizer: Callable):
        """Register a personalizer for a specific aspect"""
        self.personalizers[aspect] = personalizer
        
    def personalize(self, aspect: str, context: PersonalizationContext, 
                   default_value: Any = None) -> PersonalizationResult:
        """Get personalized value for an aspect"""
        
        if aspect in self.personalizers:
            return self.personalizers[aspect](context)
            
        return self._generic_personalize(aspect, context, default_value)
        
    def _generic_personalize(self, aspect: str, context: PersonalizationContext,
                            default_value: Any) -> PersonalizationResult:
        """Generic personalization using profile and preferences"""
        
        profile = self.profile_manager.get_profile(context.user_id)
        if not profile:
            return PersonalizationResult(
                value=default_value, confidence=0.0, source="default",
                reasoning="No user profile found"
            )
            
        category, attribute = self._parse_aspect(aspect)
        preference = self.preference_engine.get_preference(
            context.user_id, category, attribute, default_value
        )
        
        if preference["source"] != "default":
            return PersonalizationResult(
                value=preference["value"],
                confidence=preference["confidence"],
                source=preference["source"],
                reasoning=f"Retrieved from {preference['source']} preferences"
            )
            
        patterns = self.pattern_orchestrator.get_user_patterns(
            context.user_id, min_confidence=0.7
        )
        
        for pattern in patterns:
            if self._pattern_applies(pattern, aspect, context):
                inferred_value = self._extract_value_from_pattern(pattern, aspect)
                if inferred_value is not None:
                    return PersonalizationResult(
                        value=inferred_value,
                        confidence=pattern.confidence * 0.8,
                        source="predicted",
                        reasoning=f"Inferred from pattern: {pattern.description}"
                    )
                    
        return PersonalizationResult(
            value=default_value, confidence=0.0, source="default",
            reasoning="No personalization data available"
        )
        
    def _parse_aspect(self, aspect: str) -> Tuple[str, str]:
        """Parse aspect string into category and attribute"""
        parts = aspect.split(".")
        if len(parts) == 2:
            return parts[0], parts[1]
        return "general", aspect
        
    def _pattern_applies(self, pattern: BehaviorPattern, aspect: str, 
                        context: PersonalizationContext) -> bool:
        """Check if a pattern applies to current context"""
        return aspect in pattern.pattern_id or aspect in pattern.description.lower()
        
    def _extract_value_from_pattern(self, pattern: BehaviorPattern, aspect: str) -> Any:
        """Extract relevant value from pattern metadata"""
        return pattern.metadata.get("value") or pattern.metadata.get(aspect)


class ResponsePersonalizer:
    """Personalizes AI responses based on user preferences"""
    
    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine
        
    def personalize_response(self, user_id: str, base_response: str,
                            context: Dict = None) -> str:
        """Personalize a response for the user"""
        
        pers_context = PersonalizationContext(user_id=user_id, **(context or {}))
        
        tone = self.engine.personalize("content.tone", pers_context, "neutral")
        verbosity = self.engine.personalize("content.verbosity", pers_context, "moderate")
        formality = self.engine.personalize("content.formality", pers_context, "neutral")
        
        response = base_response
        response = self._apply_tone(response, tone.value)
        response = self._apply_verbosity(response, verbosity.value)
        response = self._apply_formality(response, formality.value)
        
        return response
        
    def _apply_tone(self, text: str, tone: str) -> str:
        """Adjust text tone"""
        if tone == "friendly":
            if any(text.startswith(g) for g in ["Hi", "Hello", "Hey"]):
                for cold, warm in [("Hi", "Hey there!"), ("Hello", "Hi there!")]:
                    text = text.replace(cold, warm, 1)
        elif tone == "professional":
            text = text.replace("!", ".")
            text = text.replace("Hey", "Hello")
        return text
        
    def _apply_verbosity(self, text: str, level: str) -> str:
        """Adjust response length"""
        if level == "concise":
            sentences = text.split(". ")
            if len(sentences) > 2:
                return ". ".join(sentences[:2]) + "."
        return text
        
    def _apply_formality(self, text: str, level: str) -> str:
        """Adjust formality level"""
        informal_to_formal = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "gonna": "going to", "wanna": "want to"
        }
        
        if level == "formal":
            for informal, formal in informal_to_formal.items():
                text = text.replace(informal, formal)
        elif level == "casual":
            for informal, formal in informal_to_formal.items():
                text = text.replace(formal, informal)
        return text


class UIPersonalizer:
    """Personalizes UI elements based on user preferences"""
    
    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine
        
    def get_ui_config(self, user_id: str) -> Dict:
        """Get personalized UI configuration"""
        context = PersonalizationContext(user_id=user_id)
        
        return {
            "theme": self.engine.personalize("ui.theme", context, "auto").value,
            "font_size": self.engine.personalize("ui.font_size", context, "medium").value,
            "compact_mode": self.engine.personalize("ui.compact_mode", context, False).value,
            "show_confidence": self.engine.personalize("ui.show_confidence", context, False).value,
            "notification_style": self.engine.personalize("notification.style", context, "standard").value
        }
