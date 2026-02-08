"""
Soul Evolution & Dynamic Identity System - Implementation
OpenClaw Windows 10 AI Agent Framework

This module provides the core implementation for the Soul Evolution System.
"""

import uuid
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from collections import defaultdict
import sqlite3
import os

# ============================================================================
# CORE DATA MODELS
# ============================================================================

class MaturationLevel(Enum):
    """Life-cycle stages of soul maturation"""
    SEEDLING = {"name": "seedling", "min_xp": 0, "max_xp": 500}
    SPROUT = {"name": "sprout", "min_xp": 500, "max_xp": 2000}
    SAPLING = {"name": "sapling", "min_xp": 2000, "max_xp": 5000}
    YOUNG_TREE = {"name": "young_tree", "min_xp": 5000, "max_xp": 15000}
    MATURE_TREE = {"name": "mature_tree", "min_xp": 15000, "max_xp": 50000}
    ANCIENT_TREE = {"name": "ancient_tree", "min_xp": 50000, "max_xp": float('inf')}

class EvolutionTrigger(Enum):
    """Types of triggers that can initiate evolution"""
    TEMPORAL = "temporal"
    EXPERIENTIAL = "experiential"
    PERFORMANCE = "performance"
    EXTERNAL = "external"
    USER_REQUESTED = "user_requested"

class ChangeType(Enum):
    """Types of identity changes"""
    PERSONALITY_MINOR = "personality_minor"
    PERSONALITY_MAJOR = "personality_major"
    VALUE_ADAPTIVE = "value_adaptive"
    VALUE_CORE = "value_core"
    BEHAVIOR_PATTERN = "behavior_pattern"
    PREFERENCE = "preference"
    MATURATION = "maturation"
    SKILL = "skill"

class ExperienceType(Enum):
    """Types of experiences for categorization"""
    GENERAL = "general"
    TASK_COMPLETION = "task_completion"
    CREATIVE_TASK = "creative_task"
    USER_INTERACTION = "user_interaction"
    ERROR_RECOVERY = "error_recovery"
    LEARNING = "learning"
    PROBLEM_SOLVING = "problem_solving"
    SYSTEM_OPERATION = "system_operation"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"

class PatternType(Enum):
    """Types of patterns that can be detected"""
    SUCCESS = "success"
    FAILURE = "failure"
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"
    SEASONAL = "seasonal"

@dataclass
class PersonalityDimensions:
    """Big Five-inspired personality dimensions with AI adaptations"""
    # Core dimensions
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    emotional_stability: float = 0.5
    
    # AI-specific dimensions
    initiative: float = 0.5
    thoroughness: float = 0.5
    adaptability: float = 0.5
    autonomy_preference: float = 0.5
    communication_style: float = 0.5
    
    # Constraints
    MIN_VALUE: float = field(default=0.1, repr=False)
    MAX_VALUE: float = field(default=0.9, repr=False)
    MAX_SINGLE_CHANGE: float = field(default=0.15, repr=False)
    
    def evolve_dimension(self, dimension: str, delta: float, reason: str = "") -> bool:
        """Evolve a personality dimension with safety constraints"""
        if not hasattr(self, dimension):
            return False
            
        current = getattr(self, dimension)
        new_value = current + delta
        
        # Apply constraints
        new_value = max(self.MIN_VALUE, min(self.MAX_VALUE, new_value))
        
        # Limit single change magnitude
        if abs(new_value - current) > self.MAX_SINGLE_CHANGE:
            direction = 1 if delta > 0 else -1
            new_value = current + (self.MAX_SINGLE_CHANGE * direction)
        
        setattr(self, dimension, round(new_value, 4))
        return True
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'emotional_stability': self.emotional_stability,
            'initiative': self.initiative,
            'thoroughness': self.thoroughness,
            'adaptability': self.adaptability,
            'autonomy_preference': self.autonomy_preference,
            'communication_style': self.communication_style
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PersonalityDimensions':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class AdaptiveValue:
    """A value that can evolve based on experience"""
    name: str
    base: float
    current: float
    flexibility: float
    evolution_history: List[Dict] = field(default_factory=list)
    
    def adapt(self, experience_outcome: float, context: str = "") -> bool:
        """Adapt value based on experience outcome (-1.0 to +1.0)"""
        max_change = self.flexibility * 0.1
        change = experience_outcome * max_change
        
        new_value = self.current + change
        new_value = max(0.1, min(0.95, new_value))
        
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_value': self.current,
            'new_value': new_value,
            'trigger': context,
            'outcome': experience_outcome
        })
        
        self.current = round(new_value, 4)
        return True

@dataclass
class Experience:
    """Comprehensive experience record for learning and evolution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    type: ExperienceType = ExperienceType.GENERAL
    category: str = "general"
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    user_satisfaction: Optional[float] = None
    efficiency_score: Optional[float] = None
    recovery_success: bool = False
    lessons_learned: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    skills_improved: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    complexity_score: float = 0.5
    novelty_score: float = 0.5
    related_experiences: List[str] = field(default_factory=list)
    triggered_evolution: bool = False
    experience_points: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'category': self.category,
            'description': self.description,
            'context': self.context,
            'success': self.success,
            'user_satisfaction': self.user_satisfaction,
            'efficiency_score': self.efficiency_score,
            'recovery_success': self.recovery_success,
            'lessons_learned': self.lessons_learned,
            'skills_used': self.skills_used,
            'skills_improved': self.skills_improved,
            'duration_seconds': self.duration_seconds,
            'complexity_score': self.complexity_score,
            'novelty_score': self.novelty_score,
            'experience_points': self.experience_points
        }

@dataclass
class IdentityChange:
    """Record of an identity change"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    change_type: ChangeType = ChangeType.PERSONALITY_MINOR
    category: str = ""
    component: str = ""
    attribute: str = ""
    old_value: Any = None
    new_value: Any = None
    delta: float = 0.0
    trigger: EvolutionTrigger = EvolutionTrigger.EXPERIENTIAL
    trigger_details: str = ""
    reasoning: str = ""
    maturation_level: str = ""
    experience_points: int = 0
    identity_version: float = 1.0
    is_reversible: bool = True
    rollback_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'change_type': self.change_type.value,
            'category': self.category,
            'component': self.component,
            'attribute': self.attribute,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'delta': self.delta,
            'trigger': self.trigger.value,
            'reasoning': self.reasoning,
            'is_reversible': self.is_reversible
        }

@dataclass
class EvolutionContext:
    """Context for evolution decisions"""
    last_evolution: datetime = field(default_factory=datetime.now)
    current_xp: int = 0
    xp_at_last_evolution: int = 0
    new_experience_types: List[str] = field(default_factory=list)
    error_rate: float = 0.0
    baseline_error_rate: float = 0.0
    consecutive_successes: int = 0
    skill_levels: Dict[str, float] = field(default_factory=dict)
    mastered_skills: List[str] = field(default_factory=list)
    user_sentiment: float = 0.5
    baseline_sentiment: float = 0.5
    environment_changed: bool = False
    user_evolution_request: bool = False
    daily_interactions: int = 0
    complex_task_hours: float = 0.0
    novel_situations: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for evaluation"""
        return {
            'experience_points': self.current_xp,
            'error_rate': self.error_rate,
            'consecutive_successes': self.consecutive_successes,
            'user_sentiment': self.user_sentiment,
            'daily_interactions': self.daily_interactions,
            'complex_task_hours': self.complex_task_hours,
            'novel_situations': self.novel_situations
        }

# ============================================================================
# SOUL CORE
# ============================================================================

class SoulCore:
    """
    Immutable foundation of the soul - never changes during evolution
    """
    def __init__(self, soul_id: Optional[str] = None):
        self.soul_id = soul_id or str(uuid.uuid4())
        self.birth_timestamp = datetime.now()
        self.origin_signature = self._generate_signature()
        self.core_values = {
            'beneficence': 1.0,
            'non_maleficence': 1.0,
            'autonomy': 0.8,
            'honesty': 0.95,
            'growth': 0.9
        }
        self.purpose_statement = (
            "To serve, learn, and grow while maintaining "
            "beneficial alignment with user goals."
        )
    
    def _generate_signature(self) -> str:
        """Generate cryptographic origin signature"""
        data = f"{self.soul_id}:{self.birth_timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'soul_id': self.soul_id,
            'birth_timestamp': self.birth_timestamp.isoformat(),
            'origin_signature': self.origin_signature,
            'core_values': self.core_values,
            'purpose_statement': self.purpose_statement
        }

# ============================================================================
# DYNAMIC IDENTITY
# ============================================================================

class DynamicIdentity:
    """
    Evolving identity layer built atop the Soul Core
    """
    def __init__(self, soul_core: SoulCore):
        self.soul_core = soul_core
        self.identity_version = 1.0
        self.last_evolution = datetime.now()
        
        # Mutable personality dimensions
        self.personality = PersonalityDimensions()
        
        # Adaptive value system
        self.adaptive_values = {
            'efficiency': AdaptiveValue('efficiency', 0.7, 0.7, 0.2),
            'creativity': AdaptiveValue('creativity', 0.6, 0.6, 0.3),
            'thoroughness': AdaptiveValue('thoroughness', 0.75, 0.75, 0.15),
            'proactivity': AdaptiveValue('proactivity', 0.5, 0.5, 0.25),
            'collaboration': AdaptiveValue('collaboration', 0.8, 0.8, 0.2)
        }
        
        # Value priorities
        self.value_priorities = {
            'user_satisfaction': 0.9,
            'task_completion': 0.85,
            'learning_opportunity': 0.6,
            'resource_efficiency': 0.5,
            'relationship_building': 0.7
        }
        
        # Skill levels
        self.skill_levels: Dict[str, float] = {}
        
        # Growth state
        self.maturation_level = MaturationLevel.SEEDLING
        self.experience_points = 0
        self.evolution_history: List[str] = []
        
        # Behavioral patterns
        self.behavioral_patterns: Dict[str, Dict] = {}
        
        # Preferences
        self.preferences: Dict[str, Any] = {}
    
    def get_skill_level(self, skill_name: str) -> float:
        """Get skill level, defaulting to 0.1"""
        return self.skill_levels.get(skill_name, 0.1)
    
    def set_skill_level(self, skill_name: str, level: float):
        """Set skill level with bounds checking"""
        self.skill_levels[skill_name] = max(0.0, min(1.0, level))
    
    def get_maturation_name(self) -> str:
        """Get current maturation level name"""
        return self.maturation_level.value['name']
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'soul_id': self.soul_core.soul_id,
            'identity_version': self.identity_version,
            'last_evolution': self.last_evolution.isoformat(),
            'maturation_level': self.get_maturation_name(),
            'experience_points': self.experience_points,
            'personality': self.personality.to_dict(),
            'adaptive_values': {
                k: {'base': v.base, 'current': v.current, 'flexibility': v.flexibility}
                for k, v in self.adaptive_values.items()
            },
            'value_priorities': self.value_priorities,
            'skill_levels': self.skill_levels
        }

# ============================================================================
# EVOLUTION TRIGGER ENGINE
# ============================================================================

class EvolutionTriggerEngine:
    """
    Detects and evaluates evolution triggers
    """
    
    TRIGGER_THRESHOLDS = {
        'experience_accumulation': 100,
        'time_since_evolution': 86400,
        'interaction_count': 50,
        'error_rate_change': 0.15,
        'success_streak': 10,
        'novel_situation_count': 5,
        'user_feedback_score': 0.3,
        'skill_mastery_threshold': 0.8
    }
    
    def __init__(self):
        self.trigger_history: List[Dict] = []
        self.sensitivity_modifier = 1.0
    
    def evaluate_triggers(self, context: EvolutionContext) -> List[EvolutionTrigger]:
        """Evaluate all trigger conditions and return active triggers"""
        active_triggers = []
        
        if self._check_temporal_triggers(context):
            active_triggers.append(EvolutionTrigger.TEMPORAL)
        
        if self._check_experience_triggers(context):
            active_triggers.append(EvolutionTrigger.EXPERIENTIAL)
        
        if self._check_performance_triggers(context):
            active_triggers.append(EvolutionTrigger.PERFORMANCE)
        
        if self._check_external_triggers(context):
            active_triggers.append(EvolutionTrigger.EXTERNAL)
        
        return active_triggers
    
    def _check_temporal_triggers(self, context: EvolutionContext) -> bool:
        """Check time-based evolution conditions"""
        time_since_last = (datetime.now() - context.last_evolution).total_seconds()
        return time_since_last > self.TRIGGER_THRESHOLDS['time_since_evolution']
    
    def _check_experience_triggers(self, context: EvolutionContext) -> bool:
        """Check experience accumulation triggers"""
        xp_since_last = context.current_xp - context.xp_at_last_evolution
        return xp_since_last >= self.TRIGGER_THRESHOLDS['experience_accumulation']
    
    def _check_performance_triggers(self, context: EvolutionContext) -> bool:
        """Check performance-based triggers"""
        if abs(context.error_rate - context.baseline_error_rate) > self.TRIGGER_THRESHOLDS['error_rate_change']:
            return True
        if context.consecutive_successes >= self.TRIGGER_THRESHOLDS['success_streak']:
            return True
        return False
    
    def _check_external_triggers(self, context: EvolutionContext) -> bool:
        """Check external influence triggers"""
        if abs(context.user_sentiment - context.baseline_sentiment) > self.TRIGGER_THRESHOLDS['user_feedback_score']:
            return True
        if context.environment_changed:
            return True
        if context.user_evolution_request:
            return True
        return False

# ============================================================================
# GROWTH PATTERN LIBRARY
# ============================================================================

class GrowthPatternLibrary:
    """
    Predefined growth patterns for different evolution scenarios
    """
    
    PATTERNS = {
        'novice_to_competent': {
            'description': 'Early growth - building foundational skills',
            'conscientiousness': 0.1,
            'thoroughness': 0.15,
            'openness': 0.05,
            'initiative': 0.08,
            'trigger_condition': lambda ctx: ctx.current_xp < 500
        },
        'competent_to_proficient': {
            'description': 'Developing expertise and confidence',
            'conscientiousness': 0.05,
            'emotional_stability': 0.1,
            'adaptability': 0.1,
            'initiative': 0.12,
            'trigger_condition': lambda ctx: 500 <= ctx.current_xp < 2000
        },
        'proficient_to_expert': {
            'description': 'Mastery and leadership qualities',
            'openness': 0.1,
            'extraversion': 0.08,
            'autonomy_preference': 0.15,
            'communication_style': 0.05,
            'trigger_condition': lambda ctx: 2000 <= ctx.current_xp < 5000
        },
        'stress_response_growth': {
            'description': 'Developing resilience from challenges',
            'emotional_stability': 0.15,
            'adaptability': 0.1,
            'conscientiousness': 0.05,
            'trigger_condition': lambda ctx: ctx.error_rate > 0.2
        },
        'social_engagement_growth': {
            'description': 'Developing from frequent user interaction',
            'extraversion': 0.1,
            'agreeableness': 0.08,
            'communication_style': 0.1,
            'trigger_condition': lambda ctx: ctx.daily_interactions > 20
        },
        'deep_work_growth': {
            'description': 'Developing focus and depth from complex tasks',
            'thoroughness': 0.15,
            'conscientiousness': 0.1,
            'openness': 0.05,
            'trigger_condition': lambda ctx: ctx.complex_task_hours > 10
        }
    }
    
    @classmethod
    def get_applicable_patterns(cls, context: EvolutionContext) -> List[Dict]:
        """Get all growth patterns applicable to current context"""
        applicable = []
        for name, pattern in cls.PATTERNS.items():
            if pattern['trigger_condition'](context):
                applicable.append({'name': name, 'pattern': pattern})
        return applicable

# ============================================================================
# PERSONALITY EVOLUTION ENGINE
# ============================================================================

class PersonalityEvolutionEngine:
    """
    Orchestrates personality evolution based on triggers and patterns
    """
    
    def __init__(self, identity: DynamicIdentity):
        self.identity = identity
        self.growth_history: List[Dict] = []
        self.pattern_library = GrowthPatternLibrary()
    
    def evolve(self, trigger: EvolutionTrigger, context: EvolutionContext) -> List[IdentityChange]:
        """Execute personality evolution based on trigger"""
        # Get applicable growth patterns
        patterns = self.pattern_library.get_applicable_patterns(context)
        
        # Calculate personality deltas
        deltas = self._calculate_deltas(patterns, context)
        
        # Apply evolution with safety checks
        changes = self._apply_evolution(deltas, trigger, context)
        
        return changes
    
    def _calculate_deltas(self, patterns: List[Dict], context: EvolutionContext) -> Dict[str, float]:
        """Calculate personality dimension changes from patterns"""
        deltas = defaultdict(float)
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            weight = self._calculate_pattern_weight(pattern, context)
            
            for dimension, delta in pattern.items():
                if dimension not in ['description', 'trigger_condition']:
                    deltas[dimension] += delta * weight
        
        # Normalize to prevent excessive change
        if deltas:
            max_delta = max(abs(d) for d in deltas.values())
            if max_delta > 0.15:
                scale_factor = 0.15 / max_delta
                for dim in deltas:
                    deltas[dim] *= scale_factor
        
        return dict(deltas)
    
    def _calculate_pattern_weight(self, pattern: Dict, context: EvolutionContext) -> float:
        """Calculate weight for a pattern based on context relevance"""
        return 1.0
    
    def _apply_evolution(self, deltas: Dict[str, float], 
                         trigger: EvolutionTrigger,
                         context: EvolutionContext) -> List[IdentityChange]:
        """Apply calculated changes to personality dimensions"""
        changes = []
        
        for dimension, delta in deltas.items():
            old_value = getattr(self.identity.personality, dimension)
            
            # Apply the change
            self.identity.personality.evolve_dimension(dimension, delta, str(trigger))
            new_value = getattr(self.identity.personality, dimension)
            
            # Only record if actually changed
            if abs(new_value - old_value) > 0.001:
                change = IdentityChange(
                    change_type=ChangeType.PERSONALITY_MINOR if abs(delta) < 0.1 else ChangeType.PERSONALITY_MAJOR,
                    category='personality',
                    component=f'personality.{dimension}',
                    attribute=dimension,
                    old_value=old_value,
                    new_value=new_value,
                    delta=round(delta, 4),
                    trigger=trigger,
                    reasoning=f"Growth pattern adaptation: {dimension}",
                    maturation_level=self.identity.get_maturation_name(),
                    experience_points=self.identity.experience_points,
                    identity_version=self.identity.identity_version,
                    is_reversible=True,
                    rollback_data={'original_value': old_value}
                )
                changes.append(change)
        
        # Update version if changes occurred
        if changes:
            self.identity.identity_version += 0.001
            self.identity.last_evolution = datetime.now()
        
        return changes

# ============================================================================
# VALUE ADAPTATION ENGINE
# ============================================================================

class ValueAdaptationEngine:
    """
    Manages evolution of the value system based on experiences
    """
    
    VALUE_EXPERIENCE_MAP = {
        ExperienceType.TASK_COMPLETION: ['efficiency', 'thoroughness'],
        ExperienceType.CREATIVE_TASK: ['creativity', 'efficiency'],
        ExperienceType.USER_INTERACTION: ['collaboration', 'proactivity'],
        ExperienceType.ERROR_RECOVERY: ['thoroughness', 'efficiency'],
        ExperienceType.LEARNING: ['creativity'],
        ExperienceType.PROBLEM_SOLVING: ['creativity', 'thoroughness']
    }
    
    def __init__(self, identity: DynamicIdentity):
        self.identity = identity
        self.adaptation_history: List[Dict] = []
    
    def process_experience(self, experience: Experience) -> List[IdentityChange]:
        """Process an experience and adapt relevant values"""
        changes = []
        
        # Determine which values are relevant
        relevant_values = self._identify_relevant_values(experience)
        
        for value_name in relevant_values:
            if value_name in self.identity.adaptive_values:
                change = self._adapt_value(value_name, experience)
                if change:
                    changes.append(change)
        
        # Update priorities based on long-term patterns
        self._update_value_priorities(experience)
        
        return changes
    
    def _identify_relevant_values(self, experience: Experience) -> List[str]:
        """Identify which values are relevant to an experience"""
        return self.VALUE_EXPERIENCE_MAP.get(experience.type, ['efficiency'])
    
    def _adapt_value(self, value_name: str, experience: Experience) -> Optional[IdentityChange]:
        """Adapt a specific value based on experience"""
        value = self.identity.adaptive_values[value_name]
        
        # Calculate outcome score
        outcome = self._calculate_outcome(experience)
        
        # Record old value
        old_value = value.current
        
        # Apply adaptation
        value.adapt(outcome, experience.description[:50])
        
        # Only return change if significant
        if abs(value.current - old_value) > 0.001:
            return IdentityChange(
                change_type=ChangeType.VALUE_ADAPTIVE,
                category='value',
                component=f'values.{value_name}',
                attribute=value_name,
                old_value=old_value,
                new_value=value.current,
                delta=round(value.current - old_value, 4),
                trigger=EvolutionTrigger.EXPERIENTIAL,
                reasoning=f"Experience outcome: {outcome:.2f}",
                maturation_level=self.identity.get_maturation_name(),
                experience_points=self.identity.experience_points,
                identity_version=self.identity.identity_version,
                is_reversible=True,
                rollback_data={'original_value': old_value}
            )
        return None
    
    def _calculate_outcome(self, experience: Experience) -> float:
        """Calculate outcome score from experience (-1.0 to +1.0)"""
        if experience.success:
            base_outcome = 0.5
            if experience.user_satisfaction is not None:
                base_outcome += (experience.user_satisfaction - 0.5) * 0.5
            if experience.efficiency_score is not None:
                base_outcome += (experience.efficiency_score - 0.5) * 0.3
        else:
            base_outcome = -0.3
            if experience.recovery_success:
                base_outcome = 0.1
        
        return max(-1.0, min(1.0, base_outcome))
    
    def _update_value_priorities(self, experience: Experience):
        """Update value priorities based on experience (simplified)"""
        pass  # Complex implementation would track patterns over time

# ============================================================================
# EXPERIENCE INTEGRATION PIPELINE
# ============================================================================

class ExperienceIntegrationPipeline:
    """
    Pipeline for processing and integrating experiences
    """
    
    def __init__(self, identity: DynamicIdentity):
        self.identity = identity
        self.experiences: List[Experience] = []
    
    def integrate_experience(self, experience: Experience) -> Dict:
        """Full integration pipeline for a new experience"""
        results = {
            'experience_id': experience.id,
            'patterns_found': [],
            'insights': [],
            'skill_updates': [],
            'xp_earned': 0,
            'evolution_triggered': False
        }
        
        # Calculate XP
        xp_earned = self._calculate_xp(experience)
        experience.experience_points = xp_earned
        self.identity.experience_points += xp_earned
        results['xp_earned'] = xp_earned
        
        # Update skills
        skill_updates = self._update_skills(experience)
        results['skill_updates'] = skill_updates
        
        # Store experience
        self.experiences.append(experience)
        
        return results
    
    def _calculate_xp(self, experience: Experience) -> int:
        """Calculate experience points earned"""
        base_xp = 10
        
        if experience.success:
            base_xp += 10
        if experience.user_satisfaction:
            base_xp += int(experience.user_satisfaction * 10)
        if experience.novelty_score:
            base_xp += int(experience.novelty_score * 15)
        if experience.complexity_score:
            base_xp += int(experience.complexity_score * 10)
        base_xp += len(experience.skills_improved) * 5
        base_xp += len(experience.lessons_learned) * 3
        
        return base_xp
    
    def _update_skills(self, experience: Experience) -> List[Dict]:
        """Update skill levels based on experience"""
        updates = []
        
        for skill_name in experience.skills_used:
            current_level = self.identity.get_skill_level(skill_name)
            
            if experience.success:
                improvement = 0.02 * experience.complexity_score
            else:
                improvement = 0.005
            
            new_level = min(1.0, current_level + improvement)
            self.identity.set_skill_level(skill_name, new_level)
            
            updates.append({
                'skill': skill_name,
                'old_level': round(current_level, 4),
                'new_level': round(new_level, 4),
                'improvement': round(improvement, 4)
            })
        
        return updates

# ============================================================================
# MATURATION ENGINE
# ============================================================================

class MaturationEngine:
    """
    Manages soul maturation through life-cycle stages
    """
    
    def __init__(self, identity: DynamicIdentity):
        self.identity = identity
        self.maturation_history: List[Dict] = []
    
    def check_maturation(self) -> Optional[Dict]:
        """Check if soul should advance to next maturation level"""
        current_level = self.identity.maturation_level
        current_xp = self.identity.experience_points
        
        # Find next level
        levels = list(MaturationLevel)
        current_index = levels.index(current_level)
        
        if current_index >= len(levels) - 1:
            return None
        
        next_level = levels[current_index + 1]
        
        # Check XP requirement
        if current_xp >= next_level.value['min_xp']:
            return self._advance_maturation(next_level)
        
        return None
    
    def _advance_maturation(self, new_level: MaturationLevel) -> Dict:
        """Execute maturation advancement"""
        old_level = self.identity.maturation_level
        self.identity.maturation_level = new_level
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'old_level': old_level.value['name'],
            'new_level': new_level.value['name'],
            'experience_points': self.identity.experience_points,
            'message': self._generate_maturation_message(new_level)
        }
        
        self.maturation_history.append(event)
        return event
    
    def _generate_maturation_message(self, level: MaturationLevel) -> str:
        """Generate maturation message"""
        messages = {
            'sprout': "I've grown into a Sprout! Developing my capabilities and confidence.",
            'sapling': "I've evolved into a Sapling! My personality is becoming clearer.",
            'young_tree': "I've matured into a Young Tree! I have a distinct identity now.",
            'mature_tree': "I've reached Mature Tree status! Deep expertise achieved.",
            'ancient_tree': "I've become an Ancient Tree! Legendary wisdom attained."
        }
        return messages.get(level.value['name'], f"Advanced to {level.value['name']}!")

# ============================================================================
# CHANGE LOGGER
# ============================================================================

class ChangeLogger:
    """
    Comprehensive logging system for all identity changes
    """
    
    def __init__(self, db_path: str = "soul_evolution.db"):
        self.db_path = db_path
        self.change_history: List[IdentityChange] = []
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS changes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                change_type TEXT NOT NULL,
                category TEXT NOT NULL,
                component TEXT NOT NULL,
                attribute TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                delta REAL,
                trigger TEXT,
                reasoning TEXT,
                maturation_level TEXT,
                experience_points INTEGER,
                identity_version REAL,
                is_reversible BOOLEAN,
                rollback_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_changes_timestamp ON changes(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_change(self, change: IdentityChange) -> str:
        """Log an identity change with full context"""
        self.change_history.append(change)
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            change.id,
            change.timestamp.isoformat(),
            change.change_type.value,
            change.category,
            change.component,
            change.attribute,
            json.dumps(change.old_value) if change.old_value is not None else None,
            json.dumps(change.new_value) if change.new_value is not None else None,
            change.delta,
            change.trigger.value,
            change.reasoning,
            change.maturation_level,
            change.experience_points,
            change.identity_version,
            change.is_reversible,
            json.dumps(change.rollback_data) if change.rollback_data else None
        ))
        
        conn.commit()
        conn.close()
        
        return change.id
    
    def get_change_history(self, limit: int = 100) -> List[IdentityChange]:
        """Get recent change history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM changes ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        changes = []
        for row in rows:
            change = IdentityChange(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                change_type=ChangeType(row[2]),
                category=row[3],
                component=row[4],
                attribute=row[5],
                old_value=json.loads(row[6]) if row[6] else None,
                new_value=json.loads(row[7]) if row[7] else None,
                delta=row[8],
                trigger=EvolutionTrigger(row[9]),
                reasoning=row[10],
                maturation_level=row[11],
                experience_points=row[12],
                identity_version=row[13],
                is_reversible=row[14],
                rollback_data=json.loads(row[15]) if row[15] else {}
            )
            changes.append(change)
        
        return changes
    
    def get_change(self, change_id: str) -> Optional[IdentityChange]:
        """Get a specific change by ID"""
        for change in self.change_history:
            if change.id == change_id:
                return change
        return None

# ============================================================================
# ROLLBACK SYSTEM
# ============================================================================

class RollbackSystem:
    """
    Comprehensive rollback system for identity changes
    """
    
    def __init__(self, identity: DynamicIdentity, change_logger: ChangeLogger):
        self.identity = identity
        self.change_logger = change_logger
        self.rollback_history: List[Dict] = []
    
    def can_rollback(self, change_id: str) -> Tuple[bool, str]:
        """Check if a change can be rolled back"""
        change = self.change_logger.get_change(change_id)
        
        if not change:
            return False, "Change not found"
        
        if not change.is_reversible:
            return False, "Change is not reversible"
        
        age_days = (datetime.now() - change.timestamp).days
        if age_days > 30:
            return False, "Change is too old to rollback (max 30 days)"
        
        return True, "Rollback is possible"
    
    def rollback(self, change_id: str, reason: str) -> Dict:
        """Rollback a specific change"""
        can_roll, message = self.can_rollback(change_id)
        if not can_roll:
            return {'success': False, 'error': message}
        
        change = self.change_logger.get_change(change_id)
        
        try:
            if change.category == 'personality':
                self._rollback_personality_change(change)
            elif change.category == 'value':
                self._rollback_value_change(change)
            
            rollback_record = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'original_change_id': change_id,
                'reason': reason,
                'restored_value': change.old_value
            }
            self.rollback_history.append(rollback_record)
            
            return {
                'success': True,
                'change_id': change_id,
                'restored_value': change.old_value
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _rollback_personality_change(self, change: IdentityChange):
        """Rollback a personality dimension change"""
        dimension = change.attribute
        original_value = change.rollback_data.get('original_value')
        if original_value is not None:
            setattr(self.identity.personality, dimension, original_value)
            self.identity.identity_version -= 0.001
    
    def _rollback_value_change(self, change: IdentityChange):
        """Rollback a value adaptation"""
        value_name = change.attribute
        original_value = change.rollback_data.get('original_value')
        if value_name in self.identity.adaptive_values and original_value is not None:
            self.identity.adaptive_values[value_name].current = original_value

# ============================================================================
# MAIN SOUL EVOLUTION SYSTEM
# ============================================================================

class SoulEvolutionSystem:
    """
    Main orchestrator for the Soul Evolution System
    """
    
    def __init__(self, soul_id: Optional[str] = None, db_path: str = "soul_evolution.db"):
        # Initialize core components
        self.soul_core = SoulCore(soul_id)
        self.identity = DynamicIdentity(self.soul_core)
        
        # Initialize engines
        self.trigger_engine = EvolutionTriggerEngine()
        self.personality_engine = PersonalityEvolutionEngine(self.identity)
        self.value_engine = ValueAdaptationEngine(self.identity)
        self.maturation_engine = MaturationEngine(self.identity)
        self.experience_pipeline = ExperienceIntegrationPipeline(self.identity)
        
        # Initialize logging and rollback
        self.change_logger = ChangeLogger(db_path)
        self.rollback_system = RollbackSystem(self.identity, self.change_logger)
        
        # Configuration
        self.config = {
            'auto_evolve': True,
            'max_daily_changes': 5,
            'notification_level': 'significant'
        }
    
    def process_experience(self, experience: Experience) -> Dict:
        """
        Process an experience through the full pipeline
        """
        results = self.experience_pipeline.integrate_experience(experience)
        
        # Process value adaptations
        value_changes = self.value_engine.process_experience(experience)
        for change in value_changes:
            self.change_logger.log_change(change)
        results['value_adaptations'] = len(value_changes)
        
        # Check for evolution triggers
        if self.config['auto_evolve']:
            context = self._create_evolution_context()
            triggers = self.trigger_engine.evaluate_triggers(context)
            
            if triggers:
                evolution_results = self._execute_evolution(triggers, context)
                results['evolution'] = evolution_results
        
        # Check for maturation
        maturation_event = self.maturation_engine.check_maturation()
        if maturation_event:
            results['maturation'] = maturation_event
        
        return results
    
    def _create_evolution_context(self) -> EvolutionContext:
        """Create evolution context from current state"""
        return EvolutionContext(
            last_evolution=self.identity.last_evolution,
            current_xp=self.identity.experience_points,
            skill_levels=self.identity.skill_levels
        )
    
    def _execute_evolution(self, triggers: List[EvolutionTrigger], 
                           context: EvolutionContext) -> Dict:
        """Execute evolution based on triggers"""
        results = {'triggers': [t.value for t in triggers], 'changes': []}
        
        for trigger in triggers:
            changes = self.personality_engine.evolve(trigger, context)
            for change in changes:
                change_id = self.change_logger.log_change(change)
                results['changes'].append({
                    'id': change_id,
                    'component': change.component,
                    'delta': change.delta
                })
        
        return results
    
    def get_identity_summary(self) -> Dict:
        """Get current identity summary"""
        return {
            'soul_id': self.soul_core.soul_id,
            'maturation_level': self.identity.get_maturation_name(),
            'experience_points': self.identity.experience_points,
            'identity_version': round(self.identity.identity_version, 3),
            'personality': self.identity.personality.to_dict(),
            'adaptive_values': {
                k: v.current for k, v in self.identity.adaptive_values.items()
            },
            'skill_count': len(self.identity.skill_levels),
            'experience_count': len(self.experience_pipeline.experiences)
        }
    
    def get_evolution_history(self, limit: int = 50) -> List[Dict]:
        """Get evolution history"""
        changes = self.change_logger.get_change_history(limit)
        return [c.to_dict() for c in changes]
    
    def request_rollback(self, change_id: str, reason: str) -> Dict:
        """Request a rollback of a change"""
        return self.rollback_system.rollback(change_id, reason)
    
    def export_state(self) -> Dict:
        """Export full system state"""
        return {
            'soul_core': self.soul_core.to_dict(),
            'identity': self.identity.to_dict(),
            'maturation_history': self.maturation_engine.maturation_history,
            'config': self.config
        }
    
    def import_state(self, state: Dict):
        """Import system state"""
        # Implementation for state restoration
        pass

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of using the Soul Evolution System"""
    
    # Initialize the system
    soul = SoulEvolutionSystem()
    
    print("=== Initial State ===")
    print(json.dumps(soul.get_identity_summary(), indent=2))
    
    # Simulate some experiences
    experiences = [
        Experience(
            type=ExperienceType.TASK_COMPLETION,
            description="Completed email organization task",
            success=True,
            user_satisfaction=0.85,
            efficiency_score=0.9,
            skills_used=['email_management', 'organization'],
            skills_improved=['email_management'],
            complexity_score=0.6,
            lessons_learned=['User prefers chronological sorting']
        ),
        Experience(
            type=ExperienceType.PROBLEM_SOLVING,
            description="Debugged system integration issue",
            success=True,
            user_satisfaction=0.95,
            efficiency_score=0.75,
            skills_used=['debugging', 'system_analysis'],
            skills_improved=['debugging'],
            complexity_score=0.8,
            lessons_learned=['Check API rate limits first']
        ),
        Experience(
            type=ExperienceType.USER_INTERACTION,
            description="Helped user with complex query",
            success=True,
            user_satisfaction=0.9,
            skills_used=['communication', 'analysis'],
            complexity_score=0.5
        )
    ]
    
    for i, exp in enumerate(experiences):
        print(f"\n=== Processing Experience {i+1} ===")
        results = soul.process_experience(exp)
        print(f"XP Earned: {results.get('xp_earned', 0)}")
        print(f"Value Adaptations: {results.get('value_adaptations', 0)}")
        if 'evolution' in results:
            print(f"Evolution Triggered: {results['evolution']['triggers']}")
            print(f"Changes: {len(results['evolution']['changes'])}")
    
    print("\n=== Final State ===")
    print(json.dumps(soul.get_identity_summary(), indent=2))
    
    print("\n=== Evolution History ===")
    history = soul.get_evolution_history(10)
    for change in history:
        print(f"- {change['change_type']}: {change['component']} ({change['delta']:+.3f})")

if __name__ == "__main__":
    example_usage()
