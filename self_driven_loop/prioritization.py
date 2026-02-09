"""
Self-Driven Loop: Goal Prioritization and Scheduling System
===========================================================

Implements goal prioritization using multiple factors and dynamic weighting.
Includes dynamic scheduling based on priority, context, and constraints.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class PriorityLevel(Enum):
    """Priority levels for goals."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class ScheduledGoal:
    """Represents a scheduled goal."""
    goal: Any  # Goal object
    start_time: datetime
    end_time: datetime
    confidence: float
    scheduled_at: datetime = field(default_factory=datetime.now)


@dataclass
class TimeSlot:
    """Represents a time slot for scheduling."""
    start: datetime
    end: datetime
    confidence: float


@dataclass
class Schedule:
    """Represents a schedule of goals."""
    scheduled_goals: List[ScheduledGoal] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add(self, scheduled_goal: ScheduledGoal) -> None:
        """Add a scheduled goal."""
        self.scheduled_goals.append(scheduled_goal)
        # Keep sorted by start time
        self.scheduled_goals.sort(key=lambda x: x.start_time)
    
    def get_conflicts(self) -> List[Tuple[ScheduledGoal, ScheduledGoal]]:
        """Find scheduling conflicts."""
        conflicts = []
        for i, goal1 in enumerate(self.scheduled_goals):
            for goal2 in self.scheduled_goals[i+1:]:
                if (goal1.start_time < goal2.end_time and 
                    goal2.start_time < goal1.end_time):
                    conflicts.append((goal1, goal2))
        return conflicts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_count': len(self.scheduled_goals),
            'created_at': self.created_at.isoformat(),
            'goals': [
                {
                    'goal_id': sg.goal.id,
                    'start': sg.start_time.isoformat(),
                    'end': sg.end_time.isoformat(),
                    'confidence': sg.confidence
                }
                for sg in self.scheduled_goals
            ]
        }


class GoalPrioritizer:
    """
    Prioritizes goals using multiple factors and dynamic weighting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.priority_factors = {
            'intrinsic_motivation': self.config.get('motivation_weight', 0.20),
            'user_value': self.config.get('user_value_weight', 0.25),
            'urgency': self.config.get('urgency_weight', 0.20),
            'feasibility': self.config.get('feasibility_weight', 0.15),
            'strategic_alignment': self.config.get('alignment_weight', 0.10),
            'resource_efficiency': self.config.get('efficiency_weight', 0.10)
        }
        
        # Historical tracking
        self.goal_success_rates: Dict[str, List[bool]] = {}
    
    def prioritize_goals(self, goals: List[Any],
                        context: Dict[str, Any]) -> List[Any]:
        """Prioritize goals and return sorted list."""
        
        scored_goals = []
        
        for goal in goals:
            score = self._calculate_priority_score(goal, context)
            scored_goals.append((goal, score))
        
        # Sort by score descending
        scored_goals.sort(key=lambda x: x[1], reverse=True)
        
        return [goal for goal, _ in scored_goals]
    
    def _calculate_priority_score(self, goal: Any,
                                 context: Dict[str, Any]) -> float:
        """Calculate comprehensive priority score for a goal."""
        
        # Calculate individual factor scores
        motivation_score = self._score_intrinsic_motivation(goal, context)
        value_score = self._score_user_value(goal, context)
        urgency_score = self._score_urgency(goal)
        feasibility_score = self._score_feasibility(goal, context)
        alignment_score = self._score_strategic_alignment(goal, context)
        efficiency_score = self._score_resource_efficiency(goal)
        
        # Get dynamic weights
        weights = self._get_dynamic_weights(context)
        
        # Calculate weighted sum
        total_score = (
            weights['intrinsic_motivation'] * motivation_score +
            weights['user_value'] * value_score +
            weights['urgency'] * urgency_score +
            weights['feasibility'] * feasibility_score +
            weights['strategic_alignment'] * alignment_score +
            weights['resource_efficiency'] * efficiency_score
        )
        
        return total_score
    
    def _score_intrinsic_motivation(self, goal: Any,
                                   context: Dict[str, Any]) -> float:
        """Score based on how well goal satisfies intrinsic drives."""
        
        motivation = context.get('motivation', {})
        goal_type = getattr(goal, 'type', None)
        
        if goal_type is None:
            return 0.5
        
        type_name = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)
        
        if 'competence' in type_name.lower():
            return motivation.get('competence', 0.5)
        elif 'autonomy' in type_name.lower():
            return motivation.get('autonomy', 0.5)
        elif 'curiosity' in type_name.lower():
            return motivation.get('curiosity', 0.5)
        elif 'relatedness' in type_name.lower():
            return motivation.get('relatedness', 0.5)
        
        return 0.5
    
    def _score_user_value(self, goal: Any, context: Dict[str, Any]) -> float:
        """Score based on expected value to user."""
        
        # Check if goal has explicit user value
        user_value = getattr(goal, 'user_value', None)
        if user_value is not None:
            return user_value
        
        # Infer from goal type
        goal_type = getattr(goal, 'type', None)
        if goal_type:
            type_name = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)
            if 'user' in type_name.lower():
                return 0.9
        
        # Check user priority
        user_priority = context.get('user_priority', 0.5)
        
        return user_priority
    
    def _score_urgency(self, goal: Any) -> float:
        """Score based on time sensitivity."""
        
        deadline = getattr(goal, 'deadline', None)
        
        if deadline is None:
            return 0.3  # Low urgency for no deadline
        
        time_remaining = (deadline - datetime.now()).total_seconds()
        
        if time_remaining <= 0:
            return 1.0  # Overdue
        
        estimated_effort = getattr(goal, 'estimated_effort', 1.0)
        estimated_duration = estimated_effort * 3600  # Convert hours to seconds
        
        urgency_ratio = estimated_duration / time_remaining
        
        # Scale to 0-1
        if urgency_ratio >= 1.0:
            return 1.0
        elif urgency_ratio >= 0.5:
            return 0.7 + (urgency_ratio - 0.5) * 0.6
        else:
            return urgency_ratio * 1.4
    
    def _score_feasibility(self, goal: Any, context: Dict[str, Any]) -> float:
        """Score based on likelihood of successful completion."""
        
        # Check capability match
        capability_match = self._calculate_capability_match(goal, context)
        
        # Check resource availability
        resource_availability = self._check_resource_availability(goal)
        
        # Check complexity vs skill level
        complexity_match = self._assess_complexity_match(goal, context)
        
        # Historical success rate for similar goals
        historical_success = self._get_historical_success_rate(goal)
        
        return (
            0.3 * capability_match +
            0.25 * resource_availability +
            0.25 * complexity_match +
            0.2 * historical_success
        )
    
    def _calculate_capability_match(self, goal: Any,
                                   context: Dict[str, Any]) -> float:
        """Calculate how well agent capabilities match goal requirements."""
        required_capabilities = getattr(goal, 'required_capabilities', [])
        
        if not required_capabilities:
            return 0.8  # Assume good match if no requirements
        
        agent_capabilities = context.get('capabilities', [])
        
        matches = sum(1 for cap in required_capabilities if cap in agent_capabilities)
        return matches / len(required_capabilities)
    
    def _check_resource_availability(self, goal: Any) -> float:
        """Check if required resources are available."""
        required_resources = getattr(goal, 'resources_needed', [])
        
        if not required_resources:
            return 1.0
        
        # Placeholder - would check actual resource availability
        return 0.8
    
    def _assess_complexity_match(self, goal: Any,
                                context: Dict[str, Any]) -> float:
        """Assess if goal complexity matches agent skill level."""
        difficulty = getattr(goal, 'difficulty', 0.5)
        skill_level = context.get('skill_level', 0.5)
        
        # Optimal difficulty is slightly above skill level
        optimal = skill_level * 1.1
        diff = abs(difficulty - optimal)
        
        return max(0.0, 1.0 - diff)
    
    def _get_historical_success_rate(self, goal: Any) -> float:
        """Get historical success rate for similar goals."""
        goal_type = getattr(goal, 'type', None)
        
        if goal_type is None:
            return 0.5
        
        type_key = str(goal_type)
        
        if type_key not in self.goal_success_rates:
            return 0.5
        
        history = self.goal_success_rates[type_key]
        if not history:
            return 0.5
        
        return sum(history) / len(history)
    
    def _score_strategic_alignment(self, goal: Any,
                                  context: Dict[str, Any]) -> float:
        """Score based on alignment with strategic objectives using TF-IDF cosine similarity."""

        strategic_objectives = context.get('strategic_objectives', [])

        if not strategic_objectives:
            return 0.5

        goal_description = getattr(goal, 'description', '')
        if not goal_description:
            return 0.5

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            texts = [goal_description] + list(strategic_objectives)
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            return float(max(similarities)) if len(similarities) > 0 else 0.5
        except ImportError:
            # Jaccard fallback
            alignment_scores = []
            goal_words = set(goal_description.lower().split())
            for objective in strategic_objectives:
                obj_words = set(objective.lower().split())
                if obj_words and goal_words:
                    alignment_scores.append(
                        len(obj_words & goal_words) / len(obj_words | goal_words)
                    )
            return max(alignment_scores) if alignment_scores else 0.5
    
    def _score_resource_efficiency(self, goal: Any) -> float:
        """Score based on resource efficiency."""
        
        estimated_effort = getattr(goal, 'estimated_effort', 1.0)
        
        # Prefer goals with good value/effort ratio
        # Lower effort is better, but not at expense of value
        
        # Normalize effort (assume 0.5-10 hour range)
        normalized_effort = min(1.0, estimated_effort / 10.0)
        
        # Efficiency is inverse of normalized effort
        return 1.0 - normalized_effort * 0.5  # Scale so max is 1.0
    
    def _get_dynamic_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get dynamically adjusted priority weights."""
        weights = self.priority_factors.copy()
        
        # Adjust based on context
        
        # Increase urgency weight if deadlines approaching
        if context.get('deadlines_approaching', False):
            weights['urgency'] += 0.1
            weights['feasibility'] -= 0.05
            weights['resource_efficiency'] -= 0.05
        
        # Increase motivation weight if agent needs engagement
        if context.get('motivation', {}).get('total', 0.5) < 0.4:
            weights['intrinsic_motivation'] += 0.1
            weights['strategic_alignment'] -= 0.05
            weights['resource_efficiency'] -= 0.05
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def record_goal_outcome(self, goal: Any, success: bool) -> None:
        """Record outcome for learning."""
        goal_type = getattr(goal, 'type', None)
        
        if goal_type is None:
            return
        
        type_key = str(goal_type)
        
        if type_key not in self.goal_success_rates:
            self.goal_success_rates[type_key] = []
        
        self.goal_success_rates[type_key].append(success)
        
        # Keep only last 50 outcomes
        if len(self.goal_success_rates[type_key]) > 50:
            self.goal_success_rates[type_key].pop(0)


class DynamicScheduler:
    """
    Dynamically schedules goals based on priority, context, and constraints.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.constraints: List[Any] = []
        self.energy_pattern = self.config.get('energy_pattern', {})
        
    def create_schedule(self, goals: List[Any],
                       time_horizon: timedelta = timedelta(days=7)) -> Schedule:
        """Create optimized schedule for goals."""
        
        # Filter goals within time horizon
        feasible_goals = []
        for goal in goals:
            deadline = getattr(goal, 'deadline', None)
            if deadline is None or deadline <= datetime.now() + time_horizon:
                feasible_goals.append(goal)
        
        # Prioritize goals
        prioritizer = GoalPrioritizer(self.config)
        context = self._get_scheduling_context()
        prioritized_goals = prioritizer.prioritize_goals(feasible_goals, context)
        
        # Create schedule
        schedule = Schedule()
        current_time = datetime.now()
        
        for goal in prioritized_goals:
            # Find best slot
            slot = self._find_optimal_slot(goal, current_time)
            
            if slot:
                scheduled = ScheduledGoal(
                    goal=goal,
                    start_time=slot.start,
                    end_time=slot.end,
                    confidence=slot.confidence
                )
                schedule.add(scheduled)
                current_time = slot.end
        
        return schedule
    
    def _get_scheduling_context(self) -> Dict[str, Any]:
        """Get context for scheduling."""
        return {
            'motivation': {'total': 0.6, 'competence': 0.6, 'autonomy': 0.6},
            'capabilities': ['planning', 'execution', 'learning'],
            'skill_level': 0.6,
            'strategic_objectives': ['improve_efficiency', 'learn_new_skills']
        }
    
    def _find_optimal_slot(self, goal: Any,
                          after: datetime) -> Optional[TimeSlot]:
        """Find optimal time slot for goal execution."""
        
        estimated_effort = getattr(goal, 'estimated_effort', 1.0)
        duration = timedelta(hours=estimated_effort)
        
        # Consider energy patterns
        energy_pattern = self._get_energy_pattern()
        
        # Consider context requirements
        required_context = getattr(goal, 'required_context', {})
        
        # Search for slot
        for offset_hours in range(0, 168):  # Look up to 1 week ahead
            candidate_start = after + timedelta(hours=offset_hours)
            candidate_end = candidate_start + duration
            
            # Check constraints
            if self._is_valid_slot(candidate_start, candidate_end, goal):
                # Score slot quality
                quality = self._score_slot_quality(
                    candidate_start, candidate_end,
                    goal, energy_pattern
                )
                
                if quality > 0.7:
                    return TimeSlot(
                        start=candidate_start,
                        end=candidate_end,
                        confidence=quality
                    )
        
        return None
    
    def _get_energy_pattern(self) -> Dict[int, float]:
        """Get energy level pattern throughout the day."""
        # Default pattern: higher energy in morning and afternoon
        return {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4,
            6: 0.6, 7: 0.8, 8: 0.9, 9: 0.9, 10: 0.85, 11: 0.8,
            12: 0.7, 13: 0.75, 14: 0.85, 15: 0.9, 16: 0.85, 17: 0.8,
            18: 0.7, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.3
        }
    
    def _is_valid_slot(self, start: datetime, end: datetime, goal: Any) -> bool:
        """Check if time slot is valid for goal."""
        # Check deadline
        deadline = getattr(goal, 'deadline', None)
        if deadline and end > deadline:
            return False
        
        # Check business hours if required
        required_context = getattr(goal, 'required_context', {})
        if required_context.get('business_hours_only', False):
            if start.hour < 9 or end.hour > 17:
                return False
        
        return True
    
    def _score_slot_quality(self, start: datetime, end: datetime,
                           goal: Any, energy_pattern: Dict[int, float]) -> float:
        """Score quality of time slot for goal."""
        
        # Energy match
        avg_energy = np.mean([
            energy_pattern.get(hour, 0.5)
            for hour in range(start.hour, end.hour + 1)
        ])
        
        # Deadline proximity bonus
        deadline = getattr(goal, 'deadline', None)
        deadline_bonus = 0.0
        if deadline:
            time_to_deadline = (deadline - end).total_seconds()
            if time_to_deadline < 86400:  # Less than 24 hours
                deadline_bonus = 0.2
        
        # Goal type preference
        goal_type = getattr(goal, 'type', None)
        type_preference = 0.5
        
        if goal_type:
            type_name = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)
            if 'competence' in type_name.lower() and avg_energy > 0.7:
                type_preference = 0.9
            elif 'curiosity' in type_name.lower() and avg_energy > 0.5:
                type_preference = 0.8
        
        return min(1.0, avg_energy * 0.4 + deadline_bonus + type_preference * 0.4)


# Singleton instances
_prioritizer: Optional[GoalPrioritizer] = None
_scheduler: Optional[DynamicScheduler] = None


def get_prioritizer(config: Optional[Dict] = None) -> GoalPrioritizer:
    """Get or create the global prioritizer instance."""
    global _prioritizer
    if _prioritizer is None:
        _prioritizer = GoalPrioritizer(config)
    return _prioritizer


def get_scheduler(config: Optional[Dict] = None) -> DynamicScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DynamicScheduler(config)
    return _scheduler


if __name__ == "__main__":
    # Example usage
    from goal_system import Goal, GoalType
    
    prioritizer = GoalPrioritizer()
    
    goals = [
        Goal(
            title="Learn new API",
            type=GoalType.COMPETENCE,
            priority=0.7,
            deadline=datetime.now() + timedelta(days=3),
            estimated_effort=5.0
        ),
        Goal(
            title="Optimize system",
            type=GoalType.SYSTEM,
            priority=0.8,
            deadline=datetime.now() + timedelta(days=1),
            estimated_effort=3.0
        ),
        Goal(
            title="Explore new topic",
            type=GoalType.CURIOSITY,
            priority=0.5,
            estimated_effort=2.0
        )
    ]
    
    context = {
        'motivation': {'competence': 0.7, 'curiosity': 0.8, 'autonomy': 0.6},
        'capabilities': ['coding', 'analysis', 'learning'],
        'skill_level': 0.7
    }
    
    prioritized = prioritizer.prioritize_goals(goals, context)
    
    print("Prioritized goals:")
    for i, goal in enumerate(prioritized, 1):
        print(f"  {i}. {goal.title} (Type: {goal.type.value})")
    
    # Test scheduler
    scheduler = DynamicScheduler()
    schedule = scheduler.create_schedule(goals)
    
    print(f"\nSchedule created with {len(schedule.scheduled_goals)} goals")
