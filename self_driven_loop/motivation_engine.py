"""
Self-Driven Loop: Intrinsic Motivation Engine
=============================================

Implements intrinsic motivation algorithms based on Self-Determination Theory (SDT).
Generates internal drive signals for autonomous AI agent behavior.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any
from collections import deque
from datetime import datetime
from enum import Enum
import numpy as np


class DriveType(Enum):
    """Types of intrinsic drives."""
    AUTONOMY = "autonomy"
    COMPETENCE = "competence"
    RELATEDNESS = "relatedness"
    CURIOSITY = "curiosity"


@dataclass
class MotivationState:
    """Represents the current motivation state of the agent."""
    total: float = 0.5
    autonomy: float = 0.5
    competence: float = 0.5
    relatedness: float = 0.5
    curiosity: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    trend: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'autonomy': self.autonomy,
            'competence': self.competence,
            'relatedness': self.relatedness,
            'curiosity': self.curiosity,
            'timestamp': self.timestamp.isoformat(),
            'trend': self.trend
        }


@dataclass
class AgentContext:
    """Context information for motivation calculation."""
    # Action tracking
    self_initiated_actions: int = 0
    total_actions: int = 0
    available_options: int = 0
    max_options: int = 10
    constraints_applied: int = 0
    constraints_possible: int = 10
    
    # Performance tracking
    successful_actions: int = 0
    total_attempts: int = 0
    current_skill_level: float = 0.5
    previous_skill_level: float = 0.5
    task_difficulty: float = 0.5
    
    # User interaction
    meaningful_exchanges: int = 0
    total_interactions: int = 0
    user_feedback_score: float = 0.5
    tasks_helping_others: int = 0
    total_tasks: int = 0
    
    # Environment
    environment_novelty: float = 0.5
    expected_information_gain: float = 0.5
    unexplored_opportunity_ratio: float = 0.5
    
    # State indicators
    reactive_ratio: float = 0.5
    recent_failure_rate: float = 0.0
    user_engagement_score: float = 0.5


class AutonomyDrive:
    """
    Measures the agent's sense of self-direction and choice.
    Higher when agent makes decisions independently.
    """
    
    def calculate(self, context: AgentContext) -> float:
        """Calculate autonomy drive score."""
        # Factors contributing to autonomy
        self_determined_actions = (
            context.self_initiated_actions / max(context.total_actions, 1)
        )
        choice_variety = context.available_options / max(context.max_options, 1)
        constraint_ratio = 1.0 - (
            context.constraints_applied / max(context.constraints_possible, 1)
        )
        
        autonomy_score = (
            0.4 * self_determined_actions +
            0.3 * choice_variety +
            0.3 * constraint_ratio
        )
        
        return min(max(autonomy_score, 0.0), 1.0)


class CompetenceDrive:
    """
    Measures the agent's sense of effectiveness and mastery.
    Higher when agent successfully completes challenging tasks.
    """
    
    def calculate(self, context: AgentContext) -> float:
        """Calculate competence drive score."""
        # Skill progression tracking
        success_rate = context.successful_actions / max(context.total_attempts, 1)
        skill_growth = context.current_skill_level - context.previous_skill_level
        challenge_match = self._optimal_challenge_ratio(
            context.task_difficulty,
            context.current_skill_level
        )
        
        competence_score = (
            0.35 * success_rate +
            0.35 * self._normalize(skill_growth) +
            0.30 * challenge_match
        )
        
        return min(max(competence_score, 0.0), 1.0)
    
    def _optimal_challenge_ratio(self, difficulty: float, skill: float) -> float:
        """Optimal challenge is slightly above current skill (Vygotsky's ZPD)."""
        optimal_zone = 1.1  # 10% above current skill
        ratio = difficulty / (skill * optimal_zone + 0.01)
        # Peak at 1.0 (optimal), decrease as we move away
        return max(0.0, 1.0 - abs(ratio - 1.0))
    
    def _normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        return min(max((value + 1) / 2, 0.0), 1.0)


class RelatednessDrive:
    """
    Measures the agent's sense of connection and contribution.
    Higher when agent meaningfully interacts with users/system.
    """
    
    def calculate(self, context: AgentContext) -> float:
        """Calculate relatedness drive score."""
        # Connection quality metrics
        meaningful_interactions = (
            context.meaningful_exchanges / max(context.total_interactions, 1)
        )
        user_satisfaction = context.user_feedback_score
        contribution_impact = (
            context.tasks_helping_others / max(context.total_tasks, 1)
        )
        
        relatedness_score = (
            0.3 * meaningful_interactions +
            0.4 * user_satisfaction +
            0.3 * contribution_impact
        )
        
        return min(max(relatedness_score, 0.0), 1.0)


class IntrinsicMotivationEngine:
    """
    Combines all intrinsic drives into unified motivation signal.
    Implements dynamic weighting based on agent state.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.autonomy_drive = AutonomyDrive()
        self.competence_drive = CompetenceDrive()
        self.relatedness_drive = RelatednessDrive()
        
        self.config = config or {}
        self.drive_weights = {
            'autonomy': self.config.get('autonomy_weight', 0.33),
            'competence': self.config.get('competence_weight', 0.33),
            'relatedness': self.config.get('relatedness_weight', 0.34)
        }
        self.curiosity_bonus_max = self.config.get('curiosity_bonus_max', 0.1)
        
        self.motivation_history: Deque[MotivationState] = deque(maxlen=1000)
        
    def calculate_motivation(self, context: AgentContext) -> MotivationState:
        """Calculate current intrinsic motivation level."""
        
        # Calculate individual drives
        autonomy = self.autonomy_drive.calculate(context)
        competence = self.competence_drive.calculate(context)
        relatedness = self.relatedness_drive.calculate(context)
        
        # Apply dynamic weighting based on current needs
        weights = self._adjust_weights(context)
        
        # Combined motivation score
        total_motivation = (
            weights['autonomy'] * autonomy +
            weights['competence'] * competence +
            weights['relatedness'] * relatedness
        )
        
        # Add curiosity bonus
        curiosity_bonus = self._calculate_curiosity_bonus(context)
        total_motivation = min(1.0, total_motivation + curiosity_bonus)
        
        motivation_state = MotivationState(
            total=total_motivation,
            autonomy=autonomy,
            competence=competence,
            relatedness=relatedness,
            curiosity=curiosity_bonus,
            timestamp=datetime.now(),
            trend=self._calculate_trend()
        )
        
        self.motivation_history.append(motivation_state)
        return motivation_state
    
    def _adjust_weights(self, context: AgentContext) -> Dict[str, float]:
        """Dynamically adjust drive weights based on current state."""
        weights = self.drive_weights.copy()
        
        # Increase autonomy weight if agent has been reactive
        if context.reactive_ratio > 0.7:
            weights['autonomy'] += 0.1
            weights['competence'] -= 0.05
            weights['relatedness'] -= 0.05
        
        # Increase competence weight if many failures
        if context.recent_failure_rate > 0.5:
            weights['competence'] += 0.15
            weights['autonomy'] -= 0.075
            weights['relatedness'] -= 0.075
        
        # Increase relatedness weight if low user engagement
        if context.user_engagement_score < 0.3:
            weights['relatedness'] += 0.1
            weights['autonomy'] -= 0.05
            weights['competence'] -= 0.05
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _calculate_curiosity_bonus(self, context: AgentContext) -> float:
        """Calculate additional motivation from curiosity."""
        novelty_score = context.environment_novelty
        information_gain = context.expected_information_gain
        exploration_value = context.unexplored_opportunity_ratio
        
        avg_signal = (novelty_score + information_gain + exploration_value) / 3
        return self.curiosity_bonus_max * avg_signal
    
    def _calculate_trend(self) -> float:
        """Calculate motivation trend over recent history."""
        if len(self.motivation_history) < 10:
            return 0.0
        
        recent = list(self.motivation_history)[-10:]
        values = [m.total for m in recent]
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope
    
    def get_motivation_summary(self) -> Dict[str, Any]:
        """Get summary of motivation state."""
        if not self.motivation_history:
            return {'status': 'no_data'}
        
        recent = list(self.motivation_history)[-50:]
        
        return {
            'current': self.motivation_history[-1].to_dict(),
            'average': {
                'total': np.mean([m.total for m in recent]),
                'autonomy': np.mean([m.autonomy for m in recent]),
                'competence': np.mean([m.competence for m in recent]),
                'relatedness': np.mean([m.relatedness for m in recent]),
                'curiosity': np.mean([m.curiosity for m in recent])
            },
            'trend': self._calculate_trend(),
            'history_length': len(self.motivation_history)
        }


# Singleton instance for global access
_motivation_engine: Optional[IntrinsicMotivationEngine] = None


def get_motivation_engine(config: Optional[Dict] = None) -> IntrinsicMotivationEngine:
    """Get or create the global motivation engine instance."""
    global _motivation_engine
    if _motivation_engine is None:
        _motivation_engine = IntrinsicMotivationEngine(config)
    return _motivation_engine


def reset_motivation_engine() -> None:
    """Reset the global motivation engine instance."""
    global _motivation_engine
    _motivation_engine = None


if __name__ == "__main__":
    # Example usage
    engine = IntrinsicMotivationEngine()
    
    context = AgentContext(
        self_initiated_actions=70,
        total_actions=100,
        available_options=8,
        max_options=10,
        successful_actions=85,
        total_attempts=100,
        current_skill_level=0.7,
        previous_skill_level=0.65,
        user_feedback_score=0.8,
        environment_novelty=0.6
    )
    
    motivation = engine.calculate_motivation(context)
    print(f"Motivation State: {motivation}")
    print(f"Summary: {engine.get_motivation_summary()}")
