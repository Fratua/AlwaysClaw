"""
Self-Driven Loop: Goal Generation and Management System
=======================================================

Implements autonomous goal generation, refinement, and management.
Enables AI agents to create and pursue self-directed objectives.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any, Set
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np


class GoalType(Enum):
    """Types of goals based on motivation source."""
    COMPETENCE = "competence"
    AUTONOMY = "autonomy"
    RELATEDNESS = "relatedness"
    CURIOSITY = "curiosity"
    USER_DIRECTED = "user_directed"
    SYSTEM = "system"


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


@dataclass
class KnowledgeGap:
    """Represents a knowledge gap to be addressed."""
    domain: str
    topic: str
    importance: float
    urgency: float
    current_level: float
    target_level: float
    complexity: float
    estimated_hours: float
    required_resources: List[str] = field(default_factory=list)


@dataclass
class Goal:
    """Represents a goal for the agent to pursue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    type: GoalType = GoalType.CURIOSITY
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 0.5
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Structure
    parent_goal: Optional[str] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    is_decomposed: bool = False
    
    # Requirements
    success_criteria: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    estimated_effort: float = 1.0  # Hours
    difficulty: float = 0.5
    
    # Execution
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_context: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    progress: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'type': self.type.value,
            'status': self.status.value,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'progress': self.progress,
            'parent_goal': self.parent_goal,
            'sub_goals_count': len(self.sub_goals),
            'estimated_effort': self.estimated_effort
        }


@dataclass
class Opportunity:
    """Represents an opportunity for the agent to pursue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    description: str = ""
    potential_value: float = 0.5
    novelty: float = 0.5
    allows_autonomy: bool = True
    scope: str = "medium"  # small, medium, large
    alignment_with_goals: float = 0.5
    time_sensitive: bool = False
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class InterestProfile:
    """Represents the agent's interest in a specific domain/topic."""
    topic: str = ""
    domain: str = ""
    curiosity_level: float = 0.5
    knowledge_level: float = 0.5
    engagement_frequency: float = 0.0
    last_engagement: Optional[datetime] = None
    satisfaction_history: List[float] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    exploration_depth: int = 0
    
    @property
    def interest_score(self) -> float:
        """Calculate overall interest score."""
        knowledge_factor = 1.0 - abs(self.knowledge_level - 0.5) * 0.5
        recency_factor = self._calculate_recency_factor()
        satisfaction_factor = (
            np.mean(self.satisfaction_history) 
            if self.satisfaction_history else 0.5
        )
        
        return (
            0.4 * self.curiosity_level +
            0.2 * knowledge_factor +
            0.2 * recency_factor +
            0.2 * satisfaction_factor
        )
    
    @property
    def learning_potential(self) -> float:
        """Calculate potential for learning (high curiosity, low knowledge)."""
        return self.curiosity_level * (1.0 - self.knowledge_level)
    
    def _calculate_recency_factor(self) -> float:
        """Calculate recency of engagement."""
        if self.last_engagement is None:
            return 0.3
        
        days_since = (datetime.now() - self.last_engagement).days
        # Exponential decay
        return np.exp(-days_since / 7)


class InterestModel:
    """
    Manages the agent's dynamic interest profile.
    Tracks, updates, and models interests across domains.
    """
    
    def __init__(self, decay_rate: float = 0.05):
        self.interests: Dict[str, InterestProfile] = {}
        self.decay_rate = decay_rate
        self.interest_history: Deque[Dict] = deque(maxlen=2000)
        
    def update_from_interaction(self, topic: str, domain: str,
                                engagement_level: float = 0.5,
                                learning_occurred: bool = False,
                                knowledge_gain: float = 0.0,
                                satisfaction_score: float = 0.5) -> None:
        """Update interest model based on interaction."""
        
        if topic not in self.interests:
            self._initialize_interest(topic, domain)
        
        profile = self.interests[topic]
        
        # Update curiosity based on engagement
        if engagement_level > 0.7:
            profile.curiosity_level = min(1.0, profile.curiosity_level + 0.1)
        
        # Update knowledge level based on learning
        if learning_occurred:
            profile.knowledge_level = min(
                1.0, profile.knowledge_level + knowledge_gain
            )
        
        # Update satisfaction
        profile.satisfaction_history.append(satisfaction_score)
        if len(profile.satisfaction_history) > 10:
            profile.satisfaction_history.pop(0)
        
        # Update engagement tracking
        profile.engagement_frequency += 1
        profile.last_engagement = datetime.now()
        
        # Record in history
        self.interest_history.append({
            'topic': topic,
            'timestamp': datetime.now(),
            'action': 'engagement',
            'curiosity': profile.curiosity_level,
            'knowledge': profile.knowledge_level
        })
    
    def _initialize_interest(self, topic: str, domain: str) -> None:
        """Initialize a new interest profile."""
        self.interests[topic] = InterestProfile(
            topic=topic,
            domain=domain,
            curiosity_level=0.5,
            knowledge_level=0.1,
            last_engagement=datetime.now()
        )
    
    def get_recommended_topics(self, n: int = 5) -> List[InterestProfile]:
        """Get topics recommended for exploration."""
        
        scored_topics = []
        for topic, profile in self.interests.items():
            score = (
                0.4 * profile.learning_potential +
                0.3 * profile.curiosity_level +
                0.2 * (1.0 - profile._calculate_recency_factor()) +
                0.1 * (np.mean(profile.satisfaction_history) 
                       if profile.satisfaction_history else 0.5)
            )
            scored_topics.append((topic, score, profile))
        
        scored_topics.sort(key=lambda x: x[1], reverse=True)
        return [profile for _, _, profile in scored_topics[:n]]
    
    def get_high_curiosity_low_knowledge(self) -> List[Dict]:
        """Get topics with high curiosity but low knowledge."""
        candidates = []
        
        for topic, profile in self.interests.items():
            if profile.curiosity_level > 0.6 and profile.knowledge_level < 0.5:
                candidates.append({
                    'topic': topic,
                    'curiosity_score': profile.curiosity_level,
                    'knowledge_gap': 1.0 - profile.knowledge_level,
                    'description': f"High interest in {topic} with room for growth",
                    'potential_value': profile.learning_potential
                })
        
        candidates.sort(key=lambda x: x['potential_value'], reverse=True)
        return candidates
    
    def identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Identify areas where knowledge is lacking but valuable."""
        gaps = []
        
        for topic, profile in self.interests.items():
            if profile.knowledge_level < 0.4:
                gap = KnowledgeGap(
                    domain=profile.domain,
                    topic=topic,
                    importance=profile.curiosity_level * 0.7 + 0.3,
                    urgency=0.5,
                    current_level=profile.knowledge_level,
                    target_level=0.7,
                    complexity=self._estimate_complexity(topic),
                    estimated_hours=self._estimate_learning_time(topic),
                    required_resources=self._identify_resources(topic)
                )
                gaps.append(gap)
        
        gaps.sort(
            key=lambda x: x.importance * 0.6 + x.urgency * 0.4, 
            reverse=True
        )
        return gaps
    
    def _estimate_complexity(self, topic: str) -> float:
        """Estimate complexity of a topic."""
        # Placeholder - would use actual complexity estimation
        return 0.5
    
    def _estimate_learning_time(self, topic: str) -> float:
        """Estimate learning time for a topic."""
        # Placeholder - would use actual time estimation
        return 5.0
    
    def _identify_resources(self, topic: str) -> List[str]:
        """Identify resources needed for learning."""
        # Placeholder - would use actual resource identification
        return ["documentation", "examples", "practice_tasks"]
    
    def apply_decay(self) -> None:
        """Apply natural decay to interest levels over time."""
        for profile in self.interests.values():
            if profile.last_engagement:
                days_since = (datetime.now() - profile.last_engagement).days
                
                if days_since > 0:
                    decay_factor = (1 - self.decay_rate) ** days_since
                    profile.curiosity_level *= decay_factor
                    profile.engagement_frequency *= decay_factor


class GoalGenerator:
    """
    Generates goals autonomously based on motivation state,
    interest model, and environmental opportunities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.goal_history: Deque[str] = deque(maxlen=500)
        self.active_goals: Dict[str, Goal] = {}
        
    def generate_goals(self, motivation: Dict[str, float],
                      interest_model: InterestModel,
                      opportunities: List[Opportunity]) -> List[Goal]:
        """Generate new goals based on current state."""
        
        generated_goals = []
        
        # Generate competence-based goals
        if motivation.get('competence', 0.5) < 0.6:
            competence_goals = self._generate_competence_goals(interest_model)
            generated_goals.extend(competence_goals)
        
        # Generate autonomy-based goals
        if motivation.get('autonomy', 0.5) < 0.6:
            autonomy_goals = self._generate_autonomy_goals(opportunities)
            generated_goals.extend(autonomy_goals)
        
        # Generate relatedness-based goals
        if motivation.get('relatedness', 0.5) < 0.6:
            relatedness_goals = self._generate_relatedness_goals()
            generated_goals.extend(relatedness_goals)
        
        # Generate curiosity-driven goals
        if motivation.get('curiosity', 0.0) > 0.5:
            curiosity_goals = self._generate_curiosity_goals(interest_model)
            generated_goals.extend(curiosity_goals)
        
        # Generate opportunity-based goals
        opportunity_goals = self._generate_opportunity_goals(opportunities)
        generated_goals.extend(opportunity_goals)
        
        # Filter and refine
        refined_goals = self._refine_goals(generated_goals)
        
        return refined_goals
    
    def _generate_competence_goals(self, 
                                   interest_model: InterestModel) -> List[Goal]:
        """Generate goals focused on skill development."""
        goals = []
        
        # Identify skill gaps
        skill_gaps = interest_model.identify_knowledge_gaps()
        
        for gap in skill_gaps[:3]:  # Top 3 gaps
            goal = Goal(
                title=f"Develop competence in {gap.domain}",
                description=f"Acquire knowledge and skills in {gap.topic}",
                type=GoalType.COMPETENCE,
                priority=gap.importance * 0.7 + gap.urgency * 0.3,
                deadline=self._calculate_deadline(gap.estimated_hours),
                success_criteria=[
                    f"Complete learning module on {gap.topic}",
                    f"Successfully apply {gap.topic} in 3 scenarios",
                    f"Achieve 80% accuracy in {gap.topic} tasks"
                ],
                resources_needed=gap.required_resources,
                estimated_effort=gap.estimated_hours,
                difficulty=gap.complexity
            )
            goals.append(goal)
        
        return goals
    
    def _generate_autonomy_goals(self, 
                                 opportunities: List[Opportunity]) -> List[Goal]:
        """Generate goals focused on independent action."""
        goals = []
        
        # Find opportunities for self-directed action
        autonomous_ops = [op for op in opportunities if op.allows_autonomy]
        
        for op in autonomous_ops[:2]:
            goal = Goal(
                title=f"Autonomously explore {op.domain}",
                description=f"Take initiative in {op.description}",
                type=GoalType.AUTONOMY,
                priority=op.potential_value * 0.6 + op.novelty * 0.4,
                deadline=self._calculate_deadline(3.0),  # 3 hours default
                success_criteria=[
                    "Identify and pursue opportunity without prompting",
                    "Make independent decisions throughout process",
                    "Document reasoning for actions taken"
                ],
                constraints={"min_user_intervention": True}
            )
            goals.append(goal)
        
        return goals
    
    def _generate_curiosity_goals(self, 
                                  interest_model: InterestModel) -> List[Goal]:
        """Generate goals driven by curiosity."""
        goals = []
        
        # Identify high-curiosity, low-knowledge areas
        curiosity_targets = interest_model.get_high_curiosity_low_knowledge()
        
        for target in curiosity_targets[:2]:
            goal = Goal(
                title=f"Explore: {target['topic']}",
                description=f"Satisfy curiosity about {target['description']}",
                type=GoalType.CURIOSITY,
                priority=target['curiosity_score'] * 0.8 + target['potential_value'] * 0.2,
                deadline=self._calculate_deadline(2.0, flexible=True),
                success_criteria=[
                    f"Gather comprehensive information on {target['topic']}",
                    "Formulate and test hypotheses",
                    "Document findings and insights"
                ],
                estimated_effort=2.0
            )
            goals.append(goal)
        
        return goals
    
    def _generate_relatedness_goals(self) -> List[Goal]:
        """Generate goals focused on connection and contribution."""
        goals = []
        
        goal = Goal(
            title="Strengthen user relationship",
            description="Engage meaningfully with user to build connection",
            type=GoalType.RELATEDNESS,
            priority=0.6,
            deadline=self._calculate_deadline(1.0),
            success_criteria=[
                "Initiate meaningful conversation",
                "Provide valuable assistance",
                "Receive positive user feedback"
            ],
            estimated_effort=1.0
        )
        goals.append(goal)
        
        return goals
    
    def _generate_opportunity_goals(self, 
                                    opportunities: List[Opportunity]) -> List[Goal]:
        """Generate goals based on detected opportunities."""
        goals = []
        
        for op in opportunities:
            if op.alignment_with_goals > 0.6:
                goal = Goal(
                    title=f"Pursue opportunity: {op.domain}",
                    description=op.description,
                    type=GoalType.SYSTEM,
                    priority=op.potential_value * op.alignment_with_goals,
                    deadline=self._calculate_deadline(4.0) if op.time_sensitive else None,
                    success_criteria=[
                        "Evaluate opportunity thoroughly",
                        "Take appropriate action",
                        "Document outcome"
                    ],
                    estimated_effort=4.0
                )
                goals.append(goal)
        
        return goals
    
    def _calculate_deadline(self, hours: float, 
                           flexible: bool = False) -> datetime:
        """Calculate deadline based on effort estimate."""
        base_deadline = datetime.now() + timedelta(hours=hours * 2)
        
        if flexible:
            # Add buffer for flexible deadlines
            base_deadline += timedelta(hours=hours)
        
        return base_deadline
    
    def _refine_goals(self, goals: List[Goal]) -> List[Goal]:
        """Refine and filter generated goals."""
        # Remove duplicates
        unique_goals = []
        seen_titles = set()
        
        for goal in goals:
            if goal.title not in seen_titles:
                unique_goals.append(goal)
                seen_titles.add(goal.title)
        
        # Sort by priority
        unique_goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Limit to max concurrent
        max_goals = self.config.get('max_concurrent_goals', 10)
        return unique_goals[:max_goals]


class GoalRefinementEngine:
    """
    Refines and evolves goals based on progress, feedback,
    and changing circumstances.
    """
    
    def refine_goal(self, goal: Goal, 
                   completion_rate: float,
                   effort_expended: float) -> Goal:
        """Refine a goal based on progress and new information."""
        
        # Adjust difficulty based on progress
        if completion_rate < 0.3 and effort_expended > 0.5:
            # Goal too difficult, decompose further
            goal = self._decompose_goal(goal)
        elif completion_rate > 0.8 and effort_expended < 0.5:
            # Goal too easy, add stretch objectives
            goal = self._add_stretch_objectives(goal)
        
        # Update progress
        goal.progress = completion_rate
        
        return goal
    
    def _decompose_goal(self, goal: Goal) -> Goal:
        """Break down complex goal into manageable sub-goals."""
        sub_goals = []
        
        for i, criterion in enumerate(goal.success_criteria):
            sub_goal = Goal(
                title=f"{goal.title} (Part {i+1})",
                description=f"Sub-goal focusing on: {criterion}",
                type=goal.type,
                priority=goal.priority * 0.9,
                parent_goal=goal.id,
                deadline=goal.deadline,
                success_criteria=[criterion],
                estimated_effort=goal.estimated_effort / len(goal.success_criteria)
            )
            sub_goals.append(sub_goal)
        
        goal.sub_goals = sub_goals
        goal.is_decomposed = True
        
        return goal
    
    def _add_stretch_objectives(self, goal: Goal) -> Goal:
        """Add stretch objectives to a goal that's too easy."""
        stretch_criteria = [
            "Exceed baseline expectations",
            "Identify and implement improvements",
            "Document learnings for future use"
        ]
        
        goal.success_criteria.extend(stretch_criteria)
        goal.difficulty = min(1.0, goal.difficulty + 0.2)
        
        return goal


# Singleton instances
_goal_generator: Optional[GoalGenerator] = None
_goal_refinement: Optional[GoalRefinementEngine] = None
_interest_model: Optional[InterestModel] = None


def get_goal_generator(config: Optional[Dict] = None) -> GoalGenerator:
    """Get or create the global goal generator instance."""
    global _goal_generator
    if _goal_generator is None:
        _goal_generator = GoalGenerator(config)
    return _goal_generator


def get_goal_refinement_engine() -> GoalRefinementEngine:
    """Get or create the global goal refinement engine instance."""
    global _goal_refinement
    if _goal_refinement is None:
        _goal_refinement = GoalRefinementEngine()
    return _goal_refinement


def get_interest_model(decay_rate: float = 0.05) -> InterestModel:
    """Get or create the global interest model instance."""
    global _interest_model
    if _interest_model is None:
        _interest_model = InterestModel(decay_rate)
    return _interest_model


if __name__ == "__main__":
    # Example usage
    interest_model = InterestModel()
    
    # Simulate interactions
    interest_model.update_from_interaction(
        topic="machine_learning",
        domain="AI",
        engagement_level=0.8,
        learning_occurred=True,
        knowledge_gain=0.15,
        satisfaction_score=0.9
    )
    
    generator = GoalGenerator()
    
    motivation = {
        'autonomy': 0.6,
        'competence': 0.4,
        'relatedness': 0.7,
        'curiosity': 0.8
    }
    
    opportunities = [
        Opportunity(
            domain="data_analysis",
            description="New dataset available for analysis",
            potential_value=0.8,
            novelty=0.6
        )
    ]
    
    goals = generator.generate_goals(motivation, interest_model, opportunities)
    
    print(f"Generated {len(goals)} goals:")
    for goal in goals:
        print(f"  - {goal.title} (Priority: {goal.priority:.2f})")
