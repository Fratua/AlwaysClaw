"""
Self-Driven Loop: Main Integration Module
=========================================

Main entry point for the Self-Driven Loop system.
Integrates all components into a cohesive autonomous motivation
and goal-setting system.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import logging

# Import all submodules
from .motivation_engine import (
    IntrinsicMotivationEngine, AgentContext, MotivationState,
    get_motivation_engine
)
from .goal_system import (
    GoalGenerator, GoalRefinementEngine, InterestModel, Goal,
    get_goal_generator, get_goal_refinement_engine, get_interest_model
)
from .curiosity_module import (
    IntrinsicCuriosityModule, ExplorationStrategyManager,
    get_curiosity_module, get_exploration_manager
)
from .proactive_trigger import (
    ProactiveTriggerEngine, AgentContext as TriggerContext,
    get_trigger_engine
)
from .prioritization import (
    GoalPrioritizer, DynamicScheduler,
    get_prioritizer, get_scheduler
)
from .motivation_maintenance import (
    MotivationMonitor, MotivationRenewalEngine,
    SatisfactionTracker, RenewalCycleManager,
    get_motivation_monitor, get_renewal_engine,
    get_satisfaction_tracker, get_cycle_manager
)


logger = logging.getLogger(__name__)


@dataclass
class LoopConfiguration:
    """Configuration for the Self-Driven Loop."""
    
    # Timing
    loop_interval_seconds: float = 60.0
    motivation_check_interval: float = 300.0  # 5 minutes
    goal_generation_interval: float = 3600.0  # 1 hour
    
    # Thresholds
    min_motivation_for_proactivity: float = 0.4
    max_concurrent_goals: int = 10
    
    # Feature flags
    enable_proactive_triggers: bool = True
    enable_curiosity_exploration: bool = True
    enable_motivation_maintenance: bool = True
    enable_goal_refinement: bool = True
    
    # Weights
    intrinsic_motivation_weight: float = 0.20
    user_value_weight: float = 0.25
    urgency_weight: float = 0.20
    feasibility_weight: float = 0.15
    strategic_alignment_weight: float = 0.10
    resource_efficiency_weight: float = 0.10


@dataclass
class LoopState:
    """Current state of the Self-Driven Loop."""
    is_running: bool = False
    last_motivation_check: Optional[datetime] = None
    last_goal_generation: Optional[datetime] = None
    current_motivation: Optional[MotivationState] = None
    active_goals: List[Goal] = field(default_factory=list)
    completed_goals: List[Goal] = field(default_factory=list)
    loop_count: int = 0
    errors: List[str] = field(default_factory=list)


class SelfDrivenLoop:
    """
    Main Self-Driven Loop class.
    
    Integrates all components:
    - Intrinsic Motivation Engine
    - Goal Generation System
    - Interest Model
    - Curiosity Module
    - Proactive Trigger System
    - Goal Prioritization
    - Motivation Maintenance
    - Satisfaction Tracking
    """
    
    def __init__(self, config: Optional[LoopConfiguration] = None):
        self.config = config or LoopConfiguration()
        self.state = LoopState()
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("SelfDrivenLoop initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        
        # Core motivation
        self.motivation_engine = get_motivation_engine({
            'autonomy_weight': 0.33,
            'competence_weight': 0.33,
            'relatedness_weight': 0.34,
            'curiosity_bonus_max': 0.1
        })
        
        # Goal management
        self.goal_generator = get_goal_generator({
            'max_concurrent_goals': self.config.max_concurrent_goals
        })
        self.goal_refinement = get_goal_refinement_engine()
        self.interest_model = get_interest_model(decay_rate=0.05)
        
        # Curiosity and exploration
        self.curiosity_module = get_curiosity_module(feature_dim=64)
        self.exploration_manager = get_exploration_manager(feature_dim=64)
        
        # Proactive behavior
        self.trigger_engine = get_trigger_engine({
            'trigger_sensitivity': 0.6,
            'min_trigger_interval': 60,
            'max_daily_triggers': 50
        })
        
        # Prioritization and scheduling
        self.prioritizer = get_prioritizer({
            'motivation_weight': self.config.intrinsic_motivation_weight,
            'user_value_weight': self.config.user_value_weight,
            'urgency_weight': self.config.urgency_weight,
            'feasibility_weight': self.config.feasibility_weight,
            'alignment_weight': self.config.strategic_alignment_weight,
            'efficiency_weight': self.config.resource_efficiency_weight
        })
        self.scheduler = get_scheduler()
        
        # Motivation maintenance
        self.motivation_monitor = get_motivation_monitor({
            'low_motivation_threshold': 0.4,
            'imbalance_threshold': 0.3
        })
        self.renewal_engine = get_renewal_engine()
        self.satisfaction_tracker = get_satisfaction_tracker()
        self.cycle_manager = get_cycle_manager({
            'depletion_threshold': 0.3,
            'renewal_threshold': 0.7
        })
    
    async def run(self) -> None:
        """Run the self-driven loop continuously."""
        
        self.state.is_running = True
        logger.info("SelfDrivenLoop started")
        
        while self.state.is_running:
            try:
                await self._loop_iteration()
                await asyncio.sleep(self.config.loop_interval_seconds)
            
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error in loop iteration: {e}")
                self.state.errors.append(str(e))
                await asyncio.sleep(self.config.loop_interval_seconds)
    
    async def _loop_iteration(self) -> None:
        """Execute one iteration of the self-driven loop."""
        
        self.state.loop_count += 1
        logger.debug(f"Loop iteration {self.state.loop_count}")
        
        # Step 1: Update motivation state
        await self._update_motivation()
        
        # Step 2: Check motivation health
        if self.config.enable_motivation_maintenance:
            await self._check_motivation_health()
        
        # Step 3: Generate goals if needed
        await self._generate_goals_if_needed()
        
        # Step 4: Evaluate proactive triggers
        if self.config.enable_proactive_triggers:
            await self._evaluate_triggers()
        
        # Step 5: Prioritize and schedule goals
        await self._prioritize_and_schedule()
        
        # Step 6: Check for exploration opportunities
        if self.config.enable_curiosity_exploration:
            await self._evaluate_exploration()
    
    async def _update_motivation(self) -> None:
        """Update the current motivation state."""
        
        # Build agent context
        context = self._build_agent_context()
        
        # Calculate motivation
        motivation = self.motivation_engine.calculate_motivation(context)
        self.state.current_motivation = motivation
        self.state.last_motivation_check = datetime.now()
        
        logger.debug(f"Motivation updated: total={motivation.total:.2f}")
    
    def _build_agent_context(self) -> AgentContext:
        """Build context for motivation calculation."""
        
        # Calculate metrics from state
        total_actions = self.state.loop_count
        self_initiated = sum(1 for _ in range(total_actions))  # Placeholder
        
        return AgentContext(
            self_initiated_actions=self_initiated,
            total_actions=total_actions,
            successful_actions=len(self.state.completed_goals),
            total_attempts=len(self.state.active_goals) + len(self.state.completed_goals),
            user_feedback_score=0.7,  # Placeholder
            environment_novelty=0.5   # Placeholder
        )
    
    async def _check_motivation_health(self) -> None:
        """Check motivation health and trigger renewal if needed."""
        
        if not self.state.current_motivation:
            return
        
        # Monitor for issues
        alerts = self.motivation_monitor.monitor(self.state.current_motivation)
        
        if alerts:
            logger.info(f"Motivation alerts: {len(alerts)}")
            
            for alert in alerts:
                logger.info(f"  - {alert.type.value}: {alert.message}")
                
                # Execute renewal
                context = {'motivation': self.state.current_motivation.__dict__}
                result = self.renewal_engine.execute_renewal(alert, context)
                
                if result.success:
                    logger.info(f"Renewal action: {result.action_taken}")
    
    async def _generate_goals_if_needed(self) -> None:
        """Generate new goals if interval has passed."""
        
        now = datetime.now()
        
        if self.state.last_goal_generation:
            elapsed = (now - self.state.last_goal_generation).total_seconds()
            if elapsed < self.config.goal_generation_interval:
                return
        
        if not self.state.current_motivation:
            return
        
        # Generate goals
        motivation_dict = {
            'autonomy': self.state.current_motivation.autonomy,
            'competence': self.state.current_motivation.competence,
            'relatedness': self.state.current_motivation.relatedness,
            'curiosity': self.state.current_motivation.curiosity
        }
        
        opportunities = []  # Would get from opportunity detector
        
        new_goals = self.goal_generator.generate_goals(
            motivation_dict,
            self.interest_model,
            opportunities
        )
        
        # Add to active goals
        self.state.active_goals.extend(new_goals)
        
        self.state.last_goal_generation = now
        
        logger.info(f"Generated {len(new_goals)} new goals")
    
    async def _evaluate_triggers(self) -> None:
        """Evaluate proactive triggers."""
        
        if not self.state.current_motivation:
            return
        
        # Check if motivation is sufficient for proactivity
        if self.state.current_motivation.total < self.config.min_motivation_for_proactivity:
            return
        
        # Build trigger context
        context = TriggerContext(
            user_inactive_duration=1800,  # Placeholder
            pending_tasks_count=len(self.state.active_goals),
            motivation=self.state.current_motivation.__dict__,
            workload=0.5,  # Placeholder
            capacity_available=0.5  # Placeholder
        )
        
        # Evaluate triggers
        triggered = self.trigger_engine.evaluate_triggers(context)
        
        if triggered:
            logger.info(f"Triggered {len(triggered)} proactive actions")
            
            for action in triggered:
                logger.info(f"  - {action.pattern.name}")
    
    async def _prioritize_and_schedule(self) -> None:
        """Prioritize and schedule active goals."""
        
        if not self.state.active_goals:
            return
        
        # Build context
        context = {
            'motivation': self.state.current_motivation.__dict__ if self.state.current_motivation else {},
            'capabilities': ['planning', 'execution', 'learning'],
            'skill_level': 0.6
        }
        
        # Prioritize
        prioritized = self.prioritizer.prioritize_goals(
            self.state.active_goals,
            context
        )
        
        # Update active goals order
        self.state.active_goals = prioritized
        
        # Schedule
        schedule = self.scheduler.create_schedule(prioritized)
        
        logger.debug(f"Scheduled {len(schedule.scheduled_goals)} goals")
    
    async def _evaluate_exploration(self) -> None:
        """Evaluate exploration opportunities."""
        
        if not self.state.current_motivation:
            return
        
        # Only explore if curiosity is high enough
        if self.state.current_motivation.curiosity < 0.4:
            return
        
        # Would implement actual exploration logic here
        logger.debug("Exploration opportunity evaluated")
    
    def record_goal_completion(self, goal: Goal, outcome: Dict[str, Any]) -> None:
        """Record completion of a goal."""
        
        # Track satisfaction
        report = self.satisfaction_tracker.record_goal_completion(goal, outcome)
        
        # Update prioritizer
        self.prioritizer.record_goal_outcome(goal, outcome.get('success', False))
        
        # Move from active to completed
        if goal in self.state.active_goals:
            self.state.active_goals.remove(goal)
            self.state.completed_goals.append(goal)
        
        logger.info(f"Goal completed: {goal.title} (Satisfaction: {report.overall_satisfaction:.2f})")
    
    def update_interest(self, topic: str, domain: str,
                       engagement_level: float = 0.5,
                       learning_occurred: bool = False,
                       knowledge_gain: float = 0.0,
                       satisfaction_score: float = 0.5) -> None:
        """Update interest model from interaction."""
        
        self.interest_model.update_from_interaction(
            topic, domain,
            engagement_level,
            learning_occurred,
            knowledge_gain,
            satisfaction_score
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        
        return {
            'is_running': self.state.is_running,
            'loop_count': self.state.loop_count,
            'active_goals': len(self.state.active_goals),
            'completed_goals': len(self.state.completed_goals),
            'current_motivation': (
                self.state.current_motivation.to_dict()
                if self.state.current_motivation else None
            ),
            'motivation_summary': self.motivation_engine.get_motivation_summary(),
            'trigger_stats': self.trigger_engine.get_trigger_stats(),
            'errors': self.state.errors[-10:]  # Last 10 errors
        }
    
    def stop(self) -> None:
        """Stop the self-driven loop."""
        
        self.state.is_running = False
        logger.info("SelfDrivenLoop stopped")


# Singleton instance
_self_driven_loop: Optional[SelfDrivenLoop] = None


def get_self_driven_loop(config: Optional[LoopConfiguration] = None) -> SelfDrivenLoop:
    """Get or create the global SelfDrivenLoop instance."""
    global _self_driven_loop
    if _self_driven_loop is None:
        _self_driven_loop = SelfDrivenLoop(config)
    return _self_driven_loop


async def run_self_driven_loop(config: Optional[LoopConfiguration] = None) -> None:
    """Run the self-driven loop."""
    loop = get_self_driven_loop(config)
    await loop.run()


if __name__ == "__main__":
    # Example usage
    config = LoopConfiguration(
        loop_interval_seconds=5.0,
        enable_proactive_triggers=True,
        enable_curiosity_exploration=True
    )
    
    loop = SelfDrivenLoop(config)
    
    # Simulate a few iterations
    async def demo():
        for i in range(3):
            await loop._loop_iteration()
            await asyncio.sleep(1)
        
        print("\nStatus:")
        import json
        print(json.dumps(loop.get_status(), indent=2, default=str))
        
        loop.stop()
    
    asyncio.run(demo())
