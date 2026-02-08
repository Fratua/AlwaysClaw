"""
Self-Driven Loop: Autonomous Motivation and Goal-Setting System
===============================================================

A comprehensive system for enabling intrinsic motivation generation,
autonomous goal setting, and self-directed behavior in AI agents.

Components:
-----------
- motivation_engine: Intrinsic motivation calculation (SDT-based)
- goal_system: Goal generation, refinement, and interest modeling
- curiosity_module: Curiosity-driven exploration (ICM-based)
- proactive_trigger: Proactive behavior triggering
- prioritization: Goal prioritization and scheduling
- motivation_maintenance: Motivation monitoring and renewal
- self_driven_loop: Main integration module

Usage:
------
    from self_driven_loop import SelfDrivenLoop, LoopConfiguration
    
    config = LoopConfiguration(
        loop_interval_seconds=60.0,
        enable_proactive_triggers=True
    )
    
    loop = SelfDrivenLoop(config)
    await loop.run()

Author: AI Systems Architecture Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Systems Architecture Team"

# Main exports
from .self_driven_loop import (
    SelfDrivenLoop,
    LoopConfiguration,
    LoopState,
    get_self_driven_loop,
    run_self_driven_loop
)

# Motivation engine exports
from .motivation_engine import (
    IntrinsicMotivationEngine,
    MotivationState,
    AgentContext,
    AutonomyDrive,
    CompetenceDrive,
    RelatednessDrive,
    get_motivation_engine
)

# Goal system exports
from .goal_system import (
    GoalGenerator,
    GoalRefinementEngine,
    InterestModel,
    Goal,
    GoalType,
    GoalStatus,
    KnowledgeGap,
    Opportunity,
    InterestProfile,
    get_goal_generator,
    get_goal_refinement_engine,
    get_interest_model
)

# Curiosity module exports
from .curiosity_module import (
    IntrinsicCuriosityModule,
    ExplorationStrategyManager,
    FeatureEncoder,
    ForwardDynamicsModel,
    InverseDynamicsModel,
    State,
    Action,
    ExplorationContext,
    get_curiosity_module,
    get_exploration_manager
)

# Proactive trigger exports
from .proactive_trigger import (
    ProactiveTriggerEngine,
    TriggerPattern,
    TriggeredAction,
    TriggerType,
    Condition,
    MotivationAlert,
    AlertType,
    Severity,
    get_trigger_engine
)

# Prioritization exports
from .prioritization import (
    GoalPrioritizer,
    DynamicScheduler,
    Schedule,
    ScheduledGoal,
    TimeSlot,
    PriorityLevel,
    get_prioritizer,
    get_scheduler
)

# Motivation maintenance exports
from .motivation_maintenance import (
    MotivationMonitor,
    MotivationRenewalEngine,
    SatisfactionTracker,
    RenewalCycleManager,
    MotivationAlert,
    AlertType,
    Severity,
    CycleState,
    CycleAction,
    CycleDecision,
    RenewalResult,
    SatisfactionReport,
    get_motivation_monitor,
    get_renewal_engine,
    get_satisfaction_tracker,
    get_cycle_manager
)

__all__ = [
    # Main
    'SelfDrivenLoop',
    'LoopConfiguration',
    'LoopState',
    'get_self_driven_loop',
    'run_self_driven_loop',
    
    # Motivation
    'IntrinsicMotivationEngine',
    'MotivationState',
    'AgentContext',
    'AutonomyDrive',
    'CompetenceDrive',
    'RelatednessDrive',
    'get_motivation_engine',
    
    # Goals
    'GoalGenerator',
    'GoalRefinementEngine',
    'InterestModel',
    'Goal',
    'GoalType',
    'GoalStatus',
    'KnowledgeGap',
    'Opportunity',
    'InterestProfile',
    'get_goal_generator',
    'get_goal_refinement_engine',
    'get_interest_model',
    
    # Curiosity
    'IntrinsicCuriosityModule',
    'ExplorationStrategyManager',
    'FeatureEncoder',
    'ForwardDynamicsModel',
    'InverseDynamicsModel',
    'State',
    'Action',
    'ExplorationContext',
    'get_curiosity_module',
    'get_exploration_manager',
    
    # Triggers
    'ProactiveTriggerEngine',
    'TriggerPattern',
    'TriggeredAction',
    'TriggerType',
    'Condition',
    'get_trigger_engine',
    
    # Prioritization
    'GoalPrioritizer',
    'DynamicScheduler',
    'Schedule',
    'ScheduledGoal',
    'TimeSlot',
    'PriorityLevel',
    'get_prioritizer',
    'get_scheduler',
    
    # Maintenance
    'MotivationMonitor',
    'MotivationRenewalEngine',
    'SatisfactionTracker',
    'RenewalCycleManager',
    'CycleState',
    'CycleAction',
    'CycleDecision',
    'RenewalResult',
    'SatisfactionReport',
    'get_motivation_monitor',
    'get_renewal_engine',
    'get_satisfaction_tracker',
    'get_cycle_manager'
]
