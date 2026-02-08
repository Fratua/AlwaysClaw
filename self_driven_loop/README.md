# Self-Driven Loop: Autonomous Motivation and Goal-Setting System

A comprehensive Python implementation of an autonomous motivation and goal-setting system for AI agents, inspired by Self-Determination Theory (SDT) and Intrinsic Curiosity Module (ICM) research.

## Overview

The Self-Driven Loop enables AI agents to:
- Generate intrinsic motivation autonomously
- Create and pursue self-directed goals
- Explore driven by curiosity
- Act proactively without external prompts
- Maintain sustainable motivation over time

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-DRIVEN LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  MOTIVATION  │  │    GOAL      │  │  INTEREST    │          │
│  │   ENGINE     │◀─│   GENERATOR  │◀─│    MODEL     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                      │
│         ▼                 ▼                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   CURIOSITY  │  │   PROACTIVE  │  │ PRIORITIZER  │          │
│  │    MODULE    │─▶│   TRIGGER    │─▶│              │          │
│  └──────────────┘  └──────────────┘  └──────┬───────┘          │
│                                              │                   │
│         ┌────────────────────────────────────┘                   │
│         ▼                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  MOTIVATION  │  │ SATISFACTION │  │   RENEWAL    │          │
│  │  MAINTAINER  │◀─│   TRACKER    │◀─│    CYCLE     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Intrinsic Motivation Engine (`motivation_engine.py`)

Implements Self-Determination Theory's three core drives:
- **Autonomy Drive**: Self-direction and choice
- **Competence Drive**: Mastery and effectiveness
- **Relatedness Drive**: Connection and contribution

Plus curiosity as an additional drive.

```python
from self_driven_loop import IntrinsicMotivationEngine, AgentContext

engine = IntrinsicMotivationEngine()
context = AgentContext(
    self_initiated_actions=70,
    total_actions=100,
    successful_actions=85,
    user_feedback_score=0.8
)
motivation = engine.calculate_motivation(context)
```

### 2. Goal System (`goal_system.py`)

Autonomous goal generation and refinement:
- Goal generation based on motivation state
- Interest modeling and tracking
- Goal decomposition and refinement

```python
from self_driven_loop import GoalGenerator, InterestModel

generator = GoalGenerator()
interest_model = InterestModel()

# Update interests
interest_model.update_from_interaction(
    topic="machine_learning",
    domain="AI",
    engagement_level=0.8,
    learning_occurred=True,
    knowledge_gain=0.15
)

# Generate goals
motivation = {'autonomy': 0.6, 'competence': 0.4, 'curiosity': 0.8}
goals = generator.generate_goals(motivation, interest_model, [])
```

### 3. Curiosity Module (`curiosity_module.py`)

Curiosity-driven exploration using prediction error:
- Intrinsic Curiosity Module (ICM) implementation
- Forward/inverse dynamics models
- Multiple exploration strategies

```python
from self_driven_loop import IntrinsicCuriosityModule

icm = IntrinsicCuriosityModule()

# Calculate intrinsic reward
reward = icm.calculate_intrinsic_reward(state, action, next_state)
```

### 4. Proactive Trigger System (`proactive_trigger.py`)

Detects conditions for proactive behavior:
- User inactivity detection
- Opportunity detection
- Learning opportunity triggers
- System optimization triggers

```python
from self_driven_loop import ProactiveTriggerEngine, AgentContext

engine = ProactiveTriggerEngine()
context = AgentContext(
    user_inactive_duration=2400,
    pending_tasks_count=3,
    motivation={'curiosity': 0.8}
)

triggered = engine.evaluate_triggers(context)
```

### 5. Prioritization System (`prioritization.py`)

Goal prioritization and scheduling:
- Multi-factor priority calculation
- Dynamic weight adjustment
- Intelligent scheduling

```python
from self_driven_loop import GoalPrioritizer, DynamicScheduler

prioritizer = GoalPrioritizer()
scheduler = DynamicScheduler()

prioritized = prioritizer.prioritize_goals(goals, context)
schedule = scheduler.create_schedule(prioritized)
```

### 6. Motivation Maintenance (`motivation_maintenance.py`)

Sustainable motivation management:
- Motivation monitoring
- Renewal protocols
- Satisfaction tracking
- Renewal cycle management

```python
from self_driven_loop import MotivationMonitor, MotivationRenewalEngine

monitor = MotivationMonitor()
renewal = MotivationRenewalEngine()

alerts = monitor.monitor(motivation_state)
for alert in alerts:
    result = renewal.execute_renewal(alert, context)
```

## Quick Start

### Installation

```bash
# Clone or copy the self_driven_loop package
pip install numpy  # Required dependency
```

### Basic Usage

```python
import asyncio
from self_driven_loop import SelfDrivenLoop, LoopConfiguration

async def main():
    # Configure the loop
    config = LoopConfiguration(
        loop_interval_seconds=60.0,
        enable_proactive_triggers=True,
        enable_curiosity_exploration=True,
        enable_motivation_maintenance=True
    )
    
    # Create and run the loop
    loop = SelfDrivenLoop(config)
    
    # Run for a specific duration
    try:
        await asyncio.wait_for(loop.run(), timeout=300.0)
    except asyncio.TimeoutError:
        loop.stop()
    
    # Get status
    status = loop.get_status()
    print(f"Completed {status['loop_count']} iterations")
    print(f"Active goals: {status['active_goals']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with Agent System

```python
from self_driven_loop import get_self_driven_loop

# Get singleton instance
loop = get_self_driven_loop()

# Update from agent interactions
loop.update_interest(
    topic="user_preference",
    domain="personalization",
    engagement_level=0.9,
    satisfaction_score=0.8
)

# Record goal completion
loop.record_goal_completion(goal, {
    'success': True,
    'user_satisfaction': 0.9,
    'autonomy_level': 0.8
})

# Get current status
status = loop.get_status()
```

## Configuration

### Loop Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loop_interval_seconds` | 60.0 | Main loop interval |
| `motivation_check_interval` | 300.0 | Motivation check frequency |
| `goal_generation_interval` | 3600.0 | Goal generation frequency |
| `min_motivation_for_proactivity` | 0.4 | Minimum motivation for proactive actions |
| `max_concurrent_goals` | 10 | Maximum active goals |
| `enable_proactive_triggers` | True | Enable proactive behavior |
| `enable_curiosity_exploration` | True | Enable curiosity-driven exploration |
| `enable_motivation_maintenance` | True | Enable motivation maintenance |

### Motivation Weights

| Drive | Default | Description |
|-------|---------|-------------|
| `autonomy_weight` | 0.33 | Weight for autonomy drive |
| `competence_weight` | 0.33 | Weight for competence drive |
| `relatedness_weight` | 0.34 | Weight for relatedness drive |
| `curiosity_bonus_max` | 0.1 | Maximum curiosity bonus |

## API Reference

### Main Classes

- `SelfDrivenLoop`: Main integration class
- `IntrinsicMotivationEngine`: Motivation calculation
- `GoalGenerator`: Goal generation
- `InterestModel`: Interest tracking
- `IntrinsicCuriosityModule`: Curiosity-driven exploration
- `ProactiveTriggerEngine`: Proactive behavior triggers
- `GoalPrioritizer`: Goal prioritization
- `DynamicScheduler`: Goal scheduling
- `MotivationMonitor`: Motivation monitoring
- `MotivationRenewalEngine`: Motivation renewal

### Data Classes

- `MotivationState`: Current motivation state
- `AgentContext`: Context for motivation calculation
- `Goal`: Goal representation
- `InterestProfile`: Interest profile for a topic
- `TriggerPattern`: Pattern for proactive triggers
- `Schedule`: Goal schedule

## Research Foundation

This implementation is based on:

1. **Self-Determination Theory** (Deci & Ryan)
   - Autonomy, Competence, Relatedness as core drives

2. **Intrinsic Curiosity Module** (Pathak et al.)
   - Prediction error as curiosity signal
   - Forward/inverse dynamics models

3. **Goal-Oriented Agent Architectures**
   - Hierarchical goal structures
   - Dynamic goal generation

4. **Proactive AI Systems**
   - Continuous monitoring
   - Context-aware triggering

## Success Metrics

### Motivation Health
- Average motivation level > 0.7
- Drive balance variance < 0.15
- Motivation trend stability > -0.05

### Goal Achievement
- Goal completion rate > 0.8
- Self-generated goal success > 0.7
- Goal satisfaction score > 0.75

### Proactivity
- Proactive action ratio > 0.3
- Trigger accuracy > 0.75
- User acceptance rate > 0.7

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow the existing code style and add tests for new features.

## Changelog

### v1.0.0
- Initial release
- Full SDT-based motivation engine
- ICM-based curiosity module
- Proactive trigger system
- Goal prioritization and scheduling
- Motivation maintenance and renewal
