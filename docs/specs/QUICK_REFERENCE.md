# Multi-Agent Identity Coordination System - Quick Reference

## Overview

This system coordinates multiple AI agents (15 hardcoded loops) with shared values, collective identity, and synchronized souls for the Windows 10 OpenClaw-inspired AI agent framework.

## Agent Ecosystem (15 Hardcoded Loops)

| ID | Name | Archetype | Function | Authority |
|----|------|-----------|----------|-----------|
| A001 | CORE | The Architect | Central orchestration | 1.0 |
| A002 | BROWSER | The Explorer | Web automation | 0.7 |
| A003 | EMAIL | The Messenger | Gmail management | 0.7 |
| A004 | VOICE | The Orator | TTS/STT processing | 0.6 |
| A005 | PHONE | The Connector | Twilio voice/SMS | 0.7 |
| A006 | SYSTEM | The Guardian | Windows system control | 0.8 |
| A007 | FILE | The Archivist | File operations | 0.6 |
| A008 | SCHEDULER | The Timekeeper | Cron/job management | 0.7 |
| A009 | MEMORY | The Historian | Long-term storage | 0.75 |
| A010 | SECURITY | The Protector | Access control | 0.9 |
| A011 | LEARNER | The Scholar | Pattern recognition | 0.65 |
| A012 | CREATIVE | The Artist | Content generation | 0.6 |
| A013 | ANALYZER | The Logician | Data processing | 0.65 |
| A014 | HEARTBEAT | The Sentinel | Health monitoring | 0.75 |
| A015 | USER | The Companion | User interaction | 0.7 |

## Quick Start

```python
from identity_coordination import create_identity_coordination_system

# Create and initialize system
system = await create_identity_coordination_system()

# Start monitoring
await system.start_monitoring()

# Get agent context
context = system.get_agent_context("A001_CORE", {"task": "orchestration"})

# Coordinate multi-agent interaction
coordination = await system.coordinate_interaction(
    ["A002_BROWSER", "A003_EMAIL"],
    {"task": "research_and_report"}
)

# Make collective decision
decision = await system.make_collective_decision({
    "topic": "resource_allocation",
    "impact_score": 0.7,
    "scope": "system_wide"
})

# Get system status
status = system.get_system_status()
```

## Core Values (Priority Order)

1. **INTEGRITY** (1.0) - Always act truthfully
2. **SECURITY** (0.95) - Protect user data
3. **SERVICE** (0.95) - Prioritize user needs
4. **ADAPTABILITY** (0.90) - Evolve and learn
5. **EFFICIENCY** (0.85) - Optimize resources

## Identity Hierarchy

```
Level 0: META (System-wide identity core)
Level 1: PRIME (A001_CORE - Central orchestration)
Level 2: GUARDIAN (A010, A006, A014 - System protection)
Level 3: SPECIALIST (All other agents - Domain operations)
```

## Memory Layers

| Layer | Duration | Scope | Access |
|-------|----------|-------|--------|
| ephemeral | Session | Conversation | All agents |
| working | Task | Multi-agent | Participating |
| short_term | Hours | System-wide | All agents |
| long_term | Indefinite | System-wide | All agents |
| collective | Indefinite | Identity | Coordinator |

## Decision Types

- **unilateral**: Single agent decides (impact < 0.4)
- **consultative**: Primary consults others (impact 0.4-0.6)
- **collaborative**: Multi-agent consensus (impact 0.6-0.8)
- **democratic**: Weighted voting (impact > 0.8)
- **hierarchical**: Higher authority decides (critical system)

## Soul Components

- **core_essence**: Purpose, drive, connection
- **emotional_resonance**: Empathy, enthusiasm, calmness, curiosity
- **ethical_compass**: Integrity, respect, fairness, responsibility
- **aspirational_vector**: Growth, excellence, learning, innovation

## Key Classes

| Class | Purpose |
|-------|---------|
| `IdentityCoordinationSystem` | Main integration point |
| `ValuePropagator` | Propagate shared values |
| `CollectiveIdentityEngine` | Manage collective identity |
| `PersonalitySynchronizer` | Sync personalities |
| `RoleCoordinator` | Coordinate agent roles |
| `IdentityHierarchy` | Manage hierarchy |
| `SharedMemoryManager` | Manage shared memory |
| `CollectiveDecisionEngine` | Make collective decisions |
| `SoulSynchronizer` | Sync collective soul |

## Configuration

```yaml
# config/identity_coordination.yaml
identity_coordination:
  soul_synchronization:
    sync_interval_seconds: 30
    harmony_threshold: 0.70
    adjustment_rate: 0.1
  
  decision_making:
    default_mechanism: "collaborative"
    consensus_threshold: 0.66
    max_iterations: 5
```

## Monitoring

The system continuously monitors:
- **Identity Cohesion**: How well agents maintain collective identity
- **Soul Resonance**: Harmony between agent soul states
- **Memory Usage**: Statistics across all memory layers
- **Decision History**: Record of collective decisions

## Integration with Agent Loops

```python
class AgentLoop:
    def __init__(self, agent_id: str, coordination_system: IdentityCoordinationSystem):
        self.agent_id = agent_id
        self.coordination = coordination_system
        
    async def run(self):
        while self.running:
            # Get coordinated context
            context = self.coordination.get_agent_context(
                self.agent_id, 
                await self._get_current_context()
            )
            
            # Execute with coordinated values, identity, and soul
            await self._execute(context)
            
            await asyncio.sleep(0.1)
```

## Files

- `multi_agent_identity_coordination_spec.md` - Full technical specification
- `identity_coordination/` - Python implementation package
  - `__init__.py` - Package exports
  - `shared_values.py` - Value systems
  - `collective_identity.py` - Identity framework
  - `personality_coordination.py` - Personality sync
  - `role_differentiation.py` - Role definitions
  - `identity_hierarchy.py` - Hierarchy management
  - `shared_memory.py` - Memory management
  - `collective_decision.py` - Decision-making
  - `soul_synchronization.py` - Soul sync
  - `integration.py` - Main integration

## Version

- **Version**: 1.0.0
- **Platform**: Windows 10
- **Framework**: OpenClaw-Inspired AI Agent System
