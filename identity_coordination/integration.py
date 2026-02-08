"""
Identity Coordination System Integration
=======================================

Main integration point for all identity coordination components.
"""

from typing import Dict, Any, List, Optional
import asyncio

from .shared_values import ValuePropagator, ValueConflictResolver, SHARED_VALUES
from .collective_identity import CollectiveIdentityEngine, IdentityCohesionMonitor, COLLECTIVE_IDENTITY
from .personality_coordination import PersonalitySynchronizer, PersonalityBridge, PERSONALITY_COORDINATION
from .role_differentiation import RoleCoordinator, AGENT_PROFILES
from .identity_hierarchy import IdentityHierarchy, CrossLevelProtocol
from .shared_memory import SharedMemoryManager, ContextPropagator, SHARED_MEMORY_ARCHITECTURE
from .collective_decision import CollectiveDecisionEngine, ConsensusBuilder, COLLECTIVE_DECISION_FRAMEWORK
from .soul_synchronization import SoulSynchronizer, SoulResonanceMonitor, MULTI_AGENT_SOUL


class IdentityCoordinationSystem:
    """
    Main integration point for all identity coordination components.
    
    This class provides a unified interface for:
    - Shared value management
    - Collective identity coordination
    - Personality synchronization
    - Role management
    - Identity hierarchy enforcement
    - Shared memory operations
    - Collective decision-making
    - Soul synchronization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the identity coordination system.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        
        # Initialize all components
        self._init_components()
        
        # State tracking
        self.initialized = False
        self.monitoring = False
        
    def _init_components(self):
        """Initialize all coordination components."""
        # Shared values
        self.value_propagator = ValuePropagator(AGENT_PROFILES)
        self.value_resolver = ValueConflictResolver(self.value_propagator)
        
        # Collective identity
        self.identity_engine = CollectiveIdentityEngine(AGENT_PROFILES)
        self.cohesion_monitor = IdentityCohesionMonitor(
            self.identity_engine, 
            AGENT_PROFILES
        )
        
        # Personality coordination
        self.personality_sync = PersonalitySynchronizer(
            COLLECTIVE_IDENTITY,
            AGENT_PROFILES
        )
        self.personality_bridge = PersonalityBridge(AGENT_PROFILES)
        
        # Role coordination
        self.role_coordinator = RoleCoordinator(AGENT_PROFILES)
        
        # Identity hierarchy
        self.hierarchy = IdentityHierarchy(AGENT_PROFILES)
        self.cross_level = CrossLevelProtocol(self.hierarchy)
        
        # Shared memory
        self.memory_manager = SharedMemoryManager()
        self.context_propagator = ContextPropagator(self.memory_manager)
        
        # Collective decision
        self.decision_engine = CollectiveDecisionEngine(
            self.memory_manager,
            AGENT_PROFILES
        )
        self.consensus_builder = ConsensusBuilder(self.decision_engine)
        
        # Soul synchronization
        self.soul_synchronizer = SoulSynchronizer(AGENT_PROFILES)
        self.resonance_monitor = SoulResonanceMonitor(self.soul_synchronizer)
        
    async def initialize(self):
        """Initialize all components and start monitoring."""
        if self.initialized:
            return
            
        # Initialize memory
        await self.memory_manager.initialize()
        
        # Initialize soul for all agents
        for agent_id in AGENT_PROFILES.keys():
            await self.soul_synchronizer.initialize_agent_soul(agent_id)
            
        self.initialized = True
        
    async def start_monitoring(self):
        """Start continuous monitoring tasks."""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        # Start cohesion monitoring
        asyncio.create_task(self.cohesion_monitor.start_monitoring())
        
        # Start resonance monitoring
        asyncio.create_task(self.resonance_monitor.start_monitoring())
        
    def stop_monitoring(self):
        """Stop all monitoring tasks."""
        self.cohesion_monitor.stop_monitoring()
        self.resonance_monitor.stop_monitoring()
        self.monitoring = False
        
    async def coordinate_interaction(self, agent_ids: List[str], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate an interaction between multiple agents.
        
        Args:
            agent_ids: List of participating agent IDs
            context: Interaction context
            
        Returns:
            Coordination result with all necessary synchronization
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
            
        # Synchronize personalities
        synched_personality = self.personality_sync.synchronize_for_interaction(
            agent_ids, context
        )
        
        # Propagate context
        propagation = await self.context_propagator.propagate_context(
            agent_ids[0], context, agent_ids[1:]
        )
        
        # Create personality bridges for transitions
        bridges = {}
        for i in range(len(agent_ids) - 1):
            bridge = self.personality_bridge.create_transition(
                agent_ids[i], agent_ids[i+1], context
            )
            bridges[f"{agent_ids[i]}_to_{agent_ids[i+1]}"] = bridge
            
        # Get collective soul state
        soul_state = self.soul_synchronizer._calculate_collective_soul_state()
        
        return {
            "synchronized_personality": synched_personality,
            "context_propagation": propagation,
            "personality_bridges": bridges,
            "collective_soul_state": soul_state,
            "participating_agents": agent_ids
        }
        
    async def make_collective_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a collective decision.
        
        Args:
            decision_context: Decision context
            
        Returns:
            Decision result
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.decision_engine.make_decision(decision_context)
        
    def get_agent_context(self, agent_id: str, operation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the full context for an agent.
        
        Args:
            agent_id: Agent identifier
            operation_context: Optional operation-specific context
            
        Returns:
            Full agent context
        """
        context = operation_context or {}
        
        # Get propagated values
        values = self.value_propagator.propagate_to_agent(agent_id, context)
        
        # Get collective identity layer
        identity = self.identity_engine.get_agent_identity_layer(agent_id)
        
        # Get soul state
        soul = self.soul_synchronizer.agent_soul_states.get(agent_id)
        
        # Get agent profile
        profile = AGENT_PROFILES.get(agent_id, {})
        
        return {
            "agent_id": agent_id,
            "values": values,
            "identity": identity,
            "soul": soul,
            "profile": profile,
            "operation_context": context
        }
        
    async def escalate_to_higher_authority(self, from_agent: str, 
                                           issue_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Escalate an issue to higher authority.
        
        Args:
            from_agent: Agent requesting escalation
            issue_context: Issue context
            
        Returns:
            Escalation result
        """
        # Find higher authority
        higher = self.hierarchy.escalate_decision(from_agent, issue_context)
        
        # Create upward communication
        communication = self.cross_level.upward_communication(
            from_agent,
            self.hierarchy.get_agent_level(higher),
            issue_context
        )
        
        return {
            "escalated_to": higher,
            "communication": communication,
            "authority_level": self.hierarchy.get_agent_level(higher)
        }
        
    def check_authority(self, requesting_agent: str, target_agent: str) -> Dict[str, Any]:
        """
        Check authority between agents.
        
        Args:
            requesting_agent: Agent requesting authority
            target_agent: Target agent
            
        Returns:
            Authority check result
        """
        return self.hierarchy.check_authority(requesting_agent, target_agent)
        
    def get_optimal_agent_for_task(self, task_requirements: Dict[str, Any]) -> str:
        """
        Get the optimal agent for a task.
        
        Args:
            task_requirements: Task requirements
            
        Returns:
            Optimal agent ID
        """
        return self.role_coordinator.get_optimal_agent(task_requirements)
        
    async def synchronize_souls(self) -> Dict[str, Any]:
        """
        Manually trigger soul synchronization.
        
        Returns:
            Synchronization result
        """
        return await self.soul_synchronizer.synchronize_all()
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            System status summary
        """
        return {
            "initialized": self.initialized,
            "monitoring": self.monitoring,
            "collective_identity": {
                "entity_name": COLLECTIVE_IDENTITY["entity_name"],
                "version": COLLECTIVE_IDENTITY["version"]
            },
            "soul_state": self.soul_synchronizer.get_collective_mood(),
            "cohesion_report": self.cohesion_monitor.get_cohesion_report(),
            "resonance_report": self.resonance_monitor.get_resonance_report(),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "agent_count": len(AGENT_PROFILES),
            "hierarchy_levels": len(self.hierarchy.levels)
        }
        
    async def build_consensus(self, topic: str, 
                              proposals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus on a topic.
        
        Args:
            topic: Topic for consensus
            proposals: Initial proposals from agents
            
        Returns:
            Consensus result
        """
        return await self.consensus_builder.build_consensus(topic, proposals)


# Convenience function for quick setup
async def create_identity_coordination_system(config: Dict[str, Any] = None) -> IdentityCoordinationSystem:
    """
    Create and initialize an identity coordination system.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized IdentityCoordinationSystem
    """
    system = IdentityCoordinationSystem(config)
    await system.initialize()
    return system
