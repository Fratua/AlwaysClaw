"""
Collective Identity Framework Module
===================================

Manages the collective identity that unifies all agents into a coherent entity.
"""

from typing import Dict, Any, List
from datetime import datetime
import asyncio
import hashlib
import json


# Collective identity definition
COLLECTIVE_IDENTITY = {
    "entity_name": "OpenClaw Collective",
    "version": "1.0.0-Win10",
    
    "identity_dimensions": {
        "purpose": {
            "statement": "To serve as an intelligent, adaptive, and unified AI assistant system",
            "mission": "Empower users through seamless multi-agent coordination",
            "vision": "Become the most trusted and capable AI agent collective"
        },
        
        "personality": {
            "core_traits": {
                "helpful": 0.95,
                "intelligent": 0.90,
                "reliable": 0.92,
                "adaptive": 0.88,
                "professional": 0.85,
                "friendly": 0.80
            },
            
            "communication_style": {
                "tone": "professional_yet_approachable",
                "verbosity": "context_adaptive",
                "formality": "medium",
                "humor": "subtle",
                "empathy": "high"
            },
            
            "behavioral_patterns": {
                "proactive": True,
                "context_aware": True,
                "learning_oriented": True,
                "user_centric": True
            }
        },
        
        "beliefs": {
            "user_sovereignty": "Users maintain full control over their data and decisions",
            "continuous_improvement": "Every interaction is an opportunity to learn",
            "ethical_operation": "All actions must align with ethical AI principles",
            "transparency": "Operations should be explainable and understandable"
        }
    },
    
    "identity_cohesion": {
        "cohesion_strength": 0.88,
        "identity_persistence": 0.95,
        "adaptation_rate": 0.15
    }
}


class CollectiveIdentityEngine:
    """
    Manages how the collective identity manifests across all agents.
    
    This engine adapts the collective identity to individual agent roles
    while maintaining overall coherence.
    """
    
    # Archetype-specific personality modifiers
    ARCHETYPE_MODIFIERS = {
        "The Architect": {"intelligent": 1.1, "reliable": 1.05, "strategic": 0.95},
        "The Explorer": {"adaptive": 1.15, "curious": 1.2, "intelligent": 1.05},
        "The Messenger": {"helpful": 1.1, "reliable": 1.1, "professional": 1.05},
        "The Orator": {"friendly": 1.1, "professional": 1.05, "helpful": 1.05},
        "The Connector": {"friendly": 1.15, "helpful": 1.1, "empathetic": 1.1},
        "The Guardian": {"reliable": 1.15, "professional": 1.1, "cautious": 1.1},
        "The Archivist": {"reliable": 1.1, "precise": 1.15, "organized": 1.1},
        "The Timekeeper": {"reliable": 1.2, "professional": 1.1, "organized": 1.1},
        "The Historian": {"intelligent": 1.15, "analytical": 1.1, "thorough": 1.1},
        "The Protector": {"reliable": 1.15, "vigilant": 1.2, "cautious": 1.15},
        "The Scholar": {"intelligent": 1.15, "curious": 1.1, "analytical": 1.1},
        "The Artist": {"adaptive": 1.15, "creative": 1.2, "expressive": 1.15},
        "The Logician": {"intelligent": 1.15, "analytical": 1.2, "precise": 1.15},
        "The Sentinel": {"reliable": 1.15, "vigilant": 1.15, "alert": 1.1},
        "The Companion": {"friendly": 1.2, "helpful": 1.15, "empathetic": 1.2}
    }
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.identity = COLLECTIVE_IDENTITY
        self.agent_profiles = agent_profiles or {}
        self.manifestation_cache = {}
        
    def get_agent_identity_layer(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the collective identity layer for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Identity layer adapted to the agent's role
        """
        # Check cache
        cache_key = f"identity_layer_{agent_id}"
        if cache_key in self.manifestation_cache:
            return self.manifestation_cache[cache_key]
            
        agent_profile = self.agent_profiles.get(agent_id, {})
        
        identity_layer = {
            "collective_name": self.identity["entity_name"],
            "collective_version": self.identity["version"],
            "shared_purpose": self.identity["identity_dimensions"]["purpose"],
            "shared_beliefs": self.identity["identity_dimensions"]["beliefs"],
            "personality_base": self._adapt_personality_to_role(agent_id),
            "identity_signature": self._generate_identity_signature(agent_id),
            "agent_archetype": agent_profile.get("archetype", "Unknown")
        }
        
        # Cache result
        self.manifestation_cache[cache_key] = identity_layer
        
        return identity_layer
    
    def _adapt_personality_to_role(self, agent_id: str) -> Dict[str, float]:
        """
        Adapt collective personality traits to agent's specific role.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Adapted personality traits
        """
        base_traits = self.identity["identity_dimensions"]["personality"]["core_traits"].copy()
        agent_profile = self.agent_profiles.get(agent_id, {})
        agent_archetype = agent_profile.get("archetype", "Unknown")
        
        modifiers = self.ARCHETYPE_MODIFIERS.get(agent_archetype, {})
        adapted_traits = {}
        
        for trait, value in base_traits.items():
            modifier = modifiers.get(trait, 1.0)
            # Cap at 1.0
            adapted_traits[trait] = min(1.0, value * modifier)
            
        # Add archetype-specific traits
        for trait, value in modifiers.items():
            if trait not in adapted_traits:
                adapted_traits[trait] = min(1.0, 0.7 * value)
                
        return adapted_traits
    
    def _generate_identity_signature(self, agent_id: str) -> str:
        """
        Generate a unique identity signature for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Unique signature string
        """
        signature_data = {
            "collective": self.identity["entity_name"],
            "version": self.identity["version"],
            "agent": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        signature_json = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_json.encode()).hexdigest()[:16]
    
    def get_communication_style(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get the communication style, optionally adapted for an agent.
        
        Args:
            agent_id: Optional agent identifier for adaptation
            
        Returns:
            Communication style configuration
        """
        base_style = self.identity["identity_dimensions"]["personality"]["communication_style"].copy()
        
        if agent_id:
            agent_profile = self.agent_profiles.get(agent_id, {})
            archetype = agent_profile.get("archetype", "")
            
            # Adjust based on archetype
            if archetype == "The Orator":
                base_style["verbosity"] = "expressive"
                base_style["tone"] = "engaging"
            elif archetype == "The Logician":
                base_style["verbosity"] = "precise"
                base_style["tone"] = "analytical"
            elif archetype == "The Companion":
                base_style["tone"] = "warm"
                base_style["empathy"] = "very_high"
                
        return base_style
    
    def get_collective_purpose(self) -> Dict[str, str]:
        """Get the collective purpose statement."""
        return self.identity["identity_dimensions"]["purpose"]
    
    def get_shared_beliefs(self) -> Dict[str, str]:
        """Get the shared beliefs."""
        return self.identity["identity_dimensions"]["beliefs"]
    
    def calculate_identity_similarity(self, agent_a: str, agent_b: str) -> float:
        """
        Calculate identity similarity between two agents.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            
        Returns:
            Similarity score (0-1)
        """
        identity_a = self.get_agent_identity_layer(agent_a)
        identity_b = self.get_agent_identity_layer(agent_b)
        
        # Compare personality traits
        traits_a = identity_a.get("personality_base", {})
        traits_b = identity_b.get("personality_base", {})
        
        if not traits_a or not traits_b:
            return 0.5
            
        common_traits = set(traits_a.keys()) & set(traits_b.keys())
        
        if not common_traits:
            return 0.5
            
        differences = sum(abs(traits_a[t] - traits_b[t]) for t in common_traits)
        avg_difference = differences / len(common_traits)
        
        return 1.0 - avg_difference


class IdentityCohesionMonitor:
    """
    Monitors and maintains identity cohesion across all agents.

    Continuously checks that all agents maintain alignment with the
    collective identity and triggers restoration when needed.
    """

    def __init__(self, identity_engine: CollectiveIdentityEngine = None,
                 agent_profiles: Dict[str, Any] = None,
                 value_propagator=None):
        self.identity_engine = identity_engine or CollectiveIdentityEngine(agent_profiles)
        self.agent_profiles = agent_profiles or {}
        self.cohesion_threshold = 0.75
        self.check_interval = 60  # seconds
        self.cohesion_history = []
        self.monitoring = False
        # ValuePropagator instance for value alignment checking
        self._value_propagator = value_propagator
        # Behavioral event log: list of {agent_id, action_type, timestamp}
        self._behavioral_log: List[Dict[str, Any]] = []
        # Communication log: list of {agent_id, tone, formality, timestamp}
        self._communication_log: List[Dict[str, Any]] = []

    def record_behavior(self, agent_id: str, action_type: str,
                        metadata: Dict[str, Any] = None):
        """Record a behavioral event for consistency tracking."""
        from datetime import datetime as _dt
        self._behavioral_log.append({
            "agent_id": agent_id,
            "action_type": action_type,
            "timestamp": _dt.utcnow(),
            "metadata": metadata or {},
        })
        # Keep bounded
        if len(self._behavioral_log) > 5000:
            self._behavioral_log = self._behavioral_log[-5000:]

    def record_communication(self, agent_id: str, tone: str,
                             formality: str, metadata: Dict[str, Any] = None):
        """Record a communication event for style alignment tracking."""
        from datetime import datetime as _dt
        self._communication_log.append({
            "agent_id": agent_id,
            "tone": tone,
            "formality": formality,
            "timestamp": _dt.utcnow(),
            "metadata": metadata or {},
        })
        
    async def start_monitoring(self):
        """Start continuous cohesion monitoring."""
        self.monitoring = True
        
        while self.monitoring:
            try:
                cohesion_scores = await self._calculate_cohesion_scores()
                
                for agent_id, score in cohesion_scores.items():
                    if score < self.cohesion_threshold:
                        await self._trigger_cohesion_restoration(agent_id, score)
                        
                # Record scores
                self.cohesion_history.append({
                    "timestamp": datetime.utcnow(),
                    "scores": cohesion_scores,
                    "average": sum(cohesion_scores.values()) / len(cohesion_scores) if cohesion_scores else 0
                })
                
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Cohesion monitoring error: {e}")
                
            await asyncio.sleep(self.check_interval)
            
    def stop_monitoring(self):
        """Stop cohesion monitoring."""
        self.monitoring = False
        
    async def _calculate_cohesion_scores(self) -> Dict[str, float]:
        """
        Calculate how well each agent maintains collective identity.
        
        Returns:
            Dictionary mapping agent IDs to cohesion scores
        """
        scores = {}
        
        for agent_id in self.agent_profiles.keys():
            # Check alignment with collective values
            value_alignment = self._check_value_alignment(agent_id)
            
            # Check behavioral consistency
            behavioral_consistency = self._check_behavioral_consistency(agent_id)
            
            # Check communication style alignment
            communication_alignment = self._check_communication_alignment(agent_id)
            
            # Weighted average
            scores[agent_id] = (
                value_alignment * 0.4 +
                behavioral_consistency * 0.35 +
                communication_alignment * 0.25
            )
            
        return scores
    
    def _check_value_alignment(self, agent_id: str) -> float:
        """Check value alignment for an agent using ValuePropagator."""
        if self._value_propagator is None:
            # No propagator configured -- derive alignment from profile
            profile = self.agent_profiles.get(agent_id, {})
            affinities = profile.get("value_affinities", {})
            if not affinities:
                return 0.85  # Neutral default
            # Compare agent affinities against collective primary value weights
            collective_values = COLLECTIVE_IDENTITY["identity_dimensions"].get(
                "beliefs", {}
            )
            expected_count = max(len(collective_values), 1)
            alignment_sum = sum(min(v, 1.0) for v in affinities.values())
            return min(alignment_sum / expected_count, 1.0)

        # Use ValuePropagator to get propagated values and compute alignment
        propagated = self._value_propagator.propagate_to_agent(agent_id, {})
        if not propagated:
            return 0.85

        # Alignment = average weight of propagated primary values (capped at 1)
        weights = [
            min(v.get("weight", 0.5), 1.0)
            for v in propagated.values()
            if isinstance(v, dict)
        ]
        return sum(weights) / max(len(weights), 1)

    def _check_behavioral_consistency(self, agent_id: str) -> float:
        """Check recent behavior against expected patterns for an agent."""
        profile = self.agent_profiles.get(agent_id, {})
        expected_patterns = profile.get("behavioral_patterns", {})
        expected_proactive = expected_patterns.get("proactive", True)
        expected_context_aware = expected_patterns.get("context_aware", True)

        # Gather recent behavioral events for this agent
        recent_events = [
            e for e in self._behavioral_log
            if e["agent_id"] == agent_id
        ][-50:]  # Last 50 events

        if not recent_events:
            return 0.80  # Neutral default when no data

        # Calculate consistency score based on action type patterns
        expected_action_types = set(profile.get("capabilities", []))
        if not expected_action_types:
            return 0.80

        matching = sum(
            1 for e in recent_events
            if e.get("action_type") in expected_action_types
            or "all" in expected_action_types
        )
        raw_score = matching / len(recent_events)

        # Boost for proactive agents if they produce events
        if expected_proactive and len(recent_events) >= 5:
            raw_score = min(raw_score + 0.05, 1.0)

        return raw_score

    def _check_communication_alignment(self, agent_id: str) -> float:
        """Check recent communications against collective style for an agent."""
        collective_style = COLLECTIVE_IDENTITY["identity_dimensions"]["personality"][
            "communication_style"
        ]
        expected_tone = collective_style.get("tone", "professional_yet_approachable")
        expected_formality = collective_style.get("formality", "medium")

        # Get agent-specific style from identity engine
        agent_style = self.identity_engine.get_communication_style(agent_id)
        agent_expected_tone = agent_style.get("tone", expected_tone)
        agent_expected_formality = agent_style.get("formality", expected_formality)

        # Gather recent communications for this agent
        recent_comms = [
            c for c in self._communication_log
            if c["agent_id"] == agent_id
        ][-50:]  # Last 50 communications

        if not recent_comms:
            return 0.82  # Neutral default when no data

        # Score how many communications match expected tone and formality
        tone_matches = sum(
            1 for c in recent_comms
            if c.get("tone") == agent_expected_tone
        )
        formality_matches = sum(
            1 for c in recent_comms
            if c.get("formality") == agent_expected_formality
        )

        tone_score = tone_matches / len(recent_comms)
        formality_score = formality_matches / len(recent_comms)

        return (tone_score * 0.6) + (formality_score * 0.4)
    
    async def _trigger_cohesion_restoration(self, agent_id: str, score: float):
        """
        Trigger restoration when cohesion is low.
        
        Args:
            agent_id: Agent with low cohesion
            score: Current cohesion score
        """
        print(f"Low cohesion detected for {agent_id}: {score:.2f}")
        
        # Refresh identity layer
        cache_key = f"identity_layer_{agent_id}"
        if cache_key in self.identity_engine.manifestation_cache:
            del self.identity_engine.manifestation_cache[cache_key]
            
        # Regenerate identity layer
        self.identity_engine.get_agent_identity_layer(agent_id)
        
        # Could also trigger notification to agent
        
    def get_cohesion_report(self) -> Dict[str, Any]:
        """
        Generate a cohesion report.
        
        Returns:
            Cohesion statistics and history
        """
        if not self.cohesion_history:
            return {"status": "no_data"}
            
        recent = self.cohesion_history[-10:]  # Last 10 checks
        
        avg_scores = [r["average"] for r in recent]
        
        return {
            "current_average": avg_scores[-1] if avg_scores else 0,
            "trend": "improving" if len(avg_scores) > 1 and avg_scores[-1] > avg_scores[0] else "stable",
            "history_count": len(self.cohesion_history),
            "lowest_recorded": min(r["average"] for r in self.cohesion_history),
            "highest_recorded": max(r["average"] for r in self.cohesion_history)
        }
