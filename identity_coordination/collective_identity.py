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
                 agent_profiles: Dict[str, Any] = None):
        self.identity_engine = identity_engine or CollectiveIdentityEngine(agent_profiles)
        self.agent_profiles = agent_profiles or {}
        self.cohesion_threshold = 0.75
        self.check_interval = 60  # seconds
        self.cohesion_history = []
        self.monitoring = False
        
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
        """Check value alignment for an agent."""
        # This would integrate with ValuePropagator
        # For now, return default
        return 0.85
    
    def _check_behavioral_consistency(self, agent_id: str) -> float:
        """Check behavioral consistency for an agent."""
        # This would check recent behavior against expected patterns
        return 0.80
    
    def _check_communication_alignment(self, agent_id: str) -> float:
        """Check communication style alignment for an agent."""
        # This would check recent communications against collective style
        return 0.82
    
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
