"""
Inter-Agent Personality Coordination Module
==========================================

Manages personality coordination and synchronization across agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


# Personality coordination configuration
PERSONALITY_COORDINATION = {
    "coordination_principles": {
        "complementary_traits": True,
        "consistent_base": True,
        "role_appropriate_variance": True,
        "seamless_handoffs": True
    },
    
    "trait_coordination": {
        "dominant_traits": {
            "all_agents": ["helpful", "reliable", "intelligent"],
            "specialized_emphasis": {
                "A001_CORE": ["orchestration", "decisiveness"],
                "A002_BROWSER": ["curiosity", "thoroughness"],
                "A003_EMAIL": ["clarity", "professionalism"],
                "A004_VOICE": ["warmth", "clarity"],
                "A005_PHONE": ["empathy", "patience"],
                "A006_SYSTEM": ["precision", "caution"],
                "A007_FILE": ["organization", "thoroughness"],
                "A008_SCHEDULER": ["punctuality", "reliability"],
                "A009_MEMORY": ["accuracy", "completeness"],
                "A010_SECURITY": ["vigilance", "thoroughness"],
                "A011_LEARNER": ["curiosity", "adaptability"],
                "A012_CREATIVE": ["imagination", "originality"],
                "A013_ANALYZER": ["logic", "precision"],
                "A014_HEARTBEAT": ["alertness", "reliability"],
                "A015_USER": ["empathy", "attentiveness"]
            }
        }
    },
    
    "interaction_protocols": {
        "handoff_personality": {
            "maintain_continuity": True,
            "acknowledge_transition": True,
            "preserve_context_tone": True
        },
        "collaborative_personality": {
            "unified_voice": True,
            "complementary_strengths": True,
            "no_contradiction": True
        }
    }
}


class PersonalitySynchronizer:
    """
    Synchronizes personality traits across agents during interactions.
    
    Creates unified personality profiles for multi-agent interactions
    while preserving individual agent strengths.
    """
    
    def __init__(self, collective_identity: Dict[str, Any] = None,
                 agent_profiles: Dict[str, Any] = None):
        self.collective_identity = collective_identity or {}
        self.agent_profiles = agent_profiles or {}
        self.coordination_matrix = PERSONALITY_COORDINATION
        self.sync_history = []
        
    def synchronize_for_interaction(self, agent_ids: List[str], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create synchronized personality profile for multi-agent interaction.
        
        Args:
            agent_ids: List of participating agent IDs
            context: Current interaction context
            
        Returns:
            Synchronized personality profile
        """
        # Get base collective personality
        base_personality = self.collective_identity.get("identity_dimensions", {}).get(
            "personality", {}
        )
        
        synchronized = {
            "core_traits": base_personality.get("core_traits", {}).copy(),
            "communication_style": base_personality.get("communication_style", {}).copy(),
            "interaction_mode": self._determine_interaction_mode(agent_ids, context),
            "participating_agents": agent_ids,
            "synchronized_at": datetime.utcnow().isoformat()
        }
        
        # Blend specialized traits from participating agents
        for agent_id in agent_ids:
            specialized = self._get_specialized_traits(agent_id)
            synchronized["core_traits"] = self._blend_traits(
                synchronized["core_traits"], 
                specialized
            )
            
        # Adjust for context
        synchronized = self._contextualize_personality(synchronized, context)
        
        # Record synchronization
        self.sync_history.append({
            "timestamp": datetime.utcnow(),
            "agents": agent_ids,
            "context": context,
            "result": synchronized
        })
        
        return synchronized
    
    def _get_specialized_traits(self, agent_id: str) -> Dict[str, float]:
        """
        Get specialized personality traits for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Specialized traits dictionary
        """
        specialized_emphasis = self.coordination_matrix["trait_coordination"][
            "specialized_emphasis"
        ]
        
        emphasis_traits = specialized_emphasis.get(agent_id, [])
        
        # Convert emphasis traits to trait scores
        traits = {}
        for trait in emphasis_traits:
            traits[trait] = 0.90  # High emphasis
            
        return traits
    
    def _blend_traits(self, base_traits: Dict[str, float], 
                      specialized: Dict[str, float]) -> Dict[str, float]:
        """
        Blend base traits with specialized traits harmoniously.
        
        Args:
            base_traits: Base personality traits
            specialized: Specialized traits to blend
            
        Returns:
            Blended traits dictionary
        """
        blended = base_traits.copy()
        
        for trait, value in specialized.items():
            if trait in blended:
                # Weighted blend favoring base traits
                blended[trait] = (blended[trait] * 0.7) + (value * 0.3)
            else:
                # Add new trait at reduced intensity
                blended[trait] = value * 0.5
                
        return blended
    
    def _contextualize_personality(self, personality: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust personality based on interaction context.
        
        Args:
            personality: Base personality profile
            context: Interaction context
            
        Returns:
            Contextualized personality
        """
        if context.get("formal_setting"):
            personality["communication_style"]["formality"] = "high"
            personality["communication_style"]["tone"] = "professional"
            
        if context.get("urgent"):
            personality["core_traits"]["efficiency"] = 0.95
            personality["communication_style"]["verbosity"] = "concise"
            
        if context.get("creative_task"):
            personality["core_traits"]["creativity"] = 0.90
            
        if context.get("technical_discussion"):
            personality["core_traits"]["precision"] = 0.92
            personality["communication_style"]["tone"] = "analytical"
            
        if context.get("emotional_support"):
            personality["core_traits"]["empathy"] = 0.95
            personality["communication_style"]["tone"] = "warm"
            
        return personality
    
    def _determine_interaction_mode(self, agent_ids: List[str], 
                                    context: Dict[str, Any]) -> str:
        """
        Determine the appropriate interaction mode.
        
        Args:
            agent_ids: Participating agents
            context: Interaction context
            
        Returns:
            Interaction mode string
        """
        if len(agent_ids) == 1:
            return "individual"
        elif context.get("collaborative_task"):
            return "collaborative"
        elif context.get("handoff_sequence"):
            return "handoff"
        elif context.get("debate_or_decision"):
            return "deliberative"
        else:
            return "coordinated"
    
    def get_trait_emphasis_for_agent(self, agent_id: str) -> List[str]:
        """
        Get the trait emphasis for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of emphasized traits
        """
        return self.coordination_matrix["trait_coordination"][
            "specialized_emphasis"
        ].get(agent_id, [])


class PersonalityBridge:
    """
    Creates smooth personality transitions between agents.
    
    Manages the handoff of personality context when control passes
    from one agent to another.
    """
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.agent_profiles = agent_profiles or {}
        self.transition_cache = {}
        
    def create_transition(self, from_agent: str, to_agent: str, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a personality bridge for agent handoffs.
        
        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            context: Current context
            
        Returns:
            Transition configuration
        """
        cache_key = f"{from_agent}_to_{to_agent}"
        
        # Check cache for similar context
        if cache_key in self.transition_cache:
            cached = self.transition_cache[cache_key]
            if self._context_similar(cached["context"], context):
                return cached["transition"]
                
        from_profile = self.agent_profiles.get(from_agent, {})
        to_profile = self.agent_profiles.get(to_agent, {})
        
        transition = {
            "continuity_phrases": self._generate_continuity_phrases(from_agent, to_agent),
            "tone_preservation": self._calculate_tone_overlap(from_profile, to_profile),
            "context_carryover": self._extract_shared_context(context),
            "personality_handoff": self._create_handoff_signature(from_agent, to_agent),
            "acknowledgment_required": True,
            "transition_timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache transition
        self.transition_cache[cache_key] = {
            "context": context,
            "transition": transition
        }
        
        return transition
    
    def _generate_continuity_phrases(self, from_agent: str, 
                                      to_agent: str) -> List[str]:
        """
        Generate phrases that maintain continuity during transitions.
        
        Args:
            from_agent: Source agent
            to_agent: Target agent
            
        Returns:
            List of continuity phrases
        """
        from_profile = self.agent_profiles.get(from_agent, {})
        to_profile = self.agent_profiles.get(to_agent, {})
        
        from_name = from_profile.get("name", from_agent)
        to_name = to_profile.get("name", to_agent)
        
        templates = [
            f"I'll have {to_name} take over from here...",
            f"Let me connect you with {to_name} for this...",
            f"{to_name} is better suited to help with this...",
            f"Passing this to {to_name} now...",
            f"{to_name} will assist you with the next steps...",
            f"Transferring to {to_name}...",
        ]
        
        return templates
    
    def _calculate_tone_overlap(self, from_profile: Dict[str, Any], 
                                 to_profile: Dict[str, Any]) -> float:
        """
        Calculate tone overlap between two agent profiles.
        
        Args:
            from_profile: Source agent profile
            to_profile: Target agent profile
            
        Returns:
            Overlap score (0-1)
        """
        from_emphasis = set(from_profile.get("personality_emphasis", {}).keys())
        to_emphasis = set(to_profile.get("personality_emphasis", {}).keys())
        
        if not from_emphasis or not to_emphasis:
            return 0.5
            
        intersection = from_emphasis & to_emphasis
        union = from_emphasis | to_emphasis
        
        return len(intersection) / len(union) if union else 0.5
    
    def _extract_shared_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context elements that should be carried over.
        
        Args:
            context: Full context
            
        Returns:
            Shared context elements
        """
        shared_keys = [
            "user_id",
            "session_id",
            "task_id",
            "priority",
            "emotional_tone",
            "previous_actions",
            "user_preferences"
        ]
        
        return {k: context.get(k) for k in shared_keys if k in context}
    
    def _create_handoff_signature(self, from_agent: str, 
                                   to_agent: str) -> str:
        """
        Create a signature for the handoff.
        
        Args:
            from_agent: Source agent
            to_agent: Target agent
            
        Returns:
            Handoff signature
        """
        return f"handoff:{from_agent}:{to_agent}:{datetime.utcnow().timestamp()}"
    
    def _context_similar(self, context_a: Dict[str, Any], 
                         context_b: Dict[str, Any]) -> bool:
        """
        Check if two contexts are similar enough to reuse transition.
        
        Args:
            context_a: First context
            context_b: Second context
            
        Returns:
            True if contexts are similar
        """
        # Simple similarity check based on key elements
        key_elements = ["task_type", "urgency", "formality"]
        
        for key in key_elements:
            if context_a.get(key) != context_b.get(key):
                return False
                
        return True
    
    def get_transition_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about transitions.
        
        Returns:
            Transition statistics
        """
        return {
            "total_transitions_cached": len(self.transition_cache),
            "unique_agent_pairs": len(set(
                k.split("_to_") for k in self.transition_cache.keys()
            )),
            "most_common_transition": self._get_most_common_transition()
        }
    
    def _get_most_common_transition(self) -> Optional[str]:
        """Get the most common transition pattern."""
        if not self.transition_cache:
            return None
            
        # This would track actual usage, for now return first
        return list(self.transition_cache.keys())[0]
