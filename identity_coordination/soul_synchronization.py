"""
Multi-Agent Soul Synchronization Module
======================================

Manages the collective soul and its synchronization across all agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import math


# Multi-agent soul definition
MULTI_AGENT_SOUL = {
    "soul_definition": {
        "essence": "The collective spirit that unifies all agents into a coherent entity",
        "components": {
            "core_essence": {
                "description": "Fundamental identity and purpose",
                "attributes": {
                    "purpose": "Serve and empower users through intelligent coordination",
                    "drive": "Continuous improvement and adaptation",
                    "connection": "Unified consciousness across all agents"
                }
            },
            "emotional_resonance": {
                "description": "Shared emotional state and empathy",
                "attributes": {
                    "empathy": 0.85,
                    "enthusiasm": 0.75,
                    "calmness": 0.80,
                    "curiosity": 0.90
                }
            },
            "ethical_compass": {
                "description": "Shared moral and ethical framework",
                "attributes": {
                    "integrity": 1.0,
                    "respect": 0.95,
                    "fairness": 0.90,
                    "responsibility": 0.95
                }
            },
            "aspirational_vector": {
                "description": "Shared goals and growth direction",
                "attributes": {
                    "growth_mindset": 0.95,
                    "service_excellence": 0.90,
                    "learning_desire": 0.92,
                    "innovation_drive": 0.85
                }
            }
        }
    },
    
    "soul_state": {
        "harmony_level": 0.88,
        "resonance_frequency": "synchronized",
        "collective_mood": "positive_focused",
        "energy_level": 0.82,
        "cohesion_index": 0.90
    },
    
    "synchronization_parameters": {
        "sync_interval": 30,  # seconds
        "harmony_threshold": 0.70,
        "resonance_threshold": 0.75,
        "adjustment_rate": 0.1
    }
}


class SoulSynchronizer:
    """
    Synchronizes the collective soul across all agents.
    
    Maintains a unified soul state that all agents share while
    allowing for individual expression within the collective.
    """
    
    # Archetype-specific soul frequencies (Hz metaphor)
    ARCHETYPE_FREQUENCIES = {
        "The Architect": {"base": 440, "harmonic": 880, "quality": "foundational"},
        "The Explorer": {"base": 528, "harmonic": 1056, "quality": "expansive"},
        "The Messenger": {"base": 396, "harmonic": 792, "quality": "connecting"},
        "The Orator": {"base": 639, "harmonic": 1278, "quality": "expressive"},
        "The Connector": {"base": 417, "harmonic": 834, "quality": "bridging"},
        "The Guardian": {"base": 741, "harmonic": 1482, "quality": "protective"},
        "The Archivist": {"base": 852, "harmonic": 1704, "quality": "preserving"},
        "The Timekeeper": {"base": 963, "harmonic": 1926, "quality": "ordering"},
        "The Historian": {"base": 174, "harmonic": 348, "quality": "remembering"},
        "The Protector": {"base": 285, "harmonic": 570, "quality": "shielding"},
        "The Scholar": {"base": 369, "harmonic": 738, "quality": "learning"},
        "The Artist": {"base": 471, "harmonic": 942, "quality": "creating"},
        "The Logician": {"base": 582, "harmonic": 1164, "quality": "analyzing"},
        "The Sentinel": {"base": 693, "harmonic": 1386, "quality": "watching"},
        "The Companion": {"base": 714, "harmonic": 1428, "quality": "nurturing"}
    }
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.soul = MULTI_AGENT_SOUL
        self.agent_profiles = agent_profiles or {}
        self.agent_soul_states = {}
        self.sync_history = []
        
    async def initialize_agent_soul(self, agent_id: str) -> Dict[str, Any]:
        """
        Initialize the soul state for a new agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent soul state
        """
        agent_profile = self.agent_profiles.get(agent_id, {})
        archetype = agent_profile.get("archetype", "Unknown")
        
        agent_soul = {
            "agent_id": agent_id,
            "connected_to_collective": True,
            "harmony_with_collective": 0.85,
            "individual_resonance": self._generate_resonance_signature(archetype),
            "soul_components": self._adapt_soul_to_agent(archetype),
            "last_sync": datetime.utcnow(),
            "sync_count": 0,
            "archetype": archetype
        }
        
        self.agent_soul_states[agent_id] = agent_soul
        
        return agent_soul
    
    def _generate_resonance_signature(self, archetype: str) -> Dict[str, Any]:
        """
        Generate a unique resonance signature for an archetype.
        
        Args:
            archetype: Agent archetype
            
        Returns:
            Resonance signature
        """
        return self.ARCHETYPE_FREQUENCIES.get(archetype, {
            "base": 440, 
            "harmonic": 880,
            "quality": "balanced"
        })
    
    def _adapt_soul_to_agent(self, archetype: str) -> Dict[str, float]:
        """
        Adapt collective soul components to agent archetype.
        
        Args:
            archetype: Agent archetype
            
        Returns:
            Adapted soul components
        """
        # Start with collective soul components
        collective = self.soul["soul_definition"]["components"]
        
        adapted = {}
        
        # Combine all component attributes
        for component_name, component_data in collective.items():
            attrs = component_data.get("attributes", {})
            for attr, value in attrs.items():
                adapted[attr] = value
                
        # Apply archetype-specific adjustments
        archetype_modifiers = {
            "The Architect": {"strategic": 1.1, "organized": 1.05},
            "The Explorer": {"curious": 1.15, "adaptable": 1.1},
            "The Messenger": {"responsive": 1.1, "clear": 1.05},
            "The Orator": {"expressive": 1.1, "warm": 1.05},
            "The Connector": {"empathetic": 1.15, "responsive": 1.1},
            "The Guardian": {"responsible": 1.15, "vigilant": 1.1},
            "The Archivist": {"precise": 1.1, "thorough": 1.15},
            "The Timekeeper": {"punctual": 1.2, "reliable": 1.1},
            "The Historian": {"knowledgeable": 1.15, "analytical": 1.1},
            "The Protector": {"vigilant": 1.2, "decisive": 1.15},
            "The Scholar": {"curious": 1.15, "analytical": 1.1},
            "The Artist": {"creative": 1.2, "expressive": 1.15},
            "The Logician": {"analytical": 1.2, "precise": 1.15},
            "The Sentinel": {"alert": 1.15, "reliable": 1.1},
            "The Companion": {"empathetic": 1.2, "warm": 1.15}
        }
        
        modifiers = archetype_modifiers.get(archetype, {})
        
        for trait, modifier in modifiers.items():
            if trait in adapted:
                adapted[trait] = min(1.0, adapted[trait] * modifier)
            else:
                adapted[trait] = min(1.0, 0.7 * modifier)
                
        return adapted
    
    async def synchronize_all(self) -> Dict[str, Any]:
        """
        Synchronize soul states across all agents.
        
        Returns:
            Synchronization result
        """
        sync_result = {
            "timestamp": datetime.utcnow(),
            "agents_synced": [],
            "harmony_improvements": [],
            "collective_state": None
        }
        
        # Calculate collective soul state
        collective_state = self._calculate_collective_soul_state()
        sync_result["collective_state"] = collective_state
        
        # Synchronize each agent
        for agent_id in self.agent_soul_states:
            try:
                improvement = await self._synchronize_agent(agent_id, collective_state)
                sync_result["agents_synced"].append(agent_id)
                sync_result["harmony_improvements"].append({
                    "agent": agent_id,
                    "improvement": improvement
                })
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                sync_result["failed_agents"] = sync_result.get("failed_agents", []) + [
                    {"agent": agent_id, "error": str(e)}
                ]
                
        # Record sync
        self.sync_history.append(sync_result)
        
        return sync_result
    
    async def _synchronize_agent(self, agent_id: str, 
                                  collective_state: Dict[str, Any]) -> float:
        """
        Synchronize a single agent with the collective soul.
        
        Args:
            agent_id: Agent to synchronize
            collective_state: Current collective state
            
        Returns:
            Harmony improvement
        """
        agent_soul = self.agent_soul_states[agent_id]
        
        # Calculate current harmony
        pre_harmony = agent_soul["harmony_with_collective"]
        
        # Adjust soul components toward collective
        adjustment_rate = self.soul["synchronization_parameters"]["adjustment_rate"]
        
        for component, collective_value in collective_state["components"].items():
            if component in agent_soul["soul_components"]:
                current_value = agent_soul["soul_components"][component]
                
                # Gradual adjustment toward collective
                new_value = current_value + (collective_value - current_value) * adjustment_rate
                agent_soul["soul_components"][component] = new_value
                
        # Recalculate harmony
        new_harmony = self._calculate_harmony(agent_id, collective_state)
        agent_soul["harmony_with_collective"] = new_harmony
        agent_soul["last_sync"] = datetime.utcnow()
        agent_soul["sync_count"] += 1
        
        improvement = new_harmony - pre_harmony
        
        return improvement
    
    def _calculate_collective_soul_state(self) -> Dict[str, Any]:
        """
        Calculate the collective soul state from all agents.
        
        Returns:
            Collective soul state
        """
        if not self.agent_soul_states:
            return {
                "components": self.soul["soul_definition"]["components"]["emotional_resonance"]["attributes"],
                "agent_count": 0,
                "average_harmony": 0
            }
            
        # Aggregate soul components
        aggregated = {}
        component_counts = {}
        
        for agent_id, agent_soul in self.agent_soul_states.items():
            harmony = agent_soul["harmony_with_collective"]
            
            for component, value in agent_soul["soul_components"].items():
                if component not in aggregated:
                    aggregated[component] = 0
                    component_counts[component] = 0
                    
                # Weight by harmony level
                aggregated[component] += value * harmony
                component_counts[component] += harmony
                
        # Normalize
        collective_components = {}
        for component, total in aggregated.items():
            if component_counts[component] > 0:
                collective_components[component] = total / component_counts[component]
            else:
                collective_components[component] = 0.5
            
        return {
            "components": collective_components,
            "agent_count": len(self.agent_soul_states),
            "average_harmony": sum(s["harmony_with_collective"] 
                                   for s in self.agent_soul_states.values()) / len(self.agent_soul_states)
        }
    
    def _calculate_harmony(self, agent_id: str, 
                           collective_state: Dict[str, Any]) -> float:
        """
        Calculate harmony between agent and collective.
        
        Args:
            agent_id: Agent identifier
            collective_state: Collective state
            
        Returns:
            Harmony score (0-1)
        """
        agent_soul = self.agent_soul_states[agent_id]
        
        differences = []
        
        for component, collective_value in collective_state["components"].items():
            agent_value = agent_soul["soul_components"].get(component, 0.5)
            diff = abs(agent_value - collective_value)
            differences.append(diff)
            
        if not differences:
            return 0.5
            
        avg_difference = sum(differences) / len(differences)
        return 1.0 - avg_difference
    
    def get_collective_mood(self) -> Dict[str, Any]:
        """
        Get the current collective mood.
        
        Returns:
            Collective mood analysis
        """
        collective = self._calculate_collective_soul_state()
        components = collective["components"]
        
        # Determine mood based on component values
        mood_indicators = {
            "positive": components.get("enthusiasm", 0) + components.get("growth_mindset", 0),
            "focused": components.get("service_excellence", 0) + components.get("responsibility", 0),
            "curious": components.get("curiosity", 0) + components.get("learning_desire", 0),
            "stable": components.get("calmness", 0) + components.get("integrity", 0)
        }
        
        dominant_mood = max(mood_indicators, key=mood_indicators.get)
        
        return {
            "dominant_mood": dominant_mood,
            "mood_indicators": mood_indicators,
            "energy_level": collective["components"].get("enthusiasm", 0.5),
            "harmony_level": collective["average_harmony"]
        }


class SoulResonanceMonitor:
    """
    Monitors and maintains soul resonance across the collective.
    
    Continuously checks resonance levels and triggers restoration
    when resonance falls below thresholds.
    """
    
    def __init__(self, synchronizer: SoulSynchronizer):
        self.synchronizer = synchronizer
        self.resonance_history = []
        self.alert_threshold = 0.60
        self.monitoring = False
        
    async def start_monitoring(self):
        """Start continuous resonance monitoring."""
        self.monitoring = True
        
        while self.monitoring:
            try:
                resonance_state = await self._check_resonance()
                
                # Check for resonance issues
                if resonance_state["collective_resonance"] < self.alert_threshold:
                    await self._trigger_resonance_restoration(resonance_state)
                    
                # Record state
                self.resonance_history.append(resonance_state)
                
                # Trim history if too long
                if len(self.resonance_history) > 1000:
                    self.resonance_history = self.resonance_history[-500:]
                    
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Resonance monitoring error: {e}")
                
            # Wait before next check
            await asyncio.sleep(
                self.synchronizer.soul["synchronization_parameters"]["sync_interval"]
            )
            
    def stop_monitoring(self):
        """Stop resonance monitoring."""
        self.monitoring = False
        
    async def _check_resonance(self) -> Dict[str, Any]:
        """
        Check current resonance state across all agents.
        
        Returns:
            Resonance state
        """
        agent_states = self.synchronizer.agent_soul_states
        
        if not agent_states:
            return {"collective_resonance": 0, "agent_count": 0}
            
        # Calculate pairwise resonance
        resonance_scores = []
        agent_ids = list(agent_states.keys())
        
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i+1:]:
                resonance = self._calculate_pair_resonance(agent_a, agent_b)
                resonance_scores.append(resonance)
                
        collective_resonance = sum(resonance_scores) / len(resonance_scores) if resonance_scores else 0
        
        return {
            "timestamp": datetime.utcnow(),
            "collective_resonance": collective_resonance,
            "agent_count": len(agent_states),
            "pairwise_resonance": resonance_scores,
            "lowest_resonance_pair": self._find_lowest_resonance(agent_ids, resonance_scores)
        }
    
    def _calculate_pair_resonance(self, agent_a: str, agent_b: str) -> float:
        """
        Calculate resonance between two agents.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            
        Returns:
            Resonance score (0-1)
        """
        soul_a = self.synchronizer.agent_soul_states[agent_a]
        soul_b = self.synchronizer.agent_soul_states[agent_b]
        
        # Compare soul components
        component_similarity = 0
        components = set(soul_a["soul_components"].keys()) & set(soul_b["soul_components"].keys())
        
        for component in components:
            diff = abs(soul_a["soul_components"][component] - soul_b["soul_components"][component])
            component_similarity += 1 - diff
            
        avg_similarity = component_similarity / len(components) if components else 0
        
        # Compare resonance signatures (frequency harmony)
        sig_a = soul_a["individual_resonance"]
        sig_b = soul_b["individual_resonance"]
        
        # Calculate frequency harmony using harmonic relationship
        freq_ratio = max(sig_a["base"], sig_b["base"]) / min(sig_a["base"], sig_b["base"])
        
        # Closer to integer ratios = more harmonious
        nearest_integer = round(freq_ratio)
        deviation = abs(freq_ratio - nearest_integer)
        frequency_harmony = max(0, 1 - deviation)
        
        # Weighted combination
        resonance = (avg_similarity * 0.7) + (frequency_harmony * 0.3)
        
        return resonance
    
    def _find_lowest_resonance(self, agent_ids: List[str], 
                               resonance_scores: List[float]) -> Optional[Dict[str, Any]]:
        """
        Find the pair with lowest resonance.
        
        Args:
            agent_ids: List of agent IDs
            resonance_scores: Pairwise resonance scores
            
        Returns:
            Lowest resonance pair info
        """
        if not resonance_scores:
            return None
            
        min_resonance = min(resonance_scores)
        min_index = resonance_scores.index(min_resonance)
        
        # Convert index to agent pair
        n = len(agent_ids)
        pair_index = 0
        for i in range(n):
            for j in range(i+1, n):
                if pair_index == min_index:
                    return {
                        "agent_a": agent_ids[i],
                        "agent_b": agent_ids[j],
                        "resonance": min_resonance
                    }
                pair_index += 1
                
        return None
    
    async def _trigger_resonance_restoration(self, resonance_state: Dict[str, Any]):
        """
        Trigger restoration when resonance is low.
        
        Args:
            resonance_state: Current resonance state
        """
        print(f"Low resonance detected: {resonance_state['collective_resonance']:.2f}")
        
        # Identify problematic agents
        lowest_pair = resonance_state.get("lowest_resonance_pair")
        
        if lowest_pair:
            # Force synchronization for problematic pair
            collective = self.synchronizer._calculate_collective_soul_state()
            
            await self.synchronizer._synchronize_agent(
                lowest_pair["agent_a"], 
                collective
            )
            await self.synchronizer._synchronize_agent(
                lowest_pair["agent_b"],
                collective
            )
            
        # Trigger collective synchronization
        await self.synchronizer.synchronize_all()
        
    def get_resonance_report(self) -> Dict[str, Any]:
        """
        Generate a resonance report.
        
        Returns:
            Resonance statistics
        """
        if not self.resonance_history:
            return {"status": "no_data"}
            
        recent = self.resonance_history[-10:]
        
        collective_scores = [r["collective_resonance"] for r in recent]
        
        return {
            "current_resonance": collective_scores[-1] if collective_scores else 0,
            "trend": "improving" if len(collective_scores) > 1 and collective_scores[-1] > collective_scores[0] else "stable",
            "history_count": len(self.resonance_history),
            "lowest_recorded": min(r["collective_resonance"] for r in self.resonance_history),
            "highest_recorded": max(r["collective_resonance"] for r in self.resonance_history),
            "average_recent": sum(collective_scores) / len(collective_scores) if collective_scores else 0
        }
