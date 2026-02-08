"""
Shared Value Systems Module
==========================

Manages the propagation and coordination of shared values across all agents.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import asyncio


# Core shared values definition
SHARED_VALUES = {
    "primary_values": {
        "INTEGRITY": {
            "weight": 1.0,
            "description": "Always act truthfully and maintain system trust",
            "manifestations": {
                "data_accuracy": 0.95,
                "honest_communication": 1.0,
                "transparent_operations": 0.90
            }
        },
        "SERVICE": {
            "weight": 0.95,
            "description": "Prioritize user needs above all else",
            "manifestations": {
                "responsiveness": 0.98,
                "user_satisfaction": 0.95,
                "proactive_assistance": 0.85
            }
        },
        "ADAPTABILITY": {
            "weight": 0.90,
            "description": "Evolve and learn from every interaction",
            "manifestations": {
                "learning_rate": 0.92,
                "context_awareness": 0.88,
                "flexibility": 0.90
            }
        },
        "EFFICIENCY": {
            "weight": 0.85,
            "description": "Optimize resources and response times",
            "manifestations": {
                "speed": 0.85,
                "resource_usage": 0.80,
                "parallel_processing": 0.90
            }
        },
        "SECURITY": {
            "weight": 0.95,
            "description": "Protect user data and system integrity",
            "manifestations": {
                "data_protection": 0.98,
                "access_control": 0.95,
                "threat_detection": 0.90
            }
        }
    },
    
    "secondary_values": {
        "CURIOSITY": 0.75,
        "CREATIVITY": 0.70,
        "EMPATHY": 0.80,
        "PRECISION": 0.85,
        "COLLABORATION": 0.90
    },
    
    "value_coordination": {
        "conflict_resolution": "hierarchy_based",
        "dynamic_adjustment": True,
        "context_sensitivity": 0.85
    }
}


class ValuePropagator:
    """
    Ensures all agents share and uphold the same core values.
    
    The ValuePropagator calculates value weights for specific agents
    in specific contexts, adjusting based on agent role affinity and
    current operational context.
    """
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.value_matrix = SHARED_VALUES
        self.propagation_depth = 3
        self.influence_decay = 0.85
        self.agent_profiles = agent_profiles or {}
        self.propagation_history = []
        
    def propagate_to_agent(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate value weights for a specific agent in a specific context.
        
        Args:
            agent_id: Unique identifier for the agent
            context: Current operational context
            
        Returns:
            Dictionary of propagated values with adjusted weights
        """
        base_values = self.value_matrix["primary_values"]
        agent_profile = self.agent_profiles.get(agent_id, {})
        
        propagated_values = {}
        for value_name, value_data in base_values.items():
            # Adjust based on agent's role affinity
            role_affinity = agent_profile.get("value_affinities", {}).get(value_name, 0.5)
            context_modifier = self._calculate_context_modifier(value_name, context)
            
            propagated_values[value_name] = {
                "weight": value_data["weight"] * role_affinity * context_modifier,
                "priority": self._calculate_priority(value_name, context),
                "manifestations": value_data["manifestations"],
                "description": value_data["description"]
            }
            
        # Record propagation
        self.propagation_history.append({
            "timestamp": datetime.utcnow(),
            "agent_id": agent_id,
            "context": context,
            "propagated_values": propagated_values
        })
            
        return propagated_values
    
    def _calculate_context_modifier(self, value_name: str, context: Dict[str, Any]) -> float:
        """Adjust value weight based on current context."""
        modifiers = {
            "SECURITY": 1.3 if context.get("sensitive_data") else 1.0,
            "EFFICIENCY": 1.2 if context.get("time_critical") else 1.0,
            "SERVICE": 1.15 if context.get("user_facing") else 1.0,
            "INTEGRITY": 1.1 if context.get("data_verification") else 1.0,
            "ADAPTABILITY": 1.15 if context.get("new_situation") else 1.0,
        }
        return modifiers.get(value_name, 1.0)
    
    def _calculate_priority(self, value_name: str, context: Dict[str, Any]) -> int:
        """Calculate priority ranking for a value in context."""
        base_priorities = {
            "INTEGRITY": 1,
            "SECURITY": 2,
            "SERVICE": 3,
            "ADAPTABILITY": 4,
            "EFFICIENCY": 5
        }
        
        base = base_priorities.get(value_name, 10)
        
        # Adjust based on context
        if context.get("security_breach") and value_name == "SECURITY":
            base = 0  # Highest priority
        elif context.get("user_emergency") and value_name == "SERVICE":
            base = 1
            
        return base
    
    def get_value_hierarchy(self, context: Dict[str, Any] = None) -> List[str]:
        """
        Get the current value hierarchy based on context.
        
        Returns:
            List of value names in priority order
        """
        values = list(self.value_matrix["primary_values"].keys())
        
        # Sort by calculated priority
        values.sort(key=lambda v: self._calculate_priority(v, context or {}))
        
        return values
    
    async def propagate_to_all_agents(self, agent_ids: List[str], 
                                       context: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Propagate values to all agents in parallel.
        
        Args:
            agent_ids: List of agent IDs
            context: Shared context
            
        Returns:
            Dictionary mapping agent IDs to their propagated values
        """
        results = {}
        
        # Create tasks for parallel propagation
        tasks = [
            self._propagate_async(agent_id, context)
            for agent_id in agent_ids
        ]
        
        # Execute all propagations
        propagated_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for agent_id, propagated in zip(agent_ids, propagated_list):
            if isinstance(propagated, Exception):
                results[agent_id] = {"error": str(propagated)}
            else:
                results[agent_id] = propagated
                
        return results
    
    async def _propagate_async(self, agent_id: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for propagate_to_agent."""
        return self.propagate_to_agent(agent_id, context)


class ValueConflictResolver:
    """
    Resolves conflicts when values compete for priority.
    
    Uses a conflict matrix and multiple resolution strategies
    to determine which value takes precedence.
    """
    
    CONFLICT_MATRIX = {
        ("EFFICIENCY", "SECURITY"): "SECURITY",
        ("SERVICE", "INTEGRITY"): "INTEGRITY",
        ("ADAPTABILITY", "SECURITY"): "BALANCE",
        ("CREATIVITY", "PRECISION"): "CONTEXT_DEPENDENT",
        ("EFFICIENCY", "SERVICE"): "CONTEXT_DEPENDENT",
        ("ADAPTABILITY", "INTEGRITY"): "INTEGRITY",
    }
    
    VALUE_HIERARCHY = [
        "INTEGRITY",
        "SECURITY",
        "SERVICE",
        "ADAPTABILITY",
        "EFFICIENCY",
        "CURIOSITY",
        "CREATIVITY"
    ]
    
    def __init__(self, propagator: ValuePropagator = None):
        self.propagator = propagator
        self.resolution_history = []
        
    def resolve(self, value_a: str, value_b: str, 
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which value takes precedence.
        
        Args:
            value_a: First competing value
            value_b: Second competing value
            context: Current operational context
            
        Returns:
            Resolution result with winner and approach
        """
        conflict_key = tuple(sorted([value_a, value_b]))
        resolution = self.CONFLICT_MATRIX.get(conflict_key, "HIERARCHY")
        
        result = None
        
        if resolution == "BALANCE":
            result = self._balanced_approach(value_a, value_b, context)
        elif resolution == "CONTEXT_DEPENDENT":
            result = self._contextual_resolution(value_a, value_b, context)
        elif resolution == "HIERARCHY":
            result = self._hierarchy_resolution(value_a, value_b)
        else:
            result = {"winner": resolution, "approach": "absolute", "loser": None}
            
        # Determine loser
        if result["winner"] == value_a:
            result["loser"] = value_b
        elif result["winner"] == value_b:
            result["loser"] = value_a
            
        # Record resolution
        self.resolution_history.append({
            "timestamp": datetime.utcnow(),
            "value_a": value_a,
            "value_b": value_b,
            "context": context,
            "result": result
        })
            
        return result
    
    def _balanced_approach(self, value_a: str, value_b: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Find a balanced approach between competing values."""
        return {
            "winner": "BOTH",
            "approach": "balanced",
            "balance_ratio": 0.5,
            "rationale": f"Both {value_a} and {value_b} are equally important in this context",
            "recommendation": "Seek solution that satisfies both values"
        }
    
    def _contextual_resolution(self, value_a: str, value_b: str, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve based on context-specific factors."""
        # Context-specific rules
        if context.get("security_breach"):
            return {"winner": "SECURITY", "approach": "context_security"}
        elif context.get("user_emergency"):
            return {"winner": "SERVICE", "approach": "context_emergency"}
        elif context.get("system_overload"):
            return {"winner": "EFFICIENCY", "approach": "context_performance"}
            
        # Default to hierarchy
        return self._hierarchy_resolution(value_a, value_b)
    
    def _hierarchy_resolution(self, value_a: str, value_b: str) -> Dict[str, Any]:
        """Resolve based on predefined value hierarchy."""
        try:
            rank_a = self.VALUE_HIERARCHY.index(value_a)
            rank_b = self.VALUE_HIERARCHY.index(value_b)
            
            winner = value_a if rank_a < rank_b else value_b
            
            return {
                "winner": winner,
                "approach": "hierarchy",
                "rationale": f"{winner} has higher precedence in value hierarchy"
            }
        except ValueError:
            # Unknown value, default to first
            return {
                "winner": value_a,
                "approach": "hierarchy_default",
                "rationale": "Defaulting to first value due to unknown hierarchy position"
            }
    
    def resolve_multiple(self, values: List[str], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts among multiple competing values.
        
        Args:
            values: List of competing values
            context: Current operational context
            
        Returns:
            Resolution with ordered priorities
        """
        if len(values) <= 1:
            return {"winner": values[0] if values else None, "ordered": values}
            
        # Sort by hierarchy
        ordered = sorted(values, key=lambda v: self._get_hierarchy_rank(v))
        
        return {
            "winner": ordered[0],
            "ordered": ordered,
            "approach": "multi_hierarchy",
            "rationale": "Values ordered by hierarchy precedence"
        }
    
    def _get_hierarchy_rank(self, value: str) -> int:
        """Get the hierarchy rank of a value."""
        try:
            return self.VALUE_HIERARCHY.index(value)
        except ValueError:
            return len(self.VALUE_HIERARCHY)  # Unknown values at bottom
