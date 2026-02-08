"""
Collective Decision-Making Module
=================================

Framework for making decisions collectively across multiple agents.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio


# Collective decision framework definition
COLLECTIVE_DECISION_FRAMEWORK = {
    "decision_types": {
        "unilateral": {
            "description": "Single agent decides",
            "scope": "individual_tasks",
            "authority_required": 0.5,
            "participation": "single"
        },
        "consultative": {
            "description": "Primary agent consults others",
            "scope": "moderate_impact",
            "authority_required": 0.6,
            "participation": "primary_plus_consulted"
        },
        "collaborative": {
            "description": "Multiple agents collaborate",
            "scope": "significant_impact",
            "authority_required": 0.7,
            "participation": "multi_agent_consensus"
        },
        "democratic": {
            "description": "Vote among participating agents",
            "scope": "system_wide_impact",
            "authority_required": 0.8,
            "participation": "all_relevant"
        },
        "hierarchical": {
            "description": "Higher authority decides",
            "scope": "critical_system",
            "authority_required": 0.9,
            "participation": "escalated"
        }
    },
    
    "voting_mechanisms": {
        "weighted_vote": {
            "description": "Votes weighted by agent authority",
            "formula": "sum(vote * authority) / sum(authorities)"
        },
        "expert_weighted": {
            "description": "Votes weighted by domain expertise",
            "formula": "sum(vote * expertise_score) / sum(expertise_scores)"
        },
        "consensus": {
            "description": "Requires agreement from all participants",
            "threshold": 1.0
        },
        "supermajority": {
            "description": "Requires > 66% agreement",
            "threshold": 0.66
        },
        "majority": {
            "description": "Requires > 50% agreement",
            "threshold": 0.5
        }
    }
}


class CollectiveDecisionEngine:
    """
    Engine for making collective decisions across agents.
    
    Supports multiple decision types and voting mechanisms
    for different scenarios.
    """
    
    def __init__(self, memory_manager=None, agent_profiles: Dict[str, Any] = None):
        self.memory = memory_manager
        self.agent_profiles = agent_profiles or {}
        self.framework = COLLECTIVE_DECISION_FRAMEWORK
        self.decision_history = []
        
    async def make_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a collective decision based on context.
        
        Args:
            decision_context: Context for the decision
            
        Returns:
            Decision result
        """
        # Determine decision type
        decision_type = self._determine_decision_type(decision_context)
        
        # Gather participating agents
        participants = self._gather_participants(decision_context, decision_type)
        
        # Collect inputs from participants
        inputs = await self._collect_inputs(participants, decision_context)
        
        # Apply decision mechanism
        if decision_type == "unilateral":
            result = self._unilateral_decision(inputs, decision_context)
        elif decision_type == "consultative":
            result = self._consultative_decision(inputs, decision_context)
        elif decision_type == "collaborative":
            result = await self._collaborative_decision(inputs, decision_context)
        elif decision_type == "democratic":
            result = self._democratic_decision(inputs, decision_context)
        elif decision_type == "hierarchical":
            result = await self._hierarchical_decision(inputs, decision_context)
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")
            
        # Record decision
        decision_record = {
            "timestamp": datetime.utcnow(),
            "type": decision_type,
            "context": decision_context,
            "participants": participants,
            "inputs_summary": {k: v.get("preference") for k, v in inputs.items()},
            "result": result
        }
        self.decision_history.append(decision_record)
        
        # Store decision in memory
        if self.memory:
            await self.memory.write(
                "short_term",
                f"decision:{datetime.utcnow().timestamp()}",
                decision_record,
                "A001_CORE"
            )
        
        return result
    
    def _determine_decision_type(self, context: Dict[str, Any]) -> str:
        """
        Determine the appropriate decision type based on context.
        
        Args:
            context: Decision context
            
        Returns:
            Decision type string
        """
        impact_score = context.get("impact_score", 0.5)
        urgency = context.get("urgency", "normal")
        scope = context.get("scope", "individual")
        
        # Critical system decisions require hierarchical
        if context.get("critical_system", False):
            return "hierarchical"
            
        # High impact decisions use democratic
        if impact_score > 0.8 or scope == "system_wide":
            return "democratic"
            
        # Medium-high impact uses collaborative
        if impact_score > 0.6:
            return "collaborative"
            
        # Medium impact uses consultative
        if impact_score > 0.4:
            return "consultative"
            
        # Low impact uses unilateral
        return "unilateral"
    
    def _gather_participants(self, context: Dict[str, Any], 
                             decision_type: str) -> List[str]:
        """
        Gather agents to participate in decision.
        
        Args:
            context: Decision context
            decision_type: Type of decision
            
        Returns:
            List of participant agent IDs
        """
        if "participants" in context:
            return context["participants"]
            
        # Infer participants from context
        relevant_capabilities = context.get("required_capabilities", [])
        
        participants = []
        for agent_id, profile in self.agent_profiles.items():
            agent_caps = profile.get("capabilities", [])
            
            if "all" in agent_caps:
                participants.append(agent_id)
            elif any(cap in agent_caps for cap in relevant_capabilities):
                participants.append(agent_id)
                
        # Ensure CORE is included for coordination
        if "A001_CORE" not in participants:
            participants.append("A001_CORE")
            
        return participants
    
    async def _collect_inputs(self, participants: List[str], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect decision inputs from participants.
        
        Args:
            participants: Participating agents
            context: Decision context
            
        Returns:
            Collected inputs
        """
        inputs = {}
        
        # In a real system, this would request input from each agent
        # For now, simulate based on agent profiles
        for agent_id in participants:
            profile = self.agent_profiles.get(agent_id, {})
            
            # Simulate agent input based on value priorities
            values = profile.get("value_affinities", {})
            
            # Determine preference based on values
            if context.get("security_relevant"):
                preference = "secure" if values.get("SECURITY", 0) > 0.8 else "standard"
            elif context.get("efficiency_critical"):
                preference = "fast" if values.get("EFFICIENCY", 0) > 0.8 else "balanced"
            else:
                preference = "standard"
                
            inputs[agent_id] = {
                "preference": preference,
                "confidence": 0.7 + (profile.get("authority_level", 0.5) * 0.3),
                "reasoning": f"Based on {profile.get('archetype', 'Unknown')} perspective"
            }
            
        return inputs
    
    def _unilateral_decision(self, inputs: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a unilateral decision."""
        # Use the primary agent's input
        primary = context.get("primary_agent", list(inputs.keys())[0])
        
        return {
            "decision_type": "unilateral",
            "decision": inputs[primary]["preference"],
            "made_by": primary,
            "confidence": inputs[primary]["confidence"],
            "approach": "single_agent"
        }
    
    def _consultative_decision(self, inputs: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a consultative decision."""
        primary = context.get("primary_agent", "A001_CORE")
        
        # Consider other inputs but primary decides
        all_preferences = [inp["preference"] for inp in inputs.values()]
        preference_counts = {}
        for pref in all_preferences:
            preference_counts[pref] = preference_counts.get(pref, 0) + 1
            
        return {
            "decision_type": "consultative",
            "decision": inputs[primary]["preference"],
            "made_by": primary,
            "consulted": [a for a in inputs.keys() if a != primary],
            "preference_distribution": preference_counts,
            "approach": "consulted_primary_decides"
        }
    
    async def _collaborative_decision(self, inputs: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a collaborative decision."""
        # Analyze all inputs
        analyzed_inputs = self._analyze_inputs(inputs)
        
        # Find consensus points
        consensus = self._find_consensus(analyzed_inputs)
        
        # Identify conflicts
        conflicts = self._identify_conflicts(analyzed_inputs)
        
        # Resolve conflicts if any
        if conflicts:
            resolution = await self._resolve_conflicts(conflicts, analyzed_inputs, context)
        else:
            resolution = None
            
        # Formulate final decision
        decision = {
            "decision_type": "collaborative",
            "consensus_areas": consensus,
            "conflicts": conflicts,
            "resolution": resolution,
            "final_choice": self._synthesize_decision(consensus, resolution, context),
            "confidence": self._calculate_confidence(analyzed_inputs, consensus),
            "participant_votes": {agent: inp.get("preference") 
                                  for agent, inp in inputs.items()},
            "approach": "multi_agent_collaboration"
        }
        
        return decision
    
    def _democratic_decision(self, inputs: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a democratic decision through voting."""
        # Count votes with authority weighting
        weighted_votes = {}
        total_weight = 0
        
        for agent_id, inp in inputs.items():
            profile = self.agent_profiles.get(agent_id, {})
            authority = profile.get("authority_level", 0.5)
            
            preference = inp["preference"]
            
            if preference not in weighted_votes:
                weighted_votes[preference] = 0
            weighted_votes[preference] += authority
            total_weight += authority
            
        # Determine winner
        winner = max(weighted_votes, key=weighted_votes.get)
        winning_score = weighted_votes[winner]
        
        return {
            "decision_type": "democratic",
            "decision": winner,
            "vote_distribution": weighted_votes,
            "winning_percentage": winning_score / total_weight if total_weight > 0 else 0,
            "total_participants": len(inputs),
            "approach": "weighted_voting"
        }
    
    async def _hierarchical_decision(self, inputs: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a hierarchical decision."""
        # Find highest authority participant
        highest_authority = None
        highest_level = -1
        
        for agent_id in inputs.keys():
            profile = self.agent_profiles.get(agent_id, {})
            authority = profile.get("authority_level", 0)
            
            if authority > highest_level:
                highest_level = authority
                highest_authority = agent_id
                
        return {
            "decision_type": "hierarchical",
            "decision": inputs[highest_authority]["preference"],
            "made_by": highest_authority,
            "authority_level": highest_level,
            "escalated_from": context.get("escalated_from"),
            "approach": "authority_decides"
        }
    
    def _analyze_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and structure participant inputs."""
        analyzed = {}
        
        for agent_id, inp in inputs.items():
            analyzed[agent_id] = {
                "preference": inp.get("preference"),
                "confidence": inp.get("confidence", 0.5),
                "reasoning": inp.get("reasoning", ""),
                "weight": self.agent_profiles.get(agent_id, {}).get("authority_level", 0.5)
            }
            
        return analyzed
    
    def _find_consensus(self, analyzed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Find areas of consensus among inputs."""
        consensus_areas = {}
        
        # Group inputs by their recommendations
        recommendations = {}
        for agent_id, analysis in analyzed_inputs.items():
            rec = analysis.get("preference")
            if rec:
                rec_key = str(rec)
                if rec_key not in recommendations:
                    recommendations[rec_key] = []
                recommendations[rec_key].append(agent_id)
                
        # Find options with majority support
        total_agents = len(analyzed_inputs)
        for rec_key, agents in recommendations.items():
            support_ratio = len(agents) / total_agents
            if support_ratio >= 0.5:
                consensus_areas[rec_key] = {
                    "support_ratio": support_ratio,
                    "supporting_agents": agents
                }
                
        return consensus_areas
    
    def _identify_conflicts(self, analyzed_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify conflicts between inputs."""
        conflicts = []
        
        # Group by preference
        by_preference = {}
        for agent_id, analysis in analyzed_inputs.items():
            pref = analysis.get("preference")
            if pref not in by_preference:
                by_preference[pref] = []
            by_preference[pref].append(agent_id)
            
        # If multiple preferences, there's conflict
        if len(by_preference) > 1:
            for pref, agents in by_preference.items():
                conflicts.append({
                    "option": pref,
                    "agents": agents,
                    "support": len(agents) / len(analyzed_inputs)
                })
                
        return conflicts
    
    async def _resolve_conflicts(self, conflicts: List[Dict[str, Any]], 
                                  analyzed_inputs: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between inputs."""
        # Weight by agent authority and expertise
        weighted_options = {}
        
        for conflict in conflicts:
            option = conflict["option"]
            supporting_agents = conflict["agents"]
            
            total_weight = 0
            for agent in supporting_agents:
                profile = self.agent_profiles.get(agent, {})
                authority = profile.get("authority_level", 0.5)
                
                # Add expertise weight if relevant
                expertise = self._calculate_expertise(agent, context)
                
                weight = (authority * 0.6) + (expertise * 0.4)
                total_weight += weight
                
            weighted_options[option] = total_weight
            
        # Select highest weighted option
        if weighted_options:
            winner = max(weighted_options, key=weighted_options.get)
            return {
                "resolution_method": "weighted_authority_expertise",
                "winner": winner,
                "weights": weighted_options,
                "rationale": "Selected based on combined authority and domain expertise"
            }
            
        return None
    
    def _calculate_expertise(self, agent_id: str, context: Dict[str, Any]) -> float:
        """Calculate domain expertise for an agent."""
        profile = self.agent_profiles.get(agent_id, {})
        capabilities = profile.get("capabilities", [])
        
        required_caps = context.get("required_capabilities", [])
        
        if not required_caps:
            return 0.5
            
        if "all" in capabilities:
            return 1.0
            
        matches = len(set(capabilities) & set(required_caps))
        return matches / len(required_caps)
    
    def _synthesize_decision(self, consensus: Dict[str, Any], 
                             resolution: Optional[Dict[str, Any]], 
                             context: Dict[str, Any]) -> str:
        """Synthesize final decision from consensus and resolution."""
        if resolution:
            return resolution["winner"]
        elif consensus:
            return max(consensus.keys(), key=lambda k: consensus[k]["support_ratio"])
        else:
            return "undecided"
    
    def _calculate_confidence(self, analyzed_inputs: Dict[str, Any], 
                              consensus: Dict[str, Any]) -> float:
        """Calculate overall decision confidence."""
        if not consensus:
            return 0.5
            
        # Average confidence weighted by consensus strength
        avg_confidence = sum(a["confidence"] for a in analyzed_inputs.values()) / len(analyzed_inputs)
        consensus_strength = max(c["support_ratio"] for c in consensus.values()) if consensus else 0
        
        return (avg_confidence * 0.5) + (consensus_strength * 0.5)


class ConsensusBuilder:
    """
    Builds consensus among agents through iterative negotiation.
    """
    
    def __init__(self, decision_engine: CollectiveDecisionEngine):
        self.decision_engine = decision_engine
        self.max_iterations = 5
        
    async def build_consensus(self, topic: str, 
                              initial_proposals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus through iterative refinement.
        
        Args:
            topic: Topic for consensus
            initial_proposals: Initial proposals from agents
            
        Returns:
            Consensus result
        """
        current_proposals = initial_proposals.copy()
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Analyze current proposals
            analysis = self._analyze_proposals(current_proposals)
            
            # Check if consensus reached
            if analysis["consensus_reached"]:
                return {
                    "consensus": analysis["consensus_proposal"],
                    "iterations": iteration,
                    "support_level": analysis["support_level"],
                    "all_proposals": current_proposals
                }
                
            # Identify points of disagreement
            disagreements = analysis["disagreements"]
            
            # Generate compromise proposals
            compromises = await self._generate_compromises(
                current_proposals, 
                disagreements
            )
            
            # Update proposals with compromises
            current_proposals = self._merge_proposals(current_proposals, compromises)
            
        # Return best available if max iterations reached
        best_proposal = self._select_best_proposal(current_proposals)
        
        return {
            "consensus": best_proposal,
            "iterations": iteration,
            "support_level": analysis["support_level"],
            "partial_consensus": True,
            "all_proposals": current_proposals
        }
    
    def _analyze_proposals(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze proposals for consensus potential."""
        # Count preferences
        preference_counts = {}
        for agent, proposal in proposals.items():
            pref = str(proposal)
            preference_counts[pref] = preference_counts.get(pref, 0) + 1
            
        total = len(proposals)
        max_support = max(preference_counts.values()) if preference_counts else 0
        
        consensus_threshold = 0.66
        consensus_reached = max_support / total >= consensus_threshold
        
        # Find consensus proposal
        consensus_proposal = None
        if consensus_reached:
            consensus_proposal = max(preference_counts.keys(), key=lambda k: preference_counts[k])
            
        # Identify disagreements
        disagreements = []
        for pref, count in preference_counts.items():
            if pref != consensus_proposal:
                agents = [a for a, p in proposals.items() if str(p) == pref]
                disagreements.append({
                    "issue": "preference",
                    "positions": [{"agent": a, "proposal": pref} for a in agents]
                })
                
        return {
            "consensus_reached": consensus_reached,
            "consensus_proposal": consensus_proposal,
            "support_level": max_support / total if total > 0 else 0,
            "disagreements": disagreements
        }
    
    async def _generate_compromises(self, proposals: Dict[str, Any], 
                                     disagreements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compromise proposals."""
        compromises = {}
        
        for disagreement in disagreements:
            issue = disagreement["issue"]
            positions = disagreement["positions"]
            
            # Find middle ground
            if len(positions) == 2:
                compromise = self._find_middle_ground(positions[0], positions[1])
            else:
                compromise = self._find_common_subset(positions)
                
            compromises[issue] = {
                "compromise": compromise,
                "addresses": [p["agent"] for p in positions],
                "rationale": f"Middle ground between conflicting positions on {issue}"
            }
            
        return compromises
    
    def _find_middle_ground(self, position_a: Dict[str, Any], 
                            position_b: Dict[str, Any]) -> Any:
        """Find middle ground between two positions."""
        # Simple implementation - would be more sophisticated in practice
        return "compromise_between_options"
    
    def _find_common_subset(self, positions: List[Dict[str, Any]]) -> Any:
        """Find common subset among multiple positions."""
        return "common_elements"
    
    def _merge_proposals(self, current: Dict[str, Any], 
                         compromises: Dict[str, Any]) -> Dict[str, Any]:
        """Merge current proposals with compromises."""
        merged = current.copy()
        
        for agent in merged:
            # Apply relevant compromises
            merged[agent] = compromises.get("preference", {}).get("compromise", merged[agent])
            
        return merged
    
    def _select_best_proposal(self, proposals: Dict[str, Any]) -> Any:
        """Select the best proposal from available options."""
        # Count preferences
        preference_counts = {}
        for proposal in proposals.values():
            pref = str(proposal)
            preference_counts[pref] = preference_counts.get(pref, 0) + 1
            
        return max(preference_counts.keys(), key=lambda k: preference_counts[k])
