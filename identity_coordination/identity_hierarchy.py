"""
Identity Hierarchy Module
========================

Manages and enforces the identity hierarchy across agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class IdentityHierarchy:
    """
    Manages and enforces the identity hierarchy.
    
    Defines hierarchical levels, authority relationships, and
    escalation paths for decision-making.
    """
    
    HIERARCHY_LEVELS = {
        0: {
            "name": "META",
            "agents": ["SYSTEM"],
            "scope": "universal",
            "override_capability": True,
            "description": "System-wide identity core - shared values and universal beliefs"
        },
        1: {
            "name": "PRIME",
            "agents": ["A001_CORE"],
            "scope": "global",
            "override_capability": True,
            "description": "Core orchestration - central coordination and control"
        },
        2: {
            "name": "GUARDIAN",
            "agents": ["A010_SECURITY", "A006_SYSTEM", "A014_HEARTBEAT"],
            "scope": "system_wide",
            "override_capability": True,
            "description": "System protection and integrity monitoring"
        },
        3: {
            "name": "SPECIALIST",
            "agents": [
                "A002_BROWSER", "A003_EMAIL", "A004_VOICE", "A005_PHONE",
                "A007_FILE", "A008_SCHEDULER", "A009_MEMORY",
                "A011_LEARNER", "A012_CREATIVE", "A013_ANALYZER", "A015_USER"
            ],
            "scope": "domain_specific",
            "override_capability": False,
            "description": "Domain-specific operations and specialized tasks"
        }
    }
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.levels = self.HIERARCHY_LEVELS
        self.agent_profiles = agent_profiles or {}
        self.authority_log = []
        
    def get_agent_level(self, agent_id: str) -> int:
        """
        Get the hierarchy level of an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Hierarchy level (0-3)
        """
        for level, data in self.levels.items():
            if agent_id in data["agents"]:
                return level
        return 3  # Default to specialist level
    
    def check_authority(self, requesting_agent: str, 
                        target_agent: str) -> Dict[str, Any]:
        """
        Check if requesting agent has authority over target agent.
        
        Args:
            requesting_agent: Agent requesting authority
            target_agent: Target of the authority check
            
        Returns:
            Authority check result
        """
        requester_level = self.get_agent_level(requesting_agent)
        target_level = self.get_agent_level(target_agent)
        
        result = None
        
        if requester_level < target_level:
            result = {
                "authorized": True,
                "authority_type": "hierarchical_override",
                "override_strength": 1.0 - (target_level - requester_level) * 0.2,
                "requester_level": requester_level,
                "target_level": target_level
            }
        elif requester_level == target_level:
            result = {
                "authorized": True,
                "authority_type": "peer_coordination",
                "override_strength": 0.5,
                "requester_level": requester_level,
                "target_level": target_level
            }
        else:
            result = {
                "authorized": False,
                "authority_type": "insufficient_authority",
                "override_strength": 0.0,
                "requester_level": requester_level,
                "target_level": target_level
            }
            
        # Log authority check
        self.authority_log.append({
            "timestamp": datetime.utcnow(),
            "requester": requesting_agent,
            "target": target_agent,
            "result": result
        })
        
        return result
    
    def escalate_decision(self, agent_id: str, 
                          decision_context: Dict[str, Any]) -> str:
        """
        Escalate a decision to the next higher authority level.
        
        Args:
            agent_id: Agent requesting escalation
            decision_context: Context of the decision
            
        Returns:
            ID of agent to escalate to
        """
        current_level = self.get_agent_level(agent_id)
        
        if current_level <= 1:
            return "A001_CORE"  # Already at or near top
            
        # Find appropriate higher authority
        higher_level = current_level - 1
        higher_agents = self.levels[higher_level]["agents"]
        
        # Return first available agent at higher level
        # In a real system, this would check availability
        return higher_agents[0]
    
    def get_higher_authorities(self, agent_id: str) -> List[str]:
        """
        Get all agents at higher authority levels.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of higher authority agent IDs
        """
        current_level = self.get_agent_level(agent_id)
        higher = []
        
        for level in range(current_level):
            higher.extend(self.levels[level]["agents"])
            
        return higher
    
    def get_peers(self, agent_id: str) -> List[str]:
        """
        Get all agents at the same authority level.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of peer agent IDs
        """
        current_level = self.get_agent_level(agent_id)
        peers = self.levels[current_level]["agents"].copy()
        
        if agent_id in peers:
            peers.remove(agent_id)
            
        return peers
    
    def get_subordinates(self, agent_id: str) -> List[str]:
        """
        Get all agents at lower authority levels.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of subordinate agent IDs
        """
        current_level = self.get_agent_level(agent_id)
        subordinates = []
        
        for level in range(current_level + 1, max(self.levels.keys()) + 1):
            subordinates.extend(self.levels[level]["agents"])
            
        return subordinates
    
    def can_override(self, agent_id: str) -> bool:
        """
        Check if an agent has override capability.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent can override decisions
        """
        level = self.get_agent_level(agent_id)
        return self.levels[level]["override_capability"]
    
    def get_scope(self, agent_id: str) -> str:
        """
        Get the decision scope for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Decision scope string
        """
        level = self.get_agent_level(agent_id)
        return self.levels[level]["scope"]
    
    def get_hierarchy_report(self) -> Dict[str, Any]:
        """
        Generate a hierarchy report.
        
        Returns:
            Hierarchy statistics
        """
        return {
            "levels": len(self.levels),
            "total_agents": sum(len(l["agents"]) for l in self.levels.values()),
            "level_distribution": {
                level: len(data["agents"])
                for level, data in self.levels.items()
            },
            "override_capable_agents": [
                agent for level, data in self.levels.items()
                for agent in data["agents"]
                if data["override_capability"]
            ],
            "recent_authority_checks": len(self.authority_log[-100:])
        }


class CrossLevelProtocol:
    """
    Protocols for communication across hierarchy levels.
    
    Manages the formatting and handling of messages between
    different hierarchy levels.
    """
    
    def __init__(self, hierarchy: IdentityHierarchy = None):
        self.hierarchy = hierarchy or IdentityHierarchy()
        
    def upward_communication(self, from_agent: str, to_level: int, 
                             message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle communication from lower to higher level.
        
        Args:
            from_agent: Source agent
            to_level: Target hierarchy level
            message: Message content
            
        Returns:
            Formatted communication
        """
        from_level = self.hierarchy.get_agent_level(from_agent)
        
        protocol = {
            "message_type": "escalation" if message.get("urgent") else "report",
            "from_level": from_level,
            "to_level": to_level,
            "respect_indicators": True,
            "include_context": True,
            "request_acknowledgment": True,
            "escalation_reason": message.get("reason", "standard_escalation")
        }
        
        return {
            "protocol": protocol,
            "formatted_message": self._format_upward_message(message, protocol),
            "priority_boost": message.get("urgent", False),
            "requires_immediate_attention": message.get("critical", False)
        }
    
    def downward_communication(self, from_agent: str, to_agent: str, 
                               message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle communication from higher to lower level.
        
        Args:
            from_agent: Source (higher) agent
            to_agent: Target (lower) agent
            message: Message content
            
        Returns:
            Formatted communication
        """
        authority = self.hierarchy.check_authority(from_agent, to_agent)
        
        protocol = {
            "message_type": "directive" if authority["override_strength"] > 0.7 else "guidance",
            "authority_level": authority["authority_type"],
            "binding_force": authority["override_strength"],
            "expect_compliance": authority["authorized"],
            "can_be_questioned": authority["override_strength"] < 0.9
        }
        
        return {
            "protocol": protocol,
            "formatted_message": self._format_downward_message(message, protocol),
            "compliance_required": protocol["expect_compliance"],
            "response_expected": message.get("requires_response", False)
        }
    
    def peer_communication(self, agent_a: str, agent_b: str, 
                           message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle communication between agents at the same level.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            message: Message content
            
        Returns:
            Formatted communication
        """
        protocol = {
            "message_type": "collaboration",
            "relationship": "peer",
            "consensus_required": message.get("requires_agreement", True),
            "negotiation_allowed": True,
            "equal_standing": True
        }
        
        return {
            "protocol": protocol,
            "formatted_message": self._format_peer_message(message, protocol),
            "requires_coordination": True,
            "can_delegate": True
        }
    
    def _format_upward_message(self, message: Dict[str, Any], 
                               protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Format a message for upward communication."""
        return {
            "header": {
                "direction": "upward",
                "type": protocol["message_type"],
                "priority": "high" if protocol["priority_boost"] else "normal"
            },
            "body": message.get("content", message),
            "context": message.get("context", {}),
            "request": message.get("request", {})
        }
    
    def _format_downward_message(self, message: Dict[str, Any], 
                                 protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Format a message for downward communication."""
        return {
            "header": {
                "direction": "downward",
                "type": protocol["message_type"],
                "authority": protocol["authority_level"],
                "binding": protocol["binding_force"] > 0.7
            },
            "body": message.get("content", message),
            "instruction": message.get("instruction", {}),
            "deadline": message.get("deadline")
        }
    
    def _format_peer_message(self, message: Dict[str, Any], 
                             protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Format a message for peer communication."""
        return {
            "header": {
                "direction": "peer",
                "type": protocol["message_type"],
                "consensus_needed": protocol["consensus_required"]
            },
            "body": message.get("content", message),
            "proposal": message.get("proposal", {}),
            "alternatives": message.get("alternatives", [])
        }
