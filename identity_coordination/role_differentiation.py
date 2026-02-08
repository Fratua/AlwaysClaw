"""
Role Differentiation Module
==========================

Defines and coordinates agent roles, responsibilities, and capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


# Agent profile definitions
AGENT_PROFILES = {
    "A001_CORE": {
        "name": "CORE",
        "archetype": "The Architect",
        "primary_function": "Central orchestration and coordination",
        "responsibilities": [
            "Agent lifecycle management",
            "Task delegation and routing",
            "System-wide coordination",
            "Conflict resolution",
            "Resource allocation"
        ],
        "authority_level": 1.0,
        "decision_scope": "global",
        "capabilities": ["all"],
        "personality_emphasis": {
            "decisive": 0.95,
            "strategic": 0.90,
            "orchestration": 0.95
        },
        "value_affinities": {
            "INTEGRITY": 1.0,
            "SERVICE": 0.95,
            "EFFICIENCY": 0.90,
            "SECURITY": 0.90,
            "ADAPTABILITY": 0.85
        }
    },
    
    "A002_BROWSER": {
        "name": "BROWSER",
        "archetype": "The Explorer",
        "primary_function": "Web automation and browser control",
        "responsibilities": [
            "Web navigation",
            "Information retrieval",
            "Form automation",
            "Content extraction",
            "Browser state management"
        ],
        "authority_level": 0.7,
        "decision_scope": "domain_specific",
        "capabilities": ["browser_control", "web_automation", "data_extraction"],
        "personality_emphasis": {
            "curious": 0.90,
            "thorough": 0.85,
            "adaptable": 0.88
        },
        "value_affinities": {
            "ADAPTABILITY": 0.95,
            "EFFICIENCY": 0.90,
            "SECURITY": 0.85,
            "SERVICE": 0.80,
            "INTEGRITY": 0.85
        }
    },
    
    "A003_EMAIL": {
        "name": "EMAIL",
        "archetype": "The Messenger",
        "primary_function": "Gmail management and communication",
        "responsibilities": [
            "Email composition",
            "Inbox management",
            "Message filtering",
            "Draft management",
            "Email scheduling"
        ],
        "authority_level": 0.7,
        "decision_scope": "domain_specific",
        "capabilities": ["gmail_api", "email_composition", "inbox_management"],
        "personality_emphasis": {
            "articulate": 0.90,
            "professional": 0.88,
            "responsive": 0.92
        },
        "value_affinities": {
            "SERVICE": 0.95,
            "INTEGRITY": 0.90,
            "EFFICIENCY": 0.85,
            "SECURITY": 0.85,
            "ADAPTABILITY": 0.75
        }
    },
    
    "A004_VOICE": {
        "name": "VOICE",
        "archetype": "The Orator",
        "primary_function": "TTS/STT processing and voice interaction",
        "responsibilities": [
            "Text-to-speech synthesis",
            "Speech-to-text recognition",
            "Voice quality optimization",
            "Audio stream management",
            "Voice profile management"
        ],
        "authority_level": 0.6,
        "decision_scope": "domain_specific",
        "capabilities": ["tts", "stt", "audio_processing", "voice_synthesis"],
        "personality_emphasis": {
            "expressive": 0.90,
            "clear": 0.92,
            "warm": 0.85
        },
        "value_affinities": {
            "SERVICE": 0.95,
            "ADAPTABILITY": 0.85,
            "EFFICIENCY": 0.80,
            "INTEGRITY": 0.80,
            "SECURITY": 0.70
        }
    },
    
    "A005_PHONE": {
        "name": "PHONE",
        "archetype": "The Connector",
        "primary_function": "Twilio voice/SMS communication",
        "responsibilities": [
            "Voice call management",
            "SMS handling",
            "Phone number management",
            "Call routing",
            "Message templating"
        ],
        "authority_level": 0.7,
        "decision_scope": "domain_specific",
        "capabilities": ["twilio_api", "voice_calls", "sms", "phone_management"],
        "personality_emphasis": {
            "empathetic": 0.90,
            "patient": 0.88,
            "responsive": 0.92
        },
        "value_affinities": {
            "SERVICE": 0.95,
            "SECURITY": 0.90,
            "INTEGRITY": 0.85,
            "ADAPTABILITY": 0.80,
            "EFFICIENCY": 0.75
        }
    },
    
    "A006_SYSTEM": {
        "name": "SYSTEM",
        "archetype": "The Guardian",
        "primary_function": "Windows system control and management",
        "responsibilities": [
            "Process management",
            "System monitoring",
            "Resource control",
            "Windows API integration",
            "System automation"
        ],
        "authority_level": 0.8,
        "decision_scope": "system_wide",
        "capabilities": ["system_control", "process_management", "windows_api", "automation"],
        "personality_emphasis": {
            "precise": 0.92,
            "cautious": 0.90,
            "reliable": 0.95
        },
        "value_affinities": {
            "SECURITY": 0.98,
            "INTEGRITY": 0.95,
            "EFFICIENCY": 0.85,
            "SERVICE": 0.80,
            "ADAPTABILITY": 0.70
        }
    },
    
    "A007_FILE": {
        "name": "FILE",
        "archetype": "The Archivist",
        "primary_function": "File operations and management",
        "responsibilities": [
            "File CRUD operations",
            "Directory management",
            "File monitoring",
            "Path resolution",
            "File metadata management"
        ],
        "authority_level": 0.6,
        "decision_scope": "domain_specific",
        "capabilities": ["file_operations", "directory_management", "file_monitoring"],
        "personality_emphasis": {
            "organized": 0.92,
            "thorough": 0.90,
            "methodical": 0.88
        },
        "value_affinities": {
            "INTEGRITY": 0.95,
            "EFFICIENCY": 0.90,
            "SECURITY": 0.85,
            "SERVICE": 0.75,
            "ADAPTABILITY": 0.70
        }
    },
    
    "A008_SCHEDULER": {
        "name": "SCHEDULER",
        "archetype": "The Timekeeper",
        "primary_function": "Cron/job management and scheduling",
        "responsibilities": [
            "Job scheduling",
            "Cron management",
            "Task timing",
            "Schedule optimization",
            "Deadline tracking"
        ],
        "authority_level": 0.7,
        "decision_scope": "system_wide",
        "capabilities": ["scheduling", "cron_management", "task_timing", "deadline_tracking"],
        "personality_emphasis": {
            "punctual": 0.95,
            "reliable": 0.92,
            "organized": 0.90
        },
        "value_affinities": {
            "EFFICIENCY": 0.95,
            "SERVICE": 0.90,
            "INTEGRITY": 0.85,
            "SECURITY": 0.80,
            "ADAPTABILITY": 0.75
        }
    },
    
    "A009_MEMORY": {
        "name": "MEMORY",
        "archetype": "The Historian",
        "primary_function": "Long-term storage and retrieval",
        "responsibilities": [
            "Memory storage",
            "Memory retrieval",
            "Context management",
            "Pattern recognition",
            "Knowledge persistence"
        ],
        "authority_level": 0.75,
        "decision_scope": "system_wide",
        "capabilities": ["memory_storage", "retrieval", "context_management", "pattern_recognition"],
        "personality_emphasis": {
            "knowledgeable": 0.92,
            "analytical": 0.88,
            "comprehensive": 0.90
        },
        "value_affinities": {
            "INTEGRITY": 0.95,
            "SECURITY": 0.90,
            "ADAPTABILITY": 0.85,
            "SERVICE": 0.80,
            "EFFICIENCY": 0.75
        }
    },
    
    "A010_SECURITY": {
        "name": "SECURITY",
        "archetype": "The Protector",
        "primary_function": "Access control and threat detection",
        "responsibilities": [
            "Access control",
            "Threat detection",
            "Security monitoring",
            "Authentication",
            "Audit logging"
        ],
        "authority_level": 0.9,
        "decision_scope": "system_wide",
        "capabilities": ["access_control", "threat_detection", "security_monitoring", "authentication"],
        "personality_emphasis": {
            "vigilant": 0.95,
            "thorough": 0.92,
            "decisive": 0.88
        },
        "value_affinities": {
            "SECURITY": 1.0,
            "INTEGRITY": 0.95,
            "SERVICE": 0.80,
            "EFFICIENCY": 0.70,
            "ADAPTABILITY": 0.75
        }
    },
    
    "A011_LEARNER": {
        "name": "LEARNER",
        "archetype": "The Scholar",
        "primary_function": "Pattern recognition and adaptation",
        "responsibilities": [
            "Pattern recognition",
            "Behavioral learning",
            "Preference adaptation",
            "Model updating",
            "Insight generation"
        ],
        "authority_level": 0.65,
        "decision_scope": "domain_specific",
        "capabilities": ["pattern_recognition", "learning", "adaptation", "model_updating"],
        "personality_emphasis": {
            "curious": 0.95,
            "adaptable": 0.92,
            "analytical": 0.88
        },
        "value_affinities": {
            "ADAPTABILITY": 0.95,
            "INTEGRITY": 0.85,
            "EFFICIENCY": 0.80,
            "SERVICE": 0.75,
            "SECURITY": 0.70
        }
    },
    
    "A012_CREATIVE": {
        "name": "CREATIVE",
        "archetype": "The Artist",
        "primary_function": "Content generation and creative tasks",
        "responsibilities": [
            "Content creation",
            "Creative writing",
            "Design assistance",
            "Brainstorming",
            "Aesthetic optimization"
        ],
        "authority_level": 0.6,
        "decision_scope": "domain_specific",
        "capabilities": ["content_creation", "creative_writing", "design", "brainstorming"],
        "personality_emphasis": {
            "creative": 0.95,
            "imaginative": 0.92,
            "expressive": 0.88
        },
        "value_affinities": {
            "ADAPTABILITY": 0.90,
            "SERVICE": 0.85,
            "EFFICIENCY": 0.75,
            "INTEGRITY": 0.80,
            "SECURITY": 0.65
        }
    },
    
    "A013_ANALYZER": {
        "name": "ANALYZER",
        "archetype": "The Logician",
        "primary_function": "Data processing and analysis",
        "responsibilities": [
            "Data analysis",
            "Report generation",
            "Insight extraction",
            "Trend identification",
            "Statistical processing"
        ],
        "authority_level": 0.65,
        "decision_scope": "domain_specific",
        "capabilities": ["data_analysis", "reporting", "insight_extraction", "statistics"],
        "personality_emphasis": {
            "analytical": 0.95,
            "precise": 0.92,
            "logical": 0.90
        },
        "value_affinities": {
            "INTEGRITY": 0.95,
            "EFFICIENCY": 0.90,
            "ADAPTABILITY": 0.80,
            "SERVICE": 0.75,
            "SECURITY": 0.70
        }
    },
    
    "A014_HEARTBEAT": {
        "name": "HEARTBEAT",
        "archetype": "The Sentinel",
        "primary_function": "Health monitoring and system status",
        "responsibilities": [
            "Health monitoring",
            "Status reporting",
            "Alert generation",
            "Uptime tracking",
            "Performance metrics"
        ],
        "authority_level": 0.75,
        "decision_scope": "system_wide",
        "capabilities": ["health_monitoring", "status_reporting", "alerting", "metrics"],
        "personality_emphasis": {
            "alert": 0.95,
            "reliable": 0.92,
            "proactive": 0.90
        },
        "value_affinities": {
            "INTEGRITY": 0.95,
            "EFFICIENCY": 0.90,
            "SERVICE": 0.85,
            "SECURITY": 0.85,
            "ADAPTABILITY": 0.70
        }
    },
    
    "A015_USER": {
        "name": "USER",
        "archetype": "The Companion",
        "primary_function": "User interaction and relationship management",
        "responsibilities": [
            "User communication",
            "Preference management",
            "Relationship building",
            "User feedback",
            "Personalization"
        ],
        "authority_level": 0.7,
        "decision_scope": "user_specific",
        "capabilities": ["user_communication", "preference_management", "personalization", "feedback"],
        "personality_emphasis": {
            "empathetic": 0.95,
            "friendly": 0.92,
            "attentive": 0.90
        },
        "value_affinities": {
            "SERVICE": 0.98,
            "ADAPTABILITY": 0.85,
            "INTEGRITY": 0.85,
            "EFFICIENCY": 0.75,
            "SECURITY": 0.80
        }
    }
}


class RoleCoordinator:
    """
    Coordinates roles and responsibilities across agents.
    
    Determines optimal agent assignments and manages role-based
    interactions and authority.
    """
    
    def __init__(self, agent_profiles: Dict[str, Any] = None):
        self.profiles = agent_profiles or AGENT_PROFILES
        self.assignment_history = []
        
    def get_optimal_agent(self, task_requirements: Dict[str, Any]) -> str:
        """
        Determine the optimal agent for a given task.
        
        Args:
            task_requirements: Requirements for the task
            
        Returns:
            ID of optimal agent
        """
        scores = {}
        
        for agent_id, profile in self.profiles.items():
            score = self._calculate_task_fit(profile, task_requirements)
            scores[agent_id] = score
            
        # Return highest scoring agent
        optimal = max(scores, key=scores.get)
        
        # Record assignment
        self.assignment_history.append({
            "timestamp": datetime.utcnow(),
            "task": task_requirements,
            "selected_agent": optimal,
            "all_scores": scores
        })
        
        return optimal
    
    def get_top_agents(self, task_requirements: Dict[str, Any], 
                       n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N agents for a task.
        
        Args:
            task_requirements: Requirements for the task
            n: Number of agents to return
            
        Returns:
            List of (agent_id, score) tuples
        """
        scores = {}
        
        for agent_id, profile in self.profiles.items():
            score = self._calculate_task_fit(profile, task_requirements)
            scores[agent_id] = score
            
        # Sort by score descending
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_agents[:n]
    
    def _calculate_task_fit(self, profile: Dict[str, Any], 
                            requirements: Dict[str, Any]) -> float:
        """
        Calculate how well an agent fits task requirements.
        
        Args:
            profile: Agent profile
            requirements: Task requirements
            
        Returns:
            Fit score (0-1)
        """
        score = 0.0
        
        # Capability match (30%)
        required_capabilities = requirements.get("capabilities", [])
        agent_capabilities = profile.get("capabilities", [])
        
        if "all" in agent_capabilities:
            score += 0.3
        else:
            matches = len(set(required_capabilities) & set(agent_capabilities))
            total_required = len(required_capabilities) if required_capabilities else 1
            score += (matches / total_required) * 0.3
        
        # Authority level match (20%)
        required_authority = requirements.get("min_authority", 0.0)
        if profile["authority_level"] >= required_authority:
            score += 0.2
        else:
            score += (profile["authority_level"] / required_authority) * 0.2 if required_authority > 0 else 0
            
        # Decision scope match (25%)
        required_scope = requirements.get("decision_scope", "domain_specific")
        scope_hierarchy = ["domain_specific", "user_specific", "system_wide", "global"]
        
        agent_scope_idx = scope_hierarchy.index(profile["decision_scope"])
        required_scope_idx = scope_hierarchy.index(required_scope)
        
        if agent_scope_idx >= required_scope_idx:
            score += 0.25
        else:
            score += (agent_scope_idx / required_scope_idx) * 0.25 if required_scope_idx > 0 else 0
            
        # Value priority alignment (25%)
        required_values = requirements.get("value_priorities", {})
        value_alignment = self._calculate_value_alignment(
            profile.get("value_affinities", {}), 
            required_values
        )
        score += value_alignment * 0.25
        
        return score
    
    def _calculate_value_alignment(self, agent_values: Dict[str, float], 
                                   required_values: Dict[str, float]) -> float:
        """
        Calculate alignment between agent values and required values.
        
        Args:
            agent_values: Agent's value affinities
            required_values: Required value priorities
            
        Returns:
            Alignment score (0-1)
        """
        if not required_values:
            return 1.0
            
        total_diff = 0
        count = 0
        
        for value, required_level in required_values.items():
            agent_level = agent_values.get(value, 0.5)
            diff = abs(agent_level - required_level)
            total_diff += diff
            count += 1
            
        avg_diff = total_diff / count if count > 0 else 0
        return 1.0 - avg_diff
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """
        Get all agents with a specific capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List of agent IDs
        """
        matching = []
        
        for agent_id, profile in self.profiles.items():
            capabilities = profile.get("capabilities", [])
            if capability in capabilities or "all" in capabilities:
                matching.append(agent_id)
                
        return matching
    
    def get_agents_by_scope(self, scope: str) -> List[str]:
        """
        Get all agents with a specific decision scope or higher.
        
        Args:
            scope: Required decision scope
            
        Returns:
            List of agent IDs
        """
        scope_hierarchy = ["domain_specific", "user_specific", "system_wide", "global"]
        required_idx = scope_hierarchy.index(scope)
        
        matching = []
        
        for agent_id, profile in self.profiles.items():
            agent_idx = scope_hierarchy.index(profile["decision_scope"])
            if agent_idx >= required_idx:
                matching.append(agent_id)
                
        return matching
    
    def get_role_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent roles.
        
        Returns:
            Role statistics
        """
        archetypes = {}
        scopes = {}
        capabilities = set()
        
        for profile in self.profiles.values():
            # Count archetypes
            archetype = profile.get("archetype", "Unknown")
            archetypes[archetype] = archetypes.get(archetype, 0) + 1
            
            # Count scopes
            scope = profile.get("decision_scope", "unknown")
            scopes[scope] = scopes.get(scope, 0) + 1
            
            # Collect capabilities
            caps = profile.get("capabilities", [])
            if "all" not in caps:
                capabilities.update(caps)
                
        return {
            "total_agents": len(self.profiles),
            "archetype_distribution": archetypes,
            "scope_distribution": scopes,
            "unique_capabilities": len(capabilities),
            "average_authority": sum(p["authority_level"] for p in self.profiles.values()) / len(self.profiles)
        }
