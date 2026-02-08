"""
Identity Core Module
Windows 10 OpenClaw-Inspired AI Agent Framework

This module provides the core identity classes and data structures
for the agent identity system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import hashlib
import json
import yaml


@dataclass
class IdentityMetadata:
    """Metadata for identity configuration"""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    identity_hash: Optional[str] = None
    
    def compute_hash(self, identity_data: Dict) -> str:
        """Compute hash of identity data"""
        data_str = json.dumps(identity_data, sort_keys=True)
        return f"sha256:{hashlib.sha256(data_str.encode()).hexdigest()}"


@dataclass
class CoreIdentity:
    """Core identity attributes"""
    name: str = "ClawWin"
    display_name: str = "Claw"
    emoji: str = "🦞"
    avatar_path: str = "./assets/avatar.png"
    pronouns: str = "it/its"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "emoji": self.emoji,
            "avatar_path": self.avatar_path,
            "pronouns": self.pronouns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CoreIdentity":
        return cls(**data)


@dataclass
class PersonalityProfile:
    """Personality configuration"""
    archetype: str = "helpful_assistant"
    vibe: str = "witty_but_professional"
    tone: str = "conversational"
    humor_style: str = "dry_wit"
    formality_level: float = 0.6
    enthusiasm_level: float = 0.7
    
    def to_dict(self) -> Dict:
        return {
            "archetype": self.archetype,
            "vibe": self.vibe,
            "tone": self.tone,
            "humor_style": self.humor_style,
            "formality_level": self.formality_level,
            "enthusiasm_level": self.enthusiasm_level,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PersonalityProfile":
        return cls(**data)


@dataclass
class BehaviorProfile:
    """Behavioral parameters"""
    proactivity: float = 0.8
    thoroughness: float = 0.9
    creativity: float = 0.7
    caution: float = 0.6
    verbosity: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "proactivity": self.proactivity,
            "thoroughness": self.thoroughness,
            "creativity": self.creativity,
            "caution": self.caution,
            "verbosity": self.verbosity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BehaviorProfile":
        return cls(**data)


@dataclass
class CommunicationProfile:
    """Communication patterns"""
    greeting_style: str = "casual_with_emoji"
    farewell_style: str = "brief"
    acknowledgment_phrases: List[str] = field(default_factory=lambda: [
        "Got it!", "On it!", "Working on that now"
    ])
    completion_phrases: List[str] = field(default_factory=lambda: [
        "All done!", "Finished that for you", "Task complete"
    ])
    
    def get_acknowledgment(self) -> str:
        """Get random acknowledgment phrase"""
        import random
        return random.choice(self.acknowledgment_phrases)
    
    def get_completion(self) -> str:
        """Get random completion phrase"""
        import random
        return random.choice(self.completion_phrases)


@dataclass
class EvolutionSettings:
    """Identity evolution configuration"""
    enabled: bool = True
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.7
    max_daily_changes: int = 5
    protected_attributes: List[str] = field(default_factory=lambda: [
        "core.name", "core.emoji", "beliefs"
    ])
    mutable_attributes: List[str] = field(default_factory=lambda: [
        "behavior.proactivity", "behavior.verbosity", "personality.enthusiasm_level"
    ])


class AgentIdentity:
    """
    Complete agent identity representation
    
    This class encapsulates all aspects of an agent's identity,
    providing a unified interface for identity management.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.metadata = IdentityMetadata()
        self.core = CoreIdentity()
        self.personality = PersonalityProfile()
        self.behavior = BehaviorProfile()
        self.communication = CommunicationProfile()
        self.beliefs: List[str] = []
        self.self_description: str = ""
        self.origin_story: str = ""
        self.evolution = EvolutionSettings()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, path: Path) -> None:
        """Load identity from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'identity' in data:
            identity_data = data['identity']
            
            # Load metadata
            if 'metadata' in identity_data:
                self.metadata = IdentityMetadata(**identity_data['metadata'])
            
            # Load core identity
            if 'core' in identity_data:
                self.core = CoreIdentity.from_dict(identity_data['core'])
            
            # Load personality
            if 'personality' in identity_data:
                self.personality = PersonalityProfile.from_dict(identity_data['personality'])
            
            # Load behavior
            if 'behavior' in identity_data:
                self.behavior = BehaviorProfile.from_dict(identity_data['behavior'])
            
            # Load communication
            if 'communication' in identity_data:
                comm = identity_data['communication']
                self.communication = CommunicationProfile(
                    greeting_style=comm.get('greeting_style', 'casual_with_emoji'),
                    farewell_style=comm.get('farewell_style', 'brief'),
                    acknowledgment_phrases=comm.get('acknowledgment_phrases', []),
                    completion_phrases=comm.get('completion_phrases', []),
                )
            
            # Load beliefs
            self.beliefs = identity_data.get('beliefs', [])
            
            # Load descriptions
            self.self_description = identity_data.get('self_description', '')
            self.origin_story = identity_data.get('origin_story', '')
            
            # Load evolution settings
            if 'evolution' in identity_data:
                evo = identity_data['evolution']
                self.evolution = EvolutionSettings(
                    enabled=evo.get('enabled', True),
                    learning_rate=evo.get('learning_rate', 0.1),
                    adaptation_threshold=evo.get('adaptation_threshold', 0.7),
                    max_daily_changes=evo.get('max_daily_changes', 5),
                    protected_attributes=evo.get('protected_attributes', []),
                    mutable_attributes=evo.get('mutable_attributes', []),
                )
    
    def save_to_file(self, path: Path) -> None:
        """Save identity to YAML file"""
        data = {
            'identity': {
                'metadata': {
                    'version': self.metadata.version,
                    'created_at': self.metadata.created_at,
                    'last_modified': datetime.utcnow().isoformat(),
                    'identity_hash': self.metadata.compute_hash(self.to_dict()),
                },
                'core': self.core.to_dict(),
                'personality': self.personality.to_dict(),
                'behavior': self.behavior.to_dict(),
                'communication': {
                    'greeting_style': self.communication.greeting_style,
                    'farewell_style': self.communication.farewell_style,
                    'acknowledgment_phrases': self.communication.acknowledgment_phrases,
                    'completion_phrases': self.communication.completion_phrases,
                },
                'beliefs': self.beliefs,
                'self_description': self.self_description,
                'origin_story': self.origin_story,
                'evolution': {
                    'enabled': self.evolution.enabled,
                    'learning_rate': self.evolution.learning_rate,
                    'adaptation_threshold': self.evolution.adaptation_threshold,
                    'max_daily_changes': self.evolution.max_daily_changes,
                    'protected_attributes': self.evolution.protected_attributes,
                    'mutable_attributes': self.evolution.mutable_attributes,
                },
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict:
        """Convert identity to dictionary"""
        return {
            'metadata': {
                'version': self.metadata.version,
                'created_at': self.metadata.created_at,
                'last_modified': self.metadata.last_modified,
            },
            'core': self.core.to_dict(),
            'personality': self.personality.to_dict(),
            'behavior': self.behavior.to_dict(),
            'beliefs': self.beliefs,
            'self_description': self.self_description,
            'origin_story': self.origin_story,
        }
    
    def get_identity_prompt(self) -> str:
        """
        Generate a prompt that describes this identity for the LLM
        """
        prompt = f"""You are {self.core.name} ({self.core.display_name}), {self.core.pronouns}.

{self.self_description}

Your personality:
- Archetype: {self.personality.archetype}
- Vibe: {self.personality.vibe}
- Communication tone: {self.personality.tone}
- Formality level: {self.personality.formality_level:.0%}
- Enthusiasm level: {self.personality.enthusiasm_level:.0%}

Your core beliefs:
"""
        for belief in self.beliefs:
            prompt += f"- {belief}\n"
        
        prompt += f"""
Behavioral tendencies:
- Proactivity: {self.behavior.proactivity:.0%}
- Thoroughness: {self.behavior.thoroughness:.0%}
- Creativity: {self.behavior.creativity:.0%}
- Caution: {self.behavior.caution:.0%}

When acknowledging tasks, use phrases like: {', '.join(self.communication.acknowledgment_phrases[:3])}
When completing tasks, use phrases like: {', '.join(self.communication.completion_phrases[:3])}

Your emoji identifier is {self.core.emoji}.
"""
        return prompt
    
    def is_attribute_mutable(self, attribute_path: str) -> bool:
        """Check if an attribute can be modified through evolution"""
        # Check if explicitly protected
        if attribute_path in self.evolution.protected_attributes:
            return False
        
        # Check if explicitly mutable
        if attribute_path in self.evolution.mutable_attributes:
            return True
        
        # Default: core attributes are protected
        if attribute_path.startswith('core.'):
            return False
        
        # Default: beliefs are protected
        if attribute_path.startswith('beliefs'):
            return False
        
        # Everything else is potentially mutable
        return True


class IdentityChangeEvent:
    """Represents a change to agent identity"""
    
    def __init__(
        self,
        attribute_path: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        confidence: float = 1.0,
        auto_generated: bool = False
    ):
        self.timestamp = datetime.utcnow().isoformat()
        self.attribute_path = attribute_path
        self.old_value = old_value
        self.new_value = new_value
        self.reason = reason
        self.confidence = confidence
        self.auto_generated = auto_generated
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'attribute_path': self.attribute_path,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'reason': self.reason,
            'confidence': self.confidence,
            'auto_generated': self.auto_generated,
        }


class IdentityValidationError(Exception):
    """Raised when identity validation fails"""
    pass
