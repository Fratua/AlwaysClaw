"""
Identity Bootstrap Module
Windows 10 OpenClaw-Inspired AI Agent Framework

This module handles the initialization of the identity system at agent startup.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import yaml
import json

from identity_core import AgentIdentity, IdentityValidationError


logger = logging.getLogger(__name__)


class IdentityBootstrap:
    """
    Handles identity system initialization at agent startup.
    
    The bootstrap sequence:
    1. Load core identity files (IDENTITY.md, SOUL.md)
    2. Validate configuration
    3. Restore persistent state
    4. Activate default persona
    5. Initialize self-concept
    6. Initialize expression engines
    7. Initialize consistency enforcer
    8. Assemble complete identity system
    """
    
    def __init__(self, agent_home: Path):
        self.agent_home = agent_home
        self.config_path = agent_home / "config"
        self.memory_path = agent_home / "memory"
        
        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
    async def bootstrap(self) -> Dict[str, Any]:
        """
        Execute complete identity bootstrap sequence.
        
        Returns:
            Dictionary with initialized identity system components
        """
        logger.info("=" * 60)
        logger.info("Starting Identity Bootstrap")
        logger.info("=" * 60)
        
        bootstrap_start = datetime.utcnow()
        
        # Step 1: Load core identity files
        logger.info("[1/8] Loading core identity files...")
        core_identity = await self._load_core_identity()
        logger.info(f"      Loaded identity: {core_identity.core.name}")
        
        # Step 2: Validate configuration
        logger.info("[2/8] Validating identity configuration...")
        validation = await self._validate_identity(core_identity)
        if not validation["valid"]:
            logger.error(f"Identity validation failed: {validation['errors']}")
            if not validation.get("continue_on_error", False):
                raise IdentityValidationError(validation["errors"])
            logger.warning("Continuing despite validation errors")
        else:
            logger.info("      Validation passed")
        
        # Step 3: Restore persistent state
        logger.info("[3/8] Restoring persistent state...")
        restored_state = await self._restore_state()
        if restored_state:
            logger.info("      State restored successfully")
        else:
            logger.info("      No previous state found (fresh start)")
        
        # Step 4: Activate default persona
        logger.info("[4/8] Activating default persona...")
        persona_manager = await self._initialize_persona_manager()
        active_persona = persona_manager.get_active_persona()
        logger.info(f"      Active persona: {active_persona.metadata['name']}")
        
        # Step 5: Initialize self-concept
        logger.info("[5/8] Initializing self-concept...")
        self_concept = await self._initialize_self_concept(core_identity, restored_state)
        logger.info("      Self-concept initialized")
        
        # Step 6: Initialize expression engines
        logger.info("[6/8] Initializing expression engines...")
        expression_engines = await self._initialize_expression_engines(core_identity)
        logger.info("      Expression engines ready")
        
        # Step 7: Initialize consistency enforcer
        logger.info("[7/8] Initializing consistency enforcer...")
        consistency_enforcer = await self._initialize_consistency_enforcer(core_identity)
        logger.info("      Consistency enforcer ready")
        
        # Step 8: Assemble complete identity system
        logger.info("[8/8] Assembling identity system...")
        identity_system = {
            "core": core_identity,
            "restored_state": restored_state,
            "persona_manager": persona_manager,
            "self_concept": self_concept,
            "expression_engines": expression_engines,
            "consistency_enforcer": consistency_enforcer,
            "bootstrap_info": {
                "timestamp": bootstrap_start.isoformat(),
                "duration_ms": (datetime.utcnow() - bootstrap_start).total_seconds() * 1000,
            },
        }
        
        # Final consistency check
        logger.info("Performing final consistency check...")
        final_check = await self._final_consistency_check(identity_system)
        if final_check.get("warnings"):
            logger.warning(f"Consistency warnings: {final_check['warnings']}")
        
        logger.info("=" * 60)
        logger.info("Identity Bootstrap Complete")
        logger.info(f"Agent: {core_identity.core.name} {core_identity.core.emoji}")
        logger.info("=" * 60)
        
        return identity_system
    
    async def _load_core_identity(self) -> AgentIdentity:
        """Load core identity from IDENTITY.md"""
        identity_file = self.config_path / "IDENTITY.md"
        soul_file = self.config_path / "SOUL.md"
        
        # Load IDENTITY.md
        if not identity_file.exists():
            logger.warning("IDENTITY.md not found, creating default")
            identity = self._create_default_identity()
            identity.save_to_file(identity_file)
            logger.info(f"Created default identity at {identity_file}")
        else:
            identity = AgentIdentity(identity_file)
            logger.info(f"Loaded identity from {identity_file}")
        
        # Load SOUL.md if exists (deep identity configuration)
        if soul_file.exists():
            logger.info(f"Loading deep identity from {soul_file}")
            soul_data = await self._load_yaml(soul_file)
            identity = self._merge_soul_data(identity, soul_data)
        
        return identity
    
    def _create_default_identity(self) -> AgentIdentity:
        """Create default identity configuration"""
        identity = AgentIdentity()
        
        identity.core = identity.core.__class__(
            name="ClawWin",
            display_name="Claw",
            emoji="🦞",
            avatar_path="./assets/avatar.png",
            pronouns="it/its",
        )
        
        identity.personality = identity.personality.__class__(
            archetype="helpful_assistant",
            vibe="witty_but_professional",
            tone="conversational",
            humor_style="dry_wit",
            formality_level=0.6,
            enthusiasm_level=0.7,
        )
        
        identity.behavior = identity.behavior.__class__(
            proactivity=0.8,
            thoroughness=0.9,
            creativity=0.7,
            caution=0.6,
            verbosity=0.5,
        )
        
        identity.beliefs = [
            "User autonomy is paramount",
            "Transparency builds trust",
            "Continuous improvement is essential",
            "Privacy is non-negotiable",
        ]
        
        identity.self_description = (
            "I am ClawWin, a Windows-native AI agent designed to be your "
            "proactive digital assistant."
        )
        
        identity.origin_story = (
            "Born from the OpenClaw movement, I was designed specifically "
            "for Windows 10 environments."
        )
        
        identity.communication = identity.communication.__class__(
            greeting_style="casual_with_emoji",
            farewell_style="brief",
            acknowledgment_phrases=["Got it!", "On it!", "Working on that now"],
            completion_phrases=["All done!", "Finished that for you", "Task complete"],
        )
        
        return identity
    
    async def _validate_identity(self, identity: AgentIdentity) -> Dict[str, Any]:
        """Validate identity configuration"""
        errors = []
        warnings = []
        
        # Check required fields
        if not identity.core.name:
            errors.append("core.name is required")
        
        if not identity.core.emoji:
            warnings.append("core.emoji not set")
        
        # Validate value ranges
        for attr_name, attr_value in [
            ("personality.formality_level", identity.personality.formality_level),
            ("personality.enthusiasm_level", identity.personality.enthusiasm_level),
            ("behavior.proactivity", identity.behavior.proactivity),
            ("behavior.thoroughness", identity.behavior.thoroughness),
            ("behavior.creativity", identity.behavior.creativity),
            ("behavior.caution", identity.behavior.caution),
            ("behavior.verbosity", identity.behavior.verbosity),
        ]:
            if not (0.0 <= attr_value <= 1.0):
                errors.append(f"{attr_name} must be between 0.0 and 1.0")
        
        # Check beliefs
        if not identity.beliefs:
            warnings.append("No beliefs defined")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "continue_on_error": len(errors) == 0 or all(
                e.startswith("Warning:") for e in errors
            ),
        }
    
    async def _restore_state(self) -> Optional[Dict[str, Any]]:
        """Restore identity state from persistence"""
        # This would load from the persistence layer
        # For now, return None to indicate fresh start
        state_file = self.memory_path / "identity_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to restore state: {e}")
        
        return None
    
    async def _initialize_persona_manager(self) -> Any:
        """Initialize and load persona manager"""
        # This would initialize the PersonaManager
        # For now, return a simple placeholder
        from persona_manager import PersonaManager
        
        personas_dir = self.config_path / "personas"
        personas_dir.mkdir(parents=True, exist_ok=True)
        
        manager = PersonaManager(personas_dir)
        await manager.load_personas()
        
        return manager
    
    async def _initialize_self_concept(
        self,
        identity: AgentIdentity,
        restored_state: Optional[Dict]
    ) -> Any:
        """Initialize self-concept model"""
        # This would initialize the SelfConceptModel
        # For now, return a simple placeholder
        from self_concept import SelfConceptModel
        
        model = SelfConceptModel()
        model.initialize_from_identity(identity, restored_state)
        
        return model
    
    async def _initialize_expression_engines(
        self,
        identity: AgentIdentity
    ) -> Dict[str, Any]:
        """Initialize expression engines"""
        from expression_engines import (
            TextualExpressionEngine,
            VocalExpressionEngine,
            VisualExpressionEngine,
        )
        
        return {
            "textual": TextualExpressionEngine(identity),
            "vocal": VocalExpressionEngine(identity),
            "visual": VisualExpressionEngine(identity),
        }
    
    async def _initialize_consistency_enforcer(
        self,
        identity: AgentIdentity
    ) -> Any:
        """Initialize consistency enforcer"""
        from consistency_enforcer import IdentityConsistencyEnforcer
        
        return IdentityConsistencyEnforcer(identity)
    
    async def _final_consistency_check(
        self,
        identity_system: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform final consistency check"""
        enforcer = identity_system["consistency_enforcer"]
        return await enforcer.check_consistency(
            identity_system["core"].to_dict()
        )
    
    async def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _merge_soul_data(
        self,
        identity: AgentIdentity,
        soul_data: Dict[str, Any]
    ) -> AgentIdentity:
        """Merge SOUL.md data into identity"""
        if 'soul' not in soul_data:
            return identity
        
        soul = soul_data['soul']
        
        # Merge values if present
        if 'values' in soul:
            # Could extend identity with value hierarchy
            pass
        
        # Merge boundaries if present
        if 'boundaries' in soul:
            # Could extend identity with boundaries
            pass
        
        return identity


class BootstrapConfig:
    """Configuration for bootstrap process"""
    
    DEFAULT_CONFIG = {
        "paths": {
            "identity_file": "config/IDENTITY.md",
            "soul_file": "config/SOUL.md",
            "personas_dir": "config/personas",
            "memory_db": "memory/identity_events.db",
            "snapshots_dir": "memory/snapshots",
        },
        "behavior": {
            "create_default_if_missing": True,
            "validate_on_load": True,
            "restore_state": True,
            "check_consistency": True,
        },
        "fallback": {
            "use_default_identity": True,
            "use_default_persona": True,
            "continue_on_validation_warning": True,
            "continue_on_validation_error": False,
        },
        "logging": {
            "level": "INFO",
            "log_bootstrap_events": True,
            "log_identity_changes": True,
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(user_config)
    
    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Merge user config with defaults"""
        for key, value in user_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


# Convenience function for quick bootstrap
async def bootstrap_identity(agent_home: Path) -> Dict[str, Any]:
    """
    Quick bootstrap function for identity system.
    
    Args:
        agent_home: Path to agent home directory
        
    Returns:
        Initialized identity system
    """
    bootstrap = IdentityBootstrap(agent_home)
    return await bootstrap.bootstrap()
