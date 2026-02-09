"""
Config Migration Manager for Self-Updating Loop
Handles configuration migration between versions.
"""

import logging
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MigrationStep:
    from_version: str
    to_version: str
    description: str
    changes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MigrationResult:
    success: bool = True
    steps_applied: int = 0
    errors: List[str] = field(default_factory=list)
    new_config: Optional[Dict[str, Any]] = None


class ConfigMigrationManager:
    """Manages configuration migrations between versions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._migrations: List[MigrationStep] = []
        logger.info("ConfigMigrationManager initialized")

    def register_migration(self, step: MigrationStep) -> None:
        """Register a migration step."""
        self._migrations.append(step)

    async def migrate(
        self,
        current_config: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> MigrationResult:
        """Migrate configuration from one version to another."""
        result = MigrationResult(new_config=copy.deepcopy(current_config))

        applicable = [
            m for m in self._migrations
            if m.from_version >= from_version and m.to_version <= to_version
        ]
        applicable.sort(key=lambda m: m.from_version)

        for step in applicable:
            try:
                for change in step.changes:
                    action = change.get("action", "set")
                    key = change.get("key", "")
                    value = change.get("value")

                    if action == "set":
                        result.new_config[key] = value
                    elif action == "delete" and key in result.new_config:
                        del result.new_config[key]
                    elif action == "rename":
                        old_key = change.get("old_key", "")
                        new_key = change.get("new_key", "")
                        if old_key in result.new_config:
                            result.new_config[new_key] = result.new_config.pop(old_key)

                result.steps_applied += 1
            except (KeyError, TypeError, ValueError) as e:
                result.errors.append(f"Migration {step.from_version}->{step.to_version} failed: {e}")
                result.success = False
                break

        return result

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a configuration dictionary."""
        if not isinstance(config, dict):
            return False
        return True
