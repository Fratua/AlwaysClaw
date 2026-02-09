"""
Config Migration Manager for Self-Updating Loop
Handles configuration migration between versions with full rollback support,
BFS-based migration path finding, and SQLite history persistence.
"""

import copy
import json
import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MigrationStep:
    """A single migration between two adjacent versions."""
    from_version: str
    to_version: str
    description: str
    migration_func: Callable[[dict], dict]
    rollback_func: Optional[Callable[[dict], dict]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MigrationResult:
    """Outcome of a full migration run."""
    success: bool = True
    steps_applied: int = 0
    errors: List[str] = field(default_factory=list)
    new_config: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Migration registry with BFS path-finding
# ---------------------------------------------------------------------------

class MigrationRegistry:
    """
    Registry of all known migration steps.

    Migrations are keyed by (from_version, to_version) and the registry
    can compute the shortest migration chain between any two versions
    using BFS over the version graph.
    """

    def __init__(self):
        self._migrations: Dict[Tuple[str, str], MigrationStep] = {}

    def register(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migrate_func: Callable[[dict], dict],
        rollback_func: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        """Register a migration between two versions."""
        step = MigrationStep(
            from_version=from_version,
            to_version=to_version,
            description=description,
            migration_func=migrate_func,
            rollback_func=rollback_func,
        )
        self._migrations[(from_version, to_version)] = step

    def get_path(self, from_version: str, to_version: str) -> List[MigrationStep]:
        """
        Find the shortest migration chain from *from_version* to
        *to_version* using BFS.

        Returns an ordered list of MigrationStep objects.
        Raises ValueError if no path exists.
        """
        if from_version == to_version:
            return []

        # Build adjacency map: version -> list of (next_version, step)
        adjacency: Dict[str, List[Tuple[str, MigrationStep]]] = {}
        for (src, dst), step in self._migrations.items():
            adjacency.setdefault(src, []).append((dst, step))

        # BFS
        queue: deque[Tuple[str, List[MigrationStep]]] = deque()
        queue.append((from_version, []))
        visited = {from_version}

        while queue:
            current, path = queue.popleft()
            for neighbour, step in adjacency.get(current, []):
                if neighbour == to_version:
                    return path + [step]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, path + [step]))

        raise ValueError(
            f"No migration path from {from_version} to {to_version}"
        )

    def get_all_versions(self) -> List[str]:
        """Return a sorted list of all known versions."""
        versions: set[str] = set()
        for src, dst in self._migrations:
            versions.add(src)
            versions.add(dst)
        return sorted(versions)


# ---------------------------------------------------------------------------
# Migration manager
# ---------------------------------------------------------------------------

class ConfigMigrationManager:
    """
    Manages configuration migrations between versions.

    Features:
    - Registry-based migration step management
    - BFS shortest-path migration chain resolution
    - Automatic rollback of partially applied chains on failure
    - SQLite-backed migration history
    - Schema-aware config validation
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS migration_history (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        from_version  TEXT    NOT NULL,
        to_version    TEXT    NOT NULL,
        executed_at   TEXT    NOT NULL,
        success       BOOLEAN NOT NULL,
        error_message TEXT,
        config_snapshot TEXT
    )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ):
        self.config = config or {}
        self._registry = MigrationRegistry()
        self._history: List[Dict] = []
        self._db_path: str = db_path or str(
            Path(__file__).resolve().parent / "migration_history.db"
        )
        self._ensure_db()
        self._history = self._load_history()
        logger.info("ConfigMigrationManager initialized (db=%s)", self._db_path)

    # -- public API ----------------------------------------------------------

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migrate_func: Callable[[dict], dict],
        rollback_func: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        """Register a migration step in the internal registry."""
        self._registry.register(
            from_version, to_version, description,
            migrate_func, rollback_func,
        )

    async def migrate(
        self,
        config: dict,
        from_version: str,
        to_version: str,
    ) -> MigrationResult:
        """
        Migrate *config* from *from_version* to *to_version*.

        On failure the manager rolls back all already-applied steps in
        reverse order (if rollback functions are available) and persists
        the failure record.
        """
        result = MigrationResult(new_config=copy.deepcopy(config))

        try:
            path = self._registry.get_path(from_version, to_version)
        except ValueError as exc:
            result.success = False
            result.errors.append(str(exc))
            self._persist_history({
                "from_version": from_version,
                "to_version": to_version,
                "executed_at": datetime.now().isoformat(),
                "success": False,
                "error_message": str(exc),
                "config_snapshot": json.dumps(config),
            })
            return result

        if not path:
            # Already at target version
            return result

        applied: List[Tuple[MigrationStep, dict]] = []  # (step, pre_config)

        for step in path:
            snapshot = copy.deepcopy(result.new_config)
            try:
                result.new_config = step.migration_func(result.new_config)
                applied.append((step, snapshot))
                result.steps_applied += 1
            except Exception as exc:
                result.success = False
                result.errors.append(
                    f"Migration {step.from_version}->{step.to_version} "
                    f"failed: {exc}"
                )
                # Rollback completed steps in reverse
                for prev_step, prev_config in reversed(applied):
                    if prev_step.rollback_func is not None:
                        try:
                            result.new_config = prev_step.rollback_func(
                                result.new_config
                            )
                        except Exception as rb_exc:
                            result.errors.append(
                                f"Rollback {prev_step.to_version}->"
                                f"{prev_step.from_version} failed: {rb_exc}"
                            )
                            # Last resort: restore raw snapshot
                            result.new_config = prev_config
                    else:
                        result.new_config = prev_config
                break

        record = {
            "from_version": from_version,
            "to_version": to_version,
            "executed_at": datetime.now().isoformat(),
            "success": result.success,
            "error_message": "; ".join(result.errors) if result.errors else None,
            "config_snapshot": json.dumps(result.new_config),
        }
        self._persist_history(record)
        self._history.append(record)

        return result

    async def validate_config(
        self,
        config: Dict[str, Any],
        schema: Optional[dict] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate *config*, optionally against *schema*.

        *schema* may contain:
        - ``required``: list of required top-level keys
        - ``types``: dict mapping key -> expected type name (e.g. ``"str"``)

        Returns ``(valid, errors)``.
        """
        errors: List[str] = []
        if not isinstance(config, dict):
            errors.append("Config must be a dict")
            return False, errors

        if schema is None:
            return True, errors

        # Check required keys
        for key in schema.get("required", []):
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Check types
        type_map = {
            "str": str, "string": str,
            "int": int, "integer": int,
            "float": float,
            "bool": bool, "boolean": bool,
            "list": list,
            "dict": dict,
        }
        for key, expected_type_name in schema.get("types", {}).items():
            if key in config:
                expected = type_map.get(expected_type_name)
                if expected and not isinstance(config[key], expected):
                    errors.append(
                        f"Key '{key}' expected type {expected_type_name}, "
                        f"got {type(config[key]).__name__}"
                    )

        return len(errors) == 0, errors

    def get_current_version(self, config: dict) -> str:
        """Read the version string from *config*."""
        return str(config.get("version", "0.0.0"))

    # -- SQLite persistence --------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the migration history table if it does not exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(self._SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("Failed to initialise migration DB: %s", exc)

    def _persist_history(self, record: dict) -> None:
        """Write a migration record to SQLite."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT INTO migration_history
                   (from_version, to_version, executed_at, success,
                    error_message, config_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record["from_version"],
                    record["to_version"],
                    record["executed_at"],
                    record["success"],
                    record.get("error_message"),
                    record.get("config_snapshot"),
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("Failed to persist migration record: %s", exc)

    def _load_history(self) -> List[dict]:
        """Load all migration history from SQLite."""
        rows: List[dict] = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM migration_history ORDER BY id"
            )
            for row in cursor:
                rows.append({
                    "id": row["id"],
                    "from_version": row["from_version"],
                    "to_version": row["to_version"],
                    "executed_at": row["executed_at"],
                    "success": bool(row["success"]),
                    "error_message": row["error_message"],
                    "config_snapshot": row["config_snapshot"],
                })
            conn.close()
        except sqlite3.Error as exc:
            logger.error("Failed to load migration history: %s", exc)
        return rows
