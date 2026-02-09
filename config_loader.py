"""
Central Configuration Loader for OpenClaw AI Agent System
Loads YAML configs with environment variable substitution, validation, and caching.

Usage:
    from config_loader import get_config, ConfigLoader

    # Get a specific config section
    memory_cfg = get_config("memory_config", "embedding")
    sample_rate = get_config("tts_config", "tts.audio.sample_rate")

    # Or use the loader directly
    loader = ConfigLoader.instance()
    full_config = loader.load("cpel_config")
"""

import os
import re
import yaml
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom YAML tag: !env for environment variable substitution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')


def _env_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """Handle !env tag: !env ${VAR_NAME:default_value}"""
    value = loader.construct_scalar(node)
    strict = getattr(loader, 'strict', False)
    return _substitute_env_vars(value, strict=strict)


class UnresolvedEnvVarError(Exception):
    """Raised when strict mode finds unresolved ${VAR} placeholders."""
    def __init__(self, unresolved: List[str]):
        self.unresolved = unresolved
        super().__init__(
            f"Unresolved environment variables: {', '.join(unresolved)}"
        )


def _substitute_env_vars(value: str, *, strict: bool = False) -> str:
    """Replace ${VAR_NAME} and ${VAR_NAME:default} patterns with env values.

    Args:
        value: The string containing env var placeholders.
        strict: When True, raise ``UnresolvedEnvVarError`` if any placeholder
                has no matching env var and no default value.  When False
                (the default), unresolved placeholders are left as literal
                strings (backwards-compatible behaviour).
    """
    unresolved: List[str] = []

    def _replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        # No env var found and no default provided
        unresolved.append(var_name)
        return match.group(0)  # leave unchanged if no env and no default

    result = _ENV_PATTERN.sub(_replacer, value)

    if strict and unresolved:
        raise UnresolvedEnvVarError(unresolved)

    return result


class _EnvVarLoader(yaml.SafeLoader):
    """YAML loader that substitutes ${ENV_VAR} patterns in all strings.

    Set the class-level ``strict`` attribute to ``True`` before loading
    to raise on unresolved env-var placeholders.
    """
    strict: bool = False


def _env_str_constructor(loader, node):
    """Implicitly substitute env vars in all scalar strings."""
    value = loader.construct_scalar(node)
    if isinstance(value, str) and '${' in value:
        strict = getattr(loader, 'strict', False)
        return _substitute_env_vars(value, strict=strict)
    return value


_EnvVarLoader.add_constructor('!env', _env_constructor)
_EnvVarLoader.add_implicit_resolver(
    'tag:yaml.org,2002:str',
    re.compile(r'.*\$\{.*\}.*'),
    None
)
_EnvVarLoader.add_constructor('tag:yaml.org,2002:str', _env_str_constructor)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {'; '.join(errors)}")


# Keys that are commonly expected at the top level of config files.
# A warning (not an error) is emitted when these are absent.
_COMMON_EXPECTED_KEYS: List[str] = [
    'version',
    'logging',
]


def _validate_required_keys(config: dict, required: List[str], prefix: str = "") -> List[str]:
    """Check that all *required* keys exist in the config dict.

    Args:
        config: The parsed configuration dictionary.
        required: Dot-separated key paths that must be present
                  (e.g. ``["database.host", "database.port"]``).
        prefix: Internal prefix used for nested error messages.

    Returns:
        A list of human-readable error strings for every missing key.
        An empty list means all required keys are present.

    Example::

        errors = _validate_required_keys(
            config,
            required=["database.host", "database.port", "api.key"],
        )
        if errors:
            raise ConfigValidationError(errors)
    """
    errors = []
    for key in required:
        parts = key.split('.')
        current = config
        path = prefix
        for part in parts:
            path = f"{path}.{part}" if path else part
            if not isinstance(current, dict) or part not in current:
                errors.append(f"Missing required key: {path}")
                break
            current = current[part]
    return errors


def _warn_common_keys(config: dict, config_name: str) -> None:
    """Emit warnings for commonly expected top-level keys that are absent."""
    for key in _COMMON_EXPECTED_KEYS:
        if key not in config:
            logger.warning(
                f"Config '{config_name}' is missing commonly expected key '{key}'. "
                f"Consider adding it for consistency."
            )


# ---------------------------------------------------------------------------
# ConfigLoader singleton
# ---------------------------------------------------------------------------

class ConfigLoader:
    """
    Central configuration loader with caching and env-var substitution.

    Features:
    - Loads YAML files from the project root (or specified directory)
    - Substitutes ${ENV_VAR} and ${ENV_VAR:default} patterns
    - Caches loaded configs (thread-safe)
    - Dot-notation access to nested keys
    - Optional schema validation (required keys)
    """

    _instance: Optional['ConfigLoader'] = None
    _lock = threading.Lock()

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self._config_dir = Path(config_dir)
        else:
            self._config_dir = Path(__file__).parent
        self._cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def instance(cls, config_dir: Optional[str] = None) -> 'ConfigLoader':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_dir)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def load(
        self,
        name: str,
        required_keys: Optional[List[str]] = None,
        strict: bool = False,
    ) -> dict:
        """
        Load a YAML config file by name (without .yaml extension).

        Args:
            name: Config file name (e.g., "memory_config", "cpel_config")
            required_keys: Optional list of dot-separated keys that must exist.
                Use :func:`_validate_required_keys` for programmatic validation.
            strict: When True, raise ``UnresolvedEnvVarError`` if any
                ``${VAR}`` placeholder cannot be resolved from the
                environment and has no default value.

        Returns:
            Parsed config dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If required keys are missing
            UnresolvedEnvVarError: If *strict* is True and env vars are unresolved
        """
        with self._cache_lock:
            if name in self._cache:
                return deepcopy(self._cache[name])

        config = self._load_file(name, strict=strict)

        # Warn about commonly expected keys that are absent
        _warn_common_keys(config, name)

        if required_keys:
            errors = _validate_required_keys(config, required_keys)
            if errors:
                raise ConfigValidationError(errors)

        with self._cache_lock:
            self._cache[name] = config

        return deepcopy(config)

    def _load_file(self, name: str, strict: bool = False) -> dict:
        """Load and parse a YAML file.

        Args:
            name: Config file stem (no extension).
            strict: When True the YAML loader will raise on unresolved
                ``${VAR}`` env-var placeholders.
        """
        # Try multiple locations
        candidates = [
            self._config_dir / f"{name}.yaml",
            self._config_dir / f"{name}.yml",
            self._config_dir / "config" / f"{name}.yaml",
            self._config_dir / "config" / f"{name}.yml",
        ]

        for path in candidates:
            if path.exists():
                logger.debug(f"Loading config from {path}")
                # Temporarily set strict flag on the loader class
                prev_strict = _EnvVarLoader.strict
                _EnvVarLoader.strict = strict
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.load(f, Loader=_EnvVarLoader)
                finally:
                    _EnvVarLoader.strict = prev_strict
                if data is None:
                    data = {}
                return data

        raise FileNotFoundError(
            f"Config '{name}' not found. Searched: {[str(p) for p in candidates]}"
        )

    def get(self, name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific value from a config using dot-notation.

        Args:
            name: Config file name
            key_path: Dot-separated path (e.g., "embedding.dimension")
            default: Default value if key not found

        Returns:
            The config value, or default if not found
        """
        try:
            config = self.load(name)
        except FileNotFoundError:
            logger.warning(f"Config file '{name}' not found, returning default")
            return default

        return self._get_nested(config, key_path, default)

    def get_section(self, name: str, section: str) -> dict:
        """
        Get an entire section from a config.

        Args:
            name: Config file name
            section: Dot-separated section path

        Returns:
            The section dictionary, or empty dict if not found
        """
        result = self.get(name, section, {})
        if not isinstance(result, dict):
            return {}
        return result

    def reload(self, name: Optional[str] = None):
        """
        Reload config(s) from disk, clearing cache.

        Args:
            name: Specific config to reload, or None to reload all
        """
        with self._cache_lock:
            if name:
                self._cache.pop(name, None)
            else:
                self._cache.clear()
        logger.info(f"Config cache cleared: {name or 'all'}")

    def list_configs(self) -> List[str]:
        """List all available YAML config files."""
        configs = set()
        for pattern in ['*.yaml', '*.yml']:
            for path in self._config_dir.glob(pattern):
                configs.add(path.stem)
            config_subdir = self._config_dir / 'config'
            if config_subdir.exists():
                for path in config_subdir.glob(pattern):
                    configs.add(path.stem)
        return sorted(configs)

    @staticmethod
    def _get_nested(data: dict, key_path: str, default: Any = None) -> Any:
        """Navigate nested dict using dot-separated key path."""
        current = data
        for part in key_path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __repr__(self):
        cached = list(self._cache.keys())
        return f"ConfigLoader(dir={self._config_dir}, cached={cached})"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_config(
    name: str,
    key_path: Optional[str] = None,
    default: Any = None,
    strict: bool = False,
) -> Any:
    """
    Convenience function to get config values.

    Args:
        name: Config file name (e.g., "memory_config")
        key_path: Optional dot-separated path (e.g., "embedding.dimension")
        default: Default value if not found
        strict: When True, raise ``UnresolvedEnvVarError`` for any
            ``${VAR}`` placeholder that cannot be resolved.

    Returns:
        Full config dict (if no key_path) or specific value

    Examples:
        # Get full config
        cfg = get_config("memory_config")

        # Get nested value
        dim = get_config("memory_config", "embedding.dimension", 1536)

        # Get section
        search = get_config("memory_config", "search")

        # Strict mode – blow up on unresolved env vars
        cfg = get_config("memory_config", strict=True)
    """
    loader = ConfigLoader.instance()
    if key_path is None:
        try:
            return loader.load(name, strict=strict)
        except FileNotFoundError:
            return default if default is not None else {}
    return loader.get(name, key_path, default)


def get_section(name: str, section: str) -> dict:
    """Convenience function to get a config section."""
    return ConfigLoader.instance().get_section(name, section)


def reload_config(name: Optional[str] = None):
    """Convenience function to reload configs."""
    ConfigLoader.instance().reload(name)
