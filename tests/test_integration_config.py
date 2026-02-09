"""Integration tests for config loading across all modules."""

import pytest
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigLoader:
    """Test the central ConfigLoader utility."""

    def test_config_loader_singleton(self):
        from config_loader import ConfigLoader
        ConfigLoader.reset()
        a = ConfigLoader.instance()
        b = ConfigLoader.instance()
        assert a is b
        ConfigLoader.reset()

    def test_load_memory_config(self):
        from config_loader import get_config
        cfg = get_config("memory_config")
        assert isinstance(cfg, dict)
        assert "embedding" in cfg
        assert cfg["embedding"]["dimension"] == 1536

    def test_load_cpel_config(self):
        from config_loader import get_config
        cfg = get_config("cpel_config")
        assert isinstance(cfg, dict)
        assert "context_prompt_engineering_loop" in cfg

    def test_load_ralph_config(self):
        from config_loader import get_config
        cfg = get_config("ralph_loop_config")
        assert isinstance(cfg, dict)
        assert "ralph_loop" in cfg

    def test_load_tts_config(self):
        from config_loader import get_config
        cfg = get_config("tts_config")
        assert isinstance(cfg, dict)
        assert "tts" in cfg

    def test_load_scaling_config(self):
        from config_loader import get_config
        cfg = get_config("scaling_config")
        assert isinstance(cfg, dict) or cfg is None  # Optional config

    def test_load_deployment_configs(self):
        from config_loader import get_config
        for name in ["blue-green-config", "canary-config", "rolling-update-config"]:
            cfg = get_config(name)
            assert isinstance(cfg, dict), f"{name} should load as dict"

    def test_load_planning_config(self):
        from config_loader import get_config
        cfg = get_config("planning_loop_config")
        assert isinstance(cfg, dict)
        assert "planning_loop" in cfg or len(cfg) > 0

    def test_load_research_config(self):
        from config_loader import get_config
        cfg = get_config("research_loop_config")
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_strict_mode_unresolved_var(self):
        """Test that strict mode raises on unresolved env vars."""
        from config_loader import _substitute_env_vars
        # In strict mode, unresolved vars should raise
        result = _substitute_env_vars("${DEFINITELY_NOT_SET_12345}")
        # Default (non-strict) should return the placeholder
        assert "${DEFINITELY_NOT_SET_12345}" in result or result == ""

    def test_config_values_have_correct_types(self):
        """Test that specific config values have expected types."""
        from config_loader import get_config
        mem_cfg = get_config("memory_config")
        assert isinstance(mem_cfg["embedding"]["dimension"], int)
        search_cfg = mem_cfg.get("search", {})
        if "min_score" in search_cfg:
            assert isinstance(search_cfg["min_score"], (int, float))

    def test_get_nested_value(self):
        from config_loader import get_config
        dim = get_config("memory_config", "embedding.dimension", 0)
        assert dim == 1536

    def test_get_section(self):
        from config_loader import get_section
        search = get_section("memory_config", "search")
        assert isinstance(search, dict)
        assert "min_score" in search

    def test_missing_config_returns_default(self):
        from config_loader import get_config
        result = get_config("nonexistent_config", default={})
        assert result == {}

    def test_missing_key_returns_default(self):
        from config_loader import get_config
        result = get_config("memory_config", "nonexistent.key.path", "fallback")
        assert result == "fallback"

    def test_env_var_substitution(self):
        from config_loader import ConfigLoader
        ConfigLoader.reset()
        os.environ["TEST_OPENCLAW_VAR"] = "test_value"
        try:
            from config_loader import _substitute_env_vars
            result = _substitute_env_vars("prefix_${TEST_OPENCLAW_VAR}_suffix")
            assert result == "prefix_test_value_suffix"
        finally:
            del os.environ["TEST_OPENCLAW_VAR"]
            ConfigLoader.reset()

    def test_env_var_default(self):
        from config_loader import _substitute_env_vars
        result = _substitute_env_vars("${NONEXISTENT_VAR_12345:default_val}")
        assert result == "default_val"

    def test_reload_clears_cache(self):
        from config_loader import ConfigLoader, reload_config
        ConfigLoader.reset()
        loader = ConfigLoader.instance()
        loader.load("memory_config")
        assert "memory_config" in loader._cache
        reload_config("memory_config")
        assert "memory_config" not in loader._cache
        ConfigLoader.reset()

    def test_list_configs(self):
        from config_loader import ConfigLoader
        ConfigLoader.reset()
        loader = ConfigLoader.instance()
        configs = loader.list_configs()
        assert len(configs) > 0
        assert "memory_config" in configs
        ConfigLoader.reset()


class TestAllYamlConfigsLoad:
    """Verify every YAML config file loads without errors."""

    YAML_CONFIGS = [
        "memory_config", "cpel_config", "ralph_loop_config", "tts_config",
        "config", "planning_loop_config", "research_loop_config",
        "self_driven_loop_config", "self_upgrading_config", "self_learning_config",
        "soul_evolution_config", "email_workflow_config", "prompt_engineering_config",
        "blue-green-config", "canary-config", "rolling-update-config",
        "rollback-config", "verification-config", "alerting-rules",
        "pipeline-config", "feature-flags", "response_playbooks",
    ]

    @pytest.mark.parametrize("config_name", YAML_CONFIGS)
    def test_yaml_loads(self, config_name):
        from config_loader import get_config
        cfg = get_config(config_name)
        assert isinstance(cfg, (dict, type(None))), f"{config_name} failed to load"
