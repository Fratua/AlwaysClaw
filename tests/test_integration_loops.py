"""Integration tests for loop initialization and lifecycle."""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRalphLoop:
    """Test Ralph Loop initialization with YAML config."""

    def test_init_with_yaml(self):
        from ralph_loop_implementation import RalphLoop
        loop = RalphLoop()
        assert loop.config is not None
        assert hasattr(loop, 'layer_manager')
        assert hasattr(loop, 'priority_queue')

    def test_init_with_explicit_config(self):
        from ralph_loop_implementation import RalphLoop
        config = {'layers': {}, 'storage': {'db_path': ':memory:'}}
        loop = RalphLoop(config=config)
        assert loop.config == config

    def test_layers_initialized(self):
        from ralph_loop_implementation import RalphLoop
        loop = RalphLoop()
        assert len(loop.layer_manager.layers) == 7


class TestCPELLoop:
    """Test Context Prompt Engineering Loop initialization."""

    def test_init_loads_config(self):
        from cpel_implementation import ContextPromptEngineeringLoop
        loop = ContextPromptEngineeringLoop()
        assert loop.config is not None
        assert hasattr(loop, 'registry')

    def test_init_with_explicit_config(self):
        from cpel_implementation import ContextPromptEngineeringLoop
        config = {'test': True}
        loop = ContextPromptEngineeringLoop(config=config)
        assert loop.config == config


class TestContextEngineeringLoop:
    """Test Context Engineering Loop initialization."""

    def test_init_loads_yaml(self):
        from context_engineering_loop import ContextEngineeringLoop
        loop = ContextEngineeringLoop()
        assert loop.config is not None
        assert hasattr(loop, 'monitor')


class TestPlanningLoop:
    """Test Planning Loop initialization."""

    def test_init_loads_yaml(self):
        from planning_loop_implementation import AdvancedPlanningLoop
        loop = AdvancedPlanningLoop()
        assert loop.config is not None
        assert hasattr(loop, 'goal_manager')


class TestExplorationLoop:
    """Test Exploration Loop initialization."""

    def test_init_loads_config(self):
        from exploration_loop_implementation import ExplorationLoop
        loop = ExplorationLoop()
        assert loop.config is not None
        assert loop.state == "initialized"


class TestSelfUpgradingLoop:
    """Test Self-Upgrading Loop initialization."""

    def test_init_loads_yaml(self):
        from self_upgrading_loop import UpgradeOrchestrator
        orch = UpgradeOrchestrator()
        assert orch.config is not None
        assert hasattr(orch, 'gap_analyzer')


class TestAgentLoopsConfig:
    """Test agent loops configuration loading."""

    def test_all_configs_loaded(self):
        from agent_loops_config import AGENT_LOOP_CONFIGS, COGNITIVE_LOOP_CONFIGS
        assert len(AGENT_LOOP_CONFIGS) > 0
        assert len(COGNITIVE_LOOP_CONFIGS) > 0

    def test_cognitive_loops_have_required_fields(self):
        from agent_loops_config import COGNITIVE_LOOP_CONFIGS
        for name, config in COGNITIVE_LOOP_CONFIGS.items():
            assert 'interval' in config, f"{name} missing interval"
            assert 'description' in config, f"{name} missing description"

    def test_get_cognitive_loop_config(self):
        from agent_loops_config import get_cognitive_loop_config
        config = get_cognitive_loop_config('ralph')
        assert config is not None
        assert 'interval' in config


class TestDeploymentOrchestrator:
    """Test deployment orchestrator initialization."""

    def test_import(self):
        from deployment_orchestrator import (
            BlueGreenDeployer, CanaryDeployer, RollingUpdateDeployer,
            DeploymentState
        )

    def test_blue_green_init(self):
        from deployment_orchestrator import BlueGreenDeployer
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config", "future", "blue-green-config.yaml")
        deployer = BlueGreenDeployer(config_path)
        assert deployer is not None

    def test_canary_init(self):
        from deployment_orchestrator import CanaryDeployer
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config", "future", "canary-config.yaml")
        deployer = CanaryDeployer(config_path)
        assert deployer is not None

    def test_rolling_update_init(self):
        from deployment_orchestrator import RollingUpdateDeployer
        deployer = RollingUpdateDeployer("rolling-update-config.yaml")
        assert deployer is not None


class TestHealthCheck:
    """Test system health check."""

    def test_run_health_check(self):
        from health_check import SystemHealthCheck, HealthStatus
        checker = SystemHealthCheck()
        report = checker.run_all()
        assert report.overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert len(report.components) > 0

    def test_health_report_serialization(self):
        from health_check import SystemHealthCheck
        checker = SystemHealthCheck()
        report = checker.run_all()
        d = report.to_dict()
        assert 'overall_status' in d
        assert 'components' in d
        assert len(d['components']) > 0

    def test_health_report_log_string(self):
        from health_check import SystemHealthCheck
        checker = SystemHealthCheck()
        report = checker.run_all()
        log_str = report.to_log_string()
        assert "System Health Report" in log_str
