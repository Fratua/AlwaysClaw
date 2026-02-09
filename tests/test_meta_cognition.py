"""Tests for meta_cognition_loop metric calculations."""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class MockStep:
    """Mock reasoning step for testing."""
    content: str = ""
    confidence: float = 0.5
    step_type: str = "reasoning"
    validation_status: str = "validated"
    time_taken_ms: int = 100


@dataclass
class MockTrace:
    """Mock reasoning trace for testing."""
    trace_id: str = "test-trace"
    steps: List[MockStep] = field(default_factory=list)
    used_tools: List[str] = field(default_factory=list)
    working_memory: List[str] = field(default_factory=list)
    revision_count: int = 0
    backtrack_count: int = 0


class TestHelpfulnessAssessment:
    """Test _assess_helpfulness method."""

    def _get_monitor(self):
        """Get CognitivePerformanceMonitor instance."""
        from meta_cognition_loop import CognitivePerformanceMonitor
        config = MagicMock()
        config.baseline_establishment_tasks = 10
        config.performance_window_size = 100
        return CognitivePerformanceMonitor(config)

    def test_empty_trace(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[])
        score = monitor._assess_helpfulness(trace)
        assert score == 0.5

    def test_trace_with_tools(self):
        monitor = self._get_monitor()
        trace = MockTrace(
            steps=[MockStep(confidence=0.8), MockStep(confidence=0.9)],
            used_tools=["search", "calculator"]
        )
        score = monitor._assess_helpfulness(trace)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Tools used + high confidence = helpful

    def test_trace_with_increasing_confidence(self):
        monitor = self._get_monitor()
        trace = MockTrace(
            steps=[MockStep(confidence=0.3), MockStep(confidence=0.9)],
        )
        score = monitor._assess_helpfulness(trace)
        assert score > 0.5


class TestBiasDetectionRate:
    """Test _calculate_bias_detection_rate method."""

    def _get_monitor(self):
        from meta_cognition_loop import CognitivePerformanceMonitor
        config = MagicMock()
        config.baseline_establishment_tasks = 10
        config.performance_window_size = 100
        return CognitivePerformanceMonitor(config)

    def test_no_evidence_steps(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[MockStep(content="just a plain step")])
        rate = monitor._calculate_bias_detection_rate(trace)
        assert rate == 0.5

    def test_evidence_with_counter(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[
            MockStep(content="The evidence suggests X is true"),
            MockStep(content="The data indicates Y, however there are limitations"),
        ])
        rate = monitor._calculate_bias_detection_rate(trace)
        assert rate > 0.0

    def test_evidence_without_counter(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[
            MockStep(content="The evidence suggests X"),
            MockStep(content="The data indicates Y"),
        ])
        rate = monitor._calculate_bias_detection_rate(trace)
        assert rate == 0.0  # No counter-evidence addressed

    def test_empty_trace(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[])
        rate = monitor._calculate_bias_detection_rate(trace)
        assert rate == 0.5


class TestLearningTransfer:
    """Test _calculate_learning_transfer method."""

    def _get_monitor(self):
        from meta_cognition_loop import CognitivePerformanceMonitor
        config = MagicMock()
        config.baseline_establishment_tasks = 10
        config.performance_window_size = 100
        return CognitivePerformanceMonitor(config)

    def test_no_retrieval_steps(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[
            MockStep(content="regular step", confidence=0.5),
            MockStep(content="another step", confidence=0.6),
        ])
        score = monitor._calculate_learning_transfer(trace)
        assert score == 0.5

    def test_retrieval_with_improvement(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[
            MockStep(content="I retrieved the relevant info", step_type="retrieval", confidence=0.4),
            MockStep(content="Now I understand better", confidence=0.8),
        ])
        score = monitor._calculate_learning_transfer(trace)
        assert score > 0.5

    def test_single_step(self):
        monitor = self._get_monitor()
        trace = MockTrace(steps=[MockStep()])
        score = monitor._calculate_learning_transfer(trace)
        assert score == 0.5
