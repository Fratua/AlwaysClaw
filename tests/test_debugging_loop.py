"""Tests for the debugging loop."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from debugging_loop import DebuggingLoop, DetectedError, GeneratedFix


class TestErrorClassification:
    """Test error pattern classification."""

    def setup_method(self):
        self.loop = DebuggingLoop()

    def test_classify_import_error(self):
        error = DetectedError(
            error_id="e1", source="test.py",
            error_type="ImportError",
            message="No module named 'foo'"
        )
        assert self.loop._classify_error(error) == 'import'

    def test_classify_attribute_error(self):
        error = DetectedError(
            error_id="e2", source="test.py",
            error_type="AttributeError",
            message="'NoneType' has no attribute 'bar'"
        )
        assert self.loop._classify_error(error) == 'attribute'

    def test_classify_type_error(self):
        error = DetectedError(
            error_id="e3", source="test.py",
            error_type="TypeError",
            message="unsupported operand type(s)"
        )
        assert self.loop._classify_error(error) == 'type'

    def test_classify_value_error(self):
        error = DetectedError(
            error_id="e4", source="test.py",
            error_type="ValueError",
            message="invalid literal for int()"
        )
        assert self.loop._classify_error(error) == 'value'

    def test_classify_key_error(self):
        error = DetectedError(
            error_id="e5", source="test.py",
            error_type="KeyError",
            message="'missing_key'"
        )
        assert self.loop._classify_error(error) == 'key'

    def test_classify_io_error(self):
        error = DetectedError(
            error_id="e6", source="test.py",
            error_type="FileNotFoundError",
            message="No such file or directory"
        )
        assert self.loop._classify_error(error) == 'io'

    def test_classify_unknown(self):
        error = DetectedError(
            error_id="e7", source="test.py",
            error_type="CustomError",
            message="something unusual"
        )
        assert self.loop._classify_error(error) == 'unknown'


class TestFixValidation:
    """Test fix validation."""

    def setup_method(self):
        self.loop = DebuggingLoop()

    def test_valid_python(self):
        assert self.loop._validate_fix("x = 1 + 2") is True

    def test_invalid_python(self):
        assert self.loop._validate_fix("def foo(") is False

    def test_empty_string(self):
        assert self.loop._validate_fix("") is True


class TestErrorDeduplication:
    """Test error deduplication."""

    def setup_method(self):
        self.loop = DebuggingLoop()

    def test_dedup_removes_seen(self):
        error1 = DetectedError(
            error_id="e1", source="a.py",
            error_type="ValueError", message="duplicate message"
        )
        self.loop.error_history.append(error1)

        error2 = DetectedError(
            error_id="e2", source="b.py",
            error_type="ValueError", message="duplicate message"
        )
        result = self.loop._deduplicate_errors([error2])
        assert len(result) == 0

    def test_dedup_keeps_new(self):
        error = DetectedError(
            error_id="e3", source="c.py",
            error_type="ValueError", message="brand new error"
        )
        result = self.loop._deduplicate_errors([error])
        assert len(result) == 1


class TestFixGeneration:
    """Test fix generation with mocked LLM."""

    @pytest.mark.asyncio
    async def test_generate_fix_no_llm(self):
        loop = DebuggingLoop(llm_client=None)
        error = DetectedError(
            error_id="e1", source="test.py",
            error_type="ValueError", message="test error"
        )
        fix = await loop._generate_fix(error)
        assert fix is not None
        assert fix.confidence == 0.0
        assert "Manual review" in fix.description

    @pytest.mark.asyncio
    async def test_generate_fix_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Fix: Change line 42 to use try/except"
        loop = DebuggingLoop(llm_client=mock_llm)
        error = DetectedError(
            error_id="e1", source="test.py",
            error_type="ValueError", message="test error",
            severity="high"
        )
        fix = await loop._generate_fix(error)
        assert fix is not None
        assert fix.confidence > 0
        assert mock_llm.generate.called


class TestRunCycle:
    """Test run_single_cycle."""

    @pytest.mark.asyncio
    async def test_cycle_with_errors(self):
        loop = DebuggingLoop()
        context = {
            'recent_exceptions': [
                {
                    'source': 'test.py',
                    'type': 'ValueError',
                    'message': 'test cycle error',
                    'severity': 'high',
                }
            ]
        }
        result = await loop.run_single_cycle(context)
        assert result['errors_detected'] >= 1
        assert 'cycle_time_ms' in result

    @pytest.mark.asyncio
    async def test_cycle_empty(self):
        loop = DebuggingLoop()
        result = await loop.run_single_cycle({})
        assert result['errors_detected'] >= 0
        assert 'cycle_time_ms' in result
