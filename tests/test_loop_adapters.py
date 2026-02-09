"""Unit tests for loop_adapters.py"""

import asyncio
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSafeRun:
    def test_safe_run_success(self):
        from loop_adapters import _safe_run
        result = _safe_run(
            'test_loop',
            lambda: "instance",
            lambda inst: {"data": "ok"},
        )
        assert result["loop"] == "test_loop"
        assert result["success"] is True
        assert result["result"]["data"] == "ok"

    def test_safe_run_catches_exception(self):
        from loop_adapters import _safe_run
        result = _safe_run(
            'test_loop',
            lambda: "instance",
            lambda inst: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert result["loop"] == "test_loop"
        assert result["success"] is False
        assert "boom" in result["error"]

    def test_safe_run_getter_failure(self):
        from loop_adapters import _safe_run

        def bad_getter():
            raise ImportError("module not found")

        result = _safe_run('test_loop', bad_getter, lambda inst: {})
        assert result["success"] is False
        assert "module not found" in result["error"]


class TestRunAsync:
    def test_run_async_with_coroutine(self):
        from loop_adapters import _run_async

        async def sample_coro():
            return {"value": 42}

        result = _run_async(sample_coro())
        assert result["value"] == 42

    def test_run_async_timeout(self):
        from loop_adapters import _run_async, _DEFAULT_TIMEOUT
        import loop_adapters

        # Temporarily set a very short timeout
        original = loop_adapters._DEFAULT_TIMEOUT
        loop_adapters._DEFAULT_TIMEOUT = 0.1

        async def slow_coro():
            await asyncio.sleep(10)
            return {}

        try:
            with pytest.raises(asyncio.TimeoutError):
                _run_async(slow_coro())
        finally:
            loop_adapters._DEFAULT_TIMEOUT = original


class TestGetLoopHandlers:
    def test_handlers_dict_returned(self):
        from loop_adapters import get_loop_handlers
        handlers = get_loop_handlers(llm_client=None)
        assert isinstance(handlers, dict)
        assert len(handlers) >= 15  # At least 15 cognitive loops + debugging
        assert 'loop.ralph.run_cycle' in handlers
        assert 'loop.debugging.run_cycle' in handlers

    def test_all_handlers_callable(self):
        from loop_adapters import get_loop_handlers
        handlers = get_loop_handlers(llm_client=None)
        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} is not callable"
