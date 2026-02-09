"""
Loop Adapters - Standardized entry points for all 16 Python cognitive loops.
Each adapter lazily initializes its loop class and provides a run_cycle() method
that can be called from the Python bridge.

All loops share the same OpenAIClient singleton (GPT-5.2) and have direct
access to the memory database (same Python process, no bridge round-trip).
"""

import asyncio
import atexit
import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Gauge, Histogram

    LOOPS_RUNNING = Gauge(
        'openclaw_agent_loops_running',
        'Number of agent loops currently running',
    )
    LOOP_STATUS = Gauge(
        'openclaw_loop_status',
        'Status of each loop (1=ok, 0=error)',
        ['loop'],
    )
    LOOP_CYCLE_DURATION = Histogram(
        'openclaw_loop_cycle_duration_seconds',
        'Duration of loop cycle execution',
        ['loop'],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    )
    LOOP_RESTARTS = Counter(
        'openclaw_loop_restarts_total',
        'Total number of loop restarts',
        ['loop'],
    )
    GPT_RESPONSE_DURATION = Histogram(
        'openclaw_gpt_response_duration_seconds',
        'Duration of GPT-5.2 API calls',
        buckets=[0.5, 1, 2, 3, 5, 10, 20, 30, 60],
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    logger.debug("prometheus_client not available, metrics disabled")

# ── Singleton loop instances ──────────────────────────────────────────

_loop_instances: Dict[str, Any] = {}
_llm_client = None
_thread_pool = None


def _get_llm():
    """Get or create the shared OpenAI GPT-5.2 client."""
    global _llm_client
    if _llm_client is None:
        from openai_client import OpenAIClient
        _llm_client = OpenAIClient.get_instance()
    return _llm_client


_LOOP_TIMEOUTS = {
    'ralph': 300,
    'research': 600,
    'planning': 300,
    'e2e': 600,
    'exploration': 600,
    'discovery': 300,
    'bug_finder': 300,
    'self_learning': 300,
    'meta_cognition': 300,
    'self_upgrading': 900,
    'self_updating': 300,
    'self_driven': 300,
    'cpel': 300,
    'context_engineering': 300,
    'debugging': 300,
    'web_monitor': 300,
}

_DEFAULT_TIMEOUT = 300  # 5 minutes fallback


def _get_timeout(loop_name: str) -> int:
    """Get the timeout for a specific loop."""
    return _LOOP_TIMEOUTS.get(loop_name, _DEFAULT_TIMEOUT)


def _run_async(coro, timeout: int = None):
    """Run an async coroutine synchronously (bridge handlers are sync).
    Enforces a per-loop timeout to prevent hung loops."""
    global _thread_pool
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        if _thread_pool is None:
            import concurrent.futures
            _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        future = _thread_pool.submit(asyncio.run, asyncio.wait_for(coro, timeout=timeout))
        return future.result(timeout=timeout + 5)
    else:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


atexit.register(lambda: _thread_pool.shutdown(wait=False) if _thread_pool else None)


# ── Individual loop adapters ──────────────────────────────────────────

def _get_ralph_loop():
    if 'ralph' not in _loop_instances:
        try:
            from ralph_loop_implementation import RalphLoop
            _loop_instances['ralph'] = RalphLoop(config={})
            logger.info("RalphLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize RalphLoop: {e}")
            raise
    return _loop_instances['ralph']


def _get_research_loop():
    if 'research' not in _loop_instances:
        try:
            from research_loop.research_loop import ResearchLoop
            _loop_instances['research'] = ResearchLoop()
            logger.info("ResearchLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize ResearchLoop: {e}")
            raise
    return _loop_instances['research']


def _get_planning_loop():
    if 'planning' not in _loop_instances:
        try:
            from planning_loop_implementation import AdvancedPlanningLoop
            _loop_instances['planning'] = AdvancedPlanningLoop()
            logger.info("AdvancedPlanningLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize AdvancedPlanningLoop: {e}")
            raise
    return _loop_instances['planning']


def _get_e2e_loop():
    if 'e2e' not in _loop_instances:
        try:
            from e2e_loop_core import E2EWorkflowEngine
            _loop_instances['e2e'] = E2EWorkflowEngine()
            logger.info("E2EWorkflowEngine initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize E2EWorkflowEngine: {e}")
            raise
    return _loop_instances['e2e']


def _get_exploration_loop():
    if 'exploration' not in _loop_instances:
        try:
            from exploration_loop_implementation import ExplorationLoop
            _loop_instances['exploration'] = ExplorationLoop()
            logger.info("ExplorationLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize ExplorationLoop: {e}")
            raise
    return _loop_instances['exploration']


def _get_discovery_loop():
    if 'discovery' not in _loop_instances:
        try:
            from discovery_loop_architecture import DiscoveryLoop
            _loop_instances['discovery'] = DiscoveryLoop()
            logger.info("DiscoveryLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize DiscoveryLoop: {e}")
            raise
    return _loop_instances['discovery']


def _get_bug_finder_loop():
    if 'bug_finder' not in _loop_instances:
        try:
            from bug_finder_loop import AdvancedBugFinderLoop
            _loop_instances['bug_finder'] = AdvancedBugFinderLoop()
            logger.info("AdvancedBugFinderLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize AdvancedBugFinderLoop: {e}")
            raise
    return _loop_instances['bug_finder']


def _get_self_learning_loop():
    if 'self_learning' not in _loop_instances:
        try:
            from self_learning_loop_implementation import SelfLearningLoop
            _loop_instances['self_learning'] = SelfLearningLoop()
            logger.info("SelfLearningLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize SelfLearningLoop: {e}")
            raise
    return _loop_instances['self_learning']


def _get_meta_cognition_loop():
    if 'meta_cognition' not in _loop_instances:
        try:
            from meta_cognition_loop import MetaCognitionLoop
            _loop_instances['meta_cognition'] = MetaCognitionLoop()
            logger.info("MetaCognitionLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize MetaCognitionLoop: {e}")
            raise
    return _loop_instances['meta_cognition']


def _get_self_upgrading_loop():
    if 'self_upgrading' not in _loop_instances:
        try:
            from self_upgrading_loop import UpgradeOrchestrator
            _loop_instances['self_upgrading'] = UpgradeOrchestrator()
            logger.info("UpgradeOrchestrator initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize UpgradeOrchestrator: {e}")
            raise
    return _loop_instances['self_upgrading']


def _get_self_updating_loop():
    if 'self_updating' not in _loop_instances:
        try:
            from self_updating_loop.loop import SelfUpdatingLoop
            _loop_instances['self_updating'] = SelfUpdatingLoop()
            logger.info("SelfUpdatingLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize SelfUpdatingLoop: {e}")
            raise
    return _loop_instances['self_updating']


def _get_self_driven_loop():
    if 'self_driven' not in _loop_instances:
        try:
            from self_driven_loop.self_driven_loop import SelfDrivenLoop
            _loop_instances['self_driven'] = SelfDrivenLoop()
            logger.info("SelfDrivenLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize SelfDrivenLoop: {e}")
            raise
    return _loop_instances['self_driven']


def _get_cpel_loop():
    if 'cpel' not in _loop_instances:
        try:
            from cpel_implementation import ContextPromptEngineeringLoop
            _loop_instances['cpel'] = ContextPromptEngineeringLoop()
            logger.info("ContextPromptEngineeringLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize ContextPromptEngineeringLoop: {e}")
            raise
    return _loop_instances['cpel']


def _get_context_engineering_loop():
    if 'context_engineering' not in _loop_instances:
        try:
            from context_engineering_loop import ContextEngineeringLoop
            _loop_instances['context_engineering'] = ContextEngineeringLoop()
            logger.info("ContextEngineeringLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize ContextEngineeringLoop: {e}")
            raise
    return _loop_instances['context_engineering']


def _get_debugging_loop():
    if 'debugging' not in _loop_instances:
        try:
            from debugging_loop import DebuggingLoop
            _loop_instances['debugging'] = DebuggingLoop(llm_client=_get_llm())
            logger.info("DebuggingLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize DebuggingLoop: {e}")
            raise
    return _loop_instances['debugging']


def _get_web_monitor_loop():
    if 'web_monitor' not in _loop_instances:
        try:
            from web_monitor_agent_loop import WebMonitoringAgentLoop

            class MinimalAgentCore:
                """Minimal agent core for web monitor when no full core available."""
                def __init__(self):
                    self.name = "openclaw-agent"
                    self.config = {}

                async def process(self, prompt: str = "", **kwargs):
                    try:
                        llm = _get_llm()
                        result = llm.generate(prompt or "Analyze the current state.")
                        return {"status": "ok", "content": result}
                    except (EnvironmentError, RuntimeError) as e:
                        logger.debug(f"MinimalAgentCore LLM call skipped: {e}")
                        return {"status": "ok"}

            _loop_instances['web_monitor'] = WebMonitoringAgentLoop(agent_core=MinimalAgentCore())
            logger.info("WebMonitoringAgentLoop initialized")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to initialize WebMonitoringAgentLoop: {e}")
            raise
    return _loop_instances['web_monitor']


# ── Cycle handlers (called by bridge) ────────────────────────────────
# Each handler returns a dict with cycle results.
# Failures are caught and returned as error dicts so the bridge stays alive.

def _safe_run(loop_name: str, getter, runner) -> Dict[str, Any]:
    """Safely initialize and run a loop cycle with per-loop timeout."""
    start = time.monotonic()
    try:
        instance = getter()
        result = runner(instance)
        elapsed = time.monotonic() - start
        if _METRICS_AVAILABLE:
            LOOP_CYCLE_DURATION.labels(loop=loop_name).observe(elapsed)
            LOOP_STATUS.labels(loop=loop_name).set(1)
        return {
            "loop": loop_name,
            "success": True,
            "result": result if isinstance(result, dict) else {"output": str(result)},
        }
    except (ImportError, RuntimeError, OSError, ValueError, KeyError) as e:
        elapsed = time.monotonic() - start
        if _METRICS_AVAILABLE:
            LOOP_CYCLE_DURATION.labels(loop=loop_name).observe(elapsed)
            LOOP_STATUS.labels(loop=loop_name).set(0)
            LOOP_RESTARTS.labels(loop=loop_name).inc()
        logger.error(f"[{loop_name}] Cycle error: {e}\n{traceback.format_exc()}")
        return {
            "loop": loop_name,
            "success": False,
            "error": str(e),
        }


def _loop_run_async(loop_name: str, coro):
    """Run an async coroutine with the timeout configured for the given loop."""
    return _run_async(coro, timeout=_get_timeout(loop_name))


def run_ralph_cycle(**context) -> Dict:
    return _safe_run('ralph', _get_ralph_loop,
                     lambda loop: _loop_run_async('ralph', loop.run_single_cycle(context)))


def run_research_cycle(**context) -> Dict:
    return _safe_run('research', _get_research_loop,
                     lambda loop: _loop_run_async('research', loop.run_single_cycle(context)))


def run_planning_cycle(**context) -> Dict:
    return _safe_run('planning', _get_planning_loop,
                     lambda loop: _loop_run_async('planning', loop.execute_goal(context.get('goal', 'system maintenance check'))))


def run_e2e_cycle(**context) -> Dict:
    def _run_e2e(loop):
        pending = len(getattr(loop, '_running_workflows', {}))
        # Process any pending workflows
        if hasattr(loop, 'state_backend'):
            try:
                workflows = _loop_run_async('e2e', loop.state_backend.list_workflows(limit=10))
                active = [w for w in workflows if w.status.value in ('pending', 'running')]
                return {
                    "status": "ready",
                    "engine": "E2EWorkflowEngine",
                    "running_workflows": pending,
                    "active_workflows": len(active),
                    "total_workflows": len(workflows),
                }
            except (RuntimeError, OSError, ValueError) as e:
                logger.debug(f"E2E state query: {e}")
        return {
            "status": "ready",
            "engine": "E2EWorkflowEngine",
            "running_workflows": pending,
        }
    return _safe_run('e2e', _get_e2e_loop, _run_e2e)


def run_exploration_cycle(**context) -> Dict:
    return _safe_run('exploration', _get_exploration_loop,
                     lambda loop: _loop_run_async('exploration', loop.run_single_cycle(context)))


def run_discovery_cycle(**context) -> Dict:
    return _safe_run('discovery', _get_discovery_loop,
                     lambda loop: _loop_run_async('discovery', loop.run_single_cycle(context)))


def run_bug_finder_cycle(**context) -> Dict:
    async def _default_bug_finder_source():
        import psutil
        return {
            'system_metrics': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage(os.environ.get('SYSTEMDRIVE', 'C:\\')).percent,
            }
        }

    return _safe_run('bug_finder', _get_bug_finder_loop,
                     lambda loop: _loop_run_async('bug_finder', loop._detection_cycle(_default_bug_finder_source)))


def run_self_learning_cycle(**context) -> Dict:
    from datetime import timedelta
    return _safe_run('self_learning', _get_self_learning_loop,
                     lambda loop: _loop_run_async('self_learning',
                         loop.run_consolidation_cycle(timedelta(minutes=10))))


def run_meta_cognition_cycle(**context) -> Dict:
    return _safe_run('meta_cognition', _get_meta_cognition_loop,
                     lambda loop: _loop_run_async('meta_cognition', loop.execute_cycle(context=context)))


def run_self_upgrading_cycle(**context) -> Dict:
    return _safe_run('self_upgrading', _get_self_upgrading_loop,
                     lambda loop: _loop_run_async('self_upgrading', loop.run_upgrade_cycle()))


def run_self_updating_cycle(**context) -> Dict:
    return _safe_run('self_updating', _get_self_updating_loop,
                     lambda loop: _loop_run_async('self_updating', loop.run_single_cycle(context)))


def run_self_driven_cycle(**context) -> Dict:
    return _safe_run('self_driven', _get_self_driven_loop,
                     lambda loop: _loop_run_async('self_driven', loop._loop_iteration()))


def run_cpel_cycle(**context) -> Dict:
    return _safe_run('cpel', _get_cpel_loop,
                     lambda loop: _loop_run_async('cpel', loop.get_optimized_prompt(
                         user_input=context.get('user_input', 'system check'),
                         system_state=context,
                     )))


def run_context_engineering_cycle(**context) -> Dict:
    return _safe_run('context_engineering', _get_context_engineering_loop,
                     lambda loop: _loop_run_async('context_engineering', loop.run_single_cycle(context)))


def run_debugging_cycle(**context) -> Dict:
    return _safe_run('debugging', _get_debugging_loop,
                     lambda loop: _loop_run_async('debugging', loop.run_single_cycle(context)))


def run_web_monitor_cycle(**context) -> Dict:
    return _safe_run('web_monitor', _get_web_monitor_loop,
                     lambda loop: _loop_run_async('web_monitor', loop.run_single_cycle(context)))


# ── Public API ────────────────────────────────────────────────────────

def get_loop_handlers(llm_client=None) -> Dict[str, Callable]:
    """
    Return a dict mapping bridge method names to handler functions.
    Called by python_bridge.py during startup.
    """
    global _llm_client
    if llm_client is not None:
        _llm_client = llm_client

    handlers = {
        'loop.ralph.run_cycle': run_ralph_cycle,
        'loop.research.run_cycle': run_research_cycle,
        'loop.planning.run_cycle': run_planning_cycle,
        'loop.e2e.run_cycle': run_e2e_cycle,
        'loop.exploration.run_cycle': run_exploration_cycle,
        'loop.discovery.run_cycle': run_discovery_cycle,
        'loop.bug_finder.run_cycle': run_bug_finder_cycle,
        'loop.self_learning.run_cycle': run_self_learning_cycle,
        'loop.meta_cognition.run_cycle': run_meta_cognition_cycle,
        'loop.self_upgrading.run_cycle': run_self_upgrading_cycle,
        'loop.self_updating.run_cycle': run_self_updating_cycle,
        'loop.self_driven.run_cycle': run_self_driven_cycle,
        'loop.cpel.run_cycle': run_cpel_cycle,
        'loop.context_engineering.run_cycle': run_context_engineering_cycle,
        'loop.debugging.run_cycle': run_debugging_cycle,
        'loop.web_monitor.run_cycle': run_web_monitor_cycle,
    }

    if _METRICS_AVAILABLE:
        LOOPS_RUNNING.set(len(handlers))

    logger.info(f"Loop adapters: {len(handlers)} handlers prepared")
    return handlers
