"""
Loop Adapters - Standardized entry points for all 15 Python cognitive loops.
Each adapter lazily initializes its loop class and provides a run_cycle() method
that can be called from the Python bridge.

All loops share the same OpenAIClient singleton (GPT-5.2) and have direct
access to the memory database (same Python process, no bridge round-trip).
"""

import asyncio
import logging
import traceback
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Singleton loop instances ──────────────────────────────────────────

_loop_instances: Dict[str, Any] = {}
_llm_client = None


def _get_llm():
    """Get or create the shared OpenAI GPT-5.2 client."""
    global _llm_client
    if _llm_client is None:
        from openai_client import OpenAIClient
        _llm_client = OpenAIClient.get_instance()
    return _llm_client


def _run_async(coro):
    """Run an async coroutine synchronously (bridge handlers are sync)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already an event loop running, create a new one in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=300)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists yet
        return asyncio.run(coro)


# ── Individual loop adapters ──────────────────────────────────────────

def _get_ralph_loop():
    if 'ralph' not in _loop_instances:
        try:
            from ralph_loop_implementation import RalphLoop
            _loop_instances['ralph'] = RalphLoop(config={})
            logger.info("RalphLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RalphLoop: {e}")
            raise
    return _loop_instances['ralph']


def _get_research_loop():
    if 'research' not in _loop_instances:
        try:
            from research_loop.research_loop import ResearchLoop
            _loop_instances['research'] = ResearchLoop()
            logger.info("ResearchLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ResearchLoop: {e}")
            raise
    return _loop_instances['research']


def _get_planning_loop():
    if 'planning' not in _loop_instances:
        try:
            from planning_loop_implementation import AdvancedPlanningLoop
            _loop_instances['planning'] = AdvancedPlanningLoop()
            logger.info("AdvancedPlanningLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedPlanningLoop: {e}")
            raise
    return _loop_instances['planning']


def _get_e2e_loop():
    if 'e2e' not in _loop_instances:
        try:
            from e2e_loop_core import E2EWorkflowEngine
            _loop_instances['e2e'] = E2EWorkflowEngine()
            logger.info("E2EWorkflowEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize E2EWorkflowEngine: {e}")
            raise
    return _loop_instances['e2e']


def _get_exploration_loop():
    if 'exploration' not in _loop_instances:
        try:
            from exploration_loop_implementation import ExplorationLoop
            _loop_instances['exploration'] = ExplorationLoop()
            logger.info("ExplorationLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ExplorationLoop: {e}")
            raise
    return _loop_instances['exploration']


def _get_discovery_loop():
    if 'discovery' not in _loop_instances:
        try:
            from discovery_loop_architecture import DiscoveryLoop
            _loop_instances['discovery'] = DiscoveryLoop()
            logger.info("DiscoveryLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DiscoveryLoop: {e}")
            raise
    return _loop_instances['discovery']


def _get_bug_finder_loop():
    if 'bug_finder' not in _loop_instances:
        try:
            from bug_finder_loop import AdvancedBugFinderLoop
            _loop_instances['bug_finder'] = AdvancedBugFinderLoop()
            logger.info("AdvancedBugFinderLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedBugFinderLoop: {e}")
            raise
    return _loop_instances['bug_finder']


def _get_self_learning_loop():
    if 'self_learning' not in _loop_instances:
        try:
            from self_learning_loop_implementation import SelfLearningLoop
            _loop_instances['self_learning'] = SelfLearningLoop()
            logger.info("SelfLearningLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SelfLearningLoop: {e}")
            raise
    return _loop_instances['self_learning']


def _get_meta_cognition_loop():
    if 'meta_cognition' not in _loop_instances:
        try:
            from meta_cognition_loop import MetaCognitionLoop
            _loop_instances['meta_cognition'] = MetaCognitionLoop()
            logger.info("MetaCognitionLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MetaCognitionLoop: {e}")
            raise
    return _loop_instances['meta_cognition']


def _get_self_upgrading_loop():
    if 'self_upgrading' not in _loop_instances:
        try:
            from self_upgrading_loop import UpgradeOrchestrator
            _loop_instances['self_upgrading'] = UpgradeOrchestrator()
            logger.info("UpgradeOrchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize UpgradeOrchestrator: {e}")
            raise
    return _loop_instances['self_upgrading']


def _get_self_updating_loop():
    if 'self_updating' not in _loop_instances:
        try:
            from self_updating_loop.loop import SelfUpdatingLoop
            _loop_instances['self_updating'] = SelfUpdatingLoop()
            logger.info("SelfUpdatingLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SelfUpdatingLoop: {e}")
            raise
    return _loop_instances['self_updating']


def _get_self_driven_loop():
    if 'self_driven' not in _loop_instances:
        try:
            from self_driven_loop.self_driven_loop import SelfDrivenLoop
            _loop_instances['self_driven'] = SelfDrivenLoop()
            logger.info("SelfDrivenLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SelfDrivenLoop: {e}")
            raise
    return _loop_instances['self_driven']


def _get_cpel_loop():
    if 'cpel' not in _loop_instances:
        try:
            from cpel_implementation import ContextPromptEngineeringLoop
            _loop_instances['cpel'] = ContextPromptEngineeringLoop()
            logger.info("ContextPromptEngineeringLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ContextPromptEngineeringLoop: {e}")
            raise
    return _loop_instances['cpel']


def _get_context_engineering_loop():
    if 'context_engineering' not in _loop_instances:
        try:
            from context_engineering_loop import ContextEngineeringLoop
            _loop_instances['context_engineering'] = ContextEngineeringLoop()
            logger.info("ContextEngineeringLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ContextEngineeringLoop: {e}")
            raise
    return _loop_instances['context_engineering']


def _get_web_monitor_loop():
    if 'web_monitor' not in _loop_instances:
        try:
            from web_monitor_agent_loop import WebMonitoringAgentLoop
            _loop_instances['web_monitor'] = WebMonitoringAgentLoop(agent_core=None)
            logger.info("WebMonitoringAgentLoop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebMonitoringAgentLoop: {e}")
            raise
    return _loop_instances['web_monitor']


# ── Cycle handlers (called by bridge) ────────────────────────────────
# Each handler returns a dict with cycle results.
# Failures are caught and returned as error dicts so the bridge stays alive.

def _safe_run(loop_name: str, getter, runner) -> Dict[str, Any]:
    """Safely initialize and run a loop cycle."""
    try:
        instance = getter()
        result = runner(instance)
        return {
            "loop": loop_name,
            "success": True,
            "result": result if isinstance(result, dict) else {"output": str(result)},
        }
    except Exception as e:
        logger.error(f"[{loop_name}] Cycle error: {e}\n{traceback.format_exc()}")
        return {
            "loop": loop_name,
            "success": False,
            "error": str(e),
        }


def run_ralph_cycle(**context) -> Dict:
    return _safe_run('ralph', _get_ralph_loop,
                     lambda loop: _run_async(loop.start()))


def run_research_cycle(**context) -> Dict:
    return _safe_run('research', _get_research_loop,
                     lambda loop: _run_async(loop.run()))


def run_planning_cycle(**context) -> Dict:
    return _safe_run('planning', _get_planning_loop,
                     lambda loop: _run_async(loop.execute_goal(context.get('goal'))) if context.get('goal')
                     else {"status": "idle", "note": "No goal provided"})


def run_e2e_cycle(**context) -> Dict:
    return _safe_run('e2e', _get_e2e_loop,
                     lambda loop: {"status": "ready", "engine": "E2EWorkflowEngine"})


def run_exploration_cycle(**context) -> Dict:
    return _safe_run('exploration', _get_exploration_loop,
                     lambda loop: _run_async(loop.run_single_cycle(context)))


def run_discovery_cycle(**context) -> Dict:
    return _safe_run('discovery', _get_discovery_loop,
                     lambda loop: _run_async(loop.start()))


def run_bug_finder_cycle(**context) -> Dict:
    return _safe_run('bug_finder', _get_bug_finder_loop,
                     lambda loop: _run_async(loop.run(context)))


def run_self_learning_cycle(**context) -> Dict:
    from datetime import timedelta
    return _safe_run('self_learning', _get_self_learning_loop,
                     lambda loop: _run_async(
                         loop.run_consolidation_cycle(timedelta(minutes=10))))


def run_meta_cognition_cycle(**context) -> Dict:
    return _safe_run('meta_cognition', _get_meta_cognition_loop,
                     lambda loop: _run_async(loop.execute_cycle(context=context)))


def run_self_upgrading_cycle(**context) -> Dict:
    return _safe_run('self_upgrading', _get_self_upgrading_loop,
                     lambda loop: _run_async(loop.run_upgrade_cycle()))


def run_self_updating_cycle(**context) -> Dict:
    return _safe_run('self_updating', _get_self_updating_loop,
                     lambda loop: _run_async(loop.start()))


def run_self_driven_cycle(**context) -> Dict:
    return _safe_run('self_driven', _get_self_driven_loop,
                     lambda loop: _run_async(loop.run()))


def run_cpel_cycle(**context) -> Dict:
    return _safe_run('cpel', _get_cpel_loop,
                     lambda loop: _run_async(loop.get_optimized_prompt(
                         user_input=context.get('user_input', 'system check'),
                         system_state=context,
                     )))


def run_context_engineering_cycle(**context) -> Dict:
    return _safe_run('context_engineering', _get_context_engineering_loop,
                     lambda loop: _run_async(loop.start()))


def run_web_monitor_cycle(**context) -> Dict:
    return _safe_run('web_monitor', _get_web_monitor_loop,
                     lambda loop: _run_async(loop.run()))


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
        'loop.web_monitor.run_cycle': run_web_monitor_cycle,
    }

    logger.info(f"Loop adapters: {len(handlers)} handlers prepared")
    return handlers
