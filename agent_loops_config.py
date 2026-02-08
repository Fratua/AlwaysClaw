"""
OpenClaw Windows 10 AI Agent - Agent Loop Configurations
=========================================================

This file contains TWO distinct sets of loop configurations:

1. OPERATIONAL SERVICE LOOPS (Node.js side)
   - Run inside the Node.js daemon (service.js / agent-loop-worker)
   - Handle heartbeat, Gmail sync, browser monitoring, TTS/STT, etc.
   - Defined in AGENT_LOOP_CONFIGS below

2. COGNITIVE LOOPS (Python side)
   - Run inside the Python bridge via loop_adapters.py
   - 15 AI reasoning loops: ralph, research, planning, e2e, exploration,
     discovery, bug_finder, self_learning, meta_cognition, self_upgrading,
     self_updating, self_driven, cpel, context_engineering, web_monitor
   - Defined in COGNITIVE_LOOP_CONFIGS below
   - Triggered via JSON-RPC: loop.<name>.run_cycle

Each loop has defined priority, interval, and CPU/memory limits.
"""

from resource_manager import TaskPriority

# ============================================================================
# AGENT LOOP CONFIGURATIONS
# ============================================================================

AGENT_LOOP_CONFIGS = {
    # ========================================================================
    # CORE SYSTEM LOOPS (Critical Priority)
    # ========================================================================
    
    'heartbeat': {
        'priority': TaskPriority.CRITICAL,
        'interval': 5.0,  # 5 seconds
        'cpu_limit': 2.0,
        'memory_limit': 16 * 1024 * 1024,  # 16MB
        'description': 'System health monitoring and keep-alive',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3
    },
    
    'soul_maintenance': {
        'priority': TaskPriority.CRITICAL,
        'interval': 60.0,  # 1 minute
        'cpu_limit': 5.0,
        'memory_limit': 32 * 1024 * 1024,  # 32MB
        'description': 'Agent soul/identity state maintenance',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 5
    },
    
    'identity_sync': {
        'priority': TaskPriority.CRITICAL,
        'interval': 300.0,  # 5 minutes
        'cpu_limit': 3.0,
        'memory_limit': 16 * 1024 * 1024,
        'description': 'Synchronize agent identity across sessions',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3
    },
    
    # ========================================================================
    # USER-FACING LOOPS (High Priority)
    # ========================================================================
    
    'user_input_handler': {
        'priority': TaskPriority.HIGH,
        'interval': 0.5,  # 500ms - responsive
        'cpu_limit': 10.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Process user input and commands',
        'enabled': True,
        'retry_on_failure': False,
        'max_retries': 0
    },
    
    'conversation_manager': {
        'priority': TaskPriority.HIGH,
        'interval': 1.0,  # 1 second
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Manage active conversations and context',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 2
    },
    
    'notification_dispatcher': {
        'priority': TaskPriority.HIGH,
        'interval': 2.0,  # 2 seconds
        'cpu_limit': 5.0,
        'memory_limit': 32 * 1024 * 1024,
        'description': 'Send notifications to user (desktop, voice, SMS)',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3
    },
    
    # ========================================================================
    # SERVICE LOOPS (Normal Priority)
    # ========================================================================
    
    'gmail_sync': {
        'priority': TaskPriority.NORMAL,
        'interval': 30.0,  # 30 seconds
        'cpu_limit': 10.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Synchronize Gmail inbox and send emails',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3,
        'api_calls': True,
        'api_budget_percent': 15  # 15% of daily API budget
    },
    
    'browser_monitor': {
        'priority': TaskPriority.NORMAL,
        'interval': 5.0,  # 5 seconds
        'cpu_limit': 8.0,
        'memory_limit': 256 * 1024 * 1024,  # Browser is memory-heavy
        'description': 'Monitor and control browser state',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 2
    },
    
    'tts_processor': {
        'priority': TaskPriority.NORMAL,
        'interval': 0.1,  # 100ms - low latency
        'cpu_limit': 5.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Process text-to-speech requests',
        'enabled': True,
        'retry_on_failure': False,
        'max_retries': 0
    },
    
    'stt_processor': {
        'priority': TaskPriority.NORMAL,
        'interval': 0.1,  # 100ms - low latency
        'cpu_limit': 5.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Process speech-to-text requests',
        'enabled': True,
        'retry_on_failure': False,
        'max_retries': 0
    },
    
    'twilio_handler': {
        'priority': TaskPriority.NORMAL,
        'interval': 10.0,  # 10 seconds
        'cpu_limit': 5.0,
        'memory_limit': 32 * 1024 * 1024,
        'description': 'Handle Twilio voice/SMS interactions',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3,
        'api_calls': True,
        'api_budget_percent': 10
    },
    
    # ========================================================================
    # BACKGROUND LOOPS (Low Priority)
    # ========================================================================
    
    'memory_cleanup': {
        'priority': TaskPriority.LOW,
        'interval': 300.0,  # 5 minutes
        'cpu_limit': 15.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Clean up unused memory and caches',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 1
    },
    
    'log_rotation': {
        'priority': TaskPriority.LOW,
        'interval': 3600.0,  # 1 hour
        'cpu_limit': 5.0,
        'memory_limit': 32 * 1024 * 1024,
        'description': 'Rotate and archive log files',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 3
    },
    
    'cache_maintenance': {
        'priority': TaskPriority.LOW,
        'interval': 600.0,  # 10 minutes
        'cpu_limit': 10.0,
        'memory_limit': 64 * 1024 * 1024,
        'description': 'Maintain cache freshness and expiry',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 2
    },
    
    # ========================================================================
    # MAINTENANCE LOOPS (Background Priority)
    # ========================================================================
    
    'metrics_collection': {
        'priority': TaskPriority.BACKGROUND,
        'interval': 60.0,  # 1 minute
        'cpu_limit': 3.0,
        'memory_limit': 32 * 1024 * 1024,
        'description': 'Collect system metrics and performance data',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 1
    },
    
    'self_optimization': {
        'priority': TaskPriority.BACKGROUND,
        'interval': 1800.0,  # 30 minutes
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Self-optimization analysis and tuning',
        'enabled': True,
        'retry_on_failure': True,
        'max_retries': 1,
        'api_calls': True,
        'api_budget_percent': 5
    }
}

# ============================================================================
# RESOURCE ALLOCATION SUMMARY
# ============================================================================

RESOURCE_ALLOCATION = {
    'memory': {
        'total_reserved': sum(
            loop['memory_limit'] 
            for loop in AGENT_LOOP_CONFIGS.values()
        ),
        'by_priority': {
            'CRITICAL': sum(
                loop['memory_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.CRITICAL
            ),
            'HIGH': sum(
                loop['memory_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.HIGH
            ),
            'NORMAL': sum(
                loop['memory_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.NORMAL
            ),
            'LOW': sum(
                loop['memory_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.LOW
            ),
            'BACKGROUND': sum(
                loop['memory_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.BACKGROUND
            )
        }
    },
    'cpu': {
        'total_reserved': sum(
            loop['cpu_limit'] 
            for loop in AGENT_LOOP_CONFIGS.values()
        ),
        'by_priority': {
            'CRITICAL': sum(
                loop['cpu_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.CRITICAL
            ),
            'HIGH': sum(
                loop['cpu_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.HIGH
            ),
            'NORMAL': sum(
                loop['cpu_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.NORMAL
            ),
            'LOW': sum(
                loop['cpu_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.LOW
            ),
            'BACKGROUND': sum(
                loop['cpu_limit'] 
                for name, loop in AGENT_LOOP_CONFIGS.items()
                if loop['priority'] == TaskPriority.BACKGROUND
            )
        }
    },
    'api_budget': {
        'total_percent': sum(
            loop.get('api_budget_percent', 0)
            for loop in AGENT_LOOP_CONFIGS.values()
        ),
        'by_loop': {
            name: loop.get('api_budget_percent', 0)
            for name, loop in AGENT_LOOP_CONFIGS.items()
            if loop.get('api_budget_percent', 0) > 0
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_loop_config(name: str) -> dict:
    """Get configuration for a specific agent loop"""
    return AGENT_LOOP_CONFIGS.get(name, {})

def get_loops_by_priority(priority: TaskPriority) -> dict:
    """Get all loops with a specific priority"""
    return {
        name: config 
        for name, config in AGENT_LOOP_CONFIGS.items()
        if config['priority'] == priority
    }

def get_api_enabled_loops() -> dict:
    """Get all loops that make API calls"""
    return {
        name: config 
        for name, config in AGENT_LOOP_CONFIGS.items()
        if config.get('api_calls', False)
    }

def print_resource_allocation():
    """Print resource allocation summary"""
    print("=" * 60)
    print("RESOURCE ALLOCATION SUMMARY")
    print("=" * 60)
    
    print("\n📊 Memory Allocation:")
    print(f"  Total Reserved: {RESOURCE_ALLOCATION['memory']['total_reserved'] / 1024 / 1024:.1f} MB")
    for priority, amount in RESOURCE_ALLOCATION['memory']['by_priority'].items():
        print(f"  {priority}: {amount / 1024 / 1024:.1f} MB")
    
    print("\n⚡ CPU Allocation:")
    print(f"  Total Reserved: {RESOURCE_ALLOCATION['cpu']['total_reserved']:.1f}%")
    for priority, amount in RESOURCE_ALLOCATION['cpu']['by_priority'].items():
        print(f"  {priority}: {amount:.1f}%")
    
    print("\n💰 API Budget Allocation:")
    print(f"  Total: {RESOURCE_ALLOCATION['api_budget']['total_percent']}%")
    for loop, percent in RESOURCE_ALLOCATION['api_budget']['by_loop'].items():
        print(f"  {loop}: {percent}%")
    
    print("=" * 60)

# ============================================================================
# COGNITIVE LOOP CONFIGURATIONS (Python-side, via loop_adapters.py)
# ============================================================================
# These are the 15 AI reasoning loops that run inside the Python bridge.
# They are triggered via JSON-RPC calls from the Node.js daemon:
#   method: "loop.<name>.run_cycle"
# The adapter module (loop_adapters.py) maps each name to its loop class.

COGNITIVE_LOOP_CONFIGS = {
    'ralph': {
        'priority': TaskPriority.HIGH,
        'interval': 30.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Ralph conversational agent loop',
        'module': 'ralph_loop_implementation',
        'class': 'RalphLoop',
        'enabled': True,
    },
    'research': {
        'priority': TaskPriority.NORMAL,
        'interval': 60.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Research and information gathering loop',
        'module': 'research_loop.research_loop',
        'class': 'ResearchLoop',
        'enabled': True,
    },
    'planning': {
        'priority': TaskPriority.NORMAL,
        'interval': 60.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Goal decomposition and planning loop',
        'module': 'planning_loop_implementation',
        'class': 'AdvancedPlanningLoop',
        'enabled': True,
    },
    'e2e': {
        'priority': TaskPriority.NORMAL,
        'interval': 60.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'End-to-end workflow orchestration loop',
        'module': 'e2e_loop_core',
        'class': 'E2EWorkflowEngine',
        'enabled': True,
    },
    'exploration': {
        'priority': TaskPriority.LOW,
        'interval': 120.0,
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Curiosity-driven exploration loop',
        'module': 'exploration_loop_implementation',
        'class': 'ExplorationLoop',
        'enabled': True,
    },
    'discovery': {
        'priority': TaskPriority.LOW,
        'interval': 120.0,
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Knowledge and pattern discovery loop',
        'module': 'discovery_loop_architecture',
        'class': 'DiscoveryLoop',
        'enabled': True,
    },
    'bug_finder': {
        'priority': TaskPriority.NORMAL,
        'interval': 300.0,
        'cpu_limit': 15.0,
        'memory_limit': 256 * 1024 * 1024,
        'description': 'ML-based predictive bug detection loop',
        'module': 'bug_finder_loop',
        'class': 'AdvancedBugFinderLoop',
        'enabled': True,
    },
    'self_learning': {
        'priority': TaskPriority.LOW,
        'interval': 600.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Continuous self-learning and knowledge consolidation',
        'module': 'self_learning_loop_implementation',
        'class': 'SelfLearningLoop',
        'enabled': True,
    },
    'meta_cognition': {
        'priority': TaskPriority.LOW,
        'interval': 300.0,
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Self-reflective meta-cognition loop',
        'module': 'meta_cognition_loop',
        'class': 'MetaCognitionLoop',
        'enabled': True,
    },
    'self_upgrading': {
        'priority': TaskPriority.BACKGROUND,
        'interval': 1800.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Self-upgrade orchestration loop',
        'module': 'self_upgrading_loop',
        'class': 'UpgradeOrchestrator',
        'enabled': True,
    },
    'self_updating': {
        'priority': TaskPriority.BACKGROUND,
        'interval': 1800.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Self-updating code improvement loop',
        'module': 'self_updating_loop.loop',
        'class': 'SelfUpdatingLoop',
        'enabled': True,
    },
    'self_driven': {
        'priority': TaskPriority.NORMAL,
        'interval': 60.0,
        'cpu_limit': 10.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Autonomous self-driven task loop',
        'module': 'self_driven_loop.self_driven_loop',
        'class': 'SelfDrivenLoop',
        'enabled': True,
    },
    'cpel': {
        'priority': TaskPriority.NORMAL,
        'interval': 30.0,
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Context-prompt engineering loop',
        'module': 'cpel_implementation',
        'class': 'ContextPromptEngineeringLoop',
        'enabled': True,
    },
    'context_engineering': {
        'priority': TaskPriority.NORMAL,
        'interval': 60.0,
        'cpu_limit': 8.0,
        'memory_limit': 128 * 1024 * 1024,
        'description': 'Context engineering and optimization loop',
        'module': 'context_engineering_loop',
        'class': 'ContextEngineeringLoop',
        'enabled': True,
    },
    'web_monitor': {
        'priority': TaskPriority.NORMAL,
        'interval': 300.0,
        'cpu_limit': 10.0,
        'memory_limit': 256 * 1024 * 1024,
        'description': 'Web monitoring and change detection loop',
        'module': 'web_monitor_agent_loop',
        'class': 'WebMonitoringAgentLoop',
        'enabled': True,
    },
}


def get_cognitive_loop_config(name: str) -> dict:
    """Get configuration for a specific cognitive loop."""
    return COGNITIVE_LOOP_CONFIGS.get(name, {})


def get_all_cognitive_loops() -> dict:
    """Get all cognitive loop configurations."""
    return COGNITIVE_LOOP_CONFIGS


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_resource_allocation()

    print("\n--- Operational Service Loops (Node.js side) ---")
    for priority in TaskPriority:
        loops = get_loops_by_priority(priority)
        if loops:
            print(f"\n{priority.name} ({len(loops)} loops):")
            for name, config in loops.items():
                print(f"  - {name}: {config['description']}")

    print("\n--- Cognitive Loops (Python side, via loop_adapters.py) ---")
    for name, config in COGNITIVE_LOOP_CONFIGS.items():
        print(f"  - loop.{name}.run_cycle: {config['description']}")
