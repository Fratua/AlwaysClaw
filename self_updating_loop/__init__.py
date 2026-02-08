"""
Self-Updating Loop Package
OpenClaw-Inspired AI Agent System - Windows 10 Edition

This package provides autonomous code update capabilities for the AI agent system,
including update detection, change analysis, safe modification, version control,
rollback mechanisms, and comprehensive audit logging.
"""

__version__ = "1.0.0"
__author__ = "OpenClaw AI Systems"

from .loop import (
    SelfUpdatingLoop,
    UpdateLoopState,
    UpdateSource,
    UpdateType,
    UpdatePriority,
    RiskLevel,
    UpdateEvent,
    ImpactReport,
    UpdateResult,
    get_self_updating_loop,
)

__all__ = [
    "SelfUpdatingLoop",
    "UpdateLoopState",
    "UpdateSource",
    "UpdateType",
    "UpdatePriority",
    "RiskLevel",
    "UpdateEvent",
    "ImpactReport",
    "UpdateResult",
    "get_self_updating_loop",
]
