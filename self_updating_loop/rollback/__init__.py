"""
Rollback Manager Module
Manages rollback operations to restore system to previous state.
"""

from .rollback_manager import (
    RollbackManager,
    RollbackStrategy,
    GitResetStrategy,
    BackupRestoreStrategy,
    IncrementalUndoStrategy,
    AutoRollbackTriggers,
    RollbackTarget,
    RollbackResult,
    Checkpoint,
    RollbackStrategyType,
    RollbackTargetType,
)

__all__ = [
    "RollbackManager",
    "RollbackStrategy",
    "GitResetStrategy",
    "BackupRestoreStrategy",
    "IncrementalUndoStrategy",
    "AutoRollbackTriggers",
    "RollbackTarget",
    "RollbackResult",
    "Checkpoint",
    "RollbackStrategyType",
    "RollbackTargetType",
]
