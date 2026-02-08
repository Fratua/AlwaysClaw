"""
Safety Engine Module
Ensures all modifications are performed safely with proper
validation, backup, and rollback preparation.
"""

from .safety_engine import (
    SafetyEngine,
    BackupManager,
    AtomicUpdateExecutor,
    BackupType,
    RollbackComplexity,
    BackupRecord,
    UpdateOperation,
    StagedUpdate,
    UpdateResult,
)

__all__ = [
    "SafetyEngine",
    "BackupManager",
    "AtomicUpdateExecutor",
    "BackupType",
    "RollbackComplexity",
    "BackupRecord",
    "UpdateOperation",
    "StagedUpdate",
    "UpdateResult",
]
