"""
Update Detection Module
Monitors multiple sources for available updates.
"""

from .update_detector import (
    UpdateDetector,
    FileSystemMonitor,
    GitRepositoryMonitor,
    RemoteRegistryClient,
    UpdateEvent,
    UpdateSource,
    UpdateType,
    UpdatePriority,
)

__all__ = [
    "UpdateDetector",
    "FileSystemMonitor",
    "GitRepositoryMonitor",
    "RemoteRegistryClient",
    "UpdateEvent",
    "UpdateSource",
    "UpdateType",
    "UpdatePriority",
]
