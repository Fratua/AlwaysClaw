"""
Version Manager Module
Manages semantic versioning, Git integration, and release management.
"""

from .version_manager import (
    VersionManager,
    SemanticVersionManager,
    GitIntegration,
    ReleaseManager,
    Version,
    VersionRecord,
    Release,
    BumpType,
)

__all__ = [
    "VersionManager",
    "SemanticVersionManager",
    "GitIntegration",
    "ReleaseManager",
    "Version",
    "VersionRecord",
    "Release",
    "BumpType",
]
