"""
Version Manager Module
Manages semantic versioning, Git integration, and release management.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import pygit2
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BumpType(Enum):
    """Types of version bumps"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"
    BUILD = "build"


@dataclass
class Version:
    """Semantic version representation"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    buildmetadata: Optional[str] = None
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string into Version object"""
        # SemVer regex pattern
        pattern = r'^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)' \
                  r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)' \
                  r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?' \
                  r'(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
        
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")
        
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            buildmetadata=match.group("buildmetadata"),
        )
    
    def __str__(self) -> str:
        """Convert version to string"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.buildmetadata:
            version += f"+{self.buildmetadata}"
        return version
    
    def bump(self, bump_type: BumpType, prerelease_id: str = None) -> "Version":
        """Create new version with specified bump"""
        if bump_type == BumpType.MAJOR:
            return Version(
                major=self.major + 1,
                minor=0,
                patch=0,
                prerelease=prerelease_id,
            )
        elif bump_type == BumpType.MINOR:
            return Version(
                major=self.major,
                minor=self.minor + 1,
                patch=0,
                prerelease=prerelease_id,
            )
        elif bump_type == BumpType.PATCH:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch + 1,
                prerelease=prerelease_id,
            )
        elif bump_type == BumpType.PRERELEASE:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                prerelease=prerelease_id or "alpha.1",
            )
        else:
            return self
    
    def compare(self, other: "Version") -> int:
        """
        Compare versions.
        Returns: -1 if self < other, 0 if equal, 1 if self > other
        """
        # Compare major.minor.patch
        for attr in ["major", "minor", "patch"]:
            diff = getattr(self, attr) - getattr(other, attr)
            if diff != 0:
                return 1 if diff > 0 else -1
        
        # Handle prerelease comparison
        if self.prerelease is None and other.prerelease is not None:
            return 1  # No prerelease > has prerelease
        if self.prerelease is not None and other.prerelease is None:
            return -1
        
        if self.prerelease and other.prerelease:
            # Compare prerelease identifiers
            self_parts = self.prerelease.split(".")
            other_parts = other.prerelease.split(".")
            
            for self_part, other_part in zip(self_parts, other_parts):
                # Try numeric comparison first
                try:
                    self_num = int(self_part)
                    other_num = int(other_part)
                    if self_num != other_num:
                        return 1 if self_num > other_num else -1
                except ValueError:
                    # String comparison
                    if self_part != other_part:
                        return 1 if self_part > other_part else -1
            
            # Longer prerelease > shorter prerelease
            if len(self_parts) != len(other_parts):
                return 1 if len(self_parts) > len(other_parts) else -1
        
        return 0  # Equal
    
    def __lt__(self, other: "Version") -> bool:
        return self.compare(other) < 0
    
    def __le__(self, other: "Version") -> bool:
        return self.compare(other) <= 0
    
    def __gt__(self, other: "Version") -> bool:
        return self.compare(other) > 0
    
    def __ge__(self, other: "Version") -> bool:
        return self.compare(other) >= 0
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return self.compare(other) == 0


@dataclass
class VersionRecord:
    """Records version information for audit trail"""
    version: str
    timestamp: datetime
    commit_hash: Optional[str]
    author: str
    changes: List[str]
    update_type: str
    previous_version: str
    rollback_available: bool
    rollback_commit: Optional[str] = None


@dataclass
class Release:
    """Represents a software release"""
    version: str
    timestamp: datetime
    commit_hash: str
    changelog: str
    release_notes: str
    assets: List[Dict[str, Any]] = field(default_factory=list)
    is_prerelease: bool = False


class GitIntegration:
    """
    Manages Git integration for version control.
    Uses pygit2 for programmatic Git operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_path = config.get("git_repo_path", ".")
        self.remote_url = config.get("git_remote_url", "")
        self._repo = None
        
    def _get_repo(self):
        """Get or initialize Git repository"""
        if not GIT_AVAILABLE:
            return None
        
        if self._repo is None:
            try:
                self._repo = pygit2.Repository(self.repo_path)
            except Exception as e:
                logger.error(f"Failed to open Git repository: {e}")
        
        return self._repo
    
    def get_current_commit(self) -> Optional[str]:
        """Get current commit hash"""
        repo = self._get_repo()
        if not repo:
            return None
        
        try:
            return repo.head.target.hex
        except Exception:
            return None
    
    def get_commit_message(self, commit_hash: str) -> Optional[str]:
        """Get commit message for specified commit"""
        repo = self._get_repo()
        if not repo:
            return None
        
        try:
            commit = repo.revparse_single(commit_hash)
            return commit.message
        except Exception:
            return None
    
    def get_commits_since(self, since_commit: str) -> List[Dict[str, Any]]:
        """Get commits since specified commit"""
        repo = self._get_repo()
        if not repo:
            return []
        
        commits = []
        try:
            walker = repo.walk(repo.head.target, pygit2.GIT_SORT_TIME)
            for commit in walker:
                if commit.hex == since_commit:
                    break
                commits.append({
                    "hash": commit.hex,
                    "message": commit.message,
                    "author": str(commit.author),
                    "timestamp": commit.commit_time,
                })
        except Exception as e:
            logger.error(f"Failed to get commits: {e}")
        
        return commits
    
    def create_tag(self, version: str, message: str = "") -> bool:
        """Create Git tag for version"""
        repo = self._get_repo()
        if not repo:
            return False
        
        try:
            tag_name = f"v{version}"
            repo.create_tag(
                tag_name,
                repo.head.target,
                pygit2.GIT_OBJ_COMMIT,
                repo.default_signature,
                message or f"Release {version}",
            )
            logger.info(f"Git tag created: {tag_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create tag: {e}")
            return False
    
    def push_to_remote(self, ref: str = "refs/heads/main") -> bool:
        """Push changes to remote repository"""
        repo = self._get_repo()
        if not repo:
            return False
        
        try:
            remote = repo.remotes["origin"]
            remote.push([ref])
            logger.info(f"Pushed to remote: {ref}")
            return True
        except Exception as e:
            logger.error(f"Failed to push to remote: {e}")
            return False


class SemanticVersionManager:
    """
    Manages semantic versioning for the AI agent system.
    Implements SemVer 2.0.0 specification.
    """
    
    VERSION_FILE = "./version.json"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.version_file = Path(config.get("version_file", self.VERSION_FILE))
        self._current_version: Optional[Version] = None
        
    def get_current_version(self) -> str:
        """Get current system version string"""
        version = self._get_version_object()
        return str(version)
    
    def _get_version_object(self) -> Version:
        """Get current version as Version object"""
        if self._current_version is None:
            try:
                if self.version_file.exists():
                    with open(self.version_file) as f:
                        data = json.load(f)
                        version_str = data.get("version", "0.0.0")
                        self._current_version = Version.parse(version_str)
                else:
                    self._current_version = Version(0, 0, 0)
            except Exception as e:
                logger.error(f"Failed to parse version: {e}")
                self._current_version = Version(0, 0, 0)
        
        return self._current_version
    
    def bump_version(self, bump_type: BumpType, 
                     prerelease_id: str = None) -> str:
        """
        Bump version according to SemVer rules.
        
        Args:
            bump_type: Type of version bump
            prerelease_id: Optional prerelease identifier
            
        Returns:
            New version string
        """
        current = self._get_version_object()
        new_version = current.bump(bump_type, prerelease_id)
        
        self._save_version(new_version)
        
        logger.info(f"Version bumped: {current} -> {new_version}")
        return str(new_version)
    
    def _save_version(self, version: Version):
        """Save version to file"""
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": str(version),
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(self.version_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._current_version = version
    
    def compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two version strings.
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        version1 = Version.parse(v1)
        version2 = Version.parse(v2)
        return version1.compare(version2)
    
    def is_compatible(self, required: str, current: str) -> bool:
        """
        Check if current version satisfies required version.
        
        For now, implements simple major version compatibility.
        """
        try:
            req = Version.parse(required)
            cur = Version.parse(current)
            
            # Same major version is compatible
            return req.major == cur.major
        except ValueError:
            return False


class ReleaseManager:
    """
    Manages releases including tagging, changelog generation,
    and release notes.
    """
    
    CHANGELOG_FILE = "./CHANGELOG.md"
    
    CHANGELOG_CATEGORIES = {
        "feat": "Features",
        "fix": "Bug Fixes",
        "perf": "Performance Improvements",
        "docs": "Documentation",
        "refactor": "Refactoring",
        "test": "Tests",
        "chore": "Chores",
        "security": "Security",
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.git = GitIntegration(config)
        self.changelog_file = Path(config.get("changelog_file", self.CHANGELOG_FILE))
        
    def create_release(self, version: str, changes: List[str],
                       git_commit: str = None) -> Optional[Release]:
        """
        Create a new release.
        
        Args:
            version: Release version
            changes: List of changes in this release
            git_commit: Associated Git commit hash
            
        Returns:
            Release object
        """
        timestamp = datetime.now()
        
        # Get commit hash if not provided
        if not git_commit:
            git_commit = self.git.get_current_commit() or "unknown"
        
        # Generate changelog entry
        changelog = self._generate_changelog_entry(version, changes, timestamp)
        
        # Generate release notes
        release_notes = self._generate_release_notes(version, changes)
        
        # Create Git tag
        self.git.create_tag(version, release_notes)
        
        # Update changelog file
        self._update_changelog_file(changelog)
        
        release = Release(
            version=version,
            timestamp=timestamp,
            commit_hash=git_commit,
            changelog=changelog,
            release_notes=release_notes,
        )
        
        logger.info(f"Release created: {version}")
        return release
    
    def _generate_changelog_entry(self, version: str, 
                                   changes: List[str],
                                   timestamp: datetime) -> str:
        """Generate changelog entry for version"""
        lines = [
            f"## [{version}] - {timestamp.strftime('%Y-%m-%d')}",
            "",
        ]
        
        # Categorize changes
        categorized: Dict[str, List[str]] = {cat: [] for cat in self.CHANGELOG_CATEGORIES.values()}
        
        for change in changes:
            # Try to categorize based on conventional commit prefix
            categorized_flag = False
            for prefix, category in self.CHANGELOG_CATEGORIES.items():
                if change.lower().startswith(f"{prefix}:") or change.lower().startswith(f"{prefix}(" ):
                    categorized[category].append(change)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized["Chores"].append(change)
        
        # Add categorized changes
        for category, items in categorized.items():
            if items:
                lines.append(f"### {category}")
                lines.append("")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_release_notes(self, version: str, 
                                 changes: List[str]) -> str:
        """Generate release notes"""
        lines = [
            f"# Release {version}",
            "",
            "## Changes",
            "",
        ]
        
        for change in changes:
            lines.append(f"- {change}")
        
        lines.extend([
            "",
            "## Installation",
            "",
            f"```bash",
            f"pip install openclaw=={version}",
            f"```",
        ])
        
        return "\n".join(lines)
    
    def _update_changelog_file(self, entry: str):
        """Update changelog file with new entry"""
        existing_content = ""
        
        if self.changelog_file.exists():
            existing_content = self.changelog_file.read_text()
        
        # Prepend new entry
        header = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        new_content = header + entry + "\n" + existing_content[len(header):]
        
        self.changelog_file.parent.mkdir(parents=True, exist_ok=True)
        self.changelog_file.write_text(new_content)
    
    def get_changelog(self, since_version: str = None) -> str:
        """Get changelog content"""
        if self.changelog_file.exists():
            return self.changelog_file.read_text()
        return ""


class VersionManager:
    """
    Main version manager that coordinates semantic versioning,
    Git integration, and release management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.semver = SemanticVersionManager(config)
        self.git = GitIntegration(config)
        self.release_manager = ReleaseManager(config)
        self._version_history: List[VersionRecord] = []
        self._load_version_history()
        
    def _load_version_history(self):
        """Load version history from file"""
        history_file = Path("./state/version_history.json")
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self._version_history = [
                        VersionRecord(
                            version=r["version"],
                            timestamp=datetime.fromisoformat(r["timestamp"]),
                            commit_hash=r.get("commit_hash"),
                            author=r.get("author", "system"),
                            changes=r.get("changes", []),
                            update_type=r.get("update_type", "patch"),
                            previous_version=r.get("previous_version", "0.0.0"),
                            rollback_available=r.get("rollback_available", False),
                            rollback_commit=r.get("rollback_commit"),
                        )
                        for r in data
                    ]
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")
    
    def _save_version_history(self):
        """Save version history to file"""
        history_file = Path("./state/version_history.json")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                "version": r.version,
                "timestamp": r.timestamp.isoformat(),
                "commit_hash": r.commit_hash,
                "author": r.author,
                "changes": r.changes,
                "update_type": r.update_type,
                "previous_version": r.previous_version,
                "rollback_available": r.rollback_available,
                "rollback_commit": r.rollback_commit,
            }
            for r in self._version_history
        ]
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_current_version(self) -> str:
        """Get current system version"""
        return self.semver.get_current_version()
    
    def bump_version(self, bump_type: BumpType) -> str:
        """Bump system version"""
        return self.semver.bump_version(bump_type)
    
    def record_update(self, version: str, previous_version: str,
                      update_type: str, changes: List[str],
                      rollback_available: bool = False):
        """
        Record an update in version history.
        
        Args:
            version: New version
            previous_version: Previous version
            update_type: Type of update
            changes: List of changes
            rollback_available: Whether rollback is available
        """
        record = VersionRecord(
            version=version,
            timestamp=datetime.now(),
            commit_hash=self.git.get_current_commit(),
            author="system",  # Could be enhanced to track actual user
            changes=changes,
            update_type=update_type,
            previous_version=previous_version,
            rollback_available=rollback_available,
            rollback_commit=self.git.get_current_commit() if rollback_available else None,
        )
        
        self._version_history.append(record)
        
        # Limit history size
        if len(self._version_history) > 100:
            self._version_history = self._version_history[-100:]
        
        self._save_version_history()
        
        logger.info(f"Update recorded: {previous_version} -> {version}")
    
    def create_release(self, version: str, 
                       changes: List[str]) -> Optional[Release]:
        """Create a new release"""
        return self.release_manager.create_release(version, changes)
    
    def get_version_history(self, limit: int = 10) -> List[VersionRecord]:
        """Get version history"""
        return self._version_history[-limit:]
    
    def compare_versions(self, v1: str, v2: str) -> int:
        """Compare two versions"""
        return self.semver.compare_versions(v1, v2)


if __name__ == "__main__":
    # Example usage
    config = {
        "git_repo_path": ".",
    }
    
    manager = VersionManager(config)
    
    # Get current version
    current = manager.get_current_version()
    print(f"Current version: {current}")
    
    # Bump version
    new_version = manager.bump_version(BumpType.PATCH)
    print(f"New version: {new_version}")
    
    # Record update
    manager.record_update(
        version=new_version,
        previous_version=current,
        update_type="patch",
        changes=["fix: resolved issue with update detection"],
    )
    
    # Get history
    history = manager.get_version_history()
    print(f"Version history: {len(history)} entries")
