"""
Update Detection Module
Monitors multiple sources for available updates.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

try:
    import pygit2
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


class UpdateSource(Enum):
    """Sources from which updates can be detected"""
    LOCAL_FILESYSTEM = "local_fs"
    GIT_REPOSITORY = "git_repo"
    REMOTE_REGISTRY = "remote_registry"
    API_ENDPOINT = "api_endpoint"
    MANUAL_TRIGGER = "manual"
    SCHEDULED_CHECK = "scheduled"
    WEBHOOK = "webhook"


class UpdateType(Enum):
    """Types of updates"""
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"
    HOTFIX = "hotfix"
    CONFIG = "config"
    SECURITY = "security"


class UpdatePriority(Enum):
    """Update priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class UpdateEvent:
    """Represents a detected update event"""
    event_id: str
    timestamp: datetime
    source: UpdateSource
    update_type: UpdateType
    current_version: str
    target_version: str
    changed_files: List[str] = field(default_factory=list)
    change_summary: str = ""
    priority: UpdatePriority = UpdatePriority.LOW
    requires_restart: bool = False
    requires_approval: bool = True
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    remote_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        data = f"{self.timestamp.isoformat()}{self.source.value}{self.target_version}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class FileSystemMonitor:
    """
    Monitors file system for changes that may require updates.
    Uses watchdog library for efficient cross-platform monitoring.
    """
    
    MONITORED_PATHS = [
        "./agents/",
        "./loops/",
        "./config/",
        "./modules/",
        "./skills/",
    ]
    
    WATCH_PATTERNS = [
        "*.py",
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.ini",
    ]
    
    IGNORE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        "*.log",
        "*.tmp",
        "*.bak",
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[FileSystemEventHandler] = None
        self.detected_changes: List[Dict[str, Any]] = []
        self._file_hashes: Dict[str, str] = {}
        
    def start(self):
        """Start file system monitoring"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available - file system monitoring disabled")
            return
        
        self.event_handler = self._create_event_handler()
        self.observer = Observer()
        
        for path in self.MONITORED_PATHS:
            if Path(path).exists():
                self.observer.schedule(self.event_handler, path, recursive=True)
                logger.info(f"Monitoring path: {path}")
        
        self.observer.start()
        logger.info("File system monitoring started")
    
    def stop(self):
        """Stop file system monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File system monitoring stopped")
    
    def _create_event_handler(self) -> FileSystemEventHandler:
        """Create event handler for file system events"""
        class UpdateEventHandler(FileSystemEventHandler):
            def __init__(handler_self, monitor):
                handler_self.monitor = monitor
            
            def on_modified(handler_self, event):
                if event.is_directory:
                    return
                
                # Check if file matches watch patterns
                if not handler_self.monitor._should_watch_file(event.src_path):
                    return
                
                # Calculate file hash to detect actual changes
                file_hash = handler_self.monitor._get_file_hash(event.src_path)
                
                if handler_self.monitor._file_hashes.get(event.src_path) != file_hash:
                    handler_self.monitor._file_hashes[event.src_path] = file_hash
                    handler_self.monitor.detected_changes.append({
                        "path": event.src_path,
                        "timestamp": datetime.now(),
                        "hash": file_hash,
                    })
                    logger.debug(f"File changed: {event.src_path}")
        
        return UpdateEventHandler(self)
    
    def _should_watch_file(self, path: str) -> bool:
        """Check if file should be watched based on patterns"""
        path_obj = Path(path)
        
        # Check ignore patterns
        for pattern in self.IGNORE_PATTERNS:
            if pattern in path:
                return False
        
        # Check watch patterns
        for pattern in self.WATCH_PATTERNS:
            if path_obj.match(pattern):
                return True
        
        return False
    
    def _get_file_hash(self, path: str) -> str:
        """Calculate file hash"""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except OSError as e:
            logger.warning(f"Failed to hash file {path}: {e}")
            return ""
    
    def get_changes(self) -> List[Dict[str, Any]]:
        """Get and clear detected changes"""
        changes = self.detected_changes.copy()
        self.detected_changes.clear()
        return changes


class GitRepositoryMonitor:
    """
    Monitors Git repository for new commits, tags, and branches.
    Uses pygit2 for programmatic Git operations.
    """
    
    CHECK_INTERVAL = 300  # 5 minutes
    
    TRACKED_REFS = [
        "refs/heads/main",
        "refs/heads/develop",
        "refs/heads/release/*",
        "refs/tags/v*",
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_path = config.get("git_repo_path", ".")
        self.remote_url = config.get("git_remote_url", "")
        self.last_checked_commit: Optional[str] = None
        self._repo = None
        
    def _get_repo(self):
        """Get or initialize Git repository"""
        if not GIT_AVAILABLE:
            return None
        
        if self._repo is None:
            try:
                self._repo = pygit2.Repository(self.repo_path)
            except (OSError, KeyError) as e:
                logger.error(f"Failed to open Git repository: {e}", exc_info=True)
        
        return self._repo
    
    async def check_for_updates(self) -> List[UpdateEvent]:
        """Check Git repository for new updates"""
        if not GIT_AVAILABLE:
            logger.warning("pygit2 not available - Git monitoring disabled")
            return []
        
        repo = self._get_repo()
        if not repo:
            return []
        
        updates = []
        
        try:
            # Get current HEAD
            head = repo.head
            current_commit = head.target.hex
            
            # Check if there are new commits
            if self.last_checked_commit and self.last_checked_commit != current_commit:
                # Get commits since last check
                commits = self._get_commits_since(self.last_checked_commit)
                
                for commit in commits:
                    update = self._create_update_from_commit(commit)
                    if update:
                        updates.append(update)
            
            # Check for new tags
            new_tags = self._check_new_tags()
            updates.extend(new_tags)
            
            self.last_checked_commit = current_commit
            
        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Git check failed: {e}", exc_info=True)
        
        return updates
    
    def _get_commits_since(self, since_commit: str) -> List[pygit2.Commit]:
        """Get commits since specified commit"""
        repo = self._get_repo()
        commits = []
        
        try:
            walker = repo.walk(repo.head.target, pygit2.GIT_SORT_TIME)
            for commit in walker:
                if commit.hex == since_commit:
                    break
                commits.append(commit)
        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Failed to get commits: {e}", exc_info=True)
        
        return commits
    
    def _create_update_from_commit(self, commit: pygit2.Commit) -> Optional[UpdateEvent]:
        """Create UpdateEvent from Git commit"""
        # Analyze commit message for update type
        message = commit.message.lower()
        
        update_type = UpdateType.PATCH
        priority = UpdatePriority.LOW
        
        if "breaking" in message or "!:" in message:
            update_type = UpdateType.MAJOR
            priority = UpdatePriority.HIGH
        elif "feat:" in message or "feature" in message:
            update_type = UpdateType.MINOR
            priority = UpdatePriority.MEDIUM
        elif "fix:" in message or "hotfix" in message:
            update_type = UpdateType.HOTFIX
            priority = UpdatePriority.HIGH
        elif "security" in message:
            update_type = UpdateType.SECURITY
            priority = UpdatePriority.CRITICAL
        
        return UpdateEvent(
            event_id="",
            timestamp=datetime.fromtimestamp(commit.commit_time),
            source=UpdateSource.GIT_REPOSITORY,
            update_type=update_type,
            current_version=self._get_current_version(),
            target_version=self._get_target_version(commit),
            change_summary=commit.message.strip(),
            priority=priority,
            git_commit_hash=commit.hex,
            git_branch=commit.branch if hasattr(commit, 'branch') else None,
        )
    
    def _check_new_tags(self) -> List[UpdateEvent]:
        """Check for new version tags"""
        updates = []
        repo = self._get_repo()
        
        if not repo:
            return updates
        
        try:
            for ref_name in repo.listall_references():
                if ref_name.startswith("refs/tags/v"):
                    tag = repo.lookup_reference(ref_name)
                    # Check if this is a new tag
                    # In real implementation, track seen tags
                    pass
        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Failed to check tags: {e}", exc_info=True)
        
        return updates
    
    def _get_current_version(self) -> str:
        """Get current system version"""
        try:
            version_file = Path("version.json")
            if version_file.exists():
                with open(version_file) as f:
                    data = json.load(f)
                    return data.get("version", "0.0.0")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to read version file: {e}")
        return "0.0.0"
    
    def _get_target_version(self, commit: pygit2.Commit) -> str:
        """Determine target version from commit"""
        # In real implementation, parse version from commit or tags
        current = self._get_current_version()
        parts = current.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            return ".".join(parts)
        return "0.0.1"


class RemoteRegistryClient:
    """
    Polls remote update registry for available updates.
    Supports semantic versioning for update selection.
    """
    
    ENDPOINTS = {
        "check_updates": "/api/v1/updates/check",
        "download_update": "/api/v1/updates/download",
        "verify_checksum": "/api/v1/updates/verify",
        "report_status": "/api/v1/updates/status",
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config.get("remote_endpoint", "")
        self.api_key = config.get("remote_api_key", "")
        self.current_version = config.get("current_version", "0.0.0")
        
    async def check_for_updates(self) -> List[UpdateEvent]:
        """Check remote registry for available updates"""
        if not self.endpoint:
            return []
        
        updates = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                
                url = f"{self.endpoint}{self.ENDPOINTS['check_updates']}"
                params = {"current_version": self.current_version}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for update_data in data.get("updates", []):
                            update = self._parse_remote_update(update_data)
                            if update:
                                updates.append(update)
                    else:
                        logger.warning(f"Remote check failed: {response.status}")
                        
        except (aiohttp.ClientError, OSError, TimeoutError) as e:
            logger.error(f"Remote registry check failed: {e}", exc_info=True)
        
        return updates
    
    def _parse_remote_update(self, data: Dict[str, Any]) -> Optional[UpdateEvent]:
        """Parse remote update data into UpdateEvent"""
        try:
            update_type = UpdateType(data.get("type", "patch"))
            priority = UpdatePriority(data.get("priority", 1))
            
            return UpdateEvent(
                event_id=data.get("id", ""),
                timestamp=datetime.now(),
                source=UpdateSource.REMOTE_REGISTRY,
                update_type=update_type,
                current_version=self.current_version,
                target_version=data.get("version", ""),
                change_summary=data.get("description", ""),
                priority=priority,
                requires_restart=data.get("requires_restart", False),
                requires_approval=data.get("requires_approval", True),
                remote_url=data.get("download_url"),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse remote update: {e}", exc_info=True)
            return None


class UpdateDetector:
    """
    Main update detector that aggregates results from all sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize monitors
        self.fs_monitor = FileSystemMonitor(config)
        self.git_monitor = GitRepositoryMonitor(config)
        self.remote_client = RemoteRegistryClient(config)
        
        # Track detected updates to avoid duplicates
        self._detected_update_ids: Set[str] = set()
        
    def start(self):
        """Start all monitoring"""
        self.fs_monitor.start()
        logger.info("Update detector started")
    
    def stop(self):
        """Stop all monitoring"""
        self.fs_monitor.stop()
        logger.info("Update detector stopped")
    
    async def check_all_sources(self) -> List[UpdateEvent]:
        """Check all update sources and aggregate results"""
        all_updates = []
        
        # Check file system changes
        fs_changes = self.fs_monitor.get_changes()
        if fs_changes:
            fs_updates = self._process_fs_changes(fs_changes)
            all_updates.extend(fs_updates)
        
        # Check Git repository
        git_updates = await self.git_monitor.check_for_updates()
        all_updates.extend(git_updates)
        
        # Check remote registry
        remote_updates = await self.remote_client.check_for_updates()
        all_updates.extend(remote_updates)
        
        # Filter duplicates and already processed
        unique_updates = []
        for update in all_updates:
            if update.event_id not in self._detected_update_ids:
                self._detected_update_ids.add(update.event_id)
                unique_updates.append(update)
        
        # Limit tracked IDs to prevent memory growth
        if len(self._detected_update_ids) > 1000:
            self._detected_update_ids = set(list(self._detected_update_ids)[-500:])
        
        return unique_updates
    
    def _process_fs_changes(self, changes: List[Dict[str, Any]]) -> List[UpdateEvent]:
        """Process file system changes into update events"""
        if not changes:
            return []
        
        # Group changes by type
        config_changes = [c for c in changes if "config" in c["path"]]
        code_changes = [c for c in changes if c["path"].endswith(".py")]
        
        updates = []
        
        # Create update events for significant changes
        if config_changes:
            updates.append(UpdateEvent(
                event_id="",
                timestamp=datetime.now(),
                source=UpdateSource.LOCAL_FILESYSTEM,
                update_type=UpdateType.CONFIG,
                current_version="0.0.0",
                target_version="0.0.1",
                changed_files=[c["path"] for c in config_changes],
                change_summary=f"Configuration changes detected: {len(config_changes)} files",
                priority=UpdatePriority.MEDIUM,
            ))
        
        if code_changes:
            updates.append(UpdateEvent(
                event_id="",
                timestamp=datetime.now(),
                source=UpdateSource.LOCAL_FILESYSTEM,
                update_type=UpdateType.PATCH,
                current_version="0.0.0",
                target_version="0.0.1",
                changed_files=[c["path"] for c in code_changes],
                change_summary=f"Code changes detected: {len(code_changes)} files",
                priority=UpdatePriority.LOW,
            ))
        
        return updates


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "git_repo_path": ".",
            "remote_endpoint": "",
        }
        
        detector = UpdateDetector(config)
        detector.start()
        
        try:
            await asyncio.sleep(2)
            updates = await detector.check_all_sources()
            print(f"Found {len(updates)} updates")
            for update in updates:
                print(f"  - {update.update_type.value}: {update.change_summary}")
        finally:
            detector.stop()
    
    asyncio.run(main())
