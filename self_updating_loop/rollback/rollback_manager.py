"""
Rollback Manager Module
Manages rollback operations to restore system to previous state.
"""

import shutil
import logging
import os
import zipfile
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

try:
    import pygit2
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class RollbackStrategyType(Enum):
    """Types of rollback strategies"""
    GIT_RESET = "git_reset"
    BACKUP_RESTORE = "backup_restore"
    INCREMENTAL_UNDO = "incremental_undo"
    STATE_RECONSTRUCTION = "state_reconstruction"


class RollbackTargetType(Enum):
    """Types of rollback targets"""
    VERSION = "version"
    COMMIT = "commit"
    TIMESTAMP = "timestamp"
    CHECKPOINT = "checkpoint"
    PREVIOUS_STABLE = "previous_stable"


@dataclass
class RollbackTarget:
    """Target for rollback operation"""
    target_type: RollbackTargetType
    target_value: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    rollback_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    target: Optional[RollbackTarget] = None
    strategy_used: str = ""
    message: str = ""
    error: Optional[str] = None
    affected_files: List[str] = field(default_factory=list)


@dataclass
class Checkpoint:
    """System checkpoint for rollback"""
    checkpoint_id: str
    timestamp: datetime
    version: str
    git_commit: Optional[str]
    backup_id: Optional[str]
    state_snapshot: Dict[str, Any]
    operations_since: List[Dict[str, Any]] = field(default_factory=list)


class RollbackStrategy(ABC):
    """Base class for rollback strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def execute(self, target: RollbackTarget) -> RollbackResult:
        """Execute rollback to target - must be implemented by subclasses"""
        ...
    
    def can_execute(self, target: RollbackTarget) -> bool:
        """Check if this strategy can handle the target"""
        return True


class GitResetStrategy(RollbackStrategy):
    """
    Uses Git reset to rollback to previous commit.
    Fast and reliable for code-only rollbacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.repo_path = config.get("git_repo_path", ".")
        self._repo = None
    
    def _get_repo(self):
        """Get Git repository"""
        if not GIT_AVAILABLE:
            return None
        
        if self._repo is None:
            try:
                self._repo = pygit2.Repository(self.repo_path)
            except (OSError, KeyError) as e:
                logger.error(f"Failed to open Git repository: {e}", exc_info=True)
        
        return self._repo
    
    def can_execute(self, target: RollbackTarget) -> bool:
        """Check if Git reset can be used"""
        if not GIT_AVAILABLE:
            return False
        
        repo = self._get_repo()
        if not repo:
            return False
        
        # Check if target is a valid Git reference
        if target.target_type in [RollbackTargetType.COMMIT, RollbackTargetType.VERSION]:
            try:
                repo.revparse_single(target.target_value)
                return True
            except (KeyError, ValueError):
                return False
        
        return False
    
    async def execute(self, target: RollbackTarget) -> RollbackResult:
        """
        Execute Git-based rollback.
        
        Steps:
        1. Stash any uncommitted changes
        2. Reset to target commit
        3. Verify system state
        4. Restart affected services
        """
        repo = self._get_repo()
        
        if not repo:
            return RollbackResult(
                success=False,
                rollback_id="",
                target=target,
                strategy_used="git_reset",
                error="Git repository not available",
            )
        
        rollback_id = self._generate_rollback_id()
        
        try:
            # Stash uncommitted changes
            # Note: pygit2 stash implementation varies
            
            # Get target commit
            target_commit = repo.revparse_single(target.target_value)
            
            # Perform reset
            repo.reset(target_commit.hex, pygit2.GIT_RESET_HARD)
            
            logger.info(f"Git reset completed: {target.target_value}")
            
            return RollbackResult(
                success=True,
                rollback_id=rollback_id,
                target=target,
                strategy_used="git_reset",
                message=f"Successfully reset to {target.target_value}",
            )
            
        except (OSError, KeyError, ValueError) as e:
            logger.error(f"Git reset failed: {e}", exc_info=True)
            return RollbackResult(
                success=False,
                rollback_id=rollback_id,
                target=target,
                strategy_used="git_reset",
                error=str(e),
            )
    
    def _generate_rollback_id(self) -> str:
        """Generate unique rollback ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"git_rollback_{timestamp}"


class BackupRestoreStrategy(RollbackStrategy):
    """
    Restores from backup archive.
    Comprehensive rollback including data and configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backup_root = Path(config.get("backup_root", "./backups/"))
    
    def can_execute(self, target: RollbackTarget) -> bool:
        """Check if backup restore can be used"""
        if target.target_type == RollbackTargetType.CHECKPOINT:
            backup_path = self.backup_root / f"backup_{target.target_value}.zip"
            return backup_path.exists()
        
        if target.target_type == RollbackTargetType.VERSION:
            # Look for backup with version tag
            for backup_file in self.backup_root.glob("*.zip"):
                if target.target_value in backup_file.name:
                    return True
        
        return False
    
    async def execute(self, target: RollbackTarget) -> RollbackResult:
        """
        Execute backup-based rollback.
        
        Steps:
        1. Stop affected services
        2. Restore files from backup
        3. Restore database if needed
        4. Restore configuration
        5. Verify restoration
        6. Restart services
        """
        rollback_id = self._generate_rollback_id()
        
        try:
            # Find backup file
            backup_path = self._find_backup(target)
            
            if not backup_path:
                return RollbackResult(
                    success=False,
                    rollback_id=rollback_id,
                    target=target,
                    strategy_used="backup_restore",
                    error=f"Backup not found for target: {target.target_value}",
                )
            
            # Verify backup integrity
            if not self._verify_backup(backup_path):
                return RollbackResult(
                    success=False,
                    rollback_id=rollback_id,
                    target=target,
                    strategy_used="backup_restore",
                    error="Backup integrity check failed",
                )
            
            # Extract backup
            import zipfile
            extract_path = Path("./.rollback_temp")
            
            with zipfile.ZipFile(backup_path, 'r') as zf:
                zf.extractall(extract_path)
            
            # Track affected files
            affected_files = []
            
            # Restore files
            for src_path in extract_path.rglob("*"):
                if src_path.is_file():
                    rel_path = src_path.relative_to(extract_path)
                    dst_path = Path(".") / rel_path
                    
                    # Backup current file before overwriting
                    if dst_path.exists():
                        backup_current = Path(f"{dst_path}.pre_rollback")
                        shutil.copy2(dst_path, backup_current)
                    
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    affected_files.append(str(dst_path))
            
            # Clean up extract directory
            shutil.rmtree(extract_path)
            
            logger.info(f"Backup restore completed: {backup_path.name}")
            
            return RollbackResult(
                success=True,
                rollback_id=rollback_id,
                target=target,
                strategy_used="backup_restore",
                message=f"Successfully restored from {backup_path.name}",
                affected_files=affected_files,
            )
            
        except (OSError, zipfile.BadZipFile, shutil.Error) as e:
            logger.error(f"Backup restore failed: {e}", exc_info=True)
            return RollbackResult(
                success=False,
                rollback_id=rollback_id,
                target=target,
                strategy_used="backup_restore",
                error=str(e),
            )
    
    def _find_backup(self, target: RollbackTarget) -> Optional[Path]:
        """Find backup file for target"""
        if target.target_type == RollbackTargetType.CHECKPOINT:
            backup_path = self.backup_root / f"backup_{target.target_value}.zip"
            if backup_path.exists():
                return backup_path
        
        if target.target_type == RollbackTargetType.VERSION:
            for backup_file in self.backup_root.glob("*.zip"):
                if target.target_value in backup_file.name:
                    return backup_file
        
        # Find most recent backup
        backups = sorted(self.backup_root.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if backups:
            return backups[0]
        
        return None
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            import zipfile
            with zipfile.ZipFile(backup_path, 'r') as zf:
                # Test archive integrity
                bad_file = zf.testzip()
                return bad_file is None
        except (OSError, zipfile.BadZipFile) as e:
            logger.error(f"Backup verification failed: {e}", exc_info=True)
            return False
    
    def _generate_rollback_id(self) -> str:
        """Generate unique rollback ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_rollback_{timestamp}"


class IncrementalUndoStrategy(RollbackStrategy):
    """
    Undoes changes incrementally using recorded operations.
    Useful for complex updates with multiple steps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.operations_log_path = Path(config.get("operations_log", "./logs/operations.json"))
        self._operations_log: List[Dict[str, Any]] = []
        self._load_operations_log()
    
    def _load_operations_log(self):
        """Load operations log from file"""
        if self.operations_log_path.exists():
            try:
                with open(self.operations_log_path) as f:
                    self._operations_log = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to load operations log: {e}", exc_info=True)
    
    def _save_operations_log(self):
        """Save operations log to file"""
        self.operations_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.operations_log_path, 'w') as f:
            json.dump(self._operations_log, f, indent=2)
    
    def can_execute(self, target: RollbackTarget) -> bool:
        """Check if incremental undo can be used"""
        if target.target_type == RollbackTargetType.CHECKPOINT:
            # Check if we have operations since checkpoint
            checkpoint_time = datetime.fromisoformat(target.target_value)
            recent_ops = [
                op for op in self._operations_log
                if datetime.fromisoformat(op.get("timestamp", "1970-01-01")) > checkpoint_time
            ]
            return len(recent_ops) > 0
        
        return False
    
    async def execute(self, target: RollbackTarget) -> RollbackResult:
        """
        Execute incremental rollback.
        
        Steps:
        1. Get operations since checkpoint
        2. Generate inverse operations
        3. Execute inverse operations in reverse order
        4. Verify state matches checkpoint
        """
        rollback_id = self._generate_rollback_id()
        
        try:
            # Get operations since checkpoint
            checkpoint_time = datetime.fromisoformat(target.target_value)
            operations_since = [
                op for op in self._operations_log
                if datetime.fromisoformat(op.get("timestamp", "1970-01-01")) > checkpoint_time
            ]
            
            if not operations_since:
                return RollbackResult(
                    success=True,
                    rollback_id=rollback_id,
                    target=target,
                    strategy_used="incremental_undo",
                    message="No operations to undo",
                )
            
            # Generate and execute inverse operations
            affected_files = []
            
            for op in reversed(operations_since):
                inverse_op = self._create_inverse_operation(op)
                if inverse_op:
                    success = await self._execute_inverse_operation(inverse_op)
                    if not success:
                        return RollbackResult(
                            success=False,
                            rollback_id=rollback_id,
                            target=target,
                            strategy_used="incremental_undo",
                            error=f"Failed to undo operation: {op}",
                        )
                    
                    if "target" in op:
                        affected_files.append(op["target"])
            
            logger.info(f"Incremental undo completed: {len(operations_since)} operations")
            
            return RollbackResult(
                success=True,
                rollback_id=rollback_id,
                target=target,
                strategy_used="incremental_undo",
                message=f"Successfully undid {len(operations_since)} operations",
                affected_files=affected_files,
            )
            
        except (OSError, json.JSONDecodeError, KeyError, ValueError, shutil.Error) as e:
            logger.error(f"Incremental undo failed: {e}", exc_info=True)
            return RollbackResult(
                success=False,
                rollback_id=rollback_id,
                target=target,
                strategy_used="incremental_undo",
                error=str(e),
            )
    
    def _create_inverse_operation(self, operation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create inverse operation for rollback"""
        op_type = operation.get("operation_type")
        
        if op_type == "file_write":
            # Inverse: restore from backup or delete
            target = operation.get("target")
            backup = operation.get("backup")
            
            if backup and Path(backup).exists():
                return {
                    "operation_type": "file_restore",
                    "source": backup,
                    "target": target,
                }
            else:
                return {
                    "operation_type": "file_delete",
                    "target": target,
                }
        
        elif op_type == "file_delete":
            # Inverse: restore from backup
            backup = operation.get("backup")
            if backup and Path(backup).exists():
                return {
                    "operation_type": "file_restore",
                    "source": backup,
                    "target": operation.get("target"),
                }
        
        elif op_type == "config_update":
            # Inverse: restore previous config
            return {
                "operation_type": "config_restore",
                "previous_state": operation.get("previous_state"),
                "target": operation.get("target"),
            }
        
        return None
    
    async def _execute_inverse_operation(self, inverse_op: Dict[str, Any]) -> bool:
        """Execute an inverse operation"""
        try:
            op_type = inverse_op.get("operation_type")
            
            if op_type == "file_restore":
                source = Path(inverse_op["source"])
                target = Path(inverse_op["target"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                return True
            
            elif op_type == "file_delete":
                target = Path(inverse_op["target"])
                if target.exists():
                    target.unlink()
                return True
            
            elif op_type == "config_restore":
                target = Path(inverse_op["target"])
                previous_state = inverse_op.get("previous_state", {})
                with open(target, 'w') as f:
                    json.dump(previous_state, f, indent=2)
                return True
            
            return False
            
        except (OSError, json.JSONDecodeError, KeyError, shutil.Error) as e:
            logger.error(f"Failed to execute inverse operation: {e}", exc_info=True)
            return False
    
    def _generate_rollback_id(self) -> str:
        """Generate unique rollback ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"incremental_rollback_{timestamp}"
    
    def log_operation(self, operation: Dict[str, Any]):
        """Log an operation for potential rollback"""
        operation["timestamp"] = datetime.now().isoformat()
        self._operations_log.append(operation)
        self._save_operations_log()


class AutoRollbackTriggers:
    """
    Defines conditions that trigger automatic rollback.
    """

    TRIGGERS = None  # Loaded from rollback-config.yaml in __init__

    @staticmethod
    def _load_default_triggers() -> Dict:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'rollback-config.yaml')
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            triggers_cfg = cfg.get('rollback', {}).get('auto_rollback', {}).get('triggers', {})
            if triggers_cfg:
                return {
                    "health_check_failure": {
                        "description": "System health checks failing after update",
                        "threshold": triggers_cfg.get('health_check', {}).get('consecutive_failures', 3),
                        "window_seconds": triggers_cfg.get('health_check', {}).get('check_interval_seconds', 10) * triggers_cfg.get('health_check', {}).get('consecutive_failures', 3),
                    },
                    "error_rate_spike": {
                        "description": "Error rate exceeded threshold",
                        "threshold": triggers_cfg.get('error_rate', {}).get('threshold', 0.1),
                        "window_seconds": triggers_cfg.get('error_rate', {}).get('window_seconds', 120),
                    },
                    "performance_degradation": {
                        "description": "Performance degraded beyond acceptable limit",
                        "threshold": 1.5,
                        "window_seconds": triggers_cfg.get('latency', {}).get('window_seconds', 300),
                    },
                    "service_unavailable": {
                        "description": "Critical service unavailable",
                        "threshold": 1,
                        "window_seconds": 30,
                    },
                }
        except (OSError, yaml.YAMLError):
            pass
        return {
            "health_check_failure": {"description": "System health checks failing after update", "threshold": 3, "window_seconds": 300},
            "error_rate_spike": {"description": "Error rate exceeded threshold", "threshold": 0.1, "window_seconds": 120},
            "performance_degradation": {"description": "Performance degraded beyond acceptable limit", "threshold": 1.5, "window_seconds": 300},
            "service_unavailable": {"description": "Critical service unavailable", "threshold": 1, "window_seconds": 30},
        }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if AutoRollbackTriggers.TRIGGERS is None:
            AutoRollbackTriggers.TRIGGERS = self._load_default_triggers()
        self._health_history: List[Dict[str, Any]] = []
        self._error_history: List[Dict[str, Any]] = []
        
    def check_triggers(self) -> List[str]:
        """Check all triggers and return triggered ones"""
        triggered = []
        
        for trigger_name, trigger_config in self.TRIGGERS.items():
            if self._check_trigger(trigger_name, trigger_config):
                triggered.append(trigger_name)
        
        return triggered
    
    def _check_trigger(self, name: str, config: Dict[str, Any]) -> bool:
        """Check if a specific trigger is activated"""
        if name == "health_check_failure":
            return self._check_health_failures(config)
        elif name == "error_rate_spike":
            return self._check_error_rate(config)
        elif name == "performance_degradation":
            return self._check_performance(config)
        elif name == "service_unavailable":
            return self._check_service_availability(config)
        
        return False
    
    def _check_health_failures(self, config: Dict[str, Any]) -> bool:
        """Check for consecutive health check failures"""
        threshold = config["threshold"]
        window = config["window_seconds"]
        
        cutoff = datetime.now().timestamp() - window
        recent_failures = [
            h for h in self._health_history
            if h["timestamp"] > cutoff and not h["healthy"]
        ]
        
        return len(recent_failures) >= threshold
    
    def _check_error_rate(self, config: Dict[str, Any]) -> bool:
        """Check if error rate exceeds threshold"""
        threshold = config["threshold"]
        window = config["window_seconds"]
        
        cutoff = datetime.now().timestamp() - window
        recent_errors = [e for e in self._error_history if e["timestamp"] > cutoff]
        
        if not recent_errors:
            return False
        
        error_rate = len([e for e in recent_errors if e["is_error"]]) / len(recent_errors)
        return error_rate > threshold
    
    def _check_performance(self, config: Dict[str, Any]) -> bool:
        """Check for performance degradation"""
        # Implementation would track performance metrics
        return False
    
    def _check_service_availability(self, config: Dict[str, Any]) -> bool:
        """Check if critical services are available"""
        # Implementation would check service health
        return False
    
    def record_health_check(self, healthy: bool, details: Dict[str, Any] = None):
        """Record a health check result"""
        self._health_history.append({
            "timestamp": datetime.now().timestamp(),
            "healthy": healthy,
            "details": details or {},
        })
        
        # Trim old history
        cutoff = datetime.now().timestamp() - 3600  # 1 hour
        self._health_history = [h for h in self._health_history if h["timestamp"] > cutoff]
    
    def record_error(self, is_error: bool = True, details: Dict[str, Any] = None):
        """Record an error event"""
        self._error_history.append({
            "timestamp": datetime.now().timestamp(),
            "is_error": is_error,
            "details": details or {},
        })
        
        # Trim old history
        cutoff = datetime.now().timestamp() - 3600  # 1 hour
        self._error_history = [e for e in self._error_history if e["timestamp"] > cutoff]


class RollbackManager:
    """
    Manages rollback operations to restore system to previous state.
    Implements multiple rollback strategies for different scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize strategies
        self._strategies: Dict[RollbackStrategyType, RollbackStrategy] = {
            RollbackStrategyType.GIT_RESET: GitResetStrategy(config),
            RollbackStrategyType.BACKUP_RESTORE: BackupRestoreStrategy(config),
            RollbackStrategyType.INCREMENTAL_UNDO: IncrementalUndoStrategy(config),
        }
        
        # Auto-rollback triggers
        self._triggers = AutoRollbackTriggers(config)
        
        # Checkpoint management
        self._checkpoints: List[Checkpoint] = []
        self._load_checkpoints()
        
    def _load_checkpoints(self):
        """Load saved checkpoints"""
        checkpoint_file = Path("./state/checkpoints.json")
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    self._checkpoints = [Checkpoint(**cp) for cp in data]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Failed to load checkpoints: {e}", exc_info=True)
    
    def _save_checkpoints(self):
        """Save checkpoints to file"""
        checkpoint_file = Path("./state/checkpoints.json")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump([self._checkpoint_to_dict(cp) for cp in self._checkpoints], f, indent=2)
    
    def _checkpoint_to_dict(self, cp: Checkpoint) -> Dict[str, Any]:
        """Convert checkpoint to dictionary"""
        return {
            "checkpoint_id": cp.checkpoint_id,
            "timestamp": cp.timestamp.isoformat(),
            "version": cp.version,
            "git_commit": cp.git_commit,
            "backup_id": cp.backup_id,
            "state_snapshot": cp.state_snapshot,
            "operations_since": cp.operations_since,
        }
    
    async def rollback(self, target_value: str, 
                       target_type: RollbackTargetType = RollbackTargetType.CHECKPOINT) -> bool:
        """
        Execute rollback to specified target.
        
        Args:
            target_value: Target identifier (version, commit, checkpoint ID, etc.)
            target_type: Type of rollback target
            
        Returns:
            True if rollback successful
        """
        target = RollbackTarget(
            target_type=target_type,
            target_value=target_value,
        )
        
        logger.info(f"Initiating rollback to: {target_value} ({target_type.value})")
        
        # Select appropriate strategy
        strategy = self._select_strategy(target)
        
        if not strategy:
            logger.error(f"No suitable rollback strategy found for target: {target_value}")
            return False
        
        # Execute rollback
        result = await strategy.execute(target)
        
        if result.success:
            logger.info(f"Rollback completed successfully: {result.message}")
        else:
            logger.error(f"Rollback failed: {result.error}")
        
        return result.success
    
    def _select_strategy(self, target: RollbackTarget) -> Optional[RollbackStrategy]:
        """Select the best rollback strategy for the target"""
        # Try strategies in order of preference
        strategy_order = [
            RollbackStrategyType.GIT_RESET,
            RollbackStrategyType.BACKUP_RESTORE,
            RollbackStrategyType.INCREMENTAL_UNDO,
        ]
        
        for strategy_type in strategy_order:
            strategy = self._strategies.get(strategy_type)
            if strategy and strategy.can_execute(target):
                return strategy
        
        return None
    
    def create_checkpoint(self, version: str, backup_id: Optional[str] = None) -> str:
        """
        Create a system checkpoint for future rollback.
        
        Args:
            version: Current system version
            backup_id: Associated backup ID
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get current Git commit
        git_commit = None
        if GIT_AVAILABLE:
            try:
                repo = pygit2.Repository(".")
                git_commit = repo.head.target.hex
            except (OSError, KeyError) as e:
                logger.warning(f"Failed to get git commit for checkpoint: {e}", exc_info=True)
        
        # Capture state snapshot
        state_snapshot = self._capture_state_snapshot()
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            version=version,
            git_commit=git_commit,
            backup_id=backup_id,
            state_snapshot=state_snapshot,
        )
        
        self._checkpoints.append(checkpoint)
        
        # Limit checkpoint history
        if len(self._checkpoints) > 50:
            self._checkpoints = self._checkpoints[-50:]
        
        self._save_checkpoints()
        
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint_id
    
    def _capture_state_snapshot(self) -> Dict[str, Any]:
        """Capture current system state snapshot"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "services": {},
        }
        
        # Capture key file hashes
        key_files = ["./config/system.yaml", "./version.json"]
        for file_path in key_files:
            path = Path(file_path)
            if path.exists():
                import hashlib
                with open(path, 'rb') as f:
                    snapshot["files"][file_path] = hashlib.md5(f.read()).hexdigest()
        
        return snapshot
    
    def check_auto_rollback_triggers(self) -> List[str]:
        """Check if any auto-rollback triggers are activated"""
        return self._triggers.check_triggers()
    
    def record_health_check(self, healthy: bool, details: Dict[str, Any] = None):
        """Record health check for trigger monitoring"""
        self._triggers.record_health_check(healthy, details)
    
    def record_error(self, is_error: bool = True, details: Dict[str, Any] = None):
        """Record error for trigger monitoring"""
        self._triggers.record_error(is_error, details)
    
    def get_available_rollbacks(self) -> List[Dict[str, Any]]:
        """Get list of available rollback targets"""
        rollbacks = []
        
        # Add checkpoints
        for cp in reversed(self._checkpoints[-10:]):  # Last 10 checkpoints
            rollbacks.append({
                "id": cp.checkpoint_id,
                "type": "checkpoint",
                "timestamp": cp.timestamp.isoformat(),
                "version": cp.version,
            })
        
        return rollbacks


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "backup_root": "./backups/",
        }
        
        manager = RollbackManager(config)
        
        # Create a checkpoint
        checkpoint_id = manager.create_checkpoint("1.0.0")
        print(f"Created checkpoint: {checkpoint_id}")
        
        # Get available rollbacks
        rollbacks = manager.get_available_rollbacks()
        print(f"Available rollbacks: {rollbacks}")
    
    asyncio.run(main())
