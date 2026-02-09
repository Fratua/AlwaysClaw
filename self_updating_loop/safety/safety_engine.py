"""
Safety Engine Module
Ensures all modifications are performed safely with proper
validation, backup, and rollback preparation.
"""

import shutil
import hashlib
import json
import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    STATE_ONLY = "state_only"


class RollbackComplexity(Enum):
    """Complexity levels for rollback"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass
class BackupRecord:
    """Record of a created backup"""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    paths: List[str]
    archive_path: str
    checksum: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateOperation:
    """Single update operation"""
    operation_type: str
    source_path: str
    target_path: str
    backup_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StagedUpdate:
    """Update staged for application"""
    stage_id: str
    update_id: str
    timestamp: datetime
    operations: List[UpdateOperation]
    backup_record: Optional[BackupRecord] = None
    status: str = "pending"  # pending, applied, verified, committed, rolled_back


@dataclass
class UpdateResult:
    """Result of an update operation"""
    success: bool
    update_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    rollback_available: bool = False
    rollback_id: Optional[str] = None


class BackupManager:
    """
    Manages backups before updates to enable rollback.
    Implements incremental and full backup strategies.
    """
    
    BACKUP_ROOT = "./backups/"
    MAX_BACKUPS = 10
    
    BACKUP_COMPONENTS = {
        "code": {
            "paths": ["./agents/", "./loops/", "./modules/", "./skills/"],
            "method": "git_snapshot",
        },
        "config": {
            "paths": ["./config/"],
            "method": "full_copy",
        },
        "data": {
            "paths": ["./data/", "./state/"],
            "method": "incremental_archive",
        },
        "database": {
            "paths": ["./database/"],
            "method": "sql_dump",
        },
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_root = Path(config.get("backup_root", self.BACKUP_ROOT))
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._backup_history: List[BackupRecord] = []
        
    def create_backup(self, backup_type: BackupType = BackupType.FULL,
                      specific_paths: Optional[List[str]] = None) -> BackupRecord:
        """
        Create a system backup before update.
        
        Args:
            backup_type: Type of backup to create
            specific_paths: Specific paths to backup (optional)
            
        Returns:
            BackupRecord with backup metadata
        """
        backup_id = self._generate_backup_id()
        timestamp = datetime.now()
        
        # Determine paths to backup
        if specific_paths:
            paths_to_backup = specific_paths
        else:
            paths_to_backup = []
            for component in self.BACKUP_COMPONENTS.values():
                paths_to_backup.extend(component["paths"])
        
        # Create backup archive
        archive_name = f"backup_{backup_id}.zip"
        archive_path = self.backup_root / archive_name
        
        total_size = 0
        files_backed_up = []
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for path_str in paths_to_backup:
                    path = Path(path_str)
                    if not path.exists():
                        logger.warning(f"Path does not exist: {path}")
                        continue
                    
                    if path.is_file():
                        zf.write(path, path.name)
                        total_size += path.stat().st_size
                        files_backed_up.append(str(path))
                    elif path.is_dir():
                        for file_path in path.rglob("*"):
                            if file_path.is_file():
                                # Skip certain files
                                if self._should_skip_file(file_path):
                                    continue
                                
                                arcname = file_path.relative_to(path.parent)
                                zf.write(file_path, arcname)
                                total_size += file_path.stat().st_size
                                files_backed_up.append(str(file_path))
            
            # Calculate checksum
            checksum = self._calculate_checksum(archive_path)
            
            record = BackupRecord(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type=backup_type,
                paths=paths_to_backup,
                archive_path=str(archive_path),
                checksum=checksum,
                size_bytes=total_size,
                metadata={"files_count": len(files_backed_up)},
            )
            
            self._backup_history.append(record)
            self._cleanup_old_backups()
            
            logger.info(f"Backup created: {backup_id} ({total_size} bytes)")
            return record
            
        except (OSError, zipfile.BadZipFile) as e:
            logger.error(f"Backup creation failed: {e}", exc_info=True)
            # Clean up partial backup
            if archive_path.exists():
                archive_path.unlink()
            raise
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of backup to restore
            
        Returns:
            True if restore successful
        """
        record = self._find_backup(backup_id)
        if not record:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        archive_path = Path(record.archive_path)
        if not archive_path.exists():
            logger.error(f"Backup archive not found: {archive_path}")
            return False
        
        try:
            # Verify checksum
            current_checksum = self._calculate_checksum(archive_path)
            if current_checksum != record.checksum:
                logger.error("Backup checksum mismatch - archive may be corrupted")
                return False
            
            # Extract backup
            with zipfile.ZipFile(archive_path, 'r') as zf:
                # First, backup current state
                temp_backup = self.create_backup(BackupType.STATE_ONLY)
                
                # Extract to temporary location first
                extract_path = self.backup_root / f"restore_{backup_id}"
                zf.extractall(extract_path)
                
                # Copy files to target locations
                for root, dirs, files in os.walk(extract_path):
                    for file in files:
                        src = Path(root) / file
                        rel_path = src.relative_to(extract_path)
                        dst = Path(".") / rel_path
                        
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                
                # Clean up extract directory
                shutil.rmtree(extract_path)
            
            logger.info(f"Backup restored: {backup_id}")
            return True
            
        except (OSError, zipfile.BadZipFile, shutil.Error) as e:
            logger.error(f"Backup restore failed: {e}", exc_info=True)
            return False
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        record = self._find_backup(backup_id)
        if not record:
            return False
        
        archive_path = Path(record.archive_path)
        if not archive_path.exists():
            return False
        
        try:
            current_checksum = self._calculate_checksum(archive_path)
            return current_checksum == record.checksum
        except OSError:
            logger.warning(f"Backup verification failed for: {backup_id}", exc_info=True)
            return False
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during backup"""
        skip_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.git',
            '*.log',
            '*.tmp',
            '*.bak',
            '.env',
            '*.key',
            '*.secret',
        ]
        
        path_str = str(file_path)
        for pattern in skip_patterns:
            if pattern in path_str or file_path.match(pattern):
                return True
        
        return False
    
    def _find_backup(self, backup_id: str) -> Optional[BackupRecord]:
        """Find backup record by ID"""
        for record in self._backup_history:
            if record.backup_id == backup_id:
                return record
        return None
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding max count"""
        max_backups = self.config.get("max_backups", self.MAX_BACKUPS)
        
        while len(self._backup_history) > max_backups:
            old_record = self._backup_history.pop(0)
            archive_path = Path(old_record.archive_path)
            if archive_path.exists():
                archive_path.unlink()
                logger.debug(f"Removed old backup: {old_record.backup_id}")


class AtomicUpdateExecutor:
    """
    Ensures updates are applied atomically - either fully applied
    or fully rolled back, preventing partial update states.
    """
    
    OPERATION_TYPES = {
        "file_write": "write_file",
        "file_delete": "delete_file",
        "directory_create": "create_directory",
        "config_update": "update_config",
        "service_restart": "restart_service",
        "registry_update": "update_registry",
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._operations_log: List[Dict[str, Any]] = []
        
    async def execute_atomic(self, operations: List[UpdateOperation]) -> bool:
        """
        Execute a series of update operations atomically.
        
        Args:
            operations: List of operations to execute
            
        Returns:
            True if all operations succeeded
        """
        executed = []
        
        try:
            # Phase 1: Validate all operations can be applied
            for op in operations:
                if not self._validate_operation(op):
                    logger.error(f"Operation validation failed: {op}")
                    return False
            
            # Phase 2: Create backups for rollback
            backups = []
            for op in operations:
                backup = self._create_operation_backup(op)
                backups.append(backup)
            
            # Phase 3: Apply all operations
            for i, op in enumerate(operations):
                success = await self._execute_operation(op)
                if not success:
                    logger.error(f"Operation failed: {op}")
                    # Rollback already executed operations
                    await self._rollback_operations(executed, backups[:len(executed)])
                    return False
                executed.append(op)
            
            # Phase 4: Verify all operations
            for op in operations:
                if not self._verify_operation(op):
                    logger.error(f"Operation verification failed: {op}")
                    await self._rollback_operations(executed, backups)
                    return False
            
            logger.info(f"Atomic update executed: {len(operations)} operations")
            return True
            
        except (OSError, json.JSONDecodeError, KeyError, ValueError, shutil.Error) as e:
            logger.exception(f"Atomic update failed: {e}")
            await self._rollback_operations(executed, backups[:len(executed)])
            return False
    
    def _validate_operation(self, operation: UpdateOperation) -> bool:
        """Validate an operation can be applied"""
        op_type = operation.operation_type
        
        if op_type == "file_write":
            source = Path(operation.source_path)
            target = Path(operation.target_path)
            
            # Check source exists
            if not source.exists():
                logger.error(f"Source file does not exist: {source}")
                return False
            
            # Check target directory exists or can be created
            if not target.parent.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(f"Cannot create target directory: {e}")
                    return False
            
            return True
            
        elif op_type == "file_delete":
            target = Path(operation.target_path)
            return target.exists()
            
        elif op_type == "directory_create":
            target = Path(operation.target_path)
            return not target.exists()  # Should not exist yet
            
        return True
    
    def _create_operation_backup(self, operation: UpdateOperation) -> Optional[str]:
        """Create backup for single operation"""
        if operation.operation_type in ["file_write", "file_delete"]:
            target = Path(operation.target_path)
            if target.exists():
                backup_path = f"{operation.target_path}.backup"
                shutil.copy2(target, backup_path)
                return backup_path
        return None
    
    async def _execute_operation(self, operation: UpdateOperation) -> bool:
        """Execute a single operation"""
        try:
            op_type = operation.operation_type
            
            if op_type == "file_write":
                source = Path(operation.source_path)
                target = Path(operation.target_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                
            elif op_type == "file_delete":
                target = Path(operation.target_path)
                target.unlink()
                
            elif op_type == "directory_create":
                target = Path(operation.target_path)
                target.mkdir(parents=True, exist_ok=True)
                
            elif op_type == "config_update":
                # Update configuration file
                config_path = Path(operation.target_path)
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Apply updates from metadata
                updates = operation.metadata.get("updates", {})
                config.update(updates)
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Log operation
            self._operations_log.append({
                "timestamp": datetime.now().isoformat(),
                "operation": operation.operation_type,
                "target": operation.target_path,
            })
            
            return True
            
        except (OSError, json.JSONDecodeError, KeyError, shutil.Error) as e:
            logger.error(f"Operation execution failed: {e}", exc_info=True)
            return False
    
    def _verify_operation(self, operation: UpdateOperation) -> bool:
        """Verify an operation was applied correctly"""
        op_type = operation.operation_type
        
        if op_type == "file_write":
            target = Path(operation.target_path)
            return target.exists()
            
        elif op_type == "file_delete":
            target = Path(operation.target_path)
            return not target.exists()
            
        elif op_type == "directory_create":
            target = Path(operation.target_path)
            return target.exists() and target.is_dir()
        
        return True
    
    async def _rollback_operations(self, operations: List[UpdateOperation],
                                    backups: List[Optional[str]]):
        """Rollback executed operations"""
        logger.info(f"Rolling back {len(operations)} operations")
        
        # Rollback in reverse order
        for op, backup in reversed(list(zip(operations, backups))):
            try:
                if backup and Path(backup).exists():
                    # Restore from backup
                    shutil.copy2(backup, op.target_path)
                    Path(backup).unlink()
                elif op.operation_type == "directory_create":
                    # Remove created directory
                    target = Path(op.target_path)
                    if target.exists():
                        shutil.rmtree(target)
            except (OSError, shutil.Error) as e:
                logger.error(f"Rollback failed for operation: {e}", exc_info=True)


class SafetyEngine:
    """
    Ensures all modifications are performed safely with proper
    validation, backup, and rollback preparation.
    """
    
    SAFETY_CHECKS = [
        "pre_update_backup",
        "dependency_validation",
        "conflict_detection",
        "resource_availability",
        "state_preservation",
        "permission_validation",
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_manager = BackupManager(config)
        self.atomic_executor = AtomicUpdateExecutor(config)
        self._staged_updates: Dict[str, StagedUpdate] = {}
        
    async def apply_update(self, update: Any, impact: Any) -> UpdateResult:
        """
        Execute an update with comprehensive safety measures.
        
        Flow:
        1. Pre-flight checks
        2. Create backup
        3. Stage changes
        4. Apply changes
        5. Verify application
        6. Commit or rollback
        
        Args:
            update: Update event to apply
            impact: Impact analysis report
            
        Returns:
            UpdateResult with success status and details
        """
        update_id = update.event_id if hasattr(update, 'event_id') else str(update)
        
        try:
            # Step 1: Pre-flight checks
            if not self._run_preflight_checks():
                return UpdateResult(
                    success=False,
                    update_id=update_id,
                    error="Pre-flight checks failed",
                )
            
            # Step 2: Create backup
            if self.config.get("backup_before_update", True):
                backup = self.backup_manager.create_backup(BackupType.FULL)
                logger.info(f"Backup created: {backup.backup_id}")
            else:
                backup = None
            
            # Step 3: Stage changes
            stage_id = self._stage_update(update, backup)
            
            # Step 4: Apply changes
            operations = self._create_operations(update)
            success = await self.atomic_executor.execute_atomic(operations)
            
            if not success:
                return UpdateResult(
                    success=False,
                    update_id=update_id,
                    error="Atomic update execution failed",
                    rollback_available=backup is not None,
                    rollback_id=backup.backup_id if backup else None,
                )
            
            # Step 5: Verify application
            verified = self._verify_update(update)
            
            if not verified:
                return UpdateResult(
                    success=False,
                    update_id=update_id,
                    error="Update verification failed",
                    rollback_available=backup is not None,
                    rollback_id=backup.backup_id if backup else None,
                )
            
            # Update staged status
            if stage_id in self._staged_updates:
                self._staged_updates[stage_id].status = "verified"
            
            return UpdateResult(
                success=True,
                update_id=update_id,
                message="Update applied successfully",
                rollback_available=backup is not None,
                rollback_id=backup.backup_id if backup else None,
            )
            
        except (OSError, ValueError, KeyError, AttributeError) as e:
            logger.exception(f"Update application failed: {e}")
            return UpdateResult(
                success=False,
                update_id=update_id,
                error=str(e),
            )
    
    def _run_preflight_checks(self) -> bool:
        """Run pre-flight safety checks"""
        checks = {
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory(),
            "permissions": self._check_permissions(),
        }
        
        failed = [name for name, passed in checks.items() if not passed]
        
        if failed:
            logger.error(f"Pre-flight checks failed: {failed}")
            return False
        
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024**3)
            required_gb = self.config.get("min_free_space_gb", 1.0)
            
            if free_gb < required_gb:
                logger.warning(f"Low disk space: {free_gb:.2f}GB free, {required_gb}GB required")
                return False
            
            return True
        except OSError as e:
            logger.error(f"Disk space check failed: {e}", exc_info=True)
            return False
    
    def _check_memory(self) -> bool:
        """Check available memory"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            required_gb = self.config.get("min_free_memory_gb", 0.5)
            
            if available_gb < required_gb:
                logger.warning(f"Low memory: {available_gb:.2f}GB available, {required_gb}GB required")
                return False
            
            return True
        except ImportError:
            # psutil not available, skip check
            return True
        except OSError as e:
            logger.error(f"Memory check failed: {e}", exc_info=True)
            return False
    
    def _check_permissions(self) -> bool:
        """Check required permissions"""
        try:
            # Check write permission to key directories
            test_paths = ["./", "./config/", "./logs/"]
            
            for path_str in test_paths:
                path = Path(path_str)
                if path.exists() and not os.access(path, os.W_OK):
                    logger.error(f"No write permission: {path}")
                    return False
            
            return True
        except OSError as e:
            logger.error(f"Permission check failed: {e}", exc_info=True)
            return False
    
    def _stage_update(self, update: Any, backup: Optional[BackupRecord]) -> str:
        """Stage an update for application"""
        stage_id = hashlib.md5(
            f"{update.event_id}{datetime.now().timestamp()}".encode()
        ).hexdigest()[:12]
        
        operations = self._create_operations(update)
        
        staged = StagedUpdate(
            stage_id=stage_id,
            update_id=update.event_id if hasattr(update, 'event_id') else str(update),
            timestamp=datetime.now(),
            operations=operations,
            backup_record=backup,
            status="staged",
        )
        
        self._staged_updates[stage_id] = staged
        logger.info(f"Update staged: {stage_id}")
        
        return stage_id
    
    def _create_operations(self, update: Any) -> List[UpdateOperation]:
        """Create update operations from update event."""
        operations = []

        if not hasattr(update, 'changed_files'):
            return operations

        # Determine update package source directory
        # Priority: update metadata -> git staging area -> default staging dir
        source_root = None

        # Check update metadata for an explicit package path
        if hasattr(update, 'metadata') and isinstance(update.metadata, dict):
            pkg_path = update.metadata.get("package_path")
            if pkg_path and Path(pkg_path).is_dir():
                source_root = Path(pkg_path)

        # Try git-based source: if the update carries a commit hash,
        # check for an extracted worktree in the staging area
        if source_root is None and hasattr(update, 'git_commit_hash') and update.git_commit_hash:
            git_staging = Path("./staging/git") / update.git_commit_hash
            if git_staging.is_dir():
                source_root = git_staging

        # Try remote source: look for downloaded archive extraction
        if source_root is None and hasattr(update, 'remote_url') and update.remote_url:
            event_id = update.event_id if hasattr(update, 'event_id') else "unknown"
            remote_staging = Path("./staging/remote") / event_id
            if remote_staging.is_dir():
                source_root = remote_staging

        # Fallback to the default staging directory
        if source_root is None:
            source_root = Path("./staging")

        for file_path in update.changed_files:
            source = source_root / file_path
            target = file_path

            operations.append(UpdateOperation(
                operation_type="file_write",
                source_path=str(source),
                target_path=target,
            ))

        return operations
    
    def _verify_update(self, update: Any) -> bool:
        """Verify update was applied correctly"""
        # Check all changed files exist
        if hasattr(update, 'changed_files'):
            for file_path in update.changed_files:
                if not Path(file_path).exists():
                    logger.error(f"Updated file does not exist: {file_path}")
                    return False
        
        return True
    
    def cleanup_staging(self):
        """Clean up staged updates"""
        # Remove old staged updates
        cutoff = datetime.now() - timedelta(hours=24)
        
        to_remove = [
            stage_id for stage_id, staged in self._staged_updates.items()
            if staged.timestamp < cutoff and staged.status in ["committed", "rolled_back"]
        ]
        
        for stage_id in to_remove:
            del self._staged_updates[stage_id]
        
        logger.debug(f"Cleaned up {len(to_remove)} staged updates")


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "backup_before_update": True,
            "min_free_space_gb": 0.5,
        }
        
        engine = SafetyEngine(config)
        
        # Create a test update
        @dataclass
        class TestUpdate:
            event_id: str
            changed_files: List[str]
        
        update = TestUpdate(
            event_id="test_123",
            changed_files=["./test_file.txt"],
        )
        
        # Create test file
        Path("./test_file.txt").write_text("test content")
        
        # Apply update
        result = await engine.apply_update(update, None)
        print(f"Update result: {result}")
    
    asyncio.run(main())
