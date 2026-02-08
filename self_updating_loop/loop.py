"""
Self-Updating Loop - Main Implementation
OpenClaw-Inspired AI Agent System - Windows 10 Edition

This module implements the core self-updating loop that enables autonomous
code updates, configuration management, and safe self-modification capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import hashlib
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UpdateLoopState(Enum):
    """States of the self-updating loop lifecycle"""
    IDLE = "idle"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    STAGING = "staging"
    APPROVING = "approving"
    APPLYING = "applying"
    VERIFYING = "verifying"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    ERROR = "error"


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


class RiskLevel(Enum):
    """Risk levels for updates"""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


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


@dataclass
class ImpactReport:
    """Comprehensive impact analysis report"""
    directly_affected: List[str] = field(default_factory=list)
    indirectly_affected: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MINIMAL
    risk_factors: List[str] = field(default_factory=list)
    required_tests: List[str] = field(default_factory=list)
    estimated_test_time: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    rollback_complexity: str = "simple"
    rollback_time_estimate: timedelta = field(default_factory=lambda: timedelta(minutes=2))
    recommendations: List[str] = field(default_factory=list)
    approval_required: bool = True


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


@dataclass
class AuditRecord:
    """Immutable audit record for update operations"""
    record_id: str
    timestamp: datetime
    event_type: str
    actor_type: str
    actor_id: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    update_id: Optional[str] = None
    version_from: Optional[str] = None
    version_to: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    checksum: str = ""
    previous_record_hash: str = ""


class SelfUpdatingLoop:
    """
    Main self-updating loop implementation.
    
    This class orchestrates the entire self-update process including:
    - Update detection from multiple sources
    - Change analysis and impact assessment
    - Safe modification with backup and rollback
    - Version control integration
    - Update validation and testing
    - Configuration migration
    - Comprehensive audit logging
    """
    
    DEFAULT_CONFIG = {
        "enabled": True,
        "check_interval": 300,  # 5 minutes
        "auto_apply_patch": True,
        "auto_apply_minor": False,
        "auto_apply_major": False,
        "backup_before_update": True,
        "test_before_apply": True,
        "auto_rollback_on_failure": True,
        "health_check_interval": 30,
        "max_retries": 3,
        "version_scheme": "semver",
        "log_level": "INFO",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the self-updating loop.
        
        Args:
            config: Configuration dictionary overriding defaults
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.state = UpdateLoopState.IDLE
        self.state_lock = threading.RLock()
        
        # Component references (initialized lazily)
        self._detector = None
        self._analyzer = None
        self._safety_engine = None
        self._version_manager = None
        self._rollback_manager = None
        self._validator = None
        self._config_migrator = None
        self._audit_logger = None
        
        # Runtime state
        self._running = False
        self._current_update: Optional[UpdateEvent] = None
        self._update_history: List[UpdateResult] = []
        self._pending_updates: List[UpdateEvent] = []
        
        # Event handlers
        self._state_handlers: Dict[UpdateLoopState, List[Callable]] = {
            state: [] for state in UpdateLoopState
        }
        
        logger.info("SelfUpdatingLoop initialized")
    
    @property
    def detector(self):
        """Lazy initialization of update detector"""
        if self._detector is None:
            from detection.update_detector import UpdateDetector
            self._detector = UpdateDetector(self.config)
        return self._detector
    
    @property
    def analyzer(self):
        """Lazy initialization of change analyzer"""
        if self._analyzer is None:
            from analysis.change_analyzer import ChangeAnalyzer
            self._analyzer = ChangeAnalyzer(self.config)
        return self._analyzer
    
    @property
    def safety_engine(self):
        """Lazy initialization of safety engine"""
        if self._safety_engine is None:
            from safety.safety_engine import SafetyEngine
            self._safety_engine = SafetyEngine(self.config)
        return self._safety_engine
    
    @property
    def version_manager(self):
        """Lazy initialization of version manager"""
        if self._version_manager is None:
            from version.version_manager import VersionManager
            self._version_manager = VersionManager(self.config)
        return self._version_manager
    
    @property
    def rollback_manager(self):
        """Lazy initialization of rollback manager"""
        if self._rollback_manager is None:
            from rollback.rollback_manager import RollbackManager
            self._rollback_manager = RollbackManager(self.config)
        return self._rollback_manager
    
    @property
    def validator(self):
        """Lazy initialization of update validator"""
        if self._validator is None:
            from validation.update_validator import UpdateValidator
            self._validator = UpdateValidator(self.config)
        return self._validator
    
    @property
    def config_migrator(self):
        """Lazy initialization of config migrator"""
        if self._config_migrator is None:
            from config_migration.migration_manager import ConfigMigrationManager
            self._config_migrator = ConfigMigrationManager(self.config)
        return self._config_migrator
    
    @property
    def audit_logger(self):
        """Lazy initialization of audit logger"""
        if self._audit_logger is None:
            from audit.logger import UpdateAuditLogger
            self._audit_logger = UpdateAuditLogger(self.config)
        return self._audit_logger
    
    def get_state(self) -> UpdateLoopState:
        """Get current loop state (thread-safe)"""
        with self.state_lock:
            return self.state
    
    def set_state(self, new_state: UpdateLoopState):
        """Set loop state (thread-safe)"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
            
            # Trigger state change handlers
            for handler in self._state_handlers.get(new_state, []):
                try:
                    handler(old_state, new_state)
                except Exception as e:
                    logger.error(f"State handler error: {e}")
    
    def on_state_change(self, state: UpdateLoopState, handler: Callable):
        """Register a handler for state changes"""
        self._state_handlers[state].append(handler)
    
    async def start(self):
        """Start the self-updating loop"""
        if self._running:
            logger.warning("Self-updating loop is already running")
            return
        
        self._running = True
        logger.info("Self-updating loop started")
        
        self.audit_logger.log_event(
            event_type="LOOP_STARTED",
            actor_type="system",
            actor_id="self_updating_loop",
            description="Self-updating loop started",
        )
        
        try:
            while self._running:
                await self._iteration()
                await asyncio.sleep(self.config["check_interval"])
        except asyncio.CancelledError:
            logger.info("Self-updating loop cancelled")
        except Exception as e:
            logger.exception("Error in self-updating loop")
            self.set_state(UpdateLoopState.ERROR)
            raise
    
    async def stop(self):
        """Stop the self-updating loop gracefully"""
        logger.info("Stopping self-updating loop...")
        self._running = False
        
        self.audit_logger.log_event(
            event_type="LOOP_STOPPED",
            actor_type="system",
            actor_id="self_updating_loop",
            description="Self-updating loop stopped",
        )
    
    async def _iteration(self):
        """Single iteration of the update loop"""
        try:
            # Step 1: Detect updates
            self.set_state(UpdateLoopState.DETECTING)
            updates = await self._detect_updates()
            
            if not updates:
                self.set_state(UpdateLoopState.IDLE)
                return
            
            logger.info(f"Detected {len(updates)} potential updates")
            
            # Process each update
            for update in updates:
                if not self._running:
                    break
                
                await self._process_update(update)
                
        except Exception as e:
            logger.exception("Error in update iteration")
            self.set_state(UpdateLoopState.ERROR)
    
    async def _detect_updates(self) -> List[UpdateEvent]:
        """Detect available updates from all sources"""
        updates = []
        
        try:
            updates = await self.detector.check_all_sources()
            
            # Log detection
            if updates:
                self.audit_logger.log_event(
                    event_type="UPDATES_DETECTED",
                    actor_type="system",
                    actor_id="detector",
                    description=f"Detected {len(updates)} updates",
                    details={"count": len(updates), "sources": list(set(u.source.value for u in updates))},
                )
        except Exception as e:
            logger.error(f"Update detection failed: {e}")
            self.audit_logger.log_event(
                event_type="DETECTION_FAILED",
                actor_type="system",
                actor_id="detector",
                description="Update detection failed",
                details={"error": str(e)},
                success=False,
                error_message=str(e),
            )
        
        return updates
    
    async def _process_update(self, update: UpdateEvent):
        """Process a single update through the full pipeline"""
        self._current_update = update
        
        logger.info(f"Processing update: {update.event_id} ({update.update_type.value})")
        
        try:
            # Step 2: Analyze changes
            self.set_state(UpdateLoopState.ANALYZING)
            impact = await self._analyze_update(update)
            
            if impact.risk_level == RiskLevel.CRITICAL and not self._should_auto_apply(update):
                logger.warning(f"Update {update.event_id} has critical risk - requires manual approval")
                self.set_state(UpdateLoopState.APPROVING)
                # In a real implementation, this would notify for approval
                return
            
            # Step 3: Validate update
            self.set_state(UpdateLoopState.VALIDATING)
            if not await self._validate_update(update):
                logger.error(f"Update {update.event_id} validation failed")
                return
            
            # Step 4: Apply update with safety
            result = await self._apply_update_safely(update, impact)
            
            if result.success:
                self._update_history.append(result)
                logger.info(f"Update {update.event_id} applied successfully")
            else:
                logger.error(f"Update {update.event_id} failed: {result.error}")
                
                # Attempt rollback if enabled
                if self.config["auto_rollback_on_failure"] and result.rollback_available:
                    await self._rollback_update(result.rollback_id)
            
        except Exception as e:
            logger.exception(f"Error processing update {update.event_id}")
            self.set_state(UpdateLoopState.ERROR)
        finally:
            self._current_update = None
            self.set_state(UpdateLoopState.IDLE)
    
    async def _analyze_update(self, update: UpdateEvent) -> ImpactReport:
        """Analyze the impact of an update"""
        self.set_state(UpdateLoopState.ANALYZING)
        
        impact = self.analyzer.analyze(update)
        
        self.audit_logger.log_event(
            event_type="ANALYSIS_COMPLETED",
            actor_type="system",
            actor_id="analyzer",
            description=f"Impact analysis completed for {update.event_id}",
            details={
                "risk_level": impact.risk_level.name,
                "approval_required": impact.approval_required,
            },
            update_id=update.event_id,
        )
        
        return impact
    
    async def _validate_update(self, update: UpdateEvent) -> bool:
        """Validate an update before application"""
        self.set_state(UpdateLoopState.VALIDATING)
        
        valid = self.validator.validate(update)
        
        self.audit_logger.log_event(
            event_type="VALIDATION_COMPLETED" if valid else "VALIDATION_FAILED",
            actor_type="system",
            actor_id="validator",
            description=f"Validation {'passed' if valid else 'failed'} for {update.event_id}",
            update_id=update.event_id,
            success=valid,
        )
        
        return valid
    
    async def _apply_update_safely(self, update: UpdateEvent, 
                                    impact: ImpactReport) -> UpdateResult:
        """Apply an update with full safety measures"""
        self.set_state(UpdateLoopState.STAGING)
        
        # Check if auto-apply is enabled for this update type
        if not self._should_auto_apply(update):
            logger.info(f"Update {update.event_id} requires manual approval")
            self.set_state(UpdateLoopState.APPROVING)
            return UpdateResult(
                success=False,
                update_id=update.event_id,
                message="Update requires manual approval",
            )
        
        self.set_state(UpdateLoopState.APPLYING)
        
        # Apply with safety engine
        result = await self.safety_engine.apply_update(update, impact)
        
        if result.success:
            self.set_state(UpdateLoopState.VERIFYING)
            
            # Post-update verification
            verified = await self._verify_update(update)
            
            if verified:
                self.set_state(UpdateLoopState.COMMITTING)
                await self._commit_update(update)
            else:
                # Verification failed - rollback
                if result.rollback_available:
                    await self._rollback_update(result.rollback_id)
                result.success = False
                result.error = "Post-update verification failed"
        
        return result
    
    def _should_auto_apply(self, update: UpdateEvent) -> bool:
        """Determine if update should be auto-applied based on config"""
        if update.update_type == UpdateType.PATCH:
            return self.config["auto_apply_patch"]
        elif update.update_type == UpdateType.MINOR:
            return self.config["auto_apply_minor"]
        elif update.update_type == UpdateType.MAJOR:
            return self.config["auto_apply_major"]
        elif update.update_type == UpdateType.HOTFIX:
            return True  # Always auto-apply hotfixes
        elif update.update_type == UpdateType.SECURITY:
            return True  # Always auto-apply security updates
        return False
    
    async def _verify_update(self, update: UpdateEvent) -> bool:
        """Verify update was applied correctly"""
        # Run health checks
        health_ok = self.validator.run_health_checks()
        
        # Verify version changed
        current_version = self.version_manager.get_current_version()
        version_ok = current_version == update.target_version
        
        return health_ok and version_ok
    
    async def _commit_update(self, update: UpdateEvent):
        """Finalize the update"""
        # Update version tracking
        self.version_manager.record_update(
            version=update.target_version,
            previous_version=update.current_version,
            update_type=update.update_type,
            changes=update.changed_files,
        )
        
        # Cleanup
        self.safety_engine.cleanup_staging()
        
        self.audit_logger.log_event(
            event_type="UPDATE_COMPLETED",
            actor_type="system",
            actor_id="self_updating_loop",
            description=f"Update {update.event_id} completed successfully",
            update_id=update.event_id,
            version_from=update.current_version,
            version_to=update.target_version,
        )
    
    async def _rollback_update(self, rollback_id: str):
        """Rollback a failed update"""
        self.set_state(UpdateLoopState.ROLLING_BACK)
        
        logger.info(f"Rolling back update with rollback ID: {rollback_id}")
        
        result = self.rollback_manager.rollback(rollback_id)
        
        self.audit_logger.log_event(
            event_type="ROLLBACK_COMPLETED" if result else "ROLLBACK_FAILED",
            actor_type="system",
            actor_id="rollback_manager",
            description=f"Rollback {rollback_id} {'completed' if result else 'failed'}",
            success=result,
        )
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loop status"""
        return {
            "state": self.get_state().value,
            "running": self._running,
            "current_update": self._current_update.event_id if self._current_update else None,
            "pending_updates": len(self._pending_updates),
            "update_history_count": len(self._update_history),
            "config": {
                k: v for k, v in self.config.items() 
                if not k.endswith("_key") and not k.endswith("_secret")
            },
        }
    
    def trigger_manual_update(self, update_source: str = "manual") -> str:
        """Trigger a manual update check"""
        # This would trigger an immediate update check
        logger.info(f"Manual update triggered from {update_source}")
        
        # In async context, this would schedule an immediate check
        # For now, just log the request
        self.audit_logger.log_event(
            event_type="MANUAL_UPDATE_TRIGGERED",
            actor_type="user",
            actor_id=update_source,
            description="Manual update check triggered",
        )
        
        return "Update check scheduled"


# Singleton instance
_self_updating_loop: Optional[SelfUpdatingLoop] = None


def get_self_updating_loop(config: Optional[Dict[str, Any]] = None) -> SelfUpdatingLoop:
    """Get or create the singleton self-updating loop instance"""
    global _self_updating_loop
    if _self_updating_loop is None:
        _self_updating_loop = SelfUpdatingLoop(config)
    return _self_updating_loop


if __name__ == "__main__":
    # Example usage
    async def main():
        loop = get_self_updating_loop()
        
        # Run for a short time as demonstration
        try:
            await asyncio.wait_for(loop.start(), timeout=10)
        except asyncio.TimeoutError:
            await loop.stop()
    
    asyncio.run(main())
