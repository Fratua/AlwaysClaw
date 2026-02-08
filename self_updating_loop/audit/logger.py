"""
Audit Logger Module
Comprehensive logging for all update operations.
Maintains immutable audit trail.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for audit events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of audit events"""
    # Loop lifecycle events
    LOOP_STARTED = "loop_started"
    LOOP_STOPPED = "loop_stopped"
    LOOP_ERROR = "loop_error"
    
    # Detection events
    UPDATES_DETECTED = "updates_detected"
    UPDATE_DETECTED = "update_detected"
    DETECTION_FAILED = "detection_failed"
    UPDATE_DISMISSED = "update_dismissed"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    IMPACT_ASSESSED = "impact_assessed"
    
    # Update events
    UPDATE_STARTED = "update_started"
    BACKUP_CREATED = "backup_created"
    STAGE_COMPLETED = "stage_completed"
    UPDATE_COMPLETED = "update_completed"
    UPDATE_FAILED = "update_failed"
    
    # Rollback events
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    ROLLBACK_FAILED = "rollback_failed"
    
    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    
    # Configuration events
    CONFIG_MIGRATED = "config_migrated"
    CONFIG_BACKUP = "config_backup"
    
    # Manual events
    MANUAL_UPDATE_TRIGGERED = "manual_update_triggered"
    MANUAL_ROLLBACK_TRIGGERED = "manual_rollback_triggered"


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
    
    def __post_init__(self):
        if not self.record_id:
            self.record_id = self._generate_record_id()
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID"""
        data = f"{self.timestamp.isoformat()}{self.event_type}{self.actor_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self) -> str:
        """Calculate record checksum for integrity verification"""
        # Exclude checksum and previous_record_hash from calculation
        data = {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "description": self.description,
            "details": self.details,
            "update_id": self.update_id,
            "version_from": self.version_from,
            "version_to": self.version_to,
            "success": self.success,
            "error_message": self.error_message,
            "previous_record_hash": self.previous_record_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify record integrity"""
        return self.checksum == self._calculate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary"""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "description": self.description,
            "details": self.details,
            "update_id": self.update_id,
            "version_from": self.version_from,
            "version_to": self.version_to,
            "success": self.success,
            "error_message": self.error_message,
            "checksum": self.checksum,
            "previous_record_hash": self.previous_record_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create record from dictionary"""
        return cls(
            record_id=data.get("record_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=data["event_type"],
            actor_type=data["actor_type"],
            actor_id=data["actor_id"],
            description=data["description"],
            details=data.get("details", {}),
            update_id=data.get("update_id"),
            version_from=data.get("version_from"),
            version_to=data.get("version_to"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            checksum=data.get("checksum", ""),
            previous_record_hash=data.get("previous_record_hash", ""),
        )


class LogStorageManager:
    """
    Manages storage and retention of update logs.
    """
    
    STORAGE_CONFIG = {
        "local_path": "./logs/updates/",
        "max_local_size": "1GB",
        "retention_days": 90,
        "archive_after_days": 30,
        "compression": "gzip",
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_path = Path(config.get("log_path", self.STORAGE_CONFIG["local_path"]))
        self.local_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = config.get("retention_days", self.STORAGE_CONFIG["retention_days"])
        self.archive_after_days = config.get("archive_after_days", self.STORAGE_CONFIG["archive_after_days"])
        
        self._current_log_file: Optional[Path] = None
        self._records_in_current_file = 0
        self._max_records_per_file = 1000
        
    def _get_current_log_file(self) -> Path:
        """Get or create current log file"""
        if self._current_log_file is None or self._records_in_current_file >= self._max_records_per_file:
            # Create new log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._current_log_file = self.local_path / f"audit_{timestamp}.jsonl"
            self._records_in_current_file = 0
        
        return self._current_log_file
    
    def store_record(self, record: AuditRecord):
        """Store audit record"""
        log_file = self._get_current_log_file()
        
        with open(log_file, 'a') as f:
            json.dump(record.to_dict(), f)
            f.write('\n')
        
        self._records_in_current_file += 1
    
    def read_records(self, start: datetime, end: datetime) -> List[AuditRecord]:
        """Read records from time period"""
        records = []
        
        for log_file in self.local_path.glob("audit_*.jsonl"):
            # Parse timestamp from filename
            try:
                file_timestamp = datetime.strptime(
                    log_file.stem.replace("audit_", ""),
                    "%Y%m%d_%H%M%S"
                )
                
                # Skip files outside time range
                if file_timestamp < start - timedelta(days=1) or file_timestamp > end:
                    continue
                
                with open(log_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            record = AuditRecord.from_dict(data)
                            
                            # Filter by timestamp
                            if start <= record.timestamp <= end:
                                records.append(record)
                        except Exception as e:
                            logger.warning(f"Failed to parse audit record: {e}")
                            
            except ValueError:
                continue
        
        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)
        
        return records
    
    def archive_old_logs(self):
        """Archive logs older than archive_after_days"""
        cutoff = datetime.now() - timedelta(days=self.archive_after_days)
        
        for log_file in self.local_path.glob("audit_*.jsonl"):
            try:
                file_timestamp = datetime.strptime(
                    log_file.stem.replace("audit_", ""),
                    "%Y%m%d_%H%M%S"
                )
                
                if file_timestamp < cutoff:
                    # Archive the file
                    archive_path = self.local_path / "archive"
                    archive_path.mkdir(exist_ok=True)
                    
                    import gzip
                    archive_file = archive_path / f"{log_file.name}.gz"
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_file, 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    log_file.unlink()
                    logger.debug(f"Archived log file: {log_file.name}")
                    
            except Exception as e:
                logger.error(f"Failed to archive log file: {e}")
    
    def cleanup_old_records(self):
        """Remove records older than retention_days"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove old archived files
        archive_path = self.local_path / "archive"
        if archive_path.exists():
            for archive_file in archive_path.glob("*.gz"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = archive_file.name.replace("audit_", "").replace(".jsonl.gz", "")
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if file_timestamp < cutoff:
                        archive_file.unlink()
                        logger.debug(f"Removed old archive: {archive_file.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to cleanup archive: {e}")


class ComplianceReporter:
    """
    Generates compliance reports from audit logs.
    """
    
    REPORT_TYPES = {
        "update_history": "Complete update history",
        "failed_updates": "Failed updates analysis",
        "rollback_analysis": "Rollback frequency and causes",
        "security_updates": "Security-related updates",
        "performance_impact": "Update performance impact",
        "audit_summary": "Audit trail summary",
    }
    
    def __init__(self, storage: LogStorageManager):
        self.storage = storage
    
    def generate_report(self, report_type: str, 
                        start: datetime,
                        end: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        if report_type == "update_history":
            return self._generate_update_history_report(start, end)
        elif report_type == "failed_updates":
            return self._generate_failed_updates_report(start, end)
        elif report_type == "rollback_analysis":
            return self._generate_rollback_analysis_report(start, end)
        elif report_type == "audit_summary":
            return self._generate_audit_summary_report(start, end)
        else:
            return {"error": f"Unknown report type: {report_type}"}
    
    def _generate_update_history_report(self, start: datetime, 
                                        end: datetime) -> Dict[str, Any]:
        """Generate update history report"""
        records = self.storage.read_records(start, end)
        
        update_records = [
            r for r in records 
            if r.event_type in [EventType.UPDATE_COMPLETED.value, EventType.UPDATE_FAILED.value]
        ]
        
        updates = []
        for record in update_records:
            updates.append({
                "timestamp": record.timestamp.isoformat(),
                "event_type": record.event_type,
                "update_id": record.update_id,
                "version_from": record.version_from,
                "version_to": record.version_to,
                "success": record.success,
                "description": record.description,
            })
        
        return {
            "report_type": "update_history",
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_updates": len(updates),
            "successful_updates": len([u for u in updates if u["success"]]),
            "failed_updates": len([u for u in updates if not u["success"]]),
            "updates": updates,
        }
    
    def _generate_failed_updates_report(self, start: datetime, 
                                        end: datetime) -> Dict[str, Any]:
        """Generate failed updates analysis report"""
        records = self.storage.read_records(start, end)
        
        failed_records = [
            r for r in records 
            if r.event_type == EventType.UPDATE_FAILED.value
        ]
        
        failures = []
        for record in failed_records:
            failures.append({
                "timestamp": record.timestamp.isoformat(),
                "update_id": record.update_id,
                "error_message": record.error_message,
                "description": record.description,
            })
        
        # Analyze error patterns
        error_types: Dict[str, int] = {}
        for failure in failures:
            error = failure.get("error_message", "Unknown")
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "report_type": "failed_updates",
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_failures": len(failures),
            "failures": failures,
            "error_patterns": error_types,
        }
    
    def _generate_rollback_analysis_report(self, start: datetime, 
                                           end: datetime) -> Dict[str, Any]:
        """Generate rollback analysis report"""
        records = self.storage.read_records(start, end)
        
        rollback_records = [
            r for r in records 
            if r.event_type in [EventType.ROLLBACK_COMPLETED.value, EventType.ROLLBACK_FAILED.value]
        ]
        
        rollbacks = []
        for record in rollback_records:
            rollbacks.append({
                "timestamp": record.timestamp.isoformat(),
                "event_type": record.event_type,
                "success": record.success,
                "description": record.description,
            })
        
        return {
            "report_type": "rollback_analysis",
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_rollbacks": len(rollbacks),
            "successful_rollbacks": len([r for r in rollbacks if r["success"]]),
            "failed_rollbacks": len([r for r in rollbacks if not r["success"]]),
            "rollbacks": rollbacks,
        }
    
    def _generate_audit_summary_report(self, start: datetime, 
                                       end: datetime) -> Dict[str, Any]:
        """Generate audit summary report"""
        records = self.storage.read_records(start, end)
        
        # Count events by type
        event_counts: Dict[str, int] = {}
        actor_counts: Dict[str, int] = {}
        
        for record in records:
            event_counts[record.event_type] = event_counts.get(record.event_type, 0) + 1
            actor_counts[record.actor_type] = actor_counts.get(record.actor_type, 0) + 1
        
        return {
            "report_type": "audit_summary",
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_records": len(records),
            "event_breakdown": event_counts,
            "actor_breakdown": actor_counts,
        }
    
    def export_for_audit(self, auditor: str, 
                         start: datetime,
                         end: datetime) -> Dict[str, Any]:
        """Export data for external audit"""
        records = self.storage.read_records(start, end)
        
        return {
            "auditor": auditor,
            "export_timestamp": datetime.now().isoformat(),
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "record_count": len(records),
            "records": [r.to_dict() for r in records],
            "integrity_verification": {
                "all_records_valid": all(r.verify_integrity() for r in records),
                "verification_method": "SHA-256 checksum",
            },
        }


class UpdateAuditLogger:
    """
    Comprehensive logging for all update operations.
    Maintains immutable audit trail.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage = LogStorageManager(config)
        self.reporter = ComplianceReporter(self.storage)
        
        self._lock = threading.RLock()
        self._last_record_hash = ""
        self._record_count = 0
        
        # Initialize from existing logs
        self._load_last_record_hash()
    
    def _load_last_record_hash(self):
        """Load hash of last record for chain integrity"""
        try:
            # Find most recent log file
            log_files = sorted(
                self.storage.local_path.glob("audit_*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if log_files:
                with open(log_files[0]) as f:
                    lines = f.readlines()
                    if lines:
                        last_record = json.loads(lines[-1].strip())
                        self._last_record_hash = last_record.get("checksum", "")
        except Exception as e:
            logger.warning(f"Failed to load last record hash: {e}")
    
    def log_event(self, event_type: str, actor_type: str, actor_id: str,
                  description: str, details: Dict[str, Any] = None,
                  update_id: str = None, version_from: str = None,
                  version_to: str = None, success: bool = True,
                  error_message: str = None) -> AuditRecord:
        """
        Log an update event with full context.
        
        Args:
            event_type: Type of event
            actor_type: Type of actor (system, user, agent)
            actor_id: Identifier of actor
            description: Human-readable description
            details: Structured event details
            update_id: Associated update ID
            version_from: Source version
            version_to: Target version
            success: Whether operation succeeded
            error_message: Error message if failed
            
        Returns:
            Created audit record
        """
        with self._lock:
            record = AuditRecord(
                record_id="",
                timestamp=datetime.now(),
                event_type=event_type,
                actor_type=actor_type,
                actor_id=actor_id,
                description=description,
                details=details or {},
                update_id=update_id,
                version_from=version_from,
                version_to=version_to,
                success=success,
                error_message=error_message,
                previous_record_hash=self._last_record_hash,
            )
            
            # Store record
            self.storage.store_record(record)
            
            # Update chain
            self._last_record_hash = record.checksum
            self._record_count += 1
            
            # Log to standard logger
            log_method = logger.info if success else logger.error
            log_method(f"Audit: {event_type} - {description}")
            
            return record
    
    def log_update_result(self, result: Any):
        """Log update result"""
        self.log_event(
            event_type=EventType.UPDATE_COMPLETED.value if result.success else EventType.UPDATE_FAILED.value,
            actor_type="system",
            actor_id="self_updating_loop",
            description=result.message if hasattr(result, 'message') else str(result),
            update_id=result.update_id if hasattr(result, 'update_id') else None,
            success=result.success if hasattr(result, 'success') else False,
            error_message=result.error if hasattr(result, 'error') else None,
        )
    
    def get_audit_trail(self, start: datetime, 
                        end: datetime) -> List[AuditRecord]:
        """Retrieve audit trail for time period"""
        return self.storage.read_records(start, end)
    
    def generate_report(self, report_type: str, 
                        start: datetime = None,
                        end: datetime = None) -> Dict[str, Any]:
        """Generate compliance report"""
        if start is None:
            start = datetime.now() - timedelta(days=30)
        if end is None:
            end = datetime.now()
        
        return self.reporter.generate_report(report_type, start, end)
    
    def export_for_audit(self, auditor: str,
                         start: datetime = None,
                         end: datetime = None) -> Dict[str, Any]:
        """Export data for external audit"""
        if start is None:
            start = datetime.now() - timedelta(days=90)
        if end is None:
            end = datetime.now()
        
        return self.reporter.export_for_audit(auditor, start, end)
    
    def verify_chain_integrity(self) -> bool:
        """Verify integrity of the entire audit chain"""
        records = self.storage.read_records(
            datetime.now() - timedelta(days=365),
            datetime.now()
        )
        
        for record in records:
            if not record.verify_integrity():
                logger.error(f"Record integrity check failed: {record.record_id}")
                return False
        
        logger.info(f"Audit chain integrity verified: {len(records)} records")
        return True
    
    def cleanup(self):
        """Perform cleanup of old logs"""
        self.storage.archive_old_logs()
        self.storage.cleanup_old_records()


if __name__ == "__main__":
    # Example usage
    config = {
        "log_path": "./logs/updates/",
        "retention_days": 90,
    }
    
    audit_logger = UpdateAuditLogger(config)
    
    # Log some events
    audit_logger.log_event(
        event_type=EventType.LOOP_STARTED.value,
        actor_type="system",
        actor_id="self_updating_loop",
        description="Self-updating loop started",
    )
    
    audit_logger.log_event(
        event_type=EventType.UPDATE_COMPLETED.value,
        actor_type="system",
        actor_id="self_updating_loop",
        description="Update applied successfully",
        update_id="upd_123",
        version_from="1.0.0",
        version_to="1.0.1",
    )
    
    # Generate report
    report = audit_logger.generate_report("audit_summary")
    print(f"Audit summary: {report}")
    
    # Verify integrity
    integrity_ok = audit_logger.verify_chain_integrity()
    print(f"Chain integrity: {integrity_ok}")
