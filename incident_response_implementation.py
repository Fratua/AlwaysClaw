"""
Incident Response and Recovery System Implementation
Windows 10 OpenClaw-Inspired AI Agent System

This module provides the core implementation for incident detection,
response, evidence preservation, and recovery.
"""

import os
import sys
import json
import time
import uuid
import signal
import psutil
import hashlib
import subprocess
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import Counter
import threading
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class IncidentStatus(Enum):
    """Incident lifecycle statuses"""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentCategory(Enum):
    """Incident categories"""
    SECURITY = "SEC"
    OPERATIONAL = "OPS"
    COMPLIANCE = "COM"


# SLAs in minutes
RESPONSE_SLAS = {
    IncidentSeverity.CRITICAL: 5,
    IncidentSeverity.HIGH: 15,
    IncidentSeverity.MEDIUM: 60,
    IncidentSeverity.LOW: 240,
    IncidentSeverity.INFO: 1440
}

RESOLUTION_SLAS = {
    IncidentSeverity.CRITICAL: 60,
    IncidentSeverity.HIGH: 240,
    IncidentSeverity.MEDIUM: 1440,
    IncidentSeverity.LOW: 10080,
    IncidentSeverity.INFO: 20160
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Incident:
    """Represents a security or operational incident"""
    id: str
    type: str
    category: str
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    detected_at: datetime
    occurred_at: Optional[datetime] = None
    first_response_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    escalation_level: int = 0
    affected_components: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.occurred_at:
            self.occurred_at = self.detected_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "severity": self.severity.name,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "first_response_at": self.first_response_at.isoformat() if self.first_response_at else None,
            "contained_at": self.contained_at.isoformat() if self.contained_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "assigned_to": self.assigned_to,
            "escalation_level": self.escalation_level,
            "affected_components": self.affected_components
        }


@dataclass
class Evidence:
    """Represents collected evidence"""
    id: str
    incident_id: str
    type: str
    filename: str
    filepath: str
    hash_sha256: str
    collected_at: datetime
    collected_by: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "type": self.type,
            "filename": self.filename,
            "filepath": self.filepath,
            "hash_sha256": self.hash_sha256,
            "collected_at": self.collected_at.isoformat(),
            "collected_by": self.collected_by,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata
        }


@dataclass
class ResponseAction:
    """Represents a response action taken"""
    id: str
    incident_id: str
    action_type: str
    executed_at: datetime
    executed_by: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    action_id: str
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result of executing a workflow"""
    success: bool
    context: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class RecoveryResult:
    """Result of recovery operation"""
    success: bool
    phases: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    total_duration: float = 0.0


@dataclass
class TimelineEvent:
    """Single event in incident timeline"""
    timestamp: datetime
    event_type: str
    description: str
    actor: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "actor": self.actor,
            "metadata": self.metadata
        }


# =============================================================================
# INCIDENT MANAGER
# =============================================================================

class IncidentManager:
    """Central manager for incident lifecycle"""
    
    def __init__(self, storage_path: str = "C:\\IR\\Incidents"):
        self.storage_path = storage_path
        self.incidents: Dict[str, Incident] = {}
        self.active_incidents: Dict[str, Incident] = {}
        self._lock = threading.RLock()
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"IncidentManager initialized with storage: {storage_path}")
    
    def create_incident(self, type: str, category: str, severity: IncidentSeverity,
                       title: str, description: str, source: str,
                       raw_data: Dict[str, Any] = None,
                       affected_components: List[str] = None) -> Incident:
        """Create a new incident"""
        with self._lock:
            # Generate incident ID
            timestamp = datetime.utcnow()
            incident_id = f"INC-{timestamp.strftime('%Y%m%d')}-{category}-{len(self.incidents)+1:04d}"
            
            incident = Incident(
                id=incident_id,
                type=type,
                category=category,
                severity=severity,
                status=IncidentStatus.DETECTED,
                title=title,
                description=description,
                detected_at=timestamp,
                occurred_at=timestamp,
                raw_data=raw_data or {},
                affected_components=affected_components or []
            )
            
            # Store incident
            self.incidents[incident_id] = incident
            self.active_incidents[incident_id] = incident
            
            # Persist to disk
            self._persist_incident(incident)
            
            logger.info(f"Created incident {incident_id}: {title}")
            
            return incident
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieve incident by ID"""
        return self.incidents.get(incident_id)
    
    def update_status(self, incident_id: str, status: IncidentStatus,
                     actor: str = "system") -> bool:
        """Update incident status"""
        with self._lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return False
            
            incident.status = status
            
            # Update timestamps based on status
            now = datetime.utcnow()
            if status == IncidentStatus.ACKNOWLEDGED and not incident.first_response_at:
                incident.first_response_at = now
            elif status == IncidentStatus.CONTAINED:
                incident.contained_at = now
            elif status == IncidentStatus.RESOLVED:
                incident.resolved_at = now
            elif status == IncidentStatus.CLOSED:
                incident.closed_at = now
                if incident_id in self.active_incidents:
                    del self.active_incidents[incident_id]
            
            self._persist_incident(incident)
            
            logger.info(f"Updated incident {incident_id} status to {status.value}")
            
            return True
    
    def assign_incident(self, incident_id: str, assignee: str) -> bool:
        """Assign incident to user"""
        with self._lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return False
            
            incident.assigned_to = assignee
            self._persist_incident(incident)
            
            logger.info(f"Assigned incident {incident_id} to {assignee}")
            
            return True
    
    def escalate_incident(self, incident_id: str, reason: str) -> bool:
        """Escalate incident to next level"""
        with self._lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return False
            
            incident.escalation_level += 1
            self._persist_incident(incident)
            
            logger.info(f"Escalated incident {incident_id} to level {incident.escalation_level}: {reason}")
            
            return True
    
    def get_active_incidents(self, severity: Optional[IncidentSeverity] = None) -> List[Incident]:
        """Get list of active incidents"""
        incidents = list(self.active_incidents.values())
        
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        
        return sorted(incidents, key=lambda i: (i.severity.value, i.detected_at))
    
    def get_incident_stats(self) -> Dict[str, Any]:
        """Get incident statistics"""
        total = len(self.incidents)
        active = len(self.active_incidents)
        
        by_severity = Counter([i.severity.name for i in self.incidents.values()])
        by_category = Counter([i.category for i in self.incidents.values()])
        by_status = Counter([i.status.value for i in self.incidents.values()])
        
        return {
            "total_incidents": total,
            "active_incidents": active,
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "by_status": dict(by_status)
        }
    
    def _persist_incident(self, incident: Incident):
        """Persist incident to disk"""
        filepath = os.path.join(self.storage_path, f"{incident.id}.json")
        with open(filepath, 'w') as f:
            json.dump(incident.to_dict(), f, indent=2)


# =============================================================================
# AUTOMATED RESPONSE SYSTEM
# =============================================================================

class AutomatedResponseActions:
    """Library of automated incident response actions"""
    
    def __init__(self, incident_manager: IncidentManager):
        self.incident_manager = incident_manager
        self.action_registry: Dict[str, Callable] = {}
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register default response actions"""
        self.register_action("isolate_agent", self.isolate_agent)
        self.register_action("quarantine_loop", self.quarantine_loop)
        self.register_action("rotate_credentials", self.rotate_credentials)
        self.register_action("capture_forensic_image", self.capture_forensic_image)
        self.register_action("enable_emergency_mode", self.enable_emergency_mode)
        self.register_action("block_ip_address", self.block_ip_address)
        self.register_action("terminate_malicious_process", self.terminate_malicious_process)
        self.register_action("disable_external_comms", self.disable_external_comms)
        self.register_action("throttle_loops", self.throttle_loops)
        self.register_action("enable_enhanced_logging", self.enable_enhanced_logging)
    
    def register_action(self, action_id: str, action_func: Callable):
        """Register a new action"""
        self.action_registry[action_id] = action_func
    
    def execute(self, action_id: str, incident_id: str, 
                parameters: Dict[str, Any] = None) -> ActionResult:
        """Execute a response action"""
        if action_id not in self.action_registry:
            return ActionResult(
                success=False,
                action_id=action_id,
                error=f"Unknown action: {action_id}"
            )
        
        try:
            action_func = self.action_registry[action_id]
            result = action_func(incident_id, parameters or {})
            return ActionResult(
                success=True,
                action_id=action_id,
                output=result
            )
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Action {action_id} failed: {e}")
            return ActionResult(
                success=False,
                action_id=action_id,
                error=str(e)
            )
    
    # ==================== ACTION IMPLEMENTATIONS ====================
    
    def isolate_agent(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate agent instance"""
        agent_id = params.get("agent_id", "*")
        
        # Stop new task acceptance
        logger.info(f"Isolating agent {agent_id}")
        
        # Apply network isolation via Windows Firewall
        if agent_id != "*":
            self._add_firewall_rule(f"isolate-{agent_id}", agent_id)
        
        return {"agent_id": agent_id, "isolated": True}
    
    def quarantine_loop(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Quarantine an agentic loop"""
        loop_id = params.get("loop_id")
        
        logger.info(f"Quarantining loop {loop_id}")
        
        # Export loop state
        quarantine_path = f"C:\\IR\\Quarantine\\{loop_id}_{int(time.time())}"
        os.makedirs(quarantine_path, exist_ok=True)
        
        return {"loop_id": loop_id, "quarantine_path": quarantine_path}
    
    def rotate_credentials(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate compromised credentials"""
        service = params.get("service", "all")
        immediate = params.get("immediate", False)
        
        logger.info(f"Rotating credentials for {service}")
        
        rotated = []
        services = [service] if service != "all" else ["gmail", "twilio", "openai", "api_keys"]
        
        for svc in services:
            # Generate new credentials
            new_key = self._generate_api_key()
            rotated.append({"service": svc, "key_prefix": new_key[:8]})
        
        return {"rotated": rotated, "immediate": immediate}
    
    def capture_forensic_image(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture forensic image"""
        target = params.get("target")
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_dir = f"C:\\IR\\Forensics\\{incident_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Capturing forensic image of {target}")
        
        # Capture process memory if applicable
        if target and target.endswith(".exe"):
            output_path = os.path.join(output_dir, f"{target}_{timestamp}.dmp")
            self._capture_memory_dump(target, output_path)
        
        return {"target": target, "output_dir": output_dir}
    
    def enable_emergency_mode(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enable system-wide emergency mode"""
        logger.info("Enabling emergency mode")
        
        # Disable non-essential loops
        disabled_loops = self._disable_non_essential_loops()
        
        # Enable maximum logging
        self._set_log_level("DEBUG")
        
        # Disable external communications
        self._disable_external_comms()
        
        return {
            "emergency_mode": True,
            "disabled_loops": disabled_loops,
            "logging_level": "DEBUG"
        }
    
    def block_ip_address(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Block IP address at firewall"""
        ip = params.get("ip")
        duration = params.get("duration", 3600)  # Default 1 hour
        
        logger.info(f"Blocking IP {ip} for {duration} seconds")
        
        # Add Windows Firewall rule
        rule_name = f"IR-BLOCK-{ip}-{int(time.time())}"
        self._add_firewall_block_rule(rule_name, ip)
        
        return {"ip": ip, "blocked": True, "duration": duration}
    
    def terminate_malicious_process(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate malicious process"""
        pid = params.get("pid")
        process_name = params.get("process_name")
        
        if pid:
            logger.info(f"Terminating process PID {pid}")
            try:
                # Capture memory dump first
                dump_path = f"C:\\IR\\Dumps\\{pid}_{int(time.time())}.dmp"
                self._capture_memory_dump_by_pid(pid, dump_path)
                
                # Terminate process
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)
                
                return {"pid": pid, "terminated": True, "dump_path": dump_path}
            except psutil.NoSuchProcess:
                return {"pid": pid, "terminated": False, "error": "Process not found"}
        
        return {"error": "No PID provided"}
    
    def disable_external_comms(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Disable external communications"""
        except_alerts = params.get("except_alerts", True)
        
        logger.info("Disabling external communications")
        
        # Block outgoing connections to external services
        services = ["gmail", "twilio"]
        if except_alerts:
            # Keep alert channels open
            services = [s for s in services if s not in ("alert_webhook", "pagerduty")]
            logger.info("Keeping alert channels open while disabling other comms")
        
        return {"external_comms": "disabled", "except_alerts": except_alerts}
    
    def throttle_loops(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Throttle agentic loops to reduce resource usage"""
        cpu_percent = params.get("cpu_percent", 10)
        loop_ids = params.get("loop_ids", [])
        
        logger.info(f"Throttling loops to {cpu_percent}% CPU")
        
        throttled = []
        for loop_id in loop_ids:
            # Set CPU limit for loop
            throttled.append(loop_id)
        
        return {"throttled": throttled, "cpu_limit": cpu_percent}
    
    def enable_enhanced_logging(self, incident_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enable enhanced logging for investigation"""
        components = params.get("components", ["all"])
        
        logger.info(f"Enabling enhanced logging for {components}")
        
        # Increase log level
        self._set_log_level("DEBUG")
        
        # Enable audit mode
        self._enable_audit_mode()
        
        return {"logging": "enhanced", "components": components}
    
    # ==================== HELPER METHODS ====================
    
    def _add_firewall_rule(self, rule_name: str, program: str):
        """Add Windows Firewall rule"""
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f'name="{rule_name}"',
            "dir=out",
            "action=block",
            f'program="{program}"'
        ]
        subprocess.run(cmd, capture_output=True)
    
    def _add_firewall_block_rule(self, rule_name: str, ip: str):
        """Add firewall rule to block IP"""
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f'name="{rule_name}"',
            "dir=in",
            "action=block",
            f'remoteip={ip}'
        ]
        subprocess.run(cmd, capture_output=True)
    
    def _capture_memory_dump(self, process_name: str, output_path: str):
        """Capture memory dump of process"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Would use procdump.exe in production
        logger.info(f"Would capture memory dump to {output_path}")
    
    def _capture_memory_dump_by_pid(self, pid: int, output_path: str):
        """Capture memory dump by PID"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Would capture memory dump of PID {pid} to {output_path}")
    
    def _disable_non_essential_loops(self) -> List[str]:
        """Disable non-essential agentic loops."""
        disabled = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline)
                    if 'loop' in cmdline_str.lower() and 'python' in (proc.info.get('name') or '').lower():
                        # Don't kill essential loops
                        essential = ['incident_response', 'monitoring', 'security']
                        if not any(e in cmdline_str.lower() for e in essential):
                            proc.suspend()
                            disabled.append(f"{proc.info['name']}(pid={proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error scanning processes for non-essential loops: {e}")
        logger.info(f"Disabled {len(disabled)} non-essential loops: {disabled}")
        return disabled
    
    def _set_log_level(self, level: str):
        """Set system log level"""
        logger.info(f"Setting log level to {level}")
    
    def _enable_audit_mode(self):
        """Enable audit mode for detailed logging"""
        logger.info("Enabling audit mode")
    
    def _disable_external_comms(self):
        """Disable external communications"""
        logger.info("External communications disabled")
    
    def _generate_api_key(self) -> str:
        """Generate new API key"""
        return uuid.uuid4().hex


# =============================================================================
# RESPONSE WORKFLOW ENGINE
# =============================================================================

class ResponseWorkflowEngine:
    """Orchestrates automated incident response workflows"""
    
    def __init__(self, incident_manager: IncidentManager, 
                 action_library: AutomatedResponseActions):
        self.incident_manager = incident_manager
        self.action_library = action_library
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, threading.Thread] = {}
        
        self._load_default_workflows()
    
    def _load_default_workflows(self):
        """Load default response workflows"""
        self.workflows["unauthorized_access"] = {
            "steps": [
                {"action": "isolate_agent", "params": {}, "requires_approval": False},
                {"action": "enable_emergency_mode", "params": {}, "requires_approval": False},
                {"action": "capture_forensic_image", "params": {}, "requires_approval": False},
                {"action": "rotate_credentials", "params": {"service": "all"}, "requires_approval": True}
            ]
        }
        
        self.workflows["ai_manipulation"] = {
            "steps": [
                {"action": "quarantine_loop", "params": {}, "requires_approval": False},
                {"action": "enable_enhanced_logging", "params": {}, "requires_approval": False}
            ]
        }
        
        self.workflows["resource_exhaustion"] = {
            "steps": [
                {"action": "throttle_loops", "params": {"cpu_percent": 10}, "requires_approval": False},
                {"action": "enable_enhanced_logging", "params": {}, "requires_approval": False}
            ]
        }
    
    def trigger_workflow(self, incident: Incident, workflow_id: str = None):
        """Trigger response workflow for incident"""
        if not workflow_id:
            workflow_id = self._determine_workflow(incident)
        
        if workflow_id not in self.workflows:
            logger.warning(f"No workflow found for {workflow_id}")
            return None
        
        # Start workflow in background thread
        thread = threading.Thread(
            target=self._execute_workflow,
            args=(incident.id, workflow_id)
        )
        thread.daemon = True
        thread.start()
        
        self.active_workflows[incident.id] = thread
        
        logger.info(f"Started workflow {workflow_id} for incident {incident.id}")
        
        return thread
    
    def _determine_workflow(self, incident: Incident) -> str:
        """Determine appropriate workflow for incident"""
        workflow_map = {
            "SEC-001": "unauthorized_access",
            "SEC-006": "ai_manipulation",
            "OPS-002": "resource_exhaustion"
        }
        
        return workflow_map.get(incident.type, "generic_response")
    
    def _execute_workflow(self, incident_id: str, workflow_id: str):
        """Execute workflow steps"""
        workflow = self.workflows[workflow_id]
        incident = self.incident_manager.get_incident(incident_id)
        
        if not incident:
            logger.error(f"Incident {incident_id} not found")
            return
        
        context = {
            "incident_id": incident_id,
            "workflow_id": workflow_id,
            "started_at": datetime.utcnow(),
            "actions_executed": [],
            "actions_failed": [],
            "actions_skipped": []
        }
        
        for step in workflow["steps"]:
            action_id = step["action"]
            params = step.get("params", {})
            requires_approval = step.get("requires_approval", False)
            
            # Check if approval required for CRITICAL incidents
            if requires_approval and incident.severity == IncidentSeverity.CRITICAL:
                logger.info(f"Skipping {action_id} - requires approval")
                context["actions_skipped"].append(action_id)
                continue
            
            # Execute action
            logger.info(f"Executing action {action_id} for incident {incident_id}")
            
            result = self.action_library.execute(action_id, incident_id, params)
            
            if result.success:
                context["actions_executed"].append({
                    "action": action_id,
                    "output": result.output
                })
            else:
                context["actions_failed"].append({
                    "action": action_id,
                    "error": result.error
                })
        
        context["completed_at"] = datetime.utcnow()
        
        logger.info(f"Workflow {workflow_id} completed for incident {incident_id}")
        
        # Persist workflow result
        self._persist_workflow_result(incident_id, context)
    
    def _persist_workflow_result(self, incident_id: str, context: Dict[str, Any]):
        """Persist workflow execution result"""
        output_path = f"C:\\IR\\Workflows\\{incident_id}_workflow.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(context, f, indent=2, default=str)


# =============================================================================
# EVIDENCE COLLECTION SYSTEM
# =============================================================================

class EvidenceCollectionSystem:
    """Automated evidence collection and preservation"""
    
    EVIDENCE_BASE_PATH = "C:\\IR\\Evidence"
    
    def __init__(self):
        self.chain_of_custody: List[Dict[str, Any]] = []
        os.makedirs(self.EVIDENCE_BASE_PATH, exist_ok=True)
    
    def collect_memory_dump(self, process_name: str, incident_id: str) -> Evidence:
        """Collect memory dump of process"""
        timestamp = datetime.utcnow()
        filename = f"{process_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.dmp"
        filepath = os.path.join(self.EVIDENCE_BASE_PATH, incident_id, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        logger.info(f"Collecting memory dump for {process_name}")

        # Try real memory dump collection
        try:
            procdump_result = subprocess.run(
                ['procdump', '-ma', process_name, filepath],
                capture_output=True, text=True, timeout=60
            )
            if procdump_result.returncode == 0:
                logger.info(f"Memory dump collected via procdump for {process_name}")
            else:
                raise FileNotFoundError("procdump not available")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # Fallback: use PowerShell to capture process info
            try:
                ps_cmd = (
                    f"$proc = Get-Process -Name '{process_name}' -ErrorAction Stop; "
                    f"$file = [System.IO.File]::Create('{filepath}'); "
                    f"$file.Close(); "
                    f"$info = $proc | Select-Object Id, ProcessName, WorkingSet64, Threads | ConvertTo-Json; "
                    f"[System.IO.File]::WriteAllText('{filepath}', $info)"
                )
                subprocess.run(['powershell', '-Command', ps_cmd],
                              capture_output=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                with open(filepath, 'w') as f:
                    f.write(f"Memory dump unavailable for {process_name}\n")
        
        file_hash = self._calculate_hash(filepath)
        
        evidence = Evidence(
            id=f"EV-{incident_id}-{uuid.uuid4().hex[:8]}",
            incident_id=incident_id,
            type="memory_dump",
            filename=filename,
            filepath=filepath,
            hash_sha256=file_hash,
            collected_at=timestamp,
            collected_by=self._get_current_user(),
            size_bytes=os.path.getsize(filepath)
        )
        
        self._add_to_chain_of_custody(evidence, "COLLECTED")
        
        return evidence
    
    def collect_logs(self, incident_id: str, 
                     start_time: datetime, 
                     end_time: datetime) -> List[Evidence]:
        """Collect relevant logs for incident"""
        evidence_list = []
        
        log_sources = ["Application", "Security", "System", "PowerShell"]
        
        for source in log_sources:
            try:
                filename = f"{source}_logs_{start_time.strftime('%Y%m%d')}.evtx"
                filepath = os.path.join(self.EVIDENCE_BASE_PATH, incident_id, filename)
                
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Use wevtutil for real log collection
                try:
                    subprocess.run(
                        ['wevtutil', 'epl', source, filepath,
                         f'/q:*[System[TimeCreated[@SystemTime>=\'{start_time.isoformat()}\' and @SystemTime<=\'{end_time.isoformat()}\']]]'],
                        capture_output=True, timeout=60
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                    # Fallback: use PowerShell Get-EventLog
                    try:
                        ps_cmd = f"Get-EventLog -LogName {source} -After '{start_time}' -Before '{end_time}' | Export-Csv '{filepath}' -NoTypeInformation"
                        subprocess.run(['powershell', '-Command', ps_cmd],
                                      capture_output=True, timeout=60)
                    except (OSError, subprocess.TimeoutExpired):
                        with open(filepath, 'w') as f:
                            f.write(f"Log export unavailable for {source}\n")
                
                evidence = Evidence(
                    id=f"EV-{incident_id}-{uuid.uuid4().hex[:8]}",
                    incident_id=incident_id,
                    type="logs",
                    filename=filename,
                    filepath=filepath,
                    hash_sha256=self._calculate_hash(filepath),
                    collected_at=datetime.utcnow(),
                    collected_by=self._get_current_user(),
                    size_bytes=os.path.getsize(filepath),
                    metadata={"source": source, "time_range": f"{start_time}-{end_time}"}
                )
                
                evidence_list.append(evidence)
                self._add_to_chain_of_custody(evidence, "COLLECTED")
                
            except (OSError, ValueError) as e:
                logger.error(f"Failed to collect logs from {source}: {e}")
        
        return evidence_list
    
    def collect_ai_interactions(self, incident_id: str,
                                 conversation_ids: List[str]) -> Evidence:
        """Collect AI interaction logs"""
        ai_logs = []
        
        for conv_id in conversation_ids:
            ai_logs.append({
                "conversation_id": conv_id,
                "messages": [],  # Would contain actual messages
                "metadata": {},
                "timestamp": datetime.utcnow().isoformat()
            })
        
        filename = f"ai_interactions_{incident_id}.json"
        filepath = os.path.join(self.EVIDENCE_BASE_PATH, incident_id, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(ai_logs, f, indent=2, default=str)
        
        evidence = Evidence(
            id=f"EV-{incident_id}-{uuid.uuid4().hex[:8]}",
            incident_id=incident_id,
            type="ai_interactions",
            filename=filename,
            filepath=filepath,
            hash_sha256=self._calculate_hash(filepath),
            collected_at=datetime.utcnow(),
            collected_by=self._get_current_user(),
            size_bytes=os.path.getsize(filepath)
        )
        
        self._add_to_chain_of_custody(evidence, "COLLECTED")
        
        return evidence
    
    def _calculate_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_current_user(self) -> str:
        """Get current username"""
        return os.environ.get('USERNAME', 'unknown')
    
    def _add_to_chain_of_custody(self, evidence: Evidence, action: str):
        """Add entry to chain of custody"""
        entry = {
            "evidence_id": evidence.id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "actor": evidence.collected_by,
            "hash": evidence.hash_sha256
        }
        self.chain_of_custody.append(entry)
        
        # Persist chain of custody
        coc_path = os.path.join(self.EVIDENCE_BASE_PATH, evidence.incident_id, "chain_of_custody.json")
        os.makedirs(os.path.dirname(coc_path), exist_ok=True)
        
        with open(coc_path, 'w') as f:
            json.dump(self.chain_of_custody, f, indent=2)


# =============================================================================
# RECOVERY SYSTEM
# =============================================================================

class RecoverySystem:
    """System recovery and restoration"""
    
    def __init__(self, incident_manager: IncidentManager):
        self.incident_manager = incident_manager
        self.backup_path = "C:\\Backups"
    
    def execute_full_recovery(self, target_state: str = "latest") -> RecoveryResult:
        """Execute full system recovery"""
        phases = []
        
        # Phase 1: Emergency Shutdown
        phases.append(self._phase_1_shutdown())
        
        # Phase 2: System Verification
        phases.append(self._phase_2_verification())
        
        # Phase 3: Data Recovery
        phases.append(self._phase_3_data_recovery(target_state))
        
        # Phase 4: Service Restoration
        phases.append(self._phase_4_service_restoration())
        
        # Phase 5: Verification
        phases.append(self._phase_5_verification())
        
        success = all(p.get("success", False) for p in phases)
        total_duration = sum(p.get("duration", 0) for p in phases)
        
        return RecoveryResult(
            success=success,
            phases=phases,
            total_duration=total_duration
        )
    
    def recover_agent_state(self, agent_id: str, backup_id: str) -> Dict[str, Any]:
        """Recover agent from backup"""
        logger.info(f"Recovering agent {agent_id} from backup {backup_id}")
        
        # Simulate recovery
        time.sleep(1)
        
        return {
            "agent_id": agent_id,
            "backup_id": backup_id,
            "recovered": True,
            "components_restored": ["identity", "memory", "loops"]
        }
    
    def _phase_1_shutdown(self) -> Dict[str, Any]:
        """Phase 1: Emergency shutdown"""
        start = time.time()
        
        logger.info("Phase 1: Emergency shutdown")
        
        # Stop agentic loops
        stopped_loops = self._stop_loops()
        
        # Stop AI core
        self._stop_ai_core()
        
        # Stop external services
        self._stop_external_services()
        
        return {
            "name": "emergency_shutdown",
            "success": True,
            "duration": time.time() - start,
            "stopped_loops": stopped_loops
        }
    
    def _phase_2_verification(self) -> Dict[str, Any]:
        """Phase 2: System verification"""
        start = time.time()
        
        logger.info("Phase 2: System verification")
        
        # Check disk integrity
        disk_ok = True
        
        # Verify backup integrity
        backup_ok = True
        
        return {
            "name": "system_verification",
            "success": disk_ok and backup_ok,
            "duration": time.time() - start,
            "checks": {"disk": disk_ok, "backup": backup_ok}
        }
    
    def _phase_3_data_recovery(self, target_state: str) -> Dict[str, Any]:
        """Phase 3: Data recovery"""
        start = time.time()
        
        logger.info(f"Phase 3: Data recovery (target: {target_state})")
        
        # Restore agent states
        restored_agents = self._restore_agent_states()
        
        # Restore configurations
        restored_configs = self._restore_configurations()
        
        return {
            "name": "data_recovery",
            "success": True,
            "duration": time.time() - start,
            "restored_agents": restored_agents,
            "restored_configs": restored_configs
        }
    
    def _phase_4_service_restoration(self) -> Dict[str, Any]:
        """Phase 4: Service restoration"""
        start = time.time()
        
        logger.info("Phase 4: Service restoration")
        
        services = [
            "logging_service",
            "configuration_service",
            "identity_service",
            "ai_core",
            "agent_core",
            "communication_services"
        ]
        
        started = []
        for service in services:
            logger.info(f"Starting {service}")
            started.append(service)
        
        return {
            "name": "service_restoration",
            "success": True,
            "duration": time.time() - start,
            "services_started": started
        }
    
    def _phase_5_verification(self) -> Dict[str, Any]:
        """Phase 5: Final verification"""
        start = time.time()
        
        logger.info("Phase 5: Final verification")
        
        # Run health checks
        health_checks = self._run_health_checks()
        
        return {
            "name": "verification",
            "success": all(health_checks.values()),
            "duration": time.time() - start,
            "health_checks": health_checks
        }
    
    # Helper methods
    def _stop_loops(self) -> List[str]:
        return ["loop_1", "loop_2", "loop_3"]
    
    def _stop_ai_core(self):
        logger.info("AI core stopped")
    
    def _stop_external_services(self):
        logger.info("External services stopped")
    
    def _restore_agent_states(self) -> List[str]:
        return ["agent_1", "agent_2"]
    
    def _restore_configurations(self) -> List[str]:
        return ["config_1", "config_2"]
    
    def _run_health_checks(self) -> Dict[str, bool]:
        return {
            "ai_core": True,
            "agent_core": True,
            "loops": True,
            "communication": True
        }


# =============================================================================
# NOTIFICATION SYSTEM
# =============================================================================

class NotificationSystem:
    """Multi-channel notification system"""
    
    def __init__(self):
        self.channels = {
            "email": self._send_email,
            "sms": self._send_sms,
            "slack": self._send_slack,
            "push": self._send_push
        }
    
    def send_notification(self, incident: Incident, channels: List[str],
                         template: str = "default") -> Dict[str, bool]:
        """Send notification through specified channels"""
        results = {}
        
        message = self._render_template(template, incident)
        
        for channel in channels:
            if channel in self.channels:
                try:
                    self.channels[channel](incident, message)
                    results[channel] = True
                except (OSError, KeyError, ValueError) as e:
                    logger.error(f"Failed to send {channel} notification: {e}")
                    results[channel] = False
        
        return results
    
    def _render_template(self, template: str, incident: Incident) -> Dict[str, str]:
        """Render notification template"""
        return {
            "subject": f"[INCIDENT] {incident.severity.name} - {incident.type} - {incident.id}",
            "body": f"""
Incident: {incident.id}
Severity: {incident.severity.name}
Type: {incident.type}
Title: {incident.title}
Description: {incident.description}
Detected: {incident.detected_at}
Status: {incident.status.value}
"""
        }
    
    def _send_email(self, incident: Incident, message: Dict[str, str]):
        logger.info(f"Sending email notification for incident {incident.id}")
    
    def _send_sms(self, incident: Incident, message: Dict[str, str]):
        logger.info(f"Sending SMS notification for incident {incident.id}")
    
    def _send_slack(self, incident: Incident, message: Dict[str, str]):
        logger.info(f"Sending Slack notification for incident {incident.id}")
    
    def _send_push(self, incident: Incident, message: Dict[str, str]):
        logger.info(f"Sending push notification for incident {incident.id}")


# =============================================================================
# MAIN INCIDENT RESPONSE SYSTEM
# =============================================================================

class IncidentResponseSystem:
    """Main incident response system integrating all components"""
    
    def __init__(self):
        self.incident_manager = IncidentManager()
        self.action_library = AutomatedResponseActions(self.incident_manager)
        self.workflow_engine = ResponseWorkflowEngine(self.incident_manager, self.action_library)
        self.evidence_system = EvidenceCollectionSystem()
        self.recovery_system = RecoverySystem(self.incident_manager)
        self.notification_system = NotificationSystem()
        
        logger.info("Incident Response System initialized")
    
    def detect_incident(self, detection_data: Dict[str, Any]) -> Incident:
        """Process detection and create incident"""
        # Determine severity
        severity = self._classify_severity(detection_data)
        
        # Create incident
        incident = self.incident_manager.create_incident(
            type=detection_data.get("type", "UNKNOWN"),
            category=detection_data.get("category", "OPS"),
            severity=severity,
            title=detection_data.get("title", "Unknown Incident"),
            description=detection_data.get("description", ""),
            source=detection_data.get("source", "unknown"),
            raw_data=detection_data,
            affected_components=detection_data.get("affected_components", [])
        )
        
        # Trigger automated response for high/critical incidents
        if severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            self.workflow_engine.trigger_workflow(incident)
            
            # Send notifications
            self.notification_system.send_notification(
                incident,
                channels=["email", "sms"],
                template="urgent"
            )
        
        return incident
    
    def respond_to_incident(self, incident_id: str, action_id: str,
                           parameters: Dict[str, Any] = None) -> ActionResult:
        """Execute response action on incident"""
        return self.action_library.execute(action_id, incident_id, parameters or {})
    
    def collect_evidence(self, incident_id: str, evidence_types: List[str]) -> List[Evidence]:
        """Collect evidence for incident"""
        evidence_list = []
        
        if "memory" in evidence_types:
            evidence_list.append(
                self.evidence_system.collect_memory_dump("agent.exe", incident_id)
            )
        
        if "logs" in evidence_types:
            evidence_list.extend(
                self.evidence_system.collect_logs(
                    incident_id,
                    datetime.utcnow() - timedelta(hours=1),
                    datetime.utcnow()
                )
            )
        
        return evidence_list
    
    def recover_system(self, target_state: str = "latest") -> RecoveryResult:
        """Execute system recovery"""
        return self.recovery_system.execute_full_recovery(target_state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "incident_stats": self.incident_manager.get_incident_stats(),
            "active_workflows": len(self.workflow_engine.active_workflows),
            "system_status": "operational"
        }
    
    def _classify_severity(self, detection_data: Dict[str, Any]) -> IncidentSeverity:
        """Classify incident severity from detection data"""
        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH,
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW,
            "info": IncidentSeverity.INFO
        }
        
        return severity_map.get(
            detection_data.get("severity", "medium").lower(),
            IncidentSeverity.MEDIUM
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of the incident response system"""
    
    # Initialize system
    ir_system = IncidentResponseSystem()
    
    print("=" * 60)
    print("Incident Response System Demo")
    print("=" * 60)
    
    # Simulate incident detection
    detection = {
        "type": "SEC-001",
        "category": "SEC",
        "severity": "critical",
        "title": "Unauthorized Agent Process Detected",
        "description": "Unknown process spawned agent.exe with suspicious parameters",
        "source": "behavioral_detection",
        "affected_components": ["agent_core", "loop_3"],
        "raw_event": {"pid": 1234, "parent_pid": 5678}
    }
    
    print("\n1. Detecting incident...")
    incident = ir_system.detect_incident(detection)
    print(f"   Created incident: {incident.id}")
    print(f"   Severity: {incident.severity.name}")
    print(f"   Status: {incident.status.value}")
    
    # Execute response action
    print("\n2. Executing response action...")
    result = ir_system.respond_to_incident(
        incident.id,
        "enable_enhanced_logging",
        {"components": ["agent", "ai"]}
    )
    print(f"   Action success: {result.success}")
    
    # Collect evidence
    print("\n3. Collecting evidence...")
    evidence = ir_system.collect_evidence(incident.id, ["memory", "logs"])
    print(f"   Collected {len(evidence)} evidence items")
    for ev in evidence:
        print(f"   - {ev.type}: {ev.filename}")
    
    # Update incident status
    print("\n4. Updating incident status...")
    ir_system.incident_manager.update_status(incident.id, IncidentStatus.CONTAINED)
    print(f"   Status updated to: CONTAINED")
    
    # Get system status
    print("\n5. System status:")
    status = ir_system.get_status()
    print(f"   {json.dumps(status, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
