"""
Monitoring and Visualization System for E2E Loop
OpenClaw-Inspired AI Agent System for Windows 10

This module provides comprehensive monitoring and visualization:
- Real-time metrics collection
- Web dashboard
- Audit logging
- Alerting system
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Coroutine
from collections import defaultdict
import uuid

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# METRICS CLASSES
# ============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    values: List[MetricValue] = field(default_factory=list)
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        self.values.append(MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        ))
        
        # Keep only last 10000 values
        if len(self.values) > 10000:
            self.values = self.values[-10000:]
    
    def get_latest(self, labels_filter: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get latest value matching labels."""
        for value in reversed(self.values):
            if labels_filter is None or all(
                value.labels.get(k) == v for k, v in labels_filter.items()
            ):
                return value.value
        return None
    
    def get_sum(self, labels_filter: Optional[Dict[str, str]] = None) -> float:
        """Get sum of values."""
        return sum(
            v.value for v in self.values
            if labels_filter is None or all(
                v.labels.get(k) == v for k, v in labels_filter.items()
            )
        )
    
    def get_count(self, labels_filter: Optional[Dict[str, str]] = None) -> int:
        """Get count of values."""
        return len([
            v for v in self.values
            if labels_filter is None or all(
                v.labels.get(k) == v for k, v in labels_filter.items()
            )
        ])


# ============================================================================
# METRICS REGISTRY
# ============================================================================

class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._collectors: List[Callable] = []
    
    def register(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Metric:
        """Register a new metric."""
        if name in self._metrics:
            return self._metrics[name]
        
        metric = Metric(
            name=name,
            type=metric_type,
            description=description,
            labels=labels or []
        )
        self._metrics[name] = metric
        return metric
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Metric:
        """Create or get a counter metric."""
        return self.register(name, MetricType.COUNTER, description, labels)
    
    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Metric:
        """Create or get a gauge metric."""
        return self.register(name, MetricType.GAUGE, description, labels)
    
    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Metric:
        """Create or get a histogram metric."""
        return self.register(name, MetricType.HISTOGRAM, description, labels)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return dict(self._metrics)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, metric in self._metrics.items():
            # Type line
            lines.append(f"# TYPE {name} {metric.type.value}")
            lines.append(f"# HELP {name} {metric.description}")
            
            # Values
            for value in metric.values[-100:]:  # Last 100 values
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in value.labels.items()
                )
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {value.value}")
                else:
                    lines.append(f"{name} {value.value}")
        
        return "\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        return {
            name: {
                'type': metric.type.value,
                'description': metric.description,
                'values': [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat(),
                        'labels': v.labels
                    }
                    for v in metric.values[-100:]
                ]
            }
            for name, metric in self._metrics.items()
        }


# Global metrics registry
GLOBAL_METRICS = MetricsRegistry()


# ============================================================================
# WORKFLOW METRICS COLLECTOR
# ============================================================================

class WorkflowMetricsCollector:
    """Collects metrics for workflow execution."""
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or GLOBAL_METRICS
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup workflow metrics."""
        # Execution metrics
        self.workflow_executions_total = self.registry.counter(
            'e2e_workflow_executions_total',
            'Total number of workflow executions',
            ['workflow_id', 'status']
        )
        
        self.workflow_execution_duration = self.registry.histogram(
            'e2e_workflow_execution_duration_seconds',
            'Workflow execution duration in seconds',
            ['workflow_id']
        )
        
        self.workflow_active = self.registry.gauge(
            'e2e_workflows_active',
            'Number of active workflows',
            ['workflow_id', 'status']
        )
        
        # Task metrics
        self.task_executions_total = self.registry.counter(
            'e2e_task_executions_total',
            'Total number of task executions',
            ['workflow_id', 'task_id', 'status']
        )
        
        self.task_execution_duration = self.registry.histogram(
            'e2e_task_execution_duration_seconds',
            'Task execution duration in seconds',
            ['workflow_id', 'task_id']
        )
        
        self.task_retries_total = self.registry.counter(
            'e2e_task_retries_total',
            'Total number of task retries',
            ['workflow_id', 'task_id']
        )
        
        # Checkpoint metrics
        self.checkpoints_created_total = self.registry.counter(
            'e2e_checkpoints_created_total',
            'Total number of checkpoints created',
            ['workflow_id']
        )
        
        self.checkpoint_size_bytes = self.registry.histogram(
            'e2e_checkpoint_size_bytes',
            'Checkpoint size in bytes',
            ['workflow_id']
        )
        
        # HITL metrics
        self.approval_requests_total = self.registry.counter(
            'e2e_approval_requests_total',
            'Total number of approval requests',
            ['workflow_id', 'status']
        )
        
        self.approval_response_time = self.registry.histogram(
            'e2e_approval_response_time_seconds',
            'Time to receive approval response',
            ['workflow_id']
        )
        
        # Resource metrics
        self.parallel_executions = self.registry.gauge(
            'e2e_parallel_executions',
            'Current number of parallel executions',
            ['workflow_id']
        )
        
        self.queue_depth = self.registry.gauge(
            'e2e_queue_depth',
            'Current task queue depth',
            ['workflow_id']
        )
    
    def record_workflow_start(self, workflow_id: str) -> None:
        """Record workflow start."""
        self.workflow_executions_total.record(1, {
            'workflow_id': workflow_id,
            'status': 'started'
        })
        self.workflow_active.record(1, {
            'workflow_id': workflow_id,
            'status': 'running'
        })
    
    def record_workflow_complete(
        self,
        workflow_id: str,
        duration_seconds: float,
        status: str
    ) -> None:
        """Record workflow completion."""
        self.workflow_executions_total.record(1, {
            'workflow_id': workflow_id,
            'status': status
        })
        self.workflow_execution_duration.record(duration_seconds, {
            'workflow_id': workflow_id
        })
        self.workflow_active.record(-1, {
            'workflow_id': workflow_id,
            'status': 'running'
        })
    
    def record_task_execution(
        self,
        workflow_id: str,
        task_id: str,
        duration_seconds: float,
        status: str
    ) -> None:
        """Record task execution."""
        self.task_executions_total.record(1, {
            'workflow_id': workflow_id,
            'task_id': task_id,
            'status': status
        })
        self.task_execution_duration.record(duration_seconds, {
            'workflow_id': workflow_id,
            'task_id': task_id
        })
    
    def record_task_retry(self, workflow_id: str, task_id: str) -> None:
        """Record task retry."""
        self.task_retries_total.record(1, {
            'workflow_id': workflow_id,
            'task_id': task_id
        })
    
    def record_checkpoint(self, workflow_id: str, size_bytes: int) -> None:
        """Record checkpoint creation."""
        self.checkpoints_created_total.record(1, {
            'workflow_id': workflow_id
        })
        self.checkpoint_size_bytes.record(size_bytes, {
            'workflow_id': workflow_id
        })
    
    def record_approval_request(
        self,
        workflow_id: str,
        status: str
    ) -> None:
        """Record approval request."""
        self.approval_requests_total.record(1, {
            'workflow_id': workflow_id,
            'status': status
        })
    
    def record_approval_response_time(
        self,
        workflow_id: str,
        response_time_seconds: float
    ) -> None:
        """Record approval response time."""
        self.approval_response_time.record(response_time_seconds, {
            'workflow_id': workflow_id
        })


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditEventType(Enum):
    """Types of audit events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRY = "task_retry"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESPONDED = "approval_responded"
    APPROVAL_TIMEOUT = "approval_timeout"
    COMPENSATION_EXECUTED = "compensation_executed"


@dataclass
class AuditEvent:
    """Audit event record."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    workflow_id: Optional[str]
    task_id: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    source_ip: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'details': self.details,
            'source_ip': self.source_ip,
            'session_id': self.session_id
        }


class AuditLogger:
    """Audit logging system."""
    
    def __init__(self, storage: Optional['AuditStorage'] = None):
        self.storage = storage
        self._callbacks: List[Callable[[AuditEvent], Coroutine]] = []
    
    async def log(
        self,
        event_type: AuditEventType,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            workflow_id=workflow_id,
            task_id=task_id,
            user_id=user_id,
            details=details or {},
            source_ip=kwargs.get('source_ip'),
            session_id=kwargs.get('session_id')
        )
        
        # Store event
        if self.storage:
            await self.storage.save_event(event)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(event)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Audit callback error: {e}")
        
        return event
    
    def register_callback(
        self,
        callback: Callable[[AuditEvent], Coroutine]
    ) -> None:
        """Register audit event callback."""
        self._callbacks.append(callback)
    
    async def query(
        self,
        workflow_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events."""
        if self.storage:
            return await self.storage.query_events(
                workflow_id=workflow_id,
                event_types=event_types,
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                limit=limit
            )
        return []


class AuditStorage:
    """Storage backend for audit events."""
    
    def __init__(self):
        self._events: List[AuditEvent] = []
    
    async def save_event(self, event: AuditEvent) -> None:
        """Save an audit event."""
        self._events.append(event)
        
        # Keep only last 100000 events
        if len(self._events) > 100000:
            self._events = self._events[-100000:]
    
    async def query_events(
        self,
        workflow_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        events = self._events
        
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        # Sort by timestamp descending
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]


# ============================================================================
# ALERTING SYSTEM
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    workflow_id: Optional[str]
    task_id: Optional[str]
    created_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[MetricsRegistry], bool],
        severity: AlertSeverity,
        title_template: str,
        message_template: str
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.title_template = title_template
        self.message_template = message_template
        self.last_triggered: Optional[datetime] = None
        self.cooldown: timedelta = timedelta(minutes=5)
    
    def check(self, registry: MetricsRegistry) -> Optional[Alert]:
        """Check if alert should be triggered."""
        # Check cooldown
        if self.last_triggered:
            if datetime.utcnow() - self.last_triggered < self.cooldown:
                return None
        
        # Check condition
        if self.condition(registry):
            self.last_triggered = datetime.utcnow()
            
            return Alert(
                id=str(uuid.uuid4()),
                severity=self.severity,
                title=self.title_template,
                message=self.message_template,
                workflow_id=None,
                task_id=None,
                created_at=datetime.utcnow()
            )
        
        return None


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or GLOBAL_METRICS
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self._callbacks: List[Callable[[Alert], Coroutine]] = []
        self._running = False
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)
    
    def register_callback(
        self,
        callback: Callable[[Alert], Coroutine]
    ) -> None:
        """Register alert callback."""
        self._callbacks.append(callback)
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start alert monitoring loop."""
        self._running = True
        
        while self._running:
            try:
                await self._check_rules()
                await asyncio.sleep(interval)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Alert monitoring error: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        self._running = False
    
    async def _check_rules(self) -> None:
        """Check all alert rules."""
        for rule in self.rules:
            alert = rule.check(self.registry)
            if alert:
                await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert."""
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(alert)
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Alert triggered: {alert.title}")
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str
    ) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active (unacknowledged) alerts."""
        alerts = [a for a in self.alerts if not a.acknowledged]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by severity and time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at))
        
        return alerts


def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules."""
    rules = []
    
    # High failure rate alert
    rules.append(AlertRule(
        name="high_failure_rate",
        condition=lambda r: (
            (r.get('e2e_workflow_executions_total') or Metric('dummy', MetricType.COUNTER, '')).get_count({'status': 'failed'}) >
            (r.get('e2e_workflow_executions_total') or Metric('dummy', MetricType.COUNTER, '')).get_count({'status': 'completed'}) * 0.2
        ),
        severity=AlertSeverity.ERROR,
        title_template="High Workflow Failure Rate",
        message_template="More than 20% of workflows are failing"
    ))
    
    # Long-running workflow alert
    rules.append(AlertRule(
        name="long_running_workflow",
        condition=lambda r: (
            (r.get('e2e_workflow_execution_duration_seconds') or Metric('dummy', MetricType.HISTOGRAM, '')).get_latest() or 0
        ) > 3600,
        severity=AlertSeverity.WARNING,
        title_template="Long-Running Workflow Detected",
        message_template="A workflow has been running for over 1 hour"
    ))
    
    # Approval timeout alert
    rules.append(AlertRule(
        name="approval_timeout",
        condition=lambda r: (
            (r.get('e2e_approval_requests_total') or Metric('dummy', MetricType.COUNTER, '')).get_count({'status': 'timed_out'})
        ) > 0,
        severity=AlertSeverity.WARNING,
        title_template="Approval Request Timed Out",
        message_template="An approval request has timed out and may require attention"
    ))
    
    return rules


# ============================================================================
# WEB DASHBOARD
# ============================================================================

class DashboardServer:
    """Web dashboard server for monitoring."""
    
    def __init__(
        self,
        metrics_registry: Optional[MetricsRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        self.metrics = metrics_registry or GLOBAL_METRICS
        self.audit_logger = audit_logger
        self.alert_manager = alert_manager
        self.app = None
    
    def create_app(self) -> 'FastAPI':
        """Create FastAPI application."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
        
        app = FastAPI(title="E2E Loop Dashboard")
        self.app = app
        
        # API Routes
        @app.get("/api/metrics")
        async def get_metrics():
            """Get all metrics."""
            return JSONResponse(self.metrics.export_json())
        
        @app.get("/api/metrics/prometheus")
        async def get_prometheus_metrics():
            """Get metrics in Prometheus format."""
            return PlainTextResponse(self.metrics.export_prometheus())
        
        @app.get("/api/workflows")
        async def list_workflows(
            status: Optional[str] = None,
            limit: int = 100
        ):
            """List workflows from file-based state storage."""
            import os
            import glob

            state_dir = os.path.join(
                os.path.expanduser('~'), '.openclaw', 'state', 'workflows'
            )
            workflows = []

            if os.path.isdir(state_dir):
                pattern = os.path.join(state_dir, '*.json')
                for filepath in sorted(glob.glob(pattern), reverse=True):
                    if len(workflows) >= limit:
                        break
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            wf_data = json.loads(f.read())
                        # Filter by status if specified
                        if status and wf_data.get('status') != status:
                            continue
                        workflows.append(wf_data)
                    except (json.JSONDecodeError, OSError, KeyError) as e:
                        logger.debug(f"Skipping workflow file {filepath}: {e}")
                        continue

            return JSONResponse({
                'workflows': workflows,
                'total': len(workflows)
            })
        
        @app.get("/api/alerts")
        async def get_alerts(
            severity: Optional[str] = None,
            active_only: bool = True
        ):
            """Get alerts."""
            if self.alert_manager:
                sev = AlertSeverity(severity) if severity else None
                alerts = self.alert_manager.get_active_alerts(sev)
                return JSONResponse({
                    'alerts': [a.to_dict() for a in alerts],
                    'total': len(alerts)
                })
            return JSONResponse({'alerts': [], 'total': 0})
        
        @app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, user_id: str):
            """Acknowledge an alert."""
            if self.alert_manager:
                success = await self.alert_manager.acknowledge_alert(alert_id, user_id)
                return JSONResponse({'success': success})
            return JSONResponse({'success': False})
        
        @app.get("/api/audit")
        async def query_audit(
            workflow_id: Optional[str] = None,
            limit: int = 100
        ):
            """Query audit log."""
            if self.audit_logger:
                events = await self.audit_logger.query(
                    workflow_id=workflow_id,
                    limit=limit
                )
                return JSONResponse({
                    'events': [e.to_dict() for e in events],
                    'total': len(events)
                })
            return JSONResponse({'events': [], 'total': 0})
        
        # WebSocket for real-time updates
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            try:
                while True:
                    # Send metrics update
                    await websocket.send_json({
                        'type': 'metrics',
                        'data': self.metrics.export_json()
                    })
                    
                    # Send alerts if available
                    if self.alert_manager:
                        alerts = self.alert_manager.get_active_alerts()
                        await websocket.send_json({
                            'type': 'alerts',
                            'data': [a.to_dict() for a in alerts]
                        })
                    
                    await asyncio.sleep(5)
                    
            except WebSocketDisconnect:
                logger.debug("WebSocket client disconnected")
        
        # Dashboard HTML
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Render main dashboard."""
            return HTMLResponse(self._render_dashboard())
        
        return app
    
    def _render_dashboard(self) -> str:
        """Render dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>E2E Loop Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 500;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: 600;
            color: #333;
        }
        
        .metric-label {
            font-size: 14px;
            color: #999;
            margin-top: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-running { background: #4CAF50; }
        .status-failed { background: #f44336; }
        .status-pending { background: #ff9800; }
        .status-completed { background: #2196F3; }
        
        .alert {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .alert-critical {
            background: #ffebee;
            border-left: 4px solid #c62828;
        }
        
        .alert-error {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        
        .alert-warning {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        
        .alert-info {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        
        .alert-title {
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .alert-message {
            font-size: 13px;
            color: #666;
        }
        
        .alert-time {
            font-size: 12px;
            color: #999;
            margin-left: auto;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .table th {
            font-weight: 500;
            color: #666;
            font-size: 13px;
            text-transform: uppercase;
        }
        
        .table tr:hover {
            background: #f9f9f9;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .badge-success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .badge-error {
            background: #ffebee;
            color: #c62828;
        }
        
        .badge-warning {
            background: #fff3e0;
            color: #e65100;
        }
        
        .badge-info {
            background: #e3f2fd;
            color: #1565c0;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .connection-connected {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .connection-disconnected {
            background: #ffebee;
            color: #c62828;
        }
        
        .refresh-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>E2E Loop Dashboard <span class="refresh-indicator"></span></h1>
    </div>
    
    <div class="connection-status connection-connected" id="connection-status">
        ● Connected
    </div>
    
    <div class="container">
        <!-- Metrics Grid -->
        <div class="grid" id="metrics-grid">
            <div class="card">
                <h3>Active Workflows</h3>
                <div class="metric-value" id="active-workflows">0</div>
                <div class="metric-label">Currently running</div>
            </div>
            
            <div class="card">
                <h3>Completed Today</h3>
                <div class="metric-value" id="completed-today">0</div>
                <div class="metric-label">Successful executions</div>
            </div>
            
            <div class="card">
                <h3>Failed Today</h3>
                <div class="metric-value" id="failed-today">0</div>
                <div class="metric-label">Failed executions</div>
            </div>
            
            <div class="card">
                <h3>Avg Duration</h3>
                <div class="metric-value" id="avg-duration">0s</div>
                <div class="metric-label">Average execution time</div>
            </div>
        </div>
        
        <!-- Alerts Section -->
        <div class="card" style="margin-bottom: 20px;">
            <h3>Active Alerts</h3>
            <div id="alerts-container">
                <p style="color: #999; padding: 20px 0;">No active alerts</p>
            </div>
        </div>
        
        <!-- Recent Workflows -->
        <div class="card">
            <h3>Recent Workflows</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Workflow</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Duration</th>
                        <th>Tasks</th>
                    </tr>
                </thead>
                <tbody id="workflows-table">
                    <tr>
                        <td colspan="5" style="text-align: center; color: #999; padding: 40px;">
                            No workflows to display
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        let ws;
        let reconnectInterval;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('connection-status').className = 'connection-status connection-connected';
                document.getElementById('connection-status').textContent = '● Connected';
                clearInterval(reconnectInterval);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'metrics') {
                    updateMetrics(data.data);
                } else if (data.type === 'alerts') {
                    updateAlerts(data.data);
                }
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').className = 'connection-status connection-disconnected';
                document.getElementById('connection-status').textContent = '● Disconnected';
                
                // Reconnect after 5 seconds
                reconnectInterval = setTimeout(connect, 5000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateMetrics(metrics) {
            // Update metrics display
            const activeWorkflows = metrics['e2e_workflows_active'] || { values: [] };
            const completedWorkflows = metrics['e2e_workflow_executions_total'] || { values: [] };
            const failedWorkflows = metrics['e2e_workflow_executions_total'] || { values: [] };
            const durations = metrics['e2e_workflow_execution_duration_seconds'] || { values: [] };
            
            // Count active workflows
            const activeCount = activeWorkflows.values.filter(v => v.value > 0).length;
            document.getElementById('active-workflows').textContent = activeCount;
            
            // Count completed today
            const completedCount = completedWorkflows.values.filter(
                v => v.labels.status === 'completed'
            ).length;
            document.getElementById('completed-today').textContent = completedCount;
            
            // Count failed today
            const failedCount = failedWorkflows.values.filter(
                v => v.labels.status === 'failed'
            ).length;
            document.getElementById('failed-today').textContent = failedCount;
            
            // Calculate average duration
            if (durations.values.length > 0) {
                const avgDuration = durations.values.reduce((a, b) => a + b.value, 0) / durations.values.length;
                document.getElementById('avg-duration').textContent = formatDuration(avgDuration);
            }
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (alerts.length === 0) {
                container.innerHTML = '<p style="color: #999; padding: 20px 0;">No active alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert alert-${alert.severity}">
                    <div>
                        <div class="alert-title">${alert.title}</div>
                        <div class="alert-message">${alert.message}</div>
                    </div>
                    <div class="alert-time">${formatTime(alert.created_at)}</div>
                </div>
            `).join('');
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) {
                return Math.round(seconds) + 's';
            } else if (seconds < 3600) {
                return Math.round(seconds / 60) + 'm';
            } else {
                return Math.round(seconds / 3600 * 10) / 10 + 'h';
            }
        }
        
        function formatTime(isoString) {
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        }
        
        // Connect on load
        connect();
    </script>
</body>
</html>
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_monitoring_usage():
    """Example usage of monitoring system."""
    
    # Create metrics collector
    metrics = WorkflowMetricsCollector()
    
    # Create audit logger
    audit_storage = AuditStorage()
    audit_logger = AuditLogger(audit_storage)
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Add default alert rules
    for rule in create_default_alert_rules():
        alert_manager.add_rule(rule)
    
    # Register alert callback
    async def on_alert(alert: Alert):
        print(f"ALERT: [{alert.severity.value.upper()}] {alert.title}")
    
    alert_manager.register_callback(on_alert)
    
    # Record some metrics
    metrics.record_workflow_start("workflow-1")
    metrics.record_task_execution("workflow-1", "task-1", 2.5, "completed")
    metrics.record_task_execution("workflow-1", "task-2", 1.0, "completed")
    metrics.record_workflow_complete("workflow-1", 3.5, "completed")
    
    # Log audit events
    await audit_logger.log(
        AuditEventType.WORKFLOW_STARTED,
        workflow_id="workflow-1",
        details={'inputs': {'key': 'value'}}
    )
    
    # Create dashboard
    dashboard = DashboardServer(
        metrics_registry=GLOBAL_METRICS,
        audit_logger=audit_logger,
        alert_manager=alert_manager
    )
    
    app = dashboard.create_app()
    
    print("Monitoring system initialized")
    print(f"Metrics: {len(GLOBAL_METRICS.get_all_metrics())} registered")
    
    # Export metrics
    print("\nPrometheus Metrics:")
    print(GLOBAL_METRICS.export_prometheus()[:500] + "...")


if __name__ == "__main__":
    asyncio.run(example_monitoring_usage())
