"""
Monitoring and Alerting System for OpenClaw AI Agent System
Collects metrics, generates alerts, and provides dashboards
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels"""
    CONSOLE = "console"
    LOG = "log"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"


@dataclass
class Metric:
    """Time-series metric"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric: str
    condition: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    threshold: float
    duration_seconds: int
    severity: AlertSeverity
    channels: List[AlertChannel]
    message_template: str
    auto_resolve: bool = True
    cooldown_seconds: int = 300


@dataclass
class Alert:
    """Active or historical alert"""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    triggered_at: float
    resolved_at: Optional[float] = None
    metric_value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    notification_sent: bool = False


class MetricsCollector:
    """
    Centralized metrics collection system
    
    Features:
    - Time-series data storage
    - Metric aggregation
    - Label support for dimensional metrics
    """
    
    def __init__(self, retention_seconds: int = 86400):
        self.retention_seconds = retention_seconds
        
        # Metrics storage: {metric_name: [Metric, ...]}
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Aggregated metrics
        self.aggregates: Dict[str, Dict] = {}
        
        # Custom collectors
        self.custom_collectors: List[Callable] = []
        
        # Lock
        self._lock = threading.RLock()
        
        logger.info("MetricsCollector initialized")
    
    def record(self, 
              name: str, 
              value: float, 
              labels: Dict[str, str] = None,
              unit: str = "") -> None:
        """Record a metric value"""
        if labels is None:
            labels = {}
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels,
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_metrics(self, 
                   name: str,
                   labels_filter: Dict[str, str] = None,
                   seconds: int = 3600) -> List[Metric]:
        """Get metrics for a name with optional filtering"""
        cutoff = time.time() - seconds
        
        with self._lock:
            all_metrics = self.metrics.get(name, deque())
            
            filtered = [
                m for m in all_metrics
                if m.timestamp >= cutoff
            ]
            
            if labels_filter:
                filtered = [
                    m for m in filtered
                    if all(m.labels.get(k) == v for k, v in labels_filter.items())
                ]
            
            return filtered
    
    def get_latest(self, name: str, labels_filter: Dict[str, str] = None) -> Optional[Metric]:
        """Get the latest metric value"""
        metrics = self.get_metrics(name, labels_filter, seconds=3600)
        return metrics[-1] if metrics else None
    
    def get_average(self, 
                   name: str,
                   labels_filter: Dict[str, str] = None,
                   seconds: int = 300) -> Optional[float]:
        """Get average value over time period"""
        metrics = self.get_metrics(name, labels_filter, seconds)
        
        if not metrics:
            return None
        
        return statistics.mean(m.value for m in metrics)
    
    def get_percentile(self,
                      name: str,
                      percentile: float,
                      labels_filter: Dict[str, str] = None,
                      seconds: int = 300) -> Optional[float]:
        """Get percentile value over time period"""
        metrics = self.get_metrics(name, labels_filter, seconds)
        
        if not metrics:
            return None
        
        values = sorted(m.value for m in metrics)
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def get_rate(self,
                name: str,
                labels_filter: Dict[str, str] = None,
                seconds: int = 60) -> Optional[float]:
        """Get rate of change per second"""
        metrics = self.get_metrics(name, labels_filter, seconds)
        
        if len(metrics) < 2:
            return None
        
        first = metrics[0]
        last = metrics[-1]
        
        time_diff = last.timestamp - first.timestamp
        value_diff = last.value - first.value
        
        if time_diff == 0:
            return 0
        
        return value_diff / time_diff
    
    def register_custom_collector(self, collector: Callable) -> None:
        """Register a custom metrics collector function"""
        self.custom_collectors.append(collector)
    
    def collect_all(self) -> None:
        """Run all custom collectors"""
        for collector in self.custom_collectors:
            try:
                collector(self)
            except Exception as e:
                logger.error(f"Custom collector error: {e}")


class AlertManager:
    """
    Alert management system
    
    Features:
    - Rule-based alerting
    - Multi-channel notifications
    - Alert correlation
    - Auto-resolution
    """
    
    DEFAULT_RULES = [
        # System alerts
        AlertRule(
            name="high_cpu_usage",
            metric="cpu_percent",
            condition="gt",
            threshold=70.0,
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            message_template="High CPU usage: {value}% (threshold: {threshold}%)",
            cooldown_seconds=600
        ),
        AlertRule(
            name="critical_cpu_usage",
            metric="cpu_percent",
            condition="gt",
            threshold=90.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            message_template="CRITICAL: CPU usage at {value}%",
            cooldown_seconds=300
        ),
        
        # Memory alerts
        AlertRule(
            name="high_memory_usage",
            metric="memory_percent",
            condition="gt",
            threshold=80.0,
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            message_template="High memory usage: {value}%",
            cooldown_seconds=600
        ),
        AlertRule(
            name="critical_memory_usage",
            metric="memory_percent",
            condition="gt",
            threshold=95.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            message_template="CRITICAL: Memory usage at {value}%",
            cooldown_seconds=300
        ),
        
        # GPT-5.2 alerts
        AlertRule(
            name="gpt52_high_latency",
            metric="gpt_response_time_ms",
            condition="gt",
            threshold=5000.0,
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            message_template="GPT-5.2 response time high: {value}ms",
            cooldown_seconds=600
        ),
        AlertRule(
            name="gpt52_critical_latency",
            metric="gpt_response_time_ms",
            condition="gt",
            threshold=10000.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            message_template="CRITICAL: GPT-5.2 response time: {value}ms",
            cooldown_seconds=300
        ),
        
        # Queue alerts
        AlertRule(
            name="queue_backlog",
            metric="queue_depth",
            condition="gt",
            threshold=100.0,
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            message_template="Queue backlog: {value} messages",
            cooldown_seconds=600
        ),
        
        # Instance health alerts
        AlertRule(
            name="instance_unhealthy",
            metric="healthy_instances_percent",
            condition="lt",
            threshold=80.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            message_template="Only {value}% instances healthy",
            cooldown_seconds=300
        ),
        
        # Error rate alerts
        AlertRule(
            name="error_rate_spike",
            metric="error_rate_percent",
            condition="gt",
            threshold=5.0,
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            message_template="Error rate elevated: {value}%",
            cooldown_seconds=600
        ),
        AlertRule(
            name="critical_error_rate",
            metric="error_rate_percent",
            condition="gt",
            threshold=10.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            message_template="CRITICAL: Error rate at {value}%",
            cooldown_seconds=300
        ),
    ]
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
        # Alert rules
        self.rules: List[AlertRule] = list(self.DEFAULT_RULES)
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=10000)
        
        # Rule state (for tracking duration)
        self.rule_state: Dict[str, Dict] = defaultdict(lambda: {
            "first_triggered": None,
            "last_triggered": None,
            "last_notified": None
        })
        
        # Notification handlers
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # Callbacks
        self.on_alert_triggered: Optional[Callable] = None
        self.on_alert_resolved: Optional[Callable] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._alert_thread: Optional[threading.Thread] = None
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info("AlertManager initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default notification handlers"""
        self.notification_handlers[AlertChannel.CONSOLE] = self._console_notify
        self.notification_handlers[AlertChannel.LOG] = self._log_notify
    
    def _console_notify(self, alert: Alert) -> None:
        """Console notification handler"""
        print(f"\n[ALERT] {alert.severity.value.upper()}: {alert.message}\n")
    
    def _log_notify(self, alert: Alert) -> None:
        """Log notification handler"""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.info)
        
        log_func(f"ALERT [{alert.rule_name}]: {alert.message}")
    
    def register_notification_handler(self,
                                     channel: AlertChannel,
                                     handler: Callable) -> None:
        """Register a custom notification handler"""
        self.notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for {channel.value}")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def evaluate_condition(self, 
                          rule: AlertRule, 
                          value: float) -> bool:
        """Evaluate if condition is met"""
        if rule.condition == "gt":
            return value > rule.threshold
        elif rule.condition == "lt":
            return value < rule.threshold
        elif rule.condition == "gte":
            return value >= rule.threshold
        elif rule.condition == "lte":
            return value <= rule.threshold
        elif rule.condition == "eq":
            return value == rule.threshold
        return False
    
    def check_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown"""
        state = self.rule_state[rule.name]
        last_notified = state.get("last_notified")
        
        if last_notified is None:
            return True
        
        return (time.time() - last_notified) >= rule.cooldown_seconds
    
    def send_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels"""
        rule = next((r for r in self.rules if r.name == alert.rule_name), None)
        if not rule:
            return
        
        for channel in rule.channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Notification error for {channel.value}: {e}")
        
        alert.notification_sent = True
    
    def check_rules(self) -> None:
        """Check all alert rules"""
        for rule in self.rules:
            # Get metric value
            latest = self.metrics.get_latest(rule.metric)
            
            if latest is None:
                continue
            
            value = latest.value
            state = self.rule_state[rule.name]
            
            # Check if condition is met
            if self.evaluate_condition(rule, value):
                # Track first trigger time
                if state["first_triggered"] is None:
                    state["first_triggered"] = time.time()
                
                state["last_triggered"] = time.time()
                
                # Check if duration threshold met
                duration = time.time() - state["first_triggered"]
                
                if duration >= rule.duration_seconds:
                    # Check cooldown
                    if not self.check_cooldown(rule):
                        continue
                    
                    # Create or update alert
                    alert_id = f"{rule.name}:{latest.labels.get('instance', 'global')}"
                    
                    if alert_id not in self.active_alerts:
                        message = rule.message_template.format(
                            value=round(value, 2),
                            threshold=rule.threshold
                        )
                        
                        alert = Alert(
                            alert_id=alert_id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=message,
                            triggered_at=time.time(),
                            metric_value=value,
                            labels=latest.labels
                        )
                        
                        self.active_alerts[alert_id] = alert
                        
                        # Send notifications
                        self.send_notifications(alert)
                        
                        # Callback
                        if self.on_alert_triggered:
                            self.on_alert_triggered(alert)
                        
                        logger.warning(f"Alert triggered: {rule.name}")
                    
                    # Update last notified
                    state["last_notified"] = time.time()
            
            else:
                # Condition not met - check for resolution
                if state["first_triggered"] is not None:
                    # Auto-resolve if enabled
                    if rule.auto_resolve:
                        for alert_id, alert in list(self.active_alerts.items()):
                            if alert.rule_name == rule.name:
                                alert.resolved_at = time.time()
                                self.alert_history.append(alert)
                                del self.active_alerts[alert_id]
                                
                                if self.on_alert_resolved:
                                    self.on_alert_resolved(alert)
                                
                                logger.info(f"Alert resolved: {rule.name}")
                    
                    # Reset state
                    state["first_triggered"] = None
    
    def alert_loop(self) -> None:
        """Main alert checking loop"""
        logger.info("Alert checking loop started")
        
        while not self._stop_event.is_set():
            try:
                self.check_rules()
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
            
            self._stop_event.wait(10)  # Check every 10 seconds
        
        logger.info("Alert checking loop stopped")
    
    def start(self) -> None:
        """Start alert checking"""
        self._alert_thread = threading.Thread(
            target=self.alert_loop,
            daemon=True
        )
        self._alert_thread.start()
        logger.info("Alert manager started")
    
    def stop(self) -> None:
        """Stop alert checking"""
        self._stop_event.set()
        if self._alert_thread:
            self._alert_thread.join(timeout=10)
        logger.info("Alert manager stopped")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert status"""
        return {
            "active_alerts": len(self.active_alerts),
            "by_severity": {
                severity.value: len([
                    a for a in self.active_alerts.values()
                    if a.severity == severity
                ])
                for severity in AlertSeverity
            },
            "total_rules": len(self.rules),
            "notification_channels": len(self.notification_handlers)
        }


class DashboardDataProvider:
    """
    Provides data for monitoring dashboards
    """
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 alert_manager: AlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
    
    def get_system_overview(self) -> Dict:
        """Get system overview for dashboard"""
        return {
            "timestamp": time.time(),
            "cpu": {
                "current": self.metrics.get_latest("cpu_percent"),
                "average_5m": self.metrics.get_average("cpu_percent", seconds=300),
                "peak_1h": max(
                    (m.value for m in self.metrics.get_metrics("cpu_percent", seconds=3600)),
                    default=0
                )
            },
            "memory": {
                "current": self.metrics.get_latest("memory_percent"),
                "average_5m": self.metrics.get_average("memory_percent", seconds=300),
            },
            "instances": {
                "total": self.metrics.get_latest("total_instances"),
                "healthy": self.metrics.get_latest("healthy_instances"),
                "unhealthy": self.metrics.get_latest("unhealthy_instances"),
            },
            "gpt52": {
                "response_time": self.metrics.get_latest("gpt_response_time_ms"),
                "tokens_per_min": self.metrics.get_latest("gpt_tokens_per_min"),
                "queue_depth": self.metrics.get_latest("gpt_queue_depth"),
            },
            "alerts": self.alerts.get_alert_summary()
        }
    
    def get_metrics_timeseries(self,
                               metric_names: List[str],
                               seconds: int = 3600) -> Dict:
        """Get time-series data for multiple metrics"""
        result = {}
        
        for name in metric_names:
            metrics = self.metrics.get_metrics(name, seconds=seconds)
            result[name] = [
                {"timestamp": m.timestamp, "value": m.value, "labels": m.labels}
                for m in metrics
            ]
        
        return result


# Example usage
if __name__ == "__main__":
    # Create metrics collector
    metrics = MetricsCollector()
    
    # Create alert manager
    alerts = AlertManager(metrics)
    
    # Simulate metrics
    for i in range(100):
        metrics.record("cpu_percent", 50 + (i % 30), unit="percent")
        metrics.record("memory_percent", 60 + (i % 20), unit="percent")
        metrics.record("gpt_response_time_ms", 2000 + (i % 1000), unit="ms")
        time.sleep(0.01)
    
    # Start alert manager
    alerts.start()
    
    # Run for demonstration
    try:
        time.sleep(30)
    finally:
        alerts.stop()
    
    # Show results
    print("\n=== ALERT SUMMARY ===")
    print(alerts.get_alert_summary())
    
    print("\n=== ACTIVE ALERTS ===")
    for alert in alerts.get_active_alerts():
        print(f"  {alert.severity.value}: {alert.message}")
    
    # Dashboard data
    dashboard = DashboardDataProvider(metrics, alerts)
    print("\n=== SYSTEM OVERVIEW ===")
    print(json.dumps(dashboard.get_system_overview(), indent=2, default=str))
