"""
Auto-Scaling Framework for OpenClaw AI Agent System
Implements intelligent scaling policies with predictive capabilities
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"
    EMERGENCY = "emergency"


class ScalingPolicy(Enum):
    """Scaling policy types"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


@dataclass
class ScalingRule:
    """Defines a scaling rule"""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte'
    duration_seconds: int
    action: ScalingAction
    instance_change: int
    cooldown_seconds: int
    priority: int = 1


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    queue_depth: int
    request_rate: float
    response_time_ms: float
    error_rate: float
    active_instances: int
    gpt_tokens_per_min: int
    gpt_response_time_ms: float


class AutoScalerMetricsCollector:
    """Collects and stores system metrics for auto-scaling decisions"""
    
    def __init__(self, max_history: int = 3600):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
    
    @staticmethod
    def _get_active_instances() -> int:
        """Get count of active child processes, handling permission errors."""
        try:
            import psutil as _psutil
            return len(_psutil.Process().children())
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            return 1

    def collect(self) -> MetricSnapshot:
        """Collect current metrics"""
        import psutil
        
        import os as _os

        mc = self._metrics_collector if hasattr(self, '_metrics_collector') else None
        err = mc.get_latest('error_count') if mc else 0
        total = mc.get_latest('total_count') if mc else 1

        snapshot = MetricSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent if _os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            queue_depth=len(getattr(self, 'task_queue', [])),
            request_rate=mc.get_latest('request_rate') if mc else 0.0,
            response_time_ms=mc.get_latest('response_time') if mc else 0.0,
            error_rate=err / max(total, 1),
            active_instances=self._get_active_instances(),
            gpt_tokens_per_min=mc.get_latest('gpt_tokens_per_min') if mc else 0.0,
            gpt_response_time_ms=mc.get_latest('gpt_response_time') if mc else 0.0,
        )
        
        with self._lock:
            self.metrics_history.append(snapshot)
        
        return snapshot
    
    def get_history(self, seconds: int = 300) -> List[MetricSnapshot]:
        """Get metrics history for last N seconds"""
        cutoff = time.time() - seconds
        with self._lock:
            return [
                m for m in self.metrics_history
                if m.timestamp >= cutoff
            ]
    
    def get_average(self, metric_name: str, seconds: int = 300) -> float:
        """Get average value of a metric over time period"""
        history = self.get_history(seconds)
        if not history:
            return 0.0
        
        values = [getattr(m, metric_name) for m in history]
        return statistics.mean(values)
    
    def get_trend(self, metric_name: str, seconds: int = 300) -> float:
        """Get trend (slope) of a metric over time period"""
        history = self.get_history(seconds)
        if len(history) < 2:
            return 0.0
        
        values = [getattr(m, metric_name) for m in history]
        timestamps = [m.timestamp for m in history]
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(timestamps, values))
        sum_x2 = sum(t * t for t in timestamps)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0


class PredictiveScaler:
    """
    Predictive scaling using time-series analysis
    """
    
    def __init__(self, metrics_collector: AutoScalerMetricsCollector):
        self.metrics = metrics_collector
        self.prediction_model = None
        
    def predict_load(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """
        Predict load for next N minutes
        
        Uses simple trend extrapolation
        For production, replace with ML model (LSTM, Prophet, etc.)
        """
        predictions = {}
        
        # Get current values and trends
        current_cpu = self.metrics.get_average('cpu_percent', 300)
        cpu_trend = self.metrics.get_trend('cpu_percent', 300)
        
        current_memory = self.metrics.get_average('memory_percent', 300)
        memory_trend = self.metrics.get_trend('memory_percent', 300)
        
        current_queue = self.metrics.get_average('queue_depth', 300)
        queue_trend = self.metrics.get_trend('queue_depth', 300)
        
        # Extrapolate
        horizon_seconds = horizon_minutes * 60
        
        predictions['cpu_percent'] = max(0, min(100, 
            current_cpu + cpu_trend * horizon_seconds))
        predictions['memory_percent'] = max(0, min(100,
            current_memory + memory_trend * horizon_seconds))
        predictions['queue_depth'] = max(0,
            current_queue + queue_trend * horizon_seconds)
        
        return predictions
    
    def should_pre_scale(self) -> Tuple[bool, int]:
        """
        Determine if we should pre-scale based on prediction
        
        Returns:
            (should_scale, instances_to_add)
        """
        predictions = self.predict_load(horizon_minutes=15)
        
        # Check if any metric will exceed threshold
        if predictions['cpu_percent'] > 70:
            return True, 2
        if predictions['memory_percent'] > 80:
            return True, 1
        if predictions['queue_depth'] > 50:
            return True, 2
        
        return False, 0


class AutoScaler:
    """
    Main auto-scaling controller
    
    Implements reactive, predictive, and scheduled scaling policies
    """
    
    DEFAULT_RULES = [
        # Scale out rules
        ScalingRule(
            name="scale_out_cpu",
            metric="cpu_percent",
            threshold=70.0,
            comparison="gt",
            duration_seconds=120,
            action=ScalingAction.SCALE_OUT,
            instance_change=1,
            cooldown_seconds=60,
            priority=1
        ),
        ScalingRule(
            name="scale_out_fast_cpu",
            metric="cpu_percent",
            threshold=85.0,
            comparison="gt",
            duration_seconds=60,
            action=ScalingAction.SCALE_OUT,
            instance_change=3,
            cooldown_seconds=60,
            priority=2
        ),
        ScalingRule(
            name="scale_out_queue",
            metric="queue_depth",
            threshold=100.0,
            comparison="gt",
            duration_seconds=120,
            action=ScalingAction.SCALE_OUT,
            instance_change=5,
            cooldown_seconds=60,
            priority=2
        ),
        
        # Scale in rules
        ScalingRule(
            name="scale_in_cpu",
            metric="cpu_percent",
            threshold=30.0,
            comparison="lt",
            duration_seconds=600,
            action=ScalingAction.SCALE_IN,
            instance_change=1,
            cooldown_seconds=300,
            priority=1
        ),
        ScalingRule(
            name="scale_in_safe_cpu",
            metric="cpu_percent",
            threshold=20.0,
            comparison="lt",
            duration_seconds=900,
            action=ScalingAction.SCALE_IN,
            instance_change=2,
            cooldown_seconds=300,
            priority=1
        ),
        
        # Emergency rules
        ScalingRule(
            name="emergency_high_cpu",
            metric="cpu_percent",
            threshold=95.0,
            comparison="gt",
            duration_seconds=30,
            action=ScalingAction.EMERGENCY,
            instance_change=5,
            cooldown_seconds=0,
            priority=10
        ),
    ]
    
    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 100,
                 check_interval: int = 10,
                 policy: ScalingPolicy = ScalingPolicy.HYBRID):
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.check_interval = check_interval
        self.policy = policy
        
        # Components
        self.metrics_collector = AutoScalerMetricsCollector()
        self.predictive_scaler = PredictiveScaler(self.metrics_collector)
        
        # Rules
        self.rules: List[ScalingRule] = list(self.DEFAULT_RULES)
        
        # State
        self.current_instances = min_instances
        self.last_scale_time = 0
        self.rule_violations: Dict[str, List[float]] = {}
        
        # Callbacks
        self.scale_out_callback: Optional[Callable[[int], bool]] = None
        self.scale_in_callback: Optional[Callable[[int], bool]] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._scaling_thread: Optional[threading.Thread] = None
        
        logger.info(f"AutoScaler initialized with policy: {policy.value}")
    
    def set_scale_callbacks(self,
                           scale_out: Callable[[int], bool],
                           scale_in: Callable[[int], bool]):
        """Set callbacks for scaling actions"""
        self.scale_out_callback = scale_out
        self.scale_in_callback = scale_in
    
    def add_rule(self, rule: ScalingRule):
        """Add a custom scaling rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a scaling rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed scaling rule: {rule_name}")
    
    def evaluate_rule(self, rule: ScalingRule, metrics: MetricSnapshot) -> bool:
        """Evaluate if a rule condition is met"""
        value = getattr(metrics, rule.metric, None)
        if value is None:
            return False
        
        if rule.comparison == "gt":
            return value > rule.threshold
        elif rule.comparison == "lt":
            return value < rule.threshold
        elif rule.comparison == "gte":
            return value >= rule.threshold
        elif rule.comparison == "lte":
            return value <= rule.threshold
        
        return False
    
    def check_cooldown(self, rule: ScalingRule) -> bool:
        """Check if rule is in cooldown period"""
        if rule.cooldown_seconds == 0:
            return True
        
        time_since_last_scale = time.time() - self.last_scale_time
        return time_since_last_scale >= rule.cooldown_seconds
    
    def determine_action(self, metrics: MetricSnapshot) -> Tuple[ScalingAction, int]:
        """Determine scaling action based on rules and metrics"""
        
        # Check each rule in priority order
        for rule in self.rules:
            if not self.check_cooldown(rule):
                continue
            
            if not hasattr(self, '_violation_start_times'):
                self._violation_start_times = {}

            if self.evaluate_rule(rule, metrics):
                # Track violation start time
                if rule.name not in self._violation_start_times:
                    self._violation_start_times[rule.name] = time.time()
                elapsed = time.time() - self._violation_start_times[rule.name]
                if elapsed >= rule.duration_seconds:
                    self._violation_start_times.pop(rule.name, None)
                    return rule.action, rule.instance_change
            else:
                # Violation cleared - reset start time
                self._violation_start_times.pop(rule.name, None)
        
        # Check predictive scaling if enabled
        if self.policy in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID]:
            should_scale, instances = self.predictive_scaler.should_pre_scale()
            if should_scale:
                return ScalingAction.SCALE_OUT, instances
        
        return ScalingAction.NO_ACTION, 0
    
    def execute_scaling(self, action: ScalingAction, instance_change: int) -> bool:
        """Execute scaling action"""
        
        if action == ScalingAction.NO_ACTION:
            return True
        
        if action == ScalingAction.SCALE_OUT:
            new_count = min(
                self.current_instances + instance_change,
                self.max_instances
            )
            actual_change = new_count - self.current_instances
            
            if actual_change > 0 and self.scale_out_callback:
                if self.scale_out_callback(actual_change):
                    self.current_instances = new_count
                    self.last_scale_time = time.time()
                    logger.info(f"Scaled out: +{actual_change} instances, "
                              f"total: {self.current_instances}")
                    return True
        
        elif action == ScalingAction.SCALE_IN:
            new_count = max(
                self.current_instances - instance_change,
                self.min_instances
            )
            actual_change = self.current_instances - new_count
            
            if actual_change > 0 and self.scale_in_callback:
                if self.scale_in_callback(actual_change):
                    self.current_instances = new_count
                    self.last_scale_time = time.time()
                    logger.info(f"Scaled in: -{actual_change} instances, "
                              f"total: {self.current_instances}")
                    return True
        
        elif action == ScalingAction.EMERGENCY:
            logger.critical(f"EMERGENCY scaling: +{instance_change} instances")
            new_count = min(
                self.current_instances + instance_change,
                self.max_instances
            )
            if self.scale_out_callback:
                self.scale_out_callback(new_count - self.current_instances)
                self.current_instances = new_count
                self.last_scale_time = time.time()
            return True
        
        return False
    
    def scaling_loop(self):
        """Main scaling loop"""
        logger.info("Auto-scaling loop started")
        
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect()
                
                # Determine action
                action, instance_change = self.determine_action(metrics)
                
                # Execute if needed
                if action != ScalingAction.NO_ACTION:
                    self.execute_scaling(action, instance_change)
                
            except (OSError, RuntimeError, PermissionError) as e:
                logger.error(f"Error in scaling loop: {e}")
            
            # Wait for next iteration
            self._stop_event.wait(self.check_interval)
        
        logger.info("Auto-scaling loop stopped")
    
    def start(self):
        """Start auto-scaling"""
        self._scaling_thread = threading.Thread(
            target=self.scaling_loop,
            daemon=False
        )
        self._scaling_thread.start()
        logger.info("Auto-scaling started")

    def stop(self):
        """Stop auto-scaling"""
        self._stop_event.set()
        if self._scaling_thread:
            self._scaling_thread.join(timeout=15)
            if self._scaling_thread.is_alive():
                logger.warning("Auto-scaling thread did not terminate in time")
        logger.info("Auto-scaling stopped")
    
    def get_status(self) -> Dict:
        """Get current auto-scaler status"""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "policy": self.policy.value,
            "last_scale_time": self.last_scale_time,
            "active_rules": len(self.rules),
            "rule_violations": {
                name: len(violations)
                for name, violations in self.rule_violations.items()
            }
        }


class ScheduledScaler:
    """
    Scheduled scaling based on time patterns
    """
    
    def __init__(self):
        self.schedules: List[Dict] = []
    
    def add_schedule(self, 
                    day_of_week: Optional[int] = None,
                    hour: Optional[int] = None,
                    min_instances: int = 1,
                    max_instances: int = 100):
        """Add a scheduled scaling configuration"""
        self.schedules.append({
            'day_of_week': day_of_week,
            'hour': hour,
            'min_instances': min_instances,
            'max_instances': max_instances
        })
    
    def get_config_for_time(self, dt=None) -> Dict:
        """Get scaling config for a specific time"""
        from datetime import datetime
        
        if dt is None:
            dt = datetime.now()
        
        # Find matching schedule (most specific wins)
        matching = []
        for schedule in self.schedules:
            match = True
            if schedule['day_of_week'] is not None:
                match = match and (dt.weekday() == schedule['day_of_week'])
            if schedule['hour'] is not None:
                match = match and (dt.hour == schedule['hour'])
            
            if match:
                matching.append(schedule)
        
        if matching:
            # Return most specific match
            return max(matching, key=lambda s: 
                      (s['day_of_week'] is not None) + (s['hour'] is not None))
        
        return {'min_instances': 1, 'max_instances': 100}


# Example usage
if __name__ == "__main__":
    # Create auto-scaler
    scaler = AutoScaler(
        min_instances=2,
        max_instances=10,
        policy=ScalingPolicy.HYBRID
    )
    
    # Define scaling callbacks
    def scale_out(count: int) -> bool:
        print(f"Scaling out: +{count} instances")
        return True
    
    def scale_in(count: int) -> bool:
        print(f"Scaling in: -{count} instances")
        return True
    
    scaler.set_scale_callbacks(scale_out, scale_in)
    
    # Add custom rule
    scaler.add_rule(ScalingRule(
        name="custom_memory_rule",
        metric="memory_percent",
        threshold=85.0,
        comparison="gt",
        duration_seconds=180,
        action=ScalingAction.SCALE_OUT,
        instance_change=2,
        cooldown_seconds=120,
        priority=3
    ))
    
    # Start auto-scaling
    scaler.start()
    
    # Run for demonstration
    try:
        time.sleep(60)
    finally:
        scaler.stop()
        print("Status:", scaler.get_status())
