"""
Bottleneck Detection System for OpenClaw AI Agent System
Real-time detection and analysis of performance bottlenecks
"""

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BottleneckType(Enum):
    """Types of bottlenecks"""
    GPT52_THROTTLING = "gpt_52_throttling"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SATURATION = "cpu_saturation"
    NETWORK_CONGESTION = "network_congestion"
    DISK_IO_BOTTLENECK = "disk_io_bottleneck"
    QUEUE_BACKLOG = "queue_backlog"
    DATABASE_SLOWDOWN = "database_slowdown"
    CACHE_INEFFICIENCY = "cache_inefficiency"
    BROWSER_POOL_EXHAUSTION = "browser_pool_exhaustion"


@dataclass
class BottleneckIndicator:
    """Defines a bottleneck indicator"""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte'
    weight: float = 1.0


@dataclass
class BottleneckSignature:
    """Signature for detecting a specific bottleneck type"""
    bottleneck_type: BottleneckType
    indicators: List[BottleneckIndicator]
    min_indicators: int = 2
    severity: BottleneckSeverity = BottleneckSeverity.MEDIUM
    description: str = ""
    auto_remediation: Optional[str] = None


@dataclass
class DetectedBottleneck:
    """Represents a detected bottleneck"""
    bottleneck_type: BottleneckType
    severity: BottleneckSeverity
    detected_at: float
    matched_indicators: List[str]
    metrics_snapshot: Dict[str, float]
    confidence: float
    remediation_attempted: bool = False
    remediation_result: Optional[str] = None
    resolved_at: Optional[float] = None


class BottleneckDetector:
    """
    Real-time bottleneck detection system
    
    Features:
    - Multiple bottleneck signatures
    - Confidence scoring
    - Automatic remediation
    - Historical tracking
    """
    
    # Predefined bottleneck signatures
    DEFAULT_SIGNATURES = [
        BottleneckSignature(
            bottleneck_type=BottleneckType.GPT52_THROTTLING,
            indicators=[
                BottleneckIndicator(
                    "high_response_time", "gpt_response_time_ms", 5000, "gt", 1.0
                ),
                BottleneckIndicator(
                    "low_token_rate", "gpt_tokens_per_min", 500, "lt", 0.8
                ),
                BottleneckIndicator(
                    "gpt_queue_backlog", "gpt_queue_depth", 20, "gt", 0.9
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.CRITICAL,
            description="GPT-5.2 API is throttling or experiencing high latency",
            auto_remediation="scale_out_gpt_workers"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.MEMORY_PRESSURE,
            indicators=[
                BottleneckIndicator(
                    "high_memory", "memory_percent", 90, "gt", 1.0
                ),
                BottleneckIndicator(
                    "swap_usage", "swap_percent", 50, "gt", 0.8
                ),
                BottleneckIndicator(
                    "frequent_gc", "gc_frequency_per_min", 10, "gt", 0.7
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.CRITICAL,
            description="System is experiencing memory pressure",
            auto_remediation="scale_out_and_restart_heavy"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.CPU_SATURATION,
            indicators=[
                BottleneckIndicator(
                    "high_cpu", "cpu_percent", 95, "gt", 1.0
                ),
                BottleneckIndicator(
                    "high_load", "load_avg", 8, "gt", 0.9
                ),
                BottleneckIndicator(
                    "context_switches", "context_switches_per_sec", 50000, "gt", 0.7
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.HIGH,
            description="CPU is saturated",
            auto_remediation="scale_out_instances"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.NETWORK_CONGESTION,
            indicators=[
                BottleneckIndicator(
                    "high_latency", "network_latency_ms", 200, "gt", 1.0
                ),
                BottleneckIndicator(
                    "packet_loss", "packet_loss_percent", 1, "gt", 0.9
                ),
                BottleneckIndicator(
                    "bandwidth_util", "bandwidth_percent", 90, "gt", 0.8
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.HIGH,
            description="Network is experiencing congestion",
            auto_remediation="enable_compression"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.DISK_IO_BOTTLENECK,
            indicators=[
                BottleneckIndicator(
                    "high_disk_queue", "disk_queue_depth", 10, "gt", 1.0
                ),
                BottleneckIndicator(
                    "disk_utilization", "disk_utilization_percent", 90, "gt", 0.9
                ),
                BottleneckIndicator(
                    "io_wait", "io_wait_percent", 20, "gt", 0.8
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.MEDIUM,
            description="Disk I/O is bottlenecking",
            auto_remediation="enable_async_writes"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.QUEUE_BACKLOG,
            indicators=[
                BottleneckIndicator(
                    "queue_depth", "queue_depth", 100, "gt", 1.0
                ),
                BottleneckIndicator(
                    "processing_lag", "processing_lag_seconds", 30, "gt", 0.9
                ),
                BottleneckIndicator(
                    "slow_consumers", "consumer_lag", 50, "gt", 0.8
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.HIGH,
            description="Message queue is backing up",
            auto_remediation="scale_out_workers"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.DATABASE_SLOWDOWN,
            indicators=[
                BottleneckIndicator(
                    "slow_queries", "slow_query_count", 10, "gt", 1.0
                ),
                BottleneckIndicator(
                    "db_connections", "db_connection_count", 80, "gt", 0.9
                ),
                BottleneckIndicator(
                    "query_time", "avg_query_time_ms", 500, "gt", 0.8
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.MEDIUM,
            description="Database is experiencing slowdown",
            auto_remediation="enable_query_caching"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.CACHE_INEFFICIENCY,
            indicators=[
                BottleneckIndicator(
                    "low_hit_rate", "cache_hit_rate", 50, "lt", 1.0
                ),
                BottleneckIndicator(
                    "high_eviction", "cache_eviction_rate", 100, "gt", 0.8
                ),
                BottleneckIndicator(
                    "memory_fragmentation", "memory_fragmentation", 30, "gt", 0.7
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.LOW,
            description="Cache is not being used efficiently",
            auto_remediation="resize_cache"
        ),
        
        BottleneckSignature(
            bottleneck_type=BottleneckType.BROWSER_POOL_EXHAUSTION,
            indicators=[
                BottleneckIndicator(
                    "pool_utilization", "browser_pool_utilization", 90, "gt", 1.0
                ),
                BottleneckIndicator(
                    "browser_wait_time", "browser_wait_time_ms", 5000, "gt", 0.9
                ),
                BottleneckIndicator(
                    "browser_crashes", "browser_crashes_per_min", 5, "gt", 0.8
                ),
            ],
            min_indicators=2,
            severity=BottleneckSeverity.MEDIUM,
            description="Browser automation pool is exhausted",
            auto_remediation="expand_browser_pool"
        ),
    ]
    
    def __init__(self, 
                 check_interval: int = 30,
                 history_size: int = 1000):
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Signatures
        self.signatures: List[BottleneckSignature] = list(self.DEFAULT_SIGNATURES)
        
        # Detected bottlenecks
        self.active_bottlenecks: Dict[BottleneckType, DetectedBottleneck] = {}
        self.bottleneck_history: deque = deque(maxlen=history_size)
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        
        # Remediation handlers
        self.remediation_handlers: Dict[str, Callable] = {}
        
        # Callbacks
        self.on_bottleneck_detected: Optional[Callable] = None
        self.on_bottleneck_resolved: Optional[Callable] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._detection_thread: Optional[threading.Thread] = None
        
        # Lock
        self._lock = threading.RLock()
        
        logger.info("BottleneckDetector initialized")
    
    def register_remediation_handler(self, 
                                     action: str, 
                                     handler: Callable) -> None:
        """Register a handler for automatic remediation"""
        self.remediation_handlers[action] = handler
        logger.info(f"Registered remediation handler: {action}")
    
    def add_signature(self, signature: BottleneckSignature) -> None:
        """Add a custom bottleneck signature"""
        self.signatures.append(signature)
        logger.info(f"Added bottleneck signature: {signature.bottleneck_type.value}")
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        import psutil
        
        import gc

        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'swap_percent': psutil.swap_memory().percent,
            'disk_queue_depth': 0,  # Platform-specific
            'disk_utilization_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            'io_wait_percent': 0,  # Platform-specific
            'network_latency_ms': 0,  # To be measured
            'packet_loss_percent': 0,  # To be measured
            'bandwidth_percent': 0,  # To be measured
            'load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'context_switches_per_sec': 0,  # Platform-specific
        }

        # Queue metrics
        metrics['queue_depth'] = len(getattr(self, 'task_queue', []))
        metrics['processing_lag_seconds'] = time.time() - getattr(self, '_oldest_pending_ts', time.time())
        metrics['consumer_lag'] = max(0, getattr(self, '_pending_count', 0) - getattr(self, '_processed_count', 0))

        # LLM metrics - from MetricsCollector
        mc = self._metrics_collector if hasattr(self, '_metrics_collector') else None
        metrics['gpt_response_time_ms'] = mc.get_latest('gpt_response_time') if mc else 0.0
        metrics['gpt_tokens_per_min'] = mc.get_latest('gpt_tokens_per_min') if mc else 0.0
        metrics['gpt_queue_depth'] = len(getattr(self, '_pending_llm_requests', []))

        # DB metrics
        metrics['slow_query_count'] = mc.get_latest('slow_query_count') if mc else 0
        metrics['db_connection_count'] = mc.get_latest('db_connections') if mc else 0
        metrics['avg_query_time_ms'] = mc.get_latest('avg_query_time_ms') if mc else 0.0

        # Cache metrics
        hits = mc.get_latest('cache_hits') if mc else 0
        misses = mc.get_latest('cache_misses') if mc else 0
        metrics['cache_hit_rate'] = hits / max(hits + misses, 1)
        metrics['cache_eviction_rate'] = mc.get_latest('cache_evictions') if mc else 0.0

        # System metrics via psutil
        vm = psutil.virtual_memory()
        metrics['memory_fragmentation'] = 1.0 - (vm.available / max(vm.total, 1))

        # Browser metrics
        metrics['browser_pool_utilization'] = mc.get_latest('browser_active_pages') / max(mc.get_latest('browser_max_pages') if mc else 1, 1) if mc else 0.0
        metrics['browser_wait_time_ms'] = mc.get_latest('browser_wait_time') if mc else 0.0
        metrics['browser_crashes_per_min'] = mc.get_latest('browser_crashes_per_min') if mc else 0.0

        # GC metrics
        gc_stats = gc.get_stats()
        metrics['gc_frequency_per_min'] = sum(s.get('collections', 0) for s in gc_stats) if gc_stats else 0.0
        
        with self._lock:
            self.metrics_history.append(metrics)
        
        return metrics
    
    def evaluate_indicator(self, 
                          indicator: BottleneckIndicator, 
                          metrics: Dict[str, float]) -> bool:
        """Evaluate if an indicator condition is met"""
        value = metrics.get(indicator.metric)
        if value is None:
            return False
        
        if indicator.comparison == "gt":
            return value > indicator.threshold
        elif indicator.comparison == "lt":
            return value < indicator.threshold
        elif indicator.comparison == "gte":
            return value >= indicator.threshold
        elif indicator.comparison == "lte":
            return value <= indicator.threshold
        
        return False
    
    def detect_bottlenecks(self, metrics: Dict[str, float]) -> List[DetectedBottleneck]:
        """Detect bottlenecks from current metrics"""
        detected = []
        
        for signature in self.signatures:
            matched_indicators = []
            total_weight = 0
            matched_weight = 0
            
            for indicator in signature.indicators:
                total_weight += indicator.weight
                if self.evaluate_indicator(indicator, metrics):
                    matched_indicators.append(indicator.name)
                    matched_weight += indicator.weight
            
            # Check if enough indicators matched
            if len(matched_indicators) >= signature.min_indicators:
                # Calculate confidence
                confidence = matched_weight / total_weight if total_weight > 0 else 0
                
                bottleneck = DetectedBottleneck(
                    bottleneck_type=signature.bottleneck_type,
                    severity=signature.severity,
                    detected_at=time.time(),
                    matched_indicators=matched_indicators,
                    metrics_snapshot={k: v for k, v in metrics.items() 
                                     if isinstance(v, (int, float))},
                    confidence=confidence
                )
                
                detected.append(bottleneck)
        
        return detected
    
    def update_active_bottlenecks(self, detected: List[DetectedBottleneck]) -> None:
        """Update the list of active bottlenecks"""
        with self._lock:
            current_time = time.time()
            
            # Mark resolved bottlenecks
            detected_types = {b.bottleneck_type for b in detected}
            for btype, bottleneck in list(self.active_bottlenecks.items()):
                if btype not in detected_types:
                    bottleneck.resolved_at = current_time
                    self.bottleneck_history.append(bottleneck)
                    del self.active_bottlenecks[btype]
                    
                    if self.on_bottleneck_resolved:
                        self.on_bottleneck_resolved(bottleneck)
                    
                    logger.info(f"Bottleneck resolved: {btype.value}")
            
            # Add new bottlenecks
            for bottleneck in detected:
                if bottleneck.bottleneck_type not in self.active_bottlenecks:
                    self.active_bottlenecks[bottleneck.bottleneck_type] = bottleneck
                    
                    if self.on_bottleneck_detected:
                        self.on_bottleneck_detected(bottleneck)
                    
                    logger.warning(
                        f"Bottleneck detected: {bottleneck.bottleneck_type.value} "
                        f"(severity: {bottleneck.severity.value}, "
                        f"confidence: {bottleneck.confidence:.2f})"
                    )
                    
                    # Attempt auto-remediation
                    self.attempt_remediation(bottleneck)
    
    def attempt_remediation(self, bottleneck: DetectedBottleneck) -> None:
        """Attempt automatic remediation for a bottleneck"""
        # Find signature for this bottleneck type
        signature = None
        for sig in self.signatures:
            if sig.bottleneck_type == bottleneck.bottleneck_type:
                signature = sig
                break
        
        if not signature or not signature.auto_remediation:
            return
        
        handler = self.remediation_handlers.get(signature.auto_remediation)
        if not handler:
            logger.warning(
                f"No handler for remediation: {signature.auto_remediation}"
            )
            return
        
        try:
            logger.info(
                f"Attempting remediation: {signature.auto_remediation} "
                f"for {bottleneck.bottleneck_type.value}"
            )
            
            result = handler(bottleneck)
            
            bottleneck.remediation_attempted = True
            bottleneck.remediation_result = str(result)
            
            logger.info(f"Remediation result: {result}")
            
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Remediation failed: {e}")
            bottleneck.remediation_result = f"failed: {str(e)}"
    
    def detection_loop(self) -> None:
        """Main detection loop"""
        logger.info("Bottleneck detection loop started")
        
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Detect bottlenecks
                detected = self.detect_bottlenecks(metrics)
                
                # Update active bottlenecks
                self.update_active_bottlenecks(detected)
                
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(f"Error in detection loop: {e}")
            
            # Wait for next iteration
            self._stop_event.wait(self.check_interval)
        
        logger.info("Bottleneck detection loop stopped")
    
    def start(self) -> None:
        """Start bottleneck detection"""
        self._detection_thread = threading.Thread(
            target=self.detection_loop,
            daemon=True
        )
        self._detection_thread.start()
        logger.info("Bottleneck detection started")
    
    def stop(self) -> None:
        """Stop bottleneck detection"""
        self._stop_event.set()
        if self._detection_thread:
            self._detection_thread.join(timeout=10)
        logger.info("Bottleneck detection stopped")
    
    def get_active_bottlenecks(self) -> List[DetectedBottleneck]:
        """Get list of currently active bottlenecks"""
        with self._lock:
            return list(self.active_bottlenecks.values())
    
    def get_bottleneck_history(self, 
                               hours: int = 24) -> List[DetectedBottleneck]:
        """Get bottleneck history for last N hours"""
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            return [
                b for b in self.bottleneck_history
                if b.detected_at >= cutoff
            ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of bottleneck detection status"""
        with self._lock:
            active = self.get_active_bottlenecks()
            history_24h = self.get_bottleneck_history(24)
            
            return {
                "active_bottlenecks": len(active),
                "active_by_severity": {
                    severity.value: len([
                        b for b in active 
                        if b.severity == severity
                    ])
                    for severity in BottleneckSeverity
                },
                "bottlenecks_24h": len(history_24h),
                "most_common_24h": self._get_most_common(history_24h),
                "signatures_configured": len(self.signatures),
                "remediation_handlers": len(self.remediation_handlers)
            }
    
    def _get_most_common(self, 
                        bottlenecks: List[DetectedBottleneck],
                        top_n: int = 3) -> List[Dict]:
        """Get most common bottleneck types"""
        counts = defaultdict(int)
        for b in bottlenecks:
            counts[b.bottleneck_type.value] += 1
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"type": btype, "count": count}
            for btype, count in sorted_counts[:top_n]
        ]


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = BottleneckDetector(check_interval=10)
    
    # Register remediation handlers
    def scale_out_gpt_workers(bottleneck: DetectedBottleneck) -> str:
        print(f"Scaling out GPT workers for {bottleneck.bottleneck_type.value}")
        return "scaled_out_2_workers"
    
    def scale_out_instances(bottleneck: DetectedBottleneck) -> str:
        print(f"Scaling out instances for {bottleneck.bottleneck_type.value}")
        return "scaled_out_1_instance"
    
    detector.register_remediation_handler(
        "scale_out_gpt_workers", scale_out_gpt_workers
    )
    detector.register_remediation_handler(
        "scale_out_instances", scale_out_instances
    )
    
    # Set callbacks
    def on_detected(bottleneck: DetectedBottleneck):
        print(f"ALERT: Bottleneck detected - {bottleneck.bottleneck_type.value}")
    
    def on_resolved(bottleneck: DetectedBottleneck):
        print(f"RESOLVED: {bottleneck.bottleneck_type.value}")
    
    detector.on_bottleneck_detected = on_detected
    detector.on_bottleneck_resolved = on_resolved
    
    # Start detection
    detector.start()
    
    # Run for demonstration
    try:
        time.sleep(60)
    finally:
        detector.stop()
        print("\nSummary:", detector.get_summary())
