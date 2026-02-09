"""
Ralph Loop Implementation
Multi-Layered Background Processing with Priority Queuing
OpenClaw Windows 10 AI Agent System

This module provides the core implementation of the Advanced Ralph Loop architecture.
"""

import asyncio
import heapq
import sqlite3
import psutil
import uuid
import json
import logging
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Deque
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading
import time

logger = logging.getLogger("RalphLoop")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PriorityLevel(IntEnum):
    """256-level priority system (0-255) - Lower is higher priority"""
    # Emergency Priorities (0-15)
    P0_SYSTEM_CRITICAL = 0
    P1_SECURITY_EMERGENCY = 1
    P2_DATA_CORRUPTION = 2
    P3_MEMORY_CRITICAL = 3
    
    # Real-Time Priorities (16-31)
    P16_VOICE_STREAM = 16
    P17_STT_REALTIME = 17
    P18_TTS_STREAM = 18
    P19_TWILIO_ACTIVE = 19
    P20_USER_INPUT = 20
    
    # High User-Facing (32-63)
    P32_GMAIL_URGENT = 32
    P33_BROWSER_NAV = 33
    P34_API_CRITICAL = 34
    P35_NOTIFICATION = 35
    P40_USER_COMMAND = 40
    
    # Standard Operations (64-127)
    P64_AGENT_LOOP = 64
    P65_DATA_PROCESS = 65
    P70_FILE_OPS = 70
    P80_SCHEDULED = 80
    P90_CACHE_MGMT = 90
    
    # Background (128-191)
    P128_LOG_ROTATION = 128
    P130_CLEANUP = 130
    P140_MAINTENANCE = 140
    P150_SYNC = 150
    
    # Batch Processing (192-239)
    P192_BULK_EMAIL = 192
    P195_REPORT_GEN = 195
    P200_DATA_EXPORT = 200
    P210_ML_TRAINING = 210
    
    # Archival (240-255)
    P240_CONVERSATION_ARCH = 240
    P245_LONG_TERM_STORE = 245
    P250_COLD_STORAGE = 250
    P255_BEST_EFFORT = 255


class JobStatus(IntEnum):
    """Job execution status"""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class PreemptionReason(IntEnum):
    """Reasons for task preemption"""
    HIGHER_PRIORITY = auto()
    RESOURCE_CONSTRAINT = auto()
    TIMEOUT = auto()
    SYSTEM_SHUTDOWN = auto()
    EMERGENCY = auto()


class HealthStatus(IntEnum):
    """System health status"""
    HEALTHY = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class PressureLevel(IntEnum):
    """System pressure levels"""
    NORMAL = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class CheckpointType(IntEnum):
    """Checkpoint types"""
    FULL = auto()
    INCREMENTAL = auto()
    DELTA = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Task:
    """Represents a unit of work"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    handler: Optional[Callable] = None
    priority: int = PriorityLevel.P64_AGENT_LOOP
    layer: int = 3
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    cpu_estimate: Optional[float] = None
    memory_estimate: Optional[int] = None
    io_estimate: Optional[int] = None
    time_limit: Optional[timedelta] = None
    data: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    preemption_count: int = 0
    is_io_bound: bool = False
    is_cpu_intensive: bool = False
    requires_network: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    priority_override: Optional[int] = None


@dataclass(order=True)
class PrioritizedTask:
    """Task wrapper for priority queue"""
    priority: int
    sequence: int
    created_at: datetime
    task: Task = field(compare=False)
    preemption_count: int = 0
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    
    def __post_init__(self):
        self.effective_priority = self._calculate_effective_priority()
    
    def _calculate_effective_priority(self) -> float:
        """Calculate dynamic priority based on age and deadline"""
        base = float(self.priority)
        
        # Age boost
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        age_boost = min(age_seconds / 60, 10)
        
        # Deadline urgency
        deadline_boost = 0
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.now()).total_seconds()
            if time_to_deadline < 0:
                deadline_boost = 50
            elif time_to_deadline < 10:
                deadline_boost = 30
            elif time_to_deadline < 60:
                deadline_boost = 15
        
        # Preemption boost
        preemption_boost = self.preemption_count * 2
        
        return base - age_boost - deadline_boost - preemption_boost


@dataclass
class Job:
    """Represents a background job"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: str = ""
    handler: Optional[Callable] = None
    status: JobStatus = JobStatus.PENDING
    priority: int = PriorityLevel.P64_AGENT_LOOP
    layer: int = 3
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    cron_job_id: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    
    def to_task(self) -> Task:
        """Convert job to task"""
        return Task(
            id=self.id,
            task_type=self.task_type,
            handler=self.handler,
            priority=self.priority,
            layer=self.layer
        )
    
    def clone(self) -> 'Job':
        """Create a copy of the job"""
        return Job(
            name=self.name,
            task_type=self.task_type,
            handler=self.handler,
            priority=self.priority,
            layer=self.layer
        )


@dataclass
class ResourceGrant:
    """Resource allocation grant"""
    cpu_cores: float
    memory_mb: int
    io_priority: int
    time_limit: Optional[timedelta] = None


@dataclass
class Resources:
    """Resource availability"""
    cpu: float
    memory: int
    io_bandwidth: int


@dataclass
class Context:
    """Execution context for task"""
    task_id: str
    registers: Dict[str, Any]
    stack: List[Any]
    heap_state: Dict[str, Any]
    file_descriptors: List[int]
    timestamp: datetime


@dataclass
class Checkpoint:
    """Task checkpoint for resumption"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    task_type: str = ""
    checkpoint_data: bytes = b""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PreemptedTask:
    """Preempted task state"""
    task: Task
    checkpoint: Checkpoint
    preempted_at: datetime
    reason: PreemptionReason
    original_priority: int


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics"""
    timestamp: datetime
    gauges: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    histograms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """System health report"""
    timestamp: datetime
    overall: HealthStatus = HealthStatus.HEALTHY
    checks: Dict[str, HealthCheckResult] = field(default_factory=dict)


@dataclass
class RecoveryReport:
    """Recovery operation report"""
    tasks_loaded: int = 0
    tasks_queued: int = 0
    tasks_resumed: int = 0
    inconsistencies_found: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class Bottleneck:
    """Performance bottleneck"""
    type: str
    layer: int
    severity: float
    recommendation: str


@dataclass
class Alert:
    """System alert"""
    rule: 'AlertRule'
    metrics: MetricsSnapshot
    timestamp: datetime


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    condition: str
    level: str
    channels: List[str]
    suppress_duration: timedelta
    
    def evaluate(self, metrics: MetricsSnapshot) -> bool:
        """Evaluate rule condition against current metrics using safe_eval."""
        from safe_eval import safe_eval

        # Build evaluation context from metrics
        context = {
            **metrics.gauges,
            **{k: float(v) for k, v in metrics.counters.items()},
            'timestamp': metrics.timestamp.timestamp() if metrics.timestamp else 0,
        }

        try:
            result = safe_eval(self.condition, context)
            return bool(result)
        except (NameError, TypeError, ValueError, SyntaxError, AttributeError) as e:
            logger.debug(f"AlertRule '{self.name}' evaluation failed: {e}")
            return False


@dataclass
class Inconsistency:
    """Data inconsistency"""
    type: str
    details: str


# =============================================================================
# EXCEPTIONS
# =============================================================================

class RalphLoopError(Exception):
    """Base exception for Ralph Loop"""
    pass


class ResourceUnavailable(RalphLoopError):
    """Raised when resources are unavailable"""
    pass


class PreemptionError(RalphLoopError):
    """Raised when preemption fails"""
    pass


class ResumptionError(RalphLoopError):
    """Raised when resumption fails"""
    pass


class CyclicDependencyError(RalphLoopError):
    """Raised when dependency cycle detected"""
    pass


class LogCorruptionError(RalphLoopError):
    """Raised when log corruption detected"""
    pass


# =============================================================================
# LAYER IMPLEMENTATIONS
# =============================================================================

class ProcessingLayer(ABC):
    """Abstract base class for processing layers"""
    
    def __init__(self, layer_id: int, config: Dict[str, Any]):
        self.layer_id = layer_id
        self.config = config
        self.active_tasks: Dict[str, Task] = {}
        self.task_count = 0
        self.total_executed = 0
        self.lock = asyncio.Lock()
        
    @abstractmethod
    async def execute(self, task: Task) -> Any:
        """Execute a task in this layer"""
        pass
    
    def has_capacity(self) -> bool:
        """Check if layer has capacity for more tasks"""
        max_concurrent = self.config.get('max_concurrent', 100)
        return len(self.active_tasks) < max_concurrent
    
    def get_current_load(self) -> float:
        """Get current load as ratio of capacity"""
        max_concurrent = self.config.get('max_concurrent', 100)
        return len(self.active_tasks) / max_concurrent if max_concurrent > 0 else 0
    
    async def add_task(self, task: Task):
        """Add task to active tasks"""
        async with self.lock:
            self.active_tasks[task.id] = task
            self.task_count += 1
    
    async def remove_task(self, task_id: str):
        """Remove task from active tasks"""
        async with self.lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                self.total_executed += 1


class CriticalLayer(ProcessingLayer):
    """Layer 0: Critical system operations"""
    
    LATENCY_TARGET = 0.001  # 1ms
    CPU_ALLOCATION = 0.40
    MEMORY_ALLOCATION = 512
    
    async def execute(self, task: Task) -> Any:
        """Execute critical task with highest priority"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await asyncio.wait_for(
                    task.handler(task.data),
                    timeout=0.005  # 5ms timeout for critical tasks
                )
                return result
        finally:
            await self.remove_task(task.id)


class RealTimeLayer(ProcessingLayer):
    """Layer 1: Real-time operations"""
    
    LATENCY_TARGET = 0.010  # 10ms
    CPU_ALLOCATION = 0.25
    MEMORY_ALLOCATION = 1024
    
    async def execute(self, task: Task) -> Any:
        """Execute real-time task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await asyncio.wait_for(
                    task.handler(task.data),
                    timeout=0.050  # 50ms timeout
                )
                return result
        finally:
            await self.remove_task(task.id)


class HighPriorityLayer(ProcessingLayer):
    """Layer 2: High priority user-facing operations"""
    
    LATENCY_TARGET = 0.100  # 100ms
    CPU_ALLOCATION = 0.20
    MEMORY_ALLOCATION = 2048
    
    async def execute(self, task: Task) -> Any:
        """Execute high priority task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await asyncio.wait_for(
                    task.handler(task.data),
                    timeout=0.500  # 500ms timeout
                )
                return result
        finally:
            await self.remove_task(task.id)


class StandardLayer(ProcessingLayer):
    """Layer 3: Standard operations"""
    
    LATENCY_TARGET = 1.0  # 1 second
    CPU_ALLOCATION = 0.10
    MEMORY_ALLOCATION = 4096
    
    async def execute(self, task: Task) -> Any:
        """Execute standard task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await task.handler(task.data)
                return result
        finally:
            await self.remove_task(task.id)


class BackgroundLayer(ProcessingLayer):
    """Layer 4: Background operations"""
    
    LATENCY_TARGET = 10.0  # 10 seconds
    CPU_ALLOCATION = 0.03
    MEMORY_ALLOCATION = 1024
    
    async def execute(self, task: Task) -> Any:
        """Execute background task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await task.handler(task.data)
                return result
        finally:
            await self.remove_task(task.id)


class BatchLayer(ProcessingLayer):
    """Layer 5: Batch processing"""
    
    LATENCY_TARGET = 300.0  # 5 minutes
    CPU_ALLOCATION = 0.015
    MEMORY_ALLOCATION = 2048
    
    async def execute(self, task: Task) -> Any:
        """Execute batch task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await task.handler(task.data)
                return result
        finally:
            await self.remove_task(task.id)


class ArchivalLayer(ProcessingLayer):
    """Layer 6: Archival operations"""
    
    LATENCY_TARGET = float('inf')  # Best effort
    CPU_ALLOCATION = 0.005
    MEMORY_ALLOCATION = 512
    
    async def execute(self, task: Task) -> Any:
        """Execute archival task"""
        await self.add_task(task)
        try:
            task.started_at = datetime.now()
            if task.handler:
                result = await task.handler(task.data)
                return result
        finally:
            await self.remove_task(task.id)


# =============================================================================
# LAYER MANAGER
# =============================================================================

class RalphLayerManager:
    """Manages all processing layers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.layers: Dict[int, ProcessingLayer] = {}
        self.task_type_mapping: Dict[str, tuple] = {}
        self._init_layers()
        
    def _init_layers(self):
        """Initialize all layers"""
        layer_configs = self.config.get('layers', {})
        
        self.layers[0] = CriticalLayer(0, layer_configs.get('critical', {}))
        self.layers[1] = RealTimeLayer(1, layer_configs.get('real_time', {}))
        self.layers[2] = HighPriorityLayer(2, layer_configs.get('high_priority', {}))
        self.layers[3] = StandardLayer(3, layer_configs.get('standard', {}))
        self.layers[4] = BackgroundLayer(4, layer_configs.get('background', {}))
        self.layers[5] = BatchLayer(5, layer_configs.get('batch', {}))
        self.layers[6] = ArchivalLayer(6, layer_configs.get('archival', {}))
    
    def register_task_type(self, task_type: str, layer: int, priority: int):
        """Register a task type with its default layer and priority"""
        self.task_type_mapping[task_type] = (layer, priority)
    
    def route_task(self, task: Task) -> int:
        """Route task to appropriate layer"""
        if task.task_type in self.task_type_mapping:
            layer, priority = self.task_type_mapping[task.task_type]
            task.layer = layer
            task.priority = priority
            return layer
        return task.layer
    
    def get_layer(self, layer_id: int) -> ProcessingLayer:
        """Get layer by ID"""
        return self.layers[layer_id]
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task in appropriate layer"""
        layer_id = self.route_task(task)
        layer = self.layers[layer_id]
        return await layer.execute(task)


# =============================================================================
# PRIORITY QUEUE
# =============================================================================

class RalphPriorityQueue:
    """Multi-layer priority queue"""
    
    def __init__(self, layer_manager: RalphLayerManager):
        self.layer_manager = layer_manager
        self.queues: Dict[int, List[PrioritizedTask]] = {i: [] for i in range(7)}
        self.sequence_counter = 0
        self.task_index: Dict[str, tuple] = {}
        self.lock = asyncio.Lock()
        self.metrics = {
            'enqueues': 0,
            'dequeues': 0,
            'reprioritizations': 0
        }
        
    async def enqueue(self, task: Task, priority: Optional[int] = None) -> str:
        """Add task to priority queue"""
        async with self.lock:
            if priority is None:
                priority = task.priority
            
            layer = self.layer_manager.route_task(task)
            
            self.sequence_counter += 1
            ptask = PrioritizedTask(
                priority=priority,
                sequence=self.sequence_counter,
                created_at=datetime.now(),
                task=task,
                deadline=task.deadline,
                estimated_duration=task.estimated_duration
            )
            
            heapq.heappush(self.queues[layer], ptask)
            self.task_index[task.id] = (layer, ptask)
            self.metrics['enqueues'] += 1
            
            return task.id
    
    async def dequeue(self, layer: Optional[int] = None) -> Optional[Task]:
        """Get highest priority task"""
        async with self.lock:
            if layer is not None:
                return self._dequeue_from_layer(layer)
            
            for layer_id in range(7):
                task = self._dequeue_from_layer(layer_id)
                if task is not None:
                    self.metrics['dequeues'] += 1
                    return task
            
            return None
    
    def _dequeue_from_layer(self, layer: int) -> Optional[Task]:
        """Dequeue from specific layer"""
        if self.queues[layer]:
            ptask = heapq.heappop(self.queues[layer])
            if ptask.task.id in self.task_index:
                del self.task_index[ptask.task.id]
            return ptask.task
        return None
    
    async def reprioritize(self, task_id: str, new_priority: int) -> bool:
        """Change priority of existing task"""
        async with self.lock:
            if task_id not in self.task_index:
                return False
            
            layer, old_ptask = self.task_index[task_id]
            
            # Remove and re-insert
            self.queues[layer] = [pt for pt in self.queues[layer] if pt.task.id != task_id]
            heapq.heapify(self.queues[layer])
            
            old_ptask.priority = new_priority
            old_ptask.sequence = self.sequence_counter + 1
            heapq.heappush(self.queues[layer], old_ptask)
            self.task_index[task_id] = (layer, old_ptask)
            
            self.metrics['reprioritizations'] += 1
            return True
    
    async def peek(self, layer: int, count: int = 1) -> List[PrioritizedTask]:
        """Peek at top tasks without removing"""
        async with self.lock:
            return heapq.nsmallest(count, self.queues[layer])
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "total_tasks": sum(len(q) for q in self.queues.values()),
            "layer_distribution": {i: len(q) for i, q in self.queues.items()},
            "metrics": self.metrics.copy()
        }


# =============================================================================
# RESOURCE ALLOCATOR
# =============================================================================

class ResourceAllocator:
    """Manages resource allocation with real system metrics via psutil."""

    def __init__(self):
        self.total_cpu = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.allocations: Dict[str, ResourceGrant] = {}
        # Historical resource usage per task_type for better estimation
        self._resource_history: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {'cpu': [], 'memory': [], 'io': []}
        )
        self._history_max = 50  # Keep last N measurements per task type
        # Snapshot IO counters at init for delta calculations
        try:
            self._last_io_counters = psutil.disk_io_counters()
        except (AttributeError, RuntimeError):
            self._last_io_counters = None

    def _get_available_resources(self) -> Resources:
        """Get currently available resources using psutil."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except (RuntimeError, OSError):
            cpu_percent = 50.0  # Fallback

        try:
            mem = psutil.virtual_memory()
            available_mb = int(mem.available / (1024 * 1024))
        except (RuntimeError, OSError):
            available_mb = 1024  # Fallback 1GB

        try:
            io = psutil.disk_io_counters()
            if io and self._last_io_counters:
                io_ops = (io.read_count + io.write_count) - (
                    self._last_io_counters.read_count + self._last_io_counters.write_count
                )
                self._last_io_counters = io
                io_bandwidth = max(1000 - io_ops, 100)
            else:
                io_bandwidth = 1000
        except (AttributeError, RuntimeError, OSError):
            io_bandwidth = 1000

        return Resources(
            cpu=self.total_cpu * (1 - cpu_percent / 100),
            memory=available_mb,
            io_bandwidth=io_bandwidth
        )

    def record_task_usage(self, task: Task, cpu: float, memory: int, io: int) -> None:
        """Record actual resource usage for a completed task to improve future estimates."""
        history = self._resource_history[task.task_type]
        for key, val in [('cpu', cpu), ('memory', memory), ('io', io)]:
            history[key].append(float(val))
            if len(history[key]) > self._history_max:
                history[key] = history[key][-self._history_max:]

    def _estimate_cpu(self, task: Task) -> float:
        """Estimate CPU requirement using explicit value, historical average, or psutil."""
        if task.cpu_estimate:
            return task.cpu_estimate
        # Use historical average for this task type
        hist = self._resource_history[task.task_type]['cpu']
        if hist:
            return sum(hist) / len(hist)
        # Fallback: use current per-process CPU as a baseline
        try:
            proc = psutil.Process()
            pct = proc.cpu_percent(interval=0.05)
            return max(pct / 100.0, 0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, RuntimeError, OSError):
            return 0.1

    def _estimate_memory(self, task: Task) -> int:
        """Estimate memory requirement using explicit value, historical average, or psutil."""
        if task.memory_estimate:
            return task.memory_estimate
        hist = self._resource_history[task.task_type]['memory']
        if hist:
            return int(sum(hist) / len(hist))
        # Fallback: use current process RSS as a baseline
        try:
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            return max(int(rss_mb * 0.1), 32)  # ~10% of current RSS, min 32MB
        except (psutil.NoSuchProcess, psutil.AccessDenied, RuntimeError, OSError):
            return 128

    def _estimate_io(self, task: Task) -> int:
        """Estimate I/O requirement using explicit value, historical average, or psutil."""
        if task.io_estimate:
            return task.io_estimate
        hist = self._resource_history[task.task_type]['io']
        if hist:
            return int(sum(hist) / len(hist))
        # Fallback: use disk IO counters to gauge current load
        try:
            io = psutil.disk_io_counters()
            if io:
                return max(int((io.read_count + io.write_count) / 1000), 10)
        except (AttributeError, RuntimeError, OSError):
            pass
        return 10
    
    def _can_allocate(self, available: Resources, cpu: float, 
                      mem: int, io: int) -> bool:
        """Check if resources can be allocated"""
        return (
            available.cpu >= cpu and
            available.memory >= mem and
            available.io_bandwidth >= io
        )
    
    async def allocate(self, task: Task) -> ResourceGrant:
        """Allocate resources for task"""
        cpu_req = self._estimate_cpu(task)
        mem_req = self._estimate_memory(task)
        io_req = self._estimate_io(task)
        
        available = self._get_available_resources()
        
        if self._can_allocate(available, cpu_req, mem_req, io_req):
            grant = ResourceGrant(
                cpu_cores=cpu_req,
                memory_mb=mem_req,
                io_priority=task.layer,
                time_limit=task.time_limit
            )
            self.allocations[task.id] = grant
            return grant
        
        raise ResourceUnavailable(f"Insufficient resources for task {task.id}")
    
    async def release(self, task_id: str):
        """Release allocated resources"""
        if task_id in self.allocations:
            del self.allocations[task_id]


# =============================================================================
# JOB ORCHESTRATOR
# =============================================================================

class JobOrchestrator:
    """Orchestrates background job execution"""
    
    def __init__(self, layer_manager: RalphLayerManager,
                 priority_queue: RalphPriorityQueue,
                 resource_allocator: ResourceAllocator):
        self.layer_manager = layer_manager
        self.priority_queue = priority_queue
        self.resource_allocator = resource_allocator
        self.jobs: Dict[str, Job] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.running = False
        
    async def submit_job(self, job: Job, dependencies: Optional[List[str]] = None) -> str:
        """Submit job for execution"""
        job.submitted_at = datetime.now()
        job.status = JobStatus.PENDING
        
        self.jobs[job.id] = job
        
        if dependencies:
            self.dependencies[job.id] = set(dependencies)
            for dep in dependencies:
                self.dependents[dep].add(job.id)
        
        if self._is_ready(job.id):
            await self._enqueue_job(job)
        
        return job.id
    
    def _is_ready(self, job_id: str) -> bool:
        """Check if job dependencies are satisfied"""
        for dep in self.dependencies[job_id]:
            if dep not in self.jobs:
                return False
            if self.jobs[dep].status != JobStatus.COMPLETED:
                return False
        return True
    
    async def _enqueue_job(self, job: Job):
        """Enqueue job for execution"""
        task = job.to_task()
        await self.priority_queue.enqueue(task, job.priority)
        job.status = JobStatus.QUEUED
    
    async def job_completed(self, job_id: str, result: Any):
        """Handle job completion"""
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.result = result
        
        # Notify dependents
        for dependent_id in self.dependents[job_id]:
            if self._is_ready(dependent_id):
                await self._enqueue_job(self.jobs[dependent_id])
    
    async def job_failed(self, job_id: str, error: str):
        """Handle job failure"""
        job = self.jobs[job_id]
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error = error
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        while self.running:
            task = await self.priority_queue.dequeue()
            if task:
                asyncio.create_task(self._execute_task(task))
            else:
                await asyncio.sleep(0.01)
    
    async def _execute_task(self, task: Task):
        """Execute a task"""
        try:
            grant = await self.resource_allocator.allocate(task)
            result = await self.layer_manager.execute_task(task)
            await self.resource_allocator.release(task.id)
            
            # Find and update job
            for job in self.jobs.values():
                if job.id == task.id:
                    await self.job_completed(job.id, result)
                    break
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Task execution failed: {e}")
            for job in self.jobs.values():
                if job.id == task.id:
                    await self.job_failed(job.id, str(e))
                    break
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False


# =============================================================================
# PREEMPTION MANAGER
# =============================================================================

class PreemptionManager:
    """Manages task preemption with state save/restore."""

    def __init__(self):
        self.preempted_tasks: Dict[str, PreemptedTask] = {}

    def _capture_task_state(self, task: Task) -> bytes:
        """Capture serialisable task state as JSON bytes."""
        state = {
            'task_id': task.id,
            'task_type': task.task_type,
            'priority': task.priority,
            'layer': task.layer,
            'data': None,
            'preemption_count': task.preemption_count,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'created_at': task.created_at.isoformat() if task.created_at else None,
        }
        # Attempt to capture task.data (skip if not JSON-serialisable)
        try:
            json.dumps(task.data)
            state['data'] = task.data
        except (TypeError, ValueError):
            state['data'] = repr(task.data) if task.data is not None else None
        return json.dumps(state).encode('utf-8')

    def _restore_task_state(self, task: Task, checkpoint_data: bytes) -> None:
        """Restore serialised state back onto a task."""
        if not checkpoint_data:
            return
        try:
            state = json.loads(checkpoint_data.decode('utf-8'))
            if state.get('data') is not None:
                task.data = state['data']
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            logger.warning(f"Could not restore task state for {task.id}: {e}")

    async def preempt(self, task: Task, reason: PreemptionReason) -> bool:
        """Preempt a running task, saving its state."""
        try:
            checkpoint_data = self._capture_task_state(task)

            preempted = PreemptedTask(
                task=task,
                checkpoint=Checkpoint(
                    task_id=task.id,
                    task_type=task.task_type,
                    checkpoint_data=checkpoint_data,
                ),
                preempted_at=datetime.now(),
                reason=reason,
                original_priority=task.priority
            )

            self.preempted_tasks[task.id] = preempted
            task.preemption_count += 1

            logger.info(
                f"Task {task.id} preempted (reason={reason.name}, "
                f"preemptions={task.preemption_count})"
            )
            return True
        except (RuntimeError, KeyError) as e:
            logger.error(f"Preemption failed: {e}")
            return False

    async def resume(self, task_id: str) -> Optional[Task]:
        """Resume a preempted task, restoring its state."""
        if task_id not in self.preempted_tasks:
            return None

        preempted = self.preempted_tasks[task_id]
        task = preempted.task

        # Restore saved state
        self._restore_task_state(task, preempted.checkpoint.checkpoint_data)

        # Boost priority based on wait time
        wait_seconds = (datetime.now() - preempted.preempted_at).total_seconds()
        boost = min(int(wait_seconds / 10), 20)
        task.priority = max(0, preempted.original_priority - boost)

        del self.preempted_tasks[task_id]
        logger.info(f"Task {task_id} resumed with priority {task.priority}")

        return task


# =============================================================================
# PERSISTENT STORAGE
# =============================================================================

class PersistentQueueStorage:
    """Persistent storage for queue state"""
    
    def __init__(self, db_path: str = "ralph_queue.db"):
        self.db_path = db_path
        self.connection = None
        self._init_db()
        
    def _init_db(self):
        """Initialize database"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queue_tasks (
                task_id TEXT PRIMARY KEY,
                layer INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                task_data BLOB NOT NULL,
                deadline TIMESTAMP,
                estimated_duration REAL,
                preemption_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                checkpoint_data BLOB NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        ''')
        
        self.connection.commit()
    
    def _task_to_json(self, task: Task) -> str:
        """Serialise a Task to JSON, skipping non-serialisable fields like handlers."""
        data = {
            'id': task.id,
            'task_type': task.task_type,
            'priority': task.priority,
            'layer': task.layer,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'preemption_count': task.preemption_count,
            'is_io_bound': task.is_io_bound,
            'is_cpu_intensive': task.is_cpu_intensive,
            'requires_network': task.requires_network,
            'allowed_paths': task.allowed_paths,
            'deadline': task.deadline.isoformat() if task.deadline else None,
            'estimated_duration': task.estimated_duration.total_seconds() if task.estimated_duration else None,
            'cpu_estimate': task.cpu_estimate,
            'memory_estimate': task.memory_estimate,
            'io_estimate': task.io_estimate,
        }
        # Attempt to include task.data; skip if not JSON-serialisable
        try:
            json.dumps(task.data)
            data['data'] = task.data
        except (TypeError, ValueError):
            data['data'] = None
            data['_data_repr'] = repr(task.data) if task.data is not None else None
        # Store handler reference (cannot serialise callables)
        if task.handler:
            data['_handler_ref'] = f"{task.handler.__module__}.{task.handler.__qualname__}" if hasattr(task.handler, '__qualname__') else str(task.handler)
        return json.dumps(data)

    def _task_from_json(self, raw: str) -> Task:
        """Deserialise a Task from JSON."""
        d = json.loads(raw)
        return Task(
            id=d.get('id', str(uuid.uuid4())),
            task_type=d.get('task_type', ''),
            handler=None,  # Handlers cannot be restored from JSON
            priority=d.get('priority', PriorityLevel.P64_AGENT_LOOP),
            layer=d.get('layer', 3),
            deadline=datetime.fromisoformat(d['deadline']) if d.get('deadline') else None,
            estimated_duration=timedelta(seconds=d['estimated_duration']) if d.get('estimated_duration') else timedelta(seconds=1),
            cpu_estimate=d.get('cpu_estimate'),
            memory_estimate=d.get('memory_estimate'),
            io_estimate=d.get('io_estimate'),
            data=d.get('data'),
            created_at=datetime.fromisoformat(d['created_at']) if d.get('created_at') else datetime.now(),
            started_at=datetime.fromisoformat(d['started_at']) if d.get('started_at') else None,
            preemption_count=d.get('preemption_count', 0),
            is_io_bound=d.get('is_io_bound', False),
            is_cpu_intensive=d.get('is_cpu_intensive', False),
            requires_network=d.get('requires_network', False),
            allowed_paths=d.get('allowed_paths', []),
        )

    async def save_task(self, ptask: PrioritizedTask):
        """Save task to storage using JSON serialisation."""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO queue_tasks
            (task_id, layer, priority, sequence, created_at, task_data,
             deadline, estimated_duration, preemption_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ptask.task.id,
            ptask.task.layer,
            ptask.priority,
            ptask.sequence,
            ptask.created_at.isoformat(),
            self._task_to_json(ptask.task),
            ptask.deadline.isoformat() if ptask.deadline else None,
            ptask.estimated_duration.total_seconds() if ptask.estimated_duration else None,
            ptask.preemption_count,
            'pending'
        ))
        self.connection.commit()

    async def load_all_tasks(self) -> List[PrioritizedTask]:
        """Load all pending tasks using JSON deserialisation."""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM queue_tasks
            WHERE status = 'pending'
            ORDER BY layer, priority, sequence
        ''')

        tasks = []
        for row in cursor.fetchall():
            try:
                task = self._task_from_json(row[5])
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Skipping corrupted task row: {e}")
                continue
            ptask = PrioritizedTask(
                priority=row[2],
                sequence=row[3],
                created_at=datetime.fromisoformat(row[4]),
                task=task,
                deadline=datetime.fromisoformat(row[6]) if row[6] else None,
                estimated_duration=timedelta(seconds=row[7]) if row[7] else None,
                preemption_count=row[8]
            )
            tasks.append(ptask)

        return tasks
    
    async def mark_completed(self, task_id: str):
        """Mark task as completed"""
        cursor = self.connection.cursor()
        cursor.execute('''
            UPDATE queue_tasks SET status = 'completed' WHERE task_id = ?
        ''', (task_id,))
        self.connection.commit()


# =============================================================================
# RECOVERY MANAGER
# =============================================================================

class RecoveryManager:
    """Manages system recovery"""
    
    def __init__(self, storage: PersistentQueueStorage,
                 orchestrator: JobOrchestrator):
        self.storage = storage
        self.orchestrator = orchestrator
        
    async def recover(self) -> RecoveryReport:
        """Perform system recovery"""
        report = RecoveryReport()
        
        try:
            # Load persisted tasks
            tasks = await self.storage.load_all_tasks()
            report.tasks_loaded = len(tasks)
            
            # Restore queue
            for ptask in tasks:
                await self.orchestrator.priority_queue.enqueue(
                    ptask.task,
                    ptask.priority
                )
            report.tasks_queued = len(tasks)
            
        except (OSError, RuntimeError) as e:
            report.errors.append(str(e))
            logger.error(f"Recovery failed: {e}")
        
        return report


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Collects system metrics"""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, Callable[[], float]] = {}
        self.timings: Dict[str, List[float]] = defaultdict(list)
        
    def register_gauge(self, name: str, supplier: Callable[[], float]):
        """Register a gauge metric"""
        self.gauges[name] = supplier
        
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] += value
        
    def record_timing(self, name: str, duration: float):
        """Record a timing"""
        self.timings[name].append(duration)
        
    async def collect(self) -> MetricsSnapshot:
        """Collect all metrics"""
        snapshot = MetricsSnapshot(timestamp=datetime.now())
        
        # Collect gauges
        for name, supplier in self.gauges.items():
            try:
                snapshot.gauges[name] = supplier()
            except (TypeError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to collect gauge {name}: {e}")
        
        # Collect counters
        snapshot.counters = dict(self.counters)
        
        return snapshot


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """Monitors system health"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        
    def register_check(self, name: str, check: Callable[[], HealthCheckResult]):
        """Register a health check"""
        self.checks[name] = check
        
    async def run_checks(self) -> HealthReport:
        """Run all health checks"""
        report = HealthReport(timestamp=datetime.now())
        
        for name, check in self.checks.items():
            try:
                result = check()
                report.checks[name] = result
            except (RuntimeError, OSError) as e:
                report.checks[name] = HealthCheckResult(
                    status=HealthStatus.ERROR,
                    message=str(e)
                )
        
        # Calculate overall health
        if any(c.status == HealthStatus.CRITICAL for c in report.checks.values()):
            report.overall = HealthStatus.CRITICAL
        elif any(c.status == HealthStatus.ERROR for c in report.checks.values()):
            report.overall = HealthStatus.ERROR
        elif any(c.status == HealthStatus.WARNING for c in report.checks.values()):
            report.overall = HealthStatus.WARNING
        else:
            report.overall = HealthStatus.HEALTHY
        
        return report


# =============================================================================
# MAIN RALPH LOOP CLASS
# =============================================================================

class RalphLoop:
    """
    Main Ralph Loop - Multi-Layered Background Processing with Priority Queuing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load from YAML if no config provided
        if config is None:
            try:
                from config_loader import get_config
                self.config = get_config("ralph_loop_config", "ralph_loop", {})
            except (ImportError, Exception):
                self.config = {}
        else:
            self.config = config

        # Initialize components
        self.layer_manager = RalphLayerManager(self.config.get('layers'))
        self.priority_queue = RalphPriorityQueue(self.layer_manager)
        self.resource_allocator = ResourceAllocator()
        self.orchestrator = JobOrchestrator(
            self.layer_manager,
            self.priority_queue,
            self.resource_allocator
        )
        self.preemption_manager = PreemptionManager()
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()

        # Storage
        db_path = self.config.get('storage', {}).get('db_path', 'ralph_queue.db')
        self.storage = PersistentQueueStorage(db_path)
        self.recovery_manager = RecoveryManager(self.storage, self.orchestrator)
        
        # State
        self.running = False
        self.started_at: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize the Ralph Loop"""
        logger.info("Initializing Ralph Loop...")
        
        # Register default task types
        self._register_default_task_types()
        
        # Register health checks
        self._register_health_checks()
        
        # Register metrics
        self._register_metrics()
        
        # Perform recovery
        if self.config.get('recovery', {}).get('auto_recover', True):
            report = await self.recovery_manager.recover()
            logger.info(f"Recovery complete: {report}")
        
        logger.info("Ralph Loop initialized")
        
    def _register_default_task_types(self):
        """Register default task types"""
        defaults = [
            ("heartbeat_check", 0, PriorityLevel.P0_SYSTEM_CRITICAL),
            ("identity_verification", 0, PriorityLevel.P0_SYSTEM_CRITICAL),
            ("voice_stream_process", 1, PriorityLevel.P16_VOICE_STREAM),
            ("stt_conversion", 1, PriorityLevel.P17_STT_REALTIME),
            ("tts_generation", 1, PriorityLevel.P18_TTS_STREAM),
            ("twilio_handler", 1, PriorityLevel.P19_TWILIO_ACTIVE),
            ("gmail_operation", 2, PriorityLevel.P32_GMAIL_URGENT),
            ("browser_control", 2, PriorityLevel.P33_BROWSER_NAV),
            ("api_request", 2, PriorityLevel.P34_API_CRITICAL),
            ("agent_loop", 3, PriorityLevel.P64_AGENT_LOOP),
            ("gpt_inference", 3, PriorityLevel.P65_DATA_PROCESS),
            ("file_operation", 3, PriorityLevel.P70_FILE_OPS),
            ("log_maintenance", 4, PriorityLevel.P128_LOG_ROTATION),
            ("cache_cleanup", 4, PriorityLevel.P130_CLEANUP),
            ("bulk_email", 5, PriorityLevel.P192_BULK_EMAIL),
            ("conversation_archive", 6, PriorityLevel.P240_CONVERSATION_ARCH),
        ]
        
        for task_type, layer, priority in defaults:
            self.layer_manager.register_task_type(task_type, layer, priority)
    
    def _register_health_checks(self):
        """Register health checks"""
        def queue_health():
            stats = self.priority_queue.get_queue_stats()
            total = stats['total_tasks']
            if total > 10000:
                return HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    message=f"Queue depth critical: {total}"
                )
            elif total > 5000:
                return HealthCheckResult(
                    status=HealthStatus.WARNING,
                    message=f"Queue depth high: {total}"
                )
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Queue depth normal: {total}"
            )
        
        def memory_health():
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                return HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    message=f"Memory usage critical: {mem.percent}%"
                )
            elif mem.percent > 80:
                return HealthCheckResult(
                    status=HealthStatus.WARNING,
                    message=f"Memory usage high: {mem.percent}%"
                )
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Memory usage normal: {mem.percent}%"
            )
        
        self.health_monitor.register_check("queue", queue_health)
        self.health_monitor.register_check("memory", memory_health)
    
    def _register_metrics(self):
        """Register metrics"""
        self.metrics_collector.register_gauge(
            "queue_depth",
            lambda: self.priority_queue.get_queue_stats()['total_tasks']
        )
        self.metrics_collector.register_gauge(
            "memory_usage_percent",
            lambda: psutil.virtual_memory().percent
        )
        self.metrics_collector.register_gauge(
            "cpu_usage_percent",
            lambda: psutil.cpu_percent()
        )
        
    async def start(self):
        """Start the Ralph Loop"""
        logger.info("Starting Ralph Loop...")
        self.running = True
        self.started_at = datetime.now()
        
        # Start orchestrator
        asyncio.create_task(self.orchestrator.start())
        
        # Start monitoring
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("Ralph Loop started")
        
    async def stop(self):
        """Stop the Ralph Loop"""
        logger.info("Stopping Ralph Loop...")
        self.running = False
        self.orchestrator.stop()
        logger.info("Ralph Loop stopped")
        
    async def submit_task(self, task: Task, priority: Optional[int] = None) -> str:
        """Submit a task for execution"""
        return await self.priority_queue.enqueue(task, priority)
    
    async def submit_job(self, job: Job, dependencies: Optional[List[str]] = None) -> str:
        """Submit a job for execution"""
        return await self.orchestrator.submit_job(job, dependencies)
    
    async def preempt_task(self, task_id: str, reason: PreemptionReason) -> bool:
        """Preempt a task"""
        # Find task
        task = None
        for layer_id in range(7):
            peeked = await self.priority_queue.peek(layer_id, 100)
            for ptask in peeked:
                if ptask.task.id == task_id:
                    task = ptask.task
                    break
            if task:
                break
        
        if task:
            return await self.preemption_manager.preempt(task, reason)
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "running": self.running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds() if self.started_at else 0,
            "queue_stats": self.priority_queue.get_queue_stats(),
            "metrics": self.metrics_collector.counters
        }
    
    async def get_health(self) -> HealthReport:
        """Get system health"""
        return await self.health_monitor.run_checks()
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        interval = self.config.get('monitoring', {}).get('interval_seconds', 10)
        
        while self.running:
            try:
                # Collect metrics
                metrics = await self.metrics_collector.collect()
                
                # Run health checks
                health = await self.health_monitor.run_checks()
                
                if health.overall == HealthStatus.CRITICAL:
                    logger.error(f"Critical health status detected: {health}")
                
            except (RuntimeError, OSError) as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(interval)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of Ralph Loop"""
    
    # Create configuration
    config = {
        'layers': {
            'critical': {'max_concurrent': 10},
            'real_time': {'max_concurrent': 20},
            'standard': {'max_concurrent': 100}
        },
        'recovery': {'auto_recover': True},
        'storage': {'db_path': 'data/ralph_queue.db'},
        'monitoring': {'interval_seconds': 10}
    }
    
    # Create Ralph Loop
    ralph = RalphLoop(config)
    
    # Initialize
    await ralph.initialize()
    
    # Start
    await ralph.start()
    
    # Submit some tasks
    async def sample_task(data):
        await asyncio.sleep(0.1)
        return f"Processed: {data}"
    
    task1 = Task(
        task_type="agent_loop",
        handler=sample_task,
        data="Hello World"
    )
    
    task_id = await ralph.submit_task(task1)
    print(f"Submitted task: {task_id}")
    
    # Submit a job
    job = Job(
        name="sample_job",
        task_type="file_operation",
        handler=sample_task,
        data="Job data"
    )
    
    job_id = await ralph.submit_job(job)
    print(f"Submitted job: {job_id}")
    
    # Get stats
    stats = await ralph.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Get health
    health = await ralph.get_health()
    print(f"Health: {health.overall}")
    
    # Run for a while
    await asyncio.sleep(5)
    
    # Stop
    await ralph.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
