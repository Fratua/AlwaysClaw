"""
OpenClaw Windows 10 AI Agent - Resource Management System
=========================================================

A comprehensive resource management system for 24/7 AI agent operation.
Implements memory management, CPU scheduling, API optimization, caching,
lazy loading, resource pooling, GC tuning, and monitoring.

Usage:
    from resource_manager import ResourceOrchestrator
    
    orchestrator = ResourceOrchestrator()
    orchestrator.start()
    
    # Your agent code here
    
    orchestrator.stop()
"""

import gc
import logging
import os
import sys
import time
import json
import pickle
import hashlib
import asyncio
import threading
import psutil
from typing import Dict, List, Optional, Callable, Any, TypeVar, Generic

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from enum import Enum
from contextlib import contextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import queue

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ResourceConfig:
    """Global resource configuration"""
    # Memory
    memory_warning_threshold: float = 70.0
    memory_critical_threshold: float = 85.0
    memory_emergency_threshold: float = 95.0
    memory_pool_size: int = 100 * 1024 * 1024  # 100MB
    
    # CPU
    max_cpu_percent: float = 80.0
    worker_threads: int = 4
    
    # API
    api_hourly_budget: float = 5.0
    api_daily_budget: float = 50.0
    api_batch_size: int = 10
    api_batch_window: float = 0.5
    
    # Cache
    cache_l1_size: int = 100 * 1024 * 1024  # 100MB
    cache_l1_items: int = 10000
    cache_l2_path: str = "./cache"
    cache_l2_size: int = 1024 * 1024 * 1024  # 1GB
    cache_ttl: int = 3600  # 1 hour
    
    # Pool
    pool_min_size: int = 2
    pool_max_size: int = 10
    pool_max_idle: float = 300  # 5 minutes
    
    # GC
    gc_gen0_threshold: int = 50000
    gc_gen1_threshold: int = 100
    gc_gen2_threshold: int = 100
    
    # Monitoring
    monitor_interval: int = 10  # seconds

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

@dataclass
class MemoryPool:
    """Memory pool for component allocation"""
    name: str
    min_size: int
    max_size: int
    current_usage: int = 0
    objects: OrderedDict = field(default_factory=OrderedDict)

class MemoryManager:
    """Central memory management for AI agent system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: ResourceConfig = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: ResourceConfig = None):
        if self._initialized:
            return
        
        self.config = config or ResourceConfig()
        self._initialized = True
        self.pools: Dict[str, MemoryPool] = {}
        self._callbacks: List[Callable] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._lru_cache: OrderedDict = OrderedDict()
        
        self._tune_gc()
    
    def _tune_gc(self):
        """Tune Python garbage collector"""
        gc.freeze()
        gc.set_threshold(
            self.config.gc_gen0_threshold,
            self.config.gc_gen1_threshold,
            self.config.gc_gen2_threshold
        )
    
    def create_pool(self, name: str, min_size: int, max_size: int) -> MemoryPool:
        """Create a named memory pool"""
        pool = MemoryPool(name=name, min_size=min_size, max_size=max_size)
        self.pools[name] = pool
        return pool
    
    def allocate(self, pool_name: str, obj_id: str, size: int, obj: object) -> bool:
        """Allocate memory from a pool"""
        if pool_name not in self.pools:
            return False
        
        pool = self.pools[pool_name]
        
        if pool.current_usage + size > pool.max_size:
            self._evict_lru(pool_name, size)
        
        if pool.current_usage + size <= pool.max_size:
            pool.objects[obj_id] = {
                'size': size,
                'timestamp': time.time()
            }
            pool.current_usage += size
            return True
        
        return False
    
    def _evict_lru(self, pool_name: str, required_size: int):
        """Evict LRU objects to free space"""
        pool = self.pools[pool_name]
        freed = 0
        
        sorted_objects = sorted(pool.objects.items(), key=lambda x: x[1]['timestamp'])
        
        for obj_id, obj_data in sorted_objects:
            if freed >= required_size:
                break
            freed += obj_data['size']
            del pool.objects[obj_id]
            pool.current_usage -= obj_data['size']
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            'process': {
                'rss': mem_info.rss,
                'vms': mem_info.vms,
                'percent': process.memory_percent()
            },
            'system': {
                'total': system_mem.total,
                'available': system_mem.available,
                'percent': system_mem.percent
            },
            'pools': {
                name: {
                    'current': pool.current_usage,
                    'max': pool.max_size,
                    'objects': len(pool.objects)
                }
                for name, pool in self.pools.items()
            }
        }
    
    def start_monitoring(self, interval: int = 30):
        """Start background memory monitoring"""
        def monitor():
            while not self._shutdown:
                stats = self.get_memory_stats()
                system_percent = stats['system']['percent']
                
                if system_percent >= self.config.memory_emergency_threshold:
                    self._emergency_cleanup()
                    self._notify_callbacks('emergency', stats)
                elif system_percent >= self.config.memory_critical_threshold:
                    self._critical_cleanup()
                    self._notify_callbacks('critical', stats)
                elif system_percent >= self.config.memory_warning_threshold:
                    self._notify_callbacks('warning', stats)
                
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        gc.collect(2)
        self._lru_cache.clear()
        for pool in self.pools.values():
            self._evict_lru(pool.name, pool.max_size // 2)
    
    def _critical_cleanup(self):
        """Critical memory cleanup"""
        gc.collect(1)
        cache_items = len(self._lru_cache)
        for _ in range(cache_items // 4):
            if self._lru_cache:
                self._lru_cache.popitem(last=False)
    
    def register_callback(self, callback: Callable):
        """Register memory threshold callback"""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, level: str, stats: Dict):
        """Notify callbacks"""
        for callback in self._callbacks:
            try:
                callback(level, stats)
            except Exception as e:
                print(f"Callback error: {e}")

# ============================================================================
# CPU SCHEDULING
# ============================================================================

class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class Task:
    """Schedulable task"""
    id: str
    priority: TaskPriority
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    cpu_limit: float = 10.0
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)

class CPUScheduler:
    """CPU scheduler for agent loops and tasks"""
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.process = psutil.Process()
        self.max_workers = self.config.worker_threads
        
        self._queues: Dict[TaskPriority, queue.PriorityQueue] = {
            priority: queue.PriorityQueue() for priority in TaskPriority
        }
        
        self._agent_loops: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._shutdown = False
        
        self._stats = {
            'tasks_executed': 0,
            'tasks_failed': 0,
            'avg_cpu_usage': 0.0
        }
    
    def register_agent_loop(self, name: str, priority: TaskPriority, 
                           interval: float, cpu_limit: float = 10.0):
        """Register an agent loop"""
        self._agent_loops[name] = {
            'name': name,
            'priority': priority,
            'interval': interval,
            'cpu_limit': cpu_limit,
            'last_run': 0,
            'execution_count': 0
        }
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task"""
        current_cpu = self.process.cpu_percent(interval=0.1)
        
        if current_cpu > self.config.max_cpu_percent and task.priority != TaskPriority.CRITICAL:
            return False
        
        self._queues[task.priority].put((task.priority.value, task))
        return True
    
    async def _execute_task(self, task: Task):
        """Execute a task"""
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, task.func, *task.args, **task.kwargs),
                    timeout=task.timeout
                )
            
            self._stats['tasks_executed'] += 1
            return result
            
        except Exception as e:
            self._stats['tasks_failed'] += 1
            raise
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self._shutdown:
            # Process agent loops
            current_time = time.time()
            
            for loop_name, loop_config in self._agent_loops.items():
                if current_time - loop_config['last_run'] < loop_config['interval']:
                    continue
                
                current_cpu = self.process.cpu_percent(interval=0.1)
                if current_cpu + loop_config['cpu_limit'] > 90:
                    continue
                
                loop_config['last_run'] = current_time
                loop_config['execution_count'] += 1
            
            # Process queued tasks
            for priority in TaskPriority:
                q = self._queues[priority]
                for _ in range(5):
                    if q.empty():
                        break
                    try:
                        _, task = q.get_nowait()
                        await self._execute_task(task)
                    except queue.Empty:
                        break
            
            await asyncio.sleep(0.01)
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            **self._stats,
            'current_cpu': self.process.cpu_percent(interval=0.1),
            'queue_sizes': {p.name: q.qsize() for p, q in self._queues.items()},
            'agent_loops': self._agent_loops
        }
    
    def start(self):
        """Start scheduler"""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._scheduler_loop())
    
    def stop(self):
        """Stop scheduler"""
        self._shutdown = True
        if self._loop:
            self._loop.stop()
        self._executor.shutdown(wait=True)

# ============================================================================
# API OPTIMIZER
# ============================================================================

@dataclass
class APIRequest:
    """API request"""
    id: str
    provider: str
    endpoint: str
    payload: Dict[str, Any]
    priority: int = 5
    cache_key: Optional[str] = None
    batchable: bool = True
    created_at: float = field(default_factory=time.time)

@dataclass
class APIResponse:
    """API response"""
    request_id: str
    success: bool
    data: Any
    latency: float
    tokens_used: int = 0
    cost: float = 0.0
    cached: bool = False

class CostTracker:
    """Track API costs"""
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.daily_spent: Dict[str, float] = defaultdict(float)
        self.hourly_spent: Dict[str, float] = defaultdict(float)
    
    def record_cost(self, provider: str, tokens_in: int, tokens_out: int) -> float:
        """Record API call cost"""
        # Approximate pricing
        pricing = {'input': 0.015, 'output': 0.060}
        
        input_cost = (tokens_in / 1000) * pricing['input']
        output_cost = (tokens_out / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        hour_key = time.strftime('%Y-%m-%d-%H')
        day_key = time.strftime('%Y-%m-%d')
        
        self.hourly_spent[hour_key] += total_cost
        self.daily_spent[day_key] += total_cost
        
        return total_cost
    
    def check_budget(self) -> Dict[str, Any]:
        """Check budget status"""
        hour_key = time.strftime('%Y-%m-%d-%H')
        day_key = time.strftime('%Y-%m-%d')
        
        return {
            'hourly_ok': self.hourly_spent[hour_key] < self.config.api_hourly_budget,
            'daily_ok': self.daily_spent[day_key] < self.config.api_daily_budget,
            'hourly_spent': self.hourly_spent[hour_key],
            'daily_spent': self.daily_spent[day_key]
        }

class APIOptimizer:
    """API call optimization"""
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.cost_tracker = CostTracker(config)
        
        self._queues: Dict[str, asyncio.PriorityQueue] = defaultdict(asyncio.PriorityQueue)
        self._cache: Dict[str, APIResponse] = {}
        self._cache_ttl: Dict[str, float] = {}
        
        self._stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'total_latency': 0.0
        }
    
    def _generate_cache_key(self, provider: str, endpoint: str, payload: Dict) -> str:
        """Generate cache key"""
        key_data = f"{provider}:{endpoint}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[APIResponse]:
        """Check cache"""
        if cache_key in self._cache:
            if time.time() < self._cache_ttl.get(cache_key, 0):
                self._stats['cached_requests'] += 1
                return self._cache[cache_key]
            else:
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
        return None
    
    async def submit_request(self, request: APIRequest) -> asyncio.Future:
        """Submit API request"""
        budget_status = self.cost_tracker.check_budget()
        if not budget_status['hourly_ok'] or not budget_status['daily_ok']:
            raise Exception("API budget exceeded")
        
        if request.cache_key:
            cached = self._check_cache(request.cache_key)
            if cached:
                future = asyncio.Future()
                future.set_result(cached)
                return future
        
        future = asyncio.Future()
        await self._queues[request.provider].put((request.priority, request, future))
        return future
    
    def get_stats(self) -> Dict:
        """Get API statistics"""
        total = self._stats['total_requests']
        return {
            **self._stats,
            'cache_hit_rate': (self._stats['cached_requests'] / total * 100) if total > 0 else 0,
            'cost_status': self.cost_tracker.check_budget()
        }

# ============================================================================
# CACHE MANAGER
# ============================================================================

class L1Cache:
    """In-memory LRU cache"""
    
    def __init__(self, max_size: int, max_items: int):
        self.max_size = max_size
        self.max_items = max_items
        self._cache: OrderedDict = OrderedDict()
        self._size = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]['value']
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            try:
                size = len(pickle.dumps(value))
            except Exception as e:
                logger.debug(f"Could not determine serialized size, using default: {e}")
                size = 1024
            
            while (self._size + size > self.max_size or len(self._cache) >= self.max_items):
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'size': size,
                'expires': time.time() + ttl if ttl else None
            }
            self._size += size
    
    def _evict_lru(self):
        if self._cache:
            key, data = self._cache.popitem(last=False)
            self._size -= data['size']
    
    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0
        }

class CacheManager:
    """Multi-tier cache manager"""
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self._l1 = L1Cache(self.config.cache_l1_size, self.config.cache_l1_items)
    
    def get(self, key: str) -> Optional[Any]:
        return self._l1.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._l1.set(key, value, ttl or self.config.cache_ttl)
    
    def get_stats(self) -> Dict:
        return {'l1': self._l1.get_stats()}

# ============================================================================
# RESOURCE POOL
# ============================================================================

T = TypeVar('T')

class ResourcePool(Generic[T]):
    """Generic resource pool"""
    
    def __init__(
        self,
        factory: Callable[[], T],
        validator: Callable[[T], bool] = lambda x: True,
        destroyer: Callable[[T], None] = lambda x: None,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: float = 300
    ):
        self._factory = factory
        self._validator = validator
        self._destroyer = destroyer
        self._min_size = min_size
        self._max_size = max_size
        self._max_idle_time = max_idle_time
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._in_use: Dict[int, Any] = {}
        self._lock = threading.RLock()
        
        self._stats = {'created': 0, 'destroyed': 0, 'acquired': 0, 'released': 0}
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self._min_size):
            try:
                resource = self._create_resource()
                self._pool.put_nowait(resource)
            except queue.Full:
                break
    
    def _create_resource(self):
        raw = self._factory()
        pooled = {
            'resource': raw,
            'created_at': time.time(),
            'last_used': time.time()
        }
        self._stats['created'] += 1
        return pooled
    
    def acquire(self, timeout: Optional[float] = None) -> T:
        self._stats['acquired'] += 1
        
        try:
            pooled = self._pool.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                if self._stats['created'] < self._max_size:
                    pooled = self._create_resource()
                else:
                    raise RuntimeError("Pool exhausted")
        
        pooled['last_used'] = time.time()
        
        with self._lock:
            self._in_use[id(pooled['resource'])] = pooled
        
        return pooled['resource']
    
    def release(self, resource: T):
        self._stats['released'] += 1
        
        with self._lock:
            pooled = self._in_use.pop(id(resource), None)
        
        if pooled:
            pooled['last_used'] = time.time()
            try:
                self._pool.put_nowait(pooled)
            except queue.Full:
                self._destroyer(pooled['resource'])
    
    @contextmanager
    def get_resource(self, timeout: Optional[float] = None):
        resource = self.acquire(timeout)
        try:
            yield resource
        finally:
            self.release(resource)
    
    def get_stats(self) -> Dict:
        return {
            **self._stats,
            'available': self._pool.qsize(),
            'in_use': len(self._in_use)
        }

# ============================================================================
# MONITORING
# ============================================================================

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)

class ResourceMonitor:
    """Resource monitoring system"""
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.process = psutil.Process()
        self._callbacks: List[Callable[[Alert], None]] = []
        self._alert_history: deque = deque(maxlen=1000)
        self._running = False
        self._shutdown = False
    
    def register_callback(self, callback: Callable[[Alert], None]):
        self._callbacks.append(callback)
    
    def _check_threshold(self, name: str, value: float, warning: float, critical: float, emergency: float):
        alert = None
        
        if value >= emergency:
            alert = Alert(AlertLevel.EMERGENCY, f"{name} at EMERGENCY: {value:.2f}", name, value, emergency)
        elif value >= critical:
            alert = Alert(AlertLevel.CRITICAL, f"{name} at CRITICAL: {value:.2f}", name, value, critical)
        elif value >= warning:
            alert = Alert(AlertLevel.WARNING, f"{name} at WARNING: {value:.2f}", name, value, warning)
        
        if alert:
            self._alert_history.append(alert)
            for callback in self._callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'process_memory': self.process.memory_info().rss,
            'process_cpu': self.process.cpu_percent()
        }
    
    async def _monitoring_loop(self):
        while not self._shutdown:
            try:
                metrics = self._collect_metrics()
                
                self._check_threshold(
                    'cpu_percent', metrics['cpu_percent'],
                    70, 85, 95
                )
                self._check_threshold(
                    'memory_percent', metrics['memory_percent'],
                    self.config.memory_warning_threshold,
                    self.config.memory_critical_threshold,
                    self.config.memory_emergency_threshold
                )
                
                await asyncio.sleep(self.config.monitor_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(30)
    
    def get_stats(self) -> Dict:
        return {
            'current_metrics': self._collect_metrics(),
            'alerts': len(self._alert_history)
        }
    
    def start(self):
        self._running = True
        asyncio.create_task(self._monitoring_loop())
    
    def stop(self):
        self._shutdown = True
        self._running = False

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ResourceOrchestrator:
    """
    Central orchestrator for all resource management
    Coordinates memory, CPU, API, cache, and monitoring
    """
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        
        # Initialize all managers
        self.memory = MemoryManager(self.config)
        self.cpu = CPUScheduler(self.config)
        self.api = APIOptimizer(self.config)
        self.cache = CacheManager(self.config)
        self.monitor = ResourceMonitor(self.config)
        
        # Setup default alert handler
        self.monitor.register_callback(self._default_alert_handler)
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert handler"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔥"
        }
        print(f"{emoji.get(alert.level, '❓')} [{alert.level.value.upper()}] {alert.message}")
    
    def register_agent_loop(self, name: str, priority: TaskPriority, interval: float, cpu_limit: float = 10.0):
        """Register an agent loop with the CPU scheduler"""
        self.cpu.register_agent_loop(name, priority, interval, cpu_limit)
    
    def create_memory_pool(self, name: str, min_size: int, max_size: int) -> MemoryPool:
        """Create a memory pool"""
        return self.memory.create_pool(name, min_size, max_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'memory': self.memory.get_memory_stats(),
            'cpu': self.cpu.get_stats(),
            'api': self.api.get_stats(),
            'cache': self.cache.get_stats(),
            'monitor': self.monitor.get_stats()
        }
    
    def start(self):
        """Start all resource managers"""
        print("🚀 Starting Resource Orchestrator...")
        
        # Start memory monitoring
        self.memory.start_monitoring()
        print("✅ Memory manager started")
        
        # Start CPU scheduler in background thread
        self._cpu_thread = threading.Thread(target=self.cpu.start, daemon=True)
        self._cpu_thread.start()
        print("✅ CPU scheduler started")
        
        # Start monitoring
        self.monitor.start()
        print("✅ Monitor started")
        
        print("✅ Resource Orchestrator ready")
    
    def stop(self):
        """Stop all resource managers"""
        print("🛑 Stopping Resource Orchestrator...")
        
        self.memory._shutdown = True
        self.cpu.stop()
        self.monitor.stop()
        
        print("✅ Resource Orchestrator stopped")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lazy_property(func: Callable) -> property:
    """Decorator for lazy-loading properties"""
    attr_name = f"_lazy_{func.__name__}"
    
    @property
    @wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper

def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss
        mem_diff = mem_after - mem_before
        
        print(f"📊 {func.__name__}: Memory delta = {mem_diff / 1024 / 1024:.2f} MB")
        
        return result
    
    return wrapper

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ResourceConfig',
    'ResourceOrchestrator',
    'MemoryManager',
    'MemoryPool',
    'CPUScheduler',
    'TaskPriority',
    'Task',
    'APIOptimizer',
    'APIRequest',
    'APIResponse',
    'CostTracker',
    'CacheManager',
    'ResourcePool',
    'ResourceMonitor',
    'Alert',
    'AlertLevel',
    'lazy_property',
    'profile_memory'
]

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    config = ResourceConfig(
        memory_warning_threshold=70.0,
        api_daily_budget=50.0,
        worker_threads=4
    )
    
    orchestrator = ResourceOrchestrator(config)
    
    # Register agent loops
    orchestrator.register_agent_loop('heartbeat', TaskPriority.CRITICAL, 5.0, 2.0)
    orchestrator.register_agent_loop('gmail_sync', TaskPriority.NORMAL, 30.0, 10.0)
    
    # Create memory pools
    orchestrator.create_memory_pool('gpt_responses', 10 * 1024 * 1024, 50 * 1024 * 1024)
    orchestrator.create_memory_pool('embeddings', 20 * 1024 * 1024, 100 * 1024 * 1024)
    
    # Start orchestrator
    orchestrator.start()
    
    try:
        # Run for demonstration
        print("\n📈 System Statistics:")
        import json
        print(json.dumps(orchestrator.get_stats(), indent=2, default=str))
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        orchestrator.stop()
