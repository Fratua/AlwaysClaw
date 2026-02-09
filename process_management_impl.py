"""
Process Management & Supervision Implementation
Windows 10 OpenClaw-Inspired AI Agent System

This module provides the core process management infrastructure including:
- Supervision trees (One-for-One, Rest-for-One, One-for-All)
- Worker pool management (Prefork, Thread, Async)
- Process spawning with Windows optimizations
- Inter-process communication (Named Pipes, Message Queues)
- Heartbeat monitoring
- Crash recovery and restart strategies
- Graceful shutdown
- Process isolation (Windows Job Objects)
"""

import asyncio
import multiprocessing
import threading
import time
import uuid
import signal
import sys
import os
import struct
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any, Coroutine, Set
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque

# Third-party imports
import psutil

# Windows-specific imports
win32api = None
win32process = None
win32con = None
win32pipe = None
win32file = None
win32job = None

if sys.platform == 'win32':
    try:
        import win32api
        import win32process
        import win32con
        import win32pipe
        import win32file
        import win32job
    except ImportError:
        logging.getLogger(__name__).warning(
            "win32 modules not available — Windows process management features disabled"
        )

logger = logging.getLogger('ProcessManager')


def _require_win32():
    """Raise if win32 modules are not available."""
    if win32api is None:
        raise RuntimeError(
            "win32 modules required for Windows process management. "
            "Install pywin32: pip install pywin32"
        )


# =============================================================================
# ENUMERATIONS AND DATA CLASSES
# =============================================================================

class RestartPolicy(Enum):
    """Child restart policies."""
    PERMANENT = auto()  # Always restart
    TEMPORARY = auto()  # Never restart
    TRANSIENT = auto()  # Restart only on abnormal exit


class ChildType(Enum):
    """Type of supervised child."""
    WORKER = auto()     # Task-executing process
    SUPERVISOR = auto() # Nested supervisor


class PoolType(Enum):
    """Worker pool types."""
    PREFORK = auto()    # Multiprocessing (CPU-bound)
    THREAD = auto()     # Threading (I/O-bound)
    ASYNC = auto()      # Asyncio (I/O-bound, high concurrency)


@dataclass
class ChildSpec:
    """Specification for a supervised child process."""
    id: str
    start_func: Callable
    start_args: tuple = field(default_factory=tuple)
    start_kwargs: Dict[str, Any] = field(default_factory=dict)
    restart_policy: RestartPolicy = RestartPolicy.PERMANENT
    child_type: ChildType = ChildType.WORKER
    shutdown_timeout: int = 5000  # Milliseconds
    max_restarts: int = 5
    modules: Optional[List[str]] = None


@dataclass
class HeartbeatData:
    """Heartbeat data structure."""
    process_id: str
    process_type: str
    process_name: str
    timestamp: float
    uptime_seconds: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    thread_count: int
    handle_count: int
    status: str
    active_tasks: int
    completed_tasks: int
    error_count: int
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SUPERVISOR BASE CLASS
# =============================================================================

class Supervisor:
    """Base class for all supervisors."""
    
    def __init__(self, max_restarts: int = 5, time_window: int = 60):
        self.max_restarts = max_restarts
        self.time_window = time_window
        self.children: Dict[str, ChildSpec] = {}
        self.child_pids: Dict[str, int] = {}
        self.child_tasks: Dict[str, asyncio.Task] = {}
        self.restart_history: Dict[str, List[float]] = {}
        self.running = False
        
    def add_child(self, spec: ChildSpec):
        """Add a child specification."""
        self.children[spec.id] = spec
        self.restart_history[spec.id] = []
        
    async def start(self):
        """Start all children."""
        self.running = True
        for spec in self.children.values():
            await self._start_child(spec)
            
    async def stop(self):
        """Stop all children."""
        self.running = False
        for child_id in list(self.child_tasks.keys()) + list(self.child_pids.keys()):
            await self._terminate_child(child_id)
            
    async def _start_child(self, spec: ChildSpec):
        """Start a single child as an asyncio task."""
        try:
            task = asyncio.create_task(
                spec.start_func(*spec.start_args, **spec.start_kwargs)
            )
            self.child_tasks[spec.id] = task
            logger.info(f"Started child {spec.id}")
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to start child {spec.id}: {e}")

    async def _terminate_child(self, child_id: str):
        """Terminate a child (asyncio task or OS process)."""
        # Handle asyncio tasks
        if child_id in self.child_tasks:
            task = self.child_tasks.pop(child_id)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError, OSError):
                pass
            return

        # Handle real OS PIDs (from subprocess spawners)
        if child_id in self.child_pids:
            pid = self.child_pids.pop(child_id)
            try:
                process = psutil.Process(pid)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
            
    def _exceeded_max_restarts(self, child_id: str) -> bool:
        """Check if max restart intensity exceeded."""
        now = time.time()
        history = self.restart_history.get(child_id, [])
        
        # Remove old entries
        history = [t for t in history if now - t < self.time_window]
        self.restart_history[child_id] = history
        
        return len(history) >= self.max_restarts


class OneForOneSupervisor(Supervisor):
    """
    One-for-One supervision strategy.
    Restarts only the failed child process.
    """
    
    async def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child process exit."""
        if not self.running:
            return
            
        if self._exceeded_max_restarts(child_id):
            logger.error(f"Max restarts exceeded for {child_id}, giving up")
            await self._escalate_failure(child_id)
            return
        
        child_spec = self.children.get(child_id)
        if child_spec and child_spec.restart_policy != RestartPolicy.TEMPORARY:
            logger.info(f"Restarting child {child_id}")
            await self._restart_child(child_id, child_spec)
            
    async def _restart_child(self, child_id: str, spec: ChildSpec):
        """Restart a child process."""
        await self._terminate_child(child_id)
        await asyncio.sleep(1)  # Brief delay before restart
        await self._start_child(spec)
        
        # Record restart
        self.restart_history[child_id].append(time.time())
        
    async def _escalate_failure(self, child_id: str):
        """Escalate failure to parent supervisor."""
        logger.critical(f"Escalating failure for {child_id}")
        # Implementation would notify parent


class RestForOneSupervisor(Supervisor):
    """
    Rest-for-One supervision strategy.
    Restarts failed child and all children started after it.
    """
    
    def __init__(self, max_restarts: int = 5, time_window: int = 60):
        super().__init__(max_restarts, time_window)
        self._child_order: List[str] = []
        
    def add_child(self, spec: ChildSpec):
        """Add a child in order."""
        super().add_child(spec)
        self._child_order.append(spec.id)
        
    async def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child exit with dependency awareness."""
        if not self.running:
            return
            
        try:
            failed_idx = self._child_order.index(child_id)
        except ValueError:
            return
            
        # Terminate all children from failed position onward
        to_restart = self._child_order[failed_idx:]
        
        for cid in to_restart:
            await self._terminate_child(cid)
            
        # Restart in order
        for cid in to_restart:
            spec = self.children[cid]
            await self._start_child(spec)
            await asyncio.sleep(0.5)  # Stagger restarts


class OneForAllSupervisor(Supervisor):
    """
    One-for-All supervision strategy.
    Restarts all children when any child fails.
    """
    
    async def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child exit - restart all children."""
        if not self.running:
            return
            
        logger.error(f"Child {child_id} exited with {exit_code}. "
                    f"Restarting all children.")
        
        # Terminate all children
        for cid in list(self.child_tasks.keys()) + list(self.child_pids.keys()):
            await self._terminate_child(cid)

        # Restart all children in order
        for spec in self.children.values():
            await self._start_child(spec)
            await asyncio.sleep(0.5)


# =============================================================================
# WORKER POOLS
# =============================================================================

class WorkerPool(ABC):
    """Base class for worker pools."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0
        }

    async def start(self):
        """Start the pool."""
        pass

    async def stop(self):
        """Stop the pool."""
        pass

    @abstractmethod
    def submit(self, task: Callable, *args, **kwargs):
        """Submit a task to the pool."""
        ...


class PreforkWorkerPool(WorkerPool):
    """
    Prefork pool for CPU-intensive tasks.
    Uses separate processes to bypass Python GIL.
    """
    
    def __init__(self, name: str,
                 min_workers: int = 2,
                 max_workers: int = None,
                 max_tasks_per_child: int = 1000):
        super().__init__(name)
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.max_tasks_per_child = max_tasks_per_child
        self.workers: List[Dict] = []
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self._lock = threading.Lock()
        
    async def start(self):
        """Start the worker pool."""
        for i in range(self.min_workers):
            self._spawn_worker()
        logger.info(f"Started {self.name} with {self.min_workers} workers")
        
    def _spawn_worker(self):
        """Spawn a new worker process."""
        ctx = multiprocessing.get_context('spawn')
        worker = ctx.Process(
            target=self._worker_loop,
            args=(self.task_queue, self.result_queue, self.max_tasks_per_child),
            name=f"{self.name}-Worker-{len(self.workers)}"
        )
        worker.daemon = True
        worker.start()
        
        with self._lock:
            self.workers.append({
                'process': worker,
                'pid': worker.pid,
                'tasks_completed': 0,
                'status': 'active'
            })
            
    @staticmethod
    def _worker_loop(task_queue, result_queue, max_tasks):
        """Worker process main loop."""
        task_count = 0
        
        while task_count < max_tasks:
            try:
                task_data = task_queue.get(timeout=1)
                if task_data is None:
                    break
                    
                func = task_data['func']
                args = task_data.get('args', ())
                kwargs = task_data.get('kwargs', {})
                task_id = task_data['id']
                
                try:
                    result = func(*args, **kwargs)
                    result_queue.put({
                        'id': task_id,
                        'status': 'success',
                        'result': result
                    })
                except (RuntimeError, OSError, ValueError, TypeError) as e:
                    result_queue.put({
                        'id': task_id,
                        'status': 'error',
                        'error': str(e)
                    })

                task_count += 1

            except (RuntimeError, OSError, ValueError):
                continue
                
    def submit(self, task: Callable, *args, **kwargs) -> str:
        """Submit a task to the pool."""
        task_id = str(uuid.uuid4())
        self.task_queue.put({
            'id': task_id,
            'func': task,
            'args': args,
            'kwargs': kwargs
        })
        self.metrics['tasks_submitted'] += 1
        return task_id
        
    async def stop(self):
        """Stop all workers."""
        # Send shutdown signal
        for _ in self.workers:
            self.task_queue.put(None)
            
        # Wait for workers to exit
        for worker_info in self.workers:
            worker_info['process'].join(timeout=5)
            if worker_info['process'].is_alive():
                worker_info['process'].terminate()


class ThreadWorkerPool(WorkerPool):
    """
    Thread pool for I/O-bound tasks.
    """
    
    def __init__(self, name: str,
                 min_workers: int = 5,
                 max_workers: int = 100):
        super().__init__(name)
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{name}-"
        )
        self.futures: Dict[Future, Dict] = {}
        
    def submit(self, task: Callable, *args, **kwargs) -> Future:
        """Submit a task to the thread pool."""
        future = self.executor.submit(task, *args, **kwargs)
        self.futures[future] = {
            'submitted_at': time.time(),
            'task_name': task.__name__
        }
        self.metrics['tasks_submitted'] += 1
        return future
        
    async def stop(self):
        """Stop the thread pool."""
        self.executor.shutdown(wait=True)


class AsyncWorkerPool(WorkerPool):
    """
    Asyncio-based pool for async tasks.
    """
    
    def __init__(self, name: str, max_concurrency: int = 1000):
        super().__init__(name)
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks: Set[asyncio.Task] = set()
        
    async def submit(self, coro: Coroutine) -> Any:
        """Submit an async task to the pool."""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.tasks.add(task)
            self.metrics['tasks_submitted'] += 1
            try:
                return await task
            finally:
                self.tasks.discard(task)
                self.metrics['tasks_completed'] += 1
                
    async def stop(self):
        """Stop all async tasks."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


# =============================================================================
# PROCESS SPAWNER
# =============================================================================

class ProcessSpawner:
    """
    Cross-platform process spawner with Windows optimizations.
    """
    
    # Windows priority classes
    PRIORITY_CLASSES = {
        'idle': 64,
        'below_normal': 16384,
        'normal': 32,
        'above_normal': 32768,
        'high': 128,
        'realtime': 256
    }
    
    def __init__(self):
        self.spawned_processes: Dict[int, Dict] = {}
        self._lock = threading.Lock()
        
    def spawn(self, 
              target: Callable,
              args: tuple = (),
              kwargs: Dict = None,
              name: str = None,
              daemon: bool = True,
              priority: str = None,
              affinity: List[int] = None) -> multiprocessing.Process:
        """Spawn a new process with optional priority and affinity."""
        ctx = multiprocessing.get_context('spawn')
        
        process = ctx.Process(
            target=self._process_wrapper,
            args=(target, args, kwargs, priority, affinity),
            name=name
        )
        process.daemon = daemon
        process.start()
        
        with self._lock:
            self.spawned_processes[process.pid] = {
                'process': process,
                'name': name,
                'started_at': time.time()
            }
            
        return process
    
    def _process_wrapper(self, target, args, kwargs, priority, affinity):
        """Wrapper to set process properties after spawn."""
        # Set process priority on Windows
        if sys.platform == 'win32' and priority:
            self._set_windows_priority(priority)
            
        # Set CPU affinity
        if affinity:
            p = psutil.Process()
            p.cpu_affinity(affinity)
            
        # Run the actual target
        try:
            target(*(args or ()), **(kwargs or {}))
        except (OSError, RuntimeError, ValueError) as e:
            logger.exception(f"Process failed: {e}")
            raise
            
    def _set_windows_priority(self, priority: str):
        """Set Windows process priority."""
        if sys.platform != 'win32':
            return
        _require_win32()
        try:
            handle = win32api.GetCurrentProcess()
            priority_class = self.PRIORITY_CLASSES.get(priority, 32)
            win32process.SetPriorityClass(handle, priority_class)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to set priority: {e}")


# =============================================================================
# HEARTBEAT SYSTEM
# =============================================================================

class HeartbeatEmitter:
    """Emits heartbeats from worker processes."""
    
    def __init__(self, 
                 process_id: str,
                 process_type: str,
                 interval: int = 10,
                 ipc_client = None):
        self.process_id = process_id
        self.process_type = process_type
        self.interval = interval
        self.ipc_client = ipc_client
        self.start_time = time.time()
        self.running = False
        self._task = None
        
    async def start(self):
        """Start emitting heartbeats."""
        self.running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        
    async def stop(self):
        """Stop emitting heartbeats."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        while self.running:
            try:
                heartbeat = self._collect_metrics()
                await self._send_heartbeat(heartbeat)
            except (OSError, RuntimeError, ValueError) as e:
                logger.error(f"Heartbeat error: {e}")
                
            await asyncio.sleep(self.interval)
            
    def _collect_metrics(self) -> HeartbeatData:
        """Collect process metrics."""
        process = psutil.Process()
        
        with process.oneshot():
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            handle_count = 0
            if hasattr(process, 'num_handles'):
                handle_count = process.num_handles()
                
            return HeartbeatData(
                process_id=self.process_id,
                process_type=self.process_type,
                process_name=process.name(),
                timestamp=time.time(),
                uptime_seconds=time.time() - self.start_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_info.rss / 1024 / 1024,
                thread_count=process.num_threads(),
                handle_count=handle_count,
                status='healthy',
                active_tasks=0,
                completed_tasks=0,
                error_count=0
            )
            
    async def _send_heartbeat(self, heartbeat: HeartbeatData):
        """Send heartbeat to monitor."""
        if self.ipc_client:
            await self.ipc_client.send({
                'type': 'heartbeat',
                'data': heartbeat.__dict__
            })


class HeartbeatMonitor:
    """Monitors heartbeats from all processes."""
    
    def __init__(self, 
                 heartbeat_timeout: int = 30,
                 check_interval: int = 5):
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        self.processes: Dict[str, Dict] = {}
        self.status: Dict[str, str] = {}
        self._callbacks: List[Callable] = []
        self.running = False
        
    async def start(self):
        """Start monitoring."""
        self.running = True
        while self.running:
            await self._check_health()
            await asyncio.sleep(self.check_interval)
            
    def stop(self):
        """Stop monitoring."""
        self.running = False
        
    def register_process(self, process_id: str, process_type: str):
        """Register a process for monitoring."""
        self.processes[process_id] = {
            'type': process_type,
            'last_heartbeat': time.time(),
            'registered_at': time.time()
        }
        self.status[process_id] = 'initializing'
        
    def receive_heartbeat(self, heartbeat: HeartbeatData):
        """Receive a heartbeat from a process."""
        if heartbeat.process_id not in self.processes:
            self.register_process(heartbeat.process_id, heartbeat.process_type)
            
        self.processes[heartbeat.process_id]['last_heartbeat'] = time.time()
        self.status[heartbeat.process_id] = heartbeat.status
        
    async def _check_health(self):
        """Check health of all registered processes."""
        now = time.time()
        
        for process_id, info in list(self.processes.items()):
            last_seen = info['last_heartbeat']
            time_since = now - last_seen
            
            if time_since > self.heartbeat_timeout:
                await self._handle_failure(process_id, 'heartbeat_timeout')
            elif time_since > self.heartbeat_timeout * 0.5:
                self.status[process_id] = 'degraded'
                
    async def _handle_failure(self, process_id: str, reason: str):
        """Handle a process failure."""
        logger.error(f"Process {process_id} failed: {reason}")
        self.status[process_id] = 'failed'
        
        for callback in self._callbacks:
            try:
                await callback(process_id, reason)
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failure callback error: {e}")
                
    def on_failure(self, callback: Callable):
        """Register a failure callback."""
        self._callbacks.append(callback)


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

class GracefulShutdown:
    """Manages graceful shutdown of the entire system."""
    
    def __init__(self, 
                 shutdown_timeout: int = 60,
                 task_drain_timeout: int = 30):
        self.shutdown_timeout = shutdown_timeout
        self.task_drain_timeout = task_drain_timeout
        self.shutdown_event = asyncio.Event()
        self.components: List[Dict] = []
        
    def register_component(self, name: str, 
                          stop_func: Callable,
                          priority: int = 100):
        """Register a component for graceful shutdown."""
        self.components.append({
            'name': name,
            'stop_func': stop_func,
            'priority': priority
        })
        self.components.sort(key=lambda x: x['priority'])
        
    async def initiate_shutdown(self, reason: str = "unknown"):
        """Initiate graceful shutdown."""
        logger.info(f"Initiating graceful shutdown: {reason}")
        self.shutdown_event.set()
        asyncio.create_task(self._shutdown_sequence())
        
    async def _shutdown_sequence(self):
        """Execute shutdown sequence."""
        start_time = time.time()
        
        try:
            for component in self.components:
                elapsed = time.time() - start_time
                remaining = self.shutdown_timeout - elapsed
                
                if remaining <= 0:
                    logger.warning("Shutdown timeout exceeded")
                    break
                    
                logger.info(f"Stopping component: {component['name']}")
                
                try:
                    await asyncio.wait_for(
                        component['stop_func'](),
                        timeout=min(remaining, 10)
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout stopping {component['name']}")
                except (OSError, RuntimeError) as e:
                    logger.error(f"Error stopping {component['name']}: {e}")
                    
            logger.info("Graceful shutdown complete")
            
        except (OSError, RuntimeError) as e:
            logger.exception(f"Shutdown error: {e}")
        finally:
            asyncio.get_event_loop().call_later(5, lambda: os._exit(0))
            
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()


# =============================================================================
# WINDOWS JOB OBJECTS (Process Isolation)
# =============================================================================

class JobObjectManager:
    """Manages Windows Job Objects for process isolation."""
    
    def __init__(self, job_name: str = None):
        if sys.platform != 'win32':
            raise RuntimeError("Job Objects are Windows-only")
        _require_win32()
        self.job_name = job_name or f"OpenClawJob_{os.getpid()}"
        self.job_handle = None
        self._create_job()
        
    def _create_job(self):
        """Create a Windows Job Object."""
        self.job_handle = win32job.CreateJobObject(None, self.job_name)
        
        # Set job limits
        info = win32job.QueryInformationJobObject(
            self.job_handle, 
            win32job.JobObjectExtendedLimitInformation
        )
        
        info['BasicLimitInformation']['LimitFlags'] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY |
            win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
            win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            win32job.JOB_OBJECT_LIMIT_PRIORITY_CLASS |
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        
        # Memory limits
        info['ProcessMemoryLimit'] = 2 * 1024 * 1024 * 1024  # 2GB per process
        info['JobMemoryLimit'] = 8 * 1024 * 1024 * 1024      # 8GB total
        
        # Max active processes
        info['BasicLimitInformation']['ActiveProcessLimit'] = 100
        
        # Priority class
        info['BasicLimitInformation']['PriorityClass'] = win32process.NORMAL_PRIORITY_CLASS
        
        win32job.SetInformationJobObject(
            self.job_handle,
            win32job.JobObjectExtendedLimitInformation,
            info
        )
        
    def assign_process(self, pid: int):
        """Assign a process to this job."""
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        win32job.AssignProcessToJobObject(self.job_handle, handle)
        win32api.CloseHandle(handle)
        
    def terminate(self):
        """Terminate all processes in the job."""
        win32job.TerminateJobObject(self.job_handle, 1)
        win32api.CloseHandle(self.job_handle)


# =============================================================================
# RESTART MANAGER
# =============================================================================

class RestartManager:
    """Manages restart logic with exponential backoff and history tracking."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0,
                 max_restarts: int = 10, window_seconds: float = 300.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_restarts = max_restarts
        self.window_seconds = window_seconds
        self.restart_history: Dict[str, List[float]] = {}

    def record_restart(self, child_id: str) -> None:
        """Record a restart event for a child."""
        now = time.time()
        if child_id not in self.restart_history:
            self.restart_history[child_id] = []
        self.restart_history[child_id].append(now)

    def _prune_history(self, child_id: str) -> List[float]:
        """Remove restart records older than the window."""
        now = time.time()
        history = self.restart_history.get(child_id, [])
        history = [t for t in history if now - t < self.window_seconds]
        self.restart_history[child_id] = history
        return history

    def can_restart(self, child_id: str) -> bool:
        """Return True if the child has not exceeded max restarts within the window."""
        history = self._prune_history(child_id)
        return len(history) < self.max_restarts

    def get_backoff_delay(self, child_id: str) -> float:
        """Calculate exponential backoff delay based on recent restart count."""
        history = self._prune_history(child_id)
        attempt = len(history)
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay

    def reset(self, child_id: str) -> None:
        """Reset restart history for a child."""
        self.restart_history.pop(child_id, None)


# =============================================================================
# SYSTEM SUPERVISOR (Main Entry Point)
# =============================================================================

class SystemSupervisor:
    """Root supervisor for the entire AI agent system."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.child_supervisors: Dict[str, Supervisor] = {}
        self.shutdown_handler = GracefulShutdown()
        self.heartbeat_monitor = HeartbeatMonitor()
        self.restart_manager = RestartManager()
        self.worker_pools: Dict[str, WorkerPool] = {}
        self.running = False
        
    async def start(self):
        """Start the entire system."""
        logger.info("Starting System Supervisor")
        self.running = True
        
        # Start heartbeat monitoring
        asyncio.create_task(self.heartbeat_monitor.start())
        
        # Register shutdown handler
        self.shutdown_handler.register_component(
            'system_supervisor',
            self._stop_children,
            priority=0
        )
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("System Supervisor started successfully")
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if sys.platform == 'win32' and win32api is not None:
            try:
                def console_handler(ctrl_type):
                    if ctrl_type in (0, 2, 6):
                        asyncio.create_task(
                            self.shutdown_handler.initiate_shutdown("signal")
                        )
                        return True
                    return False
                win32api.SetConsoleCtrlHandler(console_handler, True)
            except (OSError, ImportError) as e:
                logger.warning(f"Failed to set console ctrl handler: {e}")
                
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Unix signals."""
        signal_name = signal.Signals(signum).name
        asyncio.create_task(
            self.shutdown_handler.initiate_shutdown(signal_name)
        )
        
    async def _stop_children(self):
        """Stop all child supervisors."""
        for name, supervisor in self.child_supervisors.items():
            logger.info(f"Stopping {name} supervisor")
            await supervisor.stop()
            
        for name, pool in self.worker_pools.items():
            logger.info(f"Stopping {name} worker pool")
            await pool.stop()
            
    async def stop(self):
        """Stop the system."""
        self.running = False
        await self.shutdown_handler.initiate_shutdown("manual")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of how to use the process management system."""
    
    # Create system supervisor
    system = SystemSupervisor()
    
    # Create worker pools
    cpu_pool = PreforkWorkerPool(
        name="CPUWorkers",
        min_workers=4,
        max_tasks_per_child=1000
    )
    
    io_pool = ThreadWorkerPool(
        name="IOWorkers",
        min_workers=10,
        max_workers=50
    )
    
    async_pool = AsyncWorkerPool(
        name="AsyncWorkers",
        max_concurrency=500
    )
    
    # Start pools
    await cpu_pool.start()
    await io_pool.start()
    
    # Register with system
    system.worker_pools['cpu'] = cpu_pool
    system.worker_pools['io'] = io_pool
    
    # Create supervisors
    core_supervisor = OneForOneSupervisor(max_restarts=5)
    agent_supervisor = RestForOneSupervisor(max_restarts=3)
    
    # Add children to supervisors
    # (In production, these would be actual process start functions)
    
    # Start the system
    await system.start()
    
    # Submit work
    def example_task(x):
        return x * x
        
    task_id = cpu_pool.submit(example_task, 5)
    logger.info(f"Submitted task: {task_id}")
    
    # Let it run
    await asyncio.sleep(10)
    
    # Graceful shutdown
    await system.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
