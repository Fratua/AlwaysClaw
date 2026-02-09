# Process Management & Supervision Architecture
## Windows 10 OpenClaw-Inspired AI Agent System
### Technical Specification v1.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Process Supervision Trees](#1-process-supervision-trees)
3. [Worker Pool Management](#2-worker-pool-management)
4. [Process Forking and Spawning](#3-process-forking-and-spawning)
5. [Inter-Process Communication](#4-inter-process-communication)
6. [Process Monitoring and Heartbeats](#5-process-monitoring-and-heartbeats)
7. [Automatic Restart Strategies](#6-automatic-restart-strategies)
8. [Graceful Shutdown Procedures](#7-graceful-shutdown-procedures)
9. [Process Isolation](#8-process-isolation)
10. [Implementation Reference](#implementation-reference)

---

## Executive Summary

This specification defines a comprehensive process management and supervision architecture for a 24/7 AI agent system running on Windows 10. The design adapts Erlang/OTP supervision tree patterns to Windows environments while leveraging Python's multiprocessing and asyncio capabilities.

### Key Design Principles
- **Fault Isolation**: Failures in one component don't cascade
- **Automatic Recovery**: Self-healing through supervision and restart strategies
- **Resource Control**: CPU, memory, and I/O limits per process group
- **Observability**: Comprehensive heartbeat and health monitoring
- **Graceful Degradation**: System continues operating with reduced capacity

---

## 1. Process Supervision Trees

### 1.1 Architecture Overview

```
                                    ┌─────────────────────────────────────┐
                                    │     SYSTEM SUPERVISOR (Root)        │
                                    │  - System lifecycle management      │
                                    │  - Global configuration             │
                                    │  - Shutdown coordination            │
                                    └──────────────┬──────────────────────┘
                                                   │
           ┌───────────────────────────────────────┼───────────────────────────────────────┐
           │                                       │                                       │
           ▼                                       ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐              ┌─────────────────────┐
│  CORE SUPERVISOR    │              │ AGENT SUPERVISOR    │              │ SERVICE SUPERVISOR  │
│  (One-for-One)      │              │ (Rest-for-One)      │              │ (One-for-All)       │
│                     │              │                     │              │                     │
│ - Config Manager    │              │ - Loop Workers (15) │              │ - Gmail Service     │
│ - Identity Manager  │              │ - Cron Scheduler    │              │ - Browser Service   │
│ - User Manager      │              │ - Heartbeat Monitor │              │ - TTS Service       │
│ - Soul Manager      │              │ - State Manager     │              │ - STT Service       │
└──────────┬──────────┘              └──────────┬──────────┘              │ - Twilio Service    │
           │                                    │                        └──────────┬──────────┘
           │                                    │                                   │
           ▼                                    ▼                                   ▼
┌─────────────────────┐              ┌─────────────────────┐              ┌─────────────────────┐
│  HEARTBEAT WORKER   │              │  AGENT LOOP POOL    │              │  WORKER POOLS       │
│  (Permanent)        │              │  (Dynamic)          │              │  (Specialized)      │
│                     │              │                     │              │                     │
│ - System pulse      │              │ - 15 hardcoded      │              │ - I/O Workers       │
│ - Health checks     │              │   agentic loops     │              │ - CPU Workers       │
│ - Alert on failure  │              │ - Dynamic scaling   │              │ - Network Workers   │
└─────────────────────┘              └─────────────────────┘              └─────────────────────┘
```

### 1.2 Supervisor Types and Strategies

#### 1.2.1 One-for-One Strategy
- **Use Case**: Independent workers that can fail individually
- **Behavior**: When a child dies, only that child is restarted
- **Applied To**: Core services, I/O workers, independent agents

```python
class OneForOneSupervisor:
    """
    One-for-One supervision strategy.
    Restarts only the failed child process.
    """
    def __init__(self, max_restarts: int = 5, time_window: int = 60):
        self.max_restarts = max_restarts  # Max restarts in time_window
        self.time_window = time_window    # Seconds
        self.children = {}                # child_id -> ChildSpec
        self.restart_history = []         # Track restart timestamps
    
    def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child process exit."""
        if self._exceeded_max_restarts():
            self._escalate_failure()
            return
        
        child_spec = self.children.get(child_id)
        if child_spec and child_spec.restart_policy != RestartPolicy.TEMPORARY:
            self._restart_child(child_id, child_spec)
    
    def _exceeded_max_restarts(self) -> bool:
        """Check if max restart intensity exceeded."""
        now = time.time()
        # Remove old entries outside time window
        self.restart_history = [t for t in self.restart_history 
                               if now - t < self.time_window]
        return len(self.restart_history) >= self.max_restarts
```

#### 1.2.2 Rest-for-One Strategy
- **Use Case**: Dependent processes where order matters
- **Behavior**: When a child dies, it and all children started after it are restarted
- **Applied To**: Agent loops (each may depend on previous), service stacks

```python
class RestForOneSupervisor:
    """
    Rest-for-One supervision strategy.
    Restarts failed child and all children started after it.
    """
    def __init__(self, children_ordered: List[ChildSpec]):
        self.children = children_ordered  # Order matters!
        self.child_pids = {}              # child_id -> process_id
    
    def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child process exit with dependency awareness."""
        # Find position of failed child
        failed_idx = None
        for idx, spec in enumerate(self.children):
            if spec.id == child_id:
                failed_idx = idx
                break
        
        if failed_idx is None:
            return
        
        # Terminate and restart failed child and all after it
        for spec in self.children[failed_idx:]:
            self._terminate_child(spec.id)
        
        for spec in self.children[failed_idx:]:
            self._start_child(spec)
```

#### 1.2.3 One-for-All Strategy
- **Use Case**: Tightly coupled processes that must stay in sync
- **Behavior**: When one child dies, all children are terminated and restarted
- **Applied To**: Service worker pools, tightly integrated components

```python
class OneForAllSupervisor:
    """
    One-for-All supervision strategy.
    Restarts all children when any child fails.
    """
    def handle_child_exit(self, child_id: str, exit_code: int):
        """Handle child exit - restart all children."""
        # Log the failure
        logger.error(f"Child {child_id} exited with {exit_code}. "
                    f"Restarting all children.")
        
        # Terminate all children
        for cid in list(self.child_pids.keys()):
            self._terminate_child(cid)
        
        # Restart all children in order
        for spec in self.children:
            self._start_child(spec)
```

### 1.3 Child Specifications

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Dict, Any

class RestartPolicy(Enum):
    """Child restart policies."""
    PERMANENT = auto()  # Always restart
    TEMPORARY = auto()  # Never restart
    TRANSIENT = auto()  # Restart only on abnormal exit

class ChildType(Enum):
    """Type of supervised child."""
    WORKER = auto()     # Task-executing process
    SUPERVISOR = auto() # Nested supervisor

@dataclass
class ChildSpec:
    """Specification for a supervised child process."""
    id: str                          # Unique identifier
    start_func: Callable             # Function to start the child
    start_args: tuple = ()           # Arguments for start function
    start_kwargs: Dict[str, Any] = None  # Keyword arguments
    restart_policy: RestartPolicy = RestartPolicy.PERMANENT
    child_type: ChildType = ChildType.WORKER
    shutdown_timeout: int = 5000     # Milliseconds to wait for graceful shutdown
    max_restarts: int = 5            # Max restarts before giving up
    modules: List[str] = None        # Modules this child depends on
```

---

## 2. Worker Pool Management

### 2.1 Pool Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WORKER POOL MANAGER                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Task Queue  │  │  Monitor     │  │  Autoscaler  │              │
│  │  (Priority)  │  │  (Health)    │  │  (Dynamic)   │              │
│  └──────┬───────┘  └──────────────┘  └──────────────┘              │
│         │                                                           │
│  ┌──────┴──────────────────────────────────────────────────────┐   │
│  │                    WORKER POOL                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│  │  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │   ...      │   │
│  │  │[ACTIVE] │ │[ACTIVE] │ │[IDLE]  │ │[BUSY]  │            │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Pool Types

#### 2.2.1 Prefork Pool (CPU-Bound Tasks)
```python
class PreforkWorkerPool:
    """
    Prefork pool for CPU-intensive tasks.
    Uses separate processes to bypass Python GIL.
    """
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = None,
                 max_tasks_per_child: int = 1000):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.max_tasks_per_child = max_tasks_per_child
        self.workers = []
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        
    def start(self):
        """Start the worker pool."""
        for i in range(self.min_workers):
            self._spawn_worker()
    
    def _spawn_worker(self):
        """Spawn a new worker process."""
        worker = multiprocessing.Process(
            target=_worker_loop,
            args=(self.task_queue, self.result_queue, self.max_tasks_per_child),
            name=f"PreforkWorker-{len(self.workers)}"
        )
        worker.daemon = True
        worker.start()
        self.workers.append({
            'process': worker,
            'pid': worker.pid,
            'tasks_completed': 0,
            'status': 'active'
        })
        
    def submit(self, task: Callable, args: tuple = None, 
               kwargs: dict = None) -> str:
        """Submit a task to the pool."""
        task_id = str(uuid.uuid4())
        self.task_queue.put({
            'id': task_id,
            'func': task,
            'args': args or (),
            'kwargs': kwargs or {}
        })
        return task_id
```

#### 2.2.2 Thread Pool (I/O-Bound Tasks)
```python
class ThreadWorkerPool:
    """
    Thread pool for I/O-bound tasks.
    Lightweight, suitable for network operations.
    """
    def __init__(self, 
                 min_workers: int = 5,
                 max_workers: int = 100):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="IOWorker"
        )
        self.futures = {}
        
    def submit(self, task: Callable, *args, **kwargs) -> Future:
        """Submit a task to the thread pool."""
        future = self.executor.submit(task, *args, **kwargs)
        self.futures[future] = {
            'submitted_at': time.time(),
            'task_name': task.__name__
        }
        return future
```

#### 2.2.3 Async Pool (Asyncio-based)
```python
class AsyncWorkerPool:
    """
    Asyncio-based pool for async tasks.
    Highest performance for I/O-bound concurrent operations.
    """
    def __init__(self, max_concurrency: int = 1000):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks = set()
        
    async def submit(self, coro: Coroutine) -> Any:
        """Submit an async task to the pool."""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.tasks.add(task)
            try:
                return await task
            finally:
                self.tasks.discard(task)
```

### 2.3 Pool Selection Matrix

| Pool Type | Best For | Concurrency | Memory | Use Case |
|-----------|----------|-------------|--------|----------|
| Prefork | CPU-bound | Low (cores) | High | GPT inference, data processing |
| Thread | I/O-bound | Medium | Low | File operations, DB queries |
| Async | I/O-bound | High (1000s) | Low | Network requests, web scraping |

### 2.4 Autoscaling

```python
class PoolAutoscaler:
    """
    Dynamic pool scaling based on load.
    """
    def __init__(self, pool: WorkerPool,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 check_interval: int = 30):
        self.pool = pool
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval = check_interval
        self.metrics_history = []
        
    async def run(self):
        """Main autoscaling loop."""
        while True:
            await asyncio.sleep(self.check_interval)
            
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only recent history
            self.metrics_history = self.metrics_history[-10:]
            
            utilization = self._calculate_utilization()
            
            if utilization > self.scale_up_threshold:
                await self._scale_up()
            elif utilization < self.scale_down_threshold:
                await self._scale_down()
    
    def _calculate_utilization(self) -> float:
        """Calculate current pool utilization."""
        if not self.metrics_history:
            return 0.0
        
        recent = self.metrics_history[-3:]
        avg_queue_depth = sum(m['queue_depth'] for m in recent) / len(recent)
        avg_active_workers = sum(m['active_workers'] for m in recent) / len(recent)
        
        if avg_active_workers == 0:
            return 0.0
        
        return min(1.0, avg_queue_depth / avg_active_workers)
```

---

## 3. Process Forking and Spawning

### 3.1 Windows Process Creation

Windows does not support `fork()` - only `spawn`. This has important implications:

```python
import multiprocessing

# Windows requires spawn method
multiprocessing.set_start_method('spawn', force=True)

# Alternative: forkserver (if available, more efficient)
# multiprocessing.set_start_method('forkserver', force=True)
```

### 3.2 Process Spawner

```python
class ProcessSpawner:
    """
    Cross-platform process spawner with Windows optimizations.
    """
    def __init__(self):
        self.spawned_processes = {}
        self._lock = threading.Lock()
        
    def spawn(self, 
              target: Callable,
              args: tuple = (),
              kwargs: dict = None,
              name: str = None,
              daemon: bool = True,
              priority: int = None,
              affinity: List[int] = None) -> multiprocessing.Process:
        """
        Spawn a new process with optional priority and affinity.
        
        Args:
            target: Function to run in new process
            args: Positional arguments
            kwargs: Keyword arguments
            name: Process name
            daemon: Run as daemon process
            priority: Process priority (Windows-specific)
            affinity: CPU affinity mask
        """
        # Use spawn context for Windows compatibility
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
        if sys.platform == 'win32' and priority is not None:
            self._set_windows_priority(priority)
        
        # Set CPU affinity
        if affinity is not None:
            p = psutil.Process()
            p.cpu_affinity(affinity)
        
        # Run the actual target
        try:
            target(*args, **(kwargs or {}))
        except Exception as e:
            logger.exception(f"Process {target.__name__} failed: {e}")
            raise
    
    def _set_windows_priority(self, priority: int):
        """Set Windows process priority."""
        import win32api
        import win32process
        
        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, priority)
```

### 3.3 Priority Classes (Windows)

| Priority Class | Value | Description |
|----------------|-------|-------------|
| IDLE_PRIORITY_CLASS | 64 | Run only when system is idle |
| BELOW_NORMAL_PRIORITY_CLASS | 16384 | Below normal priority |
| NORMAL_PRIORITY_CLASS | 32 | Default priority |
| ABOVE_NORMAL_PRIORITY_CLASS | 32768 | Above normal priority |
| HIGH_PRIORITY_CLASS | 128 | High priority (use with caution) |
| REALTIME_PRIORITY_CLASS | 256 | Real-time (dangerous, avoid) |

### 3.4 CPU Affinity Management

```python
class CPUAffinityManager:
    """
    Manage CPU affinity for process groups.
    """
    def __init__(self):
        self.total_cores = psutil.cpu_count()
        self.affinity_map = {
            'critical': [0],           # Core 0: Critical services
            'agents': list(range(1, max(2, self.total_cores // 2))),  # Agent loops
            'workers': list(range(max(2, self.total_cores // 2), self.total_cores)),  # Workers
        }
    
    def get_affinity(self, process_type: str) -> List[int]:
        """Get CPU affinity for a process type."""
        return self.affinity_map.get(process_type, list(range(self.total_cores)))
    
    def apply_affinity(self, pid: int, process_type: str):
        """Apply CPU affinity to a process."""
        try:
            p = psutil.Process(pid)
            affinity = self.get_affinity(process_type)
            p.cpu_affinity(affinity)
            logger.info(f"Set affinity for PID {pid} ({process_type}): {affinity}")
        except Exception as e:
            logger.warning(f"Failed to set affinity for PID {pid}: {e}")
```

---

## 4. Inter-Process Communication

### 4.1 IPC Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      IPC LAYER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Message   │  │   Shared    │  │   Named     │  │   TCP/     │ │
│  │    Queue    │  │   Memory    │  │    Pipe     │  │   Socket   │ │
│  │  (asyncio)  │  │  (multiproc)│  │  (Windows)  │  │  (gRPC)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │               │        │
│         └────────────────┴────────────────┴───────────────┘        │
│                                    │                                │
│                         ┌──────────┴──────────┐                    │
│                         │   IPC Router/Dispatcher                  │
│                         │   - Message routing  │                    │
│                         │   - Serialization    │                    │
│                         │   - Protocol handling│                    │
│                         └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Named Pipes (Windows Primary IPC)

```python
import win32pipe
import win32file
import pywintypes

class NamedPipeServer:
    """
    Windows Named Pipe server for IPC.
    Primary IPC mechanism for Windows.
    """
    def __init__(self, pipe_name: str, max_instances: int = 10):
        self.pipe_name = f"\\\\.\\pipe\\{pipe_name}"
        self.max_instances = max_instances
        self.clients = {}
        self.running = False
        
    async def start(self):
        """Start the named pipe server."""
        self.running = True
        while self.running:
            try:
                pipe = win32pipe.CreateNamedPipe(
                    self.pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | 
                    win32pipe.PIPE_READMODE_MESSAGE | 
                    win32pipe.PIPE_WAIT,
                    self.max_instances,
                    65536,  # Output buffer
                    65536,  # Input buffer
                    0,      # Default timeout
                    None    # Security attributes
                )
                
                # Wait for client connection
                await asyncio.to_thread(win32pipe.ConnectNamedPipe, pipe, None)
                
                # Handle client in separate task
                asyncio.create_task(self._handle_client(pipe))
                
            except Exception as e:
                logger.error(f"Named pipe error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_client(self, pipe):
        """Handle a connected client."""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = pipe
        
        try:
            while self.running:
                # Read message length (4 bytes)
                data = await asyncio.to_thread(
                    win32file.ReadFile, pipe, 4
                )
                msg_len = struct.unpack('I', data[1])[0]
                
                # Read message
                data = await asyncio.to_thread(
                    win32file.ReadFile, pipe, msg_len
                )
                message = json.loads(data[1].decode('utf-8'))
                
                # Process message
                response = await self._process_message(message)
                
                # Send response
                response_data = json.dumps(response).encode('utf-8')
                await asyncio.to_thread(
                    win32file.WriteFile, 
                    pipe, 
                    struct.pack('I', len(response_data)) + response_data
                )
                
        except Exception as e:
            logger.warning(f"Client {client_id} disconnected: {e}")
        finally:
            win32file.CloseHandle(pipe)
            del self.clients[client_id]


class NamedPipeClient:
    """
    Windows Named Pipe client.
    """
    def __init__(self, pipe_name: str):
        self.pipe_name = f"\\\\.\\pipe\\{pipe_name}"
        self.pipe = None
        
    async def connect(self):
        """Connect to the named pipe server."""
        self.pipe = await asyncio.to_thread(
            win32file.CreateFile,
            self.pipe_name,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,  # No sharing
            None,  # Default security
            win32file.OPEN_EXISTING,
            win32file.FILE_FLAG_OVERLAPPED,
            None
        )
        
        # Set message mode
        win32pipe.SetNamedPipeHandleState(
            self.pipe,
            win32pipe.PIPE_READMODE_MESSAGE,
            None,
            None
        )
    
    async def send(self, message: dict) -> dict:
        """Send a message and receive response."""
        data = json.dumps(message).encode('utf-8')
        
        # Send length + data
        await asyncio.to_thread(
            win32file.WriteFile,
            self.pipe,
            struct.pack('I', len(data)) + data
        )
        
        # Read response
        data = await asyncio.to_thread(win32file.ReadFile, self.pipe, 4)
        resp_len = struct.unpack('I', data[1])[0]
        
        data = await asyncio.to_thread(win32file.ReadFile, self.pipe, resp_len)
        return json.loads(data[1].decode('utf-8'))
```

### 4.3 Message Queue (Cross-Platform)

```python
class IPCMessageQueue:
    """
    Cross-platform message queue using multiprocessing.Queue.
    """
    def __init__(self, maxsize: int = 1000):
        self.queue = multiprocessing.Queue(maxsize=maxsize)
        self.response_queues = {}
        
    def send(self, message: dict, target: str = None) -> str:
        """Send a message to the queue."""
        msg_id = str(uuid.uuid4())
        envelope = {
            'id': msg_id,
            'target': target,
            'timestamp': time.time(),
            'payload': message,
            'reply_to': None  # Set if expecting response
        }
        self.queue.put(envelope)
        return msg_id
    
    def send_and_wait(self, message: dict, target: str, 
                      timeout: float = 30.0) -> dict:
        """Send a message and wait for response."""
        msg_id = str(uuid.uuid4())
        response_queue = multiprocessing.Queue()
        self.response_queues[msg_id] = response_queue
        
        envelope = {
            'id': msg_id,
            'target': target,
            'timestamp': time.time(),
            'payload': message,
            'reply_to': msg_id
        }
        self.queue.put(envelope)
        
        try:
            response = response_queue.get(timeout=timeout)
            return response
        finally:
            del self.response_queues[msg_id]
    
    def reply(self, original_msg_id: str, response: dict):
        """Send a reply to a message."""
        if original_msg_id in self.response_queues:
            self.response_queues[original_msg_id].put(response)
```

### 4.4 Protocol Buffers for Serialization

```protobuf
// ipc_messages.proto
syntax = "proto3";

package openclaw.ipc;

message IPCMessage {
    string message_id = 1;
    string source = 2;
    string target = 3;
    int64 timestamp = 4;
    MessageType type = 5;
    bytes payload = 6;
    string correlation_id = 7;
}

enum MessageType {
    HEARTBEAT = 0;
    COMMAND = 1;
    EVENT = 2;
    RESPONSE = 3;
    ERROR = 4;
}

message HeartbeatMessage {
    string process_id = 1;
    string process_type = 2;
    int64 uptime_seconds = 3;
    double cpu_percent = 4;
    double memory_percent = 5;
    map<string, string> status = 6;
}

message CommandMessage {
    string command = 1;
    map<string, string> args = 2;
    int32 timeout_seconds = 3;
}

message EventMessage {
    string event_type = 1;
    map<string, string> data = 2;
}
```

---

## 5. Process Monitoring and Heartbeats

### 5.1 Heartbeat Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HEARTBEAT SYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐│
│  │   Workers    │────────▶│   Heartbeat  │────────▶│   Monitor    ││
│  │  (All Procs) │  pulse  │   Collector  │  agg    │   & Alert    ││
│  └──────────────┘         └──────────────┘         └──────────────┘│
│         │                         │                         │       │
│         │    Every 5-30s          │                         │       │
│         │                         ▼                         ▼       │
│         │                ┌──────────────┐         ┌──────────────┐  │
│         │                │   Time-Series│         │   Health     │  │
│         └───────────────▶│   Database   │         │   Dashboard  │  │
│                          │  (Metrics)   │         │  (Web UI)    │  │
│                          └──────────────┘         └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Heartbeat Implementation

```python
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
    handle_count: int  # Windows-specific
    status: str  # 'healthy', 'degraded', 'critical'
    active_tasks: int
    completed_tasks: int
    error_count: int
    custom_metrics: Dict[str, Any]

class HeartbeatEmitter:
    """
    Emits heartbeats from worker processes.
    """
    def __init__(self, 
                 process_id: str,
                 process_type: str,
                 interval: int = 10,
                 ipc_client: IPCClient = None):
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
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(self.interval)
    
    def _collect_metrics(self) -> HeartbeatData:
        """Collect process metrics."""
        process = psutil.Process()
        
        with process.oneshot():
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
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
                handle_count=process.num_handles() if hasattr(process, 'num_handles') else 0,
                status='healthy',
                active_tasks=0,  # Set by subclass
                completed_tasks=0,
                error_count=0,
                custom_metrics={}
            )
    
    async def _send_heartbeat(self, heartbeat: HeartbeatData):
        """Send heartbeat to monitor."""
        if self.ipc_client:
            await self.ipc_client.send({
                'type': 'heartbeat',
                'data': heartbeat.__dict__
            })


class HeartbeatMonitor:
    """
    Monitors heartbeats from all processes.
    Detects failures and triggers recovery.
    """
    def __init__(self, 
                 heartbeat_timeout: int = 30,
                 check_interval: int = 5):
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        self.processes = {}  # process_id -> last_heartbeat
        self.status = {}     # process_id -> health_status
        self._callbacks = []  # Failure callbacks
        
    async def start(self):
        """Start monitoring."""
        while True:
            await self._check_health()
            await asyncio.sleep(self.check_interval)
    
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
                # Process appears dead
                await self._handle_failure(process_id, 'heartbeat_timeout')
            elif time_since > self.heartbeat_timeout * 0.5:
                # Process is slow/stressed
                self.status[process_id] = 'degraded'
    
    async def _handle_failure(self, process_id: str, reason: str):
        """Handle a process failure."""
        logger.error(f"Process {process_id} failed: {reason}")
        self.status[process_id] = 'failed'
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                await callback(process_id, reason)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def on_failure(self, callback: Callable):
        """Register a failure callback."""
        self._callbacks.append(callback)
```

### 5.3 Health Check Levels

| Level | Check | Threshold | Action |
|-------|-------|-----------|--------|
| INFO | Heartbeat received | < 10s | Normal operation |
| WARNING | Slow heartbeat | 10-30s | Log warning, monitor closely |
| CRITICAL | Heartbeat timeout | > 30s | Trigger restart |
| FATAL | Multiple failures | 3 restarts | Escalate to supervisor |

---

## 6. Automatic Restart Strategies

### 6.1 Restart Policies

```python
class RestartManager:
    """
    Manages process restart with exponential backoff.
    """
    def __init__(self):
        self.restart_history = {}  # process_id -> [timestamps]
        self.max_restarts = 5
        self.max_restart_window = 300  # 5 minutes
        self.backoff_base = 2
        self.max_backoff = 300  # 5 minutes max
        
    async def restart(self, process_id: str, 
                      start_func: Callable,
                      restart_policy: RestartPolicy) -> bool:
        """
        Attempt to restart a process.
        
        Returns:
            True if restart successful, False if giving up
        """
        if restart_policy == RestartPolicy.TEMPORARY:
            logger.info(f"Process {process_id} is TEMPORARY, not restarting")
            return False
        
        if self._should_give_up(process_id):
            logger.error(f"Giving up on process {process_id} - too many restarts")
            return False
        
        # Calculate backoff delay
        delay = self._calculate_backoff(process_id)
        if delay > 0:
            logger.info(f"Waiting {delay}s before restarting {process_id}")
            await asyncio.sleep(delay)
        
        # Attempt restart
        try:
            await start_func()
            self._record_restart(process_id)
            logger.info(f"Successfully restarted process {process_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restart process {process_id}: {e}")
            return False
    
    def _should_give_up(self, process_id: str) -> bool:
        """Check if we should give up restarting."""
        history = self.restart_history.get(process_id, [])
        now = time.time()
        
        # Count restarts in window
        recent = [t for t in history if now - t < self.max_restart_window]
        return len(recent) >= self.max_restarts
    
    def _calculate_backoff(self, process_id: str) -> int:
        """Calculate exponential backoff delay."""
        history = self.restart_history.get(process_id, [])
        if not history:
            return 0
        
        # Exponential backoff: 2^n seconds
        backoff = self.backoff_base ** len(history)
        return min(backoff, self.max_backoff)
    
    def _record_restart(self, process_id: str):
        """Record a restart attempt."""
        if process_id not in self.restart_history:
            self.restart_history[process_id] = []
        self.restart_history[process_id].append(time.time())
```

### 6.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures.
    """
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = 'closed'  # closed, open, half-open
        self.failure_count = 0
        self.last_failure_time = None
        
    def call(self, func: Callable, *args, **kwargs):
        """Call a function with circuit breaker protection."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.error("Circuit breaker OPENED due to failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
```

---

## 7. Graceful Shutdown Procedures

### 7.1 Shutdown Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRACEFUL SHUTDOWN SEQUENCE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. RECEIVE SHUTDOWN SIGNAL                                         │
│     └─▶ Set shutdown event                                          │
│     └─▶ Log shutdown initiation                                     │
│                                                                     │
│  2. STOP ACCEPTING NEW WORK                                         │
│     └─▶ Close task queue                                            │
│     └─▶ Reject new connections                                      │
│     └─▶ Notify load balancer                                        │
│                                                                     │
│  3. DRAIN IN-PROGRESS WORK                                          │
│     └─▶ Wait for active tasks to complete                           │
│     └─▶ Timeout: 30-60 seconds                                      │
│                                                                     │
│  4. STOP WORKER POOLS                                               │
│     └─▶ Send SIGTERM to workers                                     │
│     └─▶ Wait for graceful exit                                      │
│     └─▶ SIGKILL after timeout                                       │
│                                                                     │
│  5. STOP SUPERVISORS (bottom-up)                                    │
│     └─▶ Stop child supervisors first                                │
│     └─▶ Then parent supervisors                                     │
│                                                                     │
│  6. FINAL CLEANUP                                                   │
│     └─▶ Close IPC channels                                          │
│     └─▶ Release resources                                           │
│     └─▶ Flush logs                                                  │
│     └─▶ Exit                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Shutdown Implementation

```python
class GracefulShutdown:
    """
    Manages graceful shutdown of the entire system.
    """
    def __init__(self, 
                 shutdown_timeout: int = 60,
                 task_drain_timeout: int = 30):
        self.shutdown_timeout = shutdown_timeout
        self.task_drain_timeout = task_drain_timeout
        self.shutdown_event = asyncio.Event()
        self.components = []  # Registered components
        self.shutdown_order = []  # Order to shutdown
        
    def register_component(self, name: str, 
                          stop_func: Callable,
                          priority: int = 100):
        """
        Register a component for graceful shutdown.
        
        Lower priority = stopped first
        """
        self.components.append({
            'name': name,
            'stop_func': stop_func,
            'priority': priority
        })
        # Sort by priority
        self.components.sort(key=lambda x: x['priority'])
    
    async def initiate_shutdown(self, reason: str = "unknown"):
        """Initiate graceful shutdown."""
        logger.info(f"Initiating graceful shutdown: {reason}")
        self.shutdown_event.set()
        
        # Create shutdown task
        asyncio.create_task(self._shutdown_sequence())
    
    async def _shutdown_sequence(self):
        """Execute shutdown sequence."""
        start_time = time.time()
        
        try:
            # Stop components in priority order
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
                except Exception as e:
                    logger.error(f"Error stopping {component['name']}: {e}")
            
            logger.info("Graceful shutdown complete")
            
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
        finally:
            # Force exit after timeout
            asyncio.get_event_loop().call_later(
                5, 
                lambda: os._exit(0)
            )
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()


class WorkerShutdownHandler:
    """
    Handles graceful shutdown for worker processes.
    """
    def __init__(self):
        self.shutdown_requested = False
        self.active_tasks = set()
        self._lock = asyncio.Lock()
        
    def request_shutdown(self):
        """Request shutdown."""
        self.shutdown_requested = True
        
    async def track_task(self, task_id: str):
        """Track an active task."""
        async with self._lock:
            self.active_tasks.add(task_id)
    
    async def untrack_task(self, task_id: str):
        """Untrack a completed task."""
        async with self._lock:
            self.active_tasks.discard(task_id)
    
    async def wait_for_tasks(self, timeout: float = 30.0):
        """Wait for all active tasks to complete."""
        start = time.time()
        
        while time.time() - start < timeout:
            async with self._lock:
                if not self.active_tasks:
                    return True
            
            await asyncio.sleep(0.5)
        
        logger.warning(f"Timeout waiting for tasks: {self.active_tasks}")
        return False
```

### 7.3 Windows Signal Handling

```python
class WindowsSignalHandler:
    """
    Handle Windows signals for graceful shutdown.
    """
    def __init__(self, shutdown_handler: Callable):
        self.shutdown_handler = shutdown_handler
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup signal handlers for Windows."""
        # Windows doesn't support SIGTERM well, use console control handlers
        if sys.platform == 'win32':
            import win32api
            
            def console_handler(ctrl_type):
                if ctrl_type in (0, 2, 6):  # CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT
                    asyncio.create_task(self.shutdown_handler("signal"))
                    return True
                return False
            
            win32api.SetConsoleCtrlHandler(console_handler, True)
        
        # Also handle standard signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Unix signals."""
        signal_name = signal.Signals(signum).name
        asyncio.create_task(self.shutdown_handler(signal_name))
```

---

## 8. Process Isolation

### 8.1 Windows Job Objects

```python
class JobObjectManager:
    """
    Manages Windows Job Objects for process isolation.
    """
    def __init__(self, job_name: str = None):
        self.job_name = job_name or f"OpenClawJob_{os.getpid()}"
        self.job_handle = None
        self._create_job()
    
    def _create_job(self):
        """Create a Windows Job Object."""
        import win32job
        import win32api
        
        self.job_handle = win32job.CreateJobObject(None, self.job_name)
        
        # Set job limits
        info = win32job.QueryInformationJobObject(
            self.job_handle, 
            win32job.JobObjectExtendedLimitInformation
        )
        
        # Configure limits
        info['BasicLimitInformation']['LimitFlags'] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY |
            win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
            win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            win32job.JOB_OBJECT_LIMIT_PRIORITY_CLASS |
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        
        # Memory limits (2GB per process, 4GB total)
        info['ProcessMemoryLimit'] = 2 * 1024 * 1024 * 1024
        info['JobMemoryLimit'] = 4 * 1024 * 1024 * 1024
        
        # Max active processes
        info['BasicLimitInformation']['ActiveProcessLimit'] = 50
        
        # Priority class
        info['BasicLimitInformation']['PriorityClass'] = win32process.NORMAL_PRIORITY_CLASS
        
        win32job.SetInformationJobObject(
            self.job_handle,
            win32job.JobObjectExtendedLimitInformation,
            info
        )
    
    def assign_process(self, pid: int):
        """Assign a process to this job."""
        import win32job
        
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        win32job.AssignProcessToJobObject(self.job_handle, handle)
        win32api.CloseHandle(handle)
    
    def set_cpu_rate(self, percent: int):
        """Set CPU rate limit for the job."""
        import win32job
        
        info = win32job.QueryInformationJobObject(
            self.job_handle,
            win32job.JobObjectCpuRateControlInformation
        )
        
        info['ControlFlags'] = win32job.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
        info['CpuRate'] = percent * 100  # Percent * 100
        
        win32job.SetInformationJobObject(
            self.job_handle,
            win32job.JobObjectCpuRateControlInformation,
            info
        )
    
    def terminate(self):
        """Terminate all processes in the job."""
        import win32job
        win32job.TerminateJobObject(self.job_handle, 1)
        win32api.CloseHandle(self.job_handle)
```

### 8.2 Resource Limits

```python
class ResourceLimiter:
    """
    Apply resource limits to processes.
    """
    def __init__(self):
        self.limits = {}
    
    def set_limits(self, process_type: str,
                   max_memory_mb: int = None,
                   max_cpu_percent: int = None,
                   max_handles: int = None):
        """Set resource limits for a process type."""
        self.limits[process_type] = {
            'max_memory_mb': max_memory_mb,
            'max_cpu_percent': max_cpu_percent,
            'max_handles': max_handles
        }
    
    def enforce_limits(self, pid: int, process_type: str):
        """Enforce resource limits on a process."""
        limits = self.limits.get(process_type)
        if not limits:
            return
        
        try:
            process = psutil.Process(pid)
            
            # Memory limit
            if limits['max_memory_mb']:
                max_bytes = limits['max_memory_mb'] * 1024 * 1024
                # Use Windows Job Object for hard limit
                # Soft limit: monitor and log
            
            # CPU limit (via psutil)
            if limits['max_cpu_percent']:
                # Note: This requires continuous monitoring
                pass
                
        except psutil.NoSuchProcess:
            pass
    
    async def monitor_and_enforce(self):
        """Continuously monitor and enforce limits."""
        while True:
            for pid in psutil.pids():
                try:
                    process = psutil.Process(pid)
                    # Determine process type from name/cmdline
                    process_type = self._get_process_type(process)
                    if process_type:
                        self._check_limits(process, process_type)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            await asyncio.sleep(5)
    
    def _check_limits(self, process: psutil.Process, process_type: str):
        """Check if process exceeds limits."""
        limits = self.limits.get(process_type)
        if not limits:
            return
        
        # Check memory
        if limits['max_memory_mb']:
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > limits['max_memory_mb']:
                logger.warning(
                    f"Process {process.pid} exceeds memory limit: "
                    f"{mem_mb:.0f}MB > {limits['max_memory_mb']}MB"
                )
                # Take action: notify, throttle, or terminate
```

### 8.3 Process Type Resource Limits

| Process Type | Max Memory | Max CPU | Max Handles | Priority |
|--------------|------------|---------|-------------|----------|
| System Supervisor | 512 MB | 10% | 1000 | Normal |
| Agent Loops | 1 GB each | 20% | 5000 | Above Normal |
| I/O Workers | 512 MB | 15% | 10000 | Normal |
| CPU Workers | 2 GB | 80% | 5000 | High |
| Services | 1 GB | 10% | 5000 | Normal |

---

## 9. Implementation Reference

### 9.1 Complete System Supervisor

```python
class SystemSupervisor:
    """
    Root supervisor for the entire AI agent system.
    """
    def __init__(self, config: dict):
        self.config = config
        self.child_supervisors = {}
        self.shutdown_handler = GracefulShutdown()
        self.signal_handler = WindowsSignalHandler(self._on_shutdown_signal)
        self.heartbeat_monitor = HeartbeatMonitor()
        self.restart_manager = RestartManager()
        self.job_manager = JobObjectManager("OpenClawSystem")
        
    async def start(self):
        """Start the entire system."""
        logger.info("Starting System Supervisor")
        
        # Start child supervisors
        await self._start_core_supervisor()
        await self._start_agent_supervisor()
        await self._start_service_supervisor()
        
        # Start monitoring
        asyncio.create_task(self.heartbeat_monitor.start())
        
        # Register shutdown handler
        self.shutdown_handler.register_component(
            'system_supervisor',
            self._stop_children,
            priority=0
        )
        
        logger.info("System Supervisor started successfully")
    
    async def _start_core_supervisor(self):
        """Start core services supervisor."""
        from supervisors import OneForOneSupervisor
        
        supervisor = OneForOneSupervisor(max_restarts=5, time_window=60)
        
        # Add core services
        supervisor.add_child(ChildSpec(
            id='config_manager',
            start_func=self._start_config_manager,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        supervisor.add_child(ChildSpec(
            id='identity_manager',
            start_func=self._start_identity_manager,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        supervisor.add_child(ChildSpec(
            id='user_manager',
            start_func=self._start_user_manager,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        supervisor.add_child(ChildSpec(
            id='soul_manager',
            start_func=self._start_soul_manager,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        await supervisor.start()
        self.child_supervisors['core'] = supervisor
    
    async def _start_agent_supervisor(self):
        """Start agent loops supervisor."""
        from supervisors import RestForOneSupervisor
        
        supervisor = RestForOneSupervisor()
        
        # Add 15 hardcoded agentic loops
        for i in range(1, 16):
            supervisor.add_child(ChildSpec(
                id=f'agent_loop_{i}',
                start_func=self._start_agent_loop,
                start_args=(i,),
                restart_policy=RestartPolicy.TRANSIENT
            ))
        
        # Add supporting services
        supervisor.add_child(ChildSpec(
            id='cron_scheduler',
            start_func=self._start_cron_scheduler,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        supervisor.add_child(ChildSpec(
            id='state_manager',
            start_func=self._start_state_manager,
            restart_policy=RestartPolicy.PERMANENT
        ))
        
        await supervisor.start()
        self.child_supervisors['agent'] = supervisor
    
    async def _start_service_supervisor(self):
        """Start external services supervisor."""
        from supervisors import OneForAllSupervisor
        
        supervisor = OneForAllSupervisor()
        
        # Add service workers
        services = [
            ('gmail_service', self._start_gmail_service),
            ('browser_service', self._start_browser_service),
            ('tts_service', self._start_tts_service),
            ('stt_service', self._start_stt_service),
            ('twilio_service', self._start_twilio_service),
        ]
        
        for service_id, start_func in services:
            supervisor.add_child(ChildSpec(
                id=service_id,
                start_func=start_func,
                restart_policy=RestartPolicy.PERMANENT
            ))
        
        await supervisor.start()
        self.child_supervisors['service'] = supervisor
    
    async def _on_shutdown_signal(self, reason: str):
        """Handle shutdown signal."""
        await self.shutdown_handler.initiate_shutdown(reason)
    
    async def _stop_children(self):
        """Stop all child supervisors."""
        for name, supervisor in self.child_supervisors.items():
            logger.info(f"Stopping {name} supervisor")
            await supervisor.stop()
```

### 9.2 Configuration

```yaml
# process_management.yaml
supervision:
  root_supervisor:
    max_restarts: 10
    time_window: 60
  
  strategies:
    core:
      type: one_for_one
      max_restarts: 5
    agent:
      type: rest_for_one
      max_restarts: 3
    service:
      type: one_for_all
      max_restarts: 5

worker_pools:
  agent_loops:
    min_workers: 15
    max_workers: 15
    pool_type: prefork
    max_tasks_per_child: 10000
  
  io_workers:
    min_workers: 5
    max_workers: 50
    pool_type: thread
  
  async_workers:
    max_concurrency: 1000
    pool_type: async

heartbeat:
  interval: 10
  timeout: 30
  check_interval: 5

restart:
  max_restarts: 5
  max_restart_window: 300
  backoff_base: 2
  max_backoff: 300

isolation:
  job_objects:
    enabled: true
    max_processes: 100
    max_memory_gb: 8
  
  resource_limits:
    agent_loops:
      max_memory_mb: 1024
      max_cpu_percent: 20
    services:
      max_memory_mb: 512
      max_cpu_percent: 10

shutdown:
  timeout: 60
  task_drain_timeout: 30
  sigkill_timeout: 10
```

---

## Appendix A: File Structure

```
openclaw/
├── core/
│   ├── __init__.py
│   ├── supervisor.py          # Base supervisor classes
│   ├── worker_pool.py         # Worker pool implementations
│   ├── process_spawner.py     # Process creation
│   └── shutdown.py            # Graceful shutdown
│
├── ipc/
│   ├── __init__.py
│   ├── named_pipe.py          # Windows named pipes
│   ├── message_queue.py       # Cross-platform queues
│   ├── protocol.py            # Protocol buffers
│   └── router.py              # Message routing
│
├── monitoring/
│   ├── __init__.py
│   ├── heartbeat.py           # Heartbeat system
│   ├── health_check.py        # Health monitoring
│   └── metrics.py             # Metrics collection
│
├── isolation/
│   ├── __init__.py
│   ├── job_object.py          # Windows Job Objects
│   └── resource_limiter.py    # Resource limits
│
├── supervisors/
│   ├── __init__.py
│   ├── system_supervisor.py   # Root supervisor
│   ├── core_supervisor.py     # Core services
│   ├── agent_supervisor.py    # Agent loops
│   └── service_supervisor.py  # External services
│
└── config/
    └── process_management.yaml
```

---

## Appendix B: Dependencies

```txt
# requirements-process.txt
psutil>=5.9.0
pywin32>=227; platform_system=="Windows"
protobuf>=4.0.0
grpcio>=1.50.0
asyncio-mqtt>=0.13.0
```

---

*Document Version: 1.0*
*Last Updated: 2025*
*Target Platform: Windows 10*
*Python Version: 3.10+*
