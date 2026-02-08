"""
Dynamic Weighted Load Balancer for OpenClaw AI Agent System
Implements intelligent request distribution across agent instances
"""

import random
import time
import psutil
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentInstance:
    """Represents a single agent instance"""
    instance_id: str
    host: str
    port: int
    capacity: float = 1.0
    weight: float = 1.0
    health_status: str = "healthy"
    last_health_check: float = field(default_factory=time.time)
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    avg_response_time: float = 0.0
    avg_thinking_time: float = 0.0
    request_count: int = 0
    error_count: int = 0


class NoHealthyInstancesError(Exception):
    """Raised when no healthy instances are available"""
    pass


class DynamicWeightedLoadBalancer:
    """
    Dynamic weighted load balancer for GPT-5.2 agent instances
    
    Features:
    - Dynamic weight calculation based on real-time metrics
    - Health-aware routing
    - GPT-5.2 thinking time consideration
    - Session affinity support
    """
    
    def __init__(self, 
                 health_check_interval: int = 10,
                 weight_update_interval: int = 5):
        self.instances: Dict[str, AgentInstance] = {}
        self.weights: Dict[str, float] = {}
        self.health_status: Dict[str, str] = {}
        
        # Session affinity mapping
        self.session_affinity: Dict[str, str] = {}
        
        # Configuration
        self.health_check_interval = health_check_interval
        self.weight_update_interval = weight_update_interval
        
        # Background threads
        self._stop_event = threading.Event()
        self._health_check_thread = None
        self._weight_update_thread = None
        
        # Locks
        self._instance_lock = threading.RLock()
        
        # Custom weight calculators
        self._weight_calculators: Dict[str, Callable] = {}
        
        logger.info("Load balancer initialized")
    
    def register_instance(self, instance: AgentInstance) -> bool:
        """Register a new agent instance"""
        with self._instance_lock:
            self.instances[instance.instance_id] = instance
            self.health_status[instance.instance_id] = "healthy"
            self.weights[instance.instance_id] = instance.capacity
            
            logger.info(f"Registered instance {instance.instance_id} at "
                       f"{instance.host}:{instance.port}")
            return True
    
    def unregister_instance(self, instance_id: str) -> bool:
        """Unregister an agent instance"""
        with self._instance_lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                del self.health_status[instance_id]
                del self.weights[instance_id]
                
                # Remove from session affinity
                sessions_to_remove = [
                    session for session, inst in self.session_affinity.items()
                    if inst == instance_id
                ]
                for session in sessions_to_remove:
                    del self.session_affinity[session]
                
                logger.info(f"Unregistered instance {instance_id}")
                return True
            return False
    
    def update_instance_metrics(self, instance_id: str, 
                                 cpu_usage: float = None,
                                 memory_usage: float = None,
                                 active_connections: int = None,
                                 avg_response_time: float = None,
                                 avg_thinking_time: float = None):
        """Update metrics for an instance"""
        with self._instance_lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                if cpu_usage is not None:
                    instance.cpu_usage = cpu_usage
                if memory_usage is not None:
                    instance.memory_usage = memory_usage
                if active_connections is not None:
                    instance.active_connections = active_connections
                if avg_response_time is not None:
                    instance.avg_response_time = avg_response_time
                if avg_thinking_time is not None:
                    instance.avg_thinking_time = avg_thinking_time
    
    def calculate_weight(self, instance_id: str) -> float:
        """
        Calculate dynamic weight for an instance
        
        Weight factors:
        - Base capacity
        - CPU usage (inverse)
        - Memory usage (inverse)
        - Response time (inverse)
        - Active connections (inverse)
        - GPT-5.2 thinking time (inverse)
        """
        with self._instance_lock:
            if instance_id not in self.instances:
                return 0.0
            
            instance = self.instances[instance_id]
            
            # Base weight from capacity
            base_weight = instance.capacity
            
            # CPU factor (lower CPU = higher weight)
            cpu_factor = max(0.1, 1 - (instance.cpu_usage / 100))
            
            # Memory factor (lower memory = higher weight)
            memory_factor = max(0.1, 1 - (instance.memory_usage / 100))
            
            # Response time factor (lower response time = higher weight)
            response_factor = 1 / (1 + instance.avg_response_time / 1000)
            
            # Connection factor (fewer connections = higher weight)
            connection_factor = 1 / (1 + instance.active_connections / 10)
            
            # GPT-5.2 thinking time factor (shorter thinking = higher weight)
            thinking_factor = 1 / (1 + instance.avg_thinking_time / 5000)
            
            # Calculate final weight
            weight = (base_weight * 
                     cpu_factor * 
                     memory_factor * 
                     response_factor * 
                     connection_factor * 
                     thinking_factor)
            
            return max(weight, 0.1)  # Minimum weight to avoid zero
    
    def select_instance(self, 
                       request_type: str = "general",
                       session_id: Optional[str] = None,
                       prefer_healthy: bool = True) -> str:
        """
        Select an instance for a request
        
        Args:
            request_type: Type of request (affects routing strategy)
            session_id: Optional session ID for affinity
            prefer_healthy: Whether to prefer healthy instances
            
        Returns:
            Selected instance ID
            
        Raises:
            NoHealthyInstancesError: If no healthy instances available
        """
        with self._instance_lock:
            # Check session affinity
            if session_id and session_id in self.session_affinity:
                affinity_instance = self.session_affinity[session_id]
                if affinity_instance in self.instances:
                    if (not prefer_healthy or 
                        self.health_status[affinity_instance] == "healthy"):
                        return affinity_instance
            
            # Filter healthy instances
            if prefer_healthy:
                healthy = [
                    i for i in self.instances
                    if self.health_status[i] == "healthy"
                ]
            else:
                healthy = list(self.instances.keys())
            
            if not healthy:
                raise NoHealthyInstancesError("No healthy instances available")
            
            # Calculate current weights
            for instance_id in healthy:
                self.weights[instance_id] = self.calculate_weight(instance_id)
            
            # Weighted random selection
            total_weight = sum(
                self.weights[i] for i in healthy
            )
            
            if total_weight == 0:
                # Fallback to round-robin if all weights are zero
                return healthy[0]
            
            pick = random.uniform(0, total_weight)
            current = 0
            
            for instance_id in healthy:
                current += self.weights[instance_id]
                if current >= pick:
                    # Update session affinity if session_id provided
                    if session_id:
                        self.session_affinity[session_id] = instance_id
                    return instance_id
            
            return healthy[-1]  # Fallback
    
    def get_all_instances(self) -> Dict[str, AgentInstance]:
        """Get all registered instances"""
        with self._instance_lock:
            return dict(self.instances)
    
    def get_healthy_instances(self) -> List[str]:
        """Get list of healthy instance IDs"""
        with self._instance_lock:
            return [
                i for i, status in self.health_status.items()
                if status == "healthy"
            ]
    
    def get_instance_metrics(self, instance_id: str) -> Optional[Dict]:
        """Get metrics for a specific instance"""
        with self._instance_lock:
            if instance_id in self.instances:
                inst = self.instances[instance_id]
                return {
                    "instance_id": inst.instance_id,
                    "host": inst.host,
                    "port": inst.port,
                    "health_status": inst.health_status,
                    "cpu_usage": inst.cpu_usage,
                    "memory_usage": inst.memory_usage,
                    "active_connections": inst.active_connections,
                    "avg_response_time": inst.avg_response_time,
                    "avg_thinking_time": inst.avg_thinking_time,
                    "request_count": inst.request_count,
                    "error_count": inst.error_count,
                    "weight": self.weights.get(instance_id, 0)
                }
            return None
    
    def get_all_metrics(self) -> Dict:
        """Get metrics for all instances"""
        with self._instance_lock:
            return {
                instance_id: self.get_instance_metrics(instance_id)
                for instance_id in self.instances
            }
    
    def start(self):
        """Start background threads"""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._weight_update_thread = threading.Thread(
            target=self._weight_update_loop,
            daemon=True
        )
        
        self._health_check_thread.start()
        self._weight_update_thread.start()
        
        logger.info("Load balancer background threads started")
    
    def stop(self):
        """Stop background threads"""
        self._stop_event.set()
        
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        if self._weight_update_thread:
            self._weight_update_thread.join(timeout=5)
        
        logger.info("Load balancer background threads stopped")
    
    def _health_check_loop(self):
        """Background thread for health checks"""
        while not self._stop_event.is_set():
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            self._stop_event.wait(self.health_check_interval)
    
    def _weight_update_loop(self):
        """Background thread for weight updates"""
        while not self._stop_event.is_set():
            try:
                self._update_all_weights()
            except Exception as e:
                logger.error(f"Weight update error: {e}")
            
            self._stop_event.wait(self.weight_update_interval)
    
    def _perform_health_checks(self):
        """Perform health checks on all instances"""
        with self._instance_lock:
            for instance_id, instance in self.instances.items():
                # Simple health check based on last update time
                time_since_update = time.time() - instance.last_health_check
                
                if time_since_update > 30:  # No update in 30 seconds
                    self.health_status[instance_id] = "unhealthy"
                    logger.warning(f"Instance {instance_id} marked unhealthy - "
                                 f"no health check in {time_since_update:.1f}s")
                elif instance.error_count > 10:  # Too many errors
                    self.health_status[instance_id] = "degraded"
                    logger.warning(f"Instance {instance_id} marked degraded - "
                                 f"error count: {instance.error_count}")
                else:
                    self.health_status[instance_id] = "healthy"
    
    def _update_all_weights(self):
        """Update weights for all instances"""
        with self._instance_lock:
            for instance_id in self.instances:
                self.weights[instance_id] = self.calculate_weight(instance_id)


class PriorityBasedRouter:
    """
    Priority-based request router
    Routes requests to appropriate instances based on priority
    """
    
    PRIORITY_STRATEGIES = {
        "P0": "least_response_time",      # Voice - fastest response
        "P1": "fastest_available",        # User input - quick
        "P2": "weighted_thinking",        # GPT-5.2 - thinking capacity
        "P3": "round_robin",              # Browser - even distribution
        "P4": "queue_based",              # Email - queue
        "P5": "scheduled",                # Cron - scheduled
        "P6": "best_effort"               # Background - any available
    }
    
    def __init__(self, load_balancer: DynamicWeightedLoadBalancer):
        self.lb = load_balancer
        self._round_robin_index = 0
    
    def route_request(self, priority: str, session_id: Optional[str] = None) -> str:
        """Route a request based on priority"""
        strategy = self.PRIORITY_STRATEGIES.get(priority, "weighted")
        
        if strategy == "least_response_time":
            return self._route_least_response_time()
        elif strategy == "fastest_available":
            return self._route_fastest_available()
        elif strategy == "weighted_thinking":
            return self._route_weighted_thinking()
        elif strategy == "round_robin":
            return self._route_round_robin()
        else:
            return self.lb.select_instance(session_id=session_id)
    
    def _route_least_response_time(self) -> str:
        """Route to instance with lowest response time"""
        instances = self.lb.get_all_instances()
        if not instances:
            raise NoHealthyInstancesError()
        
        return min(
            instances.values(),
            key=lambda i: i.avg_response_time
        ).instance_id
    
    def _route_fastest_available(self) -> str:
        """Route to fastest available instance"""
        return self.lb.select_instance()
    
    def _route_weighted_thinking(self) -> str:
        """Route based on GPT-5.2 thinking capacity"""
        instances = self.lb.get_all_instances()
        if not instances:
            raise NoHealthyInstancesError()
        
        # Prefer instances with lower thinking time
        return min(
            instances.values(),
            key=lambda i: i.avg_thinking_time
        ).instance_id
    
    def _route_round_robin(self) -> str:
        """Route using round-robin"""
        healthy = self.lb.get_healthy_instances()
        if not healthy:
            raise NoHealthyInstancesError()
        
        instance = healthy[self._round_robin_index % len(healthy)]
        self._round_robin_index += 1
        return instance


# Example usage
if __name__ == "__main__":
    # Create load balancer
    lb = DynamicWeightedLoadBalancer()
    
    # Register instances
    for i in range(3):
        instance = AgentInstance(
            instance_id=f"agent-{i+1:03d}",
            host="localhost",
            port=8080 + i,
            capacity=1.0
        )
        lb.register_instance(instance)
    
    # Start background threads
    lb.start()
    
    # Simulate requests
    try:
        for i in range(10):
            instance_id = lb.select_instance(
                request_type="general",
                session_id=f"session-{i % 3}"
            )
            print(f"Request {i+1} routed to {instance_id}")
            time.sleep(0.5)
    finally:
        lb.stop()
