"""
Resource Allocation System for OpenClaw AI Agent System
Manages CPU, memory, and API quota allocation across users and tasks
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources"""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    DISK_IO_MBPS = "disk_io_mbps"
    NETWORK_MBPS = "network_mbps"
    GPT52_TOKENS_PER_MIN = "gpt52_tokens_per_min"
    GPT52_REQUESTS_PER_MIN = "gpt52_requests_per_min"
    BROWSER_INSTANCES = "browser_instances"
    CONCURRENT_TASKS = "concurrent_tasks"


class UserTier(Enum):
    """User tiers with different resource allocations"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TaskPriority(Enum):
    """Task priorities affecting resource allocation"""
    P0_EMERGENCY = "P0"
    P1_HIGH = "P1"
    P2_NORMAL = "P2"
    P3_LOW = "P3"
    P4_BACKGROUND = "P4"


@dataclass
class ResourceQuota:
    """Resource quota for a user or task"""
    resource_type: ResourceType
    limit: float
    burst: float = 0.0  # Temporary burst allowance
    window_seconds: int = 60  # Rate limit window


@dataclass
class ResourceAllocation:
    """Current resource allocation"""
    user_id: str
    task_id: str
    resource_type: ResourceType
    allocated: float
    used: float = 0.0
    reserved: float = 0.0
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: float
    tokens: float
    last_update: float
    refill_rate: float  # tokens per second


class ResourceQuotaManager:
    """
    Manages resource quotas for users and tasks
    """
    
    # Default quotas by user tier
    DEFAULT_TIER_QUOTAS = {
        UserTier.FREE: {
            ResourceType.CPU_CORES: 0.5,
            ResourceType.MEMORY_GB: 1.0,
            ResourceType.DISK_IO_MBPS: 10.0,
            ResourceType.NETWORK_MBPS: 10.0,
            ResourceType.GPT52_TOKENS_PER_MIN: 100,
            ResourceType.GPT52_REQUESTS_PER_MIN: 10,
            ResourceType.BROWSER_INSTANCES: 1,
            ResourceType.CONCURRENT_TASKS: 2,
        },
        UserTier.BASIC: {
            ResourceType.CPU_CORES: 1.0,
            ResourceType.MEMORY_GB: 2.0,
            ResourceType.DISK_IO_MBPS: 20.0,
            ResourceType.NETWORK_MBPS: 20.0,
            ResourceType.GPT52_TOKENS_PER_MIN: 500,
            ResourceType.GPT52_REQUESTS_PER_MIN: 30,
            ResourceType.BROWSER_INSTANCES: 2,
            ResourceType.CONCURRENT_TASKS: 5,
        },
        UserTier.PREMIUM: {
            ResourceType.CPU_CORES: 2.0,
            ResourceType.MEMORY_GB: 4.0,
            ResourceType.DISK_IO_MBPS: 50.0,
            ResourceType.NETWORK_MBPS: 50.0,
            ResourceType.GPT52_TOKENS_PER_MIN: 2000,
            ResourceType.GPT52_REQUESTS_PER_MIN: 100,
            ResourceType.BROWSER_INSTANCES: 5,
            ResourceType.CONCURRENT_TASKS: 10,
        },
        UserTier.ENTERPRISE: {
            ResourceType.CPU_CORES: 4.0,
            ResourceType.MEMORY_GB: 8.0,
            ResourceType.DISK_IO_MBPS: 100.0,
            ResourceType.NETWORK_MBPS: 100.0,
            ResourceType.GPT52_TOKENS_PER_MIN: 10000,
            ResourceType.GPT52_REQUESTS_PER_MIN: 500,
            ResourceType.BROWSER_INSTANCES: 10,
            ResourceType.CONCURRENT_TASKS: 20,
        },
    }
    
    # Priority multipliers
    PRIORITY_MULTIPLIERS = {
        TaskPriority.P0_EMERGENCY: 4.0,
        TaskPriority.P1_HIGH: 2.0,
        TaskPriority.P2_NORMAL: 1.0,
        TaskPriority.P3_LOW: 0.5,
        TaskPriority.P4_BACKGROUND: 0.25,
    }
    
    def __init__(self):
        # User tier mapping
        self.user_tiers: Dict[str, UserTier] = {}
        
        # Custom quotas (override tier defaults)
        self.custom_quotas: Dict[str, Dict[ResourceType, float]] = defaultdict(dict)
        
        # Token buckets for rate limiting
        self.token_buckets: Dict[Tuple[str, ResourceType], TokenBucket] = {}
        
        # Lock
        self._lock = threading.RLock()
        
        logger.info("ResourceQuotaManager initialized")
    
    def set_user_tier(self, user_id: str, tier: UserTier) -> None:
        """Set the tier for a user"""
        with self._lock:
            self.user_tiers[user_id] = tier
            logger.info(f"Set user {user_id} tier to {tier.value}")
    
    def set_custom_quota(self, 
                        user_id: str, 
                        resource_type: ResourceType, 
                        quota: float) -> None:
        """Set a custom quota for a user"""
        with self._lock:
            self.custom_quotas[user_id][resource_type] = quota
            logger.info(
                f"Set custom quota for {user_id}: {resource_type.value} = {quota}"
            )
    
    def get_quota(self, 
                  user_id: str, 
                  resource_type: ResourceType,
                  priority: TaskPriority = TaskPriority.P2_NORMAL) -> float:
        """
        Get the quota for a user and resource type
        
        Applies priority multiplier to base quota
        """
        with self._lock:
            # Check for custom quota
            if resource_type in self.custom_quotas.get(user_id, {}):
                base_quota = self.custom_quotas[user_id][resource_type]
            else:
                # Get tier quota
                tier = self.user_tiers.get(user_id, UserTier.FREE)
                base_quota = self.DEFAULT_TIER_QUOTAS[tier].get(resource_type, 0)
            
            # Apply priority multiplier
            multiplier = self.PRIORITY_MULTIPLIERS.get(priority, 1.0)
            
            return base_quota * multiplier
    
    def get_all_quotas(self, 
                       user_id: str,
                       priority: TaskPriority = TaskPriority.P2_NORMAL) -> Dict[ResourceType, float]:
        """Get all quotas for a user"""
        return {
            resource_type: self.get_quota(user_id, resource_type, priority)
            for resource_type in ResourceType
        }
    
    def check_rate_limit(self, 
                        user_id: str, 
                        resource_type: ResourceType,
                        requested: float) -> Tuple[bool, float, float]:
        """
        Check if a request is within rate limit
        
        Returns:
            (allowed, remaining, reset_time)
        """
        with self._lock:
            key = (user_id, resource_type)
            quota = self.get_quota(user_id, resource_type)
            
            # Get or create token bucket
            if key not in self.token_buckets:
                self.token_buckets[key] = TokenBucket(
                    capacity=quota,
                    tokens=quota,
                    last_update=time.time(),
                    refill_rate=quota / 60  # Refill per minute
                )
            
            bucket = self.token_buckets[key]
            
            # Refill tokens
            now = time.time()
            elapsed = now - bucket.last_update
            bucket.tokens = min(
                bucket.capacity,
                bucket.tokens + (elapsed * bucket.refill_rate)
            )
            bucket.last_update = now
            
            # Check if request can be satisfied
            if bucket.tokens >= requested:
                bucket.tokens -= requested
                return True, bucket.tokens, now + (bucket.capacity / bucket.refill_rate)
            else:
                # Calculate when enough tokens will be available
                tokens_needed = requested - bucket.tokens
                wait_time = tokens_needed / bucket.refill_rate
                return False, bucket.tokens, now + wait_time


class DynamicResourceAllocator:
    """
    Dynamic resource allocator that manages real-time resource distribution
    """
    
    def __init__(self, quota_manager: ResourceQuotaManager):
        self.quota_manager = quota_manager
        
        # Active allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Resource usage tracking
        self.usage_by_user: Dict[str, Dict[ResourceType, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.usage_by_task: Dict[str, Dict[ResourceType, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # System-wide resource pool
        self.system_resources: Dict[ResourceType, float] = {
            ResourceType.CPU_CORES: 16.0,
            ResourceType.MEMORY_GB: 64.0,
            ResourceType.DISK_IO_MBPS: 500.0,
            ResourceType.NETWORK_MBPS: 1000.0,
            ResourceType.GPT52_TOKENS_PER_MIN: 100000,
            ResourceType.GPT52_REQUESTS_PER_MIN: 1000,
            ResourceType.BROWSER_INSTANCES: 20,
            ResourceType.CONCURRENT_TASKS: 100,
        }
        
        # Reserved resources (system overhead)
        self.reserved_resources: Dict[ResourceType, float] = {
            ResourceType.CPU_CORES: 2.0,
            ResourceType.MEMORY_GB: 4.0,
            ResourceType.DISK_IO_MBPS: 50.0,
            ResourceType.NETWORK_MBPS: 100.0,
        }
        
        # Lock
        self._lock = threading.RLock()
        
        logger.info("DynamicResourceAllocator initialized")
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get currently available resources"""
        with self._lock:
            available = {}
            
            for resource_type in ResourceType:
                total = self.system_resources.get(resource_type, 0)
                reserved = self.reserved_resources.get(resource_type, 0)
                
                # Calculate used resources
                used = sum(
                    alloc.allocated 
                    for alloc in self.allocations.values()
                    if alloc.resource_type == resource_type
                )
                
                available[resource_type] = max(0, total - reserved - used)
            
            return available
    
    def allocate(self,
                user_id: str,
                task_id: str,
                resource_type: ResourceType,
                requested: float,
                priority: TaskPriority = TaskPriority.P2_NORMAL,
                timeout_seconds: Optional[float] = None) -> Optional[ResourceAllocation]:
        """
        Allocate resources for a task
        
        Args:
            user_id: User requesting resources
            task_id: Task ID
            resource_type: Type of resource
            requested: Amount requested
            priority: Task priority
            timeout_seconds: Optional allocation timeout
            
        Returns:
            ResourceAllocation if successful, None otherwise
        """
        with self._lock:
            # Check user quota
            quota = self.quota_manager.get_quota(user_id, resource_type, priority)
            current_usage = self.usage_by_user[user_id][resource_type]
            
            if current_usage + requested > quota:
                logger.warning(
                    f"Quota exceeded for {user_id}: {resource_type.value} "
                    f"(requested: {requested}, quota: {quota}, used: {current_usage})"
                )
                return None
            
            # Check system availability
            available = self.get_available_resources()[resource_type]
            
            if requested > available:
                logger.warning(
                    f"Insufficient {resource_type.value}: "
                    f"requested {requested}, available {available}"
                )
                return None
            
            # Create allocation
            allocation_id = f"{user_id}:{task_id}:{resource_type.value}:{time.time()}"
            
            expires_at = None
            if timeout_seconds:
                expires_at = time.time() + timeout_seconds
            
            allocation = ResourceAllocation(
                user_id=user_id,
                task_id=task_id,
                resource_type=resource_type,
                allocated=requested,
                expires_at=expires_at
            )
            
            self.allocations[allocation_id] = allocation
            self.usage_by_user[user_id][resource_type] += requested
            self.usage_by_task[task_id][resource_type] += requested
            
            logger.info(
                f"Allocated {requested} {resource_type.value} to {user_id}/{task_id}"
            )
            
            return allocation
    
    def release(self, allocation_id: str) -> bool:
        """Release a resource allocation"""
        with self._lock:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            # Update usage tracking
            self.usage_by_user[allocation.user_id][allocation.resource_type] -= allocation.allocated
            self.usage_by_task[allocation.task_id][allocation.resource_type] -= allocation.allocated
            
            # Remove allocation
            del self.allocations[allocation_id]
            
            logger.info(
                f"Released {allocation.allocated} {allocation.resource_type.value} "
                f"from {allocation.user_id}/{allocation.task_id}"
            )
            
            return True
    
    def update_usage(self, 
                    allocation_id: str, 
                    used: float) -> bool:
        """Update actual usage for an allocation"""
        with self._lock:
            if allocation_id not in self.allocations:
                return False
            
            self.allocations[allocation_id].used = used
            return True
    
    def get_user_usage(self, user_id: str) -> Dict[ResourceType, float]:
        """Get current resource usage for a user"""
        with self._lock:
            return dict(self.usage_by_user[user_id])
    
    def get_task_usage(self, task_id: str) -> Dict[ResourceType, float]:
        """Get current resource usage for a task"""
        with self._lock:
            return dict(self.usage_by_task[task_id])
    
    def cleanup_expired(self) -> int:
        """Clean up expired allocations, returns count cleaned"""
        with self._lock:
            now = time.time()
            expired = [
                alloc_id for alloc_id, alloc in self.allocations.items()
                if alloc.expires_at and alloc.expires_at < now
            ]
            
            for alloc_id in expired:
                self.release(alloc_id)
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired allocations")
            
            return len(expired)
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of all allocations"""
        with self._lock:
            return {
                "total_allocations": len(self.allocations),
                "resources_allocated": {
                    rt.value: sum(
                        a.allocated for a in self.allocations.values()
                        if a.resource_type == rt
                    )
                    for rt in ResourceType
                },
                "resources_available": {
                    rt.value: avail 
                    for rt, avail in self.get_available_resources().items()
                },
                "users_active": len(set(
                    a.user_id for a in self.allocations.values()
                )),
                "tasks_active": len(set(
                    a.task_id for a in self.allocations.values()
                )),
            }


class ResourceThrottler:
    """
    Throttles requests when resources are constrained
    """
    
    def __init__(self, allocator: DynamicResourceAllocator):
        self.allocator = allocator
        
        # Throttling configuration
        self.throttle_delays: Dict[TaskPriority, Tuple[float, float]] = {
            # (initial_delay_ms, max_delay_ms)
            TaskPriority.P0_EMERGENCY: (0, 0),
            TaskPriority.P1_HIGH: (0, 100),
            TaskPriority.P2_NORMAL: (100, 1000),
            TaskPriority.P3_LOW: (500, 5000),
            TaskPriority.P4_BACKGROUND: (1000, 30000),
        }
        
        # Throttling state
        self.throttle_state: Dict[str, Dict] = {}
    
    def check_and_throttle(self,
                          user_id: str,
                          task_id: str,
                          resource_type: ResourceType,
                          requested: float,
                          priority: TaskPriority = TaskPriority.P2_NORMAL) -> Dict:
        """
        Check if request should be throttled
        
        Returns:
            Dict with throttling decision and delay
        """
        # Check if allocation is possible
        allocation = self.allocator.allocate(
            user_id, task_id, resource_type, requested, priority
        )
        
        if allocation:
            # Extract the actual allocation ID from the allocations map
            allocation_id = None
            for alloc_id, alloc_obj in self.allocator.allocations.items():
                if (alloc_obj.user_id == user_id
                        and alloc_obj.task_id == task_id
                        and alloc_obj.resource_type == resource_type
                        and alloc_obj.allocated == requested):
                    allocation_id = alloc_id
                    break

            return {
                "allowed": True,
                "allocation_id": allocation_id or f"{user_id}:{task_id}:{resource_type.value}",
                "throttle_delay_ms": 0,
            }
        
        # Calculate throttle delay
        initial_delay, max_delay = self.throttle_delays.get(
            priority, (100, 1000)
        )
        
        # Get current utilization
        available = self.allocator.get_available_resources()[resource_type]
        total = self.allocator.system_resources.get(resource_type, 1)
        utilization = 1 - (available / total) if total > 0 else 0
        
        # Calculate delay based on utilization
        if utilization < 0.8:
            delay = initial_delay
        elif utilization < 0.9:
            delay = initial_delay + (max_delay - initial_delay) * 0.5
        else:
            delay = max_delay
        
        return {
            "allowed": False,
            "throttle_delay_ms": delay,
            "reason": "insufficient_resources",
            "utilization": utilization,
            "retry_after": time.time() + (delay / 1000)
        }


# Example usage
if __name__ == "__main__":
    # Create quota manager
    quota_manager = ResourceQuotaManager()
    
    # Set user tiers
    quota_manager.set_user_tier("user1", UserTier.BASIC)
    quota_manager.set_user_tier("user2", UserTier.PREMIUM)
    
    # Create allocator
    allocator = DynamicResourceAllocator(quota_manager)
    
    # Create throttler
    throttler = ResourceThrottler(allocator)
    
    # Test allocations
    print("User1 quotas:", quota_manager.get_all_quotas("user1"))
    print("User2 quotas:", quota_manager.get_all_quotas("user2"))
    
    # Try to allocate resources
    alloc1 = allocator.allocate(
        "user1", "task1", ResourceType.CPU_CORES, 0.5, TaskPriority.P2_NORMAL
    )
    print(f"Allocation 1: {alloc1}")
    
    alloc2 = allocator.allocate(
        "user2", "task2", ResourceType.GPT52_TOKENS_PER_MIN, 1000, TaskPriority.P1_HIGH
    )
    print(f"Allocation 2: {alloc2}")
    
    # Get summary
    print("\nAllocation Summary:")
    print(json.dumps(allocator.get_allocation_summary(), indent=2))
    
    # Test throttling
    result = throttler.check_and_throttle(
        "user1", "task3", ResourceType.MEMORY_GB, 10, TaskPriority.P2_NORMAL
    )
    print("\nThrottle check:", result)
