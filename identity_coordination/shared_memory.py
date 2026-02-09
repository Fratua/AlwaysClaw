"""
Shared Memory and Context Module
===============================

Manages shared memory layers and context propagation across agents.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json
import os
import tempfile


# Shared memory architecture definition
SHARED_MEMORY_ARCHITECTURE = {
    "memory_layers": {
        "ephemeral": {
            "duration": "session",
            "scope": "conversation",
            "persistence": False,
            "access": "all_agents",
            "description": "Temporary context for current interaction",
            "default_ttl": 3600  # 1 hour
        },
        "working": {
            "duration": "task",
            "scope": "multi_agent",
            "persistence": True,
            "access": "participating_agents",
            "description": "Shared context for ongoing tasks",
            "default_ttl": 86400  # 24 hours
        },
        "short_term": {
            "duration": "hours",
            "scope": "system_wide",
            "persistence": True,
            "access": "all_agents",
            "description": "Recent system events and states",
            "default_ttl": 86400 * 7  # 7 days
        },
        "long_term": {
            "duration": "indefinite",
            "scope": "system_wide",
            "persistence": True,
            "access": "all_agents",
            "description": "Persistent knowledge and patterns",
            "default_ttl": None  # No expiration
        },
        "collective": {
            "duration": "indefinite",
            "scope": "identity",
            "persistence": True,
            "access": "identity_coordinator",
            "description": "Shared identity and values",
            "default_ttl": None  # No expiration
        }
    },
    
    "memory_types": {
        "contextual": {
            "description": "Current interaction context",
            "structure": "key_value",
            "ttl": 3600
        },
        "factual": {
            "description": "Known facts and information",
            "structure": "graph",
            "ttl": None
        },
        "procedural": {
            "description": "How-to knowledge and procedures",
            "structure": "hierarchical",
            "ttl": None
        },
        "episodic": {
            "description": "Event memories and experiences",
            "structure": "temporal",
            "ttl": 86400 * 30  # 30 days
        },
        "emotional": {
            "description": "Sentiment and emotional context",
            "structure": "vector",
            "ttl": 3600 * 24  # 24 hours
        }
    }
}


class SharedMemoryManager:
    """
    Central manager for all shared memory across agents.

    Manages multiple memory layers with different persistence,
    access patterns, and TTL configurations.
    """

    def __init__(self):
        self.memory_layers = {}
        self.access_log = []
        self.synchronization_queue = asyncio.Queue()
        self.subscribers = {}
        # Task tracking: maps task_id -> set of participating agent_ids
        self.active_tasks: Dict[str, Set[str]] = {}
        # Agent-to-task lookup: maps agent_id -> set of task_ids
        self._agent_tasks: Dict[str, Set[str]] = {}

    def register_task(self, task_id: str, agent_ids: List[str]):
        """Register a task with its participating agents."""
        self.active_tasks[task_id] = set(agent_ids)
        for agent_id in agent_ids:
            if agent_id not in self._agent_tasks:
                self._agent_tasks[agent_id] = set()
            self._agent_tasks[agent_id].add(task_id)

    def unregister_task(self, task_id: str):
        """Remove a completed or cancelled task."""
        agent_ids = self.active_tasks.pop(task_id, set())
        for agent_id in agent_ids:
            if agent_id in self._agent_tasks:
                self._agent_tasks[agent_id].discard(task_id)

    def is_agent_participating(self, agent_id: str) -> bool:
        """Check if an agent is participating in any active task."""
        return bool(self._agent_tasks.get(agent_id))
        
    async def initialize(self):
        """Initialize all memory layers."""
        for layer_name, config in SHARED_MEMORY_ARCHITECTURE["memory_layers"].items():
            self.memory_layers[layer_name] = {
                "data": {},
                "config": config,
                "last_access": {},
                "access_count": {},
                "created_at": datetime.utcnow()
            }
            
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
            
    async def write(self, layer: str, key: str, value: Any, 
                    agent_id: str, metadata: Dict[str, Any] = None,
                    ttl: Optional[int] = None) -> bool:
        """
        Write data to shared memory layer.
        
        Args:
            layer: Memory layer name
            key: Data key
            value: Data value
            agent_id: Writing agent
            metadata: Optional metadata
            ttl: Optional TTL override
            
        Returns:
            True if write successful
        """
        if layer not in self.memory_layers:
            raise ValueError(f"Unknown memory layer: {layer}")
            
        # Check access permissions
        if not self._check_access_permission(layer, agent_id, "write"):
            return False
            
        # Calculate expiration
        config = self.memory_layers[layer]["config"]
        effective_ttl = ttl or config.get("default_ttl")
        expires_at = None
        if effective_ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=effective_ttl)
            
        entry = {
            "value": value,
            "timestamp": datetime.utcnow(),
            "agent_id": agent_id,
            "metadata": metadata or {},
            "version": self._get_next_version(layer, key),
            "expires_at": expires_at
        }
        
        self.memory_layers[layer]["data"][key] = entry
        self.memory_layers[layer]["last_access"][key] = datetime.utcnow()
        self.memory_layers[layer]["access_count"][key] = \
            self.memory_layers[layer]["access_count"].get(key, 0) + 1
            
        # Log access
        self.access_log.append({
            "timestamp": datetime.utcnow(),
            "operation": "write",
            "layer": layer,
            "key": key,
            "agent_id": agent_id
        })
            
        # Notify subscribers
        await self._notify_subscribers(layer, key, entry)
        
        return True
    
    async def read(self, layer: str, key: str, 
                   agent_id: str) -> Optional[Any]:
        """
        Read data from shared memory layer.
        
        Args:
            layer: Memory layer name
            key: Data key
            agent_id: Reading agent
            
        Returns:
            Data value or None
        """
        if layer not in self.memory_layers:
            return None
            
        if not self._check_access_permission(layer, agent_id, "read"):
            return None
            
        entry = self.memory_layers[layer]["data"].get(key)
        
        if entry:
            # Check expiration
            if entry.get("expires_at") and entry["expires_at"] < datetime.utcnow():
                # Expired, remove
                del self.memory_layers[layer]["data"][key]
                return None
                
            self.memory_layers[layer]["last_access"][key] = datetime.utcnow()
            self.memory_layers[layer]["access_count"][key] = \
                self.memory_layers[layer]["access_count"].get(key, 0) + 1
                
            # Log access
            self.access_log.append({
                "timestamp": datetime.utcnow(),
                "operation": "read",
                "layer": layer,
                "key": key,
                "agent_id": agent_id
            })
                
            return entry["value"]
            
        return None
    
    async def query(self, layer: str, query_params: Dict[str, Any], 
                    agent_id: str) -> List[Dict[str, Any]]:
        """
        Query shared memory with filters.
        
        Args:
            layer: Memory layer name
            query_params: Query filters
            agent_id: Querying agent
            
        Returns:
            List of matching entries
        """
        if layer not in self.memory_layers:
            return []
            
        results = []
        data = self.memory_layers[layer]["data"]
        
        for key, entry in list(data.items()):
            # Check expiration
            if entry.get("expires_at") and entry["expires_at"] < datetime.utcnow():
                del data[key]
                continue
                
            if self._matches_query(entry, query_params):
                results.append({
                    "key": key,
                    "value": entry["value"],
                    "metadata": entry["metadata"],
                    "timestamp": entry["timestamp"]
                })
                
        return results
    
    async def delete(self, layer: str, key: str, 
                     agent_id: str) -> bool:
        """
        Delete data from shared memory.
        
        Args:
            layer: Memory layer name
            key: Data key
            agent_id: Deleting agent
            
        Returns:
            True if deleted
        """
        if layer not in self.memory_layers:
            return False
            
        if not self._check_access_permission(layer, agent_id, "write"):
            return False
            
        if key in self.memory_layers[layer]["data"]:
            del self.memory_layers[layer]["data"][key]
            
            # Log access
            self.access_log.append({
                "timestamp": datetime.utcnow(),
                "operation": "delete",
                "layer": layer,
                "key": key,
                "agent_id": agent_id
            })
            
            return True
            
        return False
    
    def _check_access_permission(self, layer: str, agent_id: str, 
                                  operation: str) -> bool:
        """Check if agent has permission for operation on layer."""
        config = self.memory_layers[layer]["config"]
        access = config["access"]
        
        if access == "all_agents":
            return True
        elif access == "identity_coordinator":
            return agent_id == "A001_CORE"
        elif access == "participating_agents":
            # Check if agent is participating in any active task
            return self.is_agent_participating(agent_id)
            
        return False
    
    def _get_next_version(self, layer: str, key: str) -> int:
        """Get next version number for a key."""
        entry = self.memory_layers[layer]["data"].get(key)
        if entry:
            return entry.get("version", 0) + 1
        return 1
    
    def _matches_query(self, entry: Dict[str, Any], 
                       query_params: Dict[str, Any]) -> bool:
        """Check if entry matches query parameters."""
        # Agent filter
        if "agent_id" in query_params:
            if entry.get("agent_id") != query_params["agent_id"]:
                return False
                
        # Time range filter
        if "since" in query_params:
            if entry["timestamp"] < query_params["since"]:
                return False
                
        # Metadata filter
        if "metadata_filter" in query_params:
            for k, v in query_params["metadata_filter"].items():
                if entry.get("metadata", {}).get(k) != v:
                    return False
                    
        return True
    
    async def _notify_subscribers(self, layer: str, key: str, 
                                   entry: Dict[str, Any]):
        """Notify subscribers of memory update."""
        subscription_key = f"{layer}:{key}"
        
        if subscription_key in self.subscribers:
            for callback in self.subscribers[subscription_key]:
                try:
                    await callback(layer, key, entry)
                except (RuntimeError, ValueError, TypeError) as e:
                    print(f"Subscriber notification error: {e}")
                    
    def subscribe(self, layer: str, key: str, callback):
        """
        Subscribe to updates for a specific key.
        
        Args:
            layer: Memory layer
            key: Data key
            callback: Async callback function
        """
        subscription_key = f"{layer}:{key}"
        
        if subscription_key not in self.subscribers:
            self.subscribers[subscription_key] = []
            
        self.subscribers[subscription_key].append(callback)
        
    def unsubscribe(self, layer: str, key: str, callback):
        """Unsubscribe from updates."""
        subscription_key = f"{layer}:{key}"
        
        if subscription_key in self.subscribers:
            if callback in self.subscribers[subscription_key]:
                self.subscribers[subscription_key].remove(callback)
                
    async def _periodic_cleanup(self):
        """Periodically clean up expired entries."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            for layer_name, layer_data in self.memory_layers.items():
                expired_keys = []
                
                for key, entry in layer_data["data"].items():
                    if entry.get("expires_at") and entry["expires_at"] < datetime.utcnow():
                        expired_keys.append(key)
                        
                for key in expired_keys:
                    del layer_data["data"][key]
                    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Memory statistics
        """
        stats = {}
        
        for layer_name, layer_data in self.memory_layers.items():
            stats[layer_name] = {
                "entry_count": len(layer_data["data"]),
                "total_accesses": sum(layer_data["access_count"].values()),
                "created_at": layer_data["created_at"]
            }
            
        return stats


class ContextPropagator:
    """
    Propagates context across agents and memory layers.
    
    Ensures that relevant context is available to all agents
    that need it for their operations.
    """
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.memory = memory_manager
        self.propagation_rules = self._load_propagation_rules()
        
    def _load_propagation_rules(self) -> Dict[str, Any]:
        """Load context propagation rules."""
        return {
            "auto_propagate": True,
            "context_inference": True,
            "priority_boost": {
                "urgent": 2.0,
                "high": 1.5,
                "normal": 1.0,
                "low": 0.5
            }
        }
        
    async def propagate_context(self, source_agent: str, context: Dict[str, Any],
                                 target_agents: List[str] = None) -> Dict[str, Any]:
        """
        Propagate context from one agent to others.
        
        Args:
            source_agent: Source agent ID
            context: Context to propagate
            target_agents: Optional specific targets
            
        Returns:
            Propagation result
        """
        propagation_result = {
            "source": source_agent,
            "targets": [],
            "propagated_keys": [],
            "failed_targets": []
        }
        
        # Store in working memory
        context_key = f"context:{source_agent}:{datetime.utcnow().timestamp()}"
        await self.memory.write(
            "working", 
            context_key, 
            context, 
            source_agent,
            {"propagation_targets": target_agents}
        )
        
        # Determine target agents
        if target_agents is None:
            target_agents = self._infer_target_agents(context)
            
        # Propagate to each target
        for target in target_agents:
            try:
                await self._send_context_to_agent(target, context, context_key)
                propagation_result["targets"].append(target)
                propagation_result["propagated_keys"].append(context_key)
            except (OSError, RuntimeError, ValueError) as e:
                propagation_result["failed_targets"].append({
                    "agent": target,
                    "error": str(e)
                })
                
        return propagation_result
    
    async def _send_context_to_agent(self, agent_id: str,
                                      context: Dict[str, Any],
                                      context_key: str):
        """Send context to a specific agent via file-based message passing."""
        # Store in agent-specific shared memory
        await self.memory.write(
            "ephemeral",
            f"received_context:{agent_id}:{context_key}",
            context,
            agent_id,
            {"source": context_key}
        )

        # File-based message passing for cross-process communication
        message_dir = Path(tempfile.gettempdir()) / "openclaw_messages" / agent_id
        try:
            message_dir.mkdir(parents=True, exist_ok=True)
            message_file = message_dir / f"{context_key.replace(':', '_')}.json"
            message_payload = {
                "context_key": context_key,
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "target_agent": agent_id,
            }
            message_file.write_text(json.dumps(message_payload, default=str))
        except OSError as e:
            raise OSError(
                f"Failed to send file-based message to {agent_id}: {e}"
            ) from e
        
    def _infer_target_agents(self, context: Dict[str, Any]) -> List[str]:
        """
        Infer which agents should receive context based on content.
        
        Args:
            context: Context content
            
        Returns:
            List of target agent IDs
        """
        targets = []
        context_str = json.dumps(context).lower()
        
        # Keyword-based inference
        agent_keywords = {
            "A002_BROWSER": ["web", "browser", "url", "http", "page", "site"],
            "A003_EMAIL": ["email", "gmail", "message", "inbox", "draft"],
            "A004_VOICE": ["voice", "speech", "audio", "speak", "listen"],
            "A005_PHONE": ["phone", "call", "sms", "twilio", "number"],
            "A006_SYSTEM": ["system", "windows", "process", "resource"],
            "A007_FILE": ["file", "directory", "folder", "path", "document"],
            "A008_SCHEDULER": ["schedule", "cron", "job", "timer", "deadline"],
            "A009_MEMORY": ["memory", "remember", "recall", "store", "history"],
            "A010_SECURITY": ["security", "access", "auth", "permission", "threat"],
            "A011_LEARNER": ["learn", "pattern", "adapt", "improve", "study"],
            "A012_CREATIVE": ["create", "write", "design", "generate", "content"],
            "A013_ANALYZER": ["analyze", "data", "report", "insight", "statistic"],
            "A014_HEARTBEAT": ["health", "status", "monitor", "alert", "metric"],
            "A015_USER": ["user", "preference", "profile", "relationship"]
        }
        
        for agent_id, keywords in agent_keywords.items():
            if any(kw in context_str for kw in keywords):
                targets.append(agent_id)
                
        # Always include CORE for coordination
        if "A001_CORE" not in targets:
            targets.append("A001_CORE")
            
        return targets
    
    async def create_shared_context(self, agent_ids: List[str], 
                                    base_context: Dict[str, Any]) -> str:
        """
        Create a shared context for multiple agents.
        
        Args:
            agent_ids: Participating agents
            base_context: Base context data
            
        Returns:
            Context key
        """
        context_key = f"shared:{':'.join(agent_ids)}:{datetime.utcnow().timestamp()}"
        
        shared_context = {
            "participants": agent_ids,
            "created_at": datetime.utcnow().isoformat(),
            "data": base_context
        }
        
        await self.memory.write(
            "working",
            context_key,
            shared_context,
            "A001_CORE",  # Written by CORE
            {"participants": agent_ids}
        )
        
        return context_key
