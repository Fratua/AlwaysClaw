#!/usr/bin/env python3
"""
AI Agent Sandbox Monitoring and Behavioral Analysis
Windows 10 AI Agent Security System

This module provides:
- Behavioral analysis for agent processes
- Real-time monitoring of suspicious activities
- Rate limiting for agent operations
- Filesystem access control
- Network policy enforcement
"""

import os
import sys
import time
import json
import functools
import threading
import statistics
import logging
import hashlib
import re
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from enum import Enum, auto
from pathlib import Path
from threading import Lock, Event

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class NetworkAccessLevel(Enum):
    """Network access levels for components"""
    NONE = "none"
    LOOPBACK = "loopback"
    LIMITED = "limited"
    PROXY = "proxy"
    FULL = "full"


# Blocked file paths (Windows-specific)
BLOCKED_PATHS: Set[str] = {
    r'C:\Windows\System32\config\SAM',
    r'C:\Windows\System32\config\SECURITY',
    r'C:\Windows\System32\config\SYSTEM',
    r'C:\Windows\System32\config\SOFTWARE',
    r'C:\Windows\System32\lsass.exe',
    r'C:\Windows\System32\csrss.exe',
    r'C:\Windows\System32\services.exe',
    r'C:\Windows\System32\winlogon.exe',
    r'C:\ProgramData\Microsoft\Windows Defender',
    r'C:\Windows\Temp',
    r'C:\Windows\Prefetch',
}

# Sensitive path patterns
SENSITIVE_PATTERNS = [
    r'.*\.ssh/.*',
    r'.*\.gnupg/.*',
    r'.*\.aws/.*',
    r'.*\.azure/.*',
    r'.*\.docker/.*',
    r'.*AppData.*Credentials.*',
    r'.*AppData.*Cookies.*',
    r'.*AppData.*Passwords.*',
]

# Blocked file extensions
BLOCKED_EXTENSIONS = {
    '.exe', '.dll', '.bat', '.cmd', '.ps1', '.vbs', 
    '.js', '.wsf', '.scr', '.com'
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BehaviorBaseline:
    """Baseline metrics for normal agent behavior"""
    avg_api_calls_per_minute: float = 50.0
    avg_file_ops_per_minute: float = 20.0
    avg_network_requests_per_minute: float = 10.0
    avg_memory_usage_mb: float = 512.0
    avg_cpu_percent: float = 20.0
    max_child_processes: int = 5
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.txt', '.json', '.xml', '.csv', '.log', '.py', '.js', '.md'
    })
    blocked_file_extensions: Set[str] = field(default_factory=lambda: BLOCKED_EXTENSIONS)


@dataclass
class SuspiciousBehavior:
    """Definition of suspicious behavior pattern"""
    name: str
    description: str
    severity: AlertSeverity
    detection_callback: Callable[[Dict], bool]


@dataclass
class NetworkPolicy:
    """Network access policy for agent component"""
    access_level: NetworkAccessLevel
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    allow_dns: bool = False
    allow_outbound: bool = False
    require_proxy: bool = False
    proxy_host: str = ""
    proxy_port: int = 0
    rate_limit_mbps: float = 10.0


@dataclass
class SecurityAlert:
    """Security alert structure"""
    timestamp: float
    behavior_name: str
    description: str
    severity: AlertSeverity
    event_data: Dict[str, Any]
    context: Dict[str, Any]


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter for agent operations"""
    
    DEFAULT_LIMITS = {
        'llm_request': {'rate': 100, 'burst': 20},
        'tool_execution': {'rate': 50, 'burst': 10},
        'file_read': {'rate': 500, 'burst': 100},
        'file_write': {'rate': 100, 'burst': 20},
        'network_request': {'rate': 30, 'burst': 5},
        'browser_action': {'rate': 60, 'burst': 10},
        'credential_access': {'rate': 10, 'burst': 2},
        'process_spawn': {'rate': 20, 'burst': 5},
        'registry_write': {'rate': 50, 'burst': 10},
    }
    
    def __init__(self, custom_limits: Optional[Dict] = None):
        self.buckets: Dict[str, Dict] = defaultdict(
            lambda: {
                'tokens': 0.0,
                'last_update': time.time(),
                'lock': Lock()
            }
        )
        self.limits = custom_limits or self.DEFAULT_LIMITS
    
    def acquire(self, operation_type: str, tokens: int = 1) -> bool:
        """Attempt to acquire tokens for an operation"""
        if operation_type not in self.limits:
            logger.warning(f"Unknown operation type: {operation_type}")
            return True
        
        limit = self.limits[operation_type]
        bucket = self.buckets[operation_type]
        
        with bucket['lock']:
            now = time.time()
            elapsed = now - bucket['last_update']
            
            # Add tokens based on elapsed time
            bucket['tokens'] = min(
                limit['burst'],
                bucket['tokens'] + elapsed * (limit['rate'] / 60.0)
            )
            bucket['last_update'] = now
            
            # Check if we have enough tokens
            if bucket['tokens'] >= tokens:
                bucket['tokens'] -= tokens
                return True
            else:
                logger.warning(
                    f"Rate limit exceeded for {operation_type}. "
                    f"Available: {bucket['tokens']:.2f}, Required: {tokens}"
                )
                return False
    
    def get_wait_time(self, operation_type: str, tokens: int = 1) -> float:
        """Calculate wait time until tokens are available"""
        if operation_type not in self.limits:
            return 0.0
        
        limit = self.limits[operation_type]
        bucket = self.buckets[operation_type]
        
        with bucket['lock']:
            deficit = tokens - bucket['tokens']
            if deficit <= 0:
                return 0.0
            return deficit / (limit['rate'] / 60.0)
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get current rate limiter statistics"""
        stats = {}
        for op_type, bucket in self.buckets.items():
            with bucket['lock']:
                stats[op_type] = {
                    'tokens': bucket['tokens'],
                    'limit': self.limits.get(op_type, {})
                }
        return stats


def rate_limited(operation_type: str, tokens: int = 1):
    """Decorator for rate-limiting agent operations"""
    limiter = RateLimiter()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire(operation_type, tokens):
                wait_time = limiter.get_wait_time(operation_type, tokens)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {operation_type}. "
                    f"Retry after {wait_time:.2f} seconds"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass


# ============================================================================
# BEHAVIORAL ANALYZER
# ============================================================================

class BehavioralAnalyzer:
    """Real-time behavioral analysis for agent processes"""
    
    def __init__(self, baseline: Optional[BehaviorBaseline] = None):
        self.baseline = baseline or BehaviorBaseline()
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.suspicious_behaviors: List[SuspiciousBehavior] = []
        self.alerts: List[SecurityAlert] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_handlers: List[Callable[[SecurityAlert], None]] = []
        
        self._setup_behaviors()
    
    def _setup_behaviors(self):
        """Define suspicious behavior patterns"""
        self.suspicious_behaviors = [
            SuspiciousBehavior(
                name="api_call_spike",
                description="Unusual spike in API calls",
                severity=AlertSeverity.MEDIUM,
                detection_callback=self._detect_api_spike
            ),
            SuspiciousBehavior(
                name="mass_file_deletion",
                description="Mass file deletion detected",
                severity=AlertSeverity.HIGH,
                detection_callback=self._detect_mass_deletion
            ),
            SuspiciousBehavior(
                name="executable_creation",
                description="Executable file created in temp directory",
                severity=AlertSeverity.HIGH,
                detection_callback=self._detect_executable_creation
            ),
            SuspiciousBehavior(
                name="network_anomaly",
                description="Unusual network connection pattern",
                severity=AlertSeverity.MEDIUM,
                detection_callback=self._detect_network_anomaly
            ),
            SuspiciousBehavior(
                name="privilege_escalation_attempt",
                description="Attempt to escalate privileges",
                severity=AlertSeverity.CRITICAL,
                detection_callback=self._detect_privilege_escalation
            ),
            SuspiciousBehavior(
                name="process_injection",
                description="Potential process injection detected",
                severity=AlertSeverity.CRITICAL,
                detection_callback=self._detect_process_injection
            ),
            SuspiciousBehavior(
                name="credential_access",
                description="Attempt to access credentials",
                severity=AlertSeverity.CRITICAL,
                detection_callback=self._detect_credential_access
            ),
            SuspiciousBehavior(
                name="registry_persistence",
                description="Registry modification for persistence",
                severity=AlertSeverity.HIGH,
                detection_callback=self._detect_persistence
            ),
            SuspiciousBehavior(
                name="data_exfiltration",
                description="Potential data exfiltration",
                severity=AlertSeverity.CRITICAL,
                detection_callback=self._detect_exfiltration
            ),
            SuspiciousBehavior(
                name="memory_anomaly",
                description="Unusual memory allocation pattern",
                severity=AlertSeverity.MEDIUM,
                detection_callback=self._detect_memory_anomaly
            ),
            SuspiciousBehavior(
                name="suspicious_child_process",
                description="Suspicious child process spawned",
                severity=AlertSeverity.HIGH,
                detection_callback=self._detect_suspicious_child
            ),
        ]
    
    def start_monitoring(self):
        """Start behavioral monitoring"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Behavioral monitoring started")
    
    def stop_monitoring(self):
        """Stop behavioral monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Behavioral monitoring stopped")
    
    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Add a handler for security alerts"""
        self._alert_handlers.append(handler)
    
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record an event for analysis"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        self.metrics_history[event_type].append(event)
        self._analyze_event(event)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            self._periodic_analysis()
            time.sleep(5)
    
    def _analyze_event(self, event: Dict):
        """Analyze a single event for suspicious behavior"""
        for behavior in self.suspicious_behaviors:
            try:
                if behavior.detection_callback(event):
                    self._trigger_alert(behavior, event)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Error in behavior detection {behavior.name}: {e}")
    
    def _periodic_analysis(self):
        """Perform periodic analysis of collected metrics"""
        for event_type, history in self.metrics_history.items():
            if len(history) < 10:
                continue
            
            cutoff = time.time() - 60
            recent_events = [e for e in history if e['timestamp'] > cutoff]
            
            if event_type == 'api_call':
                if len(recent_events) > self.baseline.avg_api_calls_per_minute * 3:
                    self._trigger_alert(
                        SuspiciousBehavior(
                            name="sustained_api_spike",
                            description=f"Sustained API call spike: {len(recent_events)}/min",
                            severity=AlertSeverity.MEDIUM,
                            detection_callback=lambda x: False
                        ),
                        {'count': len(recent_events)}
                    )
    
    # Detection callbacks
    def _detect_api_spike(self, event: Dict) -> bool:
        if event['type'] != 'api_call':
            return False
        cutoff = time.time() - 10
        recent = [e for e in self.metrics_history['api_call'] if e['timestamp'] > cutoff]
        return len(recent) > 20
    
    def _detect_mass_deletion(self, event: Dict) -> bool:
        if event['type'] != 'file_delete':
            return False
        cutoff = time.time() - 60
        recent_deletions = [e for e in self.metrics_history['file_delete'] if e['timestamp'] > cutoff]
        return len(recent_deletions) > 50
    
    def _detect_executable_creation(self, event: Dict) -> bool:
        if event['type'] != 'file_create':
            return False
        filepath = event['data'].get('path', '')
        ext = Path(filepath).suffix.lower()
        if ext in self.baseline.blocked_file_extensions:
            return 'temp' in filepath.lower() or 'tmp' in filepath.lower()
        return False
    
    def _detect_network_anomaly(self, event: Dict) -> bool:
        if event['type'] != 'network_connect':
            return False
        remote_ip = event['data'].get('remote_ip', '')
        suspicious_ips = {'0.0.0.0', '255.255.255.255'}
        if remote_ip in suspicious_ips:
            return True
        bytes_sent = event['data'].get('bytes_sent', 0)
        return bytes_sent > 10 * 1024 * 1024
    
    def _detect_privilege_escalation(self, event: Dict) -> bool:
        if event['type'] != 'token_adjust':
            return False
        privileges_added = event['data'].get('privileges_added', [])
        dangerous_privs = {'SeDebugPrivilege', 'SeTcbPrivilege', 'SeCreateTokenPrivilege'}
        return any(p in dangerous_privs for p in privileges_added)
    
    def _detect_process_injection(self, event: Dict) -> bool:
        if event['type'] != 'memory_write':
            return False
        target_pid = event['data'].get('target_pid', 0)
        source_pid = event['data'].get('source_pid', 0)
        return target_pid != 0 and target_pid != source_pid
    
    def _detect_credential_access(self, event: Dict) -> bool:
        if event['type'] != 'registry_read':
            return False
        path = event['data'].get('path', '').lower()
        credential_paths = ['sam', 'security', 'lsa', 'credman', 'password', 'credential']
        return any(p in path for p in credential_paths)
    
    def _detect_persistence(self, event: Dict) -> bool:
        if event['type'] != 'registry_write':
            return False
        path = event['data'].get('path', '').lower()
        persistence_keys = [
            r'\\currentversion\\run',
            r'\\currentversion\\runonce',
            r'\\currentversion\\winlogon',
        ]
        return any(p in path for p in persistence_keys)
    
    def _detect_exfiltration(self, event: Dict) -> bool:
        if event['type'] != 'network_send':
            return False
        cutoff = time.time() - 60
        total_sent = sum(
            e['data'].get('bytes_sent', 0)
            for e in self.metrics_history['network_send']
            if e['timestamp'] > cutoff
        )
        return total_sent > 100 * 1024 * 1024
    
    def _detect_memory_anomaly(self, event: Dict) -> bool:
        if event['type'] != 'memory_alloc':
            return False
        size = event['data'].get('size', 0)
        protection = event['data'].get('protection', 0)
        if size > 100 * 1024 * 1024:
            if protection in {0x10, 0x40}:  # PAGE_EXECUTE or PAGE_EXECUTE_READWRITE
                return True
        return False
    
    def _detect_suspicious_child(self, event: Dict) -> bool:
        if event['type'] != 'process_create':
            return False
        cmdline = event['data'].get('command_line', '').lower()
        suspicious_patterns = ['powershell', 'cmd.exe', 'wscript', 'cscript', 'mshta']
        return any(p in cmdline for p in suspicious_patterns)
    
    def _trigger_alert(self, behavior: SuspiciousBehavior, event: Dict):
        """Trigger an alert for suspicious behavior"""
        alert = SecurityAlert(
            timestamp=time.time(),
            behavior_name=behavior.name,
            description=behavior.description,
            severity=behavior.severity,
            event_data=event,
            context=self._get_alert_context()
        )
        
        self.alerts.append(alert)
        
        logger.warning(
            f"[ALERT] {behavior.severity.name}: {behavior.description}"
        )
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Alert handler error: {e}")
        
        # Take action based on severity
        if behavior.severity in {AlertSeverity.HIGH, AlertSeverity.CRITICAL}:
            self._take_defensive_action(alert)
    
    def _get_alert_context(self) -> Dict[str, Any]:
        """Get context for alert"""
        return {
            'recent_api_calls': len(self.metrics_history['api_call']),
            'recent_file_ops': len(self.metrics_history['file_write']),
            'recent_network': len(self.metrics_history['network_connect']),
            'active_processes': len(set(
                e['data'].get('pid') 
                for e in self.metrics_history['process_create']
            ))
        }
    
    def _take_defensive_action(self, alert: SecurityAlert):
        """Take defensive action for high-severity alerts"""
        severity = alert.severity
        
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"Taking immediate containment action for: {alert.behavior_name}")
            # Signal to terminate agent process
            # Capture memory dump
            # Isolate network
        elif severity == AlertSeverity.HIGH:
            logger.error(f"Escalated monitoring for: {alert.behavior_name}")
            # Increase monitoring frequency
            # Notify security team
    
    def get_alerts(self, min_severity: Optional[AlertSeverity] = None) -> List[SecurityAlert]:
        """Get alerts, optionally filtered by severity"""
        if min_severity is None:
            return self.alerts.copy()
        return [a for a in self.alerts if a.severity.value >= min_severity.value]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'total_events': sum(len(h) for h in self.metrics_history.values()),
            'total_alerts': len(self.alerts),
            'alerts_by_severity': {
                s.name: len([a for a in self.alerts if a.severity == s])
                for s in AlertSeverity
            },
            'event_types': list(self.metrics_history.keys())
        }


# ============================================================================
# FILESYSTEM SANDBOX
# ============================================================================

class FilesystemSandbox:
    """User-mode filesystem sandbox"""
    
    def __init__(self, sandbox_id: str, root_path: str):
        self.sandbox_id = sandbox_id
        self.root_path = Path(root_path).resolve()
        self.allowed_paths: Set[Path] = set()
        self.writable_paths: Set[Path] = set()
        self.access_log: List[Dict] = []
    
    def allow_path(self, path: str, writable: bool = False):
        """Allow access to a specific path"""
        resolved_path = Path(path).resolve()
        
        if self._is_blocked_path(resolved_path):
            raise PermissionError(f"Access to {path} is blocked by security policy")
        
        self.allowed_paths.add(resolved_path)
        if writable:
            self.writable_paths.add(resolved_path)
        
        logger.info(f"Allowed {'writable' if writable else 'read-only'} access to {path}")
    
    def _is_blocked_path(self, path: Path) -> bool:
        """Check if path is in blocked list"""
        path_str = str(path).lower()
        
        for blocked in BLOCKED_PATHS:
            if path_str == blocked.lower():
                return True
        
        for pattern in SENSITIVE_PATTERNS:
            if re.match(pattern, path_str, re.IGNORECASE):
                return True
        
        return False
    
    def validate_access(self, path: str, access_type: str = 'read') -> bool:
        """Validate if access to path is allowed"""
        resolved_path = Path(path).resolve()
        
        self.access_log.append({
            'timestamp': time.time(),
            'path': str(resolved_path),
            'access_type': access_type,
            'allowed': False
        })
        
        if self._is_blocked_path(resolved_path):
            logger.warning(f"Blocked access attempt to {path}")
            return False
        
        for allowed in self.allowed_paths:
            try:
                resolved_path.relative_to(allowed)
                
                if access_type == 'write':
                    for writable in self.writable_paths:
                        try:
                            resolved_path.relative_to(writable)
                            self.access_log[-1]['allowed'] = True
                            return True
                        except ValueError:
                            continue
                    logger.warning(f"Write access denied to {path}")
                    return False
                
                self.access_log[-1]['allowed'] = True
                return True
            except ValueError:
                continue
        
        logger.warning(f"Access denied to {path} - not in allowed paths")
        return False
    
    def create_jail(self):
        """Create chroot-like jail environment"""
        directories = {
            'bin': False,
            'lib': False,
            'skills': False,
            'config': False,
            'data': True,
            'logs': True,
            'temp': True,
            'cache': True,
        }
        
        for dir_name, writable in directories.items():
            dir_path = self.root_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.allow_path(str(dir_path), writable)
    
    def get_access_log(self) -> List[Dict]:
        """Get filesystem access log"""
        return self.access_log.copy()


# ============================================================================
# NETWORK POLICY MANAGER
# ============================================================================

class NetworkPolicyManager:
    """Manage network policies for agent components"""
    
    # Predefined policies for common components
    POLICIES = {
        'agent_core': NetworkPolicy(
            access_level=NetworkAccessLevel.PROXY,
            allowed_hosts=[
                'api.openai.com',
                'api.anthropic.com',
                'localhost',
                '127.0.0.1'
            ],
            allowed_ports=[443, 80],
            allow_dns=True,
            require_proxy=True,
            proxy_host='localhost',
            proxy_port=3128,
            rate_limit_mbps=10.0
        ),
        'browser_controller': NetworkPolicy(
            access_level=NetworkAccessLevel.PROXY,
            allowed_hosts=['*'],
            allowed_ports=[80, 443],
            allow_dns=True,
            require_proxy=True,
            proxy_host='localhost',
            proxy_port=3128,
            blocked_hosts=[
                '127.0.0.1',
                'localhost',
                '192.168.*',
                '10.*',
                '172.16.*'
            ],
            rate_limit_mbps=20.0
        ),
        'skill_executor': NetworkPolicy(
            access_level=NetworkAccessLevel.NONE,
            allowed_hosts=[],
            allowed_ports=[],
            allow_dns=False,
            allow_outbound=False,
            rate_limit_mbps=0.0
        ),
        'gmail_client': NetworkPolicy(
            access_level=NetworkAccessLevel.LIMITED,
            allowed_hosts=[
                'gmail.googleapis.com',
                'oauth2.googleapis.com',
                'accounts.google.com'
            ],
            allowed_ports=[443],
            allow_dns=True,
            rate_limit_mbps=5.0
        ),
        'twilio_client': NetworkPolicy(
            access_level=NetworkAccessLevel.LIMITED,
            allowed_hosts=[
                'api.twilio.com',
                'studio.twilio.com'
            ],
            allowed_ports=[443],
            allow_dns=True,
            rate_limit_mbps=5.0
        ),
        'tts_stt': NetworkPolicy(
            access_level=NetworkAccessLevel.LIMITED,
            allowed_hosts=[
                'eastus.tts.speech.microsoft.com',
                'eastus.stt.speech.microsoft.com'
            ],
            allowed_ports=[443],
            allow_dns=True,
            rate_limit_mbps=10.0
        ),
    }
    
    def __init__(self):
        self.active_policies: Dict[str, NetworkPolicy] = {}
        self._firewall_rules: List[str] = []
    
    def apply_policy(self, component: str, process_id: int = None) -> bool:
        """Apply network policy to a component"""
        if component not in self.POLICIES:
            logger.error(f"Unknown component: {component}")
            return False
        
        policy = self.POLICIES[component]
        self.active_policies[component] = policy
        
        logger.info(f"Applied network policy for {component}")
        return True
    
    def remove_policy(self, component: str) -> bool:
        """Remove network policy for a component"""
        if component not in self.active_policies:
            return True
        
        del self.active_policies[component]
        logger.info(f"Removed network policy for {component}")
        return True
    
    def is_host_allowed(self, component: str, host: str) -> bool:
        """Check if a host is allowed for a component"""
        if component not in self.active_policies:
            return False
        
        policy = self.active_policies[component]
        
        if policy.access_level == NetworkAccessLevel.NONE:
            return False
        
        if policy.access_level == NetworkAccessLevel.LOOPBACK:
            return host in ['localhost', '127.0.0.1', '::1']
        
        if host in policy.blocked_hosts:
            return False
        
        if '*' in policy.allowed_hosts:
            return True
        
        return host in policy.allowed_hosts
    
    def is_port_allowed(self, component: str, port: int) -> bool:
        """Check if a port is allowed for a component"""
        if component not in self.active_policies:
            return False
        
        policy = self.active_policies[component]
        return port in policy.allowed_ports


# ============================================================================
# AGENT SECURITY MONITOR
# ============================================================================

class AgentSecurityMonitor:
    """Main security monitor for AI agent system"""
    
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.rate_limiter = RateLimiter()
        self.filesystem_sandboxes: Dict[str, FilesystemSandbox] = {}
        self.network_manager = NetworkPolicyManager()
        self._running = False
    
    def start(self):
        """Start security monitoring"""
        self._running = True
        self.behavioral_analyzer.start_monitoring()
        logger.info("Agent security monitoring started")
    
    def stop(self):
        """Stop security monitoring"""
        self._running = False
        self.behavioral_analyzer.stop_monitoring()
        logger.info("Agent security monitoring stopped")
    
    def create_filesystem_sandbox(self, sandbox_id: str, root_path: str) -> FilesystemSandbox:
        """Create a new filesystem sandbox"""
        sandbox = FilesystemSandbox(sandbox_id, root_path)
        self.filesystem_sandboxes[sandbox_id] = sandbox
        return sandbox
    
    def on_api_call(self, api_name: str, params: Dict):
        """Hook for API calls"""
        if not self.rate_limiter.acquire('llm_request'):
            raise RateLimitExceeded("API call rate limit exceeded")
        
        self.behavioral_analyzer.record_event('api_call', {
            'api': api_name,
            'params': params,
            'pid': os.getpid()
        })
    
    def on_file_operation(self, operation: str, path: str, **kwargs):
        """Hook for file operations"""
        event_type = f'file_{operation}'
        if not self.rate_limiter.acquire(event_type):
            raise RateLimitExceeded(f"File {operation} rate limit exceeded")
        
        self.behavioral_analyzer.record_event(event_type, {
            'path': path,
            'pid': os.getpid(),
            **kwargs
        })
    
    def on_network_operation(self, operation: str, **kwargs):
        """Hook for network operations"""
        event_type = f'network_{operation}'
        if not self.rate_limiter.acquire(event_type):
            raise RateLimitExceeded(f"Network {operation} rate limit exceeded")
        
        self.behavioral_analyzer.record_event(event_type, {
            'pid': os.getpid(),
            **kwargs
        })
    
    def on_process_operation(self, operation: str, **kwargs):
        """Hook for process operations"""
        event_type = f'process_{operation}'
        if not self.rate_limiter.acquire(event_type):
            raise RateLimitExceeded(f"Process {operation} rate limit exceeded")
        
        self.behavioral_analyzer.record_event(event_type, {
            'pid': os.getpid(),
            **kwargs
        })
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        return {
            'monitoring_active': self._running,
            'behavioral_stats': self.behavioral_analyzer.get_stats(),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'filesystem_sandboxes': len(self.filesystem_sandboxes),
            'active_network_policies': len(self.network_manager.active_policies),
            'alerts': [
                {
                    'timestamp': a.timestamp,
                    'behavior': a.behavior_name,
                    'description': a.description,
                    'severity': a.severity.name
                }
                for a in self.behavioral_analyzer.get_alerts()
            ]
        }


# ============================================================================
# MAIN - EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the security monitoring system"""
    print("AI Agent Security Monitor - Example Usage")
    print("=" * 50)
    
    # Create security monitor
    monitor = AgentSecurityMonitor()
    
    # Add alert handler
    def on_alert(alert: SecurityAlert):
        print(f"[ALERT] {alert.severity.name}: {alert.description}")
    
    monitor.behavioral_analyzer.add_alert_handler(on_alert)
    
    # Start monitoring
    monitor.start()
    
    # Create filesystem sandbox
    sandbox = monitor.create_filesystem_sandbox("agent_1", "C:\\AgentSandboxes\\agent_1")
    sandbox.create_jail()
    
    # Apply network policy
    monitor.network_manager.apply_policy("agent_core")
    
    # Simulate some operations
    try:
        monitor.on_api_call("openai_chat", {"model": "gpt-4"})
        monitor.on_file_operation("read", "C:\\AgentSandboxes\\agent_1\\config\\settings.json")
        monitor.on_network_operation("connect", {"host": "api.openai.com", "port": 443})
    except RateLimitExceeded as e:
        print(f"Rate limit: {e}")
    
    # Generate report
    time.sleep(1)
    report = monitor.get_security_report()
    print("\nSecurity Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Stop monitoring
    monitor.stop()
    
    print("\nSecurity monitoring complete")


if __name__ == "__main__":
    main()
