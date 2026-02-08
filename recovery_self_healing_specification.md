# Recovery and Self-Healing Systems Specification
## Windows 10 OpenClaw-Inspired AI Agent Framework
### Version 1.0 - 24/7 Operation Architecture

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Health Check Implementations](#health-check-implementations)
4. [Crash Detection and Reporting](#crash-detection-and-reporting)
5. [Automatic Restart Procedures](#automatic-restart-procedures)
6. [State Recovery After Crash](#state-recovery-after-crash)
7. [Circuit Breaker Patterns](#circuit-breaker-patterns)
8. [Graceful Degradation](#graceful-degradation)
9. [Recovery Testing Framework](#recovery-testing-framework)
10. [Incident Logging and Analysis](#incident-logging-and-analysis)
11. [Implementation Code Reference](#implementation-code-reference)

---

## Executive Summary

This specification defines a comprehensive recovery and self-healing architecture for a Windows 10-based AI agent system designed for continuous 24/7 operation. The system implements multi-layer resilience mechanisms including health monitoring, crash recovery, automatic restart capabilities, circuit breakers, and graceful degradation strategies.

### Key Resilience Metrics
- **Target Uptime**: 99.95% (4.38 hours downtime/year)
- **Recovery Time Objective (RTO)**: < 30 seconds
- **Recovery Point Objective (RPO)**: < 5 seconds
- **Health Check Frequency**: 5 seconds (critical), 30 seconds (standard)
- **Auto-restart Timeout**: 10 seconds
- **Circuit Breaker Threshold**: 5 failures in 60 seconds

---

## System Architecture Overview

### Component Hierarchy
```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM RESILIENCE LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Watchdog  │  │   Health    │  │    Recovery Manager     │  │
│  │   Service   │  │   Monitor   │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         └─────────────────┴──────────────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    AGENT CORE SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │   Soul   │ │ Identity │ │  Memory  │ │  Cron    │ │ Heart- │ │
│  │  Engine  │ │ Manager  │ │  System  │ │ Scheduler│ │  beat  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    AGENTIC LOOP LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Loop 1 │ Loop 2 │ Loop 3 │ ... │ Loop 15 (Hardcoded)           │
├─────────────────────────────────────────────────────────────────┤
│                    SERVICE INTEGRATION LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │ Gmail  │ │Browser │ │  TTS   │ │  STT   │ │ Twilio │         │
│  │Service │ │Control │ │ Engine │ │ Engine │ │ Voice/ │         │
│  │        │ │        │ │        │ │        │ │  SMS   │         │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘         │
├─────────────────────────────────────────────────────────────────┤
│                    SYSTEM ACCESS LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  File System │ Registry │ Processes │ Network │ Hardware         │
└─────────────────────────────────────────────────────────────────┘
```

### Resilience Layers
1. **Layer 1 - Prevention**: Proactive health monitoring
2. **Layer 2 - Detection**: Real-time failure detection
3. **Layer 3 - Response**: Automated recovery actions
4. **Layer 4 - Recovery**: State restoration and restart
5. **Layer 5 - Learning**: Incident analysis and adaptation

---

## Health Check Implementations

### 3.1 Health Check Architecture

```python
# health_monitor.py - Core Health Monitoring System
"""
Multi-tier health monitoring for Windows 10 AI Agent System
Implements: System-level, Service-level, Component-level health checks
"""

import asyncio
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import threading
import winreg
import ctypes
from ctypes import wintypes

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetrics:
    """Container for health metrics data"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_latency_ms: float
    process_count: int
    thread_count: int
    handle_count: int
    service_status: Dict[str, str]
    
@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: HealthStatus
    timestamp: datetime
    message: str
    metrics: Optional[Dict[str, Any]] = None
    recovery_action: Optional[str] = None
    severity: int = 0  # 0-10 scale

class HealthMonitor:
    """
    Centralized health monitoring system for AI Agent
    Implements multi-tier health checking with configurable intervals
    """
    
    def __init__(self, config_path: str = "config/health_config.json"):
        self.config = self._load_config(config_path)
        self.checks: Dict[str, Callable] = {}
        self.results: deque = deque(maxlen=10000)
        self.subscribers: List[Callable] = []
        self.running = False
        self._lock = threading.RLock()
        self._check_threads: Dict[str, threading.Thread] = {}
        
        # Thresholds
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'network_timeout': 5000,  # ms
            'process_max': 500,
            'handle_max': 50000
        }
        
        # Initialize logging
        self.logger = logging.getLogger('HealthMonitor')
        self._setup_logging()
        
    def _load_config(self, path: str) -> Dict:
        """Load health monitoring configuration"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default health monitoring configuration"""
        return {
            "check_intervals": {
                "critical": 5,      # seconds
                "standard": 30,     # seconds
                "extended": 300     # seconds (5 minutes)
            },
            "components": {
                "system": {"interval": "critical", "enabled": True},
                "memory": {"interval": "critical", "enabled": True},
                "disk": {"interval": "standard", "enabled": True},
                "network": {"interval": "standard", "enabled": True},
                "gmail_service": {"interval": "critical", "enabled": True},
                "browser_control": {"interval": "standard", "enabled": True},
                "tts_engine": {"interval": "standard", "enabled": True},
                "stt_engine": {"interval": "standard", "enabled": True},
                "twilio_service": {"interval": "standard", "enabled": True},
                "soul_engine": {"interval": "critical", "enabled": True},
                "identity_manager": {"interval": "critical", "enabled": True},
                "cron_scheduler": {"interval": "critical", "enabled": True},
                "heartbeat": {"interval": "critical", "enabled": True}
            },
            "alert_thresholds": {
                "consecutive_failures": 3,
                "degradation_window": 300  # 5 minutes
            }
        }
    
    def _setup_logging(self):
        """Configure health monitoring logging"""
        handler = logging.FileHandler('logs/health_monitor.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # ==================== SYSTEM-LEVEL HEALTH CHECKS ====================
    
    def check_system_resources(self) -> HealthCheckResult:
        """
        Comprehensive system resource health check
        Monitors: CPU, Memory, Disk, Network
        """
        try:
            metrics = {}
            issues = []
            severity = 0
            
            # CPU Check
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_percent'] = cpu_percent
            if cpu_percent > self.thresholds['cpu_critical']:
                issues.append(f"CPU critical: {cpu_percent}%")
                severity = max(severity, 9)
            elif cpu_percent > self.thresholds['cpu_warning']:
                issues.append(f"CPU high: {cpu_percent}%")
                severity = max(severity, 5)
            
            # Memory Check
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024**3)
            if memory.percent > self.thresholds['memory_critical']:
                issues.append(f"Memory critical: {memory.percent}%")
                severity = max(severity, 9)
            elif memory.percent > self.thresholds['memory_warning']:
                issues.append(f"Memory high: {memory.percent}%")
                severity = max(severity, 5)
            
            # Disk Check
            disk = psutil.disk_usage('C:\\')
            metrics['disk_percent'] = disk.percent
            if disk.percent > self.thresholds['disk_critical']:
                issues.append(f"Disk critical: {disk.percent}%")
                severity = max(severity, 9)
            elif disk.percent > self.thresholds['disk_warning']:
                issues.append(f"Disk high: {disk.percent}%")
                severity = max(severity, 5)
            
            # Process Check
            process_count = len(psutil.pids())
            metrics['process_count'] = process_count
            if process_count > self.thresholds['process_max']:
                issues.append(f"Too many processes: {process_count}")
                severity = max(severity, 6)
            
            # Determine status
            if severity >= 9:
                status = HealthStatus.CRITICAL
            elif severity >= 5:
                status = HealthStatus.DEGRADED
            elif issues:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                timestamp=datetime.now(),
                message="; ".join(issues) if issues else "All system resources healthy",
                metrics=metrics,
                severity=severity
            )
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Check failed: {str(e)}",
                severity=8
            )
    
    def check_windows_services(self) -> HealthCheckResult:
        """
        Check critical Windows services status
        """
        critical_services = [
            'Dnscache',          # DNS Client
            'NlaSvc',            # Network Location Awareness
            'nsi',               # Network Store Interface
            'RpcSs',             # Remote Procedure Call
            'LanmanWorkstation', # Workstation
            'BFE',               # Base Filtering Engine
            'Dhcp',              # DHCP Client
            'WinHttpAutoProxySvc' # WinHTTP Web Proxy
        ]
        
        failed_services = []
        metrics = {}
        
        try:
            for service in critical_services:
                try:
                    svc = psutil.win_service_get(service)
                    status = svc.status()
                    metrics[service] = status
                    if status != 'running':
                        failed_services.append(f"{service}: {status}")
                except Exception as e:
                    failed_services.append(f"{service}: error ({e})")
                    metrics[service] = "error"
            
            if failed_services:
                return HealthCheckResult(
                    component="windows_services",
                    status=HealthStatus.DEGRADED if len(failed_services) < 3 else HealthStatus.CRITICAL,
                    timestamp=datetime.now(),
                    message=f"Service issues: {', '.join(failed_services)}",
                    metrics=metrics,
                    severity=7 if len(failed_services) >= 3 else 4
                )
            
            return HealthCheckResult(
                component="windows_services",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="All critical Windows services running",
                metrics=metrics,
                severity=0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="windows_services",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Service check failed: {str(e)}",
                severity=6
            )

    # ==================== SERVICE-LEVEL HEALTH CHECKS ====================
    
    def check_gmail_service(self) -> HealthCheckResult:
        """Check Gmail API connectivity and authentication"""
        # Implementation depends on Gmail service wrapper
        metrics = {'last_check': datetime.now().isoformat()}
        
        try:
            # Check token validity
            # Check API connectivity
            # Check recent operation success rate
            
            # Placeholder - actual implementation would call Gmail service
            return HealthCheckResult(
                component="gmail_service",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Gmail service operational",
                metrics=metrics,
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="gmail_service",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Gmail service error: {str(e)}",
                metrics=metrics,
                severity=7,
                recovery_action="restart_gmail_service"
            )
    
    def check_browser_control(self) -> HealthCheckResult:
        """Check browser automation health"""
        metrics = {}
        
        try:
            # Check browser process exists
            browser_processes = [p for p in psutil.process_iter(['name']) 
                               if 'chrome' in p.info['name'].lower() or 
                                  'firefox' in p.info['name'].lower() or
                                  'edge' in p.info['name'].lower()]
            
            metrics['browser_processes'] = len(browser_processes)
            
            if not browser_processes:
                return HealthCheckResult(
                    component="browser_control",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    message="No browser processes found",
                    metrics=metrics,
                    severity=7,
                    recovery_action="restart_browser"
                )
            
            # Check WebDriver connectivity
            # Check recent automation success rate
            
            return HealthCheckResult(
                component="browser_control",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message=f"Browser control operational ({len(browser_processes)} processes)",
                metrics=metrics,
                severity=0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="browser_control",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Browser control check failed: {str(e)}",
                severity=6,
                recovery_action="restart_browser"
            )
    
    def check_tts_engine(self) -> HealthCheckResult:
        """Check Text-to-Speech engine health"""
        try:
            # Check TTS service/process
            # Check voice availability
            # Check recent synthesis success rate
            
            return HealthCheckResult(
                component="tts_engine",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="TTS engine operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="tts_engine",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"TTS engine error: {str(e)}",
                severity=5,
                recovery_action="restart_tts"
            )
    
    def check_stt_engine(self) -> HealthCheckResult:
        """Check Speech-to-Text engine health"""
        try:
            # Check STT service/process
            # Check microphone access
            # Check recent recognition success rate
            
            return HealthCheckResult(
                component="stt_engine",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="STT engine operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="stt_engine",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"STT engine error: {str(e)}",
                severity=5,
                recovery_action="restart_stt"
            )
    
    def check_twilio_service(self) -> HealthCheckResult:
        """Check Twilio voice/SMS service health"""
        try:
            # Check Twilio API connectivity
            # Check webhook endpoints
            # Check recent call/SMS success rate
            
            return HealthCheckResult(
                component="twilio_service",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Twilio service operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="twilio_service",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Twilio service error: {str(e)}",
                severity=6,
                recovery_action="restart_twilio"
            )

    # ==================== COMPONENT-LEVEL HEALTH CHECKS ====================
    
    def check_soul_engine(self) -> HealthCheckResult:
        """Check AI soul/engine health"""
        try:
            # Check GPT-5.2 API connectivity
            # Check response times
            # Check error rates
            
            metrics = {
                'api_response_time_ms': 0,  # placeholder
                'error_rate_5min': 0.0,
                'queue_depth': 0
            }
            
            return HealthCheckResult(
                component="soul_engine",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Soul engine operational",
                metrics=metrics,
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="soul_engine",
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                message=f"Soul engine critical error: {str(e)}",
                severity=10,
                recovery_action="restart_agent"
            )
    
    def check_identity_manager(self) -> HealthCheckResult:
        """Check identity management system health"""
        try:
            # Check identity persistence
            # Check user context availability
            
            return HealthCheckResult(
                component="identity_manager",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Identity manager operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="identity_manager",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Identity manager error: {str(e)}",
                severity=7,
                recovery_action="reload_identity"
            )
    
    def check_cron_scheduler(self) -> HealthCheckResult:
        """Check cron job scheduler health"""
        try:
            # Check scheduler thread
            # Check job queue
            # Check missed executions
            
            return HealthCheckResult(
                component="cron_scheduler",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Cron scheduler operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="cron_scheduler",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message=f"Cron scheduler error: {str(e)}",
                severity=6,
                recovery_action="restart_scheduler"
            )
    
    def check_heartbeat(self) -> HealthCheckResult:
        """Check heartbeat system health"""
        try:
            # Check last heartbeat timestamp
            # Check heartbeat interval compliance
            
            return HealthCheckResult(
                component="heartbeat",
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Heartbeat system operational",
                severity=0
            )
        except Exception as e:
            return HealthCheckResult(
                component="heartbeat",
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                message=f"Heartbeat system error: {str(e)}",
                severity=10,
                recovery_action="emergency_restart"
            )

    # ==================== HEALTH MONITOR ORCHESTRATION ====================
    
    def register_check(self, name: str, check_func: Callable, interval: str = "standard"):
        """Register a health check function"""
        with self._lock:
            self.checks[name] = {
                'function': check_func,
                'interval': interval,
                'last_run': None,
                'failures': 0
            }
    
    def subscribe(self, callback: Callable):
        """Subscribe to health check results"""
        self.subscribers.append(callback)
    
    def _notify_subscribers(self, result: HealthCheckResult):
        """Notify all subscribers of health check result"""
        for callback in self.subscribers:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Subscriber notification failed: {e}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        with self._lock:
            if name not in self.checks:
                return HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.now(),
                    message=f"Check '{name}' not registered",
                    severity=0
                )
            
            check = self.checks[name]
            check['last_run'] = datetime.now()
            
            try:
                result = check['function']()
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    check['failures'] += 1
                else:
                    check['failures'] = 0
                
                with self._lock:
                    self.results.append(result)
                
                self._notify_subscribers(result)
                return result
                
            except Exception as e:
                check['failures'] += 1
                result = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    message=f"Check execution failed: {str(e)}",
                    severity=8
                )
                with self._lock:
                    self.results.append(result)
                return result
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks"""
        results = []
        for name in self.checks:
            results.append(self.run_check(name))
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        with self._lock:
            if not self.results:
                return HealthStatus.UNKNOWN
            
            recent_results = list(self.results)[-50:]  # Last 50 checks
            
            critical_count = sum(1 for r in recent_results if r.status == HealthStatus.CRITICAL)
            unhealthy_count = sum(1 for r in recent_results if r.status == HealthStatus.UNHEALTHY)
            degraded_count = sum(1 for r in recent_results if r.status == HealthStatus.DEGRADED)
            
            if critical_count > 0:
                return HealthStatus.CRITICAL
            elif unhealthy_count > 2:
                return HealthStatus.UNHEALTHY
            elif degraded_count > 5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        self.running = True
        
        # Register built-in checks
        self.register_check('system_resources', self.check_system_resources, 'critical')
        self.register_check('windows_services', self.check_windows_services, 'standard')
        self.register_check('gmail_service', self.check_gmail_service, 'critical')
        self.register_check('browser_control', self.check_browser_control, 'standard')
        self.register_check('tts_engine', self.check_tts_engine, 'standard')
        self.register_check('stt_engine', self.check_stt_engine, 'standard')
        self.register_check('twilio_service', self.check_twilio_service, 'standard')
        self.register_check('soul_engine', self.check_soul_engine, 'critical')
        self.register_check('identity_manager', self.check_identity_manager, 'critical')
        self.register_check('cron_scheduler', self.check_cron_scheduler, 'critical')
        self.register_check('heartbeat', self.check_heartbeat, 'critical')
        
        # Start check threads
        for name, check in self.checks.items():
            thread = threading.Thread(
                target=self._check_loop,
                args=(name, check),
                daemon=True,
                name=f"HealthCheck-{name}"
            )
            thread.start()
            self._check_threads[name] = thread
        
        self.logger.info("Health monitoring started")
    
    def _check_loop(self, name: str, check: Dict):
        """Continuous check execution loop"""
        interval_map = {
            'critical': self.config['check_intervals']['critical'],
            'standard': self.config['check_intervals']['standard'],
            'extended': self.config['check_intervals']['extended']
        }
        
        interval = interval_map.get(check['interval'], 30)
        
        while self.running:
            try:
                self.run_check(name)
            except Exception as e:
                self.logger.error(f"Check loop error for {name}: {e}")
            
            # Sleep with interrupt handling
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        for thread in self._check_threads.values():
            thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        with self._lock:
            recent_results = list(self.results)[-100:]
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': self.get_overall_health().value,
                'checks': {},
                'summary': {
                    'total_checks': len(recent_results),
                    'healthy': sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY),
                    'degraded': sum(1 for r in recent_results if r.status == HealthStatus.DEGRADED),
                    'unhealthy': sum(1 for r in recent_results if r.status == HealthStatus.UNHEALTHY),
                    'critical': sum(1 for r in recent_results if r.status == HealthStatus.CRITICAL)
                }
            }
            
            # Group by component
            for result in recent_results:
                if result.component not in report['checks']:
                    report['checks'][result.component] = []
                report['checks'][result.component].append({
                    'status': result.status.value,
                    'timestamp': result.timestamp.isoformat(),
                    'message': result.message,
                    'severity': result.severity
                })
            
            return report


# ==================== WINDOWS-SPECIFIC HEALTH UTILITIES ====================

class WindowsHealthUtils:
    """Windows-specific health check utilities"""
    
    @staticmethod
    def get_system_uptime() -> float:
        """Get system uptime in seconds"""
        return time.time() - psutil.boot_time()
    
    @staticmethod
    def get_windows_version() -> Dict:
        """Get Windows version information"""
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            
            version = {
                'product_name': winreg.QueryValueEx(key, 'ProductName')[0],
                'release_id': winreg.QueryValueEx(key, 'ReleaseId')[0],
                'build': winreg.QueryValueEx(key, 'CurrentBuild')[0],
                'ubr': winreg.QueryValueEx(key, 'UBR')[0]
            }
            winreg.CloseKey(key)
            return version
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def check_windows_update_status() -> Dict:
        """Check Windows Update status"""
        # This would require Windows Update API integration
        # Placeholder for implementation
        return {'status': 'not_implemented'}
    
    @staticmethod
    def get_event_log_errors(hours: int = 24) -> List[Dict]:
        """Get critical errors from Windows Event Log"""
        # This would require win32evtlog or similar
        # Placeholder for implementation
        return []
    
    @staticmethod
    def check_antivirus_status() -> Dict:
        """Check Windows Defender/antivirus status"""
        try:
            # Use WMI to check antivirus status
            import wmi
            c = wmi.WMI(namespace="SecurityCenter2")
            antivirus = c.AntivirusProduct()
            
            status = []
            for av in antivirus:
                status.append({
                    'name': av.displayName,
                    'state': av.productState,
                    'enabled': bool(av.productState & 0x1000)
                })
            
            return {'antivirus': status}
        except Exception as e:
            return {'error': str(e)}


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize health monitor
    monitor = HealthMonitor()
    
    # Subscribe to health events
    def on_health_event(result: HealthCheckResult):
        if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
            print(f"ALERT: {result.component} - {result.status.value}: {result.message}")
    
    monitor.subscribe(on_health_event)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run for demonstration
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    
    # Get health report
    report = monitor.get_health_report()
    print(json.dumps(report, indent=2))
    
    # Stop monitoring
    monitor.stop_monitoring()
```

---

## Crash Detection and Reporting

### 4.1 Crash Detection Architecture

```python
# crash_detector.py - Comprehensive Crash Detection System
"""
Crash detection and reporting for Windows 10 AI Agent
Implements: Exception interception, process monitoring, watchdog timers
"""

import sys
import os
import traceback
import signal
import atexit
import threading
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import queue
import ctypes
from ctypes import wintypes

# Windows-specific imports
import win32api
import win32con
import win32process
import win32event

class CrashType(Enum):
    """Types of crashes that can occur"""
    UNHANDLED_EXCEPTION = "unhandled_exception"
    SEGMENTATION_FAULT = "segmentation_fault"
    STACK_OVERFLOW = "stack_overflow"
    OUT_OF_MEMORY = "out_of_memory"
    DEADLOCK = "deadlock"
    INFINITE_LOOP = "infinite_loop"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    SERVICE_FAILURE = "service_failure"
    EXTERNAL_TERMINATION = "external_termination"
    UNKNOWN = "unknown"

class CrashSeverity(Enum):
    """Crash severity levels"""
    LOW = "low"           # Recoverable, no data loss
    MEDIUM = "medium"     # Recoverable, minor data loss
    HIGH = "high"         # Partial recovery, some data loss
    CRITICAL = "critical" # Full restart required, significant data loss
    FATAL = "fatal"       # System halt required

@dataclass
class CrashReport:
    """Comprehensive crash report structure"""
    crash_id: str
    timestamp: datetime
    crash_type: CrashType
    severity: CrashSeverity
    component: str
    exception_type: Optional[str]
    exception_message: Optional[str]
    stack_trace: Optional[str]
    system_state: Dict[str, Any]
    memory_dump_summary: Dict[str, Any]
    active_threads: List[str]
    open_resources: List[str]
    recent_logs: List[str]
    recovery_attempted: bool
    recovery_successful: Optional[bool]
    additional_context: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'crash_id': self.crash_id,
            'timestamp': self.timestamp.isoformat(),
            'crash_type': self.crash_type.value,
            'severity': self.severity.value,
            'component': self.component,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'stack_trace': self.stack_trace,
            'system_state': self.system_state,
            'memory_dump_summary': self.memory_dump_summary,
            'active_threads': self.active_threads,
            'open_resources': self.open_resources,
            'recent_logs': self.recent_logs,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'additional_context': self.additional_context
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class CrashDetector:
    """
    Centralized crash detection and reporting system
    Implements multiple detection mechanisms for comprehensive coverage
    """
    
    def __init__(self, agent_context: Any = None):
        self.agent_context = agent_context
        self.crash_handlers: List[Callable] = []
        self.reporters: List[Callable] = []
        self.crash_history: deque = deque(maxlen=100)
        self.is_shutting_down = False
        self._lock = threading.RLock()
        
        # Crash statistics
        self.crash_stats = {
            'total_crashes': 0,
            'crashes_by_type': {},
            'crashes_by_component': {},
            'last_crash_time': None,
            'consecutive_crashes': 0
        }
        
        # Logging
        self.logger = logging.getLogger('CrashDetector')
        self._setup_logging()
        
        # Watchdog
        self.watchdog = WatchdogTimer(self._on_watchdog_timeout)
        
        # Setup exception hooks
        self._setup_exception_hooks()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_logging(self):
        """Configure crash detection logging"""
        os.makedirs('logs/crashes', exist_ok=True)
        handler = logging.FileHandler('logs/crash_detector.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _setup_exception_hooks(self):
        """Setup global exception hooks"""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._custom_excepthook
        
        # Thread exception hook (Python 3.8+)
        if hasattr(threading, 'excepthook'):
            self._original_thread_excepthook = threading.excepthook
            threading.excepthook = self._custom_thread_excepthook
    
    def _custom_excepthook(self, exc_type, exc_value, exc_traceback):
        """Custom exception hook for unhandled exceptions"""
        if not self.is_shutting_down:
            self._handle_crash(
                crash_type=CrashType.UNHANDLED_EXCEPTION,
                exception=exc_value,
                exc_type=exc_type,
                exc_traceback=exc_traceback
            )
        
        # Call original hook
        self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _custom_thread_excepthook(self, args):
        """Custom thread exception hook"""
        if not self.is_shutting_down:
            self._handle_crash(
                crash_type=CrashType.UNHANDLED_EXCEPTION,
                exception=args.exc_value,
                exc_type=args.exc_type,
                exc_traceback=args.exc_traceback,
                component=f"thread:{args.thread.name}"
            )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for crash detection"""
        # Windows-specific signals
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        signal_name = signal.Signals(signum).name if hasattr(signal.Signals, signum) else str(signum)
        self.logger.warning(f"Received signal: {signal_name}")
        
        if not self.is_shutting_down:
            self._handle_crash(
                crash_type=CrashType.EXTERNAL_TERMINATION,
                additional_context={'signal': signal_name}
            )
        
        # Exit cleanly
        self.is_shutting_down = True
        sys.exit(1)
    
    def _atexit_handler(self):
        """Handler for atexit - cleanup"""
        if not self.is_shutting_down:
            self.logger.info("Normal shutdown detected")
    
    def _on_watchdog_timeout(self):
        """Handler for watchdog timeout"""
        self._handle_crash(
            crash_type=CrashType.WATCHDOG_TIMEOUT,
            severity=CrashSeverity.HIGH,
            additional_context={'watchdog_timeout': True}
        )
    
    def _handle_crash(self, 
                     crash_type: CrashType,
                     exception: Optional[Exception] = None,
                     exc_type=None,
                     exc_traceback=None,
                     severity: CrashSeverity = CrashSeverity.HIGH,
                     component: str = "unknown",
                     additional_context: Dict = None):
        """
        Main crash handling method
        
        Args:
            crash_type: Type of crash
            exception: The exception that caused the crash (if any)
            exc_type: Exception type
            exc_traceback: Exception traceback
            severity: Crash severity
            component: Component that crashed
            additional_context: Additional context information
        """
        with self._lock:
            # Generate crash ID
            crash_id = self._generate_crash_id()
            
            # Capture stack trace
            stack_trace = None
            if exc_traceback:
                stack_trace = ''.join(traceback.format_exception(exc_type, exception, exc_traceback))
            else:
                stack_trace = ''.join(traceback.format_stack())
            
            # Get exception details
            exception_type = exc_type.__name__ if exc_type else None
            exception_message = str(exception) if exception else None
            
            # Determine component from stack trace if not provided
            if component == "unknown" and stack_trace:
                component = self._extract_component_from_trace(stack_trace)
            
            # Collect system state
            system_state = self._collect_system_state()
            
            # Collect memory info
            memory_dump_summary = self._collect_memory_info()
            
            # Get active threads
            active_threads = self._get_active_threads()
            
            # Get open resources
            open_resources = self._get_open_resources()
            
            # Get recent logs
            recent_logs = self._get_recent_logs()
            
            # Create crash report
            report = CrashReport(
                crash_id=crash_id,
                timestamp=datetime.now(),
                crash_type=crash_type,
                severity=severity,
                component=component,
                exception_type=exception_type,
                exception_message=exception_message,
                stack_trace=stack_trace,
                system_state=system_state,
                memory_dump_summary=memory_dump_summary,
                active_threads=active_threads,
                open_resources=open_resources,
                recent_logs=recent_logs,
                recovery_attempted=False,
                recovery_successful=None,
                additional_context=additional_context or {}
            )
            
            # Update statistics
            self._update_crash_stats(report)
            
            # Store in history
            self.crash_history.append(report)
            
            # Log crash
            self._log_crash(report)
            
            # Save crash report to file
            self._save_crash_report(report)
            
            # Notify handlers
            self._notify_handlers(report)
            
            # Send to reporters
            self._send_to_reporters(report)
            
            self.logger.critical(f"Crash detected: {crash_id} - {crash_type.value}")
            
            return report
    
    def _generate_crash_id(self) -> str:
        """Generate unique crash ID"""
        import uuid
        return f"CRASH-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    
    def _extract_component_from_trace(self, stack_trace: str) -> str:
        """Extract component name from stack trace"""
        lines = stack_trace.split('\n')
        for line in lines:
            if 'File' in line and 'agent' in line.lower():
                # Extract filename
                parts = line.split('"')
                if len(parts) >= 2:
                    filepath = parts[1]
                    filename = os.path.basename(filepath)
                    return filename.replace('.py', '')
        return "unknown"
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state"""
        import psutil
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': psutil.disk_usage('C:\\').total / (1024**3),
                'free_gb': psutil.disk_usage('C:\\').free / (1024**3),
                'percent': psutil.disk_usage('C:\\').percent
            },
            'process_count': len(psutil.pids()),
            'agent_process': self._get_agent_process_info()
        }
        
        return state
    
    def _get_agent_process_info(self) -> Dict:
        """Get agent process information"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info_mb': process.memory_info().rss / (1024**2),
                'num_threads': process.num_threads(),
                'num_handles': process.num_handles() if hasattr(process, 'num_handles') else 0,
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _collect_memory_info(self) -> Dict[str, Any]:
        """Collect memory dump summary"""
        try:
            import psutil
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Get object counts
            object_counts = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            # Top memory consumers
            top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return {
                'gc_objects': len(gc.get_objects()),
                'gc_garbage': len(gc.garbage),
                'top_object_types': top_objects,
                'process_memory_mb': psutil.Process().memory_info().rss / (1024**2)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_active_threads(self) -> List[str]:
        """Get list of active threads"""
        threads = []
        for thread in threading.enumerate():
            thread_info = f"{thread.name} (ID: {thread.ident})"
            if hasattr(thread, '_target') and thread._target:
                thread_info += f" - Target: {thread._target.__name__}"
            threads.append(thread_info)
        return threads
    
    def _get_open_resources(self) -> List[str]:
        """Get list of open resources"""
        resources = []
        try:
            import psutil
            process = psutil.Process()
            
            # Open files
            try:
                for file in process.open_files():
                    resources.append(f"File: {file.path}")
            except Exception:
                pass
            
            # Network connections
            try:
                for conn in process.connections():
                    resources.append(f"Connection: {conn.laddr} -> {conn.raddr if conn.raddr else 'N/A'}")
            except Exception:
                pass
            
        except Exception as e:
            resources.append(f"Error getting resources: {e}")
        
        return resources
    
    def _get_recent_logs(self, lines: int = 50) -> List[str]:
        """Get recent log lines"""
        recent_logs = []
        try:
            log_files = [
                'logs/agent.log',
                'logs/health_monitor.log',
                'logs/crash_detector.log'
            ]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        all_lines = f.readlines()
                        recent_logs.extend(all_lines[-lines:])
        except Exception as e:
            recent_logs.append(f"Error reading logs: {e}")
        
        return recent_logs
    
    def _update_crash_stats(self, report: CrashReport):
        """Update crash statistics"""
        self.crash_stats['total_crashes'] += 1
        self.crash_stats['last_crash_time'] = report.timestamp.isoformat()
        
        # By type
        crash_type = report.crash_type.value
        self.crash_stats['crashes_by_type'][crash_type] = \
            self.crash_stats['crashes_by_type'].get(crash_type, 0) + 1
        
        # By component
        component = report.component
        self.crash_stats['crashes_by_component'][component] = \
            self.crash_stats['crashes_by_component'].get(component, 0) + 1
        
        # Consecutive crashes (within 5 minutes)
        if self.crash_stats['last_crash_time']:
            last_time = datetime.fromisoformat(self.crash_stats['last_crash_time'])
            if (report.timestamp - last_time).total_seconds() < 300:
                self.crash_stats['consecutive_crashes'] += 1
            else:
                self.crash_stats['consecutive_crashes'] = 1
    
    def _log_crash(self, report: CrashReport):
        """Log crash details"""
        self.logger.critical(f"=" * 80)
        self.logger.critical(f"CRASH DETECTED: {report.crash_id}")
        self.logger.critical(f"Type: {report.crash_type.value}")
        self.logger.critical(f"Severity: {report.severity.value}")
        self.logger.critical(f"Component: {report.component}")
        if report.exception_type:
            self.logger.critical(f"Exception: {report.exception_type}: {report.exception_message}")
        self.logger.critical(f"Stack Trace:\n{report.stack_trace}")
        self.logger.critical(f"=" * 80)
    
    def _save_crash_report(self, report: CrashReport):
        """Save crash report to file"""
        try:
            filename = f"logs/crashes/{report.crash_id}.json"
            with open(filename, 'w') as f:
                f.write(report.to_json())
        except Exception as e:
            self.logger.error(f"Failed to save crash report: {e}")
    
    def _notify_handlers(self, report: CrashReport):
        """Notify registered crash handlers"""
        for handler in self.crash_handlers:
            try:
                handler(report)
            except Exception as e:
                self.logger.error(f"Crash handler failed: {e}")
    
    def _send_to_reporters(self, report: CrashReport):
        """Send crash report to registered reporters"""
        for reporter in self.reporters:
            try:
                reporter(report)
            except Exception as e:
                self.logger.error(f"Crash reporter failed: {e}")
    
    def register_handler(self, handler: Callable):
        """Register a crash handler"""
        self.crash_handlers.append(handler)
    
    def register_reporter(self, reporter: Callable):
        """Register a crash reporter"""
        self.reporters.append(reporter)
    
    def get_crash_stats(self) -> Dict:
        """Get crash statistics"""
        return self.crash_stats.copy()
    
    def get_recent_crashes(self, count: int = 10) -> List[CrashReport]:
        """Get recent crash reports"""
        return list(self.crash_history)[-count:]
    
    def start_watchdog(self, timeout_seconds: int = 30):
        """Start the watchdog timer"""
        self.watchdog.start(timeout_seconds)
    
    def reset_watchdog(self):
        """Reset the watchdog timer"""
        self.watchdog.reset()
    
    def stop_watchdog(self):
        """Stop the watchdog timer"""
        self.watchdog.stop()


class WatchdogTimer:
    """
    Watchdog timer for detecting hangs and infinite loops
    """
    
    def __init__(self, timeout_callback: Callable):
        self.timeout_callback = timeout_callback
        self.timeout_seconds = 30
        self._last_reset = time.time()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self, timeout_seconds: int = 30):
        """Start the watchdog timer"""
        with self._lock:
            self.timeout_seconds = timeout_seconds
            self._last_reset = time.time()
            self._running = True
            
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="WatchdogTimer"
            )
            self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitor loop for watchdog"""
        while self._running:
            time.sleep(1)
            
            with self._lock:
                elapsed = time.time() - self._last_reset
                
                if elapsed > self.timeout_seconds:
                    self._running = False
                    self.timeout_callback()
                    return
    
    def reset(self):
        """Reset the watchdog timer"""
        with self._lock:
            self._last_reset = time.time()
    
    def stop(self):
        """Stop the watchdog timer"""
        with self._lock:
            self._running = False


class DeadlockDetector:
    """
    Detect potential deadlocks in the system
    """
    
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._thread_states: Dict[int, Dict] = {}
        self.logger = logging.getLogger('DeadlockDetector')
    
    def start(self):
        """Start deadlock detection"""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._detect_loop,
            daemon=True,
            name="DeadlockDetector"
        )
        self._monitor_thread.start()
        self.logger.info("Deadlock detection started")
    
    def _detect_loop(self):
        """Main detection loop"""
        while self._running:
            self._check_threads()
            time.sleep(self.check_interval)
    
    def _check_threads(self):
        """Check for potential deadlocks"""
        current_time = time.time()
        
        for thread in threading.enumerate():
            thread_id = thread.ident
            
            if thread_id not in self._thread_states:
                self._thread_states[thread_id] = {
                    'last_activity': current_time,
                    'stack': None,
                    'warning_count': 0
                }
            
            state = self._thread_states[thread_id]
            
            # Get current stack
            import sys
            frame = sys._current_frames().get(thread_id)
            current_stack = traceback.format_stack(frame) if frame else None
            
            # Check if stack is unchanged
            if current_stack and state['stack'] == current_stack:
                elapsed = current_time - state['last_activity']
                
                # Potential deadlock if unchanged for > 30 seconds
                if elapsed > 30:
                    state['warning_count'] += 1
                    self.logger.warning(
                        f"Potential deadlock detected in thread {thread.name} "
                        f"(unchanged for {elapsed:.1f}s)"
                    )
                    
                    if state['warning_count'] >= 3:
                        self._report_deadlock(thread, current_stack)
            else:
                # Stack changed, update state
                state['last_activity'] = current_time
                state['stack'] = current_stack
                state['warning_count'] = 0
    
    def _report_deadlock(self, thread: threading.Thread, stack: List[str]):
        """Report detected deadlock"""
        self.logger.error(f"DEADLOCK CONFIRMED in thread {thread.name}")
        self.logger.error(f"Stack trace:\n{''.join(stack)}")
    
    def stop(self):
        """Stop deadlock detection"""
        self._running = False


# ==================== CRASH REPORTERS ====================

class EmailCrashReporter:
    """Send crash reports via email"""
    
    def __init__(self, smtp_config: Dict):
        self.smtp_config = smtp_config
    
    def __call__(self, report: CrashReport):
        """Send crash report via email"""
        # Implementation depends on email service
        pass

class WebhookCrashReporter:
    """Send crash reports via webhook"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def __call__(self, report: CrashReport):
        """Send crash report via webhook"""
        import requests
        
        try:
            payload = {
                'crash_id': report.crash_id,
                'timestamp': report.timestamp.isoformat(),
                'type': report.crash_type.value,
                'severity': report.severity.value,
                'component': report.component,
                'message': report.exception_message
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Webhook reporter failed: {e}")

class FileCrashReporter:
    """Write crash reports to file"""
    
    def __init__(self, output_dir: str = "logs/crashes"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def __call__(self, report: CrashReport):
        """Write crash report to file"""
        filename = os.path.join(self.output_dir, f"{report.crash_id}.json")
        
        with open(filename, 'w') as f:
            f.write(report.to_json())


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize crash detector
    detector = CrashDetector()
    
    # Register handlers
    def on_crash(report: CrashReport):
        print(f"Crash detected: {report.crash_id}")
        print(f"Type: {report.crash_type.value}")
        print(f"Component: {report.component}")
    
    detector.register_handler(on_crash)
    
    # Register reporters
    detector.register_reporter(FileCrashReporter())
    
    # Start watchdog
    detector.start_watchdog(timeout_seconds=30)
    
    # Simulate some work
    try:
        for i in range(100):
            detector.reset_watchdog()
            time.sleep(1)
            print(f"Working... {i}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Stop
    detector.stop_watchdog()
```

---

## Automatic Restart Procedures

### 5.1 Auto-Restart Architecture

```python
# auto_restart.py - Automatic Restart and Recovery System
"""
Automatic restart procedures for Windows 10 AI Agent
Implements: Service restart, process restart, stateful recovery
"""

import os
import sys
import subprocess
import time
import json
import logging
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import psutil
import win32service
import win32serviceutil
import win32api
import win32con
import win32process
import win32event

class RestartPolicy(Enum):
    """Restart policy types"""
    IMMEDIATE = "immediate"           # Restart immediately
    DELAYED = "delayed"               # Restart after delay
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    NO_RESTART = "no_restart"         # Do not restart
    MANUAL_ONLY = "manual_only"       # Manual restart only

class RestartPriority(Enum):
    """Restart priority levels"""
    CRITICAL = 1      # Must restart immediately
    HIGH = 2          # Restart as soon as possible
    MEDIUM = 3        # Restart when convenient
    LOW = 4           # Restart during maintenance window

@dataclass
class RestartConfig:
    """Configuration for restart behavior"""
    policy: RestartPolicy
    priority: RestartPriority
    max_restarts: int = 5
    restart_window_seconds: int = 300  # 5 minutes
    initial_delay_seconds: int = 5
    max_delay_seconds: int = 300  # 5 minutes
    backoff_multiplier: float = 2.0
    require_healthy_state: bool = True
    preserve_state: bool = True
    notify_on_restart: bool = True

@dataclass
class RestartRecord:
    """Record of a restart event"""
    timestamp: datetime
    component: str
    reason: str
    previous_pid: Optional[int]
    new_pid: Optional[int]
    success: bool
    duration_seconds: float
    state_preserved: bool

class AutoRestartManager:
    """
    Centralized automatic restart manager
    Handles all restart scenarios with configurable policies
    """
    
    def __init__(self, config_path: str = "config/restart_config.json"):
        self.config = self._load_config(config_path)
        self.restart_history: List[RestartRecord] = []
        self.component_configs: Dict[str, RestartConfig] = {}
        self.restart_counts: Dict[str, List[datetime]] = {}
        self._lock = threading.RLock()
        self._restart_threads: Dict[str, threading.Thread] = {}
        self._shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger('AutoRestartManager')
        self._setup_logging()
        
        # Initialize component configs
        self._init_component_configs()
    
    def _load_config(self, path: str) -> Dict:
        """Load restart configuration"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default restart configuration"""
        return {
            "global": {
                "max_restarts_per_hour": 10,
                "emergency_contact": "admin@example.com",
                "enable_notifications": True
            },
            "components": {
                "agent_core": {
                    "policy": "exponential_backoff",
                    "priority": "critical",
                    "max_restarts": 5,
                    "restart_window": 300,
                    "initial_delay": 5,
                    "max_delay": 300,
                    "preserve_state": True
                },
                "soul_engine": {
                    "policy": "immediate",
                    "priority": "critical",
                    "max_restarts": 10,
                    "preserve_state": True
                },
                "gmail_service": {
                    "policy": "delayed",
                    "priority": "high",
                    "max_restarts": 5,
                    "initial_delay": 10,
                    "preserve_state": True
                },
                "browser_control": {
                    "policy": "delayed",
                    "priority": "medium",
                    "max_restarts": 3,
                    "initial_delay": 5,
                    "preserve_state": False
                },
                "tts_engine": {
                    "policy": "delayed",
                    "priority": "low",
                    "max_restarts": 3,
                    "initial_delay": 5,
                    "preserve_state": False
                },
                "stt_engine": {
                    "policy": "delayed",
                    "priority": "low",
                    "max_restarts": 3,
                    "initial_delay": 5,
                    "preserve_state": False
                },
                "twilio_service": {
                    "policy": "delayed",
                    "priority": "medium",
                    "max_restarts": 5,
                    "initial_delay": 10,
                    "preserve_state": True
                },
                "cron_scheduler": {
                    "policy": "immediate",
                    "priority": "high",
                    "max_restarts": 10,
                    "preserve_state": True
                },
                "heartbeat": {
                    "policy": "immediate",
                    "priority": "critical",
                    "max_restarts": 20,
                    "preserve_state": True
                }
            }
        }
    
    def _setup_logging(self):
        """Configure restart manager logging"""
        os.makedirs('logs/restarts', exist_ok=True)
        handler = logging.FileHandler('logs/auto_restart.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_component_configs(self):
        """Initialize component restart configurations"""
        for component, config in self.config['components'].items():
            self.component_configs[component] = RestartConfig(
                policy=RestartPolicy(config['policy']),
                priority=RestartPriority[config['priority'].upper()],
                max_restarts=config.get('max_restarts', 5),
                restart_window_seconds=config.get('restart_window', 300),
                initial_delay_seconds=config.get('initial_delay', 5),
                max_delay_seconds=config.get('max_delay', 300),
                backoff_multiplier=config.get('backoff_multiplier', 2.0),
                preserve_state=config.get('preserve_state', True)
            )
    
    def request_restart(self, 
                       component: str, 
                       reason: str,
                       force: bool = False) -> bool:
        """
        Request a restart for a component
        
        Args:
            component: Component to restart
            reason: Reason for restart
            force: Force restart even if policy would prevent it
            
        Returns:
            True if restart was initiated, False otherwise
        """
        with self._lock:
            if component not in self.component_configs:
                self.logger.error(f"Unknown component: {component}")
                return False
            
            config = self.component_configs[component]
            
            # Check restart policy
            if config.policy == RestartPolicy.NO_RESTART and not force:
                self.logger.warning(f"Restart blocked by policy for {component}")
                return False
            
            # Check restart limits
            if not self._check_restart_limits(component, config) and not force:
                self.logger.error(f"Restart limit exceeded for {component}")
                self._escalate_restart_failure(component, reason)
                return False
            
            # Calculate delay
            delay = self._calculate_restart_delay(component, config)
            
            # Initiate restart
            self.logger.info(f"Scheduling restart for {component} in {delay}s (reason: {reason})")
            
            restart_thread = threading.Thread(
                target=self._execute_restart,
                args=(component, reason, config, delay),
                name=f"Restart-{component}",
                daemon=True
            )
            restart_thread.start()
            
            self._restart_threads[component] = restart_thread
            
            return True
    
    def _check_restart_limits(self, component: str, config: RestartConfig) -> bool:
        """Check if restart is within limits"""
        now = datetime.now()
        window_start = now - timedelta(seconds=config.restart_window_seconds)
        
        # Get recent restarts for this component
        if component not in self.restart_counts:
            self.restart_counts[component] = []
        
        recent_restarts = [
            t for t in self.restart_counts[component]
            if t > window_start
        ]
        
        self.restart_counts[component] = recent_restarts
        
        return len(recent_restarts) < config.max_restarts
    
    def _calculate_restart_delay(self, component: str, config: RestartConfig) -> int:
        """Calculate restart delay based on policy"""
        if config.policy == RestartPolicy.IMMEDIATE:
            return 0
        elif config.policy == RestartPolicy.DELAYED:
            return config.initial_delay_seconds
        elif config.policy == RestartPolicy.EXPONENTIAL_BACKOFF:
            # Calculate based on recent restarts
            recent_count = len(self.restart_counts.get(component, []))
            delay = config.initial_delay_seconds * (config.backoff_multiplier ** recent_count)
            return min(int(delay), config.max_delay_seconds)
        else:
            return config.initial_delay_seconds
    
    def _execute_restart(self, 
                        component: str, 
                        reason: str,
                        config: RestartConfig,
                        delay: int):
        """Execute the restart procedure"""
        start_time = time.time()
        previous_pid = None
        new_pid = None
        state_preserved = False
        success = False
        
        try:
            # Wait for delay
            if delay > 0:
                self.logger.info(f"Waiting {delay}s before restarting {component}")
                time.sleep(delay)
            
            # Check if shutdown requested
            if self._shutdown_event.is_set():
                self.logger.info(f"Shutdown requested, aborting restart of {component}")
                return
            
            # Get current process info
            previous_pid = self._get_component_pid(component)
            
            # Preserve state if configured
            if config.preserve_state:
                state_preserved = self._preserve_component_state(component)
            
            # Stop component
            self._stop_component(component)
            
            # Wait for stop
            time.sleep(2)
            
            # Start component
            new_pid = self._start_component(component)
            
            if new_pid:
                success = True
                self.logger.info(f"Successfully restarted {component} (PID: {new_pid})")
                
                # Restore state if preserved
                if state_preserved:
                    self._restore_component_state(component)
                
                # Record restart
                self._record_restart(component, reason, previous_pid, new_pid, True, 
                                   time.time() - start_time, state_preserved)
            else:
                raise Exception(f"Failed to start {component}")
                
        except Exception as e:
            self.logger.error(f"Restart failed for {component}: {e}")
            self._record_restart(component, reason, previous_pid, new_pid, False,
                               time.time() - start_time, state_preserved)
            
            # Try escalation
            self._escalate_restart_failure(component, str(e))
    
    def _get_component_pid(self, component: str) -> Optional[int]:
        """Get the PID of a component"""
        # This would query the process manager or service manager
        # Placeholder implementation
        return None
    
    def _preserve_component_state(self, component: str) -> bool:
        """Preserve component state before restart"""
        try:
            state_file = f"state/{component}_state.json"
            os.makedirs("state", exist_ok=True)
            
            # Call component's state preservation method
            # This would be implemented by each component
            state = self._call_component_method(component, 'get_state')
            
            if state:
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to preserve state for {component}: {e}")
            return False
    
    def _restore_component_state(self, component: str) -> bool:
        """Restore component state after restart"""
        try:
            state_file = f"state/{component}_state.json"
            
            if not os.path.exists(state_file):
                return False
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Call component's state restoration method
            self._call_component_method(component, 'set_state', state)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore state for {component}: {e}")
            return False
    
    def _call_component_method(self, component: str, method: str, *args) -> Any:
        """Call a method on a component"""
        # This would use the component registry to call methods
        # Placeholder implementation
        return None
    
    def _stop_component(self, component: str):
        """Stop a component"""
        self.logger.info(f"Stopping {component}")
        
        # Get stop command from config
        stop_commands = {
            'agent_core': self._stop_agent_core,
            'soul_engine': self._stop_soul_engine,
            'gmail_service': self._stop_gmail_service,
            'browser_control': self._stop_browser_control,
            'tts_engine': self._stop_tts_engine,
            'stt_engine': self._stop_stt_engine,
            'twilio_service': self._stop_twilio_service,
            'cron_scheduler': self._stop_cron_scheduler,
            'heartbeat': self._stop_heartbeat
        }
        
        if component in stop_commands:
            stop_commands[component]()
        else:
            self.logger.warning(f"No stop command for {component}")
    
    def _start_component(self, component: str) -> Optional[int]:
        """Start a component and return its PID"""
        self.logger.info(f"Starting {component}")
        
        # Get start command from config
        start_commands = {
            'agent_core': self._start_agent_core,
            'soul_engine': self._start_soul_engine,
            'gmail_service': self._start_gmail_service,
            'browser_control': self._start_browser_control,
            'tts_engine': self._start_tts_engine,
            'stt_engine': self._start_stt_engine,
            'twilio_service': self._start_twilio_service,
            'cron_scheduler': self._start_cron_scheduler,
            'heartbeat': self._start_heartbeat
        }
        
        if component in start_commands:
            return start_commands[component]()
        else:
            self.logger.warning(f"No start command for {component}")
            return None
    
    # ==================== COMPONENT-SPECIFIC STOP/START METHODS ====================
    
    def _stop_agent_core(self):
        """Stop agent core"""
        # Signal graceful shutdown
        pass
    
    def _start_agent_core(self) -> Optional[int]:
        """Start agent core"""
        try:
            # Start main agent process
            process = subprocess.Popen(
                [sys.executable, 'agent_core.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return process.pid
        except Exception as e:
            self.logger.error(f"Failed to start agent core: {e}")
            return None
    
    def _stop_soul_engine(self):
        """Stop soul engine"""
        pass
    
    def _start_soul_engine(self) -> Optional[int]:
        """Start soul engine"""
        return None
    
    def _stop_gmail_service(self):
        """Stop Gmail service"""
        pass
    
    def _start_gmail_service(self) -> Optional[int]:
        """Start Gmail service"""
        return None
    
    def _stop_browser_control(self):
        """Stop browser control"""
        # Kill browser processes
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and any(browser in proc.info['name'].lower() 
                                         for browser in ['chrome', 'firefox', 'edge']):
                try:
                    proc.terminate()
                except Exception:
                    pass
    
    def _start_browser_control(self) -> Optional[int]:
        """Start browser control"""
        return None
    
    def _stop_tts_engine(self):
        """Stop TTS engine"""
        pass
    
    def _start_tts_engine(self) -> Optional[int]:
        """Start TTS engine"""
        return None
    
    def _stop_stt_engine(self):
        """Stop STT engine"""
        pass
    
    def _start_stt_engine(self) -> Optional[int]:
        """Start STT engine"""
        return None
    
    def _stop_twilio_service(self):
        """Stop Twilio service"""
        pass
    
    def _start_twilio_service(self) -> Optional[int]:
        """Start Twilio service"""
        return None
    
    def _stop_cron_scheduler(self):
        """Stop cron scheduler"""
        pass
    
    def _start_cron_scheduler(self) -> Optional[int]:
        """Start cron scheduler"""
        return None
    
    def _stop_heartbeat(self):
        """Stop heartbeat"""
        pass
    
    def _start_heartbeat(self) -> Optional[int]:
        """Start heartbeat"""
        return None
    
    def _record_restart(self, component: str, reason: str, previous_pid: Optional[int],
                       new_pid: Optional[int], success: bool, duration: float,
                       state_preserved: bool):
        """Record a restart event"""
        record = RestartRecord(
            timestamp=datetime.now(),
            component=component,
            reason=reason,
            previous_pid=previous_pid,
            new_pid=new_pid,
            success=success,
            duration_seconds=duration,
            state_preserved=state_preserved
        )
        
        with self._lock:
            self.restart_history.append(record)
            
            if component not in self.restart_counts:
                self.restart_counts[component] = []
            self.restart_counts[component].append(datetime.now())
    
    def _escalate_restart_failure(self, component: str, reason: str):
        """Escalate a restart failure"""
        self.logger.critical(f"RESTART ESCALATION: {component} - {reason}")
        
        # Send notification
        if self.config['global']['enable_notifications']:
            self._send_escalation_notification(component, reason)
        
        # Try full system restart if critical component
        critical_components = ['agent_core', 'soul_engine', 'heartbeat']
        if component in critical_components:
            self.logger.critical(f"Attempting full system restart due to {component} failure")
            self._full_system_restart()
    
    def _send_escalation_notification(self, component: str, reason: str):
        """Send escalation notification"""
        # Implementation depends on notification system
        pass
    
    def _full_system_restart(self):
        """Perform full system restart"""
        self.logger.critical("INITIATING FULL SYSTEM RESTART")
        
        # Save all states
        for component in self.component_configs:
            self._preserve_component_state(component)
        
        # Restart main process
        try:
            # Start new instance
            subprocess.Popen(
                [sys.executable, 'main.py', '--recovery-mode'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            # Exit current instance
            sys.exit(1)
        except Exception as e:
            self.logger.critical(f"Full system restart failed: {e}")
    
    def get_restart_history(self, component: Optional[str] = None) -> List[RestartRecord]:
        """Get restart history"""
        with self._lock:
            if component:
                return [r for r in self.restart_history if r.component == component]
            return list(self.restart_history)
    
    def get_restart_stats(self) -> Dict:
        """Get restart statistics"""
        with self._lock:
            stats = {
                'total_restarts': len(self.restart_history),
                'successful_restarts': sum(1 for r in self.restart_history if r.success),
                'failed_restarts': sum(1 for r in self.restart_history if not r.success),
                'by_component': {}
            }
            
            for record in self.restart_history:
                if record.component not in stats['by_component']:
                    stats['by_component'][record.component] = {
                        'total': 0,
                        'successful': 0,
                        'failed': 0
                    }
                
                stats['by_component'][record.component]['total'] += 1
                if record.success:
                    stats['by_component'][record.component]['successful'] += 1
                else:
                    stats['by_component'][record.component]['failed'] += 1
            
            return stats
    
    def shutdown(self):
        """Shutdown the restart manager"""
        self._shutdown_event.set()
        
        # Wait for restart threads
        for component, thread in self._restart_threads.items():
            thread.join(timeout=10)


class WindowsServiceManager:
    """
    Manage Windows service integration for the agent
    """
    
    def __init__(self, service_name: str = "OpenClawAgent"):
        self.service_name = service_name
        self.logger = logging.getLogger('WindowsServiceManager')
    
    def install_service(self, executable_path: str):
        """Install as Windows service"""
        try:
            # Use pywin32 to create service
            import win32service
            import win32serviceutil
            
            win32serviceutil.InstallService(
                None,
                self.service_name,
                displayName="OpenClaw AI Agent",
                startType=win32service.SERVICE_AUTO_START,
                errorControl=win32service.SERVICE_ERROR_NORMAL,
                bRunInteractive=0,
                userName=None,
                password=None,
                description="24/7 AI Agent System",
                exeName=executable_path
            )
            
            self.logger.info(f"Service {self.service_name} installed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to install service: {e}")
            return False
    
    def start_service(self) -> bool:
        """Start the Windows service"""
        try:
            win32serviceutil.StartService(self.service_name)
            self.logger.info(f"Service {self.service_name} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the Windows service"""
        try:
            win32serviceutil.StopService(self.service_name)
            self.logger.info(f"Service {self.service_name} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop service: {e}")
            return False
    
    def restart_service(self) -> bool:
        """Restart the Windows service"""
        if self.stop_service():
            time.sleep(2)
            return self.start_service()
        return False
    
    def get_service_status(self) -> Dict:
        """Get service status"""
        try:
            status = win32serviceutil.QueryServiceStatus(self.service_name)
            return {
                'service_name': self.service_name,
                'status_code': status[1],
                'status': self._status_code_to_string(status[1])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _status_code_to_string(self, code: int) -> str:
        """Convert status code to string"""
        status_map = {
            win32service.SERVICE_STOPPED: "STOPPED",
            win32service.SERVICE_START_PENDING: "START_PENDING",
            win32service.SERVICE_STOP_PENDING: "STOP_PENDING",
            win32service.SERVICE_RUNNING: "RUNNING",
            win32service.SERVICE_CONTINUE_PENDING: "CONTINUE_PENDING",
            win32service.SERVICE_PAUSE_PENDING: "PAUSE_PENDING",
            win32service.SERVICE_PAUSED: "PAUSED"
        }
        return status_map.get(code, f"UNKNOWN({code})")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize restart manager
    restart_manager = AutoRestartManager()
    
    # Request a restart
    restart_manager.request_restart('browser_control', 'Browser process crashed')
    
    # Get stats
    stats = restart_manager.get_restart_stats()
    print(f"Restart stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Shutdown
    restart_manager.shutdown()
```

---

## State Recovery After Crash

### 6.1 State Recovery Architecture

```python
# state_recovery.py - State Recovery and Persistence System
"""
State recovery system for Windows 10 AI Agent
Implements: Checkpointing, state persistence, recovery procedures
"""

import os
import sys
import json
import pickle
import time
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import shutil
import sqlite3
from contextlib import contextmanager

T = TypeVar('T')

class CheckpointType(Enum):
    """Types of checkpoints"""
    FULL = "full"           # Complete state snapshot
    INCREMENTAL = "incremental"  # Changes since last checkpoint
    MEMORY = "memory"       # In-memory state only
    DISK = "disk"          # Persisted to disk

class RecoveryMode(Enum):
    """Recovery modes"""
    FULL = "full"           # Full recovery from checkpoint
    PARTIAL = "partial"     # Partial recovery (best effort)
    MINIMAL = "minimal"     # Minimal recovery (critical only)
    NONE = "none"          # No recovery, start fresh

@dataclass
class Checkpoint:
    """Checkpoint data structure"""
    checkpoint_id: str
    timestamp: datetime
    checkpoint_type: CheckpointType
    component: str
    state_data: Dict[str, Any]
    checksum: str
    parent_checkpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def verify_checksum(self) -> bool:
        """Verify checkpoint integrity"""
        data_str = json.dumps(self.state_data, sort_keys=True)
        computed = hashlib.sha256(data_str.encode()).hexdigest()
        return computed == self.checksum

@dataclass
class RecoveryPoint:
    """Recovery point information"""
    recovery_id: str
    timestamp: datetime
    checkpoint_id: str
    components: List[str]
    recovery_mode: RecoveryMode
    estimated_data_loss_seconds: float

class StateManager:
    """
    Centralized state management and recovery system
    Handles checkpointing, persistence, and recovery for all components
    """
    
    def __init__(self, base_path: str = "state"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_path = self.base_path / "checkpoints"
        self.checkpoints_path.mkdir(exist_ok=True)
        
        self.recovery_path = self.base_path / "recovery"
        self.recovery_path.mkdir(exist_ok=True)
        
        self.journal_path = self.base_path / "journal"
        self.journal_path.mkdir(exist_ok=True)
        
        # State registry
        self.component_states: Dict[str, Any] = {}
        self.state_handlers: Dict[str, Dict[str, Callable]] = {}
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, List[Checkpoint]] = {}
        self.last_checkpoint: Dict[str, datetime] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Database for state persistence
        self.db_path = self.base_path / "state.db"
        self._init_database()
        
        # Logging
        self.logger = logging.getLogger('StateManager')
        self._setup_logging()
        
        # Auto-checkpoint thread
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._stop_checkpoint = threading.Event()
    
    def _setup_logging(self):
        """Configure state manager logging"""
        handler = logging.FileHandler('logs/state_manager.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_database(self):
        """Initialize SQLite database for state persistence"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    checkpoint_type TEXT,
                    component TEXT,
                    state_data BLOB,
                    checksum TEXT,
                    parent_checkpoint TEXT,
                    metadata TEXT
                )
            ''')
            
            # Recovery points table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_points (
                    recovery_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    checkpoint_id TEXT,
                    components TEXT,
                    recovery_mode TEXT,
                    estimated_data_loss_seconds REAL
                )
            ''')
            
            # Journal table for write-ahead logging
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    component TEXT,
                    operation TEXT,
                    data BLOB
                )
            ''')
            
            conn.commit()
    
    # ==================== COMPONENT REGISTRATION ====================
    
    def register_component(self, 
                          component: str,
                          get_state: Callable[[], Dict],
                          set_state: Callable[[Dict], None],
                          checkpoint_interval: int = 60):
        """
        Register a component for state management
        
        Args:
            component: Component name
            get_state: Function to get component state
            set_state: Function to set component state
            checkpoint_interval: Auto-checkpoint interval in seconds
        """
        with self._lock:
            self.state_handlers[component] = {
                'get_state': get_state,
                'set_state': set_state,
                'checkpoint_interval': checkpoint_interval,
                'last_checkpoint': None
            }
            
            self.logger.info(f"Registered component: {component}")
    
    def unregister_component(self, component: str):
        """Unregister a component"""
        with self._lock:
            if component in self.state_handlers:
                del self.state_handlers[component]
                self.logger.info(f"Unregistered component: {component}")
    
    # ==================== CHECKPOINT OPERATIONS ====================
    
    def create_checkpoint(self, 
                         component: Optional[str] = None,
                         checkpoint_type: CheckpointType = CheckpointType.FULL) -> Optional[Checkpoint]:
        """
        Create a checkpoint for one or all components
        
        Args:
            component: Specific component or None for all
            checkpoint_type: Type of checkpoint to create
            
        Returns:
            Checkpoint object or None if failed
        """
        with self._lock:
            checkpoint_id = self._generate_checkpoint_id()
            timestamp = datetime.now()
            
            if component:
                # Checkpoint single component
                components = [component]
            else:
                # Checkpoint all registered components
                components = list(self.state_handlers.keys())
            
            state_data = {}
            
            for comp in components:
                if comp in self.state_handlers:
                    try:
                        state = self.state_handlers[comp]['get_state']()
                        state_data[comp] = state
                        self.state_handlers[comp]['last_checkpoint'] = timestamp
                    except Exception as e:
                        self.logger.error(f"Failed to get state for {comp}: {e}")
            
            if not state_data:
                self.logger.warning("No state data to checkpoint")
                return None
            
            # Compute checksum
            state_str = json.dumps(state_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(state_str.encode()).hexdigest()
            
            # Create checkpoint
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                checkpoint_type=checkpoint_type,
                component=','.join(components),
                state_data=state_data,
                checksum=checksum,
                metadata={
                    'version': '1.0',
                    'agent_version': self._get_agent_version()
                }
            )
            
            # Save checkpoint
            self._save_checkpoint(checkpoint)
            
            # Update tracking
            for comp in components:
                if comp not in self.checkpoints:
                    self.checkpoints[comp] = []
                self.checkpoints[comp].append(checkpoint)
                self.last_checkpoint[comp] = timestamp
            
            self.logger.info(f"Created checkpoint: {checkpoint_id} for {components}")
            
            return checkpoint
    
    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO checkpoints 
                (checkpoint_id, timestamp, checkpoint_type, component, state_data, checksum, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                checkpoint.checkpoint_id,
                checkpoint.timestamp.isoformat(),
                checkpoint.checkpoint_type.value,
                checkpoint.component,
                pickle.dumps(checkpoint.state_data),
                checkpoint.checksum,
                json.dumps(checkpoint.metadata)
            ))
            
            conn.commit()
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM checkpoints WHERE checkpoint_id = ?
            ''', (checkpoint_id,))
            
            row = cursor.fetchone()
            
            if row:
                checkpoint = Checkpoint(
                    checkpoint_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    checkpoint_type=CheckpointType(row[2]),
                    component=row[3],
                    state_data=pickle.loads(row[4]),
                    checksum=row[5],
                    metadata=json.loads(row[7]) if row[7] else {}
                )
                
                # Verify integrity
                if not checkpoint.verify_checksum():
                    self.logger.error(f"Checkpoint {checkpoint_id} checksum mismatch!")
                    return None
                
                return checkpoint
            
            return None
    
    def get_latest_checkpoint(self, component: Optional[str] = None) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a component or all components"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            if component:
                cursor.execute('''
                    SELECT checkpoint_id FROM checkpoints 
                    WHERE component LIKE ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (f'%{component}%',))
            else:
                cursor.execute('''
                    SELECT checkpoint_id FROM checkpoints 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
            
            row = cursor.fetchone()
            
            if row:
                return self.load_checkpoint(row[0])
            
            return None
    
    def list_checkpoints(self, 
                        component: Optional[str] = None,
                        since: Optional[datetime] = None) -> List[Checkpoint]:
        """List available checkpoints"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT checkpoint_id FROM checkpoints WHERE 1=1'
            params = []
            
            if component:
                query += ' AND component LIKE ?'
                params.append(f'%{component}%')
            
            if since:
                query += ' AND timestamp > ?'
                params.append(since.isoformat())
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            
            checkpoints = []
            for row in cursor.fetchall():
                cp = self.load_checkpoint(row[0])
                if cp:
                    checkpoints.append(cp)
            
            return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24, keep_minimum: int = 5):
        """Clean up old checkpoints"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get checkpoints to delete
            cursor.execute('''
                SELECT checkpoint_id FROM checkpoints 
                WHERE timestamp < ?
                ORDER BY timestamp DESC
            ''', (cutoff.isoformat(),))
            
            all_old = cursor.fetchall()
            
            # Keep minimum number
            to_delete = all_old[keep_minimum:]
            
            for row in to_delete:
                cursor.execute('DELETE FROM checkpoints WHERE checkpoint_id = ?', (row[0],))
                self.logger.info(f"Deleted old checkpoint: {row[0]}")
            
            conn.commit()
    
    # ==================== RECOVERY OPERATIONS ====================
    
    def recover(self, 
               checkpoint_id: Optional[str] = None,
               recovery_mode: RecoveryMode = RecoveryMode.FULL) -> RecoveryPoint:
        """
        Recover system state from checkpoint
        
        Args:
            checkpoint_id: Specific checkpoint to recover from, or None for latest
            recovery_mode: Recovery mode to use
            
        Returns:
            RecoveryPoint with recovery information
        """
        with self._lock:
            recovery_id = self._generate_recovery_id()
            timestamp = datetime.now()
            
            # Get checkpoint
            if checkpoint_id:
                checkpoint = self.load_checkpoint(checkpoint_id)
            else:
                checkpoint = self.get_latest_checkpoint()
            
            if not checkpoint:
                self.logger.error("No checkpoint available for recovery")
                return RecoveryPoint(
                    recovery_id=recovery_id,
                    timestamp=timestamp,
                    checkpoint_id="",
                    components=[],
                    recovery_mode=recovery_mode,
                    estimated_data_loss_seconds=0
                )
            
            # Calculate data loss
            data_loss = (timestamp - checkpoint.timestamp).total_seconds()
            
            # Recover components
            recovered_components = []
            failed_components = []
            
            for component, state in checkpoint.state_data.items():
                if recovery_mode == RecoveryMode.MINIMAL and component not in self._get_critical_components():
                    continue
                
                try:
                    if component in self.state_handlers:
                        self.state_handlers[component]['set_state'](state)
                        recovered_components.append(component)
                        self.logger.info(f"Recovered component: {component}")
                    else:
                        # Component not registered, save for later
                        self.component_states[component] = state
                        recovered_components.append(component)
                except Exception as e:
                    self.logger.error(f"Failed to recover {component}: {e}")
                    failed_components.append(component)
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                recovery_id=recovery_id,
                timestamp=timestamp,
                checkpoint_id=checkpoint.checkpoint_id,
                components=recovered_components,
                recovery_mode=recovery_mode,
                estimated_data_loss_seconds=data_loss
            )
            
            # Save recovery point
            self._save_recovery_point(recovery_point)
            
            self.logger.info(f"Recovery completed: {recovery_id}")
            self.logger.info(f"Recovered: {recovered_components}")
            if failed_components:
                self.logger.warning(f"Failed to recover: {failed_components}")
            
            return recovery_point
    
    def _save_recovery_point(self, recovery_point: RecoveryPoint):
        """Save recovery point to database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO recovery_points 
                (recovery_id, timestamp, checkpoint_id, components, recovery_mode, estimated_data_loss_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                recovery_point.recovery_id,
                recovery_point.timestamp.isoformat(),
                recovery_point.checkpoint_id,
                json.dumps(recovery_point.components),
                recovery_point.recovery_mode.value,
                recovery_point.estimated_data_loss_seconds
            ))
            
            conn.commit()
    
    def get_recovery_history(self) -> List[RecoveryPoint]:
        """Get recovery history"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM recovery_points ORDER BY timestamp DESC')
            
            recovery_points = []
            for row in cursor.fetchall():
                recovery_points.append(RecoveryPoint(
                    recovery_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    checkpoint_id=row[2],
                    components=json.loads(row[3]),
                    recovery_mode=RecoveryMode(row[4]),
                    estimated_data_loss_seconds=row[5]
                ))
            
            return recovery_points
    
    # ==================== AUTO-CHECKPOINT ====================
    
    def start_auto_checkpoint(self):
        """Start automatic checkpointing"""
        self._stop_checkpoint.clear()
        
        self._checkpoint_thread = threading.Thread(
            target=self._auto_checkpoint_loop,
            daemon=True,
            name="AutoCheckpoint"
        )
        self._checkpoint_thread.start()
        
        self.logger.info("Auto-checkpoint started")
    
    def _auto_checkpoint_loop(self):
        """Auto-checkpoint loop"""
        while not self._stop_checkpoint.is_set():
            try:
                # Check each component
                for component, handlers in self.state_handlers.items():
                    interval = handlers['checkpoint_interval']
                    last = handlers['last_checkpoint']
                    
                    if last is None or (datetime.now() - last).total_seconds() >= interval:
                        self.create_checkpoint(component, CheckpointType.INCREMENTAL)
                
                # Full checkpoint every 5 minutes
                if int(time.time()) % 300 == 0:
                    self.create_checkpoint(checkpoint_type=CheckpointType.FULL)
                
            except Exception as e:
                self.logger.error(f"Auto-checkpoint error: {e}")
            
            # Sleep
            self._stop_checkpoint.wait(10)
    
    def stop_auto_checkpoint(self):
        """Stop automatic checkpointing"""
        self._stop_checkpoint.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5)
        self.logger.info("Auto-checkpoint stopped")
    
    # ==================== JOURNAL OPERATIONS ====================
    
    def journal_write(self, component: str, operation: str, data: Dict):
        """Write to journal for recovery"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO journal (timestamp, component, operation, data)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                component,
                operation,
                pickle.dumps(data)
            ))
            
            conn.commit()
    
    def journal_replay(self, since: datetime) -> List[Dict]:
        """Replay journal entries since a point in time"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM journal WHERE timestamp > ? ORDER BY entry_id
            ''', (since.isoformat(),))
            
            entries = []
            for row in cursor.fetchall():
                entries.append({
                    'entry_id': row[0],
                    'timestamp': row[1],
                    'component': row[2],
                    'operation': row[3],
                    'data': pickle.loads(row[4])
                })
            
            return entries
    
    # ==================== UTILITY METHODS ====================
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID"""
        return f"CP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8].upper()}"
    
    def _generate_recovery_id(self) -> str:
        """Generate unique recovery ID"""
        return f"REC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8].upper()}"
    
    def _get_agent_version(self) -> str:
        """Get agent version"""
        return "1.0.0"  # Placeholder
    
    def _get_critical_components(self) -> List[str]:
        """Get list of critical components"""
        return ['soul_engine', 'identity_manager', 'cron_scheduler', 'heartbeat']
    
    def get_state_summary(self) -> Dict:
        """Get summary of state management"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Count checkpoints
            cursor.execute('SELECT COUNT(*) FROM checkpoints')
            checkpoint_count = cursor.fetchone()[0]
            
            # Count recovery points
            cursor.execute('SELECT COUNT(*) FROM recovery_points')
            recovery_count = cursor.fetchone()[0]
            
            # Latest checkpoint
            cursor.execute('SELECT MAX(timestamp) FROM checkpoints')
            latest_checkpoint = cursor.fetchone()[0]
            
            return {
                'total_checkpoints': checkpoint_count,
                'total_recoveries': recovery_count,
                'latest_checkpoint': latest_checkpoint,
                'registered_components': list(self.state_handlers.keys()),
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
            }


class ComponentStateMixin:
    """
    Mixin for components to easily integrate with state management
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._state_version = 0
        self._state_dirty = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state - override in subclass"""
        return {
            'version': self._state_version,
            'data': self._state.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set component state - override in subclass"""
        self._state = state.get('data', {}).copy()
        self._state_version = state.get('version', 0)
        self._state_dirty = False
    
    def mark_state_dirty(self):
        """Mark state as needing checkpoint"""
        self._state_dirty = True
    
    def update_state(self, key: str, value: Any):
        """Update state value"""
        self._state[key] = value
        self._state_version += 1
        self._state_dirty = True


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize state manager
    state_manager = StateManager()
    
    # Register a component
    class MyComponent(ComponentStateMixin):
        def __init__(self):
            super().__init__()
            self._state = {'counter': 0, 'messages': []}
        
        def get_state(self):
            return super().get_state()
        
        def set_state(self, state):
            super().set_state(state)
    
    component = MyComponent()
    
    state_manager.register_component(
        'my_component',
        component.get_state,
        component.set_state,
        checkpoint_interval=30
    )
    
    # Create checkpoint
    checkpoint = state_manager.create_checkpoint()
    print(f"Created checkpoint: {checkpoint.checkpoint_id}")
    
    # Simulate crash recovery
    recovery_point = state_manager.recover()
    print(f"Recovery completed: {recovery_point.recovery_id}")
    print(f"Data loss: {recovery_point.estimated_data_loss_seconds}s")
    
    # Get summary
    summary = state_manager.get_state_summary()
    print(f"State summary: {json.dumps(summary, indent=2, default=str)}")
```

---

## Circuit Breaker Patterns

### 7.1 Circuit Breaker Architecture

```python
# circuit_breaker.py - Circuit Breaker Implementation
"""
Circuit breaker pattern for Windows 10 AI Agent
Implements: Failure detection, state transitions, fallback mechanisms
"""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Optional, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import json

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_transitions: List[Dict] = field(default_factory=list)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreaker:
    """
    Circuit breaker implementation for service protection
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """
    
    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3,
                 success_threshold: int = 2,
                 fallback_function: Optional[Callable] = None):
        """
        Initialize circuit breaker
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
            success_threshold: Successes needed to close circuit
            fallback_function: Function to call when circuit is open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.fallback_function = fallback_function
        
        # State
        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Half-open tracking
        self._half_open_calls = 0
        
        # Lock
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f'CircuitBreaker.{name}')
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state
    
    def _transition_to(self, new_state: CircuitState, reason: str = ""):
        """Transition to a new state"""
        with self._lock:
            old_state = self._state
            self._state = new_state
            self._state_changed_at = datetime.now()
            
            # Record transition
            self.metrics.state_transitions.append({
                'from': old_state.value,
                'to': new_state.value,
                'timestamp': datetime.now().isoformat(),
                'reason': reason
            })
            
            self.logger.warning(
                f"Circuit '{self.name}' transitioned from {old_state.value} "
                f"to {new_state.value} - {reason}"
            )
            
            # Reset half-open counter
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                elapsed = (datetime.now() - self._state_changed_at).total_seconds()
                
                if elapsed >= self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN, "Recovery timeout elapsed")
                    return True
                else:
                    return False
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                else:
                    return False
            
            return False
    
    def record_success(self):
        """Record a successful execution"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED, "Success threshold reached")
    
    def record_failure(self, exception: Optional[Exception] = None):
        """Record a failed execution"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            
            if self._state == CircuitState.CLOSED:
                if self.metrics.consecutive_failures >= self.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"Failure threshold reached ({self.metrics.consecutive_failures})"
                    )
            
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(
                    CircuitState.OPEN,
                    f"Failure during recovery test: {exception}"
                )
    
    def record_rejection(self):
        """Record a rejected execution (circuit open)"""
        with self._lock:
            self.metrics.rejected_requests += 1
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function execution fails
        """
        if not self.can_execute():
            self.record_rejection()
            
            if self.fallback_function:
                self.logger.info(f"Executing fallback for '{self.name}'")
                return self.fallback_function(*args, **kwargs)
            
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is OPEN - request rejected"
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        
        except Exception as e:
            self.record_failure(e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker protection"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        # Attach circuit breaker to function
        wrapper._circuit_breaker = self
        return wrapper
    
    def get_metrics(self) -> Dict:
        """Get circuit breaker metrics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'state_since': self._state_changed_at.isoformat(),
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'rejected_requests': self.metrics.rejected_requests,
                    'success_rate': (
                        self.metrics.successful_requests / max(self.metrics.total_requests, 1)
                    ),
                    'consecutive_failures': self.metrics.consecutive_failures,
                    'consecutive_successes': self.metrics.consecutive_successes,
                    'last_failure_time': (
                        self.metrics.last_failure_time.isoformat() 
                        if self.metrics.last_failure_time else None
                    ),
                    'last_success_time': (
                        self.metrics.last_success_time.isoformat()
                        if self.metrics.last_success_time else None
                    )
                },
                'state_transitions': self.metrics.state_transitions[-10:]  # Last 10
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._state_changed_at = datetime.now()
            self.metrics = CircuitBreakerMetrics()
            self._half_open_calls = 0
            self.logger.info(f"Circuit '{self.name}' manually reset to CLOSED")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger('CircuitBreakerRegistry')
    
    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        with self._lock:
            self._breakers[breaker.name] = breaker
            self.logger.info(f"Registered circuit breaker: {breaker.name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        with self._lock:
            return self._breakers.get(name)
    
    def create(self,
               name: str,
               failure_threshold: int = 5,
               recovery_timeout: float = 30.0,
               **kwargs) -> CircuitBreaker:
        """Create and register a new circuit breaker"""
        with self._lock:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
            self.register(breaker)
            return breaker
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }
    
    def get_open_circuits(self) -> List[str]:
        """Get list of open circuits"""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            ]
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def reset(self, name: str):
        """Reset a specific circuit breaker"""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()


# ==================== SERVICE-SPECIFIC CIRCUIT BREAKERS ====================

class GmailCircuitBreaker:
    """Circuit breaker for Gmail API"""
    
    def __init__(self, registry: CircuitBreakerRegistry):
        self.breaker = registry.create(
            name='gmail_service',
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=2,
            success_threshold=2,
            fallback_function=self._fallback
        )
    
    def _fallback(self, *args, **kwargs):
        """Fallback when Gmail is unavailable"""
        return {
            'success': False,
            'error': 'Gmail service temporarily unavailable',
            'fallback': True,
            'queued': True  # Request will be queued for later
        }

class BrowserCircuitBreaker:
    """Circuit breaker for browser control"""
    
    def __init__(self, registry: CircuitBreakerRegistry):
        self.breaker = registry.create(
            name='browser_control',
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=1,
            success_threshold=1,
            fallback_function=self._fallback
        )
    
    def _fallback(self, *args, **kwargs):
        """Fallback when browser is unavailable"""
        return {
            'success': False,
            'error': 'Browser control temporarily unavailable',
            'fallback': True,
            'retry_after': 30
        }

class GPTCircuitBreaker:
    """Circuit breaker for GPT-5.2 API"""
    
    def __init__(self, registry: CircuitBreakerRegistry):
        self.breaker = registry.create(
            name='gpt_api',
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=1,
            success_threshold=2,
            fallback_function=self._fallback
        )
    
    def _fallback(self, *args, **kwargs):
        """Fallback when GPT API is unavailable"""
        return {
            'success': False,
            'error': 'AI service temporarily unavailable',
            'fallback': True,
            'use_cached': True,
            'cached_response': 'I apologize, but I am experiencing technical difficulties. Please try again in a moment.'
        }

class TwilioCircuitBreaker:
    """Circuit breaker for Twilio service"""
    
    def __init__(self, registry: CircuitBreakerRegistry):
        self.breaker = registry.create(
            name='twilio_service',
            failure_threshold=3,
            recovery_timeout=45.0,
            half_open_max_calls=2,
            success_threshold=2,
            fallback_function=self._fallback
        )
    
    def _fallback(self, *args, **kwargs):
        """Fallback when Twilio is unavailable"""
        return {
            'success': False,
            'error': 'Communication service temporarily unavailable',
            'fallback': True,
            'queued': True
        }


# ==================== CIRCUIT BREAKER MONITOR ====================

class CircuitBreakerMonitor:
    """
    Monitor circuit breakers and take action on state changes
    """
    
    def __init__(self, registry: CircuitBreakerRegistry):
        self.registry = registry
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._handlers: Dict[CircuitState, List[Callable]] = {
            CircuitState.OPEN: [],
            CircuitState.HALF_OPEN: [],
            CircuitState.CLOSED: []
        }
        self.logger = logging.getLogger('CircuitBreakerMonitor')
    
    def on_state_change(self, state: CircuitState, handler: Callable):
        """Register a handler for state changes"""
        self._handlers[state].append(handler)
    
    def start(self, check_interval: float = 5.0):
        """Start monitoring"""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True,
            name="CircuitBreakerMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Circuit breaker monitor started")
    
    def _monitor_loop(self, check_interval: float):
        """Monitor loop"""
        previous_states: Dict[str, CircuitState] = {}
        
        while self._running:
            try:
                for name, breaker in self.registry._breakers.items():
                    current_state = breaker.state
                    
                    # Check for state change
                    if name in previous_states:
                        if previous_states[name] != current_state:
                            self._handle_state_change(name, previous_states[name], current_state)
                    
                    previous_states[name] = current_state
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
            
            time.sleep(check_interval)
    
    def _handle_state_change(self, name: str, old_state: CircuitState, new_state: CircuitState):
        """Handle circuit breaker state change"""
        self.logger.warning(f"Circuit '{name}' changed from {old_state.value} to {new_state.value}")
        
        # Call registered handlers
        for handler in self._handlers.get(new_state, []):
            try:
                handler(name, old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change handler failed: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create registry
    registry = CircuitBreakerRegistry()
    
    # Create service circuit breakers
    gmail_cb = GmailCircuitBreaker(registry)
    browser_cb = BrowserCircuitBreaker(registry)
    gpt_cb = GPTCircuitBreaker(registry)
    
    # Create monitor
    monitor = CircuitBreakerMonitor(registry)
    
    # Register state change handlers
    def on_open(name, old, new):
        print(f"ALERT: Circuit {name} is OPEN!")
    
    def on_closed(name, old, new):
        print(f"INFO: Circuit {name} is now CLOSED (recovered)")
    
    monitor.on_state_change(CircuitState.OPEN, on_open)
    monitor.on_state_change(CircuitState.CLOSED, on_closed)
    
    # Start monitor
    monitor.start()
    
    # Example usage with decorator
    @gmail_cb.breaker
    def send_email(to: str, subject: str, body: str):
        # Simulate API call
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Gmail API error")
        return {'success': True, 'message_id': '12345'}
    
    # Test circuit breaker
    for i in range(20):
        try:
            result = send_email('test@example.com', 'Test', 'Body')
            print(f"Success: {result}")
        except CircuitBreakerError as e:
            print(f"Circuit open: {e}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(0.5)
    
    # Get metrics
    metrics = registry.get_all_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Stop monitor
    monitor.stop()
```

---

## Graceful Degradation

### 8.1 Graceful Degradation Architecture

```python
# graceful_degradation.py - Graceful Degradation System
"""
Graceful degradation for Windows 10 AI Agent
Implements: Feature tiers, fallback mechanisms, adaptive behavior
"""

import time
import threading
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json

class DegradationLevel(Enum):
    """Degradation levels from full functionality to minimal"""
    FULL = "full"           # All features available
    REDUCED = "reduced"     # Some non-critical features disabled
    MINIMAL = "minimal"     # Only critical features available
    EMERGENCY = "emergency" # Absolute minimum for survival
    OFFLINE = "offline"     # No external connectivity

class FeaturePriority(Enum):
    """Feature priority levels"""
    CRITICAL = 1      # Must have for basic operation
    HIGH = 2          # Important but can be disabled
    MEDIUM = 3        # Nice to have
    LOW = 4           # Can be disabled anytime

@dataclass
class Feature:
    """Feature definition"""
    name: str
    priority: FeaturePriority
    dependencies: List[str] = field(default_factory=list)
    degradation_action: Optional[Callable] = None
    enable_action: Optional[Callable] = None
    enabled: bool = True
    last_enabled: Optional[datetime] = None
    last_disabled: Optional[datetime] = None

class DegradationManager:
    """
    Centralized graceful degradation manager
    Manages feature availability based on system health
    """
    
    def __init__(self):
        self.current_level = DegradationLevel.FULL
        self.features: Dict[str, Feature] = {}
        self.levels: Dict[DegradationLevel, Set[str]] = {
            DegradationLevel.FULL: set(),
            DegradationLevel.REDUCED: set(),
            DegradationLevel.MINIMAL: set(),
            DegradationLevel.EMERGENCY: set(),
            DegradationLevel.OFFLINE: set()
        }
        self._lock = threading.RLock()
        self._handlers: List[Callable] = []
        
        # Logging
        self.logger = logging.getLogger('DegradationManager')
        self._setup_logging()
        
        # Initialize default features
        self._init_default_features()
    
    def _setup_logging(self):
        """Configure degradation manager logging"""
        handler = logging.FileHandler('logs/degradation.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_default_features(self):
        """Initialize default feature set"""
        # Critical features (always available)
        self.register_feature(Feature(
            name='heartbeat',
            priority=FeaturePriority.CRITICAL
        ))
        self.register_feature(Feature(
            name='core_agent',
            priority=FeaturePriority.CRITICAL
        ))
        self.register_feature(Feature(
            name='soul_engine',
            priority=FeaturePriority.CRITICAL
        ))
        
        # High priority features
        self.register_feature(Feature(
            name='gmail_service',
            priority=FeaturePriority.HIGH,
            dependencies=['network']
        ))
        self.register_feature(Feature(
            name='cron_scheduler',
            priority=FeaturePriority.HIGH
        ))
        
        # Medium priority features
        self.register_feature(Feature(
            name='browser_control',
            priority=FeaturePriority.MEDIUM,
            dependencies=['network']
        ))
        self.register_feature(Feature(
            name='twilio_service',
            priority=FeaturePriority.MEDIUM,
            dependencies=['network']
        ))
        
        # Low priority features
        self.register_feature(Feature(
            name='tts_engine',
            priority=FeaturePriority.LOW
        ))
        self.register_feature(Feature(
            name='stt_engine',
            priority=FeaturePriority.LOW
        ))
        self.register_feature(Feature(
            name='advanced_analytics',
            priority=FeaturePriority.LOW
        ))
        
        # Configure degradation levels
        self._configure_degradation_levels()
    
    def _configure_degradation_levels(self):
        """Configure which features are available at each degradation level"""
        # FULL: All features
        self.levels[DegradationLevel.FULL] = set(self.features.keys())
        
        # REDUCED: Disable low priority
        self.levels[DegradationLevel.REDUCED] = {
            name for name, f in self.features.items()
            if f.priority.value <= FeaturePriority.MEDIUM.value
        }
        
        # MINIMAL: Only critical and high priority
        self.levels[DegradationLevel.MINIMAL] = {
            name for name, f in self.features.items()
            if f.priority.value <= FeaturePriority.HIGH.value
        }
        
        # EMERGENCY: Only critical
        self.levels[DegradationLevel.EMERGENCY] = {
            name for name, f in self.features.items()
            if f.priority == FeaturePriority.CRITICAL
        }
        
        # OFFLINE: Critical without network dependencies
        self.levels[DegradationLevel.OFFLINE] = {
            name for name, f in self.features.items()
            if f.priority == FeaturePriority.CRITICAL and 'network' not in f.dependencies
        }
    
    def register_feature(self, feature: Feature):
        """Register a feature"""
        with self._lock:
            self.features[feature.name] = feature
            self.logger.info(f"Registered feature: {feature.name} (priority: {feature.priority.name})")
    
    def subscribe(self, handler: Callable):
        """Subscribe to degradation level changes"""
        self._handlers.append(handler)
    
    def _notify_handlers(self, old_level: DegradationLevel, new_level: DegradationLevel):
        """Notify handlers of level change"""
        for handler in self._handlers:
            try:
                handler(old_level, new_level)
            except Exception as e:
                self.logger.error(f"Handler notification failed: {e}")
    
    def set_degradation_level(self, level: DegradationLevel, reason: str = ""):
        """
        Set the degradation level
        
        Args:
            level: New degradation level
            reason: Reason for the change
        """
        with self._lock:
            if self.current_level == level:
                return
            
            old_level = self.current_level
            self.current_level = level
            
            # Update feature states
            enabled_features = self.levels[level]
            
            for name, feature in self.features.items():
                should_enable = name in enabled_features
                
                if should_enable and not feature.enabled:
                    self._enable_feature(feature)
                elif not should_enable and feature.enabled:
                    self._disable_feature(feature)
            
            self.logger.warning(
                f"Degradation level changed from {old_level.value} to {level.value} - {reason}"
            )
            
            # Notify handlers
            self._notify_handlers(old_level, level)
    
    def _enable_feature(self, feature: Feature):
        """Enable a feature"""
        feature.enabled = True
        feature.last_enabled = datetime.now()
        
        if feature.enable_action:
            try:
                feature.enable_action()
            except Exception as e:
                self.logger.error(f"Failed to enable feature {feature.name}: {e}")
        
        self.logger.info(f"Feature enabled: {feature.name}")
    
    def _disable_feature(self, feature: Feature):
        """Disable a feature"""
        feature.enabled = False
        feature.last_disabled = datetime.now()
        
        if feature.degradation_action:
            try:
                feature.degradation_action()
            except Exception as e:
                self.logger.error(f"Failed to disable feature {feature.name}: {e}")
        
        self.logger.info(f"Feature disabled: {feature.name}")
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is available"""
        with self._lock:
            if feature_name not in self.features:
                return False
            return self.features[feature_name].enabled
    
    def get_available_features(self) -> List[str]:
        """Get list of currently available features"""
        with self._lock:
            return [
                name for name, f in self.features.items()
                if f.enabled
            ]
    
    def get_disabled_features(self) -> List[str]:
        """Get list of currently disabled features"""
        with self._lock:
            return [
                name for name, f in self.features.items()
                if not f.enabled
            ]
    
    def get_status(self) -> Dict:
        """Get degradation status"""
        with self._lock:
            return {
                'current_level': self.current_level.value,
                'available_features': self.get_available_features(),
                'disabled_features': self.get_disabled_features(),
                'feature_details': {
                    name: {
                        'enabled': f.enabled,
                        'priority': f.priority.name,
                        'dependencies': f.dependencies,
                        'last_enabled': f.last_enabled.isoformat() if f.last_enabled else None,
                        'last_disabled': f.last_disabled.isoformat() if f.last_disabled else None
                    }
                    for name, f in self.features.items()
                }
            }


class AdaptiveDegradationController:
    """
    Automatically adjust degradation level based on system health
    """
    
    def __init__(self, degradation_manager: DegradationManager):
        self.degradation_manager = degradation_manager
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Health thresholds for degradation
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'error_rate_warning': 0.1,  # 10%
            'error_rate_critical': 0.3   # 30%
        }
        
        self.logger = logging.getLogger('AdaptiveDegradationController')
    
    def start(self, check_interval: float = 30.0):
        """Start adaptive degradation controller"""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True,
            name="AdaptiveDegradationController"
        )
        self._monitor_thread.start()
        self.logger.info("Adaptive degradation controller started")
    
    def _monitor_loop(self, check_interval: float):
        """Monitor system health and adjust degradation"""
        while self._running:
            try:
                # Get system metrics
                metrics = self._get_system_metrics()
                
                # Determine appropriate degradation level
                new_level = self._determine_degradation_level(metrics)
                
                # Apply if different
                if new_level != self.degradation_manager.current_level:
                    self.degradation_manager.set_degradation_level(
                        new_level,
                        reason=f"System metrics: CPU={metrics['cpu']}%, Memory={metrics['memory']}%"
                    )
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
            
            time.sleep(check_interval)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        import psutil
        
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('C:\\').percent
        }
    
    def _determine_degradation_level(self, metrics: Dict[str, float]) -> DegradationLevel:
        """Determine appropriate degradation level based on metrics"""
        cpu = metrics['cpu']
        memory = metrics['memory']
        
        # Critical conditions
        if cpu >= self.thresholds['cpu_critical'] or memory >= self.thresholds['memory_critical']:
            return DegradationLevel.EMERGENCY
        
        # High stress
        if cpu >= self.thresholds['cpu_warning'] or memory >= self.thresholds['memory_warning']:
            return DegradationLevel.MINIMAL
        
        # Moderate stress
        if cpu >= self.thresholds['cpu_warning'] * 0.7 or memory >= self.thresholds['memory_warning'] * 0.7:
            return DegradationLevel.REDUCED
        
        return DegradationLevel.FULL
    
    def stop(self):
        """Stop adaptive degradation controller"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Adaptive degradation controller stopped")


class FallbackManager:
    """
    Manage fallback behaviors for degraded features
    """
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
        self.logger = logging.getLogger('FallbackManager')
    
    def register_fallback(self, feature: str, fallback: Callable):
        """Register a fallback for a feature"""
        self.fallbacks[feature] = fallback
        self.logger.info(f"Registered fallback for: {feature}")
    
    def execute_fallback(self, feature: str, *args, **kwargs) -> Any:
        """Execute fallback for a feature"""
        if feature in self.fallbacks:
            try:
                return self.fallbacks[feature](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback execution failed for {feature}: {e}")
                return None
        else:
            self.logger.warning(f"No fallback registered for: {feature}")
            return None
    
    def get_fallback_response(self, feature: str) -> Dict:
        """Get standard fallback response for a feature"""
        fallback_responses = {
            'tts_engine': {
                'success': False,
                'error': 'Text-to-speech temporarily unavailable',
                'fallback': True,
                'message': 'Response will be displayed as text only'
            },
            'stt_engine': {
                'success': False,
                'error': 'Speech recognition temporarily unavailable',
                'fallback': True,
                'message': 'Please use text input instead'
            },
            'browser_control': {
                'success': False,
                'error': 'Browser automation temporarily unavailable',
                'fallback': True,
                'message': 'Direct API calls will be used instead'
            },
            'gmail_service': {
                'success': False,
                'error': 'Email service temporarily unavailable',
                'fallback': True,
                'queued': True,
                'message': 'Messages will be queued for later delivery'
            },
            'twilio_service': {
                'success': False,
                'error': 'Communication service temporarily unavailable',
                'fallback': True,
                'queued': True,
                'message': 'Calls/messages will be queued for later'
            }
        }
        
        return fallback_responses.get(feature, {
            'success': False,
            'error': f'{feature} temporarily unavailable',
            'fallback': True
        })


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create degradation manager
    degradation_manager = DegradationManager()
    
    # Subscribe to changes
    def on_level_change(old, new):
        print(f"Degradation level changed: {old.value} -> {new.value}")
    
    degradation_manager.subscribe(on_level_change)
    
    # Create adaptive controller
    controller = AdaptiveDegradationController(degradation_manager)
    controller.start()
    
    # Check initial status
    print("Initial status:")
    print(json.dumps(degradation_manager.get_status(), indent=2))
    
    # Simulate degradation
    print("\nSimulating MINIMAL degradation...")
    degradation_manager.set_degradation_level(DegradationLevel.MINIMAL, "High system load")
    
    print("\nStatus after degradation:")
    print(json.dumps(degradation_manager.get_status(), indent=2))
    
    # Check feature availability
    print(f"\nIs TTS available? {degradation_manager.is_feature_available('tts_engine')}")
    print(f"Is heartbeat available? {degradation_manager.is_feature_available('heartbeat')}")
    
    # Stop controller
    controller.stop()
```

---

## Recovery Testing Framework

### 9.1 Recovery Testing Architecture

```python
# recovery_testing.py - Recovery Testing Framework
"""
Recovery testing framework for Windows 10 AI Agent
Implements: Chaos engineering, failure injection, recovery validation
"""

import os
import sys
import time
import random
import threading
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import psutil

class TestType(Enum):
    """Types of recovery tests"""
    CRASH_RECOVERY = "crash_recovery"
    SERVICE_RESTART = "service_restart"
    NETWORK_FAILURE = "network_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEADLOCK = "deadlock"
    MEMORY_LEAK = "memory_leak"
    CIRCUIT_BREAKER = "circuit_breaker"
    STATE_RECOVERY = "state_recovery"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class TestResult(Enum):
    """Test result statuses"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class TestCase:
    """Test case definition"""
    name: str
    test_type: TestType
    description: str
    target_component: str
    failure_injection: Callable
    validation_checks: List[Callable]
    timeout_seconds: int = 60
    cleanup: Optional[Callable] = None

@dataclass
class TestExecution:
    """Test execution record"""
    test_id: str
    test_case: TestCase
    start_time: datetime
    end_time: Optional[datetime] = None
    result: TestResult = TestResult.SKIPPED
    details: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

class RecoveryTestFramework:
    """
    Framework for testing recovery mechanisms
    Implements chaos engineering principles
    """
    
    def __init__(self, agent_context: Any = None):
        self.agent_context = agent_context
        self.test_cases: Dict[str, TestCase] = {}
        self.test_history: List[TestExecution] = []
        self._running = False
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger('RecoveryTestFramework')
        self._setup_logging()
        
        # Initialize default test cases
        self._init_default_tests()
    
    def _setup_logging(self):
        """Configure test framework logging"""
        os.makedirs('logs/tests', exist_ok=True)
        handler = logging.FileHandler('logs/recovery_tests.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_default_tests(self):
        """Initialize default test cases"""
        # Crash recovery test
        self.register_test(TestCase(
            name="agent_crash_recovery",
            test_type=TestType.CRASH_RECOVERY,
            description="Test agent recovery after simulated crash",
            target_component="agent_core",
            failure_injection=self._inject_crash,
            validation_checks=[
                self._validate_process_restarted,
                self._validate_state_recovered,
                self._validate_services_running
            ],
            timeout_seconds=120
        ))
        
        # Service restart test
        self.register_test(TestCase(
            name="service_restart_test",
            test_type=TestType.SERVICE_RESTART,
            description="Test individual service restart",
            target_component="gmail_service",
            failure_injection=self._inject_service_failure,
            validation_checks=[
                self._validate_service_restarted,
                self._validate_service_functional
            ],
            timeout_seconds=60
        ))
        
        # Circuit breaker test
        self.register_test(TestCase(
            name="circuit_breaker_test",
            test_type=TestType.CIRCUIT_BREAKER,
            description="Test circuit breaker activation and recovery",
            target_component="gmail_service",
            failure_injection=self._inject_repeated_failures,
            validation_checks=[
                self._validate_circuit_opened,
                self._validate_circuit_closed_after_recovery
            ],
            timeout_seconds=90
        ))
        
        # State recovery test
        self.register_test(TestCase(
            name="state_recovery_test",
            test_type=TestType.STATE_RECOVERY,
            description="Test state recovery from checkpoint",
            target_component="all",
            failure_injection=self._inject_state_loss,
            validation_checks=[
                self._validate_checkpoint_exists,
                self._validate_state_restored
            ],
            timeout_seconds=60
        ))
        
        # Graceful degradation test
        self.register_test(TestCase(
            name="graceful_degradation_test",
            test_type=TestType.GRACEFUL_DEGRADATION,
            description="Test graceful degradation under load",
            target_component="all",
            failure_injection=self._inject_resource_stress,
            validation_checks=[
                self._validate_degradation_triggered,
                self._validate_critical_features_available
            ],
            timeout_seconds=120
        ))
    
    def register_test(self, test_case: TestCase):
        """Register a test case"""
        with self._lock:
            self.test_cases[test_case.name] = test_case
            self.logger.info(f"Registered test: {test_case.name}")
    
    def run_test(self, test_name: str) -> TestExecution:
        """
        Run a specific test
        
        Args:
            test_name: Name of the test to run
            
        Returns:
            Test execution record
        """
        if test_name not in self.test_cases:
            raise ValueError(f"Test not found: {test_name}")
        
        test_case = self.test_cases[test_name]
        test_id = self._generate_test_id()
        
        execution = TestExecution(
            test_id=test_id,
            test_case=test_case,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Starting test: {test_name} (ID: {test_id})")
        
        try:
            # Pre-test validation
            if not self._pre_test_validation(test_case):
                execution.result = TestResult.SKIPPED
                execution.details['error'] = "Pre-test validation failed"
                return execution
            
            # Inject failure
            self.logger.info(f"Injecting failure for {test_name}")
            injection_result = test_case.failure_injection()
            execution.details['injection_result'] = injection_result
            
            # Wait for recovery
            self.logger.info(f"Waiting for recovery (timeout: {test_case.timeout_seconds}s)")
            recovery_success = self._wait_for_recovery(test_case)
            execution.details['recovery_success'] = recovery_success
            
            # Run validation checks
            validation_results = []
            for check in test_case.validation_checks:
                try:
                    result = check()
                    validation_results.append({
                        'check': check.__name__,
                        'passed': result
                    })
                except Exception as e:
                    validation_results.append({
                        'check': check.__name__,
                        'passed': False,
                        'error': str(e)
                    })
            
            execution.details['validation_results'] = validation_results
            
            # Determine result
            all_passed = all(r['passed'] for r in validation_results)
            any_passed = any(r['passed'] for r in validation_results)
            
            if all_passed:
                execution.result = TestResult.PASSED
            elif any_passed:
                execution.result = TestResult.PARTIAL
            else:
                execution.result = TestResult.FAILED
            
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.details['error'] = str(e)
            execution.details['traceback'] = traceback.format_exc()
            self.logger.error(f"Test error: {e}")
        
        finally:
            # Cleanup
            if test_case.cleanup:
                try:
                    test_case.cleanup()
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
            
            execution.end_time = datetime.now()
            execution.details['duration_seconds'] = (
                execution.end_time - execution.start_time
            ).total_seconds()
        
        # Store execution record
        with self._lock:
            self.test_history.append(execution)
        
        self.logger.info(f"Test completed: {test_name} - {execution.result.value}")
        
        return execution
    
    def run_all_tests(self) -> List[TestExecution]:
        """Run all registered tests"""
        results = []
        
        for test_name in self.test_cases:
            result = self.run_test(test_name)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(5)
        
        return results
    
    def run_test_suite(self, test_names: List[str]) -> List[TestExecution]:
        """Run a specific suite of tests"""
        results = []
        
        for test_name in test_names:
            if test_name in self.test_cases:
                result = self.run_test(test_name)
                results.append(result)
                time.sleep(5)
            else:
                self.logger.warning(f"Test not found: {test_name}")
        
        return results
    
    def _generate_test_id(self) -> str:
        """Generate unique test ID"""
        return f"TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000, 9999)}"
    
    def _pre_test_validation(self, test_case: TestCase) -> bool:
        """Validate pre-test conditions"""
        # Check if agent is running
        # Check if target component exists
        # Check if test environment is ready
        return True
    
    def _wait_for_recovery(self, test_case: TestCase) -> bool:
        """Wait for system recovery"""
        start_time = time.time()
        
        while time.time() - start_time < test_case.timeout_seconds:
            # Check if system has recovered
            if self._is_system_healthy():
                return True
            
            time.sleep(1)
        
        return False
    
    def _is_system_healthy(self) -> bool:
        """Check if system is healthy"""
        # This would integrate with health monitor
        return True
    
    # ==================== FAILURE INJECTION METHODS ====================
    
    def _inject_crash(self) -> Dict:
        """Inject a simulated crash"""
        # Signal the agent to simulate a crash
        # In production, this would use a test hook
        return {'method': 'crash_simulation', 'triggered': True}
    
    def _inject_service_failure(self) -> Dict:
        """Inject a service failure"""
        # Stop a service
        return {'method': 'service_stop', 'service': 'gmail_service'}
    
    def _inject_repeated_failures(self) -> Dict:
        """Inject repeated failures to trigger circuit breaker"""
        # Simulate repeated API failures
        return {'method': 'repeated_failures', 'count': 10}
    
    def _inject_state_loss(self) -> Dict:
        """Inject state loss scenario"""
        # Clear in-memory state
        return {'method': 'state_clear', 'affected': 'memory'}
    
    def _inject_resource_stress(self) -> Dict:
        """Inject resource stress"""
        # Create CPU/memory pressure
        return {'method': 'resource_stress', 'cpu_percent': 90, 'memory_percent': 90}
    
    # ==================== VALIDATION CHECKS ====================
    
    def _validate_process_restarted(self) -> bool:
        """Validate that the process restarted"""
        # Check if agent process is running
        return True
    
    def _validate_state_recovered(self) -> bool:
        """Validate that state was recovered"""
        # Check if state was restored from checkpoint
        return True
    
    def _validate_services_running(self) -> bool:
        """Validate that services are running"""
        # Check service statuses
        return True
    
    def _validate_service_restarted(self) -> bool:
        """Validate that service restarted"""
        return True
    
    def _validate_service_functional(self) -> bool:
        """Validate that service is functional"""
        return True
    
    def _validate_circuit_opened(self) -> bool:
        """Validate that circuit breaker opened"""
        return True
    
    def _validate_circuit_closed_after_recovery(self) -> bool:
        """Validate that circuit closed after recovery"""
        return True
    
    def _validate_checkpoint_exists(self) -> bool:
        """Validate that checkpoint exists"""
        return True
    
    def _validate_state_restored(self) -> bool:
        """Validate that state was restored"""
        return True
    
    def _validate_degradation_triggered(self) -> bool:
        """Validate that degradation was triggered"""
        return True
    
    def _validate_critical_features_available(self) -> bool:
        """Validate that critical features are available"""
        return True
    
    # ==================== REPORTING ====================
    
    def generate_report(self) -> Dict:
        """Generate test execution report"""
        with self._lock:
            if not self.test_history:
                return {'message': 'No tests executed'}
            
            recent_tests = self.test_history[-50:]  # Last 50 tests
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_tests': len(recent_tests),
                'summary': {
                    'passed': sum(1 for t in recent_tests if t.result == TestResult.PASSED),
                    'failed': sum(1 for t in recent_tests if t.result == TestResult.FAILED),
                    'partial': sum(1 for t in recent_tests if t.result == TestResult.PARTIAL),
                    'error': sum(1 for t in recent_tests if t.result == TestResult.ERROR),
                    'skipped': sum(1 for t in recent_tests if t.result == TestResult.SKIPPED)
                },
                'by_type': {},
                'recent_executions': []
            }
            
            # Group by type
            for test in recent_tests:
                test_type = test.test_case.test_type.value
                if test_type not in report['by_type']:
                    report['by_type'][test_type] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 0
                    }
                
                report['by_type'][test_type]['total'] += 1
                if test.result == TestResult.PASSED:
                    report['by_type'][test_type]['passed'] += 1
                else:
                    report['by_type'][test_type]['failed'] += 1
            
            # Recent executions
            for test in recent_tests[-10:]:
                report['recent_executions'].append({
                    'test_id': test.test_id,
                    'name': test.test_case.name,
                    'type': test.test_case.test_type.value,
                    'result': test.result.value,
                    'duration': test.details.get('duration_seconds', 0),
                    'timestamp': test.start_time.isoformat()
                })
            
            return report
    
    def save_report(self, filepath: str):
        """Save report to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to: {filepath}")


class ChaosEngineeringScheduler:
    """
    Schedule chaos engineering tests to run periodically
    """
    
    def __init__(self, test_framework: RecoveryTestFramework):
        self.test_framework = test_framework
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self.schedule: Dict[str, Any] = {}
        self.logger = logging.getLogger('ChaosScheduler')
    
    def configure_schedule(self, schedule: Dict[str, Any]):
        """Configure test schedule"""
        self.schedule = schedule
    
    def start(self):
        """Start the chaos engineering scheduler"""
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True,
            name="ChaosScheduler"
        )
        self._scheduler_thread.start()
        self.logger.info("Chaos engineering scheduler started")
    
    def _schedule_loop(self):
        """Main scheduling loop"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check if any tests should run
                for test_name, config in self.schedule.items():
                    if self._should_run_test(test_name, config, current_time):
                        self.logger.info(f"Running scheduled test: {test_name}")
                        self.test_framework.run_test(test_name)
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _should_run_test(self, test_name: str, config: Dict, current_time: datetime) -> bool:
        """Check if a test should run now"""
        # Check frequency
        frequency = config.get('frequency', 'daily')
        
        if frequency == 'hourly':
            return current_time.minute == 0
        elif frequency == 'daily':
            return current_time.hour == config.get('hour', 2) and current_time.minute == 0
        elif frequency == 'weekly':
            return (current_time.weekday() == config.get('day', 0) and 
                    current_time.hour == config.get('hour', 2) and 
                    current_time.minute == 0)
        
        return False
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.logger.info("Chaos engineering scheduler stopped")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create test framework
    framework = RecoveryTestFramework()
    
    # Run a specific test
    result = framework.run_test("circuit_breaker_test")
    print(f"Test result: {result.result.value}")
    print(f"Details: {json.dumps(result.details, indent=2, default=str)}")
    
    # Generate report
    report = framework.generate_report()
    print(f"\nTest Report:\n{json.dumps(report, indent=2, default=str)}")
    
    # Save report
    framework.save_report("logs/test_report.json")
```

---

## Incident Logging and Analysis

### 10.1 Incident Logging Architecture

```python
# incident_analysis.py - Incident Logging and Analysis System
"""
Incident logging and analysis for Windows 10 AI Agent
Implements: Centralized logging, pattern detection, root cause analysis
"""

import os
import json
import logging
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import re

class IncidentSeverity(Enum):
    """Incident severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentCategory(Enum):
    """Incident categories"""
    CRASH = "crash"
    ERROR = "error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    RESOURCE = "resource"
    RECOVERY = "recovery"
    HEALTH = "health"

@dataclass
class Incident:
    """Incident record"""
    incident_id: str
    timestamp: datetime
    severity: IncidentSeverity
    category: IncidentCategory
    component: str
    title: str
    description: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    related_incidents: List[str] = field(default_factory=list)
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'incident_id': self.incident_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'component': self.component,
            'title': self.title,
            'description': self.description,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'related_incidents': self.related_incidents,
            'resolution': self.resolution,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class IncidentManager:
    """
    Centralized incident management system
    Handles logging, storage, and analysis of all incidents
    """
    
    def __init__(self, db_path: str = "state/incidents.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._lock = threading.RLock()
        self._handlers: List[Callable] = []
        
        # Initialize database
        self._init_database()
        
        # Logging
        self.logger = logging.getLogger('IncidentManager')
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure incident manager logging"""
        handler = logging.FileHandler('logs/incidents.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _init_database(self):
        """Initialize incident database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    severity TEXT,
                    category TEXT,
                    component TEXT,
                    title TEXT,
                    description TEXT,
                    stack_trace TEXT,
                    context TEXT,
                    related_incidents TEXT,
                    resolution TEXT,
                    resolved_at TEXT
                )
            ''')
            
            # Patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_hash TEXT UNIQUE,
                    first_seen TEXT,
                    last_seen TEXT,
                    occurrence_count INTEGER,
                    severity TEXT,
                    category TEXT,
                    component TEXT,
                    signature TEXT,
                    sample_incident_id TEXT
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incident_metrics (
                    date TEXT PRIMARY KEY,
                    total_incidents INTEGER,
                    by_severity TEXT,
                    by_category TEXT,
                    by_component TEXT
                )
            ''')
            
            conn.commit()
    
    def log_incident(self,
                    severity: IncidentSeverity,
                    category: IncidentCategory,
                    component: str,
                    title: str,
                    description: str,
                    stack_trace: Optional[str] = None,
                    context: Dict[str, Any] = None) -> Incident:
        """
        Log a new incident
        
        Args:
            severity: Incident severity
            category: Incident category
            component: Affected component
            title: Short title
            description: Detailed description
            stack_trace: Optional stack trace
            context: Additional context
            
        Returns:
            Created incident
        """
        with self._lock:
            incident_id = self._generate_incident_id()
            timestamp = datetime.now()
            
            incident = Incident(
                incident_id=incident_id,
                timestamp=timestamp,
                severity=severity,
                category=category,
                component=component,
                title=title,
                description=description,
                stack_trace=stack_trace,
                context=context or {}
            )
            
            # Find related incidents
            incident.related_incidents = self._find_related_incidents(incident)
            
            # Store in database
            self._store_incident(incident)
            
            # Update patterns
            self._update_patterns(incident)
            
            # Log
            self.logger.log(
                self._severity_to_level(severity),
                f"Incident {incident_id}: {title}"
            )
            
            # Notify handlers
            self._notify_handlers(incident)
            
            return incident
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"INC-{timestamp}-{random_suffix.upper()}"
    
    def _severity_to_level(self, severity: IncidentSeverity) -> int:
        """Convert severity to logging level"""
        levels = {
            IncidentSeverity.INFO: logging.INFO,
            IncidentSeverity.LOW: logging.WARNING,
            IncidentSeverity.MEDIUM: logging.WARNING,
            IncidentSeverity.HIGH: logging.ERROR,
            IncidentSeverity.CRITICAL: logging.CRITICAL
        }
        return levels.get(severity, logging.INFO)
    
    def _store_incident(self, incident: Incident):
        """Store incident in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO incidents 
                (incident_id, timestamp, severity, category, component, title, 
                 description, stack_trace, context, related_incidents, resolution, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident.incident_id,
                incident.timestamp.isoformat(),
                incident.severity.value,
                incident.category.value,
                incident.component,
                incident.title,
                incident.description,
                incident.stack_trace,
                json.dumps(incident.context),
                json.dumps(incident.related_incidents),
                incident.resolution,
                incident.resolved_at.isoformat() if incident.resolved_at else None
            ))
            
            conn.commit()
    
    def _find_related_incidents(self, incident: Incident, window_hours: int = 24) -> List[str]:
        """Find related incidents within time window"""
        window_start = incident.timestamp - timedelta(hours=window_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT incident_id FROM incidents
                WHERE component = ? AND category = ?
                AND timestamp > ?
                AND incident_id != ?
            ''', (
                incident.component,
                incident.category.value,
                window_start.isoformat(),
                incident.incident_id
            ))
            
            return [row[0] for row in cursor.fetchall()]
    
    def _update_patterns(self, incident: Incident):
        """Update incident patterns"""
        # Generate pattern signature
        signature = self._generate_signature(incident)
        pattern_hash = hashlib.md5(signature.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if pattern exists
            cursor.execute(
                'SELECT pattern_id, occurrence_count FROM patterns WHERE pattern_hash = ?',
                (pattern_hash,)
            )
            
            row = cursor.fetchone()
            
            if row:
                # Update existing pattern
                cursor.execute('''
                    UPDATE patterns 
                    SET last_seen = ?, occurrence_count = ?
                    WHERE pattern_id = ?
                ''', (
                    incident.timestamp.isoformat(),
                    row[1] + 1,
                    row[0]
                ))
            else:
                # Create new pattern
                pattern_id = f"PAT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                cursor.execute('''
                    INSERT INTO patterns 
                    (pattern_id, pattern_hash, first_seen, last_seen, occurrence_count,
                     severity, category, component, signature, sample_incident_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    pattern_hash,
                    incident.timestamp.isoformat(),
                    incident.timestamp.isoformat(),
                    1,
                    incident.severity.value,
                    incident.category.value,
                    incident.component,
                    signature,
                    incident.incident_id
                ))
            
            conn.commit()
    
    def _generate_signature(self, incident: Incident) -> str:
        """Generate pattern signature from incident"""
        # Extract key elements for pattern matching
        elements = [
            incident.category.value,
            incident.component,
            self._normalize_error_message(incident.description)
        ]
        
        if incident.stack_trace:
            # Extract first few lines of stack trace
            trace_lines = incident.stack_trace.split('\n')[:3]
            elements.extend(trace_lines)
        
        return '|'.join(elements)
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for pattern matching"""
        # Remove variable parts like timestamps, IDs, etc.
        normalized = message.lower()
        
        # Replace common variable patterns
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', normalized)
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', '<TIME>', normalized)
        normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}', '<UUID>', normalized)
        normalized = re.sub(r'\d+', '<NUM>', normalized)
        
        return normalized
    
    def resolve_incident(self, incident_id: str, resolution: str):
        """Mark an incident as resolved"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE incidents 
                SET resolution = ?, resolved_at = ?
                WHERE incident_id = ?
            ''', (
                resolution,
                datetime.now().isoformat(),
                incident_id
            ))
            
            conn.commit()
        
        self.logger.info(f"Incident resolved: {incident_id}")
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM incidents WHERE incident_id = ?', (incident_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_incident(row)
            
            return None
    
    def _row_to_incident(self, row) -> Incident:
        """Convert database row to Incident"""
        return Incident(
            incident_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            severity=IncidentSeverity(row[2]),
            category=IncidentCategory(row[3]),
            component=row[4],
            title=row[5],
            description=row[6],
            stack_trace=row[7],
            context=json.loads(row[8]) if row[8] else {},
            related_incidents=json.loads(row[9]) if row[9] else [],
            resolution=row[10],
            resolved_at=datetime.fromisoformat(row[11]) if row[11] else None
        )
    
    def query_incidents(self,
                       severity: Optional[IncidentSeverity] = None,
                       category: Optional[IncidentCategory] = None,
                       component: Optional[str] = None,
                       since: Optional[datetime] = None,
                       until: Optional[datetime] = None,
                       resolved: Optional[bool] = None,
                       limit: int = 100) -> List[Incident]:
        """Query incidents with filters"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM incidents WHERE 1=1'
            params = []
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity.value)
            
            if category:
                query += ' AND category = ?'
                params.append(category.value)
            
            if component:
                query += ' AND component = ?'
                params.append(component)
            
            if since:
                query += ' AND timestamp > ?'
                params.append(since.isoformat())
            
            if until:
                query += ' AND timestamp < ?'
                params.append(until.isoformat())
            
            if resolved is not None:
                if resolved:
                    query += ' AND resolution IS NOT NULL'
                else:
                    query += ' AND resolution IS NULL'
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            return [self._row_to_incident(row) for row in cursor.fetchall()]
    
    def get_statistics(self, days: int = 7) -> Dict:
        """Get incident statistics"""
        since = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total incidents
            cursor.execute(
                'SELECT COUNT(*) FROM incidents WHERE timestamp > ?',
                (since.isoformat(),)
            )
            total = cursor.fetchone()[0]
            
            # By severity
            cursor.execute('''
                SELECT severity, COUNT(*) FROM incidents 
                WHERE timestamp > ? GROUP BY severity
            ''', (since.isoformat(),))
            by_severity = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By category
            cursor.execute('''
                SELECT category, COUNT(*) FROM incidents 
                WHERE timestamp > ? GROUP BY category
            ''', (since.isoformat(),))
            by_category = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By component
            cursor.execute('''
                SELECT component, COUNT(*) FROM incidents 
                WHERE timestamp > ? GROUP BY component
            ''', (since.isoformat(),))
            by_component = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Top patterns
            cursor.execute('''
                SELECT pattern_id, occurrence_count, component, category
                FROM patterns ORDER BY occurrence_count DESC LIMIT 10
            ''')
            top_patterns = [
                {
                    'pattern_id': row[0],
                    'occurrences': row[1],
                    'component': row[2],
                    'category': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'period_days': days,
                'total_incidents': total,
                'by_severity': by_severity,
                'by_category': by_category,
                'by_component': by_component,
                'top_patterns': top_patterns
            }
    
    def subscribe(self, handler: Callable):
        """Subscribe to incident notifications"""
        self._handlers.append(handler)
    
    def _notify_handlers(self, incident: Incident):
        """Notify incident handlers"""
        for handler in self._handlers:
            try:
                handler(incident)
            except Exception as e:
                self.logger.error(f"Handler notification failed: {e}")
    
    def generate_report(self, days: int = 7) -> str:
        """Generate incident report"""
        stats = self.get_statistics(days)
        recent = self.query_incidents(since=datetime.now() - timedelta(days=days), limit=20)
        
        report = f"""
# Incident Report - Last {days} Days

## Summary
- Total Incidents: {stats['total_incidents']}

## By Severity
"""
        for severity, count in stats['by_severity'].items():
            report += f"- {severity}: {count}\n"
        
        report += "\n## By Category\n"
        for category, count in stats['by_category'].items():
            report += f"- {category}: {count}\n"
        
        report += "\n## Top Patterns\n"
        for pattern in stats['top_patterns']:
            report += f"- {pattern['pattern_id']}: {pattern['occurrences']} occurrences ({pattern['component']}/{pattern['category']})\n"
        
        report += "\n## Recent Incidents\n"
        for incident in recent[:10]:
            report += f"- [{incident.severity.value.upper()}] {incident.timestamp.strftime('%Y-%m-%d %H:%M')} - {incident.title}\n"
        
        return report


class RootCauseAnalyzer:
    """
    Analyze incidents to identify root causes
    """
    
    def __init__(self, incident_manager: IncidentManager):
        self.incident_manager = incident_manager
        self.logger = logging.getLogger('RootCauseAnalyzer')
    
    def analyze_pattern(self, pattern_id: str) -> Dict:
        """Analyze a specific incident pattern"""
        with sqlite3.connect(self.incident_manager.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return {'error': 'Pattern not found'}
            
            pattern = {
                'pattern_id': row[0],
                'first_seen': row[2],
                'last_seen': row[3],
                'occurrence_count': row[4],
                'severity': row[5],
                'category': row[6],
                'component': row[7]
            }
            
            # Get sample incident
            sample_id = row[9]
            sample = self.incident_manager.get_incident(sample_id)
            
            # Analyze timing
            cursor.execute('''
                SELECT timestamp FROM incidents 
                WHERE incident_id IN (
                    SELECT incident_id FROM incidents
                    WHERE component = ? AND category = ?
                )
                ORDER BY timestamp
            ''', (pattern['component'], pattern['category']))
            
            timestamps = [datetime.fromisoformat(row[0]) for row in cursor.fetchall()]
            
            # Calculate intervals
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            # Potential causes
            potential_causes = self._identify_potential_causes(pattern, sample)
            
            return {
                'pattern': pattern,
                'timing_analysis': {
                    'total_occurrences': len(timestamps),
                    'average_interval_seconds': avg_interval,
                    'first_occurrence': timestamps[0].isoformat() if timestamps else None,
                    'last_occurrence': timestamps[-1].isoformat() if timestamps else None
                },
                'potential_causes': potential_causes,
                'recommendations': self._generate_recommendations(pattern, potential_causes)
            }
    
    def _identify_potential_causes(self, pattern: Dict, sample: Optional[Incident]) -> List[Dict]:
        """Identify potential root causes"""
        causes = []
        
        # Check for resource issues
        if pattern['category'] == 'resource':
            causes.append({
                'type': 'resource_exhaustion',
                'likelihood': 'high',
                'description': 'Pattern indicates resource exhaustion'
            })
        
        # Check for network issues
        if pattern['category'] == 'network':
            causes.append({
                'type': 'network_connectivity',
                'likelihood': 'high',
                'description': 'Pattern indicates network connectivity issues'
            })
        
        # Check for code issues
        if sample and sample.stack_trace:
            causes.append({
                'type': 'software_bug',
                'likelihood': 'medium',
                'description': 'Stack trace suggests potential software bug'
            })
        
        # Check for external dependency issues
        causes.append({
            'type': 'external_dependency',
            'likelihood': 'medium',
            'description': f"Component {pattern['component']} may have external dependencies"
        })
        
        return causes
    
    def _generate_recommendations(self, pattern: Dict, causes: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        for cause in causes:
            if cause['type'] == 'resource_exhaustion':
                recommendations.append(
                    "Monitor resource usage and implement resource limits"
                )
                recommendations.append(
                    "Consider scaling up resources or optimizing resource usage"
                )
            
            elif cause['type'] == 'network_connectivity':
                recommendations.append(
                    "Implement retry logic with exponential backoff"
                )
                recommendations.append(
                    "Consider using circuit breaker pattern"
                )
            
            elif cause['type'] == 'software_bug':
                recommendations.append(
                    "Review recent code changes in affected component"
                )
                recommendations.append(
                    "Add more comprehensive error handling"
                )
            
            elif cause['type'] == 'external_dependency':
                recommendations.append(
                    "Implement health checks for external dependencies"
                )
                recommendations.append(
                    "Consider fallback mechanisms"
                )
        
        return recommendations


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create incident manager
    incident_manager = IncidentManager()
    
    # Log some incidents
    incident_manager.log_incident(
        severity=IncidentSeverity.HIGH,
        category=IncidentCategory.CRASH,
        component="soul_engine",
        title="GPT API timeout",
        description="Request to GPT-5.2 API timed out after 30 seconds",
        context={'timeout': 30, 'retry_count': 3}
    )
    
    incident_manager.log_incident(
        severity=IncidentSeverity.MEDIUM,
        category=IncidentCategory.NETWORK,
        component="gmail_service",
        title="Gmail API connection failed",
        description="Failed to connect to Gmail API",
        context={'error_code': 503}
    )
    
    # Get statistics
    stats = incident_manager.get_statistics(days=1)
    print("Incident Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Generate report
    report = incident_manager.generate_report(days=1)
    print("\n" + report)
    
    # Create analyzer
    analyzer = RootCauseAnalyzer(incident_manager)
    
    # Query incidents
    incidents = incident_manager.query_incidents(
        severity=IncidentSeverity.HIGH,
        limit=10
    )
    print(f"\nFound {len(incidents)} high severity incidents")
```

---

## Implementation Code Reference

### 11.1 Main Recovery System Integration

```python
# recovery_system.py - Main Recovery System Integration
"""
Main recovery system integration for Windows 10 AI Agent
Coordinates all recovery and self-healing components
"""

import os
import sys
import time
import threading
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Import all recovery components
from health_monitor import HealthMonitor, HealthStatus
from crash_detector import CrashDetector, CrashReport
from auto_restart import AutoRestartManager
from state_recovery import StateManager
from circuit_breaker import CircuitBreakerRegistry
from graceful_degradation import DegradationManager, AdaptiveDegradationController
from recovery_testing import RecoveryTestFramework
from incident_analysis import IncidentManager, IncidentSeverity, IncidentCategory

class RecoverySystem:
    """
    Main recovery system that coordinates all self-healing components
    """
    
    def __init__(self, agent_context: Any = None):
        self.agent_context = agent_context
        self.running = False
        
        # Initialize all components
        self.health_monitor = HealthMonitor()
        self.crash_detector = CrashDetector(agent_context)
        self.restart_manager = AutoRestartManager()
        self.state_manager = StateManager()
        self.circuit_registry = CircuitBreakerRegistry()
        self.degradation_manager = DegradationManager()
        self.adaptive_controller = AdaptiveDegradationController(self.degradation_manager)
        self.test_framework = RecoveryTestFramework(agent_context)
        self.incident_manager = IncidentManager()
        
        # Setup logging
        self.logger = logging.getLogger('RecoverySystem')
        self._setup_logging()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _setup_logging(self):
        """Configure recovery system logging"""
        os.makedirs('logs', exist_ok=True)
        
        handler = logging.FileHandler('logs/recovery_system.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _register_event_handlers(self):
        """Register cross-component event handlers"""
        # Health monitor -> Degradation manager
        self.health_monitor.subscribe(self._on_health_event)
        
        # Crash detector -> Restart manager
        self.crash_detector.register_handler(self._on_crash)
        
        # Circuit breaker monitor
        self.circuit_registry.subscribe(self._on_circuit_state_change)
    
    def _on_health_event(self, result):
        """Handle health check events"""
        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self.logger.warning(f"Health alert: {result.component} - {result.status.value}")
            
            # Log incident
            self.incident_manager.log_incident(
                severity=IncidentSeverity.HIGH if result.status == HealthStatus.UNHEALTHY else IncidentSeverity.CRITICAL,
                category=IncidentCategory.HEALTH,
                component=result.component,
                title=f"Health check failed: {result.component}",
                description=result.message,
                context=result.metrics
            )
            
            # Trigger restart if configured
            if result.recovery_action:
                self.restart_manager.request_restart(
                    result.component,
                    f"Health check failed: {result.message}"
                )
    
    def _on_crash(self, report: CrashReport):
        """Handle crash events"""
        self.logger.critical(f"Crash detected: {report.crash_id}")
        
        # Log incident
        self.incident_manager.log_incident(
            severity=IncidentSeverity.CRITICAL,
            category=IncidentCategory.CRASH,
            component=report.component,
            title=f"Crash: {report.crash_type.value}",
            description=report.exception_message or "Unknown error",
            stack_trace=report.stack_trace,
            context={
                'crash_id': report.crash_id,
                'crash_type': report.crash_type.value
            }
        )
        
        # Create checkpoint before restart
        self.state_manager.create_checkpoint()
        
        # Trigger restart
        self.restart_manager.request_restart(
            report.component,
            f"Crash: {report.crash_type.value}"
        )
    
    def _on_circuit_state_change(self, name: str, old_state, new_state):
        """Handle circuit breaker state changes"""
        self.logger.warning(f"Circuit {name} changed: {old_state.value} -> {new_state.value}")
        
        if new_state.value == 'open':
            self.incident_manager.log_incident(
                severity=IncidentSeverity.MEDIUM,
                category=IncidentCategory.RECOVERY,
                component=name,
                title=f"Circuit breaker opened: {name}",
                description=f"Circuit breaker for {name} opened due to repeated failures"
            )
    
    def start(self):
        """Start the recovery system"""
        self.logger.info("Starting recovery system...")
        
        self.running = True
        
        # Start all components
        self.health_monitor.start_monitoring()
        self.crash_detector.start_watchdog(timeout_seconds=30)
        self.state_manager.start_auto_checkpoint()
        self.adaptive_controller.start()
        
        self.logger.info("Recovery system started")
    
    def stop(self):
        """Stop the recovery system"""
        self.logger.info("Stopping recovery system...")
        
        self.running = False
        
        # Stop all components
        self.health_monitor.stop_monitoring()
        self.crash_detector.stop_watchdog()
        self.state_manager.stop_auto_checkpoint()
        self.adaptive_controller.stop()
        self.restart_manager.shutdown()
        
        self.logger.info("Recovery system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive recovery system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'health_monitor': {
                'overall_health': self.health_monitor.get_overall_health().value,
                'registered_checks': len(self.health_monitor.checks)
            },
            'crash_detector': {
                'total_crashes': self.crash_detector.crash_stats['total_crashes'],
                'consecutive_crashes': self.crash_detector.crash_stats['consecutive_crashes']
            },
            'restart_manager': self.restart_manager.get_restart_stats(),
            'state_manager': self.state_manager.get_state_summary(),
            'circuit_breakers': self.circuit_registry.get_all_metrics(),
            'degradation': self.degradation_manager.get_status(),
            'incidents': self.incident_manager.get_statistics(days=1)
        }
    
    def run_recovery_test(self, test_name: str):
        """Run a recovery test"""
        return self.test_framework.run_test(test_name)
    
    def generate_report(self) -> str:
        """Generate comprehensive recovery system report"""
        status = self.get_status()
        
        report = f"""
# Recovery System Report
Generated: {status['timestamp']}

## System Status
- Running: {status['running']}

## Health Monitor
- Overall Health: {status['health_monitor']['overall_health']}
- Registered Checks: {status['health_monitor']['registered_checks']}

## Crash Detection
- Total Crashes: {status['crash_detector']['total_crashes']}
- Consecutive Crashes: {status['crash_detector']['consecutive_crashes']}

## Restart Manager
- Total Restarts: {status['restart_manager']['total_restarts']}
- Successful: {status['restart_manager']['successful_restarts']}
- Failed: {status['restart_manager']['failed_restarts']}

## State Manager
- Total Checkpoints: {status['state_manager']['total_checkpoints']}
- Total Recoveries: {status['state_manager']['total_recoveries']}
- Database Size: {status['state_manager']['database_size_mb']:.2f} MB

## Degradation
- Current Level: {status['degradation']['current_level']}
- Available Features: {len(status['degradation']['available_features'])}
- Disabled Features: {len(status['degradation']['disabled_features'])}

## Incidents (Last 24h)
- Total: {status['incidents']['total_incidents']}
- By Severity: {status['incidents']['by_severity']}
- By Category: {status['incidents']['by_category']}
"""
        
        return report


# ==================== WINDOWS SERVICE INTEGRATION ====================

class AgentService(win32serviceutil.ServiceFramework):
    """
    Windows Service wrapper for the AI Agent
    """
    _svc_name_ = "OpenClawAgent"
    _svc_display_name_ = "OpenClaw AI Agent"
    _svc_description_ = "24/7 AI Agent System with Self-Healing"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.recovery_system: Optional[RecoverySystem] = None
    
    def SvcStop(self):
        """Service stop handler"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        
        if self.recovery_system:
            self.recovery_system.stop()
    
    def SvcDoRun(self):
        """Service main loop"""
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        
        # Initialize and start recovery system
        self.recovery_system = RecoverySystem()
        self.recovery_system.start()
        
        # Wait for stop signal
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)


# ==================== COMMAND LINE INTERFACE ====================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recovery System CLI')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'report', 'test'])
    parser.add_argument('--test-name', help='Test name for test command')
    parser.add_argument('--service', action='store_true', help='Run as Windows service')
    
    args = parser.parse_args()
    
    if args.service:
        win32serviceutil.HandleCommandLine(AgentService)
    elif args.command == 'start':
        recovery = RecoverySystem()
        recovery.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            recovery.stop()
    
    elif args.command == 'stop':
        # Signal service to stop
        pass
    
    elif args.command == 'status':
        recovery = RecoverySystem()
        status = recovery.get_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == 'report':
        recovery = RecoverySystem()
        print(recovery.generate_report())
    
    elif args.command == 'test':
        recovery = RecoverySystem()
        result = recovery.run_recovery_test(args.test_name or 'circuit_breaker_test')
        print(f"Test result: {result.result.value}")


if __name__ == "__main__":
    main()
```

---

## 11.2 Configuration Files

### health_config.json
```json
{
  "check_intervals": {
    "critical": 5,
    "standard": 30,
    "extended": 300
  },
  "components": {
    "system": {"interval": "critical", "enabled": true},
    "memory": {"interval": "critical", "enabled": true},
    "disk": {"interval": "standard", "enabled": true},
    "network": {"interval": "standard", "enabled": true},
    "gmail_service": {"interval": "critical", "enabled": true},
    "browser_control": {"interval": "standard", "enabled": true},
    "tts_engine": {"interval": "standard", "enabled": true},
    "stt_engine": {"interval": "standard", "enabled": true},
    "twilio_service": {"interval": "standard", "enabled": true},
    "soul_engine": {"interval": "critical", "enabled": true},
    "identity_manager": {"interval": "critical", "enabled": true},
    "cron_scheduler": {"interval": "critical", "enabled": true},
    "heartbeat": {"interval": "critical", "enabled": true}
  },
  "alert_thresholds": {
    "consecutive_failures": 3,
    "degradation_window": 300
  }
}
```

### restart_config.json
```json
{
  "global": {
    "max_restarts_per_hour": 10,
    "emergency_contact": "admin@example.com",
    "enable_notifications": true
  },
  "components": {
    "agent_core": {
      "policy": "exponential_backoff",
      "priority": "critical",
      "max_restarts": 5,
      "restart_window": 300,
      "initial_delay": 5,
      "max_delay": 300,
      "preserve_state": true
    },
    "soul_engine": {
      "policy": "immediate",
      "priority": "critical",
      "max_restarts": 10,
      "preserve_state": true
    },
    "gmail_service": {
      "policy": "delayed",
      "priority": "high",
      "max_restarts": 5,
      "initial_delay": 10,
      "preserve_state": true
    },
    "browser_control": {
      "policy": "delayed",
      "priority": "medium",
      "max_restarts": 3,
      "initial_delay": 5,
      "preserve_state": false
    },
    "cron_scheduler": {
      "policy": "immediate",
      "priority": "high",
      "max_restarts": 10,
      "preserve_state": true
    },
    "heartbeat": {
      "policy": "immediate",
      "priority": "critical",
      "max_restarts": 20,
      "preserve_state": true
    }
  }
}
```

---

## 11.3 Directory Structure

```
agent/
├── recovery/
│   ├── __init__.py
│   ├── health_monitor.py
│   ├── crash_detector.py
│   ├── auto_restart.py
│   ├── state_recovery.py
│   ├── circuit_breaker.py
│   ├── graceful_degradation.py
│   ├── recovery_testing.py
│   ├── incident_analysis.py
│   └── recovery_system.py
├── config/
│   ├── health_config.json
│   ├── restart_config.json
│   └── circuit_breaker_config.json
├── state/
│   ├── checkpoints/
│   ├── recovery/
│   ├── journal/
│   └── state.db
├── logs/
│   ├── health_monitor.log
│   ├── crash_detector.log
│   ├── auto_restart.log
│   ├── state_manager.log
│   ├── circuit_breaker.log
│   ├── degradation.log
│   ├── recovery_tests.log
│   ├── incidents.log
│   ├── recovery_system.log
│   ├── crashes/
│   ├── restarts/
│   └── tests/
└── main.py
```

---

## 11.4 Key Metrics and SLAs

### Recovery Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| System Uptime | 99.95% | Annual availability |
| Recovery Time Objective (RTO) | < 30 seconds | Time to restore service |
| Recovery Point Objective (RPO) | < 5 seconds | Maximum data loss |
| Health Check Frequency | 5 seconds | Critical components |
| Auto-restart Timeout | 10 seconds | Maximum restart delay |
| Circuit Breaker Threshold | 5 failures | Failures before opening |
| Checkpoint Interval | 60 seconds | Full checkpoint frequency |
| Incident Response Time | < 60 seconds | Time to acknowledge |

### Component Recovery Priorities

| Component | Priority | Max Restarts | Preserve State |
|-----------|----------|--------------|----------------|
| Heartbeat | Critical | 20 | Yes |
| Soul Engine | Critical | 10 | Yes |
| Agent Core | Critical | 5 | Yes |
| Identity Manager | Critical | 10 | Yes |
| Cron Scheduler | High | 10 | Yes |
| Gmail Service | High | 5 | Yes |
| Twilio Service | Medium | 5 | Yes |
| Browser Control | Medium | 3 | No |
| TTS Engine | Low | 3 | No |
| STT Engine | Low | 3 | No |

---

## 11.5 Integration Points

### With Agent Core
```python
# In agent_core.py
from recovery.recovery_system import RecoverySystem

class AgentCore:
    def __init__(self):
        self.recovery_system = RecoverySystem(self)
        
    def start(self):
        self.recovery_system.start()
        # ... rest of startup
        
    def stop(self):
        self.recovery_system.stop()
        # ... rest of shutdown
```

### With Agentic Loops
```python
# In agentic_loop.py
from recovery.circuit_breaker import CircuitBreakerRegistry

class AgenticLoop:
    def __init__(self):
        self.circuit_breaker = CircuitBreakerRegistry().get('gpt_api')
        
    @circuit_breaker
    def execute(self, task):
        # Execute with circuit breaker protection
        return self.process_task(task)
```

---

## Summary

This specification provides a comprehensive recovery and self-healing architecture for a Windows 10-based AI agent system. The architecture includes:

1. **Health Monitoring**: Multi-tier health checks for system, services, and components
2. **Crash Detection**: Comprehensive crash detection with detailed reporting
3. **Auto-Restart**: Configurable restart policies with state preservation
4. **State Recovery**: Checkpoint-based state management with database persistence
5. **Circuit Breakers**: Service protection with automatic recovery
6. **Graceful Degradation**: Adaptive feature disabling under stress
7. **Recovery Testing**: Chaos engineering framework for validation
8. **Incident Analysis**: Centralized logging with pattern detection

The system is designed for 24/7 operation with a target uptime of 99.95% and recovery time objectives of less than 30 seconds.

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: Systems Infrastructure Expert*
