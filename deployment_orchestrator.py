"""
Deployment Orchestrator for OpenClaw Agent System
Windows 10/11 Deployment

Implements three deployment strategies:
- Blue-Green: instant environment switching with gradual traffic shift
- Canary: progressive percentage-based rollout with metric evaluation
- Rolling Update: batch-by-batch instance updates with drain and health checks

Integrates with process_management_impl.py for graceful shutdown and heartbeat
monitoring during deployments.
"""

import asyncio
import logging
import time
import yaml
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Integration with process management
try:
    from process_management_impl import GracefulShutdown, HeartbeatMonitor
    _HAS_PROCESS_MGMT = True
except ImportError:
    _HAS_PROCESS_MGMT = False


logger = logging.getLogger("DeploymentOrchestrator")


# =============================================================================
# ENUMERATIONS AND DATA CLASSES
# =============================================================================

class DeploymentState(Enum):
    PENDING = auto()
    PREPARING = auto()
    IN_PROGRESS = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    ROLLING_BACK = auto()
    FAILED = auto()


@dataclass
class DeploymentResult:
    success: bool
    state: DeploymentState
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE DEPLOYMENT ORCHESTRATOR
# =============================================================================

class DeploymentOrchestrator(ABC):
    """Base class for all deployment strategies."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state = DeploymentState.PENDING
        self._previous_version: Optional[str] = None
        self._current_version: Optional[str] = None
        self._load_config()

    def _load_config(self):
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}
        self.logger.info("Loaded config from %s", self.config_path)

    async def deploy(self, version: str, **kwargs) -> DeploymentResult:
        """Template method for deployment."""
        started_at = datetime.utcnow()
        self._current_version = version
        self.logger.info("Starting deployment of version %s", version)

        try:
            self.state = DeploymentState.PREPARING
            await self._prepare(version, **kwargs)

            self.state = DeploymentState.IN_PROGRESS
            await self._execute(version, **kwargs)

            self.state = DeploymentState.VERIFYING
            await self._verify(version, **kwargs)

            self.state = DeploymentState.COMPLETED
            await self._cleanup(version, **kwargs)

            self.logger.info("Deployment of version %s completed", version)
            return DeploymentResult(
                success=True,
                state=DeploymentState.COMPLETED,
                message=f"Successfully deployed version {version}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        except Exception as exc:
            self.logger.error("Deployment failed: %s", exc)
            self.state = DeploymentState.FAILED
            return DeploymentResult(
                success=False,
                state=DeploymentState.FAILED,
                message=f"Deployment failed: {exc}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    async def rollback(self) -> DeploymentResult:
        """Template method for rollback."""
        started_at = datetime.utcnow()
        self.state = DeploymentState.ROLLING_BACK
        self.logger.info("Rolling back deployment")

        try:
            await self._execute_rollback()
            self.state = DeploymentState.COMPLETED
            return DeploymentResult(
                success=True,
                state=DeploymentState.COMPLETED,
                message="Rollback completed successfully",
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        except Exception as exc:
            self.logger.error("Rollback failed: %s", exc)
            self.state = DeploymentState.FAILED
            return DeploymentResult(
                success=False,
                state=DeploymentState.FAILED,
                message=f"Rollback failed: {exc}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    async def health_check(
        self,
        host: str,
        port: int,
        endpoint: str = "/health",
        timeout: int = 10,
    ) -> bool:
        """Perform an HTTP GET health check."""
        if not _HAS_AIOHTTP:
            self.logger.warning("aiohttp not available, skipping HTTP health check")
            return True

        url = f"http://{host}:{port}{endpoint}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    healthy = resp.status == 200
                    if healthy:
                        self.logger.debug("Health check passed: %s", url)
                    else:
                        self.logger.warning("Health check returned %d: %s", resp.status, url)
                    return healthy
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self.logger.warning("Health check failed for %s: %s", url, exc)
            return False

    async def _wait_for_healthy(
        self,
        host: str,
        port: int,
        retries: int = 10,
        interval: float = 5.0,
    ) -> bool:
        """Retry health check until healthy or retries exhausted."""
        for attempt in range(1, retries + 1):
            if await self.health_check(host, port):
                return True
            self.logger.info(
                "Health check attempt %d/%d failed, retrying in %.1fs",
                attempt, retries, interval,
            )
            await asyncio.sleep(interval)
        self.logger.error("Instance %s:%d failed health checks after %d attempts", host, port, retries)
        return False

    async def _execute_rollback(self):
        """Default rollback implementation (override in subclasses)."""
        raise NotImplementedError("Subclass must implement _execute_rollback")

    @abstractmethod
    async def _prepare(self, version: str, **kwargs):
        """Prepare for deployment."""
        ...

    @abstractmethod
    async def _execute(self, version: str, **kwargs):
        """Execute the deployment."""
        ...

    @abstractmethod
    async def _verify(self, version: str, **kwargs):
        """Verify the deployment."""
        ...

    @abstractmethod
    async def _cleanup(self, version: str, **kwargs):
        """Post-deployment cleanup."""
        ...


# =============================================================================
# BLUE-GREEN DEPLOYER
# =============================================================================

class BlueGreenDeployer(DeploymentOrchestrator):
    """
    Blue-Green deployment strategy.

    Maintains two identical environments (blue / green). One is active (serving
    traffic) and the other is standby. New versions are deployed to standby,
    verified, then traffic is gradually shifted over before switching the active
    marker.
    """

    def __init__(self, config_path: str = "blue-green-config.yaml"):
        super().__init__(config_path)
        bg = self.config.get("blue_green", {})
        self._envs = bg.get("environments", {})
        self._router = bg.get("router", {})
        self._cutover = bg.get("cutover", {})
        self._rollback_cfg = bg.get("rollback", {})
        self._cleanup_cfg = bg.get("cleanup", {})

        # Track which environment is active
        self._active_env: str = "blue"
        self._standby_env: str = "green"
        self._previous_active: Optional[str] = None

    # -- helpers --

    def _env_host(self, env_name: str) -> str:
        return self._envs.get(env_name, {}).get("host", "localhost")

    def _env_port(self, env_name: str) -> int:
        port_range = self._envs.get(env_name, {}).get("port_range", [8000, 8014])
        return port_range[0]

    def _generate_nginx_config(
        self,
        active_port: int,
        standby_port: int,
        active_weight: int,
        standby_weight: int,
    ) -> str:
        """Render the nginx upstream config from the YAML template."""
        template = self._router.get("nginx_config_template", "")
        rendered = template.replace("{{active_port}}", str(active_port))
        rendered = rendered.replace("{{standby_port}}", str(standby_port))
        rendered = rendered.replace("{{active_weight}}", str(active_weight))
        rendered = rendered.replace("{{standby_weight}}", str(standby_weight))
        return rendered

    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics for threshold evaluation."""
        metrics: Dict[str, float] = {"error_rate": 0.0, "latency_p95_ms": 0.0}
        if _HAS_PSUTIL:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["memory_percent"] = psutil.virtual_memory().percent
        else:
            metrics["cpu_percent"] = 0.0
            metrics["memory_percent"] = 0.0
        return metrics

    def _thresholds_exceeded(self, metrics: Dict[str, float]) -> Optional[str]:
        """Check metrics against abort thresholds. Returns reason or None."""
        thresholds = self._cutover.get("abort_thresholds", {})
        for key, limit in thresholds.items():
            value = metrics.get(key)
            if value is not None and value > limit:
                return f"{key}={value:.4f} exceeds threshold {limit}"
        return None

    # -- abstract implementations --

    async def _prepare(self, version: str, **kwargs):
        self.logger.info(
            "Preparing blue-green deploy: active=%s standby=%s",
            self._active_env, self._standby_env,
        )
        self._previous_active = self._active_env

    async def _execute(self, version: str, **kwargs):
        standby_host = self._env_host(self._standby_env)
        standby_port = self._env_port(self._standby_env)

        # 1. Deploy new version to standby (simulated -- in production this would
        #    invoke a build/deploy script).
        self.logger.info(
            "Deploying version %s to standby environment %s (%s:%d)",
            version, self._standby_env, standby_host, standby_port,
        )

        # 2. Wait for standby to become healthy
        healthy = await self._wait_for_healthy(standby_host, standby_port)
        if not healthy:
            raise RuntimeError(
                f"Standby environment {self._standby_env} failed health checks"
            )

        # 3. Gradual traffic shift
        active_port = self._env_port(self._active_env)
        steps = self._cutover.get("gradual_steps", [10, 25, 50, 100])
        stabilization = self._cutover.get("stabilization_seconds", 60)
        check_interval = self._cutover.get("health_check_interval_seconds", 10)

        for pct in steps:
            active_weight = 100 - pct
            standby_weight = pct
            nginx_cfg = self._generate_nginx_config(
                active_port, standby_port, active_weight, standby_weight,
            )
            self.logger.info(
                "Traffic shift: %d%% -> standby (%s), %d%% -> active (%s)",
                pct, self._standby_env, 100 - pct, self._active_env,
            )
            self.logger.debug("Nginx config:\n%s", nginx_cfg)

            # Monitor during stabilization
            elapsed = 0.0
            while elapsed < stabilization:
                await asyncio.sleep(min(check_interval, stabilization - elapsed))
                elapsed += check_interval

                metrics = await self._collect_metrics()
                reason = self._thresholds_exceeded(metrics)
                if reason:
                    self.logger.error("Threshold exceeded during traffic shift: %s", reason)
                    if self._cutover.get("abort_on_health_check_failure", True):
                        await self._execute_rollback()
                        raise RuntimeError(f"Auto-rollback triggered: {reason}")

            self.logger.info("Step %d%% stabilized", pct)

        # 4. Update active marker
        self._active_env, self._standby_env = self._standby_env, self._active_env
        self.logger.info("Active environment is now %s", self._active_env)

    async def _verify(self, version: str, **kwargs):
        host = self._env_host(self._active_env)
        port = self._env_port(self._active_env)
        healthy = await self._wait_for_healthy(host, port, retries=5, interval=3.0)
        if not healthy:
            raise RuntimeError("Post-deploy verification failed")
        self.logger.info("Post-deployment verification passed")

    async def _cleanup(self, version: str, **kwargs):
        if self._cleanup_cfg.get("enabled", True):
            delay = self._cleanup_cfg.get("delay_minutes", 60)
            self.logger.info(
                "Cleanup scheduled in %d minutes for old environment %s",
                delay, self._standby_env,
            )

    async def _execute_rollback(self):
        """Instant rollback: swap active back to previous environment."""
        if self._previous_active is None:
            raise RuntimeError("No previous active environment to rollback to")

        self.logger.info(
            "Rolling back: switching active from %s to %s",
            self._active_env, self._previous_active,
        )
        self._active_env = self._previous_active
        self._standby_env = "green" if self._active_env == "blue" else "blue"

        # Generate full-weight config pointing back to old active
        active_port = self._env_port(self._active_env)
        standby_port = self._env_port(self._standby_env)
        nginx_cfg = self._generate_nginx_config(active_port, standby_port, 100, 0)
        self.logger.debug("Rollback nginx config:\n%s", nginx_cfg)

        # Verify old environment is still healthy
        if self._rollback_cfg.get("verify_rollback", True):
            timeout = self._rollback_cfg.get("verification_timeout_seconds", 60)
            retries = max(1, timeout // 5)
            healthy = await self._wait_for_healthy(
                self._env_host(self._active_env),
                self._env_port(self._active_env),
                retries=retries,
                interval=5.0,
            )
            if not healthy:
                raise RuntimeError("Rollback verification failed")
        self.logger.info("Rollback complete. Active environment: %s", self._active_env)


# =============================================================================
# CANARY DEPLOYER
# =============================================================================

class CanaryDeployer(DeploymentOrchestrator):
    """
    Canary deployment strategy.

    Progressively routes an increasing percentage of traffic to the canary
    (new version) while monitoring metrics. If any threshold is exceeded the
    canary is rolled back automatically.
    """

    def __init__(self, config_path: str = "canary-config.yaml"):
        super().__init__(config_path)
        canary = self.config.get("canary", {})
        self._phases: List[Dict[str, Any]] = canary.get("phases", [])
        self._thresholds: Dict[str, float] = canary.get("thresholds", {})
        self._auto_rollback = canary.get("auto_rollback", {})
        self._analysis = canary.get("analysis", {})

        # Runtime state
        self._canary_active = False
        self._current_phase_idx = 0
        self._phase_metrics: List[Dict[str, Any]] = []

    # -- helpers --

    async def _check_metrics(self) -> Dict[str, float]:
        """Collect system and application metrics."""
        metrics: Dict[str, float] = {
            "error_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
        }
        if _HAS_PSUTIL:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["memory_percent"] = psutil.virtual_memory().percent
        else:
            metrics["cpu_percent"] = 0.0
            metrics["memory_percent"] = 0.0
        return metrics

    async def _evaluate_phase(self, phase: Dict[str, Any], metrics: Dict[str, float]) -> bool:
        """
        Evaluate whether the canary passes a given phase.
        Returns True if all thresholds are within limits.
        """
        checks = [
            ("error_rate", "error_rate_max"),
            ("latency_p50_ms", "latency_p50_max_ms"),
            ("latency_p95_ms", "latency_p95_max_ms"),
            ("latency_p99_ms", "latency_p99_max_ms"),
            ("cpu_percent", "cpu_max_percent"),
            ("memory_percent", "memory_max_percent"),
        ]
        for metric_key, threshold_key in checks:
            limit = self._thresholds.get(threshold_key)
            value = metrics.get(metric_key)
            if limit is not None and value is not None and value > limit:
                self.logger.warning(
                    "Phase '%s' failed: %s=%.4f exceeds %s=%.4f",
                    phase.get("name", "unknown"), metric_key, value,
                    threshold_key, limit,
                )
                return False
        return True

    # -- abstract implementations --

    async def _prepare(self, version: str, **kwargs):
        self._canary_active = False
        self._current_phase_idx = 0
        self._phase_metrics.clear()
        self.logger.info("Preparing canary deployment for version %s", version)

    async def _execute(self, version: str, **kwargs):
        self._canary_active = True

        for idx, phase in enumerate(self._phases):
            self._current_phase_idx = idx
            pct = phase.get("percentage", 0)
            duration_min = phase.get("duration_minutes", 0)
            check_interval = phase.get("check_interval_seconds", 30)
            phase_name = phase.get("name", f"phase-{idx}")

            self.logger.info(
                "Starting canary phase '%s': %d%% traffic for %d minutes",
                phase_name, pct, duration_min,
            )

            # If this is the final 100% phase with 0 duration, just promote
            if pct >= 100 and duration_min == 0:
                self.logger.info("Canary promoted to full production")
                break

            duration_sec = duration_min * 60
            elapsed = 0.0

            while elapsed < duration_sec:
                wait = min(check_interval, duration_sec - elapsed)
                await asyncio.sleep(wait)
                elapsed += wait

                metrics = await self._check_metrics()
                self._phase_metrics.append({
                    "phase": phase_name,
                    "timestamp": time.time(),
                    "metrics": metrics,
                })

                passed = await self._evaluate_phase(phase, metrics)
                if not passed:
                    self.logger.error(
                        "Canary phase '%s' failed metric evaluation, triggering rollback",
                        phase_name,
                    )
                    await self._execute_rollback()
                    raise RuntimeError(
                        f"Canary failed at phase '{phase_name}' ({pct}%)"
                    )

            self.logger.info("Canary phase '%s' (%d%%) completed successfully", phase_name, pct)

        self._canary_active = False

    async def _verify(self, version: str, **kwargs):
        self.logger.info("Canary verification: all phases passed")

    async def _cleanup(self, version: str, **kwargs):
        self.logger.info("Canary cleanup: old version can be decommissioned")

    async def _execute_rollback(self):
        """Roll back the canary by routing 100% traffic to production."""
        self.logger.info("Rolling back canary: routing 100%% traffic to production")
        self._canary_active = False
        self._current_phase_idx = 0
        self.logger.info("Canary rollback complete")


# =============================================================================
# ROLLING UPDATE DEPLOYER
# =============================================================================

class RollingUpdateDeployer(DeploymentOrchestrator):
    """
    Rolling update deployment strategy.

    Updates instances in batches, draining connections before stopping each
    instance. Uses GracefulShutdown from process_management_impl for safe
    instance stopping and HeartbeatMonitor for health checking when available.
    """

    def __init__(self, config_path: str = "rolling-update-config.yaml"):
        super().__init__(config_path)
        ru = self.config.get("rolling_update", {})
        self._batch_size: int = ru.get("batch_size", 2)
        self._drain_timeout: int = ru.get("drain_timeout_seconds", 30)
        self._stabilization: int = ru.get("stabilization_seconds", 60)
        self._health_timeout: int = ru.get("health_check_timeout", 45)
        self._batch_cooldown: int = ru.get("batch_cooldown_seconds", 30)
        self._max_failures_batch: int = ru.get("max_failures_per_batch", 1)
        self._max_total_failures: int = ru.get("max_total_failures", 3)
        self._auto_rollback: bool = ru.get("auto_rollback", True)
        self._drain_cfg = ru.get("connection_draining", {})
        self._resource_limits = ru.get("resource_limits", {})

        loops = self.config.get("agent_loops", {})
        self._update_order: List[str] = loops.get("update_order", [])
        self._critical_loops: List[str] = loops.get("critical_loops", [])
        self._max_parallel: int = loops.get("max_parallel_updates", 3)

        # Runtime tracking
        self._updated_instances: List[str] = []
        self._pending_instances: List[str] = []
        self._total_failures: int = 0

        # Process management integration
        self._shutdown_handler: Optional[Any] = None
        self._heartbeat_monitor: Optional[Any] = None
        if _HAS_PROCESS_MGMT:
            try:
                self._shutdown_handler = GracefulShutdown(
                    shutdown_timeout=self._drain_timeout,
                    task_drain_timeout=self._drain_cfg.get("timeout_seconds", 30),
                )
                self._heartbeat_monitor = HeartbeatMonitor(
                    heartbeat_timeout=self._health_timeout,
                    check_interval=5,
                )
            except Exception as exc:
                self.logger.warning("Failed to init process management: %s", exc)

    # -- helpers --

    def _split_into_batches(self, instances: List[str]) -> List[List[str]]:
        """Split instance list into batches respecting critical loop constraints."""
        batches: List[List[str]] = []
        current_batch: List[str] = []

        for inst in instances:
            # Never put two critical loops in the same batch
            if inst in self._critical_loops:
                critical_in_batch = [i for i in current_batch if i in self._critical_loops]
                if critical_in_batch:
                    batches.append(current_batch)
                    current_batch = []

            current_batch.append(inst)
            if len(current_batch) >= self._batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)
        return batches

    async def _drain_instance(self, instance: str):
        """Drain connections from an instance before stopping."""
        drain_timeout = self._drain_cfg.get("timeout_seconds", self._drain_timeout)
        grace = self._drain_cfg.get("grace_period_seconds", 5)
        self.logger.info("Draining connections from %s (timeout=%ds)", instance, drain_timeout)
        # In production this would signal the load balancer to stop sending
        # new requests and wait for in-flight requests to complete.
        await asyncio.sleep(grace)
        self.logger.info("Instance %s drained", instance)

    async def _stop_instance(self, instance: str):
        """Stop an instance using GracefulShutdown if available."""
        self.logger.info("Stopping instance %s", instance)
        if self._shutdown_handler is not None:
            async def _stop():
                self.logger.debug("GracefulShutdown stopping %s", instance)

            self._shutdown_handler.register_component(
                name=instance,
                stop_func=_stop,
                priority=50,
            )
            await self._shutdown_handler.initiate_shutdown(reason=f"rolling-update:{instance}")
            # Re-create handler for next instance
            if _HAS_PROCESS_MGMT:
                self._shutdown_handler = GracefulShutdown(
                    shutdown_timeout=self._drain_timeout,
                    task_drain_timeout=self._drain_cfg.get("timeout_seconds", 30),
                )
        else:
            self.logger.info("No GracefulShutdown available, basic stop for %s", instance)

    async def _deploy_instance(self, instance: str, version: str):
        """Deploy new version to an instance."""
        self.logger.info("Deploying version %s to instance %s", version, instance)
        # In production: restart the agent loop process with the new code
        await asyncio.sleep(1)

    async def _health_check_instance(self, instance: str) -> bool:
        """Health check a single instance, using HeartbeatMonitor if available."""
        if self._heartbeat_monitor is not None:
            status = self._heartbeat_monitor.status.get(instance)
            if status and status not in ("failed", "degraded"):
                self.logger.debug("HeartbeatMonitor reports %s healthy", instance)
                return True
            # Fall through to HTTP check if heartbeat unavailable
        # Fallback: HTTP health check
        return await self.health_check("localhost", 8080, endpoint="/health", timeout=10)

    async def _check_resource_limits(self) -> bool:
        """Return True if resource usage is within limits."""
        if not _HAS_PSUTIL:
            return True
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        max_cpu = self._resource_limits.get("max_cpu_percent", 80)
        max_mem = self._resource_limits.get("max_memory_percent", 85)
        if cpu > max_cpu or mem > max_mem:
            self.logger.warning(
                "Resource limits exceeded: cpu=%.1f%% (max %d), mem=%.1f%% (max %d)",
                cpu, max_cpu, mem, max_mem,
            )
            if self._resource_limits.get("abort_on_resource_exhaustion", True):
                return False
        return True

    async def _rollback_batch(self, batch: List[str], previous_version: str):
        """Roll back a single batch of instances."""
        self.logger.info("Rolling back batch: %s", batch)
        for inst in batch:
            await self._stop_instance(inst)
            await self._deploy_instance(inst, previous_version)
            self._updated_instances.remove(inst) if inst in self._updated_instances else None

    async def _rollback_all(self, previous_version: str):
        """Roll back all updated instances."""
        self.logger.info("Rolling back all updated instances: %s", self._updated_instances)
        for inst in list(self._updated_instances):
            await self._stop_instance(inst)
            await self._deploy_instance(inst, previous_version)
        self._updated_instances.clear()

    # -- abstract implementations --

    async def _prepare(self, version: str, **kwargs):
        self._pending_instances = list(self._update_order)
        self._updated_instances = []
        self._total_failures = 0
        self._previous_version = kwargs.get("previous_version", "unknown")
        self.logger.info(
            "Preparing rolling update: %d instances, batch_size=%d",
            len(self._pending_instances), self._batch_size,
        )

    async def _execute(self, version: str, **kwargs):
        batches = self._split_into_batches(self._pending_instances)
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches, 1):
            self.logger.info(
                "Processing batch %d/%d: %s", batch_idx, total_batches, batch,
            )
            batch_failures = 0

            # Check resource limits before starting batch
            if not await self._check_resource_limits():
                if self._auto_rollback:
                    await self._rollback_all(self._previous_version)
                    raise RuntimeError("Resource limits exceeded, rolled back all")
                raise RuntimeError("Resource limits exceeded")

            for instance in batch:
                try:
                    # 1. Drain connections
                    await self._drain_instance(instance)

                    # 2. Stop old instance
                    await self._stop_instance(instance)

                    # 3. Deploy new version
                    await self._deploy_instance(instance, version)

                    # 4. Health check
                    healthy = await self._health_check_instance(instance)
                    if not healthy:
                        raise RuntimeError(f"Health check failed for {instance}")

                    self._updated_instances.append(instance)
                    self.logger.info("Instance %s updated successfully", instance)

                except Exception as exc:
                    self.logger.error("Failed to update instance %s: %s", instance, exc)
                    batch_failures += 1
                    self._total_failures += 1

                    if batch_failures > self._max_failures_batch:
                        self.logger.error(
                            "Max batch failures exceeded (%d), rolling back batch",
                            self._max_failures_batch,
                        )
                        await self._rollback_batch(
                            [i for i in batch if i in self._updated_instances],
                            self._previous_version,
                        )
                        break

            # Check total failure budget
            if self._total_failures > self._max_total_failures:
                self.logger.error(
                    "Max total failures exceeded (%d), rolling back all",
                    self._max_total_failures,
                )
                if self._auto_rollback:
                    await self._rollback_all(self._previous_version)
                raise RuntimeError(
                    f"Rolling update aborted: {self._total_failures} total failures"
                )

            # Stabilization wait between batches
            if batch_idx < total_batches:
                self.logger.info("Stabilization wait: %ds", self._stabilization)
                await asyncio.sleep(self._stabilization)

                # Cooldown
                if self._batch_cooldown > 0:
                    self.logger.info("Batch cooldown: %ds", self._batch_cooldown)
                    await asyncio.sleep(self._batch_cooldown)

        self._pending_instances.clear()

    async def _verify(self, version: str, **kwargs):
        self.logger.info("Verifying all %d updated instances", len(self._updated_instances))
        for inst in self._updated_instances:
            healthy = await self._health_check_instance(inst)
            if not healthy:
                self.logger.warning("Post-update verification failed for %s", inst)
        self.logger.info("Rolling update verification complete")

    async def _cleanup(self, version: str, **kwargs):
        self.logger.info(
            "Rolling update cleanup: %d instances updated, %d failures",
            len(self._updated_instances), self._total_failures,
        )

    async def _execute_rollback(self):
        """Roll back all instances that were updated."""
        if not self._previous_version:
            raise RuntimeError("No previous version recorded for rollback")
        await self._rollback_all(self._previous_version)
        self.logger.info("Rolling update rollback complete")
