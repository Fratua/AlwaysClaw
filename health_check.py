"""
System Health Check for OpenClaw AI Agent
Probes all optional dependencies on startup and reports availability.
"""

import sys
import os
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealthReport:
    overall_status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'overall_status': self.overall_status.value,
            'timestamp': self.timestamp.isoformat(),
            'components': [
                {
                    'name': c.name,
                    'status': c.status.value,
                    'message': c.message,
                    'details': c.details
                }
                for c in self.components
            ]
        }

    def to_log_string(self) -> str:
        lines = [f"=== System Health Report ({self.overall_status.value}) ==="]
        for c in self.components:
            icon = {"healthy": "+", "degraded": "~", "unavailable": "-", "unknown": "?"}
            lines.append(f"  [{icon.get(c.status.value, '?')}] {c.name}: {c.status.value} - {c.message}")
        return "\n".join(lines)


class StartupBlockedError(RuntimeError):
    """Raised when a critical health check fails and block_on_critical is True."""


class SystemHealthCheck:
    """Probes all optional dependencies and reports system health."""

    # Components whose failure should block startup when block_on_critical=True
    CRITICAL_COMPONENTS = {"python", "sqlite"}

    def __init__(self, critical_components: Optional[List[str]] = None):
        """
        Args:
            critical_components: Override the set of component names whose
                failure blocks startup. Defaults to CRITICAL_COMPONENTS.
        """
        if critical_components is not None:
            self.CRITICAL_COMPONENTS = set(critical_components)
        self._checks: List[callable] = [
            self._check_python_version,
            self._check_redis,
            self._check_postgresql,
            self._check_sqlite,
            self._check_playwright,
            self._check_pywin32,
            self._check_sounddevice,
            self._check_sentence_transformers,
            self._check_twilio,
            self._check_gmail,
            self._check_psutil,
            self._check_fastapi,
            self._check_openai,
            self._check_numpy,
        ]

    def run_all(self, block_on_critical: bool = False) -> SystemHealthReport:
        """
        Run all health checks and return report.

        Args:
            block_on_critical: If True, raise StartupBlockedError when any
                component listed in CRITICAL_COMPONENTS is UNAVAILABLE.
        """
        components = []
        for check in self._checks:
            try:
                result = check()
                components.append(result)
            except Exception as e:
                components.append(ComponentHealth(
                    name=check.__name__.replace('_check_', ''),
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}"
                ))

        # Determine overall status
        statuses = [c.status for c in components]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNAVAILABLE for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.DEGRADED

        report = SystemHealthReport(overall_status=overall, components=components)
        logger.info(report.to_log_string())

        # Block startup if critical components failed
        if block_on_critical:
            failed_critical = [
                c for c in components
                if c.name in self.CRITICAL_COMPONENTS
                and c.status in (HealthStatus.UNAVAILABLE, HealthStatus.UNKNOWN)
            ]
            if failed_critical:
                names = ", ".join(c.name for c in failed_critical)
                raise StartupBlockedError(
                    f"Critical health check(s) failed: {names}. "
                    f"Startup blocked. Details:\n{report.to_log_string()}"
                )

        return report

    def _check_python_version(self) -> ComponentHealth:
        version = sys.version
        major, minor = sys.version_info[:2]
        ok = major >= 3 and minor >= 10
        return ComponentHealth(
            name="python",
            status=HealthStatus.HEALTHY if ok else HealthStatus.DEGRADED,
            message=f"Python {version}",
            details={"version": version, "major": major, "minor": minor}
        )

    def _check_redis(self) -> ComponentHealth:
        try:
            import redis
            # Try connecting
            try:
                r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
                r.ping()
                return ComponentHealth(name="redis", status=HealthStatus.HEALTHY, message="Connected to localhost:6379")
            except (redis.ConnectionError, redis.TimeoutError):
                return ComponentHealth(name="redis", status=HealthStatus.UNAVAILABLE, message="redis package installed but server not reachable")
        except ImportError:
            return ComponentHealth(name="redis", status=HealthStatus.UNAVAILABLE, message="redis package not installed")

    def _check_postgresql(self) -> ComponentHealth:
        try:
            import asyncpg
        except ImportError:
            return ComponentHealth(name="postgresql", status=HealthStatus.UNAVAILABLE, message="asyncpg not installed")

        # Attempt actual connection using env config
        dsn = os.environ.get("DATABASE_URL", "")
        pg_host = os.environ.get("PG_HOST", "localhost")
        pg_port = os.environ.get("PG_PORT", "5432")
        pg_user = os.environ.get("PG_USER", "")
        pg_db = os.environ.get("PG_DATABASE", "")

        if not dsn and not pg_user:
            return ComponentHealth(
                name="postgresql",
                status=HealthStatus.DEGRADED,
                message="asyncpg installed but no connection config (set DATABASE_URL or PG_HOST/PG_USER/PG_DATABASE)",
            )

        import asyncio

        async def _try_connect():
            if dsn:
                conn = await asyncpg.connect(dsn, timeout=5)
            else:
                conn = await asyncpg.connect(
                    host=pg_host, port=int(pg_port),
                    user=pg_user, database=pg_db, timeout=5,
                )
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            return version

        try:
            version = asyncio.get_event_loop().run_until_complete(_try_connect())
            short_ver = str(version).split(",")[0] if version else "unknown"
            return ComponentHealth(
                name="postgresql", status=HealthStatus.HEALTHY,
                message=f"Connected: {short_ver}",
            )
        except Exception as e:
            return ComponentHealth(
                name="postgresql", status=HealthStatus.UNAVAILABLE,
                message=f"asyncpg installed but connection failed: {e}",
            )

    def _check_sqlite(self) -> ComponentHealth:
        import sqlite3
        version = sqlite3.sqlite_version
        return ComponentHealth(name="sqlite", status=HealthStatus.HEALTHY, message=f"SQLite {version}")

    def _check_playwright(self) -> ComponentHealth:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ComponentHealth(name="playwright", status=HealthStatus.UNAVAILABLE, message="playwright not installed")

        # Verify at least one browser is installed
        try:
            with sync_playwright() as p:
                # Try chromium first (most common)
                browser = p.chromium.launch(headless=True)
                version = browser.version
                browser.close()
                return ComponentHealth(
                    name="playwright", status=HealthStatus.HEALTHY,
                    message=f"Chromium {version} available",
                )
        except Exception as e:
            return ComponentHealth(
                name="playwright", status=HealthStatus.DEGRADED,
                message=f"Playwright installed but browser launch failed (run 'playwright install'): {e}",
            )

    def _check_pywin32(self) -> ComponentHealth:
        if sys.platform != 'win32':
            return ComponentHealth(name="pywin32", status=HealthStatus.DEGRADED, message="Not on Windows")
        try:
            import win32api
            return ComponentHealth(name="pywin32", status=HealthStatus.HEALTHY, message="pywin32 available")
        except ImportError:
            return ComponentHealth(name="pywin32", status=HealthStatus.UNAVAILABLE, message="pywin32 not installed")

    def _check_sounddevice(self) -> ComponentHealth:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return ComponentHealth(name="sounddevice", status=HealthStatus.HEALTHY, message=f"{len(devices)} audio devices found")
        except ImportError:
            return ComponentHealth(name="sounddevice", status=HealthStatus.UNAVAILABLE, message="sounddevice not installed")
        except Exception as e:
            return ComponentHealth(name="sounddevice", status=HealthStatus.DEGRADED, message=f"Installed but error: {e}")

    def _check_sentence_transformers(self) -> ComponentHealth:
        try:
            from sentence_transformers import SentenceTransformer
            return ComponentHealth(name="sentence_transformers", status=HealthStatus.HEALTHY, message="Available")
        except ImportError:
            return ComponentHealth(name="sentence_transformers", status=HealthStatus.UNAVAILABLE, message="Not installed")

    def _check_twilio(self) -> ComponentHealth:
        try:
            from twilio.rest import Client
        except ImportError:
            return ComponentHealth(name="twilio", status=HealthStatus.UNAVAILABLE, message="Not installed")

        import re
        sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
        token = os.environ.get('TWILIO_AUTH_TOKEN', '')

        if not sid or not token:
            return ComponentHealth(
                name="twilio", status=HealthStatus.DEGRADED,
                message="Installed but TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set",
            )

        # Validate SID format: starts with AC and is 34 chars
        warnings = []
        if not re.match(r'^AC[0-9a-fA-F]{32}$', sid):
            warnings.append("TWILIO_ACCOUNT_SID format invalid (expected AC + 32 hex chars)")
        # Auth token should be 32 hex chars
        if not re.match(r'^[0-9a-fA-F]{32}$', token):
            warnings.append("TWILIO_AUTH_TOKEN format invalid (expected 32 hex chars)")

        if warnings:
            return ComponentHealth(
                name="twilio", status=HealthStatus.DEGRADED,
                message=f"Credential format issues: {'; '.join(warnings)}",
            )

        return ComponentHealth(
            name="twilio", status=HealthStatus.HEALTHY,
            message="Installed with valid credential format",
        )

    def _check_gmail(self) -> ComponentHealth:
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            return ComponentHealth(name="gmail", status=HealthStatus.HEALTHY, message="Google API packages available")
        except ImportError:
            return ComponentHealth(name="gmail", status=HealthStatus.UNAVAILABLE, message="Google API packages not installed")

    def _check_psutil(self) -> ComponentHealth:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            return ComponentHealth(name="psutil", status=HealthStatus.HEALTHY, message=f"CPU: {cpu}%, Memory: {mem}%")
        except ImportError:
            return ComponentHealth(name="psutil", status=HealthStatus.UNAVAILABLE, message="Not installed")

    def _check_fastapi(self) -> ComponentHealth:
        try:
            import fastapi
            return ComponentHealth(name="fastapi", status=HealthStatus.HEALTHY, message=f"FastAPI {fastapi.__version__}")
        except ImportError:
            return ComponentHealth(name="fastapi", status=HealthStatus.UNAVAILABLE, message="Not installed")

    def _check_openai(self) -> ComponentHealth:
        try:
            import openai
            has_key = bool(os.environ.get('OPENAI_API_KEY'))
            if has_key:
                return ComponentHealth(name="openai", status=HealthStatus.HEALTHY, message="Installed with API key")
            return ComponentHealth(name="openai", status=HealthStatus.DEGRADED, message="Installed but no API key in env")
        except ImportError:
            return ComponentHealth(name="openai", status=HealthStatus.UNAVAILABLE, message="Not installed")

    def _check_numpy(self) -> ComponentHealth:
        try:
            import numpy as np
            return ComponentHealth(name="numpy", status=HealthStatus.HEALTHY, message=f"NumPy {np.__version__}")
        except ImportError:
            return ComponentHealth(name="numpy", status=HealthStatus.UNAVAILABLE, message="Not installed")


def run_startup_health_check(
    block_on_critical: bool = False,
    critical_components: Optional[List[str]] = None,
) -> SystemHealthReport:
    """
    Convenience function to run health check on startup.

    Args:
        block_on_critical: Raise StartupBlockedError if any critical component
            is unavailable.
        critical_components: Override which component names are considered critical.
    """
    checker = SystemHealthCheck(critical_components=critical_components)
    report = checker.run_all(block_on_critical=block_on_critical)

    # Write JSON report
    report_path = os.path.join(os.path.dirname(__file__), 'health_report.json')
    try:
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write health report: {e}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_startup_health_check()
    print(report.to_log_string())
    print(f"\nJSON report: {json.dumps(report.to_dict(), indent=2)}")
