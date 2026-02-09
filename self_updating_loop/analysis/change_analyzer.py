"""
Change Analyzer for Self-Updating Loop
Analyzes changes detected by the update detector and classifies their impact.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChangeAnalysisResult:
    impact: ChangeImpact = ChangeImpact.LOW
    affected_components: List[str] = field(default_factory=list)
    requires_restart: bool = False
    requires_migration: bool = False
    risk_score: float = 0.0
    summary: str = ""


class ChangeAnalyzer:
    """Analyzes detected changes and determines their impact."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info("ChangeAnalyzer initialized")

    async def analyze_change(self, change: Dict[str, Any]) -> ChangeAnalysisResult:
        """Analyze a single change and determine its impact."""
        change_type = change.get("type", "unknown")
        file_path = change.get("file_path", "")

        result = ChangeAnalysisResult(summary=f"Change analysis for {change_type}")

        # Classify by file type
        if file_path.endswith((".py", ".js", ".ts")):
            result.impact = ChangeImpact.MEDIUM
            result.affected_components.append("code")
        elif file_path.endswith((".yaml", ".yml", ".json", ".env")):
            result.impact = ChangeImpact.LOW
            result.affected_components.append("config")
            result.requires_migration = True
        elif file_path.endswith((".sql",)):
            result.impact = ChangeImpact.HIGH
            result.affected_components.append("database")

        # Critical paths
        critical_paths = ["service.js", "daemon-master.js", "python_bridge.py"]
        if any(cp in file_path for cp in critical_paths):
            result.impact = ChangeImpact.CRITICAL
            result.requires_restart = True

        result.risk_score = {
            ChangeImpact.LOW: 0.1,
            ChangeImpact.MEDIUM: 0.4,
            ChangeImpact.HIGH: 0.7,
            ChangeImpact.CRITICAL: 0.95,
        }.get(result.impact, 0.5)

        return result

    async def analyze_batch(self, changes: List[Dict[str, Any]]) -> List[ChangeAnalysisResult]:
        """Analyze a batch of changes."""
        results = []
        for change in changes:
            result = await self.analyze_change(change)
            results.append(result)
        return results
