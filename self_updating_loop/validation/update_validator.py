"""
Update Validator for Self-Updating Loop
Validates updates before they are applied to ensure safety.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0


class UpdateValidator:
    """Validates updates before application."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        logger.info("UpdateValidator initialized")

    async def validate_update(self, update: Dict[str, Any]) -> ValidationResult:
        """Validate a single update."""
        result = ValidationResult()

        checks = [
            self._check_file_size,
            self._check_syntax,
            self._check_forbidden_patterns,
        ]
        result.checks_total = len(checks)

        for check in checks:
            passed, message = await check(update)
            if passed:
                result.checks_passed += 1
            else:
                result.valid = False
                result.errors.append(message)

        return result

    async def _check_file_size(self, update: Dict[str, Any]) -> tuple:
        content = update.get("content", "")
        if len(content) > self.max_file_size:
            return False, f"File exceeds max size ({len(content)} > {self.max_file_size})"
        return True, "OK"

    async def _check_syntax(self, update: Dict[str, Any]) -> tuple:
        file_path = update.get("file_path", "")
        content = update.get("content", "")
        if file_path.endswith(".py"):
            try:
                compile(content, file_path, "exec")
                return True, "OK"
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"
        return True, "OK"

    async def _check_forbidden_patterns(self, update: Dict[str, Any]) -> tuple:
        content = update.get("content", "")
        forbidden = ["os.system(", "subprocess.call(", "__import__("]
        for pattern in forbidden:
            if pattern in content:
                return False, f"Forbidden pattern found: {pattern}"
        return True, "OK"
