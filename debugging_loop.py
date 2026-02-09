"""
Advanced Debugging Loop - Automated error detection, fix generation, and validation.

Implements the debugging loop per advanced_debugging_loop_specification.md.
Provides autonomous error detection, intelligent fix generation via LLM,
and safe validation of generated patches.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedError:
    """An error detected by the debugging loop."""
    error_id: str
    source: str
    error_type: str
    message: str
    stacktrace: str = ""
    severity: str = "medium"
    detected_at: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedFix:
    """A fix generated for a detected error."""
    fix_id: str
    error_id: str
    description: str
    patch: str = ""
    confidence: float = 0.0
    validated: bool = False
    applied: bool = False


class DebuggingLoop:
    """
    Autonomous debugging loop that detects errors, generates fixes via LLM,
    validates them, and tracks fix effectiveness.
    """

    ERROR_PATTERNS = {
        'import': r'(?i)(ImportError|ModuleNotFoundError|No module named)',
        'attribute': r'(?i)(AttributeError|has no attribute)',
        'type': r'(?i)(TypeError|unsupported operand|expected .+ got)',
        'value': r'(?i)(ValueError|invalid literal|could not convert)',
        'key': r'(?i)(KeyError|key not found)',
        'index': r'(?i)(IndexError|list index out of range)',
        'io': r'(?i)(IOError|FileNotFoundError|PermissionError|OSError)',
        'connection': r'(?i)(ConnectionError|TimeoutError|ConnectionRefused)',
        'syntax': r'(?i)(SyntaxError|IndentationError|TabError)',
        'memory': r'(?i)(MemoryError|allocation failed)',
        'runtime': r'(?i)(RuntimeError|RecursionError|StopIteration)',
    }

    def __init__(self, llm_client=None, config: Dict = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.error_history: List[DetectedError] = []
        self.fix_history: List[GeneratedFix] = []
        self._recent_logs: List[str] = []
        self._max_log_history = self.config.get('max_log_history', 500)

    async def run_single_cycle(self, context: Dict = None) -> Dict[str, Any]:
        """
        Run a single debugging cycle:
        1. Scan for recent errors
        2. Analyze and prioritize
        3. Generate fix suggestions (if LLM available)
        4. Return status report
        """
        context = context or {}
        cycle_start = datetime.utcnow()

        # Step 1: Detect errors from various sources
        errors = await self._detect_errors(context)

        # Step 2: Analyze and deduplicate
        new_errors = self._deduplicate_errors(errors)
        self.error_history.extend(new_errors)

        # Step 3: Generate fix suggestions for high-severity errors
        fixes = []
        for error in new_errors:
            if error.severity in ('critical', 'high'):
                fix = await self._generate_fix(error)
                if fix:
                    fixes.append(fix)
                    self.fix_history.append(fix)

        return {
            "cycle_time_ms": int((datetime.utcnow() - cycle_start).total_seconds() * 1000),
            "errors_detected": len(errors),
            "new_errors": len(new_errors),
            "fixes_generated": len(fixes),
            "total_error_history": len(self.error_history),
            "total_fix_history": len(self.fix_history),
        }

    async def _detect_errors(self, context: Dict) -> List[DetectedError]:
        """Scan log files and recent exceptions for errors."""
        errors = []
        import uuid

        # Check for errors passed in context
        if 'recent_exceptions' in context:
            for exc_info in context['recent_exceptions']:
                errors.append(DetectedError(
                    error_id=str(uuid.uuid4())[:12],
                    source=exc_info.get('source', 'unknown'),
                    error_type=exc_info.get('type', 'Exception'),
                    message=exc_info.get('message', ''),
                    stacktrace=exc_info.get('stacktrace', ''),
                    severity=exc_info.get('severity', 'medium'),
                    context=exc_info,
                ))

        # Scan log directory for recent ERROR/CRITICAL entries
        try:
            import os
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            if os.path.isdir(log_dir):
                for fname in sorted(os.listdir(log_dir))[-3:]:
                    if not fname.endswith('.log'):
                        continue
                    fpath = os.path.join(log_dir, fname)
                    try:
                        with open(fpath, 'r', errors='replace') as f:
                            # Read last 200 lines
                            lines = f.readlines()[-200:]
                        for line in lines:
                            if 'ERROR' in line or 'CRITICAL' in line:
                                errors.append(DetectedError(
                                    error_id=str(uuid.uuid4())[:12],
                                    source=fname,
                                    error_type='log_error',
                                    message=line.strip()[:500],
                                    severity='high' if 'CRITICAL' in line else 'medium',
                                ))
                    except OSError as e:
                        logger.debug(f"Could not read log file {fpath}: {e}")
        except (OSError, ValueError) as e:
            logger.debug(f"Log scan error: {e}")

        return errors

    def _deduplicate_errors(self, errors: List[DetectedError]) -> List[DetectedError]:
        """Remove errors we've already seen recently."""
        seen_messages = {e.message[:100] for e in self.error_history[-200:]}
        new_errors = []
        for error in errors:
            key = error.message[:100]
            if key not in seen_messages:
                new_errors.append(error)
                seen_messages.add(key)
        return new_errors

    def _classify_error(self, error: DetectedError) -> str:
        """Classify error using regex patterns."""
        import re
        combined = f"{error.error_type} {error.message} {error.stacktrace}"
        for category, pattern in self.ERROR_PATTERNS.items():
            if re.search(pattern, combined):
                return category
        return 'unknown'

    def _extract_source_context(self, error: DetectedError) -> str:
        """Extract enclosing function/class from stacktrace using AST."""
        import ast
        import os
        if not error.stacktrace:
            return ''
        # Parse last File reference from stacktrace
        lines = error.stacktrace.strip().split('\n')
        file_line = None
        line_no = None
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('File "'):
                parts = line.split('"')
                if len(parts) >= 2:
                    file_line = parts[1]
                # Extract line number
                if 'line ' in line:
                    try:
                        line_no = int(line.split('line ')[1].split(',')[0])
                    except (ValueError, IndexError):
                        pass
                break
        if not file_line or not line_no or not os.path.isfile(file_line):
            return ''
        try:
            with open(file_line, 'r', errors='replace') as f:
                source = f.read()
            tree = ast.parse(source)
            enclosing = ''
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if hasattr(node, 'end_lineno') and node.lineno <= line_no <= (node.end_lineno or node.lineno + 100):
                        enclosing = f"{type(node).__name__} {node.name} (lines {node.lineno}-{node.end_lineno})"
            return enclosing
        except (SyntaxError, OSError, ValueError):
            return ''

    def _load_fix_db(self) -> List[Dict]:
        """Load fix history from JSON file."""
        import os
        import json
        db_path = os.path.join(os.path.dirname(__file__) or '.', 'data', 'fix_history.json')
        if os.path.isfile(db_path):
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _find_similar_fixes(self, error: DetectedError) -> List[Dict]:
        """Find past fixes for similar errors."""
        history = self._load_fix_db()
        category = self._classify_error(error)
        return [
            fix for fix in history
            if fix.get('category') == category
            and fix.get('outcome') == 'success'
        ][:3]

    def record_fix_outcome(self, fix: 'GeneratedFix', outcome: str):
        """Record whether a fix succeeded or failed."""
        import os
        import json
        db_path = os.path.join(os.path.dirname(__file__) or '.', 'data', 'fix_history.json')
        history = self._load_fix_db()
        history.append({
            'fix_id': fix.fix_id,
            'error_id': fix.error_id,
            'description': fix.description[:500],
            'category': self._classify_error(
                next((e for e in self.error_history if e.error_id == fix.error_id),
                     DetectedError(error_id='', source='', error_type='', message=''))
            ),
            'outcome': outcome,
            'timestamp': datetime.utcnow().isoformat(),
        })
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            with open(db_path, 'w') as f:
                json.dump(history[-500:], f, indent=2)
        except OSError as e:
            logger.warning(f"Could not save fix history: {e}")

    def _validate_fix(self, patch: str) -> bool:
        """Validate that a generated patch is syntactically valid Python."""
        import ast
        try:
            ast.parse(patch)
            return True
        except SyntaxError:
            return False

    async def _generate_fix(self, error: DetectedError) -> Optional[GeneratedFix]:
        """Use LLM to generate a fix suggestion for an error."""
        import uuid

        if not self.llm_client:
            return GeneratedFix(
                fix_id=str(uuid.uuid4())[:12],
                error_id=error.error_id,
                description=f"Manual review needed: {error.message[:200]}",
                confidence=0.0,
            )

        try:
            category = self._classify_error(error)
            source_ctx = self._extract_source_context(error)
            similar = self._find_similar_fixes(error)

            similar_text = ''
            if similar:
                similar_text = '\n\nPrevious successful fixes for similar errors:\n'
                for s in similar:
                    similar_text += f"- {s.get('description', '')[:200]}\n"

            prompt = (
                f"Analyze this error and suggest a fix:\n"
                f"Category: {category}\n"
                f"Type: {error.error_type}\n"
                f"Message: {error.message}\n"
                f"Stacktrace: {error.stacktrace[:1000]}\n"
                f"Source: {error.source}\n"
            )
            if source_ctx:
                prompt += f"Enclosing context: {source_ctx}\n"
            prompt += similar_text
            prompt += "\nProvide a brief description of the fix and the code change needed."

            if hasattr(self.llm_client, 'generate'):
                response = self.llm_client.generate(prompt, system="You are a debugging assistant.")
            else:
                response = str(self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system="You are a debugging assistant.",
                    max_tokens=1024,
                ).get('content', ''))

            # Try to validate any code blocks in the response
            confidence = 0.5
            if '```' in response:
                code_blocks = response.split('```')
                for i in range(1, len(code_blocks), 2):
                    block = code_blocks[i]
                    if block.startswith('python\n'):
                        block = block[7:]
                    if self._validate_fix(block.strip()):
                        confidence = 0.7
                        break

            return GeneratedFix(
                fix_id=str(uuid.uuid4())[:12],
                error_id=error.error_id,
                description=response[:2000],
                confidence=confidence,
            )
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"Fix generation failed: {e}")
            return None
