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

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings based on word trigrams."""
        def trigrams(text: str):
            words = text.lower().split()
            if len(words) < 3:
                return set(words)
            return {tuple(words[i:i+3]) for i in range(len(words) - 2)}

        ta, tb = trigrams(a), trigrams(b)
        if not ta or not tb:
            return 0.0
        intersection = ta & tb
        union = ta | tb
        return len(intersection) / len(union)

    def _find_similar_fixes(self, error: DetectedError) -> List[Dict]:
        """Find past fixes for similar errors using category + text similarity."""
        history = self._load_fix_db()
        category = self._classify_error(error)
        error_text = f"{error.error_type} {error.message}"

        # First pass: same category and successful
        candidates = [
            fix for fix in history
            if fix.get('category') == category
            and fix.get('outcome') == 'success'
        ]

        # Score by text similarity to the error description
        scored = []
        for fix in candidates:
            fix_text = fix.get('description', '')
            sim = self._text_similarity(error_text, fix_text)
            scored.append((sim, fix))

        # Sort by similarity descending and return top 3
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fix for _, fix in scored[:3]]

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

    def _validate_fix(self, patch: str, error: Optional[DetectedError] = None) -> float:
        """
        Validate a generated patch beyond syntax checking.

        Returns a quality score between 0.0 and 1.0:
          - 0.0  syntax error
          - base 0.5 for valid syntax
          - penalties for anti-patterns (bare except, unused imports)
          - bonus if the patch references the error location
        """
        import ast
        import re

        # Step 1: syntax check
        try:
            tree = ast.parse(patch)
        except SyntaxError:
            return 0.0

        score = 0.5

        # Step 2: penalize bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                score -= 0.1
                break  # one penalty is enough

        # Step 3: penalize unused imports (simple heuristic)
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.append(alias.asname or alias.name)

        if imported_names:
            source_without_imports = re.sub(
                r'^\s*(import |from \S+ import ).*$', '', patch, flags=re.MULTILINE
            )
            unused = [n for n in imported_names if n not in source_without_imports]
            if unused:
                score -= min(0.1, 0.03 * len(unused))

        # Step 4: bonus if patch references the error location / message
        if error:
            error_tokens = set()
            # extract meaningful tokens from the error
            for part in (error.error_type, error.message, error.source):
                error_tokens.update(
                    tok for tok in re.findall(r'[A-Za-z_]\w+', part) if len(tok) > 3
                )
            if error_tokens:
                patch_text = patch.lower()
                matches = sum(1 for t in error_tokens if t.lower() in patch_text)
                if matches:
                    score += min(0.2, 0.05 * matches)

        return max(0.0, min(1.0, round(score, 2)))

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

            # Score confidence using multiple signals
            base_confidence = 0.3  # LLM produced a response

            # Signal 1: response length (very short responses are low quality)
            if len(response) > 200:
                base_confidence += 0.05
            if len(response) > 500:
                base_confidence += 0.05

            # Signal 2: validate code blocks with quality scoring
            best_block_score = 0.0
            if '```' in response:
                code_blocks = response.split('```')
                for i in range(1, len(code_blocks), 2):
                    block = code_blocks[i]
                    if block.startswith('python\n'):
                        block = block[7:]
                    block_score = self._validate_fix(block.strip(), error)
                    best_block_score = max(best_block_score, block_score)

            # Signal 3: similar fixes found (suggests known pattern)
            similarity_bonus = min(0.1, 0.04 * len(similar))

            # Signal 4: response references the error specifics
            import re as _re
            error_keywords = set(
                tok for tok in _re.findall(r'[A-Za-z_]\w+', f"{error.error_type} {error.message}")
                if len(tok) > 3
            )
            keyword_hits = sum(1 for kw in error_keywords if kw.lower() in response.lower())
            keyword_bonus = min(0.1, 0.02 * keyword_hits)

            confidence = min(0.95, round(
                base_confidence + best_block_score * 0.4 + similarity_bonus + keyword_bonus, 2
            ))

            return GeneratedFix(
                fix_id=str(uuid.uuid4())[:12],
                error_id=error.error_id,
                description=response[:2000],
                confidence=confidence,
            )
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"Fix generation failed: {e}")
            return None
