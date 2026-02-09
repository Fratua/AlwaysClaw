"""
Context Engineering Loop - Autonomous Context Window Optimization System
For Windows 10 OpenClaw-Inspired AI Agent Framework

This module implements the Context Engineering Loop for optimizing context windows
in a GPT-5.2 powered AI agent system with 24/7 operation capabilities.
"""

import asyncio
import hashlib
import json
import logging
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from collections import deque
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class PriorityLevel(Enum):
    """Priority levels for context elements."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class TruncationStrategy(Enum):
    """Available truncation strategies."""
    HEAD = "head"
    TAIL = "tail"
    MIDDLE = "middle"
    SMART = "smart"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SENTENCE_BOUNDARY = "sentence"
    PARAGRAPH_BOUNDARY = "paragraph"


class CompressionLevel(Enum):
    """Compression levels for archiving."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetailLevel(Enum):
    """Detail levels for context reconstruction."""
    MINIMAL = "minimal"
    MEDIUM = "medium"
    FULL = "full"


class OptimizationType(Enum):
    """Types of optimization tasks."""
    COMPRESSION = "compression"
    TRUNCATION = "truncation"
    ARCHIVE = "archive"
    RECONSTRUCT = "reconstruct"


@dataclass
class ContextElement:
    """Represents a single element in the context window."""
    type: str
    content: str
    token_count: int
    timestamp: datetime
    role: str = "assistant"
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: PriorityLevel = PriorityLevel.MEDIUM
    relevance_score: float = 0.0


@dataclass
class RelevanceScore:
    """Multi-dimensional relevance score."""
    overall: float
    semantic: float = 0.0
    temporal: float = 0.0
    structural: float = 0.0
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    compressed_elements: List[ContextElement]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    information_preserved: float
    
    @classmethod
    def empty(cls):
        return cls(
            compressed_elements=[],
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
            information_preserved=1.0
        )


@dataclass
class TruncationResult:
    """Result of a truncation operation."""
    truncated_element: Optional[ContextElement]
    was_truncated: bool
    tokens_removed: int
    truncation_method: str = ""
    
    @classmethod
    def empty(cls):
        return cls(
            truncated_element=None,
            was_truncated=False,
            tokens_removed=0
        )


@dataclass
class ContextSnapshot:
    """Snapshot of context window state."""
    timestamp: datetime
    total_tokens: int
    system_tokens: int
    conversation_tokens: int
    tool_output_tokens: int
    memory_tokens: int
    available_tokens: int
    utilization_rate: float


@dataclass
class TokenUsageEntry:
    """Single token usage entry."""
    timestamp: datetime
    component: str
    tokens_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenStats:
    """Statistics for token usage."""
    total_tokens: int
    count: int
    average: float
    min_tokens: int
    max_tokens: int
    last_hour: int
    last_day: int


@dataclass
class OptimizedContext:
    """Final optimized context for LLM call."""
    messages: List[Dict[str, str]]
    total_tokens: int
    was_compressed: bool
    elements_included: int
    elements_dropped: int


@dataclass
class MonitorConfig:
    """Configuration for context monitoring."""
    model_context_limit: int = 256000
    reserve_tokens: int = 16000
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95
    monitoring_interval: float = 5.0


@dataclass
class LoopConfig:
    """Configuration for the Context Engineering Loop."""
    model_context_limit: int = 256000
    reserve_tokens: int = 16000
    formatting_buffer: int = 500
    monitoring_interval: float = 5.0
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95
    compression_threshold: float = 0.85
    truncation_threshold: float = 0.95
    hot_memory_size: int = 1000
    warm_memory_size: int = 10000
    archive_after_days: int = 7
    min_relevance_score: float = 0.6
    max_context_elements: int = 100
    enable_caching: bool = True
    cache_size: int = 10000


@dataclass
class ConversationExchange:
    """A single conversation exchange."""
    id: str
    conversation_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """Extracted fact from conversation."""
    content: str
    source_exchange_id: str
    timestamp: datetime
    category: str


# =============================================================================
# TOKEN COUNTING
# =============================================================================

class TokenCounter:
    """
    Accurate token counting for GPT-5.2 with caching and batch operations.
    """
    
    def __init__(self, model: str = "gpt-5.2"):
        self.model = model
        self.tokenizer = self._load_tokenizer()
        self.cache = {}
        self.cache_size = 10000
        self.batch_size = 100
        
    def _load_tokenizer(self):
        """Load appropriate tokenizer."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model("gpt-4")  # Fallback
        except ImportError:
            logger.warning("tiktoken not available, using simple tokenizer")
            return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens with caching."""
        if not text:
            return 0
            
        cache_key = hashlib.md5(text[:200].encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.tokenizer:
            token_count = len(self.tokenizer.encode(text))
        else:
            # Simple approximation: ~4 characters per token
            token_count = len(text) // 4
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        
        self.cache[cache_key] = token_count
        return token_count
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens for message list including formatting overhead."""
        total = 0
        base_overhead = 3  # Formatting tokens per message
        
        for msg in messages:
            total += base_overhead
            total += self.count_tokens(msg.get('role', ''))
            total += self.count_tokens(msg.get('content', ''))
            
            if 'name' in msg:
                total += self.count_tokens(msg['name'])
            if 'function_call' in msg:
                total += self.count_tokens(str(msg['function_call']))
        
        total += 3  # Assistant priming
        return total
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """Efficient batch token counting."""
        return [self.count_tokens(t) for t in texts]


# =============================================================================
# RELEVANCE SCORING
# =============================================================================

class SemanticRelevanceScorer:
    """Semantic relevance using embeddings and similarity metrics."""

    def __init__(self):
        self.cache = {}
        self.embedding_dim = 384  # MiniLM dimension
        self._model = None
        self._use_embeddings = False
        self._init_model()

    def _init_model(self):
        """Attempt to load sentence-transformers model for real embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self._use_embeddings = True
            logger.info("SemanticRelevanceScorer: using sentence-transformers embeddings")
        except ImportError:
            logger.info(
                "sentence-transformers not installed; falling back to "
                "TF-IDF keyword matching for semantic relevance"
            )
            self._model = None
            self._use_embeddings = False

    def _tfidf_similarity(self, content: str, query: str) -> float:
        """TF-IDF weighted keyword matching as fallback."""
        import math

        content_lower = content.lower()
        query_lower = query.lower()

        # Tokenize
        content_words = re.findall(r'\b\w+\b', content_lower)
        query_words = re.findall(r'\b\w+\b', query_lower)

        if not content_words or not query_words:
            return 0.0

        # Build term frequency maps
        content_tf: Dict[str, float] = {}
        for w in content_words:
            content_tf[w] = content_tf.get(w, 0) + 1
        for w in content_tf:
            content_tf[w] /= len(content_words)

        query_tf: Dict[str, float] = {}
        for w in query_words:
            query_tf[w] = query_tf.get(w, 0) + 1
        for w in query_tf:
            query_tf[w] /= len(query_words)

        # IDF approximation using both documents as corpus
        all_words = set(content_tf.keys()) | set(query_tf.keys())
        idf: Dict[str, float] = {}
        for w in all_words:
            doc_count = (1 if w in content_tf else 0) + (1 if w in query_tf else 0)
            idf[w] = math.log(2.0 / doc_count) + 1.0

        # Compute TF-IDF vectors and cosine similarity
        dot_product = 0.0
        norm_c = 0.0
        norm_q = 0.0
        for w in all_words:
            c_val = content_tf.get(w, 0.0) * idf[w]
            q_val = query_tf.get(w, 0.0) * idf[w]
            dot_product += c_val * q_val
            norm_c += c_val ** 2
            norm_q += q_val ** 2

        if norm_c == 0 or norm_q == 0:
            return 0.0
        return dot_product / (norm_c ** 0.5 * norm_q ** 0.5)

    async def score(self, content: str, query: str) -> float:
        """Calculate semantic relevance score using embeddings or keyword fallback."""
        if not content or not query:
            return 0.0

        # Check cache
        cache_key = hashlib.md5(f"{content[:200]}|{query[:200]}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self._use_embeddings and self._model is not None:
            try:
                embeddings = self._model.encode([content, query], convert_to_numpy=True)
                # Cosine similarity
                cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
                ))
                score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug(f"Embedding scoring failed, using fallback: {e}")
                score = self._tfidf_similarity(content, query)
        else:
            score = self._tfidf_similarity(content, query)

        self.cache[cache_key] = score
        # Limit cache size
        if len(self.cache) > 10000:
            self.cache.clear()
        return score

    async def batch_score(self, contents: List[str], query: str) -> List[float]:
        """Score multiple contents efficiently."""
        if self._use_embeddings and self._model is not None and contents:
            try:
                all_texts = contents + [query]
                embeddings = self._model.encode(all_texts, convert_to_numpy=True)
                query_emb = embeddings[-1]
                query_norm = np.linalg.norm(query_emb) + 1e-8
                scores = []
                for i in range(len(contents)):
                    cos_sim = float(
                        np.dot(embeddings[i], query_emb)
                        / (np.linalg.norm(embeddings[i]) * query_norm + 1e-8)
                    )
                    scores.append(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))
                return scores
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug(f"Batch embedding scoring failed: {e}")
        return [await self.score(c, query) for c in contents]


class TemporalRelevanceScorer:
    """Temporal relevance based on recency and conversation flow."""
    
    def __init__(self, half_life_seconds: float = 300):
        self.half_life = half_life_seconds
        
    def score(self, element_timestamp: datetime, current_time: datetime) -> float:
        """Calculate temporal relevance score."""
        age_seconds = (current_time - element_timestamp).total_seconds()
        # Exponential decay
        return 0.5 ** (age_seconds / self.half_life)


class ImportanceScorer:
    """Score importance based on content characteristics."""
    
    IMPORTANCE_KEYWORDS = {
        'critical': 1.0, 'important': 0.9, 'must': 0.9, 'required': 0.85,
        'essential': 0.85, 'error': 0.8, 'failed': 0.8, 'success': 0.7,
        'decided': 0.75, 'agreed': 0.7, 'deadline': 0.8, 'urgent': 0.9,
        'password': 0.95, 'api_key': 0.95, 'token': 0.9, 'secret': 0.95,
        'decision': 0.8, 'conclusion': 0.8, 'action': 0.75, 'task': 0.7
    }
    
    def score(self, content: str) -> float:
        """Calculate importance score."""
        content_lower = content.lower()
        max_score = 0.0
        
        for keyword, weight in self.IMPORTANCE_KEYWORDS.items():
            if keyword in content_lower:
                max_score = max(max_score, weight)
        
        return max_score


class RelevanceEngine:
    """Multi-dimensional relevance scoring for context elements."""
    
    def __init__(self):
        self.semantic_scorer = SemanticRelevanceScorer()
        self.temporal_scorer = TemporalRelevanceScorer()
        self.importance_scorer = ImportanceScorer()
        
        self.weights = {
            'semantic': 0.35,
            'temporal': 0.25,
            'structural': 0.20,
            'importance': 0.20
        }
    
    async def score_context_element(
        self,
        element: ContextElement,
        current_query: str,
        current_time: datetime = None
    ) -> RelevanceScore:
        """Calculate comprehensive relevance score."""
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Semantic relevance
        semantic_score = await self.semantic_scorer.score(
            element.content, current_query
        )
        
        # Temporal relevance
        temporal_score = self.temporal_scorer.score(
            element.timestamp, current_time
        )
        
        # Importance score
        importance_score = self.importance_scorer.score(element.content)
        
        # Structural relevance (based on element type)
        structural_score = self._calculate_structural_score(element)
        
        # Combined weighted score
        combined_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['temporal'] * temporal_score +
            self.weights['structural'] * structural_score +
            self.weights['importance'] * importance_score
        )
        
        return RelevanceScore(
            overall=combined_score,
            semantic=semantic_score,
            temporal=temporal_score,
            structural=structural_score,
            importance=importance_score,
            metadata={
                'element_type': element.type,
                'element_age': (current_time - element.timestamp).total_seconds(),
                'token_count': element.token_count
            }
        )
    
    def _calculate_structural_score(self, element: ContextElement) -> float:
        """Calculate structural relevance based on element type."""
        type_scores = {
            'system_prompt': 1.0,
            'user_message': 1.0,
            'tool_output': 0.8,
            'assistant_response': 0.7,
            'memory_context': 0.6,
            'retrieved_doc': 0.5,
            'summary': 0.4
        }
        return type_scores.get(element.type, 0.5)


# =============================================================================
# COMPRESSION STRATEGIES
# =============================================================================

class CompressionStrategy(ABC):
    """Base class for compression strategies."""
    
    @abstractmethod
    async def compress(
        self,
        elements: List[ContextElement],
        target_tokens: int
    ) -> CompressionResult:
        """Compress context elements to fit target token budget."""
        ...


class ExtractionStrategy(CompressionStrategy):
    """Extract key information using pattern matching."""
    
    KEY_PATTERNS = {
        'decisions': r'\b(decided|conclusion|agreed|determined|resolved)\b.*?[.!?]',
        'actions': r'\b(will|must|should|need to|going to)\s+\w+.*?[.!?]',
        'facts': r'\b(\d+(?:\.\d+)?)\s*(?:GB|MB|KB|ms|seconds|minutes|hours|dollars|\$|%)',
        'errors': r'\b(error|failed|exception|crash|bug|issue)\b.*?[.!?]',
        'results': r'\b(success|completed|done|finished)\b.*?[.!?]',
    }
    
    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter
    
    async def compress(
        self,
        elements: List[ContextElement],
        target_tokens: int
    ) -> CompressionResult:
        """Extract key sentences matching important patterns."""
        
        extracted_by_category = {cat: [] for cat in self.KEY_PATTERNS.keys()}
        current_tokens = 50  # Base overhead
        
        for element in elements:
            content = element.content
            
            for category, pattern in self.KEY_PATTERNS.items():
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    sentence = match.group(0).strip()
                    sentence_tokens = self.counter.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens <= target_tokens:
                        extracted_by_category[category].append(sentence)
                        current_tokens += sentence_tokens
        
        # Format extracted content
        formatted = self._format_extracted(extracted_by_category)
        formatted_tokens = self.counter.count_tokens(formatted)
        
        compressed = ContextElement(
            type="extracted_key_points",
            content=formatted,
            token_count=formatted_tokens,
            timestamp=datetime.utcnow(),
            metadata={'extraction_categories': list(extracted_by_category.keys())}
        )
        
        original_tokens = sum(e.token_count for e in elements)
        
        return CompressionResult(
            compressed_elements=[compressed],
            original_tokens=original_tokens,
            compressed_tokens=formatted_tokens,
            compression_ratio=formatted_tokens / original_tokens if original_tokens > 0 else 1.0,
            information_preserved=0.7
        )
    
    def _format_extracted(self, extracted: Dict[str, List[str]]) -> str:
        """Format extracted points by category."""
        lines = ["[Key points from earlier context:]"]
        
        category_names = {
            'decisions': "Decisions",
            'actions': "Actions",
            'facts': "Facts/Numbers",
            'errors': "Issues",
            'results': "Results"
        }
        
        for category, points in extracted.items():
            if points:
                lines.append(f"\n{category_names.get(category, category)}:")
                for point in points[:5]:
                    lines.append(f"  - {point}")
        
        return "\n".join(lines)


class HierarchicalCompressionStrategy(CompressionStrategy):
    """Multi-level compression with different strategies for different ages."""
    
    def __init__(self, token_counter: TokenCounter):
        self.extraction = ExtractionStrategy(token_counter)
        self.counter = token_counter
    
    async def compress(
        self,
        elements: List[ContextElement],
        target_tokens: int,
        preserve_recent: int = 5
    ) -> CompressionResult:
        """Apply hierarchical compression."""
        
        if not elements:
            return CompressionResult.empty()
        
        # Sort by timestamp (newest last)
        sorted_elements = sorted(elements, key=lambda e: e.timestamp)
        
        # Split by age
        recent = sorted_elements[-preserve_recent:] if len(sorted_elements) > preserve_recent else sorted_elements
        older = sorted_elements[:-preserve_recent] if len(sorted_elements) > preserve_recent else []
        
        # Calculate budgets
        recent_tokens = sum(e.token_count for e in recent)
        remaining_budget = target_tokens - recent_tokens
        
        compressed_elements = list(recent)
        
        # Compress older with extraction
        if older and remaining_budget > 200:
            older_budget = min(remaining_budget * 0.8, int(target_tokens * 0.3))
            extraction = await self.extraction.compress(older, older_budget)
            compressed_elements.extend(extraction.compressed_elements)
        
        total_compressed = sum(e.token_count for e in compressed_elements)
        original_tokens = sum(e.token_count for e in elements)
        
        return CompressionResult(
            compressed_elements=compressed_elements,
            original_tokens=original_tokens,
            compressed_tokens=total_compressed,
            compression_ratio=total_compressed / original_tokens if original_tokens > 0 else 1.0,
            information_preserved=0.75
        )


class CompressionOrchestrator:
    """Orchestrates multiple compression strategies."""
    
    def __init__(self, token_counter: TokenCounter):
        self.strategies = [
            HierarchicalCompressionStrategy(token_counter),
            ExtractionStrategy(token_counter)
        ]
        self.counter = token_counter
    
    async def compress(
        self,
        elements: List[ContextElement],
        target_tokens: int
    ) -> CompressionResult:
        """Compress elements using best available strategy."""
        
        current_tokens = sum(e.token_count for e in elements)
        
        if current_tokens <= target_tokens:
            return CompressionResult(
                compressed_elements=elements,
                original_tokens=current_tokens,
                compressed_tokens=current_tokens,
                compression_ratio=1.0,
                information_preserved=1.0
            )
        
        # Try strategies in order
        for strategy in self.strategies:
            try:
                result = await strategy.compress(elements, target_tokens)
                if result.compressed_tokens <= target_tokens:
                    return result
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Compression strategy failed: {e}")
                continue
        
        # Fallback: simple truncation
        return await self._fallback_compression(elements, target_tokens)
    
    async def _fallback_compression(
        self,
        elements: List[ContextElement],
        target_tokens: int
    ) -> CompressionResult:
        """Fallback: Keep highest priority elements."""
        
        # Sort by priority
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.priority.value, e.relevance_score),
            reverse=True
        )
        
        kept = []
        total_tokens = 0
        
        for element in sorted_elements:
            if total_tokens + element.token_count <= target_tokens:
                kept.append(element)
                total_tokens += element.token_count
        
        original_tokens = sum(e.token_count for e in elements)
        
        return CompressionResult(
            compressed_elements=kept,
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
            compression_ratio=total_tokens / original_tokens if original_tokens > 0 else 1.0,
            information_preserved=0.5
        )


# =============================================================================
# TRUNCATION
# =============================================================================

class IntelligentTruncator:
    """Smart truncation that preserves meaning and critical information."""
    
    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter
    
    async def truncate(
        self,
        element: ContextElement,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.SMART
    ) -> TruncationResult:
        """Truncate content intelligently."""
        
        if element.token_count <= max_tokens:
            return TruncationResult(
                truncated_element=element,
                was_truncated=False,
                tokens_removed=0
            )
        
        # Select truncation method
        if element.type == 'conversation':
            return await self._truncate_conversation(element, max_tokens)
        elif element.type == 'tool_output':
            return await self._truncate_tool_output(element, max_tokens)
        else:
            return await self._truncate_generic(element, max_tokens)
    
    async def _truncate_conversation(
        self,
        element: ContextElement,
        max_tokens: int
    ) -> TruncationResult:
        """Truncate conversation keeping most recent messages."""
        
        # Parse messages (simplified)
        lines = element.content.split('\n')
        
        kept_lines = []
        total_tokens = 0
        
        # Work backwards
        for line in reversed(lines):
            line_tokens = self.counter.count_tokens(line)
            if total_tokens + line_tokens <= max_tokens:
                kept_lines.insert(0, line)
                total_tokens += line_tokens
            else:
                break
        
        truncated_content = '\n'.join(kept_lines)
        
        truncated = ContextElement(
            type=element.type,
            content=truncated_content,
            token_count=self.counter.count_tokens(truncated_content),
            timestamp=element.timestamp,
            metadata={**element.metadata, 'was_truncated': True}
        )
        
        return TruncationResult(
            truncated_element=truncated,
            was_truncated=True,
            tokens_removed=element.token_count - truncated.token_count,
            truncation_method='conversation_recent_first'
        )
    
    async def _truncate_tool_output(
        self,
        element: ContextElement,
        max_tokens: int
    ) -> TruncationResult:
        """Truncate tool output preserving errors and summary."""
        
        content = element.content
        
        # Look for error indicators
        error_match = re.search(r'(error|failed|exception)[^\n]*', content, re.IGNORECASE)
        
        # Keep first part (summary) and error if present
        keep_parts = []
        
        # First 30% for summary
        summary_end = int(len(content) * 0.3)
        summary = content[:summary_end]
        
        if self.counter.count_tokens(summary) < max_tokens * 0.5:
            keep_parts.append(summary)
        
        # Add error if found
        if error_match:
            error_text = error_match.group(0)
            keep_parts.append(f"\n... [Error: {error_text}] ...")
        
        truncated_content = ''.join(keep_parts)
        truncated_content += "\n[Output truncated]"
        
        truncated = ContextElement(
            type=element.type,
            content=truncated_content,
            token_count=self.counter.count_tokens(truncated_content),
            timestamp=element.timestamp,
            metadata={**element.metadata, 'was_truncated': True}
        )
        
        return TruncationResult(
            truncated_element=truncated,
            was_truncated=True,
            tokens_removed=element.token_count - truncated.token_count,
            truncation_method='tool_output_summary_error'
        )
    
    async def _truncate_generic(
        self,
        element: ContextElement,
        max_tokens: int
    ) -> TruncationResult:
        """Generic truncation at sentence boundary."""
        
        content = element.content
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        kept_sentences = []
        total_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.counter.count_tokens(sentence)
            if total_tokens + sentence_tokens <= max_tokens - 20:  # Buffer
                kept_sentences.append(sentence)
                total_tokens += sentence_tokens
            else:
                break
        
        truncated_content = ' '.join(kept_sentences)
        if len(kept_sentences) < len(sentences):
            truncated_content += " [Content truncated]"
        
        truncated = ContextElement(
            type=element.type,
            content=truncated_content,
            token_count=self.counter.count_tokens(truncated_content),
            timestamp=element.timestamp,
            metadata={**element.metadata, 'was_truncated': True}
        )
        
        return TruncationResult(
            truncated_element=truncated,
            was_truncated=True,
            tokens_removed=element.token_count - truncated.token_count,
            truncation_method='sentence_boundary'
        )


# =============================================================================
# CONTEXT MONITORING
# =============================================================================

class ContextMonitor:
    """Real-time monitoring system for context window utilization."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.alert_handlers = []
        self.monitoring_active = False
        self.current_utilization = 0.0
        
    async def start_monitoring(self):
        """Begin continuous context window monitoring."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                snapshot = await self.capture_snapshot()
                await self.analyze_snapshot(snapshot)
                await asyncio.sleep(self.config.monitoring_interval)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    def set_context_source(self, context_source) -> None:
        """Register the ContextEngineeringLoop for real token tracking."""
        self._context_source = context_source

    async def capture_snapshot(self) -> ContextSnapshot:
        """Capture current state of context window from the actual context source."""
        source = getattr(self, '_context_source', None)

        if source is not None:
            counter = source.token_tracker.counter
            memory = source.memory

            # Count system tokens (from configuration overhead)
            system_tokens = 0
            conversation_tokens = 0
            tool_output_tokens = 0
            memory_tokens = 0

            # Sum tokens across memory tiers
            for exchange in memory.hot_memory:
                user_tok = counter.count_tokens(exchange.user_message)
                asst_tok = counter.count_tokens(exchange.assistant_response)
                conversation_tokens += user_tok + asst_tok

            for exchange in memory.warm_memory:
                memory_tokens += counter.count_tokens(
                    exchange.user_message + exchange.assistant_response
                )

            # Add fact tokens
            for fact in memory.facts:
                memory_tokens += counter.count_tokens(fact.content)

            # Get recent usage from tracker for tool outputs
            tracker = source.token_tracker
            for entry in list(tracker.usage_log)[-100:]:
                if entry.component == 'tool_outputs':
                    tool_output_tokens += entry.tokens_used
                elif entry.component == 'system_prompt':
                    system_tokens += entry.tokens_used

            total_tokens = system_tokens + conversation_tokens + tool_output_tokens + memory_tokens
            available_tokens = max(0, self.config.model_context_limit - total_tokens)
            utilization_rate = total_tokens / self.config.model_context_limit if self.config.model_context_limit > 0 else 0.0

            return ContextSnapshot(
                timestamp=datetime.utcnow(),
                total_tokens=total_tokens,
                system_tokens=system_tokens,
                conversation_tokens=conversation_tokens,
                tool_output_tokens=tool_output_tokens,
                memory_tokens=memory_tokens,
                available_tokens=available_tokens,
                utilization_rate=utilization_rate
            )

        # Fallback when no context source is registered
        return ContextSnapshot(
            timestamp=datetime.utcnow(),
            total_tokens=0,
            system_tokens=0,
            conversation_tokens=0,
            tool_output_tokens=0,
            memory_tokens=0,
            available_tokens=self.config.model_context_limit,
            utilization_rate=self.current_utilization
        )
    
    async def analyze_snapshot(self, snapshot: ContextSnapshot):
        """Analyze snapshot and trigger optimizations if needed."""
        self.metrics_history.append(snapshot)
        self.current_utilization = snapshot.utilization_rate
        
        # Check thresholds
        if snapshot.utilization_rate >= self.config.critical_threshold:
            await self._trigger_critical_alert(snapshot)
        elif snapshot.utilization_rate >= self.config.warning_threshold:
            await self._trigger_warning_alert(snapshot)
    
    async def _trigger_critical_alert(self, snapshot: ContextSnapshot):
        """Trigger critical utilization alert."""
        logger.critical(f"CRITICAL: Context utilization at {snapshot.utilization_rate:.1%}")
        for handler in self.alert_handlers:
            await handler('critical', snapshot)
    
    async def _trigger_warning_alert(self, snapshot: ContextSnapshot):
        """Trigger warning utilization alert."""
        logger.warning(f"WARNING: Context utilization at {snapshot.utilization_rate:.1%}")
        for handler in self.alert_handlers:
            await handler('warning', snapshot)
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_average_utilization(self, hours: int = 1) -> float:
        """Get average utilization over time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        relevant = [m for m in self.metrics_history if m.timestamp >= cutoff]
        
        if not relevant:
            return 0.0
        
        return statistics.mean(m.utilization_rate for m in relevant)
    
    def get_peak_utilization(self, hours: int = 24) -> float:
        """Get peak utilization over time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        relevant = [m for m in self.metrics_history if m.timestamp >= cutoff]
        
        if not relevant:
            return 0.0
        
        return max(m.utilization_rate for m in relevant)


# =============================================================================
# TOKEN USAGE TRACKING
# =============================================================================

class TokenUsageTracker:
    """Comprehensive tracking of token usage across all system components."""
    
    def __init__(self):
        self.usage_log = deque(maxlen=10000)
        self.component_usage = {
            'system_prompt': [],
            'conversation': [],
            'tool_outputs': [],
            'memory_context': [],
            'retrieved_docs': [],
            'agent_state': []
        }
        self.daily_budget = 1000000
        self.hourly_budget = 50000
        self.counter = TokenCounter()
    
    async def track_usage(self, component: str, tokens: int, metadata: Dict = None):
        """Track token usage for a specific component."""
        
        entry = TokenUsageEntry(
            timestamp=datetime.utcnow(),
            component=component,
            tokens_used=tokens,
            metadata=metadata or {}
        )
        
        self.usage_log.append(entry)
        
        if component in self.component_usage:
            self.component_usage[component].append(entry)
    
    def get_usage_last_hour(self) -> int:
        """Get total usage in last hour."""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        return sum(
            e.tokens_used for e in self.usage_log
            if e.timestamp >= cutoff
        )
    
    def get_usage_last_day(self) -> int:
        """Get total usage in last day."""
        cutoff = datetime.utcnow() - timedelta(days=1)
        return sum(
            e.tokens_used for e in self.usage_log
            if e.timestamp >= cutoff
        )
    
    def get_component_breakdown(self) -> Dict[str, TokenStats]:
        """Get token usage breakdown by component."""
        
        breakdown = {}
        
        for component, entries in self.component_usage.items():
            if not entries:
                continue
            
            tokens = [e.tokens_used for e in entries]
            cutoff_hour = datetime.utcnow() - timedelta(hours=1)
            cutoff_day = datetime.utcnow() - timedelta(days=1)
            
            breakdown[component] = TokenStats(
                total_tokens=sum(tokens),
                count=len(tokens),
                average=statistics.mean(tokens) if tokens else 0,
                min_tokens=min(tokens) if tokens else 0,
                max_tokens=max(tokens) if tokens else 0,
                last_hour=sum(e.tokens_used for e in entries if e.timestamp >= cutoff_hour),
                last_day=sum(e.tokens_used for e in entries if e.timestamp >= cutoff_day)
            )
        
        return breakdown


# =============================================================================
# MEMORY STORE
# =============================================================================

class MemoryStore:
    """Multi-tier memory store for historical context management."""
    
    def __init__(self, token_counter: TokenCounter):
        self.hot_memory = deque(maxlen=1000)
        self.warm_memory = deque(maxlen=10000)
        self.facts = []
        self.token_counter = token_counter
        self.exchanges_by_conversation = {}
    
    async def add_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Dict = None
    ):
        """Add a conversation exchange to memory."""
        
        exchange = ConversationExchange(
            id=f"ex_{datetime.utcnow().timestamp()}",
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Add to hot memory
        self.hot_memory.append(exchange)
        
        # Index by conversation
        if conversation_id not in self.exchanges_by_conversation:
            self.exchanges_by_conversation[conversation_id] = []
        self.exchanges_by_conversation[conversation_id].append(exchange)
        
        # Extract facts
        await self._extract_facts(exchange)
    
    async def _extract_facts(self, exchange: ConversationExchange):
        """Extract facts from exchange."""
        
        content = f"{exchange.user_message} {exchange.assistant_response}"
        
        # Simple keyword-based extraction
        importance_scorer = ImportanceScorer()
        importance = importance_scorer.score(content)
        
        if importance > 0.7:
            fact = Fact(
                content=content[:200],
                source_exchange_id=exchange.id,
                timestamp=exchange.timestamp,
                category='extracted'
            )
            self.facts.append(fact)
    
    async def get_recent_context(
        self,
        conversation_id: str,
        max_tokens: int,
        max_messages: int = 50
    ) -> List[ContextElement]:
        """Get recent conversation context."""
        
        exchanges = self.exchanges_by_conversation.get(conversation_id, [])
        recent = list(exchanges)[-max_messages:]
        
        elements = []
        total_tokens = 0
        
        for exchange in recent:
            # User message
            user_tokens = self.token_counter.count_tokens(exchange.user_message)
            if total_tokens + user_tokens <= max_tokens:
                elements.append(ContextElement(
                    type='conversation',
                    content=exchange.user_message,
                    token_count=user_tokens,
                    timestamp=exchange.timestamp,
                    role='user'
                ))
                total_tokens += user_tokens
            
            # Assistant response
            assistant_tokens = self.token_counter.count_tokens(exchange.assistant_response)
            if total_tokens + assistant_tokens <= max_tokens:
                elements.append(ContextElement(
                    type='conversation',
                    content=exchange.assistant_response,
                    token_count=assistant_tokens,
                    timestamp=exchange.timestamp,
                    role='assistant'
                ))
                total_tokens += assistant_tokens
        
        return elements
    
    async def search_relevant_context(
        self,
        query: str,
        conversation_id: str = None,
        max_results: int = 5,
        min_relevance: float = 0.3
    ) -> List[ContextElement]:
        """Search for semantically relevant historical context."""
        
        scorer = SemanticRelevanceScorer()
        
        # Get candidate exchanges
        if conversation_id:
            candidates = self.exchanges_by_conversation.get(conversation_id, [])
        else:
            candidates = list(self.hot_memory)
        
        # Score relevance
        scored = []
        for exchange in candidates:
            combined = f"{exchange.user_message} {exchange.assistant_response}"
            score = await scorer.score(combined, query)
            if score >= min_relevance:
                scored.append((exchange, score))
        
        # Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to elements
        elements = []
        for exchange, score in scored[:max_results]:
            elements.append(ContextElement(
                type='memory_context',
                content=f"[Relevant history: {exchange.user_message} -> {exchange.assistant_response[:100]}...]",
                token_count=self.token_counter.count_tokens(exchange.user_message) + 50,
                timestamp=exchange.timestamp,
                metadata={'relevance_score': score}
            ))
        
        return elements


# =============================================================================
# CONTEXT PRIORITIZER
# =============================================================================

@dataclass
class PrioritizedContext:
    """Result of context prioritization."""
    selected_elements: List[ContextElement]
    total_tokens: int
    available_tokens: int
    utilization: float
    dropped_elements: int


class ContextPrioritizer:
    """Dynamic context prioritization based on multiple factors."""
    
    def __init__(self, token_counter: TokenCounter):
        self.relevance_engine = RelevanceEngine()
        self.token_counter = token_counter
    
    async def prioritize_context(
        self,
        context_elements: List[ContextElement],
        current_query: str,
        available_tokens: int
    ) -> PrioritizedContext:
        """Prioritize context elements to fit within token budget."""
        
        # Score all elements
        scored_elements = []
        for element in context_elements:
            relevance = await self.relevance_engine.score_context_element(
                element, current_query
            )
            element.relevance_score = relevance.overall
            scored_elements.append((element, relevance))
        
        # Sort by relevance score (descending)
        scored_elements.sort(key=lambda x: x[1].overall, reverse=True)
        
        # Select elements within budget
        selected = []
        total_tokens = 0
        
        for element, relevance in scored_elements:
            element_tokens = element.token_count
            
            if total_tokens + element_tokens <= available_tokens:
                selected.append(element)
                total_tokens += element_tokens
        
        return PrioritizedContext(
            selected_elements=selected,
            total_tokens=total_tokens,
            available_tokens=available_tokens,
            utilization=total_tokens / available_tokens if available_tokens > 0 else 0,
            dropped_elements=len(context_elements) - len(selected)
        )


# =============================================================================
# MAIN CONTEXT ENGINEERING LOOP
# =============================================================================

class ContextEngineeringLoop:
    """
    Main Context Engineering Loop for the OpenClaw agent system.
    Integrates all components for autonomous context optimization.
    """
    
    def __init__(self, config: LoopConfig = None):
        if config is None:
            config = self._load_config_from_yaml()
        self.config = config
        
        # Initialize components
        self.monitor = ContextMonitor(MonitorConfig(
            model_context_limit=self.config.model_context_limit,
            reserve_tokens=self.config.reserve_tokens,
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold,
            monitoring_interval=self.config.monitoring_interval
        ))
        
        self.token_tracker = TokenUsageTracker()
        self.prioritizer = ContextPrioritizer(TokenCounter())
        self.compressor = CompressionOrchestrator(TokenCounter())
        self.truncator = IntelligentTruncator(TokenCounter())
        self.memory = MemoryStore(TokenCounter())
        
        # State
        self.loop_active = False
        self.optimization_queue = asyncio.Queue()
        
    @staticmethod
    def _load_config_from_yaml() -> 'LoopConfig':
        """Load LoopConfig from cpel_config.yaml using ConfigLoader."""
        try:
            from config_loader import get_config
            cfg = get_config("cpel_config", "context_prompt_engineering_loop", {})
            if not cfg:
                return LoopConfig()
            perf = cfg.get('performance', {})
            opt = cfg.get('optimization', {})
            return LoopConfig(
                monitoring_interval=perf.get('collection_interval', 5.0),
                warning_threshold=opt.get('improvement_threshold', 0.80),
            )
        except (ImportError, Exception):
            return LoopConfig()

    async def start(self):
        """Start the Context Engineering Loop."""
        self.loop_active = True

        # Start monitoring
        asyncio.create_task(self.monitor.start_monitoring())

        logger.info("Context Engineering Loop started")
    
    async def stop(self):
        """Stop the Context Engineering Loop."""
        self.loop_active = False
        self.monitor.monitoring_active = False
        logger.info("Context Engineering Loop stopped")
    
    async def build_optimized_context(
        self,
        system_prompt: str,
        user_message: str,
        conversation_id: str,
        tool_outputs: List[str] = None,
        include_memory: bool = True
    ) -> OptimizedContext:
        """
        Build an optimized context window for an LLM call.
        
        This is the main entry point for context optimization.
        """
        
        counter = self.token_tracker.counter
        
        # Calculate available tokens
        system_tokens = counter.count_tokens(system_prompt)
        user_tokens = counter.count_tokens(user_message)
        
        available_for_context = (
            self.config.model_context_limit
            - self.config.reserve_tokens
            - system_tokens
            - user_tokens
            - self.config.formatting_buffer
        )
        
        # Gather context elements
        elements = []
        
        # Add recent conversation
        recent = await self.memory.get_recent_context(
            conversation_id,
            max_tokens=int(available_for_context * 0.4)
        )
        elements.extend(recent)
        
        # Add relevant historical context
        if include_memory:
            relevant = await self.memory.search_relevant_context(
                user_message,
                conversation_id=conversation_id,
                max_results=3
            )
            elements.extend(relevant)
        
        # Add tool outputs
        if tool_outputs:
            for output in tool_outputs:
                elements.append(ContextElement(
                    type='tool_output',
                    content=output,
                    token_count=counter.count_tokens(output),
                    timestamp=datetime.utcnow()
                ))
        
        # Score and prioritize
        prioritized = await self.prioritizer.prioritize_context(
            elements,
            user_message,
            available_for_context
        )
        
        # Apply compression if needed
        if prioritized.utilization > self.config.compression_threshold:
            compressed = await self.compressor.compress(
                prioritized.selected_elements,
                available_for_context
            )
            final_elements = compressed.compressed_elements
        else:
            final_elements = prioritized.selected_elements
        
        # Build final context
        messages = [{"role": "system", "content": system_prompt}]
        
        for element in final_elements:
            messages.append({
                "role": element.role,
                "content": element.content
            })
        
        messages.append({"role": "user", "content": user_message})
        
        # Track token usage
        total_tokens = counter.count_messages(messages)
        await self.token_tracker.track_usage('conversation', total_tokens)
        
        # Update monitor
        utilization = total_tokens / self.config.model_context_limit
        self.monitor.current_utilization = utilization
        
        return OptimizedContext(
            messages=messages,
            total_tokens=total_tokens,
            was_compressed=prioritized.utilization > self.config.compression_threshold,
            elements_included=len(final_elements),
            elements_dropped=prioritized.dropped_elements
        )
    
    async def record_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Dict = None
    ):
        """Record a completed exchange for future context."""
        await self.memory.add_exchange(
            conversation_id, user_message, assistant_response, metadata
        )
    
    def get_metrics(self) -> Dict:
        """Get current metrics for the loop."""
        return {
            'context_utilization': {
                'current': self.monitor.current_utilization,
                'average_1h': self.monitor.get_average_utilization(hours=1),
                'peak_24h': self.monitor.get_peak_utilization(hours=24)
            },
            'token_usage': {
                'last_hour': self.token_tracker.get_usage_last_hour(),
                'last_day': self.token_tracker.get_usage_last_day(),
                'by_component': self.token_tracker.get_component_breakdown()
            },
            'memory_stats': {
                'hot_memory_size': len(self.memory.hot_memory),
                'facts_stored': len(self.memory.facts),
                'conversations_tracked': len(self.memory.exchanges_by_conversation)
            }
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    """Example usage of the Context Engineering Loop."""
    
    # Initialize loop
    config = LoopConfig(
        model_context_limit=256000,
        reserve_tokens=16000,
        compression_threshold=0.85
    )
    
    loop = ContextEngineeringLoop(config)
    await loop.start()
    
    # Simulate conversation
    conversation_id = "conv_123"
    system_prompt = "You are a helpful AI assistant."
    
    # First exchange
    user_msg = "Hello, can you help me with a programming task?"
    
    context = await loop.build_optimized_context(
        system_prompt=system_prompt,
        user_message=user_msg,
        conversation_id=conversation_id
    )
    
    print(f"Built context with {context.total_tokens} tokens")
    print(f"Was compressed: {context.was_compressed}")
    print(f"Elements included: {context.elements_included}")
    
    # Record exchange
    assistant_response = "Hello! I'd be happy to help you with programming. What do you need assistance with?"
    await loop.record_exchange(conversation_id, user_msg, assistant_response)
    
    # Get metrics
    metrics = loop.get_metrics()
    print(f"\nMetrics: {json.dumps(metrics, indent=2, default=str)}")
    
    await loop.stop()


if __name__ == "__main__":
    asyncio.run(main())
