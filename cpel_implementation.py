"""
Context Prompt Engineering Loop - Core Implementation
Windows 10 OpenClaw-Inspired AI Agent System

This module provides the core implementation for the Context Prompt Engineering Loop,
an autonomous prompt optimization system for GPT-5.2 based AI agents.
"""

import asyncio
import json
import hashlib
import uuid
import re
import difflib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class TaskType(Enum):
    """Enumeration of task types for the AI agent."""
    SYSTEM = "system"
    BROWSER = "browser"
    EMAIL = "email"
    VOICE = "voice"
    SMS = "sms"
    FILE = "file"
    SCHEDULE = "schedule"
    RESEARCH = "research"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"


@dataclass
class ExecutionContext:
    """Represents the execution context for prompt optimization."""
    task_type: TaskType = TaskType.CONVERSATION
    user_intent: Optional[str] = None
    conversation_depth: int = 0
    previous_outcomes: List[Dict] = field(default_factory=list)
    system_state: Dict = field(default_factory=dict)
    time_context: Dict = field(default_factory=dict)
    emotional_tone: Optional[str] = None
    complexity_score: float = 0.5
    urgency_level: float = 0.0
    domain: Optional[str] = None
    user_preferences: Dict = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class PromptMetrics:
    """Comprehensive metrics for a single prompt execution."""
    # Response Quality Metrics
    response_accuracy: float = 0.0
    response_relevance: float = 0.0
    response_completeness: float = 0.0
    response_coherence: float = 0.0
    helpfulness: float = 0.0
    factual_correctness: float = 0.0
    
    # Efficiency Metrics
    token_count_input: int = 0
    token_count_output: int = 0
    token_efficiency: float = 0.0
    latency_ms: int = 0
    cost_per_request: float = 0.0
    
    # User Satisfaction Metrics
    user_rating: Optional[float] = None
    implicit_satisfaction: float = 0.0
    retry_count: int = 0
    correction_needed: bool = False
    follow_up_question_rate: float = 0.0
    
    # Context Metrics
    context_match_score: float = 0.0
    variable_substitution_success: float = 1.0
    template_render_success: bool = True
    
    # Temporal Metrics
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    prompt_id: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics over a time window."""
    prompt_id: str = ""
    time_window: str = ""
    stats: Dict[str, Dict] = field(default_factory=dict)
    anomalies: List[Dict] = field(default_factory=list)
    overall_score: float = 0.0
    sample_count: int = 0


@dataclass
class OptimizableTemplate:
    """Template with optimization metadata and versioning."""
    template_id: str = ""
    name: str = ""
    category: str = ""  # 'system', 'task', 'conversation', 'tool'
    template_str: str = ""
    variables: List[str] = field(default_factory=list)
    context_rules: Dict = field(default_factory=dict)
    performance_baseline: float = 0.0
    current_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    version: str = "1.0.0"
    optimization_history: List[Dict] = field(default_factory=list)


@dataclass
class PromptVariant:
    """A/B test variant of a prompt template."""
    variant_id: str = ""
    parent_template_id: str = ""
    template_str: str = ""
    modifications: Dict[str, str] = field(default_factory=dict)
    traffic_percentage: float = 0.0
    performance_score: float = 0.0
    sample_count: int = 0


@dataclass
class FewShotExample:
    """Single few-shot example for in-context learning."""
    example_id: str = ""
    input_text: str = ""
    output_text: str = ""
    task_type: str = ""
    difficulty: float = 0.5
    success_rate: float = 1.0
    usage_count: int = 0
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PromptVersion:
    """Represents a single version of a prompt."""
    version_id: str = ""
    prompt_id: str = ""
    version_number: str = ""
    template: OptimizableTemplate = field(default_factory=OptimizableTemplate)
    parent_version: Optional[str] = None
    author: str = "system"
    commit_message: str = ""
    performance_snapshot: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization process."""
    prompt: str = ""
    template_id: str = ""
    variant_id: Optional[str] = None
    context_used: ExecutionContext = field(default_factory=ExecutionContext)
    examples_used: List[FewShotExample] = field(default_factory=list)
    assembly_metadata: Dict = field(default_factory=dict)


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

class PromptRegistry:
    """
    Central registry for prompt templates with full metadata tracking.
    """
    
    def __init__(self):
        self._templates: Dict[str, OptimizableTemplate] = {}
        self._versions: Dict[str, List[str]] = defaultdict(list)
        self._metadata: Dict[str, Dict] = defaultdict(dict)
        self._task_mappings: Dict[str, List[str]] = defaultdict(list)
        
    async def register_template(
        self,
        template: OptimizableTemplate
    ) -> str:
        """Register a new prompt template."""
        if not template.template_id:
            template.template_id = str(uuid.uuid4())
            
        self._templates[template.template_id] = template
        self._versions[template.template_id].append(template.version)
        
        # Map to task type
        self._task_mappings[template.category].append(template.template_id)
        
        logger.info(f"Registered template: {template.name} ({template.template_id})")
        return template.template_id
        
    async def get_template(self, template_id: str) -> Optional[OptimizableTemplate]:
        """Retrieve a template by ID."""
        return self._templates.get(template_id)
        
    async def get_template_for_task(
        self,
        task_type: Union[str, TaskType],
        context: ExecutionContext
    ) -> Optional[OptimizableTemplate]:
        """Get the best template for a given task type and context."""
        category = task_type.value if isinstance(task_type, TaskType) else task_type
        
        template_ids = self._task_mappings.get(category, [])
        if not template_ids:
            return None
            
        # Score each template for the context
        scored_templates = []
        for tid in template_ids:
            template = self._templates.get(tid)
            if template:
                score = self._score_template_for_context(template, context)
                scored_templates.append((template, score))
                
        # Return highest scoring template
        if scored_templates:
            scored_templates.sort(key=lambda x: x[1], reverse=True)
            return scored_templates[0][0]
            
        return None
        
    def _score_template_for_context(
        self,
        template: OptimizableTemplate,
        context: ExecutionContext
    ) -> float:
        """Score how well a template matches the context."""
        score = template.current_score
        
        # Boost score based on context rules
        rules = template.context_rules
        
        if 'min_complexity' in rules and context.complexity_score >= rules['min_complexity']:
            score += 0.1
        if 'max_complexity' in rules and context.complexity_score <= rules['max_complexity']:
            score += 0.1
        if 'domain' in rules and context.domain == rules['domain']:
            score += 0.15
            
        return score
        
    async def update_template_score(
        self,
        template_id: str,
        new_score: float
    ):
        """Update the performance score of a template."""
        if template_id in self._templates:
            self._templates[template_id].current_score = new_score
            self._templates[template_id].updated_at = datetime.utcnow()


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """
    Comprehensive performance tracking for prompt effectiveness.
    """
    
    def __init__(self):
        self._metrics_store: List[PromptMetrics] = []
        self._aggregation_window = 3600  # 1 hour default
        
    async def collect_execution_metrics(
        self,
        prompt_id: str,
        rendered_prompt: str,
        context: ExecutionContext,
        llm_response: str,
        execution_time_ms: int,
        llm_interface: Any = None
    ) -> PromptMetrics:
        """
        Collect comprehensive metrics for a single prompt execution.
        """
        metrics = PromptMetrics()
        metrics.prompt_id = prompt_id
        metrics.session_id = context.session_id
        metrics.latency_ms = execution_time_ms
        
        # Calculate token counts (simplified estimation)
        metrics.token_count_input = len(rendered_prompt.split()) // 0.75
        metrics.token_count_output = len(llm_response.split()) // 0.75
        metrics.token_efficiency = (
            metrics.token_count_output / max(metrics.token_count_input, 1)
        )
        
        # Evaluate quality if LLM interface provided
        if llm_interface:
            quality_scores = await self._evaluate_quality(
                rendered_prompt, llm_response, context, llm_interface
            )
            metrics.response_accuracy = quality_scores.get('accuracy', 0.0)
            metrics.response_relevance = quality_scores.get('relevance', 0.0)
            metrics.response_completeness = quality_scores.get('completeness', 0.0)
            metrics.response_coherence = quality_scores.get('coherence', 0.0)
            metrics.helpfulness = quality_scores.get('helpfulness', 0.0)
            
        # Calculate context match score
        metrics.context_match_score = self._calculate_context_match(
            rendered_prompt, context
        )
        
        # Store metrics
        self._metrics_store.append(metrics)
        
        return metrics
        
    async def _evaluate_quality(
        self,
        prompt: str,
        response: str,
        context: ExecutionContext,
        llm_interface: Any
    ) -> Dict[str, float]:
        """Use LLM to evaluate response quality across dimensions."""
        evaluation_prompt = f"""
        Evaluate the following AI response based on the original prompt and context.
        
        ORIGINAL PROMPT (truncated):
        {prompt[:1500]}
        
        AI RESPONSE (truncated):
        {response[:1500]}
        
        CONTEXT:
        Task Type: {context.task_type.value if context.task_type else 'unknown'}
        User Intent: {context.user_intent or 'unknown'}
        Complexity: {context.complexity_score:.2f}
        
        Rate the response on a scale of 0.0 to 1.0 for:
        1. accuracy - factual correctness
        2. relevance - addresses the specific query
        3. completeness - fully answers the question
        4. coherence - logical flow and clarity
        5. helpfulness - practical utility
        
        Return ONLY a JSON object with these scores.
        """
        
        try:
            evaluation = await llm_interface.generate(evaluation_prompt)
            return json.loads(evaluation)
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                'accuracy': 0.5,
                'relevance': 0.5,
                'completeness': 0.5,
                'coherence': 0.5,
                'helpfulness': 0.5
            }
            
    def _calculate_context_match(
        self,
        rendered_prompt: str,
        context: ExecutionContext
    ) -> float:
        """Calculate how well the prompt matches the context."""
        score = 1.0
        
        # Check if context indicators are present
        if context.complexity_score > 0.7 and 'complex' not in rendered_prompt.lower():
            score -= 0.1
        if context.urgency_level > 0.7 and 'urgent' not in rendered_prompt.lower():
            score -= 0.1
            
        return max(0.0, score)
        
    async def aggregate_performance(
        self,
        prompt_id: str,
        time_window: str,
        dimensions: List[str]
    ) -> AggregatedMetrics:
        """Aggregate metrics for analysis and optimization decisions."""
        # Parse time window
        hours = self._parse_time_window(time_window)
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter metrics
        raw_metrics = [
            m for m in self._metrics_store
            if m.prompt_id == prompt_id and m.timestamp >= cutoff
        ]
        
        aggregated = AggregatedMetrics(
            prompt_id=prompt_id,
            time_window=time_window,
            sample_count=len(raw_metrics)
        )
        
        if not raw_metrics:
            return aggregated
            
        # Calculate statistics for each dimension
        for dimension in dimensions:
            values = []
            for m in raw_metrics:
                val = getattr(m, dimension, None)
                if val is not None:
                    values.append(val)
                    
            if values:
                aggregated.stats[dimension] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p95': float(np.percentile(values, 95)),
                    'sample_count': len(values)
                }
                
        # Calculate overall score
        aggregated.overall_score = self._calculate_overall_score(aggregated.stats)
        
        return aggregated
        
    def _parse_time_window(self, window: str) -> int:
        """Parse time window string to hours."""
        mapping = {
            '1h': 1, '6h': 6, '24h': 24,
            '7d': 168, '30d': 720
        }
        return mapping.get(window, 24)
        
    def _calculate_overall_score(self, stats: Dict) -> float:
        """Calculate weighted overall performance score."""
        if not stats:
            return 0.0
            
        weights = {
            'response_accuracy': 0.2,
            'response_relevance': 0.15,
            'response_completeness': 0.15,
            'response_coherence': 0.1,
            'helpfulness': 0.15,
            'token_efficiency': 0.15,
            'context_match_score': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in stats:
                total_score += stats[metric]['mean'] * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0.0


# =============================================================================
# CONTEXT ENGINE
# =============================================================================

class ContextEngine:
    """
    Analyzes and manages execution context for prompt optimization.
    """
    
    def __init__(self):
        self._context_history: List[ExecutionContext] = []
        
    async def detect_context(
        self,
        user_input: str,
        conversation_history: List[Dict],
        system_state: Dict
    ) -> ExecutionContext:
        """
        Comprehensive context detection for prompt adjustment.
        """
        context = ExecutionContext()
        
        # Detect task type
        context.task_type = self._detect_task_type(user_input)
        
        # Detect user intent
        context.user_intent = self._detect_intent(user_input)
        
        # Calculate complexity
        context.complexity_score = self._calculate_complexity(user_input)
        
        # Detect urgency
        context.urgency_level = self._detect_urgency(user_input)
        
        # Detect domain
        context.domain = self._detect_domain(user_input)
        
        # Detect emotional tone
        context.emotional_tone = self._detect_emotional_tone(user_input)
        
        # Calculate conversation depth
        context.conversation_depth = len(conversation_history)
        
        # Extract previous outcomes
        context.previous_outcomes = self._extract_outcomes(conversation_history)
        
        # Add system state
        context.system_state = system_state
        
        # Set session ID
        context.session_id = system_state.get('session_id')
        
        self._context_history.append(context)
        
        return context
        
    def _detect_task_type(self, user_input: str) -> TaskType:
        """Detect the type of task from user input."""
        input_lower = user_input.lower()
        
        # Browser-related
        if any(kw in input_lower for kw in ['browse', 'search', 'open', 'website', 'url', 'navigate']):
            return TaskType.BROWSER
            
        # Email-related
        if any(kw in input_lower for kw in ['email', 'gmail', 'send mail', 'inbox', 'message']):
            return TaskType.EMAIL
            
        # Voice-related
        if any(kw in input_lower for kw in ['call', 'voice', 'speak', 'phone', 'twilio']):
            return TaskType.VOICE
            
        # SMS-related
        if any(kw in input_lower for kw in ['sms', 'text', 'message']):
            return TaskType.SMS
            
        # File-related
        if any(kw in input_lower for kw in ['file', 'folder', 'directory', 'save', 'open file']):
            return TaskType.FILE
            
        # Schedule-related
        if any(kw in input_lower for kw in ['schedule', 'remind', 'calendar', 'alarm', 'cron']):
            return TaskType.SCHEDULE
            
        # Research-related
        if any(kw in input_lower for kw in ['research', 'find', 'lookup', 'investigate', 'analyze']):
            return TaskType.RESEARCH
            
        # Tool use
        if any(kw in input_lower for kw in ['use', 'run', 'execute', 'command']):
            return TaskType.TOOL_USE
            
        return TaskType.CONVERSATION
        
    def _detect_intent(self, user_input: str) -> str:
        """Detect user intent from input."""
        input_lower = user_input.lower()
        
        if any(kw in input_lower for kw in ['how', 'what is', 'explain']):
            return 'information_seeking'
        elif any(kw in input_lower for kw in ['do', 'perform', 'execute', 'run']):
            return 'action_request'
        elif any(kw in input_lower for kw in ['help', 'assist', 'support']):
            return 'help_request'
        elif any(kw in input_lower for kw in ['create', 'make', 'build', 'generate']):
            return 'creation_request'
        else:
            return 'general'
            
    def _calculate_complexity(self, user_input: str) -> float:
        """Calculate complexity score of the request."""
        # Factors contributing to complexity
        word_count = len(user_input.split())
        sentence_count = len(re.split(r'[.!?]+', user_input))
        
        # Count complex indicators
        complex_indicators = ['and', 'also', 'additionally', 'furthermore', 'moreover',
                             'however', 'although', 'whereas', 'meanwhile', 'then']
        indicator_count = sum(1 for ind in complex_indicators if ind in user_input.lower())
        
        # Calculate score
        score = min(1.0, (
            (word_count / 100) * 0.3 +
            (sentence_count / 10) * 0.3 +
            (indicator_count / 5) * 0.4
        ))
        
        return score
        
    def _detect_urgency(self, user_input: str) -> float:
        """Detect urgency level from user input."""
        urgency_keywords = {
            'high': ['urgent', 'asap', 'immediately', 'now', 'quick', 'hurry', 'emergency'],
            'medium': ['soon', 'today', 'promptly', 'timely'],
            'low': ['whenever', 'sometime', 'later', 'eventually']
        }
        
        input_lower = user_input.lower()
        
        if any(kw in input_lower for kw in urgency_keywords['high']):
            return 0.9
        elif any(kw in input_lower for kw in urgency_keywords['medium']):
            return 0.6
        elif any(kw in input_lower for kw in urgency_keywords['low']):
            return 0.2

        # LLM fallback when no keywords match
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            resp = client.generate(
                f"Rate urgency 0.0-1.0 for: {user_input[:300]}. Reply ONLY a number."
            )
            return max(0.0, min(1.0, float(resp.strip())))
        except (ImportError, ValueError, RuntimeError, EnvironmentError):
            return 0.3
        
    def _detect_domain(self, user_input: str) -> Optional[str]:
        """Detect the domain of the request."""
        domains = {
            'technical': ['code', 'program', 'script', 'api', 'database', 'server',
                          'software', 'bug', 'deploy', 'programming', 'function'],
            'business': ['meeting', 'report', 'client', 'project', 'deadline',
                         'revenue', 'profit', 'market', 'budget', 'stakeholder'],
            'creative': ['write', 'story', 'design', 'create', 'art',
                         'compose', 'illustrate', 'brainstorm'],
            'personal': ['my', 'i need', 'help me', 'for me', 'remind me'],
            'research': ['study', 'paper', 'article', 'research', 'analysis',
                         'hypothesis', 'experiment', 'data'],
            'science': ['biology', 'physics', 'chemistry', 'math', 'theorem'],
        }

        input_lower = user_input.lower()

        # Count keyword matches per domain, return best match
        scores = {}
        for domain, keywords in domains.items():
            count = sum(1 for kw in keywords if kw in input_lower)
            if count > 0:
                scores[domain] = count

        if scores:
            return max(scores, key=scores.get)

        # LLM fallback
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            resp = client.generate(
                f"Classify this request into one domain: technical, business, "
                f"creative, personal, research, science. Reply with ONLY the "
                f"domain name.\nRequest: {user_input[:300]}"
            )
            detected = resp.strip().lower()
            if detected in domains:
                return detected
        except (ImportError, ValueError, RuntimeError, EnvironmentError):
            pass

        return None
        
    def _detect_emotional_tone(self, user_input: str) -> Optional[str]:
        """Detect emotional tone from user input."""
        tones = {
            'frustrated': ['frustrated', 'annoyed', 'angry', 'upset', 'tired of', 'sick of'],
            'excited': ['excited', 'great', 'awesome', 'amazing', 'love', 'fantastic'],
            'anxious': ['worried', 'concerned', 'nervous', 'anxious', 'stressed'],
            'curious': ['curious', 'wondering', 'interested', 'how does', 'why is'],
            'neutral': []
        }
        
        input_lower = user_input.lower()
        
        for tone, keywords in tones.items():
            if any(kw in input_lower for kw in keywords):
                return tone
                
        return 'neutral'
        
    def _extract_outcomes(self, conversation_history: List[Dict]) -> List[Dict]:
        """Extract outcomes from conversation history."""
        outcomes = []
        
        for entry in conversation_history[-5:]:  # Last 5 entries
            if 'outcome' in entry:
                outcomes.append(entry['outcome'])
                
        return outcomes


# =============================================================================
# CONTEXT-BASED MODIFIER
# =============================================================================

class ContextBasedModifier:
    """
    Modifies prompts based on detected context.
    """
    
    def __init__(self):
        self._modification_rules = self._load_modification_rules()
        
    def _load_modification_rules(self) -> Dict:
        """Load context-based modification rules."""
        return {
            'complexity': {
                'high_threshold': 0.8,
                'low_threshold': 0.3
            },
            'urgency': {
                'high_threshold': 0.7
            }
        }
        
    def apply_context_modifications(
        self,
        base_prompt: str,
        context: ExecutionContext
    ) -> str:
        """Apply context-aware modifications to base prompt."""
        modified_prompt = base_prompt
        
        # Apply complexity-based modifications
        if context.complexity_score > self._modification_rules['complexity']['high_threshold']:
            modified_prompt = self._add_complexity_instructions(modified_prompt)
        elif context.complexity_score < self._modification_rules['complexity']['low_threshold']:
            modified_prompt = self._simplify_instructions(modified_prompt)
            
        # Apply urgency-based modifications
        if context.urgency_level > self._modification_rules['urgency']['high_threshold']:
            modified_prompt = self._add_urgency_instructions(modified_prompt)
            
        # Apply domain-specific modifications
        if context.domain:
            modified_prompt = self._add_domain_context(modified_prompt, context.domain)
            
        # Apply emotional tone adjustments
        if context.emotional_tone == 'frustrated':
            modified_prompt = self._add_empathy_instructions(modified_prompt)
        elif context.emotional_tone == 'excited':
            modified_prompt = self._match_enthusiasm(modified_prompt)
            
        # Add conversation context if deep in conversation
        if context.conversation_depth > 5:
            modified_prompt = self._add_conversation_continuity(modified_prompt)
            
        return modified_prompt
        
    def _add_complexity_instructions(self, prompt: str) -> str:
        """Add instructions for complex requests."""
        complexity_addition = """

[COMPLEXITY GUIDANCE]
This is a complex request. Please:
1. Break down your response into clear, numbered steps
2. Provide detailed explanations for each step
3. Consider edge cases and alternatives
4. Use examples to illustrate complex concepts
5. Summarize key points at the end
"""
        return prompt + complexity_addition
        
    def _simplify_instructions(self, prompt: str) -> str:
        """Simplify instructions for straightforward requests."""
        simplification = """

[SIMPLICITY GUIDANCE]
This is a straightforward request. Please:
1. Provide a concise, direct answer
2. Avoid unnecessary elaboration
3. Focus on the essential information
"""
        return prompt + simplification
        
    def _add_urgency_instructions(self, prompt: str) -> str:
        """Add instructions for urgent requests."""
        urgency_addition = """

[URGENCY GUIDANCE]
This request is time-sensitive. Please:
1. Prioritize speed while maintaining accuracy
2. Provide the most critical information first
3. Use concise, direct language
4. Mark any information that needs verification
"""
        return prompt + urgency_addition
        
    def _add_domain_context(self, prompt: str, domain: str) -> str:
        """Add domain-specific context."""
        domain_contexts = {
            'technical': "\n[DOMAIN: TECHNICAL] Provide technically accurate, detailed responses.",
            'business': "\n[DOMAIN: BUSINESS] Focus on professional, actionable insights.",
            'creative': "\n[DOMAIN: CREATIVE] Be imaginative and explore multiple possibilities.",
            'research': "\n[DOMAIN: RESEARCH] Provide well-sourced, analytical responses."
        }
        
        return prompt + domain_contexts.get(domain, "")
        
    def _add_empathy_instructions(self, prompt: str) -> str:
        """Add empathy for frustrated users."""
        empathy = """

[EMPATHY GUIDANCE]
The user may be frustrated. Please:
1. Acknowledge their frustration
2. Be patient and supportive
3. Provide clear, actionable solutions
4. Reassure them that you'll help resolve the issue
"""
        return prompt + empathy
        
    def _match_enthusiasm(self, prompt: str) -> str:
        """Match user's enthusiastic tone."""
        enthusiasm = """

[TONE GUIDANCE]
The user is enthusiastic! Please:
1. Match their positive energy
2. Be encouraging and supportive
3. Celebrate progress and successes
"""
        return prompt + enthusiasm
        
    def _add_conversation_continuity(self, prompt: str) -> str:
        """Add continuity for ongoing conversations."""
        continuity = """

[CONTINUITY GUIDANCE]
This is part of an ongoing conversation. Please:
1. Reference previous context when relevant
2. Build upon earlier discussion points
3. Maintain consistency with prior responses
"""
        return prompt + continuity


# =============================================================================
# FEW-SHOT EXAMPLE SELECTOR
# =============================================================================

class FewShotExampleSelector:
    """
    Selects optimal few-shot examples using multiple strategies.
    """
    
    def __init__(self):
        self._examples: List[FewShotExample] = []
        self._embeddings: Dict[str, List[float]] = {}
        
    async def add_example(
        self,
        input_text: str,
        output_text: str,
        task_type: str,
        embedding: Optional[List[float]] = None
    ) -> FewShotExample:
        """Add a new example to the database."""
        example = FewShotExample(
            example_id=str(uuid.uuid4()),
            input_text=input_text,
            output_text=output_text,
            task_type=task_type,
            embedding=embedding
        )
        
        self._examples.append(example)
        
        if embedding:
            self._embeddings[example.example_id] = embedding
            
        return example
        
    async def select_examples(
        self,
        query: str,
        task_type: str,
        n_examples: int = 3,
        method: str = 'similarity'
    ) -> List[FewShotExample]:
        """
        Select few-shot examples using specified method.
        
        Methods:
        - 'similarity': Semantic similarity to query
        - 'diversity': Maximize diversity among examples
        - 'success': Prioritize high-success-rate examples
        - 'hybrid': Combine multiple criteria
        """
        # Filter by task type
        candidates = [e for e in self._examples if e.task_type == task_type]
        
        if not candidates:
            return []
            
        if method == 'similarity':
            return await self._select_by_similarity(candidates, query, n_examples)
        elif method == 'diversity':
            return await self._select_by_diversity(candidates, n_examples)
        elif method == 'success':
            return await self._select_by_success(candidates, n_examples)
        elif method == 'hybrid':
            return await self._select_hybrid(candidates, query, n_examples)
        else:
            return candidates[:n_examples]
            
    async def _select_by_similarity(
        self,
        candidates: List[FewShotExample],
        query: str,
        n: int
    ) -> List[FewShotExample]:
        """Select examples by semantic similarity to query."""
        # Simple keyword-based similarity (in production, use embeddings)
        scored = []
        query_words = set(query.lower().split())
        
        for example in candidates:
            example_words = set(example.input_text.lower().split())
            overlap = len(query_words & example_words)
            similarity = overlap / max(len(query_words), len(example_words), 1)
            scored.append((example, similarity))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:n]]
        
    async def _select_by_diversity(
        self,
        candidates: List[FewShotExample],
        n: int
    ) -> List[FewShotExample]:
        """Select diverse examples using greedy selection."""
        if len(candidates) <= n:
            return candidates
            
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        while len(selected) < n and remaining:
            # Find example most different from already selected
            max_min_distance = -1
            best_candidate = None
            
            for candidate in remaining:
                min_distance = float('inf')
                for sel in selected:
                    dist = self._example_distance(candidate, sel)
                    min_distance = min(min_distance, dist)
                    
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                
        return selected
        
    async def _select_by_success(
        self,
        candidates: List[FewShotExample],
        n: int
    ) -> List[FewShotExample]:
        """Select examples with highest success rates."""
        scored = [(e, e.success_rate) for e in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:n]]
        
    async def _select_hybrid(
        self,
        candidates: List[FewShotExample],
        query: str,
        n: int
    ) -> List[FewShotExample]:
        """Hybrid selection combining multiple criteria."""
        # Score each example on multiple dimensions
        scored = []
        query_words = set(query.lower().split())
        
        for example in candidates:
            # Similarity score
            example_words = set(example.input_text.lower().split())
            overlap = len(query_words & example_words)
            similarity = overlap / max(len(query_words), len(example_words), 1)
            
            # Success score
            success = example.success_rate
            
            # Usage balance (prefer less-used examples)
            usage_balance = 1.0 / (1 + example.usage_count * 0.1)
            
            # Combined score
            combined = similarity * 0.4 + success * 0.4 + usage_balance * 0.2
            scored.append((example, combined))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:n]]
        
    def _example_distance(
        self,
        ex1: FewShotExample,
        ex2: FewShotExample
    ) -> float:
        """Calculate distance between two examples."""
        # Simple Jaccard distance on words
        words1 = set(ex1.input_text.lower().split())
        words2 = set(ex2.input_text.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return 1.0 - (intersection / union if union > 0 else 0)


# =============================================================================
# DYNAMIC PROMPT ASSEMBLER
# =============================================================================

class DynamicPromptAssembler:
    """
    Dynamically assembles prompts from modular components.
    """
    
    def __init__(self):
        self._components: Dict[str, Dict] = {}
        
    async def assemble_prompt(
        self,
        base_template: str,
        context: ExecutionContext,
        examples: List[FewShotExample],
        variables: Dict[str, Any],
        max_tokens: int = 4000
    ) -> str:
        """
        Dynamically assemble a complete prompt.
        """
        parts = []
        
        # Add system instruction if present
        if 'system_instruction' in variables:
            parts.append(variables['system_instruction'])
            
        # Add task description
        parts.append(base_template)
        
        # Add context-specific guidance
        context_guidance = self._generate_context_guidance(context)
        if context_guidance:
            parts.append(context_guidance)
            
        # Add few-shot examples
        if examples:
            example_section = self._format_examples(examples)
            parts.append(example_section)
            
        # Add output format if specified
        if 'output_format' in variables:
            parts.append(variables['output_format'])
            
        # Add constraints
        if 'constraints' in variables:
            parts.append(variables['constraints'])
            
        # Combine all parts
        assembled = "\n\n".join(parts)
        
        # Resolve any remaining variables
        assembled = self._resolve_variables(assembled, variables)
        
        # Truncate if exceeds max tokens (rough estimate)
        estimated_tokens = len(assembled.split()) // 0.75
        if estimated_tokens > max_tokens:
            assembled = self._truncate_to_tokens(assembled, max_tokens)
            
        return assembled
        
    def _generate_context_guidance(self, context: ExecutionContext) -> str:
        """Generate guidance based on context."""
        guidance_parts = []
        
        if context.complexity_score > 0.8:
            guidance_parts.append("Think step-by-step and explain your reasoning.")
            
        if context.urgency_level > 0.7:
            guidance_parts.append("Provide the most important information first.")
            
        if context.emotional_tone == 'frustrated':
            guidance_parts.append("Be patient and understanding in your response.")
            
        if guidance_parts:
            return "\n".join(["[GUIDANCE]"] + guidance_parts)
            
        return ""
        
    def _format_examples(self, examples: List[FewShotExample]) -> str:
        """Format few-shot examples for inclusion in prompt."""
        lines = ["[EXAMPLES]"]
        
        for i, example in enumerate(examples, 1):
            lines.append(f"\nExample {i}:")
            lines.append(f"Input: {example.input_text}")
            lines.append(f"Output: {example.output_text}")
            
        return "\n".join(lines)
        
    def _resolve_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Resolve template variables."""
        result = template
        
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
            
        return result
        
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        words = text.split()
        # Rough estimate: 0.75 words per token
        max_words = int(max_tokens * 0.75)
        
        if len(words) <= max_words:
            return text
            
        truncated = " ".join(words[:max_words])
        return truncated + "\n\n[Content truncated due to length]"


# =============================================================================
# A/B TEST FRAMEWORK
# =============================================================================

class ABTestFramework:
    """
    Comprehensive A/B testing framework for prompt optimization.
    """
    
    def __init__(self):
        self._active_tests: Dict[str, Dict] = {}
        self._variant_assignments: Dict[str, str] = {}
        
    async def create_test(
        self,
        test_name: str,
        control_template: OptimizableTemplate,
        variants: List[PromptVariant],
        success_metrics: List[str],
        min_samples: int = 100,
        confidence_level: float = 0.95
    ) -> str:
        """
        Create a new A/B test.
        """
        test_id = str(uuid.uuid4())
        
        # Calculate traffic distribution
        num_variants = len(variants) + 1  # +1 for control
        base_percentage = 1.0 / num_variants
        
        # Set traffic percentages
        for variant in variants:
            variant.traffic_percentage = base_percentage
            
        test_config = {
            'test_id': test_id,
            'name': test_name,
            'control': control_template,
            'variants': variants,
            'success_metrics': success_metrics,
            'min_samples': min_samples,
            'confidence_level': confidence_level,
            'status': 'running',
            'start_time': datetime.utcnow(),
            'results': defaultdict(lambda: defaultdict(list))
        }
        
        self._active_tests[test_id] = test_config
        
        logger.info(f"Created A/B test: {test_name} ({test_id})")
        return test_id
        
    async def get_variant_for_request(
        self,
        test_id: str,
        user_id: Optional[str] = None
    ) -> Optional[PromptVariant]:
        """
        Get the appropriate variant for a request.
        """
        test = self._active_tests.get(test_id)
        if not test or test['status'] != 'running':
            return None
            
        # Check for sticky assignment
        assignment_key = f"{test_id}:{user_id or 'anonymous'}"
        if assignment_key in self._variant_assignments:
            variant_id = self._variant_assignments[assignment_key]
            for variant in test['variants']:
                if variant.variant_id == variant_id:
                    return variant
                    
        # New assignment using consistent hashing
        all_options = [test['control']] + test['variants']
        hash_input = f"{test_id}:{user_id or str(uuid.uuid4())}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Weighted selection
        cumulative = 0
        for option in all_options:
            weight = option.traffic_percentage if hasattr(option, 'traffic_percentage') else 1.0 / len(all_options)
            cumulative += weight
            if (hash_value % 10000) < (cumulative * 10000):
                if isinstance(option, PromptVariant):
                    self._variant_assignments[assignment_key] = option.variant_id
                    return option
                else:
                    # Control variant
                    control_variant = PromptVariant(
                        variant_id=f"{test['control'].template_id}_control",
                        parent_template_id=test['control'].template_id,
                        template_str=test['control'].template_str,
                        traffic_percentage=1.0 / len(all_options)
                    )
                    self._variant_assignments[assignment_key] = control_variant.variant_id
                    return control_variant
                    
        return None
        
    async def record_result(
        self,
        test_id: str,
        variant_id: str,
        metrics: Dict[str, float]
    ):
        """Record a result for a variant."""
        test = self._active_tests.get(test_id)
        if not test:
            return
            
        for metric, value in metrics.items():
            if metric in test['success_metrics']:
                test['results'][variant_id][metric].append(value)
                
        # Check if test should conclude
        await self._check_test_completion(test_id)
        
    async def _check_test_completion(self, test_id: str):
        """Check if A/B test has enough data to conclude."""
        test = self._active_tests.get(test_id)
        if not test:
            return
            
        # Check sample sizes
        for variant_id, metrics in test['results'].items():
            for metric, values in metrics.items():
                if len(values) < test['min_samples']:
                    return
                    
        # All variants have enough samples - analyze results
        await self._analyze_test_results(test_id)
        
    async def _analyze_test_results(self, test_id: str):
        """Analyze A/B test results and determine winner."""
        test = self._active_tests.get(test_id)
        if not test:
            return
            
        analysis = {}
        
        for metric in test['success_metrics']:
            control_values = test['results'].get('control', {}).get(metric, [])
            
            for variant in test['variants']:
                variant_values = test['results'].get(variant.variant_id, {}).get(metric, [])
                
                if control_values and variant_values:
                    # Calculate statistics
                    control_mean = np.mean(control_values)
                    variant_mean = np.mean(variant_values)
                    
                    relative_change = (variant_mean - control_mean) / control_mean if control_mean != 0 else 0
                    
                    # Simple t-test
                    from statistics import stdev
                    try:
                        pooled_se = ((stdev(control_values) ** 2 / len(control_values)) + 
                                    (stdev(variant_values) ** 2 / len(variant_values))) ** 0.5
                        t_stat = (variant_mean - control_mean) / pooled_se if pooled_se > 0 else 0
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"A/B test statistics calculation failed: {e}")
                        t_stat = 0
                        
                    analysis[f"{variant.variant_id}_{metric}"] = {
                        'control_mean': control_mean,
                        'variant_mean': variant_mean,
                        'relative_change': relative_change,
                        't_statistic': t_stat
                    }
                    
        logger.info(f"A/B test {test_id} analysis complete")
        test['analysis'] = analysis
        test['status'] = 'completed'


# =============================================================================
# VERSION CONTROL
# =============================================================================

class PromptVersionControl:
    """
    Git-like version control for prompts.
    """
    
    def __init__(self):
        self._versions: Dict[str, List[PromptVersion]] = defaultdict(list)
        self._current_versions: Dict[str, str] = {}
        
    async def commit_version(
        self,
        prompt_id: str,
        template: OptimizableTemplate,
        commit_message: str,
        author: str = 'system',
        tags: List[str] = None
    ) -> PromptVersion:
        """
        Commit a new version of a prompt.
        """
        # Get current version
        current_version_id = self._current_versions.get(prompt_id)
        
        # Calculate new version number
        existing_versions = self._versions.get(prompt_id, [])
        if existing_versions:
            last_version = existing_versions[-1].version_number
            new_version = self._increment_version(last_version)
            parent = current_version_id
        else:
            new_version = "1.0.0"
            parent = None
            
        # Create version
        version = PromptVersion(
            version_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            version_number=new_version,
            template=template,
            parent_version=parent,
            author=author,
            commit_message=commit_message,
            tags=tags or [],
            timestamp=datetime.utcnow()
        )
        
        # Store version
        self._versions[prompt_id].append(version)
        self._current_versions[prompt_id] = version.version_id
        
        logger.info(f"Committed version {new_version} for prompt {prompt_id}")
        return version
        
    def _increment_version(self, current: str) -> str:
        """Increment semantic version number."""
        try:
            major, minor, patch = map(int, current.split('.'))
            patch += 1
            if patch >= 10:
                patch = 0
                minor += 1
            if minor >= 10:
                minor = 0
                major += 1
            return f"{major}.{minor}.{patch}"
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Version increment failed for '{current}': {e}")
            return "1.0.0"
            
    async def get_version(
        self,
        prompt_id: str,
        version_number: str
    ) -> Optional[PromptVersion]:
        """Get a specific version of a prompt."""
        versions = self._versions.get(prompt_id, [])
        for version in versions:
            if version.version_number == version_number:
                return version
        return None
        
    async def get_current_version(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get the current version of a prompt."""
        current_id = self._current_versions.get(prompt_id)
        if not current_id:
            return None
            
        versions = self._versions.get(prompt_id, [])
        for version in versions:
            if version.version_id == current_id:
                return version
        return None
        
    async def rollback(
        self,
        prompt_id: str,
        target_version: str
    ) -> bool:
        """
        Rollback to a previous version.
        """
        target = await self.get_version(prompt_id, target_version)
        if not target:
            logger.error(f"Version {target_version} not found for prompt {prompt_id}")
            return False
            
        # Set as current version
        self._current_versions[prompt_id] = target.version_id
        
        logger.info(f"Rolled back prompt {prompt_id} to version {target_version}")
        return True
        
    async def get_version_history(self, prompt_id: str) -> List[Dict]:
        """Get version history for a prompt."""
        versions = self._versions.get(prompt_id, [])
        return [
            {
                'version_number': v.version_number,
                'author': v.author,
                'commit_message': v.commit_message,
                'timestamp': v.timestamp.isoformat(),
                'tags': v.tags
            }
            for v in versions
        ]


# =============================================================================
# MAIN CPEL INTERFACE
# =============================================================================

class ContextPromptEngineeringLoop:
    """
    Main interface for the Context Prompt Engineering Loop.
    
    This class orchestrates all components of the CPEL system and provides
    a unified interface for prompt optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.registry = PromptRegistry()
        self.context_engine = ContextEngine()
        self.performance_tracker = PerformanceTracker()
        self.modifier = ContextBasedModifier()
        self.example_selector = FewShotExampleSelector()
        self.assembler = DynamicPromptAssembler()
        self.ab_tester = ABTestFramework()
        self.version_control = PromptVersionControl()
        
        # State
        self._initialized = False
        
    async def initialize(self):
        """Initialize the CPEL system."""
        if self._initialized:
            return
            
        logger.info("Initializing Context Prompt Engineering Loop...")
        
        # Load default templates
        await self._load_default_templates()
        
        self._initialized = True
        logger.info("CPEL initialization complete")
        
    async def _load_default_templates(self):
        """Load default prompt templates."""
        default_templates = [
            OptimizableTemplate(
                template_id="system_default",
                name="System Default",
                category="system",
                template_str="You are a helpful AI assistant. Respond to user requests accurately and helpfully.",
                performance_baseline=0.7
            ),
            OptimizableTemplate(
                template_id="browser_task",
                name="Browser Task",
                category="browser",
                template_str="You are controlling a web browser. Navigate to websites, extract information, and perform web-based tasks as requested.",
                performance_baseline=0.75
            ),
            OptimizableTemplate(
                template_id="email_task",
                name="Email Task",
                category="email",
                template_str="You are managing emails through Gmail. Compose, send, read, and organize emails as requested.",
                performance_baseline=0.8
            ),
            OptimizableTemplate(
                template_id="voice_task",
                name="Voice Task",
                category="voice",
                template_str="You are handling voice communications through Twilio. Make calls, send voice messages, and manage voice interactions.",
                performance_baseline=0.75
            )
        ]
        
        for template in default_templates:
            await self.registry.register_template(template)
            
    async def get_optimized_prompt(
        self,
        user_input: str,
        task_type: Optional[TaskType] = None,
        conversation_history: List[Dict] = None,
        system_state: Dict = None,
        llm_interface: Any = None
    ) -> OptimizedPrompt:
        """
        Main entry point: Get an optimized prompt for the given context.
        
        Args:
            user_input: The user's request or query
            task_type: Optional explicit task type
            conversation_history: Previous conversation turns
            system_state: Current system state
            llm_interface: Interface to LLM for quality evaluation
            
        Returns:
            OptimizedPrompt containing the optimized prompt and metadata
        """
        if not self._initialized:
            await self.initialize()
            
        conversation_history = conversation_history or []
        system_state = system_state or {}
        
        # Detect context
        context = await self.context_engine.detect_context(
            user_input, conversation_history, system_state
        )
        
        # Override task type if provided
        if task_type:
            context.task_type = task_type
            
        # Get base template
        base_template = await self.registry.get_template_for_task(
            context.task_type, context
        )
        
        if not base_template:
            base_template = await self.registry.get_template("system_default")
            
        # Check for active A/B test
        variant = None
        # (A/B test integration would go here)
        
        # Apply context-based modifications
        modified_template = self.modifier.apply_context_modifications(
            base_template.template_str, context
        )
        
        # Select few-shot examples
        examples = await self.example_selector.select_examples(
            query=user_input,
            task_type=context.task_type.value,
            n_examples=3,
            method='hybrid'
        )
        
        # Prepare variables
        variables = {
            'system_instruction': f"You are an AI assistant specializing in {context.task_type.value} tasks.",
            'user_input': user_input,
            'task_type': context.task_type.value
        }
        
        # Assemble final prompt
        assembled_prompt = await self.assembler.assemble_prompt(
            base_template=modified_template,
            context=context,
            examples=examples,
            variables=variables
        )
        
        return OptimizedPrompt(
            prompt=assembled_prompt,
            template_id=base_template.template_id,
            variant_id=variant.variant_id if variant else None,
            context_used=context,
            examples_used=examples
        )
        
    async def record_execution_result(
        self,
        prompt_id: str,
        rendered_prompt: str,
        context: ExecutionContext,
        response: str,
        execution_time_ms: int,
        llm_interface: Any = None
    ) -> PromptMetrics:
        """
        Record the outcome of prompt execution for optimization.
        
        Args:
            prompt_id: ID of the template used
            rendered_prompt: The actual prompt sent to LLM
            context: Execution context
            response: LLM response
            execution_time_ms: Execution time in milliseconds
            llm_interface: Interface to LLM for quality evaluation
            
        Returns:
            PromptMetrics collected from this execution
        """
        # Collect performance metrics
        metrics = await self.performance_tracker.collect_execution_metrics(
            prompt_id=prompt_id,
            rendered_prompt=rendered_prompt,
            context=context,
            llm_response=response,
            execution_time_ms=execution_time_ms,
            llm_interface=llm_interface
        )
        
        # Update template score
        await self._update_template_score(prompt_id)
        
        return metrics
        
    async def _update_template_score(self, prompt_id: str):
        """Update template performance score based on recent metrics."""
        aggregated = await self.performance_tracker.aggregate_performance(
            prompt_id=prompt_id,
            time_window='24h',
            dimensions=['response_accuracy', 'response_relevance', 'token_efficiency']
        )
        
        if aggregated.sample_count > 0:
            await self.registry.update_template_score(
                prompt_id, aggregated.overall_score
            )
            
    async def create_ab_test(
        self,
        template_id: str,
        variant_templates: List[str],
        test_name: str = None
    ) -> str:
        """
        Create an A/B test for a template.
        
        Args:
            template_id: ID of the base template
            variant_templates: List of variant template strings
            test_name: Optional name for the test
            
        Returns:
            Test ID
        """
        base = await self.registry.get_template(template_id)
        if not base:
            raise ValueError(f"Template {template_id} not found")
            
        variants = []
        for i, variant_str in enumerate(variant_templates):
            variant = PromptVariant(
                variant_id=f"{template_id}_variant_{i}",
                parent_template_id=template_id,
                template_str=variant_str,
                modifications={'template': variant_str}
            )
            variants.append(variant)
            
        return await self.ab_tester.create_test(
            test_name=test_name or f"Test for {base.name}",
            control_template=base,
            variants=variants,
            success_metrics=['response_accuracy', 'response_relevance', 'helpfulness'],
            min_samples=100
        )
        
    async def get_performance_report(
        self,
        prompt_id: str,
        time_window: str = '24h'
    ) -> Dict:
        """
        Get performance report for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            time_window: Time window for metrics ('1h', '6h', '24h', '7d', '30d')
            
        Returns:
            Performance report dictionary
        """
        aggregated = await self.performance_tracker.aggregate_performance(
            prompt_id=prompt_id,
            time_window=time_window,
            dimensions=['response_accuracy', 'response_relevance', 'response_completeness',
                       'response_coherence', 'helpfulness', 'token_efficiency']
        )
        
        template = await self.registry.get_template(prompt_id)
        
        return {
            'prompt_id': prompt_id,
            'prompt_name': template.name if template else 'Unknown',
            'time_window': time_window,
            'overall_score': aggregated.overall_score,
            'sample_count': aggregated.sample_count,
            'metrics': aggregated.stats,
            'version_history': await self.version_control.get_version_history(prompt_id)
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the Context Prompt Engineering Loop."""
    
    # Initialize CPEL
    cpel = ContextPromptEngineeringLoop()
    await cpel.initialize()
    
    # Example: Get optimized prompt for a browser task
    user_request = "Search for the latest news about AI and summarize the top 3 articles"
    
    optimized = await cpel.get_optimized_prompt(
        user_input=user_request,
        conversation_history=[],
        system_state={'session_id': 'test_session_123'}
    )
    
    print("=" * 60)
    print("OPTIMIZED PROMPT:")
    print("=" * 60)
    print(optimized.prompt)
    print("=" * 60)
    print(f"\nTemplate ID: {optimized.template_id}")
    print(f"Context: {optimized.context_used}")
    print(f"Examples used: {len(optimized.examples_used)}")
    
    # Simulate recording execution result
    mock_response = "Here are the top 3 AI news articles: 1. GPT-5 released..."
    
    metrics = await cpel.record_execution_result(
        prompt_id=optimized.template_id,
        rendered_prompt=optimized.prompt,
        context=optimized.context_used,
        response=mock_response,
        execution_time_ms=2500
    )
    
    print(f"\nMetrics recorded:")
    print(f"  Token efficiency: {metrics.token_efficiency:.2f}")
    print(f"  Latency: {metrics.latency_ms}ms")
    
    # Get performance report
    report = await cpel.get_performance_report(optimized.template_id)
    print(f"\nPerformance Report:")
    print(f"  Overall score: {report['overall_score']:.2f}")
    print(f"  Sample count: {report['sample_count']}")


if __name__ == "__main__":
    asyncio.run(main())
