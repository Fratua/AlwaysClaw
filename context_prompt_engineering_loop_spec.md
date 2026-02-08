# Context Prompt Engineering Loop - Technical Specification
## Windows 10 OpenClaw-Inspired AI Agent System

### Version: 1.0.0
### Date: 2025
### Target: GPT-5.2 with Extra High Thinking Capability

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Prompt Performance Tracking](#1-prompt-performance-tracking)
5. [Context-Aware Prompt Adjustment](#2-context-aware-prompt-adjustment)
6. [Template Optimization Strategies](#3-template-optimization-strategies)
7. [A/B Testing Framework](#4-ab-testing-framework-for-prompts)
8. [Prompt Effectiveness Metrics](#5-prompt-effectiveness-metrics)
9. [Dynamic Prompt Assembly](#6-dynamic-prompt-assembly)
10. [Few-Shot Example Selection](#7-few-shot-example-selection)
11. [Prompt Versioning and Rollback](#8-prompt-versioning-and-rollback)
12. [Implementation Guide](#implementation-guide)
13. [Integration with Agent Loops](#integration-with-agent-loops)

---

## Executive Summary

The Context Prompt Engineering Loop (CPEL) is an autonomous, self-improving subsystem designed to continuously optimize prompts for the Windows 10 OpenClaw-inspired AI agent. Leveraging GPT-5.2's advanced reasoning capabilities, CPEL implements a closed-loop feedback system that tracks prompt performance, analyzes contextual effectiveness, and automatically refines prompt templates based on real-world usage patterns.

### Key Capabilities
- **Autonomous Optimization**: Self-improving prompts without human intervention
- **Context-Aware Adaptation**: Dynamic adjustment based on task context
- **Performance Tracking**: Comprehensive metrics and analytics
- **A/B Testing**: Statistical validation of prompt variants
- **Version Control**: Full history with rollback capabilities
- **Real-time Assembly**: Dynamic prompt construction

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT PROMPT ENGINEERING LOOP                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Prompt     │───▶│  Performance │───▶│   Context    │                  │
│  │   Registry   │    │   Tracker    │    │   Analyzer   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │              OPTIMIZATION ENGINE                      │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │                 │
│  │  │ Template │  │   A/B    │  │  Dynamic │           │                 │
│  │  │Optimizer │  │  Tester  │  │ Assembler│           │                 │
│  │  └──────────┘  └──────────┘  └──────────┘           │                 │
│  └──────────────────────────────────────────────────────┘                 │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Version    │◀───│   Few-Shot   │◀───│   Metrics    │                  │
│  │   Control    │    │   Selector   │    │   Engine     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Input Request ──▶ Context Analysis ──▶ Prompt Selection ──▶ LLM Execution
                                              │                    │
                                              ▼                    ▼
                                       Template Assembly ◀── Response Capture
                                              │                    │
                                              ▼                    ▼
                                       Performance Log ──▶ Metric Calculation
                                                                   │
                                                                   ▼
                                                            Optimization Decision
                                                                   │
                                                                   ▼
                                                            Template Update
```

---

## Core Components

### 1. Prompt Registry
Central repository for all prompt templates with metadata.

```python
class PromptRegistry:
    """
    Central registry for prompt templates with full metadata tracking.
    """
    
    def __init__(self):
        self.templates = {}  # prompt_id -> PromptTemplate
        self.versions = {}   # prompt_id -> List[Version]
        self.metadata = {}   # prompt_id -> Metadata
        
    class PromptTemplate:
        def __init__(self, 
                     template_id: str,
                     name: str,
                     category: str,
                     template_str: str,
                     variables: List[str],
                     context_rules: Dict,
                     performance_baseline: float):
            self.template_id = template_id
            self.name = name
            self.category = category  # 'system', 'task', 'conversation', 'tool'
            self.template_str = template_str
            self.variables = variables
            self.context_rules = context_rules
            self.performance_baseline = performance_baseline
            self.created_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()
            self.usage_count = 0
            self.version = "1.0.0"
```

### 2. Context Engine
Analyzes execution context to inform prompt selection and optimization.

```python
class ContextEngine:
    """
    Analyzes and manages execution context for prompt optimization.
    """
    
    def __init__(self):
        self.context_history = []
        self.context_embeddings = {}
        self.similarity_threshold = 0.85
        
    class ExecutionContext:
        def __init__(self):
            self.task_type = None           # Type of task being performed
            self.user_intent = None         # Detected user intent
            self.conversation_depth = 0     # How deep in conversation
            self.previous_outcomes = []     # Recent task outcomes
            self.system_state = {}          # Current system state
            self.time_context = {}          # Time-based context
            self.emotional_tone = None      # Detected emotional tone
            self.complexity_score = 0.0     # Task complexity (0-1)
            self.urgency_level = 0.0        # Task urgency (0-1)
            self.domain = None              # Domain context
```

---

## 1. Prompt Performance Tracking

### 1.1 Performance Metrics Schema

```python
class PerformanceTracker:
    """
    Comprehensive performance tracking for prompt effectiveness.
    """
    
    def __init__(self):
        self.metrics_store = MetricsStore()
        self.aggregation_window = 3600  # 1 hour default
        
    class PromptMetrics:
        """Complete metrics for a single prompt execution."""
        
        # Response Quality Metrics
        response_accuracy: float        # 0-1 accuracy score
        response_relevance: float       # Semantic relevance to query
        response_completeness: float    # Did it answer fully?
        response_coherence: float       # Logical flow quality
        
        # Efficiency Metrics
        token_count_input: int
        token_count_output: int
        token_efficiency: float         # output_quality / token_count
        latency_ms: int
        cost_per_request: float
        
        # User Satisfaction Metrics
        user_rating: Optional[float]    # Explicit user feedback
        implicit_satisfaction: float    # Derived from behavior
        retry_count: int                # How many retries needed
        correction_needed: bool         # Did user correct?
        
        # Context Metrics
        context_match_score: float      # How well context was used
        variable_substitution_success: float
        template_render_success: bool
        
        # Temporal Metrics
        timestamp: datetime
        session_id: str
        conversation_id: str
```

### 1.2 Real-time Performance Collection

```python
class RealTimePerformanceCollector:
    """
    Collects performance data in real-time during prompt execution.
    """
    
    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.processors = []
        
    async def collect_execution_metrics(
        self,
        prompt_id: str,
        rendered_prompt: str,
        context: ExecutionContext,
        llm_response: str,
        execution_time_ms: int
    ) -> PromptMetrics:
        """
        Collect comprehensive metrics for a single prompt execution.
        """
        metrics = PromptMetrics()
        
        # Calculate quality scores using GPT-5.2 evaluation
        metrics.response_quality = await self._evaluate_quality(
            rendered_prompt, llm_response, context
        )
        
        # Calculate efficiency metrics
        metrics.token_efficiency = self._calculate_token_efficiency(
            rendered_prompt, llm_response
        )
        
        # Calculate context match score
        metrics.context_match_score = self._calculate_context_match(
            rendered_prompt, context
        )
        
        # Store metrics
        await self.metrics_store.store(metrics)
        
        return metrics
        
    async def _evaluate_quality(
        self,
        prompt: str,
        response: str,
        context: ExecutionContext
    ) -> Dict[str, float]:
        """
        Use GPT-5.2 to evaluate response quality across dimensions.
        """
        evaluation_prompt = f"""
        Evaluate the following AI response based on the original prompt and context.
        
        ORIGINAL PROMPT:
        {prompt[:2000]}
        
        AI RESPONSE:
        {response[:2000]}
        
        CONTEXT:
        Task Type: {context.task_type}
        User Intent: {context.user_intent}
        Complexity: {context.complexity_score}
        
        Rate the response on a scale of 0.0 to 1.0 for:
        1. Accuracy - factual correctness
        2. Relevance - addresses the specific query
        3. Completeness - fully answers the question
        4. Coherence - logical flow and clarity
        5. Helpfulness - practical utility
        
        Return ONLY a JSON object with these scores.
        """
        
        evaluation = await self.llm.generate(evaluation_prompt)
        return json.loads(evaluation)
```

### 1.3 Performance Aggregation and Analysis

```python
class PerformanceAggregator:
    """
    Aggregates performance data across time windows and dimensions.
    """
    
    def __init__(self):
        self.time_windows = ['1h', '6h', '24h', '7d', '30d']
        self.aggregation_strategies = {
            'mean': np.mean,
            'median': np.median,
            'p95': lambda x: np.percentile(x, 95),
            'trend': self._calculate_trend
        }
        
    async def aggregate_performance(
        self,
        prompt_id: str,
        time_window: str,
        dimensions: List[str]
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for analysis and optimization decisions.
        """
        raw_metrics = await self.metrics_store.query(
            prompt_id=prompt_id,
            start_time=self._get_window_start(time_window),
            end_time=datetime.utcnow()
        )
        
        aggregated = AggregatedMetrics()
        
        for dimension in dimensions:
            values = [getattr(m, dimension) for m in raw_metrics]
            
            aggregated.stats[dimension] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'trend': self._calculate_trend(values),
                'sample_count': len(values)
            }
            
        # Detect performance anomalies
        aggregated.anomalies = self._detect_anomalies(raw_metrics)
        
        # Calculate performance score
        aggregated.overall_score = self._calculate_overall_score(aggregated.stats)
        
        return aggregated
```

### 1.4 Performance Dashboard Schema

```python
class PerformanceDashboard:
    """
    Real-time performance visualization and alerting.
    """
    
    def generate_dashboard_data(self) -> Dict:
        return {
            'summary': {
                'total_prompts': self._count_total_prompts(),
                'active_experiments': self._count_active_experiments(),
                'avg_performance_score': self._get_avg_score(),
                'optimization_opportunities': self._find_opportunities()
            },
            'top_performing_prompts': self._get_top_prompts(limit=10),
            'underperforming_prompts': self._get_underperforming(limit=10),
            'trending_improvements': self._get_improving_prompts(),
            'trending_declines': self._get_declining_prompts(),
            'ab_test_results': self._get_ab_test_summary(),
            'optimization_queue': self._get_optimization_queue()
        }
```

---

## 2. Context-Aware Prompt Adjustment

### 2.1 Context Detection Engine

```python
class ContextDetectionEngine:
    """
    Detects and classifies execution context for prompt adjustment.
    """
    
    def __init__(self):
        self.context_classifiers = {
            'task_type': TaskTypeClassifier(),
            'complexity': ComplexityClassifier(),
            'urgency': UrgencyClassifier(),
            'domain': DomainClassifier(),
            'emotional_state': EmotionalStateClassifier()
        }
        
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
        
        # Parallel context classification
        results = await asyncio.gather(
            self.context_classifiers['task_type'].classify(user_input),
            self.context_classifiers['complexity'].classify(user_input, conversation_history),
            self.context_classifiers['urgency'].classify(user_input, system_state),
            self.context_classifiers['domain'].classify(user_input),
            self.context_classifiers['emotional_state'].classify(user_input, conversation_history)
        )
        
        context.task_type = results[0]
        context.complexity_score = results[1]
        context.urgency_level = results[2]
        context.domain = results[3]
        context.emotional_tone = results[4]
        
        # Calculate conversation depth
        context.conversation_depth = len(conversation_history)
        
        # Extract previous outcomes
        context.previous_outcomes = self._extract_outcomes(conversation_history)
        
        return context
```

### 2.2 Context-Based Prompt Modification

```python
class ContextBasedModifier:
    """
    Modifies prompts based on detected context.
    """
    
    def __init__(self):
        self.modification_rules = self._load_modification_rules()
        
    def apply_context_modifications(
        self,
        base_prompt: str,
        context: ExecutionContext
    ) -> str:
        """
        Apply context-aware modifications to base prompt.
        """
        modified_prompt = base_prompt
        
        # Apply complexity-based modifications
        if context.complexity_score > 0.8:
            modified_prompt = self._add_complexity_instructions(modified_prompt)
        elif context.complexity_score < 0.3:
            modified_prompt = self._simplify_instructions(modified_prompt)
            
        # Apply urgency-based modifications
        if context.urgency_level > 0.7:
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
        complexity_addition = """
        This is a complex request. Please:
        1. Break down your response into clear steps
        2. Provide detailed explanations for each step
        3. Consider edge cases and alternatives
        4. Use examples to illustrate complex concepts
        5. Summarize key points at the end
        """
        return prompt + "\n" + complexity_addition
        
    def _add_urgency_instructions(self, prompt: str) -> str:
        urgency_addition = """
        This request is time-sensitive. Please:
        1. Prioritize speed while maintaining accuracy
        2. Provide the most critical information first
        3. Use concise, direct language
        4. Mark any information that needs verification
        """
        return prompt + "\n" + urgency_addition
```

### 2.3 Dynamic Context Injection

```python
class DynamicContextInjector:
    """
    Injects relevant context into prompts dynamically.
    """
    
    def __init__(self):
        self.context_retriever = ContextRetriever()
        self.relevance_scorer = RelevanceScorer()
        
    async def inject_context(
        self,
        prompt: str,
        query: str,
        max_context_tokens: int = 2000
    ) -> str:
        """
        Dynamically inject relevant context into prompt.
        """
        # Retrieve potentially relevant context
        candidates = await self.context_retriever.retrieve(
            query=query,
            top_k=20
        )
        
        # Score relevance of each candidate
        scored_candidates = []
        for candidate in candidates:
            score = await self.relevance_scorer.score(query, candidate)
            scored_candidates.append((candidate, score))
            
        # Sort by relevance
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select context within token budget
        selected_context = []
        current_tokens = 0
        
        for candidate, score in scored_candidates:
            candidate_tokens = self._estimate_tokens(candidate)
            if current_tokens + candidate_tokens <= max_context_tokens:
                selected_context.append((candidate, score))
                current_tokens += candidate_tokens
            else:
                break
                
        # Format and inject context
        context_section = self._format_context(selected_context)
        
        return self._inject_into_prompt(prompt, context_section)
```

---

## 3. Template Optimization Strategies

### 3.1 Template Structure

```python
@dataclass
class OptimizableTemplate:
    """
    Template with optimization metadata and versioning.
    """
    template_id: str
    name: str
    version: str
    
    # Template components
    system_instruction: str
    task_description: str
    output_format: str
    constraints: List[str]
    examples: List[Dict]
    
    # Optimization metadata
    optimization_history: List[OptimizationRecord]
    performance_baseline: float
    current_score: float
    
    # A/B testing
    variants: List[TemplateVariant]
    active_variant: Optional[str]
    
    # Context rules
    context_rules: Dict[str, Any]
    
@dataclass
class TemplateVariant:
    """A/B test variant of a template."""
    variant_id: str
    parent_template_id: str
    modifications: Dict[str, str]  # field -> new_value
    traffic_percentage: float
    performance_score: float
    sample_count: int
```

### 3.2 Optimization Strategies

```python
class TemplateOptimizer:
    """
    Applies various optimization strategies to templates.
    """
    
    STRATEGIES = {
        'clarity_enhancement': ClarityEnhancementStrategy(),
        'specificity_boost': SpecificityBoostStrategy(),
        'example_optimization': ExampleOptimizationStrategy(),
        'constraint_refinement': ConstraintRefinementStrategy(),
        'format_optimization': FormatOptimizationStrategy(),
        'length_optimization': LengthOptimizationStrategy()
    }
    
    async def optimize_template(
        self,
        template: OptimizableTemplate,
        strategy: str,
        performance_data: AggregatedMetrics
    ) -> OptimizableTemplate:
        """
        Apply optimization strategy to template.
        """
        optimizer = self.STRATEGIES.get(strategy)
        if not optimizer:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # Analyze current performance
        analysis = await optimizer.analyze(template, performance_data)
        
        # Generate optimized version
        optimized = await optimizer.optimize(template, analysis)
        
        # Create variant for A/B testing
        variant = self._create_variant(template, optimized, strategy)
        
        return optimized, variant
```

### 3.3 Clarity Enhancement Strategy

```python
class ClarityEnhancementStrategy:
    """
    Optimizes template for clarity and understandability.
    """
    
    async def analyze(
        self,
        template: OptimizableTemplate,
        performance_data: AggregatedMetrics
    ) -> AnalysisResult:
        """
        Analyze template for clarity issues.
        """
        analysis_prompt = f"""
        Analyze the following prompt template for clarity issues:
        
        TEMPLATE:
        {template.system_instruction}
        {template.task_description}
        {template.output_format}
        
        PERFORMANCE DATA:
        - Response coherence: {performance_data.stats.get('response_coherence', {}).get('mean', 'N/A')}
        - User corrections: {performance_data.stats.get('correction_needed', {}).get('mean', 'N/A')}
        - Retry count: {performance_data.stats.get('retry_count', {}).get('mean', 'N/A')}
        
        Identify:
        1. Ambiguous instructions
        2. Confusing language
        3. Unclear expectations
        4. Missing context
        5. Overly complex sentences
        
        Return a structured analysis with specific issues and recommendations.
        """
        
        analysis = await self.llm.generate(analysis_prompt)
        return self._parse_analysis(analysis)
        
    async def optimize(
        self,
        template: OptimizableTemplate,
        analysis: AnalysisResult
    ) -> OptimizableTemplate:
        """
        Generate clarity-enhanced version of template.
        """
        optimization_prompt = f"""
        Rewrite the following prompt template to improve clarity:
        
        ORIGINAL TEMPLATE:
        {template.system_instruction}
        {template.task_description}
        
        IDENTIFIED ISSUES:
        {analysis.issues}
        
        OPTIMIZATION GUIDELINES:
        1. Use simple, direct language
        2. Break complex instructions into steps
        3. Provide clear examples
        4. Define technical terms
        5. Use consistent terminology
        6. Remove ambiguity
        
        Return the optimized template maintaining the original intent.
        """
        
        optimized = await self.llm.generate(optimization_prompt)
        return self._create_optimized_template(template, optimized)
```

### 3.4 Multi-Strategy Optimization Pipeline

```python
class MultiStrategyOptimizer:
    """
    Runs multiple optimization strategies and selects best result.
    """
    
    async def run_optimization_pipeline(
        self,
        template: OptimizableTemplate,
        performance_data: AggregatedMetrics
    ) -> List[OptimizableTemplate]:
        """
        Run multiple optimization strategies in parallel.
        """
        # Determine which strategies to apply based on performance data
        strategies = self._select_strategies(performance_data)
        
        # Run optimizations in parallel
        optimization_tasks = [
            self.optimize_template(template, strategy, performance_data)
            for strategy in strategies
        ]
        
        results = await asyncio.gather(*optimization_tasks)
        
        # Score each optimized version
        scored_results = []
        for optimized, variant in results:
            predicted_score = await self._predict_performance(optimized)
            scored_results.append((optimized, variant, predicted_score))
            
        # Sort by predicted performance
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        return scored_results
        
    def _select_strategies(
        self,
        performance_data: AggregatedMetrics
    ) -> List[str]:
        """
        Select optimization strategies based on performance gaps.
        """
        strategies = []
        
        stats = performance_data.stats
        
        # Check coherence issues
        if stats.get('response_coherence', {}).get('mean', 1.0) < 0.8:
            strategies.append('clarity_enhancement')
            
        # Check completeness issues
        if stats.get('response_completeness', {}).get('mean', 1.0) < 0.8:
            strategies.append('specificity_boost')
            
        # Check format adherence
        if stats.get('format_adherence', {}).get('mean', 1.0) < 0.9:
            strategies.append('format_optimization')
            
        # Check efficiency
        token_efficiency = stats.get('token_efficiency', {}).get('mean', 1.0)
        if token_efficiency < 0.5:
            strategies.append('length_optimization')
            
        # Check example effectiveness
        if stats.get('example_effectiveness', {}).get('mean', 1.0) < 0.7:
            strategies.append('example_optimization')
            
        return strategies if strategies else ['clarity_enhancement']
```

---

## 4. A/B Testing Framework for Prompts

### 4.1 A/B Test Architecture

```python
class ABTestFramework:
    """
    Comprehensive A/B testing framework for prompt optimization.
    """
    
    def __init__(self):
        self.test_manager = TestManager()
        self.traffic_router = TrafficRouter()
        self.statistics_engine = StatisticsEngine()
        
    class ABTest:
        """Represents a single A/B test."""
        
        def __init__(
            self,
            test_id: str,
            name: str,
            hypothesis: str,
            control_variant: PromptVariant,
            treatment_variants: List[PromptVariant],
            success_metrics: List[str],
            min_sample_size: int,
            confidence_level: float = 0.95,
            max_duration_days: int = 14
        ):
            self.test_id = test_id
            self.name = name
            self.hypothesis = hypothesis
            self.control_variant = control_variant
            self.treatment_variants = treatment_variants
            self.success_metrics = success_metrics
            self.min_sample_size = min_sample_size
            self.confidence_level = confidence_level
            self.max_duration_days = max_duration_days
            
            self.status = 'pending'  # pending, running, completed, cancelled
            self.start_time = None
            self.end_time = None
            self.results = None
            
    async def create_test(
        self,
        base_template: OptimizableTemplate,
        optimization_strategies: List[str],
        success_criteria: Dict
    ) -> ABTest:
        """
        Create a new A/B test for prompt optimization.
        """
        # Generate variants using different strategies
        variants = []
        for strategy in optimization_strategies:
            optimizer = TemplateOptimizer.STRATEGIES[strategy]
            optimized = await optimizer.optimize(base_template, None)
            variant = PromptVariant(
                variant_id=f"{base_template.template_id}_{strategy}",
                parent_template_id=base_template.template_id,
                modifications=self._extract_modifications(base_template, optimized),
                traffic_percentage=1.0 / (len(optimization_strategies) + 1),
                performance_score=0.0,
                sample_count=0
            )
            variants.append(variant)
            
        # Create control variant (current template)
        control = PromptVariant(
            variant_id=f"{base_template.template_id}_control",
            parent_template_id=base_template.template_id,
            modifications={},
            traffic_percentage=1.0 / (len(optimization_strategies) + 1),
            performance_score=base_template.current_score,
            sample_count=0
        )
        
        test = ABTest(
            test_id=str(uuid.uuid4()),
            name=f"Optimization test for {base_template.name}",
            hypothesis="Optimized templates will outperform control",
            control_variant=control,
            treatment_variants=variants,
            success_metrics=success_criteria['metrics'],
            min_sample_size=success_criteria['min_samples'],
            confidence_level=success_criteria.get('confidence', 0.95)
        )
        
        await self.test_manager.save_test(test)
        return test
```

### 4.2 Traffic Routing

```python
class TrafficRouter:
    """
    Routes traffic to appropriate test variants.
    """
    
    def __init__(self):
        self.assignment_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def get_variant_for_request(
        self,
        test_id: str,
        request_context: Dict
    ) -> PromptVariant:
        """
        Determine which variant to serve for a request.
        """
        # Check for sticky assignment
        user_id = request_context.get('user_id')
        session_id = request_context.get('session_id')
        
        cache_key = f"{test_id}:{user_id or session_id}"
        
        if cache_key in self.assignment_cache:
            cached = self.assignment_cache[cache_key]
            if cached['expires'] > time.time():
                return cached['variant']
                
        # Get test configuration
        test = await self.test_manager.get_test(test_id)
        
        if test.status != 'running':
            return test.control_variant
            
        # Use consistent hashing for user assignment
        variant = self._consistent_hash_assignment(
            test,
            user_id or session_id
        )
        
        # Cache assignment
        self.assignment_cache[cache_key] = {
            'variant': variant,
            'expires': time.time() + self.cache_ttl
        }
        
        return variant
        
    def _consistent_hash_assignment(
        self,
        test: ABTest,
        identifier: str
    ) -> PromptVariant:
        """
        Assign variant using consistent hashing for even distribution.
        """
        hash_value = hashlib.md5(
            f"{test.test_id}:{identifier}".encode()
        ).hexdigest()
        
        hash_int = int(hash_value, 16)
        
        # Calculate assignment based on traffic percentages
        all_variants = [test.control_variant] + test.treatment_variants
        
        cumulative = 0
        for variant in all_variants:
            cumulative += variant.traffic_percentage
            if hash_int % 10000 < cumulative * 10000:
                return variant
                
        return test.control_variant
```

### 4.3 Statistical Analysis

```python
class StatisticsEngine:
    """
    Statistical analysis for A/B test results.
    """
    
    def __init__(self):
        self.min_effect_size = 0.05  # 5% minimum detectable effect
        
    async def analyze_test_results(
        self,
        test: ABTest
    ) -> TestResults:
        """
        Perform statistical analysis of A/B test results.
        """
        results = TestResults()
        
        # Collect metrics for each variant
        variant_metrics = {}
        for variant in [test.control_variant] + test.treatment_variants:
            metrics = await self._collect_variant_metrics(
                test.test_id,
                variant.variant_id,
                test.success_metrics
            )
            variant_metrics[variant.variant_id] = metrics
            
        # Perform statistical tests
        for metric in test.success_metrics:
            control_data = variant_metrics[test.control_variant.variant_id][metric]
            
            for treatment in test.treatment_variants:
                treatment_data = variant_metrics[treatment.variant_id][metric]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    control_data,
                    treatment_data
                )
                
                # Calculate effect size (Cohen's d)
                effect_size = self._cohens_d(control_data, treatment_data)
                
                # Calculate confidence interval
                ci = self._confidence_interval(
                    treatment_data,
                    test.confidence_level
                )
                
                # Determine statistical significance
                is_significant = (
                    p_value < (1 - test.confidence_level) and
                    abs(effect_size) >= self.min_effect_size
                )
                
                results.comparisons.append(ComparisonResult(
                    metric=metric,
                    control_variant=test.control_variant.variant_id,
                    treatment_variant=treatment.variant_id,
                    control_mean=np.mean(control_data),
                    treatment_mean=np.mean(treatment_data),
                    relative_change=(np.mean(treatment_data) - np.mean(control_data)) / np.mean(control_data),
                    p_value=p_value,
                    effect_size=effect_size,
                    confidence_interval=ci,
                    is_significant=is_significant,
                    winner=treatment.variant_id if (is_significant and effect_size > 0) else None
                ))
                
        # Determine overall winner
        results.overall_winner = self._determine_overall_winner(results.comparisons)
        results.recommendation = self._generate_recommendation(results)
        
        return results
        
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### 4.4 Multi-Armed Bandit for Dynamic Allocation

```python
class MultiArmedBanditRouter:
    """
    Thompson Sampling based multi-armed bandit for dynamic traffic allocation.
    """
    
    def __init__(self):
        self.exploration_rate = 0.1  # 10% exploration
        
    async def thompson_sampling_assignment(
        self,
        test: ABTest,
        metric: str
    ) -> PromptVariant:
        """
        Assign variant using Thompson Sampling.
        """
        # Get current performance estimates for each variant
        variants = [test.control_variant] + test.treatment_variants
        
        # Sample from posterior distributions
        samples = []
        for variant in variants:
            # Get success count and total trials
            successes, trials = await self._get_variant_stats(
                test.test_id,
                variant.variant_id,
                metric
            )
            
            # Sample from Beta distribution (conjugate prior for Bernoulli)
            sample = np.random.beta(successes + 1, trials - successes + 1)
            samples.append((variant, sample))
            
        # Select variant with highest sample
        best_variant = max(samples, key=lambda x: x[1])[0]
        
        # Add exploration
        if random.random() < self.exploration_rate:
            best_variant = random.choice(variants)
            
        return best_variant
```

---

## 5. Prompt Effectiveness Metrics

### 5.1 Comprehensive Metrics Framework

```python
class EffectivenessMetricsFramework:
    """
    Comprehensive framework for measuring prompt effectiveness.
    """
    
    METRIC_CATEGORIES = {
        'quality': [
            'accuracy',
            'relevance',
            'completeness',
            'coherence',
            'helpfulness',
            'factual_correctness'
        ],
        'efficiency': [
            'token_efficiency',
            'response_latency',
            'cost_per_request',
            'time_to_first_token'
        ],
        'user_satisfaction': [
            'explicit_rating',
            'implicit_satisfaction',
            'retry_rate',
            'correction_rate',
            'follow_up_question_rate'
        ],
        'robustness': [
            'consistency_score',
            'edge_case_handling',
            'error_recovery_rate',
            'context_adherence'
        ]
    }
    
    async def calculate_comprehensive_score(
        self,
        prompt_id: str,
        time_window: str
    ) -> ComprehensiveScore:
        """
        Calculate comprehensive effectiveness score.
        """
        score = ComprehensiveScore()
        
        # Calculate scores for each category
        for category, metrics in self.METRIC_CATEGORIES.items():
            category_score = await self._calculate_category_score(
                prompt_id,
                metrics,
                time_window
            )
            setattr(score, category, category_score)
            
        # Calculate weighted overall score
        weights = {
            'quality': 0.35,
            'efficiency': 0.25,
            'user_satisfaction': 0.25,
            'robustness': 0.15
        }
        
        score.overall = sum(
            getattr(score, cat) * weight
            for cat, weight in weights.items()
        )
        
        return score
```

### 5.2 Automated Quality Evaluation

```python
class AutomatedQualityEvaluator:
    """
    Uses GPT-5.2 to automatically evaluate prompt response quality.
    """
    
    EVALUATION_DIMENSIONS = {
        'accuracy': {
            'description': 'Factual correctness of the response',
            'criteria': [
                'Information is factually correct',
                'No hallucinations or false claims',
                'Sources are accurate if cited'
            ]
        },
        'relevance': {
            'description': 'How well response addresses the query',
            'criteria': [
                'Directly answers the question asked',
                'Stays on topic',
                'Provides requested information'
            ]
        },
        'completeness': {
            'description': 'Thoroughness of the response',
            'criteria': [
                'All parts of query are addressed',
                'No important information is missing',
                'Appropriate level of detail'
            ]
        },
        'coherence': {
            'description': 'Logical flow and clarity',
            'criteria': [
                'Clear structure and organization',
                'Logical progression of ideas',
                'Easy to understand'
            ]
        }
    }
    
    async def evaluate_response(
        self,
        original_query: str,
        prompt_used: str,
        response: str,
        context: Dict
    ) -> QualityEvaluation:
        """
        Automatically evaluate response quality.
        """
        evaluation = QualityEvaluation()
        
        for dimension, config in self.EVALUATION_DIMENSIONS.items():
            score = await self._evaluate_dimension(
                original_query,
                prompt_used,
                response,
                dimension,
                config
            )
            setattr(evaluation, dimension, score)
            
        return evaluation
        
    async def _evaluate_dimension(
        self,
        query: str,
        prompt: str,
        response: str,
        dimension: str,
        config: Dict
    ) -> DimensionScore:
        """
        Evaluate a single quality dimension.
        """
        eval_prompt = f"""
        Evaluate the following AI response on the dimension of {dimension}.
        
        DIMENSION DEFINITION:
        {config['description']}
        
        EVALUATION CRITERIA:
        {chr(10).join(f"- {c}" for c in config['criteria'])}
        
        ORIGINAL QUERY:
        {query}
        
        AI RESPONSE:
        {response}
        
        Rate the response on a scale of 0.0 to 1.0 for {dimension}.
        Provide:
        1. A score between 0.0 and 1.0
        2. Brief justification (1-2 sentences)
        3. Specific improvement suggestions if score < 0.8
        
        Return as JSON: {{"score": float, "justification": str, "suggestions": [str]}}
        """
        
        result = await self.llm.generate(eval_prompt)
        return DimensionScore(**json.loads(result))
```

### 5.3 Metric Correlation Analysis

```python
class MetricCorrelationAnalyzer:
    """
    Analyzes correlations between different metrics.
    """
    
    async def analyze_correlations(
        self,
        prompt_id: str,
        metrics: List[str],
        time_window: str
    ) -> CorrelationMatrix:
        """
        Analyze correlations between metrics.
        """
        # Collect metric data
        data = await self._collect_metric_data(prompt_id, metrics, time_window)
        
        # Calculate correlation matrix
        df = pd.DataFrame(data)
        corr_matrix = df.corr()
        
        # Identify strong correlations
        strong_correlations = []
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'metric1': metrics[i],
                        'metric2': metrics[j],
                        'correlation': corr_value,
                        'relationship': 'positive' if corr_value > 0 else 'negative'
                    })
                    
        return CorrelationMatrix(
            matrix=corr_matrix.to_dict(),
            strong_correlations=strong_correlations,
            insights=self._generate_correlation_insights(strong_correlations)
        )
```

---

## 6. Dynamic Prompt Assembly

### 6.1 Assembly Pipeline

```python
class DynamicPromptAssembler:
    """
    Dynamically assembles prompts from modular components.
    """
    
    def __init__(self):
        self.component_registry = ComponentRegistry()
        self.assembly_strategies = {
            'standard': StandardAssemblyStrategy(),
            'minimal': MinimalAssemblyStrategy(),
            'comprehensive': ComprehensiveAssemblyStrategy(),
            'adaptive': AdaptiveAssemblyStrategy()
        }
        
    class PromptComponent:
        """Individual prompt component."""
        
        def __init__(
            self,
            component_id: str,
            component_type: str,  # 'system', 'instruction', 'context', 'example', 'constraint'
            content: str,
            priority: int,
            token_estimate: int,
            conditions: List[Condition],
            dependencies: List[str]
        ):
            self.component_id = component_id
            self.component_type = component_type
            self.content = content
            self.priority = priority
            self.token_estimate = token_estimate
            self.conditions = conditions
            self.dependencies = dependencies
            
    async def assemble_prompt(
        self,
        task_type: str,
        context: ExecutionContext,
        user_input: str,
        max_tokens: int = 4000,
        strategy: str = 'adaptive'
    ) -> AssembledPrompt:
        """
        Dynamically assemble prompt for given task and context.
        """
        # Get relevant components
        components = await self.component_registry.get_components_for_task(
            task_type,
            context
        )
        
        # Filter components based on conditions
        applicable_components = [
            c for c in components
            if self._check_conditions(c.conditions, context)
        ]
        
        # Resolve dependencies
        resolved_components = self._resolve_dependencies(applicable_components)
        
        # Sort by priority
        sorted_components = sorted(
            resolved_components,
            key=lambda c: c.priority,
            reverse=True
        )
        
        # Apply assembly strategy
        assembler = self.assembly_strategies[strategy]
        assembled = await assembler.assemble(
            sorted_components,
            context,
            max_tokens
        )
        
        return assembled
```

### 6.2 Component Selection Algorithm

```python
class ComponentSelector:
    """
    Selects optimal components for prompt assembly.
    """
    
    async def select_components(
        self,
        available_components: List[PromptComponent],
        context: ExecutionContext,
        token_budget: int
    ) -> List[PromptComponent]:
        """
        Select components to maximize prompt effectiveness within token budget.
        """
        # Score each component for current context
        scored_components = []
        for component in available_components:
            relevance = await self._score_relevance(component, context)
            importance = self._score_importance(component, context)
            effectiveness = await self._get_historical_effectiveness(component)
            
            # Combined score
            combined_score = (
                relevance * 0.4 +
                importance * 0.3 +
                effectiveness * 0.3
            )
            
            scored_components.append((component, combined_score))
            
        # Sort by score
        scored_components.sort(key=lambda x: x[1], reverse=True)
        
        # Select components within token budget (knapsack-like selection)
        selected = []
        remaining_tokens = token_budget
        
        for component, score in scored_components:
            if component.token_estimate <= remaining_tokens:
                selected.append(component)
                remaining_tokens -= component.token_estimate
                
        return selected
```

### 6.3 Context-Aware Assembly Strategy

```python
class AdaptiveAssemblyStrategy:
    """
    Adapts prompt assembly based on context and task requirements.
    """
    
    async def assemble(
        self,
        components: List[PromptComponent],
        context: ExecutionContext,
        max_tokens: int
    ) -> AssembledPrompt:
        """
        Adaptively assemble prompt based on context.
        """
        # Determine assembly approach based on context
        if context.complexity_score > 0.8:
            return await self._assemble_complex(components, context, max_tokens)
        elif context.urgency_level > 0.7:
            return await self._assemble_urgent(components, context, max_tokens)
        elif context.conversation_depth > 10:
            return await self._assemble_continued(components, context, max_tokens)
        else:
            return await self._assemble_standard(components, context, max_tokens)
            
    async def _assemble_complex(
        self,
        components: List[PromptComponent],
        context: ExecutionContext,
        max_tokens: int
    ) -> AssembledPrompt:
        """
        Assembly strategy for complex tasks.
        """
        # Prioritize detailed instructions and examples
        prioritized = self._prioritize_by_type(
            components,
            type_order=['system', 'instruction', 'example', 'context', 'constraint']
        )
        
        # Add complexity-specific guidance
        complexity_guidance = PromptComponent(
            component_id='complexity_guidance',
            component_type='instruction',
            content="""
            This is a complex request. Please:
            1. Break down the problem systematically
            2. Consider multiple approaches
            3. Explain your reasoning clearly
            4. Highlight any assumptions made
            """,
            priority=100,
            token_estimate=50,
            conditions=[],
            dependencies=[]
        )
        
        prioritized.insert(1, complexity_guidance)
        
        return self._build_prompt(prioritized, max_tokens)
```

### 6.4 Template Variable Resolution

```python
class VariableResolver:
    """
    Resolves template variables with context-aware values.
    """
    
    def __init__(self):
        self.resolvers = {
            'user': UserVariableResolver(),
            'system': SystemVariableResolver(),
            'context': ContextVariableResolver(),
            'time': TimeVariableResolver(),
            'dynamic': DynamicVariableResolver()
        }
        
    async def resolve_variables(
        self,
        template: str,
        context: ExecutionContext,
        custom_resolvers: Dict = None
    ) -> str:
        """
        Resolve all variables in template.
        """
        # Find all variables
        variables = re.findall(r'\{\{(\w+)(?::(\w+))?\}\}', template)
        
        resolved = template
        for var_name, var_type in variables:
            var_type = var_type or 'dynamic'
            
            # Get appropriate resolver
            resolver = (custom_resolvers or {}).get(var_type) or \
                      self.resolvers.get(var_type)
            
            if resolver:
                value = await resolver.resolve(var_name, context)
                resolved = resolved.replace(
                    f'{{{{{var_name}{":" + var_type if var_type else ""}}}}}}',
                    str(value)
                )
                
        return resolved
```

---

## 7. Few-Shot Example Selection

### 7.1 Example Database

```python
class ExampleDatabase:
    """
    Manages and retrieves few-shot examples.
    """
    
    def __init__(self):
        self.examples = []
        self.embeddings = {}
        self.index = None  # Vector index for similarity search
        
    class FewShotExample:
        """Single few-shot example."""
        
        def __init__(
            self,
            example_id: str,
            input_text: str,
            output_text: str,
            task_type: str,
            difficulty: float,
            success_rate: float,
            usage_count: int,
            embedding: Optional[List[float]] = None
        ):
            self.example_id = example_id
            self.input_text = input_text
            self.output_text = output_text
            self.task_type = task_type
            self.difficulty = difficulty
            self.success_rate = success_rate
            self.usage_count = usage_count
            self.embedding = embedding
            
    async def add_example(
        self,
        input_text: str,
        output_text: str,
        task_type: str,
        performance_data: Dict
    ) -> FewShotExample:
        """
        Add new example to database.
        """
        # Generate embedding
        embedding = await self._generate_embedding(
            f"{input_text}\n{output_text}"
        )
        
        example = FewShotExample(
            example_id=str(uuid.uuid4()),
            input_text=input_text,
            output_text=output_text,
            task_type=task_type,
            difficulty=performance_data.get('difficulty', 0.5),
            success_rate=performance_data.get('success_rate', 1.0),
            usage_count=0,
            embedding=embedding
        )
        
        self.examples.append(example)
        self.embeddings[example.example_id] = embedding
        
        # Update index
        await self._update_index()
        
        return example
```

### 7.2 Learn-from-Mistakes (LFM) Algorithm

```python
class LearnFromMistakesSelector:
    """
    Selects examples based on model's past mistakes.
    """
    
    def __init__(self):
        self.mistake_tracker = MistakeTracker()
        self.iteration_count = 1
        self.stacked = True
        
    async def select_examples_lfm(
        self,
        model: Any,
        candidate_examples: List[FewShotExample],
        query: str,
        n_examples: int,
        option: str = 'incorrect'  # 'incorrect', 'correct', 'gray'
    ) -> List[FewShotExample]:
        """
        Select examples using Learn-from-Mistakes algorithm.
        
        Based on: "On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection"
        """
        # Initialize sets
        examples_correct = set(e.example_id for e in candidate_examples)
        examples_incorrect = set(e.example_id for e in candidate_examples)
        
        # Multiple iterations for consistency
        for iteration in range(self.iteration_count):
            iter_correct = set()
            iter_incorrect = set()
            
            # Initialize context with empty or previously selected examples
            context_examples = []
            
            for example in candidate_examples:
                # Query model with current context
                prediction = await self._query_model_with_example(
                    model, query, example, context_examples
                )
                
                # Check if prediction matches expected output
                is_correct = self._evaluate_prediction(
                    prediction, example.output_text
                )
                
                if is_correct:
                    iter_correct.add(example.example_id)
                else:
                    iter_incorrect.add(example.example_id)
                    
                    # Update context if stacked mode
                    if self.stacked:
                        context_examples.append(example)
                        
            # Update consistent sets
            examples_correct &= iter_correct
            examples_incorrect &= iter_incorrect
            
        # Determine gray examples (neither consistently correct nor incorrect)
        examples_gray = set(e.example_id for e in candidate_examples) - \
                       examples_correct - examples_incorrect
        
        # Select based on option
        if option == 'incorrect':
            selected_ids = list(examples_incorrect)
        elif option == 'correct':
            selected_ids = list(examples_correct)
        else:  # gray
            selected_ids = list(examples_gray)
            
        # Get full examples
        selected = [
            e for e in candidate_examples
            if e.example_id in selected_ids
        ]
        
        # Return top n
        return selected[:n_examples]
```

### 7.3 Learn-from-Nearest-Neighbors (LFNN) Algorithm

```python
class LearnFromNearestNeighborsSelector:
    """
    Selects examples based on semantic similarity to query.
    """
    
    def __init__(self, encoder_model: str = 'text-embedding-3-large'):
        self.encoder_model = encoder_model
        self.example_embeddings = {}
        
    async def select_examples_lfnn(
        self,
        candidate_examples: List[FewShotExample],
        query: str,
        n_neighbors: int
    ) -> List[FewShotExample]:
        """
        Select examples using Learn-from-Nearest-Neighbors algorithm.
        """
        # Pre-compute embeddings for all examples (if not already done)
        for example in candidate_examples:
            if example.example_id not in self.example_embeddings:
                embedding = await self._encode_text(
                    f"{example.input_text}\n{example.output_text}"
                )
                self.example_embeddings[example.example_id] = embedding
                
        # Encode query
        query_embedding = await self._encode_text(query)
        
        # Calculate similarities
        similarities = []
        for example in candidate_examples:
            example_embedding = self.example_embeddings[example.example_id]
            similarity = self._cosine_similarity(query_embedding, example_embedding)
            similarities.append((example, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n neighbors
        return [ex for ex, _ in similarities[:n_neighbors]]
        
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
```

### 7.4 Combined Selection Strategy

```python
class CombinedExampleSelector:
    """
    Combines LFM and LFNN for optimal example selection.
    """
    
    def __init__(self):
        self.lfm_selector = LearnFromMistakesSelector()
        self.lfnn_selector = LearnFromNearestNeighborsSelector()
        
    async def select_examples_method1(
        self,
        model: Any,
        candidate_examples: List[FewShotExample],
        query: str,
        n_lfm: int,
        n_lfnn: int
    ) -> List[FewShotExample]:
        """
        Method 1: Union of LFM set and query-specific nearest neighbors.
        
        Most cost-effective - LFM computed once, LFNN per query.
        """
        # Get LFM examples (query-agnostic, compute once)
        lfm_examples = await self.lfm_selector.select_examples_lfm(
            model, candidate_examples, query, n_lfm
        )
        
        # Get nearest neighbors for this query
        lfnn_examples = await self.lfnn_selector.select_examples_lfnn(
            candidate_examples, query, n_lfnn
        )
        
        # Combine (union)
        combined = list(set(lfm_examples + lfnn_examples))
        
        return combined
        
    async def select_examples_method2(
        self,
        model: Any,
        candidate_examples: List[FewShotExample],
        query: str,
        n_lfm: int,
        n_lfnn: int
    ) -> List[FewShotExample]:
        """
        Method 2: Use LFNN as initial examples for LFM.
        
        More resource-intensive but potentially more effective.
        """
        # Get nearest neighbors first
        lfnn_examples = await self.lfnn_selector.select_examples_lfnn(
            candidate_examples, query, n_lfnn
        )
        
        # Use as initial set for LFM
        lfm_examples = await self.lfm_selector.select_examples_lfm(
            model, candidate_examples, query, n_lfm,
            initial_set=lfnn_examples
        )
        
        return lfm_examples
        
    async def select_examples_method3(
        self,
        model: Any,
        candidate_examples: List[FewShotExample],
        query: str,
        n_lfm1: int,
        n_lfm2: int,
        n_lfnn: int
    ) -> List[FewShotExample]:
        """
        Method 3: Two-pass LFM with LFNN filtering.
        
        Most sophisticated - balances coverage and relevance.
        """
        # First LFM pass (query-agnostic)
        lfm1_examples = await self.lfm_selector.select_examples_lfm(
            model, candidate_examples, query, n_lfm1
        )
        
        # Get nearest neighbors
        lfnn_examples = await self.lfnn_selector.select_examples_lfnn(
            candidate_examples, query, n_lfnn
        )
        
        # Second LFM pass on nearest neighbors only
        lfm2_examples = await self.lfm_selector.select_examples_lfm(
            model, lfnn_examples, query, n_lfm2,
            initial_set=lfm1_examples
        )
        
        return lfm2_examples
```

### 7.5 Example Quality Scoring

```python
class ExampleQualityScorer:
    """
    Scores example quality for selection prioritization.
    """
    
    async def score_example(
        self,
        example: FewShotExample,
        context: ExecutionContext
    ) -> ExampleScore:
        """
        Calculate comprehensive quality score for example.
        """
        scores = {}
        
        # Success rate score
        scores['success_rate'] = example.success_rate
        
        # Diversity score (how different from other selected examples)
        scores['diversity'] = await self._calculate_diversity_score(example)
        
        # Difficulty match score
        scores['difficulty_match'] = self._calculate_difficulty_match(
            example, context
        )
        
        # Recency score (prefer newer examples)
        scores['recency'] = self._calculate_recency_score(example)
        
        # Usage balance score (prefer less-used examples)
        scores['usage_balance'] = 1.0 / (1 + example.usage_count * 0.1)
        
        # Combine scores
        weights = {
            'success_rate': 0.3,
            'diversity': 0.2,
            'difficulty_match': 0.2,
            'recency': 0.15,
            'usage_balance': 0.15
        }
        
        total_score = sum(
            scores[key] * weights[key]
            for key in weights
        )
        
        return ExampleScore(
            total=total_score,
            components=scores
        )
```

---

## 8. Prompt Versioning and Rollback

### 8.1 Version Control System

```python
class PromptVersionControl:
    """
    Git-like version control for prompts.
    """
    
    def __init__(self):
        self.storage = VersionStorage()
        
    class PromptVersion:
        """Represents a single version of a prompt."""
        
        def __init__(
            self,
            version_id: str,
            prompt_id: str,
            version_number: str,
            template: OptimizableTemplate,
            parent_version: Optional[str],
            author: str,
            commit_message: str,
            performance_snapshot: Dict,
            tags: List[str],
            timestamp: datetime
        ):
            self.version_id = version_id
            self.prompt_id = prompt_id
            self.version_number = version_number
            self.template = template
            self.parent_version = parent_version
            self.author = author
            self.commit_message = commit_message
            self.performance_snapshot = performance_snapshot
            self.tags = tags
            self.timestamp = timestamp
            
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
        # Get current version info
        current_version = await self.get_current_version(prompt_id)
        
        # Calculate new version number
        if current_version:
            new_version = self._increment_version(current_version.version_number)
            parent = current_version.version_id
        else:
            new_version = "1.0.0"
            parent = None
            
        # Capture performance snapshot
        performance = await self._capture_performance_snapshot(prompt_id)
        
        # Create version
        version = PromptVersion(
            version_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            version_number=new_version,
            template=copy.deepcopy(template),
            parent_version=parent,
            author=author,
            commit_message=commit_message,
            performance_snapshot=performance,
            tags=tags or [],
            timestamp=datetime.utcnow()
        )
        
        # Store version
        await self.storage.save_version(version)
        
        # Update current version pointer
        await self.storage.set_current_version(prompt_id, version.version_id)
        
        return version
        
    def _increment_version(self, current: str) -> str:
        """Increment semantic version number."""
        major, minor, patch = map(int, current.split('.'))
        
        # Simple increment strategy - can be made more sophisticated
        patch += 1
        if patch >= 10:
            patch = 0
            minor += 1
        if minor >= 10:
            minor = 0
            major += 1
            
        return f"{major}.{minor}.{patch}"
```

### 8.2 Rollback Mechanism

```python
class RollbackManager:
    """
    Manages prompt rollback to previous versions.
    """
    
    async def rollback(
        self,
        prompt_id: str,
        target_version: str,
        reason: str,
        automatic: bool = False
    ) -> RollbackResult:
        """
        Rollback prompt to a previous version.
        """
        # Get target version
        target = await self.version_control.get_version(prompt_id, target_version)
        
        if not target:
            return RollbackResult(
                success=False,
                error=f"Version {target_version} not found"
            )
            
        # Get current version for comparison
        current = await self.version_control.get_current_version(prompt_id)
        
        # Validate rollback is safe
        validation = await self._validate_rollback(current, target)
        
        if not validation.is_safe and not automatic:
            return RollbackResult(
                success=False,
                error="Rollback validation failed",
                details=validation.issues
            )
            
        # Create rollback record
        rollback_record = RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            from_version=current.version_number,
            to_version=target_version,
            reason=reason,
            automatic=automatic,
            timestamp=datetime.utcnow(),
            performance_comparison=await self._compare_performance(current, target)
        )
        
        # Perform rollback
        await self._apply_rollback(prompt_id, target)
        
        # Notify stakeholders
        await self._notify_rollback(rollback_record)
        
        return RollbackResult(
            success=True,
            rollback_record=rollback_record
        )
        
    async def auto_rollback_on_degradation(
        self,
        prompt_id: str,
        degradation_threshold: float = 0.15
    ):
        """
        Automatically rollback if performance degrades significantly.
        """
        # Get current performance
        current_perf = await self.metrics.get_current_performance(prompt_id)
        
        # Get baseline performance
        baseline = await self.metrics.get_baseline_performance(prompt_id)
        
        # Check for degradation
        if baseline and current_perf:
            degradation = (baseline.overall_score - current_perf.overall_score) / baseline.overall_score
            
            if degradation > degradation_threshold:
                # Find last stable version
                stable_version = await self._find_last_stable_version(prompt_id)
                
                if stable_version:
                    await self.rollback(
                        prompt_id=prompt_id,
                        target_version=stable_version,
                        reason=f"Auto-rollback due to {degradation:.1%} performance degradation",
                        automatic=True
                    )
```

### 8.3 Version Comparison

```python
class VersionComparator:
    """
    Compares different versions of prompts.
    """
    
    async def compare_versions(
        self,
        prompt_id: str,
        version1: str,
        version2: str
    ) -> VersionComparison:
        """
        Compare two versions of a prompt.
        """
        v1 = await self.version_control.get_version(prompt_id, version1)
        v2 = await self.version_control.get_version(prompt_id, version2)
        
        comparison = VersionComparison()
        
        # Compare template content
        comparison.template_diff = self._generate_diff(
            v1.template.template_str,
            v2.template.template_str
        )
        
        # Compare performance
        comparison.performance_diff = self._compare_performance(
            v1.performance_snapshot,
            v2.performance_snapshot
        )
        
        # Compare metadata
        comparison.metadata_diff = {
            'author_changed': v1.author != v2.author,
            'time_delta': (v2.timestamp - v1.timestamp).total_seconds(),
            'version_delta': self._calculate_version_delta(
                v1.version_number, v2.version_number
            )
        }
        
        # Generate summary
        comparison.summary = self._generate_comparison_summary(comparison)
        
        return comparison
        
    def _generate_diff(self, text1: str, text2: str) -> str:
        """Generate unified diff between two texts."""
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile='version1',
            tofile='version2',
            lineterm=''
        )
        
        return ''.join(diff)
```

### 8.4 Branching and Merging

```python
class BranchManager:
    """
    Manages branches for parallel prompt development.
    """
    
    async def create_branch(
        self,
        prompt_id: str,
        branch_name: str,
        from_version: str,
        description: str
    ) -> PromptBranch:
        """
        Create a new branch for prompt development.
        """
        branch = PromptBranch(
            branch_id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            branch_name=branch_name,
            base_version=from_version,
            description=description,
            created_at=datetime.utcnow(),
            commits=[]
        )
        
        await self.storage.save_branch(branch)
        return branch
        
    async def merge_branch(
        self,
        branch_id: str,
        target_version: str,
        merge_strategy: str = 'auto'
    ) -> MergeResult:
        """
        Merge a branch into mainline.
        """
        branch = await self.storage.get_branch(branch_id)
        
        # Get branch tip and target
        branch_tip = branch.commits[-1]
        target = await self.version_control.get_version(
            branch.prompt_id, target_version
        )
        
        # Check for conflicts
        conflicts = self._detect_conflicts(branch_tip, target)
        
        if conflicts and merge_strategy == 'auto':
            # Attempt automatic resolution
            resolved = await self._auto_resolve_conflicts(conflicts)
            
            if not resolved.all_resolved:
                return MergeResult(
                    success=False,
                    conflicts=conflicts,
                    message="Automatic merge failed, manual resolution required"
                )
                
        # Create merge commit
        merged_template = self._merge_templates(
            branch_tip.template,
            target.template,
            conflicts
        )
        
        merge_commit = await self.version_control.commit_version(
            prompt_id=branch.prompt_id,
            template=merged_template,
            commit_message=f"Merge branch '{branch.branch_name}' into {target_version}",
            tags=['merge', branch.branch_name]
        )
        
        return MergeResult(
            success=True,
            merge_commit=merge_commit,
            conflicts_resolved=len(conflicts) if conflicts else 0
        )
```

---

## Implementation Guide

### 9.1 Directory Structure

```
/context_prompt_engineering_loop/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── registry.py          # PromptRegistry
│   ├── context.py           # ContextEngine
│   └── config.py            # Configuration
├── performance/
│   ├── __init__.py
│   ├── tracker.py           # PerformanceTracker
│   ├── aggregator.py        # PerformanceAggregator
│   ├── collector.py         # RealTimePerformanceCollector
│   └── dashboard.py         # PerformanceDashboard
├── optimization/
│   ├── __init__.py
│   ├── optimizer.py         # TemplateOptimizer
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── clarity.py       # ClarityEnhancementStrategy
│   │   ├── specificity.py   # SpecificityBoostStrategy
│   │   ├── examples.py      # ExampleOptimizationStrategy
│   │   └── format.py        # FormatOptimizationStrategy
│   └── modifier.py          # ContextBasedModifier
├── testing/
│   ├── __init__.py
│   ├── ab_test.py           # ABTestFramework
│   ├── router.py            # TrafficRouter
│   ├── statistics.py        # StatisticsEngine
│   └── bandit.py            # MultiArmedBanditRouter
├── assembly/
│   ├── __init__.py
│   ├── assembler.py         # DynamicPromptAssembler
│   ├── selector.py          # ComponentSelector
│   ├── resolver.py          # VariableResolver
│   └── strategies.py        # Assembly strategies
├── examples/
│   ├── __init__.py
│   ├── database.py          # ExampleDatabase
│   ├── lfm.py               # LearnFromMistakesSelector
│   ├── lfnn.py              # LearnFromNearestNeighborsSelector
│   ├── combined.py          # CombinedExampleSelector
│   └── quality.py           # ExampleQualityScorer
├── versioning/
│   ├── __init__.py
│   ├── control.py           # PromptVersionControl
│   ├── rollback.py          # RollbackManager
│   ├── compare.py           # VersionComparator
│   └── branch.py            # BranchManager
├── metrics/
│   ├── __init__.py
│   ├── framework.py         # EffectivenessMetricsFramework
│   ├── evaluator.py         # AutomatedQualityEvaluator
│   └── correlation.py       # MetricCorrelationAnalyzer
└── utils/
    ├── __init__.py
    ├── embeddings.py        # Embedding utilities
    ├── similarity.py        # Similarity calculations
    └── tokenization.py      # Token counting
```

### 9.2 Configuration Schema

```yaml
# config/cpel_config.yaml

context_prompt_engineering_loop:
  # Performance tracking settings
  performance:
    collection_interval: 60  # seconds
    aggregation_windows: ['1h', '6h', '24h', '7d', '30d']
    retention_days: 90
    
  # Optimization settings
  optimization:
    auto_optimize: true
    optimization_interval: 3600  # seconds
    min_samples_for_optimization: 100
    improvement_threshold: 0.05
    
  # A/B testing settings
  ab_testing:
    default_confidence_level: 0.95
    min_sample_size: 500
    max_test_duration_days: 14
    enable_bandit: true
    exploration_rate: 0.1
    
  # Assembly settings
  assembly:
    default_strategy: 'adaptive'
    max_prompt_tokens: 4000
    context_token_budget: 2000
    
  # Example selection settings
  example_selection:
    default_method: 'combined_method1'
    max_examples: 5
    min_similarity_threshold: 0.7
    
  # Versioning settings
  versioning:
    auto_commit_on_change: true
    max_versions_per_prompt: 100
    auto_rollback_on_degradation: true
    degradation_threshold: 0.15
```

### 9.3 Integration Interface

```python
class ContextPromptEngineeringLoop:
    """
    Main interface for the Context Prompt Engineering Loop.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.registry = PromptRegistry()
        self.context_engine = ContextEngine()
        self.performance_tracker = PerformanceTracker()
        self.optimizer = TemplateOptimizer()
        self.ab_tester = ABTestFramework()
        self.assembler = DynamicPromptAssembler()
        self.example_selector = CombinedExampleSelector()
        self.version_control = PromptVersionControl()
        self.rollback_manager = RollbackManager()
        
    async def get_optimized_prompt(
        self,
        task_type: str,
        user_input: str,
        conversation_history: List[Dict] = None,
        system_state: Dict = None
    ) -> OptimizedPrompt:
        """
        Main entry point: Get an optimized prompt for the given context.
        """
        # Detect context
        context = await self.context_engine.detect_context(
            user_input, conversation_history or [], system_state or {}
        )
        
        # Get base template
        base_template = await self.registry.get_template_for_task(task_type, context)
        
        # Check for active A/B test
        test_variant = await self.ab_tester.get_variant_if_testing(
            base_template.template_id,
            context
        )
        
        if test_variant:
            template = test_variant
        else:
            template = base_template
            
        # Apply context-based modifications
        modified = self.optimizer.modifier.apply_context_modifications(
            template.template_str,
            context
        )
        
        # Select few-shot examples
        examples = await self.example_selector.select_examples(
            task_type=task_type,
            query=user_input,
            context=context,
            n_examples=self.config['example_selection']['max_examples']
        )
        
        # Assemble final prompt
        assembled = await self.assembler.assemble_prompt(
            task_type=task_type,
            context=context,
            user_input=user_input,
            base_template=modified,
            examples=examples
        )
        
        # Track usage
        await self.performance_tracker.track_prompt_usage(
            template.template_id,
            context
        )
        
        return OptimizedPrompt(
            prompt=assembled.prompt,
            template_id=template.template_id,
            variant_id=test_variant.variant_id if test_variant else None,
            context_used=context,
            examples_used=examples
        )
        
    async def record_outcome(
        self,
        prompt_id: str,
        execution_result: ExecutionResult
    ):
        """
        Record the outcome of prompt execution for optimization.
        """
        # Collect performance metrics
        metrics = await self.performance_tracker.collect_execution_metrics(
            prompt_id=prompt_id,
            rendered_prompt=execution_result.prompt_used,
            context=execution_result.context,
            llm_response=execution_result.response,
            execution_time_ms=execution_result.execution_time_ms
        )
        
        # Check if optimization is needed
        await self._check_and_trigger_optimization(prompt_id)
        
    async def _check_and_trigger_optimization(self, prompt_id: str):
        """
        Check if prompt needs optimization and trigger if needed.
        """
        # Get recent performance
        recent = await self.performance_tracker.aggregate_performance(
            prompt_id=prompt_id,
            time_window='24h',
            dimensions=['response_quality', 'token_efficiency']
        )
        
        # Check against baseline
        template = await self.registry.get_template(prompt_id)
        
        if recent.overall_score < template.performance_baseline * 0.95:
            # Performance degradation detected
            await self.optimizer.run_optimization_pipeline(template, recent)
```

---

## Integration with Agent Loops

### 10.1 Integration Points

The Context Prompt Engineering Loop integrates with the broader OpenClaw-inspired agent system through:

1. **Heartbeat Loop**: CPEL reports optimization status and performance metrics
2. **Soul Loop**: CPEL contributes to agent's evolving "personality" through prompt refinement
3. **Identity Loop**: CPEL maintains consistent identity across prompt versions
4. **User Loop**: CPEL adapts prompts based on user preferences and history
5. **Task Loops**: CPEL optimizes prompts for specific task types (browser, email, voice, etc.)

### 10.2 Event-Driven Architecture

```python
class CPELEventBus:
    """
    Event bus for CPEL integration with other agent loops.
    """
    
    EVENTS = {
        'prompt_optimized': 'Fired when a prompt is optimized',
        'ab_test_started': 'Fired when A/B test begins',
        'ab_test_completed': 'Fired when A/B test ends',
        'performance_degradation': 'Fired when performance drops',
        'rollback_executed': 'Fired when rollback occurs',
        'new_example_added': 'Fired when few-shot example is added'
    }
    
    async def emit(self, event_type: str, data: Dict):
        """Emit event to all subscribed loops."""
        # Implementation for event distribution
        pass
```

---

## Conclusion

The Context Prompt Engineering Loop provides a comprehensive, autonomous system for prompt optimization in the Windows 10 OpenClaw-inspired AI agent. By combining:

- **Performance tracking** with comprehensive metrics
- **Context-aware adjustment** for dynamic adaptation
- **Template optimization** with multiple strategies
- **A/B testing** with statistical rigor
- **Dynamic assembly** for flexible prompt construction
- **Intelligent example selection** using LFM and LFNN algorithms
- **Version control** with rollback capabilities

CPEL enables the agent to continuously improve its prompting capabilities without human intervention, leading to better performance, higher user satisfaction, and more efficient operation.

---

## References

1. Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models (2025)
2. On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection (2025)
3. Evidently AI: Automated Prompt Optimization (2026)
4. Best Practices for A/B Testing AI Model Prompts (2026)
5. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
6. In-Context Learning and Example Selection (Brown et al., 2020)

---

*Document Version: 1.0.0*
*Target Platform: Windows 10*
*LLM Target: GPT-5.2 with Extra High Thinking Capability*
