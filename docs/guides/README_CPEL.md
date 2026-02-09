# Context Prompt Engineering Loop (CPEL)

## Windows 10 OpenClaw-Inspired AI Agent System

---

## Overview

The Context Prompt Engineering Loop (CPEL) is an autonomous, self-improving subsystem designed to continuously optimize prompts for AI agents. Built for Windows 10 environments and optimized for GPT-5.2 with extra high thinking capability, CPEL implements a closed-loop feedback system that tracks prompt performance, analyzes contextual effectiveness, and automatically refines prompt templates based on real-world usage patterns.

---

## Features

### 1. Prompt Performance Tracking
- **Real-time metrics collection** during prompt execution
- **Multi-dimensional quality scoring** (accuracy, relevance, completeness, coherence)
- **Efficiency metrics** (token usage, latency, cost)
- **User satisfaction tracking** (explicit and implicit feedback)
- **Performance aggregation** across configurable time windows

### 2. Context-Aware Prompt Adjustment
- **Automatic context detection** (task type, complexity, urgency, domain, emotional tone)
- **Dynamic prompt modification** based on detected context
- **Context injection** for improved relevance
- **Conversation continuity** for multi-turn interactions

### 3. Template Optimization Strategies
- **Clarity enhancement** for ambiguous instructions
- **Specificity boost** for vague requirements
- **Example optimization** for better few-shot learning
- **Format optimization** for structured outputs
- **Constraint refinement** for better adherence
- **Multi-strategy optimization pipeline**

### 4. A/B Testing Framework
- **Statistical rigor** with configurable confidence levels
- **Multi-armed bandit** for dynamic traffic allocation
- **Automatic winner detection** with effect size analysis
- **Traffic routing** with consistent user assignment
- **Test orchestration** and lifecycle management

### 5. Prompt Effectiveness Metrics
- **Comprehensive scoring** across 4 categories:
  - Quality (accuracy, relevance, completeness, coherence, helpfulness)
  - Efficiency (token efficiency, latency, cost)
  - User Satisfaction (ratings, retry rates, corrections)
  - Robustness (consistency, edge case handling)
- **Automated quality evaluation** using GPT-5.2
- **Correlation analysis** between metrics

### 6. Dynamic Prompt Assembly
- **Modular component system** for flexible construction
- **Context-aware assembly strategies**
- **Token budget management**
- **Variable resolution** with custom resolvers
- **Component prioritization** within constraints

### 7. Few-Shot Example Selection
- **Multiple selection algorithms**:
  - Similarity-based selection
  - Diversity maximization
  - Success-rate prioritization
  - Hybrid scoring
  - Learn-from-Mistakes (LFM)
  - Learn-from-Nearest-Neighbors (LFNN)
  - Combined methods
- **Example quality scoring**
- **Dynamic example database**

### 8. Prompt Versioning and Rollback
- **Git-like version control** with semantic versioning
- **Automatic commit** on template changes
- **Performance-based rollback** triggers
- **Version comparison** with diff generation
- **Branching and merging** support

---

## Architecture

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

---

## Installation

### Prerequisites
- Python 3.9+
- Windows 10
- GPT-5.2 API access

### Dependencies
```bash
pip install numpy pandas pyyaml asyncio
```

### Setup
1. Clone the repository
2. Copy `cpel_config.yaml` to your config directory
3. Initialize the CPEL system

```python
from cpel_implementation import ContextPromptEngineeringLoop

cpel = ContextPromptEngineeringLoop()
await cpel.initialize()
```

---

## Usage

### Basic Usage

```python
# Get optimized prompt for a user request
optimized = await cpel.get_optimized_prompt(
    user_input="Search for AI news and summarize",
    conversation_history=[],
    system_state={'session_id': 'session_123'}
)

print(optimized.prompt)
```

### Recording Results

```python
# Record execution outcome for optimization
metrics = await cpel.record_execution_result(
    prompt_id=optimized.template_id,
    rendered_prompt=optimized.prompt,
    context=optimized.context_used,
    response=llm_response,
    execution_time_ms=2500,
    llm_interface=gpt52_interface
)
```

### Creating A/B Tests

```python
# Create A/B test for template optimization
test_id = await cpel.create_ab_test(
    template_id="browser_task",
    variant_templates=[
        "Variant 1: Focus on speed",
        "Variant 2: Focus on accuracy"
    ],
    test_name="Browser Task Optimization"
)
```

### Performance Reports

```python
# Get performance report
report = await cpel.get_performance_report(
    prompt_id="browser_task",
    time_window='24h'
)

print(f"Overall Score: {report['overall_score']}")
print(f"Sample Count: {report['sample_count']}")
```

---

## Configuration

See `cpel_config.yaml` for all configuration options:

```yaml
context_prompt_engineering_loop:
  performance:
    collection_interval: 60
    aggregation_windows: ['1h', '6h', '24h', '7d', '30d']
    
  optimization:
    auto_optimize: true
    optimization_interval: 3600
    
  ab_testing:
    default_confidence_level: 0.95
    enable_bandit: true
```

---

## Integration with Agent Loops

CPEL integrates with the broader OpenClaw-inspired agent system:

### Heartbeat Loop
- Reports optimization status every 5 minutes
- Shares performance metrics with monitoring

### Soul Loop
- Contributes to agent's evolving personality
- Maintains consistent voice across optimizations

### Identity Loop
- Ensures prompt changes don't affect core identity
- Version control tracks identity-related changes

### User Loop
- Adapts to individual user preferences
- Learns from user feedback and corrections

### Task Loops
- Optimizes prompts per task type (browser, email, voice, etc.)
- Task-specific example selection

---

## API Reference

### ContextPromptEngineeringLoop

Main interface class for the CPEL system.

#### Methods

##### `async initialize()`
Initialize the CPEL system and load default templates.

##### `async get_optimized_prompt(user_input, task_type, conversation_history, system_state, llm_interface)`
Get an optimized prompt for the given context.

**Parameters:**
- `user_input` (str): The user's request or query
- `task_type` (TaskType, optional): Explicit task type
- `conversation_history` (List[Dict], optional): Previous conversation turns
- `system_state` (Dict, optional): Current system state
- `llm_interface` (Any, optional): Interface to LLM for quality evaluation

**Returns:** `OptimizedPrompt`

##### `async record_execution_result(prompt_id, rendered_prompt, context, response, execution_time_ms, llm_interface)`
Record the outcome of prompt execution for optimization.

**Returns:** `PromptMetrics`

##### `async create_ab_test(template_id, variant_templates, test_name)`
Create an A/B test for a template.

**Returns:** `str` (test_id)

##### `async get_performance_report(prompt_id, time_window)`
Get performance report for a prompt.

**Returns:** `Dict`

---

## Performance Metrics

### Quality Metrics
- **Accuracy**: Factual correctness (0-1)
- **Relevance**: Addresses the specific query (0-1)
- **Completeness**: Fully answers the question (0-1)
- **Coherence**: Logical flow and clarity (0-1)
- **Helpfulness**: Practical utility (0-1)

### Efficiency Metrics
- **Token Efficiency**: Output quality per token
- **Latency**: Response time in milliseconds
- **Cost per Request**: Estimated API cost

### User Satisfaction Metrics
- **Explicit Rating**: User-provided rating (1-5)
- **Implicit Satisfaction**: Derived from behavior
- **Retry Rate**: How often users retry
- **Correction Rate**: How often users correct

---

## Optimization Strategies

### Clarity Enhancement
- Identifies ambiguous instructions
- Simplifies complex language
- Adds clarifying examples
- Removes redundancy

### Specificity Boost
- Adds concrete requirements
- Specifies output format
- Defines success criteria
- Adds constraints

### Example Optimization
- Selects relevant examples
- Balances positive/negative examples
- Orders examples strategically
- Updates examples based on performance

### Format Optimization
- Structures output clearly
- Adds formatting instructions
- Defines sections
- Specifies response length

---

## A/B Testing

### Statistical Methods
- **T-test**: Compare means between variants
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: Uncertainty quantification
- **Multi-Armed Bandit**: Dynamic traffic allocation

### Test Lifecycle
1. **Creation**: Define hypothesis and variants
2. **Running**: Route traffic and collect data
3. **Analysis**: Statistical significance testing
4. **Decision**: Promote winner or continue testing
5. **Cleanup**: Archive test and apply learnings

---

## Version Control

### Semantic Versioning
- **Major**: Breaking changes to prompt behavior
- **Minor**: New features or significant improvements
- **Patch**: Bug fixes or minor adjustments

### Operations
- **Commit**: Save new version with message
- **Rollback**: Revert to previous version
- **Compare**: Diff between versions
- **Branch**: Parallel development tracks
- **Merge**: Combine branch changes

---

## Troubleshooting

### Common Issues

#### Performance Not Improving
- Check `min_samples_for_optimization` setting
- Verify metrics are being collected
- Review optimization strategies

#### A/B Test Not Completing
- Check `min_sample_size` configuration
- Verify traffic routing is working
- Review statistical parameters

#### Context Detection Inaccurate
- Review context detection thresholds
- Add custom domain keywords
- Tune complexity scoring

### Debug Mode
```python
import logging
logging.getLogger('cpel').setLevel(logging.DEBUG)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License - See LICENSE file

---

## References

1. Promptomatix: An Automatic Prompt Optimization Framework (2025)
2. On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection (2025)
3. Evidently AI: Automated Prompt Optimization (2026)
4. Best Practices for A/B Testing AI Model Prompts (2026)
5. DSPy: Compiling Declarative Language Model Calls (2023)

---

## Contact

For questions or support, please open an issue on GitHub.

---

*Version: 1.0.0*
*Target Platform: Windows 10*
*LLM Target: GPT-5.2 with Extra High Thinking Capability*
