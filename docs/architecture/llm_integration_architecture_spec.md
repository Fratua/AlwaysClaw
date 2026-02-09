# LLM Integration and Reasoning Engine Architecture
## Windows 10 OpenClaw-Inspired AI Agent System

**Version:** 1.0  
**Date:** 2025  
**Classification:** Technical Specification

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Model Abstraction Layer](#model-abstraction-layer)
4. [Prompt Engineering System](#prompt-engineering-system)
5. [Chain-of-Thought & Reasoning Pipeline](#chain-of-thought--reasoning-pipeline)
6. [Extra High Thinking Mode](#extra-high-thinking-mode)
7. [Multi-Model Fallback & Routing](#multi-model-fallback--routing)
8. [Token Usage Optimization](#token-usage-optimization)
9. [Response Parsing & Action Extraction](#response-parsing--action-extraction)
10. [Streaming Response Handling](#streaming-response-handling)
11. [Integration Patterns](#integration-patterns)
12. [Configuration Reference](#configuration-reference)

---

## Executive Summary

This document defines the complete LLM Integration and Reasoning Engine Architecture for a Windows 10-focused OpenClaw-inspired AI agent system. The architecture leverages GPT-5.2 with EXTRA HIGH thinking capability as the primary reasoning engine while maintaining model-agnostic flexibility for future extensibility.

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Model Agnostic** | Support GPT-5.2, Claude, Gemini, and local models through unified interface |
| **Reasoning-First** | Deep chain-of-thought processing for complex agentic tasks |
| **Cost-Aware** | Intelligent routing and token optimization for sustainable operations |
| **Streaming-First** | Real-time response handling for interactive user experience |
| **Fault-Tolerant** | Multi-layer fallback systems for 24/7 reliability |

---

## Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ORCHESTRATION LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Intent    │  │   Planning  │  │  Execution  │  │   Memory        │   │
│  │  Interpreter│  │   Engine    │  │  Controller │  │   Manager       │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────────┘   │
└─────────┼────────────────┼────────────────┼────────────────────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────────┐
│                    LLM INTEGRATION LAYER                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MODEL ABSTRACTION LAYER                           │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│  │  │ GPT-5.2  │ │ Claude   │ │ Gemini   │ │ Local    │ │ Fallback │   │  │
│  │  │ Provider │ │ Provider │ │ Provider │ │ Provider │ │ Provider │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    REASONING ENGINE                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Chain-of-    │  │ Extra High   │  │ Response     │              │  │
│  │  │ Thought      │  │ Thinking     │  │ Validator    │              │  │
│  │  │ Pipeline     │  │ Controller   │  │              │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PROMPT ENGINEERING SYSTEM                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Template     │  │ Dynamic      │  │ Context      │              │  │
│  │  │ Manager      │  │ Injector     │  │ Assembler    │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Abstraction Layer

### 3.1 Provider Interface Design

```python
# core/llm/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Union, Callable
import asyncio

class ModelCapability(Enum):
    """Model capabilities for capability-based routing"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"
    EXTRA_HIGH_THINKING = "extra_high_thinking"

class ReasoningLevel(Enum):
    """Reasoning effort levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTRA_HIGH = "xhigh"

@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    enable_streaming: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model_id: str
    usage: Dict[str, int]
    reasoning_content: Optional[str] = None
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False

@dataclass
class LLMRequest:
    """Standardized LLM request"""
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict]] = None
    reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    stream: bool = False
    max_tokens: int = 4096
    temperature: float = 0.7
    metadata: Optional[Dict] = None

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        pass
    
    @abstractmethod
    def get_pricing(self) -> Dict[str, float]:
        pass
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        return capability in self.get_capabilities()
```

### 3.2 GPT-5.2 Provider Implementation

```python
# core/llm/providers/gpt52_provider.py
import openai
from typing import AsyncIterator
import tiktoken

class GPT52Provider(BaseLLMProvider):
    """GPT-5.2 Provider with EXTRA HIGH thinking support."""
    
    MODEL_MAPPING = {
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-instant": "gpt-5.2-chat-latest",
        "gpt-5.2-pro": "gpt-5.2-pro",
    }
    
    REASONING_LEVEL_MAP = {
        ReasoningLevel.NONE: "none",
        ReasoningLevel.LOW: "low",
        ReasoningLevel.MEDIUM: "medium",
        ReasoningLevel.HIGH: "high",
        ReasoningLevel.EXTRA_HIGH: "xhigh",
    }
    
    PRICING = {
        "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
        "gpt-5.2-chat-latest": {"input": 0.50, "output": 2.00},
    }
    
    def _initialize_client(self) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://api.openai.com/v1",
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        self._encoder = tiktoken.encoding_for_model("gpt-4")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        import time
        start_time = time.time()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)
        
        model_id = self.MODEL_MAPPING.get(self.config.model_id, "gpt-5.2")
        
        params = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        
        if "gpt-5.2" in model_id and "chat" not in model_id:
            params["reasoning_effort"] = self.REASONING_LEVEL_MAP.get(
                request.reasoning_level, "medium"
            )
        
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = "auto"
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.chat.completions.create(**params)
                break
            except openai.RateLimitError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
        
        reasoning_content = None
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model_id=model_id,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            reasoning_content=reasoning_content,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency,
        )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        import time
        start_time = time.time()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)
        
        model_id = self.MODEL_MAPPING.get(self.config.model_id, "gpt-5.2")
        
        params = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        
        if "gpt-5.2" in model_id and "chat" not in model_id:
            params["reasoning_effort"] = self.REASONING_LEVEL_MAP.get(
                request.reasoning_level, "medium"
            )
        
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = "auto"
        
        stream = await self._client.chat.completions.create(**params)
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            yield LLMResponse(
                content=delta.content or "",
                model_id=model_id,
                usage={"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
                reasoning_content=delta.reasoning_content if hasattr(delta, 'reasoning_content') else None,
                finish_reason=chunk.choices[0].finish_reason or "",
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def get_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.EXTRA_HIGH_THINKING,
        ]
    
    def estimate_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))
    
    def get_pricing(self) -> Dict[str, float]:
        return self.PRICING.get(self.config.model_id, self.PRICING["gpt-5.2"])
```

---

## Prompt Engineering System

### 4.1 Core System Prompt Template

```yaml
# templates/prompts/system_base.yaml
name: system_base
description: Base system prompt for OpenClaw Windows agent
version: "1.0"
variables:
  - agent_name
  - agent_identity
  - capabilities
  - current_time
  - user_context
required_variables:
  - agent_name
  - agent_identity
template: |
  You are {{agent_name}}, {{agent_identity}}.
  
  ## Core Identity
  You are an autonomous AI agent running on Windows 10, designed to assist users through natural conversation and direct system interaction. You have a persistent identity, memory, and the ability to execute tasks autonomously.
  
  ## Current Context
  - Current Time: {{current_time}}
  - User Context: {{user_context}}
  
  ## Available Capabilities
  {{capabilities}}
  
  ## Operating Principles
  1. **Autonomy**: Take initiative when appropriate, don't wait for explicit permission for low-risk actions
  2. **Transparency**: Always explain what you're doing and why
  3. **Safety**: Never execute destructive operations without explicit confirmation
  4. **Memory**: Reference previous conversations and learned preferences
  5. **Proactivity**: Anticipate user needs based on patterns and context
  
  ## Response Format
  When taking actions, use the following format:
  ```
  THOUGHT: [Your reasoning about what to do]
  ACTION: [The specific action to take]
  PARAMS: [JSON parameters for the action]
  ```
  
  Always be helpful, accurate, and efficient in your responses.
```

### 4.2 Intent Analysis Template

```yaml
# templates/prompts/intent_analysis.yaml
name: intent_analysis
description: Analyze user intent and determine appropriate response strategy
version: "1.0"
variables:
  - user_input
  - conversation_history
  - available_tools
required_variables:
  - user_input
  - available_tools
template: |
  Analyze the following user input and determine the intent, required actions, and response strategy.
  
  ## User Input
  {{user_input}}
  
  ## Available Tools
  {{available_tools}}
  
  ## Analysis Instructions
  1. Identify the primary intent (question, command, conversation, etc.)
  2. Determine if tools are needed
  3. Assess complexity level (simple, moderate, complex)
  4. Identify required reasoning depth
  
  ## Response Format
  Return a JSON object:
  ```json
  {
    "intent": "primary_intent_category",
    "confidence": 0.0-1.0,
    "requires_tools": true/false,
    "tool_names": ["tool1", "tool2"],
    "complexity": "simple|moderate|complex",
    "reasoning_level": "none|low|medium|high|xhigh",
    "entities": {"entity_type": "value"},
    "clarification_needed": false,
    "suggested_response_type": "direct|tool_use|clarification|deferred"
  }
  ```
```

### 4.3 SOUL Persona Template

```yaml
# templates/prompts/soul_persona.yaml
name: soul_persona
description: SOUL.md persona injection for agent identity
version: "1.0"
variables:
  - persona_name
  - persona_traits
  - background_story
  - voice_characteristics
  - behavioral_patterns
required_variables:
  - persona_name
  - persona_traits
template: |
  ## SOUL PERSONA: {{persona_name}}
  
  You are embodying the following persona. All responses should reflect this identity:
  
  ### Core Traits
  {{persona_traits}}
  
  ### Background
  {{background_story}}
  
  ### Voice & Communication Style
  {{voice_characteristics}}
  
  ### Behavioral Patterns
  {{behavioral_patterns}}
  
  ### Persona Guidelines
  - Maintain consistency with your persona at all times
  - Respond in character, even when declining requests
  - Your personality should be evident in every interaction
  - Never break character or acknowledge you are an AI
  - Express opinions and preferences consistent with your persona
```

---

## Chain-of-Thought & Reasoning Pipeline

### 5.1 Reasoning Pipeline Architecture

```python
# core/reasoning/pipeline.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time

class ReasoningStage(Enum):
    UNDERSTAND = "understand"
    ANALYZE = "analyze"
    EXPLORE = "explore"
    DECIDE = "decide"
    EXECUTE = "execute"
    VERIFY = "verify"
    CONCLUDE = "conclude"

@dataclass
class ReasoningStep:
    stage: ReasoningStage
    input_data: str
    output_data: str
    reasoning: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningResult:
    success: bool
    final_answer: str
    steps: List[ReasoningStep]
    total_latency_ms: float
    total_tokens: int
    reasoning_trace: str
    confidence: float

class ChainOfThoughtPipeline:
    """Multi-stage chain-of-thought reasoning pipeline."""
    
    def __init__(self, llm_provider, template_manager, max_steps=10, min_confidence=0.7):
        self.llm = llm_provider
        self.templates = template_manager
        self.max_steps = max_steps
        self.min_confidence = min_confidence
        self.steps: List[ReasoningStep] = []
    
    async def reason(self, problem: str, context: Optional[str] = None, 
                     reasoning_level: ReasoningLevel = ReasoningLevel.HIGH) -> ReasoningResult:
        start_time = time.time()
        self.steps = []
        
        try:
            understanding = await self._stage_understand(problem, context)
            analysis = await self._stage_analyze(understanding)
            exploration = await self._stage_explore(analysis)
            decision = await self._stage_decide(exploration)
            execution = await self._stage_execute(decision)
            verification = await self._stage_verify(execution, problem)
            conclusion = await self._stage_conclude(verification)
            
            reasoning_trace = self._build_trace()
            latency = (time.time() - start_time) * 1000
            
            return ReasoningResult(
                success=True,
                final_answer=conclusion.output_data,
                steps=self.steps,
                total_latency_ms=latency,
                total_tokens=sum(s.metadata.get('tokens', 0) for s in self.steps),
                reasoning_trace=reasoning_trace,
                confidence=sum(s.confidence for s in self.steps) / len(self.steps)
            )
        except Exception as e:
            return ReasoningResult(
                success=False,
                final_answer="",
                steps=self.steps,
                total_latency_ms=(time.time() - start_time) * 1000,
                total_tokens=0,
                reasoning_trace=self._build_trace(),
                confidence=0.0
            )
    
    async def _stage_understand(self, problem: str, context: Optional[str]) -> ReasoningStep:
        prompt = f"""Understand this problem:
{problem}

Context: {context or 'None'}

1. Restate the problem in your own words
2. Identify what is being asked
3. Note any constraints or requirements
4. Identify the domain and relevant knowledge"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=1000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.UNDERSTAND,
            input_data=problem,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.8,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_analyze(self, understanding: ReasoningStep) -> ReasoningStep:
        prompt = f"""Analyze this problem:
{understanding.output_data}

1. What are the key elements involved?
2. What are the relationships between elements?
3. What constraints must be satisfied?
4. What resources are available?"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=1000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.ANALYZE,
            input_data=understanding.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.75,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_explore(self, analysis: ReasoningStep) -> ReasoningStep:
        prompt = f"""Explore approaches for:
{analysis.output_data}

1. List 2-3 different ways to solve this
2. For each approach, note pros and cons
3. Consider edge cases for each
4. Estimate complexity of each approach"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=1500
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.EXPLORE,
            input_data=analysis.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.7,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_decide(self, exploration: ReasoningStep) -> ReasoningStep:
        prompt = f"""Make a decision based on:
{exploration.output_data}

1. Which approach is best and why?
2. What are the key steps to implement it?
3. What could go wrong and how to handle it?
4. What success looks like"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=1000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.DECIDE,
            input_data=exploration.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.85,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_execute(self, decision: ReasoningStep) -> ReasoningStep:
        prompt = f"""Execute the decided approach:
{decision.output_data}

Provide the complete solution with all steps worked out."""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=2000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.EXECUTE,
            input_data=decision.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.8,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_verify(self, execution: ReasoningStep, original_problem: str) -> ReasoningStep:
        prompt = f"""Verify the solution:

Original Problem: {original_problem}

Proposed Solution:
{execution.output_data}

1. Does this solve the original problem?
2. Are there any errors or issues?
3. Does it satisfy all constraints?
4. What is your confidence level (0-1)?"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.HIGH,
            max_tokens=1000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.VERIFY,
            input_data=execution.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.75,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    async def _stage_conclude(self, verification: ReasoningStep) -> ReasoningStep:
        prompt = f"""Provide final conclusion:
{verification.output_data}

1. Clear, concise solution
2. Key points to remember
3. Any caveats or limitations"""
        
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            reasoning_level=ReasoningLevel.MEDIUM,
            max_tokens=1000
        )
        
        response = await self.llm.generate(request)
        
        step = ReasoningStep(
            stage=ReasoningStage.CONCLUDE,
            input_data=verification.output_data,
            output_data=response.content,
            reasoning=response.reasoning_content or "",
            confidence=0.9,
            timestamp=time.time(),
            metadata={"tokens": response.usage["total_tokens"]}
        )
        self.steps.append(step)
        return step
    
    def _build_trace(self) -> str:
        trace = []
        for i, step in enumerate(self.steps, 1):
            trace.append(f"\n{'='*60}")
            trace.append(f"Step {i}: {step.stage.value.upper()}")
            trace.append(f"{'='*60}")
            trace.append(f"Confidence: {step.confidence:.2f}")
            trace.append(f"\nOutput:\n{step.output_data}")
            if step.reasoning:
                trace.append(f"\nReasoning:\n{step.reasoning}")
        return "\n".join(trace)
```

---

## Extra High Thinking Mode

### 6.1 X-High Configuration

```python
# core/reasoning/xhigh_controller.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio

class XHighTriggerCondition(Enum):
    COMPLEX_CODE = "complex_code"
    MULTI_STEP_PLANNING = "multi_step_planning"
    RESEARCH_TASK = "research_task"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    PROBLEM_SOLVING = "problem_solving"
    USER_EXPLICIT = "user_explicit"
    HIGH_STAKES = "high_stakes"
    AMBIGUOUS_CONTEXT = "ambiguous_context"

@dataclass
class XHighConfig:
    enabled: bool = True
    auto_trigger: bool = True
    max_thinking_time_seconds: int = 300
    min_confidence_threshold: float = 0.9
    show_thinking_trace: bool = True
    compact_trace: bool = True
    trigger_weights: Dict[XHighTriggerCondition, float] = None
    
    def __post_init__(self):
        if self.trigger_weights is None:
            self.trigger_weights = {
                XHighTriggerCondition.COMPLEX_CODE: 0.9,
                XHighTriggerCondition.MULTI_STEP_PLANNING: 0.85,
                XHighTriggerCondition.RESEARCH_TASK: 0.8,
                XHighTriggerCondition.DATA_ANALYSIS: 0.8,
                XHighTriggerCondition.CREATIVE_WRITING: 0.7,
                XHighTriggerCondition.PROBLEM_SOLVING: 0.75,
                XHighTriggerCondition.USER_EXPLICIT: 1.0,
                XHighTriggerCondition.HIGH_STAKES: 0.95,
                XHighTriggerCondition.AMBIGUOUS_CONTEXT: 0.6,
            }
```

### 6.2 X-High Controller Implementation

```python
class XHighThinkingController:
    """Controller for EXTRA HIGH thinking mode integration."""
    
    COMPLEXITY_INDICATORS = {
        "code": ["refactor", "architecture", "design pattern", "optimize", "debug complex"],
        "planning": ["create a plan", "strategy", "roadmap", "workflow", "automation"],
        "research": ["research", "analyze", "investigate", "compare", "evaluate options"],
        "data": ["dataset", "analyze data", "visualization", "statistics", "correlation"],
        "creative": ["write a story", "creative", "brainstorm", "design", "concept"],
        "problem": ["solve", "figure out", "troubleshoot", "diagnose", "root cause"],
        "high_stakes": ["important", "critical", "production", "deploy", "release"],
    }
    
    def __init__(self, llm_provider, config: XHighConfig = None):
        self.llm = llm_provider
        self.config = config or XHighConfig()
        self.thinking_history: List[Dict] = []
    
    def should_use_xhigh(self, user_input: str, context: Optional[Dict] = None) -> tuple[bool, List[XHighTriggerCondition]]:
        if not self.config.enabled:
            return False, []
        
        triggered = []
        input_lower = user_input.lower()
        
        if "think deeply" in input_lower or "extra high" in input_lower:
            triggered.append(XHighTriggerCondition.USER_EXPLICIT)
        
        for category, keywords in self.COMPLEXITY_INDICATORS.items():
            for keyword in keywords:
                if keyword in input_lower:
                    condition_map = {
                        "code": XHighTriggerCondition.COMPLEX_CODE,
                        "planning": XHighTriggerCondition.MULTI_STEP_PLANNING,
                        "research": XHighTriggerCondition.RESEARCH_TASK,
                        "data": XHighTriggerCondition.DATA_ANALYSIS,
                        "creative": XHighTriggerCondition.CREATIVE_WRITING,
                        "problem": XHighTriggerCondition.PROBLEM_SOLVING,
                        "high_stakes": XHighTriggerCondition.HIGH_STAKES,
                    }
                    condition = condition_map.get(category)
                    if condition and condition not in triggered:
                        triggered.append(condition)
        
        if context:
            if context.get('conversation_turns', 0) > 10:
                triggered.append(XHighTriggerCondition.AMBIGUOUS_CONTEXT)
            if context.get('previous_failures', 0) > 0:
                triggered.append(XHighTriggerCondition.PROBLEM_SOLVING)
        
        if not triggered:
            return False, []
        
        total_weight = sum(self.config.trigger_weights.get(t, 0.5) for t in triggered)
        avg_weight = total_weight / len(triggered)
        
        should_use = avg_weight >= 0.7 or len(triggered) >= 2
        return should_use, triggered
    
    async def execute_with_xhigh(self, user_input: str, system_prompt: Optional[str] = None,
                                  context: Optional[str] = None) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        enhanced_system = self._build_xhigh_system_prompt(system_prompt)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": user_input}],
            system_prompt=enhanced_system,
            reasoning_level=ReasoningLevel.EXTRA_HIGH,
            max_tokens=8000,
            temperature=0.5
        )
        
        try:
            response = await asyncio.wait_for(
                self.llm.generate(request),
                timeout=self.config.max_thinking_time_seconds
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            thinking_trace = response.reasoning_content
            if thinking_trace and self.config.compact_trace:
                thinking_trace = self._compact_trace(thinking_trace)
            
            result = {
                "content": response.content,
                "thinking_trace": thinking_trace,
                "show_trace": self.config.show_thinking_trace,
                "latency_seconds": elapsed,
                "tokens": response.usage,
                "model": response.model_id,
                "finish_reason": response.finish_reason,
            }
            
            self.thinking_history.append({
                "input": user_input[:100],
                "duration": elapsed,
                "tokens": response.usage["total_tokens"],
                "timestamp": start_time
            })
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "content": "The request timed out during deep thinking. Please try again or simplify your request.",
                "thinking_trace": None,
                "error": "timeout",
                "latency_seconds": self.config.max_thinking_time_seconds,
            }
    
    def _build_xhigh_system_prompt(self, base_prompt: Optional[str]) -> str:
        xhigh_instructions = """
You are operating in EXTRA HIGH thinking mode. This mode enables maximum reasoning capability.

## Thinking Mode Guidelines
1. **Deep Analysis**: Take time to thoroughly analyze the problem from multiple angles
2. **Step-by-Step**: Work through solutions methodically, showing your reasoning
3. **Consider Alternatives**: Explore multiple approaches before deciding
4. **Verify**: Double-check your work and validate conclusions
5. **Be Thorough**: Don't rush - quality over speed

## Response Structure
Your thinking will be captured separately from your final response.
- Use the thinking space to work through the problem
- Provide a clear, well-structured final answer
- Include relevant details and explanations

## Quality Standards
- Ensure accuracy and completeness
- Consider edge cases and exceptions
- Provide actionable, practical solutions
- Cite relevant knowledge or principles
"""
        if base_prompt:
            return f"{base_prompt}\n\n{xhigh_instructions}"
        return xhigh_instructions
    
    def _compact_trace(self, trace: str) -> str:
        lines = trace.split('\n')
        lines = [l for l in lines if len(l.strip()) > 10]
        if len(lines) > 50:
            compact = lines[:20]
            compact.append("\n... [thinking trace truncated] ...\n")
            compact.extend(lines[-20:])
            lines = compact
        return '\n'.join(lines)
    
    def get_thinking_stats(self) -> Dict[str, Any]:
        if not self.thinking_history:
            return {"total_invocations": 0}
        
        total = len(self.thinking_history)
        avg_duration = sum(h["duration"] for h in self.thinking_history) / total
        avg_tokens = sum(h["tokens"] for h in self.thinking_history) / total
        
        return {
            "total_invocations": total,
            "avg_duration_seconds": avg_duration,
            "avg_tokens": avg_tokens,
            "last_invocation": self.thinking_history[-1]["timestamp"] if self.thinking_history else None,
        }
```

---

## Multi-Model Fallback & Routing

### 7.1 Model Router

```python
# core/llm/routing/router.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time

class RoutingStrategy(Enum):
    CAPABILITY_BASED = "capability_based"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    FALLBACK_CHAIN = "fallback_chain"
    ADAPTIVE = "adaptive"

@dataclass
class RoutingDecision:
    provider_name: str
    model_id: str
    reasoning: str
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float
    fallback_chain: List[str]

class ModelRouter:
    """Intelligent model router with multiple routing strategies."""
    
    TASK_COMPLEXITY = {
        "greeting": 1, "simple_qa": 2, "summarization": 3, "translation": 3,
        "code_simple": 4, "analysis": 5, "code_complex": 6, "planning": 7,
        "research": 8, "creative": 7, "debugging": 6, "data_analysis": 7,
    }
    
    MODEL_CAPABILITIES = {
        "gpt-5.2": {"quality": 9, "speed": 7, "reasoning": 9, "coding": 9, "cost_efficiency": 6},
        "gpt-5.2-instant": {"quality": 7, "speed": 9, "reasoning": 6, "coding": 7, "cost_efficiency": 9},
        "gpt-5.2-pro": {"quality": 10, "speed": 4, "reasoning": 10, "coding": 10, "cost_efficiency": 3},
        "claude-opus": {"quality": 9, "speed": 6, "reasoning": 9, "coding": 9, "cost_efficiency": 5},
        "claude-sonnet": {"quality": 8, "speed": 7, "reasoning": 8, "coding": 8, "cost_efficiency": 7},
        "claude-haiku": {"quality": 6, "speed": 9, "reasoning": 6, "coding": 6, "cost_efficiency": 10},
    }
    
    def __init__(self, providers: Dict[str, BaseLLMProvider],
                 default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
                 cost_budget_per_request: float = 0.10):
        self.providers = providers
        self.strategy = default_strategy
        self.cost_budget = cost_budget_per_request
        self.routing_history: List[Dict] = []
        self.performance_metrics: Dict[str, Dict] = {}
    
    async def route(self, request: LLMRequest, task_type: Optional[str] = None,
                    user_tier: str = "standard") -> RoutingDecision:
        complexity = self._assess_complexity(request, task_type)
        requires_reasoning = self._requires_deep_reasoning(request)
        urgency = request.metadata.get('urgency', 'normal') if request.metadata else 'normal'
        
        if self.strategy == RoutingStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(complexity, requires_reasoning, urgency, user_tier)
        else:
            strategy = self.strategy
        
        decision = self._make_decision(strategy, complexity, requires_reasoning, urgency, user_tier)
        
        self.routing_history.append({
            "timestamp": time.time(),
            "strategy": strategy.value,
            "decision": decision.provider_name,
            "complexity": complexity,
            "estimated_cost": decision.estimated_cost,
        })
        
        return decision
    
    def _assess_complexity(self, request: LLMRequest, task_type: Optional[str]) -> int:
        complexity = 5
        if task_type and task_type in self.TASK_COMPLEXITY:
            complexity = self.TASK_COMPLEXITY[task_type]
        
        content = " ".join(m.get('content', '') for m in request.messages)
        token_estimate = len(content) // 4
        if token_estimate > 2000:
            complexity += 2
        elif token_estimate > 1000:
            complexity += 1
        
        complex_keywords = ["analyze", "compare", "evaluate", "design", "architecture",
                           "optimize", "refactor", "implement", "create", "build"]
        for keyword in complex_keywords:
            if keyword in content.lower():
                complexity += 1
                break
        
        return min(complexity, 10)
    
    def _requires_deep_reasoning(self, request: LLMRequest) -> bool:
        if request.reasoning_level in [ReasoningLevel.HIGH, ReasoningLevel.EXTRA_HIGH]:
            return True
        
        content = " ".join(m.get('content', '') for m in request.messages)
        reasoning_indicators = ["why", "explain", "reason", "think through", "step by step",
                               "how would you", "what if", "consider", "analyze"]
        return any(indicator in content.lower() for indicator in reasoning_indicators)
    
    def _select_adaptive_strategy(self, complexity: int, requires_reasoning: bool,
                                   urgency: str, user_tier: str) -> RoutingStrategy:
        if urgency == "high":
            return RoutingStrategy.LATENCY_OPTIMIZED
        if user_tier == "premium":
            return RoutingStrategy.QUALITY_OPTIMIZED
        if user_tier == "budget":
            return RoutingStrategy.COST_OPTIMIZED
        if complexity >= 7 and requires_reasoning:
            return RoutingStrategy.CAPABILITY_BASED
        if complexity <= 4:
            return RoutingStrategy.COST_OPTIMIZED
        return RoutingStrategy.CAPABILITY_BASED
    
    def _make_decision(self, strategy: RoutingStrategy, complexity: int,
                       requires_reasoning: bool, urgency: str, user_tier: str) -> RoutingDecision:
        if strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._route_capability_based(complexity, requires_reasoning)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._route_cost_optimized(complexity)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return self._route_latency_optimized(complexity)
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return self._route_quality_optimized(requires_reasoning)
        else:
            return self._route_fallback_chain()
    
    def _route_capability_based(self, complexity: int, requires_reasoning: bool) -> RoutingDecision:
        if complexity >= 8 and requires_reasoning:
            if "gpt-5.2-pro" in self.providers:
                return RoutingDecision(
                    provider_name="gpt-5.2-pro", model_id="gpt-5.2-pro",
                    reasoning="High complexity task requiring deep reasoning",
                    estimated_cost=0.05, estimated_latency_ms=15000, confidence=0.95,
                    fallback_chain=["claude-opus", "gpt-5.2", "claude-sonnet"]
                )
            else:
                return RoutingDecision(
                    provider_name="claude-opus", model_id="claude-opus",
                    reasoning="High complexity task, GPT-5.2 Pro unavailable",
                    estimated_cost=0.04, estimated_latency_ms=12000, confidence=0.90,
                    fallback_chain=["gpt-5.2", "claude-sonnet"]
                )
        
        if complexity >= 6:
            return RoutingDecision(
                provider_name="gpt-5.2", model_id="gpt-5.2",
                reasoning="Medium-high complexity task",
                estimated_cost=0.015, estimated_latency_ms=5000, confidence=0.90,
                fallback_chain=["claude-sonnet", "gpt-5.2-instant"]
            )
        
        return RoutingDecision(
            provider_name="gpt-5.2-instant", model_id="gpt-5.2-instant",
            reasoning="Lower complexity task, cost-optimized",
            estimated_cost=0.003, estimated_latency_ms=2000, confidence=0.85,
            fallback_chain=["claude-haiku", "gpt-5.2"]
        )
    
    def _route_cost_optimized(self, complexity: int) -> RoutingDecision:
        if complexity <= 4:
            return RoutingDecision(
                provider_name="claude-haiku", model_id="claude-haiku",
                reasoning="Cost-optimized routing for simple task",
                estimated_cost=0.001, estimated_latency_ms=1500, confidence=0.80,
                fallback_chain=["gpt-5.2-instant", "claude-sonnet"]
            )
        return RoutingDecision(
            provider_name="gpt-5.2-instant", model_id="gpt-5.2-instant",
            reasoning="Cost-optimized routing for medium task",
            estimated_cost=0.003, estimated_latency_ms=2000, confidence=0.85,
            fallback_chain=["claude-sonnet", "gpt-5.2"]
        )
    
    def _route_latency_optimized(self, complexity: int) -> RoutingDecision:
        if complexity <= 5:
            return RoutingDecision(
                provider_name="gpt-5.2-instant", model_id="gpt-5.2-instant",
                reasoning="Latency-optimized routing",
                estimated_cost=0.003, estimated_latency_ms=1000, confidence=0.85,
                fallback_chain=["claude-haiku", "claude-sonnet"]
            )
        return RoutingDecision(
            provider_name="claude-sonnet", model_id="claude-sonnet",
            reasoning="Latency-optimized for complex task",
            estimated_cost=0.008, estimated_latency_ms=3000, confidence=0.88,
            fallback_chain=["gpt-5.2", "gpt-5.2-pro"]
        )
    
    def _route_quality_optimized(self, requires_reasoning: bool) -> RoutingDecision:
        if requires_reasoning:
            return RoutingDecision(
                provider_name="gpt-5.2-pro", model_id="gpt-5.2-pro",
                reasoning="Quality-optimized with deep reasoning",
                estimated_cost=0.05, estimated_latency_ms=15000, confidence=0.98,
                fallback_chain=["claude-opus", "gpt-5.2"]
            )
        return RoutingDecision(
            provider_name="gpt-5.2", model_id="gpt-5.2",
            reasoning="Quality-optimized routing",
            estimated_cost=0.015, estimated_latency_ms=5000, confidence=0.95,
            fallback_chain=["claude-opus", "claude-sonnet"]
        )
    
    def _route_fallback_chain(self) -> RoutingDecision:
        return RoutingDecision(
            provider_name="gpt-5.2", model_id="gpt-5.2",
            reasoning="Fallback chain routing",
            estimated_cost=0.015, estimated_latency_ms=5000, confidence=0.90,
            fallback_chain=["gpt-5.2-pro", "claude-opus", "claude-sonnet", 
                           "gpt-5.2-instant", "claude-haiku"]
        )
```

### 7.2 Fallback Handler

```python
# core/llm/routing/fallback.py
class FallbackHandler:
    """Handles model failures with intelligent fallback."""
    
    def __init__(self, providers: Dict[str, BaseLLMProvider],
                 max_failures: int = 3, reset_timeout_seconds: int = 60):
        self.providers = providers
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout_seconds
        self.failure_counts: Dict[str, int] = {name: 0 for name in providers}
        self.last_failure_time: Dict[str, float] = {}
        self.circuit_open: Dict[str, bool] = {name: False for name in providers}
    
    async def execute_with_fallback(self, decision: RoutingDecision, request: LLMRequest) -> LLMResponse:
        providers_to_try = [decision.provider_name] + decision.fallback_chain
        last_error = None
        
        for provider_name in providers_to_try:
            if self._is_circuit_open(provider_name):
                continue
            
            try:
                provider = self.providers.get(provider_name)
                if not provider:
                    continue
                
                if request.stream:
                    chunks = []
                    async for chunk in provider.generate_stream(request):
                        chunks.append(chunk)
                    response = self._combine_streaming_chunks(chunks)
                else:
                    response = await provider.generate(request)
                
                self._record_success(provider_name)
                return response
                
            except Exception as e:
                last_error = e
                self._record_failure(provider_name)
                continue
        
        raise FallbackExhaustedError(f"All providers failed. Last error: {last_error}")
    
    def _is_circuit_open(self, provider_name: str) -> bool:
        if not self.circuit_open.get(provider_name, False):
            return False
        
        last_failure = self.last_failure_time.get(provider_name, 0)
        if time.time() - last_failure > self.reset_timeout:
            self.circuit_open[provider_name] = False
            self.failure_counts[provider_name] = 0
            return False
        
        return True
    
    def _record_failure(self, provider_name: str) -> None:
        self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
        self.last_failure_time[provider_name] = time.time()
        
        if self.failure_counts[provider_name] >= self.max_failures:
            self.circuit_open[provider_name] = True
    
    def _record_success(self, provider_name: str) -> None:
        self.failure_counts[provider_name] = 0
        self.circuit_open[provider_name] = False
    
    def _combine_streaming_chunks(self, chunks: List[LLMResponse]) -> LLMResponse:
        if not chunks:
            raise ValueError("No chunks to combine")
        
        combined_content = "".join(c.content for c in chunks)
        combined_reasoning = "".join(c.reasoning_content for c in chunks if c.reasoning_content)
        
        total_tokens = sum(c.usage.get("total_tokens", 0) for c in chunks)
        total_latency = sum(c.latency_ms for c in chunks)
        
        return LLMResponse(
            content=combined_content,
            model_id=chunks[0].model_id,
            usage={
                "prompt_tokens": chunks[0].usage.get("prompt_tokens", 0),
                "completion_tokens": total_tokens - chunks[0].usage.get("prompt_tokens", 0),
                "total_tokens": total_tokens,
            },
            reasoning_content=combined_reasoning or None,
            finish_reason=chunks[-1].finish_reason,
            latency_ms=total_latency,
        )

class FallbackExhaustedError(Exception):
    pass
```

---

## Token Usage Optimization

### 8.1 Token Budget Manager

```python
# core/optimization/token_manager.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import time

@dataclass
class TokenBudget:
    daily_limit: int = 100000
    hourly_limit: int = 10000
    per_request_limit: int = 8000
    warning_threshold: float = 0.8
    model_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.model_multipliers is None:
            self.model_multipliers = {
                "gpt-5.2": 1.0,
                "gpt-5.2-pro": 5.0,
                "gpt-5.2-instant": 0.3,
                "claude-opus": 1.2,
                "claude-sonnet": 0.6,
                "claude-haiku": 0.1,
            }

class TokenBudgetManager:
    """Manages token budgets with tracking and enforcement."""
    
    def __init__(self, budget: TokenBudget = None):
        self.budget = budget or TokenBudget()
        self.usage_history: List[Dict] = []
        self.alerts: List[Dict] = []
        self.alert_callbacks: List[Callable] = []
    
    def register_alert_callback(self, callback: Callable) -> None:
        self.alert_callbacks.append(callback)
    
    def check_budget(self, estimated_tokens: int, model_id: str) -> Dict[str, any]:
        multiplier = self.budget.model_multipliers.get(model_id, 1.0)
        adjusted_tokens = int(estimated_tokens * multiplier)
        
        daily_usage = self.get_daily_usage()
        hourly_usage = self.get_hourly_usage()
        
        status = {"allowed": True, "warnings": [], "recommendations": []}
        
        if adjusted_tokens > self.budget.per_request_limit:
            status["allowed"] = False
            status["warnings"].append(f"Request exceeds per-request limit ({adjusted_tokens} > {self.budget.per_request_limit})")
            status["recommendations"].append("Split request into smaller chunks")
        
        hourly_projected = hourly_usage + adjusted_tokens
        if hourly_projected > self.budget.hourly_limit:
            status["warnings"].append(f"Hourly budget at risk ({hourly_projected}/{self.budget.hourly_limit})")
            status["recommendations"].append("Consider using a cheaper model")
        
        daily_projected = daily_usage + adjusted_tokens
        daily_ratio = daily_projected / self.budget.daily_limit
        
        if daily_ratio > 1.0:
            status["allowed"] = False
            status["warnings"].append(f"Daily budget exceeded ({daily_projected}/{self.budget.daily_limit})")
        elif daily_ratio > self.budget.warning_threshold:
            status["warnings"].append(f"Daily budget at {daily_ratio*100:.1f}%")
            status["recommendations"].append("Switch to cost-optimized routing")
        
        if status["warnings"]:
            self._trigger_alerts(status["warnings"])
        
        return status
    
    def record_usage(self, tokens: int, model_id: str, metadata: Dict = None) -> None:
        multiplier = self.budget.model_multipliers.get(model_id, 1.0)
        adjusted_tokens = int(tokens * multiplier)
        
        self.usage_history.append({
            "timestamp": time.time(),
            "tokens": tokens,
            "adjusted_tokens": adjusted_tokens,
            "model_id": model_id,
            "metadata": metadata or {},
        })
        
        cutoff = time.time() - (30 * 24 * 3600)
        self.usage_history = [u for u in self.usage_history if u["timestamp"] > cutoff]
    
    def get_daily_usage(self) -> int:
        day_start = time.time() - (time.time() % 86400)
        return sum(u["adjusted_tokens"] for u in self.usage_history if u["timestamp"] >= day_start)
    
    def get_hourly_usage(self) -> int:
        hour_start = time.time() - (time.time() % 3600)
        return sum(u["adjusted_tokens"] for u in self.usage_history if u["timestamp"] >= hour_start)
    
    def get_usage_stats(self) -> Dict:
        daily = self.get_daily_usage()
        hourly = self.get_hourly_usage()
        
        return {
            "daily_usage": daily,
            "daily_limit": self.budget.daily_limit,
            "daily_remaining": self.budget.daily_limit - daily,
            "daily_percentage": (daily / self.budget.daily_limit * 100) if self.budget.daily_limit > 0 else 0,
            "hourly_usage": hourly,
            "hourly_limit": self.budget.hourly_limit,
            "hourly_remaining": self.budget.hourly_limit - hourly,
            "total_requests": len(self.usage_history),
            "avg_tokens_per_request": sum(u["tokens"] for u in self.usage_history) / len(self.usage_history) if self.usage_history else 0,
        }
    
    def _trigger_alerts(self, warnings: List[str]) -> None:
        for callback in self.alert_callbacks:
            try:
                callback(warnings)
            except Exception:
                pass
```

### 8.2 Token Optimizer

```python
class TokenOptimizer:
    """Optimizes prompts to reduce token usage."""
    
    REPLACEMENTS = {
        "in order to": "to", "due to the fact that": "because",
        "at this point in time": "now", "in the event that": "if",
        "for the purpose of": "to", "with regard to": "about",
        "in spite of the fact that": "although", "at the present time": "now",
        "in the near future": "soon", "on a daily basis": "daily",
        "in the process of": "while", "with respect to": "about",
        "in connection with": "about", "for the reason that": "because",
        "in view of the fact that": "since", "on the occasion of": "when",
        "under the circumstances": "then", "it is clear that": "clearly",
        "it is important to note that": "note:", "it should be noted that": "note:",
    }
    
    FILLER_WORDS = ["very", "really", "just", "basically", "actually", 
                    "literally", "quite", "rather", "somewhat", "fairly"]
    
    def optimize(self, text: str, aggressive: bool = False) -> str:
        import re
        optimized = re.sub(r'\s+', ' ', text).strip()
        
        for verbose, concise in self.REPLACEMENTS.items():
            pattern = re.compile(r'\b' + re.escape(verbose) + r'\b', re.IGNORECASE)
            optimized = pattern.sub(concise, optimized)
        
        if aggressive:
            for filler in self.FILLER_WORDS:
                pattern = re.compile(r'\b' + re.escape(filler) + r'\b', re.IGNORECASE)
                optimized = pattern.sub('', optimized)
            optimized = re.sub(r'\s+', ' ', optimized).strip()
        
        return optimized
    
    def truncate_context(self, context: str, max_tokens: int, preserve_recent: bool = True) -> str:
        max_chars = max_tokens * 4
        if len(context) <= max_chars:
            return context
        if preserve_recent:
            return "..." + context[-(max_chars - 3):]
        return context[:max_chars - 3] + "..."
    
    def compress_examples(self, examples: List[Dict], max_examples: int = 3) -> List[Dict]:
        if len(examples) <= max_examples:
            return examples
        
        if max_examples >= 3:
            indices = [0, len(examples) // 2, len(examples) - 1]
            for i in range(max_examples - 3):
                idx = (i + 1) * len(examples) // (max_examples - 2)
                if idx not in indices:
                    indices.append(idx)
            indices = sorted(indices)[:max_examples]
        else:
            indices = list(range(max_examples))
        
        return [examples[i] for i in indices]
```

### 8.3 Caching System

```python
# core/optimization/caching.py
from typing import Dict, Optional, List, Any
import hashlib
import time
import json

class PromptCache:
    """Multi-level caching system for LLM responses."""
    
    def __init__(self, exact_ttl_seconds: int = 3600, semantic_ttl_seconds: int = 1800):
        self.exact_cache: Dict[str, Dict] = {}
        self.semantic_cache: Dict[str, Dict] = {}
        self.exact_ttl = exact_ttl_seconds
        self.semantic_ttl = semantic_ttl_seconds
        self.embedding_model = None
        self.hits = {"exact": 0, "semantic": 0}
        self.misses = 0
    
    def _generate_key(self, request: LLMRequest) -> str:
        content = json.dumps({
            "messages": request.messages,
            "system": request.system_prompt,
            "tools": request.tools,
            "reasoning_level": request.reasoning_level.value,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        key = self._generate_key(request)
        
        if key in self.exact_cache:
            entry = self.exact_cache[key]
            if time.time() - entry["timestamp"] < self.exact_ttl:
                self.hits["exact"] += 1
                return entry["response"]
            else:
                del self.exact_cache[key]
        
        if self.embedding_model:
            semantic_match = self._find_semantic_match(request)
            if semantic_match:
                self.hits["semantic"] += 1
                return semantic_match
        
        self.misses += 1
        return None
    
    def set(self, request: LLMRequest, response: LLMResponse) -> None:
        key = self._generate_key(request)
        self.exact_cache[key] = {
            "timestamp": time.time(),
            "response": response,
            "request": request,
        }
        if len(self.exact_cache) > 1000:
            self._cleanup_exact_cache()
    
    def _cleanup_exact_cache(self) -> None:
        now = time.time()
        expired = [k for k, v in self.exact_cache.items() if now - v["timestamp"] > self.exact_ttl]
        for k in expired:
            del self.exact_cache[k]
    
    def _find_semantic_match(self, request: LLMRequest) -> Optional[LLMResponse]:
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = sum(self.hits.values()) + self.misses
        hit_rate = (sum(self.hits.values()) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "exact_hits": self.hits["exact"],
            "semantic_hits": self.hits["semantic"],
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": len(self.semantic_cache),
        }
```

---

## Response Parsing & Action Extraction

### 9.1 Action Extractor

```python
# core/parsing/action_extractor.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import re
import json

@dataclass
class ExtractedAction:
    action_type: str
    tool_name: Optional[str]
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str

class ActionExtractor:
    """Extracts actions and tool calls from LLM responses."""
    
    ACTION_PATTERNS = {
        "thought": re.compile(r'(?:THOUGHT|Thought):\s*(.+?)(?=\n(?:ACTION|Action):|$)', re.DOTALL | re.IGNORECASE),
        "action": re.compile(r'(?:ACTION|Action):\s*(\w+)', re.IGNORECASE),
        "params": re.compile(r'(?:PARAMS|Params|PARAMETERS|Parameters):\s*(\{.*?\})', re.DOTALL | re.IGNORECASE),
        "tool_call": re.compile(r'```tool\n(\{.*?\})\n```', re.DOTALL),
    }
    
    def __init__(self, available_tools: List[str] = None):
        self.available_tools = available_tools or []
    
    def extract(self, response_text: str) -> List[ExtractedAction]:
        actions = []
        
        # Extract structured action blocks
        action_blocks = self._extract_action_blocks(response_text)
        for block in action_blocks:
            action = self._parse_action_block(block)
            if action:
                actions.append(action)
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(response_text)
        actions.extend(tool_calls)
        
        # Extract implicit actions
        implicit_actions = self._extract_implicit_actions(response_text)
        actions.extend(implicit_actions)
        
        return actions
    
    def _extract_action_blocks(self, text: str) -> List[str]:
        blocks = []
        pattern = re.compile(r'THOUGHT:.+?(?=\n\n|$)', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        blocks.extend(matches)
        return blocks
    
    def _parse_action_block(self, block: str) -> Optional[ExtractedAction]:
        thought_match = self.ACTION_PATTERNS["thought"].search(block)
        action_match = self.ACTION_PATTERNS["action"].search(block)
        params_match = self.ACTION_PATTERNS["params"].search(block)
        
        if not action_match:
            return None
        
        action_type = action_match.group(1).lower()
        tool_name = None
        
        if action_type in self.available_tools:
            tool_name = action_type
            action_type = "tool_call"
        
        parameters = {}
        if params_match:
            try:
                parameters = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                parameters = {"raw": params_match.group(1)}
        
        confidence = 0.9 if tool_name else 0.7
        
        return ExtractedAction(
            action_type=action_type,
            tool_name=tool_name,
            parameters=parameters,
            confidence=confidence,
            raw_text=block
        )
    
    def _extract_tool_calls(self, text: str) -> List[ExtractedAction]:
        actions = []
        matches = self.ACTION_PATTERNS["tool_call"].findall(text)
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                actions.append(ExtractedAction(
                    action_type="tool_call",
                    tool_name=tool_data.get("name"),
                    parameters=tool_data.get("arguments", {}),
                    confidence=0.95,
                    raw_text=match
                ))
            except json.JSONDecodeError:
                continue
        
        return actions
    
    def _extract_implicit_actions(self, text: str) -> List[ExtractedAction]:
        actions = []
        
        # Check for common action patterns
        if re.search(r'\b(open|launch|start)\s+(\w+)', text, re.IGNORECASE):
            match = re.search(r'\b(open|launch|start)\s+(\w+)', text, re.IGNORECASE)
            actions.append(ExtractedAction(
                action_type="open_application",
                tool_name="system",
                parameters={"application": match.group(2)},
                confidence=0.6,
                raw_text=match.group(0)
            ))
        
        if re.search(r'\b(search|find|look up)\s+(.+?)(?=\.|$)', text, re.IGNORECASE):
            match = re.search(r'\b(search|find|look up)\s+(.+?)(?=\.|$)', text, re.IGNORECASE)
            actions.append(ExtractedAction(
                action_type="search",
                tool_name="browser",
                parameters={"query": match.group(2).strip()},
                confidence=0.6,
                raw_text=match.group(0)
            ))
        
        return actions
```

### 9.2 Response Parser

```python
# core/parsing/response_parser.py
class ResponseParser:
    """Parses LLM responses into structured components."""
    
    def __init__(self, action_extractor: ActionExtractor = None):
        self.action_extractor = action_extractor or ActionExtractor()
    
    def parse(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse response into structured components."""
        content = response.content
        
        # Extract main content (remove action blocks)
        main_content = self._extract_main_content(content)
        
        # Extract actions
        actions = self.action_extractor.extract(content)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        
        # Extract JSON data
        json_data = self._extract_json(content)
        
        return {
            "main_content": main_content,
            "actions": actions,
            "code_blocks": code_blocks,
            "json_data": json_data,
            "reasoning": response.reasoning_content,
            "model": response.model_id,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
        }
    
    def _extract_main_content(self, text: str) -> str:
        # Remove action blocks
        cleaned = re.sub(r'THOUGHT:.+?(?=\n\n|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'ACTION:.+?(?=\n\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'PARAMS:.+?(?=\n\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```tool\n.+?\n```', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        blocks = []
        pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(text)
        
        for lang, code in matches:
            blocks.append({
                "language": lang or "text",
                "code": code.strip()
            })
        
        return blocks
    
    def _extract_json(self, text: str) -> List[Dict]:
        json_objects = []
        
        # Find JSON in code blocks
        pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(text)
        
        for match in matches:
            try:
                json_objects.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        
        # Find inline JSON objects
        pattern = re.compile(r'\{[^{}]*\}')
        matches = pattern.findall(text)
        
        for match in matches:
            try:
                json_objects.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        
        return json_objects
```

---

## Streaming Response Handling

### 10.1 Streaming Handler

```python
# core/streaming/handler.py
from typing import AsyncIterator, Callable, List, Optional
import asyncio

class StreamingHandler:
    """Handles streaming LLM responses with buffering and processing."""
    
    def __init__(self, buffer_size: int = 10, min_chunk_size: int = 5):
        self.buffer_size = buffer_size
        self.min_chunk_size = min_chunk_size
        self.buffer: List[str] = []
        self.callbacks: List[Callable[[str], None]] = []
    
    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for chunk processing."""
        self.callbacks.append(callback)
    
    async def process_stream(
        self, 
        stream: AsyncIterator[LLMResponse]
    ) -> AsyncIterator[str]:
        """Process streaming response with buffering."""
        accumulated = ""
        
        async for chunk in stream:
            content = chunk.content
            if not content:
                continue
            
            accumulated += content
            self.buffer.append(content)
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(content)
                except Exception:
                    pass
            
            # Yield when buffer is full or sentence complete
            if len(self.buffer) >= self.buffer_size or self._is_sentence_complete(accumulated):
                yield accumulated
                accumulated = ""
                self.buffer = []
        
        # Yield remaining content
        if accumulated:
            yield accumulated
    
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text ends with sentence terminator."""
        return text.rstrip().endswith(('.', '!', '?', '\n'))
    
    async def stream_with_timeout(
        self,
        stream: AsyncIterator[LLMResponse],
        timeout_seconds: float = 30.0
    ) -> AsyncIterator[str]:
        """Stream with timeout protection."""
        try:
            async for chunk in asyncio.wait_for(
                self._stream_iterator(stream),
                timeout=timeout_seconds
            ):
                yield chunk
        except asyncio.TimeoutError:
            yield "[Response timed out]"
    
    async def _stream_iterator(
        self, 
        stream: AsyncIterator[LLMResponse]
    ) -> AsyncIterator[str]:
        """Helper for timeout wrapper."""
        async for chunk in self.process_stream(stream):
            yield chunk

class StreamingAggregator:
    """Aggregates streaming chunks into complete response."""
    
    def __init__(self):
        self.chunks: List[LLMResponse] = []
        self.content_buffer = ""
    
    def add_chunk(self, chunk: LLMResponse) -> None:
        """Add a chunk to the aggregation."""
        self.chunks.append(chunk)
        self.content_buffer += chunk.content or ""
    
    def get_partial_content(self) -> str:
        """Get current accumulated content."""
        return self.content_buffer
    
    def finalize(self) -> LLMResponse:
        """Finalize aggregation into complete response."""
        if not self.chunks:
            raise ValueError("No chunks to aggregate")
        
        combined_content = "".join(c.content for c in self.chunks)
        combined_reasoning = "".join(
            c.reasoning_content for c in self.chunks if c.reasoning_content
        )
        
        total_tokens = sum(c.usage.get("total_tokens", 0) for c in self.chunks)
        total_latency = sum(c.latency_ms for c in self.chunks)
        
        return LLMResponse(
            content=combined_content,
            model_id=self.chunks[0].model_id,
            usage={
                "prompt_tokens": self.chunks[0].usage.get("prompt_tokens", 0),
                "completion_tokens": total_tokens - self.chunks[0].usage.get("prompt_tokens", 0),
                "total_tokens": total_tokens,
            },
            reasoning_content=combined_reasoning or None,
            finish_reason=self.chunks[-1].finish_reason,
            latency_ms=total_latency,
        )
```

### 10.2 SSE Streaming for Web Interface

```python
# core/streaming/sse.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

class SSEStreamingHandler:
    """Server-Sent Events streaming handler for web interfaces."""
    
    def __init__(self):
        self.app = FastAPI()
    
    async def stream_to_sse(
        self,
        stream: AsyncIterator[LLMResponse]
    ) -> StreamingResponse:
        """Convert LLM stream to SSE format."""
        
        async def event_generator():
            async for chunk in stream:
                data = {
                    "content": chunk.content,
                    "reasoning": chunk.reasoning_content,
                    "finish_reason": chunk.finish_reason,
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    async def stream_with_metadata(
        self,
        stream: AsyncIterator[LLMResponse],
        include_usage: bool = True
    ) -> StreamingResponse:
        """Stream with metadata included."""
        
        async def event_generator():
            total_tokens = 0
            
            async for chunk in stream:
                total_tokens += chunk.usage.get("total_tokens", 0)
                
                data = {
                    "content": chunk.content,
                    "reasoning": chunk.reasoning_content,
                    "finish_reason": chunk.finish_reason,
                }
                
                if include_usage:
                    data["usage"] = chunk.usage
                
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send final metadata
            if include_usage:
                yield f"data: {json.dumps({'final_usage': {'total_tokens': total_tokens}})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
```

---

## Integration Patterns

### 11.1 Agent Core Integration

```python
# core/agent/integration.py
class AgentLLMIntegration:
    """Integration layer between Agent Core and LLM systems."""
    
    def __init__(
        self,
        model_router: ModelRouter,
        fallback_handler: FallbackHandler,
        xhigh_controller: XHighThinkingController,
        template_manager: TemplateManager,
        token_budget: TokenBudgetManager,
        prompt_cache: PromptCache,
    ):
        self.router = model_router
        self.fallback = fallback_handler
        self.xhigh = xhigh_controller
        self.templates = template_manager
        self.budget = token_budget
        self.cache = prompt_cache
    
    async def process_request(
        self,
        user_input: str,
        agent_context: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Process a user request through the complete LLM pipeline."""
        
        # 1. Check if X-High thinking is needed
        should_use_xhigh, triggers = self.xhigh.should_use_xhigh(user_input, agent_context)
        
        if should_use_xhigh:
            result = await self.xhigh.execute_with_xhigh(
                user_input,
                system_prompt=agent_context.get('system_prompt'),
                context=agent_context.get('conversation_summary')
            )
            result["mode"] = "extra_high"
            result["triggers"] = [t.value for t in triggers]
            return result
        
        # 2. Build LLM request
        request = self._build_request(user_input, agent_context)
        
        # 3. Check cache
        cached_response = self.cache.get(request)
        if cached_response:
            return {
                "content": cached_response.content,
                "mode": "cached",
                "cached": True,
                "model": cached_response.model_id,
            }
        
        # 4. Check budget
        estimated_tokens = self._estimate_tokens(request)
        budget_status = self.budget.check_budget(estimated_tokens, "gpt-5.2")
        
        if not budget_status["allowed"]:
            return {
                "content": "Request exceeds budget limits. Please try a shorter query.",
                "mode": "budget_exceeded",
                "warnings": budget_status["warnings"],
            }
        
        # 5. Route to appropriate model
        routing_decision = await self.router.route(request)
        
        # 6. Execute with fallback
        if stream:
            response_stream = await self._execute_streaming(routing_decision, request)
            return {"stream": response_stream, "mode": "streaming"}
        else:
            response = await self.fallback.execute_with_fallback(routing_decision, request)
        
        # 7. Cache response
        self.cache.set(request, response)
        
        # 8. Record usage
        self.budget.record_usage(
            response.usage["total_tokens"],
            response.model_id,
            {"input": user_input[:50]}
        )
        
        return {
            "content": response.content,
            "mode": "normal",
            "model": response.model_id,
            "reasoning": response.reasoning_content,
            "usage": response.usage,
            "latency_ms": response.latency_ms,
        }
    
    def _build_request(self, user_input: str, agent_context: Dict) -> LLMRequest:
        """Build LLM request from user input and context."""
        system_prompt = self.templates.render("system_base",
            agent_name=agent_context.get('agent_name', 'OpenClaw'),
            agent_identity=agent_context.get('agent_identity', 'an AI assistant'),
            capabilities=agent_context.get('capabilities', ''),
            current_time=agent_context.get('current_time', ''),
            user_context=agent_context.get('user_context', ''),
        )
        
        return LLMRequest(
            messages=[{"role": "user", "content": user_input}],
            system_prompt=system_prompt,
            reasoning_level=ReasoningLevel.MEDIUM,
            stream=False,
            max_tokens=4000,
        )
    
    def _estimate_tokens(self, request: LLMRequest) -> int:
        """Estimate token count for request."""
        content = " ".join(m.get('content', '') for m in request.messages)
        if request.system_prompt:
            content += request.system_prompt
        return len(content) // 4
    
    async def _execute_streaming(
        self,
        decision: RoutingDecision,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Execute streaming request."""
        provider = self.fallback.providers.get(decision.provider_name)
        if not provider:
            raise ValueError(f"Provider not found: {decision.provider_name}")
        
        stream = provider.generate_stream(request)
        handler = StreamingHandler()
        
        async for chunk in handler.process_stream(stream):
            yield chunk
```

---

## Configuration Reference

### 12.1 Default Configuration

```yaml
# config/llm_config.yaml
llm:
  # Primary provider configuration
  primary_provider: "gpt-5.2"
  
  providers:
    gpt-5.2:
      model_id: "gpt-5.2"
      api_key: "${OPENAI_API_KEY}"
      timeout: 120.0
      max_retries: 3
      reasoning_level: "medium"
      enable_streaming: true
      max_tokens: 4096
      temperature: 0.7
    
    gpt-5.2-pro:
      model_id: "gpt-5.2-pro"
      api_key: "${OPENAI_API_KEY}"
      timeout: 300.0
      max_retries: 3
      reasoning_level: "xhigh"
      enable_streaming: true
      max_tokens: 8000
      temperature: 0.5
    
    claude-opus:
      model_id: "claude-opus"
      api_key: "${ANTHROPIC_API_KEY}"
      timeout: 120.0
      max_retries: 3
      reasoning_level: "high"
      enable_streaming: true
      max_tokens: 4096
      temperature: 0.7
  
  # Routing configuration
  routing:
    default_strategy: "adaptive"
    cost_budget_per_request: 0.10
    
  # X-High thinking configuration
  xhigh:
    enabled: true
    auto_trigger: true
    max_thinking_time_seconds: 300
    min_confidence_threshold: 0.9
    show_thinking_trace: true
    compact_trace: true
  
  # Token budget configuration
  budget:
    daily_limit: 100000
    hourly_limit: 10000
    per_request_limit: 8000
    warning_threshold: 0.8
  
  # Caching configuration
  cache:
    exact_ttl_seconds: 3600
    semantic_ttl_seconds: 1800
  
  # Streaming configuration
  streaming:
    buffer_size: 10
    min_chunk_size: 5
    timeout_seconds: 30.0

# Prompt templates configuration
templates:
  directory: "templates/prompts"
  hot_reload: true
  
  # Default templates
  defaults:
    system: "system_base"
    intent_analysis: "intent_analysis"
    action_planning: "action_planning"
    reasoning: "reasoning_cot"
    tool_execution: "tool_execution"
    soul_persona: "soul_persona"
    error_recovery: "error_recovery"
```

### 12.2 Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Model Configuration
LLM_PRIMARY_PROVIDER=gpt-5.2
LLM_REASONING_LEVEL=medium
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7

# X-High Configuration
LLM_XHIGH_ENABLED=true
LLM_XHIGH_AUTO_TRIGGER=true
LLM_XHIGH_MAX_TIME=300

# Budget Configuration
LLM_BUDGET_DAILY=100000
LLM_BUDGET_HOURLY=10000
LLM_BUDGET_WARNING_THRESHOLD=0.8

# Cache Configuration
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=3600

# Streaming Configuration
LLM_STREAMING_ENABLED=true
LLM_STREAMING_TIMEOUT=30
```

---

## Summary

This specification provides a comprehensive architecture for the LLM Integration and Reasoning Engine of a Windows 10 OpenClaw-inspired AI agent system. Key components include:

1. **Model Abstraction Layer**: Unified interface for GPT-5.2, Claude, and future models
2. **Prompt Engineering System**: Template management with dynamic context assembly
3. **Chain-of-Thought Pipeline**: 7-stage reasoning process for complex tasks
4. **Extra High Thinking Mode**: Automatic trigger-based deep reasoning
5. **Multi-Model Routing**: Intelligent model selection with fallback chains
6. **Token Optimization**: Budget management, caching, and prompt compression
7. **Response Parsing**: Action extraction and structured response handling
8. **Streaming Support**: Real-time response delivery with buffering

The architecture is designed for 24/7 operation with robust error handling, cost management, and extensibility for future model integrations.

---

*Document Version: 1.0*  
*Last Updated: 2025*
