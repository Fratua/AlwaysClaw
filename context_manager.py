"""
Context Window Management for LLM
Handles token budgeting, compression, and pre-compaction flush
"""

import logging
import re
import tiktoken
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from memory_models import (
    ContextBudget, AgentMessage, MemoryEntry, 
    FlushResult, MemoryConfig
)


@dataclass
class ContextComponent:
    """A component of the context window."""
    name: str
    content: str
    tokens: int
    priority: int  # Higher = more important
    required: bool = False


class ContextWindowManager:
    """
    Manages LLM context window allocation and optimization.
    
    Features:
    - Token budget allocation across components
    - Context compression when approaching limits
    - Pre-compaction memory flush
    - Usage monitoring and optimization
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-5.2",
        budget: Optional[ContextBudget] = None
    ):
        self.model = llm_model
        self.budget = budget or ContextBudget()
        try:
            self.encoder = tiktoken.encoding_for_model(llm_model)
        except KeyError:
            logger.info(f"tiktoken: model '{llm_model}' not found, using cl100k_base encoding")
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Thresholds for actions
        self.COMPACTION_THRESHOLD = 0.85  # 85% full → compact
        self.FLUSH_THRESHOLD = 0.90        # 90% full → flush to memory
        
        # Pre-compaction flush tracking
        self.flushed_this_cycle = False
        self.last_flush_time: Optional[datetime] = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def count_message_tokens(self, messages: List[AgentMessage]) -> int:
        """Count tokens in message list."""
        total = 0
        for msg in messages:
            # Base tokens per message (OpenAI format)
            total += 4
            # Content tokens
            total += self.count_tokens(msg.content)
            # Role tokens
            total += self.count_tokens(msg.role)
        # Add buffer for formatting
        total += 2
        return total
    
    def build_context(
        self,
        system_prompt: str,
        skills: List[Dict[str, Any]],
        retrieved_memories: str,
        conversation_history: List[AgentMessage],
        current_request: str,
        agent_state: Dict[str, Any]
    ) -> List[AgentMessage]:
        """
        Build optimized context within budget constraints.
        
        Allocates tokens in priority order:
        1. System prompt (required)
        2. Critical skills (required)
        3. Retrieved memories (high priority)
        4. Conversation history (medium priority)
        5. Current request (required)
        6. Response reserve (always reserved)
        """
        context: List[AgentMessage] = []
        used_tokens = 0
        
        # 1. System prompt (highest priority, required)
        system_content = self._build_system_content(system_prompt, agent_state)
        system_tokens = self.count_tokens(system_content)
        
        if system_tokens > self.budget.system_tokens:
            # Compress system prompt if too long
            system_content = self._compress_text(
                system_content, 
                self.budget.system_tokens
            )
            system_tokens = self.count_tokens(system_content)
        
        context.append(AgentMessage(role='system', content=system_content))
        used_tokens += system_tokens + 4
        
        # 2. Active skills (high priority)
        skill_content = self._build_skills_content(skills)
        skill_tokens = self.count_tokens(skill_content)
        
        available_for_skills = self.budget.system_tokens + self.budget.skills_tokens - used_tokens
        if skill_tokens > available_for_skills:
            # Select only critical skills
            skill_content = self._select_critical_skills(skills)
            skill_tokens = self.count_tokens(skill_content)
        
        if skill_content:
            context.append(AgentMessage(
                role='system', 
                content=f"Available skills: {skill_content}"
            ))
            used_tokens += skill_tokens + 4
        
        # 3. Retrieved memories (high priority)
        memory_budget = self.budget.memory_tokens
        memory_tokens = self.count_tokens(retrieved_memories)
        
        if memory_tokens > memory_budget:
            # Truncate memories to fit
            retrieved_memories = self._truncate_to_tokens(
                retrieved_memories, 
                memory_budget
            )
            memory_tokens = self.count_tokens(retrieved_memories)
        
        if retrieved_memories:
            context.append(AgentMessage(
                role='system', 
                content=f"Relevant context:\n{retrieved_memories}"
            ))
            used_tokens += memory_tokens + 4
        
        # 4. Conversation history (medium priority)
        history_budget = self.budget.history_tokens
        history_tokens = self.count_message_tokens(conversation_history)
        
        if history_tokens > history_budget:
            # Compress history
            compressed_history = self._compress_history(
                conversation_history, 
                history_budget
            )
            context.extend(compressed_history)
            used_tokens += self.count_message_tokens(compressed_history)
        else:
            context.extend(conversation_history)
            used_tokens += history_tokens
        
        # 5. Current request (required)
        context.append(AgentMessage(role='user', content=current_request))
        used_tokens += self.count_tokens(current_request) + 4
        
        # Check utilization
        utilization = used_tokens / self.budget.total
        
        return context
    
    def _build_system_content(
        self, 
        system_prompt: str, 
        agent_state: Dict[str, Any]
    ) -> str:
        """Build full system prompt with state."""
        parts = [system_prompt]
        
        # Add agent state if relevant
        if agent_state.get('current_task'):
            parts.append(f"\nCurrent task: {agent_state['current_task']}")
        
        if agent_state.get('active_project'):
            parts.append(f"Active project: {agent_state['active_project']}")
        
        return '\n'.join(parts)
    
    def _build_skills_content(self, skills: List[Dict[str, Any]]) -> str:
        """Build skills description."""
        if not skills:
            return ""
        
        skill_descriptions = []
        for skill in skills:
            desc = f"- {skill['name']}: {skill.get('description', 'No description')}"
            skill_descriptions.append(desc)
        
        return '\n'.join(skill_descriptions)
    
    def _select_critical_skills(
        self, 
        skills: List[Dict[str, Any]]
    ) -> str:
        """Select only critical skills when space is limited."""
        # Sort by priority/importance
        sorted_skills = sorted(
            skills, 
            key=lambda s: s.get('priority', 0), 
            reverse=True
        )
        
        # Take top skills that fit
        selected = []
        total_tokens = 0
        budget = self.budget.skills_tokens
        
        for skill in sorted_skills:
            desc = f"- {skill['name']}: {skill.get('description', '')}"
            tokens = self.count_tokens(desc)
            
            if total_tokens + tokens <= budget:
                selected.append(desc)
                total_tokens += tokens
            else:
                break
        
        return '\n'.join(selected)
    
    def _compress_history(
        self,
        history: List[AgentMessage],
        budget: int
    ) -> List[AgentMessage]:
        """
        Compress conversation history to fit budget.
        
        Strategy:
        1. Keep last N exchanges verbatim (most recent = most important)
        2. Summarize older exchanges
        3. Preserve critical messages (errors, decisions, etc.)
        """
        # Keep last 6 messages (3 exchanges) verbatim
        recent = history[-6:] if len(history) > 6 else history
        recent_tokens = self.count_message_tokens(recent)
        
        if recent_tokens >= budget:
            # Even recent is too much - aggressive compression
            return self._aggressive_compress(recent, budget)
        
        # Summarize older messages
        older = history[:-6] if len(history) > 6 else []
        
        if older:
            summary_budget = budget - recent_tokens
            
            # Extract critical messages from older history
            critical = self._extract_critical_messages(older)
            critical_tokens = self.count_message_tokens(critical)
            
            if critical_tokens < summary_budget:
                # Can include critical messages + summary
                summary_budget -= critical_tokens
                summary = self._summarize_messages(older, summary_budget)
                
                compressed = [
                    AgentMessage(
                        role='system', 
                        content=f"Earlier conversation summary: {summary}"
                    )
                ] + critical + recent
            else:
                # Just include summary
                summary = self._summarize_messages(older, summary_budget)
                compressed = [
                    AgentMessage(
                        role='system', 
                        content=f"Earlier conversation summary: {summary}"
                    )
                ] + recent
        else:
            compressed = recent
        
        return compressed
    
    def _extract_critical_messages(
        self, 
        messages: List[AgentMessage]
    ) -> List[AgentMessage]:
        """Extract messages containing critical information."""
        critical = []
        critical_patterns = [
            'decision:', 'important:', 'remember:',
            'error:', 'exception:', 'failed',
            'api key', 'password', 'credential',
            'critical', 'urgent', 'emergency'
        ]
        
        for msg in messages:
            content_lower = msg.content.lower()
            if any(pattern in content_lower for pattern in critical_patterns):
                critical.append(msg)
        
        return critical
    
    def _summarize_messages(
        self,
        messages: List[AgentMessage],
        max_tokens: int
    ) -> str:
        """
        Generate a concise summary of messages.

        Attempts LLM-based summarization via OpenAIClient, falling back
        to extractive summarization if the LLM is unavailable.
        """
        # Build conversation text for both paths
        conversation_parts = []
        for msg in messages:
            prefix = "User" if msg.role == 'user' else "Agent"
            conversation_parts.append(f"{prefix}: {msg.content[:300]}")
        conversation_text = '\n'.join(conversation_parts)

        # Attempt LLM-based summarization
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            result = client.complete(
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize the following conversation segment concisely. "
                        "Preserve: key decisions, user requests, action items, important context. "
                        "Max 150 words.\n\n"
                        f"{conversation_text}"
                    )
                }],
                system="You are a conversation summarizer. Be concise and preserve critical information.",
                max_tokens=min(300, max_tokens),
                temperature=0.3,
            )
            summary = result.get('content', '').strip()
            if summary:
                # Ensure it fits in budget
                while self.count_tokens(summary) > max_tokens and len(summary) > 10:
                    summary = summary[:int(len(summary) * 0.8)]
                return summary
        except (ImportError, EnvironmentError, OSError, KeyError, TypeError, ValueError) as e:
            logging.getLogger(__name__).debug(f"LLM summarization unavailable, using extraction: {e}")

        # Fallback: extractive summarization
        key_points = []

        for msg in messages:
            if msg.role == 'user':
                lines = msg.content.split('\n')
                for line in lines[:2]:
                    stripped = line.strip()
                    if stripped:
                        key_points.append(f"User: {stripped[:100]}")
            elif msg.role == 'assistant':
                # Extract action-oriented lines and first line
                lines = msg.content.split('\n')
                for line in lines[:2]:
                    stripped = line.strip()
                    if stripped and any(kw in stripped.lower() for kw in
                                       ('action', 'completed', 'decided', 'created',
                                        'found', 'error', 'result', 'success', 'failed')):
                        key_points.append(f"Agent: {stripped[:100]}")
                        break
                else:
                    if lines and lines[0].strip():
                        key_points.append(f"Agent: {lines[0].strip()[:100]}")

        summary = ' | '.join(key_points[:8])

        # Ensure it fits in budget using token counting
        while self.count_tokens(summary) > max_tokens and len(summary) > 10:
            summary = summary[:int(len(summary) * 0.8)]

        return summary
    
    def _aggressive_compress(
        self,
        messages: List[AgentMessage],
        budget: int
    ) -> List[AgentMessage]:
        """Aggressive compression for extreme cases."""
        # Keep only most recent messages
        result = []
        used_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(msg.content) + 4
            
            if used_tokens + msg_tokens <= budget:
                result.insert(0, msg)
                used_tokens += msg_tokens
            else:
                break
        
        return result
    
    def _compress_text(self, text: str, max_tokens: int) -> str:
        """Compress text to fit token budget."""
        tokens = self.count_tokens(text)
        
        if tokens <= max_tokens:
            return text
        
        # Truncate to fit
        encoded = self.encoder.encode(text)
        truncated = encoded[:max_tokens]
        return self.encoder.decode(truncated)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        return self._compress_text(text, max_tokens)
    
    def should_flush(self, current_tokens: int) -> bool:
        """
        Determine if pre-compaction memory flush should be triggered.
        
        Formula: current >= total - response_reserve - soft_threshold
        """
        if self.flushed_this_cycle:
            return False
        
        soft_threshold = 4000  # tokens
        threshold = self.budget.total - self.budget.response_reserve - soft_threshold
        
        return current_tokens >= threshold
    
    async def trigger_pre_compaction_flush(
        self,
        conversation: List[AgentMessage],
        llm_client,
        memory_tools
    ) -> FlushResult:
        """
        Execute pre-compaction memory flush.
        
        This gives the agent a chance to write important memories
        before older context is lost to compaction.
        """
        flush_prompt = """
        [SYSTEM: Session nearing compaction]
        
        The conversation context is approaching its limit. Important information
        from earlier in this conversation may soon be lost.
        
        Please review the conversation and identify information that should be
        preserved for future sessions:
        
        1. Key decisions made
        2. User preferences expressed
        3. Important facts or context discovered
        4. Action items or todos
        5. Any information that would be valuable to remember
        
        Use the memory_write tool to store these memories.
        
        If there's nothing important to store, reply with exactly: NO_REPLY
        """
        
        # Add flush prompt to conversation
        flush_context = conversation + [
            AgentMessage(role='system', content=flush_prompt)
        ]
        
        # Get agent response (returns a plain string)
        response_text = await llm_client.generate(flush_context)
        if not isinstance(response_text, str):
            response_text = getattr(response_text, 'content', str(response_text))

        # Check for NO_REPLY
        if "NO_REPLY" in response_text or not response_text.strip():
            return FlushResult(written=False, reason="nothing_to_store")

        # Parse and execute memory writes from response text
        memories_written = []

        # Try to extract JSON memory entries from the response
        import re
        json_blocks = re.findall(r'\{[^{}]+\}', response_text)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if 'content' in parsed:
                    result = await memory_tools.write_memory(
                        content=parsed.get('content', ''),
                        category=parsed.get('category', 'semantic'),
                        importance=parsed.get('importance', 0.5),
                    )
                    if result.success:
                        memories_written.append(result.memory_id)
            except (json.JSONDecodeError, OSError, ConnectionError, TimeoutError, ValueError) as e:
                logging.getLogger(__name__).debug(f"Failed to write memory: {e}")
        
        self.flushed_this_cycle = True
        self.last_flush_time = datetime.now()
        
        return FlushResult(
            written=len(memories_written) > 0,
            memories_written=memories_written,
            reason="success" if memories_written else "no_writes_executed"
        )
    
    def reset_flush_cycle(self) -> None:
        """Reset flush tracking for new compaction cycle."""
        self.flushed_this_cycle = False
    
    def get_utilization(self, current_tokens: int) -> Dict[str, Any]:
        """Get context window utilization statistics."""
        return {
            'total_tokens': self.budget.total,
            'used_tokens': current_tokens,
            'available_tokens': self.budget.total - current_tokens,
            'utilization_percent': round((current_tokens / self.budget.total) * 100, 2),
            'should_compact': current_tokens / self.budget.total >= self.COMPACTION_THRESHOLD,
            'should_flush': self.should_flush(current_tokens),
            'flushed_this_cycle': self.flushed_this_cycle
        }


class ContextCompressor:
    """
    Advanced context compression using LLM-based summarization.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def compress(
        self,
        messages: List[AgentMessage],
        target_tokens: int
    ) -> List[AgentMessage]:
        """
        Compress messages using LLM summarization.
        
        This is more sophisticated than the basic compression
        in ContextWindowManager and produces higher quality summaries.
        """
        current_tokens = sum(self.manager.count_tokens(m.content) + 4 for m in messages)
        
        if current_tokens <= target_tokens:
            return messages
        
        # Separate messages by importance
        system_msgs = [m for m in messages if m.role == 'system']
        user_msgs = [m for m in messages if m.role == 'user']
        assistant_msgs = [m for m in messages if m.role == 'assistant']
        
        # Keep recent messages verbatim
        recent_count = min(6, len(user_msgs))
        recent_user = user_msgs[-recent_count:] if user_msgs else []
        recent_assistant = assistant_msgs[-recent_count:] if assistant_msgs else []
        
        # Summarize older messages
        older_user = user_msgs[:-recent_count] if len(user_msgs) > recent_count else []
        older_assistant = assistant_msgs[:-recent_count] if len(assistant_msgs) > recent_count else []
        
        compressed = list(system_msgs)
        
        # Summarize older conversation
        if older_user or older_assistant:
            summary = await self._generate_summary(older_user, older_assistant)
            compressed.append(AgentMessage(
                role='system',
                content=f"Summary of earlier conversation: {summary}"
            ))
        
        # Add recent messages
        compressed.extend(recent_user)
        compressed.extend(recent_assistant)
        
        return compressed
    
    async def _generate_summary(
        self,
        user_msgs: List[AgentMessage],
        assistant_msgs: List[AgentMessage]
    ) -> str:
        """Generate LLM-based summary of conversation."""
        # Build conversation text
        conversation = []
        for msg in user_msgs + assistant_msgs:
            prefix = "User" if msg.role == 'user' else "Assistant"
            conversation.append(f"{prefix}: {msg.content}")
        
        conversation_text = '\n'.join(conversation)
        
        prompt = f"""
        Summarize the following conversation segment concisely.
        Preserve: key decisions, user requests, action items, important context.
        
        Conversation:
        {conversation_text}
        
        Summary (max 200 words):
        """
        
        summary = await self.llm.generate(prompt, max_tokens=300)
        return summary.strip()
