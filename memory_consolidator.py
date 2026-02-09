"""
Memory Consolidation Module
Converts ephemeral memories (daily logs) to durable semantic memory
"""

import re
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from memory_models import (
    DailyLogEntry, MemoryEntry, MemoryType, 
    ConsolidationReport, SemanticSummary, Fact,
    WriteResult
)


@dataclass
class ParsedEvent:
    """An event parsed from a daily log."""
    timestamp: datetime
    event_type: str
    title: str
    content: str
    importance: float
    metadata: Dict[str, Any]


class DailyLogParser:
    """Parser for daily log markdown files."""
    
    # Event type patterns for importance scoring
    EVENT_TYPE_IMPORTANCE = {
        'decision': 0.9,
        'milestone': 0.85,
        'error': 0.8,
        'user_feedback': 0.75,
        'action': 0.6,
        'observation': 0.5,
        'routine': 0.3
    }
    
    def parse_file(self, file_path: Path) -> List[ParsedEvent]:
        """Parse a daily log file into events."""
        content = file_path.read_text(encoding='utf-8')
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[ParsedEvent]:
        """Parse daily log content into events."""
        events = []
        
        # Extract date from header
        date_match = re.search(r'# Daily Log: (\d{4}-\d{2}-\d{2})', content)
        if date_match:
            log_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
        else:
            log_date = datetime.now().date()
        
        # Parse event sections
        event_sections = re.split(r'\n### ', content)
        
        for section in event_sections[1:]:  # Skip header
            event = self._parse_event_section(section, log_date)
            if event:
                events.append(event)
        
        return events
    
    def _parse_event_section(
        self, 
        section: str, 
        log_date: datetime.date
    ) -> Optional[ParsedEvent]:
        """Parse a single event section."""
        lines = section.strip().split('\n')
        if not lines:
            return None
        
        # Parse header: "HH:MM - Title"
        header_match = re.match(r'(\d{2}:\d{2})\s+-\s+(.+)', lines[0])
        if not header_match:
            return None
        
        time_str, title = header_match.groups()
        hour, minute = map(int, time_str.split(':'))
        timestamp = datetime.combine(
            log_date, 
            datetime.min.time().replace(hour=hour, minute=minute)
        )
        
        # Parse content and metadata
        event_type = 'general'
        content_lines = []
        metadata = {}
        
        i = 1
        while i < len(lines):
            line = lines[i]
            
            # Check for event type
            if line.startswith('**Type**:'):
                event_type = line.replace('**Type**:', '').strip().lower()
            
            # Check for metadata
            elif line.startswith('**Metadata**:'):
                i += 1
                while i < len(lines) and lines[i].startswith('- '):
                    parts = lines[i][2:].split(': ', 1)
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
                    i += 1
                continue
            
            # Regular content
            elif line.strip() and not line.startswith('**'):
                content_lines.append(line)
            
            i += 1
        
        content = '\n'.join(content_lines).strip()
        
        # Calculate importance
        importance = self._calculate_importance(event_type, content, metadata)
        
        return ParsedEvent(
            timestamp=timestamp,
            event_type=event_type,
            title=title,
            content=content,
            importance=importance,
            metadata=metadata
        )
    
    def _calculate_importance(
        self,
        event_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate importance score for an event."""
        # Base importance from event type
        base_score = self.EVENT_TYPE_IMPORTANCE.get(event_type, 0.5)
        
        # Content-based adjustments
        content_lower = content.lower()
        
        # Keywords indicating importance
        important_keywords = [
            'decided', 'decision', 'chose', 'selected',
            'important', 'critical', 'essential',
            'user prefers', 'user likes', 'user wants',
            'error', 'bug', 'issue', 'problem',
            'completed', 'finished', 'deployed'
        ]
        
        keyword_bonus = 0
        for keyword in important_keywords:
            if keyword in content_lower:
                keyword_bonus += 0.05
        
        keyword_bonus = min(0.2, keyword_bonus)  # Cap at 0.2
        
        # User feedback bonus
        feedback_bonus = 0.1 if 'user_feedback' in metadata else 0
        
        return min(1.0, base_score + keyword_bonus + feedback_bonus)


class MemoryConsolidator:
    """
    Consolidates ephemeral memories into durable semantic memory.
    
    Process:
    1. Parse daily logs older than threshold
    2. Extract important events
    3. Generate semantic summaries using LLM
    4. Update MEMORY.md with consolidated facts
    5. Archive processed logs
    """
    
    def __init__(
        self,
        memory_manager,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        self.memory = memory_manager
        self.llm = llm_client
        self.parser = DailyLogParser()
        
        # Configuration
        self.config = config or {}
        self.age_threshold_days = self.config.get('age_threshold_days', 7)
        self.importance_threshold = self.config.get('importance_threshold', 0.6)
        self.max_daily_logs = self.config.get('max_daily_logs', 30)
    
    async def consolidate(self, force: bool = False) -> ConsolidationReport:
        """
        Run memory consolidation process.
        
        Args:
            force: Force consolidation even if not scheduled
            
        Returns:
            Consolidation report with statistics
        """
        report = ConsolidationReport()
        report.start_time = datetime.now()
        
        # Get logs eligible for consolidation
        old_logs = self._get_consolidation_candidates()
        
        for log_file in old_logs:
            try:
                # Parse daily log
                events = self.parser.parse_file(log_file)
                
                # Filter to important events
                important_events = [
                    e for e in events 
                    if e.importance >= self.importance_threshold
                ]
                
                if important_events:
                    # Generate semantic summary
                    summary = await self._generate_semantic_summary(
                        important_events,
                        log_file.stem  # Date from filename
                    )
                    
                    # Update durable memory
                    await self._update_memory_md(summary, log_file.stem)
                    
                    # Update report
                    report.events_processed += len(events)
                    report.facts_extracted += len(summary.facts)
                    report.files_processed.append(log_file.name)
                
                # Archive the log
                await self._archive_log(log_file)
                
            except (OSError, ValueError) as e:
                report.errors.append(f"{log_file.name}: {str(e)}")
        
        # Clean up old archives
        await self._cleanup_archives()
        
        report.end_time = datetime.now()
        return report
    
    def _get_consolidation_candidates(self) -> List[Path]:
        """Get daily logs that are old enough to consolidate."""
        daily_dir = self.memory.config.daily_dir
        
        if not daily_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=self.age_threshold_days)
        candidates = []
        
        for log_file in sorted(daily_dir.glob('*.md')):
            # Parse date from filename (YYYY-MM-DD.md)
            try:
                file_date = datetime.strptime(log_file.stem, '%Y-%m-%d')
                if file_date < cutoff_date:
                    candidates.append(log_file)
            except ValueError:
                continue
        
        return candidates
    
    async def _generate_semantic_summary(
        self,
        events: List[ParsedEvent],
        date_str: str
    ) -> SemanticSummary:
        """
        Use LLM to extract semantic facts from events.
        """
        # Format events for LLM
        events_text = self._format_events_for_llm(events)
        
        prompt = f"""
        Analyze the following events from {date_str} and extract important 
        facts, decisions, and patterns that should be remembered long-term.
        
        Events:
        {events_text}
        
        Extract the following:
        1. Key decisions made
        2. User preferences expressed
        3. Important project updates
        4. Recurring patterns observed
        5. Relationship or contact information
        
        Return as JSON:
        {{
            "facts": [
                {{
                    "content": "Clear, concise fact statement",
                    "category": "decision|preference|project|pattern|relationship",
                    "importance": 0.0-1.0,
                    "confidence": 0.0-1.0
                }}
            ],
            "summary": "Brief narrative summary of the day's important events"
        }}
        
        Guidelines:
        - Be specific and factual
        - Include dates and names when relevant
        - Focus on durable information (not temporary states)
        - Assign higher importance to user preferences and decisions
        """
        
        response = await self.llm.generate(
            prompt,
            response_format={"type": "json_object"}
        )
        
        try:
            data = json.loads(response)
            facts = [Fact(**f) for f in data.get('facts', [])]
            
            return SemanticSummary(
                facts=facts,
                summary=data.get('summary', ''),
                source_date=datetime.strptime(date_str, '%Y-%m-%d')
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create simple summary
            return self._create_fallback_summary(events, date_str)
    
    def _format_events_for_llm(self, events: List[ParsedEvent]) -> str:
        """Format events for LLM consumption."""
        formatted = []
        
        for event in events:
            time_str = event.timestamp.strftime('%H:%M')
            formatted.append(f"""
[{time_str}] {event.title}
Type: {event.event_type}
Importance: {event.importance:.2f}
Content: {event.content}
---
""")
        
        return '\n'.join(formatted)
    
    def _create_fallback_summary(
        self,
        events: List[ParsedEvent],
        date_str: str
    ) -> SemanticSummary:
        """Create a simple summary when LLM fails."""
        facts = []
        
        for event in events:
            fact = Fact(
                content=f"{event.title}: {event.content[:200]}",
                category=event.event_type,
                importance=event.importance,
                confidence=0.7
            )
            facts.append(fact)
        
        summary = f"Consolidated {len(events)} important events from {date_str}"
        
        return SemanticSummary(
            facts=facts,
            summary=summary,
            source_date=datetime.strptime(date_str, '%Y-%m-%d')
        )
    
    async def _update_memory_md(
        self,
        summary: SemanticSummary,
        date_str: str
    ) -> None:
        """Update MEMORY.md with consolidated facts."""
        memory_md_path = self.memory.config.memory_dir / 'MEMORY.md'
        
        # Read existing content
        if memory_md_path.exists():
            existing_content = memory_md_path.read_text(encoding='utf-8')
        else:
            existing_content = self._create_memory_md_template()
        
        # Generate new content
        new_section = self._format_consolidated_section(summary, date_str)
        
        # Insert into appropriate section
        updated_content = self._insert_into_memory_md(
            existing_content, 
            new_section,
            summary.facts
        )
        
        # Write back
        memory_md_path.write_text(updated_content, encoding='utf-8')
        
        # Also index in vector store
        for fact in summary.facts:
            await self.memory.remember(
                content=fact.content,
                category=fact.category,
                importance=fact.importance,
                immediate=True
            )
    
    def _create_memory_md_template(self) -> str:
        """Create template for new MEMORY.md file."""
        return """# Agent Memory

> Last Updated: {date}
> Version: 1.0

---

## User Preferences

## Project Context

## Important Decisions

## Recurring Tasks

## Relationships & Contacts

## Learned Patterns

## Consolidated Events

""".format(date=datetime.now().strftime('%Y-%m-%d'))
    
    def _format_consolidated_section(
        self,
        summary: SemanticSummary,
        date_str: str
    ) -> str:
        """Format consolidated facts as markdown section."""
        lines = [f"\n### Consolidated from {date_str}\n"]
        lines.append(f"{summary.summary}\n")
        
        for fact in summary.facts:
            lines.append(f"- **{fact.category.upper()}**: {fact.content}")
            lines.append(f"  (importance: {fact.importance:.2f}, confidence: {fact.confidence:.2f})")
        
        lines.append("")
        return '\n'.join(lines)
    
    def _insert_into_memory_md(
        self,
        content: str,
        new_section: str,
        facts: List[Fact]
    ) -> str:
        """Insert new section into appropriate part of MEMORY.md."""
        # For now, append to "Consolidated Events" section
        # In production, this would be more sophisticated
        
        # Find "Consolidated Events" section
        if '## Consolidated Events' in content:
            # Insert after section header
            parts = content.split('## Consolidated Events', 1)
            return parts[0] + '## Consolidated Events' + new_section + parts[1]
        else:
            # Append to end
            return content + new_section
    
    async def _archive_log(self, log_file: Path) -> None:
        """Archive a processed daily log."""
        archive_dir = self.memory.config.sessions_dir / 'archived_logs'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move to archive
        archive_path = archive_dir / log_file.name
        log_file.rename(archive_path)
    
    async def _cleanup_archives(self) -> None:
        """Clean up old archived logs."""
        archive_dir = self.memory.config.sessions_dir / 'archived_logs'
        
        if not archive_dir.exists():
            return
        
        # Delete logs older than 1 year
        cutoff = datetime.now() - timedelta(days=365)
        
        for log_file in archive_dir.glob('*.md'):
            try:
                file_date = datetime.strptime(log_file.stem, '%Y-%m-%d')
                if file_date < cutoff:
                    log_file.unlink()
            except ValueError:
                continue


class ImportanceScorer:
    """
    Calculates importance scores for memories.
    
    Factors:
    - User feedback (explicit importance signals)
    - Content type (decisions > observations)
    - Recency (newer = more relevant)
    - Access frequency (frequently accessed = more important)
    - Semantic significance (detected via LLM)
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def score(
        self,
        content: str,
        event_type: str = 'general',
        user_feedback: Optional[str] = None,
        access_count: int = 0,
        age_days: int = 0
    ) -> float:
        """Calculate importance score."""
        score = 0.5  # Base score
        
        # Event type weight
        type_weights = {
            'decision': 0.3,
            'user_feedback': 0.25,
            'milestone': 0.2,
            'error': 0.15,
            'action': 0.1,
            'observation': 0.05,
            'routine': 0.0
        }
        score += type_weights.get(event_type, 0)
        
        # User feedback
        if user_feedback:
            if 'important' in user_feedback.lower():
                score += 0.2
            if 'remember' in user_feedback.lower():
                score += 0.25
        
        # Access frequency (diminishing returns)
        if access_count > 0:
            score += min(0.15, access_count * 0.03)
        
        # Recency decay
        recency_factor = max(0.1, 1.0 - (age_days / 365))
        score *= recency_factor
        
        return min(1.0, max(0.0, score))
    
    async def semantic_score(self, content: str) -> float:
        """Use LLM to score semantic importance."""
        if not self.llm:
            return 0.5
        
        prompt = f"""
        Rate the importance of the following information for an AI assistant
        to remember long-term. Consider:
        - Is this a user preference?
        - Is this a key decision?
        - Would this be useful in future conversations?
        
        Content: {content}
        
        Importance (0.0-1.0):
        """
        
        try:
            response = await self.llm.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, TypeError):
            return 0.5
