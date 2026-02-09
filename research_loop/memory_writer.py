"""
Memory Writer - Knowledge Consolidation into MEMORY.md
Handles writing synthesized knowledge to persistent memory storage.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from .models import (
    SynthesisResult, ResearchTask, ConsolidationResult, MergedKnowledge,
    GraphUpdate, DiscoveredSource, Fact
)


logger = logging.getLogger(__name__)


class MemoryWriter:
    """
    Write synthesized knowledge to MEMORY.md and knowledge graph.
    
    Handles:
    - MEMORY.md structure and formatting
    - Knowledge graph updates
    - Cross-reference management
    - Source tracking
    """
    
    MEMORY_SECTIONS = [
        "# Knowledge Base",
        "# Research History",
        "# Source Library",
        "# Knowledge Gaps",
        "# Confidence Map",
        "# Update Log"
    ]
    
    def __init__(self, memory_path: str = "MEMORY.md"):
        self.memory_path = Path(memory_path)
        # Dict-based knowledge graph: topic -> {related_topics, facts, last_updated}
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.update_tracker = UpdateTracker()
    
    async def consolidate(
        self,
        synthesis_result: SynthesisResult,
        research_task: ResearchTask
    ) -> ConsolidationResult:
        """
        Consolidate research findings into memory.
        
        Args:
            synthesis_result: Synthesis result to consolidate
            research_task: Original research task
            
        Returns:
            Consolidation result with statistics
        """
        logger.info(f"Consolidating research: {research_task.topic}")
        
        result = ConsolidationResult()
        
        # Step 1: Check for existing knowledge
        existing = await self._find_existing_knowledge(synthesis_result)
        
        # Step 2: Merge or create new entries
        if existing:
            merged = await self._merge_with_existing(
                synthesis_result,
                existing
            )
        else:
            merged = await self._create_new_entry(
                synthesis_result,
                research_task
            )
        
        # Step 3: Update knowledge graph dict with new topic and facts
        topic = synthesis_result.synthesis.topic
        facts_list = [kp for kp in synthesis_result.synthesis.key_points if kp]
        if topic not in self.knowledge_graph:
            self.knowledge_graph[topic] = {
                "related_topics": [],
                "facts": facts_list,
                "last_updated": datetime.now().isoformat()
            }
            result.knowledge_graph_updates = 1 + len(facts_list)
        else:
            existing_facts = self.knowledge_graph[topic].get("facts", [])
            new_facts = [f for f in facts_list if f not in existing_facts]
            existing_facts.extend(new_facts)
            self.knowledge_graph[topic]["facts"] = existing_facts
            self.knowledge_graph[topic]["last_updated"] = datetime.now().isoformat()
            result.knowledge_graph_updates = len(new_facts)
        
        # Step 4: Write to MEMORY.md
        await self._write_to_memory(merged, research_task)
        result.entries_created = len(merged.new_entries)
        result.entries_updated = len(merged.updated_entries)
        
        # Step 5: Update cross-references
        cross_refs = await self._update_cross_references(merged)
        result.cross_references_added = cross_refs
        
        logger.info(
            f"Consolidation complete: {result.entries_created} new, "
            f"{result.entries_updated} updated, {result.cross_references_added} cross-refs"
        )
        
        return result
    
    async def _find_existing_knowledge(
        self,
        synthesis: SynthesisResult
    ) -> Optional[Dict[str, Any]]:
        """Find existing knowledge about the topic"""
        if not self.memory_path.exists():
            return None
        
        try:
            content = self.memory_path.read_text(encoding='utf-8')
            
            # Look for existing entry
            topic = synthesis.synthesis.topic
            pattern = rf"## {re.escape(topic)}\b"
            
            if re.search(pattern, content, re.IGNORECASE):
                # Extract existing entry
                match = re.search(
                    rf"## {re.escape(topic)}\b.*?(?=\n## |\Z)",
                    content,
                    re.IGNORECASE | re.DOTALL
                )
                if match:
                    return {
                        "content": match.group(0),
                        "position": match.start()
                    }
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Error reading memory: {e}")
        
        return None
    
    async def _merge_with_existing(
        self,
        synthesis: SynthesisResult,
        existing: Dict[str, Any]
    ) -> MergedKnowledge:
        """Merge new synthesis with existing knowledge"""
        # Extract key facts from synthesis key_points
        key_facts = self._extract_key_facts(synthesis)

        # Extract sources from consensus areas
        sources = self._extract_sources(synthesis)

        # Create merged knowledge
        merged = MergedKnowledge(
            topic=synthesis.synthesis.topic,
            summary=synthesis.summaries.get("brief", Summary(level="brief", content="")).content,
            key_facts=key_facts,
            detailed_content=synthesis.synthesis.content,
            sources=sources,
            confidence_score=synthesis.confidence_score,
            related_topics=list(set(
                synthesis.synthesis.knowledge_gaps + synthesis.synthesis.recommendations
            ))[:10]
        )

        # Mark as updated
        merged.updated_entries.append(synthesis.synthesis.topic)

        return merged
    
    async def _create_new_entry(
        self,
        synthesis: SynthesisResult,
        task: ResearchTask
    ) -> MergedKnowledge:
        """Create new knowledge entry"""
        # Extract real key facts and sources from synthesis
        key_facts = self._extract_key_facts(synthesis)
        sources = self._extract_sources(synthesis)

        # Create merged knowledge
        merged = MergedKnowledge(
            topic=synthesis.synthesis.topic,
            summary=synthesis.summaries.get("brief", Summary(level="brief", content="")).content,
            key_facts=key_facts,
            detailed_content=synthesis.synthesis.content,
            sources=sources,
            confidence_score=synthesis.confidence_score,
            related_topics=list(set(
                synthesis.synthesis.knowledge_gaps + synthesis.synthesis.recommendations
            ))[:10]
        )

        # Mark as new
        merged.new_entries.append(synthesis.synthesis.topic)

        return merged
    
    async def _write_to_memory(
        self,
        merged: MergedKnowledge,
        task: ResearchTask
    ):
        """Write knowledge to MEMORY.md"""
        # Ensure directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing memory
        existing_content = ""
        if self.memory_path.exists():
            existing_content = self.memory_path.read_text(encoding='utf-8')
        
        # Generate memory entry
        entry = self._generate_memory_entry(merged, task)
        
        # Determine section
        section = self._determine_section(merged)
        
        # Update content
        updated_content = self._update_section(
            existing_content,
            section,
            entry,
            merged.topic
        )
        
        # Add to update log
        updated_content = self._add_update_log(
            updated_content,
            task,
            merged
        )
        
        # Write to file
        self.memory_path.write_text(updated_content, encoding='utf-8')
        
        logger.debug(f"Written to memory: {self.memory_path}")
    
    def _generate_memory_entry(
        self,
        merged: MergedKnowledge,
        task: ResearchTask
    ) -> str:
        """Generate formatted memory entry"""
        entry = f"""
## {merged.topic}

**Added:** {datetime.now().isoformat()}  
**Research ID:** {task.id}  
**Confidence:** {merged.confidence_score:.2f}  
**Sources:** {len(merged.sources)}

### Summary
{merged.summary}

### Key Facts
"""
        
        for fact in merged.key_facts[:10]:
            confidence_marker = "✓" if fact.confidence == "high" else "~" if fact.confidence == "medium" else "?"
            entry += f"- {confidence_marker} {fact.statement}\n"
        
        if merged.detailed_content:
            entry += "\n### Detailed Information\n"
            # Truncate if too long
            detail = merged.detailed_content[:2000]
            if len(merged.detailed_content) > 2000:
                detail += "\n\n*[Content truncated for brevity]*"
            entry += detail + "\n"
        
        if merged.sources:
            entry += "\n### Source References\n"
            for source in merged.sources[:5]:
                entry += f"- [{source.title or 'Source'}]({source.url})\n"
        
        if merged.related_topics:
            entry += "\n### Related Topics\n"
            for related in merged.related_topics[:10]:
                entry += f"- [[{related}]]\n"
        
        entry += "\n---\n"
        
        return entry
    
    def _determine_section(self, merged: MergedKnowledge) -> str:
        """Determine which section to add entry to"""
        # Simple logic - could be more sophisticated
        if merged.confidence_score > 0.8:
            return "# Knowledge Base"
        elif merged.confidence_score > 0.5:
            return "# Research History"
        else:
            return "# Knowledge Gaps"
    
    def _update_section(
        self,
        content: str,
        section: str,
        entry: str,
        topic: str
    ) -> str:
        """Update a section in memory content"""
        # Ensure section exists
        if section not in content:
            content += f"\n\n{section}\n\n"
        
        # Check if entry already exists
        pattern = rf"## {re.escape(topic)}\b"
        
        if re.search(pattern, content, re.IGNORECASE):
            # Update existing entry
            updated = re.sub(
                rf"(## {re.escape(topic)}\b.*?(?=\n## |\Z))",
                entry.strip(),
                content,
                flags=re.IGNORECASE | re.DOTALL
            )
            return updated
        else:
            # Add new entry after section header
            section_pattern = rf"({re.escape(section)}.*?\n)"
            match = re.search(section_pattern, content, re.DOTALL)
            
            if match:
                insert_pos = match.end()
                return content[:insert_pos] + entry + content[insert_pos:]
            else:
                # Append to end
                return content + entry
    
    def _add_update_log(
        self,
        content: str,
        task: ResearchTask,
        merged: MergedKnowledge
    ) -> str:
        """Add entry to update log"""
        log_entry = f"""
- **{datetime.now().isoformat()}** - {task.topic}
  - Type: {task.trigger_type.value}
  - Confidence: {merged.confidence_score:.2f}
  - Sources: {len(merged.sources)}
  - Task ID: {task.id}
"""
        
        # Find or create update log section
        if "# Update Log" not in content:
            content += "\n\n# Update Log\n"
        
        # Add to beginning of log
        pattern = r"(# Update Log\n)"
        match = re.search(pattern, content)
        
        if match:
            insert_pos = match.end()
            return content[:insert_pos] + log_entry + content[insert_pos:]
        
        return content + log_entry
    
    async def _update_cross_references(self, merged: MergedKnowledge) -> int:
        """Update cross-references between topics in the knowledge graph."""
        count = 0
        topic = merged.topic

        # Ensure the main topic exists in the knowledge graph
        if topic not in self.knowledge_graph:
            self.knowledge_graph[topic] = {
                "related_topics": [],
                "facts": [],
                "last_updated": datetime.now().isoformat()
            }

        # Add facts to the graph
        for fact in merged.key_facts:
            if fact.statement not in self.knowledge_graph[topic].get("facts", []):
                self.knowledge_graph[topic].setdefault("facts", []).append(fact.statement)

        self.knowledge_graph[topic]["last_updated"] = datetime.now().isoformat()

        # Build bidirectional links between this topic and related topics
        for related in merged.related_topics:
            # Add forward link
            existing_related = self.knowledge_graph[topic].get("related_topics", [])
            if related not in existing_related:
                existing_related.append(related)
                self.knowledge_graph[topic]["related_topics"] = existing_related
                count += 1

            # Add reverse link
            if related not in self.knowledge_graph:
                self.knowledge_graph[related] = {
                    "related_topics": [topic],
                    "facts": [],
                    "last_updated": datetime.now().isoformat()
                }
                count += 1
            else:
                reverse_related = self.knowledge_graph[related].get("related_topics", [])
                if topic not in reverse_related:
                    reverse_related.append(topic)
                    self.knowledge_graph[related]["related_topics"] = reverse_related
                    count += 1

        # Also link topics that share key facts with existing entries
        for other_topic, other_info in self.knowledge_graph.items():
            if other_topic == topic:
                continue
            other_facts = set(other_info.get("facts", []))
            my_facts = set(self.knowledge_graph[topic].get("facts", []))
            if other_facts & my_facts:
                # Shared facts create a cross-reference
                if other_topic not in self.knowledge_graph[topic].get("related_topics", []):
                    self.knowledge_graph[topic].setdefault("related_topics", []).append(other_topic)
                    count += 1
                if topic not in other_info.get("related_topics", []):
                    other_info.setdefault("related_topics", []).append(topic)
                    count += 1

        self.update_tracker.track({
            "type": "cross_reference_update",
            "topic": topic,
            "references_added": count
        })

        return count

    def _extract_key_facts(self, synthesis: SynthesisResult) -> List[Fact]:
        """Extract Fact objects from synthesis key_points and content."""
        facts = []

        # Extract from key_points
        for point in synthesis.synthesis.key_points:
            if not point or len(point.strip()) < 5:
                continue
            # Determine confidence from synthesis confidence score
            if synthesis.confidence_score > 0.7:
                confidence = "high"
            elif synthesis.confidence_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            facts.append(Fact(
                statement=point.strip(),
                category="key_finding",
                confidence=confidence,
                source_attribution=synthesis.synthesis.topic
            ))

        # Extract bullet points from synthesis content (lines starting with - or *)
        if synthesis.synthesis.content:
            for line in synthesis.synthesis.content.splitlines():
                stripped = line.strip()
                if stripped.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
                    # Clean up the bullet point
                    text = re.sub(r'^[-*\d.)\s]+', '', stripped).strip()
                    if len(text) > 10 and text not in [f.statement for f in facts]:
                        facts.append(Fact(
                            statement=text,
                            category="detail",
                            confidence="medium",
                            source_attribution=synthesis.synthesis.topic
                        ))

        return facts[:20]  # Limit to 20 key facts

    def _extract_sources(self, synthesis: SynthesisResult) -> List[DiscoveredSource]:
        """Extract source references from synthesis consensus areas."""
        sources = []
        seen_urls = set()

        for area in synthesis.consensus_areas:
            for source_name in area.supporting_sources:
                if source_name not in seen_urls:
                    seen_urls.add(source_name)
                    sources.append(DiscoveredSource(
                        url=source_name if source_name.startswith("http") else f"source://{source_name}",
                        title=source_name,
                        engine="synthesis",
                        query=synthesis.synthesis.topic,
                        rank=0,
                        credibility_score=area.confidence
                    ))

        return sources


class UpdateTracker:
    """Track memory updates"""
    
    def __init__(self):
        self.updates: List[Dict[str, Any]] = []
    
    def track(self, update: Dict[str, Any]):
        """Track an update"""
        update["timestamp"] = datetime.now().isoformat()
        self.updates.append(update)
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent updates"""
        return self.updates[-count:]


class Summary:
    """Lightweight summary fallback used when models.Summary is not available in dict lookups."""
    def __init__(self, level: str, content: str):
        self.level = level
        self.content = content
