"""
Synthesis and Summarization System
Multi-source synthesis, conflict resolution, and multi-level summarization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .models import (
    ResearchTask, ExtractedInformation, SynthesisResult, Synthesis,
    Summary, ResolvedFact, ConsensusArea, DisagreementArea,
    Entity, Fact, Relationship, Claim
)


logger = logging.getLogger(__name__)


class SynthesisEngine:
    """
    Multi-source synthesis engine.
    
    Synthesizes information from multiple sources:
    - Aggregates facts by topic
    - Resolves conflicts
    - Identifies consensus and disagreements
    - Generates comprehensive synthesis
    """
    
    def __init__(self):
        self.conflict_resolver = ConflictResolver()
        self.summarization_engine = SummarizationEngine()
    
    async def synthesize(
        self,
        extracted_info: List[ExtractedInformation],
        research_task: ResearchTask
    ) -> SynthesisResult:
        """
        Synthesize information from multiple sources.
        
        Args:
            extracted_info: List of extracted information from sources
            research_task: Original research task
            
        Returns:
            Complete synthesis result
        """
        logger.info(f"Synthesizing {len(extracted_info)} sources")
        
        if not extracted_info:
            return SynthesisResult(
                synthesis=Synthesis(
                    content="No information available",
                    topic=research_task.topic
                ),
                confidence_score=0.0,
                sources_used=0
            )
        
        # Step 1: Aggregate facts by topic
        topic_facts = self._aggregate_by_topic(extracted_info)
        
        # Step 2: Resolve conflicts
        resolved_facts = await self.conflict_resolver.resolve(topic_facts)
        
        # Step 3: Identify consensus and disagreements
        consensus = self._identify_consensus(resolved_facts)
        disagreements = self._identify_disagreements(resolved_facts)
        
        # Step 4: Generate synthesis
        synthesis = await self._generate_synthesis(
            resolved_facts,
            consensus,
            disagreements,
            research_task
        )
        
        # Step 5: Create summaries at different levels
        summaries = await self.summarization_engine.create_summaries(synthesis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(resolved_facts, extracted_info)
        
        return SynthesisResult(
            synthesis=synthesis,
            summaries=summaries,
            consensus_areas=consensus,
            disagreement_areas=disagreements,
            confidence_score=confidence,
            sources_used=len(extracted_info),
            facts_synthesized=len(resolved_facts),
            synthesis_timestamp=datetime.now()
        )
    
    def _aggregate_by_topic(
        self,
        extracted_info: List[ExtractedInformation]
    ) -> Dict[str, List[Fact]]:
        """Aggregate facts by topic"""
        topic_facts = defaultdict(list)
        
        for info in extracted_info:
            for fact in info.facts:
                # Use category as topic key
                topic = fact.category
                topic_facts[topic].append(fact)
        
        return dict(topic_facts)
    
    def _identify_consensus(
        self,
        resolved_facts: List[ResolvedFact]
    ) -> List[ConsensusArea]:
        """Identify areas of consensus"""
        consensus_areas = []
        
        # Group facts by similarity
        fact_groups = self._group_similar_facts(resolved_facts)
        
        for group in fact_groups:
            if len(group) > 1:  # Multiple sources agree
                sources = []
                for fact in group:
                    sources.extend(fact.sources)
                
                consensus_areas.append(ConsensusArea(
                    topic=group[0].statement[:50],
                    agreed_facts=[f.statement for f in group],
                    supporting_sources=list(set(sources)),
                    confidence=min(0.95, 0.7 + len(group) * 0.05)
                ))
        
        return consensus_areas
    
    def _identify_disagreements(
        self,
        resolved_facts: List[ResolvedFact]
    ) -> List[DisagreementArea]:
        """Identify areas of disagreement"""
        disagreement_areas = []
        
        # Look for facts with alternative statements
        for fact in resolved_facts:
            if fact.alternative_statements:
                positions = [fact.statement] + fact.alternative_statements
                
                sources_by_position = {
                    fact.statement: fact.sources
                }
                
                disagreement_areas.append(DisagreementArea(
                    topic=fact.statement[:50],
                    conflicting_positions=positions,
                    sources_by_position=sources_by_position,
                    analysis="Conflicting information from multiple sources"
                ))
        
        return disagreement_areas
    
    async def _generate_synthesis(
        self,
        facts: List[ResolvedFact],
        consensus: List[ConsensusArea],
        disagreements: List[DisagreementArea],
        task: ResearchTask
    ) -> Synthesis:
        """Generate comprehensive synthesis"""
        
        # Build synthesis content
        content_parts = []
        
        # Executive summary
        content_parts.append(self._generate_executive_summary(facts, task))
        
        # Key findings
        content_parts.append(self._generate_key_findings(facts))
        
        # Detailed analysis
        content_parts.append(self._generate_detailed_analysis(facts))
        
        # Consensus areas
        if consensus:
            content_parts.append(self._generate_consensus_section(consensus))
        
        # Disagreement areas
        if disagreements:
            content_parts.append(self._generate_disagreement_section(disagreements))
        
        # Implications
        content_parts.append(self._generate_implications(facts))
        
        # Knowledge gaps
        content_parts.append(self._generate_knowledge_gaps(facts))
        
        # Combine
        full_content = "\n\n".join(content_parts)
        
        # Extract key points
        key_points = [f.statement for f in facts[:10]]
        
        return Synthesis(
            content=full_content,
            topic=task.topic,
            key_points=key_points,
            implications=self._extract_implications(facts),
            knowledge_gaps=self._identify_gaps(facts),
            recommendations=self._generate_recommendations(facts, task)
        )
    
    def _generate_executive_summary(
        self,
        facts: List[ResolvedFact],
        task: ResearchTask
    ) -> str:
        """Generate executive summary"""
        summary = f"""## Executive Summary

This research synthesis covers **{task.topic}** based on analysis of multiple sources.

**Key Points:**
"""
        
        for i, fact in enumerate(facts[:5], 1):
            summary += f"{i}. {fact.statement}\n"
        
        summary += f"\n**Confidence Level:** {self._calculate_overall_confidence(facts):.0%}"
        
        return summary
    
    def _generate_key_findings(self, facts: List[ResolvedFact]) -> str:
        """Generate key findings section"""
        findings = "## Key Findings\n\n"
        
        for fact in facts[:15]:
            confidence_emoji = "✓" if fact.confidence > 0.8 else "~" if fact.confidence > 0.5 else "?"
            findings += f"- {confidence_emoji} {fact.statement}\n"
        
        return findings
    
    def _generate_detailed_analysis(self, facts: List[ResolvedFact]) -> str:
        """Generate detailed analysis"""
        analysis = "## Detailed Analysis\n\n"
        
        # Group by category
        by_category = defaultdict(list)
        for fact in facts:
            # Extract category from sources or use default
            category = "General"
            by_category[category].append(fact)
        
        for category, cat_facts in by_category.items():
            analysis += f"### {category}\n\n"
            for fact in cat_facts[:5]:
                analysis += f"- {fact.statement}\n"
                if fact.sources:
                    analysis += f"  *Sources: {', '.join(fact.sources[:3])}*\n"
            analysis += "\n"
        
        return analysis
    
    def _generate_consensus_section(
        self,
        consensus: List[ConsensusArea]
    ) -> str:
        """Generate consensus section"""
        section = "## Areas of Consensus\n\n"
        
        for area in consensus:
            section += f"### {area.topic}\n"
            section += f"**Confidence:** {area.confidence:.0%}\n\n"
            section += "**Agreed Facts:**\n"
            for fact in area.agreed_facts[:3]:
                section += f"- {fact}\n"
            section += f"\n**Supporting Sources:** {len(area.supporting_sources)}\n\n"
        
        return section
    
    def _generate_disagreement_section(
        self,
        disagreements: List[DisagreementArea]
    ) -> str:
        """Generate disagreement section"""
        section = "## Areas of Disagreement\n\n"
        
        for area in disagreements:
            section += f"### {area.topic}\n"
            section += f"**Analysis:** {area.analysis}\n\n"
            section += "**Positions:**\n"
            for i, position in enumerate(area.conflicting_positions, 1):
                section += f"{i}. {position}\n"
            section += "\n"
        
        return section
    
    def _generate_implications(self, facts: List[ResolvedFact]) -> str:
        """Generate implications section"""
        return "## Implications\n\nBased on the synthesized information, further analysis may be needed to fully understand the implications of these findings."
    
    def _generate_knowledge_gaps(self, facts: List[ResolvedFact]) -> str:
        """Generate knowledge gaps section"""
        return "## Knowledge Gaps\n\nSome areas require additional research for complete understanding."
    
    def _extract_implications(self, facts: List[ResolvedFact]) -> List[str]:
        """Extract implications from facts"""
        # Simplified - would use LLM in production
        return ["Further research recommended"]
    
    def _identify_gaps(self, facts: List[ResolvedFact]) -> List[str]:
        """Identify knowledge gaps"""
        # Simplified - would analyze coverage
        return ["Additional sources may provide more complete picture"]
    
    def _generate_recommendations(
        self,
        facts: List[ResolvedFact],
        task: ResearchTask
    ) -> List[str]:
        """Generate recommendations for further research"""
        recommendations = []
        
        if len(facts) < 5:
            recommendations.append("Consider additional sources for more comprehensive coverage")
        
        recommendations.append("Verify key facts with primary sources when possible")
        recommendations.append("Monitor for updates on this topic")
        
        return recommendations
    
    def _group_similar_facts(
        self,
        facts: List[ResolvedFact]
    ) -> List[List[ResolvedFact]]:
        """Group similar facts together"""
        groups = []
        used = set()
        
        for i, fact1 in enumerate(facts):
            if i in used:
                continue
            
            group = [fact1]
            used.add(i)
            
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if j in used:
                    continue
                
                if self._facts_similar(fact1, fact2):
                    group.append(fact2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _facts_similar(self, fact1: ResolvedFact, fact2: ResolvedFact) -> bool:
        """Check if two facts are similar"""
        # Simple similarity check
        words1 = set(fact1.statement.lower().split())
        words2 = set(fact2.statement.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity > 0.6
    
    def _calculate_overall_confidence(self, facts: List[ResolvedFact]) -> float:
        """Calculate overall confidence score"""
        if not facts:
            return 0.0
        
        total_confidence = sum(f.confidence for f in facts)
        return total_confidence / len(facts)
    
    def _calculate_confidence(
        self,
        resolved_facts: List[ResolvedFact],
        extracted_info: List[ExtractedInformation]
    ) -> float:
        """Calculate overall synthesis confidence"""
        if not extracted_info:
            return 0.0
        
        # Factor 1: Source diversity
        source_count = len(extracted_info)
        source_factor = min(1.0, source_count / 5)
        
        # Factor 2: Fact confidence
        if resolved_facts:
            fact_confidence = sum(f.confidence for f in resolved_facts) / len(resolved_facts)
        else:
            fact_confidence = 0.5
        
        # Factor 3: Information density
        total_facts = sum(len(info.facts) for info in extracted_info)
        density_factor = min(1.0, total_facts / 20)
        
        # Combined score
        confidence = (source_factor * 0.3) + (fact_confidence * 0.5) + (density_factor * 0.2)
        
        return round(confidence, 2)


class ConflictResolver:
    """
    Conflict resolution between information from different sources.
    """
    
    async def resolve(
        self,
        topic_facts: Dict[str, List[Fact]]
    ) -> List[ResolvedFact]:
        """
        Resolve conflicting information.
        
        Args:
            topic_facts: Facts grouped by topic
            
        Returns:
            List of resolved facts
        """
        resolved = []
        
        for topic, facts in topic_facts.items():
            if len(facts) == 1:
                # No conflict
                resolved.append(ResolvedFact(
                    statement=facts[0].statement,
                    sources=["source_1"],
                    confidence=0.6,
                    resolution_method="single_source",
                    alternative_statements=[]
                ))
            else:
                # Check for conflicts
                conflicts = self._identify_conflicts(facts)
                
                if conflicts:
                    resolved_fact = await self._resolve_conflict_group(
                        topic,
                        facts,
                        conflicts
                    )
                    resolved.append(resolved_fact)
                else:
                    # No conflicts, consolidate
                    resolved.append(self._consolidate_facts(facts))
        
        return resolved
    
    def _identify_conflicts(self, facts: List[Fact]) -> List[Dict[str, Any]]:
        """Identify conflicts between facts"""
        conflicts = []
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                if self._facts_conflict(fact1, fact2):
                    conflicts.append({
                        "fact1": fact1,
                        "fact2": fact2,
                        "type": self._classify_conflict(fact1, fact2)
                    })
        
        return conflicts
    
    def _facts_conflict(self, fact1: Fact, fact2: Fact) -> bool:
        """Check if two facts conflict"""
        # Simple conflict detection
        # Would use more sophisticated NLP in production
        
        # Check for contradictory keywords
        contradiction_markers = [
            ("is", "is not"),
            ("was", "was not"),
            ("will", "will not"),
            ("increases", "decreases"),
            ("positive", "negative")
        ]
        
        text1 = fact1.statement.lower()
        text2 = fact2.statement.lower()
        
        for pos, neg in contradiction_markers:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return True
        
        return False
    
    def _classify_conflict(self, fact1: Fact, fact2: Fact) -> str:
        """Classify the type of conflict"""
        # Check for temporal conflict
        if fact1.temporal_context != fact2.temporal_context:
            return "temporal"
        
        # Check for numerical conflict
        if re.search(r'\d', fact1.statement) and re.search(r'\d', fact2.statement):
            return "numerical"
        
        return "semantic"
    
    async def _resolve_conflict_group(
        self,
        topic: str,
        facts: List[Fact],
        conflicts: List[Dict[str, Any]]
    ) -> ResolvedFact:
        """Resolve a group of conflicting facts"""
        
        # Analyze conflict type
        conflict_types = set(c["type"] for c in conflicts)
        
        if "temporal" in conflict_types:
            # Time-based conflict - use most recent
            return self._resolve_temporal_conflict(facts)
        
        if "numerical" in conflict_types:
            # Numerical conflict - use median or most cited
            return self._resolve_numerical_conflict(facts)
        
        # Default: use most confident
        return self._resolve_by_confidence(facts)
    
    def _resolve_temporal_conflict(self, facts: List[Fact]) -> ResolvedFact:
        """Resolve conflict using most recent information"""
        # Sort by temporal context (simplified)
        sorted_facts = sorted(
            facts,
            key=lambda f: f.temporal_context or "",
            reverse=True
        )
        
        most_recent = sorted_facts[0]
        alternatives = [f.statement for f in sorted_facts[1:]]
        
        return ResolvedFact(
            statement=most_recent.statement,
            sources=["temporal_resolution"],
            confidence=0.7,
            resolution_method="temporal_priority",
            alternative_statements=alternatives
        )
    
    def _resolve_numerical_conflict(self, facts: List[Fact]) -> ResolvedFact:
        """Resolve numerical conflict"""
        # Use the fact with highest confidence
        best_fact = max(facts, key=lambda f: self._confidence_value(f.confidence))
        
        alternatives = [
            f.statement for f in facts
            if f.statement != best_fact.statement
        ]
        
        return ResolvedFact(
            statement=best_fact.statement,
            sources=["numerical_resolution"],
            confidence=self._confidence_value(best_fact.confidence),
            resolution_method="confidence_priority",
            alternative_statements=alternatives
        )
    
    def _resolve_by_confidence(self, facts: List[Fact]) -> ResolvedFact:
        """Resolve by confidence level"""
        best_fact = max(facts, key=lambda f: self._confidence_value(f.confidence))
        
        alternatives = [
            f.statement for f in facts
            if f.statement != best_fact.statement
        ]
        
        return ResolvedFact(
            statement=best_fact.statement,
            sources=["confidence_resolution"],
            confidence=self._confidence_value(best_fact.confidence),
            resolution_method="confidence_priority",
            alternative_statements=alternatives
        )
    
    def _consolidate_facts(self, facts: List[Fact]) -> ResolvedFact:
        """Consolidate non-conflicting facts"""
        # Use the most detailed fact
        best_fact = max(facts, key=lambda f: len(f.statement))
        
        return ResolvedFact(
            statement=best_fact.statement,
            sources=["consolidation"],
            confidence=0.75,
            resolution_method="consolidation",
            alternative_statements=[]
        )
    
    def _confidence_value(self, confidence: str) -> float:
        """Convert confidence string to numeric value"""
        values = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3
        }
        return values.get(confidence.lower(), 0.5)


class SummarizationEngine:
    """
    Multi-level summarization engine.
    
    Creates summaries at multiple detail levels:
    - ultra_brief: One sentence
    - brief: 2-3 sentences
    - standard: One paragraph
    - detailed: Multiple paragraphs
    - comprehensive: Full summary with key points
    """
    
    SUMMARY_LEVELS = {
        "ultra_brief": {
            "max_length": 150,
            "description": "One-sentence summary"
        },
        "brief": {
            "max_length": 300,
            "description": "2-3 sentence summary"
        },
        "standard": {
            "max_length": 600,
            "description": "Paragraph summary"
        },
        "detailed": {
            "max_length": 1500,
            "description": "Multi-paragraph summary"
        },
        "comprehensive": {
            "max_length": 3000,
            "description": "Full summary with key points"
        }
    }
    
    async def create_summaries(self, synthesis: Synthesis) -> Dict[str, Summary]:
        """
        Create summaries at all levels.
        
        Args:
            synthesis: Synthesis to summarize
            
        Returns:
            Dictionary of summaries by level
        """
        summaries = {}
        
        for level, config in self.SUMMARY_LEVELS.items():
            summary = await self._create_summary_at_level(
                synthesis,
                level,
                config
            )
            summaries[level] = summary
        
        return summaries
    
    async def _create_summary_at_level(
        self,
        synthesis: Synthesis,
        level: str,
        config: Dict[str, Any]
    ) -> Summary:
        """Create a summary at a specific detail level"""
        
        # Extract content based on level
        if level == "ultra_brief":
            content = self._create_ultra_brief(synthesis)
        elif level == "brief":
            content = self._create_brief(synthesis)
        elif level == "standard":
            content = self._create_standard(synthesis)
        elif level == "detailed":
            content = self._create_detailed(synthesis)
        else:  # comprehensive
            content = self._create_comprehensive(synthesis)
        
        # Truncate if needed
        if len(content) > config["max_length"]:
            content = content[:config["max_length"]].rsplit('.', 1)[0] + '.'
        
        return Summary(
            level=level,
            content=content,
            token_count=len(content.split()),
            generated_at=datetime.now()
        )
    
    def _create_ultra_brief(self, synthesis: Synthesis) -> str:
        """Create ultra-brief one-sentence summary"""
        if synthesis.key_points:
            return f"Research on {synthesis.topic} indicates: {synthesis.key_points[0]}"
        return f"Research synthesis completed on {synthesis.topic}."
    
    def _create_brief(self, synthesis: Synthesis) -> str:
        """Create brief 2-3 sentence summary"""
        sentences = []
        
        sentences.append(f"This synthesis covers {synthesis.topic}.")
        
        if synthesis.key_points:
            sentences.append(f"Key finding: {synthesis.key_points[0]}")
        
        if synthesis.implications:
            sentences.append(synthesis.implications[0])
        
        return " ".join(sentences)
    
    def _create_standard(self, synthesis: Synthesis) -> str:
        """Create standard paragraph summary"""
        parts = []
        
        parts.append(f"This research synthesis examines {synthesis.topic}.")
        
        if synthesis.key_points:
            parts.append(f"Key findings include: {'; '.join(synthesis.key_points[:3])}.")
        
        if synthesis.implications:
            parts.append(f"Implications: {synthesis.implications[0]}")
        
        return " ".join(parts)
    
    def _create_detailed(self, synthesis: Synthesis) -> str:
        """Create detailed multi-paragraph summary"""
        paragraphs = []
        
        # Introduction
        paragraphs.append(
            f"This comprehensive synthesis examines {synthesis.topic} "
            f"based on analysis of multiple sources. The research reveals "
            f"several key insights about this topic."
        )
        
        # Key findings
        if synthesis.key_points:
            findings_text = "Key findings include:\n\n"
            for i, point in enumerate(synthesis.key_points[:5], 1):
                findings_text += f"{i}. {point}\n"
            paragraphs.append(findings_text)
        
        # Implications
        if synthesis.implications:
            paragraphs.append(
                f"Implications: {' '.join(synthesis.implications[:2])}"
            )
        
        return "\n\n".join(paragraphs)
    
    def _create_comprehensive(self, synthesis: Synthesis) -> str:
        """Create comprehensive summary"""
        sections = []
        
        # Overview
        sections.append(f"# {synthesis.topic} - Comprehensive Summary\n")
        
        sections.append(
            f"This comprehensive research synthesis provides an in-depth "
            f"analysis of {synthesis.topic}. The synthesis draws from multiple "
            f"sources to provide a well-rounded understanding of the subject.\n"
        )
        
        # Key Findings
        if synthesis.key_points:
            sections.append("## Key Findings\n")
            for i, point in enumerate(synthesis.key_points, 1):
                sections.append(f"{i}. {point}")
            sections.append("")
        
        # Main Content
        if synthesis.content:
            sections.append("## Detailed Analysis\n")
            # Include first part of synthesis content
            content_preview = synthesis.content[:1500]
            sections.append(content_preview + "...")
            sections.append("")
        
        # Implications
        if synthesis.implications:
            sections.append("## Implications\n")
            for impl in synthesis.implications:
                sections.append(f"- {impl}")
            sections.append("")
        
        # Knowledge Gaps
        if synthesis.knowledge_gaps:
            sections.append("## Knowledge Gaps\n")
            for gap in synthesis.knowledge_gaps:
                sections.append(f"- {gap}")
            sections.append("")
        
        # Recommendations
        if synthesis.recommendations:
            sections.append("## Recommendations\n")
            for rec in synthesis.recommendations:
                sections.append(f"- {rec}")
        
        return "\n".join(sections)
