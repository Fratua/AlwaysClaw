"""
Information Extraction System
Structured information extraction from crawled content.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    CrawledContent, ExtractedInformation, Entity, Fact, Relationship,
    Claim, Statistic, Quote, ExtractionGoal
)


logger = logging.getLogger(__name__)


class InformationExtractor:
    """
    Main information extraction orchestrator.
    
    Coordinates multiple extraction subsystems:
    - Entity extraction
    - Fact extraction
    - Relationship extraction
    - Claim extraction
    - Statistics extraction
    - Quote extraction
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.fact_extractor = FactExtractor()
        self.relation_extractor = RelationExtractor()
    
    async def extract_information(
        self,
        content: CrawledContent,
        goals: List[ExtractionGoal]
    ) -> ExtractedInformation:
        """
        Extract all structured information from crawled content.
        
        Args:
            content: Crawled content to extract from
            goals: Extraction goals to guide the process
            
        Returns:
            Extracted information with all components
        """
        logger.debug(f"Extracting information from: {content.source.url}")
        
        extracted = ExtractedInformation(
            source=content.source,
            extraction_timestamp=datetime.now()
        )
        
        # Extract entities
        try:
            extracted.entities = await self.entity_extractor.extract(
                content.content.content
            )
            logger.debug(f"Extracted {len(extracted.entities)} entities")
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        # Extract facts
        try:
            extracted.facts = await self.fact_extractor.extract(
                content.content.content,
                goals
            )
            logger.debug(f"Extracted {len(extracted.facts)} facts")
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
        
        # Extract relationships
        try:
            extracted.relationships = await self.relation_extractor.extract(
                content.content.content,
                extracted.entities
            )
            logger.debug(f"Extracted {len(extracted.relationships)} relationships")
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
        
        # Extract claims
        try:
            extracted.claims = await self._extract_claims(content)
            logger.debug(f"Extracted {len(extracted.claims)} claims")
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
        
        # Extract statistics
        try:
            extracted.statistics = await self._extract_statistics(content)
            logger.debug(f"Extracted {len(extracted.statistics)} statistics")
        except Exception as e:
            logger.warning(f"Statistics extraction failed: {e}")
        
        # Extract quotes
        try:
            extracted.quotes = await self._extract_quotes(content)
            logger.debug(f"Extracted {len(extracted.quotes)} quotes")
        except Exception as e:
            logger.warning(f"Quote extraction failed: {e}")
        
        return extracted
    
    async def _extract_claims(self, content: CrawledContent) -> List[Claim]:
        """Extract verifiable claims from content"""
        claims = []
        text = content.content.content
        
        # Simple claim detection patterns
        claim_patterns = [
            r'([A-Z][^.]*?(?:is|are|was|were|will be|has been|have been)[^.]*\.)',
            r'([A-Z][^.]*?(?:shows?|demonstrates?|proves?|indicates?)[^.]*\.)',
            r'([A-Z][^.]*?(?:according to|research|study|report)[^.]*\.)'
        ]
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:10]:  # Limit claims
                claims.append(Claim(
                    text=match.strip(),
                    claim_type="fact",
                    confidence="medium"
                ))
        
        return claims
    
    async def _extract_statistics(self, content: CrawledContent) -> List[Statistic]:
        """Extract statistics from content"""
        statistics = []
        text = content.content.content
        
        # Pattern for numbers with units
        stat_patterns = [
            r'(\d+(?:\.\d+)?)\s*(percent|%|million|billion|thousand)',
            r'(\d+(?:\.\d+)?)\s*(USD|\$|€|£)',
            r'(\d+(?:\.\d+)?)\s*(people|users|customers|patients)'
        ]
        
        for pattern in stat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:10]:
                value, metric = match
                statistics.append(Statistic(
                    value=value,
                    metric=metric,
                    context=self._get_context(text, match[0])
                ))
        
        return statistics
    
    async def _extract_quotes(self, content: CrawledContent) -> List[Quote]:
        """Extract quotes from content"""
        quotes = []
        text = content.content.content
        
        # Pattern for quoted text
        quote_patterns = [
            r'"([^"]{10,200})"',  # Double quotes
            r"'([^']{10,200})'",   # Single quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:10]:
                # Try to find attribution
                context = self._get_context(text, match)
                author = self._extract_attribution(context)
                
                quotes.append(Quote(
                    text=match.strip(),
                    author=author,
                    context=context
                ))
        
        return quotes
    
    def _get_context(self, text: str, target: str, window: int = 100) -> str:
        """Get context around a target text"""
        idx = text.find(target)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(target) + window)
        
        return text[start:end].strip()
    
    def _extract_attribution(self, context: str) -> Optional[str]:
        """Try to extract quote attribution from context"""
        # Simple patterns for attribution
        patterns = [
            r'said\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'according to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+said'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)
        
        return None


class EntityExtractor:
    """
    Named Entity Recognition and extraction.
    
    Extracts entities like people, organizations, locations,
    technologies, and concepts from text.
    """
    
    ENTITY_TYPES = [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
        "MONEY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "TECHNOLOGY", "CONCEPT"
    ]
    
    # Simple patterns for entity detection
    PATTERNS = {
        "ORGANIZATION": [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Inc|Corp|Ltd|LLC|Company|Organization|Institute|University))\b',
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Foundation|Association|Society|Group))\b'
        ],
        "TECHNOLOGY": [
            r'\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)+)\b',  # Software names
            r'\b([A-Z][a-zA-Z]*\s+(?:API|SDK|Framework|Library|Platform))\b',
            r'\b(?:Python|JavaScript|Java|Go|Rust|TypeScript|React|Angular|Vue|Docker|Kubernetes)\b'
        ],
        "PERSON": [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:said|wrote|reported|claimed))\b'
        ],
        "LOCATION": [
            r'\b(?:in|at|from)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b'
        ]
    }
    
    async def extract(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        seen = set()
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    # Handle tuple matches
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    # Normalize
                    normalized = match.strip()
                    key = (normalized.lower(), entity_type)
                    
                    if key not in seen and len(normalized) > 2:
                        seen.add(key)
                        
                        # Count mentions
                        mentions = text.lower().count(normalized.lower())
                        
                        # Get context
                        context = self._get_entity_context(text, normalized)
                        
                        entities.append(Entity(
                            name=normalized,
                            type=entity_type,
                            context=context,
                            confidence=min(0.9, 0.5 + mentions * 0.1),
                            mentions=mentions,
                            normalized_name=normalized.lower()
                        ))
        
        # Sort by mention count
        entities.sort(key=lambda e: e.mentions, reverse=True)
        
        return entities[:50]  # Limit entities
    
    def _get_entity_context(self, text: str, entity: str, window: int = 50) -> str:
        """Get context around an entity mention"""
        idx = text.find(entity)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        
        return text[start:end].strip()


class FactExtractor:
    """
    Factual information extraction.
    
    Extracts facts based on extraction goals and content analysis.
    """
    
    async def extract(
        self,
        text: str,
        goals: List[ExtractionGoal]
    ) -> List[Fact]:
        """
        Extract facts based on extraction goals.
        
        Args:
            text: Text to extract from
            goals: Extraction goals
            
        Returns:
            List of extracted facts
        """
        facts = []
        
        for goal in goals:
            goal_facts = await self._extract_for_goal(text, goal)
            facts.extend(goal_facts)
        
        # Remove duplicates
        seen = set()
        unique_facts = []
        for fact in facts:
            key = fact.statement.lower().strip()
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        
        return unique_facts
    
    async def _extract_for_goal(
        self,
        text: str,
        goal: ExtractionGoal
    ) -> List[Fact]:
        """Extract facts for a specific goal"""
        facts = []
        
        # Define patterns based on goal
        patterns = self._get_patterns_for_goal(goal)
        
        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches[:5]:  # Limit per pattern
                statement = match.strip() if isinstance(match, str) else match[0].strip()
                
                if len(statement) > 20:  # Minimum length
                    facts.append(Fact(
                        statement=statement,
                        category=category,
                        confidence="medium",
                        temporal_context=self._extract_temporal_context(text, statement)
                    ))
        
        return facts
    
    def _get_patterns_for_goal(self, goal: ExtractionGoal) -> List[tuple]:
        """Get extraction patterns for a goal"""
        
        # Default patterns
        default_patterns = [
            (r'([^.]*?\bis\s+a\s+[^.]+\.)', "definition"),
            (r'([^.]*?\bconsists?\s+of\s+[^.]+\.)', "composition"),
            (r'([^.]*?\bused\s+for\s+[^.]+\.)', "purpose"),
        ]
        
        # Goal-specific patterns
        goal_patterns = {
            "definition": [
                (r'([^.]*?\bis\s+defined\s+as\s+[^.]+\.)', "definition"),
                (r'([^.]*?\brefers?\s+to\s+[^.]+\.)', "definition"),
                (r'([^.]*?\bmeans?\s+[^.]+\.)', "definition"),
            ],
            "statistics": [
                (r'(\d+(?:\.\d+)?%\s+of\s+[^.]+)', "statistic"),
                (r'(\d+(?:\.\d+)?\s+(?:million|billion|thousand)\s+[^.]+)', "statistic"),
            ],
            "process": [
                (r'([^.]*?\bfirst\s+[^.]*?then\s+[^.]+)', "process"),
                (r'([^.]*?\bsteps?\s+(?:are|include)\s+[^.]+)', "process"),
            ]
        }
        
        return goal_patterns.get(goal.topic, default_patterns)
    
    def _extract_temporal_context(self, text: str, statement: str) -> Optional[str]:
        """Extract temporal context for a statement"""
        # Look for dates near the statement
        idx = text.find(statement)
        if idx == -1:
            return None
        
        context = text[max(0, idx - 100):min(len(text), idx + len(statement) + 100)]
        
        # Date patterns
        date_patterns = [
            r'\b(\d{4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
            r'\b(in|since|as of)\s+(\d{4}|\w+\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(0)
        
        return None


class RelationExtractor:
    """
    Relationship extraction between entities.
    
    Identifies relationships like "is-a", "part-of", "uses", etc.
    """
    
    RELATION_PATTERNS = {
        "is_a": [
            r'([A-Z][a-zA-Z]+)\s+is\s+a\s+([a-z\s]+)',
            r'([A-Z][a-zA-Z]+)\s+is\s+an\s+([a-z\s]+)'
        ],
        "part_of": [
            r'([A-Z][a-zA-Z]+)\s+is\s+part\s+of\s+([A-Z][a-zA-Z]+)',
            r'([A-Z][a-zA-Z]+)\s+consists?\s+of\s+([a-z\s,]+)'
        ],
        "uses": [
            r'([A-Z][a-zA-Z]+)\s+uses?\s+([A-Z][a-zA-Z]+)',
            r'([A-Z][a-zA-Z]+)\s+is\s+used\s+by\s+([A-Z][a-zA-Z]+)'
        ],
        "created_by": [
            r'([A-Z][a-zA-Z]+)\s+was\s+created\s+by\s+([A-Z][a-zA-Z]+)',
            r'([A-Z][a-zA-Z]+)\s+developed\s+([A-Z][a-zA-Z]+)'
        ]
    }
    
    async def extract(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            text: Text to extract from
            entities: Already extracted entities
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Create entity lookup
        entity_names = {e.name.lower(): e for e in entities}
        
        for relation_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    subject, obj = match
                    
                    # Check if entities are known
                    subj_lower = subject.lower()
                    obj_lower = obj.lower()
                    
                    if subj_lower in entity_names or obj_lower in entity_names:
                        relationships.append(Relationship(
                            subject=subject.strip(),
                            predicate=relation_type,
                            object=obj.strip(),
                            confidence=0.7,
                            context=self._get_context(text, subject)
                        ))
        
        return relationships
    
    def _get_context(self, text: str, target: str, window: int = 50) -> str:
        """Get context around a target"""
        idx = text.find(target)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(target) + window)
        
        return text[start:end].strip()
