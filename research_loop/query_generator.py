"""
Search Query Generation System
Intelligent query generation and optimization for multi-engine search.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import ResearchTask, SearchQuery, ResearchContext
from .config import QUERY_TEMPLATES, ENGINE_CAPABILITIES


logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    Intelligent search query generation system.
    
    Generates optimized queries using:
    - GPT-5.2 for concept extraction and query formulation
    - Query templates for different search types
    - Multi-variant generation for comprehensive coverage
    """
    
    def __init__(self):
        self.query_templates = QUERY_TEMPLATES
        self.query_history: List[SearchQuery] = []
    
    async def generate_queries(
        self,
        research_task: ResearchTask,
        context: ResearchContext
    ) -> List[SearchQuery]:
        """
        Generate optimized search queries for a research task.
        
        Args:
            research_task: The research task to generate queries for
            context: Current research context
            
        Returns:
            List of search queries
        """
        logger.info(f"Generating queries for: {research_task.topic}")
        
        # Step 1: Extract key concepts
        concepts = await self._extract_concepts(research_task)
        
        # Step 2: Generate query variants
        variants = await self._generate_variants(concepts, research_task)
        
        # Step 3: Optimize for search engines
        optimized = await self._optimize_queries(variants, research_task)
        
        # Step 4: Deduplicate and rank
        final_queries = self._rank_and_deduplicate(optimized)
        
        # Store in history
        self.query_history.extend(final_queries)
        
        logger.info(f"Generated {len(final_queries)} queries")
        
        return final_queries
    
    async def _extract_concepts(self, task: ResearchTask) -> Dict[str, Any]:
        """
        Extract key concepts from research task.
        
        Uses GPT-5.2 to analyze the task and extract:
        - Primary subject
        - Secondary subjects
        - Key entities
        - Temporal context
        - Geographic context
        - Domain specificity
        """
        # This would use GPT-5.2 in production
        # For now, use rule-based extraction
        
        concepts = {
            "primary_subject": task.topic,
            "secondary_subjects": [],
            "key_entities": [],
            "temporal_context": None,
            "geographic_context": None,
            "domain": "general",
            "technical_level": "medium"
        }
        
        # Extract from context if available
        if task.context:
            # Simple keyword extraction
            words = task.context.lower().split()
            
            # Detect temporal context
            temporal_keywords = ["latest", "recent", "new", "2024", "2025", "current"]
            for kw in temporal_keywords:
                if kw in words:
                    concepts["temporal_context"] = "current"
                    break
            
            # Detect technical level
            technical_keywords = ["technical", "implementation", "code", "api", "documentation"]
            if any(kw in words for kw in technical_keywords):
                concepts["technical_level"] = "high"
                concepts["domain"] = "technical"
        
        return concepts
    
    async def _generate_variants(
        self,
        concepts: Dict[str, Any],
        task: ResearchTask
    ) -> List[SearchQuery]:
        """Generate query variants based on concepts"""
        variants = []
        
        topic = concepts["primary_subject"]
        
        # Generate variants from templates
        for template_name, template_config in self.query_templates.items():
            # Check if template is appropriate for this research
            if self._template_matches_task(template_config, task):
                query_text = self._apply_template(
                    template_config["template"],
                    topic,
                    concepts
                )
                
                query = SearchQuery(
                    text=query_text,
                    target_engines=template_config["engines"],
                    query_type=template_name,
                    priority=self._calculate_template_priority(template_name, task),
                    parent_task_id=task.id
                )
                
                variants.append(query)
        
        # Generate additional variants based on depth
        depth_variants = self._generate_depth_variants(concepts, task)
        variants.extend(depth_variants)
        
        return variants
    
    def _apply_template(
        self,
        template: str,
        topic: str,
        concepts: Dict[str, Any]
    ) -> str:
        """Apply a query template with the given concepts"""
        query = template.replace("{topic}", topic)
        
        # Replace optional placeholders
        if "{action}" in query:
            query = query.replace("{action}", "use")  # Default action
        
        if "{topic_a}" in query:
            query = query.replace("{topic_a}", topic)
        
        if "{topic_b}" in query:
            # Try to find a comparison topic
            query = query.replace("{topic_b}", "alternatives")
        
        if "{timeframe}" in query:
            timeframe = "2024" if concepts.get("temporal_context") == "current" else ""
            query = query.replace("{timeframe}", timeframe).strip()
        
        return query
    
    def _template_matches_task(
        self,
        template_config: Dict[str, Any],
        task: ResearchTask
    ) -> bool:
        """Check if a template is appropriate for the task"""
        use_case = template_config["use_case"]
        
        # Map task characteristics to use cases
        if "tutorial" in task.topic.lower() or "how to" in task.topic.lower():
            return use_case in ["procedural_knowledge", "basic_understanding"]
        
        if task.depth.value in ["deep", "exhaustive"]:
            return use_case in ["deep_research", "technical_details"]
        
        if task.depth.value in ["surface", "shallow"]:
            return use_case in ["basic_understanding", "current_information"]
        
        return True
    
    def _calculate_template_priority(
        self,
        template_name: str,
        task: ResearchTask
    ) -> float:
        """Calculate priority for a template based on task"""
        base_priority = 0.5
        
        # Adjust based on template type and task
        priorities = {
            "definition": 0.8 if task.depth.value == "surface" else 0.5,
            "how_to": 0.9 if "how" in task.topic.lower() else 0.4,
            "latest": 0.8 if task.trigger_type.value == "scheduled" else 0.6,
            "deep_dive": 0.9 if task.depth.value in ["deep", "exhaustive"] else 0.3,
            "technical": 0.8 if "technical" in task.topic.lower() else 0.5
        }
        
        return priorities.get(template_name, base_priority)
    
    def _generate_depth_variants(
        self,
        concepts: Dict[str, Any],
        task: ResearchTask
    ) -> List[SearchQuery]:
        """Generate additional variants based on research depth"""
        variants = []
        topic = concepts["primary_subject"]
        
        depth_modifiers = {
            "surface": ["overview", "basics"],
            "shallow": ["introduction", "guide"],
            "medium": ["detailed", "in-depth"],
            "deep": ["comprehensive", "advanced"],
            "exhaustive": ["complete", "exhaustive", "research"]
        }
        
        modifiers = depth_modifiers.get(task.depth.value, [""])
        
        for modifier in modifiers:
            if modifier:
                query_text = f"{topic} {modifier}"
                query = SearchQuery(
                    text=query_text,
                    target_engines=["google", "bing"],
                    query_type="depth_variant",
                    priority=0.6,
                    parent_task_id=task.id
                )
                variants.append(query)
        
        return variants
    
    async def _optimize_queries(
        self,
        queries: List[SearchQuery],
        task: ResearchTask
    ) -> List[SearchQuery]:
        """Optimize queries for search engines"""
        optimized = []
        
        for query in queries:
            # Remove unnecessary words
            optimized_text = self._optimize_text(query.text)
            
            # Add search operators if needed
            if task.depth.value in ["deep", "exhaustive"]:
                optimized_text = self._add_search_operators(optimized_text)
            
            query.text = optimized_text
            optimized.append(query)
        
        return optimized
    
    def _optimize_text(self, text: str) -> str:
        """Optimize query text for search engines"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common stop words at the beginning
        stop_words = ["the", "a", "an"]
        words = text.lower().split()
        if words and words[0] in stop_words:
            words = words[1:]
        
        return " ".join(words)
    
    def _add_search_operators(self, text: str) -> str:
        """Add search operators for more precise results"""
        # Add quotes around key phrases
        # This is a simplified version - production would be more sophisticated
        return text
    
    def _rank_and_deduplicate(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        """Rank queries and remove duplicates"""
        # Remove duplicates based on text
        seen_texts = set()
        unique = []
        
        for query in sorted(queries, key=lambda q: q.priority, reverse=True):
            normalized = query.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(query)
        
        return unique


class QueryDistributor:
    """
    Distribute queries across multiple search engines.
    
    Considers:
    - Engine capabilities and strengths
    - Rate limits
    - Query type appropriateness
    """
    
    def __init__(self):
        self.engine_capabilities = ENGINE_CAPABILITIES
        self.rate_limit_tracker: Dict[str, int] = {}
    
    def distribute_queries(
        self,
        queries: List[SearchQuery],
        research_depth: str
    ) -> Dict[str, List[SearchQuery]]:
        """
        Distribute queries to appropriate search engines.
        
        Args:
            queries: List of search queries
            research_depth: Research depth level
            
        Returns:
            Dictionary mapping engine names to queries
        """
        distribution: Dict[str, List[SearchQuery]] = {}
        
        for query in queries:
            # Determine best engines for this query
            engines = self._select_engines(query, research_depth)
            
            # Distribute with fallback
            assigned = False
            for engine in engines:
                if self._check_rate_limit(engine):
                    if engine not in distribution:
                        distribution[engine] = []
                    distribution[engine].append(query)
                    self._track_request(engine)
                    assigned = True
                    break
            
            if not assigned:
                logger.warning(f"Could not assign query to any engine: {query.text}")
        
        return distribution
    
    def _select_engines(
        self,
        query: SearchQuery,
        research_depth: str
    ) -> List[str]:
        """Select appropriate engines for a query"""
        engines = []
        
        # Start with query's target engines
        if query.target_engines:
            engines.extend(query.target_engines)
        
        # Add engines based on query type
        query_type_engines = {
            "definition": ["google", "bing", "duckduckgo"],
            "how_to": ["google", "youtube", "stackoverflow"],
            "deep_dive": ["google_scholar", "arxiv"],
            "technical": ["github", "stackoverflow", "docs"],
            "opinions": ["reddit", "twitter", "quora"]
        }
        
        if query.query_type in query_type_engines:
            for engine in query_type_engines[query.query_type]:
                if engine not in engines:
                    engines.append(engine)
        
        # Add general engines as fallback
        for engine in ["google", "bing", "duckduckgo"]:
            if engine not in engines:
                engines.append(engine)
        
        return engines
    
    def _check_rate_limit(self, engine: str) -> bool:
        """Check if engine is within rate limits"""
        config = self.engine_capabilities.get(engine, {})
        limit = config.get("rate_limit")
        
        if limit is None:
            return True  # No limit
        
        current = self.rate_limit_tracker.get(engine, 0)
        return current < limit
    
    def _track_request(self, engine: str):
        """Track a request to an engine"""
        self.rate_limit_tracker[engine] = self.rate_limit_tracker.get(engine, 0) + 1
    
    def reset_rate_limits(self):
        """Reset all rate limit counters"""
        self.rate_limit_tracker = {}
