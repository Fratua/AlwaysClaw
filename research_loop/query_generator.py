"""
Search Query Generation System
Intelligent query generation and optimization for multi-engine search.
"""

import re
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

        Attempts LLM-based concept extraction via OpenAIClient. Falls back
        to keyword-based extraction when the LLM is unavailable.
        """
        concepts = {
            "primary_subject": task.topic,
            "secondary_subjects": [],
            "key_entities": [],
            "temporal_context": None,
            "geographic_context": None,
            "domain": "general",
            "technical_level": "medium"
        }

        # Attempt LLM-based extraction
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()

            prompt = (
                "Analyze the following research topic and extract structured concepts.\n"
                f"Topic: {task.topic}\n"
            )
            if task.context:
                prompt += f"Context: {task.context}\n"
            prompt += (
                "\nReturn the following fields, one per line, in key: value format:\n"
                "primary_subject: <main topic>\n"
                "secondary_subjects: <comma-separated list>\n"
                "key_entities: <comma-separated list>\n"
                "temporal_context: <current/historical/none>\n"
                "geographic_context: <region or none>\n"
                "domain: <general/technical/scientific/business/other>\n"
                "technical_level: <low/medium/high>\n"
            )
            result = client.generate(prompt, system="You are a research analyst.", max_tokens=300)

            for line in result.strip().splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key == "primary_subject" and value:
                    concepts["primary_subject"] = value
                elif key == "secondary_subjects" and value.lower() != "none":
                    concepts["secondary_subjects"] = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "key_entities" and value.lower() != "none":
                    concepts["key_entities"] = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "temporal_context" and value.lower() != "none":
                    concepts["temporal_context"] = value
                elif key == "geographic_context" and value.lower() != "none":
                    concepts["geographic_context"] = value
                elif key == "domain" and value:
                    concepts["domain"] = value
                elif key == "technical_level" and value:
                    concepts["technical_level"] = value

            return concepts

        except (ImportError, EnvironmentError, RuntimeError, ValueError) as exc:
            logger.debug(f"LLM unavailable for concept extraction, using keyword fallback: {exc}")

        # Keyword-based fallback
        combined_text = (task.topic + " " + (task.context or "")).lower()
        words = combined_text.split()

        # Temporal context detection
        temporal_keywords = ["latest", "recent", "new", "2024", "2025", "2026", "current", "today", "now"]
        historical_keywords = ["history", "historical", "past", "origin", "evolution"]
        for kw in temporal_keywords:
            if kw in words:
                concepts["temporal_context"] = "current"
                break
        if not concepts["temporal_context"]:
            for kw in historical_keywords:
                if kw in words:
                    concepts["temporal_context"] = "historical"
                    break

        # Geographic context detection
        geo_keywords = {
            "us": "United States", "usa": "United States", "europe": "Europe",
            "asia": "Asia", "china": "China", "india": "India", "uk": "United Kingdom",
            "global": "Global", "worldwide": "Global",
        }
        for kw, region in geo_keywords.items():
            if kw in words:
                concepts["geographic_context"] = region
                break

        # Domain detection
        technical_keywords = ["technical", "implementation", "code", "api", "documentation",
                              "programming", "software", "algorithm", "framework", "library"]
        scientific_keywords = ["research", "study", "experiment", "hypothesis", "data",
                               "analysis", "scientific", "journal", "paper"]
        business_keywords = ["market", "business", "revenue", "startup", "company",
                             "investment", "industry", "strategy"]

        if any(kw in words for kw in technical_keywords):
            concepts["technical_level"] = "high"
            concepts["domain"] = "technical"
        elif any(kw in words for kw in scientific_keywords):
            concepts["domain"] = "scientific"
        elif any(kw in words for kw in business_keywords):
            concepts["domain"] = "business"

        # Extract multi-word entities (capitalized phrases in original text)
        original_text = task.topic + " " + (task.context or "")
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', original_text)
        concepts["key_entities"] = list(set(entities))[:10]

        # Extract secondary subjects from context keywords > 5 chars
        if task.context:
            stop_words = {"about", "their", "which", "would", "could", "should", "these", "those",
                          "there", "where", "other", "after", "before", "between"}
            secondary = [
                w for w in task.context.split()
                if len(w) > 5 and w.lower() not in stop_words and w.lower() != task.topic.lower()
            ]
            concepts["secondary_subjects"] = list(set(secondary))[:5]

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
        """Add search operators for more precise results."""
        words = text.split()
        if len(words) < 2:
            return text

        # Identify multi-word noun phrases (2-3 consecutive capitalized or
        # common compound terms) and wrap them in quotes
        # Heuristic: quote the first 2-3 word chunk that looks like a key phrase
        compound_patterns = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+){1,2}\b', text)
        quoted_text = text
        quoted_any = False
        for phrase in compound_patterns[:2]:
            # Only quote phrases longer than one word and not already quoted
            phrase_words = phrase.split()
            if len(phrase_words) >= 2 and f'"{phrase}"' not in quoted_text:
                quoted_text = quoted_text.replace(phrase, f'"{phrase}"', 1)
                quoted_any = True
                break  # Only quote the primary phrase to avoid over-constraining

        if not quoted_any and len(words) >= 3:
            # Quote the first two meaningful words as a phrase
            meaningful = [w for w in words if len(w) > 3]
            if len(meaningful) >= 2:
                phrase = f"{meaningful[0]} {meaningful[1]}"
                if phrase in quoted_text:
                    quoted_text = quoted_text.replace(phrase, f'"{phrase}"', 1)

        return quoted_text
    
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
