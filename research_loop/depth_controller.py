"""
Research Depth and Breadth Control System
Manages research depth, adaptive research, and resource allocation.
"""

import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from .models import (
    ResearchTask, ResearchConfig, CrawlConfig, ProgressStatus,
    AdaptationDecision, SearchQuery, ExtractedInformation
)
from .config import ResearchLoopConfig, default_config


logger = logging.getLogger(__name__)


class DepthController:
    """
    Control research depth and breadth.
    
    Manages:
    - Depth level configuration
    - Resource allocation
    - Progress monitoring
    - Time budget management
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or default_config
        self.active_research: Dict[str, Dict[str, Any]] = {}
    
    async def configure_research(
        self,
        task: ResearchTask
    ) -> ResearchConfig:
        """
        Configure research parameters based on depth level.
        
        Args:
            task: Research task to configure
            
        Returns:
            Research configuration
        """
        # Get depth configuration
        depth_config = self.config.get_depth_config(task.depth.value)
        
        # Check available resources
        resources = await self._check_resources()
        
        # Adjust based on resources
        if resources.get("low_tokens", False):
            depth_config = self._reduce_depth_config(depth_config)
            logger.warning("Reduced research depth due to low token availability")
        
        if resources.get("rate_limited", False):
            depth_config.max_sources = max(3, depth_config.max_sources // 2)
            logger.warning("Reduced sources due to rate limiting")
        
        # Create research config
        config = ResearchConfig(
            depth=task.depth,
            max_sources=depth_config.max_sources,
            max_depth=depth_config.max_depth,
            max_pages_per_source=depth_config.max_pages_per_source,
            synthesis_detail=depth_config.synthesis_detail,
            time_budget=timedelta(seconds=depth_config.time_budget_seconds),
            query_variants=depth_config.query_variants,
            start_time=datetime.now()
        )
        
        # Track active research
        self.active_research[task.id] = {
            "config": config,
            "start_time": datetime.now(),
            "sources_collected": 0,
            "facts_extracted": 0,
            "average_source_quality": 0.0
        }
        
        logger.debug(f"Research configured: {config}")
        
        return config
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check actual system resources using psutil (with graceful fallback)."""
        result = {
            "low_tokens": False,
            "rate_limited": False,
            "memory_available": True
        }

        try:
            import psutil

            # Check memory: flag if more than 85% used
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                result["memory_available"] = False
                result["low_tokens"] = True  # Conserve resources when memory is tight
                logger.warning(f"High memory usage: {mem.percent:.1f}%")

            # Check CPU: flag rate limiting if sustained high CPU
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if cpu_percent > 90:
                result["rate_limited"] = True
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

            # Check disk: flag if disk is nearly full (>95%)
            try:
                disk = psutil.disk_usage(os.path.abspath(os.sep))
                if disk.percent > 95:
                    result["memory_available"] = False
                    logger.warning(f"Disk nearly full: {disk.percent:.1f}%")
            except OSError:
                pass

        except ImportError:
            logger.debug("psutil not installed; resource checks skipped")

        return result
    
    def _reduce_depth_config(self, config: Any) -> Any:
        """Reduce depth configuration for limited resources"""
        # Reduce all parameters by 50%
        config.max_sources = max(3, config.max_sources // 2)
        config.max_depth = max(1, config.max_depth // 2)
        config.max_pages_per_source = max(1, config.max_pages_per_source // 2)
        config.query_variants = max(2, config.query_variants // 2)
        return config
    
    async def monitor_progress(
        self,
        research_id: str
    ) -> ProgressStatus:
        """
        Monitor research progress and determine if should continue.
        
        Args:
            research_id: Research task ID
            
        Returns:
            Progress status with continue/stop decision
        """
        progress = self.active_research.get(research_id)
        
        if not progress:
            return ProgressStatus(
                should_continue=False,
                reason="Research not found",
                progress_percent=0
            )
        
        config = progress["config"]
        elapsed = datetime.now() - progress["start_time"]
        
        # Check time budget
        if elapsed > config.time_budget:
            return ProgressStatus(
                should_continue=False,
                reason="Time budget exceeded",
                progress_percent=self._calculate_progress(progress),
                sources_collected=progress["sources_collected"],
                average_source_quality=progress["average_source_quality"]
            )
        
        # Check source limit
        if progress["sources_collected"] >= config.max_sources:
            return ProgressStatus(
                should_continue=False,
                reason="Source limit reached",
                progress_percent=100,
                sources_collected=progress["sources_collected"],
                average_source_quality=progress["average_source_quality"]
            )
        
        # Check quality threshold
        if (progress["average_source_quality"] > 0.8 and 
            progress["sources_collected"] >= 5):
            return ProgressStatus(
                should_continue=False,
                reason="Quality threshold met",
                progress_percent=self._calculate_progress(progress),
                sources_collected=progress["sources_collected"],
                average_source_quality=progress["average_source_quality"]
            )
        
        # Check minimum requirements met
        if progress["sources_collected"] >= 3 and progress["facts_extracted"] >= 10:
            # Can continue but have minimum viable data
            pass
        
        return ProgressStatus(
            should_continue=True,
            progress_percent=self._calculate_progress(progress),
            sources_collected=progress["sources_collected"],
            average_source_quality=progress["average_source_quality"]
        )
    
    def _calculate_progress(self, progress: Dict[str, Any]) -> float:
        """Calculate progress percentage"""
        config = progress["config"]
        
        # Based on sources collected
        source_progress = (progress["sources_collected"] / config.max_sources) * 100
        
        # Based on time elapsed
        elapsed = datetime.now() - progress["start_time"]
        time_progress = (elapsed / config.time_budget) * 100
        
        # Combined progress
        return min(100, max(source_progress, time_progress))
    
    def update_progress(
        self,
        research_id: str,
        sources_collected: int = None,
        facts_extracted: int = None,
        source_quality: float = None
    ):
        """Update research progress"""
        if research_id not in self.active_research:
            return
        
        progress = self.active_research[research_id]
        
        if sources_collected is not None:
            progress["sources_collected"] = sources_collected
        
        if facts_extracted is not None:
            progress["facts_extracted"] = facts_extracted
        
        if source_quality is not None:
            # Update running average
            n = progress["sources_collected"]
            if n > 0:
                old_avg = progress["average_source_quality"]
                progress["average_source_quality"] = (
                    (old_avg * (n - 1) + source_quality) / n
                )
    
    def complete_research(self, research_id: str):
        """Mark research as complete"""
        if research_id in self.active_research:
            del self.active_research[research_id]


class AdaptiveResearchController:
    """
    Adaptively control research based on findings.
    
    Adjusts research strategy based on:
    - Information density
    - Conflicts detected
    - Knowledge gaps identified
    - High-quality sources found
    """
    
    def __init__(self):
        self.adaptation_history: List[Dict[str, Any]] = []
    
    async def adapt_research(
        self,
        current_findings: List[ExtractedInformation],
        config: ResearchConfig
    ) -> AdaptationDecision:
        """
        Adapt research strategy based on current findings.
        
        Args:
            current_findings: Information extracted so far
            config: Current research configuration
            
        Returns:
            Adaptation decision
        """
        decision = AdaptationDecision()
        
        # Analyze current findings
        analysis = await self._analyze_findings(current_findings)
        
        # Make adaptation decisions
        
        # 1. Low information density - broaden search
        if analysis["information_density"] < 0.3:
            decision.should_broaden = True
            decision.new_queries = await self._generate_broader_queries(
                current_findings
            )
            logger.info("Adapting: Broadening search due to low information density")
        
        # 2. Conflicts detected - deepen research
        if analysis["has_conflicts"]:
            decision.should_deepen = True
            decision.conflict_resolution_needed = True
            logger.info("Adapting: Deepening research to resolve conflicts")
        
        # 3. Knowledge gaps identified
        if analysis["knowledge_gaps"]:
            decision.additional_topics = analysis["knowledge_gaps"]
            logger.info(f"Adapting: Adding {len(analysis['knowledge_gaps'])} topics for gap filling")
        
        # 4. High-quality sources found - explore related
        if analysis["high_quality_sources"]:
            decision.explore_related = True
            decision.related_topics = await self._find_related_topics(
                analysis["high_quality_sources"]
            )
            logger.info("Adapting: Exploring related topics from high-quality sources")
        
        # Track adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "decision": decision.model_dump()
        })
        
        return decision
    
    async def _analyze_findings(
        self,
        findings: List[ExtractedInformation]
    ) -> Dict[str, Any]:
        """Analyze findings for adaptation decisions"""
        analysis = {
            "information_density": 0.0,
            "has_conflicts": False,
            "knowledge_gaps": [],
            "high_quality_sources": []
        }
        
        if not findings:
            return analysis
        
        # Calculate information density
        total_facts = sum(len(f.facts) for f in findings)
        total_content = sum(
            len(f.source.snippet or "") 
            for f in findings
        )
        
        if total_content > 0:
            analysis["information_density"] = min(1.0, total_facts / (total_content / 100))
        
        # Check for conflicts
        for finding in findings:
            if finding.conflicts:
                analysis["has_conflicts"] = True
                break
        
        # Identify gaps
        analysis["knowledge_gaps"] = self._identify_gaps(findings)
        
        # Find high quality sources
        analysis["high_quality_sources"] = [
            f for f in findings
            if f.source.credibility_score > 0.8
        ]
        
        return analysis
    
    def _identify_gaps(
        self,
        findings: List[ExtractedInformation]
    ) -> List[str]:
        """Identify knowledge gaps from findings"""
        gaps = []
        
        # Check for missing entity types
        entity_types = set()
        for finding in findings:
            for entity in finding.entities:
                entity_types.add(entity.type)
        
        # If few entity types, might need broader search
        if len(entity_types) < 3:
            gaps.append("broader_entity_coverage")
        
        # Check for missing fact categories
        fact_categories = set()
        for finding in findings:
            for fact in finding.facts:
                fact_categories.add(fact.category)
        
        if len(fact_categories) < 2:
            gaps.append("diverse_fact_types")
        
        return gaps
    
    async def _generate_broader_queries(
        self,
        findings: List[ExtractedInformation]
    ) -> List[SearchQuery]:
        """Generate broader search queries"""
        queries = []
        
        # Extract topics from findings
        topics = set()
        for finding in findings:
            for entity in finding.entities:
                if entity.type in ["CONCEPT", "TECHNOLOGY"]:
                    topics.add(entity.name)
        
        # Generate broader queries
        for topic in list(topics)[:3]:
            queries.append(SearchQuery(
                text=f"{topic} overview introduction",
                target_engines=["google", "bing"],
                query_type="broadening",
                priority=0.6
            ))
        
        return queries
    
    async def _find_related_topics(
        self,
        high_quality_sources: List[ExtractedInformation]
    ) -> List[str]:
        """Find related topics from high-quality sources"""
        related = set()
        
        for source in high_quality_sources:
            for entity in source.entities:
                if entity.mentions > 1:
                    related.add(entity.name)
            
            for relationship in source.relationships:
                related.add(relationship.object)
        
        return list(related)[:10]
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        return self.adaptation_history


class ResourceMonitor:
    """
    Monitor system resources for research operations.
    
    Tracks:
    - Token usage
    - API rate limits
    - Memory usage
    - Disk space
    """
    
    def __init__(self):
        self.usage_history: List[Dict[str, Any]] = []
    
    async def check_resources(self) -> Dict[str, Any]:
        """Check current resource availability"""
        resources = {
            "tokens_available": await self._check_token_availability(),
            "rate_limits_ok": await self._check_rate_limits(),
            "memory_available": await self._check_memory(),
            "disk_space_ok": await self._check_disk_space(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.usage_history.append(resources)
        
        return resources
    
    async def _check_token_availability(self) -> int:
        """Check available tokens from budget tracking."""
        import os
        budget = int(os.environ.get('TOKEN_BUDGET', '100000'))
        used = sum(getattr(r, 'tokens_used', 0) for r in self.usage_history) if self.usage_history else 0
        return max(0, budget - used)
    
    async def _check_rate_limits(self) -> bool:
        """Check if rate limits are approaching."""
        import os
        if not self.usage_history:
            return True
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent = [r for r in self.usage_history
                  if getattr(r, 'timestamp', datetime.min) > one_minute_ago]
        max_rpm = int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '60'))
        return len(recent) < max_rpm
    
    async def _check_memory(self) -> bool:
        """Check memory availability."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.percent < 90  # Available if less than 90% used
        except ImportError:
            return True
    
    async def _check_disk_space(self) -> bool:
        """Check disk space availability."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 95  # Available if less than 95% used
        except ImportError:
            return True
    
    def get_usage_trends(self) -> Dict[str, Any]:
        """Get resource usage trends"""
        if not self.usage_history:
            return {}
        
        return {
            "total_checks": len(self.usage_history),
            "recent": self.usage_history[-10:]
        }


class TimeBudgetManager:
    """
    Manage time budgets for research tasks.
    
    Ensures research completes within allocated time.
    """
    
    def __init__(self):
        self.budgets: Dict[str, Dict[str, Any]] = {}
    
    def set_budget(self, task_id: str, budget_seconds: int):
        """Set time budget for a task"""
        self.budgets[task_id] = {
            "budget": timedelta(seconds=budget_seconds),
            "start_time": datetime.now(),
            "extensions": 0
        }
    
    def check_budget(self, task_id: str) -> Dict[str, Any]:
        """Check remaining budget for a task"""
        if task_id not in self.budgets:
            return {"valid": False, "reason": "No budget set"}
        
        budget_info = self.budgets[task_id]
        elapsed = datetime.now() - budget_info["start_time"]
        remaining = budget_info["budget"] - elapsed
        
        return {
            "valid": remaining.total_seconds() > 0,
            "elapsed_seconds": elapsed.total_seconds(),
            "remaining_seconds": remaining.total_seconds(),
            "percent_used": (elapsed / budget_info["budget"]) * 100
        }
    
    def extend_budget(self, task_id: str, additional_seconds: int) -> bool:
        """Extend time budget for a task"""
        if task_id not in self.budgets:
            return False
        
        self.budgets[task_id]["budget"] += timedelta(seconds=additional_seconds)
        self.budgets[task_id]["extensions"] += 1
        
        return True
    
    def complete_task(self, task_id: str):
        """Mark task as complete and free budget"""
        if task_id in self.budgets:
            del self.budgets[task_id]
