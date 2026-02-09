"""
Research Trigger Mechanisms
Implements scheduled, event-driven, and curiosity-based research triggers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from croniter import croniter

from .models import (
    ResearchTask, TriggerType, TaskStatus, ResearchDepth,
    KnowledgeGap, RelatedTopic, TrendingTopic, SerendipitousDiscovery
)
from .config import ResearchLoopConfig, default_config, EVENT_TRIGGERS


logger = logging.getLogger(__name__)


class ScheduledResearchTrigger:
    """
    Cron-like scheduled research tasks.
    
    Supports multiple schedule types:
    - Daily briefings
    - Weekly deep dives
    - Hourly monitoring
    - Continuous learning
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or default_config
        self.last_run_times: Dict[str, datetime] = {}
    
    async def check_triggers(self, force: bool = False) -> List[ResearchTask]:
        """
        Check all scheduled triggers and return pending tasks.
        
        Args:
            force: If True, ignore schedule and return all tasks
            
        Returns:
            List of research tasks to execute
        """
        tasks = []
        current_time = datetime.now()
        
        for schedule_name, schedule_config in self.config.default_schedules.items():
            if not schedule_config.enabled:
                continue
            
            should_trigger = force or self._should_trigger(
                schedule_name,
                schedule_config.cron,
                current_time
            )
            
            if should_trigger:
                task = ResearchTask(
                    trigger_type=TriggerType.SCHEDULED,
                    trigger_source=schedule_name,
                    topic=f"Scheduled: {schedule_name}",
                    rationale=f"Scheduled research: {schedule_name}",
                    depth=ResearchDepth(schedule_config.depth),
                    sources=schedule_config.sources,
                    priority=self._calculate_priority(schedule_name),
                    immediate=False
                )
                
                # Add topics as context
                task.context = f"Topics: {', '.join(schedule_config.topics)}"
                
                tasks.append(task)
                self.last_run_times[schedule_name] = current_time
                
                logger.info(f"Scheduled trigger activated: {schedule_name}")
        
        return tasks
    
    def _should_trigger(
        self,
        schedule_name: str,
        cron_expression: str,
        current_time: datetime
    ) -> bool:
        """Check if a schedule should trigger"""
        last_run = self.last_run_times.get(schedule_name)
        
        if last_run is None:
            # Never run before - check if we should run now
            cron = croniter(cron_expression, current_time - timedelta(minutes=1))
            next_run = cron.get_next(datetime)
            # Allow 1 minute window
            return abs((next_run - current_time).total_seconds()) < 60
        
        # Check if enough time has passed
        cron = croniter(cron_expression, last_run)
        next_run = cron.get_next(datetime)
        
        return current_time >= next_run
    
    def _calculate_priority(self, schedule_name: str) -> float:
        """Calculate priority based on schedule type"""
        priorities = {
            "morning_briefing": 0.7,
            "weekly_research": 0.8,
            "hourly_check": 0.5,
            "continuous_learning": 0.4
        }
        return priorities.get(schedule_name, 0.5)


class EventDrivenResearchTrigger:
    """
    Event-based research triggers.
    
    Responds to system events such as:
    - Unknown topics in conversation
    - Outdated information
    - User requests
    - Memory gaps
    - Contradictions
    - News alerts
    - Trending topics
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or default_config
        self.event_queue: List[Dict[str, Any]] = []
        self.processed_events: set = set()
    
    async def check_triggers(self, force: bool = False) -> List[ResearchTask]:
        """
        Check for event-based triggers.
        
        Returns:
            List of research tasks triggered by events
        """
        tasks = []
        
        # Process queued events
        while self.event_queue:
            event = self.event_queue.pop(0)
            task = await self._process_event(event)
            if task:
                tasks.append(task)
        
        return tasks
    
    async def _process_event(self, event: Dict[str, Any]) -> Optional[ResearchTask]:
        """Process a single event and generate research task if needed"""
        event_type = event.get("type")
        event_id = event.get("id", str(hash(str(event))))
        
        # Skip already processed events
        if event_id in self.processed_events:
            return None
        
        self.processed_events.add(event_id)
        
        # Find matching trigger configuration
        trigger_config = EVENT_TRIGGERS.get(event_type)
        if not trigger_config:
            return None
        
        # Create research task
        task = ResearchTask(
            trigger_type=TriggerType.EVENT,
            trigger_source=event_type,
            topic=event.get("topic", f"Event: {event_type}"),
            context=event.get("context"),
            rationale=trigger_config["action"],
            depth=ResearchDepth(trigger_config["depth"]),
            immediate=trigger_config["immediate"],
            priority=self._calculate_event_priority(event)
        )
        
        logger.info(f"Event trigger activated: {event_type}")
        
        return task
    
    def queue_event(self, event: Dict[str, Any]):
        """Queue an event for processing"""
        event["queued_at"] = datetime.now().isoformat()
        self.event_queue.append(event)
        logger.debug(f"Event queued: {event.get('type')}")
    
    def _calculate_event_priority(self, event: Dict[str, Any]) -> float:
        """Calculate priority based on event characteristics"""
        base_priority = 0.7
        
        # Adjust based on event type
        type_multipliers = {
            "user_request": 1.0,
            "contradiction_detected": 0.95,
            "unknown_topic": 0.85,
            "news_alert": 0.8,
            "outdated_info": 0.6,
            "memory_gap": 0.5,
            "trending_topic": 0.55
        }
        
        multiplier = type_multipliers.get(event.get("type"), 0.5)
        
        # Adjust based on user importance
        if event.get("user_importance") == "high":
            multiplier += 0.1
        
        return min(1.0, base_priority * multiplier)


class CuriosityEngine:
    """
    Curiosity-driven research engine.
    
    Generates research tasks based on:
    - Knowledge gaps
    - Related topic exploration
    - Trend detection
    - Serendipitous discovery
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or default_config
        # Knowledge graph: topic -> {related_topics, facts, last_updated}
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        # Interest model: topic -> {interest_level, exposure_count, last_accessed}
        self.interest_model: Dict[str, Dict[str, Any]] = {}
        self.curiosity_score_threshold = self.config.triggers.min_curiosity_score
        self.daily_task_count = 0
        self.last_reset = datetime.now()
    
    async def generate_research(self, force: bool = False) -> List[ResearchTask]:
        """
        Generate curiosity-driven research tasks.
        
        Args:
            force: If True, ignore daily limits
            
        Returns:
            List of research tasks
        """
        # Reset daily count if needed
        self._reset_daily_count()
        
        # Check daily limit
        if not force and self.daily_task_count >= self.config.triggers.max_daily_tasks:
            logger.debug("Daily curiosity task limit reached")
            return []
        
        tasks = []
        
        # 1. Knowledge Gap Analysis
        gap_tasks = await self._generate_gap_research()
        tasks.extend(gap_tasks)
        
        # 2. Related Topic Exploration
        related_tasks = await self._generate_related_research()
        tasks.extend(related_tasks)
        
        # 3. Trend Detection
        trend_tasks = await self._generate_trend_research()
        tasks.extend(trend_tasks)
        
        # 4. Serendipitous Discovery
        serendipity_tasks = await self._generate_serendipity_research()
        tasks.extend(serendipity_tasks)
        
        # Sort by priority and limit
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        remaining_slots = self.config.triggers.max_daily_tasks - self.daily_task_count
        limited_tasks = tasks[:remaining_slots]
        
        self.daily_task_count += len(limited_tasks)
        
        if limited_tasks:
            logger.info(f"Generated {len(limited_tasks)} curiosity-driven research tasks")
        
        return limited_tasks
    
    async def _generate_gap_research(self) -> List[ResearchTask]:
        """Generate research tasks for knowledge gaps"""
        tasks = []
        
        gaps = await self._identify_knowledge_gaps()
        
        for gap in gaps:
            if gap.importance_score >= self.curiosity_score_threshold:
                task = ResearchTask(
                    trigger_type=TriggerType.CURIOSITY,
                    trigger_source="knowledge_gap",
                    topic=gap.topic,
                    rationale=f"Knowledge gap: {gap.description}",
                    depth=gap.recommended_depth,
                    priority=gap.importance_score,
                    immediate=False
                )
                tasks.append(task)
        
        return tasks
    
    async def _generate_related_research(self) -> List[ResearchTask]:
        """Generate research tasks for related topics"""
        tasks = []
        
        related = await self._find_related_topics()
        
        for topic in related:
            if topic.novelty_score >= self.curiosity_score_threshold:
                task = ResearchTask(
                    trigger_type=TriggerType.CURIOSITY,
                    trigger_source="related_exploration",
                    topic=topic.name,
                    rationale=f"Related to known topic: {topic.connection}",
                    depth=ResearchDepth.MEDIUM,
                    priority=topic.novelty_score * 0.8,
                    immediate=False
                )
                tasks.append(task)
        
        return tasks
    
    async def _generate_trend_research(self) -> List[ResearchTask]:
        """Generate research tasks for trending topics"""
        tasks = []
        
        trends = await self._detect_interesting_trends()
        
        for trend in trends:
            task = ResearchTask(
                trigger_type=TriggerType.CURIOSITY,
                trigger_source="trend_detection",
                topic=trend.topic,
                rationale=f"Trending in interest area: {trend.significance}",
                depth=ResearchDepth.SHALLOW,
                priority=trend.relevance_score,
                immediate=False
            )
            tasks.append(task)
        
        return tasks
    
    async def _generate_serendipity_research(self) -> List[ResearchTask]:
        """Generate research tasks from serendipitous discoveries"""
        tasks = []
        
        discoveries = await self._serendipitous_discovery()
        
        for discovery in discoveries:
            task = ResearchTask(
                trigger_type=TriggerType.CURIOSITY,
                trigger_source="serendipity",
                topic=discovery.topic,
                rationale=f"Unexpected connection: {discovery.connection}",
                depth=ResearchDepth.SHALLOW,
                priority=discovery.interest_score * 0.6,
                immediate=False
            )
            tasks.append(task)
        
        return tasks
    
    async def _identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Identify gaps in current knowledge base by analyzing the knowledge graph."""
        gaps = []

        if not self.knowledge_graph:
            return gaps

        now = datetime.now()

        for topic, info in self.knowledge_graph.items():
            facts = info.get("facts", [])
            related = info.get("related_topics", [])
            last_updated = info.get("last_updated")

            # Gap: topic has few facts recorded
            if len(facts) < 3:
                importance = min(1.0, 0.6 + (3 - len(facts)) * 0.1)
                gaps.append(KnowledgeGap(
                    topic=topic,
                    description=f"Only {len(facts)} facts recorded for '{topic}'",
                    importance_score=round(importance, 2),
                    recommended_depth=ResearchDepth.MEDIUM,
                    related_topics=related[:5]
                ))

            # Gap: stale information (older than 7 days)
            if last_updated:
                try:
                    if isinstance(last_updated, str):
                        last_updated = datetime.fromisoformat(last_updated)
                    age_days = (now - last_updated).days
                    if age_days > 7:
                        importance = min(1.0, 0.5 + age_days * 0.02)
                        gaps.append(KnowledgeGap(
                            topic=topic,
                            description=f"Information is {age_days} days old for '{topic}'",
                            importance_score=round(importance, 2),
                            recommended_depth=ResearchDepth.SHALLOW,
                            related_topics=related[:5]
                        ))
                except (ValueError, TypeError):
                    pass

            # Gap: related topics not yet in the graph
            for rel in related:
                if rel not in self.knowledge_graph:
                    gaps.append(KnowledgeGap(
                        topic=rel,
                        description=f"Related topic '{rel}' (from '{topic}') not yet researched",
                        importance_score=0.65,
                        recommended_depth=ResearchDepth.SHALLOW,
                        related_topics=[topic]
                    ))

        # Deduplicate by topic, keeping highest importance
        seen: Dict[str, KnowledgeGap] = {}
        for gap in gaps:
            if gap.topic not in seen or gap.importance_score > seen[gap.topic].importance_score:
                seen[gap.topic] = gap

        return sorted(seen.values(), key=lambda g: g.importance_score, reverse=True)
    
    async def _find_related_topics(self) -> List[RelatedTopic]:
        """Find topics related to current knowledge by traversing the knowledge graph."""
        related = []

        if not self.knowledge_graph:
            return related

        # Collect all topics that appear as related but are not primary entries,
        # or have low exposure in the interest model
        known_topics = set(self.knowledge_graph.keys())

        for topic, info in self.knowledge_graph.items():
            for rel_topic in info.get("related_topics", []):
                # Compute novelty: higher if not yet in graph or low exposure
                if rel_topic in known_topics:
                    exposure = 0
                    if rel_topic in self.interest_model:
                        exposure = self.interest_model[rel_topic].get("exposure_count", 0)
                    novelty = max(0.0, min(1.0, 1.0 - exposure * 0.1))
                else:
                    novelty = 0.9  # High novelty for unknown topics

                if novelty >= self.curiosity_score_threshold * 0.8:
                    related.append(RelatedTopic(
                        name=rel_topic,
                        connection=f"Related to known topic: {topic}",
                        novelty_score=round(novelty, 2)
                    ))

        # Deduplicate by name, keeping highest novelty
        seen: Dict[str, RelatedTopic] = {}
        for rt in related:
            if rt.name not in seen or rt.novelty_score > seen[rt.name].novelty_score:
                seen[rt.name] = rt

        return sorted(seen.values(), key=lambda r: r.novelty_score, reverse=True)
    
    async def _detect_interesting_trends(self) -> List[TrendingTopic]:
        """Detect trending topics based on interest model activity patterns."""
        trends = []

        if not self.interest_model:
            return trends

        now = datetime.now()

        # Sort topics by interest level to find what the user cares about most
        sorted_interests = sorted(
            self.interest_model.items(),
            key=lambda item: item[1].get("interest_level", 0),
            reverse=True
        )

        for topic, model_info in sorted_interests[:20]:
            interest_level = model_info.get("interest_level", 0)
            exposure_count = model_info.get("exposure_count", 0)
            last_accessed = model_info.get("last_accessed")

            # Only consider topics with meaningful interest
            if interest_level < 0.3:
                continue

            # Check if the topic has recent activity (accessed within 3 days)
            recency_boost = 0.0
            if last_accessed:
                try:
                    if isinstance(last_accessed, str):
                        last_accessed = datetime.fromisoformat(last_accessed)
                    days_since = (now - last_accessed).days
                    if days_since <= 3:
                        recency_boost = 0.2
                except (ValueError, TypeError):
                    pass

            # Topics with high interest and frequent exposure are trending
            relevance = min(1.0, interest_level * 0.6 + min(exposure_count, 10) * 0.03 + recency_boost)

            if relevance >= self.curiosity_score_threshold * 0.7:
                # Determine source based on whether it exists in knowledge graph
                source = "knowledge_graph" if topic in self.knowledge_graph else "interest_model"
                trends.append(TrendingTopic(
                    topic=topic,
                    significance=f"High interest ({interest_level:.1f}) with {exposure_count} exposures",
                    relevance_score=round(relevance, 2),
                    source=source
                ))

        return sorted(trends, key=lambda t: t.relevance_score, reverse=True)[:10]
    
    async def _serendipitous_discovery(self) -> List[SerendipitousDiscovery]:
        """Discover unexpected connections by finding shared related topics across distant domains."""
        discoveries = []

        if not self.knowledge_graph or len(self.knowledge_graph) < 2:
            return discoveries

        topics = list(self.knowledge_graph.keys())

        # Compare each pair of topics to find unexpected shared related topics
        for i, topic_a in enumerate(topics):
            related_a = set(self.knowledge_graph[topic_a].get("related_topics", []))
            if not related_a:
                continue

            for topic_b in topics[i + 1:]:
                related_b = set(self.knowledge_graph[topic_b].get("related_topics", []))
                if not related_b:
                    continue

                # Find shared related topics that create a bridge
                shared = related_a & related_b

                # Only interesting if topics themselves are not directly related
                if topic_b in related_a or topic_a in related_b:
                    continue

                for bridge in shared:
                    # Calculate interest score from interest model
                    interest_a = self.interest_model.get(topic_a, {}).get("interest_level", 0.5)
                    interest_b = self.interest_model.get(topic_b, {}).get("interest_level", 0.5)
                    interest_score = min(1.0, (interest_a + interest_b) / 2 + 0.1)

                    discoveries.append(SerendipitousDiscovery(
                        topic=f"{topic_a} and {topic_b}",
                        connection=f"Unexpected link via shared concept '{bridge}'",
                        interest_score=round(interest_score, 2),
                        discovered_via="knowledge_graph_cross_analysis"
                    ))

        # Deduplicate and sort by interest score
        seen_topics = set()
        unique = []
        for d in sorted(discoveries, key=lambda x: x.interest_score, reverse=True):
            if d.topic not in seen_topics:
                seen_topics.add(d.topic)
                unique.append(d)

        return unique[:5]
    
    def _reset_daily_count(self):
        """Reset daily task count if a day has passed"""
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.daily_task_count = 0
            self.last_reset = now


class TriggerEngine:
    """
    Unified trigger engine that manages all trigger types.
    
    Coordinates scheduled, event-driven, and curiosity-based triggers.
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or default_config
        self.scheduled_trigger = ScheduledResearchTrigger(config)
        self.event_trigger = EventDrivenResearchTrigger(config)
        self.curiosity_engine = CuriosityEngine(config)
    
    async def check_scheduled_triggers(self, force: bool = False) -> List[ResearchTask]:
        """Check scheduled triggers"""
        return await self.scheduled_trigger.check_triggers(force)
    
    async def check_event_triggers(self, force: bool = False) -> List[ResearchTask]:
        """Check event triggers"""
        return await self.event_trigger.check_triggers(force)
    
    async def check_curiosity_triggers(self, force: bool = False) -> List[ResearchTask]:
        """Check curiosity triggers"""
        return await self.curiosity_engine.generate_research(force)
    
    async def check_all_triggers(self) -> List[ResearchTask]:
        """Check all trigger types and return combined tasks"""
        all_tasks = []
        
        # Check scheduled triggers
        if self.config.triggers.schedules:
            scheduled = await self.check_scheduled_triggers()
            all_tasks.extend(scheduled)
        
        # Check event triggers
        if self.config.triggers.enabled:
            events = await self.check_event_triggers()
            all_tasks.extend(events)
        
        # Check curiosity triggers
        if self.config.triggers.enabled:
            curiosity = await self.check_curiosity_triggers()
            all_tasks.extend(curiosity)
        
        # Sort by priority
        all_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        return all_tasks
    
    def queue_event(self, event_type: str, **kwargs):
        """Queue an event for processing"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.event_trigger.queue_event(event)
        logger.debug(f"Event queued: {event_type}")
    
    # Convenience methods for common events
    
    def on_unknown_topic(self, topic: str, context: Optional[str] = None):
        """Trigger research on unknown topic"""
        self.queue_event(
            "unknown_topic",
            topic=topic,
            context=context
        )
    
    def on_user_request(self, topic: str, depth: str = "deep"):
        """Trigger user-requested research"""
        self.queue_event(
            "user_request",
            topic=topic,
            requested_depth=depth,
            user_importance="high"
        )
    
    def on_contradiction_detected(
        self,
        topic: str,
        conflicting_sources: List[str]
    ):
        """Trigger research to resolve contradiction"""
        self.queue_event(
            "contradiction_detected",
            topic=topic,
            conflicting_sources=conflicting_sources
        )
    
    def on_news_alert(self, topic: str, significance: str):
        """Trigger research on news alert"""
        self.queue_event(
            "news_alert",
            topic=topic,
            significance=significance
        )
