"""
Main Research Loop Implementation
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

from .models import (
    ResearchTask, SearchQuery, DiscoveredSource, CrawledContent,
    ExtractedInformation, SynthesisResult, ResearchConfig, CrawlConfig,
    ExtractionGoal, TaskStatus, ResearchDepth, TriggerType
)
from .config import ResearchLoopConfig, default_config


logger = logging.getLogger(__name__)


class ResearchLoop:
    """
    Main research loop orchestrator for autonomous information gathering.
    
    This class coordinates all research activities including:
    - Trigger detection and task generation
    - Query generation and optimization
    - Source discovery and crawling
    - Information extraction and synthesis
    - Knowledge consolidation to memory
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        """
        Initialize the research loop.
        
        Args:
            config: Research loop configuration. Uses default if not provided.
        """
        self.config = config or default_config
        self.running = False
        
        # Initialize components
        self.trigger_engine = None
        self.query_generator = None
        self.source_discovery = None
        self.information_extractor = None
        self.synthesis_engine = None
        self.memory_writer = None
        self.citation_tracker = None
        self.depth_controller = None
        
        # Task management
        self.task_queue = asyncio.Queue(maxsize=self.config.performance.task_queue_size)
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.task_history: List[ResearchTask] = []
        
        # Statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "sources_discovered": 0,
            "facts_extracted": 0,
            "start_time": None
        }
        
        logger.info("Research Loop initialized")
    
    async def initialize(self):
        """Initialize all research components"""
        logger.info("Initializing research components...")
        
        # Import and initialize components
        from .triggers import TriggerEngine
        from .query_generator import QueryGenerator
        from .source_discovery import SourceDiscoveryEngine
        from .extractor import InformationExtractor
        from .synthesizer import SynthesisEngine
        from .memory_writer import MemoryWriter
        from .citation_tracker import CitationTracker
        from .depth_controller import DepthController
        
        self.trigger_engine = TriggerEngine(self.config)
        self.query_generator = QueryGenerator()
        self.source_discovery = SourceDiscoveryEngine(self.config)
        self.information_extractor = InformationExtractor()
        self.synthesis_engine = SynthesisEngine()
        self.memory_writer = MemoryWriter(self.config.memory.memory_file)
        self.citation_tracker = CitationTracker(self.config.memory.citation_db)
        self.depth_controller = DepthController()
        
        logger.info("All research components initialized")
    
    async def run(self):
        """
        Main research loop - runs continuously until stopped.
        
        This method:
        1. Checks for research triggers
        2. Queues research tasks
        3. Executes queued tasks
        4. Sends heartbeat signals
        """
        if not self.config.enabled:
            logger.info("Research Loop is disabled")
            return
        
        await self.initialize()
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        logger.info("Research Loop started")
        
        while self.running:
            try:
                # Check for triggers and add tasks to queue
                await self._check_triggers()
                
                # Process queued tasks
                await self._process_task_queue()
                
                # Send heartbeat
                await self._send_heartbeat()
                
                # Sleep before next iteration
                await asyncio.sleep(self.config.poll_interval)
                
            except (OSError, RuntimeError, ValueError, ConnectionError) as e:
                logger.error(f"Research Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("Research Loop stopped")
    
    async def stop(self):
        """Stop the research loop gracefully"""
        logger.info("Stopping Research Loop...")
        self.running = False
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks...")
            await asyncio.sleep(2)
    
    async def _check_triggers(self):
        """Check all trigger types and queue research tasks"""
        if not self.trigger_engine:
            return
        
        try:
            # Check scheduled triggers
            scheduled_tasks = await self.trigger_engine.check_scheduled_triggers()
            for task in scheduled_tasks:
                await self._queue_task(task)
            
            # Check event triggers
            event_tasks = await self.trigger_engine.check_event_triggers()
            for task in event_tasks:
                await self._queue_task(task)
            
            # Check curiosity triggers
            curiosity_tasks = await self.trigger_engine.check_curiosity_triggers()
            for task in curiosity_tasks:
                await self._queue_task(task)
                
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error checking triggers: {e}")
    
    async def _queue_task(self, task: ResearchTask):
        """Add a research task to the queue"""
        try:
            self.task_queue.put_nowait(task)
            logger.info(f"Queued research task: {task.id} - {task.topic}")
        except asyncio.QueueFull:
            logger.warning(f"Task queue full, dropping task: {task.id}")
    
    async def _process_task_queue(self):
        """Process tasks from the queue"""
        # Limit concurrent tasks
        active_count = len(self.active_tasks)
        max_concurrent = self.config.performance.max_concurrent_tasks
        
        available_slots = max_concurrent - active_count
        
        for _ in range(available_slots):
            if self.task_queue.empty():
                break
            
            try:
                task = self.task_queue.get_nowait()
                # Start task execution in background
                asyncio.create_task(self._execute_research(task))
            except asyncio.QueueEmpty:
                break
    
    async def _execute_research(self, task: ResearchTask):
        """
        Execute a complete research task.
        
        This is the main research pipeline that:
        1. Configures research depth
        2. Generates search queries
        3. Discovers and crawls sources
        4. Extracts information
        5. Synthesizes findings
        6. Consolidates to memory
        """
        logger.info(f"Starting research task: {task.id} - {task.topic}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.active_tasks[task.id] = task
        
        try:
            # 1. Configure research depth
            research_config = await self.depth_controller.configure_research(task)
            logger.debug(f"Research config: {research_config}")
            
            # 2. Generate search queries
            queries = await self.query_generator.generate_queries(
                task,
                self._get_research_context()
            )
            logger.info(f"Generated {len(queries)} search queries")
            
            # 3. Discover sources
            sources = await self.source_discovery.discover_sources(
                queries,
                research_config
            )
            logger.info(f"Discovered {len(sources)} sources")
            self.stats["sources_discovered"] += len(sources)
            
            # 4. Crawl and extract content
            crawled_content = await self._crawl_sources(
                sources,
                research_config
            )
            logger.info(f"Successfully crawled {len(crawled_content)} sources")
            
            # 5. Extract structured information
            extraction_goals = self._derive_extraction_goals(task)
            extracted_info = await self._extract_information(
                crawled_content,
                extraction_goals
            )
            total_facts = sum(len(info.facts) for info in extracted_info)
            logger.info(f"Extracted {total_facts} facts")
            self.stats["facts_extracted"] += total_facts
            
            # 6. Synthesize findings
            synthesis = await self.synthesis_engine.synthesize(
                extracted_info,
                task
            )
            logger.info(f"Synthesis complete with confidence {synthesis.confidence_score}")
            
            # 7. Consolidate to memory
            consolidation = await self.memory_writer.consolidate(
                synthesis,
                task
            )
            logger.info(f"Memory consolidation: {consolidation.entries_created} new entries")
            
            # 8. Track citations
            for info in extracted_info:
                await self.citation_tracker.track_source(
                    info.source,
                    {"research_task": task.id}
                )
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = {
                "sources_used": len(crawled_content),
                "facts_extracted": total_facts,
                "confidence_score": synthesis.confidence_score,
                "entries_created": consolidation.entries_created
            }
            
            self.stats["tasks_completed"] += 1
            self.task_history.append(task)
            
            # Emit completion event
            await self._emit_completion_event(task, synthesis, consolidation)
            
            logger.info(f"Research task completed: {task.id}")
            
        except (OSError, RuntimeError, ValueError, ConnectionError, asyncio.TimeoutError) as e:
            logger.error(f"Research task failed: {task.id} - {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.stats["tasks_failed"] += 1
            await self._emit_failure_event(task, e)
            
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _crawl_sources(
        self,
        sources: List[DiscoveredSource],
        config: ResearchConfig
    ) -> List[CrawledContent]:
        """Crawl discovered sources up to configured limits"""
        crawled = []
        
        # Limit sources based on config
        limited_sources = sources[:config.max_sources]
        
        # Create crawl config
        crawl_config = CrawlConfig.from_research_config(config)
        
        # Crawl sources concurrently
        semaphore = asyncio.Semaphore(self.config.crawling.max_concurrent)
        
        async def crawl_with_limit(source: DiscoveredSource) -> Optional[CrawledContent]:
            async with semaphore:
                try:
                    return await self.source_discovery.crawler.crawl_source(
                        source,
                        crawl_config
                    )
                except (OSError, ConnectionError, asyncio.TimeoutError, ValueError) as e:
                    logger.warning(f"Failed to crawl {source.url}: {e}")
                    return None
        
        # Execute crawls
        tasks = [crawl_with_limit(source) for source in limited_sources]
        results = await asyncio.gather(*tasks)
        
        # Filter successful crawls
        crawled = [r for r in results if r is not None]
        
        return crawled
    
    async def _extract_information(
        self,
        crawled_content: List[CrawledContent],
        goals: List[ExtractionGoal]
    ) -> List[ExtractedInformation]:
        """Extract structured information from crawled content"""
        extracted = []
        
        for content in crawled_content:
            try:
                info = await self.information_extractor.extract_information(
                    content,
                    goals
                )
                extracted.append(info)
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to extract from {content.source.url}: {e}")
        
        return extracted
    
    def _derive_extraction_goals(self, task: ResearchTask) -> List[ExtractionGoal]:
        """Derive information extraction goals from research task"""
        goals = []
        
        # Primary goal from task topic
        goals.append(ExtractionGoal(
            topic=task.topic,
            description=f"Extract key information about {task.topic}",
            priority=1.0
        ))
        
        # Context-based goals
        if task.context:
            goals.append(ExtractionGoal(
                topic="context",
                description=f"Understand context: {task.context}",
                priority=0.8
            ))
        
        # Depth-based goals
        depth_goals = {
            ResearchDepth.SURFACE: ["basic_facts", "definitions"],
            ResearchDepth.SHALLOW: ["basic_facts", "definitions", "key_points"],
            ResearchDepth.MEDIUM: ["facts", "relationships", "statistics", "opinions"],
            ResearchDepth.DEEP: ["comprehensive", "technical_details", "research_findings"],
            ResearchDepth.EXHAUSTIVE: ["complete_coverage", "all_aspects", "contradictions"]
        }
        
        for goal_type in depth_goals.get(task.depth, []):
            goals.append(ExtractionGoal(
                topic=goal_type,
                description=f"Extract {goal_type}",
                priority=0.6
            ))
        
        return goals
    
    def _get_research_context(self) -> Dict[str, Any]:
        """Get current research context"""
        return {
            "recent_topics": [t.topic for t in self.task_history[-10:]],
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "stats": self.stats
        }
    
    async def _send_heartbeat(self):
        """Send heartbeat signal to system by writing a heartbeat JSON file."""
        heartbeat_data = {
            "loop": "research",
            "status": "running" if self.running else "stopped",
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "stats": {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.stats.items()
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.debug(f"Heartbeat: {heartbeat_data}")

        # Write heartbeat to a file so other components can monitor liveness
        heartbeat_path = Path(os.environ.get(
            "RESEARCH_HEARTBEAT_FILE", "research_heartbeat.json"
        ))
        try:
            heartbeat_path.write_text(
                json.dumps(heartbeat_data, indent=2, default=str),
                encoding="utf-8"
            )
        except OSError as e:
            logger.debug(f"Could not write heartbeat file: {e}")
    
    async def _emit_completion_event(
        self,
        task: ResearchTask,
        synthesis: SynthesisResult,
        consolidation: Any
    ):
        """Emit research completion event by appending to event log file."""
        event_data = {
            "event": "research_completed",
            "task_id": task.id,
            "topic": task.topic,
            "sources_used": synthesis.sources_used,
            "confidence_score": synthesis.confidence_score,
            "entries_created": consolidation.entries_created if consolidation else 0,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Research completion event: {event_data}")

        await self._append_event(event_data)
    
    async def _emit_failure_event(self, task: ResearchTask, error: Exception):
        """Emit research failure event by appending to event log file."""
        event_data = {
            "event": "research_failed",
            "task_id": task.id,
            "topic": task.topic,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }

        logger.error(f"Research failure event: {event_data}")

        await self._append_event(event_data)

    async def _append_event(self, event_data: Dict[str, Any]):
        """Append an event to the research events log file."""
        event_log_path = Path(os.environ.get(
            "RESEARCH_EVENT_LOG", "research_events.jsonl"
        ))
        try:
            line = json.dumps(event_data, default=str) + "\n"
            with open(event_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.debug(f"Could not write event log: {e}")
    
    # Public API methods
    
    async def submit_research_request(
        self,
        topic: str,
        depth: str = "medium",
        context: Optional[str] = None
    ) -> str:
        """
        Submit a user research request.
        
        Args:
            topic: Research topic
            depth: Research depth (surface, shallow, medium, deep, exhaustive)
            context: Additional context
            
        Returns:
            Task ID for tracking
        """
        task = ResearchTask(
            trigger_type=TriggerType.USER,
            trigger_source="user_request",
            topic=topic,
            context=context,
            depth=ResearchDepth(depth),
            immediate=True,
            priority=0.9
        )
        
        await self._queue_task(task)
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a research task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "topic": task.topic,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "progress": "in_progress"
            }
        
        # Check history
        for task in self.task_history:
            if task.id == task_id:
                return {
                    "id": task.id,
                    "status": task.status.value,
                    "topic": task.topic,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "result": task.result,
                    "error": task.error
                }
        
        return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get research loop statistics"""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "total_tasks": len(self.task_history) + len(self.active_tasks),
            "uptime": (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0
        }
    
    async def force_trigger(self, trigger_name: str) -> List[str]:
        """Force a specific trigger to execute"""
        task_ids = []
        
        if trigger_name == "scheduled":
            tasks = await self.trigger_engine.check_scheduled_triggers(force=True)
        elif trigger_name == "curiosity":
            tasks = await self.trigger_engine.check_curiosity_triggers(force=True)
        else:
            return []
        
        for task in tasks:
            await self._queue_task(task)
            task_ids.append(task.id)
        
        return task_ids
