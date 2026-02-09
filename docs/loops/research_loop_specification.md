# Research Loop Technical Specification
## OpenClaw-Inspired AI Agent System - Windows 10 Edition

**Version:** 1.0  
**Date:** 2025-01-20  
**Component:** RESEARCH LOOP (1 of 15 Agentic Loops)  
**AI Engine:** GPT-5.2 with Enhanced Thinking Capability

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Research Trigger Mechanisms](#research-trigger-mechanisms)
4. [Search Query Generation](#search-query-generation)
5. [Source Discovery & Crawling](#source-discovery--crawling)
6. [Information Extraction & Parsing](#information-extraction--parsing)
7. [Synthesis & Summarization](#synthesis--summarization)
8. [Knowledge Consolidation](#knowledge-consolidation)
9. [Source Tracking & Citations](#source-tracking--citations)
10. [Research Depth & Breadth Control](#research-depth--breadth-control)
11. [Implementation Specifications](#implementation-specifications)
12. [Integration Points](#integration-points)

---

## Executive Summary

The Research Loop is an autonomous information gathering and synthesis subsystem designed for continuous learning and knowledge acquisition. It operates as one of 15 hardcoded agentic loops within the OpenClaw-inspired AI agent framework, enabling the system to:

- **Autonomously discover** relevant information based on context, user needs, and curiosity drivers
- **Execute multi-source research** with intelligent query generation and source evaluation
- **Synthesize findings** into structured, actionable knowledge
- **Maintain persistent knowledge** through MEMORY.md integration
- **Track and cite sources** for transparency and verification

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Research** | Self-directed information gathering based on triggers |
| **Multi-Source Aggregation** | Web search, browser automation, API integrations |
| **Intelligent Synthesis** | GPT-5.2 powered summarization and knowledge extraction |
| **Persistent Memory** | Automatic consolidation into MEMORY.md |
| **Source Verification** | Reliability scoring and citation tracking |
| **Adaptive Depth** | Configurable research breadth and depth parameters |

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH LOOP ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   TRIGGER    │───▶│   QUERY      │───▶│   SOURCE     │                  │
│  │   ENGINE     │    │   GENERATOR  │    │   DISCOVERY  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │              RESEARCH ORCHESTRATOR                    │                 │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │                 │
│  │  │  Scheduled │  │  Event-    │  │  Curiosity │     │                 │
│  │  │  Research  │  │  Driven    │  │  Engine    │     │                 │
│  │  └────────────┘  └────────────┘  └────────────┘     │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                            │                                               │
│                            ▼                                               │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │           INFORMATION PROCESSING PIPELINE             │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐│                 │
│  │  │ Extraction│─▶│ Parsing  │─▶│ Synthesis│─▶│ Summary││                 │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘│                 │
│  └──────────────────────────────────────────────────────┘                 │
│                            │                                               │
│                            ▼                                               │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │              KNOWLEDGE CONSOLIDATION                  │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │                 │
│  │  │  Memory  │  │  Source  │  │  Cross-Reference │   │                 │
│  │  │  Update  │  │  Tracking│  │  & Linking       │   │                 │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Modules

```python
# Module Structure
research_loop/
├── __init__.py
├── triggers.py          # Research trigger mechanisms
├── query_generator.py   # Search query generation
├── source_discovery.py  # Source finding and crawling
├── extractor.py         # Information extraction
├── synthesizer.py       # Content synthesis
├── memory_writer.py     # MEMORY.md integration
├── citation_tracker.py  # Source tracking
├── depth_controller.py  # Research depth management
├── models.py            # Data models
└── config.py            # Configuration
```

---

## Research Trigger Mechanisms

### 1. Scheduled Research (Time-Based)

**Purpose:** Regular, proactive information gathering on predefined topics

**Implementation:**

```python
class ScheduledResearchTrigger:
    """
    Cron-like scheduled research tasks
    """
    
    SCHEDULES = {
        # Daily briefings
        "morning_briefing": {
            "cron": "0 8 * * *",  # 8 AM daily
            "topics": ["tech_news", "market_updates", "weather"],
            "depth": "shallow",
            "sources": ["news_api", "rss_feeds"]
        },
        
        # Weekly deep dives
        "weekly_research": {
            "cron": "0 10 * * 1",  # Monday 10 AM
            "topics": ["emerging_tech", "industry_trends"],
            "depth": "deep",
            "sources": ["all"]
        },
        
        # Hourly monitoring
        "hourly_check": {
            "cron": "0 * * * *",  # Every hour
            "topics": ["alerts", "notifications"],
            "depth": "surface",
            "sources": ["priority_feeds"]
        },
        
        # Continuous learning
        "continuous_learning": {
            "cron": "*/30 * * * *",  # Every 30 minutes
            "topics": ["user_interests", "knowledge_gaps"],
            "depth": "adaptive",
            "sources": ["dynamic"]
        }
    }
    
    async def check_triggers(self) -> List[ResearchTask]:
        """Check all schedules and return pending research tasks"""
        pending_tasks = []
        current_time = datetime.now()
        
        for schedule_name, config in self.SCHEDULES.items():
            if self._should_trigger(config["cron"], current_time):
                task = ResearchTask(
                    trigger_type="scheduled",
                    trigger_source=schedule_name,
                    topics=config["topics"],
                    depth=config["depth"],
                    sources=config["sources"],
                    priority=self._calculate_priority(schedule_name)
                )
                pending_tasks.append(task)
        
        return pending_tasks
```

### 2. Event-Driven Research (Context-Based)

**Purpose:** Reactive research triggered by specific events or user actions

**Trigger Events:**

```python
class EventDrivenResearchTrigger:
    """
    Event-based research triggers
    """
    
    EVENT_TRIGGERS = {
        # User conversation triggers
        "unknown_topic": {
            "condition": "user_mentions_unknown_concept",
            "action": "research_concept",
            "depth": "medium",
            "immediate": True
        },
        
        "outdated_info": {
            "condition": "information_age > 30_days",
            "action": "refresh_knowledge",
            "depth": "medium",
            "immediate": False
        },
        
        "user_request": {
            "condition": "explicit_research_request",
            "action": "deep_research",
            "depth": "deep",
            "immediate": True
        },
        
        # System triggers
        "memory_gap": {
            "condition": "missing_critical_information",
            "action": "fill_gap",
            "depth": "deep",
            "immediate": False
        },
        
        "contradiction_detected": {
            "condition": "conflicting_information_found",
            "action": "verify_and_resolve",
            "depth": "deep",
            "immediate": True
        },
        
        # External triggers
        "news_alert": {
            "condition": "breaking_news_in_user_interest_area",
            "action": "summarize_and_notify",
            "depth": "shallow",
            "immediate": True
        },
        
        "trending_topic": {
            "condition": "topic_trending_in_user_domain",
            "action": "research_trend",
            "depth": "medium",
            "immediate": False
        }
    }
    
    async def handle_event(self, event: SystemEvent) -> Optional[ResearchTask]:
        """Process incoming events and generate research tasks"""
        
        for trigger_name, config in self.EVENT_TRIGGERS.items():
            if self._matches_condition(event, config["condition"]):
                return ResearchTask(
                    trigger_type="event",
                    trigger_source=trigger_name,
                    context=event.context,
                    depth=config["depth"],
                    immediate=config["immediate"],
                    priority=self._event_priority(event)
                )
        
        return None
```

### 3. Curiosity-Driven Research (Autonomous)

**Purpose:** Self-directed learning based on knowledge gaps and interest modeling

```python
class CuriosityEngine:
    """
    Autonomous curiosity-driven research system
    """
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.interest_model = InterestModel()
        self.curiosity_score_threshold = 0.7
    
    async def generate_curiosity_research(self) -> List[ResearchTask]:
        """Generate research tasks based on curiosity drivers"""
        
        tasks = []
        
        # 1. Knowledge Gap Analysis
        gaps = await self._identify_knowledge_gaps()
        for gap in gaps:
            if gap.importance_score > self.curiosity_score_threshold:
                tasks.append(ResearchTask(
                    trigger_type="curiosity",
                    trigger_source="knowledge_gap",
                    topic=gap.topic,
                    depth=gap.recommended_depth,
                    rationale=f"Knowledge gap identified: {gap.description}",
                    priority=gap.importance_score
                ))
        
        # 2. Related Topic Exploration
        related = await self._find_related_topics()
        for topic in related:
            if topic.novelty_score > self.curiosity_score_threshold:
                tasks.append(ResearchTask(
                    trigger_type="curiosity",
                    trigger_source="related_exploration",
                    topic=topic.name,
                    depth="medium",
                    rationale=f"Related to known topic: {topic.connection}",
                    priority=topic.novelty_score * 0.8
                ))
        
        # 3. Trending in Interest Areas
        trends = await self._detect_interesting_trends()
        for trend in trends:
            tasks.append(ResearchTask(
                trigger_type="curiosity",
                trigger_source="trend_detection",
                topic=trend.topic,
                depth="shallow",
                rationale=f"Emerging trend in interest area: {trend.significance}",
                priority=trend.relevance_score
            ))
        
        # 4. Serendipitous Discovery
        serendipity = await self._serendipitous_discovery()
        for discovery in serendipity:
            tasks.append(ResearchTask(
                trigger_type="curiosity",
                trigger_source="serendipity",
                topic=discovery.topic,
                depth="shallow",
                rationale=f"Unexpected connection discovered: {discovery.connection}",
                priority=discovery.interest_score * 0.6
            ))
        
        return sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    async def _identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Identify gaps in current knowledge base"""
        gaps = []
        
        # Query knowledge graph for incomplete nodes
        incomplete = self.knowledge_graph.find_incomplete_nodes()
        
        for node in incomplete:
            gap = KnowledgeGap(
                topic=node.topic,
                description=node.missing_attributes,
                importance_score=self._calculate_gap_importance(node),
                recommended_depth=self._recommend_depth(node)
            )
            gaps.append(gap)
        
        return gaps
```

### Trigger Priority Matrix

| Trigger Type | Priority Range | Response Time | Max Concurrent |
|--------------|----------------|---------------|----------------|
| Event-Driven (Immediate) | 0.9 - 1.0 | < 5 seconds | 3 |
| Event-Driven (Deferred) | 0.7 - 0.9 | < 5 minutes | 5 |
| Curiosity-Driven | 0.5 - 0.8 | < 30 minutes | 10 |
| Scheduled | 0.3 - 0.7 | Per schedule | Unlimited |

---

## Search Query Generation

### Query Generation Pipeline

```python
class QueryGenerator:
    """
    Intelligent search query generation system
    """
    
    def __init__(self):
        self.llm = GPT52Client()  # GPT-5.2 integration
        self.query_templates = self._load_templates()
        self.query_history = QueryHistory()
    
    async def generate_queries(
        self, 
        research_task: ResearchTask,
        context: ResearchContext
    ) -> List[SearchQuery]:
        """Generate optimized search queries for a research task"""
        
        # Step 1: Extract key concepts
        concepts = await self._extract_concepts(research_task)
        
        # Step 2: Generate query variants
        variants = await self._generate_variants(concepts, context)
        
        # Step 3: Optimize for search engines
        optimized = await self._optimize_queries(variants)
        
        # Step 4: Deduplicate and rank
        final_queries = self._rank_and_deduplicate(optimized)
        
        return final_queries
    
    async def _extract_concepts(self, task: ResearchTask) -> List[Concept]:
        """Extract key concepts from research task using GPT-5.2"""
        
        prompt = f"""
        Analyze the following research task and extract key concepts for search query generation:
        
        Task: {task.topic}
        Context: {task.context}
        Rationale: {task.rationale}
        
        Extract:
        1. Primary subject (main topic)
        2. Secondary subjects (related topics)
        3. Key entities (people, organizations, products)
        4. Temporal context (time relevance)
        5. Geographic context (location relevance)
        6. Domain specificity (technical level)
        
        Return as structured JSON.
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="high",
            response_format="json"
        )
        
        return Concept.parse(response)
```

### Query Templates

```python
QUERY_TEMPLATES = {
    # Factual queries
    "definition": {
        "template": "what is {topic} definition meaning",
        "use_case": "basic_understanding",
        "engines": ["google", "bing", "duckduckgo"]
    },
    
    "how_to": {
        "template": "how to {action} {topic} tutorial guide",
        "use_case": "procedural_knowledge",
        "engines": ["google", "youtube", "stackoverflow"]
    },
    
    "comparison": {
        "template": "{topic_a} vs {topic_b} comparison differences",
        "use_case": "comparative_analysis",
        "engines": ["google", "reddit", "quora"]
    },
    
    "latest": {
        "template": "{topic} latest news updates {timeframe}",
        "use_case": "current_information",
        "engines": ["google_news", "bing_news", "twitter"]
    },
    
    "deep_dive": {
        "template": "{topic} comprehensive analysis research paper",
        "use_case": "deep_research",
        "engines": ["google_scholar", "arxiv", "semantic_scholar"]
    },
    
    "opinions": {
        "template": "{topic} opinions reviews experiences",
        "use_case": "sentiment_analysis",
        "engines": ["reddit", "twitter", "quora"]
    },
    
    "technical": {
        "template": "{topic} technical documentation specification",
        "use_case": "technical_details",
        "engines": ["github", "docs", "stackoverflow"]
    },
    
    "trends": {
        "template": "{topic} trends statistics data {timeframe}",
        "use_case": "trend_analysis",
        "engines": ["google_trends", "statista", "industry_reports"]
    }
}
```

### Multi-Engine Query Distribution

```python
class QueryDistributor:
    """
    Distribute queries across multiple search engines
    """
    
    ENGINE_CAPABILITIES = {
        "google": {
            "strengths": ["general", "comprehensive", "fresh"],
            "rate_limit": 100,  # queries per day
            "priority": 1
        },
        "bing": {
            "strengths": ["general", "api_friendly"],
            "rate_limit": 1000,
            "priority": 2
        },
        "duckduckgo": {
            "strengths": ["privacy", "no_rate_limit"],
            "rate_limit": None,
            "priority": 3
        },
        "google_scholar": {
            "strengths": ["academic", "research_papers"],
            "rate_limit": 100,
            "priority": 1
        },
        "arxiv": {
            "strengths": ["preprints", "cs_physics_math"],
            "rate_limit": None,
            "priority": 2
        },
        "reddit": {
            "strengths": ["opinions", "discussions", "community"],
            "rate_limit": 60,
            "priority": 2
        },
        "twitter": {
            "strengths": ["real_time", "trends", "sentiment"],
            "rate_limit": 450,
            "priority": 3
        },
        "github": {
            "strengths": ["code", "technical", "documentation"],
            "rate_limit": 5000,
            "priority": 2
        },
        "youtube": {
            "strengths": ["tutorials", "visual", "explanations"],
            "rate_limit": 10000,
            "priority": 3
        }
    }
    
    def distribute_queries(
        self, 
        queries: List[SearchQuery],
        research_depth: str
    ) -> Dict[str, List[SearchQuery]]:
        """Distribute queries to appropriate search engines"""
        
        distribution = defaultdict(list)
        
        for query in queries:
            # Determine best engines for this query type
            engines = self._select_engines(query, research_depth)
            
            # Distribute with fallback
            for engine in engines:
                if self._check_rate_limit(engine):
                    distribution[engine].append(query)
                    break
            else:
                # All engines at rate limit, queue for later
                self._queue_query(query)
        
        return dict(distribution)
```

---

## Source Discovery & Crawling

### Source Discovery Pipeline

```python
class SourceDiscoveryEngine:
    """
    Multi-source discovery and crawling system
    """
    
    def __init__(self):
        self.search_engines = SearchEngineManager()
        self.browser = BrowserController()
        self.crawler = WebCrawler()
        self.source_ranker = SourceRanker()
    
    async def discover_sources(
        self, 
        queries: List[SearchQuery],
        config: ResearchConfig
    ) -> List[DiscoveredSource]:
        """Discover relevant sources for research queries"""
        
        all_sources = []
        
        # Parallel search across engines
        search_tasks = [
            self._search_engine_query(query, config)
            for query in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Aggregate and deduplicate
        for results in search_results:
            all_sources.extend(results)
        
        # Remove duplicates
        unique_sources = self._deduplicate_sources(all_sources)
        
        # Rank sources by relevance and quality
        ranked_sources = await self.source_ranker.rank(
            unique_sources, 
            queries[0]  # Primary query for context
        )
        
        # Filter by research depth requirements
        filtered_sources = self._filter_by_depth(
            ranked_sources, 
            config.depth
        )
        
        return filtered_sources
    
    async def _search_engine_query(
        self, 
        query: SearchQuery,
        config: ResearchConfig
    ) -> List[DiscoveredSource]:
        """Execute search on appropriate engines"""
        
        sources = []
        
        for engine_name in query.target_engines:
            engine = self.search_engines.get(engine_name)
            
            try:
                results = await engine.search(
                    query=query.text,
                    num_results=config.results_per_engine,
                    filters=query.filters
                )
                
                for result in results:
                    source = DiscoveredSource(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        engine=engine_name,
                        query=query.text,
                        rank=result.rank,
                        timestamp=datetime.now()
                    )
                    sources.append(source)
                    
            except Exception as e:
                logger.warning(f"Search failed on {engine_name}: {e}")
                continue
        
        return sources
```

### Web Crawling System

```python
class WebCrawler:
    """
    Intelligent web crawling with content extraction
    """
    
    def __init__(self):
        self.session = aiohttp.ClientSession(
            headers=self._default_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.content_extractor = ContentExtractor()
        self.robots_parser = RobotsParser()
    
    async def crawl_source(
        self, 
        source: DiscoveredSource,
        config: CrawlConfig
    ) -> CrawledContent:
        """Crawl and extract content from a source"""
        
        # Check robots.txt
        if not await self.robots_parser.can_fetch(source.url):
            logger.info(f"Skipping {source.url} - blocked by robots.txt")
            return None
        
        try:
            # Fetch page
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {source.url}")
                    return None
                
                html = await response.text()
                
                # Extract content
                content = await self.content_extractor.extract(
                    html=html,
                    url=source.url,
                    extract_type=config.extract_type
                )
                
                # Follow links if depth allows
                if config.follow_links and config.current_depth < config.max_depth:
                    linked_content = await self._crawl_linked_pages(
                        content.links, 
                        config
                    )
                    content.linked_content = linked_content
                
                return CrawledContent(
                    source=source,
                    content=content,
                    crawl_timestamp=datetime.now(),
                    crawl_depth=config.current_depth
                )
                
        except Exception as e:
            logger.error(f"Crawl failed for {source.url}: {e}")
            return None
    
    async def _crawl_linked_pages(
        self, 
        links: List[str],
        config: CrawlConfig
    ) -> List[CrawledContent]:
        """Crawl linked pages up to configured depth"""
        
        # Filter relevant links
        relevant_links = self._filter_relevant_links(links, config.topic)
        
        # Limit concurrent crawls
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def crawl_with_limit(url):
            async with semaphore:
                new_config = config.copy()
                new_config.current_depth += 1
                return await self.crawl_source(
                    DiscoveredSource(url=url),
                    new_config
                )
        
        # Crawl in parallel
        tasks = [crawl_with_limit(url) for url in relevant_links[:config.max_links]]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
```

### Content Extraction

```python
class ContentExtractor:
    """
    Extract structured content from web pages
    """
    
    def __init__(self):
        self.readability = ReadabilityExtractor()
        self.schema_parser = SchemaParser()
        self.media_extractor = MediaExtractor()
    
    async def extract(
        self, 
        html: str, 
        url: str,
        extract_type: str
    ) -> ExtractedContent:
        """Extract content from HTML based on extraction type"""
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract based on type
        extractors = {
            "article": self._extract_article,
            "documentation": self._extract_documentation,
            "forum": self._extract_forum_post,
            "product": self._extract_product_info,
            "research": self._extract_research_paper,
            "general": self._extract_general
        }
        
        extractor = extractors.get(extract_type, self._extract_general)
        
        content = await extractor(soup, url)
        
        # Extract metadata
        content.metadata = self._extract_metadata(soup, url)
        
        # Extract media
        content.media = await self.media_extractor.extract(soup, url)
        
        return content
    
    async def _extract_article(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract article content using readability"""
        
        # Use Mozilla Readability algorithm
        article = self.readability.parse(soup, url)
        
        return ExtractedContent(
            title=article.title,
            author=article.byline,
            publish_date=article.publish_date,
            content=article.text,
            html_content=article.content,
            summary=article.excerpt,
            reading_time=article.reading_time,
            links=article.links,
            content_type="article"
        )
    
    async def _extract_research_paper(
        self, 
        soup: BeautifulSoup, 
        url: str
    ) -> ExtractedContent:
        """Extract research paper content"""
        
        # Look for academic paper structure
        content = ExtractedContent()
        
        # Extract abstract
        abstract = soup.find(['abstract', 'section#abstract', '.abstract'])
        if abstract:
            content.abstract = abstract.get_text(strip=True)
        
        # Extract sections
        sections = soup.find_all(['h1', 'h2', 'h3', 'section'])
        content.sections = []
        
        for section in sections:
            section_data = {
                'heading': section.get_text(strip=True),
                'content': self._get_section_content(section),
                'level': self._get_heading_level(section)
            }
            content.sections.append(section_data)
        
        # Extract citations
        content.citations = self._extract_citations(soup)
        
        # Extract figures and tables
        content.figures = self._extract_figures(soup)
        content.tables = self._extract_tables(soup)
        
        return content
```

### Source Quality Ranking

```python
class SourceRanker:
    """
    Rank sources by quality, relevance, and credibility
    """
    
    CREDIBILITY_DOMAINS = {
        # Academic
        "edu": 0.9,
        "ac.uk": 0.9,
        "arxiv.org": 0.95,
        "ieee.org": 0.95,
        "acm.org": 0.95,
        "nature.com": 0.95,
        "science.org": 0.95,
        
        # Government
        "gov": 0.9,
        "gov.uk": 0.9,
        "europa.eu": 0.9,
        
        # Tech
        "github.com": 0.85,
        "stackoverflow.com": 0.85,
        "docs.microsoft.com": 0.9,
        "developer.mozilla.org": 0.9,
        
        # News
        "reuters.com": 0.85,
        "apnews.com": 0.85,
        "bbc.com": 0.8,
        
        # Reference
        "wikipedia.org": 0.75,
        "britannica.com": 0.8
    }
    
    async def rank(
        self, 
        sources: List[DiscoveredSource],
        query: SearchQuery
    ) -> List[RankedSource]:
        """Rank sources by multiple quality factors"""
        
        ranked = []
        
        for source in sources:
            scores = {
                "credibility": self._score_credibility(source),
                "relevance": await self._score_relevance(source, query),
                "freshness": self._score_freshness(source),
                "authority": self._score_authority(source),
                "engagement": await self._score_engagement(source),
                "completeness": self._score_completeness(source)
            }
            
            # Calculate weighted overall score
            overall_score = self._calculate_overall_score(scores)
            
            ranked.append(RankedSource(
                source=source,
                scores=scores,
                overall_score=overall_score,
                confidence=self._calculate_confidence(scores)
            ))
        
        # Sort by overall score
        ranked.sort(key=lambda x: x.overall_score, reverse=True)
        
        return ranked
    
    def _score_credibility(self, source: DiscoveredSource) -> float:
        """Score source credibility based on domain and signals"""
        
        domain = extract_domain(source.url)
        
        # Check known credible domains
        for cred_domain, score in self.CREDIBILITY_DOMAINS.items():
            if cred_domain in domain:
                return score
        
        # Check for negative signals
        negative_signals = [
            "blogspot", "wordpress.com", "medium.com",  # User-generated
            "clickbait", "sponsored", "advertorial"     # Commercial
        ]
        
        for signal in negative_signals:
            if signal in domain.lower():
                return 0.4
        
        # Default score for unknown domains
        return 0.6
    
    async def _score_relevance(
        self, 
        source: DiscoveredSource,
        query: SearchQuery
    ) -> float:
        """Score relevance to the research query"""
        
        # Title relevance
        title_score = self._text_similarity(
            query.text.lower(),
            source.title.lower()
        )
        
        # Snippet relevance
        snippet_score = self._text_similarity(
            query.text.lower(),
            source.snippet.lower()
        ) if source.snippet else 0.5
        
        # Combined score
        return (title_score * 0.6) + (snippet_score * 0.4)
```

---

## Information Extraction & Parsing

### Structured Information Extraction

```python
class InformationExtractor:
    """
    Extract structured information from crawled content
    """
    
    def __init__(self):
        self.llm = GPT52Client()
        self.entity_extractor = EntityExtractor()
        self.fact_extractor = FactExtractor()
        self.relation_extractor = RelationExtractor()
    
    async def extract_information(
        self, 
        content: CrawledContent,
        extraction_goals: List[ExtractionGoal]
    ) -> ExtractedInformation:
        """Extract structured information from content"""
        
        extracted = ExtractedInformation(
            source=content.source,
            extraction_timestamp=datetime.now()
        )
        
        # Extract entities
        extracted.entities = await self.entity_extractor.extract(
            content.content.content
        )
        
        # Extract facts
        extracted.facts = await self.fact_extractor.extract(
            content.content.content,
            extraction_goals
        )
        
        # Extract relationships
        extracted.relationships = await self.relation_extractor.extract(
            content.content.content,
            extracted.entities
        )
        
        # Extract key claims
        extracted.claims = await self._extract_claims(content)
        
        # Extract statistics and data
        extracted.statistics = await self._extract_statistics(content)
        
        # Extract quotes
        extracted.quotes = await self._extract_quotes(content)
        
        return extracted
    
    async def _extract_claims(self, content: CrawledContent) -> List[Claim]:
        """Extract verifiable claims from content"""
        
        prompt = f"""
        Extract verifiable claims from the following text.
        A claim is a statement that can be fact-checked or verified.
        
        Text: {content.content.content[:8000]}
        
        For each claim:
        1. The exact claim text
        2. Claim type (fact, opinion, prediction, statistic)
        3. Confidence level (high, medium, low)
        4. Supporting evidence if present
        5. Contradicting evidence if present
        
        Return as structured JSON array.
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="high",
            response_format="json"
        )
        
        return [Claim.parse(c) for c in response["claims"]]
```

### Entity Extraction

```python
class EntityExtractor:
    """
    Named Entity Recognition and extraction
    """
    
    ENTITY_TYPES = [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
        "MONEY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "TECHNOLOGY", "CONCEPT"
    ]
    
    async def extract(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        
        # Use GPT-5.2 for entity extraction
        prompt = f"""
        Extract all named entities from the following text.
        
        Text: {text[:10000]}
        
        Entity types to extract:
        - PERSON: People, characters
        - ORGANIZATION: Companies, agencies, institutions
        - LOCATION: Countries, cities, geographic features
        - TECHNOLOGY: Software, hardware, technical terms
        - CONCEPT: Ideas, theories, methodologies
        - PRODUCT: Products, services
        - EVENT: Named events, conferences, incidents
        
        For each entity:
        1. Entity name
        2. Entity type
        3. Context (surrounding text)
        4. Confidence (0-1)
        
        Return as JSON array.
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="medium",
            response_format="json"
        )
        
        entities = []
        for e in response["entities"]:
            entity = Entity(
                name=e["name"],
                type=e["type"],
                context=e["context"],
                confidence=e["confidence"],
                mentions=self._count_mentions(text, e["name"])
            )
            entities.append(entity)
        
        return entities
```

### Fact Extraction

```python
class FactExtractor:
    """
    Extract factual information from content
    """
    
    async def extract(
        self, 
        text: str,
        goals: List[ExtractionGoal]
    ) -> List[Fact]:
        """Extract facts based on extraction goals"""
        
        facts = []
        
        for goal in goals:
            goal_facts = await self._extract_for_goal(text, goal)
            facts.extend(goal_facts)
        
        return facts
    
    async def _extract_for_goal(
        self, 
        text: str,
        goal: ExtractionGoal
    ) -> List[Fact]:
        """Extract facts for a specific goal"""
        
        prompt = f"""
        Extract factual information related to: {goal.topic}
        
        Text: {text[:8000]}
        
        Extraction focus: {goal.description}
        
        For each fact:
        1. The factual statement
        2. Category (definition, statistic, relationship, process, attribute)
        3. Confidence level
        4. Temporal context (when applicable)
        5. Source attribution within text
        
        Return as JSON array.
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="high",
            response_format="json"
        )
        
        return [Fact.parse(f) for f in response["facts"]]
```

---

## Synthesis & Summarization

### Multi-Source Synthesis

```python
class SynthesisEngine:
    """
    Synthesize information from multiple sources
    """
    
    def __init__(self):
        self.llm = GPT52Client()
        self.conflict_resolver = ConflictResolver()
        self.consolidation_engine = ConsolidationEngine()
    
    async def synthesize(
        self, 
        extracted_info: List[ExtractedInformation],
        research_task: ResearchTask
    ) -> SynthesisResult:
        """Synthesize information from multiple sources"""
        
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
        summaries = await self._create_summaries(synthesis)
        
        return SynthesisResult(
            synthesis=synthesis,
            summaries=summaries,
            consensus_areas=consensus,
            disagreement_areas=disagreements,
            confidence_score=self._calculate_confidence(resolved_facts),
            sources_used=len(extracted_info),
            synthesis_timestamp=datetime.now()
        )
    
    async def _generate_synthesis(
        self,
        facts: List[ResolvedFact],
        consensus: List[ConsensusArea],
        disagreements: List[DisagreementArea],
        task: ResearchTask
    ) -> Synthesis:
        """Generate comprehensive synthesis using GPT-5.2"""
        
        # Prepare context for synthesis
        context = self._prepare_synthesis_context(facts, consensus, disagreements)
        
        prompt = f"""
        Synthesize the following research findings into a comprehensive analysis.
        
        Research Topic: {task.topic}
        Research Goal: {task.rationale}
        
        Key Facts:
        {self._format_facts(facts[:50])}
        
        Areas of Consensus:
        {self._format_consensus(consensus)}
        
        Areas of Disagreement:
        {self._format_disagreements(disagreements)}
        
        Generate:
        1. Executive Summary (2-3 paragraphs)
        2. Key Findings (bullet points)
        3. Detailed Analysis (organized by theme)
        4. Implications and Conclusions
        5. Knowledge Gaps (what remains unknown)
        6. Recommendations for Further Research
        
        Maintain objectivity and clearly indicate confidence levels.
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="high",
            max_tokens=4000
        )
        
        return Synthesis(
            content=response,
            topic=task.topic,
            generated_at=datetime.now()
        )
```

### Conflict Resolution

```python
class ConflictResolver:
    """
    Resolve conflicts between information from different sources
    """
    
    async def resolve(
        self, 
        topic_facts: Dict[str, List[Fact]]
    ) -> List[ResolvedFact]:
        """Resolve conflicting information"""
        
        resolved = []
        
        for topic, facts in topic_facts.items():
            if len(facts) == 1:
                # No conflict
                resolved.append(ResolvedFact.from_single(facts[0]))
            else:
                # Check for conflicts
                conflicts = self._identify_conflicts(facts)
                
                if conflicts:
                    # Resolve conflicts
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
    
    async def _resolve_conflict_group(
        self,
        topic: str,
        facts: List[Fact],
        conflicts: List[Conflict]
    ) -> ResolvedFact:
        """Resolve a group of conflicting facts"""
        
        # Analyze conflict
        conflict_analysis = await self._analyze_conflict(facts, conflicts)
        
        # Resolution strategies
        if conflict_analysis["type"] == "temporal":
            # Time-based conflict - use most recent
            resolved = self._resolve_temporal_conflict(facts)
        
        elif conflict_analysis["type"] == "source_credibility":
            # Credibility-based resolution
            resolved = self._resolve_by_credibility(facts)
        
        elif conflict_analysis["type"] == "contextual":
            # Context-dependent - both may be true
            resolved = self._resolve_contextual_conflict(facts)
        
        else:
            # Use LLM to reason through conflict
            resolved = await self._llm_resolve_conflict(facts, conflict_analysis)
        
        return resolved
```

### Multi-Level Summarization

```python
class SummarizationEngine:
    """
    Create summaries at multiple detail levels
    """
    
    SUMMARY_LEVELS = {
        "ultra_brief": {
            "max_tokens": 50,
            "description": "One-sentence summary"
        },
        "brief": {
            "max_tokens": 150,
            "description": "2-3 sentence summary"
        },
        "standard": {
            "max_tokens": 300,
            "description": "Paragraph summary"
        },
        "detailed": {
            "max_tokens": 800,
            "description": "Multi-paragraph summary"
        },
        "comprehensive": {
            "max_tokens": 2000,
            "description": "Full summary with key points"
        }
    }
    
    async def create_summaries(
        self, 
        synthesis: Synthesis
    ) -> Dict[str, Summary]:
        """Create summaries at all levels"""
        
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
        config: Dict
    ) -> Summary:
        """Create a summary at a specific detail level"""
        
        prompt = f"""
        Create a {level} summary of the following research synthesis.
        
        {config["description"]}
        Maximum length: {config["max_tokens"]} tokens
        
        Synthesis:
        {synthesis.content}
        
        Requirements:
        - Maintain factual accuracy
        - Preserve key insights
        - Use clear, concise language
        - Include confidence indicators where relevant
        """
        
        response = await self.llm.generate(
            prompt=prompt,
            thinking_mode="medium",
            max_tokens=config["max_tokens"]
        )
        
        return Summary(
            level=level,
            content=response,
            token_count=len(response.split()),
            generated_at=datetime.now()
        )
```

---

## Knowledge Consolidation

### MEMORY.md Integration

```python
class MemoryWriter:
    """
    Write synthesized knowledge to MEMORY.md
    """
    
    MEMORY_STRUCTURE = {
        "sections": [
            "knowledge_base",      # Core knowledge organized by topic
            "research_history",    # Record of research activities
            "source_library",      # Tracked sources with metadata
            "knowledge_gaps",      # Identified gaps for future research
            "confidence_map",      # Confidence levels for stored knowledge
            "update_log"          # History of memory updates
        ]
    }
    
    def __init__(self, memory_path: str = "MEMORY.md"):
        self.memory_path = memory_path
        self.knowledge_graph = KnowledgeGraph()
        self.update_tracker = UpdateTracker()
    
    async def consolidate(
        self, 
        synthesis_result: SynthesisResult,
        research_task: ResearchTask
    ) -> ConsolidationResult:
        """Consolidate research findings into memory"""
        
        # Step 1: Check for existing knowledge
        existing = await self._find_existing_knowledge(synthesis_result)
        
        # Step 2: Merge or create new entries
        if existing:
            merged = await self._merge_with_existing(
                synthesis_result, 
                existing
            )
        else:
            merged = await self._create_new_entry(synthesis_result)
        
        # Step 3: Update knowledge graph
        await self._update_knowledge_graph(merged)
        
        # Step 4: Write to MEMORY.md
        await self._write_to_memory(merged, research_task)
        
        # Step 5: Update cross-references
        await self._update_cross_references(merged)
        
        return ConsolidationResult(
            entries_created=len(merged.new_entries),
            entries_updated=len(merged.updated_entries),
            cross_references_added=merged.cross_references,
            consolidation_timestamp=datetime.now()
        )
    
    async def _write_to_memory(
        self, 
        merged: MergedKnowledge,
        task: ResearchTask
    ):
        """Write knowledge to MEMORY.md file"""
        
        # Read existing memory
        existing_memory = await self._read_memory()
        
        # Generate memory entry
        entry = self._generate_memory_entry(merged, task)
        
        # Update appropriate section
        section = self._determine_section(merged)
        
        # Write updated memory
        updated_memory = self._update_section(
            existing_memory,
            section,
            entry
        )
        
        # Add to update log
        updated_memory = self._add_update_log(
            updated_memory,
            task,
            entry
        )
        
        # Write to file
        await self._write_memory_file(updated_memory)
    
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
        
        for fact in merged.key_facts:
            entry += f"- {fact.statement} (Confidence: {fact.confidence})\n"
        
        entry += "\n### Detailed Information\n"
        entry += merged.detailed_content
        
        entry += "\n### Source References\n"
        for source in merged.sources:
            entry += f"- [{source.title}]({source.url}) - {source.credibility_score}\n"
        
        entry += "\n### Related Topics\n"
        for related in merged.related_topics:
            entry += f"- [[{related}]]\n"
        
        entry += "\n---\n"
        
        return entry
```

### Knowledge Graph Integration

```python
class KnowledgeGraph:
    """
    Graph-based knowledge representation
    """
    
    def __init__(self, graph_path: str = "knowledge_graph.json"):
        self.graph_path = graph_path
        self.graph = self._load_graph()
    
    async def add_knowledge(
        self, 
        synthesis: SynthesisResult
    ) -> GraphUpdate:
        """Add synthesized knowledge to graph"""
        
        updates = GraphUpdate()
        
        # Add topic node
        topic_node = self._create_topic_node(synthesis)
        self.graph.add_node(topic_node)
        updates.nodes_added.append(topic_node.id)
        
        # Add entity nodes
        for entity in synthesis.entities:
            entity_node = self._create_entity_node(entity)
            if not self.graph.has_node(entity_node.id):
                self.graph.add_node(entity_node)
                updates.nodes_added.append(entity_node.id)
            
            # Link to topic
            edge = self._create_edge(topic_node.id, entity_node.id, "contains")
            self.graph.add_edge(edge)
            updates.edges_added.append(edge.id)
        
        # Add fact nodes
        for fact in synthesis.facts:
            fact_node = self._create_fact_node(fact)
            self.graph.add_node(fact_node)
            updates.nodes_added.append(fact_node.id)
            
            # Link to topic
            edge = self._create_edge(topic_node.id, fact_node.id, "has_fact")
            self.graph.add_edge(edge)
            updates.edges_added.append(edge.id)
        
        # Create relationships
        for relationship in synthesis.relationships:
            rel_edge = self._create_relationship_edge(relationship)
            self.graph.add_edge(rel_edge)
            updates.edges_added.append(rel_edge.id)
        
        # Save graph
        await self._save_graph()
        
        return updates
    
    def find_incomplete_nodes(self) -> List[Node]:
        """Find nodes with incomplete information"""
        
        incomplete = []
        
        for node in self.graph.nodes:
            # Check for missing attributes
            missing = self._get_missing_attributes(node)
            
            # Check for low connection count
            connection_count = len(self.graph.get_edges(node.id))
            
            # Check for old information
            age = datetime.now() - node.last_updated
            
            if missing or connection_count < 3 or age.days > 30:
                incomplete.append(node)
        
        return incomplete
```

---

## Source Tracking & Citations

### Citation Management

```python
class CitationTracker:
    """
    Track and manage source citations
    """
    
    def __init__(self, citations_path: str = "citations.json"):
        self.citations_path = citations_path
        self.citations_db = self._load_citations()
    
    async def track_source(
        self, 
        source: DiscoveredSource,
        usage_context: UsageContext
    ) -> Citation:
        """Track a source usage"""
        
        # Generate citation ID
        citation_id = self._generate_citation_id(source)
        
        # Create citation record
        citation = Citation(
            id=citation_id,
            source_url=source.url,
            source_title=source.title,
            access_date=datetime.now(),
            usage_context=usage_context,
            reliability_score=source.credibility_score,
            citation_format=self._generate_formats(source)
        )
        
        # Store citation
        self.citations_db[citation_id] = citation
        await self._save_citations()
        
        return citation
    
    def _generate_formats(self, source: DiscoveredSource) -> Dict[str, str]:
        """Generate citation in multiple formats"""
        
        formats = {}
        
        # APA format
        formats["apa"] = self._apa_format(source)
        
        # MLA format
        formats["mla"] = self._mla_format(source)
        
        # Chicago format
        formats["chicago"] = self._chicago_format(source)
        
        # IEEE format
        formats["ieee"] = self._ieee_format(source)
        
        # Simple format
        formats["simple"] = f"{source.title} - {source.url}"
        
        return formats
    
    def _apa_format(self, source: DiscoveredSource) -> str:
        """Generate APA citation"""
        
        if source.content_type == "article":
            return f"{source.author if source.author else 'Unknown'}. " \
                   f"({source.publish_date.year if source.publish_date else 'n.d.'}). " \
                   f"{source.title}. Retrieved {datetime.now().strftime('%B %d, %Y')}, " \
                   f"from {source.url}"
        
        return f"{source.title}. (n.d.). Retrieved {datetime.now().strftime('%B %d, %Y')}, from {source.url}"
    
    async def verify_citation(self, citation_id: str) -> VerificationResult:
        """Verify if a citation is still valid"""
        
        citation = self.citations_db.get(citation_id)
        
        if not citation:
            return VerificationResult(
                valid=False,
                reason="Citation not found"
            )
        
        # Check if URL is still accessible
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(citation.source_url, timeout=10) as response:
                    if response.status == 200:
                        return VerificationResult(
                            valid=True,
                            last_verified=datetime.now()
                        )
                    else:
                        return VerificationResult(
                            valid=False,
                            reason=f"HTTP {response.status}",
                            last_verified=datetime.now()
                        )
        except Exception as e:
            return VerificationResult(
                valid=False,
                reason=str(e),
                last_verified=datetime.now()
            )
```

### Source Reliability Tracking

```python
class SourceReliabilityTracker:
    """
    Track and score source reliability over time
    """
    
    def __init__(self):
        self.reliability_db = ReliabilityDatabase()
    
    async def update_reliability(
        self, 
        source: DiscoveredSource,
        verification_result: VerificationResult
    ):
        """Update source reliability based on verification"""
        
        domain = extract_domain(source.url)
        
        # Get current reliability
        current = await self.reliability_db.get(domain)
        
        # Calculate new reliability score
        if verification_result.valid:
            # Increase reliability
            new_score = min(1.0, current.score + 0.05)
            current.successful_verifications += 1
        else:
            # Decrease reliability
            new_score = max(0.0, current.score - 0.1)
            current.failed_verifications += 1
        
        current.score = new_score
        current.last_verified = datetime.now()
        current.total_verifications += 1
        
        # Update database
        await self.reliability_db.update(domain, current)
    
    async def get_reliability_score(self, url: str) -> float:
        """Get reliability score for a URL"""
        
        domain = extract_domain(url)
        reliability = await self.reliability_db.get(domain)
        
        if not reliability:
            # Unknown domain - return neutral score
            return 0.5
        
        return reliability.score
```

---

## Research Depth & Breadth Control

### Depth Configuration

```python
class DepthController:
    """
    Control research depth and breadth
    """
    
    DEPTH_LEVELS = {
        "surface": {
            "max_sources": 3,
            "max_depth": 1,
            "max_pages_per_source": 1,
            "synthesis_detail": "brief",
            "time_budget_seconds": 60,
            "query_variants": 2
        },
        "shallow": {
            "max_sources": 5,
            "max_depth": 1,
            "max_pages_per_source": 2,
            "synthesis_detail": "standard",
            "time_budget_seconds": 180,
            "query_variants": 3
        },
        "medium": {
            "max_sources": 10,
            "max_depth": 2,
            "max_pages_per_source": 3,
            "synthesis_detail": "detailed",
            "time_budget_seconds": 600,
            "query_variants": 5
        },
        "deep": {
            "max_sources": 20,
            "max_depth": 3,
            "max_pages_per_source": 5,
            "synthesis_detail": "comprehensive",
            "time_budget_seconds": 1800,
            "query_variants": 8
        },
        "exhaustive": {
            "max_sources": 50,
            "max_depth": 5,
            "max_pages_per_source": 10,
            "synthesis_detail": "comprehensive",
            "time_budget_seconds": 3600,
            "query_variants": 15
        }
    }
    
    def __init__(self):
        self.active_research = {}
        self.resource_monitor = ResourceMonitor()
    
    async def configure_research(
        self, 
        task: ResearchTask
    ) -> ResearchConfig:
        """Configure research parameters based on depth level"""
        
        depth_config = self.DEPTH_LEVELS.get(
            task.depth, 
            self.DEPTH_LEVELS["medium"]
        )
        
        # Adjust based on available resources
        resources = await self.resource_monitor.check_resources()
        
        if resources.available_tokens < 50000:
            # Reduce depth if low on tokens
            depth_config = self._reduce_depth(depth_config)
        
        if resources.rate_limit_approaching:
            # Reduce sources if rate limited
            depth_config["max_sources"] = max(3, depth_config["max_sources"] // 2)
        
        # Create research config
        config = ResearchConfig(
            depth=task.depth,
            max_sources=depth_config["max_sources"],
            max_depth=depth_config["max_depth"],
            max_pages_per_source=depth_config["max_pages_per_source"],
            synthesis_detail=depth_config["synthesis_detail"],
            time_budget=timedelta(seconds=depth_config["time_budget_seconds"]),
            query_variants=depth_config["query_variants"],
            start_time=datetime.now()
        )
        
        return config
    
    async def monitor_progress(
        self, 
        research_id: str
    ) -> ProgressStatus:
        """Monitor research progress and adjust if needed"""
        
        progress = self.active_research.get(research_id)
        
        if not progress:
            return ProgressStatus.not_found()
        
        # Check time budget
        elapsed = datetime.now() - progress.start_time
        config = progress.config
        
        if elapsed > config.time_budget:
            # Time budget exceeded - wrap up
            return ProgressStatus(
                should_continue=False,
                reason="Time budget exceeded",
                progress_percent=progress.percent_complete
            )
        
        # Check if we have enough information
        if progress.sources_collected >= config.max_sources:
            return ProgressStatus(
                should_continue=False,
                reason="Source limit reached",
                progress_percent=100
            )
        
        # Check quality threshold
        if progress.average_source_quality > 0.8 and progress.sources_collected >= 5:
            # High quality sources found - can stop early
            return ProgressStatus(
                should_continue=False,
                reason="Quality threshold met",
                progress_percent=progress.percent_complete
            )
        
        return ProgressStatus(
            should_continue=True,
            progress_percent=progress.percent_complete
        )
```

### Adaptive Research

```python
class AdaptiveResearchController:
    """
    Adaptively control research based on findings
    """
    
    async def adapt_research(
        self, 
        current_findings: List[ExtractedInformation],
        config: ResearchConfig
    ) -> AdaptationDecision:
        """Adapt research strategy based on current findings"""
        
        decision = AdaptationDecision()
        
        # Analyze current findings
        analysis = await self._analyze_findings(current_findings)
        
        # Decide on adaptations
        if analysis["information_density"] < 0.3:
            # Low information density - broaden search
            decision.should_broaden = True
            decision.new_queries = await self._generate_broader_queries(
                current_findings
            )
        
        if analysis["has_conflicts"]:
            # Conflicts found - deepen research
            decision.should_deepen = True
            decision.conflict_resolution_needed = True
        
        if analysis["knowledge_gaps"]:
            # Gaps identified - targeted research
            decision.additional_topics = analysis["knowledge_gaps"]
        
        if analysis["high_quality_sources"]:
            # Found high quality sources - explore related
            decision.explore_related = True
            decision.related_topics = await self._find_related_topics(
                analysis["high_quality_sources"]
            )
        
        return decision
    
    async def _analyze_findings(
        self, 
        findings: List[ExtractedInformation]
    ) -> Dict:
        """Analyze findings for adaptation decisions"""
        
        analysis = {
            "information_density": 0,
            "has_conflicts": False,
            "knowledge_gaps": [],
            "high_quality_sources": []
        }
        
        total_facts = sum(len(f.facts) for f in findings)
        total_content = sum(len(f.source.content) for f in findings)
        
        if total_content > 0:
            analysis["information_density"] = total_facts / total_content
        
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
            if f.source.overall_score > 0.8
        ]
        
        return analysis
```

---

## Implementation Specifications

### Core Data Models

```python
# Pydantic models for type safety

class ResearchTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: str  # scheduled, event, curiosity
    trigger_source: str
    topic: str
    context: Optional[str] = None
    rationale: Optional[str] = None
    depth: str = "medium"  # surface, shallow, medium, deep, exhaustive
    sources: List[str] = Field(default_factory=list)
    priority: float = 0.5
    immediate: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed

class SearchQuery(BaseModel):
    text: str
    target_engines: List[str]
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_type: str  # definition, how_to, comparison, etc.
    priority: float = 0.5
    generated_at: datetime = Field(default_factory=datetime.now)

class DiscoveredSource(BaseModel):
    url: str
    title: str
    snippet: Optional[str] = None
    engine: str
    query: str
    rank: int
    timestamp: datetime
    content_type: Optional[str] = None
    credibility_score: float = 0.5

class ExtractedContent(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    content: str
    html_content: Optional[str] = None
    summary: Optional[str] = None
    reading_time: Optional[int] = None
    links: List[str] = Field(default_factory=list)
    content_type: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    media: List[Media] = Field(default_factory=list)

class ExtractedInformation(BaseModel):
    source: DiscoveredSource
    entities: List[Entity] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    statistics: List[Statistic] = Field(default_factory=list)
    quotes: List[Quote] = Field(default_factory=list)
    extraction_timestamp: datetime

class SynthesisResult(BaseModel):
    synthesis: Synthesis
    summaries: Dict[str, Summary]
    consensus_areas: List[ConsensusArea]
    disagreement_areas: List[DisagreementArea]
    confidence_score: float
    sources_used: int
    synthesis_timestamp: datetime

class ResearchConfig(BaseModel):
    depth: str
    max_sources: int
    max_depth: int
    max_pages_per_source: int
    synthesis_detail: str
    time_budget: timedelta
    query_variants: int
    start_time: datetime
    results_per_engine: int = 10
```

### Main Research Loop

```python
class ResearchLoop:
    """
    Main research loop orchestrator
    """
    
    def __init__(self, config: LoopConfig):
        self.config = config
        self.trigger_engine = TriggerEngine()
        self.query_generator = QueryGenerator()
        self.source_discovery = SourceDiscoveryEngine()
        self.information_extractor = InformationExtractor()
        self.synthesis_engine = SynthesisEngine()
        self.memory_writer = MemoryWriter()
        self.citation_tracker = CitationTracker()
        self.depth_controller = DepthController()
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
    
    async def run(self):
        """Main research loop"""
        
        logger.info("Research Loop started")
        
        while True:
            try:
                # Check for triggers
                triggers = await self.trigger_engine.check_all_triggers()
                
                for trigger in triggers:
                    await self.task_queue.put(trigger)
                
                # Process tasks
                while not self.task_queue.empty():
                    task = await self.task_queue.get()
                    await self._execute_research(task)
                
                # Heartbeat
                await self._send_heartbeat()
                
                # Sleep before next iteration
                await asyncio.sleep(self.config.poll_interval)
                
            except Exception as e:
                logger.error(f"Research Loop error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_research(self, task: ResearchTask):
        """Execute a complete research task"""
        
        logger.info(f"Starting research task: {task.id} - {task.topic}")
        
        try:
            # Update status
            task.status = "in_progress"
            self.active_tasks[task.id] = task
            
            # 1. Configure research depth
            research_config = await self.depth_controller.configure_research(task)
            
            # 2. Generate queries
            queries = await self.query_generator.generate_queries(
                task, 
                ResearchContext()
            )
            
            # 3. Discover sources
            sources = await self.source_discovery.discover_sources(
                queries, 
                research_config
            )
            
            # 4. Crawl and extract
            crawled_content = []
            for source in sources[:research_config.max_sources]:
                content = await self.source_discovery.crawler.crawl_source(
                    source,
                    CrawlConfig.from_research_config(research_config)
                )
                if content:
                    crawled_content.append(content)
            
            # 5. Extract information
            extraction_goals = self._derive_extraction_goals(task)
            extracted_info = []
            
            for content in crawled_content:
                info = await self.information_extractor.extract_information(
                    content,
                    extraction_goals
                )
                extracted_info.append(info)
                
                # Track citation
                await self.citation_tracker.track_source(
                    content.source,
                    UsageContext(research_task=task.id)
                )
            
            # 6. Synthesize
            synthesis = await self.synthesis_engine.synthesize(
                extracted_info,
                task
            )
            
            # 7. Consolidate to memory
            consolidation = await self.memory_writer.consolidate(
                synthesis,
                task
            )
            
            # 8. Update task status
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # 9. Emit completion event
            await self._emit_completion_event(task, synthesis, consolidation)
            
            logger.info(f"Research task completed: {task.id}")
            
        except Exception as e:
            logger.error(f"Research task failed: {task.id} - {e}")
            task.status = "failed"
            task.error = str(e)
            await self._emit_failure_event(task, e)
        
        finally:
            del self.active_tasks[task.id]
```

---

## Integration Points

### Loop Interconnections

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOOP INTERCONNECTIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RESEARCH LOOP ◄────────────────────────────────────┐           │
│       │                                             │           │
│       │ Triggers:                                   │           │
│       │   • MEMORY LOOP (knowledge gaps)            │           │
│       │   • CONVERSATION LOOP (unknown topics)      │           │
│       │   • PLANNING LOOP (research needs)          │           │
│       │   • CURIOSITY LOOP (interest areas)         │           │
│       │                                             │           │
│       │ Outputs:                                    │           │
│       │   • MEMORY.md updates                       │           │
│       │   • KNOWLEDGE GRAPH updates                 │           │
│       │   • Citation database                       │           │
│       │   • Event notifications                     │           │
│       │                                             │           │
│       ▼                                             │           │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────┐│           │
│  │ MEMORY LOOP │◄──►│  KNOWLEDGE  │◄──►│  USER    ││           │
│  │             │    │   GRAPH     │    │  SYSTEM  ││           │
│  └─────────────┘    └─────────────┘    └──────────┘│           │
│       ▲                                             │           │
│       │                                             │           │
│       └─────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### External API Integrations

| Service | Purpose | Rate Limits |
|---------|---------|-------------|
| Google Search | General web search | 100 queries/day |
| Bing Search | Backup web search | 1000 queries/month |
| DuckDuckGo | Privacy-focused search | Unlimited |
| Google Scholar | Academic research | 100 queries/day |
| ArXiv | Scientific papers | Unlimited |
| Reddit API | Community discussions | 60 requests/min |
| Twitter API | Real-time trends | 450 requests/15min |
| GitHub API | Technical documentation | 5000 requests/hour |

### Windows 10 Integration

```python
class WindowsIntegration:
    """
    Windows 10 specific integrations
    """
    
    async def schedule_research_task(self, task: ResearchTask):
        """Schedule research using Windows Task Scheduler"""
        
        # Create scheduled task
        subprocess.run([
            "schtasks", "/create",
            "/tn", f"OpenClaw_Research_{task.id}",
            "/tr", f"python research_runner.py {task.id}",
            "/sc", self._convert_to_windows_schedule(task.schedule),
            "/f"
        ])
    
    async def send_notification(self, result: ResearchResult):
        """Send Windows notification for research completion"""
        
        from win10toast import ToastNotifier
        
        toaster = ToastNotifier()
        toaster.show_toast(
            "OpenClaw Research Complete",
            f"Research on '{result.topic}' completed. "
            f"Found {result.facts_count} facts from {result.sources_count} sources.",
            duration=10
        )
```

---

## Configuration

```yaml
# research_loop_config.yaml

research_loop:
  enabled: true
  poll_interval: 30  # seconds
  
  triggers:
    scheduled:
      enabled: true
      schedules:
        morning_briefing:
          cron: "0 8 * * *"
          topics: ["tech_news", "market_updates"]
          depth: "shallow"
        
        weekly_deep_dive:
          cron: "0 10 * * 1"
          topics: ["emerging_tech"]
          depth: "deep"
    
    event_driven:
      enabled: true
      immediate_events:
        - unknown_topic
        - user_request
        - contradiction_detected
      
      deferred_events:
        - outdated_info
        - memory_gap
    
    curiosity:
      enabled: true
      min_curiosity_score: 0.7
      max_daily_tasks: 10
  
  search:
    engines:
      - google
      - bing
      - duckduckgo
    
    results_per_engine: 10
    max_query_variants: 5
    
  crawling:
    max_concurrent: 5
    timeout_seconds: 30
    respect_robots_txt: true
    user_agent: "OpenClaw Research Bot 1.0"
  
  synthesis:
    model: "gpt-5.2"
    thinking_mode: "high"
    summary_levels:
      - ultra_brief
      - brief
      - standard
      - detailed
      - comprehensive
  
  memory:
    memory_file: "MEMORY.md"
    knowledge_graph: "knowledge_graph.json"
    citation_db: "citations.json"
    auto_backup: true
    backup_interval: "daily"
  
  depth_levels:
    surface:
      max_sources: 3
      time_budget: 60
    
    shallow:
      max_sources: 5
      time_budget: 180
    
    medium:
      max_sources: 10
      time_budget: 600
    
    deep:
      max_sources: 20
      time_budget: 1800
    
    exhaustive:
      max_sources: 50
      time_budget: 3600
```

---

## Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Generation Time | < 2s | Average per query |
| Source Discovery Time | < 10s | For 10 sources |
| Content Extraction | < 5s | Per page |
| Synthesis Time | < 30s | For 10 sources |
| Memory Write Time | < 2s | Per entry |
| Research Completion | Per depth config | End-to-end |
| Source Quality Score | > 0.7 | Average credibility |
| Information Density | > 0.5 | Facts per content unit |
| Citation Accuracy | > 95% | Verified citations |

---

## Error Handling & Recovery

```python
class ResearchErrorHandler:
    """
    Handle and recover from research errors
    """
    
    ERROR_STRATEGIES = {
        "search_failed": {
            "retry": True,
            "max_retries": 3,
            "fallback": "try_alternate_engine"
        },
        "crawl_failed": {
            "retry": True,
            "max_retries": 2,
            "fallback": "skip_source"
        },
        "extraction_failed": {
            "retry": False,
            "fallback": "use_alternate_extractor"
        },
        "synthesis_failed": {
            "retry": True,
            "max_retries": 1,
            "fallback": "reduce_sources_and_retry"
        },
        "memory_write_failed": {
            "retry": True,
            "max_retries": 3,
            "fallback": "queue_for_retry"
        }
    }
    
    async def handle_error(
        self, 
        error: Exception,
        context: ErrorContext
    ) -> ErrorRecovery:
        """Handle research error and attempt recovery"""
        
        error_type = self._classify_error(error)
        strategy = self.ERROR_STRATEGIES.get(error_type)
        
        if not strategy:
            # Unknown error - log and fail
            logger.error(f"Unhandled error: {error}")
            return ErrorRecovery(should_fail=True)
        
        # Attempt recovery
        if strategy["retry"] and context.retry_count < strategy["max_retries"]:
            return ErrorRecovery(
                should_retry=True,
                retry_delay=self._calculate_backoff(context.retry_count)
            )
        
        # Use fallback
        if strategy["fallback"]:
            fallback_result = await self._execute_fallback(
                strategy["fallback"],
                context
            )
            return ErrorRecovery(
                should_retry=False,
                fallback_result=fallback_result
            )
        
        return ErrorRecovery(should_fail=True)
```

---

## Security & Privacy

```python
class ResearchSecurity:
    """
    Security and privacy controls for research
    """
    
    BLOCKED_DOMAINS = [
        "malware", "phishing", "spam",
        "adult", "gambling", "illegal"
    ]
    
    PRIVACY_SETTINGS = {
        "respect_robots_txt": True,
        "respect_noindex": True,
        "anonymize_requests": True,
        "no_personal_data_extraction": True,
        "secure_storage": True
    }
    
    async def check_url_safety(self, url: str) -> SafetyCheck:
        """Check if URL is safe to crawl"""
        
        # Check blocked domains
        domain = extract_domain(url)
        for blocked in self.BLOCKED_DOMAINS:
            if blocked in domain:
                return SafetyCheck(safe=False, reason=f"Blocked domain: {blocked}")
        
        # Check URL reputation
        reputation = await self._check_reputation(url)
        if reputation.score < 0.5:
            return SafetyCheck(safe=False, reason="Low reputation score")
        
        return SafetyCheck(safe=True)
```

---

## Conclusion

The Research Loop provides a comprehensive autonomous information gathering and synthesis system for the OpenClaw-inspired AI agent framework. Key features include:

1. **Multi-Modal Triggers**: Scheduled, event-driven, and curiosity-based research initiation
2. **Intelligent Query Generation**: GPT-5.2 powered query optimization
3. **Multi-Source Discovery**: Parallel search across multiple engines
4. **Structured Extraction**: Entity, fact, and relationship extraction
5. **Conflict Resolution**: Automated handling of contradictory information
6. **Persistent Memory**: Automatic consolidation into MEMORY.md
7. **Source Tracking**: Comprehensive citation and reliability tracking
8. **Adaptive Control**: Dynamic depth and breadth adjustment

This specification enables the AI agent to continuously learn, verify, and synthesize information from the web while maintaining transparency through proper source tracking and citation.
