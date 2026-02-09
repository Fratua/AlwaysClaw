"""
Discovery Loop - Architecture Implementation
Windows 10 OpenClaw AI Agent Framework

This module provides the complete implementation of the Discovery Loop
for autonomous exploration and mapping.
"""

import asyncio
import json
import logging
import sqlite3
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Discovery:
    """Represents a discovered concept or topic"""
    topic: str
    content: str = ""
    confidence: float = 0.0
    novelty_score: float = 0.0
    relevance_score: float = 0.0
    depth: int = 0
    parent: Optional[str] = None
    strategy: str = "unknown"
    sources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ExplorationResult:
    """Result of an exploration operation"""
    discoveries: List[Discovery]
    strategy: str
    path: List[str] = field(default_factory=list)
    coverage: float = 0.0
    gaps_filled: int = 0
    strategy_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class NoveltyScore:
    """Novelty score breakdown"""
    overall: float
    semantic: float
    structural: float
    temporal: float
    source: float
    is_novel: bool = False

@dataclass
class KnowledgeGap:
    """Identified knowledge gap"""
    type: str
    description: str
    priority: float
    category: Optional[str] = None
    node_id: Optional[str] = None
    current_depth: int = 0
    recommended_depth: int = 0
    age_days: int = 0
    current_coverage: float = 0.0
    estimated_value: float = 0.0
    bridge_candidates: List[str] = field(default_factory=list)

@dataclass
class TerritoryStats:
    """Statistics about knowledge territory"""
    total_nodes: int = 0
    total_edges: int = 0
    density: float = 0.0
    avg_clustering: float = 0.0
    connected_components: int = 0
    max_depth: int = 0
    avg_confidence: float = 0.0
    recently_discovered: int = 0
    category_distribution: Counter = field(default_factory=Counter)

@dataclass
class DiscoveryStats:
    """Discovery statistics for a time period"""
    total_discoveries: int = 0
    average_novelty: float = 0.0
    average_relevance: float = 0.0
    strategies_used: int = 0
    unique_topics: int = 0

@dataclass
class IntegrationResult:
    """Result of concept integration"""
    success: bool
    node_ids: List[str] = field(default_factory=list)
    concepts_added: int = 0
    relationships_added: int = 0
    error: str = ""
    stage: str = ""

@dataclass
class BalanceDecision:
    """Decision from exploration-exploitation balancer"""
    action: str  # "explore" or "exploit"
    strategy: str
    exploration_rate: float

@dataclass
class DiscoveryStatus:
    """Current status of discovery loop"""
    running: bool
    territory_size: int
    queue_size: int
    exploration_rate: float
    recent_discoveries: int
    active_strategies: List[str]
    next_exploration: Optional[datetime]

# ============================================================================
# EXPLORATION STRATEGIES
# ============================================================================

class ExplorationStrategy(Enum):
    """Available exploration strategies"""
    BREADTH_FIRST = "bfs"
    DEPTH_FIRST = "dfs"
    INTEREST_DRIVEN = "interest"
    SERENDIPITY = "random"
    GAP_DRIVEN = "gap"
    TREND_DRIVEN = "trend"
    HYBRID = "hybrid"

class BreadthFirstExplorer:
    """BFS Strategy for systematic territory coverage"""
    
    CONFIG = {
        "max_depth": 5,
        "max_nodes": 100,
        "branching_factor": 10,
        "parallel_workers": 3,
        "exploration_timeout": 300,
        "territory_cache_ttl": 3600,
        "novelty_threshold": 0.5
    }
    
    def __init__(self, novelty_detector=None):
        self.novelty_detector = novelty_detector
        self.max_depth = self.CONFIG["max_depth"]
        self.max_nodes = self.CONFIG["max_nodes"]
        
    async def explore(self, seed_topics: List[str]) -> ExplorationResult:
        """Execute BFS exploration from seed topics"""
        visited = set()
        frontier = deque([(topic, 0) for topic in seed_topics])
        discoveries = []
        
        while frontier and len(discoveries) < self.max_nodes:
            current_topic, depth = frontier.popleft()
            
            if current_topic in visited or depth > self.max_depth:
                continue
                
            visited.add(current_topic)
            
            # Discover neighbors (simulated)
            neighbors = await self._discover_neighbors(current_topic)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    novelty_score = 0.7  # Simulated
                    if self.novelty_detector:
                        novelty = await self.novelty_detector.score(neighbor)
                        novelty_score = novelty.overall
                        
                    if novelty_score > self.CONFIG["novelty_threshold"]:
                        frontier.append((neighbor, depth + 1))
                        discoveries.append(Discovery(
                            topic=neighbor,
                            confidence=0.8,
                            novelty_score=novelty_score,
                            relevance_score=0.7,
                            depth=depth + 1,
                            parent=current_topic,
                            strategy="bfs"
                        ))
        
        return ExplorationResult(
            discoveries=discoveries,
            strategy="bfs"
        )
    
    async def _discover_neighbors(self, topic: str) -> List[str]:
        """Discover related sub-topics using LLM"""
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            n = self.CONFIG["branching_factor"]
            response = client.generate(
                f"List exactly {n} related sub-topics for '{topic}'. "
                f"Return one sub-topic per line, no numbering or bullets.",
                max_tokens=200,
            )
            lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
            return lines[:n] if lines else [f"{topic}_subtopic_{i}" for i in range(n)]
        except (ImportError, RuntimeError, EnvironmentError):
            return [f"{topic}_subtopic_{i}" for i in range(self.CONFIG["branching_factor"])]

class DepthFirstExplorer:
    """DFS Strategy for deep territory investigation"""
    
    CONFIG = {
        "max_depth": 10,
        "min_relevance": 0.7,
        "backtrack_threshold": 0.3,
        "exploration_timeout": 600
    }
    
    def __init__(self, novelty_detector=None):
        self.novelty_detector = novelty_detector
        
    async def explore(self, start_topic: str, target_depth: int = None) -> ExplorationResult:
        """Execute DFS exploration from start topic"""
        path = [start_topic]
        discoveries = []
        current_depth = 0
        max_depth = target_depth or self.CONFIG["max_depth"]
        
        while current_depth < max_depth:
            children = await self._discover_children(path[-1])
            
            if not children:
                if len(path) > 1:
                    path.pop()
                    current_depth -= 1
                    continue
                else:
                    break
            
            # Score children by TF-IDF similarity to the exploration path
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                texts = [start_topic] + children
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform(texts)
                sims = cos_sim(tfidf[0:1], tfidf[1:])[0]
                scored_children = list(zip(children, sims.tolist()))
            except ImportError:
                scored_children = [(child, random.uniform(0.5, 1.0)) for child in children]
            scored_children.sort(key=lambda x: x[1], reverse=True)
            
            best_child, best_score = scored_children[0]
            
            if best_score < self.CONFIG["backtrack_threshold"]:
                if len(path) > 1:
                    path.pop()
                    current_depth -= 1
                    continue
                else:
                    break
            
            path.append(best_child)
            current_depth += 1
            
            novelty_score = 0.6
            if self.novelty_detector:
                novelty = await self.novelty_detector.score(best_child)
                novelty_score = novelty.overall
            
            discoveries.append(Discovery(
                topic=best_child,
                confidence=0.75,
                novelty_score=novelty_score,
                relevance_score=best_score,
                depth=current_depth,
                parent=path[-2] if len(path) > 1 else None,
                strategy="dfs"
            ))
        
        return ExplorationResult(
            discoveries=discoveries,
            strategy="dfs",
            path=path
        )
    
    async def _discover_children(self, topic: str) -> List[str]:
        """Discover child sub-topics using LLM"""
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            response = client.generate(
                f"List 5 narrower sub-topics under '{topic}'. "
                f"One per line, no numbering.",
                max_tokens=150,
            )
            lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
            return lines[:5] if lines else [f"{topic}_child_{i}" for i in range(5)]
        except (ImportError, RuntimeError, EnvironmentError):
            return [f"{topic}_child_{i}" for i in range(5)]

class InterestDrivenExplorer:
    """Exploration guided by user interests"""
    
    CONFIG = {
        "interest_decay": 0.95,
        "discovery_bonus": 1.2,
        "max_candidates": 10,
        "min_score": 0.5
    }
    
    def __init__(self, user_interests: Dict[str, float], novelty_detector=None):
        self.user_interests = user_interests
        self.novelty_detector = novelty_detector
        self.discovered_topics = set()
        
    async def explore(self, frontier: List[str]) -> ExplorationResult:
        """Execute interest-driven exploration"""
        scored_frontier = []
        
        for node in frontier:
            interest_score = self._score_interest(node)
            novelty_score = 0.7
            
            if self.novelty_detector:
                novelty = await self.novelty_detector.score(node)
                novelty_score = novelty.overall
            
            combined_score = interest_score * novelty_score
            scored_frontier.append((node, combined_score))
        
        scored_frontier.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_frontier[:self.CONFIG["max_candidates"]]
        
        discoveries = []
        for node, score in candidates:
            if score < self.CONFIG["min_score"]:
                continue
            
            self.discovered_topics.add(node)
            discoveries.append(Discovery(
                topic=node,
                confidence=0.8,
                novelty_score=0.7,
                relevance_score=score,
                strategy="interest_driven"
            ))
        
        return ExplorationResult(
            discoveries=discoveries,
            strategy="interest_driven"
        )
    
    def _score_interest(self, topic: str) -> float:
        """Calculate interest score for topic"""
        base_score = 0.5
        for interest, weight in self.user_interests.items():
            if interest.lower() in topic.lower():
                base_score = max(base_score, weight)
        
        if topic not in self.discovered_topics:
            base_score *= self.CONFIG["discovery_bonus"]
        
        return min(base_score, 1.0)

class HybridExplorer:
    """Adaptive strategy combining multiple approaches"""
    
    STRATEGY_WEIGHTS = {
        "bfs": 0.25,
        "dfs": 0.25,
        "interest": 0.30,
        "gap": 0.20
    }
    
    def __init__(self, novelty_detector=None):
        self.bfs_explorer = BreadthFirstExplorer(novelty_detector)
        self.dfs_explorer = DepthFirstExplorer(novelty_detector)
        self.novelty_detector = novelty_detector
        
    async def explore(self, seed_topics: List[str], context: Dict = None) -> ExplorationResult:
        """Dynamically select and combine strategies"""
        # Adjust weights based on context
        weights = self.STRATEGY_WEIGHTS.copy()
        
        if context:
            coverage = context.get("coverage", 0.5)
            if coverage < 0.3:
                weights["bfs"] += 0.2
                weights["dfs"] -= 0.1
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Execute strategies
        all_discoveries = []
        
        if weights["bfs"] > 0.1:
            bfs_result = await self.bfs_explorer.explore(seed_topics)
            all_discoveries.extend(bfs_result.discoveries)
        
        if weights["dfs"] > 0.1 and seed_topics:
            dfs_result = await self.dfs_explorer.explore(seed_topics[0])
            all_discoveries.extend(dfs_result.discoveries)
        
        # Deduplicate
        seen = set()
        unique_discoveries = []
        for d in all_discoveries:
            if d.topic not in seen:
                seen.add(d.topic)
                unique_discoveries.append(d)
        
        return ExplorationResult(
            discoveries=unique_discoveries,
            strategy="hybrid",
            strategy_weights=weights
        )

# ============================================================================
# TERRITORY MAPPING
# ============================================================================

class KnowledgeTerritoryGraph:
    """Graph-based representation of explored knowledge territory"""
    
    def __init__(self, storage_path: str = None):
        self.graph = nx.DiGraph()
        self.storage_path = storage_path
        self.embedding_model = None
        self.similarity_threshold = 0.7
        
    def _generate_node_id(self, topic: str) -> str:
        """Generate unique node ID"""
        import hashlib
        return hashlib.md5(topic.encode()).hexdigest()[:12]
    
    async def add_discovery(self, discovery: Discovery) -> str:
        """Add new discovery to territory graph"""
        node_id = self._generate_node_id(discovery.topic)
        
        self.graph.add_node(
            node_id,
            label=discovery.topic,
            category="general",
            confidence=discovery.confidence,
            discovery_date=datetime.now(),
            last_accessed=datetime.now(),
            visit_count=1,
            depth=discovery.depth,
            sources=discovery.sources,
            metadata=discovery.metadata
        )
        
        if discovery.parent:
            parent_id = self._generate_node_id(discovery.parent)
            self.graph.add_edge(
                parent_id,
                node_id,
                relationship="parent_child",
                weight=discovery.relevance_score,
                discovery_date=datetime.now()
            )
        
        return node_id
    
    async def get_stats(self) -> TerritoryStats:
        """Calculate territory statistics"""
        if not self.graph.nodes():
            return TerritoryStats()
        
        depths = [self.graph.nodes[n].get('depth', 0) for n in self.graph.nodes()]
        confidences = [self.graph.nodes[n].get('confidence', 0) for n in self.graph.nodes()]
        
        now = datetime.now()
        recently = sum(
            1 for n in self.graph.nodes()
            if (now - self.graph.nodes[n].get('discovery_date', datetime.min)).days < 7
        )
        
        categories = Counter(
            self.graph.nodes[n].get('category', 'unknown')
            for n in self.graph.nodes()
        )
        
        return TerritoryStats(
            total_nodes=len(self.graph.nodes()),
            total_edges=len(self.graph.edges()),
            density=nx.density(self.graph),
            avg_clustering=nx.average_clustering(self.graph.to_undirected()) if len(self.graph.nodes()) > 1 else 0,
            connected_components=nx.number_connected_components(self.graph.to_undirected()) if len(self.graph.nodes()) > 0 else 0,
            max_depth=max(depths) if depths else 0,
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            recently_discovered=recently,
            category_distribution=categories
        )
    
    async def get_size(self) -> int:
        """Get number of nodes in territory"""
        return len(self.graph.nodes())
    
    async def save(self, path: Path):
        """Save territory graph"""
        nx.write_gpickle(self.graph, path)
    
    async def load(self, path: Path):
        """Load territory graph"""
        if path.exists():
            self.graph = nx.read_gpickle(path)

# ============================================================================
# NOVELTY DETECTION
# ============================================================================

class NoveltyDetector:
    """Detect novel information using multiple signals"""
    
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.novelty_threshold = 0.7
        
    async def score(self, content: str, context: Dict = None) -> NoveltyScore:
        """Calculate novelty score for content"""
        semantic = await self._semantic_novelty(content)
        structural = await self._structural_novelty(content)
        temporal = await self._temporal_novelty(content)
        source = await self._source_novelty(content, context)
        
        # Weighted combination
        composite = (
            semantic * 0.4 +
            structural * 0.3 +
            temporal * 0.2 +
            source * 0.1
        )
        
        return NoveltyScore(
            overall=composite,
            semantic=semantic,
            structural=structural,
            temporal=temporal,
            source=source,
            is_novel=composite > self.novelty_threshold
        )
    
    async def _semantic_novelty(self, content: str) -> float:
        """Calculate semantic novelty via TF-IDF distance from known knowledge base"""
        if not hasattr(self, '_seen_contents'):
            self._seen_contents = []
        if not self._seen_contents:
            self._seen_contents.append(content)
            return 0.8  # First content is novel
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            known_texts = self._seen_contents[-50:]
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([content] + known_texts)
            max_sim = float(cos_sim(tfidf[0:1], tfidf[1:]).max())
            self._seen_contents.append(content)
            return 1.0 - max_sim
        except ImportError:
            content_words = set(content.lower().split())
            all_known = set()
            for c in self._seen_contents[-50:]:
                all_known.update(c.lower().split())
            self._seen_contents.append(content)
            if not all_known:
                return 0.8
            overlap = len(content_words & all_known) / max(len(content_words), 1)
            return 1.0 - overlap

    async def _structural_novelty(self, content: str) -> float:
        """Calculate structural novelty based on content patterns"""
        # Score based on structural features not seen before
        features = {
            'has_code': '```' in content or 'def ' in content,
            'has_url': 'http' in content,
            'has_numbers': any(c.isdigit() for c in content),
            'is_long': len(content) > 500,
            'has_list': '\n-' in content or '\n*' in content,
        }
        novel_features = sum(1 for v in features.values() if v)
        return min(1.0, novel_features / 3.0)

    async def _temporal_novelty(self, content: str) -> float:
        """Calculate temporal novelty based on recency of related concepts"""
        if not hasattr(self, '_seen_timestamps') or not self._seen_timestamps:
            return 0.9
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        content_key = content[:100].lower()
        for seen_key, ts in reversed(list(self._seen_timestamps.items())):
            if content_key in seen_key or seen_key in content_key:
                hours_ago = (now - ts).total_seconds() / 3600
                return min(1.0, hours_ago / 24.0)
        return 0.9

    async def _source_novelty(self, content: str, context: Dict) -> float:
        """Calculate source novelty based on source diversity"""
        source = context.get('source', '')
        if not source:
            return 0.7
        known_sources = getattr(self, '_known_sources', set())
        if source in known_sources:
            return 0.3
        self._known_sources = known_sources | {source}
        return 0.9

# ============================================================================
# GAP IDENTIFICATION
# ============================================================================

class KnowledgeGapIdentifier:
    """Identify gaps in knowledge territory"""
    
    def __init__(self, territory_graph: KnowledgeTerritoryGraph):
        self.graph = territory_graph
        
    async def identify_gaps(self) -> List[KnowledgeGap]:
        """Identify different types of knowledge gaps"""
        gaps = []
        
        structural = await self._find_structural_gaps()
        gaps.extend(structural)
        
        coverage = await self._find_coverage_gaps()
        gaps.extend(coverage)
        
        depth = await self._find_depth_gaps()
        gaps.extend(depth)
        
        recency = await self._find_recency_gaps()
        gaps.extend(recency)
        
        return sorted(gaps, key=lambda x: x.priority, reverse=True)
    
    async def _find_structural_gaps(self) -> List[KnowledgeGap]:
        """Find structural gaps"""
        gaps = []
        
        if len(self.graph.graph.nodes()) < 10:
            return gaps
        
        undirected = self.graph.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        if len(components) > 1:
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    gaps.append(KnowledgeGap(
                        type="structural",
                        description=f"Missing connection between {len(comp1)} and {len(comp2)} nodes",
                        priority=0.7,
                        estimated_value=len(comp1) * len(comp2)
                    ))
        
        return gaps
    
    async def _find_coverage_gaps(self) -> List[KnowledgeGap]:
        """Find coverage gaps"""
        gaps = []
        
        categories = Counter(
            self.graph.graph.nodes[n].get('category', 'unknown')
            for n in self.graph.graph.nodes()
        )
        
        total = sum(categories.values())
        
        for category, count in categories.items():
            ratio = count / total if total > 0 else 0
            if ratio < 0.1:
                gaps.append(KnowledgeGap(
                    type="coverage",
                    description=f"Underrepresented category: {category}",
                    priority=0.6,
                    category=category,
                    current_coverage=ratio
                ))
        
        return gaps
    
    async def _find_depth_gaps(self) -> List[KnowledgeGap]:
        """Find depth gaps"""
        gaps = []
        
        for node in self.graph.graph.nodes():
            degree = self.graph.graph.degree(node)
            depth = self.graph.graph.nodes[node].get('depth', 0)
            
            if degree > 3 and depth < 3:
                gaps.append(KnowledgeGap(
                    type="depth",
                    description="Shallow exploration of well-connected topic",
                    priority=0.5,
                    node_id=node,
                    current_depth=depth,
                    recommended_depth=5
                ))
        
        return gaps
    
    async def _find_recency_gaps(self) -> List[KnowledgeGap]:
        """Find recency gaps"""
        gaps = []
        now = datetime.now()
        
        for node in self.graph.graph.nodes():
            last_accessed = self.graph.graph.nodes[node].get('last_accessed', datetime.min)
            age_days = (now - last_accessed).days
            
            if age_days > 30:
                gaps.append(KnowledgeGap(
                    type="recency",
                    description=f"Outdated information: {age_days} days old",
                    priority=min(age_days / 100, 0.9),
                    node_id=node,
                    age_days=age_days
                ))
        
        return gaps

# ============================================================================
# DISCOVERY LOGGER
# ============================================================================

class DiscoveryLogger:
    """Comprehensive logging of discovery activities"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS discovery_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                topic TEXT NOT NULL,
                strategy TEXT,
                novelty_score REAL,
                relevance_score REAL,
                confidence REAL,
                depth INTEGER,
                parent_topic TEXT,
                sources TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def log_discovery(self, discovery: Discovery):
        """Log a discovery event"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO discovery_events 
            (event_type, topic, strategy, novelty_score, relevance_score, 
             confidence, depth, parent_topic, sources, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "discovery",
            discovery.topic,
            discovery.strategy,
            discovery.novelty_score,
            discovery.relevance_score,
            discovery.confidence,
            discovery.depth,
            discovery.parent,
            json.dumps(discovery.sources),
            json.dumps(discovery.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def get_discovery_stats(self, time_range: Tuple[datetime, datetime]) -> DiscoveryStats:
        """Get discovery statistics"""
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(novelty_score) as avg_novelty,
                AVG(relevance_score) as avg_relevance,
                COUNT(DISTINCT strategy) as strategies,
                COUNT(DISTINCT topic) as unique_topics
            FROM discovery_events
            WHERE timestamp BETWEEN ? AND ?
        """, (time_range[0].isoformat(), time_range[1].isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        return DiscoveryStats(
            total_discoveries=row[0] or 0,
            average_novelty=row[1] or 0,
            average_relevance=row[2] or 0,
            strategies_used=row[3] or 0,
            unique_topics=row[4] or 0
        )
    
    async def get_recent_count(self, hours: int = 24) -> int:
        """Get count of recent discoveries"""
        conn = sqlite3.connect(self.db_path)
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor = conn.execute("""
            SELECT COUNT(*) FROM discovery_events
            WHERE timestamp > ?
        """, (since,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count

# ============================================================================
# EXPLORATION-EXPLOITATION BALANCER
# ============================================================================

class ExplorationExploitationBalancer:
    """Balance between exploration and exploitation"""
    
    def __init__(self):
        self.exploration_rate = 0.5
        self.history = deque(maxlen=100)
        
    async def get_balance(self, context: Dict = None) -> BalanceDecision:
        """Determine optimal balance"""
        # Calculate factors
        coverage = context.get("coverage", 0.5) if context else 0.5
        discovery_rate = context.get("discovery_rate", 0.1) if context else 0.1
        
        # Adjust target rate
        if coverage < 0.3:
            target_rate = 0.7
        elif discovery_rate < 0.1:
            target_rate = 0.6
        else:
            target_rate = 0.5
        
        # Smooth transition
        self.exploration_rate = 0.8 * self.exploration_rate + 0.2 * target_rate
        
        # Make decision
        action = "explore" if random.random() < self.exploration_rate else "exploit"
        
        return BalanceDecision(
            action=action,
            strategy="hybrid" if action == "explore" else "value_driven",
            exploration_rate=self.exploration_rate
        )

# ============================================================================
# MAIN DISCOVERY LOOP
# ============================================================================

class DiscoveryLoop:
    """Main discovery loop implementation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.running = False
        
        # Initialize components
        self.novelty_detector = NoveltyDetector()
        self.explorer = HybridExplorer(self.novelty_detector)
        
        data_dir = Path.home() / ".openclaw" / "discovery"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self.mapper = KnowledgeTerritoryGraph(str(data_dir / "territory.graph"))
        self.gap_identifier = KnowledgeGapIdentifier(self.mapper)
        self.logger = DiscoveryLogger(str(data_dir / "discovery.db"))
        self.balancer = ExplorationExploitationBalancer()
        
        self.seed_topics = self.config.get("seeds", [
            "artificial intelligence",
            "machine learning",
            "natural language processing"
        ])
        
    async def start(self):
        """Start the discovery loop"""
        self.running = True
        
        while self.running:
            try:
                # Get balance decision
                stats = await self.mapper.get_stats()
                context = {
                    "coverage": stats.density,
                    "discovery_rate": 0.1
                }
                
                decision = await self.balancer.get_balance(context)
                
                if decision.action == "explore":
                    await self._explore()
                else:
                    await self._exploit()
                
                # Sleep before next iteration
                await asyncio.sleep(self.config.get("loop_interval", 300))
                
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Discovery loop error: {e}")
                await asyncio.sleep(60)
    
    async def _explore(self):
        """Execute exploration"""
        result = await self.explorer.explore(self.seed_topics)
        
        for discovery in result.discoveries:
            # Log discovery
            await self.logger.log_discovery(discovery)
            
            # Add to territory
            await self.mapper.add_discovery(discovery)
    
    async def _exploit(self):
        """Execute exploitation (review existing knowledge)"""
        logger.info("Exploiting existing knowledge")
        stats = await self.mapper.get_stats()
        if stats.total_nodes > 0:
            recent = await self.logger.get_recent_count(hours=24)
            logger.info(f"Reviewed {stats.total_nodes} nodes, {recent} recent discoveries")
    
    async def get_status(self) -> DiscoveryStatus:
        """Get current status"""
        stats = await self.mapper.get_stats()
        recent = await self.logger.get_recent_count(hours=24)
        
        return DiscoveryStatus(
            running=self.running,
            territory_size=stats.total_nodes,
            queue_size=0,
            exploration_rate=self.balancer.exploration_rate,
            recent_discoveries=recent,
            active_strategies=["hybrid"],
            next_exploration=datetime.now() + timedelta(seconds=300)
        )
    
    def stop(self):
        """Stop the discovery loop"""
        self.running = False

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage of Discovery Loop"""
    
    # Create discovery loop
    config = {
        "seeds": [
            "artificial intelligence",
            "machine learning",
            "deep learning"
        ],
        "loop_interval": 60  # 1 minute for demo
    }
    
    loop = DiscoveryLoop(config)
    
    # Run one exploration cycle
    print("Starting Discovery Loop...")
    
    # Explore
    result = await loop.explorer.explore(loop.seed_topics)
    
    print(f"\nExploration Results:")
    print(f"Strategy: {result.strategy}")
    print(f"Discoveries: {len(result.discoveries)}")
    
    for i, discovery in enumerate(result.discoveries[:5], 1):
        print(f"\n  {i}. {discovery.topic}")
        print(f"     Confidence: {discovery.confidence:.2f}")
        print(f"     Novelty: {discovery.novelty_score:.2f}")
        print(f"     Depth: {discovery.depth}")
    
    # Get territory stats
    for discovery in result.discoveries:
        await loop.mapper.add_discovery(discovery)
    
    stats = await loop.mapper.get_stats()
    
    print(f"\n\nTerritory Statistics:")
    print(f"Total Nodes: {stats.total_nodes}")
    print(f"Total Edges: {stats.total_edges}")
    print(f"Density: {stats.density:.3f}")
    print(f"Max Depth: {stats.max_depth}")
    
    # Identify gaps
    gaps = await loop.gap_identifier.identify_gaps()
    
    print(f"\n\nKnowledge Gaps Identified: {len(gaps)}")
    for gap in gaps[:3]:
        print(f"  - {gap.type}: {gap.description} (priority: {gap.priority:.2f})")
    
    print("\nDiscovery Loop demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())
