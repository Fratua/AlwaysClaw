"""
Advanced Self-Learning Loop Implementation
OpenClaw Windows 10 AI Agent System
Continuous Learning with Knowledge Consolidation

This module provides the core implementation for the self-learning loop,
including continuous learning, forgetting prevention, knowledge consolidation,
spaced repetition, transfer learning, and retrieval practice.
"""

import asyncio
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS AND CONSTANTS
# ============================================================================

class ExperienceType(Enum):
    INTERACTION = "interaction"
    TASK = "task"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


class ConsolidationPhase(Enum):
    REPLAY = "replay"
    INTEGRATION = "integration"
    GENERALIZATION = "generalization"
    INDEXING = "indexing"


class RetrievalMethod(Enum):
    FREE_RECALL = "free_recall"
    CUED_RECALL = "cued_recall"
    RECOGNITION = "recognition"


# Default configuration
DEFAULT_CONFIG = {
    'experience_buffer_size': 10000,
    'consolidation_idle_threshold_minutes': 15,
    'max_consolidation_window_hours': 4,
    'ewc_lambda': 1000,
    'spaced_repetition_base_intervals': [1, 3, 7, 14, 30, 60, 120],
    'target_retention': 0.9,
    'embedding_dim': 768,
    'min_learning_rate': 0.00001,
    'max_learning_rate': 0.1,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Experience:
    """Represents a learning experience."""
    id: str
    timestamp: datetime
    type: ExperienceType
    content: Dict[str, Any]
    context: Dict[str, Any]
    outcome: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    processed: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class KnowledgeUnit:
    """Represents a unit of consolidated knowledge."""
    id: str
    domain: str
    summary: str
    concepts: List[str] = field(default_factory=list)
    procedures: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    embedding: Optional[np.ndarray] = None
    retention_score: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class LearningTask:
    """Represents a learning task."""
    id: str
    domain: str
    description: str
    initial_strategy: Dict[str, Any] = field(default_factory=dict)
    strategy_adaptations: List[Dict] = field(default_factory=list)
    final_performance: float = 0.0
    time_to_proficiency: Optional[timedelta] = None
    success: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ReviewSession:
    """Represents a scheduled review session."""
    skill_id: str
    scheduled_time: datetime
    session_type: str = "regular"
    duration_minutes: int = 15
    completed: bool = False
    performance: float = 0.0


@dataclass
class RetrievalSession:
    """Represents a retrieval practice session."""
    knowledge_id: str
    scheduled_time: datetime
    method: RetrievalMethod = RetrievalMethod.FREE_RECALL
    duration_minutes: int = 5
    completed: bool = False


@dataclass
class TransferRecommendation:
    """Represents a knowledge transfer recommendation."""
    source_domain: str
    target_domain: str
    potential: float
    transferable_skills: List[str] = field(default_factory=list)
    adaptation_rules: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXPERIENCE BUFFER
# ============================================================================

class ExperienceBuffer:
    """
    Buffer for storing and managing learning experiences.
    Implements prioritized experience storage.
    """
    
    def __init__(self, max_size: int = 10000):
        self.experiences: List[Experience] = []
        self.max_size = max_size
        self.priority_queue: List[Tuple[float, str]] = []  # (importance, id)
        
    async def add(self, experience: Experience) -> bool:
        """Add experience to buffer."""
        if len(self.experiences) >= self.max_size:
            # Remove lowest priority experience
            await self._remove_lowest_priority()
        
        self.experiences.append(experience)
        self.priority_queue.append((experience.importance, experience.id))
        self.priority_queue.sort(reverse=True)  # Highest priority first
        
        logger.info(f"Added experience {experience.id} (importance: {experience.importance})")
        return True
    
    async def get_batch(self, batch_size: int = 32) -> List[Experience]:
        """Get batch of unprocessed experiences."""
        unprocessed = [e for e in self.experiences if not e.processed]
        
        # Sort by importance and time
        unprocessed.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        
        return unprocessed[:batch_size]
    
    async def mark_processed(self, experience_id: str):
        """Mark experience as processed."""
        for exp in self.experiences:
            if exp.id == experience_id:
                exp.processed = True
                break
    
    async def _remove_lowest_priority(self):
        """Remove lowest priority experience from buffer."""
        if self.priority_queue:
            _, lowest_id = self.priority_queue.pop()
            self.experiences = [e for e in self.experiences if e.id != lowest_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'total_experiences': len(self.experiences),
            'unprocessed': len([e for e in self.experiences if not e.processed]),
            'avg_importance': np.mean([e.importance for e in self.experiences]) if self.experiences else 0
        }


# ============================================================================
# CATASTROPHIC FORGETTING PREVENTION
# ============================================================================

class ElasticWeightConsolidation:
    """
    Implements Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
    Based on Kirkpatrick et al. (2017).
    """
    
    def __init__(self, lambda_ewc: float = 1000, importance_threshold: float = 0.7):
        self.parameter_importance: Dict[str, float] = {}
        self.optimal_parameters: Dict[str, float] = {}
        self.lambda_ewc = lambda_ewc
        self.importance_threshold = importance_threshold
        
    async def compute_parameter_importance(self, task_data: List[Experience]) -> Dict[str, float]:
        """Compute Fisher Information Matrix diagonal as importance measure."""
        importance_scores: Dict[str, float] = defaultdict(float)
        
        for experience in task_data:
            # Simulate gradient computation
            gradients = await self._compute_gradients(experience)
            
            # Accumulate squared gradients (Fisher Information)
            for param_name, grad in gradients.items():
                importance_scores[param_name] += grad ** 2
        
        # Normalize by number of samples
        n_samples = len(task_data) if task_data else 1
        for param_name in importance_scores:
            importance_scores[param_name] /= n_samples
        
        self.parameter_importance = dict(importance_scores)
        return self.parameter_importance
    
    async def _compute_gradients(self, experience: Experience) -> Dict[str, float]:
        """
        Compute gradients for experience using numerical differentiation.

        Approximates the gradient of a loss function with respect to each
        parameter using the experience's state, outcome, and importance.
        """
        # Extract features from the experience
        state_data = experience.state if isinstance(experience.state, dict) else {}
        outcome_val = float(experience.outcome) if isinstance(experience.outcome, (int, float)) else 0.0
        importance = float(experience.importance)

        # Build a parameter vector from current optimal params or defaults
        n_params = max(10, len(self.optimal_parameters))
        param_names = [f'param_{i}' for i in range(n_params)]

        # Get current parameter values
        params = np.array([
            self.optimal_parameters.get(name, 0.0) for name in param_names
        ], dtype=np.float64)

        # Build input features from experience state
        state_features = np.zeros(n_params, dtype=np.float64)
        if isinstance(state_data, dict):
            for i, (key, val) in enumerate(list(state_data.items())[:n_params]):
                try:
                    state_features[i] = float(val)
                except (TypeError, ValueError):
                    state_features[i] = hash(str(val)) % 100 / 100.0
        elif isinstance(experience.state, (list, np.ndarray)):
            arr = np.array(experience.state, dtype=np.float64).flatten()
            state_features[:min(len(arr), n_params)] = arr[:n_params]

        # Loss function: MSE between predicted output and actual outcome
        # predicted = dot(params, state_features), target = outcome
        epsilon = 1e-5
        gradients = {}

        predicted = np.dot(params, state_features)
        base_loss = (predicted - outcome_val) ** 2

        for i, name in enumerate(param_names):
            # Numerical gradient via central difference
            params_plus = params.copy()
            params_plus[i] += epsilon
            loss_plus = (np.dot(params_plus, state_features) - outcome_val) ** 2

            params_minus = params.copy()
            params_minus[i] -= epsilon
            loss_minus = (np.dot(params_minus, state_features) - outcome_val) ** 2

            grad = (loss_plus - loss_minus) / (2 * epsilon)
            # Scale by importance
            gradients[name] = float(grad * importance)

        return gradients
    
    async def apply_ewc_penalty(self, loss: float, current_params: Dict[str, float]) -> float:
        """Apply EWC regularization penalty to loss function."""
        ewc_penalty = 0.0
        
        for param_name, current_value in current_params.items():
            if param_name in self.parameter_importance:
                importance = self.parameter_importance[param_name]
                optimal_value = self.optimal_parameters.get(param_name, current_value)
                ewc_penalty += importance * (current_value - optimal_value) ** 2
        
        return loss + (self.lambda_ewc / 2) * ewc_penalty
    
    async def store_optimal_params(self, params: Dict[str, float]):
        """Store optimal parameters for a task."""
        self.optimal_parameters = params.copy()


class MemoryReplaySystem:
    """
    Maintains episodic memory for experience replay during learning.
    Implements prioritized experience replay.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.episodic_memory: List[Dict[str, Any]] = []
        self.buffer_size = buffer_size
        self.replay_ratio = 0.3
        self.prioritization_alpha = 0.6
        
    async def store_experience(self, experience: Experience, importance: float):
        """Store experience with importance weighting."""
        memory_item = {
            'experience': experience,
            'importance': importance,
            'last_accessed': datetime.now(),
            'access_count': 0,
            'retrieval_strength': 1.0
        }
        
        self.episodic_memory.append(memory_item)
        
        # Maintain buffer size
        if len(self.episodic_memory) > self.buffer_size:
            await self._maintain_buffer()
    
    async def sample_for_replay(self, batch_size: int) -> List[Experience]:
        """Sample experiences using prioritized experience replay."""
        if not self.episodic_memory:
            return []
        
        # Calculate sampling probabilities
        priorities = [
            m['importance'] ** self.prioritization_alpha
            for m in self.episodic_memory
        ]
        total_priority = sum(priorities)
        
        if total_priority == 0:
            probabilities = [1.0 / len(self.episodic_memory)] * len(self.episodic_memory)
        else:
            probabilities = [p / total_priority for p in priorities]
        
        # Sample based on priorities
        indices = np.random.choice(
            len(self.episodic_memory),
            size=min(batch_size, len(self.episodic_memory)),
            replace=False,
            p=probabilities
        )
        
        samples = []
        for i in indices:
            self.episodic_memory[i]['access_count'] += 1
            self.episodic_memory[i]['last_accessed'] = datetime.now()
            samples.append(self.episodic_memory[i]['experience'])
        
        return samples
    
    async def _maintain_buffer(self):
        """Maintain buffer size by removing least important items."""
        # Sort by combined score of importance and recency
        self.episodic_memory.sort(
            key=lambda m: (
                m['importance'] * 0.7 + 
                (1 / (1 + (datetime.now() - m['last_accessed']).days)) * 0.3
            ),
            reverse=True
        )
        
        # Keep top buffer_size items
        self.episodic_memory = self.episodic_memory[:self.buffer_size]


# ============================================================================
# KNOWLEDGE CONSOLIDATION
# ============================================================================

class KnowledgeConsolidationEngine:
    """
    Manages knowledge consolidation during system idle periods.
    Implements sleep-phase learning inspired by memory consolidation research.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.consolidation_queue: List[KnowledgeUnit] = []
        self.phase_durations = {
            ConsolidationPhase.REPLAY: 0.25,
            ConsolidationPhase.INTEGRATION: 0.30,
            ConsolidationPhase.GENERALIZATION: 0.25,
            ConsolidationPhase.INDEXING: 0.20
        }
        self.consolidation_active = False
        
    async def schedule_consolidation(self, knowledge_unit: KnowledgeUnit):
        """Schedule knowledge unit for consolidation."""
        self.consolidation_queue.append(knowledge_unit)
        logger.info(f"Scheduled knowledge {knowledge_unit.id} for consolidation")
    
    async def run_consolidation_cycle(self, available_time: timedelta):
        """Execute full consolidation cycle."""
        if self.consolidation_active:
            logger.warning("Consolidation already active, skipping")
            return
        
        self.consolidation_active = True
        total_seconds = available_time.total_seconds()
        
        try:
            for phase in ConsolidationPhase:
                phase_duration = timedelta(
                    seconds=total_seconds * self.phase_durations[phase]
                )
                
                logger.info(f"Starting consolidation phase: {phase.value}")
                
                if phase == ConsolidationPhase.REPLAY:
                    await self._run_replay_phase(phase_duration)
                elif phase == ConsolidationPhase.INTEGRATION:
                    await self._run_integration_phase(phase_duration)
                elif phase == ConsolidationPhase.GENERALIZATION:
                    await self._run_generalization_phase(phase_duration)
                elif phase == ConsolidationPhase.INDEXING:
                    await self._run_indexing_phase(phase_duration)
                    
        finally:
            self.consolidation_active = False
    
    async def _run_replay_phase(self, duration: timedelta):
        """Reactivate and strengthen recent knowledge."""
        start_time = datetime.now()
        
        for knowledge in self.consolidation_queue:
            if datetime.now() - start_time >= duration:
                break
            
            # Simulate memory reactivation
            await self._reactivate_knowledge(knowledge)
            logger.debug(f"Reactivated knowledge: {knowledge.id}")
    
    async def _run_integration_phase(self, duration: timedelta):
        """Merge related knowledge and resolve conflicts."""
        start_time = datetime.now()
        
        # Group related knowledge
        knowledge_groups = await self._group_related_knowledge()
        
        for group in knowledge_groups:
            if datetime.now() - start_time >= duration:
                break
            
            # Merge knowledge in group
            await self._merge_knowledge_group(group)
    
    async def _run_generalization_phase(self, duration: timedelta):
        """Extract general patterns and create transferable skills."""
        start_time = datetime.now()
        
        # Find patterns in consolidated knowledge
        patterns = await self._extract_patterns()
        
        for pattern in patterns:
            if datetime.now() - start_time >= duration:
                break
            
            # Create generalized skill
            await self._create_generalized_skill(pattern)
    
    async def _run_indexing_phase(self, duration: timedelta):
        """Optimize knowledge retrieval structures."""
        # Update embeddings
        await self._update_embeddings()
        
        # Rebuild indices
        await self._rebuild_indices()
        
        logger.info("Indexing phase completed")
    
    async def _reactivate_knowledge(self, knowledge: KnowledgeUnit):
        """Simulate knowledge reactivation."""
        knowledge.access_count += 1
        knowledge.last_accessed = datetime.now()
    
    async def _group_related_knowledge(self) -> List[List[KnowledgeUnit]]:
        """Group related knowledge units."""
        # Simple grouping by domain
        groups: Dict[str, List[KnowledgeUnit]] = defaultdict(list)
        for knowledge in self.consolidation_queue:
            groups[knowledge.domain].append(knowledge)
        return list(groups.values())
    
    async def _merge_knowledge_group(self, group: List[KnowledgeUnit]):
        """Merge knowledge units in a group."""
        if len(group) < 2:
            return
        
        # Merge concepts and procedures
        merged_concepts = set()
        merged_procedures = []
        
        for knowledge in group:
            merged_concepts.update(knowledge.concepts)
            merged_procedures.extend(knowledge.procedures)
        
        logger.info(f"Merged {len(group)} knowledge units")
    
    async def _extract_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from knowledge."""
        patterns = []
        
        # Simple pattern extraction
        domain_counts: Dict[str, int] = defaultdict(int)
        for knowledge in self.consolidation_queue:
            domain_counts[knowledge.domain] += 1
        
        for domain, count in domain_counts.items():
            if count > 2:
                patterns.append({
                    'type': 'domain_frequency',
                    'domain': domain,
                    'count': count
                })
        
        return patterns
    
    async def _create_generalized_skill(self, pattern: Dict[str, Any]):
        """Create generalized skill from pattern."""
        logger.info(f"Created generalized skill from pattern: {pattern}")
    
    async def _update_embeddings(self):
        """Update knowledge embeddings."""
        logger.info("Updating knowledge embeddings")
        if hasattr(self, 'knowledge_base') and self.knowledge_base:
            for item in self.knowledge_base:
                if not item.get('embedding'):
                    item['embedding'] = hash(item.get('content', '')) % (2**16)
        logger.debug("Embedding update complete")
    
    async def _rebuild_indices(self):
        """Rebuild search indices."""
        logger.info("Rebuilding search indices")
        if hasattr(self, 'knowledge_base') and self.knowledge_base:
            self._search_index = {}
            for i, item in enumerate(self.knowledge_base):
                for word in str(item.get('content', '')).lower().split():
                    self._search_index.setdefault(word, []).append(i)
        logger.debug("Index rebuild complete")


# ============================================================================
# SPACED REPETITION SYSTEM
# ============================================================================

class SpacedRepetitionSystem:
    """
    Implements adaptive spaced repetition for skill retention.
    Based on SM-2 algorithm with adaptive adjustments.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.base_intervals = self.config.get(
            'spaced_repetition_base_intervals',
            [1, 3, 7, 14, 30, 60, 120]
        )
        self.easiness_factor_default = 2.5
        self.min_easiness_factor = 1.3
        self.target_retention = self.config.get('target_retention', 0.9)
        self.skill_schedules: Dict[str, Dict[str, Any]] = {}
        
    async def calculate_next_review(
        self,
        skill_id: str,
        performance: float,
        previous_interval: int = 0
    ) -> datetime:
        """Calculate optimal next review time based on performance."""
        skill_data = self.skill_schedules.get(skill_id, {
            'repetition_count': 0,
            'easiness_factor': self.easiness_factor_default,
            'last_review': None,
            'performance_history': []
        })
        
        # Update easiness factor
        ef = skill_data['easiness_factor']
        ef = ef + (0.1 - (1 - performance) * (0.08 + (1 - performance) * 0.02))
        ef = max(ef, self.min_easiness_factor)
        skill_data['easiness_factor'] = ef
        
        # Calculate next interval
        if performance < 0.6:
            # Failed review - reset
            interval = self.base_intervals[0]
            skill_data['repetition_count'] = 0
        else:
            # Successful review - increase interval
            rep_count = skill_data['repetition_count']
            if rep_count < len(self.base_intervals):
                base_interval = self.base_intervals[rep_count]
            else:
                base_interval = self.base_intervals[-1] * ef
            
            interval = int(base_interval * ef)
            skill_data['repetition_count'] += 1
        
        # Calculate next review date
        next_review = datetime.now() + timedelta(days=interval)
        skill_data['next_review'] = next_review
        skill_data['last_review'] = datetime.now()
        skill_data['performance_history'].append({
            'date': datetime.now(),
            'performance': performance
        })
        
        self.skill_schedules[skill_id] = skill_data
        
        return next_review
    
    async def get_due_reviews(self) -> List[str]:
        """Get all skills due for review."""
        due_skills = []
        now = datetime.now()
        
        for skill_id, data in self.skill_schedules.items():
            next_review = data.get('next_review')
            if next_review and next_review <= now:
                due_skills.append(skill_id)
        
        # Sort by priority (lower performance = higher priority)
        due_skills.sort(
            key=lambda s: self.skill_schedules[s]['performance_history'][-1]['performance']
            if self.skill_schedules[s]['performance_history'] else 0
        )
        
        return due_skills
    
    async def get_skill_stats(self, skill_id: str) -> Dict[str, Any]:
        """Get statistics for a skill."""
        data = self.skill_schedules.get(skill_id, {})
        
        return {
            'repetition_count': data.get('repetition_count', 0),
            'easiness_factor': data.get('easiness_factor', self.easiness_factor_default),
            'next_review': data.get('next_review'),
            'last_review': data.get('last_review'),
            'performance_history': data.get('performance_history', [])
        }


# ============================================================================
# TRANSFER LEARNING SYSTEM
# ============================================================================

class TransferLearningSystem:
    """
    Manages knowledge transfer between different domains.
    """
    
    def __init__(self):
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        self.transfer_matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.domains: set = set()
        
    async def register_domain(self, domain: str, embedding: np.ndarray):
        """Register a domain with its embedding."""
        self.domain_embeddings[domain] = embedding
        self.domains.add(domain)
        
    async def calculate_domain_similarity(
        self,
        domain_a: str,
        domain_b: str
    ) -> float:
        """Calculate similarity between two domains."""
        embedding_a = self.domain_embeddings.get(domain_a)
        embedding_b = self.domain_embeddings.get(domain_b)
        
        if embedding_a is None or embedding_b is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b) + 1e-8
        )
        
        return float(similarity)
    
    async def build_transfer_matrix(self):
        """Build the complete skill transfer matrix."""
        for source in self.domains:
            self.transfer_matrix[source] = {}
            
            for target in self.domains:
                if source != target:
                    similarity = await self.calculate_domain_similarity(source, target)
                    
                    self.transfer_matrix[source][target] = {
                        'similarity': similarity,
                        'transfer_potential': max(0, similarity),
                        'adaptation_rules': await self._infer_adaptation_rules(source, target)
                    }
    
    async def get_transfer_recommendations(
        self,
        target_domain: str,
        min_potential: float = 0.5
    ) -> List[TransferRecommendation]:
        """Get recommendations for knowledge transfer to target domain."""
        recommendations = []
        
        for source in self.domains:
            if source == target_domain:
                continue
            
            transfer_data = self.transfer_matrix.get(source, {}).get(target_domain, {})
            potential = transfer_data.get('transfer_potential', 0)
            
            if potential >= min_potential:
                recommendations.append(TransferRecommendation(
                    source_domain=source,
                    target_domain=target_domain,
                    potential=potential,
                    adaptation_rules=transfer_data.get('adaptation_rules', {})
                ))
        
        # Sort by transfer potential
        recommendations.sort(key=lambda x: x.potential, reverse=True)
        
        return recommendations
    
    async def _infer_adaptation_rules(
        self,
        source: str,
        target: str
    ) -> Dict[str, Any]:
        """Infer adaptation rules between domains."""
        # Build concept mapping by finding shared terminology
        source_terms = set(source.lower().split())
        target_terms = set(target.lower().split())
        shared = source_terms & target_terms

        concept_mapping = {}
        for term in shared:
            concept_mapping[term] = term  # Direct mapping for shared concepts

        # Build procedure adaptations
        procedure_adaptations = []
        if source != target:
            procedure_adaptations.append({
                'type': 'domain_transfer',
                'source_domain': source,
                'target_domain': target,
                'shared_concepts': list(shared),
                'adaptation_needed': len(shared) < min(len(source_terms), len(target_terms)) * 0.3
            })

        return {
            'concept_mapping': concept_mapping,
            'procedure_adaptations': procedure_adaptations,
            'transfer_feasibility': len(shared) / max(len(source_terms | target_terms), 1)
        }


# ============================================================================
# RETRIEVAL PRACTICE SYSTEM
# ============================================================================

class RetrievalPracticeSystem:
    """
    Schedules and manages retrieval practice sessions.
    """
    
    def __init__(self):
        self.retrieval_calendar: Dict[str, List[RetrievalSession]] = {}
        self.retrieval_strategies = {
            RetrievalMethod.FREE_RECALL: 0.3,
            RetrievalMethod.CUED_RECALL: 0.5,
            RetrievalMethod.RECOGNITION: 0.2
        }
        
    async def schedule_retrieval_practice(
        self,
        knowledge_id: str,
        priority: float = 0.5
    ) -> List[RetrievalSession]:
        """Schedule retrieval practice sessions for knowledge."""
        sessions = []
        
        # Initial retrieval
        sessions.append(RetrievalSession(
            knowledge_id=knowledge_id,
            scheduled_time=datetime.now() + timedelta(hours=1),
            method=RetrievalMethod.FREE_RECALL,
            duration_minutes=5
        ))
        
        # Follow-up retrievals (spaced)
        intervals = [1, 3, 7, 14, 30]  # days
        
        for i, interval in enumerate(intervals):
            adjusted_interval = int(interval * (1 + (1 - priority) * 0.5))
            
            # Select retrieval method
            if i < 2:
                method = RetrievalMethod.FREE_RECALL
            elif i < 4:
                method = RetrievalMethod.CUED_RECALL
            else:
                method = np.random.choice(
                    list(self.retrieval_strategies.keys()),
                    p=list(self.retrieval_strategies.values())
                )
            
            sessions.append(RetrievalSession(
                knowledge_id=knowledge_id,
                scheduled_time=datetime.now() + timedelta(days=adjusted_interval),
                method=method,
                duration_minutes=5 + i * 2
            ))
        
        self.retrieval_calendar[knowledge_id] = sessions
        
        return sessions
    
    async def get_due_retrieval_sessions(self) -> List[RetrievalSession]:
        """Get all retrieval sessions due for practice."""
        due_sessions = []
        now = datetime.now()
        
        for knowledge_id, sessions in self.retrieval_calendar.items():
            for session in sessions:
                if not session.completed and session.scheduled_time <= now:
                    due_sessions.append(session)
        
        return due_sessions
    
    async def complete_retrieval_session(
        self,
        knowledge_id: str,
        performance: float
    ):
        """Mark retrieval session as complete."""
        sessions = self.retrieval_calendar.get(knowledge_id, [])
        
        for session in sessions:
            if not session.completed and session.scheduled_time <= datetime.now():
                session.completed = True
                logger.info(f"Completed retrieval session for {knowledge_id}, performance: {performance}")
                break


# ============================================================================
# MAIN SELF-LEARNING LOOP
# ============================================================================

class SelfLearningLoop:
    """
    Main self-learning loop that orchestrates all learning components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.experience_buffer = ExperienceBuffer(
            self.config.get('experience_buffer_size', 10000)
        )
        self.ewc = ElasticWeightConsolidation(
            lambda_ewc=self.config.get('ewc_lambda', 1000)
        )
        self.memory_replay = MemoryReplaySystem()
        self.consolidation_engine = KnowledgeConsolidationEngine(self.config)
        self.spaced_repetition = SpacedRepetitionSystem(self.config)
        self.transfer_learning = TransferLearningSystem()
        self.retrieval_practice = RetrievalPracticeSystem()
        
        # Knowledge store
        self.knowledge_store: Dict[str, KnowledgeUnit] = {}
        
        # State
        self.running = False
        self.idle_start_time: Optional[datetime] = None
        
    async def start(self):
        """Start the self-learning loop."""
        self.running = True
        logger.info("Self-Learning Loop started")
        
        # Start background tasks
        asyncio.create_task(self._experience_processing_loop())
        asyncio.create_task(self._idle_monitoring_loop())
        asyncio.create_task(self._review_scheduling_loop())
    
    async def stop(self):
        """Stop the self-learning loop."""
        self.running = False
        logger.info("Self-Learning Loop stopped")
    
    async def ingest_experience(self, experience: Experience) -> str:
        """
        Ingest a new experience for learning.
        Returns knowledge unit ID.
        """
        await self.experience_buffer.add(experience)
        logger.info(f"Ingested experience: {experience.id}")
        return experience.id
    
    async def create_knowledge_unit(
        self,
        domain: str,
        summary: str,
        concepts: List[str] = None,
        confidence: float = 0.5
    ) -> KnowledgeUnit:
        """Create a new knowledge unit."""
        knowledge = KnowledgeUnit(
            id=str(uuid.uuid4()),
            domain=domain,
            summary=summary,
            concepts=concepts or [],
            confidence=confidence
        )
        
        self.knowledge_store[knowledge.id] = knowledge
        
        # Schedule for consolidation
        await self.consolidation_engine.schedule_consolidation(knowledge)
        
        # Schedule retrieval practice
        await self.retrieval_practice.schedule_retrieval_practice(knowledge.id)
        
        logger.info(f"Created knowledge unit: {knowledge.id}")
        return knowledge
    
    async def query_knowledge(
        self,
        query: str,
        top_k: int = 10
    ) -> List[KnowledgeUnit]:
        """Query knowledge store."""
        # Simple keyword-based search (replace with semantic search in production)
        results = []
        query_terms = query.lower().split()
        
        for knowledge in self.knowledge_store.values():
            score = 0
            knowledge_text = f"{knowledge.summary} {' '.join(knowledge.concepts)}".lower()
            
            for term in query_terms:
                if term in knowledge_text:
                    score += 1
            
            if score > 0:
                results.append((knowledge, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [k for k, _ in results[:top_k]]
    
    async def assess_retention(self, knowledge_id: str) -> Dict[str, Any]:
        """Assess retention level for specific knowledge."""
        knowledge = self.knowledge_store.get(knowledge_id)
        
        if not knowledge:
            return {'status': 'unknown', 'retention': 0}
        
        # Calculate time-based decay
        days_since_access = (datetime.now() - knowledge.last_accessed).days
        half_life = 30  # Default half-life in days
        retention = np.exp(-0.693 * days_since_access / half_life)
        
        # Determine status
        if retention >= 0.8:
            status = 'retained'
        elif retention >= 0.5:
            status = 'declining'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'retention': retention,
            'days_since_access': days_since_access,
            'access_count': knowledge.access_count
        }
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        return {
            'experience_buffer': self.experience_buffer.get_stats(),
            'knowledge_units': len(self.knowledge_store),
            'scheduled_reviews': len(await self.spaced_repetition.get_due_reviews()),
            'consolidation_queue': len(self.consolidation_engine.consolidation_queue),
            'registered_domains': len(self.transfer_learning.domains)
        }
    
    async def _experience_processing_loop(self):
        """Background loop for processing experiences."""
        while self.running:
            try:
                # Get batch of unprocessed experiences
                batch = await self.experience_buffer.get_batch(batch_size=32)
                
                for experience in batch:
                    # Process experience
                    await self._process_experience(experience)
                    
                    # Mark as processed
                    await self.experience_buffer.mark_processed(experience.id)
                
                # Wait before next batch
                await asyncio.sleep(60)  # Check every minute
                
            except (OSError, RuntimeError, PermissionError) as e:
                logger.error(f"Error in experience processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_experience(self, experience: Experience):
        """Process a single experience."""
        # Store in memory replay
        await self.memory_replay.store_experience(experience, experience.importance)
        
        # Update EWC importance if applicable
        if experience.type == ExperienceType.TASK and experience.outcome:
            await self.ewc.compute_parameter_importance([experience])
        
        logger.debug(f"Processed experience: {experience.id}")
    
    async def _idle_monitoring_loop(self):
        """Monitor for idle state and trigger consolidation."""
        idle_threshold = timedelta(
            minutes=self.config.get('consolidation_idle_threshold_minutes', 15)
        )
        
        while self.running:
            try:
                is_idle = await self._check_idle_state()
                
                if is_idle:
                    if self.idle_start_time is None:
                        self.idle_start_time = datetime.now()
                    
                    idle_duration = datetime.now() - self.idle_start_time
                    
                    if idle_duration >= idle_threshold:
                        # Trigger consolidation
                        max_window = timedelta(
                            hours=self.config.get('max_consolidation_window_hours', 4)
                        )
                        consolidation_time = min(idle_duration, max_window)
                        
                        await self.consolidation_engine.run_consolidation_cycle(
                            consolidation_time
                        )
                else:
                    self.idle_start_time = None
                
                await asyncio.sleep(60)  # Check every minute
                
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Error in idle monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_idle_state(self) -> bool:
        """Check if system is in idle state using system metrics."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            # Consider idle if CPU usage is below 15%
            if cpu > 15:
                return False
            # Check if any active tasks are running
            if hasattr(self, 'active_tasks') and self.active_tasks:
                return False
            return True
        except ImportError:
            return False
    
    async def _review_scheduling_loop(self):
        """Background loop for scheduling reviews."""
        while self.running:
            try:
                # Get due reviews
                due_reviews = await self.spaced_repetition.get_due_reviews()
                
                if due_reviews:
                    logger.info(f"{len(due_reviews)} skills due for review")
                
                # Get due retrieval sessions
                due_retrievals = await self.retrieval_practice.get_due_retrieval_sessions()
                
                if due_retrievals:
                    logger.info(f"{len(due_retrievals)} retrieval sessions due")
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except (OSError, RuntimeError, PermissionError) as e:
                logger.error(f"Error in review scheduling loop: {e}")
                await asyncio.sleep(3600)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_experience(
    content: Dict[str, Any],
    exp_type: ExperienceType = ExperienceType.OBSERVATION,
    importance: float = 0.5,
    context: Dict[str, Any] = None
) -> Experience:
    """Helper function to create an experience."""
    return Experience(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        type=exp_type,
        content=content,
        context=context or {},
        importance=importance
    )


async def demo():
    """Demo function to show self-learning loop functionality."""
    # Initialize self-learning loop
    loop = SelfLearningLoop()
    
    # Start the loop
    await loop.start()
    
    # Create some experiences
    for i in range(5):
        experience = create_experience(
            content={'task': f'task_{i}', 'result': 'success'},
            exp_type=ExperienceType.TASK,
            importance=0.7 + i * 0.05
        )
        await loop.ingest_experience(experience)
    
    # Create some knowledge units
    for i in range(3):
        knowledge = await loop.create_knowledge_unit(
            domain=f'domain_{i}',
            summary=f'Knowledge about topic {i}',
            concepts=[f'concept_{i}_a', f'concept_{i}_b'],
            confidence=0.8
        )
    
    # Get stats
    stats = await loop.get_learning_stats()
    print("Learning Stats:", json.dumps(stats, indent=2, default=str))
    
    # Query knowledge
    results = await loop.query_knowledge('topic', top_k=5)
    print(f"\nQuery results: {len(results)} knowledge units found")
    
    # Assess retention
    if loop.knowledge_store:
        first_knowledge_id = list(loop.knowledge_store.keys())[0]
        retention = await loop.assess_retention(first_knowledge_id)
        print(f"\nRetention assessment: {retention}")
    
    # Run for a bit
    await asyncio.sleep(5)
    
    # Stop the loop
    await loop.stop()
    
    print("\nDemo completed successfully!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())
