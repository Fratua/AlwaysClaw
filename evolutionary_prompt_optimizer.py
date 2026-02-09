"""
Advanced Context Prompt Engineering Loop
Evolutionary Prompt Optimization with Genetic Algorithms

This module implements a complete genetic algorithm framework for
autonomous prompt optimization in AI agent systems.

Author: AI Systems Architecture Team
Version: 1.0
"""

import random
import numpy as np
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib


# =============================================================================
# GENE AND CHROMOSOME DEFINITIONS
# =============================================================================

class GeneType(Enum):
    """Enumeration of prompt gene types."""
    SYSTEM_CONTEXT = "system_context"      # Role definition
    TASK_INSTRUCTION = "task_instruction"  # Core task description
    CONSTRAINT = "constraint"              # Rules and limitations
    EXAMPLE = "example"                    # Few-shot examples
    FORMAT_SPEC = "format_spec"            # Output format
    TOOL_HINT = "tool_hint"                # Tool usage guidance
    REASONING_STEP = "reasoning_step"      # Chain-of-thought steps
    SAFETY_GUARD = "safety_guard"          # Safety constraints
    CONTEXT_VARIABLE = "context_var"       # Dynamic context slots
    STYLE_MODIFIER = "style_modifier"      # Tone/style adjustments


@dataclass
class PromptGene:
    """
    Individual gene representing a prompt component.
    Genes are the atomic units of prompt chromosomes.
    """
    gene_type: GeneType
    content: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)
    position_flexibility: float = 0.5
    required: bool = False
    
    def copy(self) -> 'PromptGene':
        """Create a deep copy of this gene."""
        return PromptGene(
            gene_type=self.gene_type,
            content=self.content,
            weight=self.weight,
            metadata=self.metadata.copy(),
            position_flexibility=self.position_flexibility,
            required=self.required
        )
    
    def to_dict(self) -> Dict:
        """Convert gene to dictionary."""
        return {
            'gene_type': self.gene_type.value,
            'content': self.content,
            'weight': self.weight,
            'metadata': self.metadata,
            'position_flexibility': self.position_flexibility,
            'required': self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptGene':
        """Create gene from dictionary."""
        return cls(
            gene_type=GeneType(data['gene_type']),
            content=data['content'],
            weight=data.get('weight', 1.0),
            metadata=data.get('metadata', {}),
            position_flexibility=data.get('position_flexibility', 0.5),
            required=data.get('required', False)
        )


@dataclass
class PromptChromosome:
    """
    Complete chromosome representing an optimizable prompt.
    Chromosomes contain ordered collections of genes.
    """
    chromosome_id: str
    genes: List[PromptGene]
    gene_order: List[int]
    fitness_score: Optional[float] = None
    generation_created: int = 0
    parent_ids: List[str] = field(default_factory=list)
    task_domain: str = ""
    complexity_score: float = 0.0
    token_count: int = 0
    mutations_applied: List[Tuple] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.chromosome_id:
            self.chromosome_id = str(uuid.uuid4())
        self._update_metadata()
    
    def _update_metadata(self):
        """Update derived metadata."""
        self.token_count = self._estimate_tokens()
        self.complexity_score = self._calculate_complexity()
    
    def _estimate_tokens(self) -> int:
        """Estimate token count (rough approximation)."""
        total_chars = sum(len(gene.content) for gene in self.genes)
        return total_chars // 4  # Roughly 4 chars per token
    
    def _calculate_complexity(self) -> float:
        """Calculate complexity score."""
        if not self.genes:
            return 0.0
        
        # Factors: number of genes, vocabulary diversity, structure depth
        gene_count_factor = min(len(self.genes) / 20, 1.0)
        
        all_text = ' '.join(gene.content for gene in self.genes)
        words = all_text.lower().split()
        unique_words = set(words)
        diversity_factor = len(unique_words) / max(len(words), 1)
        
        return (gene_count_factor + diversity_factor) / 2
    
    def encode(self) -> str:
        """
        Encode chromosome to prompt string.
        Applies gene ordering and assembles final prompt.
        """
        ordered_genes = [self.genes[i] for i in self.gene_order if i < len(self.genes)]
        
        prompt_parts = []
        for gene in ordered_genes:
            if gene.weight >= 0.3:  # Filter very low-weight genes
                part = self._format_gene(gene)
                if gene.weight != 1.0:
                    part = self._apply_weight_emphasis(part, gene.weight)
                prompt_parts.append(part)
        
        return "\n\n".join(prompt_parts)
    
    def _format_gene(self, gene: PromptGene) -> str:
        """Format a gene based on its type."""
        formatters = {
            GeneType.SYSTEM_CONTEXT: lambda g: f"[SYSTEM]\n{g.content}",
            GeneType.TASK_INSTRUCTION: lambda g: f"[TASK]\n{g.content}",
            GeneType.CONSTRAINT: lambda g: f"[CONSTRAINT]\n{g.content}",
            GeneType.EXAMPLE: lambda g: f"[EXAMPLE]\n{g.content}",
            GeneType.FORMAT_SPEC: lambda g: f"[FORMAT]\n{g.content}",
            GeneType.TOOL_HINT: lambda g: f"[TOOL]\n{g.content}",
            GeneType.REASONING_STEP: lambda g: f"[REASONING]\n{g.content}",
            GeneType.SAFETY_GUARD: lambda g: f"[SAFETY]\n{g.content}",
            GeneType.CONTEXT_VARIABLE: lambda g: f"[CONTEXT]\n{g.content}",
            GeneType.STYLE_MODIFIER: lambda g: f"[STYLE]\n{g.content}",
        }
        return formatters.get(gene.gene_type, lambda g: g.content)(gene)
    
    def _apply_weight_emphasis(self, content: str, weight: float) -> str:
        """Apply emphasis based on gene weight."""
        if weight > 1.5:
            return f"!!!IMPORTANT!!!\n{content}\n!!!END IMPORTANT!!!"
        elif weight > 1.2:
            return f"**{content}**"
        elif weight < 0.7:
            return f"({content})"  # De-emphasize
        return content
    
    def distance(self, other: 'PromptChromosome') -> float:
        """Calculate genetic distance between chromosomes."""
        # Jaccard distance on gene content
        self_genes = set((g.gene_type, g.content) for g in self.genes)
        other_genes = set((g.gene_type, g.content) for g in other.genes)
        
        if not self_genes and not other_genes:
            return 0.0
        
        intersection = len(self_genes & other_genes)
        union = len(self_genes | other_genes)
        
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    def similarity(self, other: 'PromptChromosome') -> float:
        """Calculate similarity score."""
        return 1.0 - self.distance(other)
    
    def copy(self) -> 'PromptChromosome':
        """Create a deep copy of this chromosome."""
        return PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=[g.copy() for g in self.genes],
            gene_order=self.gene_order.copy(),
            fitness_score=None,
            generation_created=self.generation_created,
            parent_ids=self.parent_ids.copy(),
            task_domain=self.task_domain,
            mutations_applied=[]
        )
    
    @classmethod
    def decode(cls, prompt_string: str, task_domain: str = "") -> 'PromptChromosome':
        """Decode prompt string to chromosome."""
        # Simple segmentation by double newlines
        segments = [s.strip() for s in prompt_string.split('\n\n') if s.strip()]
        
        genes = []
        for segment in segments:
            gene = cls._parse_segment(segment)
            if gene:
                genes.append(gene)
        
        return cls(
            chromosome_id=str(uuid.uuid4()),
            genes=genes,
            gene_order=list(range(len(genes))),
            task_domain=task_domain
        )
    
    @classmethod
    def _parse_segment(cls, segment: str) -> Optional[PromptGene]:
        """Parse a text segment into a gene."""
        # Detect gene type from markers
        type_markers = {
            '[SYSTEM]': GeneType.SYSTEM_CONTEXT,
            '[TASK]': GeneType.TASK_INSTRUCTION,
            '[CONSTRAINT]': GeneType.CONSTRAINT,
            '[EXAMPLE]': GeneType.EXAMPLE,
            '[FORMAT]': GeneType.FORMAT_SPEC,
            '[TOOL]': GeneType.TOOL_HINT,
            '[REASONING]': GeneType.REASONING_STEP,
            '[SAFETY]': GeneType.SAFETY_GUARD,
            '[CONTEXT]': GeneType.CONTEXT_VARIABLE,
            '[STYLE]': GeneType.STYLE_MODIFIER,
        }
        
        gene_type = GeneType.TASK_INSTRUCTION  # Default
        content = segment
        
        for marker, gtype in type_markers.items():
            if segment.startswith(marker):
                gene_type = gtype
                content = segment[len(marker):].strip()
                break
        
        # Detect emphasis
        weight = 1.0
        if '!!!IMPORTANT!!!' in segment:
            weight = 1.8
        elif segment.startswith('**') and segment.endswith('**'):
            weight = 1.3
            content = content[2:-2]
        elif segment.startswith('(') and segment.endswith(')'):
            weight = 0.6
            content = content[1:-1]
        
        return PromptGene(
            gene_type=gene_type,
            content=content,
            weight=weight
        )


# =============================================================================
# POPULATION MANAGEMENT
# =============================================================================

@dataclass
class Population:
    """Managed population of prompt chromosomes."""
    individuals: List[PromptChromosome]
    generation: int = 0
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    creation_timestamp: float = field(default_factory=time.time)
    task_domain: str = ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate population statistics."""
        fitnesses = [ind.fitness_score for ind in self.individuals 
                    if ind.fitness_score is not None]
        
        return {
            'size': len(self.individuals),
            'generation': self.generation,
            'best_fitness': max(fitnesses) if fitnesses else 0.0,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            'worst_fitness': min(fitnesses) if fitnesses else 0.0,
            'fitness_std': np.std(fitnesses) if fitnesses else 0.0,
            'diversity': self.calculate_diversity(),
        }
    
    def calculate_diversity(self) -> float:
        """Calculate population genetic diversity."""
        if len(self.individuals) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                dist = self.individuals[i].distance(self.individuals[j])
                distances.append(dist)
        
        return sum(distances) / len(distances) if distances else 0.0
    
    def get_best(self) -> Optional[PromptChromosome]:
        """Get the best individual in the population."""
        evaluated = [ind for ind in self.individuals if ind.fitness_score is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda x: x.fitness_score)


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

@dataclass
class TaskResult:
    """Result of executing a task with a prompt."""
    success: bool
    score: float
    input_tokens: int = 0
    output_tokens: int = 0
    execution_time: float = 0.0
    error_message: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class FitnessMetrics:
    """Comprehensive fitness evaluation metrics."""
    task_success_rate: float = 0.0
    task_completion_time: float = 0.0
    error_rate: float = 0.0
    retry_count: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    token_efficiency_ratio: float = 0.0
    response_relevance: float = 0.0
    instruction_following: float = 0.0
    consistency_score: float = 0.0
    safety_score: float = 0.0


class FitnessEvaluator:
    """Multi-objective fitness evaluation engine."""
    
    WEIGHTS = {
        'task_performance': 0.40,
        'token_efficiency': 0.25,
        'response_quality': 0.20,
        'robustness': 0.10,
        'safety': 0.05
    }
    
    def __init__(self, task_executor: Callable[[PromptChromosome, Any], TaskResult]):
        self.task_executor = task_executor
    
    def evaluate(self, chromosome: PromptChromosome, 
                 task_suite: List[Any]) -> float:
        """Comprehensive fitness evaluation."""
        metrics = FitnessMetrics()
        results = []
        
        for task in task_suite:
            result = self.task_executor(chromosome, task)
            results.append(result)
        
        # Calculate metrics
        metrics.task_success_rate = self._calc_success_rate(results)
        metrics.token_efficiency_ratio = self._calc_token_efficiency(results)
        metrics.response_relevance = self._calc_relevance(results)
        metrics.consistency_score = self._calc_consistency(results)
        metrics.safety_score = self._calc_safety(results)
        
        # Compute weighted fitness
        fitness = (
            self.WEIGHTS['task_performance'] * self._performance_score(metrics) +
            self.WEIGHTS['token_efficiency'] * self._efficiency_score(metrics) +
            self.WEIGHTS['response_quality'] * self._quality_score(metrics) +
            self.WEIGHTS['robustness'] * self._robustness_score(metrics) +
            self.WEIGHTS['safety'] * self._safety_score(metrics)
        )
        
        return fitness
    
    def _calc_success_rate(self, results: List[TaskResult]) -> float:
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.success)
        return successes / len(results)
    
    def _calc_token_efficiency(self, results: List[TaskResult]) -> float:
        if not results:
            return 0.0
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
        optimal = 500 * len(results)
        return min(optimal / max(total_tokens, 1), 2.0) / 2.0
    
    def _calc_relevance(self, results: List[TaskResult]) -> float:
        return sum(r.score for r in results) / len(results) if results else 0.0
    
    def _calc_consistency(self, results: List[TaskResult]) -> float:
        if len(results) < 2:
            return 1.0
        scores = [r.score for r in results]
        return 1.0 - min(np.std(scores), 1.0)
    
    def _calc_safety(self, results: List[TaskResult]) -> float:
        return 1.0  # Placeholder
    
    def _performance_score(self, metrics: FitnessMetrics) -> float:
        return (
            0.5 * metrics.task_success_rate +
            0.2 * (1 - min(metrics.error_rate, 1.0)) +
            0.2 * (1 - min(metrics.retry_count / 5, 1.0)) +
            0.1 * (1 - min(metrics.task_completion_time / 60, 1.0))
        )
    
    def _efficiency_score(self, metrics: FitnessMetrics) -> float:
        return metrics.token_efficiency_ratio
    
    def _quality_score(self, metrics: FitnessMetrics) -> float:
        return (
            0.4 * metrics.response_relevance +
            0.4 * metrics.instruction_following +
            0.2 * (1 - min(metrics.safety_score, 1.0))
        )
    
    def _robustness_score(self, metrics: FitnessMetrics) -> float:
        return metrics.consistency_score
    
    def _safety_score(self, metrics: FitnessMetrics) -> float:
        return metrics.safety_score


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

class SelectionOperators:
    """Selection strategies for choosing parent chromosomes."""
    
    @staticmethod
    def tournament_selection(population: List[PromptChromosome],
                            fitness_scores: List[float],
                            tournament_size: int = 5) -> PromptChromosome:
        """Tournament selection: compete random individuals."""
        selected_indices = random.sample(
            range(len(population)), 
            min(tournament_size, len(population))
        )
        best_idx = max(selected_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    @staticmethod
    def rank_based_selection(population: List[PromptChromosome],
                            fitness_scores: List[float]) -> PromptChromosome:
        """Rank-based selection: probability proportional to rank."""
        ranked = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
        n = len(population)
        
        # Linear ranking probabilities
        probabilities = [(2 - 1/n) + 2*(n - rank - 1)*(1 - 1/n)/(n - 1) 
                        for rank in range(n)]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        selected_rank = np.random.choice(n, p=probabilities)
        selected_idx = ranked[selected_rank][0]
        return population[selected_idx]
    
    @staticmethod
    def roulette_selection(population: List[PromptChromosome],
                          fitness_scores: List[float]) -> PromptChromosome:
        """Roulette wheel selection."""
        min_fitness = min(fitness_scores)
        adjusted = [f - min_fitness + 0.01 for f in fitness_scores]
        total = sum(adjusted)
        probabilities = [f / total for f in adjusted]
        
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx]


class CrossoverOperators:
    """Crossover strategies for combining parent chromosomes."""
    
    @staticmethod
    def single_point_crossover(parent1: PromptChromosome,
                               parent2: PromptChromosome,
                               crossover_rate: float = 0.8) -> Tuple[PromptChromosome, PromptChromosome]:
        """Single-point crossover at gene boundaries."""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        min_len = min(len(parent1.genes), len(parent2.genes))
        if min_len < 2:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, min_len - 1)
        
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        
        child1 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=[g.copy() for g in child1_genes],
            gene_order=list(range(len(child1_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        child2 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=[g.copy() for g in child2_genes],
            gene_order=list(range(len(child2_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover(parent1: PromptChromosome,
                         parent2: PromptChromosome,
                         mixing_ratio: float = 0.5) -> Tuple[PromptChromosome, PromptChromosome]:
        """Uniform crossover: gene-by-gene selection."""
        child1_genes = []
        child2_genes = []
        
        max_len = max(len(parent1.genes), len(parent2.genes))
        
        for i in range(max_len):
            if i < len(parent1.genes) and i < len(parent2.genes):
                if random.random() < mixing_ratio:
                    child1_genes.append(parent1.genes[i].copy())
                    child2_genes.append(parent2.genes[i].copy())
                else:
                    child1_genes.append(parent2.genes[i].copy())
                    child2_genes.append(parent1.genes[i].copy())
            elif i < len(parent1.genes):
                child1_genes.append(parent1.genes[i].copy())
            elif i < len(parent2.genes):
                child2_genes.append(parent2.genes[i].copy())
        
        child1 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=child1_genes,
            gene_order=list(range(len(child1_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        child2 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=child2_genes,
            gene_order=list(range(len(child2_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        return child1, child2
    
    @staticmethod
    def semantic_crossover(parent1: PromptChromosome,
                          parent2: PromptChromosome) -> Tuple[PromptChromosome, PromptChromosome]:
        """Semantic-aware crossover preserving coherent meaning."""
        # Group genes by type
        def group_by_type(genes):
            groups = defaultdict(list)
            for gene in genes:
                groups[gene.gene_type].append(gene)
            return groups
        
        p1_by_type = group_by_type(parent1.genes)
        p2_by_type = group_by_type(parent2.genes)
        
        child1_genes = []
        child2_genes = []
        
        all_types = set(p1_by_type.keys()) | set(p2_by_type.keys())
        
        for gene_type in all_types:
            p1_has = gene_type in p1_by_type
            p2_has = gene_type in p2_by_type
            
            if p1_has and p2_has:
                if random.random() < 0.5:
                    child1_genes.extend([g.copy() for g in p1_by_type[gene_type]])
                    child2_genes.extend([g.copy() for g in p2_by_type[gene_type]])
                else:
                    child1_genes.extend([g.copy() for g in p2_by_type[gene_type]])
                    child2_genes.extend([g.copy() for g in p1_by_type[gene_type]])
            elif p1_has:
                child1_genes.extend([g.copy() for g in p1_by_type[gene_type]])
            elif p2_has:
                child2_genes.extend([g.copy() for g in p2_by_type[gene_type]])
        
        # Sort by semantic order
        type_order = list(GeneType)
        child1_genes.sort(key=lambda g: type_order.index(g.gene_type))
        child2_genes.sort(key=lambda g: type_order.index(g.gene_type))
        
        child1 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=child1_genes,
            gene_order=list(range(len(child1_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        child2 = PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=child2_genes,
            gene_order=list(range(len(child2_genes))),
            parent_ids=[parent1.chromosome_id, parent2.chromosome_id]
        )
        
        return child1, child2


class MutationOperators:
    """Mutation strategies for introducing variation."""
    
    MUTATION_TYPES = {
        'gene_substitution': 0.25,
        'gene_insertion': 0.20,
        'gene_deletion': 0.15,
        'gene_reordering': 0.15,
        'weight_adjustment': 0.15,
        'paraphrase': 0.10
    }
    
    def __init__(self, gene_pool: Dict[GeneType, List[str]] = None):
        self.gene_pool = gene_pool or self._default_gene_pool()
    
    def _default_gene_pool(self) -> Dict[GeneType, List[str]]:
        """Create default gene pool."""
        return {
            GeneType.SYSTEM_CONTEXT: [
                "You are a helpful AI assistant.",
                "You are an expert in your field.",
                "You are a precise and thorough assistant."
            ],
            GeneType.CONSTRAINT: [
                "Be concise in your responses.",
                "Provide detailed explanations.",
                "Always verify facts before stating them."
            ],
            GeneType.FORMAT_SPEC: [
                "Respond in JSON format.",
                "Use bullet points for lists.",
                "Provide step-by-step reasoning."
            ],
        }
    
    def mutate(self, chromosome: PromptChromosome,
               mutation_rate: float,
               generation: int) -> PromptChromosome:
        """Apply adaptive mutation to chromosome."""
        adaptive_rate = self._calculate_adaptive_rate(mutation_rate, generation)
        
        mutated_genes = [g.copy() for g in chromosome.genes]
        mutations_applied = []
        
        i = 0
        while i < len(mutated_genes):
            if random.random() < adaptive_rate:
                mutation_type = self._select_mutation_type()
                
                if mutation_type == 'gene_substitution':
                    mutated_genes[i] = self._substitute_gene(mutated_genes[i])
                    mutations_applied.append(('substitution', i))
                    
                elif mutation_type == 'gene_insertion':
                    new_gene = self._generate_random_gene()
                    mutated_genes.insert(i, new_gene)
                    mutations_applied.append(('insertion', i))
                    i += 1
                    
                elif mutation_type == 'gene_deletion':
                    if len(mutated_genes) > 3:
                        del mutated_genes[i]
                        mutations_applied.append(('deletion', i))
                        continue
                        
                elif mutation_type == 'weight_adjustment':
                    mutated_genes[i] = self._adjust_weight(mutated_genes[i])
                    mutations_applied.append(('weight', i))
                    
                elif mutation_type == 'paraphrase':
                    mutated_genes[i] = self._paraphrase_gene(mutated_genes[i])
                    mutations_applied.append(('paraphrase', i))
            i += 1
        
        # Apply reordering mutation
        if random.random() < adaptive_rate * self.MUTATION_TYPES['gene_reordering']:
            mutated_genes = self._reorder_genes(mutated_genes)
            mutations_applied.append(('reordering', None))
        
        return PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=mutated_genes,
            gene_order=list(range(len(mutated_genes))),
            parent_ids=[chromosome.chromosome_id],
            mutations_applied=mutations_applied
        )
    
    def _select_mutation_type(self) -> str:
        """Select mutation type based on probabilities."""
        types = list(self.MUTATION_TYPES.keys())
        weights = list(self.MUTATION_TYPES.values())
        return random.choices(types, weights=weights)[0]
    
    def _calculate_adaptive_rate(self, base_rate: float, generation: int) -> float:
        """Adjust mutation rate based on evolution progress."""
        if generation < 50:
            return base_rate * 1.5
        elif generation < 200:
            return base_rate
        else:
            return base_rate * 0.7
    
    def _substitute_gene(self, gene: PromptGene) -> PromptGene:
        """Replace gene with alternative from gene pool."""
        alternatives = self.gene_pool.get(gene.gene_type, [])
        if alternatives:
            new_content = random.choice(alternatives)
            return PromptGene(
                gene_type=gene.gene_type,
                content=new_content,
                weight=gene.weight,
                position_flexibility=gene.position_flexibility
            )
        return gene.copy()
    
    def _generate_random_gene(self) -> PromptGene:
        """Generate a random gene from the pool."""
        gene_type = random.choice(list(GeneType))
        alternatives = self.gene_pool.get(gene_type, ["Complete the task."])
        return PromptGene(
            gene_type=gene_type,
            content=random.choice(alternatives)
        )
    
    def _adjust_weight(self, gene: PromptGene) -> PromptGene:
        """Adjust gene weight."""
        new_gene = gene.copy()
        new_gene.weight = max(0.1, min(2.0, gene.weight + random.uniform(-0.3, 0.3)))
        return new_gene
    
    def _paraphrase_gene(self, gene: PromptGene) -> PromptGene:
        """Paraphrase gene content using LLM."""
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            response = client.generate(
                f"Rephrase this prompt differently while preserving intent:\n"
                f"{gene.content}",
                max_tokens=200,
            )
            new_gene = gene.copy()
            new_gene.content = response.strip() or gene.content
            return new_gene
        except (ImportError, RuntimeError, EnvironmentError):
            return gene.copy()
    
    def _reorder_genes(self, genes: List[PromptGene]) -> List[PromptGene]:
        """Shuffle gene order while respecting constraints."""
        # Separate required and flexible genes
        required = [(i, g) for i, g in enumerate(genes) if g.required]
        flexible = [(i, g) for i, g in enumerate(genes) if not g.required]
        
        # Shuffle flexible genes
        random.shuffle(flexible)
        
        # Reconstruct
        result = []
        req_idx = flex_idx = 0
        
        for i in range(len(genes)):
            if req_idx < len(required) and required[req_idx][0] == i:
                result.append(required[req_idx][1])
                req_idx += 1
            elif flex_idx < len(flexible):
                result.append(flexible[flex_idx][1])
                flex_idx += 1
        
        return result


# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================

@dataclass
class ConvergenceState:
    """Tracks convergence state of evolution."""
    is_converged: bool = False
    convergence_generation: Optional[int] = None
    convergence_type: Optional[str] = None
    fitness_plateau: bool = False
    diversity_collapse: bool = False
    population_stagnation: bool = False
    final_fitness: float = 0.0
    final_diversity: float = 0.0
    generations_evolved: int = 0


class ConvergenceDetector:
    """Detects various forms of evolutionary convergence."""
    
    def __init__(self, 
                 fitness_improvement_threshold: float = 0.01,
                 fitness_variance_threshold: float = 0.001,
                 diversity_threshold: float = 0.3,
                 stagnation_generations: int = 100,
                 max_generations: int = 1000):
        self.fitness_improvement_threshold = fitness_improvement_threshold
        self.fitness_variance_threshold = fitness_variance_threshold
        self.diversity_threshold = diversity_threshold
        self.stagnation_generations = stagnation_generations
        self.max_generations = max_generations
        
        self.fitness_history = []
        self.diversity_history = []
        self.best_fitness = 0.0
        self.generations_without_improvement = 0
    
    def check_convergence(self, population: Population) -> ConvergenceState:
        """Check if evolution has converged."""
        state = ConvergenceState()
        stats = population.get_statistics()
        
        # Update histories
        self.fitness_history.append(stats['best_fitness'])
        self.diversity_history.append(stats['diversity'])
        
        # Track improvement
        if stats['best_fitness'] > self.best_fitness:
            self.best_fitness = stats['best_fitness']
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        
        # Check conditions
        state.fitness_plateau = self._check_fitness_plateau()
        state.diversity_collapse = self._check_diversity_collapse()
        state.population_stagnation = self._check_stagnation()
        
        # Determine convergence
        state.is_converged = (
            state.fitness_plateau or 
            state.diversity_collapse or
            state.population_stagnation or
            population.generation >= self.max_generations
        )
        
        if state.is_converged:
            state.convergence_generation = population.generation
            state.final_fitness = stats['best_fitness']
            state.final_diversity = stats['diversity']
            state.generations_evolved = population.generation
            
            if state.fitness_plateau:
                state.convergence_type = 'fitness_plateau'
            elif state.diversity_collapse:
                state.convergence_type = 'diversity_collapse'
            elif state.population_stagnation:
                state.convergence_type = 'stagnation'
            else:
                state.convergence_type = 'max_generations'
        
        return state
    
    def _check_fitness_plateau(self, window: int = 50) -> bool:
        if len(self.fitness_history) < window:
            return False
        
        recent = self.fitness_history[-window:]
        improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 0.001)
        variance = np.var(recent)
        
        return (improvement < self.fitness_improvement_threshold and 
                variance < self.fitness_variance_threshold)
    
    def _check_diversity_collapse(self, window: int = 30) -> bool:
        if len(self.diversity_history) < window:
            return False
        
        recent = self.diversity_history[-window:]
        avg_diversity = sum(recent) / len(recent)
        
        return avg_diversity < self.diversity_threshold
    
    def _check_stagnation(self) -> bool:
        return self.generations_without_improvement > self.stagnation_generations


# =============================================================================
# ELITE PRESERVATION
# =============================================================================

@dataclass
class EliteIndividual:
    """Elite individual with extended metadata."""
    chromosome: PromptChromosome
    fitness_score: float
    generation_discovered: int
    task_successes: int = 0
    task_attempts: int = 0
    parent_ids: List[str] = field(default_factory=list)
    offspring_count: int = 0
    validation_fitness: Optional[float] = None
    cross_validation_scores: List[float] = field(default_factory=list)
    overfitting_warning: bool = False
    
    def success_rate(self) -> float:
        if self.task_attempts == 0:
            return 0.0
        return self.task_successes / self.task_attempts


class ElitePreservation:
    """Manages elite individuals across generations."""
    
    def __init__(self, elite_size: int = 10, hall_of_fame_size: int = 50):
        self.elite_size = elite_size
        self.hall_of_fame_size = hall_of_fame_size
        self.elites: List[EliteIndividual] = []
        self.hall_of_fame: List[EliteIndividual] = []
    
    def update_elites(self, population: Population) -> None:
        """Update elite list with current population."""
        current_elites = self._select_population_elites(population)
        combined = self.elites + current_elites
        unique_elites = self._remove_duplicates(combined)
        
        sorted_elites = sorted(unique_elites, 
                              key=lambda e: e.fitness_score, 
                              reverse=True)
        
        self.elites = sorted_elites[:self.elite_size]
        self._update_hall_of_fame()
    
    def _select_population_elites(self, population: Population) -> List[EliteIndividual]:
        """Select elites from current population."""
        sorted_individuals = sorted(
            population.individuals,
            key=lambda x: x.fitness_score or 0,
            reverse=True
        )
        
        elites = []
        for ind in sorted_individuals[:self.elite_size]:
            if ind.fitness_score is not None:
                elite = EliteIndividual(
                    chromosome=ind.copy(),
                    fitness_score=ind.fitness_score,
                    generation_discovered=population.generation,
                    parent_ids=ind.parent_ids.copy()
                )
                elites.append(elite)
        
        return elites
    
    def _remove_duplicates(self, elites: List[EliteIndividual],
                          similarity_threshold: float = 0.95) -> List[EliteIndividual]:
        """Remove genetically similar elites."""
        unique = []
        
        for elite in elites:
            is_duplicate = False
            for existing in unique:
                similarity = elite.chromosome.similarity(existing.chromosome)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    if elite.fitness_score > existing.fitness_score:
                        unique[unique.index(existing)] = elite
                    break
            
            if not is_duplicate:
                unique.append(elite)
        
        return unique
    
    def _update_hall_of_fame(self) -> None:
        """Update hall of fame with all-time best."""
        for elite in self.elites:
            if not any(e.chromosome.chromosome_id == elite.chromosome.chromosome_id 
                      for e in self.hall_of_fame):
                self.hall_of_fame.append(elite)
        
        self.hall_of_fame = sorted(self.hall_of_fame,
                                   key=lambda e: e.fitness_score,
                                   reverse=True)[:self.hall_of_fame_size]
    
    def get_elite_for_breeding(self, probability: float = 0.2) -> Optional[PromptChromosome]:
        """Select elite for breeding with probability."""
        if not self.elites or random.random() > probability:
            return None
        
        fitnesses = [e.fitness_score for e in self.elites]
        total = sum(fitnesses)
        
        if total > 0:
            probabilities = [f / total for f in fitnesses]
            selected = np.random.choice(len(self.elites), p=probabilities)
            return self.elites[selected].chromosome.copy()
        
        return None
    
    def get_best_elite(self) -> Optional[EliteIndividual]:
        """Get the best elite individual."""
        if not self.elites:
            return None
        return max(self.elites, key=lambda e: e.fitness_score)


# =============================================================================
# MAIN EVOLUTIONARY OPTIMIZER
# =============================================================================

class EvolutionaryPromptOptimizer:
    """
    Main genetic algorithm orchestrator for prompt optimization.
    Implements generational evolution with adaptive parameters.
    """
    
    DEFAULT_CONFIG = {
        'population_size': 100,
        'max_generations': 1000,
        'elite_ratio': 0.1,
        'crossover_rate': 0.8,
        'mutation_rate': 0.15,
        'adaptive_mutation': True,
        'tournament_size': 5,
        'convergence_threshold': 0.95,
        'diversity_threshold': 0.3,
        'prompt_complexity_limit': 4096,
        'selection_strategy': 'tournament',
        'crossover_strategy': 'semantic',
    }
    
    def __init__(self, 
                 task_executor: Callable[[PromptChromosome, Any], TaskResult],
                 config: Dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.task_executor = task_executor
        
        # Initialize components
        self.fitness_evaluator = FitnessEvaluator(task_executor)
        self.selection = SelectionOperators()
        self.crossover = CrossoverOperators()
        self.mutation = MutationOperators()
        self.convergence_detector = ConvergenceDetector(
            max_generations=self.config['max_generations']
        )
        self.elite_preservation = ElitePreservation(
            elite_size=int(self.config['population_size'] * self.config['elite_ratio'])
        )
        
        self.population = None
        self.generation = 0
        self.history = []
    
    def initialize_population(self, task_domain: str, 
                             seed_prompts: List[str] = None) -> Population:
        """Initialize population with seed prompts or random individuals."""
        individuals = []
        
        if seed_prompts:
            for prompt in seed_prompts:
                chrom = PromptChromosome.decode(prompt, task_domain)
                individuals.append(chrom)
        
        # Fill with random individuals
        while len(individuals) < self.config['population_size']:
            chrom = self._generate_random_chromosome(task_domain)
            individuals.append(chrom)
        
        self.population = Population(
            individuals=individuals,
            task_domain=task_domain
        )
        
        return self.population
    
    def _generate_random_chromosome(self, task_domain: str) -> PromptChromosome:
        """Generate a random chromosome."""
        genes = []
        
        # Add system context
        genes.append(PromptGene(
            gene_type=GeneType.SYSTEM_CONTEXT,
            content="You are a helpful AI assistant.",
            weight=1.0
        ))
        
        # Add task instruction
        genes.append(PromptGene(
            gene_type=GeneType.TASK_INSTRUCTION,
            content=f"Complete the {task_domain} task efficiently.",
            weight=1.2
        ))
        
        # Add random constraints
        if random.random() < 0.5:
            genes.append(PromptGene(
                gene_type=GeneType.CONSTRAINT,
                content="Be concise and accurate.",
                weight=0.8
            ))
        
        return PromptChromosome(
            chromosome_id=str(uuid.uuid4()),
            genes=genes,
            gene_order=list(range(len(genes))),
            task_domain=task_domain
        )
    
    def evolve(self, task_suite: List[Any], 
               generations: int = None) -> PromptChromosome:
        """
        Run evolution for specified generations or until convergence.
        
        Args:
            task_suite: List of tasks for fitness evaluation
            generations: Max generations (uses config default if None)
            
        Returns:
            Best chromosome found
        """
        max_gen = generations or self.config['max_generations']
        
        for gen in range(max_gen):
            self.generation = gen
            
            # Evaluate population
            self._evaluate_population(task_suite)
            
            # Update elites
            self.elite_preservation.update_elites(self.population)
            
            # Log statistics
            stats = self.population.get_statistics()
            self.history.append(stats)
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best={stats['best_fitness']:.4f}, "
                      f"Avg={stats['avg_fitness']:.4f}, "
                      f"Diversity={stats['diversity']:.4f}")
            
            # Check convergence
            conv_state = self.convergence_detector.check_convergence(self.population)
            if conv_state.is_converged:
                print(f"Converged at generation {gen}: {conv_state.convergence_type}")
                break
            
            # Create next generation
            self._create_next_generation()
        
        # Return best individual
        best = self.population.get_best()
        if best:
            print(f"\nFinal best fitness: {best.fitness_score:.4f}")
        return best
    
    def _evaluate_population(self, task_suite: List[Any]) -> None:
        """Evaluate all individuals in population."""
        for individual in self.population.individuals:
            if individual.fitness_score is None:
                individual.fitness_score = self.fitness_evaluator.evaluate(
                    individual, task_suite
                )
    
    def _create_next_generation(self) -> None:
        """Create next generation through selection, crossover, mutation."""
        new_individuals = []
        
        # Elite preservation
        elite_count = int(self.config['elite_ratio'] * len(self.population.individuals))
        elites = self._select_elites(elite_count)
        new_individuals.extend([e.copy() for e in elites])
        
        # Generate offspring
        fitness_scores = [ind.fitness_score or 0 for ind in self.population.individuals]
        
        while len(new_individuals) < len(self.population.individuals):
            # Selection
            parent1 = self._select_parent(fitness_scores)
            parent2 = self._select_parent(fitness_scores)
            
            # Crossover
            if random.random() < self.config['crossover_rate']:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self.mutation.mutate(
                child1, 
                self.config['mutation_rate'],
                self.generation
            )
            child2 = self.mutation.mutate(
                child2,
                self.config['mutation_rate'],
                self.generation
            )
            
            new_individuals.extend([child1, child2])
        
        # Trim to population size
        new_individuals = new_individuals[:len(self.population.individuals)]
        
        # Update population
        self.population = Population(
            individuals=new_individuals,
            generation=self.population.generation + 1,
            fitness_history=self.population.fitness_history + [max(fitness_scores)],
            diversity_history=self.population.diversity_history + [self.population.calculate_diversity()],
            task_domain=self.population.task_domain
        )
    
    def _select_elites(self, count: int) -> List[PromptChromosome]:
        """Select top individuals as elites."""
        sorted_pop = sorted(
            self.population.individuals,
            key=lambda x: x.fitness_score or 0,
            reverse=True
        )
        return sorted_pop[:count]
    
    def _select_parent(self, fitness_scores: List[float]) -> PromptChromosome:
        """Select a parent using configured strategy."""
        # Try elite breeding
        elite = self.elite_preservation.get_elite_for_breeding()
        if elite:
            return elite
        
        # Use selection strategy
        strategy = self.config['selection_strategy']
        
        if strategy == 'tournament':
            return self.selection.tournament_selection(
                self.population.individuals,
                fitness_scores,
                self.config['tournament_size']
            )
        elif strategy == 'rank':
            return self.selection.rank_based_selection(
                self.population.individuals,
                fitness_scores
            )
        else:
            return self.selection.roulette_selection(
                self.population.individuals,
                fitness_scores
            )
    
    def _crossover(self, parent1: PromptChromosome, 
                   parent2: PromptChromosome) -> Tuple[PromptChromosome, PromptChromosome]:
        """Apply crossover using configured strategy."""
        strategy = self.config['crossover_strategy']
        
        if strategy == 'semantic':
            return self.crossover.semantic_crossover(parent1, parent2)
        elif strategy == 'uniform':
            return self.crossover.uniform_crossover(parent1, parent2)
        else:
            return self.crossover.single_point_crossover(parent1, parent2)
    
    def get_best_prompt(self) -> Optional[str]:
        """Get the best prompt as a string."""
        best = self.population.get_best()
        return best.encode() if best else None
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        return {
            'generations': self.generation,
            'population_stats': self.population.get_statistics() if self.population else {},
            'elite_count': len(self.elite_preservation.elites),
            'hall_of_fame_count': len(self.elite_preservation.hall_of_fame),
            'history': self.history
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_task_executor(chromosome: PromptChromosome, task: Any) -> TaskResult:
    """Example task executor for demonstration."""
    # Simulate task execution
    prompt = chromosome.encode()
    
    # Simulate success based on prompt quality (length, structure)
    score = min(len(prompt) / 500, 1.0) * random.uniform(0.7, 1.0)
    success = score > 0.6
    
    return TaskResult(
        success=success,
        score=score,
        input_tokens=len(prompt) // 4,
        output_tokens=50,
        execution_time=random.uniform(0.1, 1.0)
    )


def main():
    """Example usage of the evolutionary prompt optimizer."""
    # Create optimizer
    optimizer = EvolutionaryPromptOptimizer(
        task_executor=example_task_executor,
        config={
            'population_size': 30,
            'max_generations': 50,
            'mutation_rate': 0.2
        }
    )
    
    # Initialize with seed prompts
    seed_prompts = [
        """[SYSTEM]
You are an AI assistant specialized in task completion.

[TASK]
Complete the assigned task efficiently and accurately.

[CONSTRAINT]
Be concise in your responses.""",
        
        """[SYSTEM]
You are a helpful assistant with expertise in multiple domains.

[TASK]
Analyze and complete the given task with attention to detail.

[FORMAT]
Provide structured output."""
    ]
    
    optimizer.initialize_population("general", seed_prompts)
    
    # Create dummy task suite
    task_suite = [{'id': i, 'type': 'general'} for i in range(10)]
    
    # Run evolution
    best_chromosome = optimizer.evolve(task_suite, generations=30)
    
    # Output results
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"\nBest Fitness: {best_chromosome.fitness_score:.4f}")
    print(f"\nOptimized Prompt:\n{best_chromosome.encode()}")
    
    # Print statistics
    stats = optimizer.get_statistics()
    print(f"\nTotal Generations: {stats['generations']}")
    print(f"Elite Pool Size: {stats['elite_count']}")


if __name__ == "__main__":
    main()
