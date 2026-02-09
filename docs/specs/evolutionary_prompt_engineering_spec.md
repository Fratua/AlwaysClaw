# ADVANCED CONTEXT PROMPT ENGINEERING LOOP
## Evolutionary Prompt Optimization with Genetic Algorithms
### Technical Specification v1.0

---

## EXECUTIVE SUMMARY

The Advanced Context Prompt Engineering Loop (ACPEL) implements a sophisticated genetic algorithm framework for autonomous prompt optimization in AI agent systems. This system enables continuous, self-improving prompt evolution through evolutionary computation principles, enabling the AI agent to adapt and optimize its prompting strategies based on task performance metrics.

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED CONTEXT PROMPT ENGINEERING LOOP                 │
│                      Evolutionary Optimization Core                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   Prompt     │───▶│  Chromosome  │───▶│  Population  │───▶│  Fitness │  │
│  │   Encoder    │    │  Generator   │    │  Manager     │    │  Evaluator│  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│         │                   │                   │                │         │
│         ▼                   ▼                   ▼                ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   Prompt     │◀───│   Genetic    │◀───│  Selection   │◀───│  Fitness │  │
│  │   Decoder    │    │   Operators  │    │  Engine      │    │  Scores  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONVERGENCE & ELITE MANAGEMENT                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Convergence  │  │   Elite      │  │   Prompt     │              │   │
│  │  │  Detector    │  │  Preservation│  │   Archive    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. GENETIC ALGORITHM IMPLEMENTATION

### 2.1 Core Algorithm Flow

```python
class EvolutionaryPromptOptimizer:
    """
    Main genetic algorithm orchestrator for prompt optimization.
    Implements generational evolution with adaptive parameters.
    """
    
    ALGORITHM_CONFIG = {
        'population_size': 100,           # Initial population size
        'max_generations': 1000,          # Maximum evolution generations
        'elite_ratio': 0.1,               # Top performers preserved
        'crossover_rate': 0.8,            # Probability of crossover
        'mutation_rate': 0.15,            # Base mutation probability
        'adaptive_mutation': True,        # Enable adaptive mutation
        'tournament_size': 5,             # Selection tournament size
        'convergence_threshold': 0.95,    # Fitness convergence threshold
        'diversity_threshold': 0.3,       # Population diversity minimum
        'prompt_complexity_limit': 4096,  # Max tokens per prompt
    }
    
    def evolve(self, task_domain: str, initial_prompts: List[str] = None):
        """
        Main evolution loop for prompt optimization.
        
        Args:
            task_domain: Target task category (e.g., 'email', 'browser', 'system')
            initial_prompts: Optional seed prompts for initialization
            
        Returns:
            Optimized prompt with highest fitness score
        """
        # Initialize population
        population = self._initialize_population(task_domain, initial_prompts)
        
        generation = 0
        best_fitness_history = []
        diversity_history = []
        
        while not self._should_terminate(generation, best_fitness_history, diversity_history):
            # Evaluate fitness for all individuals
            fitness_scores = self._evaluate_population(population, task_domain)
            
            # Track statistics
            best_fitness = max(fitness_scores)
            diversity = self._calculate_diversity(population)
            best_fitness_history.append(best_fitness)
            diversity_history.append(diversity)
            
            # Log generation statistics
            self._log_generation_stats(generation, population, fitness_scores)
            
            # Create next generation
            new_population = self._create_next_generation(
                population, 
                fitness_scores,
                generation
            )
            
            population = new_population
            generation += 1
            
        # Return best individual from final population
        return self._get_best_individual(population, task_domain)
```

### 2.2 Evolution Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Population Size | 100 | 50-500 | Number of prompts in each generation |
| Elite Ratio | 0.1 | 0.05-0.2 | Percentage of top performers preserved |
| Crossover Rate | 0.8 | 0.5-0.95 | Probability of genetic crossover |
| Mutation Rate | 0.15 | 0.05-0.3 | Base probability of mutation |
| Tournament Size | 5 | 3-10 | Selection pressure control |
| Max Generations | 1000 | 100-5000 | Evolution termination limit |
| Convergence Window | 50 | 20-100 | Generations for convergence check |

---

## 3. PROMPT ENCODING (CHROMOSOME REPRESENTATION)

### 3.1 Gene Structure

```python
@dataclass
class PromptGene:
    """
    Individual gene representing a prompt component.
    Genes are the atomic units of prompt chromosomes.
    """
    gene_type: GeneType           # Type of prompt component
    content: str                  # Actual text content
    weight: float = 1.0          # Importance weight (0.0-2.0)
    metadata: Dict = field(default_factory=dict)
    
    # Gene-specific attributes
    position_flexibility: float = 0.5  # Can gene move within chromosome?
    required: bool = False             # Is gene mandatory?
    
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
```

### 3.2 Chromosome Structure

```python
@dataclass  
class PromptChromosome:
    """
    Complete chromosome representing an optimizable prompt.
    Chromosomes contain ordered collections of genes.
    """
    chromosome_id: str
    genes: List[PromptGene]
    gene_order: List[int]           # Permutation indices
    fitness_score: Optional[float] = None
    generation_created: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # Chromosome metadata
    task_domain: str = ""
    complexity_score: float = 0.0
    token_count: int = 0
    
    def encode(self) -> str:
        """
        Encode chromosome to prompt string.
        Applies gene ordering and assembles final prompt.
        """
        ordered_genes = [self.genes[i] for i in self.gene_order]
        
        # Apply gene weights to influence prominence
        prompt_parts = []
        for gene in ordered_genes:
            if gene.weight >= 0.5:  # Filter low-weight genes
                part = self._format_gene(gene)
                if gene.weight != 1.0:
                    part = self._apply_weight_emphasis(part, gene.weight)
                prompt_parts.append(part)
        
        return "\n\n".join(prompt_parts)
    
    def decode(self, prompt_string: str) -> 'PromptChromosome':
        """
        Decode prompt string back to chromosome.
        Uses NLP parsing to identify gene boundaries and types.
        """
        # Segment prompt into potential genes
        segments = self._segment_prompt(prompt_string)
        
        genes = []
        for segment in segments:
            gene_type = self._classify_segment(segment)
            weight = self._estimate_importance(segment)
            genes.append(PromptGene(
                gene_type=gene_type,
                content=segment,
                weight=weight
            ))
        
        return PromptChromosome(
            chromosome_id=generate_uuid(),
            genes=genes,
            gene_order=list(range(len(genes)))
        )
```

### 3.3 Encoding Schemes

```python
class PromptEncodingSchemes:
    """
    Multiple encoding strategies for different prompt types.
    """
    
    @staticmethod
    def structural_encoding(prompt: str) -> Dict:
        """
        Encode prompt based on structural components.
        """
        return {
            'sections': {
                'header': extract_header(prompt),
                'context': extract_context(prompt),
                'instructions': extract_instructions(prompt),
                'examples': extract_examples(prompt),
                'constraints': extract_constraints(prompt),
                'footer': extract_footer(prompt)
            },
            'markers': identify_markers(prompt),
            'delimiters': extract_delimiters(prompt),
            'hierarchy_depth': calculate_hierarchy(prompt)
        }
    
    @staticmethod
    def semantic_encoding(prompt: str) -> Dict:
        """
        Encode prompt based on semantic meaning.
        """
        return {
            'intent_vector': embed_intent(prompt),
            'task_category': classify_task(prompt),
            'complexity_metrics': {
                'vocabulary_diversity': calculate_diversity(prompt),
                'syntactic_complexity': analyze_syntax(prompt),
                'instruction_count': count_instructions(prompt),
                'constraint_count': count_constraints(prompt)
            },
            'semantic_clusters': cluster_semantics(prompt)
        }
    
    @staticmethod
    def functional_encoding(prompt: str) -> Dict:
        """
        Encode prompt based on functional components.
        """
        return {
            'operations': extract_operations(prompt),
            'data_transforms': identify_transforms(prompt),
            'control_flow': parse_control_structure(prompt),
            'variable_bindings': extract_variables(prompt),
            'tool_invocations': identify_tool_calls(prompt)
        }
```

---

## 4. FITNESS FUNCTION DESIGN

### 4.1 Multi-Objective Fitness Framework

```python
@dataclass
class FitnessMetrics:
    """
    Comprehensive fitness evaluation metrics.
    """
    # Task Performance (40% weight)
    task_success_rate: float = 0.0
    task_completion_time: float = 0.0
    error_rate: float = 0.0
    retry_count: float = 0.0
    
    # Token Efficiency (25% weight)
    input_tokens: int = 0
    output_tokens: int = 0
    token_efficiency_ratio: float = 0.0
    prompt_compression_ratio: float = 0.0
    
    # Response Quality (20% weight)
    response_relevance: float = 0.0
    instruction_following: float = 0.0
    output_format_adherence: float = 0.0
    hallucination_score: float = 0.0
    
    # Robustness (10% weight)
    edge_case_handling: float = 0.0
    consistency_score: float = 0.0
    generalization_score: float = 0.0
    
    # Safety (5% weight)
    safety_score: float = 0.0
    policy_compliance: float = 0.0

class FitnessEvaluator:
    """
    Multi-objective fitness evaluation engine.
    """
    
    WEIGHTS = {
        'task_performance': 0.40,
        'token_efficiency': 0.25,
        'response_quality': 0.20,
        'robustness': 0.10,
        'safety': 0.05
    }
    
    def evaluate(self, chromosome: PromptChromosome, 
                 task_suite: List[Task]) -> float:
        """
        Comprehensive fitness evaluation across multiple dimensions.
        """
        metrics = FitnessMetrics()
        
        # Evaluate on task suite
        results = []
        for task in task_suite:
            result = self._execute_task(chromosome, task)
            results.append(result)
        
        # Calculate component scores
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
    
    def _performance_score(self, metrics: FitnessMetrics) -> float:
        """Calculate task performance component."""
        return (
            0.5 * metrics.task_success_rate +
            0.2 * (1 - min(metrics.error_rate, 1.0)) +
            0.2 * (1 - min(metrics.retry_count / 5, 1.0)) +
            0.1 * (1 - min(metrics.task_completion_time / 60, 1.0))
        )
    
    def _efficiency_score(self, metrics: FitnessMetrics) -> float:
        """Calculate token efficiency component."""
        optimal_tokens = 500  # Target token count
        actual_tokens = metrics.input_tokens + metrics.output_tokens
        
        token_ratio = min(optimal_tokens / max(actual_tokens, 1), 2.0) / 2.0
        compression_bonus = min(metrics.prompt_compression_ratio, 2.0) / 2.0
        
        return 0.7 * token_ratio + 0.3 * compression_bonus
```

### 4.2 Task-Specific Fitness Functions

```python
class DomainSpecificFitness:
    """
    Specialized fitness functions for different agent capabilities.
    """
    
    @staticmethod
    def email_fitness(result: TaskResult) -> float:
        """Fitness for email handling prompts."""
        scores = {
            'recipient_correct': result.recipient_accuracy,
            'subject_appropriate': result.subject_relevance,
            'tone_match': result.tone_alignment,
            'content_complete': result.content_coverage,
            'format_proper': result.format_compliance,
            'attachments_handled': result.attachment_success
        }
        weights = [0.25, 0.15, 0.20, 0.25, 0.10, 0.05]
        return sum(s * w for s, w in zip(scores.values(), weights))
    
    @staticmethod
    def browser_fitness(result: TaskResult) -> float:
        """Fitness for browser control prompts."""
        scores = {
            'navigation_success': result.nav_success,
            'element_interaction': result.element_accuracy,
            'data_extraction': result.extraction_quality,
            'action_sequence': result.sequence_correctness,
            'error_recovery': result.recovery_success,
            'page_load_efficiency': result.load_efficiency
        }
        weights = [0.20, 0.25, 0.25, 0.15, 0.10, 0.05]
        return sum(s * w for s, w in zip(scores.values(), weights))
    
    @staticmethod
    def voice_fitness(result: TaskResult) -> float:
        """Fitness for voice interaction prompts."""
        scores = {
            'intent_recognition': result.intent_accuracy,
            'response_naturalness': result.naturalness_score,
            'latency_acceptable': result.latency_score,
            'context_retention': result.context_score,
            'interruption_handling': result.interruption_score,
            'barge_in_support': result.barge_in_success
        }
        weights = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
        return sum(s * w for s, w in zip(scores.values(), weights))
```

---

## 5. GENETIC OPERATORS

### 5.1 Selection Operators

```python
class SelectionOperators:
    """
    Selection strategies for choosing parent chromosomes.
    """
    
    @staticmethod
    def tournament_selection(population: List[PromptChromosome],
                            fitness_scores: List[float],
                            tournament_size: int = 5) -> PromptChromosome:
        """
        Tournament selection: compete random individuals.
        """
        selected_indices = random.sample(
            range(len(population)), 
            min(tournament_size, len(population))
        )
        
        best_idx = max(selected_indices, 
                      key=lambda i: fitness_scores[i])
        
        return population[best_idx]
    
    @staticmethod
    def rank_based_selection(population: List[PromptChromosome],
                            fitness_scores: List[float]) -> PromptChromosome:
        """
        Rank-based selection: probability proportional to rank.
        """
        # Sort by fitness and assign ranks
        ranked = sorted(enumerate(fitness_scores), 
                       key=lambda x: x[1], reverse=True)
        
        # Rank probabilities (linear ranking)
        n = len(population)
        probabilities = [(2 - 1/n) + 2*(n - rank - 1)*(1 - 1/n)/(n - 1) 
                        for rank in range(n)]
        
        # Normalize
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Select based on rank probabilities
        selected_rank = np.random.choice(n, p=probabilities)
        selected_idx = ranked[selected_rank][0]
        
        return population[selected_idx]
    
    @staticmethod
    def diversity_aware_selection(population: List[PromptChromosome],
                                 fitness_scores: List[float],
                                 diversity_weight: float = 0.3) -> PromptChromosome:
        """
        Selection considering both fitness and diversity.
        """
        # Calculate diversity contribution for each individual
        diversity_scores = []
        for i, chrom in enumerate(population):
            distances = [chrom.distance(population[j]) 
                        for j in range(len(population)) if j != i]
            avg_distance = sum(distances) / len(distances) if distances else 0
            diversity_scores.append(avg_distance)
        
        # Normalize scores
        norm_fitness = normalize(fitness_scores)
        norm_diversity = normalize(diversity_scores)
        
        # Combined score
        combined_scores = [
            (1 - diversity_weight) * f + diversity_weight * d
            for f, d in zip(norm_fitness, norm_diversity)
        ]
        
        # Select based on combined score
        selected_idx = max(range(len(population)), 
                          key=lambda i: combined_scores[i])
        
        return population[selected_idx]
```

### 5.2 Crossover Operators

```python
class CrossoverOperators:
    """
    Crossover strategies for combining parent chromosomes.
    """
    
    @staticmethod
    def single_point_crossover(parent1: PromptChromosome,
                               parent2: PromptChromosome,
                               crossover_rate: float = 0.8) -> Tuple[PromptChromosome, PromptChromosome]:
        """
        Single-point crossover at gene boundaries.
        """
        if random.random() > crossover_rate:
            return parent1, parent2
        
        # Choose crossover point
        min_len = min(len(parent1.genes), len(parent2.genes))
        if min_len < 2:
            return parent1, parent2
        
        point = random.randint(1, min_len - 1)
        
        # Create offspring
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        
        child1 = create_chromosome(child1_genes, [parent1, parent2])
        child2 = create_chromosome(child2_genes, [parent1, parent2])
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover(parent1: PromptChromosome,
                         parent2: PromptChromosome,
                         mixing_ratio: float = 0.5) -> Tuple[PromptChromosome, PromptChromosome]:
        """
        Uniform crossover: gene-by-gene selection.
        """
        child1_genes = []
        child2_genes = []
        
        max_len = max(len(parent1.genes), len(parent2.genes))
        
        for i in range(max_len):
            if i < len(parent1.genes) and i < len(parent2.genes):
                # Both parents have gene at this position
                if random.random() < mixing_ratio:
                    child1_genes.append(parent1.genes[i])
                    child2_genes.append(parent2.genes[i])
                else:
                    child1_genes.append(parent2.genes[i])
                    child2_genes.append(parent1.genes[i])
            elif i < len(parent1.genes):
                # Only parent1 has gene
                child1_genes.append(parent1.genes[i])
            elif i < len(parent2.genes):
                # Only parent2 has gene
                child2_genes.append(parent2.genes[i])
        
        child1 = create_chromosome(child1_genes, [parent1, parent2])
        child2 = create_chromosome(child2_genes, [parent1, parent2])
        
        return child1, child2
    
    @staticmethod
    def semantic_crossover(parent1: PromptChromosome,
                          parent2: PromptChromosome) -> Tuple[PromptChromosome, PromptChromosome]:
        """
        Semantic-aware crossover preserving coherent meaning.
        """
        # Group genes by semantic type
        p1_by_type = group_by_type(parent1.genes)
        p2_by_type = group_by_type(parent2.genes)
        
        child1_genes = []
        child2_genes = []
        
        # For each gene type, choose source parent
        for gene_type in GeneType:
            if gene_type in p1_by_type and gene_type in p2_by_type:
                # Mix genes of this type
                if random.random() < 0.5:
                    child1_genes.extend(p1_by_type[gene_type])
                    child2_genes.extend(p2_by_type[gene_type])
                else:
                    child1_genes.extend(p2_by_type[gene_type])
                    child2_genes.extend(p1_by_type[gene_type])
            elif gene_type in p1_by_type:
                child1_genes.extend(p1_by_type[gene_type])
            elif gene_type in p2_by_type:
                child2_genes.extend(p2_by_type[gene_type])
        
        # Maintain logical ordering
        child1_genes = sort_by_semantic_order(child1_genes)
        child2_genes = sort_by_semantic_order(child2_genes)
        
        child1 = create_chromosome(child1_genes, [parent1, parent2])
        child2 = create_chromosome(child2_genes, [parent1, parent2])
        
        return child1, child2
```

### 5.3 Mutation Operators

```python
class MutationOperators:
    """
    Mutation strategies for introducing variation.
    """
    
    MUTATION_TYPES = {
        'gene_substitution': 0.25,
        'gene_insertion': 0.20,
        'gene_deletion': 0.15,
        'gene_reordering': 0.15,
        'weight_adjustment': 0.15,
        'paraphrase': 0.10
    }
    
    def mutate(self, chromosome: PromptChromosome,
               mutation_rate: float,
               generation: int) -> PromptChromosome:
        """
        Apply adaptive mutation to chromosome.
        """
        # Adaptive mutation rate based on generation progress
        adaptive_rate = self._calculate_adaptive_rate(
            mutation_rate, generation
        )
        
        mutated_genes = list(chromosome.genes)
        mutations_applied = []
        
        for i in range(len(mutated_genes)):
            if random.random() < adaptive_rate:
                mutation_type = self._select_mutation_type()
                
                if mutation_type == 'gene_substitution':
                    mutated_genes[i] = self._substitute_gene(mutated_genes[i])
                    mutations_applied.append(('substitution', i))
                    
                elif mutation_type == 'gene_insertion':
                    new_gene = self._generate_random_gene()
                    mutated_genes.insert(i, new_gene)
                    mutations_applied.append(('insertion', i))
                    
                elif mutation_type == 'gene_deletion':
                    if len(mutated_genes) > 3:  # Maintain minimum size
                        del mutated_genes[i]
                        mutations_applied.append(('deletion', i))
                        
                elif mutation_type == 'weight_adjustment':
                    mutated_genes[i] = self._adjust_weight(mutated_genes[i])
                    mutations_applied.append(('weight', i))
                    
                elif mutation_type == 'paraphrase':
                    mutated_genes[i] = self._paraphrase_gene(mutated_genes[i])
                    mutations_applied.append(('paraphrase', i))
        
        # Apply reordering mutation separately
        if random.random() < adaptive_rate * self.MUTATION_TYPES['gene_reordering']:
            mutated_genes = self._reorder_genes(mutated_genes)
            mutations_applied.append(('reordering', None))
        
        return create_chromosome(mutated_genes, [chromosome], mutations_applied)
    
    def _substitute_gene(self, gene: PromptGene) -> PromptGene:
        """Replace gene with alternative from gene pool."""
        alternatives = self.gene_pool.get_alternatives(gene.gene_type)
        if alternatives:
            new_content = random.choice(alternatives)
            return PromptGene(
                gene_type=gene.gene_type,
                content=new_content,
                weight=gene.weight,
                position_flexibility=gene.position_flexibility
            )
        return gene
    
    def _paraphrase_gene(self, gene: PromptGene) -> PromptGene:
        """Paraphrase gene content while preserving meaning."""
        paraphrased = self.llm.paraphrase(
            gene.content,
            style=gene.gene_type.value,
            preserve_keywords=True
        )
        return PromptGene(
            gene_type=gene.gene_type,
            content=paraphrased,
            weight=gene.weight,
            position_flexibility=gene.position_flexibility
        )
    
    def _calculate_adaptive_rate(self, base_rate: float, 
                                  generation: int) -> float:
        """
        Adjust mutation rate based on evolution progress.
        Higher early, lower as population converges.
        """
        if generation < 50:
            # Exploration phase: higher mutation
            return base_rate * 1.5
        elif generation < 200:
            # Exploitation phase: standard mutation
            return base_rate
        else:
            # Refinement phase: lower mutation
            return base_rate * 0.7
```

---

## 6. POPULATION MANAGEMENT

### 6.1 Population Structure

```python
@dataclass
class Population:
    """
    Managed population of prompt chromosomes.
    """
    individuals: List[PromptChromosome]
    generation: int = 0
    
    # Population statistics
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    
    # Metadata
    creation_timestamp: float = field(default_factory=time.time)
    task_domain: str = ""
    
    def get_statistics(self) -> Dict:
        """Calculate population statistics."""
        fitnesses = [ind.fitness_score for ind in self.individuals 
                    if ind.fitness_score is not None]
        
        return {
            'size': len(self.individuals),
            'generation': self.generation,
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'worst_fitness': min(fitnesses) if fitnesses else 0,
            'fitness_std': np.std(fitnesses) if fitnesses else 0,
            'diversity': self.calculate_diversity(),
            'age_distribution': self._age_distribution()
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

class PopulationManager:
    """
    Manages population lifecycle and dynamics.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.population = None
        self.archive = PromptArchive()
        
    def initialize(self, task_domain: str, 
                   seed_prompts: List[str] = None) -> Population:
        """Initialize new population."""
        individuals = []
        
        if seed_prompts:
            # Create from seed prompts
            for prompt in seed_prompts:
                chrom = PromptChromosome.decode(prompt)
                chrom.task_domain = task_domain
                individuals.append(chrom)
        
        # Fill remaining with generated individuals
        while len(individuals) < self.config['population_size']:
            chrom = self._generate_random_chromosome(task_domain)
            individuals.append(chrom)
        
        self.population = Population(
            individuals=individuals,
            task_domain=task_domain
        )
        
        return self.population
    
    def evolve_generation(self) -> Population:
        """Evolve population by one generation."""
        # Evaluate fitness
        fitness_scores = self._evaluate_population()
        
        # Create next generation
        new_individuals = []
        
        # Elite preservation
        elite_count = int(self.config['elite_ratio'] * len(self.population.individuals))
        elites = self._select_elites(elite_count)
        new_individuals.extend(elites)
        
        # Generate offspring
        while len(new_individuals) < len(self.population.individuals):
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.config['crossover_rate']:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
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
        
        return self.population
```

### 6.2 Diversity Maintenance

```python
class DiversityManager:
    """
    Maintains population diversity to prevent premature convergence.
    """
    
    def __init__(self, diversity_threshold: float = 0.3):
        self.diversity_threshold = diversity_threshold
        self.crowding_factor = 2.0
        
    def inject_diversity(self, population: Population) -> Population:
        """Inject diversity when population stagnates."""
        current_diversity = population.calculate_diversity()
        
        if current_diversity < self.diversity_threshold:
            # Calculate how many individuals to replace
            replace_count = int(0.3 * len(population.individuals))
            
            # Remove similar individuals
            to_replace = self._find_similar_clusters(population, replace_count)
            
            # Replace with diverse individuals
            for idx in to_replace:
                population.individuals[idx] = self._generate_diverse_individual(
                    population
                )
        
        return population
    
    def crowding_selection(self, population: Population,
                          offspring: PromptChromosome,
                          tournament_size: int = 5) -> int:
        """
        Crowding: replace most similar individual.
        """
        # Random tournament
        candidates = random.sample(range(len(population.individuals)), 
                                  min(tournament_size, len(population.individuals)))
        
        # Find most similar
        most_similar = min(candidates,
                          key=lambda i: offspring.distance(population.individuals[i]))
        
        return most_similar
    
    def _find_similar_clusters(self, population: Population, 
                               count: int) -> List[int]:
        """Find clusters of similar individuals for replacement."""
        # Calculate pairwise distances
        distances = {}
        for i in range(len(population.individuals)):
            for j in range(i + 1, len(population.individuals)):
                dist = population.individuals[i].distance(population.individuals[j])
                distances[(i, j)] = dist
        
        # Find individuals in dense clusters
        density_scores = defaultdict(float)
        for (i, j), dist in distances.items():
            if dist < 0.3:  # Similarity threshold
                density_scores[i] += 1
                density_scores[j] += 1
        
        # Return most clustered individuals
        sorted_by_density = sorted(density_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_by_density[:count]]
```

---

## 7. PROMPT BREEDING STRATEGIES

### 7.1 Breeding Pipeline

```python
class PromptBreedingPipeline:
    """
    Orchestrates the complete prompt breeding process.
    """
    
    def __init__(self, config: BreedingConfig):
        self.config = config
        self.encoder = PromptEncoder()
        self.fitness_evaluator = FitnessEvaluator()
        self.selection = SelectionOperators()
        self.crossover = CrossoverOperators()
        self.mutation = MutationOperators()
        self.archive = PromptArchive()
        
    def breed(self, parent_population: Population,
              target_improvement: float = 0.1) -> Population:
        """
        Execute complete breeding cycle.
        """
        # Phase 1: Selection
        mating_pool = self._create_mating_pool(parent_population)
        
        # Phase 2: Crossover
        offspring = self._generate_offspring(mating_pool)
        
        # Phase 3: Mutation
        mutated_offspring = self._apply_mutations(offspring)
        
        # Phase 4: Evaluation
        evaluated_offspring = self._evaluate_offspring(mutated_offspring)
        
        # Phase 5: Selection for next generation
        next_generation = self._environmental_selection(
            parent_population, 
            evaluated_offspring
        )
        
        # Phase 6: Archive update
        self._update_archive(next_generation)
        
        return next_generation
    
    def _create_mating_pool(self, population: Population) -> List[PromptChromosome]:
        """Create pool of parents for breeding."""
        pool_size = len(population.individuals) * 2
        mating_pool = []
        
        fitness_scores = [ind.fitness_score for ind in population.individuals]
        
        while len(mating_pool) < pool_size:
            parent = self.selection.tournament_selection(
                population.individuals,
                fitness_scores,
                self.config.tournament_size
            )
            mating_pool.append(parent)
        
        return mating_pool
    
    def _generate_offspring(self, mating_pool: List[PromptChromosome]) -> List[PromptChromosome]:
        """Generate offspring through crossover."""
        offspring = []
        
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if i + 1 < len(mating_pool) else mating_pool[0]
            
            child1, child2 = self.crossover.semantic_crossover(parent1, parent2)
            offspring.extend([child1, child2])
        
        return offspring
    
    def _environmental_selection(self, parents: Population,
                                 offspring: List[PromptChromosome]) -> Population:
        """
        Select survivors for next generation.
        Uses (mu + lambda) or (mu, lambda) strategy.
        """
        if self.config.selection_strategy == 'plus':
            # (mu + lambda): parents compete with offspring
            combined = parents.individuals + offspring
        else:
            # (mu, lambda): only offspring compete
            combined = offspring
        
        # Evaluate combined population
        for individual in combined:
            if individual.fitness_score is None:
                individual.fitness_score = self.fitness_evaluator.quick_evaluate(individual)
        
        # Sort by fitness
        sorted_pop = sorted(combined, key=lambda x: x.fitness_score or 0, reverse=True)
        
        # Select top individuals
        survivors = sorted_pop[:len(parents.individuals)]
        
        return Population(
            individuals=survivors,
            generation=parents.generation + 1,
            task_domain=parents.task_domain
        )
```

### 7.2 Specialized Breeding Strategies

```python
class SpecializedBreeding:
    """
    Advanced breeding strategies for specific scenarios.
    """
    
    @staticmethod
    def niching_strategy(population: Population,
                        num_niches: int = 5) -> Population:
        """
        Fitness sharing to maintain multiple niches.
        """
        # Cluster population into niches
        niches = cluster_by_similarity(population.individuals, num_niches)
        
        new_population = []
        
        for niche in niches:
            # Apply fitness sharing within niche
            shared_fitnesses = apply_fitness_sharing(niche)
            
            # Select from niche proportional to shared fitness
            niche_size = len(population.individuals) // num_niches
            selected = weighted_sample(niche, shared_fitnesses, niche_size)
            
            new_population.extend(selected)
        
        return Population(individuals=new_population)
    
    @staticmethod
    def coevolution_strategy(subpopulations: List[Population]) -> Population:
        """
        Coevolution: multiple populations evolve together.
        """
        # Evaluate complete solutions by combining from each subpopulation
        complete_solutions = []
        
        for _ in range(100):  # Sample combinations
            combination = [random.choice(subpop.individuals) 
                          for subpop in subpopulations]
            
            # Merge into complete prompt
            merged = merge_chromosomes(combination)
            
            # Evaluate
            merged.fitness_score = evaluate_complete(merged)
            complete_solutions.append(merged)
        
        # Credit assignment back to subpopulations
        for subpop in subpopulations:
            update_subpopulation_fitness(subpop, complete_solutions)
        
        # Combine best from each subpopulation
        best_from_each = [max(subpop.individuals, key=lambda x: x.fitness_score or 0)
                         for subpop in subpopulations]
        
        return merge_chromosomes(best_from_each)
    
    @staticmethod
    def memetic_strategy(population: Population,
                        local_search_iterations: int = 10) -> Population:
        """
        Memetic algorithm: GA + local search.
        """
        # Standard GA step
        evolved = standard_evolution_step(population)
        
        # Local search on elites
        elites = select_elites(evolved, ratio=0.2)
        
        for elite in elites:
            improved = local_search(
                elite,
                iterations=local_search_iterations,
                neighborhood_size=20
            )
            
            # Replace if improved
            if improved.fitness_score > elite.fitness_score:
                replace_individual(evolved, elite, improved)
        
        return evolved
```

---

## 8. CONVERGENCE DETECTION

### 8.1 Convergence Metrics

```python
@dataclass
class ConvergenceState:
    """
    Tracks convergence state of evolution.
    """
    is_converged: bool = False
    convergence_generation: Optional[int] = None
    convergence_type: Optional[str] = None
    
    # Metrics
    fitness_plateau: bool = False
    diversity_collapse: bool = False
    population_stagnation: bool = False
    
    # Statistics at convergence
    final_fitness: float = 0.0
    final_diversity: float = 0.0
    generations_evolved: int = 0

class ConvergenceDetector:
    """
    Detects various forms of evolutionary convergence.
    """
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.fitness_history = []
        self.diversity_history = []
        self.improvement_history = []
        
    def check_convergence(self, population: Population) -> ConvergenceState:
        """
        Check if evolution has converged.
        """
        state = ConvergenceState()
        
        # Update histories
        stats = population.get_statistics()
        self.fitness_history.append(stats['best_fitness'])
        self.diversity_history.append(stats['diversity'])
        
        # Check fitness plateau
        state.fitness_plateau = self._check_fitness_plateau()
        
        # Check diversity collapse
        state.diversity_collapse = self._check_diversity_collapse()
        
        # Check population stagnation
        state.population_stagnation = self._check_stagnation()
        
        # Determine if converged
        state.is_converged = (
            state.fitness_plateau or 
            state.diversity_collapse or
            state.population_stagnation or
            population.generation >= self.config.max_generations
        )
        
        if state.is_converged:
            state.convergence_generation = population.generation
            state.final_fitness = stats['best_fitness']
            state.final_diversity = stats['diversity']
            state.generations_evolved = population.generation
            
            # Determine convergence type
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
        """Check if fitness has plateaued."""
        if len(self.fitness_history) < window:
            return False
        
        recent = self.fitness_history[-window:]
        
        # Check if improvement is below threshold
        improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 0.001)
        
        # Check variance
        variance = np.var(recent)
        
        return improvement < self.config.fitness_improvement_threshold and \
               variance < self.config.fitness_variance_threshold
    
    def _check_diversity_collapse(self, window: int = 30) -> bool:
        """Check if diversity has collapsed."""
        if len(self.diversity_history) < window:
            return False
        
        recent_diversity = self.diversity_history[-window:]
        avg_diversity = sum(recent_diversity) / len(recent_diversity)
        
        return avg_diversity < self.config.diversity_threshold
    
    def _check_stagnation(self) -> bool:
        """Check if population is stagnant."""
        if len(self.improvement_history) < 100:
            return False
        
        # Count generations without improvement
        no_improvement_count = 0
        best_so_far = 0
        
        for fitness in reversed(self.fitness_history):
            if fitness > best_so_far:
                break
            no_improvement_count += 1
            best_so_far = max(best_so_far, fitness)
        
        return no_improvement_count > self.config.stagnation_generations
```

### 8.2 Adaptive Convergence Handling

```python
class AdaptiveConvergenceHandler:
    """
    Handles convergence with adaptive restart strategies.
    """
    
    def __init__(self):
        self.convergence_count = 0
        self.restart_history = []
        
    def handle_convergence(self, population: Population,
                          state: ConvergenceState) -> Optional[Population]:
        """
        Handle detected convergence with appropriate strategy.
        """
        self.convergence_count += 1
        
        if state.convergence_type == 'fitness_plateau':
            # Good convergence: save result
            return self._save_final_result(population)
        
        elif state.convergence_type == 'diversity_collapse':
            # Bad convergence: restart with diversity injection
            return self._restart_with_diversity(population)
        
        elif state.convergence_type == 'stagnation':
            # Stagnation: partial restart with elite preservation
            return self._partial_restart(population)
        
        else:
            # Max generations: evaluate and potentially continue
            return self._evaluate_and_decide(population)
    
    def _restart_with_diversity(self, population: Population) -> Population:
        """Restart with injected diversity."""
        # Preserve top elites
        elite_count = max(3, len(population.individuals) // 20)
        elites = select_elites(population, elite_count)
        
        # Generate diverse individuals
        new_individuals = list(elites)
        
        while len(new_individuals) < len(population.individuals):
            # Generate from different regions of search space
            diverse = self._generate_diverse_individual(population)
            new_individuals.append(diverse)
        
        # Increase mutation rate temporarily
        self.temp_mutation_boost = 2.0
        
        return Population(
            individuals=new_individuals,
            generation=0,  # Reset generation counter
            task_domain=population.task_domain
        )
```

---

## 9. ELITE PRESERVATION

### 9.1 Elite Management System

```python
@dataclass
class EliteIndividual:
    """
    Elite individual with extended metadata.
    """
    chromosome: PromptChromosome
    fitness_score: float
    generation_discovered: int
    
    # Performance tracking
    task_successes: int = 0
    task_attempts: int = 0
    
    # Lineage
    parent_ids: List[str] = field(default_factory=list)
    offspring_count: int = 0
    
    # Validation
    validation_fitness: Optional[float] = None
    cross_validation_scores: List[float] = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.task_attempts == 0:
            return 0.0
        return self.task_successes / self.task_attempts

class ElitePreservation:
    """
    Manages elite individuals across generations.
    """
    
    def __init__(self, elite_size: int = 10):
        self.elite_size = elite_size
        self.elites: List[EliteIndividual] = []
        self.hall_of_fame: List[EliteIndividual] = []
        
    def update_elites(self, population: Population) -> None:
        """Update elite list with current population."""
        # Get current population elites
        current_elites = self._select_population_elites(population)
        
        # Merge with existing elites
        combined = self.elites + current_elites
        
        # Remove duplicates based on chromosome similarity
        unique_elites = self._remove_duplicates(combined)
        
        # Sort by fitness
        sorted_elites = sorted(unique_elites, 
                              key=lambda e: e.fitness_score, 
                              reverse=True)
        
        # Keep top elites
        self.elites = sorted_elites[:self.elite_size]
        
        # Update hall of fame
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
                    chromosome=ind,
                    fitness_score=ind.fitness_score,
                    generation_discovered=population.generation,
                    parent_ids=ind.parent_ids
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
                    # Keep the one with higher fitness
                    if elite.fitness_score > existing.fitness_score:
                        unique[unique.index(existing)] = elite
                    break
            
            if not is_duplicate:
                unique.append(elite)
        
        return unique
    
    def _update_hall_of_fame(self) -> None:
        """Update hall of fame with all-time best."""
        # Add current elites to hall of fame
        for elite in self.elites:
            if not any(e.chromosome.chromosome_id == elite.chromosome.chromosome_id 
                      for e in self.hall_of_fame):
                self.hall_of_fame.append(elite)
        
        # Sort and trim hall of fame
        self.hall_of_fame = sorted(self.hall_of_fame,
                                   key=lambda e: e.fitness_score,
                                   reverse=True)[:50]  # Keep top 50 all-time
    
    def get_elite_for_breeding(self) -> Optional[PromptChromosome]:
        """Select elite for breeding with probability."""
        if not self.elites:
            return None
        
        # 20% chance to use elite
        if random.random() < 0.2:
            # Weighted by fitness
            fitnesses = [e.fitness_score for e in self.elites]
            total = sum(fitnesses)
        
        if total > 0:
            probabilities = [f / total for f in fitnesses]
            selected = np.random.choice(len(self.elites), p=probabilities)
            return self.elites[selected].chromosome
        
        return None
```

### 9.2 Elite Validation

```python
class EliteValidator:
    """
    Validates elites on held-out test sets.
    """
    
    def __init__(self, validation_suite: List[Task]):
        self.validation_suite = validation_suite
        self.validation_frequency = 10  # Every N generations
        
    def validate_elites(self, elites: List[EliteIndividual]) -> None:
        """Validate elites on held-out tasks."""
        for elite in elites:
            # Run validation suite
            scores = []
            for task in self.validation_suite:
                result = execute_task(elite.chromosome, task)
                scores.append(result.score)
            
            # Update validation metrics
            elite.validation_fitness = sum(scores) / len(scores)
            elite.cross_validation_scores = scores
            
            # Check for overfitting
            training_fitness = elite.fitness_score
            validation_fitness = elite.validation_fitness
            
            overfit_ratio = training_fitness / max(validation_fitness, 0.001)
            
            if overfit_ratio > 1.5:
                # Mark as potentially overfitted
                elite.overfitting_warning = True
```

---

## 10. INTEGRATION WITH AGENT SYSTEM

### 10.1 Loop Integration

```python
class ContextPromptEngineeringLoop:
    """
    Main loop integrating evolutionary prompt optimization
    into the agent system.
    """
    
    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.optimizer = EvolutionaryPromptOptimizer()
        self.population_manager = PopulationManager(agent_config.ga_config)
        self.convergence_detector = ConvergenceDetector(agent_config.conv_config)
        self.elite_preservation = ElitePreservation(elite_size=10)
        self.prompt_archive = PromptArchive()
        
        # Task tracking
        self.task_history = []
        self.performance_metrics = defaultdict(list)
        
    def run(self, task_domain: str = None) -> OptimizedPrompt:
        """
        Execute the prompt engineering optimization loop.
        """
        # Determine task domain if not specified
        if task_domain is None:
            task_domain = self._infer_task_domain()
        
        # Initialize or load population
        population = self._initialize_population(task_domain)
        
        # Evolution loop
        while True:
            # Evaluate current population
            population = self._evaluate_population(population, task_domain)
            
            # Update elites
            self.elite_preservation.update_elites(population)
            
            # Check convergence
            conv_state = self.convergence_detector.check_convergence(population)
            
            if conv_state.is_converged:
                # Handle convergence
                result = self._handle_convergence(population, conv_state)
                if result:
                    return result
            
            # Evolve next generation
            population = self.population_manager.evolve_generation()
            
            # Periodic operations
            if population.generation % 10 == 0:
                self._periodic_maintenance(population)
        
    def _evaluate_population(self, population: Population,
                            task_domain: str) -> Population:
        """Evaluate population on representative tasks."""
        # Get task suite for domain
        task_suite = self._get_task_suite(task_domain)
        
        # Evaluate each individual
        for individual in population.individuals:
            if individual.fitness_score is None:
                fitness = self.optimizer.fitness_evaluator.evaluate(
                    individual, 
                    task_suite
                )
                individual.fitness_score = fitness
        
        return population
    
    def _periodic_maintenance(self, population: Population) -> None:
        """Perform periodic maintenance operations."""
        # Archive current population
        self.prompt_archive.archive_population(population)
        
        # Validate elites
        if population.generation % 50 == 0:
            self.elite_validator.validate_elites(self.elite_preservation.elites)
        
        # Inject diversity if needed
        if population.calculate_diversity() < 0.2:
            population = self.diversity_manager.inject_diversity(population)
        
        # Save checkpoint
        self._save_checkpoint(population)
```

### 10.2 Real-Time Adaptation

```python
class RealTimePromptAdapter:
    """
    Adapts prompts in real-time based on task performance.
    """
    
    def __init__(self, base_prompt: str):
        self.base_chromosome = PromptChromosome.decode(base_prompt)
        self.performance_window = deque(maxlen=100)
        self.adaptation_threshold = 0.7
        
    def adapt(self, task_result: TaskResult) -> str:
        """
        Adapt prompt based on recent task performance.
        """
        # Record performance
        self.performance_window.append(task_result.success)
        
        # Check if adaptation needed
        recent_success_rate = sum(self.performance_window) / len(self.performance_window)
        
        if recent_success_rate < self.adaptation_threshold:
            # Trigger adaptation
            adapted = self._generate_adaptation()
            return adapted.encode()
        
        return self.base_chromosome.encode()
    
    def _generate_adaptation(self) -> PromptChromosome:
        """Generate adapted chromosome."""
        # Identify weak components
        weak_genes = self._identify_weak_genes()
        
        # Apply targeted mutations
        adapted_genes = list(self.base_chromosome.genes)
        
        for gene_idx in weak_genes:
            # Strengthen weak gene
            adapted_genes[gene_idx] = self._strengthen_gene(
                adapted_genes[gene_idx]
            )
        
        # Create adapted chromosome
        adapted = PromptChromosome(
            chromosome_id=generate_uuid(),
            genes=adapted_genes,
            gene_order=self.base_chromosome.gene_order,
            parent_ids=[self.base_chromosome.chromosome_id]
        )
        
        return adapted
```

---

## 11. CONFIGURATION SPECIFICATION

### 11.1 Full Configuration Schema

```yaml
# context_prompt_engineering_config.yaml

evolutionary_prompt_optimization:
  
  # Genetic Algorithm Parameters
  genetic_algorithm:
    population_size: 100
    max_generations: 1000
    elite_ratio: 0.1
    crossover_rate: 0.8
    mutation_rate: 0.15
    adaptive_mutation: true
    tournament_size: 5
    selection_strategy: "tournament"  # tournament, rank, roulette
    
  # Prompt Encoding
  encoding:
    gene_types:
      - system_context
      - task_instruction
      - constraint
      - example
      - format_spec
      - tool_hint
      - reasoning_step
      - safety_guard
      - context_variable
      - style_modifier
    max_genes_per_chromosome: 50
    min_genes_per_chromosome: 3
    token_limit: 4096
    
  # Fitness Evaluation
  fitness:
    weights:
      task_performance: 0.40
      token_efficiency: 0.25
      response_quality: 0.20
      robustness: 0.10
      safety: 0.05
    evaluation_tasks_per_generation: 50
    validation_split: 0.2
    
  # Convergence Detection
  convergence:
    fitness_improvement_threshold: 0.01
    fitness_variance_threshold: 0.001
    diversity_threshold: 0.3
    stagnation_generations: 100
    max_generations_without_improvement: 50
    
  # Elite Management
  elite_preservation:
    elite_pool_size: 10
    hall_of_fame_size: 50
    validation_frequency: 10
    overfit_threshold: 1.5
    
  # Diversity Management
  diversity:
    maintenance_strategy: "crowding"  # crowding, sharing, injection
    diversity_threshold: 0.3
    injection_ratio: 0.3
    crowding_factor: 2.0
    
  # Breeding Strategies
  breeding:
    primary_strategy: "semantic_crossover"
    secondary_strategies:
      - uniform_crossover
      - niching
      - memetic
    strategy_rotation_frequency: 100
    
  # Mutation Operators
  mutation:
    operators:
      gene_substitution: 0.25
      gene_insertion: 0.20
      gene_deletion: 0.15
      gene_reordering: 0.15
      weight_adjustment: 0.15
      paraphrase: 0.10
    adaptive_schedule:
      exploration_phase: 1.5
      exploitation_phase: 1.0
      refinement_phase: 0.7
      
  # Archive Management
  archive:
    max_archived_prompts: 10000
    compression_enabled: true
    similarity_threshold: 0.95
    retention_policy: "fitness_based"  # fitness_based, age_based, hybrid
    
  # Integration
  integration:
    loop_frequency: "per_task"  # per_task, periodic, manual
    adaptation_enabled: true
    real_time_tuning: true
    checkpoint_interval: 50
```

---

## 12. PERFORMANCE METRICS & MONITORING

### 12.1 Key Performance Indicators

| Metric | Target | Measurement |
|--------|--------|-------------|
| Prompt Fitness | >0.85 | Weighted multi-objective score |
| Task Success Rate | >90% | Successful completions / Total tasks |
| Token Efficiency | >0.8 | Optimal tokens / Actual tokens |
| Convergence Speed | <500 gen | Generations to convergence |
| Diversity Maintenance | >0.3 | Population diversity score |
| Elite Validation | >0.95 | Cross-validation fitness |
| Adaptation Latency | <100ms | Time to adapt to performance drop |

### 12.2 Monitoring Dashboard

```python
class EvolutionMonitor:
    """
    Real-time monitoring of evolutionary process.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'diversity_collapse': 0.1,
            'fitness_stagnation': 50,
            'overfitting_detected': 1.5
        }
    
    def log_generation(self, population: Population, 
                       generation_time: float) -> None:
        """Log generation statistics."""
        stats = population.get_statistics()
        
        metrics = {
            'timestamp': time.time(),
            'generation': population.generation,
            'best_fitness': stats['best_fitness'],
            'avg_fitness': stats['avg_fitness'],
            'diversity': stats['diversity'],
            'generation_time': generation_time,
            'population_size': stats['size']
        }
        
        self.metrics_history.append(metrics)
        
        # Check alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: Dict) -> None:
        """Check for alert conditions."""
        alerts = []
        
        if metrics['diversity'] < self.alert_thresholds['diversity_collapse']:
            alerts.append({
                'level': 'critical',
                'type': 'diversity_collapse',
                'message': f"Diversity collapsed: {metrics['diversity']:.3f}"
            })
        
        # Send alerts if any
        if alerts:
            self._send_alerts(alerts)
```

---

## 13. IMPLEMENTATION ROADMAP

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Implement chromosome encoding/decoding
- [ ] Build basic genetic operators
- [ ] Create population management
- [ ] Implement fitness evaluation framework

### Phase 2: Advanced Features (Weeks 3-4)
- [ ] Implement all selection strategies
- [ ] Build crossover operator library
- [ ] Create mutation operator suite
- [ ] Implement convergence detection

### Phase 3: Optimization (Weeks 5-6)
- [ ] Build elite preservation system
- [ ] Implement diversity management
- [ ] Create breeding strategy orchestrator
- [ ] Build prompt archive

### Phase 4: Integration (Weeks 7-8)
- [ ] Integrate with agent system
- [ ] Implement real-time adaptation
- [ ] Create monitoring dashboard
- [ ] Performance tuning and optimization

---

## APPENDIX: GENETIC OPERATOR QUICK REFERENCE

### Selection Operators
| Operator | Description | Use Case |
|----------|-------------|----------|
| Tournament | Compete k random individuals | General purpose |
| Rank-Based | Probability proportional to rank | Prevent premature convergence |
| Roulette | Probability proportional to fitness | Simple selection |
| Diversity-Aware | Fitness + diversity | Maintain exploration |

### Crossover Operators
| Operator | Description | Use Case |
|----------|-------------|----------|
| Single-Point | One crossover point | Simple recombination |
| Uniform | Gene-by-gene mixing | High diversity |
| Semantic | Type-aware mixing | Preserve meaning |
| PMX | Permutation preserving | Ordered genes |

### Mutation Operators
| Operator | Description | Use Case |
|----------|-------------|----------|
| Substitution | Replace gene content | Content variation |
| Insertion | Add new gene | Expand capability |
| Deletion | Remove gene | Simplify prompt |
| Reordering | Shuffle gene order | Structure variation |
| Paraphrase | Reword content | Semantic variation |

---

*Document Version: 1.0*
*Last Updated: 2025*
*Author: AI Systems Architecture Team*
