# Advanced Self-Driven Loop Technical Specification
## Intrinsic Motivation with Curiosity-Driven Exploration
### OpenClaw-Inspired AI Agent System for Windows 10

---

## Executive Summary

This document provides a comprehensive technical specification for the **Advanced Self-Driven Loop** - a sophisticated intrinsic motivation system designed for the Windows 10 OpenClaw-inspired AI agent framework. The system implements cutting-edge curiosity algorithms, multi-modal intrinsic reward mechanisms, and autonomous exploration capabilities to enable 24/7 self-directed learning and behavior generation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Curiosity-Driven Exploration Algorithms](#2-curiosity-driven-exploration-algorithms)
3. [Information Gain Prediction](#3-information-gain-prediction)
4. [Surprise and Novelty Detection](#4-surprise-and-novelty-detection)
5. [Intrinsic Reward Calculation Framework](#5-intrinsic-reward-calculation-framework)
6. [Boredom Detection and Avoidance](#6-boredom-detection-and-avoidance)
7. [Interest Model Maintenance](#7-interest-model-maintenance)
8. [Drive Satisfaction Metrics](#8-drive-satisfaction-metrics)
9. [Motivation Decay and Renewal](#9-motivation-decay-and-renewal)
10. [Implementation Specifications](#10-implementation-specifications)

---

## 1. Architecture Overview

### 1.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED SELF-DRIVEN LOOP ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   PERCEPTION    │───▶│  STATE ENCODER  │───▶│  MEMORY SYSTEM  │          │
│  │    MODULE       │    │                 │    │                 │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │              CURIOSITY ENGINE (Core Component)                   │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │       │
│  │  │  Epistemic  │  │  Diversive  │  │ Empowerment │  │  Social │ │       │
│  │  │  Curiosity  │  │  Curiosity  │  │  Curiosity  │  │Curiosity│ │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │           INTRINSIC REWARD CALCULATOR                            │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │       │
│  │  │ Information │  │   Novelty   │  │   Learning  │  │ Surprise│ │       │
│  │  │    Gain     │  │   Reward    │  │   Progress  │  │  Reward │ │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │              MOTIVATION ORCHESTRATOR                             │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │       │
│  │  │   Boredom   │  │   Drive     │  │   Interest  │  │Motivation│ │       │
│  │  │  Detection  │  │  Satisfaction│  │    Model    │  │  Decay  │ │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │              ACTION SELECTION ENGINE                             │       │
│  │         (Integrates with 14 other agentic loops)                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Description | Priority |
|-----------|-------------|----------|
| Curiosity Engine | Multi-modal curiosity generation system | Critical |
| Intrinsic Reward Calculator | Computes various intrinsic reward signals | Critical |
| Motivation Orchestrator | Manages drive states and motivation dynamics | High |
| Surprise Detector | Identifies unexpected events and patterns | High |
| Novelty Tracker | Tracks state visitation and novelty scores | High |
| Learning Progress Monitor | Measures improvement in predictions | Medium |
| Boredom Detection System | Identifies stagnation and repetitive patterns | Medium |
| Interest Model Manager | Maintains evolving interest profiles | Medium |

---

## 2. Curiosity-Driven Exploration Algorithms

### 2.1 Multi-Modal Curiosity Framework

The system implements five distinct curiosity types based on cutting-edge research:

#### 2.1.1 Epistemic Curiosity (Uncertainty Reduction)

**Purpose**: Drive to reduce uncertainty and gain knowledge

**Mathematical Formulation**:
```
C_E(s, k) = H(S|k) - H(S|k ∪ {s})
```

Where:
- `H(S|k)` = Conditional entropy of skill space given current knowledge
- `s` = Potential skill/outcome to explore
- `k` = Current knowledge state

**Implementation**:
```python
class EpistemicCuriosity:
    """
    Implements uncertainty-based epistemic curiosity.
    Higher values indicate greater potential for knowledge gain.
    """
    
    def __init__(self, state_encoder, uncertainty_estimator):
        self.state_encoder = state_encoder
        self.uncertainty_estimator = uncertainty_estimator
        self.knowledge_base = KnowledgeBase()
        
    def compute(self, state, action, next_state):
        """
        Calculate epistemic curiosity for a state transition.
        
        Returns:
            float: Curiosity value (higher = more uncertain/interesting)
        """
        # Encode states
        encoded_current = self.state_encoder.encode(state)
        encoded_next = self.state_encoder.encode(next_state)
        
        # Estimate uncertainty before and after observation
        uncertainty_before = self.uncertainty_estimator.estimate(encoded_current)
        
        # Simulate knowledge update
        self.knowledge_base.update(state, action, next_state)
        uncertainty_after = self.uncertainty_estimator.estimate(encoded_next)
        
        # Epistemic curiosity = uncertainty reduction potential
        curiosity = uncertainty_before - uncertainty_after
        
        return max(0, curiosity)  # Ensure non-negative
```

#### 2.1.2 Diversive Curiosity (Novelty Seeking)

**Purpose**: Drive for novelty and variety

**Mathematical Formulation**:
```
C_D(s, k) = novelty(s, k) × diversity(s, recent_skills)
```

Where:
- `novelty(s, k)` = How new/unfamiliar is state `s`
- `diversity(s, recent_skills)` = How different from recent experiences

**Implementation**:
```python
class DiversiveCuriosity:
    """
    Implements novelty-seeking diversive curiosity.
    Rewards exploration of new and diverse states.
    """
    
    def __init__(self, memory_buffer, similarity_threshold=0.85):
        self.memory = memory_buffer
        self.similarity_threshold = similarity_threshold
        self.recent_skills = deque(maxlen=100)
        
    def compute_novelty(self, state_embedding):
        """
        Calculate novelty based on distance from known states.
        """
        if len(self.memory) == 0:
            return 1.0  # Maximum novelty for first experience
            
        # Find nearest neighbor distance
        similarities = self.memory.get_similarities(state_embedding)
        max_similarity = np.max(similarities) if similarities else 0
        
        # Novelty = 1 - similarity
        novelty = 1.0 - max_similarity
        return novelty
        
    def compute_diversity(self, state_embedding):
        """
        Calculate diversity from recent experiences.
        """
        if not self.recent_skills:
            return 1.0
            
        # Average distance from recent skills
        diversities = []
        for recent in self.recent_skills:
            dist = cosine_distance(state_embedding, recent)
            diversities.append(dist)
            
        return np.mean(diversities)
        
    def compute(self, state):
        state_embedding = self.encode_state(state)
        novelty = self.compute_novelty(state_embedding)
        diversity = self.compute_diversity(state_embedding)
        
        return novelty * diversity
```

#### 2.1.3 Empowerment Curiosity (Control Maximization)

**Purpose**: Drive to increase control and influence over environment

**Mathematical Formulation**:
```
C_EE(s, k) = I(A; S_future | S_current, s)
```

Where:
- `I` = Mutual information between actions and future states
- Measures how much control agent has over future outcomes

**Implementation**:
```python
class EmpowermentCuriosity:
    """
    Implements empowerment-based curiosity.
    Rewards states that maximize future action-state mutual information.
    """
    
    def __init__(self, forward_model, n_samples=100):
        self.forward_model = forward_model
        self.n_samples = n_samples
        
    def compute_empowerment(self, state):
        """
        Estimate empowerment as mutual information I(A; S').
        """
        # Sample actions and predict outcomes
        action_samples = self.sample_diverse_actions(self.n_samples)
        
        predicted_states = []
        for action in action_samples:
            predicted = self.forward_model.predict(state, action)
            predicted_states.append(predicted)
            
        # Calculate entropy of predicted states
        state_entropy = self.estimate_entropy(predicted_states)
        
        # Calculate conditional entropy (given actions)
        conditional_entropy = self.estimate_conditional_entropy(
            action_samples, predicted_states
        )
        
        # Empowerment = I(A; S') = H(S') - H(S'|A)
        empowerment = state_entropy - conditional_entropy
        
        return empowerment
```

#### 2.1.4 Social Curiosity (Interaction Drive)

**Purpose**: Drive to understand and interact with other agents/users

**Mathematical Formulation**:
```
C_S(s, k) = social_relevance(s) × interaction_potential(s)
```

**Implementation**:
```python
class SocialCuriosity:
    """
    Implements social curiosity for agent-user interaction.
    """
    
    def __init__(self, user_model, interaction_history):
        self.user_model = user_model
        self.interaction_history = interaction_history
        
    def compute(self, context):
        """
        Calculate social curiosity based on user interaction potential.
        """
        # Social relevance of current context
        relevance = self.user_model.get_interest_level(context)
        
        # Potential for meaningful interaction
        interaction_potential = self.estimate_interaction_value(context)
        
        # Recent interaction saturation (avoid over-interaction)
        time_since_last = self.time_since_last_interaction(context)
        saturation_factor = min(1.0, time_since_last / 3600)  # 1 hour normalization
        
        return relevance * interaction_potential * saturation_factor
```

#### 2.1.5 Aesthetic Curiosity (Pattern Appreciation)

**Purpose**: Drive for elegant, compressible patterns

**Mathematical Formulation**:
```
C_A(s, k) = complexity(s) × compressibility(s)
```

**Implementation**:
```python
class AestheticCuriosity:
    """
    Implements aesthetic curiosity for pattern discovery.
    Rewards states with interesting, learnable structure.
    """
    
    def __init__(self, complexity_estimator, compression_model):
        self.complexity_estimator = complexity_estimator
        self.compression_model = compression_model
        
    def compute(self, state):
        """
        Calculate aesthetic curiosity.
        High when state has complex but learnable structure.
        """
        # Estimate complexity
        complexity = self.complexity_estimator.estimate(state)
        
        # Estimate compressibility (learnability)
        original_size = self.get_representation_size(state)
        compressed_size = self.compression_model.compress(state)
        compressibility = 1.0 - (compressed_size / original_size)
        
        # Aesthetic curiosity peaks at intermediate complexity
        # Too simple = boring, too complex = overwhelming
        optimal_complexity = self.get_optimal_complexity()
        complexity_factor = np.exp(-((complexity - optimal_complexity) ** 2) / 
                                    (2 * optimal_complexity ** 2))
        
        return complexity_factor * compressibility
```

### 2.2 Combined Curiosity Score

**Mathematical Formulation**:
```
C_total(s, k) = Σ α_i × C_i(s, k)  for i ∈ {E, D, EE, S, A}
```

Where `α_i` are learned weights reflecting the agent's curiosity profile.

**Dynamic Weight Adaptation**:
```python
class AdaptiveCuriosityWeights:
    """
    Dynamically adjusts curiosity type weights based on context and performance.
    """
    
    def __init__(self):
        self.weights = {
            'epistemic': 0.25,
            'diversive': 0.25,
            'empowerment': 0.20,
            'social': 0.15,
            'aesthetic': 0.15
        }
        self.performance_history = defaultdict(list)
        
    def update_weights(self, context, outcomes):
        """
        Update weights based on which curiosity types led to successful outcomes.
        """
        for curiosity_type, success in outcomes.items():
            self.performance_history[curiosity_type].append(success)
            
        # Calculate success rates
        success_rates = {
            ct: np.mean(hist[-100:]) if hist else 0.5
            for ct, hist in self.performance_history.items()
        }
        
        # Normalize to get new weights
        total = sum(success_rates.values())
        if total > 0:
            self.weights = {
                ct: rate / total for ct, rate in success_rates.items()
            }
            
    def get_weights(self, context):
        """
        Get context-adjusted weights.
        """
        base_weights = self.weights.copy()
        
        # Context-specific adjustments
        if context.get('social_context'):
            base_weights['social'] *= 1.5
        if context.get('uncertainty_high'):
            base_weights['epistemic'] *= 1.3
        if context.get('repetitive_states'):
            base_weights['diversive'] *= 1.4
            
        # Renormalize
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}
```

---

## 3. Information Gain Prediction

### 3.1 Expected Information Gain (EIG)

**Purpose**: Predict how much an action will reduce uncertainty

**Mathematical Formulation**:
```
EIG(A) = E_{o~P(O|H,A)}[D_KL(P(H'|o) || P(H))]
```

Where:
- `D_KL` = Kullback-Leibler divergence
- `H` = Current hypothesis/world model
- `H'` = Updated hypothesis after observation

**Implementation**:
```python
class InformationGainPredictor:
    """
    Predicts expected information gain from potential actions.
    """
    
    def __init__(self, world_model, belief_updater):
        self.world_model = world_model
        self.belief_updater = belief_updater
        self.prediction_history = []
        
    def predict_information_gain(self, state, action, n_simulations=50):
        """
        Estimate expected information gain for taking an action.
        
        Args:
            state: Current state
            action: Proposed action
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            float: Expected information gain (bits)
        """
        current_belief = self.world_model.get_belief_distribution()
        
        information_gains = []
        
        for _ in range(n_simulations):
            # Simulate outcome
            simulated_outcome = self.world_model.simulate(state, action)
            
            # Update belief with simulated outcome
            updated_belief = self.belief_updater.update(
                current_belief, state, action, simulated_outcome
            )
            
            # Calculate KL divergence
            kl_divergence = self.compute_kl_divergence(
                updated_belief, current_belief
            )
            
            information_gains.append(kl_divergence)
            
        # Expected information gain
        expected_ig = np.mean(information_gains)
        
        # Track for learning
        self.prediction_history.append({
            'state': state,
            'action': action,
            'predicted_ig': expected_ig
        })
        
        return expected_ig
        
    def compute_kl_divergence(self, p, q):
        """
        Compute KL divergence D_KL(P || Q).
        """
        # Ensure valid probability distributions
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
```

### 3.2 Information Gain-Based Action Selection

```python
class InformationGainActionSelector:
    """
    Selects actions to maximize expected information gain.
    """
    
    def __init__(self, ig_predictor, exploration_factor=0.2):
        self.ig_predictor = ig_predictor
        self.exploration_factor = exploration_factor
        
    def select_action(self, state, available_actions):
        """
        Select action balancing information gain and other objectives.
        """
        action_scores = []
        
        for action in available_actions:
            # Predict information gain
            info_gain = self.ig_predictor.predict_information_gain(state, action)
            
            # Get other action values (from other loops)
            task_value = self.get_task_value(state, action)
            safety_score = self.get_safety_score(state, action)
            
            # Combined score
            combined_score = (
                self.exploration_factor * info_gain +
                (1 - self.exploration_factor) * task_value
            ) * safety_score
            
            action_scores.append((action, combined_score, info_gain))
            
        # Select action with highest combined score
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        return best_action
```

---

## 4. Surprise and Novelty Detection

### 4.1 Surprise Detection System

**Purpose**: Identify unexpected events that violate predictions

**Mathematical Formulation**:
```
Surprise(s_t, a_t, s_{t+1}) = ||f(φ(s_t), a_t) - φ(s_{t+1})||²
```

Where:
- `f` = Forward dynamics model
- `φ` = State embedding function
- `||.||²` = Squared prediction error

**Implementation**:
```python
class SurpriseDetector:
    """
    Detects surprising events based on prediction errors.
    """
    
    def __init__(self, forward_model, embedding_network, 
                 surprise_threshold=2.0, adaptation_rate=0.01):
        self.forward_model = forward_model
        self.embedding_network = embedding_network
        self.surprise_threshold = surprise_threshold
        self.adaptation_rate = adaptation_rate
        
        # Running statistics for normalization
        self.mean_error = 0.0
        self.var_error = 1.0
        self.error_history = deque(maxlen=1000)
        
    def compute_surprise(self, state, action, next_state):
        """
        Compute surprise as normalized prediction error.
        """
        # Encode states
        state_emb = self.embedding_network.encode(state)
        next_state_emb = self.embedding_network.encode(next_state)
        
        # Predict next state embedding
        predicted_emb = self.forward_model.predict(state_emb, action)
        
        # Compute prediction error
        prediction_error = np.mean((predicted_emb - next_state_emb) ** 2)
        
        # Normalize using running statistics
        normalized_surprise = (prediction_error - self.mean_error) / \
                              (np.sqrt(self.var_error) + 1e-8)
        
        # Update statistics
        self.error_history.append(prediction_error)
        self.mean_error = (1 - self.adaptation_rate) * self.mean_error + \
                          self.adaptation_rate * prediction_error
        self.var_error = np.var(self.error_history) if self.error_history else 1.0
        
        return normalized_surprise
        
    def is_surprising(self, state, action, next_state):
        """
        Determine if transition is surprising.
        """
        surprise = self.compute_surprise(state, action, next_state)
        return surprise > self.surprise_threshold
```

### 4.2 Surprise Memory Module

**Purpose**: Remember and recognize previously encountered surprises

```python
class SurpriseMemory:
    """
    Stores and retrieves surprise patterns for novelty detection.
    """
    
    def __init__(self, capacity=10000, similarity_threshold=0.9):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.surprise_patterns = []
        self.surprise_counts = []
        
    def add_surprise(self, surprise_vector, context):
        """
        Add a surprise pattern to memory.
        """
        # Check if similar surprise exists
        similar_idx = self.find_similar_surprise(surprise_vector)
        
        if similar_idx is not None:
            # Increment count for existing pattern
            self.surprise_counts[similar_idx] += 1
        else:
            # Add new surprise pattern
            if len(self.surprise_patterns) >= self.capacity:
                # Remove least frequent
                min_idx = np.argmin(self.surprise_counts)
                self.surprise_patterns.pop(min_idx)
                self.surprise_counts.pop(min_idx)
                
            self.surprise_patterns.append({
                'vector': surprise_vector,
                'context': context,
                'timestamp': time.time()
            })
            self.surprise_counts.append(1)
            
    def compute_surprise_novelty(self, surprise_vector):
        """
        Compute how novel a surprise is based on memory.
        """
        if not self.surprise_patterns:
            return 1.0  # Maximum novelty
            
        # Find maximum similarity
        max_similarity = 0.0
        for pattern in self.surprise_patterns:
            similarity = cosine_similarity(
                surprise_vector, pattern['vector']
            )
            max_similarity = max(max_similarity, similarity)
            
        # Novelty = 1 - similarity
        return 1.0 - max_similarity
```

### 4.3 Novelty Detection via State Counting

**Purpose**: Track state visitation for novelty-based rewards

```python
class StateNoveltyTracker:
    """
    Tracks state visitation counts for novelty-based exploration.
    Uses SimHash for high-dimensional state spaces.
    """
    
    def __init__(self, hash_size=64, embedding_dim=512):
        self.hash_size = hash_size
        self.embedding_dim = embedding_dim
        
        # Random projection matrix for SimHash
        self.projection = np.random.randn(embedding_dim, hash_size)
        
        # State visit counts
        self.visit_counts = defaultdict(int)
        
        # Total states visited
        self.total_visits = 0
        
    def compute_hash(self, state_embedding):
        """
        Compute SimHash for state embedding.
        """
        # Project and take sign
        projected = np.dot(state_embedding, self.projection)
        hash_code = (projected > 0).astype(int)
        
        # Convert to string for dictionary key
        return ''.join(map(str, hash_code))
        
    def get_novelty_score(self, state):
        """
        Get novelty score (inverse of visit count).
        """
        state_emb = self.encode_state(state)
        state_hash = self.compute_hash(state_emb)
        
        visit_count = self.visit_counts[state_hash]
        
        # Novelty = 1 / sqrt(visit_count + 1)
        novelty = 1.0 / np.sqrt(visit_count + 1)
        
        return novelty
        
    def update_visit(self, state):
        """
        Record state visitation.
        """
        state_emb = self.encode_state(state)
        state_hash = self.compute_hash(state_emb)
        
        self.visit_counts[state_hash] += 1
        self.total_visits += 1
        
    def get_intrinsic_reward(self, state):
        """
        Get intrinsic reward based on state novelty.
        """
        novelty = self.get_novelty_score(state)
        self.update_visit(state)
        return novelty
```

---

## 5. Intrinsic Reward Calculation Framework

### 5.1 Multi-Source Intrinsic Reward

The system combines multiple intrinsic reward sources:

```python
class IntrinsicRewardCalculator:
    """
    Calculates comprehensive intrinsic rewards from multiple sources.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize reward components
        self.curiosity_engine = CuriosityEngine()
        self.surprise_detector = SurpriseDetector()
        self.novelty_tracker = StateNoveltyTracker()
        self.learning_progress_tracker = LearningProgressTracker()
        
        # Reward weights (configurable)
        self.weights = {
            'curiosity': config.get('curiosity_weight', 0.30),
            'surprise': config.get('surprise_weight', 0.25),
            'novelty': config.get('novelty_weight', 0.20),
            'learning_progress': config.get('lp_weight', 0.25)
        }
        
        # Running statistics for normalization
        self.reward_stats = {k: RunningStats() for k in self.weights.keys()}
        
    def calculate_intrinsic_reward(self, state, action, next_state, info=None):
        """
        Calculate total intrinsic reward for a transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            info: Additional information
            
        Returns:
            dict: Breakdown of intrinsic rewards
            float: Total intrinsic reward
        """
        rewards = {}
        
        # 1. Curiosity reward
        rewards['curiosity'] = self.curiosity_engine.compute(
            state, action, next_state
        )
        
        # 2. Surprise reward
        rewards['surprise'] = self.surprise_detector.compute_surprise(
            state, action, next_state
        )
        
        # 3. Novelty reward
        rewards['novelty'] = self.novelty_tracker.get_intrinsic_reward(next_state)
        
        # 4. Learning progress reward
        rewards['learning_progress'] = self.learning_progress_tracker.compute(
            state, action, next_state
        )
        
        # Normalize each reward component
        normalized_rewards = {}
        for key, value in rewards.items():
            self.reward_stats[key].update(value)
            normalized_rewards[key] = self.reward_stats[key].normalize(value)
        
        # Compute weighted total
        total_reward = sum(
            self.weights[key] * normalized_rewards[key]
            for key in self.weights.keys()
        )
        
        return normalized_rewards, total_reward
```

### 5.2 Learning Progress Reward

**Purpose**: Reward improvement in prediction/model accuracy

**Mathematical Formulation**:
```
LP(t) = PE(t-1) - PE(t)
```

Where:
- `PE(t)` = Prediction error at time t
- Positive LP indicates learning/improvement

```python
class LearningProgressTracker:
    """
    Tracks and rewards learning progress.
    Based on the Learning Progress Hypothesis (Oudeyer et al.).
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.prediction_errors = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        
    def compute(self, state, action, next_state):
        """
        Compute learning progress reward.
        """
        # Get current prediction error
        current_error = self.get_prediction_error(state, action, next_state)
        
        if len(self.prediction_errors) < 10:
            # Not enough history
            self.prediction_errors.append(current_error)
            return 0.0
            
        # Compute learning progress
        recent_mean_error = np.mean(list(self.prediction_errors)[-10:])
        learning_progress = recent_mean_error - current_error
        
        # Store for next computation
        self.prediction_errors.append(current_error)
        self.learning_rates.append(learning_progress)
        
        # Return normalized learning progress
        return learning_progress
        
    def get_learning_trajectory(self):
        """
        Get overall learning trajectory for curriculum adaptation.
        """
        if len(self.learning_rates) < 20:
            return 'insufficient_data'
            
        recent_lp = np.mean(list(self.learning_rates)[-20:])
        
        if recent_lp > 0.01:
            return 'improving'
        elif recent_lp < -0.01:
            return 'degrading'
        else:
            return 'plateau'
```

### 5.3 Random Network Distillation (RND) Module

**Purpose**: Provide stable, noise-robust intrinsic rewards

```python
class RandomNetworkDistillation:
    """
    Implements Random Network Distillation for state novelty.
    More robust to stochastic environments than prediction-based methods.
    """
    
    def __init__(self, input_dim, output_dim=64, learning_rate=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Fixed random target network
        self.target_network = self.create_random_network()
        self.target_network.eval()  # Never train
        
        # Trainable predictor network
        self.predictor_network = self.create_predictor_network()
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate
        )
        
        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = deque(maxlen=1000)
        
    def create_random_network(self):
        """
        Create fixed random neural network.
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        
    def create_predictor_network(self):
        """
        Create trainable predictor network (same architecture as target).
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        
    def compute_intrinsic_reward(self, state):
        """
        Compute intrinsic reward as prediction error.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get target output (fixed)
            target_output = self.target_network(state_tensor)
            
            # Get predictor output
            predicted_output = self.predictor_network(state_tensor)
            
            # Compute error
            error = F.mse_loss(predicted_output, target_output)
            
        # Normalize
        normalized_reward = (error.item() - self.reward_mean) / \
                           (self.reward_std + 1e-8)
        
        # Update statistics
        self.reward_history.append(error.item())
        if len(self.reward_history) >= 100:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history)
            
        return normalized_reward
        
    def update(self, states):
        """
        Update predictor network on batch of states.
        """
        states_tensor = torch.FloatTensor(states)
        
        # Get targets (fixed)
        with torch.no_grad():
            targets = self.target_network(states_tensor)
            
        # Get predictions
        predictions = self.predictor_network(states_tensor)
        
        # Compute loss
        loss = F.mse_loss(predictions, targets)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 6. Boredom Detection and Avoidance

### 6.1 Boredom Detection System

**Purpose**: Identify when agent is stuck in repetitive, unproductive patterns

```python
class BoredomDetector:
    """
    Detects boredom based on multiple indicators.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Boredom indicators
        self.state_repetition_threshold = config.get('repetition_threshold', 0.95)
        self.reward_variance_threshold = config.get('reward_variance_threshold', 0.01)
        self.progress_stagnation_threshold = config.get('stagnation_threshold', 100)
        
        # Tracking buffers
        self.recent_states = deque(maxlen=config.get('state_buffer_size', 100))
        self.recent_rewards = deque(maxlen=config.get('reward_buffer_size', 100))
        self.recent_actions = deque(maxlen=config.get('action_buffer_size', 100))
        
        # Progress tracking
        self.learning_progress_history = deque(maxlen=500)
        self.last_progress_update = 0
        
        # Boredom state
        self.boredom_level = 0.0
        self.boredom_threshold = config.get('boredom_threshold', 0.7)
        
    def update(self, state, action, reward, learning_progress):
        """
        Update boredom detection with new experience.
        """
        self.recent_states.append(state)
        self.recent_rewards.append(reward)
        self.recent_actions.append(action)
        self.learning_progress_history.append(learning_progress)
        
        # Compute boredom indicators
        indicators = {
            'state_repetition': self.compute_state_repetition(),
            'reward_variance': self.compute_reward_variance(),
            'progress_stagnation': self.compute_progress_stagnation(),
            'action_diversity': self.compute_action_diversity()
        }
        
        # Update boredom level
        self.boredom_level = self.compute_boredom_level(indicators)
        
        return self.boredom_level, indicators
        
    def compute_state_repetition(self):
        """
        Compute how repetitive recent states are.
        """
        if len(self.recent_states) < 20:
            return 0.0
            
        # Compute pairwise similarities
        similarities = []
        states = list(self.recent_states)
        
        for i in range(len(states) - 1):
            for j in range(i + 1, len(states)):
                sim = self.state_similarity(states[i], states[j])
                similarities.append(sim)
                
        # High mean similarity = repetitive
        mean_similarity = np.mean(similarities)
        return mean_similarity
        
    def compute_reward_variance(self):
        """
        Compute variance of recent rewards.
        Low variance = potential boredom.
        """
        if len(self.recent_rewards) < 20:
            return 1.0
            
        variance = np.var(self.recent_rewards)
        
        # Normalize (lower variance = higher boredom indicator)
        normalized = max(0, 1.0 - variance / 0.1)
        return normalized
        
    def compute_progress_stagnation(self):
        """
        Compute how long learning progress has been stagnant.
        """
        if len(self.learning_progress_history) < 50:
            return 0.0
            
        recent_progress = list(self.learning_progress_history)[-50:]
        
        # Check if progress is consistently near zero
        mean_progress = np.mean(recent_progress)
        
        if abs(mean_progress) < 0.001:
            return 1.0
        else:
            return 0.0
            
    def compute_action_diversity(self):
        """
        Compute diversity of recent actions.
        Low diversity = potential boredom.
        """
        if len(self.recent_actions) < 20:
            return 1.0
            
        # Count unique actions
        unique_actions = len(set(self.recent_actions))
        total_actions = len(self.recent_actions)
        
        diversity = unique_actions / total_actions
        return diversity
        
    def compute_boredom_level(self, indicators):
        """
        Combine indicators into overall boredom level.
        """
        weights = {
            'state_repetition': 0.30,
            'reward_variance': 0.25,
            'progress_stagnation': 0.30,
            'action_diversity': 0.15
        }
        
        boredom = sum(
            weights[key] * (1 - indicators[key] if key == 'action_diversity' 
                          else indicators[key])
            for key in weights.keys()
        )
        
        return min(1.0, boredom)
        
    def is_bored(self):
        """
        Check if agent is currently bored.
        """
        return self.boredom_level > self.boredom_threshold
```

### 6.2 Boredom Avoidance Strategies

```python
class BoredomAvoidance:
    """
    Implements strategies to overcome boredom and re-engage curiosity.
    """
    
    def __init__(self, boredom_detector, curiosity_engine):
        self.boredom_detector = boredom_detector
        self.curiosity_engine = curiosity_engine
        
        # Strategy parameters
        self.exploration_boost_factor = 2.0
        self.novelty_weight_increase = 1.5
        self.skill_switch_threshold = 0.6
        
    def get_avoidance_action(self, current_context):
        """
        Get action to avoid/escape boredom.
        """
        boredom_level = self.boredom_detector.boredom_level
        indicators = self.boredom_detector.get_indicators()
        
        # Select strategy based on boredom type
        if indicators['state_repetition'] > 0.8:
            return self.strategy_explore_new_regions(current_context)
        elif indicators['progress_stagnation'] > 0.8:
            return self.strategy_reduce_difficulty(current_context)
        elif indicators['action_diversity'] < 0.3:
            return self.strategy_try_new_actions(current_context)
        else:
            return self.strategy_random_exploration(current_context)
            
    def strategy_explore_new_regions(self, context):
        """
        Strategy: Explore completely new state regions.
        """
        # Boost diversive curiosity
        self.curiosity_engine.boost_curiosity_type('diversive', 
                                                    self.exploration_boost_factor)
        
        # Temporarily increase novelty weight
        return {
            'action_type': 'explore_new_region',
            'curiosity_boost': 'diversive',
            'duration': 50,
            'target': 'farthest_from_current'
        }
        
    def strategy_reduce_difficulty(self, context):
        """
        Strategy: Switch to easier tasks to rebuild confidence.
        """
        return {
            'action_type': 'adjust_difficulty',
            'direction': 'decrease',
            'target_success_rate': 0.8,
            'duration': 30
        }
        
    def strategy_try_new_actions(self, context):
        """
        Strategy: Try actions not recently taken.
        """
        recent_actions = self.boredom_detector.recent_actions
        
        # Find underused actions
        action_counts = Counter(recent_actions)
        all_actions = self.get_all_available_actions()
        
        underused = [a for a in all_actions 
                    if action_counts.get(a, 0) < len(recent_actions) * 0.05]
        
        return {
            'action_type': 'try_underused_actions',
            'candidate_actions': underused,
            'exploration_probability': 0.5
        }
        
    def strategy_random_exploration(self, context):
        """
        Strategy: Pure random exploration as last resort.
        """
        return {
            'action_type': 'random_exploration',
            'duration': 20,
            'epsilon': 0.9  # High randomness
        }
```

---

## 7. Interest Model Maintenance

### 7.1 Dynamic Interest Profile

**Purpose**: Maintain evolving model of agent's interests

```python
class InterestModel:
    """
    Maintains dynamic model of agent's evolving interests.
    """
    
    def __init__(self, n_interest_dimensions=10):
        self.n_dimensions = n_interest_dimensions
        
        # Interest vector (0-1 scale for each dimension)
        self.interest_vector = np.ones(n_interest_dimensions) * 0.5
        
        # Interest history
        self.interest_history = deque(maxlen=1000)
        
        # Topic associations
        self.topic_associations = defaultdict(lambda: defaultdict(float))
        
        # Saturation levels
        self.saturation_levels = np.zeros(n_interest_dimensions)
        
        # Update parameters
        self.learning_rate = 0.1
        self.decay_rate = 0.001
        
    def update(self, experience, reward, curiosity_signal):
    """
        Update interest model based on experience.
        """
        # Extract interest-relevant features
        interest_features = self.extract_interest_features(experience)
        
        # Update interest vector based on positive experiences
        for dim in range(self.n_dimensions):
            feature_value = interest_features[dim]
            
            # Increase interest if experience was rewarding
            if reward > 0:
                self.interest_vector[dim] += self.learning_rate * \
                                              feature_value * reward
                                              
            # Decay over time
            self.interest_vector[dim] -= self.decay_rate
            
            # Clamp to valid range
            self.interest_vector[dim] = np.clip(
                self.interest_vector[dim], 0.0, 1.0
            )
            
        # Update saturation
        self.update_saturation(experience)
        
        # Record history
        self.interest_history.append(self.interest_vector.copy())
        
    def update_saturation(self, experience):
        """
        Update saturation levels for interest dimensions.
        """
        topic = self.get_experience_topic(experience)
        
        for dim in range(self.n_dimensions):
            if self.is_relevant_to_dimension(topic, dim):
                self.saturation_levels[dim] += 0.01
                self.saturation_levels[dim] = min(1.0, 
                                                   self.saturation_levels[dim])
            else:
                # Natural recovery from saturation
                self.saturation_levels[dim] *= 0.999
                
    def get_effective_interests(self):
        """
        Get interest levels adjusted for saturation.
        """
        effective = self.interest_vector * (1 - self.saturation_levels)
        return effective
        
    def get_top_interests(self, n=3):
        """
        Get top N interest dimensions.
        """
        effective = self.get_effective_interests()
        top_indices = np.argsort(effective)[-n:][::-1]
        return [(idx, effective[idx]) for idx in top_indices]
```

### 7.2 Interest-Driven Goal Generation

```python
class InterestDrivenGoalGenerator:
    """
    Generates goals based on current interest model.
    """
    
    def __init__(self, interest_model, goal_space):
        self.interest_model = interest_model
        self.goal_space = goal_space
        
    def generate_goal(self, context):
        """
        Generate a goal aligned with current interests.
        """
        # Get current interests
        interests = self.interest_model.get_effective_interests()
        
        # Sample goal space weighted by interests
        goal_weights = self.compute_goal_weights(interests)
        
        # Select goal
        goal = self.sample_goal(goal_weights)
        
        return goal
        
    def compute_goal_weights(self, interests):
        """
        Compute sampling weights for goals based on interests.
        """
        weights = []
        
        for goal in self.goal_space:
            # Compute goal-interest alignment
            alignment = self.compute_alignment(goal, interests)
            
            # Weight by interest and novelty
            novelty = self.compute_goal_novelty(goal)
            
            weight = alignment * (1 + novelty)
            weights.append(weight)
            
        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            
        return weights
```

---

## 8. Drive Satisfaction Metrics

### 8.1 Drive State Tracking

```python
class DriveSatisfactionTracker:
    """
    Tracks satisfaction levels for various intrinsic drives.
    """
    
    def __init__(self):
        # Drive definitions
        self.drives = {
            'knowledge_acquisition': {
                'current': 0.5,
                'target': 0.8,
                'decay_rate': 0.01,
                'satisfaction_rate': 0.05
            },
            'novelty_seeking': {
                'current': 0.5,
                'target': 0.7,
                'decay_rate': 0.02,
                'satisfaction_rate': 0.08
            },
            'competence': {
                'current': 0.5,
                'target': 0.75,
                'decay_rate': 0.005,
                'satisfaction_rate': 0.03
            },
            'autonomy': {
                'current': 0.5,
                'target': 0.8,
                'decay_rate': 0.01,
                'satisfaction_rate': 0.04
            },
            'social_connection': {
                'current': 0.5,
                'target': 0.6,
                'decay_rate': 0.015,
                'satisfaction_rate': 0.06
            }
        }
        
        # Satisfaction history
        self.satisfaction_history = defaultdict(list)
        
    def update_drive(self, drive_name, satisfaction_event=None):
        """
        Update a specific drive's satisfaction level.
        """
        drive = self.drives[drive_name]
        
        # Natural decay
        drive['current'] -= drive['decay_rate']
        
        # Satisfaction from event
        if satisfaction_event:
            drive['current'] += drive['satisfaction_rate'] * satisfaction_event
            
        # Clamp to valid range
        drive['current'] = np.clip(drive['current'], 0.0, 1.0)
        
        # Record
        self.satisfaction_history[drive_name].append(drive['current'])
        
    def get_drive_deficit(self, drive_name):
        """
        Get the deficit (unsatisfied portion) of a drive.
        """
        drive = self.drives[drive_name]
        return drive['target'] - drive['current']
        
    def get_priority_drive(self):
        """
        Get the drive with highest deficit (most unsatisfied).
        """
        deficits = {
            name: self.get_drive_deficit(name)
            for name in self.drives.keys()
        }
        
        return max(deficits.items(), key=lambda x: x[1])
        
    def get_overall_satisfaction(self):
        """
        Get overall drive satisfaction score.
        """
        satisfactions = [
            drive['current'] / drive['target']
            for drive in self.drives.values()
        ]
        
        return np.mean(satisfactions)
```

### 8.2 Satisfaction-Based Action Prioritization

```python
class SatisfactionBasedPrioritizer:
    """
    Prioritizes actions based on drive satisfaction deficits.
    """
    
    def __init__(self, drive_tracker):
        self.drive_tracker = drive_tracker
        
    def prioritize_actions(self, available_actions):
        """
        Prioritize actions to address drive deficits.
        """
        # Get priority drive
        priority_drive, deficit = self.drive_tracker.get_priority_drive()
        
        # Score each action by how well it addresses priority drive
        scored_actions = []
        
        for action in available_actions:
            drive_alignment = self.estimate_drive_satisfaction(action, 
                                                                priority_drive)
            scored_actions.append((action, drive_alignment))
            
        # Sort by alignment
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_actions
```

---

## 9. Motivation Decay and Renewal

### 9.1 Motivation Dynamics Model

```python
class MotivationDynamics:
    """
    Models motivation decay and renewal cycles.
    """
    
    def __init__(self):
        # Motivation state
        self.motivation_level = 0.7
        
        # Decay parameters
        self.base_decay_rate = 0.001
        self.activity_decay_factor = 0.1
        
        # Renewal parameters
        self.success_boost = 0.1
        self.novelty_boost = 0.15
        self.social_boost = 0.08
        
        # Cycle tracking
        self.cycle_history = []
        self.current_cycle_start = time.time()
        
    def update(self, experience_outcome):
        """
        Update motivation based on experience.
        """
        # Apply decay
        decay = self.compute_decay(experience_outcome)
        self.motivation_level -= decay
        
        # Apply renewal from positive outcomes
        renewal = self.compute_renewal(experience_outcome)
        self.motivation_level += renewal
        
        # Clamp
        self.motivation_level = np.clip(self.motivation_level, 0.0, 1.0)
        
        # Track cycles
        self.track_motivation_cycle()
        
    def compute_decay(self, outcome):
        """
        Compute motivation decay based on activity and outcomes.
        """
        # Base decay
        decay = self.base_decay_rate
        
        # Increase decay if activity is repetitive
        if outcome.get('repetitive', False):
            decay *= (1 + self.activity_decay_factor)
            
        # Increase decay if outcomes are negative
        if outcome.get('success', True) == False:
            decay *= 1.5
            
        return decay
        
    def compute_renewal(self, outcome):
        """
        Compute motivation renewal from positive experiences.
        """
        renewal = 0.0
        
        # Success renewal
        if outcome.get('success', False):
            renewal += self.success_boost * outcome.get('magnitude', 1.0)
            
        # Novelty renewal
        if outcome.get('novelty', 0) > 0.5:
            renewal += self.novelty_boost * outcome['novelty']
            
        # Social renewal
        if outcome.get('social_interaction', False):
            renewal += self.social_boost
            
        return renewal
        
    def track_motivation_cycle(self):
        """
        Track motivation cycles for pattern detection.
        """
        # Detect cycle boundaries (local minima and maxima)
        if len(self.cycle_history) > 10:
            recent = [c['level'] for c in self.cycle_history[-10:]]
            
            # Simple cycle detection
            if recent[-1] > recent[-2] and recent[-2] < recent[-3]:
                # Local minimum - new cycle start
                cycle_duration = time.time() - self.current_cycle_start
                self.cycle_history.append({
                    'type': 'cycle_end',
                    'duration': cycle_duration,
                    'min_level': recent[-2],
                    'max_level': max(recent)
                })
                self.current_cycle_start = time.time()
                
    def get_motivation_state(self):
        """
        Get current motivation state with trend.
        """
        if len(self.cycle_history) < 5:
            trend = 'insufficient_data'
        else:
            recent = [c['level'] for c in self.cycle_history[-10:]]
            if recent[-1] > np.mean(recent[:-1]):
                trend = 'increasing'
            elif recent[-1] < np.mean(recent[:-1]):
                trend = 'decreasing'
            else:
                trend = 'stable'
                
        return {
            'level': self.motivation_level,
            'trend': trend,
            'cycle_count': len([c for c in self.cycle_history 
                               if c.get('type') == 'cycle_end'])
        }
```

### 9.2 Motivation Renewal Strategies

```python
class MotivationRenewal:
    """
    Implements strategies for motivation renewal when decay is detected.
    """
    
    def __init__(self, motivation_dynamics, curiosity_engine):
        self.motivation_dynamics = motivation_dynamics
        self.curiosity_engine = curiosity_engine
        
    def check_and_renew(self):
        """
        Check motivation level and apply renewal if needed.
        """
        state = self.motivation_dynamics.get_motivation_state()
        
        if state['level'] < 0.3:
            return self.renew_motivation('critical')
        elif state['level'] < 0.5:
            return self.renew_motivation('low')
        elif state['trend'] == 'decreasing':
            return self.renew_motivation('preventive')
        else:
            return None
            
    def renew_motivation(self, renewal_type):
        """
        Apply appropriate renewal strategy.
        """
        strategies = {
            'critical': self.strategy_major_shift,
            'low': self.strategy_new_challenge,
            'preventive': self.strategy_variety_injection
        }
        
        return strategies[renewal_type]()
        
    def strategy_major_shift(self):
        """
        Major strategy: Completely change activity domain.
        """
        return {
            'renewal_type': 'major_shift',
            'action': 'switch_domain',
            'parameters': {
                'target_domain': 'least_explored',
                'commitment_duration': 100
            }
        }
        
    def strategy_new_challenge(self):
        """
        Medium strategy: Introduce new challenge in current domain.
        """
        return {
            'renewal_type': 'new_challenge',
            'action': 'increase_difficulty',
            'parameters': {
                'difficulty_increment': 0.2,
                'target_success_rate': 0.6
            }
        }
        
    def strategy_variety_injection(self):
        """
        Light strategy: Inject variety into current activities.
        """
        return {
            'renewal_type': 'variety_injection',
            'action': 'diversify_approach',
            'parameters': {
                'variation_probability': 0.3,
                'exploration_boost': 1.5
            }
        }
```

---

## 10. Implementation Specifications

### 10.1 Core Class Structure

```python
class SelfDrivenLoop:
    """
    Main Self-Driven Loop integrating all intrinsic motivation components.
    """
    
    def __init__(self, config):
        # Configuration
        self.config = config
        
        # Core components
        self.curiosity_engine = CuriosityEngine(config['curiosity'])
        self.intrinsic_reward_calc = IntrinsicRewardCalculator(config['rewards'])
        self.surprise_detector = SurpriseDetector(config['surprise'])
        self.novelty_tracker = StateNoveltyTracker(config['novelty'])
        self.boredom_detector = BoredomDetector(config['boredom'])
        self.boredom_avoidance = BoredomAvoidance(
            self.boredom_detector, 
            self.curiosity_engine
        )
        self.interest_model = InterestModel(config['interest'])
        self.drive_tracker = DriveSatisfactionTracker()
        self.motivation_dynamics = MotivationDynamics()
        self.motivation_renewal = MotivationRenewal(
            self.motivation_dynamics,
            self.curiosity_engine
        )
        
        # State
        self.loop_state = 'active'
        self.iteration_count = 0
        
    def step(self, state, available_actions, context):
        """
        Execute one iteration of the self-driven loop.
        
        Args:
            state: Current environment state
            available_actions: List of possible actions
            context: Additional context information
            
        Returns:
            selected_action: The action to take
            intrinsic_rewards: Breakdown of intrinsic rewards
            loop_info: Diagnostic information
        """
        self.iteration_count += 1
        
        # 1. Check motivation and renew if needed
        renewal_action = self.motivation_renewal.check_and_renew()
        
        # 2. Check for boredom
        boredom_level, boredom_indicators = self.boredom_detector.update(
            state, None, 0, 0  # Will be updated after action
        )
        
        if self.boredom_detector.is_bored():
            avoidance = self.boredom_avoidance.get_avoidance_action(context)
            # Modify action selection based on avoidance strategy
            
        # 3. Calculate curiosity signals
        curiosity_signals = self.curiosity_engine.compute_all(state, context)
        
        # 4. Get drive priorities
        priority_drive, deficit = self.drive_tracker.get_priority_drive()
        
        # 5. Generate interest-aligned goals
        goals = self.interest_model.generate_goals(context)
        
        # 6. Select action
        selected_action = self.select_action(
            state, 
            available_actions,
            curiosity_signals,
            priority_drive,
            goals,
            context
        )
        
        # 7. Execute action and observe outcome
        # (This happens outside the loop)
        
        return selected_action, {
            'curiosity_signals': curiosity_signals,
            'boredom_level': boredom_level,
            'priority_drive': priority_drive,
            'renewal_action': renewal_action
        }
        
    def update(self, state, action, next_state, extrinsic_reward, info):
        """
        Update loop state based on experience.
        """
        # Calculate intrinsic rewards
        reward_breakdown, total_intrinsic = \
            self.intrinsic_reward_calc.calculate_intrinsic_reward(
                state, action, next_state, info
            )
            
        # Update curiosity engine
        self.curiosity_engine.update(state, action, next_state, 
                                      total_intrinsic)
        
        # Update interest model
        self.interest_model.update(next_state, extrinsic_reward,
                                   reward_breakdown['curiosity'])
        
        # Update drive satisfaction
        self.update_drive_satisfaction(action, extrinsic_reward, 
                                       total_intrinsic)
        
        # Update motivation dynamics
        outcome = {
            'success': extrinsic_reward > 0,
            'magnitude': abs(extrinsic_reward),
            'novelty': reward_breakdown['novelty'],
            'social_interaction': info.get('social', False)
        }
        self.motivation_dynamics.update(outcome)
        
        # Update boredom detector with actual reward
        learning_progress = reward_breakdown['learning_progress']
        self.boredom_detector.update(next_state, action, 
                                      total_intrinsic, learning_progress)
        
        return reward_breakdown
        
    def select_action(self, state, actions, curiosity, drive, goals, context):
        """
        Select action balancing multiple intrinsic factors.
        """
        action_scores = []
        
        for action in actions:
            score = 0.0
            
            # Curiosity component
            score += 0.3 * curiosity.get('total', 0)
            
            # Drive satisfaction component
            drive_alignment = self.estimate_drive_alignment(action, drive)
            score += 0.25 * drive_alignment
            
            # Goal alignment component
            goal_alignment = self.estimate_goal_alignment(action, goals)
            score += 0.25 * goal_alignment
            
            # Interest alignment component
            interest_alignment = self.estimate_interest_alignment(action)
            score += 0.2 * interest_alignment
            
            action_scores.append((action, score))
            
        # Select best action
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        return best_action
```

### 10.2 Configuration Template

```yaml
# self_driven_loop_config.yaml

self_driven_loop:
  enabled: true
  priority: high
  
  curiosity:
    enabled_types:
      - epistemic
      - diversive
      - empowerment
      - social
      - aesthetic
    adaptive_weights: true
    weight_update_frequency: 100
    
  rewards:
    curiosity_weight: 0.30
    surprise_weight: 0.25
    novelty_weight: 0.20
    learning_progress_weight: 0.25
    normalization_window: 1000
    
  surprise:
    threshold: 2.0
    adaptation_rate: 0.01
    memory_capacity: 10000
    
  novelty:
    hash_size: 64
    embedding_dim: 512
    use_simhash: true
    
  boredom:
    repetition_threshold: 0.95
    reward_variance_threshold: 0.01
    stagnation_threshold: 100
    boredom_threshold: 0.7
    state_buffer_size: 100
    
  interest:
    n_dimensions: 10
    learning_rate: 0.1
    decay_rate: 0.001
    
  motivation:
    base_decay_rate: 0.001
    success_boost: 0.1
    novelty_boost: 0.15
    social_boost: 0.08
    
  integration:
    heartbeat_interval: 60  # seconds
    log_level: INFO
    metrics_enabled: true
```

### 10.3 Integration with Other Loops

The Self-Driven Loop integrates with the 14 other agentic loops as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│              SELF-DRIVEN LOOP INTEGRATION MAP                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Self-Driven Loop ◄──────────────────────────────────────┐      │
│         │                                                │      │
│         │ Provides: Intrinsic motivation, curiosity      │      │
│         │         signals, exploration directives        │      │
│         │                                                │      │
│         ▼                                                │      │
│  ┌────────────────────────────────────────────────────┐  │      │
│  │              OTHER AGENTIC LOOPS                    │  │      │
│  │                                                     │  │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │      │
│  │  │ Perception  │  │  Planning   │  │   Action   │  │  │      │
│  │  │    Loop     │  │    Loop     │  │   Loop     │  │  │      │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │      │
│  │  │  Learning   │  │   Memory    │  │ Reflection │  │  │      │
│  │  │    Loop     │  │    Loop     │  │   Loop     │  │  │      │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │      │
│  │  │   Social    │  │   Goal      │  │  Emotion   │  │  │      │
│  │  │    Loop     │  │ Management  │  │   Loop     │  │  │      │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │      │
│  │  │  Creativity │  │  Safety     │  │ Resource   │  │  │      │
│  │  │    Loop     │  │   Loop      │  │ Management │  │  │      │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │  │      │
│  │                                                     │  │      │
│  └────────────────────────────────────────────────────┘  │      │
│                           │                              │      │
│                           │ Returns: Outcomes,           │      │
│                           │          learning signals    │      │
│                           │                              │      │
│                           └──────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 Performance Metrics

| Metric | Description | Target Value |
|--------|-------------|--------------|
| Exploration Coverage | % of state space visited | >80% |
| Curiosity Signal Stability | Variance of curiosity scores | <0.1 |
| Boredom Recovery Time | Steps to escape boredom state | <50 |
| Learning Progress Rate | Improvement in predictions/step | >0.001 |
| Drive Satisfaction | Average drive fulfillment | >0.7 |
| Motivation Sustainability | Time before critical decay | >1000 steps |
| Novelty Detection Accuracy | True positive rate | >85% |
| Surprise Response Time | Steps to adapt to surprises | <10 |

---

## References

1. Pathak, D., et al. (2017). "Curiosity-driven Exploration by Self-supervised Prediction." ICML.
2. Burda, Y., et al. (2019). "Exploration by Random Network Distillation." ICLR.
3. Oudeyer, P-Y., et al. (2016). "Intrinsic motivation, curiosity, and learning." Progress in Brain Research.
4. Le, T. H., et al. (2024). "Improving Exploration Through Surprise Novelty." ICLR.
5. Forestier, S., et al. (2022). "Intrinsically Motivated Goal Exploration Processes." JMLR.
6. Schmidhuber, J. (2010). "Formal Theory of Creativity, Fun, and Intrinsic Motivation." IEEE.

---

## Document Information

- **Version**: 1.0
- **Date**: 2025
- **Author**: AI Systems Research
- **Classification**: Technical Specification
- **Status**: Complete

---

*End of Technical Specification*
