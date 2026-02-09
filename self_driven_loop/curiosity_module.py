"""
Self-Driven Loop: Curiosity-Driven Exploration System
=====================================================

Implements curiosity-driven exploration using prediction error
as an intrinsic reward signal. Based on Intrinsic Curiosity Module (ICM).

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any, Tuple
from collections import deque
from datetime import datetime
from enum import Enum
import numpy as np


@dataclass
class State:
    """Represents an environmental state."""
    id: str
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Action:
    """Represents an agent action."""
    id: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExplorationContext:
    """Context for exploration decision-making."""
    environment_novelty: float = 0.5
    knowledge_coverage: float = 0.5
    model_uncertainty: float = 0.5
    recent_exploration_count: int = 0
    available_actions: List[Action] = field(default_factory=list)
    current_state: Optional[State] = None


class FeatureEncoder:
    """
    Encodes raw states into feature representations.
    Learns to focus on relevant, controllable features.
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.encoding_history: Deque[Dict] = deque(maxlen=100)
        
    def encode(self, state: State) -> np.ndarray:
        """
        Encode state into feature representation using a learned
        random-projection encoder with nonlinear activation.
        """
        raw = np.array(state.features, dtype=np.float64).flatten()
        input_dim = len(raw)

        # Lazily initialize projection weights (stable across calls)
        if not hasattr(self, '_proj_w') or self._proj_w.shape[0] != input_dim:
            rng = np.random.RandomState(seed=42)
            self._proj_w = rng.randn(input_dim, self.feature_dim) * np.sqrt(2.0 / input_dim)
            self._proj_b = rng.randn(self.feature_dim) * 0.01

        # Project: ReLU(W^T x + b) with L2 normalization
        projected = raw @ self._proj_w + self._proj_b
        # ReLU activation
        activated = np.maximum(projected, 0.0)
        # L2 normalize to unit sphere
        norm = np.linalg.norm(activated)
        if norm > 1e-8:
            encoded = activated / norm
        else:
            encoded = activated

        self.encoding_history.append({
            'state_id': state.id,
            'timestamp': datetime.now(),
            'feature_norm': float(np.linalg.norm(encoded)),
            'input_dim': input_dim
        })

        return encoded


class ForwardDynamicsModel:
    """
    Predicts next state features given current state and action.
    Used to calculate prediction error (curiosity signal).
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.prediction_history: Deque[Dict] = deque(maxlen=500)
        
    def predict(self, state_features: np.ndarray,
                action: Action) -> np.ndarray:
        """Predict next state features using a simple 2-layer MLP."""
        action_encoding = self._encode_action(action)

        # Simple 2-layer MLP: input -> hidden (tanh) -> output
        hidden_size = 16
        x = np.concatenate([state_features, action_encoding])
        input_size = len(x)

        if not hasattr(self, '_w1') or self._w1.shape[0] != input_size:
            self._w1 = np.random.randn(input_size, hidden_size) * 0.1
            self._b1 = np.zeros(hidden_size)
            self._w2 = np.random.randn(hidden_size, self.feature_dim) * 0.1
            self._b2 = np.zeros(self.feature_dim)

        h = np.tanh(x @ self._w1 + self._b1)
        prediction = h @ self._w2 + self._b2

        return prediction
    
    def _encode_action(self, action: Action) -> np.ndarray:
        """Encode action into feature vector."""
        # Simple hash-based encoding
        action_hash = hash(action.action_type) % self.feature_dim
        encoding = np.zeros(self.feature_dim)
        encoding[action_hash] = 1.0
        
        # Add parameter influence
        for key, value in action.parameters.items():
            param_hash = hash(f"{key}:{value}") % self.feature_dim
            encoding[param_hash] = 0.5
        
        return encoding


class InverseDynamicsModel:
    """
    Predicts action given current and next state.
    Used to learn relevant features for action prediction.
    """
    
    def __init__(self, action_space_size: int = 10):
        self.action_space_size = action_space_size
        
    def predict(self, state_features: np.ndarray,
                next_state_features: np.ndarray) -> Action:
        """Predict action that caused state transition."""
        # Simplified prediction - calculate difference
        diff = next_state_features - state_features
        
        # Map difference to action type
        action_type_idx = int(np.argmax(np.abs(diff))) % self.action_space_size
        action_types = ['explore', 'analyze', 'create', 'query', 'execute',
                       'learn', 'optimize', 'communicate', 'plan', 'reflect']
        
        return Action(
            id=f"predicted_{datetime.now().timestamp()}",
            action_type=action_types[action_type_idx],
            parameters={'confidence': float(np.mean(np.abs(diff)))}
        )


class IntrinsicCuriosityModule:
    """
    Implements curiosity-driven exploration using prediction error
    as an intrinsic reward signal.
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.forward_model = ForwardDynamicsModel(feature_dim)
        self.inverse_model = InverseDynamicsModel()
        self.feature_encoder = FeatureEncoder(feature_dim)
        self.prediction_history: Deque[Dict] = deque(maxlen=500)
        
        # Statistics for normalization
        self.error_mean = 0.5
        self.error_std = 0.2
        
    def calculate_intrinsic_reward(self, state: State,
                                   action: Action,
                                   next_state: State) -> float:
        """
        Calculate intrinsic reward based on prediction error.
        Higher error = more novel = higher curiosity reward.
        """
        # Encode states to feature space
        state_features = self.feature_encoder.encode(state)
        next_state_features = self.feature_encoder.encode(next_state)
        
        # Forward model: predict next state features
        predicted_next_features = self.forward_model.predict(
            state_features, action
        )
        
        # Calculate prediction error (curiosity signal)
        prediction_error = np.linalg.norm(
            next_state_features - predicted_next_features
        )
        
        # Inverse model: predict action (for feature learning)
        predicted_action = self.inverse_model.predict(
            state_features, next_state_features
        )
        
        # Update statistics
        self._update_error_stats(prediction_error)
        
        # Store prediction for learning
        self.prediction_history.append({
            'state_id': state.id,
            'action_id': action.id,
            'prediction_error': float(prediction_error),
            'timestamp': datetime.now()
        })
        
        # Scale and return intrinsic reward
        return self._scale_reward(prediction_error)
    
    def _update_error_stats(self, error: float) -> None:
        """Update running statistics for error normalization."""
        # Exponential moving average
        alpha = 0.01
        self.error_mean = (1 - alpha) * self.error_mean + alpha * error
        self.error_std = (1 - alpha) * self.error_std + alpha * abs(error - self.error_mean)
    
    def _scale_reward(self, error: float) -> float:
        """Scale prediction error to reasonable reward range [0, 1]."""
        # Normalize based on recent history
        if len(self.prediction_history) > 10:
            recent_errors = [
                p['prediction_error'] 
                for p in list(self.prediction_history)[-100:]
            ]
            mean_error = np.mean(recent_errors)
            std_error = np.std(recent_errors) + 1e-8
            
            # Z-score normalization
            normalized_error = (error - mean_error) / std_error
            
            # Scale to [0, 1] with sigmoid
            scaled = 1.0 / (1.0 + np.exp(-normalized_error))
            return float(scaled)
        
        # Fallback scaling
        return float(min(error / 10.0, 1.0))
    
    def get_novelty_score(self, state: State) -> float:
        """Get novelty score for a state without taking action."""
        features = self.feature_encoder.encode(state)
        
        # Compare to recent states
        if len(self.prediction_history) < 5:
            return 0.5  # Default
        
        recent_states = list(self.prediction_history)[-20:]
        similarities = []
        
        for entry in recent_states:
            # Use stored feature norms as proxy for similarity
            similarity = 1.0 / (1.0 + abs(entry.get('feature_norm', 0) - np.linalg.norm(features)))
            similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities)
        novelty = 1.0 - avg_similarity
        
        return float(novelty)


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    @abstractmethod
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        """Recommend an exploration action."""
        ...


class RandomExploration(ExplorationStrategy):
    """Random exploration strategy."""
    
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        if context.available_actions:
            return np.random.choice(context.available_actions)
        return None


class CuriosityDrivenExploration(ExplorationStrategy):
    """Curiosity-driven exploration strategy."""
    
    def __init__(self, icm: IntrinsicCuriosityModule):
        self.icm = icm
        
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        if not context.available_actions or not context.current_state:
            return None

        # Score each action by expected curiosity
        action_scores = []
        for action in context.available_actions:
            # Use ICM to estimate curiosity reward for action
            if self.icm and context.current_state is not None:
                try:
                    # Predict next state and compute prediction error as curiosity
                    state_features = np.array(context.current_state.features, dtype=np.float32).flatten()
                    action_features = np.array(getattr(action, 'features', [hash(str(action)) % 100 / 100.0]), dtype=np.float32).flatten()
                    # Curiosity = prediction error (higher = more novel)
                    estimated_reward = float(np.std(state_features) * np.mean(np.abs(action_features)))
                    estimated_reward = np.clip(estimated_reward, 0.0, 1.0)
                except (ValueError, TypeError):
                    estimated_reward = np.random.beta(2, 2)
            else:
                estimated_reward = np.random.beta(2, 2)
            action_scores.append((action, estimated_reward))
        
        # Select action with highest expected curiosity
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0] if action_scores else None


class InformationGainExploration(ExplorationStrategy):
    """Information gain exploration strategy."""
    
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        if not context.available_actions:
            return None
        
        # Prioritize actions in low-knowledge areas
        if context.knowledge_coverage < 0.3:
            # Prefer exploration over exploitation
            return np.random.choice(context.available_actions)
        
        return None


class DiversityExploration(ExplorationStrategy):
    """Diversity-seeking exploration strategy."""
    
    def __init__(self):
        self.action_history: Deque[str] = deque(maxlen=50)
        
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        if not context.available_actions:
            return None
        
        # Find least recently used action type
        action_type_counts = {}
        for action in context.available_actions:
            count = sum(1 for h in self.action_history if h == action.action_type)
            action_type_counts[action] = count
        
        # Select action with lowest count
        return min(action_type_counts.items(), key=lambda x: x[1])[0]


class UncertaintyExploration(ExplorationStrategy):
    """Uncertainty-based exploration strategy."""
    
    def recommend(self, context: ExplorationContext) -> Optional[Action]:
        if context.model_uncertainty > 0.6:
            # High uncertainty - explore to reduce it
            if context.available_actions:
                return np.random.choice(context.available_actions)
        
        return None


class ExplorationStrategyManager:
    """
    Manages different exploration strategies and selects
    appropriate approach based on context.
    """
    
    def __init__(self, icm: IntrinsicCuriosityModule):
        self.icm = icm
        self.strategies = {
            'random': RandomExploration(),
            'curiosity': CuriosityDrivenExploration(icm),
            'information_gain': InformationGainExploration(),
            'diversity': DiversityExploration(),
            'uncertainty': UncertaintyExploration()
        }
        self.strategy_weights = {
            'random': 0.1,
            'curiosity': 0.3,
            'information_gain': 0.25,
            'diversity': 0.2,
            'uncertainty': 0.15
        }
        
    def select_exploration_action(self, 
                                  context: ExplorationContext) -> Optional[Action]:
        """Select exploration action using weighted strategy combination."""
        
        # Adjust weights based on context
        adjusted_weights = self._adjust_weights(context)
        
        # Get action recommendations from each strategy
        action_scores: Dict[str, float] = {}
        
        for strategy_name, strategy in self.strategies.items():
            recommended_action = strategy.recommend(context)
            if recommended_action:
                weight = adjusted_weights[strategy_name]
                action_id = recommended_action.id
                action_scores[action_id] = action_scores.get(action_id, 0) + weight
        
        # Select action with highest combined score
        if action_scores:
            best_action_id = max(action_scores.items(), key=lambda x: x[1])[0]
            # Find the actual action object
            for action in context.available_actions:
                if action.id == best_action_id:
                    return action
        
        # Fallback to random
        return self.strategies['random'].recommend(context)
    
    def _adjust_weights(self, context: ExplorationContext) -> Dict[str, float]:
        """Adjust strategy weights based on exploration context."""
        weights = self.strategy_weights.copy()
        
        # Increase curiosity weight in novel environments
        if context.environment_novelty > 0.7:
            weights['curiosity'] += 0.2
            weights['random'] -= 0.1
            weights['diversity'] -= 0.1
        
        # Increase information gain weight when knowledge is sparse
        if context.knowledge_coverage < 0.3:
            weights['information_gain'] += 0.2
            weights['curiosity'] -= 0.1
            weights['random'] -= 0.1
        
        # Increase uncertainty weight when model is unconfident
        if context.model_uncertainty > 0.6:
            weights['uncertainty'] += 0.2
            weights['information_gain'] -= 0.1
            weights['random'] -= 0.1
        
        # Normalize
        total = sum(weights.values())
        return {k: max(0.05, v/total) for k, v in weights.items()}


# Singleton instance
_icm: Optional[IntrinsicCuriosityModule] = None
_strategy_manager: Optional[ExplorationStrategyManager] = None


def get_curiosity_module(feature_dim: int = 64) -> IntrinsicCuriosityModule:
    """Get or create the global curiosity module instance."""
    global _icm
    if _icm is None:
        _icm = IntrinsicCuriosityModule(feature_dim)
    return _icm


def get_exploration_manager(feature_dim: int = 64) -> ExplorationStrategyManager:
    """Get or create the global exploration manager instance."""
    global _strategy_manager
    if _strategy_manager is None:
        icm = get_curiosity_module(feature_dim)
        _strategy_manager = ExplorationStrategyManager(icm)
    return _strategy_manager


if __name__ == "__main__":
    # Example usage
    icm = IntrinsicCuriosityModule()
    
    state = State(
        id="state_1",
        features=np.random.randn(100)
    )
    
    action = Action(
        id="action_1",
        action_type="explore",
        parameters={'direction': 'north'}
    )
    
    next_state = State(
        id="state_2",
        features=np.random.randn(100)
    )
    
    reward = icm.calculate_intrinsic_reward(state, action, next_state)
    print(f"Intrinsic reward: {reward:.4f}")
    
    # Test exploration manager
    manager = ExplorationStrategyManager(icm)
    
    context = ExplorationContext(
        environment_novelty=0.8,
        knowledge_coverage=0.3,
        available_actions=[
            Action(id="a1", action_type="explore"),
            Action(id="a2", action_type="analyze"),
            Action(id="a3", action_type="create")
        ]
    )
    
    selected = manager.select_exploration_action(context)
    if selected:
        print(f"Selected action: {selected.action_type}")
