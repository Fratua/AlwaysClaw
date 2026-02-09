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
import logging
import numpy as np

logger = logging.getLogger(__name__)


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

    def train(self, states: List[State], learning_rate: float = 0.01) -> Dict[str, float]:
        """
        Train the feature encoder using a PCA-like update.
        Adjusts the projection matrix so its columns align with the
        principal components of the observed state data.
        """
        if len(states) < 2:
            return {'loss': 0.0, 'updated': False}

        # Build data matrix from raw features
        raw_vectors = []
        for s in states:
            raw = np.array(s.features, dtype=np.float64).flatten()
            raw_vectors.append(raw)

        data = np.array(raw_vectors)  # (N, input_dim)
        input_dim = data.shape[1]

        # Ensure projection is initialised for this input_dim
        if not hasattr(self, '_proj_w') or self._proj_w.shape[0] != input_dim:
            rng = np.random.RandomState(seed=42)
            self._proj_w = rng.randn(input_dim, self.feature_dim) * np.sqrt(2.0 / input_dim)
            self._proj_b = rng.randn(self.feature_dim) * 0.01

        # Centre the data
        mean = data.mean(axis=0)
        centred = data - mean

        # Compute covariance and its top-k eigenvectors (PCA components)
        cov = centred.T @ centred / max(len(centred) - 1, 1)  # (input_dim, input_dim)
        k = min(self.feature_dim, input_dim, len(states))
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            logger.warning("Eigendecomposition failed in feature encoder training")
            return {'loss': 0.0, 'updated': False}

        # Take the top-k eigenvectors (eigh returns ascending order)
        top_indices = np.argsort(eigenvalues)[::-1][:k]
        pca_components = eigenvectors[:, top_indices]  # (input_dim, k)

        # Interpolate current projection towards PCA components
        target_w = np.zeros_like(self._proj_w)
        target_w[:, :k] = pca_components * np.sqrt(2.0 / input_dim)

        # Reconstruction loss before update
        projected = centred @ self._proj_w
        reconstructed = projected @ self._proj_w.T
        loss_before = float(np.mean((centred - reconstructed) ** 2))

        # Gradient step: move projection towards PCA basis
        self._proj_w += learning_rate * (target_w - self._proj_w)

        # Update bias towards data mean projection
        target_b = -mean @ self._proj_w
        self._proj_b += learning_rate * (target_b - self._proj_b)

        # Reconstruction loss after update
        projected_after = centred @ self._proj_w
        reconstructed_after = projected_after @ self._proj_w.T
        loss_after = float(np.mean((centred - reconstructed_after) ** 2))

        logger.debug(
            "FeatureEncoder train: loss %.4f -> %.4f (N=%d)",
            loss_before, loss_after, len(states)
        )

        return {
            'loss_before': loss_before,
            'loss_after': loss_after,
            'loss': loss_after,
            'updated': True,
            'n_samples': len(states),
            'explained_variance_ratio': float(
                eigenvalues[top_indices].sum() / max(eigenvalues.sum(), 1e-12)
            ),
        }


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
    
    def train(self, state_action_pairs: List[Tuple[np.ndarray, Action, np.ndarray]],
              learning_rate: float = 0.001, epochs: int = 5) -> Dict[str, float]:
        """
        Train the forward dynamics model using gradient descent on
        state-action -> next-state prediction pairs.

        Args:
            state_action_pairs: list of (state_features, action, next_state_features)
            learning_rate: step size for weight updates
            epochs: number of passes over the data
        Returns:
            dict with training statistics
        """
        if len(state_action_pairs) < 2:
            return {'loss': 0.0, 'updated': False}

        # Ensure weights are initialised for the right input size
        sample_input = np.concatenate([
            state_action_pairs[0][0],
            self._encode_action(state_action_pairs[0][1])
        ])
        input_size = len(sample_input)
        hidden_size = 16

        if not hasattr(self, '_w1') or self._w1.shape[0] != input_size:
            self._w1 = np.random.randn(input_size, hidden_size) * 0.1
            self._b1 = np.zeros(hidden_size)
            self._w2 = np.random.randn(hidden_size, self.feature_dim) * 0.1
            self._b2 = np.zeros(self.feature_dim)

        total_loss_start = 0.0
        total_loss_end = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Shuffle data each epoch
            indices = np.random.permutation(len(state_action_pairs))

            for idx in indices:
                state_feat, action, next_feat_target = state_action_pairs[idx]
                action_enc = self._encode_action(action)
                x = np.concatenate([state_feat, action_enc])

                # --- Forward pass ---
                z1 = x @ self._w1 + self._b1              # (hidden_size,)
                h = np.tanh(z1)                             # (hidden_size,)
                prediction = h @ self._w2 + self._b2       # (feature_dim,)

                # --- Loss: 0.5 * MSE ---
                error = prediction - next_feat_target       # (feature_dim,)
                loss = 0.5 * np.mean(error ** 2)
                epoch_loss += loss

                # --- Backward pass ---
                d_pred = error / self.feature_dim           # (feature_dim,)
                d_w2 = np.outer(h, d_pred)                  # (hidden, feature_dim)
                d_b2 = d_pred                               # (feature_dim,)
                d_h = d_pred @ self._w2.T                   # (hidden_size,)
                d_z1 = d_h * (1 - h ** 2)                   # tanh derivative
                d_w1 = np.outer(x, d_z1)                    # (input, hidden)
                d_b1 = d_z1                                 # (hidden_size,)

                # --- Gradient step ---
                self._w2 -= learning_rate * d_w2
                self._b2 -= learning_rate * d_b2
                self._w1 -= learning_rate * d_w1
                self._b1 -= learning_rate * d_b1

            avg_loss = epoch_loss / len(state_action_pairs)
            if epoch == 0:
                total_loss_start = avg_loss
            if epoch == epochs - 1:
                total_loss_end = avg_loss

        logger.debug(
            "ForwardDynamicsModel train: loss %.6f -> %.6f (%d epochs, %d samples)",
            total_loss_start, total_loss_end, epochs, len(state_action_pairs)
        )

        return {
            'loss_start': float(total_loss_start),
            'loss_end': float(total_loss_end),
            'loss': float(total_loss_end),
            'updated': True,
            'epochs': epochs,
            'n_samples': len(state_action_pairs),
        }

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
    
    MIN_TRAINING_SAMPLES = 100
    MIN_TRAINING_ITERATIONS = 3

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.forward_model = ForwardDynamicsModel(feature_dim)
        self.inverse_model = InverseDynamicsModel()
        self.feature_encoder = FeatureEncoder(feature_dim)
        self.prediction_history: Deque[Dict] = deque(maxlen=500)

        # Training buffer: stores (state, action, next_state) transitions
        self.training_buffer: Deque[Tuple[State, Action, State]] = deque(maxlen=5000)
        self.training_iterations: int = 0
        self.training_log: List[Dict[str, Any]] = []

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

        # Collect transition for training buffer
        self.training_buffer.append((state, action, next_state))

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

    @property
    def is_trained(self) -> bool:
        """Returns True only after minimum training iterations have been completed."""
        return self.training_iterations >= self.MIN_TRAINING_ITERATIONS

    def train(self, min_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Train both the feature encoder and forward dynamics model using
        the collected training buffer.

        Training only proceeds when the buffer contains at least
        ``min_samples`` transitions (default: MIN_TRAINING_SAMPLES).

        Returns:
            dict with training results from both sub-models and overall stats.
        """
        min_samples = min_samples or self.MIN_TRAINING_SAMPLES
        buffer_size = len(self.training_buffer)

        if buffer_size < min_samples:
            logger.debug(
                "Training skipped: buffer has %d samples, need %d",
                buffer_size, min_samples,
            )
            return {
                'trained': False,
                'reason': f'insufficient data ({buffer_size}/{min_samples})',
                'buffer_size': buffer_size,
            }

        transitions = list(self.training_buffer)

        # --- 1. Train feature encoder (PCA-like update) ---
        all_states = []
        for state, _action, next_state in transitions:
            all_states.append(state)
            all_states.append(next_state)

        encoder_result = self.feature_encoder.train(all_states)

        # --- 2. Prepare encoded pairs for forward model ---
        state_action_pairs: List[Tuple[np.ndarray, Action, np.ndarray]] = []
        for state, action, next_state in transitions:
            state_feat = self.feature_encoder.encode(state)
            next_feat = self.feature_encoder.encode(next_state)
            state_action_pairs.append((state_feat, action, next_feat))

        forward_result = self.forward_model.train(state_action_pairs)

        # --- 3. Update bookkeeping ---
        self.training_iterations += 1

        # Compute curiosity score statistics over recent predictions
        curiosity_stats = self.get_curiosity_stats()

        training_entry = {
            'iteration': self.training_iterations,
            'timestamp': datetime.now().isoformat(),
            'buffer_size': buffer_size,
            'encoder': encoder_result,
            'forward_model': forward_result,
            'curiosity_stats': curiosity_stats,
        }
        self.training_log.append(training_entry)

        logger.info(
            "ICM training iteration %d complete: encoder_loss=%.4f, "
            "forward_loss=%.4f, buffer=%d, curiosity_mean=%.4f",
            self.training_iterations,
            encoder_result.get('loss', 0.0),
            forward_result.get('loss', 0.0),
            buffer_size,
            curiosity_stats.get('mean', 0.0),
        )

        return {
            'trained': True,
            'iteration': self.training_iterations,
            'is_trained': self.is_trained,
            'encoder': encoder_result,
            'forward_model': forward_result,
            'curiosity_stats': curiosity_stats,
        }

    def get_curiosity_stats(self) -> Dict[str, float]:
        """Compute statistics over recent curiosity (prediction error) scores."""
        if not self.prediction_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        errors = [p['prediction_error'] for p in self.prediction_history]
        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'count': len(errors),
        }


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
    logging.basicConfig(level=logging.DEBUG)

    # Example usage
    icm = IntrinsicCuriosityModule()
    print(f"is_trained (before): {icm.is_trained}")

    # Collect transitions into the training buffer
    action_types = ['explore', 'analyze', 'create', 'query', 'execute']
    for i in range(120):
        state = State(id=f"s_{i}", features=np.random.randn(100))
        action = Action(
            id=f"a_{i}",
            action_type=action_types[i % len(action_types)],
            parameters={'step': i},
        )
        next_state = State(id=f"s_{i+1}", features=np.random.randn(100))
        reward = icm.calculate_intrinsic_reward(state, action, next_state)

    print(f"Buffer size: {len(icm.training_buffer)}")

    # Train until is_trained becomes True
    for _ in range(icm.MIN_TRAINING_ITERATIONS):
        result = icm.train()
        print(f"Train iteration {result.get('iteration', '?')}: "
              f"encoder_loss={result.get('encoder', {}).get('loss', 0):.4f}, "
              f"forward_loss={result.get('forward_model', {}).get('loss', 0):.6f}")

    print(f"is_trained (after): {icm.is_trained}")
    stats = icm.get_curiosity_stats()
    print(f"Curiosity stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

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
