"""
Advanced Bug Finder Loop - ML-Based Predictive Bug Detection
Windows 10 OpenClaw-Inspired AI Agent System

This module implements the core Bug Finder Loop with ML-based predictive
bug detection capabilities including anomaly detection, pattern learning,
and proactive alerting.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, deque
import json
import pickle
from pathlib import Path

# ML imports - optional, graceful fallback
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, Conv1D, MaxPooling1D, Flatten)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class FeatureVector:
    """Container for engineered features"""
    system_features: Dict[str, float] = field(default_factory=dict)
    log_features: Dict[str, float] = field(default_factory=dict)
    agent_features: Dict[str, float] = field(default_factory=dict)
    temporal_features: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        all_features = {
            **self.system_features,
            **self.log_features,
            **self.agent_features,
            **self.temporal_features
        }
        return np.array(list(all_features.values()))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            **self.system_features,
            **self.log_features,
            **self.agent_features,
            **self.temporal_features
        }


@dataclass
class ModelOutput:
    """Output from a single model"""
    model_name: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    raw_output: Any = None


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble"""
    individual_predictions: Dict[str, ModelOutput]
    ensemble_score: float
    is_anomaly: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def model_agreement(self) -> float:
        """Calculate agreement ratio among models"""
        predictions = [p.is_anomaly for p in self.individual_predictions.values()]
        if not predictions:
            return 0.0
        majority = max(set(predictions), key=predictions.count)
        return predictions.count(majority) / len(predictions)


@dataclass
class Prediction:
    """A bug prediction with metadata"""
    id: str
    prediction_type: str
    predicted_time: datetime
    confidence: float
    severity: AlertSeverity
    contributing_factors: List[str]
    affected_components: List[str] = field(default_factory=list)
    current_value: Optional[float] = None
    predicted_value: Optional[float] = None
    threshold: Optional[float] = None
    similar_incidents: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_anomaly(self) -> bool:
        return self.confidence >= 0.5


@dataclass
class SHAPExplanation:
    """SHAP-based explanation"""
    base_value: float
    prediction_value: float
    feature_importance: List[Tuple[str, float]]
    top_positive_contributors: List[Tuple[str, float]]
    top_negative_contributors: List[Tuple[str, float]]
    shap_values: List[float]


@dataclass
class BugAlert:
    """Bug alert with explanation"""
    prediction: Prediction
    explanation: str
    shap_explanation: Optional[SHAPExplanation] = None
    recommended_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the bug finder"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    avg_prediction_latency_ms: float
    total_predictions: int


# =============================================================================
# ML ENSEMBLE DETECTOR
# =============================================================================

class MLEnsembleDetector:
    """
    Ensemble ML detector combining multiple algorithms for robust
    anomaly detection in the AI agent system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
        # Ensemble weights
        self.ensemble_weights = self.config.get('ensemble_weights', {
            'isolation_forest': 0.25,
            'lstm_autoencoder': 0.30,
            'xgboost': 0.25,
            'one_class_svm': 0.10,
            'prophet': 0.10
        })
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ML models with graceful fallback for missing libraries."""
        if HAS_SKLEARN:
            # Isolation Forest for unsupervised anomaly detection
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=self.config.get('if_n_estimators', 200),
                contamination=self.config.get('contamination', 0.05),
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )

            # One-Class SVM for novelty detection
            self.models['one_class_svm'] = OneClassSVM(
                kernel=self.config.get('svm_kernel', 'rbf'),
                gamma='scale',
                nu=self.config.get('svm_nu', 0.05)
            )

            # Scalers
            self.scalers['standard'] = StandardScaler()
        else:
            logger.warning("scikit-learn not available: isolation_forest, one_class_svm, scaler disabled")
            self.models['isolation_forest'] = None
            self.models['one_class_svm'] = None
            self.scalers['standard'] = None

        if HAS_XGBOOST:
            # XGBoost for supervised classification (if labels available)
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=self.config.get('xgb_n_estimators', 200),
                max_depth=self.config.get('xgb_max_depth', 8),
                learning_rate=self.config.get('xgb_learning_rate', 0.05),
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False
            )
        else:
            logger.warning("xgboost not available: xgboost classifier disabled")
            self.models['xgboost'] = None

        # LSTM Autoencoder will be built when data shape is known
        self.models['lstm_autoencoder'] = None
        self.lstm_sequence_length = self.config.get('sequence_length', 50)
        if not HAS_TENSORFLOW:
            logger.warning("tensorflow not available: LSTM autoencoder disabled")

        # Prophet models per metric (initialized on demand)
        self.models['prophet'] = {}
        if not HAS_PROPHET:
            logger.warning("prophet not available: time-series forecasting disabled")
        
    def build_lstm_model(self, n_features: int = 1):
        """Build LSTM autoencoder model. Returns None if tensorflow is unavailable."""
        if not HAS_TENSORFLOW:
            logger.warning("Cannot build LSTM model: tensorflow not installed")
            return None
        seq_len = self.lstm_sequence_length

        model = Sequential([
            # Encoder
            LSTM(128, activation='relu', input_shape=(seq_len, n_features), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            
            # Bottleneck
            RepeatVector(seq_len),
            
            # Decoder
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit all models in the ensemble
        
        Args:
            X: Feature matrix
            y: Optional labels (for supervised models)
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics
        """
        logger.info(f"Fitting ensemble on {len(X)} samples")
        
        metrics = {}
        
        # Scale features
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Fit Isolation Forest
        logger.info("Training Isolation Forest...")
        self.models['isolation_forest'].fit(X_scaled)
        metrics['isolation_forest'] = {'status': 'trained'}
        
        # Fit One-Class SVM
        logger.info("Training One-Class SVM...")
        self.models['one_class_svm'].fit(X_scaled)
        metrics['one_class_svm'] = {'status': 'trained'}
        
        # Fit XGBoost if labels available
        if y is not None:
            logger.info("Training XGBoost classifier...")
            self.models['xgboost'].fit(
                X_scaled, y,
                eval_set=[(X_scaled, y)],
                verbose=False
            )
            metrics['xgboost'] = {'status': 'trained'}
        
        # Fit LSTM Autoencoder
        logger.info("Training LSTM Autoencoder...")
        if len(X) >= self.lstm_sequence_length:
            X_seq = self._create_sequences(X_scaled)
            n_features = X_seq.shape[2]
            
            self.models['lstm_autoencoder'] = self.build_lstm_model(n_features)
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.models['lstm_autoencoder'].fit(
                X_seq, X_seq,
                epochs=self.config.get('lstm_epochs', 100),
                batch_size=self.config.get('lstm_batch_size', 64),
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=1
            )
            
            metrics['lstm_autoencoder'] = {
                'status': 'trained',
                'final_loss': history.history['loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
        
        self.is_fitted = True
        logger.info("Ensemble training complete")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Generate ensemble prediction
        
        Args:
            X: Feature matrix (single sample or batch)
            
        Returns:
            Ensemble prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before prediction")
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        
        # Isolation Forest prediction
        if_pred = self.models['isolation_forest'].predict(X_scaled)
        if_scores = -self.models['isolation_forest'].score_samples(X_scaled)
        predictions['isolation_forest'] = ModelOutput(
            model_name='isolation_forest',
            is_anomaly=if_pred[0] == -1,
            anomaly_score=float(if_scores[0]),
            confidence=min(1.0, if_scores[0] / np.percentile(if_scores, 95))
        )
        
        # One-Class SVM prediction
        svm_pred = self.models['one_class_svm'].predict(X_scaled)
        svm_scores = -self.models['one_class_svm'].score_samples(X_scaled)
        predictions['one_class_svm'] = ModelOutput(
            model_name='one_class_svm',
            is_anomaly=svm_pred[0] == -1,
            anomaly_score=float(svm_scores[0]),
            confidence=min(1.0, svm_scores[0] / np.percentile(svm_scores, 95))
        )
        
        # XGBoost prediction
        if hasattr(self.models['xgboost'], 'classes_'):
            xgb_proba = self.models['xgboost'].predict_proba(X_scaled)[0]
            predictions['xgboost'] = ModelOutput(
                model_name='xgboost',
                is_anomaly=xgb_proba[1] > 0.5,
                anomaly_score=float(xgb_proba[1]),
                confidence=float(max(xgb_proba))
            )
        
        # LSTM Autoencoder prediction
        if self.models['lstm_autoencoder'] is not None:
            X_seq = self._create_sequences(X_scaled)
            reconstructed = self.models['lstm_autoencoder'].predict(X_seq, verbose=0)
            
            # Calculate reconstruction error
            mse = np.mean(np.power(X_seq - reconstructed, 2), axis=(1, 2))
            
            predictions['lstm_autoencoder'] = ModelOutput(
                model_name='lstm_autoencoder',
                is_anomaly=mse[0] > np.percentile(mse, 95),
                anomaly_score=float(mse[0]),
                confidence=min(1.0, mse[0] / np.percentile(mse, 99))
            )
        
        # Calculate ensemble score
        ensemble_score = self._calculate_ensemble_score(predictions)
        
        return EnsemblePrediction(
            individual_predictions=predictions,
            ensemble_score=ensemble_score,
            is_anomaly=ensemble_score > 0.5,
            confidence=ensemble_score
        )
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM"""
        sequences = []
        for i in range(len(data) - self.lstm_sequence_length + 1):
            seq = data[i:i + self.lstm_sequence_length]
            sequences.append(seq)
        return np.array(sequences)
    
    def _calculate_ensemble_score(self, predictions: Dict[str, ModelOutput]) -> float:
        """Calculate weighted ensemble score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 0.2)
            weighted_sum += pred.anomaly_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def save_models(self, path: str):
        """Save all models to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn models
        for name in ['isolation_forest', 'one_class_svm', 'xgboost']:
            if name in self.models:
                with open(path / f"{name}.pkl", 'wb') as f:
                    pickle.dump(self.models[name], f)
        
        # Save LSTM model
        if self.models['lstm_autoencoder'] is not None:
            self.models['lstm_autoencoder'].save(path / "lstm_autoencoder.h5")
        
        # Save scalers
        with open(path / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump({
                'ensemble_weights': self.ensemble_weights,
                'sequence_length': self.lstm_sequence_length,
                'config': self.config
            }, f)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load all models from disk"""
        path = Path(path)
        
        # Load sklearn models
        for name in ['isolation_forest', 'one_class_svm', 'xgboost']:
            model_path = path / f"{name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
        
        # Load LSTM model
        lstm_path = path / "lstm_autoencoder.h5"
        if lstm_path.exists():
            self.models['lstm_autoencoder'] = load_model(lstm_path)
        
        # Load scalers
        scalers_path = path / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                self.ensemble_weights = saved_config['ensemble_weights']
                self.lstm_sequence_length = saved_config['sequence_length']
        
        self.is_fitted = True
        logger.info(f"Models loaded from {path}")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering pipeline for bug detection
    """
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size  # seconds
        self.feature_history = deque(maxlen=1000)
        
    def extract_features(self, 
                        system_metrics: Dict[str, List[float]],
                        log_data: Dict[str, Any],
                        agent_metrics: Dict[str, Any],
                        timestamp: Optional[datetime] = None) -> FeatureVector:
        """
        Extract features from raw data
        
        Args:
            system_metrics: Dictionary of system metric time series
            log_data: Log analysis results
            agent_metrics: Agent execution metrics
            timestamp: Current timestamp
            
        Returns:
            FeatureVector with engineered features
        """
        timestamp = timestamp or datetime.now()
        
        # System features
        system_features = self._extract_system_features(system_metrics)
        
        # Log features
        log_features = self._extract_log_features(log_data)
        
        # Agent features
        agent_features = self._extract_agent_features(agent_metrics)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(timestamp)
        
        features = FeatureVector(
            system_features=system_features,
            log_features=log_features,
            agent_features=agent_features,
            temporal_features=temporal_features
        )
        
        self.feature_history.append(features)
        
        return features
    
    def _extract_system_features(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Extract features from system metrics"""
        features = {}
        
        for metric_name, values in metrics.items():
            if not values:
                continue
            
            arr = np.array(values)
            prefix = metric_name.replace(' ', '_').lower()
            
            # Basic statistics
            features[f'{prefix}_mean'] = float(np.mean(arr))
            features[f'{prefix}_std'] = float(np.std(arr))
            features[f'{prefix}_max'] = float(np.max(arr))
            features[f'{prefix}_min'] = float(np.min(arr))
            
            # Trend (slope of linear regression)
            if len(arr) > 1:
                x = np.arange(len(arr))
                slope = np.polyfit(x, arr, 1)[0]
                features[f'{prefix}_trend'] = float(slope)
            
            # Rate of change
            if len(arr) > 1:
                roc = (arr[-1] - arr[0]) / len(arr)
                features[f'{prefix}_roc'] = float(roc)
            
            # Spike detection
            threshold = np.mean(arr) + 2 * np.std(arr)
            features[f'{prefix}_spikes'] = float(np.sum(arr > threshold))
        
        return features
    
    def _extract_log_features(self, log_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from log data"""
        features = {}
        
        # Error and warning rates
        features['error_rate'] = log_data.get('error_count', 0) / max(log_data.get('total_logs', 1), 1)
        features['warning_rate'] = log_data.get('warning_count', 0) / max(log_data.get('total_logs', 1), 1)
        
        # Log volume
        features['log_volume'] = float(log_data.get('total_logs', 0))
        features['unique_templates'] = float(log_data.get('unique_templates', 0))
        
        # Exception indicators
        features['exception_count'] = float(log_data.get('exception_count', 0))
        features['stacktrace_count'] = float(log_data.get('stacktrace_count', 0))
        features['timeout_count'] = float(log_data.get('timeout_count', 0))
        
        # Log entropy (diversity of log types)
        if 'template_counts' in log_data:
            counts = np.array(list(log_data['template_counts'].values()))
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['log_entropy'] = float(entropy)
        
        return features
    
    def _extract_agent_features(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from agent metrics"""
        features = {}
        
        # Loop execution
        if 'loop_durations' in metrics:
            durations = metrics['loop_durations']
            features['loop_duration_mean'] = float(np.mean(durations))
            features['loop_duration_max'] = float(np.max(durations))
            features['loop_timeout_rate'] = float(
                np.sum(np.array(durations) > 30) / len(durations)
            )
        
        # API metrics
        features['api_error_rate'] = metrics.get('api_errors', 0) / max(metrics.get('api_calls', 1), 1)
        features['api_latency_mean'] = float(metrics.get('avg_api_latency', 0))
        features['rate_limit_hits'] = float(metrics.get('rate_limit_count', 0))
        
        # Task queue
        features['queue_depth'] = float(metrics.get('queue_depth', 0))
        features['success_rate'] = metrics.get('tasks_successful', 0) / max(metrics.get('tasks_total', 1), 1)
        features['consecutive_failures'] = float(metrics.get('consecutive_failures', 0))
        
        # Integration errors
        features['gmail_errors'] = float(metrics.get('gmail_error_count', 0))
        features['browser_errors'] = float(metrics.get('browser_error_count', 0))
        features['tts_errors'] = float(metrics.get('tts_error_count', 0))
        features['stt_errors'] = float(metrics.get('stt_error_count', 0))
        features['twilio_errors'] = float(metrics.get('twilio_error_count', 0))
        
        return features
    
    def _extract_temporal_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract temporal features"""
        features = {}
        
        # Hour of day (cyclical encoding)
        hour = timestamp.hour
        features['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
        features['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
        
        # Day of week (cyclical encoding)
        dow = timestamp.weekday()
        features['dow_sin'] = float(np.sin(2 * np.pi * dow / 7))
        features['dow_cos'] = float(np.cos(2 * np.pi * dow / 7))
        
        # Business hours
        features['is_business_hours'] = float(9 <= hour < 17 and dow < 5)
        
        # Weekend
        features['is_weekend'] = float(dow >= 5)
        
        return features


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

class ConfidenceScorer:
    """
    Multi-factor confidence scoring for predictions
    """
    
    FACTOR_WEIGHTS = {
        'model_agreement': 0.25,
        'prediction_stability': 0.20,
        'historical_accuracy': 0.20,
        'data_quality': 0.15,
        'pattern_strength': 0.10,
        'temporal_proximity': 0.10
    }
    
    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=1000)
        
    def calculate_confidence(self,
                           prediction: EnsemblePrediction,
                           context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence score
        
        Args:
            prediction: Ensemble prediction
            context: Additional context
            
        Returns:
            Confidence score details
        """
        factor_scores = {}
        
        # Model agreement
        factor_scores['model_agreement'] = prediction.model_agreement
        
        # Prediction stability
        factor_scores['prediction_stability'] = self._calculate_stability(prediction)
        
        # Historical accuracy (placeholder - would use actual feedback)
        factor_scores['historical_accuracy'] = 0.7
        
        # Data quality
        factor_scores['data_quality'] = context.get('data_quality', 0.8) if context else 0.8
        
        # Pattern strength
        factor_scores['pattern_strength'] = min(1.0, prediction.ensemble_score * 1.2)
        
        # Temporal proximity
        factor_scores['temporal_proximity'] = 0.8  # Placeholder
        
        # Calculate weighted confidence
        overall = sum(
            factor_scores[k] * self.FACTOR_WEIGHTS[k]
            for k in self.FACTOR_WEIGHTS.keys()
        )
        
        # Classify level
        level = self._classify_level(overall)
        
        return {
            'overall_confidence': overall,
            'confidence_level': level,
            'factor_scores': factor_scores,
            'is_confident': overall >= 0.7
        }
    
    def _calculate_stability(self, prediction: EnsemblePrediction) -> float:
        """Calculate prediction stability over time"""
        # Compare with recent similar predictions
        if not self.prediction_history:
            return 0.5
        
        recent = list(self.prediction_history)[-10:]
        if not recent:
            return 0.5
        
        scores = [p.ensemble_score for p in recent]
        variance = np.var(scores)
        
        return 1.0 - min(1.0, variance * 10)
    
    def _classify_level(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.75:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def record_prediction(self, prediction: EnsemblePrediction):
        """Record prediction for stability tracking"""
        self.prediction_history.append(prediction)


# =============================================================================
# EXPLANATION GENERATOR
# =============================================================================

class ExplanationGenerator:
    """
    Generate human-readable explanations for predictions
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.background_data = None
        
    def initialize(self, background_data: np.ndarray, model=None):
        """Initialize explainers with background data"""
        self.background_data = background_data
        
        # Sample background for efficiency
        if len(background_data) > 100:
            bg_sample = background_data[np.random.choice(
                len(background_data), 100, replace=False
            )]
        else:
            bg_sample = background_data
        
        # Initialize SHAP
        if model is not None:
            self.shap_explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                bg_sample
            )
    
    def generate_explanation(self,
                           prediction: EnsemblePrediction,
                           features: FeatureVector,
                           feature_array: np.ndarray) -> str:
        """
        Generate natural language explanation
        
        Args:
            prediction: Ensemble prediction
            features: Feature vector
            feature_array: Numpy array of features
            
        Returns:
            Human-readable explanation
        """
        parts = []
        
        # Summary
        severity = self._classify_severity(prediction.ensemble_score)
        parts.append(
            f"{severity.upper()}: Anomaly detected with "
            f"{prediction.confidence*100:.1f}% confidence"
        )
        
        # Contributing models
        model_contributions = []
        for name, output in prediction.individual_predictions.items():
            if output.is_anomaly:
                model_contributions.append(
                    f"{name} (score: {output.anomaly_score:.3f})"
                )
        
        if model_contributions:
            parts.append(f"Contributing models: {', '.join(model_contributions)}")
        
        # Top features
        top_features = self._identify_top_features(features)
        if top_features:
            parts.append("Key indicators:")
            for name, value in top_features[:5]:
                parts.append(f"  • {name}: {value:.3f}")
        
        # Recommendations
        recommendations = self._generate_recommendations(features, prediction)
        if recommendations:
            parts.append("Recommendations:")
            for rec in recommendations:
                parts.append(f"  • {rec}")
        
        return '\n'.join(parts)
    
    def generate_shap_explanation(self,
                                 feature_array: np.ndarray) -> Optional[SHAPExplanation]:
        """Generate SHAP explanation"""
        if self.shap_explainer is None:
            return None
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(feature_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Anomaly class
        
        # Create feature importance list
        feature_dict = dict(zip(self.feature_names, shap_values[0]))
        sorted_features = sorted(
            feature_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Split into positive and negative contributors
        positive = [(n, v) for n, v in sorted_features if v > 0][:5]
        negative = [(n, v) for n, v in sorted_features if v < 0][:5]
        
        return SHAPExplanation(
            base_value=float(self.shap_explainer.expected_value),
            prediction_value=float(self.shap_explainer.expected_value + np.sum(shap_values)),
            feature_importance=sorted_features,
            top_positive_contributors=positive,
            top_negative_contributors=negative,
            shap_values=shap_values[0].tolist()
        )
    
    def _classify_severity(self, score: float) -> str:
        """Classify severity from score"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _identify_top_features(self, features: FeatureVector) -> List[Tuple[str, float]]:
        """Identify top contributing features"""
        all_features = features.to_dict()
        
        # Sort by absolute value
        sorted_features = sorted(
            all_features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features
    
    def _generate_recommendations(self,
                                 features: FeatureVector,
                                 prediction: EnsemblePrediction) -> List[str]:
        """Generate recommendations based on features"""
        recommendations = []
        f = features.to_dict()
        
        # CPU recommendations
        if f.get('cpu_mean', 0) > 80:
            recommendations.append("Investigate high CPU usage processes")
        
        # Memory recommendations
        if f.get('memory_mean', 0) > 85:
            recommendations.append("Check for memory leaks or increase available memory")
        
        # Error recommendations
        if f.get('error_rate', 0) > 0.1:
            recommendations.append("Review recent error logs for root cause")
        
        # Loop recommendations
        if f.get('loop_timeout_rate', 0) > 0.1:
            recommendations.append("Optimize slow-running loops or increase timeouts")
        
        # Queue recommendations
        if f.get('queue_depth', 0) > 100:
            recommendations.append("Scale processing capacity or reduce incoming load")
        
        # API recommendations
        if f.get('api_error_rate', 0) > 0.05:
            recommendations.append("Check external API status and network connectivity")
        
        return recommendations


# =============================================================================
# FALSE POSITIVE REDUCTION
# =============================================================================

class FalsePositiveReducer:
    """
    Multi-layer false positive reduction system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_model_agreement = self.config.get('min_model_agreement', 0.6)
        self.min_duration_seconds = self.config.get('min_duration_seconds', 60)
        
        self.anomaly_history = {}
        self.maintenance_windows = []
        
    def filter_predictions(self, 
                          predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """
        Apply all filters to reduce false positives
        
        Args:
            predictions: List of predictions to filter
            
        Returns:
            Filtered predictions
        """
        filtered = predictions
        
        # Filter by confidence
        filtered = [p for p in filtered if p.confidence >= self.min_confidence]
        
        # Filter by model agreement
        filtered = [p for p in filtered if p.model_agreement >= self.min_model_agreement]
        
        # Filter by temporal consistency
        filtered = self._filter_by_duration(filtered)
        
        # Filter by context
        filtered = self._filter_by_context(filtered)
        
        return filtered
    
    def _filter_by_duration(self, 
                           predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Filter predictions that don't persist over time"""
        filtered = []
        current_time = datetime.now()
        
        for pred in predictions:
            # Create key based on prediction characteristics
            key = f"anomaly_{pred.ensemble_score:.3f}"
            
            if key not in self.anomaly_history:
                self.anomaly_history[key] = {
                    'first_seen': current_time,
                    'count': 1
                }
            else:
                self.anomaly_history[key]['count'] += 1
            
            # Check duration
            duration = (current_time - self.anomaly_history[key]['first_seen']).total_seconds()
            
            if duration >= self.min_duration_seconds:
                filtered.append(pred)
        
        # Cleanup old entries
        self._cleanup_history(current_time)
        
        return filtered
    
    def _filter_by_context(self, 
                          predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Filter based on contextual information"""
        filtered = []
        current_time = datetime.now()
        
        for pred in predictions:
            # Skip if during maintenance
            if self._is_maintenance_window(current_time):
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def _is_maintenance_window(self, timestamp: datetime) -> bool:
        """Check if in maintenance window"""
        for window in self.maintenance_windows:
            if window['start'] <= timestamp <= window['end']:
                return True
        return False
    
    def _cleanup_history(self, current_time: datetime):
        """Remove old history entries"""
        cutoff = current_time - timedelta(seconds=self.min_duration_seconds * 2)
        
        self.anomaly_history = {
            k: v for k, v in self.anomaly_history.items()
            if v['first_seen'] > cutoff
        }
    
    def add_maintenance_window(self, start: datetime, end: datetime):
        """Add a maintenance window"""
        self.maintenance_windows.append({'start': start, 'end': end})


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """
    Manage and dispatch bug alerts
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alert_handlers = []
        self.alert_history = deque(maxlen=1000)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 300)
        self.last_alert_time = {}
        
    def register_handler(self, handler: callable):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
        
    async def send_alert(self, alert: BugAlert):
        """
        Send alert through all registered handlers
        
        Args:
            alert: Bug alert to send
        """
        # Check cooldown
        pred_type = alert.prediction.prediction_type
        last_time = self.last_alert_time.get(pred_type)
        
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.debug(f"Alert for {pred_type} in cooldown")
                return
        
        # Send to all handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Alert handler error: {e}")
        
        # Update history
        self.alert_history.append(alert)
        self.last_alert_time[pred_type] = datetime.now()
        
        logger.info(f"Alert sent: {alert.prediction.id}")
    
    def get_recent_alerts(self, 
                         minutes: int = 60) -> List[BugAlert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.alert_history if a.timestamp > cutoff]


# =============================================================================
# MAIN BUG FINDER LOOP
# =============================================================================

class AdvancedBugFinderLoop:
    """
    Main Bug Finder Loop implementation
    
    This class orchestrates the entire ML-based bug detection pipeline,
    from data collection to alert generation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.ensemble = MLEnsembleDetector(self.config.get('ensemble', {}))
        self.feature_engineer = FeatureEngineer(
            window_size=self.config.get('feature_window_seconds', 300)
        )
        self.confidence_scorer = ConfidenceScorer()
        self.fp_reducer = FalsePositiveReducer(self.config.get('fp_reduction', {}))
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        self.explanation_generator = None
        
        # State
        self.is_running = False
        self.cycle_count = 0
        self.performance_metrics = {
            'predictions': 0,
            'alerts': 0,
            'latency_ms': []
        }
        
    async def initialize(self, 
                        model_path: Optional[str] = None,
                        training_data: Optional[Tuple] = None):
        """
        Initialize the bug finder loop
        
        Args:
            model_path: Path to load pre-trained models
            training_data: Optional (X, y) tuple for initial training
        """
        logger.info("Initializing Advanced Bug Finder Loop...")
        
        if model_path and Path(model_path).exists():
            # Load pre-trained models
            self.ensemble.load_models(model_path)
            logger.info(f"Loaded models from {model_path}")
        elif training_data:
            # Train new models
            X, y = training_data
            metrics = self.ensemble.fit(X, y)
            logger.info(f"Trained new models: {metrics}")
        else:
            logger.warning("No models loaded or trained - predictions will fail")
        
        # Initialize explanation generator
        if training_data:
            self.explanation_generator = ExplanationGenerator()
            self.explanation_generator.initialize(
                training_data[0],
                self.ensemble.models.get('xgboost')
            )
        
        logger.info("Bug Finder Loop initialized")
        
    async def run(self, 
                 data_source: callable,
                 interval_seconds: float = 30.0):
        """
        Run the main detection loop
        
        Args:
            data_source: Callable that returns current system data
            interval_seconds: Seconds between detection cycles
        """
        self.is_running = True
        logger.info(f"Starting Bug Finder Loop (interval: {interval_seconds}s)")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Execute detection cycle
                await self._detection_cycle(data_source)
                
                # Calculate sleep time
                elapsed = time.time() - cycle_start
                sleep_time = max(0, interval_seconds - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except (OSError, RuntimeError, PermissionError) as e:
                logger.error(f"Error in detection cycle: {e}")
                await asyncio.sleep(5)
                
    async def _detection_cycle(self, data_source: callable):
        """Execute a single detection cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        # 1. Collect data
        try:
            raw_data = await data_source()
        except (OSError, RuntimeError, PermissionError) as e:
            logger.error(f"Data collection error: {e}")
            return
        
        # 2. Extract features
        features = self.feature_engineer.extract_features(
            raw_data.get('system_metrics', {}),
            raw_data.get('log_data', {}),
            raw_data.get('agent_metrics', {})
        )
        
        feature_array = features.to_array().reshape(1, -1)
        
        # 3. Generate prediction
        prediction = self.ensemble.predict(feature_array)
        
        # 4. Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence(prediction)
        prediction.confidence = confidence['overall_confidence']
        
        # 5. Filter false positives
        filtered = self.fp_reducer.filter_predictions([prediction])
        
        # 6. Generate alerts
        for pred in filtered:
            if pred.confidence >= self.config.get('alert_threshold', 0.7):
                # Generate explanation
                explanation = ""
                if self.explanation_generator:
                    explanation = self.explanation_generator.generate_explanation(
                        pred, features, feature_array
                    )
                
                # Create alert
                alert = BugAlert(
                    prediction=Prediction(
                        id=f"pred_{self.cycle_count}_{int(time.time())}",
                        prediction_type='anomaly',
                        predicted_time=datetime.now(),
                        confidence=pred.confidence,
                        severity=self._classify_severity(pred.confidence),
                        contributing_factors=list(features.to_dict().keys())
                    ),
                    explanation=explanation
                )
                
                # Send alert
                await self.alert_manager.send_alert(alert)
                self.performance_metrics['alerts'] += 1
        
        # Update metrics
        latency_ms = (time.time() - cycle_start) * 1000
        self.performance_metrics['latency_ms'].append(latency_ms)
        self.performance_metrics['predictions'] += 1
        
        # Log periodic status
        if self.cycle_count % 100 == 0:
            self._log_status()
    
    def _classify_severity(self, confidence: float) -> AlertSeverity:
        """Classify alert severity from confidence"""
        if confidence >= 0.9:
            return AlertSeverity.CRITICAL
        elif confidence >= 0.75:
            return AlertSeverity.HIGH
        elif confidence >= 0.6:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _log_status(self):
        """Log current status"""
        latencies = self.performance_metrics['latency_ms'][-100:]
        avg_latency = np.mean(latencies) if latencies else 0
        
        logger.info(
            f"Bug Finder Status - Cycles: {self.cycle_count}, "
            f"Predictions: {self.performance_metrics['predictions']}, "
            f"Alerts: {self.performance_metrics['alerts']}, "
            f"Avg Latency: {avg_latency:.1f}ms"
        )
    
    def stop(self):
        """Stop the bug finder loop"""
        self.is_running = False
        logger.info("Bug Finder Loop stopped")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        latencies = self.performance_metrics['latency_ms']
        
        return {
            'total_cycles': self.cycle_count,
            'total_predictions': self.performance_metrics['predictions'],
            'total_alerts': self.performance_metrics['alerts'],
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0
        }
    
    def save_models(self, path: str):
        """Save models to disk"""
        self.ensemble.save_models(path)
        
    def load_models(self, path: str):
        """Load models from disk"""
        self.ensemble.load_models(path)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_training_data(n_samples: int = 1000,
                                n_features: int = 50,
                                anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample training data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        anomaly_ratio: Ratio of anomaly samples
        
    Returns:
        X, y arrays
    """
    np.random.seed(42)
    
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_ratio))
    X_normal = np.random.randn(n_normal, n_features)
    
    # Generate anomaly data
    n_anomaly = n_samples - n_normal
    X_anomaly = np.random.randn(n_anomaly, n_features) * 2 + 5
    
    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the Advanced Bug Finder Loop"""
    
    # Create configuration
    config = {
        'ensemble': {
            'contamination': 0.05,
            'sequence_length': 50,
            'lstm_epochs': 20
        },
        'fp_reduction': {
            'min_confidence': 0.7,
            'min_model_agreement': 0.6
        },
        'alerts': {
            'cooldown_seconds': 300
        },
        'alert_threshold': 0.75
    }
    
    # Initialize bug finder
    bug_finder = AdvancedBugFinderLoop(config)
    
    # Create sample training data
    X, y = create_sample_training_data(n_samples=5000)
    
    # Initialize with training
    await bug_finder.initialize(training_data=(X, y))
    
    # Register alert handler
    async def alert_handler(alert: BugAlert):
        print(f"\n{'='*60}")
        print(f"ALERT: {alert.prediction.severity.value.upper()}")
        print(f"{'='*60}")
        print(alert.explanation)
        print(f"{'='*60}\n")
    
    bug_finder.alert_manager.register_handler(alert_handler)
    
    # Simulate data source
    async def data_source():
        return {
            'system_metrics': {
                'cpu_percent': list(np.random.randn(10) * 10 + 50),
                'memory_percent': list(np.random.randn(10) * 5 + 60),
                'disk_io': list(np.random.randn(10) * 100 + 500)
            },
            'log_data': {
                'total_logs': 100,
                'error_count': np.random.randint(0, 5),
                'warning_count': np.random.randint(0, 10),
                'unique_templates': 20
            },
            'agent_metrics': {
                'loop_durations': list(np.random.randn(5) * 2 + 5),
                'api_errors': np.random.randint(0, 3),
                'api_calls': 50,
                'queue_depth': np.random.randint(0, 20),
                'tasks_successful': 45,
                'tasks_total': 50
            }
        }
    
    # Run for a few cycles
    print("Running Bug Finder Loop...")
    
    # Run manually instead of using run() for demonstration
    for i in range(5):
        await bug_finder._detection_cycle(data_source)
        await asyncio.sleep(1)
    
    # Print metrics
    metrics = bug_finder.get_metrics()
    print(f"\nFinal Metrics: {metrics}")
    
    # Save models
    bug_finder.save_models("models/bug_finder")
    
    print("\nExample complete!")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
