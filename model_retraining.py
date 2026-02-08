"""
Model Retraining Pipeline for Advanced Bug Finder Loop

This module implements continuous learning, concept drift detection,
and automated model retraining for the ML-based bug detection system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import json
from pathlib import Path

# ML libraries
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class RetrainingStrategy(Enum):
    INCREMENTAL = "incremental"
    FULL = "full"
    TRANSFER = "transfer"
    NONE = "none"


@dataclass
class DriftResult:
    """Result of concept drift detection"""
    drift_detected: bool
    drift_ratio: float
    ks_results: List[Dict]
    psi_scores: List[Dict]
    recommendation: str
    affected_features: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    false_positive_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_degraded(self, baseline: 'ModelPerformance', threshold: float = 0.05) -> bool:
        """Check if performance has degraded compared to baseline"""
        return (
            self.f1_score < baseline.f1_score - threshold or
            self.precision < baseline.precision - threshold or
            self.recall < baseline.recall - threshold
        )


@dataclass
class RetrainingResult:
    """Result of model retraining"""
    triggered: bool
    strategy: str
    models: Optional[Dict] = None
    performance: Optional[ModelPerformance] = None
    samples_used: int = 0
    training_time_seconds: float = 0.0
    message: str = ""


@dataclass
class TriggerCondition:
    """Base class for retraining trigger conditions"""
    name: str
    is_triggered: bool = False
    details: Dict = field(default_factory=dict)


# =============================================================================
# CONCEPT DRIFT DETECTION
# =============================================================================

class ConceptDriftDetector:
    """
    Detect concept drift in data distribution
    
    Uses multiple statistical tests to detect when the data distribution
    has changed significantly from the reference distribution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ks_threshold = self.config.get('ks_pvalue_threshold', 0.01)
        self.psi_threshold = self.config.get('psi_threshold', 0.25)
        self.drift_ratio_threshold = self.config.get('drift_ratio_threshold', 0.1)
        
        self.reference_distribution = None
        self.reference_stats = None
        
    def set_reference(self, reference_data: np.ndarray, feature_names: List[str]):
        """
        Set reference distribution for drift detection
        
        Args:
            reference_data: Reference data matrix
            feature_names: Names of features
        """
        self.reference_distribution = reference_data
        self.feature_names = feature_names
        
        # Calculate reference statistics
        self.reference_stats = {
            'mean': np.mean(reference_data, axis=0),
            'std': np.std(reference_data, axis=0),
            'percentiles': np.percentile(reference_data, [5, 25, 50, 75, 95], axis=0)
        }
        
        logger.info(f"Reference distribution set with {len(reference_data)} samples")
        
    def detect_drift(self, current_data: np.ndarray) -> DriftResult:
        """
        Detect concept drift between reference and current data
        
        Args:
            current_data: Current data matrix
            
        Returns:
            DriftResult with detection details
        """
        if self.reference_distribution is None:
            raise ValueError("Reference distribution not set")
        
        if len(current_data) < 10:
            return DriftResult(
                drift_detected=False,
                drift_ratio=0.0,
                ks_results=[],
                psi_scores=[],
                recommendation='none',
                affected_features=[],
                severity='low'
            )
        
        n_features = min(self.reference_distribution.shape[1], current_data.shape[1])
        
        # Kolmogorov-Smirnov test for each feature
        ks_results = []
        drifted_features = []
        
        for i in range(n_features):
            ref_values = self.reference_distribution[:, i]
            curr_values = current_data[:, i]
            
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
            
            result = {
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': p_value < self.ks_threshold
            }
            
            ks_results.append(result)
            
            if result['drift_detected']:
                drifted_features.append(result['feature'])
        
        # Population Stability Index (PSI)
        psi_scores = []
        
        for i in range(n_features):
            ref_values = self.reference_distribution[:, i]
            curr_values = current_data[:, i]
            
            psi = self._calculate_psi(ref_values, curr_values)
            
            result = {
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'psi': float(psi),
                'drift_detected': psi > self.psi_threshold
            }
            
            psi_scores.append(result)
        
        # Calculate overall drift ratio
        drift_ratio = len(drifted_features) / n_features
        
        # Determine severity
        severity = self._determine_severity(drift_ratio, drifted_features)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            drift_ratio, drifted_features, ks_results, psi_scores
        )
        
        return DriftResult(
            drift_detected=drift_ratio > self.drift_ratio_threshold,
            drift_ratio=drift_ratio,
            ks_results=ks_results,
            psi_scores=psi_scores,
            recommendation=recommendation,
            affected_features=drifted_features,
            severity=severity
        )
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index
        
        Args:
            expected: Expected/reference distribution
            actual: Actual/current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges[-1] += 1e-10  # Ensure last value is included
        
        # Calculate distributions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        # Convert to percentages
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Add small constant to avoid division by zero
        expected_percents = np.maximum(expected_percents, 0.0001)
        actual_percents = np.maximum(actual_percents, 0.0001)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return psi
    
    def _determine_severity(self, drift_ratio: float, 
                           affected_features: List[str]) -> str:
        """Determine drift severity"""
        if drift_ratio > 0.5:
            return 'critical'
        elif drift_ratio > 0.3:
            return 'high'
        elif drift_ratio > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendation(self, drift_ratio: float,
                                 affected_features: List[str],
                                 ks_results: List[Dict],
                                 psi_scores: List[Dict]) -> str:
        """Generate retraining recommendation based on drift"""
        if drift_ratio > 0.5:
            return 'full_retrain'
        elif drift_ratio > 0.3:
            return 'transfer_learning'
        elif drift_ratio > 0.1:
            return 'incremental_update'
        else:
            return 'none'


# =============================================================================
# RETRAINING TRIGGERS
# =============================================================================

class DataVolumeTrigger:
    """Trigger retraining when enough new data is collected"""
    
    def __init__(self, min_new_samples: int = 1000):
        self.min_new_samples = min_new_samples
        self.new_samples_count = 0
        
    def check(self, new_data: np.ndarray) -> TriggerCondition:
        """Check if enough new samples have been collected"""
        self.new_samples_count += len(new_data)
        
        is_triggered = self.new_samples_count >= self.min_new_samples
        
        return TriggerCondition(
            name='data_volume',
            is_triggered=is_triggered,
            details={
                'new_samples': self.new_samples_count,
                'threshold': self.min_new_samples
            }
        )
    
    def reset(self):
        """Reset the counter"""
        self.new_samples_count = 0


class PerformanceDegradationTrigger:
    """Trigger retraining when model performance degrades"""
    
    def __init__(self, min_f1_drop: float = 0.05, window_size: int = 100):
        self.min_f1_drop = min_f1_drop
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
        
    def check(self, current_performance: ModelPerformance) -> TriggerCondition:
        """Check if performance has degraded"""
        self.performance_history.append(current_performance)
        
        # Set baseline if not set
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return TriggerCondition(name='performance', is_triggered=False)
        
        # Check for degradation
        is_degraded = current_performance.is_degraded(
            self.baseline_performance, self.min_f1_drop
        )
        
        return TriggerCondition(
            name='performance',
            is_triggered=is_degraded,
            details={
                'current_f1': current_performance.f1_score,
                'baseline_f1': self.baseline_performance.f1_score,
                'drop': self.baseline_performance.f1_score - current_performance.f1_score
            }
        )
    
    def update_baseline(self, performance: ModelPerformance):
        """Update baseline performance"""
        self.baseline_performance = performance


class ConceptDriftTrigger:
    """Trigger retraining when concept drift is detected"""
    
    def __init__(self, drift_detector: ConceptDriftDetector,
                 drift_threshold: float = 0.1):
        self.drift_detector = drift_detector
        self.drift_threshold = drift_threshold
        
    def check(self, current_data: np.ndarray) -> TriggerCondition:
        """Check for concept drift"""
        drift_result = self.drift_detector.detect_drift(current_data)
        
        return TriggerCondition(
            name='concept_drift',
            is_triggered=drift_result.drift_detected,
            details={
                'drift_ratio': drift_result.drift_ratio,
                'severity': drift_result.severity,
                'affected_features': drift_result.affected_features,
                'recommendation': drift_result.recommendation
            }
        )


class ScheduledTrigger:
    """Trigger retraining on a schedule"""
    
    def __init__(self, interval_days: int = 7):
        self.interval_days = interval_days
        self.last_retraining = None
        
    def check(self) -> TriggerCondition:
        """Check if scheduled retraining is due"""
        if self.last_retraining is None:
            return TriggerCondition(
                name='scheduled',
                is_triggered=True,
                details={'reason': 'first_run'}
            )
        
        days_since = (datetime.now() - self.last_retraining).days
        is_triggered = days_since >= self.interval_days
        
        return TriggerCondition(
            name='scheduled',
            is_triggered=is_triggered,
            details={
                'days_since': days_since,
                'interval': self.interval_days
            }
        )
    
    def mark_retrained(self):
        """Mark that retraining has occurred"""
        self.last_retraining = datetime.now()


# =============================================================================
# RETRAINING STRATEGIES
# =============================================================================

class IncrementalRetrainingStrategy:
    """
    Incremental/online learning strategy
    
    Updates models with new data without forgetting previous knowledge.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.01)
        
    async def retrain(self,
                     models: Dict,
                     new_data: np.ndarray,
                     new_labels: Optional[np.ndarray] = None,
                     validation_data: Optional[Tuple] = None) -> Tuple[Dict, ModelPerformance]:
        """
        Perform incremental retraining
        
        Args:
            models: Current models
            new_data: New training data
            new_labels: Optional labels
            validation_data: Optional validation data
            
        Returns:
            Updated models and performance metrics
        """
        import time
        start_time = time.time()
        
        updated_models = {}
        
        # Isolation Forest - retrain with combined data
        if 'isolation_forest' in models:
            logger.info("Incremental update: Isolation Forest")
            # Combine old and new data (would need storage of old data)
            updated_models['isolation_forest'] = models['isolation_forest']
        
        # LSTM Autoencoder - continue training
        if 'lstm_autoencoder' in models and models['lstm_autoencoder'] is not None:
            logger.info("Incremental update: LSTM Autoencoder")
            
            lstm_model = models['lstm_autoencoder']
            
            # Create sequences
            seq_length = 50  # Should match training config
            sequences = []
            for i in range(len(new_data) - seq_length + 1):
                sequences.append(new_data[i:i+seq_length])
            
            if len(sequences) > 0:
                X_seq = np.array(sequences)
                
                # Continue training
                lstm_model.fit(
                    X_seq, X_seq,
                    epochs=5,
                    batch_size=32,
                    verbose=0
                )
            
            updated_models['lstm_autoencoder'] = lstm_model
        
        # XGBoost - update with new data
        if 'xgboost' in models and new_labels is not None:
            logger.info("Incremental update: XGBoost")
            
            xgb_model = models['xgboost']
            
            # Update with new data
            xgb_model.fit(
                new_data, new_labels,
                xgb_model=xgb_model.get_booster()
            )
            
            updated_models['xgboost'] = xgb_model
        
        # Calculate performance
        performance = None
        if validation_data:
            performance = self._evaluate_models(updated_models, validation_data)
        
        training_time = time.time() - start_time
        
        return updated_models, performance, training_time
    
    def _evaluate_models(self, models: Dict, 
                        validation_data: Tuple) -> ModelPerformance:
        """Evaluate model performance"""
        X_val, y_val = validation_data
        
        # Use XGBoost for evaluation if available
        if 'xgboost' in models:
            y_pred = models['xgboost'].predict(X_val)
            y_proba = models['xgboost'].predict_proba(X_val)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary'
            )
            
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            
            # Calculate FPR
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            return ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                false_positive_rate=fpr
            )
        
        return ModelPerformance(0, 0, 0, 0, 0, 0)


class FullRetrainingStrategy:
    """
    Full retraining strategy
    
    Completely retrain models from scratch with all available data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.test_size = self.config.get('test_size', 0.2)
        
    async def retrain(self,
                     historical_data: np.ndarray,
                     historical_labels: Optional[np.ndarray] = None,
                     new_data: Optional[np.ndarray] = None,
                     new_labels: Optional[np.ndarray] = None) -> Tuple[Dict, ModelPerformance]:
        """
        Perform full retraining
        
        Args:
            historical_data: Historical training data
            historical_labels: Historical labels
            new_data: New data to include
            new_labels: New labels
            
        Returns:
            New models and performance metrics
        """
        import time
        start_time = time.time()
        
        # Combine all data
        if new_data is not None:
            all_data = np.vstack([historical_data, new_data])
            if historical_labels is not None and new_labels is not None:
                all_labels = np.hstack([historical_labels, new_labels])
            else:
                all_labels = None
        else:
            all_data = historical_data
            all_labels = historical_labels
        
        # Split train/test
        if all_labels is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                all_data, all_labels,
                test_size=self.test_size,
                random_state=42,
                stratify=all_labels
            )
        else:
            X_train = all_data
            X_test = all_data[:100]  # Use subset for validation
            y_train = None
            y_test = None
        
        # Import and create new ensemble
        from bug_finder_loop import MLEnsembleDetector
        
        new_ensemble = MLEnsembleDetector(self.config.get('ensemble', {}))
        
        # Train
        logger.info(f"Full retraining with {len(X_train)} samples")
        new_ensemble.fit(X_train, y_train)
        
        # Evaluate
        performance = None
        if y_test is not None:
            y_pred = new_ensemble.models['xgboost'].predict(X_test)
            y_proba = new_ensemble.models['xgboost'].predict_proba(X_test)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                false_positive_rate=fpr
            )
        
        training_time = time.time() - start_time
        
        return new_ensemble.models, performance, training_time


class TransferLearningStrategy:
    """
    Transfer learning strategy
    
    Uses pre-trained models and fine-tunes on new data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fine_tune_epochs = self.config.get('fine_tune_epochs', 10)
        
    async def retrain(self,
                     models: Dict,
                     new_data: np.ndarray,
                     new_labels: Optional[np.ndarray] = None) -> Tuple[Dict, ModelPerformance]:
        """
        Perform transfer learning retraining
        
        Args:
            models: Pre-trained models
            new_data: New data for fine-tuning
            new_labels: Optional labels
            
        Returns:
            Fine-tuned models and performance
        """
        import time
        start_time = time.time()
        
        updated_models = dict(models)  # Copy
        
        # Fine-tune LSTM with lower learning rate
        if 'lstm_autoencoder' in models and models['lstm_autoencoder'] is not None:
            logger.info("Transfer learning: Fine-tuning LSTM")
            
            lstm_model = models['lstm_autoencoder']
            
            # Compile with lower learning rate
            import tensorflow as tf
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            lstm_model.compile(optimizer=optimizer, loss='mse')
            
            # Create sequences
            seq_length = 50
            sequences = []
            for i in range(len(new_data) - seq_length + 1):
                sequences.append(new_data[i:i+seq_length])
            
            if len(sequences) > 0:
                X_seq = np.array(sequences)
                
                # Fine-tune
                lstm_model.fit(
                    X_seq, X_seq,
                    epochs=self.fine_tune_epochs,
                    batch_size=32,
                    verbose=0
                )
            
            updated_models['lstm_autoencoder'] = lstm_model
        
        # Fine-tune XGBoost
        if 'xgboost' in models and new_labels is not None:
            logger.info("Transfer learning: Fine-tuning XGBoost")
            
            xgb_model = models['xgboost']
            
            # Fine-tune with lower learning rate
            xgb_model.set_params(learning_rate=0.01)
            xgb_model.fit(
                new_data, new_labels,
                xgb_model=xgb_model.get_booster()
            )
            
            updated_models['xgboost'] = xgb_model
        
        training_time = time.time() - start_time
        
        return updated_models, None, training_time


# =============================================================================
# MAIN RETRAINING PIPELINE
# =============================================================================

class ModelRetrainingPipeline:
    """
    Main pipeline for automated model retraining
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize drift detector
        self.drift_detector = ConceptDriftDetector(
            self.config.get('drift_detection', {})
        )
        
        # Initialize triggers
        self.triggers = {
            'data_volume': DataVolumeTrigger(
                self.config.get('min_new_samples', 1000)
            ),
            'performance': PerformanceDegradationTrigger(
                self.config.get('min_f1_drop', 0.05)
            ),
            'concept_drift': ConceptDriftTrigger(
                self.drift_detector,
                self.config.get('drift_threshold', 0.1)
            ),
            'scheduled': ScheduledTrigger(
                self.config.get('retraining_interval_days', 7)
            )
        }
        
        # Initialize strategies
        self.strategies = {
            'incremental': IncrementalRetrainingStrategy(
                self.config.get('incremental', {})
            ),
            'full': FullRetrainingStrategy(
                self.config.get('full', {})
            ),
            'transfer': TransferLearningStrategy(
                self.config.get('transfer', {})
            )
        }
        
        # State
        self.is_initialized = False
        self.historical_data = None
        self.historical_labels = None
        
    def initialize(self,
                  reference_data: np.ndarray,
                  reference_labels: Optional[np.ndarray] = None,
                  feature_names: Optional[List[str]] = None):
        """
        Initialize the retraining pipeline
        
        Args:
            reference_data: Initial reference data
            reference_labels: Optional reference labels
            feature_names: Names of features
        """
        self.historical_data = reference_data
        self.historical_labels = reference_labels
        
        # Set drift detector reference
        self.drift_detector.set_reference(
            reference_data,
            feature_names or [f'feature_{i}' for i in range(reference_data.shape[1])]
        )
        
        self.is_initialized = True
        
        logger.info(f"Retraining pipeline initialized with {len(reference_data)} samples")
        
    async def check_and_retrain(self,
                               current_models: Dict,
                               new_data: np.ndarray,
                               new_labels: Optional[np.ndarray] = None,
                               current_performance: Optional[ModelPerformance] = None) -> RetrainingResult:
        """
        Check if retraining is needed and execute if triggered
        
        Args:
            current_models: Current model collection
            new_data: New data since last check
            new_labels: Optional new labels
            current_performance: Current model performance
            
        Returns:
            RetrainingResult
        """
        if not self.is_initialized:
            return RetrainingResult(
                triggered=False,
                strategy='none',
                message='Pipeline not initialized'
            )
        
        # Check all triggers
        triggered_conditions = []
        
        # Data volume trigger
        data_trigger = self.triggers['data_volume'].check(new_data)
        if data_trigger.is_triggered:
            triggered_conditions.append(data_trigger)
        
        # Performance trigger
        if current_performance:
            perf_trigger = self.triggers['performance'].check(current_performance)
            if perf_trigger.is_triggered:
                triggered_conditions.append(perf_trigger)
        
        # Concept drift trigger
        drift_trigger = self.triggers['concept_drift'].check(new_data)
        if drift_trigger.is_triggered:
            triggered_conditions.append(drift_trigger)
        
        # Scheduled trigger
        sched_trigger = self.triggers['scheduled'].check()
        if sched_trigger.is_triggered:
            triggered_conditions.append(sched_trigger)
        
        # If no triggers, return
        if not triggered_conditions:
            return RetrainingResult(
                triggered=False,
                strategy='none',
                message='No retraining triggers activated'
            )
        
        # Determine strategy
        strategy = self._select_strategy(triggered_conditions)
        
        logger.info(f"Retraining triggered. Strategy: {strategy}")
        logger.info(f"Triggers: {[t.name for t in triggered_conditions]}")
        
        # Execute retraining
        try:
            if strategy == 'incremental':
                updated_models, performance, training_time = await self.strategies['incremental'].retrain(
                    current_models, new_data, new_labels
                )
                
            elif strategy == 'full':
                updated_models, performance, training_time = await self.strategies['full'].retrain(
                    self.historical_data,
                    self.historical_labels,
                    new_data,
                    new_labels
                )
                
            elif strategy == 'transfer':
                updated_models, performance, training_time = await self.strategies['transfer'].retrain(
                    current_models, new_data, new_labels
                )
            else:
                return RetrainingResult(
                    triggered=False,
                    strategy='none',
                    message='No retraining strategy selected'
                )
            
            # Update historical data
            self._update_historical_data(new_data, new_labels)
            
            # Reset data volume trigger
            self.triggers['data_volume'].reset()
            
            # Mark scheduled trigger
            self.triggers['scheduled'].mark_retrained()
            
            return RetrainingResult(
                triggered=True,
                strategy=strategy,
                models=updated_models,
                performance=performance,
                samples_used=len(new_data),
                training_time_seconds=training_time,
                message=f'Successfully retrained using {strategy} strategy'
            )
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return RetrainingResult(
                triggered=True,
                strategy=strategy,
                message=f'Retraining failed: {str(e)}'
            )
    
    def _select_strategy(self, triggered_conditions: List[TriggerCondition]) -> str:
        """Select retraining strategy based on triggers"""
        
        # Check for concept drift
        drift_condition = next(
            (c for c in triggered_conditions if c.name == 'concept_drift'),
            None
        )
        
        if drift_condition:
            recommendation = drift_condition.details.get('recommendation', 'incremental_update')
            
            if recommendation == 'full_retrain':
                return 'full'
            elif recommendation == 'transfer_learning':
                return 'transfer'
        
        # Check for performance degradation
        perf_condition = next(
            (c for c in triggered_conditions if c.name == 'performance'),
            None
        )
        
        if perf_condition:
            drop = perf_condition.details.get('drop', 0)
            if drop > 0.1:
                return 'full'
            else:
                return 'transfer'
        
        # Default to incremental
        return 'incremental'
    
    def _update_historical_data(self, new_data: np.ndarray, 
                               new_labels: Optional[np.ndarray]):
        """Update stored historical data"""
        if self.historical_data is not None:
            self.historical_data = np.vstack([self.historical_data, new_data])
            if self.historical_labels is not None and new_labels is not None:
                self.historical_labels = np.hstack([self.historical_labels, new_labels])
        else:
            self.historical_data = new_data
            self.historical_labels = new_labels
        
        # Limit storage size
        max_samples = self.config.get('max_historical_samples', 100000)
        if len(self.historical_data) > max_samples:
            self.historical_data = self.historical_data[-max_samples:]
            if self.historical_labels is not None:
                self.historical_labels = self.historical_labels[-max_samples:]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_retraining_pipeline():
    """Example usage of the retraining pipeline"""
    print("=" * 60)
    print("Model Retraining Pipeline Example")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    
    # Reference data (normal distribution)
    reference_data = np.random.randn(5000, 20)
    reference_labels = np.random.random(5000) < 0.05
    
    # New data with slight drift
    new_data = np.random.randn(1500, 20) * 1.2 + 0.5
    new_labels = np.random.random(1500) < 0.08
    
    print(f"\nReference data: {reference_data.shape}")
    print(f"New data: {new_data.shape}")
    
    # Initialize pipeline
    config = {
        'min_new_samples': 1000,
        'drift_threshold': 0.1,
        'retraining_interval_days': 7
    }
    
    pipeline = ModelRetrainingPipeline(config)
    pipeline.initialize(
        reference_data,
        reference_labels,
        feature_names=[f'feature_{i}' for i in range(20)]
    )
    
    # Check for drift
    print("\nChecking for concept drift...")
    drift_result = pipeline.drift_detector.detect_drift(new_data)
    
    print(f"Drift detected: {drift_result.drift_detected}")
    print(f"Drift ratio: {drift_result.drift_ratio:.3f}")
    print(f"Severity: {drift_result.severity}")
    print(f"Recommendation: {drift_result.recommendation}")
    
    if drift_result.affected_features:
        print(f"Affected features: {drift_result.affected_features[:5]}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    example_retraining_pipeline()
