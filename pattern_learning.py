"""
Pattern Learning System for Advanced Bug Finder Loop

This module implements historical pattern learning, sequential pattern mining,
and pattern matching for predictive bug detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from scipy import stats
from scipy.spatial.distance import cosine
import logging

# ML libraries for clustering
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sequence mining
from prefixspan import PrefixSpan

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TemporalPattern:
    """Pattern related to temporal characteristics"""
    hot_hours: List[int]
    hot_days: List[int]
    hour_distribution: Dict[int, float]
    day_distribution: Dict[int, float]
    hourly_anomaly_rate: Dict[int, float]
    
    def get_risk_score(self, timestamp: datetime) -> float:
        """Get anomaly risk score for a given timestamp"""
        hour_risk = self.hour_distribution.get(timestamp.hour, 0)
        day_risk = self.day_distribution.get(timestamp.weekday(), 0)
        return (hour_risk + day_risk) / 2


@dataclass
class SequentialPattern:
    """Sequential event pattern that precedes anomalies"""
    sequence: List[str]
    support: float
    confidence: float
    avg_time_to_anomaly: float  # Average seconds from pattern to anomaly
    occurrence_count: int
    
    def match_score(self, recent_events: List[str]) -> float:
        """Calculate how well recent events match this pattern"""
        if not recent_events or not self.sequence:
            return 0.0
        
        # Find longest common subsequence
        lcs_length = self._lcs_length(recent_events, self.sequence)
        return lcs_length / max(len(recent_events), len(self.sequence))
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


@dataclass
class CorrelationPattern:
    """Correlation between features and anomalies"""
    feature_name: str
    correlation_strength: float
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool
    normal_mean: float
    anomaly_mean: float
    
    def get_predictive_power(self) -> float:
        """Get predictive power score (0-1)"""
        if not self.is_significant:
            return 0.0
        return min(1.0, abs(self.effect_size) / 2)


@dataclass
class ClusterPattern:
    """Cluster of similar anomalies"""
    cluster_id: int
    centroid: np.ndarray
    feature_names: List[str]
    size: int
    avg_anomaly_score: float
    common_characteristics: Dict[str, Any]
    
    def distance_to(self, features: np.ndarray) -> float:
        """Calculate distance from features to cluster centroid"""
        return np.linalg.norm(features - self.centroid)


@dataclass
class PatternLibrary:
    """Complete library of learned patterns"""
    temporal: Optional[TemporalPattern] = None
    sequential: List[SequentialPattern] = field(default_factory=list)
    correlations: List[CorrelationPattern] = field(default_factory=list)
    clusters: List[ClusterPattern] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_top_predictors(self, n: int = 10) -> List[CorrelationPattern]:
        """Get top n predictive features"""
        sorted_corr = sorted(
            self.correlations,
            key=lambda x: x.get_predictive_power(),
            reverse=True
        )
        return sorted_corr[:n]


@dataclass
class PatternMatchResult:
    """Result of pattern matching"""
    temporal_match: Optional[Dict] = None
    sequential_matches: List[Dict] = field(default_factory=list)
    correlation_matches: List[Dict] = field(default_factory=list)
    cluster_matches: List[Dict] = field(default_factory=list)
    overall_score: float = 0.0
    is_predicted_anomaly: bool = False
    contributing_patterns: List[str] = field(default_factory=list)


# =============================================================================
# HISTORICAL PATTERN LEARNER
# =============================================================================

class HistoricalPatternLearner:
    """
    Learn patterns from historical anomaly and normal behavior data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pattern_library = PatternLibrary()
        self.min_pattern_support = self.config.get('min_pattern_support', 0.05)
        self.min_confidence = self.config.get('min_pattern_confidence', 0.6)
        self.scaler = StandardScaler()
        
    def learn_patterns(self, 
                      historical_data: pd.DataFrame,
                      event_sequences: Optional[List[List[str]]] = None) -> PatternLibrary:
        """
        Learn all patterns from historical data
        
        Args:
            historical_data: DataFrame with features and 'is_anomaly' column
            event_sequences: List of event sequences (for sequential patterns)
            
        Returns:
            PatternLibrary with all learned patterns
        """
        logger.info(f"Learning patterns from {len(historical_data)} records")
        
        # Temporal patterns
        logger.info("Learning temporal patterns...")
        self.pattern_library.temporal = self._learn_temporal_patterns(historical_data)
        
        # Sequential patterns
        if event_sequences:
            logger.info("Learning sequential patterns...")
            self.pattern_library.sequential = self._learn_sequential_patterns(
                event_sequences,
                historical_data
            )
        
        # Correlation patterns
        logger.info("Learning correlation patterns...")
        self.pattern_library.correlations = self._learn_correlation_patterns(
            historical_data
        )
        
        # Cluster patterns
        logger.info("Learning cluster patterns...")
        self.pattern_library.clusters = self._learn_cluster_patterns(
            historical_data
        )
        
        self.pattern_library.last_updated = datetime.now()
        
        logger.info("Pattern learning complete")
        return self.pattern_library
    
    def _learn_temporal_patterns(self, data: pd.DataFrame) -> TemporalPattern:
        """Learn temporal patterns from historical data"""
        # Ensure timestamp column exists
        if 'timestamp' not in data.columns:
            logger.warning("No timestamp column found for temporal patterns")
            return TemporalPattern(
                hot_hours=[],
                hot_days=[],
                hour_distribution={},
                day_distribution={},
                hourly_anomaly_rate={}
            )
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract hour and day
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Get anomaly records
        anomaly_data = data[data.get('is_anomaly', pd.Series([False] * len(data)))]
        
        # Hour distribution
        hour_counts = anomaly_data['hour'].value_counts().sort_index()
        hour_dist = (hour_counts / hour_counts.sum()).to_dict()
        
        # Day distribution
        day_counts = anomaly_data['day_of_week'].value_counts().sort_index()
        day_dist = (day_counts / day_counts.sum()).to_dict()
        
        # Identify hot periods (above 75th percentile)
        hour_threshold = np.percentile(list(hour_dist.values()), 75)
        hot_hours = [h for h, v in hour_dist.items() if v >= hour_threshold]
        
        day_threshold = np.percentile(list(day_dist.values()), 75)
        hot_days = [d for d, v in day_dist.items() if v >= day_threshold]
        
        # Hourly anomaly rate
        hourly_anomaly_rate = {}
        for hour in range(24):
            hour_data = data[data['hour'] == hour]
            if len(hour_data) > 0:
                anomaly_rate = hour_data.get('is_anomaly', pd.Series([0])).mean()
                hourly_anomaly_rate[hour] = float(anomaly_rate)
        
        return TemporalPattern(
            hot_hours=hot_hours,
            hot_days=hot_days,
            hour_distribution=hour_dist,
            day_distribution=day_dist,
            hourly_anomaly_rate=hourly_anomaly_rate
        )
    
    def _learn_sequential_patterns(self,
                                   event_sequences: List[List[str]],
                                   data: pd.DataFrame) -> List[SequentialPattern]:
        """Learn sequential patterns that precede anomalies"""
        patterns = []
        
        # Use PrefixSpan for frequent sequence mining
        ps = PrefixSpan(event_sequences)
        
        # Mine frequent sequences
        min_support = int(len(event_sequences) * self.min_pattern_support)
        frequent_sequences = ps.frequent(min_support)
        
        # Convert to SequentialPattern objects
        for support, sequence in frequent_sequences:
            if len(sequence) < 2:  # Skip single events
                continue
            
            confidence = support / len(event_sequences)
            
            if confidence >= self.min_confidence:
                # Calculate average time to anomaly
                # This would need timing information in practice
                avg_ttf = 300  # Placeholder: 5 minutes
                
                patterns.append(SequentialPattern(
                    sequence=list(sequence),
                    support=support / len(event_sequences),
                    confidence=confidence,
                    avg_time_to_anomaly=avg_ttf,
                    occurrence_count=support
                ))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns[:50]  # Keep top 50 patterns
    
    def _learn_correlation_patterns(self, data: pd.DataFrame) -> List[CorrelationPattern]:
        """Learn correlation patterns between features and anomalies"""
        patterns = []
        
        if 'is_anomaly' not in data.columns:
            logger.warning("No is_anomaly column found for correlation patterns")
            return patterns
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['timestamp', 'is_anomaly', 'hour', 'day_of_week']
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Separate normal and anomaly data
        normal_data = data[~data['is_anomaly']]
        anomaly_data = data[data['is_anomaly']]
        
        if len(anomaly_data) < 5:
            logger.warning("Too few anomaly samples for correlation analysis")
            return patterns
        
        for feature in feature_cols:
            try:
                normal_values = normal_data[feature].dropna().values
                anomaly_values = anomaly_data[feature].dropna().values
                
                if len(normal_values) < 5 or len(anomaly_values) < 5:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(anomaly_values, normal_values)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.std(normal_values)**2 + np.std(anomaly_values)**2) / 2
                )
                cohens_d = (np.mean(anomaly_values) - np.mean(normal_values)) / (pooled_std + 1e-10)
                
                # Determine significance
                is_significant = p_value < 0.05 and abs(cohens_d) > 0.5
                
                patterns.append(CorrelationPattern(
                    feature_name=feature,
                    correlation_strength=abs(cohens_d),
                    p_value=p_value,
                    effect_size=cohens_d,
                    is_significant=is_significant,
                    normal_mean=np.mean(normal_values),
                    anomaly_mean=np.mean(anomaly_values)
                ))
                
            except Exception as e:
                logger.debug(f"Error analyzing correlation for {feature}: {e}")
                continue
        
        # Sort by effect size
        patterns.sort(key=lambda x: abs(x.effect_size), reverse=True)
        
        return patterns
    
    def _learn_cluster_patterns(self, data: pd.DataFrame) -> List[ClusterPattern]:
        """Learn cluster patterns from anomaly data"""
        clusters = []
        
        if 'is_anomaly' not in data.columns:
            return clusters
        
        # Get anomaly records
        anomaly_data = data[data['is_anomaly']]
        
        if len(anomaly_data) < 10:
            logger.warning("Too few anomaly samples for clustering")
            return clusters
        
        # Get feature columns
        exclude_cols = ['timestamp', 'is_anomaly', 'hour', 'day_of_week']
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        # Extract features
        X = anomaly_data[feature_cols].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        n_components = min(10, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(X_pca)
        
        # Create cluster patterns
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            cluster_mask = labels == label
            cluster_points = X_scaled[cluster_mask]
            
            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)
            
            # Find common characteristics
            characteristics = self._extract_cluster_characteristics(
                anomaly_data[cluster_mask],
                feature_cols
            )
            
            clusters.append(ClusterPattern(
                cluster_id=int(label),
                centroid=centroid,
                feature_names=feature_cols,
                size=int(np.sum(cluster_mask)),
                avg_anomaly_score=1.0,  # All are anomalies
                common_characteristics=characteristics
            ))
        
        return clusters
    
    def _extract_cluster_characteristics(self,
                                        cluster_data: pd.DataFrame,
                                        feature_cols: List[str]) -> Dict[str, Any]:
        """Extract common characteristics of a cluster"""
        characteristics = {}
        
        for col in feature_cols:
            values = cluster_data[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                characteristics[col] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'range': [float(mean_val - 2*std_val), float(mean_val + 2*std_val)]
                }
        
        return characteristics
    
    def save_patterns(self, filepath: str):
        """Save pattern library to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pattern_library, f)
        logger.info(f"Patterns saved to {filepath}")
    
    def load_patterns(self, filepath: str) -> PatternLibrary:
        """Load pattern library from disk"""
        with open(filepath, 'rb') as f:
            self.pattern_library = pickle.load(f)
        logger.info(f"Patterns loaded from {filepath}")
        return self.pattern_library


# =============================================================================
# PATTERN MATCHER
# =============================================================================

class PatternMatcher:
    """
    Match current system state against learned patterns
    """
    
    def __init__(self, pattern_library: Optional[PatternLibrary] = None):
        self.pattern_library = pattern_library
        self.match_threshold = 0.6
        self.recent_events = deque(maxlen=100)
        
    def set_pattern_library(self, library: PatternLibrary):
        """Set the pattern library to use for matching"""
        self.pattern_library = library
        
    def match_current_state(self,
                           features: Dict[str, float],
                           recent_events: Optional[List[str]] = None,
                           timestamp: Optional[datetime] = None) -> PatternMatchResult:
        """
        Match current state against learned patterns
        
        Args:
            features: Current feature values
            recent_events: Recent event sequence
            timestamp: Current timestamp
            
        Returns:
            PatternMatchResult with match details
        """
        if self.pattern_library is None:
            return PatternMatchResult()
        
        timestamp = timestamp or datetime.now()
        result = PatternMatchResult()
        scores = []
        
        # Match temporal patterns
        temporal_match = self._match_temporal(timestamp)
        if temporal_match:
            result.temporal_match = temporal_match
            scores.append(temporal_match['score'])
        
        # Match sequential patterns
        if recent_events:
            sequential_matches = self._match_sequential(recent_events)
            result.sequential_matches = sequential_matches
            scores.extend([m['score'] for m in sequential_matches])
        
        # Match correlation patterns
        correlation_matches = self._match_correlations(features)
        result.correlation_matches = correlation_matches
        scores.extend([m['score'] for m in correlation_matches])
        
        # Match cluster patterns
        cluster_matches = self._match_clusters(features)
        result.cluster_matches = cluster_matches
        scores.extend([m['score'] for m in cluster_matches])
        
        # Calculate overall score
        if scores:
            result.overall_score = np.mean(scores)
            result.is_predicted_anomaly = result.overall_score > self.match_threshold
        
        # Track contributing patterns
        result.contributing_patterns = self._identify_contributing_patterns(result)
        
        return result
    
    def _match_temporal(self, timestamp: datetime) -> Optional[Dict]:
        """Match current time against temporal patterns"""
        if self.pattern_library.temporal is None:
            return None
        
        temporal = self.pattern_library.temporal
        
        # Check if current time is in hot periods
        hour_match = timestamp.hour in temporal.hot_hours
        day_match = timestamp.weekday() in temporal.hot_days
        
        # Get risk score
        risk_score = temporal.get_risk_score(timestamp)
        
        # Get hourly anomaly rate
        hourly_rate = temporal.hourly_anomaly_rate.get(timestamp.hour, 0)
        
        return {
            'hour_match': hour_match,
            'day_match': day_match,
            'risk_score': risk_score,
            'hourly_anomaly_rate': hourly_rate,
            'score': min(1.0, risk_score * 2 + hourly_rate)
        }
    
    def _match_sequential(self, recent_events: List[str]) -> List[Dict]:
        """Match recent events against sequential patterns"""
        matches = []
        
        for pattern in self.pattern_library.sequential:
            match_score = pattern.match_score(recent_events)
            
            if match_score > 0.3:  # Minimum match threshold
                matches.append({
                    'pattern_sequence': pattern.sequence,
                    'match_score': match_score,
                    'confidence': pattern.confidence,
                    'time_to_anomaly': pattern.avg_time_to_anomaly,
                    'score': match_score * pattern.confidence
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:5]  # Return top 5
    
    def _match_correlations(self, features: Dict[str, float]) -> List[Dict]:
        """Match features against correlation patterns"""
        matches = []
        
        for corr in self.pattern_library.correlations:
            if not corr.is_significant:
                continue
            
            feature_value = features.get(corr.feature_name)
            if feature_value is None:
                continue
            
            # Check if value is in anomaly range
            if corr.effect_size > 0:
                # Higher values indicate anomaly
                is_anomalous = feature_value > corr.normal_mean
                deviation = (feature_value - corr.normal_mean) / (corr.anomaly_mean - corr.normal_mean + 1e-10)
            else:
                # Lower values indicate anomaly
                is_anomalous = feature_value < corr.normal_mean
                deviation = (corr.normal_mean - feature_value) / (corr.normal_mean - corr.anomaly_mean + 1e-10)
            
            if is_anomalous:
                matches.append({
                    'feature': corr.feature_name,
                    'value': feature_value,
                    'normal_mean': corr.normal_mean,
                    'anomaly_mean': corr.anomaly_mean,
                    'effect_size': corr.effect_size,
                    'score': min(1.0, deviation * abs(corr.effect_size))
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:10]
    
    def _match_clusters(self, features: Dict[str, float]) -> List[Dict]:
        """Match features against cluster patterns"""
        matches = []
        
        # Convert features to array
        feature_array = np.array(list(features.values()))
        
        for cluster in self.pattern_library.clusters:
            # Calculate distance to centroid
            distance = cluster.distance_to(feature_array)
            
            # Convert distance to similarity score
            # Closer to centroid = higher score
            similarity = 1.0 / (1.0 + distance)
            
            if similarity > 0.5:  # Minimum similarity threshold
                matches.append({
                    'cluster_id': cluster.cluster_id,
                    'distance': distance,
                    'similarity': similarity,
                    'cluster_size': cluster.size,
                    'score': similarity
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:5]
    
    def _identify_contributing_patterns(self, result: PatternMatchResult) -> List[str]:
        """Identify which patterns contributed most to the prediction"""
        contributors = []
        
        if result.temporal_match and result.temporal_match['score'] > 0.5:
            contributors.append('temporal')
        
        if result.sequential_matches and result.sequential_matches[0]['score'] > 0.5:
            contributors.append('sequential')
        
        if result.correlation_matches and len(result.correlation_matches) > 2:
            contributors.append('correlation')
        
        if result.cluster_matches and result.cluster_matches[0]['score'] > 0.5:
            contributors.append('cluster')
        
        return contributors
    
    def record_event(self, event: str):
        """Record an event for sequential pattern matching"""
        self.recent_events.append(event)
    
    def get_recent_events(self, n: int = 20) -> List[str]:
        """Get recent events"""
        return list(self.recent_events)[-n:]


# =============================================================================
# PREDICTIVE MODEL
# =============================================================================

class PredictiveBugModel:
    """
    Predictive model that combines ML ensemble with pattern matching
    for proactive bug detection
    """
    
    def __init__(self,
                 ensemble_detector,
                 pattern_matcher: PatternMatcher,
                 config: Optional[Dict] = None):
        self.ensemble = ensemble_detector
        self.pattern_matcher = pattern_matcher
        self.config = config or {}
        
        # Prediction thresholds
        self.ml_threshold = self.config.get('ml_threshold', 0.6)
        self.pattern_threshold = self.config.get('pattern_threshold', 0.5)
        self.combined_threshold = self.config.get('combined_threshold', 0.65)
        
        # Weights for combining scores
        self.ml_weight = self.config.get('ml_weight', 0.6)
        self.pattern_weight = self.config.get('pattern_weight', 0.4)
        
    def predict(self,
               features: np.ndarray,
               feature_dict: Dict[str, float],
               recent_events: Optional[List[str]] = None,
               timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate combined prediction using ML and patterns
        
        Args:
            features: Feature array for ML models
            feature_dict: Feature dictionary for pattern matching
            recent_events: Recent event sequence
            timestamp: Current timestamp
            
        Returns:
            Combined prediction result
        """
        # ML ensemble prediction
        ml_prediction = self.ensemble.predict(features)
        ml_score = ml_prediction.ensemble_score
        
        # Pattern matching
        pattern_result = self.pattern_matcher.match_current_state(
            feature_dict, recent_events, timestamp
        )
        pattern_score = pattern_result.overall_score
        
        # Combine scores
        combined_score = (
            ml_score * self.ml_weight +
            pattern_score * self.pattern_weight
        )
        
        # Determine if anomaly
        is_anomaly = (
            ml_score > self.ml_threshold or
            pattern_score > self.pattern_threshold or
            combined_score > self.combined_threshold
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            ml_prediction, pattern_result, combined_score
        )
        
        return {
            'is_anomaly': is_anomaly,
            'combined_score': combined_score,
            'ml_score': ml_score,
            'pattern_score': pattern_score,
            'confidence': confidence,
            'ml_prediction': ml_prediction,
            'pattern_result': pattern_result,
            'contributing_factors': self._extract_contributing_factors(
                ml_prediction, pattern_result
            )
        }
    
    def _calculate_confidence(self,
                             ml_pred: Any,
                             pattern_result: PatternMatchResult,
                             combined_score: float) -> float:
        """Calculate overall confidence in the prediction"""
        factors = []
        
        # ML confidence
        factors.append(ml_pred.confidence)
        
        # Pattern confidence
        if pattern_result.is_predicted_anomaly:
            factors.append(0.7)  # Base confidence for pattern match
        else:
            factors.append(0.3)
        
        # Combined score confidence
        factors.append(combined_score)
        
        # Model agreement
        factors.append(ml_pred.model_agreement)
        
        return np.mean(factors)
    
    def _extract_contributing_factors(self,
                                     ml_pred: Any,
                                     pattern_result: PatternMatchResult) -> List[str]:
        """Extract contributing factors from both predictions"""
        factors = []
        
        # ML contributing models
        for name, output in ml_pred.individual_predictions.items():
            if output.is_anomaly:
                factors.append(f"{name}: {output.anomaly_score:.3f}")
        
        # Pattern contributing factors
        factors.extend(pattern_result.contributing_patterns)
        
        return factors


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data for testing"""
    np.random.seed(42)
    
    data = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(n_samples):
        timestamp = base_time + timedelta(minutes=i * 10)
        
        # Create features
        is_anomaly = np.random.random() < 0.1
        
        if is_anomaly:
            cpu = np.random.normal(85, 10)
            memory = np.random.normal(90, 5)
            error_rate = np.random.exponential(0.1)
        else:
            cpu = np.random.normal(50, 15)
            memory = np.random.normal(60, 10)
            error_rate = np.random.exponential(0.01)
        
        data.append({
            'timestamp': timestamp,
            'cpu_mean': max(0, min(100, cpu)),
            'memory_mean': max(0, min(100, memory)),
            'error_rate': error_rate,
            'api_latency': np.random.exponential(100) if is_anomaly else np.random.exponential(50),
            'queue_depth': np.random.poisson(50) if is_anomaly else np.random.poisson(10),
            'is_anomaly': is_anomaly
        })
    
    return pd.DataFrame(data)


def example_pattern_learning():
    """Example of pattern learning and matching"""
    print("=" * 60)
    print("Pattern Learning Example")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_samples=2000)
    print(f"\nCreated {len(data)} sample records")
    print(f"Anomaly ratio: {data['is_anomaly'].mean():.2%}")
    
    # Learn patterns
    learner = HistoricalPatternLearner()
    library = learner.learn_patterns(data)
    
    # Print temporal patterns
    if library.temporal:
        print(f"\nTemporal Patterns:")
        print(f"  Hot hours: {library.temporal.hot_hours}")
        print(f"  Hot days: {library.temporal.hot_days}")
    
    # Print correlation patterns
    print(f"\nTop Correlation Patterns:")
    for corr in library.correlations[:5]:
        print(f"  {corr.feature_name}: effect_size={corr.effect_size:.3f}, "
              f"significant={corr.is_significant}")
    
    # Print cluster patterns
    print(f"\nCluster Patterns: {len(library.clusters)} clusters found")
    for cluster in library.clusters:
        print(f"  Cluster {cluster.cluster_id}: {cluster.size} samples")
    
    # Create pattern matcher
    matcher = PatternMatcher(library)
    
    # Test pattern matching
    print("\n" + "=" * 60)
    print("Pattern Matching Test")
    print("=" * 60)
    
    # Simulate current state
    current_features = {
        'cpu_mean': 92.0,
        'memory_mean': 88.0,
        'error_rate': 0.15,
        'api_latency': 500,
        'queue_depth': 80
    }
    
    result = matcher.match_current_state(
        current_features,
        recent_events=['api_call', 'db_query', 'error'],
        timestamp=datetime.now()
    )
    
    print(f"\nOverall match score: {result.overall_score:.3f}")
    print(f"Predicted anomaly: {result.is_predicted_anomaly}")
    print(f"Contributing patterns: {result.contributing_patterns}")
    
    if result.temporal_match:
        print(f"\nTemporal match: {result.temporal_match}")
    
    if result.correlation_matches:
        print(f"\nTop correlation matches:")
        for match in result.correlation_matches[:3]:
            print(f"  {match['feature']}: score={match['score']:.3f}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    example_pattern_learning()
