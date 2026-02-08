"""
Audio Quality Optimization and Enhancement System
For Windows 10 OpenClaw AI Agent Framework

This module provides comprehensive audio quality management including:
- Codec selection and configuration
- Bitrate optimization
- Quality metrics (MOS, PESQ, POLQA)
- Dynamic bitrate adaptation
- Audio enhancement algorithms
- Bandwidth estimation
- Quality-based routing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATIONS
# =============================================================================

class CodecType(Enum):
    """Supported audio codecs"""
    OPUS = "opus"
    G722 = "g722"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"
    G729 = "g729"
    AAC = "aac"
    ILBC = "ilbc"


class NetworkQuality(Enum):
    """Network quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class AlertLevel(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# Opus Configuration for AI Voice Agents
OPUS_CONFIG = {
    "application": "voip",
    "sample_rate": 48000,
    "channels": 1,
    "bitrate": 0,  # Auto (adaptive)
    "vbr": True,
    "complexity": 10,
    "signal": "voice",
    "inband_fec": True,
    "packet_loss_perc": 5,
    "lsb_depth": 24,
    "use_dtx": False,
}

# Bitrate Profiles
BITRATE_PROFILES = {
    "premium": {
        "opus": {"min": 48, "target": 64, "max": 128},
        "g722": {"min": 64, "target": 64, "max": 64},
        "g711": {"min": 64, "target": 64, "max": 64},
        "conditions": "excellent_network",
        "mos_target": 4.5
    },
    "standard": {
        "opus": {"min": 24, "target": 32, "max": 48},
        "g722": {"min": 64, "target": 64, "max": 64},
        "g711": {"min": 64, "target": 64, "max": 64},
        "conditions": "good_network",
        "mos_target": 4.0
    },
    "economy": {
        "opus": {"min": 12, "target": 16, "max": 24},
        "g729": {"min": 8, "target": 8, "max": 8},
        "g711": {"min": 64, "target": 64, "max": 64},
        "conditions": "constrained_network",
        "mos_target": 3.5
    },
    "emergency": {
        "opus": {"min": 6, "target": 8, "max": 12},
        "g729": {"min": 8, "target": 8, "max": 8},
        "conditions": "poor_network",
        "mos_target": 3.0
    }
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "critical": {
        "mos": 3.0,
        "packet_loss": 5.0,
        "jitter": 100,
        "latency": 400,
        "action": "immediate_escalation"
    },
    "warning": {
        "mos": 3.5,
        "packet_loss": 2.0,
        "jitter": 50,
        "latency": 250,
        "action": "notify_operator"
    },
    "info": {
        "mos": 4.0,
        "packet_loss": 1.0,
        "jitter": 30,
        "latency": 150,
        "action": "log_only"
    }
}

# GCC Configuration
GCC_CONFIG = {
    "min_bitrate": 6000,      # 6 kbps
    "max_bitrate": 510000,    # 510 kbps
    "start_bitrate": 32000,   # 32 kbps
    "probe_initial": 300000,  # 300 kbps
    "probe_multipliers": [3, 6],
}

# AGC Configuration
AGC_CONFIG = {
    "target_level": -16,
    "compression_ratio": 2.0,
    "attack_time": 10,
    "release_time": 200,
    "max_gain": 30,
    "min_gain": -10,
    "vad_enabled": True,
}

# Voice Equalization Bands
VOICE_EQ_BANDS = {
    "low_cut": {"freq": 80, "type": "highpass"},
    "low_mud": {"freq": 200, "gain": -3, "q": 1.0},
    "warmth": {"freq": 250, "gain": 2, "q": 1.2},
    "presence": {"freq": 3000, "gain": 3, "q": 1.5},
    "clarity": {"freq": 5000, "gain": 2, "q": 1.0},
    "air": {"freq": 10000, "gain": 1, "q": 0.8},
    "high_cut": {"freq": 16000, "type": "lowpass"},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NetworkMetrics:
    """Network quality metrics"""
    bandwidth: float = 0.0  # kbps
    packet_loss: float = 0.0  # percentage
    jitter: float = 0.0  # ms
    rtt: float = 0.0  # ms
    timestamp: float = field(default_factory=time.time)
    
    def calculate_quality_score(self) -> float:
        """Calculate composite network quality score (0-1)"""
        bandwidth_score = min(self.bandwidth / 1000, 1.0)
        loss_score = max(0, 1 - (self.packet_loss / 10))
        jitter_score = max(0, 1 - (self.jitter / 100))
        rtt_score = max(0, 1 - (self.rtt / 500))
        
        return (bandwidth_score * 0.3 + 
                loss_score * 0.3 + 
                jitter_score * 0.2 + 
                rtt_score * 0.2)


@dataclass
class AudioMetrics:
    """Audio quality metrics"""
    rms_level: float = -30.0  # dB
    peak_level: float = -10.0  # dB
    noise_floor: float = -60.0  # dB
    snr: float = 20.0  # dB
    mos_estimate: float = 4.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    network: NetworkMetrics
    audio: AudioMetrics
    codec: str
    bitrate: int
    mos: float
    alerts: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# CODEC MANAGER
# =============================================================================

class CodecManager:
    """Manages audio codec selection and configuration"""
    
    CODEC_PRIORITY = [
        CodecType.OPUS,
        CodecType.G722,
        CodecType.G711_ULAW,
        CodecType.G729
    ]
    
    CODEC_SPECS = {
        CodecType.OPUS: {
            "sample_rates": [8000, 12000, 16000, 24000, 48000],
            "bitrates": {"min": 6, "max": 510},
            "latency_ms": 5,
            "license": "BSD",
            "complexity": 10
        },
        CodecType.G722: {
            "sample_rates": [16000],
            "bitrates": {"min": 64, "max": 64},
            "latency_ms": 20,
            "license": "Free",
            "complexity": 5
        },
        CodecType.G711_ULAW: {
            "sample_rates": [8000],
            "bitrates": {"min": 64, "max": 64},
            "latency_ms": 10,
            "license": "Free",
            "complexity": 2
        },
        CodecType.G711_ALAW: {
            "sample_rates": [8000],
            "bitrates": {"min": 64, "max": 64},
            "latency_ms": 10,
            "license": "Free",
            "complexity": 2
        },
        CodecType.G729: {
            "sample_rates": [8000],
            "bitrates": {"min": 8, "max": 8},
            "latency_ms": 25,
            "license": "Licensed",
            "complexity": 8
        }
    }
    
    def __init__(self):
        self.current_codec = CodecType.OPUS
        self.current_bitrate = 32000
        self.config = OPUS_CONFIG.copy()
        
    def select_codec(self, network_metrics: NetworkMetrics, 
                     available_codecs: List[CodecType] = None) -> CodecType:
        """Select optimal codec based on network conditions"""
        if available_codecs is None:
            available_codecs = self.CODEC_PRIORITY
            
        quality_score = network_metrics.calculate_quality_score()
        
        # Select codec based on quality score
        if quality_score >= 0.8 and CodecType.OPUS in available_codecs:
            return CodecType.OPUS
        elif quality_score >= 0.6 and CodecType.G722 in available_codecs:
            return CodecType.G722
        elif quality_score >= 0.4 and CodecType.G711_ULAW in available_codecs:
            return CodecType.G711_ULAW
        else:
            # Fallback to G.729 for poor networks
            return CodecType.G729 if CodecType.G729 in available_codecs else CodecType.G711_ULAW
    
    def get_recommended_bitrate(self, codec: CodecType, 
                                 network_metrics: NetworkMetrics) -> int:
        """Get recommended bitrate for codec and network conditions"""
        quality_score = network_metrics.calculate_quality_score()
        specs = self.CODEC_SPECS[codec]
        
        if quality_score >= 0.9:
            profile = BITRATE_PROFILES["premium"]
        elif quality_score >= 0.7:
            profile = BITRATE_PROFILES["standard"]
        elif quality_score >= 0.5:
            profile = BITRATE_PROFILES["economy"]
        else:
            profile = BITRATE_PROFILES["emergency"]
        
        codec_str = codec.value.lower()
        if codec_str in profile:
            return profile[codec_str]["target"] * 1000  # Convert to bps
        
        return specs["bitrates"]["min"] * 1000
    
    def get_opus_config(self) -> Dict[str, Any]:
        """Get Opus configuration"""
        return self.config.copy()


# =============================================================================
# QUALITY METRICS CALCULATOR
# =============================================================================

class QualityMetricsCalculator:
    """Calculates audio quality metrics (MOS, PESQ, etc.)"""
    
    # MOS score mapping from R-factor (E-Model)
    R_TO_MOS = {
        (90, 100): 4.5,
        (80, 90): 4.0,
        (70, 80): 3.5,
        (60, 70): 3.0,
        (50, 60): 2.5,
        (0, 50): 2.0
    }
    
    def __init__(self):
        self.mos_history = deque(maxlen=100)
        
    def calculate_mos_from_network(self, network_metrics: NetworkMetrics) -> float:
        """Estimate MOS from network metrics using E-Model approximation"""
        # R-factor calculation (simplified E-Model)
        R0 = 93.2  # Base R-factor
        
        # Packet loss impairment
        Ie = self._calculate_packet_loss_impairment(network_metrics.packet_loss)
        
        # Delay impairment
        Id = self._calculate_delay_impairment(network_metrics.rtt)
        
        # Equipment impairment (codec-related)
        Ieq = 0  # Assume good codec
        
        R = R0 - Ie - Id - Ieq
        
        # Convert R to MOS
        if R < 0:
            mos = 1.0
        elif R > 100:
            mos = 4.5
        else:
            mos = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6
        
        return min(max(mos, 1.0), 5.0)
    
    def _calculate_packet_loss_impairment(self, packet_loss: float) -> float:
        """Calculate impairment from packet loss"""
        if packet_loss <= 0:
            return 0
        return 11.0 + 8.0 * np.log(1 + 10 * packet_loss)
    
    def _calculate_delay_impairment(self, rtt: float) -> float:
        """Calculate impairment from delay"""
        if rtt <= 150:
            return 0
        return 0.024 * rtt + 0.11 * (rtt - 177.3) * (rtt > 177.3)
    
    def estimate_pesq_score(self, reference: np.ndarray, 
                           degraded: np.ndarray) -> float:
        """
        Estimate PESQ score (simplified implementation)
        Full implementation requires ITU-T P.862 compliant algorithm
        """
        # This is a simplified placeholder
        # Real PESQ requires complex psychoacoustic modeling
        snr = self._calculate_snr(reference, degraded)
        
        # Map SNR to approximate PESQ range
        if snr > 40:
            return 4.5
        elif snr > 30:
            return 4.0
        elif snr > 20:
            return 3.5
        elif snr > 10:
            return 3.0
        else:
            return 2.5
    
    def _calculate_snr(self, reference: np.ndarray, 
                       degraded: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        noise = reference - degraded
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return 100.0
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def calculate_r_factor(self, network_metrics: NetworkMetrics) -> float:
        """Calculate R-factor (E-Model)"""
        R0 = 93.2
        Ie = self._calculate_packet_loss_impairment(network_metrics.packet_loss)
        Id = self._calculate_delay_impairment(network_metrics.rtt)
        return max(0, R0 - Ie - Id)


# =============================================================================
# DYNAMIC BITRATE ADAPTER
# =============================================================================

class DynamicBitrateAdapter:
    """Dynamic bitrate adaptation using GCC-like algorithm"""
    
    def __init__(self):
        self.current_bitrate = GCC_CONFIG["start_bitrate"]
        self.min_bitrate = GCC_CONFIG["min_bitrate"]
        self.max_bitrate = GCC_CONFIG["max_bitrate"]
        self.bandwidth_estimate = GCC_CONFIG["start_bitrate"]
        self.network_state = "STABLE"
        
    def update_bandwidth_estimate(self, network_metrics: NetworkMetrics,
                                   packet_feedback: Dict[str, Any]) -> int:
        """Update bandwidth estimate based on network feedback"""
        # Delay-based estimation
        delay_estimate = self._delay_based_estimate(network_metrics, packet_feedback)
        
        # Loss-based estimation
        loss_estimate = self._loss_based_estimate(network_metrics)
        
        # Combined estimate (conservative: take minimum)
        self.bandwidth_estimate = min(delay_estimate, loss_estimate)
        
        # Apply AIMD
        new_bitrate = self._apply_aimd(self.current_bitrate, self.bandwidth_estimate)
        
        # Clamp to valid range
        self.current_bitrate = int(np.clip(new_bitrate, 
                                           self.min_bitrate, 
                                           self.max_bitrate))
        
        return self.current_bitrate
    
    def _delay_based_estimate(self, network_metrics: NetworkMetrics,
                               packet_feedback: Dict[str, Any]) -> float:
        """Delay-based bandwidth estimation"""
        # Simplified delay-based estimation
        if network_metrics.jitter > 50:
            return self.bandwidth_estimate * 0.9
        elif network_metrics.jitter < 20:
            return min(self.bandwidth_estimate * 1.05, self.max_bitrate)
        return self.bandwidth_estimate
    
    def _loss_based_estimate(self, network_metrics: NetworkMetrics) -> float:
        """Loss-based bandwidth estimation"""
        loss = network_metrics.packet_loss
        
        if loss > 10:
            return self.bandwidth_estimate * 0.5
        elif loss > 5:
            return self.bandwidth_estimate * 0.7
        elif loss > 2:
            return self.bandwidth_estimate * 0.9
        else:
            return self.bandwidth_estimate
    
    def _apply_aimd(self, current: float, estimate: float) -> float:
        """Apply Additive Increase Multiplicative Decrease"""
        if estimate > current * 1.1:
            # Additive increase
            return current + 10000  # 10 kbps increase
        elif estimate < current * 0.9:
            # Multiplicative decrease
            return current * 0.85
        return current
    
    def get_current_bitrate(self) -> int:
        """Get current bitrate in bps"""
        return self.current_bitrate


# =============================================================================
# AUDIO ENHANCEMENT PIPELINE
# =============================================================================

class AudioEnhancementPipeline:
    """Complete audio enhancement pipeline"""
    
    def __init__(self):
        self.agc_enabled = True
        self.ns_enabled = True
        self.eq_enabled = True
        self.agc_target = AGC_CONFIG["target_level"]
        self.current_gain = 0.0
        
    def process(self, audio_frame: np.ndarray, 
                reference_signal: np.ndarray = None) -> np.ndarray:
        """Process audio through enhancement chain"""
        output = audio_frame.copy()
        
        # High-pass filter (remove rumble)
        output = self._highpass_filter(output, cutoff=80)
        
        # Acoustic echo cancellation (if reference provided)
        if reference_signal is not None:
            output = self._acoustic_echo_cancellation(output, reference_signal)
        
        # Noise suppression
        if self.ns_enabled:
            output = self._noise_suppression(output)
        
        # Automatic gain control
        if self.agc_enabled:
            output = self._automatic_gain_control(output)
        
        # Equalization
        if self.eq_enabled:
            output = self._equalize(output)
        
        # Peak limiting
        output = self._peak_limiter(output, threshold=-3)
        
        return output
    
    def _highpass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        # Simplified high-pass filter
        # Real implementation would use scipy.signal.butter
        return audio  # Placeholder
    
    def _acoustic_echo_cancellation(self, microphone: np.ndarray,
                                     reference: np.ndarray) -> np.ndarray:
        """Acoustic echo cancellation"""
        # Placeholder for AEC implementation
        # Real implementation would use adaptive filter + neural post-filter
        return microphone
    
    def _noise_suppression(self, audio: np.ndarray) -> np.ndarray:
        """Noise suppression using spectral subtraction"""
        # Simplified noise suppression
        # Real implementation would use DeepFilterNet or RNNoise
        return audio
    
    def _automatic_gain_control(self, audio: np.ndarray) -> np.ndarray:
        """Automatic gain control"""
        current_level = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
        target_gain = self.agc_target - current_level
        
        # Smooth gain changes
        alpha = 0.1  # Smoothing factor
        self.current_gain = alpha * target_gain + (1 - alpha) * self.current_gain
        
        # Clamp gain
        self.current_gain = np.clip(self.current_gain, 
                                    AGC_CONFIG["min_gain"], 
                                    AGC_CONFIG["max_gain"])
        
        gain_linear = 10 ** (self.current_gain / 20)
        return audio * gain_linear
    
    def _equalize(self, audio: np.ndarray) -> np.ndarray:
        """Apply voice-optimized equalization"""
        # Placeholder for EQ implementation
        # Real implementation would use parametric EQ
        return audio
    
    def _peak_limiter(self, audio: np.ndarray, threshold: float = -3) -> np.ndarray:
        """Peak limiting to prevent clipping"""
        threshold_linear = 10 ** (threshold / 20)
        return np.clip(audio, -threshold_linear, threshold_linear)
    
    @property
    def total_latency_ms(self) -> float:
        """Estimated total pipeline latency"""
        return 50.0  # ~50ms for complete pipeline


# =============================================================================
# QUALITY MONITOR
# =============================================================================

class QualityMonitor:
    """Real-time quality monitoring and alerting"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.alert_handlers = []
        self.calculator = QualityMetricsCalculator()
        
    def register_alert_handler(self, handler: callable):
        """Register an alert handler callback"""
        self.alert_handlers.append(handler)
        
    def update_metrics(self, network_metrics: NetworkMetrics,
                       audio_metrics: AudioMetrics,
                       codec: str, bitrate: int):
        """Update quality metrics"""
        # Calculate MOS
        mos = self.calculator.calculate_mos_from_network(network_metrics)
        
        # Check thresholds
        alerts = self._check_thresholds(network_metrics, audio_metrics, mos)
        
        # Create quality report
        report = QualityReport(
            network=network_metrics,
            audio=audio_metrics,
            codec=codec,
            bitrate=bitrate,
            mos=mos,
            alerts=alerts
        )
        
        self.metrics_history.append(report)
        
        # Trigger alerts
        if alerts:
            for handler in self.alert_handlers:
                handler(alerts)
        
        return report
    
    def _check_thresholds(self, network: NetworkMetrics, 
                          audio: AudioMetrics, mos: float) -> List[Dict]:
        """Check metrics against thresholds"""
        alerts = []
        
        for level, thresholds in ALERT_THRESHOLDS.items():
            if mos < thresholds["mos"]:
                alerts.append({
                    "level": level,
                    "metric": "mos",
                    "value": mos,
                    "threshold": thresholds["mos"],
                    "action": thresholds["action"]
                })
            
            if network.packet_loss > thresholds["packet_loss"]:
                alerts.append({
                    "level": level,
                    "metric": "packet_loss",
                    "value": network.packet_loss,
                    "threshold": thresholds["packet_loss"],
                    "action": thresholds["action"]
                })
            
            if network.jitter > thresholds["jitter"]:
                alerts.append({
                    "level": level,
                    "metric": "jitter",
                    "value": network.jitter,
                    "threshold": thresholds["jitter"],
                    "action": thresholds["action"]
                })
        
        return alerts
    
    def get_average_mos(self, window_seconds: int = 60) -> float:
        """Get average MOS over time window"""
        cutoff_time = time.time() - window_seconds
        recent_reports = [r for r in self.metrics_history 
                         if r.timestamp > cutoff_time]
        
        if not recent_reports:
            return 4.0
        
        return np.mean([r.mos for r in recent_reports])


# =============================================================================
# QUALITY-BASED ROUTER
# =============================================================================

class QualityBasedRouter:
    """Makes routing decisions based on quality metrics"""
    
    def __init__(self):
        self.routes = []
        self.quality_history = {}
        
    def add_route(self, route_id: str, route_config: Dict[str, Any]):
        """Add a potential route"""
        self.routes.append({
            "id": route_id,
            "config": route_config,
            "quality_score": 1.0,
            "historical_mos": 4.0
        })
        
    def select_route(self, call_context: Dict[str, Any]) -> Optional[str]:
        """Select optimal route based on quality predictions"""
        if not self.routes:
            return None
        
        scored_routes = []
        
        for route in self.routes:
            # Predict quality for this route
            predicted_mos = self._predict_route_quality(route, call_context)
            
            # Calculate cost-quality trade-off
            cost = route["config"].get("cost", 1.0)
            priority = call_context.get("priority", 1.0)
            
            score = (predicted_mos * priority) / cost
            scored_routes.append((route["id"], score, predicted_mos))
        
        # Select best route
        best_route = max(scored_routes, key=lambda x: x[1])
        
        logger.info(f"Selected route {best_route[0]} with predicted MOS {best_route[2]:.2f}")
        
        return best_route[0]
    
    def _predict_route_quality(self, route: Dict, 
                               context: Dict[str, Any]) -> float:
        """Predict MOS score for a given route"""
        # Historical performance
        historical_mos = route.get("historical_mos", 4.0)
        
        # Current network conditions
        network_score = route.get("quality_score", 1.0)
        
        # Combine factors
        predicted_mos = historical_mos * 0.6 + network_score * 2.5 * 0.4
        
        return min(predicted_mos, 5.0)
    
    def update_route_quality(self, route_id: str, mos: float):
        """Update route quality history"""
        for route in self.routes:
            if route["id"] == route_id:
                # Exponential moving average
                alpha = 0.3
                route["historical_mos"] = alpha * mos + (1 - alpha) * route["historical_mos"]
                break


# =============================================================================
# MAIN AUDIO QUALITY SYSTEM
# =============================================================================

class AudioQualitySystem:
    """
    Main audio quality optimization system
    Integrates all components for complete quality management
    """
    
    def __init__(self):
        self.codec_manager = CodecManager()
        self.bitrate_adapter = DynamicBitrateAdapter()
        self.enhancement_pipeline = AudioEnhancementPipeline()
        self.quality_monitor = QualityMonitor()
        self.quality_router = QualityBasedRouter()
        self.metrics_calculator = QualityMetricsCalculator()
        
        self.current_codec = CodecType.OPUS
        self.current_bitrate = 32000
        
        logger.info("Audio Quality System initialized")
        
    def initialize_call(self, network_metrics: NetworkMetrics,
                       available_codecs: List[CodecType] = None):
        """Initialize audio quality for a new call"""
        # Select optimal codec
        self.current_codec = self.codec_manager.select_codec(
            network_metrics, available_codecs
        )
        
        # Set initial bitrate
        self.current_bitrate = self.codec_manager.get_recommended_bitrate(
            self.current_codec, network_metrics
        )
        
        logger.info(f"Call initialized with {self.current_codec.value} "
                   f"at {self.current_bitrate/1000:.1f} kbps")
        
    def process_audio(self, audio_frame: np.ndarray,
                     reference_signal: np.ndarray = None) -> np.ndarray:
        """Process audio through enhancement pipeline"""
        return self.enhancement_pipeline.process(audio_frame, reference_signal)
    
    def update_network_conditions(self, network_metrics: NetworkMetrics,
                                   packet_feedback: Dict[str, Any] = None):
        """Update system based on new network conditions"""
        # Update bitrate
        if packet_feedback:
            self.current_bitrate = self.bitrate_adapter.update_bandwidth_estimate(
                network_metrics, packet_feedback
            )
        
        # Update quality metrics
        audio_metrics = AudioMetrics()  # Would be measured from actual audio
        report = self.quality_monitor.update_metrics(
            network_metrics, audio_metrics,
            self.current_codec.value, self.current_bitrate
        )
        
        # Check for codec switch need
        optimal_codec = self.codec_manager.select_codec(network_metrics)
        if optimal_codec != self.current_codec:
            logger.info(f"Codec switch recommended: {self.current_codec.value} -> {optimal_codec.value}")
        
        return report
    
    def get_quality_report(self) -> QualityReport:
        """Get current quality report"""
        if self.quality_monitor.metrics_history:
            return self.quality_monitor.metrics_history[-1]
        return None
    
    def select_route(self, call_context: Dict[str, Any]) -> Optional[str]:
        """Select optimal route for call"""
        return self.quality_router.select_route(call_context)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "codec": self.current_codec.value,
            "bitrate_kbps": self.current_bitrate / 1000,
            "average_mos": self.quality_monitor.get_average_mos(),
            "enhancement_latency_ms": self.enhancement_pipeline.total_latency_ms,
            "network_state": self.bitrate_adapter.network_state
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example usage of the Audio Quality System"""
    
    # Initialize system
    audio_system = AudioQualitySystem()
    
    # Simulate network conditions
    network_metrics = NetworkMetrics(
        bandwidth=1000,      # 1 Mbps
        packet_loss=0.5,     # 0.5%
        jitter=15,           # 15ms
        rtt=100              # 100ms
    )
    
    # Initialize call
    audio_system.initialize_call(network_metrics)
    
    # Process audio frame
    sample_rate = 48000
    frame_duration = 0.02  # 20ms
    frame_samples = int(sample_rate * frame_duration)
    audio_frame = np.random.randn(frame_samples) * 0.1
    
    enhanced_audio = audio_system.process_audio(audio_frame)
    
    # Update network conditions
    report = audio_system.update_network_conditions(network_metrics)
    
    # Get system status
    status = audio_system.get_system_status()
    print(f"System Status: {status}")
    
    return audio_system


if __name__ == "__main__":
    example_usage()
