# Voice Activity Detection (VAD) and Audio Preprocessing Architecture
## Technical Specification for Windows 10 OpenClaw AI Agent System

**Version:** 1.0  
**Date:** 2025  
**Target Platform:** Windows 10  
**Python Version:** 3.10+  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Voice Activity Detection (VAD)](#3-voice-activity-detection-vad)
4. [Noise Suppression](#4-noise-suppression)
5. [Echo Cancellation (AEC)](#5-echo-cancellation-aec)
6. [Audio Normalization and Gain Control](#6-audio-normalization-and-gain-control)
7. [Audio Segmentation and Framing](#7-audio-segmentation-and-framing)
8. [Silence Detection and Trimming](#8-silence-detection-and-trimming)
9. [Audio Quality Enhancement](#9-audio-quality-enhancement)
10. [Complete Preprocessing Pipeline](#10-complete-preprocessing-pipeline)
11. [Windows Audio Capture Implementation](#11-windows-audio-capture-implementation)
12. [Performance Optimization](#12-performance-optimization)
13. [Implementation Code](#13-implementation-code)

---

## 1. Executive Summary

This specification defines a comprehensive Voice Activity Detection (VAD) and audio preprocessing architecture for a Windows 10-based AI agent system inspired by OpenClaw. The system integrates multiple advanced audio processing techniques to ensure high-quality speech recognition, real-time responsiveness, and robust performance in various acoustic environments.

### Key Features

- **Multi-Algorithm VAD:** Dual VAD approach using Silero VAD (primary) and WebRTC VAD (fallback)
- **Neural Noise Suppression:** RNNoise for real-time noise reduction
- **Acoustic Echo Cancellation:** SpeexDSP AEC for speaker-microphone echo removal
- **Adaptive Gain Control:** Automatic audio level normalization
- **Real-time Processing:** Sub-100ms latency for interactive voice applications
- **STT Optimization:** Pipeline specifically tuned for Whisper and other STT engines

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUDIO PREPROCESSING PIPELINE ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   CAPTURE    │───▶│  PREPROCESS  │───▶│   ENHANCE    │───▶│    VAD     │ │
│  │   (WASAPI)   │    │  (Resample)  │    │ (Denoise)    │    │  (Silero)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬──────┘ │
│                                                                    │        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │        │
│  │   OUTPUT     │◄───│  SEGMENT     │◄───│   TRIM       │◄─────────┘        │
│  │   (STT)      │    │  (Framing)   │    │  (Silence)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
│  PARALLEL PROCESSES:                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │   AGC    │  │   AEC    │  │  RNNoise │  │  Metrics │            │   │
│  │  │ (Gain)   │  │  (Echo)  │  │  (Denoise)│  │  (Stats) │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Voice Activity Detection (VAD)

### 3.1 VAD Algorithm Comparison

| Model | Accuracy | Latency | CPU Usage | License | Best For |
|-------|----------|---------|-----------|---------|----------|
| **Silero VAD** | 87.7% TPR @ 5% FPR | 30-100ms | Medium | MIT | Production, noisy environments |
| **WebRTC VAD** | 50% TPR @ 5% FPR | 10-30ms | Very Low | BSD | Low-resource, simple scenarios |
| **TEN VAD** | 95%+ TPR | 30ms | Low | Open Source | Real-time agents |
| **Cobra VAD** | 98.9% TPR @ 5% FPR | <30ms | Very Low | Commercial | Enterprise applications |

### 3.2 Recommended: Dual VAD Strategy

**Primary:** Silero VAD (v4.0+)  
**Fallback:** WebRTC VAD (aggressiveness mode 1-2)

### 3.3 Silero VAD Configuration

```python
# Silero VAD Parameters
SILERO_VAD_CONFIG = {
    "model_version": "4.0",
    "threshold": 0.5,           # Speech probability threshold (0.0-1.0)
    "sampling_rate": 16000,     # Supported: 8000, 16000, 32000, 48000
    "min_speech_duration_ms": 250,    # Minimum speech segment
    "max_speech_duration_s": 30,      # Maximum speech segment
    "min_silence_duration_ms": 500,   # Silence to trigger end-of-speech
    "window_size_samples": 512,       # Frame size (512 for 16kHz)
    "speech_pad_ms": 100,             # Padding around speech segments
}
```

### 3.4 WebRTC VAD Configuration

```python
# WebRTC VAD Parameters
WEBRTC_VAD_CONFIG = {
    "aggressiveness": 1,        # 0=least aggressive, 3=most aggressive
    "frame_duration_ms": 30,    # 10, 20, or 30ms
    "sampling_rate": 16000,     # 8000, 16000, 32000, or 48000
}
```

### 3.5 VAD State Machine

```
┌─────────────┐     speech_prob > 0.5      ┌─────────────┐
│   IDLE      │ ─────────────────────────▶ │  SPEAKING   │
│  (silent)   │                            │  (active)   │
└─────────────┘                            └──────┬──────┘
      ▲                                           │
      │         silence_duration > 500ms          │
      └───────────────────────────────────────────┘
```

### 3.6 Hysteresis Implementation

```python
class VADHysteresis:
    """
    Hysteresis state machine for stable VAD decisions.
    Prevents rapid toggling between speech/silence states.
    """
    def __init__(self, 
                 speech_threshold: float = 0.6,
                 silence_threshold: float = 0.4,
                 min_speech_frames: int = 5,
                 min_silence_frames: int = 10):
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
    
    def process(self, speech_prob: float) -> bool:
        if speech_prob >= self.speech_threshold:
            self.speech_frame_count += 1
            self.silence_frame_count = 0
            if self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
        elif speech_prob <= self.silence_threshold:
            self.silence_frame_count += 1
            self.speech_frame_count = 0
            if self.silence_frame_count >= self.min_silence_frames:
                self.is_speaking = False
        return self.is_speaking
```

---

## 4. Noise Suppression

### 4.1 Noise Suppression Options

| Method | Type | Latency | Quality | CPU Usage | Best For |
|--------|------|---------|---------|-----------|----------|
| **RNNoise** | Neural | 10ms | Excellent | Medium | Real-time speech |
| **SpeexDSP** | Traditional | 5ms | Good | Low | Embedded systems |
| **noisereduce** | Spectral | 20ms | Good | Medium | Offline processing |
| **Wiener Filter** | Statistical | 5ms | Moderate | Low | Simple noise |

### 4.2 Recommended: RNNoise

**Why RNNoise:**
- Deep learning-based (RNN architecture)
- Trained on extensive speech+noise datasets
- Real-time capable (< 10ms latency)
- Open source (BSD license)
- Provides VAD probability as side output

### 4.3 RNNoise Configuration

```python
# RNNoise Parameters
RNNOISE_CONFIG = {
    "sample_rate": 48000,       # RNNoise native rate
    "frame_size": 480,          # 10ms at 48kHz
    "vad_threshold": 0.6,       # Speech probability threshold
    "gain_control": True,       # Automatic gain control
}
```

### 4.4 RNNoise Integration

```python
from pyrnnoise import RNNoise
import numpy as np

class NoiseSuppressor:
    def __init__(self, sample_rate: int = 48000):
        self.denoiser = RNNoise(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single audio frame.
        
        Returns:
            (speech_probability, denoised_frame)
        """
        speech_prob, denoised = self.denoiser.denoise_frame(frame)
        return speech_prob, denoised
    
    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Process audio chunk and return denoised audio."""
        denoised_chunks = []
        for speech_prob, denoised_frame in self.denoiser.denoise_chunk(audio):
            denoised_chunks.append(denoised_frame)
        return np.concatenate(denoised_chunks, axis=-1)
```

### 4.5 Spectral Subtraction (Fallback)

```python
import numpy as np
from scipy.signal import stft, istft

class SpectralNoiseReducer:
    """
    Spectral subtraction noise reduction.
    Used when RNNoise is unavailable.
    """
    def __init__(self, 
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 noise_reduction_factor: float = 1.5):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_reduction_factor = noise_reduction_factor
        self.noise_profile = None
        
    def estimate_noise_profile(self, noise_audio: np.ndarray):
        """Estimate noise profile from silent audio segment."""
        _, _, Zxx = stft(noise_audio, nperseg=self.n_fft, 
                         noverlap=self.n_fft - self.hop_length)
        self.noise_profile = np.mean(np.abs(Zxx), axis=1)
        
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction."""
        f, t, Zxx = stft(audio, nperseg=self.n_fft,
                        noverlap=self.n_fft - self.hop_length)
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Spectral subtraction
        if self.noise_profile is not None:
            magnitude_reduced = np.maximum(
                magnitude - self.noise_reduction_factor * self.noise_profile[:, None],
                0
            )
        else:
            magnitude_reduced = magnitude
            
        # Reconstruct signal
        Zxx_reduced = magnitude_reduced * np.exp(1j * phase)
        _, audio_reduced = istft(Zxx_reduced, nperseg=self.n_fft,
                                 noverlap=self.n_fft - self.hop_length)
        
        return audio_reduced[:len(audio)]
```

---

## 5. Echo Cancellation (AEC)

### 5.1 AEC Requirements

For AI agent systems with TTS playback, acoustic echo cancellation is critical to prevent the agent from hearing itself.

### 5.2 AEC Options

| Method | Latency | Quality | Complexity | Best For |
|--------|---------|---------|------------|----------|
| **SpeexDSP AEC** | 20ms | Good | Low | Real-time systems |
| **WebRTC AEC3** | 15ms | Excellent | Medium | Web applications |
| **AEC-rs** | 20ms | Good | Low | Rust/Python hybrid |
| **Linear AEC** | 10ms | Moderate | Low | Simple echo |

### 5.3 Recommended: SpeexDSP AEC

```python
# SpeexDSP AEC Configuration
SPEEX_AEC_CONFIG = {
    "sample_rate": 16000,
    "filter_length_ms": 500,     # Echo tail length
    "frame_size": 160,           # 10ms at 16kHz
    "enable_preprocess": True,   # Enable noise suppression
    "agc_level": 8000,           # AGC target level
    "denoise_enabled": True,
    "dererb_enabled": False,     # Dereverberation
}
```

### 5.4 AEC Implementation

```python
import numpy as np

try:
    from pyspeexaec import SpeexAEC
    HAS_SPEEX = True
except ImportError:
    HAS_SPEEX = False

class EchoCanceller:
    """
    Acoustic Echo Canceller using SpeexDSP.
    
    Reference (far-end) audio is the TTS output.
    Recorded (near-end) audio is the microphone input.
    """
    def __init__(self, 
                 sample_rate: int = 16000,
                 filter_length_ms: int = 500):
        self.sample_rate = sample_rate
        self.filter_length = int(filter_length_ms * sample_rate / 1000)
        self.frame_size = int(0.01 * sample_rate)  # 10ms frames
        
        if HAS_SPEEX:
            self.aec = SpeexAEC(
                sampling_rate=sample_rate,
                filter_length=self.filter_length
            )
        else:
            self.aec = None
            # Fallback: simple delay-line subtraction
            self.ref_buffer = []
            self.delay_samples = int(0.02 * sample_rate)  # 20ms delay
            
    def process(self, 
                recorded: np.ndarray, 
                reference: np.ndarray) -> np.ndarray:
        """
        Cancel echo from recorded audio.
        
        Args:
            recorded: Microphone input (near-end + echo)
            reference: Speaker output (far-end)
            
        Returns:
            Echo-cancelled audio
        """
        if self.aec:
            return self.aec.process(recorded, reference)
        else:
            # Fallback implementation
            return self._fallback_cancel(recorded, reference)
    
    def _fallback_cancel(self, 
                         recorded: np.ndarray, 
                         reference: np.ndarray) -> np.ndarray:
        """Simple delay-and-subtract fallback."""
        # Pad reference to account for delay
        ref_padded = np.pad(reference, (self.delay_samples, 0), mode='constant')
        ref_padded = ref_padded[:len(recorded)]
        
        # Adaptive gain estimation
        corr = np.correlate(recorded, ref_padded, mode='valid')
        if len(corr) > 0:
            delay = np.argmax(np.abs(corr))
            gain = np.sum(recorded * np.roll(ref_padded, delay)) / \
                   (np.sum(ref_padded ** 2) + 1e-10)
            gain = np.clip(gain, -1.0, 1.0)
        else:
            gain = 0.0
            
        # Subtract estimated echo
        echo_estimate = gain * ref_padded
        cancelled = recorded - echo_estimate
        
        return cancelled
```

---

## 6. Audio Normalization and Gain Control

### 6.1 Automatic Gain Control (AGC)

```python
import numpy as np
from collections import deque

class AutomaticGainControl:
    """
    Automatic Gain Control with configurable attack/release times.
    
    Implements a digital AGC similar to hardware AGC circuits.
    """
    def __init__(self,
                 target_level_db: float = -20.0,
                 attack_time_ms: float = 10.0,
                 release_time_ms: float = 100.0,
                 sample_rate: int = 16000,
                 max_gain_db: float = 30.0,
                 min_gain_db: float = -20.0):
        self.target_level = 10 ** (target_level_db / 20)
        self.attack_coef = np.exp(-1.0 / (attack_time_ms * sample_rate / 1000))
        self.release_coef = np.exp(-1.0 / (release_time_ms * sample_rate / 1000))
        self.max_gain = 10 ** (max_gain_db / 20)
        self.min_gain = 10 ** (min_gain_db / 20)
        self.current_gain = 1.0
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply AGC to audio signal."""
        output = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            # Calculate instantaneous level (RMS over small window)
            input_level = np.abs(sample)
            
            # Calculate target gain
            if input_level > 0:
                target_gain = self.target_level / input_level
                target_gain = np.clip(target_gain, self.min_gain, self.max_gain)
            else:
                target_gain = self.max_gain
                
            # Smooth gain changes
            if target_gain > self.current_gain:
                # Attack phase (increasing gain)
                self.current_gain = self.attack_coef * self.current_gain + \
                                   (1 - self.attack_coef) * target_gain
            else:
                # Release phase (decreasing gain)
                self.current_gain = self.release_coef * self.current_gain + \
                                   (1 - self.release_coef) * target_gain
                
            output[i] = sample * self.current_gain
            
        return output
```

### 6.2 Peak Normalization

```python
class PeakNormalizer:
    """Simple peak normalization with headroom."""
    def __init__(self, target_peak_db: float = -3.0):
        self.target_peak = 10 ** (target_peak_db / 20)
        
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target peak level."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            gain = self.target_peak / peak
            return audio * gain
        return audio
```

### 6.3 RMS Normalization (EBU R128 Inspired)

```python
class RMSNormalizer:
    """
    RMS-based normalization for consistent perceived loudness.
    Similar to EBU R128 standard.
    """
    def __init__(self, 
                 target_rms_db: float = -23.0,
                 window_size: int = 2048):
        self.target_rms = 10 ** (target_rms_db / 20)
        self.window_size = window_size
        
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize based on RMS energy."""
        # Calculate RMS over windows
        rms_values = []
        for i in range(0, len(audio), self.window_size):
            window = audio[i:i + self.window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
            
        # Use median RMS for robustness
        median_rms = np.median(rms_values)
        
        if median_rms > 0:
            gain = self.target_rms / median_rms
            return audio * gain
        return audio
```

---

## 7. Audio Segmentation and Framing

### 7.1 Framing Parameters

```python
# Standard Speech Processing Frame Parameters
FRAME_CONFIG = {
    "sample_rate": 16000,
    "frame_length_ms": 25,      # 25ms frames (standard for speech)
    "frame_shift_ms": 10,       # 10ms shift (60% overlap)
    "window_type": "hamming",   # Hamming window for spectral analysis
}

# Calculate frame parameters
FRAME_LENGTH = int(FRAME_CONFIG["frame_length_ms"] * FRAME_CONFIG["sample_rate"] / 1000)  # 400 samples
FRAME_SHIFT = int(FRAME_CONFIG["frame_shift_ms"] * FRAME_CONFIG["sample_rate"] / 1000)    # 160 samples
```

### 7.2 Framing Implementation

```python
import numpy as np
from scipy.signal import get_window

class AudioFramer:
    """
    Audio framing with configurable window functions.
    """
    def __init__(self,
                 frame_length: int = 400,
                 frame_shift: int = 160,
                 window_type: str = "hamming"):
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.window = get_window(window_type, frame_length)
        
    def frame(self, audio: np.ndarray) -> np.ndarray:
        """
        Split audio into overlapping frames.
        
        Returns:
            2D array of shape (num_frames, frame_length)
        """
        num_frames = 1 + (len(audio) - self.frame_length) // self.frame_shift
        
        frames = np.zeros((num_frames, self.frame_length))
        for i in range(num_frames):
            start = i * self.frame_shift
            frame = audio[start:start + self.frame_length]
            
            # Pad last frame if necessary
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
                
            # Apply window
            frames[i] = frame * self.window
            
        return frames
    
    def unframe(self, frames: np.ndarray) -> np.ndarray:
        """
        Reconstruct audio from frames using overlap-add.
        """
        num_frames = len(frames)
        output_length = self.frame_length + (num_frames - 1) * self.frame_shift
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        
        for i, frame in enumerate(frames):
            start = i * self.frame_shift
            output[start:start + self.frame_length] += frame * self.window
            window_sum[start:start + self.frame_length] += self.window ** 2
            
        # Normalize by window sum
        output = output / (window_sum + 1e-10)
        
        return output
```

### 7.3 Window Functions

```python
class WindowFunctions:
    """Collection of window functions for audio processing."""
    
    @staticmethod
    def hamming(n: int) -> np.ndarray:
        """Hamming window - good for speech analysis."""
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    
    @staticmethod
    def hann(n: int) -> np.ndarray:
        """Hann window - good for general spectral analysis."""
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    
    @staticmethod
    def blackman(n: int) -> np.ndarray:
        """Blackman window - better frequency resolution."""
        a0, a1, a2 = 0.42, 0.5, 0.08
        t = 2 * np.pi * np.arange(n) / (n - 1)
        return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t)
```

---

## 8. Silence Detection and Trimming

### 8.1 Silence Detection Parameters

```python
SILENCE_CONFIG = {
    "top_db": 40,               # dB below reference to consider silence
    "frame_length": 2048,       # Analysis frame length
    "hop_length": 512,          # Hop between frames
    "min_silence_duration": 0.3,  # Minimum silence to trim (seconds)
    "keep_silence": 0.1,        # Silence to keep around speech (seconds)
}
```

### 8.2 Silence Trimming Implementation

```python
import numpy as np
from scipy.signal import get_window

class SilenceTrimmer:
    """
    Silence detection and trimming using RMS energy.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 top_db: float = 40.0,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 min_silence_duration: float = 0.3,
                 keep_silence: float = 0.1):
        self.sample_rate = sample_rate
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_silence_frames = int(min_silence_duration * sample_rate / hop_length)
        self.keep_silence_samples = int(keep_silence * sample_rate)
        
    def calculate_rms(self, audio: np.ndarray) -> np.ndarray:
        """Calculate RMS energy per frame."""
        num_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        rms = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
            
        return rms
    
    def detect_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect silent regions in audio.
        
        Returns:
            Boolean array indicating silent frames
        """
        rms = self.calculate_rms(audio)
        
        # Reference level (max RMS)
        ref = np.max(rms)
        if ref == 0:
            return np.ones_like(rms, dtype=bool)
            
        # Convert to dB
        rms_db = 20 * np.log10(rms / ref + 1e-10)
        
        # Silence is below threshold
        is_silence = rms_db < -self.top_db
        
        return is_silence
    
    def trim(self, audio: np.ndarray) -> tuple:
        """
        Trim leading and trailing silence.
        
        Returns:
            (trimmed_audio, (start_sample, end_sample))
        """
        is_silence = self.detect_silence(audio)
        
        # Find first non-silent frame
        non_silent_indices = np.where(~is_silence)[0]
        
        if len(non_silent_indices) == 0:
            return np.array([]), (0, 0)
            
        first_frame = non_silent_indices[0]
        last_frame = non_silent_indices[-1]
        
        # Convert to samples
        start_sample = max(0, first_frame * self.hop_length - self.keep_silence_samples)
        end_sample = min(len(audio), (last_frame + 1) * self.hop_length + self.keep_silence_samples)
        
        return audio[start_sample:end_sample], (start_sample, end_sample)
    
    def split_on_silence(self, audio: np.ndarray) -> list:
        """
        Split audio into segments separated by silence.
        
        Returns:
            List of audio segments
        """
        is_silence = self.detect_silence(audio)
        
        segments = []
        segment_start = None
        silence_count = 0
        
        for i, silent in enumerate(is_silence):
            if not silent:
                if segment_start is None:
                    segment_start = i
                silence_count = 0
            else:
                silence_count += 1
                if segment_start is not None and silence_count >= self.min_silence_frames:
                    # End of segment
                    end_sample = (i - silence_count + 1) * self.hop_length
                    start_sample = max(0, segment_start * self.hop_length - self.keep_silence_samples)
                    segments.append(audio[start_sample:end_sample + self.keep_silence_samples])
                    segment_start = None
                    
        # Add final segment
        if segment_start is not None:
            start_sample = max(0, segment_start * self.hop_length - self.keep_silence_samples)
            segments.append(audio[start_sample:])
            
        return segments
```

---

## 9. Audio Quality Enhancement

### 9.1 High-Pass Filter (Remove DC and Low-Frequency Noise)

```python
from scipy.signal import butter, filtfilt

class HighPassFilter:
    """High-pass filter to remove DC offset and low-frequency noise."""
    def __init__(self, cutoff_hz: float = 80.0, sample_rate: int = 16000):
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        self.b, self.a = butter(4, normalized_cutoff, btype='high')
        
    def filter(self, audio: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, audio)
```

### 9.2 Pre-emphasis Filter

```python
class PreEmphasis:
    """
    Pre-emphasis filter to boost high frequencies.
    Standard in speech processing (coefficient ~0.97).
    """
    def __init__(self, coefficient: float = 0.97):
        self.coefficient = coefficient
        
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter."""
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.coefficient * audio[:-1]
        return emphasized
```

### 9.3 Dynamic Range Compression

```python
class DynamicRangeCompressor:
    """
    Simple dynamic range compressor.
    Reduces volume of loud signals, raises quiet signals.
    """
    def __init__(self,
                 threshold_db: float = -20.0,
                 ratio: float = 4.0,
                 attack_ms: float = 5.0,
                 release_ms: float = 50.0,
                 sample_rate: int = 16000):
        self.threshold = 10 ** (threshold_db / 20)
        self.ratio = ratio
        self.attack_coef = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self.release_coef = np.exp(-1.0 / (release_ms * sample_rate / 1000))
        self.envelope = 0.0
        
    def compress(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        output = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            # Update envelope
            abs_sample = abs(sample)
            if abs_sample > self.envelope:
                self.envelope = self.attack_coef * self.envelope + \
                               (1 - self.attack_coef) * abs_sample
            else:
                self.envelope = self.release_coef * self.envelope + \
                               (1 - self.release_coef) * abs_sample
                
            # Calculate gain
            if self.envelope > self.threshold:
                gain_reduction = (self.envelope / self.threshold) ** (1 / self.ratio - 1)
                gain = gain_reduction
            else:
                gain = 1.0
                
            output[i] = sample * gain
            
        return output
```

---

## 10. Complete Preprocessing Pipeline

### 10.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE AUDIO PREPROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Raw audio from microphone (48kHz, 16-bit, mono)                     │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 1: CAPTURE & FORMAT                                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ WASAPI       │─▶│ Resample     │─▶│ Convert      │              │    │
│  │  │ Capture      │  │ 48k→16k      │  │ int16→float  │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 2: ENHANCEMENT                                                │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ High-Pass    │─▶│ RNNoise      │─▶│ AGC          │              │    │
│  │  │ Filter       │  │ Denoise      │  │ Normalize    │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 3: ECHO CANCELLATION (if TTS active)                          │    │
│  │  ┌──────────────┐  ┌──────────────┐                                  │    │
│  │  │ Reference    │─▶│ SpeexDSP     │                                  │    │
│  │  │ Buffer       │  │ AEC          │                                  │    │
│  │  └──────────────┘  └──────────────┘                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 4: VAD & SEGMENTATION                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ Silero VAD   │─▶│ Hysteresis   │─▶│ Trim Silence │              │    │
│  │  │ Detection    │  │ State Machine│  │ & Segment    │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 5: OUTPUT FORMATTING                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐                                  │    │
│  │  │ Frame        │─▶│ Window       │                                  │    │
│  │  │ (25ms/10ms)  │  │ (Hamming)    │                                  │    │
│  │  └──────────────┘  └──────────────┘                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  OUTPUT: Processed audio ready for STT (16kHz, float32, framed)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Pipeline Configuration

```python
# Complete Pipeline Configuration
PIPELINE_CONFIG = {
    # Capture settings
    "capture": {
        "sample_rate": 48000,
        "target_sample_rate": 16000,
        "channels": 1,
        "format": "int16",
        "chunk_size": 1024,
    },
    
    # Enhancement settings
    "enhancement": {
        "highpass_cutoff": 80.0,
        "preemphasis_coef": 0.97,
        "enable_denoise": True,
        "enable_agc": True,
        "agc_target_db": -20.0,
    },
    
    # Echo cancellation
    "aec": {
        "enabled": True,
        "filter_length_ms": 500,
        "enable_preprocess": True,
    },
    
    # VAD settings
    "vad": {
        "primary": "silero",
        "fallback": "webrtc",
        "silero_threshold": 0.5,
        "webrtc_aggressiveness": 1,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
    },
    
    # Silence trimming
    "silence": {
        "top_db": 40,
        "min_silence_duration": 0.3,
        "keep_silence": 0.1,
    },
    
    # Framing settings
    "framing": {
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "window_type": "hamming",
    },
}
```

---

## 11. Windows Audio Capture Implementation

### 11.1 WASAPI Capture with PyAudioWPatch

```python
import pyaudiowpatch as pyaudio
import numpy as np
import threading
import queue

class WindowsAudioCapture:
    """
    Windows audio capture using WASAPI.
    Supports both microphone input and speaker loopback.
    """
    def __init__(self,
                 sample_rate: int = 48000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 use_loopback: bool = False):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.use_loopback = use_loopback
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def get_devices(self):
        """List all available audio devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': info['defaultSampleRate'],
                'host_api': info['hostApi']
            })
        return devices
    
    def start(self, device_index: int = None):
        """Start audio capture."""
        if self.use_loopback:
            # Use WASAPI loopback for speaker capture
            if device_index is None:
                device_info = self.audio.get_default_wasapi_loopback()
            else:
                device_info = self.audio.get_wasapi_loopback_analogue_by_index(device_index)
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.chunk_size,
                stream_callback=self._callback
            )
        else:
            # Standard microphone capture
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._callback
            )
            
        self.is_recording = True
        self.stream.start_stream()
        
    def _callback(self, in_data, frame_count, time_info, status):
        """Audio callback - called for each chunk."""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to queue for processing
        self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def read(self, timeout: float = 0.1) -> np.ndarray:
        """Read audio data from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop audio capture."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
```

---

## 12. Performance Optimization

### 12.1 Latency Budget

| Stage | Target Latency | Maximum Latency |
|-------|---------------|-----------------|
| Capture | 10ms | 20ms |
| Resampling | 5ms | 10ms |
| Denoising | 10ms | 20ms |
| AEC | 20ms | 30ms |
| VAD | 30ms | 50ms |
| Framing | 5ms | 10ms |
| **Total** | **80ms** | **140ms** |

### 12.2 CPU Usage Targets

| Component | Target CPU | Notes |
|-----------|-----------|-------|
| VAD (Silero) | 5-10% | On modern CPU |
| RNNoise | 3-5% | GPU acceleration optional |
| AEC | 2-3% | SpeexDSP optimized |
| Full Pipeline | 15-20% | Including overhead |

### 12.3 Memory Usage

| Component | Memory Usage |
|-----------|-------------|
| Silero VAD Model | ~50MB |
| RNNoise Model | ~5MB |
| AEC Filter | ~2MB |
| Audio Buffers | ~10MB |
| **Total** | **~70MB** |

### 12.4 Optimization Strategies

1. **Use ring buffers** for audio streaming
2. **Batch processing** where possible
3. **Numba/JIT compilation** for hot paths
4. **SIMD operations** via NumPy
5. **Multi-threading** for parallel stages
6. **GPU acceleration** for neural models (optional)

---

## 13. Implementation Code

### 13.1 Complete AudioPreprocessor Class

```python
"""
Complete Audio Preprocessing Pipeline for Windows 10 AI Agent System
"""

import numpy as np
import torch
import librosa
from scipy import signal
from scipy.signal import resample, butter, filtfilt, get_window
from typing import Optional, Tuple, List, Callable
import queue
import threading
from dataclasses import dataclass
from enum import Enum
import logging

# Optional imports with fallbacks
try:
    import webrtcvad
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

try:
    from pyrnnoise import RNNoise
    HAS_RNNOISE = True
except ImportError:
    HAS_RNNOISE = False

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VADType(Enum):
    """VAD algorithm types."""
    SILERO = "silero"
    WEBRTC = "webrtc"
    ENERGY = "energy"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    # Capture
    capture_sample_rate: int = 48000
    target_sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # Enhancement
    highpass_cutoff: float = 80.0
    preemphasis_coef: float = 0.97
    enable_denoise: bool = True
    enable_agc: bool = True
    agc_target_db: float = -20.0
    
    # VAD
    vad_type: VADType = VADType.SILERO
    silero_threshold: float = 0.5
    webrtc_aggressiveness: int = 1
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 100
    
    # Silence
    silence_top_db: float = 40.0
    min_silence_duration: float = 0.3
    keep_silence: float = 0.1
    
    # Framing
    frame_length_ms: int = 25
    frame_shift_ms: int = 10
    window_type: str = "hamming"


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for voice AI agents.
    
    Pipeline stages:
    1. Capture (WASAPI)
    2. Resampling
    3. High-pass filtering
    4. Noise suppression (RNNoise)
    5. AGC normalization
    6. VAD detection
    7. Silence trimming
    8. Framing
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._init_vad()
        self._init_enhancement()
        self._init_framing()
        self._init_silence_trimmer()
        
        # Processing state
        self.is_processing = False
        self.audio_buffer = []
        self.vad_state = False
        self.speech_start_time = None
        
    def _init_vad(self):
        """Initialize VAD engine."""
        self.vad_model = None
        self.vad_utils = None
        
        if self.config.vad_type == VADType.SILERO:
            try:
                # Load Silero VAD
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model = model
                self.vad_utils = utils
                logger.info("Silero VAD loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD: {e}")
                if HAS_WEBRTC:
                    self.config.vad_type = VADType.WEBRTC
                    logger.info("Falling back to WebRTC VAD")
                else:
                    self.config.vad_type = VADType.ENERGY
                    logger.info("Falling back to energy-based VAD")
                    
        if self.config.vad_type == VADType.WEBRTC and HAS_WEBRTC:
            self.vad_model = webrtcvad.Vad(self.config.webrtc_aggressiveness)
            logger.info("WebRTC VAD initialized")
            
    def _init_enhancement(self):
        """Initialize audio enhancement components."""
        # High-pass filter
        nyquist = self.config.target_sample_rate / 2
        normalized_cutoff = self.config.highpass_cutoff / nyquist
        self.hp_b, self.hp_a = butter(4, normalized_cutoff, btype='high')
        
        # Denoiser
        self.denoiser = None
        if self.config.enable_denoise and HAS_RNNOISE:
            try:
                self.denoiser = RNNoise(sample_rate=self.config.target_sample_rate)
                logger.info("RNNoise initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RNNoise: {e}")
        
        # AGC state
        self.agc_gain = 1.0
        self.agc_envelope = 0.0
        
    def _init_framing(self):
        """Initialize framing parameters."""
        self.frame_length = int(self.config.frame_length_ms * 
                               self.config.target_sample_rate / 1000)
        self.frame_shift = int(self.config.frame_shift_ms * 
                              self.config.target_sample_rate / 1000)
        self.window = get_window(self.config.window_type, self.frame_length)
        
    def _init_silence_trimmer(self):
        """Initialize silence trimmer."""
        self.silence_frame_length = 2048
        self.silence_hop_length = 512
        
    def resample(self, audio: np.ndarray, 
                 orig_sr: int, 
                 target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples)
    
    def apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter."""
        return filtfilt(self.hp_b, self.hp_a, audio)
    
    def apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter."""
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.config.preemphasis_coef * audio[:-1]
        return emphasized
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise suppression."""
        if self.denoiser is None:
            return audio
        
        try:
            # Process in chunks
            chunk_size = 480  # RNNoise frame size at 48kHz
            if len(audio) <= chunk_size:
                return audio
            
            denoised_chunks = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                speech_prob, denoised = self.denoiser.denoise_frame(chunk)
                denoised_chunks.append(denoised[:len(chunk)])
            
            return np.concatenate(denoised_chunks)[:len(audio)]
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return audio
    
    def apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic gain control."""
        if not self.config.enable_agc:
            return audio
        
        target_level = 10 ** (self.config.agc_target_db / 20)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_gain = target_level / rms
            # Smooth gain change
            self.agc_gain = 0.9 * self.agc_gain + 0.1 * target_gain
            self.agc_gain = np.clip(self.agc_gain, 0.1, 10.0)
        
        return audio * self.agc_gain
    
    def detect_speech(self, audio: np.ndarray, 
                      sample_rate: int) -> float:
        """
        Detect speech probability in audio.
        
        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self.config.vad_type == VADType.SILERO and self.vad_model:
            # Silero VAD
            try:
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio).float()
                
                # Get speech probability
                speech_prob = self.vad_model(audio_tensor, sample_rate).item()
                return speech_prob
            except Exception as e:
                logger.warning(f"Silero VAD failed: {e}")
                
        elif self.config.vad_type == VADType.WEBRTC and self.vad_model:
            # WebRTC VAD
            try:
                # Convert to int16 bytes
                audio_int16 = (audio * 32767).astype(np.int16)
                frame_bytes = audio_int16.tobytes()
                
                # VAD requires specific frame sizes
                frame_duration = int(len(audio) / sample_rate * 1000)
                if frame_duration in [10, 20, 30]:
                    is_speech = self.vad_model.is_speech(frame_bytes, sample_rate)
                    return 1.0 if is_speech else 0.0
            except Exception as e:
                logger.warning(f"WebRTC VAD failed: {e}")
        
        # Energy-based fallback
        rms = np.sqrt(np.mean(audio ** 2))
        return min(rms * 10, 1.0)  # Simple threshold
    
    def trim_silence(self, audio: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Trim leading and trailing silence."""
        # Calculate RMS
        num_frames = 1 + (len(audio) - self.silence_frame_length) // self.silence_hop_length
        rms = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.silence_hop_length
            frame = audio[start:start + self.silence_frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        
        # Reference level
        ref = np.max(rms)
        if ref == 0:
            return audio, (0, len(audio))
        
        # Silence threshold
        rms_db = 20 * np.log10(rms / ref + 1e-10)
        is_silence = rms_db < -self.config.silence_top_db
        
        # Find speech boundaries
        non_silent = np.where(~is_silence)[0]
        if len(non_silent) == 0:
            return np.array([]), (0, 0)
        
        first_frame = non_silent[0]
        last_frame = non_silent[-1]
        
        keep_samples = int(self.config.keep_silence * self.config.target_sample_rate)
        start_sample = max(0, first_frame * self.silence_hop_length - keep_samples)
        end_sample = min(len(audio), (last_frame + 1) * self.silence_hop_length + keep_samples)
        
        return audio[start_sample:end_sample], (start_sample, end_sample)
    
    def frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """Split audio into overlapping frames."""
        num_frames = 1 + (len(audio) - self.frame_length) // self.frame_shift
        
        frames = np.zeros((num_frames, self.frame_length))
        for i in range(num_frames):
            start = i * self.frame_shift
            frame = audio[start:start + self.frame_length]
            
            if len(frame) < self.frame_length:
                frame = np.pad(frame, (0, self.frame_length - len(frame)))
            
            frames[i] = frame * self.window
        
        return frames
    
    def process(self, audio: np.ndarray, 
                sample_rate: int) -> dict:
        """
        Process audio through complete pipeline.
        
        Args:
            audio: Input audio (float32, any sample rate)
            sample_rate: Input sample rate
            
        Returns:
            Dictionary with processed audio and metadata
        """
        result = {
            'original': audio.copy(),
            'sample_rate': sample_rate,
        }
        
        # 1. Resample to target rate
        audio = self.resample(audio, sample_rate, self.config.target_sample_rate)
        result['resampled'] = audio.copy()
        
        # 2. High-pass filter
        audio = self.apply_highpass(audio)
        result['highpassed'] = audio.copy()
        
        # 3. Pre-emphasis
        audio = self.apply_preemphasis(audio)
        result['preemphasized'] = audio.copy()
        
        # 4. Denoise
        audio = self.denoise(audio)
        result['denoised'] = audio.copy()
        
        # 5. AGC
        audio = self.apply_agc(audio)
        result['normalized'] = audio.copy()
        
        # 6. VAD
        speech_prob = self.detect_speech(audio, self.config.target_sample_rate)
        result['speech_probability'] = speech_prob
        result['is_speech'] = speech_prob > self.config.silero_threshold
        
        # 7. Trim silence
        trimmed, (start, end) = self.trim_silence(audio)
        result['trimmed'] = trimmed
        result['trim_indices'] = (start, end)
        
        # 8. Frame
        if len(trimmed) >= self.frame_length:
            frames = self.frame_audio(trimmed)
            result['frames'] = frames
        else:
            result['frames'] = np.array([])
        
        result['final'] = trimmed
        result['target_sample_rate'] = self.config.target_sample_rate
        
        return result
    
    def process_stream(self, audio_chunk: np.ndarray, 
                       sample_rate: int) -> Optional[dict]:
        """
        Process streaming audio chunk.
        
        Returns:
            Result dict when speech segment is complete, None otherwise.
        """
        # Add to buffer
        self.audio_buffer.append(audio_chunk)
        
        # Process accumulated buffer
        audio = np.concatenate(self.audio_buffer)
        
        # Quick VAD check on latest chunk
        speech_prob = self.detect_speech(audio_chunk, sample_rate)
        
        # State machine
        if speech_prob > self.config.silero_threshold:
            if not self.vad_state:
                # Speech started
                self.vad_state = True
                self.speech_start_time = len(audio) / sample_rate
        else:
            if self.vad_state:
                # Speech ended
                silence_duration = (len(audio) / sample_rate) - self.speech_start_time
                if silence_duration > self.config.min_silence_duration_ms / 1000:
                    # Process complete segment
                    self.vad_state = False
                    result = self.process(audio, sample_rate)
                    self.audio_buffer = []
                    return result
        
        # Check buffer size limit
        buffer_duration = len(audio) / sample_rate
        if buffer_duration > 30:  # Max 30 seconds
            result = self.process(audio, sample_rate)
            self.audio_buffer = []
            return result
        
        return None


class RealtimeAudioProcessor:
    """
    Real-time audio processor with callback-based architecture.
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.preprocessor = AudioPreprocessor(self.config)
        self.callbacks: List[Callable] = []
        
    def add_callback(self, callback: Callable):
        """Add callback for processed audio."""
        self.callbacks.append(callback)
        
    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int):
        """Process audio chunk and trigger callbacks."""
        result = self.preprocessor.process_stream(audio_chunk, sample_rate)
        
        if result and result.get('is_speech'):
            for callback in self.callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = AudioConfig(
        capture_sample_rate=48000,
        target_sample_rate=16000,
        enable_denoise=True,
        enable_agc=True,
        vad_type=VADType.SILERO,
        silero_threshold=0.5
    )
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Example: Process audio file
    import soundfile as sf
    
    audio, sr = sf.read("test_audio.wav")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono
    
    result = preprocessor.process(audio, sr)
    
    print(f"Original duration: {len(result['original']) / sr:.2f}s")
    print(f"Speech probability: {result['speech_probability']:.2f}")
    print(f"Is speech: {result['is_speech']}")
    print(f"Trimmed duration: {len(result['trimmed']) / config.target_sample_rate:.2f}s")
    print(f"Number of frames: {len(result['frames'])}")
```

---

## Appendix A: Dependencies

```
# requirements.txt

# Core audio processing
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.10.0
soundfile>=0.12.0

# VAD
webrtcvad-wheels>=2.0.14
torch>=2.0.0
torchaudio>=2.0.0

# Noise suppression
pyrnnoise>=0.2.0

# Windows audio capture
PyAudioWPatch>=0.2.12

# Optional enhancements
noisereduce>=3.0.0
pedalboard>=0.7.0

# Utilities
pydub>=0.25.0
```

## Appendix B: Installation Commands

```bash
# Install core dependencies
pip install numpy scipy librosa soundfile

# Install VAD
pip install webrtcvad-wheels torch torchaudio

# Install noise suppression
pip install pyrnnoise

# Install Windows audio capture
pip install PyAudioWPatch

# Optional
pip install noisereduce pedalboard pydub
```

## Appendix C: Testing Commands

```python
# Test VAD
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"

# Test audio capture
python -m pyaudiowpatch

# Test RNNoise
python -c "from pyrnnoise import RNNoise; RNNoise()"
```

---

**Document End**

*This specification provides a comprehensive architecture for VAD and audio preprocessing in Windows 10 AI agent systems. For questions or updates, refer to the implementation code in Section 13.*
