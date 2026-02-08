# VAD and Audio Preprocessing - Research Summary

## Task Completed: Voice Activity Detection and Audio Preprocessing Architecture

### Generated Deliverable
**Main Document:** `/mnt/okcomputer/output/vad_audio_preprocessing_specification.md`

---

## Key Findings and Recommendations

### 1. Voice Activity Detection (VAD)

**Recommended Strategy: Dual VAD Approach**
- **Primary:** Silero VAD (v4.0+) - 87.7% TPR @ 5% FPR
- **Fallback:** WebRTC VAD - Lightweight, low latency

**Performance Comparison:**
| VAD Engine | TPR @ 5% FPR | Latency | CPU Usage | Best For |
|------------|--------------|---------|-----------|----------|
| Silero VAD | 87.7% | 30-100ms | Medium | Production, noisy environments |
| WebRTC VAD | 50% | 10-30ms | Very Low | Low-resource scenarios |
| Cobra VAD | 98.9% | <30ms | Very Low | Enterprise (commercial) |
| TEN VAD | 95%+ | 30ms | Low | Real-time agents |

### 2. Noise Suppression

**Recommended: RNNoise**
- Deep learning-based RNN architecture
- Real-time capable (< 10ms latency)
- Provides VAD probability as side output
- Open source (BSD license)

**Alternative:** SpeexDSP (traditional, lower CPU usage)

### 3. Echo Cancellation (AEC)

**Recommended: SpeexDSP AEC**
- Filter length: 500ms echo tail
- Latency: ~20ms
- Includes noise suppression preprocessing
- Suitable for real-time systems

**Key Requirement:** Reference buffer for TTS output

### 4. Audio Normalization & Gain Control

**Implementation:**
- **AGC:** Adaptive gain with configurable attack/release times
- **Peak Normalization:** Target -3dB with headroom
- **RMS Normalization:** EBU R128 inspired (-23dB target)

### 5. Audio Segmentation & Framing

**Standard Speech Parameters:**
- Frame length: 25ms (400 samples @ 16kHz)
- Frame shift: 10ms (160 samples @ 16kHz)
- Window: Hamming (optimal for speech)
- Overlap: 60%

### 6. Silence Detection & Trimming

**Parameters:**
- Silence threshold: 40dB below reference
- Min silence to trim: 300ms
- Keep silence around speech: 100ms
- Frame analysis: 2048 samples with 512 hop

### 7. Audio Quality Enhancement

**Processing Chain:**
1. High-pass filter (80Hz cutoff) - remove DC/low-freq noise
2. Pre-emphasis (0.97 coefficient) - boost high frequencies
3. Dynamic range compression - consistent levels

### 8. Complete Preprocessing Pipeline

**Pipeline Stages:**
1. **Capture** (WASAPI) - 48kHz, 16-bit, mono
2. **Resample** - 48kHz → 16kHz
3. **High-pass filter** - 80Hz cutoff
4. **Denoise** - RNNoise
5. **AGC** - Automatic gain control
6. **AEC** - Echo cancellation (if TTS active)
7. **VAD** - Silero/WebRTC detection
8. **Trim silence** - Remove leading/trailing silence
9. **Frame** - 25ms/10ms with Hamming window

**Latency Budget:**
| Stage | Target | Maximum |
|-------|--------|---------|
| Capture | 10ms | 20ms |
| Processing | 60ms | 100ms |
| VAD | 30ms | 50ms |
| **Total** | **100ms** | **170ms** |

### 9. Windows Audio Capture

**Recommended: PyAudioWPatch**
- WASAPI support with loopback recording
- Captures system audio (for TTS monitoring)
- Context manager support
- Pre-compiled wheels for Windows

---

## Python Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.10.0
soundfile>=0.12.0
webrtcvad-wheels>=2.0.14
torch>=2.0.0
pyrnnoise>=0.2.0
PyAudioWPatch>=0.2.12
```

---

## Implementation Highlights

### Complete AudioPreprocessor Class
- 8-stage processing pipeline
- Streaming and batch processing modes
- Configurable via AudioConfig dataclass
- Fallback mechanisms for missing dependencies

### Key Classes:
1. `AudioPreprocessor` - Main processing pipeline
2. `VADHysteresis` - Stable VAD state machine
3. `AutomaticGainControl` - Adaptive gain control
4. `SilenceTrimmer` - Silence detection/removal
5. `AudioFramer` - Overlapping frame extraction
6. `WindowsAudioCapture` - WASAPI audio capture

### Performance Targets:
- **CPU Usage:** 15-20% on modern hardware
- **Memory:** ~70MB (including models)
- **Latency:** < 170ms end-to-end

---

## Files Generated

1. `/mnt/okcomputer/output/vad_audio_preprocessing_specification.md` - Complete technical specification (800+ lines)

---

## Next Steps for Integration

1. Install dependencies: `pip install -r requirements.txt`
2. Test audio capture: `python -m pyaudiowpatch`
3. Verify Silero VAD: `torch.hub.load('snakers4/silero-vad', 'silero_vad')`
4. Integrate AudioPreprocessor class into agent system
5. Configure pipeline parameters for specific use case
6. Add metrics/logging for performance monitoring

---

*Research completed successfully. All VAD, noise suppression, echo cancellation, and preprocessing pipeline specifications documented.*
