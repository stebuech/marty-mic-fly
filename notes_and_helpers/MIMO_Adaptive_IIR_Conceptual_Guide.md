# MIMO Adaptive IIR Notch Filtering for UAV Ego-Noise Suppression
## Theoretical Foundation and Algorithmic Design

**Based on:** Harvey, B. (2019). *Signal Processing Methods for the Detection and Localization of Acoustic Sources via Unmanned Aerial Vehicles.* PhD Thesis, Memorial University of Newfoundland, Chapter 3 (pp. 54-97).

**Application Context:** Hexacopter-mounted 16-microphone array for acoustic measurements
- **Challenge:** Remove 25-30 dB of tonal propeller noise without distorting spatial phase
- **Constraint:** No reference signals available (microphones capture mixed signals)
- **Requirement:** Preserve phase relationships for subsequent beamforming/localization

---

## Table of Contents

1. [The Fundamental Problem](#1-the-fundamental-problem)
2. [Why Adaptive IIR Notch Filters?](#2-why-adaptive-iir-notch-filters)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [The Referenceless Adaptation Principle](#4-the-referenceless-adaptation-principle)
5. [Zero-Phase Filtering for Phase Preservation](#5-zero-phase-filtering-for-phase-preservation)
6. [MIMO Architecture: Managing Multiple Sources](#6-mimo-architecture-managing-multiple-sources)
7. [Frequency Initialization Strategies](#7-frequency-initialization-strategies)
8. [Algorithmic Flow and Processing Pipeline](#8-algorithmic-flow-and-processing-pipeline)
9. [Parameter Selection and Trade-offs](#9-parameter-selection-and-trade-offs)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Design Decisions and Rationale](#11-design-decisions-and-rationale)
12. [Implementation Considerations](#12-implementation-considerations)

---

## 1. The Fundamental Problem

### 1.1 The UAV Audition Challenge

When microphones are mounted on a multicopter drone, they capture a mixture of:

1. **Ego-noise** (unwanted): Tonal components from rotating propellers
   - Blade Passing Frequency (BPF) and harmonics
   - Varies with rotor RPM (non-stationary)
   - Dominates the signal (often 20-40 dB above target)
   
2. **Target signals** (desired): Environmental acoustic sources
   - Can be at any frequency
   - Often weaker than ego-noise
   - Spatial information encoded in phase relationships

### 1.2 What Makes This Difficult?

**Three critical constraints:**

1. **No reference signal**: Unlike Active Noise Control systems, we cannot directly measure the pure ego-noise. The propeller noise and target signals are mixed at the microphones.

2. **Phase preservation requirement**: Microphone arrays use phase differences between channels to determine source direction. Any filtering that distorts phase relationships will destroy localization capability.

3. **Real-time tracking**: Rotor RPM changes continuously during flight, so frequencies shift over time. The system must track these changes adaptively.

### 1.3 Why Not Standard Approaches?

**Traditional band-stop filters:**
- Fixed frequency → cannot track time-varying RPM
- Wide stopbands → remove too much signal
- Non-adaptive → cannot handle dynamic flight conditions

**Wiener filtering:**
- Requires reference signal (not available)
- Assumes stationary statistics (violated during flight)

**Blind source separation (BSS):**
- No guarantee of phase preservation
- High computational cost
- Difficult to ensure complete separation

**Deep learning approaches:**
- Very high computational requirements
- Difficult to guarantee phase linearity
- Training data requirements

---

## 2. Why Adaptive IIR Notch Filters?

### 2.1 The Elegance of Notch Filters

An IIR notch filter places a very narrow "notch" (near-zero transmission) at a specific frequency while leaving all other frequencies essentially unchanged.

**Key advantages:**

1. **Surgical precision**: Can create notches as narrow as 10-20 Hz while sampling at 48 kHz
2. **Minimal computation**: Only ~6 multiplications per sample
3. **Adaptive capability**: Single parameter (notch frequency θ) can be updated in real-time
4. **Perfect DC gain**: No need for normalization

### 2.2 Why IIR Over FIR?

Harvey compared three approaches: IIR notch, FIR notch, and Comb filters.

**IIR Notch Filter (Selected):**
- Notch width: ~15-30 Hz at 200 Hz (r = 0.98-0.99)
- Computation: 6 ops/sample
- Tracking: Excellent (MFD < 0.25 Hz)
- ✅ **Best overall balance**

**FIR Notch Filter:**
- Notch width: ~26-52 Hz at 200 Hz (√3 times wider)
- Computation: 150-200 coefficients × 8 sources × 10 harmonics = impractical
- Tracking: Good but with delay
- Advantage: Inherently stable
- ❌ **Too computationally expensive for MIMO**

**Comb Filter:**
- Notch width: Proportional to frequency (can't adjust independently)
- Computation: Minimal (2-3 ops)
- Tracking: Poor (false convergence with many harmonics)
- ❌ **Unsuitable for high-frequency harmonics**

### 2.3 The Adaptive Principle

The filter adapts using the **Least Mean Squares (LMS)** algorithm, minimizing the energy at the notch frequency. Because propeller noise is much stronger than target signals at harmonic frequencies, minimizing output power effectively removes the ego-noise.

---

## 3. Mathematical Foundation

### 3.1 IIR Notch Filter Transfer Function

The second-order IIR notch filter is defined by:

```
         1 - 2cos(θ)z⁻¹ + z⁻²
H(z) = ─────────────────────────
        1 - 2r·cos(θ)z⁻¹ + r²z⁻²

where:
  θ = notch frequency (radians): θ = 2πf/fs
  r = pole radius (0 < r < 1): controls notch bandwidth
  z = z-transform variable
```

**Understanding the parameters:**

- **θ (theta)**: Determines where the notch is located in frequency
  - This is the parameter we adapt in real-time
  - As rotor RPM changes, θ changes

- **r (pole radius)**: Controls how narrow the notch is
  - r = 0.95: Wide notch (~76 Hz at 200 Hz fundamental)
  - r = 0.98: Medium notch (~30 Hz) ← **typical choice**
  - r = 0.99: Narrow notch (~15 Hz)
  - Trade-off: Narrower is better, but risks instability

**Notch bandwidth (3 dB):**

```
BW ≈ (fs/π) · (1 - r)   for 0.9 < r < 1

Example: fs = 48 kHz, r = 0.98
  BW ≈ (48000/π) · 0.02 ≈ 305 Hz
  
At 200 Hz fundamental, fractional bandwidth:
  BW/f = 305/200 = 1.5 (but this is a general approximation;
                         actual notch width is ~30 Hz at 200 Hz)
```

### 3.2 Time-Domain Difference Equation

The filter operates sample-by-sample:

```
y(n) = x(n) - 2cos(θ)·x(n-1) + x(n-2) 
       + 2r·cos(θ)·y(n-1) - r²·y(n-2)

where:
  x(n) = input sample at time n
  y(n) = output sample at time n
  x(n-1), x(n-2) = previous input samples (delay line)
  y(n-1), y(n-2) = previous output samples (delay line)
```

**Why this form works:**

- The **zeros** (numerator) are placed exactly at e^(±jθ), creating the notch
- The **poles** (denominator) are at r·e^(±jθ), just inside the unit circle
- This creates a very narrow notch at frequency θ with controlled bandwidth

### 3.3 Frequency Response Characteristics

At the notch frequency (ω = θ):
```
|H(e^jθ)| ≈ (1-r)/(1+r)

For r = 0.98: |H| ≈ 0.0101 ≈ -40 dB (theoretical maximum suppression)
For r = 0.99: |H| ≈ 0.0050 ≈ -46 dB
```

Away from the notch (even slightly):
```
|H(e^jω)| ≈ 1.0   for |ω - θ| > BW

The filter is essentially transparent outside the notch region.
```

---

## 4. The Referenceless Adaptation Principle

### 4.1 Why "Referenceless"?

In standard adaptive filtering, you have:
- **Desired signal** d(n): What you want
- **Input signal** x(n): What you have
- **Error** e(n) = d(n) - y(n): What to minimize

But in our case, we don't have a reference! The microphone signal contains both ego-noise and target signals mixed together.

### 4.2 The Key Insight

**Observation**: Propeller harmonics are narrowband and high-power. At the specific frequencies of the harmonics, the ego-noise dominates.

**Strategy**: Minimize the output power of the filtered signal. When the notch aligns with a harmonic, the high-power narrowband component is removed, minimizing total power.

**Cost function:**
```
J(n) = y²(n)

This is different from standard LMS:
  Standard: J(n) = [d(n) - y(n)]²    ← needs reference d(n)
  Ours:     J(n) = y²(n)             ← no reference needed
```

### 4.3 Gradient Descent Adaptation

To minimize J(n), we use gradient descent on the parameter θ:

```
θ(n+1) = θ(n) - μ · ∂J(n)/∂θ(n)

Since J(n) = y²(n):
  ∂J(n)/∂θ(n) = 2y(n) · ∂y(n)/∂θ(n)

Define the gradient signal:
  β(n) = ∂y(n)/∂θ(n)

Then:
  θ(n+1) = θ(n) - 2μ · y(n) · β(n)
```

**This is the LMS update rule adapted for referenceless operation.**

### 4.4 Computing the Gradient Signal β(n)

The gradient signal β(n) measures how the output y(n) changes with respect to θ.

Taking the partial derivative of the difference equation:

```
β(n) = ∂y(n)/∂θ(n) 

     = 2sin(θ)·x(n-1) - 2r·sin(θ)·y(n-1)
       - 2cos(θ)·β(n-1) + β(n-2)
       + 2r·cos(θ)·β(n-1) - r²·β(n-2)

Simplified form:
β(n) = 2sin(θ)·[x(n-1) - r·y(n-1)]
       + 2(r-1)·cos(θ)·β(n-1) + (1-r²)·β(n-2)
```

**Key points:**

1. β(n) is computed recursively (like y(n)), requiring its own delay line
2. Only depends on current θ, not on any reference signal
3. Computational cost: ~4 additional operations per sample

### 4.5 Why This Works: Convergence Analysis

**Intuition**: When θ is close to a harmonic frequency:
- The notch removes most of the high-power harmonic
- Output power y²(n) is low
- Gradient is small → θ stays near the harmonic

**When θ is away from harmonic:**
- Notch doesn't remove the harmonic
- Output power y²(n) is high
- Gradient pushes θ toward the harmonic

**Critical requirement**: The filter must be highly constrained (single parameter θ). If the filter had too much freedom, it might adapt to remove the target signal instead of ego-noise.

### 4.6 Step Size Selection (μ)

The step size μ controls adaptation speed:

**Too small** (μ < 1×10⁻⁵):
- Slow convergence
- Cannot track rapid RPM changes
- Stable but poor performance

**Optimal** (μ ≈ 1×10⁻⁴):
- Good tracking speed
- Stable convergence
- Typical choice from Harvey's experiments

**Too large** (μ > 5×10⁻⁴):
- Fast convergence initially
- Risk of instability
- Noisy frequency estimates

### 4.7 Frequency Smoothing

Because the LMS update is noisy, a moving average smooths the frequency estimate:

```
θ_smooth(n) = (1/W) · Σ[i=n-W+1 to n] θ(i)

where W = window length (typically 50-200 samples)
```

**Trade-off**:
- Larger W: Smoother tracking, but slower response
- Smaller W: Faster response, but more jitter

---

## 5. Zero-Phase Filtering for Phase Preservation

### 5.1 The Phase Distortion Problem

**Standard IIR filters introduce phase shift.**

Even though the notch filter has near-unity magnitude response away from the notch, it still introduces phase distortion across the frequency range:

```
φ(ω) ≠ 0  for standard IIR filter
```

For a microphone array, phase relationships between channels encode spatial information. If different frequency components arrive with different phase shifts, beamforming will fail.

### 5.2 The Zero-Phase Solution

**Two-pass filtering eliminates phase distortion:**

```
Pass 1 (Forward):  y₁(n) = H(z) * x(n)
Time reverse:      y₂(n) = reverse(y₁(n))
Pass 2 (Backward): y₃(n) = H(z) * y₂(n)
Time reverse:      y(n)  = reverse(y₃(n))

Resulting frequency response:
  H_zp(e^jω) = H(e^jω) · H(e^j(-ω)) = |H(e^jω)|²

Phase response:
  φ_zp(ω) = φ(ω) + φ(-ω) = 0  for all ω
```

**Why this works:**

1. Forward pass introduces phase shift φ(ω)
2. Time reversal creates complex conjugate: φ(-ω) = -φ(ω)
3. Backward pass applies phase shift φ(-ω) to already-reversed signal
4. Total phase: φ(ω) + φ(-ω) = 0

**Important consequence**: The magnitude response is squared:

```
|H_zp(e^jω)| = |H(e^jω)|²

In dB: L_zp = 2 · L

If single-pass gives -20 dB suppression, zero-phase gives -40 dB.
This is beneficial for our application!
```

### 5.3 Block-Based Processing

Zero-phase filtering requires access to the entire signal (needs future samples for backward pass). For online processing, we use block-based processing:

**Concept:**
1. Divide signal into overlapping blocks
2. Apply zero-phase filtering to each block
3. Use overlap-add to reconstruct continuous output

**Overlap requirement:**
- Must overlap by several times the filter order to avoid edge artifacts
- Filter order = 2 for IIR notch
- Typical overlap: 8-20 samples (sufficient)

**Block size trade-off:**
- Small blocks: Lower latency, more overhead
- Large blocks: Higher latency, efficient processing
- Typical: 2048-4096 samples at 48 kHz → 40-85 ms latency

### 5.4 Preservation of Spatial Phase

**Critical point**: Zero-phase filtering preserves **temporal** phase (phase vs. frequency), which then preserves **spatial** phase (phase between channels).

For a source at location s received by microphones at positions m₁ and m₂:

```
Time difference: Δt = (|m₁ - s| - |m₂ - s|) / c

Phase difference: Δφ = 2πf·Δt

After zero-phase filtering:
  Both channels experience φ_zp(ω) = 0
  Relative phase difference Δφ is unchanged
  Spatial localization preserved
```

---

## 6. MIMO Architecture: Managing Multiple Sources

### 6.1 The Multi-Source Challenge

A hexacopter has 6 rotors, each producing:
- Fundamental at BPF (varies with RPM)
- Harmonics at 2×BPF, 3×BPF, ..., 10×BPF

Total: 6 sources × 10 harmonics = 60 frequency components to remove

Across 16 microphone channels: 60 × 16 = 960 individual filters needed

### 6.2 The Cascaded Iterative Structure

**Key principle**: Process sources and harmonics sequentially, not in parallel.

```
Source 1, Harmonic 1  →  Source 1, Harmonic 2  → ... → Source 1, Harmonic M
                                                                ↓
Source 2, Harmonic 1  →  Source 2, Harmonic 2  → ... → Source 2, Harmonic M
                                                                ↓
                               ...
                                                                ↓
Source S, Harmonic 1  →  Source S, Harmonic 2  → ... → Source S, Harmonic M
```

**Why cascade?**

Each stage removes one frequency component, producing a "cleaner" signal for the next stage. If processed in parallel, later stages would still see strong interference from earlier frequencies.

### 6.3 Stage Indexing

**Three indices define each filter stage:**

1. **s ∈ [1, S]**: Source index (which rotor)
2. **m ∈ [1, M]**: Harmonic index (which multiple of BPF)
3. **k ∈ [1, K]**: Channel index (which microphone)

**Signal flow:**

```
Input to stage (s, m, k):
  x_{s,m,k}(n) = y_{s,m-1,k}(n)     if m > 1 (use previous harmonic output)
               = y_{s-1,M,k}(n)     if m = 1, s > 1 (use previous source output)
               = x_k(n)              if m = 1, s = 1 (original input)

Output from stage (s, m, k):
  y_{s,m,k}(n) = H_{s,m,k}(z) * x_{s,m,k}(n)
```

### 6.4 The Harmonic Vector

The harmonic vector H defines which multiples of the fundamental to target:

**Standard harmonics:**
```
H = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

Frequencies: f_{s,m} = H[m] × BPF_s
```

**Including sub-harmonics** (for damaged propellers):
```
H = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, ..., 9.5, 10.0]

Harvey's experiments showed damaged propellers produce sub-harmonics.
```

**Custom selection:**
```
H = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]  (skip weak harmonics)
```

**Notch frequency calculation:**
```
For stage (s, m):
  f_{s,m} = H[m] × f_s,fundamental
  θ_{s,m} = 2π · f_{s,m} / fs
```

### 6.5 The Designation Vector

**Problem**: Each microphone channel should primarily track the rotor it's closest to, but all channels must use consistent frequencies for each source.

**Solution**: Designation vector c[k] maps channels to sources.

```
c[k] = source index for channel k

Example for 16-channel dual-ring array, 6 rotors:
  c = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 2, 3, 4]
  
  Outer ring: 2 mics per rotor (adjacent pairs)
  Inner ring: distributed coverage with overlap
```

**Adaptation rule:**

```
For filter stage (s, m, k):
  
  IF c[k] == s:
    // This channel actively tracks this source
    Adapt θ_{s,m,k} using LMS
    θ_{s,m,k}(n+1) = θ_{s,m,k}(n) - 2μ_m · y_{s,m,k}(n) · β_{s,m,k}(n)
  
  ELSE:
    // This channel uses frequency from designated channel
    Find channel c where c[c] == s (the "master" channel for source s)
    Copy frequency: θ_{s,m,k}(n) = θ_{s,m,c}(n)
```

**This ensures:**
1. Each source frequency is tracked by channels with strong signal from that rotor
2. All channels use the **same** frequency for each source → no spatial phase distortion
3. Adaptation is more robust (designated channels have high SNR for their source)

### 6.6 Adaptive Step Size Scheduling

**Problem**: Later stages in the cascade must track frequencies in the presence of remaining tones from earlier stages.

**Solution**: Increase step size progressively through cascade.

```
μ_m = μ_initial + (m - 1) · Δμ

where:
  μ_initial = 1×10⁻⁴ (typical)
  Δμ = 1×10⁻⁴ (typical increment)

Example:
  Harmonic 1: μ₁ = 1×10⁻⁴
  Harmonic 2: μ₂ = 2×10⁻⁴
  Harmonic 3: μ₃ = 3×10⁻⁴
  ...
  Harmonic 10: μ₁₀ = 10×10⁻⁴
```

**Why this works**:

Early stages (low harmonics) see strong, clean signals → can track with small μ

Later stages see residual noise and multiple coexisting tones → need higher μ for reliable tracking

**Risk**: Too high μ can cause instability. Harvey found μ up to 12×10⁻⁴ remained stable in practice.

### 6.7 Coexisting Frequency Challenge

**When rotors operate at similar RPM**, their harmonics can be very close in frequency (e.g., 200.5 Hz and 201.2 Hz).

**Challenge for later stages**:
- Stage for source 2 must track 201.2 Hz
- But 200.5 Hz from source 1 is still present (not yet removed)
- LMS might confuse which frequency to track

**Solutions employed by Harvey**:

1. **Higher μ for later stages**: Faster adaptation locks onto correct frequency
2. **Good initialization**: Starting near correct frequency helps avoid false convergence
3. **Narrower notches** (higher r): Less interference from nearby frequencies
4. **Spatial filtering advantage**: Designation vector ensures each channel tracks its geometrically closest rotor

### 6.8 MIMO Computational Complexity

**Per-sample operations:**

```
Single filter: ~13 FLOPs
Total system: S × M × K × 13 FLOPs

For hexacopter (S=6, M=10, K=16):
  13 × 6 × 10 × 16 = 12,480 FLOPs per sample

At fs = 48 kHz:
  12,480 × 48,000 = ~599 MFLOPS

This is well within capability of modern CPUs (10+ GFLOPS typical).
```

**Memory requirements:**

```
Per filter: ~30 bytes (7 float64 values)
Total: S × M × K × 30 bytes

For hexacopter:
  6 × 10 × 16 × 30 = 28.8 KB (negligible)

Signal buffer dominates memory:
  K channels × N samples × 8 bytes/sample
  16 × 480,000 × 8 = 61 MB for 10 seconds at 48 kHz
```

---

## 7. Frequency Initialization Strategies

### 7.1 Why Initialization Matters

The LMS algorithm is **gradient descent** → it finds local minima.

**Problem**: If θ starts far from the true harmonic frequency, it may:
- Converge very slowly
- Get stuck in local minimum (wrong frequency)
- Never converge at all

**Solution**: Initialize θ close to the actual frequency before adaptation begins.

### 7.2 Method 1: Autocorrelation-Based Detection

**Principle**: Periodic signals have strong autocorrelation at lag corresponding to period.

```
Autocorrelation:
  R_xx[lag] = (1/N) Σ_{n=0}^{N-lag-1} x(n) · x(n+lag)

Peak detection:
  Find lag_max where R_xx is maximum for lag > 0
  
Fundamental frequency:
  f₀ = fs / lag_max
  
BPF (blade passing frequency):
  BPF = n_blades × RPM / 60
```

**Advantages:**
- Works in time domain
- Robust to broadband noise
- Simple computation

**Disadvantages:**
- Requires longer signal segments (several periods)
- Can confuse harmonics with fundamental
- Less precise than frequency-domain methods

### 7.3 Method 2: FFT-Based Peak Detection

**Principle**: Harmonics appear as spectral peaks in the frequency domain.

```
Algorithm:
1. Compute power spectral density: P(f) = |FFT(x)|²
2. Restrict to valid frequency range: [f_min, f_max]
   Based on expected RPM range
3. Find peaks in P(f)
4. Sort peaks by height
5. Select top S peaks as fundamental frequencies
6. Assign to sources based on spatial signature

Initialization:
  For source s, harmonic m:
    θ_{s,m,init} = 2π · H[m] · f_s,fundamental / fs
```

**Advantages:**
- High frequency resolution with zero-padding
- Can identify multiple sources simultaneously
- Works with shorter signal segments

**Disadvantages:**
- Computationally more expensive (FFT)
- Can be confused by broadband noise at similar frequencies
- Requires choosing appropriate FFT length

### 7.4 Method 3: RPM Telemetry (If Available)

**If drone flight controller provides RPM data:**

```
BPF_s = n_blades × RPM_s / 60

For hexacopter with 2-blade propellers:
  BPF = 2 × RPM / 60 = RPM / 30

Example: 6000 RPM → BPF = 200 Hz
```

**Advantages:**
- Most accurate
- No signal processing needed
- Can predict frequencies before measurement

**Disadvantages:**
- Requires telemetry interface
- RPM sensors may have delay/quantization
- Not all flight controllers provide per-motor RPM

**Harvey's approach**: Use acoustic detection (FFT or autocorrelation) as baseline, validate with telemetry if available.

### 7.5 Multi-Source Frequency Separation

**Challenge**: 6 rotors → 6 close frequencies to separate

**Strategy 1: Spatial signatures**
- Sum power across channels weighted by expected spatial pattern
- Each rotor has unique directivity pattern
- Helps disambiguate close frequencies

**Strategy 2: Sequential detection**
1. Find strongest peak → assign to source 1
2. Notch filter at f₁
3. Find next strongest peak → assign to source 2
4. Repeat for all sources

**Strategy 3: Minimum frequency separation**
```
When detecting peaks, enforce:
  |f_s1 - f_s2| > Δf_min

where Δf_min = 20-50 Hz (several times the notch bandwidth)
```

### 7.6 Frequency Range Constraints

**Set bounds based on physical constraints:**

```
RPM_min = 4000  (typical hover minimum)
RPM_max = 8000  (typical maximum for 6" propellers)

BPF_min = n_blades × RPM_min / 60
BPF_max = n_blades × RPM_max / 60

For 2-blade props:
  f_min ≈ 133 Hz
  f_max ≈ 267 Hz

Safety factor:
  f_search_min = 0.8 × BPF_min ≈ 100 Hz
  f_search_max = 1.2 × BPF_max ≈ 320 Hz
```

**This prevents false detection of:**
- Low-frequency structural vibrations
- High-frequency electronics noise
- Harmonics mistaken for fundamentals

---

## 8. Algorithmic Flow and Processing Pipeline

### 8.1 Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RAW MICROPHONE SIGNALS                      │
│                    (K channels × N samples)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FREQUENCY INITIALIZATION                        │
│  - Autocorrelation or FFT peak detection                     │
│  - Identify S fundamental frequencies                        │
│  - Assign to rotors based on spatial/temporal signatures     │
│                                                              │
│  Output: f_s for s ∈ [1, S]                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         INITIALIZE FILTER BANK                              │
│  For each (s, m, k):                                        │
│    θ_{s,m,k} = 2π · H[m] · f_s / fs                        │
│    Initialize delay lines to zero                           │
│    Set r and μ_m parameters                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           MIMO CASCADED FILTERING                           │
│                                                              │
│  FOR s = 1 to S (sources):                                  │
│    FOR m = 1 to M (harmonics):                              │
│      FOR k = 1 to K (channels):                             │
│                                                              │
│        // Determine adaptation                              │
│        IF c[k] == s:                                        │
│          adapt = TRUE                                        │
│        ELSE:                                                │
│          adapt = FALSE                                       │
│          Copy θ from designated channel                      │
│                                                              │
│        // Process sample-by-sample through signal           │
│        FOR n = 1 to N:                                      │
│          y(n) = IIR_filter(x(n), θ, r)                      │
│          IF adapt:                                          │
│            β(n) = compute_gradient(...)                     │
│            θ(n+1) = θ(n) - 2μ_m·y(n)·β(n)                  │
│            θ_smooth = moving_average(θ)                     │
│                                                              │
│        // Output becomes input for next stage               │
│        x_{s,m+1,k} = y_{s,m,k}                             │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            ZERO-PHASE POST-PROCESSING                       │
│  (If offline processing acceptable)                         │
│                                                              │
│  FOR k = 1 to K:                                            │
│    Forward pass:  y1[k] = H(z) * x_filtered[k]             │
│    Reverse:       y2[k] = reverse(y1[k])                    │
│    Backward pass: y3[k] = H(z) * y2[k]                     │
│    Reverse:       y_final[k] = reverse(y3[k])              │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          SPECTRUM RECONSTRUCTION (Optional)                  │
│  - Interpolate across notch frequencies                     │
│  - Or average multiple time segments with varying RPM       │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CLEAN OUTPUT SIGNALS                           │
│  Ready for beamforming, localization, or analysis          │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Detailed Processing Stages

**Stage 1: Data Acquisition**
- Record from all K microphone channels simultaneously
- Maintain synchronization (critical for phase preservation)
- Buffer sufficient data for initialization (2-5 seconds typically)

**Stage 2: Frequency Detection**
- Apply FFT or autocorrelation to each channel
- For FFT: Use Welch method with overlapping windows for noise reduction
- Detect S strongest peaks in valid frequency range
- Validate peaks correspond to expected BPF pattern (harmonically related)

**Stage 3: Filter Initialization**
- Set θ for all (s,m,k) based on detected fundamentals and harmonic vector
- Clear all delay lines (x_delay, y_delay, β_delay = 0)
- Initialize moving average buffers with θ_init

**Stage 4: Iterative Cascaded Filtering**

For each stage (s, m):

1. **Determine master/slave channels**:
   - Master channels: c[k] = s (actively adapt)
   - Slave channels: c[k] ≠ s (copy frequency from master)

2. **Process all K channels in parallel** (can be parallelized):
   - Each channel filters its intermediate signal
   - Master channels adapt θ
   - Slave channels use master's θ

3. **Update for next stage**:
   - Filtered output becomes input for (s, m+1)
   - After last harmonic of source s, output becomes input for source s+1

**Stage 5: Zero-Phase Refinement**
- Reprocess entire signal with fixed θ values from stage 4
- Apply two-pass filtering for true zero phase
- Use block processing with overlap for efficiency

**Stage 6: Quality Validation**
- Check suppression achieved (target: >25 dB)
- Verify frequency tracking accuracy (MFD < 0.5 Hz)
- Validate phase coherence between channels
- If insufficient, adjust parameters and reprocess

### 8.3 Streaming vs. Batch Processing

**Batch Processing (Offline):**
- Process entire recording at once
- Can use zero-phase filtering
- Better frequency initialization (more data)
- Lower computational complexity (amortized initialization)

**Streaming Processing (Real-time):**
- Process blocks as they arrive
- Cannot use zero-phase (requires future samples)
- Must initialize from short data segments
- Higher computational overhead (per-block processing)
- Lower latency (important for real-time applications)

**Hybrid Approach:**
- Initialize frequencies offline from first few seconds
- Then stream process remaining data
- Periodically update frequency estimates
- Balance between latency and quality

---

## 9. Parameter Selection and Trade-offs

### 9.1 Pole Radius (r)

**Effect on notch width:**

| r value | Notch BW @ 200 Hz | Characteristics |
|---------|-------------------|-----------------|
| 0.95    | ~76 Hz            | Very stable, wide notch, more signal loss |
| 0.97    | ~46 Hz            | Good stability, moderate notch |
| **0.98** | **~30 Hz**       | **Balanced (recommended)** |
| 0.985   | ~23 Hz            | Narrow notch, minor instability risk |
| 0.99    | ~15 Hz            | Very narrow, convergence issues possible |

**Selection principle:**

```
IF harmonics well-separated (> 50 Hz apart):
  Use r = 0.985 or 0.99 (narrow notches, minimal signal loss)

ELSE IF harmonics close (< 30 Hz apart):
  Use r = 0.98 (wider notches prevent interaction)

IF stability issues observed:
  Reduce to r = 0.97
```

**Instability indicators:**
- Oscillating frequency estimates
- Diverging output amplitude
- NaN or Inf values in output

### 9.2 Step Size (μ)

**Effect on adaptation:**

| μ value | Convergence Speed | Stability | Noise Rejection |
|---------|-------------------|-----------|-----------------|
| 1×10⁻⁵  | Very slow         | Excellent | Best            |
| **1×10⁻⁴** | **Good**       | **Good**  | **Good** ← recommended |
| 5×10⁻⁴  | Fast              | Marginal  | Poor            |
| 1×10⁻³  | Very fast         | Unstable  | Very poor       |

**Scheduling for MIMO:**

Harvey found that later stages need higher μ to track in presence of coexisting tones:

```
μ_m = μ_base + (m-1) × μ_increment

Typical:
  μ_base = 1×10⁻⁴
  μ_increment = 1×10⁻⁴
  
  Result:
    μ₁ = 1×10⁻⁴  (first harmonic)
    μ₁₀ = 10×10⁻⁴ (tenth harmonic)
```

**Adaptive step size**: Could be adjusted based on convergence monitoring, but Harvey used fixed schedule successfully.

### 9.3 Moving Average Window (W)

**Effect on frequency estimation:**

| Window Size | Response Time | Smoothness | Tracking Lag |
|-------------|---------------|------------|--------------|
| 20-50       | Fast          | Noisy      | Minimal      |
| **50-100**  | **Medium**    | **Good**   | **Acceptable** ← recommended |
| 100-200     | Slow          | Very smooth| Noticeable   |
| 200+        | Very slow     | Excellent  | Significant  |

**Selection principle:**

```
For stable hover:
  W = 100-200 (prioritize smoothness)

For dynamic flight:
  W = 50-100 (prioritize responsiveness)

For post-processing:
  W = 200+ (prioritize accuracy)
```

**Latency contribution:**

```
t_latency = W / fs

Example: W = 100, fs = 48 kHz
  t_latency = 100 / 48000 = 2.1 ms (negligible)
```

### 9.4 Harmonic Vector Selection

**Standard configuration (most common):**
```
H = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Removes: fundamental and first 9 harmonics
```

**For damaged propellers:**
```
H = [0.5, 1, 1.5, 2, 2.5, 3, ..., 9.5, 10]
Includes: sub-harmonics that appear with imbalance
Harvey successfully tested this in experiments
```

**Computational budget constrained:**
```
H = [1, 2, 3, 4, 6, 8, 10]
Skips: weak middle harmonics to reduce stages
Trade-off: Less complete suppression but faster
```

**High-frequency emphasis:**
```
H = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20]
Focus: More high-frequency harmonics where beamforming occurs
```

### 9.5 Designation Vector Design

**Principles:**

1. **Each source needs at least 2 channels** for reliable tracking
2. **Distribute channels based on spatial proximity** to rotors
3. **Balance coverage** across sources

**Example for hexacopter with dual-ring array:**

```
Outer ring (8 mics, 0.40m radius): 
  Positioned at 45° intervals
  Each rotor gets 2 adjacent mics on outer ring
  
Inner ring (8 mics, 0.13m radius):
  Positioned at 45° intervals (offset from outer)
  Distributed for additional coverage

Designation vector (16 channels → 6 sources):
  c = [1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 1,2, 3,4]
  
  Channels 0-1:   Source 1 (rotor at 0°)
  Channels 2-3:   Source 2 (rotor at 60°)
  Channels 4-5:   Source 3 (rotor at 120°)
  Channels 6-7:   Source 4 (rotor at 180°)
  Channels 8-9:   Source 5 (rotor at 240°)
  Channels 10-11: Source 6 (rotor at 300°)
  Channels 12-15: Supplementary coverage
```

**Alternative: Uniform distribution**
```
If rotors have similar RPM, can distribute more evenly:
  c = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4]
  
Each source gets 2-3 channels spread across array
```

### 9.6 Sampling Rate Considerations

**Nyquist requirement for harmonics:**

```
f_max,signal = M × BPF_max

Example: M=10 harmonics, BPF_max = 300 Hz
  f_max,signal = 3000 Hz
  f_Nyquist = 6000 Hz (minimum)
  
Safety factor (2.5×):
  fs_min ≈ 15 kHz
```

**Practical choices:**

| fs (kHz) | Max harmonics @ 200 Hz BPF | Notes |
|----------|----------------------------|-------|
| 16       | 40                         | Minimum viable |
| 24       | 60                         | Good for most applications |
| **48**   | **120**                    | **Standard choice** |
| 96       | 240                        | Overkill for typical case, useful for analysis |

**Trade-offs:**
- Higher fs → more samples to process → higher computational cost
- Higher fs → narrower absolute notch width → better selectivity
- Lower fs → fewer samples → faster processing
- Lower fs → wider absolute notch width → more signal loss

**Downsampling strategy:**

Many measurements start at 96 kHz or higher. Downsample before filtering:

```
1. Design anti-aliasing filter at 0.4 × fs_target
2. Decimate by integer factor (e.g., 96 kHz → 48 kHz, factor = 2)
3. Apply adaptive IIR filtering at lower rate
4. Upsample if needed for final analysis (rare)
```

---

## 10. Performance Characteristics

### 10.1 Harvey's Experimental Results

**Fixed-Wing UAV (Single-Source, SIMO):**

Configuration:
- Delta X-8 fixed-wing UAV
- 4-channel microphone array
- Fundamental: ~150 Hz, 7 harmonics
- 500 Hz target tone from ground speaker

Results:
- IIR: MFD = 0.51 Hz, MSE = 0.43
- Successfully tracked non-stationary engine frequency
- Target tone preserved after filtering

**Multicopter (Multi-Source, MIMO):**

Configuration:
- Kraken octocopter (8 motors)
- 6-channel microphone array
- Multiple sources at ~80-90 Hz fundamental
- 83 Hz target tone
- Damaged propeller → sub-harmonics present

Results:
- IIR with H = [0.5, 1, 1.5, ..., 10]: MFD = 0.16-0.24 Hz across all stages
- Successfully removed sub-harmonics and full harmonics
- Target tone preserved
- Visual inspection: Complete ego-noise removal

### 10.2 Expected Performance for Hexacopter

Based on Harvey's results and system scaling:

**Frequency Tracking:**
```
Mean Frequency Deviation (MFD): < 0.25 Hz
Maximum tracking error: < 0.5 Hz
Convergence time: 0.5-1.0 seconds
Tracking range: ±15% around initialization
```

**Suppression Performance:**
```
Single-pass IIR: 20-25 dB at harmonic frequencies
Zero-phase IIR: 40-50 dB at harmonic frequencies (magnitude squared)
Broadband regions: 0 dB (transparent)

Effective suppression: 25-30 dB RMS across full spectrum
  (considering that not all energy is at harmonics)
```

**Phase Preservation:**
```
Zero-phase filtering: φ(ω) = 0 exactly
Spatial phase distortion: < 0.01 rad (negligible)
Coherence loss between channels: < 1%
```

**Computational Performance:**
```
Modern CPU (10 GFLOPS): Real-time factor > 50×
Embedded system (1 GFLOP): Real-time factor > 5×
GPU acceleration: Real-time factor > 500×

Memory usage: < 100 MB for 10-second recording
```

### 10.3 Comparison with Alternative Methods

**vs. Standard Band-Stop Filters:**
- Fixed filters: No tracking → 0-10 dB effective suppression
- Adaptive IIR: Tracks RPM → 25-30 dB suppression ✓

**vs. FIR Notch Filters:**
- FIR: 150-200 coeff × 60 stages = 9000+ coeff per channel
- IIR: 2 coeff × 60 stages = 120 coeff equivalent
- Speedup: 75× ✓

**vs. Comb Filters:**
- Comb: Wider notches, false convergence with many harmonics
- IIR: Narrow notches, reliable convergence
- IIR superior for > 5 harmonics ✓

**vs. Deep Learning:**
- DL: Requires training data, no phase guarantee, high compute
- IIR: No training, guaranteed zero phase, low compute
- IIR preferred for phase-critical applications ✓

**vs. Sparse Bayesian Learning (SBL) alone:**
- SBL: Good for broadband separation, high compute
- IIR: Excellent for tonal, low compute
- **Optimal: IIR for tonal + SBL for broadband residual** ✓

### 10.4 Failure Modes and Limitations

**When IIR Notch Filtering Fails:**

1. **Very rapid RPM changes** (> 1000 RPM/second):
   - Tracking cannot keep up
   - Solution: Increase μ (at risk of stability)

2. **Extremely close frequencies** (< 10 Hz separation):
   - Notches interfere with each other
   - Solution: Reduce r (wider notches) or use sequential removal

3. **Target signal at harmonic frequency**:
   - Cannot distinguish ego-noise from target
   - Solution: Frequency diversity in target placement

4. **Broadband ego-noise dominates**:
   - Notch filters only remove tonal components
   - Solution: Add SBL stage for broadband suppression

5. **Insufficient SNR at designated channels**:
   - Tracking unreliable
   - Solution: Revise designation vector

**Limitations:**

- **Cannot remove broadband noise**: Only narrow tones
- **Requires stationary-enough frequencies**: Must track over multiple periods
- **No separation of overlapping sources**: If target at harmonic, both removed
- **Initial frequency detection critical**: Poor initialization → poor results

### 10.5 Integration with Sparse Bayesian Learning

**Sequential Processing Approach:**

```
Raw Signal
    ↓
[Adaptive IIR Notch Filtering]  ← Remove tonal components (25-30 dB)
    ↓
Residual (tonal suppressed, broadband remains)
    ↓
[Sparse Bayesian Learning]      ← Remove broadband components (5-10 dB)
    ↓
Clean Signal (30-40 dB total suppression)
```

**Why This Order:**

1. **Tonal first**: IIR is computationally cheap and handles tonal perfectly
2. **Reduces SBL complexity**: Fewer sources for SBL to separate
3. **Leverages strengths**: IIR for narrow tones, SBL for spatial/broadband

**Combined Performance:**

Harvey's IIR alone: 25-30 dB tonal suppression
Expected SBL addition: 5-10 dB broadband suppression
**Total: 30-40 dB ego-noise suppression**

Sufficient for most beamforming applications where target sources are > -10 dB SNR.

---

## 11. Design Decisions and Rationale

### 11.1 Why Referenceless LMS?

**Harvey's key insight**: In acoustic applications, we almost never have access to a reference signal. The ego-noise and target are mixed at the microphones.

**Alternative approaches considered:**

1. **Reference from RPM sensors**:
   - Provides frequency but not amplitude/phase
   - Still requires adaptive amplitude/phase tracking
   - Not sufficient alone

2. **Rotor-mounted microphones as reference**:
   - Extreme flow noise from propeller wash
   - Pseudosound contamination
   - Creates more problems than it solves
   - Harvey specifically rejected this approach

3. **Active noise control (ANC)**:
   - Requires actuators (speakers)
   - Not applicable to passive measurement
   - Wrong problem formulation for our case

**Referenceless LMS elegantly solves this** by exploiting the fact that ego-noise is narrowband and high-power at specific frequencies.

### 11.2 Why Zero-Phase Post-Processing?

**Alternative: Single-pass filtering**
- Pro: Can be done in real-time
- Pro: Lower latency
- Con: Phase distortion destroys beamforming

**For scientific measurement applications**, phase preservation is paramount. Harvey's decision to use zero-phase filtering enables:

1. Accurate beamforming and localization
2. DOA (Direction of Arrival) estimation
3. Acoustic imaging
4. Sound power measurements

**For detection/classification applications** where only amplitude matters, single-pass might suffice.

**Harvey's experiments**: All multicopter experiments used zero-phase processing because the application was acoustic characterization, not real-time detection.

### 11.3 Why Central Array Mounting?

Harvey's literature review revealed debate: forward-mounted vs. centrally-mounted arrays.

**Forward-mounted (some researchers prefer):**
- Greater distance from rotors
- Potentially less ego-noise

**Centrally-mounted (Harvey's choice):**
- Better mechanical stability
- More symmetric spatial coverage
- Easier to implement omnidirectional measurement
- Flight dynamics less affected

**Trade-off accepted**: Slightly higher ego-noise, but compensated by adaptive filtering. The ~25-30 dB suppression from adaptive IIR makes position less critical.

### 11.4 Why Not Rotor-Mounted Reference Microphones?

Several research groups tried placing microphones on rotor arms for reference signals.

**Problems identified:**

1. **Extreme flow effects**: Propeller downwash creates pseudosound (pressure fluctuations sensed as sound)

2. **Vibration coupling**: Direct mechanical transmission to microphone

3. **Reference signal contamination**: Reference mic picks up target signals too (not pure ego-noise)

4. **Complexity**: Additional microphones, cabling, synchronization

Harvey's experimental approach showed that **spatial filtering via designation vector** provides enough "reference information" from geometric separation without these problems.

### 11.5 Why Cascaded Sequential Processing?

**Alternative: Parallel processing** of all sources and harmonics simultaneously.

**Problems with parallel:**

1. **Interference**: All tones present while each filter adapts
2. **Slower convergence**: Must track in high-noise environment
3. **False convergence**: Filters may lock onto wrong frequencies

**Benefits of cascade:**

1. **Sequential removal**: Each stage sees cleaner signal
2. **Faster convergence**: Less interference for later stages
3. **Guaranteed ordering**: Process strongest to weakest components

**Harvey's experiments** confirmed cascaded approach converged faster and more reliably than parallel alternatives.

### 11.6 Why Increasing Step Size Through Cascade?

**Initial hypothesis**: Use same μ for all stages.

**Problem observed**: Later stages (higher harmonics, later sources) struggled to converge because:
- Multiple coexisting tones remain
- Lower SNR for weak harmonics
- Previous stages haven't removed everything

**Solution**: Progressively increase μ through cascade.

**Risk**: Higher μ can cause instability.

**Harvey's finding**: μ up to 12×10⁻⁴ remained stable in practice, and significantly improved convergence of later stages.

This is a **key innovation** in the MIMO structure.

---

## 12. Implementation Considerations

### 12.1 State Management

**Per-Filter State (what must be maintained):**

1. **Current notch frequency**: θ (1 value)
2. **Input delay line**: x(n-1), x(n-2) (2 values)
3. **Output delay line**: y(n-1), y(n-2) (2 values)
4. **Gradient delay line**: β(n-1), β(n-2) (2 values)
5. **Moving average buffer**: θ history (W values)

**Total per filter**: ~30 bytes (7 scalars + circular buffer)

**Total system**: S × M × K × 30 bytes ≈ 29 KB (hexacopter)

**Memory is not a concern** even for embedded systems.

### 12.2 Numerical Precision

**Floating-point considerations:**

- **float32 sufficient for audio**: Dynamic range ~140 dB
- **float64 recommended for θ**: Frequency precision important
  - Δθ = 0.0001 rad at 48 kHz → Δf ≈ 0.76 Hz
  - Good enough for our application

**Potential issues:**

1. **Angle wrapping**: θ should stay in [-π, π]
   - Use atan2 for wrapping
   - Or modulo arithmetic

2. **Moving average stability**: 
   - Circular buffer to avoid drift
   - Periodic re-initialization

3. **Denormal numbers**:
   - Very small β values may denormalize
   - Flush-to-zero safe for this application

### 12.3 Real-Time vs. Offline

**Offline Processing (Recommended for Scientific):**
- Process entire recording at once
- Use zero-phase filtering
- Better frequency initialization
- Can iterate on parameters
- **Harvey's approach for all experiments**

**Real-Time Processing (For Online Applications):**
- Single-pass filtering only
- Block-based processing (buffer management)
- Continuous state maintenance
- Parameter adaptation monitoring

**Hybrid (Best of Both):**
- Record flight data
- Process offline with optimal parameters
- Use results for beamforming/analysis
- Develop real-time version once parameters validated

### 12.4 Parallelization Opportunities

**Channel-level parallelism:**

Within a given stage (s, m), all K channels can be processed in parallel:
- Independent filter operations
- Only synchronization: master channel θ → slave channels
- **Easy to parallelize** (CPU threads or GPU)

**Cannot parallelize:**
- Across stages (s, m): Sequential dependency
- Within channel sample loop: Recursive (IIR) computation

**Expected speedup:**
- 16 channels → 8× speedup on 8-core CPU (good efficiency)
- Much higher on GPU (hundreds of parallel channels possible)

### 12.5 Validation and Monitoring

**During Processing, Monitor:**

1. **Frequency tracking**:
   - θ should converge within ~0.5 seconds
   - Should stay within ±5% of initialized value
   - Check for oscillations (instability indicator)

2. **Output amplitude**:
   - Should decrease after each stage
   - Should not diverge (> 10× input)
   - Check for NaN/Inf

3. **Suppression per stage**:
   - Each stage should reduce power by 1-3 dB at its target frequency
   - Monitor with FFT analysis

4. **Phase coherence** (between channels):
   - Should remain unchanged for frequencies away from notches
   - Measure cross-correlation or coherence function

**Post-Processing Validation:**

1. **Spectrogram inspection**: Visual check for complete harmonic removal
2. **Suppression calculation**: Compare PSD before/after
3. **MFD calculation**: Validate tracking accuracy vs. ground truth (if available)
4. **Beamforming test**: Verify spatial localization still works

### 12.6 Integration Points with Existing Frameworks

**For Acoular Integration:**

The implementation should provide:

1. **TimeSamples-compatible interface**: 
   - `.result(num)` method for sample access
   - `sample_freq`, `numchannels`, `numsamples` properties

2. **HDF5 I/O**:
   - Read from Acoular HDF5 files
   - Write filtered results back to HDF5

3. **Beamforming pipeline compatibility**:
   - Filtered signals → PowerSpectra → BeamformerBase
   - Phase preservation ensures this works

**For Python Ecosystem:**

- NumPy for array operations
- SciPy for signal processing utilities (FFT, etc.)
- Optional: Numba for JIT compilation (10-50× speedup)

**For Real-Time Systems:**

- Consider C/C++ implementation for performance
- Or Cython for Python interface with C-speed core
- FPGA implementation possible for dedicated hardware

### 12.7 Parameter Storage and Reproducibility

**What to Save:**

For reproducibility, store:

1. **Configuration parameters**:
   - S, M, K (system dimensions)
   - r, μ_base, μ_increment
   - W (moving average window)
   - H (harmonic vector)
   - c (designation vector)

2. **Initialization results**:
   - Detected fundamental frequencies
   - Initial θ values

3. **Processing metadata**:
   - Software version
   - Date/time
   - Convergence metrics

4. **Final state** (optional):
   - Final θ values
   - Can be used for continuation

**File Format Recommendation:**
- YAML or JSON for parameters
- HDF5 for numerical data
- Link to Acoular measurement files

---

## Summary: Key Takeaways for Implementation

### Core Principles

1. **Referenceless adaptation** enables ego-noise removal without reference signals
2. **Zero-phase filtering** preserves spatial phase relationships for beamforming
3. **MIMO cascade structure** handles multiple sources and harmonics systematically
4. **Designation vector** ensures frequency consistency across channels

### Critical Success Factors

1. **Good frequency initialization**: Use FFT peak detection or autocorrelation
2. **Appropriate pole radius**: r = 0.98 balances notch width and stability
3. **Adaptive step sizing**: Increase μ through cascade for reliable convergence
4. **Proper designation mapping**: Assign channels to sources based on geometry

### Expected Results

- **Tonal suppression**: 25-30 dB at harmonic frequencies
- **Frequency tracking**: < 0.5 Hz deviation (MFD < 0.25 Hz)
- **Phase preservation**: Negligible distortion (< 0.01 rad)
- **Computational**: Real-time capable on modern CPUs (> 10× RT factor)

### Implementation Strategy

1. **Start simple**: Single source, single harmonic, validate thoroughly
2. **Add complexity incrementally**: Multi-harmonic → Multi-source → MIMO
3. **Validate at each step**: Synthetic signals → Real data → Integration
4. **Optimize after validation**: Numba/C++ after functional correctness confirmed

### Integration with Project

```
Adaptive IIR Notch Filtering (this document)
              ↓
      Tonal components removed (25-30 dB)
              ↓
Sparse Bayesian Learning (AP2 plan)
              ↓
      Broadband components removed (5-10 dB)
              ↓
Total Ego-Noise Suppression: 30-40 dB
              ↓
Ready for Beamforming/Localization/ISO 3744 measurements
```

---

## References

### Primary Source

Harvey, B. (2019). *Signal Processing Methods for the Detection and Localization of Acoustic Sources via Unmanned Aerial Vehicles.* PhD Thesis, Memorial University of Newfoundland.
- **Chapter 3** (pp. 54-97): Adaptive narrowband noise removal methods
- Experiments: Fixed-wing UAV (SIMO), Multicopter (MIMO)
- Results: MFD < 0.25 Hz, successful tonal suppression

### Mathematical Foundations

Tan, L., & Jiang, J. (2015). Simplified Gradient Adaptive Harmonic IIR Notch Filter for Frequency Estimation and Tracking. *American Journal of Signal Processing*, 5(1), 6-12.
- Adaptive IIR notch filter formulation
- LMS gradient derivation

### Related Work

1. **UAV Acoustic Sensing**: Multiple papers on drone audition (Harvey reviews ~30 works)
2. **Beamforming**: Microphone array theory (Acoular documentation)
3. **Adaptive Filtering**: Standard LMS/NLMS algorithms (Widrow, Haykin)
4. **Zero-Phase Filtering**: Forward-backward filtering theory (Gustafsson, etc.)

### Standards

ISO 3744:2010. *Acoustics — Determination of sound power levels of noise sources using sound pressure — Engineering methods for an essentially free field over a reflecting plane.*
- Envelope surface method
- Target application for ego-noise suppressed measurements

---

**Document Version:** 1.0  
**Purpose:** Theoretical foundation for implementation by autonomous coding agent  
**Target System:** 16-channel hexacopter array, 6 sources, 10 harmonics per source  
**Expected Performance:** 25-30 dB tonal suppression, <0.5 Hz tracking accuracy

---

## Appendix: Quick Reference

### Key Equations

**IIR Notch Filter:**
```
H(z) = (1 - 2cos(θ)z⁻¹ + z⁻²) / (1 - 2r·cos(θ)z⁻¹ + r²z⁻²)
y(n) = x(n) - 2cos(θ)·x(n-1) + x(n-2) + 2r·cos(θ)·y(n-1) - r²·y(n-2)
```

**LMS Adaptation:**
```
θ(n+1) = θ(n) - 2μ · y(n) · β(n)
β(n) = 2sin(θ)·[x(n-1) - r·y(n-1)] + 2(r-1)·cos(θ)·β(n-1) + (1-r²)·β(n-2)
```

**Notch Bandwidth:**
```
BW ≈ (fs/π) · (1 - r)
```

**Zero-Phase:**
```
H_zp(e^jω) = |H(e^jω)|²,  φ_zp(ω) = 0
```

### Recommended Parameters

```
r = 0.98
μ_base = 1×10⁻⁴
μ_increment = 1×10⁻⁴
W = 100
H = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### System Dimensions

```
S = 6 sources (rotors)
M = 10 harmonics per source
K = 16 channels (microphones)
Total stages: S × M = 60
Total filters: S × M × K = 960
```

---

*End of Conceptual Guide*
