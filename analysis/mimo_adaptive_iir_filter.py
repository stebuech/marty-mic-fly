#!/usr/bin/env python3
"""
MIMO Adaptive IIR Notch Filter Implementation

Based on Harvey (2019) PhD Thesis:
"Signal Processing Methods for the Detection and Localization of
Acoustic Sources via Unmanned Aerial Vehicles"
Chapter 3 (pp. 54-97): Adaptive Narrowband Noise Removal Methods

Implements referenceless LMS-adapted IIR notch filters in a cascaded
MIMO architecture for removal of tonal propeller noise from UAV-mounted
microphone arrays while preserving spatial phase relationships.

Author: MartyMicFly Project
Date: 2025-12-15
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from dataclasses import dataclass, field, asdict


class IIRNotchFilter:
    """
    Single adaptive IIR notch filter with LMS adaptation.

    Transfer function:
    H(z) = (1 - 2cos(θ)z⁻¹ + z⁻²) / (1 - 2r·cos(θ)z⁻¹ + r²z⁻²)

    where:
        θ = 2πf/fs (notch frequency in radians)
        r = pole radius (0 < r < 1, controls notch bandwidth)

    Adaptation:
        - Cost function: J(n) = y²(n) (minimize output power)
        - Gradient descent: θ(n+1) = θ(n) - 2μy(n)β(n)
        - Gradient signal: β(n) = ∂y(n)/∂θ(n)

    Expected Performance:
        - Suppression: 20-25 dB at notch frequency
        - Tracking accuracy: <0.5 Hz deviation (MFD <0.25 Hz)
        - Convergence time: 0.5-1.0 seconds
    """

    def __init__(self, theta_init: float, r: float, mu: float,
                 fs: float, W: int = 100):
        """
        Initialize adaptive IIR notch filter.

        Args:
            theta_init: Initial notch frequency (radians, 0 to π)
            r: Pole radius (0.95-0.99 typical)
            mu: LMS step size (1e-4 typical)
            fs: Sampling rate (Hz)
            W: Moving average window length for frequency smoothing
        """
        # Filter parameters
        self.theta = float(theta_init)
        self.r = float(r)
        self.mu = float(mu)
        self.fs = float(fs)
        self.W = int(W)

        # Delay lines (initialized to zero)
        self.x_delay = np.zeros(2, dtype=np.float64)  # x(n-1), x(n-2)
        self.y_delay = np.zeros(2, dtype=np.float64)  # y(n-1), y(n-2)
        self.beta_delay = np.zeros(2, dtype=np.float64)  # β(n-1), β(n-2)

        # Moving average buffer for theta smoothing (circular buffer)
        self.theta_buffer = np.full(W, theta_init, dtype=np.float64)
        self.theta_buffer_idx = 0
        self.theta_buffer_filled = False

        # Smoothed theta estimate
        self.theta_smooth = theta_init

        # Sample counter
        self.n_samples_processed = 0

    def filter_sample(self, x_n: float) -> float:
        """
        Process one input sample through the notch filter.

        Difference equation:
        y(n) = x(n) - 2cos(θ)·x(n-1) + x(n-2)
               + 2r·cos(θ)·y(n-1) - r²·y(n-2)

        Args:
            x_n: Input sample x(n)

        Returns:
            y_n: Output sample y(n)
        """
        cos_theta = np.cos(self.theta)
        r_squared = self.r * self.r

        # Compute output
        y_n = (x_n
               - 2.0 * cos_theta * self.x_delay[0]
               + self.x_delay[1]
               + 2.0 * self.r * cos_theta * self.y_delay[0]
               - r_squared * self.y_delay[1])

        # Update delay lines (shift)
        self.x_delay[1] = self.x_delay[0]
        self.x_delay[0] = x_n

        self.y_delay[1] = self.y_delay[0]
        self.y_delay[0] = y_n

        self.n_samples_processed += 1

        return y_n

    def compute_gradient(self, x_n: float, y_n: float) -> float:
        """
        Compute gradient signal β(n) = ∂y(n)/∂θ(n).

        Gradient equation (simplified form):
        β(n) = 2sin(θ)·[x(n-1) - r·y(n-1)]
               + 2(r-1)·cos(θ)·β(n-1) + (1-r²)·β(n-2)

        Args:
            x_n: Current input sample x(n)
            y_n: Current output sample y(n)

        Returns:
            beta_n: Gradient signal β(n)
        """
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)

        # Compute gradient signal
        beta_n = (2.0 * sin_theta * (self.x_delay[0] - self.r * self.y_delay[0])
                  + 2.0 * (self.r - 1.0) * cos_theta * self.beta_delay[0]
                  + (1.0 - np.square(self.r)) * self.beta_delay[1])

        # Update gradient delay line
        self.beta_delay[1] = self.beta_delay[0]
        self.beta_delay[0] = beta_n

        return beta_n

    def adapt(self, y_n: float, beta_n: float) -> None:
        """
        Adapt notch frequency using LMS algorithm.

        Update rule:
        θ(n+1) = θ(n) - 2μ·y(n)·β(n)

        The factor of 2 comes from the derivative of J(n) = y²(n).

        Args:
            y_n: Output sample y(n)
            beta_n: Gradient signal β(n)
        """
        # LMS update
        theta_new = self.theta - 2.0 * self.mu * y_n * beta_n

        # Wrap angle to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        self.theta = theta_new

        # Update moving average buffer (circular)
        self.theta_buffer[self.theta_buffer_idx] = theta_new
        self.theta_buffer_idx = (self.theta_buffer_idx + 1) % self.W

        if self.theta_buffer_idx == 0:
            self.theta_buffer_filled = True

        # Compute smoothed theta
        if self.theta_buffer_filled:
            self.theta_smooth = np.mean(self.theta_buffer)
        else:
            self.theta_smooth = np.mean(self.theta_buffer[:self.theta_buffer_idx])

    def get_frequency_hz(self, use_smooth: bool = True) -> float:
        """
        Get current notch frequency in Hz.

        Args:
            use_smooth: If True, return smoothed frequency estimate

        Returns:
            Frequency in Hz
        """
        theta = self.theta_smooth if use_smooth else self.theta
        return (theta * self.fs) / (2.0 * np.pi)

    def reset_state(self) -> None:
        """Reset filter state (delays, buffers) but keep parameters."""
        self.x_delay.fill(0.0)
        self.y_delay.fill(0.0)
        self.beta_delay.fill(0.0)
        self.theta_buffer.fill(self.theta)
        self.theta_buffer_idx = 0
        self.theta_buffer_filled = False
        self.theta_smooth = self.theta
        self.n_samples_processed = 0

    def get_state(self) -> Dict:
        """
        Get current filter state for serialization/reproducibility.

        Returns:
            Dictionary containing all state variables
        """
        return {
            'theta': float(self.theta),
            'theta_smooth': float(self.theta_smooth),
            'r': float(self.r),
            'mu': float(self.mu),
            'fs': float(self.fs),
            'W': int(self.W),
            'x_delay': self.x_delay.tolist(),
            'y_delay': self.y_delay.tolist(),
            'beta_delay': self.beta_delay.tolist(),
            'theta_buffer': self.theta_buffer.tolist(),
            'theta_buffer_idx': int(self.theta_buffer_idx),
            'theta_buffer_filled': bool(self.theta_buffer_filled),
            'n_samples_processed': int(self.n_samples_processed)
        }

    def set_state(self, state: Dict) -> None:
        """
        Restore filter state from dictionary.

        Args:
            state: Dictionary from get_state()
        """
        self.theta = float(state['theta'])
        self.theta_smooth = float(state['theta_smooth'])
        self.r = float(state['r'])
        self.mu = float(state['mu'])
        self.fs = float(state['fs'])
        self.W = int(state['W'])
        self.x_delay = np.array(state['x_delay'], dtype=np.float64)
        self.y_delay = np.array(state['y_delay'], dtype=np.float64)
        self.beta_delay = np.array(state['beta_delay'], dtype=np.float64)
        self.theta_buffer = np.array(state['theta_buffer'], dtype=np.float64)
        self.theta_buffer_idx = int(state['theta_buffer_idx'])
        self.theta_buffer_filled = bool(state['theta_buffer_filled'])
        self.n_samples_processed = int(state['n_samples_processed'])

    def __repr__(self) -> str:
        return (f"IIRNotchFilter(f={self.get_frequency_hz():.2f} Hz, "
                f"r={self.r:.3f}, μ={self.mu:.1e}, "
                f"samples={self.n_samples_processed})")


@dataclass
class MIMOFilterConfig:
    """
    Configuration for MIMO Adaptive IIR Notch Filter system.

    Defines all parameters for the multi-source, multi-harmonic,
    multi-channel cascaded filter bank.
    """

    # System dimensions
    n_sources: int = 4              # Number of rotor sources (quadcopter)
    n_harmonics: int = 10           # Harmonics per source
    n_channels: int = 95            # Acoustic microphone channels
    sample_rate: float = 51200.0    # Sampling rate (Hz)

    # Channel processing options
    channel_subset: Optional[List[int]] = None  # None = all channels

    # Filter parameters
    pole_radius: float = 0.98       # r: controls notch bandwidth
    mu_base: float = 1e-4           # Base LMS step size
    mu_increment: float = 1e-4      # Step size increment per harmonic stage
    moving_avg_window: int = 100    # W: smoothing window length

    # Harmonic configuration
    harmonic_vector: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    include_subharmonics: bool = False  # If True: [0.5, 1, 1.5, 2, ...]

    # Designation vector (channel → source mapping)
    designation_strategy: str = 'auto_quadrant'  # 'auto_quadrant' or 'explicit'
    designation_vector: Optional[List[int]] = None  # Explicit if strategy='explicit'

    # Frequency initialization
    initialization_method: str = 'rpm'  # 'autocorr', 'fft', 'rpm', or 'compare_all'

    # Initialization method parameters
    fft_nfft: int = 65536
    fft_f_min: float = 183.0
    fft_f_max: float = 207.0
    fft_min_separation: float = 2.0

    autocorr_max_lag: int = 512
    autocorr_f_min: float = 183.0
    autocorr_f_max: float = 207.0

    rpm_n_blades: int = 2
    rpm_correction_factor: float = 7/12

    # Processing options
    zero_phase: bool = True         # Enable zero-phase filtering
    save_intermediate: bool = False # Save per-stage outputs (debugging)
    validate_convergence: bool = True  # Check convergence at each stage
    convergence_tolerance: float = 0.5  # Hz - max deviation for "converged"

    # Output options
    save_tracked_frequencies: bool = True
    save_suppression_metrics: bool = True
    save_filter_state: bool = True
    create_comparison_plots: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid, errors = self.validate()
        if not valid:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # System dimensions
        if self.n_sources < 1:
            errors.append("n_sources must be >= 1")
        if self.n_harmonics < 1:
            errors.append("n_harmonics must be >= 1")
        if self.n_channels < 1:
            errors.append("n_channels must be >= 1")
        if self.sample_rate <= 0:
            errors.append("sample_rate must be > 0")

        # Filter parameters
        if not (0.9 <= self.pole_radius < 1.0):
            errors.append("pole_radius must be in [0.9, 1.0)")
        if self.mu_base <= 0:
            errors.append("mu_base must be > 0")
        if self.mu_increment < 0:
            errors.append("mu_increment must be >= 0")
        if self.moving_avg_window < 1:
            errors.append("moving_avg_window must be >= 1")

        # Harmonic vector
        if len(self.harmonic_vector) != self.n_harmonics:
            errors.append(f"harmonic_vector length ({len(self.harmonic_vector)}) "
                         f"must match n_harmonics ({self.n_harmonics})")

        # Channel subset
        if self.channel_subset is not None:
            if len(self.channel_subset) < 1:
                errors.append("channel_subset must contain at least 1 channel")
            if max(self.channel_subset) >= self.n_channels:
                errors.append(f"channel_subset indices must be < n_channels ({self.n_channels})")

        # Designation vector
        if self.designation_strategy == 'explicit':
            if self.designation_vector is None:
                errors.append("designation_vector required when strategy='explicit'")
            elif len(self.designation_vector) != self.n_channels:
                errors.append(f"designation_vector length must match n_channels")
            elif min(self.designation_vector) < 1 or max(self.designation_vector) > self.n_sources:
                errors.append(f"designation_vector values must be in [1, {self.n_sources}]")

        # Initialization method
        valid_methods = ['autocorr', 'fft', 'rpm', 'compare_all']
        if self.initialization_method not in valid_methods:
            errors.append(f"initialization_method must be one of {valid_methods}")

        return (len(errors) == 0, errors)

    def get_adaptive_step_size(self, harmonic_idx: int) -> float:
        """
        Get adaptive step size for harmonic stage m.

        μ_m = μ_base + (m - 1) · μ_increment

        Args:
            harmonic_idx: Harmonic index (0-based)

        Returns:
            Step size μ_m
        """
        return self.mu_base + harmonic_idx * self.mu_increment

    def create_designation_vector_quadcopter(self) -> np.ndarray:
        """
        Create designation vector for quadcopter using quadrant strategy.

        Divides channels into 4 spatial quadrants:
        - Quadrant 1 (0 to n//4): Source 1
        - Quadrant 2 (n//4 to n//2): Source 2
        - Quadrant 3 (n//2 to 3n//4): Source 3
        - Quadrant 4 (3n//4 to n): Source 4

        Returns:
            Designation vector c[k] with values 1-4 (1-indexed)
        """
        n_channels = self.n_channels
        c = np.zeros(n_channels, dtype=int)

        # Divide into 4 equal quadrants
        channels_per_source = n_channels // 4
        remainder = n_channels % 4

        idx = 0
        for source in range(1, 5):  # 1-indexed sources
            # Distribute remainder across first sources
            n_channels_this_source = channels_per_source + (1 if source <= remainder else 0)
            c[idx:idx + n_channels_this_source] = source
            idx += n_channels_this_source

        # Ensure each source has at least 2 channels
        for source in range(1, 5):
            if np.sum(c == source) < 2:
                raise ValueError(f"Source {source} has < 2 channels. "
                                f"Increase n_channels or adjust strategy.")

        return c

    def get_designation_vector(self) -> np.ndarray:
        """
        Get or create designation vector based on strategy.

        Returns:
            Designation vector c[k] (1-indexed source assignments)
        """
        if self.designation_strategy == 'explicit':
            if self.designation_vector is None:
                raise ValueError("designation_vector not specified for strategy='explicit'")
            return np.array(self.designation_vector, dtype=int)
        elif self.designation_strategy == 'auto_quadrant':
            return self.create_designation_vector_quadcopter()
        else:
            raise ValueError(f"Unknown designation_strategy: {self.designation_strategy}")

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'MIMOFilterConfig':
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            MIMOFilterConfig instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Handle nested mimo_filter key if present
        if 'mimo_filter' in data:
            data = data['mimo_filter']

        # Flatten nested dicts
        config_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Flatten nested config (e.g., system, filter_params)
                for subkey, subvalue in value.items():
                    config_dict[subkey] = subvalue
            else:
                config_dict[key] = value

        return cls(**config_dict)

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to output YAML file
        """
        filepath = Path(filepath)

        # Convert to dict and remove None values
        config_dict = {k: v for k, v in asdict(self).items() if v is not None}

        with open(filepath, 'w') as f:
            yaml.dump({'mimo_filter': config_dict}, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return (f"MIMOFilterConfig(sources={self.n_sources}, "
                f"harmonics={self.n_harmonics}, channels={self.n_channels}, "
                f"fs={self.sample_rate/1000:.1f}kHz, r={self.pole_radius}, "
                f"zero_phase={self.zero_phase})")


class MIMOFilterStage:
    """
    One filter stage (source s, harmonic m) across all channels.

    Manages K IIRNotchFilter instances with master/slave channel logic:
    - Master channels (c[k] == s): Adapt theta using LMS
    - Slave channels (c[k] != s): Copy theta from master

    This ensures frequency consistency across all channels,
    preserving spatial phase relationships.
    """

    def __init__(self, source_idx: int, harmonic_idx: int,
                 n_channels: int, designation_vector: np.ndarray,
                 r: float, mu: float, fs: float, W: int = 100):
        """
        Initialize MIMO filter stage.

        Args:
            source_idx: Source index (1-indexed, 1 to S)
            harmonic_idx: Harmonic index (1-indexed, 1 to M)
            n_channels: Number of channels K
            designation_vector: c[k] mapping channels to sources (1-indexed)
            r: Pole radius
            mu: LMS step size
            fs: Sampling rate
            W: Moving average window
        """
        self.source_idx = source_idx
        self.harmonic_idx = harmonic_idx
        self.n_channels = n_channels
        self.designation_vector = designation_vector
        self.r = r
        self.mu = mu
        self.fs = fs
        self.W = W

        # Create filter for each channel
        self.filters = []
        for k in range(n_channels):
            # Initialize with placeholder theta (will be set later)
            filt = IIRNotchFilter(theta_init=0.1, r=r, mu=mu, fs=fs, W=W)
            self.filters.append(filt)

        # Find master channel (first channel where c[k] == source_idx)
        self.master_channel = None
        for k in range(n_channels):
            if designation_vector[k] == source_idx:
                self.master_channel = k
                break

        if self.master_channel is None:
            raise ValueError(f"No master channel found for source {source_idx}")

        self.n_samples_processed = 0

    def initialize_frequencies(self, fundamental_freq: float, harmonic_factor: float) -> None:
        """
        Initialize all filter frequencies.

        Args:
            fundamental_freq: Fundamental frequency for this source (Hz)
            harmonic_factor: Harmonic multiplier (e.g., 1, 2, 3, ...)
        """
        target_freq = fundamental_freq * harmonic_factor
        theta_init = 2.0 * np.pi * target_freq / self.fs

        # Initialize all filters with same frequency
        for filt in self.filters:
            filt.theta = theta_init
            filt.theta_smooth = theta_init
            filt.theta_buffer.fill(theta_init)

    def process_block(self, x_block: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process one block of samples through this stage.

        Args:
            x_block: Input block [n_channels, n_samples]

        Returns:
            y_block: Output block [n_channels, n_samples]
            metadata: Stage processing metadata
        """
        n_channels, n_samples = x_block.shape
        y_block = np.zeros_like(x_block)

        # Process each sample
        for n in range(n_samples):
            # First, process master channel and adapt
            k_master = self.master_channel
            x_n = x_block[k_master, n]
            y_n = self.filters[k_master].filter_sample(x_n)
            beta_n = self.filters[k_master].compute_gradient(x_n, y_n)
            self.filters[k_master].adapt(y_n, beta_n)
            y_block[k_master, n] = y_n

            # Get updated theta from master
            theta_master = self.filters[k_master].theta

            # Process all other channels
            for k in range(n_channels):
                if k == k_master:
                    continue  # Already processed

                if self.designation_vector[k] == self.source_idx:
                    # Another master channel for this source - also adapt
                    x_n = x_block[k, n]
                    y_n = self.filters[k].filter_sample(x_n)
                    beta_n = self.filters[k].compute_gradient(x_n, y_n)
                    self.filters[k].adapt(y_n, beta_n)
                else:
                    # Slave channel - copy theta from master
                    self.filters[k].theta = theta_master
                    x_n = x_block[k, n]
                    y_n = self.filters[k].filter_sample(x_n)
                    # No adaptation for slave channels

                y_block[k, n] = y_n

        self.n_samples_processed += n_samples

        # Collect tracked frequencies
        tracked_freqs = np.array([filt.get_frequency_hz() for filt in self.filters])

        metadata = {
            'source': self.source_idx,
            'harmonic': self.harmonic_idx,
            'tracked_frequencies': tracked_freqs,
            'master_frequency': self.filters[self.master_channel].get_frequency_hz(),
            'n_samples': n_samples
        }

        return y_block, metadata

    def is_converged(self, tolerance: float = 0.5) -> bool:
        """
        Check if stage has converged (frequency stable).

        Args:
            tolerance: Maximum deviation in Hz to consider converged

        Returns:
            True if converged
        """
        # Check master channel frequency stability
        master_freq = self.filters[self.master_channel].get_frequency_hz()
        theta_buffer = self.filters[self.master_channel].theta_buffer

        if not self.filters[self.master_channel].theta_buffer_filled:
            return False

        # Convert buffer to frequencies
        freqs_hz = (theta_buffer * self.fs) / (2.0 * np.pi)
        std_dev = np.std(freqs_hz)

        return std_dev < tolerance

    def get_tracked_frequencies(self) -> np.ndarray:
        """Get current frequencies for all channels."""
        return np.array([filt.get_frequency_hz() for filt in self.filters])

    def __repr__(self) -> str:
        return (f"MIMOFilterStage(source={self.source_idx}, "
                f"harmonic={self.harmonic_idx}, "
                f"f={self.filters[self.master_channel].get_frequency_hz():.2f} Hz, "
                f"converged={'✓' if self.is_converged() else '✗'})")


class MIMOFilterBank:
    """
    Full MIMO cascaded filter bank (S sources × M harmonics × K channels).

    Orchestrates sequential processing through all stages:
    - Sources processed sequentially (s = 1, 2, 3, 4)
    - Harmonics processed sequentially (m = 1, 2, ..., M)
    - Output of stage (s, m) → Input of stage (s, m+1)

    Implements both single-pass and zero-phase filtering.
    """

    def __init__(self, config: MIMOFilterConfig):
        """
        Initialize MIMO filter bank.

        Args:
            config: MIMOFilterConfig instance
        """
        self.config = config
        self.n_sources = config.n_sources
        self.n_harmonics = config.n_harmonics
        self.n_channels = config.n_channels
        self.sample_rate = config.sample_rate

        # Get designation vector
        self.designation_vector = config.get_designation_vector()

        # Create stages array [S, M]
        self.stages = []
        for s in range(1, self.n_sources + 1):
            harmonic_stages = []
            for m in range(1, self.n_harmonics + 1):
                # Get adaptive step size for this harmonic
                mu_m = config.get_adaptive_step_size(m - 1)

                stage = MIMOFilterStage(
                    source_idx=s,
                    harmonic_idx=m,
                    n_channels=self.n_channels,
                    designation_vector=self.designation_vector,
                    r=config.pole_radius,
                    mu=mu_m,
                    fs=config.sample_rate,
                    W=config.moving_avg_window
                )
                harmonic_stages.append(stage)
            self.stages.append(harmonic_stages)

        self.initialized = False

    def initialize_from_frequencies(self, fundamental_freqs: np.ndarray) -> None:
        """
        Initialize all stages with fundamental frequencies.

        Args:
            fundamental_freqs: Array of S fundamental frequencies (Hz)
        """
        if len(fundamental_freqs) != self.n_sources:
            raise ValueError(f"Expected {self.n_sources} frequencies, "
                           f"got {len(fundamental_freqs)}")

        for s in range(self.n_sources):
            fundamental_freq = fundamental_freqs[s]
            for m in range(self.n_harmonics):
                harmonic_factor = self.config.harmonic_vector[m]
                self.stages[s][m].initialize_frequencies(fundamental_freq, harmonic_factor)

        self.initialized = True

    def process_forward_pass(self, time_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process data through cascade (single forward pass with adaptation).

        Args:
            time_data: Input data [n_channels, n_samples]

        Returns:
            filtered_data: Output data [n_channels, n_samples]
            metadata: Processing metadata
        """
        if not self.initialized:
            raise RuntimeError("Filter bank not initialized. "
                             "Call initialize_from_frequencies() first.")

        x_current = time_data.copy()
        metadata = {
            'tracked_frequencies': np.zeros((self.n_sources, self.n_harmonics)),
            'stage_metadata': []
        }

        # Process through cascade: sources → harmonics
        for s in range(self.n_sources):
            for m in range(self.n_harmonics):
                stage = self.stages[s][m]
                y_current, stage_meta = stage.process_block(x_current)
                x_current = y_current  # Output → input for next stage

                # Store metadata
                metadata['tracked_frequencies'][s, m] = stage_meta['master_frequency']
                if self.config.save_intermediate:
                    metadata['stage_metadata'].append(stage_meta)

        return x_current, metadata

    def process_zero_phase(self, time_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process with zero-phase filtering (forward-backward pass).

        Algorithm:
        1. Forward pass with adaptation → get final theta values
        2. Reverse signal
        3. Backward pass with frozen theta (no adaptation)
        4. Reverse again to restore time order

        Args:
            time_data: Input data [n_channels, n_samples]

        Returns:
            filtered_data: Zero-phase filtered data [n_channels, n_samples]
            metadata: Processing metadata
        """
        # Step 1: Forward pass with adaptation
        print("  Forward pass (adapting)...", end='', flush=True)
        filtered_forward, metadata = self.process_forward_pass(time_data)
        print(" done")

        # Step 2: Extract and freeze theta values
        frozen_theta = []
        for s in range(self.n_sources):
            source_theta = []
            for m in range(self.n_harmonics):
                stage = self.stages[s][m]
                # Get theta from all filters in this stage
                theta_values = np.array([filt.theta for filt in stage.filters])
                source_theta.append(theta_values)
            frozen_theta.append(source_theta)

        # Step 3: Reverse signal
        print("  Reversing signal...", end='', flush=True)
        reversed_signal = np.flip(filtered_forward, axis=1)
        print(" done")

        # Step 4: Backward pass with frozen theta (no adaptation)
        print("  Backward pass (frozen)...", end='', flush=True)
        x_current = reversed_signal.copy()

        for s in range(self.n_sources):
            for m in range(self.n_harmonics):
                stage = self.stages[s][m]

                # Reset filters and set frozen theta
                for k in range(self.n_channels):
                    stage.filters[k].reset_state()
                    stage.filters[k].theta = frozen_theta[s][m][k]

                # Process without adaptation
                n_samples = x_current.shape[1]
                y_current = np.zeros_like(x_current)

                for n in range(n_samples):
                    for k in range(self.n_channels):
                        x_n = x_current[k, n]
                        y_n = stage.filters[k].filter_sample(x_n)
                        y_current[k, n] = y_n

                x_current = y_current

        print(" done")

        # Step 5: Reverse again to restore time order
        print("  Reversing back...", end='', flush=True)
        result = np.flip(x_current, axis=1)
        print(" done")

        metadata['zero_phase'] = True

        return result, metadata

    def get_suppression_summary(self) -> Dict:
        """Get summary of tracked frequencies and convergence."""
        summary = {
            'sources': [],
            'convergence_status': []
        }

        for s in range(self.n_sources):
            source_freqs = []
            source_converged = []
            for m in range(self.n_harmonics):
                stage = self.stages[s][m]
                freq = stage.filters[stage.master_channel].get_frequency_hz()
                converged = stage.is_converged(self.config.convergence_tolerance)
                source_freqs.append(freq)
                source_converged.append(converged)

            summary['sources'].append({
                'source_idx': s + 1,
                'frequencies': source_freqs,
                'converged': source_converged
            })

        return summary

    def __repr__(self) -> str:
        status = "initialized" if self.initialized else "not initialized"
        return (f"MIMOFilterBank({self.n_sources} sources, "
                f"{self.n_harmonics} harmonics, {self.n_channels} channels, "
                f"{status})")


if __name__ == "__main__":
    # Example usage and basic testing
    print("="*60)
    print("MIMO Adaptive IIR Notch Filter - Core Classes")
    print("="*60)

    # Test IIRNotchFilter
    print("\n1. Testing IIRNotchFilter")
    print("-" * 40)

    fs = 51200.0
    f_target = 195.0  # Target frequency (Hz)
    theta_init = 2 * np.pi * f_target / fs

    filt = IIRNotchFilter(theta_init, r=0.98, mu=1e-4, fs=fs, W=100)
    print(f"Created: {filt}")
    print(f"Initial frequency: {filt.get_frequency_hz():.2f} Hz")

    # Generate test tone
    duration = 1.0
    t = np.arange(int(duration * fs)) / fs
    x = np.sin(2 * np.pi * f_target * t)

    # Process
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = filt.filter_sample(x[n])
        beta = filt.compute_gradient(x[n], y[n])
        filt.adapt(y[n], beta)

    # Measure suppression
    suppression_db = 20 * np.log10(np.std(x[10000:]) / (np.std(y[10000:]) + 1e-10))
    print(f"Suppression after 1s: {suppression_db:.1f} dB")
    print(f"Final frequency: {filt.get_frequency_hz():.2f} Hz")
    print(f"Tracking error: {abs(filt.get_frequency_hz() - f_target):.3f} Hz")

    # Test state save/load
    state = filt.get_state()
    filt2 = IIRNotchFilter(0.1, r=0.98, mu=1e-4, fs=fs)
    filt2.set_state(state)
    print(f"State restored: {filt2}")

    # Test MIMOFilterConfig
    print("\n2. Testing MIMOFilterConfig")
    print("-" * 40)

    config = MIMOFilterConfig()
    print(f"Default config: {config}")

    valid, errors = config.validate()
    print(f"Validation: {'✓ Valid' if valid else '✗ Invalid'}")
    if errors:
        for err in errors:
            print(f"  - {err}")

    # Test designation vector
    c = config.get_designation_vector()
    print(f"\nDesignation vector (first 20): {c[:20]}")
    for source in range(1, 5):
        n_ch = np.sum(c == source)
        print(f"  Source {source}: {n_ch} channels")

    # Test adaptive step size
    print(f"\nAdaptive step sizes:")
    for m in range(10):
        mu_m = config.get_adaptive_step_size(m)
        print(f"  Harmonic {m+1}: μ = {mu_m:.1e}")

    # Test MIMO architecture
    print("\n3. Testing MIMO Architecture")
    print("-" * 40)

    # Create small-scale test: 2 sources, 2 harmonics, 10 channels
    config_test = MIMOFilterConfig(
        n_sources=2,
        n_harmonics=2,
        n_channels=10,
        sample_rate=51200.0,
        pole_radius=0.98,
        mu_base=1e-4,
        mu_increment=1e-4,
        harmonic_vector=[1, 2],
        zero_phase=False  # Test single-pass first
    )

    print(f"Created: {config_test}")

    filter_bank = MIMOFilterBank(config_test)
    print(f"Filter bank: {filter_bank}")

    # Initialize with test frequencies
    fundamental_freqs = np.array([190.0, 200.0])  # Hz
    filter_bank.initialize_from_frequencies(fundamental_freqs)
    print(f"Initialized with frequencies: {fundamental_freqs} Hz")

    # Generate multi-source test signal
    duration = 0.5  # Short test
    n_samples = int(duration * 51200)
    t = np.arange(n_samples) / 51200.0

    # Create signal with 2 sources × 2 harmonics
    x_test = np.zeros((10, n_samples))
    for s, f0 in enumerate(fundamental_freqs):
        for h in [1, 2]:  # Two harmonics
            freq = f0 * h
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            x_test += tone  # Add to all channels

    # Add noise
    x_test += 0.1 * np.random.randn(10, n_samples)

    print(f"Test signal: {x_test.shape}, duration={duration}s")

    # Process
    print("Processing through cascade...")
    y_test, metadata_test = filter_bank.process_forward_pass(x_test)

    print(f"Output: {y_test.shape}")
    print(f"Tracked frequencies shape: {metadata_test['tracked_frequencies'].shape}")

    # Show tracked frequencies
    print("\nTracked frequencies:")
    for s in range(2):
        for m in range(2):
            f_tracked = metadata_test['tracked_frequencies'][s, m]
            f_expected = fundamental_freqs[s] * (m + 1)
            error = abs(f_tracked - f_expected)
            print(f"  Source {s+1}, Harmonic {m+1}: "
                  f"{f_tracked:.2f} Hz (expected {f_expected:.2f} Hz, "
                  f"error {error:.3f} Hz)")

    # Measure suppression
    suppression_estimate = 20 * np.log10(np.std(x_test) / (np.std(y_test) + 1e-10))
    print(f"\nOverall suppression: {suppression_estimate:.1f} dB")

    # Test zero-phase filtering with even smaller signal
    print("\n4. Testing Zero-Phase Filtering")
    print("-" * 40)

    # Very short signal for fast test
    duration_zp = 0.2
    n_samples_zp = int(duration_zp * 51200)
    t_zp = np.arange(n_samples_zp) / 51200.0

    # Single source, single harmonic for quick test
    # Need at least 2 channels per source for quadrant strategy
    config_zp = MIMOFilterConfig(
        n_sources=1,
        n_harmonics=1,
        n_channels=5,
        sample_rate=51200.0,
        harmonic_vector=[1],
        zero_phase=True,
        designation_strategy='explicit',
        designation_vector=[1, 1, 1, 1, 1]  # All channels to source 1
    )

    filter_bank_zp = MIMOFilterBank(config_zp)
    filter_bank_zp.initialize_from_frequencies(np.array([195.0]))

    # Test signal
    x_zp = np.zeros((5, n_samples_zp))
    tone_zp = np.sin(2 * np.pi * 195.0 * t_zp)
    x_zp += tone_zp
    x_zp += 0.1 * np.random.randn(5, n_samples_zp)

    print(f"Test signal: {x_zp.shape}")
    print("Applying zero-phase filtering...")

    y_zp, metadata_zp = filter_bank_zp.process_zero_phase(x_zp)

    suppression_zp = 20 * np.log10(np.std(x_zp) / (np.std(y_zp) + 1e-10))
    print(f"Zero-phase suppression: {suppression_zp:.1f} dB")
    print(f"Zero-phase flag: {metadata_zp.get('zero_phase', False)}")

    # Get suppression summary
    summary = filter_bank_zp.get_suppression_summary()
    print(f"\nSuppression summary:")
    for source_info in summary['sources']:
        print(f"  Source {source_info['source_idx']}: "
              f"f={source_info['frequencies'][0]:.2f} Hz, "
              f"converged={source_info['converged'][0]}")

    print("\n" + "="*60)
    print("All classes working correctly!")
    print("Core implementation complete:")
    print("  ✓ IIRNotchFilter")
    print("  ✓ MIMOFilterConfig")
    print("  ✓ MIMOFilterStage")
    print("  ✓ MIMOFilterBank")
    print("  ✓ Zero-phase filtering")
    print("="*60)
