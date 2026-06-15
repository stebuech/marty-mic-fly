#!/usr/bin/env python3
"""
Frequency Initialization Methods for MIMO Adaptive IIR Filter

Implements three methods for detecting BPF fundamental frequencies:
1. Autocorrelation-based detection
2. FFT peak detection
3. RPM telemetry-based (direct from hardware)

Based on Harvey (2019) Section 3.4.2: Frequency Initialization Strategies

Author: MartyMicFly Project
Date: 2025-12-15
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple
from sklearn.cluster import KMeans


class FrequencyInitializer:
    """
    Frequency initialization for MIMO adaptive IIR notch filters.

    Provides three methods for determining fundamental frequencies of
    rotor sources before adaptive filtering begins.
    """

    def __init__(self, sample_rate: float, n_sources: int, n_channels: int):
        """
        Initialize frequency detector.

        Args:
            sample_rate: Sampling rate (Hz)
            n_sources: Number of rotor sources to detect
            n_channels: Number of microphone channels available
        """
        self.sample_rate = sample_rate
        self.n_sources = n_sources
        self.n_channels = n_channels

    def autocorrelation_method(self, time_data: np.ndarray,
                               f_min: float = 183.0, f_max: float = 207.0,
                               max_lag: Optional[int] = None) -> np.ndarray:
        """
        Detect fundamental frequencies using autocorrelation.

        Algorithm:
        1. Compute autocorrelation for each channel
        2. Find peak lag in valid range [lag_min, lag_max]
        3. Convert lag to frequency: f = fs / lag
        4. Cluster candidates into n_sources groups
        5. Return cluster centers as fundamentals

        Args:
            time_data: Input signal [n_channels, n_samples]
            f_min: Minimum expected BPF (Hz)
            f_max: Maximum expected BPF (Hz)
            max_lag: Maximum lag to search (auto if None)

        Returns:
            fundamental_freqs: Array of n_sources fundamental frequencies (Hz)
        """
        if max_lag is None:
            max_lag = int(self.sample_rate / f_min) + 100

        # Lag range corresponding to frequency range
        lag_min = int(self.sample_rate / f_max)
        lag_max = int(self.sample_rate / f_min)

        freq_candidates = []

        # Process each channel
        for ch in range(min(self.n_channels, time_data.shape[0])):
            x = time_data[ch, :]

            # Compute autocorrelation
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags only

            # Limit to search range
            autocorr_search = autocorr[lag_min:min(lag_max, len(autocorr))]

            if len(autocorr_search) == 0:
                continue

            # Find peak
            peak_idx = np.argmax(autocorr_search)
            peak_lag = peak_idx + lag_min

            # Convert to frequency
            freq = self.sample_rate / peak_lag

            if f_min <= freq <= f_max:
                freq_candidates.append(freq)

        if len(freq_candidates) < self.n_sources:
            raise ValueError(f"Only found {len(freq_candidates)} candidates, "
                           f"need {self.n_sources}")

        # Cluster into n_sources groups
        fundamental_freqs = self._cluster_frequencies(
            np.array(freq_candidates), self.n_sources
        )

        return fundamental_freqs

    def fft_peak_detection_method(self, time_data: np.ndarray,
                                   f_min: float = 183.0, f_max: float = 207.0,
                                   nfft: int = 65536,
                                   min_separation: float = 2.0) -> np.ndarray:
        """
        Detect fundamental frequencies using FFT peak detection.

        Algorithm:
        1. Compute averaged PSD using Welch's method
        2. Find peaks in [f_min, f_max] range
        3. Select n_sources strongest peaks
        4. Enforce minimum frequency separation
        5. Sort and return

        Args:
            time_data: Input signal [n_channels, n_samples]
            f_min: Minimum expected BPF (Hz)
            f_max: Maximum expected BPF (Hz)
            nfft: FFT length for frequency resolution
            min_separation: Minimum separation between peaks (Hz)

        Returns:
            fundamental_freqs: Array of n_sources fundamental frequencies (Hz)
        """
        # Compute PSD for all channels and average
        psd_list = []

        for ch in range(min(self.n_channels, time_data.shape[0])):
            freqs, psd = signal.welch(
                time_data[ch, :],
                fs=self.sample_rate,
                nperseg=min(nfft, time_data.shape[1]),
                scaling='density',
                window='hann'
            )
            psd_list.append(psd)

        # Average across channels (linear scale)
        psd_avg = np.mean(psd_list, axis=0)

        # Find frequency range indices
        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_search = freqs[freq_mask]
        psd_search = psd_avg[freq_mask]

        if len(freqs_search) == 0:
            raise ValueError(f"No frequencies in range [{f_min}, {f_max}] Hz")

        # Find peaks
        peak_indices, peak_properties = signal.find_peaks(
            psd_search,
            height=np.max(psd_search) * 0.1,  # At least 10% of max
            distance=int(min_separation / (freqs[1] - freqs[0]))  # Min separation in bins
        )

        if len(peak_indices) == 0:
            # No peaks found - use highest point
            peak_idx = np.argmax(psd_search)
            fundamental_freqs = np.array([freqs_search[peak_idx]])
            # Duplicate if needed
            while len(fundamental_freqs) < self.n_sources:
                fundamental_freqs = np.append(fundamental_freqs, fundamental_freqs[0])
            return fundamental_freqs

        # Get peak frequencies and heights
        peak_freqs = freqs_search[peak_indices]
        peak_heights = peak_properties['peak_heights']

        # Sort by height (descending)
        sorted_indices = np.argsort(peak_heights)[::-1]
        peak_freqs_sorted = peak_freqs[sorted_indices]

        # Select top n_sources with minimum separation
        selected_freqs = []
        for freq in peak_freqs_sorted:
            # Check minimum separation from already selected
            if len(selected_freqs) == 0:
                selected_freqs.append(freq)
            else:
                separations = np.abs(np.array(selected_freqs) - freq)
                if np.all(separations >= min_separation):
                    selected_freqs.append(freq)

            if len(selected_freqs) >= self.n_sources:
                break

        # If not enough found, add duplicates
        while len(selected_freqs) < self.n_sources:
            selected_freqs.append(selected_freqs[0])

        fundamental_freqs = np.array(sorted(selected_freqs))

        return fundamental_freqs

    def rpm_telemetry_method(self, rpm_data: Dict,
                            n_blades: int = 2,
                            correction_factor: float = 7/12) -> np.ndarray:
        """
        Extract fundamental frequencies from RPM telemetry.

        This is the most accurate method when RPM data is available
        and properly synchronized with microphone data.

        Algorithm:
        1. Extract per-motor RPM from telemetry
        2. Apply correction factor (if not already applied)
        3. Calculate BPF = (RPM / 60) × n_blades
        4. Return as fundamental frequencies

        Args:
            rpm_data: Dictionary from load_rpm_telemetry()
                     Must contain 'rpm_per_esc' or similar structure
            n_blades: Number of blades per rotor
            correction_factor: RPM correction (default 7/12)
                             Note: May already be applied in load_rpm_telemetry()

        Returns:
            fundamental_freqs: Array of n_sources fundamental frequencies (Hz)
        """
        # Try to extract RPM data from various possible structures
        fundamental_freqs = []

        # Method 1: rpm_per_esc dict
        if 'rpm_per_esc' in rpm_data:
            rpm_per_esc = rpm_data['rpm_per_esc']
            for esc_name in sorted(rpm_per_esc.keys()):
                esc_data = rpm_per_esc[esc_name]

                if 'rpm' in esc_data:
                    rpm_values = esc_data['rpm']
                elif 'mean_rpm' in esc_data:
                    rpm_values = esc_data['mean_rpm']
                else:
                    continue

                # Take mean RPM over time
                if isinstance(rpm_values, np.ndarray):
                    rpm_mean = np.mean(rpm_values)
                else:
                    rpm_mean = rpm_values

                # Check if correction already applied
                # (Heuristic: corrected RPM for quadcopter at hover is ~3000-4000)
                if rpm_mean > 4500:  # Likely uncorrected
                    rpm_corrected = rpm_mean * correction_factor
                else:  # Likely already corrected
                    rpm_corrected = rpm_mean

                # Calculate BPF
                bpf = (rpm_corrected / 60.0) * n_blades

                fundamental_freqs.append(bpf)

                if len(fundamental_freqs) >= self.n_sources:
                    break

        # Method 2: Direct rpm_per_motor array
        elif 'rpm_per_motor' in rpm_data:
            rpm_per_motor = rpm_data['rpm_per_motor']
            for rpm_mean in rpm_per_motor[:self.n_sources]:
                if rpm_mean > 4500:
                    rpm_corrected = rpm_mean * correction_factor
                else:
                    rpm_corrected = rpm_mean

                bpf = (rpm_corrected / 60.0) * n_blades
                fundamental_freqs.append(bpf)

        # Method 3: Use extract_stable_segment per_motor_stats
        elif 'per_motor_stats' in rpm_data:
            stats = rpm_data['per_motor_stats']
            if 'mean_rpm' in stats:
                for rpm_mean in stats['mean_rpm'][:self.n_sources]:
                    if rpm_mean > 4500:
                        rpm_corrected = rpm_mean * correction_factor
                    else:
                        rpm_corrected = rpm_mean

                    bpf = (rpm_corrected / 60.0) * n_blades
                    fundamental_freqs.append(bpf)

        if len(fundamental_freqs) == 0:
            raise ValueError("Could not extract RPM data from provided dictionary. "
                           "Expected 'rpm_per_esc', 'rpm_per_motor', or 'per_motor_stats'.")

        # Ensure we have exactly n_sources frequencies
        while len(fundamental_freqs) < self.n_sources:
            fundamental_freqs.append(fundamental_freqs[0])  # Duplicate if needed

        fundamental_freqs = np.array(fundamental_freqs[:self.n_sources])

        return fundamental_freqs

    def compare_methods(self, time_data: np.ndarray,
                       rpm_data: Optional[Dict] = None,
                       f_min: float = 183.0, f_max: float = 207.0) -> Dict:
        """
        Run all available methods and compare results.

        Args:
            time_data: Input signal [n_channels, n_samples]
            rpm_data: Optional RPM telemetry data
            f_min: Minimum expected BPF (Hz)
            f_max: Maximum expected BPF (Hz)

        Returns:
            Dictionary with:
                - 'autocorr': Frequencies from autocorrelation
                - 'fft': Frequencies from FFT peaks
                - 'rpm': Frequencies from RPM (if available)
                - 'recommended': Best method's frequencies
                - 'agreement': Agreement metrics
        """
        results = {}

        # Method 1: Autocorrelation
        try:
            freqs_autocorr = self.autocorrelation_method(time_data, f_min, f_max)
            results['autocorr'] = freqs_autocorr
        except Exception as e:
            results['autocorr'] = None
            results['autocorr_error'] = str(e)

        # Method 2: FFT
        try:
            freqs_fft = self.fft_peak_detection_method(time_data, f_min, f_max)
            results['fft'] = freqs_fft
        except Exception as e:
            results['fft'] = None
            results['fft_error'] = str(e)

        # Method 3: RPM (if available)
        if rpm_data is not None:
            try:
                freqs_rpm = self.rpm_telemetry_method(rpm_data)
                results['rpm'] = freqs_rpm
            except Exception as e:
                results['rpm'] = None
                results['rpm_error'] = str(e)
        else:
            results['rpm'] = None

        # Determine recommended method
        if results['rpm'] is not None:
            # RPM is most reliable when available
            results['recommended'] = results['rpm']
            results['recommended_method'] = 'rpm'
        elif results['fft'] is not None:
            # FFT is second choice
            results['recommended'] = results['fft']
            results['recommended_method'] = 'fft'
        elif results['autocorr'] is not None:
            # Autocorrelation as fallback
            results['recommended'] = results['autocorr']
            results['recommended_method'] = 'autocorr'
        else:
            results['recommended'] = None
            results['recommended_method'] = None

        # Calculate agreement metrics
        agreement = {}
        available_methods = [k for k in ['autocorr', 'fft', 'rpm']
                           if results[k] is not None]

        if len(available_methods) >= 2:
            # Pairwise comparisons
            all_freqs = np.array([results[m] for m in available_methods])
            agreement['mean'] = np.mean(all_freqs, axis=0)
            agreement['std'] = np.std(all_freqs, axis=0)
            agreement['max_deviation'] = np.max(agreement['std'])
            agreement['good_agreement'] = agreement['max_deviation'] < 2.0  # < 2 Hz

        results['agreement'] = agreement

        return results

    def _cluster_frequencies(self, freq_candidates: np.ndarray,
                           n_clusters: int) -> np.ndarray:
        """
        Cluster frequency candidates into n_clusters groups using K-means.

        Args:
            freq_candidates: Array of frequency candidates
            n_clusters: Number of clusters (sources)

        Returns:
            Cluster centers sorted by frequency
        """
        if len(freq_candidates) < n_clusters:
            # Not enough candidates - duplicate
            freq_candidates = np.tile(freq_candidates,
                                    (n_clusters // len(freq_candidates)) + 1)

        # Reshape for sklearn
        X = freq_candidates.reshape(-1, 1)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)

        # Get cluster centers
        centers = kmeans.cluster_centers_.flatten()

        # Sort by frequency
        centers = np.sort(centers)

        return centers


if __name__ == "__main__":
    # Test frequency initialization methods
    print("="*60)
    print("Frequency Initialization Methods - Testing")
    print("="*60)

    # Generate synthetic test signal
    fs = 51200.0
    duration = 2.0
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Create 4-source signal with known frequencies
    true_freqs = np.array([185.0, 192.0, 198.0, 205.0])
    n_channels = 16

    print(f"\nTrue fundamental frequencies: {true_freqs} Hz")
    print(f"Channels: {n_channels}, Duration: {duration}s")

    x_test = np.zeros((n_channels, n_samples))
    for freq in true_freqs:
        # Add fundamental and 2 harmonics
        for h in [1, 2, 3]:
            tone = 0.3 * np.sin(2 * np.pi * freq * h * t)
            x_test += tone

    # Add noise
    x_test += 0.1 * np.random.randn(n_channels, n_samples)

    # Create initializer
    initializer = FrequencyInitializer(fs, n_sources=4, n_channels=n_channels)

    # Test Method 1: Autocorrelation
    print("\n1. Autocorrelation Method")
    print("-" * 40)
    freqs_autocorr = initializer.autocorrelation_method(x_test, f_min=180, f_max=210)
    print(f"Detected: {freqs_autocorr}")
    errors_autocorr = np.abs(freqs_autocorr - true_freqs)
    print(f"Errors: {errors_autocorr}")
    print(f"Mean error: {np.mean(errors_autocorr):.3f} Hz")

    # Test Method 2: FFT
    print("\n2. FFT Peak Detection Method")
    print("-" * 40)
    freqs_fft = initializer.fft_peak_detection_method(x_test, f_min=180, f_max=210)
    print(f"Detected: {freqs_fft}")
    errors_fft = np.abs(freqs_fft - true_freqs)
    print(f"Errors: {errors_fft}")
    print(f"Mean error: {np.mean(errors_fft):.3f} Hz")

    # Test Method 3: RPM (simulated)
    print("\n3. RPM Telemetry Method")
    print("-" * 40)

    # Create mock RPM data
    rpm_per_motor = (true_freqs / 2.0) * 60  # BPF = (RPM/60)*2, so RPM = BPF*30
    rpm_data_mock = {
        'rpm_per_motor': rpm_per_motor * (12/7)  # Uncorrected RPM
    }

    freqs_rpm = initializer.rpm_telemetry_method(rpm_data_mock, n_blades=2, correction_factor=7/12)
    print(f"Detected: {freqs_rpm}")
    errors_rpm = np.abs(freqs_rpm - true_freqs)
    print(f"Errors: {errors_rpm}")
    print(f"Mean error: {np.mean(errors_rpm):.3f} Hz")

    # Test comparison
    print("\n4. Method Comparison")
    print("-" * 40)
    comparison = initializer.compare_methods(x_test, rpm_data_mock, f_min=180, f_max=210)

    print(f"Autocorrelation: {comparison['autocorr']}")
    print(f"FFT:             {comparison['fft']}")
    print(f"RPM:             {comparison['rpm']}")
    print(f"\nRecommended ({comparison['recommended_method']}): {comparison['recommended']}")

    if 'agreement' in comparison and comparison['agreement']:
        agreement = comparison['agreement']
        print(f"\nAgreement metrics:")
        print(f"  Mean: {agreement['mean']}")
        print(f"  Std: {agreement['std']}")
        print(f"  Max deviation: {agreement['max_deviation']:.3f} Hz")
        print(f"  Good agreement: {agreement['good_agreement']}")

    print("\n" + "="*60)
    print("Frequency initialization methods working!")
    print("="*60)
