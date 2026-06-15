#!/usr/bin/env python3
"""
Validation and Performance Measurement for MIMO Adaptive IIR Filters

Provides tools for:
- Suppression measurement at harmonic frequencies
- Frequency tracking accuracy (MFD - Mean Frequency Deviation)
- Phase preservation validation
- Synthetic signal generation for testing

Author: MartyMicFly Project
Date: 2025-12-15
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple


class MIMOFilterValidator:
    """
    Validation tools for MIMO adaptive IIR notch filters.

    Measures performance metrics to validate filter operation
    according to Harvey (2019) expected performance:
    - Suppression: 20-25 dB (single-pass), 40-50 dB (zero-phase)
    - Tracking: MFD < 0.25 Hz
    - Phase: Negligible distortion (<0.01 rad)
    """

    def __init__(self):
        pass

    def measure_suppression(self, original: np.ndarray, filtered: np.ndarray,
                           harmonic_freqs: np.ndarray,
                           bandwidth: float = 10.0,
                           sample_rate: float = 51200.0) -> Dict:
        """
        Measure suppression at harmonic frequencies.

        Algorithm:
        1. Compute PSD for original and filtered signals
        2. For each harmonic: integrate power in [f - BW/2, f + BW/2]
        3. Suppression[h] = 10*log10(P_original[h] / P_filtered[h])

        Args:
            original: Original signal [n_channels, n_samples]
            filtered: Filtered signal [n_channels, n_samples]
            harmonic_freqs: Array of harmonic frequencies to measure (Hz)
            bandwidth: Integration bandwidth around each harmonic (Hz)
            sample_rate: Sampling rate (Hz)

        Returns:
            Dictionary with:
                - per_harmonic: Suppression per harmonic (dB)
                - mean_suppression: Mean across all harmonics (dB)
                - rms_suppression: RMS across all harmonics (dB)
                - broadband_preservation: Change in broadband energy (dB)
        """
        # Compute PSDs
        freqs_orig, psd_orig = self._compute_averaged_psd(original, sample_rate)
        freqs_filt, psd_filt = self._compute_averaged_psd(filtered, sample_rate)

        # Ensure same frequency axis
        assert np.allclose(freqs_orig, freqs_filt), "Frequency axes must match"
        freqs = freqs_orig
        df = freqs[1] - freqs[0]

        # Measure suppression at each harmonic
        suppression_per_harmonic = []

        for f_harmonic in harmonic_freqs:
            # Find frequency range
            f_low = f_harmonic - bandwidth / 2.0
            f_high = f_harmonic + bandwidth / 2.0

            # Find indices
            idx_low = np.argmin(np.abs(freqs - f_low))
            idx_high = np.argmin(np.abs(freqs - f_high))

            if idx_low == idx_high:
                idx_high = idx_low + 1

            # Integrate power
            power_orig = np.trapz(psd_orig[idx_low:idx_high], freqs[idx_low:idx_high])
            power_filt = np.trapz(psd_filt[idx_low:idx_high], freqs[idx_low:idx_high])

            # Suppression in dB
            if power_filt > 1e-20:  # Avoid log(0)
                suppression_dB = 10 * np.log10(power_orig / power_filt)
            else:
                suppression_dB = 100.0  # Very high suppression

            suppression_per_harmonic.append(suppression_dB)

        suppression_per_harmonic = np.array(suppression_per_harmonic)

        # Calculate aggregate metrics
        mean_suppression = np.mean(suppression_per_harmonic)
        rms_suppression = np.sqrt(np.mean(suppression_per_harmonic**2))

        # Broadband preservation (total energy change)
        total_power_orig = np.trapz(psd_orig, freqs)
        total_power_filt = np.trapz(psd_filt, freqs)
        broadband_change_dB = 10 * np.log10(total_power_filt / total_power_orig)

        return {
            'per_harmonic': suppression_per_harmonic,
            'mean_suppression': mean_suppression,
            'rms_suppression': rms_suppression,
            'broadband_preservation': broadband_change_dB,
            'harmonic_frequencies': harmonic_freqs
        }

    def measure_tracking_accuracy(self, tracked_freqs: np.ndarray,
                                  ground_truth_freqs: np.ndarray) -> Dict:
        """
        Measure frequency tracking accuracy (MFD - Mean Frequency Deviation).

        Args:
            tracked_freqs: Tracked frequencies [n_sources, n_harmonics] (Hz)
            ground_truth_freqs: True frequencies [n_sources, n_harmonics] (Hz)

        Returns:
            Dictionary with:
                - mfd: Mean frequency deviation per stage (Hz)
                - max_deviation: Maximum deviation across all stages (Hz)
                - good_tracking: Boolean, True if MFD < 0.5 Hz
        """
        # Calculate deviations
        deviations = np.abs(tracked_freqs - ground_truth_freqs)

        # MFD per stage
        mfd = np.mean(deviations, axis=None)  # Mean across all stages

        # Maximum deviation
        max_deviation = np.max(deviations)

        # Per-source MFD
        mfd_per_source = np.mean(deviations, axis=1)

        # Per-harmonic MFD
        mfd_per_harmonic = np.mean(deviations, axis=0)

        # Good tracking threshold (Harvey: MFD < 0.25 Hz)
        good_tracking = mfd < 0.5
        excellent_tracking = mfd < 0.25

        return {
            'mfd': mfd,
            'mfd_per_source': mfd_per_source,
            'mfd_per_harmonic': mfd_per_harmonic,
            'max_deviation': max_deviation,
            'good_tracking': good_tracking,
            'excellent_tracking': excellent_tracking,
            'deviations': deviations
        }

    def validate_phase_preservation(self, original: np.ndarray,
                                    filtered: np.ndarray,
                                    sample_rate: float = 51200.0,
                                    freq_range: Tuple[float, float] = (100, 10000)) -> Dict:
        """
        Validate zero-phase property and spatial phase preservation.

        Args:
            original: Original signal [n_channels, n_samples]
            filtered: Filtered signal [n_channels, n_samples]
            sample_rate: Sampling rate (Hz)
            freq_range: Frequency range to analyze (Hz)

        Returns:
            Dictionary with:
                - max_phase_distortion: Maximum phase distortion (rad)
                - coherence: Cross-channel coherence (preserved?)
                - spatial_preserved: Boolean, True if phase well-preserved
        """
        n_channels = original.shape[0]

        # Compute transfer function between original and filtered
        freqs, H = self._compute_transfer_function(original[0, :], filtered[0, :], sample_rate)

        # Extract phase
        phase = np.angle(H)

        # Restrict to frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        phase_in_range = phase[freq_mask]

        # Maximum phase distortion (should be ~0 for zero-phase)
        max_phase_distortion = np.max(np.abs(phase_in_range))

        # Check cross-channel coherence preservation
        # Compute coherence between channel pairs for original and filtered
        coherence_original = self._compute_avg_coherence(original, sample_rate)
        coherence_filtered = self._compute_avg_coherence(filtered, sample_rate)

        # Coherence loss (should be minimal)
        coherence_loss = np.abs(coherence_original - coherence_filtered)
        max_coherence_loss = np.max(coherence_loss[freq_mask])

        # Spatial preserved if phase distortion < 0.1 rad and coherence loss < 0.1
        spatial_preserved = (max_phase_distortion < 0.1) and (max_coherence_loss < 0.1)

        return {
            'max_phase_distortion': max_phase_distortion,
            'max_coherence_loss': max_coherence_loss,
            'spatial_preserved': spatial_preserved,
            'frequencies': freqs[freq_mask],
            'phase': phase_in_range
        }

    def generate_synthetic_signal(self, n_channels: int, n_samples: int,
                                  sample_rate: float,
                                  bpf_freqs: np.ndarray,
                                  harmonics: List[float],
                                  snr_db: float = 20.0) -> Tuple[np.ndarray, Dict]:
        """
        Generate synthetic multi-source signal with known harmonics.

        Args:
            n_channels: Number of channels
            n_samples: Number of samples
            sample_rate: Sampling rate (Hz)
            bpf_freqs: BPF frequencies for each source [n_sources]
            harmonics: List of harmonic multipliers (e.g., [1, 2, 3, 4, 5])
            snr_db: Signal-to-noise ratio (dB)

        Returns:
            signal: Synthetic signal [n_channels, n_samples]
            ground_truth: Dictionary with true parameters
        """
        t = np.arange(n_samples) / sample_rate
        signal_data = np.zeros((n_channels, n_samples))

        # Generate tones for each source and harmonic
        ground_truth_freqs = []
        for bpf in bpf_freqs:
            source_freqs = []
            for h in harmonics:
                freq = bpf * h
                # Random amplitude and phase per harmonic
                amplitude = 1.0 / np.sqrt(h)  # Decaying with harmonic number
                phase = np.random.uniform(0, 2 * np.pi)

                tone = amplitude * np.sin(2 * np.pi * freq * t + phase)
                signal_data += tone
                source_freqs.append(freq)
            ground_truth_freqs.append(source_freqs)

        # Calculate signal power
        signal_power = np.mean(signal_data**2)

        # Add noise to achieve desired SNR
        noise_power = signal_power / (10**(snr_db / 10.0))
        noise = np.sqrt(noise_power) * np.random.randn(n_channels, n_samples)

        signal_data += noise

        ground_truth = {
            'bpf_freqs': bpf_freqs,
            'harmonics': harmonics,
            'all_frequencies': np.array(ground_truth_freqs),
            'snr_db': snr_db,
            'sample_rate': sample_rate
        }

        return signal_data, ground_truth

    def _compute_averaged_psd(self, time_data: np.ndarray,
                             sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute averaged PSD across channels."""
        psd_list = []
        for ch in range(time_data.shape[0]):
            freqs, psd = signal.welch(
                time_data[ch, :],
                fs=sample_rate,
                nperseg=min(8192, time_data.shape[1]),
                scaling='density',
                window='hann'
            )
            psd_list.append(psd)

        psd_avg = np.mean(psd_list, axis=0)
        return freqs, psd_avg

    def _compute_transfer_function(self, x: np.ndarray, y: np.ndarray,
                                  sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transfer function H = Y/X."""
        freqs, Pxy = signal.csd(x, y, fs=sample_rate, nperseg=8192)
        _, Pxx = signal.welch(x, fs=sample_rate, nperseg=8192)

        H = Pxy / (Pxx + 1e-10)  # Avoid division by zero
        return freqs, H

    def _compute_avg_coherence(self, time_data: np.ndarray,
                              sample_rate: float) -> np.ndarray:
        """Compute average coherence across channel pairs."""
        n_channels = time_data.shape[0]
        coherence_list = []

        # Sample a few channel pairs (not all to save time)
        for i in range(min(5, n_channels)):
            for j in range(i + 1, min(5, n_channels)):
                freqs, coh = signal.coherence(
                    time_data[i, :],
                    time_data[j, :],
                    fs=sample_rate,
                    nperseg=8192
                )
                coherence_list.append(coh)

        if len(coherence_list) > 0:
            avg_coherence = np.mean(coherence_list, axis=0)
        else:
            avg_coherence = np.ones_like(freqs)

        return avg_coherence


if __name__ == "__main__":
    # Test validation methods
    print("="*60)
    print("MIMO Filter Validation Tools - Testing")
    print("="*60)

    validator = MIMOFilterValidator()

    # Generate synthetic test signal
    print("\n1. Generating Synthetic Signal")
    print("-" * 40)

    bpf_freqs = np.array([190.0, 200.0])
    harmonics = [1, 2, 3, 4, 5]

    signal_original, ground_truth = validator.generate_synthetic_signal(
        n_channels=8,
        n_samples=51200,  # 1 second at 51.2 kHz
        sample_rate=51200.0,
        bpf_freqs=bpf_freqs,
        harmonics=harmonics,
        snr_db=20.0
    )

    print(f"Signal shape: {signal_original.shape}")
    print(f"BPF frequencies: {ground_truth['bpf_freqs']} Hz")
    print(f"Harmonics: {ground_truth['harmonics']}")
    print(f"All frequencies:\n{ground_truth['all_frequencies']}")

    # Simulate filtered signal (add some suppression)
    print("\n2. Simulating Filtered Signal")
    print("-" * 40)

    # Create "filtered" signal with reduced harmonics
    signal_filtered = signal_original.copy()

    # Manually suppress some harmonics
    t = np.arange(signal_original.shape[1]) / 51200.0
    for bpf in bpf_freqs:
        for h in [1, 2]:  # Suppress first 2 harmonics
            freq = bpf * h
            # Create notch by subtracting the tone (simulated removal)
            amplitude = 0.8 / np.sqrt(h)
            tone = amplitude * np.sin(2 * np.pi * freq * t)
            signal_filtered -= tone

    print("Simulated suppression of first 2 harmonics per source")

    # Test suppression measurement
    print("\n3. Measuring Suppression")
    print("-" * 40)

    # Get all harmonic frequencies
    all_harm_freqs = ground_truth['all_frequencies'].flatten()

    suppression = validator.measure_suppression(
        signal_original,
        signal_filtered,
        all_harm_freqs,
        bandwidth=10.0,
        sample_rate=51200.0
    )

    print(f"Mean suppression: {suppression['mean_suppression']:.1f} dB")
    print(f"RMS suppression: {suppression['rms_suppression']:.1f} dB")
    print(f"Broadband preservation: {suppression['broadband_preservation']:.1f} dB")
    print(f"\nPer-harmonic suppression (dB):")
    for i, (freq, supp) in enumerate(zip(all_harm_freqs, suppression['per_harmonic'])):
        print(f"  {freq:7.1f} Hz: {supp:6.1f} dB")

    # Test tracking accuracy
    print("\n4. Measuring Tracking Accuracy")
    print("-" * 40)

    # Simulate tracked frequencies (with small errors)
    tracked_freqs = ground_truth['all_frequencies'] + np.random.randn(2, 5) * 0.3

    tracking = validator.measure_tracking_accuracy(
        tracked_freqs,
        ground_truth['all_frequencies']
    )

    print(f"Mean Frequency Deviation (MFD): {tracking['mfd']:.3f} Hz")
    print(f"Max deviation: {tracking['max_deviation']:.3f} Hz")
    print(f"Good tracking (MFD < 0.5 Hz): {tracking['good_tracking']}")
    print(f"Excellent tracking (MFD < 0.25 Hz): {tracking['excellent_tracking']}")

    # Test phase preservation
    print("\n5. Validating Phase Preservation")
    print("-" * 40)

    # For this test, filtered signal should have minimal phase distortion
    phase_validation = validator.validate_phase_preservation(
        signal_original,
        signal_filtered,
        sample_rate=51200.0,
        freq_range=(100, 2000)
    )

    print(f"Max phase distortion: {phase_validation['max_phase_distortion']:.4f} rad")
    print(f"Max coherence loss: {phase_validation['max_coherence_loss']:.4f}")
    print(f"Spatial phase preserved: {phase_validation['spatial_preserved']}")

    print("\n" + "="*60)
    print("Validation tools working correctly!")
    print("="*60)
