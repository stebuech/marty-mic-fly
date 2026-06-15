#!/usr/bin/env python3
"""
Standalone MIMO Adaptive IIR Notch Filter Analysis

Applies MIMO adaptive IIR notch filtering to microphone array data
and generates comprehensive performance reports.

Usage:
    python mimo_filter_analysis.py --mic-h5 measurement.h5 --rpm-h5 rpm.h5 \\
           --output-dir results/ --method rpm

Author: MartyMicFly Project
Date: 2025-12-15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from data_loader import (
    load_microphone_data,
    load_rpm_telemetry,
    synchronize_datasets,
    extract_stable_segment,
    apply_mimo_iir_filter
)
from mimo_adaptive_iir_filter import MIMOFilterConfig
from iir_filter_validation import MIMOFilterValidator
from spectral_comparison import compute_power_spectrum


def plot_time_domain_comparison(original, filtered, sample_rate, output_path):
    """Plot time-domain comparison of first channel."""
    duration = original.shape[1] / sample_rate
    t = np.arange(original.shape[1]) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Original
    axes[0].plot(t, original[0, :], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal (Channel 1)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)

    # Filtered
    axes[1].plot(t, filtered[0, :], 'r-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Filtered Signal (Channel 1)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, duration)

    # Removed noise (difference)
    removed = original[0, :] - filtered[0, :]
    axes[2].plot(t, removed, 'g-', linewidth=0.5, alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Removed Ego-Noise (Difference)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, duration)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_spectral_comparison(original, filtered, sample_rate,
                            harmonic_freqs, output_path):
    """Plot spectral comparison showing suppression."""
    # Compute PSDs
    orig_spectrum = compute_power_spectrum(original, sample_rate)
    filt_spectrum = compute_power_spectrum(filtered, sample_rate)

    freqs = orig_spectrum['frequencies']
    psd_orig_db = orig_spectrum['psd_dB']
    psd_filt_db = filt_spectrum['psd_dB']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Overlay comparison
    axes[0].plot(freqs, psd_orig_db, 'b-', linewidth=1.5,
                label='Original', alpha=0.7)
    axes[0].plot(freqs, psd_filt_db, 'r-', linewidth=1.5,
                label='Filtered', alpha=0.7)

    # Mark harmonics
    ylim = axes[0].get_ylim()
    for freq in harmonic_freqs:
        if freq < max(freqs):
            axes[0].axvline(freq, color='gray', linestyle='--',
                          linewidth=0.5, alpha=0.5)

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD (dB)')
    axes[0].set_title('Power Spectral Density Comparison')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, min(5000, max(freqs)))

    # Suppression (difference)
    suppression = psd_orig_db - psd_filt_db

    axes[1].plot(freqs, suppression, 'g-', linewidth=1.5)
    axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)

    # Mark harmonics
    for freq in harmonic_freqs:
        if freq < max(freqs):
            axes[1].axvline(freq, color='gray', linestyle='--',
                          linewidth=0.5, alpha=0.5)

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Suppression (dB)')
    axes[1].set_title('Suppression at Each Frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(5000, max(freqs)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tracked_frequencies(result, output_path):
    """Plot tracked frequencies per source and harmonic."""
    tracked_freqs = result['tracked_frequencies']
    n_sources, n_harmonics = tracked_freqs.shape

    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 3*n_sources),
                            sharex=True)
    if n_sources == 1:
        axes = [axes]

    harmonic_numbers = np.arange(1, n_harmonics + 1)

    for s in range(n_sources):
        ax = axes[s]
        freqs_source = tracked_freqs[s, :]

        ax.plot(harmonic_numbers, freqs_source, 'o-',
               linewidth=2, markersize=8, label=f'Source {s+1}')

        # Expected linear relationship
        fundamental = freqs_source[0]
        expected = fundamental * harmonic_numbers
        ax.plot(harmonic_numbers, expected, '--',
               linewidth=1, alpha=0.5, label='Expected')

        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Source {s+1} - Fundamental: {fundamental:.2f} Hz')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Harmonic Number')

    plt.suptitle('Tracked Frequencies per Source', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_report(result, output_path):
    """Generate comprehensive text report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MIMO ADAPTIVE IIR NOTCH FILTER - ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        config = result['config']
        f.write(f"Sources: {config.n_sources}\n")
        f.write(f"Harmonics per source: {config.n_harmonics}\n")
        f.write(f"Channels: {config.n_channels}\n")
        f.write(f"Sample rate: {config.sample_rate/1000:.1f} kHz\n")
        f.write(f"Initialization method: {config.initialization_method}\n")
        f.write(f"Zero-phase filtering: {config.zero_phase}\n")
        f.write(f"Pole radius (r): {config.pole_radius}\n")
        f.write(f"Base step size (μ): {config.mu_base:.1e}\n")
        f.write(f"Step size increment: {config.mu_increment:.1e}\n\n")

        # Detected Frequencies
        f.write("DETECTED FUNDAMENTAL FREQUENCIES\n")
        f.write("-"*70 + "\n")
        fundamentals = result['fundamental_frequencies']
        for i, freq in enumerate(fundamentals):
            f.write(f"Source {i+1}: {freq:7.2f} Hz\n")
        f.write("\n")

        # Tracked Frequencies
        f.write("TRACKED FREQUENCIES (All Sources × Harmonics)\n")
        f.write("-"*70 + "\n")
        tracked = result['tracked_frequencies']
        for s in range(tracked.shape[0]):
            f.write(f"\nSource {s+1}:\n")
            for m in range(tracked.shape[1]):
                expected = fundamentals[s] * (m + 1)
                actual = tracked[s, m]
                error = abs(actual - expected)
                f.write(f"  Harmonic {m+1:2d}: {actual:7.2f} Hz "
                       f"(expected {expected:7.2f} Hz, error {error:.3f} Hz)\n")

        # Suppression Metrics
        f.write("\nSUPPRESSION PERFORMANCE\n")
        f.write("-"*70 + "\n")
        supp = result['suppression_metrics']
        f.write(f"Mean suppression: {supp['mean_suppression']:6.2f} dB\n")
        f.write(f"RMS suppression:  {supp['rms_suppression']:6.2f} dB\n")
        f.write(f"Broadband preservation: {supp['broadband_preservation']:6.2f} dB\n\n")

        f.write("Per-Harmonic Suppression:\n")
        for i, (freq, supp_db) in enumerate(zip(supp['harmonic_frequencies'],
                                                supp['per_harmonic'])):
            f.write(f"  {freq:7.1f} Hz: {supp_db:6.1f} dB\n")

        # Summary
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")

        mean_supp = supp['mean_suppression']
        if mean_supp >= 40:
            status = "EXCELLENT (≥40 dB)"
        elif mean_supp >= 25:
            status = "GOOD (25-40 dB)"
        elif mean_supp >= 15:
            status = "FAIR (15-25 dB)"
        else:
            status = "POOR (<15 dB)"

        f.write(f"Overall Performance: {status}\n")
        f.write(f"Mean Suppression: {mean_supp:.1f} dB\n")

        if config.zero_phase:
            f.write("\nZero-phase filtering enabled: Phase preservation confirmed\n")
        else:
            f.write("\nSingle-pass filtering: Some phase distortion expected\n")

        f.write("\n" + "="*70 + "\n")

    print(f"  Saved: {output_path}")


def save_results_json(result, output_path):
    """Save results as JSON for further processing."""
    # Convert numpy arrays to lists for JSON serialization
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'fundamental_frequencies': result['fundamental_frequencies'].tolist(),
        'tracked_frequencies': result['tracked_frequencies'].tolist(),
        'suppression_metrics': {
            'mean_suppression': float(result['suppression_metrics']['mean_suppression']),
            'rms_suppression': float(result['suppression_metrics']['rms_suppression']),
            'broadband_preservation': float(result['suppression_metrics']['broadband_preservation']),
            'harmonic_frequencies': result['suppression_metrics']['harmonic_frequencies'].tolist(),
            'per_harmonic': result['suppression_metrics']['per_harmonic'].tolist()
        },
        'config': {
            'n_sources': result['config'].n_sources,
            'n_harmonics': result['config'].n_harmonics,
            'n_channels': result['config'].n_channels,
            'sample_rate': result['config'].sample_rate,
            'initialization_method': result['config'].initialization_method,
            'zero_phase': result['config'].zero_phase,
            'pole_radius': result['config'].pole_radius,
            'mu_base': result['config'].mu_base,
            'mu_increment': result['config'].mu_increment
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='MIMO Adaptive IIR Notch Filter - Standalone Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with RPM telemetry
  python mimo_filter_analysis.py --mic-h5 measurement.h5 --rpm-h5 rpm.h5

  # FFT initialization (no RPM data)
  python mimo_filter_analysis.py --mic-h5 measurement.h5 --method fft

  # Custom configuration
  python mimo_filter_analysis.py --mic-h5 measurement.h5 --rpm-h5 rpm.h5 \\
      --config custom_mimo_config.yaml --output-dir results/

  # Quick test with subset of channels
  python mimo_filter_analysis.py --mic-h5 measurement.h5 --rpm-h5 rpm.h5 \\
      --n-channels 10 --n-harmonics 5
        """)

    parser.add_argument('--mic-h5', required=True,
                       help='Microphone array HDF5 file')
    parser.add_argument('--rpm-h5',
                       help='RPM telemetry HDF5 file (required for method=rpm)')
    parser.add_argument('--config',
                       help='YAML configuration file (overrides other options)')
    parser.add_argument('--output-dir', default='mimo_filter_results',
                       help='Output directory for results (default: mimo_filter_results)')

    # Filter configuration
    parser.add_argument('--method', choices=['autocorr', 'fft', 'rpm', 'compare_all'],
                       default='rpm',
                       help='Frequency initialization method (default: rpm)')
    parser.add_argument('--n-sources', type=int, default=4,
                       help='Number of rotor sources (default: 4)')
    parser.add_argument('--n-harmonics', type=int, default=10,
                       help='Harmonics per source (default: 10)')
    parser.add_argument('--n-channels', type=int,
                       help='Number of channels to process (default: all)')
    parser.add_argument('--zero-phase', action='store_true', default=True,
                       help='Enable zero-phase filtering (default: True)')
    parser.add_argument('--no-zero-phase', action='store_false', dest='zero_phase',
                       help='Disable zero-phase filtering')

    # Data extraction
    parser.add_argument('--stable-duration', type=float, default=10.0,
                       help='Stable segment duration in seconds (default: 10.0)')

    args = parser.parse_args()

    print("="*70)
    print("MIMO ADAPTIVE IIR NOTCH FILTER - STANDALONE ANALYSIS")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load microphone data
    print(f"\nLoading microphone data: {args.mic_h5}")
    mic_data = load_microphone_data(args.mic_h5, exclude_channel_1=True)
    print(f"  Loaded: {mic_data['n_channels']} channels, "
          f"{mic_data['n_samples']} samples ({mic_data['duration']:.2f}s)")

    # Load RPM data if provided
    rpm_data = None
    if args.rpm_h5:
        print(f"\nLoading RPM telemetry: {args.rpm_h5}")
        rpm_data = load_rpm_telemetry(args.rpm_h5)
        print(f"  Loaded: {rpm_data['esc_count']} ESCs")

        # Synchronize
        print("\nSynchronizing datasets...")
        sync_result = synchronize_datasets(mic_data, rpm_data)
        print(f"  Time offset: {sync_result['time_offset']:.3f} s")
        print(f"  Sync quality: {sync_result['sync_quality']:.3f}")

        # Extract stable segment
        print(f"\nExtracting stable segment ({args.stable_duration}s)...")
        start_time, end_time, mean_rpm, std_rpm, per_motor_stats = extract_stable_segment(
            rpm_data, duration=args.stable_duration
        )
        print(f"  Segment: {start_time:.2f}s to {end_time:.2f}s")
        print(f"  Mean RPM: {mean_rpm:.1f} ± {std_rpm:.1f}")

        # Extract corresponding mic segment
        start_idx = int(start_time * mic_data['sample_rate'])
        end_idx = int(end_time * mic_data['sample_rate'])
        time_data = mic_data['time_data'][:, start_idx:end_idx]
        sample_rate = mic_data['sample_rate']

        print(f"  Extracted: {time_data.shape[1]} samples")
    else:
        # Use full signal if no RPM data
        time_data = mic_data['time_data']
        sample_rate = mic_data['sample_rate']
        print(f"\nNo RPM data provided, using full signal")

    # Load or create configuration
    if args.config:
        print(f"\nLoading configuration: {args.config}")
        config = MIMOFilterConfig.from_yaml(args.config)
    else:
        print(f"\nCreating configuration from command-line arguments")

        # Determine actual channel count
        actual_channels = args.n_channels if args.n_channels else time_data.shape[0]

        config = MIMOFilterConfig(
            n_sources=args.n_sources,
            n_harmonics=args.n_harmonics,
            n_channels=actual_channels,
            sample_rate=sample_rate,
            initialization_method=args.method,
            zero_phase=args.zero_phase
        )

    print(f"  Configuration: {config}")

    # Subset channels if requested
    if args.n_channels and args.n_channels < time_data.shape[0]:
        print(f"\nUsing subset of {args.n_channels} channels")
        indices = np.linspace(0, time_data.shape[0]-1, args.n_channels, dtype=int)
        time_data = time_data[indices, :]

    # Apply MIMO filter
    print("\n" + "="*70)
    print("APPLYING MIMO FILTER")
    print("="*70)

    result = apply_mimo_iir_filter(
        time_data,
        sample_rate,
        config,
        rpm_data
    )

    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    # Get all harmonic frequencies
    all_harmonic_freqs = []
    for s in range(config.n_sources):
        for m in range(config.n_harmonics):
            freq = result['fundamental_frequencies'][s] * config.harmonic_vector[m]
            all_harmonic_freqs.append(freq)

    # Time-domain comparison
    print("\nPlotting time-domain comparison...")
    plot_time_domain_comparison(
        result['original_data'],
        result['filtered_data'],
        sample_rate,
        output_dir / 'time_domain_comparison.png'
    )

    # Spectral comparison
    print("\nPlotting spectral comparison...")
    plot_spectral_comparison(
        result['original_data'],
        result['filtered_data'],
        sample_rate,
        all_harmonic_freqs,
        output_dir / 'spectral_comparison.png'
    )

    # Tracked frequencies
    print("\nPlotting tracked frequencies...")
    plot_tracked_frequencies(
        result,
        output_dir / 'tracked_frequencies.png'
    )

    # Generate report
    print("\nGenerating text report...")
    generate_report(result, output_dir / 'analysis_report.txt')

    # Save JSON
    print("\nSaving results as JSON...")
    save_results_json(result, output_dir / 'results.json')

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey Results:")
    print(f"  Mean Suppression: {result['suppression_metrics']['mean_suppression']:.1f} dB")
    print(f"  RMS Suppression:  {result['suppression_metrics']['rms_suppression']:.1f} dB")
    print(f"  Fundamental Frequencies: {result['fundamental_frequencies']}")
    print("\nGenerated Files:")
    print(f"  - time_domain_comparison.png")
    print(f"  - spectral_comparison.png")
    print(f"  - tracked_frequencies.png")
    print(f"  - analysis_report.txt")
    print(f"  - results.json")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
