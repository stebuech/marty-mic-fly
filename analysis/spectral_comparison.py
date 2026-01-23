#!/usr/bin/env python3
"""
Spectral Analysis and Comparison Module

Performs frequency-domain analysis of acoustic data, extracts blade passing
frequency (BPF) harmonics, and creates comparison visualizations for different
rotor configurations.

Author: MartyMicFly Project
Date: 2025-12-11
"""

import numpy as np
from scipy import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import yaml

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Interactive plots will not be available.")


def compute_power_spectrum(time_data: np.ndarray, sample_rate: float,
                           nperseg: int = 32768, noverlap: Optional[int] = None,
                           channel_subset: Optional[List[int]] = None) -> Dict:
    """
    Compute power spectral density using Welch's method.

    Args:
        time_data: numpy array [n_channels, n_samples]
        sample_rate: sampling rate (Hz)
        nperseg: length of each segment for Welch's method
        noverlap: number of points to overlap between segments
        channel_subset: optional list of channel indices to process

    Returns:
        Dictionary containing:
            - frequencies: frequency array (Hz)
            - psd_avg: averaged PSD across channels (linear scale)
            - psd_dB: averaged PSD in dB
            - psd_per_channel: PSD for each channel
            - n_channels: number of channels processed
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if channel_subset is not None:
        time_data = time_data[channel_subset, :]
    n_channels = time_data.shape[0]

    # Compute PSD for each channel
    psd_list = []
    for ch_idx in range(n_channels):
        frequencies, psd = signal.welch(
            time_data[ch_idx, :],
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            window='hann'
        )
        psd_list.append(psd)

    psd_per_channel = np.array(psd_list)

    # Average PSD across channels (linear average)
    psd_avg = np.mean(psd_per_channel, axis=0)

    # Convert to dB (reference: 1 unit²/Hz)
    psd_dB = 10 * np.log10(psd_avg + 1e-12)  # Add small value to avoid log(0)

    return {
        'frequencies': frequencies,
        'psd_avg': psd_avg,
        'psd_dB': psd_dB,
        'psd_per_channel': psd_per_channel,
        'n_channels': n_channels
    }


def calculate_bpf_harmonics(rpm_avg: float, rpm_per_motor: np.ndarray,
                            n_blades: int = 2, n_harmonics: int = 20,
                            max_freq: float = 25600.0) -> Dict:
    """
    Calculate theoretical BPF harmonic frequencies from RPM data.

    Note: RPM data should already be corrected before passing to this function.

    Args:
        rpm_avg: average RPM across all motors (already corrected)
        rpm_per_motor: array of RPM values for each motor (already corrected)
        n_blades: number of blades per rotor
        n_harmonics: number of harmonics to calculate
        max_freq: maximum frequency to consider (Nyquist frequency)

    Returns:
        Dictionary containing:
            - bpf_avg: average BPF across all motors (Hz)
            - bpf_per_motor: list of BPF for each motor (Hz)
            - rpm_avg: average RPM
            - rpm_per_motor: list of RPM for each motor
            - harmonic_freqs: array of harmonic frequencies based on average BPF (Hz)
            - harmonic_numbers: array of harmonic numbers [1, 2, 3, ...]
    """
    # Calculate BPF per motor
    bpf_per_motor = (rpm_per_motor / 60.0) * n_blades

    # Calculate average BPF
    bpf_avg = (rpm_avg / 60.0) * n_blades

    print(f"  Average RPM: {rpm_avg:.1f}")
    print(f"  Average BPF: {bpf_avg:.2f} Hz")
    print(f"  BPF per motor:")
    for i, (bpf, rpm) in enumerate(zip(bpf_per_motor, rpm_per_motor), 1):
        print(f"    Motor {i}: {bpf:.2f} Hz (RPM: {rpm:.1f})")

    # Calculate harmonic frequencies based on average BPF
    harmonic_freqs = []
    harmonic_numbers = []

    for n in range(1, n_harmonics + 1):
        freq = n * bpf_avg
        if freq > max_freq:
            break
        harmonic_freqs.append(freq)
        harmonic_numbers.append(n)

    print(f"  Harmonics to mark: {len(harmonic_freqs)} (up to {harmonic_freqs[-1]:.0f} Hz)")

    return {
        'bpf_avg': bpf_avg,
        'bpf_per_motor': bpf_per_motor,
        'rpm_avg': rpm_avg,
        'rpm_per_motor': rpm_per_motor,
        'harmonic_freqs': np.array(harmonic_freqs),
        'harmonic_numbers': np.array(harmonic_numbers)
    }


def plot_spectral_comparison(pull_spectrum: Dict, push_spectrum: Dict,
                             pull_harmonics: Dict, push_harmonics: Dict,
                             output_dir: Path, interactive: bool = True):
    """
    Create comparison plots for spectral analysis.

    Creates three-panel layout:
    - Panel 1: Pull spectrum with BPF harmonics marked
    - Panel 2: Push spectrum with BPF harmonics marked
    - Panel 3: Difference (Pull - Push)

    Args:
        pull_spectrum: spectrum dict from compute_power_spectrum()
        push_spectrum: spectrum dict from compute_power_spectrum()
        pull_harmonics: harmonics dict from calculate_bpf_harmonics()
        push_harmonics: harmonics dict from calculate_bpf_harmonics()
        output_dir: directory for output files
        interactive: if True and plotly available, create interactive HTML plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = pull_spectrum['frequencies']
    pull_psd_dB = pull_spectrum['psd_dB']
    push_psd_dB = push_spectrum['psd_dB']

    # Compute difference
    diff_dB = pull_psd_dB - push_psd_dB

    # Create static matplotlib plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Colors for different motors
    motor_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Red, Blue, Green, Purple
    xlim = [50, 8000]
    # Panel 1: Pull spectrum
    axes[0].semilogx(frequencies, pull_psd_dB, 'k-', linewidth=0.8, alpha=0.7, label='Pull spectrum')
    # Mark per-motor BPF harmonics with vertical lines (solid for pull)
    for motor_idx, bpf in enumerate(pull_harmonics['bpf_per_motor']):
        color = motor_colors[motor_idx]
        # Plot harmonics for this motor
        for harm_n in range(1, 21):  # Plot up to 20 harmonics
            freq = bpf * harm_n
            if freq > frequencies[-1]:
                break
            if harm_n == 1:
                axes[0].axvline(freq, color=color, alpha=0.5, linestyle='-', linewidth=0.8,
                              label=f'Motor {motor_idx+1}')
            else:
                axes[0].axvline(freq, color=color, alpha=0.5, linestyle='-', linewidth=0.8)
    axes[0].set_ylabel('PSD (dB)')
    axes[0].set_title(f'Pull Configuration (rotors upward) - Mean RPM: {pull_harmonics["rpm_avg"]:.1f}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlim(xlim)

    # Panel 2: Push spectrum
    axes[1].semilogx(frequencies, push_psd_dB, 'k-', linewidth=0.8, alpha=0.7, label='Push spectrum')
    # Mark per-motor BPF harmonics with vertical lines (dashed for push)
    for motor_idx, bpf in enumerate(push_harmonics['bpf_per_motor']):
        color = motor_colors[motor_idx]
        # Plot harmonics for this motor
        for harm_n in range(1, 21):  # Plot up to 20 harmonics
            freq = bpf * harm_n
            if freq > frequencies[-1]:
                break
            if harm_n == 1:
                axes[1].axvline(freq, color=color, alpha=0.5, linestyle='--', linewidth=0.8,
                              label=f'Motor {motor_idx+1}')
            else:
                axes[1].axvline(freq, color=color, alpha=0.5, linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('PSD (dB)')
    axes[1].set_title(f'Push Configuration (rotors downward) - Mean RPM: {push_harmonics["rpm_avg"]:.1f}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(xlim)

    # Panel 3: Difference
    axes[2].semilogx(frequencies, diff_dB, 'k-', linewidth=1.0, alpha=0.7, label='Difference (Pull - Push)')
    axes[2].axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[2].fill_between(frequencies, 0, diff_dB, where=(diff_dB > 0),
                        color='green', alpha=0.15, label='Pull higher')
    axes[2].fill_between(frequencies, 0, diff_dB, where=(diff_dB < 0),
                        color='red', alpha=0.15, label='Push higher')

    # Mark per-motor harmonics from both configurations
    # Pull harmonics (solid lines)
    for motor_idx, bpf in enumerate(pull_harmonics['bpf_per_motor']):
        color = motor_colors[motor_idx]
        for harm_n in range(1, 21):
            freq = bpf * harm_n
            if freq > frequencies[-1]:
                break
            if harm_n == 1:
                axes[2].axvline(freq, color=color, alpha=0.4, linestyle='-', linewidth=0.6,
                              label=f'Pull M{motor_idx+1}')
            else:
                axes[2].axvline(freq, color=color, alpha=0.4, linestyle='-', linewidth=0.6)

    # Push harmonics (dashed lines)
    for motor_idx, bpf in enumerate(push_harmonics['bpf_per_motor']):
        color = motor_colors[motor_idx]
        for harm_n in range(1, 21):
            freq = bpf * harm_n
            if freq > frequencies[-1]:
                break
            if harm_n == 1:
                axes[2].axvline(freq, color=color, alpha=0.4, linestyle='--', linewidth=0.6,
                              label=f'Push M{motor_idx+1}')
            else:
                axes[2].axvline(freq, color=color, alpha=0.4, linestyle='--', linewidth=0.6)

    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Difference (dB)')
    axes[2].set_title('Spectral Difference (Pull - Push)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=7, ncol=2)
    axes[2].set_xlim(xlim)

    plt.tight_layout()

    # Save static plot
    png_path = output_dir / 'spectral_comparison.png'
    pdf_path = output_dir / 'spectral_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved spectral comparison plots:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

    # Create interactive plotly plot if available
    if interactive and PLOTLY_AVAILABLE:
        fig_plotly = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'Pull Configuration - Mean RPM: {pull_harmonics["rpm_avg"]:.1f}, Mean BPF: {pull_harmonics["bpf_avg"]:.1f} Hz',
                f'Push Configuration - Mean RPM: {push_harmonics["rpm_avg"]:.1f}, Mean BPF: {push_harmonics["bpf_avg"]:.1f} Hz',
                'Spectral Difference (Pull - Push)'
            ),
            vertical_spacing=0.08
        )

        # Pull spectrum
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=pull_psd_dB, mode='lines',
                      name='Pull spectrum', line=dict(color='black', width=1),
                      opacity=0.7, showlegend=True),
            row=1, col=1
        )
        # Add vertical lines for per-motor harmonics (solid for pull)
        for motor_idx, bpf in enumerate(pull_harmonics['bpf_per_motor']):
            color = motor_colors[motor_idx]
            for harm_n in range(1, 21):
                freq = bpf * harm_n
                if freq > frequencies[-1]:
                    break
                fig_plotly.add_vline(x=freq, line_dash="solid", line_color=color,
                                   opacity=0.5, line_width=0.8, row=1, col=1)

        # Push spectrum
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=push_psd_dB, mode='lines',
                      name='Push spectrum', line=dict(color='black', width=1),
                      opacity=0.7, showlegend=True),
            row=2, col=1
        )
        # Add vertical lines for per-motor harmonics (dashed for push)
        for motor_idx, bpf in enumerate(push_harmonics['bpf_per_motor']):
            color = motor_colors[motor_idx]
            for harm_n in range(1, 21):
                freq = bpf * harm_n
                if freq > frequencies[-1]:
                    break
                fig_plotly.add_vline(x=freq, line_dash="dash", line_color=color,
                                   opacity=0.5, line_width=0.8, row=2, col=1)

        # Difference
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=diff_dB, mode='lines',
                      name='Difference (Pull - Push)', line=dict(color='black', width=1),
                      opacity=0.7, showlegend=True),
            row=3, col=1
        )

        # Add vertical lines for per-motor harmonics from both configurations
        # Pull harmonics (solid lines)
        for motor_idx, bpf in enumerate(pull_harmonics['bpf_per_motor']):
            color = motor_colors[motor_idx]
            for harm_n in range(1, 21):
                freq = bpf * harm_n
                if freq > frequencies[-1]:
                    break
                fig_plotly.add_vline(x=freq, line_dash="solid", line_color=color,
                                   opacity=0.4, line_width=0.6, row=3, col=1)

        # Push harmonics (dashed lines)
        for motor_idx, bpf in enumerate(push_harmonics['bpf_per_motor']):
            color = motor_colors[motor_idx]
            for harm_n in range(1, 21):
                freq = bpf * harm_n
                if freq > frequencies[-1]:
                    break
                fig_plotly.add_vline(x=freq, line_dash="dash", line_color=color,
                                   opacity=0.4, line_width=0.6, row=3, col=1)

        # Update axes
        fig_plotly.update_xaxes(type='log', title_text='Frequency (Hz)', row=3, col=1)
        fig_plotly.update_xaxes(type='log', row=1, col=1)
        fig_plotly.update_xaxes(type='log', row=2, col=1)
        fig_plotly.update_yaxes(title_text='PSD (dB)', row=1, col=1)
        fig_plotly.update_yaxes(title_text='PSD (dB)', row=2, col=1)
        fig_plotly.update_yaxes(title_text='Difference (dB)', row=3, col=1)

        fig_plotly.update_layout(
            height=1000,
            title_text='Pull vs Push Rotor Configuration - Spectral Comparison',
            showlegend=True
        )

        html_path = output_dir / 'spectral_comparison.html'
        fig_plotly.write_html(str(html_path))
        print(f"  {html_path}")

    # Save CSV data
    # _save_spectral_data_csv(pull_spectrum, output_dir / 'pull_spectrum.csv')
    # _save_spectral_data_csv(push_spectrum, output_dir / 'push_spectrum.csv')
    #
    # # Save harmonics comparison
    # _save_harmonics_comparison(pull_harmonics, push_harmonics,
    #                            output_dir / 'bpf_harmonics_comparison.csv')
    return fig


def _save_spectral_data_csv(spectrum: Dict, output_file: Path):
    """Save spectrum data to CSV file."""
    df = pd.DataFrame({
        'frequency_Hz': spectrum['frequencies'],
        'psd_dB': spectrum['psd_dB'],
        'psd_linear': spectrum['psd_avg']
    })
    df.to_csv(output_file, index=False)
    print(f"Saved spectrum data: {output_file}")


def _save_harmonics_comparison(pull_harmonics: Dict, push_harmonics: Dict,
                               output_file: Path):
    """Save BPF harmonics comparison to CSV file."""
    # Combine harmonics data
    max_len = max(len(pull_harmonics['harmonic_freqs']), len(push_harmonics['harmonic_freqs']))

    data = {
        'harmonic_number': [],
        'pull_harmonic_freq_Hz': [],
        'push_harmonic_freq_Hz': [],
    }

    for i in range(max_len):
        harm_num = i + 1 if i < max_len else None

        pull_freq = pull_harmonics['harmonic_freqs'][i] if i < len(pull_harmonics['harmonic_freqs']) else np.nan
        push_freq = push_harmonics['harmonic_freqs'][i] if i < len(push_harmonics['harmonic_freqs']) else np.nan

        data['harmonic_number'].append(harm_num)
        data['pull_harmonic_freq_Hz'].append(pull_freq)
        data['push_harmonic_freq_Hz'].append(push_freq)

    df = pd.DataFrame(data)

    # Add summary info at the top
    summary_data = {
        'parameter': ['pull_avg_rpm', 'pull_avg_bpf', 'push_avg_rpm', 'push_avg_bpf'],
        'value': [
            pull_harmonics['rpm_avg'],
            pull_harmonics['bpf_avg'],
            push_harmonics['rpm_avg'],
            push_harmonics['bpf_avg']
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Write summary and harmonics to file
    with open(output_file, 'w') as f:
        f.write("# Summary\n")
        summary_df.to_csv(f, index=False)
        f.write("\n# Per-motor BPF (Pull)\n")
        for i, bpf in enumerate(pull_harmonics['bpf_per_motor'], 1):
            f.write(f"motor_{i},{bpf:.2f}\n")
        f.write("\n# Per-motor BPF (Push)\n")
        for i, bpf in enumerate(push_harmonics['bpf_per_motor'], 1):
            f.write(f"motor_{i},{bpf:.2f}\n")
        f.write("\n# Harmonic Frequencies\n")
        df.to_csv(f, index=False)

    print(f"Saved harmonics comparison: {output_file}")


def analyze_broadband_levels(spectrum: Dict, frequency_bands: List[Tuple[float, float]]) -> Dict:
    """
    Analyze broadband levels in specified frequency bands.

    Args:
        spectrum: spectrum dict from compute_power_spectrum()
        frequency_bands: list of (f_min, f_max) tuples in Hz

    Returns:
        Dictionary with band levels in dB
    """
    frequencies = spectrum['frequencies']
    psd = spectrum['psd_avg']

    band_levels = {}
    for f_min, f_max in frequency_bands:
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        if np.any(mask):
            # Integrate PSD in band
            psd_band = psd[mask]
            # Convert to overall level (energy sum)
            level_dB = 10 * np.log10(np.sum(psd_band) + 1e-12)
            band_levels[f'{f_min}-{f_max}Hz'] = level_dB

    return band_levels


if __name__ == '__main__':
    # Example usage and testing

    from data_loader import load_microphone_data, load_rpm_telemetry, synchronize_datasets, extract_stable_segment

    parser = argparse.ArgumentParser(
        description='Spectral analysis comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using command-line arguments
  python spectral_comparison.py --pull-mic pull.h5 --pull-rpm pull_rpm.h5 \\
                                --push-mic push.h5 --push-rpm push_rpm.h5 \\
                                --output-dir results/spectral

  # Using YAML config file
  python spectral_comparison.py --config config.yaml --output-dir results/spectral
        '''
    )
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--pull-mic', help='Pull configuration mic HDF5 file')
    parser.add_argument('--pull-rpm', help='Pull configuration RPM HDF5 file')
    parser.add_argument('--push-mic', help='Push configuration mic HDF5 file')
    parser.add_argument('--push-rpm', help='Push configuration RPM HDF5 file')
    parser.add_argument('--output-dir', default='spectral_analysis', help='Output directory')
    parser.add_argument('--duration', type=float, help='Stable segment duration (s)')

    args = parser.parse_args()

    # Load from config or use command-line arguments
    config_path = 'my_comparison.yaml'#args.config if args.config else 'my_comparison.yaml'
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        pull_mic = config['comparison']['pull_config']['mic_h5']
        pull_rpm = config['comparison']['pull_config']['rpm_h5']
        push_mic = config['comparison']['push_config']['mic_h5']
        push_rpm = config['comparison']['push_config']['rpm_h5']
        duration = config['comparison']['analysis_params'].get('stable_duration', 30.0)
        if args.output_dir == 'spectral_analysis':  # Use config output if default not changed
            output_dir = Path(config['comparison']['output_dir']) / 'spectral_analysis'
        else:
            output_dir = Path(args.output_dir)
    else:
        if not all([args.pull_mic, args.pull_rpm, args.push_mic, args.push_rpm]):
            parser.error("Either --config or all of --pull-mic, --pull-rpm, --push-mic, --push-rpm are required")
        pull_mic = args.pull_mic
        pull_rpm = args.pull_rpm
        push_mic = args.push_mic
        push_rpm = args.push_rpm
        duration = args.duration if args.duration else 30.0
        output_dir = Path(args.output_dir)

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load Pull data
    print("\nLoading Pull microphone data...")
    pull_mic_data = load_microphone_data(pull_mic)
    print(f"  Duration: {pull_mic_data['duration']:.2f} s, Channels: {pull_mic_data['n_channels']}")

    print("\nLoading Pull RPM data...")
    pull_rpm_data = load_rpm_telemetry(pull_rpm)
    print(f"  Duration: {pull_rpm_data['metadata']['duration']:.2f} s, ESCs: {pull_rpm_data['esc_count']}")

    # Load Push data
    print("\nLoading Push microphone data...")
    push_mic_data = load_microphone_data(push_mic)
    print(f"  Duration: {push_mic_data['duration']:.2f} s, Channels: {push_mic_data['n_channels']}")

    print("\nLoading Push RPM data...")
    push_rpm_data = load_rpm_telemetry(push_rpm)
    print(f"  Duration: {push_rpm_data['metadata']['duration']:.2f} s, ESCs: {push_rpm_data['esc_count']}")

    print("\n" + "=" * 70)
    print("SYNCHRONIZATION")
    print("=" * 70)

    # Synchronize Pull mic ↔ Pull RPM
    print("\nSynchronizing Pull mic ↔ Pull RPM...")
    pull_sync = synchronize_datasets(pull_mic_data, pull_rpm_data)
    print(f"  Sync quality: {pull_sync['sync_quality']:.3f}")
    print(f"  Time offset: {pull_sync['time_offset']:.3f} s")

    # Synchronize Push mic ↔ Push RPM
    print("\nSynchronizing Push mic ↔ Push RPM...")
    push_sync = synchronize_datasets(push_mic_data, push_rpm_data)
    print(f"  Sync quality: {push_sync['sync_quality']:.3f}")
    print(f"  Time offset: {push_sync['time_offset']:.3f} s")

    # Synchronize Pull mic ↔ Push mic using heartbeat signals
    print("\nSynchronizing Pull mic ↔ Push mic using heartbeat...")
    from data_loader import detect_edges

    pull_heartbeat = pull_mic_data['heartbeat_signal']
    push_heartbeat = push_mic_data['heartbeat_signal']
    sample_rate = pull_mic_data['sample_rate']

    # Detect edges in both heartbeat signals
    pull_edges = detect_edges(pull_heartbeat, threshold=0.5, min_interval=0.8, sample_rate=sample_rate)
    push_edges = detect_edges(push_heartbeat, threshold=0.5, min_interval=0.8, sample_rate=sample_rate)

    pull_edge_times = pull_edges / sample_rate
    push_edge_times = push_edges / sample_rate

    print(f"  Pull edges detected: {len(pull_edge_times)}")
    print(f"  Push edges detected: {len(push_edge_times)}")

    # Find best alignment between pull and push heartbeat edges
    # Try different offsets and find best match
    n_edges = min(len(pull_edge_times), len(push_edge_times), 50)  # Use up to 50 edges

    if n_edges < 3:
        raise ValueError("Not enough heartbeat edges to align pull and push recordings")

    pull_edges_subset = pull_edge_times[:n_edges]
    push_edges_subset = push_edge_times[:n_edges]

    # Compute intervals for pattern matching
    pull_intervals = np.diff(pull_edges_subset)
    push_intervals = np.diff(push_edges_subset)

    # Cross-correlate interval patterns to find time offset
    offset_range = np.linspace(-60, 60, 2000)  # ±60 seconds with fine resolution
    correlations = []

    for offset in offset_range:
        aligned_push = push_edges_subset + offset
        # Compute similarity (negative sum of squared differences)
        diff = np.sum((pull_edges_subset - aligned_push)**2)
        correlations.append(-diff)

    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    pull_push_offset = offset_range[best_idx]

    print(f"  Pull ↔ Push time offset: {pull_push_offset:.3f} s")
    print(f"  (Push time + {pull_push_offset:.3f} = Pull time)")

    print("\n" + "=" * 70)
    print("STABLE SEGMENT EXTRACTION")
    print("=" * 70)

    # Find stable segment in Pull configuration
    print("\nFinding stable segment in Pull RPM data...")
    pull_start_rpm, pull_end_rpm, pull_mean_rpm, pull_std_rpm, pull_per_motor = extract_stable_segment(
        pull_rpm_data, duration=duration
    )
    print(f"  RPM time base: {pull_start_rpm:.2f} - {pull_end_rpm:.2f} s")
    print(f"  Mean RPM: {pull_mean_rpm:.1f} ± {pull_std_rpm:.1f}")

    # Convert to Pull mic time base
    pull_start_mic = pull_start_rpm + pull_sync['time_offset']
    pull_end_mic = pull_end_rpm + pull_sync['time_offset']
    print(f"  Pull mic time base: {pull_start_mic:.2f} - {pull_end_mic:.2f} s")

    # Convert to Push mic time base using pull↔push alignment
    push_start_mic = pull_start_mic - pull_push_offset
    push_end_mic = pull_end_mic - pull_push_offset
    print(f"  Push mic time base: {push_start_mic:.2f} - {push_end_mic:.2f} s")

    # Convert to Push RPM time base
    push_start_rpm = push_start_mic - push_sync['time_offset']
    push_end_rpm = push_end_mic - push_sync['time_offset']
    print(f"  Push RPM time base: {push_start_rpm:.2f} - {push_end_rpm:.2f} s")

    # Validate that the aligned segment exists in push data
    if push_start_mic < 0 or push_end_mic > push_mic_data['duration']:
        print(f"\nWarning: Aligned segment in push mic ({push_start_mic:.2f} - {push_end_mic:.2f}) exceeds recording bounds (0 - {push_mic_data['duration']:.2f})")
        print("Falling back to independent stable segments for each configuration.")

        # Fall back to finding independent stable segment in push
        push_start_rpm, push_end_rpm, push_mean_rpm, push_std_rpm, push_per_motor = extract_stable_segment(
            push_rpm_data, duration=duration
        )
        push_start_mic = push_start_rpm + push_sync['time_offset']
        push_end_mic = push_end_rpm + push_sync['time_offset']

        print(f"\nPush stable segment (independent):")
        print(f"  RPM time base: {push_start_rpm:.2f} - {push_end_rpm:.2f} s")
        print(f"  Mic time base: {push_start_mic:.2f} - {push_end_mic:.2f} s")
    else:
        # Compute RPM statistics for the aligned segment in push configuration
        print(f"\nComputing Push RPM statistics for aligned segment...")
        rpm_timestamps = push_rpm_data['timestamp']
        rpm_avg = push_rpm_data['rpm_avg']

        # Find samples within the aligned segment
        mask = (rpm_timestamps >= push_start_rpm) & (rpm_timestamps <= push_end_rpm)
        rpm_segment = rpm_avg[mask]

        if len(rpm_segment) > 0:
            push_mean_rpm = np.mean(rpm_segment)
            push_std_rpm = np.std(rpm_segment)

            # Per-motor stats
            per_motor_mean = []
            per_motor_std = []
            for esc_name, esc_data in push_rpm_data['rpm_per_esc'].items():
                esc_timestamps = esc_data['timestamp']
                esc_rpm = esc_data['rpm']
                mask_esc = (esc_timestamps >= push_start_rpm) & (esc_timestamps <= push_end_rpm)
                esc_rpm_segment = esc_rpm[mask_esc]
                if len(esc_rpm_segment) > 0:
                    per_motor_mean.append(np.mean(esc_rpm_segment))
                    per_motor_std.append(np.std(esc_rpm_segment))
                else:
                    per_motor_mean.append(np.nan)
                    per_motor_std.append(np.nan)

            push_per_motor = {
                'mean_rpm': np.array(per_motor_mean),
                'std_rpm': np.array(per_motor_std)
            }

            print(f"  Mean RPM: {push_mean_rpm:.1f} ± {push_std_rpm:.1f}")
        else:
            raise ValueError("No RPM data found in aligned push segment")

    print("\n" + "=" * 70)
    print("EXTRACTING MIC SEGMENTS")
    print("=" * 70)

    # Extract Pull mic segment
    print("\nExtracting Pull mic segment...")
    pull_sample_rate = pull_mic_data['sample_rate']
    pull_start_idx = int(pull_start_mic * pull_sample_rate)
    pull_end_idx = int(pull_end_mic * pull_sample_rate)

    if pull_start_idx < 0 or pull_end_idx > pull_mic_data['n_samples']:
        raise ValueError(f"Pull segment ({pull_start_idx}:{pull_end_idx}) out of bounds (0:{pull_mic_data['n_samples']})")

    pull_time_segment = pull_mic_data['time_data'][:, pull_start_idx:pull_end_idx]
    print(f"  Duration: {pull_time_segment.shape[1] / pull_sample_rate:.2f} s ({pull_time_segment.shape[1]} samples)")

    # Extract Push mic segment
    print("\nExtracting Push mic segment...")
    push_sample_rate = push_mic_data['sample_rate']
    push_start_idx = int(push_start_mic * push_sample_rate)
    push_end_idx = int(push_end_mic * push_sample_rate)

    if push_start_idx < 0 or push_end_idx > push_mic_data['n_samples']:
        raise ValueError(f"Push segment ({push_start_idx}:{push_end_idx}) out of bounds (0:{push_mic_data['n_samples']})")

    push_time_segment = push_mic_data['time_data'][:, push_start_idx:push_end_idx]
    print(f"  Duration: {push_time_segment.shape[1] / push_sample_rate:.2f} s ({push_time_segment.shape[1]} samples)")

    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS")
    print("=" * 70)

    print("\nComputing Pull spectrum...")
    pull_spectrum = compute_power_spectrum(pull_time_segment, pull_sample_rate)

    print("\nCalculating Pull BPF harmonics...")
    pull_harmonics = calculate_bpf_harmonics(
        pull_mean_rpm, pull_per_motor['mean_rpm'],
        n_blades=2, n_harmonics=20,
        max_freq=pull_spectrum['frequencies'][-1]
    )
    print(f"Harmonics calculated: {len(pull_harmonics['harmonic_freqs'])}")

    print("\nComputing Push spectrum...")
    push_spectrum = compute_power_spectrum(push_time_segment, push_sample_rate)

    print("\nCalculating Push BPF harmonics...")
    push_harmonics = calculate_bpf_harmonics(
        push_mean_rpm, push_per_motor['mean_rpm'],
        n_blades=2, n_harmonics=20,
        max_freq=push_spectrum['frequencies'][-1]
    )
    print(f"Harmonics calculated: {len(push_harmonics['harmonic_freqs'])}")

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    fig = plot_spectral_comparison(
        pull_spectrum, push_spectrum,
        pull_harmonics, push_harmonics,
        output_dir, interactive=True
    )

    print("\nSpectral analysis complete!")
