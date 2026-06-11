#!/usr/bin/env python3
"""
Data Loading and Synchronization Module

Handles loading of RPM telemetry and microphone array data from HDF5 files,
synchronizes datasets using heartbeat/trigger signals, and exports data
in formats compatible with existing analysis scripts.

Author: MartyMicFly Project
Date: 2025-12-11
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import signal, interpolate
from typing import Dict, Tuple, Optional
import json


def load_rpm_telemetry(h5_file: Path, rpm_correction_factor: float = 7/12) -> Dict:
    """
    Extract RPM data from HDF5 ESC telemetry file.

    Args:
        h5_file: Path to RPM telemetry HDF5 file
        rpm_correction_factor: Multiplicative correction for RPM values (default: 7/12)
                              This corrects for the pole pair configuration in the motor/ESC

    Returns:
        Dictionary containing:
            - timestamp: numpy array of averaged timestamps
            - rpm_avg: numpy array of averaged RPM across all ESCs (corrected)
            - rpm_per_esc: dict with RPM data for each ESC (corrected)
            - trigger_events: dict with trigger timestamps and states
            - esc_count: number of ESCs found
            - metadata: timing, file information, and rpm_correction_factor
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        raise FileNotFoundError(f"RPM telemetry file not found: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        # Extract timing metadata
        metadata = {}
        if 'timing' in f:
            timing_group = f['timing']
            if hasattr(timing_group, 'attrs'):
                metadata['start_time_wall'] = timing_group.attrs.get('start_time_wall', None)
                metadata['start_time_perf'] = timing_group.attrs.get('start_time_perf', None)

        # Extract trigger events
        trigger_events = {}
        if 'trigger' in f:
            trigger_group = f['trigger']
            trigger_events['timestamp'] = trigger_group['timestamp'][:]
            trigger_events['state'] = trigger_group['state'][:]

        # Extract RPM data from all ESCs
        rpm_per_esc = {}
        esc_data_list = []

        if 'esc_telemetry' not in f:
            raise ValueError(f"No 'esc_telemetry' group found in {h5_file}")

        esc_telemetry = f['esc_telemetry']

        for esc_name in ['ESC1', 'ESC2', 'ESC3', 'ESC4']:
            if esc_name in esc_telemetry:
                esc_group = esc_telemetry[esc_name]
                esc_data = {
                    'timestamp': esc_group['timestamp'][:],
                    'rpm': esc_group['rpm'][:],
                    'temperature': esc_group['temperature'][:],
                    'voltage': esc_group['voltage'][:],
                    'current': esc_group['current'][:]
                }
                rpm_per_esc[esc_name] = esc_data
                esc_data_list.append(esc_data)

    if not esc_data_list:
        raise ValueError(f"No ESC data found in {h5_file}")

    # Find common time range across all ESCs
    t_start = max(esc['timestamp'][0] for esc in esc_data_list)
    t_end = min(esc['timestamp'][-1] for esc in esc_data_list)

    # Create common time base with high resolution
    # Use minimum sample rate among ESCs
    sample_rates = [len(esc['timestamp']) / (esc['timestamp'][-1] - esc['timestamp'][0])
                    for esc in esc_data_list]
    target_rate = min(sample_rates) * 1.5  # 1.5x to ensure adequate sampling
    n_samples = int((t_end - t_start) * target_rate)
    timestamps_common = np.linspace(t_start, t_end, n_samples)

    # Interpolate all ESC RPM data to common time base
    rpm_interpolated = []
    for esc_data in esc_data_list:
        interp_func = interpolate.interp1d(
            esc_data['timestamp'],
            esc_data['rpm'],
            kind='linear',
            fill_value='extrapolate'
        )
        rpm_interpolated.append(interp_func(timestamps_common))

    # Average RPM across all ESCs
    rpm_avg = np.mean(rpm_interpolated, axis=0)

    # Apply RPM correction factor
    rpm_avg = rpm_avg * rpm_correction_factor
    for esc_name, esc_data in rpm_per_esc.items():
        esc_data['rpm'] = esc_data['rpm'] * rpm_correction_factor

    metadata['file_path'] = str(h5_file)
    metadata['time_range'] = (float(t_start), float(t_end))
    metadata['duration'] = float(t_end - t_start)
    metadata['rpm_correction_factor'] = rpm_correction_factor

    return {
        'timestamp': timestamps_common,
        'rpm_avg': rpm_avg,
        'rpm_per_esc': rpm_per_esc,
        'trigger_events': trigger_events,
        'esc_count': len(esc_data_list),
        'metadata': metadata
    }


def load_microphone_data(h5_file: Path, exclude_channel_1: bool = True) -> Dict:
    """
    Load microphone array time-domain data from HDF5 file.

    Args:
        h5_file: Path to microphone array HDF5 file
        exclude_channel_1: If True, exclude Channel 1 (heartbeat signal)

    Returns:
        Dictionary containing:
            - time_data: numpy array [n_channels, n_samples]
            - sample_rate: sampling rate (Hz)
            - n_channels: number of channels (excluding channel 1 if excluded)
            - n_samples: number of time samples
            - duration: recording duration (seconds)
            - heartbeat_signal: Channel 1 data (heartbeat)
            - metadata: environmental conditions and file info
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        raise FileNotFoundError(f"Microphone array file not found: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        # Load time domain data
        if 'time_data' not in f:
            raise ValueError(f"No 'time_data' dataset found in {h5_file}")

        time_data_full = f['time_data'][:]  # Shape: [n_samples, n_channels]

        # Transpose to [n_channels, n_samples] for processing
        time_data_full = time_data_full.T

        # Extract heartbeat signal (Channel 1 = index 0)
        heartbeat_signal = time_data_full[0, :]

        # Extract acoustic data (Channels 2-96 = indices 1-95)
        if exclude_channel_1:
            time_data = time_data_full[1:, :]
        else:
            time_data = time_data_full

        # Extract metadata
        metadata = {}
        if 'metadata' in f:
            meta_group = f['metadata']
            if 'temperature_celsius' in meta_group:
                metadata['temperature_celsius'] = float(meta_group['temperature_celsius'][()])
            if 'humidity_percent' in meta_group:
                metadata['humidity_percent'] = float(meta_group['humidity_percent'][()])
            if 'dewpoint_celsius' in meta_group:
                metadata['dewpoint_celsius'] = float(meta_group['dewpoint_celsius'][()])

        # sample rate (assumed 51.2 kHz from measurement doc)
        sample_rate = 51200.0  # Hz

        # Try to read from file if available
        if hasattr(f, 'attrs') and 'sample_rate' in f.attrs:
            sample_rate = float(f.attrs['sample_rate'])
        elif hasattr(f, 'attrs') and 'fs' in f.attrs:
            sample_rate = float(f.attrs['fs'])

    n_channels = time_data.shape[0]
    n_samples = time_data.shape[1]
    duration = n_samples / sample_rate

    metadata['file_path'] = str(h5_file)
    metadata['sample_rate'] = sample_rate

    return {
        'time_data': time_data,
        'sample_rate': sample_rate,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'duration': duration,
        'heartbeat_signal': heartbeat_signal,
        'metadata': metadata
    }


def detect_edges(signal_data, threshold=0.5, min_interval=1.0, sample_rate=51200.0):
    """
    Detect rising edges in a signal.

    Args:
        signal_data: 1D numpy array
        threshold: threshold for edge detection (normalized amplitude)
        min_interval: minimum time between edges (seconds). Set to 0 to disable.
        sample_rate: sampling rate (Hz)

    Returns:
        Array of edge timestamps (samples)
    """
    # Normalize signal
    signal_norm = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))

    # Detect crossings
    above_threshold = signal_norm > threshold
    edges = np.where(np.diff(above_threshold.astype(int)) > 0)[0]

    # Filter edges with minimum interval
    if len(edges) > 0 and min_interval > 0.0:
        min_samples = int(min_interval * sample_rate)
        filtered_edges = [edges[0]]
        for edge in edges[1:]:
            if edge - filtered_edges[-1] >= min_samples:
                filtered_edges.append(edge)
        edges = np.array(filtered_edges)

    return edges


def synchronize_datasets(mic_data: Dict, rpm_data: Dict) -> Dict:
    """
    Synchronize microphone and RPM datasets using heartbeat/trigger cross-correlation.

    Args:
        mic_data: Dictionary from load_microphone_data()
        rpm_data: Dictionary from load_rpm_telemetry()

    Returns:
        Dictionary containing:
            - time_offset: time offset to align datasets (seconds)
            - aligned_rpm_timestamps: RPM timestamps aligned to mic time base
            - sync_quality: correlation coefficient (0-1)
            - heartbeat_intervals_mic: intervals between mic heartbeat edges
            - heartbeat_intervals_rpm: intervals between RPM trigger events
            - edge_count_mic: number of edges detected in mic heartbeat
            - edge_count_rpm: number of trigger events in RPM data
    """
    # Extract heartbeat signal from microphone data
    heartbeat_signal = mic_data['heartbeat_signal']
    mic_sample_rate = mic_data['sample_rate']

    # Detect edges in microphone heartbeat signal
    mic_edges = detect_edges(heartbeat_signal, threshold=0.5,
                              min_interval=5.0, sample_rate=mic_sample_rate)
    mic_edge_times = mic_edges / mic_sample_rate

    # Extract trigger events from RPM data
    if 'trigger_events' not in rpm_data or len(rpm_data['trigger_events']) == 0:
        raise ValueError("No trigger events found in RPM data")

    rpm_trigger_times = rpm_data['trigger_events']['timestamp']
    rpm_trigger_states = rpm_data['trigger_events']['state']

    # Find rising edges in trigger (state = 1)
    rising_edges = rpm_trigger_times[rpm_trigger_states == 1]

    if len(rising_edges) == 0:
        raise ValueError("No rising trigger edges found in RPM data")

    if len(mic_edge_times) == 0:
        raise ValueError("No heartbeat edges detected in microphone data")

    # Compute intervals
    if len(mic_edge_times) > 1:
        mic_intervals = np.diff(mic_edge_times)
    else:
        mic_intervals = np.array([])

    if len(rising_edges) > 1:
        rpm_intervals = np.diff(rising_edges)
    else:
        rpm_intervals = np.array([])

    # Find time offset using cross-correlation of edge sequences
    # Try different offsets and find best alignment
    n_edges_to_use = min(len(mic_edge_times), len(rising_edges))

    if n_edges_to_use < 2:
        # Not enough edges for correlation, use simple offset
        time_offset = rising_edges[0] - mic_edge_times[0]
        sync_quality = 0.5  # Low confidence
    else:
        # Use first N edges for correlation
        mic_edges_subset = mic_edge_times[:n_edges_to_use]
        rpm_edges_subset = rising_edges[:n_edges_to_use]

        # Try offsets in a reasonable range
        offset_range = np.linspace(-60, 60, 1000)  # ±60 seconds
        correlations = []

        for offset in offset_range:
            aligned_rpm = rpm_edges_subset + offset
            # Compute similarity (inverse of sum of squared differences)
            diff = np.sum((mic_edges_subset - aligned_rpm)**2)
            correlations.append(-diff)

        correlations = np.array(correlations)
        best_idx = np.argmax(correlations)
        time_offset = offset_range[best_idx]

        # Normalize correlation to 0-1
        sync_quality = np.clip(1.0 / (1.0 + np.abs(correlations[best_idx]) / 100), 0, 1)

    # Align RPM timestamps to microphone time base
    aligned_rpm_timestamps = rpm_data['timestamp'] + time_offset

    return {
        'time_offset': float(time_offset),
        'aligned_rpm_timestamps': aligned_rpm_timestamps,
        'sync_quality': float(sync_quality),
        'heartbeat_intervals_mic': mic_intervals,
        'heartbeat_intervals_rpm': rpm_intervals,
        'edge_count_mic': len(mic_edge_times),
        'edge_count_rpm': len(rising_edges),
        'mic_edge_times': mic_edge_times,
        'rpm_trigger_times': rising_edges
    }


def extract_stable_segment(rpm_data: Dict, duration: float = 30.0,
                           max_cv: float = 0.05) -> Tuple[float, float, float, float, Dict]:
    """
    Find stable hover segment with minimal RPM variance.

    Args:
        rpm_data: Dictionary from load_rpm_telemetry()
        duration: desired segment duration (seconds)
        max_cv: maximum coefficient of variation (std/mean)

    Returns:
        Tuple: (start_time, end_time, mean_rpm, std_rpm, per_motor_stats)
        where per_motor_stats is a dict with 'mean_rpm' and 'std_rpm' arrays for each motor
    """
    timestamps = rpm_data['timestamp']
    rpm = rpm_data['rpm_avg']

    # Window size in samples
    window_duration = duration
    total_duration = timestamps[-1] - timestamps[0]

    if total_duration < window_duration:
        raise ValueError(f"Recording duration ({total_duration:.1f}s) shorter than requested segment ({duration}s)")

    # Sample rate
    sample_rate = len(timestamps) / total_duration
    window_samples = int(window_duration * sample_rate)

    # Slide window and find segment with lowest variance
    best_cv = float('inf')
    best_start_idx = 0

    for start_idx in range(0, len(rpm) - window_samples, int(sample_rate)):  # Check every second
        end_idx = start_idx + window_samples
        rpm_window = rpm[start_idx:end_idx]

        mean_rpm = np.mean(rpm_window)
        std_rpm = np.std(rpm_window)
        cv = std_rpm / mean_rpm if mean_rpm > 0 else float('inf')

        if cv < best_cv:
            best_cv = cv
            best_start_idx = start_idx

    best_end_idx = best_start_idx + window_samples
    start_time = timestamps[best_start_idx]
    end_time = timestamps[best_end_idx]

    rpm_segment = rpm[best_start_idx:best_end_idx]
    mean_rpm = np.mean(rpm_segment)
    std_rpm = np.std(rpm_segment)

    # Calculate per-motor statistics for the same stable segment
    per_motor_mean = []
    per_motor_std = []

    for esc_name, esc_data in rpm_data['rpm_per_esc'].items():
        # Find indices in per-ESC data that correspond to the stable segment
        esc_timestamps = esc_data['timestamp']
        esc_rpm = esc_data['rpm']

        # Find samples within the stable segment time range
        mask = (esc_timestamps >= start_time) & (esc_timestamps <= end_time)
        esc_rpm_segment = esc_rpm[mask]

        if len(esc_rpm_segment) > 0:
            per_motor_mean.append(np.mean(esc_rpm_segment))
            per_motor_std.append(np.std(esc_rpm_segment))
        else:
            per_motor_mean.append(np.nan)
            per_motor_std.append(np.nan)

    per_motor_stats = {
        'mean_rpm': np.array(per_motor_mean),
        'std_rpm': np.array(per_motor_std)
    }

    if best_cv > max_cv:
        print(f"Warning: Best segment CV={best_cv:.3f} exceeds max_cv={max_cv:.3f}")

    return start_time, end_time, mean_rpm, std_rpm, per_motor_stats


def export_rpm_as_txt(rpm_h5_file: Path, output_txt_file: Path,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None):
    """
    Convert HDF5 RPM telemetry to txt format for existing analysis scripts.

    Existing scripts (ego_noise_suppression.py, sound_power_calculation.py)
    expect 2-column text file: [timestamp, rpm_avg]

    Args:
        rpm_h5_file: Path to RPM telemetry HDF5 file
        output_txt_file: Path to output text file
        start_time: Optional start time for segment extraction
        end_time: Optional end time for segment extraction
    """
    # Load RPM data
    rpm_data = load_rpm_telemetry(rpm_h5_file)

    timestamps = rpm_data['timestamp']
    rpm_avg = rpm_data['rpm_avg']

    # Extract segment if specified
    if start_time is not None or end_time is not None:
        if start_time is None:
            start_time = timestamps[0]
        if end_time is None:
            end_time = timestamps[-1]

        mask = (timestamps >= start_time) & (timestamps <= end_time)
        timestamps = timestamps[mask]
        rpm_avg = rpm_avg[mask]

        # Adjust timestamps to start at 0
        timestamps = timestamps - timestamps[0]

    # Save as text file
    output_txt_file = Path(output_txt_file)
    output_txt_file.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_txt_file, np.column_stack([timestamps, rpm_avg]),
               fmt='%.6f %.2f', header='timestamp rpm', comments='')

    print(f"Exported RPM data to: {output_txt_file}")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f} s")
    print(f"  Mean RPM: {np.mean(rpm_avg):.1f}")
    print(f"  Std RPM: {np.std(rpm_avg):.1f}")


def save_synchronization_report(sync_result: Dict, output_file: Path):
    """
    Save synchronization results to JSON file for documentation.

    Args:
        sync_result: Dictionary from synchronize_datasets()
        output_file: Path to output JSON file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    report = {}
    for key, value in sync_result.items():
        if isinstance(value, np.ndarray):
            report[key] = value.tolist()
        else:
            report[key] = value

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Synchronization report saved to: {output_file}")


if __name__ == '__main__':
    # Example usage and testing
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Test data loading and synchronization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using command-line arguments
  python data_loader.py --mic-h5 mic.h5 --rpm-h5 rpm.h5 --output-dir results/

  # Using YAML config file (pull configuration)
  python data_loader.py --config config.yaml --use-pull

  # Using YAML config file (push configuration)
  python data_loader.py --config config.yaml --use-push
        '''
    )
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--use-pull', action='store_true',
                       help='Use pull configuration from YAML (default if neither specified)')
    parser.add_argument('--use-push', action='store_true',
                       help='Use push configuration from YAML')
    parser.add_argument('--mic-h5', help='Microphone array HDF5 file')
    parser.add_argument('--rpm-h5', help='RPM telemetry HDF5 file')
    parser.add_argument('--output-dir', default='test_output', help='Output directory')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Stable segment duration (seconds)')

    args = parser.parse_args()

    # Load from config or use command-line arguments
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Determine which configuration to use (default to pull)
        if args.use_push:
            config_key = 'push_config'
            print("Using PUSH configuration from YAML")
        else:
            config_key = 'pull_config'
            print("Using PULL configuration from YAML")

        mic_h5 = config['comparison'][config_key]['mic_h5']
        rpm_h5 = config['comparison'][config_key]['rpm_h5']
        output_dir = config['comparison'].get('output_dir', 'test_output')
        duration = config['comparison']['analysis_params'].get('stable_duration', 30.0)
    else:
        mic_h5 = args.mic_h5
        rpm_h5 = args.rpm_h5
        output_dir = args.output_dir
        duration = args.duration

    if mic_h5 and rpm_h5:
        print(f"\nProcessing:")
        print(f"  Microphone: {mic_h5}")
        print(f"  RPM: {rpm_h5}")
        print(f"  Output: {output_dir}\n")

        print("Loading microphone data...")
        mic_data = load_microphone_data(mic_h5)
        print(f"  Channels: {mic_data['n_channels']}")
        print(f"  Duration: {mic_data['duration']:.2f} s")
        print(f"  Sample rate: {mic_data['sample_rate']} Hz")

        print("\nLoading RPM telemetry...")
        rpm_data = load_rpm_telemetry(rpm_h5)
        print(f"  ESC count: {rpm_data['esc_count']}")
        print(f"  Duration: {rpm_data['metadata']['duration']:.2f} s")
        print(f"  Mean RPM: {np.mean(rpm_data['rpm_avg']):.1f}")

        print("\nSynchronizing datasets...")
        sync_result = synchronize_datasets(mic_data, rpm_data)
        print(f"  Time offset: {sync_result['time_offset']:.3f} s")
        print(f"  Sync quality: {sync_result['sync_quality']:.3f}")
        print(f"  Mic edges: {sync_result['edge_count_mic']}")
        print(f"  RPM triggers: {sync_result['edge_count_rpm']}")

        print(f"\nFinding stable segment ({duration}s)...")
        start_time, end_time, mean_rpm, std_rpm, per_motor_stats = extract_stable_segment(
            rpm_data, duration=duration
        )
        print(f"  Start time: {start_time:.2f} s")
        print(f"  End time: {end_time:.2f} s")
        print(f"  Mean RPM (avg): {mean_rpm:.1f} ± {std_rpm:.1f}")
        print(f"  CV: {std_rpm/mean_rpm:.3f}")
        print(f"  Per-motor RPM in stable segment:")
        for i, (mean, std) in enumerate(zip(per_motor_stats['mean_rpm'], per_motor_stats['std_rpm']), 1):
            print(f"    Motor {i}: {mean:.1f} ± {std:.1f}")

        # Export for existing scripts
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        export_rpm_as_txt(rpm_h5, output_path / 'rpm_data.txt',
                         start_time=start_time, end_time=end_time)

        save_synchronization_report(sync_result, output_path / 'synchronization_report.json')

        print(f"\nTest complete! Results saved to: {output_path}")
    else:
        print("Error: Please provide measurement files via --config or --mic-h5 and --rpm-h5")
        parser.print_help()


