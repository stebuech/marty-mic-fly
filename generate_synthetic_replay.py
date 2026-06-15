#!/usr/bin/env python3
"""
Generate synthetic H5 file for replay_and_record.py testing

Creates an HDF5 file with a controlled RPM profile:
- Ramp up motor 1 from idle (5% throttle equivalent) to target RPM over specified time
- Hold at target RPM
- Ramp down back to idle

Usage:
    python generate_synthetic_replay.py --output-file ./test_replay.h5 --target-rpm 3400
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic RPM profile for replay testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default='./synthetic_replay.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--target-rpm',
        type=float,
        default=3400.0,
        help='Target RPM to reach and hold'
    )
    parser.add_argument(
        '--idle-rpm',
        type=float,
        default=500.0,
        help='Idle RPM (corresponds to ~5%% throttle)'
    )
    parser.add_argument(
        '--ramp-up-time',
        type=float,
        default=3.0,
        help='Ramp-up duration (seconds)'
    )
    parser.add_argument(
        '--hold-time',
        type=float,
        default=10.0,
        help='Hold duration at target RPM (seconds)'
    )
    parser.add_argument(
        '--ramp-down-time',
        type=float,
        default=3.0,
        help='Ramp-down duration (seconds)'
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=100.0,
        help='Sample rate (Hz) for generating synthetic data'
    )
    parser.add_argument(
        '--num-escs',
        type=int,
        default=4,
        help='Number of ESCs in the system'
    )
    parser.add_argument(
        '--active-motor',
        type=int,
        default=1,
        help='Which motor to apply the RPM profile (1-4, others will idle)'
    )

    return parser.parse_args()


def generate_rpm_profile(
    idle_rpm: float,
    target_rpm: float,
    ramp_up_time: float,
    hold_time: float,
    ramp_down_time: float,
    sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate RPM profile with ramp-up, hold, and ramp-down phases

    Parameters
    ----------
    idle_rpm : float
        Starting and ending RPM (corresponds to ~5% throttle)
    target_rpm : float
        Peak RPM to reach and hold
    ramp_up_time : float
        Duration of ramp-up phase (seconds)
    hold_time : float
        Duration of hold phase (seconds)
    ramp_down_time : float
        Duration of ramp-down phase (seconds)
    sample_rate : float
        Sampling frequency (Hz)

    Returns
    -------
    timestamps : np.ndarray
        Time vector (seconds)
    rpms : np.ndarray
        RPM values
    """
    dt = 1.0 / sample_rate

    # Calculate number of samples for each phase
    n_ramp_up = int(ramp_up_time * sample_rate)
    n_hold = int(hold_time * sample_rate)
    n_ramp_down = int(ramp_down_time * sample_rate)

    # Generate timestamps
    total_duration = ramp_up_time + hold_time + ramp_down_time
    timestamps = np.arange(0, total_duration, dt)

    # Initialize RPM array
    rpms = np.zeros_like(timestamps)

    # Phase 1: Ramp up
    t_ramp_up = np.linspace(0, ramp_up_time, n_ramp_up)
    rpm_ramp_up = idle_rpm + (target_rpm - idle_rpm) * (t_ramp_up / ramp_up_time)
    rpms[:n_ramp_up] = rpm_ramp_up

    # Phase 2: Hold at target
    rpms[n_ramp_up:n_ramp_up + n_hold] = target_rpm

    # Phase 3: Ramp down
    t_ramp_down = np.linspace(0, ramp_down_time, n_ramp_down)
    rpm_ramp_down = target_rpm - (target_rpm - idle_rpm) * (t_ramp_down / ramp_down_time)
    rpms[n_ramp_up + n_hold:n_ramp_up + n_hold + n_ramp_down] = rpm_ramp_down

    # Ensure we return exactly the right length
    min_len = min(len(timestamps), len(rpms))

    return timestamps[:min_len], rpms[:min_len]


def create_synthetic_h5(
    output_path: str,
    num_escs: int,
    active_motor: int,
    idle_rpm: float,
    target_rpm: float,
    ramp_up_time: float,
    hold_time: float,
    ramp_down_time: float,
    sample_rate: float
) -> None:
    """
    Create HDF5 file with synthetic ESC telemetry data

    Parameters
    ----------
    output_path : str
        Path to output HDF5 file
    num_escs : int
        Number of ESCs (1-4)
    active_motor : int
        Which motor gets the RPM profile (1-indexed)
    idle_rpm : float
        Idle RPM level
    target_rpm : float
        Target RPM to reach
    ramp_up_time : float
        Ramp-up duration (seconds)
    hold_time : float
        Hold duration (seconds)
    ramp_down_time : float
        Ramp-down duration (seconds)
    sample_rate : float
        Sample rate (Hz)
    """
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC REPLAY FILE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Number of ESCs: {num_escs}")
    print(f"Active motor: ESC{active_motor}")
    print(f"Target RPM: {target_rpm:.0f} RPM")
    print(f"Idle RPM: {idle_rpm:.0f} RPM")
    print(f"Profile: {ramp_up_time}s ramp-up → {hold_time}s hold → {ramp_down_time}s ramp-down")
    print(f"Sample rate: {sample_rate:.0f} Hz")
    print("=" * 70 + "\n")

    # Generate RPM profile
    print("Generating RPM profile...")
    timestamps, active_rpms = generate_rpm_profile(
        idle_rpm, target_rpm, ramp_up_time, hold_time, ramp_down_time, sample_rate
    )

    # Create idle RPM profile (constant idle)
    idle_rpms = np.full_like(timestamps, idle_rpm)

    total_duration = timestamps[-1]
    num_samples = len(timestamps)

    print(f"  ✓ Generated {num_samples} samples")
    print(f"  ✓ Total duration: {total_duration:.2f}s")
    print(f"  ✓ RPM range: {idle_rpm:.0f} → {target_rpm:.0f} → {idle_rpm:.0f} RPM\n")

    # Create HDF5 file
    print("Creating HDF5 file...")

    with h5py.File(output_path, 'w') as f:
        # Create timing group
        timing_grp = f.create_group('timing')
        timing_grp.attrs['start_time_wall'] = 1704067200.0  # Arbitrary Unix timestamp
        timing_grp.attrs['start_time_perf'] = 0.0

        # Create esc_telemetry group
        esc_grp = f.create_group('esc_telemetry')

        # Create datasets for each ESC
        for esc_num in range(1, num_escs + 1):
            esc_id = f'ESC{esc_num}'
            esc_subgrp = esc_grp.create_group(esc_id)

            # Use active profile for specified motor, idle for others
            if esc_num == active_motor:
                rpm_data = active_rpms
                profile_type = "active (ramp-up/hold/ramp-down)"
            else:
                rpm_data = idle_rpms
                profile_type = "idle"

            # Create datasets
            esc_subgrp.create_dataset('timestamp', data=timestamps, compression='gzip')
            esc_subgrp.create_dataset('rpm', data=rpm_data, compression='gzip')

            # Add dummy telemetry data (not used by replay but may be expected)
            esc_subgrp.create_dataset('temperature', data=np.full_like(timestamps, 25.0), compression='gzip')
            esc_subgrp.create_dataset('voltage', data=np.full_like(timestamps, 16.8), compression='gzip')
            esc_subgrp.create_dataset('current', data=np.full_like(timestamps, 1.0), compression='gzip')
            esc_subgrp.create_dataset('consumption', data=np.cumsum(np.full_like(timestamps, 0.001)), compression='gzip')

            # Add attributes
            esc_subgrp.attrs['port'] = f'/dev/ttyAMA{esc_num-1}'
            esc_subgrp.attrs['total_samples'] = num_samples
            esc_subgrp.attrs['avg_sample_rate'] = sample_rate
            esc_subgrp.attrs['sample_rate_std'] = 0.0

            print(f"  ✓ {esc_id}: {profile_type}")

    print(f"\n✓ Synthetic H5 file created: {output_path}")
    print("=" * 70)
    print("\nYou can now use this file with replay_and_record.py:")
    print(f"  python replay_and_record.py --h5-file {output_path} \\")
    print(f"                                --calibration ./calibration_data/throttle_rpm_mapping.csv \\")
    print(f"                                --output-folder ./replay_recordings")
    print("\nIMPORTANT: Ensure you have a valid throttle-RPM calibration file that covers")
    print(f"          the RPM range {idle_rpm:.0f} - {target_rpm:.0f} RPM")
    print("=" * 70 + "\n")


def main():
    """Main execution function"""
    args = parse_arguments()

    # Validate arguments
    if args.active_motor < 1 or args.active_motor > args.num_escs:
        print(f"Error: --active-motor must be between 1 and {args.num_escs}")
        return

    if args.target_rpm <= args.idle_rpm:
        print("Error: --target-rpm must be greater than --idle-rpm")
        return

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic H5 file
    create_synthetic_h5(
        output_path=str(output_path),
        num_escs=args.num_escs,
        active_motor=args.active_motor,
        idle_rpm=args.idle_rpm,
        target_rpm=args.target_rpm,
        ramp_up_time=args.ramp_up_time,
        hold_time=args.hold_time,
        ramp_down_time=args.ramp_down_time,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()