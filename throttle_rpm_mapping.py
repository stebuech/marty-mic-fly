#!/usr/bin/env python3
"""
Throttle-to-RPM Calibration Script

Maps ESC throttle percentages to motor RPM values with hysteresis testing.
Outputs results to CSV format for analysis.

Usage:
    python throttle_rpm_mapping.py [options]

Example:
    python throttle_rpm_mapping.py --max-throttle 30 --throttle-step 5
"""

# CPU Affinity (MUST BE FIRST - before other imports)
import os
os.sched_setaffinity(0, {0})  # Use core 0 for ESC telemetry (no sounddevice conflict)

# Standard Library
import time
import threading
import csv
import argparse
import traceback
from datetime import datetime

# External
import numpy as np

# Local
from esc_throttle_set import MultiESCControler
from daq import Timer, ESCTelemtry

# CONSTANTS
MAX_THROTTLE_SAFETY_LIMIT = 55  # Hard-coded safety limit (percent)
ARMING_DURATION = 5.0  # Seconds to arm ESCs
DISARMING_DURATION = 1.0  # Seconds to ensure motors stop
DEFAULT_STABILIZATION_TIME = 4.0  # Seconds per throttle point
STABILIZATION_WAIT = 2.0  # Seconds to wait before data collection
COLLECTION_DURATION = 2.0  # Seconds to collect data


def parse_arguments():
    """Parse and validate command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Throttle-to-RPM Calibration Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hardware configuration
    parser.add_argument(
        '--esc-pins',
        type=int,
        nargs='+',
        default=[18, 19, 20, 21],
        help='GPIO pins for ESCs (BCM numbering)'
    )
    parser.add_argument(
        '--serial-ports',
        type=str,
        nargs='+',
        default=['/dev/ttyAMA0', '/dev/ttyAMA4', '/dev/ttyAMA2', '/dev/ttyAMA3'],
        help='Serial ports for ESC telemetry'
    )
    parser.add_argument(
        '--dshot-speed',
        type=int,
        choices=[150, 300, 600],
        default=300,
        help='DShot protocol speed'
    )
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Serial baudrate for telemetry'
    )
    parser.add_argument(
        '--pole-pairs',
        type=int,
        default=24,
        help='Motor pole pair count for RPM calculation'
    )

    # Calibration configuration
    parser.add_argument(
        '--max-throttle',
        type=int,
        default=55,
        help=f'Maximum throttle percentage (max: {MAX_THROTTLE_SAFETY_LIMIT})'
    )
    parser.add_argument(
        '--throttle-step',
        type=int,
        default=5,
        help='Throttle increment percentage'
    )
    parser.add_argument(
        '--stabilization-time',
        type=float,
        default=DEFAULT_STABILIZATION_TIME,
        help='Seconds to wait at each throttle point'
    )

    # Output configuration
    parser.add_argument(
        '--output-folder',
        type=str,
        default='./calibration_data',
        help='Folder for CSV output'
    )

    # Options
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip connection validation (not recommended)'
    )

    args = parser.parse_args()

    # Validation
    if args.max_throttle > MAX_THROTTLE_SAFETY_LIMIT:
        parser.error(f"Maximum throttle cannot exceed {MAX_THROTTLE_SAFETY_LIMIT}% for safety")

    if len(args.esc_pins) != len(args.serial_ports):
        parser.error("Number of ESC pins must match number of serial ports")

    if args.max_throttle % args.throttle_step != 0:
        print(f"⚠ Warning: throttle-step {args.throttle_step}% does not evenly divide max-throttle {args.max_throttle}%")

    return args


def initialize_hardware(esc_pins, serial_ports, dshot_speed, baudrate, pole_pairs):
    """
    Initialize ESC controllers and telemetry monitors

    Returns:
        tuple: (timer, multi_esc_controller, esc_monitors, telemetry_threads)
    """
    print("\n" + "=" * 70)
    print("INITIALIZING HARDWARE")
    print("=" * 70)

    # Create shared timer for synchronized timestamps
    timer = Timer()
    print(f"Timer initialized (start time: {timer.start_time_wall:.3f})")

    # Initialize multi-ESC controller (starts command thread automatically)
    print(f"\nInitializing {len(esc_pins)} ESC controllers...")
    print(f"  GPIO pins: {esc_pins}")
    print(f"  DShot speed: {dshot_speed}")
    multi_esc = MultiESCControler(esc_pins, dshot_speed=dshot_speed)
    multi_esc.start()
    print("  ESC command stream started")

    # Create telemetry monitors
    print(f"\nInitializing ESC telemetry monitors...")
    print(f"  Serial ports: {serial_ports}")
    print(f"  Baudrate: {baudrate}")
    print(f"  Pole pairs: {pole_pairs}")

    esc_monitors = []
    telemetry_threads = []

    for i, port in enumerate(serial_ports):
        esc_id = f'ESC{i+1}'
        monitor = ESCTelemtry(
            port=port,
            baudrate=baudrate,
            esc_id=esc_id,
            timer=timer,
            pole_count=pole_pairs
        )

        # Start monitoring thread
        thread = threading.Thread(target=monitor.monitor_thread, daemon=True)
        thread.start()

        esc_monitors.append(monitor)
        telemetry_threads.append(thread)
        print(f"  {esc_id}: {port} - thread started")

    # Wait for connections to establish
    print("\nWaiting for connections to establish...")
    time.sleep(0.5)

    print("=" * 70 + "\n")

    return timer, multi_esc, esc_monitors, telemetry_threads


def validate_connections(esc_monitors, timeout=3.0):
    """
    Verify telemetry is being received from all ESCs

    Returns:
        bool: True to proceed, False to abort
    """
    print("=" * 70)
    print("VALIDATING ESC CONNECTIONS")
    print("=" * 70)
    print(f"Checking telemetry (timeout: {timeout}s)...\n")

    start_time = time.time()
    connected = {esc.esc_id: False for esc in esc_monitors}

    # Poll for telemetry from each ESC
    while time.time() - start_time < timeout:
        for esc in esc_monitors:
            if not connected[esc.esc_id]:
                sample = esc.get_latest_sample()
                if sample is not None:
                    connected[esc.esc_id] = True
                    print(f"✓ {esc.esc_id}: Connected (RPM={sample['rpm']:.0f}, V={sample['voltage']:.2f}V)")

        if all(connected.values()):
            break

        time.sleep(0.1)

    # Report results
    print()
    all_connected = all(connected.values())

    if all_connected:
        print("✓ All ESCs connected and receiving telemetry")
    else:
        print("⚠ Warning: Some ESCs not responding:")
        for esc_id, status in connected.items():
            if not status:
                print(f"  ✗ {esc_id}: No telemetry received")

        print("\nPossible causes:")
        print("  - Wrong serial port configuration")
        print("  - ESC not powered or disconnected")
        print("  - Telemetry wire not connected")

        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            return False

    print("=" * 70 + "\n")
    return True


def measure_throttle_point(timer, multi_esc, esc_monitors, throttle_pct, direction):
    """
    Measure RPM at a specific throttle point

    Args:
        timer: Shared Timer instance
        multi_esc: MultiESCControler instance
        esc_monitors: List of ESCTelemtry instances
        throttle_pct: Throttle percentage (0-100)
        direction: 'UP' or 'DOWN' for hysteresis tracking

    Returns:
        dict: Measurement data with statistics
    """
    # Set throttle
    multi_esc.set_all_throttle(throttle_pct)
    timestamp = timer.get_time()

    print(f"  [{direction}] {throttle_pct:3.0f}% throttle... ", end='', flush=True)

    # Wait for motor speed to stabilize
    time.sleep(STABILIZATION_WAIT)

    # Clear telemetry buffers
    for esc in esc_monitors:
        esc.pop_all_data()

    # Collect data
    time.sleep(COLLECTION_DURATION)

    # Retrieve samples
    esc_data = {}
    total_samples = 0

    for esc in esc_monitors:
        samples = esc.pop_all_data()

        if len(samples) > 0:
            # Extract RPM arrays only
            rpms = np.array([s['rpm'] for s in samples])

            # Calculate RPM statistics only
            esc_data[esc.esc_id] = {
                'rpm_mean': float(np.mean(rpms)),
                'rpm_std': float(np.std(rpms)),
                'rpm_min': float(np.min(rpms)),
                'rpm_max': float(np.max(rpms)),
                'sample_count': len(samples)
            }
            total_samples += len(samples)
        else:
            # No data (motor not spinning or connection lost)
            esc_data[esc.esc_id] = {
                'rpm_mean': 0.0,
                'rpm_std': 0.0,
                'rpm_min': 0.0,
                'rpm_max': 0.0,
                'sample_count': 0
            }

    avg_samples = total_samples / len(esc_monitors) if esc_monitors else 0
    print(f"collected {avg_samples:.0f} samples/ESC")

    return {
        'timestamp': timestamp,
        'throttle_percent': throttle_pct,
        'direction': direction,
        'esc_data': esc_data
    }


def run_calibration(timer, multi_esc, esc_monitors, max_throttle, throttle_step,
                    stabilization_time):
    """
    Run complete calibration sequence with hysteresis testing

    Skips 0% throttle measurement (motors not spinning anyway)

    Returns:
        list: Calibration data points
    """
    print("=" * 70)
    print("RUNNING CALIBRATION SEQUENCE")
    print("=" * 70)

    # Generate throttle points (skip 0%)
    throttle_points = list(range(throttle_step, max_throttle + 1, throttle_step))
    total_points = len(throttle_points) * 2  # UP and DOWN
    estimated_time = total_points * stabilization_time + ARMING_DURATION + DISARMING_DURATION

    print(f"Throttle range: {throttle_step}% to {max_throttle}% in {throttle_step}% steps (0% skipped)")
    print(f"Throttle points: {throttle_points}")
    print(f"Stabilization time: {stabilization_time}s per point")
    print(f"Total points: {total_points} ({len(throttle_points)} UP + {len(throttle_points)} DOWN)")
    print(f"Estimated duration: {estimated_time:.0f}s (~{estimated_time/60:.1f} min)")
    print()

    calibration_data = []

    # RAMP UP: throttle_step% → max_throttle%
    print(f"RAMP UP ({throttle_step}% → {max_throttle}%):")
    for throttle in throttle_points:
        data_point = measure_throttle_point(
            timer, multi_esc, esc_monitors, throttle,
            stabilization_time, direction='UP'
        )
        calibration_data.append(data_point)

    print()

    # RAMP DOWN: max_throttle% → throttle_step%
    print(f"RAMP DOWN ({max_throttle}% → {throttle_step}%):")
    for throttle in reversed(throttle_points):
        data_point = measure_throttle_point(
            timer, multi_esc, esc_monitors, throttle,
            stabilization_time, direction='DOWN'
        )
        calibration_data.append(data_point)

    print()
    print("=" * 70 + "\n")

    return calibration_data


def process_calibration_to_averaged(
    input_file: str,
    output_path: str = None,
    extrapolate_from: float = 35.0,
    extrapolate_to: float = 100.0
) -> str:
    """
    Process raw throttle-RPM calibration CSV into averaged calibration curve

    Averages RPM values across all ESCs, combines UP/DOWN directions,
    and extrapolates to target throttle percentage.

    Args:
        input_file: Path to raw throttle_rpm_calibration CSV file
        output_path: Output file path (default: same dir as input with name 'throttle_rpm_mapping.csv')
        extrapolate_from: Starting throttle % for extrapolation slope (default: 35.0)
        extrapolate_to: Target throttle % for extrapolation (default: 100.0)

    Returns:
        str: Path to saved averaged calibration file
    """
    print("\n" + "=" * 70)
    print("PROCESSING CALIBRATION TO AVERAGED CURVE")
    print("=" * 70)

    # Load and parse input CSV
    print(f"Loading: {input_file}")
    metadata = {}

    with open(input_file, 'r') as f:
        # Parse metadata from header comments
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                break
            if ':' in line:
                parts = line[1:].split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                metadata[key] = value

        # Read CSV data
        f.seek(0)
        reader = csv.DictReader(line for line in f if not line.startswith('#'))
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError("CSV file is empty")

    # Identify ESC columns
    esc_ids = [col.replace('_rpm_mean', '') for col in rows[0].keys() if col.endswith('_rpm_mean')]
    print(f"  Found {len(esc_ids)} ESCs: {', '.join(esc_ids)}")
    print(f"  Loaded {len(rows)} data rows")

    # Aggregate data by throttle point and direction
    print("\nAveraging across ESCs and directions...")
    throttle_data = {}

    for row in rows:
        throttle = float(row['throttle_percent'])
        direction = row['direction']

        if throttle not in throttle_data:
            throttle_data[throttle] = {'UP': [], 'DOWN': []}

        # Collect RPM values from all ESCs
        rpms = []
        for esc_id in esc_ids:
            rpm_col = f'{esc_id}_rpm_mean'
            if rpm_col in row:
                rpms.append(float(row[rpm_col]))

        # Average across ESCs for this throttle/direction
        if rpms:
            throttle_data[throttle][direction].append(np.mean(rpms))

    # Average UP and DOWN directions for each throttle point
    throttles = []
    rpms = []

    for throttle in sorted(throttle_data.keys()):
        all_rpms = throttle_data[throttle]['UP'] + throttle_data[throttle]['DOWN']
        if all_rpms:
            throttles.append(throttle)
            rpms.append(np.mean(all_rpms))

    throttles = np.array(throttles)
    rpms = np.array(rpms)

    print(f"  Measured range: {throttles[0]:.1f}% to {throttles[-1]:.1f}%")
    print(f"  RPM range: {rpms[0]:.1f} to {rpms[-1]:.1f}")

    # Extrapolate to target throttle
    if extrapolate_to > throttles[-1]:
        print(f"\nExtrapolating to {extrapolate_to}% throttle...")

        # Find points >= extrapolate_from for slope calculation
        mask = throttles >= extrapolate_from

        if np.sum(mask) < 2:
            print(f"  ⚠ Warning: Not enough points >= {extrapolate_from}% for extrapolation")
            print(f"  Using all available points instead")
            fit_throttles = throttles
            fit_rpms = rpms
        else:
            fit_throttles = throttles[mask]
            fit_rpms = rpms[mask]

        # Linear fit: RPM = slope * throttle + intercept
        coeffs = np.polyfit(fit_throttles, fit_rpms, deg=1)
        slope = coeffs[0]
        intercept = coeffs[1]

        print(f"  Linear fit: {len(fit_throttles)} points from {fit_throttles[0]:.1f}% to {fit_throttles[-1]:.1f}%")
        print(f"  Slope: {slope:.2f} RPM per % throttle")

        # Calculate and append extrapolated point
        extrapolated_rpm = slope * extrapolate_to + intercept
        print(f"  Extrapolated RPM at {extrapolate_to}%: {extrapolated_rpm:.1f}")

        throttles = np.append(throttles, extrapolate_to)
        rpms = np.append(rpms, extrapolated_rpm)

    print(f"\nFinal curve: {len(throttles)} points ({throttles[0]:.1f}% to {throttles[-1]:.1f}%)")

    # Determine output path
    if output_path is None:
        input_dir = os.path.dirname(input_file) or './calibration_data'
        output_path = os.path.join(input_dir, 'throttle_rpm_mapping.csv')

    # Save averaged calibration
    print(f"\nSaving to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        # Write header comments
        f.write("# Averaged Throttle-RPM Calibration\n")
        f.write(f"# Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if metadata:
            f.write("#\n# Original file metadata:\n")
            for key, value in metadata.items():
                f.write(f"#   {key}: {value}\n")

        f.write("#\n")

        # Write CSV data
        writer = csv.writer(f)
        writer.writerow(['throttle_percent', 'rpm_mean'])

        for throttle, rpm in zip(throttles, rpms):
            writer.writerow([f'{throttle:.1f}', f'{rpm:.1f}'])

    print(f"✓ Saved {len(throttles)} data points")
    print("=" * 70 + "\n")

    return output_path


def save_csv_results(calibration_data, output_folder, num_escs, dshot_speed,
                     stabilization_time, max_throttle):
    """
    Save calibration results to CSV file

    Adds synthetic 0% throttle rows (UP and DOWN) with RPM = 0

    Returns:
        str: Full path to saved CSV file
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"throttle_rpm_calibration_{timestamp}.csv"
    filepath = os.path.join(output_folder, filename)

    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    print(f"Output file: {filepath}")

    with open(filepath, 'w', newline='') as csvfile:
        # Write header comments
        csvfile.write(f"# Throttle-to-RPM Calibration Data\n")
        csvfile.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        csvfile.write(f"# ESCs: {num_escs}\n")
        csvfile.write(f"# DShot Speed: {dshot_speed}\n")
        csvfile.write(f"# Stabilization Time: {stabilization_time}s\n")
        csvfile.write(f"# Maximum Throttle: {max_throttle}%\n")
        csvfile.write(f"#\n")

        # Build column headers (RPM data only, no voltage/current/temp)
        headers = ['timestamp', 'direction', 'throttle_percent']

        # Add per-ESC columns (RPM statistics only)
        for i in range(1, num_escs + 1):
            esc_id = f'ESC{i}'
            headers.extend([
                f'{esc_id}_rpm_mean',
                f'{esc_id}_rpm_std',
                f'{esc_id}_rpm_min',
                f'{esc_id}_rpm_max',
                f'{esc_id}_sample_count'
            ])

        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        # Add synthetic 0% throttle row for UP direction (at beginning)
        zero_row_up = {
            'timestamp': '0.000',
            'direction': 'UP',
            'throttle_percent': '0.0'
        }
        for i in range(1, num_escs + 1):
            esc_id = f'ESC{i}'
            zero_row_up[f'{esc_id}_rpm_mean'] = '0.0'
            zero_row_up[f'{esc_id}_rpm_std'] = '0.0'
            zero_row_up[f'{esc_id}_rpm_min'] = '0.0'
            zero_row_up[f'{esc_id}_rpm_max'] = '0.0'
            zero_row_up[f'{esc_id}_sample_count'] = '0'
        writer.writerow(zero_row_up)

        # Write measured data rows
        for point in calibration_data:
            row = {
                'timestamp': f"{point['timestamp']:.3f}",
                'direction': point['direction'],
                'throttle_percent': f"{point['throttle_percent']:.1f}"
            }

            # Add ESC data (RPM only)
            for i in range(1, num_escs + 1):
                esc_id = f'ESC{i}'
                esc_data = point['esc_data'].get(esc_id, {})

                row[f'{esc_id}_rpm_mean'] = f"{esc_data.get('rpm_mean', 0.0):.1f}"
                row[f'{esc_id}_rpm_std'] = f"{esc_data.get('rpm_std', 0.0):.1f}"
                row[f'{esc_id}_rpm_min'] = f"{esc_data.get('rpm_min', 0.0):.1f}"
                row[f'{esc_id}_rpm_max'] = f"{esc_data.get('rpm_max', 0.0):.1f}"
                row[f'{esc_id}_sample_count'] = f"{esc_data.get('sample_count', 0)}"

            writer.writerow(row)

        # Add synthetic 0% throttle row for DOWN direction (at end)
        zero_row_down = {
            'timestamp': f"{calibration_data[-1]['timestamp']:.3f}",
            'direction': 'DOWN',
            'throttle_percent': '0.0'
        }
        for i in range(1, num_escs + 1):
            esc_id = f'ESC{i}'
            zero_row_down[f'{esc_id}_rpm_mean'] = '0.0'
            zero_row_down[f'{esc_id}_rpm_std'] = '0.0'
            zero_row_down[f'{esc_id}_rpm_min'] = '0.0'
            zero_row_down[f'{esc_id}_rpm_max'] = '0.0'
            zero_row_down[f'{esc_id}_sample_count'] = '0'
        writer.writerow(zero_row_down)

    print(f"Saved {len(calibration_data) + 2} data points to CSV (including 0% throttle)")
    print("=" * 70 + "\n")

    return filepath


def cleanup_hardware(multi_esc, esc_monitors):
    """
    Safe shutdown sequence for ESCs and telemetry monitors
    Always runs in finally block to ensure motors stop
    """
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)

    try:
        if multi_esc is not None:
            print("Disarming ESCs...")
            multi_esc.disarm_all()
            time.sleep(DISARMING_DURATION)

            print("Stopping command stream...")
            multi_esc.stop()

            print("Releasing GPIO resources...")
            multi_esc.cleanup()
    except Exception as e:
        print(f"⚠ Error during ESC cleanup: {e}")

    try:
        if esc_monitors:
            print("Stopping telemetry monitors...")
            for esc in esc_monitors:
                esc.stop()
    except Exception as e:
        print(f"⚠ Error during telemetry cleanup: {e}")

    print("Cleanup complete")
    print("=" * 70)


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()

    # Safety prompt
    print("\n" + "=" * 70)
    print("THROTTLE-TO-RPM CALIBRATION SCRIPT")
    print("=" * 70)
    print(f"⚠  SAFETY CHECK")
    print(f"   Motors will spin up to {args.max_throttle}% throttle")
    print(f"   Ensure propellers are REMOVED or area is clear")
    print("=" * 70)

    response = input("\nContinue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Calibration cancelled.")
        return

    # Initialize variables for cleanup
    multi_esc = None
    esc_monitors = []

    try:
        # Initialize hardware
        timer, multi_esc, esc_monitors, telemetry_threads = initialize_hardware(
            args.esc_pins,
            args.serial_ports,
            args.dshot_speed,
            args.baudrate,
            args.pole_pairs
        )

        # Validate connections
        if not args.skip_validation:
            if not validate_connections(esc_monitors):
                print("\nCalibration aborted.")
                return

        # Arm ESCs
        print("=" * 70)
        print("ARMING ESCs")
        print("=" * 70)
        print(f"Sending arming signal for {ARMING_DURATION}s...")
        multi_esc.arm_all(duration=ARMING_DURATION)
        print("ESCs armed and ready")
        print("=" * 70 + "\n")

        # Run calibration
        calibration_data = run_calibration(
            timer,
            multi_esc,
            esc_monitors,
            args.max_throttle,
            args.throttle_step,
            args.stabilization_time
        )

        # Save results
        output_file = save_csv_results(
            calibration_data,
            args.output_folder,
            len(args.esc_pins),
            args.dshot_speed,
            args.stabilization_time,
            args.max_throttle
        )

        print("✓ Calibration complete!")
        print(f"✓ Results saved to: {output_file}")

        # Process to averaged calibration curve
        averaged_file = process_calibration_to_averaged(
            output_file,
            extrapolate_from=35.0,
            extrapolate_to=100.0
        )

        print(f"✓ Averaged calibration saved to: {averaged_file}\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Calibration interrupted by user!")

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        traceback.print_exc()

    finally:
        # ALWAYS cleanup regardless of how we exit
        cleanup_hardware(multi_esc, esc_monitors)


if __name__ == "__main__":
    main()
