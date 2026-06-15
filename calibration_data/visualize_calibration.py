#!/usr/bin/env python3
"""
Throttle-to-RPM Calibration Data Visualization

Visualizes calibration data from throttle_rpm_mapping.py with multiple plots:
- Throttle vs RPM curves (with hysteresis)
- RPM variance and statistics
- Voltage, current, and temperature profiles

Usage:
    python visualize_calibration.py <csv_file_path>
    python visualize_calibration.py calibration_data/throttle_rpm_calibration_20251202_145757.csv
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_calibration_csv(csv_path):
    """
    Load calibration data from CSV file

    Returns:
        dict: Parsed calibration data with metadata
    """
    print(f"Loading calibration data from: {csv_path}")

    # Read metadata from comments
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            else:
                break

    # Read CSV data (skip comments)
    data = {
        'throttle_up': [],
        'throttle_down': [],
        'escs': {}
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(line for line in f if not line.startswith('#'))
        rows = list(reader)

    # Detect number of ESCs
    num_escs = 0
    for key in rows[0].keys():
        if key.startswith('ESC') and key.endswith('_rpm_mean'):
            num_escs += 1

    print(f"  ESCs detected: {num_escs}")
    print(f"  Data points: {len(rows)}")

    # Initialize ESC data structures
    for i in range(1, num_escs + 1):
        esc_id = f'ESC{i}'
        data['escs'][esc_id] = {
            'up': {'throttle': [], 'rpm_mean': [], 'rpm_std': [], 'rpm_min': [], 'rpm_max': [],
                   'sample_count': []},
            'down': {'throttle': [], 'rpm_mean': [], 'rpm_std': [], 'rpm_min': [], 'rpm_max': [],
                     'sample_count': []}
        }

    # Parse data rows
    for row in rows:
        direction = row['direction']
        throttle = float(row['throttle_percent'])

        for i in range(1, num_escs + 1):
            esc_id = f'ESC{i}'
            dir_key = 'up' if direction == 'UP' else 'down'

            data['escs'][esc_id][dir_key]['throttle'].append(throttle)
            data['escs'][esc_id][dir_key]['rpm_mean'].append(float(row[f'{esc_id}_rpm_mean']))
            data['escs'][esc_id][dir_key]['rpm_std'].append(float(row[f'{esc_id}_rpm_std']))
            data['escs'][esc_id][dir_key]['rpm_min'].append(float(row[f'{esc_id}_rpm_min']))
            data['escs'][esc_id][dir_key]['rpm_max'].append(float(row[f'{esc_id}_rpm_max']))
            data['escs'][esc_id][dir_key]['sample_count'].append(int(row[f'{esc_id}_sample_count']))

    # Convert lists to numpy arrays
    for esc_id in data['escs']:
        for direction in ['up', 'down']:
            for key in data['escs'][esc_id][direction]:
                data['escs'][esc_id][direction][key] = np.array(data['escs'][esc_id][direction][key])

    data['metadata'] = metadata
    data['num_escs'] = num_escs

    return data


def plot_throttle_rpm_curves(data, ax):
    """Plot throttle vs RPM curves with UP/DOWN comparison"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    for i, esc_id in enumerate(sorted(data['escs'].keys())):
        esc_data = data['escs'][esc_id]
        color = colors[i]

        # Plot UP direction (solid line)
        ax.plot(esc_data['up']['throttle'], esc_data['up']['rpm_mean'],
                'o-', color=color, linewidth=2, markersize=6,
                label=f'{esc_id} UP', alpha=0.8)

        # Plot DOWN direction (dashed line)
        ax.plot(esc_data['down']['throttle'], esc_data['down']['rpm_mean'],
                's--', color=color, linewidth=2, markersize=5,
                label=f'{esc_id} DOWN', alpha=0.6)

    ax.set_xlabel('Throttle (%)', fontsize=12)
    ax.set_ylabel('RPM', fontsize=12)
    ax.set_title('Throttle vs RPM (Hysteresis Comparison)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc='upper left')


def plot_rpm_variance(data, ax):
    """Plot RPM standard deviation vs throttle"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, esc_id in enumerate(sorted(data['escs'].keys())):
        esc_data = data['escs'][esc_id]
        color = colors[i]

        # Average UP and DOWN std
        throttle = esc_data['up']['throttle']
        rpm_std = (esc_data['up']['rpm_std'] + esc_data['down']['rpm_std'][::-1]) / 2

        ax.plot(throttle, rpm_std, 'o-', color=color, linewidth=2,
                markersize=6, label=esc_id, alpha=0.8)

    ax.set_xlabel('Throttle (%)', fontsize=12)
    ax.set_ylabel('RPM Std Dev', fontsize=12)
    ax.set_title('RPM Stability (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_hysteresis(data, ax):
    """Plot hysteresis (UP vs DOWN RPM difference)"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, esc_id in enumerate(sorted(data['escs'].keys())):
        esc_data = data['escs'][esc_id]
        color = colors[i]

        throttle = esc_data['up']['throttle']
        rpm_up = esc_data['up']['rpm_mean']
        rpm_down = esc_data['down']['rpm_mean'][::-1]

        # Calculate percentage difference
        hysteresis = ((rpm_down - rpm_up) / rpm_up * 100)
        hysteresis[rpm_up == 0] = 0  # Avoid division by zero at 0% throttle

        ax.plot(throttle, hysteresis, 'o-', color=color, linewidth=2,
                markersize=6, label=esc_id, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Throttle (%)', fontsize=12)
    ax.set_ylabel('Hysteresis (%)', fontsize=12)
    ax.set_title('Hysteresis Effect (DOWN - UP)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)




def plot_sample_count(data, ax):
    """Plot sample count vs throttle to assess data quality"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, esc_id in enumerate(sorted(data['escs'].keys())):
        esc_data = data['escs'][esc_id]
        color = colors[i]

        # Average UP and DOWN
        throttle = esc_data['up']['throttle']
        samples = (esc_data['up']['sample_count'] + esc_data['down']['sample_count']) / 2

        ax.plot(throttle, samples, 'o-', color=color, linewidth=2,
                markersize=6, label=esc_id, alpha=0.8)

    ax.set_xlabel('Throttle (%)', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Data Quality (Samples per Point)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def create_visualization(csv_path, output_path=None):
    """
    Create comprehensive calibration visualization

    Args:
        csv_path: Path to calibration CSV file
        output_path: Optional path to save figure (if None, displays interactively)
    """
    # Load data
    data = load_calibration_csv(csv_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Throttle-to-RPM Calibration Analysis\n{Path(csv_path).name}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add metadata text
    metadata_text = []
    for key, value in data['metadata'].items():
        metadata_text.append(f'{key}: {value}')

    fig.text(0.02, 0.98, '\n'.join(metadata_text), fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Create subplots (3 plots total)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, :])  # Throttle vs RPM (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # RPM variance
    ax3 = fig.add_subplot(gs[1, 1])  # Hysteresis

    # Generate plots
    print("\nGenerating plots...")
    plot_throttle_rpm_curves(data, ax1)
    plot_rpm_variance(data, ax2)
    plot_hysteresis(data, ax3)

    # Save or display
    if output_path:
        print(f"\nSaving figure to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("✓ Figure saved")
    else:
        print("\nDisplaying figure...")
        plt.show()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Visualize throttle-to-RPM calibration data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--csv-file',
        type=str,
        default='/home/steffen/MartyMicFly/Code/marty-mic-fly/calibration_data/throttle_rpm_calibration_20251202_160343.csv',
        help='Path to calibration CSV file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for figure (if not specified, displays interactively)'
    )

    args = parser.parse_args()

    # Validate input file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)

    # Create visualization
    try:
        create_visualization(str(csv_path), args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
