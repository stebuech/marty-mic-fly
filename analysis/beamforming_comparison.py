#!/usr/bin/env python3
"""
Beamforming Spatial Analysis Comparison Module

Wrapper around existing ego_noise_suppression.py for spatial acoustic mapping
and comparison between different rotor configurations.

Author: MartyMicFly Project
Date: 2025-12-11
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import matplotlib.pyplot as plt
from matplotlib import cm

# Import existing ego-noise suppression module
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude/skills/drone-acoustics-analysis/scripts'))

try:
    from ego_noise_suppression import process_drone_measurement
    BEAMFORMING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ego_noise_suppression: {e}")
    BEAMFORMING_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def run_beamforming(mic_h5: Path, rpm_txt: Path, array_xml: Path,
                   freq: float, config: Dict) -> Dict:
    """
    Run beamforming analysis using existing process_drone_measurement() function.

    Args:
        mic_h5: Path to microphone array HDF5 file
        rpm_txt: Path to RPM telemetry text file
        array_xml: Path to microphone array geometry XML
        freq: Center frequency for analysis (Hz)
        config: Configuration dictionary with keys:
            - grid_size: grid extent ±from center (m)
            - grid_increment: grid spacing (m)
            - grid_z: grid distance from array (m)
            - target_band: frequency band (1=octave, 3=third-octave)
            - n_rotors: number of rotors
            - n_blades: number of blades per rotor

    Returns:
        Dictionary with beamforming results:
            - map_dB: 2D beamforming map in dB
            - grid: RectGrid object
            - freq: center frequency
            - band: frequency band
            - max_level: peak level in dB
            - processing_info: metadata
    """
    if not BEAMFORMING_AVAILABLE:
        raise ImportError("ego_noise_suppression module not available")

    print(f"\nRunning beamforming at {freq} Hz...")

    # Extract configuration
    grid_size = config.get('grid_size', 4.0)
    grid_increment = config.get('grid_increment', 0.1)
    grid_z = config.get('grid_z', 3.0)
    target_band = config.get('target_band', 1)

    # Grid parameters
    grid_params = {
        'x_min': -grid_size,
        'x_max': grid_size,
        'y_min': -grid_size,
        'y_max': grid_size,
        'z': grid_z,
        'increment': grid_increment
    }

    results = process_drone_measurement(
        h5_file=str(mic_h5),
        rpm_file=str(rpm_txt),
        array_xml=str(array_xml),
        target_freq=freq,
        target_band=target_band,
        grid_params=grid_params,
        output_dir=None  # Don't save intermediate files
    )

    return results


def compute_source_location(map_dB: np.ndarray, grid) -> tuple:
    """
    Find peak location in beamforming map.

    Args:
        map_dB: 2D beamforming map in dB
        grid: RectGrid object with position information

    Returns:
        Tuple: (x, y, z, peak_level_dB)
    """
    # Find peak
    peak_idx = np.argmax(map_dB)
    peak_idx_2d = np.unravel_index(peak_idx, map_dB.shape)

    # Get grid positions
    grid_pos = grid.gpos()  # 3 x n_points array

    # Convert 2D index to 1D index in grid
    flat_idx = peak_idx_2d[0] * map_dB.shape[1] + peak_idx_2d[1]

    x = grid_pos[0, flat_idx]
    y = grid_pos[1, flat_idx]
    z = grid_pos[2, flat_idx]
    peak_level = map_dB.flat[peak_idx]

    return (float(x), float(y), float(z), float(peak_level))


def plot_side_by_side_maps(pull_map: np.ndarray, push_map: np.ndarray,
                           freq: float, grid, output_dir: Path,
                           drone_position: List[float] = None,
                           interactive: bool = True):
    """
    Create side-by-side beamforming comparison maps.

    Args:
        pull_map: 2D beamforming map for pull configuration (dB)
        push_map: 2D beamforming map for push configuration (dB)
        freq: center frequency (Hz)
        grid: RectGrid object
        output_dir: directory for output files
        drone_position: [x, y, z] position of drone in meters
        interactive: if True, create interactive HTML plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute difference
    diff_map = pull_map - push_map

    # Get grid extent
    grid_pos = grid.gpos()
    x_min, x_max = np.min(grid_pos[0]), np.max(grid_pos[0])
    y_min, y_max = np.min(grid_pos[1]), np.max(grid_pos[1])

    # Standardized color scale for pull and push
    vmin = min(np.min(pull_map), np.min(push_map))
    vmax = max(np.max(pull_map), np.max(push_map))

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pull map
    im1 = axes[0].imshow(pull_map, origin='lower', cmap='jet',
                        extent=[x_min, x_max, y_min, y_max],
                        vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title(f'Pull Configuration - {freq} Hz')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    if drone_position is not None:
        axes[0].plot(drone_position[0], drone_position[1], 'w+',
                    markersize=15, markeredgewidth=2, label='Drone')
        axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label='SPL (dB)')

    # Push map
    im2 = axes[1].imshow(push_map, origin='lower', cmap='jet',
                        extent=[x_min, x_max, y_min, y_max],
                        vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title(f'Push Configuration - {freq} Hz')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    if drone_position is not None:
        axes[1].plot(drone_position[0], drone_position[1], 'w+',
                    markersize=15, markeredgewidth=2, label='Drone')
        axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='SPL (dB)')

    # Difference map
    diff_max = max(abs(np.min(diff_map)), abs(np.max(diff_map)))
    im3 = axes[2].imshow(diff_map, origin='lower', cmap='RdBu_r',
                        extent=[x_min, x_max, y_min, y_max],
                        vmin=-diff_max, vmax=diff_max, aspect='equal')
    axes[2].set_title(f'Difference (Pull - Push) - {freq} Hz')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    if drone_position is not None:
        axes[2].plot(drone_position[0], drone_position[1], 'k+',
                    markersize=15, markeredgewidth=2, label='Drone')
        axes[2].legend()
    plt.colorbar(im3, ax=axes[2], label='Difference (dB)')

    plt.suptitle(f'Beamforming Comparison at {freq} Hz', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save static plots
    png_path = output_dir / 'side_by_side.png'
    pdf_path = output_dir / 'side_by_side.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved beamforming comparison:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    plt.close()

    # Save difference map separately
    fig_diff, ax_diff = plt.subplots(figsize=(8, 7))
    im = ax_diff.imshow(diff_map, origin='lower', cmap='RdBu_r',
                       extent=[x_min, x_max, y_min, y_max],
                       vmin=-diff_max, vmax=diff_max, aspect='equal')
    ax_diff.set_xlabel('X (m)')
    ax_diff.set_ylabel('Y (m)')
    ax_diff.set_title(f'Acoustic Difference (Pull - Push) at {freq} Hz')
    if drone_position is not None:
        ax_diff.plot(drone_position[0], drone_position[1], 'k+',
                    markersize=15, markeredgewidth=2, label='Drone position')
        ax_diff.legend()
    plt.colorbar(im, ax=ax_diff, label='Difference (dB)')
    plt.tight_layout()

    diff_path = output_dir / 'difference_map.png'
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    print(f"  {diff_path}")
    plt.close()

    # Save raw data
    np.save(output_dir / 'pull_map.npy', pull_map)
    np.save(output_dir / 'push_map.npy', push_map)
    np.save(output_dir / 'difference_map.npy', diff_map)

    # Create interactive plotly plot
    if interactive and PLOTLY_AVAILABLE:
        x = np.linspace(x_min, x_max, pull_map.shape[1])
        y = np.linspace(y_min, y_max, pull_map.shape[0])

        fig_plotly = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Pull Configuration', 'Push Configuration', 'Difference'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
        )

        # Pull map
        fig_plotly.add_trace(
            go.Heatmap(z=pull_map, x=x, y=y, colorscale='Jet',
                      zmin=vmin, zmax=vmax, colorbar=dict(x=0.3)),
            row=1, col=1
        )

        # Push map
        fig_plotly.add_trace(
            go.Heatmap(z=push_map, x=x, y=y, colorscale='Jet',
                      zmin=vmin, zmax=vmax, colorbar=dict(x=0.65)),
            row=1, col=2
        )

        # Difference map
        fig_plotly.add_trace(
            go.Heatmap(z=diff_map, x=x, y=y, colorscale='RdBu',
                      zmid=0, zmin=-diff_max, zmax=diff_max, colorbar=dict(x=1.0)),
            row=1, col=3
        )

        # Add drone position markers if provided
        if drone_position is not None:
            for col in [1, 2, 3]:
                fig_plotly.add_trace(
                    go.Scatter(x=[drone_position[0]], y=[drone_position[1]],
                             mode='markers', marker=dict(symbol='x', size=12, color='white'),
                             name='Drone', showlegend=(col == 1)),
                    row=1, col=col
                )

        fig_plotly.update_xaxes(title_text="X (m)")
        fig_plotly.update_yaxes(title_text="Y (m)")
        fig_plotly.update_layout(
            height=500,
            title_text=f'Beamforming Comparison at {freq} Hz',
            showlegend=True
        )

        html_path = output_dir / 'side_by_side.html'
        fig_plotly.write_html(str(html_path))
        print(f"  {html_path}")


def analyze_multiple_frequencies(pull_results_list: List[Dict],
                                 push_results_list: List[Dict],
                                 output_dir: Path):
    """
    Create summary analysis across multiple frequencies.

    Args:
        pull_results_list: list of beamforming results for pull configuration
        push_results_list: list of beamforming results for push configuration
        output_dir: directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = [res['freq'] for res in pull_results_list]
    pull_max_levels = [res['max_level'] for res in pull_results_list]
    push_max_levels = [res['max_level'] for res in push_results_list]

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frequencies, pull_max_levels, 'b-o', label='Pull', markersize=8, linewidth=2)
    ax.plot(frequencies, push_max_levels, '-o', color='orange', label='Push',
           markersize=8, linewidth=2)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak SPL (dB)')
    ax.set_title('Peak Acoustic Levels vs Frequency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    summary_path = output_dir / 'frequency_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved frequency summary: {summary_path}")
    plt.close()

    # Save data
    summary_data = {
        'frequencies_Hz': frequencies,
        'pull_peak_levels_dB': pull_max_levels,
        'push_peak_levels_dB': push_max_levels,
        'differences_dB': [p - ps for p, ps in zip(pull_max_levels, push_max_levels)]
    }

    with open(output_dir / 'frequency_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)


if __name__ == '__main__':
    # Example usage
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Beamforming comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using command-line arguments
  python beamforming_comparison.py --pull-mic pull.h5 --pull-rpm-txt pull_rpm.txt \\
                                   --push-mic push.h5 --push-rpm-txt push_rpm.txt \\
                                   --array-xml array.xml --output-dir results/beamforming

  # Using YAML config file
  python beamforming_comparison.py --config config.yaml
        '''
    )
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--pull-mic', help='Pull configuration mic HDF5')
    parser.add_argument('--pull-rpm-txt', help='Pull configuration RPM txt')
    parser.add_argument('--push-mic', help='Push configuration mic HDF5')
    parser.add_argument('--push-rpm-txt', help='Push configuration RPM txt')
    parser.add_argument('--array-xml', help='Microphone array geometry XML')
    parser.add_argument('--frequencies', nargs='+', type=float,
                       help='Analysis frequencies (Hz)')
    parser.add_argument('--output-dir', default='beamforming', help='Output directory')

    args = parser.parse_args()

    # Load from config or use command-line arguments
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)

        pull_mic = config_data['comparison']['pull_config']['mic_h5']
        pull_rpm_txt = config_data['comparison']['pull_config'].get('rpm_txt')
        push_mic = config_data['comparison']['push_config']['mic_h5']
        push_rpm_txt = config_data['comparison']['push_config'].get('rpm_txt')
        array_xml = config_data['comparison'].get('array_xml')

        if args.output_dir == 'beamforming':  # Use config output if default not changed
            output_base = Path(config_data['comparison']['output_dir']) / 'beamforming'
        else:
            output_base = Path(args.output_dir)

        # Beamforming specific configuration
        beamforming_config = config_data['comparison'].get('beamforming_params', {})
        frequencies = args.frequencies if args.frequencies else beamforming_config.get('frequencies', [500, 1000, 2000, 4000, 8000])
    else:
        if not all([args.pull_mic, args.pull_rpm_txt, args.push_mic, args.push_rpm_txt, args.array_xml]):
            parser.error("Either --config or all of --pull-mic, --pull-rpm-txt, --push-mic, --push-rpm-txt, --array-xml are required")

        pull_mic = args.pull_mic
        pull_rpm_txt = args.pull_rpm_txt
        push_mic = args.push_mic
        push_rpm_txt = args.push_rpm_txt
        array_xml = args.array_xml
        output_base = Path(args.output_dir)
        frequencies = args.frequencies if args.frequencies else [500, 1000, 2000, 4000, 8000]
        beamforming_config = {}

    # Configuration with defaults
    config = {
        'grid_size': beamforming_config.get('grid_size', 4.0),
        'grid_increment': beamforming_config.get('grid_increment', 0.1),
        'grid_z': beamforming_config.get('grid_z', 3.0),
        'target_band': beamforming_config.get('target_band', 1),
        'n_rotors': beamforming_config.get('n_rotors', 4),
        'n_blades': beamforming_config.get('n_blades', 2)
    }

    drone_position = beamforming_config.get('drone_position', [0.0, 0.375, 0.85])

    pull_results_all = []
    push_results_all = []

    for freq in frequencies:
        print("\n" + "="*70)
        print(f"BEAMFORMING AT {freq} Hz")
        print("="*70)

        # Pull configuration
        print("\nProcessing Pull configuration...")
        pull_results = run_beamforming(
            pull_mic, pull_rpm_txt, array_xml, freq, config
        )
        pull_results_all.append(pull_results)

        # Push configuration
        print("\nProcessing Push configuration...")
        push_results = run_beamforming(
            push_mic, push_rpm_txt, array_xml, freq, config
        )
        push_results_all.append(push_results)

        # Create comparison
        freq_dir = output_base / f'{int(freq)}Hz'
        plot_side_by_side_maps(
            pull_results['map_dB'], push_results['map_dB'],
            freq, pull_results['grid'], freq_dir,
            drone_position=drone_position,
            interactive=True
        )

        # Compute source locations
        pull_loc = compute_source_location(pull_results['map_dB'], pull_results['grid'])
        push_loc = compute_source_location(push_results['map_dB'], push_results['grid'])

        print(f"\nPull peak: ({pull_loc[0]:.2f}, {pull_loc[1]:.2f}, {pull_loc[2]:.2f}) m, {pull_loc[3]:.1f} dB")
        print(f"Push peak: ({push_loc[0]:.2f}, {push_loc[1]:.2f}, {push_loc[2]:.2f}) m, {push_loc[3]:.1f} dB")

    # Create frequency summary
    analyze_multiple_frequencies(pull_results_all, push_results_all, output_base)

    print("\nBeamforming comparison complete!")
