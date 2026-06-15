#!/usr/bin/env python3
"""
Sound Power Comparison Module

Wrapper around existing sound_power_calculation.py for comparative analysis
of sound power levels between different rotor configurations using ISO 3744 standard.

Author: MartyMicFly Project
Date: 2025-12-11
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict
import json
import matplotlib.pyplot as plt
import pandas as pd

# Import existing sound power calculation module
# sys.path.insert(0, str(Path(__file__).parent.parent / '.claude/skills/drone-acoustics-analysis/scripts'))

try:
    from analysis.sound_power_calculation import measure_sound_power_envelope
    SOUND_POWER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sound_power_calculation: {e}")
    SOUND_POWER_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def run_sound_power_analysis(mic_h5: Path, rpm_txt: Path, array_xml: Path,
                             config: Dict) -> Dict:
    """
    Run sound power measurement using existing measure_sound_power_envelope() function.

    Args:
        mic_h5: Path to microphone array HDF5 file
        rpm_txt: Path to RPM telemetry text file (timestamp, rpm format)
        array_xml: Path to microphone array geometry XML
        config: Configuration dictionary with keys:
            - source_position: [x, y, z] in meters
            - radius: hemisphere radius in meters
            - n_points: number of measurement points
            - freq_range: (f_min, f_max) tuple in Hz
            - use_notch: boolean for ego-noise suppression
            - n_rotors: number of rotors
            - n_blades: number of blades per rotor

    Returns:
        Dictionary with ISO 3744 sound power results
    """
    if not SOUND_POWER_AVAILABLE:
        raise ImportError("sound_power_calculation module not available")

    print(f"\nRunning sound power analysis...")
    print(f"  Microphone file: {mic_h5}")
    print(f"  RPM file: {rpm_txt}")
    print(f"  Array geometry: {array_xml}")

    # Extract configuration
    source_position = config.get('source_position', [0.0, 0.375, 0.85])
    radius = config.get('radius', 2.0)
    n_points = config.get('n_points', 20)
    freq_range = config.get('freq_range', (100, 10000))
    use_notch = config.get('use_notch', True)

    results = measure_sound_power_envelope(
        h5_file=str(mic_h5),
        rpm_file=str(rpm_txt),
        array_xml=str(array_xml),
        source_position=source_position,
        radius=radius,
        freq_range=freq_range,
        n_points=n_points,
        use_notch=use_notch
    )

    return results


def compare_sound_power(pull_results: Dict, push_results: Dict) -> Dict:
    """
    Compute differences and statistical summary between two sound power measurements.

    Args:
        pull_results: Results dict from run_sound_power_analysis() for pull config
        push_results: Results dict from run_sound_power_analysis() for push config

    Returns:
        Dictionary containing:
            - delta_Lw_total: overall sound power difference (dB)
            - delta_Lw_spectrum: frequency-dependent differences (array)
            - frequencies: frequency array (Hz)
            - pull_Lw_total: total sound power for pull (dB)
            - push_Lw_total: total sound power for push (dB)
            - statistical_summary: dict with statistics
    """
    pull_Lw_total = pull_results['sound_power_total']
    push_Lw_total = push_results['sound_power_total']

    delta_Lw_total = pull_Lw_total - push_Lw_total

    # Frequency-dependent comparison
    pull_spectrum = pull_results['sound_power_spectrum']
    push_spectrum = push_results['sound_power_spectrum']

    pull_freqs = np.array(pull_spectrum['frequencies'])
    push_freqs = np.array(push_spectrum['frequencies'])

    if not np.array_equal(pull_freqs, push_freqs):
        print("Warning: Frequency arrays differ between configurations")

    pull_Lw = np.array(pull_spectrum['Lw'])
    push_Lw = np.array(push_spectrum['Lw'])

    delta_Lw_spectrum = pull_Lw - push_Lw

    # Statistical summary
    statistical_summary = {
        'mean_difference': float(np.mean(delta_Lw_spectrum)),
        'std_difference': float(np.std(delta_Lw_spectrum)),
        'max_difference': float(np.max(delta_Lw_spectrum)),
        'min_difference': float(np.min(delta_Lw_spectrum)),
        'pull_louder_bands': int(np.sum(delta_Lw_spectrum > 0)),
        'push_louder_bands': int(np.sum(delta_Lw_spectrum < 0)),
        'total_bands': len(delta_Lw_spectrum)
    }

    return {
        'delta_Lw_total': float(delta_Lw_total),
        'delta_Lw_spectrum': delta_Lw_spectrum,
        'frequencies': pull_freqs,
        'pull_Lw_total': float(pull_Lw_total),
        'push_Lw_total': float(push_Lw_total),
        'pull_Lw_spectrum': pull_Lw,
        'push_Lw_spectrum': push_Lw,
        'statistical_summary': statistical_summary
    }


def plot_sound_power_comparison(pull_results: Dict, push_results: Dict,
                                comparison: Dict, output_dir: Path,
                                interactive: bool = True):
    """
    Create comparison visualizations for sound power analysis.

    Args:
        pull_results: Results from run_sound_power_analysis() for pull
        push_results: Results from run_sound_power_analysis() for push
        comparison: Results from compare_sound_power()
        output_dir: Directory for output files
        interactive: If True, create interactive HTML plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = comparison['frequencies']
    pull_Lw = comparison['pull_Lw_spectrum']
    push_Lw = comparison['push_Lw_spectrum']
    delta_Lw = comparison['delta_Lw_spectrum']

    # Create matplotlib figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Total sound power bar chart
    ax1 = fig.add_subplot(gs[0, :])
    configs = ['Pull', 'Push']
    total_levels = [comparison['pull_Lw_total'], comparison['push_Lw_total']]
    colors = ['blue', 'orange']
    bars = ax1.bar(configs, total_levels, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Sound Power Level (dB)')
    ax1.set_title('Total Sound Power Comparison (ISO 3744)')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, level in zip(bars, total_levels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{level:.1f} dB', ha='center', va='bottom', fontweight='bold')

    # Add delta annotation
    ax1.text(0.5, max(total_levels) * 0.95, f'Δ = {comparison["delta_Lw_total"]:.1f} dB',
            ha='center', transform=ax1.transData, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Frequency-dependent Lw comparison
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(frequencies, pull_Lw, 'b-o', label='Pull', markersize=4, linewidth=1.5)
    ax2.plot(frequencies, push_Lw, '-o', color='orange', label='Push', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Sound Power Level (dB)')
    ax2.set_title('Frequency-Dependent Sound Power')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Difference spectrum
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(frequencies, delta_Lw, 'purple', marker='o', markersize=4, linewidth=1.5)
    ax3.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax3.fill_between(frequencies, 0, delta_Lw, where=(delta_Lw > 0),
                    color='green', alpha=0.2, label='Pull higher')
    ax3.fill_between(frequencies, 0, delta_Lw, where=(delta_Lw < 0),
                    color='red', alpha=0.2, label='Push higher')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Difference (dB)')
    ax3.set_title('Sound Power Difference (Pull - Push)')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Statistics summary (text box)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    stats = comparison['statistical_summary']
    stats_text = (
        f"Statistical Summary\n"
        f"{'='*30}\n\n"
        f"Total Difference:  {comparison['delta_Lw_total']:+.2f} dB\n\n"
        f"Mean Difference:   {stats['mean_difference']:+.2f} dB\n"
        f"Std Deviation:     {stats['std_difference']:.2f} dB\n"
        f"Max Difference:    {stats['max_difference']:+.2f} dB\n"
        f"Min Difference:    {stats['min_difference']:+.2f} dB\n\n"
        f"Pull Louder:       {stats['pull_louder_bands']}/{stats['total_bands']} bands\n"
        f"Push Louder:       {stats['push_louder_bands']}/{stats['total_bands']} bands"
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transData,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.suptitle('Pull vs Push Rotor Configuration - Sound Power Analysis (ISO 3744)',
                fontsize=14, fontweight='bold')

    # Save static plots
    png_path = output_dir / 'sound_power_comparison.png'
    pdf_path = output_dir / 'sound_power_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved sound power comparison plots:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    plt.close()

    # Create interactive plotly plot
    if interactive and PLOTLY_AVAILABLE:
        fig_plotly = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Frequency-Dependent Sound Power', 'Sound Power Difference'),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # Lw comparison
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=pull_Lw, mode='lines+markers',
                      name='Pull', line=dict(color='blue', width=2),
                      marker=dict(size=6)),
            row=1, col=1
        )
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=push_Lw, mode='lines+markers',
                      name='Push', line=dict(color='orange', width=2),
                      marker=dict(size=6)),
            row=1, col=1
        )

        # Difference
        fig_plotly.add_trace(
            go.Scatter(x=frequencies, y=delta_Lw, mode='lines+markers',
                      name='Difference (Pull - Push)', line=dict(color='purple', width=2),
                      marker=dict(size=6), fill='tozeroy'),
            row=2, col=1
        )

        fig_plotly.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
        fig_plotly.update_xaxes(type="log", row=1, col=1)
        fig_plotly.update_yaxes(title_text="Sound Power Level (dB)", row=1, col=1)
        fig_plotly.update_yaxes(title_text="Difference (dB)", row=2, col=1)

        fig_plotly.update_layout(
            height=800,
            title_text=f'Sound Power Comparison - Total: Pull={comparison["pull_Lw_total"]:.1f} dB, Push={comparison["push_Lw_total"]:.1f} dB (Δ={comparison["delta_Lw_total"]:+.1f} dB)',
            showlegend=True
        )

        html_path = output_dir / 'sound_power_comparison.html'
        fig_plotly.write_html(str(html_path))
        print(f"  {html_path}")

    # Save CSV data
    _save_sound_power_csv(comparison, output_dir / 'sound_power_summary.csv')


def _save_sound_power_csv(comparison: Dict, output_file: Path):
    """Save sound power comparison data to CSV."""
    df = pd.DataFrame({
        'frequency_Hz': comparison['frequencies'],
        'pull_Lw_dB': comparison['pull_Lw_spectrum'],
        'push_Lw_dB': comparison['push_Lw_spectrum'],
        'difference_dB': comparison['delta_Lw_spectrum']
    })
    df.to_csv(output_file, index=False)
    print(f"Saved sound power summary: {output_file}")


if __name__ == '__main__':
    # Example usage
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Sound power comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using command-line arguments
  python sound_power_comparison.py --pull-mic pull.h5 --pull-rpm-txt pull_rpm.txt \\
                                   --push-mic push.h5 --push-rpm-txt push_rpm.txt \\
                                   --array-xml array.xml --output-dir results/sound_power

  # Using YAML config file
  python sound_power_comparison.py --config config.yaml
        '''
    )
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--pull-mic', help='Pull configuration mic HDF5')
    parser.add_argument('--pull-rpm-txt', help='Pull configuration RPM txt')
    parser.add_argument('--push-mic', help='Push configuration mic HDF5')
    parser.add_argument('--push-rpm-txt', help='Push configuration RPM txt')
    parser.add_argument('--array-xml', help='Microphone array geometry XML')
    parser.add_argument('--output-dir', default='sound_power', help='Output directory')

    args = parser.parse_args()

    # Load from config or use command-line arguments
    config_path = args.config
    if config_path:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        pull_mic = config_data['comparison']['pull_config']['mic_h5']
        pull_rpm_txt = config_data['comparison']['pull_config'].get('rpm_txt')
        push_mic = config_data['comparison']['push_config']['mic_h5']
        push_rpm_txt = config_data['comparison']['push_config'].get('rpm_txt')
        array_xml = config_data['comparison'].get('array_xml')

        if args.output_dir == 'sound_power':  # Use config output if default not changed
            output_dir = Path(config_data['comparison']['output_dir']) / 'sound_power'
        else:
            output_dir = Path(args.output_dir)

        # Sound power specific configuration
        sound_power_config = config_data['comparison'].get('sound_power_params', {})
    else:
        if not all([args.pull_mic, args.pull_rpm_txt, args.push_mic, args.push_rpm_txt, args.array_xml]):
            parser.error("Either --config or all of --pull-mic, --pull-rpm-txt, --push-mic, --push-rpm-txt, --array-xml are required")

        pull_mic = args.pull_mic
        pull_rpm_txt = args.pull_rpm_txt
        push_mic = args.push_mic
        push_rpm_txt = args.push_rpm_txt
        array_xml = args.array_xml
        output_dir = Path(args.output_dir)
        sound_power_config = {}

    # Configuration with defaults
    config = {
        'source_position': sound_power_config.get('source_position', [0.0, 0.375, 0.85]),
        'radius': sound_power_config.get('radius', 2.0),
        'n_points': sound_power_config.get('n_points', 20),
        'freq_range': tuple(sound_power_config.get('freq_range', [100, 10000])),
        'use_notch': sound_power_config.get('use_notch', True),
        'n_rotors': sound_power_config.get('n_rotors', 4),
        'n_blades': sound_power_config.get('n_blades', 2)
    }

    print("="*70)
    print("SOUND POWER ANALYSIS: PULL CONFIGURATION")
    print("="*70)

    pull_results = run_sound_power_analysis(
        pull_mic, pull_rpm_txt, array_xml, config
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'pull_sound_power.json', 'w') as f:
        json.dump(pull_results, f, indent=2)

    print("\n" + "="*70)
    print("SOUND POWER ANALYSIS: PUSH CONFIGURATION")
    print("="*70)

    push_results = run_sound_power_analysis(
        push_mic, push_rpm_txt, array_xml, config
    )

    # Save results
    with open(output_dir / 'push_sound_power.json', 'w') as f:
        json.dump(push_results, f, indent=2)

    print("\n" + "="*70)
    print("COMPARING SOUND POWER RESULTS")
    print("="*70)

    comparison = compare_sound_power(pull_results, push_results)

    print(f"\nTotal Sound Power:")
    print(f"  Pull: {comparison['pull_Lw_total']:.1f} dB")
    print(f"  Push: {comparison['push_Lw_total']:.1f} dB")
    print(f"  Difference: {comparison['delta_Lw_total']:+.1f} dB")

    plot_sound_power_comparison(pull_results, push_results, comparison,
                                output_dir, interactive=True)

    print("\nSound power analysis complete!")
