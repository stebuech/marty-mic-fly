#!/usr/bin/env python3
"""
Sound Power Measurement Using Envelope Surface Method (ISO 3744)

Implements ISO 3744 Class 2 sound power measurement using drone-mounted
microphone arrays with ego-noise suppression.

Usage:
    python sound_power_calculation.py <h5_file> <rpm_file> <array_xml> [options]

Example:
    python sound_power_calculation.py measurement.h5 rpm_data.txt array_8x8.xml \\
        --source-pos 0 0 1.5 --radius 2.0 --freq-range 100 10000 \\
        --output results/sound_power.json

Dependencies:
    - acoular
    - numpy
    - scipy
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import acoular as ac
    from acoular import (
        MaskedTimeSamples, PowerSpectra, MicGeom,
        SteeringVector, BeamformerBase, L_p
    )
except ImportError:
    print("Error: Acoular not installed. Install with: pip install acoular")
    exit(1)

# Import ego-noise suppression from companion script
try:
    from ego_noise_suppression import DroneNotchFilter
except ImportError:
    print("Warning: Could not import DroneNotchFilter. Notch filtering disabled.")
    DroneNotchFilter = None


class PointGrid:
    """Simple point grid for beamforming to specific locations."""
    
    def __init__(self, pos):
        """
        Args:
            pos: 3 x n_points array of positions
        """
        self.pos = pos
        self.size = pos.shape[1]
        
    def gpos(self):
        """Return grid positions."""
        return self.pos


def generate_hemisphere_points(center, radius, n_points=20):
    """
    Generate measurement points on hemisphere following ISO 3744.
    
    Uses golden spiral distribution for uniform coverage.
    
    Args:
        center: 3D position of hemisphere center (array)
        radius: Hemisphere radius (meters)
        n_points: Number of measurement points
    
    Returns:
        n_points x 3 array of positions
    """
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n_points):
        # Golden spiral on hemisphere (y is "up")
        y = 1 - (i / (n_points - 1))  # Range [1, 0]
        r = np.sqrt(1 - y**2)
        theta = 2 * np.pi * i / phi
        
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        
        # Scale to radius and translate to center
        point = radius * np.array([x, y, z]) + center
        points.append(point)
    
    return np.array(points)


def third_octave_bands(f_min, f_max):
    """
    Generate third-octave band center frequencies.
    
    Args:
        f_min: Minimum frequency (Hz)
        f_max: Maximum frequency (Hz)
    
    Returns:
        List of center frequencies
    """
    # Standard third-octave center frequencies
    base_freqs = np.array([
        100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
    ])
    
    # Filter to range
    freqs = base_freqs[(base_freqs >= f_min) & (base_freqs <= f_max)]
    return freqs.tolist()


def measure_spl_at_point(freq_data, mics, point, freq, band=3):
    """
    Measure SPL at a single point using beamforming.
    
    Args:
        freq_data: PowerSpectra object
        mics: MicGeom object
        point: 3D position (array)
        freq: Center frequency (Hz)
        band: Frequency band (1=octave, 3=third-octave)
    
    Returns:
        SPL in dB
    """
    # Create grid for single point
    point_grid = PointGrid(pos=point.reshape(3, 1))
    st = SteeringVector(grid=point_grid, mics=mics)
    
    # Beamform to extract SPL
    bf = BeamformerBase(freq_data=freq_data, steer=st)
    result = bf.synthetic(freq, band)
    
    return L_p(result)[0]


def measure_sound_power_envelope(h5_file, rpm_file, array_xml,
                                  source_position, radius=2.0,
                                  freq_range=(100, 10000),
                                  n_points=20,
                                  use_notch=False):
    """
    Measure sound power using envelope surface method (ISO 3744 Class 2).
    
    Args:
        h5_file: Path to HDF5 measurement file
        rpm_file: Path to RPM telemetry file
        array_xml: Path to microphone array geometry XML
        source_position: 3D position of sound source (array [x, y, z])
        radius: Hemisphere radius (meters)
        freq_range: Tuple (f_min, f_max) frequency range
        n_points: Number of envelope measurement points
        use_notch: Enable adaptive notch filtering
    
    Returns:
        dict with sound power results
    """
    print(f"Sound Power Measurement - ISO 3744")
    print(f"Source position: {source_position}")
    print(f"Hemisphere radius: {radius} m")
    print(f"Measurement points: {n_points}")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    
    # 1. Define envelope measurement points
    print("\nGenerating envelope measurement points...")
    center = np.array(source_position)
    measurement_points = generate_hemisphere_points(center, radius, n_points)
    
    # 2. Load and process measurement data
    print("Loading measurement data...")
    ts = MaskedTimeSamples(name=str(h5_file))
    rpm_data = np.loadtxt(rpm_file)
    
    # Apply notch filtering if enabled
    if use_notch and DroneNotchFilter is not None:
        print("Applying ego-noise suppression...")
        notch_filter = DroneNotchFilter(
            source=ts,
            rpm_data=rpm_data,
            n_rotors=6,
            n_harmonics=10
        )
        ps = PowerSpectra(
            source=notch_filter,
            block_size=4096,
            window='Hanning',
            overlap='50%',
            cached=True
        )
    else:
        print("Processing without notch filtering...")
        ps = PowerSpectra(
            source=ts,
            block_size=4096,
            window='Hanning',
            overlap='50%',
            cached=True
        )
    
    # 3. Get frequency bands
    freq_bands = third_octave_bands(freq_range[0], freq_range[1])
    print(f"Frequency bands: {len(freq_bands)}")
    
    # 4. Load microphone geometry
    mics = MicGeom(from_file=str(array_xml))
    print(f"Microphone array: {mics.num_mics} channels")
    
    # 5. Measure SPL at each envelope point
    print("\nMeasuring SPL at envelope points...")
    spl_measurements = np.zeros((n_points, len(freq_bands)))
    
    for i, point in enumerate(measurement_points):
        print(f"  Point {i+1}/{n_points}: {point}")
        
        for j, fc in enumerate(freq_bands):
            spl = measure_spl_at_point(ps, mics, point, fc, band=3)
            spl_measurements[i, j] = spl
        
        print(f"    SPL range: {np.min(spl_measurements[i]):.1f} - "
              f"{np.max(spl_measurements[i]):.1f} dB")
    
    # 6. Calculate sound power from envelope measurements
    print("\nCalculating sound power...")
    
    # ISO 3744: Lw = Lp_avg + 10*log10(S/S0)
    S = 2 * np.pi * radius**2  # Hemisphere surface area
    S0 = 1.0  # Reference area (1 m²)
    K = 10 * np.log10(S / S0)  # Area correction
    
    # Average SPL across measurement points (energy average)
    Lp_avg = 10 * np.log10(np.mean(10**(spl_measurements / 10), axis=0))
    
    # Sound power level
    Lw = Lp_avg + K
    
    # Overall sound power (A-weighted sum)
    # For simplicity, use unweighted sum across bands
    Lw_total = 10 * np.log10(np.sum(10**(Lw / 10)))
    
    print(f"\nResults:")
    print(f"  Average SPL: {np.mean(Lp_avg):.1f} dB")
    print(f"  Area correction: {K:.1f} dB")
    print(f"  Sound power (total): {Lw_total:.1f} dB")
    
    # 7. Compile results
    results = {
        'sound_power_total': float(Lw_total),
        'sound_power_spectrum': {
            'frequencies': freq_bands,
            'Lw': Lw.tolist(),
            'Lp_avg': Lp_avg.tolist()
        },
        'measurement_points': measurement_points.tolist(),
        'spl_measurements': spl_measurements.tolist(),
        'parameters': {
            'source_position': source_position,
            'radius': radius,
            'n_points': n_points,
            'freq_range': freq_range,
            'hemisphere_area': float(S),
            'area_correction': float(K)
        },
        'standard': 'ISO 3744 Class 2',
        'processing': {
            'h5_file': str(h5_file),
            'rpm_file': str(rpm_file),
            'array_xml': str(array_xml),
            'ego_noise_suppression': use_notch
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Sound power measurement using ISO 3744 envelope surface method'
    )
    parser.add_argument('h5_file', help='HDF5 measurement file')
    parser.add_argument('rpm_file', help='RPM telemetry file')
    parser.add_argument('array_xml', help='Microphone array geometry XML file')
    parser.add_argument('--source-pos', nargs=3, type=float, 
                       default=[0.0, 0.0, 1.5],
                       help='Sound source position x y z (m, default: 0 0 1.5)')
    parser.add_argument('--radius', type=float, default=2.0,
                       help='Hemisphere radius (m, default: 2.0)')
    parser.add_argument('--freq-range', nargs=2, type=float,
                       default=[100, 10000],
                       help='Frequency range min max (Hz, default: 100 10000)')
    parser.add_argument('--n-points', type=int, default=20,
                       help='Number of measurement points (default: 20)')
    parser.add_argument('--no-notch', action='store_true',
                       help='Disable adaptive notch filtering')
    parser.add_argument('--output', required=True,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Measure sound power
    results = measure_sound_power_envelope(
        args.h5_file,
        args.rpm_file,
        args.array_xml,
        source_position=args.source_pos,
        radius=args.radius,
        freq_range=tuple(args.freq_range),
        n_points=args.n_points,
        use_notch=not args.no_notch
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
