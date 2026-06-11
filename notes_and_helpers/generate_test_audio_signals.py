"""
Test Signal Generator for Drone Acoustic Measurements
======================================================
Generates test signals for validating noise suppression algorithms
on drone-mounted microphone arrays.

Target Application: Hexacopter/Quadcopter with RPM range 5500-6200
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, lfilter, chirp
import os
from datetime import datetime
import json


class DroneTestSignalGenerator:
    """
    Generator for acoustic test signals adapted to drone rotor frequencies.

    Parameters
    ----------
    rpm_min : float
        Minimum rotor RPM (default: 5500)
    rpm_max : float
        Maximum rotor RPM (default: 6200)
    n_blades : int
        Number of rotor blades (default: 2)
    fs : int
        Sampling frequency in Hz (default: 48000)
    duration : float
        Signal duration in seconds (default: 60)
    """

    def __init__(self, rpm_min=5500, rpm_max=6200, n_blades=2,
                 fs=48000, duration=60):
        self.rpm_min = rpm_min
        self.rpm_max = rpm_max
        self.n_blades = n_blades
        self.fs = fs
        self.duration = duration

        # Calculate blade passing frequencies (BPF)
        self.f0_min = rpm_min / 60  # Fundamental frequency
        self.f0_max = rpm_max / 60
        self.bpf_min = self.f0_min * n_blades
        self.bpf_max = self.f0_max * n_blades
        self.bpf_center = (self.bpf_min + self.bpf_max) / 2

        # Time vector
        self.t = np.arange(duration * fs) / fs
        self.n_samples = len(self.t)

        # Storage for generated signals
        self.signals = {}

        print(f"Drone Test Signal Generator initialized:")
        print(f"  RPM range: {rpm_min} - {rpm_max}")
        print(f"  BPF range: {self.bpf_min:.1f} - {self.bpf_max:.1f} Hz")
        print(f"  Duration: {duration} s @ {fs} Hz")
        print(f"  Samples: {self.n_samples}")
        print()

    def _normalize(self, signal, target_level=0.7):
        """Normalize signal to target level"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val * target_level
        return signal

    def _bandpass_filter(self, lowcut, highcut, order=4):
        """Design a Butterworth bandpass filter"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def generate_white_noise(self, name='white_noise'):
        """Generate white noise signal"""
        signal = np.random.randn(self.n_samples)
        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_pink_noise(self, name='pink_noise'):
        """Generate pink (1/f) noise signal"""
        white = np.random.randn(self.n_samples)
        # 1/f filter approximation
        pink = lfilter([1], [1, -0.99], white)
        self.signals[name] = self._normalize(pink)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_single_tone(self, frequency, name=None):
        """
        Generate a single tone at specified frequency

        Parameters
        ----------
        frequency : float
            Tone frequency in Hz
        name : str, optional
            Signal name (auto-generated if None)
        """
        if name is None:
            name = f'tone_{frequency:.0f}Hz'

        signal = np.sin(2 * np.pi * frequency * self.t)
        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_tones_near_bpf_harmonics(self, harmonics=[1, 2, 3, 5, 10, 15, 20]):
        """
        Generate tones near BPF harmonics (to test notch filters)

        Parameters
        ----------
        harmonics : list
            List of harmonic numbers to generate
        """
        print("\nGenerating tones near BPF harmonics...")
        for h in harmonics:
            # Slightly offset from exact BPF to test filter selectivity
            freq = self.bpf_center * h + 5  # +5 Hz offset
            name = f'tone_near_{h}xBPF_{freq:.0f}Hz'
            self.generate_single_tone(freq, name)

    def generate_tones_between_bpf_harmonics(self):
        """
        Generate tones between BPF harmonics (should NOT be filtered)
        """
        print("\nGenerating tones between BPF harmonics...")
        # Frequencies strategically placed between harmonics
        between_freqs = [
            (self.bpf_center * 1.5, 'between_1_2_BPF'),
            (self.bpf_center * 2.5, 'between_2_3_BPF'),
            (self.bpf_center * 4.5, 'between_4_5_BPF'),
            (self.bpf_center * 7.5, 'between_7_8_BPF'),
            (self.bpf_center * 12.5, 'between_12_13_BPF'),
        ]

        for freq, name in between_freqs:
            self.generate_single_tone(freq, f'tone_{name}_{freq:.0f}Hz')

    def generate_chirp_linear_bpf(self, name='chirp_bpf_linear'):
        """
        Generate linear chirp over BPF range (simulates RPM change)
        """
        signal = chirp(self.t, self.bpf_min, self.bpf_max, self.duration,
                       method='linear')
        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_chirp_logarithmic(self, f_start=180, f_end=4200,
                                   name='chirp_logarithmic'):
        """
        Generate logarithmic chirp over wide frequency range
        """
        signal = chirp(self.t, f_start, f_end, self.duration,
                       method='logarithmic')
        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_realistic_rpm_ramp(self, name='chirp_realistic_rpm_ramp'):
        """
        Generate realistic RPM ramp (continuous phase for realistic rotor simulation)
        """
        # Linearly changing RPM
        rpm_ramp = np.linspace(self.rpm_min, self.rpm_max, self.n_samples)
        # Calculate instantaneous phase
        phase = np.cumsum(2 * np.pi * (rpm_ramp / 60) * self.n_blades / self.fs)
        signal = np.sin(phase)
        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_multitone_bpf(self, n_harmonics=15, name='multitone_bpf'):
        """
        Generate multitone signal with BPF harmonics

        Parameters
        ----------
        n_harmonics : int
            Number of harmonics to include
        name : str
            Signal name
        """
        signal = np.zeros(self.n_samples)

        # Realistic amplitude decay (approximately 1/f^0.7)
        for h in range(1, n_harmonics + 1):
            amplitude = 1.0 / (h ** 0.7)
            freq = self.bpf_center * h
            signal += amplitude * np.sin(2 * np.pi * freq * self.t)

        self.signals[name] = self._normalize(signal)
        print(f"✓ Generated: {name} ({n_harmonics} harmonics)")
        return self.signals[name]

    def generate_bandpass_noise(self, lowcut, highcut, name=None):
        """
        Generate bandpass-filtered white noise

        Parameters
        ----------
        lowcut : float
            Lower cutoff frequency in Hz
        highcut : float
            Upper cutoff frequency in Hz
        name : str, optional
            Signal name (auto-generated if None)
        """
        if name is None:
            name = f'bandpass_noise_{lowcut:.0f}_{highcut:.0f}Hz'

        # Generate white noise
        noise = np.random.randn(self.n_samples)

        # Apply bandpass filter
        b, a = self._bandpass_filter(lowcut, highcut)
        filtered = filtfilt(b, a, noise)

        self.signals[name] = self._normalize(filtered)
        print(f"✓ Generated: {name}")
        return self.signals[name]

    def generate_bandpass_noise_set(self):
        """Generate set of bandpass-filtered noise signals"""
        print("\nGenerating bandpass noise signals...")

        noise_bands = [
            (150, 250, 'around_bpf'),
            (300, 500, 'low_freq'),
            (800, 1200, 'mid_harmonics'),
            (1800, 2200, 'high_harmonics'),
            (3000, 5000, 'high_freq'),
        ]

        for low, high, label in noise_bands:
            self.generate_bandpass_noise(low, high,
                                         f'bandpass_noise_{label}')

    def generate_combined_signals(self):
        """Generate realistic combined signals (noise + tones)"""
        print("\nGenerating combined signals...")

        # White noise + multitone
        if 'white_noise' not in self.signals:
            self.generate_white_noise()
        if 'multitone_bpf' not in self.signals:
            self.generate_multitone_bpf()

        combined1 = (self.signals['white_noise'] * 0.3 +
                     self.signals['multitone_bpf'] * 0.5)
        self.signals['combined_white_multitone'] = self._normalize(combined1)
        print(f"✓ Generated: combined_white_multitone")

        # Pink noise + single tone near BPF
        if 'pink_noise' not in self.signals:
            self.generate_pink_noise()

        tone_190 = np.sin(2 * np.pi * 190 * self.t)
        combined2 = (self.signals['pink_noise'] * 0.4 +
                     tone_190 * 0.4)
        self.signals['combined_pink_tone_190Hz'] = self._normalize(combined2)
        print(f"✓ Generated: combined_pink_tone_190Hz")

        # Bandpass noise + chirp
        self.generate_bandpass_noise(500, 2000, 'temp_bandpass')
        if 'chirp_bpf_linear' not in self.signals:
            self.generate_chirp_linear_bpf()

        combined3 = (self.signals['temp_bandpass'] * 0.5 +
                     self.signals['chirp_bpf_linear'] * 0.3)
        self.signals['combined_bandpass_chirp'] = self._normalize(combined3)
        print(f"✓ Generated: combined_bandpass_chirp")

        # Remove temporary signal
        if 'temp_bandpass' in self.signals:
            del self.signals['temp_bandpass']

    def generate_all_signals(self):
        """Generate complete test signal set"""
        print("\n" + "=" * 60)
        print("GENERATING COMPLETE TEST SIGNAL SET")
        print("=" * 60)

        # 1. Broadband signals
        print("\n[1/7] Broadband reference signals...")
        self.generate_white_noise()
        self.generate_pink_noise()

        # 2. Tones near BPF harmonics
        print("\n[2/7] Tones near BPF harmonics...")
        self.generate_tones_near_bpf_harmonics()

        # 3. Tones between BPF harmonics
        print("\n[3/7] Tones between BPF harmonics...")
        self.generate_tones_between_bpf_harmonics()

        # 4. Chirps
        print("\n[4/7] Chirp signals...")
        self.generate_chirp_linear_bpf()
        self.generate_chirp_logarithmic()
        self.generate_realistic_rpm_ramp()

        # 5. Multitone
        print("\n[5/7] Multitone signals...")
        self.generate_multitone_bpf(n_harmonics=15)
        self.generate_multitone_bpf(n_harmonics=8, name='multitone_bpf_8harm')

        # 6. Bandpass noise
        print("\n[6/7] Bandpass noise signals...")
        self.generate_bandpass_noise_set()

        # 7. Combined signals
        print("\n[7/7] Combined realistic signals...")
        self.generate_combined_signals()

        print(f"\n{'=' * 60}")
        print(f"TOTAL SIGNALS GENERATED: {len(self.signals)}")
        print(f"{'=' * 60}\n")

    def save_signals(self, output_dir='./test_signals',
                     create_level_variants=True):
        """
        Save all generated signals to WAV files

        Parameters
        ----------
        output_dir : str
            Output directory path
        create_level_variants : bool
            If True, create variants at different SPL levels
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving signals to: {output_dir}")
        print("-" * 60)

        # SPL variants (if requested)
        if create_level_variants:
            spl_levels = [50, 60, 70, 80, 90]  # dB SPL
            # Note: Actual SPL depends on calibration
            # These are just relative levels
            level_factors = {
                50: 0.01,  # -40 dB
                60: 0.03162,  # -30 dB
                70: 0.1,  # -20 dB
                80: 0.3162,  # -10 dB
                90: 1.0,  # 0 dB (reference)
            }
        else:
            spl_levels = [None]
            level_factors = {None: 0.7}

        saved_count = 0

        for signal_name, signal in self.signals.items():
            for spl in spl_levels:
                # Prepare filename
                if spl is not None:
                    filename = f"{signal_name}_{spl}dB.wav"
                    scaled_signal = signal * level_factors[spl]
                else:
                    filename = f"{signal_name}.wav"
                    scaled_signal = signal

                # Save
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, scaled_signal, self.fs)
                saved_count += 1

        print(f"✓ Saved {saved_count} WAV files")

        # Save metadata
        self._save_metadata(output_dir)

        print(f"\n{'=' * 60}")
        print(f"All signals saved successfully!")
        print(f"{'=' * 60}\n")

    def _save_metadata(self, output_dir):
        """Save metadata about generated signals"""

        # Text README
        readme_path = os.path.join(output_dir, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DRONE ACOUSTIC TEST SIGNAL SET\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DRONE PARAMETERS\n")
            f.write("-" * 60 + "\n")
            f.write(f"RPM Range:           {self.rpm_min} - {self.rpm_max} rpm\n")
            f.write(f"Number of Blades:    {self.n_blades}\n")
            f.write(f"Fundamental Freq:    {self.f0_min:.1f} - {self.f0_max:.1f} Hz\n")
            f.write(f"BPF Range:           {self.bpf_min:.1f} - {self.bpf_max:.1f} Hz\n")
            f.write(f"BPF Center:          {self.bpf_center:.1f} Hz\n\n")

            f.write("SIGNAL PARAMETERS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Sampling Rate:       {self.fs} Hz\n")
            f.write(f"Duration:            {self.duration} s\n")
            f.write(f"Samples:             {self.n_samples}\n")
            f.write(f"Bit Depth:           16 bit (WAV format)\n\n")

            f.write("BPF HARMONICS\n")
            f.write("-" * 60 + "\n")
            for h in [1, 2, 3, 5, 10, 15, 20]:
                f_min = self.bpf_min * h
                f_max = self.bpf_max * h
                f.write(f"{h:2d}× BPF: {f_min:7.1f} - {f_max:7.1f} Hz\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"SIGNAL LIST ({len(self.signals)} base signals)\n")
            f.write("=" * 60 + "\n\n")

            # Group signals by type
            signal_types = {
                'Broadband': ['white_noise', 'pink_noise'],
                'Tones near BPF': [s for s in self.signals.keys()
                                   if 'tone_near' in s],
                'Tones between BPF': [s for s in self.signals.keys()
                                      if 'tone_between' in s],
                'Chirps': [s for s in self.signals.keys() if 'chirp' in s],
                'Multitone': [s for s in self.signals.keys() if 'multitone' in s],
                'Bandpass Noise': [s for s in self.signals.keys()
                                   if 'bandpass_noise' in s],
                'Combined': [s for s in self.signals.keys() if 'combined' in s],
            }

            for category, signals in signal_types.items():
                if signals:
                    f.write(f"\n{category}:\n")
                    for sig in sorted(signals):
                        f.write(f"  - {sig}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("USAGE NOTES\n")
            f.write("=" * 60 + "\n\n")
            f.write("Signal Level Variants:\n")
            f.write("  Signals are available at 50, 60, 70, 80, 90 dB SPL levels\n")
            f.write("  Note: Actual SPL depends on loudspeaker calibration\n\n")
            f.write("Recommended Test Sequence:\n")
            f.write("  1. Baseline (no drone): All signals\n")
            f.write("  2. Notch filter tests: Tones near/between BPF, chirps\n")
            f.write("  3. Spatial filter tests: Bandpass noise, combined signals\n")
            f.write("  4. Realistic scenarios: Combined signals, multitone\n\n")

        print(f"✓ Saved: README.txt")

        # JSON metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'parameters': {
                'rpm_min': self.rpm_min,
                'rpm_max': self.rpm_max,
                'n_blades': self.n_blades,
                'bpf_min_hz': self.bpf_min,
                'bpf_max_hz': self.bpf_max,
                'bpf_center_hz': self.bpf_center,
                'sampling_rate_hz': self.fs,
                'duration_s': self.duration,
                'n_samples': self.n_samples,
            },
            'signals': list(self.signals.keys()),
            'n_signals': len(self.signals),
        }

        json_path = os.path.join(output_dir, 'metadata.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved: metadata.json")


def main():
    """Main execution function"""

    print("\n" + "=" * 60)
    print("DRONE ACOUSTIC TEST SIGNAL GENERATOR")
    print("=" * 60 + "\n")

    # Initialize generator
    generator = DroneTestSignalGenerator(
        rpm_min=5500,
        rpm_max=6200,
        n_blades=2,
        fs=48000,
        duration=60
    )

    # Generate all signals
    generator.generate_all_signals()

    # Save to disk
    output_directory = './test_signals'
    generator.save_signals(
        output_dir=output_directory,
        create_level_variants=True  # Creates 50, 60, 70, 80, 90 dB variants
    )

    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"Test signals are ready in: {output_directory}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()