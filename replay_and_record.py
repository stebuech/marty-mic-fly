#!/usr/bin/env python3
"""
Replay and Record Helper Script

Replays a previously recorded RPM signal using MultiESCController and records
it back using DAQ. The script coordinates the timing to ensure proper data capture.
Optionally plays an audio file during replay for acoustic measurements.

Usage:
    python replay_and_record.py [options]

Examples:
    # Basic replay and record
    python replay_and_record.py --h5-file ./data/telemetry_data_20240115.h5 \
                                 --calibration ./calibration_data/throttle_rpm_mapping.csv \
                                 --output-folder ./replay_recordings

    # Replay with audio playback (20s after replay starts)
    python replay_and_record.py --h5-file ./data/telemetry_data_20240115.h5 \
                                 --calibration ./calibration_data/throttle_rpm_mapping.csv \
                                 --enable-mic-array \
                                 --audio-file ./test_signals/white_noise_70dB.wav \
                                 --audio-delay 20.0 \
                                 --audio-volume 0.7

    # List available audio devices
    python replay_and_record.py --list-audio-devices
"""

import argparse
import time
import os
import traceback
import threading
from pathlib import Path
from esc_throttle_set import MultiESCControler
from daq import DAQ

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    from scipy import signal as scipy_signal
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def parse_arguments():
    """Parse and validate command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Replay RPM signal and record with DAQ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Replay configuration
    parser.add_argument(
        '--h5-file',
        type=str,
        default='./synthetic_motor1_3400rpm.h5',
        help='Path to HDF5 recording to replay'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default='./calibration_data/throttle_rpm_mapping.csv',
        help='Path to averaged throttle-RPM calibration CSV'
    )
    parser.add_argument(
        '--playback-speed',
        type=float,
        default=1.0,
        help='Playback speed multiplier (0.5 = half speed, 2.0 = double speed)'
    )
    parser.add_argument(
        '--max-throttle',
        type=float,
        default=100.0,
        help='Maximum throttle safety limit (percent)'
    )
    parser.add_argument(
        '--invert-spin',
        action='store_true',
        help='Reverse motor spin direction'
    )

    # ESC configuration
    parser.add_argument(
        '--esc-pins',
        type=int,
        nargs='+',
        default=[18, 19, 20, 21],
        help='GPIO pins for ESCs (BCM numbering)'
    )
    parser.add_argument(
        '--dshot-speed',
        type=int,
        choices=[150, 300, 600],
        default=300,
        help='DShot protocol speed'
    )
    parser.add_argument(
        '--arming-duration',
        type=float,
        default=5.0,
        help='ESC arming duration (seconds)'
    )

    # DAQ configuration
    parser.add_argument(
        '--esc-ports',
        type=str,
        nargs='+',
        default=['/dev/ttyAMA0', '/dev/ttyAMA4', '/dev/ttyAMA2', '/dev/ttyAMA3'],
        help='Serial ports for ESC telemetry'
    )
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Serial baudrate for telemetry'
    )
    parser.add_argument(
        '--enable-mic-array',
        action='store_true',
        help='Enable microphone array recording'
    )
    parser.add_argument(
        '--mic-channels',
        type=int,
        default=16,
        help='Number of microphone channels'
    )
    parser.add_argument(
        '--mic-sample-rate',
        type=int,
        default=48000,
        help='Microphone array sample rate (Hz)'
    )

    # Output configuration
    parser.add_argument(
        '--output-folder',
        type=str,
        default='/home/steffen/MMFDataLogs/replay_recordings',
        help='Output folder for recordings'
    )
    parser.add_argument(
        '--heartbeat-pin',
        type=int,
        default=22,
        help='GPIO pin for heartbeat output'
    )
    parser.add_argument(
        '--log-pin',
        type=int,
        default=23,
        help='GPIO pin to monitor and log heartbeat events (connect to heartbeat-pin with wire)'
    )
    parser.add_argument(
        '--heartbeat-interval',
        type=float,
        default=10.0,
        help='Heartbeat pulse interval (seconds)'
    )

    # Audio playback configuration
    parser.add_argument(
        '--audio-file',
        type=str,
        default=None,#'./test_signals/white_noise_90dB.wav',
        help='Path to audio file to play during replay (optional)'
    )
    parser.add_argument(
        '--audio-delay',
        type=float,
        default=20.0,
        help='Delay in seconds after replay starts before playing audio'
    )
    parser.add_argument(
        '--audio-device',
        type=int,
        default=None,
        help='Audio output device index (use --list-audio-devices to see options)'
    )
    parser.add_argument(
        '--audio-volume',
        type=float,
        default=10,
        help='Audio playback volume (0.0 to 1.0)'
    )
    parser.add_argument(
        '--list-audio-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--audio-only',
        action='store_true',
        default=False,
        help='Play only audio, no rpm replay'
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.h5_file):
        parser.error(f"HDF5 file not found: {args.h5_file}")

    if not os.path.exists(args.calibration):
        parser.error(f"Calibration file not found: {args.calibration}")

    if args.audio_file and not os.path.exists(args.audio_file):
        parser.error(f"Audio file not found: {args.audio_file}")

    # if args.audio_volume < 0.0 or args.audio_volume > 1.0:
    #     parser.error("Audio volume must be between 0.0 and 1.0")

    return args


def play_audio_signal(audio_file, volume=0.7, device=None):
    """
    Play audio signal from file

    Parameters
    ----------
    audio_file : str or Path
        Path to audio file (WAV format)
    volume : float
        Playback volume (0.0 to 1.0)
    device : int, optional
        Audio output device index

    Returns
    -------
    float : Duration of playback in seconds
    """
    if not AUDIO_AVAILABLE:
        raise ImportError("sounddevice and soundfile are required for audio playback. "
                         "Install with: uv add sounddevice soundfile")

    print(f"Loading audio file: {audio_file}")

    # Load audio signal
    signal, fs = sf.read(audio_file)

    # Get file info for diagnostics
    info = sf.info(audio_file)
    print(f"File info: {info.samplerate} Hz, {info.channels} channels, {info.duration:.2f}s")
    print(f"Read signal: shape={signal.shape}, detected fs={fs} Hz")

    # Calculate expected duration
    if signal.ndim == 1:
        num_samples = len(signal)
    else:
        num_samples = signal.shape[0]

    expected_duration = num_samples / fs
    print(f"Signal samples: {num_samples}, expected duration: {expected_duration:.2f}s")

    # Check audio device info
    if device is None:
        device_info = sd.query_devices(kind='output')
        print(f"Default output device: {device_info['name']}, default samplerate={device_info['default_samplerate']} Hz")
    else:
        device_info = sd.query_devices(device)
        print(f"Target device: {device_info['name']}, default samplerate={device_info['default_samplerate']} Hz")

    # Get device sample rate
    device_fs = int(device_info['default_samplerate'])

    # Resample if needed to match device sample rate
    if fs != device_fs:
        print(f"Resampling from {fs} Hz to {device_fs} Hz...")
        # Calculate resampling ratio
        num_samples_out = int(num_samples * device_fs / fs)
        signal = scipy_signal.resample(signal, num_samples_out)
        fs = device_fs
        expected_duration = num_samples_out / fs
        print(f"Resampled signal: {num_samples_out} samples, new duration: {expected_duration:.2f}s")

    # Apply volume
    signal = signal * volume

    # Play audio with explicit samplerate
    print(f"Starting playback: {expected_duration:.1f}s audio @ {fs} Hz, volume={volume:.2f}")
    start_time = time.time()
    sd.play(signal, samplerate=fs, device=device)

    # Wait for playback to finish
    sd.wait()

    actual_duration = time.time() - start_time
    print(f"Actual playback time: {actual_duration:.2f}s")

    return expected_duration


def main():
    """Main execution function"""
    args = parse_arguments()

    # Handle --list-audio-devices
    if args.list_audio_devices:
        if not AUDIO_AVAILABLE:
            print("Error: sounddevice is not available. Install with: uv add sounddevice soundfile")
            return
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return

    print("\n" + "=" * 70)
    print("REPLAY AND RECORD")
    print("=" * 70)
    print(f"Input H5: {args.h5_file}")
    print(f"Calibration: {args.calibration}")
    print(f"Output: {args.output_folder}")
    print(f"Playback speed: {args.playback_speed}x")
    print(f"Invert spin: {args.invert_spin}")
    if args.audio_file:
        print(f"Audio file: {args.audio_file}")
        print(f"Audio delay: {args.audio_delay}s after replay starts")
    print("=" * 70 + "\n")

    # Safety prompt
    print("⚠  SAFETY CHECK")
    print("   Motors will spin according to the recorded RPM profile")
    print(f"   Maximum throttle: {args.max_throttle}%")
    print("   Ensure propellers are REMOVED or drone is fixed in place and area is clear")
    print()

    response = input("Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    multi_esc = None
    daq = None

    try:
        # ========================================
        # STEP 1: Initialize ESC controller
        # ========================================
        print("\n" + "=" * 70)
        print("STEP 1: INITIALIZING ESC CONTROLLER")
        print("=" * 70)

        multi_esc = MultiESCControler(
            gpio_pins=args.esc_pins,
            dshot_speed=args.dshot_speed
        )
        multi_esc.start()
        print(f"✓ ESC controller initialized ({len(args.esc_pins)} ESCs)")
        print("=" * 70 + "\n")

        # ========================================
        # STEP 2: Arm ESCs
        # ========================================
        print("=" * 70)
        print("STEP 2: ARMING ESCs")
        print("=" * 70)
        print(f"Sending arming signal for {args.arming_duration}s...")
        multi_esc.arm_all(duration=args.arming_duration)
        print("✓ ESCs armed and ready")
        print("=" * 70 + "\n")

        # ========================================
        # STEP 3: Initialize DAQ
        # ========================================
        print("=" * 70)
        print("STEP 3: INITIALIZING DAQ")
        print("=" * 70)

        daq = DAQ(
            # Audio
            enable_mic_array=args.enable_mic_array,
            mic_array_channels=args.mic_channels,
            mic_array_sample_rate=args.mic_sample_rate,
            # ESC telemetry
            enable_telemetry=True,
            esc_ports=args.esc_ports,
            baudrate=args.baudrate,
            # Signal monitoring with heartbeat
            enable_signal_monitor=True,
            listen_pin=None,  # No trigger control
            output_pin=args.heartbeat_pin,
            log_pin=args.log_pin,  # Monitor heartbeat output for logging
            heartbeat=True,
            heartbeat_interval=args.heartbeat_interval,
            # Control
            trigger_controlled=False,  # Manual mode
            # Data logging
            enable_logging=True,
            log_folder=args.output_folder,
            flush_interval=1.0
        )

        print(f"✓ DAQ initialized")
        print(f"  - ESC telemetry: {len(args.esc_ports)} channels")
        print(f"  - Mic array: {'enabled' if args.enable_mic_array else 'disabled'}")
        print(f"  - Heartbeat pin: GPIO {args.heartbeat_pin}")
        print(f"  - Log pin: {'GPIO ' + str(args.log_pin) if args.log_pin else 'disabled (no trigger logging)'}")
        print(f"  - Heartbeat interval: {args.heartbeat_interval}s")
        print("=" * 70 + "\n")

        # ========================================
        # STEP 4: Start DAQ measurement
        # ========================================
        print("=" * 70)
        print("STEP 4: STARTING DAQ MEASUREMENT")
        print("=" * 70)

        daq.start()
        print("✓ DAQ measurement started")
        print("=" * 70 + "\n")

        # Brief pause to ensure DAQ is fully started
        time.sleep(0.5)

        # ========================================
        # STEP 5: Replay RPM signal (with optional audio)
        # ========================================
        print("=" * 70)
        print("STEP 5: REPLAYING RPM SIGNAL")
        if args.audio_file:
            print(f"         (Audio will play {args.audio_delay}s after replay starts)")
        print("=" * 70)
        print("Starting replay in 2 seconds...")
        time.sleep(2.0)

        # If audio file is specified, start background thread to play it after delay
        audio_thread = None
        if args.audio_file:
            def audio_thread_func():
                try:
                    time.sleep(args.audio_delay)
                    print("\n" + "-" * 70)
                    print("AUDIO PLAYBACK")
                    print("-" * 70)
                    audio_duration = play_audio_signal(
                        args.audio_file,
                        volume=args.audio_volume,
                        device=args.audio_device
                    )
                    print(f"✓ Audio playback completed ({audio_duration:.1f}s)")
                    print("-" * 70 + "\n")
                except Exception as e:
                    print(f"\n⚠ Audio playback failed: {e}\n")

            audio_thread = threading.Thread(target=audio_thread_func, daemon=False)
            audio_thread.start()
            print(f"Audio playback scheduled for {args.audio_delay}s from now\n")

        if not args.audio_only:
            # Run replay in main thread
            multi_esc.replay(
                h5file_path=args.h5_file,
                throttle_mapping_path=args.calibration,
                playback_speed=args.playback_speed,
                max_throttle_limit=args.max_throttle,
                arm_before_replay=False,  # Already armed
                invert_spin=args.invert_spin
            )

        print("\n✓ Replay completed")
        print("=" * 70 + "\n")

        # Wait for audio thread to finish if it's still playing
        if audio_thread and audio_thread.is_alive():
            print("Waiting for audio playback to complete...")
            audio_thread.join()
            print("✓ Audio playback finished\n")

        # Brief pause before stopping DAQ
        print("Continuing recording for 2 more seconds...")
        time.sleep(2.0)

        # ========================================
        # STEP 6: Stop DAQ
        # ========================================
        print("\n" + "=" * 70)
        print("STEP 6: STOPPING DAQ")
        print("=" * 70)

        daq.stop()
        print("✓ DAQ stopped")

        if daq.logger:
            print(f"✓ Recording saved to: {daq.logger.filename}")

        print("=" * 70 + "\n")

        print("=" * 70)
        print("✓ REPLAY AND RECORD COMPLETE")
        print("=" * 70)
        print(f"\nOutput file: {daq.logger.filename if daq.logger else 'N/A'}\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        traceback.print_exc()

    finally:
        # ========================================
        # CLEANUP
        # ========================================
        print("\n" + "=" * 70)
        print("CLEANUP")
        print("=" * 70)

        # Stop DAQ first
        if daq is not None:
            try:
                print("Stopping DAQ...")
                daq.stop()
                print("✓ DAQ stopped")
            except Exception as e:
                print(f"⚠ Error stopping DAQ: {e}")

        # Then cleanup ESC controller
        if multi_esc is not None:
            try:
                print("Disarming ESCs...")
                multi_esc.disarm_all()
                time.sleep(0.5)

                print("Stopping ESC controller...")
                multi_esc.cleanup()
                print("✓ ESC controller cleaned up")
            except Exception as e:
                print(f"⚠ Error cleaning up ESC controller: {e}")

        print("=" * 70)
        print("Cleanup complete\n")


if __name__ == "__main__":
    main()
