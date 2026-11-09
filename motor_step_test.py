#!/usr/bin/env python3
"""Short motor step test: all motors through a few throttle steps, then stop.

Only drives the motors (DShot from the Pi). Record in parallel with daq.py in
a second terminal; this script does not open the Pico link.

    python daq.py --no-trigger-controlled --motors 4     # terminal 1
    python motor_step_test.py                            # terminal 2

PROPELLERS OFF.
"""

import argparse
import time

from esc_throttle_set import MultiESCControler
from hw_config import dshot_pins

MAX_THROTTLE = 35  # percent, hard limit for this test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--motors', type=int, choices=[4, 6], default=4)
    ap.add_argument('--steps', type=float, nargs='+', default=[10, 20, 30], help='throttle steps in percent')
    ap.add_argument('--hold', type=float, default=5.0, help='seconds per step')
    ap.add_argument('--dshot-speed', type=int, choices=[150, 300, 600], default=300)
    ap.add_argument('--arming-duration', type=float, default=3.0)
    args = ap.parse_args()

    if max(args.steps) > MAX_THROTTLE:
        ap.error(f'steps above {MAX_THROTTLE}% are not allowed in this test')

    pins = dshot_pins(args.motors)
    total = args.arming_duration + len(args.steps) * args.hold
    print(f"Motors {args.motors} on GPIO {pins}, DShot{args.dshot_speed}")
    print(f"Steps {args.steps} %, {args.hold:.0f} s each, about {total:.0f} s in total")
    print("PROPELLERS OFF? Is daq.py already recording in the other terminal?")
    if input("Start motors? [y/N]: ").strip().lower() != 'y':
        print("Cancelled.")
        return

    esc = MultiESCControler(pins, dshot_speed=args.dshot_speed)
    try:
        esc.arm_all(duration=args.arming_duration)
        for pct in args.steps:
            print(f"{time.strftime('%H:%M:%S')}  throttle {pct:.0f}%")
            esc.set_all_throttle(pct)
            time.sleep(args.hold)
        print(f"{time.strftime('%H:%M:%S')}  stop")
    except KeyboardInterrupt:
        print("\nInterrupted, stopping motors")
    finally:
        esc.disarm_all()
        time.sleep(1.0)
        esc.cleanup()
        print("Motors stopped.")


if __name__ == '__main__':
    main()
