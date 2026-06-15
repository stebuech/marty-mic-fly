#!/usr/bin/env python3
"""Quick visualization of synthetic RPM profile"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data
with h5py.File('synthetic_motor1_3400rpm.h5', 'r') as f:
    timestamps = f['esc_telemetry/ESC1/timestamp'][:]
    rpms = f['esc_telemetry/ESC1/rpm'][:]

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps, rpms, linewidth=2, label='ESC1 RPM')
ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Idle RPM (5% throttle)')
ax.axhline(y=3400, color='red', linestyle='--', alpha=0.5, label='Target RPM')

# Mark phases
ax.axvline(x=3.0, color='green', linestyle=':', alpha=0.7)
ax.axvline(x=13.0, color='orange', linestyle=':', alpha=0.7)
ax.text(1.5, 3200, 'Ramp Up\n(3s)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.text(8.0, 3200, 'Hold\n(10s)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(14.5, 3200, 'Ramp Down\n(3s)', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('RPM', fontsize=12)
ax.set_title('Synthetic Motor RPM Profile for Replay', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlim(0, timestamps[-1])
ax.set_ylim(0, 3600)

plt.tight_layout()
plt.savefig('synthetic_rpm_profile.png', dpi=150)
print("✓ Plot saved to: synthetic_rpm_profile.png")