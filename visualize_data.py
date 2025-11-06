import h5py
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


def visualize_data(filepath, show_mic_channels=None, figsize=(14, 10)):

    with h5py.File(filepath, 'r') as f:
        has_mic_array, has_esc, has_trigger = 'mic_array' in f, 'esc_telemetry' in f, 'trigger' in f

    n_subplots = 0
    if has_trigger:
        n_subplots += 1
    if has_mic_array:
        if show_mic_channels is None:
            show_mic_channels = [0, 1]
        n_subplots += len(show_mic_channels)
    if has_esc:
        n_subplots += 1  # Only RPM subplot


    # Create figure with linked time axes
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]

    ax_idx = 0

    # Plot trigger events at the top
    if has_trigger:
        with h5py.File(filepath, 'r') as f:
            trigger_group = f['trigger']
            timestamps = trigger_group['timestamp'][:]
            states = trigger_group['state'][:]

            ax = axes[ax_idx]
            # Plot trigger events as vertical lines
            for ts, state in zip(timestamps, states):
                color = 'green' if state == 1 else 'red'
                ax.axvline(ts, color=color, alpha=0.6, linewidth=1)

            ax.set_ylabel('Trigger')
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([])
            ax_idx += 1

    # Plot microphone array data
    if has_mic_array:
        with h5py.File(filepath, 'r') as f:
            mic_group = f['mic_array']
            timestamps = mic_group['timestamp'][:]
            audio_data = mic_group['audio_data'][:]

            for ch_idx in show_mic_channels:
                ax = axes[ax_idx]
                ax.plot(timestamps, audio_data[:, ch_idx], linewidth=0.5)
                ax.set_ylabel(f'Mic Ch{ch_idx}')
                ax.grid(True, alpha=0.3)
                ax_idx += 1

    # Plot ESC RPM data
    if has_esc:
        with h5py.File(filepath, 'r') as f:
            esc_group = f['esc_telemetry']
            ax = axes[ax_idx]

            for esc_id in esc_group:
                esc_subgroup = esc_group[esc_id]
                timestamps = esc_subgroup['timestamp'][:]
                rpm = esc_subgroup['rpm'][:]

                ax.plot(timestamps, rpm, label=esc_id, marker='.')

            ax.set_ylabel('RPM')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # Set xlabel on bottom subplot
    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()

    return fig