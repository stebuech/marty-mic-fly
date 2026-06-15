import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import h5py

filepath='/home/steffen/MMFDataLogs/telemetry_data_20251113_152221.h5'
with h5py.File(filepath, 'r') as f:
    mic_group = f['mic_array']
    timestamps = mic_group['timestamp'][:]
    audio_data = mic_group['audio_data'][:]
    time_drift = mic_group['timing_drift']['drift'][:]
    time_drift_mean = mic_group['timing_drift']['drift_mean'][:]
    time_drift_std = mic_group['timing_drift']['drift_std'][:]
    time_drift_correction = mic_group['timing_drift']['correction_factor'][:]
    adc_time = mic_group['timing_drift']['adc_timestamp'][:]
    hw_time = mic_group['timing_drift']['hw_timestamp'][:]
    time = mic_group['timing_drift']['timestamp'][:]

dt = np.diff(timestamps)
dt_block = dt[1023::1024]

np.savetxt("dt_check.csv", dt_block, delimiter=",")
