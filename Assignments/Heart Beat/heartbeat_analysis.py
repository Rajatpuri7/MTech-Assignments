"""
Heartbeat Audio Analysis
========================
Assignment: Analyze a 20+ minute heartbeat audio recording.
Uses a Schmitt Trigger (dual-threshold hysteresis) approach on the 
amplitude envelope to detect individual heartbeats and calculate:
  - Total number of beats
  - Beats per 10-second window
  - Beats per minute (BPM) over time

Method:
  1. Load and downsample audio to 4000 Hz
  2. Apply bandpass filter (20-200 Hz) to isolate heart sounds
  3. Compute amplitude envelope via Hilbert transform
  4. Apply Schmitt Trigger with adaptive thresholds for beat detection
  5. Calculate beat statistics and generate visualizations
"""

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.ndimage import uniform_filter1d
import os
import csv

# ============================================================
# 1. LOAD THE AUDIO FILE (downsample to 4 kHz)
# ============================================================
AUDIO_FILE = "Heartbeat recording 21 minutes .mp3"
TARGET_SR = 4000  # Downsample to 4 kHz (sufficient for heart sounds)

print(f"Loading audio file: {AUDIO_FILE} ...")
y, sr = librosa.load(AUDIO_FILE, sr=TARGET_SR, mono=True)
duration = len(y) / sr
print(f"  Sample rate      : {sr} Hz")
print(f"  Total samples    : {len(y)}")
print(f"  Duration         : {duration:.2f} seconds ({duration/60:.2f} minutes)")
print()

# ============================================================
# 2. PRE-PROCESSING: BANDPASS FILTER (SOS for stability)
# ============================================================
def bandpass_filter_sos(signal, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter using SOS (numerically stable)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

print("Applying bandpass filter (20-200 Hz, SOS form) ...")
y_filtered = bandpass_filter_sos(y, lowcut=20, highcut=200, fs=sr, order=4)
print(f"  Filtered signal range: [{np.min(y_filtered):.4f}, {np.max(y_filtered):.4f}]")

# ============================================================
# 3. COMPUTE AMPLITUDE ENVELOPE
# ============================================================
print("Computing amplitude envelope (Hilbert transform) ...")
analytic_signal = hilbert(y_filtered)
envelope = np.abs(analytic_signal)

# Smooth the envelope to reduce noise spikes
smooth_window = int(0.05 * sr)  # 50ms smoothing window
envelope_smooth = uniform_filter1d(envelope, size=max(1, smooth_window))
print(f"  Envelope range: [{np.min(envelope_smooth):.6f}, {np.max(envelope_smooth):.6f}]")
print(f"  Envelope mean : {np.mean(envelope_smooth):.6f}")
print()

# ============================================================
# 4. SCHMITT TRIGGER BEAT DETECTION
# ============================================================
print("Detecting beats using Schmitt Trigger approach ...")

def schmitt_trigger_detection(envelope, sr, high_thresh_factor=0.35,
                                low_thresh_factor=0.15, min_beat_gap=0.35,
                                window_duration=15.0):
    """
    Schmitt Trigger (dual-threshold hysteresis) beat detector.
    
    Uses a sliding window to adapt thresholds to local signal amplitude,
    making it robust to volume changes over the 21-minute recording.
    
    How it works:
    - The detector has two states: ARMED and TRIGGERED
    - ARMED -> TRIGGERED: when envelope crosses ABOVE the high threshold
      (a beat is registered at this crossing)
    - TRIGGERED -> ARMED: when envelope drops BELOW the low threshold
      (detector is ready for the next beat)
    - This hysteresis prevents false triggers from noise
    
    Parameters
    ----------
    envelope : np.ndarray
        Smoothed amplitude envelope of the filtered signal.
    sr : int
        Sample rate.
    high_thresh_factor : float
        Fraction of local max for the upper threshold.
    low_thresh_factor : float
        Fraction of local max for the lower threshold.
    min_beat_gap : float
        Minimum time (seconds) between consecutive beats (refractory period).
    window_duration : float
        Duration (seconds) of the sliding window for adaptive threshold.
    
    Returns
    -------
    beat_times : list of float
        Timestamps (seconds) of detected beats.
    """
    n_samples = len(envelope)
    min_gap_samples = int(min_beat_gap * sr)
    
    beat_times = []
    state = 'armed'
    last_beat_sample = -min_gap_samples
    
    # Pre-compute adaptive thresholds using sliding window local max
    block_size = int(0.5 * sr)  # 0.5 second blocks for efficiency
    n_blocks = (n_samples + block_size - 1) // block_size
    window_samples = int(window_duration * sr)
    
    local_maxes = np.zeros(n_samples)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_samples)
        win_start = max(0, start - window_samples // 2)
        win_end = min(n_samples, end + window_samples // 2)
        local_max = np.max(envelope[win_start:win_end])
        # Ensure minimum threshold to avoid triggering on silence
        local_max = max(local_max, np.percentile(envelope, 80))
        local_maxes[start:end] = local_max
    
    high_thresh = high_thresh_factor * local_maxes
    low_thresh = low_thresh_factor * local_maxes
    
    # Run the Schmitt trigger state machine
    for i in range(n_samples):
        if state == 'armed':
            if envelope[i] > high_thresh[i]:
                if (i - last_beat_sample) >= min_gap_samples:
                    beat_times.append(i / sr)
                    last_beat_sample = i
                state = 'triggered'
        elif state == 'triggered':
            if envelope[i] < low_thresh[i]:
                state = 'armed'
    
    return beat_times, high_thresh, low_thresh

# Run detection
beat_times, high_thresh, low_thresh = schmitt_trigger_detection(envelope_smooth, sr)
total_beats = len(beat_times)
print(f"  Total beats detected: {total_beats}")

# Validate: expect roughly 60-100 BPM for 21 minutes
expected_min = int(50 * duration / 60)
expected_max = int(120 * duration / 60)
if expected_min <= total_beats <= expected_max:
    print(f"  [OK] Beat count is within expected range ({expected_min}-{expected_max})")
else:
    print(f"  [NOTE] Expected {expected_min}-{expected_max} beats for this duration")
print()

# ============================================================
# 5. CALCULATE BEATS PER 10 SECONDS & BEATS PER MINUTE
# ============================================================
print("Calculating beats per 10s and BPM ...")

# --- Beats per 10-second window ---
window_10s = 10.0
n_windows_10s = int(np.ceil(duration / window_10s))
beats_per_10s = []
window_centers_10s = []

beat_times_arr = np.array(beat_times)

for i in range(n_windows_10s):
    t_start = i * window_10s
    t_end = t_start + window_10s
    count = np.sum((beat_times_arr >= t_start) & (beat_times_arr < t_end))
    beats_per_10s.append(count)
    window_centers_10s.append(t_start + window_10s / 2)

beats_per_10s = np.array(beats_per_10s)
window_centers_10s = np.array(window_centers_10s)

# --- Instantaneous BPM from inter-beat intervals ---
if len(beat_times_arr) > 1:
    ibi = np.diff(beat_times_arr)  # inter-beat intervals in seconds
    instantaneous_bpm = 60.0 / ibi
    bpm_times = beat_times_arr[1:]
else:
    ibi = np.array([])
    instantaneous_bpm = np.array([])
    bpm_times = np.array([])

# --- Rolling average BPM (60-second window) ---
window_60s = 60.0
bpm_rolling = []
bpm_rolling_times = []

for i in range(n_windows_10s):
    t_center = i * window_10s + window_10s / 2
    t_start = max(0, t_center - window_60s / 2)
    t_end = min(duration, t_center + window_60s / 2)
    actual_window = t_end - t_start
    count = np.sum((beat_times_arr >= t_start) & (beat_times_arr < t_end))
    if actual_window > 0:
        bpm = count * (60.0 / actual_window)
        bpm_rolling.append(bpm)
        bpm_rolling_times.append(t_center)

bpm_rolling = np.array(bpm_rolling)
bpm_rolling_times = np.array(bpm_rolling_times)

# Overall average BPM
avg_bpm = total_beats / (duration / 60.0) if duration > 0 else 0

print(f"  Average BPM       : {avg_bpm:.1f}")
if len(instantaneous_bpm) > 0:
    print(f"  Min instant. BPM  : {np.min(instantaneous_bpm):.1f}")
    print(f"  Max instant. BPM  : {np.max(instantaneous_bpm):.1f}")
    print(f"  Median inst. BPM  : {np.median(instantaneous_bpm):.1f}")
print()

# ============================================================
# 6. SAVE BEAT TRANSCRIPTION TO CSV
# ============================================================
csv_file = "beat_timestamps.csv"
print(f"Saving beat timestamps to {csv_file} ...")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Beat_Number", "Time_seconds", "Time_mm:ss.ms"])
    for i, bt in enumerate(beat_times, 1):
        mins = int(bt // 60)
        secs = bt % 60
        writer.writerow([i, f"{bt:.4f}", f"{mins:02d}:{secs:06.3f}"])
print(f"  Saved {total_beats} beat timestamps.")
print()

# ============================================================
# 7. SAVE 10-SECOND WINDOW SUMMARY TO CSV
# ============================================================
summary_csv = "beats_per_10s_summary.csv"
print(f"Saving 10-second window counts to {summary_csv} ...")
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Window_Start_s", "Window_End_s", "Beats_Count", "Est_BPM"])
    for i in range(n_windows_10s):
        t_start = i * window_10s
        t_end = t_start + window_10s
        count = beats_per_10s[i]
        est_bpm = count * 6  # beats per 10s * 6 = BPM estimate
        writer.writerow([f"{t_start:.1f}", f"{t_end:.1f}", count, est_bpm])
print(f"  Saved {n_windows_10s} windows.")
print()

# ============================================================
# 8. GENERATE PLOTS
# ============================================================
print("Generating plots ...")
os.makedirs("plots", exist_ok=True)

time_axis = np.arange(len(y)) / sr

# ------ PLOT 1: Full waveform overview ------
fig, axes = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True)
fig.suptitle("Heartbeat Audio Analysis - Full Recording Overview",
             fontsize=16, fontweight='bold')

# 1a. Raw waveform
axes[0].plot(time_axis / 60, y, linewidth=0.15, color='#2196F3', alpha=0.7)
axes[0].set_title("Raw Audio Waveform (Time Domain)", fontsize=12)
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, duration / 60)
axes[0].grid(True, alpha=0.3)

# 1b. Filtered waveform + envelope
axes[1].plot(time_axis / 60, y_filtered, linewidth=0.15, color='#4CAF50', alpha=0.5,
             label='Filtered signal')
axes[1].plot(time_axis / 60, envelope_smooth, linewidth=0.4, color='#FF5722',
             label='Envelope')
axes[1].set_title("Bandpass Filtered Signal (20-200 Hz) + Amplitude Envelope", fontsize=12)
axes[1].set_ylabel("Amplitude")
axes[1].set_xlim(0, duration / 60)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

# 1c. Beats per 10 seconds
axes[2].bar(window_centers_10s / 60, beats_per_10s, width=window_10s / 60 * 0.9,
            color='#9C27B0', alpha=0.7, edgecolor='#7B1FA2')
axes[2].set_title("Number of Beats per 10-Second Window", fontsize=12)
axes[2].set_ylabel("Beat Count")
axes[2].set_xlim(0, duration / 60)
axes[2].grid(True, alpha=0.3, axis='y')

# 1d. BPM over time
if len(bpm_rolling) > 0:
    axes[3].plot(np.array(bpm_rolling_times) / 60, bpm_rolling,
                 linewidth=1.5, color='#E91E63', label='Rolling BPM (60s window)')
    axes[3].axhline(y=avg_bpm, color='#FFC107', linestyle='--', linewidth=1.5,
                    label=f'Average BPM = {avg_bpm:.1f}')
    axes[3].fill_between(np.array(bpm_rolling_times) / 60, bpm_rolling,
                         alpha=0.15, color='#E91E63')
    axes[3].legend(loc='upper right', fontsize=9)
axes[3].set_title("Heart Rate (BPM) Over Time", fontsize=12)
axes[3].set_ylabel("BPM")
axes[3].set_xlabel("Time (minutes)")
axes[3].set_xlim(0, duration / 60)
axes[3].grid(True, alpha=0.3)

plt.savefig("plots/01_full_overview.png", dpi=150, bbox_inches='tight')
print("  Saved: plots/01_full_overview.png")
plt.close()

# ------ PLOT 2: Zoomed 30 seconds showing individual beats ------
fig, axes = plt.subplots(2, 1, figsize=(18, 8), constrained_layout=True)
fig.suptitle("Heartbeat Detection - Zoomed View (First 30 Seconds)",
             fontsize=14, fontweight='bold')

zoom_end = 30
zoom_mask = time_axis <= zoom_end
beat_mask_30 = beat_times_arr[beat_times_arr <= zoom_end]

# 2a. Raw waveform zoomed
axes[0].plot(time_axis[zoom_mask], y[zoom_mask], linewidth=0.5, color='#2196F3')
for bt in beat_mask_30:
    axes[0].axvline(x=bt, color='red', linewidth=0.8, alpha=0.6)
axes[0].set_title("Raw Waveform with Detected Beats (red lines)", fontsize=12)
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, zoom_end)
axes[0].grid(True, alpha=0.3)

# 2b. Envelope zoomed
axes[1].plot(time_axis[zoom_mask], envelope_smooth[zoom_mask], linewidth=1,
             color='#FF5722', label='Envelope')
axes[1].plot(time_axis[zoom_mask], high_thresh[zoom_mask], linewidth=0.8,
             color='#4CAF50', linestyle='--', alpha=0.7, label='High threshold')
axes[1].plot(time_axis[zoom_mask], low_thresh[zoom_mask], linewidth=0.8,
             color='#2196F3', linestyle='--', alpha=0.7, label='Low threshold')
for bt in beat_mask_30:
    axes[1].axvline(x=bt, color='red', linewidth=0.8, alpha=0.6)
axes[1].set_title("Envelope with Schmitt Trigger Thresholds & Detected Beats", fontsize=12)
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_xlim(0, zoom_end)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.savefig("plots/02_zoomed_30s.png", dpi=150, bbox_inches='tight')
print("  Saved: plots/02_zoomed_30s.png")
plt.close()

# ------ PLOT 3: 5-second detail showing Schmitt Trigger operation ------
fig, ax = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)

zoom_s = 5
zoom_e = 10
detail_mask = (time_axis >= zoom_s) & (time_axis <= zoom_e)
detail_beats = beat_times_arr[(beat_times_arr >= zoom_s) & (beat_times_arr <= zoom_e)]

ax.plot(time_axis[detail_mask], envelope_smooth[detail_mask], linewidth=1.5,
        color='#FF5722', label='Envelope', zorder=3)
ax.plot(time_axis[detail_mask], high_thresh[detail_mask], linewidth=1.5,
        color='#4CAF50', linestyle='--', label='High threshold (arm->triggered)')
ax.plot(time_axis[detail_mask], low_thresh[detail_mask], linewidth=1.5,
        color='#2196F3', linestyle='--', label='Low threshold (triggered->armed)')
ax.fill_between(time_axis[detail_mask], low_thresh[detail_mask], high_thresh[detail_mask],
                alpha=0.1, color='green', label='Hysteresis band')
for bt in detail_beats:
    ax.axvline(x=bt, color='red', linewidth=1.5, alpha=0.8, linestyle='-',
               label='Beat detected' if bt == detail_beats[0] else None)

ax.set_title("Schmitt Trigger Detection - 5-Second Detail View",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Envelope Amplitude")
ax.set_xlabel("Time (seconds)")
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.savefig("plots/03_schmitt_trigger_detail.png", dpi=150, bbox_inches='tight')
print("  Saved: plots/03_schmitt_trigger_detail.png")
plt.close()

# ------ PLOT 4: BPM histogram ------
fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
if len(instantaneous_bpm) > 0:
    valid_bpm = instantaneous_bpm[(instantaneous_bpm > 30) & (instantaneous_bpm < 200)]
    ax.hist(valid_bpm, bins=50, color='#E91E63', alpha=0.7, edgecolor='#C2185B')
    ax.axvline(x=avg_bpm, color='#FFC107', linewidth=2, linestyle='--',
               label=f'Average = {avg_bpm:.1f} BPM')
    ax.axvline(x=np.median(valid_bpm), color='#4CAF50', linewidth=2, linestyle='--',
               label=f'Median = {np.median(valid_bpm):.1f} BPM')
    ax.legend(fontsize=11)
ax.set_title("Distribution of Instantaneous Heart Rate", fontsize=13, fontweight='bold')
ax.set_xlabel("BPM")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3, axis='y')
plt.savefig("plots/04_bpm_histogram.png", dpi=150, bbox_inches='tight')
print("  Saved: plots/04_bpm_histogram.png")
plt.close()

# ------ PLOT 5: Beats per 10s table (first 30 windows as sample) ------
fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
ax.axis('off')
table_data = []
for i in range(min(30, n_windows_10s)):
    t_s = i * window_10s
    t_e = t_s + window_10s
    table_data.append([f"{t_s:.0f}-{t_e:.0f}s", str(beats_per_10s[i]), str(beats_per_10s[i] * 6)])

table = ax.table(cellText=table_data,
                 colLabels=["Time Window", "Beats/10s", "Est. BPM"],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(0.8, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#E91E63')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#FCE4EC')

ax.set_title("Sample: Beats per 10-Second Window (first 5 minutes)",
             fontsize=13, fontweight='bold', pad=20)
plt.savefig("plots/05_beats_table_sample.png", dpi=150, bbox_inches='tight')
print("  Saved: plots/05_beats_table_sample.png")
plt.close()

# ============================================================
# 9. PRINT FINAL SUMMARY
# ============================================================
print()
print("=" * 60)
print("           HEARTBEAT ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Audio File         : {AUDIO_FILE}")
print(f"  Duration           : {duration:.2f}s ({duration/60:.2f} min)")
print(f"  Sample Rate (used) : {sr} Hz")
print(f"  Detection Method   : Schmitt Trigger (dual-threshold hysteresis)")
print(f"  Filter Band        : 20-200 Hz (Butterworth SOS, order 4)")
print(f"  ------------------------------------------------")
print(f"  Total Beats        : {total_beats}")
print(f"  Average BPM        : {avg_bpm:.1f}")
if len(instantaneous_bpm) > 0:
    print(f"  Median BPM         : {np.median(instantaneous_bpm):.1f}")
    print(f"  Min BPM            : {np.min(instantaneous_bpm):.1f}")
    print(f"  Max BPM            : {np.max(instantaneous_bpm):.1f}")
    print(f"  Std Dev BPM        : {np.std(instantaneous_bpm):.1f}")
print(f"  Avg beats/10s      : {np.mean(beats_per_10s):.1f}")
print(f"  ------------------------------------------------")
print(f"  Outputs:")
print(f"    - {csv_file}         (beat-by-beat timestamps)")
print(f"    - {summary_csv}    (10s window summary)")
print(f"    - plots/01_full_overview.png")
print(f"    - plots/02_zoomed_30s.png")
print(f"    - plots/03_schmitt_trigger_detail.png")
print(f"    - plots/04_bpm_histogram.png")
print(f"    - plots/05_beats_table_sample.png")
print("=" * 60)
print("\nDone!")
