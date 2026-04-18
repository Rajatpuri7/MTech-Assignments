"""
Google Maps Journey Analysis - Assignment 1
============================================
Extracts and analyses journey data from Google Maps screen recording.

Tasks:
a) Speed vs Time graph
b) Instantaneous time-to-travel-10km vs Time graph
c1) Distance covered every 2 minutes (rolling window) graph
c2) Total distance from rolling window + percentage error
c3) Instantaneous speed comparison + absolute error, mean error, RMSE
d) Instantaneous acceleration vs Time graph
e) Instantaneous jerk vs Time graph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional plot styling
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#cccccc',
})

# Color palette
C_BLUE = '#1976D2'
C_RED = '#D32F2F'
C_GREEN = '#388E3C'
C_ORANGE = '#F57C00'
C_PURPLE = '#7B1FA2'
C_TEAL = '#00796B'
C_PINK = '#C2185B'

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
print("=" * 70)
print("GOOGLE MAPS JOURNEY ANALYSIS")
print("=" * 70)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_data.csv")
df = pd.read_csv(data_path)

# Time conversions
df['time_min'] = df['video_time_s'] / 60.0
df['time_s'] = df['video_time_s'].astype(float)

# Speed conversions
df['speed_ms'] = df['speed_kmh'] / 3.6

# Distance covered from distance_remaining (note: distance_remaining is coarse integer km)
total_route_distance_km = df['distance_remaining_km'].iloc[0]  # initial distance remaining
df['dist_covered_gmap_km'] = total_route_distance_km - df['distance_remaining_km']

# ---- Build a SMOOTH distance profile from the recorded speed ----
# Integrate speed over time using trapezoidal rule → gives distance covered
# This is the most reliable method since speed readings are fine-grained
time_arr = df['time_s'].values
speed_ms_arr = df['speed_ms'].values

# Cumulative distance by integrating speed
dist_integrated_m = np.zeros(len(time_arr))
for i in range(1, len(time_arr)):
    dt = time_arr[i] - time_arr[i-1]
    dist_integrated_m[i] = dist_integrated_m[i-1] + 0.5 * (speed_ms_arr[i] + speed_ms_arr[i-1]) * dt

dist_integrated_km = dist_integrated_m / 1000.0
df['dist_covered_integrated_km'] = dist_integrated_km

# Also smooth the Google Maps distance_remaining using a spline
# to get a better differentiation basis
from scipy.signal import savgol_filter

# Use Savitzky-Golay filter on distance_remaining for smoothing
# (window length must be odd and > polyorder)
if len(df) > 11:
    dist_remaining_smooth = savgol_filter(df['distance_remaining_km'].values, 
                                          window_length=11, polyorder=3)
else:
    dist_remaining_smooth = df['distance_remaining_km'].values

df['dist_remaining_smooth_km'] = dist_remaining_smooth
df['dist_covered_smooth_km'] = dist_remaining_smooth[0] - dist_remaining_smooth

# ---- Summary ----
actual_distance = df['dist_covered_gmap_km'].iloc[-1]
integrated_distance = df['dist_covered_integrated_km'].iloc[-1]
journey_duration_min = df['time_min'].iloc[-1]

print(f"\nJourney Summary:")
print(f"  Route total (Google Maps initial): {total_route_distance_km} km")
print(f"  Journey duration: {df['time_s'].iloc[-1]:.0f} s = {journey_duration_min:.1f} min")
print(f"  Distance covered (Google Maps): {actual_distance:.1f} km")
print(f"  Distance covered (speed integration): {integrated_distance:.2f} km")
print(f"  Speed range: {df['speed_kmh'].min()} – {df['speed_kmh'].max()} km/h")
print(f"  Average speed: {df['speed_kmh'].mean():.1f} km/h")

# ============================================================================
# TASK (a): Speed vs Time Graph
# ============================================================================
print("\n--- Task (a): Speed vs Time ---")

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['time_min'], df['speed_kmh'], color=C_BLUE, linewidth=2.2, 
        label='Speed (from Google Maps)', marker='o', markersize=3, alpha=0.9)
ax.fill_between(df['time_min'], 0, df['speed_kmh'], alpha=0.10, color=C_BLUE)
avg_speed = df['speed_kmh'].mean()
ax.axhline(y=avg_speed, color=C_RED, linestyle='--', linewidth=1.5,
           label=f'Average Speed = {avg_speed:.1f} km/h')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Speed (km/h)')
ax.set_title('Task (a): Speed vs Time')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, df['time_min'].iloc[-1])
ax.set_ylim(0, max(df['speed_kmh']) * 1.15)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "a_speed_vs_time.png"), bbox_inches='tight')
plt.close()
print("  [OK] Saved: a_speed_vs_time.png")

# ============================================================================
# TASK (b): Instantaneous time-to-travel-10km vs Time
# ============================================================================
print("\n--- Task (b): Instantaneous time to travel 10 km vs Time ---")

df['time_to_10km_min'] = np.where(df['speed_kmh'] > 0, (10.0 / df['speed_kmh']) * 60, np.nan)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['time_min'], df['time_to_10km_min'], color=C_PURPLE, linewidth=2.2,
        label='Time to travel 10 km', marker='s', markersize=3)
ax.fill_between(df['time_min'], 0, df['time_to_10km_min'], alpha=0.10, color=C_PURPLE)
avg_t10 = np.nanmean(df['time_to_10km_min'])
ax.axhline(y=avg_t10, color=C_ORANGE, linestyle='--', linewidth=1.5,
           label=f'Average = {avg_t10:.1f} min')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Time to travel 10 km (minutes)')
ax.set_title('Task (b): Instantaneous Time to Travel 10 km vs Time')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, df['time_min'].iloc[-1])
ax.set_ylim(0)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "b_time_to_10km_vs_time.png"), bbox_inches='tight')
plt.close()
print("  [OK] Saved: b_time_to_10km_vs_time.png")

# ============================================================================
# TASK (c1): Distance covered every 2 minutes (rolling window)
# ============================================================================
print("\n--- Task (c1): Distance covered in rolling 2-min windows ---")

# Create fine time series by interpolation of the integrated distance
time_fine = np.arange(0, df['time_s'].iloc[-1] + 1, 1)  # 1-second resolution
dist_fine = np.interp(time_fine, df['time_s'], df['dist_covered_integrated_km'])

# Rolling 2-minute windows as per assignment specification:
# 0min          (plotted at 0min mark) → x0 = ZERO
# 0min–2min     (plotted at 1min mark) → x1
# 1min–3min     (plotted at 2min mark) → x2
# 2min–4min     (plotted at 3min mark) → x3
# ... and so on
window_s = 120  # 2 minutes
max_time_s = int(df['time_s'].iloc[-1])

rolling_plot_min = [0]
rolling_dist = [0.0]
rolling_labels = ["0 min (ZERO)"]

window_idx = 1
start_s = 0
while start_s + window_s <= max_time_s:
    end_s = start_s + window_s
    d_start = np.interp(start_s, time_fine, dist_fine)
    d_end = np.interp(end_s, time_fine, dist_fine)
    dx = d_end - d_start
    rolling_plot_min.append(window_idx)
    rolling_dist.append(dx)
    rolling_labels.append(f"{start_s//60}–{end_s//60} min")
    start_s += 60
    window_idx += 1

rolling_plot_min = np.array(rolling_plot_min)
rolling_dist = np.array(rolling_dist)

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(rolling_plot_min, rolling_dist, width=0.65, color=C_GREEN, alpha=0.80,
              edgecolor='#2E7D32', linewidth=0.8, label='Distance in 2-min window')
ax.plot(rolling_plot_min, rolling_dist, color=C_RED, linewidth=1.5, marker='o',
        markersize=5, label='Trend', zorder=5)

for t, d in zip(rolling_plot_min, rolling_dist):
    if d > 0:
        ax.text(t, d + 0.03, f'{d:.2f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', rotation=45)

ax.set_xlabel('Time (minute mark on graph)')
ax.set_ylabel('Distance covered in 2-min window (km)')
ax.set_title('Task (c1): Distance Covered Every Two Minutes (Rolling Window)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(-0.5, rolling_plot_min[-1] + 0.5)
ax.set_ylim(0, max(rolling_dist) * 1.35)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "c1_distance_every_2min.png"), bbox_inches='tight')
plt.close()
print("  [OK] Saved: c1_distance_every_2min.png")

# Print rolling window table
print("\n  Rolling 2-min Window Data Table:")
print(f"  {'Plot Mark':>10s} | {'Window':>14s} | {'Distance (km)':>14s}")
print(f"  {'─'*10} | {'─'*14} | {'─'*14}")
for t, lbl, d in zip(rolling_plot_min, rolling_labels, rolling_dist):
    print(f"  {t:>8.0f}   | {lbl:>14s} | {d:>12.4f}")

# ============================================================================
# TASK (c2): Total distance from rolling window + percentage error
# ============================================================================
print("\n--- Task (c2): Total distance from c1 graph & percentage error ---")

# Method: Since rolling windows overlap by 1 min, non-overlapping sum uses every other window
# But a more rigorous approach: extract 1-minute distance increments
one_min_dists = []
for t_start_s in range(0, max_time_s, 60):
    t_end_s = min(t_start_s + 60, max_time_s)
    d_s = np.interp(t_start_s, time_fine, dist_fine)
    d_e = np.interp(t_end_s, time_fine, dist_fine)
    one_min_dists.append(d_e - d_s)

total_from_integration = sum(one_min_dists)

# From the rolling window graph: non-overlapping pairs
total_non_overlap = sum(rolling_dist[i] for i in range(1, len(rolling_dist), 2))

# Ground truth from Google Maps
actual_gmap = df['dist_covered_gmap_km'].iloc[-1]

pct_err_integrated = abs(total_from_integration - actual_gmap) / actual_gmap * 100
pct_err_non_overlap = abs(total_non_overlap - actual_gmap) / actual_gmap * 100

print(f"  Actual distance (Google Maps Δ remaining): {actual_gmap:.2f} km")
print(f"  Distance from speed integration:           {total_from_integration:.2f} km")
print(f"  Distance from non-overlapping 2-min sums:  {total_non_overlap:.2f} km")
print(f"  Percentage error (integration vs Google):   {pct_err_integrated:.2f}%")
print(f"  Percentage error (non-overlap vs Google):   {pct_err_non_overlap:.2f}%")

# ============================================================================
# TASK (c3): Instantaneous speed comparison + error metrics
# ============================================================================
print("\n--- Task (c3): Speed comparison and error analysis ---")

# Calculated speed = d(dist_integrated)/dt using central differences
# Smooth the integrated distance slightly to reduce noise
dist_smooth = savgol_filter(df['dist_covered_integrated_km'].values, 
                            window_length=min(7, len(df) if len(df) % 2 == 1 else len(df)-1), 
                            polyorder=2)

# Central difference for interior, forward/backward for edges
calc_speed_kmh = np.zeros(len(df))
for i in range(len(df)):
    if i == 0:
        dt = time_arr[1] - time_arr[0]
        dd = dist_smooth[1] - dist_smooth[0]
    elif i == len(df) - 1:
        dt = time_arr[-1] - time_arr[-2]
        dd = dist_smooth[-1] - dist_smooth[-2]
    else:
        dt = time_arr[i+1] - time_arr[i-1]
        dd = dist_smooth[i+1] - dist_smooth[i-1]
    calc_speed_kmh[i] = (dd / dt) * 3600  # km/s → km/h

df['calc_speed_kmh'] = calc_speed_kmh

# Error analysis
abs_error = np.abs(df['speed_kmh'].values - df['calc_speed_kmh'].values)
mean_abs_error = np.mean(abs_error)
rmse = np.sqrt(np.mean((df['speed_kmh'].values - df['calc_speed_kmh'].values) ** 2))
max_abs_error = np.max(abs_error)

print(f"  Mean Absolute Error:  {mean_abs_error:.2f} km/h")
print(f"  RMSE:                 {rmse:.2f} km/h")
print(f"  Max Absolute Error:   {max_abs_error:.2f} km/h")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

# Speed comparison
ax1 = axes[0]
ax1.plot(df['time_min'], df['speed_kmh'], color=C_BLUE, linewidth=2,
         label='Recorded Speed (Google Maps)', marker='o', markersize=3)
ax1.plot(df['time_min'], df['calc_speed_kmh'], color=C_ORANGE, linewidth=1.8,
         label='Calculated Speed (Δd/Δt)', linestyle='--', marker='x', markersize=4)
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Speed (km/h)')
ax1.set_title('Task (c3): Recorded vs Calculated Instantaneous Speed')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_xlim(0, df['time_min'].iloc[-1])
ax1.set_ylim(0, max(max(df['speed_kmh']), max(df['calc_speed_kmh'])) * 1.15)

# Error plot
ax2 = axes[1]
ax2.bar(df['time_min'], abs_error, width=0.12, color=C_RED, alpha=0.7,
        label='|Error|', edgecolor='none')
ax2.axhline(y=mean_abs_error, color=C_ORANGE, linestyle='--', linewidth=2,
            label=f'Mean Abs Error = {mean_abs_error:.2f} km/h')
ax2.axhline(y=rmse, color=C_PURPLE, linestyle=':', linewidth=2,
            label=f'RMSE = {rmse:.2f} km/h')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Absolute Error (km/h)')
ax2.set_title('Speed Estimation Error: |Recorded − Calculated|')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_xlim(0, df['time_min'].iloc[-1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "c3_speed_comparison.png"), bbox_inches='tight')
plt.close()
print("  ✓ Saved: c3_speed_comparison.png")

# Print error table (sample every 5th point)
print(f"\n  Sample Error Table (every 5th data point):")
print(f"  {'Time(min)':>10s} | {'Recorded':>10s} | {'Calculated':>10s} | {'Abs Error':>10s}")
print(f"  {'─'*10} | {'─'*10} | {'─'*10} | {'─'*10}")
for i in range(0, len(df), 5):
    print(f"  {df['time_min'].iloc[i]:>8.1f}   | {df['speed_kmh'].iloc[i]:>8.1f}   | "
          f"{df['calc_speed_kmh'].iloc[i]:>8.1f}   | {abs_error[i]:>8.2f}")

# ============================================================================
# TASK (d): Instantaneous Acceleration vs Time
# ============================================================================
print("\n--- Task (d): Instantaneous Acceleration vs Time ---")

# Smooth speed first for cleaner acceleration
speed_smooth_ms = savgol_filter(df['speed_ms'].values,
                                 window_length=min(7, len(df) if len(df) % 2 == 1 else len(df)-1),
                                 polyorder=2)

# Central differences for acceleration
accel = np.zeros(len(df))
for i in range(len(df)):
    if i == 0:
        accel[i] = (speed_smooth_ms[1] - speed_smooth_ms[0]) / (time_arr[1] - time_arr[0])
    elif i == len(df) - 1:
        accel[i] = (speed_smooth_ms[-1] - speed_smooth_ms[-2]) / (time_arr[-1] - time_arr[-2])
    else:
        accel[i] = (speed_smooth_ms[i+1] - speed_smooth_ms[i-1]) / (time_arr[i+1] - time_arr[i-1])

df['acceleration_ms2'] = accel

fig, ax = plt.subplots(figsize=(14, 7))
colors_acc = [C_GREEN if a >= 0 else C_RED for a in accel]
ax.bar(df['time_min'], accel, width=0.13, color=colors_acc, alpha=0.75, edgecolor='none')
ax.plot(df['time_min'], accel, color='#333333', linewidth=0.8, alpha=0.4)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Acceleration (m/s²)')
ax.set_title('Task (d): Instantaneous Acceleration vs Time')

from matplotlib.patches import Patch
legend_acc = [Patch(facecolor=C_GREEN, alpha=0.75, label='Acceleration (+)'),
              Patch(facecolor=C_RED, alpha=0.75, label='Deceleration (−)')]
ax.legend(handles=legend_acc, loc='upper right', framealpha=0.9)
ax.set_xlim(0, df['time_min'].iloc[-1])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "d_acceleration_vs_time.png"), bbox_inches='tight')
plt.close()
print(f"  Max acceleration:  {np.max(accel):.4f} m/s² ({np.max(accel)*3.6:.2f} km/h/s)")
print(f"  Max deceleration:  {np.min(accel):.4f} m/s² ({np.min(accel)*3.6:.2f} km/h/s)")
print(f"  Mean |acceleration|: {np.mean(np.abs(accel)):.4f} m/s²")
print("  ✓ Saved: d_acceleration_vs_time.png")

# ============================================================================
# TASK (e): Instantaneous Jerk vs Time
# ============================================================================
print("\n--- Task (e): Instantaneous Jerk vs Time ---")

# Jerk = da/dt — further smooth acceleration before differencing
accel_smooth = savgol_filter(accel, 
                              window_length=min(7, len(df) if len(df) % 2 == 1 else len(df)-1),
                              polyorder=2)

jerk = np.zeros(len(df))
for i in range(len(df)):
    if i == 0:
        jerk[i] = (accel_smooth[1] - accel_smooth[0]) / (time_arr[1] - time_arr[0])
    elif i == len(df) - 1:
        jerk[i] = (accel_smooth[-1] - accel_smooth[-2]) / (time_arr[-1] - time_arr[-2])
    else:
        jerk[i] = (accel_smooth[i+1] - accel_smooth[i-1]) / (time_arr[i+1] - time_arr[i-1])

df['jerk_ms3'] = jerk

fig, ax = plt.subplots(figsize=(14, 7))
colors_jrk = [C_BLUE if j >= 0 else C_ORANGE for j in jerk]
ax.bar(df['time_min'], jerk, width=0.13, color=colors_jrk, alpha=0.75, edgecolor='none')
ax.plot(df['time_min'], jerk, color='#333333', linewidth=0.8, alpha=0.4)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Jerk (m/s³)')
ax.set_title('Task (e): Instantaneous Jerk vs Time')

legend_jrk = [Patch(facecolor=C_BLUE, alpha=0.75, label='Positive Jerk'),
              Patch(facecolor=C_ORANGE, alpha=0.75, label='Negative Jerk')]
ax.legend(handles=legend_jrk, loc='upper right', framealpha=0.9)
ax.set_xlim(0, df['time_min'].iloc[-1])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "e_jerk_vs_time.png"), bbox_inches='tight')
plt.close()
print(f"  Max jerk:  {np.max(jerk):.6f} m/s³")
print(f"  Min jerk:  {np.min(jerk):.6f} m/s³")
print("  ✓ Saved: e_jerk_vs_time.png")

# ============================================================================
# BONUS: Distance Covered & Remaining vs Time
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['time_min'], df['dist_covered_integrated_km'], color=C_TEAL, linewidth=2.5,
        label='Distance Covered (speed integration)')
ax.plot(df['time_min'], df['dist_covered_gmap_km'], color=C_GREEN, linewidth=1.5,
        linestyle=':', marker='s', markersize=4, alpha=0.6,
        label='Distance Covered (Google Maps)')
ax.plot(df['time_min'], df['distance_remaining_km'], color=C_RED, linewidth=1.5,
        linestyle='--', marker='^', markersize=4, alpha=0.6,
        label='Distance Remaining')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Distance (km)')
ax.set_title('Distance Covered & Remaining vs Time')
ax.legend(loc='center right', framealpha=0.9)
ax.set_xlim(0, df['time_min'].iloc[-1])
ax.set_ylim(0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bonus_distance_vs_time.png"), bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: bonus_distance_vs_time.png")

# ============================================================================
# EXPORT PROCESSED DATA
# ============================================================================
export_df = df[['frame', 'video_time_s', 'time_min', 'phone_clock', 'speed_kmh',
                'calc_speed_kmh', 'distance_remaining_km', 'dist_covered_integrated_km',
                'acceleration_ms2', 'jerk_ms3', 'time_to_10km_min']].copy()
export_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
export_df.to_csv(export_path, index=False, float_format='%.4f')
print(f"\n  ✓ Saved processed data: processed_data.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"\nFiles generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  • {f} ({size_kb:.0f} KB)")

print(f"\n{'─'*50}")
print(f"  SUMMARY STATISTICS")
print(f"{'─'*50}")
print(f"  Journey distance (Google Maps):  {actual_gmap:.1f} km")
print(f"  Journey distance (integrated):   {integrated_distance:.2f} km")
print(f"  Journey duration:                {journey_duration_min:.1f} min")
print(f"  Speed — avg: {avg_speed:.1f}, min: {df['speed_kmh'].min()}, max: {df['speed_kmh'].max()} km/h")
print(f"  Speed RMSE (calc vs recorded):   {rmse:.2f} km/h")
print(f"  Distance % error:                {pct_err_integrated:.2f}%")
print(f"  Max |acceleration|:              {np.max(np.abs(accel)):.4f} m/s²")
print(f"  Max |jerk|:                      {np.max(np.abs(jerk)):.6f} m/s³")
print(f"{'─'*50}")
