import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys
import os

# Add src to path for config
sys.path.append(os.path.dirname(__file__))
from config import *

# --- Matplotlib High-Visibility Settings ---
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.3

# --- 1. Load Hardware Noise Data (from noise_fsp logic) ---
INPUT_DIR = BASE_DIR / "input" / "Noise_file"
freq_noise = []
amp_noise_sum = []
count_noise = 0

for i in range(5):
    fpath = INPUT_DIR / f"Noise_{i}.csv"
    if fpath.exists():
        with open(fpath, 'r', encoding='UTF-8') as f:
            reader = csv.DictReader(f)
            freqs, vals = [], []
            for row in reader:
                freqs.append(float(row['Hz'])) # Hz
                vals.append(float(row['dBc/Hz']))
            if count_noise == 0:
                freq_noise = np.array(freqs)
                amp_noise_sum = np.zeros_like(freq_noise)
            amp_noise_sum += np.array(vals)
            count_noise += 1

avg_noise_floor = amp_noise_sum / count_noise if count_noise > 0 else []
# Convert dBc/Hz to dBm/Hz assuming Carrier = 0 dBm
noise_dbm_hz = avg_noise_floor + 0.0 

import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Plot Noise vs Signal Peaks for multiple deltas.')
parser.add_argument('--deltas', type=float, nargs='+', default=None, help='List of energy spread values (e.g., 1.0 1.6 2.2)')
args = parser.parse_args()

# --- 2. Load Simulation Signal Data ---
try:
    amp_df = pd.read_csv(LATEST_SCAN_DIR / "scan_amplitudes.csv")
    tune_df = pd.read_csv(LATEST_SCAN_DIR / "scan_tunes.csv")
    df = pd.merge(amp_df, tune_df, on=['alpha_c_1e4', 'delta_1e4'])
    available_deltas = sorted(df['delta_1e4'].unique())
    
    if args.deltas is None:
        # Pick 3 representative deltas if none specified
        target_deltas = [available_deltas[0], available_deltas[len(available_deltas)//2], available_deltas[-1]]
    else:
        target_deltas = []
        for d in args.deltas:
            val = d * 1e4 if d < 0.01 else d
            matched = min(available_deltas, key=lambda x: abs(x - val))
            if matched not in target_deltas:
                target_deltas.append(matched)
    
    print(f"Selected Deltas for Overlay: {target_deltas}")

except Exception as e:
    print(f"Error loading simulation data: {e}")
    sys.exit(1)

# --- 3. Plotting ---
plt.figure(figsize=(12, 7))

# A. Plot Noise Floor (Continuous)
plt.semilogx(freq_noise / 1000.0, noise_dbm_hz, color='blue', alpha=0.6, label='Hardware Noise Floor')

# B. Plot Signal Peaks for each Delta
import matplotlib.colors as mcolors
num_colors = 20 # Increased for smoother scale if needed
cmap = plt.get_cmap('turbo', num_colors)
norm = mcolors.BoundaryNorm(np.linspace(df['alpha_c_1e4'].min(), df['alpha_c_1e4'].max(), num_colors + 1), cmap.N)

line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D', '*']

for i, d in enumerate(target_deltas):
    subset = df[df['delta_1e4'] == d].copy()
    subset = subset.sort_values('alpha_c_1e4')
    
    # Decimation for clarity
    if len(subset) > 15:
        idx = np.linspace(0, len(subset)-1, 15, dtype=int)
        subset = subset.iloc[idx]
    # Calculate Power and Frequency
    v_sig_mv = POS_SENSITIVITY * (subset['Z_amp_um']/1000.0) * BUNCH_CHARGE_NC
    subset['Power_dBm'] = 10 * np.log10( (((v_sig_mv/1000.0)**2 / (2 * IMPEDANCE)) * 1000.0).replace(0, np.nan) )
    subset['fs_kHz'] = subset['vz_from_Z_MHz'] * 1000.0
    
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    
    # Plot connecting line
    plt.plot(subset['fs_kHz'], subset['Power_dBm'], color='black', linestyle=style, linewidth=1.2, alpha=0.7)
    
    # Plot scatter points with distinct markers
    sc = plt.scatter(subset['fs_kHz'], subset['Power_dBm'], 
                     c=subset['alpha_c_1e4'], cmap=cmap, norm=norm,
                     marker=marker,
                     s=50, edgecolors='black', linewidth=0.3, zorder=3, 
                     label=f'Delta={d:.2f}e-4')

# C. Decorations
cbar = plt.colorbar(sc, ticks=np.linspace(df['alpha_c_1e4'].min(), df['alpha_c_1e4'].max(), 10))
cbar.set_label(r'Momentum Compaction $\alpha_c$ [$10^{-4}$]', fontweight='bold')

plt.xlabel('Frequency [kHz]', fontsize=14)
plt.ylabel('Power [dBm or dBm/Hz]', fontsize=14)
plt.title(f'Multi-Delta Overlay: Noise vs Synchrotron Signal Peaks (Z)\n(Current={BEAM_CURRENT}uA)', fontsize=15)

plt.xlim(0.01, 100)
plt.ylim(-110, -60)
plt.grid(True, which='both', linestyle=':', alpha=0.4)
plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

# Save result
deltas_str = "_".join([f"{d:.1f}" for d in target_deltas])
output_path = LATEST_SCAN_DIR / f"Noise_ZSignal_Overlay_MultiDelta.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Multi-delta overlay plot saved to: {output_path}")
plt.show()
