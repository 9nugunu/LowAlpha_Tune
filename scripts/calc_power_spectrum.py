import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys
import os

# Add src to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import *

# --- Matplotlib Settings ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['grid.alpha'] = 0.3

# --- Paths ---
BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "input"
NOISE_DIR = INPUT_DIR / "Noise_file"
# TODO: Scan directory might need partial matching if hash varies. 
# For now, hardcoding the known output path or searching for latest.
OUTPUT_ROOT = BASE_DIR / "output" / "scan_alphac_pyele"
# Find latest scan dir
# LATEST_SCAN_DIR = scan_dirs[-1]
print(f"Using scan result: {LATEST_SCAN_DIR}")

# --- 1. Load Simulation Data ---
try:
    amp_df = pd.read_csv(LATEST_SCAN_DIR / "scan_amplitudes.csv")
    tune_df = pd.read_csv(LATEST_SCAN_DIR / "scan_tunes.csv")
    
    # Merge on alpha, delta to ensure alignment
    # Start by rounding to avoid float mismatch if needed, but CSV should differ.
    df = pd.merge(amp_df, tune_df, on=['alpha_c_1e4', 'delta_1e4'])
    print(f"Loaded {len(df)} simulation points.")
except Exception as e:
    print(f"Error loading CSVs: {e}")
    sys.exit(1)

# --- 2. Calculate Power (dBm) ---
# Signal Power (modulated by offset x)
# V_sig_peak = S_pos * x_mm * Q
# Power (dBm) = 10 * log10( (V_peak/sqrt(2))^2 / R / 1mW )
df['V_sig_peak_mV'] = POS_SENSITIVITY * (df['X_amp_um']/1000.0) * BUNCH_CHARGE_NC
# P_mW = (V_peak/1000)^2 / (2 * 50) * 1000
df['Power_mW'] = ((df['V_sig_peak_mV']/1000.0)**2) / (2 * IMPEDANCE) * 1000.0
df['Power_dBm'] = 10 * np.log10(df['Power_mW'].replace(0, np.nan))

# Simulation Noise Floor in dBm (using Noise_X_raw)
df['Noise_X_mW'] = ((POS_SENSITIVITY * (df['Noise_X_raw']/1e6) * BUNCH_CHARGE_NC/1000.0)**2) / (2 * IMPEDANCE) * 1000.0
df['Noise_X_dBm'] = 10 * np.log10(df['Noise_X_mW'].replace(0, np.nan))
avg_sim_noise = df['Noise_X_dBm'].mean()

# --- 3. Hardware Noise Baseline ---
# User requested fixed -90 dBm or Carrier=0 reference
P_HARDWARE_NOISE_DBM = ASSUMED_NOISE_FLOOR

# --- 4. Final Power-Alpha Plot ---
plt.figure(figsize=(10, 6))
deltas = df['delta_1e4'].unique()

for d in deltas:
    subset = df[df['delta_1e4'] == d]
    plt.plot(subset['alpha_c_1e4'], subset['Power_dBm'], 'o-', linewidth=2, label=f'Signal Power (delta={d}e-4)')

# Overlay Hardware Noise Baseline
plt.axhline(y=P_HARDWARE_NOISE_DBM, color='red', linestyle='--', linewidth=1.5, label=f'Hardware Noise Floor ({P_HARDWARE_NOISE_DBM} dBm)')

# Overlay Simulation Noise Baseline
plt.axhline(y=avg_sim_noise, color='green', linestyle=':', alpha=0.6, label=f'Sim-Noise Floor ({avg_sim_noise:.1f} dBm)')

plt.xlabel(r'$\alpha_c$ index ($10^{-4}$)', fontsize=14)
plt.ylabel('Power [dBm]', fontsize=14)
plt.title(f'Power vs Alpha_c (Current={BEAM_CURRENT}uA, Sensitivity={POS_SENSITIVITY}mV/mm/nC)', fontsize=14)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(frameon=True, loc='best')

# Dominance Annotation
dom_subset = df[df['Power_dBm'] > P_HARDWARE_NOISE_DBM]
if not dom_subset.empty:
    min_a = dom_subset['alpha_c_1e4'].min()
    plt.annotate(f'Detectable @ alpha > {min_a}', 
                 xy=(min_a, P_HARDWARE_NOISE_DBM), xytext=(min_a, P_HARDWARE_NOISE_DBM+10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=11, fontweight='bold', color='darkred')

plt.tight_layout()
save_path = LATEST_SCAN_DIR / "Power_vs_Alpha_Simple.png"
plt.savefig(save_path, dpi=200)
print(f"Simple plot saved to: {save_path}")
plt.show()
