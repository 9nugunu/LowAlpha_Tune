"""
Combine theoretical offsets (from long_offsets) with simulation results (result_elegant).

Plots x/z offsets vs. momentum compaction for a chosen energy spread (delta) and saves
the combined figure under output/<runfilename>/.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

import os
import sys

# Non-interactive backend for headless runs.
plt.switch_backend("Agg")

# Add src to path for config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *


from src.visualization import PlotConfig


PLOT_CONFIG = PlotConfig()
PLOT_CONFIG.apply_settings()

DEFAULT_RUNFILENAME = "compare_powers"
# Grid definitions will be auto-loaded from metadata if possible,
# but for simplicity we'll try to extract them from the csv indices later or assume the previous scan range.
# Let's use the ones already in the file but update paths to the new config.
X_PATH = LATEST_SCAN_DIR / "X.csv"
Z_PATH = LATEST_SCAN_DIR / "Z.csv"

# Grid definitions: These should ideally match the simulation grid.
# We'll try to read them from the CSV if index_col=0 is used.


class MachineParams:
    def __init__(self):
        # Constants
        self.e_charge = 1.602176634e-19
        self.c = 299792458                 # m/s
        self.E_0 = 629e6                  # eV
        self.U_0 = 9.1e3                  # eV
        self.T_0 = T_REV                  # Use config REV period
        self.gam = self.E_0 / (0.511e6) + 1

        # Lattice / RF
        self.beta_x = 1.8
        self.V_rf = 0.5e6                 # V
        self.f_rf = 500e6                 # Hz
        self.w_rf = 2 * np.pi * self.f_rf # rad/s
        self.e_x = 190e-9                 # m * rad

        # Derived
        self.phi_s = np.pi - np.arcsin(self.U_0 / self.V_rf)

 
PARAMS = MachineParams()


def style_axes(axes):
    """Apply minor ticks and grids to axes (works for single axis or array)."""
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-")
        ax.grid(which="minor", linestyle=":", alpha=0.5)


def synchrotron_frequency(alpha, params=PARAMS):
    """Synchrotron angular frequency w_s(alpha)."""
    return np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)


def x_offset(alpha, delta, params=PARAMS):
    """Transverse offset from 00_long_x (uses Bessel J1)."""
    w_s = synchrotron_frequency(alpha, params)
    mu_s = w_s * params.T_0 / (2 * np.pi)
    return np.sqrt(params.e_x * params.beta_x) * sp.j1(delta / mu_s) #/ (2*np.sqrt(2)) # factor retained


def z_offset(alpha, delta, params=PARAMS):
    """Longitudinal offset from 00_long_z."""
    w_s = synchrotron_frequency(alpha, params)
    return (alpha - 1 / params.gam**2) * params.c / w_s * delta

def offset_to_power(offset_um):
    """Convert offset in micrometers to Power in dBm using config parameters."""
    # V_peak (mV) = S (mV/mm/nC) * x (mm) * Q (nC)
    v_peak_mv = POS_SENSITIVITY * (offset_um / 1000.0) * BUNCH_CHARGE_NC
    # P_mW = (V_peak/1000)^2 / (2 * R) * 1000
    p_mw = ((v_peak_mv / 1000.0)**2) / (2 * IMPEDANCE) * 1000.0
    # Avoid log of zero
    with np.errstate(divide='ignore'):
        p_dbm = 10 * np.log10(p_mw)
    return p_dbm


def ensure_output_dir(runfilename: str) -> Path:
    out_dir = Path("output") / runfilename
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_simulation_slice(delta_target: float):
    import pandas as pd
    if not X_PATH.exists() or not Z_PATH.exists():
        raise SystemExit(f"Missing simulation data: {X_PATH} or {Z_PATH}")

    X_df = pd.read_csv(X_PATH, index_col=0)
    Z_df = pd.read_csv(Z_PATH, index_col=0)
    
    alpha_axis = np.array(X_df.columns, dtype=float)
    alpha_grid = alpha_axis * 1e-4
    delta_norm = np.array(X_df.index, dtype=float)
    delta_grid = delta_norm * 1e-4

    X = X_df.values
    Z = Z_df.values

    delta_target_norm = delta_target / 1e-4
    idx = int(np.argmin(np.abs(delta_norm - delta_target_norm)))
    delta_used = delta_grid[idx]

    x_offsets_um = X[idx, :]
    z_offsets_um = Z[idx, :]
    return alpha_grid, alpha_axis, x_offsets_um, z_offsets_um, delta_used


def theoretical_offsets_um(alpha_grid: np.ndarray, delta: float):
    x_um = x_offset(alpha_grid, delta) / 1e-6
    z_um = z_offset(alpha_grid, delta) / 1e-6
    return x_um, z_um


def plot_combined_powers(alpha_axis, sim_x, sim_z, th_x, th_z, delta_target, delta_used, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert um to dBm
    sim_x_dbm = offset_to_power(sim_x)
    th_x_dbm = offset_to_power(th_x)
    sim_z_dbm = offset_to_power(sim_z)
    th_z_dbm = offset_to_power(th_z)

    # Plot X curves (Red theme)
    ax.plot(alpha_axis, th_x_dbm, color="tab:red", linewidth=2.5, label=f"X Theory")
    ax.plot(alpha_axis, sim_x_dbm, color="tab:red", linestyle="--", marker="o", markersize=6, alpha=0.7, label=f"X Sim")

    # Plot Z curves (Blue theme)
    ax.plot(alpha_axis, th_z_dbm, color="tab:blue", linewidth=2.5, label=f"Z Theory")
    ax.plot(alpha_axis, sim_z_dbm, color="tab:blue", linestyle="--", marker="s", markersize=6, alpha=0.7, label=f"Z Sim")

    # Overlay Noise Floor
    ax.axhline(y=ASSUMED_NOISE_FLOOR, color='black', linestyle=':', linewidth=1.5, label=f'Hardware Noise Floor ({ASSUMED_NOISE_FLOOR} dBm)')

    ax.set_xlabel(r"momentum compaction $\alpha_c \times 10^{-4}$", fontsize=16, fontweight='bold')
    ax.set_ylabel(r"Signal Power (dBm)", fontsize=16, fontweight='bold')
    ax.set_title(f"Comparison of X/Z Signal Power\n(delta={delta_target:.2e}, beam={BEAM_CURRENT}uA)", fontsize=18, fontweight='bold')
    
    style_axes(ax)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    out_path = out_dir / f"overlay_powers_delta{delta_target:.1e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved overlay power plot to {out_path}")
    return out_path


    print(synchrotron_frequency(50e3))


def calculate_alpha_from_fs(fs_Hz):
    """
    Inverse calculation of alpha_c from synchrotron frequency fs using MachineParams.
    Formula: alpha_c = (fs / f_rev)**2 * (2*pi*E_beam) / (h * V_rf * cos(phi_s))
    """
    # Use globally defined PARAMS
    T_rev = PARAMS.T_0
    f_rev = 1.0 / T_rev
    E_beam = PARAMS.E_0
    V_rf = PARAMS.V_rf
    
    # Calculate Harmonic Number h = f_rf / f_rev
    h = PARAMS.f_rf / f_rev
    
    # Use derived synchronous phase phi_s
    # Note: PARAMS.phi_s is calculated as pi - arcsin(U0/Vrf)
    # Cosine term usually takes the absolute value or accounts for the specific bucket
    cos_phi = np.abs(np.cos(PARAMS.phi_s))
    
    numerator = 2 * np.pi * E_beam
    denominator = h * V_rf * cos_phi
    
    # alpha_c calculation
    alpha = (fs_Hz / f_rev)**2 * (numerator / denominator)
    
    # Debug info for verification
    print(f"\n[Params Used] E={E_beam/1e6:.1f}MeV, V_rf={V_rf/1e3:.1f}kV, h={h:.1f}, f_rev={f_rev/1e6:.3f}MHz")
    
    return alpha


def main():
    parser = argparse.ArgumentParser(description="Compare theoretical and simulation offsets vs. alphac for a chosen delta.")
    parser.add_argument("--delta", type=float, default=2.2e-4, help="Energy spread (delta) to use for theory curve and nearest simulation slice.")
    parser.add_argument(
        "--runfilename",
        default=DEFAULT_RUNFILENAME,
        help="Folder name under output/ where plots will be saved.",
    )
    # Add argument for calculating alpha from measured fs
    parser.add_argument("--measured_fs", type=float, default=None, help="Measured synchrotron frequency (Hz) to calc alpha")
    
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.runfilename)

    alpha_grid, alpha_axis, sim_x, sim_z, delta_used = load_simulation_slice(args.delta)
    th_x, th_z = theoretical_offsets_um(alpha_grid, args.delta)

    plot_combined_powers(alpha_axis, sim_x, sim_z, th_x, th_z, args.delta, delta_used, out_dir)
    
    if args.measured_fs is not None:
        alpha_calc = calculate_alpha_from_fs(args.measured_fs)
        print(f"\n[Inverse Calculation]")
        print(f"Measured fs: {args.measured_fs/1000.:.3f} kHz")
        print(f"Calculated Alpha_c: {alpha_calc:.3e}")
        print(f"Calculated Alpha_c (1e-4 scale): {alpha_calc*1e4:.4f}")
    
    # print(synchrotron_frequency(50e3)) 


if __name__ == "__main__":
    main()
