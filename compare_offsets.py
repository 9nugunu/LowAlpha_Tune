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
        self.beta_x = 7.08
        self.V_rf = 0.5e6                 # V
        self.f_rf = 500e6                 # Hz
        self.w_rf = 2 * np.pi * self.f_rf # rad/s
        self.e_x = 190e-9                 # m * rad
        
        # Higher order momentum compaction (from ELEGANT)
        self.alphac2 = 6.15e-2            # Quadratic term
        self.alphac3 = -9.77              # Cubic term

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
    """Synchrotron angular frequency w_s(alpha). Linear approximation."""
    return np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)


def synchrotron_frequency_higher_order(alpha, delta, order=2, params=PARAMS):
    """Synchrotron angular frequency w_s(alpha, delta) with higher-order momentum compaction.
    
    order=1: Linear (alpha_1 only)
    order=2: Quadratic (alpha_1 + alpha_2 * delta)
    order=3: Cubic (alpha_1 + alpha_2 * delta + alpha_3 * delta^2)
    """
    if order == 1:
        alpha_eff = alpha
    elif order == 2:
        alpha_eff = alpha + params.alphac2 * delta
    else:  # order >= 3
        alpha_eff = alpha + params.alphac2 * delta + params.alphac3 * delta**2
    
    # Ensure alpha_eff is positive to avoid sqrt of negative
    alpha_eff = np.maximum(alpha_eff, 1e-10)
    return np.sqrt(-params.V_rf * params.w_rf * alpha_eff * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)


def x_offset(alpha, delta, params=PARAMS):
    """Transverse offset from 00_long_x (uses Bessel J1).
    Note: Factor of 2 for both sidebands (vx±vz).
    """
    w_s = synchrotron_frequency(alpha, params)
    mu_s = w_s * params.T_0 / (2 * np.pi)
    return np.sqrt(params.e_x * params.beta_x) * 2 * sp.j1(delta / mu_s)  # ×2 for two sidebands


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
    """Creates a subfolder under the current scan directory for results."""
    out_dir = LATEST_SCAN_DIR / runfilename
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
    
    # Load X_main if available
    x_main_amps = None
    X_MAIN_PATH = LATEST_SCAN_DIR / "X_main.csv"
    if X_MAIN_PATH.exists():
        X_main_df = pd.read_csv(X_MAIN_PATH, index_col=0)
        X_main_vals = X_main_df.values
        x_main_amps = X_main_vals[idx, :]

    return alpha_grid, alpha_axis, x_offsets_um, z_offsets_um, delta_used, x_main_amps


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
    ax.plot(alpha_axis, th_x_dbm, color="red", linewidth=2.5, label=f"X Theory")
    ax.plot(alpha_axis, sim_x_dbm, color="red", linestyle="--", marker="o", markersize=6, alpha=0.7, label=f"X Sim")

    # Plot Z curves (Blue theme)
    ax.plot(alpha_axis, th_z_dbm, color="blue", linewidth=2.5, label=f"Z Theory")
    ax.plot(alpha_axis, sim_z_dbm, color="blue", linestyle="--", marker="s", markersize=6, alpha=0.7, label=f"Z Sim")

    # Overlay Noise Floor
    # ax.axhline(y=ASSUMED_NOISE_FLOOR, color='black', linestyle=':', linewidth=1.5, label=f'Hardware Noise Floor ({ASSUMED_NOISE_FLOOR} dBm)')

    ax.set_xlabel(r"momentum compaction $\alpha_c \times 10^{-4}$")
    ax.set_ylabel(r"Signal Power (dBm)")
    ax.set_title(f"Comparison of X/Z Signal Power\n(delta={delta_target:.2e}, beam={BEAM_CURRENT}uA)")
    
    style_axes(ax)
    ax.legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    out_path = out_dir / f"overlay_powers_delta{delta_target:.2e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved overlay power plot to {out_path}")
    return out_path


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
    parser.add_argument("--delta", type=float, default=2.3e-4, help="Energy spread (delta) to use for theory curve and nearest simulation slice.")
    parser.add_argument(
        "--runfilename",
        default=DEFAULT_RUNFILENAME,
        help="Subfolder name under the current scan directory where plots will be saved.",
    )
    # Add argument for calculating alpha from measured fs
    parser.add_argument("--measured_fs", type=float, default=None, help="Measured synchrotron frequency (Hz) to calc alpha")
    
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.runfilename)

    alpha_grid, alpha_axis, sim_x, sim_z, delta_used, x_main_amps = load_simulation_slice(args.delta)
    th_x, th_z = theoretical_offsets_um(alpha_grid, args.delta)

    plot_combined_powers(alpha_axis, sim_x, sim_z, th_x, th_z, args.delta, delta_used, out_dir)
    plot_combined_offsets(alpha_axis, sim_x, sim_z, th_x, th_z, args.delta, delta_used, out_dir, x_main_amps)
    plot_modulation_index(alpha_grid, alpha_axis, sim_x, args.delta, out_dir, x_main_amps)
    plot_total_amplitude_comparison(alpha_axis, sim_x, th_x, args.delta, out_dir, x_main_amps)
    plot_alpha2_comparison(alpha_grid, alpha_axis, sim_x, args.delta, out_dir, x_main_amps)
    
    if args.measured_fs is not None:
        alpha_calc = calculate_alpha_from_fs(args.measured_fs)
        print(f"\n[Inverse Calculation]")
        print(f"Measured fs: {args.measured_fs/1000.:.3f} kHz")
        print(f"Calculated Alpha_c: {alpha_calc:.3e}")
        print(f"Calculated Alpha_c (1e-4 scale): {alpha_calc*1e4:.4f}")


def plot_combined_offsets(alpha_axis, sim_x, sim_z, th_x, th_z, delta_target, delta_used, out_dir: Path, x_main_amps=None):
    """Plot offsets. Scales THEORY to match simulation carrier amplitude."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate theoretical carrier amplitude (Envelope)
    th_carrier_um = np.sqrt(PARAMS.e_x * PARAMS.beta_x) / 1e-6

    # Scaling Logic: Scale THEORY DOWN to simulation carrier
    if x_main_amps is not None and np.all(x_main_amps > 1e-9):
        avg_main = np.median(x_main_amps)
        scale_factor = avg_main / th_carrier_um  # < 1 typically
        
        # Scale theory to match simulation carrier
        th_x_scaled = th_x * scale_factor
        label_suffix = " (Scaled)"
        
        # Plot original theory for reference (transparent)
        # ax.plot(alpha_axis, th_x, color="red", linestyle=":", alpha=0.3, label="X Theory (Original)")
    else:
        th_x_scaled = th_x
        avg_main = th_carrier_um
        scale_factor = 1.0
        label_suffix = ""

    # Plot X curves (Red theme) - Theory scaled, Simulation raw
    # ax.plot(alpha_axis, th_x_scaled, color="red", linewidth=2.5, label=f"X Theory{label_suffix}")
    ax.plot(alpha_axis, sim_x, color="red", linestyle="--", marker="o", markersize=6, alpha=0.7, label=f"X ELEGANT")

    # Plot Z curves (Blue theme)
    ax.plot(alpha_axis, th_z, color="blue", linewidth=2.5, label=f"Z Theory")
    ax.plot(alpha_axis, sim_z, color="blue", linestyle="--", marker="s", markersize=6, alpha=0.7, label=f"Z ELEGANT")
    
    # Text annotation for normalization
    ax.text(0.05, 0.95, f"Theory Carrier: {th_carrier_um:.1f} um\nSim Carrier (avg): {avg_main:.1f} um\nScale Factor: {scale_factor:.4f}", 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax.set_ylabel(r"Amplitude [$\mu m$]")
    ax.set_title(f"delta={delta_target:.2e}")
    
    style_axes(ax)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = out_dir / f"overlay_offsets_delta{delta_target:.2e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved overlay offset plot to {out_path}")


def plot_total_amplitude_comparison(alpha_axis, sim_x, th_x, delta_target, out_dir: Path, x_main_amps=None):
    """Compare Raw Theory Offset vs Simulation Total Amplitude (Main + Side)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Theory (Raw Offset)
    ax.plot(alpha_axis, th_x, color="magenta", linewidth=2.5, label=f"X Theory (Offset Only)")

    # Plot Simulation
    if x_main_amps is not None and np.all(x_main_amps > 1e-9):
        # Calculate Total Amplitude (Main + Sideband)
        sim_x_total = sim_x + x_main_amps
        
        # Plot Total
        ax.plot(alpha_axis, sim_x_total, color="darkgreen", linestyle="-", marker="^", markersize=6, alpha=0.7, label=f"X ELEGANT (Total = Main + Side)")
        
        # Plot Sideband Only (for reference)
        ax.plot(alpha_axis, sim_x, color="red", linestyle="--", marker="o", markersize=6, alpha=0.3, label=f"X ELEGANT (Sideband Only)")
        
        # Annotate average values
        avg_theory = np.mean(th_x)
        avg_sim_total = np.mean(sim_x_total)
        avg_sim_main = np.median(x_main_amps)
        
        text_str = (f"Theory (Offset) avg: {avg_theory:.1f} um\n"
                    f"Sim (Total) avg: {avg_sim_total:.1f} um\n"
                    f"Sim (Main) avg: {avg_sim_main:.1f} um")
        
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    else:
        ax.text(0.5, 0.5, "X_main data missing", transform=ax.transAxes, ha='center')

    ax.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax.set_ylabel(r"Amplitude [$\mu m$]")
    ax.set_title(f"Hypothesis Check: Theory Offset vs Sim Total (delta={delta_target:.2e})")
    
    style_axes(ax)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = out_dir / f"hypothesis_total_amp_delta{delta_target:.2e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved total amplitude comparison to {out_path}")


def plot_modulation_index(alpha_grid, alpha_axis, sim_x, delta_target, out_dir: Path, x_main_amps=None):
    """Compare dimensionless Modulation Index: Theory J₁(δ/μs) vs Sim (X_side/X_main)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Theory Modulation Index = 2 × J₁(δ/μs) for both sidebands
    w_s = synchrotron_frequency(alpha_grid)
    mu_s = w_s * PARAMS.T_0 / (2 * np.pi)
    th_mod_idx = 2 * sp.j1(delta_target / mu_s)  # ×2 for two sidebands
    
    # Simulation Modulation Index = X_side / X_main
    if x_main_amps is not None and np.all(x_main_amps > 1e-9):
        sim_mod_idx = sim_x / x_main_amps
        
        # Calculate correlation / ratio
        valid_mask = (th_mod_idx > 1e-9) & (sim_mod_idx > 1e-9)
        ratio_label = ""
        if np.any(valid_mask):
            ratio = np.median(sim_mod_idx[valid_mask] / th_mod_idx[valid_mask])
            ratio_label = f" (Ratio: {ratio:.3f})"
        
        # Plot both
        ax.plot(alpha_axis, th_mod_idx, color="purple", linewidth=2.5, label=r"Modeling: $2 J_1(\delta/\mu_s)$")
        ax.plot(alpha_axis, sim_mod_idx, color="green", linestyle="-", marker="o", markersize=6, alpha=0.7, 
                label=r"ELEGANT: $X_{side}/X_{main}$" + "\n" +ratio_label)
    else:
        ax.text(0.5, 0.5, "X_main data not available", transform=ax.transAxes, 
                fontsize=16, ha='center', va='center')
        return
    
    ax.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax.set_ylabel("Modulation ratio")
    ax.set_title(fr"Comparison of modulation ($\delta={delta_target:.2e}$)")
    
    # Set Y-axis limits for consistent comparison (2*J1 max is ~1.16)
    y_max = max(1.0, np.max(th_mod_idx) * 1.1)
    if sim_mod_idx is not None:
        y_max = max(y_max, np.max(sim_mod_idx) * 1.1)
    ax.set_ylim(0, y_max)
    
    style_axes(ax)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = out_dir / f"modulation_ratio_delta_{delta_target:.2e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved modulation ratio plot to {out_path}")


def plot_alpha2_comparison(alpha_grid, alpha_axis, sim_x, delta_target, out_dir: Path, x_main_amps=None):
    """Compare Modulation Index with different orders of momentum compaction (alpha_1, alpha_2, alpha_3).
    
    Shows how higher-order terms cause 'saturation' at low alpha values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Synchrotron Tune comparison
    ax1 = axes[0]
    
    # Linear (1st order): nu_s(alpha_1 only)
    w_s_linear = synchrotron_frequency_higher_order(alpha_grid, delta_target, order=1)
    mu_s_linear = w_s_linear * PARAMS.T_0 / (2 * np.pi)
    nu_s_linear_kHz = w_s_linear / (2 * np.pi * 1000)
    
    # Quadratic (2nd order): nu_s(alpha_1 + alpha_2 * delta)
    w_s_quad = synchrotron_frequency_higher_order(alpha_grid, delta_target, order=2)
    mu_s_quad = w_s_quad * PARAMS.T_0 / (2 * np.pi)
    nu_s_quad_kHz = w_s_quad / (2 * np.pi * 1000)
    
    # Cubic (3rd order): nu_s(alpha_1 + alpha_2 * delta + alpha_3 * delta^2)
    w_s_cubic = synchrotron_frequency_higher_order(alpha_grid, delta_target, order=3)
    mu_s_cubic = w_s_cubic * PARAMS.T_0 / (2 * np.pi)
    nu_s_cubic_kHz = w_s_cubic / (2 * np.pi * 1000)
    
    ax1.plot(alpha_axis, nu_s_linear_kHz, color="blue", linewidth=2.5, label=r"$\nu_s$ (1st: $\alpha_1$)")
    ax1.plot(alpha_axis, nu_s_quad_kHz, color="orange", linewidth=2.5, linestyle="--", label=r"$\nu_s$ (2nd: $+\alpha_2 \delta$)")
    ax1.plot(alpha_axis, nu_s_cubic_kHz, color="red", linewidth=2.5, linestyle="-.", label=r"$\nu_s$ (3rd: $+\alpha_3 \delta^2$)")
    
    ax1.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax1.set_ylabel(r"$\nu_s$ [kHz]")
    ax1.set_title(r"$\nu_s$ variation")
    
    # Text box with coefficients
    text_str = (f"$\\alpha_2$ = {PARAMS.alphac2:.2e}\n"
                f"$\\alpha_3$ = {PARAMS.alphac3:.2e}\n"
                f"$\\alpha_2 \\delta$ = {PARAMS.alphac2 * delta_target:.2e}\n"
                f"$\\alpha_3 \\delta^2$ = {PARAMS.alphac3 * delta_target**2:.2e}")
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    style_axes(ax1)
    ax1.legend(loc='center right')
    
    # Right plot: Modulation Index comparison
    ax2 = axes[1]
    
    # Modulation Index for each order
    mod_idx_linear = 2 * sp.j1(delta_target / mu_s_linear)
    mod_idx_quad = 2 * sp.j1(delta_target / mu_s_quad)
    mod_idx_cubic = 2 * sp.j1(delta_target / mu_s_cubic)
    
    ax2.plot(alpha_axis, mod_idx_linear, color="blue", linewidth=2.5, label=r"$2J_1$ (1st order)")
    ax2.plot(alpha_axis, mod_idx_quad, color="orange", linewidth=2.5, linestyle="--", label=r"$2J_1$ (2nd order)")
    ax2.plot(alpha_axis, mod_idx_cubic, color="red", linewidth=2.5, linestyle="-.", label=r"$2J_1$ (3rd order)")
    
    # Add simulation data if available
    if x_main_amps is not None and np.all(x_main_amps > 1e-9):
        sim_mod_idx = sim_x / x_main_amps
        ax2.plot(alpha_axis, sim_mod_idx, color="green", linestyle="-", marker="o", markersize=5, alpha=0.7, label="Simulation")
    
    ax2.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax2.set_ylabel("Modulation")
    ax2.set_title(r"Modulation variation")
    
    # Set Y-axis limits for consistent comparison
    y_max_th = max(np.max(mod_idx_linear), np.max(mod_idx_quad), np.max(mod_idx_cubic))
    y_max = max(1.0, y_max_th * 1.1)
    if x_main_amps is not None and np.all(x_main_amps > 1e-9):
        y_max = max(y_max, np.max(sim_x / x_main_amps) * 1.1)
    ax2.set_ylim(0, y_max)
    
    style_axes(ax2)
    ax2.legend(loc='upper right')

    fig.suptitle(fr"$\alpha$ higher order effects ($\delta = {delta_target:.2e}$)", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = out_dir / f"alpha_order_comparison_delta{delta_target:.2e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved higher-order alpha comparison plot to {out_path}")


if __name__ == "__main__":
    main()
