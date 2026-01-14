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

# Non-interactive backend for headless runs.
plt.switch_backend("Agg")


@dataclass
class PlotConfig:
    """Plot style configuration."""
    label_size: int = 24
    label_weight: str = "bold"
    font_weight: str = "bold"
    font_size: int = 24
    cmap: str = "turbo"
    title_size: int = 28
    title_weight: str = "bold"
    legend_fontsize: int = 14
    line_width: int = 3
    marker_size: int = 8
    zero_transparent: bool = True  # Whether to treat zeros as transparent

    def apply_settings(self):
        """Apply settings to matplotlib."""
        plt.rcParams["axes.labelsize"] = self.label_size
        plt.rcParams["axes.labelweight"] = self.label_weight
        plt.rcParams["font.weight"] = self.font_weight
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["image.cmap"] = self.cmap
        plt.rcParams["axes.titlesize"] = self.title_size
        plt.rcParams["axes.titleweight"] = self.title_weight
        plt.rcParams["legend.fontsize"] = self.legend_fontsize
        plt.rcParams["lines.linewidth"] = self.line_width
        plt.rcParams["lines.markersize"] = self.marker_size


PLOT_CONFIG = PlotConfig()
PLOT_CONFIG.apply_settings()

DEFAULT_RUNFILENAME = "compare_offsets"
RESULT_DIR = Path("input/results")
X_PATH = RESULT_DIR / "X_ss.txt"
Z_PATH = RESULT_DIR / "Z_ss.txt"

# Grid definitions copied from result_elegant.py / plot_alpha_offset.py
SCAN_START_D, SCAN_STOP_D, SCAN_STEP_D = 10 ** -4, 2.2 * 10 ** -4, 0.2 * 10 ** -5
SCAN_START_A, SCAN_STOP_A, SCAN_STEP_A = 10 ** -5, 1.01 * 10 ** -4, 0.1 * 10 ** -5


class MachineParams:
    def __init__(self):
        # Constants
        self.e_charge = 1.602176634e-19
        self.c = 299792458                 # m/s
        self.E_0 = 629e6                  # eV
        self.U_0 = 9.1e3                  # eV
        self.T_0 = 48 / self.c            # s
        self.gam = self.E_0 / (0.511e6) + 1

        # Lattice / RF
        self.beta_x = 0.6428098                  # m
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
    return np.sqrt(params.e_x * params.beta_x) * sp.j1(delta / mu_s) # (2*np.sqrt(2)) factor retained


def z_offset(alpha, delta, params=PARAMS):
    """Longitudinal offset from 00_long_z."""
    w_s = synchrotron_frequency(alpha, params)
    return (alpha - 1 / params.gam**2) * params.c / w_s * delta


def ensure_output_dir(runfilename: str) -> Path:
    out_dir = Path("output") / runfilename
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_simulation_slice(delta_target: float):
    if not X_PATH.exists() or not Z_PATH.exists():
        raise SystemExit(f"Missing simulation data: {X_PATH} or {Z_PATH}")

    alpha_grid = np.arange(SCAN_START_A, SCAN_STOP_A, SCAN_STEP_A)  # absolute alpha values
    alpha_axis = alpha_grid / 1e-4  # normalized for plotting
    delta_grid = np.arange(SCAN_START_D, SCAN_STOP_D, SCAN_STEP_D)
    delta_norm = delta_grid / 1e-4

    X = np.loadtxt(X_PATH)
    Z = np.loadtxt(Z_PATH)

    if X.shape != (len(delta_grid), len(alpha_grid)) or Z.shape != (len(delta_grid), len(alpha_grid)):
        raise SystemExit(
            f"Unexpected X/Z shape {X.shape} / {Z.shape}; expected {(len(delta_grid), len(alpha_grid))}"
        )

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


def plot_combined(alpha_axis, sim_x, sim_z, th_x, th_z, delta_target, delta_used, out_dir: Path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax[0].plot(alpha_axis, th_x, color="tab:red", label=f"Theory (delta={delta_target:.1e})")
    ax[0].plot(alpha_axis, sim_x, color="k", linestyle="--", marker="o", markersize=4, label=f"Simulation (delta~{delta_used:.1e})")
    ax[0].set_ylabel(r"x offset ($\mu$m)")
    ax[0].legend()

    ax[1].plot(alpha_axis, th_z, color="tab:blue", label=f"Theory (delta={delta_target:.1e})")
    ax[1].plot(alpha_axis, sim_z, color="k", linestyle="--", marker="s", markersize=4, label=f"Simulation (delta~{delta_used:.1e})")
    ax[1].set_xlabel(r"momentum compaction $\alpha_c \times 10^{-4}$")
    ax[1].set_ylabel(r"z offset ($\mu$m)")
    ax[1].legend()

    style_axes(ax)

    fig.suptitle(f"Offsets vs. alpha_c | target delta={delta_target:.1e}, sim slice ~{delta_used:.1e}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = out_dir / f"combined_offsets_delta{delta_target:.0e}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined plot to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare theoretical and simulation offsets vs. alphac for a chosen delta.")
    parser.add_argument("--delta", type=float, default=2e-4, help="Energy spread (delta) to use for theory curve and nearest simulation slice.")
    parser.add_argument(
        "--runfilename",
        default=DEFAULT_RUNFILENAME,
        help="Folder name under output/ where plots will be saved.",
    )
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.runfilename)

    alpha_grid, alpha_axis, sim_x, sim_z, delta_used = load_simulation_slice(args.delta)
    th_x, th_z = theoretical_offsets_um(alpha_grid, args.delta)

    plot_combined(alpha_axis, sim_x, sim_z, th_x, th_z, args.delta, delta_used, out_dir)


if __name__ == "__main__":
    main()
