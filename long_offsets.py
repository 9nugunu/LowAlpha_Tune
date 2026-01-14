from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


@dataclass
class PlotConfig:
    """플롯 스타일 설정을 위한 데이터 클래스"""
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
    zero_transparent: bool = True  # 0값 투명 처리 여부

    def apply_settings(self):
        """matplotlib에 설정 적용"""
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


DEFAULT_RUNFILENAME = Path(__file__).stem


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
        self.beta_x = 10                  # m
        self.V_rf = 0.5e6                 # V
        self.f_rf = 500e6                 # Hz
        self.w_rf = 2 * np.pi * self.f_rf # rad/s
        self.e_x = 190e-9                 # m * rad

        # Derived
        self.phi_s = np.pi - np.arcsin(self.U_0 / self.V_rf)

PARAMS = MachineParams()
THRESHOLD_UM = 2.0


def style_axes(axes):
    """Apply minor ticks and grids to axes (works for single axis or array)."""
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-")
        ax.grid(which="minor", linestyle=":", alpha=0.5)


# --- Physics helpers ---
def synchrotron_frequency(alpha, params=PARAMS):
    """Synchrotron angular frequency w_s(alpha)."""
    return np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)


def x_offset(alpha, delta, params=PARAMS):
    """Transverse offset from 00_long_x (uses Bessel J1)."""
    w_s = synchrotron_frequency(alpha, params)
    mu_s = w_s * params.T_0 / (2 * np.pi)
    return np.sqrt(params.e_x * params.beta_x) * sp.j1(delta / mu_s)


def z_offset(alpha, delta, params=PARAMS):
    """Longitudinal offset from 00_long_z."""
    w_s = synchrotron_frequency(alpha, params)
    return (alpha - 1 / params.gam**2) * params.c / w_s * delta


# --- Plot helpers ---
def plot_contour(ax, alpha_vals, delta_vals, offset_fn, title):
    A, D = np.meshgrid(alpha_vals, delta_vals)
    Z = offset_fn(A, D) / 1e-6  # convert to micrometers

    contour = ax.contourf(A / 1e-4, D / 1e-4, Z, 100, cmap='turbo')
    ax.set_xlabel(r'$\alpha$ x 10$^{-4}$')
    ax.set_ylabel(r'relative energy spread x 10$^{-4}$')
    ax.set_title(title)
    return contour


def plot_fixed_delta_lines(ax, alpha_vals, delta_list, offset_fn, title, show_baseline=True):
    offset_data = []
    max_offset_um = 0.0
    for d in delta_list:
        offsets_um = offset_fn(alpha_vals, d) / 1e-6
        offset_data.append((d, offsets_um))
        max_offset_um = max(max_offset_um, offsets_um.max())

    # Colored regions for SNR interpretation (draw first)
    upper_ylim = max(max_offset_um * 1.05, THRESHOLD_UM * 1.2)
    ax.axhspan(0, THRESHOLD_UM, color='red', alpha=0.12)
    ax.axhspan(THRESHOLD_UM, upper_ylim, color='green', alpha=0.10)

    if show_baseline:
        ax.axhline(y=THRESHOLD_UM, color='k', linestyle='--', linewidth=1.5, label=r'2 $\mu$m threshold')

    # Now draw the curves on top of the colored regions
    for d, offsets_um in offset_data:
        low_mask = np.ma.masked_where(offsets_um >= THRESHOLD_UM, offsets_um)
        high_mask = np.ma.masked_where(offsets_um < THRESHOLD_UM, offsets_um)
        high_all_masked = np.ma.getmaskarray(high_mask).all()
        low_all_masked = np.ma.getmaskarray(low_mask).all()
        delta_label = rf'$\delta = {d:.0e}$'

        if not high_all_masked:
            ax.plot(alpha_vals / 1e-4, high_mask, color='green', marker='o', markersize=3, label=delta_label)
        if not low_all_masked:
            ax.plot(alpha_vals / 1e-4, low_mask, color='red', marker='o', markersize=3, label=delta_label if high_all_masked else None)

    x_mid = (alpha_vals.min() + alpha_vals.max()) / (2 * 1e-4)  # scaled axis units
    # ax.text(x_mid, THRESHOLD_UM * 0.5, 'Very low SNR', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    # ax.text(x_mid, (THRESHOLD_UM + upper_ylim) * 0.5, 'Measurable region', ha='center', va='center', color='green', fontsize=12, fontweight='bold')

    ax.set_xlabel(r'$\alpha\times$10$^{-4}$')
    ax.set_ylabel(r'offset [um]')
    ax.set_title(title)
    ax.set_ylim(0, upper_ylim)
    ax.grid(True, linestyle=':')
    ax.legend()


def ensure_output_dir(runfilename: str) -> Path:
    """Create (if needed) and return output directory under output/runfilename/."""
    out_dir = Path("output") / runfilename
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main(runfilename: str = DEFAULT_RUNFILENAME):
    # Ranges
    alpha_range_contour = np.linspace(0.1e-4, 1e-4, 200)
    delta_range_contour = np.linspace(1e-4, 2.2e-4, 200)
    alpha_range_line = np.linspace(alpha_range_contour.min(), alpha_range_contour.max(), 200)
    delta_line_values = [1e-5]

    out_dir = ensure_output_dir(runfilename)

    # --- Contours for x and z ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_z_contour, ax_x_contour = axes1  # z on the left (subplot 1,1), x on the right (1,2)
    c_z = plot_contour(ax_z_contour, alpha_range_contour, delta_range_contour, z_offset, 'z-offset contour')
    fig1.colorbar(c_z, ax=ax_z_contour).set_label("z offset [um]")

    c_x = plot_contour(ax_x_contour, alpha_range_contour, delta_range_contour, x_offset, 'x-offset contour')
    fig1.colorbar(c_x, ax=ax_x_contour).set_label("x offset [um]")
    style_axes(axes1)

    # --- Fixed-delta lines: z on the left, x on the right ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_z_line, ax_x_line = axes2  # z on the left (subplot 1,1), x on the right (1,2)
    plot_fixed_delta_lines(ax_z_line, alpha_range_line, delta_line_values, z_offset, r'$\Delta z$ vs. $\alpha_c$')
    plot_fixed_delta_lines(ax_x_line, alpha_range_line, delta_line_values, x_offset, r'$\Delta x$ vs. $\alpha_c$')
    style_axes(axes2)

    # --- Overlay figure: z and x on the same axes ---
    fig3, ax3 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    delta_val = delta_line_values[0]
    offsets_z = z_offset(alpha_range_line, delta_val) / 1e-6
    offsets_x = x_offset(alpha_range_line, delta_val) / 1e-6
    ax3.plot(alpha_range_line / 1e-4, offsets_z, color='tab:blue', label=r'$\Delta z$')
    ax3.plot(alpha_range_line / 1e-4, offsets_x, color='tab:red', label=r'$\Delta x$')

    ax3.axhline(y=THRESHOLD_UM, color='k', linestyle='--', linewidth=1.5, label=r'2 $\mu$m threshold')
    ax3.set_xlabel(r'$\alpha$ x 10$^{-4}$')
    ax3.set_ylabel(r'offset [um]')
    ax3.set_title(r'$\Delta x$ and $\Delta z$ vs. $\alpha$')
    ax3.grid(True, linestyle=':')
    ax3.legend()
    style_axes(ax3)

    # Save figures under output/runfilename/
    contour_path = out_dir / "offset_contours.png"
    lines_path = out_dir / "offset_vs_alpha.png"
    overlay_path = out_dir / "offset_vs_alpha_overlay.png"
    fig1.savefig(contour_path, dpi=300)
    fig2.savefig(lines_path, dpi=300)
    fig3.savefig(overlay_path, dpi=300)
    print(f"Saved contour plots to {contour_path}")
    print(f"Saved fixed-delta plots to {lines_path}")
    print(f"Saved overlay plot to {overlay_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and save offset contours and fixed-delta lines.")
    parser.add_argument(
        "--runfilename",
        default=DEFAULT_RUNFILENAME,
        help="Folder name under output/ where plots will be saved.",
    )
    args = parser.parse_args()
    main(runfilename=args.runfilename)
