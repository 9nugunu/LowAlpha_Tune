from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.visualization import PlotConfig, apply_axis_style, apply_colorbar_style
from src.physics import x_offset, z_offset

logger = logging.getLogger(__name__)


PLOT_CONFIG = PlotConfig()
PLOT_CONFIG.apply_settings()


DEFAULT_RUNFILENAME = Path(__file__).stem
THRESHOLD_UM = 2.0


def style_axes(axes):
    """Apply shared scaled styling to axes (works for single axis or array)."""
    for ax in np.ravel(np.atleast_1d(axes)):
        figsize = tuple(ax.figure.get_size_inches())
        apply_axis_style(ax, PLOT_CONFIG, figsize)


def style_colorbar(cbar):
    """Apply shared scaled styling to colorbars based on the owning figure size."""
    figsize = tuple(cbar.ax.figure.get_size_inches())
    apply_colorbar_style(cbar, PLOT_CONFIG, figsize, cbar.ax.yaxis.label.get_text())


# --- Plot helpers ---
def plot_contour(ax, alpha_vals, delta_vals, offset_fn, title, vmin=None, vmax=None):
    A, D = np.meshgrid(alpha_vals, delta_vals)
    Z = offset_fn(A, D) / 1e-6  # convert to micrometers

    levels = 100
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 100)

    contour = ax.contourf(A / 1e-4, D / 1e-4, Z, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax)
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

def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    """Helper to save figures following the convention."""
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved: %s", out_path)

def main(runfilename: str = DEFAULT_RUNFILENAME):
    # Ranges
    alpha_range_contour = np.linspace(0.1e-4, 1e-4, 200)
    delta_range_contour = np.linspace(1e-4, 2.4e-4, 200)
    alpha_range_line = np.linspace(alpha_range_contour.min(), alpha_range_contour.max(), 200)
    delta_line_values = [1e-5]

    out_dir = ensure_output_dir(runfilename)

    # Calculate shared limits for x and z contours
    A_grid, D_grid = np.meshgrid(alpha_range_contour, delta_range_contour)
    val_z = z_offset(A_grid, D_grid) / 1e-6
    val_x = x_offset(A_grid, D_grid) / 1e-6
    vmin_shared = min(val_z.min(), val_x.min())
    vmax_shared = max(val_z.max(), val_x.max())

    # --- Contours for z ---
    fig_z_contour, ax_z_contour = plt.subplots(figsize=(8, 6), constrained_layout=True)
    c_z = plot_contour(ax_z_contour, alpha_range_contour, delta_range_contour, z_offset, 'z-offset contour', 
                       vmin=vmin_shared, vmax=vmax_shared)
    cbar_z = fig_z_contour.colorbar(c_z, ax=ax_z_contour)
    cbar_z.set_label("z offset [um]")
    style_colorbar(cbar_z)
    style_axes(ax_z_contour)

    # --- Contours for x ---
    fig_x_contour, ax_x_contour = plt.subplots(figsize=(8, 6), constrained_layout=True)
    c_x = plot_contour(ax_x_contour, alpha_range_contour, delta_range_contour, x_offset, 'x-offset contour', 
                       vmin=vmin_shared, vmax=vmax_shared)
    cbar_x = fig_x_contour.colorbar(c_x, ax=ax_x_contour)
    cbar_x.set_label("x offset [um]")
    style_colorbar(cbar_x)
    style_axes(ax_x_contour)

    # --- New: Plot Difference (X - Z) ---
    diff = val_x - val_z
    max_abs = max(abs(diff.min()), abs(diff.max()))
    
    fig_diff, ax_diff = plt.subplots(figsize=(10, 8), constrained_layout=True)
    # Use 'seismic' for symmetric divergence from 0
    cp_diff = ax_diff.contourf(A_grid / 1e-4, D_grid / 1e-4, diff, levels=100, cmap='seismic', vmin=-max_abs, vmax=max_abs)
    
    # Highlight zero difference (X = Z)
    zero_contour = ax_diff.contour(A_grid / 1e-4, D_grid / 1e-4, diff, levels=[0.0], colors='magenta', linewidths=2.5)
    
    # Label placement optimization
    zi = np.where(np.abs(diff) < np.nanpercentile(np.abs(diff), 2))
    if len(zi[0]) > 20:
        p1_idx = len(zi[0]) // 14
        p2_idx = (8 * len(zi[0])) // 10
        label_locs = [
            (A_grid[zi[0][p1_idx], zi[1][p1_idx]] / 1e-4, D_grid[zi[0][p1_idx], zi[1][p1_idx]] / 1e-4),
            (A_grid[zi[0][p2_idx], zi[1][p2_idx]] / 1e-4, D_grid[zi[0][p2_idx], zi[1][p2_idx]] / 1e-4)
        ]
        texts = ax_diff.clabel(zero_contour, inline=True, fontsize=18, fmt={0.0: '0'}, manual=label_locs)
        for t in texts:
            t.set_rotation(0)
    else:
        texts = ax_diff.clabel(zero_contour, inline=True, fontsize=18, fmt={0.0: '0'})
        for t in texts:
            t.set_rotation(0)

    cbar_diff = fig_diff.colorbar(cp_diff, ax=ax_diff)
    apply_colorbar_style(cbar_diff, PLOT_CONFIG, tuple(fig_diff.get_size_inches()), r"$\Delta x-\Delta z$ ($\mu$m)")
    ax_diff.set_title("Theoretical Comparison: X and Z amplitudes")
    ax_diff.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax_diff.set_ylabel(r"$\delta \times 10^{-4}$")
    style_axes(ax_diff)

    # --- Fixed-delta lines: z ---
    fig_z_line, ax_z_line = plt.subplots(figsize=(8, 6), constrained_layout=True)
    plot_fixed_delta_lines(ax_z_line, alpha_range_line, delta_line_values, z_offset, r'$\Delta z$ vs. $\alpha_c$')
    style_axes(ax_z_line)

    # --- Fixed-delta lines: x ---
    fig_x_line, ax_x_line = plt.subplots(figsize=(8, 6), constrained_layout=True)
    plot_fixed_delta_lines(ax_x_line, alpha_range_line, delta_line_values, x_offset, r'$\Delta x$ vs. $\alpha_c$')
    style_axes(ax_x_line)

    # --- Overlay figure: z and x on the same axes ---
    fig_overlay, ax_overlay = plt.subplots(figsize=(8, 6), constrained_layout=True)
    delta_val = delta_line_values[0]
    offsets_z = z_offset(alpha_range_line, delta_val) / 1e-6
    offsets_x = x_offset(alpha_range_line, delta_val) / 1e-6
    ax_overlay.plot(alpha_range_line / 1e-4, offsets_z, color='tab:blue', label=r'$\Delta z$')
    ax_overlay.plot(alpha_range_line / 1e-4, offsets_x, color='tab:red', label=r'$\Delta x$')

    ax_overlay.axhline(y=THRESHOLD_UM, color='k', linestyle='--', linewidth=1.5, label=r'2 $\mu$m threshold')
    ax_overlay.set_xlabel(r'$\alpha$ x 10$^{-4}$')
    ax_overlay.set_ylabel(r'offset [um]')
    ax_overlay.set_title(r'$\Delta x$ and $\Delta z$ vs. $\alpha$')
    ax_overlay.grid(True, linestyle=':')
    ax_overlay.legend()
    style_axes(ax_overlay)

    # Save figures under output/runfilename/
    save_figure(fig_z_contour, out_dir / "offset_z_contour.png")
    save_figure(fig_x_contour, out_dir / "offset_x_contour.png")
    save_figure(fig_z_line, out_dir / "offset_z_vs_alpha.png")
    save_figure(fig_x_line, out_dir / "offset_x_vs_alpha.png")
    save_figure(fig_overlay, out_dir / "offset_vs_alpha_overlay.png")
    save_figure(fig_diff, out_dir / "XZ_diff_contour.png")

    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Plot and save offset contours and fixed-delta lines.")
    parser.add_argument(
        "--runfilename",
        default=DEFAULT_RUNFILENAME,
        help="Folder name under output/ where plots will be saved.",
    )
    args = parser.parse_args()
    main(runfilename=args.runfilename)
