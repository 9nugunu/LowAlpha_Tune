"""
Process elegant scan results for the current alpha/delta ranges and plot contours.

Logic follows ref_result_elegant.py; only the scan ranges and file naming/location
are updated for the new sweep (files like opt_A{alpha}_D{delta}_check.w2 under results/).
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sdds
import json
import re
from scipy.fftpack import fft, fftshift, fftfreq, ifft, ifftshift
from scipy.signal import find_peaks
import pandas as pd
from src.visualization import PlotConfig

# Plot style
plt.switch_backend("Agg")
config = PlotConfig(cmap='jet', font_size=20)
config.apply_settings()

c0 = 299792458
T0 = 48 / c0

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
# OUT_DIR will be initialized after finding FDIR

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Process scan results and plot contours.")
    parser.add_argument("--dir", type=str, default=None, help="Specific scan directory name to process.")
    return parser.parse_args()

# Set from CLI args in __main__
TARGET_SCAN_DIR = None


def get_latest_scan_dir():
    """Find the specified or most recently modified directory starting with 'scan_'."""
    # Define potential search paths for scan results
    search_paths = [
        PROJECT_ROOT / "scan_elegant" / "results",
        PROJECT_ROOT / "output" / "scan_alphac_pyele",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "input" / "results" / "scan_elegant" / "results",
        PROJECT_ROOT / "Archieves",
    ]
    
    # 1. Priority: Explicitly specified directory
    if TARGET_SCAN_DIR:
        for base in search_paths:
            path = base / TARGET_SCAN_DIR
            if path.exists() and path.is_dir():
                return path
        print(f"Warning: Specified directory '{TARGET_SCAN_DIR}' not found. Searching for latest...")

    # 2. Fallback: Find latest scan session
    all_scans = []
    for base in search_paths:
        if base.exists():
            scans = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("scan_")]
            all_scans.extend(scans)
    
    if not all_scans:
        return None
    return max(all_scans, key=os.path.getmtime)


def load_metadata(fdir: Path):
    """Load scan parameters from metadata.json or fallback to directory name parsing."""
    meta_path = fdir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            return (
                meta.get("SCAN_START_A"), meta.get("SCAN_STOP_A"), meta.get("SCAN_STEP_A"),
                meta.get("SCAN_START_D"), meta.get("SCAN_STOP_D"), meta.get("SCAN_STEP_D")
            )
    
    # Fallback: Parse from directory name
    pattern = r"scan_A([\d\.eE+-]+)-([\d\.eE+-]+)_D([\d\.eE+-]+)-([\d\.eE+-]+)"
    match = re.search(pattern, fdir.name)
    if match:
        try:
            return (
                float(match.group(1)), float(match.group(2)), 1e-6, # Default step if missing
                float(match.group(3)), float(match.group(4)), 0.2e-6 # Default step if missing
            )
        except ValueError:
            pass
    return None


# Initialized in __main__
FDIR = None
OUT_DIR = None


def col_page(filename, column):
    temp = sdds.SDDS(0)
    temp.load(str(filename))
    col_data = temp.columnData[temp.columnName.index(column)]
    return np.array(col_data)

def filtering_norm(freq, data, cen_f, cutoff, ttt, vz=0.0):
    """
    Frequency filtering logic.
    ttt=0: Main peak bandpass (BP filter around cen_f)
    ttt=2: Sideband bandpass (BP filter around cen_f +/- vz)
    """
    lowband = cen_f - cutoff/2.0
    highband = cen_f + cutoff/2.0
    tmp = np.zeros_like(data) # Initialize with zeros to act like the legacy 'else: 0'
    
    for i in range(len(freq)):
        f_abs = np.abs(freq[i])
        if ttt == 2:
            # Sideband filtering: Keeps vx - vz and vx + vz
            if (lowband - vz < f_abs < highband - vz) or (lowband + vz < f_abs < highband + vz):
                tmp[i] = data[i]
        else:
            # Standard bandpass: Keeps cen_f (used for vz in longitudinal)
            if lowband < f_abs < highband:
                tmp[i] = data[i]
    return tmp

def amp_cal(p):
    fdir, valD, valA = p
    file_name = f"opt_A{valA:.2e}_D{valD:.2e}_check.w2"
    fpath = fdir / file_name

    if not fpath.exists():
        print(f"Missing: {fpath}")
        return [0.0, 0.0, 0.0, 0.0]

    try:
        x = col_page(str(fpath), "x")
        t = col_page(str(fpath), "dt") * c0
        
        freq = fftshift(fftfreq(len(x), T0)) / 10**6
        fft_x = fftshift(fft(x.ravel()))
        fft_z = fftshift(fft(t.ravel()))

        # --- Peak Detection (Initial for Tune Determination) ---
        amp_x = np.abs(fft_x)
        h_threshold = np.max(amp_x) / 20.
        peaks, _ = find_peaks(amp_x, prominence=h_threshold, height=h_threshold, distance=3)

        cal_peak = []
        for i in peaks:
            if freq[i] > 0:
                cal_peak.append([freq[i], amp_x[i]])
        cal_peak = np.array(cal_peak)

        vxs, vzs = [], []
        for k in range(len(cal_peak)):
            if 0.001 < cal_peak[k][0] < 0.1:
                vzs.append(cal_peak[k])
            elif cal_peak[k][0] > 1:
                vxs.append(cal_peak[k])
        vxs = np.array(vxs)
        vzs = np.array(vzs)

        vx, vz = 0.0, 0.0
        if len(vxs) > 0:
            vx = vxs[vxs[:, 1].argsort()][-1, 0]
        if len(vzs) > 0:
            vz = vzs[vzs[:, 1].argsort()][-1, 0]

        # --- Sideband Detection (X Spectrum) ---
        vz_sb_left = 0.0
        vz_sb_right = 0.0
        
        if vx > 0 and len(cal_peak) > 0:
            # Find closest peaks to vx
            candidates = cal_peak[cal_peak[:, 0] > 1.0] # Only look at high freq
            
            # Left Sideband
            left_candidates = candidates[candidates[:, 0] < vx]
            if len(left_candidates) > 0:
                # Get the one with highest frequency (closest to vx from left)
                sb_left_idx = np.argmax(left_candidates[:, 0])
                sb_left_freq = left_candidates[sb_left_idx, 0]
                # Check if it's within reasonable range (e.g. < 100kHz distance)
                if (vx - sb_left_freq) < 0.1:
                    vz_sb_left = vx - sb_left_freq

            # Right Sideband
            right_candidates = candidates[candidates[:, 0] > vx]
            if len(right_candidates) > 0:
                # Get the one with lowest frequency (closest to vx from right)
                sb_right_idx = np.argmin(right_candidates[:, 0])
                sb_right_freq = right_candidates[sb_right_idx, 0]
                if (sb_right_freq - vx) < 0.1:
                    vz_sb_right = sb_right_freq - vx

        # --- Debug Plot with Subplots (Zoomed vz and vx sidebands) ---
        # --- Filtering (User's Physics: X=Sidebands, Z=Main) ---
        bw = 0.001 # User preference: Bandwidth used for filtering

        # --- Debug Plot with Subplots (Zoomed vz and vx sidebands) ---
        if (vx > 0 or vz > 0) and np.random.random() < 0.02:
            debug_dir = OUT_DIR / "debug_peaks"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            # bw used for visualization spans
            
            # Subplot 1: Z-Tune (Synchrotron) - Focused on 0-0.15 MHz
            mask1 = (freq >= 0) & (freq < 0.1)
            ax1.semilogy(freq[mask1], np.abs(fft_z[mask1]), color='royalblue', alpha=0.7, label='Z Spectrum')
            if vz > 0:
                ax1.axvline(vz, color='g', linestyle='--', label=f'vz={vz:.4f} MHz')
                ax1.axvspan(vz - bw/2, vz + bw/2, color='green', alpha=0.2, label='vz filter')
            ax1.set_title(f"Z-Tune Zoom (Synchrotron)")
            ax1.set_xlabel("Frequency [MHz]")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, which='both', ls=':', alpha=0.5)
            ax1.legend(fontsize=9)

            # Subplot 2: X-Tune & Sidebands - Zoomed around vx
            span = max(vz * 2.5, 0.05) # Dynamic zoom based on vz, at least 50kHz
            mask2 = (freq > vx - span) & (freq < vx + span)
            ax2.semilogy(freq[mask2], np.abs(fft_x[mask2]), color='indianred', alpha=0.7, label='X Spectrum')
            if vx > 0:
                ax2.axvline(vx, color='blue', linestyle='--', alpha=0.4, label=f'vx={vx:.4f} MHz')
                if vz > 0:
                    # Sidebands vx +/- vz
                    ax2.axvline(vx - vz, color='orange', linestyle=':', label=f'vx-vz ({vx-vz:.4f})')
                    ax2.axvline(vx + vz, color='orange', linestyle=':', label=f'vx+vz ({vx+vz:.4f})')
                    ax2.axvspan(vx - vz - bw/2, vx - vz + bw/2, color='orange', alpha=0.1)
                    ax2.axvspan(vx + vz - bw/2, vx + vz + bw/2, color='orange', alpha=0.1)
            ax2.set_title(f"X-Tune & Sidebands Zoom")
            ax2.set_xlabel("Frequency [MHz]")
            ax2.grid(True, which='both', ls=':', alpha=0.5)
            ax2.legend(fontsize=8, loc='upper right')

            plt.suptitle(f"Debug Analysis (A={valA:.2e}, D={valD:.2e}, bw={bw:.1e})", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(debug_dir / f"check_A{valA:.2e}_D{valD:.2e}.png", dpi=250)
            plt.close()

        if vx == 0 or vz == 0:
            return [round(valA / 10**-4, 8), round(valD / 10**-4, 8), 0.0, 0.0, 0.0, vx, vz, 0.0, 0.0, 0.0, 0.0, 0.0]

        # --- Filtering applied using defining 'bw' above ---
        
        # fx_main: Extract ONLY the main betatron peak (ttt=0) - Carrier amplitude
        fx_main = filtering_norm(freq, fft_x, vx, bw, 0)
        
        # fx_side: Extract ONLY the sidebands (ttt=2) - Modulation amplitude
        fx_side = filtering_norm(freq, fft_x, vx, bw, 2, vz=vz)
        
        # fz_filt: Extract ONLY the main synchrotron peak (ttt=0)
        fz_filt = filtering_norm(freq, fft_z, vz, bw, 0)

        # fx_comb: Combine Main + Sidebands (Carrier + Modulation)
        fx_comb = fx_main + fx_side
        
        filtered_x_main = np.real(ifftshift(ifft(fx_main)))
        filtered_x_side = np.real(ifftshift(ifft(fx_side)))
        filtered_x_comb = np.real(ifftshift(ifft(fx_comb)))
        filtered_z = np.real(ifftshift(ifft(fz_filt)))

        # --- Noise Floor Calculation (Median of Spectrum) ---
        noise_x = np.median(np.abs(fft_x))
        noise_z = np.median(np.abs(fft_z))
        
        # X amplitudes
        X_main_amp = np.max(np.abs(filtered_x_main)) / 10**-6
        X_side_amp = np.max(np.abs(filtered_x_side)) / 10**-6
        X_comb_amp = np.max(np.abs(filtered_x_comb)) / 10**-6  # Combined Amplitude

        return [
            round(valA / 10**-4, 8),   # 0: alpha
            round(valD / 10**-4, 8),   # 1: delta
            X_main_amp,                 # 2: X main peak
            X_side_amp,                 # 3: X sideband
            np.max(np.abs(filtered_z)) / 10**-6,  # 4: Z amplitude
            vx,                         # 5: vx
            vz,                         # 6: vz
            vz_sb_left,                 # 7: vz_sb_l
            vz_sb_right,                # 8: vz_sb_r
            noise_x,                    # 9: noise X
            noise_z,                    # 10: noise Z
            X_side_amp / X_main_amp if X_main_amp > 0 else 0.0, # 11: mod_idx
            X_comb_amp                  # 12: X combined (Main + Side)
        ]
    except Exception as e:
        # print(f"Error processing {fpath.name}: {e}")
        return [round(valA / 10**-4, 8), round(valD / 10**-4, 8), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

if __name__ == "__main__":
    args = get_parser()
    TARGET_SCAN_DIR = args.dir
    FDIR = get_latest_scan_dir()
    if FDIR:
        OUT_DIR = PROJECT_ROOT / "output" / Path(__file__).stem / FDIR.name
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUT_DIR = PROJECT_ROOT / "output" / Path(__file__).stem

    if FDIR:
        print(f"Processing latest scan results from: {FDIR}")
    else:
        print("Warning: No scan results found. Please check your search paths.")
        # We can't proceed without data
        import sys
        sys.exit(1)

    # Attempt to load scan ranges automatically
    params = load_metadata(FDIR)
    if params:
        scan_startA, scan_stopA, scan_stepA, scan_startD, scan_stopD, scan_stepD = params
        print(f"Ranges Auto-detected:")
        print(f"  A: {scan_startA:.2e} to {scan_stopA:.2e} (step: {scan_stepA:.2e})")
        print(f"  D: {scan_startD:.2e} to {scan_stopD:.2e} (step: {scan_stepD:.2e})")
    else:
        # Final fallback to manual values if everything else fails
        scan_startA, scan_stopA, scan_stepA = 10**-5, 1.01*10**-4, 0.1*10**-5
        scan_startD, scan_stopD, scan_stepD = 1.4*10**-5, 2.6*10**-5, 0.2*10**-6
        print("Using hardcoded fallback ranges (no metadata found).")

    display_scale = 10**-4
    alpha_new = np.round(np.arange(scan_startA, scan_stopA, scan_stepA) / display_scale, 6)
    delD_new = np.round(np.arange(scan_startD, scan_stopD, scan_stepD) / display_scale, 6)
    plotX, plotY = np.meshgrid(alpha_new, delD_new)

    x_csv = FDIR / "X.csv"
    z_csv = FDIR / "Z.csv"

    if x_csv.exists() and z_csv.exists():
        X = pd.read_csv(x_csv, index_col=0).values
        Z = pd.read_csv(z_csv, index_col=0).values
    else:
        new_value = []
        # Round the ranges to avoid floating point artifacts
        d_range = np.round(np.arange(scan_startD, scan_stopD, scan_stepD), 12)
        a_range = np.round(np.arange(scan_startA, scan_stopA, scan_stepA), 12)
        for i in d_range:
            for k in a_range:
                new_value.append([FDIR, i, k])

        processes = os.cpu_count() or 1
        # Windows limit for WaitForMultipleObjects is 64 handles. 
        # Capping to 60 for safety on high-core-count machines.
        if os.name == 'nt' and processes > 60:
            processes = 60
            
        if processes > 1:
            print(f"multiprocessing works (using {processes} cores)")
            mp_pool = __import__("multiprocessing").Pool(processes)
            
            total_tasks = len(new_value)
            completed = 0
            results_list = []
            
            for res in mp_pool.imap(amp_cal, new_value):
                results_list.append(res)
                completed += 1
                percent = (completed / total_tasks) * 100
                import sys
                sys.stdout.write(f"\rProcessing Progress: [{completed}/{total_tasks}] {percent:.1f}% ")
                sys.stdout.flush()
            
            results = np.array(results_list)
            mp_pool.close()
            mp_pool.join()
            print("\nProcessing complete.")
        else:
            results = np.array([amp_cal(new_value[i]) for i in range(len(new_value))])

        # Results shape is now (N, 13): 
        # [0:alpha, 1:delta, 2:X_main, 3:X_side, 4:Z_amp, 5:vx, 6:vz, 7:vz_sb_l, 8:vz_sb_r, 9:nx, 10:nz, 11:mod_idx, 12: X_comb]
        results = results.reshape((-1, 13))

        # 1. Save Amplitudes CSV (including main/sideband separation and modulation index)
        amp_df = pd.DataFrame(
            results[:, [0, 1, 2, 3, 12, 4, 9, 10, 11]], 
            columns=['alpha_c_1e4', 'delta_1e4', 'X_main_um', 'X_side_um', 'X_comb_um', 'Z_amp_um', 'Noise_X_raw', 'Noise_Z_raw', 'Mod_Index']
        )
        amp_csv = FDIR / "scan_amplitudes.csv"
        amp_df.to_csv(amp_csv, index=False)
        print(f"Amplitudes saved to: {amp_csv}")

        # 2. Save Tunes CSV
        # Columns: alpha, delta, vx, vz, vz_sb_minus, vz_sb_plus
        tune_data = results[:, [0, 1, 5, 6, 7, 8]]
        tune_df = pd.DataFrame(tune_data, columns=['alpha_c_1e4', 'delta_1e4', 'vx_MHz', 'vz_from_Z_MHz', 'vz_sb_minus_MHz', 'vz_sb_plus_MHz'])
        tune_csv = FDIR / "scan_tunes.csv"
        tune_df.to_csv(tune_csv, index=False)
        print(f"Tunes saved to: {tune_csv}")

        Z = np.zeros((len(delD_new), len(alpha_new)))
        X = np.zeros((len(delD_new), len(alpha_new)))       # This will store X_SIDEBAND (raw)
        X_side = np.zeros((len(delD_new), len(alpha_new)))  # Store sideband separately (redundant but safe)
        X_main = np.zeros((len(delD_new), len(alpha_new)))

        # 1. Create a fast lookup dictionary (O(N))
        result_map = {}
        for res in results:
            key = (round(res[0], 6), round(res[1], 6))
            # Store: (X_main, X_side, X_comb, Z_amp)
            result_map[key] = (res[2], res[3], res[12], res[4])

        # 2. Fill the grid using the lookup table
        for i, a in enumerate(alpha_new):
            for k, d in enumerate(delD_new):
                match = result_map.get((round(a, 6), round(d, 6)))
                if match:
                    X_main[k][i] = match[0]  # Main X peak
                    X[k][i]      = match[1]  # <--- REVERT: Save SIDEBAND amp to X (raw data)
                    Z[k][i]      = match[3]  # Z amplitude

        # Save as Pandas DataFrame for labeled CSV
        df_x = pd.DataFrame(X, index=delD_new, columns=alpha_new)
        df_z = pd.DataFrame(Z, index=delD_new, columns=alpha_new)
        df_x_main = pd.DataFrame(X_main, index=delD_new, columns=alpha_new)
        df_x_side = pd.DataFrame(X, index=delD_new, columns=alpha_new) # Same as X
        
        # X.csv now contains the SIDEBAND amplitude (raw)
        df_x.to_csv(x_csv)
        df_z.to_csv(z_csv)
        
        # Save X_main and X_side separately
        x_main_csv = FDIR / "X_main.csv"
        x_side_csv = FDIR / "X_sidebands.csv"
        df_x_main.to_csv(x_main_csv)
        df_x_side.to_csv(x_side_csv)
        print(f"X_main (carrier) saved to: {x_main_csv}")
        print(f"X_side (sideband) saved to: {x_side_csv}")

    # Plot Z contour
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    cp = ax.contourf(plotX, plotY, Z, levels=100)
    cbar = fig.colorbar(cp)
    cbar.set_label(r"offset ($\mu$m)")
    plt.xlabel(r"momentum compaction factor $\times 10^{-4}$")
    plt.ylabel(r"relative energy spread $\times 10^{-4}$")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Z_offset_contour.png", dpi=300)

    # Plot X contour
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    cp = ax.contourf(plotX, plotY, X, levels=100)
    cbar = fig.colorbar(cp)
    cbar.set_label(r"offset ($\mu$m)")
    plt.xlabel(r"momentum compaction factor $\times 10^{-4}$")
    plt.ylabel(r"relative energy spread $\times 10^{-4}$")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "X_offset_contour.png", dpi=300)

    # Side-by-side with unified color scale
    vmin = min(X.min(), Z.min())
    vmax = max(X.max(), Z.max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cf1 = axes[0].contourf(plotX, plotY, Z, levels=100, vmin=vmin, vmax=vmax)
    axes[0].set_title("Z offset")
    axes[0].set_xlabel(r"$\alpha_c \times 10^{-4}$")
    axes[0].set_ylabel(r"$\delta \times 10^{-4}$")
    cf2 = axes[1].contourf(plotX, plotY, X, levels=100, vmin=vmin, vmax=vmax)
    axes[1].set_title("X offset")
    axes[1].set_xlabel(r"$\alpha_c \times 10^{-4}$")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cbar = fig.colorbar(cf1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label(r"offset ($\mu$m)")
    fig.savefig(OUT_DIR / "XZ_offset_contour_side_by_side.png", dpi=300)

    # --- New: Plot Difference (X - Z) ---
    diff = X - Z
    max_abs = max(abs(diff.min()), abs(diff.max())) # For symmetric color scale
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    # 'seismic' provides very high contrast between blue (negative) and red (positive)
    cp = ax.contourf(plotX, plotY, diff, levels=100, cmap='seismic', vmin=-max_abs, vmax=max_abs)

    # --- Highlight zero difference (X = Z) ---
    zero_contour = ax.contour(plotX, plotY, diff, levels=[0.0], colors='magenta', linewidths=2.5)
    
    # Place labels at specific coordinates to widen the interval
    # We pick points where diff is near 0
    zi = np.where(np.abs(diff) < np.nanpercentile(np.abs(diff), 2))
    if len(zi[0]) > 20:
        # Pick two points (roughly 1/4 and 3/4 along the detected indices)
        p1_idx = len(zi[0]) // 14
        p2_idx = (8 * len(zi[0])) // 10
        label_locs = [
            (plotX[zi[0][p1_idx], zi[1][p1_idx]], plotY[zi[0][p1_idx], zi[1][p1_idx]]),
            (plotX[zi[0][p2_idx], zi[1][p2_idx]], plotY[zi[0][p2_idx], zi[1][p2_idx]])
        ]
        texts = ax.clabel(zero_contour, inline=True, fontsize=18, fmt={0.0: '0'}, manual=label_locs)
        for t in texts:
            t.set_rotation(0)
    else:
        texts = ax.clabel(zero_contour, inline=True, fontsize=18, fmt={0.0: '0'})
        for t in texts:
            t.set_rotation(0)
    
    cbar = fig.colorbar(cp)
    cbar.set_label(r"$\Delta x-\Delta z$ ($\mu$m)", rotation=270, labelpad=20)
    ax.set_title("Comparison of X and Z amplitudes")
    ax.set_xlabel(r"$\alpha_c \times 10^{-4}$")
    ax.set_ylabel(r"$\delta \times 10^{-4}$")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "XZ_diff_contour.png", dpi=300)
    print(f"Difference map saved to: {OUT_DIR / 'XZ_diff_contour.png'}")

    # plt.show()
