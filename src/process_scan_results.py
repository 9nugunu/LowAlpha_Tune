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

# Plot style
plt.switch_backend("Agg")
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 20
plt.rcParams["image.cmap"] = "jet"

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

args = get_parser()

# --- User Configuration ---
TARGET_SCAN_DIR = args.dir


def get_latest_scan_dir():
    """Find the specified or most recently modified directory starting with 'scan_'."""
    # Define potential search paths for scan results
    search_paths = [
        PROJECT_ROOT / "scan_elegant" / "results",
        PROJECT_ROOT / "output" / "scan_alphac_pyele",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "input" / "results" / "scan_elegant" / "results"
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


FDIR = get_latest_scan_dir()

# Initialize OUT_DIR based on the scan directory name to avoid overwriting
if FDIR:
    OUT_DIR = PROJECT_ROOT / "output" / Path(__file__).stem / FDIR.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
else:
    OUT_DIR = PROJECT_ROOT / "output" / Path(__file__).stem


def col_page(filename, column):
    temp = sdds.SDDS(0)
    temp.load(str(filename))
    col_data = temp.columnData[temp.columnName.index(column)]
    return np.array(col_data)

def amp_cal(p):
    fdir, valD, valA = p
    # os.chdir(fdir)  # Removed unsafe chdir
    file_name = f"opt_A{valA:.2e}_D{valD:.2e}_check.w2"
    fpath = fdir / file_name

    if fpath.exists():
        x = col_page(str(fpath), "x")
        # xp = col_page(str(fpath), "xp") # Use fpath (absolute) not just filename
        t = col_page(str(fpath), "dt") * c0
        # p = col_page(str(fpath), "p")

    
        freq = fftshift(fftfreq(len(x), T0)) / 10**6
        fz = fftshift(fft(t.ravel()))
        fx = fftshift(fft(x.ravel()))

        # Correct prominence/height to use absolute amplitude
        amp_x = np.abs(fx)
        h_threshold = np.max(amp_x) / 20.
        peaks, _ = find_peaks(amp_x, prominence=h_threshold, height=h_threshold, distance=3)

        cal_peak = []
        for i in peaks:
            if freq[i] > 0:
                cal_peak.append([freq[i], amp_x[i]])
        cal_peak = np.array(cal_peak)

        vxs, vzs = [], []
        if len(cal_peak) > 0:
            for k in range(len(cal_peak)):
                if 0.0 < cal_peak[k][0] < 0.1:
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

        # --- Debug Plot for Peak Verification (Sample ~2%) ---
        if (vx > 0 or vz > 0) and np.random.random() < 0.02:
            debug_dir = OUT_DIR / "debug_peaks"
            debug_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 6))
            mask = freq > 0
            plt.semilogy(freq[mask], amp_x[mask], label='X Spectrum', color='gray', alpha=0.6)
            if vx > 0:
                plt.axvline(vx, color='r', linestyle='--', label=f'vx={vx:.4f} MHz')
            if vz > 0:
                plt.axvline(vz, color='g', linestyle='--', label=f'vz={vz:.4f} MHz')
            plt.title(f"Debug Peaks: A={valA:.2e}, D={valD:.2e}")
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, which='both', ls=':', alpha=0.5)
            plt.savefig(debug_dir / f"check_A{valA:.2e}_D{valD:.2e}.png", dpi=120)
            plt.close()

        if vx == 0 or vz == 0:
            return [valA / 10 ** -4, valD / 10 ** -4, 0.0, 0.0]

            
        # vxn1 = vx-vz
        # vxp1 = vx+vz
        # bw = vz/3.
        bw = 0.001

        for i in range(len(freq)):
            if  vx-vz -bw/2 < np.abs(freq[i]) < vx-vz+bw/2 or vx+vz-bw/2 < np.abs(freq[i]) < vx+vz+bw/2 :
                fz[i] = 0
                # pass
            elif vz -bw/2 < np.abs(freq[i]) < vz + bw/2:
                fx[i] = 0
        try:
            x = col_page(str(fpath), "x")
            # xp = col_page(str(fpath), "xp") # Use fpath (absolute) not just filename
            t = col_page(str(fpath), "dt") * c0
            # p = col_page(str(fpath), "p")

        
            freq = fftshift(fftfreq(len(x), T0)) / 10**6
            fz = fftshift(fft(t.ravel()))
            fx = fftshift(fft(x.ravel()))

            # Correct prominence/height to use absolute amplitude
            amp_x = np.abs(fx)
            h_threshold = np.max(amp_x) / 20.
            peaks, _ = find_peaks(amp_x, prominence=h_threshold, height=h_threshold, distance=3)

            cal_peak = []
            for i in peaks:
                if freq[i] > 0:
                    cal_peak.append([freq[i], amp_x[i]])
            cal_peak = np.array(cal_peak)

            vxs, vzs = [], []
            if len(cal_peak) > 0:
                for k in range(len(cal_peak)):
                    if 0.0 < cal_peak[k][0] < 0.1:
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

            # --- Debug Plot for Peak Verification (Sample ~2%) ---
            if (vx > 0 or vz > 0) and np.random.random() < 0.02:
                debug_dir = OUT_DIR / "debug_peaks"
                debug_dir.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(10, 6))
                mask = freq > 0
                plt.semilogy(freq[mask], amp_x[mask], label='X Spectrum', color='gray', alpha=0.6)
                if vx > 0:
                    plt.axvline(vx, color='r', linestyle='--', label=f'vx={vx:.4f} MHz')
                if vz > 0:
                    plt.axvline(vz, color='g', linestyle='--', label=f'vz={vz:.4f} MHz')
                plt.title(f"Debug Peaks: A={valA:.2e}, D={valD:.2e}")
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid(True, which='both', ls=':', alpha=0.5)
                plt.savefig(debug_dir / f"check_A{valA:.2e}_D{valD:.2e}.png", dpi=120)
                plt.close()

            if vx == 0 or vz == 0:
                return [round(valA / 10 ** -4, 8), round(valD / 10 ** -4, 8), 0.0, 0.0]

                
            # vxn1 = vx-vz
            # vxp1 = vx+vz
            # bw = vz/3.
            bw = 0.001

            for i in range(len(freq)):
                if  vx-vz -bw/2 < np.abs(freq[i]) < vx-vz+bw/2 or vx+vz-bw/2 < np.abs(freq[i]) < vx+vz+bw/2 :
                    fz[i] = 0
                    # pass
                elif vz -bw/2 < np.abs(freq[i]) < vz + bw/2:
                    fx[i] = 0
                else:
                    fx[i] = 0
                    fz[i] = 0
            
            filtered_x = np.real(ifftshift(ifft(fx)))
            filtered_z = np.real(ifftshift(ifft(fz)))
            return [
                round(valA / 10**-4, 8), 
                round(valD / 10**-4, 8), 
                np.max(np.abs(filtered_x)) / 10**-6, 
                np.max(np.abs(filtered_z)) / 10**-6
            ]
        except Exception:
            return [round(valA / 10**-4, 8), round(valD / 10**-4, 8), 0.0, 0.0]
    else:
        # Log missing file
        print(f"Missing: {fpath}")
        return [0,0,0,0]

if __name__ == "__main__":
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

    x_txt = FDIR / "X.txt"
    z_txt = FDIR / "Z.txt"

    if x_txt.exists() and z_txt.exists():
        X = np.loadtxt(x_txt, unpack=True).T
        Z = np.loadtxt(z_txt, unpack=True).T
    else:
        new_value = []
        # Round the ranges to avoid floating point artifacts
        d_range = np.round(np.arange(scan_startD, scan_stopD, scan_stepD), 12)
        a_range = np.round(np.arange(scan_startA, scan_stopA, scan_stepA), 12)
        for i in d_range:
            for k in a_range:
                new_value.append([FDIR, i, k])

        processes = os.cpu_count() or 1
        if processes > 1:
            print("multiprocessing works")
            mp_pool = __import__("multiprocessing").Pool(processes)
            results = np.array(mp_pool.map(amp_cal, new_value))
            mp_pool.close()
            mp_pool.join()
        else:
            results = np.array([amp_cal(new_value[i]) for i in range(len(new_value))])

        results = results.reshape((-1, 4))
        Z = np.zeros((len(delD_new), len(alpha_new)))
        X = np.zeros((len(delD_new), len(alpha_new)))

        for i in range(len(alpha_new)):
            for k in range(len(delD_new)):
                # Use a tolerance check or simpler matching since we rounded grid definition
                # But to fully respect "find rounding point", we round the result values too
                target_a = alpha_new[i]
                target_d = delD_new[k]
                
                # Use np.isclose for robust float matching
                for l in range(len(results)):
                    if np.isclose(results[l][0], target_a, atol=1e-6) and \
                       np.isclose(results[l][1], target_d, atol=1e-6):
                        X[k][i] = results[l][2]
                        Z[k][i] = results[l][3]
                        break
 # Found it, break inner loop

        np.savetxt(x_txt, X)
        np.savetxt(z_txt, Z)

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
    plt.show()
