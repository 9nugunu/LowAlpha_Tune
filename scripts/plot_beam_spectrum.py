import sdds
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq
from pathlib import Path
import os
import argparse
from scipy.signal import find_peaks
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.visualization import PlotConfig, escape_latex

PlotConfig().apply_settings()

# --- Constants from MLS.py ---
c0 = 299792458
T0 = 48 / c0  # Revolution period for MLS (Circumference 48m)
bw = 0.001

# --- Paths ---
BASE_DIR = Path(__file__).parent
RESULTS_BASE = BASE_DIR / "output" / "scan_alphac_pyele"
BASELINE_PATH = BASE_DIR / "input" / "MLS_tune_elegant" / "run1.w2"

def get_latest_scan_dir():
    """Find the most recently modified directory starting with 'scan_'."""
    if not RESULTS_BASE.exists():
        return None
    scans = [d for d in RESULTS_BASE.iterdir() if d.is_dir() and d.name.startswith("scan_")]
    if not scans:
        return None
    return max(scans, key=os.path.getmtime)

def get_column(fpath, col_name):
    """Load a specific column from an SDDS file."""
    ds = sdds.SDDS(0)
    ds.load(str(fpath))
    try:
        col_index = ds.columnName.index(col_name)
        return np.array(ds.columnData[col_index]).ravel()
    except ValueError:
        print(f"Warning: Column '{col_name}' not found in {fpath.name}")
        return None

def filtering_norm(freq, data, cen_f, cutoff, ttt, vz=0.0):
    """Frequency filtering logic from MLS.py."""
    lowband = cen_f - cutoff/2.0
    highband = cen_f + cutoff/2.0
    tmp = data.copy()
    
    for i in range(len(freq)):
        f_abs = np.abs(freq[i])
        if ttt == 2:
            if (lowband - vz < f_abs < highband - vz) or (lowband + vz < f_abs < highband + vz):
                tmp[i] = data[i]
            else:
                tmp[i] = 0
        else:
            if lowband < f_abs < highband:
                tmp[i] = data[i]
            else:
                tmp[i] = 0
    return tmp

def plot_spectrum(fpath):
    """Compute FFT and plot Amplitude Spectrum (MLS.py position_retrive style)."""
    if not fpath.exists():
        print(f"Error: File not found -> {fpath}")
        return

    print(f"Processing spectrum for: {fpath}")
    
    x = get_column(fpath, "x")
    dt = get_column(fpath, "dt")
    
    if x is None or dt is None:
        return

    n = len(x)
    freq = fftshift(fftfreq(n, T0)) * 1e-6  # Frequency in MHz
    
    # --- Horizontal (X) Analysis ---
    x_fft_main = fftshift(fft(x.ravel()))
    amp_x = np.abs(x_fft_main)

    # --- Peak Detection (Same as process_scan_results.py) ---
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
            elif cal_peak[k][0] > 1.0:
                vxs.append(cal_peak[k])
    
    vxs = np.array(vxs)
    vzs = np.array(vzs)
    
    vx_det, vz_det = 0.0, 0.0
    if len(vxs) > 0:
        vx_det = vxs[vxs[:, 1].argsort()][-1, 0]
    if len(vzs) > 0:
        vz_det = vzs[vzs[:, 1].argsort()][-1, 0]
    
    print(f"Detected vx: {vx_det:.6f} MHz, vz: {vz_det:.6f} MHz")

    # Replicate filtering from process_scan_results.py
    # For X: Take sidebands of vx (ttt=2)
    x_filt_side = filtering_norm(freq, x_fft_main, vx_det, bw, 2, vz=vz_det)

    # --- Longitudinal (Z) Analysis ---
    z_dist = dt.ravel() * c0
    z_fft_main = fftshift(fft(z_dist))
    # For Z: Take main sync peak (ttt=0)
    z_filt_main = filtering_norm(freq, z_fft_main, vz_det, bw, 0)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Positive mask for plotting
    mask = freq > 0
    freq_p = freq[mask]
    
    # X Plot (overlay original and filtered)
    axes[0].semilogy(freq_p, np.abs(x_fft_main[mask]), label='X Original', color='gray', alpha=0.3)
    axes[0].semilogy(freq_p, np.abs(x_filt_side[mask]), label='X Sidebands (Filtered)', color='red', linewidth=1.5)
    
    if vx_det > 0:
        axes[0].axvline(vx_det, color='blue', linestyle='--', alpha=0.5, label=f'vx={vx_det:.4f}')
        if vz_det > 0:
            # Mark sidebands
            axes[0].axvline(vx_det - vz_det, color='orange', linestyle=':', label='vx-vz')
            axes[0].axvline(vx_det + vz_det, color='orange', linestyle=':', label='vx+vz')
            axes[0].axvspan(vx_det - vz_det - bw/2, vx_det - vz_det + bw/2, color='orange', alpha=0.2)
            axes[0].axvspan(vx_det + vz_det - bw/2, vx_det + vz_det + bw/2, color='orange', alpha=0.2)

    axes[0].set_title(f"Horizontal Spectrum (X) - {escape_latex(fpath.name)}")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].set_xlim(vx_det - 0.05, vx_det + 0.05) if vx_det > 0 else axes[0].set_xlim(0, 2)

    # Z Plot
    axes[1].semilogy(freq_p, np.abs(z_fft_main[mask]), label='Z Original', color='gray', alpha=0.3)
    axes[1].semilogy(freq_p, np.abs(z_filt_main[mask]), label='Z Main Peak (Filtered)', color='green', linewidth=1.5)
    
    if vz_det > 0:
        axes[1].axvline(vz_det, color='green', linestyle='--', label=f'vz={vz_det:.4e}')
        axes[1].axvspan(vz_det - bw/2, vz_det + bw/2, color='green', alpha=0.2)

    axes[1].set_title(f"Longitudinal Spectrum (Z) - {escape_latex(fpath.name)}")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].set_xlim(0, 0.1)
    axes[1].grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    
    # --- Save image in output/plot_beam_spectrum/ ---
    output_dir = BASE_DIR / "output" / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_img = output_dir / f"{fpath.stem}_spectrum_filtering.png"
    plt.savefig(out_img, dpi=300)
    print(f"Spectrum plot saved to: {out_img}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot Beam Spectrum with filtering options.")
    parser.add_argument("--file", type=str, help="Path to a specific .w2 file to plot.")
    parser.add_argument("--base", action="store_true", help="Plot the baseline run1.w2 from input folder.")
    args = parser.parse_args()

    target_file = None

    if args.file:
        target_file = Path(args.file)
    elif args.base:
        target_file = BASELINE_PATH
        print("Mode: Plotting baseline run1.w2")
    else:
        # Default: Latest scan result
        scan_dir = get_latest_scan_dir()
        if not scan_dir:
            print(f"No scan results found in {RESULTS_BASE}")
            return
        
        print(f"Mode: Latest scan directory -> {scan_dir.name}")
        w2_files = list(scan_dir.glob("*_check.w2"))
        if not w2_files:
            print("No '_check.w2' files found in scan directory.")
            return
        target_file = w2_files[0]

    if target_file:
        plot_spectrum(target_file)

if __name__ == "__main__":
    main()
