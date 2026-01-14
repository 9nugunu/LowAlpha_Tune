import sdds
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq
from pathlib import Path
import os
import argparse

# --- Constants from MLS.py ---
c0 = 299792458
T0 = 48 / c0  # Revolution period for MLS (Circumference 48m)
vx = 1.1144067
vz = 0.007082
bw = 0.05

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

def filtering_norm(freq, data, cen_f, cutoff, ttt):
    """Frequency filtering logic from MLS.py."""
    lowband = cen_f - cutoff/2.0
    highband = cen_f + cutoff/2.0
    tmp = data.copy()
    
    for i in range(len(freq)):
        f_abs = np.abs(freq[i])
        if ttt == 2:
            # Sideband filtering logic
            if (lowband - vz < f_abs < highband - vz) or (lowband + vz < f_abs < highband + vz):
                pass
            else:
                tmp[i] = 0
        else:
            # Standard bandpass logic
            if lowband < f_abs < highband:
                pass
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
    x_fft_main = fftshift(fft(x))
    
    # Replicate filtering from MLS.py
    x_filt_1 = filtering_norm(freq, x_fft_main, vx, bw, 0)
    x_filt_2 = filtering_norm(freq, x_fft_main, vz, bw, 1) # Note: naming suggests vz but filtering logic depends on ttt
    x_filt_3 = filtering_norm(freq, x_fft_main, vx, bw, 2) # Sideband?

    # --- Longitudinal (Z) Analysis ---
    z_dist = dt * c0
    z_fft_main = fftshift(fft(z_dist))

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Positive mask for plotting
    mask = freq > 0
    freq_p = freq[mask]
    
    # X Plot (overlay original and filtered)
    axes[0].semilogy(freq_p, np.abs(x_fft_main[mask]), label='Original', color='gray', alpha=0.5)
    axes[0].semilogy(freq_p, np.abs(x_filt_1[mask]), label=f'Filtered (Tune X={vx:.4f})', color='red')
    axes[0].semilogy(freq_p, np.abs(x_filt_2[mask]), label=f'Filtered (Tune Z peak?)', color='green')
    axes[0].semilogy(freq_p, np.abs(x_filt_3[mask]), label=f'Filtered (Sideband?)', color='blue')
    
    axes[0].set_title(f"Horizontal Spectrum (X) - {fpath.name}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Amplitude", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, which="both", ls=":", alpha=0.5)

    # Z Plot
    axes[1].semilogy(freq_p, np.abs(z_fft_main[mask]), label='Original', color='blue')
    # Add horizontal/longitudinal labels consistent with MLS.py if needed
    axes[1].set_title(f"Longitudinal Spectrum (Z) - {fpath.name}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Frequency [MHz]", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Amplitude", fontsize=12, fontweight='bold')
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
