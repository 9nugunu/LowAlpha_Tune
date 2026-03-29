import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Project root on sys.path so `from src.xxx` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_theory_vs_simulation import load_simulation_slice, theoretical_offsets_um, ensure_output_dir, DEFAULT_RUNFILENAME
from src.config import LATEST_SCAN_DIR

def check_offset_deviation():
    # 1. Determine Scan Range from Metadata (X.csv)
    X_PATH = LATEST_SCAN_DIR / "X.csv"
    if not X_PATH.exists():
        print(f"Error: {X_PATH} not found.")
        return

    print(f"Loading metadata from: {X_PATH}")
    df = pd.read_csv(X_PATH, index_col=0)
    
    # Get all delta values available in the scan
    # Note: index is usually normalized (e.g. 1.0, 1.1...), we need to confirm scale.
    # load_simulation_slice expects 'delta_target' in *real units* (e.g. 1.0e-4)
    # The index in csv is usually multiplied by 10^4.
    deltas_norm = np.array(df.index, dtype=float)
    deltas_real = deltas_norm * 1e-4
    
    alphas_norm = np.array(df.columns, dtype=float)
    alphas_real = alphas_norm * 1e-4
    
    # Target: Lowest Alpha (First Column)
    target_alpha_idx = 0
    target_alpha_val = alphas_real[target_alpha_idx]
    
    print(f"Analyzing deviation at Lowest Alpha: {target_alpha_val:.3e} ({alphas_norm[target_alpha_idx]:.2f}e-4)")
    
    results = []
    
    # 2. Iterate through ALL deltas using the shared loader
    print(f"Scanning {len(deltas_real)} delta points...")
    
    for delta in deltas_real:
        # A. Load Simulation Slice (returns arrays for ALL alphas at this delta)
        # alpha_grid, alpha_axis, sim_x, sim_z, delta_used = load_simulation_slice(delta)
        alpha_grid, _, sim_x_arr, _, _ = load_simulation_slice(delta)
        
        # B. Load Theoretical Slice
        th_x_arr, _ = theoretical_offsets_um(alpha_grid, delta)
        
        # C. Extract value at lowest alpha index
        sim_val = sim_x_arr[target_alpha_idx]
        th_val = th_x_arr[target_alpha_idx]
        
        # Calculations
        diff = sim_val - th_val
        ratio = th_val / sim_val if sim_val != 0 else np.nan
        
        results.append({
            "delta_scan": delta * 1e4,
            "delta": delta,
            "sim_x": sim_val,
            "th_x": th_val,
            "diff": diff,
            "ratio": ratio
        })
        
    res_df = pd.DataFrame(results)
    
    # 3. Output & Plotting
    out_dir = ensure_output_dir("debug_deviation")
    res_df.to_csv(out_dir / "deviation_results_linked.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Deviation (Left Axis)
    color1 = 'tab:purple'
    ax1.set_xlabel(r"$\delta \times 10^{-4}$")
    ax1.set_ylabel(r"Deviation ($X_{sim} - X_{theory}$) [$\mu m$]", color=color1, fontweight='bold')
    ax1.plot(res_df['delta_scan'], res_df['diff'], 'o-', color=color1, label='Deviation')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot Ratio (Right Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:red'
    ax2.set_ylabel(r"Ratio ($X_{theory} / X_{sim}$)", color=color2, fontweight='bold')  # we already handled the x-label with ax1
    ax2.plot(res_df['delta_scan'], res_df['ratio'], 's--', color=color2, label='Ratio (Th/Sim)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined Title
    plt.title(f"X Offset Analysis at $\\alpha_c={target_alpha_val*1e4:.2f}\\times 10^{-4}$")
    
    # Grid usually on ax1
    ax1.grid(True, alpha=0.3)
    
    plot_path = out_dir / "deviation_plot_linked.png"
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    print("\n--- Top 5 Rows (Linked Logic) ---")
    print(res_df.head())

if __name__ == "__main__":
    check_offset_deviation()
