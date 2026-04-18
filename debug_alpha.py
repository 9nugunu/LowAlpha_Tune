import sdds
import subprocess
import sys
import os
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Add src to path for config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import LATEST_SCAN_DIR

# ============================================================
# Quick Debug Runner: Verify alphac and plot LPS distributions
# Uses LATEST_SCAN_DIR from config.py
# Supports multiprocessing with progress bars
# ============================================================

PYTHON_EXE = sys.executable

# Non-interactive backend for multiprocessing
plt.switch_backend("Agg")


def _verify_single_result(f_path):
    """Worker: Verify alphac for a single file."""
    f = Path(f_path)
    try:
        s = sdds.SDDS(0)
        match = re.search(r"A([\d\.eE+-]+)", f.name)
        target = float(match.group(1)) if match else 0.0
        
        s.load(str(f))
        actual = s.parameterData[s.parameterName.index("alphac")][0]
        diff = actual - target
        return (f.name, target, actual, diff)
    except Exception as e:
        return (f.name, 0.0, 0.0, str(e))


def verify_results(base_path, n_workers=None):
    """Verify alphac values match targets from filenames using multiprocessing."""
    print(f"\n--- Verifying Results in {base_path.name} ---")
    
    twi_files = sorted(list(base_path.glob("*_final.twi")))
    if not twi_files:
        print("No final.twi files found. Did the simulation fail?")
        return

    if n_workers is None:
        n_workers = max(1, cpu_count() - 4)

    print(f"  Verifying {len(twi_files)} files using {n_workers} workers...")
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_verify_single_result, [str(f) for f in twi_files]), 
                            total=len(twi_files), desc="Verifying alphac", unit="file", leave=False))

    print(f"\n{'Filename':<40} | {'Target APVAL':<15} | {'Actual alphac':<15} | {'Diff':<10}")
    print("-" * 90)
    for fname, target, actual, diff in results:
        if isinstance(diff, str):
            print(f"{fname:<40} | ERROR: {diff}")
        else:
            print(f"{fname:<40} | {target:<15.2e} | {actual:<15.2e} | {diff:<10.2e}")


def _verify_single_physics(args):
    """Worker: Verify DP and BetaX for a single root."""
    f_twi_path, base_path_str = args
    f_twi = Path(f_twi_path)
    base_path = Path(base_path_str)
    
    root = f_twi.name.replace("_final.twi", "")
    p0 = 1230.922029484751 # Central momentum reference
    
    try:
        s = sdds.SDDS(0)
        
        # Extract target delta
        d_match = re.search(r"D([\d\.eE+-]+)", root)
        target_dp = float(d_match.group(1)) if d_match else 0.0

        applied_dp = 0.0
        actual_dp = 0.0
        betax_wis = 0.0

        # 1. Applied DP from .param
        f_param = base_path / f"{root}_check.param"
        if f_param.exists():
            s.load(str(f_param))
            names = [str(n).strip().upper() for n in s.columnData[s.columnName.index("ElementName")][0]]
            items = [str(i).strip().upper() for i in s.columnData[s.columnName.index("ElementParameter")][0]]
            vals = s.columnData[s.columnName.index("ParameterValue")][0]
            for i, n in enumerate(names):
                if n == "MAL" and items[i] == "DP":
                    applied_dp = vals[i]
                    break

        # 2. Actual DP from .w2
        f_w2 = base_path / f"{root}_check.w2"
        if f_w2.exists():
            s.load(str(f_w2))
            p_data = s.columnData[s.columnName.index("p")]
            p_vals = []
            if isinstance(p_data[0], (list, np.ndarray)):
                for page in p_data: p_vals.extend(page)
            else: p_vals = p_data
            actual_dp = (np.mean(p_vals) - p0) / p0

        # 3. BetaX at WISLAND from _final.twi
        s.load(str(f_twi))
        names = [str(n).strip().upper() for n in s.columnData[s.columnName.index("ElementName")][0]]
        bx_data = s.columnData[s.columnName.index("betax")]
        bx = bx_data[0] if isinstance(bx_data[0], (list, np.ndarray)) else bx_data
        try:
            betax_wis = bx[names.index("WISLAND")]
        except:
            betax_wis = 0.0
            
        return (root, target_dp, applied_dp, actual_dp, betax_wis)
    except Exception as e:
        return (root, 0.0, 0.0, 0.0, f"Error: {e}")


def verify_physics(base_path, n_workers=None):
    """Verify physics: DP applied vs actual, BetaX at watch point using multiprocessing."""
    print(f"\n--- Verifying Physics for All Points (DP & Beta) ---")
    
    twi_files = sorted(list(base_path.glob("*_final.twi")))
    if not twi_files:
        print("No results found.")
        return

    if n_workers is None:
        n_workers = max(1, cpu_count() - 4)

    print(f"  Analysing {len(twi_files)} points using {n_workers} workers...")
    
    task_args = [(str(f), str(base_path)) for f in twi_files]
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_verify_single_physics, task_args), 
                            total=len(twi_files), desc="Checking Physics", unit="point", leave=False))

    print(f"\n{'Rootname':<40} | {'Targ DP':<10} | {'Appl DP':<10} | {'Act DP':<10} | {'BetaX':<8}")
    print("-" * 95)
    for root, target_dp, applied_dp, actual_dp, betax_info in results:
        if isinstance(betax_info, str):
            print(f"{root:<40} | {betax_info}")
        else:
            print(f"{root:<40} | {target_dp:<10.2e} | {applied_dp:<10.2e} | {actual_dp:<10.2e} | {betax_info:<8.3f}")


def _get_single_w2_limits(f_w2_path):
    """Worker: Find p and dt limits for a single .w2 file."""
    try:
        s = sdds.SDDS(0)
        s.load(f_w2_path)
        p_data = s.columnData[s.columnName.index("p")]
        dt_data = s.columnData[s.columnName.index("dt")]
        p_vals = np.concatenate(p_data) if isinstance(p_data[0], (list, np.ndarray)) else np.array(p_data)
        dt_vals = np.concatenate(dt_data) if isinstance(dt_data[0], (list, np.ndarray)) else np.array(dt_data)
        return p_vals.min(), p_vals.max(), dt_vals.min(), dt_vals.max()
    except:
        return None


def _plot_single_lps(args):
    """Worker: Load a single .w2 file and create its LPS plot."""
    f_w2_path, plot_dir, x_lims, y_lims = args
    f_w2 = Path(f_w2_path)
    
    try:
        s = sdds.SDDS(0)
        root = f_w2.name.replace("_check.w2", "")
        s.load(str(f_w2))
        
        p_data = s.columnData[s.columnName.index("p")]
        dt_data = s.columnData[s.columnName.index("dt")]
        p_vals = np.concatenate(p_data) if isinstance(p_data[0], (list, np.ndarray)) else np.array(p_data)
        dt_vals = np.concatenate(dt_data) if isinstance(dt_data[0], (list, np.ndarray)) else np.array(dt_data)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(dt_vals * 1e12, p_vals, s=2, alpha=0.5, color='red')
        ax.set_title(f"Longitudinal Phase Space (LPS)\n{root}")
        ax.set_xlabel("dt [ps]")
        ax.set_ylabel("p")
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        save_path = Path(plot_dir) / f"{root}_LPS.png"
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
        return f"OK: {f_w2.name}"
    except Exception as e:
        return f"ERROR: {f_w2.name} - {e}"


def plot_distributions(base_path, n_workers=None):
    """Plot LPS (dt vs p) for all .w2 files using multiprocessing."""
    plot_dir = base_path / "debug_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Saving Diagnostic Plots to {plot_dir} ---")
    
    w2_files = sorted(list(base_path.glob("*_check.w2")))
    if not w2_files:
        print("No .w2 files found for plotting.")
        return

    if n_workers is None:
        n_workers = max(1, cpu_count() - 4)

    # Pass 1: Global limits - Only scan files with max delta per alpha (optimization)
    # Group files by alpha and find max delta per group
    alpha_max_delta_files = {}
    for f in w2_files:
        # Extract alpha and delta from filename: opt_A{alpha}_D{delta}_check.w2
        a_match = re.search(r"A([\d\.eE+-]+)", f.name)
        d_match = re.search(r"D([\d\.eE+-]+)", f.name)
        if a_match and d_match:
            alpha_val = a_match.group(1)
            delta_val = float(d_match.group(1))
            if alpha_val not in alpha_max_delta_files or delta_val > alpha_max_delta_files[alpha_val][1]:
                alpha_max_delta_files[alpha_val] = (f, delta_val)
    
    representative_files = [v[0] for v in alpha_max_delta_files.values()]
    print(f"  Calculating global axis limits from {len(representative_files)} representative files (max delta per alpha)...")
    
    with Pool(processes=n_workers) as pool:
        limit_results = list(tqdm(pool.imap(_get_single_w2_limits, [str(f) for f in representative_files]), 
                                  total=len(representative_files), desc="Scanning for limits", unit="file", leave=False))
    
    # Aggregate results
    valid_results = [r for r in limit_results if r is not None]
    if not valid_results:
        print("Error: Could not extract limits from any .w2 files.")
        return
        
    global_p_min = min(r[0] for r in valid_results)
    global_p_max = max(r[1] for r in valid_results)
    global_dt_min = min(r[2] for r in valid_results)
    global_dt_max = max(r[3] for r in valid_results)

    p_buf = (global_p_max - global_p_min) * 0.05 if global_p_max > global_p_min else 0.1
    dt_buf = (global_dt_max - global_dt_min) * 0.05 if global_dt_max > global_dt_min else 0.1
    y_lims = (global_p_min - p_buf, global_p_max + p_buf)
    x_lims = ((global_dt_min - dt_buf) * 1e12, (global_dt_max + dt_buf) * 1e12)

    # Pass 2: Plotting (Parallel)
    print(f"  Generating {len(w2_files)} plots using {n_workers} workers...")
    task_args = [(str(f_w2), str(plot_dir), x_lims, y_lims) for f_w2 in w2_files]
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_plot_single_lps, task_args), 
                            total=len(task_args), desc="Generating LPS Plots", unit="plot"))
    
    success_count = sum(1 for r in results if r.startswith("OK"))
    print(f"\n  Completed: {success_count}/{len(w2_files)} plots generated successfully.")
    print(f"All LPS plots saved to: {plot_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug alpha verification and LPS plotting.")
    parser.add_argument("--run-scan", action="store_true", 
                        help="Run a new test scan instead of using LATEST_SCAN_DIR from config.")
    # Default workers to cpu_count() - 4
    default_workers = max(1, cpu_count() - 4)
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Number of parallel workers (default: {default_workers}).")
    args = parser.parse_args()
    
    if args.run_scan:
        print("--- Running Test Scan Points ---")
        cmd = [
            PYTHON_EXE, "src/scan_alphac_pyele.py",
            "--startA", "1.0e-5", "--stopA", "1.1e-4", "--stepA", "5.0e-5",
            "--startD", "1.0e-4", "--stopD", "2.4e-4", "--stepD", "0.6e-4"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        match = re.search(r"Simulation session: (scan_[\w\.-]+)", result.stdout)
        if match:
            session_name = match.group(1)
            base_path = Path("output/scan_alphac_pyele") / session_name
        else:
            print("Failed to start/identify simulation session.")
            sys.exit(1)
    else:
        base_path = LATEST_SCAN_DIR
        print(f"Using scan directory from config: {base_path}")
    
    if not base_path.exists():
        print(f"Error: Scan directory does not exist: {base_path}")
        sys.exit(1)
    
    # 1. Verify results (Parallel)
    verify_results(base_path, n_workers=args.workers)
    
    # 2. Verify physics (Parallel)
    verify_physics(base_path, n_workers=args.workers)
    
    # 3. Plot distributions (Parallel)
    plot_distributions(base_path, n_workers=args.workers)
