import sdds
import numpy as np
from pathlib import Path
import re
import csv
import os
import sys
from multiprocessing import Pool

# Constants
C0 = 299792458

def get_column(fpath, col_name):
    """Load a specific column from an SDDS file."""
    ds = sdds.SDDS(0)
    ds.load(str(fpath))
    try:
        col_index = ds.columnName.index(col_name)
        # Handle potential multi-page data by flattening
        data = ds.columnData[col_index]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
            return np.concatenate(data)
        return np.array(data).ravel()
    except Exception as e:
        print(f"Warning: Could not load '{col_name}' from {fpath.name}: {e}")
        return None

def process_single_file(fpath):
    """Worker function to process a single .w2 file."""
    # Extract parameters from filename (e.g., opt_A1.00e-05_D1.00e-05_check.w2)
    match = re.search(r"A([\d\.eE+-]+)_D([\d\.eE+-]+)", fpath.name)
    alpha = float(match.group(1)) if match else 0.0
    delta = float(match.group(2)) if match else 0.0

    x = get_column(fpath, "x")
    dt = get_column(fpath, "dt")
    
    if x is not None and dt is not None:
        # Stats for X
        x_rms = np.std(x) * 1e6 
        x_max = np.max(np.abs(x)) * 1e6
        
        # Stats for Z (ct)
        z_rms = np.std(dt) * C0 * 1e6
        z_max = np.max(np.abs(dt)) * C0 * 1e6
        
        return {
            "Filename": fpath.name,
            "Alpha": alpha,
            "Delta": delta,
            "X_RMS_um": x_rms,
            "X_Max_um": x_max,
            "Z_RMS_um": z_rms,
            "Z_Max_um": z_max
        }
    return None

def analyze_scan_offsets(scan_dir_path):
    scan_dir = Path(scan_dir_path)
    if not scan_dir.exists():
        print(f"Error: Directory not found: {scan_dir}")
        return

    w2_files = sorted(list(scan_dir.glob("*_check.w2")))
    if not w2_files:
        print(f"No *_check.w2 files found in {scan_dir}")
        return

    print(f"\nAnalyzing offsets for scan: {scan_dir.name}")
    print(f"Total files to process: {len(w2_files)}")
    
    # Multiprocessing setup
    num_cores = os.cpu_count() or 1
    # Windows limit for WaitForMultipleObjects is 64 handles.
    if os.name == 'nt' and num_cores > 60:
        num_cores = 60
    
    print(f"Using {num_cores} cores for multiprocessing...")
    
    results = []
    total_files = len(w2_files)
    completed = 0
    
    with Pool(num_cores) as pool:
        for res in pool.imap(process_single_file, w2_files):
            results.append(res)
            completed += 1
            percent = (completed / total_files) * 100
            sys.stdout.write(f"\rAnalysis Progress: [{completed}/{total_files}] {percent:.1f}% ")
            sys.stdout.flush()
    
    print("\nProcessing complete.")

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("No valid data processed.")
        return

    # Display results (first 20 for brevity if many)
    print(f"\n{'Filename':<40} | {'Alpha':<10} | {'Delta':<10} | {'X RMS [um]':<12} | {'X Max [um]':<12} | {'Z RMS [um]':<12} | {'Z Max [um]':<12}")
    print("-" * 135)
    for res in results[:50]: # Show up to 50 results in console
        print(f"{res['Filename']:<40} | {res['Alpha']:<10.2e} | {res['Delta']:<10.2e} | {res['X_RMS_um']:<12.4f} | {res['X_Max_um']:<12.4f} | {res['Z_RMS_um']:<12.4f} | {res['Z_Max_um']:<12.4f}")
    
    if len(results) > 50:
        print(f"... and {len(results) - 50} more entries (see CSV for full data).")

    # Save to CSV
    csv_path = scan_dir / "scan_rms_offsets.csv"
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\n✅ Full results saved to: {csv_path}")

    return results

if __name__ == "__main__":
    # Use relative path from the script's location
    base_dir = Path(__file__).parent
    target_path = base_dir / "output" / "scan_alphac_pyele" / "scan_A1.00e-05-1.00e-04_D1.00e-04-2.40e-04"
    
    print(f"Targeting: {target_path}")
    analyze_scan_offsets(target_path)
