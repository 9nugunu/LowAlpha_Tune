import os
import json
from pathlib import Path
import sys

def check_progress():
    results_base = Path("output/scan_alphac_pyele")
    if not results_base.exists():
        print("No simulation output directory found.")
        return

    # Find the latest session
    sessions = sorted([d for d in results_base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    if not sessions:
        print("No simulation sessions found.")
        return

    session_dir = sessions[0]
    metadata_file = session_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Metadata not found in {session_dir.name}")
        return

    with open(metadata_file, "r") as f:
        meta = json.load(f)

    # Calculate expected total tasks
    import numpy as np
    a_vals = np.arange(meta["SCAN_START_A"], meta["SCAN_STOP_A"] + meta["SCAN_STEP_A"]/100, meta["SCAN_STEP_A"])
    d_vals = np.arange(meta["SCAN_START_D"], meta["SCAN_STOP_D"] + meta["SCAN_STEP_D"]/100, meta["SCAN_STEP_D"])
    total_expected = len(a_vals) * len(d_vals)

    # Count produced files
    opt_files = list(session_dir.glob("*_opt.new"))
    check_files = list(session_dir.glob("*_check.w2"))
    
    opt_count = len(opt_files)
    check_count = len(check_files)

    print(f"\nSimulation Session: {session_dir.name}")
    print(f"Total Points Expected: {total_expected}")
    print("-" * 40)
    
    opt_perc = (opt_count / total_expected) * 100
    print(f"Optimization Stage:  [{opt_count}/{total_expected}] {opt_perc:.1f}%")
    
    check_perc = (check_count / total_expected) * 100
    print(f"Verification Stage:  [{check_count}/{total_expected}] {check_perc:.1f}%")
    
    if opt_count == total_expected and check_count == total_expected:
        print("\nStatus: COMPLETE")
    elif opt_count > 0 or check_count > 0:
        print("\nStatus: RUNNING")
    else:
        print("\nStatus: PENDING/STARTING")

if __name__ == "__main__":
    check_progress()
