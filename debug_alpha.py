import sdds
import subprocess
import sys
import os
import json
import re
from pathlib import Path
import numpy as np

# ============================================================
# Quick Debug Runner: Run 2 scan points and verify alphac
# ============================================================

PYTHON_EXE = sys.executable

def run_test_scan():
    print("--- Running 5 Test Scan Points ---")
    # Small range for testing
    cmd = [
        PYTHON_EXE, "src/scan_alphac_pyele.py",
        "--startA", "1.0e-5", "--stopA", "1.0e-5", "--stepA", "1.0e-5",
        "--startD", "1.0e-5", "--stopD", "4.0e-5", "--stepD", "1.0e-5"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract session dir from output
    match = re.search(r"Simulation session: (scan_[\w\.-]+)", result.stdout)
    if match:
        return match.group(1)
    return None

def verify_results(session_name, s):
    base_path = Path("output/scan_alphac_pyele") / session_name
    print(f"\n--- Verifying Results in {session_name} ---")
    
    twi_files = list(base_path.glob("*_final.twi"))
    if not twi_files:
        print("No final.twi files found. Did the simulation fail?")
        return

    print(f"{'Filename':<40} | {'Target APVAL':<15} | {'Actual alphac':<15} | {'Diff':<10}")
    print("-" * 90)

    # Reuse the provided SDDS object
    
    for f in twi_files:
        # Extract target from filename (e.g., opt_A1.00e-05_D1.00e-05_final.twi)
        match = re.search(r"A([\d\.eE+-]+)", f.name)
        target = float(match.group(1)) if match else 0.0
        
        # Load file into the existing SDDS object
        s.load(str(f))
        actual = s.parameterData[s.parameterName.index("alphac")][0]
        
        diff = actual - target
        print(f"{f.name:<40} | {target:<15.2e} | {actual:<15.2e} | {diff:<10.2e}")

def verify_physics(session_name, s):
    base_path = Path("output/scan_alphac_pyele") / session_name
    print(f"\n--- Verifying Physics for All Points (DP & Beta) ---")
    
    # Get all unique roots from twi files
    twi_files = sorted(list(base_path.glob("*_final.twi")))
    if not twi_files:
        print("No results found.")
        return

    print(f"{'Rootname':<40} | {'Targ DP':<10} | {'Appl DP':<10} | {'Act DP':<10} | {'BetaX':<8}")
    print("-" * 95)

    p0 = 1230.922029484751 # Central momentum reference

    for f_twi in twi_files:
        root = f_twi.name.replace("_final.twi", "")
        # Extract target delta from filename
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

        print(f"{root:<40} | {target_dp:<10.2e} | {applied_dp:<10.2e} | {actual_dp:<10.2e} | {betax_wis:<8.3f}")

if __name__ == "__main__":
    session = run_test_scan()
    if session:
        # Initialize SDDS object once and pass it to verification functions
        s = sdds.SDDS(0)
        verify_results(session, s)
        verify_physics(session, s)
    else:
        print("Failed to start/identify simulation session.")
