import concurrent.futures
from pathlib import Path
import subprocess
import numpy as np
import os
import shutil
import json
import argparse
import time
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Scan AlphaC and Delta parameters.")
    parser.add_argument("--startA", type=float, default=10**-5)
    parser.add_argument("--stopA", type=float, default=10**-4)
    parser.add_argument("--stepA", type=float, default=0.1*10**-5)
    parser.add_argument("--startD", type=float, default=2.6*10**-5)
    parser.add_argument("--stopD", type=float, default=3.2*10**-5)
    parser.add_argument("--stepD", type=float, default=0.2*10**-6)
    return parser.parse_args()


def safe_rename(src: Path, dst: Path, retries: int = 5, delay: float = 0.5):
    """Reliably rename/move a file with retries for Windows file lock issues."""
    for i in range(retries):
        try:
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            return True
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            print(f"Unexpected error renaming {src} -> {dst}: {e}")
            raise
    return False

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR
PROJECT_ROOT = BASE_DIR.parent
ELEGANT_CMD = "elegant"
OPT_FILE_NAME = "opt.ele"
CHECK_FILE_NAME = "check.ele"
LTE_FILE_NAME = "mlsLA.LTE"
RESULTS_BASE_DIR = PROJECT_ROOT / "output" / "scan_alphac_pyele"

# parallel workers for sweep (adjust as needed)
MAX_WORKERS = os.cpu_count() - 4 or 4
TEST_MODE = False  # Set to True for quick testing with limited steps
TEST_MAX_STEPS = 2

# Scan ranges (set from CLI args in __main__)
SCAN_START_A = SCAN_STOP_A = SCAN_STEP_A = 0.0
SCAN_START_D = SCAN_STOP_D = SCAN_STEP_D = 0.0


def run(cmd: str, cwd: Path) -> bool:
    """Run a shell command in specified cwd, return True if successful."""
    print(f"Running: {cmd} (cwd={cwd})")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode()}")
        return False


def run_opt_task(alpha_target: float, delta_target: float, work_dir: Path) -> tuple[str, bool]:
    """Run a single optimization job and return (rootname, success)."""
    rootname = f"opt_A{alpha_target:.2e}_D{delta_target:.2e}"
    cmd = (
        f"{ELEGANT_CMD} {OPT_FILE_NAME} "
        f"-macro=rootname={rootname},APVAL={alpha_target:.6e},DELTA={delta_target:.6e}"
    )
    success = run(cmd, cwd=work_dir)
    # Verification: Does the lattice file exist? (Matching opt.ele: <rootname>_opt.new)
    if success and not (work_dir / f"{rootname}_opt.new").exists():
        success = False
        print(f"Alert: {rootname}_opt.new was not generated despite successful command exit.")
    return rootname, success


def sweep_optics(work_dir: Path) -> list[str]:
    """Run the optimization scan and return generated root paths that succeeded."""
    alpha_values = np.round(np.arange(SCAN_START_A, SCAN_STOP_A + SCAN_STEP_A / 100, SCAN_STEP_A), 10)
    delta_values = np.round(np.arange(SCAN_START_D, SCAN_STOP_D + SCAN_STEP_D / 100, SCAN_STEP_D), 10)
    # Ensure coordinates are clean rounded floats
    tasks = [(round(float(a), 12), round(float(d), 12)) for a in alpha_values for d in delta_values]

    if TEST_MODE:
        tasks = tasks[:TEST_MAX_STEPS]
        print(f"--- TEST MODE ACTIVE: Limiting sweep to {TEST_MAX_STEPS} steps ---")

    print(f"Starting sweep: {len(tasks)} jobs total.")

    results: dict[str, bool] = {}
    total_tasks = len(tasks)
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_opt_task, a, d, work_dir): (a, d) for a, d in tasks}
        for future in concurrent.futures.as_completed(futures):
            rootname, success = future.result()
            results[rootname] = success
            completed += 1
            percent = (completed / total_tasks) * 100
            sys.stdout.write(f"\rOptimization Progress: [{completed}/{total_tasks}] {percent:.1f}% ")
            sys.stdout.flush()
    print() # New line after progress

    success_roots = [r for r, s in results.items() if s]
    failed_roots = [r for r, s in results.items() if not s]

    print(f"\nOptimization Sweep Summary:")
    print(f"- Total requested: {len(tasks)}")
    print(f"- Successful: {len(success_roots)}")
    if failed_roots:
        print(f"- FAILED: {len(failed_roots)}")
        for r in failed_roots:
            print(f"  [!] Missing result: {r}")
    
    return sorted(success_roots)


def run_single_check(rootname: str, work_dir: Path) -> bool:
    """Run check.ele for a specific rootname and return success."""
    lattice_file = work_dir / f"{rootname}_opt.new"
    if not lattice_file.exists():
        return False

    # Extract delta from the rootname (format opt_A{alpha}_D{delta})
    delta_target = None
    try:
        delta_str = rootname.split("_D")[-1]
        delta_target = float(delta_str)
    except Exception:
        pass

    cmd = f"{ELEGANT_CMD} {CHECK_FILE_NAME} -macro=lattice={lattice_file.name},rootname={rootname}"
    if delta_target is not None:
        cmd += f",DELTA={delta_target:.6e}"
    
    success = run(cmd, cwd=work_dir)
    
    # Rename watch outputs and verify
    found_w2 = False
    for ext in (".w1", ".w2", ".param"):
        src = work_dir / f"{rootname}{ext}"
        if src.exists():
            dst = work_dir / f"{rootname}_check{ext}"
            # Use the new robust rename function
            try:
                if safe_rename(src, dst):
                    if ext == ".w2":
                        found_w2 = True
            except Exception as e:
                print(f"Failed to rename {src.name} after retries: {e}")
    
    return success and found_w2


def run_checks(rootnames: list[str], work_dir: Path) -> None:
    """Run check.ele on each lattice result in parallel with verification."""
    print(f"Starting parallel checks for {len(rootnames)} results...")
    
    results: dict[str, bool] = {}
    total_checks = len(rootnames)
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_check, r, work_dir): r for r in rootnames}
        for future in concurrent.futures.as_completed(futures):
            rootname = futures[future]
            results[rootname] = future.result()
            completed += 1
            percent = (completed / total_checks) * 100
            sys.stdout.write(f"\rCheck Progress: [{completed}/{total_checks}] {percent:.1f}% ")
            sys.stdout.flush()
    print() # New line after progress

    success_count = sum(1 for s in results.values() if s)
    failed_list = [r for r, s in results.items() if not s]

    print(f"\nCheck Summary:")
    print(f"- Total attempted: {len(rootnames)}")
    print(f"- Successful: {success_count}")
    if failed_list:
        print(f"- FAILED: {len(failed_list)}")
        for r in failed_list:
            print(f"  [!] Missing check output: {r}")


def main() -> None:
    # 1. Create session directory based on scan ranges
    a_range = f"A{SCAN_START_A:.2e}-{SCAN_STOP_A:.2e}"
    d_range = f"D{SCAN_START_D:.2e}-{SCAN_STOP_D:.2e}"
    session_dir = RESULTS_BASE_DIR / f"scan_{a_range}_{d_range}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Simulation session: {session_dir.name}")

    # 1.1 Store metadata for analysis scripts
    metadata = {
        "SCAN_START_A": float(SCAN_START_A),
        "SCAN_STOP_A": float(SCAN_STOP_A),
        "SCAN_STEP_A": float(SCAN_STEP_A),
        "SCAN_START_D": float(SCAN_START_D),
        "SCAN_STOP_D": float(SCAN_STOP_D),
        "SCAN_STEP_D": float(SCAN_STEP_D),
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # 2. Copy necessary source files into session directory
    required_files = [
        "run0.new", "run0.twi", 
        OPT_FILE_NAME, CHECK_FILE_NAME, LTE_FILE_NAME
    ]
    for filename in required_files:
        src = SRC_DIR / filename
        if src.exists():
            shutil.copy(src, session_dir)
        else:
            print(f"Warning: {filename} not found in {SRC_DIR}")

    # 3. Execute sweep and checks with strict verification
    success_opt_roots = sweep_optics(session_dir)
    run_checks(success_opt_roots, session_dir)
    
    # Final check: compare requested points with produced files
    print(f"\nFinal Directory Audit of {session_dir}:")
    w2_count = len(list(session_dir.glob("*_check.w2")))
    print(f"Total *_check.w2 files found: {w2_count}")
    
    print(f"\nAll tasks processed. Results stored in: {session_dir}")


if __name__ == "__main__":
    args = get_args()
    SCAN_START_A = args.startA
    SCAN_STOP_A = args.stopA
    SCAN_STEP_A = args.stepA
    SCAN_START_D = args.startD
    SCAN_STOP_D = args.stopD
    SCAN_STEP_D = args.stepD
    main()
