import concurrent.futures
from pathlib import Path
import subprocess
import numpy as np
import os
import shutil


BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
PROJECT_ROOT = BASE_DIR.parent
ELEGANT_CMD = "elegant"
OPT_FILE_NAME = "opt.ele"
CHECK_FILE_NAME = "check.ele"
LTE_FILE_NAME = "mlsLA.LTE"
RESULTS_BASE_DIR = PROJECT_ROOT / "output" / Path(__file__).stem

# alpha_c scan range
SCAN_START_A = 10** -5
SCAN_STOP_A = 1.01 * 10 ** -4
SCAN_STEP_A = 0.1 * 10 ** -5

# delta (energy spread) scan range
SCAN_START_D = 1.4 * 10** -5
SCAN_STOP_D = 2.6 * 10 ** -5
SCAN_STEP_D = 0.2 * 10 ** -6

# parallel workers for sweep (adjust as needed)
MAX_WORKERS = os.cpu_count() or 4
TEST_MODE = False  # Set to True for quick testing with limited steps
TEST_MAX_STEPS = 2


def run(cmd: str, cwd: Path) -> None:
    """Run a shell command in specified cwd, raising on failure."""
    print(f"Running: {cmd} (cwd={cwd})")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def run_opt_task(alpha_target: float, delta_target: float, work_dir: Path) -> str:
    """Run a single optimization job and return its rootname."""
    rootname = f"opt_A{alpha_target:.2e}_D{delta_target:.2e}"
    cmd = (
        f"{ELEGANT_CMD} {OPT_FILE_NAME} "
        f"-macro=rootname={rootname},APVAL={alpha_target:.6e},DELTA={delta_target:.6e}"
    )
    run(cmd, cwd=work_dir)
    return rootname


def sweep_optics(work_dir: Path) -> list[str]:
    """Run the optimization scan and return generated root paths (as strings)."""
    alpha_values = np.round(np.arange(SCAN_START_A, SCAN_STOP_A + SCAN_STEP_A / 100, SCAN_STEP_A), 10)
    delta_values = np.round(np.arange(SCAN_START_D, SCAN_STOP_D + SCAN_STEP_D / 100, SCAN_STEP_D), 10)
    tasks = [(float(a), float(d)) for a in alpha_values for d in delta_values]

    if TEST_MODE:
        tasks = tasks[:TEST_MAX_STEPS]
        print(f"--- TEST MODE ACTIVE: Limiting sweep to {TEST_MAX_STEPS} steps ---")

    print(
        f"Starting sweep over {len(alpha_values)} alpha_c values x "
        f"{len(delta_values)} deltas -> {len(tasks)} jobs (max_workers={MAX_WORKERS})."
    )

    rootnames: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_opt_task, a, d, work_dir) for a, d in tasks]
        for future in concurrent.futures.as_completed(futures):
            rootnames.append(future.result())

    rootnames.sort()
    print("Sweep finished.")
    return rootnames


def run_single_check(rootname: str, work_dir: Path) -> None:
    """Run check.ele for a specific rootname."""
    lattice_file = work_dir / f"{rootname}.new"
    if not lattice_file.exists():
        print(f"Skipping {rootname}: {lattice_file.name} not found.")
        return

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
    run(cmd, cwd=work_dir)

    # Rename watch outputs (w1/w2) to include _check suffix for consistency
    for ext in (".w1", ".w2"):
        src = work_dir / f"{rootname}{ext}"
        if src.exists():
            dst = work_dir / f"{rootname}_check{ext}"
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            print(f"Renamed {src.name} -> {dst.name}")


def run_checks(rootnames: list[str], work_dir: Path) -> None:
    """Run check.ele on each lattice result in parallel."""
    print(f"Starting parallel checks for {len(rootnames)} results (max_workers={MAX_WORKERS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_check, r, work_dir) for r in rootnames]
        concurrent.futures.wait(futures)
    print("All checks completed.")


def main() -> None:
    # 1. Create session directory based on scan ranges
    a_range = f"A{SCAN_START_A:.2e}-{SCAN_STOP_A:.2e}"
    d_range = f"D{SCAN_START_D:.2e}-{SCAN_STOP_D:.2e}"
    session_dir = RESULTS_BASE_DIR / f"scan_{a_range}_{d_range}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Simulation session: {session_dir.name}")

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

    # 3. Execute sweep and checks in the session directory
    rootnames = sweep_optics(session_dir)
    run_checks(rootnames, session_dir)
    
    print(f"\nAll tasks complete. Results stored in: {session_dir}")


if __name__ == "__main__":
    main()
