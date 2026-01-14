"""
Sweep alpha_c targets with elegant and post-check each lattice using check.ele.

Optimization outputs follow the defaults set in opt.ele (rootname only), and
each checked Twiss file is written under ./results/ with a name
matching its root (e.g., opt_A1.00e-05_D1.00e-04.check.twi).
"""

import concurrent.futures
from pathlib import Path
import subprocess
import numpy as np
import os


BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
PROJECT_ROOT = BASE_DIR.parents[2]
ELEGANT_CMD = "elegant"
OPT_FILE = SRC_DIR / "opt.ele"
CHECK_FILE = SRC_DIR / "check.ele"
CHECK_OUTPUT_DIR = BASE_DIR / "results"

# alpha_c scan range
SCAN_START_A = 10 ** -5
SCAN_STOP_A = 1.01 * 10 ** -4
SCAN_STEP_A = 0.1 * 10 ** -5

# delta (energy spread) scan range
SCAN_START_D = 1.4 * 10 ** -5
SCAN_STOP_D = 2.6 * 10 ** -5
SCAN_STEP_D = 0.2 * 10 ** -6

# parallel workers for sweep (adjust as needed)
MAX_WORKERS = os.cpu_count() or 4
# limit number of jobs for quick test runs
TEST_MAX_STEPS = 1


def run(cmd: str, cwd: Path = BASE_DIR) -> None:
    """Run a shell command in BASE_DIR, raising on failure."""
    print(f"Running: {cmd} (cwd={cwd})")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def run_opt_task(alpha_target: float, delta_target: float) -> str:
    """Run a single optimization job and return its rootname."""
    rootname = f"opt_A{alpha_target:.2e}_D{delta_target:.2e}"
    cmd = (
        f"{ELEGANT_CMD} {OPT_FILE.name} "
        f"-macro=rootname={rootname},APVAL={alpha_target:.6e},DELTA={delta_target:.6e}"
    )
    run(cmd, cwd=SRC_DIR)
    return rootname


def sweep_optics() -> list[str]:
    """Run the optimization scan and return generated root paths (as strings)."""
    if not OPT_FILE.exists():
        raise FileNotFoundError(f"Missing {OPT_FILE}")

    alpha_values = np.arange(SCAN_START_A, SCAN_STOP_A + SCAN_STEP_A / 100, SCAN_STEP_A)
    delta_values = np.arange(SCAN_START_D, SCAN_STOP_D + SCAN_STEP_D / 100, SCAN_STEP_D)
    tasks = [(float(a), float(d)) for a in alpha_values for d in delta_values]

    # For quick testing, only run a handful of jobs
    # tasks = tasks[:TEST_MAX_STEPS]

    print(
        f"Starting sweep over {len(alpha_values)} alpha_c values x "
        f"{len(delta_values)} deltas -> capped to {len(tasks)} jobs (max_workers={MAX_WORKERS})."
    )

    rootnames: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_opt_task, a, d) for a, d in tasks]
        for future in concurrent.futures.as_completed(futures):
            rootnames.append(future.result())

    # keep deterministic ordering by rootname
    rootnames.sort()

    print("Sweep finished.")
    return rootnames


def run_checks(rootnames: list[str]) -> None:
    """Run check.ele on each lattice result and save twiss into ./results/."""
    if not CHECK_FILE.exists():
        raise FileNotFoundError(f"Missing {CHECK_FILE}")

    CHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rootname in rootnames:
        root_path = Path(rootname)
        lattice_file = SRC_DIR / f"{rootname}.new"
        if not lattice_file.exists():
            print(f"Skipping {root_path.name}: {lattice_file.name} not found.")
            continue

        # Extract delta from the rootname (format opt_A{alpha}_D{delta})
        delta_target = None
        try:
            delta_str = root_path.name.split("_D")[-1]
            delta_target = float(delta_str)
        except Exception:
            pass

        tmp_twi = SRC_DIR / f"{root_path.name}_check.twi"
        if tmp_twi.exists():
            tmp_twi.unlink()

        lattice_macro = lattice_file.resolve().as_posix()
        root_macro = root_path.name
        cmd = f"{ELEGANT_CMD} {CHECK_FILE.name} -macro=lattice={lattice_macro},rootname={root_macro}"
        if delta_target is not None:
            cmd += f",DELTA={delta_target:.6e}"
        run(cmd, cwd=SRC_DIR)

        if not tmp_twi.exists():
            print(f"Warning: {tmp_twi.name} not produced for {root_path.name}.")
            continue

        target_twi = CHECK_OUTPUT_DIR / f"{root_path.name}_check.twi"
        if target_twi.exists():
            target_twi.unlink()
        tmp_twi.replace(target_twi)
        rel_path = target_twi.relative_to(PROJECT_ROOT)
        print(f"Saved {rel_path}")

        # Move watch outputs (w1/w2) if present
        for ext in (".w1", ".w2"):
            src = SRC_DIR / f"{root_path.name}{ext}"
            if src.exists():
                dst = CHECK_OUTPUT_DIR / f"{root_path.name}_check{ext}"
                if dst.exists():
                    dst.unlink()
                src.replace(dst)
                print(f"Saved {dst.relative_to(PROJECT_ROOT)}")


def main() -> None:
    rootnames = sweep_optics()
    run_checks(rootnames)


if __name__ == "__main__":
    main()
