"""Equilibrium-regime master pipeline.

Mirrors scripts/run_pipeline.py but drives the eq launcher + analysis:
  1. scan_alphac_pyele_eq.py   -> Elegant multi-particle tracking scan
  2. process_scan_results.py   -> FFT / amplitude grids / contour plots

No CLI arguments are required. All defaults come from src/config.py
(DEFAULT_SCAN_CONFIG for the alpha/delta grid, DEFAULT_EQ_CONFIG for
n_particles/n_passes). Override only if you want something other than
the standard production run.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_EQ_CONFIG, DEFAULT_SCAN_CONFIG, ScanConfig

PYTHON_EXE = sys.executable
EQ_OUTPUT_ROOT = PROJECT_ROOT / "output" / "scan_alphac_pyele_eq"


def scan_config_from_args(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        startA=args.startA,
        stopA=args.stopA,
        stepA=args.stepA,
        startD=args.startD,
        stopD=args.stopD,
        stepD=args.stepD,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master pipeline: equilibrium-regime scan + analysis."
    )
    parser.add_argument("--startA", type=float, default=DEFAULT_SCAN_CONFIG.startA)
    parser.add_argument("--stopA", type=float, default=DEFAULT_SCAN_CONFIG.stopA)
    parser.add_argument("--stepA", type=float, default=DEFAULT_SCAN_CONFIG.stepA)
    parser.add_argument("--startD", type=float, default=DEFAULT_SCAN_CONFIG.startD)
    parser.add_argument("--stopD", type=float, default=DEFAULT_SCAN_CONFIG.stopD)
    parser.add_argument("--stepD", type=float, default=DEFAULT_SCAN_CONFIG.stepD)
    parser.add_argument("--n-particles", type=int, default=DEFAULT_EQ_CONFIG.n_particles)
    parser.add_argument("--n-passes", type=int, default=DEFAULT_EQ_CONFIG.n_passes)
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Run the scan only and stop before FFT / plotting.",
    )
    return parser.parse_args(argv)


def run_scan(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Equilibrium-regime tracking scan")
    print("=" * 60)
    cmd = [
        PYTHON_EXE, "src/scan_alphac_pyele_eq.py",
        "--startA", str(args.startA),
        "--stopA",  str(args.stopA),
        "--stepA",  str(args.stepA),
        "--startD", str(args.startD),
        "--stopD",  str(args.stopD),
        "--stepD",  str(args.stepD),
        "--n-particles", str(args.n_particles),
        "--n-passes",    str(args.n_passes),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("\n[!] Scan failed. Aborting pipeline.")
        sys.exit(result.returncode)


def run_analysis(session_dir_name: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: FFT + contour analysis")
    print("=" * 60)
    # Point analysis scripts at the eq output root for this invocation.
    env = os.environ.copy()
    env["LOW_ALPHA_OUTPUT_ROOT"] = str(EQ_OUTPUT_ROOT)
    cmd = [
        PYTHON_EXE, "src/process_scan_results.py",
        "--dir", session_dir_name,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print("\n[!] Analysis failed.")
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    session_dir_name = scan_config_from_args(args).session_dir_name()
    print("Equilibrium-regime pipeline started.")
    run_scan(args)
    if args.skip_analysis:
        print("\n--skip-analysis set; stopping after scan.")
        return
    run_analysis(session_dir_name)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"Scan results: {EQ_OUTPUT_ROOT / session_dir_name}")
    print("Analysis plots: output/process_scan_results/<session>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
