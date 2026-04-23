from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalized(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True)
class ScanConfig:
    startA: float
    stopA: float
    stepA: float
    startD: float
    stopD: float
    stepD: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "startA", _normalized(self.startA))
        object.__setattr__(self, "stopA", _normalized(self.stopA))
        object.__setattr__(self, "stepA", _normalized(self.stepA))
        object.__setattr__(self, "startD", _normalized(self.startD))
        object.__setattr__(self, "stopD", _normalized(self.stopD))
        object.__setattr__(self, "stepD", _normalized(self.stepD))

    def session_dir_name(self) -> str:
        a_range = f"A{self.startA:.2e}-{self.stopA:.2e}"
        a_step = f"sA{self.stepA:.2e}"
        d_range = f"D{self.startD:.2e}-{self.stopD:.2e}"
        d_step = f"sD{self.stepD:.2e}"
        return f"scan_{a_range}_{a_step}_{d_range}_{d_step}"

    def metadata(self) -> dict[str, float]:
        return {
            "SCAN_START_A": self.startA,
            "SCAN_STOP_A": self.stopA,
            "SCAN_STEP_A": self.stepA,
            "SCAN_START_D": self.startD,
            "SCAN_STOP_D": self.stopD,
            "SCAN_STEP_D": self.stepD,
        }


# Default scan grid shared by run_pipeline.py, scan_alphac_pyele.py,
# and process_scan_results.py when metadata is unavailable.
DEFAULT_SCAN_CONFIG = ScanConfig(
    startA=5e-5,
    stopA=1e-4,
    stepA=1e-6,
    startD=1.0e-5,
    stopD=2.2e-5,
    stepD=0.2e-6,
)


@dataclass(frozen=True)
class EqBunchConfig:
    """Equilibrium-regime tracking knobs injected into check_eq.ele as macros.

    emit_x / sigma_dp / sigma_s are NOT stored here: elegant computes them
    at run time from twiss radiation_integrals and rf_setup, and
    check_eq.ele wires the derived values back into &bunched_beam via
    &rpn_load tags (tw.ex0, tw.Sdelta0, Sz0). This keeps every (alphac,
    delta) scan point physically self-consistent without the launcher
    having to re-derive lattice-dependent quantities.
    """

    n_passes: int  # tracking turns in Run 2
    n_particles: int  # particles per bunch in Run 1

    def metadata(self) -> dict[str, int | str]:
        return {
            "regime": "equilibrium",
            "n_particles": self.n_particles,
            "n_passes": self.n_passes,
        }


DEFAULT_EQ_CONFIG = EqBunchConfig(
    n_passes=50000,
    n_particles=10000,
)


def build_scan_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Build a scan axis with inclusive endpoint logic shared by scan and analysis."""
    start = _normalized(start)
    stop = _normalized(stop)
    step = _normalized(step)
    return np.round(np.arange(start, stop + step / 100, step), 12)


def scan_config_from_metadata(metadata: dict[str, float | None]) -> ScanConfig | None:
    required_keys = (
        "SCAN_START_A",
        "SCAN_STOP_A",
        "SCAN_STEP_A",
        "SCAN_START_D",
        "SCAN_STOP_D",
        "SCAN_STEP_D",
    )
    if any(metadata.get(key) is None for key in required_keys):
        return None

    return ScanConfig(
        startA=metadata["SCAN_START_A"],
        stopA=metadata["SCAN_STOP_A"],
        stepA=metadata["SCAN_STEP_A"],
        startD=metadata["SCAN_START_D"],
        stopD=metadata["SCAN_STOP_D"],
        stepD=metadata["SCAN_STEP_D"],
    )

# --- Path Configuration ---
# Set the specific scan folder you want to analyze by default when running
# standalone analysis scripts without an explicit --dir override.
# Example: "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_100000turn_Ok"
# SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_70000/2.5kHz_result"
# SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_70000/4kHz_result"
# SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_70000/1kHz_result"
# SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04-x_signal"
SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04-x_signal_50000_onpass"


BASE_DIR = Path(__file__).parent.parent

# Analysis scripts default to the induced-regime output root but will fall
# back to the equilibrium-regime root (scan_alphac_pyele_eq) when the
# induced root has no matching scan -- this lets the same analysis code
# operate on either scan type. Override with LOW_ALPHA_OUTPUT_ROOT if
# you want a custom location.
_OUTPUT_ROOT_CANDIDATES = (
    BASE_DIR / "output" / "scan_alphac_pyele",
    BASE_DIR / "output" / "scan_alphac_pyele_eq",
)

import os as _os
_env_root = _os.environ.get("LOW_ALPHA_OUTPUT_ROOT")
if _env_root:
    OUTPUT_ROOT = Path(_env_root)
    # When an explicit root is chosen, restrict the search to that root.
    _OUTPUT_ROOT_CANDIDATES = (OUTPUT_ROOT,)
else:
    OUTPUT_ROOT = _OUTPUT_ROOT_CANDIDATES[0]

CONFIGURED_SCAN_DIR = OUTPUT_ROOT / SCAN_FOLDER_NAME


def _resolve_latest_scan_dir(output_root: Path, configured_scan_dir: Path) -> Path:
    if configured_scan_dir.exists():
        return configured_scan_dir

    # Candidate roots: the explicit output_root first (honors env override),
    # then the other built-in candidates as a fallback.
    search_roots: list[Path] = [output_root]
    for root in _OUTPUT_ROOT_CANDIDATES:
        if root != output_root:
            search_roots.append(root)

    # First pass: look for the configured scan name in any search root.
    for root in search_roots:
        candidate = root / SCAN_FOLDER_NAME
        if candidate.exists():
            return candidate

    # Second pass: pick the most recent scan across all search roots.
    all_scans: list[Path] = []
    for root in search_roots:
        if root.exists():
            all_scans.extend(
                path for path in root.iterdir()
                if path.is_dir() and path.name.startswith("scan_")
            )
    if all_scans:
        return max(all_scans, key=lambda path: path.stat().st_mtime)

    return configured_scan_dir


LATEST_SCAN_DIR = _resolve_latest_scan_dir(OUTPUT_ROOT, CONFIGURED_SCAN_DIR)

# --- Instrumentation & Calibration Parameters ---
POS_SENSITIVITY = 388.0    # mV / mm / nC
BEAM_CURRENT = 10.0        # uA
IMPEDANCE = 50.0           # Ohm
T_REV = 160e-9             # seconds
ASSUMED_NOISE_FLOOR = -90.0 # dBm

# Derived
BUNCH_CHARGE_NC = (BEAM_CURRENT * 1e-6) * T_REV * 1e9 # nC
