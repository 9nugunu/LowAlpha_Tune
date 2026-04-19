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
OUTPUT_ROOT = BASE_DIR / "output" / "scan_alphac_pyele"
CONFIGURED_SCAN_DIR = OUTPUT_ROOT / SCAN_FOLDER_NAME


def _resolve_latest_scan_dir(output_root: Path, configured_scan_dir: Path) -> Path:
    if configured_scan_dir.exists():
        return configured_scan_dir

    if output_root.exists():
        scan_dirs = [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("scan_")]
        if scan_dirs:
            return max(scan_dirs, key=lambda path: path.stat().st_mtime)

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
