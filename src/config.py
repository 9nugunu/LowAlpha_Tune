from pathlib import Path

# --- Path Configuration ---
# Set the specific scan folder you want to analyze
# Example: "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_100000turn_Ok"
SCAN_FOLDER_NAME = "scan_A1.00e-05-1.10e-04_D1.00e-04-2.40e-04_70000"

BASE_DIR = Path(__file__).parent.parent
OUTPUT_ROOT = BASE_DIR / "output" / "scan_alphac_pyele"
LATEST_SCAN_DIR = OUTPUT_ROOT / SCAN_FOLDER_NAME

# --- Instrumentation & Calibration Parameters ---
POS_SENSITIVITY = 388.0    # mV / mm / nC
BEAM_CURRENT = 10.0        # uA
IMPEDANCE = 50.0           # Ohm
T_REV = 160e-9             # seconds
ASSUMED_NOISE_FLOOR = -90.0 # dBm

# Derived
BUNCH_CHARGE_NC = (BEAM_CURRENT * 1e-6) * T_REV * 1e9 # nC
