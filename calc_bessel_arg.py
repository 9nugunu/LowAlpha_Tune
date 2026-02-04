import numpy as np
import pandas as pd
import scipy.special as sp
import sys
import os
from pathlib import Path

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import LATEST_SCAN_DIR, T_REV

# --- Physics Constants (Copy from compare_offsets.py for standalone usage) ---
class MachineParams:
    def __init__(self):
        self.e_charge = 1.602176634e-19
        self.E_0 = 629e6                  # eV
        self.U_0 = 9.1e3                  # eV
        self.T_0 = T_REV                  # Use config REV period
        self.V_rf = 0.5e6                 # V
        self.f_rf = 500e6                 # Hz
        self.w_rf = 2 * np.pi * self.f_rf # rad/s
        
        # Derived
        self.phi_s = np.pi - np.arcsin(self.U_0 / self.V_rf)
        self.gam = self.E_0 / (0.511e6) + 1 # Gamma (not strictly needed for nu_s but good to have)

PARAMS = MachineParams()

def synchrotron_frequency(alpha, params=PARAMS):
    """Synchrotron angular frequency w_s(alpha)."""
    # w_s = sqrt( eta * h * e * V * cos(phi_s) / (2*pi*E*T_0) ) ... check formula
    # Formula used in compare_offsets:
    # return np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)
    # Note: alpha is momentum compaction. eta approx alpha for high energy.
    # The formula in compare_offsets.py:
    # np.sqrt(-V * w_rf * alpha * cos(phi)) / sqrt(E * T0)
    # = sqrt( -(V * 2pi f_rf * alpha * cos(phi)) / (E * T0) )
    # This seems correct for w_s^2 = (alpha * 2pi * f_rf * V * |cos(phi)|) / (E * T0) ? 
    # Let's trust the one in compare_offsets.py since it matches the user's setup.
    return np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s)) / np.sqrt(params.E_0 * params.T_0)

def calculate_grid():
    if not LATEST_SCAN_DIR.exists():
        print(f"Error: Scan directory not found: {LATEST_SCAN_DIR}")
        return

    x_path = LATEST_SCAN_DIR / "X.csv"
    if not x_path.exists():
        print(f"Error: X.csv not found in {LATEST_SCAN_DIR}")
        return

    print(f"Loading grid from {x_path}...")
    # Load grid definition
    # Columns are alpha * 1e4
    # Index is delta * 1e4
    df_template = pd.read_csv(x_path, index_col=0)
    
    alpha_vals_disp = np.array(df_template.columns, dtype=float)
    delta_vals_disp = np.array(df_template.index, dtype=float)
    
    # Restore physical units
    alpha_phys = alpha_vals_disp * 1e-4
    delta_phys = delta_vals_disp * 1e-4
    
    # Create meshgrids
    # We want rows = delta, cols = alpha
    # Alpha varies along columns (axis 1), Delta varies along rows (axis 0)
    Alpha, Delta = np.meshgrid(alpha_phys, delta_phys)
    
    # Calculate Synchrotron Tune (nu_s)
    # w_s = synchrotron_frequency(alpha)
    # nu_s = w_s * T_0 / (2 * pi)
    w_s_grid = synchrotron_frequency(Alpha)
    nu_s_grid = w_s_grid * PARAMS.T_0 / (2 * np.pi)
    
    # Calculate Argument: delta / nu_s
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        bessel_arg = Delta / nu_s_grid
        
    # Handle NaNs or Infs if alpha is 0 or negative (though alpha > 0 usually)
    bessel_arg[nu_s_grid == 0] = 0
    
    # Create DataFrame
    df_arg = pd.DataFrame(bessel_arg, index=delta_vals_disp, columns=alpha_vals_disp)
    
    # Save
    out_path = LATEST_SCAN_DIR / "Bessel_Arg_Grid.csv"
    df_arg.to_csv(out_path)
    print(f"Saved Bessel Argument (delta/nu_s) grid to: {out_path}")
    
    # Also save the Synchrotron Tune grid for reference?
    df_nus = pd.DataFrame(nu_s_grid, index=delta_vals_disp, columns=alpha_vals_disp)
    out_path_nus = LATEST_SCAN_DIR / "Nu_s_Grid.csv"
    df_nus.to_csv(out_path_nus)
    print(f"Saved Synchrotron Tune (nu_s) grid to: {out_path_nus}")

if __name__ == "__main__":
    calculate_grid()
