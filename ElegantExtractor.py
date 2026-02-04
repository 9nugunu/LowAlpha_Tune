import subprocess
import os
import numpy as np
from pathlib import Path

class ElegantDataExtractor:
    def __init__(self, twiss_file, param_file, rf_element_name="RF1", watch_element_name="WISLANDP"):
        """
        twiss_file: Path to .twi file from &twiss_output
        param_file: Path to .param file
        rf_element_name: Element name for RF cavity (e.g., RF1)
        watch_element_name: Element name for Beta checking (e.g., WISLAND)
        """
        self.twiss_file = Path(twiss_file).resolve()
        self.param_file = Path(param_file).resolve()
        self.rf_name = rf_element_name
        self.watch_name = watch_element_name
        
        # Physics Constants
        self.e_charge = 1.602176634e-19
        self.c = 299792458
        self.electron_mass_eV = 0.510998950e6 

        self.E_0 = 0.0
        self.gam = 0.0
        self.U_0 = 0.0
        self.T_0 = 0.0
        self.beta_x = 0.0
        self.e_x = 0.0
        self.alphac = 0.0
        self.alphac2 = 0.0
        self.V_rf = 0.0
        self.f_rf = 0.0
        self.w_rf = 0.0

        if not self.twiss_file.exists():
            print(f"Error: Twiss file not found: {self.twiss_file}")
            return
        # param file might be optional or checked later
        
        self.extract_data()

    def _run_cmd(self, cmd_parts):
        """Run standard subprocess command and return output string."""
        # subprocess.check_output expects a string if shell=True.
        # Ensure paths are POSIX style (forward slashes) to avoid escape issues in some shells
        if isinstance(cmd_parts, list):
            cmd_str = " ".join(str(p).replace('\\', '/') for p in cmd_parts)
        else:
            cmd_str = str(cmd_parts).replace('\\', '/')
            
        try:
            # print(f"Debug: Running {cmd_str}") 
            result = subprocess.check_output(cmd_str, shell=True, stderr=subprocess.STDOUT)
            return result.decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {cmd_str}\nMSG: {e.output.decode('utf-8')}")
            return None

    def extract_data(self):
        # Use .as_posix() for all paths
        twi = f'"{self.twiss_file.as_posix()}"'
        param = f'"{self.param_file.as_posix()}"'

        # 1. Beam Energy / Gamma (pCentral)
        res = self._run_cmd(f"sdds2stream {twi} -parameter=pCentral")
        if res:
            try:
                self.gam = float(res)
                self.E_0 = self.gam * self.electron_mass_eV
            except ValueError: pass

        # 2. Radiation Loss (U0)
        res = self._run_cmd(f"sdds2stream {twi} -parameter=U0")
        if res:
            try: self.U_0 = float(res)
            except ValueError: pass

        # 3. Revolution Period (T0) from Length / c
        # Calculate Length (max of s)
        cmd_len = f"sddsprocess {twi} -pipe=out -process=s,max,Length | sdds2stream -pipe=in -parameter=Length"
        res = self._run_cmd(cmd_len)
        if res:
            try:
                length = float(res)
                self.T_0 = length / self.c
            except ValueError: pass

        # 4. Beta_x at specific element
        # Use sddsprocess to filter rows where ElementName matches, then pipe to sdds2stream
        cmd_beta = f"sddsprocess {twi} -pipe=out -match=col,ElementName={self.watch_name} | sdds2stream -pipe=in -column=betax"
        res = self._run_cmd(cmd_beta)
        if res:
            try:
                # If multiple lines, take the first value
                vals = res.split()
                if vals: self.beta_x = float(vals[0])
            except ValueError: pass

        # 5. Emittance (ex0)
        res = self._run_cmd(f"sdds2stream {twi} -parameter=ex0")
        if res:
            try: self.e_x = float(res)
            except ValueError: pass

        # 6. RF Parameters from .param file
        if self.param_file.exists():
            # Voltage (VOLT)
            cmd_vrf = f"sddsprocess {param} -pipe=out -match=col,ElementName={self.rf_name} -match=col,ElementParameter=VOLT | sdds2stream -pipe=in -column=ParameterValue"
            res = self._run_cmd(cmd_vrf)
            if res:
                try: self.V_rf = float(res.split()[0])
                except ValueError: pass
            
            # Frequency (FREQ)
            cmd_frf = f"sddsprocess {param} -pipe=out -match=col,ElementName={self.rf_name} -match=col,ElementParameter=FREQ | sdds2stream -pipe=in -column=ParameterValue"
            res = self._run_cmd(cmd_frf)
            if res:
                try: self.f_rf = float(res.split()[0])
                except ValueError: pass
        else:
            print(f"Warning: Param file not found: {self.param_file}")

        # 7. Momentum Compaction Factors (alphac, alphac2)
        # These are parameters in the .twi file (Global values), not columns
        res = self._run_cmd(f"sdds2stream {twi} -parameter=alphac")
        if res:
            try: self.alphac = float(res)
            except ValueError: pass
            
        res = self._run_cmd(f"sdds2stream {twi} -parameter=alphac2")
        if res:
            try: self.alphac2 = float(res)
            except ValueError: pass

        # Derived
        self.w_rf = 2 * np.pi * self.f_rf

    def print_summary(self):
        print(f"\n--- Extracted Parameters for {self.twiss_file.name} ---")
        print(f"E_0 (Beam Energy): {self.E_0:.4e} eV")
        print(f"Gamma: {self.gam:.4f}")
        print(f"U_0 (Energy Loss): {self.U_0:.4e} eV")
        print(f"T_0 (Revolution Period): {self.T_0:.4e} s")
        print(f"Beta_x at {self.watch_name}: {self.beta_x:.4f} m")
        print(f"Emittance_x: {self.e_x:.4e} m*rad")
        print(f"Alpha_c (Linear): {self.alphac:.4e}")
        print(f"Alpha_c2 (Quadratic): {self.alphac2:.4e}")
        print(f"RF Voltage ({self.rf_name}): {self.V_rf:.4e} V")
        print(f"RF Frequency ({self.rf_name}): {self.f_rf:.4e} Hz")

if __name__ == "__main__":
    # Automatic detection of a sample file for testing
    import glob
    
    # Try to find any _final.twi file in output directory
    search_pattern = "output/scan_alphac_pyele/**/opt_*_final.twi"
    twi_files = glob.glob(search_pattern, recursive=True)
    
    if twi_files:
        target_twi = Path(twi_files[0])
        # Try to find corresponding check.param file
        # Pattern: opt_A..._D..._final.twi -> opt_A..._D..._check.param
        target_param = target_twi.parent / target_twi.name.replace("_final.twi", "_check.param")
        
        extractor = ElegantDataExtractor(
            twiss_file=target_twi,
            param_file=target_param,
            rf_element_name="RF1",
            watch_element_name="WISLANDP"
        )
        extractor.print_summary()
    else:
        print("No sample .twi files found in output/ directory to test.")