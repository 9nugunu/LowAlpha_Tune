"""MLS machine parameters and beam physics functions.

Single source of truth for MachineParams and synchrotron/offset calculations.
"""

import numpy as np
import scipy.special as sp

from src.config import T_REV, POS_SENSITIVITY, BUNCH_CHARGE_NC, IMPEDANCE


class MachineParams:
    """MLS low-alpha machine parameters."""

    def __init__(self) -> None:
        # Constants
        self.e_charge = 1.602176634e-19       # C
        self.c = 299792458                    # m/s
        self.E_0 = 629e6                      # eV
        self.U_0 = 9.1e3                      # eV
        self.T_0 = T_REV                      # s (from config)
        self.gam = self.E_0 / 0.511e6 + 1

        # Lattice / RF
        self.beta_x = 7.08                    # m
        self.V_rf = 0.5e6                     # V
        self.f_rf = 500e6                     # Hz
        self.w_rf = 2 * np.pi * self.f_rf     # rad/s
        self.e_x = 190e-9                     # m * rad

        # Higher-order momentum compaction (from ELEGANT)
        self.alphac2 = 6.15e-2                # Quadratic term
        self.alphac3 = -9.77                  # Cubic term

        # Derived
        self.phi_s = np.pi - np.arcsin(self.U_0 / self.V_rf)


PARAMS = MachineParams()


def synchrotron_frequency(alpha: np.ndarray, params: MachineParams = PARAMS) -> np.ndarray:
    """Synchrotron angular frequency w_s(alpha). Linear approximation."""
    return (
        np.sqrt(-params.V_rf * params.w_rf * alpha * np.cos(params.phi_s))
        / np.sqrt(params.E_0 * params.T_0)
    )


def synchrotron_frequency_higher_order(
    alpha: np.ndarray,
    delta: np.ndarray,
    order: int = 2,
    params: MachineParams = PARAMS,
) -> np.ndarray:
    """Synchrotron angular frequency with higher-order momentum compaction.

    Parameters
    ----------
    alpha : array_like
        First-order momentum compaction factor.
    delta : array_like
        Relative energy deviation.
    order : int
        1 = linear, 2 = quadratic (+alpha2*delta), 3 = cubic (+alpha3*delta^2).
    params : MachineParams
        Machine parameters.
    """
    if order == 1:
        alpha_eff = alpha
    elif order == 2:
        alpha_eff = alpha + params.alphac2 * delta
    else:
        alpha_eff = alpha + params.alphac2 * delta + params.alphac3 * delta**2

    alpha_eff = np.maximum(alpha_eff, 1e-10)
    return (
        np.sqrt(-params.V_rf * params.w_rf * alpha_eff * np.cos(params.phi_s))
        / np.sqrt(params.E_0 * params.T_0)
    )


def x_offset(alpha: np.ndarray, delta: np.ndarray, params: MachineParams = PARAMS) -> np.ndarray:
    """Transverse offset via Bessel J1. Factor of 2 for both sidebands (vx +/- vz)."""
    w_s = synchrotron_frequency(alpha, params)
    mu_s = w_s * params.T_0 / (2 * np.pi)
    return np.sqrt(params.e_x * params.beta_x) * 2 * sp.j1(delta / mu_s)


def z_offset(alpha: np.ndarray, delta: np.ndarray, params: MachineParams = PARAMS) -> np.ndarray:
    """Longitudinal offset from dispersion relation."""
    w_s = synchrotron_frequency(alpha, params)
    return (alpha - 1 / params.gam**2) * params.c / w_s * delta


def offset_to_power_dbm(offset_um: np.ndarray) -> np.ndarray:
    """Convert offset in micrometers to power in dBm."""
    v_peak_mv = POS_SENSITIVITY * (offset_um / 1000.0) * BUNCH_CHARGE_NC
    p_mw = ((v_peak_mv / 1000.0) ** 2) / (2 * IMPEDANCE) * 1000.0
    with np.errstate(divide="ignore"):
        return 10 * np.log10(p_mw)
