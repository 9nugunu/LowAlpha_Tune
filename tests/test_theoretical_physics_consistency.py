from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_theory_vs_simulation import plot_combined_powers
from scripts.plot_theoretical_offsets import x_offset as theoretical_x_offset
from scripts.plot_theoretical_offsets import z_offset as theoretical_z_offset
from src.physics import x_offset as shared_x_offset
from src.physics import z_offset as shared_z_offset


def test_plot_theoretical_offsets_matches_shared_physics_module() -> None:
    alpha = np.array([[1.0e-5, 4.0e-5], [7.0e-5, 1.0e-4]])
    delta = np.array([[1.0e-4, 1.5e-4], [2.0e-4, 2.3e-4]])

    assert np.allclose(theoretical_x_offset(alpha, delta), shared_x_offset(alpha, delta), rtol=1e-3, atol=0.0)
    assert np.allclose(theoretical_z_offset(alpha, delta), shared_z_offset(alpha, delta), rtol=1e-3, atol=0.0)


def test_plot_combined_powers_smoke_saves_output(tmp_path: Path) -> None:
    alpha_axis = np.array([0.1, 0.2, 0.3])
    sim_x = np.array([1.0, 2.0, 3.0])
    sim_z = np.array([0.8, 1.6, 2.4])
    th_x = np.array([0.9, 1.8, 2.7])
    th_z = np.array([0.7, 1.4, 2.1])

    out_path = plot_combined_powers(alpha_axis, sim_x, sim_z, th_x, th_z, 2.3e-4, 2.3e-4, tmp_path)

    assert out_path.exists()
