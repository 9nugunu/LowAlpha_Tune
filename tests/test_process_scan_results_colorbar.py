from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.process_scan_results as process_scan_results



def test_get_contour_color_limits_returns_shared_limits_when_enabled() -> None:
    x = np.array([[1.0, 4.0], [2.0, 3.0]])
    z = np.array([[0.5, 5.0], [1.5, 2.5]])

    vmin, vmax = process_scan_results.get_contour_color_limits(x, z, unify_individual_colorbar_range=True)

    assert vmin == 0.5
    assert vmax == 5.0



def test_get_contour_color_limits_disables_limits_when_not_enabled() -> None:
    x = np.array([[1.0, 4.0], [2.0, 3.0]])
    z = np.array([[0.5, 5.0], [1.5, 2.5]])

    vmin, vmax = process_scan_results.get_contour_color_limits(x, z, unify_individual_colorbar_range=False)

    assert vmin is None
    assert vmax is None



def test_build_contour_levels_uses_shared_linspace_when_enabled() -> None:
    x = np.array([[1.0, 4.0], [2.0, 3.0]])
    z = np.array([[0.5, 5.0], [1.5, 2.5]])

    levels = process_scan_results.build_contour_levels(x, z, unify_individual_colorbar_range=True, n_levels=5)

    assert isinstance(levels, np.ndarray)
    assert np.allclose(levels, np.linspace(0.5, 5.0, 5))



def test_build_contour_levels_keeps_default_count_when_not_enabled() -> None:
    x = np.array([[1.0, 4.0], [2.0, 3.0]])
    z = np.array([[0.5, 5.0], [1.5, 2.5]])

    levels = process_scan_results.build_contour_levels(x, z, unify_individual_colorbar_range=False, n_levels=5)

    assert levels == 5



def test_parser_accepts_unify_individual_colorbar_range_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_scan_results.py",
            "--dir",
            "scan_example",
            "--unify-individual-colorbar-range",
        ],
    )

    args = process_scan_results.get_parser()

    assert args.dir == "scan_example"
    assert args.unify_individual_colorbar_range is True
