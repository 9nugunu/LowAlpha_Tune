from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import PlotConfig


def test_plotconfig_enables_true_latex_rendering_and_smoke_renders(tmp_path: Path) -> None:
    PlotConfig().apply_settings()

    assert plt.rcParams["text.usetex"] is True

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\alpha_c$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title(r"$\Delta x$ vs. $\alpha_c$")

    output_path = tmp_path / "usetex_smoke.png"
    fig.savefig(output_path, dpi=72)
    plt.close(fig)

    assert output_path.exists()


def test_plotconfig_matches_lps_pytomo_defaults_and_scaling() -> None:
    cfg = PlotConfig()

    assert cfg.label_size == 32
    assert cfg.font_size == 30
    assert cfg.title_size == 32
    assert cfg.legend_fontsize == 16
    assert cfg.line_width == 4

    assert cfg.font_scale((10, 8)) == 1.0
    assert round(cfg.font_scale((12, 9)), 6) == round((108.0 / 80.0) ** 0.5, 6)

    scaled = cfg.scaled_font_sizes((12, 9))
    assert round(scaled["tick"], 6) == round(cfg.font_size * cfg.font_scale((12, 9)) * 0.8, 6)
    assert round(scaled["label"], 6) == round(cfg.label_size * cfg.font_scale((12, 9)), 6)
    assert round(scaled["title"], 6) == round(cfg.title_size * cfg.font_scale((12, 9)), 6)
    assert round(scaled["legend"], 6) == round(cfg.legend_fontsize * cfg.font_scale((12, 9)), 6)
