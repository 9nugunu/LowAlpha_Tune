from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import PlotConfig, apply_axis_style, apply_colorbar_style


def test_plotconfig_defaults_to_titles_off() -> None:
    cfg = PlotConfig()

    assert cfg.show_title is False


def test_apply_axis_style_scales_axis_text_for_figure_size() -> None:
    cfg = PlotConfig(show_title=True)
    figsize = (12, 9)
    scaled = cfg.scaled_font_sizes(figsize)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("alpha")
    ax.set_ylabel("delta")

    apply_axis_style(ax, cfg, figsize, title="demo")

    assert round(ax.xaxis.label.get_size(), 6) == round(scaled["label"], 6)
    assert round(ax.yaxis.label.get_size(), 6) == round(scaled["label"], 6)
    assert round(ax.title.get_size(), 6) == round(scaled["title"], 6)
    assert ax.get_title() == "demo"
    assert ax.xaxis.label.get_weight() == cfg.font_weight
    assert ax.yaxis.label.get_weight() == cfg.font_weight

    plt.close(fig)


def test_apply_axis_style_clears_title_when_titles_are_disabled_by_default() -> None:
    cfg = PlotConfig()
    figsize = (12, 9)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("existing")

    apply_axis_style(ax, cfg, figsize, title="demo")

    assert ax.get_title() == ""

    plt.close(fig)


def test_apply_colorbar_style_sets_scaled_label_rotation_and_padding() -> None:
    cfg = PlotConfig()
    figsize = (12, 9)
    scaled = cfg.scaled_font_sizes(figsize)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(np.arange(4).reshape(2, 2))
    cbar = fig.colorbar(image)

    apply_colorbar_style(cbar, cfg, figsize, r"Offset ($\mu\mathrm{m}$)")

    assert cbar.ax.yaxis.label.get_text() == r"Offset ($\mu\mathrm{m}$)"
    assert round(cbar.ax.yaxis.label.get_size(), 6) == round(scaled["label"], 6)
    assert cbar.ax.yaxis.label.get_rotation() in (-90.0, 270.0)
    assert cbar.ax.yaxis.labelpad == 35

    plt.close(fig)
