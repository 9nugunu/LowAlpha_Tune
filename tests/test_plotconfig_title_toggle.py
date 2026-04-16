from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.process_scan_results as process_scan_results


def test_process_scan_results_style_axis_hides_title_when_plot_config_titles_are_disabled() -> None:
    process_scan_results.config.show_title = False
    figsize = (12, 9)

    fig, ax = plt.subplots(figsize=figsize)
    process_scan_results.style_axis(ax, figsize, title="hidden")

    assert ax.get_title() == ""

    plt.close(fig)


def test_process_scan_results_style_axis_shows_title_when_plot_config_titles_are_enabled() -> None:
    process_scan_results.config.show_title = True
    figsize = (12, 9)

    fig, ax = plt.subplots(figsize=figsize)
    process_scan_results.style_axis(ax, figsize, title="shown")

    assert ax.get_title() == "shown"

    plt.close(fig)

    process_scan_results.config.show_title = False
