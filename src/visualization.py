from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import texmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
TEX_CACHE_DIR = MPL_CACHE_DIR / "tex.cache"
TEX_WRAPPER_DIR = PROJECT_ROOT / "tools" / "wsl_texlive_bin"


def ensure_tex_tool_path() -> None:
    wrapper_dir = str(TEX_WRAPPER_DIR)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if wrapper_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([wrapper_dir, *path_entries]) if path_entries else wrapper_dir

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
    texmanager.TexManager._texcache = str(TEX_CACHE_DIR)


def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def apply_axis_style(ax, config: "PlotConfig", figsize: Tuple[float, float], *, title: str | None = None) -> Dict[str, float]:
    sizes = config.scaled_font_sizes(figsize)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-")
    ax.grid(which="minor", linestyle=":", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=sizes["tick"], width=2)
    plt.setp(ax.get_xticklabels(), weight="bold")
    plt.setp(ax.get_yticklabels(), weight="bold")
    ax.xaxis.label.set_size(sizes["label"])
    ax.yaxis.label.set_size(sizes["label"])
    ax.xaxis.label.set_weight(config.font_weight)
    ax.yaxis.label.set_weight(config.font_weight)
    if config.show_title:
        if title is not None:
            ax.set_title(title)
    else:
        ax.set_title("")
    ax.title.set_size(sizes["title"])
    ax.title.set_weight(config.title_weight)
    return sizes


def apply_colorbar_style(cbar, config: "PlotConfig", figsize: Tuple[float, float], label: str) -> Dict[str, float]:
    sizes = config.scaled_font_sizes(figsize)
    cbar.set_label(label, rotation=-90, labelpad=35, size=sizes["label"], weight=config.font_weight)
    cbar.ax.tick_params(labelsize=sizes["tick"], width=2)
    plt.setp(cbar.ax.get_yticklabels(), weight="bold")
    return sizes


@dataclass
class PlotConfig:


    label_size: int = 32
    label_weight: str = "bold"
    font_weight: str = "bold"
    font_size: int = 30
    font_family: str = "serif"
    cmap: str = "turbo"
    title_size: int = 32
    title_weight: str = "bold"
    show_title: bool = False
    legend_fontsize: int = 16
    line_width: int = 4
    marker_size: int = 8
    zero_transparent: bool = True
    use_tex: bool = True
    ref_area: float = 10.0 * 8.0

    def apply_settings(self) -> None:
        if self.use_tex:
            ensure_tex_tool_path()
            plt.rcParams["text.usetex"] = True
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
            plt.rcParams["text.latex.preamble"] = (
                r"\usepackage[T1]{fontenc} "
                r"\usepackage{amsmath,bm,amssymb} "
                r"\usepackage{type1cm} "
                r"\boldmath "
                r"\renewcommand{\familydefault}{\rmdefault} "
                r"\renewcommand{\seriesdefault}{b} "
                r"\AtBeginDocument{\boldmath\bfseries}"
            )
        else:
            plt.rcParams["text.usetex"] = False
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "Bitstream Vera Sans",
                "sans-serif",
            ]
            plt.rcParams["mathtext.fontset"] = "cm"

        plt.rcParams["axes.labelsize"] = self.label_size
        plt.rcParams["axes.labelweight"] = self.label_weight
        plt.rcParams["font.weight"] = self.font_weight
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["image.cmap"] = self.cmap
        plt.rcParams["axes.titlesize"] = self.title_size
        plt.rcParams["axes.titleweight"] = self.title_weight
        plt.rcParams["legend.fontsize"] = self.legend_fontsize
        plt.rcParams["xtick.labelsize"] = self.font_size * 0.8
        plt.rcParams["ytick.labelsize"] = self.font_size * 0.8
        plt.rcParams["xtick.major.width"] = 2
        plt.rcParams["ytick.major.width"] = 2
        plt.rcParams["lines.linewidth"] = self.line_width
        plt.rcParams["lines.markersize"] = self.marker_size

    def font_scale(self, figsize: Tuple[float, float]) -> float:
        return (figsize[0] * figsize[1] / self.ref_area) ** 0.5

    def scaled_font_sizes(self, figsize: Tuple[float, float], font_scale_override: float | None = None) -> Dict[str, float]:
        scale = font_scale_override if font_scale_override is not None else self.font_scale(figsize)
        return {
            "tick": self.font_size * scale * 0.8,
            "label": self.label_size * scale,
            "title": self.title_size * scale,
            "legend": self.legend_fontsize * scale,
        }
