"""Create ablation plots for the bird-song VI-PLL experiment."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
MAX_WIDTH_PT = 480.0
PT_PER_INCH = 72.27
PLOT_BOX_WIDTH_PT = 180.0
PLOT_BOX_HEIGHT_PT = 122.0
LEFT_MARGIN_WITH_LABEL_PT = 35.0
LEFT_MARGIN_NO_LABEL_PT = 28.0
RIGHT_MARGIN_PT = 10.0
BOTTOM_MARGIN_PT = 32.0
TOP_MARGIN_PT = 8.0
RBF_GAMMA = 1000.0
CSV_COLUMNS = [
    "dataset",
    "algorithm",
    "seed",
    "train_acc",
    "test_acc",
    "train_mcc",
    "test_mcc",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel PDF for the bird-song VI-PLL ablation study."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for abl_plot1.pdf and abl_plot2.pdf.",
    )
    return parser.parse_args()


def collect_series(pattern: str, regex: str) -> pd.DataFrame:
    """Collect mean/std test accuracy for a filename pattern."""

    rows: list[dict[str, float]] = []
    compiled = re.compile(regex)

    for csv_path in sorted(RESULTS_DIR.glob(pattern)):
        match = compiled.fullmatch(csv_path.name)
        if match is None:
            continue

        sweep_value = float(match.group(1))
        frame = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
        rows.append(
            {
                "x": sweep_value,
                "mean": float(frame["test_acc"].mean()),
                "std": float(frame["test_acc"].std(ddof=1)),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No result files matched pattern {pattern!r} in {RESULTS_DIR}."
        )

    return pd.DataFrame(rows).sort_values("x", kind="stable").reset_index(
        drop=True
    )


def plot_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    xlabel: str,
    color: str,
    ylabel: str | None,
    show_y_ticks: bool,
) -> None:
    """Draw line plot with a standard-deviation band."""

    x = normalize_x(frame["x"])
    mean = smooth_series(frame["mean"])
    std = smooth_series(frame["std"])

    ax.plot(x, mean, color=color, marker="o", linewidth=1.6, markersize=3.0)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.2,
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
    )
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        ax.yaxis.labelpad = 1.5
    ax.xaxis.labelpad = 4.0
    ax.set_axisbelow(True)
    ax.grid(True, color="#e6e6e6", linewidth=0.6, alpha=1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.xaxis.set_major_formatter(tex_number_formatter_x)
    ax.yaxis.set_major_formatter(tex_number_formatter_y)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.7, 0.8)
    ax.tick_params(axis="y", labelleft=show_y_ticks)


def normalize_x(series: pd.Series) -> np.ndarray:
    """Normalize x-values to the [0, 1] interval."""

    values = series.to_numpy(dtype=float)
    x_min = values.min()
    x_max = values.max()
    if np.isclose(x_min, x_max):
        return np.zeros_like(values)
    return (values - x_min) / (x_max - x_min)


def smooth_series(series: pd.Series) -> np.ndarray:
    """Apply RBF-kernel smoothing over the normalized x positions."""

    values = series.to_numpy(dtype=float)
    positions = np.linspace(0.0, 1.0, len(values), dtype=float)
    sq_dist = (positions[:, None] - positions[None, :]) ** 2
    weights = np.exp(-RBF_GAMMA * sq_dist)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values


def tex_number_formatter_x(value: float, _: int) -> str:
    """Format x-axis ticks through TeX to match paper fonts."""

    return rf"\textrm{{{value:.1f}}}"


def tex_number_formatter_y(value: float, _: int) -> str:
    """Format y-axis ticks through TeX to match paper fonts."""

    return rf"\textrm{{{value:.2f}}}"


def set_icml_style() -> None:
    """Configure matplotlib to approximate the ICML paper template."""

    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{times}",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def create_figure(include_ylabel: bool) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with a fixed-size plot box."""

    left_margin_pt = (
        LEFT_MARGIN_WITH_LABEL_PT if include_ylabel else LEFT_MARGIN_NO_LABEL_PT
    )
    figure_width_pt = left_margin_pt + PLOT_BOX_WIDTH_PT + RIGHT_MARGIN_PT
    figure_height_pt = BOTTOM_MARGIN_PT + PLOT_BOX_HEIGHT_PT + TOP_MARGIN_PT
    figure_width_in = figure_width_pt / PT_PER_INCH
    figure_height_in = figure_height_pt / PT_PER_INCH

    fig = plt.figure(figsize=(figure_width_in, figure_height_in))
    ax = fig.add_axes(
        [
            left_margin_pt / figure_width_pt,
            BOTTOM_MARGIN_PT / figure_height_pt,
            PLOT_BOX_WIDTH_PT / figure_width_pt,
            PLOT_BOX_HEIGHT_PT / figure_height_pt,
        ]
    )
    return fig, ax


def save_plot(
    frame: pd.DataFrame,
    xlabel: str,
    color: str,
    output_path: Path,
    ylabel: str | None,
) -> None:
    """Render one ablation plot to a PDF."""

    fig, ax = create_figure(include_ylabel=ylabel is not None)
    plot_series(
        ax,
        frame,
        xlabel=xlabel,
        color=color,
        ylabel=ylabel,
        show_y_ticks=True,
    )
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main() -> None:
    """Create the ablation PDFs."""

    set_icml_style()
    beta_frame = collect_series(
        "bird-song_vi-pll_l*_p5.000.csv",
        r"bird-song_vi-pll_l([0-9]+\.[0-9]{3})_p5\.000\.csv",
    )
    delta_frame = collect_series(
        "bird-song_vi-pll_l2.500_p*.csv",
        r"bird-song_vi-pll_l2\.500_p([0-9]+\.[0-9]{3})\.csv",
    )

    output_dir = parse_args().output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path1 = output_dir / "abl_plot1.pdf"
    output_path2 = output_dir / "abl_plot2.pdf"

    save_plot(
        beta_frame,
        xlabel=r"Regularization parameter $\beta$",
        color="tab:blue",
        output_path=output_path1,
        ylabel=r"\textrm{Test-set accuracy}",
    )
    save_plot(
        delta_frame,
        xlabel=r"Prior weight $\delta$",
        color="tab:orange",
        output_path=output_path2,
        ylabel=None,
    )
    print(f"Saved ablation plots to {output_path1} and {output_path2}")


if __name__ == "__main__":
    main()
