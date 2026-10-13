"""Where the distributional assumption fails (Section 6.3 of the notes).

Double-proxy graph with linear mechanisms, the certificate with the pair
(Z, {W}) against the linear proximal bridge, exactly as in the linear-proxy
design of `iv_and_proxy.py`, but with light or lighter-tailed noise:

    Part A  Gaussian noise            n over SAMPLE_SIZES
    Part B  Student-t noise, ν d.f.   ν over DEGREES_OF_FREEDOM, n in STUDENT_SIZES

Under Gaussian noise the safeguard no longer monitors U (Example 2 of the notes),
so the certificate should plateau while the bridge converges; Student-t noise
traces the transition between the regularly varying regime and that limit.

    uv run experiments/distributional_failure.py
    uv run experiments/distributional_failure.py --plot-only
    uv run experiments/distributional_failure.py --smoke

The figure is written as a PDF preview and as the pgfplots fragment
`distributional_failure.tex` that the notes input (copied to notes/Figures/experiments/).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from graphical_toolbox.generate_data.dag import GaussianDag, StudentTDag
from graphical_toolbox.models.proximal import LinearProximal
from plotly.subplots import make_subplots
from tqdm import tqdm

from certification_estimators.classical_graphs import PROXY
from iv_and_proxy import (
    BRIDGE, CERTIFIED, FAILURE_TOLERANCE, N_REPETITIONS, OUTCOME, SAMPLE_SIZES, SEED,
    TREATMENT, TRUE_BETA, as_float, certificate,
)

DEGREES_OF_FREEDOM = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 30.0]
STUDENT_SIZES = [10_000, 100_000, 1_000_000]
OUTPUT = Path("experiments/distributional_failure")

ESTIMATORS = {
    BRIDGE: lambda: LinearProximal(TREATMENT, OUTCOME, "Z", "W"),
    CERTIFIED: lambda: certificate("Z", ["W"]),
}


def run_repetition(
    repetition: int,
    gaussian_sizes: list[int] = SAMPLE_SIZES,
    student_sizes: list[int] = STUDENT_SIZES,
    degrees_of_freedom: list[float] = DEGREES_OF_FREEDOM,
) -> list[dict]:
    rng = np.random.default_rng([SEED, 2, repetition])
    coefficients = (
        GaussianDag.from_dag(PROXY)
        .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
        .set_edge_strength(TREATMENT, OUTCOME, TRUE_BETA)
    )
    student = StudentTDag(coefficients.G, list(coefficients.hidden_nodes))  # same coefficients

    def record(noise, df, n, data):
        return [{
            "noise": noise, "df": df, "sample_size": n, "repetition": repetition,
            "estimator": label, "estimate": as_float(make().fit_or_null(data)),
        } for label, make in ESTIMATORS.items()]

    rows = []
    for n in gaussian_sizes:
        rows += record("gaussian", None, n, coefficients.sample(n=n, rng=rng))
    for df in degrees_of_freedom:
        for n in student_sizes:
            rows += record("student", df, n, student.sample(n=n, tail_index=df, rng=rng))
    return rows


def run(n_repetitions: int, workers: int, output: Path, **kwargs) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r, **kwargs) for r in range(n_repetitions)]
        rows = [row for f in tqdm(futures, desc="distributional_failure") for row in f.result()]
    estimates = pl.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    estimates.write_parquet(output / "estimates.parquet")
    return estimates


def summarise(estimates: pl.DataFrame) -> pl.DataFrame:
    error = pl.col("estimate") - TRUE_BETA
    return (
        estimates
        .group_by(["noise", "df", "estimator", "sample_size"])
        .agg(
            error.abs().median().alias("median_abs_error"),
            error.median().alias("median_bias"),
            (pl.col("estimate") - pl.col("estimate").median()).abs().median().alias("mad"),
            (error.abs() > FAILURE_TOLERANCE).mean().alias("failure_rate"),
            pl.col("estimate").is_null().sum().alias("failed_fits"),
            pl.len().alias("repetitions"),
        )
        .sort(["noise", "df", "estimator", "sample_size"], nulls_last=False)
    )


def figure(stats: pl.DataFrame) -> go.Figure:
    from figure_style import COLORS, apply_layout, legend_entry, line

    colors = {BRIDGE: COLORS["classical"], CERTIFIED: COLORS["certified"]}
    student_sizes = sorted(stats.filter(pl.col("noise") == "student")["sample_size"].unique())
    dashes = dict(zip(student_sizes, ["dot", "dash", "solid"]))
    symbols = dict(zip(student_sizes, ["triangle-up", "square", "circle"]))
    superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=["(a) Gaussian noise", "(b) Student-t noise"],
    )
    gaussian = stats.filter(pl.col("noise") == "gaussian")
    for label in ESTIMATORS:
        curve = gaussian.filter(pl.col("estimator") == label)
        fig.add_trace(line(curve["sample_size"], curve["median_abs_error"], colors[label]), row=1, col=1)
        for n in student_sizes:
            curve = stats.filter((pl.col("noise") == "student") & (pl.col("estimator") == label)
                                 & (pl.col("sample_size") == n))
            fig.add_trace(line(curve["df"], curve["median_abs_error"], colors[label], dashes[n], symbols[n]),
                          row=1, col=2)
        fig.add_trace(legend_entry(label, colors[label]))
    for n in student_sizes:
        mantissa, exponent = f"{n:.0e}".split("e")
        label_n = f"10{str(int(exponent)).translate(superscript)}" if mantissa == "1" else f"{n:,}"
        fig.add_trace(legend_entry(f"n = {label_n}", dash=dashes[n]))

    fig.update_xaxes(type="log", tickvals=SAMPLE_SIZES[::2], ticktext=["10⁴", "10⁵", "10⁶"],
                     title_text="sample size n", showgrid=True, minor_showgrid=False, row=1, col=1)
    nu_ticks = [1.5, 2.5, 5, 10, 30]
    fig.update_xaxes(type="log", tickvals=nu_ticks, ticktext=[f"{d:g}" for d in nu_ticks], tickangle=0,
                     title_text="degrees of freedom ν", showgrid=True, minor_showgrid=False, row=1, col=2)
    fig.update_yaxes(type="log", exponentformat="power", showgrid=True, minor_showgrid=False, dtick=1)
    fig.update_yaxes(title_text="median absolute error", row=1, col=1)
    return apply_layout(fig, width=900, height=420, legend_y=-0.22)


def tikz_figure(stats: pl.DataFrame) -> str:
    """The two panels of `figure` as a pgfplots groupplot for the notes: the
    same colours, dashes and markers, the shared legend below the panels."""
    from figure_style import COLORS
    from tikz_style import AXIS_STYLE, COLOR_DEFINITIONS, LEGEND_STYLE, color_name, coordinates, tickten

    colors = {BRIDGE: color_name(COLORS["classical"]), CERTIFIED: color_name(COLORS["certified"])}
    student_sizes = sorted(stats.filter(pl.col("noise") == "student")["sample_size"].unique())
    dashes = dict(zip(student_sizes, ["densely dotted", "dashed", "solid"]))
    marks = dict(zip(student_sizes, ["triangle*", "square*", "*"]))
    width, height, h_sep = 0.312, 0.248, 0.08  # fractions of \textwidth (0.39, 0.31, 0.10 scaled by 0.8)
    gaussian = stats.filter(pl.col("noise") == "gaussian").sort("sample_size")
    student = stats.filter(pl.col("noise") == "student").sort("df")
    left, right = [], []
    for label in ESTIMATORS:
        curve = gaussian.filter(pl.col("estimator") == label)
        left.append(f"\\addplot[{colors[label]}, mark=*, forget plot] "
                    f"coordinates {{{coordinates(curve['sample_size'], curve['median_abs_error'])}}};")
        for n in student_sizes:
            curve = student.filter((pl.col("estimator") == label) & (pl.col("sample_size") == n))
            right.append(f"\\addplot[{colors[label]}, {dashes[n]}, mark={marks[n]}, forget plot] "
                         f"coordinates {{{coordinates(curve['df'], curve['median_abs_error'])}}};")
    legend = [(label, colors[label], "solid") for label in ESTIMATORS]
    for n in student_sizes:
        exponent = np.log10(n)
        label_n = f"10^{{{int(exponent)}}}" if exponent.is_integer() else f"{n:,}".replace(",", "{,}")
        legend.append((f"$n = {label_n}$", "black", dashes[n]))
    gaussian_sizes = sorted(gaussian["sample_size"].unique().to_list())
    xtick = [n for n in gaussian_sizes if np.log10(n).is_integer()] or gaussian_sizes
    nu_ticks = ",".join(f"{d:g}" for d in [1.5, 2.5, 5, 10, 30])
    return f"""\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}\\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep={h_sep}\\textwidth}},
  scale only axis, width={width}\\textwidth, height={height}\\textwidth,
  xmode=log, ymode=log, xminorticks=false, yminorticks=false,
  xmajorgrids, ymajorgrids, {AXIS_STYLE},
  every axis plot/.append style={{line width=0.9pt, mark size=1.5pt, mark options={{solid}}}},
]
\\nextgroupplot[title={{(a) Gaussian noise}}, xtick={{{','.join(str(n) for n in xtick)}}},
  xlabel={{sample size $n$}}, ylabel={{median absolute error}},
  ytickten={{{tickten(gaussian['median_abs_error'])}}}]
{chr(10).join(left)}
\\nextgroupplot[title={{(b) Student-$t$ noise}}, xtick={{{nu_ticks}}}, xticklabels={{{nu_ticks}}},
  xlabel={{degrees of freedom $\\nu$}}, ytickten={{{tickten(student['median_abs_error'])}}},
  legend columns=5, legend style={{at={{({-h_sep / 2 / width:.4f},0)}}, anchor=north, yshift=-1cm,
  {LEGEND_STYLE}}}]
{chr(10).join(right)}
{chr(10).join(f"\\addlegendimage{{{color}, {dash}, line width=1pt}}\\addlegendentry{{{name}}}" for name, color, dash in legend)}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reps", type=int, default=N_REPETITIONS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plot_only:
        estimates = pl.read_parquet(args.output / "estimates.parquet")
    elif args.smoke:
        estimates = run(4, args.workers, args.output, gaussian_sizes=[20_000, 40_000],
                        student_sizes=[20_000, 40_000, 80_000])
    else:
        estimates = run(args.reps, args.workers, args.output)

    stats = summarise(estimates)
    stats.write_parquet(args.output / "stats.parquet")
    stats.write_csv(args.output / "stats.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=6):
        print(stats)

    from figure_style import save
    from tikz_style import write_fragment
    save(figure(stats), args.output / "distributional_failure")
    write_fragment(args.output / "distributional_failure.tex", tikz_figure(stats), "experiments")


if __name__ == "__main__":
    main()
