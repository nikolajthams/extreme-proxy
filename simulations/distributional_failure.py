"""Failure of the distributional assumption (Section 6.3 of the paper).

Design (b) of `iv_and_proxy.py`, the linear double-proxy graph with the pair
(Z, {W}) against the linear proximal bridge, but with light or lighter-tailed
noise in place of the Pareto noise:

    Part A  standard Gaussian noise         n over SAMPLE_SIZES
    Part B  Student-t noise, nu d.f.        nu over DEGREES_OF_FREEDOM, n in STUDENT_SIZES

Under Gaussian noise the safeguard no longer monitors U, so the certified OLS
plateaus while the bridge converges; Student-t noise traces the transition
between the regularly varying regime and that limit.

    uv run python -m simulations.distributional_failure
    uv run python -m simulations.distributional_failure --plot-only

Writes to distributional_failure/: estimates.parquet, stats.csv and the
pgfplots fragment distributional_failure.tex of the figure.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from certification_estimators.baselines import LinearProximal
from certification_estimators.dag import GaussianDag, StudentTDag
from certification_estimators.graphs import PROXY
from simulations.iv_and_proxy import (
    BRIDGE, CERTIFIED, COEFFICIENT_RANGE, FAILURE_TOLERANCE, N_REPETITIONS, OUTCOME, SAMPLE_SIZES, SEED,
    TREATMENT, TRUE_BETA, certificate,
)
from tikz_style import COLOR_DEFINITIONS, COLORS, color_name, coordinates, tickten, write_fragment

DEGREES_OF_FREEDOM = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 30.0]
STUDENT_SIZES = [10_000, 100_000, 1_000_000]
OUTPUT = Path(__file__).resolve().parent / "distributional_failure"

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
    """One repetition of every cell, seeded by the repetition index alone: one set
    of coefficients, shared by the Gaussian and the Student-t samples."""
    rng = np.random.default_rng([SEED, 2, repetition])
    gaussian = (
        GaussianDag.from_dag(PROXY)
        .sample_coefficients(*COEFFICIENT_RANGE, rng=rng)
        .set_edge_strength(TREATMENT, OUTCOME, TRUE_BETA)
    )
    student = StudentTDag(gaussian.G, list(gaussian.hidden_nodes))  # the same coefficients

    def record(noise: str, df: float | None, n: int, data: pl.DataFrame) -> list[dict]:
        return [{
            "noise": noise, "df": df, "sample_size": n, "repetition": repetition,
            "estimator": label, "estimate": make().fit_or_null(data),
        } for label, make in ESTIMATORS.items()]

    rows = []
    for n in gaussian_sizes:
        rows += record("gaussian", None, n, gaussian.sample(n=n, rng=rng))
    for df in degrees_of_freedom:
        for n in student_sizes:
            rows += record("student", df, n, student.sample(n=n, rng=rng, df=df))
    return rows


def run(n_repetitions: int, workers: int, output: Path) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r) for r in range(n_repetitions)]
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


def tikz_figure(stats: pl.DataFrame) -> str:
    """Two panels: (a) the Gaussian case against n, (b) the Student-t case
    against nu with one dash and marker per n; the legend to the right of (b)."""
    colors = {BRIDGE: color_name(COLORS["classical"]), CERTIFIED: color_name(COLORS["certified"])}
    student_sizes = sorted(stats.filter(pl.col("noise") == "student")["sample_size"].unique())
    dashes = dict(zip(student_sizes, ["densely dotted", "dashed", "solid"]))
    marks = dict(zip(student_sizes, ["triangle*", "square*", "*"]))
    width, height, h_sep = 0.23, 0.15, 0.08  # fractions of \textwidth; the height is that of iv_and_proxy.tex
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
  xmajorgrids, ymajorgrids, grid style={{applight!50}}, axis line style={{applight}}, tick style={{applight}},
  tick label style={{font=\\scriptsize}}, label style={{font=\\small}},
  title style={{font=\\footnotesize, yshift=-3pt}},
  every axis plot/.append style={{line width=0.9pt, mark size=1.3pt, mark options={{solid}}}},
]
\\nextgroupplot[title={{(a) Gaussian noise}}, xtick={{{','.join(str(n) for n in xtick)}}},
  xlabel={{sample size $n$}}, ylabel={{median absolute error}},
  ytickten={{{tickten(gaussian['median_abs_error'])}}}]
{chr(10).join(left)}
\\nextgroupplot[title={{(b) Student-$t$ noise}}, xtick={{{nu_ticks}}}, xticklabels={{{nu_ticks}}},
  xlabel={{degrees of freedom $\\nu$}}, ytickten={{{tickten(student['median_abs_error'])}}},
  legend columns=1, legend style={{at={{(1.08,0.5)}}, anchor=west,
  draw=none, fill=none, font=\\footnotesize, legend cell align=left, row sep=1pt}}]
{chr(10).join(right)}
{chr(10).join(f"\\addlegendimage{{{color}, {dash}, line width=1pt}}\\addlegendentry{{{name}}}" for name, color, dash in legend)}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="reuse the stored estimates")
    parser.add_argument("--reps", type=int, default=N_REPETITIONS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plot_only:
        estimates = pl.read_parquet(args.output / "estimates.parquet")
    else:
        estimates = run(args.reps, args.workers, args.output)

    stats = summarise(estimates)
    stats.write_csv(args.output / "stats.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=6):
        print(stats)
    write_fragment(args.output / "distributional_failure.tex", tikz_figure(stats),
                   "simulations/distributional_failure.py", "Figures/experiments")


if __name__ == "__main__":
    main()
