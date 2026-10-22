"""Certified OLS against instruments and proxies (Section 6.2 and the
simulation appendix of the paper).

Four designs on the instrumental-variable and double-proxy graphs, arranged
so that the classical method's assumption holds in (a) and (b) and fails in
(c) and (d):

    valid_iv         (a) IV graph                      2SLS consistent
    linear_proxy     (b) double-proxy graph, linear    linear proximal bridge consistent
    invalid_iv       (c) IV graph with U -> Z          2SLS inconsistent
    nonlinear_proxy  (d) double-proxy graph, squared   linear proximal bridge inconsistent

In each design the certified estimator uses the source-safeguard pair the
graphical criterion of the paper selects: (Z, {}) in the valid-IV graph,
(T, {Z}) in the invalid-IV graph and (Z, {W}) in the double-proxy graph. It
is the certified OLS, least squares of Y on T through the origin over the
N_CERTIFIED observations with the largest source value among those with the
safeguard in the ball, whose radius is the SAFEGUARD_QUANTILE quantile of the
safeguard's l1 norm. The number of certified observations is fixed rather
than a quantile of the source, so that the certification threshold grows like
n^{1/alpha} and the estimator converges at that rate. Design (c) additionally
runs the pair (Z, {}) that the criterion disqualifies there, and design (d)
the correctly specified quadratic bridge, computed in exact arithmetic
(`exact_iv.py`) and, for reference, in double precision.

Every repetition draws fresh coefficients, then samples each (tail index, n)
cell independently; a repetition is seeded by its index alone. The
per-repetition estimates are stored, so the summary and the figure can be
recomputed without rerunning.

    uv run python -m simulations.iv_and_proxy                # full run
    uv run python -m simulations.iv_and_proxy --plot-only    # figure and tables from stored estimates

Writes to iv_and_proxy/: estimates.parquet (one row per repetition, cell and
estimator), stats.csv (median absolute error, median bias, failure rate and
more per cell), slopes.csv (the fitted slopes of the appendix table) and the
pgfplots fragment iv_and_proxy.tex of the figure.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from tqdm import tqdm

from certification_estimators.baselines import LinearCIV, LinearProximal
from certification_estimators.certificate import QuantileBall, Unrestricted
from certification_estimators.dag import Dag, NonlinearParetoDag, ParetoDag
from certification_estimators.estimators import CertifiedEstimator
from certification_estimators.graphs import INVALID_IV, IV, PROXY
from simulations.exact_iv import ExactQuadraticProximal
from tikz_style import AXIS_STYLE, COLOR_DEFINITIONS, COLORS, LEGEND_STYLE, color_name, coordinates, tickten, write_fragment

TRUE_BETA = 1.0
SAMPLE_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
TAIL_INDICES = [1.5, 2.5]
N_REPETITIONS = 1000
N_CERTIFIED = 3  # certified observations per fit; the source threshold then grows like n^{1/alpha}
SAFEGUARD_QUANTILE = 0.1
COEFFICIENT_RANGE = (0.2, 1.0)  # every structural coefficient other than beta_0
FAILURE_TOLERANCE = 0.1  # an estimate further than this from beta_0 counts as a failure
SEED = 1
OUTPUT = Path(__file__).resolve().parent / "iv_and_proxy"

TREATMENT, OUTCOME = "X", "Y"  # the paper writes T for the treatment; the graphs use X

CERTIFIED = "certified OLS"
DISQUALIFIED = "certified OLS, disqualified pair"
TSLS = "2SLS"
BRIDGE = "proximal bridge"
BRIDGE_REPAIRED = "proximal bridge, correctly specified"
BRIDGE_REPAIRED_FLOAT64 = "proximal bridge, correctly specified (double precision)"


def certificate(source: str, safeguard: list[str]) -> CertifiedEstimator:
    """The certified OLS of the pair (source, safeguard) on N_CERTIFIED observations."""
    ball = QuantileBall(safeguard, SAFEGUARD_QUANTILE) if safeguard else Unrestricted()
    return CertifiedEstimator(source, ball, TREATMENT, OUTCOME, N_CERTIFIED, "ols")


class QuadraticProximalFloat64:
    """The correctly specified bridge as `LinearProximal` computes it (QR least
    squares in double precision) on data augmented with Z^2 and W^2; stored to
    document where the floating-point solve breaks down, not plotted."""

    def fit_or_null(self, data: pl.DataFrame) -> float | None:
        augmented = data.with_columns((pl.col("W") ** 2).alias("W2"), (pl.col("Z") ** 2).alias("Z2"))
        return LinearProximal(TREATMENT, OUTCOME, ["Z", "Z2"], ["W", "W2"]).fit_or_null(augmented)


@dataclass(frozen=True)
class Design:
    key: str
    title: str
    graph: Dag
    dag_class: type[Dag]
    estimators: dict[str, Callable[[], object]]
    classical: str  # label of the classical competitor, for colouring


# The order of DESIGNS is the sampling order of a repetition and fixes its random
# stream; PANEL_ORDER below is the order of the figure.
DESIGNS: list[Design] = [
    Design(
        "valid_iv", "(a) Valid instrument", IV, ParetoDag,
        {TSLS: lambda: LinearCIV(TREATMENT, OUTCOME, "Z"),
         CERTIFIED: lambda: certificate("Z", [])},
        classical=TSLS,
    ),
    Design(
        "invalid_iv", "(c) Invalid instrument", INVALID_IV, ParetoDag,
        {TSLS: lambda: LinearCIV(TREATMENT, OUTCOME, "Z"),
         CERTIFIED: lambda: certificate(TREATMENT, ["Z"]),
         DISQUALIFIED: lambda: certificate("Z", [])},
        classical=TSLS,
    ),
    Design(
        "linear_proxy", "(b) Double proxy, linear", PROXY, ParetoDag,
        {BRIDGE: lambda: LinearProximal(TREATMENT, OUTCOME, "Z", "W"),
         CERTIFIED: lambda: certificate("Z", ["W"])},
        classical=BRIDGE,
    ),
    Design(
        "nonlinear_proxy", "(d) Double proxy, nonlinear", PROXY, NonlinearParetoDag,
        {BRIDGE: lambda: LinearProximal(TREATMENT, OUTCOME, "Z", "W"),
         CERTIFIED: lambda: certificate("Z", ["W"]),
         BRIDGE_REPAIRED: lambda: ExactQuadraticProximal(TREATMENT, OUTCOME, "Z", "W"),
         BRIDGE_REPAIRED_FLOAT64: QuadraticProximalFloat64},
        classical=BRIDGE,
    ),
]
NOT_PLOTTED = {BRIDGE_REPAIRED_FLOAT64}
PANEL_ORDER = ["valid_iv", "linear_proxy", "invalid_iv", "nonlinear_proxy"]
PANELS = [next(d for d in DESIGNS if d.key == key) for key in PANEL_ORDER]


def run_repetition(
    repetition: int,
    sample_sizes: list[int] = SAMPLE_SIZES,
    tail_indices: list[float] = TAIL_INDICES,
    designs: list[Design] = DESIGNS,
) -> list[dict]:
    """One repetition of every design and cell, seeded by the repetition index alone."""
    rng = np.random.default_rng([SEED, repetition])
    rows = []
    for design in designs:
        dag = (
            design.dag_class
            .from_dag(design.graph)
            .sample_coefficients(*COEFFICIENT_RANGE, rng=rng)
            .set_edge_strength(TREATMENT, OUTCOME, TRUE_BETA)
        )
        for tail_index in tail_indices:
            for n in sample_sizes:
                data = dag.sample(n=n, tail_index=tail_index, rng=rng)
                for label, make in design.estimators.items():
                    rows.append({
                        "design": design.key,
                        "repetition": repetition,
                        "tail_index": tail_index,
                        "sample_size": n,
                        "estimator": label,
                        "estimate": make().fit_or_null(data),
                    })
    return rows


def run(n_repetitions: int, workers: int, output: Path) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r) for r in range(n_repetitions)]
        rows = [row for f in tqdm(futures, desc="iv_and_proxy") for row in f.result()]
    estimates = pl.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    estimates.write_parquet(output / "estimates.parquet")
    return estimates


def summarise(estimates: pl.DataFrame) -> pl.DataFrame:
    error = pl.col("estimate") - TRUE_BETA
    return (
        estimates
        .group_by(["design", "estimator", "tail_index", "sample_size"])
        .agg(
            error.abs().median().alias("median_abs_error"),
            error.median().alias("median_bias"),
            (pl.col("estimate") - pl.col("estimate").median()).abs().median().alias("mad"),
            (error.abs() > FAILURE_TOLERANCE).mean().alias("failure_rate"),
            pl.col("estimate").is_null().sum().alias("failed_fits"),
            pl.len().alias("repetitions"),
        )
        .sort(["design", "estimator", "tail_index", "sample_size"])
    )


def loglog_slopes(stats: pl.DataFrame) -> pl.DataFrame:
    """Least-squares slope of log(median absolute error) on log(n), per curve,
    next to the slope -1/alpha of the paper's rate (-2/alpha in design (d))."""
    rows = []
    for (design, estimator, tail_index), group in stats.group_by(
        ["design", "estimator", "tail_index"], maintain_order=True
    ):
        x = np.log(group["sample_size"].to_numpy())
        y = np.log(group["median_abs_error"].to_numpy())
        rows.append({"design": design, "estimator": estimator, "tail_index": tail_index,
                     "slope": np.polyfit(x, y, 1)[0], "minus_one_over_alpha": -1 / tail_index})
    return pl.DataFrame(rows).sort(["design", "estimator", "tail_index"])


def tikz_figure(stats: pl.DataFrame) -> str:
    """One row of four panels, median absolute error against n on log-log axes;
    one colour per estimator, one dash and marker per tail index, the shared
    legend centred below the row."""
    dashes = {1.5: "solid", 2.5: "dashed"}
    marks = {1.5: "*", 2.5: "square*"}
    width, height, h_sep = 0.17, 0.15, 0.055  # fractions of \textwidth
    sizes = sorted(stats["sample_size"].unique().to_list())
    xtick = [n for n in sizes if np.log10(n).is_integer()] or sizes
    panels = []
    for i, design in enumerate(PANELS):
        plots, errors = [], []
        for label in design.estimators:
            if label in NOT_PLOTTED:
                continue
            role = ("classical" if label == design.classical
                    else "disqualified" if label == DISQUALIFIED
                    else "repaired" if label == BRIDGE_REPAIRED else "certified")
            for tail_index in TAIL_INDICES:
                curve = stats.filter(
                    (pl.col("design") == design.key) & (pl.col("estimator") == label)
                    & (pl.col("tail_index") == tail_index)
                ).sort("sample_size")
                if curve.height == 0:
                    continue
                errors.extend(curve["median_abs_error"].to_list())
                plots.append(f"\\addplot[{color_name(COLORS[role])}, {dashes[tail_index]}, mark={marks[tail_index]}, forget plot] "
                             f"coordinates {{{coordinates(curve['sample_size'], curve['median_abs_error'])}}};")
        options = [f"title={{{design.title}}}", f"ytickten={{{tickten(errors)}}}"]
        if i == len(PANELS) - 1:
            legend = [(f"{TSLS} / {BRIDGE}", "apporange", "solid"), (CERTIFIED, "appblue", "solid"),
                      (DISQUALIFIED, "appgreen", "solid"), (BRIDGE_REPAIRED, "appviolet", "solid")]
            legend += [(f"$\\alpha = {tail_index}$", "black", dashes[tail_index]) for tail_index in TAIL_INDICES]
            # the centre of the row, in the last panel's relative axis coordinates
            centre = 0.5 - (len(PANELS) - 1) / 2 * (width + h_sep) / width
            options.append(f"legend columns=3, legend style={{at={{({centre:.4f},0)}}, anchor=north, "
                           f"yshift=-1cm,\n  {LEGEND_STYLE}}}")
            plots += [f"\\addlegendimage{{{color}, {dash}, line width=1pt}}\\addlegendentry{{{name}}}"
                      for name, color, dash in legend]
        panels.append(f"\\nextgroupplot[{', '.join(options)}]\n" + "\n".join(plots))
    return f"""\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}\\begin{{groupplot}}[
  group style={{group size={len(PANELS)} by 1, horizontal sep={h_sep}\\textwidth,
    xlabels at=edge bottom, ylabels at=edge left}},
  scale only axis, width={width}\\textwidth, height={height}\\textwidth,
  xmode=log, ymode=log, xtick={{{','.join(str(n) for n in xtick)}}}, xminorticks=false, yminorticks=false,
  xmajorgrids, ymajorgrids, {AXIS_STYLE}, tick label style={{font=\\scriptsize}}, title style={{font=\\footnotesize}},
  xlabel={{sample size $n$}}, ylabel={{median absolute error}},
  every axis plot/.append style={{line width=0.9pt, mark size=1.3pt, mark options={{solid}}}},
]
{chr(10).join(panels)}
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
    slopes = loglog_slopes(stats)
    slopes.write_csv(args.output / "slopes.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=6):
        print(stats)
        print(slopes)
    write_fragment(args.output / "iv_and_proxy.tex", tikz_figure(stats), "simulations/iv_and_proxy.py",
                   "Figures/experiments")


if __name__ == "__main__":
    main()
