"""Coverage of the rank-based confidence interval on the certified sample
(the confidence-interval appendix of the paper).

Designs (b) and (c) of `iv_and_proxy.py` with the same Pareto noise and
coefficients, the certified sample being the k observations with the largest
source value among those with the safeguard in the ball; on it the interval
C_{k,gamma} of `certification_estimators.rank_interval` and, for reference,
the certified OLS. Design (b) is repeated with Gaussian noise, where the
safeguard does not monitor U and the interval inherits the failure of the
certificate.

    uv run python -m simulations.rank_interval                # full run
    uv run python -m simulations.rank_interval --plot-only    # table from stored estimates

Writes to rank_interval/: estimates.parquet (one row per repetition, cell, k
and level), coverage.csv and the table fragment rank_interval_sim_table.tex.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from certification_estimators.certificate import Certificate, QuantileBall
from certification_estimators.dag import Dag, GaussianDag, ParetoDag
from certification_estimators.graphs import INVALID_IV, PROXY
from certification_estimators.rank_interval import rank_interval
from simulations.iv_and_proxy import COEFFICIENT_RANGE, OUTCOME, SAFEGUARD_QUANTILE, SEED, TREATMENT, TRUE_BETA
from tikz_style import write_fragment

SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
TAIL_INDICES = [1.5, 2.5]
CERTIFIED_SIZES = [5, 10]
LEVELS = [0.1, 0.05]  # gamma; the table reports gamma = 0.1, the level of the application's intervals
TABLE_LEVEL = 0.1
N_REPETITIONS = 1000
OUTPUT = Path(__file__).resolve().parent / "rank_interval"


@dataclass(frozen=True)
class Design:
    key: str
    label: str
    graph: Dag
    dag_class: type[Dag]
    source: str
    safeguard: list[str]
    tail_indices: tuple[float | None, ...]


DESIGNS = [
    Design("linear_proxy", "(b) linear double proxy", PROXY, ParetoDag, "Z", ["W"], tuple(TAIL_INDICES)),
    Design("invalid_iv", "(c) invalid instrument", INVALID_IV, ParetoDag, TREATMENT, ["Z"], tuple(TAIL_INDICES)),
    Design("linear_proxy_gaussian", "(b) with Gaussian noise", PROXY, GaussianDag, "Z", ["W"], (None,)),
]


def certified_sample(data: pl.DataFrame, source: str, safeguard: list[str], k: int) -> pl.DataFrame:
    ball = QuantileBall(safeguard, SAFEGUARD_QUANTILE)
    return Certificate(source, ball).top_k_certified_subset(data, k)


def certified_ols(sub: pl.DataFrame) -> float:
    t, y = sub[TREATMENT].to_numpy(), sub[OUTCOME].to_numpy()
    return float(np.sum(t * y) / np.sum(t * t))


def run_repetition(repetition: int, sample_sizes: list[int] = SAMPLE_SIZES,
                   designs: list[Design] = DESIGNS) -> list[dict]:
    """One repetition of every design and cell, seeded by the repetition index alone."""
    rng = np.random.default_rng([SEED, 3, repetition])
    rows = []
    for design in designs:
        dag = (design.dag_class.from_dag(design.graph)
               .sample_coefficients(*COEFFICIENT_RANGE, rng=rng)
               .set_edge_strength(TREATMENT, OUTCOME, TRUE_BETA))
        for tail_index in design.tail_indices:
            for n in sample_sizes:
                data = dag.sample(n=n, rng=rng) if tail_index is None else dag.sample(n=n, tail_index=tail_index, rng=rng)
                for k in CERTIFIED_SIZES:
                    sub = certified_sample(data, design.source, design.safeguard, k)
                    t, y = sub[TREATMENT].to_numpy(), sub[OUTCOME].to_numpy()
                    ols = certified_ols(sub)
                    for gamma in LEVELS:
                        ci = rank_interval(t, y, gamma, rng)
                        rows.append({
                            "design": design.key, "repetition": repetition, "tail_index": tail_index,
                            "sample_size": n, "k": k, "gamma": gamma,
                            "lo": ci.lo, "hi": ci.hi, "covers": ci.covers(TRUE_BETA), "length": ci.length,
                            "exact_coverage": ci.coverage, "certified_ols": ols,
                        })
    return rows


def run(n_repetitions: int, workers: int, output: Path) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r) for r in range(n_repetitions)]
        rows = [row for f in tqdm(futures, desc="rank_interval") for row in f.result()]
    estimates = pl.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    estimates.write_parquet(output / "estimates.parquet")
    return estimates


def summarise(estimates: pl.DataFrame) -> pl.DataFrame:
    return (
        estimates
        .group_by(["design", "tail_index", "k", "gamma", "sample_size"])
        .agg(
            pl.col("covers").mean().alias("coverage"),
            pl.col("exact_coverage").first(),
            pl.col("length").median().alias("median_length"),
            (pl.col("certified_ols") - TRUE_BETA).abs().median().alias("ols_median_abs_error"),
            ((pl.col("certified_ols") >= pl.col("lo")) & (pl.col("certified_ols") <= pl.col("hi"))).mean()
            .alias("ols_inside"),
            pl.len().alias("repetitions"),
        )
        .sort(["design", "tail_index", "k", "gamma", "sample_size"], nulls_last=True)
    )


def _num(x: float) -> str:
    """Two significant digits, trailing zeros kept, for the lengths."""
    return f"{x:#.2g}"


def table(stats: pl.DataFrame, gamma: float = TABLE_LEVEL) -> str:
    """Coverage (per cent) and median length of C_{k,gamma} against n, one
    row per design, tail index and k, with the exact asymptotic coverage."""
    sizes = sorted(stats["sample_size"].unique().to_list())
    head = " & ".join(f"$n = 10^{{{int(round(np.log10(n)))}}}$" for n in sizes)
    lines = ["\\begin{tabular}{llcc" + "c" * len(sizes) + "}", "\\hline",
             f"Design & $\\alpha$ & $k$ & limit & {head} \\\\", "\\hline"]
    for design in DESIGNS:
        first = True
        for tail_index in design.tail_indices:
            for k in CERTIFIED_SIZES:
                f = (pl.col("design") == design.key) & (pl.col("k") == k) & (pl.col("gamma") == gamma)
                f = f & (pl.col("tail_index").is_null() if tail_index is None else (pl.col("tail_index") == tail_index))
                rows = stats.filter(f).sort("sample_size")
                if rows.height == 0:
                    continue
                cells = " & ".join(f"{100 * r['coverage']:.1f} ({_num(r['median_length'])})" for r in rows.to_dicts())
                label = design.label if first else ""
                alpha = "--" if tail_index is None else f"{tail_index}"
                lines.append(f"{label} & {alpha} & {k} & {100 * rows['exact_coverage'][0]:.1f} & {cells} \\\\")
                first = False
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


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
    stats.write_csv(args.output / "coverage.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(stats)
    write_fragment(args.output / "rank_interval_sim_table.tex", table(stats), "simulations/rank_interval.py",
                   "sections/Appendices")


if __name__ == "__main__":
    main()
