"""Coverage of the rank-based confidence interval on the certified sample
(the confidence-interval appendix of the notes).

Designs (b) and (c) of `iv_and_proxy.py` with the same Pareto noise and
coefficients, the certified sample being the k observations with the largest
source value among those with the safeguard in the ball; on it the interval
C_{k,gamma} of `certification_estimators.rank_interval` and, for reference,
the certified OLS. Design (b) is repeated with Gaussian noise, where the
safeguard does not monitor U and the interval inherits the failure of the
certificate.

    uv run experiments/rank_interval.py                 # full run
    uv run experiments/rank_interval.py --plot-only     # table from stored estimates
    uv run experiments/rank_interval.py --smoke         # tiny run into a scratch folder

Writes estimates.parquet (one row per repetition, cell, k and level),
coverage.csv and the table fragment rank_interval_sim_table.tex, which is
copied to notes/sections/Appendices/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from graphical_toolbox.generate_data.dag import Dag, GaussianDag, ParetoDag
from tqdm import tqdm

from certification_estimators.certificate import Certificate, QuantileBall
from certification_estimators.classical_graphs import INVALID_IV, PROXY
from certification_estimators.rank_interval import cutoff, rank_interval
from iv_and_proxy import OUTCOME, SAFEGUARD_QUANTILE, SEED, TREATMENT, TRUE_BETA

SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
TAIL_INDICES = [1.5, 2.5]
CERTIFIED_SIZES = [5, 10]
LEVELS = [0.1, 0.05]          # gamma; the table reports 1 - gamma = 0.9, the application's level
TABLE_LEVEL = 0.1
N_REPETITIONS = 1000
OUTPUT = Path("experiments/rank_interval")
NOTES_TABLE = Path(__file__).resolve().parents[2] / "notes" / "sections" / "Appendices" / "rank_interval_sim_table.tex"


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
    """One replication of every design and cell, seeded by the repetition index alone."""
    rng = np.random.default_rng([SEED, 3, repetition])
    rows = []
    for design in designs:
        dag = (design.dag_class.from_dag(design.graph)
               .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
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


def run(n_repetitions: int, workers: int, output: Path, **kwargs) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r, **kwargs) for r in range(n_repetitions)]
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
    parser.add_argument("--plot-only", action="store_true", help="reuse stored estimates")
    parser.add_argument("--smoke", action="store_true", help="tiny run into a scratch folder")
    parser.add_argument("--reps", type=int, default=N_REPETITIONS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plot_only:
        estimates = pl.read_parquet(args.output / "estimates.parquet")
    elif args.smoke:
        estimates = run(4, args.workers, args.output, sample_sizes=[20_000, 40_000])
    else:
        estimates = run(args.reps, args.workers, args.output)

    stats = summarise(estimates)
    stats.write_csv(args.output / "coverage.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(stats)
    body = table(stats)
    out = args.output / "rank_interval_sim_table.tex"
    out.write_text("% generated by experiments/rank_interval.py; do not edit by hand\n" + body)
    print(f"wrote {out}")
    if not args.smoke and NOTES_TABLE.parent.is_dir():
        shutil.copy(out, NOTES_TABLE)
        print(f"  copied to {NOTES_TABLE}")


if __name__ == "__main__":
    main()
