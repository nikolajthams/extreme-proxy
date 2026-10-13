"""Precision-weighted combination of the certified estimator and the bridge
(the combining appendix of the notes).

Design (b) of `iv_and_proxy.py`, the linear double-proxy graph with Pareto
noise, the same coefficients and the same safeguard ball. The threshold v_n is
the (K + 1)th largest source value Z among the observations with W in the ball,
the smallest threshold at which exactly K observations are certified. Three
estimators of beta_0:

    certified   least squares of Y on T with an intercept over the K certified
                observations {Z > v_n, W in ball}
    bridge      the linear proximal bridge of `iv_and_proxy.py`, two-stage least
                squares of Y on (1, T, W) with instruments (1, T, Z), fitted to
                the body {Z < v_n}
    combined    w beta_bridge + (1 - w) beta_certified with the precision weight
                w = var_cert / (var_cert + var_bridge), both variances being
                heteroscedasticity-consistent sandwich estimates

By the conditional-independence lemma of the appendix the certified tail and
the body are independent samples given v_n, so the variance of the combination
is w^2 var_bridge + (1 - w)^2 var_cert and the precision weight minimises it.
The bridge is also fitted to all observations, for reference (stored in the
estimates and the summary, not tabulated). Per-replication estimates are
stored, so the table can be recomputed without rerunning (`--plot-only`).

    uv run experiments/combine_estimators.py              # full run
    uv run experiments/combine_estimators.py --plot-only  # table from stored estimates
    uv run experiments/combine_estimators.py --smoke      # tiny run into a scratch folder

Writes estimates.parquet, summary.csv and the table fragment
combine_estimators_sim_table.tex, which is copied to notes/sections/Appendices/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from graphical_toolbox.generate_data.dag import ParetoDag
from tqdm import tqdm

from certification_estimators.certificate import Certificate, QuantileBall
from certification_estimators.classical_graphs import PROXY
from iv_and_proxy import OUTCOME, SAFEGUARD_QUANTILE, SEED, TREATMENT, TRUE_BETA

SOURCE, SAFEGUARD = "Z", ["W"]  # the pair (E, S) = (Z, {W}) of design (b)
SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
TAIL_INDICES = [1.5, 2.5]
N_CERTIFIED = 10
N_REPETITIONS = 1000
OUTPUT = Path("experiments/combine_estimators")
NOTES_TABLE = (Path(__file__).resolve().parents[2] / "notes" / "sections" / "Appendices"
               / "combine_estimators_sim_table.tex")

CERTIFIED = "certified OLS with intercept"
BRIDGE_BODY = "bridge on body"
COMBINED = "precision-weighted combination"
BRIDGE_ALL = "bridge on all observations"  # reference only, not in the table
COLUMNS = {"certified": CERTIFIED, "bridge_body": BRIDGE_BODY, "combined": COMBINED, "bridge_all": BRIDGE_ALL}
TABULATED = [CERTIFIED, BRIDGE_BODY, COMBINED]


def split_sample(data: pl.DataFrame, k: int = N_CERTIFIED) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """(certified tail, body, v_n): the k observations with the largest |source| among
    those with the safeguard in the ball, the observations with |source| below the
    threshold, and the threshold v_n itself, the (k + 1)th largest |source| in the ball.
    The absolute value follows `Certificate`; in design (b) every variable is positive."""
    ball = QuantileBall(SAFEGUARD, SAFEGUARD_QUANTILE)
    top = Certificate(SOURCE, ball).top_k_certified_subset(data, k + 1)
    if top.height < k + 1:
        raise ValueError(f"fewer than {k + 1} observations in the safeguard ball")
    v = float(top[SOURCE].abs()[k])
    certified = top.head(k)
    body = data.filter(pl.col(SOURCE).abs() < v)
    return certified, body, v


def ols_with_intercept(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of y on t with an intercept, and its HC1 sandwich variance."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    k = len(t)
    tc, yc = t - t.mean(), y - y.mean()
    stt = float(np.sum(tc * tc))
    beta = float(np.sum(tc * yc) / stt)
    resid = yc - beta * tc
    var = float(np.sum(tc * tc * resid * resid) / stt**2) * k / (k - 2)
    return beta, var


def proximal_bridge(data: pl.DataFrame) -> tuple[float, float]:
    """The linear proximal bridge with an intercept, as two-stage least squares of Y on
    (1, T, W) with instruments (1, T, Z), and the HC0 sandwich variance of the coefficient
    of T. The point estimate is that of `LinearProximal(T, Y, Z, W).fit(data)` with its
    default demeaning (tests/test_combine_estimators.py)."""
    t, y, z, w = (data[c].to_numpy().astype(float) for c in (TREATMENT, OUTCOME, SOURCE, SAFEGUARD[0]))
    one = np.ones_like(t)
    instruments = np.column_stack([one, t, z])
    regressors = np.column_stack([one, t, w])
    first_stage, *_ = np.linalg.lstsq(instruments, regressors, rcond=None)
    fitted = instruments @ first_stage
    theta, *_ = np.linalg.lstsq(fitted, y, rcond=None)
    resid = y - regressors @ theta
    bread = np.linalg.pinv(fitted.T @ fitted)
    scores = fitted * resid[:, None]
    var = bread @ (scores.T @ scores) @ bread
    return float(theta[1]), float(var[1, 1])


def precision_weight(var_certified: float, var_bulk: float) -> float:
    """The weight on the bulk estimator that minimises w^2 var_bulk + (1 - w)^2 var_certified."""
    return var_certified / (var_certified + var_bulk)


def run_repetition(repetition: int, sample_sizes: list[int] = SAMPLE_SIZES,
                   tail_indices: list[float] = TAIL_INDICES, k: int = N_CERTIFIED) -> list[dict]:
    """One replication of every cell, seeded by the repetition index alone: fresh
    coefficients, then one sample per (tail index, n)."""
    rng = np.random.default_rng([SEED, 4, repetition])
    dag = (ParetoDag.from_dag(PROXY)
           .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
           .set_edge_strength(TREATMENT, OUTCOME, TRUE_BETA))
    rows = []
    for tail_index in tail_indices:
        for n in sample_sizes:
            data = dag.sample(n=n, tail_index=tail_index, rng=rng)
            certified, body, v = split_sample(data, k)
            beta_cert, var_cert = ols_with_intercept(certified[TREATMENT].to_numpy(), certified[OUTCOME].to_numpy())
            beta_bridge, var_bridge = proximal_bridge(body)
            beta_all, _ = proximal_bridge(data)
            w = precision_weight(var_cert, var_bridge)
            rows.append({
                "repetition": repetition, "tail_index": tail_index, "sample_size": n, "k": k,
                "threshold": v, "body_size": body.height,
                "certified": beta_cert, "bridge_body": beta_bridge,
                "combined": w * beta_bridge + (1 - w) * beta_cert, "bridge_all": beta_all,
                "weight": w, "var_certified": var_cert, "var_bridge": var_bridge,
            })
    return rows


def run(n_repetitions: int, workers: int, output: Path, **kwargs) -> pl.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_repetition, r, **kwargs) for r in range(n_repetitions)]
        rows = [row for f in tqdm(futures, desc="combine_estimators") for row in f.result()]
    estimates = pl.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    estimates.write_parquet(output / "estimates.parquet")
    return estimates


def summarise(estimates: pl.DataFrame) -> pl.DataFrame:
    """Per cell and estimator: median and 0.9-quantile of |beta_hat - beta_0|, root mean
    squared error, and the median precision weight on the bridge in that cell."""
    rows = []
    cells = estimates.select(["tail_index", "sample_size"]).unique().sort(["tail_index", "sample_size"])
    for tail_index, n in cells.iter_rows():
        group = estimates.filter((pl.col("tail_index") == tail_index) & (pl.col("sample_size") == n))
        weight = float(group["weight"].median())
        for column, label in COLUMNS.items():
            err = (group[column] - TRUE_BETA).abs().to_numpy()
            rows.append({
                "tail_index": tail_index, "sample_size": n, "estimator": label,
                "median_abs_error": float(np.median(err)), "q90_abs_error": float(np.quantile(err, 0.9)),
                "rmse": float(np.sqrt(np.mean(err**2))), "median_weight": weight, "repetitions": len(err),
            })
    return pl.DataFrame(rows)


def _num(x: float) -> str:
    """Two significant digits, trailing zeros kept."""
    return f"{x:#.2g}"


def table(stats: pl.DataFrame) -> str:
    """Median absolute error with the 0.9-quantile in parentheses against n, one block per
    tail index with the three estimators and the median weight on the bridge."""
    sizes = sorted(stats["sample_size"].unique().to_list())
    head = " & ".join(f"$n = 10^{{{int(round(np.log10(n)))}}}$" for n in sizes)
    lines = ["\\begin{tabular}{ll" + "c" * len(sizes) + "}", "\\hline",
             f"$\\alpha$ & Estimator & {head} \\\\", "\\hline"]
    for tail_index in sorted(stats["tail_index"].unique().to_list()):
        first = True
        for label in TABULATED:
            rows = stats.filter((pl.col("tail_index") == tail_index) & (pl.col("estimator") == label)).sort("sample_size")
            cells = " & ".join(f"{_num(r['median_abs_error'])} ({_num(r['q90_abs_error'])})" for r in rows.to_dicts())
            lines.append(f"{tail_index if first else ''} & {label} & {cells} \\\\")
            first = False
        rows = stats.filter((pl.col("tail_index") == tail_index) & (pl.col("estimator") == COMBINED)).sort("sample_size")
        cells = " & ".join(f"{r['median_weight']:.2f}" for r in rows.to_dicts())
        lines.append(f" & median weight on the bridge & {cells} \\\\")
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
    stats.write_csv(args.output / "summary.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(stats)
    body = table(stats)
    out = args.output / "combine_estimators_sim_table.tex"
    out.write_text("% generated by experiments/combine_estimators.py; do not edit by hand\n" + body)
    print(f"wrote {out}")
    if not args.smoke and NOTES_TABLE.parent.is_dir():
        shutil.copy(out, NOTES_TABLE)
        print(f"  copied to {NOTES_TABLE}")


if __name__ == "__main__":
    main()
