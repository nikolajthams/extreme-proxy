"""Rank-based confidence intervals on the declustered certified days (the
confidence-interval appendix of the paper).

    uv run python -m hydro.experiments.rank_interval

Reads pairs.parquet. The certified days of a certified estimator are
declustered to one day per flood event: certified days fewer than
`Decluster.run_length_days` apart belong to the same event, which is
represented by its day of largest T. On the declustered days the certified
OLS is refitted and the interval C_{k,gamma} of
`certification_estimators.rank_interval` is computed at the level
`Estimator.interval_level` of the bootstrap intervals in the main text.

Writes
  * headline_rank_interval.parquet: the headline pair at the tabulated q_P,
    one row per method (certified OLS and the ablations V1-V3, with v and
    the certified days as in all_pairs/headline); V2 reports the medians over
    the permutations of the estimate and of the two endpoints;
  * all_pairs_rank_interval.parquet: the certified OLS of every pair.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from certification_estimators.rank_interval import rank_interval
from hydro.config import CONFIG, OUTPUT_DIR, REGION_BY_NAME
from hydro.estimation import certified_events, naive_ols, operating_point, threshold
from hydro.pairs import headline_pair, main_pairs, panel_for
from hydro.preprocess.decluster import runs_decluster
from hydro.preprocess.panel import PairSpec, build_panel

EST = CONFIG.estimator
GAMMA = 1 - EST.interval_level


def decluster_certified(cert: pl.DataFrame) -> pl.DataFrame:
    """One certified day per flood event, the day of largest T."""
    return runs_decluster(cert, "T", -np.inf, CONFIG.decluster.run_length_days)


def interval_row(cert: pl.DataFrame, gamma: float = GAMMA) -> dict:
    """Declustered certified OLS and rank interval of a certified sample."""
    d = decluster_certified(cert)
    ci = rank_interval(d["T"].to_numpy(), d["Y"].to_numpy(), gamma) if d.height >= 2 else None
    return {"n_days": cert.height, "n_cert": d.height, "beta": naive_ols(d) if d.height else None,
            "ci_lo": None if ci is None else ci.lo, "ci_hi": None if ci is None else ci.hi,
            "ci_level": None if ci is None else ci.coverage, "ci_r": None if ci is None else ci.r}


def headline_rows(row: dict, panel) -> list[dict]:
    dy, p = panel.daily, panel.p_cols
    q_p = EST.q_p
    q_t, v = operating_point(dy, p, q_p)
    base = {"pair": row["pair"], "q_p": q_p, "q_t": q_t, "v": v, "level": EST.interval_level}
    out = [base | {"method": "certified OLS"} | interval_row(certified_events(dy, p, q_p, v)),
           base | {"method": "V1 tail only"} | interval_row(certified_events(dy, [], q_p, v))]
    # V2: the permutations of headline.py, in the same order (same seed and draws)
    rng = np.random.default_rng(EST.seed)
    rows = []
    for _ in range(EST.n_permutations):
        perm = dy.drop(p).hstack(dy.select(p)[rng.permutation(dy.height)])
        r = interval_row(certified_events(perm, p, q_p, threshold(perm, p, q_p, q_t)))
        if r["beta"] is not None:
            rows.append(r)
    med = lambda key: float(np.median([r[key] for r in rows if r[key] is not None]))  # noqa: E731
    out.append(base | {"method": "V2 permuted safeguard", "n_days": med("n_days"), "n_cert": med("n_cert"),
                       "beta": med("beta"), "ci_lo": med("ci_lo"), "ci_hi": med("ci_hi"), "ci_level": med("ci_level"),
                       "ci_r": None,
                       "note": f"medians over {EST.n_permutations} permutations of the estimate and the endpoints"})
    a3 = build_panel(PairSpec(row["pair"], REGION_BY_NAME[row["region"]], row["t"], row["y"],
                              list(EST.other_basin_proxies)), distance_km=row["distance_km"])
    op3 = operating_point(a3.daily, a3.p_cols, q_p)
    out.append(base | {"method": "V3 other-basin safeguard", "q_t": op3[0], "v": op3[1]}
               | interval_row(certified_events(a3.daily, a3.p_cols, q_p, op3[1])))
    return out


def all_pairs_rows() -> list[dict]:
    out = []
    for row in main_pairs().iter_rows(named=True):
        panel = panel_for(row)
        op = operating_point(panel.daily, panel.p_cols, EST.q_p)
        if op is None:
            continue
        out.append({"pair": row["pair"], "region": row["region"], "q_p": EST.q_p, "q_t": op[0], "v": op[1],
                    "level": EST.interval_level}
                   | interval_row(certified_events(panel.daily, panel.p_cols, EST.q_p, op[1])))
    return out


def main() -> None:
    row = headline_pair()
    h = pl.DataFrame(headline_rows(row, panel_for(row)), infer_schema_length=None)
    h.write_parquet(OUTPUT_DIR / "headline_rank_interval.parquet")
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=3):
        print(h.select("method", "q_t", "v", "n_days", "n_cert", "beta", "ci_lo", "ci_hi", "ci_level"))
    a = pl.DataFrame(all_pairs_rows(), infer_schema_length=None)
    a.write_parquet(OUTPUT_DIR / "all_pairs_rank_interval.parquet")
    defined = a.filter(pl.col("ci_lo").is_not_null())
    covers = defined.filter((pl.col("ci_lo") <= CONFIG.beta_target) & (pl.col("ci_hi") >= CONFIG.beta_target))
    print(f"all pairs: declustered certified days median {a['n_cert'].median():.0f} (range {a['n_cert'].min()}-{a['n_cert'].max()}), "
          f"intervals on {defined.height} of {a.height} pairs, {covers.height} contain the target, "
          f"median length {(defined['ci_hi'] - defined['ci_lo']).median():.2f}")


if __name__ == "__main__":
    main()
