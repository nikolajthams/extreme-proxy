"""The headline pair: threshold-stability grid and the ablations V1-V3.

    uv run python -m hydro.pipeline.headline

Reads pairs.parquet. Writes headline.parquet with
  * method "certified OLS" on the full (q_p, q_t) grid of the daily record:
    beta and n_cert against the threshold v, for the appendix stability figure;
  * the ablations at the certified OLS's operating point at each q_p:
      V1  tail only: OLS on {T > v} with no safeguard, the same v
      V2  permuted safeguard: the proxy block permuted across days, median
          over permutations
      V3  a safeguard from another basin: the Iowa/Cedar gauges of
          `Estimator.other_basin_proxies`, on their own panel and at their
          own operating point.
    At the tabulated q_P all three carry a 90% year-level bootstrap interval,
    computed as for the estimators (v held fixed); for V2 every resample
    also draws a fresh permutation of the proxy block.
The estimates and intervals of the headline pair itself come from
all_pairs.parquet.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR, REGION_BY_NAME
from hydro.estimation import OLS, certified_events, fit_certified, operating_point, threshold, year_bootstrap_interval
from hydro.pairs import headline_pair, panel_for
from hydro.preprocess.panel import PairSpec, build_panel

EST = CONFIG.estimator


def stability_rows(dy, p, base) -> list[dict]:
    out = []
    for q_p in EST.q_p_grid:
        for q_t in EST.q_t_grid:
            v = threshold(dy, p, q_p, q_t)
            f = fit_certified(dy, p, q_p, v, OLS, q_t=q_t)
            out.append(base | {"method": "certified OLS", "q_p": q_p, "q_t": q_t, "v": v,
                               "n_cert": f.n_cert, "beta": f.beta})
    return out


def _permuted_fit(dy, p, q_p, q_t, rng) -> float | None:
    perm = dy.drop(p).hstack(dy.select(p)[rng.permutation(dy.height)])
    return fit_certified(perm, p, q_p, threshold(perm, p, q_p, q_t), OLS).beta


def permuted_safeguard(dy, p, q_p, q_t) -> list[float]:
    rng = np.random.default_rng(EST.seed)
    out = []
    for _ in range(EST.n_permutations):
        b = _permuted_fit(dy, p, q_p, q_t, rng)
        if b is not None:
            out.append(b)
    return out


def _interval(table, fit_fn, q_p):
    """Bootstrap only at the tabulated q_P, as in all_pairs."""
    return year_bootstrap_interval(table, fit_fn) if q_p == EST.q_p else (None, None, 0)


def ablation_rows(row, dy, p, base) -> list[dict]:
    out = []
    a3_spec = PairSpec(row["pair"], REGION_BY_NAME[row["region"]], row["t"], row["y"],
                       list(EST.other_basin_proxies))
    a3 = build_panel(a3_spec, distance_km=row["distance_km"])
    for q_p in EST.q_p_grid:
        op = operating_point(dy, p, q_p)
        if op is None:
            continue
        q_t, v = op
        common = base | {"q_p": q_p, "q_t": q_t, "v": v}
        a1 = fit_certified(dy, [], q_p, v, OLS)
        lo, hi, nb = _interval(dy, lambda d, v=v, q=q_p: fit_certified(d, [], q, v, OLS).beta, q_p)
        out.append(common | {"method": "V1 tail only", "n_cert": a1.n_cert, "beta": a1.beta,
                             "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
        draws = permuted_safeguard(dy, p, q_p, q_t)
        rng = np.random.default_rng(EST.seed + 1)
        lo, hi, nb = _interval(dy, lambda d, q=q_p, qt=q_t: _permuted_fit(d, p, q, qt, rng), q_p)
        out.append(common | {"method": "V2 permuted safeguard", "n_cert": certified_events(dy, p, q_p, v).height,
                             "beta": float(np.median(draws)) if draws else None,
                             "ci_lo": lo, "ci_hi": hi, "n_boot": nb,
                             "note": f"median over {EST.n_permutations} permutations; "
                                     "bootstrap with a fresh permutation per resample"})
        if a3 is not None:
            op3 = operating_point(a3.daily, a3.p_cols, q_p)
            if op3 is not None:
                f3 = fit_certified(a3.daily, a3.p_cols, q_p, op3[1], OLS, q_t=op3[0])
                lo, hi, nb = _interval(
                    a3.daily, lambda d, v3=f3.v, q=q_p: fit_certified(d, a3.p_cols, q, v3, OLS).beta, q_p)
                out.append(base | {"method": "V3 other-basin safeguard", "q_p": q_p, "q_t": f3.q_t, "v": f3.v,
                                   "n_cert": f3.n_cert, "beta": f3.beta, "n_days": a3.n_daily,
                                   "ci_lo": lo, "ci_hi": hi, "n_boot": nb,
                                   "note": "P = " + ",".join(EST.other_basin_proxies)})
    return out


def main() -> None:
    row = headline_pair()
    panel = panel_for(row)
    dy, p = panel.daily, panel.p_cols
    print(f"headline {row['pair']}: {row['t_name']} -> {row['y_name']}, P = {row['proxies']}, "
          f"lag {panel.lag.lag_days} d, {panel.n_daily} days")
    base = {"pair": row["pair"], "region": row["region"], "n_days": panel.n_daily}
    res = pl.DataFrame(stability_rows(dy, p, base) + ablation_rows(row, dy, p, base), infer_schema_length=None)
    res.write_parquet(OUTPUT_DIR / "headline.parquet")
    with pl.Config(tbl_rows=30):
        print(res.filter(pl.col("method") != "certified OLS")
              .select("method", "q_p", "q_t", "v", "n_cert", "beta", "ci_lo", "ci_hi"))


if __name__ == "__main__":
    main()
