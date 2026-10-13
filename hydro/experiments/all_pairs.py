"""Every estimator on every pair of the main design.

    uv run python -m hydro.experiments.all_pairs              (~10 min)
    uv run python -m hydro.experiments.all_pairs --no-bootstrap

Reads pairs.parquet (hydro.pairs). Writes all_pairs.parquet, long, one row
per (pair, method, q_p), everything fitted on the full daily record: naive
OLS, the proximal bridge, and the certified OLS at every safeguard level of
the grid at its operating point. Intervals are year-level bootstraps,
computed for the comparison estimators and for the certified OLS at the
tabulated q_P.
"""

from __future__ import annotations

import sys

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import (
    OLS,
    fit_at_operating_point,
    fit_certified,
    naive_ols,
    proximal_bridge,
    year_bootstrap_interval,
)
from hydro.pairs import main_pairs, panel_for

EST = CONFIG.estimator


def _interval(table, fit_fn, bootstrap: bool):
    return year_bootstrap_interval(table, fit_fn) if bootstrap else (None, None, 0)


def rows_for(row: dict, bootstrap: bool) -> list[dict]:
    panel = panel_for(row)
    dy, p = panel.daily, panel.p_cols
    base = {"pair": row["pair"], "region": row["region"], "t_name": row["t_name"], "y_name": row["y_name"],
            "n_days": panel.n_daily, "area_ratio": row["area_ratio"]}
    out = []
    for name, fn in (("naive OLS", naive_ols), ("proximal bridge", lambda d: proximal_bridge(d, p))):
        b = fn(dy)
        if b is None:
            print(f"  {row['pair']}: {name} not defined, row skipped")
            continue
        lo, hi, nb = _interval(dy, fn, bootstrap)
        out.append(base | {"method": name, "n_cert": dy.height, "beta": b, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    for q_p in EST.q_p_grid:
        f = fit_at_operating_point(dy, p, q_p, OLS)
        if f is None:
            out.append(base | {"method": "certified OLS", "q_p": q_p, "note": "no operating point"})
            continue
        lo, hi, nb = _interval(
            dy, lambda d, v=f.v, q=q_p: fit_certified(d, p, q, v, OLS).beta, bootstrap and q_p == EST.q_p,
        )
        out.append(base | {"method": "certified OLS", "q_p": q_p, "q_t": f.q_t, "v": f.v, "n_cert": f.n_cert,
                           "beta": f.beta, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    return out


def main() -> None:
    bootstrap = "--no-bootstrap" not in sys.argv
    rows = []
    for row in main_pairs().iter_rows(named=True):
        rows.extend(rows_for(row, bootstrap))
        print(f"{row['pair']}: {row['t_name']} -> {row['y_name']} done", flush=True)
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUTPUT_DIR / "all_pairs.parquet")
    print(res.filter(pl.col("beta").is_not_null() & (pl.col("q_p").is_null() | (pl.col("q_p") == EST.q_p)))
          .group_by("method", maintain_order=True).agg(pl.len().alias("pairs"), pl.col("beta").median()))


if __name__ == "__main__":
    main()
