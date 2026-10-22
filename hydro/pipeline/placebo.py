"""The disconnected-pair placebo: the estimators of the main design on pairs
whose true routing coefficient is zero.

    uv run python -m hydro.pipeline.placebo

Reads placebo_pairs.parquet (hydro.pairs). Y is read on the same day as T
(no travel time between disconnected gauges). Writes placebo.parquet, long,
one row per (pair, method, q_p), all on the daily record: naive OLS, the
proximal bridge and the certified OLS at every safeguard level, at the main
design's operating rule. No intervals are computed.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import OLS, fit_at_operating_point, naive_ols, proximal_bridge
from hydro.pairs import panel_for

EST = CONFIG.estimator


def rows_for(row: dict) -> list[dict]:
    panel = panel_for(row, lag=0)
    dy, p = panel.daily, panel.p_cols
    base = {"pair": row["pair"], "region": row["region"], "t_name": row["t_name"], "y_name": row["y_name"],
            "n_days": panel.n_daily, "area_ratio": row["area_ratio"], "distance_km": row["distance_km"]}
    out = [base | {"method": "naive OLS", "n_cert": dy.height, "beta": naive_ols(dy)}]
    b = proximal_bridge(dy, p)
    if b is not None:
        out.append(base | {"method": "proximal bridge", "n_cert": dy.height, "beta": b})
    for q_p in EST.q_p_grid:
        f = fit_at_operating_point(dy, p, q_p, OLS)
        if f is not None:
            out.append(base | {"method": "certified OLS", "q_p": q_p, "q_t": f.q_t, "v": f.v,
                               "n_cert": f.n_cert, "beta": f.beta})
    return out


def main() -> None:
    rows = []
    for row in pl.read_parquet(OUTPUT_DIR / "placebo_pairs.parquet").iter_rows(named=True):
        rows.extend(rows_for(row))
        print(f"placebo {row['pair']}: {row['t_name']} | {row['y_name']} done", flush=True)
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUTPUT_DIR / "placebo.parquet")
    print(res.filter(pl.col("beta").is_not_null() & (pl.col("q_p").is_null() | (pl.col("q_p") == EST.q_p)))
          .group_by("method", maintain_order=True).agg(pl.len().alias("pairs"), pl.col("beta").median()))


if __name__ == "__main__":
    main()
