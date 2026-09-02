"""E2 + F2: disconnected-pair placebo, and E3: reverse direction.

`uv run python -m hydro.experiments.e2_placebo`

E2: X and Y in hydrologically disconnected catchments inside the same basin.
The true causal effect is exactly 0 and the shared weather driver U is strong,
so a naive through-the-origin regression is positive by construction. Note that
discharge is non-negative, so the certified estimate can only approach 0 from
above: Y keeps its baseflow however extreme X is.

E3: the same connected pairs as E1 with X and Y swapped, with the forward lag
applied in reverse (see `build_panel(force_lag=...)`).

Writes e2_placebo.parquet, e3_reverse.parquet, e2_placebo_dumbbell.{html,tex}.
"""

from __future__ import annotations

import plotly.express as px
import polars as pl
import tikzplotly

from hydro.baselines import naive_ols
from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import fit_certificate, year_bootstrap_interval
from hydro.experiments.candidates import (
    design_a_candidates,
    disconnected_candidates,
    panel_for,
    spec_from_row,
)
from hydro.experiments.e1_headline import Q_W_GRID, operating_point
from hydro.preprocess.panel import build_panel

EST = CONFIG.estimator
MAX_PLACEBO_PAIRS = 20
MIN_EVENTS = 60


def _estimates(events, w_cols, pair_meta: dict, experiment: str, daily=None) -> list[dict]:
    # Under the null the certified estimate cannot reach 0: discharge is
    # non-negative, so Y sits at its ordinary level while X sits above v. The
    # smallest value the estimator can plausibly return is therefore
    # median(daily Y) / mean(certified X), reported per row as `beta_null_floor`
    # so the distance of beta_hat from 0 can be read against it.
    y_med = float(daily["Y"].median()) if daily is not None else None
    rows = []
    naive = naive_ols(events)
    lo, hi, nb = year_bootstrap_interval(events, naive_ols)
    rows.append(pair_meta | {"experiment": experiment, "method": "naive OLS",
                             "estimator_type": "-", "n_cert": events.height,
                             "beta": naive, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    for q_w in Q_W_GRID:
        q_x, v, _ = operating_point(events, w_cols, q_w)
        if q_x is None:
            continue
        for t in ("ols", "median"):
            f = fit_certificate(events, w_cols, estimator_type=t, level=v, safeguard_quantile=q_w)
            lo, hi, nb = year_bootstrap_interval(
                events,
                lambda d, t=t, v=v, q=q_w: fit_certificate(
                    d, w_cols, estimator_type=t, level=v, safeguard_quantile=q
                ).beta,
            )
            rows.append(pair_meta | {
                "experiment": experiment, "method": "certificate", "estimator_type": t,
                "safeguard_quantile": q_w, "certification_quantile": q_x, "threshold_v": v,
                "n_cert": f.n_cert, "beta": f.beta, "ci_lo": lo, "ci_hi": hi, "n_boot": nb,
                "y_median_daily": y_med,
                "beta_null_floor": (
                    y_med / float(events.filter(pl.col("X") > v)["X"].mean())
                    if y_med is not None and f.n_cert else None
                ),
            })
    return rows


def e2() -> pl.DataFrame:
    rows: list[dict] = []
    cand = disconnected_candidates(MAX_PLACEBO_PAIRS * 3)
    for rule in ("nearest", "largest"):
        seen: set[str] = set()
        for row in cand.iter_rows(named=True):
            if len(seen) >= MAX_PLACEBO_PAIRS:
                break
            spec = spec_from_row(row, kind="disconnected", safeguard_rule=rule)
            if len(spec.w) < EST.n_safeguard_gauges:
                continue
            panel = build_panel(spec, CONFIG, distance_km=row["distance_km"])
            if panel is None or panel.n_events < MIN_EVENTS:
                continue
            seen.add(spec.label)
            meta = {"pair": spec.label, "region": row["region"], "x_name": row["x_name"],
                    "y_name": row["y_name"], "area_ratio": row["area_ratio"],
                    "distance_km": row["distance_km"], "n_events": panel.n_events,
                    "lag_days": panel.lag_xy.lag_days, "safeguards": ",".join(spec.w),
                    "safeguard_rule": rule, "reference_beta_routing": 0.0}
            rows.extend(_estimates(panel.events, panel.w_cols, meta, "E2", panel.daily))
    return pl.DataFrame(rows, infer_schema_length=None)


def e3() -> pl.DataFrame:
    rows: list[dict] = []
    for row in design_a_candidates().iter_rows(named=True):
        fwd = panel_for(row)
        if fwd is None or fwd.n_events < MIN_EVENTS:
            continue
        spec = spec_from_row(row, reverse=True)
        rev = build_panel(spec, CONFIG, distance_km=row["distance_km"],
                          force_lag=-fwd.lag_xy.lag_days)
        if rev is None or rev.n_events < MIN_EVENTS:
            continue
        meta = {"pair": spec.label, "region": row["region"], "x_name": row["y_name"],
                "y_name": row["x_name"], "area_ratio": 1 / row["area_ratio"],
                "distance_km": row["distance_km"], "n_events": rev.n_events,
                "lag_days": rev.lag_xy.lag_days, "forward_lag_days": fwd.lag_xy.lag_days,
                "safeguards": ",".join(spec.w), "safeguard_rule": "largest",
                "reference_beta_routing": 0.0}
        rows.extend(_estimates(rev.events, rev.w_cols, meta, "E3", rev.daily))
    return pl.DataFrame(rows, infer_schema_length=None)


def dumbbell(df: pl.DataFrame, q_w: float, title: str, stem: str, rule: str = "largest") -> None:
    """F2: one row per pair, naive vs certified.

    The pairs are ordered by the naive estimate so the figure reads as a single
    monotone ladder. beta is on a log x axis and clipped at 8: two placebo pairs
    have a naive coefficient above 20 (a large catchment regressed on a small
    one), which on a linear axis would push every certified marker onto the
    y axis. The clipped values are in the parquet.
    """
    wide = (
        df.filter(pl.col("safeguard_rule") == rule)
        .filter(
            (pl.col("method") == "naive OLS")
            | ((pl.col("estimator_type") == "ols") & (pl.col("safeguard_quantile") == q_w))
        )
        .with_columns(
            pl.when(pl.col("method") == "naive OLS")
            .then(pl.lit("naive OLS"))
            .otherwise(pl.lit(f"certified (q_w={q_w})"))
            .alias("arm")
        )
        .select("pair", "arm", "beta", "n_cert", "n_events")
    )
    order = (
        wide.filter(pl.col("arm") == "naive OLS").sort("beta")["pair"].to_list()
    )
    keep = set(wide.filter(pl.col("arm") != "naive OLS")["pair"])
    plot = wide.filter(pl.col("pair").is_in(keep)).with_columns(
        pl.col("beta").clip(0.01, 8.0).alias("beta_plot")
    )
    fig = px.scatter(
        plot, x="beta_plot", y="pair", color="arm", log_x=True,
        category_orders={"pair": [p for p in order if p in keep]},
        labels={"beta_plot": "beta hat (clipped to [0.01, 8])", "pair": ""},
        title=title, hover_data=["n_cert", "n_events"],
    )
    fig.update_traces(marker_size=10)
    fig.update_yaxes(dtick=1)  # label every pair; plotly otherwise shows every second one
    fig.add_vline(x=1.0, line_dash="dot", line_width=1, opacity=0.4)
    fig.write_html(OUTPUT_DIR / f"{stem}.html")
    tikzplotly.save(OUTPUT_DIR / f"{stem}.tex", fig)


def main() -> None:
    d2, d3 = e2(), e3()
    d2.write_parquet(OUTPUT_DIR / "e2_placebo.parquet")
    d3.write_parquet(OUTPUT_DIR / "e3_reverse.parquet")

    q_w = EST.safeguard_quantile
    r2 = pl.read_parquet(OUTPUT_DIR / "e2_placebo.parquet")
    r3 = pl.read_parquet(OUTPUT_DIR / "e3_reverse.parquet")
    dumbbell(r2, q_w,
             f"F2 disconnected-pair placebo (true beta = 0), certificate at q_w={q_w}, "
             f"safeguard = 3 disconnected gauges nearest Y",
             "e2_placebo_dumbbell", rule="nearest")
    dumbbell(r3, q_w, f"E3 reverse direction (true beta = 0), certificate at q_w={q_w}",
             "e3_reverse_dumbbell")

    for name, r in (("E2", r2), ("E3", r3)):
        print(f"\n== {name}: {r['pair'].n_unique()} pairs ==")
        print(
            r.filter((pl.col("method") == "naive OLS") | (pl.col("estimator_type") == "ols"))
            .group_by("safeguard_rule", "method", "safeguard_quantile")
            .agg(pl.len().alias("n_pairs"), pl.median("beta").alias("median_beta"),
                 pl.quantile("beta", 0.1).alias("q10"), pl.quantile("beta", 0.9).alias("q90"),
                 pl.median("n_cert").alias("median_n_cert"),
                 pl.median("beta_null_floor").alias("median_null_floor"))
            .sort("safeguard_rule", "method", "safeguard_quantile")
        )


if __name__ == "__main__":
    main()
