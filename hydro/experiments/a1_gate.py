"""A1 -- safeguard-off ablation. The gate for the whole application (spec 7.1).

`uv run python -m hydro.experiments.a1_gate`

A1 estimates beta from {X > v} alone. The certified estimator adds
{W in S}. If the two coincide, the safeguard is inert on this data and the
paper's central claim is unsupported here.

The comparison is run at an **identical absolute threshold v** for both arms, so
the only difference between them is the safeguard. (Comparing at a common
*quantile* would confound the safeguard with a shift in v, because the certified
convention takes the quantile of X on the safeguard-satisfying subsample, which
is a lower-flow subsample.) The A1-own-quantile arm is reported too, since that
is what a practitioner ignoring W would actually do.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import certified_threshold, fit_certificate
from hydro.experiments.candidates import design_a_candidates, headline_row, spec_from_row
from hydro.preprocess.panel import build_panel

Q_W = CONFIG.estimator.safeguard_quantile
GRID = CONFIG.estimator.certification_quantile_grid
TYPES = ("ols", "median")


def rows_for_pair(row: dict) -> list[dict]:
    spec = spec_from_row(row)
    panel = build_panel(spec, CONFIG, distance_km=row["distance_km"])
    if panel is None or panel.n_events < 30:
        return []
    ev, w = panel.events, panel.w_cols
    naive = fit_certificate(ev, [], estimator_type="ols", level=0.0)
    out = []
    for q_x in GRID:
        v = certified_threshold(ev, w, q_x, Q_W, "per_variable")
        for t in TYPES:
            cert = fit_certificate(ev, w, estimator_type=t, level=v, safeguard_quantile=Q_W)
            a1_v = fit_certificate(ev, [], estimator_type=t, level=v)
            a1_q = fit_certificate(ev, [], estimator_type=t, certification_quantile=q_x)
            out.append(
                {
                    "pair": spec.label,
                    "region": row["region"],
                    "x_name": row["x_name"],
                    "y_name": row["y_name"],
                    "area_ratio": row["area_ratio"],
                    "overlap_years": row["overlap_years"],
                    "lag_days": panel.lag_xy.lag_days,
                    "n_events": panel.n_events,
                    "estimator_type": t,
                    "certification_quantile": q_x,
                    "threshold_v": v,
                    "beta_certified": cert.beta,
                    "n_cert": cert.n_cert,
                    "beta_a1_same_v": a1_v.beta,
                    "n_a1_same_v": a1_v.n_cert,
                    "beta_a1_own_q": a1_q.beta,
                    "n_a1_own_q": a1_q.n_cert,
                    "beta_naive_ols": naive.beta,
                }
            )
    return out


def main() -> None:
    cand = design_a_candidates()
    headline = headline_row()["x"] + "->" + headline_row()["y"]
    print(f"{cand.height} Design-A candidates; headline = {headline}")
    rows: list[dict] = []
    for row in cand.iter_rows(named=True):
        rows.extend(rows_for_pair(row))

    df = pl.DataFrame(rows).with_columns(
        (pl.col("beta_certified") - pl.col("beta_a1_same_v")).alias("delta_same_v"),
        (pl.col("beta_certified") - pl.col("beta_a1_own_q")).alias("delta_own_q"),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_DIR / "a1_gate.parquet")

    read = pl.read_parquet(OUTPUT_DIR / "a1_gate.parquet")
    print(f"\nwrote a1_gate.parquet: {read.height} rows, {read['pair'].n_unique()} pairs\n")
    print(
        read.filter(pl.col("n_cert") >= CONFIG.estimator.unreliable_n_cert)
        .group_by("estimator_type", "certification_quantile")
        .agg(
            pl.len().alias("n_rows"),
            pl.median("beta_certified").alias("med_cert"),
            pl.median("beta_a1_same_v").alias("med_a1_v"),
            pl.median("beta_a1_own_q").alias("med_a1_q"),
            pl.median("delta_same_v").alias("med_delta_v"),
            (pl.col("delta_same_v") < 0).mean().alias("frac_cert_below_a1"),
        )
        .sort("estimator_type", "certification_quantile")
    )
    print(f"\nHeadline pair {headline}:")
    print(
        read.filter(pl.col("pair") == headline).select(
            "estimator_type", "certification_quantile", "threshold_v", "n_cert",
            "beta_certified", "beta_a1_same_v", "n_a1_same_v", "beta_a1_own_q", "beta_naive_ols",
        )
    )


if __name__ == "__main__":
    main()
