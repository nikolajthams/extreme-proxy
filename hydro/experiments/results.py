"""Consolidate every experiment into `results.parquet` and the pair artifact.

`uv run python -m hydro.experiments.results`

One row per (experiment, pair, method, hyperparameter setting). Run the three
experiment modules first; this module only reads what they wrote.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import hill_estimate
from hydro.experiments.candidates import design_a_candidates, panel_for, spec_from_row

KEYS = [
    "experiment", "pair", "region", "method", "estimator_type",
    "safeguard_quantile", "certification_quantile", "safeguard_rule",
    "threshold_v", "n_cert", "n_events", "beta", "ci_lo", "ci_hi", "n_boot",
    "reference_beta_routing", "reference_area_ratio", "hill_alpha_x",
    "beta_null_floor", "lag_days", "note",
]


def _a1_long() -> pl.DataFrame:
    """A1 gate, reshaped to the common long format (all Design-A pairs)."""
    d = pl.read_parquet(OUTPUT_DIR / "a1_gate.parquet")
    common = ["pair", "region", "certification_quantile", "estimator_type",
              "threshold_v", "n_events", "lag_days"]
    parts = []
    for beta_col, n_col, method in (
        ("beta_certified", "n_cert", "certificate"),
        ("beta_a1_same_v", "n_a1_same_v", "A1 tail only (no safeguard)"),
        ("beta_naive_ols", None, "naive OLS"),
    ):
        parts.append(
            d.select(
                *common,
                pl.col(beta_col).alias("beta"),
                (pl.col(n_col) if n_col else pl.col("n_events")).alias("n_cert"),
                pl.col("area_ratio").alias("reference_area_ratio"),
                pl.lit(method).alias("method"),
                pl.lit(CONFIG.estimator.safeguard_quantile if method == "certificate" else None,
                       dtype=pl.Float64).alias("safeguard_quantile"),
                pl.lit("E1_all_pairs").alias("experiment"),
                pl.lit(CONFIG.beta_routing_target).alias("reference_beta_routing"),
            )
        )
    return pl.concat(parts, how="vertical_relaxed")


def pair_artifact() -> pl.DataFrame:
    """Design-A candidate table with the panel diagnostics each estimate rests on."""
    rows = []
    for row in design_a_candidates().iter_rows(named=True):
        spec = spec_from_row(row)
        p = panel_for(row)
        alpha_d, _ = hill_estimate(p.daily["X"].to_numpy()) if p else (None, 0)
        alpha_e, _ = hill_estimate(p.events["X"].to_numpy()) if p else (None, 0)
        rows.append({
            k: row[k] for k in ("region", "pair", "x", "y", "x_name", "y_name",
                                "area_x_km2", "area_y_km2", "area_ratio",
                                "n_tributaries", "n_safeguards_used", "tributaries",
                                "ungauged_gain_fraction", "overlap_years", "distance_km",
                                "z", "provenance")
        } | {
            "safeguards_used": ",".join(spec.w),
            "panel_built": p is not None,
            "n_daily": p.n_daily if p else None,
            "n_events": p.n_events if p else None,
            "lag_days": p.lag_xy.lag_days if p else None,
            "lag_correlation": p.lag_xy.correlation if p else None,
            "lag_plausible": p.lag_xy.plausible if p else None,
            "decluster_threshold_m3s": p.threshold_u if p else None,
            "dropped_on_quality": p.dropped_quality if p else None,
            "hill_alpha_daily_x": alpha_d,
            "hill_alpha_event_x": alpha_e,
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    sources = ["e1_results", "e1_prereg_v1", "e2_placebo", "e3_reverse"]
    frames = [_a1_long()]
    for name in sources:
        path = OUTPUT_DIR / f"{name}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            if "experiment" not in df.columns:
                df = df.with_columns(pl.lit(name.split("_")[0].upper()).alias("experiment"))
            frames.append(df)

    res = pl.concat(frames, how="diagonal_relaxed")
    res = res.select([pl.col(c) for c in KEYS if c in res.columns]
                     + [c for c in res.columns if c not in KEYS])
    res.write_parquet(OUTPUT_DIR / "results.parquet")
    pair_artifact().write_parquet(OUTPUT_DIR / "design_a_pairs.parquet")

    read = pl.read_parquet(OUTPUT_DIR / "results.parquet")
    print(f"results.parquet: {read.height} rows, {read['pair'].n_unique()} pairs")
    print(read.group_by("experiment", "method").agg(
        pl.len().alias("rows"), pl.median("beta").alias("median_beta")
    ).sort("experiment", "method"))
    pairs = pl.read_parquet(OUTPUT_DIR / "design_a_pairs.parquet")
    print(f"\ndesign_a_pairs.parquet: {pairs.height} pairs, "
          f"{pairs['panel_built'].sum()} with a panel, "
          f"median n_events {pairs['n_events'].median()}, "
          f"all lags plausible = {bool(pairs['lag_plausible'].drop_nulls().all())}")


if __name__ == "__main__":
    main()
