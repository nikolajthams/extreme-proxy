"""E1 + F1 + F4: headline Design-A pair, threshold stability, baselines, A2/A3.

`uv run python -m hydro.experiments.e1_headline`

Writes under experiments/river_discharge/:
  e1_stability.parquet                                     (F1 source table)
  e1_threshold_stability.{html,tex}                        (F1, beta vs v by q_w)
  e1_estimator_types.{html,tex}                            (F1b, ols/median/trimmed/max)
  e1_method_comparison.{html,tex}                          (F4)
  e1_certified_events.parquet                              (audit trail)
  e1_results.parquet, e1_prereg_v1.parquet                 (long, for results.parquet)
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import polars as pl
import tikzplotly

from hydro.baselines import BASELINES
from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import (
    certified_events,
    certified_threshold,
    fit_certificate,
    hill_estimate,
    year_bootstrap_interval,
)
from hydro.experiments.candidates import design_a_candidates, headline_row, panel_for, spec_from_row
from hydro.preprocess.panel import PairSpec, build_panel

EST = CONFIG.estimator
Q_X_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Q_W_GRID = (0.25, 0.5, 0.75, 0.9)
TYPES = ("ols", "median", "trimmed", "max")
# A3: a gauge in a climatically unrelated basin (Iowa/Cedar) used as the
# safeguard for a Susquehanna pair. Largest Iowa gauges with long records.
IRRELEVANT_W = ("05451500", "05449500", "05452000")
N_PERMUTATIONS = 50


def operating_point(events, w_cols, q_w: float) -> tuple[float | None, float, int]:
    """Pre-registered rule: largest q_x on the grid with n_cert >= min_n_cert."""
    best = (None, float("nan"), 0)
    for q_x in EST.certification_quantile_grid:
        v = certified_threshold(events, w_cols, q_x, q_w, "per_variable")
        f = fit_certificate(events, w_cols, level=v, safeguard_quantile=q_w)
        if f.n_cert >= EST.min_n_cert:
            best = (q_x, v, f.n_cert)
    return best


def stability(events, w_cols) -> pl.DataFrame:
    rows = []
    for q_w in Q_W_GRID:
        for q_x in Q_X_GRID:
            v = certified_threshold(events, w_cols, q_x, q_w, "per_variable")
            for t in TYPES:
                f = fit_certificate(events, w_cols, estimator_type=t, level=v, safeguard_quantile=q_w)
                rows.append(
                    {
                        "safeguard_quantile": q_w,
                        "certification_quantile": q_x,
                        "threshold_v": v,
                        "estimator_type": t,
                        "beta": f.beta,
                        "n_cert": f.n_cert,
                        "exceedance_prob": 1 - q_x,
                    }
                )
    return pl.DataFrame(rows)


def permuted_safeguard(events, w_cols, q_w, q_x, t, seed) -> list[float]:
    """A2: time-permute the safeguard block, preserving its joint marginal."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_PERMUTATIONS):
        idx = rng.permutation(events.height)
        perm = events.drop(w_cols).hstack(events.select(w_cols)[idx])
        v = certified_threshold(perm, w_cols, q_x, q_w, "per_variable")
        b = fit_certificate(perm, w_cols, estimator_type=t, level=v, safeguard_quantile=q_w).beta
        if b is not None and np.isfinite(b):
            out.append(b)
    return out


def main() -> None:
    row = headline_row()
    spec = spec_from_row(row)
    panel = panel_for(row)
    ev, w = panel.events, panel.w_cols
    area_ratio = row["area_ratio"]
    alpha, k = hill_estimate(ev["X"].to_numpy())

    print(
        f"headline {spec.label}: {row['x_name']} -> {row['y_name']}\n"
        f"  W = {spec.w}  Z = {spec.z}  lag = {panel.lag_xy.lag_days} d "
        f"(corr {panel.lag_xy.correlation:.3f}, celerity ok = {panel.lag_xy.plausible})\n"
        f"  {panel.n_daily} in-season complete days, {panel.n_events} declustered events, "
        f"decluster threshold {panel.threshold_u:.1f} m3/s, {panel.dropped_quality} days dropped on quality\n"
        f"  area ratio A_Y/A_X = {area_ratio:.3f}, Hill alpha(X) = {alpha:.2f} (k={k})"
    )

    # ---------------------------------------------------------------- F1
    stab = stability(ev, w)
    stab.write_parquet(OUTPUT_DIR / "e1_stability.parquet")

    # Two single-panel figures rather than one faceted one: tikzplotly flattens
    # facets into a single axis, and the .tex is what goes into the paper.
    # Points with fewer than `unreliable_n_cert` certified events are dropped
    # rather than drawn -- at n_cert < 20 one event moves beta by more than the
    # whole vertical range and would flatten every other curve.
    plot = stab.filter(
        (pl.col("n_cert") >= EST.unreliable_n_cert) & pl.col("beta").is_not_null()
    )
    caption = (
        f"{spec.label} ({row['x_name']} -> {row['y_name']}), n_events={panel.n_events}, "
        f"n_cert<{EST.unreliable_n_cert} omitted; refs: routing target 1.0, "
        f"area ratio {area_ratio:.2f}"
    )
    figs = {
        "e1_threshold_stability": px.line(
            plot.filter(pl.col("estimator_type") == "ols")
            .with_columns(pl.col("safeguard_quantile").cast(pl.Utf8).alias("q_w"))
            .sort("threshold_v"),
            x="threshold_v", y="beta", color="q_w", markers=True, log_x=True,
            labels={"threshold_v": "treatment threshold v (m3/s)", "beta": "beta hat"},
            title=f"F1 threshold stability by safeguard level (OLS); {caption}",
        ),
        "e1_estimator_types": px.line(
            plot.filter(pl.col("safeguard_quantile") == EST.safeguard_quantile).sort("threshold_v"),
            x="threshold_v", y="beta", color="estimator_type", markers=True, log_x=True,
            labels={"threshold_v": "treatment threshold v (m3/s)", "beta": "beta hat"},
            title=f"F1b estimator types at q_w {EST.safeguard_quantile}; {caption}",
        ),
    }
    for name, fig in figs.items():
        for yv in (CONFIG.beta_routing_target, area_ratio):
            fig.add_hline(y=yv, line_dash="dot", line_width=1, opacity=0.5)
        fig.write_html(OUTPUT_DIR / f"{name}.html")
        tikzplotly.save(OUTPUT_DIR / f"{name}.tex", fig)

    # ------------------------------------------------- operating point + F4
    rows: list[dict] = []
    base = dict(experiment="E1", pair=spec.label, region=row["region"],
                n_events=panel.n_events, reference_beta_routing=CONFIG.beta_routing_target,
                reference_area_ratio=area_ratio, hill_alpha_x=alpha)

    for q_w in Q_W_GRID:
        q_x, v, n = operating_point(ev, w, q_w)
        if q_x is None:
            rows.append(base | {"method": "certificate", "safeguard_quantile": q_w,
                                "note": f"no grid point reaches n_cert>={EST.min_n_cert}"})
            continue
        for t in TYPES:
            f = fit_certificate(ev, w, estimator_type=t, level=v, safeguard_quantile=q_w)
            lo, hi, nb = year_bootstrap_interval(
                ev,
                lambda d, t=t, v=v, q=q_w: fit_certificate(
                    d, w, estimator_type=t, level=v, safeguard_quantile=q
                ).beta,
            )
            rows.append(base | {
                "method": "certificate", "estimator_type": t, "safeguard_quantile": q_w,
                "certification_quantile": q_x, "threshold_v": v, "n_cert": f.n_cert,
                "beta": f.beta, "ci_lo": lo, "ci_hi": hi, "n_boot": nb,
            })
        # A1 at the same v, and the two safeguard ablations
        rows.append(base | {"method": "A1 tail only (no safeguard)", "estimator_type": "ols",
                            "safeguard_quantile": q_w, "certification_quantile": q_x,
                            "threshold_v": v,
                            **_fit_row(fit_certificate(ev, [], estimator_type="ols", level=v))})
        draws = permuted_safeguard(ev, w, q_w, q_x, "ols", CONFIG.seed)
        rows.append(base | {
            "method": "A2 permuted safeguard", "estimator_type": "ols",
            "safeguard_quantile": q_w, "certification_quantile": q_x, "threshold_v": v,
            "beta": float(np.median(draws)) if draws else None,
            "ci_lo": float(np.quantile(draws, 0.05)) if draws else None,
            "ci_hi": float(np.quantile(draws, 0.95)) if draws else None,
            "n_boot": len(draws), "note": f"{N_PERMUTATIONS} permutations",
        })

    # A3 needs its own panel (different sites => different common record).
    a3_spec = PairSpec(f"{spec.x}->{spec.y}", spec.region, spec.x, spec.y,
                       list(IRRELEVANT_W), spec.z, "connected")
    a3 = build_panel(a3_spec, CONFIG, distance_km=row["distance_km"])
    if a3 is not None:
        for q_w in Q_W_GRID:
            q_x, v, n = operating_point(a3.events, a3.w_cols, q_w)
            if q_x is None:
                continue
            f = fit_certificate(a3.events, a3.w_cols, estimator_type="ols", level=v,
                                safeguard_quantile=q_w)
            rows.append(base | {
                "method": "A3 irrelevant safeguard", "estimator_type": "ols",
                "safeguard_quantile": q_w, "certification_quantile": q_x, "threshold_v": v,
                "n_cert": f.n_cert, "beta": f.beta, "n_events": a3.n_events,
                "note": f"W = {','.join(IRRELEVANT_W)} (Iowa/Cedar basin)",
            })

    for name, fn in BASELINES.items():
        b = fn(ev, w_cols=w)
        lo, hi, nb = year_bootstrap_interval(ev, lambda d, fn=fn: fn(d, w_cols=w))
        rows.append(base | {"method": name, "estimator_type": "-", "n_cert": panel.n_events,
                            "beta": b, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    rows.append(base | {"method": "drainage-area ratio (confounded reference)",
                        "estimator_type": "-", "beta": area_ratio,
                        "note": "not an estimator; the value OLS returns under spatially uniform runoff"})

    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUTPUT_DIR / "e1_results.parquet")

    # F4 forest: one row per method at the pre-registered q_w, plus the two refs.
    q_w0 = EST.safeguard_quantile
    forest = res.filter(
        pl.col("beta").is_not_null()
        & (pl.col("estimator_type").is_in(["ols", "-"]))
        & (pl.col("safeguard_quantile").is_null() | (pl.col("safeguard_quantile") == q_w0))
    ).with_columns(
        (pl.col("method") + pl.when(pl.col("safeguard_quantile").is_not_null())
         .then(pl.lit(f" [q_w {q_w0}]")).otherwise(pl.lit(""))).alias("label"),
        (pl.col("ci_hi") - pl.col("beta")).fill_null(0.0).alias("err_hi"),
        (pl.col("beta") - pl.col("ci_lo")).fill_null(0.0).alias("err_lo"),
    )
    # x is clipped to [0, 4]: the bridge baselines can land far outside that at
    # alpha < 2 and would otherwise compress every other method into one pixel.
    # Their true values stay in e1_results.parquet and are printed below.
    fig4 = px.scatter(
        forest.sort("beta"), x="beta", y="label", error_x="err_hi", error_x_minus="err_lo",
        range_x=[0, 4],
        title=(f"F4 method comparison, {spec.label}; certificate at q_w {q_w0}, "
               f"90% year-level bootstrap; routing target 1.0, area ratio {area_ratio:.2f}"),
        labels={"beta": "beta hat (axis clipped to [0, 4])", "label": ""},
    )
    fig4.update_traces(marker_size=10)
    for xv in (CONFIG.beta_routing_target, area_ratio):
        fig4.add_vline(x=xv, line_dash="dot", line_width=1, opacity=0.5)
    fig4.write_html(OUTPUT_DIR / "e1_method_comparison.html")
    tikzplotly.save(OUTPUT_DIR / "e1_method_comparison.tex", fig4)

    # The pre-amendment headline (see preregistration.md) is estimated too, so
    # that changing the rule cannot hide a result.
    prereg_rows = _prereg_v1_rows()
    if prereg_rows:
        pl.DataFrame(prereg_rows, infer_schema_length=None).write_parquet(
            OUTPUT_DIR / "e1_prereg_v1.parquet"
        )

    # ------------------------------------------------------------ audit trail
    q_x0, v0, _ = operating_point(ev, w, q_w0)
    certified_events(ev, w, level=v0, safeguard_quantile=q_w0).write_parquet(
        OUTPUT_DIR / "e1_certified_events.parquet"
    )

    read = pl.read_parquet(OUTPUT_DIR / "e1_results.parquet")
    print("\n-- certificate across safeguard levels (ols) --")
    print(read.filter((pl.col("method") == "certificate") & (pl.col("estimator_type") == "ols"))
          .select("safeguard_quantile", "certification_quantile", "threshold_v", "n_cert",
                  "beta", "ci_lo", "ci_hi"))
    print("\n-- ablations / baselines --")
    print(read.filter(pl.col("method") != "certificate")
          .select("method", "safeguard_quantile", "n_cert", "beta", "ci_lo", "ci_hi", "note"))


PREREG_V1_PAIR = ("01559000", "01563500")


def _prereg_v1_rows() -> list[dict]:
    pairs = pl.read_parquet(OUTPUT_DIR / "station_pairs.parquet")
    hit = pairs.filter((pl.col("x") == PREREG_V1_PAIR[0]) & (pl.col("y") == PREREG_V1_PAIR[1]))
    if hit.height == 0:
        return []
    row = hit.to_dicts()[0]
    panel = panel_for(row)
    if panel is None:
        return []
    ev, w = panel.events, panel.w_cols
    base = dict(experiment="E1_prereg_v1", pair=f"{row['x']}->{row['y']}", region=row["region"],
                n_events=panel.n_events, reference_beta_routing=CONFIG.beta_routing_target,
                reference_area_ratio=row["area_ratio"],
                note=f"pre-amendment headline; only {len(w)} usable safeguard gauges")
    out = []
    for q_w in Q_W_GRID:
        q_x, v, _ = operating_point(ev, w, q_w)
        if q_x is None:
            continue
        f = fit_certificate(ev, w, estimator_type="ols", level=v, safeguard_quantile=q_w)
        a1 = fit_certificate(ev, [], estimator_type="ols", level=v)
        out.append(base | {"method": "certificate", "estimator_type": "ols",
                           "safeguard_quantile": q_w, "certification_quantile": q_x,
                           "threshold_v": v, "n_cert": f.n_cert, "beta": f.beta})
        out.append(base | {"method": "A1 tail only (no safeguard)", "estimator_type": "ols",
                           "safeguard_quantile": q_w, "certification_quantile": q_x,
                           "threshold_v": v, "n_cert": a1.n_cert, "beta": a1.beta})
    out.append(base | {"method": "naive OLS", "estimator_type": "-",
                       "n_cert": panel.n_events, "beta": BASELINES["naive OLS"](ev, w_cols=w)})
    return out


def _fit_row(f) -> dict:
    return {"n_cert": f.n_cert, "beta": f.beta}


if __name__ == "__main__":
    main()
