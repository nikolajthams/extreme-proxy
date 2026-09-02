"""Bias decomposition and safeguard structures on every Design-A pair.

Where does the safeguard sit, and what does the certificate remove? Water is
conserved, so on the event table Y = X + sum(W) + U, with U the inflow the
safeguard gauges do not measure, and the through-the-origin slope splits as
beta_naive = 1 + b_W + b_U (`hydro.estimation.bias_decomposition`). The same
split on a certified subset says whether the certificate removed the gauged
part, the ungauged part, or neither.

Two safeguard structures are run on every pair:

  tributary  W = the (<= 3) largest non-nested gauged tributaries entering
             between X and Y. Direct causes of Y (coefficient 1 by water
             conservation) and children of the storm.
  proxy      W = the (<= 3) gauges nearest to Y that are hydrologically
             disconnected from both X and Y. Children of the storm with no
             edge into X or Y: a pure proxy, the paper's double-proxy graph.

For each structure the file holds naive OLS, the adjustment baselines (free
coefficients on W, and W subtracted at its physical coefficient 1), the
instrumental and proximal baselines, the tail-only ablation A1, and the
certificate along the safeguard grid, each at the largest certification
quantile on the grid that keeps at least MIN_N events. Intervals are year-level
bootstraps at the tabulated safeguard level only; at MIN_N events they are
indicative at best, and the paper prints n_cert next to every estimate.

    uv run python -m hydro.experiments.decomposition            # ~25 min
    uv run python -m hydro.experiments.decomposition --no-bootstrap

Writes under experiments/river_discharge/:
    decomposition.parquet        long: one row per (pair, structure, method, q_w)
    decomposition_pairs.parquet  one row per (pair, structure): gauges, coverage
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from graphical_toolbox.models.proximal import LinearProximal

from hydro.baselines import BASELINES
from hydro.config import CONFIG, OUTPUT_DIR
from hydro.estimation import (
    bias_decomposition,
    certified_events,
    certified_threshold,
    fit_certificate,
    hill_estimate,
    year_bootstrap_interval,
)
from hydro.experiments.candidates import (
    _site_areas,
    design_a_candidates,
    network_for,
    panel_for,
    region_of,
    spec_from_row,
)
from hydro.preprocess.panel import PairSpec, build_panel

EST = CONFIG.estimator
Q_W_GRID = (0.9, 0.75, 0.5, 0.25)
# The tabulated safeguard level and the smallest certified sample we report.
# The retained fraction is ~ q_W^k (1 - q_x), so at q_W = 0.25 not every pair
# keeps MIN_N events; the figures then fall back to q_W = 0.5 with an open
# marker, and the sweep shows every level that exists.
TABULATED_Q_W = 0.25
FALLBACK_Q_W = 0.5
MIN_N = 10
STRUCTURES = ("tributary", "proxy")
# Baselines per structure. The water balance and the tributary-side proximal
# bridge are defined only when W are the tributaries; the two-proxy bridge
# only when W are pure proxies.
BASELINES_FOR = {
    "tributary": ("naive OLS", "OLS adjusting for W", "OLS subtracting W",
                  "2SLS (headwater Z)", "proximal (linear bridge)"),
    "proxy": ("naive OLS", "OLS adjusting for W", "2SLS (headwater Z)",
              "proximal (linear bridge)", "proximal (two proxies)"),
}


def operating_point(events: pl.DataFrame, w_cols: list[str], q_w: float,
                    min_n: int = MIN_N) -> tuple[float | None, float, int]:
    """Largest q_x on the grid whose certified sample keeps >= min_n events."""
    best = (None, float("nan"), 0)
    for q_x in EST.certification_quantile_grid:
        v = certified_threshold(events, w_cols, q_x, q_w, "per_variable")
        f = fit_certificate(events, w_cols, level=v, safeguard_quantile=q_w)
        if f.n_cert >= min_n:
            best = (q_x, v, f.n_cert)
    return best


def _interval(events, fit_fn, bootstrap: bool):
    if not bootstrap:
        return (None, None, 0)
    return year_bootstrap_interval(events, fit_fn)


def _ols_t_coef(df: pl.DataFrame, adj: list[str]) -> float | None:
    """Coefficient on X in a through-the-origin regression of Y on (X, adj)."""
    a = df.select(["X", *adj]).to_numpy()
    b = df["Y"].to_numpy()
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(coef[0])


def _subtract_w(df: pl.DataFrame, adj: list[str]) -> float:
    """Slope of Y - sum(adj) on X: the tributaries taken out at their physical
    coefficient one, so only the intercept-free routing slope is estimated."""
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy() - df.select(adj).to_numpy().sum(axis=1)
    return float(x @ y / (x @ x))


def adjusted_rows(row: dict, structure: str, bootstrap: bool, base: dict) -> list[dict]:
    """The certificate with the tributaries regressed out, plus the (P, W)
    bridge. Safeguard = the structure's own (W or P); the adjustment always
    uses the tributaries W.

    The certified samples are small (>= MIN_N events), so the adjustment is
    reported in two forms: `minus W`, which subtracts the tributaries at their
    physical coefficient one and estimates a single parameter, and
    `adjusting for W`, which fits one free coefficient per tributary and is
    computed only when the certified sample holds at least ADJUST_MIN_PER_COEF
    events per parameter. Which form the paper carries is decided by the
    stability check in the module docstring's spike, not here.

    For structure="proxy" this needs the proxy gauges and the tributaries in
    one panel, whose record intersection is its own; its event count is
    reported in `n_events` for these rows.
    """
    spec = spec_from_row(row, kind="connected" if structure == "tributary" else "proxy")
    trib = spec_from_row(row)
    if structure == "proxy":
        if len(spec.w) < 2:
            return []
        comb = PairSpec(row["pair"], region_of(row["region"]), row["x"], row["y"],
                        spec.w + [t for t in trib.w if t not in spec.w])
        panel = build_panel(comb, CONFIG, distance_km=row["distance_km"])
        if panel is None:
            return []
        safeguard = panel.w_cols[:len(spec.w)]
        adj = panel.w_cols[len(spec.w):]
    else:
        panel = panel_for(row)
        if panel is None:
            return []
        safeguard = adj = panel.w_cols
    if not adj:
        return []
    ev = panel.events
    out = []
    base = base | {"n_events": panel.n_events, "k_adjust": len(adj)}
    zcol, wcol = (safeguard[0], adj[0])
    if structure == "proxy":
        # The (P, W) bridge: a disconnected gauge is a valid treatment-side
        # proxy (independent of Y given the storm) and the largest tributary a
        # valid outcome-side proxy (it may cause Y), so unlike (W1, W2) this
        # pairing satisfies the proximal exclusion conditions.
        b = LinearProximal("X", "Y", zcol, wcol).fit_or_null(ev)
        if b is not None:
            fn = lambda d: (lambda r: None if r is None else float(np.asarray(r).flatten()[0]))(
                LinearProximal("X", "Y", zcol, wcol).fit_or_null(d))
            lo, hi, nb = _interval(ev, fn, bootstrap)
            out.append(base | {"method": "proximal (P, W)", "n_cert": ev.height,
                               "beta": float(np.asarray(b).flatten()[0]),
                               "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    for q_w in Q_W_GRID:
        q_x, v, n = operating_point(ev, safeguard, q_w)
        if q_x is None:
            continue
        sub = certified_events(ev, safeguard, level=v, safeguard_quantile=q_w)
        boot = bootstrap and q_w in (TABULATED_Q_W, FALLBACK_Q_W)
        common = base | {"safeguard_quantile": q_w, "certification_quantile": q_x,
                         "threshold_v": v, "n_cert": n}

        def certified(d, fn):
            s = make_sub(d, v, q_w)
            return fn(s) if s.height > len(adj) + 1 else None

        def make_sub(d, v=v, q=q_w):
            return certified_events(d, safeguard, level=v, safeguard_quantile=q)

        lo, hi, nb = _interval(ev, lambda d: certified(d, lambda s: _subtract_w(s, adj)), boot)
        out.append(common | {"method": "certificate minus W", "beta": _subtract_w(sub, adj),
                             "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
        if sub.height >= ADJUST_MIN_PER_COEF * (len(adj) + 1):
            lo, hi, nb = _interval(ev, lambda d: certified(d, lambda s: _ols_t_coef(s, adj)), boot)
            out.append(common | {"method": "certificate adjusting for W",
                                 "beta": _ols_t_coef(sub, adj), "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    return out


ADJUST_MIN_PER_COEF = 3


def rows_for(row: dict, structure: str, bootstrap: bool) -> tuple[list[dict], dict | None]:
    kind = "connected" if structure == "tributary" else "proxy"
    spec = spec_from_row(row, kind=kind)
    if structure == "proxy" and len(spec.w) < 2:
        # One disconnected gauge certifies, but the two-proxy bridge needs
        # two; pairs with fewer than two are recorded and skipped.
        return [], {"pair": row["pair"], "structure": structure, "k": len(spec.w),
                    "safeguards": ",".join(spec.w), "note": "fewer than two disconnected gauges"}
    panel = panel_for(row, kind=kind)
    if panel is None:
        return [], None
    ev, w = panel.events, panel.w_cols
    net = network_for(row["region"])
    areas = _site_areas(row["region"])
    gain = max(row["area_y_km2"] - row["area_x_km2"], 1e-9)
    alpha, _ = hill_estimate(ev["X"].to_numpy())
    water_balance = structure == "tributary"

    base = dict(
        experiment="decomposition", pair=row["pair"], region=row["region"],
        x_name=row["x_name"], y_name=row["y_name"], structure=structure,
        n_events=panel.n_events, k=len(w), reference_area_ratio=row["area_ratio"],
        safeguard_area_coverage=(sum(areas.get(s) or 0.0 for s in spec.w) / gain
                                 if water_balance else None),
        ungauged_gain_fraction=row["ungauged_gain_fraction"], hill_alpha_x=alpha,
        lag_days=panel.lag_xy.lag_days,
    )
    pair_row = base | {
        "safeguards": ",".join(spec.w),
        "safeguard_names": "; ".join(net.name(s) for s in spec.w),
        "safeguard_areas_km2": ",".join(f"{areas.get(s) or 0:.0f}" for s in spec.w),
        "safeguard_distance_to_y_km": ",".join(
            f"{net.distance_km(s, row['y']) or float('nan'):.0f}" for s in spec.w),
        "z": spec.z,
    }
    out: list[dict] = []

    def split(df):
        d = bias_decomposition(df, w)
        return {"beta": d["beta"], "b_W": d["b_W"] if water_balance else None,
                "b_U": d["b_U"] if water_balance else None}

    for name in BASELINES_FOR[structure]:
        fn = BASELINES[name]
        b = fn(ev, w_cols=w)
        if b is None:
            continue
        lo, hi, nb = _interval(ev, lambda d, fn=fn: fn(d, w_cols=w), bootstrap)
        r = base | {"method": name, "n_cert": ev.height, "beta": b,
                    "ci_lo": lo, "ci_hi": hi, "n_boot": nb}
        if name == "naive OLS":
            r |= split(ev)
        out.append(r)

    for q_w in Q_W_GRID:
        q_x, v, n = operating_point(ev, w, q_w)
        if q_x is None:
            out.append(base | {"method": "certificate", "safeguard_quantile": q_w,
                               "note": f"no grid point keeps {MIN_N} certified events"})
            continue
        sub = certified_events(ev, w, level=v, safeguard_quantile=q_w)
        lo, hi, nb = _interval(
            ev, lambda d, v=v, q=q_w: fit_certificate(d, w, level=v, safeguard_quantile=q).beta,
            bootstrap and q_w in (TABULATED_Q_W, FALLBACK_Q_W),
        )
        out.append(base | {"method": "certificate", "safeguard_quantile": q_w,
                           "certification_quantile": q_x, "threshold_v": v, "n_cert": n,
                           "ci_lo": lo, "ci_hi": hi, "n_boot": nb} | split(sub))
        # The single-maximum estimator of the theory: Y/T at the certified
        # event with the largest T (eq:single-maximum-estimator).
        mx = fit_certificate(ev, w, estimator_type="max", level=v, safeguard_quantile=q_w)
        lo, hi, nb = _interval(
            ev, lambda d, v=v, q=q_w: fit_certificate(
                d, w, estimator_type="max", level=v, safeguard_quantile=q).beta,
            bootstrap and q_w in (TABULATED_Q_W, FALLBACK_Q_W),
        )
        out.append(base | {"method": "certificate (single max)", "safeguard_quantile": q_w,
                           "certification_quantile": q_x, "threshold_v": v, "n_cert": mx.n_cert,
                           "beta": mx.beta, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
        # A1 at the identical absolute threshold: the arms differ in the
        # safeguard alone.
        a1 = fit_certificate(ev, [], level=v)
        out.append(base | {"method": "A1 tail only (no safeguard)", "safeguard_quantile": q_w,
                           "certification_quantile": q_x, "threshold_v": v, "n_cert": a1.n_cert}
                   | split(ev.filter(pl.col("X").abs() > v)))
    out.extend(adjusted_rows(row, structure, bootstrap, base))
    return out, pair_row


def main() -> None:
    bootstrap = "--no-bootstrap" not in sys.argv
    rows, pairs = [], []
    for row in design_a_candidates().iter_rows(named=True):
        for structure in STRUCTURES:
            r, p = rows_for(row, structure, bootstrap)
            rows.extend(r)
            if p is not None:
                pairs.append(p)
        print(f"{row['pair']}: {row['x_name']} -> {row['y_name']} done", flush=True)
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUTPUT_DIR / "decomposition.parquet")
    pl.DataFrame(pairs, infer_schema_length=None).write_parquet(OUTPUT_DIR / "decomposition_pairs.parquet")

    read = pl.read_parquet(OUTPUT_DIR / "decomposition.parquet")
    print("\n-- medians by structure and method (certificate at the tabulated q_W) --")
    print(
        read.filter(
            (pl.col("method") != "certificate")
            | (pl.col("safeguard_quantile") == TABULATED_Q_W)
        )
        .filter(pl.col("beta").is_not_null())
        .group_by("structure", "method", maintain_order=True)
        .agg(pl.len().alias("pairs"), pl.col("beta").median(), pl.col("b_W").median(),
             pl.col("b_U").median())
        .sort("structure", "method")
    )


if __name__ == "__main__":
    main()
