"""Thin wrappers that run the certificate estimator on a `Panel` and always
return the effective certified sample size alongside beta (spec 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import polars as pl

from certification_estimators.certificate import Certificate, QuantileBall, Unrestricted
from certification_estimators.extensions import (
    PerVariableQuantileBall,
    TailCertificateEstimator,
)
from hydro.config import CONFIG, HydroConfig


@dataclass
class Fit:
    beta: float | None
    n_cert: int
    threshold_v: float | None
    estimator_type: str
    safeguard: str
    certification_quantile: float | None
    safeguard_quantile: float | None


def make_safeguard(w_cols: list[str], q: float, kind: str = "per_variable"):
    """Safeguard set S.

    kind="per_variable": each safeguard gauge below its own q-quantile
                         (the semantics stated in the paper; see
                         `PerVariableQuantileBall`).
    kind="l_infty"/"l1": the existing `QuantileBall`, quantile of the
                         aggregated norm -- reported as a robustness check.
    kind="none":         no safeguard at all (ablation A1, tail only).
    """
    if kind == "none" or not w_cols:
        return Unrestricted()
    if kind == "per_variable":
        return PerVariableQuantileBall(w_cols, q)
    return QuantileBall(w_cols, q, kind)  # type: ignore[arg-type]


def certified_threshold(
    events: pl.DataFrame, w_cols: list[str], q_x: float, q_w: float, kind: str
) -> float:
    """The treatment threshold v implied by a certification quantile.

    Follows the `quantile_certified_subset` convention: the quantile of X is
    taken on the safeguard-satisfying subsample, not on the full sample.
    """
    s = make_safeguard(w_cols, q_w, kind)
    sub = events.filter(s.is_bounded)
    if sub.height == 0:
        return float("nan")
    return float(sub.select(pl.col("X").abs().quantile(q_x)).item())


def fit_certificate(
    events: pl.DataFrame,
    w_cols: list[str],
    *,
    estimator_type: str = "ols",
    certification_quantile: float | None = None,
    level: float | None = None,
    safeguard_quantile: float = 0.75,
    safeguard_kind: str = "per_variable",
    anchor: str = "X",
) -> Fit:
    s = make_safeguard(w_cols, safeguard_quantile, safeguard_kind)
    est = TailCertificateEstimator(
        anchor, s, "X", "Y",
        certification_quantile=certification_quantile,
        estimator_type=estimator_type,  # type: ignore[arg-type]
        level=level,
        trim_fraction=CONFIG.estimator.trim_fraction,
    )
    subset = est.certified_subset(events)
    v = level
    if v is None:
        sub = events.filter(s.is_bounded)
        v = float(sub.select(pl.col(anchor).abs().quantile(certification_quantile)).item()) \
            if sub.height else float("nan")
    try:
        beta = float(est.fit_on_certified_subset(subset))
    except Exception:
        beta = None
    return Fit(
        beta=beta,
        n_cert=subset.height,
        threshold_v=v,
        estimator_type=estimator_type,
        safeguard=safeguard_kind if w_cols else "none",
        certification_quantile=certification_quantile,
        safeguard_quantile=safeguard_quantile if w_cols else None,
    )


def certified_events(
    events: pl.DataFrame, w_cols: list[str], *, certification_quantile=None, level=None,
    safeguard_quantile: float = 0.75, safeguard_kind: str = "per_variable", anchor: str = "X",
) -> pl.DataFrame:
    s = make_safeguard(w_cols, safeguard_quantile, safeguard_kind)
    cert = Certificate(e=anchor, safeguard=s)
    if level is not None:
        return cert.certified_subset(events, level)
    return cert.quantile_certified_subset(events, certification_quantile)


# --------------------------------------------------------------- diagnostics


def bias_decomposition(events: pl.DataFrame, w_cols: list[str]) -> dict[str, float]:
    """Split a through-the-origin slope of Y on X into what the safeguard sees.

    Water is conserved, so Y = X + sum(W) + U with U the inflow the safeguard
    gauges do not measure (other tributaries, ungauged area, timing). For the
    through-the-origin slope this is an identity, not a model:

        <X, Y>/<X, X> = 1 + <X, sum W>/<X, X> + <X, U>/<X, X>
                      = 1 + b_W + b_U.

    b_W is the part of the naive bias carried by the gauged tributaries, b_U the
    part they do not see. Evaluated on a certified subset it says which part
    the certificate removed. With a pure-proxy safeguard (gauges that feed
    neither X nor Y) the split is not a water balance and only `beta` is
    meaningful; callers should not report b_W then.
    """
    x = events["X"].to_numpy()
    y = events["Y"].to_numpy()
    sw = events.select(w_cols).to_numpy().sum(axis=1) if w_cols else np.zeros_like(x)
    xx = float(x @ x)
    b_w = float(x @ sw) / xx
    b_u = float(x @ (y - x - sw)) / xx
    return {"beta": 1.0 + b_w + b_u, "b_W": b_w, "b_U": b_u}


def hill_estimate(x: np.ndarray, k: int | None = None,
                  cfg: HydroConfig = CONFIG) -> tuple[float, int]:
    """Hill estimate of the tail index alpha of a positive sample."""
    x = np.sort(x[np.isfinite(x) & (x > 0)])[::-1]
    n = x.size
    if n < 20:
        return (float("nan"), 0)
    if k is None:
        k = max(10, int(cfg.uncertainty.hill_k_fraction * n))
    k = min(k, n - 1)
    alpha = 1.0 / np.mean(np.log(x[:k]) - np.log(x[k]))
    return (float(alpha), int(k))


def hill_path(x: np.ndarray, ks: list[int]) -> list[tuple[int, float]]:
    return [(k, hill_estimate(x, k)[0]) for k in ks]


def year_bootstrap_interval(
    events: pl.DataFrame,
    fit_fn,
    *,
    n_boot: int | None = None,
    level: float | None = None,
    seed: int | None = None,
) -> tuple[float | None, float | None, int]:
    """Year-level bootstrap (spec 6.4 option 3).

    Whole hydrological years are resampled with replacement, which respects
    within-year serial dependence while staying transparent. This is the ONLY
    interval we report: subsampling and the stationary block bootstrap were cut
    for time, and the interval should be read as indicative, not as a
    calibrated confidence statement under infinite variance.
    """
    cfg = CONFIG.uncertainty
    n_boot = n_boot or cfg.n_bootstrap
    level = level or cfg.level
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    years = events["year"].unique().sort().to_list()
    if len(years) < 5:
        return (None, None, 0)
    by_year = {y: events.filter(pl.col("year") == y) for y in years}

    draws: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(len(years), size=len(years), replace=True)
        sample = pl.concat([by_year[years[i]] for i in pick])
        b = fit_fn(sample)
        if b is not None and np.isfinite(b):
            draws.append(float(b))
    if len(draws) < 20:
        return (None, None, len(draws))
    a = (1 - level) / 2
    return (float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)), len(draws))
