"""The estimators of the application, all on the daily table of a pair.

The certified estimator (`certification_estimators`):
  * certified OLS      least squares of Y on T through the origin over the
                       certified days {T > v, P in B}; with a single certified
                       day it is the ratio Y/T at that day, the single-maximum
                       estimator of the paper (`MAX` fits that estimator on any
                       certified sample; it is used by the tests only)
The comparison estimators, fitted to every day:
  * naive OLS          least squares of Y on T through the origin
  * proximal bridge    the linear two-proxy bridge with P0 as treatment-inducing
                       and P1 as outcome-inducing proxy, with an intercept
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from certification_estimators.baselines import LinearProximal
from certification_estimators.certificate import Certificate, CompactSet, PerVariableQuantileBall, Unrestricted
from certification_estimators.estimators import fit_on_certified_subset
from hydro.config import CONFIG

EST = CONFIG.estimator
OLS, MAX = "ols", "max"


@dataclass
class Fit:
    beta: float | None
    n_cert: int
    v: float           # treatment threshold, m3/s
    q_t: float | None  # certification quantile behind v, if chosen on the grid


def safeguard(p_cols: list[str], q_p: float) -> CompactSet:
    """B = {p : |p_j| < Q_{q_p}(|P_j|) for every j}; unrestricted without proxies."""
    return PerVariableQuantileBall(p_cols, q_p) if p_cols else Unrestricted()


def threshold(days: pl.DataFrame, p_cols: list[str], q_p: float, q_t: float) -> float:
    """v = the q_t-quantile of T over the safeguard-satisfying days."""
    sub = days.filter(safeguard(p_cols, q_p).is_bounded)
    return float(sub["T"].abs().quantile(q_t)) if sub.height else float("nan")


def certified_events(days: pl.DataFrame, p_cols: list[str], q_p: float, v: float) -> pl.DataFrame:
    """The certified days {T > v, P in B}."""
    return Certificate("T", safeguard(p_cols, q_p)).certified_subset(days, v)


def fit_certified(days: pl.DataFrame, p_cols: list[str], q_p: float, v: float,
                  estimator: str = OLS, q_t: float | None = None) -> Fit:
    sub = certified_events(days, p_cols, q_p, v)
    beta = fit_on_certified_subset(sub, "T", "Y", estimator) if sub.height else None
    return Fit(beta=beta if beta is not None and np.isfinite(beta) else None, n_cert=sub.height, v=v, q_t=q_t)


def operating_point(days: pl.DataFrame, p_cols: list[str], q_p: float,
                    min_events: int = EST.min_events, grid: tuple[float, ...] = EST.q_t_grid
                    ) -> tuple[float, float] | None:
    """(q_t, v): the largest q_t on the grid whose certified sample keeps at
    least `min_events` days, or None if none does."""
    best = None
    for q_t in grid:
        v = threshold(days, p_cols, q_p, q_t)
        if np.isfinite(v) and certified_events(days, p_cols, q_p, v).height >= min_events:
            best = (q_t, v)
    return best


def fit_at_operating_point(days: pl.DataFrame, p_cols: list[str], q_p: float,
                           estimator: str = OLS) -> Fit | None:
    op = operating_point(days, p_cols, q_p)
    if op is None:
        return None
    q_t, v = op
    return fit_certified(days, p_cols, q_p, v, estimator, q_t=q_t)


# -------------------------------------------------------------- comparisons


def naive_ols(df: pl.DataFrame) -> float | None:
    # np.sum (pairwise, single-threaded) rather than a BLAS dot product, whose
    # reduction order can vary between runs on long daily tables.
    t, y = df["T"].to_numpy(), df["Y"].to_numpy()
    den = float(np.sum(t * t))
    return float(np.sum(t * y)) / den if den > 0 else None


def proximal_bridge(df: pl.DataFrame, p_cols: list[str]) -> float | None:
    if len(p_cols) < 2:
        return None
    b = LinearProximal("T", "Y", p_cols[0], p_cols[1]).fit_or_null(df)
    return b if b is not None and np.isfinite(b) else None


# -------------------------------------------------------------- diagnostics


def hill_estimate(x: np.ndarray, k: int | None = None) -> float:
    """Hill estimate of the tail index of a positive sample."""
    x = np.sort(x[np.isfinite(x) & (x > 0)])[::-1]
    n = x.size
    if n < 20:
        return float("nan")
    if k is None:
        k = max(10, int(EST.hill_k_fraction * n))
    k = min(k, n - 1)
    return float(1.0 / np.mean(np.log(x[:k]) - np.log(x[k])))


def year_bootstrap_interval(table: pl.DataFrame, fit_fn, *, seed: int | None = None
                            ) -> tuple[float | None, float | None, int]:
    """Percentile interval from resampling whole years with replacement; the
    only interval reported, indicative rather than calibrated at a handful
    of certified days."""
    rng = np.random.default_rng(EST.seed if seed is None else seed)
    years = table["year"].unique().sort().to_list()
    if len(years) < 5:
        return (None, None, 0)
    by_year = {yr: table.filter(pl.col("year") == yr) for yr in years}
    draws = []
    for _ in range(EST.n_bootstrap):
        pick = rng.choice(len(years), size=len(years), replace=True)
        b = fit_fn(pl.concat([by_year[years[i]] for i in pick]))
        if b is not None and np.isfinite(b):
            draws.append(float(b))
    if len(draws) < 20:
        return (None, None, len(draws))
    a = (1 - EST.interval_level) / 2
    return (float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)), len(draws))
