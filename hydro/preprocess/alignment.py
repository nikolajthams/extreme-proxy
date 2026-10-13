"""Travel-time alignment of Y to T.

X and Y see the same flood wave on different days. The lag we want is the
*flood-wave celerity* lag, not the bulk-flow lag, so the cross-correlation is
computed only on windows containing the largest X values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hydro.config import Alignment

EPS = 1e-3


@dataclass
class LagEstimate:
    lag_days: int
    correlation: float
    n_tail_days: int
    implied_celerity_lo: float | None
    implied_celerity_hi: float | None
    plausible: bool
    reason: str


def _to_regular_grid(dates: pl.Series, values: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Place a (date, value) series on a contiguous daily grid, nan-filled."""
    d = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    v = values.to_numpy().astype(float)
    lo, hi = d.min(), d.max()
    grid = np.full(hi - lo + 1, np.nan)
    grid[d - lo] = v
    return grid, np.array([lo, hi])


def estimate_lag(
    dates: pl.Series,
    x: pl.Series,
    y: pl.Series,
    cfg: Alignment,
    distance_km: float | None = None,
) -> LagEstimate:
    """Integer lag (days) maximising tail-restricted cross-correlation of log-flow.

    Positive lag means the flood wave reaches Y `lag` days after X.
    """
    gx, _ = _to_regular_grid(dates, x)
    gy, _ = _to_regular_grid(dates, y)
    lx, ly = np.log(gx + EPS), np.log(gy + EPS)

    finite = np.isfinite(gx)
    if finite.sum() < 365:
        return LagEstimate(0, float("nan"), 0, None, None, False, "record too short")

    # Tail window mask: days within `tail_window_days` of a top-`tail_fraction` X day.
    thresh = np.nanquantile(gx, 1 - cfg.tail_fraction)
    peaks = np.where(np.nan_to_num(gx, nan=-np.inf) >= thresh)[0]
    mask = np.zeros(len(gx), dtype=bool)
    w = cfg.tail_window_days
    for p in peaks:
        mask[max(0, p - w) : min(len(mask), p + w + 1)] = True

    best_lag, best_corr = 0, -np.inf
    for lag in range(0, cfg.max_lag_days + 1):
        a = lx[: len(lx) - lag]
        b = ly[lag:]
        m = mask[: len(mask) - lag] & np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50:
            continue
        c = np.corrcoef(a[m], b[m])[0, 1]
        if np.isfinite(c) and c > best_corr:
            best_lag, best_corr = lag, float(c)

    # Plausibility: treat the integer lag as a travel time in
    # [(lag-0.5)+, (lag+0.5)] days and require the implied celerity range to
    # intersect the physical band.
    lo = hi = None
    plausible, reason = True, "ok"
    if distance_km is not None and distance_km > 0:
        d_m = distance_km * 1000.0
        t_hi = (best_lag + 0.5) * 86400.0
        t_lo = max(best_lag - 0.5, 1 / 24) * 86400.0
        lo, hi = d_m / t_hi, d_m / t_lo
        c_lo, c_hi = cfg.celerity_m_per_s
        if hi < c_lo or lo > c_hi:
            plausible = False
            reason = f"implied celerity [{lo:.2f}, {hi:.2f}] m/s outside physical band"

    return LagEstimate(
        lag_days=best_lag,
        correlation=best_corr if np.isfinite(best_corr) else float("nan"),
        n_tail_days=int(mask.sum()),
        implied_celerity_lo=lo,
        implied_celerity_hi=hi,
        plausible=plausible,
        reason=reason,
    )
