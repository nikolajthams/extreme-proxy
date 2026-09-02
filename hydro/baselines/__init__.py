"""Spec 7 baselines, all fitted on the *identical* declustered event table.

Every baseline uses the whole event table, not the certified subset: that is
what makes them baselines. All regressions are through the origin, matching the
certificate estimand Y/X -> beta (spec 6.1); the two proximal baselines are the
exception and keep `LinearProximal`'s own `demean=True`, since a bridge method
is centred by construction and handicapping it would not be a fair comparison.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from graphical_toolbox.models.proximal import LinearProximal


def _ols(df: pl.DataFrame, xs: list[str], y: str) -> np.ndarray | None:
    a = df.select(xs).to_numpy()
    b = df.select(y).to_numpy()
    try:
        return np.linalg.solve(a.T @ a, a.T @ b).flatten()
    except np.linalg.LinAlgError:
        return None


def naive_ols(df: pl.DataFrame, **_) -> float | None:
    """Baseline 1: Y ~ X, no intercept."""
    b = _ols(df, ["X"], "Y")
    return None if b is None else float(b[0])


def ols_adjusting_for_w(df: pl.DataFrame, w_cols: list[str], **_) -> float | None:
    """Baseline 2: Y ~ X + W, no intercept; the coefficient on X."""
    b = _ols(df, ["X", *w_cols], "Y")
    return None if b is None else float(b[0])


def ols_subtracting_w(df: pl.DataFrame, w_cols: list[str], **_) -> float | None:
    """Baseline 2b: (Y - sum W) ~ X, no intercept: the water-balance adjustment.

    Water is conserved, so the structural coefficient of every tributary on the
    downstream gauge is 1 and need not be estimated. Where `ols_adjusting_for_w`
    fits the tributary coefficients freely (and lets small creeks absorb the
    ungauged inflow with coefficients of 5 or more), this fixes them at their
    physical value. Only meaningful when W are the intervening tributaries.
    """
    if not w_cols:
        return None
    d = df.with_columns((pl.col("Y") - pl.sum_horizontal([pl.col(c) for c in w_cols])).alias("Y_adj"))
    b = _ols(d, ["X"], "Y_adj")
    return None if b is None else float(b[0])


def two_sls(df: pl.DataFrame, **_) -> float | None:
    """Baseline 3: just-identified 2SLS with the headwater gauge Z, no intercept.

    Reduces to (Z'Y)/(Z'X). Design B: Z is confounded by the same weather and
    exclusion fails through ungauged inflow, so this is expected to be biased.
    """
    if "Z" not in df.columns:
        return None
    z, x, y = (df[c].to_numpy() for c in ("Z", "X", "Y"))
    den = float(z @ x)
    return None if den == 0 else float(z @ y) / den


def linear_proximal(df: pl.DataFrame, w_cols: list[str], **_) -> float | None:
    """Baseline 4: linear bridge, Z = headwater gauge, W = largest tributary.

    Only one safeguard gauge enters: with a single instrument, bridging several
    W's makes the second-stage Gram matrix singular (every bridged W lives in
    span{X, Z}).
    """
    if "Z" not in df.columns or not w_cols:
        return None
    b = LinearProximal("X", "Y", "Z", w_cols[0]).fit_or_null(df)
    return None if b is None else float(np.asarray(b).flatten()[0])


def valid_bridge_proximal(df: pl.DataFrame, w_cols: list[str], **_) -> float | None:
    """Baseline 5: bridge augmented with squared proxies (`valid_bridge_note.md`).

    Correctly specified for a quadratic confounding channel; the simulation note
    shows it is still inconsistent at tail index alpha <= 2 and degrades
    gradually above it. The Hill estimates are reported alongside so the reader
    can place this data on that axis.
    """
    if "Z" not in df.columns or not w_cols:
        return None
    w = w_cols[0]
    aug = df.with_columns((pl.col("Z") ** 2).alias("Z2"), (pl.col(w) ** 2).alias("W2"))
    b = LinearProximal("X", "Y", ["Z", "Z2"], [w, "W2"]).fit_or_null(aug)
    return None if b is None else float(np.asarray(b).flatten()[0])


def proximal_two_proxies(df: pl.DataFrame, w_cols: list[str], **_) -> float | None:
    """Baseline 6: linear bridge with two pure proxies, Z = W_0 and W = W_1.

    Two gauges disconnected from both X and Y are each independent of X and Y
    given the storm U, so either can play the treatment-side or the
    outcome-side proxy: this is the double-proxy graph of the simulations with
    no edge from the proxy into Y. Only meaningful with a pure-proxy safeguard.
    """
    if len(w_cols) < 2:
        return None
    b = LinearProximal("X", "Y", w_cols[0], w_cols[1]).fit_or_null(df)
    return None if b is None else float(np.asarray(b).flatten()[0])


BASELINES = {
    "naive OLS": naive_ols,
    "OLS adjusting for W": ols_adjusting_for_w,
    "OLS subtracting W": ols_subtracting_w,
    "2SLS (headwater Z)": two_sls,
    "proximal (linear bridge)": linear_proximal,
    "proximal (valid bridge)": valid_bridge_proximal,
    "proximal (two proxies)": proximal_two_proxies,
}
