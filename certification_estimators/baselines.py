"""The classical estimators the certified OLS is compared with.

    LinearCIV       two-stage least squares of Y on X with the instrument Z
    LinearProximal  proximal two-stage least squares (Tchetgen Tchetgen et al.,
                    2020) with a linear bridge: the outcome-inducing proxy W is
                    regressed on (X, Z), then Y on X and the fitted bridge

Both demean every column they use, so the fits carry an intercept. Least
squares is solved by QR (`numpy.linalg.lstsq`) rather than by the normal
equations: with heavy-tailed regressors whose components carry different tail
indices, such as W and W^2 of a regularly varying proxy, the condition number
of X'X exhausts double precision at moderate n, and the normal equations
return noise where the estimator itself is still consistent.

Adapted from the graphical-toolbox package
(https://github.com/nikolajthams/graphical-toolbox).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def ols(data: pl.DataFrame, x: list[str], y: str) -> np.ndarray:
    """Least-squares coefficients of y on the columns x, without intercept, by QR."""
    x_arr = data.select(x).to_numpy()
    y_arr = data.select([y]).to_numpy()
    beta, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    return beta


def demean(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return data.with_columns(pl.col(c) - pl.col(c).mean() for c in columns)


def fitted(data: pl.DataFrame, covariates: list[str], target: str) -> pl.Series:
    """The least-squares fit of `target` on `covariates`, as a series named `target`."""
    beta = ols(data, covariates, target).flatten()
    return data.select(
        pl.sum_horizontal(pl.col(c) * beta[j] for j, c in enumerate(covariates)).alias(target)
    ).to_series()


def _as_list(columns: str | list[str]) -> list[str]:
    return [columns] if isinstance(columns, str) else list(columns)


class _Estimator:
    def fit(self, data: pl.DataFrame) -> float:
        raise NotImplementedError

    def fit_or_null(self, data: pl.DataFrame) -> float | None:
        try:
            return self.fit(data)
        except Exception:  # noqa: BLE001 - a failed fit is recorded as missing
            return None


class LinearCIV(_Estimator):
    """Two-stage least squares of y on x with the instruments z."""

    def __init__(self, x: str, y: str, z: str | list[str]):
        self.x, self.y, self.z = x, y, _as_list(z)

    def fit(self, data: pl.DataFrame) -> float:
        data = demean(data, [self.x, self.y, *self.z])
        data = data.with_columns(fitted(data, self.z, self.x).alias("instrumented_x"))
        return float(ols(data, ["instrumented_x"], self.y)[0, 0])


class LinearProximal(_Estimator):
    """Proximal two-stage least squares of y on x with the treatment-inducing
    proxies z and the outcome-inducing proxies w and a linear bridge."""

    def __init__(self, x: str, y: str, z: str | list[str], w: str | list[str]):
        self.x, self.y, self.z, self.w = x, y, _as_list(z), _as_list(w)

    def fit(self, data: pl.DataFrame) -> float:
        data = demean(data, [self.x, self.y, *self.z, *self.w])
        bridges = [f"bridge_{w}" for w in self.w]
        data = data.with_columns(
            fitted(data, [self.x, *self.z], w).alias(bridge) for w, bridge in zip(self.w, bridges)
        )
        return float(ols(data, [self.x, *bridges], self.y)[0, 0])
