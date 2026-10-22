"""The certified estimators of beta_0 on a certified sample (T_i, Y_i).

    certified OLS    least squares of Y on T through the origin
    single maximum   Y/T at the certified observation with the largest T,
                     which is the certified OLS of a single observation
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from certification_estimators.certificate import Certificate, CompactSet

EstimatorType = Literal["ols", "max"]


def certified_ols(t: np.ndarray, y: np.ndarray) -> float:
    """Least squares of y on t through the origin, solved by QR."""
    t, y = np.asarray(t, float).reshape(-1, 1), np.asarray(y, float).reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(t, y, rcond=None)
    return float(beta.item())


def single_maximum(t: np.ndarray, y: np.ndarray) -> float:
    """The ratio y/t at the (first) observation with the largest t."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    i = int(np.argmax(t))
    return float(y[i] / t[i])


def fit_on_certified_subset(sub: pl.DataFrame, treatment: str, outcome: str,
                            estimator: EstimatorType = "ols") -> float:
    t, y = sub[treatment].to_numpy(), sub[outcome].to_numpy()
    if estimator == "ols":
        return certified_ols(t, y)
    if estimator == "max":
        return single_maximum(t, y)
    raise ValueError(f"unknown estimator {estimator!r}")


class CertifiedEstimator:
    """A certified estimator with a fixed number of certified observations: the
    `n_certified` observations with the largest source value among those with
    the safeguard in its set. Keeping that number fixed as n grows sends the
    certification threshold to infinity like n^{1/alpha}, the regime of the
    paper's convergence rates; a fixed quantile of the source would leave a
    bias of order 1/threshold. Used by the simulations."""

    def __init__(self, source: str, safeguard: CompactSet, treatment: str, outcome: str,
                 n_certified: int, estimator: EstimatorType = "ols"):
        self.certificate = Certificate(source, safeguard)
        self.treatment, self.outcome = treatment, outcome
        self.n_certified = n_certified
        self.estimator = estimator

    @property
    def source(self) -> str:
        return self.certificate.e

    @property
    def safeguard(self) -> CompactSet:
        return self.certificate.safeguard

    def certified_subset(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.certificate.top_k_certified_subset(data, self.n_certified)

    def fit(self, data: pl.DataFrame) -> float:
        return fit_on_certified_subset(self.certified_subset(data), self.treatment, self.outcome, self.estimator)

    def fit_or_null(self, data: pl.DataFrame) -> float | None:
        try:
            return self.fit(data)
        except Exception:  # noqa: BLE001 - a failed fit is recorded as missing
            return None
