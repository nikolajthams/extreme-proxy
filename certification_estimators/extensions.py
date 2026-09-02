"""Additive extensions required by the river-discharge application (spec 6.1).

Nothing here modifies the behaviour of `certificate.py` or
`certificate_estimator.py`; the simulation experiments in `experiments/` are
unaffected.

Added:
  * `PerVariableQuantileBall` -- an l_infty safeguard with per-variable
    quantiles, which is what "each safeguard gauge below its own 90th
    percentile" actually means.
  * `TailCertificateEstimator` -- absolute threshold path (a level v in
    physical units) plus the `"trimmed"` estimator type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from certification_estimators.certificate import Certificate, CompactSet
from certification_estimators.certificate_estimator import CertificateEstimator


@dataclass
class PerVariableQuantileBall(CompactSet):
    """S = { w : |w_j| < Q_{q}(|W_j|) for every j }.

    Semantics note (spec 6.2). The existing `QuantileBall` with
    `norm="l_infty"` takes the quantile of the *aggregated* norm
    max_j |W_j|, i.e. a single scalar radius applied to every coordinate. When
    the safeguard gauges have very different scales -- which is always the case
    for river gauges, whose drainage areas differ by an order of magnitude --
    that radius is set by the largest gauge and the small gauges are then
    effectively unconstrained. This class instead thresholds each coordinate at
    its own marginal quantile, which is the semantics we want for the
    application and the one stated in the paper ("all safeguard variables
    individually non-extreme"). `QuantileBall` is left untouched and is still
    reported as a robustness check.
    """

    quantile: float

    @property
    def is_bounded(self) -> pl.Expr:
        if not self.variables:
            return pl.lit(True)
        conds = [pl.col(v).abs() < pl.col(v).abs().quantile(self.quantile) for v in self.variables]
        expr = conds[0]
        for c in conds[1:]:
            expr = expr & c
        return expr


TailEstimatorType = Literal["ols", "median", "max", "trimmed"]


class TailCertificateEstimator(CertificateEstimator):
    """`CertificateEstimator` with an absolute-threshold path and a trimmed mean.

    Exactly one of `certification_quantile` and `level` must be given.
    `level` is the treatment threshold v in the data's own physical units
    (m3/s here), so thresholds can be reported and compared with published
    flood quantiles.

    OLS is through the origin, inherited unchanged: the estimand is the ratio
    Y/X -> beta and an intercept would change it.
    """

    def __init__(
        self,
        e: str,
        safeguard: CompactSet,
        x: str,
        y: str,
        certification_quantile: float | None = None,
        estimator_type: TailEstimatorType = "ols",
        level: float | None = None,
        trim_fraction: float = 0.1,
    ):
        if (certification_quantile is None) == (level is None):
            raise ValueError("Provide exactly one of `certification_quantile` or `level`")
        super().__init__(e, safeguard, x, y, certification_quantile or 0.0, estimator_type)
        self.level = level
        self.trim_fraction = trim_fraction

    # -- subset ---------------------------------------------------------
    def certified_subset(self, data: pl.DataFrame) -> pl.DataFrame:
        cert = Certificate(e=self.e, safeguard=self.safeguard)
        if self.level is not None:
            return cert.certified_subset(data, self.level)
        return cert.quantile_certified_subset(data, quantile=self.certification_quantile)

    def fit(
        self,
        data: pl.DataFrame | pd.DataFrame | np.ndarray,
        names: list[str] | None = None,
        demean: bool = False,
    ) -> np.ndarray:
        data = self.prepare_data(data, names)
        if demean:  # never on by default: demeaning inverts the safeguard set
            from graphical_toolbox.models.abstract_model import demean_data

            data = demean_data(data)
        return self.fit_on_certified_subset(self.certified_subset(data))

    # -- estimators -----------------------------------------------------
    def fit_on_certified_subset(self, data: pl.DataFrame):
        if self.estimator_type != "trimmed":
            return super().fit_on_certified_subset(data)
        ratios = (
            data.select((pl.col(self.y) / pl.col(self.x)).alias("r"))
            .drop_nulls()["r"]
            .to_numpy()
        )
        if ratios.size == 0:
            raise RuntimeError("Empty certified subset")
        k = int(np.floor(self.trim_fraction * ratios.size))
        r = np.sort(ratios)
        r = r[k : ratios.size - k] if ratios.size - 2 * k > 0 else r
        return float(np.mean(r))
