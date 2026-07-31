from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from graphical_toolbox.models.abstract_model import AbstractEstimator, demean_data

from certification_estimators.certificate import (
    Certificate,
    CompactSet,
)

CertificateEstimatorType = Literal["ols", "median", "max"]
class CertificateEstimator(AbstractEstimator):
    def __init__(self, e: str, safeguard: CompactSet, x: str, y: str, certification_quantile: float, estimator_type: CertificateEstimatorType="ols"):
        self.e = e
        self.x = x
        self.y = y
        self.safeguard = safeguard
        self.certification_quantile = certification_quantile
        self.estimator_type = estimator_type

    def fit(self, data: pl.DataFrame | pd.DataFrame | np.ndarray, names: list[str] | None = None, demean: bool=False) -> np.ndarray:
        data = self.prepare_data(data, names)
        if demean:
            data = demean_data(data)
        certified_subset = Certificate(e=self.e, safeguard=self.safeguard).quantile_certified_subset(data, quantile=self.certification_quantile)
        beta = self.fit_on_certified_subset(certified_subset)

        return beta

    def fit_or_null(self, data: pl.DataFrame | pd.DataFrame | np.ndarray, names: list[str] | None = None, demean: bool=False) -> np.ndarray | None:
        try:
            return self.fit(data, names, demean)
        except:
            return None


    def fit_on_certified_subset(self, data: pl.DataFrame) -> np.ndarray:
        if self.estimator_type == "ols":
            beta = self.ols(data, [self.x], [self.y]).item()
        elif self.estimator_type == "median":
            beta = (
                data
                .select(
                    (pl.col(self.y) / pl.col(self.x))
                    .median()
                ).item()
            )
        elif self.estimator_type == "max":
            beta = (
                data
                .select(
                    (pl.col(self.y) / pl.col(self.x))
                    .filter(pl.col(self.x) == pl.col(self.x).max())
                    .first()
                ).item()
            )
        else:
            raise RuntimeError(f"Estimator type {self.estimator_type} unknown")

        return beta


