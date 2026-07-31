from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

Norm = Literal["l1", "l2", "l_infty"]


@dataclass
class CompactSet(ABC):
    variables: list[str]

    @property
    @abstractmethod
    def is_bounded(self) -> pl.Expr:
       pass


@dataclass
class Unrestricted(CompactSet):
    """The empty safeguard S = {}, i.e. no restriction. Used for the valid-IV anchor E = Z."""
    variables: list[str] = field(default_factory=list)

    @property
    def is_bounded(self) -> pl.Expr:
        return pl.lit(True)


@dataclass
class Ball(CompactSet): 
    radius: float
    norm: Norm = "l1"

    @property
    def is_bounded(self) -> pl.Expr:
       if self.norm == "l1":
          return pl.sum_horizontal(pl.col(v).abs() for v in self.variables) < self.radius
       elif self.norm == "l2":
          return pl.sum_horizontal(pl.col(v) ** 2 for v in self.variables).sqrt() < self.radius
       elif self.norm == "l_infty":
          return pl.max_horizontal(pl.col(v).abs() for v in self.variables) < self.radius
       else:
          raise RuntimeError(f"Norm {self.norm} not supported")

@dataclass
class QuantileBall(CompactSet): 
    quantile: float
    norm: Norm = "l1"

    @property
    def is_bounded(self) -> pl.Expr:
        if self.norm == "l1":
            expr = pl.sum_horizontal(pl.col(v).abs() for v in self.variables)
        elif self.norm == "l2":
            expr = pl.sum_horizontal(pl.col(v) ** 2 for v in self.variables).sqrt()
        elif self.norm == "l_infty":
            expr = pl.max_horizontal(pl.col(v).abs() for v in self.variables)
        else:
            raise RuntimeError(f"Norm {self.norm} not supported")

        return expr < expr.quantile(self.quantile)

class Certificate:
    def __init__(
            self,
            e: str,
            safeguard: CompactSet,
            ):
       self.e = e
       self.safeguard = safeguard

    def _check_names_valid(self, df: pl.DataFrame) -> None:
        required = {self.e, *self.safeguard.variables}
        missing = required - set(df.columns)
        assert not missing, f"Not all columns present in the data. Missing columns: {missing}"

    def certified_subset(self, df: pl.DataFrame, tau: float) -> pl.DataFrame:
        self._check_names_valid(df)

        return df.filter((pl.col(self.e).abs() > tau) & self.safeguard.is_bounded)

    def quantile_certified_subset(self, df: pl.DataFrame, quantile: float) -> pl.DataFrame:
        self._check_names_valid(df)
        tau = df.filter(self.safeguard.is_bounded).select(pl.col(self.e).abs().quantile(quantile)).item()

        return self.certified_subset(df, tau)


