"""The certifying event of a source-safeguard pair (E, S).

A certificate selects the observations whose source E exceeds a threshold v
while the safeguard variables S lie in a compact set B, the event
A_v = {E > v, S in B} of the paper. The set B is a quantile ball of the
safeguard, so that it is fixed by the data alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl


@dataclass
class CompactSet(ABC):
    """A set B for the safeguard, as a polars expression that is true on the
    observations with S in B."""

    variables: list[str]

    @property
    @abstractmethod
    def is_bounded(self) -> pl.Expr: ...


@dataclass
class Unrestricted(CompactSet):
    """The empty safeguard S = {}: no restriction, as for the pair (Z, {}) of
    the valid-instrument design."""

    variables: list[str] = field(default_factory=list)

    @property
    def is_bounded(self) -> pl.Expr:
        return pl.lit(True)


@dataclass
class QuantileBall(CompactSet):
    """The l1-ball {s : ||s||_1 < r} whose radius r is the `quantile` of
    ||S||_1 over all observations (the simulations: the 10% quantile)."""

    quantile: float

    @property
    def is_bounded(self) -> pl.Expr:
        norm = pl.sum_horizontal(pl.col(v).abs() for v in self.variables)
        return norm < norm.quantile(self.quantile)


@dataclass
class PerVariableQuantileBall(CompactSet):
    """{s : |s_j| < Q_q(|S_j|) for every j}: every safeguard variable below its
    own `quantile` (the application, whose proxy gauges differ in scale by an
    order of magnitude, so that a single radius would be set by the largest)."""

    quantile: float

    @property
    def is_bounded(self) -> pl.Expr:
        if not self.variables:
            return pl.lit(True)
        return pl.all_horizontal(
            pl.col(v).abs() < pl.col(v).abs().quantile(self.quantile) for v in self.variables
        )


class Certificate:
    """The certifying events of the pair (E, S) with the safeguard set B.

    The source enters through |E|; every source of the paper is non-negative,
    so the certified subset is {E > v, S in B}."""

    def __init__(self, e: str, safeguard: CompactSet):
        self.e = e
        self.safeguard = safeguard

    def _check_columns(self, df: pl.DataFrame) -> None:
        missing = {self.e, *self.safeguard.variables} - set(df.columns)
        if missing:
            raise KeyError(f"columns missing from the data: {sorted(missing)}")

    def certified_subset(self, df: pl.DataFrame, v: float) -> pl.DataFrame:
        """The observations with |E| > v and S in B."""
        self._check_columns(df)
        return df.filter((pl.col(self.e).abs() > v) & self.safeguard.is_bounded)

    def top_k_certified_subset(self, df: pl.DataFrame, k: int) -> pl.DataFrame:
        """The k observations with the largest |E| among those with S in B, in
        decreasing order of |E| (a stable sort, so the result is reproducible)."""
        self._check_columns(df)
        return (
            df.filter(self.safeguard.is_bounded)
            .sort(pl.col(self.e).abs(), descending=True, maintain_order=True)
            .head(k)
        )
