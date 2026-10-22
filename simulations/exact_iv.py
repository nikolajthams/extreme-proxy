"""Exactly identified instrumental-variable estimates in exact rational arithmetic.

The correctly specified quadratic proximal bridge in the nonlinear double-proxy
design regresses Y on (T, W, W^2) with instruments (T, Z, Z^2). Because W and Z
are regularly varying, the squared columns have tail index alpha/2 and the
design matrix becomes ill-conditioned as n grows (condition number of order
n^2.7 at alpha = 1.5). A double-precision least-squares solve then returns
noise although the estimator is consistent, which a recomputation in 60-digit
arithmetic confirms: the apparent failure is conditioning, not statistics.
With as many instruments as regressors, two-stage least squares equals
(Z'X)^{-1} Z'y, so it suffices to accumulate the moment matrices exactly:
every double is an integer times a power of two, so products and sums of
products are exact integers, and the 4x4 solve is done in `fractions.Fraction`.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import polars as pl

_HALF = 27
_MASK = (1 << _HALF) - 1
_CHUNK = 256  # 256 * 2**54 < 2**63, so chunk sums of int64 partial products cannot overflow


def _mantissa_exponent(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a = M * 2**e exactly, with integer |M| < 2**53."""
    m, e = np.frexp(np.asarray(a, dtype=np.float64))
    return np.ldexp(m, 53).astype(np.int64), e.astype(np.int64) - 53


def _exact_scaled_sum(values: np.ndarray, exponents: np.ndarray, order: np.ndarray) -> tuple[int, int]:
    """sum_i values[i] * 2**exponents[i] as (integer, emin) meaning integer * 2**emin."""
    values, exponents = values[order], exponents[order]
    uniq, start = np.unique(exponents, return_index=True)
    ends = np.append(start[1:], len(exponents))
    emin = int(uniq[0])
    total = 0
    for exponent, s, t in zip(uniq.tolist(), start.tolist(), ends.tolist()):
        chunks = np.add.reduceat(values[s:t], np.arange(0, t - s, _CHUNK))
        total += sum(int(c) for c in chunks) << (exponent - emin)
    return total, emin


def exact_dot(a: np.ndarray, b: np.ndarray) -> Fraction:
    """The exact value of sum_i a_i * b_i for float64 arrays."""
    ma, ea = _mantissa_exponent(a)
    mb, eb = _mantissa_exponent(b)
    e = ea + eb
    order = np.argsort(e, kind="stable")
    ha, la = ma >> _HALF, ma & _MASK  # M = h * 2**27 + l with 0 <= l < 2**27, |h| < 2**26
    hb, lb = mb >> _HALF, mb & _MASK
    total = Fraction(0)
    for partial, shift in ((ha * hb, 2 * _HALF), (ha * lb + la * hb, _HALF), (la * lb, 0)):
        integer, emin = _exact_scaled_sum(partial, e + shift, order)
        total += Fraction(integer) * (Fraction(2) ** emin)
    return total


def solve_exact(A: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """Gaussian elimination with partial pivoting over the rationals."""
    k = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(M[r][col]))
        if M[pivot][col] == 0:
            raise ZeroDivisionError("singular moment matrix")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(k):
            if r != col and M[r][col] != 0:
                f = M[r][col] / M[col][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][k] / M[i][i] for i in range(k)]


def exact_iv(instruments: np.ndarray, regressors: np.ndarray, y: np.ndarray) -> list[Fraction]:
    """(Z'X)^{-1} Z'y computed exactly; columns of `instruments` and `regressors` must match in number."""
    assert instruments.shape == regressors.shape
    ZX = [[exact_dot(instruments[:, i], regressors[:, j]) for j in range(regressors.shape[1])]
          for i in range(instruments.shape[1])]
    Zy = [exact_dot(instruments[:, i], y) for i in range(instruments.shape[1])]
    return solve_exact(ZX, Zy)


class ExactQuadraticProximal:
    """Proximal two-stage least squares with bridge h(W) = c0 + c1 W + c2 W^2 and
    proxies (Z, Z^2), computed exactly. Includes an intercept, which is what
    demeaning every column before the two stages amounts to."""

    def __init__(self, x: str, y: str, z: str, w: str):
        self.x, self.y, self.z, self.w = x, y, z, w

    def fit(self, data: pl.DataFrame) -> float:
        x = data[self.x].to_numpy()
        z = data[self.z].to_numpy()
        w = data[self.w].to_numpy()
        y = data[self.y].to_numpy()
        ones = np.ones_like(x)
        instruments = np.column_stack([ones, x, z, z * z])
        regressors = np.column_stack([ones, x, w, w * w])
        return float(exact_iv(instruments, regressors, y)[1])

    def fit_or_null(self, data: pl.DataFrame) -> float | None:
        try:
            return self.fit(data)
        except (ZeroDivisionError, ValueError):
            return None
