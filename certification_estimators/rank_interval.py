"""The rank-based confidence interval for beta_0 on a certified sample
(the confidence-interval appendix of the paper).

For k certified observations (T_i, Y_i) let s_ij = (Y_j - Y_i) / (T_j - T_i)
be the m = k(k-1)/2 pairwise slopes. Kendall's statistic between T and the
residual Y - beta T equals the number of pairwise slopes above beta minus
the number below, so the test of beta_0 that rejects when at most r slopes
lie on one side of beta inverts to the interval

    C_{k,gamma} = [s_(r+1), s_(m-r)],    r = r_{k,gamma},

the interval of Sen (1968). Under the null the number of slopes below beta_0
is distributed as the number of inversions of a uniform random permutation
of k elements (Kendall 1938), whose distribution is tabulated here exactly,
so the interval needs no moments and no variance estimate. r_{k,gamma} is the
largest r whose exact level 2 P(D_k <= r) is at most gamma; no interval of
level gamma exists when 2/k! > gamma.

Ties among the T_i carry no information on the slope and are broken at
random, which keeps the null distribution exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial

import numpy as np


@lru_cache(maxsize=None)
def inversion_counts(k: int) -> tuple[int, ...]:
    """The Mahonian numbers: entry d is the number of permutations of k
    elements with exactly d inversions (the coefficients of
    prod_{j=1}^k (1 + q + ... + q^{j-1}))."""
    counts = [1]
    for j in range(1, k + 1):
        new = [0] * (len(counts) + j - 1)
        for d, c in enumerate(counts):
            for s in range(j):
                new[d + s] += c
        counts = new
    return tuple(counts)


def exact_level(k: int, r: int) -> float:
    """gamma_{k,r} = 2 P(D_k <= r): the level of the test that rejects beta
    when at most r pairwise slopes lie on one side of it."""
    return 2 * sum(inversion_counts(k)[: r + 1]) / factorial(k)


def cutoff(k: int, gamma: float) -> tuple[int, float] | None:
    """(r_{k,gamma}, gamma_{k,r}): the largest r with 2r < m and exact level
    at most gamma, or None when no interval of level gamma exists (k too
    small: 2/k! > gamma)."""
    m = comb(k, 2)
    best = None
    for r in range(m // 2 + 1):
        if 2 * r >= m:
            break
        level = exact_level(k, r)
        if level <= gamma:
            best = (r, level)
        else:
            break
    return best


def pairwise_slopes(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(t), 1)
    return (y[j] - y[i]) / (t[j] - t[i])


def kendall_statistic(t: np.ndarray, y: np.ndarray, beta: float) -> int:
    """K(beta) = sum_{i<j} sgn(T_j - T_i) sgn((Y_j - beta T_j) - (Y_i - beta T_i))."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    i, j = np.triu_indices(len(t), 1)
    r = y - beta * t
    return int(np.sum(np.sign(t[j] - t[i]) * np.sign(r[j] - r[i])))


@dataclass(frozen=True)
class RankInterval:
    k: int
    m: int
    r: int
    level: float   # the exact asymptotic level gamma_{k,r} <= gamma
    lo: float
    hi: float

    @property
    def coverage(self) -> float:
        return 1.0 - self.level

    def covers(self, beta: float) -> bool:
        return self.lo <= beta <= self.hi

    @property
    def length(self) -> float:
        return self.hi - self.lo


def rank_interval(t, y, gamma: float, rng: np.random.Generator | None = None) -> RankInterval | None:
    """The interval C_{k,gamma} of the certified sample (t, y), or None when
    k is too small for the level gamma. Ties in t are broken at random
    (`rng`, default seed 0)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    k = len(t)
    if k != len(y):
        raise ValueError("t and y must have the same length")
    c = cutoff(k, gamma)
    if c is None:
        return None
    r, level = c
    if len(np.unique(t)) < k:
        rng = np.random.default_rng(0) if rng is None else rng
        scale = np.ptp(t) if np.ptp(t) > 0 else max(abs(t).max(), 1.0)
        t = t + rng.uniform(-1e-9, 1e-9, size=k) * scale
    s = np.sort(pairwise_slopes(t, y))
    m = len(s)
    return RankInterval(k=k, m=m, r=r, level=level, lo=float(s[r]), hi=float(s[m - r - 1]))
