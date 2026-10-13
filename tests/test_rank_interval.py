"""Tests of the rank-based confidence interval on a certified sample."""

from __future__ import annotations

from math import comb, factorial

import numpy as np
import pytest

from certification_estimators.rank_interval import (
    cutoff, exact_level, inversion_counts, kendall_statistic, rank_interval,
)


def test_inversion_counts_are_the_mahonian_numbers():
    assert inversion_counts(3) == (1, 2, 2, 1)
    assert inversion_counts(4) == (1, 3, 5, 6, 5, 3, 1)
    for k in range(2, 9):
        c = inversion_counts(k)
        assert sum(c) == factorial(k) and len(c) == comb(k, 2) + 1
        assert c == c[::-1]  # D_k is symmetric about m/2


def test_cutoffs_match_the_values_quoted_in_the_paper():
    # gamma = 0.05: r_{k,gamma} for k = 5, ..., 12, and no interval for k <= 4
    assert [cutoff(k, 0.05)[0] for k in range(5, 13)] == [0, 1, 3, 5, 8, 11, 14, 18]
    assert cutoff(4, 0.05) is None and cutoff(3, 0.05) is None
    # gamma = 0.1 (the application's level): the range of all slopes at k = 4
    assert cutoff(4, 0.1) == (0, pytest.approx(2 / 24))
    assert [cutoff(k, 0.1)[0] for k in range(5, 13)] == [1, 2, 4, 6, 9, 12, 16, 20]
    for k in range(4, 13):
        r, level = cutoff(k, 0.1)
        assert level == exact_level(k, r) <= 0.1 < exact_level(k, r + 1)


def test_interval_is_the_inversion_of_the_kendall_test():
    rng = np.random.default_rng(1)
    for k in (5, 8, 10):
        t, y = rng.pareto(2.0, k) + 1, rng.normal(size=k)
        ci = rank_interval(t, y, 0.1)
        bound = ci.m - 2 * ci.r
        grid = np.linspace(ci.lo - 2, ci.hi + 2, 4001)
        inside = np.array([abs(kendall_statistic(t, y, b)) < bound for b in grid])
        assert np.all(inside == ((grid >= ci.lo - 1e-12) & (grid <= ci.hi + 1e-12)))
        assert ci.covers(np.median((y[:, None] - y[None, :])[np.triu_indices(k, 1)]
                                   / (t[:, None] - t[None, :])[np.triu_indices(k, 1)]))


def test_coverage_under_independence_equals_the_exact_level():
    rng = np.random.default_rng(2)
    k, reps = 7, 4000
    r, level = cutoff(k, 0.1)
    hits = sum(rank_interval(rng.pareto(1.5, k) + 1, 2.0 * 0 + rng.standard_t(2, k), 0.1).covers(0.0)
               for _ in range(reps))
    assert hits / reps == pytest.approx(1 - level, abs=0.02)


def test_ties_in_t_are_broken_at_random_and_reproducibly():
    t = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    y = np.array([0.5, 1.5, 2.0, 3.2, 2.8, 4.1])
    a, b = rank_interval(t, y, 0.1), rank_interval(t, y, 0.1)
    assert a == b and np.isfinite([a.lo, a.hi]).all()
    assert rank_interval([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.1) is None
