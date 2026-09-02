"""The exact dot product is exact, and the exact IV estimate agrees with the
double-precision proximal estimator wherever the latter is well conditioned."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from exact_iv import ExactQuadraticProximal, exact_dot, exact_iv, solve_exact  # noqa: E402
from graphical_toolbox.generate_data.dag import NonlinearParetoDag  # noqa: E402
from graphical_toolbox.models.proximal import LinearProximal  # noqa: E402
from certification_estimators.classical_graphs import PROXY  # noqa: E402


def test_exact_dot_matches_fraction_arithmetic():
    rng = np.random.default_rng(0)
    a = rng.standard_cauchy(2000) * 10.0 ** rng.integers(-5, 12, 2000)
    b = rng.standard_cauchy(2000) * 10.0 ** rng.integers(-5, 12, 2000)
    a[::7] = 0.0
    b[3::11] = -b[3::11]
    expected = sum(Fraction(x) * Fraction(y) for x, y in zip(a.tolist(), b.tolist()))
    assert exact_dot(a, b) == expected


def test_exact_dot_handles_chunk_boundaries_and_single_exponent():
    a = np.full(1000, 3.0)
    b = np.full(1000, 5.0)
    assert exact_dot(a, b) == Fraction(15000)
    assert exact_dot(np.array([1e300, -1e300]), np.array([1e8, 1e8])) == Fraction(0)


def test_solve_exact():
    A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
    b = [Fraction(5), Fraction(10)]
    assert solve_exact(A, b) == [Fraction(1), Fraction(3)]


def test_exact_iv_recovers_a_noiseless_linear_relation():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    x = 2.0 * z + rng.normal(size=500)
    y = 0.5 + 3.0 * x
    ones = np.ones(500)
    beta = exact_iv(np.column_stack([ones, z]), np.column_stack([ones, x]), y)
    assert abs(float(beta[1]) - 3.0) < 1e-12 and abs(float(beta[0]) - 0.5) < 1e-12


def test_exact_quadratic_bridge_agrees_with_lstsq_when_well_conditioned():
    rng = np.random.default_rng(2)
    dag = (
        NonlinearParetoDag.from_dag(PROXY)
        .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
        .set_edge_strength("X", "Y", 1.0)
    )
    data = dag.sample(n=10_000, tail_index=2.5, rng=rng)
    augmented = data.with_columns((pl.col("W") ** 2).alias("W2"), (pl.col("Z") ** 2).alias("Z2"))
    float64 = LinearProximal("X", "Y", ["Z", "Z2"], ["W", "W2"]).fit(augmented).item()
    exact = ExactQuadraticProximal("X", "Y", "Z", "W").fit(data)
    assert abs(exact - float64) < 1e-6 * max(1.0, abs(exact))
    assert abs(exact - 1.0) < 0.05
