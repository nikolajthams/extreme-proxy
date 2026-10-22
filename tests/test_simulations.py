"""Tests of the simulation scripts in `simulations/`: the designs encode the
source-safeguard pairs the paper names, a repetition is reproducible from its
index alone, the summary statistics are what the figure plots, and the stored
estimates are reproduced by the current code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from certification_estimators.certificate import QuantileBall, Unrestricted
from certification_estimators.estimators import CertifiedEstimator
from simulations import distributional_failure as dist
from simulations import iv_and_proxy as sim

ROOT = Path(__file__).resolve().parents[1]


def _certificate(design: sim.Design, label: str) -> CertifiedEstimator:
    return design.estimators[label]()


def test_designs_use_the_pairs_selected_by_the_criterion():
    by_key = {d.key: d for d in sim.DESIGNS}
    assert list(by_key) == ["valid_iv", "invalid_iv", "linear_proxy", "nonlinear_proxy"]
    # the figure shows the designs in which the classical assumption holds first
    assert sim.PANEL_ORDER == ["valid_iv", "linear_proxy", "invalid_iv", "nonlinear_proxy"]
    assert [d.title[:3] for d in sim.PANELS] == ["(a)", "(b)", "(c)", "(d)"]

    valid = _certificate(by_key["valid_iv"], sim.CERTIFIED)
    assert valid.source == "Z" and isinstance(valid.safeguard, Unrestricted)

    invalid = _certificate(by_key["invalid_iv"], sim.CERTIFIED)
    assert invalid.source == sim.TREATMENT and invalid.safeguard.variables == ["Z"]
    disqualified = _certificate(by_key["invalid_iv"], sim.DISQUALIFIED)
    assert disqualified.source == "Z" and isinstance(disqualified.safeguard, Unrestricted)

    for key in ("linear_proxy", "nonlinear_proxy"):
        cert = _certificate(by_key[key], sim.CERTIFIED)
        assert cert.source == "Z" and cert.safeguard.variables == ["W"]
        assert by_key[key].classical == sim.BRIDGE
    assert by_key["valid_iv"].classical == sim.TSLS


def test_designs_certify_three_observations_with_the_certified_ols():
    for design in sim.DESIGNS:
        cert = _certificate(design, sim.CERTIFIED)
        assert cert.n_certified == sim.N_CERTIFIED == 3
        assert cert.estimator == "ols"


def test_repetition_is_reproducible_and_finite():
    first = sim.run_repetition(3, sample_sizes=[20_000], tail_indices=[2.5])
    second = sim.run_repetition(3, sample_sizes=[20_000], tail_indices=[2.5])
    assert first == second
    estimates = [row["estimate"] for row in first]
    assert all(e is not None and np.isfinite(e) for e in estimates)
    # 2 + 3 + 2 + 4 estimators, one cell each
    assert len(first) == 11
    assert sim.run_repetition(4, sample_sizes=[20_000], tail_indices=[2.5]) != first


def test_repetition_is_reproducible_across_processes():
    """String hashing is randomised per process; the seeded repetition must not depend on it."""
    code = ("import json; from simulations import iv_and_proxy as sim; "
            "print(json.dumps(sim.run_repetition(5, sample_sizes=[20_000], tail_indices=[2.5])))")
    outputs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                cwd=ROOT, check=True)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_summary_statistics():
    rows = [
        {"design": "d", "estimator": "e", "tail_index": 1.5, "sample_size": 10, "repetition": r, "estimate": v}
        for r, v in enumerate([1.0, 1.2, 0.95, None])
    ]
    stats = sim.summarise(pl.DataFrame(rows))
    assert stats.height == 1
    row = stats.row(0, named=True)
    assert row["failed_fits"] == 1
    assert row["repetitions"] == 4
    # |errors| among the non-null estimates: 0, 0.2, 0.05 -> median 0.05
    assert abs(row["median_abs_error"] - 0.05) < 1e-12
    assert abs(row["failure_rate"] - 1 / 3) < 1e-12


def test_certified_estimator_uses_the_k_largest_source_values_in_the_safeguard_set():
    rng = np.random.default_rng(0)
    n = 400
    data = pl.DataFrame({"Z": rng.pareto(1.5, n), "W": rng.pareto(1.5, n),
                         "X": rng.pareto(1.5, n) + 1, "Y": rng.pareto(1.5, n)})
    ball = QuantileBall(["W"], 0.5)
    est = CertifiedEstimator("Z", ball, "X", "Y", n_certified=7, estimator="ols")

    certified = est.certified_subset(data)
    in_ball = data.filter(pl.col("W").abs() < pl.col("W").abs().quantile(0.5))
    expected = in_ball.sort(pl.col("Z").abs(), descending=True).head(7)
    assert certified.height == 7
    assert sorted(certified["Z"].to_list()) == sorted(expected["Z"].to_list())

    # through-the-origin least squares of Y on X over the certified rows
    x, y = expected["X"].to_numpy(), expected["Y"].to_numpy()
    assert abs(est.fit(data) - float(np.sum(x * y) / np.sum(x * x))) < 1e-12

    # with a single certified observation it is the ratio Y/X at the largest source,
    # which is the single-maximum estimator with the same certifying event
    one = CertifiedEstimator("Z", ball, "X", "Y", n_certified=1)
    top = expected.row(0, named=True)
    assert abs(one.fit(data) - top["Y"] / top["X"]) < 1e-12
    # the single-maximum estimator on the same seven observations: Y/X at the largest treatment
    single = CertifiedEstimator("Z", ball, "X", "Y", n_certified=7, estimator="max")
    largest = expected.sort("X", descending=True).row(0, named=True)
    assert abs(single.fit(data) - largest["Y"] / largest["X"]) < 1e-12


@pytest.mark.skipif(not (sim.OUTPUT / "estimates.parquet").exists(), reason="stored estimates not present")
def test_stored_estimates_are_reproduced():
    """The first cell of repetition 0 consumes the random stream before any other
    cell, so it can be recomputed alone and compared with the stored estimates."""
    design, tail_index, n = sim.DESIGNS[0], sim.TAIL_INDICES[0], sim.SAMPLE_SIZES[0]
    stored = pl.read_parquet(sim.OUTPUT / "estimates.parquet").filter(
        (pl.col("repetition") == 0) & (pl.col("design") == design.key)
        & (pl.col("tail_index") == tail_index) & (pl.col("sample_size") == n)
    )
    fresh = sim.run_repetition(0, sample_sizes=[n], tail_indices=[tail_index], designs=[design])
    assert {r["estimator"] for r in fresh} == set(stored["estimator"]) == {sim.TSLS, sim.CERTIFIED}
    for r in fresh:
        assert r["estimate"] == pytest.approx(stored.filter(pl.col("estimator") == r["estimator"])["estimate"][0], rel=1e-9)


@pytest.mark.skipif(not (dist.OUTPUT / "estimates.parquet").exists(), reason="stored estimates not present")
def test_stored_distributional_failure_estimates_are_reproduced():
    n = sim.SAMPLE_SIZES[0]
    stored = pl.read_parquet(dist.OUTPUT / "estimates.parquet").filter(
        (pl.col("repetition") == 0) & (pl.col("noise") == "gaussian") & (pl.col("sample_size") == n)
    )
    fresh = dist.run_repetition(0, gaussian_sizes=[n], student_sizes=[], degrees_of_freedom=[])
    assert {r["estimator"] for r in fresh} == set(stored["estimator"]) == {sim.BRIDGE, sim.CERTIFIED}
    for r in fresh:
        assert r["estimate"] == pytest.approx(stored.filter(pl.col("estimator") == r["estimator"])["estimate"][0], rel=1e-9)
