"""Tests for experiments/combine_estimators.py: the split of the sample at the
certification threshold, the bridge against the toolbox's LinearProximal, the
least-squares pieces, the precision weight, and the reproducibility of a
replication."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import combine_estimators as ce  # noqa: E402


def _data(n: int = 5000, seed: int = 0, tail_index: float = 2.5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dag = (ce.ParetoDag.from_dag(ce.PROXY)
           .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
           .set_edge_strength(ce.TREATMENT, ce.OUTCOME, ce.TRUE_BETA))
    return dag.sample(n=n, tail_index=tail_index, rng=rng)


def test_split_certifies_k_observations_above_the_threshold_and_the_body_below():
    data = _data()
    certified, body, v = ce.split_sample(data, k=7)
    in_ball = data.filter(pl.col("W").abs() < pl.col("W").abs().quantile(ce.SAFEGUARD_QUANTILE))
    z_desc = np.sort(in_ball["Z"].abs().to_numpy())[::-1]
    assert certified.height == 7
    assert v == z_desc[7]  # the (k + 1)th largest source value in the ball
    assert (certified["Z"].abs().to_numpy() > v).all()
    assert (body["Z"].abs().to_numpy() < v).all()
    assert body.height == int((data["Z"].abs() < v).sum())
    # exactly k observations are certified at v, one more at any smaller threshold
    assert in_ball.filter(pl.col("Z").abs() > v).height == 7
    assert in_ball.filter(pl.col("Z").abs() > v * (1 - 1e-12)).height == 8
    # the certified rows and the body are disjoint, and the uncertified tail is in neither
    assert certified.join(body, on=data.columns, how="inner").height == 0
    uncertified_tail = data.filter((pl.col("Z").abs() > v) & ~(pl.col("W").abs() < pl.col("W").abs().quantile(ce.SAFEGUARD_QUANTILE)))
    assert certified.height + body.height + uncertified_tail.height == data.height - 1


def test_bridge_matches_the_toolbox_point_estimate():
    from graphical_toolbox.models.proximal import LinearProximal

    data = _data()
    beta, var = ce.proximal_bridge(data)
    reference = float(np.asarray(LinearProximal(ce.TREATMENT, ce.OUTCOME, "Z", "W").fit(data)).reshape(-1)[0])
    assert abs(beta - reference) < 1e-8 * max(1.0, abs(reference))
    assert var > 0


def test_ols_with_intercept_matches_polyfit_and_its_sandwich():
    rng = np.random.default_rng(1)
    t = rng.pareto(2.5, 10) + 1
    y = 1.0 + 0.5 * t + rng.normal(size=10)
    beta, var = ce.ols_with_intercept(t, y)
    assert abs(beta - np.polyfit(t, y, 1)[0]) < 1e-12
    tc = t - t.mean()
    r = (y - y.mean()) - beta * tc
    assert abs(var - np.sum(tc**2 * r**2) / np.sum(tc**2) ** 2 * 10 / 8) < 1e-12


def test_precision_weight_minimises_the_variance_of_the_combination():
    var_cert, var_bulk = 1.0, 3.0
    w = ce.precision_weight(var_cert, var_bulk)
    assert abs(w - 0.25) < 1e-12  # the weight on the bulk estimator
    variance = lambda w: w**2 * var_bulk + (1 - w) ** 2 * var_cert  # noqa: E731
    grid = np.linspace(0, 1, 1001)
    assert variance(w) <= min(variance(g) for g in grid) + 1e-12
    assert abs(variance(w) - 1 / (1 / var_cert + 1 / var_bulk)) < 1e-12


def test_repetition_is_reproducible_and_finite():
    first = ce.run_repetition(2, sample_sizes=[20_000], tail_indices=[2.5])
    second = ce.run_repetition(2, sample_sizes=[20_000], tail_indices=[2.5])
    assert first == second and len(first) == 1
    row = first[0]
    assert row["k"] == ce.N_CERTIFIED and row["body_size"] < 20_000
    for key in ("certified", "bridge_body", "combined", "bridge_all", "weight", "threshold"):
        assert np.isfinite(row[key])
    assert 0 <= row["weight"] <= 1
    w = row["weight"]
    assert abs(row["combined"] - (w * row["bridge_body"] + (1 - w) * row["certified"])) < 1e-12
    assert abs(w - row["var_certified"] / (row["var_certified"] + row["var_bridge"])) < 1e-12
    assert ce.run_repetition(3, sample_sizes=[20_000], tail_indices=[2.5]) != first


def test_summary_and_table():
    rows = [r for rep in range(3) for r in ce.run_repetition(rep, sample_sizes=[20_000], tail_indices=[1.5, 2.5])]
    stats = ce.summarise(pl.DataFrame(rows))
    assert stats.height == 2 * len(ce.COLUMNS)
    assert set(stats["estimator"]) == set(ce.COLUMNS.values())
    assert (stats["repetitions"] == 3).all()
    body = ce.table(stats)
    assert body.count("\\hline") == 4
    for label in ce.TABULATED:
        assert label in body
    assert ce.BRIDGE_ALL not in body and "median weight on the bridge" in body
