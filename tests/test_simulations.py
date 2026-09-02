"""Smoke tests for the simulation scripts in `experiments/`.

They check the design registry encodes the anchor-safeguard pairs the notes
claim, that a replication is reproducible from its index alone, and that the
summary statistics are computed from the stored per-replication estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import iv_and_proxy as sim  # noqa: E402
from certification_estimators.certificate import Unrestricted  # noqa: E402


def _certificate(design: sim.Design, label: str):
    return design.estimators[label]()


def test_designs_use_the_pairs_selected_by_the_criterion():
    by_key = {d.key: d for d in sim.DESIGNS}
    assert list(by_key) == ["valid_iv", "invalid_iv", "linear_proxy", "nonlinear_proxy"]

    valid = _certificate(by_key["valid_iv"], sim.CERTIFIED)
    assert valid.e == "Z" and isinstance(valid.safeguard, Unrestricted)

    invalid = _certificate(by_key["invalid_iv"], sim.CERTIFIED)
    assert invalid.e == sim.TREATMENT and invalid.safeguard.variables == ["Z"]
    disqualified = _certificate(by_key["invalid_iv"], sim.DISQUALIFIED)
    assert disqualified.e == "Z" and isinstance(disqualified.safeguard, Unrestricted)

    for key in ("linear_proxy", "nonlinear_proxy"):
        cert = _certificate(by_key[key], sim.CERTIFIED)
        assert cert.e == "Z" and cert.safeguard.variables == ["W"]
        assert by_key[key].classical == sim.BRIDGE
    assert by_key["valid_iv"].classical == sim.TSLS


def test_repetition_is_reproducible_and_finite():
    first = sim.run_repetition(3, sample_sizes=[20_000], tail_indices=[2.5])
    second = sim.run_repetition(3, sample_sizes=[20_000], tail_indices=[2.5])
    assert first == second
    estimates = [row["estimate"] for row in first]
    assert all(e is not None and np.isfinite(e) for e in estimates)
    # 2 + 3 + 2 + 4 estimators, one cell each
    assert len(first) == 11
    other = sim.run_repetition(4, sample_sizes=[20_000], tail_indices=[2.5])
    assert other != first


def test_summary_statistics():
    rows = [
        {"design": "d", "estimator": "e", "tail_index": 1.5, "sample_size": 10, "repetition": r,
         "estimate": v}
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


def test_repetition_is_reproducible_across_processes():
    """String hashing is randomised per process; the seeded replication must not depend on it."""
    import json
    import os
    import subprocess

    code = (
        "import json, sys; sys.path.insert(0, 'experiments'); import iv_and_proxy as sim; "
        "print(json.dumps(sim.run_repetition(5, sample_sizes=[20_000], tail_indices=[2.5])))"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                cwd=Path(__file__).resolve().parents[1], check=True)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
