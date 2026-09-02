"""Unit tests required by spec 10: lag estimation, declustering counts,
safeguard semantics, no-intercept OLS, and the `demean` fix.

Everything here runs on synthetic data; nothing touches the network or cache.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from certification_estimators.certificate import Certificate, QuantileBall, Unrestricted
from certification_estimators.certificate_estimator import CertificateEstimator
from certification_estimators.extensions import PerVariableQuantileBall, TailCertificateEstimator
from hydro.config import AlignmentConfig
from hydro.preprocess.alignment import estimate_lag
from hydro.preprocess.decluster import runs_decluster
from hydro.preprocess.panel import _shift, _window_max

RNG = np.random.default_rng(0)


def _dates(n: int) -> pl.Series:
    day = np.timedelta64(1, "D")
    start = np.datetime64("1990-01-01")
    return pl.Series(np.arange(start, start + n * day, day)).cast(pl.Date)


# ----------------------------------------------------------------- alignment


@pytest.mark.parametrize("true_lag", [0, 1, 3, 5])
def test_estimate_lag_recovers_synthetic_shift(true_lag):
    n = 4000
    x = np.exp(RNG.normal(size=n)) + 0.05
    y = np.empty(n)
    y[true_lag:] = x[: n - true_lag] * 1.3
    y[:true_lag] = x[:true_lag] * 1.3
    y = y + 0.01 * np.exp(RNG.normal(size=n))
    est = estimate_lag(_dates(n), pl.Series(x), pl.Series(y), AlignmentConfig())
    assert est.lag_days == true_lag


def test_estimate_lag_flags_implausible_celerity():
    n = 2000
    x = np.exp(RNG.normal(size=n)) + 0.05
    y = np.roll(x, 3) * 1.1
    # 3 days of travel over 2 km implies ~0.008 m/s, far below the physical band.
    est = estimate_lag(_dates(n), pl.Series(x), pl.Series(y), AlignmentConfig(), distance_km=2.0)
    assert est.lag_days == 3 and not est.plausible


def test_shift_and_window_max():
    a = np.array([1.0, 5.0, 2.0, np.nan, 4.0])
    assert _shift(a, 1)[0] == 5.0 and np.isnan(_shift(a, 1)[-1])
    assert _window_max(a, 1)[0] == 5.0 and _window_max(a, 1)[3] == 4.0


# --------------------------------------------------------------- declustering


def test_runs_decluster_counts_and_peaks():
    # Two clusters: days 0-2 and days 12-13. With r = 7 the 9-day gap splits
    # them; with r = 14 it does not.
    q = [10.0, 30.0, 12.0] + [1.0] * 9 + [20.0, 40.0]
    df = pl.DataFrame({"date": _dates(len(q)), "X": q})
    out7 = runs_decluster(df, "X", 5.0, 7)
    assert out7.height == 2
    assert sorted(out7["X"].to_list()) == [30.0, 40.0]
    assert runs_decluster(df, "X", 5.0, 14).height == 1
    assert runs_decluster(df, "X", 100.0, 7).height == 0


# ------------------------------------------------------------ safeguard sets


def test_per_variable_quantile_ball_differs_from_aggregated_l_infty():
    # W0 is 100x the scale of W1. The aggregated l_infty radius is set by W0, so
    # QuantileBall leaves W1 unconstrained; PerVariableQuantileBall does not.
    df = pl.DataFrame({"W0": 100 * RNG.random(500), "W1": RNG.random(500)})
    agg = df.filter(QuantileBall(["W0", "W1"], 0.5, "l_infty").is_bounded)
    per = df.filter(PerVariableQuantileBall(["W0", "W1"], 0.5).is_bounded)
    assert agg["W1"].max() > per["W1"].max()
    assert per["W0"].max() < df["W0"].quantile(0.5) + 1e-9
    assert per["W1"].max() < df["W1"].quantile(0.5) + 1e-9
    # Per-variable is an intersection of half-spaces, so it is the smaller set.
    assert per.height <= agg.height


def test_empty_safeguard_is_unrestricted():
    df = pl.DataFrame({"W0": [1.0, 2.0, 3.0]})
    assert df.filter(PerVariableQuantileBall([], 0.5).is_bounded).height == 3
    assert df.filter(Unrestricted().is_bounded).height == 3


# ------------------------------------------------------------- the estimators


def _toy(n: int = 400) -> pl.DataFrame:
    u = RNG.pareto(2.0, n) + 1
    w = 0.5 * u + RNG.pareto(3.0, n)
    x = u + RNG.pareto(3.0, n) + 1
    return pl.DataFrame({"X": x, "Y": 2.0 * x + u, "W": w})


def test_ols_is_through_the_origin():
    df = pl.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [3.0, 5.0, 7.0]})  # Y = 2X + 1
    est = TailCertificateEstimator("X", Unrestricted(), "X", "Y", level=0.0)
    beta = est.fit_on_certified_subset(df)
    xs, ys = df["X"].to_numpy(), df["Y"].to_numpy()
    assert beta == pytest.approx(float(xs @ ys) / float(xs @ xs))
    assert beta != pytest.approx(2.0)  # an intercept model would return exactly 2


def test_demean_is_off_by_default_and_changes_the_safeguard():
    df = _toy()
    safeguard = PerVariableQuantileBall(["W"], 0.5)
    est = TailCertificateEstimator("X", safeguard, "X", "Y", certification_quantile=0.8)
    assert est.fit(df) == pytest.approx(est.fit(df, demean=False))
    # Demeaning sends |W| through zero, so "small |W|" stops meaning "W small":
    # the certified subsets differ, which is the bug valid_bridge_note.md flags.
    plain = Certificate("X", safeguard).quantile_certified_subset(df, 0.8)
    centred = df.with_columns(pl.col(c) - pl.col(c).mean() for c in df.columns)
    demeaned = Certificate("X", safeguard).quantile_certified_subset(centred, 0.8)
    assert plain.height != demeaned.height or plain["W"].max() != pytest.approx(
        demeaned["W"].max() + df["W"].mean()
    )
    # The base class must not force demeaning on either entry point.
    base = CertificateEstimator("X", safeguard, "X", "Y", 0.8)
    assert base.fit(df) == pytest.approx(base.fit_or_null(df))


def test_level_and_quantile_paths_are_exclusive_and_agree():
    df = _toy()
    s = PerVariableQuantileBall(["W"], 0.5)
    with pytest.raises(ValueError):
        TailCertificateEstimator("X", s, "X", "Y", certification_quantile=0.8, level=1.0)
    with pytest.raises(ValueError):
        TailCertificateEstimator("X", s, "X", "Y")
    v = df.filter(s.is_bounded).select(pl.col("X").abs().quantile(0.8)).item()
    a = TailCertificateEstimator("X", s, "X", "Y", certification_quantile=0.8).fit(df)
    b = TailCertificateEstimator("X", s, "X", "Y", level=v).fit(df)
    assert a == pytest.approx(b)


def test_trimmed_sits_between_median_and_mean_of_ratios():
    df = _toy()
    s = Unrestricted()
    kw = dict(e="X", safeguard=s, x="X", y="Y", level=0.0)
    r = (df["Y"] / df["X"]).to_numpy()
    trimmed = TailCertificateEstimator(**kw, estimator_type="trimmed", trim_fraction=0.1).fit(df)
    med = TailCertificateEstimator(**kw, estimator_type="median").fit(df)
    assert min(med, float(r.mean())) <= trimmed <= max(med, float(r.mean()))


def test_safeguard_reduces_confounding_on_synthetic_data():
    """The certificate must beat the tail-only estimate on data built to confound."""
    df = _toy(4000)
    v = df["X"].quantile(0.9)
    cert = TailCertificateEstimator(
        "X", PerVariableQuantileBall(["W"], 0.3), "X", "Y", level=v
    ).fit(df)
    tail_only = TailCertificateEstimator("X", Unrestricted(), "X", "Y", level=v).fit(df)
    assert abs(cert - 2.0) < abs(tail_only - 2.0)


# ------------------------------------------------------------ safeguards


def _toy_network():
    """X -> Y on a main stem; tributary gauges A (downstream) and A_up (nested
    above A) join between them; P is on a separate stream."""
    from hydro.data.network import Network

    upstream = {
        "X": {"X"},
        "Y": {"Y", "X", "A", "A_up"},
        "A": {"A", "A_up"},
        "A_up": {"A_up"},
        "P": {"P"},
    }
    return Network(upstream=upstream, mainstem={"X", "Y"}, meta=pl.DataFrame())


def test_drop_nested_keeps_the_downstream_gauge_and_order():
    from hydro.experiments.candidates import drop_nested

    net = _toy_network()
    assert drop_nested(["A", "A_up", "P"], net) == ["A", "P"]
    assert drop_nested(["A_up", "P", "A"], net) == ["P", "A"]
    assert drop_nested(["P"], net) == ["P"]


def test_bias_decomposition_is_an_identity():
    from hydro.estimation import bias_decomposition

    rng = np.random.default_rng(0)
    x = rng.pareto(2.0, 500) + 1
    w = rng.pareto(2.0, 500) + 0.5
    y = 0.9 * x + w + rng.normal(size=500)
    df = pl.DataFrame({"X": x, "Y": y, "W0": w})
    d = bias_decomposition(df, ["W0"])
    assert d["beta"] == pytest.approx(float(x @ y / (x @ x)))
    assert d["beta"] == pytest.approx(1 + d["b_W"] + d["b_U"])
    assert d["b_W"] == pytest.approx(float(x @ w / (x @ x)))


def test_subtracting_w_recovers_beta_when_w_enters_with_coefficient_one():
    from hydro.baselines import ols_adjusting_for_w, ols_subtracting_w

    rng = np.random.default_rng(1)
    u = rng.pareto(2.5, 4000) + 1
    x = u + rng.pareto(2.5, 4000)
    w = u + rng.pareto(2.5, 4000)
    y = 0.9 * x + w + 0.01 * rng.normal(size=4000)
    df = pl.DataFrame({"X": x, "Y": y, "W0": w})
    assert ols_subtracting_w(df, ["W0"]) == pytest.approx(0.9, abs=1e-3)
    assert ols_adjusting_for_w(df, ["W0"]) == pytest.approx(0.9, abs=1e-2)
