"""Tests of the river-discharge application: alignment, declustering, the
safeguard and the estimators on synthetic data, and one reproducibility check
of the headline pair against the cached USGS data.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from certification_estimators.certificate import QuantileBall, Unrestricted
from certification_estimators.extensions import PerVariableQuantileBall
from hydro import estimation as est
from hydro.config import CONFIG, Alignment, OUTPUT_DIR
from hydro.data.network import Network
from hydro.preprocess.alignment import estimate_lag
from hydro.preprocess.decluster import runs_decluster
from hydro.preprocess.panel import _shift

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
    assert estimate_lag(_dates(n), pl.Series(x), pl.Series(y), Alignment()).lag_days == true_lag


def test_estimate_lag_flags_implausible_celerity():
    n = 2000
    x = np.exp(RNG.normal(size=n)) + 0.05
    y = np.roll(x, 3) * 1.1
    # 3 days over 2 km is ~0.008 m/s, far below the physical band.
    e = estimate_lag(_dates(n), pl.Series(x), pl.Series(y), Alignment(), distance_km=2.0)
    assert e.lag_days == 3 and not e.plausible


def test_shift():
    a = np.array([1.0, 5.0, 2.0, np.nan, 4.0])
    assert _shift(a, 1)[0] == 5.0 and np.isnan(_shift(a, 1)[-1])
    assert _shift(a, -1)[1] == 1.0 and np.isnan(_shift(a, -1)[0])


# --------------------------------------------------------------- declustering


def test_runs_decluster_counts_and_peaks():
    # Two clusters: days 0-2 and days 12-13. With r = 7 the 9-day gap splits
    # them; with r = 14 it does not.
    q = [10.0, 30.0, 12.0] + [1.0] * 9 + [20.0, 40.0]
    df = pl.DataFrame({"date": _dates(len(q)), "T": q})
    out7 = runs_decluster(df, "T", 5.0, 7)
    assert out7.height == 2 and sorted(out7["T"].to_list()) == [30.0, 40.0]
    assert runs_decluster(df, "T", 5.0, 14).height == 1
    assert runs_decluster(df, "T", 100.0, 7).height == 0


# ------------------------------------------------------------ safeguard sets


def test_per_variable_quantile_ball_bounds_each_coordinate():
    # P0 is 100x the scale of P1: an aggregated l_infty radius is set by P0
    # and leaves P1 unconstrained; the per-variable ball does not.
    df = pl.DataFrame({"P0": 100 * RNG.random(500), "P1": RNG.random(500)})
    agg = df.filter(QuantileBall(["P0", "P1"], 0.5, "l_infty").is_bounded)
    per = df.filter(PerVariableQuantileBall(["P0", "P1"], 0.5).is_bounded)
    assert agg["P1"].max() > per["P1"].max()
    assert per["P0"].max() < df["P0"].quantile(0.5) + 1e-9
    assert per["P1"].max() < df["P1"].quantile(0.5) + 1e-9
    assert per.height <= agg.height


def test_empty_safeguard_is_unrestricted():
    df = pl.DataFrame({"P0": [1.0, 2.0, 3.0]})
    assert df.filter(est.safeguard([], 0.5).is_bounded).height == 3
    assert df.filter(Unrestricted().is_bounded).height == 3


# ------------------------------------------------------------- the estimators


def _toy(n: int = 400) -> pl.DataFrame:
    u = RNG.pareto(2.0, n) + 1
    p = 0.5 * u + RNG.pareto(3.0, n)
    t = u + RNG.pareto(3.0, n) + 1
    return pl.DataFrame({"T": t, "Y": 2.0 * t + u, "P0": p, "year": np.arange(n) // 20})


def test_certified_ols_is_through_the_origin():
    df = pl.DataFrame({"T": [1.0, 2.0, 3.0], "Y": [3.0, 5.0, 7.0]})  # Y = 2T + 1
    beta = est.fit_certified(df, [], 0.5, 0.0, est.OLS).beta
    t, y = df["T"].to_numpy(), df["Y"].to_numpy()
    assert beta == pytest.approx(float(t @ y) / float(t @ t))
    assert beta == pytest.approx(est.naive_ols(df))
    assert beta != pytest.approx(2.0)


def test_certified_maximum_is_the_ratio_at_the_largest_t():
    df = pl.DataFrame({"T": [1.0, 5.0, 3.0], "Y": [2.0, 15.0, 3.0]})
    assert est.fit_certified(df, [], 0.5, 0.0, est.MAX).beta == pytest.approx(3.0)


def test_threshold_is_taken_on_the_safeguard_satisfying_events():
    df = _toy()
    v = est.threshold(df, ["P0"], 0.5, 0.8)
    sub = df.filter(pl.col("P0") < df["P0"].quantile(0.5))
    assert v == pytest.approx(float(sub["T"].quantile(0.8)))
    cert = est.certified_events(df, ["P0"], 0.5, v)
    assert cert["T"].min() > v and cert["P0"].max() < df["P0"].quantile(0.5)


def test_operating_point_is_the_largest_quantile_keeping_min_events():
    df = _toy()
    q_t, v = est.operating_point(df, ["P0"], 0.5, 5)
    assert q_t in CONFIG.estimator.q_t_grid
    assert est.certified_events(df, ["P0"], 0.5, v).height >= 5
    higher = [q for q in CONFIG.estimator.q_t_grid if q > q_t]
    for q in higher:
        assert est.certified_events(df, ["P0"], 0.5, est.threshold(df, ["P0"], 0.5, q)).height < 5
    assert est.operating_point(df, ["P0"], 0.5, 10 ** 6) is None


def test_safeguard_reduces_confounding_on_synthetic_data():
    df = _toy(4000)
    v = df["T"].quantile(0.9)
    cert = est.fit_certified(df, ["P0"], 0.3, v, est.OLS).beta
    tail_only = est.fit_certified(df, [], 0.3, v, est.OLS).beta
    assert abs(cert - 2.0) < abs(tail_only - 2.0)


def test_year_bootstrap_is_seeded_and_covers_the_estimate():
    df = _toy(600)
    a = est.year_bootstrap_interval(df, est.naive_ols)
    b = est.year_bootstrap_interval(df, est.naive_ols)
    assert a == b and a[0] <= est.naive_ols(df) <= a[1]


# ------------------------------------------------------------------ network


def test_drop_nested_keeps_the_downstream_gauge_and_order():
    # T -> Y on a main stem; A_up is nested above A; P is on a separate stream.
    net = Network(upstream={"T": {"T"}, "Y": {"Y", "T", "A", "A_up"}, "A": {"A", "A_up"},
                            "A_up": {"A_up"}, "P": {"P"}}, meta=pl.DataFrame())
    assert net.drop_nested(["A", "A_up", "P"]) == ["A", "P"]
    assert net.drop_nested(["A_up", "P", "A"]) == ["P", "A"]
    assert net.disconnected_from_both("T", "Y") == ["P"]
    assert net.is_disconnected("A", "P") and not net.is_disconnected("A", "Y")


# ---------------------------------------------------------- reproducibility


@pytest.mark.skipif(not (OUTPUT_DIR / "pairs.parquet").exists(), reason="pipeline outputs not present")
def test_headline_pair_reproduces_from_the_cache():
    """The headline pair's panel and naive OLS rebuilt from the cached USGS
    data equal the persisted values."""
    from hydro.pairs import headline_pair, panel_for

    row = headline_pair()
    panel = panel_for(row)
    assert panel.n_daily == row["n_daily"] and panel.lag.lag_days == row["lag_days"]
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet").filter(pl.col("pair") == row["pair"])
    naive = d.filter(pl.col("method") == "naive OLS")["beta"][0]
    assert est.naive_ols(panel.daily) == pytest.approx(naive)
    c = d.filter((pl.col("method") == "certified OLS") & (pl.col("q_p") == CONFIG.estimator.q_p)).row(0, named=True)
    f = est.fit_at_operating_point(panel.daily, panel.p_cols, CONFIG.estimator.q_p, est.OLS)
    assert f.n_cert == c["n_cert"] and f.beta == pytest.approx(c["beta"])
