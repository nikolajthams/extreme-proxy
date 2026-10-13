"""The analysis table of one pair: aligned daily record and declustered flood events.

season filter -> quality filter -> travel-time alignment of Y -> complete days
-> runs declustering on T. Everything sits on a contiguous daily grid held as
numpy arrays, so shifts are exact and gaps stay gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from hydro.config import CONFIG, HydroConfig, Region
from hydro.data import nwis
from hydro.preprocess.alignment import LagEstimate, estimate_lag
from hydro.preprocess.decluster import runs_decluster

DAY = np.timedelta64(1, "D")


@dataclass
class PairSpec:
    label: str
    region: Region
    t: str
    y: str
    proxies: list[str] = field(default_factory=list)

    @property
    def sites(self) -> list[str]:
        return [self.t, self.y, *self.proxies]


@dataclass
class Panel:
    spec: PairSpec
    events: pl.DataFrame     # one row per declustered flood event: date, year, T, Y, P0..
    daily: pl.DataFrame      # every complete aligned in-season day, same columns
    lag: LagEstimate
    threshold_u: float       # declustering threshold on T (m3/s)
    p_cols: list[str]
    dropped_quality: int

    @property
    def n_events(self) -> int:
        return self.events.height

    @property
    def n_daily(self) -> int:
        return self.daily.height


def _grid(site: str, region: Region, cfg: HydroConfig,
          lo: np.datetime64, hi: np.datetime64) -> tuple[np.ndarray, int]:
    """In-season, quality-filtered daily discharge on the grid [lo, hi]."""
    df = nwis.daily_discharge(site, cfg.start_date, cfg.end_date)
    df = df.filter(pl.col("date").dt.month().is_in(list(region.season_months)))
    n = len(np.arange(lo, hi + DAY))
    out = np.full(n, np.nan)
    if df.height == 0:
        return out, 0
    bad = pl.lit(False)
    for q in cfg.quality.excluded_qualifiers:
        bad = bad | pl.col("qualifiers").str.split(",").list.contains(q)
    dropped = df.filter(bad).height
    df = df.filter(~bad)
    idx = (df["date"].to_numpy().astype("datetime64[D]") - lo).astype(int)
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = df["q"].to_numpy()[keep]
    return out, dropped


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """result[t] = a[t + k]."""
    out = np.full_like(a, np.nan)
    if k == 0:
        return a.copy()
    if k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def _common_range(sites: list[str], cfg: HydroConfig):
    los, his = [], []
    for s in sites:
        df = nwis.daily_discharge(s, cfg.start_date, cfg.end_date)
        if df.height:
            los.append(df["date"].min())
            his.append(df["date"].max())
    if not los:
        return None
    lo, hi = np.datetime64(max(los), "D"), np.datetime64(min(his), "D")
    return (lo, hi) if hi > lo else None


def build_panel(spec: PairSpec, cfg: HydroConfig = CONFIG, *,
                distance_km: float | None = None, lag: int | None = None) -> Panel | None:
    """The panel of a pair, or None if the common record is too short.

    Y is read `lag` days after T. By default the lag is estimated from the
    flood waves (`alignment.estimate_lag`); the placebo passes `lag=0`, since
    two disconnected gauges have no travel time.
    """
    rng = _common_range(spec.sites, cfg)
    if rng is None:
        return None
    lo, hi = rng
    dates = np.arange(lo, hi + DAY)

    grids, dropped = {}, 0
    for s in dict.fromkeys(spec.sites):
        grids[s], d = _grid(s, spec.region, cfg, lo, hi)
        dropped += d
    gt, gy = grids[spec.t], grids[spec.y]

    both = np.isfinite(gt) & np.isfinite(gy)
    if both.sum() < 365:
        return None
    lag_est = estimate_lag(
        pl.Series(dates[both].astype("datetime64[ms]")).cast(pl.Date),
        pl.Series(gt[both]), pl.Series(gy[both]), cfg.alignment, distance_km,
    )
    if lag is not None:
        lag_est = replace(lag_est, lag_days=lag, plausible=True, reason=f"fixed lag {lag}")

    cols = {"T": gt, "Y": _shift(gy, lag_est.lag_days)}
    p_cols = []
    for i, p in enumerate(spec.proxies):
        cols[f"P{i}"] = grids[p]
        p_cols.append(f"P{i}")

    complete = np.all(np.isfinite(np.vstack(list(cols.values()))), axis=0)
    daily = pl.DataFrame(
        {"date": pl.Series(dates[complete].astype("datetime64[ms]")).cast(pl.Date)}
        | {k: v[complete] for k, v in cols.items()}
    ).with_columns(pl.col("date").dt.year().alias("year"))
    if daily.height < 365:
        return None

    threshold_u = float(np.quantile(daily["T"].to_numpy(), cfg.decluster.threshold_quantile))
    events = runs_decluster(daily, "T", threshold_u, cfg.decluster.run_length_days)
    return Panel(spec=spec, events=events, daily=daily, lag=lag_est, threshold_u=threshold_u,
                 p_cols=p_cols, dropped_quality=dropped)
