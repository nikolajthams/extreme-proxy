"""Assemble the analysis-ready event table for one station pair.

Pipeline (all steps in spec 5):
  season filter -> quality filter -> travel-time alignment -> missing-data drop
  -> runs declustering on X.

Everything is done on a contiguous daily grid held as numpy arrays, so shifts
and window maxima are exact and gaps stay gaps (never silently closed).
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
    """A causal design instantiated on the gauge network."""

    label: str
    region: Region
    x: str
    y: str
    w: list[str] = field(default_factory=list)   # safeguard gauges
    z: str | None = None                          # instrument (Design B)
    kind: str = "connected"                       # connected | disconnected | reverse

    @property
    def sites(self) -> list[str]:
        s = [self.x, self.y, *self.w]
        if self.z:
            s.append(self.z)
        return s


@dataclass
class Panel:
    spec: PairSpec
    events: pl.DataFrame           # one row per declustered flood event
    daily: pl.DataFrame            # aligned daily frame before declustering
    lag_xy: LagEstimate
    lag_zx: LagEstimate | None
    threshold_u: float             # declustering threshold on X (m3/s)
    w_cols: list[str]
    n_daily: int
    n_events: int
    definition: str
    run_length_days: int
    dropped_quality: int


def _grid(site: str, region: Region, cfg: HydroConfig,
          lo: np.datetime64, hi: np.datetime64) -> tuple[np.ndarray, int]:
    """In-season, quality-filtered daily discharge on a contiguous grid."""
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

    d = df["date"].to_numpy().astype("datetime64[D]")
    idx = (d - lo).astype(int)
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = df["q"].to_numpy()[keep]
    return out, dropped


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a shifted so that result[t] = a[t + k]."""
    out = np.full_like(a, np.nan)
    if k == 0:
        return a.copy()
    if k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def _window_max(a: np.ndarray, half: int) -> np.ndarray:
    """Centred window maximum over +-half days, ignoring nan."""
    if half <= 0:
        return a.copy()
    stack = np.vstack([_shift(a, k) for k in range(-half, half + 1)])
    with np.errstate(invalid="ignore"):
        out = np.nanmax(stack, axis=0)
    out[np.all(np.isnan(stack), axis=0)] = np.nan
    return out


def _common_range(sites: list[str], region: Region, cfg: HydroConfig):
    los, his = [], []
    for s in sites:
        df = nwis.daily_discharge(s, cfg.start_date, cfg.end_date)
        if df.height == 0:
            continue
        los.append(df["date"].min())
        his.append(df["date"].max())
    if not los:
        return None
    lo = np.datetime64(max(los), "D")
    hi = np.datetime64(min(his), "D")
    return (lo, hi) if hi > lo else None


def build_panel(
    spec: PairSpec,
    cfg: HydroConfig = CONFIG,
    *,
    definition: str | None = None,
    run_length_days: int | None = None,
    distance_km: float | None = None,
    force_lag: int | None = None,
) -> Panel | None:
    definition = definition or cfg.alignment.default_definition
    run_length_days = run_length_days or cfg.decluster.run_length_days

    rng = _common_range(spec.sites, spec.region, cfg)
    if rng is None:
        return None
    lo, hi = rng
    dates = np.arange(lo, hi + DAY)

    grids: dict[str, np.ndarray] = {}
    dropped = 0
    for s in set(spec.sites):
        g, d = _grid(s, spec.region, cfg, lo, hi)
        grids[s] = g
        dropped += d

    gx, gy = grids[spec.x], grids[spec.y]

    # --- travel-time alignment (spec 5.1) -------------------------------
    both = np.isfinite(gx) & np.isfinite(gy)
    if both.sum() < 365:
        return None
    lag_xy = estimate_lag(
        pl.Series(dates[both].astype("datetime64[ms]")).cast(pl.Date),
        pl.Series(gx[both]),
        pl.Series(gy[both]),
        cfg.alignment,
        distance_km,
    )

    # The lag search runs over non-negative lags only, because water flows
    # downhill. The reverse-direction placebo (E3) therefore passes the forward
    # lag with its sign flipped via `force_lag`, so that placebo cannot be won
    # by leaving the anti-causal direction misaligned (spec 5.1 step 3).
    if force_lag is not None:
        lag_xy = replace(lag_xy, lag_days=force_lag, reason=f"forced lag {force_lag}")
    half = cfg.alignment.window_half_width_days if definition == "window" else 0
    y_aligned = _shift(_window_max(gy, half), lag_xy.lag_days)

    cols: dict[str, np.ndarray] = {"X": gx, "Y": y_aligned}
    w_cols = []
    for i, w in enumerate(spec.w):
        name = f"W{i}"
        w_cols.append(name)
        # The safeguard asks whether the intervening catchment was in flood
        # around the event, so a window maximum is the conservative reading;
        # with definition="point" it degenerates to the event-day value.
        cols[name] = _window_max(grids[w], half)

    lag_zx = None
    if spec.z:
        gz = grids[spec.z]
        both_zx = np.isfinite(gz) & np.isfinite(gx)
        if both_zx.sum() >= 365:
            lag_zx = estimate_lag(
                pl.Series(dates[both_zx].astype("datetime64[ms]")).cast(pl.Date),
                pl.Series(gz[both_zx]),
                pl.Series(gx[both_zx]),
                cfg.alignment,
                None,
            )
            # Z is observed lag_zx days before X, so bring it forward.
            cols["Z"] = _shift(gz, -lag_zx.lag_days)
        else:
            cols["Z"] = gz

    stacked = np.vstack(list(cols.values()))
    complete = np.all(np.isfinite(stacked), axis=0)

    daily = pl.DataFrame(
        {"date": pl.Series(dates[complete].astype("datetime64[ms]")).cast(pl.Date)}
        | {k: v[complete] for k, v in cols.items()}
    ).with_columns(pl.col("date").dt.year().alias("year"))

    if daily.height < 365:
        return None

    threshold_u = float(np.quantile(daily["X"].to_numpy(), cfg.decluster.threshold_quantile))
    events = runs_decluster(daily, "X", threshold_u, run_length_days)

    return Panel(
        spec=spec,
        events=events,
        daily=daily,
        lag_xy=lag_xy,
        lag_zx=lag_zx,
        threshold_u=threshold_u,
        w_cols=w_cols,
        n_daily=daily.height,
        n_events=events.height,
        definition=definition,
        run_length_days=run_length_days,
        dropped_quality=dropped,
    )
