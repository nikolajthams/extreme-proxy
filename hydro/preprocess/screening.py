"""Gauge-level screen and per-gauge record completeness.

Every screen is applied before any estimate exists and the reason a gauge is
rejected is recorded; screens use only record length, drainage area, the
station name and the curated regulation list.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from hydro.config import CACHE_DIR, CONFIG, REGION_BY_NAME, Region
from hydro.data import nldi, nwis
from hydro.data.network import Network, build_network

SCREEN_DIR = CACHE_DIR / "screen"
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def screen_region_sites(region: Region, *, refresh: bool = False) -> pl.DataFrame:
    """Every gauge upstream of the region outlet with pass/fail and reason."""
    path = SCREEN_DIR / f"{region.name}_sites.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)

    scr = CONFIG.gauges
    sites = nldi.upstream_tributaries(region.outlet, region.navigation_km)["site_no"].to_list()
    catalog = nwis.series_catalog(sites).with_columns(
        ((pl.col("end_date") - pl.col("begin_date")).dt.total_days() / 365.25).alias("record_years")
    )
    long_enough = catalog.filter(
        (pl.col("record_years") >= scr.min_record_years)
        & (pl.col("n_daily_values") >= scr.min_daily_values)
    )
    meta = nwis.site_metadata(long_enough["site_no"].to_list())
    name_hit = pl.lit(False)
    for kw in scr.regulation_name_keywords:
        name_hit = name_hit | pl.col("station_nm").str.to_uppercase().str.contains(kw, literal=True)
    df = (
        long_enough.join(meta, on="site_no", how="inner")
        .with_columns(pl.lit(region.name).alias("region"))
        .with_columns(
            pl.when(pl.col("site_tp_cd") != "ST").then(pl.lit("not a stream gauge"))
            .when(pl.col("area_km2").is_null()).then(pl.lit("no drainage area published"))
            .when(pl.col("area_km2") < scr.min_drainage_area_km2).then(pl.lit("drainage area below minimum"))
            .when(name_hit | pl.col("site_no").is_in(scr.regulated_sites)).then(pl.lit("regulation screen"))
            .otherwise(pl.lit("pass"))
            .alias("screen_result")
        )
        .sort("site_no")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


@lru_cache(maxsize=None)
def region_network(region_name: str) -> Network:
    """Network over the gauges of a region that pass the gauge screen."""
    region = REGION_BY_NAME[region_name]
    passing = screen_region_sites(region).filter(pl.col("screen_result") == "pass")
    return build_network(passing, region.navigation_km)


def season_days(region: Region) -> int:
    return sum(_DAYS_IN_MONTH[m - 1] for m in region.season_months)


@lru_cache(maxsize=None)
def good_years(site: str, region_name: str) -> frozenset[int]:
    """Years in which the gauge's in-season record is complete (spec: at most
    `max_missing_fraction_in_year` of the season's days missing)."""
    region = REGION_BY_NAME[region_name]
    d = nwis.daily_discharge(site, CONFIG.start_date, CONFIG.end_date)
    d = d.filter(pl.col("date").dt.month().is_in(list(region.season_months)))
    if d.height == 0:
        return frozenset()
    thresh = (1 - CONFIG.gauges.max_missing_fraction_in_year) * season_days(region)
    counts = d.group_by(pl.col("date").dt.year().alias("year")).agg(pl.len().alias("n"))
    return frozenset(counts.filter(pl.col("n") >= thresh)["year"].to_list())
