"""Spec 4.3: deterministic, auditable gauge and pair selection.

Two rules govern this module:

1. Every screen is applied *before* any beta is estimated, and the reason a
   gauge or pair was rejected is recorded in the artifact, so nothing can be
   dropped after the fact for looking bad.
2. Screens use only record length, coverage, drainage area, network topology
   and a documented regulation list -- never an estimate.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CONFIG, CACHE_DIR, Region
from hydro.data import nldi, nwis
from hydro.data.network import Network, build_network

SCREEN_DIR = CACHE_DIR / "screen"


def _regulation_flag(name_col: str) -> pl.Expr:
    kws = CONFIG.screening.regulation_name_keywords
    expr = pl.lit(False)
    upper = pl.col(name_col).str.to_uppercase()
    for kw in kws:
        expr = expr | upper.str.contains(kw, literal=True)
    return expr


def screen_region_sites(region: Region, *, refresh: bool = False) -> pl.DataFrame:
    """Every gauge upstream of the region outlet, with pass/fail and reason."""
    path = SCREEN_DIR / f"{region.name}_sites.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)

    ut = nldi.upstream_tributaries(region.outlet, region.navigation_km)
    sites = ut["site_no"].to_list()

    catalog = nwis.series_catalog(sites)
    scr = CONFIG.screening
    cat = (
        catalog.with_columns(
            (
                (pl.col("end_date") - pl.col("begin_date")).dt.total_days() / 365.25
            ).alias("record_years")
        )
    )
    # Stage 1: record length. Cheap, removes ~95% of NWIS sites.
    stage1 = cat.filter(
        (pl.col("record_years") >= scr.min_overlap_years + 5)
        & (pl.col("n_daily_values") >= scr.min_overlap_years * 365)
    )

    meta = nwis.site_metadata(stage1["site_no"].to_list())
    df = (
        stage1.join(meta, on="site_no", how="inner")
        .join(ut.select("site_no", "mainstem", "comid"), on="site_no", how="left")
        .with_columns(
            pl.lit(region.name).alias("region"),
            _regulation_flag("station_nm").alias("regulation_name_hit"),
            pl.col("site_no").is_in(scr.regulated_sites).alias("regulation_listed"),
        )
        .with_columns(
            pl.when(pl.col("site_tp_cd") != "ST")
            .then(pl.lit("not a stream gauge"))
            .when(pl.col("area_km2").is_null())
            .then(pl.lit("no drainage area published"))
            .when(pl.col("area_km2") < scr.min_drainage_area_km2)
            .then(pl.lit("drainage area below minimum"))
            .when(pl.col("regulation_name_hit") | pl.col("regulation_listed"))
            .then(pl.lit("regulation screen"))
            .otherwise(pl.lit("pass"))
            .alias("screen_result")
        )
        .sort("site_no")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def region_network(region: Region, *, refresh: bool = False) -> Network:
    """Network over the gauges that pass the site-level screen."""
    sites_df = screen_region_sites(region, refresh=refresh)
    passing = sites_df.filter(pl.col("screen_result") == "pass")
    sites = passing["site_no"].to_list()
    mainstem = nldi.mainstem_set(region.outlet, region.navigation_km)
    return build_network(sites, mainstem, passing, region.navigation_km)


# ------------------------------------------------------------------ coverage


def in_season_coverage(site: str, region: Region) -> tuple[int, int, float]:
    """(first_year, last_year, coverage) of in-season daily values for a gauge."""
    df = nwis.daily_discharge(site, CONFIG.start_date, CONFIG.end_date)
    if df.height == 0:
        return (0, 0, 0.0)
    df = df.filter(pl.col("date").dt.month().is_in(list(region.season_months)))
    if df.height == 0:
        return (0, 0, 0.0)
    y0 = df["date"].dt.year().min()
    y1 = df["date"].dt.year().max()
    n_days_per_year = sum(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1] for m in region.season_months
    )
    expected = (y1 - y0 + 1) * n_days_per_year
    return (int(y0), int(y1), df.height / expected if expected else 0.0)


_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def season_days(region: Region) -> int:
    return sum(_DAYS_IN_MONTH[m - 1] for m in region.season_months)


def good_years(site: str, region: Region) -> set[int]:
    """Years in which the gauge has at least (1 - max_missing) in-season days.

    This is the per-year completeness screen of spec 4.3, applied per gauge so
    that a pair's usable record is just the intersection of the two sets.
    """
    d = nwis.daily_discharge(site, CONFIG.start_date, CONFIG.end_date)
    d = d.filter(pl.col("date").dt.month().is_in(list(region.season_months)))
    if d.height == 0:
        return set()
    thresh = (1 - CONFIG.screening.max_missing_fraction_in_any_year) * season_days(region)
    counts = d.group_by(pl.col("date").dt.year().alias("year")).agg(pl.len().alias("n"))
    return set(counts.filter(pl.col("n") >= thresh)["year"].to_list())


def overlap_years(a: str, b: str, region: Region) -> int:
    """Number of years in which both gauges have >=95% of in-season days."""
    return len(good_years(a, region) & good_years(b, region))
