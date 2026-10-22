"""USGS NWIS loaders: site metadata, series catalogue, daily discharge.

All outputs are polars frames cached as parquet under `hydro/cache/`.
Discharge is converted to m3/s at load time.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hydro.config import CACHE_DIR, CFS_TO_CUMECS, SQMI_TO_SQKM
from hydro.data.http import fetch, fetch_json, parse_rdb

DV_DIR = CACHE_DIR / "dv"

DISCHARGE_PARAM = "00060"


def _chunks(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


# ---------------------------------------------------------------- metadata


def site_metadata(sites: list[str]) -> pl.DataFrame:
    """Expanded site metadata (name, lat/lon, drainage area) for `sites`."""
    sites = sorted(set(sites))
    rows: list[dict] = []
    for chunk in _chunks(sites, 100):
        url = (
            "https://waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&sites={','.join(chunk)}&siteOutput=expanded&siteStatus=all"
        )
        rows.extend(parse_rdb(fetch(url)))
    if not rows:
        return pl.DataFrame(schema={"site_no": pl.Utf8})

    df = pl.DataFrame(rows, infer_schema_length=None)
    keep = [
        c
        for c in ("site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va",
                  "huc_cd", "drain_area_va", "contrib_drain_area_va", "alt_va")
        if c in df.columns
    ]
    df = df.select(keep)
    return df.with_columns(
        pl.col("dec_lat_va").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("dec_long_va").cast(pl.Float64, strict=False).alias("lon"),
        (pl.col("drain_area_va").cast(pl.Float64, strict=False) * SQMI_TO_SQKM).alias("area_km2"),
        pl.col("alt_va").cast(pl.Float64, strict=False).alias("alt_ft"),
    ).drop([c for c in ("dec_lat_va", "dec_long_va", "drain_area_va", "alt_va") if c in df.columns])


def series_catalog(sites: list[str]) -> pl.DataFrame:
    """Period of record for the daily-mean-discharge series of each site.

    Cheap screen: lets us reject short records before downloading any data.
    """
    sites = sorted(set(sites))
    rows: list[dict] = []
    for chunk in _chunks(sites, 50):
        url = (
            "https://nwis.waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&sites={','.join(chunk)}&seriesCatalogOutput=true"
            f"&parameterCd={DISCHARGE_PARAM}&siteStatus=all"
        )
        rows.extend(parse_rdb(fetch(url)))
    if not rows:
        return pl.DataFrame(schema={"site_no": pl.Utf8, "begin_date": pl.Utf8})

    df = pl.DataFrame(rows, infer_schema_length=None)
    return (
        df.filter(
            (pl.col("data_type_cd") == "dv")
            & (pl.col("parm_cd") == DISCHARGE_PARAM)
            & (pl.col("stat_cd") == "00003")
        )
        .select("site_no", "begin_date", "end_date", "count_nu")
        .with_columns(
            pl.col("begin_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("end_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("count_nu").cast(pl.Int64, strict=False).alias("n_daily_values"),
        )
        .drop("count_nu")
        .group_by("site_no")
        .agg(pl.min("begin_date"), pl.max("end_date"), pl.sum("n_daily_values"))
    )


# ---------------------------------------------------------------- daily values


def _dv_cache(site: str) -> Path:
    return DV_DIR / f"{site}.parquet"


def daily_discharge(site: str, start: str, end: str, *, refresh: bool = False) -> pl.DataFrame:
    """Daily-mean discharge for one site.

    Returns columns: date, q (m3/s), qualifiers (str), site_no.
    Negative / missing sentinel values (-999999) are dropped.
    """
    path = _dv_cache(site)
    if path.exists() and not refresh:
        return pl.read_parquet(path)

    url = (
        "https://waterservices.usgs.gov/nwis/dv/?format=json"
        f"&sites={site}&startDT={start}&endDT={end}"
        f"&parameterCd={DISCHARGE_PARAM}&statCd=00003&siteStatus=all"
    )
    payload = fetch_json(url)
    series = payload.get("value", {}).get("timeSeries", []) if payload else []

    records: list[dict] = []
    for ts in series:
        if ts["variable"]["variableCode"][0]["value"] != DISCHARGE_PARAM:
            continue
        for block in ts["values"]:
            for v in block["value"]:
                records.append(
                    {
                        "date": v["dateTime"][:10],
                        "value": v["value"],
                        "qualifiers": ",".join(v.get("qualifiers", []) or []),
                    }
                )

    if not records:
        df = pl.DataFrame(
            schema={"date": pl.Date, "q": pl.Float64, "qualifiers": pl.Utf8, "site_no": pl.Utf8}
        )
    else:
        df = (
            pl.DataFrame(records)
            .with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                pl.col("value").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("value") > -1e5)
            .with_columns((pl.col("value") * CFS_TO_CUMECS).alias("q"))
            .drop("value")
            .with_columns(pl.lit(site).alias("site_no"))
            .unique(subset=["date"], keep="first")
            .sort("date")
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df

