"""NLDI (Network Linked Data Index) navigation: the river-network topology.

The only query used is "UT" (upstream with tributaries): the set of gauges
upstream of a gauge. Every topological statement in the application ("T is
upstream of Y", "P is disconnected from both") is a set expression over it.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CACHE_DIR
from hydro.data.http import fetch_json

NAV_DIR = CACHE_DIR / "nldi"
BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}"


def upstream_tributaries(site: str, distance_km: int = 500) -> pl.DataFrame:
    """All gauges upstream of `site` on any tributary (plus `site`), cached."""
    path = NAV_DIR / f"{site}_UT_{distance_km}.parquet"
    if path.exists():
        return pl.read_parquet(path)

    url = f"{BASE.format(site=site)}/navigation/UT/nwissite?distance={distance_km}"
    payload = fetch_json(url)
    rows = []
    for f in payload.get("features", []) if payload else []:
        p = f.get("properties", {})
        ident = p.get("identifier", "")
        if not ident.startswith("USGS-"):
            continue
        coords = (f.get("geometry") or {}).get("coordinates") or [None, None]
        rows.append({
            "site_no": ident.removeprefix("USGS-"),
            "name": p.get("name"),
            "comid": str(p.get("comid")),
            "mainstem": p.get("mainstem"),
            "lon": float(coords[0]) if coords[0] is not None else None,
            "lat": float(coords[1]) if coords[1] is not None else None,
        })
    df = pl.DataFrame(rows, schema={
        "site_no": pl.Utf8, "name": pl.Utf8, "comid": pl.Utf8,
        "mainstem": pl.Utf8, "lon": pl.Float64, "lat": pl.Float64,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def upstream_set(site: str, distance_km: int = 500) -> set[str]:
    return set(upstream_tributaries(site, distance_km)["site_no"].to_list())
