"""NLDI (Network Linked Data Index) navigation: the river-network topology.

Everything about "upstream of", "on the main stem" and "tributary entering
between X and Y" in this project is derived from these two queries and nothing
else, so the topology is deterministic and auditable (spec 4.3).

  UM = upstream main stem, UT = upstream with tributaries.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CACHE_DIR
from hydro.data.http import fetch_json

NAV_DIR = CACHE_DIR / "nldi"
BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}"


def _navigate(site: str, mode: str, distance_km: int) -> pl.DataFrame:
    path = NAV_DIR / f"{site}_{mode}_{distance_km}.parquet"
    if path.exists():
        return pl.read_parquet(path)

    url = f"{BASE.format(site=site)}/navigation/{mode}/nwissite?distance={distance_km}"
    payload = fetch_json(url)
    feats = payload.get("features", []) if payload else []
    rows = []
    for f in feats:
        p = f.get("properties", {})
        ident = p.get("identifier", "")
        if not ident.startswith("USGS-"):
            continue
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        rows.append(
            {
                "site_no": ident.removeprefix("USGS-"),
                "name": p.get("name"),
                "comid": str(p.get("comid")),
                "mainstem": p.get("mainstem"),
                "lon": float(coords[0]) if coords[0] is not None else None,
                "lat": float(coords[1]) if coords[1] is not None else None,
            }
        )
    df = pl.DataFrame(rows, schema={
        "site_no": pl.Utf8, "name": pl.Utf8, "comid": pl.Utf8,
        "mainstem": pl.Utf8, "lon": pl.Float64, "lat": pl.Float64,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def upstream_mainstem(site: str, distance_km: int = 500) -> pl.DataFrame:
    """Gauges on the main stem strictly upstream of `site` (plus `site`)."""
    return _navigate(site, "UM", distance_km)


def upstream_tributaries(site: str, distance_km: int = 500) -> pl.DataFrame:
    """All gauges upstream of `site` on any tributary (plus `site`)."""
    return _navigate(site, "UT", distance_km)


def upstream_set(site: str, distance_km: int = 500) -> set[str]:
    return set(upstream_tributaries(site, distance_km)["site_no"].to_list())


def mainstem_set(site: str, distance_km: int = 500) -> set[str]:
    return set(upstream_mainstem(site, distance_km)["site_no"].to_list())
