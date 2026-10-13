"""River-network topology over a basin's screened gauges, from NLDI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from hydro.config import CONFIG
from hydro.data import nldi

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class Network:
    """`upstream[g]` is the set of gauges whose water passes gauge g (g itself
    included); `meta` has one row per gauge (station_nm, lat, lon, area_km2)."""

    upstream: dict[str, set[str]]
    meta: pl.DataFrame

    def is_upstream_of(self, a: str, b: str) -> bool:
        return a != b and a in self.upstream.get(b, set())

    def is_disconnected(self, a: str, b: str) -> bool:
        return a != b and not self.is_upstream_of(a, b) and not self.is_upstream_of(b, a)

    def disconnected_from_both(self, a: str, b: str) -> list[str]:
        return sorted(g for g in self.upstream
                      if g not in (a, b) and self.is_disconnected(g, a) and self.is_disconnected(g, b))

    def drop_nested(self, gauges: list[str]) -> list[str]:
        """Collapse gauges on one stream to the most downstream one, keeping order."""
        return [g for g in gauges if not any(self.is_upstream_of(g, o) for o in gauges if o != g)]

    def distance_km(self, a: str, b: str) -> float | None:
        """River distance, approximated as great-circle distance times a sinuosity factor."""
        rows = {r["site_no"]: (r["lat"], r["lon"])
                for r in self.meta.filter(pl.col("site_no").is_in([a, b])).iter_rows(named=True)}
        if len(rows) != 2 or any(v is None for xy in rows.values() for v in xy):
            return None
        return CONFIG.alignment.sinuosity * haversine_km(*rows[a], *rows[b])

    def area(self, site: str) -> float | None:
        row = self.meta.filter(pl.col("site_no") == site)
        return row["area_km2"].item() if row.height else None

    def name(self, site: str) -> str:
        row = self.meta.filter(pl.col("site_no") == site)
        return row["station_nm"].item() if row.height else site


def build_network(meta: pl.DataFrame, distance_km: int) -> Network:
    """Upstream sets (from cached NLDI navigations) for the gauges in `meta`."""
    universe = set(meta["site_no"].to_list())
    upstream = {s: (nldi.upstream_set(s, distance_km) & universe) | {s} for s in sorted(universe)}
    return Network(upstream=upstream, meta=meta)
