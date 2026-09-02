"""River-network topology derived from NLDI, and the geometry helpers built on it.

The only primitive is `UT(g)` = the set of gauges upstream of `g` on any
tributary (NLDI navigation "UT"). Everything else -- "X is upstream of Y",
"T is a tributary entering between X and Y", "A and B are hydrologically
disconnected" -- is a set expression over those, so the topology is
deterministic and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from hydro.data import nldi

EARTH_RADIUS_KM = 6371.0
# River distance is not returned by NLDI, so we approximate it as the
# great-circle distance inflated by a channel sinuosity factor. This is only
# used for the flood-wave celerity plausibility check (spec 5.1 step 2), where
# the accepted band is wide, so the approximation is harmless.
SINUOSITY = 1.35


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class Network:
    """Upstream sets for a fixed collection of gauges."""

    upstream: dict[str, set[str]]   # site -> set of gauges upstream of it (incl. itself)
    mainstem: set[str]              # gauges on the region outlet's main stem
    meta: pl.DataFrame              # one row per site: name, lat, lon, area_km2, ...

    # ------------------------------------------------------------ predicates
    def is_upstream_of(self, a: str, b: str) -> bool:
        """True if water at gauge `a` flows past gauge `b`."""
        return a != b and a in self.upstream.get(b, set())

    def is_disconnected(self, a: str, b: str) -> bool:
        return a != b and not self.is_upstream_of(a, b) and not self.is_upstream_of(b, a)

    # ------------------------------------------------------------ selections
    def intervening_tributaries(self, x: str, y: str) -> list[str]:
        """Gauges on tributaries joining the stem strictly between `x` and `y`.

        T qualifies iff T is upstream of Y, NOT upstream of X, and X is NOT
        upstream of T. The last condition removes main-stem gauges between X
        and Y, which are mediators (descendants of X) and therefore invalid
        safeguards.

        Only *maximal* such gauges are returned -- if two qualifying gauges sit
        on the same tributary, the downstream one supersedes the upstream one.
        Otherwise nested gauges would be counted twice in the intervening-area
        bookkeeping and would add near-duplicate coordinates to the safeguard
        vector, shrinking the certified set for no information gain.
        """
        ux, uy = self.upstream.get(x, set()), self.upstream.get(y, set())
        cand = [
            t
            for t in uy
            if t not in ux
            and t not in (x, y)
            and not self.is_upstream_of(x, t)
        ]
        cand_set = set(cand)
        return sorted(
            t for t in cand
            if not any(self.is_upstream_of(t, o) for o in cand_set if o != t)
        )

    def upstream_mainstem_of(self, x: str) -> list[str]:
        """Main-stem gauges strictly upstream of X: candidate instruments Z."""
        return sorted(g for g in self.upstream.get(x, set()) if g != x and g in self.mainstem)

    def disconnected_from_both(self, a: str, b: str) -> list[str]:
        """Gauges in the region hydrologically disconnected from both a and b.

        These are the admissible safeguards for the disconnected-pair placebo:
        they share the regional weather driver U but are neither ancestors nor
        descendants of either gauge.
        """
        return sorted(
            g
            for g in self.upstream
            if g not in (a, b) and self.is_disconnected(g, a) and self.is_disconnected(g, b)
        )

    # ------------------------------------------------------------ geometry
    def distance_km(self, a: str, b: str) -> float | None:
        row = self.meta.filter(pl.col("site_no").is_in([a, b]))
        if row.height != 2:
            return None
        d = {r["site_no"]: (r["lat"], r["lon"]) for r in row.iter_rows(named=True)}
        if any(v is None for pair in d.values() for v in pair):
            return None
        return SINUOSITY * haversine_km(*d[a], *d[b])

    def area(self, site: str) -> float | None:
        row = self.meta.filter(pl.col("site_no") == site)
        return row["area_km2"].item() if row.height else None

    def name(self, site: str) -> str:
        row = self.meta.filter(pl.col("site_no") == site)
        return row["station_nm"].item() if row.height else site


def build_network(sites: list[str], mainstem: set[str], meta: pl.DataFrame,
                  distance_km: int = 500) -> Network:
    """Fetch UT() for each site (cached) and assemble the network."""
    universe = set(sites)
    upstream: dict[str, set[str]] = {}
    for s in sorted(universe):
        ut = nldi.upstream_set(s, distance_km)
        upstream[s] = (ut & universe) | {s}
    return Network(upstream=upstream, mainstem=mainstem & universe, meta=meta)
