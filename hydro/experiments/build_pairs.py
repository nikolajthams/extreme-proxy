"""Build the station-pair table (spec 4.3).

`uv run python -m hydro.experiments.build_pairs`

Writes two artifacts under experiments/river_discharge/:
  * `station_pairs.parquet` / `.csv`  -- every candidate connected pair, with
    the screen it passed or the rule that rejected it.
  * `disconnected_pairs.parquet`      -- the placebo pairs for E2.

Every row carries a `screen_result` and a `provenance` column. No pair is ever
removed after the fact: rejected pairs stay in the table with their reason.
"""

from __future__ import annotations

import json
from itertools import combinations

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR, Region
from hydro.data.network import Network
from hydro.preprocess.screening import good_years, region_network

PROVENANCE = (
    "USGS NWIS daily values (param 00060) + NWIS site service; topology from "
    "the USGS NLDI navigation API (UT/UM). Screens: hydro/config.py "
    "ScreeningConfig, fixed before any estimation."
)


def _tributary_candidates(net: Network, x: str, y: str, area_x: float) -> list[str]:
    scr = CONFIG.screening
    out = []
    for t in net.intervening_tributaries(x, y):
        a = net.area(t)
        if a is None:
            continue
        if a >= scr.min_tributary_area_ratio * area_x:
            out.append(t)
    # Largest first: safeguard power comes from the big intervening catchments.
    return sorted(out, key=lambda t: -(net.area(t) or 0))


def connected_pairs(region: Region, net: Network) -> pl.DataFrame:
    scr = CONFIG.screening
    sites = sorted(net.upstream)
    gy = {s: good_years(s, region) for s in sites}

    rows = []
    for x in sites:
        for y in sites:
            if not net.is_upstream_of(x, y):
                continue
            ax, ay = net.area(x), net.area(y)
            if ax is None or ay is None or ax <= 0:
                continue
            ratio = ay / ax
            tribs = _tributary_candidates(net, x, y, ax)
            n_overlap = len(gy[x] & gy[y])
            trib_area = sum(net.area(t) or 0.0 for t in tribs)
            zs = net.upstream_mainstem_of(x)
            z = max(zs, key=lambda s: net.area(s) or 0.0) if zs else None

            if ratio < scr.min_area_ratio_y_over_x:
                res = "area ratio too small (gauges effectively co-located)"
            elif ratio > scr.max_area_ratio_y_over_x:
                res = "area ratio too large (upstream gauge negligible)"
            elif not tribs:
                res = "no gauged intervening tributary (Design A impossible)"
            elif n_overlap < scr.min_overlap_years:
                res = f"overlap {n_overlap} y < {scr.min_overlap_years} y"
            else:
                res = "pass"

            rows.append(
                {
                    "region": region.name,
                    "x": x,
                    "y": y,
                    "x_name": net.name(x),
                    "y_name": net.name(y),
                    "area_x_km2": ax,
                    "area_y_km2": ay,
                    "area_ratio": ratio,
                    "n_tributaries": len(tribs),
                    "tributaries": ",".join(tribs),
                    "tributary_area_km2": trib_area,
                    # Fraction of the area gained between X and Y that is NOT
                    # gauged by a safeguard tributary: the paper's "W must cover
                    # every latent source" caveat, quantified.
                    "ungauged_gain_fraction": max(0.0, 1.0 - trib_area / max(ay - ax, 1e-9)),
                    "overlap_years": n_overlap,
                    "distance_km": net.distance_km(x, y),
                    "z": z,
                    "screen_result": res,
                    "provenance": PROVENANCE,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def disconnected_pairs(region: Region, net: Network, max_pairs: int = 40) -> pl.DataFrame:
    """Placebo pairs: neither gauge is upstream of the other (true effect = 0).

    They must still share the regional weather driver, so we require them to be
    geographically close (both are inside the same basin by construction).
    Safeguard gauges are gauges disconnected from BOTH members of the pair.
    """
    scr = CONFIG.screening
    sites = sorted(net.upstream)
    gy = {s: good_years(s, region) for s in sites}
    rows = []
    for a, b in combinations(sites, 2):
        if not net.is_disconnected(a, b):
            continue
        aa, ab = net.area(a), net.area(b)
        if aa is None or ab is None:
            continue
        n_overlap = len(gy[a] & gy[b])
        d = net.distance_km(a, b)
        safeguards = [
            s for s in net.disconnected_from_both(a, b)
            if (net.area(s) or 0) >= scr.min_tributary_area_ratio * aa
        ]
        safeguards = sorted(safeguards, key=lambda s: -(net.area(s) or 0))
        if n_overlap < scr.min_overlap_years:
            res = f"overlap {n_overlap} y < {scr.min_overlap_years} y"
        elif not safeguards:
            res = "no disconnected safeguard gauge available"
        elif d is not None and d > 300:
            res = "gauges too far apart to share a weather system"
        else:
            res = "pass"
        rows.append(
            {
                "region": region.name,
                "x": a,
                "y": b,
                "x_name": net.name(a),
                "y_name": net.name(b),
                "area_x_km2": aa,
                "area_y_km2": ab,
                "area_ratio": ab / aa,
                "n_safeguards": len(safeguards),
                "safeguards": ",".join(safeguards[:4]),
                "overlap_years": n_overlap,
                "distance_km": d,
                "screen_result": res,
                "provenance": PROVENANCE,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn, disc = [], []
    for region in CONFIG.regions:
        net = region_network(region)
        c = connected_pairs(region, net)
        d = disconnected_pairs(region, net)
        print(
            f"{region.name}: {len(net.upstream)} screened gauges, "
            f"{c.height} connected candidates ({c.filter(pl.col('screen_result') == 'pass').height} pass), "
            f"{d.height} disconnected candidates ({d.filter(pl.col('screen_result') == 'pass').height} pass)"
        )
        conn.append(c)
        disc.append(d)

    pairs = pl.concat(conn, how="vertical_relaxed")
    dpairs = pl.concat(disc, how="vertical_relaxed")
    pairs.write_parquet(OUTPUT_DIR / "station_pairs.parquet")
    pairs.write_csv(OUTPUT_DIR / "station_pairs.csv")
    dpairs.write_parquet(OUTPUT_DIR / "disconnected_pairs.parquet")

    summary = {
        "connected_candidates": pairs.height,
        "connected_pass": pairs.filter(pl.col("screen_result") == "pass").height,
        "disconnected_candidates": dpairs.height,
        "disconnected_pass": dpairs.filter(pl.col("screen_result") == "pass").height,
        "reject_reasons": dict(
            pairs.group_by("screen_result").len().sort("len", descending=True).iter_rows()
        ),
    }
    (OUTPUT_DIR / "pair_table_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
