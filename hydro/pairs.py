"""Pair selection for the main design and the placebo.

`select_pairs()` lists every connected pair (T upstream of Y) among a basin's
screened gauges with the rule that rejected it or "pass", and returns the
passing pairs with their proxies and panel diagnostics. `placebo_pairs()`
does the same for disconnected pairs. Every rule is a property of the records
and the network topology (hydro/config.py, `PairRules`), never of an estimate.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR, REGION_BY_NAME, Region
from hydro.data import nwis
from hydro.data.network import Network
from hydro.estimation import hill_estimate
from hydro.preprocess.panel import PairSpec, Panel, build_panel
from hydro.preprocess.screening import good_years, region_network

RULES = CONFIG.pairs


@lru_cache(maxsize=None)
def daily_tail_index(site: str, region_name: str) -> float:
    """Hill index of a gauge's in-season daily discharge over its whole record."""
    region = REGION_BY_NAME[region_name]
    df = nwis.daily_discharge(site, CONFIG.start_date, CONFIG.end_date)
    df = df.filter(pl.col("date").dt.month().is_in(list(region.season_months)))
    return hill_estimate(df["q"].to_numpy())


def proxies_for(net: Network, region: Region, t: str, y: str, n: int = RULES.n_proxies) -> list[str]:
    """The safeguard P: the `n` gauges nearest to Y (river distance) that are
    disconnected from both T and Y, drain at least `min_proxy_area_ratio` of
    T's area, cover `min_proxy_record_coverage` of the pair's complete years,
    one gauge per stream."""
    base = good_years(t, region.name) & good_years(y, region.name)
    min_area = RULES.min_proxy_area_ratio * (net.area(t) or 0.0)
    cand = [
        s for s in net.disconnected_from_both(t, y)
        if (net.area(s) or 0.0) >= min_area
        and net.distance_km(s, y) is not None
        and (not base or len(good_years(s, region.name) & base) >= RULES.min_proxy_record_coverage * len(base))
    ]
    cand.sort(key=lambda s: net.distance_km(s, y))
    return net.drop_nested(cand)[:n]


def spec_for(row: dict) -> PairSpec:
    return PairSpec(row["pair"], REGION_BY_NAME[row["region"]], row["t"], row["y"], row["proxies"].split(","))


def panel_for(row: dict, **kw) -> Panel | None:
    return build_panel(spec_for(row), distance_km=row["distance_km"], **kw)


def _base_row(region: Region, net: Network, t: str, y: str) -> dict:
    return {
        "region": region.name, "pair": f"{t}->{y}", "t": t, "y": y,
        "t_name": net.name(t), "y_name": net.name(y),
        "area_t_km2": net.area(t), "area_y_km2": net.area(y),
        "area_ratio": net.area(y) / net.area(t),
        "overlap_years": len(good_years(t, region.name) & good_years(y, region.name)),
        "distance_km": net.distance_km(t, y),
    }


def _connected_candidates(region: Region, net: Network) -> list[dict]:
    """Every (T, Y) with T upstream of Y, with the cheap rules applied."""
    lo, hi = RULES.area_ratio
    rows = []
    sites = sorted(s for s in net.upstream if net.area(s))
    for t in sites:
        for y in sites:
            if not net.is_upstream_of(t, y):
                continue
            r = _base_row(region, net, t, y)
            if not lo <= r["area_ratio"] <= hi:
                r["screen_result"] = f"area ratio outside [{lo}, {hi}]"
            elif r["overlap_years"] < RULES.min_overlap_years:
                r["screen_result"] = f"fewer than {RULES.min_overlap_years} common complete years"
            elif r["distance_km"] is None or r["distance_km"] > RULES.max_distance_km:
                r["screen_result"] = f"more than {RULES.max_distance_km:.0f} km apart"
            else:
                r["screen_result"] = "pass"
            rows.append(r)
    return rows


def select_pairs(verbose: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(candidates, pairs): every connected pair with its screen result, and
    the passing pairs with proxies, lag, events and tail index."""
    cands, pairs = [], []
    for region in CONFIG.regions:
        net = region_network(region.name)
        for r in _connected_candidates(region, net):
            if r["screen_result"] == "pass":
                proxies = proxies_for(net, region, r["t"], r["y"])
                r["proxies"] = ",".join(proxies)
                r["daily_tail_index_t"] = daily_tail_index(r["t"], region.name)
                if len(proxies) < RULES.n_proxies:
                    r["screen_result"] = f"fewer than {RULES.n_proxies} proxy gauges"
                elif not r["daily_tail_index_t"] <= RULES.max_daily_tail_index:
                    r["screen_result"] = f"daily tail index above {RULES.max_daily_tail_index}"
                else:
                    p = panel_for(r)
                    if p is None:
                        r["screen_result"] = "common record too short"
                    elif not p.lag.plausible:
                        r["screen_result"] = "travel-time lag fails the celerity check"
                    else:
                        r.update({
                            "proxy_names": "; ".join(net.name(s) for s in proxies),
                            "proxy_distance_to_y_km": ",".join(f"{net.distance_km(s, r['y']):.0f}" for s in proxies),
                            "lag_days": p.lag.lag_days, "lag_correlation": p.lag.correlation,
                            "n_daily": p.n_daily, "n_events": p.n_events,
                            "decluster_threshold_m3s": p.threshold_u,
                            "dropped_on_quality": p.dropped_quality,
                            "event_tail_index_t": hill_estimate(p.events["T"].to_numpy()),
                        })
                        pairs.append(r)
            cands.append(r)
        if verbose:
            n_pass = sum(c["screen_result"] == "pass" for c in cands if c["region"] == region.name)
            print(f"{region.name}: {len(net.upstream)} screened gauges, "
                  f"{sum(c['region'] == region.name for c in cands)} connected candidates, {n_pass} pass")
    return (pl.DataFrame(cands, infer_schema_length=None),
            pl.DataFrame(pairs, infer_schema_length=None).sort("region", "pair"))


def placebo_pairs(verbose: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(candidates, pairs) of the disconnected-pair placebo: gauges of one
    basin with no flow path between them, comparable areas, close together,
    with a full proxy vector and enough events; the nearest `placebo_n_pairs`."""
    lo, hi = RULES.placebo_area_ratio
    cands = []
    for region in CONFIG.regions:
        net = region_network(region.name)
        sites = sorted(s for s in net.upstream if net.area(s))
        for i, a in enumerate(sites):
            for b in sites[i + 1:]:
                if not net.is_disconnected(a, b):
                    continue
                r = _base_row(region, net, a, b)
                if not lo <= r["area_ratio"] <= hi:
                    r["screen_result"] = f"area ratio outside [{lo}, {hi}]"
                elif r["overlap_years"] < RULES.min_overlap_years:
                    r["screen_result"] = f"fewer than {RULES.min_overlap_years} common complete years"
                elif r["distance_km"] is None or r["distance_km"] > RULES.placebo_max_distance_km:
                    r["screen_result"] = f"more than {RULES.placebo_max_distance_km:.0f} km apart"
                else:
                    r["screen_result"] = "pass"
                cands.append(r)
    cands.sort(key=lambda r: r["distance_km"] if r["distance_km"] is not None else 1e9)
    pairs = []
    for r in cands:
        if len(pairs) >= RULES.placebo_n_pairs:
            break
        if r["screen_result"] != "pass":
            continue
        region, net = REGION_BY_NAME[r["region"]], region_network(r["region"])
        proxies = proxies_for(net, region, r["t"], r["y"])
        r["proxies"] = ",".join(proxies)
        if len(proxies) < RULES.n_proxies:
            r["screen_result"] = f"fewer than {RULES.n_proxies} proxy gauges"
            continue
        p = panel_for(r, lag=0)
        if p is None:
            r["screen_result"] = "common record too short"
            continue
        r.update({"n_events": p.n_events, "n_daily": p.n_daily})
        pairs.append(r)
    if verbose:
        n_pass = sum(c["screen_result"] == "pass" for c in cands)
        print(f"placebo: {len(cands)} disconnected candidates, {n_pass} pass the screens, "
              f"{len(pairs)} used (nearest first)")
    return pl.DataFrame(cands, infer_schema_length=None), pl.DataFrame(pairs, infer_schema_length=None)


# ------------------------------------------------------------- persisted


def main_pairs() -> pl.DataFrame:
    return pl.read_parquet(OUTPUT_DIR / "pairs.parquet")


def headline_pair() -> dict:
    """The admitted pair with the largest drainage-area ratio, i.e. the most
    room for confounding inflow between the gauges; ties by common years."""
    return main_pairs().sort(["area_ratio", "overlap_years", "pair"],
                             descending=[True, True, False]).row(0, named=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cands, pairs = select_pairs()
    cands.write_parquet(OUTPUT_DIR / "pair_candidates.parquet")
    pairs.write_parquet(OUTPUT_DIR / "pairs.parquet")
    print(cands.group_by("screen_result").len().sort("len", descending=True))
    print(f"{pairs.height} pairs: " + ", ".join(f"{r['region']} {r['len']}" for r in
                                               pairs.group_by("region").len().sort("region").to_dicts()))
    pc, pp = placebo_pairs()
    pc.write_parquet(OUTPUT_DIR / "placebo_candidates.parquet")
    pp.write_parquet(OUTPUT_DIR / "placebo_pairs.parquet")


if __name__ == "__main__":
    main()
