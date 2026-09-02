"""Pair selection, shared by every experiment (see `preregistration.md`).

The rules here use only columns of `station_pairs.parquet` -- record length,
drainage area, NLDI topology, the regulation screen -- and never an estimate.
"""

from __future__ import annotations

import polars as pl

from hydro.config import CACHE_DIR, CONFIG, OUTPUT_DIR, Region
from hydro.preprocess.panel import PairSpec
from hydro.preprocess.screening import good_years

MIN_AREA_RATIO = 1.2
MAX_AREA_RATIO = 2.5
# The pre-registered rule "ungauged intervening area <= 10%" was computed
# from a tributary list with nested gauges, whose areas double-counted, so it
# read 0.0 for every pair and never excluded anything. Computed correctly the
# median is 0.35 and the rule keeps two pairs. Rather than tune a threshold
# after seeing estimates we drop the screen and REPORT the fraction per pair:
# it is the quantity that should predict the residual bias, and the paper
# treats it as such (preregistration.md, amendment 3).
MAX_UNGAUGED_GAIN = None
# One non-nested tributary can carry most of the intervening inflow (the West
# Branch above Sunbury), so a pair needs at least one, not three. What the
# safeguard must do is see most of the water arriving between X and Y, so the
# rule that replaces both is on the area the safeguard vector actually covers:
# the (<= n_safeguard_gauges) gauges that enter the certificate must drain at
# least this fraction of the intervening area. Fixed from the pair table
# before any estimate on the new pair set was computed (amendment 3).
MIN_TRIBUTARIES = 1
MIN_SAFEGUARD_AREA_COVERAGE = 0.5
MIN_OVERLAP_YEARS = 40
MAX_DISTANCE_KM = 200.0
MIN_SAFEGUARD_COVERAGE = 0.8

REGION_BY_NAME = {r.name: r for r in CONFIG.regions}


def region_of(name: str) -> Region:
    return REGION_BY_NAME[name]


def design_a_candidates() -> pl.DataFrame:
    """Connected pairs passing the pre-registered Design-A filter, best first."""
    return (
        pl.read_parquet(OUTPUT_DIR / "station_pairs.parquet")
        .filter(
            (pl.col("screen_result") == "pass")
            & pl.col("area_ratio").is_between(MIN_AREA_RATIO, MAX_AREA_RATIO)
            & (pl.col("n_tributaries") >= MIN_TRIBUTARIES)
            & (pl.col("overlap_years") >= MIN_OVERLAP_YEARS)
            & (pl.col("distance_km") <= MAX_DISTANCE_KM)
        )
        .sort(["overlap_years", "distance_km"], descending=[True, False])
        .with_columns(
            (pl.col("x") + "->" + pl.col("y")).alias("pair"),
            pl.col("area_ratio").alias("area_ratio_reference"),
        )
        .with_columns(
            pl.struct(pl.all())
            .map_elements(
                lambda r: len(select_safeguards(r, CONFIG.estimator.n_safeguard_gauges, "tributaries")),
                return_dtype=pl.Int64,
            )
            .alias("n_safeguards_used"),
            pl.struct(pl.all())
            .map_elements(safeguard_area_coverage, return_dtype=pl.Float64)
            .alias("safeguard_area_coverage"),
        )
        # Rule 4 of the pre-registration asked for a full n_safeguard_gauges
        # safeguard vector. Two things changed that (see preregistration.md,
        # amendment 2): safeguards are restricted to gauges that cover the
        # pair's record, and nested gauges on one tributary are collapsed to
        # the downstream one (`select_safeguards`). A pair whose intervening
        # inflow arrives through one or two large tributaries then has one or
        # two safeguards, not three, and dropping it would discard exactly the
        # best-covered pairs. We keep every pair with at least one usable
        # safeguard and report the count. Evaluated on the pair table alone.
        .filter(
            (pl.col("n_safeguards_used") >= 1)
            & (pl.col("safeguard_area_coverage") >= MIN_SAFEGUARD_AREA_COVERAGE)
        )
    )


def safeguard_area_coverage(row: dict) -> float:
    """Fraction of the area gained between X and Y drained by the safeguard
    gauges that enter the certificate (non-nested, record-covering, largest
    first, at most n_safeguard_gauges)."""
    areas = _site_areas(row["region"])
    w = select_safeguards(row, CONFIG.estimator.n_safeguard_gauges, "tributaries")
    gain = max(row["area_y_km2"] - row["area_x_km2"], 1e-9)
    return sum(areas.get(s) or 0.0 for s in w) / gain


def disconnected_candidates(max_pairs: int = 20) -> pl.DataFrame:
    """Placebo pairs for E2, best first.

    Requires a long common record, comparable catchment sizes (so a naive
    regression is not trivially far from 1 for scale reasons alone) and enough
    disconnected safeguard gauges to fill the 3-coordinate safeguard vector.
    Closest pairs first: the placebo is only interesting where the two basins
    genuinely share weather.
    """
    return (
        pl.read_parquet(OUTPUT_DIR / "disconnected_pairs.parquet")
        .filter(
            (pl.col("screen_result") == "pass")
            & (pl.col("n_safeguards") >= MIN_TRIBUTARIES)
            & (pl.col("overlap_years") >= MIN_OVERLAP_YEARS)
            & pl.col("area_ratio").is_between(0.5, 2.0)
            & (pl.col("distance_km") <= 100.0)
        )
        .sort(["distance_km"])
        .head(max_pairs)
        .with_columns((pl.col("x") + "->" + pl.col("y")).alias("pair"))
    )


def _site_areas(region_name: str) -> dict[str, float]:
    df = pl.read_parquet(CACHE_DIR / "screen" / f"{region_name}_sites.parquet")
    return dict(zip(df["site_no"], df["area_km2"]))


def select_safeguards(row: dict, n: int, field: str) -> list[str]:
    """The `n` safeguard gauges: largest catchment *among those that are gauged
    over the pair's own record*.

    The panel is built on the intersection of every site's record, so a large
    tributary with a short record destroys more sample than it buys safeguard
    power: taking the three largest outright cut some pairs from 95 usable
    years to 13. We therefore require a candidate to cover at least
    `MIN_SAFEGUARD_COVERAGE` of the X-Y common good years and rank the
    survivors by drainage area. Record length is a data-availability property,
    never an estimate, so this stays inside the pre-registration.
    """
    region = region_of(row["region"])
    cand = [s for s in row[field].split(",") if s]
    base = good_years(row["x"], region) & good_years(row["y"], region)
    if not base:
        return cand[:n]
    areas = _site_areas(row["region"])
    ok = [s for s in cand if len(good_years(s, region) & base) >= MIN_SAFEGUARD_COVERAGE * len(base)]
    ok.sort(key=lambda s: -(areas.get(s) or 0.0))
    return drop_nested(ok or cand, network_for(row["region"]))[:n]


def drop_nested(gauges: list[str], net) -> list[str]:
    """Collapse gauges that lie on the same stream to the downstream one.

    Two gauges on one tributary measure the same water twice: their records
    are nearly collinear, so as safeguard coordinates the pair costs a factor
    q_W of certified sample for no extra monitoring, and as regressors they
    make the adjustment coefficients uninterpretable. Order is preserved, so
    a list sorted by area or distance stays sorted.
    """
    keep: list[str] = []
    for g in gauges:
        if any(net.is_upstream_of(g, o) or net.is_upstream_of(o, g) for o in gauges if o != g):
            # Keep only the most downstream gauge of each nested chain.
            if any(net.is_upstream_of(g, o) for o in gauges if o != g):
                continue
        keep.append(g)
    return keep


_NETS: dict[str, object] = {}


def network_for(region_name: str):
    from hydro.preprocess.screening import region_network

    if region_name not in _NETS:
        _NETS[region_name] = region_network(region_of(region_name))
    return _NETS[region_name]


def nearest_disconnected_safeguards(row: dict, n: int) -> list[str]:
    """Safeguards for the placebo design: the `n` gauges disconnected from both
    X and Y that lie **closest to Y**.

    This is the structural analogue of Design A, where W is the tributary
    entering just above Y: what the safeguard has to certify is that the latent
    driver acting on *Y's* catchment was not extreme, and a gauge 200 km away in
    another sub-basin sees a different part of the storm field. Ranking the same
    admissible set by drainage area instead is reported as `safeguard_rule =
    "largest"`, and the difference between the two is evidence about the paper's
    "W must monitor the latent source" caveat rather than a tuning knob.
    """
    net = network_for(row["region"])
    region = region_of(row["region"])
    base = good_years(row["x"], region) & good_years(row["y"], region)
    min_area = CONFIG.screening.min_tributary_area_ratio * (net.area(row["x"]) or 0.0)
    cand = [
        s for s in net.disconnected_from_both(row["x"], row["y"])
        if (net.area(s) or 0.0) >= min_area
        and net.distance_km(s, row["y"]) is not None
        and (not base or len(good_years(s, region) & base) >= MIN_SAFEGUARD_COVERAGE * len(base))
    ]
    cand.sort(key=lambda s: net.distance_km(s, row["y"]))
    return drop_nested(cand, net)[:n]


def spec_from_row(
    row: dict, *, kind: str = "connected", reverse: bool = False,
    safeguard_rule: str = "largest",
) -> PairSpec:
    n_w = CONFIG.estimator.n_safeguard_gauges
    w_field = "tributaries" if kind in ("connected", "proxy") else "safeguards"
    if kind == "proxy" or (kind == "disconnected" and safeguard_rule == "nearest"):
        # kind="proxy": a connected pair certified with the pure-proxy
        # safeguard -- gauges disconnected from both X and Y, which share the
        # storm but feed neither. Same rule as the placebo's safeguard.
        w = nearest_disconnected_safeguards(row, n_w)
    else:
        w = select_safeguards(row, n_w, w_field)
    x, y = (row["y"], row["x"]) if reverse else (row["x"], row["y"])
    return PairSpec(
        label=f"{x}->{y}",
        region=region_of(row["region"]),
        x=x,
        y=y,
        w=w,
        z=row.get("z"),
        kind="reverse" if reverse else ("connected" if kind == "proxy" else kind),
    )


def panel_for(row: dict, **kw):
    from hydro.preprocess.panel import build_panel

    return build_panel(spec_from_row(row, **kw), CONFIG, distance_km=row["distance_km"])


HEADLINE_MIN_AREA_RATIO = 2.0
HEADLINE_Q_W = 0.25


def headline_row() -> dict:
    """The pair the headline figures are drawn on (see preregistration.md).

    Rule: most declustered events among the Design-A candidates that admit
    both safeguard structures with a full vector -- n_safeguard_gauges
    non-nested tributaries and as many disconnected proxy gauges -- and whose
    drainage-area ratio is at least HEADLINE_MIN_AREA_RATIO, so that the
    confounded reference sits well away from the target, and that keep at
    least `min_n_cert` certified events at the tabulated safeguard level
    HEADLINE_Q_W in both structures, so that the headline figures can show
    both at one level. All are properties of the records and of the safeguard
    sample sizes, never of an estimate (amendment 4). Outlet pairs never
    qualify: every gauge in the basin is upstream of the outlet, so no proxy
    exists.
    """
    from hydro.estimation import certified_threshold, fit_certificate

    est = CONFIG.estimator
    n_w = est.n_safeguard_gauges

    def reaches(panel) -> bool:
        for q_x in est.certification_quantile_grid:
            v = certified_threshold(panel.events, panel.w_cols, q_x, HEADLINE_Q_W, "per_variable")
            if fit_certificate(panel.events, panel.w_cols, level=v,
                               safeguard_quantile=HEADLINE_Q_W).n_cert >= est.min_n_cert:
                return True
        return False

    best, best_n = None, -1
    for row in design_a_candidates().iter_rows(named=True):
        if row["n_safeguards_used"] < n_w or row["area_ratio"] < HEADLINE_MIN_AREA_RATIO:
            continue
        if len(nearest_disconnected_safeguards(row, n_w)) < n_w:
            continue
        p, pp = panel_for(row), panel_for(row, kind="proxy")
        if p is None or pp is None or not (reaches(p) and reaches(pp)):
            continue
        if p.n_events > best_n:
            best, best_n = row, p.n_events
    assert best is not None
    return best
