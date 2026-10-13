"""Every analysis constant of the river-discharge application.

Nothing in `hydro/` hard-codes a constant; it lives here and the paper's
appendix quotes it from here. See hydro/README.md for the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CFS_TO_CUMECS = 0.028316846592  # ft^3/s -> m^3/s
SQMI_TO_SQKM = 2.589988110336

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "hydro" / "cache"
OUTPUT_DIR = REPO_ROOT / "experiments" / "river_discharge"
NOTES_DIR = REPO_ROOT.parent / "notes"
NOTES_FIG_DIR = NOTES_DIR / "Figures" / "application"
NOTES_TABLE_DIR = NOTES_DIR / "sections" / "Appendices"


@dataclass(frozen=True)
class Region:
    """A river basin, identified by the USGS gauge at its outlet, restricted to
    one flood-generating season so that snowmelt and convective floods are not
    mixed."""

    name: str
    label: str
    outlet: str
    season_months: tuple[int, ...]
    navigation_km: int = 500  # NLDI upstream navigation distance


REGIONS: tuple[Region, ...] = (
    Region("susquehanna", "Susq.", "01570500", (5, 6, 7, 8, 9, 10, 11)),   # Harrisburg, PA
    Region("wabash", "Wabash", "03377500", (3, 4, 5, 6, 7, 8)),             # Mt. Carmel, IL
    Region("iowa_cedar", "Iowa/Cedar", "05465500", (4, 5, 6, 7, 8, 9)),     # Wapello, IA
)


@dataclass(frozen=True)
class GaugeScreen:
    """Which gauges of a basin enter the analysis at all."""

    min_record_years: int = 25
    min_daily_values: int = 7300
    min_drainage_area_km2: float = 200.0
    # A year of a gauge's record is complete if at most this fraction of its
    # in-season days is missing.
    max_missing_fraction_in_year: float = 0.05
    # Regulation screen: station-name keywords indicating an impoundment or
    # diversion, plus a curated list of gauges immediately below flood-control
    # dams in the three basins.
    regulation_name_keywords: tuple[str, ...] = (
        " BL ", " BLW ", "BELOW ", "DAM", "RESERVOIR", " RES ", "CANAL",
        "DIVERSION", "LAKE ", "LOCK", "OUTLET", "TAILWATER", "AQUEDUCT",
        "SPILLWAY", "POWERPLANT", "PWR", " CN ",
    )
    regulated_sites: tuple[str, ...] = (
        "05454500",  # Iowa R at Iowa City, below Coralville Reservoir
        "05453100",  # Iowa R below Coralville Dam
        "05455500",  # Iowa R below Iowa City
        "03360500",  # Eel R, below Cagles Mill Lake
        "03373500",  # E Fk White R at Shoals, below Monroe Lake
        "01531000",  # Chemung R at Chemung, Cowanesque/Tioga flood control
        "01518862",  # Cowanesque R below Cowanesque Lake
        "01516350",  # Tioga R below Tioga Dam
    )


@dataclass(frozen=True)
class PairRules:
    """Which (T, Y) pairs enter the main design and which disconnected pairs
    the placebo. Every rule is a property of the records and of the network
    topology, never of an estimate."""

    min_overlap_years: int = 40
    area_ratio: tuple[float, float] = (1.2, 2.5)
    max_distance_km: float = 200.0
    # The treatment gauge must be properly heavy-tailed: Hill index of its
    # in-season daily discharge at most this (the infinite-variance regime).
    max_daily_tail_index: float = 2.0
    # The safeguard P: the n gauges nearest to Y that are hydrologically
    # disconnected from both T and Y, with at least this fraction of T's
    # drainage area and a record covering this fraction of the pair's
    # complete years; gauges nested on one stream count once.
    n_proxies: int = 3
    min_proxy_area_ratio: float = 0.02
    min_proxy_record_coverage: float = 0.8
    # The headline pair: the admitted pair with the largest area ratio, i.e.
    # the most room for confounding inflow between the gauges.
    # Placebo: disconnected pairs of the same basin, nearest first.
    placebo_area_ratio: tuple[float, float] = (0.5, 2.0)
    placebo_max_distance_km: float = 100.0
    placebo_n_pairs: int = 40


@dataclass(frozen=True)
class Alignment:
    """Travel-time alignment of Y to T."""

    max_lag_days: int = 6
    # The lag maximises the cross-correlation of log flows on windows around
    # the top `tail_fraction` of T days (the flood-wave lag, not bulk flow).
    tail_fraction: float = 0.05
    tail_window_days: int = 10
    # The implied flood-wave celerity must intersect this band (m/s); a pair
    # whose lag fails the check is excluded.
    celerity_m_per_s: tuple[float, float] = (0.3, 5.0)
    # River distance is approximated by the great-circle distance times this
    # sinuosity factor.
    sinuosity: float = 1.35


@dataclass(frozen=True)
class Decluster:
    """Runs declustering of T: one event per flood, at its peak day. The
    event table is a diagnostic only (pairs.parquet reports the count); every
    estimator is fitted on the daily record."""

    threshold_quantile: float = 0.90
    run_length_days: int = 7


@dataclass(frozen=True)
class Quality:
    # NWIS daily-value qualifiers that drop a day: provisional, estimated,
    # ice-affected, equipment, seasonal, maintenance, discontinued, unknown.
    excluded_qualifiers: tuple[str, ...] = ("P", "e", "Ice", "Eqp", "Ssn", "Mnt", "Dis", "***")


@dataclass(frozen=True)
class Estimator:
    # Everything is fitted on the daily record. Safeguard levels q_P (each
    # proxy below its own q_P-quantile over all days) and the tabulated level.
    q_p_grid: tuple[float, ...] = (0.9, 0.75, 0.5, 0.25, 0.1, 0.05)
    q_p: float = 0.75
    # Certification quantile q_T of T over the safeguard-satisfying days: the
    # largest on this grid that leaves at least `min_events` certified days.
    # A daily 0.995-quantile is roughly a flood peak, hence the deep grid.
    q_t_grid: tuple[float, ...] = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
    # The certified OLS is fitted on whatever survives certification; with a
    # single certified day it is the ratio Y/T at that day.
    min_events: int = 1
    # Hill estimator: fraction of upper order statistics.
    hill_k_fraction: float = 0.10
    # Ablation V3 on the headline pair: a safeguard from a climatically
    # unrelated basin: Iowa River at Marshalltown and near Rowan, Salt Creek
    # near Elberon (Iowa/Cedar; Rowan lies upstream of Marshalltown).
    other_basin_proxies: tuple[str, ...] = ("05451500", "05449500", "05452000")
    # Ablation V2: number of time permutations of the safeguard block.
    n_permutations: int = 50
    # Year-level bootstrap.
    n_bootstrap: int = 400
    interval_level: float = 0.90
    seed: int = 20260731


@dataclass(frozen=True)
class HydroConfig:
    regions: tuple[Region, ...] = REGIONS
    gauges: GaugeScreen = field(default_factory=GaugeScreen)
    pairs: PairRules = field(default_factory=PairRules)
    alignment: Alignment = field(default_factory=Alignment)
    decluster: Decluster = field(default_factory=Decluster)
    quality: Quality = field(default_factory=Quality)
    estimator: Estimator = field(default_factory=Estimator)
    start_date: str = "1930-01-01"
    end_date: str = "2024-12-31"
    # The causal target: a marginal cubic metre past T passes Y one travel
    # time later, so beta = 1 up to attenuation of the daily mean.
    beta_target: float = 1.0


CONFIG = HydroConfig()
REGION_BY_NAME = {r.name: r for r in REGIONS}
