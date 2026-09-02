"""All tuning constants for the river-discharge application (spec section 5).

Nothing in `hydro/` may hard-code an analysis constant; it belongs here.
"""

from dataclasses import dataclass, field
from pathlib import Path

CFS_TO_CUMECS = 0.028316846592  # ft^3/s -> m^3/s
SQMI_TO_SQKM = 2.589988110336

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "hydro" / "cache"
OUTPUT_DIR = REPO_ROOT / "experiments" / "river_discharge"


@dataclass(frozen=True)
class Region:
    """A river basin, identified by the gauge at its downstream outlet.

    `outlet` is the NWIS site id from which the NLDI upstream navigations are run.
    `season_months` restricts the analysis to a single flood-generating regime
    (spec 5.3): mixing snowmelt and convective floods violates the
    single-mechanism premise.
    """

    name: str
    outlet: str
    season_months: tuple[int, ...]
    season_label: str
    # NLDI navigation distance in km along the flow network.
    navigation_km: int = 500
    rationale: str = ""


# --------------------------------------------------------------------------
# Regions. Chosen on hydrological grounds only (record length, basin size,
# known dominance of basin-wide frontal/synoptic storms => strong latent U),
# BEFORE any beta was estimated. See hydro/experiments/preregistration.md.
# --------------------------------------------------------------------------
REGIONS: tuple[Region, ...] = (
    Region(
        name="susquehanna",
        outlet="01570500",  # Susquehanna River at Harrisburg, PA
        season_months=(5, 6, 7, 8, 9, 10, 11),
        season_label="warm season (May-Nov)",
        rationale=(
            "Large (>60,000 km2) Appalachian basin whose floods of record are "
            "basin-wide tropical/frontal systems (Agnes 1972, Ivan 2004, Lee 2011); "
            "summer convective storms produce strictly local floods. That contrast "
            "is exactly the latent-U vs local-driver contrast the certificate needs. "
            "Dense long-record gauge network, mostly run-of-river."
        ),
    ),
    Region(
        name="iowa_cedar",
        outlet="05465500",  # Iowa River at Wapello, IA
        season_months=(4, 5, 6, 7, 8, 9),
        season_label="warm season (Apr-Sep)",
        rationale=(
            "Midwestern basin dominated by organised mesoscale convective systems "
            "which are basin-wide in spring (1993, 2008) and local in mid-summer. "
            "Coralville Reservoir sits on the Iowa main stem and is screened out."
        ),
    ),
    Region(
        name="wabash",
        outlet="03377500",  # Wabash River at Mt. Carmel, IL
        season_months=(3, 4, 5, 6, 7, 8),
        season_label="spring-summer (Mar-Aug)",
        rationale=(
            "Large low-relief basin; frontal systems load the whole basin at once. "
            "Long records back to the 1920s on both main stem and tributaries."
        ),
    ),
)


@dataclass(frozen=True)
class ScreeningConfig:
    """Spec 4.3: deterministic, auditable pair selection."""

    min_overlap_years: int = 20
    min_coverage_fraction: float = 0.90
    max_missing_fraction_in_any_year: float = 0.05
    # Minimum drainage area (km2) for a gauge to enter the candidate set at all.
    min_drainage_area_km2: float = 200.0
    # A tributary counts towards the safeguard only if it is at least this
    # fraction of the upstream gauge's drainage area (tiny creeks carry no
    # information about a basin-wide storm).
    min_tributary_area_ratio: float = 0.02
    # Pair must gain area going downstream, but not so much that the upstream
    # gauge is irrelevant.
    min_area_ratio_y_over_x: float = 1.02
    max_area_ratio_y_over_x: float = 6.0
    # Regulation screen (spec 4.3). We have no programmatic GAGES-II feed, so the
    # screen is (a) station-name keywords indicating an impoundment/diversion and
    # (b) a curated list of gauges immediately below major flood-control dams.
    # This is weaker than GAGES-II and is recorded as such in the pair table.
    regulation_name_keywords: tuple[str, ...] = (
        " BL ", " BLW ", "BELOW ", "DAM", "RESERVOIR", " RES ", "CANAL",
        "DIVERSION", "LAKE ", "LOCK", "OUTLET", "TAILWATER", "AQUEDUCT",
        "SPILLWAY", "POWERPLANT", "PWR", " CN ",
    )
    # Gauges known to sit immediately downstream of a major flood-control
    # impoundment in the selected regions. Fixed before estimation.
    regulated_sites: tuple[str, ...] = (
        "05454500",  # Iowa R at Iowa City -- below Coralville Reservoir
        "05453100",  # Iowa R below Coralville Dam
        "05455500",  # Iowa R below Iowa City
        "03360500",  # Eel R -- below Cagles Mill Lake influence
        "03373500",  # E Fk White R at Shoals, below Monroe Lake
        "01531000",  # Chemung R at Chemung -- Cowanesque/Tioga flood control
        "01518862",  # Cowanesque R below Cowanesque Lake
        "01516350",  # Tioga R below Tioga Dam
    )


@dataclass(frozen=True)
class AlignmentConfig:
    """Spec 5.1: travel-time alignment."""

    max_lag_days: int = 6
    # The lag is estimated from the flood-wave regime only, not bulk flow:
    # cross-correlation is computed on windows around the top
    # `tail_fraction` of X.
    tail_fraction: float = 0.05
    tail_window_days: int = 10
    # Plausible flood-wave celerity band (m/s) used for the sanity check.
    min_celerity_m_per_s: float = 0.3
    max_celerity_m_per_s: float = 5.0
    # Y is either taken at t+lag ("point") or as the max over
    # [t+lag-w, t+lag+w] ("window").
    window_half_width_days: int = 1
    default_definition: str = "point"


@dataclass(frozen=True)
class DeclusterConfig:
    """Spec 5.2: runs declustering."""

    run_length_days: int = 7
    run_length_sweep: tuple[int, ...] = (3, 5, 7, 14)
    # Declustering threshold: quantile of in-season daily X. Every downstream
    # certification threshold v must be at least this level.
    threshold_quantile: float = 0.90


@dataclass(frozen=True)
class QualityConfig:
    """Spec 5.5."""

    # NWIS daily-value qualifier codes. 'P' provisional, 'e' estimated,
    # 'Ice'/'Ssn'/'Eqp'/'Mnt'/'Dis'/'Rat'/'Bkw' etc.
    excluded_qualifiers: tuple[str, ...] = ("P", "e", "Ice", "Eqp", "Ssn", "Mnt", "Dis", "***")
    drop_events_with_missing: bool = True


@dataclass(frozen=True)
class EstimatorConfig:
    """Spec 6."""

    # Certification quantile of X, evaluated on the safeguard-satisfying
    # subsample (the `quantile_certified_subset` convention).
    certification_quantile: float = 0.80
    certification_quantile_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    # Safeguard: each safeguard variable below its own `safeguard_quantile`
    # (PerVariableQuantileBall == l_infty with per-variable scaling).
    safeguard_quantile: float = 0.75
    safeguard_quantile_grid: tuple[float, ...] = (0.5, 0.6, 0.75, 0.9)
    safeguard_norm: str = "l_infty"
    # How many intervening tributary gauges enter the safeguard vector, taken
    # largest-catchment first. Every extra coordinate multiplies the retained
    # fraction by roughly `safeguard_quantile`, so the certified sample dies
    # exponentially in this number; 3 is the compromise fixed in advance.
    n_safeguard_gauges: int = 3
    trim_fraction: float = 0.10
    estimator_types: tuple[str, ...] = ("ols", "median", "max", "trimmed")
    # Operating point rule (spec 6.3): take the largest certification
    # quantile whose certified sample still has at least `min_n_cert`
    # declustered events. Pre-registered at 30; lowered to 10 on the
    # coauthor's request (preregistration.md, 2026-09-01), with n_cert printed
    # next to every estimate and intervals read as indicative.
    min_n_cert: int = 10
    # Below this, a figure/table row is flagged unreliable.
    unreliable_n_cert: int = 20


@dataclass(frozen=True)
class UncertaintyConfig:
    """Spec 6.4. We implement option 3 (year-level bootstrap) only."""

    n_bootstrap: int = 400
    level: float = 0.90
    seed: int = 20260731
    # Hill estimator: fraction of upper order statistics used.
    hill_k_fraction: float = 0.10


@dataclass(frozen=True)
class HydroConfig:
    regions: tuple[Region, ...] = REGIONS
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    decluster: DeclusterConfig = field(default_factory=DeclusterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    start_date: str = "1930-01-01"
    end_date: str = "2024-12-31"
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    seed: int = 20260731

    # ------------------------------------------------------------------
    # External reference values. There are TWO, and conflating them is the
    # single easiest way to get this application wrong.
    #
    # (a) beta_routing: the STRUCTURAL effect of upstream daily discharge on
    #     downstream daily discharge, holding the intervening catchment's
    #     contribution fixed. Water is conserved, so a marginal cubic metre
    #     passing X passes Y one travel time later; the only losses are
    #     channel/floodplain storage, which attenuate the *daily mean* peak
    #     mildly. Hence beta ~ 1, with a plausibility band allowing
    #     attenuation. This is the causal target the certificate estimates.
    #
    # (b) area ratio A_Y / A_X: the value a regression of Y on X through the
    #     origin returns when runoff is spatially uniform over the basin, i.e.
    #     when the basin-wide latent driver U dominates. It is the
    #     *confounded* benchmark, NOT the causal effect. Spec 7 baseline 8
    #     writes it as (A_Y/A_X)^theta; we report it under that name but label
    #     it as the confounded reference, because that is what it is.
    # ------------------------------------------------------------------
    beta_routing_target: float = 1.0
    # Band tightened from (0.80, 1.05) on the coauthor's request, 2026-09-02.
    beta_routing_band: tuple[float, float] = (0.90, 1.10)
    area_ratio_theta: float = 1.0
    area_ratio_theta_band: tuple[float, float] = (0.7, 1.0)


CONFIG = HydroConfig()
