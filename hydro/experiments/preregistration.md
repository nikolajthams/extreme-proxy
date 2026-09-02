# Pre-registration: pair selection and operating point

Fixed before any $\hat\beta$ was computed. Spec 11.5 ("pre-register the headline
pair on hydrological grounds ... and report the full pair set").

Every rule below uses **only** columns of `station_pairs.parquet`, all of which
are derived from record length, drainage area, NLDI topology and the regulation
screen — never from an estimate.

## Design-A candidate set

A connected pair enters the Design-A candidate set iff

1. `screen_result == "pass"` (spec 4.3: >= 20 y overlap, >= 1 gauged intervening
   tributary, area ratio in [1.02, 6], both gauges past the site-level
   regulation and drainage-area screens);
2. `area_ratio` in [1.2, 2.5] — the downstream gauge must gain enough area for
   the intervening catchment (and hence the latent basin-wide driver) to matter,
   but not so much that the upstream gauge is a minor contributor;
3. `ungauged_gain_fraction <= 0.10` — the safeguard tributaries must gauge at
   least 90% of the area gained between X and Y. This is the empirical
   counterpart of the paper's "$W$ must monitor every latent source" caveat;
4. `n_tributaries >= 3` — the safeguard vector has the pre-registered
   `EstimatorConfig.n_safeguard_gauges = 3` coordinates available;
5. `overlap_years >= 40` and `distance_km <= 200`.

## Headline pair

The candidate with the largest **effective sample size**, i.e. the most
declustered in-season flood events on the pair's complete-record panel; ties
broken by the shortest river distance. No estimate enters the choice.

*Amendment, recorded when made.* The rule originally read "longest overlapping
record, ties broken by shortest distance". Running the A1 gate showed that
`overlap_years` is a poor proxy for power: it is computed on X and Y alone,
whereas the panel is built on the intersection of X, Y and the safeguard gauges'
records, so pairs with 95 overlap years produced as few as 25 declustered
events. `n_events` measures the quantity the consistency argument actually
depends on. The amendment changes only which of the qualifying pairs is called
"headline"; the full qualifying set is reported either way, and `n_events` is a
record-length quantity, not an estimate.

Rule 4 is evaluated on the *usable* safeguards (`n_safeguards_used == 3`) for
the same reason. Both amendments were made after the A1 gate had been run, so
its estimates were visible; neither uses one. The pre-amendment headline
(01559000 -> 01563500, Juniata Huntingdon -> Mapleton Depot, 285 events but only
2 usable safeguard gauges) is carried in `results.parquet` under
`experiment = "E1_prereg_v1"` so that nothing is hidden by the change.

The same diagnosis motivated `select_safeguards`: the safeguard vector is the
three largest intervening tributaries **among those gauged over at least 80% of
the pair's own good years**, rather than the three largest outright.

## Operating point

Spec 6.3: the **largest** certification quantile $q_x$ on the grid
`EstimatorConfig.certification_quantile_grid` whose certified sample still has
at least `EstimatorConfig.min_n_cert = 30` declustered events, at the
pre-registered safeguard quantile $q_w = 0.75$ over the 3 largest intervening
tributaries. If no grid point reaches 30, the pair is reported as
under-powered rather than run at a lower threshold.

## Reference value

Two reference numbers, never conflated (see `hydro/config.py`):

* $\beta_{\text{routing}} = 1$ (band [0.80, 1.05]) — the causal target. Mass
  conservation: a marginal cubic metre past X passes Y one travel time later,
  minus mild channel/floodplain attenuation of the daily mean.
* $A_Y/A_X$ — the **confounded** benchmark, i.e. what a through-the-origin
  regression returns when runoff is spatially uniform and the basin-wide latent
  driver dominates. Spec 7 baseline 8 writes it $(A_Y/A_X)^\theta$; we report it
  under that name but label it as the confounded reference, because that is
  what it is.

*Correction, 2026-08-31.* The sentence "pairs with 95 overlap years produced as
few as 25 declustered events" above overstates the case: `design_a_pairs.parquet`
shows the 95-year pairs at 138--236 events. The verified version of the
diagnosis, and the one the paper carries, is that overlap ranks pairs
differently from the panel's event count -- e.g. 48 overlap years -> 144 events
(01533400->01554000) against 95 overlap years -> 138 events (05464500->05465500)
-- because the panel is the intersection of X, Y *and* the safeguard gauges'
records. The amendment and its reasoning are otherwise unchanged.

## Amendments of 2026-09-01 (nested safeguards, coverage rule, headline)

*Amendment 2 -- nested safeguard gauges.* `station_pairs.parquet` had been
built before the maximality filter in `Network.intervening_tributaries` was
saved (parquet 13:30:02, network.py 13:30:37 on 2026-07-31), so the tributary
lists contained nested gauges on the same stream. 19 of the 21 Design-A
safeguard vectors held two such gauges (e.g. West Branch at Lewisburg *and* at
Williamsport). Consequences: one safeguard coordinate was a near-duplicate,
`ungauged_gain_fraction` read 0.0 for every pair (double-counted areas), and the
OLS-on-(X, W) coefficients on the old headline pair were collinearity artefacts
([2.17, -1.62, 2.04]). The pair table was rebuilt and `select_safeguards` /
`nearest_disconnected_safeguards` now drop nested gauges (`drop_nested`).

*Amendment 3 -- the ungauged screen.* Computed correctly, the "<= 10% ungauged"
rule admits 2 pairs (median ungauged fraction 0.35). It never excluded a pair
before, since its input was always 0.0. Replaced by: the safeguard vector that
enters the certificate (<= 3 non-nested, record-covering gauges) must drain at
least 50% of the area gained between X and Y (`MIN_SAFEGUARD_AREA_COVERAGE`),
and `MIN_TRIBUTARIES` is 1. The threshold was chosen from the pair table (26
pairs across all three basins, 15/8/3) before any estimate on the new pair set
was computed. The ungauged fraction is reported per pair.

*Amendment 4 -- headline rule.* With one-coordinate safeguards admitted, the
most-events rule picks 03351000->03353000 (White River, area ratio 1.34, one
tributary, no proxy). The rule is now: most events among candidates with three
tributary safeguards, three disconnected proxy gauges and area ratio >= 2
(`HEADLINE_MIN_AREA_RATIO`), which the headline figures need to show both
structures. Selected: 01531500->01554000, Towanda -> Sunbury.

*Operating point.* The tabulated safeguard level is now q_w = 0.5 with at least
10 certified events (`hydro/experiments/decomposition.py`), with the full grid
{0.9, 0.75, 0.5, 0.25} reported; `e1_headline` keeps the >= 30 rule for its
own tables.

*Follow-up, 2026-09-01 (coauthor request).* Tabulated safeguard level q_w = 0.25
(fallback 0.5 where fewer than 10 events are certified), `min_n_cert` 30 -> 10,
`e1_headline` grid {0.25, 0.5, 0.75, 0.9}. The headline rule gains the clause
"keeps >= 10 certified events at q_w = 0.25 in both structures" (a sample-size
property): selected 03351000->03354000, Nora -> Centerton (White River).

*Band change, 2026-09-02 (coauthor request).* The plausible routing band was
tightened from [0.80, 1.05] to [0.90, 1.10] (`beta_routing_band`); all in-band
counts in the paper were recomputed from the parquets.
