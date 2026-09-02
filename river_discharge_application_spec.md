# Spec: River-discharge real-data application for "Extreme Proxies for Causal Identification"

**Audience:** coding agent implementing the empirical section (`sections/8.tex`, Experiments) of the extreme-proxy paper.
**Status:** design spec. Implement in the existing repo at `extreme_proxy/code`, reusing and extending `certification_estimators`.
**Prime directive:** the deliverable is a *credible* empirical section, not a good-looking number. Several of the experiments below are designed so that the certificate estimator can visibly fail. Do not suppress those; they are the point.

---

## 1. Context and goal

The paper identifies the structural effect $\beta$ of a treatment $X$ on an outcome $Y$ under latent confounding $U$, by restricting to a **certifying tail event**

$$A_v^{XW} = \{X > v,\ W \in \mathcal{S}\}$$

where $\mathcal{S}$ is a bounded "safeguard" set. The logic: under **extreme propagation** (extreme parent $\Rightarrow$ extreme child), observing $W$ inside a bounded set certifies that $U$ was *not* extreme, so among observations with $X$ extreme the confounding contribution to $Y$ is asymptotically negligible and $Y/X \to \beta$.

The application must demonstrate three things, in order of importance:

1. **The certificate does real work.** The safeguard restriction $W \in \mathcal{S}$, not merely the tail restriction $X > v$, is what removes confounding. (Ablation-driven; see §8.2.)
2. **The estimate is right.** $\hat\beta$ matches an externally known physical target on pairs where such a target exists.
3. **Baselines fail where theory says they should.** Naive regression, 2SLS with a confounded instrument, and proximal/bridge methods are biased on the same data.

---

## 2. Why river discharge (keep this reasoning in the write-up)

- **Non-negative support and no cancellation.** Discharge is $\ge 0$ and tributary contributions add. The paper's support-on-$[0,\infty)$ / no-cancellation assumption is mass conservation, not a convenience assumption.
- **Extreme propagation is a routing law.** An upstream flood wave arrives downstream, delayed by travel time. Assumption is physically derivable, not assumed.
- **Non-extreme non-propagation** holds: flow routing is approximately linear, so no mechanism manufactures extremes from moderate parents (cf. `notes/my_notes.md`).
- **A known target exists.** For two gauges on the same river stem, the structural effect of upstream discharge on downstream discharge is $\approx 1$ for closely spaced pairs, and predictable from drainage-area ratio / attenuation more generally.
- **Genuinely heavy tails.** Flood peaks are regularly varying with small tail index, the regime where §"valid_bridge_note.md" shows bridge/moment methods degrade.

---

## 3. Causal designs

Notation follows the existing code (`certification_estimators/classical_graphs.py`): $X$ treatment, $Y$ outcome, $W$ safeguard/proxy, $Z$ instrument, $U$ latent.

### Design A — proxy graph `PROXY` (primary)

| Symbol | Hydrological realisation |
|---|---|
| $X$ | Daily discharge at an **upstream gauge** on the main stem |
| $Y$ | Daily discharge at a **downstream gauge** on the same stem |
| $U$ | Regional meteorological driver: synoptic/frontal storm or basin-wide snowmelt |
| $W$ | Discharge at a gauge on a **tributary entering between $X$ and $Y$** |

This realises the graph *exactly*, including the $W \to Y$ edge (the tributary's water physically reaches $Y$), which is the case standard proximal methods handle badly. Interpretation:

- $X$ large, $W$ small $\Rightarrow$ a **local** driver hit the upstream sub-catchment only (isolated convective cell, local snowmelt, reservoir release) while the rest of the basin stayed dry. That flood wave is exogenous to $U$.
- $X$ large, $W$ large $\Rightarrow$ basin-wide event, $U$ extreme, $Y$ inflated through every channel. Discard.

**Estimand:** $\beta$ = flood-peak routing/attenuation coefficient from the upstream to the downstream gauge, in m³/s per m³/s.

**Multi-variable safeguard.** Prefer $W$ = a *vector* of all gauged tributaries and side-catchments between $X$ and $Y$ (plus, if available, catchment-average precipitation outside the $X$ sub-catchment). The existing `QuantileBall(variables, quantile, norm)` already supports this; use `norm="l_infty"` as the default (all safeguard variables individually non-extreme) and report `"l1"` as a robustness check. The paper's caveat — $W$ must monitor *every* latent source capable of extreme confounding — translates directly into "gauge every tributary you can".

### Design B — confounded IV graph `INVALID_IV`

| Symbol | Realisation |
|---|---|
| $Z$ | Discharge at a **far-upstream headwater gauge**, or catchment-average precipitation over the $X$ sub-catchment |
| $X$, $Y$ | Mid-basin and downstream gauges as above |
| $U$ | Regional weather |

Classical IV fails twice: $Z \not\perp U$ (same weather system), and exclusion is violated because water from $Z$'s catchment reaches $Y$ via ungauged tributaries, floodplain storage and groundwater, not only through the $X$ gauge cross-section. Report 2SLS here as a baseline that is *expected* to be biased, and show the certificate anchored at $Z$ (`Certificate(e="Z", ...)`, as in `compare_to_gmm.py`) recovers $\beta$.

### Design C — negative controls (must be included)

1. **Disconnected-pair placebo.** $X$, $Y$ gauges in hydrologically **disconnected** catchments sharing a weather regime (e.g. adjacent basins draining opposite sides of a divide, or two independent tributaries of a common river measured *upstream* of their confluence). True causal effect is exactly 0; confounding by $U$ is strong. Expected: naive OLS gives a large positive coefficient, certificate estimator $\to 0$. **This is the single most persuasive figure in the paper.** Design the plot for it deliberately.
2. **Reverse-direction placebo.** Swap $X$ and $Y$ on a genuine upstream/downstream pair. Water does not flow uphill, so the certified estimate should collapse toward 0 (up to the travel-time alignment caveat — align lags in the *reverse* direction too, so this is not won by misalignment).
3. **Distance gradient.** Estimate $\beta$ for many pairs at increasing separation / increasing intervening ungauged area. $\hat\beta$ should decline smoothly and monotonically. A method that reproduces a coherent physical gradient across dozens of pairs is stronger evidence than a single number.

---

## 4. Data

### 4.1 Primary: raw daily (or sub-daily) discharge, many pairs

Sample size is the binding constraint: the certified subsample is $\{X > v\} \cap \{W \in \mathcal{S}\}$ *after declustering*, so it shrinks fast. Budget for this from the start.

Recommended, in priority order:

1. **CAMELS-GB** — 671 GB catchments, daily streamflow **and** catchment-average precipitation, ~1970–2015. Best single source: long records, precipitation available as $Z$, many candidate pairs, permissive licence. <https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9> (v2: <https://essd.copernicus.org/articles/18/4345/2026/>).
2. **USGS NWIS** via the `dataretrieval` Python package (<https://github.com/DOI-USGS/dataretrieval-python>) — programmatic access, and **15-minute / hourly** instantaneous values, which resolve travel time properly. Use this if daily aggregation turns out to blur flood-peak alignment (§5.1).
3. **GRDC** — global, for a second-basin robustness check.

### 4.2 Secondary: Upper Danube (recognisability anchor)

The `danube` dataset in the R package `graphicalExtremes` (<https://search.r-project.org/CRAN/refmans/graphicalExtremes/html/danube.html>) is *the* canonical dataset in the graphical-extremes literature (Asadi, Davison & Engelke 2015, *AoAS* 9(4):2023–2050; Bavarian Environmental Agency). It ships `info` (river name, gauge lat/lon, catchment centroid, **catchment area**, mean altitude, mean slope) and `flow_edges` (pairs of stations directly connected by river) — i.e. the network topology and areas you need for target values, for free.

**Critical caveat:** the packaged data is summer-months-only (1960–2010) and **already declustered to ~7–10 observations per year** (~350–500 rows). That is far too small to threshold twice. Therefore:

- Use the Danube **`info` / `flow_edges` metadata** for network topology and catchment areas regardless.
- For the actual estimation, obtain **raw daily** Danube discharge from the Bavarian Environmental Agency (Gewässerkundlicher Dienst Bayern) if licensing permits; otherwise run the headline analysis on CAMELS-GB/USGS and present Danube only as a small-sample illustration with the sample-size limitation stated explicitly.
- Extract the R data to Parquet once via a small R or `pyreadr` script; do **not** add an R runtime dependency to the main pipeline.

### 4.3 Station-pair selection (implement as a deterministic, auditable step)

Build a candidate-pair table by rule, then filter:

- Same river stem, $X$ strictly upstream of $Y$ (from network topology).
- Overlapping record $\ge$ 20 years, $\ge$ 90% daily coverage in the overlap, no more than 5% missing in any single year.
- At least one gauged tributary joining between $X$ and $Y$ (required for Design A).
- **Regulation screen:** exclude pairs with a major flood-control reservoir between them. Use catchment attributes where available (CAMELS-GB has reservoir/abstraction indices); otherwise document the screen used. Regulation is the principal way extreme propagation genuinely breaks.
- Record for every pair: catchment areas $A_X, A_Y, A_W$, distance along river, intervening ungauged area fraction.

Persist the pair table as a Parquet/CSV artifact with a provenance column. Every downstream figure should be reproducible from it.

---

## 5. Preprocessing

Each step below is a documented, separately testable transform. Expose every tuning constant in a config object; do not bury constants in the analysis script.

### 5.1 Travel-time alignment (highest-risk step)

$X$ and $Y$ observe the *same flood wave at different times*. On daily data in a large basin the peak can fall on different calendar days, which breaks the extreme-propagation link and will silently bias everything.

Procedure:

1. Estimate lag $\ell_{XY}$ per pair by maximising cross-correlation of log-discharge, **restricted to the upper tail** (e.g. correlate only over windows containing the top 5% of $X$) — bulk-flow lag differs from flood-wave celerity, and you want the latter.
2. Sanity-check $\ell$ against river distance / plausible wave celerity (order 1–3 m/s); flag pairs where the estimated lag is implausible and drop them.
3. Apply the same procedure independently for the reverse-direction placebo, so that placebo does not succeed merely because of misalignment.
4. **Robustness:** rather than a point lag, define $Y$ as the maximum discharge in a window $[t+\ell-1, t+\ell+1]$ days. Report results for both point-lag and window definitions.

### 5.2 Event declustering

A single flood spans several days; raw daily exceedances are massively serially dependent, and the "$\sum_i \mathbb{1}\{X_i > v_n, W_i \in \mathcal{S}\} \xrightarrow{P} \infty$" condition in the paper's consistency result concerns *effective* sample size.

- Apply runs-declustering on $X$: an exceedance cluster ends after $r$ consecutive days below threshold; take one observation per cluster (the one at the cluster's $X$-peak, with $Y$, $W$, $Z$ taken at the correspondingly aligned times).
- Default $r = 7$ days; sweep $r \in \{3, 5, 7, 14\}$ as a robustness axis.
- **Report the effective sample size $n_{\text{cert}}$ (number of declustered certified observations) on every figure and in every table.** A $\hat\beta$ based on 6 events must be visibly labelled as such.

### 5.3 Seasonality and regime

Snowmelt floods and summer convective floods propagate differently and mixing them violates the single-mechanism premise.

- Default: restrict to one hydrological season per basin (for the Danube, summer — which also matches the canonical dataset's convention).
- Report a season-stratified robustness table.
- Do **not** deseasonalise by subtracting a seasonal mean: that destroys the non-negativity that makes the no-cancellation assumption defensible.

### 5.4 Units and scale

- Work in **m³/s on the original scale**. $\beta$ is then a physical routing coefficient directly comparable to the drainage-area ratio, which is what makes the "known ground truth" claim work.
- Do **not** transform to standard Pareto/Fréchet margins for the headline estimates. Marginal standardisation is standard in multivariate EVT but destroys the interpretation of $\beta$. If a standardised variant is run for comparison with the extremal literature, present it separately and label it clearly.
- Never log-transform $X$ or $Y$ before estimating $\beta$.
- Area-normalised (mm/day) versions may be computed as a *secondary* analysis for cross-pair comparability in the distance-gradient plot.

### 5.5 Missing data and quality flags

- Drop event windows containing missing values in any of $X, Y, W, Z$ (do not interpolate over flood peaks).
- Honour provider quality flags: estimated/ice-affected/provisional values must be excluded at high flows.
- **Rating-curve caveat.** Discharge at extreme stage is extrapolated from a rating curve, so the most extreme observations are the least accurately measured — precisely the ones the method leans on. Where the provider publishes a rating-curve validity range or measurement-uncertainty band, record whether each certified observation lies inside it, and report a sensitivity analysis restricted to in-range observations. This must be discussed honestly in the paper; a referee will raise it.

---

## 6. The estimator

### 6.1 Reuse the existing abstractions

Extend, don't rewrite: `certification_estimators/certificate.py` (`CompactSet`, `Ball`, `QuantileBall`, `Certificate`) and `certificate_estimator.py` (`CertificateEstimator`, types `"ols" | "median" | "max"`) already implement the core. They operate on `polars` frames; keep that.

Required changes / additions:

- **Fix the `demean` bug** flagged in `experiments/valid_bridge_note.md`: `fit_or_null` forces `demean=True`, which inverts the `QuantileBall` safeguard in infinite-mean regimes. Real discharge data is in exactly that regime. Demeaning must be off by default for this application, and the interaction between demeaning and the safeguard should be covered by a unit test.
- **No intercept.** The estimand is a ratio through the origin ($Y/X \to \beta$). Ensure the OLS variant is through the origin, as `fit_on_certified_subset` currently is. Do not add an intercept "to be safe" — it changes the estimand.
- **Threshold as a level, not only a quantile.** Add a path that takes an absolute $v$ (m³/s) in addition to `certification_quantile`, so thresholds can be reported in physical units and compared with published flood-quantile estimates (e.g. the 10-year return level).
- **Trimmed-ratio variant.** Add an estimator type that takes a trimmed mean of $Y_i/X_i$ over the certified set, as a robustness option between `"median"` (robust, less efficient) and `"ols"`.
- **Report all of `ols`, `median`, `max`.** Their agreement on real data is itself evidence; `"max"` is the natural pure-extreme-ratio estimator and its divergence from `"ols"` signals threshold problems.

### 6.2 Safeguard construction for the application

- $\mathcal{S}$ = `QuantileBall(variables=[all intervening tributary gauges (+ precipitation if used)], quantile=q_w, norm="l_infty")`.
- Default $q_w = 0.9$ interpreted as "each safeguard variable below its own 90th percentile"; note the current `QuantileBall` computes the quantile of the *aggregated norm*, which for `l_infty` is not the same as a per-variable quantile — decide explicitly which semantics you want, document it, and if per-variable is wanted, add a `PerVariableQuantileBall`.
- **Condition the safeguard quantile on the safeguard-only subsample**, as `quantile_certified_subset` already does for $\tau$; keep that convention and state it.

### 6.3 Threshold selection and diagnostics

There is no free lunch here; the paper should present the sensitivity rather than pretend a threshold was known.

- Produce a **threshold-stability plot** for every headline pair: $\hat\beta$ (with uncertainty band) as a function of the treatment threshold $v$ (x-axis in physical units and/or exceedance probability), one panel per safeguard level $q_w$. This is the EVT idiom (cf. Hill / mean-excess plots) and referees will expect it.
- Produce a **2-D heatmap** of $\hat\beta$ over $(v, q_w)$ for the primary pair, with the region where $n_{\text{cert}} < 20$ masked out and visibly marked as unreliable.
- Report a default operating point chosen by a *pre-registered* rule (e.g. highest $v$ such that $n_{\text{cert}} \ge 30$), fixed before looking at $\hat\beta$, and state the rule in the paper.

### 6.4 Uncertainty quantification

Standard bootstrap is unreliable under infinite variance and serial dependence. Use, in order of preference:

1. **Subsampling** (Politis–Romano–Wolf) with block size $b = n^{2/3}$ over declustered events — valid under heavy tails where the bootstrap is not.
2. **Stationary block bootstrap** over the declustered event series (mean block length ~2× typical flood duration) as a cross-check.
3. **Year-level bootstrap** (resample whole hydrological years) — crude but transparently respects within-year dependence.

Report intervals from at least two of these and note disagreement rather than picking the narrowest. Also estimate and report the **tail index** $\alpha$ of $X$ (Hill estimator with a stability plot) per basin, since the `valid_bridge_note.md` analysis shows baseline behaviour depends sharply on whether $\alpha \lessgtr 2$.

---

## 7. Baseline methods

Implement each against the identical preprocessed, aligned, declustered dataset. Any preprocessing advantage given to the certificate estimator invalidates the comparison.

| # | Baseline | Implementation | Expected behaviour |
|---|---|---|---|
| 1 | **Naive OLS** $Y \sim X$ (no intercept) | trivial | Upward biased: regional storms load both |
| 2 | **OLS adjusting for the proxy** $Y \sim X + W$ | trivial | Still biased; $W$ is a noisy proxy, and adjusting on it does not block $U$ |
| 3 | **2SLS** with $Z$ (headwater gauge or precipitation) | `statsmodels` / hand-rolled | Biased: $Z$ is confounded and exclusion fails (Design B) |
| 4 | **Linear proximal / bridge** | existing `LinearProximal("X","Y",Z_vars,W_vars)` from `graphical_toolbox` | Biased *and* high-variance at $\alpha \le 2$, per `valid_bridge_note.md` |
| 5 | **Valid-bridge proximal** (augmented with squared proxies) | as in `nonlinear_compare_to_gmm.py` | Correctly specified yet still inconsistent at heavy tails — a key selling point; verify it replicates on real data |
| 6 | **Causal tail coefficient** $\Gamma$ (Gnecco, Meinshausen, Peters, Engelke 2021) | implement or port | Estimates causal *direction/ordering*, **not** $\beta$. Use it for the direction and placebo experiments only; do not tabulate it against $\beta$ as if it were an effect estimate |
| 7 | **Extremal quantile regression** (Chernozhukov) | `statsmodels` quantile regression at $\tau \to 1$ | Comparison point from the extreme-value regression literature |
| 8 | **Hydrological benchmark**: drainage-area ratio $\left(A_Y/A_X\right)^{\theta}$ | from `info` metadata | Not an estimator — the external *target* to compare against |

### 7.1 Required ablations (as important as the baselines)

- **A1 — Tail-only, no safeguard.** $\hat\beta$ from $\{X > v\}$ alone, ignoring $W$. If this equals the certified estimate, the safeguard is doing nothing and the paper's central claim is unsupported on this data. **Run this first, before building anything else.** If the certified and uncertified tail estimates coincide on the primary pair, stop and report that to the human before proceeding — the design needs a pair with genuinely strong extremal confounding.
- **A2 — Permuted safeguard.** Replace $W$ by a time-permuted copy (destroying its dependence on $U$ while preserving its marginal). The certified estimate should degrade to the A1 value.
- **A3 — Irrelevant safeguard.** Use a gauge from a climatically unrelated basin as $W$. Should behave like A2.
- **A4 — Safeguard sufficiency.** Progressively add intervening tributaries to the safeguard vector and show $\hat\beta$ stabilising. This is the empirical counterpart of the paper's "$W$ must cover all latent sources" caveat: instability under A4 is *evidence about the assumption*, and should be reported either way.

---

## 8. Experiment list and success criteria

State the criterion before running each; record pass/fail in the results artifact.

**E1 — Headline pair (Design A).** One well-chosen upstream/downstream pair with a gauged intervening tributary and long record.
*Criterion:* $\hat\beta$ within a defensible band of the physical target (e.g. within 15% of the area-ratio-adjusted value), stable across $v$ in the stability plot, with baselines 1–5 outside that band.

**E2 — Disconnected placebo (Design C1).** ~10–20 disconnected pairs.
*Criterion:* certified $\hat\beta$ concentrated near 0; naive OLS clearly positive. Report as a paired scatter/dumbbell plot, one row per pair.

**E3 — Reverse direction (Design C2).**
*Criterion:* certified $\hat\beta \approx 0$ in the anti-causal direction on the same pairs where E1 succeeds.

**E4 — Distance gradient (Design C3).** All qualifying pairs.
*Criterion:* monotone decline of $\hat\beta$ with intervening ungauged area; report the fitted trend and the scatter, not just the trend.

**E5 — Confounded-IV design (Design B).**
*Criterion:* certificate anchored at $Z$ recovers the E1 value; 2SLS does not.

**E6 — Semi-synthetic calibration.** Take the *real* $(U\text{-driven}, W, X)$ dependence from data, replace $Y$ by a synthetic $Y = \beta_0 X + \text{(real confounding channel)} + \varepsilon$ with known $\beta_0$.
*Criterion:* coverage of the subsampling intervals near nominal; bias small. This is the only place with true ground truth and it bridges §7's simulations to the application — do not skip it.

**E7 — Robustness grid.** Sweep: declustering run length $r$, season, lag definition (point vs window), safeguard norm (`l_infty` / `l1`), estimator type (`ols`/`median`/`max`/trimmed), rating-curve-in-range restriction.
*Criterion:* headline conclusion unchanged; report the full grid in an appendix table.

---

## 9. Outputs and conventions

Match the existing repo's conventions (`experiments/compare_to_gmm.py`):

- `polars` for all data handling; long-format results frame with an explicit `method` column via `unpivot`.
- Figures with `plotly.express`, exported **both** as `.html` (interactive, for inspection) and `.tex` via `tikzplotly` (for the paper). Output under `experiments/<experiment_name>/`.
- Aggregate with median bias / MAD / quantile bands rather than mean and SD — means are meaningless at $\alpha < 2$.
- Every figure caption must carry $n_{\text{cert}}$ and the thresholds used.

**Required figures:**

1. Threshold-stability plot for the headline pair (F1).
2. Placebo dumbbell plot: naive vs certified across disconnected pairs (F2) — likely the paper's main empirical figure.
3. Distance-gradient scatter with fitted trend (F3).
4. Method-comparison table/forest plot: all baselines + ablations on the headline pair, with intervals (F4).
5. $(v, q_w)$ heatmap with the low-$n$ region masked (F5, appendix).
6. Map/schematic of the chosen gauges and the river network, marking $X$, $Y$, $W$, $Z$ (F6) — readers need to see the geography to believe the DAG.

**Required artifacts:** the station-pair table, the per-event certified subsample for the headline pair (so results are auditable), and a `results.parquet` with one row per (experiment, pair, method, hyperparameter setting).

---

## 10. Repository layout

```
code/
  certification_estimators/        # existing; extend per §6.1
  hydro/                           # new
    data/
      sources.py                   # CAMELS-GB / USGS / GRDC loaders -> parquet
      danube_metadata.py           # info + flow_edges extraction
      network.py                   # upstream/downstream topology, areas, distances
    preprocess/
      alignment.py                 # §5.1 travel-time lag estimation
      decluster.py                 # §5.2 runs declustering
      screening.py                 # §4.3 pair selection + regulation screen
    baselines/                     # §7 wrappers, common interface
    experiments/                   # E1..E7, one module each
    config.py                      # all constants, dataclass-based
  experiments/                     # existing simulation scripts, unchanged
  tests/
```

Engineering requirements:
- Cache raw downloads to disk; the pipeline must be re-runnable offline.
- Fix the RNG seed for every stochastic step (subsampling, permutation ablations).
- Unit tests for: lag estimation on synthetic shifted series, declustering counts, safeguard set semantics, no-intercept OLS, the `demean` fix.
- One entry point per experiment, `uv run python -m hydro.experiments.e1_headline`.

---

## 11. Failure modes to design around, not discover late

1. **Certified subsample too small.** The most likely outcome. Mitigate by choosing long records and sub-daily data early, not by lowering thresholds until numbers appear.
2. **Regulation.** Reservoirs break extreme propagation. Screen at selection time (§4.3), and where a pair fails, report it as a *predicted* failure of the assumption — that is a strength.
3. **Overbank / floodplain flow.** Above bankfull, propagation becomes strongly nonlinear and attenuating. The linear-$\beta$ estimand may not exist for the very largest events; check whether $\hat\beta$ drifts downward at the highest thresholds in F1 and discuss.
4. **Lag misalignment** silently destroying the result — see §5.1, and note that it can also make placebos *look* successful.
5. **Multiple comparisons.** With dozens of pairs and a hyperparameter grid it is easy to find a pair that works. Pre-register the headline pair on hydrological grounds (record length, gauged tributary, no regulation) *before* estimating anything, and report the full pair set.
6. **Circular validation.** The drainage-area-ratio target is itself a model. Present it as a plausibility band, not as truth, and lean more on the placebos (E2/E3), which need no target at all.

---

## 12. Suggested order of work

1. Data loaders + network topology + pair table (§4).
2. **Ablation A1 on 2–3 candidate pairs** — cheapest possible check that the safeguard changes anything. Report back before building further.
3. Preprocessing pipeline (§5) with tests.
4. Estimator extensions + `demean` fix (§6.1).
5. E1 + F1, then E2 + F2.
6. Baselines (§7), E5, E4.
7. E6 semi-synthetic, E7 robustness grid, remaining figures.

---

## 13. Open questions for the authors (do not guess — ask)

- Which basin is the headline? (Danube = referee recognisability but small $n$; CAMELS-GB / USGS = statistical power.)
- Is the paper's estimand the routing coefficient on the original m³/s scale (recommended here), or something scale-free?
- Should the IV design (B) be a full experiment or a short remark?
- Is a nonlinear certificate variant in scope, given that overbank flow is genuinely nonlinear at the top of the range?
