# River-discharge application: progress log

Scope agreed with the user: **Core + placebos**. In: A1 gate, preprocessing
verification, estimator extension decisions, E1+F1, E2+F2, E3, baselines 1-5 +
drainage-area-ratio reference, ablations A2/A3, `results.parquet`, pair-table
artifact, `tests/`. Out: E4, E6, E7, causal tail coefficient, extremal quantile
regression, F5 heatmap, F6 map.

Entry points (all offline from `hydro/cache/`):

    uv run python -m hydro.experiments.a1_gate
    uv run python -m hydro.experiments.e1_headline
    uv run python -m hydro.experiments.e2_placebo
    uv run python -m hydro.experiments.results
    uv run pytest tests

---

## Done and verified

1. **Orientation.** Spec, `hydro/`, `certification_estimators/`, repo style.
2. **`demean` fix verified.** `CertificateEstimator.fit`/`.fit_or_null` and
   `TailCertificateEstimator.fit` all default `demean=False`. No change needed;
   covered by `tests/test_hydro.py`.
3. **`z` column understood.** Null for 164/232 passing pairs, whenever X is on a
   tributary or is the topmost main-stem gauge. Expected.
4. **Preprocessing chain verified end-to-end** (headline pair 01540500 ->
   01570500): 17592 in-season complete days, 236 declustered events, lag 0 d
   (tail cross-correlation 0.932, implied celerity inside the physical band),
   695 day-values dropped on quality flags, declustering threshold 671.1 m3/s,
   no nulls in X, Y, W0-W2, Z. Lag estimation, declustering and safeguard
   semantics also covered by unit tests on synthetic data.
5. **A1 gate: PASS.** `a1_gate.parquet`, 21 pairs x 6 certification quantiles.
   At an identical absolute threshold v the certified estimate is below the
   tail-only estimate in 17 of 17 pairs with n_cert >= 20 (median delta -0.20 to
   -0.27 depending on q_x). Tail-only is indistinguishable from naive OLS
   (headline: 1.895 vs 1.876), so on this data **the safeguard, not the tail
   restriction, is what moves the estimate**.
6. **E1 + F1** (`e1_*`), **E2 + F2 + E3** (`e2_*`, `e3_*`), baselines 1-5,
   ablations A1/A2/A3, `results.parquet`, tests.

## Decisions made (spec left these open)

* **`QuantileBall` l_infty semantics.** Kept as-is (quantile of the aggregated
  max), and used `PerVariableQuantileBall` for the application: gauges differ by
  an order of magnitude in drainage area, so a single shared radius is set by
  the largest gauge and leaves the small ones unconstrained. Per-variable is the
  semantics the paper states. `QuantileBall` remains available through
  `make_safeguard(kind="l_infty"|"l1")`.
* **Absolute-threshold path.** `TailCertificateEstimator(level=...)` takes v in
  m3/s; exactly one of `level` / `certification_quantile` may be given. Every
  comparison that must hold the tail fixed (A1, A2, A3) uses `level`, because
  the quantile path conditions on the safeguard subsample and so moves v when
  the safeguard changes.
* **`"trimmed"` estimator.** Symmetric trimmed mean of Y/X over the certified
  set at `trim_fraction = 0.10`, i.e. between `median` and `ols`. Reported
  alongside `ols`, `median`, `max` everywhere.
* **Safeguard selection.** Three largest intervening tributaries *among gauges
  covering >= 80% of the pair's good years*. Taking the three largest outright
  collapsed the panel (the panel is the intersection of all records) from 82
  usable years to 13 on several pairs. Record length, not an estimate.
* **Headline pair rule amended** to "most declustered events" — see
  `hydro/experiments/preregistration.md`, which records the amendment, the
  reason, and the fact that it was made after A1 had been run. The
  pre-amendment headline is carried in `results.parquet` as `E1_prereg_v1`.
* **E2 safeguard.** Primary rule = the three admissible gauges *nearest Y*
  (structural analogue of Design A's intervening tributary); the area-ranked
  rule is reported as `safeguard_rule = "largest"`. Both are in the parquet.
* **E3 alignment.** The lag search is non-negative by construction, so the
  reverse panel is built with `force_lag = -lag_forward`; the placebo cannot be
  won by leaving the anti-causal direction misaligned.
* **Uncertainty.** Year-level bootstrap only (spec 6.4 option 3), 400 draws,
  90%. Subsampling and the block bootstrap were not implemented; the intervals
  are indicative, not calibrated under infinite variance, and `estimation.py`
  says so.

## Results (every number read back from the written parquet)

### A1 gate — PASS
`results.parquet`, `experiment = "E1_all_pairs"`, q_x = 0.7, OLS, 17 pairs with
n_cert >= 20: the certified estimate is below the tail-only estimate at the same
absolute v in **17 of 17**; median certificate 1.411, median A1 1.713, median
naive OLS 1.710, median delta -0.223. Tail-only tracks naive OLS to ~0.003. The
one pair where certificate > A1 (03343000 -> 03377500, +0.008) has n_cert = 14.

### F1 — threshold stability (`e1_stability.parquet`)
beta is flat in v within each safeguard level and monotone in q_w:

| q_w | beta (ols) across v | n_cert |
|-----|---------------------|--------|
| 0.2 | 0.989 | 20 |
| 0.3 | 1.090 - 1.108 | 22-31 |
| 0.5 | 1.180 - 1.230 | 26-64 |
| 0.75 | 1.418 - 1.489 | 31-110 |
| 0.9 | 1.521 - 1.619 | 20-136 |

That is the theory's prediction (S must shrink) traded against n_cert.

### E2 — disconnected placebo (20 pairs): does NOT reach 0
Nearest-Y safeguard, q_w 0.75: median certified 0.502 (q10 0.250, q90 0.794),
median naive 0.664, median n_cert 34.5. At q_w 0.5 (9 pairs reach n_cert>=30):
median 0.447. Area-ranked safeguard, q_w 0.75: median 0.570. The arithmetic
floor under the null, median(daily Y)/mean(certified X), has median 0.042, so
the certified estimate sits ~12x above the floor. Modest reduction, not collapse.

### E3 — reverse direction (20 pairs): FAILS
Reverse lag forced to -lag_forward. q_w 0.75: median certified 0.534 vs median
naive 0.450 — the certificate is *higher* than naive, i.e. further from 0.
Structural reason: the safeguard is the set of other parents of the new
treatment, so "X (downstream) extreme, tributaries bounded" selects exactly the
events where the upstream gauge was large. The certificate returns the reverse
routing ratio, not 0. It does not by itself determine causal direction here.

## Headline numbers (all read back from the written parquet)

Pair 01540500 -> 01570500, Susquehanna at Danville -> at Harrisburg.
236 events, A_Y/A_X = 2.148, Hill alpha 2.14 (daily X) / 2.73 (event peaks).

| q_w | q_x | v (m3/s) | n_cert | beta (ols) | 90% CI |
|-----|-----|----------|--------|------------|--------|
| 0.5 | 0.6 | 979.8 | 36 | 1.198 | [1.073, 1.334] |
| 0.75 | 0.8 | 1430.0 | 31 | 1.418 | [1.274, 1.513] |
| 0.9 | 0.8 | 1817.9 | 39 | 1.585 | [1.481, 1.669] |

Estimator types at q_w 0.75, n_cert 31: ols 1.418, median 1.424, trimmed 1.425,
max 1.556.

Ablations at the same v: A1 1.881 / 1.895 / 1.902, A2 (50 permutations) 1.893 /
1.904 / 1.898, A3 (Iowa safeguard) 1.858 / 1.824 / 1.872 — all three land on the
naive value, so both ablations behave as the design requires.

Baselines on the same 236 events: naive OLS 1.876 [1.792, 1.952] · OLS adjusting
for W **1.001** [0.838, 1.118] · 2SLS with Z 1.865 [1.780, 1.947] · proximal
linear bridge 2.848 [-5.28, 9.38] · proximal valid bridge 2.766 [-3.67, 7.11] ·
drainage-area ratio 2.148 (confounded reference) · routing target 1.0.

**Caveat the write-up must carry:** baseline 2 (OLS on X and W) hits the routing
target almost exactly. In this application Y is close to a water-balance sum of
X and the intervening tributaries, so a linear adjustment on W is unusually
strong here — the same mass conservation that makes the no-cancellation
assumption defensible. It is not a noisy proxy for U in the sense the proximal
literature assumes.

Pre-amendment headline 01559000 -> 01563500 (`E1_prereg_v1`, 2 usable safeguard
gauges, 285 events): certificate 1.378 / 1.515 / 1.723 / 1.994 at q_w
0.3 / 0.5 / 0.75 / 0.9, A1 1.906 / 1.905 / 1.906 / 1.900, naive 1.907. The
safeguard still works, but never reaches 1 — consistent with only 2 of the 5
intervening tributaries being gauged over the record.

## Open / not done

Nothing outstanding in scope. Out-of-scope items (E4, E6, E7, causal tail
coefficient, extremal quantile regression, F5, F6) were not started.

## 2026-08-31: presentation pass (paper figures + Section rewrite)

The Application section was rewritten for clarity; the analysis itself is
unchanged (no estimate recomputed, every number read back from the parquets).

* **`hydro/experiments/paper_figures.py` (new).** Builds the four publication
  figures from the persisted parquets only, in the `experiments/figure_style.py`
  conventions (role colours, direct-to-PDF, no tikzplotly):
  `app_all_pairs` (new figure: naive vs certified per Design-A pair at the
  pre-registered operating point, drainage-area ticks, routing band),
  `app_threshold_stability`, `app_method_comparison`, `app_placebos`
  (E2 + E3 combined, arithmetic-null-floor ticks). Staged to
  `../notes/Figures/application/`; the old `e1_*`/`e2_*`/`e3_*` PDFs there are
  superseded.
* **Correction.** The paper previously reported "20 usable pairs at the
  pre-registered operating point"; the pre-registered rule (>= 30 certified
  events) yields **18** pairs (18/18 closer to target, median 1.398 vs 1.667).
  The 20-pair variant corresponds to a >= 20-event rule. The rewrite reports 18
  powered pairs plus the 3 under-powered ones (shown open in the figure, at the
  largest threshold retaining >= 10 events); all 21 move toward the target.
* **Now discussed in the paper:** OLS of Y on (X, W) attains the routing target
  (1.001) on the headline pair — the mass-balance caveat PROGRESS flagged.
* `figures_to_pdf.py` clobber fixed (skips figure_style-owned folders and
  `app_*` stems). `preregistration.md` stale "25 events" claim corrected in an
  appended note.

## 2026-09-01 rework
Pair table rebuilt (nested safeguards removed, coverage rule); 26 Design-A pairs; headline
01531500->01554000. New: decomposition.parquet / decomposition_pairs.parquet (two safeguard
structures per pair, bootstrap at q_W = 0.5), pair_table.py -> appendix table. All of a1_gate,
e1_headline, e2_placebo, results, paper_figures re-run on the new pair set. See HANDOFF.md addendum.
