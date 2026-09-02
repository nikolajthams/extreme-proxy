# Handoff: building simulation + application experiments for the extreme-proxy paper

Written 2026-08-31 for whoever picks this up next. It documents the *procedure*
I followed, not just the results, because the results are already in the paper
and the parquets — what is easy to lose is the discipline that produced them.

Read this together with:

* `docs/river_discharge_application_spec.md` — the application spec the user wrote.
* `experiments/river_discharge/PROGRESS.md` — the river agent's own state log.
* `hydro/experiments/preregistration.md` — the pair-selection rules and their amendment.

---

## 1. Geography: there are two repositories, and one of them moves

| Path | What it is |
|---|---|
| `code/` (here) | Python. Simulations, `hydro/`, tests, figure PDFs. Git repo. |
| `../notes/` | The LaTeX paper. **A separate git repo, synced from Overleaf.** |
| `code/notes/` | Not the paper. Do not confuse it with `../notes/`. |

**The paper's section files get renumbered by Overleaf syncs.** On 2026-08-18 a
sync split `3+4.tex`, added a new `6.tex`, and moved Experiments from `8.tex` to
`6.tex` and Application from `9.tex` to `7.tex`. Nothing was lost, but every
path written by number went stale.

> **Never address a section by filename.** Resolve it each session:
> ```
> grep -n "input" ../notes/Main.tex
> for f in ../notes/sections/*.tex; do echo "$f: $(grep -m1 '\\section{' $f)"; done
> ```
> Figures under `../notes/Figures/` survive syncs untouched.

As of writing: `sections/6.tex` = Simulations, `sections/7.tex` = Application.

---

## 2. Environment gotchas (all cost me time)

* **`python` is not on PATH** (macOS/zsh). Everything is `uv run python ...`.
* **`uv run --directory X python` resets the working directory**, which breaks
  every relative path in the scripts. Use absolute paths, or `cd` first.
* **`pdftk` is not installed.** Use `pdftoppm` to rasterise PDF pages for visual
  inspection.
* **Dropbox writes lazily.** `pdfinfo Main.pdf` once returned syntax errors and
  exit 12 purely because Dropbox was mid-write. Check `Main.log` for
  `Output written on Main.pdf (N pages)` before believing a PDF is broken.

---

## 3. Orchestration procedure

The user asked for subagents/workflows, with Sonnet acceptable for the easier
experiments. What worked:

1. **Simulation experiments → one agent each, in parallel.** They are
   independent and each is a single self-contained script. Sonnet was adequate.
2. **The river application → one Opus agent, long-running.** It is a single
   dependent chain (fetch → screen → pair → panel → estimate → placebo) with
   real judgment calls at every step; splitting it would have produced
   inconsistent conventions.
3. **Agents die.** The river agent was lost to session limits and could **not**
   be resumed — `ListAgents` showed it gone. What made relaunch cheap was that
   it had been maintaining `experiments/river_discharge/PROGRESS.md` on disk.
   I relaunched a fresh agent with a brief stating: what already exists on
   disk, what remains, the requirement to keep `PROGRESS.md` current, and the
   read-back rule in §4.

> **Give every long-running agent a `PROGRESS.md` and require it to update the
> file as it goes.** Treat the agent as disposable and the file as the state.

---

## 4. The verification rule (the most important section)

**Do not write a number into the paper that you have not personally read back
from the persisted artifact.** Not from the agent's report, not from its stdout,
not from your own earlier message.

This is not paranoia. Two of the river agent's reported claims did not survive
the check:

* It reported *"pairs with 95 overlap years produced as few as 25 declustered
  events"* as the motivation for amending the pre-registration rule.
  `design_a_pairs.parquet` shows the 95-year pairs at 226/236/138/154/149
  events. Replaced with a verified comparison (48 years → 144 events vs
  95 years → 138 events). **Note this stale claim is still in
  `hydro/experiments/preregistration.md`; the paper carries the corrected
  version.**
* It described the pre-amendment Juniata pair as having *"only 2 of the 5
  intervening tributaries gauged."* In fact `n_tributaries = 5` and
  `ungauged_gain_fraction = 0.0` — all intervening area is gauged; only 2 gauges
  have records long enough to enter the safeguard. Its overlap is 83 years, not
  95. Rewritten, and the downstream inference softened accordingly.

I also caught three of my own errors this way: an overcount of usable pairs
(22 pairs → 21 with a panel → 20 with a certified estimate, not "22, 20 of
which"), a direction inconsistency between body text and the reverse-placebo
caption, and a silently truncated axis (§6).

Supporting habits:

* **Re-run before writing.** After editing `e1_headline.py`, `e2_placebo.py`
  and `results.py` I reran all three; figures reproduced identically. That is
  simultaneously a check on the edit and a reproducibility test of the pipeline.
* **Establish a compile baseline before touching the paper.** Measure it with
  a command you can repeat, not by eye:
  ```
  grep -c "Reference .* undefined" ../notes/Main.log
  grep -o "Output written on Main.pdf ([0-9]* pages" ../notes/Main.log | tail -1
  ```
  Before my edits: exit 0, **10** undefined references, 42 pages. After: exit 0,
  the same 10, 47 pages. Without the baseline, "10 warnings" is uninterpretable.
  All 10 are pre-existing forward references into the appendix
  (`ass:relaxed-propagation`, `lem:as-implies-prob`, `eq:tail-race`, …) and are
  not mine. **The page count is not a stable baseline** — Overleaf syncs change
  it; at the time of writing this file the log reads 46. The undefined-reference
  count is the number to watch.
* **Look at the rendered pages.** I rasterised pages 25–29 of `Main.pdf` and
  read them, rather than assuming the figures were legible.

---

## 5. Coding style

The user asked for "the same style and condensed form of coding as in the
existing experiments." Concretely, as practised in `experiments/*.py` and
`hydro/`:

* `from __future__ import annotations`; type hints on public functions.
* A module docstring that states what the experiment shows, **why**, and the
  exact run command(s) including `--smoke` / `--plot-only` flags.
* polars, not pandas. plotly express or graph_objects. Module-level `CONFIG`
  / constants in caps rather than argument-threading.
* Short functions, few classes, no framework. One `main()`, one
  `if __name__ == "__main__"`.
* **Comments explain the non-obvious decision, not the code.** Look at the
  comment above `beta_null_floor` in `e2_placebo.py`, or the `dumbbell`
  docstring explaining why the x axis is clipped. Every clip, every cap, every
  arbitrary constant carries a sentence saying why.
* Results go to `.parquet`, always. Figures to `.html` + `.tex` (tikzplotly)
  + `.pdf`.

---

## 6. Figure pipeline, and a landmine

Two different paths exist, and they conflict:

**Path A — hand-styled (the simulation figures).**
`experiments/figure_style.py` defines the shared palette (one colour per
*role*: certified / classical / disqualified / repaired — validated all-pairs
for colour-vision deficiency), dash per tail index, one horizontal legend.
`figure_style.save()` writes `.html` and `.pdf` directly at a deliberate size
(675×525, 675×315 pts).

**Path B — bulk conversion (`experiments/figures_to_pdf.py`).**
tikzplotly emits `table[y=Certified, 1.5]` for plotly traces whose names contain
a comma, which is invalid pgfplots and will not compile. So this script re-reads
the JSON embedded in each `.html` and re-renders to PDF at 760×520.

> ### ⚠️ Landmine
> `figure_style.py`'s docstring claims *"`figures_to_pdf.py` skips these
> folders."* **It does not.** `figures_to_pdf.py` globs `experiments/*/*.html`
> with no exclusion, and both `iv_and_proxy.html` and
> `distributional_failure.html` match its regex. Running it as-is will
> **overwrite the hand-styled simulation PDFs** with squashed 760×520
> re-renders (the 2×2 subplot figure especially).
>
> Fix before next use — add to the loop in `figures_to_pdf.py`:
> ```python
> STYLED = {"iv_and_proxy", "distributional_failure"}  # written by figure_style.save
> if html_path.parent.name in STYLED:
>     continue
> ```
> I documented rather than applied this, as the user asked for documentation
> only. Do it before you run the script.

Two further fixes already applied to `figures_to_pdf.py`, both worth
understanding before editing it:

* It demotes the plot title into the y-axis label **only when the y title is
  literally `"value"`** — that is the plotly-express default for the simulation
  panels, which have no real y label and whose title is their only description.
  The river panels label their own axes and have very long titles; an earlier
  version clobbered their labels. An intermediate attempt
  (`y_title in ("", "value")`) was also wrong: horizontal dot plots have a
  categorical, empty-titled y axis.
* The docstring is a raw string (`r"""`) because it contains `\includegraphics`.

**Staging into the paper is manual**: copy the produced PDFs into
`../notes/Figures/application/` or `../notes/Figures/experiments/`. Currently
staged: 4 application PDFs, 2 experiment PDFs.

---

## 7. Plot legibility rules the user asked for

"Make sure they're easily visible, so e.g. there's no extreme value obfuscating
the view (crop / log-y / whatever suits)." In practice:

* **Cropping is fine; silent cropping is not.** `e1_method_comparison` was
  truncating the proximal bridge intervals at x = 4 (true extent
  [−5.28, 9.38]) with no indication. Fixed by putting it in the axis label:
  `"beta hat (axis clipped to [0, 4])"`, with the true extents given in the
  caption. Same pattern in `dumbbell()`: log x axis, clipped to [0.01, 8],
  stated in the label, unclipped values in the parquet.
* **Check every category label renders.** plotly silently labels every *second*
  categorical tick. The dumbbell figures needed `fig.update_yaxes(dtick=1)`.
* **Size for the width you will actually include at.** The dumbbells carry one
  station-pair label per row. At `0.49\textwidth` they were illegible; my first
  fix (a narrow 520×620 canvas) made it worse, because the label column ate the
  plot area. The working fix was to keep 760×520 and give each placebo its own
  full-width `0.8\textwidth` figure in LaTeX.

---

## 8. Writing into the paper

The user's rule: **all added text in `\nikolajbot{}`, all removals in
`\out{}`.** Defined in `Main.tex`:

```latex
\newcommand{\nikolajbot}[1]{{\color{teal}{\scriptsize NB:}#1}}
\newcommand{\out}[1]{{\color{gray}\st{\tiny #1}}}
```

Mechanics that matter:

* **Keep `figure`/`table` environments *outside* the macro argument.** Wrap only
  the caption text. A float inside `\nikolajbot{...}` breaks.
* Use `\cref` / `\Cref`, and `\labelcref` for bare assumption numbers, matching
  the surrounding text.
* Compile, then diff the warning count against the baseline (§4). Ten
  undefined-reference warnings are pre-existing; any eleventh is yours.

Section 7 structure, as a template: `sec:app-design`, `sec:app-gate`,
`sec:app-headline`, `sec:app-placebo`, `sec:app-summary` — design, then the
falsification gate, then the headline, then the placebos, then a summary that
states plainly what did not work.

---

## 9. Reporting standard

The user asked for the five messages to be conveyed *"to the extent they're
supported by the data."* That qualifier drove the whole write-up. Three results
do not flatter the method and are reported in full rather than buried:

1. Across 20 pairs the certificate beats naive OLS in 20/20, but only **1 of 20**
   lands inside the plausible routing band [0.80, 1.05]. The text says it
   removes roughly half the confounding bias, not all of it.
2. The disconnected-pair placebo does **not** collapse to zero: certified median
   0.502 vs naive 0.664, about 12× the arithmetic null floor of 0.042.
3. The reverse-direction placebo **fails outright** — the certificate is
   *further* from zero than naive in 15 of 17 pairs. There is a structural
   reason (the reversed safeguard selects exactly the events where the upstream
   gauge was large, so the estimator recovers the reciprocal routing ratio), and
   the section states that this estimator does not test causal direction.

Also disclosed rather than dropped: the pre-registration rule was amended
**after** the A1 gate had been run, hence with some estimates visible. Both
headlines are reported, and the pre-amendment estimates are kept in
`results.parquet` under `experiment = "E1_prereg_v1"`.

> **The gate-first pattern is worth reusing.** Before any headline number, the
> agent ran the ablation that could invalidate the exercise (A1: certificate vs
> tail-only at an *identical absolute threshold*, so the arms differ in the
> safeguard alone) plus two permutation ablations (A2 shuffled safeguard,
> A3 safeguard from a different basin). Only after those passed did the headline
> mean anything.

---

## 10. Current state

**Done and in the paper.** All five deliverables. `sections/6.tex` (Simulations,
2 figures) and `sections/7.tex` (Application, 3 figures) both written and
compiling: exit 0, 10 undefined references = baseline, i.e. my edits introduced
none. `tests/test_hydro.py` (14 tests) passes — re-verified 2026-08-31.

Note the simulation section now covers the four-panel `iv_and_proxy` figure and
`distributional_failure`; the separate linear/nonlinear `compare_to_gmm` studies
were folded into panels (c) and (d) of the former by a later revision. Their
scripts and parquets remain under `experiments/compare_to_gmm/` and
`experiments/nonlinear_compare_to_gmm/` but are no longer cited by the paper.

**Run everything (all offline from `hydro/cache/`):**

```
uv run python -m hydro.experiments.a1_gate
uv run python -m hydro.experiments.e1_headline
uv run python -m hydro.experiments.e2_placebo
uv run python -m hydro.experiments.results
uv run pytest tests
```

Simulations: `uv run experiments/<name>.py` (most accept `--smoke` and
`--plot-only`).

**Open items for you:**

1. **Fix the `figures_to_pdf.py` clobber (§6) before running it.**
2. **A live `\nikolaj{}` request in `sections/6.tex`** asks to add the
   correctly-specified quadratic-bridge estimator to panel (d). This appears to
   be **already done** — the caption documents a violet line for it, computed in
   exact arithmetic — so the request is likely stale. Confirm with the user
   before deleting it.
3. **The pre-registration amendment (§9)** is a judgment call the user has not
   yet ruled on. I kept it and disclosed it; the agent flagged that the user
   "may want to overrule this." Reverting would change which pair is the
   headline, so do not touch it unilaterally.
4. **Stale claim in `hydro/experiments/preregistration.md`** — the "95 overlap
   years → 25 events" figure (§4). The paper is correct; the prereg file is not.
5. **Out of scope by the user's own choice**, not oversight: E4, E6, E7, causal
   tail coefficient, extremal quantile regression, F5 heatmap, F6 map.
6. **Uncertainty is indicative, not calibrated.** Year-level bootstrap only
   (400 draws, 90%); no subsampling, no block bootstrap. Under infinite variance
   these intervals are not justified, and `hydro/estimation.py` says so.
   `README.md` at the repo root is empty.

---

## Addendum, 2026-08-31 (presentation pass)

The Application section and its figures were reworked for clarity; no estimate
was recomputed. What changed, and what it supersedes above:

* **`hydro/experiments/paper_figures.py` (new)** builds all four application
  figures from the persisted parquets, in the `figure_style.py` conventions,
  direct to PDF (no tikzplotly). The paper now includes `app_all_pairs.pdf`
  (new: naive vs certified per pair, the section's main figure),
  `app_threshold_stability.pdf`, `app_method_comparison.pdf`,
  `app_placebos.pdf` (E2+E3 combined). The old `e1_*`/`e2_*`/`e3_*` PDFs were
  removed from `../notes/Figures/application/`; their scripts still write them
  under `experiments/river_discharge/`.
* **The §6 landmine is defused**: `figures_to_pdf.py` now skips the
  figure_style-owned folders and `app_*` stems.
* **A number in the paper was corrected.** "20 usable pairs at the
  pre-registered operating point" actually corresponded to a >= 20-event rule;
  the pre-registered rule (>= 30) gives 18 pairs (18/18 closer to target,
  median 1.398 vs 1.667). The paper now reports 18 powered + 3 under-powered
  (open markers), all 21 moving toward the target.
* **The OLS-on-(X, W) caveat** (§9 above; it attains 1.001 on the headline
  pair) is now discussed in the paper rather than omitted.
* `preregistration.md` carries an appended correction for its stale
  "95 years -> 25 events" claim (open item 4 above: resolved).

## Addendum, 2026-09-01 (rework of the application: nested safeguards, decomposition, pure proxies)

Triggered by the coauthor's questions on §7 ("where exactly does W sit, why is
adjusting for W so good, is there any unobserved confounding?"). What changed:

* **Defect found and fixed.** `station_pairs.parquet` predated the maximality
  filter in `Network.intervening_tributaries` by 35 seconds, so 19 of 21
  safeguard vectors contained nested gauges on the same tributary, the
  ungauged-area screen read 0.0 everywhere, and the headline "OLS adjusting for
  W = 1.001" rested on collinear coefficients. See `preregistration.md`,
  amendments 2--4. `candidates.drop_nested` now guards both safeguard builders.
* **Pair set is now 26** (coverage rule replaces the vacuous ungauged screen;
  k = 1..3 non-nested safeguards allowed). Headline is Towanda -> Sunbury.
* **New experiment `hydro/experiments/decomposition.py`** (~15 min with
  bootstrap): the exact split beta = 1 + b_W + b_U of every through-the-origin
  estimate (`estimation.bias_decomposition`), run for two safeguard structures
  per pair -- the tributaries (parents of Y) and the *pure proxies* (gauges
  disconnected from both X and Y, `spec_from_row(kind="proxy")`). New baselines
  in `hydro/baselines`: `ols_subtracting_w` (W fixed at coefficient 1) and
  `proximal_two_proxies` (bridge with (P1, P2)).
* **`paper_figures.py` rewritten**: app_map, app_decomposition, app_proxy,
  app_headline_{sweep,methods}, app_placebos, app_threshold_stability
  (appendix). `pair_table.py` writes the appendix table straight into
  `../notes/sections/Appendices/app_pairs_table.tex`; never type those numbers.
* **Paper**: `sections/7.tex` rewritten (the previous bot draft was replaced,
  not struck out: it was uncommitted bot text, and `\st` cannot hold two pages
  of math); `sections/Appendices/Section7_appendix.tex` written (pipeline,
  rules, amendments, table, appendix figure).

Headline messages the data support (medians over pairs): naive 1.52 = 1 + 0.44
(gauged tributaries) + 0.07 (unseen); certificate with tributaries at q_W = 0.5
1.18 (59% of bias removed, 25/25 closer), at 0.25 1.10 (79%, 18 pairs); pure
proxy certificate 1.32 / 1.21 (46% / 70%); OLS adjusting for W 1.02 (17/26 in
band) -- adjustment wins where the water balance holds, and the paper says so;
two-proxy bridge 1.16 median but range 0.32--18.7. Certified b_U is negative in
13/25 pairs (attenuation/daily lag), stated in the paper.

Run order now: build_pairs -> decomposition -> a1_gate -> e1_headline ->
e2_placebo -> results -> paper_figures -> pair_table -> copy PDFs to
`../notes/Figures/application/`.

## Addendum, 2026-09-02 (W-proxy bridge and the adjusted certificate)

Coauthor request. `decomposition.py adjusted_rows` now also computes, per pair:
`certificate minus W` (Y - sum(W_trib) regressed on T over the certified
events, tributary coefficients FIXED at 1) under both safeguards -- under P via
a combined proxies+tributaries panel with its own record intersection;
`certificate adjusting for W` (free coefficients) -- verified UNSUPPORTED at
10-20 certified events (drifts to -0.06, k>=2 collapses toward 0.65), stored
but not reported; and `proximal (P, W)` (pure proxy treatment-side, largest
tributary outcome-side -- the correctly posed W-proxy bridge; (W1,W2) would put
two outcome-side proxies in one bridge and is not run). Headline messages:
cert-minus-W at q_W = 0.25 has median 0.99 (tributary safeguard, 11/18 in band)
and 0.96 (pure proxy, 5/8); (P,W) bridge median 0.90, range -0.86 to 1.68.
Fig app_proxy gained green crosses, the headline forest three rows (with
q_W = 0.5 fallback labels), the appendix table a cert.-W column.
