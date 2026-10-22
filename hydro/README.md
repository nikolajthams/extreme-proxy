# River-discharge application

The code behind Section 7 (the application) and its appendix. The routing
coefficient of upstream on downstream daily discharge is estimated with the
certified OLS of the paper on the full daily record, with gauges disconnected
from both river gauges as the proxy safeguard, and compared with naive OLS and
the two-proxy proximal bridge on the same days.

## Reproduce

```
uv run python -m hydro.pipeline.run_all    # ~10 min, every step below in order
```

All USGS responses (NWIS daily values and site metadata, NLDI navigations) are
cached under `hydro/cache/` and versioned, so the pipeline runs offline and
deterministically: the seeds are fixed in `config.py`, and re-running
reproduces every parquet, figure and table. Two derived caches are keyed by
name only: `cache/screen/` (the gauge screen; delete it after changing
`GaugeScreen`) and `cache/dv/` (daily values per site; delete it after
changing the date range). Deleting the cache re-downloads the raw data from
USGS; the gauge screen and the pair set regenerate from the raw responses
(`tests/test_hydro.py::test_headline_pair_reproduces_from_the_cache` checks
the headline pair against the persisted results).

## Pipeline

| step | module | writes (`hydro/results/`) |
|---|---|---|
| 1 | `hydro.pairs` | `pair_candidates.parquet` (every connected pair with its screen result), `pairs.parquet` (the 40 pairs with proxies, lag and events), `placebo_candidates.parquet`, `placebo_pairs.parquet` |
| 2 | `hydro.pipeline.all_pairs` | `all_pairs.parquet`: naive OLS, the proximal bridge and the certified OLS along the safeguard grid, all on the daily record, with year-bootstrap intervals |
| 3 | `hydro.pipeline.headline` | `headline.parquet`: the headline pair's threshold grid and the ablations V1--V3 |
| 4 | `hydro.pipeline.placebo` | `placebo.parquet`: the same estimators on 40 disconnected pairs (true coefficient 0) |
| 5 | `hydro.pipeline.tikz` | `app_map.tex`, `app_all_pairs.tex`, `app_headline_sweep.tex`, `app_headline_methods.tex`, `app_placebos.tex`, `app_threshold_stability.tex` (pgfplots fragments the paper inputs) |
| 6 | `hydro.pipeline.pair_table` | `app_pairs_table.tex`: the per-pair table of the appendix |
| 7 | `hydro.pipeline.paper_numbers` | prints every number quoted in the paper, in the order of the text |

Each module has a `main()` and can be run alone with
`uv run python -m <module>`; steps 5--7 read only the parquets of steps 1--4.
With the environment variable `PAPER_DIR` set to the paper's source directory,
steps 5 and 6 also copy their output into `Figures/application/` and
`sections/Appendices/` there.

## Modules

- `config.py`: every constant (regions and seasons, gauge screen, pair rules,
  alignment, declustering, quality flags, estimator grids and operating rules,
  bootstrap). The appendix quotes these values.
- `data/`: `http.py` (cached GET), `nwis.py` (USGS daily discharge and site
  metadata, converted to m3/s and km2), `nldi.py` (upstream navigation),
  `network.py` (upstream sets, disconnection, nested gauges, distances).
- `preprocess/`: `screening.py` (gauge screen, complete years per gauge),
  `alignment.py` (travel-time lag from flood waves, celerity check),
  `decluster.py` (runs declustering), `panel.py` (the aligned daily table and
  the event table of a pair).
- `estimation.py`: the safeguard set, threshold and operating point, the
  certified OLS (through `certification_estimators`), naive OLS, the proximal
  bridge, the Hill estimate, the year-level bootstrap.
- `pairs.py`: pair selection for the main design and the placebo, the
  headline-pair rule.
- `pipeline/`: the steps above.

## Conventions

Columns are named as in the paper: `T` (treatment, upstream), `Y` (outcome,
downstream, read one travel time later), `P0..P2` (proxy safeguard). Parquets
are long: one row per (pair, method, `q_p`) with `q_t`, `v`, `n_cert`, `beta`,
`ci_lo`, `ci_hi`; everything is fitted on the daily record (the declustered
event table of a panel is a diagnostic only). Method names are the paper's:
`naive OLS`, `proximal bridge`, `certified OLS`, `V1 tail only`,
`V2 permuted safeguard`, `V3 other-basin safeguard`.
