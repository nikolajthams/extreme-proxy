# Extreme proxies for causal identification: deconfounding with outliers

Code and data for the paper *Extreme proxies for causal identification:
deconfounding with outliers* by Juraj Bodik and Nikolaj Thams. Everything
in the paper that is computed, the simulations of Section 6, the
confidence-interval and combining appendices and the river-discharge
application of Section 7, can be reproduced from this repository.

## Layout

| directory | contents |
|---|---|
| `certification_estimators/` | the method: the certifying event (`certificate.py`), the certified OLS and single-maximum estimator (`estimators.py`), the rank-based confidence interval (`rank_interval.py`); the structural equation models (`dag.py`), graphs (`graphs.py`) and classical baselines (`baselines.py`) of the simulations |
| `simulations/` | one module per simulation study, each with its stored per-repetition estimates, summary and figure or table fragment in the directory of its name |
| `hydro/` | the river-discharge application: data access, preprocessing, estimation, the pipeline and its results, and the cached USGS data; see `hydro/README.md` |
| `tests/` | `uv run pytest` (~1 min) |
| `tikz_style.py` | colours, axis styles and the writer shared by every figure and table of the paper |

## What produces what

| in the paper | script | output |
|---|---|---|
| Figure 5, Table S3 and the numbers of the simulation appendix | `simulations/iv_and_proxy.py` | `simulations/iv_and_proxy/iv_and_proxy.tex`, `stats.csv`, `slopes.csv` |
| Figure 6 and its numbers | `simulations/distributional_failure.py` | `simulations/distributional_failure/distributional_failure.tex`, `stats.csv` |
| Table S1 (coverage of the rank interval) | `simulations/rank_interval.py` | `simulations/rank_interval/rank_interval_sim_table.tex`, `coverage.csv` |
| Table S2 (combining the certified OLS and the bridge) | `simulations/combine_estimators.py` | `simulations/combine_estimators/combine_estimators_sim_table.tex`, `summary.csv` |
| Figures 7, 8, S1, S2, Table S4 and every number of Section 7 and its appendix | `hydro/pipeline/run_all.py` | `hydro/results/app_*.tex`, `hydro/results/*.parquet`, the printout of `hydro/pipeline/paper_numbers.py` |

Figure and table numbers refer to the current draft; the fragments are the
files the paper `\input`s.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/); `uv sync`
creates the environment from `uv.lock` (Python 3.14, numpy, polars, networkx,
tqdm, pytest). Every command below is run from the repository root.

```
uv sync
uv run pytest
```

## Reproducing

Every simulation stores its per-repetition estimates in `estimates.parquet`,
so the figures and tables can be regenerated without rerunning:

```
uv run python -m simulations.iv_and_proxy --plot-only
uv run python -m simulations.distributional_failure --plot-only
uv run python -m simulations.rank_interval --plot-only
uv run python -m simulations.combine_estimators --plot-only
```

Without `--plot-only` a script reruns its 1,000 repetitions (`--reps` and
`--workers` control the size and the parallelism; a repetition is seeded by
its index alone, so any subset reproduces the stored rows). The application
runs offline from the cached USGS data in about ten minutes:

```
uv run python -m hydro.pipeline.run_all
```

Set the environment variable `PAPER_DIR` to the paper's source directory to
have the figure and table fragments copied into its `Figures/experiments/`,
`Figures/application/` and `sections/Appendices/` as they are written.

## Data

The application uses daily mean discharge (USGS NWIS parameter 00060) for
184 gauges in the Susquehanna, Wabash and Iowa/Cedar basins, 1930 to 2024,
and the river-network topology from the USGS NLDI service. Every response is
cached under `hydro/cache/` (49 MB), so no download is needed; deleting the
cache makes the pipeline fetch the data again.
