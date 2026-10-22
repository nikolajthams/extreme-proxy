"""Reproduce the whole application from the cached USGS data.

    uv run python -m hydro.pipeline.run_all                 (~10 min)
    uv run python -m hydro.pipeline.run_all --no-bootstrap

Steps, each a module with its own `main()`:
  1. hydro.pairs               pair selection -> pairs.parquet, placebo_pairs.parquet
  2. pipeline.all_pairs        every estimator on every pair -> all_pairs.parquet
  3. pipeline.headline         threshold grid and ablations of the headline pair -> headline.parquet
  4. pipeline.placebo          disconnected pairs -> placebo.parquet
  5. pipeline.tikz             the pgfplots figures
  6. pipeline.pair_table       the per-pair table of the appendix
  7. pipeline.paper_numbers    every number the text quotes
"""

from __future__ import annotations

import time

from hydro import pairs
from hydro.pipeline import all_pairs, headline, pair_table, paper_numbers, placebo, tikz


def main() -> None:
    for step in (pairs, all_pairs, headline, placebo, tikz, pair_table, paper_numbers):
        t = time.time()
        print(f"\n===== {step.__name__} =====", flush=True)
        step.main()
        print(f"----- {step.__name__} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
