"""Reproduce the whole application from the cached USGS data.

    uv run python -m hydro.experiments.run_all               (~10 min)
    uv run python -m hydro.experiments.run_all --no-bootstrap

Steps, each a module with its own `main()`:
  1. hydro.pairs                 pair selection -> pairs.parquet, placebo_pairs.parquet
  2. experiments.all_pairs       every estimator on every pair -> all_pairs.parquet
  3. experiments.headline        threshold grid + ablations -> headline.parquet
  4. experiments.rank_interval   rank intervals on the declustered certified days
                                 -> headline_rank_interval.parquet, all_pairs_rank_interval.parquet
  5. experiments.placebo         disconnected pairs -> placebo.parquet
  6. experiments.figures         PDF previews of the figures
  7. experiments.tikz            TikZ figures (copied to the notes)
  8. experiments.pair_table      appendix tables (written into the notes)
  9. experiments.paper_numbers   every number the text quotes
"""

from __future__ import annotations

import time

from hydro import pairs
from hydro.experiments import all_pairs, figures, headline, pair_table, paper_numbers, placebo, rank_interval, tikz


def main() -> None:
    for step in (pairs, all_pairs, headline, rank_interval, placebo, figures, tikz, pair_table, paper_numbers):
        t = time.time()
        print(f"\n===== {step.__name__} =====", flush=True)
        step.main()
        print(f"----- {step.__name__} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
