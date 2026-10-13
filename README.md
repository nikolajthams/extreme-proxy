# extreme-proxy: code

- `certification_estimators/`: the certified estimators of the paper (certificate, safeguard sets, estimator types).
- `experiments/`: the simulation experiments of Section 6 (`iv_and_proxy.py`, `distributional_failure.py`, `exact_iv.py`; shared figure style in `figure_style.py`) the coverage study of the rank interval in the confidence-interval appendix (`rank_interval.py`), and the precision-weighted combination of the certified estimator with the bridge in the combining appendix (`combine_estimators.py`).
- `hydro/`: the river-discharge application of Section 7; see `hydro/README.md`.
- `tests/`: `uv run pytest`.

Depends on the editable `graphical-toolbox` package (`pyproject.toml`).

