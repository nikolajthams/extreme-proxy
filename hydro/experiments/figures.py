"""PDF previews of the application figures, from the persisted parquets only.

    uv run python -m hydro.experiments.figures

    app_all_pairs.pdf            all pairs: one kernel density per estimator
    app_placebos.pdf             the disconnected-pair placebo, same style
    app_threshold_stability.pdf  headline pair: certified OLS against v by q_P

Ridgelines: one Gaussian kernel density per estimator over pairs on a common
density scale, values clipped to the axis window before dots and density (so
an outlier appears as mass at the edge, and the clipped count is printed on
the right), the pairs as faint dots, the interquartile range shaded more
strongly under the density, a vertical line at the median. Written to
experiments/river_discharge/ only: the paper uses the pgfplots versions that
`tikz.py` writes from the same geometry (`ridge_geometry`, `THRESHOLD_BLUES`).
"""

from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from figure_style import COLORS, LINE_WIDTH, MARKER_SIZE, apply_layout, legend_entry, save  # noqa: E402

EST = CONFIG.estimator
Q_P = EST.q_p
GREY = "#8a8a8a"
CERT = COLORS["certified"]    # blue: the certified estimators
BRIDGE = COLORS["repaired"]   # violet: the proximal bridge
NAIVE = COLORS["classical"]   # orange: naive OLS
JITTER_SEED = 20260902
# Bandwidth factor relative to Silverman's 0.9 (settled with the coauthor).
KDE_FACTOR = 0.675
# One blue per safeguard level of the threshold figure, darker as it tightens.
THRESHOLD_BLUES = {0.9: "#c4d7ec", 0.75: "#8fb3dd", 0.5: "#5a8bcb", 0.25: "#2b5fa8", 0.1: "#123f7e",
                   0.05: "#061d3d"}


def arm(d: pl.DataFrame, method: str, q_p: float | None = None) -> pl.DataFrame:
    f = (pl.col("method") == method) & pl.col("beta").is_not_null()
    if q_p is not None:
        f = f & (pl.col("q_p") == q_p)
    return d.filter(f)


def _save(fig: go.Figure, name: str) -> None:
    save(fig, OUTPUT_DIR / name)


def _kde(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE on the span of `grid`, bandwidth KDE_FACTOR * min(sd,
    IQR/1.34) * n^(-1/5) with a floor of 0.03; values are clipped to the span
    and the kernel reflected at both ends."""
    lo, hi = float(grid[0]), float(grid[-1])
    x = np.clip(np.asarray(vals, float), lo, hi)
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = (q75 - q25) / 1.34
    s = min(sd, iqr) if iqr > 0 else sd
    h = max(KDE_FACTOR * s * n ** (-0.2), 0.03)

    def k(points):
        z = (points[:, None] - x[None, :]) / h
        return np.exp(-0.5 * z ** 2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))

    return k(grid) + k(2 * lo - grid) + k(2 * hi - grid)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def ridge_geometry(rows: list[tuple[str, list[float], str]], lo: float, hi: float,
                   peak: float = 0.85) -> list[dict]:
    """What the ridgelines are drawn from, one dict per estimator: the density
    `height` over `grid` on a common scale (`peak` row units at the highest
    mode), the clipped quartiles, the jittered pair dots as (value, height
    above the baseline) split into `inside` the window and `clipped` to it,
    the median with the density height there, and the number of clipped
    pairs. Shared by the plotly figure and its TikZ version in `tikz.py`."""
    grid = np.linspace(lo, hi, 400)
    dens = [_kde(values, grid) for _, values, _ in rows]
    scale = peak / max(float(d.max()) for d in dens)
    rng = np.random.default_rng(JITTER_SEED)
    inset = 0.015 * (hi - lo)
    out = []
    for (label, values, color), d in zip(rows, dens):
        vals = np.asarray(values, float)
        q25, q75 = (float(np.clip(np.quantile(vals, q), lo, hi)) for q in (0.25, 0.75))
        ys = rng.uniform(0.04, 0.26, len(vals))
        med = float(np.median(vals))
        out.append(dict(
            label=label, color=color, grid=grid, height=scale * d, q25=q25, q75=q75,
            inside=[(float(v), float(y)) for v, y in zip(vals, ys) if lo <= v <= hi],
            clipped=[(min(max(float(v), lo + inset), hi - inset), float(y)) for v, y in zip(vals, ys)
                     if not lo <= v <= hi],
            median=med, median_height=scale * float(np.interp(med, grid, d)) if lo <= med <= hi else 0.0,
            n_clipped=int(((vals < lo) | (vals > hi)).sum()),
        ))
    return out


def _ridges(fig: go.Figure, rows: list[tuple[str, list[float], str]], lo: float, hi: float) -> list[str]:
    order = []
    for idx, g in enumerate(ridge_geometry(rows, lo, hi)):
        order.append(g["label"])
        color, grid = g["color"], g["grid"]
        fig.add_trace(go.Scatter(x=[lo, hi], y=[idx, idx], mode="lines", showlegend=False,
                                 line=dict(color=color, width=0.5), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=idx - g["height"], mode="lines", showlegend=False,
                                 fill="tonexty", fillcolor=_rgba(color, 0.22),
                                 line=dict(color=color, width=1.5), hoverinfo="skip"))
        # the interquartile range, shaded more strongly under the density
        inner = (grid >= g["q25"]) & (grid <= g["q75"])
        if inner.any():
            fig.add_trace(go.Scatter(x=[g["q25"], *grid[inner], g["q75"]],
                                     y=[idx, *(idx - g["height"][inner]), idx], mode="lines",
                                     showlegend=False, fill="toself", fillcolor=_rgba(color, 0.3),
                                     line=dict(width=0), hoverinfo="skip"))
        for pts, sym in ((g["inside"], "circle"), (g["clipped"], "circle-open")):
            if pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[idx - p[1] for p in pts], mode="markers", showlegend=False,
                    marker=dict(color=color, size=MARKER_SIZE - 1, symbol=sym, opacity=0.2,
                                line=dict(width=1, color=color)), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[g["median"], g["median"]], y=[idx, idx - g["median_height"]], mode="lines",
                                 showlegend=False, line=dict(color=color, width=3)))
        if g["n_clipped"]:
            fig.add_annotation(x=hi, y=idx, text=f"{g['n_clipped']} clipped", showarrow=False, xanchor="left",
                               yanchor="bottom", font=dict(size=10, color=GREY))
    return order


def _ridge_figure(rows, lo, hi, target, name, width: int = 675, left: int = 180, row_h: int = 64) -> None:
    fig = go.Figure()
    fig.add_vline(x=target, line_dash="dot", line_width=1, opacity=0.6)
    order = _ridges(fig, rows, lo, hi)
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, autorange="reversed",
                     range=[len(order) - 0.6, -1.0], showgrid=False, zeroline=False)
    fig.update_xaxes(title=f"beta hat (values clipped to [{lo:g}, {hi:g}])", range=[lo - 0.02, hi + 0.02])
    apply_layout(fig, width, 90 + row_h * len(order), legend_y=-0.2)
    fig.update_layout(margin=dict(l=left, r=70, b=60))
    _save(fig, name)


def _rows(d: pl.DataFrame) -> list[tuple[str, list[float], str]]:
    return [
        ("naive OLS", arm(d, "naive OLS")["beta"].to_list(), NAIVE),
        ("proximal<br>(P1, P2) bridge", arm(d, "proximal bridge")["beta"].to_list(), BRIDGE),
        ("certified OLS", arm(d, "certified OLS", Q_P)["beta"].to_list(), CERT),
    ]


def all_pairs() -> None:
    """Compact, to sit next to the map in the paper; the taller rows make the
    panel as high as the map so that the two subfigures align."""
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet")
    _ridge_figure(_rows(d), -1.0, 3.0, CONFIG.beta_target, "app_all_pairs", width=480, left=120, row_h=108)


def placebos() -> None:
    d = pl.read_parquet(OUTPUT_DIR / "placebo.parquet")
    _ridge_figure(_rows(d), -2.0, 2.0, 0.0, "app_placebos")


def threshold_stability() -> None:
    """Headline pair: certified OLS against the treatment threshold v, one
    line per safeguard level, every point with at least one certified day."""
    from hydro.pairs import headline_pair

    pair = headline_pair()["pair"]
    stab = (pl.read_parquet(OUTPUT_DIR / "headline.parquet")
            .filter((pl.col("method") == "certified OLS") & (pl.col("n_cert") >= EST.min_events)
                    & pl.col("beta").is_not_null())
            .sort("v"))
    naive = float(arm(pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet").filter(pl.col("pair") == pair),
                      "naive OLS")["beta"][0])
    fig = go.Figure()
    fig.add_hline(y=CONFIG.beta_target, line_dash="dot", line_width=1, opacity=0.6)
    fig.add_hline(y=naive, line_dash="dash", line_width=LINE_WIDTH, line_color=NAIVE, opacity=0.9)
    for q_p, color in THRESHOLD_BLUES.items():
        sub = stab.filter(pl.col("q_p") == q_p)
        if sub.height:
            fig.add_trace(go.Scatter(x=sub["v"], y=sub["beta"], mode="lines+markers", name=f"q_P = {q_p}",
                                     line=dict(color=color, width=LINE_WIDTH),
                                     marker=dict(color=color, size=MARKER_SIZE)))
    fig.add_trace(legend_entry("naive OLS", NAIVE, dash="dash"))
    fig.update_xaxes(title="treatment threshold v (m3/s)")
    fig.update_yaxes(title="beta hat", range=[0.4, 2.4])
    apply_layout(fig, 675, 480, legend_y=-0.18)
    _save(fig, "app_threshold_stability")


def main() -> None:
    all_pairs()
    placebos()
    threshold_stability()


if __name__ == "__main__":
    main()
