"""Shared figure conventions for the simulation figures in the notes.

One colour per *role* (the certified estimator, the classical competitor, and a
certificate the graphical criterion disqualifies), one dash per tail index or
sample size, one legend per figure. Every figure is written straight to PDF
here; `figures_to_pdf.py` skips these folders.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

# Three categorical slots that validate all-pairs for colour-vision deficiency.
COLORS = {
    "certified": "#2a78d6",
    "classical": "#eb6834",
    "disqualified": "#1baf7a",
    "repaired": "#4a3aa7",
}
LINE_WIDTH = 2
MARKER_SIZE = 7
FONT_SIZE = 15


def legend_entry(name: str, color: str = "#000000", dash: str = "solid") -> go.Scatter:
    """An empty trace that only contributes a legend entry."""
    return go.Scatter(
        x=[None], y=[None], mode="lines", name=name, showlegend=True,
        line=dict(color=color, dash=dash, width=LINE_WIDTH),
    )


def line(x, y, color: str, dash: str = "solid", marker_symbol: str = "circle") -> go.Scatter:
    return go.Scatter(
        x=x, y=y, mode="lines+markers", showlegend=False,
        line=dict(color=color, dash=dash, width=LINE_WIDTH),
        marker=dict(color=color, size=MARKER_SIZE, symbol=marker_symbol),
    )


def apply_layout(fig: go.Figure, width: int, height: int, legend_y: float = -0.14) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        width=width, height=height,
        font=dict(size=FONT_SIZE),
        margin=dict(l=70, r=20, t=40, b=90),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=legend_y, yanchor="top",
                    font=dict(size=FONT_SIZE - 1)),
    )
    fig.update_annotations(font_size=FONT_SIZE)
    return fig


def save(fig: go.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(stem.with_suffix(".html"))
    fig.write_image(stem.with_suffix(".pdf"))
    print(f"wrote {stem.with_suffix('.pdf')}")
