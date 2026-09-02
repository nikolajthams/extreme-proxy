"""Publication figures for the river-discharge application (the paper's Application section).

Reads ONLY the persisted parquets under experiments/river_discharge/ (run
decomposition, a1_gate, e1_headline, e2_placebo and results first) and writes
the styled PDFs the paper includes, in the shared conventions of
experiments/figure_style.py: one colour per role, horizontal legend,
direct-to-PDF (no tikzplotly).

    app_map.pdf                   the headline pair on the gauge network: where T, Y,
                                  the tributary safeguards and the pure proxies sit
    app_decomposition.pdf         every pair: naive and certified bias split into the
                                  gauged-tributary share and the unmonitored share
    app_proxy.pdf                 every pair with a pure-proxy safeguard: tributary vs
                                  proxy certificate vs the two-proxy bridge
    app_headline_sweep.pdf        headline pair: beta along the safeguard grid, both structures
    app_headline_methods.pdf      headline pair: every method at q_W = 0.5 with intervals
    app_placebos.pdf              disconnected-pair and reverse-direction placebos
    app_threshold_stability.pdf   (appendix) headline pair, beta vs v by safeguard level

`uv run python -m hydro.experiments.paper_figures`
"""

from __future__ import annotations

import math
import re
import sys

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from hydro.config import CONFIG, OUTPUT_DIR, REPO_ROOT
from hydro.experiments.decomposition import FALLBACK_Q_W, Q_W_GRID, TABULATED_Q_W

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from figure_style import COLORS, LINE_WIDTH, MARKER_SIZE, apply_layout, legend_entry, save

EST = CONFIG.estimator
Q_W = TABULATED_Q_W
BAND = CONFIG.beta_routing_band
GREY = "#8a8a8a"
LIGHT = "#cccccc"
PROXY = COLORS["disqualified"]   # green: the pure-proxy safeguard structure
ADJUST = COLORS["repaired"]      # violet: the adjustment / bridge estimators
TRIB = COLORS["certified"]       # blue: the tributary safeguard structure
NAIVE = COLORS["classical"]      # orange


def town(name: str) -> str:
    """'Susquehanna River at Danville, PA' -> 'Danville'. Figure rows need names
    a reader can hold; the station ids stay in the parquets."""
    part = re.split(r"\s+(?:at|near|nr|AT|NEAR|NR)\s+", name)[-1]
    part = re.sub(r"[,.]?\s*(PA|NY|IN|IL|IA|MD|IND)\.?\s*$", "", part.strip())
    return part.strip(" ,").title()


def _label(df: pl.DataFrame, sep: str = " → ") -> pl.DataFrame:
    """Row labels 'Town → Town'. Two gauges can share a town (the White River
    has one gauge at Noblesville and one near it), so a label that collides
    gets the upstream station id appended."""
    df = df.with_columns(
        (pl.col("x_name").map_elements(town, return_dtype=pl.Utf8) + sep
         + pl.col("y_name").map_elements(town, return_dtype=pl.Utf8)).alias("label")
    )
    dup = df.group_by("label").len().filter(pl.col("len") > 1)["label"]
    return df.with_columns(
        pl.when(pl.col("label").is_in(dup.implode()))
        .then(pl.col("label") + " (" + pl.col("pair").str.slice(0, 8) + ")")
        .otherwise(pl.col("label")).alias("label")
    )


def _decomp() -> pl.DataFrame:
    return pl.read_parquet(OUTPUT_DIR / "decomposition.parquet")


def _cert(d: pl.DataFrame, structure: str, method: str = "certificate") -> pl.DataFrame:
    """A certified estimator at the tabulated level, falling back to
    FALLBACK_Q_W for pairs that do not keep ten events there; `fallback` says
    which."""
    main = _arm(d, structure, method, Q_W).with_columns(pl.lit(False).alias("fallback"))
    fb = _arm(d, structure, method, FALLBACK_Q_W).with_columns(pl.lit(True).alias("fallback"))
    fb = fb.filter(~pl.col("pair").is_in(main["pair"].implode()))
    return pl.concat([main, fb], how="vertical")


def _arm(d: pl.DataFrame, structure: str, method: str, q_w: float | None = None) -> pl.DataFrame:
    f = (pl.col("structure") == structure) & (pl.col("method") == method) & pl.col("beta").is_not_null()
    if q_w is not None:
        f = f & (pl.col("safeguard_quantile") == q_w)
    return d.filter(f)


def _clip_markers(fig, xs, ys, lo, hi, color, symbol, size=MARKER_SIZE + 2, open_all=False, **kw) -> None:
    """Markers clipped to [lo, hi]; a clipped value is drawn open at the edge so
    the clipping is visible, never silent. Unclipped values stay in the parquet.
    `open_all` draws every marker open (used for fallback-level estimates)."""
    inset = 0.015 * (hi - lo)  # a marker on the axis boundary renders half-cut
    inside = [(x, y) for x, y in zip(xs, ys) if x is not None and lo <= x <= hi and not open_all]
    outside = [(min(max(x, lo + inset), hi - inset), y) for x, y in zip(xs, ys)
               if x is not None and (open_all or not lo <= x <= hi)]
    for pts, sym in ((inside, symbol), (outside, symbol + "-open")):
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", showlegend=False,
                marker=dict(color=color, size=size, symbol=sym, line=dict(width=1.5, color=color)),
            ), **kw)


# ------------------------------------------------------------------ map


def geometry_map() -> None:
    """The headline pair on its gauge network.

    Every screened gauge of the basin is a grey dot joined to the next gauge
    downstream. The treatment T and outcome Y are black; the main stem between
    them is drawn black. Tributary safeguards W (blue) sit on side streams
    entering between T and Y: water passing T never passes them, they feed Y.
    Pure proxies P (green) are disconnected from both T and Y: they share the
    storm and feed neither. Z (open square) is the headwater gauge used as
    instrument. Everything else the theory needs is visible: no safeguard is
    downstream of T, and P has no path into Y.
    """
    from hydro.experiments.candidates import headline_row, network_for, spec_from_row

    row = headline_row()
    net = network_for(row["region"])
    trib, prox = spec_from_row(row), spec_from_row(row, kind="proxy")
    meta = net.meta.select("site_no", "station_nm", "lat", "lon")
    pos = {r["site_no"]: (r["lon"], r["lat"]) for r in meta.iter_rows(named=True)}
    names = {r["site_no"]: r["station_nm"] for r in meta.iter_rows(named=True)}

    def parent(g: str) -> str | None:
        down = [d for d, up in net.upstream.items() if g in up and d != g and d in pos]
        return min(down, key=lambda d: len(net.upstream[d])) if down else None

    on_stem = {g for g in pos if net.is_upstream_of(row["x"], g) or g == row["x"]} & (
        net.upstream[row["y"]] | {row["y"]})
    fig = go.Figure()
    for g in pos:
        p = parent(g)
        if p is None:
            continue
        stem = g in on_stem and p in on_stem
        (x0, y0), (x1, y1) = pos[g], pos[p]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="black" if stem else LIGHT, width=3 if stem else 1),
        ))
    special = {row["x"], row["y"], *trib.w, *prox.w, *([trib.z] if trib.z else [])}
    rest = [g for g in pos if g not in special]
    fig.add_trace(go.Scatter(
        x=[pos[g][0] for g in rest], y=[pos[g][1] for g in rest], mode="markers", showlegend=False,
        marker=dict(color=GREY, size=5), hovertext=[names[g] for g in rest],
    ))
    # Label sides alternate so that gauges a few km apart do not overprint;
    # the map is zoomed to the labelled gauges, the rest of the basin's
    # network is drawn as far as the frame allows.
    groups = [
        ([row["x"]], "T (treatment)", "black", "square", ["T"], ["top left"]),
        ([row["y"]], "Y (outcome)", "black", "square", ["Y"], ["middle right"]),
        (trib.w, "W: tributary safeguard", TRIB, "circle", [f"W{i + 1}" for i in range(len(trib.w))],
         ["bottom left", "bottom left", "middle right"]),
        (prox.w, "P: pure-proxy safeguard", PROXY, "diamond", [f"P{i + 1}" for i in range(len(prox.w))],
         ["bottom left", "middle right", "bottom right"]),
    ]
    if trib.z:
        groups.append(([trib.z], "Z (headwater instrument)", "black", "square-open", ["Z"], ["middle right"]))
    for gauges, name, color, symbol, tags, sides in groups:
        fig.add_trace(go.Scatter(
            x=[pos[g][0] for g in gauges], y=[pos[g][1] for g in gauges], mode="markers+text",
            name=name, text=[f"{t}: {town(names[g])}" for t, g in zip(tags, gauges)],
            textposition=sides[:len(gauges)], textfont=dict(size=12),
            marker=dict(color=color, size=12, symbol=symbol, line=dict(width=2, color=color)),
        ))
    lat0 = sum(p[1] for p in pos.values()) / len(pos)
    lons = [pos[g][0] for g in special]
    lats = [pos[g][1] for g in special]
    pad = 0.35
    # Coordinates carry no information the reader needs (coauthor request):
    # the map is a sketch of who sits where, not a locator.
    fig.update_xaxes(visible=False, range=[min(lons) - pad, max(lons) + pad])
    fig.update_yaxes(visible=False, range=[min(lats) - pad, max(lats) + pad],
                     scaleanchor="x", scaleratio=1 / math.cos(math.radians(lat0)))
    apply_layout(fig, 675, 600, legend_y=-0.1)
    save(fig, OUTPUT_DIR / "app_map")


# ------------------------------------------------------- decomposition


def decomposition_all_pairs() -> None:
    """Every pair: the naive and the certified bias, each split into the share
    carried by the gauged tributaries (b_W) and the share they do not see (b_R;
    `b_U` in the parquet).

    Bars start at the target 1; a negative b_R points left. Naive bars are
    light, certified bars (q_W = Q_W, largest q_x keeping >= 10 events; the
    FALLBACK_Q_W level, so labelled, where Q_W keeps fewer) dark, with n_cert
    printed at their end. The violet diamonds are the two
    adjustment estimators, which fit or fix the tributary coefficients.
    """
    d = _decomp()
    naive = _arm(d, "tributary", "naive OLS").select(
        "pair", "x_name", "y_name", "reference_area_ratio",
        pl.col("beta").alias("naive"), pl.col("b_W").alias("bW0"), pl.col("b_U").alias("bU0"))
    cert = _cert(d, "tributary").select(
        "pair", pl.col("beta").alias("cert"), pl.col("b_W").alias("bW1"), pl.col("b_U").alias("bU1"),
        "n_cert", "fallback")
    adj = _arm(d, "tributary", "OLS adjusting for W").select("pair", pl.col("beta").alias("adj"))
    sub = _arm(d, "tributary", "OLS subtracting W").select("pair", pl.col("beta").alias("sub"))
    df = _label(naive.join(cert, on="pair", how="left").join(adj, on="pair", how="left")
                .join(sub, on="pair", how="left")).sort("naive")
    assert df["label"].n_unique() == df.height, "short pair labels must be unique"

    fig = go.Figure()
    fig.add_vrect(x0=BAND[0], x1=BAND[1], fillcolor=TRIB, opacity=0.08, line_width=0)
    fig.add_vline(x=1.0, line_dash="dot", line_width=1, opacity=0.6)
    for arm, (bw, bu), opacity in (("naive", ("bW0", "bU0"), 0.4), ("certified", ("bW1", "bU1"), 1.0)):
        sub_df = df.filter(pl.col(bw).is_not_null())
        fig.add_trace(go.Bar(
            y=sub_df["label"], x=sub_df[bw], base=[1.0] * sub_df.height, orientation="h",
            offsetgroup=arm, marker=dict(color=TRIB, opacity=opacity), showlegend=False, width=0.38,
        ))
        fig.add_trace(go.Bar(
            y=sub_df["label"], x=sub_df[bu], base=(1.0 + sub_df[bw]).to_list(), orientation="h",
            offsetgroup=arm, marker=dict(color=GREY, opacity=opacity), showlegend=False, width=0.38,
        ))
    for r in df.filter(pl.col("n_cert").is_not_null()).iter_rows(named=True):
        # The certified bar ends at 1 + b_W + max(b_U, 0): a negative b_U is
        # drawn back over the blue segment, so the label sits past the blue.
        end = 1.0 + r["bW1"] + max(r["bU1"], 0.0)
        tag = f"n={r['n_cert']}" + (f" (q_W = {FALLBACK_Q_W})" if r["fallback"] else "")
        fig.add_annotation(x=end + 0.02, y=r["label"], yshift=-7, text=tag,
                           showarrow=False, xanchor="left", font=dict(size=9, color=GREY))
    fig.add_trace(go.Scatter(x=df["adj"], y=df["label"], mode="markers", showlegend=False,
                             marker=dict(color=ADJUST, size=MARKER_SIZE + 1, symbol="diamond")))
    fig.add_trace(go.Scatter(x=df["sub"], y=df["label"], mode="markers", showlegend=False,
                             marker=dict(color=ADJUST, size=MARKER_SIZE + 1, symbol="diamond-open",
                                         line=dict(width=1.5, color=ADJUST))))
    fig.add_trace(go.Scatter(x=df["reference_area_ratio"], y=df["label"], mode="markers", showlegend=False,
                             marker=dict(color=GREY, symbol="line-ns-open", size=11,
                                         line=dict(width=1.5, color=GREY))))
    fig.add_traces([
        go.Bar(x=[None], y=[None], name="gauged-tributary share b_W", marker=dict(color=TRIB)),
        go.Bar(x=[None], y=[None], name="unmonitored share b_R", marker=dict(color=GREY)),
        go.Scatter(x=[None], y=[None], mode="markers", name="OLS adjusting for W",
                   marker=dict(color=ADJUST, size=MARKER_SIZE + 1, symbol="diamond")),
        go.Scatter(x=[None], y=[None], mode="markers", name="OLS subtracting W",
                   marker=dict(color=ADJUST, size=MARKER_SIZE + 1, symbol="diamond-open",
                               line=dict(width=1.5, color=ADJUST))),
        go.Scatter(x=[None], y=[None], mode="markers", name="drainage-area ratio",
                   marker=dict(color=GREY, symbol="line-ns-open", size=11, line=dict(width=1.5, color=GREY))),
    ])
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(dtick=1, categoryorder="array", categoryarray=df["label"].to_list())
    fig.update_xaxes(title="beta hat = 1 + b_W + b_R", range=[0.55, 2.25])
    apply_layout(fig, 675, 900, legend_y=-0.07)
    fig.update_layout(margin=dict(b=110), font=dict(size=14))
    save(fig, OUTPUT_DIR / "app_decomposition")


# ----------------------------------------------------- proxy vs tributary


def proxy_vs_tributary() -> None:
    """Every pair with a pure-proxy safeguard: the certificate with the
    tributary safeguard, with the pure-proxy safeguard, least squares on
    (T, W), and the two-proxy bridge, against naive OLS. Certificates are at
    Q_W where ten events exist and otherwise at FALLBACK_Q_W, drawn open. The
    proxy structure has its own event panel (the record intersection differs),
    so naive OLS is the proxy panel's.
    """
    d = _decomp()
    naive = _label(_arm(d, "proxy", "naive OLS").select("pair", "x_name", "y_name", pl.col("beta").alias("naive")))
    cp = _cert(d, "proxy").select("pair", pl.col("beta").alias("cert_proxy"), pl.col("fallback").alias("fb_p"))
    ct = _cert(d, "tributary").select("pair", pl.col("beta").alias("cert_trib"), pl.col("fallback").alias("fb_t"))
    cm = _cert(d, "proxy", "certificate minus W").select(
        "pair", pl.col("beta").alias("cert_minus"), pl.col("fallback").alias("fb_m"))
    cmt = _cert(d, "tributary", "certificate minus W").select(
        "pair", pl.col("beta").alias("cert_minus_t"), pl.col("fallback").alias("fb_mt"))
    mxt = _cert(d, "tributary", "certificate (single max)").select(
        "pair", pl.col("beta").alias("max_trib"), pl.col("fallback").alias("fb_xt"))
    mxp = _cert(d, "proxy", "certificate (single max)").select(
        "pair", pl.col("beta").alias("max_proxy"), pl.col("fallback").alias("fb_xp"))
    br = _arm(d, "proxy", "proximal (two proxies)").select("pair", pl.col("beta").alias("bridge"))
    bw = _arm(d, "proxy", "proximal (P, W)").select("pair", pl.col("beta").alias("bridge_pw"))
    aw = _arm(d, "tributary", "OLS adjusting for W").select("pair", pl.col("beta").alias("adj_w"))
    df = (naive.join(cp, on="pair", how="left").join(ct, on="pair", how="left")
          .join(cm, on="pair", how="left").join(cmt, on="pair", how="left")
          .join(mxt, on="pair", how="left").join(mxp, on="pair", how="left")
          .join(br, on="pair", how="left").join(bw, on="pair", how="left")
          .join(aw, on="pair", how="left").sort("naive"))
    assert df["label"].n_unique() == df.height

    # The summary row the reader can rank methods by (coauthor request): the
    # median over the pairs each method is defined on, drawn below a separator.
    MEDIAN_LABEL = "median over pairs"
    med = {c: float(df[c].drop_nulls().median()) for c in
           ("naive", "cert_trib", "cert_proxy", "cert_minus", "cert_minus_t", "max_trib", "max_proxy",
            "bridge", "bridge_pw", "adj_w")}

    lo, hi = 0.4, 2.3
    fig = go.Figure()
    fig.add_vrect(x0=BAND[0], x1=BAND[1], fillcolor=TRIB, opacity=0.08, line_width=0)
    fig.add_vline(x=1.0, line_dash="dot", line_width=1, opacity=0.6)
    for r in df.iter_rows(named=True):
        xs = [v for v in (r["naive"], r["cert_proxy"], r["cert_trib"]) if v is not None]
        fig.add_trace(go.Scatter(x=[max(min(xs), lo), min(max(xs), hi)], y=[r["label"]] * 2, mode="lines",
                                 line=dict(color=LIGHT, width=1), showlegend=False, hoverinfo="skip"))
    MARKS = (("naive", NAIVE, "circle"), ("cert_trib", TRIB, "circle"), ("cert_proxy", PROXY, "diamond"),
             ("cert_minus", PROXY, "cross"), ("cert_minus_t", TRIB, "cross"),
             ("max_trib", TRIB, "star"), ("max_proxy", PROXY, "star"), ("adj_w", ADJUST, "triangle-up"),
             ("bridge", ADJUST, "square"), ("bridge_pw", ADJUST, "x"))
    for col, color, symbol in MARKS:
        _clip_markers(fig, df[col], df["label"], lo, hi, color, symbol)
        _clip_markers(fig, [med[col]], [MEDIAN_LABEL], lo, hi, color, symbol)
    for name, color, symbol in (
        ("naive OLS", NAIVE, "circle"),
        (f"certified, tributary safeguard W (q_W = {Q_W})", TRIB, "circle"),
        (f"certified, pure-proxy safeguard P (q_W = {Q_W})", PROXY, "diamond"),
        (f"certified P, minus W (q_W = {Q_W})", PROXY, "cross"),
        (f"certified W, minus W (q_W = {Q_W})", TRIB, "cross"),
        (f"single max, safeguard W (q_W = {Q_W})", TRIB, "star"),
        (f"single max, safeguard P (q_W = {Q_W})", PROXY, "star"),
        ("OLS adjusting for W", ADJUST, "triangle-up"),
        ("proximal (P1, P2) bridge", ADJUST, "square"),
        ("proximal (P, W) bridge", ADJUST, "x"),
    ):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(color=color, size=MARKER_SIZE + 2, symbol=symbol)))
    fig.add_hline(y=0.5, line_width=0.7, line_color=GREY, opacity=0.7)
    fig.update_yaxes(dtick=1, categoryorder="array",
                     categoryarray=[MEDIAN_LABEL, *df["label"].to_list()])
    fig.update_xaxes(title=f"beta hat (axis clipped to [{lo}, {hi}])", range=[lo, hi])
    apply_layout(fig, 675, 800, legend_y=-0.08)
    fig.update_layout(margin=dict(b=120))
    save(fig, OUTPUT_DIR / "app_proxy")


# ------------------------------------------------------------- headline


def headline() -> None:
    """Headline pair. (a) The certificate along the safeguard grid for both
    structures, n_cert at each point; naive OLS and the two adjustments for
    reference. (b) Every method at q_W = Q_W with 90% year-level bootstrap
    intervals (A2's band is the 5-95% range over permutations)."""
    from hydro.experiments.candidates import headline_row

    pair = headline_row()["pair"]
    d = _decomp().filter(pl.col("pair") == pair)
    e1 = pl.read_parquet(OUTPUT_DIR / "e1_results.parquet").filter(pl.col("pair") == pair)
    area_ratio = float(d["reference_area_ratio"][0])

    def val(structure, method, q_w=None):
        a = _arm(d, structure, method, q_w)
        return a.to_dicts()[0] if a.height else None

    # (a) -- its own PDF; the two panels are placed side by side in LaTeX,
    # because the long method labels of (b) do not fit next to (a).
    fig = go.Figure()
    cats = [str(q) for q in Q_W_GRID]
    fig.add_hrect(y0=BAND[0], y1=BAND[1], fillcolor=TRIB, opacity=0.08, line_width=0)
    fig.add_hline(y=1.0, line_dash="dot", line_width=1, opacity=0.6)
    fig.add_hline(y=area_ratio, line_dash="dot", line_width=1, line_color=GREY)
    for structure, color, symbol, tpos in (("tributary", TRIB, "circle", "top center"),
                                           ("proxy", PROXY, "diamond", "bottom center")):
        pts = [val(structure, "certificate", q) for q in Q_W_GRID]
        xs = [c for c, p in zip(cats, pts) if p]
        fig.add_trace(go.Scatter(
            x=xs, y=[p["beta"] for p in pts if p], mode="lines+markers+text",
            text=[f"n={p['n_cert']}" for p in pts if p], textposition=tpos,
            textfont=dict(size=10, color=color), showlegend=False,
            line=dict(color=color, width=LINE_WIDTH), marker=dict(color=color, size=MARKER_SIZE, symbol=symbol),
        ))
    for method, color, dash in (("naive OLS", NAIVE, "dash"), ("OLS adjusting for W", ADJUST, "dot"),
                                ("OLS subtracting W", ADJUST, "dashdot")):
        v = val("tributary", method)
        fig.add_hline(y=v["beta"], line_dash=dash, line_width=LINE_WIDTH, line_color=color)
    fig.update_xaxes(title="safeguard level q_W", type="category", categoryorder="array", categoryarray=cats)
    fig.update_yaxes(title="beta hat", range=[0.75, 2.5])
    for name, color, dash, symbol in (
        ("certified, tributary W", TRIB, "solid", "circle"), ("certified, pure proxy P", PROXY, "solid", "diamond"),
        ("naive OLS", NAIVE, "dash", None), ("OLS adjusting for W", ADJUST, "dot", None),
        ("OLS subtracting W", ADJUST, "dashdot", None), ("drainage-area ratio", GREY, "dot", None),
    ):
        fig.add_trace(legend_entry(name, color, dash))
        if symbol:
            fig.data[-1].update(mode="lines+markers", marker=dict(color=color, size=MARKER_SIZE, symbol=symbol))
    apply_layout(fig, 675, 480, legend_y=-0.18)
    save(fig, OUTPUT_DIR / "app_headline_sweep")

    # (b)
    fig = go.Figure()
    rows = [  # (label, structure, method, q_w, color, source)
        (f"certificate, tributary W", "tributary", "certificate", Q_W, TRIB, d),
        (f"certificate, pure proxy P", "proxy", "certificate", Q_W, PROXY, d),
        ("naive OLS", "tributary", "naive OLS", None, NAIVE, d),
        ("OLS adjusting for W", "tributary", "OLS adjusting for W", None, ADJUST, d),
        ("OLS subtracting W", "tributary", "OLS subtracting W", None, ADJUST, d),
        ("OLS adjusting for P", "proxy", "OLS adjusting for W", None, ADJUST, d),
        ("2SLS, headwater Z", "tributary", "2SLS (headwater Z)", None, NAIVE, d),
        ("proximal (Z, W): linear bridge", "tributary", "proximal (linear bridge)", None, ADJUST, d),
        ("proximal (P1, P2): linear bridge", "proxy", "proximal (two proxies)", None, ADJUST, d),
        ("proximal (P, W): linear bridge", "proxy", "proximal (P, W)", None, ADJUST, d),
        ("certificate W, minus W", "tributary", "certificate minus W", Q_W, TRIB, d),
        ("certificate P, minus W", "proxy", "certificate minus W", Q_W, PROXY, d),
        ("certificate W, single max", "tributary", "certificate (single max)", Q_W, TRIB, d),
        ("certificate P, single max", "proxy", "certificate (single max)", Q_W, PROXY, d),
        ("A1: tail only, no safeguard", "tributary", "A1 tail only (no safeguard)", Q_W, GREY, d),
        ("A2: permuted safeguard W", None, "A2 permuted safeguard", Q_W, GREY, e1),
        ("A3: safeguard from another basin", None, "A3 irrelevant safeguard", Q_W, GREY, e1),
    ]
    lo, hi = 0.0, 3.0
    order, clipped = [], []
    for label, structure, method, q_w, color, src in rows:
        def rows_at(q):
            f = (pl.col("method") == method) & pl.col("beta").is_not_null()
            if structure:
                f = f & (pl.col("structure") == structure)
            if q is not None:
                f = f & (pl.col("safeguard_quantile") == q)
            if src is e1:
                f = f & (pl.col("estimator_type") == "ols")
            return src.filter(f)
        a = rows_at(q_w)
        # A certified method that does not reach ten events at the tabulated
        # level is shown at the fallback level, labelled.
        if a.height == 0 and q_w == Q_W and method.startswith("certificate"):
            a = rows_at(FALLBACK_Q_W)
            label = f"{label} (q_W = {FALLBACK_Q_W})"
        if a.height == 0:
            continue
        r = a.to_dicts()[0]
        order.append(label)
        b, cl, ch = r["beta"], r.get("ci_lo"), r.get("ci_hi")
        if not lo <= b <= hi or (cl is not None and (cl < lo or ch > hi)):
            clipped.append((label, b, cl, ch))
        fig.add_trace(go.Scatter(
            x=[min(max(b, lo), hi)], y=[label], mode="markers", showlegend=False,
            marker=dict(color=color, size=MARKER_SIZE + 3, symbol="circle" if lo <= b <= hi else "circle-open",
                        line=dict(width=1.5, color=color)),
            error_x=dict(type="data", symmetric=False, width=4, thickness=1.5, color=color,
                         array=[max(min(ch, hi) - b, 0) if ch is not None else 0],
                         arrayminus=[max(b - max(cl, lo), 0) if cl is not None else 0]),
        ))
    fig.add_vrect(x0=BAND[0], x1=BAND[1], fillcolor=TRIB, opacity=0.08, line_width=0)
    fig.add_vline(x=1.0, line_dash="dot", line_width=1, opacity=0.6)
    fig.add_vline(x=area_ratio, line_dash="dot", line_width=1, line_color=GREY)
    fig.update_yaxes(dtick=1, categoryorder="array", categoryarray=order[::-1])
    fig.update_xaxes(title=f"beta hat (axis clipped to [{lo:g}, {hi:g}])", range=[lo, hi])
    apply_layout(fig, 675, 480, legend_y=-0.2)
    fig.update_layout(margin=dict(b=60))
    save(fig, OUTPUT_DIR / "app_headline_methods")
    if clipped:
        print("  clipped in (b):", ", ".join(f"{l}: {b:.2f} [{cl}, {ch}]" for l, b, cl, ch in clipped))


# ------------------------------------------------- appendix + placebos


def threshold_stability() -> None:
    """Appendix: beta vs the treatment threshold v on the headline pair, one
    line per safeguard level, blues darkening as the safeguard tightens."""
    stab = (
        pl.read_parquet(OUTPUT_DIR / "e1_stability.parquet")
        .filter((pl.col("estimator_type") == "ols") & (pl.col("n_cert") >= 10)
                & pl.col("beta").is_not_null())
        .sort("threshold_v")
    )
    e1 = pl.read_parquet(OUTPUT_DIR / "e1_results.parquet")
    naive = float(e1.filter(pl.col("method") == "naive OLS")["beta"].item())
    area_ratio = float(e1["reference_area_ratio"].drop_nulls()[0])

    blues = {0.9: "#b3cde8", 0.75: "#7fa8d9", 0.5: "#4a80c4", 0.25: "#123f7e"}
    fig = go.Figure()
    fig.add_hrect(y0=BAND[0], y1=BAND[1], fillcolor=TRIB, opacity=0.08, line_width=0)
    fig.add_hline(y=1.0, line_dash="dot", line_width=1, opacity=0.6)
    fig.add_hline(y=naive, line_dash="dash", line_width=LINE_WIDTH, line_color=NAIVE, opacity=0.9)
    fig.add_hline(y=area_ratio, line_dash="dot", line_width=1, line_color=GREY, opacity=0.9)
    for q_w, color in blues.items():
        sub = stab.filter(pl.col("safeguard_quantile") == q_w)
        if sub.height == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["threshold_v"], y=sub["beta"], mode="lines+markers",
            name=f"q_W = {q_w}", line=dict(color=color, width=LINE_WIDTH),
            marker=dict(color=color, size=MARKER_SIZE),
        ))
    fig.add_trace(legend_entry("naive OLS", NAIVE, dash="dash"))
    fig.update_xaxes(title="treatment threshold v (m3/s)")
    fig.update_yaxes(title="beta hat", range=[0.75, 2.5])
    apply_layout(fig, 675, 480, legend_y=-0.18)
    save(fig, OUTPUT_DIR / "app_threshold_stability")


def _dumbbell_rows(fig, df, y_col, naive_col, cert_col, row=None, col=None, tick_col=None) -> None:
    kw = {} if row is None else {"row": row, "col": col}
    for r in df.iter_rows(named=True):
        fig.add_trace(go.Scatter(x=[r[naive_col], r[cert_col]], y=[r[y_col]] * 2, mode="lines",
                                 line=dict(color=LIGHT, width=1), showlegend=False, hoverinfo="skip"), **kw)
    if tick_col is not None:
        fig.add_trace(go.Scatter(x=df[tick_col], y=df[y_col], mode="markers", showlegend=False,
                                 marker=dict(color=GREY, symbol="line-ns-open", size=11,
                                             line=dict(width=1.5, color=GREY))), **kw)
    fig.add_trace(go.Scatter(x=df[naive_col], y=df[y_col], mode="markers", showlegend=False,
                             marker=dict(color=NAIVE, size=MARKER_SIZE + 2)), **kw)
    fig.add_trace(go.Scatter(x=df[cert_col], y=df[y_col], mode="markers", showlegend=False,
                             marker=dict(color=TRIB, size=MARKER_SIZE + 2)), **kw)


def placebos() -> None:
    """(a) disconnected pairs, (b) reverse direction; linear axis clipped to
    [0, 2.2] (naive estimates in (a) reach past 20; clipped values are open at
    the edge and the parquet keeps them). The grey tick is the arithmetic null
    floor median(daily Y)/mean(certified X): where a null estimate would land,
    discharge being non-negative.
    """
    res = pl.read_parquet(OUTPUT_DIR / "results.parquet")

    def arms(exp: str, rule: str) -> pl.DataFrame:
        d = res.filter((pl.col("experiment") == exp) & (pl.col("safeguard_rule") == rule))
        naive = d.filter(pl.col("method") == "naive OLS").select(
            "pair", "x_name", "y_name", pl.col("beta").alias("naive"))
        cert = d.filter(
            (pl.col("method") == "certificate") & (pl.col("estimator_type") == "ols")
            & (pl.col("safeguard_quantile") == Q_W) & pl.col("beta").is_not_null()
        ).select("pair", pl.col("beta").alias("cert"), "beta_null_floor", "n_cert")
        return _label(naive.join(cert, on="pair", how="inner"), sep=" × ").sort("naive")

    e2, e3 = arms("E2", "nearest"), arms("E3", "largest")
    e3 = e3.with_columns((pl.col("label").str.replace(" × ", " → ")).alias("label"))
    fig = make_subplots(
        rows=2, cols=1, row_heights=[e2.height, e3.height], vertical_spacing=0.09,
        subplot_titles=("(a) disconnected pairs (true beta = 0)",
                        "(b) reverse direction (true beta = 0)"),
    )
    lo, hi = 0.0, 2.2  # naive estimates reach past 20 on one pair; clipped open at the edge
    for r, d in ((1, e2), (2, e3)):
        for row_ in d.iter_rows(named=True):
            fig.add_trace(go.Scatter(
                x=[min(row_["naive"], hi), min(row_["cert"], hi)], y=[row_["label"]] * 2, mode="lines",
                line=dict(color=LIGHT, width=1), showlegend=False, hoverinfo="skip"), row=r, col=1)
        fig.add_trace(go.Scatter(x=d["beta_null_floor"], y=d["label"], mode="markers", showlegend=False,
                                 marker=dict(color=GREY, symbol="line-ns-open", size=11,
                                             line=dict(width=1.5, color=GREY))), row=r, col=1)
        _clip_markers(fig, d["naive"], d["label"], lo, hi, NAIVE, "circle", row=r, col=1)
        _clip_markers(fig, d["cert"], d["label"], lo, hi, TRIB, "circle", row=r, col=1)
        fig.update_yaxes(dtick=1, categoryorder="array", categoryarray=d["label"].to_list(), row=r, col=1)
        fig.update_xaxes(range=[-0.06, hi], dtick=0.5, row=r, col=1)
        fig.add_vline(x=0.0, line_dash="dot", line_width=1, opacity=0.6, row=r, col=1)
        fig.add_vline(x=1.0, line_dash="dot", line_width=1, line_color=GREY, opacity=0.5, row=r, col=1)
    fig.update_xaxes(title=f"beta hat (axis clipped to [0, {hi}])", row=2, col=1)
    fig.add_traces([
        legend_entry("naive OLS", NAIVE),
        legend_entry(f"certified (q_W = {Q_W})", TRIB),
        legend_entry("arithmetic null floor", GREY),
    ])
    for tr in fig.data[-3:]:
        tr.update(mode="markers", marker=dict(color=tr.line.color, size=MARKER_SIZE + 2,
                                              symbol="line-ns-open" if "floor" in tr.name else "circle"))
    apply_layout(fig, 675, 40 * (e2.height + e3.height) + 160, legend_y=-0.075)
    fig.update_layout(margin=dict(b=130))
    save(fig, OUTPUT_DIR / "app_placebos")


def main() -> None:
    geometry_map()
    decomposition_all_pairs()
    proxy_vs_tributary()
    headline()
    threshold_stability()
    placebos()


if __name__ == "__main__":
    main()
