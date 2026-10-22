"""The pgfplots figures of the application (Section 7 and its appendix), as
bare `\\begin{tikzpicture} ... \\end{tikzpicture}` fragments that the paper
`\\input`s.

    uv run python -m hydro.pipeline.tikz

    app_map.tex                  the headline pair on its gauge network, with
                                 the causal structure as an inset
    app_all_pairs.tex            all pairs: one kernel density per estimator
    app_headline_sweep.tex       the headline pair: certified OLS along q_P
    app_headline_methods.tex     the headline pair: the estimators and the
                                 ablations V1-V3 at the tabulated q_P with 90%
                                 year-bootstrap intervals
    app_placebos.tex             the disconnected-pair placebo, as app_all_pairs
    app_threshold_stability.tex  the headline pair: certified OLS against v,
                                 one line per q_P

Reads pairs.parquet, all_pairs.parquet, placebo.parquet and headline.parquet
from hydro/results/ and the gauge network metadata; writes to hydro/results/
and, when PAPER_DIR is set, copies to the paper's Figures/application/.
Colours and axis styling come from `tikz_style`.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from hydro.config import CONFIG, OUTPUT_DIR
from hydro.pairs import headline_pair
from hydro.pipeline.pair_table import town
from hydro.preprocess.screening import region_network
from tikz_style import AXIS_STYLE, COLOR_DEFINITIONS, coordinates, write_fragment

EST = CONFIG.estimator
Q_P = EST.q_p
GENERATOR = "hydro/pipeline/tikz.py"
# The two panels of the headline figure share this plot-area height so they align.
HEADLINE_AXIS_HEIGHT = "0.36\\linewidth"
# Vertical room of the map around its special gauges, in cm: the causal-structure
# inset sits in the top margin, the P1 label in the bottom one. The all-pairs ridge
# beside the map takes its height from the map (`map_height`).
MAP_PAD_TOP_CM = 1.6
MAP_PAD_BOTTOM_CM = 0.85
# Ridgelines: Gaussian kernel densities with bandwidth KDE_FACTOR times
# Silverman's 0.9 min(sd, IQR/1.34) n^(-1/5), the pairs as jittered dots.
KDE_FACTOR = 0.675
JITTER_SEED = 20260902
# One blue per safeguard level of the threshold figure, darker as it tightens.
THRESHOLD_BLUES = {0.9: "#c4d7ec", 0.75: "#8fb3dd", 0.5: "#5a8bcb", 0.25: "#2b5fa8", 0.1: "#123f7e",
                   0.05: "#061d3d"}


def _write(name: str, body: str) -> None:
    write_fragment(OUTPUT_DIR / name, body, GENERATOR, "Figures/application")


def _f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def arm(d: pl.DataFrame, method: str, q_p: float | None = None) -> pl.DataFrame:
    f = (pl.col("method") == method) & pl.col("beta").is_not_null()
    if q_p is not None:
        f = f & (pl.col("q_p") == q_p)
    return d.filter(f)


# ------------------------------------------------------------------ map


def _dag_inset(ox: float, oy: float) -> list[str]:
    """The causal structure of the design as a small DAG with its top-centre
    node U at (ox, oy): U drives T, Y (unobserved inflow) and the proxies P1,
    P2, laid out as the double-proxy graph of the paper but without the
    edges P1 -> T and P2 -> Y."""
    dx, dy = 0.72, 0.72
    nodes = {"U": (0, 0), "P1": (-dx, 0), "P2": (dx, 0), "T": (-dx / 2, -dy), "Y": (dx / 2, -dy)}
    lines = [f"\\begin{{scope}}[shift={{({_f(ox)},{_f(oy)})}}, every node/.style={{draw, circle, minimum size=3.2mm, "
             "inner sep=0.5pt, font=\\scriptsize, fill=white}]",
             f"\\fill[white, fill opacity=0.85, rounded corners=2pt] ({_f(-dx - 0.3)},{_f(-dy - 0.25)}) "
             f"rectangle ({_f(dx + 0.3)},{_f(0.25)});"]
    for name, (x, y) in nodes.items():
        label = {"P1": "$P_1$", "P2": "$P_2$"}.get(name, f"${name}$")
        style = "dashed" if name == "U" else ""
        lines.append(f"\\node[{style}] ({name}) at ({_f(x)},{_f(y)}) {{{label}}};")
    for a, b in (("U", "P1"), ("U", "P2"), ("U", "T"), ("U", "Y")):
        lines.append(f"\\draw[-{{Stealth[length=1.4mm]}}, line width=0.5pt] ({a}) -- ({b});")
    lines.append("\\draw[-{Stealth[length=1.4mm]}, line width=0.9pt] (T) -- node[draw=none, fill=none, below, "
                 "font=\\tiny, inner sep=1pt] {$\\beta_0$} (Y);")
    lines.append("\\end{scope}")
    return lines


def _map_frame(width_cm: float = 7.4) -> dict:
    """The headline pair's network and the crop window of its map: the
    special gauges plus a margin, latitude scaled by 1/cos(mean latitude);
    W and H are the width and height of the map in cm."""
    row = headline_pair()
    net = region_network(row["region"])
    proxies = row["proxies"].split(",")
    pos = {r["site_no"]: (r["lon"], r["lat"]) for r in net.meta.iter_rows(named=True)}
    names = {r["site_no"]: r["station_nm"] for r in net.meta.iter_rows(named=True)}
    special = [row["t"], row["y"], *proxies]
    pad = 0.35  # degrees of longitude either side of the special gauges
    lo_x, hi_x = min(pos[g][0] for g in special) - pad, max(pos[g][0] for g in special) + pad
    lat0 = sum(p[1] for p in pos.values()) / len(pos)
    sx = width_cm / (hi_x - lo_x)
    sy = sx / math.cos(math.radians(lat0))
    lo_y = min(pos[g][1] for g in special) - MAP_PAD_BOTTOM_CM / sy
    hi_y = max(pos[g][1] for g in special) + MAP_PAD_TOP_CM / sy
    return dict(row=row, net=net, proxies=proxies, pos=pos, names=names, lo_x=lo_x, hi_x=hi_x,
                lo_y=lo_y, hi_y=hi_y, sx=sx, sy=sy, W=width_cm, H=(hi_y - lo_y) * sy)


def map_height(width_cm: float = 7.4) -> float:
    """Height in cm of the map picture's clipped area, for the figure beside it."""
    return _map_frame(width_cm)["H"]


def geometry_map(width_cm: float = 7.4) -> None:
    """Every gauge of the basin is joined to the next gauge downstream by a
    light line, the main stem between T and Y is black and thick, T and Y are
    black squares, the proxies blue diamonds; flow arrows sit on the stem
    edges and on the two edges downstream of Y. Cropped to the special gauges
    plus a margin, latitude scaled by 1/cos(mean latitude). The causal
    structure sits as an inset in the top-left corner."""
    f = _map_frame(width_cm)
    row, net, proxies, pos, names = f["row"], f["net"], f["proxies"], f["pos"], f["names"]
    lo_x, hi_x, lo_y, hi_y, sx, sy, W, H = (f[k] for k in ("lo_x", "hi_x", "lo_y", "hi_y", "sx", "sy", "W", "H"))

    def parent(g: str) -> str | None:
        down = [d for d, up in net.upstream.items() if g in up and d != g and d in pos]
        return min(down, key=lambda d: len(net.upstream[d])) if down else None

    on_stem = {g for g in pos if net.is_upstream_of(row["t"], g) or g == row["t"]} & (net.upstream[row["y"]] | {row["y"]})
    special = [row["t"], row["y"], *proxies]

    def xy(g: str) -> tuple[float, float]:
        return (pos[g][0] - lo_x) * sx, (pos[g][1] - lo_y) * sy

    def inside(g: str) -> bool:
        return lo_x <= pos[g][0] <= hi_x and lo_y <= pos[g][1] <= hi_y

    arrow_edges = [(g, parent(g)) for g in on_stem if parent(g) is not None and parent(g) in on_stem]
    g = row["y"]
    for _ in range(2):
        pa = parent(g)
        if pa is None:
            break
        arrow_edges.append((g, pa))
        g = pa

    lines = ["\\begin{tikzpicture}[x=1cm, y=1cm]", COLOR_DEFINITIONS.rstrip(),
             f"\\begin{{scope}}\n\\clip (0,0) rectangle ({_f(W)},{_f(H)});"]
    for g in pos:
        pa = parent(g)
        if pa is None or not (inside(g) or inside(pa)):
            continue
        stem = g in on_stem and pa in on_stem
        (x0, y0), (x1, y1) = xy(g), xy(pa)
        style = "black, line width=1.2pt" if stem else "applight, line width=0.45pt"
        lines.append(f"\\draw[{style}] ({_f(x0)},{_f(y0)}) -- ({_f(x1)},{_f(y1)});")
    for g, pa in arrow_edges:
        (x0, y0), (x1, y1) = xy(g), xy(pa)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        stem = g in on_stem and pa in on_stem
        eps, head = (0.09, "length=1.7mm, width=1.4mm") if stem else (0.12, "length=2.2mm, width=1.9mm")
        color = "black" if stem else "appgrey"
        lines.append(f"\\draw[{color}, line width=0.6pt, -{{Stealth[{head}]}}] "
                     f"({_f(mx - eps * ux)},{_f(my - eps * uy)}) -- ({_f(mx + eps * ux)},{_f(my + eps * uy)});")
    for g in pos:
        if g not in special and inside(g):
            x, y = xy(g)
            lines.append(f"\\fill[appgrey] ({_f(x)},{_f(y)}) circle (0.5mm);")
    lines.append("\\end{scope}")
    lines.extend(_dag_inset(1.35, H - 0.32))

    label_pos = {
        "top left": ("south east", -1.3, 1.3), "middle right": ("west", 1.7, 0.0),
        "middle left": ("east", -1.7, 0.0),
        "bottom left": ("north east", -1.3, -1.3), "bottom right": ("north west", 1.3, -1.3),
        "top right": ("north west", 1.3, 1.3),
    }
    sq, dm = 1.1, 1.5  # half side of the squares / half diagonal of the diamonds, mm

    def square(x, y, color):
        return f"\\fill[{color}] ({_f(x)},{_f(y)}) ++(-{sq}mm,-{sq}mm) rectangle ++({2 * sq}mm,{2 * sq}mm);"

    def diamond(x, y, color):
        return (f"\\fill[{color}] ({_f(x)},{_f(y)}) ++(0,-{dm}mm) -- ++({dm}mm,{dm}mm) -- ++(-{dm}mm,{dm}mm) "
                f"-- ++(-{dm}mm,-{dm}mm) -- cycle;")

    def label(x, y, text, side):
        anchor, ddx, ddy = label_pos[side]
        return (f"\\node[anchor={anchor}, font=\\scriptsize, inner sep=1pt] "
                f"at ($({_f(x)},{_f(y)})+({ddx}mm,{ddy}mm)$) {{{text}}};")

    groups = [(row["t"], "T", "black", square, "middle left"), (row["y"], "Y", "black", square, "middle right")]
    p_sides = ["bottom left", "top right", "bottom right"]
    for i, g in enumerate(proxies):
        groups.append((g, f"P{i + 1}", "appblue", diamond, p_sides[i % len(p_sides)]))
    for g, tag, color, marker, side in groups:
        x, y = xy(g)
        lines.append(marker(x, y, color))
        lines.append(label(x, y, f"{tag}: {town(names[g])}", side))
    lines.append(f"\\coordinate (leg) at ({_f(W / 2)},-0.400);")
    lines.append(
        "\\node[anchor=center, inner sep=0pt, font=\\footnotesize] at (leg) {%\n"
        "  \\begin{tabular}{@{}c@{\\,}l@{\\hspace{3.2mm}}c@{\\,}l@{\\hspace{3.2mm}}c@{\\,}l@{}}\n"
        f"  \\tikz\\fill[black] (0,0) rectangle ({2 * sq}mm,{2 * sq}mm); & $T$ (treatment) &\n"
        f"  \\tikz\\fill[black] (0,0) rectangle ({2 * sq}mm,{2 * sq}mm); & $Y$ (outcome) &\n"
        f"  \\tikz\\fill[appblue] (0,0) -- ({dm}mm,{dm}mm) -- (0,{2 * dm}mm) -- (-{dm}mm,{dm}mm) -- cycle; "
        "& $P$: proxy safeguard\n  \\end{tabular}};")
    lines.append("\\end{tikzpicture}")
    _write("app_map.tex", "\n".join(lines) + "\n")
    print("  map: T", town(names[row["t"]]), "Y", town(names[row["y"]]), "P", [town(names[g]) for g in proxies])


# ------------------------------------------------------------- headline


def _target(sym: str = "") -> str:
    y = lambda v: f"{{axis cs:{sym}{v}}}"  # noqa: E731
    return (f"\\draw[densely dotted, black!60, line width=0.6pt] ({{rel axis cs:0,0}}|-{y(CONFIG.beta_target)}) -- ({{rel axis cs:1,0}}|-{y(CONFIG.beta_target)});\n")


def headline_sweep() -> None:
    """Certified OLS along the safeguard grid, the number of certified days
    next to each point; dashed line at naive OLS, dotted line at the target."""
    pair = headline_pair()["pair"]
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet").filter(pl.col("pair") == pair)

    def at(method: str, q: float) -> dict | None:
        a = d.filter((pl.col("method") == method) & (pl.col("q_p") == q) & pl.col("beta").is_not_null())
        return a.to_dicts()[0] if a.height else None

    naive = float(d.filter(pl.col("method") == "naive OLS")["beta"][0])
    cats = ",".join(str(q) for q in EST.q_p_grid)
    sym0 = str(EST.q_p_grid[0])
    ymin, ymax = 0.6, 2.4

    def table(method: str) -> str:
        rows = []
        for q in EST.q_p_grid:
            r = at(method, q)
            if r is not None:
                rows.append(f"{q} {r['beta']:.4f} {int(r['n_cert'])} south \\\\")
                print(f"  {method} q_P={q}: {r['beta']:.3f} (n={int(r['n_cert'])}, q_T={r['q_t']})")
        return "\n".join(rows)

    print(f"  naive OLS {naive:.3f}")
    body = f"""\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}\\pgfmathsetlengthmacro{{\\appaxw}}{{\\linewidth-1.2cm}}
\\begin{{axis}}[
  scale only axis, width=\\appaxw, height={HEADLINE_AXIS_HEIGHT},
  symbolic x coords={{{cats}}}, xtick={{{cats}}}, enlarge x limits=0.1,
  xlabel={{safeguard level $q_P$}}, ylabel={{$\\hat\\beta$}},
  ymin={ymin}, ymax={ymax}, ytick={{0.8,1.2,...,2.4}},
  ymajorgrids, grid style={{applight!50}},
  axis line style={{applight}}, tick style={{applight}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\small}},
  ylabel style={{yshift=-2pt}}, xlabel style={{yshift=2pt}},
  clip=false,
]
{_target(sym=f"{sym0},")}\\draw[dashed, apporange, line width=1.2pt] ({{rel axis cs:0,0}}|-{{axis cs:{sym0},{naive:.4f}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:{sym0},{naive:.4f}}});
\\node[anchor=south east, font=\\scriptsize, apporange, inner sep=1pt] at ({{rel axis cs:1,0}}|-{{axis cs:{sym0},{naive:.4f}}}) {{naive OLS}};
\\node[anchor=north east, font=\\scriptsize, black!60, inner sep=1pt] at ({{rel axis cs:1,0}}|-{{axis cs:{sym0},{CONFIG.beta_target}}}) {{target}};
\\addplot[appblue, line width=1pt, mark=*, mark size=2pt,
  nodes near coords={{$n{{=}}\\pgfplotspointmeta$}}, point meta=explicit symbolic,
  visualization depends on={{value \\thisrow{{anchor}} \\as \\appanchor}},
  every node near coord/.style={{font=\\scriptsize, color=appblue, anchor=\\appanchor, inner sep=1.5pt}}]
  table[x=q, y=beta, meta=n, row sep=\\\\] {{
  q beta n anchor \\\\
{table("certified OLS")}
  }};
\\end{{axis}}
\\end{{tikzpicture}}
"""
    _write("app_headline_sweep.tex", body)


def headline_methods() -> None:
    """Forest of the estimators (all_pairs.parquet) and the ablations V1-V3
    (headline.parquet) at the tabulated q_P with their 90% year-bootstrap
    intervals; x axis clipped to [0, 3]. Raised a little so that it aligns
    with the sweep beside it."""
    pair = headline_pair()["pair"]
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet").filter(pl.col("pair") == pair)
    h = pl.read_parquet(OUTPUT_DIR / "headline.parquet")
    spec = [
        ("certified OLS", "certified OLS", Q_P, "appblue", d),
        ("naive OLS", "naive OLS", None, "apporange", d),
        ("proximal $(P_1, P_2)$ bridge", "proximal bridge", None, "appviolet", d),
        ("V1: tail only, no safeguard", "V1 tail only", Q_P, "appgrey", h),
        ("V2: permuted safeguard $P$", "V2 permuted safeguard", Q_P, "appgrey", h),
        ("V3: other-basin safeguard", "V3 other-basin safeguard", Q_P, "appgrey", h),
    ]
    lo, hi = 0.0, 3.0
    rows = []
    for label, method, q, color, src in spec:
        f = (pl.col("method") == method) & pl.col("beta").is_not_null()
        if q is not None:
            f = f & (pl.col("q_p") == q)
        a = src.filter(f)
        if a.height == 0:
            print(f"  no row for {method}; skipped")
            continue
        r = a.to_dicts()[0]
        rows.append((label, float(r["beta"]), r.get("ci_lo"), r.get("ci_hi"), color))
        ci = "" if r.get("ci_lo") is None else f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"  {label}: {r['beta']:.3f}{ci}")

    labels = ",".join("{" + lab + "}" for lab, *_ in rows)
    plots = []
    for i, (label, b, cl, ch, color) in enumerate(rows, start=1):
        outside = not lo <= b <= hi
        bx = min(max(b, lo), hi)
        mark = "o" if outside else "*"
        if cl is not None and ch is not None:
            em, ep = max(bx - max(cl, lo), 0.0), max(min(ch, hi) - bx, 0.0)
            plots.append(
                f"\\addplot[{color}, only marks, mark={mark}, mark size=2.8pt, line width=0.9pt,\n"
                f"  error bars/.cd, x dir=both, x explicit, error bar style={{line width=0.9pt}},\n"
                f"  error mark options={{rotate=90, mark size=2.2pt, line width=0.9pt}}]\n"
                f"  coordinates {{({bx:.4f},{i}) -= ({em:.4f},0) += ({ep:.4f},0)}};")
        else:
            plots.append(f"\\addplot[{color}, only marks, mark={mark}, mark size=2.8pt, line width=0.9pt] "
                         f"coordinates {{({bx:.4f},{i})}};")
    n = len(rows)
    body = f"""\\raisebox{{5mm}}{{\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}\\pgfmathsetlengthmacro{{\\appaxw}}{{\\linewidth-3.4cm}}
\\begin{{axis}}[
  scale only axis, width=\\appaxw, height={HEADLINE_AXIS_HEIGHT},
  xmin={lo:g}, xmax={hi:g}, xtick={{0,1,2,3}},
  xlabel={{$\\hat\\beta$}}, xlabel style={{yshift=2pt}},
  ymin=0.4, ymax={n + 0.6}, y dir=reverse,
  ytick={{1,...,{n}}}, yticklabels={{{labels}}},
  xmajorgrids, grid style={{applight!50}},
  axis line style={{applight}}, tick style={{applight}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\small}},
  y tick label style={{font=\\scriptsize}},
  clip=false,
]
\\draw[densely dotted, black!60, line width=0.6pt] ({{axis cs:1,0}}|-{{rel axis cs:0,0}}) -- ({{axis cs:1,0}}|-{{rel axis cs:0,1}});
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}}}
"""
    _write("app_headline_methods.tex", body)


# ----------------------------------------------------------- ridgelines


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


def ridge_geometry(rows: list[tuple[str, list[float], str]], lo: float, hi: float,
                   peak: float = 0.85) -> list[dict]:
    """What the ridgelines are drawn from, one dict per estimator: the density
    `height` over `grid` on a common scale (`peak` row units at the highest
    mode), the clipped quartiles, the jittered pair dots as (value, height
    above the baseline) split into `inside` the window and `clipped` to it,
    the median with the density height there, and the number of clipped pairs."""
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


def _rows(d: pl.DataFrame) -> list[tuple[str, list[float], str]]:
    """(label, estimates over pairs, colour) per estimator; labels are TeX."""
    return [
        ("naive OLS", arm(d, "naive OLS")["beta"].to_list(), "apporange"),
        ("proximal\\\\$(P_1, P_2)$ bridge", arm(d, "proximal bridge")["beta"].to_list(), "appviolet"),
        ("certified OLS", arm(d, "certified OLS", Q_P)["beta"].to_list(), "appblue"),
    ]


def _ridge(rows: list[tuple[str, list[float], str]], lo: float, hi: float, target: float,
           axis_width: str, axis_height: str) -> str:
    """One ridgeline per estimator: a baseline, the density filled lightly with
    the interquartile range filled more strongly on top, the pairs as faint
    dots (open where clipped to the window), the median as a thick stroke and
    the clipped count at the right edge; the dotted target across the axis.
    `axis_width` and `axis_height` are lengths of the plot area."""
    geo = ridge_geometry(rows, lo, hi)
    n = len(geo)
    labels = ",".join("{" + g["label"] + "}" for g in geo)
    lines = [f"""\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}\\begin{{axis}}[
  scale only axis, width={axis_width}, height={axis_height},
  xmin={lo - 0.02:g}, xmax={hi + 0.02:g}, xtick={{{','.join(str(k) for k in range(int(lo), int(hi) + 1))}}},
  ymin={-(n - 1) - 0.6:g}, ymax=1, ytick={{{','.join(str(-k) for k in range(n))}}}, yticklabels={{{labels}}},
  y tick label style={{align=right, font=\\scriptsize}}, ytick style={{draw=none}},
  xmajorgrids, {AXIS_STYLE},
  xlabel={{$\\hat\\beta$ (values clipped to $[{lo:g}, {hi:g}]$)}},
  clip=false,
]
\\draw[densely dotted, black!60, line width=0.6pt] ({{axis cs:{target:g},0}}|-{{rel axis cs:0,0}}) -- ({{axis cs:{target:g},0}}|-{{rel axis cs:0,1}});"""]
    for k, g in enumerate(geo):
        c, base = g["color"], -k
        curve = coordinates(g["grid"], base + g["height"], digits=5)
        lines.append(f"\\draw[{c}, line width=0.3pt] (axis cs:{lo:g},{base}) -- (axis cs:{hi:g},{base});")
        lines.append(f"\\addplot[draw=none, fill={c}, fill opacity=0.22] "
                     f"coordinates {{({lo:g},{base}) {curve} ({hi:g},{base})}} -- cycle;")
        lines.append(f"\\addplot[{c}, line width=0.7pt] coordinates {{{curve}}};")
        inner = (g["grid"] >= g["q25"]) & (g["grid"] <= g["q75"])
        if inner.any():
            iqr = coordinates(g["grid"][inner], base + g["height"][inner], digits=5)
            lines.append(f"\\addplot[draw=none, fill={c}, fill opacity=0.3] "
                         f"coordinates {{({g['q25']:.4f},{base}) {iqr} ({g['q75']:.4f},{base})}} -- cycle;")
        dots = ((g["inside"], "*", f"fill={c}, draw={c}, fill opacity=0.2, draw opacity=0.2"),
                (g["clipped"], "o", f"draw={c}, draw opacity=0.2"))
        for pts, mark, options in dots:
            if pts:
                lines.append(f"\\addplot[only marks, mark={mark}, mark size=1.3pt, mark options={{{options}}}] "
                             f"coordinates {{{coordinates([p[0] for p in pts], [base + p[1] for p in pts], 4)}}};")
        if g["median_height"] > 0:
            lines.append(f"\\draw[{c}, line width=1.5pt] (axis cs:{g['median']:.4f},{base}) -- "
                         f"(axis cs:{g['median']:.4f},{base + g['median_height']:.4f});")
        if g["n_clipped"]:
            lines.append(f"\\node[anchor=south west, font=\\tiny, appgrey, inner sep=1pt, xshift=2pt] "
                         f"at (axis cs:{hi:g},{base}) {{{g['n_clipped']} clipped}};")
        print(f"  {g['label']}: median {g['median']:.3f}, {g['n_clipped']} clipped")
    lines.append("\\end{axis}\n\\end{tikzpicture}\n")
    return "\n".join(lines)


def all_pairs() -> None:
    """Sits next to the map in the paper (a subfigure of 0.42\\textwidth): the
    plot area is the line width less the room for the labels, and as high as
    the map so that the two subfigures align."""
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet")
    body = _ridge(_rows(d), -1.0, 3.0, CONFIG.beta_target, "\\appaxw", f"{map_height() - 0.3:.2f}cm")
    _write("app_all_pairs.tex", "\\pgfmathsetlengthmacro{\\appaxw}{\\linewidth-3cm}\n" + body)


def placebos() -> None:
    """Included at 0.85\\textwidth in the appendix, the plot area in a 7:3 proportion."""
    d = pl.read_parquet(OUTPUT_DIR / "placebo.parquet")
    body = _ridge(_rows(d), -2.0, 2.0, 0.0, "\\appaxw", "\\appaxh")
    _write("app_placebos.tex", "\\pgfmathsetlengthmacro{\\appaxw}{0.85\\textwidth-2.7cm}\n"
           "\\pgfmathsetlengthmacro{\\appaxh}{0.43*\\appaxw}\n" + body)


# ------------------------------------------------------------ threshold


def threshold_stability() -> None:
    """Headline pair: certified OLS against the treatment threshold v, one
    line per safeguard level (darker as it tightens), the naive OLS dashed
    and the target dotted; the legend below in two rows."""
    pair = headline_pair()["pair"]
    stab = (pl.read_parquet(OUTPUT_DIR / "headline.parquet")
            .filter((pl.col("method") == "certified OLS") & (pl.col("n_cert") >= EST.min_events)
                    & pl.col("beta").is_not_null())
            .sort("v"))
    naive = float(arm(pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet").filter(pl.col("pair") == pair),
                      "naive OLS")["beta"][0])
    blues = "".join(f"\\definecolor{{appq{i}}}{{HTML}}{{{c[1:].upper()}}}\n"
                    for i, c in enumerate(THRESHOLD_BLUES.values()))
    plots = []
    for i, q_p in enumerate(THRESHOLD_BLUES):
        sub = stab.filter(pl.col("q_p") == q_p)
        if sub.height:
            plots.append(f"\\addplot[appq{i}, mark=*] coordinates {{{coordinates(sub['v'], sub['beta'], 5)}}};\n"
                         f"\\addlegendentry{{$q_P = {q_p}$}}")
            print(f"  q_P={q_p}: {sub.height} thresholds, beta {sub['beta'].min():.3f}-{sub['beta'].max():.3f}")
    plots.append("\\addlegendimage{apporange, dashed}\\addlegendentry{naive OLS}")
    body = f"""\\begin{{tikzpicture}}
{COLOR_DEFINITIONS}{blues}\\begin{{axis}}[
  width=0.60\\textwidth, height=0.38\\textwidth,
  ymin=0.4, ymax=2.4, ytick={{0.5,1,1.5,2}}, xtick={{20,40,...,100}}, enlarge x limits=0.05,
  xlabel={{treatment threshold $v$ (m$^3$/s)}}, ylabel={{$\\hat\\beta$}},
  xmajorgrids, ymajorgrids, grid style={{applight!50}}, axis line style={{applight}}, tick style={{applight}},
  tick label style={{font=\\scriptsize}}, label style={{font=\\footnotesize}},
  title style={{font=\\footnotesize, yshift=-3pt}},
  every axis plot/.append style={{line width=0.75pt, mark size=1.2pt}},
  legend columns=4, legend style={{at={{(0.5,0)}}, anchor=north, yshift=-0.9cm, draw=none, fill=none, font=\\scriptsize,
  legend cell align=left, /tikz/every even column/.append style={{column sep=0.2cm}}}},
  clip=false,
]
{_target(sym="0,")}\\draw[dashed, apporange, line width=0.75pt] ({{rel axis cs:0,0}}|-{{axis cs:0,{naive:.4f}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:0,{naive:.4f}}});
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    _write("app_threshold_stability.tex", body)
    print(f"  naive OLS {naive:.3f}")


def main() -> None:
    geometry_map()
    headline_sweep()
    headline_methods()
    all_pairs()
    placebos()
    threshold_stability()


if __name__ == "__main__":
    main()
