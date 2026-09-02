"""The per-pair table of the Application appendix, generated from the persisted
parquets so that no number is typed by hand.

`uv run python -m hydro.experiments.pair_table`  ->  ../notes/sections/Appendices/app_pairs_table.tex
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hydro.config import OUTPUT_DIR, REPO_ROOT
from hydro.experiments.decomposition import FALLBACK_Q_W, TABULATED_Q_W
from hydro.experiments.paper_figures import town

OUT = REPO_ROOT.parent / "notes" / "sections" / "Appendices" / "app_pairs_table.tex"
BASIN = {"susquehanna": "Susq.", "wabash": "Wabash", "iowa_cedar": "Iowa/Cedar"}


def _f(x, nd=2) -> str:
    return "--" if x is None else f"${x:.{nd}f}$"


def main() -> None:
    d = pl.read_parquet(OUTPUT_DIR / "decomposition.parquet")

    def arm(structure, method, q_w=None, cols=("beta",)):
        f = (pl.col("structure") == structure) & (pl.col("method") == method)
        if q_w is not None:
            f = f & (pl.col("safeguard_quantile") == q_w)
        return d.filter(f).select("pair", *cols)

    base = arm("tributary", "naive OLS", cols=("region", "x_name", "y_name", "n_events", "k",
                                              "safeguard_area_coverage", "beta", "b_W", "b_R" if "b_R" in d.columns else "b_U"))
    base = base.rename({"beta": "naive"})
    joins = [
        (arm("tributary", "certificate", TABULATED_Q_W, ("beta", "n_cert")), {"beta": "c25", "n_cert": "n25"}),
        (arm("tributary", "certificate", FALLBACK_Q_W, ("beta", "n_cert")), {"beta": "c5", "n_cert": "n5"}),
        (arm("tributary", "OLS adjusting for W"), {"beta": "adj"}),
        (arm("tributary", "OLS subtracting W"), {"beta": "sub"}),
        (arm("proxy", "certificate", TABULATED_Q_W, ("beta", "n_cert")), {"beta": "p5", "n_cert": "np5"}),
        (arm("tributary", "certificate minus W", TABULATED_Q_W, ("beta", "n_cert")), {"beta": "m25", "n_cert": "nm25"}),
        (arm("proxy", "proximal (two proxies)"), {"beta": "bridge"}),
        (arm("proxy", "proximal (P, W)"), {"beta": "bridge_pw"}),
        (arm("tributary", "certificate (single max)", TABULATED_Q_W, ("beta", "n_cert")), {"beta": "mx25", "n_cert": "nmx25"}),
        (arm("proxy", "certificate (single max)", TABULATED_Q_W, ("beta", "n_cert")), {"beta": "px25", "n_cert": "npx25"}),
    ]
    for j, names in joins:
        base = base.join(j.rename(names), on="pair", how="left")
    base = base.sort("naive")

    lines = [
        r"\begin{tabular}{llrrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Pair & Basin & $n$ & $k$ & cov. & naive & $b_W$ & $b_R$ & cert.\ $0.25$ ($n$) & cert.\ $0.5$ ($n$) & cert.$-W$ $0.25$ ($n$) & adj.\ $W$ & subtr.\ $W$ & cert.\ $P$ $0.25$ ($n$) & max $W$ & max $P$ & br.\ $(P_1,P_2)$ & br.\ $(P,W)$ \\",
        r"\midrule",
    ]
    labels = [f"{town(r['x_name'])} $\\to$ {town(r['y_name'])}" for r in base.iter_rows(named=True)]
    for r, pair in zip(base.iter_rows(named=True), labels):
        if labels.count(pair) > 1:  # two White River gauges share a town; see paper_figures._label
            pair += f" ({r['pair'][:8]})"
        c5 = f"{_f(r['c5'])} ({r['n5']})" if r["c5"] is not None else "--"
        c25 = f"{_f(r['c25'])} ({r['n25']})" if r["c25"] is not None else "--"
        p5 = f"{_f(r['p5'])} ({r['np5']})" if r["p5"] is not None else "--"
        m25 = f"{_f(r['m25'])} ({r['nm25']})" if r["m25"] is not None else "--"
        mx = _f(r["mx25"]) if r["mx25"] is not None else "--"
        px = _f(r["px25"]) if r["px25"] is not None else "--"
        lines.append(
            f"{pair} & {BASIN[r['region']]} & {r['n_events']} & {r['k']} & {_f(r['safeguard_area_coverage'])} & "
            f"{_f(r['naive'])} & {_f(r['b_W'])} & {_f(r['b_U'])} & {c25} & {c5} & {m25} & {_f(r['adj'])} & {_f(r['sub'])} & {p5} & {mx} & {px} & {_f(r['bridge'])} & {_f(r['bridge_pw'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({base.height} rows)")


if __name__ == "__main__":
    main()
