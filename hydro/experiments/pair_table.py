"""The per-pair table of the appendix, from all_pairs.parquet, written into
the notes as app_pairs_table.tex.

    uv run python -m hydro.experiments.pair_table
"""

from __future__ import annotations

import re

import polars as pl

from hydro.config import CONFIG, NOTES_TABLE_DIR, OUTPUT_DIR, REGION_BY_NAME

EST = CONFIG.estimator
HEADER = (r"Pair & Basin & days & $A_Y/A_T$ & naive OLS & cert.\ OLS ($n$, $q_T$) & bridge $(P_1,P_2)$ \\")


def town(name: str) -> str:
    """'Susquehanna River at Danville, PA' -> 'Danville'."""
    part = re.split(r"\s+(?:at|near|nr|AT|NEAR|NR)\s+", name)[-1]
    part = re.sub(r"[,.]?\s*(PA|NY|IN|IL|IA|MD|IND)\.?\s*$", "", part.strip())
    return part.strip(" ,").title()


def _f(x, nd: int = 2) -> str:
    return "--" if x is None else f"${x:.{nd}f}$"


def main() -> None:
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet")

    def col(method, q_p=None, cols=("beta",)):
        f = (pl.col("method") == method) & pl.col("beta").is_not_null()
        if q_p is not None:
            f = f & (pl.col("q_p") == q_p)
        return d.filter(f).select("pair", *cols)

    base = col("naive OLS", cols=("region", "t_name", "y_name", "n_days", "area_ratio", "beta")).rename({"beta": "naive"})
    base = base.with_columns(pl.col("pair").str.split("->").list.get(0).alias("t"),
                             pl.col("pair").str.split("->").list.get(1).alias("y"))
    for j, names in (
        (col("certified OLS", EST.q_p, ("beta", "n_cert", "q_t")), {"beta": "cert", "n_cert": "n_cert", "q_t": "q_t"}),
        (col("proximal bridge"), {"beta": "bridge"}),
    ):
        base = base.join(j.rename(names), on="pair", how="left")
    base = base.sort("naive")

    rows = base.to_dicts()
    labels = [f"{town(r['t_name'])} $\\to$ {town(r['y_name'])}" for r in rows]
    lines = []
    for r, label in zip(rows, labels):
        if labels.count(label) > 1:  # two gauges can share a town name: add the site id that differs
            twins = [o for o, lab in zip(rows, labels) if lab == label and o is not r]
            label += f" ({r['y'] if all(o['t'] == r['t'] for o in twins) else r['t']})"
        cert = f"{_f(r['cert'])} ({r['n_cert']}, {r['q_t']})" if r["cert"] is not None else "--"
        lines.append(f"{label} & {REGION_BY_NAME[r['region']].label} & {r['n_days']} & {_f(r['area_ratio'])} & "
                     f"{_f(r['naive'])} & {cert} & {_f(r['bridge'])} \\\\")
    NOTES_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join([r"\begin{tabular}{llrrrrr}", r"\toprule", HEADER, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (NOTES_TABLE_DIR / "app_pairs_table.tex").write_text(body + "\n")
    print(f"wrote app_pairs_table.tex ({len(lines)} rows)")


if __name__ == "__main__":
    main()
