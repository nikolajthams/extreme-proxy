"""Read back every number quoted in the paper's application section and its
appendix from the persisted parquets, in the order of the text.

    uv run python -m hydro.pipeline.paper_numbers

Inter-decile ranges use polars' default ('nearest') quantile.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hydro.config import CACHE_DIR, CONFIG, OUTPUT_DIR
from hydro.pairs import headline_pair

EST = CONFIG.estimator
TARGET = CONFIG.beta_target


def arm(d: pl.DataFrame, method: str, q_p: float | None = None) -> pl.DataFrame:
    f = (pl.col("method") == method) & pl.col("beta").is_not_null()
    if q_p is not None:
        f = f & (pl.col("q_p") == q_p)
    return d.filter(f)


def spread(x: pl.Series) -> str:
    return (f"IQR {x.quantile(0.25):.2f}-{x.quantile(0.75):.2f}, IDR {x.quantile(0.1):.2f}-{x.quantile(0.9):.2f}, "
            f"min {x.min():.2f}, max {x.max():.2f}")


def ci(r: dict) -> str:
    return "" if r.get("ci_lo") is None else f" [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"


def main() -> None:
    pairs = pl.read_parquet(OUTPUT_DIR / "pairs.parquet")
    cands = pl.read_parquet(OUTPUT_DIR / "pair_candidates.parquet")
    d = pl.read_parquet(OUTPUT_DIR / "all_pairs.parquet")
    naive = arm(d, "naive OLS")
    nv = dict(zip(naive["pair"], naive["beta"]))

    def summary(label: str, a: pl.DataFrame, target: float = TARGET) -> None:
        closer = sum(abs(b - target) < abs(nv[p] - target) for p, b in zip(a["pair"], a["beta"]))
        removed = np.median([(nv[p] - b) / (nv[p] - target) for p, b in zip(a["pair"], a["beta"])])
        print(f"  {label:<26} pairs {a.height:>2}  median {a['beta'].median():.2f}  {spread(a['beta'])}  "
              f"closer than naive {closer}/{a.height}  bias removed {100 * removed:.0f}%")

    print("== Design ==")
    print(f"  connected candidates {cands.height}; pairs {pairs.height}: " + ", ".join(
        f"{r['region']} {r['len']}" for r in pairs.group_by("region").len().sort("region").to_dicts()))
    print("  screen results:", dict(cands.group_by("screen_result").len().sort("len", descending=True).iter_rows()))
    print(f"  days per pair: median {naive['n_cert'].median():.0f}, range {naive['n_cert'].min()}-{naive['n_cert'].max()}; "
          f"common years {pairs['overlap_years'].min()}-{pairs['overlap_years'].max()}; "
          f"daily tail index {pairs['daily_tail_index_t'].min():.2f}-{pairs['daily_tail_index_t'].max():.2f}; "
          f"lags {dict(pairs['lag_days'].value_counts().sort('lag_days').iter_rows())}")
    print("  gauges passing the screen per basin: " + ", ".join(
        f"{r.name} {pl.read_parquet(CACHE_DIR / 'screen' / f'{r.name}_sites.parquet').filter(pl.col('screen_result') == 'pass').height}"
        for r in CONFIG.regions))

    print("\n== Results, all pairs ==")
    print(f"  naive OLS: pairs {naive.height}, median {naive['beta'].median():.2f}, {spread(naive['beta'])}")
    for q in EST.q_p_grid:
        a = arm(d, "certified OLS", q)
        if a.height:
            summary(f"certified OLS q_P = {q}", a)
            print(f"      certified days: median {a['n_cert'].median():.0f}, range {a['n_cert'].min()}-{a['n_cert'].max()}; "
                  f"q_T counts {dict(a['q_t'].value_counts().sort('q_t').iter_rows())}; "
                  f"v median {a['v'].median():.0f} m3/s")
    summary("proximal bridge", arm(d, "proximal bridge"))
    b = arm(d, "proximal bridge")["beta"]
    print(f"      bridge outside [-1, 3]: {int(((b < -1) | (b > 3)).sum())}")

    hp = headline_pair()
    h = d.filter(pl.col("pair") == hp["pair"])
    print(f"\n== Headline pair {hp['pair']}: {hp['t_name']} -> {hp['y_name']} ==")
    print(f"  area {hp['area_t_km2']:.0f} -> {hp['area_y_km2']:.0f} km2 (ratio {hp['area_ratio']:.2f}), "
          f"{hp['overlap_years']} common years, {hp['n_daily']} days, lag {hp['lag_days']} d, "
          f"daily tail index {hp['daily_tail_index_t']:.2f}")
    print(f"  proxies {hp['proxy_names']} at {hp['proxy_distance_to_y_km']} km from Y")
    for m in ("naive OLS", "proximal bridge"):
        r = arm(h, m).row(0, named=True)
        print(f"  {m:<28} {r['beta']:.2f}{ci(r)} (n = {r['n_cert']})")
    for q in EST.q_p_grid:
        a = arm(h, "certified OLS", q)
        if a.height:
            r = a.row(0, named=True)
            print(f"  certified OLS q_P = {q:<5} {r['beta']:.2f}{ci(r)} on {r['n_cert']} days (q_T {r['q_t']}, v {r['v']:.0f} m3/s)")
        else:
            print(f"  certified OLS q_P = {q:<5} no operating point")
    c = arm(h, "certified OLS", EST.q_p).row(0, named=True)
    print(f"  fraction of the naive bias remaining at q_P = {EST.q_p}: {(c['beta'] - TARGET) / (nv[hp['pair']] - TARGET):.2f}")
    hl = pl.read_parquet(OUTPUT_DIR / "headline.parquet")
    for m in ("V1 tail only", "V2 permuted safeguard", "V3 other-basin safeguard"):
        r = arm(hl, m, EST.q_p).row(0, named=True)
        print(f"  {m:<28} {r['beta']:.2f}{ci(r)} (n = {r['n_cert']}, v {r['v']:.0f})")

    print("\n== Placebo (true beta = 0) ==")
    pc = pl.read_parquet(OUTPUT_DIR / "placebo_candidates.parquet")
    pp = pl.read_parquet(OUTPUT_DIR / "placebo_pairs.parquet")
    print(f"  disconnected candidates {pc.height}, passing the screens {pc.filter(pl.col('screen_result') == 'pass').height}, "
          f"used {pp.height}; distance {pp['distance_km'].min():.0f}-{pp['distance_km'].max():.0f} km, "
          f"days median {pp['n_daily'].median():.0f}")
    p = pl.read_parquet(OUTPUT_DIR / "placebo.parquet")
    pn = dict(zip(arm(p, "naive OLS")["pair"], arm(p, "naive OLS")["beta"]))
    for label, a in (("naive OLS", arm(p, "naive OLS")), ("proximal bridge", arm(p, "proximal bridge")),
                     (f"certified OLS q_P = {EST.q_p}", arm(p, "certified OLS", EST.q_p))):
        closer = sum(abs(b) < abs(pn[q]) for q, b in zip(a["pair"], a["beta"]))
        out = int(((a["beta"] < -2) | (a["beta"] > 2)).sum())
        print(f"  {label:<26} pairs {a.height:>2}  median {a['beta'].median():.2f}  {spread(a['beta'])}  "
              f"closer to zero than naive {closer}/{a.height}  outside [-2, 2]: {out}")
        if "certified" in label:
            print(f"      certified days: median {a['n_cert'].median():.0f}; q_T counts {dict(a['q_t'].value_counts().sort('q_t').iter_rows())}")


if __name__ == "__main__":
    main()
