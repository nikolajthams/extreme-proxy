"""Spec 5.2: runs declustering.

A flood spans several days, so raw daily exceedances are massively serially
dependent. The consistency condition in the paper concerns the *effective*
number of certified observations, so all thresholding downstream operates on
one row per flood event.
"""

from __future__ import annotations

import polars as pl


def runs_decluster(
    df: pl.DataFrame,
    value_col: str,
    threshold: float,
    run_length_days: int,
    date_col: str = "date",
) -> pl.DataFrame:
    """One row per exceedance cluster: the row at the cluster's `value_col` peak.

    A cluster ends once `run_length_days` consecutive days have passed without
    an exceedance. Gaps in the record count towards the run length, which is the
    conservative choice (a gap can only split a cluster, never merge two).
    """
    if df.height == 0:
        return df.clear()

    exceed = (
        df.sort(date_col)
        .filter(pl.col(value_col) > threshold)
    )
    if exceed.height == 0:
        return exceed

    exceed = exceed.with_columns(
        (
            (pl.col(date_col) - pl.col(date_col).shift(1)).dt.total_days() > run_length_days
        )
        .fill_null(True)
        .cum_sum()
        .alias("_cluster")
    )
    return (
        exceed.sort(["_cluster", value_col], descending=[False, True])
        .group_by("_cluster", maintain_order=True)
        .first()
        .drop("_cluster")
        .sort(date_col)
    )


def cluster_count(
    df: pl.DataFrame, value_col: str, threshold: float, run_length_days: int
) -> int:
    return runs_decluster(df, value_col, threshold, run_length_days).height
