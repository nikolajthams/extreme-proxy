from itertools import product
from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl
import tikzplotly
from graphical_toolbox.generate_data.dag import (
    NonlinearParetoDag,
    NonlinearStudentTDag,
)
from graphical_toolbox.models.proximal import LinearProximal
from tqdm import tqdm

from certification_estimators.certificate import QuantileBall
from certification_estimators.certificate_estimator import (
    CertificateEstimator,
    CertificateEstimatorType,
)
from certification_estimators.classical_graphs import PROXY

true_beta = 1.0
sample_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
certification_quantiles = [0.99, 0.999]
safeguard_quantiles = [0.1]
tail_coefficients = [1.5, 2.0, 2.5]

estimator_types: list[CertificateEstimatorType] = ["ols", "max"]
rng = np.random.default_rng(seed=1)

for distribution in (NonlinearParetoDag, NonlinearStudentTDag):
    results = []
    for _ in tqdm(range(100)):
        # Draw dag
        dag = (
            distribution
            .from_dag(PROXY)
            .sample_coefficients(lower=0.2, upper=1.0, two_sided=False, rng=rng)
            # .set_extreme_nodes([x for x in PROXY.topological_sorted_nodes if not x in ["X"]])
            .set_edge_strength("X", "Y", true_beta)
        )
        for sample_size, certification_quantile, safeguard_quantile, tail_coefficient in product(
            sample_sizes, 
            certification_quantiles, 
            safeguard_quantiles,
            tail_coefficients
        ):
            
            # Sample data
            data = dag.sample(n=sample_size, tail_index=tail_coefficient, rng=rng)
            
            # Certificates
            certificates = {
                "Z": QuantileBall(["W"], safeguard_quantile), 
                "X": QuantileBall(["W", "Z"], safeguard_quantile)
            }
            
            # Baseline
            proxy = LinearProximal("X", "Y", "Z", "W").fit_or_null(data)
            if proxy is not None:
                proxy = proxy.item()

            # Valid-bridge baseline: confounding enters Y via W^2 and U^2, so the
            # bridge h(W) = c1*W^2 + c2*W requires squared proxies/instruments
            augmented_data = data.with_columns(
                (pl.col("W") ** 2).alias("W2"),
                (pl.col("Z") ** 2).alias("Z2"),
            )
            proxy_valid_bridge = LinearProximal("X", "Y", ["Z", "Z2"], ["W", "W2"]).fit_or_null(augmented_data)
            if proxy_valid_bridge is not None:
                proxy_valid_bridge = proxy_valid_bridge.item()
            results.append({
                "tail_coefficient": tail_coefficient,
                "sample_size": sample_size,
                "certification_quantile": certification_quantile,
                "safeguard_quantile": safeguard_quantile,
                "proxy": proxy,
                "proxy_valid_bridge": proxy_valid_bridge
                } | {
                    f"certified_{anchor}_{estimator_type}": 
                    CertificateEstimator(anchor, safeguard_set, "X", "Y", certification_quantile, estimator_type).fit_or_null(data)
                    for anchor, safeguard_set in certificates.items()
                    for estimator_type in estimator_types
            })


    hyperparameters = ["tail_coefficient", "sample_size", "certification_quantile", "safeguard_quantile"]
    results_df = (
        pl.DataFrame(results)
        .unpivot(variable_name="method", value_name="estimate", index=hyperparameters)
        .with_columns((pl.col("estimate") - true_beta).alias("bias"))
    )

    stats = (
        results_df
        .group_by(["method"] + hyperparameters)
        .agg(
            pl.median("bias").alias("median bias"),
            pl.mean("bias").alias("mean bias"), 
            pl.std("estimate").alias("standard deviation"),
            (pl.col("estimate") - pl.median("estimate")).abs().median().alias("MAD"),
        )
        .unpivot(index=["method"] + hyperparameters, variable_name="metric")
        .sort(["method", "metric"] + hyperparameters)
    )



    fig_mad = px.line(stats.filter(pl.col("metric") == "MAD"), "sample_size", "value", color="method", facet_row="tail_coefficient", facet_col="certification_quantile", title="Estimator median absolute deviation", log_x=True)
    fig_median_bias = px.line(stats.filter(pl.col("metric") == "median bias"), "sample_size", "value", color="method", facet_row="tail_coefficient", facet_col="certification_quantile", title="Estimator median bias", log_x=True)


    output_folder = Path("experiments/nonlinear_compare_to_gmm")
    output_folder.mkdir(exist_ok=True, parents=True)

    fig_mad.write_html(output_folder / f"{distribution.distribution}_mad.html")
    fig_median_bias.write_html(output_folder / f"{distribution.distribution}_median_bias.html")
    tikzplotly.save(output_folder / f"{distribution.distribution}_mad.tex", fig_mad)
    tikzplotly.save(output_folder / f"{distribution.distribution}_median_bias.tex", fig_median_bias)


print("Done")