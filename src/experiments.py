from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from analysis import assign_kmeans_clusters, fit_factor_model
from data_prep import prepare_feature_matrix
from datatypes import Config
from helpers import build_factor_model_comparison


METRIC_COLUMNS = (
    "mean_primary_abs_loading",
    "mean_cross_loading_gap",
    "simple_feature_share",
    "covariance_reconstruction_rmse",
)


def summarize_config(df: pd.DataFrame, config: Config) -> dict[str, object]:
    _, x_cluster = prepare_feature_matrix(df, config.clustering_features)
    cluster_labels = assign_kmeans_clusters(x_cluster, config)
    cluster_counts = np.unique(cluster_labels, return_counts=True)[1]

    _, x_factor = prepare_feature_matrix(df, config.factor_features)
    global_result = fit_factor_model(x_factor, config)
    comparison, _ = build_factor_model_comparison(
        df,
        cluster_labels,
        x_factor,
        global_result,
        config,
    )

    global_row = comparison.loc[comparison["scope"] == "global"].iloc[0]
    clustered_row = comparison.loc[
        comparison["scope"] == "within_clusters_weighted_mean"
    ].iloc[0]

    summary: dict[str, object] = {
        "clustering_features": " | ".join(config.clustering_features),
        "n_clustering_features": len(config.clustering_features),
        "n_clusters": config.n_clusters,
        "n_factors": config.n_factors,
        "rotation": config.rotation or "none",
        "silhouette_score": float(silhouette_score(x_cluster, cluster_labels)),
        "smallest_cluster": int(cluster_counts.min()),
        "largest_cluster": int(cluster_counts.max()),
        "clusters_below_efa_rule": int(
            (cluster_counts < 5 * len(config.factor_features)).sum()
        ),
    }

    for metric in METRIC_COLUMNS:
        global_value = float(global_row[metric])
        clustered_value = float(clustered_row[metric])
        summary[f"global_{metric}"] = global_value
        summary[f"clustered_{metric}"] = clustered_value
        summary[f"clustered_minus_global_{metric}"] = (
            clustered_value - global_value
        )

    return summary


def run_config_sweep(
    df: pd.DataFrame,
    configs: Sequence[Config],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for experiment_id, config in enumerate(configs, start=1):
        try:
            row = summarize_config(df, config)
            row["experiment_id"] = experiment_id
            row["status"] = "ok"
            row["error"] = ""
        except (ValueError, np.linalg.LinAlgError) as error:
            row = {
                "experiment_id": experiment_id,
                "clustering_features": " | ".join(config.clustering_features),
                "n_clustering_features": len(config.clustering_features),
                "n_clusters": config.n_clusters,
                "n_factors": config.n_factors,
                "rotation": config.rotation or "none",
                "status": "failed",
                "error": str(error),
            }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("experiment_id")
