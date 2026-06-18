from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from analysis import assign_kmeans_clusters, fit_factor_model
from data_prep import prepare_feature_matrix
from datatypes import Config, FactorModelResult
from helpers import build_factor_model_comparison, factor_structure_metrics
from model_selection import likelihood_metrics


METRIC_COLUMNS = (
    "mean_primary_abs_loading",
    "mean_cross_loading_gap",
    "simple_feature_share",
    "covariance_reconstruction_rmse",
)


def summarize_config(
    df: pd.DataFrame,
    config: Config,
    *,
    cluster_data: tuple[np.ndarray, np.ndarray] | None = None,
    factor_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    global_result: FactorModelResult | None = None,
) -> dict[str, object]:
    if cluster_data is None:
        _, x_cluster = prepare_feature_matrix(df, config.clustering_features)
        cluster_labels = assign_kmeans_clusters(x_cluster, config)
    else:
        x_cluster, cluster_labels = cluster_data

    cluster_counts = np.unique(cluster_labels, return_counts=True)[1]
    max_cluster_factors = min(
        int(cluster_counts.min()),
        len(config.factor_features),
    )

    if factor_data is None:
        factor_values, x_factor = prepare_feature_matrix(
            df,
            config.factor_features,
        )
    else:
        factor_values, x_factor = factor_data

    if global_result is None:
        global_result = fit_factor_model(x_factor, config)

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
        "max_cluster_supported_factors": max_cluster_factors,
    }

    global_metrics = factor_structure_metrics(x_factor, global_result)
    for metric in METRIC_COLUMNS:
        summary[f"global_{metric}"] = float(global_metrics[metric])

    global_likelihood = likelihood_metrics(
        x_factor,
        global_result,
        n_features=x_factor.shape[1],
        n_factors=config.n_factors,
        eigenvalue_floor=config.likelihood_eigenvalue_floor,
    )

    for key, value in global_likelihood.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            summary[f"global_{key}"] = value

    if config.n_factors > max_cluster_factors:
        summary["clustered_status"] = "skipped_sample_limit"
        summary["clustered_reason"] = (
            f"{config.n_factors} factors exceed the smallest cluster's "
            f"supported maximum of {max_cluster_factors}"
        )
        for metric in METRIC_COLUMNS:
            summary[f"clustered_{metric}"] = np.nan
            summary[f"clustered_minus_global_{metric}"] = np.nan

        likelihood_nan_columns = (
            "clustered_n_parameters",
            "clustered_exact_total_log_likelihood",
            "clustered_regularized_total_log_likelihood",
            "clustered_regularized_aic",
            "clustered_regularized_bic",
            "clustered_unstable_models",
            "clustered_minus_global_regularized_bic",
            "clustered_minus_global_regularized_aic",
            "clustered_minus_global_regularized_total_log_likelihood",
        )

        for column in likelihood_nan_columns:
            summary[column] = np.nan
        return summary

    clustered_regularized_log_likelihood = 0.0
    clustered_exact_log_likelihood = 0.0
    clustered_parameters = 0
    clustered_unstable_models = 0
    for cluster in sorted(np.unique(cluster_labels)):
        cluster_mask = cluster_labels == cluster

        cluster_result = fit_factor_model(
            x_factor[cluster_mask],
            config,
        )

        cluster_likelihood = likelihood_metrics(
            x_factor[cluster_mask],
            cluster_result,
            n_features=x_factor.shape[1],
            n_factors=config.n_factors,
            eigenvalue_floor=config.likelihood_eigenvalue_floor,
        )

        clustered_regularized_log_likelihood += float(
            cluster_likelihood["regularized_total_log_likelihood"]
        )
        clustered_exact_log_likelihood += float(
            cluster_likelihood["exact_total_log_likelihood"]
        )
        clustered_parameters += int(cluster_likelihood["n_parameters"])

        if int(cluster_likelihood["uniquenesses_below_floor"]) > 0:
            clustered_unstable_models += 1

    n_observations = len(df)

    clustered_regularized_aic = (
        2 * clustered_parameters - 2 * clustered_regularized_log_likelihood
    )

    clustered_regularized_bic = (
        clustered_parameters * np.log(n_observations)
        - 2 * clustered_regularized_log_likelihood
    )

    summary["clustered_n_parameters"] = clustered_parameters
    summary["clustered_exact_total_log_likelihood"] = clustered_exact_log_likelihood
    summary["clustered_regularized_total_log_likelihood"] = (
        clustered_regularized_log_likelihood
    )
    summary["clustered_regularized_aic"] = clustered_regularized_aic
    summary["clustered_regularized_bic"] = clustered_regularized_bic
    summary["clustered_unstable_models"] = clustered_unstable_models

    summary["clustered_minus_global_regularized_bic"] = (
        clustered_regularized_bic - float(global_likelihood["regularized_bic"])
    )
    summary["clustered_minus_global_regularized_aic"] = (
        clustered_regularized_aic - float(global_likelihood["regularized_aic"])
    )
    summary["clustered_minus_global_regularized_total_log_likelihood"] = (
        clustered_regularized_log_likelihood
        - float(global_likelihood["regularized_total_log_likelihood"])
    )

    comparison, _ = build_factor_model_comparison(
        df,
        cluster_labels,
        factor_values,
        x_factor,
        global_result,
        config,
    )
    clustered_row = comparison.loc[
        comparison["scope"] == "within_clusters_weighted_mean"
    ].iloc[0]
    summary["clustered_status"] = "ok"
    summary["clustered_reason"] = ""

    for metric in METRIC_COLUMNS:
        global_value = float(global_metrics[metric])
        clustered_value = float(clustered_row[metric])
        summary[f"clustered_{metric}"] = clustered_value
        summary[f"clustered_minus_global_{metric}"] = clustered_value - global_value

    return summary


def run_config_sweep(
    df: pd.DataFrame,
    configs: Sequence[Config],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cluster_cache: dict[
        tuple[tuple[str, ...], int, int, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    factor_data_cache: dict[
        tuple[str, ...],
        tuple[pd.DataFrame, np.ndarray],
    ] = {}
    global_result_cache: dict[
        tuple[tuple[str, ...], int, object, int],
        FactorModelResult,
    ] = {}

    for experiment_id, config in enumerate(configs, start=1):
        try:
            cluster_key = (
                tuple(config.clustering_features),
                config.n_clusters,
                config.seed,
                config.kmeans_restarts,
            )
            if cluster_key not in cluster_cache:
                _, x_cluster = prepare_feature_matrix(
                    df,
                    config.clustering_features,
                )
                labels = assign_kmeans_clusters(x_cluster, config)
                cluster_cache[cluster_key] = (x_cluster, labels)

            factor_key = tuple(config.factor_features)
            if factor_key not in factor_data_cache:
                factor_data_cache[factor_key] = prepare_feature_matrix(
                    df,
                    config.factor_features,
                )

            global_key = (
                factor_key,
                config.n_factors,
                config.rotation,
                config.seed,
            )
            if global_key not in global_result_cache:
                _, x_factor = factor_data_cache[factor_key]
                global_result_cache[global_key] = fit_factor_model(
                    x_factor,
                    config,
                )

            row = summarize_config(
                df,
                config,
                cluster_data=cluster_cache[cluster_key],
                factor_data=factor_data_cache[factor_key],
                global_result=global_result_cache[global_key],
            )
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
