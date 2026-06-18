from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import assign_kmeans_clusters, fit_factor_model
from data_prep import prepare_feature_matrix
from datatypes import Config, FactorModelResult
from model_selection import (
    likelihood_metrics,
    regularized_gaussian_log_likelihood,
)


def save_adaptive_heatmap(
    loadings: np.ndarray,
    *,
    cluster: int,
    factor_features: tuple[str, ...],
    output_path: Path,
) -> None:
    display_features = [feature.replace("_", " ") for feature in factor_features]
    table = pd.DataFrame(
        loadings,
        index=pd.Index(display_features),
        columns=pd.Index(
            [
                f"factor_{factor_index}"
                for factor_index in range(1, loadings.shape[1] + 1)
            ]
        ),
    )
    plt.figure(figsize=(max(10, loadings.shape[1]), 22))
    sns.heatmap(
        table,
        cmap="vlag",
        center=0.0,
        cbar_kws={"shrink": 0.6},
    )
    plt.title(f"Adaptive Factor Loadings: Cluster {cluster}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def select_pa_factor_counts(
    pa_summary: pd.DataFrame,
    config: Config,
) -> dict[int, int]:
    feature_label = " | ".join(config.clustering_features)
    rows = pa_summary.loc[
        (pa_summary["clustering_features"] == feature_label)
        & (pa_summary["n_clusters"] == config.n_clusters)
        & (pa_summary["scope"] != "global")
    ]
    if len(rows) != config.n_clusters:
        raise ValueError(
            "PA summary does not contain every cluster for the adaptive regime"
        )
    return {int(row.cluster): int(row.suggested_factors) for row in rows.itertuples()}


def run_adaptive_regime(
    df: pd.DataFrame,
    x_factor: np.ndarray,
    pa_summary: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    output_dir = config.out_dir / "adaptive_regime"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, x_cluster = prepare_feature_matrix(
        df,
        config.clustering_features,
    )
    cluster_labels = assign_kmeans_clusters(x_cluster, config)
    factor_counts = select_pa_factor_counts(pa_summary, config)

    assignments = df.loc[:, list(config.id_columns)].copy()
    assignments["cluster"] = cluster_labels
    assignments["n_factors"] = [
        factor_counts[int(cluster)] for cluster in cluster_labels
    ]
    assignments.to_csv(output_dir / "assignments.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    communality_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    likelihood_rows: list[dict[str, object]] = []
    adaptive_exact_log_likelihood = 0.0
    adaptive_log_likelihood = 0.0
    adaptive_parameters = 0
    adaptive_bic = 0.0
    unstable_clusters = 0

    for cluster in sorted(factor_counts):
        cluster_mask = cluster_labels == cluster
        n_factors = factor_counts[cluster]
        cluster_config = replace(config, n_factors=n_factors)
        cluster_result = fit_factor_model(
            x_factor[cluster_mask],
            cluster_config,
        )
        metrics = likelihood_metrics(
            x_factor[cluster_mask],
            cluster_result,
            n_features=x_factor.shape[1],
            n_factors=n_factors,
            eigenvalue_floor=config.likelihood_eigenvalue_floor,
        )
        adaptive_log_likelihood += float(metrics["regularized_total_log_likelihood"])
        adaptive_exact_log_likelihood += float(metrics["exact_total_log_likelihood"])
        adaptive_parameters += int(metrics["n_parameters"])
        adaptive_bic += float(metrics["regularized_bic"])
        if int(metrics["uniquenesses_below_floor"]) > 0:
            unstable_clusters += 1

        summary_rows.append(
            {
                "model": "adaptive_cluster",
                "cluster": cluster,
                "n_factors": n_factors,
                "criterion_scope": "conditional_on_fixed_clusters",
                "warning": (
                    "regularized likelihood; near-singular uniquenesses"
                    if int(metrics["uniquenesses_below_floor"]) > 0
                    else ""
                ),
                **metrics,
            }
        )

        communalities = (cluster_result.loadings**2).sum(axis=1)
        for feature_index, feature in enumerate(config.factor_features):
            communality_rows.append(
                {
                    "cluster": cluster,
                    "n_factors": n_factors,
                    "feature": feature,
                    "communality": communalities[feature_index],
                    "uniqueness": cluster_result.uniqueness[feature_index],
                }
            )
            for factor_index in range(n_factors):
                loading_rows.append(
                    {
                        "cluster": cluster,
                        "n_factors": n_factors,
                        "feature": feature,
                        "factor": factor_index + 1,
                        "loading": cluster_result.loadings[
                            feature_index,
                            factor_index,
                        ],
                    }
                )

        cluster_ids = df.loc[
            cluster_mask,
            list(config.id_columns),
        ].reset_index(drop=True)
        regularized_log_likelihood = regularized_gaussian_log_likelihood(
            x_factor[cluster_mask],
            cluster_result,
            eigenvalue_floor=config.likelihood_eigenvalue_floor,
        )
        for country_index, country in cluster_ids.iterrows():
            likelihood_rows.append(
                {
                    **country.to_dict(),
                    "cluster": cluster,
                    "n_factors": n_factors,
                    "exact_log_likelihood": (
                        cluster_result.log_likelihood[country_index]
                    ),
                    "regularized_log_likelihood": (
                        regularized_log_likelihood[country_index]
                    ),
                }
            )
            for factor_index in range(n_factors):
                score_rows.append(
                    {
                        **country.to_dict(),
                        "cluster": cluster,
                        "n_factors": n_factors,
                        "factor": factor_index + 1,
                        "score": cluster_result.scores[
                            country_index,
                            factor_index,
                        ],
                    }
                )

        save_adaptive_heatmap(
            cluster_result.loadings,
            cluster=cluster,
            factor_features=tuple(config.factor_features),
            output_path=output_dir / f"cluster_{cluster}_loadings.png",
        )

    total_observations = len(df)
    adaptive_aic = 2 * adaptive_parameters - 2 * adaptive_log_likelihood
    summary_rows.append(
        {
            "model": "adaptive_aggregate",
            "cluster": "all",
            "n_factors": "/".join(
                str(factor_counts[cluster]) for cluster in sorted(factor_counts)
            ),
            "criterion_scope": "conditional_on_fixed_clusters",
            "warning": (
                f"{unstable_clusters} cluster models required likelihood "
                "regularization"
            ),
            "n_observations": total_observations,
            "n_parameters": adaptive_parameters,
            "exact_total_log_likelihood": adaptive_exact_log_likelihood,
            "exact_mean_log_likelihood": (
                adaptive_exact_log_likelihood / total_observations
            ),
            "regularized_total_log_likelihood": adaptive_log_likelihood,
            "regularized_mean_log_likelihood": (
                adaptive_log_likelihood / total_observations
            ),
            "regularized_aic": adaptive_aic,
            "regularized_bic": adaptive_bic,
            "likelihood_eigenvalue_floor": (config.likelihood_eigenvalue_floor),
            "exact_likelihood_finite": bool(np.isfinite(adaptive_exact_log_likelihood)),
            "minimum_uniqueness": np.nan,
            "uniquenesses_below_floor": np.nan,
        }
    )

    global_factor_count = int(
        pa_summary.loc[
            pa_summary["scope"] == "global",
            "suggested_factors",
        ].iloc[0]
    )
    global_config = replace(config, n_factors=global_factor_count)
    global_result = fit_factor_model(x_factor, global_config)
    global_metrics = likelihood_metrics(
        x_factor,
        global_result,
        n_features=x_factor.shape[1],
        n_factors=global_factor_count,
        eigenvalue_floor=config.likelihood_eigenvalue_floor,
    )
    summary_rows.append(
        {
            "model": "global_pa",
            "cluster": "all",
            "n_factors": global_factor_count,
            "criterion_scope": "global",
            "warning": "",
            **global_metrics,
        }
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    pd.DataFrame(loading_rows).to_csv(
        output_dir / "loadings_long.csv",
        index=False,
    )
    pd.DataFrame(communality_rows).to_csv(
        output_dir / "communalities.csv",
        index=False,
    )
    pd.DataFrame(score_rows).to_csv(
        output_dir / "scores_long.csv",
        index=False,
    )
    pd.DataFrame(likelihood_rows).to_csv(
        output_dir / "observation_likelihoods.csv",
        index=False,
    )
    return summary
