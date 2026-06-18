from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from analysis import assign_kmeans_clusters
from data_prep import prepare_feature_matrix, standardize_features
from datatypes import Config


def remove_constant_features(
    values: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    minimums = values.min(axis=0)
    maximums = values.max(axis=0)
    numerically_constant = np.isclose(
        minimums.to_numpy(dtype=float),
        maximums.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    retained_columns = values.columns[~numerically_constant]
    dropped_columns = [
        str(column)
        for column in values.columns
        if column not in retained_columns
    ]
    return values.loc[:, retained_columns], dropped_columns


def parallel_analysis(
    x_scaled: np.ndarray,
    *,
    iterations: int,
    percentile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n_rows, n_features = x_scaled.shape

    observed = np.linalg.eigvalsh(
        np.corrcoef(x_scaled, rowvar=False)
    )[::-1]
    random_eigenvalues = np.empty((iterations, n_features))

    for iteration in range(iterations):
        permuted = np.empty_like(x_scaled)
        for feature_index in range(n_features):
            permuted[:, feature_index] = rng.permutation(
                x_scaled[:, feature_index]
            )
        random_eigenvalues[iteration] = np.linalg.eigvalsh(
            np.corrcoef(permuted, rowvar=False)
        )[::-1]

    thresholds = np.percentile(
        random_eigenvalues,
        percentile,
        axis=0,
    )
    max_estimable_factors = min(n_features, n_rows - 1)
    retain = np.zeros(n_features, dtype=bool)
    retain[:max_estimable_factors] = (
        observed[:max_estimable_factors]
        > thresholds[:max_estimable_factors]
    )
    suggested_factors = int(retain.sum())
    return observed, thresholds, suggested_factors


def save_parallel_analysis_plot(
    observed: np.ndarray,
    thresholds: np.ndarray,
    *,
    scope: str,
    output_path: Path,
) -> None:
    ranks = np.arange(1, len(observed) + 1)
    tick_step = max(1, len(ranks) // 15)

    plt.figure(figsize=(11, 6))
    plt.plot(ranks, observed, marker="o", markersize=4, label="Observed")
    plt.plot(
        ranks,
        thresholds,
        marker="o",
        markersize=3,
        label="Permutation threshold",
    )
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Correlation-matrix eigenvalue")
    plt.xticks(ranks[::tick_step])
    plt.title(f"Parallel Analysis: {scope}")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_parallel_analysis_diagnostics(
    factor_values: pd.DataFrame,
    cluster_labels: np.ndarray,
    config: Config,
) -> pd.DataFrame:
    output_dir = config.out_dir / "diagnostics" / "parallel_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, str]] = []

    scopes: list[tuple[str, pd.DataFrame]] = [("global", factor_values)]
    scopes.extend(
        (
            f"cluster_{int(cluster)}",
            factor_values.loc[cluster_labels == cluster],
        )
        for cluster in sorted(np.unique(cluster_labels))
    )

    for scope_index, (scope, values) in enumerate(scopes):
        diagnostic_values, dropped_features = remove_constant_features(values)
        x_scaled = standardize_features(diagnostic_values)
        observed, thresholds, suggested_factors = parallel_analysis(
            x_scaled,
            iterations=config.parallel_analysis_iterations,
            percentile=config.parallel_analysis_percentile,
            seed=config.seed + scope_index,
        )

        rows.append(
            {
                "scope": scope,
                "n_observations": x_scaled.shape[0],
                "n_input_variables": values.shape[1],
                "n_variables": diagnostic_values.shape[1],
                "n_dropped_constant": len(dropped_features),
                "iterations": config.parallel_analysis_iterations,
                "percentile": config.parallel_analysis_percentile,
                "suggested_factors": suggested_factors,
            }
        )
        pd.DataFrame(
            {
                "eigenvalue_rank": np.arange(1, len(observed) + 1),
                "observed_eigenvalue": observed,
                "permutation_threshold": thresholds,
                "retain": (
                    (np.arange(len(observed)) < min(x_scaled.shape[1], x_scaled.shape[0] - 1))
                    & (observed > thresholds)
                ),
            }
        ).to_csv(output_dir / f"{scope}_eigenvalues.csv", index=False)
        save_parallel_analysis_plot(
            observed,
            thresholds,
            scope=scope,
            output_path=output_dir / f"{scope}.png",
        )
        dropped_rows.extend(
            {"scope": scope, "feature": feature}
            for feature in dropped_features
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(
        dropped_rows,
        columns=pd.Index(["scope", "feature"]),
    ).to_csv(output_dir / "dropped_constant_features.csv", index=False)
    return summary


def run_parallel_analysis_sweep(
    df: pd.DataFrame,
    factor_values: pd.DataFrame,
    configs: Sequence[Config],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    eigenvalue_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    global_values, global_dropped = remove_constant_features(factor_values)
    global_scaled = standardize_features(global_values)
    global_observed, global_thresholds, global_suggested = parallel_analysis(
        global_scaled,
        iterations=configs[0].parallel_analysis_iterations,
        percentile=configs[0].parallel_analysis_percentile,
        seed=configs[0].seed,
    )
    summary_rows.append(
        {
            "specification_id": 0,
            "clustering_features": "global",
            "n_clustering_features": 0,
            "n_clusters": 0,
            "scope": "global",
            "cluster": "all",
            "n_observations": global_scaled.shape[0],
            "n_input_variables": factor_values.shape[1],
            "n_variables": global_values.shape[1],
            "n_dropped_constant": len(global_dropped),
            "silhouette_score": np.nan,
            "suggested_factors": global_suggested,
        }
    )
    for rank, (observed, threshold) in enumerate(
        zip(global_observed, global_thresholds, strict=True),
        start=1,
    ):
        eigenvalue_rows.append(
            {
                "specification_id": 0,
                "scope": "global",
                "cluster": "all",
                "eigenvalue_rank": rank,
                "observed_eigenvalue": observed,
                "permutation_threshold": threshold,
                "retain": (
                    rank
                    <= min(global_scaled.shape[1], global_scaled.shape[0] - 1)
                    and observed > threshold
                ),
            }
        )

    for specification_id, config in enumerate(configs, start=1):
        _, x_cluster = prepare_feature_matrix(
            df,
            config.clustering_features,
        )
        labels = assign_kmeans_clusters(x_cluster, config)
        silhouette = float(silhouette_score(x_cluster, labels))
        feature_label = " | ".join(config.clustering_features)

        for cluster in sorted(np.unique(labels)):
            scope = f"cluster_{int(cluster)}"
            values = factor_values.loc[labels == cluster]
            diagnostic_values, dropped_features = remove_constant_features(
                values
            )
            x_scaled = standardize_features(diagnostic_values)
            observed, thresholds, suggested_factors = parallel_analysis(
                x_scaled,
                iterations=config.parallel_analysis_iterations,
                percentile=config.parallel_analysis_percentile,
                seed=config.seed + specification_id * 100 + int(cluster),
            )

            summary_rows.append(
                {
                    "specification_id": specification_id,
                    "clustering_features": feature_label,
                    "n_clustering_features": len(config.clustering_features),
                    "n_clusters": config.n_clusters,
                    "scope": scope,
                    "cluster": int(cluster),
                    "n_observations": x_scaled.shape[0],
                    "n_input_variables": values.shape[1],
                    "n_variables": diagnostic_values.shape[1],
                    "n_dropped_constant": len(dropped_features),
                    "silhouette_score": silhouette,
                    "suggested_factors": suggested_factors,
                }
            )
            dropped_rows.extend(
                {
                    "specification_id": specification_id,
                    "scope": scope,
                    "cluster": int(cluster),
                    "feature": feature,
                }
                for feature in dropped_features
            )
            for rank, (observed_value, threshold) in enumerate(
                zip(observed, thresholds, strict=True),
                start=1,
            ):
                eigenvalue_rows.append(
                    {
                        "specification_id": specification_id,
                        "scope": scope,
                        "cluster": int(cluster),
                        "eigenvalue_rank": rank,
                        "observed_eigenvalue": observed_value,
                        "permutation_threshold": threshold,
                        "retain": (
                            rank
                            <= min(x_scaled.shape[1], x_scaled.shape[0] - 1)
                            and observed_value > threshold
                        ),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    pd.DataFrame(eigenvalue_rows).to_csv(
        output_dir / "sweep_eigenvalues.csv",
        index=False,
    )
    pd.DataFrame(
        dropped_rows,
        columns=pd.Index(
            ["specification_id", "scope", "cluster", "feature"]
        ),
    ).to_csv(output_dir / "sweep_dropped_constant_features.csv", index=False)
    return summary
