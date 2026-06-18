from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_prep import standardize_features
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
    suggested_factors = int(np.sum(observed > thresholds))
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
                "retain": observed > thresholds,
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
