from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from analysis import fit_factor_model
from data_prep import standardize_features
from datatypes import Config, FactorModelResult


def factor_columns(config: Config) -> list[str]:
    return [f"factor_{i}" for i in range(1, config.n_factors + 1)]


def save_cluster_assignments(
    df: pd.DataFrame,
    labels: np.ndarray,
    config: Config,
) -> pd.DataFrame:
    assignments = df.loc[:, list(config.id_columns)].copy()
    assignments["cluster"] = labels
    assignments.to_csv(config.out_dir / "cluster_assignments.csv", index=False)
    return assignments


def save_cluster_profiles(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    config: Config,
) -> None:
    profiles = raw_features.copy()
    profiles["cluster"] = labels
    profiles.groupby("cluster").mean().sort_index().to_csv(
        config.out_dir / "cluster_profiles.csv"
    )


def save_income_crosstab(assignments: pd.DataFrame, config: Config) -> None:
    counts = pd.crosstab(
        assignments["cluster"],
        assignments["world_bank_income_group"],
    )
    counts.to_csv(config.out_dir / "cluster_income_crosstab.csv")


def save_cluster_plot(
    assignments: pd.DataFrame,
    x_scaled: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> None:
    plt.figure(figsize=(10, 6))

    if x_scaled.shape[1] == 1:
        plot_df = pd.DataFrame(
            {
                "standardized_value": x_scaled[:, 0],
                "cluster": labels.astype(str),
                "income": assignments["world_bank_income_group"],
            }
        )
        sns.stripplot(
            data=plot_df,
            x="standardized_value",
            y="cluster",
            hue="income",
            order=[str(cluster) for cluster in sorted(np.unique(labels))],
            jitter=0.25,
            alpha=0.75,
        )
        plt.xlabel(f"{config.clustering_features[0]} (standardized)")
        plt.ylabel("Cluster")
        plt.title("WDI 2022 Countries by K-means Cluster")
    else:
        pca_points = PCA(n_components=2, random_state=config.seed).fit_transform(
            x_scaled
        )
        plot_df = pd.DataFrame(
            {
                "PC1": pca_points[:, 0],
                "PC2": pca_points[:, 1],
                "cluster": labels.astype(str),
                "income": assignments["world_bank_income_group"],
            }
        )
        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            hue="cluster",
            style="income",
            s=55,
        )
        plt.title("WDI 2022 Countries by K-means Cluster")

    plt.tight_layout()
    plt.savefig(config.out_dir / "cluster_plot.png", dpi=180)
    plt.close()


def save_all_cluster_outputs(
    df: pd.DataFrame,
    raw_features: pd.DataFrame,
    x_scaled: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> None:
    assignments = save_cluster_assignments(df, labels, config)
    save_cluster_profiles(raw_features, labels, config)
    save_income_crosstab(assignments, config)
    save_cluster_plot(assignments, x_scaled, labels, config)


def save_factor_loadings(
    result: FactorModelResult,
    config: Config,
) -> pd.DataFrame:
    loading_table = pd.DataFrame(
        result.loadings,
        index=pd.Index(config.factor_features, name="feature"),
        columns=pd.Index(factor_columns(config), name="factor"),
    )
    loading_table.to_csv(config.out_dir / "factor_loadings.csv")
    return loading_table


def save_factor_communalities(
    result: FactorModelResult,
    config: Config,
) -> None:
    communalities = (result.loadings**2).sum(axis=1)
    table = pd.DataFrame(
        {
            "feature": config.factor_features,
            "communality": communalities,
            "uniqueness": result.uniqueness,
        }
    )
    table.to_csv(config.out_dir / "factor_communalities.csv", index=False)


def save_factor_scores(
    df: pd.DataFrame,
    result: FactorModelResult,
    config: Config,
) -> None:
    score_table = df.loc[:, list(config.id_columns)].copy()
    for factor_idx, factor_name in enumerate(factor_columns(config)):
        score_table[factor_name] = result.scores[:, factor_idx]
    score_table.to_csv(config.out_dir / "factor_scores.csv", index=False)


def save_factor_heatmap(
    loading_table: pd.DataFrame,
    config: Config,
) -> None:
    display_table = loading_table.copy()
    display_table.index = [
        textwrap.fill(str(feature).replace("_", " "), width=42)
        for feature in display_table.index
    ]

    height = max(7.0, 0.7 * len(config.factor_features))
    plt.figure(figsize=(12, height))
    sns.heatmap(
        display_table,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        cbar_kws={"shrink": 0.8},
    )
    rotation_name = config.rotation or "unrotated"
    plt.title(f"{rotation_name.title()} Factor Loadings")
    plt.tight_layout()
    plt.savefig(config.out_dir / "factor_loadings_heatmap.png", dpi=180)
    plt.close()


def save_factor_scree_plot(
    x_scaled: np.ndarray,
    config: Config,
) -> None:
    correlation = np.corrcoef(x_scaled, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    explained_variance = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(explained_variance)
    component_numbers = np.arange(1, len(eigenvalues) + 1)

    figure, eigenvalue_axis = plt.subplots(figsize=(11, 6))
    eigenvalue_axis.plot(
        component_numbers,
        eigenvalues,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Eigenvalue",
    )
    eigenvalue_axis.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Kaiser threshold",
    )
    eigenvalue_axis.set_xlabel("Number of factors")
    eigenvalue_axis.set_ylabel("Correlation-matrix eigenvalue")
    tick_step = max(1, len(component_numbers) // 15)
    eigenvalue_axis.set_xticks(component_numbers[::tick_step])
    eigenvalue_axis.grid(axis="y", alpha=0.25)

    variance_axis = eigenvalue_axis.twinx()
    variance_axis.plot(
        component_numbers,
        cumulative_variance,
        color="tab:orange",
        linewidth=2,
        label="Cumulative explained variance",
    )
    variance_axis.set_ylabel("Cumulative explained variance")
    variance_axis.set_ylim(0.0, 1.05)

    lines = eigenvalue_axis.lines + variance_axis.lines
    labels = [str(line.get_label()) for line in lines]
    eigenvalue_axis.legend(lines, labels, loc="center right")
    plt.title("Factor Scree Plot")
    figure.tight_layout()
    figure.savefig(config.out_dir / "factor_scree_plot.png", dpi=180)
    plt.close(figure)


def factor_structure_metrics(
    x_scaled: np.ndarray,
    result: FactorModelResult,
) -> dict[str, float]:
    absolute_loadings = np.abs(result.loadings)
    sorted_loadings = np.sort(absolute_loadings, axis=1)
    primary_loadings = sorted_loadings[:, -1]

    if result.loadings.shape[1] > 1:
        secondary_loadings = sorted_loadings[:, -2]
    else:
        secondary_loadings = np.zeros_like(primary_loadings)

    empirical_covariance = np.cov(x_scaled, rowvar=False, bias=True)
    model_covariance = result.loadings @ result.loadings.T + np.diag(result.uniqueness)
    residual = empirical_covariance - model_covariance

    return {
        "mean_primary_abs_loading": float(primary_loadings.mean()),
        "mean_cross_loading_gap": float((primary_loadings - secondary_loadings).mean()),
        "simple_feature_share": float(
            ((primary_loadings >= 0.40) & (secondary_loadings <= 0.30)).mean()
        ),
        "covariance_reconstruction_rmse": float(np.sqrt((residual**2).mean())),
    }


def efa_sample_warning(n_rows: int, config: Config) -> str:
    recommended_rows = 5 * len(config.factor_features)
    if n_rows < recommended_rows:
        return f"small sample for EFA; prefer at least {recommended_rows} rows"
    return ""


def factor_model_comparison_row(
    scope: str,
    cluster: int | str,
    x_scaled: np.ndarray,
    result: FactorModelResult,
    config: Config,
) -> dict[str, object]:
    return {
        "scope": scope,
        "cluster": cluster,
        "n_countries": x_scaled.shape[0],
        "rotation": config.rotation or "none",
        **factor_structure_metrics(x_scaled, result),
        "warning": efa_sample_warning(x_scaled.shape[0], config),
    }


def add_cluster_loadings(
    cluster: int,
    result: FactorModelResult,
    rows: list[dict[str, object]],
    config: Config,
) -> None:
    factor_names = factor_columns(config)
    for feature_idx, feature in enumerate(config.factor_features):
        row: dict[str, object] = {
            "cluster": cluster,
            "feature": feature,
        }
        for factor_idx, factor_name in enumerate(factor_names):
            row[factor_name] = result.loadings[feature_idx, factor_idx]
        rows.append(row)


def build_factor_model_comparison(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    global_factor_values: pd.DataFrame,
    global_x_scaled: np.ndarray,
    global_result: FactorModelResult,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = [
        factor_model_comparison_row(
            "global",
            "all",
            global_x_scaled,
            global_result,
            config,
        )
    ]
    cluster_loading_rows: list[dict[str, object]] = []

    for cluster in sorted(np.unique(cluster_labels).tolist()):
        cluster_mask = cluster_labels == cluster
        cluster_x_scaled = standardize_features(global_factor_values.loc[cluster_mask])
        cluster_result = fit_factor_model(cluster_x_scaled, config)

        comparison_rows.append(
            factor_model_comparison_row(
                "cluster",
                int(cluster),
                cluster_x_scaled,
                cluster_result,
                config,
            )
        )
        add_cluster_loadings(
            int(cluster),
            cluster_result,
            cluster_loading_rows,
            config,
        )

    comparison = pd.DataFrame(comparison_rows)
    cluster_rows = comparison.loc[comparison["scope"] == "cluster"]
    metric_columns = [
        "mean_primary_abs_loading",
        "mean_cross_loading_gap",
        "simple_feature_share",
        "covariance_reconstruction_rmse",
    ]

    weighted_summary: dict[str, object] = {
        "scope": "within_clusters_weighted_mean",
        "cluster": "all",
        "n_countries": int(cluster_rows["n_countries"].sum()),
        "rotation": config.rotation or "none",
        "warning": "weighted summary of cluster-specific EFAs",
    }
    for metric in metric_columns:
        weighted_summary[metric] = float(
            np.average(
                cluster_rows[metric],
                weights=cluster_rows["n_countries"],
            )
        )

    comparison = pd.concat(
        [comparison, pd.DataFrame([weighted_summary])],
        ignore_index=True,
    )
    return comparison, pd.DataFrame(cluster_loading_rows)


def save_factor_model_comparison(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    global_factor_values: pd.DataFrame,
    global_x_scaled: np.ndarray,
    global_result: FactorModelResult,
    config: Config,
) -> pd.DataFrame:
    comparison, cluster_loadings = build_factor_model_comparison(
        df,
        cluster_labels,
        global_factor_values,
        global_x_scaled,
        global_result,
        config,
    )
    comparison.to_csv(
        config.out_dir / "factor_model_comparison.csv",
        index=False,
    )
    cluster_loadings.to_csv(
        config.out_dir / "factor_loadings_by_cluster.csv",
        index=False,
    )
    return comparison


def save_all_factor_outputs(
    df: pd.DataFrame,
    factor_values: pd.DataFrame,
    x_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    result: FactorModelResult,
    config: Config,
) -> pd.DataFrame:
    loading_table = save_factor_loadings(result, config)
    save_factor_communalities(result, config)
    save_factor_scores(df, result, config)
    save_factor_heatmap(loading_table, config)
    save_factor_scree_plot(x_scaled, config)
    return save_factor_model_comparison(
        df,
        cluster_labels,
        factor_values,
        x_scaled,
        result,
        config,
    )

def rank_configs(
    config_sweep: pd.DataFrame,
    *,
    max_factors: int | None = None,
) -> pd.DataFrame:
    mask = (
        (config_sweep["status"] == "ok")
        & (config_sweep["clustered_status"] == "ok")
        & config_sweep["clustered_regularized_bic"].notna()
    )

    if max_factors is not None:
        mask &= (
            (config_sweep["clustered_unstable_models"] == 0)
            & (config_sweep["n_factors"] <= max_factors)
            & (config_sweep["n_factors"] <= config_sweep["smallest_cluster"])
        )

    return config_sweep.loc[mask].sort_values(
        [
            "clustered_regularized_bic",
            "clustered_unstable_models",
            "clustered_covariance_reconstruction_rmse",
            "clustered_mean_cross_loading_gap",
        ],
        ascending=[True, True, True, False],
    )

def print_best(label: str, row: pd.Series) -> None:
    print(label)
    print(f"  experiment_id: {row['experiment_id']}")
    print(f"  clustering features: {row['clustering_features']}")
    print(f"  n_clusters: {row['n_clusters']}")
    print(f"  n_factors: {row['n_factors']}")
    print(f"  clustered BIC: {row['clustered_regularized_bic']:.2f}")
    print(
        "  clustered - global BIC: "
        f"{row['clustered_minus_global_regularized_bic']:.2f}"
    )
