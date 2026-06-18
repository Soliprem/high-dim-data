import pandas as pd
from sklearn.metrics import silhouette_score

from adaptive import run_adaptive_regime
from analysis import assign_kmeans_clusters, fit_factor_model
from config import (
    CLUSTERING_CONFIGS,
    CONFIG,
    EXPERIMENT_CONFIGS,
    PA_REGIME_CONFIG,
)
from data_prep import validate_dataset, prepare_feature_matrix
from datatypes import Config
from model_selection import (
    run_parallel_analysis_diagnostics,
    run_parallel_analysis_sweep,
)
from experiments import run_config_sweep
from helpers import save_all_cluster_outputs, save_all_factor_outputs


def main(config: Config = CONFIG) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.data_path)
    validate_dataset(
        df,
        config,
    )
    raw_cluster_features, x_cluster = prepare_feature_matrix(
        df,
        config.clustering_features,
    )
    cluster_labels = assign_kmeans_clusters(
        x_cluster,
        config,
    )
    cluster_score = silhouette_score(x_cluster, cluster_labels)
    save_all_cluster_outputs(
        df, raw_cluster_features, x_cluster, cluster_labels, config
    )

    factor_values, x_factor = prepare_feature_matrix(df, config.factor_features)
    factor_result = fit_factor_model(x_factor, config)
    factor_comparison = save_all_factor_outputs(
        df,
        factor_values,
        x_factor,
        cluster_labels,
        factor_result,
        config,
    )

    global_metrics = factor_comparison.loc[factor_comparison["scope"] == "global"].iloc[
        0
    ]
    clustered_metrics = factor_comparison.loc[
        factor_comparison["scope"] == "within_clusters_weighted_mean"
    ].iloc[0]

    parallel_analysis_summary = run_parallel_analysis_diagnostics(
        factor_values,
        cluster_labels,
        config,
    )
    parallel_analysis_sweep = run_parallel_analysis_sweep(
        df,
        factor_values,
        CLUSTERING_CONFIGS,
        config.out_dir / "diagnostics" / "parallel_analysis",
    )
    adaptive_regime_summary = run_adaptive_regime(
        df,
        x_factor,
        parallel_analysis_sweep,
        PA_REGIME_CONFIG,
    )

    config_sweep = run_config_sweep(df, EXPERIMENT_CONFIGS)
    config_sweep.to_csv(config.out_dir / "config_sweep.csv", index=False)
    ranked_config_sweep = config_sweep.loc[
        (config_sweep["status"] == "ok")
        & (config_sweep["clustered_status"] == "ok")
        & config_sweep["clustered_regularized_bic"].notna()
    ].sort_values(
        [
            "clustered_regularized_bic",
            "clusters_below_efa_rule",
            "clustered_covariance_reconstruction_rmse",
        ],
        ascending=[True, True, True],
    )

    ranked_config_sweep.to_csv(
        config.out_dir / "config_sweep_ranked_by_bic.csv",
        index=False,
    )

    best_bic = ranked_config_sweep.iloc[0]

    print("Best clustered BIC specification:")
    print(f"  experiment_id: {best_bic['experiment_id']}")
    print(f"  clustering features: {best_bic['clustering_features']}")
    print(f"  n_clusters: {best_bic['n_clusters']}")
    print(f"  n_factors: {best_bic['n_factors']}")
    print(f"  clustered BIC: {best_bic['clustered_regularized_bic']:.2f}")
    print(
        "  clustered - global BIC: "
        f"{best_bic['clustered_minus_global_regularized_bic']:.2f}"
    )

    print(f"Rows: {len(df)}")
    print(f"Clustering features: {raw_cluster_features.shape[1]}")
    print(f"Clusters: {sorted(set(cluster_labels.tolist()))}")
    print(f"Silhouette score: {cluster_score:.4f}")
    print(f"Factor features: {x_factor.shape[1]}")
    print(
        "Global EFA reconstruction RMSE: "
        f"{global_metrics['covariance_reconstruction_rmse']:.4f}"
    )
    print(
        "Clustered EFA weighted RMSE: "
        f"{clustered_metrics['covariance_reconstruction_rmse']:.4f}"
    )
    global_pa_factors = parallel_analysis_summary.loc[
        parallel_analysis_summary["scope"] == "global",
        "suggested_factors",
    ].iloc[0]
    print(f"Global PA suggested factors: {global_pa_factors}")
    print(
        "PA clustering specifications: "
        f"{parallel_analysis_sweep['specification_id'].max()}"
    )
    adaptive_factors = adaptive_regime_summary.loc[
        adaptive_regime_summary["model"] == "adaptive_aggregate",
        "n_factors",
    ].iloc[0]
    print(f"Adaptive regime factors: {adaptive_factors}")
    print(f"Config experiments: {len(config_sweep)}")
    print(f"Outputs written to {config.out_dir}")


if __name__ == "__main__":
    main()
