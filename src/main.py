import pandas as pd
from sklearn.metrics import silhouette_score

from analysis import assign_kmeans_clusters, fit_factor_model
from config import CONFIG, EXPERIMENT_CONFIGS
from data_prep import validate_dataset, prepare_feature_matrix
from datatypes import Config
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
    save_all_cluster_outputs(df, raw_cluster_features, x_cluster, cluster_labels, config)

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

    global_metrics = factor_comparison.loc[
        factor_comparison["scope"] == "global"
    ].iloc[0]
    clustered_metrics = factor_comparison.loc[
        factor_comparison["scope"] == "within_clusters_weighted_mean"
    ].iloc[0]

    config_sweep = run_config_sweep(df, EXPERIMENT_CONFIGS)
    config_sweep.to_csv(config.out_dir / "config_sweep.csv", index=False)

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
    print(f"Config experiments: {len(config_sweep)}")
    print(f"Outputs written to {config.out_dir}")


if __name__ == "__main__":
    main()
