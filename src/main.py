import pandas as pd
from analysis import assign_kmeans_clusters, fit_factor_model
from config import CONFIG
from data_prep import validate_dataset, prepare_feature_matrix
from sklearn.metrics import silhouette_score
from datatypes import Config
from helpers import save_all_cluster_outputs


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

    _, x_factor = prepare_feature_matrix(df, config.factor_features)
    fit_factor_model(x_factor, config)

    print(f"Rows: {len(df)}")
    print(f"Clustering features: {raw_cluster_features.shape[1]}")
    print(f"Clusters: {sorted(set(cluster_labels.tolist()))}")
    print(f"silhouette_score: {cluster_score}")
    print(f"factor features: {x_factor.shape[1]}")


if __name__ == "__main__":
    main()
