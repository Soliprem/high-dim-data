import pandas as pd
from analysis import assign_kmeans_clusters
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
        id_columns=config.id_columns,
        clustering_features=config.clustering_features,
        factor_features=config.factor_features,
    )
    raw_cluster_features, x_cluster = prepare_feature_matrix(
        df,
        config.clustering_features,
    )
    cluster_labels = assign_kmeans_clusters(
        x_cluster,
        n_clusters=config.n_clusters,
        seed=config.seed,
        kmeans_restarts=config.kmeans_restarts,
    )
    cluster_score = silhouette_score(x_cluster, cluster_labels)
    save_all_cluster_outputs(df, raw_cluster_features, x_cluster, cluster_labels, config)

    print(f"Rows: {len(df)}")
    print(f"Clustering features: {raw_cluster_features.shape[1]}")
    print(f"Clusters: {sorted(set(cluster_labels.tolist()))}")
    print(f"silhouette_score: {cluster_score}")


if __name__ == "__main__":
    main()
