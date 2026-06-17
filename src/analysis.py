from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis


def assign_kmeans_clusters(
    x_scaled: np.ndarray,
    config,
) -> np.ndarray:
    model = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.seed,
        n_init=cast(Any, config.kmeans_restarts),
    )
    # For readability in output, we name the clusters starting from 1 rather than 0
    return model.fit_predict(x_scaled) + 1


def fit_factor_model(
    x_scaled: np.ndarray,
    config
):
    model = FactorAnalysis(
        n_components=config.n_factors,
        rotation=None,
        random_state=config.seed,
    )
    model.fit(x_scaled)
    return model.components_.T
