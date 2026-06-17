from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans


def assign_kmeans_clusters(
    x_scaled: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    kmeans_restarts: int,
) -> np.ndarray:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=cast(Any, kmeans_restarts),
    )
    # For readability in output, we name the clusters starting from 1 rather than 0
    return model.fit_predict(x_scaled) + 1
