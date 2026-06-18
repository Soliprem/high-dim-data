from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

from datatypes import Config, FactorModelResult


def assign_kmeans_clusters(
    x_scaled: np.ndarray,
    config: Config,
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
    config: Config,
) -> FactorModelResult:
    max_factors = min(x_scaled.shape)
    if config.n_factors > max_factors:
        raise ValueError(
            f"Cannot fit {config.n_factors} factors to a matrix with shape "
            f"{x_scaled.shape}; maximum is {max_factors}"
        )

    model = FactorAnalysis(
        n_components=config.n_factors,
        rotation=config.rotation,
        random_state=config.seed,
    )
    scores = model.fit_transform(x_scaled)

    return FactorModelResult(
        loadings=np.asarray(model.components_.T, dtype=float),
        uniqueness=np.asarray(model.noise_variance_, dtype=float),
        scores=np.asarray(scores, dtype=float),
    )
