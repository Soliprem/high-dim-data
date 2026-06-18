from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np


PathT = TypeVar("PathT", bound=Path)
FeaturesT = TypeVar("FeaturesT", bound=Sequence[str])
Rotation = Literal["varimax", "quartimax"] | None


@dataclass(frozen=True)
class Config(Generic[PathT, FeaturesT]):
    data_path: PathT
    out_dir: PathT

    seed: int
    n_clusters: int
    n_factors: int
    kmeans_restarts: int

    clustering_features: FeaturesT
    factor_features: FeaturesT
    id_columns: FeaturesT

    rotation: Rotation


@dataclass(frozen=True)
class FactorModelResult:
    loadings: np.ndarray
    uniqueness: np.ndarray
    scores: np.ndarray
