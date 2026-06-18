from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_dataset(
    df: pd.DataFrame,
    config,
) -> None:
    required_columns = {
        *config.id_columns,
        *config.clustering_features,
        *config.factor_features,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    years = sorted(df["observation_year"].dropna().unique().tolist())
    if years != [2022]:
        raise ValueError(f"Expected only year 2022, got {years}")


def select_numeric_features(
    df: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    numeric = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.columns[numeric.notna().sum() == 0].tolist()
    if all_missing:
        raise ValueError(f"Features have no numeric values: {all_missing}")
    return numeric


# We fill missing values with their respective feature medians to avoid shedding most of the data
def fill_missing_with_feature_medians(values: pd.DataFrame) -> pd.DataFrame:
    medians = values.median(numeric_only=True)
    return values.fillna(medians).astype(float)


def log_transform_service_employment(values: pd.DataFrame) -> pd.DataFrame:
    transformed = values.copy()
    feature = "services_value_added_per_worker_constant_2015_usd"
    if feature not in transformed.columns:
        return transformed
    min_value = transformed[feature].min()
    if min_value <= -1:
        raise ValueError(f"Cannot log1p {feature} because it contains values <= -1")
    transformed[feature] = np.log1p(transformed[feature])
    return transformed


def winsorize_extreme_values(values: pd.DataFrame) -> pd.DataFrame:
    lower = values.quantile(0.01)
    upper = values.quantile(0.99)
    return values.clip(lower=lower, upper=upper, axis=1)


def standardize_features(values: pd.DataFrame) -> np.ndarray:
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values")
    return StandardScaler().fit_transform(values)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    winsorized = (
        select_numeric_features(df, feature_names)
        .pipe(fill_missing_with_feature_medians)
        .pipe(log_transform_service_employment)
        .pipe(winsorize_extreme_values)
    )
    scaled = standardize_features(winsorized)
    return winsorized, scaled
