from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_dataset(
    df: pd.DataFrame,
    *,
    id_columns: list[str],
    clustering_features: list[str],
    factor_features: list[str],
) -> None:
    required_columns = set(id_columns + clustering_features + factor_features)
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    years = sorted(df["year"].dropna().unique().tolist())
    if years != [2022]:
        raise ValueError(f"Expected only year 2022, got {years}")


def select_numeric_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    numeric = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.columns[numeric.notna().sum() == 0].tolist()
    if all_missing:
        raise ValueError(f"Features have no numeric values: {all_missing}")
    return numeric


def fill_missing_with_medians(values: pd.DataFrame) -> pd.DataFrame:
    medians = values.median(numeric_only=True)
    return values.fillna(medians).astype(float)


def log_transform_service_employment(values: pd.DataFrame) -> pd.DataFrame:
    feature = "NV.SRV.EMPL.KD"
    transformed = values.copy()

    if feature not in transformed.columns:
        return transformed

    min_value = transformed[feature].min()
    if min_value <= -1:
        raise ValueError(f"Cannot log1p {feature} because it contains values <= -1")

    transformed[feature] = np.log1p(transformed[feature])
    return transformed


def winsorize_extreme_values(values: pd.DataFrame) -> pd.DataFrame:
    lower_limits = values.quantile(0.01)
    upper_limits = values.quantile(0.99)
    return values.clip(lower=lower_limits, upper=upper_limits, axis=1)


def standardize_features(values: pd.DataFrame) -> np.ndarray:
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values")
    return StandardScaler().fit_transform(values)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    numeric = select_numeric_features(df, feature_names)
    imputed = fill_missing_with_medians(numeric)
    transformed = log_transform_service_employment(imputed)
    winsorized = winsorize_extreme_values(transformed)
    scaled = standardize_features(winsorized)
    return winsorized, scaled
