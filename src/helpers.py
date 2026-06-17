from config import CONFIG
import pandas as pd
import numpy as np

def save_cluster_assignments(df: pd.DataFrame, labels: np.ndarray, config) -> pd.DataFrame:
    assignments = df.loc[:, config.id_columns].copy()
    assignments["cluster"] = labels
    assignments.to_csv(config.out_dir / "cluster_assignments.csv", index=False)
    return assignments

def save_cluster_profiles(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    config,
) -> None:
    profiles = raw_features.copy()
    profiles["cluster"] = labels
    profiles.groupby("cluster").mean().sort_index().to_csv(
        config.out_dir / "cluster_profiles.csv"
    )

def save_income_crosstab(assignments: pd.DataFrame, config) -> None:
    counts = pd.crosstab(assignments["cluster"], assignments["world_bank_income_group"])
    counts.to_csv(config.out_dir / "cluster_income_crosstab.csv")

def save_all_cluster_outputs(
    df: pd.DataFrame,
    raw_features: pd.DataFrame,
    x_scaled: np.ndarray,
    labels: np.ndarray,
    config=CONFIG
) -> None:
    assignments = save_cluster_assignments(df, labels, config)
    save_cluster_profiles(raw_features, labels, config)
    save_income_crosstab(assignments, config)
