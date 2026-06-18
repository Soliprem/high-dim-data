from __future__ import annotations

import csv
from dataclasses import replace
from itertools import product
from pathlib import Path

from datatypes import Config


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "WDI_2022_descriptive.csv"

ID_COLUMNS = (
    "country_name",
    "iso3_country_code",
    "world_bank_region",
    "world_bank_income_group",
    "observation_year",
)

CURRENT_FACTOR_FEATURES = (
    "foreign_direct_investment_net_inflows_percent_of_gdp",
    "foreign_direct_investment_net_outflows_percent_of_gdp",
    "trade_in_services_percent_of_gdp",
    "exports_of_goods_and_services_percent_of_gdp",
    "imports_of_goods_and_services_percent_of_gdp",
    "external_balance_on_goods_and_services_percent_of_gdp",
    "services_value_added_percent_of_gdp",
    "services_value_added_per_worker_constant_2015_usd",
    "logistics_performance_index_competence_and_quality_of_logistics_services_one_equals_low_to_five_equals_high",
    "statistical_performance_indicators_spi_pillar_2_data_services_score_scale_0_100",
    "inflation_consumer_prices_annual_percent",
)

with DATA_PATH.open(newline="") as data_file:
    dataset_columns = next(csv.reader(data_file))

ALL_FACTOR_FEATURES = tuple(
    column for column in dataset_columns if column not in ID_COLUMNS
)

FACTOR_FEATURE_OPTIONS = {
    "current": CURRENT_FACTOR_FEATURES,
    "all": ALL_FACTOR_FEATURES,
}

CONFIG = Config[Path, tuple[str, ...]](
    data_path=DATA_PATH,
    out_dir=ROOT_DIR / "outputs",
    seed=42,
    n_clusters=4,
    n_factors=2,
    kmeans_restarts=50,
    # we know that the XX.XXX.XXXX.XX codes are unambiguous and more consistent,
    # but neither of us ever worked with WDI data, and it was getting very confusing
    clustering_features=(
        "services_value_added_per_worker_constant_2015_usd",
        # "logistics_performance_index_competence_and_quality_of_logistics_services_one_equals_low_to_five_equals_high",
        # "statistical_performance_indicators_spi_pillar_2_data_services_score_scale_0_100",
    ),
    factor_features=FACTOR_FEATURE_OPTIONS["all"],
    id_columns=ID_COLUMNS,
    rotation="varimax",
    parallel_analysis_iterations=500,
    parallel_analysis_percentile=95.0,
)

CLUSTERING_FEATURE_SETS = (
    (
        "services_value_added_per_worker_constant_2015_usd",
    ),
    (
        "services_value_added_per_worker_constant_2015_usd",
        "logistics_performance_index_competence_and_quality_of_logistics_services_one_equals_low_to_five_equals_high",
    ),
    (
        "services_value_added_per_worker_constant_2015_usd",
        "logistics_performance_index_competence_and_quality_of_logistics_services_one_equals_low_to_five_equals_high",
        "statistical_performance_indicators_spi_pillar_2_data_services_score_scale_0_100",
    ),
)

EXPERIMENT_CONFIGS = tuple(
    replace(
        CONFIG,
        clustering_features=clustering_features,
        n_clusters=n_clusters,
        n_factors=n_factors,
    )
    for clustering_features, n_clusters, n_factors in product(
        CLUSTERING_FEATURE_SETS,
        range(2, 7),
        range(1, len(CONFIG.factor_features)),
    )
)
