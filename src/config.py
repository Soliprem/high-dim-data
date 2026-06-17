from __future__ import annotations

from pathlib import Path

from datatypes import Config


ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG = Config[Path, tuple[str, ...]](
    data_path=ROOT_DIR / "data" / "WDI_2022_descriptive.csv",
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
    factor_features=(
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
    ),
    id_columns=(
        "country_name",
        "iso3_country_code",
        "world_bank_region",
        "world_bank_income_group",
        "observation_year",
    ),
)
