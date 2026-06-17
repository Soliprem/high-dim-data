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
    clustering_features=(
        "trade_in_services_percent_of_gdp",
        "insurance_and_financial_services_percent_of_service_imports_bop",
        "transport_services_percent_of_service_imports_bop",
        "travel_services_percent_of_service_imports_bop",
        "foreign_direct_investment_net_outflows_percent_of_gdp",
        "insurance_and_financial_services_percent_of_service_exports_bop",
        "transport_services_percent_of_service_exports_bop",
        "travel_services_percent_of_service_exports_bop",
        "foreign_direct_investment_net_inflows_percent_of_gdp",
        "short_term_debt_percent_of_exports_of_goods_services_and_primary_income",
        "net_oda_received_percent_of_imports_of_goods_services_and_primary_income",
        "total_debt_service_percent_of_exports_of_goods_services_and_primary_income",
        "debt_service_ppg_and_imf_only_percent_of_exports_of_goods_services_and_primary_income",
        "public_and_publicly_guaranteed_debt_service_percent_of_exports_of_goods_services_and_primary_income",
        "inflation_consumer_prices_annual_percent",
        "net_investment_in_nonfinancial_assets_percent_of_gdp",
        "taxes_on_goods_and_services_percent_of_revenue",
        "taxes_on_goods_and_services_percent_value_added_of_industry_and_services",
        "goods_and_services_expense_percent_of_expense",
        "statistical_performance_indicators_spi_pillar_2_data_services_score_scale_0_100",
        "logistics_performance_index_competence_and_quality_of_logistics_services_one_equals_low_to_five_equals_high",
        "exports_of_goods_and_services_annual_percent_growth",
        "exports_of_goods_and_services_percent_of_gdp",
        "imports_of_goods_and_services_annual_percent_growth",
        "imports_of_goods_and_services_percent_of_gdp",
        "external_balance_on_goods_and_services_percent_of_gdp",
        "services_value_added_per_worker_constant_2015_usd",
        "services_value_added_annual_percent_growth",
        "services_value_added_percent_of_gdp",
        "inflation_gdp_deflator_annual_percent",
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
