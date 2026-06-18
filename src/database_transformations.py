from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "WDI_World_2022.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "WDI_2022_analysis.csv"

COLUMN_RENAMES = {
    "country": "country_name",
    "iso3c": "iso3_country_code",
    "region": "world_bank_region",
    "income": "world_bank_income_group",
    "year": "observation_year",
    "BG.GSR.NFSV.GD.ZS": "trade_in_services_percent_of_gdp",
    "BM.GSR.GNFS.CD": "imports_of_goods_and_services_bop_current_usd",
    "BM.GSR.INSF.ZS": (
        "insurance_and_financial_services_percent_of_service_imports_bop"
    ),
    "BM.GSR.TOTL.CD": (
        "imports_of_goods_services_and_primary_income_bop_current_usd"
    ),
    "BM.GSR.TRAN.ZS": "transport_services_percent_of_service_imports_bop",
    "BM.GSR.TRVL.ZS": "travel_services_percent_of_service_imports_bop",
    "BM.KLT.DINV.CD.WD": (
        "foreign_direct_investment_net_outflows_bop_current_usd"
    ),
    "BM.KLT.DINV.WD.GD.ZS": (
        "foreign_direct_investment_net_outflows_percent_of_gdp"
    ),
    "BN.GSR.GNFS.CD": "net_trade_in_goods_and_services_bop_current_usd",
    "BN.KLT.DINV.CD": "foreign_direct_investment_net_bop_current_usd",
    "BN.KLT.PTXL.CD": "portfolio_investment_net_bop_current_usd",
    "BX.GSR.GNFS.CD": "exports_of_goods_and_services_bop_current_usd",
    "BX.GSR.INSF.ZS": (
        "insurance_and_financial_services_percent_of_service_exports_bop"
    ),
    "BX.GSR.TOTL.CD": (
        "exports_of_goods_services_and_primary_income_bop_current_usd"
    ),
    "BX.GSR.TRAN.ZS": "transport_services_percent_of_service_exports_bop",
    "BX.GSR.TRVL.ZS": "travel_services_percent_of_service_exports_bop",
    "BX.KLT.DINV.CD.WD": (
        "foreign_direct_investment_net_inflows_bop_current_usd"
    ),
    "BX.KLT.DINV.WD.GD.ZS": (
        "foreign_direct_investment_net_inflows_percent_of_gdp"
    ),
    "DT.DOD.DSTC.XP.ZS": (
        "short_term_debt_percent_of_exports_of_goods_services_and_primary_income"
    ),
    "DT.ODA.ODAT.MP.ZS": (
        "net_oda_received_percent_of_imports_of_goods_services_and_primary_income"
    ),
    "DT.TDS.DECT.EX.ZS": (
        "total_debt_service_percent_of_exports_of_goods_services_and_primary_income"
    ),
    "DT.TDS.DPPF.XP.ZS": (
        "debt_service_ppg_and_imf_only_percent_of_exports_of_goods_services_and_"
        "primary_income"
    ),
    "DT.TDS.DPPG.XP.ZS": (
        "public_and_publicly_guaranteed_debt_service_percent_of_exports_of_goods_"
        "services_and_primary_income"
    ),
    "FP.CPI.TOTL.ZG": "inflation_consumer_prices_annual_percent",
    "GC.NFN.TOTL.CN": "net_investment_in_nonfinancial_assets_current_lcu",
    "GC.NFN.TOTL.GD.ZS": (
        "net_investment_in_nonfinancial_assets_percent_of_gdp"
    ),
    "GC.TAX.GSRV.CN": "taxes_on_goods_and_services_current_lcu",
    "GC.TAX.GSRV.RV.ZS": "taxes_on_goods_and_services_percent_of_revenue",
    "GC.TAX.GSRV.VA.ZS": (
        "taxes_on_goods_and_services_percent_value_added_of_industry_and_services"
    ),
    "GC.XPN.GSRV.CN": "goods_and_services_expense_current_lcu",
    "GC.XPN.GSRV.ZS": "goods_and_services_expense_percent_of_expense",
    "IQ.SPI.PIL2": (
        "statistical_performance_indicators_spi_pillar_2_data_services_score_"
        "scale_0_100"
    ),
    "LP.LPI.LOGS.XQ": (
        "logistics_performance_index_competence_and_quality_of_logistics_services_"
        "one_equals_low_to_five_equals_high"
    ),
    "NE.EXP.GNFS.CD": "exports_of_goods_and_services_current_usd",
    "NE.EXP.GNFS.CN": "exports_of_goods_and_services_current_lcu",
    "NE.EXP.GNFS.KD": "exports_of_goods_and_services_constant_2015_usd",
    "NE.EXP.GNFS.KD.ZG": "exports_of_goods_and_services_annual_percent_growth",
    "NE.EXP.GNFS.KN": "exports_of_goods_and_services_constant_lcu",
    "NE.EXP.GNFS.ZS": "exports_of_goods_and_services_percent_of_gdp",
    "NE.IMP.GNFS.CD": "imports_of_goods_and_services_current_usd",
    "NE.IMP.GNFS.CN": "imports_of_goods_and_services_current_lcu",
    "NE.IMP.GNFS.KD": "imports_of_goods_and_services_constant_2015_usd",
    "NE.IMP.GNFS.KD.ZG": "imports_of_goods_and_services_annual_percent_growth",
    "NE.IMP.GNFS.KN": "imports_of_goods_and_services_constant_lcu",
    "NE.IMP.GNFS.ZS": "imports_of_goods_and_services_percent_of_gdp",
    "NE.RSB.GNFS.CD": "external_balance_on_goods_and_services_current_usd",
    "NE.RSB.GNFS.CN": "external_balance_on_goods_and_services_current_lcu",
    "NE.RSB.GNFS.KN": "external_balance_on_goods_and_services_constant_lcu",
    "NE.RSB.GNFS.ZS": (
        "external_balance_on_goods_and_services_percent_of_gdp"
    ),
    "NV.SRV.EMPL.KD": "services_value_added_per_worker_constant_2015_usd",
    "NV.SRV.TOTL.CD": "services_value_added_current_usd",
    "NV.SRV.TOTL.CN": "services_value_added_current_lcu",
    "NV.SRV.TOTL.KD": "services_value_added_constant_2015_usd",
    "NV.SRV.TOTL.KD.ZG": "services_value_added_annual_percent_growth",
    "NV.SRV.TOTL.KN": "services_value_added_constant_lcu",
    "NV.SRV.TOTL.ZS": "services_value_added_percent_of_gdp",
    "NY.GDP.DEFL.KD.ZG": "inflation_gdp_deflator_annual_percent",
    "NY.GDP.DEFL.KD.ZG.AD": (
        "inflation_gdp_deflator_linked_series_annual_percent"
    ),
}

LOCAL_CURRENCY_INDICATORS = (
    "GC.NFN.TOTL.CN",
    "GC.TAX.GSRV.CN",
    "GC.XPN.GSRV.CN",
    "NE.EXP.GNFS.CN",
    "NE.EXP.GNFS.KN",
    "NE.IMP.GNFS.CN",
    "NE.IMP.GNFS.KN",
    "NE.RSB.GNFS.CN",
    "NE.RSB.GNFS.KN",
    "NV.SRV.TOTL.CN",
    "NV.SRV.TOTL.KN",
)

DROPPED_LOCAL_CURRENCY_FEATURES = tuple(
    COLUMN_RENAMES[indicator] for indicator in LOCAL_CURRENCY_INDICATORS
)

REPLACED_USD_FLOW_FEATURES = (
    "foreign_direct_investment_net_outflows_bop_current_usd",
    "foreign_direct_investment_net_bop_current_usd",
    "foreign_direct_investment_net_inflows_bop_current_usd",
    "net_trade_in_goods_and_services_bop_current_usd",
    "external_balance_on_goods_and_services_current_usd",
)

DERIVED_FDI_NET_PERCENT_FEATURE = (
    "foreign_direct_investment_net_bop_percent_of_gdp"
)


def derive_gdp_current_usd(values: pd.DataFrame) -> pd.Series:
    candidates = pd.DataFrame(
        {
            "exports": (
                values["exports_of_goods_and_services_current_usd"]
                / (values["exports_of_goods_and_services_percent_of_gdp"] / 100)
            ),
            "imports": (
                values["imports_of_goods_and_services_current_usd"]
                / (values["imports_of_goods_and_services_percent_of_gdp"] / 100)
            ),
            "services": (
                values["services_value_added_current_usd"]
                / (values["services_value_added_percent_of_gdp"] / 100)
            ),
        }
    ).replace([np.inf, -np.inf], np.nan)
    return candidates.bfill(axis=1).iloc[:, 0]


def transform_wdi_database(raw: pd.DataFrame) -> pd.DataFrame:
    expected_columns = set(COLUMN_RENAMES)
    actual_columns = set(raw.columns)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected:
        raise ValueError(
            "Unexpected WDI schema; "
            f"missing columns: {missing}; unexpected columns: {unexpected}"
        )

    transformed = raw.rename(columns=COLUMN_RENAMES)
    gdp_current_usd = derive_gdp_current_usd(transformed)
    transformed[DERIVED_FDI_NET_PERCENT_FEATURE] = (
        100
        * transformed["foreign_direct_investment_net_bop_current_usd"]
        / gdp_current_usd
    )
    return transformed.drop(
        columns=[
            *DROPPED_LOCAL_CURRENCY_FEATURES,
            *REPLACED_USD_FLOW_FEATURES,
        ]
    )


def transform_wdi_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    transformed = transform_wdi_database(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(output_path, index=False)
    return transformed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename WDI indicator codes and remove incomparable LCU levels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformed = transform_wdi_file(args.input, args.output)
    print(f"Wrote {len(transformed)} rows and {len(transformed.columns)} columns")
    print("Dropped local-currency features:")
    for feature in DROPPED_LOCAL_CURRENCY_FEATURES:
        print(f"  {feature}")
    print("Replaced current-USD flow features:")
    for feature in REPLACED_USD_FLOW_FEATURES:
        print(f"  {feature}")
    print(f"Derived feature: {DERIVED_FDI_NET_PERCENT_FEATURE}")


if __name__ == "__main__":
    main()
