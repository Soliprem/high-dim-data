from __future__ import annotations

from pathlib import Path

from datatypes import Config


ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG = Config[Path, tuple[str, ...]](
    data_path=ROOT_DIR / "data" / "WDI_World_2022.csv",
    out_dir=ROOT_DIR / "outputs",
    seed=42,
    n_clusters=4,
    n_factors=2,
    kmeans_restarts=50,
    clustering_features=(
        "BG.GSR.NFSV.GD.ZS",
        "BM.GSR.INSF.ZS",
        "BM.GSR.TRAN.ZS",
        "BM.GSR.TRVL.ZS",
        "BM.KLT.DINV.WD.GD.ZS",
        "BX.GSR.INSF.ZS",
        "BX.GSR.TRAN.ZS",
        "BX.GSR.TRVL.ZS",
        "BX.KLT.DINV.WD.GD.ZS",
        "DT.DOD.DSTC.XP.ZS",
        "DT.ODA.ODAT.MP.ZS",
        "DT.TDS.DECT.EX.ZS",
        "DT.TDS.DPPF.XP.ZS",
        "DT.TDS.DPPG.XP.ZS",
        "FP.CPI.TOTL.ZG",
        "GC.NFN.TOTL.GD.ZS",
        "GC.TAX.GSRV.RV.ZS",
        "GC.TAX.GSRV.VA.ZS",
        "GC.XPN.GSRV.ZS",
        "IQ.SPI.PIL2",
        "LP.LPI.LOGS.XQ",
        "NE.EXP.GNFS.KD.ZG",
        "NE.EXP.GNFS.ZS",
        "NE.IMP.GNFS.KD.ZG",
        "NE.IMP.GNFS.ZS",
        "NE.RSB.GNFS.ZS",
        "NV.SRV.EMPL.KD",
        "NV.SRV.TOTL.KD.ZG",
        "NV.SRV.TOTL.ZS",
        "NY.GDP.DEFL.KD.ZG",
    ),
    factor_features=(
        "BX.KLT.DINV.WD.GD.ZS",
        "BM.KLT.DINV.WD.GD.ZS",
        "BG.GSR.NFSV.GD.ZS",
        "NE.EXP.GNFS.ZS",
        "NE.IMP.GNFS.ZS",
        "NE.RSB.GNFS.ZS",
        "NV.SRV.TOTL.ZS",
        "NV.SRV.EMPL.KD",
        "LP.LPI.LOGS.XQ",
        "IQ.SPI.PIL2",
        "FP.CPI.TOTL.ZG",
    ),
    id_columns=(
        "country",
        "iso3c",
        "region",
        "income",
        "year",
    ),
)
