import pandas as pd
from pathlib import Path

DATA_PATH = Path("../data/WDI_World_2022.csv")
OUT_DIR = Path("../outputs")
SEED = 42
N_CLUSTERS = range(10)
N_FACTORS = range(10)

CLUSTERING_FETURES = []
FACTOR_FEATURES = []

def main():
    df = pd.read_csv(DATA_PATH)
if __name__ == "__main__":
    main()
