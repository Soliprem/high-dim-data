# Clustering and Factor Analysis of WDI 2022 Data

This project investigates whether clustering countries before exploratory
factor analysis improves model quality for a high-dimensional World Development
Indicators dataset.

## Environment

The repository provides a Nix development shell containing Python and the
required analysis libraries:

```sh
nix develop
```
The main dependencies are NumPy, pandas, SciPy, scikit-learn, Matplotlib,
Seaborn, and tqdm.

## Reproducing the analysis

### 1. Transform the source database

```sh
python src/database_transformations.py
```

This reads `data/WDI_World_2022.csv` and generates
`data/WDI_2022_analysis.csv`.

The transformation:

- replaces WDI indicator codes with descriptive names;
- removes 11 local-currency level variables that are not comparable across
  countries;
- replaces selected current-USD flows with GDP-normalized alternatives;
- derives net FDI as a percentage of GDP.

### 2. Run the full analysis

```sh
nix develop -c python src/main.py
```

This runs the clustering diagnostics, factor models, parallel-analysis sweep,
adaptive regime, and fixed-factor configuration sweep. The complete run is
computationally intensive and may take several minutes.

### 3. Compile the report

With [Typst](https://typst.app/) installed:

```sh
typst compile writeup.typ writeup.pdf
```

## Important outputs

- `outputs/config_sweep.csv`: all fixed-factor experiments.
- `outputs/config_sweep_ranked_by_bic.csv`: unrestricted BIC ranking.
- `outputs/config_sweep_ranked_by_bic_admissible.csv`: stable ranking subject
  to the global parallel-analysis factor cap.
- `outputs/diagnostics/parallel_analysis/`: global, cluster, and sweep
  diagnostics.
- `outputs/adaptive_regime/model_summary.csv`: adaptive-regime likelihood and
  stability results.
- `outputs/factor_loadings.csv`: loadings for the default global model.
- `outputs/factor_model_comparison.csv`: default global and clustered
  diagnostics.
- `outputs/cluster_assignments.csv`: default cluster assignments.

## Methodological notes

- Clustered BIC is computed conditionally on fixed k-means assignments (more details in the writeup)
- Near-singular models are identified through uniqueness estimates below the
  likelihood eigenvalue floor.
- The unrestricted ranking contains unstable high-factor models and should not
  be used as the final model ranking.
- Parallel analysis uses 500 permutations and the 95th percentile of the
  permuted eigenvalues.
- Cluster-specific factor numbers and signs are not directly comparable because
  each cluster model is fitted independently.

For more details, check `writeup`

## Repository structure

```text
data/                         Source and transformed datasets
src/database_transformations.py
                              Database renaming and comparability filtering
src/data_prep.py              Imputation and numerical transformations
src/analysis.py               K-means and factor-model fitting
src/model_selection.py        Parallel analysis and likelihood criteria
src/experiments.py            Fixed-factor configuration sweep
src/adaptive.py               Adaptive parallel-analysis regime
src/main.py                   Full analysis entry point
outputs/                      Generated tables, diagnostics, and figures
writeup.typ                   Project report
```
