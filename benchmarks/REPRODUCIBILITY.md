# Reproducing the Paper

This document gives the exact pixi commands to reproduce every artifact that appears in the paper: the regression and classification benchmarks, the conditional-diagnostics analysis, the appendix studies, and the final PDF build. Reviewers are not expected to rerun the full benchmark suite; the result files that back the paper's tables and plots are tracked in the repository under `benchmarks/results/`. The commands below exist so that the pipeline can be re-executed end-to-end if desired.

A full rerun on a single workstation takes hours for the headline regression/classification suites and days for the appendix studies (convergence rates, distribution misspecification, sample-size sweeps, noise-features). All real-data commands are deterministic: the global seed is `1234`, the fold count is `10` (tune on fold 0, evaluate on folds 1–9), and the tuning metric is CRPS for regression / log-loss for classification. The main synthetic DGP benchmark uses seed `42` with the same fold convention and 50 Optuna trials per model/DGP.

## Prerequisites

- [`pixi`](https://pixi.sh/) (the locked dependency manager; everything else is installed from `pixi.toml` / `pixi.lock`).
- A Rust toolchain (for the BDF split-finding extension). One-time setup:

  ```bash
  pixi run setup        # builds the Rust extension and installs pre-commit hooks
  ```

The exact dependency versions are pinned in `pixi.lock`, which is checked into the repository. The two environments used below are defined in `pixi.toml`:

| Environment | Used for | Notable contents |
|---|---|---|
| `default` | BDF, sklearn baselines, paper build, tests, lint | `bdf`, sklearn, matplotlib, hydra, optuna, scoringrules |
| `benchmark` | External baselines (NGBoost, LightGBM, XGBoostLSS, CatBoost, QRF, DRF, ConformalRF, BART, Treeffuser) | adds `lightgbm`, `xgboost`, `xgboostlss`, `ngboost`, `catboost`, `quantile-forest`, `rpy2`, `pymc-bart`, `treeffuser`, `torch` |

The deprecated `bartpy` dependency is intentionally not part of the benchmark environment; the maintained PyMC-BART wrapper is retained for supplementary BART runs.

## Pipeline Overview

```
fetch-data ──► process-data ──► run benchmark suites ──► aggregate/plot ──► paper
                                  │
                                  └─► conditional-diagnostics (two-pass) ──► merged plots/tables
```

Each stage is a pixi task; nothing requires bare `python -m …` invocations.

## 1. Data

UCI/OpenML datasets are downloaded, not bundled. The repository intentionally provides fetch/process scripts and tracked result artifacts rather than redistributing raw or processed benchmark datasets. The two-step pipeline writes processed numeric matrices under `data/processed/` together with `*.meta.json` provenance files that record the source, processed sample/feature counts, and the target domain.

```bash
pixi run fetch-data           # downloads raw files (one-time, network required)
pixi run process-data         # produces data/processed/*.parquet and *.meta.json
```

The benchmark inventory and source per dataset is listed in `paper/appendix/experimental_setup.tex` (Table~\ref{tab:dataset-inventory}).

## 2. Main Benchmarks

### Regression suite

```bash
pixi run reg-suite                                  # BDF + sklearn baselines (default env)
pixi run -e benchmark reg-suite-models              # NGBoost / LightGBM / XGBoostLSS / CatBoost / QRF / DRF / ConformalRF / BART
pixi run reg                                        # aggregate, run Friedman/Nemenyi, render plots and LaTeX tables
```

Results are written to `benchmarks/results/regression/res_*.yaml` (one file per model). Aggregated tables and critical-difference diagrams land in `benchmarks/plots/regression/` and `benchmarks/tables/regression/`. The `pixi run reg` step copies the figures and `.tex` snippets used in the paper into `paper/plots/` and `paper/tables/`.

The TMLR revision uses the `score_regime` protocol for canonical BDF configs. Effective regimes expand before model construction as `nle -> (score_method=nle, score_correction=None)`, `nll_bic -> (score_method=nll, score_correction=bic)`, and `nll -> (score_method=nll, score_correction=None)`; result YAMLs store the expanded fields for compatibility with existing table scripts. Canonical BDF-Full runs use one tuning budget per family, while fixed-regime configs are ablations and are not extra BDF-Full candidates.

Targeted revision reruns:

```bash
# Canonical BDF-Full candidates affected by score_regime / OOB-off protocol
pixi run reg-suite model=bdf_normalmunormal
pixi run reg-suite model=bdf_freqstudentt
pixi run reg-suite model=bdf_gammamvlambdapoisson

# Fixed-regime BDF-Core / leaf x score ablations
pixi run reg-suite model=bdf_normalmunormal_nle
pixi run reg-suite model=bdf_normalmunormal_nll
pixi run reg-suite model=bdf_normalmunormal_nll_bic
pixi run reg-suite model=bdf_freqstudentt_nll
pixi run reg-suite model=bdf_freqstudentt_nll_bic

# Classification Core-vs-Full check
pixi run clas-suite model=bdf_betamvbernoulli
pixi run clas-suite model=bdf_betamvbernoulli_nle
pixi run -e benchmark clas-suite-models model=callgbm_clas
```

If Gamma--Exponential is added to any positive-outcome appendix aggregate, run `pixi run reg-suite model=bdf_gammamvlambdaexponential` under the same protocol before reporting it.

### Classification suite

```bash
pixi run clas-suite                                 # BDF + sklearn baselines
pixi run -e benchmark clas-suite-models             # NGBoost / LightGBM / CalRF / CalLGBM / KNN
pixi run -e benchmark clas-suite-models model=callgbm_clas
pixi run clas                                       # aggregate + plot + table
```

Results: `benchmarks/results/classification/res_*.yaml`. Both suites use Hydra; passing `model=<config>` to `bench-bdf` / `bench-models` runs a single model rather than the whole sweep.

### Single-model spot run

```bash
pixi run bench-bdf model=bdf_normalmunormal                 # default env
pixi run -e benchmark bench-models model=qrf                # benchmark env
pixi run -e benchmark bench-models model=drf                # benchmark env, requires R package `drf`
```

XGBoostLSS is evaluated as four separate distributional variants and fused during aggregation by fold-0 tuning CRPS among complete variants:

```bash
pixi run -e benchmark reg-suite-models -m \
    model=xgboostlss_gaussian,xgboostlss_studentt,xgboostlss_laplace,xgboostlss_gaussian_mixture
```

The aggregation step `pixi run reg` writes the selected XGBoostLSS distribution table and excludes incomplete variant-dataset pairs rather than imputing missing folds. In the locked real-data artifacts, the incomplete XGBoostLSS candidates are Gaussian on `yacht_hydrodynamics` (8/9 CRPS folds), Student-t on `energy_efficiency` (8/9), Student-t on `parkinsons_updrs` (8/9), and Student-t on `yacht_hydrodynamics` (6/9). All BDF, QRF, DRF, conformalized forest/boosting, NGBoost, CatBoost-UQ, Gaussian deep-ensemble, Bayesian-ridge, kNN-KDE, and classification files consumed by the main paper have complete reported folds for the metrics used there.

## 3. Conditional Diagnostics

The conditional-by-predicted-uncertainty analysis (appendix calibration plots) reuses the tuned hyperparameters stored in `benchmarks/results/regression/`. Model imports are deliberately lazy: BDF must run in the `default` environment, external baselines must run in the `benchmark` environment. Each pass writes its own JSON; a third command merges them and produces the plots and LaTeX tables.

```bash
# Pass 1 — BDF in default env
pixi run python -m benchmarks.conditional_diagnostics_eval \
    --models BDF --env-tag default

# Pass 2 — external baselines in benchmark env
pixi run -e benchmark python -m benchmarks.conditional_diagnostics_eval \
    --models qrf conflgbm ngboost_reg catbunc_reg --env-tag benchmark

# Merge + plot + tables
pixi run python -m benchmarks.plot_conditional_diagnostics
```

The two passes drop JSON into `benchmarks/results/conditional_diagnostics/`; the plot script consumes every `conditional_*.json` in that directory and refuses to merge if the same `(model, dataset)` pair appears twice. DRF is covered in the real-data aggregate and synthetic DGP studies, but the locked conditional-diagnostics appendix does not use DRF-specific reliability claims.

## 4. Appendix Studies

| Study | Command | Output dir |
|---|---|---|
| Synthetic DGP benchmark | `pixi run synth-study`, `pixi run -e benchmark synth-study-models --models QRF --result-suffix forests`, `pixi run -e benchmark synth-study-models --models DRF --result-suffix drf`, then `pixi run synth-output` | `benchmarks/results/synthetic_dgp/` |
| Convergence rate study | `pixi run convergence` | `benchmarks/results/convergence_rate/` |
| Complexity analysis | `pixi run complexity` | `benchmarks/results/complexity_analysis/` |
| BDF parameter ablation | `pixi run effect-params` | `benchmarks/results/effect_bdf_params/` |
| Scoring-method study | `pixi run scoring-study` | `benchmarks/results/scoring_method/` |
| Distribution misspecification | `pixi run dist-study` then `pixi run dist-output` | `benchmarks/results/dist_misspecification/` |
| OOD / covariate shift | `pixi run ood` | `benchmarks/results/effect_ood/` |
| Effect of sample size | `pixi run sample-size` | `benchmarks/results/effect_sample_size/` |
| Effect of n_trees | `pixi run ensemble-size-study` then `pixi run ensemble-size-output` | `benchmarks/results/effect_n_trees/` |
| Noise-features robustness | `pixi run noise-study` then `pixi run -e benchmark noise-study-models` then `pixi run noise-output` | `benchmarks/results/effect_noise_features/` |
| Tree prior comparison | `pixi run effect-tree-prior` | `benchmarks/results/effect_tree_prior/` |
| Label noise (clas.) | `pixi run label-noise-clas` | `benchmarks/results/label_noise/` |
| Beta prior effect (clas.) | `pixi run prior-clas` | `benchmarks/results/prior_effects/` |
| Conformalization | `pixi run conform` then `pixi run conform-output` | `benchmarks/results/conformalization/` |

Each study script also writes an auto-generated Markdown report alongside its results. The main synthetic DGP definitions are in `benchmarks/pipeline/synthetic_dgps.py` and are summarized in the paper appendix. To run a subset, pass `--dgps heteroscedastic_sinusoidal heavy_tailed` or `--models BDFNormal QRF`; external models such as QRF and DRF require the `benchmark` environment.

## 5. Revision Tables

After the targeted BDF reruns finish, regenerate the TMLR revision tables:

```bash
pixi run python -m benchmarks.make_revision_bdf_tables
pixi run reg
pixi run clas
```

`benchmarks.make_revision_bdf_tables` reads only existing YAML result artifacts and writes `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` plus `bdf_leaf_score_ablation.tex`. It warns about missing result files and does not run models.

## 6. TMLR Artifact Manifest

After regenerating tables/plots and rebuilding the paper, write the locked submission manifest:

```bash
pixi run tmlr-manifest
```

This writes `benchmarks/results/TMLR_ARTIFACT_MANIFEST.md`. The manifest records the current `paper/main.pdf` hash, the locked real-data result YAML hashes, generated table/plot hashes consumed by the LaTeX build, the repository-local files listed in `paper/main.fls`, the real-data dataset lists, the deterministic split policy, and the regeneration commands. Split indices are not serialized separately; they are exactly reconstructed from the processed data order with `KFold(n_splits=10, shuffle=True, random_state=1234)` for regression and `StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)` for classification. Fold 0 is tuning only, and folds 1--9 are evaluation folds.

## 7. Paper Build

```bash
pixi run paper          # latexmk -pdf paper/main.tex
```

Produces `paper/main.pdf`. The locked build should have no undefined references or citations; known layout warnings from dense appendix tables and float-only pages should be inspected before final submission but do not indicate missing artifacts.

## 8. Result Provenance

Every result YAML written by the orchestrator (`benchmarks/pipeline/orchestrators.py:181-196`) carries:

- `git_commit` and `git_dirty` flags captured at run start,
- `seed`, fold count, `sample_size`, `tuning_sample_size`, `tuning_metric`,
- the resolved model configuration (post Hydra+Optuna),
- per-fold metric values, fold timing, and dataset metadata.

We do **not** rerun finished suites just to refresh git hashes after unrelated commits — older `git_commit` fields record the commit at which each result was produced, which is the relevant provenance information.

Final TMLR tables and plots should be generated only from result YAMLs produced under the post-`score_regime`, OOB-off BDF protocol. Older BDF result files can remain in the repository for transparency, but any table script used for the final paper must either exclude them or regenerate them. Selection-frequency statements are descriptive summaries of the pre-specified fold-0 tuning split unless an explicit repeated-tuning sensitivity study is cited. The paper build records which `.tex` tables and plot PDFs are consumed through `paper/main.fls`, and the result YAMLs record the command and resolved config where the orchestrator provides that metadata.

## 9. Hardware and Timing Notes

Reported runtimes were measured on an Apple M-series workstation (single host, multi-core). All wall-clock numbers in the paper come from `pixi run perf` and `pixi run complexity` and assume:

- `n_jobs = -1` for BDF (all physical cores).
- Default thread counts for sklearn / LightGBM / XGBoost / CatBoost / NGBoost.
- The `default` and `benchmark` pixi environments as locked in `pixi.lock`.

Relative orderings should transfer across hardware; absolute fit/predict times will not.

## 10. Known Caveats

- BART (`bartpy`) and GP baselines do not finish on every fold of every dataset within the time budget and are therefore excluded from the main regression aggregate (see Appendix~\ref{app:experimental-setup-reg-models}). The partial result files are kept for transparency but are not consumed by `pixi run reg`.
- DRF (distributional random forests) uses the R `drf` package through `rpy2`. The Python dependencies are locked, but the R package must be installed once into the system R that `rpy2` binds to. The Python wrapper runs DRF inside an isolated worker process and the synthetic runner closes that worker after each tuning trial and evaluation fold. Failed or partial DRF shards are not merged into synthetic summary plots unless they contain aggregate metrics.
- BDF currently does not handle native categorical features, native missing values, sample weights, or multiclass classification. Categorical inputs are one-hot encoded in `process-data`, which is what every model in the suite receives. This keeps the input matrix identical across methods; production CatBoost/LightGBM deployments may additionally use native categorical splits.
- A Treeffuser config is available for exploratory runs, but no locked complete Treeffuser result artifact is consumed by the TMLR paper.
