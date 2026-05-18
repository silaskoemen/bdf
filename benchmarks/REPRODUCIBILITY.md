# Reproducing the Paper

This document gives the exact pixi commands to reproduce every artifact that appears in the paper: the regression and classification benchmarks, the conditional-diagnostics analysis, the appendix studies, and the final PDF build. Reviewers are not expected to rerun the full benchmark suite; the result files that back the paper's tables and plots are tracked in the repository under `benchmarks/results/`. The commands below exist so that the pipeline can be re-executed end-to-end if desired.

A full rerun on a single workstation takes hours for the headline regression/classification suites and days for the appendix studies (convergence rates, distribution misspecification, sample-size sweeps, noise-features). All commands are deterministic: the global seed is `1234`, the fold count is `10` (tune on fold 0, evaluate on folds 1–9), and the tuning metric is CRPS for regression / log-loss for classification.

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
| `benchmark` | External baselines (NGBoost, LightGBM, XGBoost, CatBoost, QRF, ConformalRF, BART, Treeffuser) | adds `lightgbm`, `xgboost`, `ngboost`, `catboost`, `quantile-forest`, `pymc-bart`, `bartpy`, `treeffuser`, `torch` |

`bartpy` depends on a deprecated `sklearn` shim; export `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` before solving the `benchmark` environment if the install fails.

## Pipeline Overview

```
fetch-data ──► process-data ──► run benchmark suites ──► aggregate/plot ──► paper
                                  │
                                  └─► conditional-diagnostics (two-pass) ──► merged plots/tables
```

Each stage is a pixi task; nothing requires bare `python -m …` invocations.

## 1. Data

UCI/OpenML datasets are downloaded, not bundled. The two-step pipeline writes processed numeric matrices under `data/processed/` together with `*.meta.json` provenance files that record the source, processed sample/feature counts, and the target domain.

```bash
pixi run fetch-data           # downloads raw files (one-time, network required)
pixi run process-data         # produces data/processed/*.parquet and *.meta.json
```

The benchmark inventory and source per dataset is listed in `paper/appendix/experimental_setup.tex` (Table~\ref{tab:dataset-inventory}).

## 2. Main Benchmarks

### Regression suite

```bash
pixi run reg-suite                                  # BDF + sklearn baselines (default env)
pixi run -e benchmark reg-suite-models              # NGBoost / LightGBM / XGBoost / CatBoost / QRF / ConformalRF / BART
pixi run reg                                        # aggregate, run Friedman/Nemenyi, render plots and LaTeX tables
```

Results are written to `benchmarks/results/regression/res_*.yaml` (one file per model). Aggregated tables and critical-difference diagrams land in `benchmarks/plots/regression/` and `benchmarks/tables/regression/`. The `pixi run reg` step copies the figures and `.tex` snippets used in the paper into `paper/plots/` and `paper/tables/`.

### Classification suite

```bash
pixi run clas-suite                                 # BDF + sklearn baselines
pixi run -e benchmark clas-suite-models             # NGBoost / LightGBM / CalRF / KNN
pixi run clas                                       # aggregate + plot + table
```

Results: `benchmarks/results/classification/res_*.yaml`. Both suites use Hydra; passing `model=<config>` to `bench-bdf` / `bench-models` runs a single model rather than the whole sweep.

### Single-model spot run

```bash
pixi run bench-bdf model=bdf_normalmunormal                 # default env
pixi run -e benchmark bench-models model=qrf                # benchmark env
```

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

The two passes drop JSON into `benchmarks/results/conditional_diagnostics/`; the plot script consumes every `conditional_*.json` in that directory and refuses to merge if the same `(model, dataset)` pair appears twice.

## 4. Appendix Studies

| Study | Command | Output dir |
|---|---|---|
| Synthetic DGP benchmark | `pixi run synth-study` then `pixi run synth-output` | `benchmarks/results/synthetic_dgp/` |
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

Each study script also writes an auto-generated Markdown report alongside its results.

## 5. Paper Build

```bash
pixi run paper          # latexmk -pdf paper/main.tex
```

Produces `paper/main.pdf`. The build is clean: no undefined refs, citations, or overfull/underfull warnings on the locked artifact set.

## 6. Result Provenance

Every result YAML written by the orchestrator (`benchmarks/pipeline/orchestrators.py:181-196`) carries:

- `git_commit` and `git_dirty` flags captured at run start,
- `seed`, fold count, `sample_size`, `tuning_sample_size`, `tuning_metric`,
- the resolved model configuration (post Hydra+Optuna),
- per-fold metric values, fold timing, and dataset metadata.

We do **not** rerun finished suites just to refresh git hashes after unrelated commits — older `git_commit` fields record the commit at which each result was produced, which is the relevant provenance information.

## 7. Hardware and Timing Notes

Reported runtimes were measured on an Apple M-series workstation (single host, multi-core). All wall-clock numbers in the paper come from `pixi run perf` and `pixi run complexity` and assume:

- `n_jobs = -1` for BDF (all physical cores).
- Default thread counts for sklearn / LightGBM / XGBoost / CatBoost / NGBoost.
- The `default` and `benchmark` pixi environments as locked in `pixi.lock`.

Relative orderings should transfer across hardware; absolute fit/predict times will not.

## 8. Known Caveats

- BART (`bartpy`) and GP baselines do not finish on every fold of every dataset within the time budget and are therefore excluded from the main regression aggregate (see Appendix~\ref{app:experimental-setup-reg-models}). The partial result files are kept for transparency but are not consumed by `pixi run reg`.
- DRF (distributional random forests) is omitted from the comparison because its only available implementation depends on `rpy2`, which does not install in any platform of `pixi.lock`. This is documented as a limitation rather than as evidence against the method.
- BDF currently does not handle native categorical features, native missing values, sample weights, or multiclass classification. Categorical inputs must be one-hot encoded in `process-data`, which is what every model in the suite receives.
