# TMLR Artifact Manifest

Generated: `2026-09-13T15:05:37.093325+00:00`
Git commit: `edbb6698b9a2d1536243f7b8b1a2b7f709c73dcc`
Git status: `clean`

## Protocol Lock

- Real-data seed: `1234`.
- Real-data splitters: regression uses `KFold`, classification uses `StratifiedKFold`.
- Fold convention: 10 shuffled folds; fold 0 is tuning only, folds 1--9 are evaluation folds.
- Tuning metrics: CRPS for regression, log-loss for classification.
- Main synthetic DGP seed: `42`, with the same fold convention.
- Result YAMLs record the run command, git commit, dirty flag, seed, fold count, tuning metric, resolved model config, best parameters, per-fold metrics, and dataset metadata.

Regression datasets: `abalone_age`, `bike_sharing`, `boston_housing`, `california_housing`, `combined_cycle_power_plant`, `concrete_strength`, `energy_efficiency`, `forest_fires`, `kin8nm`, `naval_propulsion`, `parkinsons_updrs`, `protein_tertiary_structure`, `superconductor`, `wine_quality`, `yacht_hydrodynamics`.

Classification datasets: `ionosphere`, `breast_cancer_wisconsin`, `credit_approval`, `heart_disease`, `pima_diabetes`, `titanic`, `german_credit`, `aids_ctg_175`, `spambase`, `default_credit_card`, `bank_marketing`, `adult_income`.

## Regeneration Commands

Run these from the repository root, after installing dependencies with `pixi install`.

```bash
pixi run fetch-data
pixi run process-data
pixi run reg-suite
pixi run -e benchmark reg-suite-models
pixi run clas-suite
pixi run -e benchmark clas-suite-models
pixi run python -m benchmarks.make_revision_bdf_tables
pixi run reg
pixi run clas
pixi run synth-study
pixi run -e benchmark synth-study-models --models QRF --result-suffix forests
pixi run -e benchmark synth-study-models --models DRF --result-suffix drf
pixi run synth-output
pixi run python -m benchmarks.make_synthetic_summary_table
pixi run paper
pixi run tmlr-manifest
```

## Paper PDF

| Path | SHA-256 |
|---|---|
| `paper/main.pdf` | `dc531ad6869391b6eb620c2ec10a237f3208d6ffb8fb575f703af4fb563a0b17` |

## Locked Real-Data Result YAMLs

### Regression

| Path | SHA-256 |
|---|---|
| `benchmarks/results/regression/res_bayesridge_reg.yaml` | `4e1689f8fc97bf322e4730642e652154856130cf5058cd7caab69fe9b885aeb0` |
| `benchmarks/results/regression/res_bdf_freqstudentt.yaml` | `6410c2058de0c1f4ceb2cbb904dc75c65cc77c7395682a3560e227b42058d03a` |
| `benchmarks/results/regression/res_bdf_freqstudentt_nll.yaml` | `0de1d65453a5626df4307b6a5c36dd4dca512ed333ef460018bc6f038b2369ea` |
| `benchmarks/results/regression/res_bdf_freqstudentt_nll_bic.yaml` | `19de2f18d1226fa3706a9ad127622739cefb5c19ccaed642584d0d9b0018f61d` |
| `benchmarks/results/regression/res_bdf_gammamvlambdapoisson.yaml` | `93690c8c44e6edc9112af66369de122be2290957c86ceeec00848f1f1d018002` |
| `benchmarks/results/regression/res_bdf_kde.yaml` | `abc1ca246e7d8fd67f6666784196fc6e05eb5e0b99558fc325e8b2580d504cd8` |
| `benchmarks/results/regression/res_bdf_normalmunormal.yaml` | `3164c5da5bcea55c79dc7f9917f5a78b7e32a44830a09a764739273192a2c78b` |
| `benchmarks/results/regression/res_bdf_normalmunormal_nle.yaml` | `ca3df358a63779ea0b6fa9f0577c2b2c7f16e409154d237063399d3cf79d8f8b` |
| `benchmarks/results/regression/res_bdf_normalmunormal_nll.yaml` | `271fba35a11a3e016dabbcc65a95356f17d64755e09ff7eb2d5f56c87115e08f` |
| `benchmarks/results/regression/res_bdf_normalmunormal_nll_bic.yaml` | `8ab4a901fa4cf5510219505c4f1bd47aaada951f59316ff15aeb82123c2f27fa` |
| `benchmarks/results/regression/res_catbunc_reg.yaml` | `4458d06a1af62f2d2605d551cedbcf5086b70aab99981b3922d4f993cab933ef` |
| `benchmarks/results/regression/res_climatological.yaml` | `6152a7f82916fb3e67c4d0475479ef6ab7934bdfec38c4ff30588098dbb21fd0` |
| `benchmarks/results/regression/res_conflgbm.yaml` | `f0c1ccd643a0534fc8f6a6827bbd0e2f43011b28dde08795734205c8bfab1804` |
| `benchmarks/results/regression/res_confrf.yaml` | `ab0ceeab73b9d400ab004a6ca105a97a912ccf0baad7003d13ba6629afc78f98` |
| `benchmarks/results/regression/res_drf.yaml` | `4861b21b134e8f20b98ffe475a4410ad64521a7b7c264db398efb29a0a734322` |
| `benchmarks/results/regression/res_gaussian_de.yaml` | `27a7a828840f20dda907ef7ce02f24343c15f033b9c21aef52d94f2ad471fe28` |
| `benchmarks/results/regression/res_knnkde.yaml` | `738d806633921364e40bee3a1c78ef13079745fb36af1d705400bfc00ba5818a` |
| `benchmarks/results/regression/res_lgbm_reg.yaml` | `a94b45100055d81fcd4b243ffbd5a365f728d73490c64a6b1dfbd4b590760306` |
| `benchmarks/results/regression/res_ngboost_exponential.yaml` | `c808bbb2da99c548a09ed62896bbc3b54427832db891a1cd44664cc9b1037563` |
| `benchmarks/results/regression/res_ngboost_laplace.yaml` | `fee851476cce3f196080f4a62904e9d6fddfaf7e4b224d57617a594021623078` |
| `benchmarks/results/regression/res_ngboost_lognormal.yaml` | `6221f205ffddb792a7c88e7bef1391ce0c2bd6623499943826d80b631bc79258` |
| `benchmarks/results/regression/res_ngboost_normal.yaml` | `158fa3e0277e6cb702ecd0afacbb3060add9ef98e60abb7b14a204d88e58e079` |
| `benchmarks/results/regression/res_ngboost_poisson.yaml` | `abafc1e536a2f71f031de875dbc226b23e709ea8915db5b0df95a8ffc7fe6188` |
| `benchmarks/results/regression/res_qrf.yaml` | `3ce22bfc99fdb7d1952a510b0559579dc59a27ac3d8294e6c88f60e96f4e2a1a` |
| `benchmarks/results/regression/res_xgboostlss_gaussian.yaml` | `a3fdad20b67944c2a7c5f8e714d879b13df6e3429c5c8ce57f6e2cd788636cbe` |
| `benchmarks/results/regression/res_xgboostlss_gaussian_mixture.yaml` | `12c4991d55dd484178f2b993598c6a5c91e868860f671816ec96984286fca01c` |
| `benchmarks/results/regression/res_xgboostlss_laplace.yaml` | `6566e0053a751c6f45aa6edb43051be292193beaec70d8b30831fb7718632316` |
| `benchmarks/results/regression/res_xgboostlss_studentt.yaml` | `01732157800e846440c03f6cff8140045b34b45a53574ecc021e38d958f1727b` |

### Classification

| Path | SHA-256 |
|---|---|
| `benchmarks/results/classification/res_bdf_betamvbernoulli.yaml` | `386a61a9e59a9ff56a2b980a31e4e7eac1bf3ec412dc81b66736b7d0103bac08` |
| `benchmarks/results/classification/res_bdf_betamvbernoulli_nle.yaml` | `e17652b5ce3e54e3247f22a436c7917d95e5872fd638d645ca67a575af36aecf` |
| `benchmarks/results/classification/res_callgbm_clas.yaml` | `aba117ef08c2f1a4ac4e61283a5b46311ee02af716a8eb36fc8bfd9903de7d00` |
| `benchmarks/results/classification/res_calrf_clas.yaml` | `130edc2734e5394c3f19718ab7d683bad842bead4c7381a75cb9d6e392d1653c` |
| `benchmarks/results/classification/res_knn_clas.yaml` | `cefcdd1af73c6ee30c22f2064bc2a5d4876835f3433d16fc32b25c8042015349` |
| `benchmarks/results/classification/res_lgbm_clas.yaml` | `f0da4df5cca3da213618bda043961814ec5aeb75d29cdd9bcf7d0f4ae801aa94` |
| `benchmarks/results/classification/res_ngboost_clas.yaml` | `e0490d4435b6a4aab16dfb4a11e945de9b29d14939473c7bc73a2b6d7ac29cde` |
| `benchmarks/results/classification/res_rf_clas.yaml` | `d99cc15018a28b3ba889ee4bc5706a951281f99e56a7dd879a783ca6630f8a18` |

## Generated Tables Consumed By The Paper

| Path | SHA-256 |
|---|---|
| `benchmarks/results/classification/tables/bdf_core_vs_full_classification.tex` | `3b07eff886fe87cd95fca778072c8131f6a0f258c1ce1713ec7deb08b34d16e1` |
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `14830b2b032305a82d3741acc545efb8f411b2c1a27cf9d73c69ec7d1908d114` |
| `benchmarks/results/classification/tables/log_loss_per_dataset.tex` | `53ec109b10137712a21c5036dbc02d97e5a17cb448b24de5bc3e964d19ac26e8` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `9e818960fff6ee59f9250c497bb96a30002afd0233d121df798df67c374b2c35` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `5bfb840e30be1b720b141843de0bfbdce6af1e737d1d4a00ca0f3fadf5aa2532` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `07c471b6a337f958a0a774cc11fba1f0957a2493fbcb5ba37fd75ad6c7879b1d` |
| `benchmarks/results/factorial_evidence/tables/factorial_evidence.tex` | `d76b3968e7fd3bc10c8b00fb1abdf92e6c1917c6ecbf0a73514e6f0374f7bb72` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `22c16a6d3d0fee5891ea3a1eea58668ad8e5660eeb715466ad51a85695ba4348` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `7362c77c1b4cf98bcb78c0400d39cf11debf41b1c877cf5b922ed406de5643b0` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `2d737f765660880b1fa095c5ceea2dc737f6dc422bb1e791e53f85ee223c07cc` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `726b806c2d72b7a6e80cd7d2ac76abd9e3ead412e95218af3732d0f999624c44` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `bb9d36a6f041b710d26b4a69029dd1931bb417907da968011d16e4e6a211e2ef` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `a8f5ed178406718d0c7b6499f8870cb5877b24d053b707e9f2ebbc08ecca609d` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `085dbb323707bef55c21c28dba9b499d272357887118c03124e4edc15baad2c1` |
| `benchmarks/results/regression/tables/ngboost_distribution_selection.tex` | `7311e88fcf11d6af24365913324dd2f526f5920225d22cb6dc16c0ee9cb8d719` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `0c97f99b7fa1a2693ad26dae9480d9f60a0229c8b699e7b4fe6c11217cee2cf8` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `83b76b4ec49d904447851b93f705ee0590b1221aa4283abe9390747abf0caf19` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `1132ed8ae6e410cb3b88b7af0f2c5d577f30adcc1b0a5bd436d542191d987fdf` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `43bde24cad612f714de017f5cdf52e9e55dac22e39d2ae9a4f06198068bb318f` |
| `benchmarks/results/synthetic_dgp/tables/synthetic_dgp_summary.tex` | `f6e0ebf37dd1dda434a6edb394739fd0f2431b08ea1647e151890d349383931f` |

## Generated Plot PDFs Consumed By The Paper

| Path | SHA-256 |
|---|---|
| `paper/plots/classification/classification_cd_brier.pdf` | `847ffb666a27f32af9b21676df7b6b489adcc3284040baf32b088dc2c61bf6ca` |
| `paper/plots/classification/classification_cd_log_loss.pdf` | `4d812345930c5b93980e24e20bde2428e07f62a49abd3759b041f157c8759dd7` |
| `paper/plots/classification/classification_rel_to_best_log_loss.pdf` | `22c0216673bcb1063eecaad295f63bb839b089c28b70f51098de2a8dd6da3890` |
| `paper/plots/classification/classification_reliability_diagram.pdf` | `271e943e1b235ad129c92f21c85a178dd618baadd4c189a0cf4658a16bb403e0` |
| `paper/plots/complexity/memory_comparison.pdf` | `7dc670864d49c75c6865a0cbd553a29f53dc5070d471ed02bf0e01a4b4889c5d` |
| `paper/plots/complexity/overhead_ratio.pdf` | `1d013e3c869b1f5e44785ee266892ab42ed506f60260f4192b419e78cd317101` |
| `paper/plots/complexity/scaling_comparison.pdf` | `de17e836fc6c7f8e74149c0b933e6d26d41e5d24814ed6d84431a559cfa3a191` |
| `paper/plots/conditional_diagnostics/conditional_coverage_heatmap_90.pdf` | `eafdd886a6255e3664a0ad7a47db4536aa68b8b0a02bd9532258a25647ad48a4` |
| `paper/plots/conditional_diagnostics/conditional_interval_score_90.pdf` | `c2171d5c2222e5d3aa7da77a05afc3284d11079dcc76e5f5c3dd39efd15fc5df` |
| `paper/plots/conformalization/global_coverage_summary.pdf` | `bd0cc641ce968044f0cb486e30e127362bb61eefd42b5dd48feb6f437bf74ef9` |
| `paper/plots/convergence/rate_comparison.pdf` | `d48e9b2d8bf3442d47b24d49da9722341a4d407b670a167994e8e7832aded2d2` |
| `paper/plots/ensemble_size/combined_summary.pdf` | `d8a8997cc44be9864c1bb312f52f62068fa984f1f2c019e365100943eb3c5e59` |
| `paper/plots/misspecification/degradation_summary.pdf` | `50746310fc9af3291d33c039c4a13b4328f12d4aeb2e0c2dcc3337e02ea50f9b` |
| `paper/plots/noise/crps_relative_degradation.pdf` | `2bb4fd3bb3c3d800e67229a6b44aa3e832799a7eaddf21ad5e51548e4b16ea21` |
| `paper/plots/noise/feature_selection_rates.pdf` | `a4ccca88a35f4e627e8cb1240865fdc998dacbdb4b236def11194485d51d8b6a` |
| `paper/plots/parameters/classification/clas_combined_sensitivity.pdf` | `b593ad41511b3a67ab8e615e3ec46b6254b59c4f6baa66621a1ef1df1bb6689e` |
| `paper/plots/parameters/regression/combined_sensitivity.pdf` | `7e5901cb902d8f23d12e39d91d3497f02f8ef8b7a46ddaad75dada44895ea55d` |
| `paper/plots/regression/regression_cd_crps.pdf` | `0ecdceea20f5e5ccd8379e632a31c7a83538ecf563b973be778883f173c5d31a` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `1afdb4aabd40073a586c3295322dace29082ecdd042ba494ba4f103fa51795e0` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `151e791bee7e31962d5361147bf0e1b5138554f1df8d4be4fd029d720b8f7347` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `dbb4cc15b012b12fcfef305b9ab9b5612f323b44d930020e4d49738eaa674bf5` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `7d9838985d104a70e94c8da2e69002801a1e348ebcd4cabefb389983791577bc` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |

## Paper Build Inputs

These are repository-local files listed as `INPUT` entries in `paper/main.fls`. They include manuscript sources, generated table snippets, and plot PDFs consumed by LaTeX.

| Path | SHA-256 |
|---|---|
| `benchmarks/results/classification/tables/bdf_core_vs_full_classification.tex` | `3b07eff886fe87cd95fca778072c8131f6a0f258c1ce1713ec7deb08b34d16e1` |
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `14830b2b032305a82d3741acc545efb8f411b2c1a27cf9d73c69ec7d1908d114` |
| `benchmarks/results/classification/tables/log_loss_per_dataset.tex` | `53ec109b10137712a21c5036dbc02d97e5a17cb448b24de5bc3e964d19ac26e8` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `9e818960fff6ee59f9250c497bb96a30002afd0233d121df798df67c374b2c35` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `5bfb840e30be1b720b141843de0bfbdce6af1e737d1d4a00ca0f3fadf5aa2532` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `07c471b6a337f958a0a774cc11fba1f0957a2493fbcb5ba37fd75ad6c7879b1d` |
| `benchmarks/results/factorial_evidence/tables/factorial_evidence.tex` | `d76b3968e7fd3bc10c8b00fb1abdf92e6c1917c6ecbf0a73514e6f0374f7bb72` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `22c16a6d3d0fee5891ea3a1eea58668ad8e5660eeb715466ad51a85695ba4348` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `7362c77c1b4cf98bcb78c0400d39cf11debf41b1c877cf5b922ed406de5643b0` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `2d737f765660880b1fa095c5ceea2dc737f6dc422bb1e791e53f85ee223c07cc` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `726b806c2d72b7a6e80cd7d2ac76abd9e3ead412e95218af3732d0f999624c44` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `bb9d36a6f041b710d26b4a69029dd1931bb417907da968011d16e4e6a211e2ef` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `a8f5ed178406718d0c7b6499f8870cb5877b24d053b707e9f2ebbc08ecca609d` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `085dbb323707bef55c21c28dba9b499d272357887118c03124e4edc15baad2c1` |
| `benchmarks/results/regression/tables/ngboost_distribution_selection.tex` | `7311e88fcf11d6af24365913324dd2f526f5920225d22cb6dc16c0ee9cb8d719` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `0c97f99b7fa1a2693ad26dae9480d9f60a0229c8b699e7b4fe6c11217cee2cf8` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `83b76b4ec49d904447851b93f705ee0590b1221aa4283abe9390747abf0caf19` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `1132ed8ae6e410cb3b88b7af0f2c5d577f30adcc1b0a5bd436d542191d987fdf` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `43bde24cad612f714de017f5cdf52e9e55dac22e39d2ae9a4f06198068bb318f` |
| `benchmarks/results/synthetic_dgp/tables/synthetic_dgp_summary.tex` | `f6e0ebf37dd1dda434a6edb394739fd0f2431b08ea1647e151890d349383931f` |
| `paper/appendix/additional_results.tex` | `03f01cb00e48858b52afe7c7182141af53c74117f617852219c6b9e80787d4c6` |
| `paper/appendix/diagnostic_robustness.tex` | `1d04261bbc683bece292af9f21fde2b71f22656a11123420961353f644c97400` |
| `paper/appendix/experimental_setup.tex` | `4986465b699e4c412ff3338d499e0b64e851470897f28f5ff38e2cd5e8de67bb` |
| `paper/appendix/proofs.tex` | `ca627357f0c7e3f9d053ce9c88bb109082c2c44820c9b654031097255b0614e6` |
| `paper/appendix/real_data_results.tex` | `f858a28395320b3713a6ea37c16259e971a1c796b87a24bd57c1f5ea312e6539` |
| `paper/fancyhdr.sty` | `2bcdf00b7ff35411e1fc3dece5e1489b467e9e455a71e437bf07c41f359ab584` |
| `paper/main.aux` | `e1154d6b193913e8bc1a1652d91a5f2835d5e7b37ccaf265406c769c9de10d34` |
| `paper/main.bbl` | `d019ef823b9b1673da5dbc50f03b6db01e89648d9de6f5761c565fa6e2dc0b80` |
| `paper/main.out` | `1809b8f3658ec4dbc7dbff3754ec96be286d4f8c74a50bf8e4d6b8d0f764404e` |
| `paper/main.tex` | `5be713565bdd4db67cafd95bca3ce3e9dd5f8ce67d731d3d7e8392d793855144` |
| `paper/plots/classification/classification_cd_brier.pdf` | `847ffb666a27f32af9b21676df7b6b489adcc3284040baf32b088dc2c61bf6ca` |
| `paper/plots/classification/classification_cd_log_loss.pdf` | `4d812345930c5b93980e24e20bde2428e07f62a49abd3759b041f157c8759dd7` |
| `paper/plots/classification/classification_rel_to_best_log_loss.pdf` | `22c0216673bcb1063eecaad295f63bb839b089c28b70f51098de2a8dd6da3890` |
| `paper/plots/classification/classification_reliability_diagram.pdf` | `271e943e1b235ad129c92f21c85a178dd618baadd4c189a0cf4658a16bb403e0` |
| `paper/plots/complexity/memory_comparison.pdf` | `7dc670864d49c75c6865a0cbd553a29f53dc5070d471ed02bf0e01a4b4889c5d` |
| `paper/plots/complexity/overhead_ratio.pdf` | `1d013e3c869b1f5e44785ee266892ab42ed506f60260f4192b419e78cd317101` |
| `paper/plots/complexity/scaling_comparison.pdf` | `de17e836fc6c7f8e74149c0b933e6d26d41e5d24814ed6d84431a559cfa3a191` |
| `paper/plots/conditional_diagnostics/conditional_coverage_heatmap_90.pdf` | `eafdd886a6255e3664a0ad7a47db4536aa68b8b0a02bd9532258a25647ad48a4` |
| `paper/plots/conditional_diagnostics/conditional_interval_score_90.pdf` | `c2171d5c2222e5d3aa7da77a05afc3284d11079dcc76e5f5c3dd39efd15fc5df` |
| `paper/plots/conformalization/global_coverage_summary.pdf` | `bd0cc641ce968044f0cb486e30e127362bb61eefd42b5dd48feb6f437bf74ef9` |
| `paper/plots/convergence/convergence_rates_table.tex` | `686df125c41241b8098c62b72bf588cc6be36c2d12c0d81fe93a5bfbeb29725c` |
| `paper/plots/convergence/rate_comparison.pdf` | `d48e9b2d8bf3442d47b24d49da9722341a4d407b670a167994e8e7832aded2d2` |
| `paper/plots/ensemble_size/combined_summary.pdf` | `d8a8997cc44be9864c1bb312f52f62068fa984f1f2c019e365100943eb3c5e59` |
| `paper/plots/ensemble_size/crps_summary_table.tex` | `8eb0246b84995b13b2a4174de5608958004e9bcd955121086e13dedc6da7cfb0` |
| `paper/plots/misspecification/degradation_summary.pdf` | `50746310fc9af3291d33c039c4a13b4328f12d4aeb2e0c2dcc3337e02ea50f9b` |
| `paper/plots/noise/crps_relative_degradation.pdf` | `2bb4fd3bb3c3d800e67229a6b44aa3e832799a7eaddf21ad5e51548e4b16ea21` |
| `paper/plots/noise/crps_summary_table.tex` | `fdbc8a554134b1e1bb3f7b9c91f6f7af460ee461fc8cad26bffe2450cbacf4b0` |
| `paper/plots/noise/feature_selection_rates.pdf` | `a4ccca88a35f4e627e8cb1240865fdc998dacbdb4b236def11194485d51d8b6a` |
| `paper/plots/parameters/classification/clas_combined_sensitivity.pdf` | `b593ad41511b3a67ab8e615e3ec46b6254b59c4f6baa66621a1ef1df1bb6689e` |
| `paper/plots/parameters/classification/clas_param_recommendation_table.tex` | `ce96823629cc179d61ff1f87b7987fd3f80d9d833f8521090b1efa3e2ed8726c` |
| `paper/plots/parameters/regression/combined_sensitivity.pdf` | `7e5901cb902d8f23d12e39d91d3497f02f8ef8b7a46ddaad75dada44895ea55d` |
| `paper/plots/parameters/regression/param_recommendation_table.tex` | `ac4c1ec4daab1fdd67f92a171a110ee3f5f733cafe189069be44f765a560ba0a` |
| `paper/plots/regression/regression_cd_crps.pdf` | `0ecdceea20f5e5ccd8379e632a31c7a83538ecf563b973be778883f173c5d31a` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `1afdb4aabd40073a586c3295322dace29082ecdd042ba494ba4f103fa51795e0` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `151e791bee7e31962d5361147bf0e1b5138554f1df8d4be4fd029d720b8f7347` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `dbb4cc15b012b12fcfef305b9ab9b5612f323b44d930020e4d49738eaa674bf5` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `7d9838985d104a70e94c8da2e69002801a1e348ebcd4cabefb389983791577bc` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |
| `paper/sections/experiments.tex` | `b56167743f95db46145b6fbf8889f8e59cd7a75ba00b0f70732e94d322fbbabc` |
| `paper/tables/complexity/complexity_table.tex` | `9c843a2c0d6cbb57d11c33d6fc3b7a7c51b7ce15520edcc6199c2ebb2bdef5dc` |
| `paper/tables/conditional_diagnostics/conditional_diagnostics_summary.tex` | `f2eb31ec3196d74ed42074050c326fd42170be4a064c146405bf1da0724010b5` |
| `paper/tables/misspecification/misspecification_crps.tex` | `3daa085722a8e6d7a0741fd67a864b89d454eefe4f9e0ad8324fd343ef158316` |
| `paper/tables/scoring_method/real_data/merged_real_data.tex` | `0a572ed84bda6ded7c725a01de8e0bb3b8e275e5c5fec0060d77ac0b2628cc1b` |
| `paper/tmlr.sty` | `5137789fbcfcf6188e67f3ccf28363472997a1a9498ce382e4cf4bee10d542e9` |
