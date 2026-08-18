# TMLR Artifact Manifest

Generated: `2026-07-12T21:56:02.666385+00:00`
Git commit: `b7ecf2dcdda5f341b124a270456ade3dad37cf1c`
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
| `paper/main.pdf` | `b9cd30eb1a00056d4b7aebca0b24d3b6e111260b788ef7f8e260e0fe6b3d4d7f` |

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
| `benchmarks/results/regression/res_bdf_normalmunormal.yaml` | `f3decb81a384fb569ab1e0bd5e1fdfe60a32961506e84279544eeeae432ddcc4` |
| `benchmarks/results/regression/res_bdf_normalmunormal_nle.yaml` | `b7f3b6e349e2b8a98a43279c089d3148dc772625216f9f69411e74ba290d35ca` |
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
| `benchmarks/results/classification/tables/rankings_brier.tex` | `9e818960fff6ee59f9250c497bb96a30002afd0233d121df798df67c374b2c35` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `5bfb840e30be1b720b141843de0bfbdce6af1e737d1d4a00ca0f3fadf5aa2532` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `07c471b6a337f958a0a774cc11fba1f0957a2493fbcb5ba37fd75ad6c7879b1d` |
| `benchmarks/results/factorial_evidence/tables/factorial_evidence.tex` | `6da9c4dbd2cab38df2f1c4756394d629905fc91b6e15d4721b876d6857fde24d` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `8254f813d66f4e0b283e6c4495c9e01c239ead531ed666b68d580b79dfa7e2ea` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `6aea63a66e162fc62ee67584e729ee1986e11c52b3bffacf93bf5118a297e4cc` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `88c44b73c24b6161d8e2fda3f6aab97ef72506da38422de92dbb0a8869bfe36c` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `eb7c649dc6a23c594ef8e43821ec53a5439c16710682a17928bc7af1c7bbefad` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `50ed7fbd4efbff01fe0a7611043354b1d11b8881d007d97a509c34757b9c9a5d` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `971c54d8b38d85ace1cd6e4619296867e8c03d9176512802bc5465acf9601d2f` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `35645311b3d9eb393ae1af5597057c68fc271d7bf252b6f5d9d7bd94b4066924` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `e1e0f8c854510254c66ac2bb208aa6a4f75203c1d20d4d3d5c1e369b3ab757c8` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `a0dd68de4a9b650072eea3902a527a93e8ffdf62b5d93d9a18ceeec4247c1ec3` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `273e0b6b95618cf862e2fe095206d84f67a0ac1cd9f663f7799f8e56247ab0de` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `c2f4f263d20955c9bad0410a02630c718a6f440687119253c747eac967e3f94b` |
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
| `paper/plots/regression/regression_cd_crps.pdf` | `8c5d3e2268efe6adbe00ee13b54aa6960e8d1105f8b39d171b3df96e233dcd77` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `aef7d032ed1fe891bd2dba7aeefee90cc89cb8abd46672744e3ab7a83bc6a15b` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `2d6b51bd23c3a7a45cbc59193e87f93a34e6bf93335bc5c3eaaa5d8b390f411a` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `3e79e60033f70464dc459ec112edc9b11222c96f4faa52ba8fd83dad53aa6c35` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `933d1ac19ba6b613d00fadd56a9cb097d85b4dec10b13ead563d8eb963b12f45` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |

## Paper Build Inputs

These are repository-local files listed as `INPUT` entries in `paper/main.fls`. They include manuscript sources, generated table snippets, and plot PDFs consumed by LaTeX.

| Path | SHA-256 |
|---|---|
| `benchmarks/results/classification/tables/bdf_core_vs_full_classification.tex` | `3b07eff886fe87cd95fca778072c8131f6a0f258c1ce1713ec7deb08b34d16e1` |
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `14830b2b032305a82d3741acc545efb8f411b2c1a27cf9d73c69ec7d1908d114` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `9e818960fff6ee59f9250c497bb96a30002afd0233d121df798df67c374b2c35` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `5bfb840e30be1b720b141843de0bfbdce6af1e737d1d4a00ca0f3fadf5aa2532` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `07c471b6a337f958a0a774cc11fba1f0957a2493fbcb5ba37fd75ad6c7879b1d` |
| `benchmarks/results/factorial_evidence/tables/factorial_evidence.tex` | `6da9c4dbd2cab38df2f1c4756394d629905fc91b6e15d4721b876d6857fde24d` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `8254f813d66f4e0b283e6c4495c9e01c239ead531ed666b68d580b79dfa7e2ea` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `6aea63a66e162fc62ee67584e729ee1986e11c52b3bffacf93bf5118a297e4cc` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `88c44b73c24b6161d8e2fda3f6aab97ef72506da38422de92dbb0a8869bfe36c` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `eb7c649dc6a23c594ef8e43821ec53a5439c16710682a17928bc7af1c7bbefad` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `50ed7fbd4efbff01fe0a7611043354b1d11b8881d007d97a509c34757b9c9a5d` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `971c54d8b38d85ace1cd6e4619296867e8c03d9176512802bc5465acf9601d2f` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `35645311b3d9eb393ae1af5597057c68fc271d7bf252b6f5d9d7bd94b4066924` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `e1e0f8c854510254c66ac2bb208aa6a4f75203c1d20d4d3d5c1e369b3ab757c8` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `a0dd68de4a9b650072eea3902a527a93e8ffdf62b5d93d9a18ceeec4247c1ec3` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `273e0b6b95618cf862e2fe095206d84f67a0ac1cd9f663f7799f8e56247ab0de` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `c2f4f263d20955c9bad0410a02630c718a6f440687119253c747eac967e3f94b` |
| `benchmarks/results/synthetic_dgp/tables/synthetic_dgp_summary.tex` | `f6e0ebf37dd1dda434a6edb394739fd0f2431b08ea1647e151890d349383931f` |
| `paper/appendix/additional_results.tex` | `03f01cb00e48858b52afe7c7182141af53c74117f617852219c6b9e80787d4c6` |
| `paper/appendix/diagnostic_robustness.tex` | `0039b4e0b09fd30e3a2adfc0dbe814c84ca66305d5d2ff0f7e6c68f1113681a6` |
| `paper/appendix/experimental_setup.tex` | `95b487c337c7f814dcc6d9528f9337f10e132aaa60108e920099b5fd76f2a9d3` |
| `paper/appendix/proofs.tex` | `a04295c13b034a75d858a31d1a31c6197834ea2db7e4468aefda54a7e6dc8a85` |
| `paper/appendix/real_data_results.tex` | `cbe10634a13d19b33170049e083c9a09575a14a70ead38a8ceb0c8995af8a592` |
| `paper/fancyhdr.sty` | `2bcdf00b7ff35411e1fc3dece5e1489b467e9e455a71e437bf07c41f359ab584` |
| `paper/main.aux` | `7a38aa1959fb6e902328f30e98e7a265c124cff2ece14fa75df7f4a91d33ca1b` |
| `paper/main.bbl` | `b2124fc052e5f035cc770d7694b264d50a07bf417cc9eaaa831df48985dd884a` |
| `paper/main.out` | `021767fea2a4055115220872105aeb87f8c8ad3df2c9ded39564133dbeb53090` |
| `paper/main.tex` | `4d6f70749bbd5ee3bb2c8c4a37fa87f27975d84aa46e712a0945cbf4b945dc32` |
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
| `paper/plots/parameters/classification/clas_param_recommendation_table.tex` | `0f704fb63de490f9b1ad8159bbf26c6bb5609ee17838f922c9c5c769206dce70` |
| `paper/plots/parameters/regression/combined_sensitivity.pdf` | `7e5901cb902d8f23d12e39d91d3497f02f8ef8b7a46ddaad75dada44895ea55d` |
| `paper/plots/parameters/regression/param_recommendation_table.tex` | `6307cd49c3fd14b1e6ac2ce23698755cb604a2f9c6fb0fbe1cc9b7ef3f5c73b4` |
| `paper/plots/regression/regression_cd_crps.pdf` | `8c5d3e2268efe6adbe00ee13b54aa6960e8d1105f8b39d171b3df96e233dcd77` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `aef7d032ed1fe891bd2dba7aeefee90cc89cb8abd46672744e3ab7a83bc6a15b` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `2d6b51bd23c3a7a45cbc59193e87f93a34e6bf93335bc5c3eaaa5d8b390f411a` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `3e79e60033f70464dc459ec112edc9b11222c96f4faa52ba8fd83dad53aa6c35` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `933d1ac19ba6b613d00fadd56a9cb097d85b4dec10b13ead563d8eb963b12f45` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |
| `paper/sections/experiments.tex` | `c5d9155ba133cbbe8d929f7f1586c823fe845ea15a69e8785c66ec9c30154bf8` |
| `paper/tables/complexity/complexity_table.tex` | `9c843a2c0d6cbb57d11c33d6fc3b7a7c51b7ce15520edcc6199c2ebb2bdef5dc` |
| `paper/tables/conditional_diagnostics/conditional_diagnostics_summary.tex` | `f2eb31ec3196d74ed42074050c326fd42170be4a064c146405bf1da0724010b5` |
| `paper/tables/misspecification/misspecification_crps.tex` | `3daa085722a8e6d7a0741fd67a864b89d454eefe4f9e0ad8324fd343ef158316` |
| `paper/tables/scoring_method/real_data/merged_real_data.tex` | `0a572ed84bda6ded7c725a01de8e0bb3b8e275e5c5fec0060d77ac0b2628cc1b` |
| `paper/tmlr.sty` | `5137789fbcfcf6188e67f3ccf28363472997a1a9498ce382e4cf4bee10d542e9` |
