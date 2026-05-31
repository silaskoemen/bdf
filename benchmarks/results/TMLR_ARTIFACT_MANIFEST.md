# TMLR Artifact Manifest

Generated: `2026-05-31T19:13:17.062256+00:00`
Git commit: `4fb9b8b3a64125d8e2dfc9f1f01cb8f295f76566`
Git status: `M benchmarks/REPRODUCIBILITY.md
 M benchmarks/calc_plot_regression_metrics.py
 M benchmarks/results/regression/tables/bdf_distribution_selection.tex
 M paper/appendix/additional_results.tex
 M paper/appendix/experimental_setup.tex
 M paper/main.tex
 M paper/sections/experiments.tex
 M pixi.toml
?? benchmarks/make_tmlr_artifact_manifest.py
?? benchmarks/results/TMLR_ARTIFACT_MANIFEST.md`

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
pixi run paper
pixi run tmlr-manifest
```

## Paper PDF

| Path | SHA-256 |
|---|---|
| `paper/main.pdf` | `82e28d387ab2de2ae1ad5bbf0937f028493e64fa26528ec4fbcca6fa66b8ffb7` |

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
| `benchmarks/results/regression/res_ngboost_reg.yaml` | `7429335b47a53f630822f0188f8c53cfd1beefaef57fca0e79b13cd223f5f942` |
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
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `67162291ba4236493c485714ec34b2233c0882ac11f8953834235d0fc3841a4d` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `447ecafce47d461a89c32ec49589faa734130405d2985276db53f54a50a7c079` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `e36db3f0a62aab2f0afe102a99c7bb498eff0ebc2ed615b4641eda9a7996a280` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `9d1668cf16ad0b5888ce777658cb9b36362f075254b61d03a8130fd3c94bd9c4` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `8254f813d66f4e0b283e6c4495c9e01c239ead531ed666b68d580b79dfa7e2ea` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `6aea63a66e162fc62ee67584e729ee1986e11c52b3bffacf93bf5118a297e4cc` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `8be038eecf0c7ec1c10bfec4eb455e331b236f3ac25993c39f97798ceadad2e1` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `728b8a3a2842b79daa6a0649365f9b2fb32da4e626b13755f89fe480fdd5e33b` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `50ed7fbd4efbff01fe0a7611043354b1d11b8881d007d97a509c34757b9c9a5d` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `21432a62c5d1c43f77d0050cdb3a64aef4f48044668cc5d1349b9515cee80ef1` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `2b66cee435c917eed1b8ee272a784f1c9a1ded0a18aed697555a5448c1f9863f` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `59e8c8f065f6fdade0971bb340b7956db5c7c4a9d9781c43513e34e71ee69557` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `f1235175f3bdca9dff5754f06de2e7fa080cb30b616f57036b1aff52b1fe6a8e` |

## Generated Plot PDFs Consumed By The Paper

| Path | SHA-256 |
|---|---|
| `paper/plots/classification/classification_cd_brier.pdf` | `847ffb666a27f32af9b21676df7b6b489adcc3284040baf32b088dc2c61bf6ca` |
| `paper/plots/classification/classification_cd_log_loss.pdf` | `4d812345930c5b93980e24e20bde2428e07f62a49abd3759b041f157c8759dd7` |
| `paper/plots/classification/classification_rel_to_best_log_loss.pdf` | `22c0216673bcb1063eecaad295f63bb839b089c28b70f51098de2a8dd6da3890` |
| `paper/plots/classification/classification_reliability_diagram.pdf` | `271e943e1b235ad129c92f21c85a178dd618baadd4c189a0cf4658a16bb403e0` |
| `paper/plots/classification/classification_scatter_brier.pdf` | `cddb989454a1ef9990e703f2c4b201eef27be29655a264f8831aba0425751953` |
| `paper/plots/classification/classification_scatter_ece.pdf` | `18bf57198dafb6ac2dc2751d71eb3b629808a205bc286c13cd5ff37856bc8346` |
| `paper/plots/classification/classification_scatter_log_loss.pdf` | `7341ff69a5e04a3752ca244051e9824ef3825672aa7c4cab2cfcf923e6e777c9` |
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
| `paper/plots/regression/regression_cd_crps.pdf` | `0580597f05ba6c42901c80a643074e704575108d0b0134b18dbe5953417ceb02` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `7186fe5fe42577992ee0360747809cd40077a5cf11e320b382a95fe31419e4f2` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `af299603dc0e1f5651eb199ea28c9d800df27667609a31781c4512ecc6a8550e` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `d9b066b9ea0e00c51c5b4d9dd19459a55648eb8ba68b70b4ff526b766cd241ff` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `b100f554473d9f24c164cfc2e8abf7498d13d77a297ef7855c1fc5e73ef5959f` |
| `paper/plots/regression/regression_scatter_crps.pdf` | `69f3b72a1e3fece0719251b58f35ac6c7cff6aab47fda747281eecb6e1322bea` |
| `paper/plots/regression/regression_scatter_rmse.pdf` | `0e9b5702da3995ce0cfe070731f520a379e11fa3949d14e16b4b5f00f2ab8e79` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/dgp_overview.pdf` | `a4ddf62c72c17236d328eae7c4bb8556d170a018f09b386e2089867ad7d8815c` |
| `paper/plots/synthetic/summary/main_metric_table.pdf` | `01f9cd447e6ab37b2fdaea483e6ebc0bbab56c08bff811d86dfcafca4c8e8985` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |

## Paper Build Inputs

These are repository-local files listed as `INPUT` entries in `paper/main.fls`. They include manuscript sources, generated table snippets, and plot PDFs consumed by LaTeX.

| Path | SHA-256 |
|---|---|
| `benchmarks/results/classification/tables/bdf_core_vs_full_classification.tex` | `3b07eff886fe87cd95fca778072c8131f6a0f258c1ce1713ec7deb08b34d16e1` |
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `67162291ba4236493c485714ec34b2233c0882ac11f8953834235d0fc3841a4d` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `447ecafce47d461a89c32ec49589faa734130405d2985276db53f54a50a7c079` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `e36db3f0a62aab2f0afe102a99c7bb498eff0ebc2ed615b4641eda9a7996a280` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `9d1668cf16ad0b5888ce777658cb9b36362f075254b61d03a8130fd3c94bd9c4` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `8254f813d66f4e0b283e6c4495c9e01c239ead531ed666b68d580b79dfa7e2ea` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `6aea63a66e162fc62ee67584e729ee1986e11c52b3bffacf93bf5118a297e4cc` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `8be038eecf0c7ec1c10bfec4eb455e331b236f3ac25993c39f97798ceadad2e1` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `728b8a3a2842b79daa6a0649365f9b2fb32da4e626b13755f89fe480fdd5e33b` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `50ed7fbd4efbff01fe0a7611043354b1d11b8881d007d97a509c34757b9c9a5d` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `21432a62c5d1c43f77d0050cdb3a64aef4f48044668cc5d1349b9515cee80ef1` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `2b66cee435c917eed1b8ee272a784f1c9a1ded0a18aed697555a5448c1f9863f` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `59e8c8f065f6fdade0971bb340b7956db5c7c4a9d9781c43513e34e71ee69557` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `f1235175f3bdca9dff5754f06de2e7fa080cb30b616f57036b1aff52b1fe6a8e` |
| `paper/appendix/additional_results.tex` | `9ecf88762fe067bc0060d6bc42ed5bf3c79dda792c18666279923d7e0eae3191` |
| `paper/appendix/experimental_setup.tex` | `e5f8ddb06ec96795e30eb78e5d3b45b98ad492c3d7a3eea73f385324aa8d3b88` |
| `paper/appendix/proofs.tex` | `9cb10196cf398c53a0edbd0e0f13bc6f3056e2276b8763ebee4d323609e11055` |
| `paper/fancyhdr.sty` | `2bcdf00b7ff35411e1fc3dece5e1489b467e9e455a71e437bf07c41f359ab584` |
| `paper/main.aux` | `66e4964ec2fca357efc1ae0404d4dab1296929ceb4ced21569aac6e91ce3e85b` |
| `paper/main.bbl` | `ba62f897233a0be5e38c9159181ce946e4d414e77d889ea433b60fbee7504de4` |
| `paper/main.out` | `43d97b38523316a384c05d5e993a5f64aa6ce2a75120ad1ff5481370c943d36c` |
| `paper/main.tex` | `7050b663517309204b93fef97d8eb2747f5167f94ff1a60c8540706ce6656dee` |
| `paper/plots/classification/classification_cd_brier.pdf` | `847ffb666a27f32af9b21676df7b6b489adcc3284040baf32b088dc2c61bf6ca` |
| `paper/plots/classification/classification_cd_log_loss.pdf` | `4d812345930c5b93980e24e20bde2428e07f62a49abd3759b041f157c8759dd7` |
| `paper/plots/classification/classification_rel_to_best_log_loss.pdf` | `22c0216673bcb1063eecaad295f63bb839b089c28b70f51098de2a8dd6da3890` |
| `paper/plots/classification/classification_reliability_diagram.pdf` | `271e943e1b235ad129c92f21c85a178dd618baadd4c189a0cf4658a16bb403e0` |
| `paper/plots/classification/classification_scatter_brier.pdf` | `cddb989454a1ef9990e703f2c4b201eef27be29655a264f8831aba0425751953` |
| `paper/plots/classification/classification_scatter_ece.pdf` | `18bf57198dafb6ac2dc2751d71eb3b629808a205bc286c13cd5ff37856bc8346` |
| `paper/plots/classification/classification_scatter_log_loss.pdf` | `7341ff69a5e04a3752ca244051e9824ef3825672aa7c4cab2cfcf923e6e777c9` |
| `paper/plots/complexity/memory_comparison.pdf` | `7dc670864d49c75c6865a0cbd553a29f53dc5070d471ed02bf0e01a4b4889c5d` |
| `paper/plots/complexity/overhead_ratio.pdf` | `1d013e3c869b1f5e44785ee266892ab42ed506f60260f4192b419e78cd317101` |
| `paper/plots/complexity/scaling_comparison.pdf` | `de17e836fc6c7f8e74149c0b933e6d26d41e5d24814ed6d84431a559cfa3a191` |
| `paper/plots/conditional_diagnostics/conditional_coverage_heatmap_90.pdf` | `eafdd886a6255e3664a0ad7a47db4536aa68b8b0a02bd9532258a25647ad48a4` |
| `paper/plots/conditional_diagnostics/conditional_interval_score_90.pdf` | `c2171d5c2222e5d3aa7da77a05afc3284d11079dcc76e5f5c3dd39efd15fc5df` |
| `paper/plots/conformalization/global_coverage_summary.pdf` | `bd0cc641ce968044f0cb486e30e127362bb61eefd42b5dd48feb6f437bf74ef9` |
| `paper/plots/convergence/convergence_rates_table.tex` | `c706dc27b45d23c14440a7803f4e2360d8e2f3d9bed84ba98c6ef8371c63e940` |
| `paper/plots/convergence/rate_comparison.pdf` | `d48e9b2d8bf3442d47b24d49da9722341a4d407b670a167994e8e7832aded2d2` |
| `paper/plots/ensemble_size/combined_summary.pdf` | `d8a8997cc44be9864c1bb312f52f62068fa984f1f2c019e365100943eb3c5e59` |
| `paper/plots/ensemble_size/crps_summary_table.tex` | `8eb0246b84995b13b2a4174de5608958004e9bcd955121086e13dedc6da7cfb0` |
| `paper/plots/misspecification/degradation_summary.pdf` | `50746310fc9af3291d33c039c4a13b4328f12d4aeb2e0c2dcc3337e02ea50f9b` |
| `paper/plots/noise/crps_relative_degradation.pdf` | `2bb4fd3bb3c3d800e67229a6b44aa3e832799a7eaddf21ad5e51548e4b16ea21` |
| `paper/plots/noise/crps_summary_table.tex` | `fdbc8a554134b1e1bb3f7b9c91f6f7af460ee461fc8cad26bffe2450cbacf4b0` |
| `paper/plots/noise/feature_selection_rates.pdf` | `a4ccca88a35f4e627e8cb1240865fdc998dacbdb4b236def11194485d51d8b6a` |
| `paper/plots/parameters/classification/clas_combined_sensitivity.pdf` | `b593ad41511b3a67ab8e615e3ec46b6254b59c4f6baa66621a1ef1df1bb6689e` |
| `paper/plots/parameters/classification/clas_param_recommendation_table.tex` | `00fa9eb3ffe5a82d0540c1792c52dc5976f44abfa7fedb9ae7fb81da2be0e8a7` |
| `paper/plots/parameters/regression/combined_sensitivity.pdf` | `7e5901cb902d8f23d12e39d91d3497f02f8ef8b7a46ddaad75dada44895ea55d` |
| `paper/plots/parameters/regression/param_recommendation_table.tex` | `2eee66be15040f21eba0b6acdb1e63fa61a499b91a7bd0d2f356745decc4b0a9` |
| `paper/plots/regression/regression_cd_crps.pdf` | `0580597f05ba6c42901c80a643074e704575108d0b0134b18dbe5953417ceb02` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `7186fe5fe42577992ee0360747809cd40077a5cf11e320b382a95fe31419e4f2` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `af299603dc0e1f5651eb199ea28c9d800df27667609a31781c4512ecc6a8550e` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `d9b066b9ea0e00c51c5b4d9dd19459a55648eb8ba68b70b4ff526b766cd241ff` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `b100f554473d9f24c164cfc2e8abf7498d13d77a297ef7855c1fc5e73ef5959f` |
| `paper/plots/regression/regression_scatter_crps.pdf` | `69f3b72a1e3fece0719251b58f35ac6c7cff6aab47fda747281eecb6e1322bea` |
| `paper/plots/regression/regression_scatter_rmse.pdf` | `0e9b5702da3995ce0cfe070731f520a379e11fa3949d14e16b4b5f00f2ab8e79` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/dgp_overview.pdf` | `a4ddf62c72c17236d328eae7c4bb8556d170a018f09b386e2089867ad7d8815c` |
| `paper/plots/synthetic/summary/main_metric_table.pdf` | `01f9cd447e6ab37b2fdaea483e6ebc0bbab56c08bff811d86dfcafca4c8e8985` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |
| `paper/sections/experiments.tex` | `f610bd0bd9a8f6056e6b86bab904fa87d30d281157bbaece97cc56ded68b72f0` |
| `paper/tables/complexity/complexity_table.tex` | `9c843a2c0d6cbb57d11c33d6fc3b7a7c51b7ce15520edcc6199c2ebb2bdef5dc` |
| `paper/tables/conditional_diagnostics/conditional_diagnostics_summary.tex` | `3f2e51a8b1b110757438eab449076003a70f55f792208e5768dc016b977accfa` |
| `paper/tables/misspecification/misspecification_crps.tex` | `3daa085722a8e6d7a0741fd67a864b89d454eefe4f9e0ad8324fd343ef158316` |
| `paper/tables/scoring_method/real_data/merged_real_data.tex` | `37d62ec05ecaf46e16cadee9bc8edf6c6fb34a09c97bf0696e717624af233acc` |
| `paper/tmlr.sty` | `5137789fbcfcf6188e67f3ccf28363472997a1a9498ce382e4cf4bee10d542e9` |
