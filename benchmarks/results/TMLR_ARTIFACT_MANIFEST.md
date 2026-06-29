# TMLR Artifact Manifest

Generated: `2026-06-28T21:53:54.677457+00:00`
Git commit: `4838917c59ad7c192fa36a39eada4def1eb45048`
Git status: `M benchmarks/REPRODUCIBILITY.md
 M benchmarks/calc_plot_regression_metrics.py
 M benchmarks/make_revision_bdf_tables.py
 M benchmarks/make_tmlr_artifact_manifest.py
 M benchmarks/results/TMLR_ARTIFACT_MANIFEST.md
 M benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex
 M benchmarks/results/regression/tables/bdf_distribution_selection.tex
 M benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex
 M benchmarks/results/regression/tables/bdf_student_t_stability.tex
 M benchmarks/results/regression/tables/bdf_unified_ablation.tex
 M benchmarks/results/regression/tables/crps_per_dataset.tex
 M benchmarks/results/regression/tables/crpss_per_dataset.tex
 M benchmarks/results/regression/tables/rankings_crps.tex
 M benchmarks/results/regression/tables/rankings_crpss.tex
 M benchmarks/results/regression/tables/real_benchmark_timing.tex
 M benchmarks/results/regression/tables/regression_main_results.tex
 M benchmarks/results/regression/tables/rel_to_best_crps.tex
 M benchmarks/results/regression/tables/speedup_normal.tex
 M benchmarks/results/regression/tables/speedup_selected.tex
 M benchmarks/results/regression/tables/win_tie_loss.tex
 M benchmarks/results/regression/tables/xgboostlss_distribution_selection.tex
 M benchmarks/utils/plotting.py
 M benchmarks/utils/yaml_loader.py
 M paper/appendix/experimental_setup.tex
 M paper/appendix/real_data_results.tex
 M paper/sections/experiments.tex
 M tests/test_benchmark_variant_aggregation.py
?? benchmarks/results/regression/tables/ngboost_distribution_selection.tex`

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
| `paper/main.pdf` | `cc83ba7c18bf86e5b632bf03b9af1dc1474da2c7956b89e0b99a975d1e99d9e6` |

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
| `benchmarks/results/classification/tables/classification_win_tie_loss.tex` | `67162291ba4236493c485714ec34b2233c0882ac11f8953834235d0fc3841a4d` |
| `benchmarks/results/classification/tables/rankings_brier.tex` | `447ecafce47d461a89c32ec49589faa734130405d2985276db53f54a50a7c079` |
| `benchmarks/results/classification/tables/rankings_log_loss.tex` | `e36db3f0a62aab2f0afe102a99c7bb498eff0ebc2ed615b4641eda9a7996a280` |
| `benchmarks/results/classification/tables/rel_to_best_log_loss.tex` | `9d1668cf16ad0b5888ce777658cb9b36362f075254b61d03a8130fd3c94bd9c4` |
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `235769dd3375db4ab62ab9c8be394462e56b346a34cd164ccf2db0e529c0fd24` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `fa1ec4fbd6a0ae451424a37e833ecaee867ab746b2fb42a2214e8a5357aea907` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `88c44b73c24b6161d8e2fda3f6aab97ef72506da38422de92dbb0a8869bfe36c` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `eb7c649dc6a23c594ef8e43821ec53a5439c16710682a17928bc7af1c7bbefad` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `914b37f51b212da367325c2f671b84b61ba3dc28fb948fd38d89e2d7bc174f51` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `971c54d8b38d85ace1cd6e4619296867e8c03d9176512802bc5465acf9601d2f` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `7c02b9ad31a6b63d7841255cf0a234f59cd5e6f91013fe20302a101cc4e57562` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `6f99f65da5ca44b5f5b8dff594a58775795c9c42817b5a873018ceeb3032c296` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `a0dd68de4a9b650072eea3902a527a93e8ffdf62b5d93d9a18ceeec4247c1ec3` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `baf020febee51539534fe4706dff2cd65e36b974d3c5d61e9d5ba7cd15d75703` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `ef7377ec0bd4c40e6ac30b6718d18e58d213b8f42f033d2622f3a018a3f3bd79` |
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
| `paper/plots/regression/regression_cd_crps.pdf` | `860c7312fc648cf74b61bf5877d274ae1843579aa9892a3cb12a34743cbd5428` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `4cb5a98f0174a58d7690caaf6e2a54b93a9f660b0a1ac28a5320c699cf4c6356` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `c475b19481446e8e54e508298845cf42aa0bdc886c3ac4dd943b7e196c51d047` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `b74a0adaa997d12c50a52c8ef520ad1f99319e866c078d8c6efd65e3ae515f28` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `8ba13e4966a6fd36be3904d4236ce503e8e674ca7460d38aaa5fc8a7aedaaf89` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
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
| `benchmarks/results/regression/tables/bdf_conjugate_vs_selected_gap.tex` | `235769dd3375db4ab62ab9c8be394462e56b346a34cd164ccf2db0e529c0fd24` |
| `benchmarks/results/regression/tables/bdf_distribution_selection.tex` | `fa1ec4fbd6a0ae451424a37e833ecaee867ab746b2fb42a2214e8a5357aea907` |
| `benchmarks/results/regression/tables/bdf_leaf_score_ablation.tex` | `88c44b73c24b6161d8e2fda3f6aab97ef72506da38422de92dbb0a8869bfe36c` |
| `benchmarks/results/regression/tables/bdf_normal_nle_vs_full.tex` | `eb7c649dc6a23c594ef8e43821ec53a5439c16710682a17928bc7af1c7bbefad` |
| `benchmarks/results/regression/tables/bdf_student_t_stability.tex` | `914b37f51b212da367325c2f671b84b61ba3dc28fb948fd38d89e2d7bc174f51` |
| `benchmarks/results/regression/tables/bdf_unified_ablation.tex` | `971c54d8b38d85ace1cd6e4619296867e8c03d9176512802bc5465acf9601d2f` |
| `benchmarks/results/regression/tables/crps_per_dataset.tex` | `7c02b9ad31a6b63d7841255cf0a234f59cd5e6f91013fe20302a101cc4e57562` |
| `benchmarks/results/regression/tables/rankings_crpss.tex` | `6f99f65da5ca44b5f5b8dff594a58775795c9c42817b5a873018ceeb3032c296` |
| `benchmarks/results/regression/tables/real_benchmark_timing.tex` | `a0dd68de4a9b650072eea3902a527a93e8ffdf62b5d93d9a18ceeec4247c1ec3` |
| `benchmarks/results/regression/tables/rel_to_best_crps.tex` | `baf020febee51539534fe4706dff2cd65e36b974d3c5d61e9d5ba7cd15d75703` |
| `benchmarks/results/regression/tables/win_tie_loss.tex` | `ef7377ec0bd4c40e6ac30b6718d18e58d213b8f42f033d2622f3a018a3f3bd79` |
| `benchmarks/results/synthetic_dgp/tables/synthetic_dgp_summary.tex` | `f6e0ebf37dd1dda434a6edb394739fd0f2431b08ea1647e151890d349383931f` |
| `paper/appendix/additional_results.tex` | `03f01cb00e48858b52afe7c7182141af53c74117f617852219c6b9e80787d4c6` |
| `paper/appendix/diagnostic_robustness.tex` | `10d39c4da3b88a74d4ebcfde2414deed23d1142e2303a3e7d723029a4a47c658` |
| `paper/appendix/experimental_setup.tex` | `d9c161563d45c60c1c2cf6445cf8b7ceeb8c121d46fa72e07ae51fda71c29154` |
| `paper/appendix/proofs.tex` | `fc562cd0e7e27f89f480c8a91e0fc51967439c69d4608613999ef0f037524db6` |
| `paper/appendix/real_data_results.tex` | `41953fead0f8bc0102d6d02cd691703565f25c787645b8d53c1acd546f92722a` |
| `paper/fancyhdr.sty` | `2bcdf00b7ff35411e1fc3dece5e1489b467e9e455a71e437bf07c41f359ab584` |
| `paper/main.aux` | `0b655b2d8eff8fe830a1f85ac7be792b8d8d96f08ed5d86b0ae9fb061cf39d49` |
| `paper/main.bbl` | `b2124fc052e5f035cc770d7694b264d50a07bf417cc9eaaa831df48985dd884a` |
| `paper/main.out` | `f11f5e6daf31e64643c0e5aead95dd294c1425e544547acc25dd72fbbfa24599` |
| `paper/main.tex` | `66d4d35b6533933df3d4c7d1ecc8d00b96c3ecf56f49295cc890a17dd7fec5a5` |
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
| `paper/plots/regression/regression_cd_crps.pdf` | `860c7312fc648cf74b61bf5877d274ae1843579aa9892a3cb12a34743cbd5428` |
| `paper/plots/regression/regression_coverage_vs_iscore_rank.pdf` | `4cb5a98f0174a58d7690caaf6e2a54b93a9f660b0a1ac28a5320c699cf4c6356` |
| `paper/plots/regression/regression_pit_histograms.pdf` | `c475b19481446e8e54e508298845cf42aa0bdc886c3ac4dd943b7e196c51d047` |
| `paper/plots/regression/regression_rel_to_best_crps.pdf` | `b74a0adaa997d12c50a52c8ef520ad1f99319e866c078d8c6efd65e3ae515f28` |
| `paper/plots/regression/regression_rel_to_best_rmse.pdf` | `8ba13e4966a6fd36be3904d4236ce503e8e674ca7460d38aaa5fc8a7aedaaf89` |
| `paper/plots/synthetic/summary/calibration_curves.pdf` | `41178f7aaeb577e456bd9bd542e9e2a9ad649d7784c871af2788e2ae9b5dfc14` |
| `paper/plots/synthetic/summary/pit_histograms.pdf` | `7f54d9a8672eedebef82e8ba1c4092e98c8dcbbf9e9d12823b7220ba1f421f54` |
| `paper/plots/synthetic/summary/predictions_grid.pdf` | `6eb070369ee1c224a6172283c9d654de54b105e9b2135206bd8c64b360b0ad60` |
| `paper/sections/experiments.tex` | `a1f8e976a1c43b43996eb21af7e9b9f47acb3d0b2e7326fb926fd4f01548b1ed` |
| `paper/tables/complexity/complexity_table.tex` | `9c843a2c0d6cbb57d11c33d6fc3b7a7c51b7ce15520edcc6199c2ebb2bdef5dc` |
| `paper/tables/conditional_diagnostics/conditional_diagnostics_summary.tex` | `3f2e51a8b1b110757438eab449076003a70f55f792208e5768dc016b977accfa` |
| `paper/tables/misspecification/misspecification_crps.tex` | `3daa085722a8e6d7a0741fd67a864b89d454eefe4f9e0ad8324fd343ef158316` |
| `paper/tables/scoring_method/real_data/merged_real_data.tex` | `37d62ec05ecaf46e16cadee9bc8edf6c6fb34a09c97bf0696e717624af233acc` |
| `paper/tmlr.sty` | `5137789fbcfcf6188e67f3ccf28363472997a1a9498ce382e4cf4bee10d542e9` |
