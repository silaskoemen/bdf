// scoring.rs - CORRECTED (line 5)
use ndarray::{ArrayView1, Array1, s};
use crate::distributions::{DistributionPrimitives, ScoringSpec, SufficientStats};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;  // CHANGED: rand::slice -> rand::seq

/// SINGLE scoring function that handles ALL methods (routing based on spec)
pub fn score_split(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    spec: &ScoringSpec,
) -> f64 {
    match spec.score_method.as_str() {
        "nle" => {
            dist.nle(data)
                .expect("BUG: NLE requested but not supported (should be caught in Python)")
        }

        "nll" => {
            let base_nll = dist.nll(data, spec.use_posterior_predictive);

            match spec.score_correction.as_deref() {
                None => base_nll,

                Some("aic") => {
                    base_nll + (spec.num_parameters as f64)
                }

                Some("bic") => {
                    let n = data.len() as f64;
                    base_nll + 0.5 * (spec.num_parameters as f64) * n.ln()
                }

                Some("loo_cv") => {
                    loo_cv_nll(data, dist, spec.use_posterior_predictive)
                }

                Some("kfold_cv") => {
                    kfold_cv_nll(
                        data, dist, spec.use_posterior_predictive,
                        spec.cv_folds, spec.cv_shuffle, spec.cv_seed
                    )
                }

                _ => unreachable!("Unknown correction (should be validated in Python)"),
            }
        }

        _ => unreachable!("Unknown score_method (should be validated in Python)"),
    }
}

/// FAST scoring function using sufficient statistics.
pub fn score_split_from_stats(
    stats: &SufficientStats,
    dist: &dyn DistributionPrimitives,
    spec: &ScoringSpec,
) -> Option<f64> {
    match spec.score_method.as_str() {
        "nle" => {
            dist.nle_suff_stats(stats)
        },
        "nll" => {
            if spec.score_correction.as_deref() == Some("loo_cv") ||
               spec.score_correction.as_deref() == Some("kfold_cv") {
                return None;
            }

            if let Some(nll) = dist.nll_suff_stats(stats, spec.use_posterior_predictive) {
                let correction = match spec.score_correction.as_deref() {
                    None => 0.0,
                    Some("aic") => spec.num_parameters as f64,
                    Some("bic") => 0.5 * (spec.num_parameters as f64) * stats.n.ln(),
                    _ => 0.0,
                };
                Some(nll + correction)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn loo_cv_nll(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    use_posterior_predictive: bool,
) -> f64 {
    let n = data.len();
    let mut total_nll = 0.0;

    for i in 0..n {
        let train: Array1<f64> = data.iter()
            .enumerate()
            .filter(|(idx, _)| *idx != i)
            .map(|(_, &val)| val)
            .collect();

        let test_point = data.slice(s![i..i+1]);
        total_nll += dist.nll_train_test(&train.view(), &test_point, use_posterior_predictive);
    }

    total_nll
}

fn kfold_cv_nll(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    use_posterior_predictive: bool,
    k: usize,
    shuffle: bool,
    seed: u64,
) -> f64 {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    }

    let fold_size = n / k;
    let mut total_nll = 0.0;

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { (fold + 1) * fold_size };

        let test_indices = &indices[test_start..test_end];
        let train_indices: Vec<usize> = indices.iter()
            .filter(|&&idx| idx < test_start || idx >= test_end)
            .copied()
            .collect();

        let train_data: Array1<f64> = train_indices.iter().map(|&idx| data[idx]).collect();
        let test_data: Array1<f64> = test_indices.iter().map(|&idx| data[idx]).collect();

        total_nll += dist.nll_train_test(&train_data.view(), &test_data.view(), use_posterior_predictive);
    }

    total_nll
}
