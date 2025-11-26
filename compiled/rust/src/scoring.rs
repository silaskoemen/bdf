use ndarray::{ArrayView1, Array1, s};
use crate::distributions::DistributionPrimitives;
use crate::distributions::ScoringSpec;

/// SINGLE scoring function that handles ALL methods (routing based on spec)
pub fn score_split(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    spec: &ScoringSpec,
) -> f64 {
    match spec.score_method.as_str() {
        "nle" => {
            // Use closed-form evidence (panic if not available - Python validated this!)
            -dist.log_evidence(data)
                .expect("BUG: NLE requested but not supported (should be caught in Python)")
        }
        
        "nll" => {
            let base_nll = compute_nll(data, dist, spec.use_posterior_predictive);
            
            // Apply correction (if any)
            match spec.score_correction.as_deref() {
                None => base_nll,
                
                Some("aic") => {
                    // AIC correction (Python pre-computed num_parameters!)
                    base_nll + (spec.num_parameters as f64)
                }
                
                Some("bic") => {
                    // BIC correction
                    let n = data.len() as f64;
                    base_nll + 0.5 * (spec.num_parameters as f64) * n.ln()
                }
                
                Some("loo_cv") => {
                    // LOO-CV: compute leave-one-out log-likelihood
                    let loo_ll = loo_cv_log_likelihood(data, dist, spec.use_posterior_predictive);
                    -loo_ll.mean().unwrap()
                }
                
                Some("kfold_cv") => {
                    // K-fold CV
                    let cv_ll = kfold_cv_log_likelihood(
                        data, dist, spec.use_posterior_predictive,
                        spec.cv_folds, spec.cv_shuffle, spec.cv_seed
                    );
                    -cv_ll.mean().unwrap()
                }
                
                _ => unreachable!("Unknown correction (should be validated in Python)"),
            }
        }
        
        _ => unreachable!("Unknown score_method (should be validated in Python)"),
    }
}

// Helper: compute NLL (routes to PP or plug-in)
fn compute_nll(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    use_posterior_predictive: bool,
) -> f64 {
    let params = dist.calc_posterior_params(data);
    
    let ll = if use_posterior_predictive {
        dist.posterior_predictive_log_likelihood(data, &params)
            .unwrap_or_else(|| dist.plugin_log_likelihood(data, &params))
    } else {
        dist.plugin_log_likelihood(data, &params)
    };
    
    -ll.sum()
}

// Helper: LOO-CV (generic implementation using primitives)
fn loo_cv_log_likelihood(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    use_posterior_predictive: bool,
) -> Array1<f64> {
    let n = data.len();
    let mut loo_ll = Array1::zeros(n);
    
    for i in 0..n {
        // Create train set (delete i-th point)
        let train: Array1<f64> = data.iter()
            .enumerate()
            .filter(|(idx, _)| *idx != i)
            .map(|(_, &val)| val)
            .collect();
        
        // Fit on train
        let params = dist.calc_posterior_params(&train.view());
        
        // Predict on test
        let test_point = data.slice(s![i..i+1]);
        let test_ll = if use_posterior_predictive {
            dist.posterior_predictive_log_likelihood(&test_point, &params)
                .unwrap_or_else(|| dist.plugin_log_likelihood(&test_point, &params))
        } else {
            dist.plugin_log_likelihood(&test_point, &params)
        };
        
        loo_ll[i] = test_ll[0];
    }
    
    loo_ll
}

// Helper: K-fold CV (generic implementation)
fn kfold_cv_log_likelihood(
    data: &ArrayView1<f64>,
    dist: &dyn DistributionPrimitives,
    use_posterior_predictive: bool,
    k: usize,
    shuffle: bool,
    seed: u64,
) -> Array1<f64> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    
    if shuffle {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    }
    
    let fold_size = n / k;
    let mut cv_ll = Array1::zeros(n);
    
    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { (fold + 1) * fold_size };
        
        let test_indices = &indices[test_start..test_end];
        let train_indices: Vec<usize> = indices.iter()
            .filter(|&&idx| idx < test_start || idx >= test_end)
            .copied()
            .collect();
        
        // Extract train data
        let train_data: Array1<f64> = train_indices.iter()
            .map(|&idx| data[idx])
            .collect();
        
        let params = dist.calc_posterior_params(&train_data.view());
        
        // Evaluate on test fold
        for &test_idx in test_indices {
            let test_point = data.slice(s![test_idx..test_idx+1]);
            let test_ll = if use_posterior_predictive {
                dist.posterior_predictive_log_likelihood(&test_point, &params)
                    .unwrap_or_else(|| dist.plugin_log_likelihood(&test_point, &params))
            } else {
                dist.plugin_log_likelihood(&test_point, &params)
            };
            cv_ll[test_idx] = test_ll[0];
        }
    }
    
    cv_ll
}
