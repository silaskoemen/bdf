use ndarray::{ArrayView1, ArrayView2, Array1, s};
use crate::distributions::{DistributionPrimitives, ScoringSpec, SufficientStats};
use crate::scoring::{score_split, score_split_from_stats}; // Import both
use rayon::prelude::*;
use std::sync::Mutex;
use std::collections::HashMap;

pub fn find_best_split(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution: &dyn DistributionPrimitives,
    scoring_spec: &ScoringSpec,
    eta: f64,
    col_idcs: Option<Array1<usize>>,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>, Option<HashMap<String, f64>>, Option<HashMap<String, f64>>) {
    let n_features = x.shape()[1];
    let n_samples = y.len();

    // 1. Pre-calculate Parent Stats
    let mut parent_stats = SufficientStats::default();
    for &val in y { parent_stats.add(val); }

    // Calculate parent score
    // Try fast path first, then slow path
    let current_score = score_split_from_stats(&parent_stats, distribution, scoring_spec)
        .unwrap_or_else(|| score_split(y, distribution, scoring_spec));

    // Shared best results
    let best_results = Mutex::new((
        None as Option<usize>,
        None as Option<f64>,
        0.0f64,
        None as Option<Array1<bool>>,
        None as Option<Array1<bool>>
    ));

    let feature_idcs: Vec<usize> = match col_idcs {
        Some(ref indices) => indices.to_vec(),
        None => (0..n_features).collect(),
    };

    let stride = (n_samples as f64 * eta).max(1.0) as usize;

    feature_idcs.par_iter().for_each(|&feature_idx| {
        let column = x.slice(s![.., feature_idx]);

        // 2. Sort indices by feature value (O(N log N))
        let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
        sorted_indices.sort_unstable_by(|&a, &b| column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_stats = SufficientStats::default();
        let mut right_stats = parent_stats.clone();

        let mut local_best_loss = 0.0;
        let mut local_best_threshold = None;
        let mut local_best_split_idx = None;

        // 3. Scan (O(N))
        for i in 0..(n_samples - 1) {
            let idx = sorted_indices[i];
            let val = y[idx];

            // Update stats incrementally (O(1))
            left_stats.add(val);
            right_stats.remove(val);

            // Evaluate only at quantile boundaries
            if (i + 1) % stride == 0 {
                let feat_val = column[idx];
                let next_feat_val = column[sorted_indices[i+1]];

                if feat_val < next_feat_val {
                    if (left_stats.n as usize) < min_samples_leaf || (right_stats.n as usize) < min_samples_leaf {
                        continue;
                    }
                    if left_stats.n < min_child_weight || right_stats.n < min_child_weight {
                        continue;
                    }

                    // 4. Score
                    // Try fast path (Sufficient Stats) via scoring.rs
                    let left_score_opt = score_split_from_stats(&left_stats, distribution, scoring_spec);
                    let right_score_opt = score_split_from_stats(&right_stats, distribution, scoring_spec);

                    let (left_score, right_score) = match (left_score_opt, right_score_opt) {
                        (Some(l), Some(r)) => (l, r),
                        _ => {
                            // Slow path: Reconstruct arrays
                            let left_indices_iter = sorted_indices[0..=i].iter();
                            let right_indices_iter = sorted_indices[i+1..].iter();

                            let left_y: Array1<f64> = left_indices_iter.map(|&ix| y[ix]).collect();
                            let right_y: Array1<f64> = right_indices_iter.map(|&ix| y[ix]).collect();

                            (
                                score_split(&left_y.view(), distribution, scoring_spec),
                                score_split(&right_y.view(), distribution, scoring_spec)
                            )
                        }
                    };

                    let improvement = current_score - (left_score + right_score);

                    if improvement > local_best_loss {
                        local_best_loss = improvement;
                        local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
                        local_best_split_idx = Some(i);
                    }
                }
            }
        }

        // Update global best
        if let Some(threshold) = local_best_threshold {
            let mut best = best_results.lock().unwrap();
            if local_best_loss > best.2 {
                let split_idx = local_best_split_idx.unwrap();

                let mut left_mask = Array1::from_elem(n_samples, false);
                let mut right_mask = Array1::from_elem(n_samples, false);

                for k in 0..=split_idx { left_mask[sorted_indices[k]] = true; }
                for k in (split_idx+1)..n_samples { right_mask[sorted_indices[k]] = true; }

                *best = (
                    Some(feature_idx),
                    Some(threshold),
                    local_best_loss,
                    Some(left_mask),
                    Some(right_mask)
                );
            }
        }
    });

    // Extract results
    let best = best_results.lock().unwrap();

    // Calculate final params for the best split (Cold Path - OK to use HashMap)
    let (left_params, right_params) = if let (Some(l_mask), Some(r_mask)) = (&best.3, &best.4) {
        if distribution.supports_rust_params() {
            let left_y: Array1<f64> = y.iter().zip(l_mask.iter()).filter(|&(_, &m)| m).map(|(&v, _)| v).collect();
            let right_y: Array1<f64> = y.iter().zip(r_mask.iter()).filter(|&(_, &m)| m).map(|(&v, _)| v).collect();
            (
                Some(distribution.calc_posterior_params(&left_y.view())),
                Some(distribution.calc_posterior_params(&right_y.view()))
            )
        } else {
            // KDE: skip Rust calculation, let Python do it
            (None, None)
        }
    } else {
        (None, None)
    };

    (best.0, best.1, best.2, best.3.clone(), best.4.clone(), left_params, right_params)
}
