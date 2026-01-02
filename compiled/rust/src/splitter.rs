use ndarray::{ArrayView1, ArrayView2, Array1, s};
use crate::distributions::{DistributionPrimitives, ScoringSpec, SufficientStats};
use crate::scoring::{score_split, score_split_from_stats};
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
    reg_gamma: f64,
    col_idcs: Option<Array1<usize>>,
    split_gain_method: &str,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>, Option<HashMap<String, f64>>, Option<HashMap<String, f64>>) {
    let n_features = x.shape()[1];
    let n_samples = y.len();

    // Check ONCE: does this distribution support sufficient statistics?
    let moment_order = distribution.required_moment_order();
    let use_suff_stats = moment_order > 0;

    // Parent score
    let (parent_stats, current_score) = if use_suff_stats {
        let mut stats = SufficientStats::default();
        for &val in y { stats.add(val, moment_order as u8); }
        let score = score_split_from_stats(&stats, distribution, scoring_spec).unwrap();
        (Some(stats), score)
    } else {
        (None, score_split(y, distribution, scoring_spec))
    };

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
    let num_quantiles = (n_samples / stride).max(1);
    let num_features_tried = feature_idcs.len();

    if use_suff_stats {
        // ====================================================================
        // FAST PATH: O(N) scan with O(1) scoring via sufficient statistics
        // ====================================================================
        let parent_stats = parent_stats.unwrap();
        let order = moment_order as u8;

        feature_idcs.par_iter().for_each(|&feature_idx| {
            let column = x.slice(s![.., feature_idx]);

            let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            sorted_indices.sort_unstable_by(|&a, &b|
                column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            );

            let mut left_stats = SufficientStats::default();
            let mut right_stats = parent_stats;

            let mut local_best_loss = 0.0;
            let mut local_best_threshold = None;
            let mut local_best_split_idx = None;

            let mut num_thresholds_tried = 0;

            for i in 0..(n_samples - 1) {
                let idx = sorted_indices[i];
                let val = y[idx];

                left_stats.add(val, order);
                right_stats.remove(val, order);

                if (i + 1) % stride == 0 {
                    let feat_val = column[idx];
                    let next_feat_val = column[sorted_indices[i + 1]];

                    if feat_val < next_feat_val {
                        if (left_stats.n as usize) < min_samples_leaf ||
                           (right_stats.n as usize) < min_samples_leaf {
                            continue;
                        }
                        if left_stats.n < min_child_weight || right_stats.n < min_child_weight {
                            continue;
                        }

                        num_thresholds_tried += 1;
                        // Direct unwrap - we know suff stats is supported
                        let left_score = score_split_from_stats(&left_stats, distribution, scoring_spec).unwrap();
                        let right_score = score_split_from_stats(&right_stats, distribution, scoring_spec).unwrap();

                        let improvement = current_score - (left_score + right_score);

                        if improvement > local_best_loss {
                            local_best_loss = improvement;
                            local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
                            local_best_split_idx = Some(i);
                        }
                    }
                }
            }
            if (reg_gamma > 0.0) && (num_thresholds_tried > 0) && (local_best_threshold.is_some()) {
                // Apply complexity penalty of gamma*(ln(k) + ln(m_j))
                local_best_loss -= reg_gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
            }

            update_best(&best_results, feature_idx, n_samples, local_best_loss,
                        local_best_threshold, local_best_split_idx, &sorted_indices);
        });
    } else {
        // ====================================================================
        // SLOW PATH: O(Q) quantile jumps with direct slice scoring
        // No stats tracking, no full O(N) scan
        // ====================================================================
        feature_idcs.par_iter().for_each(|&feature_idx| {
            let column = x.slice(s![.., feature_idx]);

            let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            sorted_indices.sort_unstable_by(|&a, &b|
                column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            );

            // Pre-sort y for direct slicing
            let sorted_y: Vec<f64> = sorted_indices.iter().map(|&i| y[i]).collect();

            let mut local_best_loss = 0.0;
            let mut local_best_threshold = None;
            let mut local_best_split_idx = None;

            let mut num_thresholds_tried = 0;

            // Jump directly to quantile positions
            for q in 1..num_quantiles {
                let split_idx = (q * stride).min(n_samples - 1) - 1;

                if split_idx == 0 || split_idx >= n_samples - 1 {
                    continue;
                }

                let left_n = split_idx + 1;
                let right_n = n_samples - left_n;

                if left_n < min_samples_leaf || right_n < min_samples_leaf {
                    continue;
                }

                let feat_val = column[sorted_indices[split_idx]];
                let next_feat_val = column[sorted_indices[split_idx + 1]];

                if feat_val >= next_feat_val {
                    continue;
                }

                num_thresholds_tried += 1;

                // Direct slice views - no allocation
                let left_y = ArrayView1::from(&sorted_y[..=split_idx]);
                let right_y = ArrayView1::from(&sorted_y[split_idx + 1..]);

                // Check min_child_weight
                let left_weight: f64 = left_y.sum();
                let right_weight: f64 = right_y.sum();
                if left_weight < min_child_weight || right_weight < min_child_weight {
                    continue;
                }

                let left_score = score_split(&left_y, distribution, scoring_spec);
                let right_score = score_split(&right_y, distribution, scoring_spec);

                let improvement = current_score - (left_score + right_score);

                if improvement > local_best_loss {
                    local_best_loss = improvement;
                    local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
                    local_best_split_idx = Some(split_idx);
                }
            }
            if (reg_gamma > 0.0) && (num_thresholds_tried > 0) && (local_best_threshold.is_some()) {
                // Apply complexity penalty of gamma*(ln(k) + ln(m_j))
                local_best_loss -= reg_gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
            }

            update_best(&best_results, feature_idx, n_samples, local_best_loss,
                        local_best_threshold, local_best_split_idx, &sorted_indices);
        });
    }

    // Extract results
    let best = best_results.lock().unwrap();

    let (left_params, right_params) = if let (Some(l_mask), Some(r_mask)) = (&best.3, &best.4) {
        if distribution.supports_rust_params() {
            let left_y: Array1<f64> = y.iter().zip(l_mask.iter())
                .filter(|&(_, &m)| m).map(|(&v, _)| v).collect();
            let right_y: Array1<f64> = y.iter().zip(r_mask.iter())
                .filter(|&(_, &m)| m).map(|(&v, _)| v).collect();
            (
                Some(distribution.calc_posterior_params(&left_y.view())),
                Some(distribution.calc_posterior_params(&right_y.view()))
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    (best.0, best.1, best.2, best.3.clone(), best.4.clone(), left_params, right_params)
}

#[inline]
fn update_best(
    best_results: &Mutex<(Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>)>,
    feature_idx: usize,
    n_samples: usize,
    local_best_loss: f64,
    local_best_threshold: Option<f64>,
    local_best_split_idx: Option<usize>,
    sorted_indices: &[usize],
) {
    if let Some(threshold) = local_best_threshold {
        let mut best = best_results.lock().unwrap();
        if local_best_loss > best.2 {
            let split_idx = local_best_split_idx.unwrap();

            let mut left_mask = Array1::from_elem(n_samples, false);
            let mut right_mask = Array1::from_elem(n_samples, false);

            for k in 0..=split_idx { left_mask[sorted_indices[k]] = true; }
            for k in (split_idx + 1)..n_samples { right_mask[sorted_indices[k]] = true; }

            *best = (
                Some(feature_idx),
                Some(threshold),
                local_best_loss,
                Some(left_mask),
                Some(right_mask)
            );
        }
    }
}
