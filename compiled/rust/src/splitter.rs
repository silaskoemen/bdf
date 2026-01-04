use ndarray::{ArrayView1, ArrayView2, Array1, s};
use crate::distributions::{DistributionPrimitives, ScoringSpec, SufficientStats};
use crate::scoring::{score_split, score_split_from_stats};
use std::cmp::Ordering;
use crate::distributions::kde::{KernelType, BandwidthRule, Kde as KdeDist};
use rayon::prelude::*;
use std::sync::Mutex;
use std::collections::HashMap;
use realfft::RealFftPlanner;
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum KdeBackend {
    Pairwise,
    // Fft, // deferred
    Fft,
    Switch,
}

#[derive(Clone, Copy, Debug)]
pub enum BandwidthPolicy {
    Parent,
    PerSplit,
}

#[derive(Clone, Debug)]
pub struct KdeSplitConfig {
    pub kernel: KernelType,
    pub bandwidth_rule: BandwidthRule,
    pub min_bandwidth: f64,
    pub backend: KdeBackend,
    pub bandwidth_policy: BandwidthPolicy,
    pub kde_backend_switch_size: usize,
    // If both are present, treat as BayesianKDE
    pub prior_h: Option<f64>,
    pub m_h: Option<f64>,
    pub use_compact_support: bool,
    pub fft_grid_min: Option<f64>,
    pub fft_grid_max: Option<f64>,
    pub fft_grid_points: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct BestKdeSplit {
    feature_idx: usize,
    threshold: f64,
    gain: f64,
    thresholds_tried: usize,
}

#[inline]
fn sort_indices_by_feature(column: &ArrayView1<f64>) -> Vec<usize> {
    let n = column.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();

    // Deterministic + robust: place NaNs at the end
    sorted_indices.sort_unstable_by(|&a, &b| {
        let va = column[a];
        let vb = column[b];
        match (va.is_nan(), vb.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => va.partial_cmp(&vb).unwrap_or(Ordering::Equal),
        }
    });

    sorted_indices
}

#[inline]
fn split_idx_from_threshold(
    column: &ArrayView1<f64>,
    sorted_indices: &[usize],
    threshold: f64,
) -> Option<usize> {
    if sorted_indices.len() < 2 {
        return None;
    }

    // Find the last index on the left with value <= threshold.
    // Since NaNs are sorted to the end, we treat them as "right".
    let mut split_idx: Option<usize> = None;
    for (pos, &idx) in sorted_indices.iter().enumerate() {
        let v = column[idx];
        if v.is_nan() || v > threshold {
            break;
        }
        split_idx = Some(pos);
    }

    match split_idx {
        Some(i) if i + 1 < sorted_indices.len() => Some(i),
        _ => None,
    }
}

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

    let _ = split_gain_method; // (evidence/map handling can be added later)

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

            // let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            // sorted_indices.sort_unstable_by(|&a, &b|
            //     column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            // );
            let sorted_indices = sort_indices_by_feature(&column);

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
                        // if left_stats.n < min_child_weight || right_stats.n < min_child_weight {
                        //     continue;
                        // }

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

            // let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            // sorted_indices.sort_unstable_by(|&a, &b|
            //     column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            // );
            let sorted_indices = sort_indices_by_feature(&column);

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
                // let left_weight: f64 = left_y.sum();
                // let right_weight: f64 = right_y.sum();
                // if left_weight < min_child_weight || right_weight < min_child_weight {
                //     continue;
                // }

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


// NOTE: Could add graph sparsification (each neighbor just low/hi indices) and incremental log sums
// mainly makes sense for truly sparse graphs, epanechnikov, small eta
pub fn find_best_split_kde(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    config: &KdeSplitConfig,
    eta: f64,
    reg_gamma: f64,
    col_idcs: Option<Array1<usize>>,
    split_gain_method: &str,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>, Option<HashMap<String, f64>>, Option<HashMap<String, f64>>) {
    let n_features = x.shape()[1];
    let n_samples = y.len();
    let _ = split_gain_method;

    if n_samples < 2 {
        return (None, None, 0.0, None, None, None, None);
    }

    let base_kde = KdeDist {
        kernel: config.kernel,
        bandwidth_rule: config.bandwidth_rule,
        min_bandwidth: config.min_bandwidth,
    };

    let full_indices: Vec<usize> = (0..n_samples).collect();
    let parent_h = compute_bandwidth_for_indices(&base_kde, config, y, &full_indices);

    // FFT backend: only implemented for Gaussian + Parent bandwidth (fast + consistent).
    let n2: u128 = (n_samples as u128) * (n_samples as u128);
    let want_fft = match config.backend {
        KdeBackend::Fft => true,
        KdeBackend::Switch => n2 > (config.kde_backend_switch_size as u128),
        KdeBackend::Pairwise => false,
    };

    if want_fft {
        if !matches!(config.kernel, KernelType::Gaussian) || !matches!(config.bandwidth_policy, BandwidthPolicy::Parent) {
            // Fall back to existing implementations
        } else {
            let (f, t, g, lm, rm) = find_best_split_kde_fft(
                x, y, min_samples_leaf, min_child_weight, config, parent_h, eta, reg_gamma, col_idcs
            );
            return (f, t, g, lm, rm, None, None);
        }
    }

    let feature_idcs: Vec<usize> = match col_idcs {
        Some(ref indices) => indices.to_vec(),
        None => (0..n_features).collect(),
    };

    let stride = (n_samples as f64 * eta).max(1.0) as usize;
    let num_features_tried = feature_idcs.len();

    // Try the fast path: Parent bandwidth + dense kernel matrix
    let mut kernel_mat: Option<Vec<f64>> = None;
    let mut full_row_sums: Option<Vec<f64>> = None;

    if matches!(config.bandwidth_policy, BandwidthPolicy::Parent) {
        if let Some((k, rs)) = try_precompute_kernel_matrix(y, parent_h, config.kernel, config.use_compact_support) {
            kernel_mat = Some(k);
            full_row_sums = Some(rs);
        }
    }

    // Current node score
    let current_score = if let Some(rs) = full_row_sums.as_ref() {
        kde_full_nll_from_row_sums(rs)
    } else {
        // Fallback: old pairwise-distance implementation
        let d2 = precompute_pairwise_d2(y);
        kde_subset_nll(&full_indices, &d2, n_samples, parent_h, config.kernel, config.use_compact_support)
    };

    let best_results: Mutex<Option<BestKdeSplit>> = Mutex::new(None);
    // Track best as a single, self-contained struct.

    if let (Some(k_mat), Some(rs_total)) = (kernel_mat.as_ref(), full_row_sums.as_ref()) {
        // ================================================================
        // FAST KDE PATH (Parent bandwidth):
        // - kernel matrix built once per node
        // - per feature scan is ~O(n^2) adds, no exp/log inside candidate loop
        // ================================================================
        feature_idcs.par_iter().for_each(|&feature_idx| {
            let column = x.slice(s![.., feature_idx]);

            // let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            // sorted_indices.sort_unstable_by(|&a, &b| {
            //     column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            // });

            let sorted_indices = sort_indices_by_feature(&column);

            let mut sum_to_left = vec![0.0f64; n_samples];

            let mut local_best_gain = 0.0;
            let mut local_best_threshold = None;
            let mut num_thresholds_tried = 0usize;

            for i in 0..(n_samples - 1) {
                let idx = sorted_indices[i];
                let row = &k_mat[idx * n_samples..(idx + 1) * n_samples];

                // Single accumulator:
                // sum_to_left[p] = sum_{q in Left} K[p,q]
                for j in 0..n_samples {
                    sum_to_left[j] += row[j];
                }

                if (i + 1) % stride != 0 {
                    continue;
                }

                let feat_val = column[idx];
                let next_feat_val = column[sorted_indices[i + 1]];
                if feat_val >= next_feat_val {
                    continue;
                }

                let left_n = i + 1;
                let right_n = n_samples - left_n;

                if left_n < min_samples_leaf || right_n < min_samples_leaf {
                    continue;
                }

                // KDE: treat min_child_weight as a count constraint (matches suff-stats path)
                // if (left_n as f64) < min_child_weight || (right_n as f64) < min_child_weight {
                //     continue;
                // }

                num_thresholds_tried += 1;

                let left_score = kde_nll_from_row_sums(&sorted_indices[..=i], &sum_to_left);
                let right_score = kde_nll_from_row_sums_right(&sorted_indices[i + 1..], &sum_to_left, rs_total);
                if !left_score.is_finite() || !right_score.is_finite() {
                    continue;
                }

                let improvement = current_score - (left_score + right_score);
                if improvement > local_best_gain {
                    local_best_gain = improvement;
                    local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
                }
            }

            if (reg_gamma > 0.0) && (num_thresholds_tried > 0) && (local_best_threshold.is_some()) {
                local_best_gain -= reg_gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
            }

            if let Some(threshold) = local_best_threshold {
                let cand = BestKdeSplit {
                    feature_idx,
                    threshold,
                    gain: local_best_gain,
                    thresholds_tried: num_thresholds_tried,
                };
                let mut best = best_results.lock().unwrap();
                if best.as_ref().map_or(true, |b| cand.gain > b.gain) {
                    *best = Some(cand);
                }
            }
        });
    } else {
        // ================================================================
        // FALLBACK KDE PATH (old):
        // Still uses parent_h if Parent policy, but no kernel-matrix cache.
        // Also fixes min_child_weight to be count-based (no y-summing).
        // ================================================================
        let d2 = precompute_pairwise_d2(y);

        let num_quantiles = (n_samples / stride).max(1);

        feature_idcs.par_iter().for_each(|&feature_idx| {
            let column = x.slice(s![.., feature_idx]);

            // let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            // sorted_indices.sort_unstable_by(|&a, &b| {
            //     column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
            // });

            let sorted_indices = sort_indices_by_feature(&column);

            let mut local_best_gain = 0.0;
            let mut local_best_threshold = None;
            let mut num_thresholds_tried = 0usize;

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

                // if (left_n as f64) < min_child_weight || (right_n as f64) < min_child_weight {
                //     continue;
                // }

                let feat_val = column[sorted_indices[split_idx]];
                let next_feat_val = column[sorted_indices[split_idx + 1]];
                if feat_val >= next_feat_val {
                    continue;
                }

                num_thresholds_tried += 1;

                let left_idx_slice = &sorted_indices[..=split_idx];
                let right_idx_slice = &sorted_indices[split_idx + 1..];

                let left_h = match config.bandwidth_policy {
                    BandwidthPolicy::Parent => parent_h,
                    BandwidthPolicy::PerSplit => compute_bandwidth_for_indices(&base_kde, config, y, left_idx_slice),
                };
                let right_h = match config.bandwidth_policy {
                    BandwidthPolicy::Parent => parent_h,
                    BandwidthPolicy::PerSplit => compute_bandwidth_for_indices(&base_kde, config, y, right_idx_slice),
                };

                let left_score = kde_subset_nll(left_idx_slice, &d2, n_samples, left_h, config.kernel, config.use_compact_support);
                let right_score = kde_subset_nll(right_idx_slice, &d2, n_samples, right_h, config.kernel, config.use_compact_support);

                let improvement = current_score - (left_score + right_score);
                if improvement > local_best_gain {
                    local_best_gain = improvement;
                    local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
                }
            }

            if (reg_gamma > 0.0) && (num_thresholds_tried > 0) && (local_best_threshold.is_some()) {
                local_best_gain -= reg_gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
            }

            if let Some(threshold) = local_best_threshold {
                let cand = BestKdeSplit {
                    feature_idx,
                    threshold,
                    gain: local_best_gain,
                    thresholds_tried: num_thresholds_tried,
                };
                let mut best = best_results.lock().unwrap();
                if best.as_ref().map_or(true, |b| cand.gain > b.gain) {
                    *best = Some(cand);
                }
            }
        });
    }

    // Extract
    let best = best_results.lock().unwrap().clone();
    let Some(best) = best else {
        return (None, None, 0.0, None, None, None, None);
    };

    // Reconstruct masks
    let feature_idx = best.feature_idx;
    let threshold = best.threshold;

    let column = x.slice(s![.., feature_idx]);
    let sorted_indices = sort_indices_by_feature(&column);
    let Some(split_idx) = split_idx_from_threshold(&column, &sorted_indices, threshold) else {
        return (None, None, 0.0, None, None, None, None);
    };

    let mut left_mask = Array1::from_elem(n_samples, false);
    let mut right_mask = Array1::from_elem(n_samples, false);
    for k in 0..=split_idx {
        left_mask[sorted_indices[k]] = true;
    }
    for k in (split_idx + 1)..n_samples {
        right_mask[sorted_indices[k]] = true;
    }

    // Refinement: re-score winning split with per-child bandwidth (top-K=1).
    // This keeps the returned gain comparable vs penalties/regularization.
    let mut final_gain = best.gain;
    if matches!(config.bandwidth_policy, BandwidthPolicy::Parent) {
        let left_indices = &sorted_indices[..=split_idx];
        let right_indices = &sorted_indices[split_idx + 1..];

        let left_h = compute_bandwidth_for_indices(&base_kde, config, y, left_indices);
        let right_h = compute_bandwidth_for_indices(&base_kde, config, y, right_indices);

        let d2 = precompute_pairwise_d2(y);
        let left_score = kde_subset_nll(left_indices, &d2, n_samples, left_h, config.kernel, config.use_compact_support);
        let right_score = kde_subset_nll(right_indices, &d2, n_samples, right_h, config.kernel, config.use_compact_support);

        if left_score.is_finite() && right_score.is_finite() {
            let mut refined_gain = current_score - (left_score + right_score);
            if (reg_gamma > 0.0) && (best.thresholds_tried > 0) {
                refined_gain -= reg_gamma * ((num_features_tried as f64).ln() + (best.thresholds_tried as f64).ln());
            }
            final_gain = refined_gain;
        }
    }

    (Some(feature_idx), Some(threshold), final_gain, Some(left_mask), Some(right_mask), None, None)
}

const INV_SQRT_2PI: f64 = 0.3989422804014327; // 1/sqrt(2*pi)
// Guardrail: dense n×n kernel matrix memory (f64) ~ 8*n*n bytes.
const MAX_KERNEL_MATRIX_ELEMS: usize = 12_000_000; // ~96MB
const GAUSSIAN_CUTOFF_STDDEVS: f64 = 4.0;

fn find_best_split_kde_fft(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    config: &KdeSplitConfig,
    parent_h: f64,
    eta: f64,
    reg_gamma: f64,
    col_idcs: Option<Array1<usize>>,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>) {
    let n_features = x.shape()[1];
    let n_samples = y.len();
    if n_samples < 2 || !parent_h.is_finite() || parent_h <= 0.0 {
        return (None, None, 0.0, None, None);
    }

    let base_kde = KdeDist {
        kernel: config.kernel,
        bandwidth_rule: config.bandwidth_rule,
        min_bandwidth: config.min_bandwidth,
    };

    // Grid
    let (grid_min, grid_max, n_bins) = pick_fft_grid(y, parent_h, config);
    if n_bins < 8 || !(grid_max > grid_min) {
        return (None, None, 0.0, None, None);
    }
    let dx = (grid_max - grid_min) / (n_bins as f64);
    if !dx.is_finite() || dx <= 0.0 {
        return (None, None, 0.0, None, None);
    }

    let fft_len = n_bins.next_power_of_two();
    let spec_len = fft_len / 2 + 1;

    // Bin each y once
    let mut y_bins: Vec<usize> = vec![0; n_samples];
    let mut total_counts_bins: Vec<f64> = vec![0.0; n_bins];
    for (i, &v) in y.iter().enumerate() {
        let b = bin_index(v, grid_min, dx, n_bins);
        y_bins[i] = b;
        total_counts_bins[b] += 1.0;
    }

    // FFT plans
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(fft_len);
    let c2r = planner.plan_fft_inverse(fft_len);

    // Gaussian frequency multiplier: exp(-0.5 * (h * omega_k)^2)
    let ldx = (fft_len as f64) * dx;
    let mut gauss_mult: Vec<f64> = vec![0.0; spec_len];
    for k in 0..spec_len {
        let omega = 2.0 * PI * (k as f64) / ldx;
        let a = parent_h * omega;
        gauss_mult[k] = (-0.5 * a * a).exp();
    }

    // Convolve total counts once (circular conv on padded length)
    let mut total_in: Vec<f64> = vec![0.0; fft_len];
    total_in[..n_bins].copy_from_slice(&total_counts_bins);
    let mut total_spec: Vec<Complex64> = r2c.make_output_vec();
    let mut scratch_fwd = r2c.make_scratch_vec();
    r2c.process_with_scratch(&mut total_in, &mut total_spec, &mut scratch_fwd)
        .expect("FFT forward failed");
    for (z, &m) in total_spec.iter_mut().zip(gauss_mult.iter()) {
        *z *= m;
    }
    let mut total_conv: Vec<f64> = c2r.make_output_vec();
    let mut scratch_inv = c2r.make_scratch_vec();
    c2r.process_with_scratch(&mut total_spec, &mut total_conv, &mut scratch_inv)
        .expect("FFT inverse failed");
    let norm = 1.0 / (fft_len as f64);
    for v in total_conv.iter_mut() {
        *v *= norm;
    }

    // Parent score (LOO-style): N * ln(N-1) - sum_i ln(sum_{j!=i} K(y_i,y_j))
    let k0 = INV_SQRT_2PI / parent_h;
    let current_score = kde_loo_nll_from_hist(&total_counts_bins, &total_conv[..n_bins], n_samples, k0);
    if !current_score.is_finite() {
        return (None, None, 0.0, None, None);
    }

    let feature_idcs: Vec<usize> = match col_idcs {
        Some(ref indices) => indices.to_vec(),
        None => (0..n_features).collect(),
    };
    let stride = (n_samples as f64 * eta).max(1.0) as usize;
    let num_features_tried = feature_idcs.len().max(1);

    // Track best (don’t store split_idx; we can derive it from threshold at the end)
    let best_results: Mutex<Option<BestKdeSplit>> = Mutex::new(None);

    feature_idcs.par_iter().for_each(|&feature_idx| {
        let column = x.slice(s![.., feature_idx]);
        // let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
        // sorted_indices.sort_unstable_by(|&a, &b| {
        //     column[a].partial_cmp(&column[b]).unwrap_or(std::cmp::Ordering::Equal)
        // });

        let sorted_indices = sort_indices_by_feature(&column);

        let mut left_counts_bins: Vec<f64> = vec![0.0; n_bins];

        // Reusable FFT buffers per feature
        let mut fft_in: Vec<f64> = vec![0.0; fft_len];
        let mut spec: Vec<Complex64> = r2c.make_output_vec();
        let mut scratch_fwd_local = r2c.make_scratch_vec();
        let mut conv_left: Vec<f64> = c2r.make_output_vec();
        let mut scratch_inv_local = c2r.make_scratch_vec();

        let mut local_best_gain = 0.0;
        let mut local_best_threshold: Option<f64> = None;
        let mut num_thresholds_tried = 0usize;

        for i in 0..(n_samples - 1) {
            // Move one point to the left
            let idx = sorted_indices[i];
            let b = y_bins[idx];
            left_counts_bins[b] += 1.0;

            if (i + 1) % stride != 0 {
                continue;
            }

            let feat_val = column[idx];
            let next_feat_val = column[sorted_indices[i + 1]];
            if feat_val >= next_feat_val {
                continue;
            }

            let left_n = i + 1;
            let right_n = n_samples - left_n;
            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }
            // if (left_n as f64) < min_child_weight || (right_n as f64) < min_child_weight {
            //     continue;
            // }

            num_thresholds_tried += 1;

            // FFT(left_counts)
            fft_in.fill(0.0);
            fft_in[..n_bins].copy_from_slice(&left_counts_bins);
            r2c.process_with_scratch(&mut fft_in, &mut spec, &mut scratch_fwd_local)
                .expect("FFT forward failed");
            for (z, &m) in spec.iter_mut().zip(gauss_mult.iter()) {
                *z *= m;
            }
            c2r.process_with_scratch(&mut spec, &mut conv_left, &mut scratch_inv_local)
                .expect("FFT inverse failed");
            for v in conv_left.iter_mut() {
                *v *= norm;
            }

            // Right conv = total_conv - left_conv (linearity)
            let left_score = kde_loo_nll_from_hist(&left_counts_bins, &conv_left[..n_bins], left_n, k0);
            if !left_score.is_finite() {
                continue;
            }

            // Compute right NLL without allocating right arrays
            let right_score = kde_loo_nll_from_hist_right(
                &total_counts_bins,
                &left_counts_bins,
                &total_conv[..n_bins],
                &conv_left[..n_bins],
                right_n,
                k0,
            );
            if !right_score.is_finite() {
                continue;
            }

            let gain = current_score - (left_score + right_score);
            if gain > local_best_gain {
                local_best_gain = gain;
                local_best_threshold = Some((feat_val + next_feat_val) / 2.0);
            }
        }

        if (reg_gamma > 0.0) && (num_thresholds_tried > 0) && local_best_threshold.is_some() {
            local_best_gain -= reg_gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
        }

        if let Some(threshold) = local_best_threshold {
            let cand = BestKdeSplit {
                feature_idx,
                threshold,
                gain: local_best_gain,
                thresholds_tried: num_thresholds_tried,
            };
            let mut best = best_results.lock().unwrap();
            if best.as_ref().map_or(true, |b| cand.gain > b.gain) {
                *best = Some(cand);
            }
        }
    });

    let best = best_results.lock().unwrap().clone();
    let Some(best) = best else {
        return (None, None, 0.0, None, None);
    };

    // Reconstruct masks
    let feature_idx = best.feature_idx;
    let threshold = best.threshold;

    let column = x.slice(s![.., feature_idx]);
    let sorted_indices = sort_indices_by_feature(&column);

    // let split_idx = (0..n_samples).find(|&i| x[[i, feature_idx]] > threshold).unwrap_or(n_samples - 1);
    let Some(split_idx) = split_idx_from_threshold(&column, &sorted_indices, threshold) else {
        return (None, None, 0.0, None, None);
    };

    let mut left_mask = Array1::from_elem(n_samples, false);
    let mut right_mask = Array1::from_elem(n_samples, false);
    for k in 0..=split_idx {
        left_mask[sorted_indices[k]] = true;
    }
    for k in (split_idx + 1)..n_samples {
        right_mask[sorted_indices[k]] = true;
    }

    // Refinement: re-score winning split with per-child bandwidth (top-K=1),
    // matching the non-FFT winner-only refinement convention.
    let mut final_gain = best.gain;
    if matches!(config.bandwidth_policy, BandwidthPolicy::Parent) {
        let left_indices = &sorted_indices[..=split_idx];
        let right_indices = &sorted_indices[split_idx + 1..];

        let left_h = compute_bandwidth_for_indices(&base_kde, config, y, left_indices);
        let right_h = compute_bandwidth_for_indices(&base_kde, config, y, right_indices);

        if left_h.is_finite() && left_h > 0.0 && right_h.is_finite() && right_h > 0.0 {
            let left_n = left_indices.len();
            let right_n = right_indices.len();

            // Build child histograms on the same grid
            let mut left_counts_bins: Vec<f64> = vec![0.0; n_bins];
            for &idx in left_indices {
                left_counts_bins[y_bins[idx]] += 1.0;
            }
            let mut right_counts_bins: Vec<f64> = vec![0.0; n_bins];
            for b in 0..n_bins {
                right_counts_bins[b] = total_counts_bins[b] - left_counts_bins[b];
            }

            // Convolve histograms with Gaussian kernel at the requested bandwidth.
            // (Two extra FFTs total; still O(n log n) overhead for refinement.)
            let mut gauss_mult_child: Vec<f64> = vec![0.0; spec_len];
            let mut fft_in: Vec<f64> = vec![0.0; fft_len];
            let mut spec: Vec<Complex64> = r2c.make_output_vec();
            let mut scratch_fwd_local = r2c.make_scratch_vec();
            let mut conv: Vec<f64> = c2r.make_output_vec();
            let mut scratch_inv_local = c2r.make_scratch_vec();

            let mut convolve_bins = |counts: &[f64], h: f64, out_bins: &mut [f64]| {
                for k in 0..spec_len {
                    let omega = 2.0 * PI * (k as f64) / ldx;
                    let a = h * omega;
                    gauss_mult_child[k] = (-0.5 * a * a).exp();
                }
                fft_in.fill(0.0);
                fft_in[..n_bins].copy_from_slice(counts);
                r2c.process_with_scratch(&mut fft_in, &mut spec, &mut scratch_fwd_local)
                    .expect("FFT forward failed");
                for (z, &m) in spec.iter_mut().zip(gauss_mult_child.iter()) {
                    *z *= m;
                }
                c2r.process_with_scratch(&mut spec, &mut conv, &mut scratch_inv_local)
                    .expect("FFT inverse failed");
                for v in conv.iter_mut() {
                    *v *= norm;
                }
                out_bins.copy_from_slice(&conv[..n_bins]);
            };

            let mut conv_left_bins: Vec<f64> = vec![0.0; n_bins];
            convolve_bins(&left_counts_bins, left_h, &mut conv_left_bins);
            let left_score = kde_loo_nll_from_hist(
                &left_counts_bins,
                &conv_left_bins,
                left_n,
                INV_SQRT_2PI / left_h,
            );

            let mut conv_right_bins: Vec<f64> = vec![0.0; n_bins];
            convolve_bins(&right_counts_bins, right_h, &mut conv_right_bins);
            let right_score = kde_loo_nll_from_hist(
                &right_counts_bins,
                &conv_right_bins,
                right_n,
                INV_SQRT_2PI / right_h,
            );

            if left_score.is_finite() && right_score.is_finite() {
                let mut refined_gain = current_score - (left_score + right_score);
                if (reg_gamma > 0.0) && (best.thresholds_tried > 0) {
                    refined_gain -= reg_gamma
                        * ((num_features_tried as f64).ln()
                            + (best.thresholds_tried as f64).ln());
                }
                final_gain = refined_gain;
            }
        }
    }

    (Some(feature_idx), Some(threshold), final_gain, Some(left_mask), Some(right_mask))
}

fn pick_fft_grid(y: &ArrayView1<f64>, parent_h: f64, config: &KdeSplitConfig) -> (f64, f64, usize) {
    let n_bins = config.fft_grid_points.unwrap_or(1024).max(64);
    let y_min = y.fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut min_v = config.fft_grid_min.unwrap_or(y_min);
    let mut max_v = config.fft_grid_max.unwrap_or(y_max);

    if !(max_v > min_v) {
        // Degenerate target range; create a small span
        min_v = y_min - 1.0;
        max_v = y_max + 1.0;
    }

    // Ensure enough padding to reduce circular wrap-around (Gaussian cutoff)
    let range = (max_v - min_v).abs();
    let pad_min = GAUSSIAN_CUTOFF_STDDEVS * parent_h;
    let pad = (0.2 * range).max(pad_min);
    (min_v - pad, max_v + pad, n_bins)
}

#[inline]
fn bin_index(v: f64, grid_min: f64, dx: f64, n_bins: usize) -> usize {
    let mut t = ((v - grid_min) / dx).floor();
    if !t.is_finite() {
        return 0;
    }
    if t < 0.0 {
        t = 0.0;
    }
    let mut idx = t as usize;
    if idx >= n_bins {
        idx = n_bins - 1;
    }
    idx
}

#[inline]
fn kde_loo_nll_from_hist(counts: &[f64], conv: &[f64], n: usize, k0: f64) -> f64 {
    if n < 2 {
        return f64::INFINITY;
    }
    let log_n1 = ((n - 1) as f64).ln();
    let mut sum_ln = 0.0f64;
    for i in 0..counts.len() {
        let c = counts[i];
        if c <= 0.0 {
            continue;
        }
        // Remove self contribution per point.
        // In exact Gaussian LOO-KDE, this quantity is strictly positive, but the FFT/hist
        // approximation can produce tiny negative/zero values (discretization/roundoff).
        // Clamp to a tiny positive floor to avoid returning INF (which causes "no split").
        let s0 = conv[i] - k0;
        if !s0.is_finite() {
            return f64::INFINITY;
        }
        let s = if s0 > 0.0 { s0 } else { f64::MIN_POSITIVE };
        sum_ln += c * s.ln();
    }
    (n as f64) * log_n1 - sum_ln
}

#[inline]
fn kde_loo_nll_from_hist_right(
    total_counts: &[f64],
    left_counts: &[f64],
    total_conv: &[f64],
    left_conv: &[f64],
    n_right: usize,
    k0: f64,
) -> f64 {
    if n_right < 2 {
        return f64::INFINITY;
    }
    let log_n1 = ((n_right - 1) as f64).ln();
    let mut sum_ln = 0.0f64;
    for i in 0..total_counts.len() {
        let c = total_counts[i] - left_counts[i];
        if c <= 0.0 {
            continue;
        }
        let s0 = (total_conv[i] - left_conv[i]) - k0;
        if !s0.is_finite() {
            return f64::INFINITY;
        }
        let s = if s0 > 0.0 { s0 } else { f64::MIN_POSITIVE };
        sum_ln += c * s.ln();
    }
    (n_right as f64) * log_n1 - sum_ln
}

fn try_precompute_kernel_matrix(
    y: &ArrayView1<f64>,
    h: f64,
    kernel: KernelType,
    use_compact_support: bool,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let nn = n.checked_mul(n)?;
    if nn > MAX_KERNEL_MATRIX_ELEMS {
        return None;
    }
    if h <= 0.0 || !h.is_finite() {
        return None;
    }

    let inv_h = 1.0 / h;
    let inv_h2 = inv_h * inv_h;

    let mut k = vec![0.0f64; nn];
    let mut row_sums = vec![0.0f64; n];

    match kernel {
        KernelType::Gaussian => {
            let norm = INV_SQRT_2PI * inv_h;
            let cutoff2 = if use_compact_support { (GAUSSIAN_CUTOFF_STDDEVS * h).powi(2) } else { f64::INFINITY };
            for i in 0..n {
                for j in (i + 1)..n {
                    let diff = y[i] - y[j];
                    if diff * diff > cutoff2 {
                        continue;
                    }
                    let kij = norm * (-0.5 * diff * diff * inv_h2).exp();
                    k[i * n + j] = kij;
                    k[j * n + i] = kij;
                    row_sums[i] += kij;
                    row_sums[j] += kij;
                }
            }
        }
        KernelType::Epanechnikov => {
            let norm = 0.75 * inv_h;
            for i in 0..n {
                for j in (i + 1)..n {
                    let diff = y[i] - y[j];
                    let u2 = diff * diff * inv_h2;
                    let kij = if u2 <= 1.0 {
                        let one_minus = 1.0 - u2;
                        if one_minus > 0.0 { norm * one_minus } else { 0.0 }
                    } else {
                        0.0
                    };
                    k[i * n + j] = kij;
                    k[j * n + i] = kij;
                    row_sums[i] += kij;
                    row_sums[j] += kij;
                }
            }
        }
    }

    Some((k, row_sums))
}

#[inline]
fn kde_nll_from_row_sums_right(indices: &[usize], sum_to_left: &[f64], rs_total: &[f64]) -> f64 {
    let m = indices.len();
    if m < 2 {
        return f64::INFINITY;
    }
    let log_m1 = ((m - 1) as f64).ln();
    let mut sum_ln = 0.0f64;
    for &idx in indices {
        let s = rs_total[idx] - sum_to_left[idx];
        if s <= 0.0 || !s.is_finite() {
            return f64::INFINITY;
        }
        sum_ln += s.ln();
    }
    (m as f64) * log_m1 - sum_ln
}


#[inline]
fn kde_nll_from_row_sums(indices: &[usize], row_sums: &[f64]) -> f64 {
    let m = indices.len();
    if m < 2 {
        return f64::INFINITY;
    }
    let log_m1 = ((m - 1) as f64).ln();
    let mut sum_ln = 0.0f64;
    for &idx in indices {
        let s = row_sums[idx];
        if s <= 0.0 || !s.is_finite() {
            return f64::INFINITY;
        }
        sum_ln += s.ln();
    }
    (m as f64) * log_m1 - sum_ln
}

#[inline]
fn kde_full_nll_from_row_sums(row_sums: &[f64]) -> f64 {
    let n = row_sums.len();
    if n < 2 {
        return f64::INFINITY;
    }
    let log_n1 = ((n - 1) as f64).ln();
    let mut sum_ln = 0.0f64;
    for &s in row_sums {
        if s <= 0.0 || !s.is_finite() {
            return f64::INFINITY;
        }
        sum_ln += s.ln();
    }
    (n as f64) * log_n1 - sum_ln
}

fn precompute_pairwise_d2(y: &ArrayView1<f64>) -> Vec<f64> {
    let n = y.len();
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = y[i] - y[j];
            let val = diff * diff;
            d2[i * n + j] = val;
            d2[j * n + i] = val;
        }
    }
    d2
}

fn compute_bandwidth_for_indices(
    base: &KdeDist,
    config: &KdeSplitConfig,
    y: &ArrayView1<f64>,
    indices: &[usize],
) -> f64 {
    // Parent/full subset: compute on the subset data (used for Parent once, or for PerSplit repeatedly)
    let subset: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
    let data_h = base.compute_bandwidth(&subset.view());

    if let (Some(prior_h), Some(m_h)) = (config.prior_h, config.m_h) {
        let n = subset.len() as f64;
        if n <= 0.0 {
            return base.min_bandwidth;
        }
        ((m_h * prior_h + n * data_h) / (m_h + n)).max(base.min_bandwidth)
    } else {
        data_h
    }
}

fn kde_subset_nll(
    indices: &[usize],
    d2: &[f64],
    n_total: usize,
    h: f64,
    kernel: KernelType,
    use_compact_support: bool,
) -> f64 {
    let m = indices.len();
    if m < 2 {
        return f64::INFINITY;
    }

    let inv_h2 = 1.0 / (h * h);
    let cutoff2 = if use_compact_support { (GAUSSIAN_CUTOFF_STDDEVS * h).powi(2) } else { f64::INFINITY };
    let log_norm = match kernel {
        KernelType::Gaussian => {
            const LOG_2PI: f64 = 1.8378770664093453;
            -0.5 * LOG_2PI - h.ln()
        }
        KernelType::Epanechnikov => {
            0.75f64.ln() - h.ln()
        }
    };

    let log_m1 = ((m - 1) as f64).ln();
    let mut total_ll = 0.0f64;

    for &i in indices {
        let mut max_val = f64::NEG_INFINITY;
        let mut sum_exp = 0.0f64;

        for &j in indices {
            if i == j {
                continue;
            }
            let dij2 = d2[i * n_total + j];
            let v = match kernel {
                KernelType::Gaussian => {
                    if dij2 > cutoff2 { f64::NEG_INFINITY } else { log_norm - 0.5 * dij2 * inv_h2 }
                }
                KernelType::Epanechnikov => {
                    let u2 = dij2 * inv_h2;
                    if u2 <= 1.0 {
                        let one_minus = 1.0 - u2;
                        if one_minus > 0.0 {
                            log_norm + one_minus.ln()
                        } else {
                            f64::NEG_INFINITY
                        }
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            };

            if v > max_val {
                sum_exp = sum_exp * (max_val - v).exp() + 1.0;
                max_val = v;
            } else {
                sum_exp += (v - max_val).exp();
            }
        }

        if max_val == f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        let ll_i = max_val + sum_exp.ln() - log_m1;
        total_ll += ll_i;
    }

    -total_ll
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
