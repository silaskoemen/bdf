use ndarray::{ArrayView1, ArrayView2, Array1, s};
use crate::distribution::Distribution;
use rayon::prelude::*;
use std::sync::Mutex;

pub fn find_best_split(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution: &dyn Distribution,
    eta: f64,
    col_idcs: Option<Array1<usize>>,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>) {
    let n_features = x.shape()[1];

    // Shared best results using Mutex for thread safety
    let best_results = Mutex::new((
        None as Option<usize>,
        None as Option<f64>,
        0.0f64,
        None as Option<Array1<bool>>,
        None as Option<Array1<bool>>
    ));

    let n_thresholds = (1.0 / eta).ceil() as usize;

    // Current node NLL (calculated once)
    let current_nll = distribution.nll(y);

    // Determine which features to iterate through
    let feature_idcs: Vec<usize> = match col_idcs {
        Some(ref indices) => indices.to_vec(),
        None => (0..n_features).collect(),
    };

    feature_idcs.par_iter().for_each(|&feature_idx| {
        // Local best tracking variables
        let mut local_best_loss = 0.0;
        let mut local_best_threshold = None;
        let mut local_best_left = None;
        let mut local_best_right = None;

        // Extract column and compute quantiles
        let column = x.slice(s![.., feature_idx]);
        let mut values: Vec<f64> = column.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Generate thresholds from quantiles
        let mut thresholds: Vec<f64> = Vec::with_capacity(n_thresholds);
        for i in 0..n_thresholds {
            let q = i as f64 / (n_thresholds as f64);
            let idx = (q * (values.len() as f64)) as usize;
            if idx < values.len() {
                thresholds.push(values[idx]);
            }
        }

        // Deduplicate thresholds
        thresholds.dedup();

        for i in 1..thresholds.len() {
            if thresholds[i] == thresholds[i-1] {
                continue;
            }
            let threshold = (thresholds[i] + thresholds[i-1]) / 2.0;
            let (left_indices, right_indices) = create_threshold_masks(&column, threshold);

            // Check constraints
            let left_count = left_indices.iter().filter(|&&v| v).count();
            let right_count = right_indices.iter().filter(|&&v| v).count();

            if left_count < min_samples_leaf || right_count < min_samples_leaf ||
            (left_count as f64) < min_child_weight || (right_count as f64) < min_child_weight {
                continue;
            }

            // Calculate NLL for each child
            let left_y = Array1::from_iter(
                y.iter()
                .zip(left_indices.iter())
                .filter(|(_, mask)| **mask)
                .map(|(&val, _)| val)
            );

            let right_y = Array1::from_iter(
                y.iter()
                .zip(right_indices.iter())
                .filter(|(_, mask)| **mask)
                .map(|(&val, _)| val)
            );

            let left_nll = distribution.nll(&left_y.view());
            let right_nll = distribution.nll(&right_y.view());

            let loss_reduction = current_nll - (left_nll + right_nll);

            // Update local best (no mutex needed yet)
            if loss_reduction > local_best_loss {
                local_best_loss = loss_reduction;
                local_best_threshold = Some(threshold);
                local_best_left = Some(left_indices);
                local_best_right = Some(right_indices);
            }
        }

        if let Some(threshold) = local_best_threshold {
            let mut best = best_results.lock().unwrap();
            if local_best_loss > best.2 {
                *best = (
                    Some(feature_idx),
                    Some(threshold),
                    local_best_loss,
                    local_best_left,
                    local_best_right
                );
            }
        }
    });

    // Extract results from mutex
    let best = best_results.lock().unwrap();
    (best.0, best.1, best.2, best.3.clone(), best.4.clone())
}

fn create_threshold_masks(column: &ArrayView1<f64>, threshold: f64) -> (Array1<bool>, Array1<bool>) {
    let len = column.len();
    let mut left_mask = Array1::from_elem(len, false);
    let mut right_mask = Array1::from_elem(len, false);

    // Simple scalar implementation - no SIMD
    for i in 0..len {
        let val = column[i];
        left_mask[i] = val <= threshold;
        right_mask[i] = val > threshold;
    }

    (left_mask, right_mask)
}
