// In splitter.rs
use ndarray::{ArrayView1, ArrayView2, Array1, s};
use crate::distribution::Distribution;

pub fn find_best_split_rust(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution: &dyn Distribution,
    eta: f64,
    col_idcs: Option<ArrayView1<usize>>,
) -> (Option<usize>, Option<f64>, f64, Option<Array1<bool>>, Option<Array1<bool>>) {
    let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
    
    let mut best_feature: Option<usize> = None;
    let mut best_threshold: Option<f64> = None;
    let mut best_loss_reduction: f64 = 0.0;
    let mut best_left_indices: Option<Array1<bool>> = None;
    let mut best_right_indices: Option<Array1<bool>> = None;
    
    let n_thresholds = (1.0 / eta).ceil() as usize;
    
    // Current node NLL
    let current_nll = distribution.nll(y);
    
    // Determine which features to iterate through
    let feature_idcs: Vec<usize> = match col_idcs {
        Some(indices) => indices.to_vec(),
        None => (0..n_features).collect(),
    };
    
    for &feature_idx in feature_idcs.iter() {
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
        
        // For each pair of adjacent thresholds, use the midpoint
        for i in 1..thresholds.len() {
            if thresholds[i] == thresholds[i-1] {
                continue;
            }
            
            let threshold = (thresholds[i] + thresholds[i-1]) / 2.0;
            
            // Split data
            let left_indices: Array1<bool> = column.mapv(|val| val <= threshold);
            let right_indices: Array1<bool> = column.mapv(|val| val > threshold);
            
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
                .filter(|(_, mask)| **mask)  // Just dereference once with *mask
                .map(|(&val, _)| val)
            );
            
            let left_nll = distribution.nll(&left_y.view());
            let right_nll = distribution.nll(&right_y.view());
            
            // Calculate loss reduction
            let loss_reduction = current_nll - (left_nll + right_nll);
            
            if loss_reduction > best_loss_reduction {
                best_loss_reduction = loss_reduction;
                best_feature = Some(feature_idx);
                best_threshold = Some(threshold);
                best_left_indices = Some(left_indices.clone());
                best_right_indices = Some(right_indices.clone());
            }
        }
    }
    
    (best_feature, best_threshold, best_loss_reduction, best_left_indices, best_right_indices)
}
