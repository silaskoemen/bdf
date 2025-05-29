#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, arr1, arr2};
    use crate::distribution::{Distribution, NormalNormal, NormalNormalSpec};
    use crate::splitter::{find_best_split, create_threshold_masks};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_normal_distribution_nll() {
        // Test NLL calculation with known data
        let dist = NormalNormal::new(&NormalNormalSpec {
            prior_mean: 0.0,
            prior_std: 1.0
        });

        // Single value
        let data_single = arr1(&[2.0]);
        let nll_single = dist.nll(&data_single.view());
        assert!(nll_single > 0.0);

        // Multiple values
        let data_multi = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let nll_multi = dist.nll(&data_multi.view());
        assert!(nll_multi > 0.0);

        // Empty array should return zero
        let data_empty = arr1(&[]);
        let nll_empty = dist.nll(&data_empty.view());
        assert_eq!(nll_empty, 0.0);

        // Test that higher variance data has higher NLL
        let data_low_var = arr1(&[1.0, 1.1, 0.9, 1.0]);
        let data_high_var = arr1(&[0.0, 0.0, 10.0, 10.0]);

        let nll_low_var = dist.nll(&data_low_var.view());
        let nll_high_var = dist.nll(&data_high_var.view());

        assert!(nll_high_var > nll_low_var);
    }

    #[test]
    fn test_threshold_masks_creation() {
        let column = arr1(&[1.0, 3.0, 2.0, 4.0]);
        let threshold = 2.5;

        let (left, right) = create_threshold_masks(&column.view(), threshold);

        // Check correct classification
        assert_eq!(left[0], true);  // 1.0 <= 2.5
        assert_eq!(left[1], false); // 3.0 > 2.5
        assert_eq!(left[2], true);  // 2.0 <= 2.5
        assert_eq!(left[3], false); // 4.0 > 2.5

        // Right mask should be the complement
        for i in 0..column.len() {
            assert_eq!(right[i], !left[i]);
        }
    }

    #[test]
    fn test_find_best_split_basic() {
        // Create a step function for testing
        let x = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],
                       [5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]);
        let y = arr1(&[1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]);

        let dist = NormalNormal::new(&NormalNormalSpec {
            prior_mean: 0.0,
            prior_std: 1.0
        });

        let result = find_best_split(
            &x.view(),
            &y.view(),
            1, // min_samples_leaf
            0.0, // min_child_weight
            &dist,
            0.1, // eta
            None // col_indices
        );

        // Extract results
        let (feat_idx, threshold, loss, left_mask, right_mask) = result;

        // Should find the obvious split on first feature
        assert_eq!(feat_idx, Some(0));

        // Threshold should be between 4 and 5
        if let Some(t) = threshold {
            assert!(t > 4.0 && t < 5.0);
        } else {
            panic!("No threshold found");
        }

        // Loss reduction should be positive
        assert!(loss > 0.0);

        // Check masks are complementary and correct
        if let (Some(left), Some(right)) = (left_mask, right_mask) {
            for i in 0..x.shape()[0] {
                assert_eq!(left[i], !right[i]);
                if i < 4 {
                    assert_eq!(left[i], true);
                } else {
                    assert_eq!(left[i], false);
                }
            }
        } else {
            panic!("No masks returned");
        }
    }

    #[test]
    fn test_find_best_split_constraints() {
        // Test that min_samples_leaf constraint is respected
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
        let y = arr1(&[1.0, 1.0, 1.0, 5.0, 5.0, 5.0]);

        let dist = NormalNormal::new(&NormalNormalSpec {
            prior_mean: 0.0,
            prior_std: 1.0
        });

        // Set min_samples_leaf too high to allow any split
        let result = find_best_split(
            &x.view(),
            &y.view(),
            4, // min_samples_leaf - won't find valid split
            0.0,
            &dist,
            0.1,
            None
        );

        // Should not find a valid split
        assert_eq!(result.0, None);
        assert_eq!(result.1, None);
    }

    #[test]
    fn test_parallel_split_finding() {
        // Test that parallel split finding gives same results
        // Create data with two features, only one has a good split
        let x = arr2(&[
            [1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0],
            [5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0]
        ]);
        let y = arr1(&[1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]);

        let dist = NormalNormal::new(&NormalNormalSpec {
            prior_mean: 0.0,
            prior_std: 1.0
        });

        // Run split finding with all features
        let result_all = find_best_split(
            &x.view(),
            &y.view(),
            1,
            0.0,
            &dist,
            0.1,
            None
        );

        // Run with explicit column indices for each feature
        let result_feat0 = find_best_split(
            &x.view(),
            &y.view(),
            1,
            0.0,
            &dist,
            0.1,
            Some(arr1(&[0]))
        );

        let result_feat1 = find_best_split(
            &x.view(),
            &y.view(),
            1,
            0.0,
            &dist,
            0.1,
            Some(arr1(&[1]))
        );

        // The overall result should match the feature 0 result (the better split)
        assert_eq!(result_all.0, Some(0));
        assert_eq!(result_all.0, result_feat0.0);

        // Feature 1 should not find a valid split (constant feature)
        assert_eq!(result_feat1.0, None);
    }

    #[test]
    fn test_mutex_thread_safety() {
        // Test thread safety of the mutex implementation
        use std::thread;

        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut num = counter_clone.lock().unwrap();
                    *num += 1;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*counter.lock().unwrap(), 1000);
    }
}
