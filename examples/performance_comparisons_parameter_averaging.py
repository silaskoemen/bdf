import time

import numpy as np
import pandas as pd


def benchmark_global_parameter_averaging(n_trees=100, n_samples=500, n_params=2):
    """Benchmark methods for averaging params across all samples and trees."""

    # Generate toy parameter dictionaries
    np.random.seed(42)
    all_params = []
    param_names = [f"param_{i}" for i in range(n_params)]

    # Create dictionaries for each tree
    for _ in range(n_trees):
        tree_params = {}
        for i in range(n_samples):
            tree_params[i] = {name: np.random.random() for name in param_names}
        all_params.append(tree_params)

    # Method 1: Loop with lists
    start_time = time.time()
    param_keys = list(all_params[0][0].keys())
    param_values = {key: [] for key in param_keys}

    for tree_dict in all_params:
        for sample_id, sample_dict in tree_dict.items():
            for key, value in sample_dict.items():
                param_values[key].append(value)

    result1 = {key: np.mean(values) for key, values in param_values.items()}
    time1 = time.time() - start_time
    print(f"Loop with lists: {time1:.4f} seconds")

    # Method 2: Using pandas
    start_time = time.time()
    flattened_dicts = []
    for tree_dict in all_params:
        for sample_id, sample_dict in tree_dict.items():
            flattened_dicts.append(sample_dict)

    df = pd.DataFrame(flattened_dicts)
    result2 = df.mean().to_dict()
    time2 = time.time() - start_time
    print(f"Pandas DataFrame: {time2:.4f} seconds")

    # Verify results
    for key in result1:
        assert np.isclose(result1[key], result2[key])

    print("All methods produce identical results.")
    return {"lists": time1, "pandas": time2}


benchmark_global_parameter_averaging(n_trees=100, n_samples=500, n_params=2)
