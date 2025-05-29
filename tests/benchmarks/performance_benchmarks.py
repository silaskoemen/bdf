import time

import bdf_rust
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from bdf.distributions.distribution_manager import DistributionManager as DM
from bdf.tree_classes.bdf_node import BDFNode
from bdf.tree_classes.bdf_regressor import BDFRegressor


def benchmark_split_finding(n_samples=1000, n_features=10, n_runs=3):
    """Benchmark Python vs Rust split finding implementations"""
    # Generate synthetic data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)  # type: ignore

    # Create distribution
    dist = DM.create_distribution("normal", {"mean": 0, "std": 1})
    rust_spec = DM.to_rust_spec(dist)

    # Python implementation timing
    python_times = []
    for _ in range(n_runs):
        start = time.time()
        # Call Python version
        result_py = BDFNode(distribution=dist)._find_best_split_python(X, y, 1, 0.0, None)
        python_times.append(time.time() - start)

    # Rust implementation timing
    rust_times = []
    for _ in range(n_runs):
        start = time.time()
        # Call Rust version
        result_rust = bdf_rust.find_best_split(X, y, 1, 0.0, rust_spec, 0.1, None)  # type: ignore
        rust_times.append(time.time() - start)

    # Print results
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Python: {np.mean(python_times):.4f}s (±{np.std(python_times):.4f}s)")
    print(f"Rust:   {np.mean(rust_times):.4f}s (±{np.std(rust_times):.4f}s)")
    print(f"Speedup: {np.mean(python_times)/np.mean(rust_times):.2f}x\n")

    # Return results
    return np.mean(python_times), np.mean(rust_times)


def benchmark_training(n_samples=1000, n_features=10, n_trees=100):
    """Benchmark full model training"""
    # Generate synthetic data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)  # type: ignore

    # BDF timing
    start = time.time()
    bdf = BDFRegressor(dist="normal", prior_params={"mean": 0, "std": 1})
    bdf.fit(X, y)
    bdf_time = time.time() - start

    # RandomForest timing
    start = time.time()
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    rf.fit(X, y)
    rf_time = time.time() - start

    print(f"Training time for {n_trees} trees, {n_samples} samples, {n_features} features:")
    print(f"BDF: {bdf_time:.4f}s")
    print(f"RandomForest: {rf_time:.4f}s")
    print(f"Ratio: {bdf_time/rf_time:.2f}x\n")

    return bdf_time, rf_time


if __name__ == "__main__":
    print("Running split finding benchmarks...")
    for samples in [100, 1000, 10000]:
        for features in [5, 20]:
            benchmark_split_finding(samples, features)

    print("\nRunning training benchmarks...")
    benchmark_training(1000, 10, 100)
    benchmark_training(10000, 20, 100)
