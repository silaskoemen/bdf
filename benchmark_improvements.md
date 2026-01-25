# ⚠️ High-Priority Issues

## Small Dataset Collection

File: benchmarks/pipeline/data.py:143-222

Only 11 regression + 3 classification datasets:
- All UCI/standard benchmarks
- No synthetic datasets with known ground truth
- No large-scale datasets (100k+ samples)
- No high-dimensional datasets (100+ features)

Fix: Add diverse datasets including synthetic ones to validate
calibration.

## No Dataset Characteristics Analysis

File: benchmarks/pipeline/data.py:27-40

Missing metadata:
- Target skewness/kurtosis (affects distribution choice)
- Signal-to-noise ratio
- Feature correlations
- Class imbalance

Fix: Can't explain why BDF works better on some datasets without this
analysis.
