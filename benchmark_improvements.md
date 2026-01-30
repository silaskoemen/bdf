# ⚠️ High-Priority Issues

## No Dataset Characteristics Analysis

File: benchmarks/pipeline/data.py:27-40

Missing metadata:
- Target skewness/kurtosis (affects distribution choice)
- Signal-to-noise ratio
- Feature correlations
- Class imbalance

Fix: Can't explain why BDF works better on some datasets without this
analysis.
