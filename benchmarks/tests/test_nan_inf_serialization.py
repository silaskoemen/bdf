"""Test NaN/inf handling with YAML serialization for benchmark results."""

import os
import tempfile

import numpy as np
import pytest
import yaml

from ..metrics.regression import _safe_mae, _safe_mape, _safe_mse, _safe_rmse


def test_yaml_nan_inf_roundtrip():
    """Test that NaN and inf values are preserved through YAML save/load."""
    # Example results with NaN and inf values (like from NGBoost overflows)
    results = {
        "model": "test_model",
        "metrics": {
            "mse": {"mean": 0.123, "std": 0.045},
            "mae": {"mean": float("nan"), "std": float("nan")},  # Failed metric
            "rmse": {"mean": float("inf"), "std": 0.1},  # Infinite prediction
            "r2": {"mean": -float("inf"), "std": 0.2},  # Negative infinity
            "normal": {"mean": 0.5, "std": 0.05},  # Normal values
        },
    }

    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)

    try:
        # Load back - NaN/inf should be preserved perfectly
        with open(temp_path, "r") as f:
            loaded_results = yaml.safe_load(f)

        # Verify normal values
        assert loaded_results["metrics"]["mse"]["mean"] == 0.123
        assert loaded_results["metrics"]["mse"]["std"] == 0.045
        assert loaded_results["metrics"]["normal"]["mean"] == 0.5

        # Verify NaN preservation
        assert np.isnan(loaded_results["metrics"]["mae"]["mean"])
        assert np.isnan(loaded_results["metrics"]["mae"]["std"])

        # Verify positive infinity
        assert np.isinf(loaded_results["metrics"]["rmse"]["mean"])
        assert loaded_results["metrics"]["rmse"]["mean"] > 0

        # Verify negative infinity
        assert np.isinf(loaded_results["metrics"]["r2"]["mean"])
        assert loaded_results["metrics"]["r2"]["mean"] < 0

    finally:
        # Clean up
        os.unlink(temp_path)


def test_yaml_format_readability():
    """Test that the YAML output is human-readable."""
    results = {
        "metrics": {
            "valid": 1.23,
            "failed": float("nan"),
            "overflow": float("inf"),
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name
        yaml.dump(results, f, default_flow_style=False)

    try:
        # Read the raw YAML text
        with open(temp_path, "r") as f:
            yaml_text = f.read()

        # Check that special values use YAML syntax
        assert ".nan" in yaml_text or "nan" in yaml_text.lower()
        assert ".inf" in yaml_text or "inf" in yaml_text.lower()

    finally:
        os.unlink(temp_path)


def test_safe_metrics_return_nan_on_inf():
    """Test that safe metrics return NaN when predictions contain inf."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_with_inf = np.array([1.1, float("inf"), 3.1, 4.1])

    # All metrics should return NaN when predictions contain inf
    assert np.isnan(_safe_mse(y_true, y_pred_with_inf))
    assert np.isnan(_safe_mae(y_true, y_pred_with_inf))
    assert np.isnan(_safe_rmse(y_true, y_pred_with_inf))
    assert np.isnan(_safe_mape(y_true, y_pred_with_inf))


def test_safe_metrics_work_on_finite():
    """Test that safe metrics work correctly on finite predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])

    # All metrics should return finite values
    assert np.isfinite(_safe_mse(y_true, y_pred))
    assert np.isfinite(_safe_mae(y_true, y_pred))
    assert np.isfinite(_safe_rmse(y_true, y_pred))
    assert np.isfinite(_safe_mape(y_true, y_pred))

    # Check approximate expected values
    assert _safe_mse(y_true, y_pred) == pytest.approx(0.015, abs=0.001)
    assert _safe_mae(y_true, y_pred) == pytest.approx(0.125, abs=0.001)
