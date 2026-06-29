from pathlib import Path

import pytest
import yaml

from benchmarks.utils.yaml_loader import aggregate_model_variants


def _write_result(results_dir: Path, model: str, tuning_value: float) -> None:
    result = {
        "datasets": {
            "dataset": {
                "tuning": {"metric": "crps", "best_value": tuning_value},
                "metrics": {"crps": [tuning_value] * 9},
            }
        }
    }
    with (results_dir / f"res_{model}.yaml").open("w") as stream:
        yaml.safe_dump(result, stream)


def test_fused_variant_selection_can_require_complete_family(tmp_path):
    _write_result(tmp_path, "normal", 2.0)

    with pytest.warns(UserWarning, match="Result file not found for model 'laplace'"):
        with pytest.raises(ValueError, match="Missing required NGBoost models"):
            aggregate_model_variants(
                results_dir=tmp_path,
                model_variants=["normal", "laplace"],
                family_name="NGBoost",
                expected_eval_folds=9,
                require_all_variants=True,
            )

    _write_result(tmp_path, "laplace", 1.0)
    aggregated, selected = aggregate_model_variants(
        results_dir=tmp_path,
        model_variants=["normal", "laplace"],
        family_name="NGBoost",
        expected_eval_folds=9,
        require_all_variants=True,
    )

    assert selected == {"dataset": "laplace"}
    assert aggregated["datasets"]["dataset"]["tuning"]["best_value"] == 1.0
    assert aggregated["metadata"]["require_all_variants"] is True


def test_fused_variant_selection_excludes_nonfinite_required_metrics(tmp_path):
    _write_result(tmp_path, "normal", 2.0)
    _write_result(tmp_path, "laplace", 1.0)

    for model, rmse_values in (("normal", [2.0] * 9), ("laplace", [1.0] * 8 + [float("inf")])):
        result_path = tmp_path / f"res_{model}.yaml"
        result = yaml.safe_load(result_path.read_text())
        result["datasets"]["dataset"]["metrics"]["rmse"] = rmse_values
        result_path.write_text(yaml.safe_dump(result))

    with pytest.warns(UserWarning, match="rmse contains non-finite values"):
        aggregated, selected = aggregate_model_variants(
            results_dir=tmp_path,
            model_variants=["normal", "laplace"],
            family_name="NGBoost",
            expected_eval_folds=9,
            required_eval_metrics=["crps", "rmse"],
            require_all_variants=True,
        )

    assert selected == {"dataset": "normal"}
    assert aggregated["metadata"]["required_eval_metrics"] == ["crps", "rmse"]
    assert aggregated["metadata"]["excluded_variants"] == {"dataset": {"laplace": ["rmse contains non-finite values"]}}
