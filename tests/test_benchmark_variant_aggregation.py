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
