"""Helpers for effective split-score regime aliases in benchmark configs."""

from __future__ import annotations

from typing import Any


def expand_score_regime(params: dict[str, Any]) -> dict[str, Any]:
    """Expand ``score_regime`` into concrete distribution parameters.

    The alias keeps the Optuna search space aligned with the paper-facing
    regimes while preserving downstream result fields: ``score_method`` and
    ``score_correction``.
    """
    expanded = dict(params)
    regime = expanded.pop("score_regime", None)
    if regime is None:
        return expanded

    match regime:
        case "nle":
            expanded["score_method"] = "nle"
            expanded["score_correction"] = None
        case "nll_bic":
            expanded["score_method"] = "nll"
            expanded["score_correction"] = "bic"
        case "nll":
            expanded["score_method"] = "nll"
            expanded["score_correction"] = None
        case _:
            raise ValueError(f"Unknown score_regime: {regime}")

    return expanded
