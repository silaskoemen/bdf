"""Generate the synthetic DGP summary table consumed by the TMLR paper."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks/results/synthetic_dgp"
OUT_PATH = RESULTS_DIR / "tables/synthetic_dgp_summary.tex"

DGPS = [
    ("heteroscedastic_sinusoidal", "Heteroscedastic"),
    ("bimodal_mixture", "Bimodal"),
    ("heavy_tailed", "Heavy-tailed"),
    ("step_function", "Step"),
    ("sparse_sampling", "Sparse"),
]

MODELS = [
    ("BDFNormal", "BDF-N"),
    ("BDFKDE", "BDF-KDE"),
    ("QRF", "QRF"),
    ("DRF", "DRF"),
    ("NGBoost", "NGBoost"),
    ("CatBoostUncertainty", "CatBoost-UQ"),
    ("ConformalRF", "ConfRF"),
    ("BART", "BART"),
    ("Climatological", "Clim."),
]

METRICS = [
    ("crps", "CRPS", "lower"),
    ("coverage_90", "Cov. 90\\%", "coverage90"),
    ("interval_score_90", "IS90", "lower"),
]


def _has_metrics(model_data: dict) -> bool:
    metrics = model_data.get("aggregated_metrics", {})
    return isinstance(metrics, dict) and bool(metrics)


def _load_dgp_results(dgp_name: str) -> dict:
    preferred = [
        RESULTS_DIR / f"{dgp_name}_results.json",
        RESULTS_DIR / f"{dgp_name}_results_core.json",
    ]
    shards = sorted(
        p for p in RESULTS_DIR.glob(f"{dgp_name}_results_*.json") if p.name != f"{dgp_name}_results_core.json"
    )
    paths = [p for p in preferred if p.exists()] + shards
    if not paths:
        raise FileNotFoundError(f"No synthetic results found for {dgp_name}")

    with paths[0].open() as handle:
        merged = json.load(handle)
    merged.setdefault("models", {})
    merged["models"] = {name: data for name, data in merged["models"].items() if _has_metrics(data)}

    for path in paths[1:]:
        with path.open() as handle:
            data = json.load(handle)
        for model_name, model_data in data.get("models", {}).items():
            if _has_metrics(model_data) and model_name not in merged["models"]:
                merged["models"][model_name] = model_data
    return merged


def _metric_value(results: dict, model: str, metric: str) -> float | None:
    info = results.get("models", {}).get(model, {}).get("aggregated_metrics", {}).get(metric)
    if not isinstance(info, dict) or "mean" not in info:
        return None
    return float(info["mean"])


def _score_for_best(metric_kind: str, value: float) -> float:
    if metric_kind == "coverage90":
        return abs(value - 0.9)
    return value


def _format(metric_kind: str, value: float | None, is_best: bool) -> str:
    if value is None:
        text = "--"
    elif metric_kind == "coverage90":
        text = f"{100 * value:.1f}\\%"
    else:
        text = f"{value:.3f}"
    if is_best and value is not None:
        return rf"\textbf{{{text}}}"
    return text


def _build_table() -> str:
    all_results = {name: _load_dgp_results(name) for name, _ in DGPS}
    col_spec = "ll" + "r" * len(MODELS)
    header = "DGP & Metric & " + " & ".join(label for _, label in MODELS) + r" \\"
    rows: list[str] = []

    for dgp_index, (dgp_name, dgp_label) in enumerate(DGPS):
        if dgp_index:
            rows.append(r"\midrule")
        results = all_results[dgp_name]
        for metric_index, (metric, metric_label, metric_kind) in enumerate(METRICS):
            values = [_metric_value(results, model, metric) for model, _ in MODELS]
            valid_scores = [_score_for_best(metric_kind, value) for value in values if value is not None]
            best_score = min(valid_scores) if valid_scores else None
            cells = []
            for value in values:
                is_best = (
                    value is not None
                    and best_score is not None
                    and abs(_score_for_best(metric_kind, value) - best_score) < 1e-12
                )
                cells.append(_format(metric_kind, value, is_best))
            dgp_cell = dgp_label if metric_index == 0 else ""
            rows.append(f"{dgp_cell} & {metric_label} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[!p]
\centering
\scriptsize
\caption{{Synthetic DGP summary by dataset and metric. CRPS and IS90 are lower-is-better; 90\% coverage is bolded by smallest absolute deviation from nominal. Values are means over evaluation folds.}}
\label{{tab:synthetic-dgp-summary}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{{col_spec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_build_table())
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
