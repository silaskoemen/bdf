"""Merge conditional diagnostics JSON outputs and generate plots/tables.

Run this after ``benchmarks.conditional_diagnostics_eval`` has been executed in
the relevant environments. The script consumes only JSON, so it does not import
model wrappers or BDF classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.utils.style import MODEL_COLORS, MODEL_DISPLAY_NAMES, apply_paper_style

INPUT_DIR = Path("benchmarks/results/conditional_diagnostics")
PLOTS_DIR = Path("benchmarks/plots/conditional_diagnostics")
TABLES_DIR = Path("benchmarks/results/conditional_diagnostics/tables")


def _label(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#888888")


def load_json_files(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"metadata": {"inputs": [str(p) for p in paths]}, "models": {}}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for model, model_data in data.get("models", {}).items():
            dst = merged["models"].setdefault(model, {"datasets": {}})
            for dataset, ds_data in model_data.get("datasets", {}).items():
                if dataset in dst["datasets"]:
                    raise ValueError(f"Duplicate model/dataset pair in inputs: {model}/{dataset}")
                dst["datasets"][dataset] = ds_data
    return merged


def flatten_results(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []

    for model, model_data in data.get("models", {}).items():
        for dataset, ds_data in model_data.get("datasets", {}).items():
            source_model = ds_data.get("source_model", model)
            summary = ds_data.get("summary", {})

            global_summary = summary.get("global", {})
            global_rows.append(
                {
                    "model": model,
                    "display_model": _label(model),
                    "source_model": source_model,
                    "dataset": dataset,
                    **global_summary,
                }
            )

            for row in summary.get("bins", []):
                bin_rows.append(
                    {
                        "model": model,
                        "display_model": _label(model),
                        "source_model": source_model,
                        "dataset": dataset,
                        **row,
                    }
                )

    return pd.DataFrame(global_rows), pd.DataFrame(bin_rows)


def aggregate_by_model(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model", sort=False):
        n = group["n"].astype(float).to_numpy()
        row = {
            "model": model,
            "display_model": _label(model),
            "n_datasets": group["dataset"].nunique(),
            "n": int(n.sum()),
        }
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(n) & (n > 0)
            row[col] = float(np.average(vals[mask], weights=n[mask])) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_bins(bin_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_cols = [
        "coverage_90",
        "coverage_error_90",
        "mean_uncertainty",
        "mean_iqr",
        "mean_width_90",
        "interval_score_90",
        "rmse",
        "mae",
        "crps",
    ]
    for (model, bin_id), group in bin_df.groupby(["model", "bin"], sort=False):
        n = group["n"].astype(float).to_numpy()
        row = {"model": model, "display_model": _label(model), "bin": int(bin_id), "n": int(n.sum())}
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(n) & (n > 0)
            row[col] = float(np.average(vals[mask], weights=n[mask])) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coverage_heatmap(bin_summary: pd.DataFrame, save_path: Path) -> None:
    models = list(dict.fromkeys(bin_summary["model"].tolist()))
    bins = sorted(bin_summary["bin"].unique())
    grid = np.full((len(models), len(bins)), np.nan)
    for i, model in enumerate(models):
        sub = bin_summary[bin_summary["model"] == model]
        for j, bin_id in enumerate(bins):
            val = sub.loc[sub["bin"] == bin_id, "coverage_error_90"]
            if len(val):
                grid[i, j] = float(val.iloc[0])

    fig, ax = plt.subplots(figsize=(0.65 * len(bins) + 2.4, 0.45 * len(models) + 1.7))
    from matplotlib.colors import TwoSlopeNorm

    im = ax.imshow(grid, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25), aspect="auto")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([str(b) for b in bins])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_label(m) for m in models])
    ax.set_xlabel("Predicted-IQR decile")
    ax.set_title("Conditional 90% Coverage Error")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Empirical coverage - 0.90")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_interval_score_by_bin(bin_summary: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model, group in bin_summary.groupby("model", sort=False):
        group = group.sort_values("bin")
        ax.plot(
            group["bin"],
            group["interval_score_90"],
            marker="o",
            color=_color(model),
            label=_label(model),
            linewidth=1.6,
        )
    ax.set_xlabel("Predicted-IQR decile")
    ax.set_ylabel("Mean 90% interval score")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def write_latex_summary(global_summary: pd.DataFrame, bin_summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in global_summary.iterrows():
        model = row["model"]
        bins = bin_summary[bin_summary["model"] == model]
        coverage_errors = pd.to_numeric(bins["coverage_error_90"], errors="coerce")
        rows.append(
            {
                "Model": _label(model),
                "Global Cov@90": row.get("coverage_90", np.nan),
                "Mean |Cond Err|": float(np.nanmean(np.abs(coverage_errors))),
                "Min Cond. Err.": float(np.nanmin(coverage_errors)),
                "IS@90": row.get("interval_score_90", np.nan),
                "CRPS": row.get("crps", np.nan),
            }
        )

    table_df = pd.DataFrame(rows)

    def fmt(x: Any) -> str:
        if pd.isna(x):
            return "---"
        if isinstance(x, str):
            return x
        return f"{float(x):.3f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Conditional calibration diagnostics by predicted-IQR decile.}",
        r"\label{tab:conditional-diagnostics}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Global Cov@90 & Mean $|\Delta|$ & Min $\Delta$ & IS@90 & CRPS \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        lines.append(" & ".join(fmt(row[col]) for col in table_df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Conditional diagnostics JSON files.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    inputs = args.inputs or sorted(args.input_dir.glob("conditional_*.json"))
    if not inputs:
        raise FileNotFoundError(f"No conditional diagnostics JSON files found in {args.input_dir}")

    merged = load_json_files(inputs)
    global_df, bin_df = flatten_results(merged)
    if global_df.empty or bin_df.empty:
        raise ValueError("No conditional diagnostics found in input JSON files")

    global_summary = aggregate_by_model(
        global_df,
        ["coverage_90", "coverage_error_90", "mean_iqr", "mean_width_90", "interval_score_90", "rmse", "mae", "crps"],
    )
    bin_summary = aggregate_bins(bin_df)

    global_df.to_csv(args.tables_dir / "conditional_global_by_dataset.csv", index=False)
    bin_df.to_csv(args.tables_dir / "conditional_bins_by_dataset.csv", index=False)
    global_summary.to_csv(args.tables_dir / "conditional_global_summary.csv", index=False)
    bin_summary.to_csv(args.tables_dir / "conditional_bin_summary.csv", index=False)

    plot_coverage_heatmap(bin_summary, args.plots_dir / "conditional_coverage_heatmap_90.pdf")
    plot_interval_score_by_bin(bin_summary, args.plots_dir / "conditional_interval_score_90.pdf")
    write_latex_summary(global_summary, bin_summary, args.tables_dir / "conditional_diagnostics_summary.tex")

    print(f"Wrote conditional diagnostics plots to {args.plots_dir}")
    print(f"Wrote conditional diagnostics tables to {args.tables_dir}")


if __name__ == "__main__":
    main()
