"""Generate publication-ready LaTeX tables for benchmark results."""

from pathlib import Path

import numpy as np
from loguru import logger

from .statistical_tests import WilcoxonResult, WinTieLossResult


def format_metric_value(
    mean: float,
    std: float | None = None,
    is_best: bool = False,
    significance_marker: str = "",
    precision: int = 3,
) -> str:
    """Format a metric value for LaTeX.

    Args:
        mean: Mean value.
        std: Standard deviation (optional).
        is_best: If True, bold the value.
        significance_marker: Marker like *, †, ‡ for significance.
        precision: Number of decimal places.

    Returns:
        Formatted LaTeX string.
    """
    if np.isnan(mean):
        return "---"

    # Format number
    fmt = f"{{:.{precision}f}}"
    val_str = fmt.format(mean)

    if std is not None and not np.isnan(std):
        std_str = fmt.format(std)
        val_str = f"{val_str} ± {std_str}"

    if is_best:
        val_str = f"\\textbf{{{val_str}}}"

    if significance_marker:
        val_str = f"{val_str}$^{{{significance_marker}}}$"

    return val_str


def generate_main_results_table(
    models: list[str],
    datasets: list[str],
    metric_data: dict[str, dict[str, tuple[float, float]]],  # metric -> model -> (mean, std)
    best_per_metric: dict[str, str],  # metric -> best model name
    wilcoxon_results: dict[str, dict[str, WilcoxonResult]] | None = None,  # metric -> model -> result
    metrics_display: dict[str, str] | None = None,  # metric -> display name
    lower_is_better: dict[str, bool] | None = None,  # metric -> direction
    caption: str = "Main benchmark results",
    label: str = "tab:main-results",
) -> str:
    """Generate main comparison table with all metrics.

    Args:
        models: List of model names (rows).
        datasets: List of dataset names (for N count).
        metric_data: Nested dict of metric values.
        best_per_metric: Best model for each metric.
        wilcoxon_results: Pairwise test results (control vs others).
        metrics_display: Display names for metrics.
        lower_is_better: Direction for each metric.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if metrics_display is None:
        metrics_display = {}
    if lower_is_better is None:
        lower_is_better = {}

    metrics = list(metric_data.keys())
    n_metrics = len(metrics)
    n_datasets = len(datasets)

    # Build header
    header_parts = ["Model"]
    for m in metrics:
        display = metrics_display.get(m, m.upper())
        direction = r"$\downarrow$" if lower_is_better.get(m, True) else r"$\uparrow$"
        header_parts.append(f"{display} {direction}")

    header = " & ".join(header_parts) + r" \\"

    # Column specification
    col_spec = "l" + "c" * n_metrics

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    # Model rows
    for model in models:
        row_parts = [model.replace("_", r"\_")]

        for metric in metrics:
            data = metric_data.get(metric, {}).get(model, (np.nan, np.nan))
            mean, std = data

            is_best = best_per_metric.get(metric) == model

            # Significance marker (requires Holm-adjusted p-values)
            sig_marker = ""
            if wilcoxon_results and metric in wilcoxon_results:
                wresult = wilcoxon_results[metric].get(model)
                if wresult:
                    if wresult.adjusted_p_value is None:
                        logger.warning(
                            f"Wilcoxon result for {model}/{metric} has no adjusted_p_value. "
                            "Use pairwise_wilcoxon_tests() which applies Holm correction. "
                            "Significance markers will be omitted."
                        )
                    else:
                        if wresult.adjusted_p_value < 0.001:
                            sig_marker = "***"
                        elif wresult.adjusted_p_value < 0.01:
                            sig_marker = "**"
                        elif wresult.adjusted_p_value < 0.05:
                            sig_marker = "*"

            val_str = format_metric_value(mean, std, is_best, sig_marker)
            row_parts.append(val_str)

        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\smallskip\footnotesize{{Results averaged over {n_datasets} datasets. "
            r"Bold = best. Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$ (Wilcoxon vs control).}}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_per_dataset_table(
    models: list[str],
    datasets: list[str],
    metric_matrix: np.ndarray,  # shape (n_datasets, n_models)
    std_matrix: np.ndarray | None = None,
    metric_name: str = "CRPS",
    caption: str = "Per-dataset results",
    label: str = "tab:per-dataset",
    lower_is_better: bool = True,
) -> str:
    """Generate per-dataset comparison table.

    Args:
        models: List of model names (columns).
        datasets: List of dataset names (rows).
        metric_matrix: Shape (n_datasets, n_models).
        std_matrix: Standard deviations, same shape.
        metric_name: Name of metric for caption.
        caption: Table caption.
        label: LaTeX label.
        lower_is_better: If True, highlight minimum per row.

    Returns:
        LaTeX table string.
    """
    n_datasets, n_models = metric_matrix.shape

    # Column specification
    col_spec = "l" + "c" * n_models

    # Header
    header_parts = ["Dataset"] + [m.replace("_", r"\_") for m in models]
    header = " & ".join(header_parts) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for i, dataset in enumerate(datasets):
        row = metric_matrix[i, :]
        stds = std_matrix[i, :] if std_matrix is not None else [None] * n_models

        # Find best in row
        valid = ~np.isnan(row)
        if valid.any():
            if lower_is_better:
                best_idx = np.nanargmin(row)
            else:
                best_idx = np.nanargmax(row)
        else:
            best_idx = -1

        row_parts = [dataset.replace("_", r"\_")]
        for j, (val, std) in enumerate(zip(row, stds)):
            is_best = j == best_idx
            val_str = format_metric_value(val, std, is_best)
            row_parts.append(val_str)

        lines.append(" & ".join(row_parts) + r" \\")

    # Add average row
    avg_row = np.nanmean(metric_matrix, axis=0)
    avg_stds = np.nanstd(metric_matrix, axis=0) if std_matrix is None else np.nanmean(std_matrix, axis=0)

    valid = ~np.isnan(avg_row)
    if valid.any():
        if lower_is_better:
            best_idx = np.nanargmin(avg_row)
        else:
            best_idx = np.nanargmax(avg_row)
    else:
        best_idx = -1

    lines.append(r"\midrule")
    row_parts = [r"\textit{Average}"]
    for j, (val, std) in enumerate(zip(avg_row, avg_stds)):
        is_best = j == best_idx
        val_str = format_metric_value(val, std, is_best)
        row_parts.append(val_str)
    lines.append(" & ".join(row_parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",  # End resizebox
            rf"\par\smallskip\footnotesize{{{metric_name} values. Bold = best per row.}}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_win_tie_loss_table(
    control_name: str,
    challengers: list[str],
    wtl_results: dict[str, WinTieLossResult],
    caption: str = "Win/Tie/Loss comparison",
    label: str = "tab:win-tie-loss",
) -> str:
    """Generate Win/Tie/Loss table.

    Args:
        control_name: Name of control algorithm.
        challengers: List of challenger names.
        wtl_results: Dict mapping challenger -> WinTieLossResult.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Comparison & Wins & Ties & Losses & Total & Sign test $p$ \\",
        r"\midrule",
    ]

    control_display = control_name.replace("_", r"\_")

    for challenger in challengers:
        result = wtl_results.get(challenger)
        if result is None:
            continue

        challenger_display = challenger.replace("_", r"\_")

        # Significance marker
        if result.sign_test_p_value < 0.01:
            p_str = rf"\textbf{{{result.sign_test_p_value:.3f}}}"
        else:
            p_str = f"{result.sign_test_p_value:.3f}"

        row = (
            f"{control_display} vs {challenger_display} & "
            f"{result.wins} & {result.ties} & {result.losses} & "
            f"{result.n_total} & {p_str} \\\\"
        )
        lines.append(row)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip\footnotesize{Wins = control better, Losses = challenger better. "
            r"Ties = $<$1\% relative difference. Bold $p$ = significant.}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_bdf_selection_table(
    selection_map: dict[str, str],
    caption: str = "BDF distribution selection per dataset",
    label: str = "tab:bdf-selection",
) -> str:
    """Generate table showing which BDF distribution was selected per dataset.

    Args:
        selection_map: Dict mapping dataset -> selected BDF model name.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Dataset & Selected Distribution \\",
        r"\midrule",
    ]

    for dataset, model in sorted(selection_map.items()):
        dataset_display = dataset.replace("_", r"\_")
        model_display = model.replace("bdf_", "").replace("_", r"\_")
        lines.append(f"{dataset_display} & {model_display} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip\footnotesize{Selection based on CRPS from hyperparameter tuning (fold 0).}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_ranking_table(
    avg_ranks: dict[str, float],
    metric_name: str = "RMSE",
    friedman_p: float | None = None,
    caption: str = "Algorithm rankings",
    label: str = "tab:rankings",
) -> str:
    """Generate average ranking table.

    Args:
        avg_ranks: Dict mapping model name -> average rank.
        metric_name: Metric used for ranking.
        friedman_p: P-value from Friedman test.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    # Sort by rank
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Rank & Model & Avg. Rank \\",
        r"\midrule",
    ]

    for i, (model, rank) in enumerate(sorted_ranks, 1):
        model_display = model.replace("_", r"\_")
        lines.append(f"{i} & {model_display} & {rank:.2f} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    if friedman_p is not None:
        sig = "significant" if friedman_p < 0.05 else "not significant"
        lines.append(rf"\par\smallskip\footnotesize{{Friedman test: $p = {friedman_p:.4f}$ ({sig}).}}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_speedup_table(
    speedup_data: dict[str, dict[str, float]],
    control_name: str = "BDF",
    model_display_names: dict[str, str] | None = None,
    caption: str = "Geometric mean speedup of BDF relative to baselines",
    label: str = "tab:speedup",
) -> str:
    """Generate table showing BDF speedup vs each baseline.

    Args:
        speedup_data: Output of compute_speedup_table.
            Maps model -> {"fit_speedup": ..., "tune_speedup": ..., "n_datasets_fit": ..., ...}
        control_name: Name of the control model.
        model_display_names: Optional display name mapping.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if model_display_names is None:
        model_display_names = {}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Fit Speedup & $N$ & Tune Speedup & $N$ \\",
        r"\midrule",
    ]

    # Sort by fit speedup descending (BDF fastest first)
    sorted_models = sorted(
        speedup_data.items(),
        key=lambda x: x[1].get("fit_speedup", 0),
        reverse=True,
    )

    for model, data in sorted_models:
        display = model_display_names.get(model, model).replace("_", r"\_")

        fit_spd = data.get("fit_speedup")
        n_fit = data.get("n_datasets_fit", 0)
        tune_spd = data.get("tune_speedup")
        n_tune = data.get("n_datasets_tune", 0)

        fit_str = _format_speedup(fit_spd) if fit_spd is not None else "---"
        tune_str = _format_speedup(tune_spd) if tune_spd is not None else "---"

        lines.append(f"{display} & {fit_str} & {n_fit} & {tune_str} & {n_tune} \\\\")

    control_display = model_display_names.get(control_name, control_name)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\smallskip\footnotesize{{Geometric mean of per-dataset time ratios (model / {control_display}). "
            r"Values $>1$: \mbox{" + control_display + r"} is faster. $N$ = number of shared datasets.}}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def _format_speedup(value: float) -> str:
    """Format a speedup factor for display."""
    if value >= 100:
        return f"{value:.0f}$\\times$"
    if value >= 10:
        return f"{value:.1f}$\\times$"
    if value >= 0.01:
        return f"{value:.2f}$\\times$"
    return f"{value:.1e}$\\times$"


def save_latex_table(table_str: str, path: Path) -> None:
    """Save LaTeX table to file.

    Args:
        table_str: LaTeX table string.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table_str)
    print(f"Saved: {path}")
