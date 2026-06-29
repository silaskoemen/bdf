from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.utils.style import apply_paper_style

apply_paper_style()

# Default color palette for models (tab10-based fallback)
_DEFAULT_CMAP = plt.get_cmap("tab10")
_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]


def _get_model_style(
    models: list[str],
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Get consistent colors, display names, and markers for models.

    Args:
        models: List of model names.
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.

    Returns:
        Tuple of (colors dict, display_names dict, markers dict)
    """
    colors = {}
    display_names = {}
    markers = {}

    for i, model in enumerate(models):
        # Color: use provided or fall back to colormap
        if model_colors and model in model_colors:
            colors[model] = model_colors[model]
        else:
            colors[model] = _DEFAULT_CMAP(i % 10)

        # Display name: use provided or fall back to model name
        if model_display_names and model in model_display_names:
            display_names[model] = model_display_names[model]
        else:
            display_names[model] = model

        # Marker
        markers[model] = _MARKERS[i % len(_MARKERS)]

    return colors, display_names, markers


def plot_average_rmse_rank(avg_rank_df, save_path: str | None = None) -> None:
    """Plot average RMSE rank for regression models.

    Args:
        avg_rank_df: DataFrame with columns 'model' and 'avg_rmse_rank'.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots()
    ax.bar(avg_rank_df["model"], avg_rank_df["avg_rmse_rank"], color="skyblue")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average RMSE Rank")
    ax.set_title("Average RMSE Rank of Regression Models")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_rel_to_best(
    rel_to_best_data: dict[str, float],  # model -> relative-to-best value
    metric: str,
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    outlier_cap_ratio: float = 5.0,
) -> None:
    """Plot relative-to-best metric for regression models.

    Args:
        rel_to_best_data: Dict mapping model name -> relative-to-best value.
        metric: Metric name for axis label.
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Optional path to save the plot (str or Path).
        figsize: Figure size.
        outlier_cap_ratio: If the largest value exceeds this multiple of the
            second-largest value, truncate its displayed bar and label the
            retained value. This prevents one catastrophic result from making
            every other bar unreadable without altering the reported metric.
    """
    # Sort by value (best = 1.0 should be first)
    sorted_items = sorted(rel_to_best_data.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors_map, display_names, _ = _get_model_style(names, model_colors, model_display_names)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Colors: use model colors if provided, else gradient
    if model_colors:
        colors = [colors_map[n] for n in names]
    else:
        colors = plt.cm.RdYlGn_r(np.linspace(0, 0.8, len(values)))  # pyright: ignore[reportAttributeAccessIssue]

    display_values = list(values)
    display_cap = None
    if len(values) > 1 and values[-1] > outlier_cap_ratio * values[-2]:
        display_cap = 1.15 * values[-2]
        display_values = [min(value, display_cap) for value in values]

    bars = ax.bar(range(len(names)), display_values, color=colors, edgecolor="black", linewidth=0.5)

    # Add horizontal line at 1.0 (best)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Best")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([display_names[n] for n in names], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(f"Relative to Best {metric.upper()}", fontsize=12)
    ax.set_title(f"Relative to Best {metric.upper()}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f"{val:.2f}"
        if display_cap is not None and val > display_cap:
            label = f"↑ {val:.2e}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if display_cap is not None:
        ax.set_ylim(top=display_cap * 1.12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pica_and_pit(
    df,  # polars.DataFrame with columns ['model','dataset','pica','pit_ks_statistic']
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Polars-native plotting:
    - Uses polars for filtering/grouping/means.
    - Converts only small Series -> numpy arrays for matplotlib.
    """
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        raise ValueError("Expected a polars.DataFrame")

    # Ensure expected metric cols are present
    available = set(df.columns)
    required = {"model", "dataset"}
    if not required.issubset(available):
        raise ValueError("Input must contain 'model' and 'dataset' columns")

    metric_info = [("pica", "PICA"), ("pit_ks_statistic", "PIT KS statistic")]

    # Optional filtering (Polars)
    if models is not None:
        df = df.filter(pl.col("model").is_in(models))
    if datasets is not None:
        df = df.filter(pl.col("dataset").is_in(datasets))

    # Determine model order (preserve provided order if given)
    if models is not None:
        model_order = list(models)
    else:
        model_order = df.select("model").unique().to_series().to_list()
        model_order.sort()

    model_to_x = {m: i for i, m in enumerate(model_order)}
    n_models = len(model_order)

    # Dataset names (sorted)
    dataset_names = df.select("dataset").unique().to_series().to_list()
    dataset_names.sort()

    # Colors
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i % 10) for i, ds in enumerate(dataset_names)}

    rng = np.random.default_rng(12345)  # deterministic jitter

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=False)

    for ax, (metric_col, title) in zip(axes, metric_info):
        if metric_col not in df.columns:
            ax.set_visible(False)
            continue

        # Per-dataset points
        for ds in dataset_names:
            sub = df.filter(pl.col("dataset") == ds)
            if sub.is_empty():
                continue

            # get model list and metric values as numpy arrays
            models_list = sub.select("model").to_series().to_list()
            x = np.array([model_to_x.get(m, np.nan) for m in models_list], dtype=float)

            # metric values: polars Series -> numpy
            y_series = sub.select(metric_col).to_series()
            y = y_series.to_numpy().astype(float)

            mask = ~np.isnan(y)
            if not mask.any():
                continue

            jitter = rng.normal(loc=0.0, scale=0.08, size=mask.sum())
            x_j = x[mask] + jitter
            ax.scatter(x_j, y[mask], color=colors[ds], alpha=0.8, s=30, label=ds)

        # Compute model means (Polars groupby -> collect)
        means_df = (
            df.group_by("model").agg(pl.col(metric_col).mean().alias("mean")).filter(pl.col("model").is_in(model_order))
        )

        # Map means to model_order for plotting
        mean_x = []
        mean_y = []
        # convert means_df to dict for fast lookup
        means_dict = {r["model"]: r["mean"] for r in means_df.iter_rows(named=True)}

        for m in model_order:
            v = means_dict.get(m, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            mean_x.append(model_to_x[m])
            mean_y.append(float(v))

        if mean_x:
            ax.scatter(mean_x, mean_y, marker="D", color="red", s=70, zorder=10, label="mean (all datasets)")

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_order, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.grid(True, alpha=0.25)

    # Build shared legend (from visible axes)
    handles, labels = [], []
    for ax in axes:
        if not ax.get_visible():
            continue
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)

    # dedupe legend entries while preserving order
    seen = set()
    dedup_h, dedup_l = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            dedup_h.append(h)
            dedup_l.append(lab)
    if dedup_h:
        fig.legend(dedup_h, dedup_l, loc="lower center", ncol=min(len(dedup_l), 6), bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_auroc_and_ece(
    df,  # polars.DataFrame with columns ['model','dataset','auroc','ece']
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Polars-native plotting:
    - Uses polars for filtering/grouping/means.
    - Converts only small Series -> numpy arrays for matplotlib.
    """
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        raise ValueError("Expected a polars.DataFrame")

    # Ensure expected metric cols are present
    available = set(df.columns)
    required = {"model", "dataset"}
    if not required.issubset(available):
        raise ValueError("Input must contain 'model' and 'dataset' columns")

    metric_info = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("ece", "ECE")]

    # Optional filtering (Polars)
    if models is not None:
        df = df.filter(pl.col("model").is_in(models))
    if datasets is not None:
        df = df.filter(pl.col("dataset").is_in(datasets))

    # Determine model order (preserve provided order if given)
    if models is not None:
        model_order = list(models)
    else:
        model_order = df.select("model").unique().to_series().to_list()
        model_order.sort()

    model_to_x = {m: i for i, m in enumerate(model_order)}
    n_models = len(model_order)

    # Dataset names (sorted)
    dataset_names = df.select("dataset").unique().to_series().to_list()
    dataset_names.sort()

    # Colors
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i % 10) for i, ds in enumerate(dataset_names)}

    rng = np.random.default_rng(12345)  # deterministic jitter

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=False)

    for ax, (metric_col, title) in zip(axes, metric_info):
        if metric_col not in df.columns:
            ax.set_visible(False)
            continue

        # Per-dataset points
        for ds in dataset_names:
            sub = df.filter(pl.col("dataset") == ds)
            if sub.is_empty():
                continue

            # get model list and metric values as numpy arrays
            models_list = sub.select("model").to_series().to_list()
            x = np.array([model_to_x.get(m, np.nan) for m in models_list], dtype=float)

            # metric values: polars Series -> numpy
            y_series = sub.select(metric_col).to_series()
            y = y_series.to_numpy().astype(float)

            mask = ~np.isnan(y)
            if not mask.any():
                continue

            jitter = rng.normal(loc=0.0, scale=0.08, size=mask.sum())
            x_j = x[mask] + jitter
            ax.scatter(x_j, y[mask], color=colors[ds], alpha=0.8, s=30, label=ds)

        # Compute model means (Polars groupby -> collect)
        means_df = (
            df.group_by("model").agg(pl.col(metric_col).mean().alias("mean")).filter(pl.col("model").is_in(model_order))
        )

        # Map means to model_order for plotting
        mean_x = []
        mean_y = []
        # convert means_df to dict for fast lookup
        means_dict = {r["model"]: r["mean"] for r in means_df.iter_rows(named=True)}

        for m in model_order:
            v = means_dict.get(m, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            mean_x.append(model_to_x[m])
            mean_y.append(float(v))

        if mean_x:
            ax.scatter(mean_x, mean_y, marker="D", color="red", s=70, zorder=10, label="mean (all datasets)")

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_order, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.grid(True, alpha=0.25)

    # Build shared legend (from visible axes)
    handles, labels = [], []
    for ax in axes:
        if not ax.get_visible():
            continue
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)

    # dedupe legend entries while preserving order
    seen = set()
    dedup_h, dedup_l = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            dedup_h.append(h)
            dedup_l.append(lab)
    if dedup_h:
        fig.legend(dedup_h, dedup_l, loc="lower center", ncol=min(len(dedup_l), 6), bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_coverage_vs_interval_score_grid(
    df,  # polars.DataFrame with columns ['model','dataset','coverage_90','interval_score_90']
    datasets: list[str],
    models: list[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    2x3 grid: one subplot per dataset (up to 6). X = coverage_90, Y = interval_score_90.
    Vertical line at 0.9 indicates ideal coverage. One legend (color per model).
    Expects Polars DataFrame `df`. Uses models list to fix colors/order; if None uses sorted unique models.
    """
    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.lines import Line2D

    if not isinstance(df, pl.DataFrame):
        raise ValueError("Expected a polars.DataFrame")

    # Ensure columns exist
    required = {"model", "dataset", "coverage_90", "interval_score_90"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns {required}")

    # Optional filter to provided model list
    if models is not None:
        df = df.filter(pl.col("model").is_in(models))

    # Determine consistent model order and colors
    if models is not None:
        model_order = list(models)
    else:
        model_order = df.select("model").unique().to_series().to_list()
        model_order.sort()

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(model_order)}

    # Prepare 2x3 grid
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=300, sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= len(datasets):
            ax.set_visible(False)
            continue

        ds = datasets[ax_idx]
        ax.set_title(ds)
        # Filter dataset rows
        sub = df.filter(pl.col("dataset") == ds)
        if sub.is_empty():
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # For each model, plot (single point) if present
        for m in model_order:
            msub = sub.filter(pl.col("model") == m)
            if msub.is_empty():
                continue
            # extract arrays (very small)
            x_arr = msub.select("coverage_90").to_series().to_numpy().astype(float)
            y_arr = msub.select("interval_score_90").to_series().to_numpy().astype(float)
            # keep only finite
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if not mask.any():
                continue
            ax.scatter(
                x_arr[mask], y_arr[mask], color=colors[m], s=40, alpha=0.9, label=m, edgecolor="k", linewidth=0.2
            )

        # vertical ideal coverage line
        ax.axvline(0.9, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("coverage_90")
        ax.set_ylabel("interval_score_90")
        ax.grid(alpha=0.2)

    # Build single legend (one entry per model)
    # Create proxy artists
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[m],
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=0.2,
        )
        for m in model_order
    ]
    fig.legend(
        legend_handles, model_order, loc="lower center", ncol=min(len(model_order), 6), bbox_to_anchor=(0.5, -0.02)
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))  # leave room for legend at bottom

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_critical_difference_diagram(
    avg_ranks: dict[str, float],
    n_datasets: int,
    cd: float | None = None,
    alpha: float = 0.05,
    title: str = "Critical Difference Diagram",
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """Plot a Critical Difference (CD) diagram for algorithm comparison.

    Based on Demšar (2006). Algorithms are ordered by average rank on a number line.
    Algorithms connected by a horizontal bar are NOT significantly different.

    Args:
        avg_ranks: Dict mapping algorithm name -> average rank.
        n_datasets: Number of datasets used for ranking.
        cd: Critical difference value. If None, computed from Nemenyi test.
        alpha: Significance level for CD computation.
        title: Plot title.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    print(avg_ranks)
    from .statistical_tests import nemenyi_critical_difference

    # Sort algorithms by rank
    sorted_algs = sorted(avg_ranks.items(), key=lambda x: x[1])
    names = [a[0] for a in sorted_algs]
    ranks = [a[1] for a in sorted_algs]

    _, display_names, _ = _get_model_style(names, None, model_display_names)

    k = len(names)

    if cd is None:
        cd = nemenyi_critical_difference(k, n_datasets, alpha)

    # Find groups of algorithms that are not significantly different
    # (their rank difference is <= CD)
    groups = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(ranks[i] - ranks[j]) <= cd:
                # Check if this pair extends an existing group
                merged = False
                for g in groups:
                    if i in g or j in g:
                        g.add(i)
                        g.add(j)
                        merged = True
                        break
                if not merged:
                    groups.append({i, j})

    # Merge overlapping groups
    merged_groups = []
    for g in groups:
        found = False
        for mg in merged_groups:
            if mg & g:  # intersection
                mg.update(g)
                found = True
                break
        if not found:
            merged_groups.append(g)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Rank axis limits
    min_rank = 1
    max_rank = k
    # rank_range = max_rank - min_rank

    # Draw the axis
    ax.axhline(y=0.5, color="black", linewidth=1.5, zorder=1)

    # Draw tick marks and labels
    for i, (name, rank) in enumerate(zip(names, ranks)):
        # Tick mark
        ax.plot([rank, rank], [0.45, 0.55], color="black", linewidth=1.5, zorder=2)

        # Label (alternate above and below to avoid overlap)
        if i % 2 == 0:
            y_pos = 0.7
            va = "bottom"
        else:
            y_pos = 0.3
            va = "top"

        # Draw line from tick to label
        ax.plot(
            [rank, rank],
            [0.55 if i % 2 == 0 else 0.45, y_pos - 0.05 if i % 2 == 0 else y_pos + 0.05],
            color="gray",
            linewidth=0.8,
            linestyle="-",
            zorder=1,
        )

        ax.text(rank, y_pos, f"{display_names[name]}\n({rank:.2f})", ha="center", va=va, fontsize=10)

    # Draw CD bar at top
    cd_y = 0.9
    ax.plot([1, 1 + cd], [cd_y, cd_y], color="black", linewidth=2, zorder=3)
    ax.plot([1, 1], [cd_y - 0.02, cd_y + 0.02], color="black", linewidth=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.02, cd_y + 0.02], color="black", linewidth=2)
    ax.text(1 + cd / 2, cd_y + 0.05, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=10)

    # Draw non-significance bars (groups)
    bar_y_positions = np.linspace(0.1, 0.35, len(merged_groups) + 1)[:-1] if merged_groups else []

    for bar_y, group in zip(bar_y_positions, merged_groups):
        group_list = sorted(group)
        left_rank = ranks[group_list[0]]
        right_rank = ranks[group_list[-1]]

        # Draw thick bar
        ax.plot([left_rank, right_rank], [bar_y, bar_y], color="black", linewidth=3, solid_capstyle="butt", zorder=3)

    # Set axis properties
    ax.set_xlim(min_rank - 0.3, max_rank + 0.3)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Average Rank", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Remove y-axis
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add rank ticks on x-axis
    ax.set_xticks(range(1, k + 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_metric_scatter(
    metric_matrix: np.ndarray,  # shape (n_datasets, n_models)
    models: list[str],
    datasets: list[str],
    metric_name: str = "CRPS",
    lower_is_better: bool = True,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Plot metric values with dots per dataset and mean diamond.

    Args:
        metric_matrix: Shape (n_datasets, n_models) with metric values.
        models: List of model names (columns).
        datasets: List of dataset names (rows).
        metric_name: Name of metric for title.
        lower_is_better: If True, highlight minimum mean.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    _, display_names, _ = _get_model_style(models, None, model_display_names)
    n_datasets, n_models = metric_matrix.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Colors for datasets
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    rng = np.random.default_rng(12345)  # deterministic jitter

    # Plot each dataset as dots
    for i, ds in enumerate(datasets):
        row = metric_matrix[i, :]
        valid = ~np.isnan(row)
        if not valid.any():
            continue

        x_positions = np.arange(n_models)[valid]
        y_values = row[valid]

        # Add jitter
        jitter = rng.normal(loc=0.0, scale=0.08, size=len(x_positions))
        ax.scatter(
            x_positions + jitter,
            y_values,
            color=colors[ds],
            alpha=0.7,
            s=40,
            label=ds,
            edgecolor="white",
            linewidth=0.5,
        )

    # Compute and plot means
    means = np.nanmean(metric_matrix, axis=0)
    valid_means = ~np.isnan(means)

    ax.scatter(
        np.arange(n_models)[valid_means],
        means[valid_means],
        marker="D",
        color="red",
        s=100,
        zorder=10,
        label="Mean",
        edgecolor="black",
        linewidth=1,
    )

    # Highlight best mean
    if valid_means.any():
        if lower_is_better:
            best_idx = np.nanargmin(means)
        else:
            best_idx = np.nanargmax(means)
        ax.scatter(
            [best_idx],
            [means[best_idx]],
            marker="*",
            color="gold",
            s=300,
            zorder=11,
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([display_names[m] for m in models], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} per Model (dots = datasets, diamond = mean)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Legend (outside plot)
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_metric_comparison_bars(
    metric_data: dict[str, tuple[float, float]],  # model -> (mean, std)
    metric_name: str = "CRPS",
    lower_is_better: bool = True,
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    highlight_best: bool = True,
) -> None:
    """Plot bar chart comparing models on a single metric.

    Args:
        metric_data: Dict mapping model name -> (mean, std).
        metric_name: Name of metric for axis label.
        lower_is_better: If True, highlight minimum as best.
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
        highlight_best: If True, highlight the best model in a different color.
    """
    # Sort by metric value
    sorted_items = sorted(metric_data.items(), key=lambda x: x[1][0], reverse=not lower_is_better)
    names = [item[0] for item in sorted_items]
    means = [item[1][0] for item in sorted_items]
    stds = [item[1][1] for item in sorted_items]

    colors_map, display_names, _ = _get_model_style(names, model_colors, model_display_names)

    # Find best
    if lower_is_better:
        best_idx = np.argmin(means)
    else:
        best_idx = np.argmax(means)

    # Colors: use model colors, but highlight best with border or different shade
    if model_colors:
        colors = [colors_map[n] for n in names]
    else:
        colors = ["#2ecc71" if i == best_idx and highlight_best else "#3498db" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([display_names[n] for n in names], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} Comparison", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01 * max(means)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_calibration_curve(
    coverage_data: dict[str, dict[str, dict]],
    models: list[str] | None = None,
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> None:
    """Plot calibration curve: average empirical coverage vs nominal coverage.

    Shows how well each model's prediction intervals are calibrated.
    Perfect calibration follows the 45-degree diagonal.

    Args:
        coverage_data: Nested dict from extract_coverage_curves:
            model -> dataset -> {levels: [...], mean_curve: [...]}
        models: List of models to include (preserves order). If None, uses all.
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    if models is None:
        models = sorted(coverage_data.keys())

    colors, display_names, markers = _get_model_style(models, model_colors, model_display_names)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect calibration")

    for model in models:
        if model not in coverage_data:
            continue

        model_ds = coverage_data[model]
        if not model_ds:
            continue

        # Collect all curves across datasets
        all_levels = None
        all_curves = []

        for _, ds_data in model_ds.items():
            levels = ds_data.get("levels", [])
            mean_curve = ds_data.get("mean_curve", [])

            if not levels or not mean_curve:
                continue

            if all_levels is None:
                all_levels = levels
            all_curves.append(mean_curve)

        if all_levels is None or not all_curves:
            continue

        # Average across datasets
        avg_curve = np.nanmean(np.array(all_curves), axis=0)

        # Plot
        ax.plot(
            all_levels,
            avg_curve,
            marker=markers[model],
            markersize=6,
            linewidth=2,
            color=colors[model],
            label=display_names[model],
            alpha=0.9,
        )

    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title("Calibration Curve (Averaged Across Datasets)", fontsize=14)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pit_histograms(
    pit_data: dict[str, dict[str, dict]],
    models: list[str] | None = None,
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    n_cols: int = 3,
) -> None:
    """Plot PIT histograms in a grid, one per model, averaged across datasets.

    For well-calibrated models, PIT values should be uniformly distributed.
    Deviations indicate miscalibration:
    - U-shaped: underdispersed (prediction intervals too narrow)
    - Inverse U-shaped: overdispersed (intervals too wide)
    - Skewed: biased predictions

    Args:
        pit_data: Nested dict from extract_pit_histograms:
            model -> dataset -> {bin_counts: [...], n_bins: int, bin_proportions: [...]}
        models: List of models to include (preserves order). If None, uses all.
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
        n_cols: Number of columns in the grid.
    """
    if models is None:
        models = sorted(pit_data.keys())

    # Filter to models that have PIT data
    models = [m for m in models if m in pit_data and pit_data[m]]

    if not models:
        print("No PIT histogram data available")
        return

    colors, display_names, _ = _get_model_style(models, model_colors, model_display_names)

    n_models = len(models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=300)
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        color = colors[model]

        model_ds = pit_data[model]
        if not model_ds:
            ax.set_visible(False)
            continue

        # Collect all bin proportions across datasets
        all_proportions = []
        n_bins = None

        for ds_data in model_ds.values():
            props = ds_data.get("bin_proportions", [])
            if props:
                all_proportions.append(props)
                if n_bins is None:
                    n_bins = ds_data.get("n_bins", len(props))

        if not all_proportions or n_bins is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(display_names[model])
            continue

        # Average across datasets
        avg_proportions = np.nanmean(np.array(all_proportions), axis=0)

        # Create bin edges
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / n_bins

        # Plot histogram bars
        ax.bar(
            bin_centers,
            avg_proportions,
            width=bin_width * 0.9,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        # Plot uniform reference line
        uniform_height = 1.0 / n_bins
        ax.axhline(uniform_height, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Uniform")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(avg_proportions.max() * 1.1, uniform_height * 1.5))
        ax.set_xlabel("PIT value", fontsize=10)
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_title(display_names[model], fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add legend to first subplot
    axes_flat[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("PIT Histograms (Averaged Across Datasets)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_coverage_vs_sharpness(
    df,  # polars DataFrame with metrics
    models: list[str],
    levels: list[int] = [50, 90, 95],
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """Plot coverage vs interval score rank at multiple confidence levels.

    Shows the sharpness-calibration tradeoff using rank of interval score
    (1 = sharpest) as a scale-free metric, averaged across datasets.

    Args:
        df: Polars DataFrame from build_comparison_dataframe with metrics.
        models: List of models to include.
        levels: Coverage levels to plot (e.g., [50, 90, 95]).
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    import polars as pl
    from scipy.stats import rankdata

    colors, display_names, markers = _get_model_style(models, model_colors, model_display_names)

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize, dpi=300)
    if n_levels == 1:
        axes = [axes]

    for ax_idx, level in enumerate(levels):
        ax = axes[ax_idx]
        nominal = level / 100.0

        coverage_metric = f"coverage_{level}"
        iscore_metric = f"interval_score_{level}"

        # Get all datasets that have data for this level
        iscore_df = df.filter(pl.col("metric") == iscore_metric)
        all_datasets = set(iscore_df.select("dataset").unique().to_series().to_list())

        # For each dataset, compute ranks of interval score across models
        # model -> list of (coverage, rank) per dataset
        model_data = {m: {"coverages": [], "ranks": []} for m in models}

        for ds in all_datasets:
            # Get interval scores for all models on this dataset
            ds_scores = {}
            ds_coverages = {}
            for model in models:
                iscore_row = df.filter(
                    (pl.col("model") == model) & (pl.col("dataset") == ds) & (pl.col("metric") == iscore_metric)
                )
                cov_row = df.filter(
                    (pl.col("model") == model) & (pl.col("dataset") == ds) & (pl.col("metric") == coverage_metric)
                )
                if not iscore_row.is_empty() and not cov_row.is_empty():
                    ds_scores[model] = iscore_row.select("mean").to_series()[0]
                    ds_coverages[model] = cov_row.select("mean").to_series()[0]

            if len(ds_scores) < 2:
                continue

            # Compute ranks (1 = lowest interval score = sharpest)
            model_names = list(ds_scores.keys())
            scores = [ds_scores[m] for m in model_names]
            ranks = rankdata(scores, method="average")

            for m, rank in zip(model_names, ranks):
                model_data[m]["ranks"].append(rank)
                model_data[m]["coverages"].append(ds_coverages[m])

        # Plot each model
        for model in models:
            if not model_data[model]["ranks"]:
                continue

            avg_coverage = float(np.mean(model_data[model]["coverages"]))
            avg_rank = float(np.mean(model_data[model]["ranks"]))

            ax.scatter(
                avg_coverage,
                avg_rank,
                s=100,
                color=colors[model],
                marker=markers[model],
                label=display_names[model],
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )

        # Add vertical line at nominal coverage
        ax.axvline(nominal, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Empirical Coverage", fontsize=11)
        ax.set_ylabel("Avg Interval Score Rank", fontsize=11)
        ax.set_title(f"{level}% Prediction Interval", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set x-axis limits around nominal
        x_margin = 0.15
        ax.set_xlim(max(nominal - x_margin, 0), min(nominal + x_margin, 1.0))

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        fontsize=10,
    )

    fig.suptitle("Coverage vs Interval Score Rank (1 = sharpest, averaged across datasets)", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_coverage_vs_width_rank(
    df,  # polars DataFrame with metrics
    models: list[str],
    levels: list[int] = [50, 90],
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot coverage vs CI width rank at multiple confidence levels.

    Shows the calibration-sharpness tradeoff using rank of CI width
    (1 = narrowest) as a scale-free metric, averaged across datasets.

    Args:
        df: Polars DataFrame from build_comparison_dataframe with metrics.
        models: List of models to include.
        levels: Coverage levels to plot (e.g., [50, 90]).
        model_colors: Optional dict mapping model -> color.
        model_display_names: Optional dict mapping model -> display name.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    import polars as pl
    from scipy.stats import rankdata

    colors, display_names, markers = _get_model_style(models, model_colors, model_display_names)

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize, dpi=300)
    if n_levels == 1:
        axes = [axes]

    for ax_idx, level in enumerate(levels):
        ax = axes[ax_idx]
        nominal = level / 100.0

        coverage_metric = f"coverage_{level}"
        width_metric = f"ci_width_{level}"

        # Get all datasets that have data for this level
        width_df = df.filter(pl.col("metric") == width_metric)
        all_datasets = set(width_df.select("dataset").unique().to_series().to_list())

        # For each dataset, compute ranks of CI width across models
        model_data = {m: {"coverages": [], "ranks": []} for m in models}

        for ds in all_datasets:
            # Get CI widths for all models on this dataset
            ds_widths = {}
            ds_coverages = {}
            for model in models:
                width_row = df.filter(
                    (pl.col("model") == model) & (pl.col("dataset") == ds) & (pl.col("metric") == width_metric)
                )
                cov_row = df.filter(
                    (pl.col("model") == model) & (pl.col("dataset") == ds) & (pl.col("metric") == coverage_metric)
                )
                if not width_row.is_empty() and not cov_row.is_empty():
                    ds_widths[model] = width_row.select("mean").to_series()[0]
                    ds_coverages[model] = cov_row.select("mean").to_series()[0]

            if len(ds_widths) < 2:
                continue

            # Compute ranks (1 = lowest width = narrowest)
            model_names = list(ds_widths.keys())
            widths = [ds_widths[m] for m in model_names]
            ranks = rankdata(widths, method="average")

            for m, rank in zip(model_names, ranks):
                model_data[m]["ranks"].append(rank)
                model_data[m]["coverages"].append(ds_coverages[m])

        # Plot each model
        for model in models:
            if not model_data[model]["ranks"]:
                continue

            avg_coverage = float(np.mean(model_data[model]["coverages"]))
            avg_rank = float(np.mean(model_data[model]["ranks"]))

            ax.scatter(
                avg_coverage,
                avg_rank,
                s=100,
                color=colors[model],
                marker=markers[model],
                label=display_names[model],
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )

        # Add vertical line at nominal coverage
        ax.axvline(nominal, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Empirical Coverage", fontsize=11)
        ax.set_ylabel("Avg Width Rank", fontsize=11)
        ax.set_title(f"{level}% Prediction Interval", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set x-axis limits around nominal
        x_margin = 0.15
        ax.set_xlim(max(nominal - x_margin, 0), min(nominal + x_margin, 1.0))

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        fontsize=10,
    )

    fig.suptitle("Coverage vs Width Rank (1 = narrowest, averaged across datasets)", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
