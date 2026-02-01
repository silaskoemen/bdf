from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.size": 12.5,
        "figure.figsize": (10, 6),
        "font.family": "serif",
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
)


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
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot relative-to-best metric for regression models.

    Args:
        rel_to_best_data: Dict mapping model name -> relative-to-best value.
        metric: Metric name for axis label.
        save_path: Optional path to save the plot (str or Path).
        figsize: Figure size.
    """
    # Sort by value (best = 1.0 should be first)
    sorted_items = sorted(rel_to_best_data.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Color: green for best (closest to 1), gradient to red for worse
    colors = plt.cm.RdYlGn_r(np.linspace(0, 0.8, len(values)))

    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)

    # Add horizontal line at 1.0 (best)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Best")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(f"Relative to Best {metric.upper()}", fontsize=12)
    ax.set_title(f"Relative to Best {metric.upper()}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

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

    metric_info = [("auroc", "AUROC"), ("ece", "ECE")]

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
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    from .statistical_tests import nemenyi_critical_difference

    # Sort algorithms by rank
    sorted_algs = sorted(avg_ranks.items(), key=lambda x: x[1])
    names = [a[0] for a in sorted_algs]
    ranks = [a[1] for a in sorted_algs]

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

        ax.text(rank, y_pos, f"{name}\n({rank:.2f})", ha="center", va=va, fontsize=10)

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
        save_path: Path to save the figure.
        figsize: Figure size.
    """
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
    ax.set_xticklabels([m.replace("_", "\n") for m in models], rotation=0, ha="center", fontsize=10)
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
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    highlight_best: bool = True,
) -> None:
    """Plot bar chart comparing models on a single metric.

    Args:
        metric_data: Dict mapping model name -> (mean, std).
        metric_name: Name of metric for axis label.
        lower_is_better: If True, highlight minimum as best.
        save_path: Path to save the figure.
        figsize: Figure size.
        highlight_best: If True, highlight the best model in a different color.
    """
    # Sort by metric value
    sorted_items = sorted(metric_data.items(), key=lambda x: x[1][0], reverse=not lower_is_better)
    names = [item[0] for item in sorted_items]
    means = [item[1][0] for item in sorted_items]
    stds = [item[1][1] for item in sorted_items]

    # Find best
    if lower_is_better:
        best_idx = np.argmin(means)
    else:
        best_idx = np.argmax(means)

    # Colors
    colors = ["#2ecc71" if i == best_idx and highlight_best else "#3498db" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=0, ha="center", fontsize=10)
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
