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


def plot_rel_to_best(rel_to_best_df, metric: str, save_path: Path | None = None) -> None:
    """Plot relative-to-best RMSE for regression models.

    Args:
        rel_to_best_df: DataFrame with columns 'model' and f'rel_to_best_{metric}'.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots()
    ax.bar(rel_to_best_df["model"], rel_to_best_df[f"rel_to_best_{metric}"], color="limegreen")
    ax.set_xlabel("Model")
    ax.set_ylabel(f"Relative to Best {metric.upper()}")
    # ax.set_title(f'Relative to Best {metric.upper()} of Regression Models')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
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
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # dedupe legend entries while preserving order
    seen = set()
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            dedup_h.append(h)
            dedup_l.append(l)
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
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # dedupe legend entries while preserving order
    seen = set()
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            dedup_h.append(h)
            dedup_l.append(l)
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
