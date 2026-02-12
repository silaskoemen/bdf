import numpy as np

from bdf.tree_classes.bdf_tree import BDFTree


def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    distribution,
    alpha: float,
    gamma: float,
    delta: float,
    tree_prior_mode: str,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    min_child_weight: float | int,
    random_state: int,
    colsample: float,
    subsample: float,
    penalty: float,
    eta: float,
    # col_idcs: list | np.ndarray | None = None,
    verbose: bool = False,
    bootstrap: bool = True,
) -> BDFTree:
    """Helper function to fit a single tree, used for parallel fitting."""
    rng = np.random.default_rng(random_state)
    iter_tree = BDFTree(
        distribution=distribution,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        tree_prior_mode=tree_prior_mode,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_child_weight=min_child_weight,
        penalty=penalty,
        random_state=random_state,
    )
    # Subsample rows and columns if specified
    if subsample < 1.0:
        n_samples = int(X.shape[0] * subsample)
        # Could allow kw bootstrap to allow replacement, do replacement below too
        row_indices = rng.choice(
            X.shape[0],
            n_samples,
            replace=bootstrap,
        )
        X_iter = X[row_indices]
        y_iter = y[row_indices]
    else:
        X_iter = X
        y_iter = y
    # col_idcs = rng.choice(X.shape[1], n_features_iter, replace=False) if colsample < 1.0 else None
    iter_tree.fit(X_iter, y_iter, rng=rng, colsample=colsample, verbose=verbose, eta=eta)
    return iter_tree


# def _logsumexp(a: np.ndarray) -> float:
#     #Stable logsumexp; returns -inf for empty arrays.
#     if a.size == 0:
#         return -np.inf
#     m = np.max(a)
#     return float(m + np.log(np.sum(np.exp(a - m))))
