import numpy as np

from bdf.tree_classes.bdf_tree import BDFTree


def _fit_single_tree(
    X: np.ndarray,
    y: np.ndarray,
    n_features_iter,
    distribution,
    reg_lambda: float,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    min_child_weight: float | int,
    random_state: int,
    reg_beta: float,
    colsample: float,
    subsample: float,
    penalty: float,
    eta: float,
    col_idcs: list | np.ndarray | None = None,
    verbose: bool = False,
) -> BDFTree:
    """Helper function to fit a single tree, used for parallel fitting."""
    rng = np.random.default_rng(random_state)
    iter_tree = BDFTree(
        distribution=distribution,
        reg_beta=reg_beta,
        reg_lambda=reg_lambda,
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
            replace=False,
        )
        X_iter = X[row_indices]
        y_iter = y[row_indices]
    else:
        X_iter = X
        y_iter = y
    col_idcs = rng.choice(X.shape[1], n_features_iter, replace=False) if colsample < 1.0 else None
    iter_tree.fit(X_iter, y_iter, col_idcs=col_idcs, verbose=verbose, eta=eta)
    return iter_tree
