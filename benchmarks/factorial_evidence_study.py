"""Factorial evidence study: when does integrated evidence help split selection?

Crosses four factors that the real-data benchmark confounds:

1. leaf-family specification: correctly specified vs misspecified for the DGP;
2. training sample size (small vs large leaves at fixed min_samples_leaf);
3. split-score regime: exact/profile NLE vs NLL+BIC vs pure plug-in NLL;
4. single deterministic tree vs randomized forest aggregation.

All other hyperparameters are held at the public estimator defaults, so any
difference between cells is attributable to the crossed factors rather than
per-cell tuning. Metrics are held-out CRPS (sample-based), held-out negative
log predictive density of the exact tree mixture, 90% coverage, and mean
leaves per tree.

Run:
    pixi run python -m benchmarks.factorial_evidence_study

Outputs:
    benchmarks/results/factorial_evidence/factorial_evidence.csv
    benchmarks/results/factorial_evidence/tables/factorial_evidence.tex
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.stats import t as student_t

from bdf.tree_classes.bdf_regressor import BDFRegressor

RESULTS_DIR = Path("benchmarks/results/factorial_evidence")
TABLES_DIR = RESULTS_DIR / "tables"

N_SEEDS = 10
N_TEST = 1000
N_PRED_SAMPLES = 1000
TRAIN_SIZES = (300, 3000)
FOREST_SIZES = (1, 50)

NORMAL_PARAMS = {"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 5.0}

# (leaf label, dist name, base params, score_method, score_correction)
MODEL_GRID = [
    ("Normal", "NormalMuNormal", NORMAL_PARAMS, "nle", None),
    ("Normal", "NormalMuNormal", NORMAL_PARAMS, "nll", "bic"),
    ("Normal", "NormalMuNormal", NORMAL_PARAMS, "nll", None),
    ("Student-$t$", "FrequentistStudentT", {}, "nll", "bic"),
    ("Student-$t$", "FrequentistStudentT", {}, "nll", None),
]

SCORE_LABEL = {("nle", None): "NLE", ("nll", "bic"): "NLL+BIC", ("nll", None): "NLL"}


def make_data(dgp: str, n: int, rng: np.random.Generator):
    """Mean function and noise follow the heavy-tailed synthetic DGP of the paper."""
    X = rng.uniform(-2, 2, size=(n, 2))
    mean = X[:, 0] ** 2 + X[:, 1]
    if dgp == "gaussian":
        y = mean + 0.5 * rng.standard_normal(n)
    elif dgp == "heavy_tailed":
        y = mean + 0.5 * rng.standard_t(df=3, size=n)
    else:
        raise ValueError(dgp)
    return X, y


def sample_crps(samples: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS from predictive samples via the sorted-sample estimator."""
    n_obs, n_s = samples.shape
    s_sorted = np.sort(samples, axis=1)
    term1 = np.abs(samples - y[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, n_s + 1) - n_s - 1
    term2 = (s_sorted * weights).sum(axis=1) / (n_s * n_s)
    return float((term1 - term2).mean())


def mixture_nll(model: BDFRegressor, X: np.ndarray, y: np.ndarray) -> float:
    """Exact held-out negative log density of the uniform tree mixture."""
    params = model.predict_params(X)  # (n_obs, n_trees) of dicts
    n_obs, n_trees = params.shape
    logpdf = np.empty((n_obs, n_trees))
    for i in range(n_obs):
        for m in range(n_trees):
            p = params[i, m]
            if "df" in p:
                logpdf[i, m] = student_t.logpdf(y[i], df=p["df"], loc=p["mu"], scale=p["sigma"])
            else:
                scale = np.sqrt(p["sample_std"] ** 2 + p["posterior_sigma_mu"] ** 2)
                logpdf[i, m] = norm.logpdf(y[i], loc=p["posterior_mu"], scale=scale)
    return float(-(logsumexp(logpdf, axis=1) - np.log(n_trees)).mean())


def count_leaves(model: BDFRegressor) -> float:
    """Mean number of leaves per tree (leaves = internal nodes + 1 in a binary tree)."""
    counts = []
    for tree in model.trees:
        try:
            counts.append((tree.root.count_nodes() + 1) / 2)
        except AttributeError:
            return float("nan")
    return float(np.mean(counts))


def run() -> pd.DataFrame:
    rows = []
    cells = list(itertools.product(("gaussian", "heavy_tailed"), TRAIN_SIZES, MODEL_GRID, FOREST_SIZES))
    t_start = time.time()
    for idx, (dgp, n_train, (leaf, dist, base_params, method, correction), n_trees) in enumerate(cells):
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(10_000 * idx + seed)
            X_train, y_train = make_data(dgp, n_train, rng)
            X_test, y_test = make_data(dgp, N_TEST, rng)

            params = {**base_params, "score_method": method}
            if correction is not None:
                params["score_correction"] = correction

            single_tree = n_trees == 1
            model = BDFRegressor(
                dist=dist,
                params=params,
                n_trees=n_trees,
                bootstrap=not single_tree,
                subsample=1.0 if single_tree else 0.75,
                colsample=1.0 if single_tree else 0.9,
                random_state=seed,
            )
            model.fit(X_train, y_train)

            samples = model.predict_samples(X_test, n_samples=N_PRED_SAMPLES)
            lo, hi = np.quantile(samples, [0.05, 0.95], axis=1)
            rows.append(
                {
                    "dgp": dgp,
                    "n_train": n_train,
                    "leaf": leaf,
                    "score": SCORE_LABEL[(method, correction)],
                    "n_trees": n_trees,
                    "seed": seed,
                    "crps": sample_crps(samples, y_test),
                    "nll": mixture_nll(model, X_test, y_test),
                    "coverage_90": float(((y_test >= lo) & (y_test <= hi)).mean()),
                    "mean_leaves": count_leaves(model),
                }
            )
        done = idx + 1
        rate = (time.time() - t_start) / done
        print(
            f"[{done}/{len(cells)}] {dgp} n={n_train} {leaf} {SCORE_LABEL[(method, correction)]} "
            f"M={n_trees}  ({rate:.1f}s/cell, ETA {rate * (len(cells) - done) / 60:.1f} min)"
        )
    return pd.DataFrame(rows)


def make_table(df: pd.DataFrame) -> str:
    """One row per (DGP, n, leaf, score); CRPS/NLL for tree and forest side by side."""
    agg = (
        df.groupby(["dgp", "n_train", "leaf", "score", "n_trees"])
        .agg(
            crps=("crps", "mean"),
            crps_se=("crps", "sem"),
            nll=("nll", "mean"),
            cov=("coverage_90", "mean"),
            leaves=("mean_leaves", "mean"),
        )
        .reset_index()
    )
    dgp_label = {"gaussian": "Gaussian noise", "heavy_tailed": "Student-$t_3$ noise"}
    spec = {
        ("gaussian", "Normal"): "correct",
        ("gaussian", "Student-$t$"): "over-flexible",
        ("heavy_tailed", "Normal"): "misspecified",
        ("heavy_tailed", "Student-$t$"): "correct",
    }

    lines = [
        r"\begin{table}[!p]",
        r"\centering",
        r"\small",
        r"\caption{Factorial evidence study on synthetic DGPs: held-out CRPS and negative log predictive"
        r" density (NLL) by leaf specification, training size, split-score regime, and ensemble size."
        r" Means over 10 seeds; bold marks the best score regime within each (DGP, $n$, leaf, $M$) block.}",
        r"\label{tab:factorial-evidence}",
        # The 10-column body overruns \textwidth by ~25pt at \small; box it so the
        # table scales to the margin instead of emitting an overfull \hbox.
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllcccccc}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{Single tree ($M{=}1$)} & \multicolumn{3}{c}{Forest ($M{=}50$)} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"DGP & $n$ & Leaf (spec.) & Score & CRPS & NLL & Leaves & CRPS & NLL & Leaves \\",
        r"\midrule",
    ]

    block_keys = agg[["dgp", "n_train", "leaf"]].drop_duplicates().values.tolist()
    prev_dgp_n = None
    for dgp, n_train, leaf in block_keys:
        sub = agg[(agg["dgp"] == dgp) & (agg["n_train"] == n_train) & (agg["leaf"] == leaf)]
        scores = [s for s in ("NLE", "NLL+BIC", "NLL") if s in set(sub["score"])]
        best = {}
        for m in (1, 50):
            for col in ("crps", "nll"):
                vals = {s: sub[(sub["score"] == s) & (sub["n_trees"] == m)][col].iloc[0] for s in scores}
                best[(m, col)] = min(vals, key=vals.get)
        if prev_dgp_n is not None and prev_dgp_n != (dgp, n_train):
            lines.append(r"\midrule")
        prev_dgp_n = (dgp, n_train)
        for si, s in enumerate(scores):
            cells_out = []
            for m in (1, 50):
                r = sub[(sub["score"] == s) & (sub["n_trees"] == m)].iloc[0]
                crps_s = f"{r['crps']:.3f}"
                nll_s = f"{r['nll']:.3f}"
                if best[(m, "crps")] == s:
                    crps_s = rf"\textbf{{{crps_s}}}"
                if best[(m, "nll")] == s:
                    nll_s = rf"\textbf{{{nll_s}}}"
                cells_out += [crps_s, nll_s, f"{r['leaves']:.0f}"]
            row_head = (
                f"{dgp_label[dgp]} & {n_train} & {leaf} ({spec[(dgp, leaf)]}) & {s}" if si == 0 else f" & & & {s}"
            )
            lines.append(row_head + " & " + " & ".join(cells_out) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\par\smallskip\footnotesize{All non-score hyperparameters are fixed at the public estimator"
        r" defaults; single trees disable bootstrap and row/feature subsampling. NLL is the exact"
        r" negative log density of the uniform tree mixture. `Leaves' is the mean leaf count per tree.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(RESULTS_DIR / "factorial_evidence.csv", index=False)
    (TABLES_DIR / "factorial_evidence.tex").write_text(make_table(df) + "\n")
    print(f"Saved {RESULTS_DIR / 'factorial_evidence.csv'} and {TABLES_DIR / 'factorial_evidence.tex'}")


if __name__ == "__main__":
    main()
