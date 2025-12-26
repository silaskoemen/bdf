"""
Experiment file to benchmark the effect of label noise on classification performance.
Currently, uses sklearn's `make_classification` to generate synthetic data with varying levels of label noise.

Compares performance of BDFClassifier at varying levels of label noise AND dataset size to RF
and NGBoost classifiers.

Writes out results to a markdown file in benchmarks/results/label_noise/classification_label_noise_results.md
"""

from loguru import logger
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split as TTS
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFClassifier

from .models.wrappers import NGBClassifierWrapper


def benchmark_label_noise_clas():
    logger.info("💥 Starting classification label noise benchmark")
    n_samples_list = [200, 500, 1000, 2000, 5000]
    label_noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    out = "# Classification Label Noise Benchmark Results\n\n"
    out += "| Model | N Samples | Label Noise | Accuracy ↑ | Precision ↑ | Recall ↑ | AUROC ↑ | F1 Score ↑ | Log Loss ↓ |\n"
    out += "|-------|-----------|-------------|----------|-----------|--------|-------|----------|----------|\n"
    for n_samples in n_samples_list:
        logger.info(f"Running benchmark for n_samples={n_samples}")
        out += f"|N SAMPLES: {n_samples}| | | | | | | | |\n"
        for noise_level in tqdm(
            label_noise_levels, "Iterating through label noise levels", total=len(label_noise_levels)
        ):
            X, y = make_classification(
                n_samples=n_samples, n_features=20, n_informative=15, n_redundant=5, random_state=42, flip_y=noise_level
            )
            X_train, X_val, y_train, y_val = TTS(X, y, test_size=0.2, random_state=1234)
            # BDFClassifier
            bdf_model = BDFClassifier(
                dist="BetaMVBernoulli",
                n_trees=100,
                params={"mean_p": "auto", "var_p": 0.1, "score_method": "nll", "score_correction": "bic"},
            )
            bdf_model.fit(X_train, y_train)
            y_preds_bdf = bdf_model.predict(X_val)
            y_proba_bdf = bdf_model.predict_proba(X_val)[:, 1]
            out += f"| BDFClassifier | {n_samples} | {noise_level} | "
            out += f"{accuracy_score(y_val, y_preds_bdf):.4f} | "
            out += f"{precision_score(y_val, y_preds_bdf):.4f} | "
            out += f"{recall_score(y_val, y_preds_bdf):.4f} | "
            out += f"{roc_auc_score(y_val, y_proba_bdf):.4f} | "
            out += f"{f1_score(y_val, y_preds_bdf):.4f} | "
            out += f"{log_loss(y_val, y_proba_bdf):.4f} |\n"
            # RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_preds_rf = rf_model.predict(X_val)
            y_proba_rf = rf_model.predict_proba(X_val)[:, 1]
            out += f"| RandomForest | {n_samples} | {noise_level} | "
            out += f"{accuracy_score(y_val, y_preds_rf):.4f} | "
            out += f"{precision_score(y_val, y_preds_rf):.4f} | "
            out += f"{recall_score(y_val, y_preds_rf):.4f} | "
            out += f"{roc_auc_score(y_val, y_proba_rf):.4f} | "
            out += f"{f1_score(y_val, y_preds_rf):.4f} | "
            out += f"{log_loss(y_val, y_proba_rf):.4f} |\n"
            # NGBClassifier
            ngb_model = NGBClassifierWrapper(n_estimators=100, random_state=42)
            ngb_model.fit(X_train, y_train)
            y_preds_ngb = ngb_model.predict(X_val)
            y_proba_ngb = ngb_model.predict_proba(X_val)[:, 1]
            out += f"| NGBClassifier | {n_samples} | {noise_level} | "
            out += f"{accuracy_score(y_val, y_preds_ngb):.4f} | "
            out += f"{precision_score(y_val, y_preds_ngb):.4f} | "
            out += f"{recall_score(y_val, y_preds_ngb):.4f} | "
            out += f"{roc_auc_score(y_val, y_proba_ngb):.4f} | "
            out += f"{f1_score(y_val, y_preds_ngb):.4f} | "
            out += f"{log_loss(y_val, y_proba_ngb):.4f} |\n"

            # Add horizontal line after each noise level block for readability
            out += "<br>\n"

    with open("./benchmarks/results/label_noise/classification_label_noise_results.md", "w") as f:
        f.write(out)


if __name__ == "__main__":
    benchmark_label_noise_clas()
