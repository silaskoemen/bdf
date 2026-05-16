"""Investigate the effect of the prior mean and variance of the beta prior
of the Bernoulli mean on accuracy, precision, recall, auroc, f1 and log loss.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split as TTS
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFClassifier

from .pipeline.data import load


def plot_prior_heatmap():
    _, X, y = load("titanic")
    X_train, X_val, y_train, y_val = TTS(X, y, test_size=0.2, random_state=1234)
    results = np.zeros((21, 10, 6))  # [means, vars, (acc/prec/rec/auroc/f1/log)]
    for i, m in tqdm(enumerate(np.arange(0.0, 1.01, 0.05)), "Iterating through prior means", total=20):
        for j, v in enumerate(np.arange(0.025, 0.251, 0.025)):
            if v < m * (1 - m):
                model = BDFClassifier(dist="BetaMVBernoulli", params={"mean_p": m, "var_p": v})
                model.fit(X_train, y_train)
                y_preds = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                results[i, j, 0] = accuracy_score(y_val, y_preds)
                results[i, j, 1] = precision_score(y_val, y_preds)
                results[i, j, 2] = recall_score(y_val, y_preds)
                results[i, j, 3] = roc_auc_score(y_val, y_proba)
                results[i, j, 4] = f1_score(y_val, y_preds)
                results[i, j, 5] = log_loss(y_val, y_proba)
    # Per metric define a heatmap of scores
    # True train mean and correct plot labels
    x_plot_labels = [f"{i:.3f}" for i in np.arange(0.025, 0.251, 0.025)]
    y_plot_labels = [f"{i:.2f}" for i in np.arange(0.0, 1.01, 0.05)]
    true_mean = y_train.mean()
    for idx, name in enumerate(["accuracy", "precision", "recall", "auroc", "f1", "log_loss"]):
        # Manually adjust the colorbar to leave invalid (outside parabola) as black, but scale scores
        # only from min to max observed
        # TODO: Set invalid to grey or separate color
        # TODO: Add arrow to indicate what's better
        plt.figure(figsize=(9, 6))
        sns.heatmap(results[:, :, idx], vmin=np.min(results[:, :, idx][results[:, :, idx] > 0]))
        plt.title(f"{name.title()}")
        plt.xticks(range(10), x_plot_labels, rotation=20)
        plt.yticks(range(21), y_plot_labels, rotation=25)
        plt.ylabel("Prior mean")
        plt.xlabel("Prior variance")
        plt.axhline(21 * true_mean, label=f"True mean: {true_mean:.2f}", color="crimson")
        plt.legend()
        plt.savefig(f"./benchmarks/results/prior_effects/prior_heatmap_{name}.png")
        plt.close()


if __name__ == "__main__":
    plot_prior_heatmap()
