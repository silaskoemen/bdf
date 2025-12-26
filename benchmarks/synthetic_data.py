# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as TTS

from bdf.tree_classes.bdf_regressor import BDFModel

mpl.rcParams.update({"font.family": "serif", "font.size": 12.5, "axes.grid": True, "grid.alpha": 0.2})

# Generate synthetic data for x in [0, pi], with y|x N(sin(x), sigma(x))
# sigma(x) is |x| for x < pi/2, then 3|x|

x = np.expand_dims(np.linspace(0, 2 * np.pi, 5000), -1)


def synth_y(x: np.ndarray) -> np.ndarray:
    means = np.sin(x)
    stds = np.where(x < np.pi, np.sqrt(np.abs(x)) / 10, np.sqrt(4 * np.abs(x)) / 10)
    return np.array(norm.rvs(loc=means, scale=stds))


y = synth_y(x).reshape(-1)

# %%
x_train, x_val, y_train, y_val = TTS(x, y, test_size=0.2, random_state=1234)
fig = plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, color="black", label="train", alpha=0.75)
plt.scatter(x_val, y_val, color="crimson", label="test")
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# %%
bdf = BDFModel(
    dist="NormalMuNormal",
    params={"mu_mu": "auto", "sigma_mu": 1.0, "score_method": "nll", "score_correction": "bic"},
    max_depth=50,
    colsample=1.0,
)
bdf.fit(x_train, y_train)

# %%
sparse_x = np.expand_dims(np.linspace(0, 2 * np.pi, 100), -1)
y_preds = bdf.predict(sparse_x)
y_val_preds = bdf.predict(x_val)
y_pred_samples = bdf.predict_samples(sparse_x, n_samples=1000)
y_90_u = np.quantile(y_pred_samples, 0.95, -1)
y_90_l = np.quantile(y_pred_samples, 0.05, -1)
y_50_u = np.quantile(y_pred_samples, 0.75, -1)
y_50_l = np.quantile(y_pred_samples, 0.25, -1)

# %%
plt.figure(figsize=(9, 6))
plt.axvline(np.pi, color="black", linestyle="--", label="variance change")
plt.scatter(x_val, y_val, color="black", alpha=0.5, label="test")
plt.scatter(x_val, y_val_preds, color="crimson", label="preds")
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
plt.figure(figsize=(9, 6))
plt.axvline(np.pi, color="black", linestyle="--", label="variance change")
plt.fill_between(sparse_x.squeeze(), y_90_u, y_90_l, label="95% CI", color="dodgerblue")
plt.fill_between(sparse_x.squeeze(), y_50_u, y_50_l, label="50% CI", color="navy")
plt.scatter(x_val, y_val, color="black", label="test")
plt.plot(sparse_x, y_preds, color="crimson", label="Mean pred", linestyle="--", linewidth=3)
plt.grid(alpha=0.2)
plt.legend()
# plt.errorbar(x_val, y_preds, yerr=y_90_u-y_90_l)
plt.show()
# %%
# Plot conditional densities (histogram) for 0, 3.1 3.13, 3.14, 3.2 to show effect
plt.rcParams["text.usetex"] = True
plot_x_vals = [3.0, 3.14, 3.15, 3.3]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()

for j, pxv in enumerate(plot_x_vals):
    idx = [i for i in range(len(sparse_x)) if sparse_x[i] > pxv][0]
    idx_samples = y_pred_samples[idx]
    ax[j].hist(idx_samples, bins=35, color="dodgerblue" if pxv < np.pi else "navy")
    true_sigma = np.sqrt(np.abs(pxv)) / 10 if pxv < np.pi else np.sqrt(4 * np.abs(pxv)) / 10
    ax[j].set_title(rf"X={pxv}, $\sigma(x)$={true_sigma:.2f}, $\hat \sigma(x)$={np.std(idx_samples):.2f}")
# Set global legend below the subplots indicating dodgerblue is below variance change @ pi
# and navy blue is larger
from matplotlib.patches import Patch

handles = [
    Patch(facecolor="dodgerblue", label=r"$X < \pi$"),
    Patch(facecolor="navy", label=r"$X \ge \pi$"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
plt.subplots_adjust(bottom=0.18)
plt.tight_layout()
plt.show()


# %%
### Multimodal data ###
# Parallel linear lines with different intercepts, constant variance
# Mixture probability varies linearly in x, for x in [0, 1]
def synth_multi_y(x: np.ndarray) -> np.ndarray:
    """Generate y values for a mixture of two linear regressions with different slopes and intercepts.
    The mixture probability varies linearly with x[0], from 0.1 to 0.9
    as x goes from 0 to 1. X itself is one-dimensional.
    """
    n_samples = x.shape[0]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        mix_prob = 0.1 + 0.8 * x[i, 0]
        if np.random.rand() < mix_prob:
            y[i] = x[i, 0] + norm.rvs(scale=0.1)
        else:
            y[i] = x[i, 0] + 2 + norm.rvs(scale=0.1)
    return y


x_multi = np.expand_dims(np.linspace(0, 1, 5000), -1)
y_multi = synth_multi_y(x_multi)

# %%
fig = plt.figure(figsize=(9, 6))
plt.plot(x_multi, y_multi, ".", alpha=0.5, color="crimson")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
# %%
x_multi_train, x_multi_val, y_multi_train, y_multi_val = TTS(x_multi, y_multi, test_size=0.2, random_state=1234)
bdf_multi = BDFModel(
    dist="NormalMeanPseudoAlphaSkewNormal",  # NormalMuNormal
    params={"mu_mu": "auto", "sigma_mu": 10.0, "score_method": "nll", "score_correction": "bic"},
    max_depth=50,
)
bdf_multi.fit(x_multi_train, y_multi_train)
# %%
sparse_x_multi = np.expand_dims(np.linspace(0, 1, 100), -1)
y_multi_preds = bdf_multi.predict(sparse_x_multi)
y_multi_pred_samples = bdf_multi.predict_samples(sparse_x_multi, n_samples=1000)
y_multi_90_u = np.quantile(y_multi_pred_samples, 0.95, -1)
y_multi_90_l = np.quantile(y_multi_pred_samples, 0.05, -1)
y_multi_50_u = np.quantile(y_multi_pred_samples, 0.75, -1)
y_multi_50_l = np.quantile(y_multi_pred_samples, 0.25, -1)

# %%
plt.figure(figsize=(9, 6))
plt.fill_between(sparse_x_multi.squeeze(), y_multi_90_u, y_multi_90_l, label="95% CI", color="dodgerblue")
plt.fill_between(sparse_x_multi.squeeze(), y_multi_50_u, y_multi_50_l, label="50% CI", color="navy")
plt.scatter(x_multi_val, y_multi_val, color="black", label="test")
plt.plot(sparse_x_multi, y_multi_preds, color="crimson", label="Mean pred", linestyle="--", linewidth=3)
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
plot_multi_x_vals = [0.1, 0.4, 0.6, 0.9]
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
ax = ax.flatten()

for j, pxv in enumerate(plot_multi_x_vals):
    idx = [i for i in range(len(sparse_x_multi)) if sparse_x_multi[i] > pxv][0]
    idx_samples = y_multi_pred_samples[idx]
    ax[j].hist(idx_samples, bins=35, color="dodgerblue")
    mix_prob = 0.1 + 0.8 * pxv
    ax[j].set_title(rf"X={pxv}, Mixture Prob.={mix_prob:.2f}")
plt.show()
# %%
### Compare to KDE ###
bdf_multi = BDFModel(
    dist="KDE",
    params={"kernel": "gaussian", "score_method": "nll", "score_correction": "bic"},
    max_depth=50,
)
bdf_multi.fit(x_multi_train, y_multi_train)
# %%
sparse_x_multi = np.expand_dims(np.linspace(0, 1, 100), -1)
y_multi_preds = bdf_multi.predict(sparse_x_multi)
y_multi_pred_samples = bdf_multi.predict_samples(sparse_x_multi, n_samples=1000)
y_multi_90_u = np.quantile(y_multi_pred_samples, 0.95, -1)
y_multi_90_l = np.quantile(y_multi_pred_samples, 0.05, -1)
y_multi_50_u = np.quantile(y_multi_pred_samples, 0.75, -1)
y_multi_50_l = np.quantile(y_multi_pred_samples, 0.25, -1)

# %%
plt.figure(figsize=(9, 6))
plt.fill_between(sparse_x_multi.squeeze(), y_multi_90_u, y_multi_90_l, label="95% CI", color="dodgerblue")
plt.fill_between(sparse_x_multi.squeeze(), y_multi_50_u, y_multi_50_l, label="50% CI", color="navy")
plt.scatter(x_multi_val, y_multi_val, color="black", label="test")
plt.plot(sparse_x_multi, y_multi_preds, color="crimson", label="Mean pred", linestyle="--", linewidth=3)
plt.grid(alpha=0.2)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
# %%
plot_multi_x_vals = [0.1, 0.4, 0.6, 0.9]
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
ax = ax.flatten()

for j, pxv in enumerate(plot_multi_x_vals):
    idx = [i for i in range(len(sparse_x_multi)) if sparse_x_multi[i] > pxv][0]
    idx_samples = y_multi_pred_samples[idx]
    ax[j].hist(idx_samples, bins=35, color="dodgerblue")
    mix_prob = 0.1 + 0.8 * pxv
    ax[j].set_title(rf"X={pxv}, Mixture Prob.={mix_prob:.2f}")
plt.show()
# %%
# Compare to random forest
rf_multi = RandomForestRegressor(random_state=1234)
rf_multi.fit(x_multi_train, y_multi_train)
rf_multi_preds = rf_multi.predict(sparse_x_multi)

plt.figure(figsize=(9, 6))
plt.scatter(x_multi_val, y_multi_val, color="black", label="test")
plt.plot(sparse_x_multi, rf_multi_preds)
plt.grid(alpha=0.2)
plt.legend()
plt.show()


# %%
### Multimodal data BUT with 2nd covariate ###
# 2nd covariate determines which mode data comes from
def synth_2multi_y(x: np.ndarray) -> np.ndarray:
    """Generate y values for a mixture of two linear regressions with different slopes and intercepts.
    The mixture probability varies linearly with x[0], from 0.1 to 0.9
    as x goes from 0 to 1. X itself is one-dimensional.
    """
    n_samples = x.shape[0]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if np.random.rand() < x[i, 1]:
            y[i] = x[i, 0] + norm.rvs(scale=0.1)
        else:
            y[i] = x[i, 0] + 2 + norm.rvs(scale=0.1)
    return y


x_2multi = np.hstack([np.expand_dims(np.linspace(0, 1, 5000), -1), np.expand_dims(1 - np.linspace(0, 1, 5000), -1)])
y_2multi = synth_2multi_y(x_2multi)
# %%
fig = plt.figure(figsize=(9, 6))
plt.plot(x_2multi[:, 0], y_2multi, ".", alpha=0.5, color="crimson")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
# %%
x_2multi_train, x_2multi_val, y_2multi_train, y_2multi_val = TTS(x_2multi, y_2multi, test_size=0.2, random_state=1234)
bdf_2multi = BDFModel(
    dist="NormalMeanPseudoAlphaSkewNormal",  # NormalMuNormal
    params={"mu_mu": "auto", "sigma_mu": 10.0, "score_method": "nll", "score_correction": "bic"},
    max_depth=50,
)
bdf_2multi.fit(x_2multi_train, y_2multi_train)
# %%
y_2multi_preds = bdf_2multi.predict(x_2multi)
y_2multi_val_preds = bdf_2multi.predict(x_2multi_val)
y_2multi_pred_samples = bdf_2multi.predict_samples(x_2multi, n_samples=1000)
y_2multi_90_u = np.quantile(y_2multi_pred_samples, 0.95, -1)
y_2multi_90_l = np.quantile(y_2multi_pred_samples, 0.05, -1)
y_2multi_50_u = np.quantile(y_2multi_pred_samples, 0.75, -1)
y_2multi_50_l = np.quantile(y_2multi_pred_samples, 0.25, -1)
# %%
plt.figure(figsize=(9, 6))
plt.fill_between(x_2multi[:, 0].squeeze(), y_2multi_90_u, y_2multi_90_l, label="95% CI", color="dodgerblue")
plt.fill_between(x_2multi[:, 0].squeeze(), y_2multi_50_u, y_2multi_50_l, label="50% CI", color="navy")
plt.scatter(x_2multi_val[:, 0], y_2multi_val, color="black", label="test")
plt.plot(x_2multi[:, 0], y_2multi_preds, color="crimson", label="Mean pred", linestyle="--", linewidth=3)
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
plot_2multi_x_vals = [0.1, 0.4, 0.6, 0.9]
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
ax = ax.flatten()

for j, pxv in enumerate(plot_2multi_x_vals):
    idx = [i for i in range(x_2multi.shape[0]) if x_2multi[i, 0] > pxv][0]
    idx_samples = y_2multi_pred_samples[idx]
    ax[j].hist(idx_samples, bins=35, color="dodgerblue")
    ax[j].set_title(rf"X={pxv}")
plt.show()


# %%
### EFFECT OF SPARSITY ###
# Back to sinusoidal data, now just sparse in train/test, view effect
def generate_sparse_sinusoidal(low: float = 0, high: float = 2 * np.pi, size: int = 1000) -> tuple:
    # Sample x vals with probability from 1 linearly to 0
    x_vals = []
    while len(x_vals) < size:
        trial_x = np.random.uniform(low=low, high=high)
        if 1 - trial_x / high > np.random.uniform():
            x_vals.append(trial_x)
    y = np.sin(x_vals) + norm.rvs(scale=0.1, size=size)
    return np.expand_dims(np.array(x_vals), -1), y


x_sparse, y_sparse = generate_sparse_sinusoidal()

# %%
plt.scatter(x_sparse, y_sparse)
plt.show()
# %%
x_sparse_train, x_sparse_test, y_sparse_train, y_sparse_test = TTS(x_sparse, y_sparse, test_size=0.2, random_state=1234)
bdf_sparse = BDFModel(
    dist="NormalMuNormal",  # NormalMuNormal
    params={"mu_mu": "auto", "sigma_mu": 10.0, "score_method": "nll", "score_correction": "bic"},
)
bdf_sparse.fit(x_sparse_train, y_sparse_train)
# %%
x_sparse_plot = np.expand_dims(np.linspace(0, 2 * np.pi, 500), -1)
y_sparse_preds = bdf_sparse.predict(x_sparse_plot)
y_sparse_val_preds = bdf_sparse.predict(x_sparse_test)
y_sparse_pred_samples = bdf_sparse.predict_samples(x_sparse_plot, n_samples=1000)
y_sparse_90_u = np.quantile(y_sparse_pred_samples, 0.95, -1)
y_sparse_90_l = np.quantile(y_sparse_pred_samples, 0.05, -1)
y_sparse_50_u = np.quantile(y_sparse_pred_samples, 0.75, -1)
y_sparse_50_l = np.quantile(y_sparse_pred_samples, 0.25, -1)

# %%
plt.figure(figsize=(9, 6))
plt.fill_between(x_sparse_plot.squeeze(), y_sparse_90_u, y_sparse_90_l, label="95% CI", color="dodgerblue")
plt.fill_between(x_sparse_plot.squeeze(), y_sparse_50_u, y_sparse_50_l, label="50% CI", color="navy")
plt.scatter(x_sparse_test, y_sparse_test, color="black", label="test")
plt.plot(x_sparse_plot.squeeze(), y_sparse_preds, color="crimson", label="Mean pred", linestyle="--", linewidth=3)
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
plot_sparse_x_vals = [0.1, 0.4, 0.6, 0.9]
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
ax = ax.flatten()

for j, pxv in enumerate(plot_sparse_x_vals):
    idx = [i for i in range(x_sparse_plot.shape[0]) if x_sparse_plot[i] > pxv][0]
    idx_samples = y_sparse_pred_samples[idx]
    ax[j].hist(idx_samples, bins=35, color="dodgerblue")
    ax[j].set_title(rf"X={pxv}, $\hat\sigma(x)$={np.std(idx_samples):.3f}")
plt.show()
# %%
