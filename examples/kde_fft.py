#%%
import numpy as np
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt


def silverman_bandwidth(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    if std == 0:
        std = 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    sigma = sigma if sigma > 0 else 1.0
    return 0.9 * sigma * n ** (-1 / 5)


def prepare_grid(values: np.ndarray, bandwidth: float, num_points: int = 1024):
    grid_min = values.min() - 3 * bandwidth
    grid_max = values.max() + 3 * bandwidth
    if grid_min == grid_max:
        grid_min -= 1.0
        grid_max += 1.0
    grid_x = np.linspace(grid_min, grid_max, num_points)
    dx = grid_x[1] - grid_x[0]
    edges = np.linspace(grid_min - 0.5 * dx, grid_max + 0.5 * dx, num_points + 1)
    freqs = np.fft.fftfreq(num_points, d=dx)
    return grid_x, edges, dx, freqs


def fft_kde_density(samples: np.ndarray, edges: np.ndarray, dx: float, freqs: np.ndarray, bandwidth: float):
    if len(samples) == 0:
        return np.full(len(freqs), 1e-12)
    counts, _ = np.histogram(samples, bins=edges)
    mass = counts.astype(float) / (len(samples) * dx)
    kernel_fft = np.exp(-0.5 * (2 * np.pi * bandwidth * freqs) ** 2)
    density = np.fft.ifft(np.fft.fft(mass) * kernel_fft).real
    density = np.clip(density, 1e-12, None)
    density /= density.sum() * dx
    return density


def negative_log_likelihood(samples: np.ndarray, grid_x: np.ndarray, density: np.ndarray) -> float:
    if len(samples) == 0:
        return float("nan")
    values = np.interp(samples, grid_x, density, left=1e-12, right=1e-12)
    return -np.mean(np.log(values))


X, _ = make_regression(n_samples=512, n_features=3, noise=0.1, random_state=42)
n_features = X.shape[1]


#%%
fig, axes = plt.subplots(n_features, 2, figsize=(12, 9), sharex=False, sharey=False)
if n_features == 1:
    axes = np.expand_dims(axes, axis=0)

results = []

for feature_idx in range(n_features):
    values = X[:, feature_idx]
    global_bandwidth = max(silverman_bandwidth(values), 1e-3)
    grid_x, edges, dx, freqs = prepare_grid(values, global_bandwidth)

    threshold = np.median(values)
    mask_left = values <= threshold
    mask_right = ~mask_left

    if not mask_left.any():
        mask_left[0] = True
        mask_right[0] = False
    if not mask_right.any():
        mask_right[-1] = True
        mask_left[-1] = False

    for col, (side_name, mask) in enumerate((("left", mask_left), ("right", mask_right))):
        samples = values[mask]
        bandwidth = max(silverman_bandwidth(samples), 1e-3)
        density = fft_kde_density(samples, edges, dx, freqs, bandwidth)
        nll = negative_log_likelihood(samples, grid_x, density)

        ax = axes[feature_idx, col]
        ax.plot(grid_x, density, label="FFT KDE")
        ax.scatter(samples, np.zeros_like(samples), marker="|", color="k", alpha=0.3, label="samples")
        ax.set_title(
            f"Feature {feature_idx} - {side_name} (n={len(samples)}, h={bandwidth:.4f}, NLL={nll:.3f})"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

        results.append(
            {
                "feature": feature_idx,
                "side": side_name,
                "threshold": threshold,
                "bandwidth": bandwidth,
                "n_samples": len(samples),
                "nll": nll,
            }
        )

plt.tight_layout()
plt.show()

print("feature\tside\tn_samples\tbandwidth\tNLL")
for res in results:
    print(
        f"{res['feature']}\t{res['side']}\t{res['n_samples']}\t"
        f"{res['bandwidth']:.6f}\t{res['nll']:.6f}"
    )

# %%
