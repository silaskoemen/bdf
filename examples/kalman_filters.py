# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------
# Simulate strong-movement toy data (price series)
# ----------------------
np.random.seed(42)
n = 100
dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")

price = np.zeros(n)
price[0] = 200
slope = 0.5
for t in range(1, n):
    if t == 50:  # reversal point
        slope = -0.6
    slope += np.random.normal(0, 0.05)  # slope noise
    price[t] = price[t - 1] + slope + np.random.normal(0, 2.0)

df = pd.DataFrame({"price": price}, index=dates)

# ----------------------
# Convert to one-day log returns or log prices
# ----------------------
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
df["log_return"] = np.log(df["price"])
df = df.dropna()


# ----------------------
# Kalman filter functions
# ----------------------
def kalman_local_linear(y, Q_level=1e-4, Q_slope=1e-5, R=1e-4):
    """Local linear trend on returns: state = [level, slope]"""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[Q_level, 0.0], [0.0, Q_slope]])
    Rm = np.array([[R]])

    x = np.array([y[0], 0.0])  # start with first return, zero slope
    P = np.eye(2)
    levels, slopes = [], []
    for z in y:
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # update
        y_obs = np.array([[z]])
        S = H @ P_pred @ H.T + Rm
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ (y_obs - H @ x_pred)).flatten()
        P = (np.eye(2) - K @ H) @ P_pred
        levels.append(x[0])
        slopes.append(x[1])
    return np.array(levels), np.array(slopes)


def kalman_ar1(y, phi=0.9, Q_level=1e-4, R=1e-4):
    """AR(1) state-space on returns: state = [level]"""
    F = np.array([[phi]])
    H = np.array([[1.0]])
    Q = np.array([[Q_level]])
    Rm = np.array([[R]])

    x = np.array([y[0]])
    P = np.eye(1)
    levels = []
    for z in y:
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # update
        y_obs = np.array([[z]])
        S = H @ P_pred @ H.T + Rm
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ (y_obs - H @ x_pred)).flatten()
        P = (np.eye(1) - K @ H) @ P_pred
        levels.append(x[0])
    return np.array(levels)


# ----------------------
# Run both filters
# ----------------------
rets = df["log_return"].values
levels_ll, slopes_ll = kalman_local_linear(rets, Q_level=1e-5, Q_slope=1e-6, R=1e-4)
levels_ar = kalman_ar1(rets, phi=0.9, Q_level=1e-5, R=1e-4)

df["kalman_ll_level"] = levels_ll
df["kalman_ll_slope"] = slopes_ll
df["kalman_ar_level"] = levels_ar

# ----------------------
# Plot returns + Kalman smoothing
# ----------------------
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["log_return"], label="Log return", color="gray", alpha=0.6)
plt.plot(df.index, df["kalman_ll_level"], label="Kalman LL level (smoothed return)")
plt.plot(df.index, df["kalman_ar_level"], label="Kalman AR(1) level")
plt.axhline(0, linestyle="--", color="black", alpha=0.5)
plt.title("One-day log returns: Local Linear vs AR(1) Kalman smoothing")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------
# Plot estimated slope (Local Linear)
# ----------------------
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["kalman_ll_slope"], label="Kalman LL slope (change in return)")
plt.axhline(0, linestyle="--", color="black", alpha=0.5)
plt.title("Local Linear Kalman slope on log returns")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------
# Q and R sensitivity
# ----------------------
Q_vals = [1e-6, 1e-5, 1e-4]
R_vals = [1e-5, 1e-4, 1e-3]
fig, axes = plt.subplots(len(Q_vals), len(R_vals), figsize=(14, 9), sharex=True, sharey=True)
for i, q in enumerate(Q_vals):
    for j, r in enumerate(R_vals):
        lv, _ = kalman_local_linear(rets, Q_level=q, Q_slope=q / 10, R=r)
        axes[i, j].plot(df.index, df["log_return"], color="gray", alpha=0.4)
        axes[i, j].plot(df.index, lv, color="red")
        axes[i, j].set_title(f"Q_level={q}, R={r}")
plt.suptitle("Effect of Q and R on Kalman LL smoothed returns", y=0.93)
plt.tight_layout()
plt.show()

# %%
# ----------------------
# Overlay slope on actual price for signal inspection
# ----------------------
fig, ax_price = plt.subplots(figsize=(10, 5))
ax_price.plot(df.index, df["price"], label="Price", color="navy")

ax_slope = ax_price.twinx()
ax_slope.plot(df.index, df["kalman_ll_slope"], label="Kalman slope", color="darkorange", alpha=0.85)

# Turning points in slope (zero-crossings)
slope = df["kalman_ll_slope"]
turn_up = (slope > 0) & (slope.shift(1) <= 0)
turn_down = (slope < 0) & (slope.shift(1) >= 0)

ax_price.scatter(df.index[turn_up], df["price"][turn_up], marker="^", color="green", s=60, label="Slope turns up")
ax_price.scatter(df.index[turn_down], df["price"][turn_down], marker="v", color="red", s=60, label="Slope turns down")

ax_price.set_title("Price with Kalman Local Linear slope signal")
ax_price.set_ylabel("Price")
ax_slope.set_ylabel("Slope (log price change)")
ax_slope.axhline(0, linestyle="--", color="black", alpha=0.5)

lines1, labels1 = ax_price.get_legend_handles_labels()
lines2, labels2 = ax_slope.get_legend_handles_labels()
ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()
# %%
