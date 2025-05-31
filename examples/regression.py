# %%
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as TTS
from xgboost import XGBRegressor  # type: ignore

from bdf.tree_classes.bdf_regressor import BDFRegressor

"""
########################## Abalone dataset ######################
data = pd.read_csv("../data/abaloneage.csv")
X = data.iloc[:, 1:-1]
X.iloc[:, 0] = X.iloc[:, 0].map({"M": 0, "F": 1, "I": 2})
X.fillna(0, inplace=True)
y = data.iloc[:, -1].astype(float)
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
# %%
start_time = time()
bdf = BDFRegressor(
    dist="sn",
    prior_params={"mean": 0, "std": 1, "mean_alpha": 0, "m_alpha": 5},
    n_trees=25,
    reg_beta=0.05,
    reg_lambda=0.01,
    max_depth=10,
    subsample=0.9,
    colsample=0.9,
    min_samples_leaf=9,
)
bdf.fit(X_train.values, y_train.values, standardize_y=True)
print(f"BDF fit time: {time() - start_time:.2f} seconds")
start_time = time()
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
print(f"RF fit time: {time() - start_time:.2f} seconds")
start_time = time()
xgb = XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
xgb.fit(X_train.values, y_train.values)
print(f"XGB fit time: {time() - start_time:.2f} seconds")
# %%
print(
    f"MSE BDF: {np.mean((bdf.predict(X_test.values, method='mean', values={'total_size': 1000}) - y_test.values) ** 2)} | ",
    f"MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2)} | MSE XGB: {np.mean((xgb.predict(X_test.values) - y_test.values) ** 2)} | "
)
# %%
###################### Real estate dataset ######################
data = pd.read_csv("../data/realestate.csv")
data.describe()
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=1234)
# %%
start_time = time()
bdf = BDFRegressor(
    dist="sn",
    prior_params={"mean": 0, "std": 3, "mean_alpha": 0, "m_alpha": 5},
    n_trees=25,
    reg_beta=0.5,
    reg_lambda=0,
    max_depth=10,
    subsample=0.9,
    colsample=1.0,
    min_samples_leaf=10,
)
bdf.fit(X_train.values, y_train.values, standardize_y=True)
print(f"BDF fit time: {time() - start_time:.2f} seconds")
start_time = time()
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
print(f"RF fit time: {time() - start_time:.2f} seconds")
start_time = time()
xgb = XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
xgb.fit(X_train.values, y_train.values)
print(f"XGB fit time: {time() - start_time:.2f} seconds")

# %%
print(
    f"MSE BDF: {np.mean((bdf.predict(X_test.values, method='mean', values={'total_size': 1000}) - y_test.values) ** 2)} | ",
    f"MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2)} | MSE XGB: {np.mean((xgb.predict(X_test.values) - y_test.values) ** 2)} | "
)
# %%
######################## Wine quality dataset ######################
data = pd.read_csv("../data/winequality.csv", sep=";")
data.describe()
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=1234)
# %%
start_time = time()
bdf = BDFRegressor(
    dist="sn",
    prior_params={"mean": 0, "std": 10, "mean_alpha": 0, "m_alpha": 5},
    n_trees=25,
    reg_beta=0.05,
    reg_lambda=0,
    max_depth=10,
    subsample=0.8,
    colsample=1.0,
    min_samples_leaf=10,
)
bdf.fit(X_train.values, y_train.values, standardize_y=True)
print(f"BDF fit time: {time() - start_time:.2f} seconds")
start_time = time()
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
print(f"RF fit time: {time() - start_time:.2f} seconds")
start_time = time()
xgb = XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
xgb.fit(X_train.values, y_train.values)
print(f"XGB fit time: {time() - start_time:.2f} seconds")
# %%
print(
    f"MSE BDF: {np.mean((bdf.predict(X_test.values, method='mean', values={'total_size': 1000}) - y_test.values) ** 2)} | ",
    f"MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2)} | MSE XGB: {np.mean((xgb.predict(X_test.values) - y_test.values) ** 2)} | "
)"""
# %%
########################## Parkinsons dataset ######################
data = pd.read_csv("../data/parkinsons_updrs.csv")
data.describe()
features = (
    "age",
    "sex",
    "test_time",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
)
target = "total_UPDRS"
X = data.loc[:, features]  # type: ignore
X.columns = [f.replace(" ", "_").replace("%", "perc").replace(":", "_") for f in X.columns]
y = data[target]
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=1234)
# %%
start_time = time()
bdf = BDFRegressor(
    dist="sn",
    prior_params={"mean": 0, "std": 3, "mean_alpha": 0, "m_alpha": 10},
    n_trees=25,
    reg_beta=0.1,
    reg_lambda=0.01,
    max_depth=15,
    subsample=0.9,
    colsample=0.9,
    min_samples_leaf=10,
)
bdf.fit(X_train.values, y_train.values, standardize_y=True)
print(f"BDF fit time: {time() - start_time:.2f} seconds")
start_time = time()
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
print(f"RF fit time: {time() - start_time:.2f} seconds")
start_time = time()
xgb = XGBRegressor(max_depth=10, n_estimators=100, random_state=42)
xgb.fit(X_train.values, y_train.values)
print(f"XGB fit time: {time() - start_time:.2f} seconds")
# %%
print(
    f"MSE BDF: {np.mean((bdf.predict(X_test.values, method='median', values={'total_size': 1000}) - y_test.values) ** 2):.3f} | ",
    f"MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2):.3f} | MSE XGB: {np.mean((xgb.predict(X_test.values) - y_test.values) ** 2):.3f} | ",
)
# %%
# bdf.plot_tree(0)
# %%
