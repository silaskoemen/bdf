#%% 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split as TTS
from time import time

from bdf.tree_classes.bdf_regressor import BDFRegressor


data = pd.read_csv("../data/abaloneage.csv")
X = data.iloc[:, 1:-1]
X.iloc[:, 0] = X.iloc[:, 0].map({'M': 0, 'F': 1, 'I': 2})
X.fillna(0, inplace=True)
y = data.iloc[:,-1].astype(float)
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
# %%
start_time = time()
bdf = BDFRegressor(
    dist='normal_normal',
    prior_params={'mean': 0, 'std': 5},
    n_trees=100,
    reg_beta=3,
    reg_lambda=1.5,
    max_depth=10,
    subsample=.9,
    colsample=.9
)
bdf.fit(X_train.values, y_train.values, standardize_y=True)
print(f"BDF fit time: {time() - start_time:.2f} seconds")
start_time = time()
rf = RandomForestRegressor(max_depth=8, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
print(f"RF fit time: {time() - start_time:.2f} seconds")
start_time = time()
gb = GradientBoostingRegressor(max_depth=8, n_estimators=100, random_state=42)
gb.fit(X_train.values, y_train.values)
print(f"GB fit time: {time() - start_time:.2f} seconds")
# %%
print(f"MSE BDF: {np.mean((bdf.predict(X_test.values) - y_test.values) ** 2)} | MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2)}, MSE GB: {np.mean((gb.predict(X_test.values) - y_test.values) ** 2)}")
# %%
