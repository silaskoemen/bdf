#%% 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split as TTS

from bdf.tree_classes.BDFRegressor import BDFRegressor


data = pd.read_csv("../data/abaloneage.csv")
X = data.iloc[:, 1:-1]
X.iloc[:, 0] = X.iloc[:, 0].map({'M': 0, 'F': 1, 'I': 2})
X.fillna(0, inplace=True)
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
# %%
bdf = BDFRegressor(
    data_dist='normal',
    prior_params={'mean': 0, 'std': 5},
    n_trees=15,
    reg_beta=1,
    reg_lambda=3,
    max_depth=15,
    colsample=.8
)
bdf.fit(X_train.values, y_train.values, standardize_y=True, verbose=True)
rf = RandomForestRegressor(max_depth=8, n_estimators=100, random_state=42)
rf.fit(X_train.values, y_train.values)
gb = GradientBoostingRegressor(max_depth=8, n_estimators=100, random_state=42)
gb.fit(X_train.values, y_train.values)
# %%
print(f"MSE BDF: {np.mean((bdf.predict(X_test.values) - y_test.values) ** 2)} | MSE RF: {np.mean((rf.predict(X_test.values) - y_test.values) ** 2)}, MSE GB: {np.mean((gb.predict(X_test.values) - y_test.values) ** 2)}")
# %%
