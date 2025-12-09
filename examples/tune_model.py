# %%
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS

from bdf.tree_classes.bdf_regressor import BDFRegressor

data = pd.read_csv("../data/abaloneage.csv")
X = data.iloc[:, 1:-1]
X.iloc[:, 0] = X.iloc[:, 0].map({"M": 0, "F": 1, "I": 2})
X.fillna(0, inplace=True)
y = data.iloc[:, -1].astype(float)
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
# %%
import warnings

warnings.filterwarnings("ignore")
# Configure the study
study_name = "bdf_parameter_tuning"
storage_name = f"sqlite:///{study_name}.db"


def objective(trial):
    # Define the parameter search space
    n_trees = trial.suggest_int("n_trees", 5, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    colsample = trial.suggest_float("colsample", 0.5, 1.0)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    reg_beta = trial.suggest_float("reg_beta", 0.0, 2.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 2.0)
    max_depth = trial.suggest_int("max_depth", 3, 15)

    # Prior parameters
    std = trial.suggest_float("prior_std", 0.1, 10.0)
    m_alpha = trial.suggest_float("prior_m_alpha", 1.0, 500.0)

    # Create and train the model with the suggested parameters
    bdf = BDFRegressor(
        dist="sn",
        params={"mean": 0, "std": std, "mean_alpha": 0, "m_alpha": m_alpha},
        n_trees=n_trees,
        min_samples_leaf=min_samples_leaf,
        reg_beta=reg_beta,
        reg_lambda=reg_lambda,
        max_depth=max_depth,
        subsample=subsample,
        colsample=colsample,
    )

    try:
        bdf.fit(X_train.values, y_train.values, standardize_y=True)
        # Evaluate on test set
        y_pred = bdf.predict(X_test.values)
        mse = mean_squared_error(y_test.values, y_pred)
        return mse
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float("inf")  # Return a high value for failed trials


# Create the study
study = optuna.create_study(
    study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize", sampler=TPESampler(seed=42)
)

# Run optimization with 10 warmup trials followed by regular trials
n_warmup_trials = 10
n_regular_trials = 200
total_trials = n_warmup_trials + n_regular_trials

print(f"Starting optimization with {n_warmup_trials} warmup trials and {n_regular_trials} regular trials")
study.optimize(objective, n_trials=total_trials)  # , show_progress_bar=True)

# %%
# Print results
print("Best trial:")
trial = study.best_trial
print(f"  Value (MSE): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %%
# Save the best model
best_params = trial.params
best_bdf = BDFRegressor(
    dist="sn",
    params={
        "mean": best_params["prior_mean"],
        "std": best_params["prior_std"],
        "mean_alpha": best_params["prior_mean_alpha"],
        "m_alpha": best_params["prior_m_alpha"],
    },
    n_trees=best_params["n_trees"],
    min_samples_leaf=best_params["min_samples_leaf"],
    reg_beta=best_params["reg_beta"],
    reg_lambda=best_params["reg_lambda"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample=best_params["colsample"],
)

best_bdf.fit(X_train.values, y_train.values, standardize_y=True)
best_mse = mean_squared_error(y_test.values, best_bdf.predict(X_test.values))
print(f"Best model MSE on test set: {best_mse}")
