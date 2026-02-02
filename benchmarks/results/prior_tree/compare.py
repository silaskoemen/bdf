# %%
import yaml

# Load linear, bernoulli and defer configs
with open("linear.yaml", "r") as f:
    linear_config = yaml.safe_load(f)
with open("bernoulli.yaml", "r") as f:
    bernoulli_config = yaml.safe_load(f)
with open("defer.yaml", "r") as f:
    defer_config = yaml.safe_load(f)


# %%
ALL_DATASETS = list(linear_config["datasets"].keys())
for d in ALL_DATASETS:
    print(50 * "-")
    print(d)
    print("----- CRPS RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["crps"]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["crps"]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["crps"]) / 9)
    print("----- PICA RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["pica"]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["pica"]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["pica"]) / 9)
    print("----- PIT KS STAT RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["pit_ks_statistic"]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["pit_ks_statistic"]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["pit_ks_statistic"]) / 9)
    print("----- WEIGHTED INTERVAL SCORE RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["weighted_interval_score"]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["weighted_interval_score"]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["weighted_interval_score"]) / 9)
    print("----- RMSE RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["mse"]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["mse"]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["mse"]) / 9)
    print("----- COVERAGE 90% RESULTS -----")
    print("Linear:", sum(linear_config["datasets"][d]["metrics"]["coverage_curve"]["empirical"][0.9]) / 9)
    print("Bernoulli:", sum(bernoulli_config["datasets"][d]["metrics"]["coverage_curve"]["empirical"][0.9]) / 9)
    print("Defer:", sum(defer_config["datasets"][d]["metrics"]["coverage_curve"]["empirical"][0.9]) / 9)

# %%
