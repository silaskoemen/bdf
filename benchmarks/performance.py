"""
Compare performance of BDF, RF and NGBoost (what else?) for a range
of feature and sample sizes from sklearn.make_regression.

Models are left at default values to represent comparable, direct
out of the box usage, fit is timed.
"""

from time import time

import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from bdf.tree_classes.bdf_regressor import BDFModel

N_REPS = 5
SEED = 1234
models = [BDFModel, RandomForestRegressor, LGBMRegressor]  # , NGBRegressorWrapper]

logger.info("🚀 Starting performance benchmark")
np.random.seed(1234)
out = "# Performance benchmark\nComparing BDF to benchmark models in terms of fit time"

# NOTE: At 50k@10 or 5k@100 locks/stops/breaks - fix in code?
for d in [10, 100, 1000]:
    out += f"\n## {d} features\n"
    logger.info(f"##### {d=} features #####")
    for n in [1000, 2000, 5000, 10000, 20000]:  # , 50000]:#, 100000]:
        logger.info(f"### {n=} samples ###")
        out += f"\n### {n} samples\n"
        for m in models:
            # logger.info(f"Fitting {m.__name__}")
            X, y = make_regression(
                n_samples=n,
                n_features=d,
                n_informative=int(d * 0.7),  # leave 70% informative features
                random_state=SEED,
            )
            _model = m(random_state=SEED, verbose=0)
            fit_times = []
            for _ in range(N_REPS):
                start_time = time()
                _model.fit(X, y)
                fit_time = time() - start_time
                fit_times.append(fit_time)
            fit_mean = np.mean(fit_times)
            fit_std = np.std(fit_times)
            logger.info(f"Fitted model {m.__name__} in {fit_mean:.3}({fit_std:.4}) seconds.")
            out += f"\n{m.__name__}: {fit_mean:.4}({fit_std:.5})"

with open("./benchmarks/results/performance/performance_timings.md") as f:
    f.write(out)
