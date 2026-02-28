from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .wrappers import (  # BARTRegressor,
    KNNKDE,
    BARTPyRegressorWrapper,
    BayesianRidgeWrapper,
    CalibratedRFWrapper,
    CatBoostUncertaintyWrapper,
    ClimatologicalRegressor,
    ConformalizedLGBMWrapper,
    ConformalizedRFWrapper,
    GaussianDeepEnsembleWrapper,
    GPClassifierWrapper,
    GPRegressorWrapper,
    LGBMQuantileRegressorWrapper,
    NGBClassifierWrapper,
    NGBRegressorWrapper,
    QuantileForestWrapper,
    TreeffuserWrapper,
)

# from treeffuser import Treeffuser


class ModelFactory:
    @staticmethod
    def get(cfg: OmegaConf) -> Any:
        match cfg.class_name:  # type: ignore[reportAttributeAccessIssue]
            case "LGBMRegressor":
                return LGBMRegressor
            case "LGBMClassifier":
                return LGBMClassifier
            case "RandomForestClassifier":
                return RandomForestClassifier
            case "RandomForestRegressor":
                return RandomForestRegressor
            case "GaussianProcessRegressor":
                return GPRegressorWrapper
            case "GaussianProcessClassifier":
                return GPClassifierWrapper
            case "NGBRegressor":
                return NGBRegressorWrapper
            case "NGBClassifier":
                return NGBClassifierWrapper
            case "LGBMQuantileRegressor":
                return LGBMQuantileRegressorWrapper
            case "CatBoostUncertaintyRegressor":
                return CatBoostUncertaintyWrapper
            case "BayesianRidgeRegressor":
                return BayesianRidgeWrapper
            case "LogisticRegression":
                return LogisticRegression
            case "LinearRegression":
                return LinearRegression
            case "KNeighborsClassifier":
                return KNeighborsClassifier
            case "KNeighborsRegressor":
                return KNeighborsRegressor
            case "DecisionTreeClassifier":
                return DecisionTreeClassifier
            case "DecisionTreeRegressor":
                return DecisionTreeRegressor
            case "CalibratedRandomForestClassifier":
                return CalibratedRFWrapper
            case "RandomForestQuantileRegressor":
                return QuantileForestWrapper
            case "BARTPyRegressor":
                return BARTPyRegressorWrapper
            case "Treeffuser":
                return TreeffuserWrapper
            case "KNNKDE":
                return KNNKDE
            case "ConformalizedLGBM":
                return ConformalizedLGBMWrapper
            case "ConformalizedRF":
                return ConformalizedRFWrapper
            case "GaussianDeepEnsemble":
                return GaussianDeepEnsembleWrapper
            case "ClimatologicalRegressor":
                return ClimatologicalRegressor
            case _:
                raise ValueError(f"Model class name '{cfg.name}' not recognized.")  # type: ignore[reportUnboundVariable]
