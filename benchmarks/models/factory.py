from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from omegaconf import OmegaConf
from quantile_forest import RandomForestQuantileRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from bdf.tree_classes.bdf_regressor import BDFModel

from .wrappers import (  # BARTRegressor,
    CalibratedRFWrapper,
    CatBoostUncertaintyWrapper,
    DeepEnsembleWrapper,
    GPClassifierWrapper,
    GPRegressorWrapper,
    LGBMQuantileRegressorWrapper,
    MapieQuantileRegressorWrapper,
    NGBClassifierWrapper,
    NGBRegressorWrapper,
)

# from ngboost import NGBoostClassifier, ...


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
            case "BDFModel":
                return BDFModel
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
            case "DeepEnsembleRegressor":
                return DeepEnsembleWrapper
            case "MapieQuantileRegressor":
                return MapieQuantileRegressorWrapper
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
            # case "RandomForestQuantileRegressor":
            #     return RandomForestQuantileRegressor
            # case "BARTRegressor":
            #     return BARTRegressor
            case _:
                raise ValueError(f"Model class name '{cfg.name}' not recognized.")  # type: ignore[reportUnboundVariable]
