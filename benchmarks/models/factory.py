from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

from bdf.tree_classes.bdf_regressor import BDFModel

# from ngboost import NGBoostClassifier, ...


class ModelFactory:
    @staticmethod
    def get(cfg: OmegaConf) -> Any:
        match cfg.class_name:
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
            case _:
                raise ValueError(f"Model class name '{cfg.name}' not recognized.")
