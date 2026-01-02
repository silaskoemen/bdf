from typing import Any

from omegaconf import OmegaConf

from bdf.tree_classes.bdf_regressor import BDFClassifier, BDFRegressor


class ModelFactory:
    @staticmethod
    def get(cfg: OmegaConf) -> Any:
        match cfg.class_name:  # type: ignore[reportAttributeAccessIssue]
            case "BDFRegressor":
                return BDFRegressor
            case "BDFClassifier":
                return BDFClassifier
            case _:
                raise ValueError(f"Model class name '{cfg.name}' not recognized.")  # type: ignore[reportUnboundVariable]
