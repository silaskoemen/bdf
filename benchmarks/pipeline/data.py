"""File for benchmark utility functions.

This file contains utility functions to load various regression and classification datasets
for benchmarking purposes. Includes loading of datasets in pandas format, as well as a registry
and possible variable transformations to ensure compatibility with the benchmarking pipeline.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Literal

import pandas as pd


class TargetDomain(Enum):
    REAL = "real"  # (-∞, +∞)
    POSITIVE_REAL = "positive_real"  # (0, +∞)
    NONNEGATIVE_REAL = "nonnegative_real"  # [0, +∞)
    INTEGER = "integer"  # {..., -1, 0, 1, ...}
    POSITIVE_INTEGER = "positive_integer"  # {1, 2, 3, ...}
    NONNEGATIVE_INTEGER = "nonnegative_integer"  # {0, 1, 2, ...}
    BINARY = "binary"  # {0, 1}
    MULTICLASS = "multiclass"  # {0, 1, ..., k}


@dataclass
class DatasetMetadata:
    name: str
    target_domain: TargetDomain

    def to_dict(self) -> dict:
        return {"name": self.name, "target_domain": self.target_domain.value}


# ===========================================================
# REGRESSION DATASETS
# ===========================================================


def _load_abalone_age():
    columns = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "age",
    ]
    data = pd.read_csv("data/raw/abalone_age.csv", names=columns)
    X = data.drop("age", axis=1)
    # Encode sex in [M, F, I]
    X = pd.get_dummies(X, columns=["sex"], drop_first=True)
    X = X.astype(float)
    y = data["age"]
    return X, y, "positive_integer"


def _load_parkinsons_updrs():
    data = pd.read_csv("data/raw/parkinsons_updrs.csv")
    # Should prob rename columns with '%' char
    X = data.drop(columns=["motor_UPDRS", "total_UPDRS", "subject#"], axis=1)
    X.columns = X.columns.str.replace("%", "percent").str.replace(":", "_")
    y = data["total_UPDRS"]
    return X, y, "positive_real"


def _load_boston_housing():
    data = pd.read_csv("data/raw/boston_housing.csv")
    data = data.dropna()
    X = data.drop(columns=["MEDV"])
    y = data["MEDV"]
    return X, y, "positive_real"


def _load_realestate():
    data = pd.read_csv("data/raw/realestate.csv")
    X = data.drop(columns=["Y house price of unit area", "No"])
    y = data["Y house price of unit area"]
    return X, y, "positive_real"


def _load_wine_quality():
    data = pd.read_csv("data/raw/wine_quality.csv", sep=";")
    X = data.drop("quality", axis=1)
    y = data["quality"]
    return X, y, "positive_integer"


# Registry of name - load functions for datasets
REGRESSION_DATASET_REGISTRY: dict[str, Callable] = {
    "abalone_age": _load_abalone_age,
    "boston_housing": _load_boston_housing,
    "parkinsons_updrs": _load_parkinsons_updrs,
    "realestate": _load_realestate,
    "wine_quality": _load_wine_quality,
}


def available_regression_datasets() -> Iterator[tuple[DatasetMetadata, pd.DataFrame, pd.Series]]:
    for name, load_fct in REGRESSION_DATASET_REGISTRY.items():
        X, y, target_domain = load_fct()
        yield DatasetMetadata(name=name, target_domain=TargetDomain(target_domain)), X, y


# ===========================================================
# CLASSIFICATION DATASETS
# ===========================================================


def _load_breast_cancer():
    data = pd.read_csv("data/raw/breast_cancer.csv")
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"])
    y = (data["diagnosis"] == "M").astype(int)  # Malignant = 1, Benign = 0, convert to int
    return X, y, "binary"


def _load_iris():
    data = pd.read_csv("data/raw/iris.csv", header=0)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y, "multiclass"


def _load_wine_quality_classification():
    data = pd.read_csv("data/raw/wine_quality_classification.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y, "multiclass"


def _load_boston_housing_classification():
    data = pd.read_csv("data/raw/boston_housing.csv")
    data = data.dropna()
    X = data.drop(columns=["MEDV"])
    y = data["MEDV"]
    y = (y > y.mean()).astype(float)  # Convert to binary classification problem
    return X, y, "binary"


def _load_titanic():
    data = pd.read_csv("data/raw/titanic.csv")
    drop_cols = ["PassengerId", "Ticket", "Embarked", "Name", "Cabin"]
    data = data.dropna(subset=data.columns.difference(drop_cols))
    X = data.drop(columns=["Survived", "PassengerId", "Ticket", "Embarked", "Name", "Cabin"], axis=1)
    X["Sex"] = (X["Sex"] == "male").astype(int)
    y = data["Survived"]
    return X, y, "binary"


CLASSIFICATION_DATASET_REGISTRY: dict[str, Callable] = {
    "breast_cancer": _load_breast_cancer,
    # "iris": _load_iris,
    # "wine_quality_classification": _load_wine_quality_classification,
    "boston_housing_classification": _load_boston_housing_classification,
    "titanic": _load_titanic,
}


def available_classification_datasets() -> Iterator[tuple[DatasetMetadata, pd.DataFrame, pd.Series]]:
    for name, load_fct in CLASSIFICATION_DATASET_REGISTRY.items():
        X, y, target_domain = load_fct()
        yield DatasetMetadata(name=name, target_domain=TargetDomain(target_domain)), X, y
