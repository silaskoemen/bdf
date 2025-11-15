"""File for benchmark utility functions.

This file contains utility functions to load various regression and classification datasets
for benchmarking purposes. Includes loading of datasets in pandas format, as well as a registry
and possible variable transformations to ensure compatibility with the benchmarking pipeline.
"""
from typing import Callable

import pandas as pd

# TODO: Add `target_range` or domain or type (str) to return statement
# Certain ML models (especially distributions of BDF) do not support certain
# types and can directly skip calculations (Gamma for real-valued)
# TODO: Add standardized_targets: none, both, only to evaluate as well on standardized
# Could e.g. do for R distributions acting on smaller target space, GP might need either way

# ===========================================================
# REGRESSION DATASETS
# ===========================================================


def _load_abalone_age():
    data = pd.read_csv("../../data/abalone_age.csv")
    X = data.drop("age", axis=1)
    y = data["age"]
    return X, y


def _load_parkinsons_updrs():
    data = pd.read_csv("../../data/parkinsons_updrs.csv")
    X = data.drop("updrs", axis=1)
    y = data["updrs"]
    return X, y


def _load_boston_housing():
    data = pd.read_csv("../../../data/boston_housing.csv")
    X = data.drop(columns=["MEDV"])
    y = data["MEDV"]
    return X, y


def _load_realestate():
    data = pd.read_csv("../../../data/realestate.csv")
    X = data.drop(columns=["Y house price of unit area", "No"])
    y = data["Y house price of unit area"]
    return X, y


def _load_wine_quality():
    data = pd.read_csv("../../data/wine_quality.csv")
    X = data.drop("quality", axis=1)
    y = data["quality"]
    return X, y


# Registry of name - load functions for datasets
REGRESSION_DATASET_REGISTRY: dict[str, Callable] = {
    "realestate": _load_realestate,
    "boston_housing": _load_boston_housing,
    # "abalone_age": _load_abalone_age,
    # "parkinsons_updrs": _load_parkinsons_updrs,
    # "wine_quality": _load_wine_quality,
}


def available_regression_datasets():
    for name, load_fct in REGRESSION_DATASET_REGISTRY.items():
        X, y = load_fct()
        yield name, X, y


# ===========================================================
# CLASSIFICATION DATASETS
# ===========================================================


def _load_breast_cancer():
    data = pd.read_csv("../../../data/breast_cancer.csv")
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"])
    y = data["diagnosis"] == "B"
    return X, y


def _load_iris():
    data = pd.read_csv("../../data/iris.csv", header=0)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def _load_wine_quality_classification():
    data = pd.read_csv("../../data/wine_quality_classification.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def _load_boston_housing_classification():
    data = pd.read_csv("../../data/boston_housing_classification.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def _load_titanic():
    data = pd.read_csv("../../../data/titanic.csv")
    X = data.drop(columns=["Survived", "PassengerId", "Ticket", "Embarked", "Name", "Cabin"], axis=1)
    X["Sex"] = X["Sex"] == "male"
    y = data["Survived"]
    return X, y


CLASSIFICATION_DATASET_REGISTRY: dict[str, Callable] = {
    "breast_cancer": _load_breast_cancer,
    # "iris": _load_iris,
    # "wine_quality_classification": _load_wine_quality_classification,
    # "boston_housing_classification": _load_boston_housing_classification,
    "titanic": _load_titanic,
}


def available_classification_datasets():
    for name, load_fct in CLASSIFICATION_DATASET_REGISTRY.items():
        X, y = load_fct()
        yield name, X, y
