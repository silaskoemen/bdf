"""Dataset loading for benchmarks.

Datasets are pre-fetched (scripts/fetch_datasets.py) and pre-processed
(scripts/process_datasets.py) into data/processed/<name>.parquet plus
data/processed/<name>.meta.json. This module discovers them by task and
exposes uniform (metadata, X, y) tuples.

Target column in every processed parquet is literally `target`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

import pandas as pd

PROCESSED_DIR = Path("data/processed")


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
    n_samples: int
    n_features: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_domain": self.target_domain.value,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
        }


def _load_meta(name: str) -> dict:
    with open(PROCESSED_DIR / f"{name}.meta.json") as f:
        return json.load(f)


def load(name: str) -> tuple[DatasetMetadata, pd.DataFrame, pd.Series]:
    """Load a processed dataset by name.

    Returns:
        Tuple (metadata, X, y) where X has only feature columns and y is the
        `target` column cast to the appropriate dtype.
    """
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    meta = _load_meta(name)
    y = df["target"]
    X = df.drop(columns=["target"])
    return (
        DatasetMetadata(
            name=meta["name"],
            target_domain=TargetDomain(meta["domain"]),
            n_samples=X.shape[0],
            n_features=X.shape[1],
        ),
        X,
        y,
    )


def _discover(task: str) -> list[str]:
    """List processed datasets of a given task, sorted by name."""
    if not PROCESSED_DIR.exists():
        return []
    names = []
    for meta_path in sorted(PROCESSED_DIR.glob("*.meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("task") == task:
            names.append(meta["name"])
    return names


def available_regression_datasets() -> Iterator[tuple[DatasetMetadata, pd.DataFrame, pd.Series]]:
    for name in _discover("regression"):
        yield load(name)


def available_classification_datasets() -> Iterator[tuple[DatasetMetadata, pd.DataFrame, pd.Series]]:
    for name in _discover("classification"):
        yield load(name)
