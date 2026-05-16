"""Fetch benchmark datasets from UCI, OpenML, and sklearn.

Writes immutable raw CSVs to data/raw/<name>.csv and a provenance manifest
(data/raw/provenance.json) with source ID, sha256, and row/column counts.

Usage:
    python -m scripts.fetch_datasets              # fetch all, skip cached
    python -m scripts.fetch_datasets --force      # re-fetch all
    python -m scripts.fetch_datasets --only a,b   # fetch subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

RAW_DIR = Path("data/raw")
PROVENANCE_PATH = RAW_DIR / "provenance.json"

Source = Literal["uci", "openml", "sklearn", "url"]


@dataclass
class DatasetSpec:
    name: str
    source: Source
    id: int | None = None
    sklearn_loader: str | None = None
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


# ===========================================================
# REGISTRY — confirmed UCI/OpenML IDs
# ===========================================================

REGRESSION: list[DatasetSpec] = [
    DatasetSpec(
        name="yacht_hydrodynamics",
        source="url",
        id=243,
        note="ucimlrepo API does not import id=243; direct UCI archive file",
        extras={
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            "pandas_kwargs": {"sep": r"\s+", "header": None, "engine": "python"},
            "column_names": [
                "longitudinal_position",
                "prismatic_coefficient",
                "length_displacement",
                "beam_draught_ratio",
                "length_beam_ratio",
                "froude_number",
                "residuary_resistance",
            ],
            "target_columns": ["residuary_resistance"],
        },
    ),
    DatasetSpec(name="boston_housing", source="openml", id=531, note="removed from sklearn; OpenML mirror"),
    DatasetSpec(name="forest_fires", source="uci", id=162),
    DatasetSpec(name="energy_efficiency", source="uci", id=242),
    DatasetSpec(name="concrete_strength", source="uci", id=165),
    DatasetSpec(
        name="wine_quality",
        source="url",
        id=186,
        note="ucimlrepo merges red+white without color column; fetch red-only directly",
        extras={
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "pandas_kwargs": {"sep": ";"},
            "target_columns": ["quality"],
        },
    ),
    DatasetSpec(name="abalone_age", source="uci", id=1),
    DatasetSpec(name="parkinsons_updrs", source="uci", id=189),
    DatasetSpec(name="kin8nm", source="openml", id=189),
    DatasetSpec(name="combined_cycle_power_plant", source="uci", id=294),
    DatasetSpec(
        name="naval_propulsion",
        source="url",
        id=316,
        note="ucimlrepo API does not import id=316; direct UCI archive zip",
        extras={
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
            "zip_member": "UCI CBM Dataset/data.txt",
            "pandas_kwargs": {"sep": r"\s+", "header": None, "engine": "python"},
            "column_names": [
                "lever_position",
                "ship_speed",
                "gt_shaft_torque",
                "gt_rate_of_revolutions",
                "gg_rate_of_revolutions",
                "starboard_propeller_torque",
                "port_propeller_torque",
                "hp_turbine_exit_temp",
                "gt_compressor_inlet_air_temp",
                "gt_compressor_outlet_air_temp",
                "hp_turbine_exit_pressure",
                "gt_compressor_inlet_air_pressure",
                "gt_compressor_outlet_air_pressure",
                "gt_exhaust_gas_pressure",
                "turbine_injection_control",
                "fuel_flow",
                "gt_compressor_decay",
                "gt_turbine_decay",
            ],
            "target_columns": ["gt_compressor_decay", "gt_turbine_decay"],
        },
    ),
    DatasetSpec(
        name="bike_sharing", source="uci", id=275, note="ships hourly+daily; hourly subset applied in processing"
    ),
    DatasetSpec(name="california_housing", source="sklearn", sklearn_loader="fetch_california_housing"),
    DatasetSpec(name="superconductor", source="uci", id=464),
    DatasetSpec(
        name="protein_tertiary_structure",
        source="url",
        id=265,
        note="ucimlrepo API does not import id=265; direct UCI archive CSV (CASP.csv)",
        extras={
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
            "pandas_kwargs": {},
            "target_columns": ["RMSD"],
        },
    ),
]

CLASSIFICATION: list[DatasetSpec] = [
    DatasetSpec(name="ionosphere", source="uci", id=52),
    DatasetSpec(name="breast_cancer_wisconsin", source="uci", id=17, note="Diagnostic variant"),
    DatasetSpec(name="credit_approval", source="uci", id=27),
    DatasetSpec(name="heart_disease", source="uci", id=45, note="Cleveland subset; target binarized in processing"),
    DatasetSpec(name="pima_diabetes", source="openml", id=37, note="UCI id=34 is a different dataset"),
    DatasetSpec(name="titanic", source="openml", id=40945),
    DatasetSpec(name="german_credit", source="uci", id=144, note="Statlog variant"),
    DatasetSpec(name="aids_ctg_175", source="uci", id=890),
    DatasetSpec(name="spambase", source="uci", id=94),
    DatasetSpec(name="default_credit_card", source="uci", id=350),
    DatasetSpec(
        name="bank_marketing",
        source="url",
        id=222,
        note="ucimlrepo returns old bank-full.csv with massive 'unknown'→NaN; fetch bank-additional-full directly",
        extras={
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
            "zip_member": "bank-additional/bank-additional-full.csv",
            "pandas_kwargs": {"sep": ";"},
            "target_columns": ["y"],
        },
    ),
    DatasetSpec(name="adult_income", source="uci", id=2),
]

ALL_SPECS: list[DatasetSpec] = REGRESSION + CLASSIFICATION


# ===========================================================
# FETCHERS
# ===========================================================


def _fetch_uci(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=spec.id)
    features = ds.data.features
    targets = ds.data.targets
    if targets is None or len(targets.columns) == 0:
        df = features
    else:
        df = pd.concat([features, targets], axis=1)

    meta = {
        "url": f"https://archive.ics.uci.edu/dataset/{spec.id}",
        "feature_columns": list(features.columns),
        "target_columns": list(targets.columns) if targets is not None else [],
    }
    return df, meta


def _fetch_openml(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=spec.id, as_frame=True, parser="auto")
    features = bunch.data
    target = bunch.target
    target_name = target.name if hasattr(target, "name") and target.name else "target"
    df = features.copy()
    df[target_name] = target.values

    meta = {
        "url": f"https://www.openml.org/d/{spec.id}",
        "feature_columns": list(features.columns),
        "target_columns": [target_name],
    }
    return df, meta


def _fetch_sklearn(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    import sklearn.datasets as sk

    assert spec.sklearn_loader is not None, f"sklearn.datasets has no loader named {spec.sklearn_loader}"
    loader = getattr(sk, spec.sklearn_loader)
    bunch = loader(as_frame=True)
    features = bunch.data
    target = bunch.target
    target_name = target.name if hasattr(target, "name") and target.name else "target"
    df = features.copy()
    df[target_name] = target.values

    meta = {
        "url": f"sklearn.datasets.{spec.sklearn_loader}",
        "feature_columns": list(features.columns),
        "target_columns": [target_name],
    }
    return df, meta


def _fetch_url(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Direct download for UCI datasets without ucimlrepo API support.

    Supports plain CSV/whitespace files and ZIP archives (via `zip_member`).
    """
    import io
    import urllib.request
    import zipfile

    url = spec.extras["url"]
    pandas_kwargs = spec.extras.get("pandas_kwargs", {})
    column_names = spec.extras.get("column_names")
    zip_member = spec.extras.get("zip_member")

    with urllib.request.urlopen(url) as resp:
        content = resp.read()

    if zip_member is not None:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            with zf.open(zip_member) as f:
                df = pd.read_csv(f, **pandas_kwargs)
    else:
        df = pd.read_csv(io.BytesIO(content), **pandas_kwargs)

    if column_names is not None:
        if len(column_names) != df.shape[1]:
            raise ValueError(
                f"{spec.name}: column_names has {len(column_names)} entries but file has {df.shape[1]} columns"
            )
        df.columns = column_names

    target_cols = spec.extras.get("target_columns", [df.columns[-1]])
    feature_cols = [c for c in df.columns if c not in target_cols]
    meta = {
        "url": url,
        "feature_columns": feature_cols,
        "target_columns": list(target_cols),
    }
    return df, meta


FETCHERS = {"uci": _fetch_uci, "openml": _fetch_openml, "sklearn": _fetch_sklearn, "url": _fetch_url}


# ===========================================================
# IO
# ===========================================================


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_provenance() -> dict[str, dict[str, Any]]:
    if PROVENANCE_PATH.exists():
        with open(PROVENANCE_PATH) as f:
            return json.load(f)
    return {}


def _save_provenance(prov: dict[str, dict[str, Any]]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_PATH, "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True, default=str)


def _pkg_version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "unknown"


def fetch_one(spec: DatasetSpec, *, force: bool = False) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{spec.name}.csv"
    prov = _load_provenance()

    if not force and out_path.exists() and spec.name in prov:
        existing_sha = prov[spec.name].get("sha256")
        actual_sha = _sha256(out_path)
        if existing_sha == actual_sha:
            print(f"  [cached] {spec.name}")
            return prov[spec.name]

    print(f"  [fetch]  {spec.name} ({spec.source} id={spec.id})")
    df, extra = FETCHERS[spec.source](spec)
    df.to_csv(out_path, index=False)

    entry = {
        "name": spec.name,
        "source": spec.source,
        "id": spec.id,
        "sklearn_loader": spec.sklearn_loader,
        "note": spec.note,
        "url": extra["url"],
        "feature_columns": extra["feature_columns"],
        "target_columns": extra["target_columns"],
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "sha256": _sha256(out_path),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ucimlrepo_version": _pkg_version("ucimlrepo"),
        "scikit_learn_version": _pkg_version("scikit-learn"),
    }
    prov[spec.name] = entry
    _save_provenance(prov)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated dataset names")
    args = parser.parse_args()

    selected = ALL_SPECS
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        selected = [s for s in ALL_SPECS if s.name in names]
        missing = names - {s.name for s in selected}
        if missing:
            print(f"Unknown datasets: {missing}", file=sys.stderr)
            return 1

    print(f"Fetching {len(selected)} datasets to {RAW_DIR}/")
    errors: list[tuple[str, Exception]] = []
    for spec in selected:
        try:
            fetch_one(spec, force=args.force)
        except Exception as e:
            print(f"  [ERROR]  {spec.name}: {e}", file=sys.stderr)
            errors.append((spec.name, e))

    if errors:
        print(f"\nFailed: {len(errors)}/{len(selected)}", file=sys.stderr)
        return 1
    print(f"\nDone. Provenance: {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
