"""Process raw datasets into numeric parquet + metadata.

Reads data/raw/<name>.csv (written by fetch_datasets.py) and writes
data/processed/<name>.parquet plus data/processed/<name>.meta.json.

Each processed parquet has all numeric columns and a single target column
named `target`. Schemas were inspected once; processors below are
straight-line. If a raw CSV's schema changes (e.g. ucimlrepo upstream
update), the processor will fail loudly — which is the point.

Usage:
    python -m scripts.process_datasets              # process all
    python -m scripts.process_datasets --only a,b   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv")


def _one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype=np.int8)


def _finalize(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str,
    task: str,
    domain: str,
    original_target: str,
    n_before: int,
    encoding: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)
    X = X.astype(float)
    out = X.copy()
    out["target"] = y.values
    meta = {
        "name": name,
        "task": task,
        "domain": domain,
        "target": "target",
        "original_target": original_target,
        "n_samples": len(out),
        "n_samples_before_na_drop": n_before,
        "n_features": X.shape[1],
        "encoding_applied": encoding,
    }
    return out, meta


# ===========================================================
# REGRESSION
# ===========================================================


def process_yacht_hydrodynamics():
    df = _read("yacht_hydrodynamics")
    n = len(df)
    df = df.dropna()
    y = df.pop("residuary_resistance")
    return _finalize(
        df,
        y,
        name="yacht_hydrodynamics",
        task="regression",
        domain="positive_real",
        original_target="residuary_resistance",
        n_before=n,
        encoding="all numeric",
    )


def process_boston_housing():
    df = _read("boston_housing")
    n = len(df)
    df = df.dropna()
    y = df.pop("MEDV")
    return _finalize(
        df,
        y,
        name="boston_housing",
        task="regression",
        domain="positive_real",
        original_target="MEDV",
        n_before=n,
        encoding="all numeric",
    )


def process_forest_fires():
    df = _read("forest_fires")
    n = len(df)
    df = df.dropna()
    y = pd.Series(np.log1p(df.pop("area").to_numpy()), name="area")
    df = _one_hot(df, ["month", "day"])
    return _finalize(
        df,
        y,
        name="forest_fires",
        task="regression",
        domain="real",
        original_target="area (log1p)",
        n_before=n,
        encoding="one-hot month/day; log1p(area)",
    )


def process_energy_efficiency():
    df = _read("energy_efficiency")
    n = len(df)
    df = df.dropna()
    y = df.pop("Y1")
    df = df.drop(columns=["Y2"])
    return _finalize(
        df,
        y,
        name="energy_efficiency",
        task="regression",
        domain="positive_real",
        original_target="Y1 (heating load)",
        n_before=n,
        encoding="drop Y2 (cooling load)",
    )


def process_concrete_strength():
    df = _read("concrete_strength")
    n = len(df)
    df = df.dropna()
    y = df.pop("Concrete compressive strength")
    return _finalize(
        df,
        y,
        name="concrete_strength",
        task="regression",
        domain="positive_real",
        original_target="Concrete compressive strength",
        n_before=n,
        encoding="all numeric",
    )


def process_wine_quality():
    df = _read("wine_quality")
    n = len(df)
    df = df.dropna()
    y = df.pop("quality")
    return _finalize(
        df,
        y,
        name="wine_quality",
        task="regression",
        domain="positive_integer",
        original_target="quality",
        n_before=n,
        encoding="red subset (fetched directly)",
    )


def process_abalone_age():
    df = _read("abalone_age")
    n = len(df)
    df = df.dropna()
    y = df.pop("Rings")
    df = _one_hot(df, ["Sex"])
    return _finalize(
        df,
        y,
        name="abalone_age",
        task="regression",
        domain="positive_integer",
        original_target="Rings",
        n_before=n,
        encoding="one-hot Sex (M/F/I)",
    )


def process_parkinsons_updrs():
    df = _read("parkinsons_updrs")
    n = len(df)
    df = df.dropna()
    y = df.pop("total_UPDRS")
    df = df.drop(columns=["motor_UPDRS"])
    df.columns = [c.replace("(%)", "_pct").replace("(", "_").replace(")", "").replace(":", "_") for c in df.columns]
    return _finalize(
        df,
        y,
        name="parkinsons_updrs",
        task="regression",
        domain="positive_real",
        original_target="total_UPDRS",
        n_before=n,
        encoding="drop motor_UPDRS; sanitize column names",
    )


def process_kin8nm():
    df = _read("kin8nm")
    n = len(df)
    df = df.dropna()
    y = df.pop("y")
    return _finalize(
        df, y, name="kin8nm", task="regression", domain="real", original_target="y", n_before=n, encoding="all numeric"
    )


def process_combined_cycle_power_plant():
    df = _read("combined_cycle_power_plant")
    n = len(df)
    df = df.dropna()
    y = df.pop("PE")
    return _finalize(
        df,
        y,
        name="combined_cycle_power_plant",
        task="regression",
        domain="positive_real",
        original_target="PE",
        n_before=n,
        encoding="all numeric",
    )


def process_naval_propulsion():
    df = _read("naval_propulsion")
    n = len(df)
    df = df.dropna()
    y = df.pop("gt_compressor_decay")
    df = df.drop(columns=["gt_turbine_decay"])
    return _finalize(
        df,
        y,
        name="naval_propulsion",
        task="regression",
        domain="positive_real",
        original_target="gt_compressor_decay",
        n_before=n,
        encoding="drop gt_turbine_decay",
    )


def process_bike_sharing():
    df = _read("bike_sharing")
    n = len(df)
    df = df.dropna()
    y = df.pop("cnt")
    df = df.drop(columns=["dteday"])
    return _finalize(
        df,
        y,
        name="bike_sharing",
        task="regression",
        domain="positive_integer",
        original_target="cnt",
        n_before=n,
        encoding="drop dteday (date)",
    )


def process_california_housing():
    df = _read("california_housing")
    n = len(df)
    df = df.dropna()
    y = df.pop("MedHouseVal")
    return _finalize(
        df,
        y,
        name="california_housing",
        task="regression",
        domain="positive_real",
        original_target="MedHouseVal",
        n_before=n,
        encoding="all numeric",
    )


def process_superconductor():
    df = _read("superconductor")
    n = len(df)
    df = df.dropna()
    y = df.pop("critical_temp")
    return _finalize(
        df,
        y,
        name="superconductor",
        task="regression",
        domain="positive_real",
        original_target="critical_temp",
        n_before=n,
        encoding="all numeric",
    )


def process_protein_tertiary_structure():
    df = _read("protein_tertiary_structure")
    n = len(df)
    df = df.dropna()
    y = df.pop("RMSD")
    return _finalize(
        df,
        y,
        name="protein_tertiary_structure",
        task="regression",
        domain="nonnegative_real",
        original_target="RMSD",
        n_before=n,
        encoding="all numeric",
    )


# ===========================================================
# CLASSIFICATION
# ===========================================================


def process_ionosphere():
    df = _read("ionosphere")
    n = len(df)
    df = df.dropna()
    y = (df.pop("Class") == "g").astype(int)
    return _finalize(
        df,
        y,
        name="ionosphere",
        task="classification",
        domain="binary",
        original_target="Class (g/b)",
        n_before=n,
        encoding="g=1, b=0",
    )


def process_breast_cancer_wisconsin():
    df = _read("breast_cancer_wisconsin")
    n = len(df)
    df = df.dropna()
    y = (df.pop("Diagnosis") == "M").astype(int)
    return _finalize(
        df,
        y,
        name="breast_cancer_wisconsin",
        task="classification",
        domain="binary",
        original_target="Diagnosis (M/B)",
        n_before=n,
        encoding="M=1, B=0",
    )


def process_credit_approval():
    df = _read("credit_approval").replace("?", np.nan).dropna()
    n = len(_read("credit_approval"))
    bin_map = {"a": 1, "b": 0, "t": 1, "f": 0, "+": 1, "-": 0}
    for c in ["A1", "A9", "A10", "A12", "A16"]:
        df[c] = df[c].map(bin_map).astype(int)
    y = df.pop("A16")
    df = _one_hot(df, ["A4", "A5", "A6", "A7", "A13"])
    return _finalize(
        df,
        y,
        name="credit_approval",
        task="classification",
        domain="binary",
        original_target="A16 (+/-)",
        n_before=n,
        encoding="? → NA drop; binary a/b,t/f,+/-; one-hot A4/A5/A6/A7/A13",
    )


def process_heart_disease():
    df = _read("heart_disease").replace("?", np.nan).dropna()
    n = len(_read("heart_disease"))
    y = (df.pop("num").astype(int) > 0).astype(int)
    df = _one_hot(df, ["cp", "restecg", "slope", "thal"])
    return _finalize(
        df,
        y,
        name="heart_disease",
        task="classification",
        domain="binary",
        original_target="num (binarized >0)",
        n_before=n,
        encoding="binarize num>0; one-hot cp/restecg/slope/thal",
    )


def process_pima_diabetes():
    df = _read("pima_diabetes")
    n = len(df)
    df = df.dropna()
    y = (df.pop("class") == "tested_positive").astype(int)
    return _finalize(
        df,
        y,
        name="pima_diabetes",
        task="classification",
        domain="binary",
        original_target="class (tested_positive/negative)",
        n_before=n,
        encoding="tested_positive=1",
    )


def process_titanic():
    df = _read("titanic")
    n = len(df)
    df = df.drop(columns=["name", "ticket", "cabin", "boat", "body", "home.dest"])
    df = df.dropna()
    y = df.pop("survived").astype(int)
    df["sex"] = (df["sex"] == "male").astype(int)
    df = _one_hot(df, ["embarked"])
    return _finalize(
        df,
        y,
        name="titanic",
        task="classification",
        domain="binary",
        original_target="survived",
        n_before=n,
        encoding="drop name/ticket/cabin/boat/body/home.dest; sex→0/1; one-hot embarked",
    )


def process_german_credit():
    df = _read("german_credit")
    n = len(df)
    df = df.dropna()
    # Statlog: A1,A3,A4,A6,A7,A9,A10,A12,A14,A15,A17,A19,A20 categorical
    cat = [f"Attribute{i}" for i in [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]]
    y = (df.pop("class") == 1).astype(int)  # 1=good→1, 2=bad→0
    df = _one_hot(df, cat)
    return _finalize(
        df,
        y,
        name="german_credit",
        task="classification",
        domain="binary",
        original_target="class (1=good, 2=bad)",
        n_before=n,
        encoding="good=1, bad=0; one-hot 13 categorical attributes",
    )


def process_aids_ctg_175():
    df = _read("aids_ctg_175")
    n = len(df)
    df = df.dropna()
    y = df.pop("cid").astype(int)
    return _finalize(
        df,
        y,
        name="aids_ctg_175",
        task="classification",
        domain="binary",
        original_target="cid",
        n_before=n,
        encoding="all numeric (categoricals pre-coded)",
    )


def process_spambase():
    df = _read("spambase")
    n = len(df)
    df = df.dropna()
    y = df.pop("Class").astype(int)
    df.columns = [
        c.replace("char_freq_;", "char_freq_semi")
        .replace("char_freq_(", "char_freq_paren")
        .replace("char_freq_[", "char_freq_bracket")
        .replace("char_freq_!", "char_freq_bang")
        .replace("char_freq_$", "char_freq_dollar")
        .replace("char_freq_#", "char_freq_hash")
        for c in df.columns
    ]
    return _finalize(
        df,
        y,
        name="spambase",
        task="classification",
        domain="binary",
        original_target="Class",
        n_before=n,
        encoding="all numeric; sanitize char_freq column names",
    )


def process_default_credit_card():
    df = _read("default_credit_card")
    n = len(df)
    df = df.dropna()
    y = df.pop("Y").astype(int)
    return _finalize(
        df,
        y,
        name="default_credit_card",
        task="classification",
        domain="binary",
        original_target="Y (default next month)",
        n_before=n,
        encoding="all numeric",
    )


def process_bank_marketing():
    df = _read("bank_marketing")
    n = len(df)
    df = df.dropna()
    y = (df.pop("y") == "yes").astype(int)
    cat = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
    df = _one_hot(df, cat)
    return _finalize(
        df,
        y,
        name="bank_marketing",
        task="classification",
        domain="binary",
        original_target="y (yes/no)",
        n_before=n,
        encoding="yes=1; one-hot 10 categorical columns",
    )


def process_adult_income():
    df = _read("adult_income").replace("?", np.nan).dropna()
    n = len(_read("adult_income"))
    y = (df.pop("income").str.strip().str.rstrip(".") == ">50K").astype(int)
    df = df.drop(columns=["fnlwgt"])
    cat = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    df = _one_hot(df, cat)
    df.columns = [c.replace("-", "_") for c in df.columns]
    return _finalize(
        df,
        y,
        name="adult_income",
        task="classification",
        domain="binary",
        original_target="income (>50K)",
        n_before=n,
        encoding="? → NA drop; >50K=1; drop fnlwgt; one-hot 8 categoricals",
    )


# ===========================================================
# REGISTRY + DRIVER
# ===========================================================

REGRESSION_PROCESSORS: dict[str, Callable] = {
    "yacht_hydrodynamics": process_yacht_hydrodynamics,
    "boston_housing": process_boston_housing,
    "forest_fires": process_forest_fires,
    "energy_efficiency": process_energy_efficiency,
    "concrete_strength": process_concrete_strength,
    "wine_quality": process_wine_quality,
    "abalone_age": process_abalone_age,
    "parkinsons_updrs": process_parkinsons_updrs,
    "kin8nm": process_kin8nm,
    "combined_cycle_power_plant": process_combined_cycle_power_plant,
    "naval_propulsion": process_naval_propulsion,
    "bike_sharing": process_bike_sharing,
    "california_housing": process_california_housing,
    "superconductor": process_superconductor,
    "protein_tertiary_structure": process_protein_tertiary_structure,
}

CLASSIFICATION_PROCESSORS: dict[str, Callable] = {
    "ionosphere": process_ionosphere,
    "breast_cancer_wisconsin": process_breast_cancer_wisconsin,
    "credit_approval": process_credit_approval,
    "heart_disease": process_heart_disease,
    "pima_diabetes": process_pima_diabetes,
    "titanic": process_titanic,
    "german_credit": process_german_credit,
    "aids_ctg_175": process_aids_ctg_175,
    "spambase": process_spambase,
    "default_credit_card": process_default_credit_card,
    "bank_marketing": process_bank_marketing,
    "adult_income": process_adult_income,
}

ALL_PROCESSORS: dict[str, Callable] = {**REGRESSION_PROCESSORS, **CLASSIFICATION_PROCESSORS}


def process_one(name: str) -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = ALL_PROCESSORS[name]()

    prov_path = RAW_DIR / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        if name in prov:
            meta["source_provenance"] = {
                "source": prov[name].get("source"),
                "id": prov[name].get("id"),
                "sha256": prov[name].get("sha256"),
            }

    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    (PROCESSED_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(
        f"  [ok]  {name:<30s}  n={meta['n_samples']:>6d}  d={meta['n_features']:>3d}  "
        f"task={meta['task']}  ({meta['n_samples_before_na_drop']} → {meta['n_samples']})"
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated dataset names")
    args = parser.parse_args()

    names = list(ALL_PROCESSORS)
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        unknown = want - set(ALL_PROCESSORS)
        if unknown:
            print(f"Unknown processors: {unknown}", file=sys.stderr)
            return 1
        names = [n for n in names if n in want]

    print(f"Processing {len(names)} datasets → {PROCESSED_DIR}/")
    errors: list[tuple[str, Exception]] = []
    for name in names:
        try:
            process_one(name)
        except Exception as e:
            print(f"  [ERROR]  {name}: {e}", file=sys.stderr)
            errors.append((name, e))

    if errors:
        print(f"\nFailed: {len(errors)}/{len(names)}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
