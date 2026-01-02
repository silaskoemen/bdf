"""
Load and save datasets for benchmarking from e.g. ucimlrepo.
"""

import pandas as pd
from loguru import logger


def _join_drop_na(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Join X and y, drop na, return combined DataFrame."""
    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    return data


def main():
    logger.info("Downloading and saving breast cancer wisconsin dataset...")
    # Load original breast cancer wisconsin dataset
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_original.data.features  # type: ignore[attr-defined]
    y = breast_cancer_wisconsin_original.data.targets["Class"]  # type: ignore[attr-defined]
    y = (y == 4).astype(int)  # convert to binary: benign (2) -> 0, malignant (4) -> 1

    breast_cancer_df = _join_drop_na(X, y)

    # Save to csv
    breast_cancer_df.to_csv("./data/raw/breast_cancer_wisconsin.csv", index=False)
    logger.success("Saved breast cancer wisconsin dataset to ./data/raw/breast_cancer_wisconsin.csv")

    # Load concrete strength dataset
    logger.info("Downloading and saving concrete strength dataset...")
    concrete_data = fetch_ucirepo(id=165)
    X_concrete = concrete_data.data.features  # type: ignore[attr-defined]
    y_concrete = concrete_data.data.targets["Concrete compressive strength"]  # type: ignore[attr-defined]

    concrete_df = _join_drop_na(X_concrete, y_concrete)
    concrete_df.to_csv("./data/raw/concrete_strength.csv", index=False)
    logger.success("Saved concrete strength dataset to ./data/raw/concrete_strength.csv")

    # Load energy efficiency dataset
    logger.info("Downloading and saving energy efficiency dataset...")
    energy_data = fetch_ucirepo(id=242)
    X_energy = energy_data.data.features  # type: ignore[attr-defined]
    y_energy = energy_data.data.targets["Y1"]  # type: ignore[attr-defined]

    energy_df = _join_drop_na(X_energy, y_energy)
    energy_df.to_csv("./data/raw/energy_efficiency.csv", index=False)
    logger.success("Saved energy efficiency dataset to ./data/raw/energy_efficiency.csv")

    # Load combined cycle power plant dataset
    logger.info("Downloading and saving combined cycle power plant dataset...")
    power_plant_data = fetch_ucirepo(id=294)
    X_power = power_plant_data.data.features  # type: ignore[attr-defined]
    y_power = power_plant_data.data.targets["PE"]  # type: ignore[attr-defined]
    power_plant_df = _join_drop_na(X_power, y_power)
    power_plant_df.to_csv("./data/raw/combined_cycle_power_plant.csv", index=False)
    logger.success("Saved combined cycle power plant dataset to ./data/raw/combined_cycle_power_plant.csv")

    # Load superconductor dataset
    logger.info("Downloading and saving superconductor dataset...")
    superconductor_data = fetch_ucirepo(id=464)
    X_super = superconductor_data.data.features  # type: ignore[attr-defined]
    y_super = superconductor_data.data.targets["critical_temp"]  # type: ignore[attr-defined]
    superconductor_df = _join_drop_na(X_super, y_super)
    superconductor_df.to_csv("./data/raw/superconductor.csv", index=False)
    logger.success("Saved superconductor dataset to ./data/raw/superconductor.csv")

    # Load bike sharing dataset
    logger.info("Downloading and saving bike sharing dataset...")
    bike_data = fetch_ucirepo(id=275)
    X_bike = bike_data.data.features  # type: ignore[attr-defined]
    y_bike = bike_data.data.targets["cnt"]  # type: ignore[attr-defined]
    bike_df = _join_drop_na(X_bike, y_bike)
    bike_df.to_csv("./data/raw/bike_sharing.csv", index=False)
    logger.success("Saved bike sharing dataset to ./data/raw/bike_sharing.csv")


if __name__ == "__main__":
    main()
