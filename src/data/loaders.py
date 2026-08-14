"""Loaders for the simulation's input artifacts.

The pool and its geography are built by separate scripts (build_property_pool.py,
build_pc4_geography.py) because the geography needs listing counts from the pool.
Joining them belongs in one place rather than in every consumer.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

POOL_PATH = DATA_DIR / "property_pool.csv"
GEOGRAPHY_PATH = DATA_DIR / "pc4_geography.csv"

GEO_COLUMNS = ["pc4", "dist_to_centre_km", "area_km2", "listings_per_km2"]


def load_property_pool(with_geography: bool = True) -> pd.DataFrame:
    """Load the property pool, optionally joined to PC4 geography.

    Raises if either artifact is missing rather than silently returning a partial
    pool — a missing join would show up downstream as properties no buyer can
    reach, which is indistinguishable from a modelling result.
    """
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"{POOL_PATH} not found. Run: python scripts/build_property_pool.py"
        )
    pool = pd.read_csv(POOL_PATH)

    if not with_geography:
        return pool

    if not GEOGRAPHY_PATH.exists():
        raise FileNotFoundError(
            f"{GEOGRAPHY_PATH} not found. Run: python scripts/build_pc4_geography.py"
        )
    geo = pd.read_csv(GEOGRAPHY_PATH)[GEO_COLUMNS]

    merged = pool.merge(geo, on="pc4", how="left", validate="many_to_one")

    unmatched = merged["dist_to_centre_km"].isna().sum()
    if unmatched:
        codes = sorted(merged.loc[merged["dist_to_centre_km"].isna(), "pc4"].unique())
        raise ValueError(
            f"{unmatched} properties have no geography for PC4 {codes}. "
            "Rebuild pc4_geography.csv."
        )

    if len(merged) != len(pool):
        raise ValueError(
            f"Geography join changed row count: {len(pool)} -> {len(merged)}"
        )

    return merged
