"""
Build the simulation property pool from cached preprocessed Funda listings.

Supersedes extract_clustering_features.py, which dropped two columns the simulation
cannot run without:

  - price_num   the asking price. Without it the pool has no price level at all, only
                price per m2, so archetype budget filters and the valuation anchor are
                both unevaluable. The archetype config claims budgets are "IQR from
                data" but the committed CSV could not have produced them.
  - neighborhood / postal code beyond a raw string, so location preferences could not
                be evaluated either.

Outputs data/property_pool.csv, one row per property, carrying everything the
simulation, the valuation module and the segmentation need.

Run from the repo root:
    python scripts/build_property_pool.py
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "data" / "df_preprocessed_ca6f817b.pkl"
ZONES_PATH = REPO_ROOT / "config" / "location_zones.yaml"
OUTPUT_PATH = REPO_ROOT / "data" / "property_pool.csv"

# Weights carried over unchanged from the prediction project's feature_engineering.py
# so luxury_score stays comparable across the two repos.
LUXURY_AMENITY_WEIGHTS = {
    "has_lift": 1.0,
    "has_sauna": 2.0,
    "has_domotica": 1.5,
    "has_airconditioning": 1.0,
    "has_zwembad": 3.0,
}

# Size band edges recovered from the prediction project's held-out residual file
# (reports/test_residuals_enriched.csv), where bands were pd.qcut(size_num, q=5).
# The valuation module samples model error within these bands, so the pool must be
# banded on exactly the same edges or the error profile is applied to the wrong
# properties. Edges are inclusive upper bounds in m2.
SIZE_BAND_EDGES = [
    ("XS", 50.0),
    ("S", 66.0),
    ("M", 79.0),
    ("L", 102.0),
    ("XL", np.inf),
]


def load_cached_listings(path: Path = CACHE_PATH) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    print(f"Loaded cached listings: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_per_m2"] = df["price_num"] / df["size_num"]

    present = [c for c in LUXURY_AMENITY_WEIGHTS if c in df.columns]
    missing = sorted(set(LUXURY_AMENITY_WEIGHTS) - set(present))
    if missing:
        print(f"  WARNING: luxury amenity columns absent from cache: {missing}")
    df["luxury_score"] = sum(df[c] * LUXURY_AMENITY_WEIGHTS[c] for c in present)

    print(f"  Derived price_per_m2 and luxury_score (from {len(present)} amenities)")
    return df


def assign_size_band(size_num: pd.Series) -> pd.Series:
    """Band properties on the prediction audit's held-out qcut edges."""
    band = pd.Series(index=size_num.index, dtype="object")
    lower = -np.inf
    for label, upper in SIZE_BAND_EDGES:
        band[(size_num > lower) & (size_num <= upper)] = label
        lower = upper
    return band


def load_zone_lookup(path: Path = ZONES_PATH) -> tuple[dict[int, str], set[int]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lookup: dict[int, str] = {}
    for zone_key, zone in cfg["zones"].items():
        for pc4 in zone["pc4"]:
            if pc4 in lookup:
                raise ValueError(
                    f"PC4 {pc4} assigned to both '{lookup[pc4]}' and '{zone_key}'"
                )
            lookup[pc4] = zone_key

    return lookup, set(cfg["canal_belt_pc4"])


def assign_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    postal = df["postal_code_clean"].astype(str).str.strip()
    df["pc4"] = pd.to_numeric(postal.str[:4], errors="coerce")
    df["postal_district"] = postal.str[:3]

    lookup, canal_belt = load_zone_lookup()
    df["zone"] = df["pc4"].map(lookup)
    df["is_canal_belt"] = df["pc4"].isin(canal_belt)

    no_postal = df["pc4"].isna().sum()
    unmapped = df.loc[df["pc4"].notna() & df["zone"].isna(), "pc4"]
    if len(unmapped):
        codes = sorted(unmapped.unique().astype(int))
        raise ValueError(
            f"{len(unmapped)} properties have PC4 codes absent from "
            f"config/location_zones.yaml: {codes}. Add them before continuing — "
            "silently dropping them would bias the pool geographically."
        )

    print(f"  Mapped all {df['pc4'].notna().sum():,} located rows to {df['zone'].nunique()} zones")
    if no_postal:
        print(f"  ({no_postal:,} rows carry no postal code and cannot be zoned)")

    return df


def fix_vve_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Undo a 100x inflation of the VvE service charge inherited from upstream parsing.

    The prediction project's parse_price() is re.sub(r"[^\\d]", "", s), which strips
    every non-digit including the Dutch decimal comma. That is harmless for asking
    prices, which Funda renders without decimals, but VvE contributions always carry
    cents:

        "EUR 105,46 per maand"  ->  10546      (should be 105.46)
        "EUR 254,00 per maand"  ->  25400      (should be 254.00)

    Uncorrected, the median service charge reads as EUR 16,001/month against a
    realistic EUR 159. The error is a uniform factor of 100, so it is monotonic and
    does not affect tree-based models in the prediction project, but the values are
    wrong and cannot be used as euros here.
    """
    df = df.copy()
    before = df["contribution_vve_num"].median()
    df["contribution_vve_num"] = df["contribution_vve_num"] / 100.0
    after = df["contribution_vve_num"].median()
    print(f"  Corrected VvE scale: median {before:,.0f} -> {after:,.2f} EUR/month")
    return df


def normalise_city(df: pd.DataFrame) -> pd.DataFrame:
    """The cache contains 134 rows with city 'STERDAM', a scraper truncation bug.

    Left uncorrected these would look like a second municipality. Every listing in
    this dataset is Amsterdam (confirmed: PC4 codes are all 10xx/11xx), so the value
    is repaired rather than dropped.
    """
    df = df.copy()
    before = df["city"].value_counts().to_dict()
    df["city"] = df["city"].replace({"STERDAM": "AMSTERDAM"})
    if before.get("STERDAM"):
        print(f"  Repaired {before['STERDAM']} truncated city values ('STERDAM')")
    return df


def build_pool(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "price_num",
        "size_num",
        "bedrooms",
        "nr_rooms",
        "luxury_score",
        "price_per_m2",
        "size_band",
        "postal_code_clean",
        "pc4",
        "postal_district",
        "zone",
        "is_canal_belt",
        "neighborhood",
        "city",
        "contribution_vve_num",
        "year_of_construction_num",
        "energy_label",
    ]

    n_before = len(df)
    # Same dropna subset the original archetype clustering used, so the new pool is
    # row-identical to the 4,054 the existing configs describe.
    pool = df.dropna(subset=["price_per_m2", "size_num", "bedrooms"]).copy()
    print(f"  Dropped {n_before - len(pool):,} rows missing price/size/bedrooms")

    pool["size_band"] = assign_size_band(pool["size_num"])

    missing_cols = [c for c in keep if c not in pool.columns]
    if missing_cols:
        raise KeyError(f"Expected columns absent from cache: {missing_cols}")

    pool = pool[keep].reset_index(drop=False).rename(columns={"index": "listing_id"})

    # Every surviving property must be locatable — buyer preference filters are
    # evaluated on `zone`, and a null zone would silently make a property invisible
    # to every buyer rather than raising.
    unzoned = pool["zone"].isna().sum()
    if unzoned:
        raise ValueError(
            f"{unzoned} properties survived cleaning without a zone. "
            "They would be unreachable by every buyer filter."
        )

    return pool


def report(pool: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("PROPERTY POOL SUMMARY")
    print("=" * 78)
    print(f"Properties: {len(pool):,}")

    print("\nAsking price:")
    q = pool["price_num"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    for p, v in q.items():
        print(f"  p{int(p * 100):<3} EUR {v:>12,.0f}")

    print("\nBy zone:")
    z = pool.groupby("zone").agg(
        n=("price_num", "size"),
        median_price=("price_num", "median"),
        median_ppm2=("price_per_m2", "median"),
    ).sort_values("median_ppm2", ascending=False)
    for zone, row in z.iterrows():
        print(
            f"  {zone:<12} n={int(row['n']):>5}  "
            f"median EUR {row['median_price']:>10,.0f}  "
            f"EUR {row['median_ppm2']:>7,.0f}/m2"
        )

    print("\nBy size band (edges from prediction-audit held-out qcut):")
    b = pool.groupby("size_band").agg(
        n=("size_num", "size"), min_m2=("size_num", "min"), max_m2=("size_num", "max")
    ).reindex([label for label, _ in SIZE_BAND_EDGES])
    for band, row in b.iterrows():
        print(f"  {band:<3} n={int(row['n']):>5}  {row['min_m2']:>5.0f}-{row['max_m2']:>5.0f} m2")

    vve = pool["contribution_vve_num"]
    vve_missing = vve.isna().sum()
    print(
        f"\nVvE contribution present for {len(pool) - vve_missing:,}/{len(pool):,} "
        f"({100 * (1 - vve_missing / len(pool)):.1f}%), median "
        f"EUR {vve.median():,.0f}/month"
    )
    print("  Missing is informative, not a gap: VvE applies to apartments, so an")
    print("  absent value usually means a freehold house with no service charge.")
    median_price = pool["price_num"].median()
    print(
        f"  Materiality: 90 days of median VvE is EUR {vve.median() * 3:,.0f}, "
        f"{100 * vve.median() * 3 / median_price:.2f}% of the median asking price — "
        "far below the discount-rate term, so carrying cost is modelled but is not "
        "the time cost that drives pricing."
    )


def main() -> Path:
    print("=" * 78)
    print("BUILDING PROPERTY POOL")
    print("=" * 78)

    df = load_cached_listings()
    df = compute_derived_features(df)
    df = fix_vve_scale(df)
    df = normalise_city(df)
    df = assign_location(df)
    pool = build_pool(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT_PATH, index=False)

    report(pool)
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
