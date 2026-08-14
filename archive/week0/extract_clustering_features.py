"""
Extract clustering features from cached preprocessed data.

Loads cached preprocessed pickle, computes missing features,
and exports clustering-ready CSV.

Output:
  data/clustering_features.csv (price_per_m2, size_num, bedrooms, luxury_score, postal_code_clean, nr_rooms)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Luxury amenity weights (from feature_engineering.py)
LUXURY_AMENITIES_WEIGHTS = {
    "has_lift": 1,
    "has_sauna": 2,
    "has_domotica": 1.5,
    "has_airconditioning": 1,
    "has_zwembad": 3,
}


def load_cached_preprocessed_data(cache_path="data/cache/df_preprocessed_ca6f817b.pkl"):
    """Load cached preprocessed dataframe."""
    with open(cache_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Loaded cached data: {df.shape}")
    return df


def compute_price_per_m2(df):
    """Compute price per m² from price_num and size_num."""
    df = df.copy()
    df['price_per_m2'] = df['price_num'] / df['size_num']
    print(f"Computed price_per_m2")
    return df


def compute_luxury_score(df):
    """Compute luxury_score as weighted sum of luxury amenities."""
    df = df.copy()

    # Check which luxury amenity columns exist
    luxury_cols = [col for col in LUXURY_AMENITIES_WEIGHTS.keys() if col in df.columns]

    if not luxury_cols:
        print("Warning: No luxury amenity columns found. Setting luxury_score to 0.")
        df['luxury_score'] = 0
        return df

    # Compute weighted sum
    df['luxury_score'] = sum(
        df[col] * LUXURY_AMENITIES_WEIGHTS[col]
        for col in luxury_cols
    )

    print(f"Computed luxury_score from {len(luxury_cols)} luxury amenities")
    return df


def select_clustering_features(df):
    """Select features for clustering."""
    clustering_features = [
        'price_per_m2',
        'size_num',
        'bedrooms',
        'luxury_score',
        'postal_code_clean',
        'nr_rooms',
    ]

    # Keep only clustering features
    df_cluster = df[clustering_features].copy()

    print(f"Selected {len(clustering_features)} clustering features")
    return df_cluster


def clean_for_clustering(df):
    """Remove rows with missing values in key clustering features."""
    initial_rows = len(df)

    # Drop rows with NaN in price_per_m2, size_num, bedrooms
    df_clean = df.dropna(subset=['price_per_m2', 'size_num', 'bedrooms'])

    rows_dropped = initial_rows - len(df_clean)
    print(f"Cleaned data: {rows_dropped} rows dropped due to missing values")
    print(f"Final dataset: {len(df_clean)} rows")

    return df_clean


def export_to_csv(df, output_path="data/clustering_features.csv"):
    """Export to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported to: {output_path}")
    return output_path


def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("CLUSTERING FEATURES SUMMARY")
    print("="*80)

    numeric_features = df.select_dtypes(include=[np.number]).columns

    for col in numeric_features:
        print(f"\n{col}:")
        print(f"  Mean:   {df[col].mean():,.2f}")
        print(f"  Median: {df[col].median():,.2f}")
        print(f"  Min:    {df[col].min():,.2f}")
        print(f"  Max:    {df[col].max():,.2f}")
        print(f"  Std:    {df[col].std():,.2f}")
        print(f"  NaN:    {df[col].isna().sum()}")

    # Postal code stats
    print(f"\npostal_code_clean:")
    print(f"  Unique: {df['postal_code_clean'].nunique()}")
    print(f"  NaN:    {df['postal_code_clean'].isna().sum()}")


def main():
    """Main extraction pipeline."""
    print("="*80)
    print("EXTRACTING CLUSTERING FEATURES")
    print("="*80)

    # 1. Load cached data
    df = load_cached_preprocessed_data()

    # 2. Compute missing features
    df = compute_price_per_m2(df)
    df = compute_luxury_score(df)

    # 3. Select clustering features
    df_cluster = select_clustering_features(df)

    # 4. Clean for clustering
    df_clean = clean_for_clustering(df_cluster)

    # 5. Export
    csv_path = export_to_csv(df_clean)

    # 6. Print summary
    print_summary_stats(df_clean)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Ready for clustering: {csv_path}")

    return csv_path


if __name__ == "__main__":
    main()
