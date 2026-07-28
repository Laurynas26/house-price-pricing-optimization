"""
Run K-Means clustering on property features and generate property segmentation analysis.

PURPOSE:
  Segment the 4,054 real Funda properties into 8 clusters for valuation and analysis.
  Map elasticity values to property clusters for ANALYSIS ONLY.

  **Important:** Elasticity here is for property segmentation tier (used in validation/analysis).
  Buyer elasticity is defined separately in config/buyer_archetypes.yaml (for simulation behavior).

Clusters on: price_per_m2, size_num, bedrooms, luxury_score, nr_rooms
Output: config/elasticity_mapping.yaml (property cluster → segmentation tier mapping)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import yaml

# Property segmentation elasticity tiers (for ANALYSIS, not simulation)
# These tiers group properties by price level and are used for:
#  - Validation: comparing sim prices to real Funda prices by segment
#  - Analysis: understanding how different property segments respond
# Buyer elasticity (for simulation behavior) is defined separately in config/buyer_archetypes.yaml
# Based on: Kaplan-Violante (2020) - luxury market theory
ELASTICITY_TIERS_PERCENTILE = {
    "ultra_luxury": {
        "percentile_min": 99,
        "elasticity": -0.35,
        "description": "Top 1% (ultra-luxury, affluent buyers, price-insensitive)"
    },
    "luxury_upper": {
        "percentile_min": 80,
        "percentile_max": 99,
        "elasticity": -0.50,
        "description": "80-99th percentile (luxury, less price-sensitive)"
    },
    "luxury_mid": {
        "percentile_min": 50,
        "percentile_max": 80,
        "elasticity": -0.65,
        "description": "50-80th percentile (mid-to-upper market, moderate elasticity)"
    },
    "mid_market": {
        "percentile_min": 25,
        "percentile_max": 50,
        "elasticity": -0.80,
        "description": "25-50th percentile (mid-market, balanced)"
    },
    "lower_mid": {
        "percentile_min": 5,
        "percentile_max": 25,
        "elasticity": -0.95,
        "description": "5-25th percentile (lower-mid, price-conscious)"
    }
}


def load_clustering_features(csv_path="data/clustering_features.csv"):
    """Load clustering features from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} properties with {len(df.columns)} features")
    return df


def prepare_clustering_data(df):
    """Prepare data for clustering: select and standardize features."""
    clustering_cols = ['price_per_m2', 'size_num', 'bedrooms', 'luxury_score', 'nr_rooms']

    X = df[clustering_cols].copy()

    # Remove rows with NaN (should be minimal)
    X_clean = X.dropna()
    print(f"After removing NaN: {len(X_clean)} rows")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print(f"Standardized {len(clustering_cols)} features")

    return X_scaled, X_clean, clustering_cols, scaler


def run_kmeans(X_scaled, n_clusters=8):
    """Run K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    print(f"K-Means completed: {n_clusters} clusters")
    return kmeans, clusters


def analyze_clusters(df_clean, clusters, clustering_cols):
    """Analyze characteristics of each cluster."""
    df_clean = df_clean.copy()
    df_clean['cluster'] = clusters

    cluster_stats = []

    for cluster_id in range(8):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]

        stats = {
            'cluster': cluster_id,
            'count': len(cluster_data),
            'price_per_m2_median': cluster_data['price_per_m2'].median(),
            'price_per_m2_mean': cluster_data['price_per_m2'].mean(),
            'size_median': cluster_data['size_num'].median(),
            'bedrooms_mean': cluster_data['bedrooms'].mean(),
            'luxury_score_mean': cluster_data['luxury_score'].mean(),
            'nr_rooms_mean': cluster_data['nr_rooms'].mean(),
        }
        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats).sort_values('price_per_m2_median', ascending=False)

    print("\n" + "="*100)
    print("CLUSTER ANALYSIS (sorted by price/m²)")
    print("="*100)
    print(stats_df.to_string(index=False))

    return stats_df


def assign_elasticity(stats_df):
    """Assign elasticity values to clusters based on Amsterdam-relative percentiles."""
    elasticity_mapping = {}

    # Calculate price/m² percentile for each cluster (0-100, higher price = higher percentile)
    # Use percentileofscore to get correct percentile ranks
    from scipy.stats import percentileofscore

    prices = stats_df['price_per_m2_median'].values
    price_percentiles = {}
    for cluster_id in stats_df['cluster'].values:
        price = stats_df[stats_df['cluster'] == cluster_id]['price_per_m2_median'].values[0]
        percentile = percentileofscore(prices, price)
        price_percentiles[cluster_id] = percentile

    for idx, row in stats_df.iterrows():
        cluster_id = int(row['cluster'])
        price_m2 = row['price_per_m2_median']
        percentile = price_percentiles[cluster_id]

        # Determine tier based on percentile (Amsterdam-relative)
        if percentile >= 99:
            tier = "ultra_luxury"
        elif percentile >= 80:
            tier = "luxury_upper"
        elif percentile >= 50:
            tier = "luxury_mid"
        elif percentile >= 25:
            tier = "mid_market"
        else:
            tier = "lower_mid"

        tier_info = ELASTICITY_TIERS_PERCENTILE[tier]

        elasticity_mapping[cluster_id] = {
            'cluster_id': cluster_id,
            'tier': tier,
            'description': tier_info['description'],
            'elasticity': tier_info['elasticity'],
            'price_per_m2_median': float(price_m2),
            'price_percentile': float(percentile),
            'count_listings': int(row['count']),
            'size_median': float(row['size_median']),
            'bedrooms_mean': float(row['bedrooms_mean']),
            'luxury_score_mean': float(row['luxury_score_mean']),
            'reasoning': f"Percentile {percentile:.1f}% of Amsterdam market -> {tier} tier (elasticity {tier_info['elasticity']})"
        }

    return elasticity_mapping


def export_elasticity_mapping(elasticity_mapping, output_path="config/elasticity_mapping.yaml"):
    """Export elasticity mapping to YAML."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mapping_dict = {
        'elasticity_mapping': elasticity_mapping,
        'metadata': {
            'source': 'K-Means clustering on 4,054 Amsterdam properties',
            'features': ['price_per_m2', 'size_num', 'bedrooms', 'luxury_score', 'nr_rooms'],
            'n_clusters': len(elasticity_mapping),
            'research_based': True,
            'elasticity_range': 'Kaplan-Violante (2020)'
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(mapping_dict, f, default_flow_style=False)

    print(f"\nExported elasticity mapping to: {output_path}")
    return output_path


def print_elasticity_summary(elasticity_mapping):
    """Print elasticity mapping summary."""
    print("\n" + "="*100)
    print("ELASTICITY MAPPING (by cluster)")
    print("="*100)

    for cluster_id in sorted(elasticity_mapping.keys()):
        mapping = elasticity_mapping[cluster_id]
        print(f"\nCluster {cluster_id} ({mapping['tier'].upper()}):")
        print(f"  Elasticity: {mapping['elasticity']:.2f}")
        print(f"  Price/m²: €{mapping['price_per_m2_median']:,.0f}")
        print(f"  Size (median): {mapping['size_median']:.0f} m²")
        print(f"  Bedrooms: {mapping['bedrooms_mean']:.1f}")
        print(f"  Luxury score: {mapping['luxury_score_mean']:.2f}")
        print(f"  Listings: {mapping['count_listings']}")
        print(f"  Reasoning: {mapping['reasoning']}")


def main():
    """Main clustering pipeline."""
    print("="*100)
    print("K-MEANS CLUSTERING + ELASTICITY MAPPING")
    print("="*100)

    # 1. Load data
    df = load_clustering_features()

    # 2. Prepare for clustering
    X_scaled, X_clean, clustering_cols, scaler = prepare_clustering_data(df)

    # 3. Run K-Means
    kmeans, clusters = run_kmeans(X_scaled, n_clusters=8)

    # 4. Analyze clusters
    stats_df = analyze_clusters(X_clean, clusters, clustering_cols)

    # 5. Assign elasticity
    elasticity_mapping = assign_elasticity(stats_df)

    # 6. Export mapping
    export_elasticity_mapping(elasticity_mapping)

    # 7. Print summary
    print_elasticity_summary(elasticity_mapping)

    print("\n" + "="*100)
    print("CLUSTERING COMPLETE")
    print("="*100)
    print("Elasticity mapping locked and exported to config/elasticity_mapping.yaml")
    print("Ready for Week 1 simulation work!")


if __name__ == "__main__":
    main()
