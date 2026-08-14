"""
Derive PC4 geography for the property pool from the Dutch PC4 shapefile.

WHAT THIS PROVIDES
------------------
  dist_to_centre_km   straight-line distance from the PC4 centroid to Dam Square
  area_km2            PC4 polygon area
  listings_per_km2    pool listings per km2 (supply thickness)

WHY GEOMETRY AND NOT THE EXISTING CLUSTER LABELS
------------------------------------------------
The prediction project's data_exploration notebook already clusters PC4 areas on
price_per_m2, size, rooms and bedrooms (cells 39-54). Those cluster labels are
deliberately NOT reused. They are fitted on price, so adopting them as "location
zones" and then reporting that central zones command a premium would restate the
fitting procedure as a finding — the circularity flagged in DESIGN.md section 5.

Two further reasons not to inherit them directly:
  - The applied clusterings pass the whole frame to the scaler, so `num_listings`
    acts as a clustering feature. That is a scrape-volume artifact, not a market
    property, and standardised it carries the same weight as price.
  - The elbow curve (cell 53) is computed on four features, while the k=5 clustering
    actually applied uses five. The elbow describes a different feature space than
    the one that was fitted.

The three quantities above are derived from polygon geometry and listing counts only.
No price information enters them, so they can be used as explanatory inputs without
making later location results self-fulfilling.

A CORRECTION TO THE NOTEBOOK'S DISTANCE CALCULATION
---------------------------------------------------
The notebook (cell 54) computes:

    gdf = gdf.to_crs(epsg=4326)
    gdf["dist_to_center_km"] = gdf["centroid"].distance(amsterdam_center) * 111

That measures distance in DEGREES on an unprojected lat/lon CRS and converts with a
flat 111 km/degree. The factor 111 is only correct for latitude. At Amsterdam's
latitude a degree of longitude spans 111 * cos(52.37 deg) ~= 67.8 km, so east-west
separation is overstated by roughly 64%. Any area displaced east or west of the
centre looks further away than it is, and the distortion grows with that displacement.

This script projects to EPSG:28992 (Amersfoort / RD New, the metre-based Dutch
national grid) and measures true planar distance. Centroids are computed in the
projected CRS, where they are geometrically meaningful. Both values are reported so
the size of the correction is auditable rather than asserted.

Run from the repo root (after build_property_pool.py):
    python scripts/build_pc4_geography.py
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_REPO = REPO_ROOT.parent / "house_price_prediction_project"
SHAPEFILE = (
    PREDICTION_REPO
    / "data"
    / "georef-netherlands-postcode-pc4"
    / "georef-netherlands-postcode-pc4.shp"
)
POOL_PATH = REPO_ROOT / "data" / "property_pool.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "pc4_geography.csv"

# Dam Square, the conventional centre of Amsterdam (WGS84 lon/lat).
AMSTERDAM_CENTRE_WGS84 = Point(4.8926, 52.3730)

# Amersfoort / RD New — the standard Dutch projected CRS, units in metres.
DUTCH_CRS = "EPSG:28992"


def load_pc4_shapes(path: Path = SHAPEFILE) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"PC4 shapefile not found at {path}\n"
            "Expected the prediction project as a sibling checkout."
        )
    gdf = gpd.read_file(path)
    gdf["pc4"] = pd.to_numeric(gdf["pc4_code"].astype(str).str.strip(), errors="coerce")
    print(f"Loaded {len(gdf):,} PC4 polygons  (source CRS: {gdf.crs})")
    return gdf


def compute_geometry_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    projected = gdf.to_crs(DUTCH_CRS)
    centre = (
        gpd.GeoSeries([AMSTERDAM_CENTRE_WGS84], crs="EPSG:4326")
        .to_crs(DUTCH_CRS)
        .iloc[0]
    )
    print(f"Projected to {DUTCH_CRS} (metres) before measuring")

    centroids = projected.geometry.centroid
    out = pd.DataFrame(
        {
            "pc4": projected["pc4"].values,
            "dist_to_centre_km": centroids.distance(centre).values / 1_000.0,
            "area_km2": projected.geometry.area.values / 1e6,
        }
    )

    # Reproduce the notebook's degree-based figure so the correction is measurable.
    # geopandas correctly warns that centroid/distance on a geographic CRS are
    # unreliable — that warning IS the bug being quantified here, so it is silenced
    # rather than fixed.
    wgs = gdf.to_crs("EPSG:4326")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*")
        out["dist_naive_deg111_km"] = (
            wgs.geometry.centroid.distance(AMSTERDAM_CENTRE_WGS84) * 111.0
        ).values

    return out.dropna(subset=["pc4"]).astype({"pc4": int})


def attach_pool_counts(geo: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    counts = pool.groupby("pc4").size().rename("n_listings")
    geo = geo.merge(counts, left_on="pc4", right_index=True, how="right")

    missing_geom = geo["area_km2"].isna().sum()
    if missing_geom:
        codes = sorted(geo.loc[geo["area_km2"].isna(), "pc4"].tolist())
        raise ValueError(
            f"{missing_geom} PC4 areas in the pool have no polygon in the "
            f"shapefile: {codes}"
        )

    geo["listings_per_km2"] = geo["n_listings"] / geo["area_km2"]
    return geo.sort_values("dist_to_centre_km").reset_index(drop=True)


def report(geo: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("DISTANCE CORRECTION — projected (EPSG:28992) vs notebook (degrees x 111)")
    print("=" * 78)

    geo = geo.copy()
    geo["overstated_%"] = 100 * (
        geo["dist_naive_deg111_km"] / geo["dist_to_centre_km"] - 1
    )
    finite = geo.replace([np.inf, -np.inf], np.nan).dropna(subset=["overstated_%"])

    print(f"  mean overstatement:   {finite['overstated_%'].mean():>6.1f}%")
    print(f"  median overstatement: {finite['overstated_%'].median():>6.1f}%")
    print(f"  max overstatement:    {finite['overstated_%'].max():>6.1f}%")
    print("\n  Largest distortions (most east-west displaced from the centre):")
    for _, r in finite.nlargest(5, "overstated_%").iterrows():
        print(
            f"    PC4 {int(r['pc4'])}: true {r['dist_to_centre_km']:>5.2f} km  "
            f"vs naive {r['dist_naive_deg111_km']:>5.2f} km  "
            f"(+{r['overstated_%']:.1f}%)"
        )

    print("\n" + "=" * 78)
    print("PC4 GEOGRAPHY")
    print("=" * 78)
    print(f"  PC4 areas: {len(geo)}")
    print(
        f"  distance to centre: {geo['dist_to_centre_km'].min():.2f}"
        f" – {geo['dist_to_centre_km'].max():.2f} km"
    )

    print("\n  Closest to centre:")
    for _, r in geo.nsmallest(5, "dist_to_centre_km").iterrows():
        print(
            f"    PC4 {int(r['pc4'])}  {r['dist_to_centre_km']:>5.2f} km  "
            f"n={int(r['n_listings']):>4}  {r['listings_per_km2']:>7.1f} listings/km2"
        )

    print("\n  Furthest from centre:")
    for _, r in geo.nlargest(5, "dist_to_centre_km").iterrows():
        print(
            f"    PC4 {int(r['pc4'])}  {r['dist_to_centre_km']:>5.2f} km  "
            f"n={int(r['n_listings']):>4}  {r['listings_per_km2']:>7.1f} listings/km2"
        )


def main() -> Path:
    print("=" * 78)
    print("BUILDING PC4 GEOGRAPHY")
    print("=" * 78)

    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"{POOL_PATH} not found. Run scripts/build_property_pool.py first."
        )
    pool = pd.read_csv(POOL_PATH)

    gdf = load_pc4_shapes()
    geo = compute_geometry_features(gdf)
    geo = attach_pool_counts(geo, pool)

    geo.to_csv(OUTPUT_PATH, index=False)
    report(geo)
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
