"""
Reproducible audit of the Week 0 repo state.

Every claim used to justify the Week 1 rebuild is checked here so it can be
re-run rather than taken on trust. Each check prints PASS / FAIL / INFO and the
numbers behind it.

CHECK 1  Are the buyer archetypes reproducible?
CHECK 2  Do elasticity_mapping.yaml and buyer_archetypes.yaml describe the same
         segmentation?
CHECK 3  Can the committed clustering_features.csv regenerate the archetypes?
CHECK 4  Do the archetype location filters match anything in the data?
CHECK 5  How many properties receive zero eligible buyers?
CHECK 6  Are tolerance bands a function of elasticity?
CHECK 7  What error profile does the prediction audit actually supply?

Run from the repo root:
    python scripts/audit_week0_state.py
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "data" / "df_preprocessed_ca6f817b.pkl"

# The audited artifacts now live in archive/week0/. They were moved rather than
# deleted precisely so this audit still runs — it is the evidence for retiring them,
# and evidence that cannot be re-executed is just an assertion.
ARCHIVE = REPO_ROOT / "archive" / "week0"
FEATURES_CSV = ARCHIVE / "clustering_features.csv"
ARCHETYPES_YAML = ARCHIVE / "buyer_archetypes.yaml"
ELASTICITY_YAML = ARCHIVE / "elasticity_mapping.yaml"

# The prediction project is a sibling checkout. Its held-out residuals are the only
# honest measurement of valuation error available to this project.
PREDICTION_REPO = REPO_ROOT.parent / "house_price_prediction_project"
RESIDUALS_CSV = PREDICTION_REPO / "reports" / "test_residuals_enriched.csv"

SEED = 42


def banner(n: int, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"CHECK {n}  {title}")
    print("=" * 78)


def load_cache() -> pd.DataFrame:
    with open(CACHE_PATH, "rb") as f:
        df = pickle.load(f)
    df["price_per_m2"] = df["price_num"] / df["size_num"]
    return df


def cleaned_pool(df: pd.DataFrame) -> pd.DataFrame:
    """The 4,054-row subset every Week 0 config claims to describe."""
    return df.dropna(subset=["price_per_m2", "size_num", "bedrooms"]).copy()


def load_archetypes() -> dict:
    with open(ARCHETYPES_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- CHECK 1

def check_archetype_reproducibility(pool: pd.DataFrame, archetypes: dict) -> pd.DataFrame:
    banner(1, "Are the buyer archetypes reproducible from the raw cache?")

    X = StandardScaler().fit_transform(pool[["price_num", "size_num", "price_per_m2"]])
    pool = pool.copy()
    pool["cluster"] = KMeans(n_clusters=8, random_state=SEED, n_init=10).fit_predict(X)

    got = pool.groupby("cluster").agg(
        n=("price_num", "size"),
        price_med=("price_num", "median"),
        size_med=("size_num", "median"),
        ppm2=("price_per_m2", "median"),
    ).sort_values("price_med")

    claimed = sorted(
        a["count_properties"] for a in archetypes["buyer_archetypes"].values()
    )
    reproduced = sorted(got["n"].tolist())

    print(f"  rows clustered:     {len(pool):,}")
    print(f"  counts in YAML:     {claimed}")
    print(f"  counts reproduced:  {reproduced}")

    if claimed == reproduced:
        print("\n  PASS  Archetypes reproduce exactly.")
        print("        Recipe: dropna(price_per_m2, size_num, bedrooms) -> 4,054 rows,")
        print("        KMeans(k=8, seed=42) on [price_num, size_num, price_per_m2].")
        print("        This recipe is NOT committed as a script anywhere in the repo.")
    else:
        print("\n  FAIL  Archetype counts do not reproduce.")

    print()
    print(got.to_string())
    return got


# ---------------------------------------------------------------- CHECK 2

def check_two_segmentations(archetypes: dict) -> None:
    banner(2, "Do the two config files describe the same segmentation?")

    with open(ELASTICITY_YAML, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)["elasticity_mapping"]

    arch_counts = sorted(
        a["count_properties"] for a in archetypes["buyer_archetypes"].values()
    )
    map_counts = sorted(c["count_listings"] for c in mapping.values())

    print(f"  buyer_archetypes.yaml   counts: {arch_counts}  (sum {sum(arch_counts):,})")
    print(f"  elasticity_mapping.yaml counts: {map_counts}  (sum {sum(map_counts):,})")
    print()
    print("  buyer_archetypes.yaml  clusters on: price_num, size_num, price_per_m2")
    print("  elasticity_mapping.yaml clusters on: price_per_m2, size_num, bedrooms,")
    print("                                       luxury_score, nr_rooms")

    if arch_counts == map_counts:
        print("\n  PASS  Same partition.")
    else:
        print("\n  FAIL  Two different 8-way partitions of the same 4,054 properties.")
        print("        Both are documented as 'the eight segments'. They are not the")
        print("        same object and cannot both be the segmentation.")


# ---------------------------------------------------------------- CHECK 3

def check_committed_csv_sufficiency() -> None:
    banner(3, "Can the committed CSV regenerate the archetypes?")

    cols = pd.read_csv(FEATURES_CSV, nrows=1).columns.tolist()
    needed = ["price_num", "size_num", "price_per_m2"]
    missing = [c for c in needed if c not in cols]

    print(f"  clustering_features.csv columns: {cols}")
    print(f"  needed to rebuild archetypes:    {needed}")

    if missing:
        print(f"\n  FAIL  Missing: {missing}")
        print("        The pool has no total price column, so archetype budget ranges")
        print("        (documented as 'IQR from data') cannot be derived from it, and")
        print("        neither can the valuation anchor. The committed pipeline cannot")
        print("        reproduce its own central config.")
    else:
        print("\n  PASS  CSV is sufficient.")


# ---------------------------------------------------------------- CHECK 4

def check_location_filters(pool: pd.DataFrame, archetypes: dict) -> None:
    banner(4, "Do archetype location filters match anything in the data?")

    print(f"  city values in pool: {pool['city'].value_counts().to_dict()}")
    print(f"  ('STERDAM' is a scraper truncation of AMSTERDAM, not a second city)")
    print(f"  distinct neighborhoods: {pool['neighborhood'].nunique()}")
    print(f"  example neighborhoods: {list(pool['neighborhood'].dropna().unique()[:6])}")

    referenced: set[str] = set()
    for a in archetypes["buyer_archetypes"].values():
        loc = a["location_preference"]
        referenced.update(loc.get("preferred", []))
        referenced.update(loc.get("excluded", []))

    neighborhoods = {str(n).lower() for n in pool["neighborhood"].dropna().unique()}
    matched = {t for t in referenced if t.lower() in neighborhoods}

    print(f"\n  distinct location tokens used by archetypes: {len(referenced)}")
    print(f"  {sorted(referenced)}")
    print(f"\n  tokens that match a real neighborhood value: {len(matched)} {sorted(matched)}")

    if not matched:
        print("\n  FAIL  No archetype location token appears anywhere in the data.")
        print("        Every location filter is unevaluable as written.")
        print("        Worse: 'amstelveen', 'diemen' and 'weesp' are PREFERRED locations")
        print("        for Space-Seekers Suburban (15.6% of buyers), and the pool is")
        print("        Amsterdam-only — that archetype can never match a property.")


# ---------------------------------------------------------------- CHECK 5

def check_buyer_coverage(pool: pd.DataFrame, archetypes: dict) -> pd.DataFrame:
    banner(5, "How many properties receive zero eligible buyers?")

    print("  Applying budget + size filters only (location filters are unevaluable,")
    print("  so this is the OPTIMISTIC bound — real coverage can only be worse).\n")

    eligible_any = pd.Series(False, index=pool.index)
    rows = []

    for key, a in archetypes["buyer_archetypes"].items():
        size = a["size_preference"]
        mask = (
            (pool["size_num"] >= size["min_sqm"])
            & (pool["size_num"] <= size["max_sqm"])
            & (pool["price_num"] >= a["budget_min"])
            & (pool["price_num"] <= a["budget_max"])
        )
        eligible_any |= mask
        rows.append({
            "archetype": key,
            "buyer_share_%": a["distribution_percent"],
            "eligible_properties": int(mask.sum()),
            "size_hard_filter": size["hard_filter"],
        })

    summary = pd.DataFrame(rows).sort_values("eligible_properties")
    print(summary.to_string(index=False))

    unreachable = int((~eligible_any).sum())
    pct = 100 * unreachable / len(pool)
    print(f"\n  Properties reachable by >=1 archetype: {len(pool) - unreachable:,}/{len(pool):,}")
    print(f"  Properties reachable by NO archetype:  {unreachable:,} ({pct:.1f}%)")

    if unreachable:
        print("\n  FAIL  Coverage claimed in the docs is 100%, but that is coverage of")
        print("        properties BY CLUSTER, not by buyer preference filter. Those are")
        print("        different quantities. A property no buyer can bid on never sells,")
        print("        which silently biases every aggregate the simulation reports.")

    return summary


# ---------------------------------------------------------------- CHECK 6

def check_tolerance_bands(archetypes: dict) -> None:
    banner(6, "Are tolerance bands a function of elasticity?")

    rows = [
        {
            "archetype": k,
            "elasticity": a["elasticity"],
            "tolerance_band_%": a["tolerance_band_percent"],
        }
        for k, a in archetypes["buyer_archetypes"].items()
    ]
    df = pd.DataFrame(rows).sort_values(["elasticity", "tolerance_band_%"])
    print(df.to_string(index=False))

    conflicts = df.groupby("elasticity")["tolerance_band_%"].nunique()
    ambiguous = conflicts[conflicts > 1]

    if len(ambiguous):
        print("\n  FAIL  The same elasticity maps to multiple different bands:")
        for e in ambiguous.index:
            bands = sorted(df.loc[df["elasticity"] == e, "tolerance_band_%"].tolist())
            print(f"        elasticity {e}: bands {bands}")
        print("\n        So the band is not derived from elasticity — it is not even a")
        print("        function of it. Monotonicity also fails: elasticity -0.25 gets a")
        print("        25% band while -0.35 gets up to 20%, but another -0.35 gets 12%.")
        print("        Separately, DESIGN.md section 4 says budget buyers bid 'within 5%'")
        print("        while the config gives them a 0% band.")
    else:
        print("\n  PASS  Band is a well-defined function of elasticity.")


# ---------------------------------------------------------------- CHECK 7

def check_error_profile(pool: pd.DataFrame) -> None:
    banner(7, "What error profile does the prediction audit supply?")

    if not RESIDUALS_CSV.exists():
        print(f"  INFO  Residual file not found at {RESIDUALS_CSV}")
        print("        Expected the prediction project as a sibling checkout.")
        return

    resid = pd.read_csv(RESIDUALS_CSV, index_col=0)
    print(f"  held-out listings: {len(resid):,}")
    print(f"  rel_error convention: (actual - predicted) / actual\n")

    by_band = resid.groupby("size_band").agg(
        n=("rel_error", "size"),
        mean_pct=("rel_error", lambda s: 100 * s.mean()),
        sd_pct=("rel_error", lambda s: 100 * s.std()),
        min_m2=("size_num", "min"),
        max_m2=("size_num", "max"),
    ).reindex(["XS", "S", "M", "L", "XL"])
    print(by_band.to_string())

    print(f"\n  Dispersion ranges {by_band['sd_pct'].min():.1f}% to "
          f"{by_band['sd_pct'].max():.1f}% across size bands — a factor of "
          f"{by_band['sd_pct'].max() / by_band['sd_pct'].min():.1f}.")
    print("  Using point valuations as if exact would discard all of this.")

    audit_max = resid["size_num"].max()
    pool_max = pool["size_num"].max()
    print(f"\n  INFO  Audit observed sizes up to {audit_max:.0f} m2; the pool contains")
    print(f"        properties up to {pool_max:.0f} m2. Error sampled for the largest")
    print("        properties extrapolates beyond measured range — a stated caveat,")
    print("        not a silent one.")


def main() -> None:
    print("=" * 78)
    print("WEEK 0 STATE AUDIT")
    print("=" * 78)

    df = load_cache()
    pool = cleaned_pool(df)
    archetypes = load_archetypes()

    check_archetype_reproducibility(pool, archetypes)
    check_two_segmentations(archetypes)
    check_committed_csv_sufficiency()
    check_location_filters(pool, archetypes)
    check_buyer_coverage(pool, archetypes)
    check_tolerance_bands(archetypes)
    check_error_profile(pool)

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
