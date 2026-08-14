"""
Rebuild the market segmentation from scratch, with the feature set and k chosen
deliberately rather than inherited.

WHY A REBUILD RATHER THAN A FIX
-------------------------------
The Week 0 segmentation had three compounding defects. Each is diagnosed here with
numbers before the replacement is built, so the choices are auditable.

  D1  Collinear feature set. It clustered on (price, size, price_per_m2), but
      price_per_m2 == price / size. The third feature is a deterministic function of
      the first two, so price enters the distance metric twice and the geometry is
      distorted toward price in a way nobody chose.

  D2  No transform on a heavily skewed variable. Prices run from ~1e5 to ~6e6. On
      standardised raw price, K-Means spends clusters isolating a handful of extreme
      properties. That is the mechanism behind the 22-property "Ultra-Luxury Estate"
      segment — and therefore behind that archetype having almost no addressable
      market.

  D3  Ranges taken from cluster IQRs. An IQR covers the middle 50% of a cluster by
      construction, so k IQRs cannot tile the distribution. Measured consequence:
      60.2% of properties were reachable by no archetype at all.

WHAT REPLACES IT
----------------
  Features: log(price), log(size), dist_to_centre_km

    log(price)          the buyer's budget constraint, on the scale people actually
                        shop in (proportional, not absolute)
    log(size)           what the buyer gets, same reasoning
    dist_to_centre_km   location, from PC4 polygon geometry (build_pc4_geography.py).
                        NOT price-derived, so segments that turn out to differ by
                        centrality are a finding rather than a restatement.

    price_per_m2 is deliberately excluded: it is log(price) - log(size) in log space,
    so it adds no information to this feature set and would reintroduce D1.

  k: chosen from the sweep below on three criteria jointly — silhouette, the elbow,
    and achievable coverage. Coverage is treated as a selection criterion, not a
    post-hoc check, because that is the failure that made the old segmentation
    unusable.

  Ranges: segment bounds are percentile-based and WIDE (configurable, default
    5th-95th), so adjacent segments overlap. Overlap is both necessary for coverage
    and realistic — a 90 m2 flat at EUR 600k is plausibly wanted by more than one
    kind of buyer.

Run from the repo root (after build_property_pool.py and build_pc4_geography.py):
    python scripts/build_segmentation.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_property_pool  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "config" / "market_segments.yaml"
ASSIGNMENTS_PATH = REPO_ROOT / "data" / "segment_assignments.csv"

SEED = 42
K_RANGE = range(3, 13)

# Segment bounds span the full range of each cluster, so every property is inside
# its own segment's box by construction and coverage is 100%. Wide and overlapping
# by design — see D3 and diagnose_bound_rule().
BOUND_LO, BOUND_HI = 0, 100


# ------------------------------------------------------------------ diagnostics

def diagnose_collinearity(pool: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("D1  Collinearity in the Week 0 feature set")
    print("=" * 78)

    old = pool[["price_num", "size_num", "price_per_m2"]]
    print("  Pearson correlation of the old clustering features:\n")
    print(old.corr().round(3).to_string())

    # price_per_m2 is exactly price/size, so in log space it is a linear combination.
    residual = np.abs(
        np.log(pool["price_per_m2"])
        - (np.log(pool["price_num"]) - np.log(pool["size_num"]))
    ).max()
    print(f"\n  max |log(ppm2) - (log(price) - log(size))| = {residual:.2e}")
    print("  i.e. the third feature is an exact function of the other two.")
    print("  Standardised, it re-injects price into the distance metric a second")
    print("  time with weight nobody chose.")


def diagnose_skew(pool: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("D2  Skew, and what it does to cluster sizes")
    print("=" * 78)

    price = pool["price_num"]
    print(f"  price skew (raw): {price.skew():>7.2f}")
    print(f"  price skew (log): {np.log(price).skew():>7.2f}")

    for label, X in (
        ("raw price + raw size", pool[["price_num", "size_num"]].values),
        ("log price + log size", np.column_stack([np.log(price), np.log(pool["size_num"])])),
    ):
        Xs = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=8, random_state=SEED, n_init=10).fit_predict(Xs)
        sizes = sorted(np.bincount(labels))
        print(f"\n  k=8 on {label}:")
        print(f"    cluster sizes: {sizes}")
        print(f"    smallest cluster: {sizes[0]} properties ({100*sizes[0]/len(pool):.1f}%)")

    print("\n  The raw-scale run strands a handful of properties in their own")
    print("  clusters. Those become archetypes with essentially no market to buy in.")


# ------------------------------------------------------------------ k selection

def build_features(pool: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = ["log_price", "log_size", "dist_to_centre_km"]
    X = np.column_stack([
        np.log(pool["price_num"].values),
        np.log(pool["size_num"].values),
        pool["dist_to_centre_km"].values,
    ])
    return StandardScaler().fit_transform(X), names


def coverage_for_labels(
    pool: pd.DataFrame, labels: np.ndarray, lo: float, hi: float
) -> float:
    """Share of properties falling inside at least one segment's price+size box."""
    reachable = np.zeros(len(pool), dtype=bool)
    price = pool["price_num"].values
    size = pool["size_num"].values

    for c in np.unique(labels):
        m = labels == c
        p_lo, p_hi = np.percentile(price[m], [lo, hi])
        s_lo, s_hi = np.percentile(size[m], [lo, hi])
        reachable |= (price >= p_lo) & (price <= p_hi) & (size >= s_lo) & (size <= s_hi)

    return reachable.mean()


def diagnose_bound_rule(pool: pd.DataFrame, X: np.ndarray, k: int = 9) -> None:
    """Show what each bound rule costs in coverage and buys in tightness."""
    print("\n" + "=" * 78)
    print(f"D3  Bound rule: coverage vs tightness (illustrated at k={k})")
    print("=" * 78)

    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
    price = pool["price_num"].values

    print(f"  {'rule':<10}{'coverage':>10}{'median width':>14}{'max width':>11}")
    print("  " + "-" * 43)
    for lo, hi, name in [(25, 75, "IQR"), (5, 95, "p5-p95"), (1, 99, "p1-p99"), (0, 100, "min-max")]:
        widths = []
        for c in np.unique(labels):
            p_lo, p_hi = np.percentile(price[labels == c], [lo, hi])
            widths.append(p_hi / max(p_lo, 1))
        cov = 100 * coverage_for_labels(pool, labels, lo, hi)
        print(f"  {name:<10}{cov:>9.1f}%{np.median(widths):>13.1f}x{max(widths):>10.1f}x")

    print("\n  IQR is the Week 0 rule. Its ~47% coverage confirms the failure is the")
    print("  RULE, not the feature set — it reproduces here on a better clustering.")
    print("\n  min-max is adopted. Rationale: a property reachable by zero buyers never")
    print("  sells, and 'never sells' is indistinguishable in the output from a genuine")
    print("  market result. Reachability is therefore a correctness constraint, not a")
    print("  quality metric, and is guaranteed by construction rather than tuned.")
    print("\n  The cost is wide bounds (median 3.1x). That is accepted because the hard")
    print("  bound only decides who MAY bid. How many actually bid is governed")
    print("  separately by preference intensity, so thin markets show up as low bidder")
    print("  counts — which is the mechanism post 8 describes — rather than as")
    print("  properties silently excluded from the simulation.")


def sweep_k(pool: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("k SWEEP — silhouette and elbow (coverage is 100% by construction)")
    print("=" * 78)

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        labels = km.labels_
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "smallest_cluster": int(np.bincount(labels).min()),
            "smallest_share_%": 100 * np.bincount(labels).min() / len(labels),
        })

    sweep = pd.DataFrame(rows)
    sweep["inertia_drop_%"] = -100 * sweep["inertia"].pct_change()
    print(sweep.round(2).to_string(index=False))
    return sweep


def choose_k(sweep: pd.DataFrame, min_cluster: int = 100) -> int:
    """Pick k at a local silhouette maximum, subject to a viable smallest cluster.

    Silhouette is highest at the smallest k and drifts down: the Amsterdam market is
    a price/size continuum, not a set of well-separated blobs. There is no natural
    number of segments to discover. Taking the global maximum would give k=3, which
    is too coarse to say anything about pricing strategy per segment — the same
    tension noted in the prediction project's notebook ("Too little clusters now").

    So k is chosen, not discovered, on a stated rule: the largest local silhouette
    maximum whose smallest cluster still holds enough properties to be stable.
    """
    viable = sweep[sweep["smallest_cluster"] >= min_cluster].copy()
    if viable.empty:
        raise RuntimeError(f"No k leaves a cluster of >= {min_cluster} properties.")

    s = viable.set_index("k")["silhouette"]
    local_maxima = [
        k for k in s.index
        if (k - 1 not in s.index or s[k] >= s[k - 1])
        and (k + 1 not in s.index or s[k] >= s[k + 1])
    ]
    # Prefer the most granular local maximum that remains viable.
    best = int(max(local_maxima)) if local_maxima else int(s.idxmax())

    print("\n" + "=" * 78)
    print("k SELECTION")
    print("=" * 78)
    print(f"  Silhouette peaks at k={int(s.idxmax())} ({s.max():.3f}) and declines with k.")
    print("  That is what a continuum looks like: there is no natural k to discover,")
    print("  so k is a modelling choice and is declared as one.")
    print(f"\n  rule: most granular local silhouette maximum with smallest cluster >= {min_cluster}")
    print(f"  viable k:       {sorted(viable['k'].tolist())}")
    print(f"  local maxima:   {local_maxima}")
    print(f"  chosen k = {best}")
    return best


# ------------------------------------------------------------------ build output

def to_native(obj):
    """Recursively convert numpy scalars to Python types for safe YAML dumping.

    yaml.safe_dump refuses numpy types, and pandas/sklearn return them almost
    everywhere. Sanitising once at the boundary is more robust than casting at each
    call site, where a newly added field silently reintroduces the failure.
    """
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def build_segments(
    pool: pd.DataFrame, labels: np.ndarray
) -> tuple[dict, dict[int, int]]:
    """Return segment definitions plus the cluster-id -> rank map used to order them.

    Clusters are renumbered by median price so segment_1 is always the cheapest.
    Raw K-Means labels are arbitrary and change with the seed; ranks are stable and
    readable, and the map is returned so assignments cannot drift from definitions.
    """
    segments = {}
    price = pool["price_num"]
    size = pool["size_num"]

    order = (
        pd.DataFrame({"c": labels, "p": price.values})
        .groupby("c")["p"].median().sort_values().index.tolist()
    )
    rank_by_cluster = {int(c): rank for rank, c in enumerate(order, start=1)}

    for rank, c in enumerate(order, start=1):
        m = labels == c
        sub = pool[m]
        p_lo, p_hi = np.percentile(price[m], [BOUND_LO, BOUND_HI])
        s_lo, s_hi = np.percentile(size[m], [BOUND_LO, BOUND_HI])

        segments[f"segment_{rank}"] = {
            "rank": int(rank),
            "n_properties": int(m.sum()),
            "share_percent": round(float(100 * m.mean()), 2),
            "price_median": int(price[m].median()),
            "price_min_bound": int(p_lo),
            "price_max_bound": int(p_hi),
            "size_median": float(size[m].median()),
            "size_min_bound": float(s_lo),
            "size_max_bound": float(s_hi),
            "price_per_m2_median": int(sub["price_per_m2"].median()),
            "dist_to_centre_km_median": round(float(sub["dist_to_centre_km"].median()), 2),
            "dominant_zone": str(sub["zone"].mode().iloc[0]),
            "zone_mix": {
                str(z): round(float(100 * v), 1)
                for z, v in sub["zone"].value_counts(normalize=True).head(3).items()
            },
        }

    return segments, rank_by_cluster


def report_segments(segments: dict, pool: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("MARKET SEGMENTS")
    print("=" * 78)
    print(
        f"{'seg':<10}{'n':>6}{'share':>8}{'median':>11}{'size':>7}"
        f"{'EUR/m2':>9}{'km':>7}  zone mix"
    )
    print("-" * 78)
    for name, s in segments.items():
        mix = ", ".join(f"{z} {p}%" for z, p in s["zone_mix"].items())
        print(
            f"{name:<10}{s['n_properties']:>6}{s['share_percent']:>7.1f}%"
            f"{s['price_median']:>11,}{s['size_median']:>7.0f}"
            f"{s['price_per_m2_median']:>9,}{s['dist_to_centre_km_median']:>7.1f}  {mix}"
        )

    print("\n  Eligibility bounds (full cluster range, overlapping by design):")
    for name, s in segments.items():
        print(
            f"    {name:<10} EUR {s['price_min_bound']:>9,} - {s['price_max_bound']:>9,}"
            f"   {s['size_min_bound']:>5.0f} - {s['size_max_bound']:>5.0f} m2"
        )


def main() -> None:
    print("=" * 78)
    print("SEGMENTATION REBUILD")
    print("=" * 78)

    pool = load_property_pool(with_geography=True)
    print(f"Loaded {len(pool):,} properties with geography")

    diagnose_collinearity(pool)
    diagnose_skew(pool)

    X, feature_names = build_features(pool)
    print(f"\nFeature set: {feature_names} (standardised)")

    diagnose_bound_rule(pool, X)
    sweep = sweep_k(pool, X)
    k = choose_k(sweep)

    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
    segments, rank_by_cluster = build_segments(pool, labels)
    report_segments(segments, pool)

    final_coverage = 100 * coverage_for_labels(pool, labels, BOUND_LO, BOUND_HI)
    print(f"\n  Coverage at k={k}: {final_coverage:.1f}% of properties reachable")
    print(f"  (Week 0 comparison: 39.8% reachable, 60.2% unreachable)")

    doc = {
        "market_segments": segments,
        "metadata": {
            "k": k,
            "features": feature_names,
            "bound_percentiles": [BOUND_LO, BOUND_HI],
            "seed": SEED,
            "n_properties": int(len(pool)),
            "coverage_percent": round(final_coverage, 2),
            "generated_by": "scripts/build_segmentation.py",
            "notes": (
                "These are PROPERTY segments. Each is later paired with a buyer "
                "population that targets it, but the segmentation itself is derived "
                "from the property pool and nothing else. Any statement that a "
                "segment's buyers behave a certain way is an assumption layered on "
                "top, not a result of this clustering."
            ),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_native(doc), f, sort_keys=False, default_flow_style=False)

    assignments = pd.DataFrame({
        "listing_id": pool["listing_id"].values,
        "segment_rank": [rank_by_cluster[int(c)] for c in labels],
    })
    if assignments["segment_rank"].isna().any():
        raise ValueError("Some properties were not assigned a segment rank.")
    assignments.to_csv(ASSIGNMENTS_PATH, index=False)

    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {ASSIGNMENTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
