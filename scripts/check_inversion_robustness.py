"""
Does the pricing inversion survive market thickness?

THE CLAIM UNDER TEST
--------------------
At CBS-derived equity, optimal asking price is ABOVE the seller's estimated value for
the bottom four price quintiles (1.06-1.09x) and BELOW it for the top quintile
(0.92x). The proposed mechanism: at the bottom the binding constraint is willingness,
so able buyers are plentiful and price can be pushed up; at the top the binding
constraint is budget, the buyer pool is thin, and the price-versus-probability
trade-off tips the other way.

WHY IT NEEDS CHECKING
---------------------
That result was measured at one arrivals rate (260). The thickness sweep already
showed that arrivals modulates the elasticity sensitivity by a factor of nearly five,
and arrivals is UNIDENTIFIED from this data — the listing date was never scraped, so
there is no time-on-market to calibrate against. A result quoted at a single value of
an unidentified parameter is not yet a result.

The test is specifically of the SIGN, not the magnitude. If Q5 prices below 1.0 and
Q1-Q4 above it across the plausible thickness range, the inversion is a property of
the model rather than of one arbitrary setting. Magnitude is expected to move.

Run from the repo root:
    python scripts/check_inversion_robustness.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_property_pool  # noqa: E402
from src.simulation.demand import generate_buyers  # noqa: E402
from src.simulation.market import (  # noqa: E402
    AbilityIndex,
    MarketConfig,
    calibrate_dispersion,
    optimize_prices,
)
from src.simulation.valuation import ValuationError, make_valuations  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "simulation.yaml"
RESULTS_PATH = REPO_ROOT / "data" / "inversion_robustness.csv"

ARRIVALS_VALUES = [80, 160, 260, 500, 1000]
ELASTICITY_VALUES = [-0.85, -0.65, -0.45]
EQUITY_MULTIPLIER = 1.0  # the CBS-derived central case
N_BINS = 5


def main() -> None:
    print("=" * 78)
    print("IS THE PRICING INVERSION ROBUST TO MARKET THICKNESS?")
    print("=" * 78)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s, m = cfg["sweep"], cfg["market"]
    rng = np.random.default_rng(s["seed"])

    pool_full = load_property_pool()
    pool = pool_full.sample(
        n=min(s["n_property_sample"], len(pool_full)), random_state=s["seed"]
    ).reset_index(drop=True)

    err = ValuationError(pool_full, overlap_only=cfg["valuation"]["use_overlap_only"])
    valued = make_valuations(pool, err, rng)
    v_true = valued["v_true"].to_numpy()
    v_est = valued["v_est"].to_numpy()
    price_bin = pd.qcut(
        v_true, N_BINS, labels=[f"Q{i}" for i in range(1, N_BINS + 1)]
    ).astype(str)

    buyers = generate_buyers(cfg, pool_full, rng, equity_multiplier=EQUITY_MULTIPLIER)
    ability = AbilityIndex(buyers, pool, cfg)
    print(f"\nPool sample {len(pool)}, equity multiplier {EQUITY_MULTIPLIER} (derived)")

    rows = []
    for arrivals in ARRIVALS_VALUES:
        mc = MarketConfig(
            arrivals_per_day=arrivals,
            horizon_days=m["horizon_days"],
            discount_rate_daily=m["discount_rate_annual"] / 365.0,
            carrying_cost_daily=m["carrying_cost_monthly_eur"] * 12.0 / 365.0,
            pool_decay_daily=m["buyer_pool_decay_daily"],
            n_properties_total=len(pool),
        )
        for target_e in ELASTICITY_VALUES:
            dispersion, achieved = calibrate_dispersion(
                v_true, v_true, ability, mc, target_e
            )
            res = optimize_prices(v_est, v_true, ability, dispersion, mc, cfg)
            res["price_bin"] = price_bin
            for label, g in res.groupby("price_bin"):
                rows.append({
                    "arrivals_per_day": arrivals,
                    "target_elasticity": target_e,
                    "calibration_ok": abs(achieved - target_e) < 0.02,
                    "price_bin": label,
                    "median_optimal_multiple": g["optimal_multiple"].median(),
                    "mean_p_sale": g["p_sale"].mean(),
                    "share_at_grid_bound": g["at_grid_bound"].mean(),
                })
        print(f"  arrivals {arrivals} done")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    report(results)
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def report(results: pd.DataFrame) -> None:
    for e in sorted(results["target_elasticity"].unique(), reverse=True):
        sub = results[results["target_elasticity"] == e]
        piv = sub.pivot(index="price_bin", columns="arrivals_per_day",
                        values="median_optimal_multiple")
        bound = sub.pivot(index="price_bin", columns="arrivals_per_day",
                          values="share_at_grid_bound")
        calib = sub["calibration_ok"].all()

        print("\n" + "=" * 78)
        print(f"OPTIMAL MULTIPLE BY BIN AND ARRIVALS   (elasticity {e:.2f})"
              + ("" if calib else "   [SOME CELLS UNCALIBRATED]"))
        print("  '*' = >5% of properties on the grid edge (optimum is a bound)")
        print("=" * 78)
        print("        " + "".join(f"{c:>9}" for c in piv.columns))
        for b in piv.index:
            cells = "".join(
                f"{piv.loc[b, c]:>8.2f}" + ("*" if bound.loc[b, c] > 0.05 else " ")
                for c in piv.columns
            )
            print(f"  {b:<6}{cells}")

    print("\n" + "=" * 78)
    print("VERDICT — is the SIGN stable?")
    print("=" * 78)

    ok = results[results["calibration_ok"]]
    low = ok[ok["price_bin"].isin(["Q1", "Q2", "Q3", "Q4"])]
    top = ok[ok["price_bin"] == "Q5"]

    low_above = (low["median_optimal_multiple"] > 1.0).mean()
    top_below = (top["median_optimal_multiple"] < 1.0).mean()

    print(f"\n  Q1-Q4 priced ABOVE the estimate: {100*low_above:.0f}% of cells "
          f"({len(low)} cells)")
    print(f"  Q5    priced BELOW the estimate: {100*top_below:.0f}% of cells "
          f"({len(top)} cells)")
    print(f"\n  Q5 optimal multiple range: {top['median_optimal_multiple'].min():.2f} "
          f"to {top['median_optimal_multiple'].max():.2f}")
    print(f"  Q5 clearing range:         {top['mean_p_sale'].min():.3f} "
          f"to {top['mean_p_sale'].max():.3f}")

    if low_above > 0.9 and top_below > 0.9:
        print("\n  INVERSION HOLDS across the thickness range. The sign is a property")
        print("  of the model, not of one arbitrary arrivals rate. Magnitude does move,")
        print("  so quote the direction as the finding and the magnitude as a range.")
    elif top_below > 0.9:
        print("\n  PARTIAL. Q5 prices below estimate throughout, but the bottom bins do")
        print("  not consistently price above it. State the top-bin result only.")
    else:
        print("\n  INVERSION DOES NOT HOLD. It was an artifact of the arrivals rate")
        print("  used, and must not be reported as a finding.")


if __name__ == "__main__":
    main()
