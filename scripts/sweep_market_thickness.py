"""
Does the elasticity result survive a different market thickness?

`arrivals_per_day` controls how many buyers reach the market. It is ASSUMED, and at
the value used so far (260) the bottom four price bins clear at 95-99.7% within 90
days. That is implausibly liquid, and it threatens both headline results: when almost
everything sells regardless of price, there is little pricing tension left and the
optimal multiple is driven mostly by the calibrated WTP dispersion rather than by a
real trade-off between price and probability of sale.

WHY THIS CANNOT SIMPLY BE CALIBRATED AWAY
-----------------------------------------
The obvious fix would be to pick the arrivals rate that reproduces observed
time-on-market. That data does not exist here: the listing date was never scraped, so
neither time-on-market nor sale/withdrawal outcomes are available. Arrivals is
therefore UNIDENTIFIED from this dataset, and the only honest treatment is to sweep it
and report how much the conclusions depend on it.

Choosing an arrivals rate because it produced sensible-looking clearing would be
calibrating the market to the answer — the same trap flagged for equity.

WHAT THIS DECIDES
-----------------
If the elasticity sensitivity is roughly stable across a wide thickness range, the
33% headline stands and thickness is a nuisance parameter. If it moves a lot, then
the headline is partly an artifact of an assumed arrival rate and must be reported as
conditional on it.

Run from the repo root:
    python scripts/sweep_market_thickness.py
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
RESULTS_PATH = REPO_ROOT / "data" / "sweep_market_thickness.csv"

ARRIVALS_VALUES = [20, 40, 80, 160, 260, 500, 1000]
ELASTICITY_VALUES = [-1.05, -0.85, -0.65, -0.45]
EQUITY_FIXED = 1.0  # the CBS-derived central case
N_BINS = 5


def main() -> None:
    print("=" * 78)
    print("MARKET THICKNESS SWEEP")
    print("=" * 78)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s = cfg["sweep"]
    m = cfg["market"]
    rng = np.random.default_rng(s["seed"])

    pool_full = load_property_pool()
    pool = pool_full.sample(
        n=min(s["n_property_sample"], len(pool_full)), random_state=s["seed"]
    ).reset_index(drop=True)

    err = ValuationError(pool_full, overlap_only=cfg["valuation"]["use_overlap_only"])
    valued = make_valuations(pool, err, rng)
    v_true = valued["v_true"].to_numpy()
    v_est = valued["v_est"].to_numpy()
    price_bin = pd.qcut(v_true, N_BINS, labels=[f"Q{i}" for i in range(1, N_BINS + 1)])

    # Buyers and the ability index depend only on the buyer population and the pool,
    # not on the arrival rate, so they are built once.
    buyers = generate_buyers(cfg, pool_full, rng, equity_multiplier=EQUITY_FIXED)
    ability = AbilityIndex(buyers, pool, cfg)
    print(f"\nPool sample {len(pool)}, equity fixed at {EQUITY_FIXED}")

    rows = []
    print(f"\n{'arrivals':>9}{'elast':>8}{'achieved':>10}{'disp':>8}"
          f"{'med_mult':>10}{'p_sale':>9}{'Q5_psale':>10}{'days':>7}")
    print("-" * 78)

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
            res["price_bin"] = price_bin.astype(str)
            q5 = res[res["price_bin"] == "Q5"]

            row = {
                "arrivals_per_day": arrivals,
                "target_elasticity": target_e,
                "achieved_elasticity": achieved,
                "calibration_ok": abs(achieved - target_e) < 0.02,
                "wtp_dispersion": dispersion,
                "median_optimal_multiple": res["optimal_multiple"].median(),
                "mean_p_sale": res["p_sale"].mean(),
                "q5_p_sale": q5["p_sale"].mean(),
                "mean_days": res["expected_days"].mean(),
                "share_at_grid_bound": res["at_grid_bound"].mean(),
            }
            rows.append(row)
            flag = "" if row["calibration_ok"] else "  <- CALIB MISSED"
            if row["share_at_grid_bound"] > 0.05:
                flag += f"  <- {100*row['share_at_grid_bound']:.0f}% bound"
            print(f"{arrivals:>9}{target_e:>8.2f}{achieved:>10.3f}{dispersion:>8.3f}"
                  f"{row['median_optimal_multiple']:>10.3f}{row['mean_p_sale']:>9.3f}"
                  f"{row['q5_p_sale']:>10.3f}{row['mean_days']:>7.1f}{flag}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    report(results)
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def report(results: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("CLEARING RATE BY ARRIVAL RATE  (elasticity -0.65)")
    print("=" * 78)
    base = results[results["target_elasticity"] == -0.65]
    print(f"  {'arrivals':>9}{'overall p_sale':>16}{'Q5 p_sale':>12}{'mean days':>12}")
    print("  " + "-" * 47)
    for _, r in base.iterrows():
        print(f"  {int(r['arrivals_per_day']):>9}{r['mean_p_sale']:>16.3f}"
              f"{r['q5_p_sale']:>12.3f}{r['mean_days']:>12.1f}")

    print("\n" + "=" * 78)
    print("DOES THE ELASTICITY RESULT SURVIVE?")
    print("=" * 78)

    ok = results[results["calibration_ok"] & (results["share_at_grid_bound"] <= 0.05)]
    piv = ok.pivot(index="arrivals_per_day", columns="target_elasticity",
                   values="median_optimal_multiple")
    print("\nMedian optimal multiple (converged cells only):\n")
    print(piv.round(3).to_string())

    sens = ((piv.max(axis=1) - piv.min(axis=1)) / piv.mean(axis=1) * 100).dropna()
    print("\n  Elasticity sensitivity at each arrival rate:")
    for a, v in sens.items():
        n_cells = piv.loc[a].notna().sum()
        note = "" if n_cells == piv.shape[1] else f"  (only {n_cells}/{piv.shape[1]} cells)"
        print(f"    arrivals {int(a):>5}: {v:>6.2f}%{note}")

    full = sens[[piv.loc[a].notna().sum() == piv.shape[1] for a in sens.index]]
    if len(full) >= 2:
        print(f"\n  Across arrival rates with complete data: "
              f"{full.min():.1f}% to {full.max():.1f}%")
        if full.max() / max(full.min(), 1e-9) > 2:
            print("\n  NOT STABLE. The elasticity sensitivity itself depends strongly on")
            print("  an assumed arrival rate that this data cannot identify. The headline")
            print("  must be reported as conditional on market thickness, and thickness")
            print("  belongs in the sweep permanently rather than fixed at one value.")
        else:
            print("\n  STABLE. The elasticity result holds across a wide thickness range,")
            print("  so arrivals_per_day is a nuisance parameter rather than a driver.")
    else:
        print("\n  Too few arrival rates yield complete data to judge stability.")


if __name__ == "__main__":
    main()
