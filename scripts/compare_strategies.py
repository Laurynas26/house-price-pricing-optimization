"""
Revenue-max vs speed-max vs balanced — and whether the comparison is even meaningful.

THE DEGENERACY THIS TESTS FOR
-----------------------------
If waiting is free, revenue-max trivially dominates: the seller holds out for the
best price at no cost, so "time to sale" has no price and the comparison has a
predetermined winner. The original design had exactly this problem — valuation did
not decay, there was no holding cost, and the strategy comparison was therefore
decided before it was run.

Three time-cost channels are now available, and this script runs the comparison with
each switched on and off so the degeneracy is demonstrated rather than assumed:

  discount rate      opportunity cost of capital. ~1.5%/quarter at 6%/yr.
  carrying cost      VvE service charge, per property. Real but small: 90 days of
                     the median charge is ~0.09% of the median asking price, roughly
                     seventeen times smaller than the discount term. Included for
                     correctness, not because it drives anything.
  buyer-pool decay   stale listings attract fewer viewers. UNIDENTIFIED from this
                     data — the listing date was never scraped — so it is swept
                     rather than fitted.

WHAT A MEANINGFUL RESULT LOOKS LIKE
-----------------------------------
The strategies should choose DIFFERENT prices, and each should win on its own metric.
If revenue-max and balanced pick the same price, time is not priced strongly enough
for the comparison to say anything.

Run from the repo root:
    python scripts/compare_strategies.py
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
    STRATEGIES,
    AbilityIndex,
    MarketConfig,
    calibrate_dispersion,
    make_market_config,
    optimize_prices,
)
from src.simulation.valuation import ValuationError, make_valuations  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "simulation.yaml"
RESULTS_PATH = REPO_ROOT / "data" / "strategy_comparison.csv"

ELASTICITY = -0.65
EQUITY_MULTIPLIER = 1.0

# (label, annual discount rate, use per-property VvE, daily buyer-pool decay)
# Decay of 0.023/day halves the arrival rate after 30 days on market.
#
# NOTE on what each channel actually does. Decay is a DEMAND-side effect: it changes
# the arrival rate and therefore what price is optimal for every seller alike. Only
# the discount rate and the carrying cost enter the objective differently across
# strategies, so only they can separate revenue-max from balanced.
TIME_COST_SCENARIOS = [
    ("none", 0.0, False, 0.0),
    ("discount only", 0.06, False, 0.0),
    ("discount + VvE", 0.06, True, 0.0),
    ("discount + VvE + decay", 0.06, True, 0.023),
    ("heavy decay", 0.06, True, 0.046),
]

# Market thickness matters more than any time-cost switch here. At 260 arrivals a
# property sells in about 24 days, so a 6%/yr discount costs roughly 0.4% — smaller
# than one step of the price grid, which makes the comparison unresolvable by
# construction. Slow markets are where a price on time can actually bite.
ARRIVALS_VALUES = [40, 80, 260]

# Seller's outside option, as a fraction of their own estimated value. ASSUMED.
RESERVE_FRACTION = 0.95

# Finer than the shared config grid: a 2% step cannot resolve a sub-1% time penalty.
FINE_GRID = {"price_multiple_min": 0.80, "price_multiple_max": 1.40,
             "price_multiple_steps": 121}  # 0.5% steps


def main() -> None:
    print("=" * 78)
    print("STRATEGY COMPARISON UNDER DIFFERENT TIME COSTS")
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

    buyers = generate_buyers(cfg, pool_full, rng, equity_multiplier=EQUITY_MULTIPLIER)
    ability = AbilityIndex(buyers, pool, cfg)

    vve_daily = pool["contribution_vve_num"].fillna(0.0).to_numpy() * 12.0 / 365.0
    print(f"\nPool sample {len(pool)}, elasticity {ELASTICITY}, "
          f"equity multiplier {EQUITY_MULTIPLIER} (derived)")
    print(f"Median VvE carrying cost: EUR {np.median(vve_daily) * 90:,.0f} over 90 days")

    cfg_fine = {**cfg, "optimizer": FINE_GRID}

    # The seller's outside option: withdraw rather than accept less than this. Without
    # it, speed_max is unbounded below — the fastest sale is always a giveaway — and
    # it pinned to the grid floor in every cell of the previous run.
    #
    # ASSUMED, and it should be swept before any strategy claim is published. Set as a
    # fraction of the seller's own estimate, because that is what a seller anchors on;
    # they do not observe v_true.
    reserve = v_est * RESERVE_FRACTION
    print(f"Reservation price: {RESERVE_FRACTION:.2f} x the seller's own estimate")

    rows = []
    for arrivals in ARRIVALS_VALUES:
        for label, disc, use_vve, decay in TIME_COST_SCENARIOS:
            mc = MarketConfig(
                arrivals_per_day=arrivals,
                horizon_days=m["horizon_days"],
                discount_rate_daily=disc / 365.0,
                carrying_cost_daily=vve_daily if use_vve else 0.0,
                pool_decay_daily=decay,
                n_properties_total=len(pool),
            )
            dispersion, achieved = calibrate_dispersion(
                v_true, v_true, ability, mc, ELASTICITY
            )

            for strategy in STRATEGIES:
                res = optimize_prices(
                    v_est, v_true, ability, dispersion, mc, cfg_fine,
                    strategy=strategy, reservation_price=reserve,
                )
                rows.append({
                    "arrivals_per_day": arrivals,
                    "time_costs": label,
                    "strategy": strategy,
                    "calibration_ok": abs(achieved - ELASTICITY) < 0.02,
                    "median_multiple": res["optimal_multiple"].median(),
                    "mean_p_sale": res["p_sale"].mean(),
                    "mean_days": res["expected_days"].mean(),
                    "share_at_grid_bound": res["at_grid_bound"].mean(),
                })
        print(f"  arrivals {arrivals} done")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    report(results)
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def report(results: pd.DataFrame) -> None:
    order = [s[0] for s in TIME_COST_SCENARIOS]

    for arrivals in ARRIVALS_VALUES:
        sub = results[results["arrivals_per_day"] == arrivals]
        piv = sub.pivot(index="time_costs", columns="strategy",
                        values="median_multiple").reindex(order)[list(STRATEGIES)]
        days = sub.pivot(index="time_costs", columns="strategy",
                         values="mean_days").reindex(order)[list(STRATEGIES)]

        print("\n" + "=" * 78)
        print(f"ARRIVALS {arrivals}  —  price multiple (days on market)")
        print("=" * 78)
        print(f"  {'time costs':<24}" + "".join(f"{s:>20}" for s in STRATEGIES))
        for label in piv.index:
            cells = "".join(
                f"{piv.loc[label, s]:>13.3f} ({days.loc[label, s]:>4.0f})"
                for s in STRATEGIES
            )
            print(f"  {label:<24}{cells}")

    print("\n" + "=" * 78)
    print("IS THE COMPARISON MEANINGFUL?")
    print("=" * 78)
    print("\n  revenue_max minus balanced, in price multiple. A zero gap means time")
    print("  is not priced strongly enough to change the decision.\n")
    print(f"  {'arrivals':>9}  {'time costs':<24}{'gap':>8}{'days saved':>12}   verdict")
    print("  " + "-" * 70)

    any_meaningful = False
    for arrivals in ARRIVALS_VALUES:
        sub = results[results["arrivals_per_day"] == arrivals]
        piv = sub.pivot(index="time_costs", columns="strategy",
                        values="median_multiple").reindex(order)
        days = sub.pivot(index="time_costs", columns="strategy",
                         values="mean_days").reindex(order)
        for label in order:
            gap = piv.loc[label, "revenue_max"] - piv.loc[label, "balanced"]
            saved = days.loc[label, "revenue_max"] - days.loc[label, "balanced"]
            meaningful = abs(gap) >= 0.005
            any_meaningful |= meaningful
            verdict = "meaningful" if meaningful else "degenerate"
            print(f"  {arrivals:>9}  {label:<24}{gap:>+8.3f}{saved:>12.1f}   {verdict}")

    # revenue_max vs balanced answers "is time priced". revenue_max vs speed_max
    # answers "what does speed cost", which is the question a seller actually asks —
    # and unlike the first, it has a non-degenerate answer.
    print("\n" + "=" * 78)
    print("WHAT DOES SPEED COST?  (revenue_max vs speed_max)")
    print("=" * 78)
    print(f"\n  {'arrivals':>9}  {'time costs':<24}{'price given up':>16}{'days saved':>12}")
    print("  " + "-" * 63)
    for arrivals in ARRIVALS_VALUES:
        sub = results[results["arrivals_per_day"] == arrivals]
        piv = sub.pivot(index="time_costs", columns="strategy",
                        values="median_multiple").reindex(order)
        days = sub.pivot(index="time_costs", columns="strategy",
                         values="mean_days").reindex(order)
        for label in order:
            rev, spd = piv.loc[label, "revenue_max"], piv.loc[label, "speed_max"]
            pct = 100 * (rev - spd) / rev
            saved = days.loc[label, "revenue_max"] - days.loc[label, "speed_max"]
            note = "  <- revenue price is BELOW the reserve" if rev < spd else ""
            print(f"  {arrivals:>9}  {label:<24}{pct:>15.1f}%{saved:>12.1f}{note}")

    print("\n  Where the revenue-maximising price falls below the reservation price,")
    print("  the seller's best move is to withdraw rather than transact — the market")
    print("  will not pay their floor. That is a genuine outcome of the model, not an")
    print("  error, and it only appears in thin markets with a decaying buyer pool.")

    print("\n" + "=" * 78)
    if any_meaningful:
        print("  The comparison separates the strategies in at least one regime. Report")
        print("  it as conditional on that regime rather than as a general result.")
    else:
        print("  DEGENERATE EVERYWHERE. Revenue-max IS the optimal strategy under every")
        print("  time cost tested, and that is a finding rather than a bug: when a")
        print("  property sells in a few weeks, waiting is close to free and there is")
        print("  nothing to trade away. The 'revenue vs speed' framing only has content")
        print("  in a slow market, and Amsterdam at these arrival rates is not one.")
        print("\n  Do NOT fix this by inflating the discount rate until the strategies")
        print("  separate. That would be choosing a parameter to manufacture a result.")
    print("=" * 78)


if __name__ == "__main__":
    main()
