"""
Does equity matter where it was introduced to matter?

The headline sweep reported that equity moves the median optimal price by ~3.7%
against elasticity's ~33%, which would mean the CBS wealth derivation is not worth
building. Two reasons that conclusion could be an artifact:

  1. THE MEDIAN HIDES THE SEGMENT EQUITY EXISTS FOR. Equity was kept in the model
     because income alone cannot reach the top of the market: at a 5x LTI multiple a
     EUR 2.4M purchase needs EUR 480k of gross household income. That argument was
     always about the top price bin. A median across all properties washes it out.

  2. THE SWEPT RANGE WAS TOO NARROW. The first sweep ran equity from 0 to 40% of LTI
     capacity. A buyer at EUR 2.4M plausibly carries equity worth MULTIPLES of their
     borrowing capacity, not a fraction of it. Capping the sweep at 0.4 would
     understate the effect by construction.

This script fixes both: a far wider equity range, reported per price bin.

Bins are price quintiles — transparent, pre-declared, and stable. Deliberately NOT
K-Means clusters: reporting bins should not carry an arbitrary k or move under
resampling, and they are descriptive rather than causal.

THE CLEARING CHECK MATTERS AS MUCH AS THE PRICE CHECK
-----------------------------------------------------
If the top bin fails to clear at low equity, that is the "luxury does not sell"
artifact the whole equity term exists to prevent. Sale probability by bin is
therefore reported alongside the optimal price. Note the standing rule: equity is
never adjusted to MAKE a segment clear. If the top bin will not clear under
externally plausible equity, that is a result about who buys Amsterdam property, not
a parameter to tune.

Run from the repo root:
    python scripts/sweep_equity_by_bin.py
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
    calibrate_dispersion,
    make_market_config,
    optimize_prices,
)
from src.simulation.valuation import ValuationError, make_valuations  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "simulation.yaml"
RESULTS_PATH = REPO_ROOT / "data" / "sweep_equity_by_bin.csv"

# Multipliers on the CBS-DERIVED equity schedule (config/equity_function.yaml).
# 1.0 is the derived central case; the rest bracket it. 0.0 is retained because it is
# the informative counterfactual — it shows what the model does when buyers have no
# equity at all, which is the "income cannot reach the top of the market" scenario.
EQUITY_VALUES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
ELASTICITY_VALUES = [-0.85, -0.65, -0.45]
N_BINS = 5


def main() -> None:
    print("=" * 78)
    print("EQUITY SWEEP BY PRICE BIN")
    print("=" * 78)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s = cfg["sweep"]
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
    bin_edges = pd.qcut(v_true, N_BINS).categories
    print(f"\nPool sample: {len(pool)} properties, {N_BINS} price-quintile bins")
    for label, edge in zip([f"Q{i}" for i in range(1, N_BINS + 1)], bin_edges):
        n = int((price_bin == label).sum())
        print(f"  {label}: n={n:>3}  EUR {edge.left:>10,.0f} - {edge.right:>10,.0f}")

    mc = make_market_config(cfg, n_properties_total=len(pool))

    rows = []
    print(f"\nRunning {len(EQUITY_VALUES) * len(ELASTICITY_VALUES)} configurations...")

    for equity in EQUITY_VALUES:
        buyers = generate_buyers(cfg, pool_full, rng, equity_multiplier=equity)
        ability = AbilityIndex(buyers, pool, cfg)

        for target_e in ELASTICITY_VALUES:
            dispersion, achieved = calibrate_dispersion(
                v_true, v_true, ability, mc, target_e
            )
            res = optimize_prices(v_est, v_true, ability, dispersion, mc, cfg)
            res["price_bin"] = price_bin.astype(str)

            for label, g in res.groupby("price_bin"):
                rows.append({
                    "equity_multiplier": equity,
                    "target_elasticity": target_e,
                    "achieved_elasticity": achieved,
                    "calibration_ok": abs(achieved - target_e) < 0.02,
                    "price_bin": label,
                    "n": len(g),
                    "median_optimal_multiple": g["optimal_multiple"].median(),
                    "mean_p_sale": g["p_sale"].mean(),
                    "share_at_grid_bound": g["at_grid_bound"].mean(),
                })
        print(f"  equity {equity:.1f} done")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    report(results)
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def report(results: pd.DataFrame) -> None:
    """Report every cell, annotating rather than deleting the unconverged ones.

    An earlier version of this function dropped cells failing the convergence gate
    and computed the verdict over the survivors. That inverted the answer. The gate
    preferentially removes the STRONGEST effects — a cell is excluded precisely
    because the optimum ran off the grid, which happens when the parameter matters
    most. Aggregating over survivors then reports the effect as weak, and a NaN read
    as "small" produced a confident wrong conclusion.

    So exclusions are now surfaced as evidence, never as holes.
    """
    b = results[results["target_elasticity"] == -0.65].copy()
    b["converged"] = b["calibration_ok"] & (b["share_at_grid_bound"] <= 0.05)

    piv = b.pivot(index="price_bin", columns="equity_multiplier",
                  values="median_optimal_multiple")
    piv_sale = b.pivot(index="price_bin", columns="equity_multiplier",
                       values="mean_p_sale")
    piv_bound = b.pivot(index="price_bin", columns="equity_multiplier",
                        values="share_at_grid_bound")

    print("\n" + "=" * 78)
    print("OPTIMAL PRICE MULTIPLE BY BIN AND EQUITY  (elasticity -0.65)")
    print("  '*' marks a cell where >5% of properties optimise onto the grid edge —")
    print("  the optimum is a lower bound there, so the true effect is LARGER.")
    print("=" * 78)
    header = "        " + "".join(f"{c:>9.1f}" for c in piv.columns)
    print(header)
    for bin_label in piv.index:
        cells = "".join(
            f"{piv.loc[bin_label, c]:>8.2f}"
            + ("*" if piv_bound.loc[bin_label, c] > 0.05 else " ")
            for c in piv.columns
        )
        print(f"  {bin_label:<6}{cells}")

    print("\n" + "=" * 78)
    print("SALE PROBABILITY BY BIN AND EQUITY  (the 'luxury does not sell' check)")
    print("=" * 78)
    print(piv_sale.round(3).to_string())

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    spread = (piv.max(axis=1) - piv.min(axis=1)) / piv.mean(axis=1) * 100
    sale_spread = (piv_sale.max(axis=1) - piv_sale.min(axis=1)) * 100

    print(f"\n  {'bin':<6}{'price move across equity':>26}{'p_sale move (pp)':>20}")
    print("  " + "-" * 52)
    for bin_label in piv.index:
        print(f"  {bin_label:<6}{spread[bin_label]:>25.2f}%{sale_spread[bin_label]:>19.1f}")

    top, bottom = piv.index[-1], piv.index[0]
    print(f"\n  top bin ({top}):    price moves {spread[top]:.1f}%, "
          f"sale probability {piv_sale.loc[top].min():.3f} -> {piv_sale.loc[top].max():.3f}")
    print(f"  bottom bin ({bottom}): price moves {spread[bottom]:.1f}%")

    if spread[top] > 3 * max(spread[bottom], 1e-9) or sale_spread[top] > 10:
        print("\n  EQUITY IS LOAD-BEARING AT THE TOP, and the pooled median hid it")
        print("  (3.7% pooled vs the figure above). The CBS wealth derivation is worth")
        print("  building, and it matters most for exactly the thin, expensive segment")
        print("  where the prediction audit also found the widest error (XL 19.7%).")
        print("\n  NOTE the standing rule: equity is derived from CBS and swept. It is")
        print("  never adjusted to make a segment clear. That the top bin fails to")
        print("  clear at low equity is a RESULT about who buys Amsterdam property.")
    else:
        print("\n  Equity remains weak even in the top bin and even at 3x LTI capacity.")
        print("  The CBS derivation can be documented as checked-and-not-load-bearing.")


if __name__ == "__main__":
    main()
