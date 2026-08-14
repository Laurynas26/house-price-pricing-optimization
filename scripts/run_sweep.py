"""
The central experiment: how far does the optimal asking price move when the
assumptions move?

Two parameters are swept together:

  elasticity        assumed, from literature. Cannot be estimated from listing data.
  equity_share      assumed, pending CBS derivation. Enormous leverage on budgets.

Elasticity is swept because it is the project's stated unknown. Equity is swept for a
different reason: to find out whether the elaborate two-stage CBS wealth construction
is worth building at all. If the optimal price barely moves across a wide equity
range, that construction is insurance rather than substance and can be documented as
"checked, does not drive the result" — a stronger position than a derivation that has
to be defended. If it does move the answer, the CBS work is load-bearing and that is
worth knowing BEFORE spending the effort.

That is the same sensitivity discipline the project applies to elasticity, turned on
the parts that were about to be treated as solid.

Run from the repo root:
    python scripts/run_sweep.py
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
RESULTS_PATH = REPO_ROOT / "data" / "sweep_results.csv"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    print("=" * 78)
    print("ELASTICITY x EQUITY SWEEP")
    print("=" * 78)

    cfg = load_config()
    s = cfg["sweep"]
    rng = np.random.default_rng(s["seed"])

    pool_full = load_property_pool()
    pool = pool_full.sample(
        n=min(s["n_property_sample"], len(pool_full)), random_state=s["seed"]
    ).reset_index(drop=True)
    print(f"Pool: {len(pool_full):,} properties, sampling {len(pool)} for the sweep")

    err = ValuationError(pool_full, overlap_only=cfg["valuation"]["use_overlap_only"])
    print(f"\nValuation error profile: {err.n_used} of {err.n_audit} audit listings "
          f"({'overlap only' if cfg['valuation']['use_overlap_only'] else 'all'})")
    print(err.summary().round(2).to_string())
    print(f"\n  audit observed sizes to {err.audit_max_size:.0f} m2; pool reaches "
          f"{pool_full['size_num'].max():.0f} m2 — error for the largest properties "
          "is extrapolated, not measured.")

    valued = make_valuations(pool, err, rng)
    v_true = valued["v_true"].to_numpy()
    v_est = valued["v_est"].to_numpy()

    mc = make_market_config(cfg, n_properties_total=len(pool))

    rows = []
    print("\n" + "=" * 78)
    print("SWEEPING")
    print("=" * 78)
    print(f"{'equity':>8}{'target_e':>10}{'achieved_e':>12}{'dispersion':>12}"
          f"{'med_mult':>10}{'p_sale':>9}{'days':>7}")
    print("-" * 78)

    for equity in s["equity_multiplier_values"]:
        buyers = generate_buyers(cfg, pool_full, rng, equity_multiplier=equity)
        ability = AbilityIndex(buyers, pool, cfg)

        for target_e in s["elasticity_values"]:
            dispersion, achieved = calibrate_dispersion(
                v_true, v_true, ability, mc, target_e
            )
            res = optimize_prices(v_est, v_true, ability, dispersion, mc, cfg)

            row = {
                "equity_multiplier": equity,
                "target_elasticity": target_e,
                "achieved_elasticity": achieved,
                "calibration_ok": abs(achieved - target_e) < 0.02,
                "wtp_dispersion": dispersion,
                "median_optimal_multiple": res["optimal_multiple"].median(),
                "share_at_grid_bound": res["at_grid_bound"].mean(),
                "share_at_floor": res["at_floor"].mean(),
                "share_at_ceiling": res["at_ceiling"].mean(),
                "mean_p_sale": res["p_sale"].mean(),
                "mean_days": res["expected_days"].mean(),
                "median_optimal_price": res["optimal_price"].median(),
            }
            rows.append(row)
            flag = "" if row["calibration_ok"] else "  <- CALIB MISSED"
            if row["share_at_grid_bound"] > 0.05:
                flag += (f"  <- bound {100*row['share_at_floor']:.0f}%lo/"
                         f"{100*row['share_at_ceiling']:.0f}%hi")
            print(f"{equity:>8.2f}{target_e:>10.2f}{achieved:>12.3f}{dispersion:>12.3f}"
                  f"{row['median_optimal_multiple']:>10.3f}"
                  f"{row['mean_p_sale']:>9.3f}{row['mean_days']:>7.1f}{flag}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    report(results)
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def report(results: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("HOW FAR DOES THE OPTIMAL PRICE MOVE?")
    print("=" * 78)

    piv = results.pivot(
        index="equity_multiplier",
        columns="target_elasticity",
        values="median_optimal_multiple",
    )
    print("\nMedian optimal price as a multiple of the seller's estimated value:\n")
    print(piv.round(4).to_string())

    # Cells where calibration missed its target, or where a large share of
    # properties optimise onto the grid edge, are not solved results. Including them
    # in the headline range would measure the grid and the calibrator rather than
    # the model.
    bad = results[(~results["calibration_ok"]) | (results["share_at_grid_bound"] > 0.05)]
    clean = results.drop(bad.index)

    print(f"\n  usable cells: {len(clean)}/{len(results)}")
    if len(bad):
        print(f"  excluded {len(bad)} cell(s):")
        for _, r in bad.iterrows():
            reasons = []
            if not r["calibration_ok"]:
                reasons.append(
                    f"calibration reached {r['achieved_elasticity']:.3f} "
                    f"vs target {r['target_elasticity']:.2f}"
                )
            if r["share_at_grid_bound"] > 0.05:
                reasons.append(f"{100*r['share_at_grid_bound']:.0f}% on grid bound")
            print(f"    equity {r['equity_multiplier']:.2f}, "
                  f"e={r['target_elasticity']:.2f}: {'; '.join(reasons)}")

    if len(clean) < 4:
        print("\n  TOO FEW USABLE CELLS to state a sensitivity result. Widen the grid,")
        print("  or restrict the elasticity range to what the demand model can reach.")
        return

    cpiv = clean.pivot(
        index="equity_multiplier",
        columns="target_elasticity",
        values="median_optimal_multiple",
    )
    across_e = (cpiv.max(axis=1) - cpiv.min(axis=1)).dropna()
    across_eq = (cpiv.max(axis=0) - cpiv.min(axis=0)).dropna()
    base = np.nanmean(cpiv.values)

    print("\n  On usable cells only:")
    print(f"    elasticity moves the optimal price by up to "
          f"{100*across_e.max()/base:.2f}%")
    print(f"    equity     moves the optimal price by up to "
          f"{100*across_eq.max()/base:.2f}%")

    print("\n  READ THIS AS: whichever parameter moves the optimal price least is the")
    print("  one not worth deriving precisely. That is the decision this sweep exists")
    print("  to make, and it is meant to be made BEFORE the effort is spent.")


if __name__ == "__main__":
    main()
