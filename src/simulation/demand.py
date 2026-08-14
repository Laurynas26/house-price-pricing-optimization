"""Buyer population and willingness to pay.

BUDGETS ARE DERIVED, PREFERENCES ARE ASSUMED
--------------------------------------------
    budget = LTI capacity(income) + equity(income) - kosten koper(price, FTB status)
             |____ formula ____|   |__ swept ___|   |____ tax policy ____|

Buyers are movers and first-time buyers, not the resident population.

Preferences (size, location, willingness to trade one for the other) cannot be
derived from this data at all. Funda shows the supply side only: we observe what
sellers listed, never what buyers chose or rejected, and sold/withdrawn status was
never scraped. The standard instrument for preferences is a discrete choice model,
and it requires choice data that does not exist here. Preference parameters are
therefore stipulated and swept.

EQUITY IS THE DANGEROUS PARAMETER
---------------------------------
Equity has enormous leverage: any segment can be made to clear by giving its buyers
more of it. Tuning equity until the market clears would calibrate the demand side to
reproduce the very prices the project is trying to explain — the archetype
circularity one level down, and harder to spot because it would feel like sensible
calibration. Equity is swept, never fitted. If the top of the market fails to clear
under externally plausible equity, that is a result about who buys Amsterdam
property, not a parameter to adjust.

Equity is modelled as a function of the BUYER's income, not of the property's price.
Making it depend on property price would be circular by construction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class BuyerPopulation:
    """A sample of prospective buyers with budgets and size preferences."""

    budget: np.ndarray          # max affordable purchase price, EUR
    preferred_size: np.ndarray  # m2
    income: np.ndarray
    equity: np.ndarray

    def __len__(self) -> int:
        return len(self.budget)


EQUITY_FUNCTION_PATH = Path(__file__).resolve().parents[2] / "config" / "equity_function.yaml"


def load_equity_schedule() -> dict[int, float]:
    """Deployable equity in EUR by income decile, derived by scripts/derive_equity_from_cbs.py."""
    if not EQUITY_FUNCTION_PATH.exists():
        raise FileNotFoundError(
            f"{EQUITY_FUNCTION_PATH} not found. Run: "
            "python scripts/derive_equity_from_cbs.py\n"
            "Equity must be derived from CBS, never invented — it has enough leverage "
            "on the result that a hand-picked value would decide the answer."
        )
    with open(EQUITY_FUNCTION_PATH, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return {int(k): float(v) for k, v in doc["equity_by_income_decile_eur"].items()}


def generate_buyers(
    cfg: dict,
    pool: pd.DataFrame,
    rng: np.random.Generator,
    n_buyers: int = 4000,
    equity_multiplier: float = 1.0,
    equity_schedule: dict[int, float] | None = None,
) -> BuyerPopulation:
    """Draw a buyer sample with exogenously-constructed budgets.

    `equity_multiplier` scales the CBS-derived schedule for sensitivity analysis.
    1.0 is the derived central case. The schedule itself is never edited — sweeping a
    multiplier keeps the derivation auditable and makes any departure from it visible
    as a number rather than as a quiet change to the source data.
    """
    b = cfg["budget"]
    schedule = equity_schedule if equity_schedule is not None else load_equity_schedule()

    # Income: lognormal, median-anchored.
    median_income = b["income_median_eur"]
    income = rng.lognormal(np.log(median_income), b["income_sigma_log"], n_buyers)

    # LTI capacity — stand-in for the NIBUD/AFM published formula.
    capacity = b["lti_multiple"] * income

    # Equity comes from the CBS schedule, looked up by the buyer's income decile.
    #
    # ASSUMPTION, stated because it is doing real work: a buyer's rank in the BUYER
    # income distribution is treated as their rank in the national HOUSEHOLD income
    # distribution. Buyers skew richer than households at large, so this likely
    # understates equity. CBS cannot resolve it — its wealth table conditions on
    # income decile or on owner-occupier status, never on both, and never on
    # "recently purchased".
    decile = np.clip(
        (pd.Series(income).rank(pct=True).to_numpy() * 10).astype(int) + 1, 1, 10
    )
    equity = np.array([schedule[d] for d in decile]) * equity_multiplier

    # Kosten koper must come from own funds: the mortgage is capped at the property
    # value, so transfer tax and fees are never borrowed. First-time buyers below
    # the cap are exempt from transfer tax (startersvrijstelling), which makes
    # buyer costs segment-varying by POLICY rather than by assumption.
    funds = capacity + equity - b["fixed_costs_eur"]
    is_starter = rng.random(n_buyers) < b["starters_share_of_buyers"]

    budget_with_tax = funds / (1.0 + b["transfer_tax_rate"])
    budget_exempt = funds
    budget = np.where(
        is_starter & (budget_exempt <= b["starters_exemption_price_cap"]),
        budget_exempt,
        budget_with_tax,
    )
    budget = np.maximum(budget, 0.0)

    # Preferred size, drawn from the market's own size distribution. ASSUMED —
    # there is no revealed-preference signal to estimate it from.
    preferred_size = rng.choice(pool["size_num"].to_numpy(), size=n_buyers, replace=True)

    return BuyerPopulation(
        budget=budget, preferred_size=preferred_size, income=income, equity=equity
    )


def eligible_fraction(
    buyers: BuyerPopulation,
    pool: pd.DataFrame,
    prices: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Fraction of buyers ABLE to buy each property: budget and preference filters.

    Returns one fraction per property. Willingness (a function of price relative to
    value) is handled separately in willing_fraction, because ability depends on the
    buyer's finances while willingness depends on the price-to-value ratio, and the
    sweep needs to move them independently.
    """
    d = cfg["demand"]
    sizes = pool["size_num"].to_numpy()
    dists = pool["dist_to_centre_km"].to_numpy()

    # (n_buyers, n_properties) boolean, chunked to bound memory.
    can_afford = buyers.budget[:, None] >= prices[None, :]

    size_ok = (
        np.abs(sizes[None, :] - buyers.preferred_size[:, None])
        <= d["size_tolerance_frac"] * buyers.preferred_size[:, None]
    )
    dist_ok = dists[None, :] <= d["max_distance_km"]

    return (can_afford & size_ok & dist_ok).mean(axis=0)


def willing_fraction(
    v_true: np.ndarray, prices: np.ndarray, wtp_dispersion: float
) -> np.ndarray:
    """Fraction of buyers whose willingness to pay covers the asking price.

    A buyer's WTP for a property is v_true * (1 + t) with t ~ Normal(0, dispersion).
    So the willing fraction is P(t >= price/v_true - 1), available in closed form —
    no need to draw t per buyer-property pair.

    `dispersion` is NOT elasticity and is not set from the literature. It is
    calibrated so the AGGREGATE demand curve exhibits the target elasticity
    (see market.calibrate_dispersion). Citing an elasticity from a paper and using
    it directly as a bid-tolerance width would be a category error: elasticity is
    percentage change in quantity demanded per percentage change in price, while a
    tolerance band is an individual willingness-to-pay spread. They are different
    objects and one does not imply the other.
    """
    from scipy.stats import norm

    threshold = prices / v_true - 1.0
    return 1.0 - norm.cdf(threshold / wtp_dispersion)
