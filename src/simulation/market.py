"""Market outcome model, elasticity calibration, and the price optimizer.

OUTCOME MODEL
-------------
For each property, bids arrive as a Poisson process whose daily rate depends on how
many buyers are both able (budget) and willing (price relative to value) to bid:

    lambda_j = arrivals_per_day * f_able_j * f_willing_j / n_properties

The division by n_properties is a crude attention-sharing assumption: buyers spread
across the market rather than all viewing every listing. It is the weakest part of
this model and is flagged rather than hidden.

The property sells on the first bid. Days-to-sale is accumulated day by day rather
than in closed form, so that buyer-pool decay (a rate falling with days on market)
can be switched on without changing the machinery.

PROCEEDS ARE THE ASKING PRICE, DELIBERATELY
-------------------------------------------
When a property sells, proceeds are the asking price — no overbid is modelled. This
is not an oversight. DESIGN.md section 8 puts realised transaction prices out of
scope: the asking-to-sale gap varies with bidder count, bidder counts are thinner for
expensive and unusual properties, and none of it is observable in Funda data. Adding
a competition premium here would silently reinstate exactly the claim that was
retracted. The objective is therefore an internally consistent model quantity, not a
prediction of Dutch sale prices.

ELASTICITY IS CALIBRATED, NOT ASSERTED
--------------------------------------
The literature gives a market-level elasticity: the percentage change in quantity
demanded for a percentage change in price. That is not a willingness-to-pay band.
So the WTP dispersion is solved for numerically, such that raising every asking price
by 1% reduces expected sales by the target percentage. Note that budgets contribute
to the measured elasticity as well as WTP dispersion does — raising prices pushes
buyers over their borrowing limit — which is why equity and elasticity interact and
why both belong in the same sweep.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .demand import BuyerPopulation


@dataclass
class MarketConfig:
    arrivals_per_day: float
    horizon_days: int
    discount_rate_daily: float
    carrying_cost_daily: float | np.ndarray  # scalar, or one value per property
    pool_decay_daily: float
    n_properties_total: int


# The seller's objective. Without a price on time, revenue-max trivially dominates
# speed-max and the comparison has a predetermined winner — which is why at least one
# time-cost channel must be active for this comparison to mean anything.
STRATEGIES = ("revenue_max", "speed_max", "balanced")


def objective_for(
    strategy: str,
    res: dict[str, np.ndarray],
    mc: MarketConfig,
    prices: np.ndarray | None = None,
    reservation_price: np.ndarray | None = None,
) -> np.ndarray:
    """Score a candidate price under one seller strategy.

    revenue_max  expected proceeds, ignoring time and holding costs entirely. The
                 seller who only cares what the property fetches.
    speed_max    sell as fast as possible, but never below a reservation price.
    balanced     discounted proceeds net of carrying cost — time is priced.

    WHY speed_max NEEDS A RESERVATION PRICE
    ---------------------------------------
    Scored purely on probability of sale, speed_max is unbounded below: the fastest
    way to sell is to give the property away, so the optimum is always the lowest
    price on the grid. That is not a strategy, it is a missing constraint, and it
    made every earlier run of the comparison uninformative.

    A real seller has an outside option — withdraw and relist, or stay put. The
    reservation price represents it. Below the reserve the objective is -inf, so
    speed_max minimises time subject to still clearing the floor, which is an actual
    decision a seller faces.
    """
    if strategy == "revenue_max":
        return res["undiscounted_proceeds"]

    if strategy == "speed_max":
        score = res["p_sale"].astype(float)
        if reservation_price is not None:
            if prices is None:
                raise ValueError("speed_max with a reservation price needs `prices`")
            score = np.where(prices >= reservation_price, score, -np.inf)
        return score

    if strategy == "balanced":
        return res["discounted_proceeds"] - mc.carrying_cost_daily * res["expected_days"]

    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


class AbilityIndex:
    """Precomputed budget lookup: fraction of buyers able to buy each property.

    Preference matching (size, distance) does not depend on price, so it is computed
    once. Within the matching buyers, the affordable fraction at any price is a
    binary search over sorted budgets rather than a full rescan — which is what makes
    a two-dimensional sweep tractable.
    """

    def __init__(self, buyers: BuyerPopulation, pool: pd.DataFrame, cfg: dict):
        d = cfg["demand"]
        sizes = pool["size_num"].to_numpy()
        dists = pool["dist_to_centre_km"].to_numpy()

        tol = d["size_tolerance_frac"] * buyers.preferred_size[:, None]
        pref_ok = np.abs(sizes[None, :] - buyers.preferred_size[:, None]) <= tol
        pref_ok &= dists[None, :] <= d["max_distance_km"]

        self.n_buyers = len(buyers)
        self.sorted_budgets = [
            np.sort(buyers.budget[pref_ok[:, j]]) for j in range(pool.shape[0])
        ]

    def able_fraction(self, prices: np.ndarray) -> np.ndarray:
        out = np.empty(len(prices))
        for j, budgets in enumerate(self.sorted_budgets):
            if budgets.size == 0:
                out[j] = 0.0
            else:
                idx = np.searchsorted(budgets, prices[j], side="left")
                out[j] = (budgets.size - idx) / self.n_buyers
        return out


def willing_fraction(
    v_true: np.ndarray, prices: np.ndarray, dispersion: float
) -> np.ndarray:
    return 1.0 - norm.cdf((prices / v_true - 1.0) / dispersion)


def outcome(
    prices: np.ndarray,
    v_true: np.ndarray,
    ability: AbilityIndex,
    dispersion: float,
    mc: MarketConfig,
) -> dict[str, np.ndarray]:
    """Expected sale probability, discounted proceeds and days on market."""
    f_able = ability.able_fraction(prices)
    f_willing = willing_fraction(v_true, prices, dispersion)
    lam0 = mc.arrivals_per_day * f_able * f_willing / mc.n_properties_total

    survival = np.ones_like(prices)
    p_sale = np.zeros_like(prices)
    disc_proceeds = np.zeros_like(prices)
    undisc_proceeds = np.zeros_like(prices)
    expected_days = np.zeros_like(prices)

    for day in range(mc.horizon_days):
        lam = lam0 * np.exp(-mc.pool_decay_daily * day)
        sell_today = survival * (1.0 - np.exp(-lam))

        p_sale += sell_today
        disc_proceeds += sell_today * prices * np.exp(-mc.discount_rate_daily * day)
        undisc_proceeds += sell_today * prices
        expected_days += survival  # time still on market during this day

        survival = survival * np.exp(-lam)

    return {
        "p_sale": p_sale,
        "discounted_proceeds": disc_proceeds,
        "undiscounted_proceeds": undisc_proceeds,
        "expected_days": expected_days,
        "objective": disc_proceeds - mc.carrying_cost_daily * expected_days,
    }


def measure_elasticity(
    prices: np.ndarray,
    v_true: np.ndarray,
    ability: AbilityIndex,
    dispersion: float,
    mc: MarketConfig,
    bump: float = 0.01,
) -> float:
    """Arc elasticity of expected sales with respect to a uniform price change."""
    q0 = outcome(prices, v_true, ability, dispersion, mc)["p_sale"].sum()
    q1 = outcome(prices * (1 + bump), v_true, ability, dispersion, mc)["p_sale"].sum()
    if q0 <= 0 or q1 <= 0:
        return np.nan
    return float(np.log(q1 / q0) / np.log(1 + bump))


def calibrate_dispersion(
    prices: np.ndarray,
    v_true: np.ndarray,
    ability: AbilityIndex,
    mc: MarketConfig,
    target_elasticity: float,
    lo: float = 0.01,
    hi: float = 3.0,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> tuple[float, float]:
    """Solve for the WTP dispersion that reproduces the target market elasticity.

    Demand gets less elastic as dispersion widens (buyers are spread over a wider
    WTP range, so a given price move converts fewer of them), making measured
    elasticity increasing in dispersion. Bisection is therefore well behaved.

    Returns (dispersion, achieved_elasticity). If the target lies outside what the
    model can produce, returns the closest achievable endpoint rather than silently
    reporting a value it did not hit.
    """
    def f(d: float) -> float:
        return measure_elasticity(prices, v_true, ability, d, mc)

    e_lo, e_hi = f(lo), f(hi)
    if not (min(e_lo, e_hi) <= target_elasticity <= max(e_lo, e_hi)):
        best = lo if abs(e_lo - target_elasticity) < abs(e_hi - target_elasticity) else hi
        return best, f(best)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e_mid = f(mid)
        if abs(e_mid - target_elasticity) < tol:
            return mid, e_mid
        if (e_mid > target_elasticity) == (e_lo > target_elasticity):
            lo, e_lo = mid, e_mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, f(mid)


def optimize_prices(
    v_est: np.ndarray,
    v_true: np.ndarray,
    ability: AbilityIndex,
    dispersion: float,
    mc: MarketConfig,
    cfg: dict,
    strategy: str = "balanced",
    reservation_price: np.ndarray | None = None,
) -> pd.DataFrame:
    """Grid search the asking price that maximises the objective, per property.

    The seller sets price from v_est — their own noisy estimate — while outcomes
    depend on v_true, which is what buyers assess. That asymmetry is the whole point:
    it makes valuation error cost something measurable rather than assuming the
    seller knows the value exactly.
    """
    o = cfg["optimizer"]
    multiples = np.linspace(
        o["price_multiple_min"], o["price_multiple_max"], o["price_multiple_steps"]
    )

    best_obj = np.full(len(v_est), -np.inf)
    best_mult = np.zeros(len(v_est))
    best_psale = np.zeros(len(v_est))
    best_days = np.zeros(len(v_est))

    for m in multiples:
        prices = v_est * m
        res = outcome(prices, v_true, ability, dispersion, mc)
        score = objective_for(strategy, res, mc, prices, reservation_price)
        better = score > best_obj
        best_obj = np.where(better, score, best_obj)
        best_mult = np.where(better, m, best_mult)
        best_psale = np.where(better, res["p_sale"], best_psale)
        best_days = np.where(better, res["expected_days"], best_days)

    # A property whose optimum lands on either end of the grid has not been solved —
    # the objective was still improving when the grid ran out. Floor and ceiling are
    # tracked separately because they have different causes: a ceiling-bound property
    # has demand too inelastic for the grid, while a floor-bound one is usually a
    # property whose seller estimate v_est sits far above v_true, so even the lowest
    # multiple of that estimate is priced above what buyers will pay.
    at_floor = best_mult <= multiples[0] + 1e-9
    at_ceiling = best_mult >= multiples[-1] - 1e-9
    at_bound = at_floor | at_ceiling

    return pd.DataFrame({
        "v_est": v_est,
        "v_true": v_true,
        "optimal_multiple": best_mult,
        "optimal_price": v_est * best_mult,
        "objective": best_obj,
        "p_sale": best_psale,
        "expected_days": best_days,
        "at_grid_bound": at_bound,
        "at_floor": at_floor,
        "at_ceiling": at_ceiling,
    })


def make_market_config(
    cfg: dict, n_properties_total: int, pool: pd.DataFrame | None = None
) -> MarketConfig:
    """Build the market config, using per-property VvE for carrying cost when available.

    A missing VvE is treated as zero rather than imputed. VvE applies to apartments,
    so an absent value usually means a freehold house with no service charge — the
    missingness is informative, and imputing a median would invent a cost that the
    property does not incur.

    Carrying cost is modelled for completeness but is not the time cost that drives
    pricing: 90 days of the median service charge is about 0.09% of the median asking
    price, against roughly 1.5% for the discount-rate term over the same period.
    """
    m = cfg["market"]

    if pool is not None and "contribution_vve_num" in pool:
        carrying = pool["contribution_vve_num"].fillna(0.0).to_numpy() * 12.0 / 365.0
    else:
        carrying = m["carrying_cost_monthly_eur"] * 12.0 / 365.0

    return MarketConfig(
        arrivals_per_day=cfg["demand"]["arrivals_per_day"],
        horizon_days=m["horizon_days"],
        discount_rate_daily=m["discount_rate_annual"] / 365.0,
        carrying_cost_daily=carrying,
        pool_decay_daily=m["buyer_pool_decay_daily"],
        n_properties_total=n_properties_total,
    )
