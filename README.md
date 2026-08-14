# House Price Pricing Optimization

**A sensitivity tool for pricing under unknown elasticity.** Not an elasticity
estimation project — the distinction is the point.

Elasticity cannot be estimated from Funda listing data. Transaction prices are absent,
and the listing date was never scraped, so neither time-on-market nor sale outcomes
exist. Elasticity is therefore an input taken from the literature, and a simulation
that propagates an assumed elasticity cannot discover anything about it.

So this project asks a different question:

> What does optimal pricing look like under assumed elasticity, and how much does the
> answer depend on the assumption?

If the optimal price barely moves across the plausible range, pricing is robust to the
uncertainty. If it moves a lot, the sweep has quantified what the missing Kadaster
transaction data is worth.

Extends [house_price_prediction_project](../house_price_prediction_project), which
supplies both the property pool and — more importantly — the **measured error profile**
that gives this simulation a defensible uncertainty structure.

## Results

**Optimal pricing inverts at the top of the market.** The bottom four price quintiles
should be listed *above* the model's estimated value (1.06–1.09×); the top quintile
*below* it (0.92×). Robust across market thickness — 98% of Q1–Q4 cells and 93% of Q5
cells, over five arrival rates and three elasticities.

The mechanism: at the bottom, the binding constraint is willingness, so able buyers are
plentiful and price can be pushed up. At the top, the binding constraint is budget, the
buyer pool is thin, and the trade-off reverses. The top-end discount tracks how thin
the pool is — 0.74× at 80 arrivals/day, 0.99× at 1000.

| finding | value |
|---|---|
| Elasticity moves optimal price by | **3–14%**, conditional on market thickness |
| Speed costs | **10.2% of price** to sell ~7 days faster |
| CBS wealth explains top-quintile clearing | 0.685 → **0.839** (2.5× derived reaches only 0.888) |
| Time costs (discount, VvE) change the decision | **never** — identical prices in all 15 cells |

Two of these are negative results and they are kept as results. Market elasticity does
not pin down optimal pricing: two markets with identical measured elasticity but
different thickness give materially different answers. And in a fast market, revenue-max
*is* the right strategy — there is nothing to trade away.

## Method

**Sweep before you invest.** Any parameter about to receive expensive derivation gets
swept crudely first, to find out whether the answer depends on it. Applied to the
project's own construction choices, not only to elasticity. It has changed decisions in
both directions — the CBS equity derivation was justified by a sweep showing it
mattered; refinement of carrying cost was abandoned because a sweep showed it did not.

**Derived vs assumed, labelled throughout.**

```
budget = LTI capacity(income) + equity(income) − kosten koper(price, FTB status)
         └── NIBUD formula ─┘   └── CBS ────┘   └── Dutch tax policy ─────────┘

preferences (size, location, trade-offs)  = ASSUMED — no revealed-preference signal exists
elasticity                                 = ASSUMED — from literature, swept
market thickness                           = UNIDENTIFIED — no time-on-market data, swept
```

**Elasticity is calibrated, not asserted.** The literature gives a market-level
elasticity; a willingness-to-pay band is a different object. The WTP dispersion is
solved for numerically so that aggregate demand exhibits the target elasticity.

**Equity is derived and never tuned.** It has enough leverage that any segment could be
made to clear by handing its buyers more of it — which would calibrate the demand side
to reproduce the prices being explained. That the top of the market does not fully
clear is reported as a result.

## Data

- **Source:** Funda.nl listings, Amsterdam, September 2025 — 4,054 properties
- **Location:** PC4 → stadsdeel mapping plus distance to Dam Square from polygon
  geometry. Both independent of price, so location findings are not circular.
- **Valuation error:** bootstrapped from 820 genuinely held-out listings in the
  prediction project, resampled within size band (dispersion 8.7%–19.7%)
- **Equity:** CBS StatLine 83834NED and 86160NED

Two upstream data bugs found and corrected: `city` truncated to `"STERDAM"` (153 rows),
and `contribution_vve_num` inflated 100× because `parse_price` strips the Dutch decimal
comma. The VvE bug is live in the prediction project, where the column is a model
feature — monotonic, so tree models are unaffected in accuracy, but the values are
wrong.

## Running it

```bash
pip install -r requirements.txt

python scripts/build_property_pool.py      # 4,054 properties with price + location
python scripts/build_pc4_geography.py      # distance to centre from the PC4 shapefile
python scripts/derive_equity_from_cbs.py   # equity schedule from CBS StatLine

python scripts/run_sweep.py                # elasticity x equity
python scripts/sweep_equity_by_bin.py      # equity by price quintile
python scripts/sweep_market_thickness.py   # does the elasticity result survive?
python scripts/check_inversion_robustness.py
python scripts/compare_strategies.py       # revenue vs speed vs balanced
```

Diagnostics, kept because they are the evidence for design decisions:

```bash
python scripts/audit_week0_state.py        # why the original archetypes were retired
python scripts/build_segmentation.py       # clustering diagnostics
python scripts/validate_segmentation.py    # adversarial checks on that clustering
```

## Repository

```
config/
  simulation.yaml        every parameter labelled DERIVED or ASSUMED
  location_zones.yaml    PC4 -> stadsdeel, not fitted to price
  equity_function.yaml   generated from CBS; never hand-edited
src/
  data/loaders.py        pool + geography join, fails loudly on a bad join
  simulation/
    valuation.py         V_true vs V_est, bootstrapped error
    demand.py            budgets and willingness to pay
    market.py            outcomes, elasticity calibration, optimizer, strategies
scripts/                 pipeline, sweeps, diagnostics
archive/week0/           retired artifacts + why (see its README)
docs/DESIGN.md           full design, assumptions, limitations
```

## Known limitations

Preferences are stipulated — there is no choice data. Market thickness is unidentified
and modulates the headline by a factor of nearly five. Attention sharing across the
pool is crude. Valuation error is extrapolated above 312 m². The Amsterdam equity tail
is a two-moment lognormal fit and probably understates the top. Full list in
[docs/DESIGN.md](docs/DESIGN.md#10-known-limitations).

**Realised transaction prices are explicitly out of scope.** The asking-to-sale gap
varies with bidder count, bidder counts are thinner for expensive properties, and none
of it is observable in this data. This project optimizes asking price and makes no
claim about what properties actually sell for.

## Blog posts

- **Post 10** — Simulation design: what you can and cannot ask of listing data
- **Post 11** — Building buyers from Dutch income and mortgage rules
- **Post 12** — How much does optimal pricing depend on elasticity?
- **Post 13** — What speed costs, and when strategy stops mattering
- **Post 14** — From prediction to pricing: lessons learned

## License

Educational project. Data from Funda.nl for learning purposes only.
