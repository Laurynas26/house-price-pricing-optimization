# Design

## 1. What this project is

A **sensitivity tool for pricing under unknown elasticity** — not an elasticity
estimation project.

The distinction is load-bearing. Elasticity cannot be estimated from Funda listing
data: transaction prices are absent, and the listing date was never scraped, so
neither time-on-market nor sale outcomes exist. Elasticity is therefore an **input**,
taken from the literature. A simulation that propagates an assumed elasticity cannot
discover anything about it — every result about optimal pricing is entailed by the
assignment.

So the question the project actually answers is:

> I could not observe elasticity, so I built a simulator that asks what optimal
> pricing looks like under assumed elasticity, and measures how much the answer
> depends on that assumption.

That makes the central experiment obvious: sweep the unknowns across their plausible
ranges and measure how far the optimal price moves. If it barely moves, optimal
pricing is robust to the uncertainty — a real result. If it moves a lot, the sweep has
quantified what the missing Kadaster transaction data is worth. Either outcome is
publishable, and neither requires pretending something was measured that was assumed.

### The standing methodological rule

**Sweep before you invest.** Any parameter about to receive expensive derivation gets
swept crudely first, to find out whether the answer depends on it. This is applied to
the project's own construction choices, not just to elasticity, and it has already
changed decisions in both directions — the CBS equity work was justified by a sweep
that showed it mattered, and refinement of carrying cost was abandoned because a sweep
showed it did not.

---

## 2. Property pool

4,054 Amsterdam listings from Funda (September 2025), built by
`scripts/build_property_pool.py` from the prediction project's cached preprocessed
data.

Each property carries asking price, size, bedrooms, rooms, a luxury amenity score,
postal code, PC4, zone, CBS neighbourhood, VvE service charge, and a size band matched
to the prediction audit's held-out quantile edges.

**Location** comes from `config/location_zones.yaml` (PC4 → Amsterdam stadsdeel, all
71 PC4 areas in the pool) and from `scripts/build_pc4_geography.py`, which derives
distance to Dam Square from PC4 polygon geometry. Both are deliberately independent of
price: zones are administrative and distance is geometric, so a later finding that
central properties command a premium is a finding rather than a restatement of how the
variables were built.

Two data corrections are applied and documented in the build script:

| issue | effect | fix |
|---|---|---|
| `city` contained 153 rows of `"STERDAM"` | looked like a second municipality | repaired to `AMSTERDAM` (all PC4s are 10xx/11xx) |
| `contribution_vve_num` inflated 100× | median read as €16,001/month | divided by 100 — upstream `parse_price` strips the Dutch decimal comma, so `"€ 105,46"` became `10546` |

The VvE bug is inherited from the prediction project, where `contribution_vve_num` is
a live model feature. The error is a uniform ×100 and therefore monotonic, so
tree-based models are unaffected in accuracy, but the values are wrong.

---

## 3. Valuation: what is true, and what the seller believes

The simulation needs two distinct objects, and conflating them is what imports false
precision:

- **`V_true`** — the property's value, anchored at the Funda asking price. It is the
  best available proxy and the only price the market actually saw.
- **`V_est`** — what the seller and the optimizer see: `V_true × (1 + ε)`.

The noise goes on the **seller's belief**, not on the property's value. Jittering
`V_true` alone would perturb the pool and average out; perturbing the belief makes
"what does valuation error cost a seller?" a measurable output.

`ε` is **bootstrapped** from the prediction project's held-out residuals, resampled
within size band — not drawn from a fitted Gaussian, since the empirical distribution
is available.

**Why not run the prediction model.** It was trained to predict asking price, so its
output is a smoothed estimate of asking price, not an independent valuation, and its
residual is model error rather than mispricing. Roughly 79% of this pool was in its
training data, so predictions over it would be leakage-optimistic. What is needed from
that project is not the model but its *error profile*, which is already measured
out-of-sample.

**Bootstrapped from the 820-listing overlap, not all 860.** The 40 non-overlapping
audit rows are records the prediction pipeline imputed and this pool drops. Their
errors are atypical: including them inflates the M band from 9.5% to 12.7%, a 34%
overstatement of valuation uncertainty for mid-size properties.

Measured dispersion by size band: XS 13.4%, S 8.7%, M 9.5%, L 14.9%, XL 19.7%.

> **Caveat.** The audit observed sizes to 312 m²; the pool reaches 500 m². Error for
> the largest properties is extrapolated, not measured.

---

## 4. Demand: budgets are derived, preferences are assumed

```
budget = LTI capacity(income) + equity(income) − kosten koper(price, FTB status)
         └── formula ──────┘   └── CBS ─────┘   └── tax policy ──────────────┘
```

Buyers are movers and first-time buyers, **not** the resident population.

### What is derived

**LTI capacity.** Dutch maximum borrowing is a published formula keyed to income,
set annually via NIBUD/AFM norms. Currently a flat multiple as a placeholder.

**Equity**, from CBS via `scripts/derive_equity_from_cbs.py`. Two points matter:

- *Deployable* equity, not total wealth: `eigen woning − hypotheekschuld + financiële
  bezittingen`. CBS `vermogen` includes business assets and other real estate that
  cannot fund a house purchase, so using the headline figure would overstate budgets.
- Computed from **aggregates on a common denominator**. `GemiddeldVermogen` is a mean
  among households *possessing* a component, and possession varies enormously by
  income (11% homeownership in decile 1, 91% in decile 10). Summing those means
  produces an inverted gradient in which the poorest look wealthiest. Verified against
  CBS's own `totaal − vermogen excl. eigen woning`: €363.6k vs €363.7k.

**Kosten koper**, from Dutch tax policy: 2% transfer tax, zero for qualifying
first-time buyers below the price cap (startersvrijstelling), plus fixed fees. It must
come from own funds because the mortgage is capped at the property value. This makes
buyer costs segment-varying *by policy* rather than by assumption.

### What is assumed, and cannot be otherwise

**Preferences** — size, location, willingness to trade one for the other. Funda shows
the **supply side only**: what sellers listed, never what buyers chose or rejected, and
sold/withdrawn status was never scraped. The standard instrument for preferences is a
discrete choice model, and it requires choice data that does not exist here. This is
not a weak signal; it is no signal.

**Elasticity** — from the literature. See §5.

**Market thickness** (`arrivals_per_day`) — unidentified, since there is no
time-on-market to calibrate against. Swept permanently, never fixed.

### The equity rule

Equity has enormous leverage: any segment can be made to clear by handing its buyers
more of it. Tuning it until the market clears would calibrate the demand side to
reproduce the very prices the project is trying to explain — the archetype circularity
one level down, and harder to spot because it would feel like sensible calibration.

**Equity is derived and swept. It is never adjusted to make a segment clear.** If the
top of the market fails to clear under CBS-derived equity, that is a result about who
buys Amsterdam property, not a parameter to fix.

> **Disclosure.** The sensitivity sweep ran *before* the derivation, so the equity
> level that clears the top bin was known when the derivation was written. The
> derivation uses no simulation input, but a reader should weigh the ordering.

### A standing test: external ≠ exogenous

For any external variable, ask whether it correlates with the outcome *through the
same causal channel being modelled*. Third-party provenance is not sufficient.

Worked example: CBS publishes household income at wijk level, and it is genuinely
external. But Amsterdam wijk income correlates hard with property prices, so using it
to drive buyer *location* preference would smuggle the target back into the demand
side under a new name — the same circularity, harder to see because the source is
independent. Income is therefore used for **budget capacity**, where the LTI link is
mechanical and documented, and **not** for location preference. PC4 polygon geometry
passes the same test; wijk income fails it.

---

## 5. Elasticity is calibrated, not asserted

The literature gives a **market-level** elasticity: percentage change in quantity
demanded per percentage change in price. A willingness-to-pay band is a different
object — an individual's tolerance spread. Citing −0.35 from a paper and using it
directly as a 15% bid band is a category error, and an economist reading the post
would flag exactly that.

So the WTP dispersion is **solved for numerically**: bisect on the dispersion
parameter until raising every asking price by 1% reduces expected sales by the target
percentage.

Budgets contribute to measured elasticity alongside WTP dispersion — raising prices
pushes buyers over their borrowing limit — which is why equity and elasticity interact
and why both belong in the same sweep.

**Model limit.** Thin markets cannot reach inelastic targets. At 20–40 arrivals/day the
dispersion parameter saturates and achieved elasticity stays near −1.1 regardless of
target, because budget constraints bind hard enough to make demand inherently elastic.
This is reported, not silently accepted.

---

## 6. Market outcome model

For each property, bids arrive as a Poisson process:

```
λ = arrivals_per_day × f_able × f_willing / n_properties
```

`f_able` is the share of buyers whose budget and preferences admit the property;
`f_willing` the share whose WTP covers the asking price. The property sells on the
first bid. Days-to-sale accumulates day by day rather than in closed form, so
buyer-pool decay can be switched on without changing the machinery.

> **Weakest part of the model, flagged rather than hidden.** Dividing by
> `n_properties` is a crude attention-sharing assumption: buyers spread across the
> market rather than viewing every listing.

**Proceeds are the asking price. No overbid is modelled.** This is deliberate — see §8.

---

## 7. Optimizer and seller strategies

Grid search over asking price as a multiple of `V_est`. **Not a MIP.** Building solver
machinery before knowing whether the objective surface requires it would be exactly
the premature investment this project argues against; the MIP goes in if and when the
surface warrants it.

The seller prices from `V_est` — their own noisy estimate — while outcomes depend on
`V_true`, which is what buyers assess. That asymmetry is the point: it makes valuation
error cost something measurable.

Optima landing on a grid edge are **flagged, never silently reported**. The grid needed
widening twice: the ceiling bound at inelastic settings, and the floor bound for
properties whose `V_est` sits far above `V_true`. The floor therefore has to span the
valuation-error distribution, not just the plausible pricing range.

Three strategies:

| strategy | objective |
|---|---|
| `revenue_max` | expected proceeds, ignoring time entirely |
| `speed_max` | probability of sale, subject to a **reservation price** |
| `balanced` | discounted proceeds net of carrying cost |

`speed_max` needs the reservation price to be well-posed at all: scored purely on
probability of sale it is unbounded below, since the fastest sale is a giveaway. The
reserve represents the seller's outside option — withdraw and relist rather than
accept less.

---

## 8. Asking vs transaction prices

Funda provides asking prices. Transaction prices are not in this data and cannot be
recovered from it.

The gap between the two is not a constant. An asking price with competing bids behaves
like an auction, so the premium over asking rises with the number of bidders — and
bidders are not spread evenly across the market. Expensive and unusual properties face
structurally thinner buyer pools. That makes the asking-to-sale gap correlate with
size, price tier and location: the same three dimensions the segmentation is built on.

The gap is therefore **segment-varying and unobserved**. Its shape cannot be estimated
here — transaction prices are absent, and the listing date was never scraped, so even
time-on-market is unavailable as a proxy.

**Scope:** the simulation optimizes **asking price only**. Realised revenue is
**explicitly out of scope**. Any quantity depending on the transaction price —
realised revenue, seller profit, actual overbid — is not produced by this project and
should not be inferred from its outputs.

This is a real limitation, not a formality. Because a segment-varying gap means the
mapping from asking price to realised revenue differs per segment, optimizing asking
price is a well-posed problem while optimizing realised revenue is not, on this data.

### A related market fact

Dutch mortgage appraisals are typically performed *after* a bid is accepted, and
commonly record the agreed price rather than an independent valuation. So the LTV cap
does not bind overbidding — the overbid is capitalised into the appraisal and is
financeable — and the binding budget constraint is LTI, not LTV.

This also means the market has no independent valuation anchor: price disciplines
valuation rather than the reverse. It supports anchoring `V_true` at the asking price,
and it is a caveat on the tolerance-band design, which assumes `V` exerts a discipline
on bidding that the real market apparently does not.

*Provenance: consistent with documented appraisal smoothing and with one first-hand
transaction; not verified at market scale.*

---

## 9. Results so far

**Optimal pricing inverts at the top of the market.** At CBS-derived equity, the bottom
four price quintiles price *above* the seller's estimate (1.06–1.09×) while the top
quintile prices *below* it (0.92×). Robust across market thickness: 98% of Q1–Q4 cells
above 1.0, 93% of Q5 cells below, over five arrival rates and three elasticities.

The mechanism: at the bottom the binding constraint is willingness, so able buyers are
plentiful and price can be pushed up; at the top the binding constraint is budget, the
buyer pool is thin, and the trade-off tips the other way. The top-end discount is a
function of how thin the pool is, and it vanishes as the pool deepens (0.74× at 80
arrivals, 0.99× at 1000).

**Elasticity moves the optimal price by 3–14%**, depending on market thickness — which
this data cannot identify. Market elasticity does not pin down optimal pricing: two
markets with identical measured elasticity but different thickness have materially
different optimal prices.

**CBS wealth accounts for most, not all, of the top of the market.** Derived equity
takes top-quintile clearing from 0.685 to 0.839, while pushing to 2.5× the derived
level reaches only 0.888. The residual gap is consistent with buyers outside Dutch
household statistics — international and corporate purchasers.

**Speed is ruinously expensive.** At baseline thickness, selling ~7 days faster costs
10.2% of price. Time costs are otherwise negligible: `revenue_max` and `balanced` pick
identical prices in all 15 tested cells, because a 90-day horizon at realistic discount
rates cannot generate enough carrying penalty to change a decision. In a thin market
with a decaying buyer pool, the revenue-maximising price falls *below* the reservation
price — the seller should withdraw.

---

## 10. Known limitations

1. **Preferences are stipulated.** No revealed-preference signal exists in the data.
2. **Market thickness is unidentified** and modulates the elasticity result by a factor
   of nearly five.
3. **Attention sharing is crude** — buyers spread uniformly across the pool.
4. **Valuation error is extrapolated above 312 m².**
5. **The Amsterdam equity tail is a two-moment lognormal fit.** Household wealth is not
   lognormal — it admits negative values and has a heavier tail — but CBS publishes no
   wealth distribution below gemeente level. The fit likely *understates* the top.
6. **Amsterdam wealth is measured over all households, not buyers.** Buyers skew
   wealthier, so equity is probably understated.
7. **CBS cannot cross income decile with owner-occupier status** — the table offers
   marginals, not a cross-tabulation.
8. **Segment membership is sample-dependent** near boundaries (bootstrap ARI down to
   0.76), which is one reason reporting uses transparent price quintiles rather than
   clusters.

---

## 11. What was retired

`archive/week0/` holds the superseded artifacts with a full account of why. In short:
the original eight "buyer archetypes" were property clusters relabelled as buyers,
their IQR-derived bounds left 60.2% of properties reachable by no buyer, their location
filters matched nothing in the data, and their tolerance bands were not a function of
elasticity.

The audit that established all of this is reproducible: `python
scripts/audit_week0_state.py`.

Segments survive only as **reporting bins**, and those are price quintiles —
transparent, pre-declared, stable under resampling, and descriptive rather than causal.
`scripts/build_segmentation.py` and `scripts/validate_segmentation.py` are retained as
diagnostics: they are the evidence that a properly-built clustering still was not the
right object.
