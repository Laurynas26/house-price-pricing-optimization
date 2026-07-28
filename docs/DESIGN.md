# Simulation Architecture & Design

## 1. Overview

A pricing optimization simulator that models how demand elasticity shapes optimal strategy, demonstrating the gap between price prediction and pricing optimization in the housing market.

---

## 2. Property Pool

We use 4,054 real Amsterdam properties from Funda (September 2025), each with listing price, size, bedrooms, luxury score, location, and a cluster ID (0-7). Property clusters reflect price segmentation (budget to luxury) and are used for valuation and segment-level analysis — NOT for elasticity assignment. Each property is valued using a trained price prediction model. Real properties ground the simulation in actual market data and enable validation: simulated prices can be compared directly against real Funda asking prices and segment-level patterns.

---

## 3. System Components

**Property**

Represents a real Funda listing with immutable attributes (ID, price, size, bedrooms, cluster, luxury_score, location). Property has NO elasticity attribute. Each property has an intrinsic valuation (from price prediction model) and cluster membership (for analysis). Tracks mutable state: current list price, days on market, bids received, sold/unsold status.

**Buyer**

An agent belonging to one of 5 archetypes (Luxury Central, Luxury Spacious, Mid-Urban, Mid-Suburban, Budget). Each archetype has: a fixed budget range, size/location preferences, and an elasticity value (from -0.35 to -1.0). When presented a property matching their preferences, buyer decides whether to bid based on: property valuation relative to listed price, filtered through elasticity-determined tolerance band. Low elasticity (luxury) = willing to pay 15% over valuation. High elasticity (budget) = only bids at or below valuation.

**Seller**

Owns a property and executes a pricing strategy. If strategy is "revenue-max," holds price steady. If "speed-max," lowers price daily to encourage bids. Accepts bids when satisfactory, tracks outcomes (days to sale, final price vs. list price).

**Market**

Orchestrates the simulation. Each day: (1) generates buyers from archetypes, (2) buyers filter properties by preferences, (3) buyers submit sealed bids on filtered properties, (4) highest bid wins property, (5) seller receives bid, updates metrics. Runs for N days (default 90) and collects aggregate outcomes: total revenue, average time-to-sale, occupancy rate by segment.

---

## 4. Elasticity Flow

Buyers are generated from archetypes (Luxury Central, Mid-Urban, Budget, etc.), each with a fixed elasticity value that determines price sensitivity. When a buyer encounters a property at price P with intrinsic valuation V, their elasticity determines their willingness-to-pay tolerance band: luxury buyers (elasticity -0.35) will bid up to 15% above V, while budget buyers (elasticity -1.00) only bid within 5% of V. These tolerance bands are calibrated from elasticity: luxury buyers with low elasticity (e.g., -0.35) have wider bands; budget buyers with high elasticity (e.g., -1.00) have narrower bands. Across the 90-day simulation, these individual elasticity-driven bid decisions aggregate into market outcomes: luxury segments command premium prices and experience faster sales (more willing buyers), while budget segments see tighter margins and slower clearing. This shows why pricing strategy must be segment-specific—the same price has fundamentally different effects depending on buyer type elasticity.

---

## 5. Buyer Archetypes

| Buyer Type | Budget | Elasticity | Size Pref | Location | Example | % |
|---|---|---|---|---|---|---|
| Luxury Central | €800k+ | -0.35 | Small (30-50m²) | City center | Penthouse, Grachtengordel | 12% |
| Luxury Spacious | €800k+ | -0.35 | Large (150m²+) | Suburb/Gooi | Villa with garden | 18% |
| Mid-Urban | €400-600k | -0.70 | Medium (60-100m²) | City/Canal | Apartment near metro | 30% |
| Mid-Suburban | €400-600k | -0.70 | Medium (100-150m²) | Suburb | Family home | 25% |
| Budget | €150-300k | -1.00 | Any | Flexible | Affordable unit | 15% |

---

## 6. Pricing Optimizer (Built Week 6-8)

Given a property's valuation V, buyer elasticity distribution, and a target objective (maximize revenue, minimize time-to-sale, or balanced), the optimizer recommends an optimal list price P*. Implemented as a Mixed Integer Program (MIP) using PuLP + CBC solver. Constraints: elasticity-derived willingness-to-pay bounds, market clearing conditions (at least one buyer willing to bid). Output: recommended list price for each property that maximizes the chosen objective. Comparison of strategies (revenue-max vs. speed-max vs. balanced) shows trade-offs across segments.

---

## 7. Assumptions (LOCKED)

1. **Elasticity is fixed per buyer archetype** — Luxury buyers maintain -0.35 elasticity throughout; elasticity doesn't vary by property type or market conditions.

2. **Property valuation is fixed for the 90-day simulation** — Each property's intrinsic valuation (from trained model) does not decay over time. (Note: In reality, time-on-market signals lower value; this dynamic is not modeled, treating all 90 days as equivalent market conditions.)

3. **Buyer preferences are hard filters** — Luxury Central only considers city center, 30-50m²; preferences don't bend even for exceptional properties.

4. **Sealed-bid auction, one round per property** — When a buyer encounters a property, they submit one sealed bid; highest bid wins; no negotiation or multi-round bidding.

5. **Overbidding applies uniformly but is not modeled** — Dutch market overbidding (10-20% above list) is acknowledged; simulation uses list price as baseline; actual transaction prices would be ~15% higher across all segments proportionally.

6. **No dynamic market factors** — Simulation assumes constant demand, no seasonality, no interest rate changes, no market shocks; buyer distribution and elasticity remain constant across all 90 days.

---

## 8. Asking vs. Transaction Prices

Funda.nl provides asking prices, not transaction prices. Dutch market typically sees 10-20% overbidding. Scope: This simulation optimizes asking prices (what sellers list). Actual transaction prices would be ~15% higher uniformly. Why it doesn't break validation: Elasticity ratios hold regardless. Budget buyers are still more price-sensitive than luxury buyers. Relative pricing optimization is valid.

---

## 9. Next Steps (Week 1)

1. Load & validate clustering_features.csv
2. Create buyer archetypes configuration (config/buyer_archetypes.yaml)
3. Build agent class stubs (Property, Buyer, Seller, Market)
4. Export property pool as pickled objects
5. Finalize design doc in Week 2
