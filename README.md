# House Price Pricing Optimization

Simulation-based pricing optimization system for Amsterdam residential properties.

**Goal:** Build an agent-based market simulator to optimize property prices under demand elasticity constraints.

**Scope:** Aug 1 - Dec 15, 2026 (5 months, ~145 hours)

## Project Overview

This project extends the house price prediction work with:
1. **Demand modeling** - Elasticity curves based on property segment (luxury, mid, budget)
2. **Market simulation** - Agent-based model with buyers, sellers, and market dynamics
3. **Pricing optimization** - Mixed Integer Program to find optimal prices
4. **Strategy comparison** - Revenue-max vs. speed-max vs. balanced approaches

## Data

- **Source:** Funda.nl listings (September 2025, Amsterdam)
- **Size:** 4,054 properties (after cleaning)
- **Features:** Price, size, bedrooms, luxury amenities, location, neighborhood

## Architecture

- **Simulation:** Agent-based market model (Property, Buyer, Seller, Market)
- **Demand:** Elasticity-driven demand curves (-0.35 to -0.85 by segment)
- **Optimizer:** PuLP + CBC solver with piecewise linear constraints
- **Validation:** 3-layer validation (held-out test, Funda comparison, optional re-scrape)

## Timeline

| Phase | Weeks | Status |
|-------|-------|--------|
| Foundation | 1-2 | Planning (Week 0) |
| Simulation | 3-5 | Next |
| Optimization | 6-8 | Q3 2026 |
| Analysis | 9-14 | Q4 2026 |
| Polish | 15 | Dec 2026 |

## Repository Structure

```
house-price-pricing-optimization/
├── data/
│   ├── clustering_features.csv        (features for clustering/segmentation)
│   ├── df_preprocessed_ca6f817b.pkl   (cached preprocessed listings)
│   └── simulation_results/
├── config/
│   └── elasticity_mapping.yaml        (cluster → elasticity mapping)
├── notebooks/
│   ├── 01_clustering.ipynb
│   ├── 02_demand_model.ipynb
│   ├── 03_simulation.ipynb
│   ├── 04_optimization.ipynb
│   └── 05_analysis.ipynb
├── src/
│   ├── simulation/
│   │   ├── market.py
│   │   ├── agents.py
│   │   └── dynamics.py
│   ├── optimization/
│   │   ├── optimizer.py
│   │   └── strategies.py
│   └── analysis/
│       ├── validation.py
│       └── metrics.py
├── scripts/
│   ├── extract_clustering_features.py
│   ├── run_clustering.py
│   └── run_simulation.py
└── README.md
```

## Week 0: Planning (Current)

✅ Elasticity research (academic-grounded values)
✅ MIP formulation (optimization approach)
✅ Clustering features extraction (4,054 properties)
⏳ K-Means clustering (8 neighborhood segments)
⏳ Elasticity mapping to clusters

## Next Steps

1. Run K-Means clustering on 4,054 properties
2. Analyze each cluster (price/m², size, luxury level)
3. Map elasticity values to clusters
4. Lock elasticity_mapping.yaml
5. Begin Phase 1 implementation (Week 1)

## Blog Posts

This project is documented across 5 blog posts:
- **Post 10:** "What drives Airbnb/housing prices? Simulation design"
- **Post 11:** "Estimating elasticity from market patterns"
- **Post 12:** "Optimizing prices under uncertainty"
- **Post 13:** "Pricing strategies: Revenue vs. speed"
- **Post 14:** "From prediction to pricing: Lessons learned"

## License

Educational project. Data from Funda.nl for learning purposes only.
