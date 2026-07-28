# House Price Pricing Optimization

Simulation-based pricing optimization system for Amsterdam residential properties.

**Goal:** Build an agent-based market simulator to optimize property prices under demand elasticity constraints.

**Scope:** Aug 1 - Dec 15, 2026 (5 months, ~145 hours)

## Project Overview

This project extends the house price prediction work with:
1. **Demand modeling** - Elasticity curves based on buyer type (luxury, mid, budget) — NOT property type
2. **Market simulation** - Agent-based model with 5 buyer archetypes, 4,054 real properties, and market dynamics
3. **Pricing optimization** - Mixed Integer Program to find optimal prices
4. **Strategy comparison** - Revenue-max vs. speed-max vs. balanced approaches

**Key insight:** Elasticity is a buyer attribute (price sensitivity), not a property attribute. Luxury buyers are less price-sensitive regardless of property type.

## Data

- **Source:** Funda.nl listings (September 2025, Amsterdam)
- **Size:** 4,054 properties (after cleaning)
- **Features:** Price, size, bedrooms, luxury amenities, location, neighborhood

### Data Pipeline

```
house_price_prediction_project/
  └── data/cache/df_preprocessed_ca6f817b.pkl (4,960 listings, 64 features)
       ↓
extract_clustering_features.py (this repo)
  - Loads cached preprocessed data
  - Computes: price_per_m2, luxury_score
  - Selects: price_per_m2, size_num, bedrooms, luxury_score, nr_rooms, postal_code_clean
  - Outputs: data/clustering_features.csv (4,054 clean rows)
       ↓
run_clustering.py (this repo)
  - Loads clustering_features.csv
  - Runs K-Means (8 clusters)
  - Maps elasticity to clusters (for PROPERTY SEGMENTATION ANALYSIS, not buyer behavior)
  - Outputs: config/elasticity_mapping.yaml
       ↓
Simulation (Weeks 3-5)
  - Uses clustering_features.csv (for property pool valuation)
  - Uses config/buyer_archetypes.yaml (for buyer elasticity and behavior)
  - elasticity_mapping.yaml used only for segment-level validation/analysis
```

**Important:** elasticity_mapping.yaml is for property segmentation and analysis. Buyer elasticity is defined separately in buyer_archetypes.yaml.

**Files in this repo:**
- `data/df_preprocessed_ca6f817b.pkl` - Original cached data (reference)
- `data/clustering_features.csv` - Extracted features for clustering
- `scripts/extract_clustering_features.py` - Feature extraction script
- `scripts/run_clustering.py` - Clustering + elasticity mapping script
- `config/elasticity_mapping.yaml` - Locked elasticity values

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for detailed architecture, system components, elasticity flow, buyer archetypes, and locked assumptions.

## Architecture

- **Simulation:** Agent-based market model (Property, Buyer, Seller, Market)
- **Buyers:** 5 archetypes (Luxury Central, Luxury Spacious, Mid-Urban, Mid-Suburban, Budget) with fixed elasticity values (-0.35 to -1.0)
- **Properties:** 4,054 real Funda listings with cluster-based segmentation (for valuation, NOT elasticity)
- **Demand:** Elasticity determines buyer tolerance band; sealed-bid auction with preference filtering
- **Optimizer:** PuLP + CBC solver with piecewise linear constraints (Week 6+)
- **Validation:** Segment-level comparison (sim prices vs. real Funda by cluster), strategy outcome comparison

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
├── docs/
│   └── DESIGN.md                      (architecture, elasticity flow, assumptions)
├── data/
│   ├── clustering_features.csv        (4,054 properties for segmentation)
│   ├── df_preprocessed_ca6f817b.pkl   (cached preprocessed listings, reference)
│   └── simulation_results/
├── config/
│   ├── elasticity_mapping.yaml        (property cluster → segmentation tier, for analysis)
│   └── buyer_archetypes.yaml          (buyer types → elasticity, budgets, preferences)
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
│   ├── data/
│   │   ├── loaders.py
│   │   └── utils.py
│   └── analysis/
│       ├── validation.py
│       └── metrics.py
├── scripts/
│   ├── extract_clustering_features.py
│   ├── run_clustering.py
│   └── run_simulation.py
└── README.md
```

## Week 0: Planning (COMPLETE)

✅ Elasticity research (academic-grounded values: -0.35 to -1.0)
✅ MIP formulation (optimization approach, pseudocode ready)
✅ Clustering features extraction (4,054 properties)
✅ K-Means clustering (8 property segments)
✅ Elasticity mapping to clusters (for property segmentation analysis)
✅ Architecture design (buyer archetypes, elasticity flow, assumptions locked)
✅ Design doc (docs/DESIGN.md)

## Week 1: Foundation (Aug 1-15)

⏳ Load & validate property data (clustering_features.csv)
⏳ Create buyer archetypes config (config/buyer_archetypes.yaml)
⏳ Build agent class stubs (Property, Buyer, Seller, Market)
⏳ Export property pool (4,054 pickled Property objects)
⏳ Finalize architecture in code

## Blog Posts

This project is documented across 5 blog posts:
- **Post 10:** "What drives Airbnb/housing prices? Simulation design"
- **Post 11:** "Estimating elasticity from market patterns"
- **Post 12:** "Optimizing prices under uncertainty"
- **Post 13:** "Pricing strategies: Revenue vs. speed"
- **Post 14:** "From prediction to pricing: Lessons learned"

## License

Educational project. Data from Funda.nl for learning purposes only.
