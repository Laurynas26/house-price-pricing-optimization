# Retired Week 0 artifacts

These files are kept, not deleted. They are the subject of `scripts/audit_week0_state.py`,
which is the evidence for why each was retired — deleting them would delete the audit
trail along with them.

Nothing here is on the live pipeline. Do not import from this directory.

## What was retired, and why

### `buyer_archetypes.yaml` — eight K-Means "buyer archetypes"

Retired because the archetypes were **endogenous**. They were produced by clustering
the property pool on `(price, size, price_per_m2)` and then relabelled as buyer types,
so the elasticity assigned to each cluster was read off the same features it claimed to
explain. Any statement about buyer behaviour derived from them was circular.

Three further defects, all measured in the audit:

- **Budget and size ranges came from cluster IQRs.** An IQR covers the middle 50% of a
  cluster by construction, so eight of them cannot tile the distribution. Measured
  result: **60.2% of properties were reachable by no archetype at all** — and that is
  the optimistic bound, since the location filters could not even be evaluated.
- **The location filters matched nothing.** All 23 location tokens
  (`grachtengordel`, `city_center`, `amstelveen`, …) were free text that appears
  nowhere in the data. `neighborhood` holds CBS buurt names, a finer level of
  aggregation, so the Jordaan appears as Anjeliersbuurt-Noord, Bloemgrachtbuurt and a
  dozen others. Worse, `amstelveen`, `diemen` and `weesp` were *preferred* locations
  for an archetype covering 15.6% of buyers, and the pool is Amsterdam-only.
- **Tolerance bands were not a function of elasticity.** Three archetypes shared
  elasticity −0.35 with bands of 12%, 15% and 20%. The band was never derived from the
  elasticity; it was assigned alongside it.

Replaced by: buyers drawn from an income distribution with budgets built from LTI
capacity plus CBS-derived equity minus kosten koper (`src/simulation/demand.py`), and
price-quintile bins for reporting.

### `elasticity_mapping.yaml` — property clusters mapped to elasticity tiers

Retired because it was a **second, incompatible** 8-way partition of the same
properties, clustering on a different feature set
(`price_per_m2, size_num, bedrooms, luxury_score, nr_rooms`). Both files were
documented as "the eight segments"; they were never the same object. It also silently
covered 4,051 properties rather than 4,054, because `run_clustering.py` dropped rows
with missing `nr_rooms` or `luxury_score`.

### `clustering_features.csv` — the Week 0 property pool

Retired because it **could not regenerate the config that depended on it**. It carried
no total price column, so the archetype budget ranges documented as "IQR from data"
were underivable from it, and there was no valuation anchor. It also lacked any usable
location field.

Replaced by: `data/property_pool.csv` via `scripts/build_property_pool.py`.

### `extract_clustering_features.py` and `run_clustering.py`

Superseded by `scripts/build_property_pool.py` and `scripts/build_segmentation.py`.

`run_clustering.py` additionally clustered on `(price_per_m2, size_num, bedrooms, …)`
without any transform, on a variable with skew 5.87. K-Means then spent clusters
isolating outliers: the smallest cluster held 11 properties. That is the mechanism
behind an "Ultra-Luxury Estate" archetype with four eligible properties.

## Reproducing the audit

```bash
python scripts/audit_week0_state.py
```

Seven checks, each printing PASS/FAIL with the numbers behind it. Every figure quoted
above comes from there.
