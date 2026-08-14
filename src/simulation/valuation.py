"""Valuation: what is true, and what the seller believes.

The simulation needs two distinct objects, and conflating them is what imports
false precision:

  V_true   the property's value. Anchored at the Funda asking price — the best
           available proxy, and the only price the market actually saw.
  V_est    what the seller and the optimizer see. V_true * (1 + eps), where eps is
           the prediction model's measured error.

Putting the noise on the SELLER'S BELIEF rather than on the property's value is the
point. Jittering V_true alone would just perturb the pool and average out. Perturbing
the belief makes "what does valuation error cost a seller?" a measurable output.

WHY NOT RUN THE PREDICTION MODEL
--------------------------------
The model was trained to predict asking price, so its output is a smoothed estimate
of asking price, not an independent valuation. Its residual is model error, not
mispricing. Running it over the pool would also be leakage-optimistic, since ~79% of
the pool was in its training data. What is needed from that project is not the model
but its ERROR PROFILE, and that is already measured out-of-sample in
reports/test_residuals_enriched.csv.

WHY BOOTSTRAP RATHER THAN ASSUME NORMALITY
------------------------------------------
The 860 held-out listings give an empirical error distribution per size band.
Sampling from it directly beats fitting a Gaussian to its standard deviation.

WHY THE OVERLAP ONLY
--------------------
Only 820 of the 860 audit listings are in this pool. The missing 40 are records the
prediction pipeline imputed and this pool drops for missing price/size/bedrooms.
Their errors are atypical: including them inflates the M band from 9.5% to 12.7%, a
34% overstatement of valuation uncertainty for mid-size properties. Where the two
sets do overlap, price and size agree exactly, so this is a cleaning difference
rather than two different datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_CSV = (
    REPO_ROOT.parent
    / "house_price_prediction_project"
    / "reports"
    / "test_residuals_enriched.csv"
)

SIZE_BANDS = ["XS", "S", "M", "L", "XL"]


class ValuationError:
    """Empirical prediction-error distribution, sampled within size band."""

    def __init__(self, pool: pd.DataFrame, overlap_only: bool = True):
        if not RESIDUALS_CSV.exists():
            raise FileNotFoundError(
                f"{RESIDUALS_CSV} not found — expected the prediction project as a "
                "sibling checkout. The error profile cannot be invented."
            )

        resid = pd.read_csv(RESIDUALS_CSV, index_col=0)
        self.n_audit = len(resid)

        if overlap_only:
            keep = resid.index.intersection(pool["listing_id"])
            resid = resid.loc[keep]
        self.n_used = len(resid)

        self.errors_by_band = {
            band: g["rel_error"].to_numpy()
            for band, g in resid.groupby("size_band")
        }

        missing = [b for b in SIZE_BANDS if b not in self.errors_by_band]
        if missing:
            raise ValueError(f"No residuals for size bands {missing}")

        # Largest property observed in the audit. The pool extends beyond it, so
        # error for the very largest properties is an extrapolation, not a
        # measurement. Recorded rather than silently applied.
        self.audit_max_size = float(resid["size_num"].max())

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "n": {b: len(e) for b, e in self.errors_by_band.items()},
                "sd_%": {b: 100 * e.std() for b, e in self.errors_by_band.items()},
            }
        ).reindex(SIZE_BANDS)

    def draw(self, size_bands: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one relative error per property, resampled within its size band."""
        eps = np.empty(len(size_bands), dtype=float)
        for band in SIZE_BANDS:
            mask = size_bands == band
            n = int(mask.sum())
            if n:
                eps[mask] = rng.choice(self.errors_by_band[band], size=n, replace=True)
        return eps


def make_valuations(
    pool: pd.DataFrame, error: ValuationError, rng: np.random.Generator
) -> pd.DataFrame:
    """Attach V_true and V_est to the pool.

    rel_error is defined in the audit as (actual - predicted) / actual, so
    predicted = actual * (1 - rel_error). V_est plays the role of the prediction.
    """
    eps = error.draw(pool["size_band"].to_numpy(), rng)

    out = pool.copy()
    out["v_true"] = out["price_num"].to_numpy(dtype=float)
    out["v_est"] = out["v_true"] * (1.0 - eps)
    out["valuation_error"] = eps
    return out
