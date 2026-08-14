"""
Derive the buyer equity function from CBS household wealth statistics.

WHY THIS IS BEING BUILT
-----------------------
The equity-by-price-bin sweep found that equity moves the optimal price by 42% in the
top price bin (and sale probability from 0.71 to 0.93), against ~4% pooled. It is the
largest single effect measured so far, and it matters most in exactly the thin,
expensive segment where the prediction audit also found the widest error. So the
derivation is worth doing — which was NOT obvious before the sweep, and is precisely
why the sweep came first.

THE STANDING RULE
-----------------
Equity has enormous leverage: any segment can be made to clear by handing its buyers
more of it. Tuning it until the market clears would calibrate the demand side to
reproduce the prices the project is trying to explain. So equity is DERIVED here and
never adjusted afterwards. If the top of the market fails to clear under CBS-derived
equity, that is a result about who buys Amsterdam property, not a parameter to fix.

DISCLOSURE ABOUT ORDERING
-------------------------
The sensitivity sweep ran BEFORE this derivation, so the value that makes the top bin
clear (roughly 3x LTI capacity) was already known when this was written. That
ordering cannot be undone. The protection is that this script derives its number
purely from CBS inputs with no reference to the simulation, and the result is
reported whatever it turns out to be. A reader should weigh it accordingly.

WHAT IS DERIVED VS ASSUMED
--------------------------
DERIVED: the shape of deployable equity across the income distribution, and its
         national level, from CBS 83834NED.
ASSUMED: that Amsterdam buyers resemble high-income Dutch owner-occupiers after the
         Amsterdam level anchor is applied. See the skewness check below.

DEPLOYABLE EQUITY, NOT TOTAL WEALTH
-----------------------------------
CBS `vermogen` includes business assets, substantial holdings and other real estate
that cannot be moved into a house purchase. What a mover actually deploys is:

    eigen woning value - mortgage debt on it + financial assets

Using total wealth would overstate budgets. This is computed from components rather
than from the headline wealth figure for that reason.

A KNOWN LIMITATION OF THE SOURCE TABLE
--------------------------------------
KenmerkenVanHuishoudens is a single dimension of MARGINALS, not a cross-tabulation.
Income decile and owner-occupier status cannot be crossed: the table offers
"9th income decile" or "owner-occupiers", never "owner-occupiers in the 9th income
decile". The conditioning is therefore on income decile alone, which is the more
relevant axis here because the simulation generates buyers from an income
distribution and maps them by income percentile.

Run from the repo root:
    python scripts/derive_equity_from_cbs.py
"""

import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "config" / "equity_function.yaml"
RAW_PATH = REPO_ROOT / "data" / "cbs_equity_raw.csv"

WEALTH_TABLE = "83834NED"   # Vermogen; huishoudenskenmerken, vermogensbestanddelen
REGION_TABLE = "86160NED"   # Vermogen; huishoudenskenmerken, regio

# Income deciles, 1st (lowest) to 10th (highest).
INCOME_DECILES = {
    f"{i}": key
    for i, key in enumerate(
        [
            "1020870", "1020880", "1020890", "1020900", "1020910",
            "1020920", "1020930", "1020940", "1020950", "1020960",
        ],
        start=1,
    )
}

COMPONENTS = {
    "eigen_woning": "1021431",           # 1.2.1 owner-occupied home value
    "hypotheekschuld": "1021461",        # 2.1 mortgage debt on that home
    "financiele_bezittingen": "1021420",  # 1.1 financial assets
    "totaal_vermogen": "T001126",        # headline wealth, for comparison only
}


def _is_nan(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _starters_share() -> float:
    with open(REPO_ROOT / "config" / "simulation.yaml", "r", encoding="utf-8") as f:
        return float(yaml.safe_load(f)["budget"]["starters_share_of_buyers"])


def odata(table: str, resource: str, query: str = "") -> list[dict]:
    url = f"https://opendata.cbs.nl/ODataApi/odata/{table}/{resource}"
    if query:
        url += "?" + query
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["value"]


def latest_period(table: str) -> str:
    periods = odata(table, "Perioden")
    return sorted(p["Key"] for p in periods)[-1]


def fetch_wealth_by_decile(period: str) -> pd.DataFrame:
    """Wealth components per income decile, on a CONSISTENT denominator.

    CRITICAL SUBTLETY. `GemiddeldVermogen_3` is the mean among households that
    POSSESS the component, and `ParticuliereHuishoudens_1` is the count of those
    possessors — both vary by component. In decile 1 only 93.1k of 825.9k households
    own a home, so its mean home value of EUR 484k describes outright-owning
    pensioners, not the decile. Summing those means across components mixes
    denominators and produces an inverted equity gradient in which the poorest look
    wealthiest. An earlier version of this script did exactly that.

    The fix is to use the aggregate `TotaalVermogen_2` (in EUR millions) divided by
    the total household count of the decile.

    Verified: net housing equity per household in decile 10 computed this way is
    EUR 363.6k, against EUR 363.7k from CBS's own
    (totaal - vermogen excl. eigen woning). The reconstruction is sound.
    """
    dec_filter = " or ".join(
        f"KenmerkenVanHuishoudens eq '{k}'" for k in INCOME_DECILES.values()
    )
    comp_filter = " or ".join(
        f"Vermogensbestanddelen eq '{k}'" for k in COMPONENTS.values()
    )
    q = urllib.parse.urlencode({
        "$filter": f"({dec_filter}) and ({comp_filter}) and (Perioden eq '{period}')"
    })
    rows = odata(WEALTH_TABLE, "TypedDataSet", q)
    if not rows:
        raise RuntimeError(f"No rows returned for period {period}")

    key_to_decile = {v: k for k, v in INCOME_DECILES.items()}
    key_to_comp = {v: k for k, v in COMPONENTS.items()}

    df = pd.DataFrame(rows)
    df["decile"] = df["KenmerkenVanHuishoudens"].str.strip().map(key_to_decile)
    df["component"] = df["Vermogensbestanddelen"].str.strip().map(key_to_comp)
    df = df.dropna(subset=["decile", "component"])
    df["decile"] = df["decile"].astype(int)

    df["aggregate_eur"] = pd.to_numeric(df["TotaalVermogen_2"], errors="coerce") * 1e6
    df["possessors"] = pd.to_numeric(df["ParticuliereHuishoudens_1"], errors="coerce")
    df["mean_among_possessors_eur"] = (
        pd.to_numeric(df["GemiddeldVermogen_3"], errors="coerce") * 1000
    )

    agg = df.pivot_table(
        index="decile", columns="component", values="aggregate_eur", aggfunc="first"
    ).sort_index()
    poss = df.pivot_table(
        index="decile", columns="component", values="possessors", aggfunc="first"
    ).sort_index()
    among = df.pivot_table(
        index="decile", columns="component",
        values="mean_among_possessors_eur", aggfunc="first",
    ).sort_index()

    # Households in the decile: the `totaal` row covers every household.
    households = poss["totaal_vermogen"]

    out = pd.DataFrame(index=agg.index)
    out["households"] = households
    out["home_ownership_rate"] = poss["eigen_woning"] / households
    for c in ("eigen_woning", "hypotheekschuld", "financiele_bezittingen", "totaal_vermogen"):
        out[f"{c}_per_hh"] = agg[c] / households
    out["eigen_woning_among_owners"] = among["eigen_woning"]
    out["hypotheek_among_holders"] = among["hypotheekschuld"]
    out["net_housing_per_owner"] = (
        (agg["eigen_woning"] - agg["hypotheekschuld"]) / poss["eigen_woning"]
    )
    return out


def fetch_amsterdam_anchor(period: str) -> dict:
    """Amsterdam vs national wealth level and skewness."""
    try:
        regions = odata(REGION_TABLE, "RegioS")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not read {REGION_TABLE} regions ({e})")
        return {}

    ams = [r for r in regions if "amsterdam" in r["Title"].lower()]
    nl = [r for r in regions if r["Title"].strip().lower() in ("nederland",)]
    if not ams or not nl:
        print("  WARNING: could not locate Amsterdam or Nederland in region list")
        return {}

    # NOTE the topic suffixes differ between tables: 83834NED numbers them _3/_4
    # while 86160NED uses _4/_5, because it carries an extra relative-households
    # topic. Reading 83834NED's names here returned None for every row and produced a
    # NaN "skewness check" that still printed a conclusion. Names are asserted rather
    # than assumed now.
    period_region = sorted(p["Key"] for p in odata(REGION_TABLE, "Perioden"))[-1]
    all_households = "1050010"  # Particuliere huishoudens (total)

    out = {}
    for label, reg in (("amsterdam", ams[0]), ("nederland", nl[0])):
        q = urllib.parse.urlencode({
            "$filter": (
                f"(RegioS eq '{reg['Key']}') and (Perioden eq '{period_region}') "
                f"and (KenmerkenVanHuishoudens eq '{all_households}')"
            )
        })
        try:
            rows = odata(REGION_TABLE, "TypedDataSet", q)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: region fetch failed for {label} ({e})")
            return {}
        if not rows:
            print(f"  WARNING: no regional rows for {label}")
            continue
        r0 = rows[0]
        missing = [k for k in ("GemiddeldVermogen_4", "MediaanVermogen_5") if k not in r0]
        if missing:
            print(f"  WARNING: {REGION_TABLE} lacks expected topics {missing}; "
                  f"available: {sorted(r0)}")
            return {}
        out[label] = {
            "mean_k": pd.to_numeric(r0["GemiddeldVermogen_4"], errors="coerce"),
            "median_k": pd.to_numeric(r0["MediaanVermogen_5"], errors="coerce"),
            "period": period_region,
            "region": reg["Title"],
        }

    return out


def fit_amsterdam_tail(anchor: dict) -> dict:
    """Fit a lognormal to Amsterdam wealth from its mean and median.

    WHY NOT JUST SCALE THE NATIONAL SHAPE
    -------------------------------------
    The skewness check shows Amsterdam's mean/median ratio is roughly four times the
    national one. The two distributions differ in SHAPE, not level, so multiplying
    the national profile by the level anchor (1.05) leaves the top tail badly
    understated — and the top tail is the whole reason the equity term exists.

    Mean and median jointly identify a two-parameter lognormal:
        median = exp(mu)                 ->  mu    = ln(median)
        mean   = exp(mu + sigma^2 / 2)   ->  sigma = sqrt(2 * ln(mean / median))

    ASSUMPTION, stated rather than buried: household wealth is not truly lognormal.
    It takes negative values (net debt) and its upper tail is closer to Pareto. This
    fit is a two-moment approximation used to get an Amsterdam-specific TAIL where
    CBS publishes no distribution at gemeente level. It is a better approximation
    than a level shift, and worse than real microdata, which is not public.
    """
    a = anchor.get("amsterdam") or {}
    if not a or _is_nan(a.get("mean_k")) or _is_nan(a.get("median_k")):
        return {}

    mean_eur = float(a["mean_k"]) * 1000
    median_eur = float(a["median_k"]) * 1000
    if median_eur <= 0 or mean_eur <= median_eur:
        return {}

    mu = math.log(median_eur)
    sigma = math.sqrt(2.0 * math.log(mean_eur / median_eur))

    # Decile boundaries and the conditional mean above the 90th percentile.
    z90 = 1.2815515655446004
    p90 = math.exp(mu + z90 * sigma)
    # E[X | X > q] for a lognormal, via the standard truncated-moment identity.
    from scipy.stats import norm as _norm

    mean_top_decile = mean_eur * (1 - _norm.cdf((math.log(p90) - mu - sigma**2) / sigma)) / 0.10

    print("\n" + "=" * 78)
    print("AMSTERDAM-SPECIFIC TAIL (two-moment lognormal fit)")
    print("=" * 78)
    print(f"  mu = ln(median) = {mu:.3f}      sigma = sqrt(2 ln(mean/median)) = {sigma:.3f}")
    print(f"  implied 90th percentile of household wealth: EUR {p90:>12,.0f}")
    print(f"  implied mean wealth of the top decile:       EUR {mean_top_decile:>12,.0f}")
    print("\n  Compare the NATIONAL top-decile deployable equity derived above.")
    print("  The gap is the amount by which a national-shape transfer would have")
    print("  understated Amsterdam's top of market.")

    return {
        "mu": mu,
        "sigma": sigma,
        "p90_wealth_eur": p90,
        "mean_top_decile_wealth_eur": mean_top_decile,
        "method": "two-moment lognormal fit to Amsterdam mean and median",
        "caveat": (
            "Household wealth is not lognormal — it admits negative values and has a "
            "heavier-than-lognormal upper tail. This is a two-moment approximation "
            "used because CBS publishes no wealth distribution below gemeente level."
        ),
    }


def build_amsterdam_schedule(w: pd.DataFrame, ams_fit: dict) -> pd.DataFrame:
    """Final equity-by-income-decile schedule, adjusted for Amsterdam's fatter tail.

    Two inputs of very different quality are combined here, and the difference is
    worth being explicit about:

      SOLID    national deployable equity by income decile, from CBS aggregates on a
               consistent denominator.
      WEAKER   one Amsterdam-specific fact — that its wealth distribution is roughly
               four times more skewed than the national one, so a level transfer
               understates its top tail by about 6x.

    CBS publishes no Amsterdam wealth distribution by income decile, so the tail
    correction cannot be applied decile by decile. It is applied as a ramp over the
    top three deciles, reaching the full factor at decile 10 and leaving deciles 1-7
    untouched. That shape is a CHOICE, not a measurement: the ramp is smooth because
    a step would put a discontinuity in buyer budgets at an arbitrary income
    threshold, not because CBS says the correction ramps.

    The multiplier is deliberately applied only upward. Amsterdam's MEDIAN wealth is
    far below national (EUR 33k vs EUR 136k) because the city is majority-renter, but
    that must not be pushed into buyer equity: our buyers are already split into
    first-time buyers (financial assets only) and movers (who own by construction).
    Scaling them down by an all-household ratio would count the renter effect twice.
    """
    out = w[["deployable_equity"]].copy()
    out.columns = ["equity_national_eur"]

    factor = 1.0
    note = "no Amsterdam tail adjustment applied (fit unavailable)"
    if ams_fit:
        # Deployable share of total wealth at the top, taken from the national data,
        # applied to Amsterdam's fitted top-decile wealth.
        deployable_share_top = float(
            w.loc[10, "deployable_equity"] / w.loc[10, "totaal_vermogen_per_hh"]
        )
        ams_top_deployable = ams_fit["mean_top_decile_wealth_eur"] * deployable_share_top
        factor = ams_top_deployable / float(w.loc[10, "deployable_equity"])
        note = (
            f"top-decile equity scaled by {factor:.2f}x, from the Amsterdam lognormal "
            f"fit (top-decile wealth EUR {ams_fit['mean_top_decile_wealth_eur']:,.0f} "
            f"x national deployable share {deployable_share_top:.2f}), ramped over "
            "deciles 8-10"
        )

    ramp = {8: 1.0 / 3.0, 9: 2.0 / 3.0, 10: 1.0}
    out["amsterdam_multiplier"] = [
        1.0 + ramp.get(d, 0.0) * (factor - 1.0) for d in out.index
    ]
    out["equity_amsterdam_eur"] = out["equity_national_eur"] * out["amsterdam_multiplier"]

    print("\n" + "=" * 78)
    print("FINAL EQUITY SCHEDULE BY INCOME DECILE")
    print("=" * 78)
    print(f"  {note}\n")
    print(out.to_string(formatters={
        "equity_national_eur": "{:,.0f}".format,
        "amsterdam_multiplier": "{:.3f}".format,
        "equity_amsterdam_eur": "{:,.0f}".format,
    }))
    return out


def main() -> None:
    print("=" * 78)
    print("DERIVING EQUITY FUNCTION FROM CBS")
    print("=" * 78)

    period = latest_period(WEALTH_TABLE)
    print(f"\nTable {WEALTH_TABLE}, latest period: {period}")

    w = fetch_wealth_by_decile(period)
    print("\nHome ownership rate by income decile "
          "(why component means could not simply be summed):\n")
    print(w[["households", "home_ownership_rate"]].round(3).to_string())

    starters_share = _starters_share()

    # A first-time buyer has no housing equity; a mover carries the net equity of the
    # home they are selling. Both hold financial assets. Rather than a single figure,
    # the two are computed separately and blended by the assumed first-time-buyer
    # share, which is already a parameter of the simulation.
    w["equity_starter"] = w["financiele_bezittingen_per_hh"]
    w["equity_mover"] = w["net_housing_per_owner"] + w["financiele_bezittingen_per_hh"]
    w["deployable_equity"] = (
        starters_share * w["equity_starter"] + (1 - starters_share) * w["equity_mover"]
    )

    print("\n" + "=" * 78)
    print("DEPLOYABLE EQUITY BY INCOME DECILE")
    print("=" * 78)
    print("  starter = financial assets only (no housing equity)")
    print("  mover   = net housing equity of an owner + financial assets")
    print("  blended = net housing equity of an owner + financial assets")
    print(f"  blend uses starters_share = {starters_share:.2f} from config/simulation.yaml")
    print("\n  Business assets, other real estate and substantial holdings are")
    print("  excluded throughout: they sit in totaal_vermogen but cannot fund a")
    print("  house purchase, so using total wealth would overstate budgets.\n")
    print(
        w[["equity_starter", "equity_mover", "deployable_equity", "totaal_vermogen_per_hh"]]
        .round(0)
        .to_string()
    )

    anchor = fetch_amsterdam_anchor(period)
    skew_note = None

    print("\n" + "=" * 78)
    print("AMSTERDAM ANCHOR AND SKEWNESS CHECK")
    print("=" * 78)

    a = anchor.get("amsterdam") or {}
    n = anchor.get("nederland") or {}
    values = [a.get("mean_k"), a.get("median_k"), n.get("mean_k"), n.get("median_k")]
    have_all = all(v is not None and pd.notna(v) for v in values)

    if not have_all:
        # Never conclude from a check that did not run. An earlier version printed
        # "skewness is comparable to national" off a set of NaNs, which is the same
        # defect as reading an excluded sweep cell as a small effect.
        skew_note = (
            "NOT CHECKED — the regional wealth table could not be read, so the "
            "national-to-Amsterdam transfer is unverified. Treat the level anchor as "
            "un-applied and the national shape as a placeholder."
        )
        print(f"  {skew_note}")
    else:
        print(f"  Amsterdam:  mean {a['mean_k']:,.0f}k   median {a['median_k']:,.0f}k")
        print(f"  Nederland:  mean {n['mean_k']:,.0f}k   median {n['median_k']:,.0f}k")
        a_ratio = a["mean_k"] / a["median_k"]
        n_ratio = n["mean_k"] / n["median_k"]
        level = a["mean_k"] / n["mean_k"]
        print(f"\n  mean/median ratio — Amsterdam {a_ratio:.2f}, Nederland {n_ratio:.2f}")
        print(f"  level anchor (Amsterdam mean / national mean): {level:.3f}")
        if a_ratio > 1.3 * n_ratio:
            skew_note = (
                f"Amsterdam is materially more skewed (mean/median {a_ratio:.2f} vs "
                f"{n_ratio:.2f} nationally), so a pure level shift of the national "
                "shape understates its top tail. The level anchor is applied but is a "
                "LOWER BOUND at the top of the market."
            )
            print(f"\n  WARNING: {skew_note}")
        else:
            skew_note = (
                f"Amsterdam skewness (mean/median {a_ratio:.2f}) is comparable to "
                f"national ({n_ratio:.2f}), so transferring the national shape with a "
                "level anchor is a reasonable approximation."
            )
            print(f"\n  {skew_note}")
        anchor["level_multiplier"] = float(level)

    ams_fit = fit_amsterdam_tail(anchor)
    schedule = build_amsterdam_schedule(w, ams_fit)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(RAW_PATH)

    doc = {
        # The schedule the simulation reads. Keyed by income decile, in euros.
        "equity_by_income_decile_eur": {
            int(d): round(float(v), 0)
            for d, v in schedule["equity_amsterdam_eur"].items()
        },
        "equity_national_by_income_decile_eur": {
            int(d): round(float(v), 0)
            for d, v in schedule["equity_national_eur"].items()
        },
        "amsterdam_multiplier_by_decile": {
            int(d): round(float(v), 4)
            for d, v in schedule["amsterdam_multiplier"].items()
        },
        "amsterdam_anchor": anchor,
        "amsterdam_tail_fit": ams_fit,
        "metadata": {
            "source_tables": [WEALTH_TABLE, REGION_TABLE],
            "period": period,
            "definition": (
                "deployable equity = eigen woning - hypotheekschuld eigen woning "
                "+ financiele bezittingen. Business assets, other real estate and "
                "substantial holdings are excluded as not deployable into a purchase."
            ),
            "conditioning": (
                "Income decile only. CBS KenmerkenVanHuishoudens is a set of "
                "marginals, not a cross-tabulation, so owner-occupier status cannot "
                "be crossed with income decile in this table."
            ),
            "transfer_assumption": skew_note,
            "standing_rule": (
                "This function is DERIVED and must never be adjusted to make a market "
                "segment clear. Failure of the top segment to clear is a result, not "
                "a calibration error."
            ),
            "ordering_disclosure": (
                "The sensitivity sweep was run before this derivation, so the equity "
                "level that clears the top bin was known when this was written. The "
                "derivation uses no simulation input, but a reader should weigh the "
                "ordering."
            ),
            "generated_by": "scripts/derive_equity_from_cbs.py",
        },
    }

    def clean(o):
        """Strip NaN and numpy scalars so the config never records a silent NaN."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not _is_nan(v)}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean(doc), f, sort_keys=False, default_flow_style=False)

    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {RAW_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
