"""Empirical GAB curve fitted to (moisture, aw) pairs reported by storage trials (spec T3, B1 empirical option).

    python -m packr.sim.sorption_fit                 # fit + bootstrap, print
    python -m packr.sim.sorption_fit --write         # also store the curve in reference/sorption_params.csv

Points: every pairing in reference/literature_pairings.csv for the commodity with a numeric `obs_moisture_db_pct` AND `obs_aw`,
plus the paper's initial (initial_moisture_db_pct, initial_aw) once per DOI. Weights: effective variance, so both reported SDs
count: r = (M - G(aw)) / sqrt(sd_M^2 + (G'(aw) sd_aw)^2). A point with no reported SD (the initial point) gets the median SD of
the others (stated in the output). Constants are searched in log space inside BOUNDS with several starts.

Uncertainty: bootstrap over the trial points with the initial point(s) always kept (the design has one low-aw point; dropping it
changes the experiment, not just the sample). The constants are declared UNIDENTIFIABLE when, for any constant, more than 5 % of
resamples sit on a bound, its 95 % range spans more than a factor 2, or two log-constants correlate beyond |0.9|. Then only the
curve inside the data's aw range may be used (the simulator enforces valid_aw_min..valid_aw_max).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..paths import REFERENCE_DIR
from .physics import gab_aw, gab_dmoisture_daw, gab_moisture

SORPTION_CSV = REFERENCE_DIR / "sorption_params.csv"
BOUNDS = {"m0": (1e-3, 1.0), "C": (1e-2, 1e4), "K": (0.05, 1.0)}       # K <= 1: GAB reduces to BET at 1
STARTS = [(0.03, 5.0, 0.9), (0.05, 2.0, 0.8), (0.02, 50.0, 0.95), (0.1, 1.0, 0.6), (0.04, 0.5, 0.99), (0.01, 200.0, 0.999)]
N_BOOT = 500
TRUST_UNIDENTIFIABLE = "empirical_fit_constants_unidentifiable"
TRUST_IDENTIFIED = "empirical_fit"


def points(pairings: pd.DataFrame, commodity: str = "Banana Chips", exclude: tuple = ()) -> pd.DataFrame:
    p = pairings[(pairings["commodity_in_table"] == commodity) & ~pairings["pairing_id"].isin(exclude)]
    num = lambda s: pd.to_numeric(s, errors="coerce")
    films = p[num(p["obs_moisture_db_pct"]).notna() & num(p["obs_aw"]).notna()]
    out = pd.DataFrame({"label": films["pairing_id"], "moisture_db": num(films["obs_moisture_db_pct"]) / 100.0, "aw": num(films["obs_aw"]),
                        "sd_m": num(films["obs_moisture_sd"]) / 100.0, "sd_aw": num(films["obs_aw_sd"]), "doi": films["doi"]})
    init = films[num(films["initial_moisture_db_pct"]).notna() & num(films["initial_aw"]).notna()].drop_duplicates("doi")
    first = pd.DataFrame({"label": "initial", "moisture_db": num(init["initial_moisture_db_pct"]) / 100.0, "aw": num(init["initial_aw"]),
                          "sd_m": np.nan, "sd_aw": np.nan, "doi": init["doi"]})
    out = pd.concat([first, out], ignore_index=True)
    out["sd_source"] = np.where(out["sd_m"].isna() | out["sd_aw"].isna(), "median of the reported SDs (none reported)", "reported")
    out["sd_m"] = out["sd_m"].fillna(out["sd_m"].median())
    out["sd_aw"] = out["sd_aw"].fillna(out["sd_aw"].median())
    return out.reset_index(drop=True)


def _residuals(theta, m, aw, sm, sa):
    m0, c, k = np.exp(theta)
    return (m - gab_moisture(aw, m0, c, k)) / np.sqrt(sm ** 2 + (gab_dmoisture_daw(aw, m0, c, k) * sa) ** 2)


def fit_gab(pts: pd.DataFrame, weighted: bool = True) -> dict:
    """weighted=False: plain least squares on moisture (unit weights, aw taken as exact), for comparison only."""
    from scipy.optimize import least_squares
    lb, ub = np.log([b[0] for b in BOUNDS.values()]), np.log([b[1] for b in BOUNDS.values()])
    args = tuple(pts[c].to_numpy(float) for c in ("moisture_db", "aw", "sd_m", "sd_aw"))
    if not weighted:
        args = (args[0], args[1], np.ones(len(pts)), np.zeros(len(pts)))
    best = None
    for s in STARTS:
        r = least_squares(_residuals, np.log(s), bounds=(lb, ub), args=args)
        if best is None or r.cost < best.cost:
            best = r
    m0, c, k = np.exp(best.x)
    return dict(m0=float(m0), C=float(c), K=float(k), chi2=float(2 * best.cost), n=len(pts), dof=len(pts) - 3,
                aw_min=float(pts["aw"].min()), aw_max=float(pts["aw"].max()))


def bootstrap(pts: pd.DataFrame, n: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(pts["label"] == "initial")
    films = np.flatnonzero(pts["label"] != "initial")
    rows = []
    for _ in range(n):
        idx = np.r_[keep, rng.choice(films, len(films), replace=True)]
        f = fit_gab(pts.iloc[idx])
        rows.append({k: f[k] for k in ("m0", "C", "K")})
    return pd.DataFrame(rows)


def identifiability(boot: pd.DataFrame) -> dict:
    reasons = []
    for k, (lo, hi) in BOUNDS.items():
        on_bound = float(((boot[k] <= lo * 1.001) | (boot[k] >= hi * 0.999)).mean())
        q_lo, q_hi = np.percentile(boot[k], [2.5, 97.5])
        if on_bound > 0.05:
            reasons.append(f"{k} on a search bound in {on_bound:.0%} of resamples")
        if q_hi / q_lo > 2.0:
            reasons.append(f"{k} 95 % range spans a factor {q_hi / q_lo:.1f}")
    corr = np.corrcoef(np.log(boot[["m0", "C", "K"]].to_numpy()).T)
    names = ["m0", "C", "K"]
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(corr[i, j]) > 0.9:
                reasons.append(f"log {names[i]} and log {names[j]} correlate at {corr[i, j]:.2f}")
    return dict(identifiable=not reasons, reasons=reasons, corr=corr.round(2).tolist())


def aw_at(moisture_db: float, consts: dict | pd.DataFrame) -> np.ndarray:
    if isinstance(consts, dict):
        return gab_aw(moisture_db, consts["m0"], consts["C"], consts["K"])
    return gab_aw(moisture_db, consts["m0"].to_numpy(), consts["C"].to_numpy(), consts["K"].to_numpy())


def moisture_at(aw: float, consts: dict | pd.DataFrame) -> np.ndarray:
    if isinstance(consts, dict):
        return gab_moisture(aw, consts["m0"], consts["C"], consts["K"])
    return gab_moisture(aw, consts["m0"].to_numpy(), consts["C"].to_numpy(), consts["K"].to_numpy())


def sorption_row(fit: dict, ident: dict, commodity: str, doi: str, temp_c: str, source: str) -> dict:
    trust = TRUST_IDENTIFIED if ident["identifiable"] else TRUST_UNIDENTIFIABLE
    note = (f"EMPIRICAL GAB fit to {fit['n']} (moisture d.b., aw) pairs from {source} (initial point + day-120 film pouches; unverified "
            f"against the PDF); effective-variance weights from the reported SDs; chi2 {fit['chi2']:.1f} on {fit['dof']} dof. ")
    if not ident["identifiable"]:
        note += ("Constants are NOT identifiable (" + "; ".join(ident["reasons"]) + "): use ONLY the curve, between valid_aw_min and "
                 "valid_aw_max, never the individual constants. ")
    note += "Used only by Constants(isotherm='gab_empirical'). Regenerate with python -m packr.sim.sorption_fit --write"
    sig = lambda x: float(f"{x:.6g}")
    return dict(commodity=commodity, m0=sig(fit["m0"]), C=sig(fit["C"]), K=sig(fit["K"]), temp_c=temp_c, crit_aw="", crit_basis="", doi=doi,
                notes=note, trust=trust, valid_aw_min=round(fit["aw_min"], 4), valid_aw_max=round(fit["aw_max"], 4))


def write_row(row: dict, path=SORPTION_CSV) -> pd.DataFrame:
    """Replace this commodity's empirical-fit row (literature rows are never touched)."""
    t = pd.read_csv(path, keep_default_na=False, dtype=str)
    for col in ("valid_aw_min", "valid_aw_max"):
        if col not in t.columns:
            t[col] = ""
    drop = (t["commodity"] == row["commodity"]) & t["trust"].str.startswith("empirical_fit")
    t = pd.concat([t[~drop], pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)[list(t.columns)]
    t.to_csv(path, index=False)
    from .physics import _sorption_table
    _sorption_table.cache_clear()
    return t


def run(pairings: pd.DataFrame, commodity: str = "Banana Chips", n_boot: int = N_BOOT) -> dict:
    pts = points(pairings, commodity)
    fit = fit_gab(pts)
    boot = bootstrap(pts, n_boot)
    return dict(commodity=commodity, points=pts, fit=fit, boot=boot, ident=identifiability(boot))


def main(argv: list[str] | None = None) -> None:
    from .calibrate import load_pairings
    ap = argparse.ArgumentParser(prog="packr.sim.sorption_fit")
    ap.add_argument("--commodity", default="Banana Chips")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    pairings = load_pairings()
    r = run(pairings, a.commodity)
    f, b = r["fit"], r["boot"]
    print(f"{len(r['points'])} points; fit m0={f['m0']:.4g} C={f['C']:.4g} K={f['K']:.4g} chi2={f['chi2']:.2f}/{f['dof']} dof")
    print(b.quantile([0.025, 0.5, 0.975]).round(4).to_string())
    print("identifiable:", r["ident"]["identifiable"], r["ident"]["reasons"])
    if a.write:
        src = pairings[pairings["commodity_in_table"] == a.commodity].iloc[0]
        row = sorption_row(f, r["ident"], a.commodity, str(r["points"]["doi"].iloc[0]), str(src["temp_c"]), str(src["source"]).split(",")[0])
        write_row(row)
        print("written:", row)


if __name__ == "__main__":
    main()
